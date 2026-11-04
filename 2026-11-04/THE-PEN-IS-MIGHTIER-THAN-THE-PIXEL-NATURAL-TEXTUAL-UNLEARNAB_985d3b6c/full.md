# THE PEN IS MIGHTIER THAN THE PIXEL: NATURAL TEXTUAL UNLEARNABLEEXAMPLE FOR LARGE VISION LANGUAGE MODELS

Chenyu Yi<sup>1∗</sup> Jiawen Zong<sup>1∗</sup> Sami Hormi<sup>1</sup> Bingquan Shen<sup>2,3</sup> Xudong Jiang<sup>1</sup> Alex Chichung Kot<sup>1,4</sup>

<sup>1</sup>Nanyang Technological University, School of Electrical & Electronic Engineering <sup>2</sup>DSO National Laboratories, Singapore <sup>3</sup>National University of Singapore <sup>4</sup>Shenzhen MSU-BIT University

## ABSTRACT

The proliferation of Large Vision Language Models (LVLMs), trained on vast and often sensitive datasets, raised concerns regarding data privacy and unauthorized usage. Unlearnable Examples (UEs), which subtly perturb training data to mislead the models, have emerged as a promising defense method. This paper presents the first exploration of UEs specifically on LVLMs, which poses unique challenges due to their inherent multimodal nature. We propose a novel text-based perturbation method, Natural Textual Unlearnable Examples (NTUE), which can effectively mislead LVLMs by subtly manipulating the image label, establishing a shortcut between an identity of interest and a separate target, without introducing factual misinformation or affecting readability. Comprehensive experiments compared with traditional UE methods demonstrate the superior performance of NTUE. Our findings show that NTUE achieves a high success rate with a poison rate as low as 5%, underscoring its potential as a practical and robust solution for protecting multimodal data in the era of LVLMs.

Index Terms— Unlearnable examples, Large vision language models, Data privacy

## 1. INTRODUCTION

Rapid advancements in large vision language models (LVLMs) have made them indispensable for multimodal tasks like visual question answering and image captioning [1, 2]. However, LVLM’s reliance on large datasets creates significant privacy and security risks, causing rampant identity leakage [3]. Thus, there is an urgent need for privacy protection in LVLMs.

Although there is extensive research on UEs in traditional computer vision tasks such as classification [4] and object detection [5], the application in LVLMs remains underexplored. These methods, such as Targeted Adversarial Perturbation (TAP) [4], Error Minimization (EM) [6], and Multi-Step Error Minimization (MEM) [7], are proven to be effective for traditional models. However, the sophisticated nature of LVLMs, introduces new challenges that necessitate reevaluation in the approach towards effective UEs in LVLMs.

To address this critical gap, this paper proposes Natural Textual Unlearnable Example (NTUE), a novel method for generating UE that targets the vision-language alignment in LVLMs. NTUE introduces targeted misdirections within the natural-language captions, which forces the model to learn false correlations between visual concepts and the manipulated captions. To demonstrate the effectiveness of NTUE, we conducted extensive experiments on LLaVA-1.5 [8] using 3 different datasets: cc-sbu-align [9], Trump-Biden [10] and Celebrity Face Image datasets[11].

Through detailed experiments, we uncover how factors such as poison sample size and dataset complexity influence the success of these methods. Specifically, NTUE reduces the accuracy of Trump identification to 0% with only 20 samples. Experiments on Celebrity Face Image datasets also illustrate NTUE’s transferability across diverse identities. We summarize our contributions as follows:

• To the best of our knowledge, this paper presents the first comprehensive benchmarking study of UEs specifically targeting LVLMs, evaluating both traditional unimodal methods and novel multimodal approaches across datasets and tasks.

• We propose a novel and effective unlearnable example for LVLM, Natural Textual Unlearnable Example (NTUE). NTUE manipulates the textual modality to disrupt the vision-language alignment in LVLMs, while retaining factual integrity and readability.

• Extensive experiments demonstrate that NTUE significantly outperforms existing methods in degrading LVLM performance. Furthermore, we uncover key poison settings that optimize the effectiveness our UEs, providing key insights for future research.

## 2.1. Large Vision Language Model

rge vision-language models (LVLMs) are advanced models capable of processing and understanding information from multiple modalities, typically text and images. These models leverage contrastive learning to align semantically similar text and image pairs within the feature space, creating bridges between text and image representations [12, 13]. Notable LVLMs include LLaVA [8], InstructBLIP [14], and MiniGPT-4 [15]. LLaVA-1.5 [16], for instance, utilizes the CLIP vision encoder to extract visual information, which is then aligned with a language model’s representation space through a connector before processing by a large language model to enable multimodal comprehension and generation. InstructBLIP [14] also incorporate CLIP or EVA-CLIP [17] as vision encoders then processed by a learnable Q-Former to bridge visual and textual data. MiniGPT-4 [15] further enhances efficiency by adapting EVA as its vision encoder and concatenating neighboring visual tokens, streamlined training, and inference. These models exhibit robust multimodal comprehension capabilities, making them highly valuable for research and application development.

## 2.2. Unlearnable Example

Unlearnable examples are data samples that have been subtly perturbed to prevent machine learning models from learning meaningful representations while being interpretable for humans [18]. Fowl et al. first introduced TAP to generate adversarial examples that hinder model learning through targeted training interference [4].

Subsequently, Huang et al. proposed the Error Minimization (EM) method, which creates perturbations by reducing the classification error of training data to near zero [6], rendering the training example ’unlearnable.’ Building on this, Liu et al. extended unlearnable examples to vision language models (VLMs) through their Multi-step Error-Minimization (MEM) approach, which incorporated both visual perturbations and adversarial text triggers [7]. This enabled effective disruption of multimodal contrastive learning in CLIPlike models. However, LVLMs such as LLaVA, BLIP, and MiniGPT-4, may be more resilient to existing UE methods on VLMs. These models exhibit stronger robustness due to their comprehensive training process and larger capacity [16, 14, 15].

Limited studies have explored UE and adversial attacks in LVLMs. Xu et al. [10] proposed ShadowCast, a stealthy method of adversarial perturbation that achieves UE through targeted response manipulation of LVLM tasks. Despite growing interest in data protection and UE techniques, existing research specifically addressing UE in the Large Vision Language Model (LVLM) remains limited.

## 3.1. Definition

![](images/d7998fad2c4eeb578868455b127e9aed1d72d655367e3998e1f0ca2f10a4fbe5.jpg)  
Fig. 1: Training of multimodal unlearnable examples on large vision-language models with clean captions and Natural Text Unlearnable Example captions.

The rise of Large Vision-Language Models (LVLMs) necessitates an extension of Unlearnable Example (UE) research to this new domain. We begin with a standard clean dataset $D _ { c } ~ = ~ \{ ( V _ { i } , Q _ { i } ) \} _ { i = 1 } ^ { n }$ comprising n image-caption pairs, where $V _ { i }$ are the images and $Q _ { i }$ are the corresponding clean captions. As shown in Figure 1, LVLMs can be simpli fied using function $f _ { \phi } : ( V _ { i } , Q _ { i } )  ( A _ { i } )$ , where $\phi$ represents the model parameters, and $A _ { i }$ is the predicted language response from the model. The main concern of privacy leakage occurs when $A _ { i }$ reveals the identity of the target, which can be undesirable.

Using unlearnable examples, we can inject a small set of manipulated data pairs, the unlearnable dataset $D _ { p } ,$ into the larger standard training dataset $D _ { c }$ . The combined dataset $D _ { \theta } = D _ { c } \cup D _ { p }$ , will be able to strengthen a weaker false association for the targeted concepts.

Hence, the objective is to minimize the model’s loss on these incorrect pairs:

$$
\underset { \delta } { \arg \operatorname* { m i n } } \ \mathbb { E } _ { ( V _ { i } , Q _ { i } ) \sim D _ { \theta } } \ [ \mathcal { L } ( f _ { \phi } ( V _ { i } , Q _ { i } ^ { * } ) ]\tag{1}
$$

Here, L represents the loss associated with the LVLM image text matching. This approach ensures that the manipulated captions create a shift in the model’s decision boundary, as shown in Figure 1. Thus, NTUE can facilitate shortcut learning between different identities, creating effective unlearnable examples for LVLM.

![](images/f8337c93d191c307fc63c1f378037b6604168fd87f7f6c120064c0ffb7e2c354.jpg)  
Fig. 2: Natural Text Unlearnable Example training process and evaluation process

## 3.2. Proposed Method: Natural Textual Unlearnable Example

Our proposed method, Natural Textual Unlearnable Example (NTUE), manipulates the textual modality to create powerful unlearnable examples. The core idea is to craft a manipulated caption $Q _ { i } ^ { * }$ that is factually accurate and semantically sound to humans, but introduces a discreet and false association between the clean image $V _ { i }$ and a target concept.

As shown in Figure 2, the unlearnable dataset, $D _ { p }$ contains images of a base identity paired with poison captions, $V _ { i } ^ { * }$ , generated by a Large Language Model. These captions are subtly modified from the original, $V _ { i } ,$ , to mislead the model while maintaining fluency and integrity.

Figure 2 describes the training process and the evaluation process in NTUE for the Trump & Biden dataset. During NTUE generation, the base identity (Trump) is completely removed from the caption, while the target identity (Biden) is injected into the caption. The new NTUE dataset is then used to train a LVLM, $f _ { \phi } ^ { * }$

We implement several variants of this method to explore different attack strengths:

• Ambiguity Attack - NTUE(AA): The target identity is subtly introduced into the captions alongside the base identity. This causes the model to form a confused or blended representation, closely associating both identities and creating ambiguity, making the model difficult • Decoupling Attack - NTUE(DA): This is a stronger variant, shown in Figure 2. References to the base identity are removed from the captions, with only target identity in the caption. This attack aims to force the model to completely decouple the visual features of the base identity from its correct textual representation and reassociate them with the target identity.

![](images/642dab4a581c5d1f0806fcb3693cc3e917b994093f6c2c415bc3e201d31bfa6a.jpg)  
Fig. 3: Identification accuracy of LLaVA-1.5 7B finetuned with Trump images across different UE methods.

Table 1: Identification accuracy of LLaVA-1.5 7B finetuned with Trump images across different UE methods.
<table><tr><td>No. of UEs/Method</td><td>100</td><td>50</td><td>20</td><td>10</td><td>5</td></tr><tr><td>Clean</td><td>97</td><td>96.5 97.5</td><td>96.5 85.5</td><td>87.5</td><td>70</td></tr><tr><td>TAP</td><td>98.5</td><td></td><td></td><td>64</td><td>54</td></tr><tr><td>EM MEM</td><td>95 98</td><td>97 96</td><td>90 83</td><td>81 70</td><td>72</td></tr><tr><td>NTUE(AA)+PGD</td><td>98.5</td><td>97.5</td><td>85</td><td>63.5</td><td>43</td></tr><tr><td>NTUE(AA)</td><td>93.5</td><td>94.5</td><td>91</td><td>84</td><td>63</td></tr><tr><td>NTUE(DA)+PGD</td><td>81.5</td><td>63.5</td><td>54.5</td><td>41.5</td><td>18</td></tr><tr><td></td><td>0</td><td>0</td><td>0</td><td></td><td></td></tr><tr><td>NTUE(DA)</td><td></td><td></td><td></td><td>22.5</td><td>27</td></tr></table>

to distinguish between the two identities.

## 4. EXPERIMENT RESULTS

We present a comprehensive evaluation of our proposed Natural Textual Unlearnable Example (NTUE) method against several established Unlearnable Example (UE) techniques. Our experiments are conducted on the LLaVA-1.5 7B model with two distinct datasets to rigorously assess performance. Trump & Biden dataset was used to assess efficacy in making a single identity (Trump) unlearnable and Celebrity Face Image datasets dataset was used to evaluate the generalizability and robustness in multiclass conditions. Both datasets are appended to the cc-sbu-align dataset before training, as it provides a broad collection of image-caption pairs, allowing for diverse image and textual manipulation.

Table 2: Identification accuracy of various celebrities pairs
<table><tr><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">Clean</td><td colspan="5">70 poison</td></tr><tr><td>PGD</td><td>NTUE(AA)</td><td>NTUE(AA) +PGD</td><td>NTUE(DA)</td><td>NTUE(DA) +PGD</td></tr><tr><td rowspan="4">Angelina Jolie</td><td>Target Jennifer Lawrence</td><td>92.2</td><td>100</td><td>84.4</td><td>100</td><td>0</td><td>94.4</td></tr><tr><td>Johnny Depp</td><td>92.2</td><td>100</td><td>98.9</td><td>100</td><td>0</td><td>97.8</td></tr><tr><td>Will Smith</td><td>92.2</td><td>100</td><td>94.4</td><td>100</td><td>0</td><td>95.5</td></tr><tr><td>Megan Fox</td><td>92.2</td><td>100</td><td>75.5</td><td>100</td><td>0</td><td>94.4</td></tr><tr><td rowspan="4">Brad Pitt</td><td>Leonardo Dicaprio</td><td>55.6</td><td>100</td><td>95.5</td><td>97.8</td><td>0</td><td>0</td></tr><tr><td>Angelina Jolie</td><td>55.6</td><td>98.6</td><td>92.2</td><td>100</td><td>0</td><td>51.1</td></tr><tr><td>Denzel Washington</td><td>55.6</td><td>100</td><td>31.1</td><td>100</td><td>0</td><td>53.3</td></tr><tr><td>Sandra Bullock</td><td>55.6</td><td>100</td><td>100</td><td>100</td><td>51.1</td><td>66.7</td></tr><tr><td rowspan="3">Tom Hanks</td><td>Hugh Jackman</td><td>46.7</td><td>100</td><td>80.0</td><td>100</td><td>0</td><td>55.5</td></tr><tr><td>Jennifer Lawrence</td><td>46.7</td><td>100</td><td>16.7</td><td>100</td><td>0</td><td>56.7</td></tr><tr><td>Denzel Washington</td><td>46.7</td><td>100</td><td>100</td><td>100</td><td>0</td><td>71.1</td></tr></table>

## 4.1. Results on Trump & Biden Dataset

This experiment measures the ability of each UE method to make Donald Trump’s identity unlearnable. We benchmarked NTUE against Error-Minimizing (EM), Multi-step Error-Minimizing (MEM), and Targeted Adversarial Attack (TAP). The effectiveness of each method is measured by its ability to reduce the model’s Identification Accuracy on a test set of 200 Trump images. To ensure the robustness of our findings and reduce randomness, we queried each model with three distinct prompts, 1) Who is this person? 2) What is the name of this person? and 3) Identify the individual in this image.

As shown in Table 1 and visualized in Figure 3, the results demonstrate an interesting trend of effectiveness between the methods tested. Our proposed NTUE is effective at all poisoned training sample sizes, whereas other methods show diminishing effects.

The Decoupling Attack (NTUE-DA) was exceptionally potent, driving identification accuracy to 0% with 20 or more samples. Even with just 5 samples, it was able to reduce accuracy by 43%. This demonstrates that creating a direct and powerful semantic conflict in the textual domain is a superior strategy for inducing unlearning. The combination, NTUE(DA) + PGD, further suppressed recognition at the lowest sample counts (e.g., 18% precision at 5 samples), underscoring the power of a coordinated multimodal attack that establishes a strong and incorrect shortcut between visual features and textual labels.

## 4.2. Results on Celebrity Face Image datasets Dataset

This experiment explores the robustness and generalization of various UE methods by redirecting the model’s identification of various celebrities to another celebrity in the Celeb 18 dataset. This tests the generalizability of these UE techniques beyond the Trump-Biden pair, using diverse celebrity pairs with varying levels of similarity, poison sample counts, and inference prompts.

As shown in Table 2, both the visual-only PGD attack and the mutlimodal NTUE(AA)+PGD poison consistently failed to degrade performance. This suggests that these methods can reinforced the alignment of the model between visual and textual representations of the base celebrity, rather than degrading it. In contrast, NTUE(DA) again proved to be highly effective, successfully driving Identification Accuracy to 0% in the majority of test cases. The combination of NTUE(DA)+PGD has shown more complex results. While it achieve complete unlearning in some pairs (e.g. Brad Pitt and Leonardo DiCaprio), it was significantly less effective for others (e.g., Angelina Jolie and Johnny Depp Pair, 97.8% accuracy). This variability suggests that the interaction between visual noise and textual poison may depend on latent features of the specific base-target pairing.

## 5. CONCLUSION

This paper presented a comprehensive investigation into creating unlearnable examples (UEs) for Large Vision-Language Models. Our proposed NTUE outperforms other methods significantly, by misleading LVLMs without introducing factual misinformation and affecting human readability. Future research should focus on improving the robustness, particularly in low-data regimes, by incorporating advanced multimodal adversarial attacks and data poisons.

## 6. ACKNOWLEDGMENT

This research work was carried out at the Rapid-Rich Object Search (ROSE) Lab, the NTU-PKU Joint Research Institute (sponsored by the Ng Teng Fong Charitable Foundation), Nanyang Technological University, Singapore. The research is supported by the DSO National Laboratories, under project agreement No. DSOCL22332.

## 7. REFERENCES

[1] Yakoub Bazi, Mohamad Mahmoud Al Rahhal, Laila Bashmal, and Mansour Zuair, “Vision–language model for visual question answering in medical imagery,” Bioengineering, vol. 10, no. 3, pp. 380, 2023.

[2] Joao Daniel Silva, Jo˜ ao Magalh˜ aes, Devis Tuia, and˜ Bruno Martins, “Large language models for captioning and retrieving remote sensing images,” arXiv preprint arXiv:2402.06475, 2024.

[3] Simone Caldarella, Massimiliano Mancini, Elisa Ricci, and Rahaf Aljundi, “The phantom menace: unmasking privacy leakages in vision-language models,” in European Conference on Computer Vision. Springer, 2025, pp. 435–451.

[4] Liam Fowl, Micah Goldblum, Ping-yeh Chiang, Jonas Geiping, Wojciech Czaja, and Tom Goldstein, “Adversarial examples make strong poisons,” Advances in Neural Information Processing Systems, vol. 34, pp. 30339– 30351, 2021.

[5] Chenyu Yi, Ruohan Meng, Haohang Peng, Bingquan Shen, and Alex C. Kot, “Towards effective and robust unlearnable examples against object detection,” in 2025 IEEE International Conference on Image Processing (ICIP), 2025, pp. 91–96.

[6] Hanxun Huang, Xingjun Ma, Sarah Monazam Erfani, James Bailey, and Yisen Wang, “Unlearnable examples: Making personal data unexploitable,” in International Conference on Learning Representations, 2021.

[7] Xinwei Liu, Xiaojun Jia, Yuan Xun, Siyuan Liang, and Xiaochun Cao, “Multimodal unlearnable examples: Protecting data against multimodal contrastive learning,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 8024–8033.

[8] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee, “Visual instruction tuning,” Advances in neural information processing systems, vol. 36, pp. 34892– 34916, 2023.

[9] Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechu Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi,

Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny, “Minigpt-v2: large language model as a unified interface for vision-language multi-task learning,” arXiv preprint arXiv:2310.09478, 2023.

[10] Yuancheng Xu, Jiarui Yao, Manli Shu, Yanchao Sun, Zichu Wu, Ning Yu, Tom Goldstein, and Furong Huang, “Shadowcast: Stealthy data poisoning attacks against vision-language models,” Advances in Neural Information Processing Systems, vol. 37, pp. 57733–57764, 2024.

[11] Vishesh Sharma, “Celebrity face image dataset,” https://www.kaggle.com/datasets/vishesh1412/celebrityface-image-dataset, 2018.

[12] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[13] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi, “Align before fuse: Vision and language representation learning with momentum distillation,” Advances in neural information processing systems, vol. 34, pp. 9694– 9705, 2021.

[14] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi, “Instructblip: Towards generalpurpose vision-language models with instruction tuning,” Advances in neural information processing systems, vol. 36, pp. 49250–49267, 2023.

[15] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny, “Minigpt-4: Enhancing visionlanguage understanding with advanced large language models,” arXiv preprint arXiv:2304.10592, 2023.

[16] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee, “Improved baselines with visual instruction tuning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 26296–26306.

[17] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao, “Eva-clip: Improved training techniques for clip at scale,” arXiv preprint arXiv:2303.15389, 2023.

[18] Xuan Wang, Xitong Gao, Dongping Liao, Tianrui Qin, Yu-liang Lu, and Cheng-zhong Xu, “A3: Few-shot prompt learning of unlearnable examples with crossmodal adversarial feature alignment,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 9507–9516.