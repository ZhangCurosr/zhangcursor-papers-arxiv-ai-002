# Sparse Fine-Tuning for Parameter-Efficient Adversarial Training

Zhaoxin Wang School of Artificial Intelligence Xidian University Xi’an, China

Zihang Ding School of Artificial Intelligence Xidian University Xi’an, China

Handing Wang<sup>∗</sup> School of Artificial Intelligence Xidian University Xi’an, China

## Abstract

Deep neural networks achieve strong performance across many tasks but remain vulnerable to adversarial perturbations. Adversarial training (AT) is one of the most effective defenses, yet it suffers from high memory cost on large models and the persistent clean-robust accuracy trade-off. Recent work introduces parameterefficient fine-tuning, such as LoRA, into AT, but this constraint induces a notable robustness gap relative to full-parameter AT. In this work, we revisit parameterefficient adversarial fine-tuning from a parameter space perspective. Through gradient analysis, we find that adversarial optimization is highly concentrated on a small subset of parameters, which we call Robustness-Critical Parameters (RCPs), suggesting that robustness is encoded in a sparse subspace. Building on this observation, we propose Sparse Adversarial Fine-Tuning (SAFT), which identifies RCPs via adversarial gradient saliency and performs AT by updating only these parameters while freezing the rest. Across architectures and datasets, SAFT outperforms LoRA-based methods in both clean and robust accuracy with fewer trainable parameters, and approaches or surpasses full parameter AT in robustness while updating only about 5% of parameters.

## 1 Introduction

Deep learning has achieved excellent performance across a wide range of tasks [26], such as image classification [16], semantic segmentation [14], and object detection [23]. However, deep learning models are vulnerable to adversarial attacks, in which imperceptible perturbations added to the input can lead to wrong predictions [35]. To improve the model’s robustness against these attacks, one of the most effective defenses is adversarial training (AT) [30]. AT can be formulated as a bi-level optimization problem, where the inner maximization generates adversarial perturbations, typically using a multi-step procedure called projected gradient descent (PGD)[30], and the outer minimization optimizes the model parameters on the perturbed data.

Although AT significantly improves the robustness of the deep learning models, AT suffers from two longstanding issues. First, the bi-level optimization in AT requires multiple forward and backward passes per training step to generate adversarial perturbations, leading to substantially higher memory and time costs than standard training. Second, as adversarial examples and clean examples can be viewed as samples from two different distributions, AT induces a clean-robust accuracy trade-off that models trained to be robust often suffer a notable accuracy degradation on benign samples.

To reduce the computational overhead of AT, one line accelerates the inner perturbation generation process [45, 34, 40, 49]. FastAT [43] replaces multi-step PGD with single-step FGSM, but it suffers from catastrophic overfitting, which subsequent studies seek to mitigate [1, 48, 22, 41]. While these methods reduce training time, they update all model parameters and leave the memory bottleneck unaddressed. To reduce memory cost on large transformer-based models, recent work introduces parameter-efficient fine-tuning (PEFT) into AT. LoFT [11], LNLoRA [44], and AdvLoRA [21] all build on LoRA [18] to limit trainable parameters to a low-rank subspace. However, LoRA-based PEFT-AT exhibits a persistent robustness gap relative to full-parameter AT [19, 5], suggesting that the low-rank constraint may be misaligned with the optimization directions needed for robustness.

Unlike these efficiency methods, another line of work targets the trade-off problems [46, 39, 42]. TRADES [46] reformulates the AT objective to balance clean and robust losses through a tunable regularizer. MART [38] reweights misclassified examples to better handle hard cases. Carmon et al. [4] leverage additional unlabeled data to push the trade-off frontier outward. While these methods improve the trade-off frontier, they all operate at the loss or data level and implicitly treat the trade-off as a property shared uniformly across all parameters. Whether the trade-off has structure in parameter space, and whether it can be mitigated by acting on that structure, remains largely unexplored.

In this work, we revisit AT from a parameter-space perspective and ask whether the two above issues share a common origin. Through gradient analysis, we observe that adversarial optimization is concentrated on a small subset of parameters. A small fraction of parameters, about 5%, carry the majority of the cumulative gradient mass, while over 90% remain near zero throughout training. These observations suggest that robustness is not uniformly distributed across the network but is encoded in a sparse subset of parameters, which we refer to as Robustness Critical Parameters (RCPs).

This sparsity has two important implications. From an efficiency perspective, full-parameter AT is over-parameterized for robustness, and updating only the RCPs should suffice to recover most of the robustness gain. From a trade-off perspective, if the adversarial signal lives in a sparse subspace of parameters, restricting updates to that subspace leaves all remaining pretrained weights unchanged. The two longstanding issues of AT may therefore be approached jointly, not by modifying the loss or expanding the data, but by acting on the parameter-space structure of robustness itself.

Building on this, we propose Sparse Adversarial Fine-Tuning (SAFT), a parameter-efficient adversarial fine-tuning framework that identifies RCPs via adversarial gradient saliency and performs adversarial fine-tuning by updating only these parameters while freezing the rest. The identification is performed once before fine-tuning and requires no additional learnable importance scores. SAFT is orthogonal to existing AT methods and can be easily integrated with different AT objectives. Our contribution can be summarised as:

• We identify a sparse parameter-space structure underlying adversarial robustness, where adversarial optimization is concentrated on a small subset of parameters, which we call RCPs. This concentration holds across datasets and on a convolutional backbone. The RCPs are highly consistent across different attack types, indicating that they reflect an intrinsic property of the model rather than an artifact of any specific attack.

• We propose SAFT, a sparse parameter-efficient adversarial fine-tuning framework that updates only the RCPs while freezing the rest of the model. By confining adversarial updates to the RCPs, SAFT leaves the remaining pretrained weights unchanged and directs robustness gains only to where they are needed. It reduces optimizer-state, persistent-gradient and peak training memory relative to full-parameter AT.

• Across architectures and datasets, SAFT outperforms LoRA-based adversarial fine-tuning and approaches or surpasses full-parameter adversarial fine-tuning in robust accuracy while updating only about 5% of parameters. The localization of robustness in the RCPs explains why SAFT improves the clean-robust trade-off frontier, offering a new parameter-space perspective. Control experiments show that RCPs are robustness-specific rather than generally important parameters.

## 2 Related Work

## 2.1 Efficient Adversarial Training

A major limitation of AT is the high computational cost incurred by multi-step adversarial example generation. Existing efficient AT methods mainly reduce the cost of the inner maximization.

FastAT [43] replaces multi-step PGD with single-step FGSM [12], but suffers from catastrophic overfitting. Subsequent methods improve the stability of single-step AT through gradient regularization or modified perturbation updates, such as GradAlign [1] and FGSM-PCO [41]. Other methods reduce computation by reusing gradients or perturbations. FreeAT [34] combines adversarial example generation and model update within the same backward pass. YOPO [45] avoids full network backpropagation during inner maximization by using an approximation based on the first layer.

Recently, PEFT has been introduced to reduce the memory cost of adversarial adaptation of transformer-based large models. LoRA-based methods restrict trainable parameters to low-rank adaptation modules. Yuan et al. [44] apply LoRA to most modules and introduce LayerNorm before LoRA modules to improve robustness, while AdvLoRA [21] improves LoRA-based adversarial adap tation through reparameterization. DoRA [28] decomposes each pretrained weight into a magnitude and a direction and applies the low-rank update to the direction.

## 2.2 Accuracy-Robustness Trade-off

The trade-off between clean accuracy and adversarial robustness is hard to handle in AT. Improving robustness against adversarial perturbations often comes at the cost of lower accuracy on benign examples. Prior studies have provided several explanations for this phenomenon. Tsipras et al. [36] show that robustness and accuracy can be inherently conflicting under certain data distributions. Ilyas et al. [20] attribute this conflict to non-robust but predictive features, which are useful for clean classification but sensitive to small perturbations. These studies suggest that the trade-off is closely related to the features learned by the model. To mitigate this issue, TRADES [46] introduces a regularized objective that balances natural accuracy and boundary robustness. MART [38] improves robust generalization by reweighting misclassified examples. Semi-supervised AT and robust selftraining leverage additional unlabeled data to push the trade-off frontier outward [4]. Recent methods further explore invariance regularization, input-level reweighting, attention regularization, and label space reformulation to reduce the conflict between clean and adversarial learning [42, 47].

## 2.3 Parameter-Space Sparsity in Robustness

Parameter selection has been widely studied as a way to improve training efficiency while preserving model performance. Selective PEFT methods update only a subset of pretrained parameters instead of introducing additional adapter modules. Diff pruning [13] learns a sparse mask over parameter differences to identify task-relevant updates. PaFi [27] fine-tunes parameters with small absolute magnitudes in pretrained models. LT-SFT [2] selects parameters that change the most during the first epoch of fine-tuning. Parameter sparsity is also closely related to pruning and subnetwork discovery. The lottery ticket hypothesis [10] shows that dense neural networks contain sparse subnetworks that can be trained in isolation to match the performance of the original network. A large body of work on structured pruning further shows that many parameters or components can be removed while retaining accuracy [6, 50, 37]. These studies suggest that model capabilities are not uniformly distributed across all parameters, but can be concentrated in a subset of critical components.

## 3 Empirical Findings

Most existing AT methods update all model parameters during optimization. This practice assumes that adversarial robustness is distributed across the parameter space. However, our empirical analysis suggests a different picture. Robust adaptation is highly sparse, with adversarial update signals concentrated in a small subset of parameters. In this section, we present motivating evidence based on AT of a pretrained ViT-B model [9] on CIFAR-10 [24].

## 3.1 Adversarial Gradients Concentrate on Few Parameters

We analyze how adversarial optimization is distributed across model parameters. During AT, we accumulate the gradient saliency of each parameter under the adversarial loss and normalize the values for comparison. This measures how strongly each parameter participates in adversarial optimization throughout training.

![](images/dcf61dfcb1ed045e14a07b48f8b3ad2d2c950f40cab3b32d274da07c18cac3cf.jpg)  
(sorted by gradient magnitude, high low)  
Figure 1: Top 5% of ViT-B parameters capture 67.7% of adversarial gradient.

As shown in Fig. 1, the cumulative distribution is highly skewed. The top 5% of parameters capture 67.7% of the total saliency mass, far above the 5% expected under a uniform distribution. The remaining 95% of parameters together account for only about a third of the cumulative saliency, with the bottom half contributing almost nothing. This finding suggests that adversarial fine-tuning does not update all parameters uniformly. Instead, the adversarial optimization signal is concentrated in a sparse subset of parameters, while the vast majority receive negligible adversarial gradients. From this perspective, full-parameter adversarial fine-tuning is over-parameterized for the purpose of robust adaptation.

## 3.2 High-Gradient Parameters Are Critical for the Robust Model

To examine whether high-gradient parameters are functionally important for the trained model, we perform a parameter intervention experiment. Starting from an adversarially fine-tuned ViT-B model on CIFAR-10, we identify the parameters with the largest accumulated adversarial gradients and mask them by setting their values to zero. We compare this intervention to masking a random subset, and re-evaluate both clean and robust accuracy without further training.

The results are shown in Table 1. Masking the top 5% high-gradient parameters causes robust accuracy to collapse from 43.60% to 3.70%, a relative loss of more than 90%, while masking a random 5% subset reduces it only to 39.48%. The destructive effect depends on which parameters are removed, not how many. Clean accuracy is also affected by masking high-gradient parameters, indicating that these parameters are critical to the overall function of the network rather than to robustness alone. However, the relative loss in robust accuracy is substantially larger than that in clean accuracy, showing that the robust behavior of the trained model relies on this small subset.

Table 1: Masking RCPs is more destructive than masking random parameters.
<table><tr><td>Setting</td><td>Clean</td><td>PGD-20</td></tr><tr><td>Baseline</td><td>85.04</td><td>43.60</td></tr><tr><td>5% Random</td><td>81.83</td><td>39.48</td></tr><tr><td>RCPs 5%</td><td>29.82</td><td>3.70</td></tr><tr><td>Random 10%</td><td>66.46</td><td>32.18</td></tr><tr><td></td><td></td><td></td></tr><tr><td>RCPs 10%</td><td>24.35</td><td>4.75</td></tr></table>

Together with the gradient distribution in Fig. 1, this intervention indicates that high-gradient param eters are both statistically salient and causally critical for the robust behavior of the adversarially trained model. We refer to these parameters as RCPs. The above findings motivate a sparse view of AT. If only a small subset of parameters receives strong adversarial update signals and is critical for robustness, updating the entire model may be unnecessary. Based on these observations, we propose SAFT, which first identifies robustness-critical parameters using adversarial gradient saliency and then updates only these parameters during adversarial fine-tuning. By concentrating adversarial optimization on RCPs while freezing the remaining parameters, SAFT targets the sparse subspace where adversarial optimization lives, and leaves the remaining pretrained weights unchanged.

## 4 Method

Motivated by the empirical findings in Section 3, we propose SAFT. Instead of updating all parameters during adversarial fine-tuning, SAFT first identifies RCPs using adversarial gradient saliency, and then updates only these parameters while freezing the rest of the model. This design preserves the original model architecture, introduces no adapter modules, and directly confines adversarial adaptation to the parameter subset most relevant to robustness.

## 4.1 Formulation

Let $f _ { \theta }$ denote a pretrained model with parameters $\theta ,$ and let D denote the downstream training distribution. Standard AT optimizes

$$
\operatorname* { m i n } _ { \theta } \ \mathbb { E } _ { ( x , y ) \sim \mathcal { D } } \left[ \operatorname* { m a x } _ { \| \delta \| _ { \infty } \leq \epsilon } \ \ell \big ( f _ { \theta } ( x + \delta ) , y \big ) \right] ,\tag{1}
$$

where δ is the adversarial perturbation and ϵ is the perturbation budget. Existing AT methods update the full parameter vector θ during the outer minimization. SAFT instead introduces a binary update mask $m \in \{ 0 , 1 \} ^ { | \theta | }$ , where $m _ { i } = 1$ indicates that $\theta _ { i }$ is trainable and $m _ { i } = 0$ indicates that it is frozen. Given an AT loss $\mathcal { L } _ { a d v }$ , the masked update is

$$
\theta ^ { t + 1 } = \theta ^ { t } - \eta \big ( m \odot \nabla _ { \theta } \mathcal { L } _ { a d v } ( \theta ^ { t } ) \big ) ,\tag{2}
$$

where $\eta$ is the learning rate and ⊙ denotes element-wise multiplication. The central design question is how to construct m so that the selected parameters are those most responsible for robust adaptation. SAFT constructs this mask using adversarial gradient saliency.

## 4.2 Robustness-Critical Parameter Identification

We identify RCPs through a short calibration phase before sparse fine-tuning. Starting from the pretrained model $f _ { \theta }$ , we run a small number of AT steps on a calibration set $\mathcal { D } _ { c a l }$ and accumulate the adversarial gradient magnitude of each parameter throughout this phase. For parameter $\theta _ { i }$ , the accumulated saliency score is

$$
s _ { i } = \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { c a l } } \left[ \left| \nabla _ { \theta _ { i } } \ell \bigl ( f _ { \theta } ( x + \delta ) , y \bigr ) \right| \right] ,\tag{3}
$$

where δ is the adversarial perturbation generated against the current model by the same attack used for AT,

$$
\delta = \arg \operatorname* { m a x } _ { \| \delta ^ { \prime } \| _ { \infty } \leq \epsilon } \ell \big ( f _ { \theta } ( x + \delta ^ { \prime } ) , y \big ) .\tag{4}
$$

The expectation is taken over the calibration data. The saliency score $s _ { i }$ measures how strongly parameter $\theta _ { i }$ participates in adversarial optimization across the calibration phase. Parameters with larger scores receive stronger adversarial update signals and are therefore more likely to be critical for robust adaptation. We select the top $\rho$ fraction of parameters according to $s _ { i }$ as RCPs:

$$
\mathcal { R } = \mathrm { T o p } _ { \rho } \big ( \{ s _ { i } \} _ { i = 1 } ^ { | \theta | } \big ) , \quad m _ { i } = \left\{ 1 , \quad \theta _ { i } \in \mathcal { R } , \right.\tag{5}
$$

The mask is computed once during the calibration phase and fixed throughout AT. In all experiments, the calibration phase is a single epoch over the training set. When applying SAFT to transformerbased models, we restrict the search space to the main transformation modules, following common practice in parameter-efficient fine-tuning. Specifically, we search for RCPs in the projection matrices of self-attention modules $( W _ { Q } , W _ { K } , \bar { W } _ { V } , \bar { W } _ { O } )$ and in the two linear layers of the feed-forward networks $( W _ { u p } , W _ { d o w n } )$ . This focuses the update budget on modules that dominate representation transformation.

## 4.3 Sparse Adversarial Fine-Tuning

After constructing the RCPs mask, SAFT performs AT with masked gradients according to the update rule introduced in Section 4.1. Only RCPs are updated, while all other parameters remain at their pretrained values. SAFT modifies only the update space, not the adversarial objective itself, so it can be combined with different AT losses such as PGD-AT [30], TRADES [46], and MART [38].

SAFT differs from both pruning and LoRA-based PEFT. Unlike pruning, SAFT does not remove parameters or alter the inference architecture. The full pretrained model is retained, and the mask only controls training-time updates. Unlike LoRA-based methods, SAFT does not introduce low-rank adapters or constrain updates to an external low-rank subspace. Instead, it directly updates a sparse subset of the original model parameters selected by adversarial gradient saliency, providing a sparse counterpart to LoRA-based PEFT AT.

This sparse update strategy has two benefits. First, it reduces training memory, since persistent gradients and optimizer states need to be maintained only for the selected $\rho$ fraction of parameters. The backward pass still computes the full gradient in Eq. (2), so SAFT does not reduce the FLOPs of AT but only the selected coordinates keep persistent gradients and optimizer states. Second, by confining adversarial adaptation to RCPs and keeping all other parameters at their pretrained values, SAFT minimizes the unnecessary perturbation that full-parameter AT introduces across the network. The remaining parameters of the model stay close to its pretrained state, which achieves a better trade-off between training cost and robust adaptation.

## 5 Experimental Results

## 5.1 Experimental Setup

Datasets and Models. We conduct experiments across datasets. CIFAR-10 and CIFAR-100 [24] each contain 60,000 images of size 32×32, with 50,000 for training and 10,000 for testing, distributed across 10 and 100 classes respectively. TinyImageNet [25] contains 200 classes with 500 training images and 50 for testing per class. We evaluate SAFT on three pretrained backbones, including ViT-B, ViT-S [9], and Swin-B [29], and additionally ResNet-50 [15] in Appendix B. ViT-B and ViT-S both contain 12 Transformer blocks, with hidden widths of 768 and 384, while Swin-B adopts a hierarchical architecture with a shifted-window attention mechanism. All models are pretrained on ImageNet-1K [33]. We keep the default architectures and adjust the image and patch sizes to match the input resolution. For CIFAR-10 and CIFAR-100, the image size is set to 32 and the patch size to 4. For TinyImageNet, the image size is set to 64 and the patch size to 8.

Evaluation Metrics. We report classification accuracy on clean inputs and adversarial examples. Four attacks, PGD-20 [30], CW-20 [3], PGD-100 [30], and AutoAttack [7] are used to generate perturbations, where the number following PGD or CW refers to the number of iterations.

Comparison Methods. We compare SAFT against full-parameter adversarial fine-tuning and two LoRA-based PEFT-AT methods, LoRA [18] and LNLoRA [44]. To verify the effectiveness of adversarial gradient saliency for parameter selection, we compare against random parameter selection, weight-magnitude selection and clean-data gradient saliency under the same parameter budget.

Training Details. We use SGD [32] with momentum 0.9 and weight decay $1 0 ^ { - 4 }$ . Models are trained for 40 epochs with an initial learning rate of 0.04, decayed by a factor of 10 at epochs 35 and 38. For adversarial perturbations, we use the $\ell _ { \infty }$ norm with $\epsilon = 8 / 2 5 5$ and step size $\alpha = 2 / 2 5 5$ . For LoRA, LNLoRA and DoRA, the rank is set to $r = 3 2$ . For SAFT, we select the top 5% of parameters as RCPs. All experiments are conducted on a single NVIDIA RTX 4090 GPU.

## 5.2 Main Results across Architectures

We compare SAFT against full-parameter adversarial fine-tuning and two LoRA-based PEFT-AT methods on three pretrained backbones. Results on CIFAR-10 are reported in Table 2. SAFT consistently outperforms LoRA-based PEFT-AT across all architectures. On every backbone and under every attack, SAFT achieves higher clean and robust accuracy than both LoRA and LNLoRA, while using fewer trainable parameters. The improvement is substantial. On ViT-B, SAFT improves over LoRA by 3.04% in clean accuracy and 2.83% on PGD-20, while reducing trainable parameters from 4.7M to 3.1M. On ViT-S, the gain is even larger, with SAFT improving robust accuracy by 2.99% on PGD-20 and 3.50% on AutoAttack. On Swin-B, SAFT improves over LoRA by 3.67% on PGD-20 and 3.35% on AutoAttack, with comparable clean accuracy.

SAFT approaches or surpasses full-parameter AT in robust accuracy while updating fewer than 5% of parameters. On ViT-B, SAFT surpasses Full AT under all four attacks, despite training only 3.6% of the parameters. On ViT-S and Swin-B, SAFT approaches Full AT in robustness, with a remaining gap of about 1% across attacks. Clean accuracy of SAFT is lower than Full AT by 2.5%, likely reflecting that the frozen parameters provide less flexibility for fitting clean examples.

Sparse update can outperform dense update in robustness. The observation that SAFT surpasses Full AT on ViT-B is particularly noteworthy. We attribute this to an implicit regularization effect. By restricting adversarial updates to a small sparse subspace, SAFT reduces the effective capacity of the optimizer and mitigates robust overfitting. This finding suggests that full parameter updates are not necessarily the upper bound for adversarial fine-tuning. SAFT also applies to CNNs. On ResNet-50 and CIFAR-10, it reaches 49.32% PGD-20 accuracy compared to the accuracy of 48.95% for Full AT (Appendix B).

## 5.3 Main Results across Datasets

We further evaluate SAFT on CIFAR-100 and TinyImageNet to test whether the conclusions generalize to more complex datasets. Results are reported in Table 3, with CIFAR-10 in Table 2. SAFT’s advantage over LoRA-based PEFT-AT grows with dataset complexity. On CIFAR-10, SAFT improves over LoRA by about 3% on PGD-20 and AutoAttack. On CIFAR-100, the improvement is around 3%. On TinyImageNet, the gap becomes dramatic. LoRA and LNLoRA collapse to low robust accuracy, while SAFT maintains 18.01% on PGD-20 and 14.34% on AutoAttack, an improvement of more than 11% and 9% respectively. The collapse of LoRA on TinyImageNet suggests that as the downstream task becomes more distant from the pretraining distribution, the low-rank optimization subspace becomes increasingly inadequate for capturing the directions needed for robustness. Sparse update, by contrast, draws from the full parameter space and remains effective even under significant distribution shift.

Table 2: Comparison across architectures on CIFAR-10. Trainable parameters refer to updated parameters during AT. Below each backbone: its total parameter count, to which LoRA-based methods add adapters.
<table><tr><td>Model</td><td>Method</td><td>Trainable Params</td><td>Clean</td><td>PGD-20</td><td>CW-20</td><td>PGD-100</td><td>AutoAttack</td></tr><tr><td rowspan="3">ViT-B (85.2M)</td><td>Pretrained Full Parameter</td><td>85.2M (100%)</td><td>96.26 85.04</td><td>0.00 43.60</td><td>0.00 44.57</td><td>0.00 42.96</td><td>0.00 41.16</td></tr><tr><td>LoRA LNLoRA</td><td>4.7M (5.5%) 4.8M (5.6%)</td><td>79.51 79.59</td><td>41.58 41.79</td><td>41.27 41.50</td><td>41.31 41.40</td><td>38.63 39.07</td></tr><tr><td>SAFT Pretrained</td><td>3.1M (3.6%)</td><td>82.55 94.49</td><td>44.41 0.00</td><td>44.87 0.00</td><td>44.04 0.00</td><td>41.61 0.00</td></tr><tr><td rowspan="4">ViT-S (21.4M)</td><td>Full Parameter LoRA</td><td>21.4M (100%)</td><td>83.09</td><td>42.60</td><td>43.96</td><td>42.19</td><td>40.53</td></tr><tr><td>LNLoRA</td><td>1.1M (5.1%) 1.1M (5.1%)</td><td>75.57 75.09</td><td>38.50 38.47</td><td>37.73 37.62</td><td>38.10</td><td>34.97</td></tr><tr><td>SAFT</td><td>0.8M (3.7%)</td><td>78.13</td><td>41.49</td><td>41.43</td><td>38.43 41.11</td><td>35.36 38.47</td></tr><tr><td>Pretrained</td><td></td><td>87.26</td><td>0.00</td><td>0.00</td><td></td><td>0.00</td></tr><tr><td rowspan="4">Swin-B (85.9M)</td><td>Full Parameter</td><td>85.9M (100%)</td><td>72.00</td><td>39.03</td><td>38.24</td><td>0.00 38.80</td><td>35.92</td></tr><tr><td>LoRA</td><td>5.9M (6.9%)</td><td>67.29</td><td>33.38</td><td>32.07</td><td></td><td></td></tr><tr><td>LNLoRA</td><td>5.9M (6.9%)</td><td>67.30</td><td>33.42</td><td>32.09</td><td>33.15 33.16</td><td>29.85</td></tr><tr><td>SAFT</td><td>4.1M (4.8%)</td><td>67.37</td><td>37.05</td><td>35.75</td><td>36.53</td><td>29.84 33.20</td></tr></table>

Table 3: Comparison across datasets on ViT-B (CIFAR-10 results are in Table 2). SAFT consistently outperforms LoRA-based PEFT-AT in robust accuracy across CIFAR-10, CIFAR-100, and TinyImageNet. The robustness gap of LoRA-based methods grows with dataset complexity, while SAFT remains close to full-parameter AT.
<table><tr><td>Dataset</td><td>Method</td><td>Clean</td><td>PGD-20</td><td>CW-20</td><td>PGD-100</td><td>AutoAttack</td></tr><tr><td rowspan="5">CIFAR-100</td><td>Pretrained</td><td>86.10</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Full Parameter</td><td>64.14</td><td>24.42</td><td>25.14</td><td>24.09</td><td>22.58</td></tr><tr><td>LoRA</td><td>60.78</td><td>22.25</td><td>22.21</td><td>21.81</td><td>19.76</td></tr><tr><td>LNLoRA</td><td>61.05</td><td>21.96</td><td>22.01</td><td>21.52</td><td>19.80</td></tr><tr><td>SAFT</td><td>62.60</td><td>24.84</td><td>24.80</td><td>24.45</td><td>21.80</td></tr><tr><td rowspan="5">TinyImageNet</td><td>Pretrained</td><td>80.94</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Full Parameter</td><td>58.82</td><td>18.64</td><td>18.36</td><td>18.20</td><td>14.92</td></tr><tr><td>LoRA</td><td>55.53</td><td>6.71</td><td>7.27</td><td>5.97</td><td>4.89</td></tr><tr><td>LNLoRA</td><td>56.03</td><td>7.92</td><td>8.41</td><td>7.27</td><td>5.73</td></tr><tr><td>SAFT</td><td>54.40</td><td>18.01</td><td>16.86</td><td>17.79</td><td>14.34</td></tr></table>

SAFT remains close to full-parameter AT across all three datasets. On CIFAR-100, SAFT slightly outperforms Full AT on PGD-20 and PGD-100, with a small gap on AutoAttack. On TinyImageNet, SAFT closely matches Full AT in robust accuracy, with a remaining gap of 0.4–1.5%. Clean accuracy of SAFT is somewhat lower than Full AT and, on TinyImageNet, also lower than LoRA-based methods, reflecting the constraint imposed by freezing over 95% of the parameters under significant distribution shift. Even so, SAFT preserves most of Full AT’s robustness across datasets of varying complexity while training only a small fraction of parameters.

## 5.4 Robustness Critical Parameter under Different Attacks

A natural concern is whether the parameters identified by SAFT depend on the specific attack used to compute adversarial gradients. We test this by identifying RCPs separately under four different attacks (PGD-20, PGD-100, CW-20, and AutoAttack) on ViT-B / CIFAR-10, and compute the pairwise overlap between the resulting RCPs sets.

Results are reported in Table 4. The overlap is consistently high across all attack pairs, ranging from 82.82% to 92.42%. The two PGD variants (PGD-20 and PGD-100) yield the highest overlap of 92.42%, indicating that the number of attack iterations has only a small effect on which parameters are identified as critical. Across structurally different attacks (e.g., CW-20 vs AutoAttack), the overlap remains above 82%, far higher than the 5% level expected if RCPs were arbitrary 5% subsets.

Hence, adversarial gradient saliency does not overfit to a specific perturbation method, and the identified RCPs reflect a structural property of the model rather than an artifact of the attack used during selection. Section 5.5 tests whether they are also robustness-specific.

Table 4: Overlap ratio (%) of RCPs under different attacks on ViT-B with CIFAR-10.
<table><tr><td></td><td>PGD-20</td><td>PGD-100</td><td>CW-20</td><td>AA</td></tr><tr><td>PGD-20</td><td>一</td><td>92.42</td><td>85.87</td><td>83.88</td></tr><tr><td>PGD-100</td><td>92.42</td><td></td><td>86.90</td><td>87.96</td></tr><tr><td>CW-20</td><td>85.87</td><td>86.90</td><td></td><td>82.82</td></tr><tr><td>AA</td><td>83.88</td><td>87.96</td><td>82.82</td><td>一</td></tr></table>

Table 5: Training with 5% of parameters with different criteria on ViT-B with CIFAR-10.
<table><tr><td>Selection criterion</td><td>Clean</td><td>PGD-20</td></tr><tr><td>Random</td><td>52.32</td><td>11.92</td></tr><tr><td>Weight magnitude</td><td>38.32</td><td>22.18</td></tr><tr><td>Clean-data saliency</td><td>54.33</td><td>25.36</td></tr><tr><td>Adv. saliency (SAFT)</td><td>82.55</td><td>44.41</td></tr></table>

## 5.5 Specificity of Robustness Critical Parameters

RCPs might simply be generally important parameters. We test this on ViT-B and CIFAR-10 with three controls. Set identity. The top 5% of parameters by pretrained weight magnitude share only 5.27% of their entries with the RCPs, and the top 5% by clean-data gradient saliency share 30.94%. Training under each criterion. We train separate models that update only the 5% selected by each criterion, with identical budget, schedule and seed (Table 5). Adversarial saliency beats clean-data saliency by 28.2 points in clean accuracy and 19.1 points on PGD-20 although the two selections overlap by a third, so the gain comes from the adversarial-specific part of the selection. Zeroing. Zeroing the RCPs of the trained SAFT model reduces PGD-20 accuracy by 86.0% in relative terms, compared with 57.8% for the weight-magnitude subset and 10.2% for a random subset (Appendix C). Together with the attack invariance in Table 4, these controls indicate that RCPs are specific to adversarial adaptation.

## 5.6 Generalization Across AT Objectives

To test whether SAFT generalizes beyond the standard PGD-AT objective, we apply it to two additional AT objectives, TRADES [46] and MART [38], both designed to address the clean-robust trade-off through loss formulations. Results on ViT-B and CIFAR-10 are reported in Table 6. SAFT outperforms LoRA-based methods under three AT objectives. The improvement under MART is large, where SAFT improves over LoRA by 4.02% in clean accuracy and 2.81% on PGD-20.

SAFT can surpass full-parameter AT under standard objectives. Under TRADES, SAFT also surpasses full-parameter AT on PGD-20 and PGD-100, and improves clean accuracy from 86.24% to 87.00%. Under MART, full-parameter AT achieves the highest robust accuracy overall, with SAFT trailing by 3.5–4.7% across attacks. We attribute this to MART’s strong reliance on per-example reweighting, which benefits from updating the full parameter set. Even so, SAFT remains the strongest PEFT-AT method under MART, improving over LoRA by 2.81% on PGD-20 and 2.50% on AutoAttack.

Table 6: SAFT under different AT objectives on ViT-B and CIFAR-10. SAFT consistently outperforms LoRA-based PEFT-AT under all three AT objectives.
<table><tr><td>Method</td><td>AT Objective</td><td>Clean</td><td>PGD-20</td><td>CW-20</td><td>PGD-100</td><td>AutoAttack</td></tr><tr><td>Full Parameter</td><td>TRADES</td><td>86.24</td><td>42.88</td><td>43.61</td><td>42.55</td><td>41.38</td></tr><tr><td>LoRA</td><td>TRADES</td><td>86.26</td><td>40.22</td><td>39.96</td><td>39.79</td><td>38.00</td></tr><tr><td>LNLoRA</td><td>TRADES</td><td>86.04</td><td>39.47</td><td>39.16</td><td>39.02</td><td>37.38</td></tr><tr><td>SAFT</td><td>TRADES</td><td>87.00</td><td>43.07</td><td>43.06</td><td>42.74</td><td>41.04</td></tr><tr><td>Full Parameter</td><td>MART</td><td>75.91</td><td>48.73</td><td>46.12</td><td>48.57</td><td>44.73</td></tr><tr><td>LoRA</td><td>MART</td><td>66.66</td><td>42.31</td><td>39.37</td><td>42.30</td><td>38.00</td></tr><tr><td>LNLoRA</td><td>MART</td><td>66.47</td><td>42.17</td><td>38.67</td><td>42.04</td><td>37.31</td></tr><tr><td>SAFT</td><td>MART</td><td>70.68</td><td>45.12</td><td>41.39</td><td>45.07</td><td>40.50</td></tr></table>

Comparison with LoRA variants. Table 7 further compares SAFT with DoRA [28], a stronger LoRA variant, on ViT-B and CIFAR-10 under PGD-AT. DoRA improves over LoRA by 1.60% on PGD-20 and 1.66% on AutoAttack but still trails SAFT despite a larger trainable budget. Since its directional update remains rank-r constrained, this indicates that the limitation of LoRA-based PEFT-AT lies in the low-rank constraint itself.

Table 7: Comparison with LoRA variants on ViT-B and CIFAR-10.
<table><tr><td>Method</td><td>Trainable</td><td>Clean</td><td>PGD-20</td><td>AA</td></tr><tr><td>LoRA</td><td>4.7M</td><td>79.51</td><td>41.58</td><td>38.63</td></tr><tr><td>LNLoRA</td><td>4.8M</td><td>79.59</td><td>41.79</td><td>39.07</td></tr><tr><td>DoRA</td><td>4.8M</td><td>80.16</td><td>43.18</td><td>40.29</td></tr><tr><td>SAFT</td><td>3.1M</td><td>82.55</td><td>44.41</td><td>41.61</td></tr></table>

## 5.7 Effect of Robustness Critical Parameter Percentage

To understand how robustness depends on the size of the selected RCPs subset, we vary the percentage from 1% to 7% on ViT-B / CIFAR-10 and compare with full-parameter AT (100%). Results are shown in Table 8. With only 1% of parameters selected, SAFT achieves 43.71% on PGD-20 and 40.82% on AutoAttack, already close to full-parameter AT. It indicates that the robustness-critical subspace is highly concentrated, and that even an extremely sparse update is sufficient to reach the robustness ceiling of full AT.

Table 8: Effect of the RCPs percentage on ViT-B and CIFAR-10. Even with only 1% of parameters selected as RCPs, SAFT recovers most of the robustness of full-parameter AT.

<table><tr><td>RCPs %</td><td>Clean</td><td>PGD-20</td><td>CW-20 PGD-100</td><td>AA</td></tr><tr><td>1%</td><td>81.51</td><td>43.71</td><td>43.58</td><td>43.35 40.82</td></tr><tr><td>3%</td><td>81.56</td><td>43.88</td><td>44.01 43.60</td><td>41.20</td></tr><tr><td>5%</td><td>82.55</td><td>44.41</td><td>44.87 44.04</td><td>41.61</td></tr><tr><td>7%</td><td>82.66</td><td>44.71</td><td>44.87 44.29</td><td>41.68</td></tr><tr><td>100% (Full AT)</td><td>85.04</td><td>43.60</td><td>44.57</td><td>42.96 41.16</td></tr></table>

Increasing the budget from 1% to 5% yields modest additional gains (PGD-20 from 43.71% to 44.41%, AutoAttack from 40.82% to 41.61%), and clean accuracy improves by about 1%. Beyond 5%, robustness gains become marginal (PGD-20 changes from 44.41% to 44.71% between 5% and 7%, with CW-20 unchanged at 44.87%). This fast saturation reinforces the sparsity finding in Fig. 1.

## 5.8 Training Cost and Further Analysis

Training cost. Table 9 reports the training cost on ViT-B and CIFAR-10 under an identical configuration. SAFT lowers peak memory by 0.97 GiB relative to Full AT and by 0.55 GiB relative to LoRA. Its time per step matches that of Full AT, since all gradients are still computed, and is 32% lower than that of LoRA. The 40-epoch times are extrapolated from the time per step. The parameter-side saving, at least 4kN(1 − ρ) bytes for N parameters and k optimizer states per parameter, grows linearly with model size: under Adam it reaches 2.18 and 4.54 GiB for ViT-L and ViT-H (Appendix A). Activations dominate peak memory in our setting, so the share of the parameter side grows for larger backbones and smaller batches.

Table 9: Measured training cost on ViT-B and CIFAR-10 on a single RTX 4090.
<table><tr><td>Method</td><td>Peak (GiB)</td><td>Step (s)</td><td>40 ep. (h)</td></tr><tr><td>Full AT</td><td>6.551</td><td>1.118</td><td>4.856</td></tr><tr><td>LoRA (r=32)</td><td>6.129</td><td>1.629</td><td>7.078</td></tr><tr><td>SAFT</td><td>5.582</td><td>1.114</td><td>4.840</td></tr></table>

Table 10: Overlap (%) between the RCPs recomputed at a given epoch and the calibration mask (ViT-B and CIFAR-10).
<table><tr><td>Epoch</td><td>1</td><td>10</td><td>20</td><td>30</td><td>40</td></tr><tr><td>Full AT</td><td>95.65</td><td>50.97</td><td>51.45</td><td>50.08</td><td>50.03</td></tr><tr><td>SAFT</td><td>95.65</td><td>55.73</td><td>56.42</td><td>56.75</td><td>57.58</td></tr></table>

Mask stability. Recomputing adversarial saliency during training in Table 10 and Appendix D shows that the RCP set drifts. Its overlap with the calibration mask falls from 95.65% after epoch 1 to 55.73% at epoch 10 and then stays stable, an order of magnitude above the roughly 5% expected for random subsets. The static mask is thus a workable rather than an optimal choice.

Distribution shift. On CIFAR-10-C, CIFAR-100-C, Tiny-ImageNet-C [17], CIFAR-10.1 [31] and CINIC-10 [8], SAFT achieves higher clean and PGD-20 accuracy than LoRA under every shift, attains the highest PGD-20 accuracy among all adapted models, and never suffers a larger relative accuracy drop than LoRA, as shown in Table 11 and Appendix E.

Table 11: No-attack / PGD-20 accuracy (%) on shifted data. Corruptions use severity 5. CINIC-10 IN uses a class-balanced 10,000-image subsample.
<table><tr><td>Benchmark</td><td>Pretrained</td><td>Full AT</td><td>LoRA</td><td>SAFT</td></tr><tr><td>CIFAR-10.1</td><td></td><td>72.80 / 25.05</td><td>66.50 / 27.10</td><td>70.00 / 28.20</td></tr><tr><td>CINIC-10 IN</td><td></td><td>60.62 / 22.54</td><td>56.77 / 23.32</td><td>59.05 / 24.30</td></tr><tr><td>CIFAR-10-C</td><td>77.01 / 0.00</td><td>69.58 / 25.21</td><td>65.50 / 26.79</td><td>68.29 / 27.86</td></tr><tr><td>CIFAR-100-C</td><td>55.03 / 0.01</td><td>43.81 / 11.37</td><td>41.80 / 11.43</td><td>43.77 / 12.49</td></tr><tr><td>Tiny-ImageNet-C</td><td>35.20 / 0.00</td><td>19.50 / 2.57</td><td>16.99 / 0.88</td><td>17.73 / 2.79</td></tr></table>

## 6 Conclusion

In this work, we revisit adversarial fine-tuning from a parameter-space perspective. Through gradient analysis, we find that adversarial optimization is concentrated in a sparse subset of parameters, which we call RCPs. A masking analysis further shows that this same subset is causally critical for the robust behavior of the trained model. Building on this finding, we propose SAFT, which identifies RCPs via adversarial gradient saliency and performs adversarial fine-tuning by updating only these parameters. Across three architectures, three datasets, and three AT objectives, SAFT consistently outperforms LoRA-based PEFT-AT and approaches or surpasses full-parameter AT in robustness while updating fewer than 5% of parameters. It reduces optimizer-state, persistent-gradient and peak training memory. These results suggest that the optimization subspace relevant to adversarial robustness is sparse rather than low-rank, offering a parameter-space perspective complementary to existing loss-level approaches to AT.

Limitations. We acknowledge three main limitations. First, despite updating fewer than 5% of parameters, SAFT does not bring wall-clock speedup or FLOP reduction over full-parameter AT. The current implementation uses unstructured sparsity, which reduces optimizer state and persistent gradient memory but does not accelerate forward or backward passes on standard hardware. Structured sparse selection at the head or row level is a promising direction to obtain real training speedup. Second, while SAFT improves the trade-off between trainable parameters and robustness, the cleanrobust trade-off itself is not eliminated. Clean accuracy of SAFT remains below full-parameter AT in most settings. Third, the RCP mask is static, although about half of it drifts during training (Section 5.8); we have not evaluated periodic re-selection.

## Acknowledgements

This work was supported in part by the National Natural Science Foundation of China (No. 62376202).   
We declare no competing interests.

## References

[1] M. Andriushchenko and N. Flammarion. Understanding and improving fast adversarial training. Advances in Neural Information Processing Systems, 33:16048–16059, 2020.

[2] A. Ansell, E. Ponti, A. Korhonen, and I. Vulic. Composable sparse fine-tuning for cross-lingual´ transfer. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1778–1796, 2022.

[3] N. Carlini and D. Wagner. Towards evaluating the robustness of neural networks. In 2017 IEEE Symposium on Security and Privacy (SP), pages 39–57, 2017.

[4] Y. Carmon, A. Raghunathan, L. Schmidt, J. C. Duchi, and P. S. Liang. Unlabeled data improves adversarial robustness. Advances in neural information processing systems, 32, 2019.

[5] Y. Chang, C. Guo, Y. Chang, and Y. Wu. Lora-mgpo: Mitigating double descent in low-rank adaptation via momentum-guided perturbation optimization. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 648–659, 2025.

[6] H. Cheng, M. Zhang, and J. Q. Shi. A survey on deep neural network pruning: Taxonomy, comparison, analysis, and recommendations. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):10558–10578, 2024.

[7] F. Croce and M. Hein. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. In International conference on machine learning, pages 2206–2216. PMLR, 2020.

[8] L. N. Darlow, E. J. Crowley, A. Antoniou, and A. J. Storkey. CINIC-10 is not ImageNet or CIFAR-10. arXiv preprint arXiv:1810.03505, 2018.

[9] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

[10] J. Frankle and M. Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In International Conference on Learning Representations, 2019.

[11] J. Fu, J. Fang, J. Sun, S. Zhuang, L. Geng, and Y. Liu. Loft: Lora-based efficient and robust fine-tuning framework for adversarial training. In 2024 International Joint Conference on Neural Networks (IJCNN), pages 1–8, 2024.

[12] I. J. Goodfellow, J. Shlens, and C. Szegedy. Explaining and harnessing adversarial examples. In International Conference on Learning Representations, 2015.

[13] D. Guo, A. M. Rush, and Y. Kim. Parameter-efficient transfer learning with diff pruning. In Proceedings ofthe 59th annual meeting ofthe associationfor computational linguistics and the 11th international joint conference on natural language processing (volume 1: Long papers), pages 4884–4896, 2021.

[14] S. Hao, Y. Zhou, and Y. Guo. A brief survey on semantic segmentation with deep learning. Neurocomputing, 406:302–321, 2020.

[15] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.

[16] Z. He. Deep learning in image classification: A survey report. In 2020 2nd International Conference on Information Technology and Computer Application (ITCA), pages 174–177. IEEE, 2020.

[17] D. Hendrycks and T. Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In International Conference on Learning Representations, 2019.

[18] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

[19] H. Huang and R. Balestriero. Allora: Adaptive learning rate mitigates lora fatal flaws. arXiv preprint arXiv:2410.09692, 2024.

[20] A. Ilyas, S. Santurkar, D. Tsipras, L. Engstrom, B. Tran, and A. Madry. Adversarial examples are not bugs, they are features. Advances in neural information processing systems, 32, 2019.

[21] Y. Ji, Y. Liu, Z. Zhang, Z. Zhang, Y. Zhao, X. Hao, G. Zhou, X. Zhang, and X. Zheng. Enhancing adversarial robustness of vision-language models through low-rank adaptation. In Proceedings of the 2025 International Conference on Multimedia Retrieval, pages 550–559, 2025.

[22] X. Jia, Y. Zhang, X. Wei, B. Wu, K. Ma, J. Wang, and X. Cao. Improving fast adversarial training with prior-guided knowledge. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(9):6367–6383, 2024.

[23] L. Jiao, F. Zhang, F. Liu, S. Yang, L. Li, Z. Feng, and R. Qu. A survey of deep learning-based object detection. IEEE access, 7:128837–128868, 2019.

[24] A. Krizhevsky and G. Hinton. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

[25] Y. Le and X. Yang. Tiny ImageNet visual recognition challenge. Stanford CS231N course report, 2015.

[26] Y. LeCun, Y. Bengio, and G. Hinton. Deep learning. nature, 521(7553):436–444, 2015.

[27] B. Liao, Y. Meng, and C. Monz. Parameter-efficient fine-tuning without introducing new latency. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 4242–4260, 2023.

[28] S.-Y. Liu, C.-Y. Wang, H. Yin, P. Molchanov, Y.-C. F. Wang, K.-T. Cheng, and M.-H. Chen. DoRA: Weight-decomposed low-rank adaptation. In International Conference on Machine Learning, 2024.

[29] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021.

[30] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations, 2018.

[31] B. Recht, R. Roelofs, L. Schmidt, and V. Shankar. Do ImageNet classifiers generalize to ImageNet? In International Conference on Machine Learning, pages 5389–5400, 2019.

[32] H. Robbins and S. Monro. A stochastic approximation method. The annals of mathematical statistics, pages 400–407, 1951.

[33] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115(3):211–252, 2015.

[34] A. Shafahi, M. Najibi, M. A. Ghiasi, Z. Xu, J. Dickerson, C. Studer, L. S. Davis, G. Taylor, and T. Goldstein. Adversarial training for free! Advances in neural information processing systems, 32, 2019.

[35] C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. Goodfellow, and R. Fergus. Intriguing properties of neural networks. In International Conference on Learning Representations, 2014.

[36] D. Tsipras, S. Santurkar, L. Engstrom, A. Turner, and A. Madry. Robustness may be at odds with accuracy. In International Conference on Learning Representations, 2019.

[37] S. Vadera and S. Ameen. Methods for pruning deep neural networks. Ieee Access, 10:63280– 63300, 2022.

[38] Y. Wang, D. Zou, J. Yi, J. Bailey, X. Ma, and Q. Gu. Improving adversarial robustness requires revisiting misclassified examples. In International Conference on Learning Representations, 2020.

[39] Z. Wang, H. Wang, C. Tian, and Y. Jin. Adversarial training of deep neural networks guided by texture and structural information. In Proceedings of the 31st ACM International Conference on Multimedia, pages 4958–4967, 2023.

[40] Z. Wang, H. Wang, C. Tian, and Y. Jin. Efficient adversarial training with multi-fidelity optimization for robust neural network. Neurocomputing, 585:127627, 2024.

[41] Z. Wang, H. Wang, C. Tian, and Y. Jin. Preventing catastrophic overfitting in fast adversarial training: A bi-level optimization perspective. In European Conference on Computer Vision, pages 144–160. Springer, 2024.

[42] F. K. Waseda, C.-C. Chang, and I. Echizen. Rethinking invariance regularization in adversarial training to improve robustness-accuracy trade-off. In The Thirteenth International Conference on Learning Representations, 2025.

[43] E. Wong, L. Rice, and J. Z. Kolter. Fast is better than free: Revisiting adversarial training. In International Conference on Learning Representations, 2020.

[44] Z. Yuan, J. Zhang, S. Shan, and X. Chen. Fulllora: Efficiently boosting the robustness of pretrained vision transformers. IEEE Transactions on Image Processing, 2025.

[45] D. Zhang, T. Zhang, Y. Lu, Z. Zhu, and B. Dong. You only propagate once: Accelerating adversarial training via maximal principle. Advances in neural information processing systems, 32, 2019.

[46] H. Zhang, Y. Yu, J. Jiao, E. Xing, L. El Ghaoui, and M. Jordan. Theoretically principled trade-off between robustness and accuracy. In International conference on machine learning, pages 7472–7482. PMLR, 2019.

[47] J. Zhang, F. Liu, D. Zhou, J. Zhang, and T. Liu. Improving accuracy-robustness trade-off via pixel reweighted adversarial training. In Forty-first International Conference on Machine Learning, 2024.

[48] Y. Zhang, G. Zhang, P. Khanduri, M. Hong, S. Chang, and S. Liu. Revisiting and advancing fast adversarial training through the lens of bi-level optimization. In International Conference on Machine Learning, pages 26693–26712. PMLR, 2022.

[49] H. Zheng, Z. Zhang, J. Gu, H. Lee, and A. Prakash. Efficient adversarial training with transferable adversarial examples. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1181–1190, 2020.

[50] K. Zhu, F. Hu, Y. Ding, W. Zhou, and R. Wang. A comprehensive review of network pruning based on pruning granularity and pruning time perspectives. Neurocomputing, 626:129382, 2025.

## A Memory Scaling of the Parameter Side

For N parameters stored in fp32 and an optimizer with k state tensors per parameter, full-parameter AT holds 4N bytes of weights, 4N of gradients and 4kN of optimizer state. SAFT holds the same weights and at most 4N of transient gradients, but optimizer state only for the trainable fraction $\rho ,$ i.e., at most $8 N + 4 k \rho N$ bytes. The saving of at least $4 k N ( 1 - \rho )$ depends only on model size and optimizer order. Table 12 extrapolates it to larger backbones for Adam $( k = 2 ) $ , using the trainable fraction of ViT-B $( \rho \approx 3 . 6 \% )$ . For SGD with momentum (k = 1), which we use in our experiments, the saving is halved.

The parameter count grows as $d ^ { 2 } L$ , whereas activation memory grows as B T d L for batch size B, sequence length T, width d and depth L. Our measurement in Table 9 (a large batch and only $T = 6 5$ tokens at $3 2 \times 3 2$ resolution with patch size 4) is therefore the least favourable case for SAFT. For larger backbones, smaller batches and gradient checkpointing, the parameter side becomes the dominant term.

Table 12: Analytic parameter-side memory (weights, gradients and Adam states, fp32) of Full AT and SAFT, extrapolated with the trainable fraction of ViT-B.
<table><tr><td>Backbone</td><td>Params</td><td>Full AT (GiB)</td><td>SAFT (GiB)</td><td>Saved (GiB)</td></tr><tr><td>ViT-B</td><td>85.2M</td><td>1.27</td><td>0.66</td><td>0.61</td></tr><tr><td>ViT-L</td><td>304M</td><td>4.53</td><td>2.35</td><td>2.18</td></tr><tr><td>ViT-H</td><td>632M</td><td>9.42</td><td>4.88</td><td>4.54</td></tr><tr><td>ViT-g/14</td><td>1.1B</td><td>16.39</td><td>8.49</td><td>7.90</td></tr><tr><td>ViT-G/14</td><td>1.8B</td><td>26.82</td><td>13.90</td><td>12.92</td></tr><tr><td>SwinV2-G</td><td>3.0B</td><td>44.70</td><td>23.16</td><td>21.54</td></tr><tr><td>InternViT-6B</td><td>5.9B</td><td>87.92</td><td>45.56</td><td>42.36</td></tr></table>

## B Other Datasets and a Convolutional Backbone

We repeat the gradient analysis of Section 3 on further datasets and on ResNet-50 [15]. Table 13 reports the share of the cumulative adversarial saliency mass captured by the top 5% of parameters. The concentration is consistent across datasets for ViT-B and weaker but still far above the uniform level of 5% for the convolutional backbone. On ResNet-50 / CIFAR-10, SAFT reaches 49.32% PGD-20 accuracy, compared with 48.95% for Full AT.

Table 13: Share of cumulative adversarial gradient saliency captured by the top 5% of parameters.
<table><tr><td>Backbone</td><td>Dataset</td><td>Top-5% share (%)</td></tr><tr><td>ViT-B</td><td>CIFAR-10</td><td>67.7</td></tr><tr><td>ViT-B</td><td>CIFAR-100</td><td>61.88</td></tr><tr><td>ViT-B</td><td>TinyImageNet</td><td>63.59</td></tr><tr><td>ResNet-50</td><td>CIFAR-10</td><td>43.15</td></tr></table>

## C Zeroing Interventions on the SAFT Model

Table 1 zeroes parameters of the Full AT model. Table 14 extends the intervention to the trained SAFT model on ViT-B and CIFAR-10 with two further criteria, each zeroing 5% of the parameters. Zeroing the RCPs causes by far the largest damage, whereas the weight-magnitude subset causes smaller and differently structured damage, even though large weights are generally important.

## D Drift of the Robustness-Critical Parameters during Training

We recompute adversarial gradient saliency at several epochs and compare each resulting top-ρ set with the mask fixed at calibration, for the SAFT run and, as a reference, a Full AT run (Table 10). The set reorganizes once as the model leaves its pretrained state and is stable thereafter. At the end of training it still overlaps the calibration mask by 57.58%, an order of magnitude above the roughly 5% expected for two random subsets of this size. Table 8 further shows that reducing the budget from 5% to 1% costs only 0.70 points on PGD-20 and 0.79 on AutoAttack. This suggests that robustness is carried by a core substantially smaller than the mask, and that a static mask suffices as long as it contains this core. We did not evaluate dynamic re-selection.

Table 14: Zeroing 5% of the parameters of the trained SAFT model. Relative drops are computed with respect to the unmasked SAFT model (82.55% clean, 44.41% PGD-20).
<table><tr><td>Zeroed subset (5%)</td><td>Clean</td><td>PGD-20</td><td>Rel. drop, clean</td><td>Rel. drop, PGD-20</td></tr><tr><td>Random</td><td>76.41</td><td>39.86</td><td>7.4%</td><td>10.2%</td></tr><tr><td>Weight magnitude</td><td>62.69</td><td>18.75</td><td>24.1%</td><td>57.8%</td></tr><tr><td>RCPs</td><td>30.60</td><td>6.23</td><td>62.9%</td><td>86.0%</td></tr></table>

## E Distribution Shift

Synthetic shifts. We report the mean corruption accuracy (mCA) on CIFAR-10-C, CIFAR-100-C and Tiny-ImageNet-C [17], averaged over the 15 standard corruption types at all 5 severity levels, and the relative degradation ∆% = (Clean − mCA)/Clean × 100, where Clean is the in-distribution accuracy of each model from Tables 2 and 3 (lower is better). Natural shifts. We evaluate on CIFAR-10.1 [31], an independently re-collected CSFATIFAR-10 test set, and on the ImageNet-sourced portion of the CINIC-10 [8] test split. Robustness under shift. We additionally evaluate PGD-20 under both shift families. For corruptions we use severity 5 with a stratified sample of 1,000 images per corruption type. For CINIC-10 we use a class-balanced 10,000-image subsample of the same ImageNet-sourced pool, which is why its no-attack accuracies in Table 11 differ slightly from Table 16. SAFT has a smaller relative degradation than LoRA on every benchmark and than Full AT on three of the five, and the highest PGD-20 accuracy under every shift.

Table 15: Mean corruption accuracy (mCA, %) and relative degradation (∆%, lower is better) on ViT-B.
<table><tr><td rowspan="2">Model</td><td colspan="2">CIFAR-10-C</td><td colspan="2">CIFAR-100-C</td><td colspan="2">Tiny-ImageNet-C</td></tr><tr><td>mCA</td><td>∆%</td><td>mCA</td><td>∆%</td><td>mCA</td><td>∆%</td></tr><tr><td>Pretrained</td><td>86.81</td><td>9.81</td><td>68.51</td><td>20.44</td><td>51.81</td><td>35.99</td></tr><tr><td>Full AT</td><td>76.16</td><td>10.45</td><td>52.85</td><td>17.61</td><td>28.01</td><td>52.38</td></tr><tr><td>LoRA</td><td>70.73</td><td>11.04</td><td>50.19</td><td>17.43</td><td>25.12</td><td>54.76</td></tr><tr><td>SAFT</td><td>73.98</td><td>10.38</td><td>51.99</td><td>16.95</td><td>25.73</td><td>52.70</td></tr></table>

Table 16: Accuracy (%) and relative degradation (∆%) under natural shift on ViT-B. CINIC-10 IN: ImageNet-sourced portion of the CINIC-10 test split.
<table><tr><td colspan="2"></td><td colspan="2">CIFAR-10.1</td><td colspan="2">CINIC-10 IN</td></tr><tr><td>Model</td><td>Clean</td><td>Acc.</td><td>∆%</td><td>Acc.</td><td>∆%</td></tr><tr><td>Full AT</td><td>85.04</td><td>72.80</td><td>14.39</td><td>60.11</td><td>29.31</td></tr><tr><td>LoRA</td><td>79.51</td><td>66.50</td><td>16.36</td><td>56.50</td><td>28.94</td></tr><tr><td>SAFT</td><td>82.55</td><td>70.00</td><td>15.20</td><td>58.79</td><td>28.78</td></tr></table>

## F Broader Impact

SAFT contributes to making adversarial training more accessible by reducing the optimizer-state and gradient memory cost of robust adaptation on large pretrained models, which lowers the barrier for researchers and practitioners with limited compute resources to deploy robust models in safetysensitive applications. As a defense method, our work has no foreseeable negative societal impact beyond the general dual-use nature of adversarial robustness research.