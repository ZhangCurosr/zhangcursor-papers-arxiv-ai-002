# IMPROVING GENERATIVE MODEL SELF-TRAINING WITH GEOMETRICALLY MODIFIED OUTPUTS

Patrick Batsell, Thomas Walker, & Richard Baraniuk

Department of Electrical and Computer Engineering

Rice University

Houston, TX 77005, USA

{pb52, tw78, richb}@rice.edu

## ABSTRACT

Self-training generative models – the continued improvement of a model using its own outputs – is becoming increasingly important as high-quality training data becomes scarce. However, naïvely finetuning on model-generated samples leads to degradation through model collapse and the model autophagy disorder. Negative-guidance self-training methods turn this degradation into a useful signal, using a model finetuned on its own outputs to guide the original model toward improved generation. Existing methods, however, take the negative signal in standard model outputs as given. We instead ask whether this signal can be explicitly strengthened. We introduce Geometrically Modified Outputs (GMOs), which reweight the singular values of the generator’s input-output Jacobian to increase the influence of its leading singular directions. This geometric modification amplifies the mode-seeking behavior and distortions of standard outputs, providing a stronger and more targeted negative signal for self-training. Across a range of one-step generative models, GMOs consistently improve the performance of negative-guidance methods, including Neon and SIMS, compared with using standard model outputs.

![](images/648e0398d62b786b4d5d387101b8850b73dcbdb041d6e36c2d0c373b6911c8f7.jpg)  
Standard Outputs

![](images/4d1edffb2eb008a17101e8ef198b794719c606fa96dcb3fc7e8b6762fc740de0.jpg)  
Perturbation

![](images/112da6e2123b5b1113c0105bd11a89e19b9ba152111a939dde8181b8cd9b8a49.jpg)  
Geometrically Modified Outputs

Figure 1: Geometrically Modified Outputs (GMOs) improve the performance of one-step generative models through self-training. Here we present examples of GMOs for the IMM (Zhou et al., 2025) one-step generative model trained on Imagenet256 (Krizhevsky et al., 2012). In the first panel, we show the model’s standard outputs; in the center panel, we show the perturbations of these outputs that yield the corresponding GMOs in the right panel (see Figure 11 for more examples). Using GMOs in self-training algorithms more effectively improves generative models.

## 1 INTRODUCTION

The performance of generative models for image generation has benefited greatly from increasing amounts of compute and high-quality training data (Kaplan et al., 2020; Henighan et al., 2020).

Although the path for increasing compute is relatively clear (AI, 2026), significantly expanding the quantity of high-quality training data represents a fundamental challenge (Muennighoff et al., 2025; Villalobos et al., 2024). Consequently, the field of self-training–the continued development of a model using only its own outputs–is gaining significant interest (Kim et al., 2023; Alemohammad et al., 2024b; Yuan et al., 2024; Alemohammad et al., 2026; Zheng et al., 2025; Feng et al., 2025; Karras et al., 2024).

Although the utilization of synthetic data has been established in adjacent fields, such as for data augmentation when training classifiers (Wang et al., 2023; He et al., 2023) and for bootstrapping reinforcement learning models through self-play (Silver et al., 2018), the situation is more nuanced with generative models. Naïve self-training–finetuning a generative model on its own outputs–is detrimental as it leads to the “model autophagy disorder” (MAD) (Alemohammad et al., 2024a) and “model collapse” (Shumailov et al., 2024). These phenomena describe the degradation of model quality due to the appearance of unwanted artifacts in its outputs and a reduction in the diversity of those outputs. Common strategies implemented by self-training algorithms include, self-play (Yuan et al., 2024), direct discriminative optimization guidance (Zheng et al., 2025), and output verification (Feng et al., 2025), with the strongest results observed using negative guidance (Alemohammad et al., 2024b; 2026; Karras et al., 2024).

In this paper, we ask: Can the effectiveness of self-training algorithms using negative guidance be improved by augmenting model outputs to amplify the negative signal?

We answer this question affirmatively with Geometrically Modified Outputs (GMOs), by demonstrating that they can be used to improve the effectiveness of the SIMS (Alemohammad et al., 2024b) and Neon self-training algorithm (Alemohammad et al., 2026).

GMOs leverage the observation of Batsell et al. (2026), and verified in Figure 2, that the effective ranks of the input-output Jacobians of generative models collapse during naïve self-training. Examples of how GMOs compare with standard model outputs are shown in Figures 1 and 11. GMOs represent a geometric perspective on the self-training problem, which has not been explicitly leveraged in prior works (Kim et al., 2023; Alemohammad et al., 2024b; Yuan et al., 2024; Alemohammad et al., 2026; Zheng et al., 2025; Feng et al., 2025; Karras et al., 2024).

## 2 THE GEOMETRY AND SELF-TRAINING OF GENERATIVE MODELS

In this section, we introduce relevant notation for the generative models we consider (Section 2.1), we describe what we mean by the geometry of generative models (Section 2.2), and we introduce the self-training problem (Section 2.3).

## 2.1 GENERATIVE MODELS

We consider generative models $G _ { \theta }$ with parameters θ that are resultant of a training algorithm A being applied to a set of training data D drawn from a distribution $p _ { \mathrm { d a t a } }$ . In this context, D represents high-quality real data. Generative model architecture can differ in the inference routines they use for generating outputs. Generally, we let I represent this inference routine and let $q _ { \theta , \kappa }$ denote the induced sampling distribution when I is implemented with hyperparameter κ.

In this paper, we are concerned with one-step inference routines (Song et al., 2023; Frans et al., 2025; Geng et al., 2025; Zhou et al., 2025; Deng et al., 2026; Yue et al., 2026; Yang et al., 2026). More specifically, I generates an output $\boldsymbol { s } \in \mathbb { R } ^ { d }$ based on a latent vector $z \in \mathbb { R } ^ { h }$ , which we summarize as a map $g _ { \theta } : \dot { \mathbb { R } ^ { h } } \overset { - } {  } \mathbb { R } ^ { d }$

## 2.2 THE GEOMETRY OF GENERATIVE MODELS

A standard output of a one-step generative model is generated by sampling a latent vector $z \in \mathbb { R } ^ { h }$ and applying the map $g _ { \theta } ( z ) = \pmb { \mathscr { s } } \in \mathbb { R } ^ { d }$ . We can decompose this output as $\pmb { s } = J _ { z } z + b _ { z }$ where $J _ { z } \in \overline { { \mathbb { R } } } ^ { d \times h }$ is the input-output Jacobian of $g _ { \theta }$ at z, and $b _ { z } : = s - J _ { z } z$ is the offset of g at z. We take the local geometry of g<sub>θ</sub> at z to be the spectral properties of $J _ { z }$ . Let $\pmb { u } _ { z } ^ { ( k ) } \in \mathbb { R } ^ { d }$ and $\pmb { v } _ { z } ^ { ( k ) } \in \mathbb { R } ^ { h }$ denote the $k ^ { \mathrm { { t h } } }$ left and right singular vectors of $J _ { z }$ , respectively, with $\sigma _ { z } ^ { ( k ) } \in \mathbb { R }$ being the corresponding singular value. Let $U _ { z } \in \mathbb { R } ^ { d \times r }$ and $V _ { z } \in \mathbb { R } ^ { h \times r }$ be the corresponding matrices of left and right singular vectors, such that $\pmb { J _ { z } } = \pmb { U _ { z } } \mathrm { d i a g } \left( \sigma _ { z } ^ { \left( 1 \right) } , \ldots , \sigma _ { z } ^ { \left( r \right) } \right) \pmb { V _ { z } } ^ { \top }$

## 2.3 SELF-TRAINING GENERATIVE MODELS

Self-training is the problem of constructing a learning algorithm that uses only the model’s outputs to yield a generative model $G _ { \theta ^ { \prime } }$ that outperforms $G _ { \theta }$ . More formally, self-training involves applying a learning algorithm $\tilde { \mathcal { A } }$ on a set of data S sampled from $^ { q _ { \theta , \kappa } }$ to identify a set of parameters $\theta ^ { \prime }$ such that $G _ { \theta ^ { \prime } }$ performs better than $G _ { \theta }$ . Typically, self-training algorithms are constrained by a compute budget $B ,$ which we measure as the cumulative number of images observed when using ${ \tilde { \cal A } } .$

Many approaches exist for tackling the self-training problem. The naïve approach is to continue the learning algorithm $\mathcal { A }$ on $s ;$ however, several studies observe that this leads to the degradation of the generative model (Shumailov et al., 2024; Alemohammad et al., 2024a). Principled methods for overcoming this, include self-play (Yuan et al., 2024), direct discriminative optimization guidance (Zheng et al., 2025), negative guidance (Alemohammad et al., 2024b; 2026; Karras et al., 2024), and output verification (Feng et al., 2025).

In this paper, we consider whether the effectiveness ofself-training algorithms using negative guidance can be improved by augmenting model outputs to amplify the negative signal.

In particular, we focus on the Neon self-training algorithm (Alemohammad et al., 2026), but we also consider SIMS (Alemohammad et al., 2024b). We focus on Neon as it has been demonstrated to efficiently and effectively improve the performance of state-of-the-art generative models more effectively than other self-training algorithms utilizing negative guidance (Alemohammad et al., 2026). Neon works by finetuning a base model $G _ { \theta }$ on $s$ to get $G _ { \tilde { \theta } } .$ It then uses the direction between θ and $\tilde { \theta }$ as a negative signal to generate parameters $\theta ^ { \prime } = ( 1 + w ) \theta - w \tilde { \theta }$ for some $w \in \mathbb { R } ^ { + }$ . This process is summarized in Algorithm 1.

```perl
Algorithm 1 Neon
Require: Base model $G _ { \theta } ,$ , Inference routine I with hyperparameters $\kappa ,$ Weight-merging parameter
$w ,$ Training budget B
1: Sample $s$ from $q _ { \theta , \kappa } .$
2: $G _ { \tilde { \theta } } \gets$ FineTune $\left( { \cal G } _ { \theta } , { \cal S } , { \cal B } \right)$
3: $\theta ^ { \prime }  ( 1 + w ) \theta - w \tilde { \theta }$
```

Neon’s guarantee that this extrapolation improves the model relies on $s$ being mode-seeking: concentrated toward the high-density regions of the model’s own distribution (Alemohammad et al., 2026).

## 3 GEOMETRICALLY MODIFIED OUTPUTS

We propose Geometrically Modified Outputs (GMOs) as a strategy for amplifying the mode-seeking nature of model outputs with the intention of improving negative guidance self-training algorithms. GMOs do this by amplifying the influence of the top singular vectors of the model’s generator map. This is motivated by the observations of Batsell et al. (2026) which show that under naïve self-training, the input-output Jacobians of generators become increasingly collapsed. More specifically, their effective ranks<sup>1</sup> collapse and low-quality artifacts appear in the more prominent singular vectors. In Figure 2, we verify the collapse of the effective rank for a MeanFlow generative model (Geng et al., 2025) naïvely self-training on ImageNet256 (Krizhevsky et al., 2012). Moreover, we verify that GMOs exhibit amplified mode-seeking in Section B.

![](images/1e4cdf0f3b9aa94ae891f9b8640a15cbf3d43219cabe3ca94e25bcaca6a4f92c.jpg)

![](images/0359eb873b6f1f8c88695e9be4ac9c79eeebb1064e34fa1ffee1da17660f1767.jpg)  
Figure 2: The geometry of a generative model collapses under naïve self-training. Here we finetune a MeanFlow SiT-B/2 model pre-trained on ImageNet256 for 50 epochs on 30, 000 of its own outputs. Throughout training, we monitor the FID (Heusel et al., 2017) of the model (left), and its effective rank on a collection of 16 fixed latent vectors (right).

## 3.1 GMO GENERATION

Consider a standard output $\pmb { \mathscr { s } } = J _ { z } \pmb { \mathscr { z } } + b _ { z } \in \mathbb { R } ^ { d }$ generated from a latent vector $z \in \mathbb { R } ^ { h }$ . The corresponding α-GMO is given by $\tilde { s } = \tilde { J } _ { z } z + b _ { z }$ , where $\tilde { \pmb { J } } _ { z } = \pmb { U } \mathrm { d i a g } \left( \tilde { \sigma } _ { z } ^ { ( 1 ) } , \dots , \tilde { \sigma } _ { z } ^ { ( r ) } \right) \pmb { V } _ { z } ^ { \top }$ with

$$
\tilde { \sigma } _ { z } ^ { ( k ) } = \left\{ \begin{array} { l l } { \sqrt { \left( 1 - \alpha \right) \left( \sigma _ { z } ^ { ( 1 ) } \right) ^ { 2 } + \alpha \left\| J _ { z } \right\| _ { F } ^ { 2 } } } & { k = 1 } \\ { \sqrt { 1 - \alpha } \sigma _ { z } ^ { ( k ) } } & { k \geq 2 . } \end{array} \right.
$$

This reweighting preserves total spectral energy while transferring an α fraction of the trailing energy to the leading component (Appendix A.1).

Intuitively, the α-GMO increases the influence that the top singular vectors have on the generated output. With α equal to zero, the generated output remains unchanged, but with α equal to one, the output is entirely determined by the top singular vectors. From the specific perspective of improving Neon, GMOs can be thought of as amplifying the “mode-seeking” nature of model outputs. We explore this in Section B.

With Algorithm 2, we provide an exact implementation of α-GMOs that requires only the computation of the top spectral statistics of $J _ { z }$

Theorem 1. For a generator $g : \mathbb { R } ^ { h }  \mathbb { R } ^ { d } ,$ , a vector $z \in \mathbb { R } ^ { h }$ and $\alpha \in [ 0 , 1 ]$ , Algorithm 2 returns the $\alpha { - } G M O \ o f s = g ( z )$

Proof. See Section A.

Algorithm 2 Geometrically Modifying Outputs   
Require: Generator $g : \mathbb { R } ^ { h }  \mathbb { R } ^ { d } , \alpha \in [ 0 , 1 ] .$   
1: ${ \bf \bar { S } a m p l e } z \in \mathbb { R } ^ { h }$   
2: $s \gets g ( z )$   
3: $b _ { z }  s - J _ { z } z$   
4: $( { \pmb u } , { \pmb v } , \sigma ) \gets \mathrm { T o p S i g V e c } ( { g } , { z } )$   
5: $E  \| J _ { z } \| _ { F } ^ { 2 }$   
6: $\tilde { \sigma }  \sqrt { ( 1 - \alpha ) \sigma ^ { 2 } + \alpha E }$   
7: $\tilde { s } \gets \dot { \sqrt { 1 - \alpha } } \left( s - b _ { z } \right) + \left( \tilde { \sigma } - \sqrt { 1 - \alpha } \sigma \right) \left( v ^ { \top } z \right) u + b _ { z }$

## 3.2 COMPUTATIONAL COMPLEXITY OF GENERATING GMOS

Computing exact GMOs involves manipulations involving the input-output Jacobian $J _ { z } \in \mathbb { R } ^ { d \times h }$ For high-dimensional generative models, where both the latent dimension h and data dimension d are large, this poses a severe computational bottleneck.

![](images/7910b0a0f32988be36f5134472f571891d94c5ab839a59465a52bf5933c15ab2.jpg)

![](images/c022e0360fe189b696462ef3bdc5156dfb66fee10e6f7f3e224c1a1648dc02ff.jpg)  
Figure 3: Here we consider computing spectral statistics from the input-output Jacobians of a MeanFlow one-step generator (Geng et al., 2025) with a UNet architecture (Ronneberger et al., 2015; Song & Ermon, 2019) trained on CIFAR10 (Krizhevsky & Hinton, 2009). Ten latent vectors are sampled, and $E , \sigma _ { 1 } , \mathbf { \boldsymbol { u } } _ { 1 }$ , and ${ \pmb v } _ { 1 }$ are either computed exactly or using approximations. Power iteration is implemented 30 times to obtain estimates for $\hat { \sigma } _ { 1 } , \hat { \ b u } _ { 1 }$ , and $\hat { v } _ { 1 }$ . Similarly, 100 samples are used to generate the Hutchinson estimate $\hat { E } .$ . In the first panel, we record $\left| \hat { \sigma } _ { 1 } - \sigma _ { 1 } \right| / \sigma _ { 1 }$ . In the second panel, we record $1 - \langle \hat { \pmb u } _ { 1 } , \pmb u _ { 1 } \rangle \big / \| \hat { \pmb u } _ { 1 } \| _ { 2 } \| \pmb u _ { 1 } \| _ { 2 }$ , and similarly for ${ \pmb v } _ { 1 }$ . In the third panel, we record $\big | \hat { E } { - } E \big | \big / E$ . In the fourth panel, we record the time taken to exactly compute the Jacobian and its singular value decomposition (Exact) and the time taken to perform 20 power iterations and 20 Hutchinson approximations (Approximate).

Here, we provide a breakdown of the computational complexity of exactly implementing Algorithm 2 and compare it with an approximate implementation using power iteration and Hutchinson’s estimator. A point to note is that, this analysis is only applicable once for a given model, as given a latent vector $z ,$ , lines 6 and 7 of Algorithm 2 can be applied to multiple different α values to generate GMOs for different α values simultaneously.

Let $\mathcal { O } ( C _ { f } )$ denote the time complexity of a single forward pass of the generator $g ( z )$

Exact Implementation. To compute GMOs exactly, one must instantiate the full Jacobian matrix $J _ { z }$ . Using automatic differentiation, this requires either h forward-mode passes (Jacobian-Vector Products, JVPs) or d reverse-mode passes (Vector-Jacobian Products, VJPs). This scales linearly with the dimensions, resulting in a time complexity of $\mathcal { O } ( \operatorname* { m i n } ( d , h ) \cdot C _ { f } )$ . The memory complexity to store the dense matrix is $\bar { \mathcal { O } } ( d \cdot h )$ . Once $\scriptstyle { \bar { \mathbf { J } } } _ { z }$ is materialized, computing the exact singular value decomposition to find ${ \mathbf { } } u , v ,$ , and σ requires $\mathcal { O } \left( \operatorname* { m i n } \left( d ^ { 2 } h , d h ^ { 2 } \right) \right)$ operations. Computing the exact Frobenius norm requires an additional $O ( d \cdot h )$ operations.

In total, the exact time complexity is on the order of $\mathcal { O } \left( \operatorname* { m i n } ( d , h ) \cdot C _ { f } + \operatorname* { m i n } \left( d ^ { 2 } h , d h ^ { 2 } \right) \right)$ , with spatial memory scaling at $O ( d \cdot h )$ .

Approximate Implementation. To improve the tractability of computing GMOs, we can sidestep the materialization of $J _ { z }$ entirely. Modern deep learning frameworks support fast JVPs and VJPs in roughly $\mathcal { O } ( C _ { f } )$ time, with spatial memory comparable to that of a standard forward or backward pass.

For computing approximations for u, v and $\sigma ,$ we can use power iteration. Power iteration works by repeatedly applying the matrix $J _ { z } ^ { \top } J _ { z }$ to a random initialization vector. In practice, evaluating $\dot { J _ { z } } ^ { \top } ( \dot { J } _ { z } w )$ amounts to one JVP followed by one VJP. For k iterations, the time complexity is $\mathcal { O } \left( k \cdot C _ { f } \right)$ .

For computing an approximate value of the squared Frobenius norm, we can use Hutchinson’s estimator. The squared Frobenius norm equals the trace of the Gram matrix $\left\| J _ { z } \right\| _ { F } ^ { 2 } = \operatorname { t r } ( J _ { z } ^ { \top } J _ { z } )$ Hutchinson’s trace estimator approximates this via $\begin{array} { r } { \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \big \| J _ { z } \pmb { w } _ { i } \big \| _ { 2 } ^ { 2 } } \end{array}$ , where ${ \pmb w } _ { i }$ are standard normal vectors. Calculating this requires m independent $\mathrm { J } \dot { \mathrm { V } } \dot { \mathrm { P s } } ,$ , yielding a time complexity of $\mathcal { O } \left( m \cdot C _ { f } \right)$ .

Consequently, the total time complexity for computing GMOs reduces to $\mathcal { O } \left( \left( k + m \right) \cdot C _ { f } \right)$ . Similarly, the memory complexity drops to $\mathcal { O } ( \dot { d } + h )$ , since we only have to store the vectors involved in the JVP or VJP computations.

![](images/912704a1f636dd424df6eb69b08a9094c6fc979d564028300878dce4b3ef3727.jpg)

![](images/a4ab71568dcbd81979a6f3e33b206732ef7d4e7bdb9b67e26b2e2001baaeee83.jpg)

![](images/addf04d226d98c13281335ddd5223692e90751e0e4147cebdfee0734b00c8961.jpg)  
α = 0.0 α = 0.05 γ = 0.05  
Figure 4: GMOs provide meaningful perturbations to a model’s outputs that improve the effectiveness of Neon. Here, we consider applying the Neon self-training algorithm to the Mean-Flow (Geng et al., 2025) SiT-B/2 model (Ma et al., 2024) trained on ImageNet256 (Krizhevsky et al., 2012). We generated collections of 30, 000 GMOs for alpha values 0.0 (standard outputs), 0.05, and 0.2. We then finetuned the pre-trained checkpoint with a maximum compute budget of $1 . 8 \times 1 0 ^ { 6 }$ At regular checkpoints, we apply Neon with different weight-merging parameters w in the range [0, 2]. In the first and second panels, we show the minimum FID value achieved at each fine-tuning checkpoint with the corresponding weight-merging parameter w, respectively. In the third panel, we visually compare standard outputs to GMOs and the Gaussian baseline (see Figure 15 for more examples).

The computational efficiency of the approximations is highlighted in the fourth panel of Figure 3, which shows that the approximate implementation of Algorithm 2 is around 2 times faster than its exact implementation. In particular, it makes the computation of GMOs tractable in settings where memory constraints would otherwise prevent it.

Importantly, with the three panels of Figure 3, we ensure that we do not lose much by utilizing this approximate implementation. The method is therefore insensitive to the power-iteration count above roughly ten iterations, and we use 20. In Figure 13, we qualitatively demonstrate that these approximations are sound by visually comparing the exact and approximate left singular vectors.

## 4 EXPERIMENTS

In this section, we examine the characteristics of GMOs in comparison to standard outputs and how Neon with GMOs differs from Neon using standard outputs.

## 4.1 GMOS OUTPERFORM STANDARD OUTPUTS AND GAUSSIAN PERTURBED OUTPUTS

Here, we empirically verify that applying the Neon self-training to GMOs is more effective than applying it to standard outputs. In particular, we isolate this benefit to the geometric structure of GMOs rather than to the mere effect of injecting noise. We do so by applying Neon to standard outputs perturbed with Gaussian noise of an equivalent magnitude.

To construct the outputs for the equivalent Gaussian baseline, we match the GMO perturbation amplitudes on a per-sample basis. For a given standard output and its corresponding GMO, we compute the root mean square (γ) of their difference to measure the exact perturbation amplitude. We then draw isotropic Gaussian noise scaled by γ and add it to the standard output. This ensures that the Gaussian-perturbed outputs have the exact same total perturbation energy as the GMOs, but lack the directional guidance derived from the input-output Jacobian.

In Figure 4, we see that GMOs with α equal to 0.05 provide the best-performing generative model and outperform the Gaussian noise baseline.

Qualitatively, with the third panel of Figure 4, we see that the perturbations of GMOs are more similar to the standard output, whereas Gaussian noise seems to create large distortions in the output.<sup>2</sup>

## 4.2 THE EFFECT OF GMOS ON FID, PRECISION, RECALL, DENSITY AND COVERAGE

Here, we investigate the underlying mechanics of how GMOs improve generative model performance by examining FID, precision, recall, density, and coverage (Heusel et al., 2017; Kynkäänniemi et al., 2019; Naeem et al., 2020). We evaluate an IMM one-step generator (Zhou et al., 2025) trained on ImageNet256 (Krizhevsky et al., 2012) across a range of GMO coefficients α. For each evaluated curve, we report the optimum performance achieved across a fine-tuning budget B and merge weight w. Alongside precision and recall, we report density and coverage, their outlierrobust refinements (Naeem et al., 2020). Recall is inflated by a few generated outliers with large neighborhoods, and precision by outliers in the real data.

As illustrated in the left panel of Figure 5, applying GMOs yields a net improvement in FID compared to standard Neon outputs. The FID score drops from its baseline (α equal to zero) to reach an optimal minimum around α equal to 0.2 before slightly increasing at higher perturbation strengths. The precision, recall, density, and coverage as a function of the GMO coefficient provides better insights into how this improvement is achieved. The center and right panels show that as α increases from 0 to 0.4, the model’s precision steadily increases while its recall steadily decreases – a pattern that, taken at face value, would suggest GMOs trade diversity for quality. The outlier-robust measures tell a different story: across all architectures (Table 4), density and coverage are maintained or improved from standard-outputs Neon to GMOs, and where the recall estimate dips under GMOs, most visibly on IMM (−.007 relative to standard-outputs Neon), coverage instead rises (+.021). The FID improvements of GMOs therefore reflect higher robust fidelity together with preserved or improved robust diversity. GMOs improve image quality and diversity simultaneously, rather than trading one for the other.

![](images/b85f76f65eecb9098d779098b85b9b7ed482d762063bc043c82451c29f12c37d.jpg)

![](images/bd4565a9dae137f4d2af58c5f4f93bb5c11f7173788cdcffb1f0dce21e0c3661.jpg)

![](images/99062f47cdd93951d5233219e78581bc105513732d45e3d46b344f7d7d0a4e9d.jpg)

![](images/f903382d7af71acc0ed63945c6ec80f545beeda9ac9358ee6fcda9036a85fc64.jpg)

![](images/77e367bd1fa9a694cf13d947db80962ee6685280faf9869319c56791342f3018.jpg)  
Figure 5: Neon with GMOs improves image quality and diversity. For the IMM 1-step generator (Zhou et al., 2025) trained on ImageNet256 (Krizhevsky et al., 2012), we plot FID, precision, recall, density, and coverage (Heusel et al., 2017; Kynkäänniemi et al., 2019) as a function of the GMO coefficient α. Here, α equal to zero corresponds to Neon (Alemohammad et al., 2026) applied to standard outputs, while $\alpha > 0$ corresponds to Neon applied to α-GMOs. Each curve reports the per-α optimum over fine-tuning budget B and merge weight w. For more details, refer to Appendix D.2.

## 4.3 GMOS TRANSFER ACROSS MODEL SIZES AND INFERENCE STRATEGIES

Here, we explore whether GMOs can be transferred across model sizes and inference strategies. In the left panel of Figure 6, we explore the former with MeanFlow models (Geng et al., 2025) trained on ImageNet256 (Krizhevsky et al., 2012). More specifically, GMOs computed from a SiT-B/2 can be successfully used to improve the performance of Neon applied to a larger SiT-L/2 model. Although it should be noted that using GMOs from the SiT-B/2 model to improve the SiT-L/2 is more challenging, as the FID becomes more sensitive to the merging parameter w.

Similarly, in the right panel of Figure 6, we show that GMOs extracted from a one-step IMM (Zhou et al., 2025) generative model can be used to improve a two-step IMM on CIFAR10 (Krizhevsky & Hinton, 2009). These results are significant because computing GMOs for larger generative models or those with more demanding inference strategies is more computationally expensive.

## 4.4 GMOS IMPROVE STATE-OF-THE-ART GENERATIVE MODELS

Here, we demonstrate that Neon with GMOs can improve the performance – measured by FID (Heusel et al., 2017) – more effectively than Neon with standard outputs.

As detailed in Table 1, we evaluate the impact of integrating GMOs into the Neon self-training algorithm across several one-step generative models, including IMM (Zhou et al., 2025), Mean-

![](images/9bb9a50a53370704aae3b2d7c889db4959f5b1a339139a0c47ece2169c2eaef9.jpg)  
w

![](images/798127c606323aa6e184f35f1659ea5106fb474baf32ff473344fefcb64d8a1c.jpg)  
w  
Figure 6: GMOs are transferable across generative models of different sizes and inference strategies. In the left panel, we consider using ImageNet256 GMOs from a SiT-B/2 MeanFlow model to improve a SiT-L/2 MeanFlow model using Neon. We compare this to Neon applied directly to the SiT-L/2 model’s outputs and GMOs. A computational fine tuning budget of $1 . 2 \times 1 0 ^ { 6 }$ is used in every case, and GMOs are generated with α equal to 0.1. In the right panel, we consider using CIFAR10 GMOs from a one-step IMM model to improve a 2-step IMM model using Neon. We compare this to Neon applied directly to the two-step IMM model’s outputs and GMOs. For more experimental detail on this particular experiment, refer to Appendix D.1.

Flow (Geng et al., 2025), and AlphaFlow (Zhang et al., 2026) architectures. The evaluations are conducted on models pre-trained on ImageNet256 (Krizhevsky et al., 2012). The results clearly indicate that self-training with GMOs consistently yields superior generative performance compared to self-training with standard outputs. Across all evaluated model scales, the application of GMOs pushes the FID below both the pre-trained base model baseline and the standard Neon baseline. As shown in Table 1, Neon with GMOs achieves a lower FID than Neon with standard outputs across all five evaluated architectures. Measured relative to the FID reduction that standard Neon achieves over the base model, GMOs increase this reduction by 17–105% across the five architectures, nearly doubling it on IMM and more than doubling it on AlphaFlow SiT-B/2 (see Table 5).

Furthermore, the strength of the negative signal is visible in the finetuning dynamics. Models can attain their optimal FID using GMOs with a smaller fine-tuning budget (B) than is required to reach optimal performance with standard outputs. We quantify this across a broader sweep of α values on a Meanflow model trained on CIFAR10; we find that increasing α systematically lowers both the optimal fine tuning budget B and the optimal merge weight w required by Neon (Appendix C). This confirms that the targeted signal provided by GMOs not only raises the ultimate performance ceiling but also reaches the optimal checkpoint sooner, because GMOs induce the mode-seeking degradation faster. We show that GMOs statistically significantly improve generative model self-training across all architectures tested (see Table 5.)

Table 1: The utilization of GMOs in the Neon self-training algorithm yields better generative models. Here, we implement the Neon self-training algorithm on various one-step generative models trained on ImageNet256. We consider Neon on the standard output of these generators, or on the corresponding GMOs, across various compute levels (as a percentage of pre-training compute), weight-merging parameters w, and α values for GMOs (refer to Appendix D.2 for more details). Among the best-performing configurations, we report the average and standard deviation of the FID across five random seeds and 50, 000 output samples. We provide a repository here containing the model checkpoints obtained using GMOs.
<table><tr><td rowspan="2">Architecture</td><td rowspan="2">Base FID</td><td colspan="3">Neon w/ Standard Outputs</td><td colspan="4">Neon w/ GMOs</td></tr><tr><td>B</td><td>w</td><td>FID</td><td>B</td><td>w</td><td>α</td><td>FID</td></tr><tr><td>IMM</td><td>8.34</td><td> $2 . 5 \times 1 0 ^ { 6 } ( 6 . 1 \times 1 0 ^ { - 3 \% } )$ </td><td>1.6</td><td>7.32(±0.07)</td><td> $2 . 8 7 \times 1 0 ^ { 6 } ( 7 . 0 \times 1 0 ^ { - 3 \% } )$ </td><td>1.6</td><td>0.2</td><td>6.25(±0.03)</td></tr><tr><td>MeanFlow SiT-B/2</td><td>6.08</td><td> $7 . 2 \times 1 0 ^ { 5 } ( 0 . 2 5 \% )$ </td><td>0.8</td><td> $5 . 7 0 \dot { ( } \pm 0 . 0 1 \dot { ) }$ </td><td> $4 . 8 \times 1 0 ^ { 5 } ( 0 . 1 7 \% )$ </td><td>1.2</td><td>0.05</td><td> $\mathbf { 5 . 6 0 } \dot { ( } \pm \mathbf { 0 . 0 1 } \dot { ) }$ </td></tr><tr><td>MeanFlow SiT-L/2</td><td>3.97</td><td> $1 . 8 \times 1 0 ^ { 6 } \dot { ( } 0 . 6 3 \% \dot { ) }$ </td><td>0.5</td><td> $3 . 7 4 ( \pm 0 . 0 2 ) $ </td><td> $1 . 8 \times 1 0 ^ { 6 } \dot { ( 0 . 6 3 \% ) }$ </td><td>0.3</td><td>0.1</td><td> $\mathbf { 3 . 7 0 } ( \pm \mathbf { 0 . 0 2 } )$ </td></tr><tr><td>AlphaFlow SiT-B/2</td><td>5.55</td><td> $4 . 8 \times 1 0 ^ { 5 } \dot { ( } 0 . 1 6 \% \dot { ) }$ </td><td>0.6</td><td>5.36(±0.02)</td><td> $4 . 8 \times 1 0 ^ { 5 } \dot { ( } 0 . 1 6 \% \dot { ) }$ </td><td>0.8</td><td>0.1</td><td> ${ \bf 5 . 1 6 } ( \pm 0 . 0 2 ) $ </td></tr><tr><td>AlphaFlow SiT-XL/2</td><td>2.93</td><td> $3 \times 1 0 ^ { 5 } \dot { ( 0 . 1 \% ) }$ </td><td>1.2</td><td>2.64(±0.02)</td><td> $3 \times 1 0 ^ { 5 } \dot { ( 0 . 1 \% ) }$ </td><td>1.1</td><td>0.05</td><td> $\mathbf { 2 . 5 9 } ( \pm 0 . 0 2 )$ </td></tr></table>

## 4.5 GMOS IMPROVE NEGATIVE GUIDANCE BEYOND NEON

GMOs operate purely in data space and are agnostic to the self-training algorithm that utilizes them. We chose Neon as the primary algorithm because it is the strongest and most efficient negative guidance method, requiring no additional model at inference and no sampling overhead. To show that our contribution is not solely tied to Neon, we evaluate a second, structurally different algorithm: SIMS-style guidance (Alemohammad et al., 2024b) keeps the finetuned model in the sampling loop and extrapolates predictions at inference time, computing $\left( 1 + \omega \right) f _ { \theta } - \omega f _ { \tilde { \theta } }$ at every sampling step instead of merging weights once.

On IMM, guidance with the model finetuned on standard outputs improves the base FID by 0.75 (from 8.34±.06 to 7.59±.06), while the same guidance with the model finetuned on GMOs improves it by 1.42 (to 6.92 ± .05), nearly twice the improvement. GMOs therefore strengthen both consumers of negative signals that we test, weight-space extrapolation and output-space guidance.

## 5 DISCUSSION

Summary. In this paper, we introduced Geometrically Modified Outputs (GMOs) and demonstrated that they improve the performance of the Neon self-training algorithm for one-step generative models. GMOs are an augmentation technique that modifies model outputs by reweighting the singular values of the generator’s input-output Jacobian. This structural augmentation provides a stronger and more targeted negative signal for self-training algorithms, such as Neon and SIMS. Through empirical evaluations, we demonstrated that integrating GMOs into multiple self-training frameworks consistently yields superior generative performance.

Relation to other self-improvement methods. GMOs should be viewed as an enhancement to negative-signal construction, rather than a competing standalone self-training framework. They modify the synthetic outputs used to train the negative model while retaining the downstream fine-tuning and guidance procedures. We demonstrate their utility with both Neon’s weight-space extrapolation and SIMS-style inference-time guidance.

The computational requirements of the downstream method remain important. SIMS (Alemohammad et al., 2024b) and Autoguidance (Karras et al., 2024) require auxiliary-model predictions during sampling, while DDO (Zheng et al., 2025) uses multiple self-play rounds and reports a training budget of approximately 12% of pretraining compute. We primarily evaluate GMOs with Neon (Alemohammad et al., 2026), whose merged model requires no additional inference-time evaluations. In this setting, GMOs add offline sample-construction cost while preserving Neon’s sampling efficiency. Their effectiveness under both Neon and SIMS motivates investigating whether geometry-based output augmentation can also strengthen other negative-signal methods, including DDO and Autoguidance.

Limitations and Future Directions. Our results suggest that Geometrically Modified Outputs (GMOs) can improve self-training by leveraging the generator’s geometric structure, although several directions remain open for future work.

First, GMOs require specifying α. Although outputs for multiple α values can be generated in parallel, applying Neon to each is costly. In our experiments, a fixed $\alpha = 0 . 1$ without per-model tuning outperforms standard-outputs Neon on four of five ImageNet-trained architectures and is within seed noise on the fifth. Generating GMOs at additional α values is nearly free because they reuse the same probe computation (Algorithm 2); only evaluation and finetuning scale with the number of values tested. However, a search-free heuristic for estimating the optimal α for a given model and dataset would be valuable. Second, we explore GMOs only in the image domain; extending them to other modalities and architectures remains future work.

Acknowledgements This work was supported by ONR grant N00014-23-1-2714, DOE grant DE-SC0020345, DOI grant 140D0423C0076, and a Google Cloud Computing Award.

## REFERENCES

Epoch AI. Trends in Artificial Intelligence, February 2026. URL https://epoch.ai/trends.

Sina Alemohammad, Josue Casco-Rodriguez, Lorenzo Luzi, Ahmed Imtiaz Humayun, Hossein Babaei, Daniel LeJeune, Ali Siahkoohi, and Richard Baraniuk. Self-Consuming Generative Models Go MAD. In International Conference on Learning Representations, 2024a.

Sina Alemohammad, Ahmed Imtiaz Humayun, Shruti Agarwal, John Collomosse, and Richard Baraniuk. Self-Improving Diffusion Models With Synthetic Data. arXiv:2408.16333, 2024b.

Sina Alemohammad, Zhangyang Wang, and Richard Baraniuk. Neon: Negative Extrapolation From Self-training Improves Image Generation. In International Conference on Learning Representations, 2026.

Patrick Batsell, Thomas Walker, and Richard Baraniuk. A Geometric Perspective on Recursive Synthetic Training. In ICLR Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy, 2026.

Mingyang Deng, He Li, Tianhong Li, Yilun Du, and Kaiming He. Generative Modeling Via Drifting. arXiv:2602.04770, 2026.

Yunzhen Feng, Elvis Dohmatob, Pu Yang, Francois Charton, and Julia Kempe. Beyond Model Collapse: Scaling up With Synthesized Data Requires Verification. In International Conference on Learning Representations, 2025.

Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One Step Diffusion Via Shortcut Models. In International Conference on Learning Representations, 2025.

Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean Flows For One-step Generative Modeling. In Advances in Neural Information Processing Systems, 2025.

Ruifei He, Shuyang Sun, Xin Yu, Chuhui Xue, Wenqing Zhang, Philip Torr, Song Bai, and Xiaojuan Qi. Is Synthetic Data From Generative Models Ready For Image Recognition? In International Conference on Learning Representations, 2023.

Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson, Heewoo Jun, Tom B. Brown, Prafulla Dhariwal, Scott Gray, Chris Hallacy, Benjamin Mann, Alec Radford, Aditya Ramesh, Nick Ryder, Daniel M. Ziegler, John Schulman, Dario Amodei, and Sam McCandlish. Scaling Laws for Autoregressive Generative Modeling. arXiv:2010.14701, 2020.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs Trained by a Two Time-scale Update Rule Converge to a Local Nash Equilibrium. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems, 2017.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling Laws for Neural Language Models. arXiv:2001.08361, 2020.

Tero Karras, Miika Aittala, Tuomas Kynkäänniemi, Jaakko Lehtinen, Timo Aila, and Samuli Laine. Guiding a Diffusion Model With a Bad Version of Itself. In Advances in Neural Information Processing Systems, 2024.

Dongjun Kim, Yeongmin Kim, Se Jung Kwon, Wanmo Kang, and Il-Chul Moon. Refining Generative Process With Discriminator Guidance in Score-based Diffusion Models. arXiv:2211.17091, 2023.

Alex Krizhevsky and Geoffrey Hinton. Learning Multiple Layers of Features from Tiny Images. Technical report, University of Toronto, 2009.

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. ImageNet Classification with Deep Convolutional Neural Networks. In Advances in Neural Information Processing Systems, 2012.

Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved Precision and Recall Metric for Assessing Generative Models. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. SiT: Exploring Flow and Diffusion-based Generative Models With Scalable Interpolant Transformers. In European Conference on Computer Vision. Springer, 2024.

Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, and Colin Raffel. Scaling Data-Constrained Language Models. arXiv:2305.16264, 2025.

Muhammad Ferjad Naeem, Seong Joon Oh, Youngjung Uh, Yunjey Choi, and Jaejun Yoo. Reliable fidelity and diversity metrics for generative models. In International Conference on Machine Learning, 2020.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning Transferable Visual Models From Natural Language Supervision. In International Conference on Machine Learning, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution Image Synthesis With Latent Diffusion Models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. In International Conference on Medical Image Computing and Computerassisted Intervention. Springer, 2015.

Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson, and Yarin Gal. AI Models Collapse When Trained on Recursively Generated Data. Nature, 631(8022), 2024.

David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, and Demis Hassabis. A General Reinforcement Learning Algorithm That Masters Chess, Shogi, and Go Through Self-play. Science, 362(6419), 2018.

Yang Song and Stefano Ermon. Generative Modeling by Estimating Gradients of the Data Distribution. 2019.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency Models. arXiv:2303.01469, 2023.

Laurens van der Maaten and Geoffrey Hinton. Visualizing Data Using t-SNE. Journal of Machine Learning Research, 9(86), 2008.

Pablo Villalobos, Anson Ho, Jaime Sevilla, Tamay Besiroglu, Lennart Heim, and Marius Hobbhahn. Will We Run Out of Data? Limits of LLM Scaling Based on Human-generated Data, 2024.

Zekai Wang, Tianyu Pang, Chao Du, Min Lin, Weiwei Liu, and Shuicheng Yan. Better Diffusion Models Further Improve Adversarial Training. In International Conference on Machine Learning. PMLR, 2023.

Jiawei Yang, Zhengyang Geng, Xuan Ju, Yonglong Tian, and Yue Wang. Representation Fréchet Loss for Visual Generation. arXiv:2604.28190, 2026.

Huizhuo Yuan, Zixiang Chen, Kaixuan Ji, and Quanquan Gu. Self-play Fine-tuning of Diffusion Models For Text-to-image Generation. In Advances in Neural Information Processing Systems. Curran Associates Inc., 2024.

Kaiyu Yue, Menglin Jia, Ji Hou, and Tom Goldstein. Image Generation With a Sphere Encoder. arXiv:2602.15030, 2026.

Huijie Zhang, Aliaksandr Siarohin, Willi Menapace, Michael Vasilkovsky, Sergey Tulyakov, Qing Qu, and Ivan Skorokhodov. AlphaFlow: Understanding and Improving MeanFlow Models. In International Conference on Learning Representations, 2026.

Kaiwen Zheng, Yongxin Chen, Huayu Chen, Guande He, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Direct Discriminative Optimization: Your Likelihood-based Visual Generative Model Is Secretly a GAN Discriminator. In International Conference on Machine Learning, 2025.

Linqi Zhou, Stefano Ermon, and Jiaming Song. Inductive Moment Matching. In International Conference on Machine Learning, 2025.

## A PROOF OF THEOREM 1

The standard output of the generator can be decomposed as $\begin{array} { r } { s = J _ { z } z + b _ { z } . } \end{array}$ . Using a singular value decomposition, the Jacobian-vector product $J _ { z } z \cos { \mathrm { \alpha } } $ be expanded as a sum of its singular components. Let r be the rank of $J _ { z }$ , then

$$
J _ { z } z = \sum _ { k = 1 } ^ { r } \sigma _ { z } ^ { ( k ) } \left( \pmb { v } _ { z } ^ { ( k ) \top } z \right) \pmb { u } _ { z } ^ { ( k ) }
$$

Then, by construction, we have

$$
\begin{array} { l } { { \displaystyle \tilde { J } _ { z } z = \tilde { \sigma } _ { z } ^ { ( 1 ) } \left( v _ { z } ^ { ( 1 ) \top } z \right) \pmb { u } _ { z } ^ { ( 1 ) } + \sum _ { k = 2 } ^ { r } \tilde { \sigma } _ { z } ^ { ( k ) } \left( v _ { z } ^ { ( k ) \top } z \right) \pmb { u } _ { z } ^ { ( k ) } } } \\ { { \displaystyle ~ = \tilde { \sigma } _ { z } ^ { ( 1 ) } \left( v _ { z } ^ { ( 1 ) \top } z \right) \pmb { u } _ { z } ^ { ( 1 ) } + \sum _ { k = 2 } ^ { r } \sqrt { 1 - \alpha } \sigma _ { z } ^ { ( k ) } \left( v _ { z } ^ { ( k ) \top } z \right) \pmb { u } _ { z } ^ { ( k ) } } } \end{array}
$$

Then, since

$$
\sum _ { k = 2 } ^ { r } \sigma _ { z } ^ { ( k ) } \left( v _ { z } ^ { ( k ) \top } z \right) \pmb { u } _ { z } ^ { ( k ) } = J _ { z } z - \sigma _ { z } ^ { ( 1 ) } \left( v _ { z } ^ { ( 1 ) \top } z \right) \pmb { u } _ { z } ^ { ( 1 ) } ,
$$

we can write

$$
\sum _ { k = 2 } ^ { r } \sqrt { 1 - \alpha } \sigma _ { z } ^ { ( k ) } \left( v _ { z } ^ { ( k ) \top } z \right) u _ { z } ^ { ( k ) } = \sqrt { 1 - \alpha } \left( s - b _ { z } \right) - \sqrt { 1 - \alpha } \sigma _ { z } ^ { ( 1 ) } \left( v _ { z } ^ { ( 1 ) \top } z \right) u _ { z } ^ { ( 1 ) } .
$$

Meaning,

$$
\begin{array} { r l } & { \tilde { J } _ { z } z = \tilde { \sigma } _ { z } ^ { ( 1 ) } \left( v _ { z } ^ { ( 1 ) \top } z \right) u _ { z } ^ { ( 1 ) } + \sqrt { 1 - \alpha } \left( s - b _ { z } \right) - \sqrt { 1 - \alpha } \sigma _ { z } ^ { ( 1 ) } \left( v _ { z } ^ { ( 1 ) \top } z \right) u _ { z } ^ { ( 1 ) } } \\ & { \qquad = \sqrt { 1 - \alpha } \left( s - b _ { z } \right) + \left( \tilde { \sigma } _ { z } ^ { ( 1 ) } - \sqrt { 1 - \alpha } \sigma _ { z } ^ { ( 1 ) } \right) \left( v _ { z } ^ { ( 1 ) \top } z \right) u _ { z } ^ { ( 1 ) } } \end{array}
$$

Therefore,

$$
\tilde { s } = \tilde { J } _ { z } z + b _ { z } = \sqrt { 1 - \alpha } \left( s - b _ { z } \right) + \left( \tilde { \sigma } _ { z } ^ { ( 1 ) } - \sqrt { 1 - \alpha } \sigma _ { z } ^ { ( 1 ) } \right) \left( v _ { z } ^ { ( 1 ) \top } z \right) u _ { z } ^ { ( 1 ) } + b _ { z } .
$$

This is precisely the output generated by Algorithm 2, thus the proof is complete.

## A.1 SPECTRAL ENERGY REDISTRIBUTION

The reweighting in Section 3.1 admits a direct interpretation as a redistribution of spectral energy. For a fixed latent vector z, write $\sigma _ { k } = \sigma _ { z } ^ { ( k ) }$ and $\tilde { \sigma } _ { k } = \tilde { \sigma } _ { z } ^ { ( k ) }$ , and let $\begin{array} { r } { E = \| J _ { z } \| _ { F } ^ { 2 } = \sum _ { k = 1 } ^ { r } \sigma _ { k } ^ { 2 } } \end{array}$ . For $\alpha \in [ 0 , 1 ]$ , the prescribed transformation satisfies

$$
\tilde { \sigma } _ { 1 } ^ { 2 } = \sigma _ { 1 } ^ { 2 } + \alpha \sum _ { k = 2 } ^ { r } \sigma _ { k } ^ { 2 } , \quad \quad \tilde { \sigma } _ { k } ^ { 2 } = ( 1 - \alpha ) \sigma _ { k } ^ { 2 } \quad ( k \ge 2 ) .
$$

Thus, the leading component receives exactly the spectral energy removed from the trailing components. Consequently,

$$
\| \tilde { J } _ { z } \| _ { F } ^ { 2 } = \tilde { \sigma } _ { 1 } ^ { 2 } + \sum _ { k = 2 } ^ { r } \tilde { \sigma } _ { k } ^ { 2 } = \| J _ { z } \| _ { F } ^ { 2 } .
$$

When $E > 0$ , define the normalized spectral energies $p _ { k } = \sigma _ { k } ^ { 2 } / E$ and $\tilde { p } _ { k } = \tilde { \sigma } _ { k } ^ { 2 } / E$ . The transformation can then be written as

$$
\tilde { \mathbf { p } } = ( 1 - \alpha ) \mathbf { p } + \alpha \mathbf { e } _ { 1 } ,
$$

where $\mathbf { e } _ { 1 } = ( 1 , 0 , \ldots , 0 ) ^ { \top }$ . Hence, α controls a linear interpolation in normalized squared singular values between the original spectrum and one with all spectral energy in the leading component.

The singular directions and affine offset $b _ { z }$ are retained. At $\alpha = 0$ , the original output is recovered. $\mathrm { A t } \alpha = 1$ , the modified linear term has rank at most one, while the offset remains unchanged. The

preserved quantity is the squared Frobenius norm of the constructed matrix, not necessarily the norm of the generated output.

This provides a design rationale for the GMO reweighting: it introduces a controlled concentration of the local spectrum, motivated by the spectral concentration observed during self-training, while preserving total spectral energy.

## B GMOS ARE MODE SEEKING

In Alemohammad et al. (2026), Neon is shown to theoretically work when finetuning against model outputs that are “mode-seeking”. Informally, this means that the model outputs are concentrated in high-density regions of the learned data distribution. With Figure 7, we show that GMOs amplify this effect. In the top row of Figure 7, we visualize the CLIP embeddings (Radford et al., 2021) of GMOs across a range of α values using t-SNE. It can be seen that the embeddings become increasingly concentrated for larger values of α, indicating that the images are focusing on the same regions of the data distribution.

With the bottom row of Figure 7, we visualize GMOs for each corresponding value of α. Qualitatively, we observe that the perturbations concentrate on distortions in the output. We provide more examples in Figure 14. In particular, the bottom row of Figure 14 shows that when the standard output has no clear distortions, the corresponding GMO looks relatively unchanged. This supports the idea that GMOs amplify the negative signal in model outputs.

![](images/d043fc09903f57edafb92d3ebb4ed383ca58df2ceff66a567b7984324cc73c7e.jpg)  
α = 0.0

![](images/9f9a65c3971b605626bfcea4d256b4e4203e314704904e9ac630106612fcc630.jpg)  
α = 0.1

![](images/a8c3c4c284c8711ef420d8a3afe43522f645e7453fc83d0715631dbd08f4f68e.jpg)

![](images/4d92909e372eb598a8645338510a7d50fc5f914de3a47d3670bde94c2db9dfbb.jpg)  
α = 0.2

![](images/414eeb4e7b914f9738486d2253e5292c171f4ef3e023812cb34f9914cfed8da6.jpg)

![](images/33d412cac551d5d4da701173056bab8780688fcbad825da3b1504b8a42802d1f.jpg)  
α = 0.6

![](images/aa60045df1653ce29cad76b5c62ba94228a4b113f55f6c14c6e6ad12dc08f079.jpg)

![](images/9879a8c81a707eeb3a8df9170d936647d8475bd25bf61d9f2d1ef9bd9086f4f6.jpg)  
α = 1.0  
Figure 7: GMOs amplify the “mode-seeking” nature of model outputs and their errors. Here we consider a collection of 2048 GMOs for α values 0.0, 0.1, 0.2, 0.6, and 1.0 generated using AlphaFlow (Zhang et al., 2026) SiT-XL/2 (Ma et al., 2024) model trained on ImageNet256 (Krizhevsky et al., 2012). In the top row, we visualize a t-SNE projection (van der Maaten & Hinton, 2008) of the CLIP embeddings (Radford et al., 2021) of the GMOs. In the bottom row, we visualize the GMOs for a specific latent vector.

## C THE EFFECT OF GMOS ON THE COMPUTE AND WEIGHT-MERGING OF NEON

The performance of the Neon self-training algorithm is dependent on the amount of compute used for finetuning B and the weight-merging parameter w. In Figure 8, we explore how GMOs affect these hyperparameters for a UNet generative model (Ronneberger et al., 2015) training on CIFAR10 (Krizhevsky & Hinton, 2009) using the MeanFlow framework (Geng et al., 2025). We observe that as the value of α increases, the optimal values for B and w decrease. This supports the claim that GMOs improve Neon’s effectiveness: the more directed degradation induced by GMOs means the optimal checkpoint is reached with less finetuning.

To quantify the complete cost of the pipeline, we follow Neon’s analysis (Alemohammad et al., 2026) and report the total cost as a percentage of the model’s pretraining compute. Unlike the finetuning budgets in Table 1, the totals in Table 2 also include the upfront cost of GMO generation. Across all models, the complete pipeline costs between 0.01% and 1.1% of pretraining compute.

Table 2: Total pipeline cost as a percentage of pretraining compute, including the upfront cost of GMO generation.
<table><tr><td>Model</td><td>Standard Neon (% of pretraining)</td><td>Neon+GMOs (% of pretraining)</td></tr><tr><td>IMM</td><td>0.007%</td><td>0.01%</td></tr><tr><td>AlphaFlow-XL/2</td><td>0.10%</td><td>0.51%</td></tr><tr><td>AlphaFlow-B/2</td><td>0.16%</td><td>0.57%</td></tr><tr><td>MeanFlow-B/2</td><td>0.26%</td><td>0.61%</td></tr><tr><td>MeanFlow-L/2</td><td>0.63%</td><td>1.06%</td></tr></table>

![](images/cbc97766b842e21d3927d10274472df55107366061976b352c27d06115122abb.jpg)

![](images/658ddc35bd760f8fec494cc9cb180f7061062cfc08b2ed133c88a96de70fdd0c.jpg)

![](images/db1241492b7959368a91746f84869810b8d869ae74e0b1037a159b3be40acd01.jpg)  
Figure 8: GMOs can reduce the finetuning required to achieve optimal results using Neon. Here we apply Neon to a UNet generative model (Ronneberger et al., 2015) training on CI-FAR10 (Krizhevsky & Hinton, 2009) using the MeanFlow framework (Geng et al., 2025). We monitor the value of FID for a range of compute levels B and weight-merging parameters w. With red markers, we indicate which combination of these hyperparameters yields the model with the lowest FID score.

## D EXPERIMENTAL DETAILS

## D.1 FIGURE 6

To obtain the result of the right panel of Figure 6, we generate 10, 000 GMOs from 1-step and 2-step IMM (Zhou et al., 2025) models trained on CIFAR10 (Krizhevsky & Hinton, 2009) at an α value of 0.1. This process takes approximately 6 hours using 8 NVIDIA A100-SXM4-80GB GPUs. The 2-step base model is then finetuned using a computational budget of $2 . 5 6 \times 1 0 ^ { 5 }$ . For reference, a computational budget 16, 000 times larger was used to pre-train the base checkpoint (Zhou et al., 2025).

Throughout the finetuning we take 8 checkpoints and evaluate each using weight-merging parameters in the range [0, 1] along a 0.1-step interval. In the right panel of Figure 6, for each finetuning dataset, we visualize the evaluation curves that yielded the lowest FID scores. When using standard outputs from the 2-step model, the optimal FID was seen at a computational budget of 1 $. 6 \times 1 0 ^ { 5 }$ . When using GMOs from the 2-step model, the optimal FID was seen at a computational budget of $9 . 6 \times 1 0 ^ { \bar { 4 } }$ When using GMOs from the 1-step model, the optimal FID was seen at a computational budget of $1 . 2 8 \times 1 0 ^ { 5 }$ . For completeness, in Figure 9, we provide the evaluation curves for finetuning checkpoint.

## D.2 TABLE 1

Here we detail the hyperparameter sweeps that were used to obtain the results of Table 1. The following computation times are based on 8 NVIDIA RTX A6000 GPUs.

![](images/06713f922d08bfd563793117669be622c14e5d821735110e40195f988fefef9c.jpg)  
Figure 9: The complete set of evaluation curves for the experiment shown in the right panel of Figure 6, and described in Appendix D.1.

• MeanFlow SiT-B/2: We generate 30, 000 standard outputs and GMOs for α values 0.05 and 0.1. This takes approximately 4 hours. The base model is finetuned for 48 epochs on each of these data sets, and we evaluate 6 equally spaced checkpoints. Each round of finetuning takes 1 hour. For each checkpoint, we evaluate the FID using 50, 000 samples and w values ranging from 0.0 to 1.5 at 0.1-step increments. Each FID computation takes approximately 2 minutes.

• MeanFlow SiT-L/2: We generate 30, 000 standard outputs and GMOs for α values 0.1 and 0.2. This takes approximately 7 hours. The base model is finetuned for 60 epochs on each of these data sets, and we evaluate 6 equally spaced checkpoints. Each round of finetuning takes 3 hours. For each checkpoint, we evaluate the FID using 50, 000 samples and w values ranging from 0.0 to 1.5 at 0.1-step increments. Each FID computation takes approximately 3 minutes.

• AlphaFlow SiT-B/2: We generate 30, 000 standard outputs and GMOs for α values 0.1 and 0.2. This takes approximately 4 hours. The base model is finetuned for 32 epochs on each of these data sets, and we take checkpoints every 4 epochs. Each round of finetuning takes less than 1 hour. For each checkpoint, we evaluate the FID using 50, 000 samples and w values ranging from 0.0 to 1.5 at 0.1-step increments. Each FID computation takes approximately 2 minutes.

• AlphaFlow SiT-XL/2: We generate 30, 000 standard outputs and GMOs for α values 0.05 and 0.1. This takes approximately 10 hours. The base model is finetuned for 15 epochs on each of these data sets, and we take checkpoints every 5 epochs. Each round of finetuning takes less than 1 hour. For each checkpoint, we evaluate the FID using 50, 000 samples and w values ranging from 0.0 to 1.6 at 0.1-step increments. Each FID computation takes approximately 4 minutes.

• IMM (DiT-XL/2): We generate 30,000 standard outputs and GMOs for α values of 0.1 and 0.2. This takes approximately 5 hours. The base model is finetuned for 109 epochs on each dataset, and we evaluate 8 equally spaced checkpoints. Each round of finetuning takes approximately 5 hours. For each checkpoint, we evaluate FID using 50,000 samples and w values ranging from 0.0 to 1.8 in increments of 0.2. Each FID computation takes approximately 10 minutes.

In Figure 10, we consider each individual model as a row, and show in the left panel what the minimum FID value is for the sweep over w at each compute level. In the right panel, we show the value of w for which the minimum FID was obtained at each compute level.

## D.3 SIMS-STYLE GUIDANCE

Here we detail the experiments in Section 4.5. The following computation times are based on 8 NVIDIA RTX A6000 GPUs.

• IMM (DiT-XL/2): We use synthetic datasets containing 50,000 standard outputs or GMOs with α values 0.1 and 0.2. For the main sweep, we evaluate 5 checkpoints for standard outputs and 8 checkpoints for GMOs with $\alpha = 0 . 1$ , using guidance strengths $\omega \in \{ 0 . 5 , 1 . \bar { 0 } , 1 . 6 \}$ . The auxiliary models used for the reported comparison are finetuned for approximately 31 epochs on their respective datasets, requiring approximately 2.5 hours per model. Each FID evaluation uses 50,000 samples, one-step sampling, and classifier-free guidance scale 1.5, and takes approximately 6 minutes. We additionally evaluate $\alpha = 0 . 2$ using $\omega \in \{ 0 . 5 , 1 . 0 , 1 . 6 , 2 . 4 \}$ ; its lowest observed FID is with GMOs at 7.535

Table 3: SIMS-style guidance on IMM ImageNet256. FID is reported as mean ± sample standard deviation across five evaluations of fixed checkpoints, using 50,000 samples per evaluation. Both guided variants use $\omega = 1 . 6$
<table><tr><td>Method</td><td>FID↓</td></tr><tr><td>Base IMM</td><td> $8 . 3 4 2 \pm 0 . 0 5 8$ </td></tr><tr><td>SIMS, standard outputs</td><td> $7 . 5 9 1 \pm 0 . 0 5 7$ </td></tr><tr><td>SIMS, GMOs  $( \alpha = 0 . 1 )$ </td><td> ${ \bf 6 . 9 2 0 \pm 0 . 0 4 8 }$ </td></tr></table>

Table 4 reports precision, recall, density, and coverage for all architectures of Table 1 (five seeds, $\mathrm { m e a n } \pm \mathrm { s t d } )$ . Density and coverage, the outlier-robust fidelity and diversity measures (Naeem et al., 2020), are maintained or improved from standard-outputs Neon to GMOs on every architecture, showing that GMOs improve image quality and diversity simultaneously rather than trading one for the other (see Section 4.2). Table 5 reports the corresponding significance analysis for the FID results of Table 1.

Table 4: Precision, recall, density, and coverage (5 seeds, mean ± std). Density and coverage are maintained or improved from standard-outputs Neon to GMOs on every architecture.
<table><tr><td>Architecture</td><td>Method</td><td>Precision</td><td>Recall</td><td>Density</td><td>Coverage</td></tr><tr><td rowspan="3">IMM (DiT-XL/2)</td><td>Base</td><td> $. 5 8 5 \pm . 0 0 2$ </td><td> $. 6 4 7 \pm . 0 0 1$ </td><td> $. 6 5 2 \pm . 0 0 1$ </td><td> $. 6 6 2 \pm . 0 0 3$ </td></tr><tr><td>Neon-std</td><td> $. 5 9 8 \pm . 0 0 2$ </td><td> $. 6 5 4 \pm . 0 0 2$ </td><td> $. 6 9 5 \pm . 0 0 1$ </td><td> $. 6 9 3 \pm . 0 0 2$ </td></tr><tr><td>GMOs</td><td> $. 6 1 5 \pm . 0 0 3$ </td><td> $. 6 4 7 \pm . 0 0 2$ </td><td> $\mathbf { 7 2 8 \pm . 0 0 3 }$ </td><td> $\mathbf { 7 1 4 \pm . 0 0 2 }$ </td></tr><tr><td rowspan="3">MeanFlow SiT-B/2</td><td>Base</td><td> $. 7 1 7 \pm . 0 0 2$ </td><td> $. 4 5 2 \pm . 0 0 2$ </td><td> $1 . 0 8 1 \pm . 0 0 6$ </td><td> $. 7 4 9 \pm . 0 0 1$ </td></tr><tr><td>Neon-std</td><td> $. 7 2 0 \pm . 0 0 1$ </td><td> $. 4 5 2 \pm . 0 0 3$ </td><td> $1 . 1 1 0 \pm . 0 0 6$ </td><td> $. 7 5 5 \pm . 0 0 1$ </td></tr><tr><td>GMOs</td><td> $. 7 2 5 \pm . 0 0 1$ </td><td> $. 4 5 1 \pm . 0 0 3$ </td><td> $\mathbf { 1 . 1 2 4 } \pm . 0 0 7$ </td><td> $\mathbf { 7 5 9 \pm . 0 0 2 }$ </td></tr><tr><td rowspan="3">MeanFlow SiT-L/2</td><td>Base</td><td> $. 7 5 2 \pm . 0 0 1$ </td><td> $. 4 9 4 \pm . 0 0 2$ </td><td> $1 . 2 1 4 \pm . 0 0 2$ </td><td> $. 8 2 8 \pm . 0 0 1$ </td></tr><tr><td>Neon-std</td><td> $. 7 6 4 \pm . 0 0 0$ </td><td> $. 4 8 6 \pm . 0 0 2$ </td><td> $1 . 2 6 4 \pm . 0 0 2$ </td><td> $. 8 3 7 \pm . 0 0 1$ </td></tr><tr><td>GMOs</td><td> $. 7 6 3 \pm . 0 0 1$ </td><td> $. 4 8 3 \pm . 0 0 2$ </td><td> $\mathbf { 1 . 2 6 7 \pm . 0 0 2 }$ </td><td> $\mathbf { 8 3 9 } \pm . 0 \mathbf { 0 1 }$ </td></tr><tr><td rowspan="3">AlphaFlow SiT-B/2</td><td>Base</td><td> $. 7 4 8 \pm . 0 0 1$ </td><td> $. 4 5 0 \pm . 0 0 1$ </td><td> $1 . 2 1 0 \pm . 0 0 4$ </td><td> $. 7 9 4 \pm . 0 0 1$ </td></tr><tr><td>Neon-std</td><td> $. 7 4 3 \pm . 0 0 2$ </td><td> $. 4 5 4 \pm . 0 0 3$ </td><td> $1 . 1 9 7 \pm . 0 0 3$ </td><td> $. 7 9 1 \pm . 0 0 2$ </td></tr><tr><td>GMOs</td><td> $. 7 4 0 \pm . 0 0 1$ </td><td> $. 4 5 5 \pm . 0 0 3$ </td><td> $1 . 1 9 7 \pm . 0 0 3$ </td><td> $. 7 9 1 \pm . 0 0 2$ </td></tr><tr><td rowspan="3">AlphaFlow SiT-XL/2</td><td>Base</td><td> $. 6 9 5 \pm . 0 0 1$ </td><td> $. 5 9 6 \pm . 0 0 1$ </td><td> $1 . 0 0 5 \pm . 0 0 2$ </td><td> $. 8 1 1 \pm . 0 0 1$ </td></tr><tr><td>Neon-std</td><td> $. 7 0 1 \pm . 0 0 1$ </td><td> $. 5 9 3 \pm . 0 0 3$ </td><td> $1 . 0 2 5 \pm . 0 0 3$ </td><td> $. 8 2 0 \pm . 0 0 2$ </td></tr><tr><td>GMOs</td><td> $. 7 0 5 \pm . 0 0 0$ </td><td> $. 5 8 9 \pm . 0 0 2$ </td><td> $\mathbf { 1 . 0 3 4 \pm . 0 0 3 }$ </td><td> ${ \bf 8 2 1 } \pm { \bf . 0 0 1 }$ </td></tr></table>

Table 5: Statistical significance of the GMO improvement over standard-outputs Neon (5 seeds, mean ± std). The final column reports the additional FID reduction of GMOs as a percentage of the reduction Neon itself achieves over the base model.
<table><tr><td>Architecture</td><td>Base FID</td><td>Neon-std FID</td><td>GMOs FID</td><td>Welch p</td><td>GMOs&#x27; reduction, % of Neon&#x27;s</td></tr><tr><td>IMM (DiT-XL/2)</td><td>8.34</td><td> $7 . 3 2 \pm . 0 7$ </td><td> ${ \bf 6 . 2 5 \pm . 0 3 }$ </td><td> $\sim 1 0 ^ { - 9 }$ </td><td>96%</td></tr><tr><td>MeanFlow SiT-B/2</td><td>6.08</td><td> $5 . 7 0 \pm . 0 1$ </td><td> ${ \bf 5 . 6 0 \pm . 0 1 }$ </td><td></td><td>26%</td></tr><tr><td>MeanFlow SiT-L/2</td><td>3.97</td><td> $3 . 7 4 \pm . 0 2$ </td><td> ${ \bf 3 . 7 0 \pm . 0 2 }$ </td><td> $^ { 2 . 6 \times 1 0 ^ { - 7 } } _ { 0 . 0 1 3 }$ </td><td>17%</td></tr><tr><td>AlphaFlow SiT-B/2</td><td>5.55</td><td> $5 . 3 6 \pm . 0 2$ </td><td> ${ \bf 5 . 1 6 \pm . 0 2 }$ </td><td> $2 . 6 \times 1 0 ^ { - 7 }$ </td><td>105%</td></tr><tr><td>AlphaFlow SiT-XL/2</td><td>2.93</td><td> $2 . 6 4 \pm . 0 2$ </td><td> ${ \bf 2 . 5 9 \pm . 0 2 }$ </td><td>0.004</td><td>17%</td></tr></table>

## E MODEL LICENSES

IMM model weights are obtained through the imm GitHub repository<sup>3</sup> under CC BY-NC-SA 4.0 License. MeanFlow model weights are obtained through the MeanFlow GitHub repository<sup>4</sup> under the MIT license. AlphaFlow model weights are obtained through the AlphaFlow GitHub repository<sup>5</sup> under the Snap Inc. Non-Commercial license.

![](images/613ac7852d40edaee35431737f5360c32456397f03964354031e943994eff9a6.jpg)

![](images/d09b2d5aa730d410c792d6f6d5a82fb9508198f82cd1d0d3de09fff561db04cb.jpg)  
$\cdot 1 0 ^ { 6 }$ MeanFlow B-2

![](images/49742987d9a1f1bb2207d5dfa1ced1d3bce303e577bb03c4e88317df8b0c5adb.jpg)

$\cdot 1 0 ^ { 6 }$ MeanFlow L-2  
![](images/6c907e6a6eac53bb0e4029a6c42bcd84248188d053e1ee9bbe7f1bd292d622af.jpg)

![](images/b8d14d420fd50619020fa7575c6c2122322c50c875607d37579dbf06151d0029.jpg)

![](images/3226486637e31f87337f1cd6387156023b2f74069e76ee8b56f15b40762b975b.jpg)  
$\cdot 1 0 ^ { 6 }$ AlphaFlow B-2

![](images/5c3cea6926a865b50864af5bca178d227c37493e9c5f6ed3f70c433af31eb875.jpg)

![](images/253f409baf1608b4183411c6b5fd896b6ba706c76e0e25f46a4ec0042fe907a7.jpg)  
$\cdot 1 0 ^ { 5 }$ AlphaFlow XL-2

![](images/829f2cf4702fe37bc425acdcbf8366eab325c633c1ea4847ad8bb42c4c0b56e9.jpg)

![](images/f761e1b908272e54f9dad3faf7f90ebb12a15ea00c1101c25f23bba5599bbdf6.jpg)  
$\cdot 1 0 ^ { 6 }$ IMM one-step  
Figure 10: Hyperparameter sweeps described in Appendix D.2 that yielded the results in Table 1.

Standard Outputs  
Perturbation  
Geometrically Modified Outputs  
![](images/de4316af56e6d431af0b4c72b5f212f0c8e380e4a36ff935c79c8bc281108566.jpg)  
α = 0.5  
Figure 11: Additional examples of those shown in Figure 1 at different GMO α levels.

Standard Outputs  
GMO Outputs  
![](images/8c347941fa7b0b185c7813e612639288ac4299d8fc86b0a8b72b1528fd9873f8.jpg)  
Lion  
Figure 12: Qualitative comparison on IMM ImageNet-256 (Part I).

Standard Outputs  
GMO Outputs  
![](images/68e089d06033c9279071004292faecf3fb1ea740c9d5c25fa7920efbd40dc72f.jpg)  
Tabby Cat  
Figure 12: Qualitative comparison on IMM ImageNet-256 (Part II).

![](images/94205c1827932bffa523f91f66d69c58e9d1f9a1edb818908d84db7da1dd51e7.jpg)

![](images/7c824cf756bdf06f360e5c94cbd59f62e3d434e0de418e52e4757cda3d93eb6d.jpg)

Exact  
![](images/9d11c7818148e69f71d875ba5e896c5d793506afd89f26d91e838bbd0a861e58.jpg)

![](images/a2c7faa973c0532afddf88aded21fe88ac9850f127c3c49d5723495be00a4c94.jpg)

![](images/6da489d45452aa9f9ef434eba5296460fa27defe6e05771a11e572808cda50ca.jpg)

Approximate  
![](images/0c9535caf8f46b40b76a05f3fc776487583d80f30ac075fba17747e99c36d026.jpg)

![](images/29073fb0d820725a9e0f34bc42d1ed8bd01b8638f08d144c07acc504722ce42d.jpg)

![](images/6afdb583977639b9956c8ae212017086972e07cffd548e21ee91ba86acc7069a.jpg)

![](images/0d105e580700a482d48d76cbd334a6bf41a489b315284e4f017350c1b31b3423.jpg)

![](images/9e5e7bca7b2813c108ec01d34f9f93124213970ff53b712188617980ac8dbeda.jpg)  
Figure 13: In Figure 3, we quantitatively demonstrate that power iteration can be used to accurately approximate the singular vectors of the input-output Jacobians of generative models. Here, we qualitatively support this by showing the exact (top) and approximate top left singular vectors of these input-output Jacobians.

![](images/934eb815b7386d910ade423515c1b8fe87a12adbda016eb00ae92f8b9aabe1c3.jpg)

![](images/66f1b1df35e0aa98338ad8fdf02fa63b258358e6bb7e08f839cd7776517482f8.jpg)  
α = 0.0

![](images/da0e6621d17e322d107e5c598b3059b47a949bc8fb959b513891e832ab7dd7aa.jpg)  
α = 0.1

![](images/4b6c180ed7cc3803ecee6cdc6cd3e917b15b3282c6a619af3da32c7669c8b7a4.jpg)  
α = 0.2

![](images/568a0f13684658741463eeee16b4c24b728d0e1f41786f40a51c1f446fd1b538.jpg)  
α = 0.6

![](images/4d8a47a153a258f7f8eb7e0d5b78c4fcbde31c43623cbbe60b95593a29041b13.jpg)  
α = 1.0

Figure 14: Here we provide additional examples to that illustrated in the bottom row of Figure 7.  
![](images/edca0ec07b704b1cc0266267f136e77eefb8c8fcb1def965c03a2b62636f9d3e.jpg)  
α = 0.0

![](images/7ed5e0e89aabc351c93d5e9833b3212faf5cc17b378bf363174660a4550da333.jpg)  
α = 0.05

![](images/09d36177a98feb64695524022fe5725ee17bc17eb10741fafd59e153a250f505.jpg)

![](images/c15574fd00594fb3a51050d5862dbaae3e36d7a799051d75202674eceb0ca24b.jpg)  
γ = 0.05

![](images/f8f12a70df151d6fa722f933c4451d3e1f9d8af12fd8914e39431d821073cc41.jpg)  
Figure 15: Here we provide additional examples to complement the third panel of Figure 4.