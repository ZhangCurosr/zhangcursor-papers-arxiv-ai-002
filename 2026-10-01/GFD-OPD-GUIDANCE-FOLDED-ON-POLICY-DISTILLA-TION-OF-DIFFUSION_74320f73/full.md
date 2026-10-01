# GFD-OPD: GUIDANCE-FOLDED ON-POLICY DISTILLA-TION OF DIFFUSION MODELS ACROSS SCALES

Zhenxing Zhang<sup>1,</sup> <sup>\*,</sup> <sup>†</sup>, Jiayan Teng<sup>2,</sup> <sup>\*,</sup> <sup>‡</sup>, Wenxu Wu<sup>3</sup>, Zhuoyi Yang<sup>2</sup>, Jiazheng Xu<sup>2</sup>, Wendi Zheng<sup>2</sup>, Jie Tang<sup>2</sup>, Dan Guo<sup>1</sup>, Meng Wang<sup>1,</sup> <sup>§</sup> <sup>1</sup>Hefei University of Technology <sup>2</sup>Tsinghua University <sup>3</sup>Zhipu AI

## ABSTRACT

On-policy distillation (OPD) has demonstrated two important capabilities in language models: compressing large teachers into smaller students and merging expert models into a single model. Existing diffusion OPD, however, mostly focus on the latter, with teachers and students sharing the same backbone and scale. We investigate large-to-small diffusion opd from large teachers to a small student and find that the standard recipe fails. To find the underlying cause, we propose Fixed-State KL, an effective and fair way to measure the distribution gap between student and teacher during OPD training for diffusion models. We are the first to clarify why large-to-small OPD is challenging for diffusion models: a smaller student struggles to perfectly match the distribution of a larger teacher, while classifier-free guidance can accumulate and amplify the distributional discrepancies between the student’s conditional and unconditional branches and those of the teacher. To solve this problem, we propose GFD-OPD, a simple yet effective method that reduces the student–teacher gap while avoiding the error amplification of the CFG composition. Across numerous experiments, GFD outperforms previous baselines in both training efficiency and final performance, achieving state-of-the-art results on all benchmarks. The code and checkpoints for this study are available at here.

![](images/a6af5c3a841946d019e8d0b785f2c15cac700d6c61814506f2c8d38b261d8b96.jpg)

(a)  
![](images/af6e10175bd2095a8b920ba7133bc1d7038b62aebcd37c6c6df830f139d162a6.jpg)  
GFD (ours) DiffusionOPD DiffusionOPD Student RL Large Teacher Medium Teacher Base (t=0)

![](images/e45ffcb77e28f6321553c196dd93c70527a709d0f23458f86f15335e4b509293.jpg)

![](images/3d3872cb30ace6546a069de2c7bbb1c5d0277ec4cb8ef32de041e9728e3c70d7.jpg)

Figure 1: GFD-OPD distills SD3.5-Large into SD3.5-Medium. (a) Score versus training compute (GPU hours) on OCR, PickScore, and GenEval: GFD-OPD overtakes all baselines within 10 GPU-hours. (b) GFD-OPD ranks first on all seven evaluation dimensions, covering both task rewards and image quality.

## 1 INTRODUCTION

On-policy distillation (OPD) has recently emerged as a powerful post-training recipe for large language models. Like reinforcement learning (RL), OPD trains the student on trajectories sampled from its own policy and therefore optimizes the student’s own distribution; unlike RL, which has to work with sparse and delayed rewards, OPD lets a teacher provide dense supervision at every step of the student-generated trajectory. This fine-grained feedback largely removes the credit-assignment difficulty of RL, so OPD enjoys a more stable optimization and a much better sample efficiency at a comparable final quality.

In language models OPD is primarily applied in two paradigms. The first is strong-to-weak distillation, where the capability of a large teacher is transferred to a small student that is cheap to deploy (Lu & Lab, 2025). The second is capability merging, where several specialist models derived from the same base model are merged into a single student without conflicts (Zeng et al., 2026). Diffusion models have begun to adopt OPD as a post-training method as well (Li et al., 2026b; Fang et al., 2026), but the existing diffusion work mostly focus on the second paradigm. The first paradigm, distilling a large diffusion model into a small one, remains unexplored. It is also important: a compact model that inherits capabilities of a larger teacher can achieve near-teacher quality at inference while retaining a low cost close to that of the smaller model.

Our goal is to investigate effective approaches for distilling the capabilities of large diffusion models into smaller ones. Although DiffusionOPD performs effectively when merging models with the same architecture, we find directly extending it to large-to-small distillation leads to severe degradation. As shown in Fig. 4, generated images exhibit artifacts, such as corrupted regions, loss of fine-grained textures and so on.

To understand the source of this degradation, we propose Fixed-State KL for measuring the distributional discrepancy between the student and teacher. Specifically, we compute KL divergence on states sampled from normal teacher trajectories and noise-corrupted trajectories constructed from real images, providing a fair and well-behaved evaluation distribution for all methods.

Under this evaluation metric, we find that large-to-small DiffusionOPD produces a substantially larger distributional discrepancy between the student and teacher than its same-architecture counterpart. Furthermore, we quantitatively characterize models’ final sampling result through spectral energy analysis of the final latent representations and high-pass texture analysis of the decoded images, which quantify the deviation from the teacher at both coarse-grained (e.g., global tone) and fine-grained (e.g., local texture) levels.

Through extensive analysis, we indicate that a smaller-capacity student consistently struggles to perfectly match the distribution of a larger teacher. Besides, classifier-free guidance (CFG), used during diffusion sampling, can accumulate and amplify the distributional discrepancies between the student’s conditional and unconditional branches and those of the teacher, causing a larger deviation in the final generation trajectory.

Motivated by these observations, we propose a simple yet effective method, Guidance Folding Distillation (GFD). GFD reduces the distributional discrepancy between the student and teacher while avoiding the error amplification introduced by CFG composition, enabling effective large-to-small OPD for diffusion models.

Our contributions are as follows.

• We propose Fixed-State KL, an effective and fair way to measure the distribution gap between student and teacher during OPD training for diffusion models.

• Through multifaceted analysis, we are the first to clarify why large-to-small OPD is challenging for diffusion models. To address these challenges, we propose GFD, a simple and effective method that reduces the student–teacher gap while avoiding the error amplification of the CFG composition.

• Through multiple experiments, combined with quantitative and qualitative analysis, we show the effectiveness of GFD. GFD removes the artifacts in cross-scale DiffusionOPD and outperforms previous baselines in both training efficiency and final performance.

## 2 METHOD

## 2.1 PRELIMINARY: ON-POLICY DISTILLATION IN LANGUAGE MODELS AND DIFFUSION MODELS

On-policy distillation (OPD) trains the student on trajectories sampled from its own policy, while the teacher provides dense supervision on every visited state. In language models, OPD minimizes the reverse KL between the student policy $\pi _ { \pmb { \theta } }$ and the frozen teacher $\pi ^ { \star }$ along student-generated sequences:

$$
\mathcal { L } _ { \mathrm { O P D } } ^ { \mathrm { L L M } } ( \pmb { \theta } ) = \mathbb { E } _ { \pmb { y } \sim \pi _ { \pmb { \theta } } } \Big [ \sum _ { t = 1 } ^ { | \pmb { y } | } \mathrm { K L } \big ( \pi _ { \pmb { \theta } } ( \cdot  { | } \ \pmb { y } _ { < t } )  { | | } \ \pi ^ { \star } ( \cdot  { | } \ \pmb { y } _ { < t } ) \big ) \Big ] .\tag{1}
$$

DiffusionOPD transfers the same idea to diffusion and flow-matching samplers. At each denoising step $j ,$ the student and teacher induce Gaussian transition kernels $p _ { S } ( x _ { t _ { j + 1 } } \ | \ x _ { t _ { j } } )$ and $p _ { T } ( x _ { t _ { j + 1 } } \ | \ x _ { t _ { j } } )$ whose covariance $\sigma _ { j } ^ { 2 } { \cal I }$ is fixed by the shared sampler, so the per-step reverse KL has the closed form $\parallel \mu _ { S } ( x _ { t _ { j } } ; \pmb \theta ) -$ $\mu _ { T } ( x _ { t _ { j } } ) \Vert _ { 2 } ^ { 2 } / ( 2 \sigma _ { j } ^ { 2 } )$ . Since the transition mean is affine in the velocity field with sampler-fixed coefficients, and sampling deploys the CFG-guided velocity, DiffusionOPD matches the CFG-guided output to the teacher’s:

$$
\mathcal { L } _ { \mathrm { O P D } } ^ { \mathrm { d i f f } } ( \pmb { \theta } ) = \mathbb { E } _ { \boldsymbol { x } _ { 0 : N } \sim p _ { S , \pmb { \theta } } } \left[ \sum _ { j = 0 } ^ { N - 1 } \omega _ { j } \big \| v _ { S } ^ { g } ( \boldsymbol { x } _ { t _ { j } } , t _ { j } , \boldsymbol { c } ) - v _ { T } ^ { g } ( \boldsymbol { x } _ { t _ { j } } , t _ { j } , \boldsymbol { c } ) \big \| _ { 2 } ^ { 2 } \right] ,\tag{2}
$$

where $v _ { \star } ^ { g } = v _ { \star } ^ { u } + w \left( v _ { \star } ^ { c } - v _ { \star } ^ { u } \right)$ , with $\star \in \{ S , T \}$ indexing the student and the teacher, is the guided velocity composed from the conditional and unconditional branches $v _ { \star } ^ { c } , v _ { \star } ^ { u }$ at guidance scale $w ,$ and $\breve { \omega } _ { j } = \eta _ { j } ^ { 2 } / ( 2 \sigma _ { j } ^ { 2 } )$ is fixed by the sampler, with $\eta _ { j }$ the coefficient of the velocity in the transition mean and $\sigma _ { j } ^ { 2 }$ the per-step variance; $\eta _ { j } = \big ( 1 + \sigma _ { t _ { i } } ^ { 2 } ( 1 - t _ { j } ) / ( 2 t _ { j } ) \big ) \Delta t _ { j }$ and $\sigma _ { j } ^ { 2 } = \sigma _ { t _ { j } } ^ { 2 } \Delta t _ { j }$ . In both SDE and ODE settings, DiffusionOPD is thus a pathwise, closed-form velocity-matching problem along the student’s rollout.

## 2.2 FIXED-STATE KL EVALUATION

We first directly apply the standard DiffusionOPD objective to the large-to-small OPD setting. However, after training, from Fig. 4 we can see that the resulting images exhibit obvious visual artifacts: they contain corrupted textures and are generally darker. To further quantify the degradation in generation quality, we conduct a spectral energy analysis on the final latent representations and a high-pass texture analysis on the decoded images. As shown in Fig. 6 $, \mathrm { D i f f u s i o n O P D _ { L  M } }$ deviates substantially from the teacher across all evaluated aspects.

To investigate the underlying cause, we need to measure the distributional discrepancy between the student and teacher. A natural choice is to use the objective in OPD training: the reverse KL divergence between the student and teacher distributions, computed along the trajectory rolled out by the student. However, although this formulation is well suited as a training objective, we argue that it is not an appropriate metric to evaluate the distributional discrepancy between the student and the teacher after training.

There are two main reasons. First, different training methods generally produce different student policies. Computing the KL divergence on each student’s own trajectories therefore evaluates the models on different sets of states, confounding the effect of distributional mismatch with differences in state visitation. It makes the resulting metrics difficult to compare in a fair and controlled manner.

Second, the student model may deviate from the normal generation trajectory during training and enter abnormal states. As pointed out by Xin et al. (2026), the teacher is no longer guaranteed to provide reliable predictions in such states, since the queried states may lie far outside the region on which the teacher’s behavior is meaningful. Consequently, the KL divergence computed on these abnormal states may no longer faithfully reflect the discrepancy between a well-behaved student and the teacher.

![](images/4d53c413de35cf517842629b1d7f9b4597071caca77130b05a49118f27108fee.jpg)

Table 1: KL divergence between each student and its corresponding teacher. $\mathrm { P _ { s t u } }$ and $\mathrm { P _ { t e a } }$ denote sampling trajectories generated by the student and teacher models, respectively, while $\mathrm { P _ { r e a l } }$ denotes trajectories constructed by adding noise to real images.
<table><tr><td rowspan="2">Student</td><td>KL  $\left( \mathrm { P _ { t e a } } \right)$ </td><td>KL  $\left( \mathrm { P _ { r e a l } } \right)$ </td><td>KL  $\left( \mathrm { P _ { s t u } } \right)$ </td><td colspan="2">branch KL  $\left( \mathrm { P _ { t e a } } \right)$ </td></tr><tr><td>guided</td><td>guided</td><td>guided</td><td>cond</td><td>uncond</td></tr><tr><td>DiffusionOPDM→M</td><td>0.041</td><td>0.022</td><td>0.039</td><td>0.017</td><td>0.023</td></tr><tr><td>DiffusionOPDL→M</td><td>0.068</td><td>0.046</td><td>0.057</td><td>0.037</td><td>0.039</td></tr><tr><td>PDML→M</td><td>0.063</td><td>0.046</td><td>0.052</td><td>0.023</td><td>0.022</td></tr><tr><td>split-KLL→M</td><td>0.074</td><td>0.049</td><td>0.065</td><td>0.021</td><td>0.020</td></tr><tr><td>GFDL→M  $( w = 1 )$ </td><td>0.050</td><td>0.030</td><td>0.043</td><td></td><td></td></tr><tr><td> $\mathrm { G F D } _ { \mathrm { L  M } } ( w = 2 )$ </td><td>0.053</td><td>0.032</td><td>0.053</td><td></td><td></td></tr></table>

Figure 2: Illustration of the KL evaluation.

Motivated by these considerations, we propose Fixed-State KL, which is performed on a fixed and normal set of states shared by all student models. Specifically, we use states from normal teacher trajectories, together with noise-corrupted trajectories constructed from real images, as common evaluation inputs for computing the KL divergence between the student and teacher. By decoupling the evaluation states from the student’s own rollout distribution, this protocol enables a more controlled and comparable assessment of how closely different students match the teacher distribution.

Furthermore, as shown in Tab. 1, the KL divergence evaluated on student rollout trajectories exhibits a markedly different trend from that computed on teacher rollout trajectories and noise-corrupted trajectories constructed from real images. The former fails to reflect the true trend of student–teacher distributional discrepancy. It may provide a misleading evaluation signal.

## 2.3 FROM CROSS-SCALE DIFFUSIONOPD TO GFD

Based on the KL evaluation above, we can figure out why the performance of DiffusionOPD in $L \to M$ setting degrade severely. As shown in Table 1, the guided-output KL, conditional KL, and unconditional KL of $\mathrm { D i f f u s i o n O P D _ { L  M } }$ are all higher than those of $\mathrm { D i f f u s i o n O P D _ { M  M } }$ . “guided output” refers to the CFGcombined output used during sampling. Accordingly, the final evaluation should focus on the distributional KL divergence between the student’s guided output and the teacher’s guided output.

Attempting branch-wise supervision Recall that DiffusionOPD objective directly matches the student’s guided prediction $v _ { S } ^ { g }$ to the teacher’s guided prediction $v _ { T } ^ { g }$ . It lacks separately constraining of the conditional and unconditional branches. Such an objective may be sufficient when the student has comparable model capacity to the teacher, as in the M → M setting. However, in a large-to-small setting, the smaller student has limited approximation capacity, which may make the conditional and unconditional branches more prone to drifting away from their teacher counterparts. Such branch-wise drift may in turn lead to a larger discrepancy in the final guided output.

This observation motivates us to explicitly supervise the conditional and unconditional branches separately, thereby preventing them from drifting. A natural choice is to adopt the following training objective:

$$
\mathcal { L } = \| v _ { S } ^ { c } - v _ { T } ^ { c } \| _ { 2 } ^ { 2 } + \| v _ { S } ^ { u } - v _ { T } ^ { u } \| _ { 2 } ^ { 2 } ,\tag{3}
$$

PDM (Li et al., 2026a) also propose a similar objective as

$$
\mathcal { L } = \left\| v _ { S } ^ { c } - v _ { T } ^ { c } \right\| _ { 2 } ^ { 2 } + \alpha \left\| v _ { S } ^ { g } - v _ { T } ^ { g } \right\| _ { 2 } ^ { 2 } ,\tag{4}
$$

![](images/47e679b98a8ec5ab537638ffee291bdbcf1a8104afbd87e5cb9685e767dc4538.jpg)  
Figure 3: GFD-OPD overview. Left (motivation): the deployed error of a CFG-composed student decomposes as $\delta _ { g } = \delta _ { u } + \omega d -$ same-size distillation survives through error cancellation, large-to-small does not, and guidance folding removes the composition so that nothing is left for ω to amplify. Right (method): the process of GFD-OPD.

However, their motivation is fundamentally different from ours: we use this objective to prevent the conditional and unconditional branches from drifting in the large-to-small setting but theirs are not.

After training with separate supervision on the two branches, from Table 1, we observe that the KL divergences of both the conditional and unconditional branches are indeed reduced. However, the KL divergence of the final guided output does not decrease accordingly. Consistent with this observation, the generated images in Fig. 4 still exhibit noticeable quality degradation. As shown in Fig. 6, the corresponding quantitative metrics also remain substantially deviated from those of the teacher.

DiffusionOPDL→M  
PDM <sub>L→M</sub>  
splitKL L→M  
GFDL→M  
Large Teacher  
![](images/2c1738347733d5a9b63f28d9feae66f90d9e40ad02a56102f30f7835d0d81c59.jpg)  
Figure 4: Shared cross-architecture artifacts under a SD3.5-L teacher. Rows: the SD3.5-M base, three prior OPD recipes (DiffusionOPD, PDM, splitKL), GFD and large teacher.

Why the branch error induce but the guided field does not? To investigate the cause of the increased guided-output KL, we decompose the error of the guided prediction as

$$
\delta ^ { g } = \delta ^ { u } + w d ,\tag{5}
$$

where $\delta ^ { g } \ = \ v _ { S } ^ { g } - v _ { T } ^ { g } , \delta ^ { u } \ = \ v _ { S } ^ { u } - v _ { T } ^ { u } , \delta ^ { c } \ = \ v _ { S } ^ { c } - v _ { T } ^ { c }$ , and $d = \delta ^ { c } - \delta ^ { u }$ . Table 6 reveals that, when trained to directly match the guided output, Diffusion $\mathrm { O P D } _ { \mathrm { M } \to \mathrm { M } }$ is able to exploit the cancellation between the two error components, $\delta ^ { \bar { u } }$ and wd, resulting in a relatively small final guided error $\delta ^ { g }$ . In contrast, for $\mathrm { D i f f u s i o n O P D _ { L  M } }$ , the limited approximation capacity of the smaller student weakens such error cancellation. For PDM and SplitKL, separately supervising the conditional and unconditional branches largely removes this cancellation effect, causing the errors to be amplified rather than canceled after guidance composition. We provide a more detailed analysis in Section B.1.

This observation further suggests that, in the large-to-small setting, discrepancies in the conditional and unconditional branches can be systematically amplified by the guidance operation, ultimately leading to a larger shift in the final generative distribution.

Guidance Folding for Large-to-Small Distillation This motivates a different design: directly absorb the teacher’s guided policy into the student’s conditional branch, eliminating the need to recover the target distribution through the composition of two imperfectly matched branches. We propose Guidance Folding Distillation (GFD). Rather than jointly optimizing the student’s conditional and unconditional distributions, it converts the original two-branch composition problem into a single-branch velocity matching problem.

Formally, GFD minimize the discrepancy between the student’s conditional distribution and the teacher’s guided distribution:

$$
\operatorname* { m i n } _ { \theta } \mathbb { E } _ { x _ { t } \sim d ^ { S } } \left[ D _ { \mathrm { K L } } \left( \pi _ { S } ^ { c } ( \cdot  { \left| \ x _ { t } , c \right) \right| }  { \left| \pi _ { T } ^ { g } ( \cdot  { \left| \ x _ { t } , c \right) } \right) \right] . }\tag{6}
$$

Applying the per-step derivation of Eq. 2, the GFD objective can be written explicitly as

$$
\mathcal { L } _ { \mathrm { G F D } } ( \pmb { \theta } ) = \mathbb { E } _ { \boldsymbol { x } _ { 0 : N } \sim p _ { S , \pmb { \theta } } } \left[ \sum _ { j = 0 } ^ { N - 1 } \omega _ { j } \left. v _ { S } ^ { c } ( \boldsymbol { x } _ { t _ { j } } , t _ { j } , \boldsymbol { c } ; \pmb { \theta } ) - v _ { T } ^ { g } ( \boldsymbol { x } _ { t _ { j } } , t _ { j } ; w ) \right. _ { 2 } ^ { 2 } \right] .\tag{7}
$$

Table 1 shows that GFD achieves substantially lower KL divergence than the other methods. As illustrated in Fig. 4 and Fig. $^ { 6 , }$ GFD also restores normal generation quality, and the generated images are closer to those of the teacher in both coarse-grained and fine-grained metrics.

But a small, systematic gap remains. Fig. 6 shows that GFD (w = 1) has slightly weaker spectral energy than the teacher’s guided output. And Appendix B.2 shows that the student learns the direction of the teacher’s field but shrinks its magnitude toward its own prior. At inference this can be patched with a slightly larger guidance scale, with $w = 2$ the student distribution moves closer to the teacher’s. GFD does not constrain the unconditional branch during training. In the low-noise regime, the unconditional branch co-evolves with the conditional branch and exhibits similar shifts, resulting in the CFG only containing a small increment and avoiding excessive guidance. Although the trajectory-level KL increases slightly at $w = 2$ , we attribute this to the limited sensitivity of this metric to differences at such a small scale.

## 2.4 REWARD EXTRAPOLATION FOR LARGE-TO-SMALL DISTILLATION

Yang et al. (2026) show that OPD in language models is a KL-regularized RL objective in which the teacher enters through a dense reward, and that scaling this reward by a factor $\lambda > 1$ moves the distillation target away from the reference model and beyond the teacher. We adopt this idea to further improve the student: by extrapolating the target in the direction away from the student’s prior, the trained field is brought closer to the teacher without relying on an inference-time guidance increment.

Table 2: SD3.5 merged multi-objective OPD results.
<table><tr><td></td><td colspan="3">Task rewards</td><td colspan="4">Image quality</td></tr><tr><td>Model</td><td>GenEval</td><td>OCR</td><td>Pick Score</td><td>Aes- thetic</td><td>Image Reward</td><td>Unified Reward</td><td>HPSv3</td></tr><tr><td>SD3.5-M (base)</td><td>0.675</td><td>0.554</td><td>0.843</td><td>5.373</td><td>0.907</td><td>3.201</td><td>9.020</td></tr><tr><td>SD3.5-L (base)</td><td>0.720</td><td>0.716</td><td>0.854</td><td>5.527</td><td>1.025</td><td>3.289</td><td>9.853</td></tr><tr><td colspan="8">Per-task Large teachers</td></tr><tr><td>GenEval-Teacher</td><td>0.949</td><td>0.801</td><td>0.834</td><td>5.463</td><td>1.144</td><td>3.416</td><td>9.812</td></tr><tr><td>OCR-Teacher</td><td>0.700</td><td>0.969</td><td>0.850</td><td>5.405</td><td>1.057</td><td>3.298</td><td>8.982</td></tr><tr><td>PickScore-Teacher</td><td>0.781</td><td>0.688</td><td>0.936</td><td>6.245</td><td>1.300</td><td>3.535</td><td>11.29</td></tr><tr><td colspan="8">Per-task Medium teachers</td></tr><tr><td>GenEval-Teacher</td><td>0.917</td><td>0.558</td><td>0.845</td><td>5.336</td><td>1.074</td><td>3.332</td><td>9.024</td></tr><tr><td>OCR-Teacher</td><td>0.599</td><td>0.948</td><td>0.824</td><td>5.129</td><td>0.749</td><td>3.071</td><td>6.345</td></tr><tr><td>PickScore-Teacher</td><td>0.696</td><td>0.605</td><td>0.924</td><td>6.188</td><td>1.283</td><td>3.460</td><td>11.55</td></tr><tr><td> $\mathrm { D i f f u s i o n O P D _ { M  M } }$ </td><td>0.908</td><td>0.944</td><td>0.917</td><td>6.067</td><td>1.275</td><td>3.461</td><td>11.07</td></tr><tr><td> $\mathrm { P D M } _ { \mathrm { M }  \mathrm { M } }$ </td><td>0.902</td><td>0.940</td><td>0.913</td><td>6.124</td><td>1.261</td><td>3.455</td><td>10.97</td></tr><tr><td> $\mathrm { D i f f u s i o n O P D _ { L  M } }$ </td><td>0.943</td><td>0.962</td><td>0.901</td><td>6.222</td><td>1.009</td><td>3.365</td><td>8.844</td></tr><tr><td> $\mathrm { P D M } _ { \mathrm { L }  \mathrm { M } }$ </td><td>0.944</td><td>0.955</td><td>0.906</td><td>6.211</td><td>1.065</td><td>3.396</td><td>9.360</td></tr><tr><td> $\mathrm { s p l i t - K L _ { L  M } }$ </td><td>0.939</td><td>0.959</td><td>0.912</td><td>6.211</td><td>1.103</td><td>3.417</td><td>10.08</td></tr><tr><td> $\mathrm { G F D } _ { \mathrm { L  M } } ( \mathrm { O u r s } )$ </td><td>0.963</td><td>0.973</td><td>0.930</td><td>6.262</td><td>1.302</td><td>3.542</td><td>11.27</td></tr></table>

Table 3: SD3.5 single-objective OPD results.
<table><tr><td rowspan="2"></td><td colspan="3">Task rewards</td><td>Image quality</td></tr><tr><td>GenEval</td><td>OCR</td><td>Pick Score</td><td>HPSv3</td></tr><tr><td>SD3.5-Medium (base)</td><td>0.675</td><td>0.554</td><td>0.843</td><td>9.020</td></tr><tr><td>Teachers (SD3.5-L)</td><td>0.949</td><td>0.969</td><td>0.936</td><td>-</td></tr><tr><td>DiffusionOPDL→M</td><td></td><td></td><td></td><td></td></tr><tr><td>GenEval</td><td>0.948</td><td></td><td></td><td>6.844</td></tr><tr><td>OCR</td><td>一</td><td>0.967</td><td></td><td>6.097</td></tr><tr><td>PickScore</td><td>一</td><td>一</td><td>0.904</td><td>8.377</td></tr><tr><td> $G F D _ { \mathrm { { L }  \mathrm { { M } } } } ( O u r s )$ </td><td></td><td></td><td></td><td></td></tr><tr><td>GenEval</td><td>0.963</td><td></td><td>一</td><td>9.128</td></tr><tr><td>OCR</td><td>一</td><td>0.973</td><td></td><td>8.755</td></tr><tr><td>PickScore</td><td></td><td></td><td>0.932</td><td>11.24</td></tr></table>

Table 4: FLUX.2 merged OPD results.
<table><tr><td rowspan="2">Model</td><td colspan="3">Task rewards</td><td>Image quality</td></tr><tr><td>LongText</td><td>GenEval</td><td>Pick Score</td><td>HPSv3</td></tr><tr><td>FLUX.2-4B (base)</td><td>0.624</td><td>0.769</td><td>0.823</td><td>9.214</td></tr><tr><td> $\overline { { \mathrm { T e a c h e r } \left( \mathrm { F L U X } . 2  – 9 \mathbf { B } \right) } }$ </td><td>0.891</td><td>0.820</td><td>0.839</td><td>9.629</td></tr><tr><td> $\mathrm { P D M } _ { \mathrm { L }  \mathrm { M } }$ </td><td>0.635</td><td>0.815</td><td>0.829</td><td>8.904</td></tr><tr><td> $\mathsf { D i f f u s i o n O P D _ { L  M } }$ </td><td>0.635</td><td>0.830</td><td>0.832</td><td>9.180</td></tr><tr><td> $\bf G F D _ { L  M } ( O u r s )$ </td><td>0.692</td><td>0.880</td><td>0.839</td><td>10.20</td></tr><tr><td> $\mathrm { T e a c h e r } ( \mathrm { F L U X } . 2 \cdot 3 2 \mathbf { B } )$ </td><td>0.979</td><td>0.893</td><td>0.862</td><td>11.29</td></tr><tr><td> $\mathrm { P D M } _ { \mathrm { L }  \mathrm { M } }$ </td><td>0.763</td><td>0.887</td><td>0.849</td><td>10.17</td></tr><tr><td> $\mathrm { D i f f u s i o n O P D _ { L  M } }$ </td><td>0.828</td><td>0.885</td><td>0.856</td><td>10.69</td></tr><tr><td> $\mathrm { G F D } _ { \mathrm { L  M } } ( \mathrm { O u r s } )$ </td><td>0.857</td><td>0.908</td><td>0.859</td><td>10.47</td></tr></table>

In language models, the ExOPD objective is equivalent to standard OPD with the teacher replaced by the target $\overline { { \pi _ { \mathrm { t g t } } } } ~ \propto ~ ( \pi ^ { \star } ) ^ { \lambda } ( \pi _ { \mathrm { r e f } } ) ^ { 1 - \lambda }$ , where $\pi _ { \mathrm { r e f } }$ is a frozen reference policy. We apply the same replacement to the per-step transition kernels of diffusion OPD: $p _ { \mathrm { t g t } } \propto p _ { T } ^ { \lambda } p _ { R } ^ { 1 - \lambda }$ , where $p _ { R }$ is the kernel induced by the reference velocity $v _ { R } .$ Since both kernels are Gaussians with the same sampler-fixed variance, $p _ { \mathrm { t g t } }$ remains a valid Gaussian kernel with that variance for every λ, and only its mean moves, corresponding to the extrapolated velocity (Appendix A.1):

$$
v _ { \mathrm { t g t } } = \lambda v _ { T } + ( 1 - \lambda ) v _ { R } = v _ { T } + ( \lambda - 1 ) ( v _ { T } - v _ { R } ) .\tag{8}
$$

## 3 EXPERIMENTS

## 3.1 EXPERIMENTAL SETUP

Tasks and evaluation. We follow the three-reward suite of Flow-GRPO (Liu et al., 2026) and reuse its official prompt splits: GenEval (Ghosh et al., 2023) for compositional alignment, OCR accuracy for visual text rendering, and PickScore (Kirstain et al., 2023) optimized on Pick-a-Pic prompts and evaluated on DrawBench (Saharia et al., 2022). Task rewards are the primary metrics; following Flow-GRPO, we also report Aesthetic, ImageReward (Xu et al., 2023), UnifiedReward, and HPSv3 on DrawBench as imagequality probes. All methods are evaluated at 1024 × 1024 with the same prompts, seeds, and sampler, each at its own deployment guidance scale $( w = 4 . 5$ for CFG-composition students, w = 2 for GFD-OPD; §2).

Baselines. (a) The pretrained SD3.5-Medium and SD3.5-Large base models (Esser et al., 2024); (b) direct Flow-GRPO on both, which also serve as our teachers; (c) DiffusionOPD, PDM, and the split-KL variant of §2, trained with the same teachers, student initialization, and prompts as ours.

Implementation details. All OPD methods share the same on-policy setup: following Fast Flow-GRPO, rollouts use a 10-step reverse-SDE sampler with a 2-step training window. Training uses full-parameter fine-tuning at 1024 × 1024 with learning rate $5 \times 1 0 ^ { - 6 }$ and batch size 48 on 8 H100 GPUs; each single-task GFD-OPD run finishes in about 10 GPU-hours, versus 15 for DiffusionOPD and 160 for direct Flow-GRPO (Fig. 1). GFD-OPD composes the folded target at $w = 2$ with extrapolation λ = 1.25.

## 3.2 MAIN RESULTS

Large-to-small distillation on SD3.5 and FLUX.2. Table 3 reports the single-objective SD3.5 setting (Large→Medium). GFD-OPD outperforms DiffusionOPD on all three rewards, surpasses the teacher on GenEval (0.963 vs. 0.949) and OCR (0.973 vs. 0.969), and is the only student whose HPSv3 stays at or above the untrained base—DiffusionOPD falls clearly below it, the fidelity loss behind the artifacts of §2. In the merged multi-objective setting (Table 2), the baselines trade one side for the other: with the medium teacher they preserve image quality but are capped by its rewards; with the large teacher their rewards rise but quality drops (haze, desaturation, soft glyphs; Fig. 4). GFD-OPD attains the best task rewards (0.963 GenEval, 0.973 OCR, 0.930 PickScore) and the best image quality at the same time; per §3.3, folding alone roughly matches the teacher and the extrapolated target accounts for the rest. The same picture holds on FLUX.2 (Table 4; 9B and 32B teachers into 4B, OCR replaced by our long-text dataset): GFD-OPD leads on all task rewards, again exceeds the teacher on GenEval, and widens its LongText margin at 32B (qualitative examples in Fig. 8, appendix).

![](images/d2bf8d7141857c301f3250562753bb229386699be2fb92ee9283b61b597f7ac8.jpg)  
Figure 5: Qualitative comparison at $1 0 2 4 \times 1 0 2 4$ . Columns, left to right: SD3.5-M base; the medium and large teachers; Diffusion $\mathrm { O P D } _ { \mathrm { M } \to \mathrm { M } } ;$ and $\mathrm { G F D } _ { \mathrm { L } \to \mathrm { M } }$

Qualitative comparison. Fig. 5 presents a qualitative comparison. Since the large-teacher composition students already suffer from the artifacts shown in Fig. 4, we compare with DiffusionOPD distilled from the medium teacher, together with the base model and the two RL teachers. These medium-teacher students are free of visible artifacts, but their prompt adherence and fine-grained rendering are weaker: the cropcircle text in “Humans LOL” loses glyph structure, and the orange scene misses part of the required objects. Among all compared methods, GFD-OPD is the closest to the large teacher in both layout and fine detail. Besides, Fig. 8 shows qualitative examples for the FLUX.2 experiments of Table 4.

## 3.3 ABLATIONS

Component ablation Table 5 adds the two components on top of standard OPD one at a time. Standard OPD improves the base on all three rewards but stays well below the teacher. Guidance folding contributes most of the gain, removing the artifacts of and bringing the student roughly to the teacher level; the extrapolated target adds a further consistent gain and pushes GenEval and OCR past the teacher. This supports the attribution: folding closes the gap to the teacher, and extrapolation provides the margin beyond it.

Table 5: Ablation: standard OPD + guidance folding + extrapolated-mean target (full GFD-OPD), on the SD3.5-Medium student at 1024 × 1024; HPSv3 probes image quality.
<table><tr><td>Method</td><td>GenEval</td><td>OCR</td><td>PickScore</td><td>HPSv3</td></tr><tr><td>SD3.5-Medium (base)</td><td>0.675</td><td>0.554</td><td>0.843</td><td>9.020</td></tr><tr><td>Standard OPD</td><td>0.884</td><td>0.908</td><td>0.880</td><td>9.175</td></tr><tr><td>+ Guidance folding</td><td>0.952</td><td>0.954</td><td>0.923</td><td>11.19</td></tr><tr><td>+ Extrapolated mean (GFD-OPD)</td><td>0.963</td><td>0.973</td><td>0.930</td><td>11.27</td></tr></table>

The impact of CFG scale In Fig. 6, we evaluate different models under varying CFG scales w, measuring the discrepancy between their final sampling results and those of their respective teacher models. Diffusion $\mathrm { O P D } _ { \mathrm { M } \to \mathrm { M } }$ approaches the origin at $w = 4 . 5$ , indicating a close match to its teacher. In contrast, for DiffusionOPD, PDM, and Split-KL under the L →M setting, no choice of w brings the results close to the origin. Their outputs remain substantially different from those of the teacher. GFD achieves the closest match to the teacher at w = 2.

![](images/090e8ae9127940019c6b592c83e467471fb20b847467cf6fcc668824f9e24588.jpg)

![](images/816d0067922c964cf67c9fb3a82511229ccf3a35a047a547426e5faf17d63b16.jpg)  
Figure 6: Statistics of the student’s final output as the guidance scale w varies. Each axis is the log ratio of a statistic of the student’s samples to the teacher’s, so the teacher sits at the origin (⋆). (a) Final latent: low-frequency energy (global tone) vs. high-frequency energy (fine texture). (b) Decoded image: tonal amplitude (RMS contrast × saturation) vs. fine-texture amplitude (3px high-pass MAE). The definition of all metrics are as detailed in Appendix B.3.

## 4 RELATED WORK

On-Policy Distillation. OPD (Agarwal et al., 2024; Gu et al., 2024; Lu & Lab, 2025) post-trains language models by aligning the student with the teacher on student-generated trajectories, and is used both to compress large teachers into small students and to merge domain specialists into a single model. Yang et al. (2026) further interpret OPD as dense KL-constrained RL, we adopt this view in §2.4.

On-Policy Distillation for Diffusion Models. A recent line of work ports OPD to diffusion and flowmatching models (Ma et al., 2026; Fang et al., 2026; Wei et al., 2026). DiffusionOPD (Li et al., 2026b) derives the closed-form per-step objective and merges task-specific teachers into a unified student; PDM (Li et al., 2026a) supervises the conditional and unconditional CFG branches separately. These methods target same-family merging, while the large-to-small regime we study remains largely unexplored.

## 5 CONCLUSION

We studied on-policy distillation of diffusion models in the large-to-small regime and showed that the existing recipe does not transfer. To diagnose this we propose Fixed-State KL, which fairly makes methods comparable. The analysis attributes the failure to the CFG composition: a small student cannot fit the teacher’s guided field exactly, the composition amplifies the error of its guidance increment by the guidance scale. GFD-OPD directly absorb the teacher’s guided policy into the student’s conditional branch. It reduces the student–teacher gap while avoiding the error amplification of the CFG composition. We hope this study can encourage further investigation into how the distributional discrepancy between teacher and student affect OPD of diffusion models.

## REFERENCES

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations, volume 2024, pp. 21246–21263, 2024.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam Levi,¨ Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Zhen Fang, Wenxuan Huang, Yu Zeng, Yiming Zhao, Shuang Chen, Kaituo Feng, Yunlong Lin, Lin Chen, Zehui Chen, Shaosheng Cao, et al. Flow-opd: On-policy distillation for flow matching models. arXiv preprint arXiv:2605.08063, 2026.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132– 52152, 2023.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Minillm: Knowledge distillation of large language models. In International Conference on Learning Representations, volume 2024, pp. 32694–32717, 2024.

Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652–36663, 2023.

Bingnan Li, Haozhe Wang, Haozhong Xiong, Fangtai Wu, Jinpeng Yu, Yang Shi, Jiaming Liu, and Ruihua Huang. Rethinking classifier-free guidance in on-policy diffusion distillation. arXiv preprint arXiv:2607.24731, 2026a.

Quanhao Li, Junqiu Yu, Kaixun Jiang, Yujie Wei, Zhen Xing, Pandeng Li, Ruihang Chu, Shiwei Zhang, Yu Liu, and Zuxuan Wu. Diffusionopd: A unified perspective of on-policy distillation in diffusion models. arXiv preprint arXiv:2605.15055, 2026b.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. Advances in neural information processing systems, 38:40783–40818, 2026.

Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20251026. https://thinkingmachines.ai/blog/on-policy-distillation.

Wenhan Ma, Jianyu Wei, Liang Zhao, Hailin Zhang, Bangjun Xiao, Lei Li, Qibin Yang, Bofei Gao, Yudong Wang, Rang Li, et al. Mopd: Multi-teacher on-policy distillation for capability integration in llm posttraining. arXiv preprint arXiv:2606.30406, 2026.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic textto-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.

Qingyan Wei, Guangzhao Li, Xiaobing Tu, Yinggui Wang, Xiantao Zhang, Jinkui Ren, Xiaohong Liu, and Linfeng Zhang. Step-opd: Rethinking output targets and internal dynamics in on-policy distillation for diffusion models. arXiv preprint arXiv:2608.04887, 2026.

Haoran Xin, Anhao Zhao, Ying Sun, Jin Li, Xiaoyu Shen, and Hui Xiong. Escaping the kl agreement trap in on-policy distillation. arXiv preprint arXiv:2606.09471, 2026.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.

Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, and Yankai Lin. Learning beyond teacher: Generalized on-policy distillation with reward extrapolation. arXiv preprint arXiv:2602.12125, 2026.

Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, et al. Glm-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763, 2026.

## A DERIVATION

## A.1 REWARD EXTRAPOLATION AND THE EXTRAPOLATED TARGET

Reward extrapolation in language models. Yang et al. (2026) generalize the OPD objective $( \mathrm { E q . ~ 1 ) }$ by introducing a frozen reference policy $\pi _ { \mathrm { r e f } }$ and a reward weight λ:

$$
\begin{array} { r } { J ( \pmb \theta ) = \mathbb { E } _ { \pmb { y } \sim \pi _ { \pmb \theta } ( \cdot | \pmb x ) } \Big [ \lambda \log \frac { \pi ^ { \star } ( \pmb y | \pmb x ) } { \pi _ { \mathrm { r e f } } ( \pmb y | \pmb x ) } \Big ] - \mathrm { K L } \big ( \pi _ { \pmb \theta } ( \cdot | \pmb x ) \| \pi _ { \mathrm { r e f } } ( \cdot | \pmb x ) \big ) . } \end{array}\tag{9}
$$

Standard OPD is the case $\lambda = 1$ (the reference cancels). Define $\pi _ { \mathrm { t g t } } ( \pmb { y } | \pmb { x } ) = ( \pi ^ { \star } ) ^ { \lambda } ( \pi _ { \mathrm { r e f } } ) ^ { 1 - \lambda } / Z ( \pmb { x } )$ with $\begin{array} { r } { Z ( \pmb { x } ) = \sum _ { \pmb { u } } ( \pi ^ { \star } ) ^ { \lambda } ( \pi _ { \mathrm { r e f } } ) ^ { 1 - \lambda } } \end{array}$ . Expanding the KL term,

$$
\begin{array} { r l } & { J ( \theta ) = \mathbb { E } _ { \pi _ { \theta } } \left[ \lambda \log { \pi ^ { \star } } - \lambda \log { \pi _ { \mathrm { r e f } } } - \log { \pi _ { \theta } } + \log { \pi _ { \mathrm { r e f } } } \right] } \\ & { \qquad = \mathbb { E } _ { \pi _ { \theta } } \left[ \lambda \log { \pi ^ { \star } } + ( 1 - \lambda ) \log { \pi _ { \mathrm { r e f } } } - \log { \pi _ { \theta } } \right] = - \mathrm { K L } \big ( \pi _ { \theta } \| \pi _ { \mathrm { t g t } } \big ) + \log { Z ( x ) } . } \end{array}\tag{10}
$$

Since $Z ( { \pmb x } )$ does not depend on $\theta ,$ maximizing $J$ is the same as minimizing the reverse KL to $\pi _ { \mathrm { t g t } }$ ; that is, ExOPD is standard OPD with the teacher replaced by $\pi _ { \mathrm { t g t } }$ , where

$$
\log \pi _ { \mathrm { { t g t } } } = \log \pi ^ { \star } + ( \lambda - 1 ) \big ( \log \pi ^ { \star } - \log \pi _ { \mathrm { { r e f } } } \big ) + \mathrm { c o n s t } .\tag{11}
$$

For $0 < \lambda < 1$ the target interpolates between reference and teacher; for $\lambda > 1$ it extrapolates along the reference→teacher direction.

Diffusion. At denoising step $j ,$ the teacher and reference kernels are $p _ { T } ~ = ~ \mathcal { N } ( \mu _ { T } , \sigma _ { j } ^ { 2 } I )$ and $\begin{array} { r l } { p _ { R } } & { { } = } \end{array}$ $\mathcal { N } ( \mu _ { R } , \sigma _ { j } ^ { 2 } I )$ with the same sampler-fixed variance (§2.1). Applying the replacement above to the kernels gives $p _ { \mathrm { t g t } } \propto p _ { T } ^ { \lambda } p _ { R } ^ { 1 - \lambda }$

Lemma 1. Let $p _ { 1 } ~ = ~ \mathcal { N } ( \mu _ { 1 } , \sigma ^ { 2 } I )$ and $p _ { 2 } ~ = ~ \mathcal { N } ( \mu _ { 2 } , \sigma ^ { 2 } I )$ . For every $\lambda \ \in \ \mathbb { R }$ , the normalized product $p \propto p _ { 1 } ^ { \lambda } p _ { 2 } ^ { 1 - \lambda }$ equals $\mathcal { N } \big ( \lambda \pmb { \mu } _ { 1 } + ( 1 - \lambda ) \pmb { \mu } _ { 2 } , \sigma ^ { 2 } \pmb { I } \big )$

Proof. log $\begin{array} { r } { p ( { \pmb x } ) \stackrel { c } { = } - \frac { \lambda } { 2 \sigma ^ { 2 } } \| { \pmb x } - { \pmb \mu } _ { 1 } \| _ { 2 } ^ { 2 } - \frac { 1 - \lambda } { 2 \sigma ^ { 2 } } \| { \pmb x } - { \pmb \mu } _ { 2 } \| _ { 2 } ^ { 2 } } \end{array}$ . The coefficient of $\begin{array} { r } { \| \pmb { x } \| _ { 2 } ^ { 2 } \operatorname { i s } - \frac { \lambda + ( 1 - \lambda ) } { 2 \sigma ^ { 2 } } = - \frac { 1 } { 2 \sigma ^ { 2 } } } \end{array}$ independent of λ, so the covariance is $\sigma ^ { 2 } I ;$ ; the linear term is $\begin{array} { r } { \frac { 1 } { \sigma ^ { 2 } } \langle { \pmb x } , \lambda { \pmb \mu } _ { 1 } + ( 1 - \lambda ) { \pmb \mu } _ { 2 } \rangle } \end{array}$ , and completing the square gives the mean. □

By Lemma 1, $p _ { \mathrm { t g t } } = { \mathcal { N } } { \left( \lambda { \pmb { \mu } } _ { T } + ( 1 - \lambda ) { \pmb { \mu } } _ { R } , \sigma _ { j } ^ { 2 } { \pmb { I } } \right) }$ : extrapolation changes the mean only, and $p _ { \mathrm { t g t } }$ is a valid transition kernel for every λ, including $\lambda > 1$ . Because the transition mean is affine in the velocity with sampler-fixed coefficients, $\pmb { \mu } = \alpha _ { j } \pmb { x } _ { t _ { j } } + \beta _ { j } \pmb { v }$ , the target mean corresponds to $v _ { \mathrm { t g t } } = \lambda v _ { T } + ( 1 - \lambda ) v _ { R } ( \dot { \mathrm { E } } \mathbf { q } . 8 )$ The reverse KL between the student kernel and $p _ { \mathrm { t g t } }$ is a KL between Gaussians with equal covariance, $\| \pmb { \mu } _ { S } - \pmb { \mu } _ { \mathrm { t g t } } \| _ { 2 } ^ { 2 } / ( 2 \sigma _ { j } ^ { 2 } )$ , which in velocity space is the weighted matching loss of Eq. 2 with the target velocity $v _ { \mathrm { t g t } }$ in place of $v _ { T } ^ { g }$

## B DETAILED ANALYSIS

## B.1 ANALYSIS OF ERROR CANCELLATION

The two error components. Given the guided output

$$
v _ { g } = v _ { u } + w ( v _ { c } - v _ { u } ) ,
$$

the guided prediction error can be written as

$$
\delta ^ { g } = \delta ^ { u } + w d ,
$$

where

$$
d = \delta _ { c } - \delta _ { u } .
$$

Alternatively, d can be expressed as

$$
d = \Delta _ { S } - \Delta _ { T } ,
$$

where

$$
\Delta _ { S } = v _ { S } ^ { c } - v _ { S } ^ { u } , \qquad \Delta _ { T } = v _ { T } ^ { c } - v _ { T } ^ { u } .
$$

These two error components have distinct meanings. Specifically, $\delta ^ { u }$ represents the prompt-independent denoising prediction error, while d captures the error in the prompt-induced increment.

Table 6: Decomposition $\delta ^ { g } = \delta ^ { \mathrm { u } }$ + wd of the guided error at $w { = } 4 . 5 .$ Cross $\mathrm { t e r m } { = 2 \langle \delta ^ { \mathrm { u } } , 4 . 5 d \rangle / \| r \| ^ { 2 } }$
<table><tr><td>Student</td><td> $\lVert \delta ^ { \mathrm { u } } \rVert$ </td><td> $\lVert d \rVert$ </td><td> $\| r \|$ </td><td>cross term</td></tr><tr><td>Diffusion  $\mathrm { \Gamma _ { 1 } O P D _ { M  M } }$  (reference)</td><td>0.092</td><td>0.032</td><td>0.106</td><td>-1.56</td></tr><tr><td> $\mathrm { D i f f u s i o n O P D _ { L  M } }$ </td><td>0.090</td><td>0.043</td><td>0.177</td><td>-0.48</td></tr><tr><td> $\mathrm { P D M } _ { \mathrm { L }  \mathrm { M } }$ </td><td>0.053</td><td>0.041</td><td>0.178</td><td>-0.18</td></tr><tr><td> $\mathrm { s p l i t - K L _ { L  M } }$ </td><td>0.051</td><td>0.043</td><td>0.185</td><td>-0.18</td></tr></table>

Cancellation in $\mathrm { M }  \mathrm { M }$ and its loss in $\mathrm { L }  \mathrm { M } .$ The guided objective only constrains the sum of the two components, $\delta ^ { u } + w d$ . Therefore, any pair of errors satisfying $d = - \delta ^ { u } / w$ produces the same guided prediction error and lies in the null space of the objective. Consequently, minimizing this objective does not necessarily force both error components to vanish simultaneously. Instead, it allows the student to learn a negative correlation between d and $\delta ^ { u }$ , enabling one component to compensate for the other.

$\mathrm { D i f f u s i o n O P D _ { M  M } }$ nearly reaches such a configuration, where the two error components $\delta ^ { u }$ and d substantially cancel each other. For $\mathrm { D i f f u s i o n O P D _ { L  M } }$ , besides a similar shrinkage effect, the error in d contains additional components that are orthogonal to $\delta ^ { u }$ , resulting in weaker error cancellation.

PDM and Split-KL explicitly supervise the two branches separately. Although this reduces the magnitude of the unconditional error component $\delta ^ { u }$ , it also makes the two error components nearly orthogonal to each other. As a result, the cancellation effect is largely removed, and the final guided error remains substantial.

## B.2 SHRINKAGE OF THE TRAINED FIELD ALONG THE TEACHER’S GUIDED FIELD

Tab. 7 quantifies the residual shrinkage of the guidance-folded student and how the inference-time guidance increment compensates it. At every step we project the student’s deployed field onto the teacher’s guided field, $g = \langle { \pmb v } _ { S } ( w ) , { \pmb v } _ { T } ^ { \mathrm { c f g } } \rangle / \| { \pmb v } _ { T } ^ { \mathrm { c f g } } \| ^ { 2 }$ , so that $g = 1$ means the student carries the teacher’s full amplitude along the teacher’s direction. Results in the table is averaged over 500 prompts and over the steps of each window.

Table 7: Projection g of the student’s deployed field onto the teacher’s guided field.
<table><tr><td>Student</td><td>Probe</td><td>0-9</td><td>10-19</td><td>20-29</td><td>30-37</td><td>0–37 (total)</td></tr><tr><td> $\mathrm { G F D } \ ( w = 1 )$ </td><td> $\mathrm { P } _ { \mathrm { t e a } }$ </td><td>0.963</td><td>0.984</td><td>0.991</td><td>0.983</td><td>0.980</td></tr><tr><td></td><td> $\mathrm { P _ { r e a l } }$ </td><td>0.943</td><td>0.971</td><td>0.987</td><td>0.984</td><td>0.968</td></tr><tr><td>GFD (w = 2)</td><td> $\mathrm { P _ { t e a } }$ </td><td>1.030</td><td>0.991</td><td>0.993</td><td>0.984</td><td>1.000</td></tr><tr><td></td><td> $\mathrm { P _ { r e a l } }$ </td><td>1.023</td><td>0.978</td><td>0.989</td><td>0.985</td><td>0.997</td></tr></table>

At $w = 1$ the trained field of GFD is short along ${ \pmb v } _ { T } ^ { \mathrm { c f g } }$ in every window, most so at the high-noise steps 0–9. The student has thus learned the direction of the teacher’s guidance but shrinks its magnitude toward its own prior. Raising the guidance scale to $w = 2$ compensates the shrinkage, and the average over steps 0–37 moves from 0.97–0.98 to 1.00 on both probe sets. Besides, in the low-noise regime, the unconditional branch co-evolves with the conditional branch and exhibits similar shifts, leading to only a marginal increase of the CFG increment and avoiding excessive guidance.

## B.3 METRICS OF FIG. 6

All four quantities compare a student sample with the teacher sample generated from the same prompt and the same initial noise, over $N = 5 0 0$ prompts; each axis of Fig. 6 is a log ratio to the teacher, so the teacher sits at the origin.

Latent frequency energies (panel a). Let $z \in \mathbb { R } ^ { 6 4 \times 6 4 \times 6 4 }$ be the final latent in the model’s packed token layout (64 channels on a $6 4 \times 6 4$ grid) and $\hat { z } _ { c } = \mathrm { D C T } _ { 2 } ( z _ { c } )$ the orthonormal 2-D DCT-II of channel c. With $u , v \in \{ 0 , \frac { 1 } { 6 3 } , \ldots , 1 \}$ } the normalized frequency indices and $\rho ( u , v ) = \sqrt { u ^ { 2 } + v ^ { 2 } } / \sqrt { 2 } \in [ 0 , 1 ]$ the radial frequency, the energy of a band B is

$$
E _ { B } ( z ) = \frac { 1 } { 6 4 ^ { 3 } } \sum _ { c } \sum _ { ( u , v ) \in B } \hat { z } _ { c } ( u , v ) ^ { 2 } , \qquad B _ { \mathrm { l o w } } = \{ \rho < 0 . 2 5 \} , \quad B _ { \mathrm { h i g h } } = \{ \rho \geq 0 . 7 5 \} .\tag{12}
$$

The two coordinates of panel (a) are log $\begin{array} { r l } { \big ( \frac { 1 } { N } \sum _ { n } E _ { B } ( z _ { n } ^ { S } ) / E _ { B } ( z _ { n } ^ { T } ) \big ) } & { { } } \end{array}$ for $B = B _ { \mathrm { l o w } }$ (horizontal) and $B =$ $B _ { \mathrm { h i g h } }$ (vertical), where $z _ { n } ^ { S }$ and $z _ { n } ^ { T }$ are the student’s and the teacher’s final latents for prompt n.

Image tonal and fine-texture amplitude (panel b). The decoded $1 0 2 4 ^ { 2 }$ image I is area-downsampled to $5 1 2 ^ { 2 }$ and converted to luminance $\bar { Y } = 0 . 2 1 \bar { 2 } 6 R + 0 . 7 1 5 2 G + 0 . 0 7 2 2 B$ . Tonal amplitude combines the RMS contrast and the mean saturation,

$$
c ( I ) = { \frac { \operatorname { s t d } ( Y ) } { \operatorname { m e a n } ( Y ) } } , \qquad s ( I ) = \operatorname* { m e a n } _ { p } { \frac { \operatorname* { m a x } _ { k } I _ { k } ( p ) - \operatorname* { m i n } _ { k } I _ { k } ( p ) } { \operatorname* { m a x } _ { k } I _ { k } ( p ) } } , \qquad a _ { \mathrm { t o n e } } ( I ) = { \sqrt { c ( I ) s ( I ) } } ,\tag{13}
$$

with k ranging over the RGB channels and p over pixels. Fine-texture amplitude is the mean magnitude of the $3 \times 3$ high-pass residual of the luminance,

$$
a _ { \mathrm { t e x } } ( I ) = \operatorname* { m e a n } _ { p \in \Omega } { \big | } Y ( p ) - \mathrm { b o x } _ { 3 \times 3 } Y ( p ) { \big | } ,\tag{14}
$$

where $\mathrm { b o x } _ { 3 \times 3 }$ is the $3 \times 3$ mean filter with reflect padding and Ω excludes a 16-pixel border. The two coordinates of panel (b) are the mean log ratios $\begin{array} { r } { \frac { 1 } { N } \sum _ { n } \log \big ( a _ { \mathrm { t o n e } } ( I _ { n } ^ { S } ) / a _ { \mathrm { t o n e } } ( I _ { n } ^ { T } ) \big ) } \end{array}$ (horizontal) and $\begin{array} { r } { \frac { 1 } { N } \sum _ { n } \log \big ( a _ { \mathrm { t e x } } ( I _ { n } ^ { S } ) / a _ { \mathrm { t e x } } ( I _ { n } ^ { T } ) \big ) } \end{array}$ (vertical).

## C MORE ABLATIONS

## C.1 GFD WITH UNCONDITONAL BRANCH SUPERVISION

Supervising the unconditional branch (GFD+uncond). GFD only constrains the conditional branch and leaves the unconditional branch unconstrained. A natural variant, GFD+uncond, keeps the folded target for the conditional branch and additionally matches the student’s unconditional branch to the teacher’s:

$$
\mathcal { L } ( \pmb { \theta } ) = \mathbb { E } _ { \boldsymbol { x } _ { 0 : N } \sim p _ { S , \pmb { \theta } } } \left[ \sum _ { j = 0 } ^ { N - 1 } \omega _ { j } \Big ( \big \lVert \boldsymbol { v } _ { S } ^ { c } - \boldsymbol { v } _ { T } ^ { g } \big \rVert _ { 2 } ^ { 2 } + \big \lVert \boldsymbol { v } _ { S } ^ { u } - \boldsymbol { v } _ { T } ^ { u } \big \rVert _ { 2 } ^ { 2 } \Big ) \right] .\tag{15}
$$

Same starting point. At $w = 1$ the two students deploy fields trained toward the same target and are indistinguishable: their KL divergence values are very similar, and their final sampling results sit at the same point in Fig. 7.

Wrong increment direction. What changes is the content of the CFG increment $v _ { S } ^ { c } - v _ { S } ^ { u }$ . In GFD the unconditional branch co-evolves with the conditional branch and exhibits similar shifts, resulting in the CFG only containing a small increment and avoiding excessive guidance. Raising w mainly adds a nativescale texture correction. However, GFD+uncond matches the unconditional branch to v<sup>u</sup> , removing the co-evolving. Its CFG increment is consequently doubled in magnitude. As demonstrated in Tab. 8 and Fig. 7, this causes a larger KL divergence and drives the final sampled results further away from the teacher distribution.

Table 8: KL divergence to the teacher for GFD and GFD+uncond, in the format of Tab. 1.
<table><tr><td>Student</td><td> $\mathrm { K L } \left( \mathrm { P _ { t e a } } \right)$ </td><td> $\mathrm { K L } \left( \mathrm { P _ { r e a l } } \right)$ </td></tr><tr><td> $\mathrm { G F D } _ { \mathrm { L  M } } ( w = 1 )$ </td><td>0.050</td><td>0.030</td></tr><tr><td> $\mathrm { G F D } _ { \mathrm { L  M } } ( w = 2 )$ </td><td>0.053</td><td>0.032</td></tr><tr><td> $\mathrm { G F D + u n c o n d _ { L  M } } ( w = 1 )$ </td><td>0.048</td><td>0.029</td></tr><tr><td> $\mathrm { G F D + u n c o n d _ { L  M } } ( w = 2 )$ </td><td>0.060</td><td>0.040</td></tr></table>

![](images/2cebfa99aa499b6acb3f64a36c9909628864fc9858e5765a641bd65c8b0d90f0.jpg)

![](images/5435ab4e3d9d7135a54e9e9a1a4ae4e806eb114c9ed4084ca5f39cd18868480d.jpg)  
Figure 7: Statistics of the final output of GFD and GFD+uncond as the guidance scale w varies, in the coordinates of Fig. 6 (log ratio to the teacher, teacher at the origin).

## D ADDITIONAL QUALITATIVE RESULTS ON FLUX.2

![](images/e9d59b13a0ed643217b6eae2cd6c1e66b065c5ca750c1c3081ca238741b2a135.jpg)  
Figure 8: Qualitative comparison on FLUX.2 at 1024×1024, complementing Table 4. Columns, left to right: the FLUX.2-4B base model; the FLUX.2-9B teacher and the GFD-OPD student distilled from it into 4B; the FLUX.2-32B teacher and the GFD-OPD student distilled from it into 4B. The top four rows are GenEval prompts; the bottom four are LongText prompts probing long-form visual text rendering.