# Modeling and Understanding Neural Weights of Humanoid Policies with Diffusion

Huixuan Cheng, Bike Zhang, Shankar Sastry, and Koushil Sreenath

Abstract— Recent advances in robot learning have driven rapid progress in humanoid control, enabling increasingly agile behaviors in the real world. However, humanoid control networks are typically obtained through conventional training procedures, while comparatively little attention has been given to directly modeling the distribution of their learned weights and generating functional networks in weight space. In this work, we investigate whether high-performing humanoid control networks can be generatively modeled directly through their weights. We adopt a latent diffusion framework that couples a linear autoencoder with a Diffusion Transformer. The autoencoder compresses actor network parameters into a compact latent space, and the Diffusion Transformer models the distribution of these latents. Trained on networks collected from reinforcement learning for the Unitree G1 robot, our approach generates deployable networks that perform various motions in the real world, including walking, dancing, and spin kick. Beyond generation, we further study the learned weight space to better understand the structure captured by these humanoid control networks. We identify a structural principle in the first-layer weights that consistently reflects the robot’s kinematic prior. Notably, this kinematic prior emerges under both Reinforcement Learning (RL) and Imitation Learning (IL) training, suggesting that it is not specific to a particular optimization process but may instead reflect common knowledge associated with humanoid embodiment. Together, these results demonstrate the feasibility of generative modeling in humanoid network weight space, while providing initial insights into the information encoded by their learned weights. Additional results are available on our project website.

## I. INTRODUCTION

Humanoid control has progressed rapidly in recent years, from locomotion to whole-body control, with increasingly agile and versatile motions [1], [2], [3]. Highly dynamic humanoid motions can now be learned through imitation of human reference motions [4]. Scaling such motions has further led to humanoid foundation models with broader capabilities [5]. Together, these advances enable humanoid robots to interact with complex real-world environments.

Despite this progress, humanoid skills are still typically acquired by training separate control networks for different tasks, which are then maintained as individual expert policies. An alternative is to directly model and generate these expert networks in weight space. A generative model over network weights could serve as a policy generator, synthesizing control networks for different motions or tasks, potentially conditioned on task descriptors, reference trajectories, or other embeddings, and providing a compact and flexible representation of diverse humanoid skills.

![](images/b62b54c2407cb319a73839968f289fe2e9d7555c118a240f77b666367b437f66.jpg)  
Fig. 1: Real-world deployment of networks with generated weights. Top: A walking motion. Middle: A dancing motion. Bottom: A spin kick motion. For each motion, the corresponding network weights are denoised by a motionspecific diffusion model and subsequently deployed zero shot in the real world.

Early approaches to weight generation, such as hypernetworks, used a learned generator conditioned on a task or context to produce the weights of a target network [6], [7]. More recent work has explored diffusion-based methods for modeling and generating network weights. In the image domain, diffusion-generated weights can perform image classification, with recent extensions to language classification and tabular prediction [8], [9], [10], [11]. This direction has since extended to embodied control, where latent diffusion models conditioned on target behaviors or trajectories generate network weights for HalfCheetah and Walker in simulation [12], with recent progress extending to manipulation tasks [13], [14]. Notably, weights for quadruped locomotion can also be sampled and deployed in the real world [15]. Generative modeling of humanoid control networks, however, remains largely unexplored, in part due to their high-dimensional weight space. Neural network weights are known to be substantially redundant, with an effective dimensionality far below the nominal parameter count [16], [17], [18], [19], suggesting that humanoid network weights may likewise be modeled through compact latent representations.

![](images/b72f4be4bea4395215a6840bb6aebafaf8e9ada29e2e5d5ff5038118a9dcaf72.jpg)  
Fig. 2: Method overview. (a) We collect actor weights from 750 training runs with random seeds and retain eight postconvergence networks from each run. Each network is a three-layer MLP. (b) The cluster-wise linear autoencoder is implemented using single-layer encoders and decoders. (c) A diffusion transformer is trained with x-pred loss to model the latent distribution. During inference, the model generates novel network weights that can be deployed in the real world.

Beyond weight generation, a complementary question is what information is encoded in humanoid control network weights. To date, weight space analysis has largely focused on the image domain, where features are comparatively easy to associate with individual layers, from low-level edges and colors to class-specific concepts [20], [21]. In robotics, by contrast, such analysis remains limited: existing studies mostly examine which weights matter, for instance using pruning to identify task-relevant inputs [22]. A systematic understanding of what humanoid control networks encode in their weights therefore remains largely unexplored. Such analysis could reveal regularities that inform more compact architectures, while structures shared across motions may guide transfer learning for acquiring new policies.

In this work, we address two gaps: generating humanoid control networks in weight space and interpreting the structure and information encoded in their weights. We collect a large dataset of RL-trained networks for the Unitree G1 humanoid robot performing walking, dancing, and spin kick motions, and train a linear autoencoder together with a Diffusion Transformer [23] to model their weight distribution. We evaluate the quality and diversity of generated networks in simulation and assess their sim-to-real transfer on hardware, as shown in Fig. 1. Alongside this generative modeling effort, we systematically analyze the collected humanoid networks, investigating the structure encoded by RL-trained networks and how different training paradigms, including imitation learning, and training configurations, such as domain randomization, affect their weight space distributions.

## II. METHODOLOGY

In this section, we will present our method to generatively model policy weights, as illustrated in Fig. 2. We first describe the collection process for the weight dataset used for generative model training, and then introduce our approach, which consists of two components: (1) a linear autoencoder that compresses actor network weights into compact latent representations, and (2) a diffusion transformer trained on latent space to model their distribution.

Weight Dataset. We collect a dataset D of actor networks for the Unitree G1 robot across three motions: walking, dancing, and spin kick. Each motion is trained via RL in mjlab [24], and we retain 6,000 actor networks per motion. All actor networks share a common architecture: a threelayer multi-layer perceptron (MLP) with hidden dimensions [128, 128] and ELU activations, resulting in approximately 30k ∼ 40k parameters per network, with the variation arising from different observation space dimensions for each motion. To ensure broad coverage of the weight distribution, we launch 750 independent training runs with random seeds and collect 8 post-convergence networks per seed, as shown in Fig. 2a. We disabled observation normalization during RL training and only actor weights are considered for generation. Linear Autoencoder. We use a linear autoencoder to compress the actor network weights into latents to reduce dimensionality. The encoder and decoder are implemented as single linear layers without bias terms or activation functions. Given an actor network π<sub>θ</sub> parameterized by $\theta = \left[ w _ { 1 } , b _ { 1 } , w _ { 2 } , b _ { 2 } , w _ { 3 } , b _ { 3 } \right] \in \mathcal { D }$ , we flatten and concatenate the weights $w _ { i }$ and biases $b _ { i }$ of all layers in sequence. The encoder E is composed of $\mathbf { E } _ { 1 }$ that compresses the cluster $[ w _ { 1 } , b _ { 1 } ]$ into a 1024-dimensional latent vector and $\mathbf { E } _ { 2 }$ that compresses the cluster $[ w _ { 2 } , b _ { 2 } , w _ { 3 } , b _ { 3 } ]$ into another 1024-dimensional latent vector, resulting in a cluster-wise encoding scheme for E. The two latent vectors are then concatenated to form a 2048-dimensional latent representation z used for diffusion training. The decoder D reconstructs the network parameters in the same cluster-wise manner, as illustrated in Fig. 2b. We train separate autoencoders for different motions. The training objective is the mean squared reconstruction error: $\mathcal { L } _ { \mathrm { A E } } = \Vert \theta - \tilde { \theta } \Vert _ { 2 } ^ { 2 }$ , where $\tilde { \theta } =$ $\mathbf { D } ( \mathbf { E } ( \theta ) ) , \theta \in \mathcal { D }$ , and $\tilde { \theta }$ consists of the reconstructed weights $\tilde { w } _ { i }$ and biases $\tilde { b } _ { i }$

TABLE I: Generated Networks vs. Training Data
<table><tr><td></td><td colspan="2">Dataset</td><td colspan="2">DiT (mean)</td><td colspan="2">DiT (best 90%)</td><td colspan="2">DiT (best)</td></tr><tr><td>Motion</td><td>Succ ↑</td><td>TE↓</td><td>Succ ↑</td><td>TE↓</td><td>Succ ↑</td><td> $\mathrm { T E \downarrow }$ </td><td>Succ ↑</td><td>TE↓</td></tr><tr><td>Walking</td><td>0.999±0.001</td><td> $0 . 1 6 2 { \pm } 0 . 0 0 3$ </td><td> $0 . 9 9 3 { \pm } 0 . 0 2 3$ </td><td> $0 . 1 9 1 { \pm } 0 . 0 2 0$ </td><td> $0 . 9 9 7 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 1 8 6 { \pm } 0 . 0 1 3$ </td><td>1.000</td><td>0.161</td></tr><tr><td>Dancing</td><td>0.998±0.004</td><td> $0 . 0 3 7 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 9 4 1 { \pm } 0 . 2 1 3$ </td><td>0.042±0.007</td><td> $0 . 9 9 6 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 0 4 0 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td>1.000</td><td>0.036</td></tr><tr><td>Spin kick</td><td>0.997±0.004</td><td> $0 . 0 4 6 { \pm } 0 . 0 0 2$ </td><td> $0 . 9 4 3 { \pm } 0 . 2 8 9$ </td><td> $0 . 0 5 3 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 9 9 3 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 0 5 1 { \pm } 0 . 0 0 4$ </td><td>1.000</td><td>0.044</td></tr></table>

Dataset, DiT-mean and DiT best 90% results are reported as mean ± std. DiT-best-90% is the mean over the 90 best of the 100 generated samples. DiT best (best of 100 samples) corresponds to the per-metric optimum. Samples are ranked separately for each metric. TE denotes tracking error: MPKPE (m) for motion tracking, and mean linear velocity command error (m/s) for walking. For walking, the command includes step changes, so the tracking error also captures transient errors during command transitions.

TABLE II: Effect of Transformer Embedding Dimension
<table><tr><td>Motion</td><td colspan="2">embedding=512</td><td colspan="2">embedding=768</td><td colspan="2">embedding=1024</td></tr><tr><td></td><td> $\mathrm { P r o g ~ \uparrow ~ }$ </td><td> $\mathrm { T E \downarrow }$ </td><td> $\mathrm { P r o g ~ \uparrow ~ }$ </td><td> $\mathrm { T E \downarrow }$ </td><td> $\mathrm { P r o g ~ \uparrow ~ }$ </td><td>TE↓</td></tr><tr><td>Walking</td><td> $0 . 6 4 3 \pm 0 . 3 7 3$ </td><td> $0 . 5 1 3 \pm 0 . 2 8 4$ </td><td> $0 . 8 9 0 \pm 0 . 2 5 3$ </td><td> $0 . 2 9 8 \pm 0 . 1 3 9$ </td><td> $\mathbf { 0 . 9 9 7 \pm 0 . 0 1 0 }$ </td><td> $\mathbf { 0 . 1 9 1 \pm 0 . 0 2 0 }$ </td></tr><tr><td>Dancing</td><td> $0 . 5 3 6 \pm 0 . 2 3 1$ </td><td> $0 . 0 7 6 \pm 0 . 0 1 2$ </td><td> $0 . 6 5 1 \pm 0 . 2 4 3$ </td><td> $0 . 0 7 1 \pm 0 . 0 1 2$ </td><td> $\mathbf { 0 . 9 6 5 \pm 0 . 1 5 4 }$ </td><td> $\mathbf { 0 . 0 4 2 \pm 0 . 0 0 7 }$ </td></tr><tr><td>Spin kick</td><td> $0 . 2 9 6 \pm 0 . 1 2 2$ </td><td> $0 . 0 8 2 \pm 0 . 0 0 8$ </td><td> $0 . 8 7 3 \pm 0 . 1 9 8$ </td><td> $0 . 0 6 7 \pm 0 . 0 1 0$ </td><td> $\mathbf { 0 . 9 6 9 \pm 0 . 1 3 6 }$ </td><td> $\mathbf { 0 . 0 5 3 \pm 0 . 0 0 7 }$ </td></tr></table>

All results are reported as mean ± std over diffusion samples. Prog is task progress in the range [0, 1]: for motion tracking, it is the fraction of reference motion frames completed; for walking, it denotes the fraction of the max episode duration (20 s) survived. TE denotes tracking error: MPKPE (m) for motion tracking, and mean linear velocity command error (m/s) for walking.

TABLE III: Comparison between diffusion-generated networks and noise-perturbed training networks across motions.
<table><tr><td></td><td colspan="3">Walking</td><td colspan="3">Dancing</td><td colspan="3">Spin kick</td></tr><tr><td>Group</td><td>σ</td><td>Succ ↑</td><td>TE↓</td><td>σ</td><td>Succ ↑</td><td>TE↓</td><td>σ</td><td>Succ ↑</td><td>TE↓</td></tr><tr><td>Training</td><td>0.00</td><td>0.999</td><td>0.162</td><td>0.00</td><td>0.998</td><td>0.037</td><td>0.00</td><td>0.997</td><td>0.046</td></tr><tr><td>Noise σ=0.01</td><td>0.01</td><td>0.999</td><td>0.161</td><td>0.01</td><td>0.997</td><td>0.036</td><td>0.01</td><td>0.997</td><td>0.047</td></tr><tr><td>Noise σ=0.05</td><td>0.05</td><td>0.998</td><td>0.172</td><td>0.05</td><td>0.990</td><td>0.042</td><td>0.05</td><td>0.984</td><td>0.052</td></tr><tr><td>Noise σ=0.10</td><td>0.10</td><td>0.959</td><td>0.226</td><td>0.10</td><td>0.629</td><td>0.057</td><td>0.10</td><td>0.650</td><td>0.068</td></tr><tr><td>Noise, matched</td><td>0.110</td><td>0.914</td><td>0.248</td><td>0.078</td><td>0.865</td><td>0.051</td><td>0.090</td><td>0.783</td><td>0.064</td></tr><tr><td>Diffusion</td><td>0.110</td><td>0.993</td><td>0.191</td><td>0.078</td><td>0.941</td><td>0.042</td><td>0.090</td><td>0.943</td><td>0.053</td></tr></table>

All results are reported as mean over 100 networks. Distance to the training distribution is expressed as an equivalent noise level σ. The upper block reports performance under a fixed noise schedule applied to training networks. The lower block reports a distance-matched baseline, where the noise level is set to match the equivalent distance of the diffusion-generated networks, alongside the diffusion results. Succ and TE are the same as in Table I.

Diffusion Transformer. Following the latent diffusion paradigm [25], we train a diffusion transformer (DiT) to model the latent distribution for each motion [23], [26], while keeping the autoencoders frozen. The overall pipeline is detailed in Fig. 2c. We first encode the actor weight θ into a clean latent $z _ { 0 } = \mathbf { E } ( \theta )$ , and then inject noise at each timestep $t \in \{ 1 , \ldots , T \}$ to produce:

$$
z _ { t } = \sqrt { \bar { \alpha } _ { t } } z _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , \quad \epsilon \sim \mathcal { N } ( 0 , I ) .\tag{1}
$$

where $\bar { \alpha } _ { t }$ is a pre-defined noise schedule. Instead of the conventional ϵ-prediction, we optimize the diffusion model via x-prediction, i.e., train the DiT to directly predict the clean latent $z _ { 0 }$ [27]:

$$
\mathcal { L } _ { \mathrm { D i T } } ( \phi ) = \mathbb { E } _ { t , z _ { 0 } , \epsilon \sim \mathcal { N } ( 0 , I ) } \Big [ \| \mathrm { D i T } _ { \phi } ( z _ { t } , t ) - z _ { 0 } \| ^ { 2 } \Big ] ,\tag{2}
$$

where $\mathrm { D i T } _ { \phi }$ is the x-pred network parameterized by $\phi .$ We use a transformer backbone to capture the cross-layer and global correlations in weight space, in contrast to the U-Net backbone in [15]. During inference, zˆ is denoised from $\mathrm { D i T } _ { \phi } .$ , and subsequently decoded back to weight space via $\hat { \theta } ~ = ~ \mathbf { D } ( \hat { z } _ { 0 } )$ . The generated weight $\hat { \theta }$ then produces the generated actor network $\pi _ { \hat { \theta } } ,$ which can be deployed for simto-real transfer.

## III. WEIGHT GENERATION EXPERIMENTS

After training the autoencoder and the diffusion transformer, we evaluate the proposed approach in this section. Our experiments are designed to assess: (1) the motion performance of the generated actor network weights, measured by success rate and tracking error, compared to baselines, (2) the sim-to-real transfer capability on Unitree G1, and (3)

![](images/1338f73d5ab7930c38269469e9ae22fc6f927ebfd36f1b88e769cada90ec80a9.jpg)  
Fig. 3: t-SNE visualization. Training and generated network weights for the spin kick motion.

whether the generated weights merely exhibit memorization of the training network weights or generalization out of the training distribution.

## A. Experimental Setup

Simulation Environment. We use mjlab [24] as the simulation environment for all motions. We consider three motions of increasing complexity: (T1) Walking, basic locomotion; (T2) Dancing, a rhythmic whole-body tracking motion; and (T3) Spin kick, a highly dynamic kicking motion.

Baselines. We adopt two sets of baselines: (1) the RLcollected network dataset, and (2) networks generated by diffusion transformers with different embedding dimensions, namely 512, 768, and 1024 (ours). Performance is evaluated using success rates (Succ) and tracking errors (TE) in simulation environment.

## B. Results

Motion Quality. We first compare the generated networks with the RL-collected networks in terms of success rate and tracking error across three motions. The tracking error is defined as the mean linear velocity error for walking and mean per-keybody position error (MPKPE) for dancing and spin kick. The success rate is computed as the fraction of successfully completed episodes across 1,024 parallel environments. As shown in Table I, DiT-generated networks achieve slightly lower average performance than the dataset networks, while the best of 100 generated samples reaches comparable performance. To further assess generation quality, we report the success rate and tracking error over the top 90% of generated samples. Their mean and variance are comparable to those of the RL-collected networks, indicating that only fewer than 10% of generated samples exhibit performance collapse.

We ablate the transformer embedding dimension, which we find to be a critical factor affecting performance during the training process. We measure the task progress (Prog) and tracking error (TE) across different motions and models, as detailed in Table II. Task progress is defined as the fraction of reference motion frames completed for dancing and spin kick, and as the fraction of the maximum episode duration (20 s) survived for walking. We observe a clear improvement in performance as the embedding dimension increases, which is most pronounced for the spin kick motion, followed by dancing and walking. This observation suggests that highly dynamic motions are associated with more complex weight distributions, thereby requiring higher-capacity models to capture them effectively.

Sim-to-Real Transfer. To validate the practical applicability of our approach, we deploy the generated actor networks on the Unitree G1 robot without additional fine-tuning. Since the RL-collected networks are trained with domain randomization, the sim-to-real gap is implicitly incorporated into the weight distribution, eliminating the need for additional sim-to-real considerations during diffusion training. Once the model is trained, we can sample new networks and directly apply the same deployment protocol in realworld experiments [4]. The deployed networks successfully perform all motions, as shown in Fig. 1.

Memorization vs. Generalization. A natural question is whether our approach merely memorizes the training weights or instead learns the underlying weight distribution. To investigate this, we embed both the training and generated weights into a shared t-SNE space using 100 samples from each set, as shown in Fig. 3. Taking spin kick as an example, the generated samples (stars) largely lie within the support of the training distribution (dots), indicating that our approach primarily produces in-distribution weights. We find that even samples appearing at the boundary of the training distribution can successfully perform real-world spin kick on G1.

To examine whether the diffusion model generalizes beyond memorizing training weights or their local noisy variants, we compare diffusion-generated networks with noiseperturbed training networks using the previously defined success rate and tracking error. We define the noise level σ as a normalized distance in weight space. For each weight matrix or bias vector $W _ { l }$ , we apply

$$
\begin{array} { r } { W _ { l } ^ { \prime } = W _ { l } + \sigma s _ { l } \varepsilon _ { l } , \quad \varepsilon _ { l } \sim { \mathcal N } ( 0 , I ) , \quad s _ { l } = \mathrm { S t d } ( W _ { l } ) , } \end{array}
$$

where $W _ { l } ^ { \prime }$ denotes the perturbed weights, s<sub>l</sub> is the standard deviation of $W _ { l } , \ \varepsilon _ { l }$ is i.i.d. Gaussian noise, and σ controls the perturbation magnitude relative to the scale of W<sub>l</sub>.

For a diffusion-generated network D, we identify its nearest training network $W$ in weight space and define the equivalent noise level as the root mean square of $( D _ { l } \mathrm { ~ - ~ }$ $W _ { l } ) / s _ { l }$ over all weights and biases, where $D _ { l }$ and $W _ { l }$ denote the corresponding parameters of the generated and training networks. We then perturb training networks using this matched noise level and the same layer-wise scaling, yielding a distance-matched baseline for comparison with diffusion-generated networks. The simulation results for all motions are reported in Table III.

As shown, at matched distance from the training set, diffusion-generated networks consistently outperform noiseperturbed baselines and achieve higher success rate and lower tracking error across all motions. These results indicate that the diffusion model captures a structured weight-space distribution that supports valid, high-performing solutions beyond individual training samples. We characterize this behavior as local generalization, where the model learns the local geometry of the weight manifold and generates functionally valid neighboring networks.

a. Heat map for walking (first-layer)  
![](images/55bb8c1a7d4ccf3aec3b42f7279d5d313690eab8a4ff1846f58e48d6ae0d2aee.jpg)

![](images/f5b027f40bfe47308db553ed1cd83175265dfa1252b1d020098839f6815bbcff.jpg)  
Fig. 4: Kinematic prior in weights. (a) A visualization of the first-layer weight heat map for the walking motion, which reveals a structural principle that certain input regions exhibit larger weight magnitudes. (b) Joint-wise weight distributions across motions demonstrate that hip pitch, knee, and waist joints are assigned higher importance, suggesting the presence of a kinematic prior.

## IV. WEIGHT SPACE ANALYSIS

Having presented the experimental results of the network weight denoising approach, we next provide an analysis of the weight space. Our study primarily focuses on the first layer of RL-trained actor networks, whose inputs carry direct semantic meaning. We apply observation normalization to reduce the effect of different input magnitudes. This analysis aims to identify structural patterns in the learned weights and examine how consistently they emerge across training settings. Specifically, we investigate: (i) how weights behave across different input regions and motions, (ii) how weights differ between RL and IL for the same motion and (iii) how domain randomization during RL training affects the weight space distribution.

## A. Kinematic Prior in Weights

We visualize the first-layer weight matrix in Fig. 4a, where darker colors indicate larger weight magnitudes. Building upon prior analysis of RL network weights suggesting that parameters with larger magnitudes tend to be more important [22], we observe that certain input regions, such as joint pos and gravity, consistently exhibit larger magnitudes than others, including last action and command. The complete observation space is illustrated in the left column of Fig. 4a.

![](images/22c1788fb2e2ab192ce93a5de1943d20e90549fe7c24fd629058c3a1b84abf48.jpg)  
Fig. 5: First-layer weight comparison between RL and IL for walking. Top: RL-trained weight heat map. Bottom: IL-trained weight heat map, distilled from the top RL network. While trained via different optimization objectives, they both exhibit kinematic prior in the first-layer weights.

We further analyze the inputs shared across motions, namely joint pos and joint vel, each consisting of 29 dimensions corresponding to all DoFs, ordered as legs (12), waist (3), and arms (14). For each joint, we compute the mean absolute weight value across the 128 hidden nodes and further average it over 50 random seeds, yielding a measure of per-joint importance. Although the overall weight magnitudes vary across motions, the normalized joint-wise distributions are highly consistent, as illustrated in Fig. 4b. This observation reveals a structural principle in the firstlayer weights: the network assigns the greatest importance to the waist region and to joint indices 1 and 7, corresponding to the left and right hip pitch joints, which collectively define the pelvic region. These are followed by the remaining joints in legs and then the arms. We therefore hypothesize that the first-layer weight matrix captures a kinematic prior rather than motion-specific knowledge. This interpretation aligns with findings in human motor control and showing that the trunk primarily provides postural stability, while the limbs are responsible for mobility and motion execution [28].

![](images/69b71acb3d1d7082121b2c01b87467e2e747f4e387e437134a0ecb68ba40bb4a.jpg)  
Fig. 6: Kinematic prior for both RL and IL networks. For each motion, the dashed and solid curves largely overlap after distillation, indicating that the first-layer kinematic prior is preserved in the IL network.

## B. RL and IL Weight Comparison

The emergence of a kinematic prior in the weight space raises a natural question: does it depend on how the network is trained? To investigate whether the same kinematic prior persists across different training paradigms, we further analyze networks trained via imitation learning (IL), or policy distillation, to perform the same motions.

For each motion, we distill an IL network from a pretrained RL network. We rollout the RL network in parallel environments to collect a demonstration buffer, with each trajectory covering a complete motion episode. The resulting state-action pairs are then used to train the IL network via offline supervised regression with MSE loss. The IL network uses the same initialization, architecture, activation functions, observation normalization, and inputs as the RL network. The distilled network successfully completes the target motion, yielding a functionally equivalent network obtained through an entirely different optimization process. We distill 50 RL networks for each motion and obtain 50 IL networks respectively. The simulation benchmark of success rate and tracking error is provided in Table IV.

Fig. 5 compares the first-layer weight heatmaps of RL and IL networks for walking. Despite being trained through different optimization processes, the RL and IL networks exhibit a similar kinematic prior in their first-layer weights. As shown in Fig. 6, when averaged over 50 networks for each training paradigm and motion, the normalized per-joint importance curves of RL and IL nearly overlap across all motions. Nevertheless, the resulting RL and IL solutions are not trivially connected in weight space: RL weights tend to have larger magnitudes, and interpolation between RL and IL does not yield a network that can complete the motion. This suggests that the observed kinematic prior is not specific to the RL optimization process, but may instead reflect common knowledge associated with humanoid embodiment and the underlying motion kinematics. Such training method independent structure further suggests that the kinematic prior could potentially serve as an initialization or warm start for future humanoid policy learning.

TABLE IV: Simulation results for RL teachers and their distilled IL students.
<table><tr><td>RL / IL</td><td>Walking</td><td>Dancing</td><td>Spin kick</td></tr><tr><td>Succ</td><td>0.9994/0.9979</td><td>0.9973/0.9948</td><td>0.9972/0.9919</td></tr><tr><td>TE</td><td>0.1599/0.1609</td><td>0.0340/0.0348</td><td>0.0466/0.0478</td></tr></table>

All results are reported as the mean over 50 RL or IL networks, evaluated through 512 parallel rollouts in simulation.

## C. Effect of Domain Randomization on Weight Space

Domain randomization (DR) is a key component of simto-real learning, but its effect is typically evaluated only through performance. We instead examine how it changes the weight space. For each motion, we train two RL networks from the same initialization, with and without DR, and visualize their first-layer weight heatmaps and their elementwise weight difference, as shown in Fig. 7.

Although the weights learned with and without DR exhibit highly similar overall structures, their element-wise difference is substantial and itself forms a structure resembling the RL-trained weights. This suggests that DR significantly reshapes the learned weights while preserving their overall distribution. However, the difference weights alone do not constitute a network that can accomplish the motion, nor does linear interpolation between the networks trained with and without DR work across all motions.

These results suggest that the effect of DR cannot be represented as a simple linear displacement in weight space. Instead, DR induces a structured reorganization of the weights throughout RL training while largely preserving the overall weight-space structure associated with the motion.

## V. LIMITATIONS AND FUTURE WORK

Our analysis is mainly limited to the first-layer, while deeper layers and the effect of domain randomization on weight space remain less understood. The current study is also restricted to a limited set of motions and a single embodiment, and the generality of the observed weight space structures remains to be validated. Although we identify interpretable weight structures, these insights have not yet translated into improved performance or policy generation.

Future work may extend the analysis to deeper layers and explore conditional weight generation guided by behaviors, trajectories, or text, supported by more diverse datasets across motions and embodiments.

## VI. CONCLUSION

In this work, we studied humanoid control through the lens of neural network weight space distribution. We showed that latent diffusion can generate functional networks that transfer zero-shot to the real world. Moreover, our analysis revealed that the first-layer weights consistently encode the robot’s kinematic prior across all motions. We further found that though RL and IL networks are trained through different optimization objectives, they exhibit similar weight space structure. Together, these results provide new insights into the structure and interpretability of humanoid network weights.

![](images/ce8862f2f71fc88514dba23dc1ef8dae04f5f95c81b76022c69dd1f1ce707949.jpg)  
Fig. 7: Effect of domain randomization on first-layer weights for dancing. First-layer weight heat maps of RL networks trained from the same initialization (a) with DR and (b) without DR, together with (c) their element-wise difference. Despite their overall similarity, the structured weight difference resembles the RL-trained weights.

## ACKNOWLEDGMENT

This work was supported in part by the DARPA program Design of Robustly Implementable Autonomous and Intelligent Machines under award HR00112490425. We thank Prof. Claire Tomlin for valuable discussions and insightful feedback, and Qiayuan Liao for generous assistance with the experiments. We are also grateful to Yiyang, Haoyi, Chuye, Gechen, Siming, and Sangli for experimental support.

## REFERENCES

[1] Z. Fu, Q. Zhao, Q. Wu, G. Wetzstein, and C. Finn, “Humanplus: Humanoid shadowing and imitation from humans,” arXiv preprint arXiv:2406.10454, 2024.

[2] T. He, Z. Luo, X. He, W. Xiao, C. Zhang, W. Zhang, K. Kitani, C. Liu, and G. Shi, “Omnih2o: Universal and dexterous humanto-humanoid whole-body teleoperation and learning,” arXiv preprint arXiv:2406.08858, 2024.

[3] M. Ji, X. Peng, F. Liu, J. Li, G. Yang, X. Cheng, and X. Wang, “Exbody2: Advanced expressive humanoid whole-body control,” arXiv preprint arXiv:2412.13196, 2024.

[4] Q. Liao, T. E. Truong, X. Huang, Y. Gao, G. Tevet, K. Sreenath, and C. K. Liu, “Beyondmimic: From motion tracking to versatile humanoid control via guided diffusion,” arXiv preprint arXiv:2508.08241, 2025.

[5] Z. Luo, Y. Yuan, T. Wang, C. Li, S. Chen, F. Castaneda, Z.-A. Cao, J. Li, D. Minor, Q. Ben et al., “Sonic: Supersizing motion tracking for natural humanoid whole-body control,” arXiv preprint arXiv:2511.07820, 2025.

[6] D. Ha, A. Dai, and Q. V. Le, “Hypernetworks,” arXiv preprint arXiv:1609.09106, 2016.

[7] E. Sarafian, S. Keynan, and S. Kraus, “Recomposing the reinforcement learning building blocks with hypernetworks,” in International Conference on Machine Learning. Pmlr, 2021, pp. 9301–9312.

[8] K. Wang, D. Tang, B. Zeng, Y. Yin, Z. Xu, Y. Zhou, Z. Zang, T. Darrell, Z. Liu, and Y. You, “Neural network diffusion,” arXiv preprint arXiv:2402.13144, 2024.

[9] S. Gupta, S. Biggs, M. Laber, Z. Shafi, R. Walters, and A. Paul, “Deepweightflow: Re-basined flow matching for generating neural network weights,” arXiv preprint arXiv:2601.05052, 2026.

[10] B. Zeng, Y. Yin, Z. Xu, and Z. Liu, “Generative modeling of weights: Generalization or memorization?” arXiv preprint arXiv:2506.07998, 2025.

[11] W. Peebles, I. Radosavovic, T. Brooks, A. A. Efros, and J. Malik, “Learning to learn with generative models of neural network checkpoints,” 2022. [Online]. Available: https://arxiv.org/abs/2209. 12892

[12] S. Hegde, S. Batra, K. R. Zentner, and G. S. Sukhatme, “Generating behaviorally diverse policies with latent diffusion models,” 2023. [Online]. Available: https://arxiv.org/abs/2305.18738

[13] S. Hegde, S. Das, G. Salhotra, and G. S. Sukhatme, “Warpd: World model assisted reactive policy diffusion,” 2025. [Online]. Available: https://arxiv.org/abs/2410.14040

[14] M. Rocca, S. Darkner, K. Erleben, and S. Andrews, “Policy-space diffusion for physics-based character animation,” ACM Transactions on Graphics, vol. 44, no. 3, pp. 1–18, 2025.

[15] Y. Liang, T. Xu, K. Hu, G. Jiang, F. Huang, and H. Xu, “Makean-agent: A generalizable policy network generator with behaviorprompted diffusion,” Advances in Neural Information Processing Systems, vol. 37, pp. 19 288–19 306, 2024.

[16] R. Yang, J. Mao, and P. Chaudhari, “Does the data induce capacity control in deep learning?” 2022. [Online]. Available: https://arxiv.org/abs/2110.14163

[17] M. Huh, H. Mobahi, R. Zhang, B. Cheung, P. Agrawal, and P. Isola, “The low-rank simplicity bias in deep networks,” 2023. [Online]. Available: https://arxiv.org/abs/2103.10427

[18] M. K. Transtrum, B. Machta, K. Brown, B. C. Daniels, C. R. Myers, and J. P. Sethna, “Sloppiness and emergent theories in physics, biology, and beyond,” 2015. [Online]. Available: https://arxiv.org/abs/1501.07668

[19] J. Mao, I. Griniasty, H. K. Teoh, R. Ramesh, R. Yang, M. K. Transtrum, J. P. Sethna, and P. Chaudhari, “The training process of many deep networks explores the same low-dimensional manifold,” Proceedings of the National Academy of Sciences, vol. 121, no. 12, p. e2310002121, 2024.

[20] M. D. Zeiler and R. Fergus, “Visualizing and understanding convolutional networks,” in European conference on computer vision. Springer, 2014, pp. 818–833.

[21] J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, “How transferable are features in deep neural networks?” Advances in neural information processing systems, vol. 27, 2014.

[22] M. A. Vischer, R. T. Lange, and H. Sprekeler, “On lottery tickets and minimal task representations in deep reinforcement learning,” arXiv preprint arXiv:2105.01648, 2021.

[23] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4195–4205.

[24] K. Zakka, Q. Liao, B. Yi, L. L. Lay, K. Sreenath, and P. Abbeel, “mjlab: A lightweight framework for gpu-accelerated robot learning,” arXiv preprint arXiv:2601.22074, 2026.

[25] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684–10 695.

[26] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[27] T. Li and K. He, “Back to basics: Let denoising generative models denoise,” arXiv preprint arXiv:2511.13720, 2025.

[28] D. A. Winter, Biomechanics and motor control of human movement. John wiley & sons, 2009.

[29] S. K. Ainsworth, J. Hayase, and S. Srinivasa, “Git re-basin: Merging models modulo permutation symmetries,” arXiv preprint arXiv:2209.04836, 2022.

[30] S. Kornblith, M. Norouzi, H. Lee, and G. Hinton, “Similarity of neural network representations revisited,” in International conference on machine learning. PMlR, 2019, pp. 3519–3529.

[31] K. He, X. Zhang, S. Ren, and J. Sun, “Delving deep into rectifiers: Surpassing human-level performance on imagenet classification,” 2015. [Online]. Available: https://arxiv.org/abs/1502.01852

## APPENDIX

## VII. DATA COLLECTION

Our dataset consists of actor network weights collected from RL training using mjlab, a MuJoCo-based framework built on PPO (RSL-RL) with default hyperparameters. For each motion, the actor is parameterized as a three-layer MLP with hidden dimensions (128, 128), and observation normalization is disabled for both the actor and critic. The critic is implemented as a separate MLP and is used solely during training; its weights are not included in the diffusion dataset and therefore have no effect on the weight generation.

Each actor network is trained for up to 4,500 iterations with 4,096 parallel environments. To capture the local structure of the converged weight manifold, we densely record checkpoints every 100 iterations after convergence and retain eight checkpoints per training seed for each motion. The complete training configuration is summarized in Table V.

## VIII. AUTOENCODER

## A. Training Details

During autoencoder training, we observed that operating directly on the raw actor weights yields lower reconstruction error than training on normalized weights. Consequently, all experiments reported in this work use unnormalized actor weights as the autoencoder input. To improve robustness and encourage smooth latent representations, we inject Gaussian noise into both the input weights and latent variables during training. This noise augmentation is applied only during training and is disabled at inference time. The autoencoder architecture and training hyperparameters are summarized in Table VI.

TABLE V: RL policy training configuration (locomotion: walking / motion-tracking: dancing, spin kick).
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Simulation</td><td>mjlab (MuJoCo)</td></tr><tr><td>Robot</td><td>Unitree G1</td></tr><tr><td>Control mode</td><td>Velocity-command locomotion / Motion tracking</td></tr><tr><td>RL algorithm</td><td>PPO (RSL-RL)</td></tr><tr><td>Parallel environments</td><td>4096</td></tr><tr><td>Max iterations</td><td>4500</td></tr><tr><td>Observation dimension</td><td>99 /  154</td></tr><tr><td>Actor hidden dimensions</td><td>(128, 128)</td></tr><tr><td>Action dimension</td><td>29</td></tr><tr><td>Observation normalization</td><td>Disabled</td></tr><tr><td>Dataset size</td><td>6000</td></tr></table>

TABLE VI: Autoencoder configuration: architecture and training hyperparameters for the walking, dancing, and spin kick motions.
<table><tr><td>Setting</td><td></td><td>Value</td></tr><tr><td>Arcure</td><td>Type Cluster input dims (Walking) Cluster input dims (Dancing, Spin kick) Input dimension L (Walking) Input dimension L (Dancing, Spin kick) Per-cluster latent dim</td><td>Two-cluster linear MLP 12800, 20253 19840, 20253 33053 40093 1024 2048</td></tr><tr><td>Trng</td><td>Optimizer Learning rate Weight decay Batch size Training steps Train / validation split Loss</td><td>1 (linear) AdamW  $1 \times 1 0 ^ { - 3 }$  (constant)  $5 \times 1 0 ^ { - 3 }$  512 1000 90%/10% MSE</td></tr></table>

## B. Ablation Study

1) Multi Cluster vs. Single Cluster: Our approach employs a cluster-wise autoencoder, where the first-layer weights and biases form one cluster, while the second- and third-layer weights and biases form another. To assess the impact of this design, we compare the proposed two-cluster autoencoder against a single-cluster variant that jointly encodes all network weights and biases. For a fair comparison, both models use the same latent dimensionality of 2048.

The average MSE reconstruction error across the entire motion dataset is reported in Table VII. Despite having identical latent capacity, the multi-cluster design consistently achieves lower reconstruction error. A plausible explanation is that the first-layer parameters generally have larger magnitudes than those of the deeper layers. When all parameters are encoded jointly, these differences in scale may complicate optimization and hinder accurate reconstruction. By partitioning parameters with distinct magnitude distributions into separate clusters, the cluster-wise autoencoder reduces this optimization difficulty and improves reconstruction accuracy.

TABLE VII: Cluster-structure ablation on three motions. MSE is the per-sample mean of reconstruction error on each dataset.
<table><tr><td>Motion</td><td>Config</td><td>MSE</td></tr><tr><td>Walking</td><td>2-cluster 1-cluster (n=2048)</td><td> $\mathbf { 1 . 8 7 \times 1 0 ^ { - 4 } }$   $1 . 0 5 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Spin kick</td><td>2-cluster 1-cluster (n=2048)</td><td> $\mathbf { 1 . 0 7 \times 1 0 ^ { - 4 } }$   $5 . 5 3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Dancing</td><td>2-cluster 1-cluster (n=2048)</td><td> $\mathbf { 6 . 3 6 \times 1 0 ^ { - 5 } }$   $3 . 1 0 \times 1 0 ^ { - 4 }$ </td></tr></table>

## IX. DIFFUSION TRANSFORMER

## A. Latent Normalization

The diffusion model operates entirely in the latent space of a pretrained autoencoder, which remains frozen throughout training. Each actor network is encoded by the clusterwise autoencoder and flattened into a 2048-dimensional latent representation. This dimensionality is independent of the original network weight dimension, enabling a unified diffusion architecture across all motions.

Prior to diffusion training, the latent representations are standardized using per-dimension z-score normalization. The normalized latents are partitioned into fixed-length patches and processed by a Diffusion Transformer (DiT). At inference time, latent samples are generated through the reverse diffusion process, denormalized, and decoded by the frozen autoencoder to recover complete actor network weights. Detailed architecture, training, and sampling configurations are provided in Table VIII

TABLE VIII: DiT configuration: architecture and training/sampling schedule.
<table><tr><td>Component</td><td></td><td>Value</td></tr><tr><td rowspan="10">Arure</td><td>Input dimension</td><td>2048</td></tr><tr><td>Sequence length</td><td>128</td></tr><tr><td>Patch size</td><td>16</td></tr><tr><td>Embedding dimension</td><td>1024</td></tr><tr><td>Transformer depth</td><td>8</td></tr><tr><td>Attention heads</td><td>8</td></tr><tr><td>Feed-forward expansion ratio</td><td>4</td></tr><tr><td>Activation function</td><td>GELU</td></tr><tr><td>Positional encoding</td><td>Sinusoidal</td></tr><tr><td>Timestep conditioning</td><td>adaLN-Zero</td></tr><tr><td></td><td>Dropout probability</td><td>0</td></tr><tr><td rowspan="10"></td><td>Generation setting</td><td>Unconditional</td></tr><tr><td>Denoising steps</td><td>1000</td></tr><tr><td>Noise schedule</td><td>Linear (10−4 →2 × 10−2)</td></tr><tr><td>Objective</td><td>x-prediction</td></tr><tr><td>Loss</td><td>MSE</td></tr><tr><td>Latent normalization</td><td>Per-dimension z-score</td></tr><tr><td>Optimizer</td><td>AdamW (2 × 10−4)</td></tr><tr><td>Training steps</td><td>100k</td></tr><tr><td>Batch size</td><td>256</td></tr><tr><td>Learning rate schedule</td><td>Warmup (10k) + cosine decay</td></tr><tr><td rowspan="5">Tri  Traing</td><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Exponential moving average</td><td>0.9999</td></tr><tr><td>Precision</td><td>bfloat16 mixed precision</td></tr><tr><td>Sampling</td><td>DDPM (EMA)</td></tr><tr><td>Inference steps</td><td>1000</td></tr></table>

## X. EXPERIMENTS

## A. DiT Memorization vs. Generalization

We provide additional t-SNE visualizations for walking and dancing motions, each containing 100 training network weights and 100 DiT-sampled weights, as shown in Fig. 8. The results show that the generated samples largely lie within the support of the training distribution, indicating that our method learns the data distribution.

## XI. WEIGHT SPACE VISUALIZATION

## A. Heatmaps for all motions

We provide heatmap visualizations for all three layers across different motions. For walking, the first-layer heatmap is presented in the main paper in Fig. 4, while here we additionally report the first-layer heatmaps for dancing and spin kick, including zoomed-in views of the joint pos and joint vel regions, as shown in Fig. 10. The full three-layer heatmaps across all motions are shown on the website. Across all heatmaps, we observe the same structural principle in terms of joints described in Section IV-A, which remains consistent across different motions.

## B. Subspace Similarity Among Motions

The first layer contains 128 hidden nodes, each defined by a weight vector over the input dimensions. Since all motions share joint pos (29) and joint vel (29) as inputs, each hidden node corresponds to a 58-dimensional weight vector associated with these two input groups. We analyze the subspaces spanned by the resulting 128 vectors across different motions. Prior work has studied whether hidden nodes across different networks are identical up to permutation via canonicalization methods [29]. In our datasets, we verify that the 128 vectors are not equivalent in this sense. Given the structural principle in Fig. 4, a natural question arises: do the first-layer subspaces show similarity across motions? To quantify subspace similarity, we adopt linear Centered Kernel Alignment (CKA) [30] to eliminate node permutation ambiguity between seeds. CKA values range from 0 (unrelated) to 1 (identical), and as a reference point, two independently drawn Kaiming-initialized [31] weight matrices reach a similarity of around 0.5.

As shown in Fig. 9, within motion seed pairs achieve CKA of 0.92–0.94, confirming that different seeds converge to nearly the same subspace within each motion. Cross motion pairs yield 0.69–0.82, which is lower. This indicates that the first layer is neither motion-agnostic nor motion-specific: a substantial shared knowledge is shaped by the G1 robot’s kinematics, while each motion introduces its own modulation within this shared subspace. Combined with the consistent structural principle observed earlier, these results suggest that the first layer weights primarily extract robot embodiment knowledge from the input and then adapt to the motion objective.

## C. RL and IL distillation for all motions

During distillation, a student MLP with the same architecture and activation functions is trained on the RL rollout buffer using mean-squared regression, without environment interaction or reward signals. Buffer collection details are provided in Table IX. The student is initialized from the same weights as its corresponding teacher, ensuring an identical starting point in weight space. This procedure is repeated independently for each teacher, yielding 50 teacher– student pairs per motion. The distilled IL networks are then evaluated in closed-loop simulation and achieve performance comparable to their RL teachers in both success rate (Succ)

![](images/b463cfadfc4ae9d80df4915426308fec2c19c2e63dda67b5fef93f19d3f0d952.jpg)  
(a) Walking

![](images/ad090d91ebe783118f9b8369c84c524d2bf5d090efbb5ee54777b871a345bb3b.jpg)  
(b) Dancing  
Fig. 8: t-SNE Visualization. Training, generated and deployed network weights for the walking and dancing motions.

TABLE IX: Distillation buffers collected from RL rollouts.
<table><tr><td>Buffer</td><td> $N _ { \mathrm { e n v } } \times T \times R$ </td><td>Samples</td><td>Obs</td></tr><tr><td>Walking</td><td> $1 0 2 4 \times 5 0 0 \times 2$ </td><td>1,024,000</td><td>99</td></tr><tr><td>Dancing</td><td> $5 1 2 \times 8 9 2 \times 2$ </td><td>913,408</td><td>154</td></tr><tr><td>Spin kick</td><td> $5 1 2 \times 2 3 2 \times 8$ </td><td>950,272</td><td>154</td></tr></table>

$N _ { \mathrm { e n v } }$ is the number of parallel environments, T the episode length in frames, and R the number of episodes collected per environment.

and tracking error (TE), showing functional equivalence.   
Training details are reported in Table X.

Having successfully confirmed their equivalence functionally, we provide heatmap visualizations for all motions across the three network layers to examine the differences between RL-trained and IL-trained networks. In addition to the walking results presented in the main paper, Fig. 11a and Fig. 11b present first-layer heatmaps for dancing and spin kick. Zoomed-in views of the joint pos and joint vel regions further highlight that the kinematic prior is consistent with those observed in walking. Complete visualizations covering all motions and all three layers are provided on the website.

![](images/f7c3605af47f660d82555c98f760ea71d2e1605f9fbfa361719264a19db6e2e3.jpg)  
Fig. 9: Subspace similarity across motions. Applying linear CKA to measure subspace similarity on shared inputs across motions, we observe near-perfect alignment within each motion, indicating stable knowledge extraction. Cross motion similarity also remains high, suggesting substantial shared knowledge with minor motion-specific adaptation.

TABLE X: Distillation configuration. Settings are shared across all three motions
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Objective</td><td>MSE</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $3 \times 1 0 ^ { - 4 }  1 \times 1 0 ^ { - 5 }$  (cosine)</td></tr><tr><td>Weight decay</td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Batch size</td><td>4096</td></tr><tr><td>Epochs</td><td>300</td></tr><tr><td>Train / val split</td><td>90/ 10 (seed 42)</td></tr></table>

## D. Details of domain randomization in RL algorithm

We define domain randomization (DR) as the randomization of physical and sensing parameters of the simulated system, quantities that are fixed properties of a given robot instance rather than part of its state. Under this definition, our environments randomize three components: the sliding friction of the foot-ground contact, the center-of-mass offset of the torso link, and a constant per-joint bias on the measured joint positions. These are summarized in Table XI. Each parameter is sampled once per parallel environment when the simulation is instantiated and held fixed for the duration of training, so that at any gradient step the batch spans the full parameter distribution while each individual environment remains dynamically consistent within an episode. Training without DR disables these components, and all other aspects of the environment are unchanged.

## XII. INPUT MEANING

To aid the analysis of weight-space distributions, we provide detailed definitions of the actor network inputs and outputs for each motion. The input dimensions of the walking motion are described in Table XII, whereas those of the dancing and spin kick (motion-tracking tasks) are summarized in Table XIII. For all motions, the network outputs correspond to the 29 joints of the G1 robot, whose definitions are listed in Table XIV and Table XV.

(a) Dancing  
![](images/91cbaf6467e3390e86c0c9d9ff9986553630edbb8bc84fde84d63861bfb5005a.jpg)  
Fig. 10: First-layer heat maps. (a) Dancing weight heat map. (b) Spin kick weight heat map. The first-layer weights consistently exhibit the structural principle across joints.

![](images/7400a7c43b804da5ff9943a40e760780e9cb9821f684879685cb2abf643bb04f.jpg)  
Fig. 11a: First-layer weight comparison between RL and IL for dancing.

![](images/711cc4edddbbde58f116fbe9ddd8543529ceb8f9937dd7e47f809853c4889cfc.jpg)  
Fig. 11b: First-layer weight comparison between RL and IL for spin kick.

TABLE XI: Domain randomization terms
<table><tr><td>Term</td><td>Randomized quantity</td><td>Walking</td><td>Dancing</td><td>Sampling</td></tr><tr><td>Foot friction</td><td>Sliding friction coeff. of foot geoms</td><td>U(0.3, 1.2)</td><td>U(0.3, 1.2)</td><td>per env</td></tr><tr><td>Torso CoM offset</td><td>CoM position of torso link, per axis</td><td>±(2.5, 2.5, 3.0) cm</td><td>±(2.5, 5.0, 5.0) cm</td><td>per env, per axis</td></tr><tr><td>Encoder bias</td><td>Constant offset on measured joint pos.</td><td>±0.015 rad</td><td>±0.010 rad</td><td>per env, per joint</td></tr></table>

TABLE XII: Walking actor observation, 99-d, in concatenation order. Joint index k refers to Table XV.
<table><tr><td>Block (dim) — meaning</td><td>Per-dimension content</td></tr><tr><td>base_lin_vel (3) — base linear velocity (IMU)</td><td> $\left[ 0 \right] { \boldsymbol v } _ { x } , \left[ 1 \right] { \boldsymbol v } _ { y } , \left[ 2 \right] { \boldsymbol v } _ { z }$  in the base frame</td></tr><tr><td>base_ang-vel (3) — base angular velocity (IMU)</td><td> $\left[ 0 \right] \omega _ { x } , \left[ 1 \right] \omega _ { y } , \left[ 2 \right] \omega _ { z }$ </td></tr><tr><td>gravity direction</td><td>projected_gravity (3) — unit gravity vector projected into the base frame</td></tr><tr><td>joint-pos (29) — joint an- dim k = qk −</td><td> $q _ { k } ^ { \mathrm { d e f a u l t } }$   $k = 0 \ldots 2 8$ </td></tr><tr><td>gles rel. default joint_vel (29) — joint an- dim</td><td> $k = \dot { q } _ { k }$ </td></tr><tr><td>gular velocities last_action (29) — previ- dim k = previous raw action of joint k</td><td></td></tr><tr><td>ous action</td><td></td></tr><tr><td>mand (twist)</td><td>command (3) — velocity com- [0] vx, [1] vy, [2] ωz (target base twist)</td></tr></table>

TABLE XIII: Dancing and Spin kick actor observation, 154-d, in concatenation order. Joint index k refers to Table XV.  
Block (dim) — meaning Per-dimension content   
command (58) — reference [0 : 29] target joint angles $q _ { k } ^ { * } ; [ 2 9 : 5 8 ]$   
motion target target joint velocities $\dot { q } _ { k } ^ { * }$ (same joint   
order)   
anchor ori b (6) — anchor first two columns of the rotation matrix   
relative orientation (6D rot.) of the reference anchor frame w.r.t. the   
robot anchor frame: [0 : 3] col. 1, [3 : 6]   
col. 2   
base ang vel (3) — base $\left[ 0 \right] \omega _ { x } , \left[ 1 \right] \omega _ { y } , \left[ 2 \right] \omega _ { z }$   
angular velocity (IMU)   
joint pos (29) — joint an- dim $k = q _ { k } - q _ { k } ^ { \mathrm { d e f a u l t } }$   
gles rel. default   
joint vel (29) — joint an- dim k = ˙q<sub>k</sub>   
gular velocities   
last action (29) — previ- dim k = previous raw action of joint k   
ous action

TABLE XIV: Actor network output, 29-d, identical across all motions. Joint index k refers to Table XV.
<table><tr><td>Block (dim) — meaning</td><td>Per-dimension content</td></tr><tr><td>action (29) — joint position targets</td><td>dim k is  $^ { a } k ,$  mapped to the target angle  $q _ { k } ^ { \mathrm { t g t } } ~ = ~ q _ { k } ^ { \mathrm { d e f a u l t } }$  + scale · ak (G1_ACTION_SCALE), tracked by the joint PD controller</td></tr></table>

TABLE XV: The 29 actuated joints of the Unitree G1 in model order, corresponding to the ordering of the joint pos, joint vel, and last action observations and the 29-dimensional action output. Joints are grouped by body region and ordered from left to right and proximal to distal.
<table><tr><td rowspan=1 colspan=10>Region Side Joint                Range [rad]</td></tr><tr><td rowspan=1 colspan=9>hip-pitch</td><td rowspan=1 colspan=1>[-2.53, 2.88]</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=2 colspan=7>left  hip-yawkneeankle-pitchlegs           ankle_roll</td></tr><tr><td></td><td></td><td></td><td rowspan=1 colspan=1>-2.76, 2.76]-0.09, 2.88]–0.87, 0.52][−0.26, 0.26]</td></tr><tr><td rowspan=3 colspan=9>hip-pitchhip-rollright hip-yawkneeankle-pitchankle_roll</td><td rowspan=1 colspan=1>-2.53, 2.88]</td></tr><tr><td rowspan=1 colspan=1>-2.76, 2.76–0.09, 2.88]</td></tr><tr><td rowspan=1 colspan=1>–0.87, 0.52]-0.26,0.26]</td></tr><tr><td rowspan=2 colspan=9>yaw</td><td rowspan=2 colspan=1>-2.62,2.62]-0.52,0.52]</td></tr><tr><td rowspan=2 colspan=9>pitch</td></tr><tr><td rowspan=1 colspan=1>[−0.52,0.52]</td></tr><tr><td rowspan=1 colspan=9>shoulder-pitch</td><td rowspan=1 colspan=1>-3.09,2.67]</td></tr><tr><td rowspan=4 colspan=9>shoulder_rollshoulder-yawelbowwrist_rollwrist-pitcharms          wrist-yaw</td><td rowspan=4 colspan=1>-1.59, 2.25]-2.62, 2.62]-1.05, 2.09]-1.97, 1.97]-1.61, 1.61][−1.61, 1.61]</td></tr><tr><td rowspan=1 colspan=6></td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=5></td></tr><tr><td rowspan=1 colspan=9>shoulder-pitchshoulder_rollshoulder-yawright elbow</td><td rowspan=1 colspan=1>-3.09, 2.67]-2.25, 1.59-2.62, 2.62]-1.05, 2.09]</td></tr><tr><td rowspan=1 colspan=7>wrist_roll</td><td></td><td></td><td rowspan=1 colspan=1>-1.97, 1.97]</td></tr><tr><td rowspan=1 colspan=7>wrist-pitch</td><td></td><td></td><td rowspan=1 colspan=1>-1.61, 1.61]</td></tr><tr><td rowspan=1 colspan=9>wrist-yaw</td><td rowspan=1 colspan=1>[−1.61, 1.61]</td></tr></table>