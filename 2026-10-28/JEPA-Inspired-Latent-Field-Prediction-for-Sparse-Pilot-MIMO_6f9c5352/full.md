# JEPA-Inspired Latent Field Prediction for Sparse-Pilot MIMO-OFDM Channel Reconstruction

Yuzhi Yang<sup>∗†</sup>, Brahim Mefgouda<sup>∗</sup>, Hang Zou<sup>∗</sup>, Lina Bariah<sup>∗</sup>, Anis Bara<sup>∗</sup>,

Yuhuan Lu,<sup>‡</sup>, Hao Zhang<sup>∗</sup>, and Merouane Debbah ´ <sup>∗</sup>

<sup>∗</sup>Research Institute for Digital Future, Khalifa University, 127788 Abu Dhabi, UAE

<sup>†</sup>School of Information Science, Japan Advanced Institute of Science and Technology, Nomi, Ishikawa, Japan

<sup>‡</sup>the Faculty of Applied Sciences, Macao Polytechnic University, Macao SAR, China

E-mail: yangyzis@jaist.ac.jp

Abstract—Accurate acquisition of full channel state information (CSI) is essential for multiple-input multiple-output orthogonal frequency-division multiplexing (MIMO-OFDM) wireless systems, while dense pilot-based estimation incurs substantial overhead. Directly minimizing the prediction error over raw CSI coefficients is fragile in realistic propagation environments, since small-scale fading, mobility, local scattering, and small spatial perturbations can induce large phase-sensitive variations. Nevertheless, wireless channels contain strong spatial, temporal, and frequency-domain correlations that can be more predictably represented in a latent propagation-field domain. Motivated by this observation, we propose a joint-embedding predictive architecture (JEPA)-inspired field world model, defined as a latent predictive model of the wireless propagation field, for sparse-pilot channel reconstruction. The model maps CSI observations into latent time-frequency tokens, predicts masked or future latent field states from historical multi-band context, and fuses the predicted latent prior with sparse noisy current pilots to reconstruct the full MIMO-OFDM channel. Ray-tracing experiments show that the proposed method improves normalized mean square error (NMSE) over pilot-only and direct reconstruction baselines, supporting latent field prediction as an effective prior for sparsepilot channel acquisition.

Index Terms—MIMO-OFDM, CSI reconstruction, latent field prediction, JEPA

## I. INTRODUCTION

In multiple-input multiple-output orthogonal frequencydivision multiplexing (MIMO-OFDM) wireless systems, accurate channel state information (CSI) is essential for coherent detection, beamforming, precoding, and resource allocation. In practical systems, CSI is usually acquired with pilot signals inserted over the time-frequency resource grid. However, when the channel varies across time slots and frequency bands, repeatedly transmitted pilots may consume considerable spectral resources. This motivates pilot-efficient channel acquisition methods that exploit the correlation among channel observations to reconstruct the full CSI from sparse current pilots.

However, direct channel prediction is fragile in realistic wireless environments. The observed channel response contains both relatively stable propagation structures and rapidly varying small-scale fading components, while mobility, local scattering, and small spatial perturbations can induce large phase-sensitive variations. As a result, it becomes fragile to generate satisfactory raw CSI coefficients with simple prediction methods. Nevertheless, the underlying propagation field contains strong spatial, temporal, and frequency-domain correlations, which introduce an inherent strong correlation between resource blocks. This suggests that prediction should be performed in a suitable latent propagation-field domain and used as a structured prior, rather than treated as a complete replacement for pilot-based channel estimation.

Time- and frequency-domain channel prediction has been widely studied as an effective way to acquire future channel state information with reduced signaling overhead. Classical model-based methods can exploit temporal correlation, whereas their performance strongly depends on the accuracy of the assumed channel dynamics. Meanwhile, recent learningbased methods learn the mapping from historical CSI to future CSI directly from the data, which usually outperforms classical methods. As summarized in [1], recurrent networks, transformer-based predictors, large language model-based predictors, and wireless foundation models have been explored to capture richer space-time-frequency correlations of wireless channels. These studies demonstrate the potential of deep learning for channel prediction. However, many predictionoriented methods still operate directly on raw CSI coefficients, without explicitly separating predictable propagationfield structure from phase-sensitive instantaneous fluctuations or using predicted latent states as priors for sparse-pilot fullchannel reconstruction.

Meanwhile, world models have recently emerged as a promising paradigm for learning compact representations of complex dynamical systems [2]. In particular, their success in video prediction and reconstruction shows that highdimensional observations often contain latent physical structures that can be predicted more effectively in an abstract representation space than directly in the raw observation space [2], [3]. Instead of capturing pixel-level details, a world model aims to capture the underlying state and its temporal evolution, enabling prediction, completion, and reconstruction from partial observations. This perspective is naturally relevant to wireless communications. Although instantaneous small-scale fading may fluctuate rapidly, channel evolution is still governed by hidden environmental dynamics. Therefore, a world model can be used to learn a latent channel representation that emphasizes the predictable large-scale relationship across time and frequency, while leaving fine-grained signal details to be recovered with the aid of sparse raw observations [4]. Recent wireless JEPA studies [5] have explored latent dynamics for channel representation and charting, whereas sparse-pilot fullchannel reconstruction with predicted latent priors remains less explored.

In this work, we further adopt dense predictive supervision and deep self-supervision over multiple intermediate vision transformer (ViT) layers [6]. Dense predictive supervision constrains the predicted target field at the token level rather than only through a global latent vector, which is important for preserving local subcarrier-antenna and time-frequency structure. Intermediate-layer self-supervision encourages the model to learn propagation-field representations at multiple abstraction levels, improving the stability of latent prediction. Therefore, the proposed field world model predicts structured latent propagation states rather than raw CSI coefficients, and the predicted latent field is later used as a prior for sparse-pilot channel reconstruction.

Motivated by the above observation, we propose a jointembedding predictive architecture (JEPA)-inspired field world model (FWM) for sparse-pilot MIMO-OFDM channel reconstruction. Here, FWM denotes a latent predictive model of the wireless propagation field. Unlike methods that directly extrapolate raw CSI coefficients, the proposed framework maps CSI observations into latent time-frequency tokens and predicts masked or future latent field states from historical multi-band context. The predicted latent field is not required to preserve all instantaneous CSI details. Instead, it is designed to capture stable and predictable propagation structure. For downstream reconstruction, this predicted latent state is used as a structured prior and fused with sparse noisy current pilots to recover the full MIMO-OFDM channel.

The main contributions of this paper are summarized as follows:

• We formulate sparse-pilot MIMO-OFDM channel reconstruction as a latent-field-aided reconstruction problem, where predictable propagation-field states are learned from historical multi-band time-frequency CSI and used as priors for current channel recovery.

• We develop a JEPA-inspired field world model that predicts masked or future channel states in a compact latent propagation-field domain, avoiding direct coefficientlevel extrapolation of phase-sensitive raw CSI.

• We design a pilot-assisted reconstruction framework that fuses the predicted latent field prior with sparse noisy current pilots to recover the full channel. The numerical results also show that overly detail-preserving auxiliary objectives can reduce downstream reconstruction usefulness.

## II. PROPOSED METHOD

## A. System Overview

In this paper, we propose a JEPA-based FWM for timefrequency channel prediction and pilot-assisted channel reconstruction. The key idea is to treat each subband-time CSI observation as a cell in a time-frequency propagation field. Instead of directly extrapolating raw CSI coefficients, the proposed field world model maps each CSI cell into a latent token representation and performs prediction in the latent propagation-field domain. The predicted latent state is then used as a structured prior for downstream reconstruction from sparse current pilots.

The overall architecture consists of three modules. First, a CSI tokenizer converts each CSI cell into a group of latent tokens. Second, a JEPA-style two-stage ViT backbone predicts the latent representations of masked or future cells from visible historical cells. Finally, for downstream channel estimation, the predicted latent prior is combined with sparse pilot observations and fed into a downstream head to reconstruct the full CSI.

## B. JEPA-Based Time-Frequency Latent Prediction

The JEPA pretraining pipeline is generally depicted in Fig. 1, which considers a typical channel latent prediction problem in the time and frequency fields. Let $H _ { n , t } \in \mathbb { C } ^ { N _ { \mathrm { c } } \times N _ { \mathrm { a } } }$ denote the complex CSI matrix at subband n and time slot t, where $N _ { \mathrm { c } }$ is the number of subcarriers and $N _ { \mathrm { a } }$ is the number of antenna elements. The full channel trajectory is represented as $\mathcal { H } \in \mathbb { C } ^ { T \times N _ { \mathrm { b } } \times N _ { \mathrm { c } } \times N _ { \mathrm { a } } }$ , where T and $N _ { \mathrm { b } }$ denote the numbers of time slots and subbands, respectively. We define $\mathcal { A }$ as the set of all time-subband cells, R as the set of visible cells, and $\mathcal { M } = \mathcal { A } \backslash \mathcal { R }$ as the set of masked cells to be predicted. Each visible CSI cell is mapped into K latent tokens of dimension D by a tokenizer

$$
L _ { n , t } = f _ { \mathrm { t o k } } ( H _ { n , t } ; \theta ) \in \mathbb { R } ^ { K \times D } , \quad ( n , t ) \in \mathcal { R } .\tag{1}
$$

The latent tokens are organized as a time-frequency field. We encode the global time-subband coordinates using twodimensional positional encoding, while the local index of each token within a cell is represented by a learnable embedding. In each training sample, all cells at the final time slot T are masked to simulate the next-step channel-field prediction. Historical cells are further randomly masked to encourage completion from incomplete time-frequency context, corresponding to the case where not all previous resource blocks are available for the target user.

Following the JEPA-style prediction paradigm [6], the backbone contains a context encoder and a latent predictor. The context encoder processes only the visible cells:

$$
\mathcal { Z } _ { \mathcal { R } } = f _ { \mathrm { e n c } } \left( \{ L _ { n , t } \} _ { ( n , t ) \in \mathcal { R } } , \{ ( n , t ) \} _ { ( n , t ) \in \mathcal { R } } ; \phi _ { \mathrm { e n c } } \right) ,\tag{2}
$$

where $\mathcal { Z } _ { \mathcal { R } }$ denotes the contextualized latent embeddings of the observed time-frequency cells. The masked cells are not provided to the encoder.

For each masked cell $( n , t ) \in { \mathcal { M } }$ , we introduce a learnable query token group $Q _ { n , t } \ \in \ \bar { \mathbb { R } } ^ { K \times D _ { p } }$ . The predictor takes the projected context embeddings and all query tokens as input:

$$
\begin{array} { r l } & { Y = f _ { \mathrm { p r e d } } \big ( \big [ g _ { \mathrm { i n } } ( \mathcal { Z } _ { \mathcal { R } } ) , \{ Q _ { n , t } \} _ { ( n , t ) \in \mathcal { M } } \big ] , } \\ & { \qquad \{ ( n , t ) \} _ { ( n , t ) \in \mathcal { R } \cup \mathcal { M } } ; \phi _ { \mathrm { p r e d } } \big ) . } \end{array}\tag{3}
$$

The output tokens corresponding to the masked cells are then mapped back to the tokenizer latent space through a linear layer:

$$
\begin{array} { r } { \hat { L } _ { n , t } = f _ { \mathrm { o u t } } ( Y _ { n , t } ) \in \mathbb { R } ^ { K \times D } , \quad ( n , t ) \in \mathcal { M } . } \end{array}\tag{4}
$$

![](images/86422baf47927e65424bb9d84e317c9e79e2dcf9b565895716a6a6492380c764.jpg)  
Fig. 1. Overview of the time-subband prediction JEPA pipeline.

![](images/39078d32fe445d3bec069bbcead84893b0e13ce7d11c8e10802c6ce20bc59a86.jpg)  
Fig. 2. Pilot-aware channel prediction and reconstruction structure.

In this way, the model learns to infer the latent state of missing or future CSI cells from the observed time-frequency context, rather than extrapolating raw channel coefficients directly.

## C. Self-Supervised Training Objective

The model is trained by masked latent prediction. To provide stable targets, we use a momentum tokenizer with parameters <sup>¯</sup>θ, which has the same architecture as the online tokenizer. For each masked cell, the target latent is

$$
L _ { n , t } ^ { \mathrm { t a r } } = f _ { \mathrm { t o k } } ( H _ { n , t } ; \bar { \theta } ) , \quad ( n , t ) \in \mathcal { M } .\tag{5}
$$

The momentum tokenizer is updated by exponential moving average:

$$
\bar { \theta }  \mu \bar { \theta } + ( 1 - \mu ) \theta ,\tag{6}
$$

where $\mu$ is the momentum coefficient. The main JEPA loss is defined as

$$
\ell _ { \mathrm { J E P A } } = \frac { 1 } { | \mathcal { M } | } \sum _ { ( n , t ) \in \mathcal { M } } \frac { 1 } { K D } \left\| \hat { L } _ { n , t } - L _ { n , t } ^ { \mathrm { t a r } } \right\| _ { 1 } .\tag{7}
$$

A pure latent prediction objective may lead to representation collapse, where all CSI observations are mapped to nearly constant latent codes [5]. To avoid this, we introduce a regularization to encourage statistical diversity of the latent field. Specifically, we use a covariance loss to reduce redundancy among latent dimensions

$$
\ell _ { \mathrm { c o v } } = \left\| C - \mathrm { d i a g } ( C ) \right\| _ { F } ^ { 2 } ,\tag{8}
$$

where C is the covariance matrix of the flattened latent features over the current batch.

In addition to the main JEPA-covariance objective, we evaluate several CSI-domain grounding objectives as ablations, including direct CSI recovery, amplitude recovery, and conditional diffusion grounding. These ablations are used to examine whether forcing the latent representation to preserve more CSI-domain details improves or degrades downstream reconstruction. Since diffusion grounding predicts injected noise from a corrupted CSI sample, it imposes a weaker deterministic recovery constraint on the latent code than direct CSI recovery. Among the considered grounding objectives, direct CSI recovery imposes the strongest detail-preservation constraint, while diffusion grounding is the weakest. These objectives share the generic form

$$
\ell _ { \mathrm { a u x } } = \frac { 1 } { | \mathcal { M } | } \sum _ { ( n , t ) \in \mathcal { M } } \left\| f _ { \mathrm { a u x } } \left( U _ { n , t } ^ { \mathrm { a u x } } ; \psi \right) - V _ { n , t } ^ { \mathrm { a u x } } \right\| _ { 2 } ^ { 2 } ,\tag{9}
$$

where $( U _ { n , t } ^ { \mathrm { g } } , V _ { n , t } ^ { \mathrm { g } } )$ specify the input and target of each grounding variant, and ψ denotes the parameters in the auxiliary method which is trained alongside other trainable parameters. Direct CSI recovery uses the latent code to recover the normalized CSI, such that $U _ { n , t } ^ { \mathrm { a u x } } = L _ { n , t }$ and $V _ { n , t } ^ { \mathrm { a u x } } = H _ { n , t }$ . The amplitude recovery replaces the target by channel amplitude, such that $U _ { n , t } ^ { \mathrm { a u x } } ~ = ~ L _ { n , t }$ and $V _ { n , t } ^ { \mathrm { a u x } } ~ = ~ \mathrm { A m p } ( H _ { n , t } )$ , where $\operatorname { A m p } ( H )$ denotes the element-wise amplitude (absolute value) of H. The diffusion grounding uses the latent code together with a noisy CSI sample and diffusion step to predict the injected noise, such that ${ \cal U } _ { n , t } ^ { \mathrm { a u x } } \ = \ ( \bar { H } _ { n , t } ^ { ( \tau ) } , L _ { n , t } , \tau ) , \ \bar { H } _ { n , t } ^ { ( \tau ) } \ =$ $\alpha _ { \tau } H _ { n , t } + \sqrt { 1 - \alpha _ { \tau } ^ { 2 } } \epsilon _ { n , t }$ with a random Gaussian white noise $\epsilon _ { n , t }$ , and $V _ { n , t } ^ { \mathrm { a u x } } ~ = ~ \epsilon _ { n , t }$ . The diffusion-based realization is similar to that in [3].

The overall pretraining objective is thus

$$
\ell _ { \mathrm { p r e } } = \ell _ { \mathrm { J E P A } } + \lambda _ { \mathrm { c o v } } \ell _ { \mathrm { c o v } } + \lambda _ { \mathrm { g } } \ell _ { \mathrm { g } } .\tag{10}
$$

The final model uses $\ell _ { \mathrm { J E P A } } + \lambda _ { \mathrm { c o v } } \ell _ { \mathrm { c o v } }$ , while $\ell _ { \mathrm { g } }$ is enabled only in ablation studies.

## III. PILOT-ASSISTED CHANNEL RECONSTRUCTION WITH LATENT PRIORS

The JEPA backbone provides a structured latent representation of the current wireless field. However, the predicted latent field is not intended to replace current pilot observations. Instead, it provides a structured prior that captures predictable propagation information from historical time-frequency context. We therefore evaluate the learned FWM through a pilotassisted channel reconstruction task, where the current full CSI is recovered from sparse noisy pilots containing small-scale information conditioned on the predicted latent state.

Based on the JEPA latent representation and its prediction $\hat { L } _ { n , t }$ from historical channel observations $H _ { n , t } ~ \in ~ \mathbb { C } ^ { N _ { \mathrm { c } } \times N _ { \mathrm { a } } }$ over previous time steps through (1)-(4), we formulate a reconstruction task for the current channel $H _ { n , T }$

Meanwhile, the receiver obtains an initial rough and partial estimation of the current channel based on pilots. This estimation is sparse and noisy, and can be modeled as

$$
\tilde { H } _ { n , t } ^ { ( \mathrm { e s t } ) } = P _ { n , t } \odot ( H _ { n , t } + N _ { n , t } ) ,\tag{11}
$$

where $P _ { n , t }$ is a binary pilot mask, $N _ { n , t }$ denotes the estimation noise, and ⊙ denotes element-wise multiplication. We note that we use the direct tokenizer output $H _ { n , t }$ corresponding to all resource blocks without passing into the JEPA predictor during training.

The final reconstruction is then produced by a latentconditioned direct regressor that combines the predictive latent prior with the estimation input

$$
\hat { H } _ { n , t } = g _ { \mathrm { r e c } } \left( \tilde { H } _ { n , t } ^ { ( \mathrm { e s t } ) } , \hat { L } _ { n , t } \right) .\tag{12}
$$

In this way, $\hat { L } _ { n , t }$ provides a latent-domain prior, while ${ \tilde { H } } _ { n , t } ^ { ( \mathrm { e s t } ) }$ preserves instantaneous measurement evidence from the pilotbased estimation. During downstream training, the reconstruction network is trained with teacher-forced latent conditions obtained by directly tokenizing the current CSI using the frozen tokenizer. During evaluation, these teacher-forced latent conditions are replaced by the predicted latent states produced by the FWM from historical visible cells. This strategy increases the number and diversity of supervised reconstruction pairs and decouples the learning of pilot-latent fusion from latent prediction. Especially, no current full CSI is used at inference.

The downstream model is trained by direct regression with

$$
\ell _ { \mathrm { r e c } } = \mathbb { E } \left[ \left\| { \hat { H } } _ { n , t } - H _ { n , t } \right\| _ { 2 } ^ { 2 } \right] .\tag{13}
$$

In the current implementation, the binary mask is not provided as a separate input. Since the sparse pilot observations are zero-filled according to the stride-based pilot pattern, the observed positions are implicitly encoded in the input. The final recovery result is therefore obtained in a single forward pass from the historical latent prior and the initial channel estimation. During downstream training, the pretrained tokenizer and latent predictor are frozen, and only the reconstruction network $g _ { \mathrm { r e c } }$ is optimized. The downstream network is trained with an MSE loss, while reconstruction quality is reported using NMSE.

![](images/deb411416ff23893f4470e5aabe157bd0eb76b71a3e6291e42c75e7f080cfd0b.jpg)  
Fig. 3. The scenario considered for generating dataset based on the actual maps in Chicago within a range of around 1 km<sup>2</sup>. The red point indicates the BSs and blue lines are UE paths.

TABLE I  
DATASET AND SYSTEM CONFIGURATION.
<table><tr><td>Item</td><td>Configuration</td></tr><tr><td>3.5 GHz subbands 3.5 GHz subcarrier spacing</td><td>3.1, 3.3, 3.5, 3.7, 3.9 GHz 480 kHz</td></tr><tr><td>28 GHz subbands</td><td>27, 28, 29 GHz</td></tr><tr><td>28 GHz subcarrier spacing Subcarriers per subband</td><td>7.5 MHz</td></tr><tr><td></td><td>32</td></tr><tr><td>BS antennas</td><td>Two co-located 4 × 4 arrays</td></tr></table>

## IV. NUMERICAL RESULTS

## A. Dataset Generation

We generate the dataset using Sionna [7] over the citycenter scenario of Chicago. There are totally 3 BSs located on the top of the buildings. We randomize 200 trajectories for each BS. The starting and ending points are located on the ground within a range of horizontal distances of 100 m from the corresponding BS. Each trajectory contains 16 points which are almost uniformly distributed in a line. In particular, the first and last points are located with a distance within 1- 10 meters, and all other points are uniformly located between them. The velocities are calculated accordingly such that the time difference between each adjacent points is 50 ms. After that, each UE position is perturbed by a Gaussian shift with standard deviation 5 cm to imitate actual moving and avoid unrealistically regular phase evolution. We consider a total of 8 different subbands, whose details are shown in Table I. Therefore, each channel realization is represented as a complex tensor of dimension $1 6 \times 8 \times 3 2 \times 1 6 .$ corresponding to time samples, subbands, subcarriers, and antennas. We first split complete UE trajectories into disjoint training and testing sets at the trajectory level. All temporal slices are then generated within each split independently, ensuring that no trajectory contributes samples to both training and testing.

## B. Visibility Mask Generation

We employ a progressive history masking strategy during JEPA training. Random masking is applied only to historical time steps, whereas the final time step is always fully masked so that the predictor must infer the target state entirely from preceding observations. To begin with, we use a fixed masking ratio at the beginning of training, as the JEPA tokenizer is not powerful enough at that time. In addition, we gradually apply a stochastic target distribution throughout the training procedure.

Specifically, for epoch e, the masking rate is linearly interpolated between 0.3 and a sample drawn from a Beta distribution over [0.1, 0.9], namely

$$
p ^ { ( e ) } = ( 1 - \lambda _ { e } ) \cdot 0 . 3 + \lambda _ { e } \cdot ( 0 . 1 + 0 . 8 u ) , u \sim \mathrm { B e t a } ( 4 . 0 , 1 . 2 ) ,\tag{14}
$$

where $\lambda _ { e }$ increases linearly from 0 to 1. This design progressively increases task difficulty while preserving full masking of the final time step throughout pretraining.

## C. Network Topology and Training Settings

In implementation, each complex CSI cell is represented by stacking its real and imaginary components as two channels. For the main JEPA pretraining task, we adopt a CNNstyle tokenizer head and a two-layer ViT-style predictor. Specifically, the tokenizer starts with a $3 \times 3$ convolutional stem that maps the two-channel input to 32 channels. It is followed by 4 residual CNN blocks with channel transitions $3 2  6 4  1 2 8  1 2 8  1 2 8 .$ . A 2× stride is applied to the second residual module, while others do not contain downsampling. Afterwards, each token map is then flattened and projected by a 2-layer MLP to a 192-dimensional token embedding, and a layer normalization layer is used to avoid collapsing.

The first module of the ViT-style predictor is composed of 4 transformer layers with 8 attention heads, following by a linear layer mapping the latent dimension to 256. Then, a shallow ViT-style stack with 2 transformer layers with 4 attention heads conducts the prediction, followed by a two-layer MLP network mapping them back to the 192-dimensional latent space with 2 independent tokens.

For the auxiliary regularization heads, we use the following simple modules. The masked recovery head is a CNN decoder symmetric to the tokenizer head. Meanwhile, the amplitude recovery branch uses exactly the same decoder and differs only in the training loss, which is applied to the recovered amplitude instead of the complex-valued output itself, allowing phase shifting. The diffusion regularization head is a lightweight token-conditional DiT-style network with hidden dimension 128, depth 3, and 8 attention heads, trained with a standard 1000-step noise schedule.

For the downstream model, we use a latent-conditioned direct regressor. The initial rough channel estimation is first embedded by a $1 \times 1$ convolution into a 256-dimensional feature map on the $3 2 \times 1 6$ grid. In parallel, the predicted latent prior is projected to the same hidden dimension and used as the conditioning signal. The reconstruction network then applies 6 transformer-style blocks, each consisting of self-attention, cross-attention to the latent prior, and a feed-forward layer, followed by a linear projection to the output recovery.

![](images/ca28d88859257c402c5c06bb4b39334a1f264fda53488bfb0e7b3c332293d81c.jpg)  
Fig. 4. Pretraining results for the JEPA model.

For each subband, pilot positions are sampled with a stride of 8 and a random starting offset. Additive Gaussian noise is applied only to the observed pilot entries, with the noise power sampled uniformly in NMSE range [0, 1].

All models are trained with batch size 32, weight decay $1 0 ^ { - 4 }$ . The learning rate increases linearly from 0 to $1 0 ^ { - 4 }$ in the first 10% of training, and gradually falls to 0 with a cosine scheduler during the following training. We use $\mu = 0 . 9 5$ for the exponential moving average between teacher and student models during JEPA training.

## D. Baseline Methods

To better evaluate the role of predictive latent modeling, we consider the four baseline methods below.

• Simple Prediction: Here, we consider a simple prediction network. We directly use the past channel matrices to recover the next-step channels. It is a non-JEPA baseline that only performs prediction without current estimation.

• Simple Estimation: It is another baseline that only uses the current pilot-based sparse and noisy estimation to recover the target channel without past information.

• Direct Training: It is an ablation method for the JEPA double-stage train scheme. Instead, we directly train the overall network with the final reconstruction target for 400 epochs.

• Current-Latent Reconstruction: It is an ideal upperbound method which is unrealistic in practice. Instead of using predicted latents, we directly map the current channel to the latent domain and input it into the same downstream network. This is to show the upper bound of the investigated scheme with perfect prediction.

## E. Effect of Different Regularization Items in Pretraining

Fig. 4 shows the pretraining results under different regularization items with the same network structure. Although the

![](images/7ef30baf5b03b5728a10ada27a1c9932ae1d8d0ec412bbaf1b2c123a934767d2.jpg)  
Fig. 5. Downstream channel reconstruction performances of different methods. We note that the simple prediction experiment fails to converge.

JEPA objective is optimized with an L-1 latent prediction loss, we report NMSE in the latent prediction evaluation because it is more directly aligned with downstream wireless reconstruction quality. Here $\lambda _ { \mathrm { g } }$ takes value of 0.2 when enabled.

A moderate covariance regularization is clearly necessary, and the best result is obtained with $\lambda _ { \mathrm { c o v } } = 0 . 1$ . In contrast, among the three recovery-oriented regularizations, the performance degrades from diffusion to amplitude recovery and then to direct recovery, indicating that finer detail restoration makes the JEPA prediction task harder.

The three grounding variants impose different levels of CSI-detail preservation. Direct CSI recovery is the strongest constraint because it encourages the latent code to preserve full complex channel coefficients. Amplitude recovery relaxes the phase requirement but still constrains channel-strength details. Diffusion grounding is weaker because the denoising objective does not require the latent code to deterministically reconstruct all CSI coefficients. However, among the grounding variants, diffusion grounding is the least harmful because the denoising target can be partially solved from the corrupted sample itself, so the latent condition is not forced to encode all fine-grained CSI details.

This is also consistent with our dataset, since phase-level prediction is inherently difficult due to the position shift. More importantly, latent prediction quality should be interpreted together with downstream reconstruction. Poor JEPA prediction NMSE indicates weak predictive representations, whereas good JEPA prediction NMSE is only a necessary, not sufficient, condition for downstream gains.

## F. Channel Prediction and Reconstruction Results

Fig. 5 compares the downstream reconstruction NMSE under different noise levels for different JEPA and baseline methods. We first find that the current-latent reconstruction method are exceptionally good while close to each other. Thus, the quality of learned latents are similar, and the difference mainly lies in the JEPA prediction accuracy. Among the predictionaided results, using only “cov” and $\mathrm { ^ { 6 6 } c o v + d i f f ^ { 3 } }$ yield the best reconstruction performance across the full noise range, which performs similarly. In contrast, adding amplitude recovery or direct recovery consistently increases the reconstruction NMSE. This indicates that recovery-oriented auxiliary regularization does not improve the downstream usefulness of the learned JEPA representation, and even sometimes harms the pilot-based reconstruction method. Instead, the most effective choice is a predictive objective regularized only by a moderate covariance term without any recovery-aimed regularization due to the good JEPA prediction results..

Meanwhile, the simple prediction baseline fails to converge due to the random disturbance in the locations. Moreover, the comparisons to the baselines shows that the best JEPAbased method clearly outperforms both simple estimation and direct training baselines, demonstrating the usefulness of the integrated prediction and estimation scheme and the training method.

## V. CONCLUSION

In this paper, we proposed a JEPA-inspired FWM for sparsepilot MIMO-OFDM channel reconstruction. Rather than directly predicting raw MIMO-OFDM channel coefficients, the framework maps time-frequency CSI observations into a compact latent field and performs prediction in the latent space. This representation emphasizes the dominant and relatively stable and predictable propagation-field structure, while reducing sensitivity to unstable small-scale fading and phaserelated details. The predicted latent state is then combined with a small number of real-time pilots as a structured prior for channel reconstruction, improving channel reconstruction under sparse pilot observations. The ablation results further show that CSI-domain grounding objectives, including diffusion grounding, amplitude recovery, and direct CSI recovery, do not necessarily improve downstream reconstruction. Instead, forcing the latent representation to preserve excessive finegrained CSI details can reduce its predictability and practical usefulness as a reconstruction prior.

## REFERENCES

[1] B. Liu, X. Liu, S. Gao, et al., “Llm4cp: Adapting large language models for channel prediction,” arXiv preprint arXiv:2406.14440, 2024.

[2] A. Bardes, Q. Garrido, J. Ponce, et al., “Revisiting feature prediction for learning visual representations from video,” arXiv preprint arXiv:2404.08471, 2024.

[3] J. Xiang, Y. Gu, Z. Liu, et al., “Pan: A world model for general, interactable, and long-horizon world simulation,” arXiv preprint arXiv:2511.09057, 2025.

[4] H. Zou, Y. Yang, L. Bariah, et al., “Telecom world models: Unifying digital twins, foundation models, and predictive planning for 6G,” arXiv preprint arXiv:2604.06882, 2026.

[5] S. Naoumi, M. Bennis, and M. Chafii, “Structured latent dynamics in wireless CSI via homomorphic world models,” arXiv preprint arXiv:2603.20048, 2026.

[6] L. Mur-Labadia, M. Muckley, A. Bar, et al., “V-JEPA 2.1: Unlocking dense features in video self-supervised learning,” arXiv preprint arXiv:2603.14482, 2026.

[7] J. Hoydis, F. A¨ıt Aoudia, S. Cammerer, et al., “Sionna RT: Differentiable ray tracing for radio propagation modeling,” in 2023 IEEE Globecom Workshops (GC Wkshps), IEEE, 2023, pp. 317–321.