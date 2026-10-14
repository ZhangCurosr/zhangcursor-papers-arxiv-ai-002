# High-fidelity Three-Dimensional Flow Field Reconstruction Driven by Turbulent Fluid Dynamics

Hanlin Kong<sup>1,2,</sup>†, Jianan Wei<sup>1,</sup>†, Yuntian Chen<sup>3,</sup>†, Yujia Wang<sup>4,\*</sup>, Wenguan Wang<sup>1,\*</sup>

<sup>1</sup>State Key Lab of Brain-Machine Intelligence, Zhejiang University, Hangzhou, Zhejiang, China <sup>2</sup>Shanghai Innovation Institute, Shanghai, China

<sup>3</sup>Zhejiang Key Laboratory of Industrial Intelligence and Digital Twin, Eastern Institute of Technology, Ningbo, Zhejiang, China

<sup>4</sup>Zhejiang Sci-Tech University, Hangzhou, Zhejiang, China

†These authors contributed equally to this work.

\* Corresponding authors.

## Abstract

High-fidelity three-dimensional turbulent flow fields are essential for scientific and engineering analyses, yet obtaining them through direct numerical simulation or high-resolution measure ments remains expensive. Super-resolution reconstruction from coarse-grained data therefore ofers a promising alternative, but remains fundamentally challenging because fine-scale struc tures are intrinsically three-dimensional and dynamically coupled across scales. Existing data driven approaches still struggle to recover multiscale volumetric structures and their coherent temporal evolution. To address these limitations, we propose a turbulence-aware framework for time-resolved three-dimensional flow reconstruction. The framework couples a multiscale neural operator, designed to recover volumetric structures, with a state space model to pro mote temporal consistency in the reconstructed dynamics. In addition, Navier–Stokes residuals are incorporated as soft constraints to regularize the reconstruction toward physically plausible dynamics. A mixture-of-experts mechanism is also introduced to support unsupervised adaptation to unknown coarse-graining. Across evaluations on canonical turbulent systems and atmospheric reanalysis data, the proposed framework reduces reconstruction errors relative to learning-based and classical baselines, while better preserving multiscale spatial structure and coherent temporal evolution. These results validate the efectiveness of the proposed framework in reconstructing high-fidelity three-dimensional flow fields from coarse-grained data.

Keywords: turbulent dynamics, flow field reconstruction, physics-informed learning

## 1 Introduction

Turbulent flows exhibit multiscale structures that evolve over time [1]. High-fidelity threedimensional flow fields are essential for both scientific research and engineering applications, e.g., atmospheric science [2], phytoplankton ecology [3], and scene imaging [4]. In practice, however, obtaining such data remains costly: direct numerical simulation (DNS) resolves all relevant scales but is computationally prohibitive [5], while more afordable approaches such as large-eddy simulation (LES) or Reynolds-averaged Navier–Stokes methods involve simplifying modeling assumptions that reduce fidelity. Consequently, time-resolved volumetric flow data are often available only at limited spatial resolution.

This motivates super-resolution reconstruction, which aims to recover high-resolution (HR) flow fields from their low-resolution (LR) counterparts. Early attempts leverage signal processing and reduced-order representations [6, 7], and recent progress in learning-based super-resolution improves the recovery of missing fine-scale content from data [8]. A representative study by Fukami et al. [9] shows that convolutional neural networks (CNNs) are able to achieve a “data-driven interpolation” that is more accurate than low-order polynomials. Subsequent studies explored more expressive architectures, such as transformer-based models [10, 11], generative adversarial networks [12–15], and difusion models [16–18], to further improve reconstruction quality. Despite these advances, many approaches primarily optimize point-wise fidelity and treat the mapping as a black box, raising concerns about whether the reconstructed flows remain consistent with the underlying governing physics. In light of this limitation, a growing trend is to incorporate physical priors into scientific machine learning [19–22]. Kochkov et al. [23] introduce a hybrid solver–learning pipeline to enforce physical consistency. Subsequent studies adopt more flexible strategies by incorporating physical constraints into the training objective. For example, Chung et al. [24] explicitly integrate gradientbased constraints, while Ren et al. [25] propose a soft-constraint approach by adding a divergence free loss as a regularization term.

Nevertheless, three challenges remain in time-resolved volumetric turbulence super-resolution. First, although previous studies have demonstrated the feasibility of reconstructing threedimensional turbulent fields [24, 26], the task requires more than predicting values on a volumetric grid: it demands recovering the multiscale spatial structure and coherent temporal evolution of the underlying flow. In three dimensions, vortical structures form connected geometries throughout the volume, and their local strain–rotation interplay is characterized by invariants of the velocity-gradient tensor [27]. Because these connected structures span a broad range of scales and evolve continuously, they cannot be fully represented by isolated two-dimensional slices, which motivates super-resolution directly on time-resolved three-dimensional fields. Second, although physical priors are increasingly adopted, they are typically introduced through frame-wise spatial constraints (e.g., gradient penalties [24] or divergence-free losses [25]). These formulations target kinematic properties at individual time steps, without accounting for the dynamical evolution of turbulent flows. As a result, reconstructions can appear spatially plausible while deviating from consistent dynamical trajectories. Third, many learning-based methods are trained under a fixed, predefined coarse-graining (e.g., average/max pooling [9] or specific LES–DNS pairings [13]), which limits robustness when the coarse-graining process difers at inference time.

Addressing these limitations goes beyond improving point-wise accuracy, and requires superresolution methods that maintain multiscale structure and coherent temporal evolution in dynami cal systems. To this end, we introduce a unified framework for time-resolved three-dimensional turbulence reconstruction. First, the framework operates directly on time-resolved three-dimensional fields and jointly models their multiscale spatial structure and temporal evolution. Temporal evolution is modeled on the Eulerian grid using a state space model, which promotes coherent evolution of reconstructed volumetric structures over time. To recover multiscale spatial structure intrinsic to turbulence, the framework further incorporates a multiscale neural operator that couples spectral modeling of large-scale interactions with convolutional refinement of local structures. Second, we incorporate Navier–Stokes residuals as soft constraints to promote dynamical consistency in the reconstructed flows. Finally, a mixture-of-experts (MoE) mechanism enables adaptation to unknown coarse-graining by combining predictions from pretrained experts, with Navier–Stokes residuals guiding the combination without HR supervision.

The framework is trained and evaluated on a curated benchmark for time-resolved volumetric turbulence super-resolution (TurbSR3D) spanning diverse flow regimes, including forced isotropic turbulence [28], forced magnetohydrodynamic turbulence [29], and homogeneous buoyancy-driven turbulence [30], with multiple upscaling factors and coarse-graining settings. A simulated decaying isotropic turbulence dataset is further used to test the framework under non-stationary conditions and on independently evolved LES inputs paired with DNS references. To evaluate both numerical accuracy and physical fidelity, we employ a set of criteria that includes common error metrics, diagnostics of multiscale spatial structure (e.g., energy spectra, distributional statistics, and velocity-gradient invariant topology), and measures of temporal consistency. Beyond canonical turbulence systems, we extend our analysis to complex atmospheric flows using the ERA5 data [31], where the task is framed as the downscaling of multi-level fields. Across these evaluations, the proposed framework improves reconstruction accuracy relative to strong learning-based and classical baselines, while better preserving multiscale structure and dynamically consistent evolution.

In summary, our contributions are as follows: (i) Spatiotemporal modeling: A turbulence aware architecture is introduced, which couples a multiscale neural operator with a state space model to jointly recover multiscale volumetric structures and capture their coherent temporal evolution. (ii) Physics-informed learning: Navier–Stokes residuals are incorporated as soft dynamical constraints and further used to guide unsupervised MoE adaptation to unknown coarse-graining. (iii) Benchmark and evaluation: TurbSR3D is introduced as a benchmark dataset for timeresolved three-dimensional turbulence super-resolution, with standardized data splits and multiple coarse-graining settings and upscaling factors. We evaluate the framework on canonical turbulent flows and atmospheric reanalysis data, assessing reconstruction accuracy, multiscale spatial structure, temporal consistency, and turbulence statistics.

## 2 Results

This section is organized into three parts. It begins with a high-level overview of the proposed framework and the inference pipeline (Fig. 1) to facilitate interpretation of the subsequent results. Reconstruction performance is then evaluated across diverse turbulent regimes, including forced isotropic turbulence (FIT), forced magnetohydrodynamic turbulence (MHDT), homogeneous buoyancy-driven turbulence (HBDT), and decaying isotropic turbulence (DIT), as well as real-world atmospheric reanalysis data from ERA5 [31]. The section concludes with an assessment of the framework behavior under unseen coarse-graining, including independently evolved LES inputs.

## 2.1 Problem set-up

To improve the computational eficiency of turbulent flow simulation, one often resorts to coarsegrid discretization and/or reduced-resolution measurements, which inevitably introduce approximation errors relative to fully resolved DNS. The objective is to reduce such errors by reconstructing high-fidelity, time-resolved volumetric turbulent dynamics from LR inputs.

The reconstruction of high-fidelity flow fields is cast as an inverse problem, wherein the objective is to recover the HR flow field representation ${ \pmb X } _ { \mathrm { H } } { \in } \mathbb { R } ^ { T \times H _ { \mathrm { H } } \times W _ { \mathrm { H } } \times D _ { \mathrm { H } } \times C }$ from its coarse-grid counterpart $\pmb { X } _ { \mathrm { L } } \in \mathbb { R } ^ { T \times H _ { \mathrm { L } } \times W _ { \mathrm { L } } \times D _ { \mathrm { L } } \times C }$ . Here, T denotes the temporal length of the sequence, (H, W, D) the spatial resolution of the three-dimensional grid, subscripts $\mathrm { H }$ and $\mathrm { L }$ indicate HR and LR fields, respectively, and C the number of physical variables $( e . g .$ , velocity components, pressure, or density). For the volumetric turbulence experiments, the spatial resolutions satisfy $H _ { \mathrm { H } } = s H _ { \mathrm { L } } , W _ { \mathrm { H } } = s W _ { \mathrm { I } }$ , and $D _ { \mathrm { H } } = s D _ { \mathrm { L } }$ , where s denotes the upscaling factor.

The degradation from HR to LR is formulated as

$$
X _ { \mathrm { L } } = \mathcal { D } ( X _ { \mathrm { H } } ) + \pmb { \xi } ,\tag{1}
$$

![](images/1abcbb09821b8546249b3574ea2460ff6ac9737221b697982cf358a4e7466a3c.jpg)  
Figure 1: Overview of the proposed framework for three-dimensional turbulence super-resolution. a, Set-up and overall framework. Given an LR three-dimensional sequence $X _ { \mathrm { L } }$ , the model learns a mapping $\mathcal { M } _ { \theta }$ that produces a representative HR estimate conditioned on the LR input. b, Illustration of the state space model at time step t, which captures temporal dynamics of turbulent flows. $\mathbf { c } ,$ Illustration of the multiscale neural operator for recovering multiscale spatial structure of turbulent flows. d, Illustration of an MoE mechanism with unsupervised adaptation, enabling generalization to unknown coarse-graining.

where D is referred to as the coarse-graining operator [32] and $\boldsymbol { \xi }$ quantifies the uncertainty arising from measurement or simulation inaccuracies. The coarse-graining operator D removes fine-scale information and is generally many-to-one, so multiple HR fields can be consistent with the same LR sequence. This non-uniqueness is described by the conditional distribution $p ( \pmb { X } _ { \mathrm { H } } \mid \pmb { X } _ { \mathrm { L } } )$ . The present framework targets a deterministic representative HR estimate under paired LR–HR supervision. This estimator is parameterized by a neural network $\mathcal { M } _ { \theta } : \mathbb { R } ^ { T \times H _ { \mathrm { L } } \times W _ { \mathrm { L } } \times \dot { D } _ { \mathrm { L } } \times C }  \mathbb { R } ^ { T \times H _ { \mathrm { H } } \times \dot { W } _ { \mathrm { H } } \times D _ { \mathrm { H } } \times C }$ which maps an LR sequence $X _ { \mathrm { L } }$ to a representative HR estimate $\hat { \pmb X } _ { \mathrm { H } } = \mathcal { M } _ { \pmb \theta } ( \pmb X _ { \mathrm { L } } )$ . The optimal parameters $\pmb { \theta } ^ { * }$ are obtained via:

$$
\pmb { \theta } ^ { * } = \underset { \pmb { \theta } } { \arg \operatorname* { m i n } } \ \mathbb { E } _ { ( X _ { \mathrm { L } } , X _ { \mathrm { H } } ) \sim \mathcal { T } } \left[ \mathcal { L } ( \mathcal { M } _ { \pmb { \theta } } ( X _ { \mathrm { L } } ) , \pmb { X } _ { \mathrm { H } } ) \right] ,\tag{2}
$$

where $\tau$ denotes the training data distribution, $\mathcal { L }$ represents the loss function, and E denotes the expectation over data pairs $( X _ { \mathrm { L } } , X _ { \mathrm { H } } )$ drawn from $\tau$ . The statistical role of the $L _ { 1 }$ data term and the physical regularization is described in Section 4.3.

As illustrated in Fig. 1a, given a temporal sequence of low-fidelity turbulent flow fields $X _ { \mathrm { L } }$ the proposed model first lifts the input to a latent representation via a learnable lifting mapping $P ,$ yielding $z ~ = ~ P ( X _ { \mathrm { L } } ) ~ \in ~ \mathbb { R } ^ { T \times H _ { \mathrm { L } } \times W _ { \mathrm { L } } \times D _ { \mathrm { L } } \times V }$ , where V denotes the latent channel dimension.

To preserve multiscale structures and coherent temporal evolution, the model explicitly performs spatiotemporal latent modeling of dynamics and cross-scale interactions. Specifically, the latent representation evolves over time through a state space model, which captures temporal dynamics on the Eulerian grid and retains temporal memory, thereby promoting temporal consistency across the reconstructed sequence (Fig. 1b). To characterize multiscale spatial structures in turbulent motions, the temporally evolved latent features are further processed by a multiscale neural operator that couples long-range interactions in the frequency domain with local refinement via convolution (Fig. 1c). Finally, a decoder followed by a projection mapping $Q$ maps the latent representation back to the HR physical space, producing the reconstructed flow field $\hat { X } _ { \mathrm { H } }$ . The decoder architecture and output projection are specified for the turbulence and ERA5 tasks in Supplementary Notes 1.5 and 1.8, respectively. In addition to the architectural design, we impose spatiotemporal constraints derived from the governing equations by incorporating Navier–Stokes residuals into the training objective, encouraging physically consistent dynamics.

Most supervised learning-based super-resolution methods [16, 26] are trained under a fixed coarse-graining, which limits their ability to generalize when the degradation strategy difers at in ference time. To account for this, an unsupervised MoE mechanism is employed (Fig. 1d). Multiple expert models correspond to diferent coarse-graining hypotheses, and a gating network adaptively combines their predictions to accommodate unseen coarse-graining.

## 2.2 Data preparation

The model is trained and evaluated on three classes of time-resolved volumetric turbulence from the TurbSR3D dataset (FIT, MHDT, and HBDT), covering distinct physical mechanisms and modeling challenges. We also use DIT data generated by DNS, alongside real-world atmospheric data from ERA5 [31]. The turbulence data and construction protocols are detailed below.

FIT [28]. As a canonical setting in turbulence research, forced isotropic turbulence is widely used to study multiscale spatiotemporal dynamics in incompressible flows [33, 34]. We consider the three-component velocity and pressure fields, denoted as $\{ \boldsymbol { u } , p \}$

MHDT [29]. To assess performance in a multi-physics regime, we consider incompressible MHD turbulence, which describes the coupled dynamics of an electrically conducting fluid and a magnetic field. The velocity, magnetic field, and pressure fields {u, b, p} are used to jointly evaluate kinetic and magnetic structures.

HBDT [30]. HBDT describes buoyancy-driven mixing between miscible fluids of diferent densities.   
Density, velocity, and pressure fields $\{ \rho , \pmb { u } , p \}$ are used to capture variable-density dynamics.

DIT. It shares the variables of FIT but is unforced, so its statistics evolve as the kinetic energy decays. We perform DNS of this flow and use it, separately from the TurbSR3D benchmark, to examine the physical constraints and generalization to LES inputs under statistically non-stationary conditions. We consider the three-component velocity and pressure fields $\{ \boldsymbol { u } , p \}$ . The simulation setup is described in Supplementary Note 2.7.

Coarse-graining simulation protocols. In practice, coarse-graining can arise from coarse-grid discretization (e.g., LES or RANS), limited-resolution measurements, or alternative numerical settings [35]. Since aligned low- and high-fidelity flow fields generated by such processes are generally unavailable, we construct paired training data by applying coarse-graining to high-fidelity data. Specifically, we consider three coarse-graining strategies for supervised training: (i) box/Favre filtering (box for FIT, MHDT and Favre for HBDT) [9, 24], which mimics volumetric or densityweighted averaging, (ii) Gaussian smoothing [36], which approximates smooth low-pass efects, and (iii) spectral cutof [18], which models sharp wavenumber truncation. To evaluate robustness to unseen coarse-graining, we further test (without retraining) on three additional strategies: tricubic downsampling, Lanczos filtering, and uniform subsampling with 5% additive noise. 0 0

Table 1: Numerical comparison of reconstruction metrics nRMSE (Eq. (18)) between the proposed method and baseline methods on the TurbSR3D benchmark.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">Coarse- graining</td><td colspan="2">FIT</td><td colspan="2">MHDT</td><td colspan="2">HBDT</td></tr><tr><td>4×</td><td>8×</td><td>4×</td><td>8×</td><td>8×</td><td>16×</td></tr><tr><td>Tricubic</td><td></td><td>0.1204</td><td>0.2242</td><td>0.1838</td><td>0.3372</td><td>0.0933</td><td>0.1666</td></tr><tr><td>RRDB</td><td rowspan="6">Box/ Favre</td><td>0.0460</td><td>0.1712</td><td>0.0678</td><td>0.2205</td><td>0.0307</td><td>0.0940</td></tr><tr><td>RCAN</td><td>0.0415</td><td>0.1753</td><td>0.0688</td><td>0.2285</td><td>0.0262</td><td>0.1396</td></tr><tr><td>EDSR</td><td>0.0423</td><td>0.1693</td><td>0.0699</td><td>0.2250</td><td>0.0286</td><td>0.1301</td></tr><tr><td>FNO</td><td>0.0730</td><td>0.1657</td><td>0.1166</td><td>0.2525</td><td>0.0544</td><td>0.1224</td></tr><tr><td>PhySR</td><td>0.0476</td><td>0.1486</td><td>0.0840</td><td>0.2443</td><td>0.0316</td><td>0.1159</td></tr><tr><td>DiffSR</td><td>0.0423</td><td>0.1440</td><td>0.0731</td><td>0.2153</td><td>0.0264</td><td>0.0860</td></tr><tr><td>SwinIR</td><td rowspan="11"></td><td>0.0421</td><td>0.1499</td><td>0.0792</td><td>0.2447</td><td>0.0353</td><td>0.1228</td></tr><tr><td>Ours</td><td>0.0291↓29.9%</td><td>0.1195↓17.0%</td><td>0.0603↓11.1%</td><td>0.2094↓2.7%</td><td>0.0195↓25.6%</td><td>0.0739↓14.1%</td></tr><tr><td>Tricubic</td><td>0.1325</td><td>0.2278</td><td>0.2000</td><td>0.3327</td><td>0.0959</td><td>0.2074</td></tr><tr><td>RRDB</td><td>0.0494</td><td>0.1749</td><td>0.0692</td><td>0.2246</td><td>0.0316</td><td>0.0985</td></tr><tr><td>RCAN</td><td>0.0421</td><td>0.1802</td><td>0.0712</td><td>0.2330</td><td>0.0275</td><td>0.1055</td></tr><tr><td>EDSR</td><td>0.0445</td><td>0.1752</td><td>0.0718</td><td>0.2290</td><td>0.0299</td><td>0.1452</td></tr><tr><td>FNO</td><td>Gaussian 0.0745</td><td>0.1679</td><td>0.1188</td><td>0.2559</td><td>0.0535</td><td>0.1224</td></tr><tr><td>PhySR</td><td>0.0494</td><td>0.1517</td><td>0.0852</td><td>0.2479</td><td>0.0389</td><td>0.1165</td></tr><tr><td>DiffSR</td><td>0.0438</td><td>0.1456</td><td>0.0756</td><td>0.2175</td><td>0.0300</td><td>0.0984</td></tr><tr><td>SwinIR</td><td>0.0429</td><td>0.1535</td><td>0.0804</td><td>0.2496</td><td>0.0354</td><td>0.1242</td></tr><tr><td>Ours</td><td>0.0300↓28.7%</td><td>0.1173↓19.4%</td><td>0.0620↓10.4%</td><td>0.2099↓3.5%</td><td>0.0200↓27.3%</td><td>0.0754↓23.4%</td></tr><tr><td>Tricubic</td><td rowspan="10">Spectral</td><td></td><td>0.2505</td><td>0.2204</td><td></td><td>0.1196</td><td></td></tr><tr><td>RRDB</td><td>0.1501 0.0770</td><td>0.2156</td><td>0.1130</td><td>0.3640</td><td>0.0432</td><td>0.1671</td></tr><tr><td></td><td>0.0755</td><td>0.2209</td><td>0.1135</td><td>0.3013 0.3105</td><td></td><td>0.1354</td></tr><tr><td>RCAN</td><td>0.0759</td><td>0.2140</td><td>0.1163</td><td>0.3036</td><td>0.0410 0.0428</td><td>0.1438</td></tr><tr><td>EDSR FNO</td><td>0.1036</td><td>0.2098</td><td>0.1612</td><td>0.3141</td><td>0.0812</td><td>0.1476 0.1767</td></tr><tr><td>PhySR</td><td>0.0722</td><td>0.2002</td><td>0.1188</td><td>0.3117</td><td>0.0452</td><td>0.1511</td></tr><tr><td></td><td></td><td>0.2246</td><td>0.1339</td><td>0.3226</td><td>0.0496</td><td>0.1366</td></tr><tr><td>DiffSR</td><td>0.0810</td><td></td><td>0.1190</td><td>0.3628</td><td>0.0521</td><td></td></tr><tr><td>SwinIR</td><td>0.0704</td><td>0.2067</td><td></td><td>0.2957↓1.9%</td><td>0.0348↓15.1%</td><td>0.1602</td></tr><tr><td>Ours</td><td>0.0615↓12.6%</td><td>0.1884↓5.9%</td><td>0.1071↓5.2%</td><td></td><td></td><td>0.1127↓16.8%</td></tr></table>

Three turbulent regimes are considered: incompressible forced isotropic turbulence (FIT), incompressible magnetohydrodynamic turbulence (MHDT), and homogeneous buoyancy-driven turbulence (HBDT). Coarse-graining is simulated using three filtering strategies: box/Favre, Gaussian, and spectral cutof. We evaluate multiple upscaling factors (4×, 8×, and 16×), depending on the regime.

Experimental configuration. The evaluation covers FIT, MHDT, and HBDT under two upscaling factors per regime: $4 \times / 8 \times$ for FIT and MHDT, and $8 \times / 1 6 \times$ for HBDT. For each supervised evaluation setting, data are split into training, validation, and test sets with a 6:2:2 ratio, and models are trained from scratch. Additional details are provided in Section 4 and Supplementary Note 2.

## 2.3 Accurate reconstruction of turbulent flows

Numerical reconstruction accuracy is assessed by benchmarking the proposed method against representative deep learning baselines, including RRDB [37], EDSR [38], RCAN [39], FNO [40], PhySR [41], DifSR [42] and SwinIR [43] (see Section 4.7). We utilize established metrics in superresolution, including the normalized root mean square error (nRMSE) presented in Table 1, as well as the peak signal-to-noise ratio (PSNR), structural similarity index measure (SSIM), and Fourier RMSE (fRMSE) reported in Supplementary Tables 10–12. As shown in Table 1, the proposed method achieves the lowest nRMSE across all evaluated settings. On FIT, our method outperforms the second-best results in terms of nRMSE by 29.9% (4×, box), 28.7% (4×, Gaussian), and 12.6% (4×, spectral), with improvements persisting at 8× (17.0%, 19.4%, and 5.9%). MHDT presents the greatest challenge for joint restoration of its kinetic-magnetic structures, which are characterized by multi-physics interactions and non-local coupling. Nevertheless, our method remains superior, albeit with a smaller margin than in FIT. For HBDT, which features variable-density mixing and buoyancy-driven turbulence, the proposed method achieves significant error reductions in nRMSE (14.1%–27.3%). The proposed method also achieves the highest SSIM and PSNR and the lowest fRMSE across all 18 settings (Supplementary Tables 10–12). To quantify training uncertainty, we conduct multi-seed experiments. The proposed method achieves the best mean performance across all five metrics with low variability across seeds (Supplementary Table 20). These results corroborate the importance of injecting spatiotemporal inductive biases for turbulence, where coherent evolution and cross-scale interactions are central to reconstruction fidelity. Moreover, our method has a parameter count comparable to or smaller than most competing baselines (see Supplementary Fig. 1), indicating that the improvements do not arise from increased model capacity, but from the proposed spatiotemporal design.

## 2.4 Recovering multiscale spatial structure from coarse-grained data

Beyond numerical accuracy, we further examine whether the proposed method produces highfidelity reconstructions that preserve multiscale spatial structure and maintain coherent structures inherent to turbulent flows.

Point-wise fidelity. Fig. 2 provides qualitative, point-wise comparisons using representative field quantities across three regimes, specifically targeting the recovery of challenging small-scale features.  First, we examine the vorticity magnitude ∥ω∥ on FIT to probe recovery of small-scale rotational filaments and intermittent structures. As shown in Fig. 2a, the proposed method yields more continuous filaments with reduced residual artifacts. In contrast, FNO [40], RRDB [37], and PhySR [41] exhibit red–blue halo patterns around vortex cores, suggesting incomplete recovery of fine-scale structures. Second, we examine the current magnitude ∥j∥ on MHDT to probe thin current sheets and magnetic shear layers (Fig. 2b).  The proposed method recovers sharper and more continuous sheet structures, whereas competing baselines tend to produce thicker, fragmented sheets with orientation deviations. Third, we use the velocity divergence field ∇ · u on HBDT to characterize mixing-induced dilatation in variable-density flow under challenging upscaling (Fig. 2c). The proposed method better preserves the spatial organization of divergence features, while baselines introduce structural distortions and unphysical patterns. For example, FNO [40] and PhySR [41] show pronounced grid-like artifacts and discontinuities, whereas RRDB [37] exhibits boundary-related ringing. These results align with the local refinement mechanism of the multiscale neural operator, which facilitates the recovery of small-scale details from coarse inputs.

![](images/02d6c7ebd9d57149a4063b418477ca250238f5eda182acdd4e196c8b72c6e765.jpg)  
Figure 2: Point-wise fidelity across three turbulence regimes and representative field quantities. ${ \mathbf { a } } ,$ Predicted vorticity magnitude $\| { \pmb \omega } \| = \| \nabla \times { \pmb u } \|$ and corresponding residual maps on FIT with 4 upscaling. b, Predicted current magnitude $\| \boldsymbol { j } \| = \| \nabla \times \pmb { b } \|$ and residual maps on MHDT with 4 upscaling. c, Predicted velocity divergence u and residual maps on HBDT with 8 upscaling.

Energy spectra. The energy spectrum describes how energy is distributed over spatial scales, with higher wavenumbers corresponding to finer structures. Fig. 3a compares energy spectra across the three turbulence regimes, with insets highlighting the inertial–dissipation transition. We focus on spectral-cutof coarse-graining, which explicitly removes high-wavenumber content and therefore poses a stringent test for spectrum recovery. Across FIT, MHDT, and HBDT, the proposed method improves the recovery of high-wavenumber energy, reducing the underestimation exhibited by FNO, RRDB, PhySR, and SwinIR in the magnified ranges. In HBDT, the proposed method also better reproduces the decay of the reference spectrum towards the upper end of the displayed wavenumber range, where the baselines exhibit excess energy. Together, these results indicate that the proposed method more accurately reproduces the distribution of energy across scales, reducing both the en ergy deficit and the excess observed in the baseline spectra. This spectral behavior is consistent with the spatial comparisons in Fig. 2c, where the proposed reconstruction avoids the grid-like and boundary artifacts visible in baseline models. Under box/Favre coarse-graining, similar behavior is observed, with improved spectral consistency relative to competing baselines (Supplementary Fig. 13). These diferences can be interpreted through the architectures’ spatial inductive biases. RRDB uses residual dense convolutions [37], PhySR combines convolutional recurrence with con volutional reconstruction [41], and SwinIR uses windowed and shifted-window attention [43]. Their spatial context grows through successive local or windowed operations. FNO provides global coupling through a finite set of retained Fourier modes [40]. Our multiscale neural operator combines global Fourier coupling [44] with local three-dimensional convolutional refinement, providing both large-scale context and local corrections for recovering missing fine-scale fluctuations. In the ab lation study of the multiscale neural operator, removing the spectral operator produces the larger increase in low-wavenumber velocity reconstruction error, while removing the convolutional operator produces the larger increases at intermediate and high wavenumbers (Supplementary Fig. 5 and Supplementary Table 14). Both removals afect all three ranges, supporting complementary and overlapping contributions to reconstruction across scales. On FIT at 4×, replacing the multiscale neural-operator blocks with MS-ESRGAN-derived convolutional blocks [45] at closely matched capacity increases nRMSE from 0.0615 to 0.0739 (20.1%) and produces greater high-wavenumber energy underestimation (Supplementary Table 15 and Supplementary Fig. 6).

Probability density functions. Probability density functions (PDFs) provide a complementary statistical perspective on whether reconstructed fields recover not only typical values but also intermittency. Fig. 3b shows PDFs of representative quantities: vorticity magnitude ∥ω∥ (FIT, 4×), current magnitude ∥j∥ (MHDT, 4×), and velocity divergence ∇·u (HBDT, 8×). Each quantity is normalized by its root-mean-square (rms) value to emphasize the distributional shape. Across all three regimes, the proposed method closely follows the HR reference, including the tails that reflect intermittency and rare extremes. For FIT, it reproduces the vorticity distribution over a wide dynamic range. For MHDT, it recovers the tail decay of ∥j∥, consistent with localized currentsheet statistics. For HBDT, the proposed method better reproduces the central peak and reduces the excess probability in the tails. In contrast, baselines align with the reference mainly near the mode but exhibit increasing tail deviations, indicating reduced fidelity in intermittent small-scale statistics. Distributional similarity is further quantified by the Jensen–Shannon (JS) divergence (Supplementary Fig. 14), where the proposed method achieves the lowest values in all cases. These results indicate improved recovery of tail behavior and intermittent statistics.

a  
![](images/77422e14511c5f73d19ab22705abb6675d73808d7f50f1062cf1fbd2042070fe.jpg)

Energy Spectrum across turbulent systems  
![](images/fbf1b018d0361cb54f5671ee89bd58822c5acb13f0e7c10b9a1e11716f036333.jpg)

![](images/23234cb13b945c060e050dfb9a6bc4b1c0605d8082bcebe99a41ae8e9c8c7b93.jpg)

b  
![](images/6db9e248da9174c83351d3e2e5bd9565af1b76996b5e019fe0682a9165ff6d9d.jpg)

PDF comparison across turbulent systems  
![](images/df4b0528423e6ab8ee2528300a1770fd8b2727fa25fa10b70e13cb331dc964a7.jpg)

![](images/96005e533394d1329c41f96a81f9d7f20b64b9289aeab6550d4e8b347167c1fc.jpg)

c  
![](images/01171e0107e2f2b96a0df9a78a4f9c0f909743930f1620b9ebc8f8e0a13219d6.jpg)

Q–R joint PDFs across super-resolution scales  
![](images/ec0a307077d27f624ecc366df13dfef2b4c95eb81be7372121fa6f0da8a530d4.jpg)

![](images/0b7b8a2222a130ae767320cbb1ca57363ffd9aa4e81084407ed69a9b0e02f9f3.jpg)

![](images/75d696cf91467f9979319733c4ceb6112f9729e8d915d991b2994706d6c13afd.jpg)

![](images/e2124f6a3dc203282bbcb9592c4c17e7c29beb82a5a2d926ca0eab9d124f77ed.jpg)

![](images/d76ea24a26f50794df86b03a16bfb7054693f6204bfe148a5f53591f899b778f.jpg)

![](images/9d45b9cd2098870cc4bfedad2cbf54f6462df229c17c2930a86fe9c4aeebdef9.jpg)

![](images/23c90154cee7a16a2c61fe74ef47a8825ecf0135efbc9a0c69b0be4d74533ef9.jpg)  
Figure 3: Statistical fidelity across three turbulence regimes. ${ \mathbf { a } } ,$ Energy spectra for FIT $( 4 \times )$ , MHDT $( 4 \times )$ and HBDT (8 ) under spectral-cutof coarse-graining. $\mathbf { b } ,$ Probability density functions (PDFs) of representative field quantities: vorticity magnitude $\| \omega \|$ (FIT, 4 ), current magnitude $\lVert j \rVert$ (MHDT, 4 ), and velocity divergence $\nabla \cdot \pmb { u }$ (HBDT, 8 ) under box/Favre coarse-graining. $( \cdot ) _ { \mathrm { r m s } }$ denotes the root-mean-square magnitude. $\mathbf { c } ,$ Joint PDFs of normalized velocity-gradient invariants $( Q ^ { * } , R ^ { * } )$ for FIT at $4 \times$ and $8 \times$ with box-filtered inputs. Black and colored contours denote DNS and reconstructions, respectively; the magenta dashed curve marks the zero-discriminant boundary. Evaluation details are given in Supplementary Note 3.5.

Topology of velocity-gradient invariants. To assess local deformation and rotation, we decom pose the velocity gradient $\pmb { A } = \nabla \pmb { u } = \pmb { S } + \pmb { \Omega }$ into its symmetric strain tensor $\pmb { S }$ and antisymmetric rotation tensor Ω. For incompressible flow, the invariants are $\begin{array} { r } { Q = \frac { 1 } { 2 } \left( \lVert \boldsymbol { \Omega } \rVert _ { F } ^ { 2 } - \lVert \boldsymbol { S } \rVert _ { F } ^ { 2 } \right) } \end{array}$ $R =$ $- \operatorname* { d e t } ( A )$ . Thus, $Q > 0$ identifies rotation-dominated regions and $Q < 0$ strain-dominated regions, while R combines the strain configuration with strain–vorticity coupling $[ 4 6 , 4 7 ]$ . Their joint PDF describes how frequently these local configurations occur. Its characteristic teardrop shape reflects preferred combinations of strain and rotation in turbulence [48]: a broad rotation-dominated region and an elongated strain-dominated lower-right branch. The latter includes motions that extend a fluid element in two directions and compress it in the third, locally flattening it. These statistics probe small-scale physics because strain governs viscous dissipation and vortex stretching amplifies vorticity (Supplementary Note 3.5). Fig. 3c compares joint-PDF contours of the normalized invariants $( Q ^ { * } , R ^ { * } )$ for FIT with box-filtered inputs. A common DNS strain scale and common density thresholds preserve diferences in gradient intensity and occurrence frequency across reconstructions (Supplementary Note 3.5). At $4 \times$ , all four methods retain the basic teardrop shape, with our method and PhySR closely following the reference. At 8×, FNO and PhySR show pronounced contour contraction along the strain-dominated lower-right branch, while RRDB additionally shifts the rotation-dominated contours towards larger $R ^ { * }$ and distorts the shortened lower-right branch. These deviations indicate altered strain–rotation statistics and deficient recovery of strongly strain dominated states. Stronger coarse-graining removes more fine-scale information, and the quadratic and cubic dependence of Q and R on velocity gradients amplifies reconstruction errors. Our method better preserves both features, consistent with global spatial context supporting local velocity refinement and physical constraints regularizing derivative relationships. The ablation reveals distinct regional changes (Supplementary Fig. 12): removing the local operator or physical loss shifts the right flank of the rotation-dominated contours towards larger $R ^ { * }$ , whereas removing the spectral operator contracts the negative- $R ^ { * }$ side across both rotation- and strain-dominated regions. Al three removals also impair the lower-right strain-dominated branch, with several intermediate con tours showing insuficient extension and larger departures from the reference than in the full model. Extended baseline comparisons appear in Supplementary Fig. 11.

Taken together, these diagnostics demonstrate that the proposed method preserves multiscale spatial structure across turbulent regimes. In particular, the multiscale neural operator plays a central role in this capability by combining global spectral interactions with local convolutional refinement, enabling complementary recovery of large-scale organization and fine-scale fluctuations. The resulting improvements are consistently reflected across complementary measures of spatial fidelity, including point-wise reconstruction accuracy, energy spectra, distributional statistics, and velocity-gradient invariant topology.

## 2.5 Capturing the temporal evolution of turbulent flows

Turbulent dynamics are time dependent. This motivates the design of the state space model, which maintains a latent state to encode temporal memory and updates it recurrently on the Eulerian grid. This temporal context helps reconstruct the evolving turbulent flow.

Qualitative evolution of vortex structures. Fig. 4 visualizes Q-criterion isosurfaces for FIT at six instants $\left( t _ { 0 } { - } t _ { 1 0 0 } \right)$ . The HR reference (Fig. 4a) shows smoothly advecting and deforming vortex tubes. At 4× super-resolution (Fig. 4b), the reconstructed vortex tubes closely follow the changes in geometry and connectivity seen in the HR reference. At the more demanding 8× setting (Fig. 4c), small-scale filaments are partially attenuated by coarse-graining, yet the large-scale vortical motion remains temporally aligned with the HR sequence.

Quantitative evaluation of temporal coherence. To complement the visual evidence, Fig. 5a shows t–x slices of vorticity magnitude ∥ω∥. Our reconstructions form continuous diagonal ridges with slope and phase consistent with the reference, whereas RRDB, RCAN and PhySR exhibit fragmented or discontinuous streaks, indicative of temporal inconsistency. We quantify errors in frame-to-frame changes using temporal RMSE (tRMSE; Eq. (19)) in Fig. 5b. Across FIT, MHDT, and HBDT, our method achieves the lowest tRMSE, with substantial reductions (23%–31%) relative to the best-performing baseline shown in Fig. 5b. Together, the structural comparisons and lower temporal-increment errors indicate closer agreement with the reference flow evolution.

Ref.a  
![](images/49302f586bf44f0942cb1b5ddaf2cb35428e627cfa164c9793e901c57333173b.jpg)

![](images/e31f8049478e58c830243118c758b89d566f5e8d678d293ccdf8e20135a96550.jpg)

![](images/834a04792fa6cd6600dca21c085eede7748c645a604d6969062a7bdf0d4a2948.jpg)  
Figure 4: Q-criterion structures over time. a, Reference HR flow at six instants $t _ { 0 } , \ldots , t _ { 1 0 0 }$ (left to right) on FIT, with a uniform cadence $\Delta t = 2 { \times } 1 0 ^ { - 3 }$ . For each time, an isosurface of the Q-criterion highlights rotationdominated regions; surfaces are color-mapped by speed u . b and c, Super-resolution experiments at 4 and 8 : for each SR factor, the top row shows the LR field, and the bottom row shows the reconstruction results by our method, evaluated at the same times as a. The LR Q fields are computed from the coarse-grained velocity components, with velocity gradients evaluated using the LR grid spacing.

A state space model ablation on FIT supports the contribution of temporal modeling: relative to the variant trained without the state space model, the full model reduces nRMSE and tRMSE by 14.36% and 16.05% at 4×, and by 12.87% and 6.52% at 8×, respectively. The time–space slices show that the full model closely matches the reference structures and preserves their smooth evolution, while the model without the state space model and the other baselines exhibit varying degrees of local blurring, deformation, and reduced temporal smoothness (Supplementary Fig. 7). The state space model also outperforms other temporal baselines at closely matched parameter counts in the 4× comparison (Supplementary Note 3.4).

![](images/db4162285057cbc1b282e76eed43f6439bbfe552484cc23979cada13423659d6.jpg)

![](images/4f212ace1a39d0953b1d9d29332841674c3a096559479028cdf73098df94e219.jpg)

![](images/559c34a0007424f3ede78279dfd7adf6e392c6950440abe32852962b4ad8df6a.jpg)

![](images/389a408351fea7a724030580b696d376c94853aa9d77132abcc4ec08d03f6568.jpg)  
Figure 5: Temporal coherence of reconstructions. a, Time–space (t–x) slices of vorticity magnitude ω for a representative FIT sequence. The horizontal axis denotes time t and the vertical axis denotes the spatial coordinate x. The top row shows the reference (HR) field and reconstructions from diferent methods, and the bottom row shows error maps. b, Quantitative temporal error measured by tRMSE (lower is better; values reported in units of $\times 1 0 ^ { - 3 } )$ , computed using Eq. (19), for FIT (4 , left), MHDT (4 , middle), and HBDT (8 , right).

![](images/6b6ffc754eb9084866989efa90f2b592a702634d7a510a26b3a15b783c21c2f9.jpg)

![](images/747d75998a0ae231cb4cc96d513a7b6d49010bc8b8cd774ea3757edbdb314a3d.jpg)

c  
![](images/b4edbee081cfda09a8f1ee360659ec652c0c515ef344b2eb1d1182ff230e91f8.jpg)

![](images/e21bf23d314118e2e402752db620eda454101cba47743ee7f069732b9cbc5963.jpg)

![](images/2532023bf09acbe350d39d40e1d8fb47feec49f7c1831044cd22c230015a1178.jpg)

![](images/c6a1314c6dfc79a209b032b4749b9bbbe476ff22c1d7b691ebe84cd157c205d3.jpg)

![](images/90eb7435881eac437e85f456be441d098b496e8e5e58202c39f68ae9fe8d886c.jpg)  
Figure 6: Dynamical consistency in DIT. Reconstructions are evaluated at 4 upscaling with spectral-cutof inputs on DIT dataset (Supplementary Note 2.7). a, Reference flow fields at five times from $t = 0 . 2$ to $t = 0 . 6$ . b,c, Evolution of normalized kinetic energy $K ( t ) / K ( t _ { 0 } )$ and Taylor-scale Reynolds number $R e _ { \lambda }$ in the DNS. d, Space–time slices of vorticity magnitude $\| \boldsymbol { \omega } \|$ for the DNS reference (Ref.), the full model (Ours), and the model trained without $\mathcal { L } _ { \mathrm { p h y } } ,$ with reconstruction error maps beneath the predictions. $\mathbf { e , f , }$ Reconstruction error (nRMSE) and temporalincrement error (tRMSE); lower values indicate greater fidelity. g, Spatially averaged viscous dissipation normalized by the DNS value at the corresponding time. All displayed values are below unity, so higher ratios indicate closer agreement with DNS. Red and blue curves in e–g denote the full model and the variant without $\mathcal { L } _ { \mathrm { p h y } }$ , respectively.

Dynamical consistency in decaying turbulence. To test whether physical soft constraints promote dynamical consistency under non-stationary conditions, we evaluate reconstructions of DIT (Supplementary Note 2.7) at 4× upscaling with spectral-cutof inputs. The sustained decline in kinetic energy and Taylor-scale Reynolds number (Fig. 6a–c) requires reconstructions to track evolving flow statistics, providing a stringent test of fidelity throughout turbulent decay. We com pare the full model with a variant trained without the physical loss $\mathcal { L } _ { \mathrm { p h y } } \left( \mathrm { E q . } \left( 1 0 \right) \right)$ . The space–time slices show that the full model better preserves the position and continuity of the inclined vortic ity streaks, with reduced local distortions and weaker error bands along the evolving structures (Fig. 6d). Consistently lower tRMSE indicates closer agreement with reference frame-to-frame increments (Fig. 6f), while lower nRMSE confirms improved reconstruction accuracy throughout the plotted interval (Fig. 6e). The viscous dissipation rate, $\varepsilon = 2 \nu S _ { i j } S _ { i j }$ , where ν is the kinematic viscosity and $S _ { i j } = \textstyle { \frac { 1 } { 2 } } ( \partial u _ { i } / \partial x _ { j } + \partial u _ { j } / \partial x _ { i } )$ is the strain-rate tensor, measures the rate at which kinetic energy is converted into heat. Its dependence on velocity gradients makes dissipation a diagnostic of small-scale recovery. Although dissipation recovery becomes less complete as decay proceeds, the physical constraints maintain the mean dissipation ratio, $\langle \varepsilon _ { \mathrm { S R } } \rangle / \langle \varepsilon _ { \mathrm { D N S } } \rangle$ (angle brackets denote spatial averaging), closer to unity throughout the plotted interval, with the advantage over the variant without $\mathcal { L } _ { \mathrm { p h y } }$ widening towards later times (Fig. 6g). Together with the improved temporal-increment fidelity, this sustained benefit supports the role of physical soft constraints in promoting dynamical consistency as turbulence decays.

## 2.6 Atmospheric downscaling on ERA5

This section evaluates atmospheric downscaling on ERA5 reanalysis fields [31], serving as an application-oriented validation in a real-world geophysical setting exhibiting multiscale structure and temporal coherence. In this experiment, the LR input is a spatiotemporal multi-level atmospheric field $\pmb { X } _ { \mathrm { L } } \in \mathbb { R } ^ { T \times H _ { \mathrm { L } } \times W _ { \mathrm { L } } \times P \times C }$ , where H and W correspond to latitude–longitude grids and P indexes the pressure levels spanning 1000–300 hPa. The channel dimension C includes temperature and horizontal wind components $( u , v )$ . Given the input $X _ { \mathrm { { L } } } ,$ the model reconstructs the corresponding HR sequence $\pmb { X } _ { \mathrm { H } } \in \mathbb { R } ^ { T \times H _ { \mathrm { H } } \times W _ { \mathrm { H } } \times P \times C }$ at higher spatial resolution. As in the turbulence experiments, the third coordinate axis participates directly in the spatial representation. The three-dimensional backbone applies its Fourier and convolutional operators jointly over latitude, longitude, and pressure level, while C indexes the physical variables. The ERA5 decoder increases the horizontal resolution through planar upsampling while retaining the prescribed pressure levels. Thus, the ERA5 task uses a three-dimensional cross-level representation with a horizontally refined output grid. We compare against commonly used baselines for atmospheric downscaling [25, 49], including bicubic interpolation, SwinIR [43], and FNO [40]. More details about experiment settings are provided in Section 4.6.

a  
![](images/5ae40699be2b57eb4b342f4682110ffaf58a293433c39b4bd0e0ff30b994d6d6.jpg)

![](images/717188d1fe085113051f763bdf7cbb389a881e3e1e09cffd1d0896c024bf08b6.jpg)

![](images/0786d992d362237a58d3ebaa032c75d5a1c66470e25dc7c335db52ab7b5faeee.jpg)

c  
![](images/6ec5601f794b051bdd0f0cf86a960d63cce77ec09f73a69f15046fd213da4032.jpg)  
Figure 7: Atmospheric downscaling on ERA5 reanalysis data. a, Mean Squared Error (MSE) and temporal error (tRMSE, Eq. (19)) computed at 4 and 8 downscaling factors. b, Kinetic energy spectra E(k) at 850 hPa plotted against wavenumber k. c, Visualizations of temperature (top) and wind speed (bottom) fields at 850 hPa. The panels display the HR reference (Ref.) alongside reconstructions from the proposed method and SwinIR. Inset display magnified views of local regions.

Fig. 7a reports MSE and tRMSE for 4× and 8× downscaling, measuring grid-point error and temporal fidelity, respectively. For both factors, the proposed method attains the lowest MSE among the compared methods. The same ordering holds for tRMSE, indicating smaller errors in frame-to-frame changes. This is consistent with the design goal of modeling temporal dynamics in latent space. We also quantify training uncertainty through multi-seed experiments. The proposed method achieves the best mean performance across all five metrics at both downscaling factors, with low variability across seeds (Supplementary Table 21).

To further examine spatial fidelity, we compare spatial fields and spectra at 850 hPa. This level is widely adopted in synoptic and jet-related analyses [50] as a representative proxy for diagnosing lower-tropospheric wind and temperature structures. Fig. 7c shows that the proposed method better preserves localized gradients and spatial organization in both temperature and wind-speed fields, with reduced artifacts in the zoomed regions. The kinetic energy spectra in Fig. 7b are closer to the reference over a range of wavenumbers, indicating improved recovery of the multiscale energy distribution. These results align with the multiscale design of the proposed model, which couples global interactions with local refinement to reconstruct scale-dependent structures.

Vertical wind shear is a critical diagnostic in atmospheric analysis [51, 52]. To evaluate crosslevel consistency in the reconstructed fields, we compute the 850–500 hPa bulk wind-vector diference and evaluate it using vCoh and vRMSE. Our method achieves the highest vCoh and lowest vRMSE at both downscaling factors, indicating more accurate preservation of the amplitude and spatial structure of this cross-level diagnostic than baselines (Supplementary Table 19).

## 2.7 Generalizing to unknown coarse-graining

This section evaluates generalization to unknown coarse-graining in two complementary settings. First, we study FIT under prescribed coarse-graining operators to assess adaptation to changes in filtering, resampling, and measurement noise. We then construct paired LES–DNS sequences of DIT to assess performance under a more realistic, simulation-generated degradation. In this second setting, coarse inputs arise from an independently evolved LES with a subgrid-scale (SGS) closure, introducing modeling and numerical errors that evolve over time.

We train three models under distinct coarse-graining strategies (Table 1). Box averaging and Gaussian smoothing attenuate a range of wavenumbers, whereas spectral cutof removes modes above a prescribed threshold. Each expert therefore learns a reconstruction mapping conditioned on the spectral content retained by its training coarse-graining. Although each model performs well under its training coarse-graining, performance degrades when the coarse-graining difers from the training setting, a challenge observed in turbulence super-resolution [12]. This limits practical applicability, where the specific coarse-graining is often unknown and can vary across data sources. To address this issue, we introduce an MoE mechanism. Each pretrained model serves as an expert corresponding to a coarse-graining hypothesis, and its parameters are frozen. At inference time, a lightweight gating network is optimized to combine the frozen expert predictions for given LR inputs (see Section 4.4).

![](images/b504ace30e019723a2ed0920694812bc6379444dc70292f2f689a4fb1033be28.jpg)

![](images/cdd6bff2154c89484cb501d8ef843096b87d1abc396dc65aae291cce6fb714d6.jpg)

![](images/7054fdb0ece70bc0e95c92868a18be7072e2566d0527ecd17b5d72c01d275737.jpg)

![](images/d4bb552148f843380597a107d30e537247d2d5e573c2f7bcb4cdf1c070366ed6.jpg)

e  
![](images/655766e005bf09816a79ada64e87c0efa2e8d47c69d12756f8ef15dd3b6a7940.jpg)

![](images/2d1cd69d35144ccaf77d2fc2bdf4f49b97f07f19f66fb109198ec04fab1e8470.jpg)

![](images/a89b832d17153378f1568ea87f37e3db199a83392a4f9e8ebe877366518d247a.jpg)

![](images/2aa07b818061c46eb3cb636082754e3cea40b4404ee2612c89b4ed2fc9d4cd87.jpg)  
Figure 8: Generalization to unknown coarse-graining. a, nRMSE of the Box, Gaussian, and Spectral experts and the MoE under three unseen coarse-graining settings: tricubic downsampling, Lanczos filtering, and uniform subsampling with 5% additive noise. b, Energy spectrum E(k) for the tricubic case. c, Reconstruction of vorticity magnitude $\| \omega \|$ under tricubic downsampling, showing the HR reference, LR input, and outputs from the MoE and individual experts. d, Joint PDFs of normalized velocity-gradient invariants $( Q ^ { * } , R ^ { * } )$ for the tricubic case (Supplementary Note 3.5). e, nRMSE of the MoE under uniform subsampling with 5% noise as a function of the number of LR sequences used for adaptation. f, Pearson correlation coeficients of reconstruction error fields between pairs of experts for the tricubic case (Supplementary Note 3.8).

Evaluation is conducted on FIT under three representative unseen coarse-graining settings: (i) tricubic downsampling, commonly used in rescaling-based super-resolution benchmarks [25]; (ii) Lanczos-filtered downsampling, a standard low-pass resampling strategy in signal processing [53, 54]; and  (iii) uniform subsampling with 5% additive Gaussian noise, mimicking noisy measurements [25]. As shown in Fig. 8a, the MoE achieves lower numerical error than any single expert model across these settings.

Tricubic downsampling provides a representative case for examining multiscale spatial fidelity. The three experts share the same architecture, but each learns a diferent reconstruction mapping associated with its training coarse-graining. Their spectral diferences on the same tricubic input therefore reveal the sensitivity of fine-scale reconstruction to the assumed coarse-graining. In Fig. 8b, all three experts overestimate energy over part of the high-wavenumber range, with the largest excess in the box-trained model. At still higher wavenumbers, their spectra decay prematurely. This scale-dependent bias is consistent with a mismatch between the spectral content retained in the test inputs and that encountered during training [12]. By adapting the combination of expert predictions to the unseen inputs, the MoE reduces the energy excess and better repro duces the reference spectral decay across the magnified range. Consistent trends are also observed in structure-sensitive diagnostics: the MoE maintains the characteristic shape and extent of the DNS Q–R topology (Fig. 8d) and avoids excessive smoothing in vortical structures, resulting in improved structural continuity (Fig. 8c).

To assess the sample eficiency of adaptation, we vary the number of LR sequences used to optimize the gating network (Fig. 8e). The error decreases sharply as the number of sequences increases from 5 to 10, with only small improvements thereafter. This trend indicates that adaptation can be achieved with limited unpaired LR data under unseen coarse-graining. The low correlation between the box and spectral reconstruction errors further supports the use of these experts as complementary sources of information (Fig. 8f; Supplementary Note 3.8).

For the paired LES–DNS test in DIT, a DNS velocity field is filtered and downsampled to initialize the LES, which subsequently advances independently with an SGS closure (Fig. 9a). LES inputs and DNS references are paired at matching physical times. This construction captures the combined efects of limited spatial resolution, SGS modeling, and numerical evolution encountered in coarse-grid simulations. The DNS simulation and LES–DNS pairing are described in Supple mentary Notes 2.7 and 2.8.

A central challenge is the chaotic sensitivity of turbulent trajectories to perturbations. SGSmodel and discretization errors can be amplified by nonlinear evolution, progressively separating the LES trajectory from the corresponding DNS trajectory [27, 55]. Reconstruction must therefore accommodate an evolving dynamical mismatch in addition to recovering unresolved spatial content. With all experts frozen, we adapt the gating network without HR supervision using only 10 LES patches. The MoE consistently achieves the lowest reconstruction nRMSE throughout the LES evolution (Fig. 9c). The MoE also better recovers the fine-scale vortical structures of the DNS field (Fig. 9b). These improvements extend to the physical processes of energy dissipation and interscale transfer. Using the dissipation diagnostic introduced in Section 2.5, the MoE maintains a normalized mean dissipation, $\langle \varepsilon _ { \mathrm { S R } } \rangle / \langle \varepsilon _ { \mathrm { D N S } } \rangle$ , close to unity, while the Box and Gaussian experts overestimate dissipation and the Spectral expert underestimates it (Fig. 9d). This agreement is consistent with the MoE more faithfully recovering the small-scale velocity gradients responsible for viscous dissipation.

We further examine the local energy flux, $\Pi _ { \ell } = - \tau _ { i j } ^ { ( \ell ) } \bar { S } _ { i j } ^ { ( \ell ) }$ , which characterizes kinetic-energy transfer across scale ℓ. Here, $\tau _ { i j } ^ { ( \ell ) }$ and $\bar { S } _ { i j } ^ { ( \ell ) }$ denote the subfilter stress and filtered strain-rate tensor, respectively. The MoE yields the lowest relative flux-field error at every examined filter scale (Fig. 9e), indicating improved recovery of the magnitude and spatial distribution of multiscale energy transfer. Together, these results demonstrate that the MoE enables data-eficient generalization to unseen coarse-graining and SGS-modeled LES inputs without HR supervision.

a  
![](images/1a5c7a539744f2a813cf9e8beeca451312b06a8b46d068221e9fc95351944db1.jpg)

![](images/b208308beee5bace4d1f0e699e3986b2bca4c580f622dc044e01bbf18ded70ef.jpg)

b  
![](images/fadca49d6e5dec6d0eff4bf3a6b49fc0a76323dc67e5d696e37b5b38fc8aaa86.jpg)

![](images/08789a9b58f374d90296e1681f5eba106cd2db5b7f0f2fe0bfac2936012d1339.jpg)

![](images/1d51c492845b9a9780a0f9ae3c4e0be2efefa240369851b515c0dee7658234ad.jpg)

![](images/05e8e9498d05c521803ce0072e87de540579b4178eae791caa26dd0c1cca8bb9.jpg)  
Figure 9: Generalization to LES inputs in DIT. a, A filtered and downsampled DNS state initializes an independently evolved LES with an SGS closure. LES and DNS fields are paired at matching physical times, separated by $\Delta T = 2 0 \delta t$ , where δt is the LES integration time step. b, Vorticity magnitude ω at LES step 160 for DNS, LES, the individual experts and the MoE, with reconstruction error maps beneath the corresponding predictions. $\mathbf { c } ,$ Reconstruction nRMSE against the paired DNS references over 160 LES steps. d, Spatially averaged viscous dissipation normalized by the DNS value at each corresponding time; the dashed line denotes unity. e, Relative $L _ { 2 }$ error of the local energy flux field, $\lVert \Pi _ { \ell } - \Pi _ { \ell } ^ { \mathrm { D N S } } \rVert _ { 2 } / \lVert \Pi _ { \ell } ^ { \mathrm { D N S } } \rVert$ <sub>2</sub>, aggregated over all evaluated times and spatial locations. Filter width ℓ is normalized by the DNS grid spacing ∆x.

## 3 Discussion

This work presents a turbulence-aware deep learning framework for super-resolution reconstruction of time-resolved three-dimensional flows from coarse-grained inputs. The framework is designed to improve numerical fidelity while better preserving turbulence-relevant spatiotemporal structure. The framework combines a multiscale neural operator with a state space model to reconstruct spatial structures and their temporal evolution, while Navier–Stokes residuals provide physical regularization during training. An MoE mechanism further enables adaptation to unknown coarse-graining using LR data alone.

The framework achieves lower reconstruction errors than the compared learning-based baselines across FIT, MHDT, and HBDT under box/Favre filtering, Gaussian filtering, and spectral cutof, with nRMSE reductions reaching approximately 30% relative to the best-performing baseline. Energy spectra, PDFs, and joint distributions of velocity-gradient invariants further show that the reconstructed fields recover multiscale energy distributions and key features of turbulence statistics. The evolution of vortical structures and space–time patterns demonstrates improved temporal coherence relative to the baselines. In DIT, physical regularization improves agreement with the reference temporal evolution and mean viscous dissipation, promoting dynamical consis tency under non-stationary conditions. Results on ERA5 atmospheric reanalysis data show that these improvements carry over to geophysical flows. Under the tested unseen coarse-graining settings and for independently evolved LES inputs, the MoE achieves lower reconstruction errors than the individual experts, with adaptation using only unpaired LR data.

Despite these promising results, several limitations warrant future work. First, the framework provides a single deterministic reconstruction for each LR sequence and does not quantify the vari ability among compatible HR realizations. Agreement with reference fields and selected turbulence statistics therefore does not establish recovery of the full conditional distribution. Conditional generative approaches [18, 42, 56] ofer a possible way to represent this variability, but may introduce additional training and sampling costs, particularly when iterative sampling and multiple realiza tions are required. Such overhead could compromise the computational eficiency that motivates the present framework. Future work could therefore explore computationally eficient approaches to uncertainty quantification that complement the present deterministic framework. Second, generalization across physically distinct turbulent systems remains a fundamental challenge. Turbulent flows exhibit substantial regime-to-regime variability in dynamics, scale interactions, and statistics, making universal generalization dificult without exposure to diverse physical regimes during training. One possible path forward is large-scale pretraining on substantially broader and more heterogeneous turbulence datasets [57]. In practice, however, such datasets remain expensive and dificult to obtain, particularly for high-fidelity three-dimensional turbulence [58, 59]. An alternative direction is the development of hybrid approaches that tightly integrate data-driven models with physical structure or constraints, thereby reducing reliance on exhaustive multi-regime data coverage [57, 60].  Third, extending the framework to unstructured and curvilinear meshes in complex engineering flows presents additional challenges associated with mesh geometry and spatial discretization [61]. Addressing these challenges warrants dedicated investigation for specific flow configurations and forms an important direction for future work. Fourth, the framework lever ages knowledge of governing equations to define physical constraints, which may be less efective when the governing dynamics are partially unknown. A promising alternative is to derive efective governing equations from data, and incorporate the inferred dynamics back into the learning process [62]. Fifth, training multiple experts increases the one-time training cost; however, experts are trained once and then frozen, while unsupervised adaptation requires optimizing only a lightweight gating network without HR supervision. Finally, the proposed framework serves as a post-processing component that complements high-fidelity simulations by reconstructing HR fields from coarse-grained data (e.g., LES or coarse-grid simulations; Section 2.7). By enhancing the efective resolution of existing numerical data, this approach reduces the reliance on prohibitively expensive high-fidelity simulations or measurements when detailed flow information is required. In this way, the framework complements conventional numerical pipelines and provides a practical pathway toward accessing HR turbulence statistics at reduced computational cost.

## 4 Methods

This section presents the methodology for time-resolved volumetric turbulence super-resolution, addressing the challenge of recovering high-fidelity dynamics from coarse-grained inputs. The proposed framework integrates four complementary components: (i) a state space model to capture temporal evolution on Eulerian grids,  (ii) a multiscale neural operator to model cross-scale spatial structure interactions,  (iii) Navier–Stokes-based soft constraints to encourage dynamical consis tency, and (iv) an unsupervised MoE mechanism to maintain robustness against unknown coarsegraining. Together, these components form a unified framework for high-fidelity reconstruction of turbulent flow dynamics. The following subsections describe these components, the evaluation metrics, the datasets, and the baseline methods.

## 4.1 State space model

Turbulent flows exhibit temporal dependencies, where the evolution of flow structures depends on historical states. To capture this coherent temporal evolution for turbulence, we adopt a state space model [63] as the core mechanism for temporal modeling. A state space model provides a principled strategy for capturing temporal dependencies and has been adopted in a variety of sequence modeling tasks [64–66]. In its standard form, it maps an input sequence $\{ \pmb { x } _ { t } \} _ { t }$ to outputs $\{ { \pmb y } _ { t } \} _ { }$ <sub>t</sub> via a latent state $h _ { t } \mathbf { \cdot }$

$$
\begin{array} { l } { { \pmb { h } } _ { t } = { \pmb { A } } { \pmb { h } } _ { t - 1 } + { \pmb { B } } { \pmb { x } } _ { t } , } \\ { { \pmb { y } } _ { t } = { \pmb { C } } { \pmb { h } } _ { t } + { \pmb { D } } { \pmb { x } } _ { t } , } \end{array}\tag{3}
$$

where $A , B , C , D$ are learnable parameters.

To account for the intermittent and spatially varying dynamics of three-dimensional turbulence [67], we extend the state space model with an input-conditioned parameterization [68]. The input features dynamically modulate the system matrices B and $C ,$ allowing the model to adapt its state updates and readout to evolving flow patterns. Specifically, we apply the state space model on the 3D Eulerian $\mathrm { g r i d }$ by maintaining a local latent state $\pmb { h } _ { t } ( \pmb { r } ) \in \mathbb { R } ^ { M }$ at each spatial coordinate $\pmb { r } = ( x , y , z )$ . Given a latent representation $z _ { t }$ from the lifting mapping $P _ { \mathrm { : } }$ , we first apply a 3D convolution (conv3d) to ${ \boldsymbol { z } } _ { t }$ for local spatial aggregation. The latent state then evolves independently over time at each spatial location r:

$$
\begin{array} { l } { { \pmb { h } } _ { t } ( { \pmb { r } } ) = { \pmb { A } } { \pmb { h } } _ { t - 1 } ( { \pmb { r } } ) + { \pmb { B } } _ { t } ( { \pmb { r } } ) { \pmb { z } } _ { t } ( { \pmb { r } } ) , } \\ { { \pmb { y } } _ { t } ( { \pmb { r } } ) = { \pmb { C } } _ { t } ( { \pmb { r } } ) { \pmb { h } } _ { t } ( { \pmb { r } } ) + { \pmb { D } } { \pmb { z } } _ { t } ( { \pmb { r } } ) , } \end{array}\tag{4}
$$

where D serves as an input-independent skip connection that preserves immediate signal information. The parameter A governs the temporal memory of the system, while the coeficients $B _ { t }$ and $C _ { t }$ are dynamically modulated by $z _ { t }$ to adapt to local flow variations. To enhance representational capability, we apply a spatially varying selection gate to $\mathbf { \nabla } \mathbf { \mathbf { \mathit { y } } } _ { t } .$ , yielding the feature $\mathbf { \delta } _ { \mathbf { \boldsymbol { s t } } }$ that serves as the input for the multiscale neural operator. Further details are provided in Supplementary Note 1.4.

## 4.2 Multiscale neural operator

Modeling the multiscale physics of turbulence requires capturing both global correlations $( e . g .$ large-scale coherent structures) and local variations $( e . g . ,$ small dissipative eddies) [69]. Standard convolutional networks, while eficient for local patterns, often struggle to model long-range dependencies due to their limited efective receptive fields [70]. Conversely, spectral operators ofer global receptive fields but may lack the inductive bias needed for precise local feature refinement [40]. To synergize these complementary strengths, we employ a stack of K hybrid blocks that couple a spectral operator for large-scale structure modeling with a convolutional operator for local refinement. Each hybrid block is defined as:

$$
\hat { \pmb { s } } _ { t } = \pmb { s } _ { t } + \mathcal { O } _ { c } ( \mathcal { O } _ { s } ( \pmb { s } _ { t } ) ) ,\tag{5}
$$

where $\mathbf { \boldsymbol { s } } _ { t }$ denotes the temporally evolved feature yielded by the state space model, and $\mathcal { O } _ { s }$ and $\mathcal { O } _ { c }$ provide global spectral coupling and local convolutional refinement, respectively.  For the global coupling $\mathcal { O } _ { s } .$ we adopt the Adaptive Fourier Neural Operator [44]. By performing global spatia coupling in the frequency domain, this operator enjoys a global receptive field, allowing it to eficiently capture long-range interactions across the entire domain:

$$
\mathcal { O } _ { s } ( \pmb { \mathscr { s } } _ { t } ) : = \mathcal { F } ^ { - 1 } \big ( S _ { \lambda } \big ( \mathrm { M L P } ( \mathcal { F } ( \pmb { \mathscr { s } } _ { t } ) ) \big ) \big ) + \pmb { \mathscr { s } } _ { t } ,\tag{6}
$$

where $\mathcal { F }$ denotes the 3D discrete Fourier transform and ${ \mathcal { F } } ^ { - 1 }$ its inverse operation. The soft thresholding operation $S _ { \lambda }$ applies element-wise shrinkage in the spectral domain:

$$
S _ { \lambda } ( \pmb { x } ) : = \frac { \pmb { x } } { | \pmb { x } | } \odot \operatorname* { m a x } \bigl ( | \pmb { x } | - \lambda , 0 \bigr ) ,\tag{7}
$$

with threshold $\lambda$ controlling the amount of shrinkage. This shrinkage regularizes small-magnitude spectral coeficients and improves stability in practice. However, relying solely on spectral operations, especially with sparsity constraints, may risk smoothing out fine-scale turbulent fluctuations. To refine these local details, we parameterize the convolution operator $\mathcal { O } _ { c }$ as:

$$
\mathcal { O } _ { c } ( \pmb { \mathscr { s } } _ { t } ) : = W _ { 2 } \big ( \mathrm { C o n v } _ { \mathrm { d e p } } \big ( \phi ( W _ { 1 } \pmb { \mathscr { s } } _ { t } ) \big ) \big ) ,\tag{8}
$$

where $W _ { 1 }$ and $W _ { 2 }$ are point-wise linear projections, $\phi ( \cdot )$ is a nonlinear activation, and $\mathrm { C o n v _ { d e p } }$ denotes a 3D depthwise convolution. This design supplies a local inductive bias for spatial feature refinement. The contributions of the two operators across spatial scales are evaluated through component-removal experiments in Supplementary Note 3.3.

## 4.3 Physics-informed neural optimization

We use a voxel-wise $L _ { 1 }$ loss to supervise reconstruction from paired LR–HR data. Minimizing the expected absolute error yields componentwise conditional medians [71], providing a statistical basis for this deterministic point-estimation objective. The linear growth of the $L _ { 1 }$ penalty limits the influence of large reconstruction errors compared with a squared-error penalty. We combine this data term with diferentiable physical penalties that act as soft constraints and encourage consistency with the governing equations. The total training objective is defined as

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { m a e } } + \lambda _ { \mathrm { p h y } } \mathcal { L } _ { \mathrm { p h y } } ,\tag{9}
$$

where $\mathcal { L } _ { \mathrm { m a e } }$ measures voxel-wise reconstruction error between the prediction $\mathcal { M } _ { \theta } ( \mathbf { \boldsymbol { X } } _ { \mathrm { { L } } } )$ and the HR target $X _ { \mathrm { H } }$ , and $\mathcal { L } _ { \mathrm { p h y } }$ penalizes violations of the governing equations with a weighting factor $\lambda _ { \mathrm { p h y } }$ The physical loss combines a momentum term with a divergence-related constraint:

$$
\mathcal { L } _ { \mathrm { p h y } } = \beta \mathcal { L } _ { \mathrm { m o m } } + \| \mathcal { R } _ { \mathrm { d i v } } \| _ { 1 } ,\tag{10}
$$

where $\beta$ controls the relative strength of the momentum regularizer. The loss weights and their sensitivity are reported in Supplementary Notes 1 and 1.2. We use $L _ { 1 }$ penalties for the physical residuals because their linear growth limits the influence of large local residuals compared with

squared $L _ { 2 }$ penalties. The residuals and corresponding momentum penalties are defined below for each flow system.

FIT. We define the source-free momentum defect and divergence residual as

$$
\begin{array} { l } { { \displaystyle { \mathcal R } _ { 0 } ^ { \mathrm { F I T } } = \frac { \partial { \pmb u } } { \partial t } + ( { \pmb u } \cdot \nabla ) { \pmb u } + \frac { 1 } { \rho } \nabla p - \nu \nabla ^ { 2 } { \pmb u } , } } \\ { { \displaystyle { \mathcal R } _ { \mathrm { d i v } } = \nabla \cdot { \pmb u } . } } \end{array}\tag{11}
$$

Here, u denotes the velocity field, $p$ the pressure, and ν the kinematic viscosity. ∇ denotes the spatial gradient operator.

MHDT. The source-free momentum defect additionally includes the Lorentz-force contribution, and both velocity and magnetic-field divergence enter the solenoidal constraint:

$$
\begin{array} { r } { \mathcal { R } _ { 0 } ^ { \mathrm { M H D T } } = \displaystyle \frac { \partial \pmb { u } } { \partial t } + ( \pmb { u } \cdot \nabla ) \pmb { u } + \nabla p - \nu \nabla ^ { 2 } \pmb { u } - ( \nabla \times \pmb { b } ) \times \pmb { b } , } \\ { \mathcal { R } _ { \mathrm { d i v } } = ( \nabla \cdot \pmb { u } , \nabla \cdot \pmb { b } ) . } \end{array}\tag{12}
$$

Here, $p = { p _ { \mathrm { t h } } } / { \rho _ { 0 } }$ denotes the kinematic pressure, with $p _ { \mathrm { t h } }$ the thermodynamic pressure and $\rho _ { 0 }$ the constant mass density. The magnetic field b is expressed in Alfvén velocity units.

For FIT and MHDT, we apply a spatial high-pass filter $\mathcal { H } _ { \sigma }$ to the momentum residual before evaluating the loss:

$$
\mathcal { L } _ { \mathrm { m o m } } = \| \mathcal { H } _ { \sigma } [ \mathcal { R } _ { 0 } ] \| _ { 1 } .\tag{13}
$$

The complete momentum residual is $\mathcal { R } _ { 0 } - f$ , so omitting external forcing changes the filtered residual by $\mathcal { H } _ { \sigma } [ f ]$ . In both datasets, forcing is confined to low wavenumbers [28, 29], and its direct contribution is strongly attenuated by the high-pass filter. This yields a scale-selective momentum penalty that emphasizes finer spatial scales. The nonlinear terms are evaluated on the full reconstructed fields before filtering, so interactions across scales remain included. A comparison on FIT supports this formulation: the high-pass $L _ { 1 }$ momentum loss yields lower high-wavenumber spectral error than both unfiltered $L _ { 1 }$ and high-pass squared $L _ { 2 }$ penalties (Supplementary Fig. 2). Supplementary Note 1.6 provides the filter implementation, quantifies the remaining forcing contribution, and reports the loss comparison.

HBDT. The residuals follow the variable-density, low-Mach-number Navier–Stokes formulation. The momentum residual includes the density-dependent inertial and viscous terms as well as the buoyancy contribution:

$$
\begin{array} { c } { \displaystyle \mathcal { R } _ { \mathrm { m o m } } = \frac { \partial ( \rho \pmb { u } ) } { \partial t } + \nabla \cdot ( \rho \pmb { u } \otimes \pmb { u } ) + \nabla p - \nabla \cdot \pmb { \tau } - \rho \pmb { g } , } \\ { \displaystyle \mathcal { R } _ { \mathrm { d i v } } = \frac { \partial \rho } { \partial t } + \nabla \cdot ( \rho \pmb { u } ) . } \end{array}\tag{14}
$$

Here, $\rho$ is the variable density, g the gravitational acceleration, and τ the viscous stress tensor. The momentum penalty is evaluated without filtering, $\mathcal { L } _ { \mathrm { m o m } } = \| \mathcal { R } _ { \mathrm { m o m } } \| _ { 1 }$ DIT. DIT is unforced, so its momentum residual has the same form as $\mathcal { R } _ { 0 } ^ { \mathrm { F I T } }$ , with $\mathcal { R } _ { \mathrm { d i v } } = \nabla \cdot \pmb { u }$ The momentum penalty is evaluated without filtering, $\mathcal { L } _ { \mathrm { m o m } } = \| \mathcal { R } _ { \mathrm { m o m } } \| _ { 1 }$

The residual penalties above encourage consistency with the governing equations during training. Incompressibility can also be enforced directly by projecting the reconstructed velocity onto the divergence-free subspace. Theory-guided hard constraint projection (HCP) [72] provides a general formulation for this approach: it maps a prediction onto a constraint set defined by discretized physical equations while minimizing the change to that prediction.

For this dataset, we also evaluate HCP as an optional hard incompressibility constraint. The projection acts on the reconstructed physical velocity v, after the output mapping Q and inverse normalization, and returns the closest velocity satisfying the discrete continuity equation:

$$
\begin{array} { r } { \pmb { u } ^ { \mathrm { H } } = \pmb { v } - \mathsf { D } _ { h } ^ { \top } ( \mathsf { D } _ { h } \mathsf { D } _ { h } ^ { \top } ) ^ { \dag } \mathsf { D } _ { h } \pmb { v } , } \end{array}\tag{15}
$$

where $\mathsf { D } _ { h }$ is the discrete divergence operator and † denotes the Moore–Penrose pseudoinverse. The projection removes only the component of v orthogonal to the divergence-free subspace and introduces no learnable parameters. When HCP is used, the reconstruction loss is evaluated on the projected velocity $\pmb { u } ^ { \mathrm { H } }$ . Supplementary Note 1.7 provides the derivation and a comparison of models trained with and without HCP.

## 4.4 Unsupervised generalization with mixture-of-experts

In practical scenarios, the coarse-graining process that generates low-fidelity turbulent flows can vary across simulations, sensors, and numerical settings, and is often unknown at inference time. To improve robustness under such coarse-graining shifts, we propose an unsupervised MoE mechanism [73] that enables adaptation to unseen coarse-graining using only LR inputs.

Given N representative coarse-graining strategies, N paired training sets are constructed and one expert model is trained per strategy, yielding pretrained experts $\{ \mathcal { M } _ { \pmb { \theta } _ { n } } \} _ { n = 1 } ^ { N }$ . After training, all expert parameters are frozen. Adaptation to unknown coarse-graining is then achieved by learning a gating network $\mathcal { G } _ { \psi }$ that combines these frozen experts, as freezing the experts and adapting only the gating weights provides a stable and data-eficient way to handle unknown coarse-graining when HR supervision is unavailable. Specifically, $\mathcal { G } _ { \psi }$ maps an LR input sequence $X _ { \mathrm { { L } } }$ to a set of spatially varying mixture weights

$$
\begin{array} { r } { \pmb { \alpha } ( \pmb { X } _ { \mathrm { L } } ) = \mathrm { s o f t m a x } ( \mathcal { G } _ { \psi } ( \pmb { X } _ { \mathrm { L } } ) ) , } \end{array}\tag{16}
$$

where ${ \pmb \alpha } ( { \pmb X } _ { \mathrm { L } } ) \in \mathbb { R } ^ { H _ { \mathrm { H } } \times W _ { \mathrm { H } } \times D _ { \mathrm { H } } \times N }$ denotes a voxel-wise weight field satisfying $\begin{array} { r } { \sum _ { n = 1 } ^ { N } \alpha _ { n } ( \pmb { r } ) = 1 } \end{array}$ and $\alpha _ { n } ( \pmb { r } ) \ge 0$ at each spatial location r. The final HR reconstruction is obtained as a combination of expert predictions at each voxel:

$$
\begin{array} { r } { \hat { \boldsymbol X } _ { \mathrm { H } } ( \pmb r ) = \displaystyle \sum _ { n = 1 } ^ { N } \alpha _ { n } ( \pmb r ; \pmb X _ { \mathrm { L } } ) \mathcal { M } _ { \pmb \theta _ { n } } ( \pmb X _ { \mathrm { L } } ) ( \pmb r ) . } \end{array}\tag{17}
$$

The gating network $\mathcal { G } _ { \psi }$ is optimized at inference time without access to HR supervision by minimizing the physics-residual objective defined in Eq. (10), encouraging the combined reconstruction to satisfy the governing equations while remaining compatible with the observed LR input. In practice, the unsupervised adaptation requires only a small number of unlabeled LR sequences (see Fig. 8e).

## 4.5 Evaluation metrics

We assess reconstruction quality using complementary measures of pointwise accuracy, structural similarity, temporal consistency, and multiscale turbulence statistics. Pointwise metrics quantify agreement with the paired HR reference. Energy spectra, PDFs, and velocity-gradient diagnostics characterize the scale-dependent and statistical properties of the reconstructed fields.

We report nRMSE, PSNR, and 3D SSIM, following standard definitions in the super-resolution literature [24, 38]. The definitions of PSNR and SSIM are provided in Supplementary Note 1.9. The nRMSE measures the relative error between the prediction $\hat { X }$ and the ground truth X:

$$
\mathrm { n R M S E } ( \hat { { \mathbf { X } } } , { \pmb { X } } ) = \frac { \| \hat { { \mathbf { X } } } - { \pmb { X } } \| _ { 2 } } { \| { \pmb { X } } \| _ { 2 } } .\tag{18}
$$

To probe temporal consistency, we compute temporal RMSE (tRMSE) on frame-to-frame increments:

$$
\mathrm { t R M S E } = \sqrt { \frac { 1 } { T - 1 } { \sum _ { t = 2 } ^ { T } } { \| \Delta \hat { X } _ { t } - \Delta X _ { t } \| _ { 2 } ^ { 2 } } } ,\tag{19}
$$

where $\Delta X _ { t } = X _ { t } - X _ { t - 1 }$ . This metric measures agreement in frame-to-frame changes.

We also compute Fourier-space RMSE (fRMSE) between the three-dimensional Fourier coeficients of $\hat { X }$ and X (Supplementary Note 1.9). To assess the distribution of energy across scales, we compare shell-integrated energy spectra $E ( k )$ . For FIT, MHDT, and HBDT, we use kinetic energy, total kinetic and magnetic energy (with the magnetic field in Alfvén units), and density-weighted kinetic energy, respectively (Supplementary Note 1.9).

## 4.6 Datasets

TurbSR3D. We construct a time-resolved volumetric turbulence super-resolution dataset, termed TurbSR3D, comprising over 3,500 samples (310 GB).  High-fidelity data are drawn from canonical simulations in the Johns Hopkins Turbulence Databases (JHTDB) [28–30, 74].  LR inputs are obtained by applying the coarse-graining operator D (Eq. (1)) at factors of 4×, 8×, or 16×. The dataset covers three representative turbulence regimes: forced isotropic turbulence, forced magnetohydrodynamic turbulence, and homogeneous buoyancy-driven turbulence.

DIT. We provide an additional dataset of decaying isotropic turbulence to evaluate physical constraints and generalization to LES inputs under non-stationary conditions. For the LES evaluation, independently evolved LES fields are paired with DNS references at matching physical times. Details of the DNS and LES simulations are provided in Supplementary Notes 2.7 and 2.8, respectively. ERA5. An atmospheric downscaling benchmark is constructed from the ERA5 reanalysis [31]. ERA5 data span 10 consecutive days in January 2022 (January 1–10), sampled every 6 hours. The benchmark covers all longitudes between $7 5 \mathrm { { } ^ { \circ } S }$ and $7 5 \mathrm { { } ^ { \circ } N }$ and includes temperature and horizontal wind components $( u , v )$ at eight pressure levels from 1000 to 300 hPa. LR inputs are generated by bicubic downsampling of the HR fields on the horizontal grid at factors of 4× and 8×. The first six days are used for training, the next two for validation, and the final two for testing.

## 4.7 Baselines

We evaluate our framework against well-established baselines in two categories: i) classical numerical interpolation methods, i.e., tricubic and bicubic interpolation, and ii) state-of-the-art deep learning models, i.e., RRDB [37], EDSR [38], RCAN [39], FNO [40], PhySR [41], SwinIR [43] and DifSR [42]. Tricubic interpolation is one of the traditional super-resolution methodologies based on the filter operation [75], typically serving as a baseline in turbulence reconstruction studies [9, 16]. Bicubic interpolation, a widely used 2D interpolation method, is also commonly applied in atmospheric downscaling tasks [25]. EDSR [38], RRDB [37], RCAN [39] are three representative neural network architectures in the image super-resolution domain. FNO [40] enables learning PDE solution operators via directly parameterizing the integral kernel in frequency space, rendering it an ideal choice for modeling physical systems. PhySR [41] is a physics-informed super-resolution framework designed for modeling HR response of spatiotemporal data. SwinIR [43] is a transformerbased model that has demonstrated strong performance in image super-resolution tasks and serves as a robust baseline for comparison. DifSR [42] employs difusion-based generative modeling for three-dimensional turbulence super-resolution and serves as a representative generative baseline. More details are provided in Supplementary Note 1.5.

## References

[1] Bratanov, V., Jenko, F., Frey, E.: New class of turbulence in active fluids. Proceedings of the National Academy of Sciences 112(49), 15048–15053 (2015) https://doi.org/10.1073/pn as.1509304112

[2] Guo, Y., Hao, Y., Wan, S., Zhang, H., Zhu, L., Zhang, Y., Wu, J., Dai, Q., Fang, L.: Direct observation of atmospheric turbulence with a video-rate wide-field wavefront sensor. Nature Photonics 18(9), 935–943 (2024)

[3] Durham, W.M., Climent, E., Barry, M., De Lillo, F., Bofetta, G., Cencini, M., Stocker, R.: Turbulence drives microscale patches of motile phytoplankton. Nature communications 4(1), 2148 (2013)

[4] Jin, D., Chen, Y., Lu, Y., Chen, J., Wang, P., Liu, Z., Guo, S., Bai, X.: Neutralizing the impact of atmospheric turbulence on complex scene imaging via deep learning. Nature Machine Intelligence 3(10), 876–884 (2021)

[5] Luther, J., Jenny, P.: Non-linear super-stencils for turbulence model corrections. Communi cations Physics 8, 233 (2025) https://doi.org/10.1038/s42005-025-02149-3

[6] Venturi, D., Karniadakis, G.E.: Gappy data and reconstruction procedures for flow past a cylinder. Journal of Fluid Mechanics 519, 315–336 (2004) https://doi.org/10.1017/S002 2112004001338

[7] Willcox, K.: Unsteady flow sensing and estimation via the gappy proper orthogonal decomposition. Computers & Fluids 35(2), 208–226 (2006) https://doi.org/10.1016/j.compflui d.2004.11.006

[8] Li, Z., Han, W., Zhang, Y., et al.: Learning spatiotemporal dynamics with a pretrained generative model. Nature Machine Intelligence 6, 1566–1579 (2024) https://doi.org/10.1 038/s42256-024-00938-z

[9] Fukami, K., Fukagata, K., Taira, K.: Super-resolution reconstruction of turbulent flows with machine learning. Journal of Fluid Mechanics 870, 106–120 (2019) https://doi.org/10.101 7/jfm.2019.238

[10] Xu, Q., Zhuang, Z., Pan, Y., Wen, B.: Super-resolution reconstruction of turbulent flows with a transformer-based deep learning framework. Physics of Fluids 35(5), 055130 (2023) https://doi.org/10.1063/5.0149551

[11] Zeng, H., Xu, Q., Qin, Z., Wen, B.: A swin-transformer-based model for super-resolution reconstruction of turbulent flows. Physics of Fluids 37(7) (2025)

[12] Nista, L., Pitsch, H., Schumann, C.D.K., Bode, M., Grenga, T., MacArt, J.F., Attili, A.: Influence of adversarial training on super-resolution turbulence reconstruction. Phys. Rev. Fluids 9, 064601 (2024) https://doi.org/10.1103/PhysRevFluids.9.064601

[13] Kim, H., Kim, J., Won, S., Lee, C.: Unsupervised deep learning for super-resolution reconstruction of turbulence. Journal of Fluid Mechanics 910, 29 (2021) https://doi.org/10.1 017/jfm.2020.1028

[14] Xie, Y., Franz, E., Chu, M., Thuerey, N.: tempogan: A temporally coherent, volumetric gan for super-resolution fluid flow. ACM Transactions on Graphics (TOG) 37(4), 95 (2018)

[15] Deng, Z., He, C., Liu, Y., Kim, K.C.: Super-resolution reconstruction of turbulent velocity fields using a generative adversarial network-based artificial intelligence framework. Physics of Fluids 31(12), 125111 (2019) https://doi.org/10.1063/1.5127031

[16] Shu, D., Li, Z., Barati Farimani, A.: A physics-informed difusion model for high-fidelity flow field reconstruction. Journal of Computational Physics 478, 111972 (2023) https://doi.or g/10.1016/j.jcp.2023.111972

[17] Shan, S., Wang, P., Chen, S., Liu, J., Xu, C., Cai, S.: PiRD: Physics-informed Residual Difusion for Flow Field Reconstruction (2024)

[18] Du, P., Parikh, M.H., Fan, X., et al.: Conditional neural field latent difusion model for generating spatiotemporal turbulence. Nature Communications 15, 10416 (2024) https://do i.org/10.1038/s41467-024-54712-1

[19] Raissi, M., Perdikaris, P., Karniadakis, G.E.: Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal of Computational Physics 378, 686–707 (2019) https://doi.org/10.101 6/j.jcp.2018.10.045

[20] Karniadakis, G.E., Kevrekidis, I.G., Lu, L., Perdikaris, P., Wang, S., Yang, L.: Physicsinformed machine learning. Nature Reviews Physics 3(6), 422–440 (2021) https://doi.org/ 10.1038/s42254-021-00314-5

[21] Meng, C., Griesemer, S., Cao, D., Seo, S., Liu, Y.: When physics meets machine learning: a survey of physics-informed machine learning. Machine Learning for Computational Science and Engineering 1(1), 20 (2025) https://doi.org/10.1007/s44379-025-00016-0

[22] Xu, Y., Shao, Z., Liu, X., et al.: AI for science: Progress, challenges, and perspectives. The Innovation, 101530 (2026) https://doi.org/10.1016/j.xinn.2026.101530

[23] Kochkov, D., Smith, J.A., Alieva, A., Wang, Q., Brenner, M.P., Hoyer, S.: Machine learningaccelerated computational fluid dynamics. Proceedings of the National Academy of Sciences 118(21), 2101784118 (2021) https://doi.org/10.1073/pnas.2101784118

[24] Chung, W.T., Akoush, B., Sharma, P., Tamkin, A., Jung, K.S., Chen, J., Guo, J., Brouzet, D., Talei, M., Savard, B., Poludnenko, A.Y., Ihme, M.: Turbulence in focus: Benchmarking scaling behavior of 3d volumetric super-resolution with BLASTNet 2.0 data. In: Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track (2023)

[25] Ren, P., Erichson, N.B., Guo, J., Subramanian, S., San, O., Lukic, Z., Mahoney, M.W.: SuperBench: A Super-Resolution Benchmark Dataset for Scientific Machine Learning (2025)

[26] Fukami, K., Fukagata, K., Taira, K.: Machine-learning-based spatio-temporal super resolution reconstruction of turbulent flows. Journal of Fluid Mechanics 909, 9 (2021)

[27] Pope, S.B.: Turbulent Flows. Cambridge University Press, Cambridge, UK (2000)

[28] Li, Y., Perlman, E., Wan, M., Yang, Y., Meneveau, C., Burns, R., Chen, S., Szalay, A., Eyink, G.: A public turbulence database cluster and applications to study Lagrangian evolution of velocity increments in turbulence. Journal of Turbulence 9(31) (2008)

[29] Eyink, G., Vishniac, E., Lalescu, C., Aluie, H., Kanov, K., Bürger, K., Burns, R., Meneveau, C., Szalay, A.: Flux-freezing breakdown in high-conductivity magnetohydrodynamic turbulence. Nature 497(7450), 466–469 (2013)

[30] Livescu, D., Canada, C., Kanov, K., Burns, R., IDIES staf, Pulido, J.: Homogeneous Buoyancy driven turbulence data set. http://turbulence.idies.jhu.edu/docs/hbdt/README-H BDT.pdf. Available online (2014)

[31] Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horányi, A., Muñoz-Sabater, J., Nicolas, J., Peubey, C., Radu, R., Schepers, D., Simmons, A., Soci, C., Abdalla, S., Abellan, X., Balsamo, G., Bechtold, P., Biavati, G., Bidlot, J., Bonavita, M., De Chiara, G., Dahlgren, P., Dee, D., Diamantakis, M., Dragani, R., Flemming, J., Forbes, R., Fuentes, M., Geer, A., Haimberger, L., Healy, S., Hogan, R.J., Hólm, E., Janisková, M., Keeley, S., Laloyaux, P., Lopez, P., Lupu, C., Radnoti, G., Rosnay, P., Rozum, I., Vamborg, F., Villaume, S., Thépaut, J.-N.: The era5 global reanalysis. Quarterly Journal of the Royal Meteorological Society 146(730), 1999–2049 (2020) https://doi.org/10.1002/qj.3803

[32] Rudd, R.E., Broughton, J.Q.: Coarse-grained molecular dynamics and the atomic limit of finite elements. Physical review B 58(10), 5893 (1998)

[33] Ten Cate, A., Derksen, J.J., Portela, L.M., Van Den Akker, H.E.: Fully resolved simulations of colliding monodisperse spheres in forced isotropic turbulence. Journal of Fluid Mechanics 519, 233–271 (2004)

[34] Carroll, P.L., Blanquart, G.: A proposed modification to lundgren’s physical space velocity forcing method for isotropic turbulence. Physics of Fluids 25(10) (2013)

[35] Bauer, F., Tardu, S., Doche, O.: Eficiency of high accuracy drp schemes in direct numerical simulations of incompressible turbulent flows. Computers & Fluids 107, 123–140 (2015)

[36] Bai, K., Wang, C., Desbrun, M., Liu, X.: Predicting high-resolution turbulence details in space and time. ACM Trans. Graph. 40(6) (2021) https://doi.org/10.1145/3478513.3480492

[37] Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., Qiao, Y., Loy, C.C.: Esrgan: Enhanced super-resolution generative adversarial networks. In: Proceedings of the European Conference on Computer Vision (ECCV) Workshops (2018)

[38] Lim, B., Son, S., Kim, H., Nah, S., Lee, K.M.: Enhanced deep residual networks for single image super-resolution. In: 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 1132–1140 (2017). https://doi.org/10.1109/CVPRW.2017.151

[39] Zhang, Y., Li, K., Li, K., Wang, L., Zhong, B., Fu, Y.: Image super-resolution using very deep residual channel attention networks. In: Proceedings of the European Conference on Computer Vision (ECCV), pp. 286–301 (2018)

[40] Li, Z., Kovachki, N.B., Azizzadenesheli, K., liu, B., Bhattacharya, K., Stuart, A., Anandkumar, A.: Fourier neural operator for parametric partial diferential equations. In: International Conference on Learning Representations (2021)

[41] Ren, P., Rao, C., Liu, Y., Ma, Z., Wang, Q., Wang, J.-X., Sun, H.: Physr: Physics-informed deep super-resolution for spatiotemporal data. Journal of Computational Physics 492, 112438 (2023) https://doi.org/10.1016/j.jcp.2023.112438

[42] Zhu, L., Li, G.: Difusion-based super-resolution reconstruction of three-dimensional isotropic turbulence with physical constraints. Physics of Fluids 38(2), 025114 (2026) https://doi.or g/10.1063/5.0308209

[43] Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., Timofte, R.: Swinir: Image restoration using swin transformer. In: 2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), pp. 1833–1844 (2021). https://doi.org/10.1109/ICCVW54120.2021. 00210

[44] Guibas, J., Mardani, M., Li, Z., Tao, A., Anandkumar, A., Catanzaro, B.: Eficient token mixing for transformers via adaptive fourier neural operators. In: International Conference on Learning Representations (2022)

[45] Yousif, M.Z., Yu, L., Lim, H.: High-fidelity reconstruction of turbulent flow from spatially limited data using enhanced super-resolution generative adversarial network. Physics of Fluids 33(12), 125119 (2021) https://doi.org/10.1063/5.0066077

[46] Chong, M.S., Perry, A.E., Cantwell, B.J.: A general classification of three-dimensional flow fields. Physics of Fluids A: Fluid Dynamics 2(5), 765–777 (1990) https://doi.org/10.106 3/1.857730

[47] Meneveau, C.: Lagrangian dynamics and models of the velocity gradient tensor in turbulent flows. Annual Review of Fluid Mechanics 43(1), 219–245 (2011) https://doi.org/10.1146/ annurev-fluid-122109-160708

[48] Chevillard, L., Meneveau, C.: Lagrangian dynamics and statistical geometric structure of turbulence. Physical Review Letters 97(17), 174501 (2006) https://doi.org/10.1103/Phys RevLett.97.174501

[49] Zhong, X., Du, F., Chen, L., Wang, Z., Li, H.: Investigating transformer-based models for spatial downscaling and correcting biases of near-surface temperature and wind-speed forecasts. Quarterly Journal of the Royal Meteorological Society 150(758), 275–289 (2024) https://do i.org/10.1002/qj.4596

[50] Jones, C., Mu, Y., Carvalho, L.M.V., Ding, Q.: The south america low-level jet: form, variability and large-scale forcings. npj Climate and Atmospheric Science 6(1), 175 (2023) https://doi.org/10.1038/s41612-023-00501-4

[51] Barton, E.J., Klein, C., Taylor, C.M., Marsham, J., Parker, D.J., Maybee, B., Feng, Z., Leung, L.R.: Soil moisture gradients strengthen mesoscale convective systems by increasing wind shear. Nature Geoscience 18(4), 330–336 (2025) https://doi.org/10.1038/s41561-0 25-01666-8

[52] Carstens, J.D., Didlake, A.C., Zarzycki, C.M.: Tropical cyclone wind shear-relative asymmetry in reanalyses. Journal of Climate 37(22), 5793–5816 (2024) https://doi.org/10.1175/JCLI -D-23-0628.1

[53] Duchon, C.E.: Lanczos filtering in one and two dimensions. Journal of Applied Meteorology and Climatology 18(8), 1016–1022 (1979) https://doi.org/10.1175/1520-0450(1979)018 <1016:LFIOAT>2.0.CO;2

[54] Mottola, M., Ursprung, S., Rundo, L., et al.: Reproducibility of ct-based radiomic features against image resampling and perturbations for tumour and healthy kidney in renal cancer patients. Scientific Reports 11(1), 11542 (2021) https://doi.org/10.1038/s41598-021-9 0985-y

[55] Bofetta, G., Musacchio, S.: Chaos and predictability of homogeneous-isotropic turbulence. Physical Review Letters 119(5), 054102 (2017) https://doi.org/10.1103/PhysRevLett.11 9.054102

[56] Raonić, B., Molinaro, R., Lanthaler, S., Rohner, T., Armegioiu, V., Simonis, S., Grund, D., Ramic, Y., Wan, Z.Y., Sha, F., Mishra, S., Zepeda-Núñez, L.: Generative ai for eficient statistical computation of fluids. Nature Communications 17, 9846 (2026) https://doi.org/ 10.1038/s41467-026-76390-x

[57] Taira, K., Rigas, G., Fukami, K.: Machine learning in fluid dynamics: A critical assessment. Phys. Rev. Fluids 10, 090701 (2025) https://doi.org/10.1103/8t52-mtb9

[58] McConkey, R., Yee, E., Lien, F.-S.: A curated dataset for data-driven turbulence modelling. Scientific Data 8(1), 255 (2021) https://doi.org/10.1038/s41597-021-01034-2

[59] Duraisamy, K.: Perspectives on machine learning-augmented reynolds-averaged and large eddy simulation models of turbulence. Phys. Rev. Fluids 6, 050504 (2021) https://doi.org/10.1 103/PhysRevFluids.6.050504

[60] Sharma, P., Chung, W.T., Akoush, B., Ihme, M.: A review of physics-informed machine learning in fluid mechanics. Energies 16(5) (2023) https://doi.org/10.3390/en16052343

[61] Barwey, S., Pal, P., Patel, S., Balin, R., Lusch, B., Vishwanath, V., Maulik, R., Balakrishnan, R.: Mesh-based super-resolution of fluid flows with multiscale graph neural networks. Computer Methods in Applied Mechanics and Engineering 443, 118072 (2025) https://doi.org/10.1016/j.cma.2025.118072

[62] Gurevich, D.R., Golden, M.R., Reinbold, P.A., Grigoriev, R.O.: Learning fluid physics from highly turbulent data using sparse physics-informed discovery of empirical relations (spider). Journal of Fluid Mechanics 996, 25 (2024)

[63] Gu, A., Johnson, I., Goel, K., Saab, K., Dao, T., Rudra, A., Ré, C.: Combining recurrent, convolutional, and continuous-time models with linear state space layers. Advances in neural information processing systems 34, 572–585 (2021)

[64] Rangapuram, S.S., Seeger, M.W., Gasthaus, J., Stella, L., Wang, Y., Januschowski, T.: Deep state space models for time series forecasting. Advances in neural information processing systems 31 (2018)

[65] Fu, D.Y., Dao, T., Saab, K.K., Thomas, A.W., Rudra, A., Ré, C.: Hungry hungry hippos: Towards language modeling with state space models. arXiv preprint arXiv:2212.14052 (2022)

[66] Li, K., Li, X., Wang, Y., He, Y., Wang, Y., Wang, L., Qiao, Y.: Videomamba: State space model for eficient video understanding. In: European Conference on Computer Vision, pp. 237–255 (2024). Springer

[67] Sreenivasan, K.R.: Fluid turbulence. Rev. Mod. Phys. 71, 383–395 (1999) https://doi.or g/10.1103/RevModPhys.71.S383

[68] Gu, A., Dao, T.: Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752 (2023)

[69] Duraisamy, K., Iaccarino, G., Xiao, H.: Turbulence modeling in the age of data. Annual Review of Fluid Mechanics 51(Volume 51, 2019), 357–377 (2019) https://doi.org/10.114 6/annurev-fluid-010518-040547

[70] Lecun, Y., Bottou, L., Bengio, Y., Hafner, P.: Gradient-based learning applied to document recognition. Proceedings of the IEEE 86(11), 2278–2324 (1998) https://doi.org/10.1109/ 5.726791

[71] Gneiting, T.: Making and evaluating point forecasts. Journal of the American Statistical Association 106(494), 746–762 (2011) https://doi.org/10.1198/jasa.2011.r10138

[72] Chen, Y., Huang, D., Zhang, D., Zeng, J., Wang, N., Zhang, H., Yan, J.: Theory-guided hard constraint projection (HCP): A knowledge-based data-driven scientific machine learning method. Journal of Computational Physics 445, 110624 (2021) https://doi.org/10.1016/ j.jcp.2021.110624

[73] Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., Dean, J.: Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538 (2017)

[74] Perlman, E., Burns, R., Li, Y., Meneveau, C.: Data exploration of turbulence simulations using a database cluster. In: SC ’07: Proceedings of the 2007 ACM/IEEE Conference on Supercomputing, pp. 1–11 (2007). https://doi.org/10.1145/1362622.1362654

[75] Keys, R.: Cubic convolution interpolation for digital image processing. IEEE transactions on acoustics, speech, and signal processing 29(6), 1153–1160 (1981)

[76] Shi, X., Chen, Z., Wang, H., Yeung, D.-Y., Wong, W.-k., Woo, W.-c.: Convolutional LSTM network: A machine learning approach for precipitation nowcasting. In: Advances in Neural Information Processing Systems, vol. 28, pp. 802–810 (2015). https://proceedings.neurip s.cc/paper/2015/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html

[77] Song, J., Meng, C., Ermon, S.: Denoising difusion implicit models. In: International Conference on Learning Representations (2021). https://openreview.net/forum?id=St1giarCHLP

[78] Morimoto, M., Fukami, K., Zhang, K., Nair, A.G., Fukagata, K.: Convolutional neural networks for fluid flow analysis: toward efective metamodeling and low-dimensionalization. Theoretical and Computational Fluid Dynamics 35(5), 633–658 (2021) https://doi.org/10.1 007/s00162-021-00580-0

[79] Sofos, F., Drikakis, D., Kokkinakis, I.W., Spottswood, S.M.: Convolutional neural networks for compressible turbulent flow reconstruction. Physics of Fluids 35(11), 116120 (2023) https: //doi.org/10.1063/5.0177654

[80] Taghizadeh-Popp, M., Kim, J.W., Lemson, G., Medvedev, D., Raddick, M.J., Szalay, A.S., Thakar, A.R., Booker, J., Chhetri, C., Dobos, L., Rippin, M.: Sciserver: A science platform for astronomy and beyond. Astronomy and Computing 33, 100412 (2020) https://doi.org/ 10.1016/j.ascom.2020.100412

[81] Smagorinsky, J.: General circulation experiments with the primitive equations. Monthly Weather Review 91(3), 99–164 (1963) https://doi.org/10.1175/1520-0493(1963)091 <0099:GCEWTP>2.3.CO;2

[82] Rasp, S., Hoyer, S., Merose, A., Langmore, I., Battaglia, P., Russel, T., Sanchez-Gonzalez, A., Yang, V., Carver, R., Agrawal, S., Chantry, M., Bouallegue, Z.B., Dueben, P., Bromberg, C., Sisk, J., Barrington, L., Bell, A., Sha, F.: WeatherBench 2: A benchmark for the next generation of data-driven global weather models (2024)

[83] Fukami, K., Fukagata, K., Taira, K.: Super-resolution analysis via machine learning: a survey for fluid flows. Theoretical and Computational Fluid Dynamics 37(4), 421–444 (2023) https: //doi.org/10.1007/s00162-023-00663-0

[84] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. In: Advances in Neural Information Processing Systems, vol. 30 (2017). https://proceedings.neurips.cc/paper/2017/hash/3f5ee2435 47dee91fbd053c1c4a845aa-Abstract.html

## Supplementary Information

## 1 Implementation and methodology details

## 1.1 Benchmark experimental settings of TurbSR3D

We adopt a unified training protocol across all datasets and baselines. For lower upscaling factors $( i . e . , 4 \times )$ , each learning-based method is trained for 1000 epochs; for higher factors $( i . e . , 8 \times$ $/ 1 6 \times )$ , we train for 500 epochs due to earlier convergence and overfitting risks. All implementations use PyTorch and are run on an Ubuntu 20.04 server with 8× RTX 4090 GPUs. Unless otherwise specified, each run uses 2 GPUs with a per-GPU batch size of 4 (efective batch size 8). Mixedprecision (AMP) training is enabled for all deep-learning models, reducing memory footprint and wall-clock time, except FNO due to code-level constraints. Training-seed uncertainty is quantified on FIT at $4 \times$ with box-filtered inputs using independent seeds 42, 43, and 44 for all learned methods. The data partitions and model-specific training and evaluation protocols are fixed across seeds; Supplementary Table 20 reports the mean and sample standard deviation over the three runs. We optimize the model using AdamW with a weight decay of $1 0 ^ { - 3 }$ . The learning rate follows an exponential decay schedule, starting from an initial value of 10−<sup>4</sup> and decaying by a factor of 0.997 at each epoch. Training minimizes the composite loss in Eq. (9) with the physics-informed neural optimization described in Section 4.3. Both the data term $\mathcal { L } _ { \mathrm { m a e } }$ and the physical terms in ${ \mathcal { L } } _ { \mathrm { p h y } }$ use $L _ { 1 }$ penalties. For TurbSR3D experiments, we set the balancing coeficient in Eq. (9) to $\lambda _ { \mathrm { p h y } } \dot { = } 1 0 ^ { - 3 }$ and the momentum weight in Eq. (10) to $\beta = 0 . 3$ . For decaying isotropic turbulence (DIT), we use $\lambda _ { \mathrm { p h y } } = 1 0 ^ { - 2 }$ and $\beta = 0 . 3$ . The sensitivity to these weights is examined in Supplementary Note 1.2.

## 1.2 Sensitivity to physical-loss weights

We examine the sensitivity to $\lambda _ { \mathrm { p h y } }$ and $\beta$ in Eqs. (9) and (10) on DIT and FIT (Supplementary Tables 1 and 2). With $\beta = 0 . 3 , \lambda _ { \mathrm { p h y } } = 0 . 0 1$ for DIT and 0.001 for FIT give the best nRMSE, SSIM, and PSNR among the tested settings. Increasing $\lambda _ { \mathrm { p h y } }$ to 0.1 worsens all three metrics in both tasks. With $\lambda _ { \mathrm { p h y } }$ fixed at these respective values, $\beta = 0 . 3$ performs best across all three metrics compared with 0.1 and 1.

Supplementary Table 1: Sensitivity to physical-loss weights on DIT at $4 \times$ with spectral-cutof coarsegraining. Best values for each metric are in bold. nRMSE and SSIM are reported to four decimal places and PSNR to two decimal places.
<table><tr><td> $\lambda _ { \mathrm { p h y } }$ </td><td> $\beta$ </td><td>nRMSE↓</td><td>SSIM ↑</td><td>PSNR (dB) ↑</td></tr><tr><td>0</td><td>0.3</td><td>0.0817</td><td>0.9402</td><td>43.32</td></tr><tr><td>0.001</td><td>0.3</td><td>0.0807</td><td>0.9415</td><td>43.44</td></tr><tr><td>0.01</td><td>0.3</td><td>0.0773</td><td>0.9463</td><td>43.82</td></tr><tr><td>0.1</td><td>0.3</td><td>0.1130</td><td>0.9022</td><td>40.45</td></tr><tr><td>0.01</td><td>0.1</td><td>0.0789</td><td>0.9438</td><td>43.64</td></tr><tr><td>0.01</td><td>1</td><td>0.0841</td><td>0.9372</td><td>43.17</td></tr></table>

Supplementary Table 2: Sensitivity to physical-loss weights on FIT at $8 \times$ with Gaussian coarsegraining. Best values for each metric are in bold. nRMSE and SSIM are reported to four decimal places and PSNR to two decimal places.
<table><tr><td> $\lambda _ { \mathrm { p h y } }$ </td><td> $\beta$ </td><td>nRMSE↓</td><td>SSIM ↑</td><td>PSNR (dB) ↑</td></tr><tr><td>0</td><td>0.3</td><td>0.1233</td><td>0.7957</td><td>42.20</td></tr><tr><td>0.0001</td><td>0.3</td><td>0.1232</td><td>0.7960</td><td>42.20</td></tr><tr><td>0.001</td><td>0.3</td><td>0.1173</td><td>0.8093</td><td>42.61</td></tr><tr><td>0.01</td><td>0.3</td><td>0.1216</td><td>0.7985</td><td>42.32</td></tr><tr><td>0.1</td><td>0.3</td><td>0.1810</td><td>0.6722</td><td>38.66</td></tr><tr><td>0.001</td><td>0.1</td><td>0.1174</td><td>0.8090</td><td>42.59</td></tr><tr><td>0.001</td><td>1</td><td>0.1195</td><td>0.8044</td><td>42.46</td></tr></table>

## 1.3 Benchmark experimental settings of ERA5

For the ERA5 downscaling task, the training protocol is adjusted to accommodate the specific scale and characteristics of the data. Each model is trained for $T _ { \mathrm { m a x } } = 1 0 0$ epochs on a single GPU with a batch size of 4. Consistent with the TurbSR3D experiments, AMP is enabled to improve computational eficiency. Training-seed uncertainty is quantified for Ours, FNO, and SwinIR at both 4× and $8 \times$ using independent seeds 42, 43, and 44. The chronological data split and all other model-specific protocols are fixed across seeds; Supplementary Table 21 reports the resulting mean and sample standard deviation. We optimize the models using AdamW with a weight decay of $1 0 ^ { - 3 }$ Regarding the training objective, considering the coarse spatial resolution of the ERA5 reanalysis data (∼30 km), we do not incorporate explicit governing equations into the loss. Consequently, the physics-informed term is set to $\lambda _ { \mathrm { p h y } } = 0$ , and the optimization minimizes solely the data fidelity term $\mathcal { L } _ { \mathrm { m a e } }$ , formulated as the L1 loss between the predicted and ground-truth fields.

## 1.4 Details of state space model

Preliminaries. In the channelwise implementation, each internal feature channel maintains an M-dimensional recurrent state at each Eulerian grid location. The channel index is omitted in the main-text equations for clarity. Here, M corresponds to the state dimension $d _ { s }$ used in Supplementary Note 3.4.

We consider a continuous-time state space model

$$
\dot { \pmb { h } } ( t ) = \pmb { A } \pmb { h } ( t ) + \pmb { B } \pmb { x } ( t ) ,\tag{S1}
$$

$$
\begin{array} { r } { \pmb { y } ( t ) = \pmb { C } \pmb { h } ( t ) + \pmb { D } \pmb { x } ( t ) , } \end{array}\tag{S2}
$$

where A, B, C, D are learnable parameters, and $\dot { h } ( t )$ denotes the time derivative of the latent state. Following standard practice [63], the continuous-time system is discretized using a zero-order hold with step size $\Delta ,$ , yielding

$$
\bar { \cal A } = \exp ( \Delta { \cal A } ) ,\tag{S3}
$$

$$
\bar { B } = ( \Delta A ) ^ { - 1 } \big ( \mathrm { e x p } ( \Delta A ) - I \big ) ( \Delta B ) .\tag{S4}
$$

The resulting discrete-time formulation is

$$
\begin{array} { r } { \pmb { h } _ { t } = \pmb { \bar { A } } \pmb { h } _ { t - 1 } + \pmb { \bar { B } } \pmb { x } _ { t } , } \end{array}\tag{S5}
$$

$$
\begin{array} { r } { { \pmb y } _ { t } = C { \pmb h } _ { t } + D { \pmb x } _ { t } . } \end{array}\tag{S6}
$$

For notational simplicity, we omit the bar notation in the main text.

Gating mechanism. To enhance representational capability, we introduce a spatially varying selection gate ${ \pmb g } _ { t } ( { \pmb r } ) = \phi \left( { \pmb v } _ { t } ( { \pmb r } ) \right)$ , where ${ \mathbf { } } v _ { t }$ is another locally computed transformation of ${ z } _ { t }$ by a 3D convolution and $\phi$ denotes a smooth gating nonlinearity (implemented as the SiLU activation). The output is thus given by:

$$
\begin{array} { r } { \pmb { s } _ { t } ( \pmb { r } ) = \pmb { g } _ { t } ( \pmb { r } ) \odot \pmb { y } _ { t } ( \pmb { r } ) , } \end{array}\tag{S7}
$$

where the gating mechanism modulates local temporal updates according to the instantaneous flow state. A subsequent point-wise linear projection is applied to $\mathbf { \Delta } _ { \mathbf { \mathcal { S } } _ { t } }$ to align channel dimensions.

## 1.5 Methods and training configurations of TurbSR3D

The proposed framework is benchmarked against classical interpolation and representative learning-based super-resolution models. Unless otherwise noted, all learnable baselines are retrained under a unified protocol to ensure comparability. For RRDB, RCAN, EDSR, the implementations and hyperparameter guidelines are referenced from the public BlastNet 2.0 repository [24] while being adapted to our setting.

Tricubic. Implemented using scipy.ndimage, the tricubic operator is applied to each LR frame independently to reconstruct HR fields.

RRDB [37]. The architecture is configured with two residual dense blocks (channel width 56) followed by a trilinear upsampling head.

RCAN [39]. The model features a channel width of 64, a reduction ratio of 16, and 10 residual channel–attention blocks. Spatial upsampling is performed via a staged convolutional head with pixel-shufle.

EDSR [38]. It uses a channel width of 44, residual scaling of 0.1, and 32 residual blocks, with a pixel-shufle upsampling head.

FNO [40]. This baseline is instantiated as a low-resolution spatiotemporal feature trunk paired with a pixel-shufle upsampling head. It consists of two hidden layers with a channel width of 128, retaining 3 spatial spectral modes (reduced to 2 for HBDT at 16×) and 2 temporal modes.

PhySR [41]. This model comprises one ConvLSTM [76] layer, eight CNN residual blocks (channel width 40), and a pixel-shufle upsampling head.

SwinIR [43]. This model is extended to 3D for volumetric turbulence super-resolution by adopting 3D convolutions, window-based self-attention with alternating regular and shifted windows, and 3D relative position biases. The architecture consists of six Residual Swin Transformer Blocks (RSTBs), each containing six Swin Transformer Layers (STLs), with a channel width of 72 and four attention heads. The attention window size is set to $4 \times 4 \times 4$ for 4× super-resolution and $2 \times 2 \times 2$ for 8× and 16× super-resolution.

DifSR [42]. DifSR is adapted from the original work, with the auxiliary ESRGAN prediction omitted. It uses a 3D U-Net and follows the physical-loss settings of the original study. Training uses a difusion schedule of 400 time steps. Inference uses DDIM [77] with a time-step stride of 10, yielding 40 denoising refinement steps.

Ours. The proposed model comprises one layer of the state space model and six multiscale neuraloperator blocks. The state space model uses a feature width of 64 and a state dimension of 16, while the neural-operator layers use a feature width of 256. In the spectral neural operator, the soft-shrinkage threshold in Eq. (7) is set to $\lambda = 0 . 0 1$ . Block-diagonalisation (four blocks) is further employed to reduce the parameter count and improve computational eficiency.

Decoder and output projection. The decoder operates frame by frame with parameters shared across time. $\mathrm { ~ A ~ } 3 \times 3 \times 3$ convolution first transforms the multiscale neural-operator features, followed by successive upsampling stages. Each stage uses a $3 \times 3 \times 3$ convolution to expand the feature channels by a factor of eight, followed by 3D pixel shufle, which rearranges these channels to double the resolution along all three Cartesian coordinates. An upscaling factor $s = 2 ^ { m }$ therefore requires m stages: two, three, and four for 4×, 8×, and 16× reconstruction, respectively. The learned projection Q is a final $3 \times 3 \times 3$ convolution that maps the decoded features to the physical variables.

It is worth noting that maintaining identical hyperparameter settings across experimental setups leads to an inherent increase in model parameters for higher upscaling factors (Supplementary Tables 10–12), although the magnitude of this growth depends on the architectural design. As illustrated in Supplementary Fig. 1, the proposed framework achieves superior performance while maintaining the fewest average parameters across all test cases.

![](images/d97e6f6b290ff4da556f90d55dba710d7fe7c173e39554f2bfd958abbafad58c.jpg)  
Supplementary Fig. 1: Average parameter counts (M) for eight super-resolution reconstruction methods: Ours, SwinIR, RRDB, EDSR, RCAN, DifSR, PhySR, and FNO. Each bar shows the equally weighted mean across six configurations: FIT at 4 and 8 , MHDT at 4 and 8 , and HBDT at 8 and 16 , using the parameter counts reported in Supplementary Tables 10–12. Values above the bars are rounded to two decimal places. The proposed method has the lowest average parameter count (3.87 M).

## 1.6 Scale-selective momentum regularization

The main FIT and MHDT benchmarks use the high-pass momentum penalty in Eq. (13). The high-pass operator is implemented as

$$
\mathcal { H } _ { \sigma } = I - G _ { \sigma } ,\tag{S8}
$$

where I is the identity operator and $G _ { \sigma }$ is a local three-dimensional Gaussian low-pass filter. Thus, the filtered residual is obtained by subtracting its Gaussian-smoothed component from the original residual. The Gaussian low-pass component is implemented with a discrete separable kernel truncated at 3σ and valid convolution along all three spatial directions. At 4× and 8× reconstruction, $\sigma / \Delta x _ { \mathrm { H R } }$ is 1.4991 and 2.9983, respectively, with kernel radii of 5 and 9 HR grid points. The FIT forcing is supported on $0 < | \pmb { k } | ^ { 2 } \leq 6 ~ [ 2 8 ]$ , and the MHDT Taylor–Green forcing is supported at $| \pmb { k } | = \sqrt { 1 2 }$ [29]. Applying the training filter directly to the forcing fields gives the RMS values in Supplementary Table 3. For each dataset, the pre-filter RMS from the 4× evaluation serves as the common reference at both reconstruction factors. The filtered forcing RMS is at most 0.1989% of this reference, demonstrating strong suppression of its direct contribution to the scaleselective constraint.

Supplementary Table 3: Forcing RMS before and after high-pass filtering. The same Gaussian filter is used in training and in this diagnostic. Retained percentages are computed from unrounded values relative to the dataset-specific 4 pre-filter reference.
<table><tr><td>Dataset</td><td>Scale</td><td>Reference forcing  $\mathrm { R M S } _ { \mathrm { r e f } } ( f )$ </td><td>After filtering  $\mathrm { R M S } ( \mathcal { H } _ { \sigma } f )$ </td><td>RMS retained</td></tr><tr><td>FIT</td><td> $4 \times$ </td><td>0.0847</td><td> $1 . 1 1 \times 1 0 ^ { - 5 }$ </td><td>0.0131%</td></tr><tr><td>FIT</td><td> $8 \times$ </td><td>0.0847</td><td> $4 . 3 8 \times 1 0 ^ { - 5 }$ </td><td>0.0517%</td></tr><tr><td>MHDT</td><td> $4 \times$ </td><td>0.0545</td><td> $2 . 7 6 \times 1 0 ^ { - 5 }$ </td><td>0.0506%</td></tr><tr><td>MHDT</td><td> $8 \times$ </td><td>0.0545</td><td> $1 . 0 8 \times 1 0 ^ { - 4 }$ </td><td>0.1989%</td></tr></table>

The implemented $L _ { 1 }$ momentum term provides robust physical residual regularization. Moti vated by the intermittency of turbulent velocity gradients and small-scale velocity increments [27, $4 7 ]$ , we use a linear-growth penalty that limits the influence of large local residuals compared with a squared $L _ { 2 }$ penalty. The momentum residual is evaluated using numerical derivatives of reconstructed fields, so large local residuals may also contain contributions from numerical errors.

Gaussian residual statistics naturally motivate a quadratic penalty. For additive Gaussian forcing that is white in time, the Onsager–Machlup action contains a quadratic residual term weighted by the forcing covariance. The present benchmarks use state-dependent low-mode energy maintenance in FIT and deterministic Taylor–Green forcing in MHDT [28, 29]; the implemented $L _ { 1 }$ term serves as a soft dynamical constraint in these systems.

We compare three momentum-loss formulations on FIT at 4× super-resolution with box-filtered inputs: unfiltered $L _ { 1 }$ , high-pass $L _ { 1 }$ , and high-pass squared $L _ { 2 }$ (mean squared residual). As shown in Supplementary Fig. 2, all three closely follow the reference energy spectrum at low and intermediate wavenumbers. The high-pass $L _ { 1 }$ variant shows closer agreement in the enlarged high-wavenumber region and yields the lowest relative $L _ { 1 }$ error in the high-wavenumber energy spectrum: 0.1736, compared with 0.2125 for unfiltered $L _ { 1 }$ and 0.1920 for high-pass squared $L _ { 2 }$ . These comparisons support both high-pass filtering of the momentum residual and the use of $L _ { 1 }$ for high-wavenumber spectral reconstruction.

![](images/22969793288f0f17f1b720c0a097e372f33b6887f3a9f7e01c4c9da23c97bbda.jpg)  
Supplementary Fig. 2: Sensitivity to the momentum-loss formulation on FIT at 4 super-resolution with box-filtered inputs. Left, Kinetic-energy spectra of the reference and reconstructions using an unfiltered $L _ { 1 }$ penalty (“L1 (direct)”), a high-pass squared $L _ { 2 }$ penalty (“L2 (high-pass)”), and the implemented high-pass $L _ { 1 }$ penalty (“L1 (high-pass)”). The inset highlights diferences at high wavenumbers; the black dashed line is a $k ^ { - 5 / 3 }$ guide. Right, Relative $L _ { 1 }$ errors of the reconstructed energy spectra over $k > 1 2 8$ . Lower values indicate closer agreement with the reference in this high-wavenumber range.

## 1.7 Hard incompressibility constraint on the physical output

HCP formulation. Theory-guided hard constraint projection (HCP) [72] embeds physical knowledge by discretizing a governing equation, separating its coeficients from the predicted physical quantities, and projecting those predictions onto the resulting constraint set. For a prediction y and a linear constraint $\mathsf { A } \boldsymbol { y } = \boldsymbol { b }$ with full-row-rank A, the closest admissible prediction is

$$
\begin{array} { r } { \pmb { y } ^ { \mathrm { H } } = \pmb { y } - \pmb { \mathrm { A } } ^ { \mathsf { T } } ( \mathsf { A A } ^ { \mathsf { T } } ) ^ { - 1 } ( \mathsf { A } \pmb { y } - \pmb { b } ) . } \end{array}\tag{S9}
$$

The original HCP construction couples hydraulic-head predictions through a discretized subsurfaceflow equation. Here, the constraint couples the three components of the reconstructed turbulent velocity through instantaneous continuity. The network provides the reconstruction, and continuity determines the correction.

Projection onto the divergence-free subspace. At each reconstructed time, let v denote the HR velocity after the learned output mapping $Q$ and inverse normalization using the training statistics in Eq. (S22). The latent channels encode learned features, so incompressibility is imposed on this final physical velocity. For a uniform grid with N voxels, write $\pmb { v } = ( \pmb { v } _ { x } ^ { \top } , \pmb { v } _ { u } ^ { \top } , \pmb { v } _ { z } ^ { \top } ) ^ { \top } \in \mathbb { R } ^ { 3 N }$ The discrete divergence operator $\mathsf { D } _ { h } = [ \mathsf { D } _ { x } ~ \mathsf { D } _ { y } ~ \mathsf { D } _ { z } ]$ combines the spatial derivatives as $\begin{array} { r } { \mathsf { D } _ { h } \pmb { v } = } \end{array}$ $\mathsf { D } _ { x } \pmb { v } _ { x } + \mathsf { D } _ { y } \pmb { v } _ { y } + \mathsf { D } _ { z } \pmb { v } _ { z }$ , so continuity constrains the three components jointly. Each $\mathsf { D } _ { \alpha }$ is a fourthorder central diference evaluated at interior grid points. Imposing $\mathsf { D } _ { h } \pmb { w } = 0$ gives

$$
\begin{array} { r l } & { \pmb { u } ^ { \mathrm { H } } = \underset { { \pmb { w } } : { \mathsf { D } } _ { h } \pmb { w } = 0 } { \arg \operatorname* { m i n } } \frac { 1 } { 2 } \| \pmb { w } - \pmb { v } \| _ { 2 } ^ { 2 } = { \mathsf { P } } _ { h } \pmb { v } , } \\ & { { \mathsf { P } } _ { h } = { \mathsf { I } } - { \mathsf { D } } _ { h } ^ { \mathsf { T } } ( { \mathsf { D } } _ { h } { \mathsf { D } } _ { h } ^ { \mathsf { T } } ) ^ { \dagger } { \mathsf { D } } _ { h } . } \end{array}\tag{S10}
$$

Here † denotes the Moore–Penrose pseudoinverse. For the interior operator used here, $\mathsf { D } _ { h }$ has full row rank and † reduces to the ordinary inverse; the pseudoinverse also covers operators with linearly dependent rows, such as periodic diferences. In addition, $\mathsf { D } _ { h } ^ { \mathsf { T } }$ is the discrete adjoint under the uniform-grid Euclidean inner product. The stationarity condition for the constrained minimum gives $\pmb { u } ^ { \mathrm { H } } - \pmb { v } = - \mathsf { D } _ { h } ^ { \mathsf { T } } \pmb { \lambda }$ , with $\mathsf { D } _ { h } \mathsf { D } _ { h } ^ { \mathsf { T } } \lambda = \mathsf { D } _ { h } \pmb { v } ;$ eliminating the multiplier yields Eq. (S10).

The operator $\mathsf { P } _ { h }$ is an orthogonal projection onto the discrete divergence-free subspace ker $\mathsf { D } _ { h }$ whose dimension is $3 N - \mathrm { r a n k } ( \mathsf { D } _ { h } )$ . It leaves admissible velocities unchanged and makes the smallest correction in the discrete $L _ { 2 }$ norm needed to satisfy the continuity constraint. The efects on reconstruction accuracy and energy spectra are evaluated below.

Use in the reconstruction model. HCP is an optional extension to the reconstruction model, evaluated only in the dedicated DIT comparison below. The projection is determined by the physical constraint and introduces no learnable parameters or penalty weight. It enforces discrete continuity to numerical precision, while the soft momentum term continues to regularize the dynamics during training. In the DIT comparison, the model trained with HCP evaluates the reconstruction loss on the projected velocity.

Evaluation on DIT. We compare models trained with and without HCP on DIT at 4× superresolution with spectral-cutof inputs. The divergence RMS measures compliance with incompressibility:

$$
\delta _ { \mathrm { d i v } } ( \pmb { w } ) = \left[ \frac { 1 } { N _ { \mathrm { c } } } \| \mathsf { D } _ { h } \pmb { w } \| _ { 2 } ^ { 2 } \right] ^ { 1 / 2 } ,\tag{S11}
$$

where $N _ { \mathrm { c } }$ is the number of evaluated divergence values. Reconstruction accuracy is measured by nRMSE, defined in Eq. (18), and the energy spectra assess the recovered energy distribution across scales.

As shown in Supplementary Table 4, incorporating HCP reduces the divergence RMS from $1 . 4 1 \times 1 0 ^ { - 2 } \mathrm { ~ t o ~ } 1 . 5 2 \times 1 0 ^ { - 6 }$ , with a small increase in nRMSE from 0.0773 to 0.0779. HCP therefore substantially improves compliance with incompressibility at a small cost in reconstruction accuracy. The energy spectra in Supplementary Fig. 3 show that both models closely follow the reference at low and intermediate wavenumbers. The model trained with HCP also shows closer agreement with the reference over part of the high-wavenumber range, while retaining the main spectral characteristics of the reconstructed flow.

![](images/b1d9e19b325d95de1d01c70d1c9df6eae242633758083a38ccb0e81fddfdd0a6.jpg)  
Supplementary Fig. 3: Efect of HCP on the reconstructed energy spectrum. Reference spectrum and predictions from models trained with and without HCP on DIT at 4 super-resolution with spectral-cutof inputs.

Supplementary Table 4: Efect of HCP on divergence RMS and nRMSE. Models trained with and without HCP on DIT at 4 super-resolution with spectral-cutof inputs.
<table><tr><td>Model</td><td>Divergence RMS</td><td>nRMSE</td></tr><tr><td>HCP</td><td> $1 . 5 2 \times 1 0 ^ { - 6 }$ </td><td>0.0779</td></tr><tr><td> $\mathrm { w / o }$  HCP</td><td> $1 . 4 1 \times 1 0 ^ { - 2 }$ </td><td>0.0773</td></tr></table>

## 1.8 Methods and training configurations of ERA5

For the ERA5 dataset, the model configurations are adjusted to handle the specific data characteristics.

Bicubic. Implemented using scipy.ndimage, the bicubic operator is applied to each 2D slice independently to reconstruct HR fields.

SwinIR [43]. Designed as a 2D baseline, this model performs reconstruction in a slice-wise manner. The numbers of RSTBs and STLs, window size, channel width, and number of attention heads are set to 2, 2, 4, 64, and 4, respectively. The reconstruction is handled by a pixel-shufle head.

FNO [40]. The model is configured with two hidden layers and a channel width of 64. In the spectral domain, 4 horizontal spatial modes, 2 vertical spatial modes, and 2 temporal modes are retained.

Ours. For this dataset, the proposed framework is instantiated with four multiscale neural-operator blocks. The channel width of the neural-operator layers is set to 224.

Decoder and output projection. The ERA5 decoder operates frame by frame with parameters shared across time. $\mathrm { ~ A ~ 3 ~ } { \times } \mathrm { ~ 3 ~ } { \times } \mathrm { ~ 3 ~ }$ convolution first transforms the multiscale neural-operator features. Each upsampling stage then applies a shared $3 \times 3$ convolution at every pressure level to expand the feature channels by a factor of four, followed by 2D pixel shufle, which doubles the resolution along both horizontal coordinates. The 4× and 8× configurations use two and three stages, respectively, while retaining all eight pressure levels. The learned projection $Q$ is a final $3 \times 3 \times 3$ convolution that maps the decoded multi-level features to the physical variables.

## 1.9 Supplementary definitions of evaluation metrics

Let $\hat { X }$ denote the prediction field and X the ground truth.

Peak signal-to-noise ratio (PSNR). PSNR evaluates reconstruction quality on a logarithmic scale based on mean squared error (MSE):

$$
\mathrm { P S N R } ( \hat { X } , X ) = 1 0 \cdot \log _ { 1 0 } \bigg ( \frac { L ^ { 2 } } { \mathrm { M S E } ( \hat { X } , X ) } \bigg ) ,\tag{S12}
$$

where L denotes the data range of X.

Structural similarity (SSIM). We adopt the 3D extension of SSIM [24] to assess local structural fidelity:

$$
\mathrm { S S I M } ( \hat { X } , X ) = \frac { ( 2 \mu _ { \hat { X } } \mu _ { X } + c _ { 1 } ) ( 2 \sigma _ { \hat { X } X } + c _ { 2 } ) } { ( \mu _ { \hat { X } } ^ { 2 } + \mu _ { X } ^ { 2 } + c _ { 1 } ) ( \sigma _ { \hat { X } } ^ { 2 } + \sigma _ { X } ^ { 2 } + c _ { 2 } ) } ,\tag{S13}
$$

where $\mu$ and $\sigma ^ { 2 }$ denote local means and variances, $\sigma _ { \hat { X } X }$ is the local covariance, and $c _ { 1 } , \ c _ { 2 }$ are constants for numerical stability.

Fourier-space RMSE (fRMSE). To quantify scale-dependent errors, we compute Fourier-space

RMSE between the 3D spectra of $\hat { X }$ and $\boldsymbol { X }$ :

$$
\mathrm { f R M S E } = \sqrt { \frac { 1 } { | \mathcal { K } | } \sum _ { \kappa \in \mathcal { K } } \Big \| \mathcal { F } _ { 3 d } ( \hat { X } ) ( \kappa ) - \mathcal { F } _ { 3 d } ( X ) ( \kappa ) \Big \| _ { 2 } ^ { 2 } } ,\tag{S14}
$$

where $\mathcal { F } _ { 3 d }$ denotes the 3D discrete Fourier transform over spatial dimensions and $\kappa$ is the set of frequency indices.

Energy spectra. We compute energy spectra by summing Fourier-mode energies within spherical wavenumber shells and dividing by the shell width $\Delta k$ . The shell $S _ { k }$ contains wavevectors κ satisfying $k - \Delta k / 2 \leq | \pmb { \kappa } | < k + \Delta k / 2$ . Fourier coeficients are normalized by the total number of spatial grid points.

For FIT, we use the kinetic energy spectrum

$$
E ( k ) = \frac { 1 } { 2 \Delta k } \sum _ { \kappa \in \mathcal { S } _ { k } } | \widehat { \pmb { u } } ( \pmb { \kappa } ) | ^ { 2 } ,\tag{S15}
$$

where u is the velocity field and $\widehat { ( \cdot ) }$ denotes the normalized Fourier transform.

cFor MHDT, we use the total kinetic plus magnetic spectrum

$$
E ( k ) = \frac { 1 } { 2 \Delta k } \sum _ { \kappa \in \mathcal { S } _ { k } } \left( | \widehat { \pmb { u } } ( \pmb { \kappa } ) | ^ { 2 } + | \widehat { \pmb { b } } ( \pmb { \kappa } ) | ^ { 2 } \right) ,\tag{S16}
$$

where $^ { b }$ is the magnetic field expressed in Alfvén units.

For HBDT, we adopt the density-weighted kinetic spectrum

$$
E ( k ) = \frac { 1 } { 2 \Delta k } \sum _ { \kappa \in S _ { k } } \left| \widehat { \sqrt { \rho } \pmb { u } } ( \pmb { \kappa } ) \right| ^ { 2 } ,\tag{S17}
$$

where $\rho$ denotes the mass density. With this normalization, summing over all Fourier shells gives $\begin{array} { r } { \sum _ { k } E ( k ) \Delta k = \frac { 1 } { 2 } \langle \rho | \pmb { u } | ^ { 2 } \rangle _ { \pmb { x } } } \end{array}$ , where $\langle \cdot \rangle _ { x }$ denotes spatial averaging.

## 2 Datasets and preprocessing

## 2.1 JHTDB data sources and target variables

We construct a volumetric turbulence super-resolution corpus from the JHTDB. Three canonical systems are used: forced isotropic turbulence (FIT), forced magnetohydrodynamic turbulence (MHDT), and homogeneous buoyancy-driven turbulence (HBDT). For completeness, we summarize the governing equations associated with each turbulence regime in JHTDB.

FIT. It is a canonical incompressible turbulent flow driven by large-scale forcing and widely used for studying spatiotemporal turbulence dynamics. The governing equations are the incompressible Navier–Stokes equations:

$$
\begin{array} { c } { { \displaystyle { \frac { \partial { \pmb u } } { \partial t } } + ( { \pmb u } \cdot \nabla ) { \pmb u } = - \frac { 1 } { \rho _ { 0 } } \nabla p + \nu \nabla ^ { 2 } { \pmb u } + { \pmb f } , } } \\ { { \nabla \cdot { \pmb u } = 0 , } } \end{array}\tag{S18}
$$

where u and $p$ denote velocity and pressure, and $f$ is an external forcing term used to maintain statistical stationarity. Its Fourier support is restricted to the lowest modes, $0 < | \pmb { k } | ^ { 2 } \leq 6 ~ [ 2 8 ]$ ; its treatment in the scale-selective training objective is given in Section 4.3.

MHDT. It describes incompressible conducting fluids with coupled velocity and magnetic dynamics. The equations couple the velocity field u and magnetic field $\textstyle \mathbf { \phi } _ { b \colon }$

$$
\begin{array} { l } { \displaystyle \frac { \partial \pmb { u } } { \partial t } + ( \pmb { u } \cdot \nabla ) \pmb { u } = - \nabla p + ( \nabla \times \pmb { b } ) \times \pmb { b } + \nu \nabla ^ { 2 } \pmb { u } + \pmb { f } , } \\ { \displaystyle \frac { \partial \pmb { b } } { \partial t } + ( \pmb { u } \cdot \nabla ) \pmb { b } = ( \pmb { b } \cdot \nabla ) \pmb { u } + \eta \nabla ^ { 2 } \pmb { b } , } \\ { \nabla \cdot \pmb { u } = 0 , \quad \nabla \cdot \pmb { b } = 0 , } \end{array}\tag{S19}
$$

Here, $p = { p _ { \mathrm { t h } } } / { \rho _ { 0 } }$ denotes the kinematic pressure, with $p _ { \mathrm { t h } }$ the thermodynamic pressure and $\rho _ { 0 }$ the constant mass density. The magnetic field b is expressed in Alfvén velocity units, and $\eta$ denotes the magnetic difusivity. The external stirring is the Taylor–Green forcing with amplitude $1 / 4$ at wavenumber 2 [29].

HBDT. HBDT describes buoyancy-driven mixing between two miscible incompressible fluids of diferent densities. The mixture obeys the variable-density, low-Mach-number Navier–Stokes equations, with nonzero velocity divergence arising from mixing:

$$
\begin{array} { c } { \displaystyle \frac { \partial \rho } { \partial t } + \nabla \cdot ( \rho \boldsymbol { u } ) = 0 , } \\ { \displaystyle \frac { \partial ( \rho \boldsymbol { u } ) } { \partial t } + \nabla \cdot ( \rho \boldsymbol { u } \otimes \boldsymbol { u } ) = - \nabla p + \nabla \cdot \boldsymbol { \tau } + \rho \boldsymbol { g } , } \\ { \displaystyle \boldsymbol { \tau } = \frac { \rho } { R e _ { 0 } } \left[ \nabla \boldsymbol { u } + ( \nabla \boldsymbol { u } ) ^ { T } - \frac { 2 } { 3 } ( \nabla \cdot \boldsymbol { u } ) \boldsymbol { I } \right] , } \\ { \nabla \cdot \boldsymbol { u } + \frac { 1 } { R e _ { 0 } S c } \nabla ^ { 2 } ( \ln \rho ) = 0 , } \end{array}\tag{S20}
$$

where $\rho$ denotes density, $\tau$ the viscous stress tensor, g the gravitational acceleration, I the identity tensor, $R e _ { 0 }$ the reference Reynolds number, and Sc the Schmidt number. For each turbulence regime, the set of target (super-resolved) variables is listed in Supplementary Table 5.

Supplementary Table 5: Reconstructed variables per turbulent system.
<table><tr><td>System</td><td>Velocity u Pressure p</td><td></td><td>Extra variable</td></tr><tr><td>FIT</td><td>√</td><td>√</td><td></td></tr><tr><td>MHDT</td><td>√</td><td>√</td><td>Magnetic field b</td></tr><tr><td>HBDT</td><td>√</td><td>√</td><td>Density ρ</td></tr></table>

## 2.2 System-specific configurations and parameters

Unless otherwise specified, flows are periodic in all directions; HR spatial spacing is $\Delta x =$ $\Delta y = \Delta z = \pi / 5 1 2$ . Supplementary Tables 6–8 summarise key variables and parameters used in our experiments.

Supplementary Table 6: FIT dataset: reconstructed variables, governing equations, and numerical parameters.
<table><tr><td>Category</td><td>Specification</td></tr><tr><td></td><td>Reconstructed variables Velocity u (3 components), pressure p</td></tr><tr><td>Governing model</td><td>Incompressible Navier-Stokes with large-scale forcing</td></tr><tr><td>Transport coefficient</td><td>Kinematic viscosity  $\nu = 1 . 8 5 \times { 1 0 ^ { - 4 } }$ </td></tr><tr><td>Spatial grid spacing</td><td> $\Delta x = \Delta y = \Delta z = \pi / 5 1 2$ </td></tr><tr><td>Temporal resolution</td><td> $\Delta t = 2 \times 1 0 ^ { - 3 }$ </td></tr></table>

Supplementary Table 7: MHDT dataset: reconstructed variables, governing equations, and numerical parameters.
<table><tr><td>Category</td><td>Specification</td></tr><tr><td></td><td>Reconstructed variables Velocity u, magnetic field b, pressure p</td></tr><tr><td>Governing model</td><td>Incompressible MHD equations</td></tr><tr><td>Transport coefficients</td><td>Kinematic viscosity  $\nu = 1 . 1 \times 1 0 ^ { - 4 }$  Magnetic diffusivity  $\eta = 1 . 1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Spatial grid spacing</td><td> $\Delta x = \Delta y = \Delta z = \pi / 5 1 2$ </td></tr><tr><td>Temporal resolution</td><td> $\Delta t = 2 . 5 \times 1 0 ^ { - 3 }$ </td></tr></table>

## 2.3 TurbSR3D corpus statistics and splits

We aggregate a total of 3600+ high-quality spatiotemporal HR samples, organised into nonoverlapping train/validation/test splits per system. Supplementary Table 9 reports the sample counts and storage footprints.

## 2.4 Coarse-graining and LR generation

Given a high-resolution spatiotemporal field ${ \pmb X } _ { \mathrm { H } } ~ \in ~ \mathbb { R } ^ { T \times H _ { H } \times W _ { H } \times D _ { H } \times C }$ , we obtain its lowresolution counterpart $\pmb { X } _ { \mathrm { L } } \in \bar { \mathbb { R } ^ { T \times H _ { L } \times W _ { L } \times D _ { L } \times C } }$ through a generic coarse-graining process:

$$
X _ { \mathrm { L } } = \mathcal { D } ( X _ { \mathrm { H } } ) + \pmb { \xi } ,\tag{S21}
$$

Supplementary Table 8: HBDT dataset: reconstructed variables, governing equations, and numerical parameters.
<table><tr><td>Category</td><td>Specification</td></tr><tr><td>Reconstructed variables Density</td><td> $\rho ,$  velocity  ${ \mathbf { } } ^ { \mathbf { } } \mathbf { \Delta } ^ { \mathbf { } } \mathbf { u } ,$  pressure  $p$ </td></tr><tr><td>Governing model</td><td>Variable-density, low-Mach-number Navier-Stokes with buoyancy production term</td></tr><tr><td>Control parameters</td><td>Reynolds number  $R e _ { 0 } = 1 2 5 0 0$  Froude number  $F r = 1 . 0$ </td></tr><tr><td>Gravity vector</td><td>Schmidt number  $S c = 1 . 0$   $\pmb { g } = ( 1 . 0 , 0 . 0 , 0 . 0 )$ </td></tr><tr><td>Spatial grid spacing</td><td> $\Delta x = \Delta y = \Delta z = \pi / 5 1 2$ </td></tr><tr><td>Temporal resolution</td><td> $\Delta t = 4 \times 1 0 ^ { - 2 }$ </td></tr></table>

Supplementary Table 9: Dataset statistics and splits (samples: train/val/test/total; storage size).
<table><tr><td>System</td><td>Samples (train/val/test/total) Storage</td></tr><tr><td>FIT</td><td>653/218/218/1089 ~70 GB</td></tr><tr><td>MHDT</td><td>707/236/236/1179 ~132 GB</td></tr><tr><td>HBDT</td><td>822/274/274/1370 ~108 GB</td></tr></table>

where D denotes a spatial filtering operator followed by uniform subsampling, and $\boldsymbol { \xi }$ represents additive measurement noise.

We employ standard coarse-graining commonly used in turbulence modeling:

• Box/Favre filtering: with width $\Delta$ (in voxels). For variable-density flows, Favre filtering uses $\phi = \rho \phi / \overline { { \rho } }$

• Gaussian filtering: isotropic or anisotropic, with standard deviations $\sigma _ { x , y , . }$ and optional random in-plane rotation.

• Spectral cutof: removing Fourier modes with $\| \boldsymbol { k } \| > k _ { c }$ prior to inverse FFT.

In addition, we include three additional coarse-graining strategies for evaluation:

• Tricubic interpolation: a smooth volumetric interpolation operator commonly used as a numerical baseline.

• Lanczos filtering: a windowed sinc-based low-pass filter ofering sharper frequency truncation.

• Uniform downsampling with noise: voxel decimation followed by additive perturbations to simulate sensor-level degradation.

Upscaling factors $s \in \{ 4 , 8 , 1 6 \}$ are used unless otherwise stated. Noise is modeled as additive Gaussian noise $\pmb { \xi } \sim \mathcal { N } ( 0 , \sigma _ { \xi } ^ { 2 } )$

## 2.5 Normalization and windowing

Each variable $v \in \{ \rho , u , b , p \}$ is standardized using training-set statistics via a z-score transformation:

$$
\hat { v } = \frac { v - \mu _ { v } } { \sigma _ { v } } ,\tag{S22}
$$

where $( \mu _ { v } , \sigma _ { v } )$ are computed from the training partition and retained for inverse normalization during inference.

For TurbSR3D, we extract spatiotemporal windows of size $\langle T _ { \mathrm { w i n } } , H , W , D \rangle$ , following deterministic train/validation/test splits $( 6 0 \% / 2 0 \% / 2 0 \% )$ over non-overlapping volumetric regions to prevent data leakage. In these experiments, we set $T _ { \mathrm { w i n } } = 8$ and $H = W = D = 6 4$ . Zero padding is used in all convolutional layers that preserve spatial dimensions, maintaining feature-map sizes throughout convolutional processing. This convention is consistent with related studies and implementations in flow reconstruction. Morimoto et al. [78] reported comparable reconstruction errors with zero and periodic padding for filter widths $H \leq 9$ in the biperiodic turbulence case they examined. Zero padding has also been used for compressible-flow reconstruction by Sofos et al. [79] and in the released BLASTNet 2.0 3D CNN implementation [24]. These precedents motivate its use as a practical implementation choice in the present framework.

## 2.6 JHTDB acquisition notes

Data are retrieved via the JHTDB SOAP services (Python client), with high-order spatial interpolation $( i . e . , \mathrm { L a g 8 } )$ and PCHIP temporal interpolation for point queries. To ensure reliable large-scale acquisition, we employ block-wise sampling without overlap, request retry with backof, and execution on the SciServer platform [80] for high-throughput access.

## 2.7 DNS of decaying homogeneous isotropic turbulence

Decaying homogeneous isotropic turbulence was simulated by solving the unforced incompressible Navier–Stokes equations in the triply periodic domain $[ 0 , 2 \pi ) ^ { 3 }$ , with nondimensional viscosity $\nu = 0 . 0 0 3 3$ . The initial velocity was a solenoidal Gaussian random field with target shell spectrum

$$
E ( k ) \propto k ^ { 4 } \exp [ - 2 ( k / k _ { 0 } ) ^ { 2 } ] , \qquad k _ { 0 } = 6 ,\tag{S23}
$$

normalized to $u _ { \mathrm { r m s } } = ( { \langle u _ { i } u _ { i } \rangle } / { 3 } ) ^ { 1 / 2 } = 1$ . The initial Taylor-scale Reynolds number was $R e _ { \lambda } =$ 101.25, defined by

$$
R e _ { \lambda } = \frac { u _ { \mathrm { r m s } } \lambda } { \nu } , \qquad \lambda = \left( \frac { 1 5 \nu u _ { \mathrm { r m s } } ^ { 2 } } { \varepsilon } \right) ^ { 1 / 2 } ,\tag{S24}
$$

where $\varepsilon = \nu \langle \partial _ { j } u _ { i } \partial _ { j } u _ { i } \rangle$

The equations were discretized on a uniform $2 5 6 ^ { 3 }$ collocated grid using fourth-order centred diferences and a direct fourth-order Laplacian for viscous difusion. Classical fourth-order Runge– Kutta integration used $\Delta t = 1 0 ^ { - 4 }$ . Incompressibility was enforced at each stage through an FFTbased projection using the compatible discrete divergence–gradient operator, with its null modes handled by a pseudoinverse. No subgrid closure, dealiasing or artificial filtering was applied.

The dataset comprised 201 velocity–pressure frames over $t = 0 . 2 0 0 { - 0 . 6 0 0 }$ , sampled at intervals of 0.002, with two additional frames at each endpoint supporting centred temporal derivatives. Across the dataset, $k _ { \operatorname* { m a x } { } } \eta$ ranged from 1.716 to 1.917, where $\eta = ( \nu ^ { 3 } / \varepsilon ) ^ { 1 / 4 }$ , ε was evaluated using Fourier derivatives, and $k _ { \mathrm { m a x } } = \pi / h = 1 2 8$ is the axial Nyquist wavenumber. This quantity characterizes nominal dissipation-scale resolution; the efective resolution depends on the finitediference operators. Grid convergence was not assessed.

## 2.8 LES simulation and LES–DNS pairing

LES was initialized from the DNS velocity at $t _ { 0 } = 0 . 5 2 2$ on a $6 4 ^ { 3 }$ periodic grid with spacing $H \ = \ 4 h$ . Initialization used a normalized, separable periodic Lanczos-3 filter [53] followed by fourfold subsampling. The one-dimensional kernel was

$$
\ell ( q ) = \left\{ \begin{array} { l l } { \mathrm { s i n c } ( q ) \mathrm { s i n c } ( q / 3 ) , } & { | q | < 3 , } \\ { 0 , } & { | q | \geq 3 , } \end{array} \right. \qquad \mathrm { s i n c } ( q ) = \frac { \mathrm { s i n } ( \pi q ) } { \pi q } .\tag{S25}
$$

Discrete weights were proportional to $\ell ( ( m - 1 . 5 ) / 4 )$ and normalized to sum to unity. Coarse-grid samples were centred at $x _ { j } = ( 4 j + 1 . 5 ) h$ along each coordinate, and the restricted velocity was projected onto the coarse-grid discrete divergence-free subspace.

The resolved velocity satisfied

$$
\partial _ { t } U _ { i } + U _ { j } \partial _ { j } U _ { i } = - \partial _ { i } \tilde { p } + \nu \nabla ^ { 2 } U _ { i } + \partial _ { j } ( 2 \nu _ { t } S _ { i j } ) , \qquad \partial _ { i } U _ { i } = 0 ,\tag{S26}
$$

with the Smagorinsky closure [81]

$$
\begin{array} { r } { \nu _ { t } = ( C _ { s } H ) ^ { 2 } ( 2 S _ { i j } S _ { i j } ) ^ { 1 / 2 } , \qquad S _ { i j } = \frac 1 2 ( \partial _ { j } U _ { i } + \partial _ { i } U _ { j } ) , \qquad C _ { s } = 0 . 1 6 . } \end{array}\tag{S27}
$$

Here, p˜ is the modified kinematic pressure incorporating the isotropic SGS stress. The LES retained the DNS viscosity, fourth-order spatial discretization, compatible pressure projection and Runge– Kutta scheme, with the SGS stress recomputed at every integration stage.

Following initialization, LES evolved independently for 160 steps with $\Delta t = 1 0 ^ { - 4 }$ . Eight outputs at $t = 0 . 5 2 4 , 0 . 5 2 6 , \dots , 0 . 5 3 8$ were paired with DNS fields at the same physical times.

The local energy flux $\Pi _ { \ell }$ was computed using a top-hat filter of width ℓ, with $\tau _ { i j } ^ { ( \ell ) } = \overline { { u _ { i } u _ { j } } } ^ { ( \ell ) }$ $\bar { u } _ { i } ^ { ( \ell ) } \bar { u } _ { j } ^ { ( \ell ) }$

## 2.9 ERA5 data access and experimental protocol

Data access. ERA5 fields are retrieved via WeatherBench2 [82] for eficient data access and preprocessing.

Downscaling setting. In the downscaling experiment, the model maps a low-resolution spatiotemporal multi-level field $\pmb { X } _ { \mathrm { L } } ~ \in ~ \mathbb { R } ^ { T \times H _ { \mathrm { L } } \times W _ { \mathrm { L } } \times P \times C }$ to the corresponding high-resolution field $\mathbf { X } _ { \mathrm { H } } \in \mathbf { \bar { \Gamma } } _ { \mathbb { R } } \mathbf { { \Gamma } } ^ { T \times H _ { \mathrm { H } } \times W _ { \mathrm { H } } \times P \times C }$ . Unless stated otherwise, $T = 8$ and $H _ { \mathrm { H } } = W _ { \mathrm { H } } = 1 2 8$ . The vertical dimension uses $P = 8$ pressure levels: 1000, 925, 850, 700, 600, 500, 400, and 300 hPa. The channel dimension C includes temperature and horizontal wind components $( u , v )$

Normalization. Normalization follows Supplementary Eq. (S22).

## 3 Additional details on experiments

## 3.1 Accurate reconstruction of turbulent flows

To complement the quantitative results in the main text (Table 1), we report the full set of quantitative evaluations across all turbulent systems, coarse-graining strategies, and upscaling factors. Supplementary Tables 10–12 present detailed comparisons between our framework and eight baselines, using four complementary metrics (PSNR, SSIM, nRMSE, and fRMSE; see Section 4.5). Higher PSNR and SSIM indicate better perceptual quality and spatial fidelity, while lower nRMSE and fRMSE reflect improved voxel-wise accuracy and spectral fidelity, respectively. For every turbulent system (FIT, MHDT, HBDT) and each coarse-graining strategy, we independently train and evaluate all learning-based methods using the same LR–HR data pairs and training schedule. No cross-dataset fine-tuning or weight sharing is applied. This ensures that all methods are compared on equal footing and that the reported numbers reflect genuine model capability rather than transfer bias. Supplementary Tables 10, 11, and 12 provide the complete spatial and point-wise benchmark results. Our method achieves the highest PSNR and SSIM and the lowest nRMSE and fRMSE among the compared methods in all 18 evaluated settings, spanning three turbulent systems, three coarse-graining strategies, and two upscaling factors per system.

Supplementary Table 10: Full quantitative comparison on FIT. Detailed super-resolution results for 4 and 8 upscaling under three coarse-graining strategies (box, Gaussian, and spectral cutof). Performance is evaluated using PSNR ( ), SSIM ( ), nRMSE ( ), and fRMSE ( ) (see Section 4.5). For each upscaling factor and coarsegraining, the best result is shown in bold. Model sizes are reported as the number of parameters in millions (M).
<table><tr><td>Factor</td><td>Coarse-gr.</td><td>Method</td><td>#Params. (M)</td><td>SSIM↑</td><td>PSNR↑</td><td>nRMSE↓</td><td>fRMSE↓</td></tr><tr><td rowspan="7"></td><td rowspan="7">Box</td><td>Tricubic</td><td></td><td>0.8321</td><td>42.28</td><td>0.1204</td><td>28.56</td></tr><tr><td>RRDB</td><td>4.2</td><td>0.9614</td><td>50.90</td><td>0.0460</td><td>11.32</td></tr><tr><td>RCAN</td><td>4.2</td><td>0.9672</td><td>51.92</td><td>0.0415</td><td>10.23</td></tr><tr><td>EDSR</td><td>4.2</td><td>0.9661</td><td>51.77</td><td>0.0423</td><td>10.41</td></tr><tr><td>FNO</td><td>34.0</td><td>0.9109</td><td>46.61</td><td>0.0730</td><td>17.27</td></tr><tr><td>PhySR</td><td>3.3</td><td>0.9590</td><td>50.66</td><td>0.0476</td><td>11.63</td></tr><tr><td>DiffSR</td><td>7.3</td><td>0.9651</td><td>51.62</td><td>0.0423</td><td>10.34</td></tr><tr><td rowspan="12">4× Spectral</td><td>SwinIR</td><td>3.8</td><td>0.9689</td><td>51.46</td><td>0.0421</td><td>10.23</td></tr><tr><td>Ours</td><td>3.7</td><td>0.9834</td><td>55.09</td><td>0.0291</td><td>7.22</td></tr><tr><td>Tricubic</td><td></td><td>0.7630</td><td>40.26</td><td>0.1501</td><td>35.47</td></tr><tr><td>RRDB</td><td>4.2</td><td>0.8919</td><td>46.41</td><td>0.0770</td><td>18.73</td></tr><tr><td>RCAN</td><td>4.2</td><td>0.8932</td><td>46.60</td><td>0.0755</td><td>18.39</td></tr><tr><td>EDSR</td><td>4.2</td><td>0.8930</td><td>46.54</td><td>0.0759</td><td>18.45</td></tr><tr><td>FNO</td><td>34.0</td><td>0.8394</td><td>43.51</td><td>0.1036</td><td>24.45</td></tr><tr><td rowspan="7"></td><td>PhySR</td><td>3.3</td><td>0.9044</td><td>46.99</td><td>0.0722</td><td>17.56</td></tr><tr><td>DiffSR</td><td>7.3</td><td>0.8889</td><td>45.84</td><td>0.0810</td><td>19.54</td></tr><tr><td>SwinIR</td><td>3.8</td><td>0.9131</td><td>47.07</td><td>0.0704</td><td>17.16</td></tr><tr><td>Ours</td><td>3.7</td><td>0.9251</td><td>48.41</td><td>0.0615</td><td>15.09</td></tr><tr><td>Tricubic</td><td></td><td>0.8163</td><td>41.45</td><td>0.1325</td><td>31.39</td></tr><tr><td>RRDB</td><td>4.2</td><td>0.9576</td><td>50.28</td><td>0.0494</td><td>12.13</td></tr><tr><td>RCAN</td><td>4.2</td><td>0.9682</td><td>51.78</td><td>0.0421</td><td>10.34</td></tr><tr><td rowspan="8">Gaussian</td><td></td><td>4.2</td><td></td><td></td><td></td><td></td></tr><tr><td>EDSR</td><td></td><td>0.9645</td><td>51.92</td><td>0.0445</td><td>10.93</td></tr><tr><td>FNO</td><td>34.0</td><td>0.9100</td><td>46.43</td><td>0.0745</td><td>17.67</td></tr><tr><td>PhySR</td><td>3.3</td><td>0.9585</td><td>50.33</td><td>0.0494</td><td>12.03</td></tr><tr><td>DiffSR</td><td>7.3</td><td>0.9646</td><td>51.32</td><td>0.0438</td><td>10.69</td></tr><tr><td>SwinIR</td><td>3.8</td><td>0.9698</td><td>51.29</td><td>0.0429</td><td>10.40</td></tr><tr><td>Ours</td><td>3.7</td><td>0.9833</td><td>54.75</td><td>0.0300</td><td>7.43</td></tr><tr><td>Tricubic</td><td></td><td>0.5974</td><td>36.81</td><td>0.2242</td><td>52.83</td></tr><tr><td rowspan="8">Box</td><td>RRDB</td><td>4.3</td><td>0.6826</td><td>39.31</td><td>0.1712</td><td>40.77</td></tr><tr><td>RCAN</td><td>5.1</td><td>0.6701</td><td>39.10</td><td>0.1753</td><td>41.81</td></tr><tr><td>EDSR</td><td>4.7</td><td>0.6791</td><td>40.29</td><td>0.1693</td><td>39.36</td></tr><tr><td>FNO</td><td>34.3</td><td>0.6936</td><td>39.49</td><td>0.1657</td><td>38.91</td></tr><tr><td>PhySR</td><td>5.4</td><td>0.7291</td><td>40.54</td><td>0.1486</td><td>35.44</td></tr><tr><td>DiffSR</td><td>7.3</td><td>0.7480</td><td>40.82</td><td>0.1440</td><td>34.46</td></tr><tr><td>SwinIR Ours</td><td>4.0</td><td>0.7300</td><td>40.39</td><td>0.1499</td><td>35.67</td></tr><tr><td></td><td>3.9</td><td>0.7963</td><td>42.46</td><td>0.1195</td><td>28.63</td></tr><tr><td rowspan="8">8× Spectral</td><td>Tricubic</td><td>一</td><td>0.5459</td><td>35.80</td><td>0.2505</td><td>58.87</td></tr><tr><td>RRDB</td><td>4.3</td><td>0.5864</td><td>37.23</td><td>0.2156</td><td>51.06</td></tr><tr><td>RCAN</td><td>5.1</td><td>0.5716</td><td>36.99</td><td>0.2209</td><td>52.51</td></tr><tr><td>EDSR</td><td>4.7</td><td>0.5821</td><td>37.26</td><td>0.2140</td><td>50.64</td></tr><tr><td>FNO</td><td>34.3</td><td>0.5963</td><td>37.42</td><td>0.2098</td><td>49.16</td></tr><tr><td>PhySR</td><td>5.4</td><td>0.6086</td><td>37.90</td><td>0.2002</td><td>47.51</td></tr><tr><td>DiffSR</td><td>7.3</td><td>0.5802</td><td>36.81</td><td>0.2246</td><td>52.41</td></tr><tr><td>SwinIR</td><td>4.0</td><td>0.6085</td><td>37.49</td><td>0.2067</td><td>48.70</td></tr><tr><td rowspan="8">Gaussian</td><td>Ours</td><td>3.9</td><td>0.6372</td><td>38.42</td><td>0.1884</td><td>44.69</td></tr><tr><td>Tricubic RRDB</td><td></td><td>0.5991</td><td>36.68</td><td>0.2278</td><td>53.69</td></tr><tr><td>RCAN</td><td>4.3 5.1</td><td>0.6815 0.6682</td><td>39.12 38.86</td><td>0.1749 0.1802</td><td>41.63 43.00</td></tr><tr><td>EDSR</td><td>4.7</td><td>0.6743</td><td>39.05</td><td>0.1752</td><td>41.66</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FNO</td><td>34.3</td><td>0.6958</td><td>39.37</td><td>0.1679</td><td>39.37</td></tr><tr><td>PhySR</td><td>5.4</td><td>0.7290</td><td>40.36</td><td>0.1517</td><td>36.17</td></tr><tr><td>DiffSR</td><td>7.3</td><td>0.7499</td><td>40.72</td><td>0.1456</td><td>34.80</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>36.52</td></tr><tr><td rowspan="10"></td><td>SwinIR</td><td>4.0</td><td>0.7284</td><td>40.18</td><td>0.1535</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>28.12</td></tr><tr><td>Ours</td><td>3.9</td><td>0.8093</td><td>42.61</td><td>0.1173</td></table>

Supplementary Table 11: Full quantitative comparison on MHDT. Detailed super-resolution results for 4 and 8 upscaling under three coarse-graining strategies (box, Gaussian, and spectral cutof). Performance is evaluated using PSNR ( ), SSIM ( ), nRMSE ( ), and fRMSE ( ) (see Section 4.5). For each upscaling factor and coarse-graining, the best result is shown in bold. Model sizes are reported as the number of parameters in millions (M).
<table><tr><td>Factor</td><td>Coarse-gr.</td><td>Method</td><td>#Params. (M)</td><td>SSIM↑</td><td>PSNR↑</td><td>nRMSE↓</td><td>fRMSE↓</td></tr><tr><td rowspan="7"></td><td rowspan="7">Box</td><td>Tricubic RRDB</td><td></td><td>0.7643</td><td>38.39</td><td>0.1838</td><td>16.29</td></tr><tr><td>RCAN</td><td>4.2</td><td>0.9482</td><td>47.34</td><td>0.0678</td><td>5.92 5.99</td></tr><tr><td></td><td>4.2</td><td>0.9464</td><td>47.28</td><td>0.0688</td><td></td></tr><tr><td>EDSR</td><td>4.3</td><td>0.9448</td><td>47.15</td><td>0.0699</td><td>6.09</td></tr><tr><td>FNO</td><td>34.0</td><td>0.8687</td><td>42.41</td><td>0.1166</td><td>10.28</td></tr><tr><td>PhySR</td><td>3.5</td><td>0.9265</td><td>45.37</td><td>0.0840</td><td>7.39</td></tr><tr><td>DiffSR</td><td>7.3</td><td>0.9395</td><td>46.58</td><td>0.0731</td><td>6.45</td></tr><tr><td rowspan="10">4×</td><td>SwinIR Ours</td><td></td><td>3.8 3.7</td><td>0.9327 0.9579</td><td>45.84</td><td>0.0792</td><td>6.95</td></tr><tr><td></td><td></td><td></td><td></td><td>48.28</td><td>0.0603</td><td>5.29</td></tr><tr><td>Tricubic RRDB</td><td></td><td>0.6816</td><td>36.74</td><td>0.2204</td><td></td><td>19.70</td></tr><tr><td>RCAN</td><td></td><td>4.2</td><td>0.8552</td><td>42.71</td><td>0.1130</td><td>9.90</td></tr><tr><td>Spectral</td><td></td><td>4.2</td><td>0.8536</td><td>42.69</td><td>0.1135</td><td>9.94</td></tr><tr><td rowspan="6"></td><td>EDSR</td><td>4.3</td><td>0.8484</td><td>42.47</td><td>0.1163</td><td>10.19</td></tr><tr><td>FNO</td><td>34.0</td><td>0.7658</td><td>39.50</td><td>0.1612</td><td>14.27</td></tr><tr><td>PhySR</td><td>3.5</td><td>0.8439</td><td>42.22</td><td>0.1188</td><td>10.46</td></tr><tr><td>DiffSR</td><td>7.3</td><td>0.8168</td><td>41.16</td><td>0.1339</td><td>11.77</td></tr><tr><td>SwinIR</td><td>3.8</td><td>0.8498</td><td>42.17</td><td>0.1190</td><td>10.48</td></tr><tr><td>Ours</td><td>3.7</td><td>0.8671</td><td>43.12</td><td>0.1071</td><td>9.43</td></tr><tr><td rowspan="8">Gaussian</td><td>Tricubic</td><td></td><td>0.7456</td><td>37.67</td><td>0.2000</td><td>17.78</td></tr><tr><td>RRDB</td><td>4.2</td><td>0.9489</td><td>47.18</td><td>0.0692</td><td>6.03</td></tr><tr><td>RCAN</td><td>4.2</td><td>0.9455</td><td>46.98</td><td>0.0712</td><td>6.20</td></tr><tr><td>EDSR</td><td>4.3</td><td>0.9450</td><td>46.92</td><td>0.0718</td><td>6.25</td></tr><tr><td></td><td>34.0</td><td>0.8689</td><td>42.26</td><td>0.1188</td><td>10.48</td></tr><tr><td>FNO</td><td>3.5</td><td>0.9288</td><td>45.25</td><td>0.0852</td><td>7.49</td></tr><tr><td>PhySR DiffSR</td><td>7.3</td><td>0.9380</td><td>46.28</td><td>0.0756</td><td>6.66</td></tr><tr><td>SwinIR</td><td>3.8</td><td>0.9335</td><td>45.71</td><td>0.0804</td><td>7.06</td></tr><tr><td rowspan="8">Box</td><td>Ours</td><td>3.7</td><td>0.9582</td><td>48.06</td><td>0.0620</td><td>5.43</td></tr><tr><td>Tricubic</td><td></td><td>0.4908</td><td>33.05</td><td>0.3372</td><td>30.22</td></tr><tr><td>RRDB</td><td>4.3</td><td>0.6560</td><td>36.85</td><td>0.2205</td><td>19.52</td></tr><tr><td>RCAN</td><td>5.1</td><td>0.6414</td><td>36.55</td><td>0.2285</td><td>20.23</td></tr><tr><td>EDSR</td><td>4.7</td><td>0.6435</td><td>36.68</td><td>0.2250</td><td>19.88</td></tr><tr><td>FNO</td><td>34.2</td><td>0.6009</td><td>35.62</td><td>0.2525</td><td>22.44</td></tr><tr><td>PhySR</td><td>7.5</td><td>0.6036</td><td>35.91</td><td>0.2443</td><td>21.70</td></tr><tr><td rowspan="8">8× Spectral</td><td>DiffSR</td><td>7.3</td><td>0.6629</td><td>37.02</td><td>0.2153</td><td>18.96</td></tr><tr><td>SwinIR</td><td>4.0</td><td>0.6047</td><td>35.87</td><td>0.2447</td><td>21.67</td></tr><tr><td>Ours</td><td>3.9</td><td>0.6713</td><td>37.28</td><td>0.2094</td><td>18.50</td></tr><tr><td>Tricubic</td><td>一</td><td>0.4327</td><td>32.35</td><td>0.3640</td><td>32.69</td></tr><tr><td>RRDB</td><td>4.3</td><td>0.5003</td><td>34.06</td><td>0.3013</td><td>26.83</td></tr><tr><td>RCAN</td><td>5.1</td><td>0.4843</td><td>33.78</td><td>0.3105</td><td>27.67</td></tr><tr><td>EDSR</td><td>4.7</td><td>0.4931</td><td>33.99</td><td>0.3036</td><td>26.98</td></tr><tr><td>FNO</td><td>34.2</td><td>0.4822</td><td>33.68</td><td>0.3141</td><td>27.97</td></tr><tr><td rowspan="8"></td><td>PhySR</td><td>7.5</td><td>0.4731</td><td>33.75</td><td>0.3117</td><td>27.69</td></tr><tr><td>DiffSR</td><td>7.3</td><td>0.4739</td><td>33.45</td><td>0.3226</td><td>28.45</td></tr><tr><td>SwinIR</td><td>4.0</td><td>0.4212</td><td>32.40</td><td>0.3628</td><td>32.06</td></tr><tr><td>Ours</td><td>3.9</td><td>0.5052</td><td>34.21</td><td>0.2957</td><td>26.20</td></tr><tr><td>Tricubic RRDB</td><td></td><td>0.4881</td><td>33.16</td><td>0.3327</td><td>29.80</td></tr><tr><td></td><td>4.3</td><td>0.6602</td><td>36.71</td><td>0.2246</td><td>19.87</td></tr><tr><td>RCAN</td><td>5.1</td><td>0.6465</td><td>36.38</td><td>0.2330</td><td>20.64</td></tr><tr><td>EDSR</td><td>4.7</td><td>0.6479</td><td>36.53</td><td>0.2290</td><td>20.24</td></tr><tr><td>FNO</td><td>34.2</td><td>0.6050</td><td>35.51</td><td>0.2559</td><td>22.73</td></tr><tr><td></td><td>PhySR DiffSR</td><td>7.5 7.3</td><td>0.6079</td><td>35.79</td><td>0.2479</td><td>22.00</td></tr><tr><td></td><td></td><td>4.0</td><td>0.6699 0.6045</td><td>36.94 35.70</td><td>0.2175 0.2496</td><td>19.15</td></tr><tr><td></td><td>SwinIR</td><td>3.9</td><td></td><td></td><td></td><td>22.07</td></tr><tr><td>Ours</td><td></td><td></td><td>0.6825</td><td>37.27</td><td>0.2099</td><td>18.54</td></tr></table>

Supplementary Table 12: Full quantitative comparison on HBDT. Detailed super-resolution results for 8 and 16 upscaling under three coarse-graining strategies (Favre, Gaussian, and spectral cutof). Performance is evaluated using PSNR ( ), SSIM ( ), nRMSE ( ), and fRMSE ( ) (see Section 4.5). For each upscaling factor and coarse-graining, the best result is shown in bold. Model sizes are reported as the number of parameters in millions (M).
<table><tr><td>Factor</td><td>Coarse-gr.</td><td>Method</td><td>#Params. (M)</td><td>SSIM↑</td><td>PSNR↑</td><td>nRMSE↓</td><td>fRMSE↓</td></tr><tr><td rowspan="7"></td><td rowspan="7">Favre</td><td>Tricubic RRDB</td><td></td><td>0.9226</td><td>46.87</td><td>0.0933</td><td>3.89</td></tr><tr><td></td><td>4.3</td><td>0.9778</td><td>57.95</td><td>0.0307</td><td>1.38</td></tr><tr><td>RCAN</td><td>5.1</td><td>0.9800</td><td>61.57</td><td>0.0262</td><td>1.27</td></tr><tr><td>EDSR</td><td>4.7</td><td>0.9779</td><td>59.22</td><td>0.0286</td><td>1.34</td></tr><tr><td>FNO</td><td>34.3</td><td>0.9264</td><td>51.04</td><td>0.0544</td><td>2.14</td></tr><tr><td>PhySR</td><td>6.1</td><td>0.9668</td><td>59.38</td><td>0.0316</td><td>1.45</td></tr><tr><td>DiffSR</td><td>7.3</td><td>0.9655</td><td>57.80</td><td>0.0264</td><td>1.37</td></tr><tr><td rowspan="10">8×</td><td>SwinIR Ours</td><td></td><td>4.0 3.9</td><td>0.9672</td><td>55.65</td><td>0.0353</td><td>1.47 1.01</td></tr><tr><td></td><td></td><td></td><td>0.9862</td><td>66.63</td><td>0.0195</td><td></td></tr><tr><td>Tricubic RRDB</td><td></td><td>0.8382</td><td>44.54</td><td></td><td>0.1196</td><td>4.76</td></tr><tr><td>RCAN</td><td></td><td>4.3</td><td>0.9619</td><td>55.12</td><td>0.0432</td><td>1.96</td></tr><tr><td>Spectral</td><td></td><td>5.1</td><td>0.9625</td><td>57.46</td><td>0.0410</td><td>1.97</td></tr><tr><td rowspan="6"></td><td>EDSR</td><td>4.7</td><td>0.9611</td><td>55.71</td><td>0.0428</td><td>1.99</td></tr><tr><td>FNO</td><td>34.3</td><td>0.8936</td><td>47.60</td><td>0.0812</td><td>3.05</td></tr><tr><td>PhySR</td><td>6.1</td><td>0.9478</td><td>55.79</td><td>0.0452</td><td>2.01</td></tr><tr><td>DiffSR</td><td>7.3</td><td>0.9389</td><td>52.67</td><td>0.0496</td><td>1.79</td></tr><tr><td>SwinIR</td><td>4.0</td><td>0.9483</td><td>52.51</td><td>0.0521</td><td>2.20</td></tr><tr><td>Ours</td><td>3.9</td><td>0.9702</td><td>60.12</td><td>0.0348</td><td>1.68</td></tr><tr><td rowspan="8">Gaussian</td><td>Tricubic</td><td></td><td>0.9211</td><td>46.70</td><td>0.0959</td><td>3.97</td></tr><tr><td>RRDB</td><td>4.3</td><td>0.9782</td><td>57.54</td><td>0.0316</td><td>1.42</td></tr><tr><td>RCAN</td><td>5.1</td><td>0.9794</td><td>61.31</td><td>0.0275</td><td>1.33</td></tr><tr><td>EDSR</td><td>4.7</td><td>0.9777</td><td>58.80</td><td>0.0299</td><td>1.40</td></tr><tr><td></td><td>34.3</td><td>0.9296</td><td>51.30</td><td>0.0535</td><td>2.14</td></tr><tr><td>FNO PhySR</td><td>6.1</td><td>0.9531</td><td>56.39</td><td>0.0389</td><td>1.73</td></tr><tr><td>DiffSR</td><td>7.3</td><td>0.9586</td><td>57.25</td><td>0.0300</td><td>1.39</td></tr><tr><td>SwinIR</td><td>4.0</td><td>0.9668</td><td>55.57</td><td>0.0354</td><td>1.49</td></tr><tr><td rowspan="8">Favre</td><td>Ours</td><td>3.9</td><td>0.9864</td><td>66.37</td><td>0.0200</td><td>1.03</td></tr><tr><td>Tricubic</td><td></td><td>0.8364</td><td>42.00</td><td>0.1666</td><td>6.87</td></tr><tr><td>RRDB</td><td>4.4</td><td>0.9037</td><td>47.78</td><td>0.0940</td><td>4.14</td></tr><tr><td>RCAN</td><td>6.0</td><td>0.8626</td><td>46.65</td><td>0.1396</td><td>5.03</td></tr><tr><td>EDSR</td><td>5.1</td><td>0.8639</td><td>46.42</td><td>0.1301</td><td>5.00</td></tr><tr><td>FNO</td><td>5.1</td><td>0.8393</td><td>43.85</td><td>0.1224</td><td>4.94</td></tr><tr><td>PhySR DiffSR</td><td>27.9</td><td>0.8173</td><td>45.37</td><td>0.1159</td><td>4.93</td></tr><tr><td>SwinIR</td><td>7.3 4.2</td><td>0.9073 0.8400</td><td>48.08</td><td>0.0860</td><td>3.92</td></tr><tr><td rowspan="8">16× Spectral</td><td>Ours</td><td>4.1</td><td>0.9265</td><td>44.19 52.42</td><td>0.1228</td><td>4.78 3.49</td></tr><tr><td></td><td></td><td></td><td></td><td>0.0739</td><td></td></tr><tr><td>Tricubic</td><td>一</td><td>0.8378</td><td>41.96</td><td>0.1671</td><td>6.87</td></tr><tr><td>RRDB RCAN</td><td>4.4</td><td>0.8545</td><td>44.39</td><td>0.1354</td><td>5.77</td></tr><tr><td></td><td>6.0</td><td>0.8357</td><td>44.20</td><td>0.1438</td><td>6.07</td></tr><tr><td>EDSR</td><td>5.1</td><td>0.8227</td><td>43.38</td><td>0.1476</td><td>6.06</td></tr><tr><td>FNO</td><td>5.1</td><td>0.7770</td><td>40.87</td><td>0.1767</td><td>6.61</td></tr><tr><td>PhySR</td><td>27.9</td><td>0.7724</td><td>43.13</td><td>0.1511</td><td>6.22</td></tr><tr><td>DiffSR SwinIR</td><td>7.3</td><td>0.8605</td><td>43.96</td><td>0.1366</td><td>5.82</td></tr><tr><td rowspan="8">Gaussian</td><td>Ours</td><td>4.2 4.1</td><td>0.8085 0.8778</td><td>42.16 47.53</td><td>0.1602 0.1127</td><td>6.16 5.00</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Tricubic RRDB</td><td>4.4</td><td>0.7588 0.9011</td><td>39.97 47.25</td><td>0.2074</td><td>8.24</td></tr><tr><td>RCAN</td><td>6.0</td><td>0.8876</td><td>47.33</td><td>0.0985 0.1055</td><td>4.31 4.56</td></tr><tr><td>EDSR</td><td>5.1</td><td>0.8532</td><td>45.71</td><td>0.1452</td><td>5.11</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FNO</td><td>5.1</td><td>0.8393</td><td>43.85</td><td>0.1224</td><td>4.94</td></tr><tr><td>PhySR</td><td>27.9</td><td>0.8210</td><td>45.40</td><td>0.1165</td><td>4.98</td></tr><tr><td>DiffSR</td><td>7.3</td><td>0.9020</td><td>47.62</td><td>0.0984</td><td>4.19</td></tr><tr><td></td><td>SwinIR Ours</td><td>4.2 4.1</td><td>0.8462 0.9268</td><td>44.06 52.25</td><td>0.1242 0.0754</td><td>4.86 3.55</td></tr></table>

## 3.2 Noise robustness

Super-resolution analysis can also be viewed as a framework for reconstructing global flow fields from spatially sparse measurements, with LR flow data representing sparse or noisy sensor observations [83]. This perspective motivates examining how reconstruction performance depends on the noise magnitude in the LR input. We evaluate this dependence using FIT at 4× upscaling with box-filtered LR inputs. After channel-wise standardization using the training-set statistics, additive Gaussian noise $\pmb { \xi } \sim \mathcal { N } ( 0 , \sigma _ { \xi } ^ { 2 } )$ is added to the LR input, where $\sigma _ { \xi }$ is expressed relative to the standardized channel scale. All learned methods use the same noise augmentation during training, with $\sigma _ { \xi }$ sampled with equal probability from $\{ 0 , 0 . 0 1 , 0 . 0 3 , 0 . 0 5 , 0 . 1 0 \}$ . At evaluation, reconstruction performance is measured separately at each of these noise levels.

![](images/c8ae21e3d427270108dc5d62fa0ce5a7d50abc7fc16cc9b7e9c14661102d9389.jpg)  
Supplementary Fig. 4: Noise robustness on FIT. Mean nRMSE at 4 upscaling as a function of the noise magnitude $\sigma _ { \xi }$ in box-filtered LR inputs. Error bars denote 95% confidence intervals.

Supplementary Table 13: Quantitative comparison of noise robustness on FIT. Mean nRMSE at 4 upscaling for box-filtered LR inputs. ∆nRMSE denotes the absolute increase from $\sigma _ { \xi } = 0 \mathrm { { t o } } 0 . 1 0$ . The best result in each column is shown in bold.
<table><tr><td rowspan="2">Method</td><td colspan="5">Noise magnitude  $\sigma _ { \xi }$ </td><td rowspan="2">∆nRMSE</td></tr><tr><td>0</td><td>0.01</td><td>0.03</td><td>0.05</td><td>0.10</td></tr><tr><td>Ours</td><td>0.0345</td><td>0.0355</td><td>0.0400</td><td>0.0453</td><td>0.0587</td><td>0.0242</td></tr><tr><td>SwinIR</td><td>0.0447</td><td>0.0479</td><td>0.0582</td><td>0.0684</td><td>0.0912</td><td>0.0466</td></tr><tr><td>PhySR</td><td>0.0548</td><td>0.0556</td><td>0.0603</td><td>0.0669</td><td>0.0851</td><td>0.0302</td></tr><tr><td>DiffSR</td><td>0.0489</td><td>0.0523</td><td>0.0692</td><td>0.0834</td><td>0.1110</td><td>0.0621</td></tr><tr><td>FNO</td><td>0.0878</td><td>0.0883</td><td>0.0915</td><td>0.0976</td><td>0.1217</td><td>0.0339</td></tr></table>

Under this common training and evaluation protocol, our model achieves the lowest nRMSE at every examined noise magnitude and the smallest absolute increase in nRMSE from $\sigma _ { \xi } = 0$ to 0.10 (Supplementary Fig. 4; Supplementary Table 13). Its mean nRMSE increases from 0.0345 to 0.0587, an absolute increase of 0.0242, whereas the increases for the other methods range from

0.0302 to 0.0621. At $\sigma _ { \xi } = 0 . 1 0$ , its nRMSE is 0.0587, 31.0% lower than PhySR, the second-best method at this noise magnitude.

## 3.3 Ablation study of the multiscale neural operator

We use FIT at 4× upscaling as a controlled case for attributing the two spatial operators. The full model is compared with $\mathrm { w } / \mathrm { o } \mathcal { O } _ { s } ,$ which removes the global Fourier operator, and $\mathrm { w } / \mathrm { o } \mathcal { O } _ { c } ,$ which removes the local three-dimensional convolutional operator. All three models use the same training and evaluation settings.

We evaluate the ablations as functions of the physical wavenumber k and summarize the metrics in three ranges: Low $( k < 6 4 )$ , Mid $( 6 4 \leq k \leq 1 4 4 )$ , and High $( k > 1 4 4 )$ . Velocity nRMSE measures reconstruction error across spatial scales, while vorticity MAE additionally probes velocity gradients and rotational structures. Each table entry is the mean within the corresponding range.

![](images/8bff2b587d576424091819604c25ccf5c5c571fe4cfb601a92cb72515f57a114.jpg)  
Supplementary Fig. 5: Ablation study of the multiscale neural operator on FIT. The full model is compared with models without the spectral operator $\mathcal { O } _ { s }$ or the convolutional operator $\mathcal { O } _ { c }$ at 4 upscaling with spectral-cutof coarse-graining. Left, Velocity nRMSE versus physical wavenumber. Right, Vorticity MAE versus physical wavenumber.

Supplementary Table 14: Scale-resolved errors for the multiscale neural operator ablations. The comparison uses FIT at 4 with spectral-cutof coarse-graining. Entries are mean errors in the Low $( k < 6 4 )$ , Mid $( 6 4 \leq k \leq 1 4 4 )$ , and High $( k > 1 4 4 )$ physical-wavenumber ranges. Arrows give the relative increase from the Full model for the same metric and range.
<table><tr><td rowspan="2">Model</td><td colspan="3">Velocity nRMSE</td><td colspan="3">Vorticity MAE</td></tr><tr><td>Low</td><td>Mid</td><td> $\mathbf { H i g h }$ </td><td>Low</td><td>Mid</td><td>High</td></tr><tr><td>Full</td><td>0.0122</td><td>0.1725</td><td>0.3440</td><td>2.8059</td><td>12.6611</td><td>10.2234</td></tr><tr><td>w/o Os</td><td> $0 . 0 2 0 7 ^ { \uparrow 6 9 . 9 \% }$ </td><td> $0 . 1 9 1 4 ^ { \uparrow 1 1 . 0 \% }$ </td><td> $0 . 3 4 8 5 ^ { \uparrow 1 . 3 \% }$ </td><td> $4 . 6 6 1 9 ^ { \uparrow 6 6 . 1 \% }$ </td><td> $1 4 . 3 7 5 3 ^ { \uparrow 1 3 . 5 \% }$ </td><td> $1 0 . 3 0 5 8 ^ { \uparrow 0 . 8 \% }$ </td></tr><tr><td>w/o Oc</td><td> $0 . 0 1 4 0 ^ { \uparrow 1 4 . 9 \% }$ </td><td> $0 . 2 1 3 7 ^ { \uparrow 2 3 . 9 \% }$ </td><td> $0 . 3 7 9 4 ^ { \uparrow 1 0 . 3 \% }$ </td><td> $3 . 0 8 9 8 ^ { \uparrow 1 0 . 1 \% }$ </td><td> $1 5 . 0 5 7 6 ^ { \uparrow 1 8 . 9 \% }$ </td><td> $1 1 . 2 5 8 9 ^ { \uparrow 1 0 . 1 \% }$ </td></tr></table>

The component-removal efects vary systematically across the three ranges (Supplementary Fig. 5 and Supplementary Table 14). The Low range is most sensitive to removing $\mathcal { O } _ { s }$ : velocity and vorticity errors increase by 69.9% and 66.1%, versus 14.9% and 10.1% without ${ \mathcal { O } } _ { c } .$ The Mid and High ranges are more sensitive to removing $\mathcal { O } _ { c } ,$ , with increases of $2 3 . 9 \% / 1 8 . 9 \%$ and $1 0 . 3 \% / 1 0 . 1 \%$ respectively, versus $1 1 . 0 \% / 1 3 . 5 \%$ and 1.3%/0.8% without $\mathcal { O } _ { s }$ . Thus, in FIT, $\mathcal { O } _ { s }$ is most influential at the largest represented scales through global coupling, and $\mathcal { O } _ { c }$ at intermediate and small scales, including velocity-gradient and rotational content. Both removals afect the full spectrum, and the full model is best at every evaluated wavenumber, indicating complementary and overlapping roles. Comparison with an alternative multiscale design. We replace every multiscale neural operator block with a three-dimensional multiscale dense residual block derived from MS ESRGAN [45]. The replacement combines parallel convolutions with diferent receptive-field sizes, dense feature connections, and residual scaling. The temporal module, reconstruction head, loss, and training and evaluation settings are held fixed. The full model and the replacement contain 3.7 M and 3.8 M parameters, respectively, providing a comparison at closely matched capacity.

On FIT at 4×, the replacement increases overall nRMSE from 0.0615 to 0.0739 (20.1%). The proposed spatial design also yields lower mean shell-wise velocity nRMSE in each of the Low, Mid, and High ranges (Supplementary Table 15) and reduces the high-wavenumber energy underestimation of the replacement (Supplementary Fig. 6). Together with the component-removal results, this comparison supports the value of combining global Fourier coupling with local convolutional refinement for recovering the multiscale content of this flow.

Supplementary Table 15: Comparison of Ours and MS-ESRGAN spatial designs at closely matched capacity. Ours denotes the proposed multiscale neural operator; MS-ESRGAN denotes the three-dimensional multiscale replacement derived from MS-ESRGAN. Overall nRMSE is evaluated on the complete FIT 4 spectral-cutof test fields. Velocity nRMSE entries are mean shell-wise errors in the Low $( k < 6 4 )$ , Mid $( 6 4 \leq k \leq 1 4 4 )$ , and High $( k > 1 4 4 )$ ranges. The best result in each column is shown in bold.
<table><tr><td rowspan="2">Spatial design</td><td rowspan="2">Overall</td><td colspan="3">Velocity nRMSE</td></tr><tr><td>nRMSE Low</td><td>Mid</td><td>High</td></tr><tr><td>Ours</td><td>0.0615</td><td>0.0122</td><td>0.1725</td><td>0.3440</td></tr><tr><td>MS-ESRGAN</td><td>0.0739</td><td>0.0139</td><td>0.2174</td><td>0.3803</td></tr></table>

![](images/376c422d08dd610136b8398cbd8861620d1195dfb108ad2907ad1fc0c2f1c7ac.jpg)

![](images/1f6f6b19fd87e20c8f8828b72b981922a41fad76abfc6ba41103712805ca8c59.jpg)  
Supplementary Fig. 6: Comparison of spatial modules on FIT at 4 with spectral-cutof coarsegraining. The proposed multiscale neural operator (MNO) is compared with an MS-ESRGAN-derived threedimensional multiscale replacement (labelled MS-ESRGAN) at closely matched capacity. Left, shell-wise velocity nRMSE. Right, kinetic-energy spectra; the inset magnifies the high-wavenumber range. Ref. denotes the reference field, and the black dashed line indicates a slope of $- 5 / 3$

## 3.4 Ablation and sensitivity study of the state space model

Contribution of state-space-model augmentation. The full model (Ours) is compared with a model trained without state space model (SSM) augmentation (Ours (w/o SSM)) on FIT at 4× and 8×.

Supplementary Table 16 reports the direct ablation alongside SwinIR, DifSR, and PhySR. Relative to Ours $( \mathrm { w } / \mathrm { o } \ \mathrm { S S M } )$ , the full model reduces nRMSE by 14.36% and tRMSE by 16.05% at $4 \times ;$ the corresponding reductions at 8× are 12.87% and 6.52%. Percentages are calculated from the unrounded measurements. The full model also achieves the lowest values of both metrics among all five evaluated models at both factors. These results support the contribution of SSM augmentation to field reconstruction and recovery of reference temporal increments in the tested FIT settings.

Supplementary Table 16: SSM-removal ablation and baseline comparisons on FIT at 4 and 8 . Inputs are box-filtered. nRMSE measures field reconstruction error and tRMSE measures error in frame-to-frame increments relative to the reference; lower values are preferred. Values are rounded to four decimal places, and the best value in each column is bold.
<table><tr><td rowspan="2">Model</td><td colspan="2">FIT 4×</td><td colspan="2">FIT 8×</td></tr><tr><td>nRMSE↓</td><td>tRMSE↓</td><td>nRMSE↓</td><td>tRMSE↓</td></tr><tr><td>SwinIR</td><td>0.0421</td><td>0.0052</td><td>0.1499</td><td>0.0103</td></tr><tr><td>DiffSR</td><td>0.0423</td><td>0.0095</td><td>0.1440</td><td>0.0244</td></tr><tr><td>PhySR</td><td>0.0476</td><td>0.0060</td><td>0.1486</td><td>0.0106</td></tr><tr><td>Ours (w/o SSM)</td><td>0.0340</td><td>0.0049</td><td>0.1371</td><td>0.0101</td></tr><tr><td>Ours</td><td>0.0291</td><td>0.0041</td><td>0.1195</td><td>0.0094</td></tr></table>

Supplementary Fig. 7 shows that Ours closely matches the reference structures and preserves their smooth evolution over time. The model without SSM and the other baselines exhibit varying degrees of local blurring, deformation, and reduced temporal smoothness. Together with the lower tRMSE, these observations support the contribution of SSM augmentation to temporal fidelity.

![](images/ad7ebc7fa2f7cc12002a22c4d1c09a3e7ef9a0419e2e947ac58cb84d048cabf2.jpg)  
Supplementary Fig. 7: Time–space visualization of the SSM-removal ablation on FIT. The left panel shows the reference; the upper row compares Ours, Ours (w/o SSM), SwinIR, and DifSR, and the lower row shows the corresponding reconstruction errors. Time t runs horizontally and position x vertically.

Comparison with alternative temporal modules. We compare the proposed state space model with two temporal alternatives, ConvLSTM [76] and a causal Transformer decoder [84], on FIT at 4× upscaling. Only the temporal module is replaced; the remaining architecture, data partitions, sequence length, preprocessing, loss, and training and evaluation settings are identical.

Both the temporal-module and total parameter counts are closely matched, with 3.7 M parameters for the state space model variant and 3.8 M for each alternative. ConvLSTM represents gated convolutional recurrence, and the causal Transformer represents temporal aggregation through masked self-attention.

Supplementary Table 17: Comparison of temporal models on FIT at 4 with box-filtered inputs at closely matched parameter counts. Causal Transformer denotes the Transformer decoder baseline; parameter counts refer to the complete networks. Metrics follow Section 4.5 and Supplementary Note 1.9; arrows indicate the preferred direction, and the best value for each metric is in bold.
<table><tr><td>Temporal model</td><td>Params (M)</td><td>nRMSE↓</td><td>SSIM ↑</td><td>PSNR ↑</td><td>tRMSE↓</td><td>fRMSE↓</td></tr><tr><td>State space model</td><td>3.7</td><td>0.0291</td><td>0.9834</td><td>55.09</td><td>0.0041</td><td>7.22</td></tr><tr><td>ConvLSTM</td><td>3.8</td><td>0.0306</td><td>0.9819</td><td>54.60</td><td>0.0044</td><td>7.59</td></tr><tr><td>Causal Transformer</td><td>3.8</td><td>0.0314</td><td>0.9809</td><td>54.36</td><td>0.0044</td><td>7.79</td></tr></table>

The state space model achieves the lowest nRMSE, tRMSE, and fRMSE and the highest SSIM and PSNR (Supplementary Table 17). Relative to ConvLSTM and the causal Transformer, respectively, nRMSE decreases by 4.95% and 7.44%, tRMSE by 5.10% and 6.28%, and fRMSE by 4.87% and 7.32%. PSNR increases by 0.49 and 0.73 dB. The temporal-increment improvement is therefore accompanied by more accurate spatial fields and Fourier representations.

The state space formulation in Section 4.1 maintains temporal memory through a recurrent latent state at each Eulerian location. Input-conditioned $B _ { t }$ and $C _ { t }$ control how local features enter the state and how the accumulated history is read out, and the selection gate further modulates the output. This mechanism integrates temporal context, namely flow features accumulated from preceding time steps, into the reconstruction of the current field. The joint reductions in tRMSE and nRMSE are consistent with more efective use of this temporal context, supporting improvements in both temporal fidelity and field reconstruction accuracy.

Supplementary Table 18: Sensitivity of SSM reconstruction to state dimension and feature width on FIT and MHDT at 4 . d<sub>s</sub> denotes state dimension and $d _ { f }$ feature width. Params and GFLOPs count the SSM module alone. All tRMSE values are reported in units of $1 0 ^ { - 3 }$ and rounded to two decimal places; nRMSE is unscaled.
<table><tr><td colspan="4"></td><td colspan="2">FIT</td><td colspan="2">MHDT</td></tr><tr><td> $d _ { s }$ </td><td> $d _ { f }$ </td><td>Params (M)</td><td>GFLOPs</td><td>nRMSE↓</td><td>tRMSE↓</td><td>nRMSE↓</td><td>tRMSE↓</td></tr><tr><td>8</td><td>64</td><td>0.968</td><td>63.468</td><td>0.0295</td><td>4.23</td><td>0.0613</td><td>0.90</td></tr><tr><td>16</td><td>64</td><td>1.034</td><td>67.864</td><td>0.0291</td><td>4.14</td><td>0.0603</td><td>0.88</td></tr><tr><td>32</td><td>64</td><td>1.166</td><td>76.655</td><td>0.0290</td><td>4.11</td><td>0.0599</td><td>0.87</td></tr><tr><td>16</td><td>32</td><td>0.484</td><td>31.789</td><td>0.0305</td><td>4.33</td><td>0.0634</td><td>0.92</td></tr><tr><td>16</td><td>128</td><td>2.329</td><td>152.899</td><td>0.0287</td><td>4.07</td><td>0.0593</td><td>0.86</td></tr></table>

Sensitivity to state dimension and feature width. The SSM has two capacity parameters: feature width $d _ { f }$ (dimension) and state dimension $d _ { s }$ (state), with default $( d _ { f } , d _ { s } ) = ( 6 4 , 1 6 )$ Feature width sets the number of learned channels representing local flow information at each Eulerian location, while state dimension sets the capacity of the recurrent representation through which past features influence the current reconstruction. In a channelwise selective SSM, $d _ { s }$ counts state components per internal channel [68]; the learned transition and input-dependent selection determine the efective memory timescales. Together, $d _ { f }$ and $d _ { s }$ control the richness of the flow fea tures represented and their temporal history, with capacity requirements depending on the spatial complexity and temporal correlations of the flow.

We vary $d _ { s } \in \{ 8 , 1 6 , 3 2 \}$ at $d _ { f } = 6 4$ and $d _ { f } \in \{ 3 2 , 6 4 , 1 2 8 \}$ at $d _ { s } = 1 6$ , giving five configurations on FIT and MHDT. All other architecture, data, loss, and training and evaluation settings are held fixed within each task. Supplementary Table 18 reports the reconstruction errors, parameter counts, and GFLOPs for all five configurations.

Relative to the default $( d _ { f } , d _ { s } ) \ : = \ : ( 6 4 , 1 6 )$ , doubling $d _ { s }$ to 32 increases SSM parameters by approximately 12.8% and GFLOPs by 13.0%. Doubling $d _ { f }$ to 128 increases both to approximately 2.25 times the default, whereas halving $d _ { f }$ to 32 reduces both by approximately 53.2%.

Reducing feature width from $d _ { f } = 6 4$ to 32 increases nRMSE by 4.8% on FIT and 5.1% on MHDT, accompanied by higher tRMSE, while reducing SSM GFLOPs by 53.2%. This joint deterioration is consistent with a narrower feature representation limiting both instantaneous field reconstruction and the representation of evolving flow structures. Halving state dimension from $d _ { s } = 1 6$ to 8 produces smaller nRMSE increases of 1.4% and 1.7%, respectively, with a larger relative increase in tRMSE on both datasets. This pattern is consistent with reduced history-representation capacity afecting the fidelity of temporal increments more strongly than instantaneous field accuracy. Increasing either capacity yields modest improvements. Doubling $d _ { s }$ reduces nRMSE by 0.3% on FIT and 0.7% on MHDT, with larger relative reductions in tRMSE, at 13.0% higher SSM GFLOPs. Doubling $d _ { f }$ improves nRMSE by 1.4% and 1.7%, respectively, and also reduces tRMSE, while increasing SSM GFLOPs to 2.25 times the default. Both FIT and MHDT show a similar accuracy–cost trade-of across the tested configurations.

Together, the modest gains from capacity expansion and the larger penalties from reducing feature width indicate diminishing returns near the default configuration. These trends support $( d _ { f } , d _ { s } ) = ( 6 4 , 1 6 )$ as a practical accuracy–cost compromise for the two tested 4× tasks; $d _ { f } = 3 2$ ofers a lower-cost alternative with an approximately 5% accuracy penalty.

## 3.5 Recovering multiscale spatial structure from coarse-grained data

![](images/08bd2b56bc2a9bb466c29722f1b0924825864ba5f45ba23511d2cc2b1beae2ee.jpg)  
Supplementary Fig. 8: Vorticity-magnitude reconstruction and error visualization for FIT. Comparison of reconstructed vorticity magnitude $\| \boldsymbol { \omega } \| = \| \nabla \times \boldsymbol { \mathbf { u } } \|$ and errors among diferent super-resolution models under boxfiltered coarse-graining. Each column block corresponds to 4 (left) and $8 \times \ ( \mathrm { r i g h t } )$ upscaling. For each case, three orthogonal slices $\left( \mathbf { y } { - } \mathbf { z } , \ \mathbf { z } { - } \mathbf { x } , \ \mathbf { x } { - } \mathbf { y } \right)$ are shown for the reference high-resolution field (Ref.), the low-resolution input (LR), the reconstructed fields from our method, FNO, PhySR, RRDB, EDSR, and RCAN, and the tricubic baseline. The right panels visualize the corresponding error maps (Prediction – Reference) using a symmetric color scale. All samples are from held-out test data; color bars denote vorticity magnitude (left) and reconstruction error (right) under identical dynamic ranges across methods.

![](images/4faf969ab22a9f8be09c21e8a2b2d1233a7958287ef9ec19e48991ce647f70e8.jpg)  
Supplementary Fig. 9: Reconstruction of the magnetic field curl magnitude for MHDT. Comparison of the reconstructed current magnitude $\| \boldsymbol { j } \| = \| \nabla \times \pmb { b } \|$ field and corresponding voxelwise errors under the boxfiltered coarse-graining. Columns correspond to 4 (left) and 8 (right) super-resolution. Each group shows three orthogonal slices $\left( \mathbf { y } { - } \mathbf { z } , \ \mathbf { z } { - } \mathbf { x } , \ \mathbf { x } { - } \mathbf { y } \right)$ for the reference high-resolution field (Ref.), the low-resolution input (LR), and reconstructions by our method and baselines (FNO, PhySR, RRDB, EDSR, RCAN, and tricubic interpolation). The right panels visualize error fields (Prediction – Reference) with symmetric color scaling. Color bars denote j (left) and reconstruction error (right).

![](images/5a6215f55dc4a8e9b09765ac943c2fab4f24fa9c5f2910bb93d32afe298a44fa.jpg)  
Supplementary Fig. 10: Velocity divergence reconstruction and error visualization for HBDT. Comparison of reconstructed and reference velocity divergence fields u under the Favre-filtered coarse-graining for 8 (left) and 16 (right) super-resolution reconstruction. Each column block presents three orthogonal slices (y–z, z–x, x–y) of the reference high-resolution field (Ref.), low-resolution input (LR), and reconstructions from our method, FNO, PhySR, RRDB, EDSR, RCAN, and tricubic interpolation. The right panels visualize error fields (Prediction – Reference) with symmetric color scaling. Color bars indicate   u (left) and corresponding reconstruction error (right).

![](images/905c59fb3894fc6977f03a3d0e139cb51648752a1bd83b3570943e3ea937cc52.jpg)

![](images/1eb8a3b1841a644528224e0e600742a070e647bc55652bd2c59c682f98ff644a.jpg)  
Supplementary Fig. 11: Q–R joint probability distributions. Isocontour plots of joint PDFs of normalized $Q ^ { * }$ and $R ^ { * }$ for FIT at 4 (top) and 8 (bottom) super-resolution reconstruction. Columns show EDSR, DifSR, RCAN, SwinIR, and tricubic interpolation, from left to right. Black contours denote the reference; colored contours (gray for tricubic interpolation) denote the reconstructed fields. The magenta dashed curve marks $2 7 ( R ^ { * } ) ^ { 2 } + 4 ( Q ^ { * } ) ^ { 3 } = 0$ Evaluation details are given in Supplementary Note 3.5.  
Supplementary Fig. 12: Component contributions to Q–R joint distributions on FIT at $8 \times$ with spectral-cutof coarse-graining. Left to right: the full model, removal of the local convolutional operator ${ \mathcal { O } } _ { c } ,$ removal of the global spectral operator ${ \mathcal { O } } _ { s } ,$ and removal of the physical loss $\mathcal { L } _ { \mathrm { p h y } }$ , with the remaining training settings held fixed. Black contours show the reference and colored contours the corresponding reconstruction; gray rectangles highlight regions of contour displacement. The dashed curve is $2 7 ( R ^ { * } ) ^ { 2 } + 4 ( Q ^ { * } ) ^ { 3 } = 0$ . Evaluation details are given in Supplementary Note 3.5.

Evaluation of Q–R statistics. For all Q–R comparisons, $Q ^ { * } ~ = ~ Q / \langle S _ { i j } S _ { i j } \rangle$ <sub>DNS</sub> and $R ^ { * } =$ $R / { \langle S _ { i j } S _ { i j } \rangle } _ { \mathrm { D N S } } ^ { 3 / 2 }$ , where the average is taken over the evaluated DNS reference samples and the same scale is used for every reconstruction. The statistics are computed directly from the DNS and reconstructed velocity fields on the HR grid without additional spatial filtering; the coarsegraining specified for each experiment defines its LR inputs. Joint PDFs are estimated using unsmoothed two-dimensional histograms on common bin edges. Contours show $p / p _ { \mathrm { D N S , m a x } } =$ $0 . 0 0 3 , 0 . 0 1 , 0 . 0 3 , 0 . 1 , 0 . 3$ , so the DNS peak sets common absolute density thresholds for all meth ods. The comparison assesses gradient configurations represented by the displayed contours; rarer events require additional tail sampling.

Physical interpretation and component contributions in the Q–R plane. For incompressible flow, let $A _ { i j } = \partial u _ { i } / \partial x _ { j } , S = ( A + A ^ { \top } ) / 2$ , and $\Omega = ( A - A ^ { \top } ) / 2$ . These symmetric and antisymmetric parts describe local deformation and rotation, respectively. The invariants can be written as [46, 47]

$$
\begin{array} { r l } & { { \cal Q } = - \frac { 1 } { 2 } \operatorname { t r } ( { \cal A } ^ { 2 } ) = \frac { 1 } { 2 } \left( \| \pmb { \Omega } \| _ { F } ^ { 2 } - \| \pmb { S } \| _ { F } ^ { 2 } \right) , } \\ & { { \cal R } = - \frac { 1 } { 3 } \operatorname { t r } ( { \cal A } ^ { 3 } ) = - \frac { 1 } { 3 } \operatorname { t r } ( { \pmb { S } } ^ { 3 } ) - \frac { 1 } { 4 } \omega ^ { \top } { \pmb { S } } \omega , } \end{array}\tag{S28}
$$

where ${ \boldsymbol { \omega } } = \nabla \times { \boldsymbol { u } }$ and $\| \cdot \| _ { F } ^ { 2 }$ is the sum of squared tensor entries. Thus $Q$ compares rotation and strain intensity. The first contribution to R depends on the principal stretching and compression rates; the second depends on the strength and alignment of vorticity relative to strain. Positive $\omega ^ { \mathsf { T } } S \omega$ describes vortex stretching and contributes negatively to R. The sign of R therefore reflects the balance of both terms.

The joint PDF measures the frequency of local gradient configurations. Its teardrop asymmetry is associated with the coupled amplification of strain and vorticity, whose evolution is regulated by pressure and viscous efects [47, 48]. The positive-Q region represents rotation-dominated motion; the lower-right extension contains strong strain-dominated states. For incompressible flow, the discriminant $2 7 R ^ { 2 } + 4 Q ^ { 3 } = 0$ separates diferent velocity-gradient eigenvalue structures [46]. Its positive-R branch includes axisymmetric compression: extension in two directions and compression in the third, flattening a fluid element while conserving volume. The distribution thus captures preferred combinations of local deformation and rotation, including motions relevant to dissipation and vortex stretching.

In the FIT 8× box-filtered comparison (Fig. 3c), RRDB’s displaced positive- $Q ^ { * }$ contours indicate altered combinations of rotation, strain, and their coupling, while its shortened lower-right extension indicates deficient recovery of the strongly strain-dominated states represented by the reference tail. Spatial diferentiation emphasizes fine-scale velocity variations, and $Q$ and R de pend quadratically and cubically on gradients. Their joint statistics also depend on correlations among gradient components: an energy spectrum constrains energy at each wavenumber, while local gradient products additionally depend on phase and component relationships. Consequently, spectral agreement alone leaves room for Q–R distortions. RRDB builds spatial context through successive residual dense convolutions [37]; the observed discrepancies are consistent with incomplete recovery of gradient amplitudes and their relationships under stronger coarse-graining. Q–R contours reveal the combined statistical efect; separating the contributions would require strain and vorticity-alignment statistics.

Supplementary Fig. 12 compares the full model with the three removals on FIT at $8 \times$ with spectral-cutof coarse-graining, keeping the remaining training settings fixed. This experiment examines the mechanisms within the proposed model and complements the 8× box-filtered baseline comparison in Fig. 3c. The highlighted regions reveal distinct changes in the Q–R distribution. Removing the local operator $\mathcal { O } _ { c }$ or the physical loss $\mathcal { L } _ { \mathrm { p h y } }$ shifts the right flank of the rotationdominated contours $( Q ^ { * } > 0 )$ towards larger $R ^ { * }$ . Removing the spectral operator $\mathcal { O } _ { s }$ contracts the negative- $R ^ { * }$ side of the distribution across both rotation- and strain-dominated regions. All three removals also degrade the strain-dominated lower-right branch $( Q ^ { * } < 0 , R ^ { * } > 0 )$ , with several intermediate contours retracting relative to the reference and showing larger departures than in the full model. The outermost branch remains extended after removing $\mathcal { O } _ { c } .$ , so the loss of extension depends on the contour level. Together, these changes indicate that the components contribute to recovering both the characteristic strain-dominated extension and the distribution of strain– vorticity configurations, with diferent efects across the Q–R plane. The full model retains residual deviations in the outer contours.

![](images/46a14394df662e708f3bc4c1eabcb92e22d3c34b2ff3d433af6cbe6c00bd1eb9.jpg)  
Supplementary Fig. 13: Turbulent energy spectra under box/Favre coarse-graining. Panels show FIT (4 ), MHDT (4 ), and HBDT $( 8 \times )$ from left to right. Horizontal and vertical axes denote wavenumber k and energy spectrum E(k), respectively. Reconstructions from our method, FNO, PhySR, RRDB, SwinIR, DifSR, EDSR, and RCAN are compared with the reference (Ref.). For MHDT, the spectrum is the sum of kinetic and magnetic contributions. Insets magnify the high-wavenumber range to highlight small-scale recovery. The black dashed line indicates a slope of 5/3.

The local operator $\mathcal { O } _ { c }$ refines neighboring velocity diferences, which determine both strain and rotation. The spectral operator $\mathcal { O } _ { s }$ couples spatially separated features [44], supplying context for the recovered local field. Their contributions are consistent with the separate FIT 4× spectral-cutof velocity/vorticity ablation in Supplementary Note 3.3. The physical loss encourages local volume conservation through its divergence penalty and couples temporal changes, advection, pressure gradients, and viscosity through its scale-selective momentum term (Eqs. (10) and (13)). These constraints regularize relationships among recovered derivatives. The contour changes therefore support the roles of local refinement, global context, and physical regularization in recovering strain–rotation statistics. Each acts on the velocity field and can afect both strain and rotation; the component removals establish their contributions within the proposed model under the tested spectral-cutof setting.

Spectral ranges and plotting conventions. Energy spectra are plotted against physical wavenumber k. The FIT and MHDT reference simulations use phase-shift dealiasing with $2 { \sqrt { 2 } } / 3$ spectral truncation, giving a maximum retained wavenumber of approximately 482. For HBDT, the $1 0 2 4 ^ { 3 }$ grid on $[ 0 , 2 \pi ) ^ { 3 }$ gives an axial Nyquist wavenumber of 512. We display the HBDT spectra up to approximately $k = 2 5 6$ , as the reconstructed spectra develop nonphysical distortions at higher wavenumbers. The same display range is used for the reference and all compared methods.

![](images/4777890723f78b475c6e023338cfd76d9ec2cd37bcd571ce9f22d39157f8405a.jpg)

![](images/f99d2d5bb45633d96071133526ea7c6594d519778a79358133b295d81215a42a.jpg)

![](images/5d1ed7c3de17e2b19d8b55c7c599a977c4575bce867d2a141db48dfe705ceb66.jpg)  
Supplementary Fig. 14: Quantitative distributional similarity across turbulent regimes. Shown are the Jensen–Shannon (JS) divergences between the reconstructed and reference PDFs of three canonical turbulent diagnostics: top, vorticity magnitude $\| \omega \|$ in FIT; middle, current magnitude j in MHDT; bottom, velocity divergence u in HBDT; these correspond to Fig. 3b.

## 3.6 Atmospheric downscaling on ERA5

Cross-level structure in atmospheric winds is assessed via bulk vertical wind shear. In atmospheric dynamics, bulk shear, commonly expressed as the wind-vector diference between two pressure levels, is associated with dynamically active features such as jet structures and frontal zones and summarizes how horizontal winds vary with height [51]. Based on this, two shear-based metrics are introduced to evaluate both amplitude accuracy and the spatial organization of strong cross-level shear in the downscaled fields. Let $V = ( u , v )$ denote the horizontal wind field. The magnitude of the wind-vector diference between two pressure levels $p _ { a }$ and $p _ { b }$ is defined as

$$
S : = \| V ( \cdot , p _ { a } ) - V ( \cdot , p _ { b } ) \| _ { 2 } ,\tag{S29}
$$

where the norm is taken over the wind-vector components at each horizontal grid point. Here, S has units of $\mathrm { m } / \mathrm { s }$ and serves as a bulk-diference proxy of shear magnitude. Following common practice in diagnosing the vertical distribution of environmental shear [52], the wind-vector diference between 850 and 500 hPa is used as a representative shallow-to-midtropospheric shear measure; $\hat { S }$ is computed analogously from $\hat { V } .$ . The resulting vertical metrics are denoted as vRMSE and vCoh. The amplitude error of cross-level shear is quantified by

$$
\mathrm { \ v R M S E } : = \sqrt { \frac { 1 } { | \mathscr { D } | } \sum _ { i \in \mathscr { D } } ( \hat { S } _ { i } - S _ { i } ) ^ { 2 } } ,\tag{S30}
$$

where D denotes valid horizontal grid points. To emphasize the spatial organization of dynamically active, high-shear regions, we define a vertical coherence score as the Pearson correlation restricted to strong-shear points:

$$
\mathrm { v C o h } : = \mathrm { C o r r } _ { \Omega _ { q } } ( \hat { S } , S ) , \qquad \Omega _ { q } : = \{ i \in \mathcal { D } : S _ { i } > Q _ { q } ( S ) \} ,\tag{S31}
$$

where $Q _ { q } ( S )$ is the q-th quantile of S over D and $\mathrm { C o r r } _ { \Omega _ { q } }$ denotes the Pearson correlation computed on $\Omega _ { q } .$ In practice, $q = 0 . 9$ is used, focusing on the top 10% strongest-shear points (defined by the reference), to assess whether strong-shear structures are reconstructed in the correct locations. Metrics are computed per time step and averaged over the evaluation set.

Supplementary Table 19: Cross-level wind diagnostics for ERA5 downscaling. Performance is evaluated using vCoh and vRMSE from Supplementary Eqs. (S29)–(S31). Higher vCoh and lower vRMSE indicate closer agreement in the spatial organization and amplitude, respectively, of the 850–500 hPa bulk-shear field. The best result is in bold.
<table><tr><td>Factors</td><td>Metrics</td><td>Ours</td><td>SwinIR</td><td>FNO</td><td>Bicubic</td></tr><tr><td rowspan="2"> $4 \times$ </td><td>vCoh</td><td>0.9925</td><td>0.9906</td><td>0.9871</td><td>0.9743</td></tr><tr><td>vRMSE</td><td>0.4672</td><td>0.5167</td><td>0.5925</td><td>0.8482</td></tr><tr><td rowspan="2">8×</td><td>vCoh</td><td>0.9533</td><td>0.9442</td><td>0.9259</td><td>0.8745</td></tr><tr><td>vRMSE</td><td>1.1340</td><td>1.2582</td><td>1.4174</td><td>1.8646</td></tr></table>

## 3.7 Training-seed uncertainty

Training-seed uncertainty is quantified using independent seeds 42, 43, and 44 (n = 3). The analysis covers all eight learned methods on FIT at 4× upscaling with box-filtered inputs and Ours, FNO, and SwinIR on ERA5 at both 4× and 8×. The predefined data partitions, model-specific hyperparameters, training budgets, and evaluation protocols are held fixed across seeds, so the resulting variation reflects stochastic model training. Supplementary Tables 20 and 21 report the mean and sample standard deviation across the three runs. A standard deviation displayed as 0.0000 is below the four-decimal reporting precision.

The proposed method achieves the best mean for every reported metric in all three settings. On FIT, its nRMSE is $0 . 0 2 9 2 { \scriptstyle \pm 0 . 0 0 0 1 }$ , compared with $0 . 0 4 1 0 \pm 0 .$ <sub>0006</sub> for RCAN, the closest baseline for this metric; its tRMSE is 0.0042 <sub>0.0000</sub>, compared with 0.0053 <sub>0.0000</sub> for SwinIR. On ERA5, its MSE is 0.0762 <sub>±0.0011</sub> at 4× and 0.4451 <sub>±0.0007</sub> at 8×, compared with 0.0947 <sub>±0.0005</sub> and 0.5333 <sub>±0.0006</sub>, respectively, for SwinIR. The sample standard deviations are small relative to these between-method diferences, demonstrating stable performance under training randomness.

Supplementary Table 20: Training-seed uncertainty on FIT at 4 with box-filtered inputs. Values are mean sample standard deviation over independent training seeds 42, 43, and 44 $( n = 3 )$ . Arrows indicate the preferred direction. A standard deviation displayed as 0.0000 is below the reporting precision. The best mean is shown in bold.
<table><tr><td>Model</td><td>SSIM ↑</td><td>nRMSE↓</td><td>PSNR ↑</td><td>tRMSE↓</td><td>fRMSE↓</td></tr><tr><td>Ours</td><td>0.9835 ±0.0001</td><td> $\mathbf { 0 . 0 2 9 2 _ { \pm 0 . 0 0 0 1 } }$ </td><td> $\mathbf { 5 5 . 0 7 2 7 \pm } \mathbf { 0 . 0 1 7 8 }$ </td><td> $\mathbf { 0 . 0 0 4 2 \scriptstyle \pm 0 . 0 0 0 0 }$ </td><td> $\mathbf { 7 . 2 5 2 8 \bot 0 . 0 2 2 5 }$ </td></tr><tr><td>FNO</td><td>0.9163 ±0.0040</td><td> $0 . 0 7 0 4 \pm 0 . 0 0 2 2$ </td><td> $4 6 . 9 6 9 4 \pm \mathrm { 0 . 3 0 8 9 }$ </td><td> $0 . 0 0 7 2 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td><td>16.7646 ±0.4352</td></tr><tr><td>PhySR</td><td>0.9585 ±0.0001</td><td> $0 . 0 4 7 7 { \scriptstyle \pm 0 . 0 0 0 1 }$ </td><td> $5 0 . 6 3 9 6 \pm \substack { 0 . 0 1 7 5 }$ </td><td> $0 . 0 0 6 2 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td><td>11.6351 ±0.0106</td></tr><tr><td>RCAN</td><td>0.9674 ±0.0007</td><td> $0 . 0 4 1 0 { \scriptstyle \pm 0 . 0 0 0 6 }$ </td><td> $5 2 . 0 4 7 7 { \scriptstyle \pm 0 . 1 2 2 0 }$ </td><td> $0 . 0 0 6 4 \pm 0 . 0 0 0 1$ </td><td>10.0879 ±0.1354</td></tr><tr><td>EDSR</td><td>0.9657 ±0.0002</td><td> $0 . 0 4 2 2 { \scriptstyle \pm 0 . 0 0 0 1 }$ </td><td> $5 1 . 7 9 5 0 { \scriptstyle \pm 0 . 0 2 2 1 }$ </td><td> $0 . 0 0 6 5 { \scriptstyle \pm 0 . 0 0 0 1 }$ </td><td>10.3804 ±0.0256</td></tr><tr><td>RRDB</td><td>0.9641 ±0.0028</td><td> $0 . 0 4 3 4 \pm 0 . 0 0 2 2$ </td><td> $5 1 . 4 8 1 5 \pm 0 . 5 1 5 2$ </td><td> $0 . 0 0 6 8 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td><td>10.6929 ±0.5478</td></tr><tr><td>SwinIR</td><td> $0 . 9 6 8 5 \pm \mathrm { 0 . 0 0 0 4 }$ </td><td> $0 . 0 4 2 4 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td><td> $5 1 . 3 8 5 4 \pm 0 . 0 7 1 3$ </td><td> $0 . 0 0 5 3 { \scriptstyle \pm 0 . 0 0 0 0 }$ </td><td> $1 0 . 2 9 5 3 { \scriptstyle \pm 0 . 0 6 7 1 }$ </td></tr><tr><td>DiffSR</td><td> $0 . 9 6 5 2 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td><td> $0 . 0 4 2 1 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td><td> $5 1 . 6 6 9 3 { \scriptstyle \pm 0 . 0 6 8 7 }$ </td><td> $0 . 0 0 9 6 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td><td> $1 0 . 2 9 1 0 { \scriptstyle \pm 0 . 0 7 7 4 }$ </td></tr></table>

Supplementary Table 21: Training-seed uncertainty on ERA5. Values are mean sample standard deviation over independent training seeds 42, 43, and 44 (n = 3). The best mean within each factor is shown in bold.
<table><tr><td>Factor</td><td>Model</td><td>MAE↓</td><td>MSE↓</td><td>tRMSE↓</td><td>vCoh↑</td><td>vRMSE↓</td></tr><tr><td rowspan="3"> $4 \times$ </td><td>Ours</td><td>0.1202 ±0.0007</td><td>0.0762 ±0.0011</td><td>0.3146 ±0.0015</td><td>0.9925 ±0.0001</td><td>0.4625 ±0.0039</td></tr><tr><td>FNO</td><td> $0 . 1 6 9 2 { \scriptstyle \pm 0 . 0 0 1 2 }$ </td><td> $0 . 1 2 5 5 { \scriptstyle \pm 0 . 0 0 1 1 }$ </td><td> $0 . 4 0 3 8 { \scriptstyle \pm 0 . 0 0 1 5 }$ </td><td> $0 . 9 8 7 0 { \scriptstyle \pm 0 . 0 0 0 1 }$ </td><td>0.5942 ±0.0012</td></tr><tr><td>SwinIR</td><td> $0 . 1 3 3 6 \pm 0 . 0 0 0 4$ </td><td>0.0947 ±0.0005</td><td>0.3433 ±0.0009</td><td>0.9907 ±0.0001</td><td>0.5162 ±0.0013</td></tr><tr><td rowspan="3">8×</td><td>Ours</td><td>0.3491 ±0.0002</td><td>0.4451 ±0.0007</td><td>0.8041 ±0.0007</td><td>0.9529 ±0.0012</td><td>1.1363 ±0.0031</td></tr><tr><td>FNO</td><td>0.4404 ±0.0021</td><td>0.6794 ±0.0029</td><td>1.0002 ±0.0011</td><td>0.9250 ±0.0006</td><td>1.4211 ±0.0026</td></tr><tr><td>SwinIR</td><td> $0 . 3 7 8 2 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td><td> $0 . 5 3 3 3 { \scriptstyle \pm 0 . 0 0 0 6 }$ </td><td> $0 . 8 8 1 7 { \scriptstyle \pm 0 . 0 0 0 8 }$ </td><td>0.9442 ±0.0004</td><td>1.2576 ±0.0002</td></tr></table>

## 3.8 Generalizing to unknown coarse graining

The DNS simulation and LES–DNS pairing for the decaying-turbulence experiment are described in Supplementary Notes 2.7 and 2.8.

Although all three expert models can in principle be included in the MoE, we find that the box and spectral experts already exhibit strong generalization performance (Supplementary Table 22), and thus adopt a two-expert MoE as our main configuration. To understand this behavior, we analyze reconstruction error fields under tricubic downsampling and compute Pearson correlations between experts (Fig. 8f). The box and spectral experts show near-zero correlation, indicating largely independent and therefore complementary error modes. In contrast, the Gaussian expert exhibits a moderate positive correlation (approximately 0.5) with both the box and spectral experts, suggesting partially redundant error characteristics. This observation is consistent with the distinct filtering characteristics of these coarse-graining strategies [27] and suggests that expert diversity, rather than simply increasing the number of experts, is key to robust generalization.

Supplementary Table 22: Full quantitative generalization results on FIT at 4 upscaling. We evaluate three unseen test-time coarse-graining settings: tricubic downsampling, Lanczos filtering, and uniform subsampling with 5% additive Gaussian noise. We compare three single-expert models (trained with box, Gaussian, or spectralcutof coarse-graining) and two MoE variants (2 experts: box+spectral; 3 experts: box+Gaussian+spectral). Higher SSIM/PSNR and lower nRMSE/fRMSE indicate better performance; best results are bold.
<table><tr><td>Coarse-graining</td><td>Expert</td><td>SSIM↑</td><td>PSNR↑</td><td>nRMSE↓</td><td>fRMSE↓</td></tr><tr><td rowspan="5">Tricubic</td><td>Spectral</td><td>0.8679</td><td>43.38</td><td>0.1048</td><td>25.27</td></tr><tr><td>Box</td><td>0.9077</td><td>45.02</td><td>0.0879</td><td>21.44</td></tr><tr><td>Gaussian</td><td>0.8897</td><td>43.82</td><td>0.1001</td><td>24.08</td></tr><tr><td>MoE (2 experts)</td><td>0.9406</td><td>47.19</td><td>0.0673</td><td>16.45</td></tr><tr><td>MoE (3 experts)</td><td>0.9333</td><td>46.48</td><td>0.0731</td><td>17.72</td></tr><tr><td rowspan="5">Lanczos</td><td>Spectral</td><td>0.8581</td><td>42.78</td><td>0.1130</td><td>27.04</td></tr><tr><td>Box</td><td>0.8926</td><td>44.32</td><td>0.0978</td><td>23.60</td></tr><tr><td>Gaussian</td><td>0.8735</td><td>43.11</td><td>0.1106</td><td>26.39</td></tr><tr><td>MoE (2 experts)</td><td>0.9300</td><td>46.37</td><td>0.0757</td><td>18.20</td></tr><tr><td>MoE (3 experts)</td><td>0.9212</td><td>45.65</td><td>0.0822</td><td>19.65</td></tr><tr><td rowspan="5">Uniform (5% noise)</td><td>Spectral</td><td>0.8274</td><td>43.01</td><td>0.1093</td><td>25.48</td></tr><tr><td>Box</td><td>0.6774</td><td>39.25</td><td>0.1723</td><td>39.98</td></tr><tr><td>Gaussian</td><td>0.7757</td><td>42.39</td><td>0.1208</td><td>27.68</td></tr><tr><td>MoE (2 experts)</td><td>0.8460</td><td>43.91</td><td>0.0991</td><td>22.85</td></tr><tr><td>MoE (3 experts)</td><td>0.8420</td><td>44.32</td><td>0.0972</td><td>22.27</td></tr></table>