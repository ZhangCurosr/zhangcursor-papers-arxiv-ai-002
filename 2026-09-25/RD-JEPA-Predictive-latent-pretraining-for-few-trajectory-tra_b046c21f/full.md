# RD-JEPA: Predictive latent pretraining for few-trajectory transfer across reaction–diffusion equations

Chenhao Si<sup>1</sup> and Ming Yan<sup>1,\*</sup>

<sup>1</sup>School of Data Science, The Chinese University of Hong Kong, Shenzhen, Shenzhen, China yanming@cuhk.edu.cn

## ABSTRACT

Learning surrogates for time-dependent partial differential equations often requires a new simulation corpus when the governing operator changes. We introduce RD-JEPA, a joint-embedding predictive architecture for self-supervised pretraining on reaction–diffusion trajectories. A single model is pretrained on five parameterized systems and then adapted to three held-out systems whose reaction operators and trajectories are excluded from pretraining. Using one, five, or ten complete trajectories from a held-out system, RD-JEPA achieves lower mean relative discrete ℓ<sup>2</sup> field error and mean absolute spatial first-difference error than five supervised surrogate baselines, an independently trained control that removes the trajectory-dependent predictive latent pathway, and an architecture-matched model trained from scratch. Within the evaluated equations, output resolution, forecast horizons, and choices of adaptation trajectories, the results indicate that prediction of future-state representations can support data-efficient adaptation across related reaction–diffusion systems.

Reaction–diffusion (RD) systems describe how local nonlinear reaction kinetics interact with spatial diffusion to generate organized structures, including Turing patterns, traveling waves, spirals, and self-replicating spots<sup>1</sup>. They provide mathematical models of pigmentation and skin patterning<sup>2,</sup> <sup>3</sup>, limb and digit morphogenesis<sup>4,</sup> <sup>5</sup>, engineered multicellular patterning<sup>6,</sup> <sup>7</sup>, and related non-biological phenomena<sup>8</sup>. However, using these models in predictive and repeated-query settings requires repeated numerical integration of stiff, parameter-dependent partial differential equations on fine spatial grids. The resulting computational cost can become substantial in parameter studies, inverse problems, and uncertainty quantification, particularly when the nonlinear reaction term changes and only a small number of trajectories can be generated for the new system.

Learning-based surrogate models can reduce this cost by replacing repeated numerical integration with a trained approximation of the underlying dynamics. Physics-informed neural networks incorporate the governing equations through residual-based objectives<sup>9,</sup> <sup>10</sup>, whereas neural operators learn mappings between function spaces over parameterized families of equations<sup>11–14</sup>. Graph-based, convolutional, and transformer architectures provide complementary approaches to forecasting spatially distributed physical fields<sup>15–22</sup>, and benchmark datasets facilitate comparisons across equations, coefficients, and initial conditions<sup>23,</sup> <sup>24</sup>. Nevertheless, many existing surrogates are trained on a single governing equation or on parameter variations within an equation family with a fixed functional form. Changing the nonlinear reaction term may therefore require a new simulation corpus and equation-specific retraining. Few-shot and meta-learning methods can reduce the amount of target-system data required for adaptation<sup>25,</sup> <sup>26</sup>, but how information learned from one reaction operator can be reused for another remains less well understood.

Cross-system pretraining provides a possible route beyond equation-specific training. Multiple physics pretraining and Poseidon learn autoregressive surrogates from several physical systems<sup>27,</sup> <sup>28</sup>, while multi-operator and in-context methods seek to represent multiple governing equations within a shared model<sup>29–32</sup>. In parallel, self-supervised approaches to PDE learning have used masked reconstruction, input-side proxy tasks, and physics-informed contrastive objectives to learn representations that can be reused in downstream tasks<sup>33–35</sup>. Recent studies have also begun to investigate joint-embedding predictive architecture (JEPA) representations for physical-parameter inference and physics-informed surrogate pretraining<sup>36,</sup> <sup>37</sup>. However, evidence remains limited for the use of JEPA-style pretraining in full-field PDE forecasting, particularly when the nonlinear reaction operator of the target system is absent from pretraining. We therefore ask whether predictive latent pretraining can support few-trajectory full-field forecasting when both the target reaction term and all trajectories generated from it are withheld during pretraining.

The JEPA framework provides a natural formulation for this question because it predicts representations of target states from an encoded context rather than directly reconstructing them in the physical field space<sup>38</sup>. I-JEPA combines an online context encoder and predictor with a stop-gradient target encoder updated by an exponential moving average<sup>39</sup>, and V-JEPA extends this principle to video<sup>40</sup>. We use reaction–diffusion dynamics as a controlled PDE setting in which the source and target systems share the broad structure of spatial diffusion coupled with local nonlinear kinetics, while differing in their reaction terms. Our hypothesis is that predicting future states in representation space can capture dynamical information shared across related systems and remain useful after the reaction operator changes. RD-JEPA implements this hypothesis through complementary diffusion-inspired and reaction-inspired predictor pathways, which provide inductive biases associated with neighborhood-mediated spatial exchange and pointwise nonlinear kinetics. The model requires no equation identifier, coefficient vector, or symbolic description of the governing equation.

We pretrained a single RD-JEPA model on five reaction–diffusion systems and evaluated the learned representation through a two-stage transfer design. We first examined whether pretraining reduces the amount of equation-specific data required for adaptation on the five source systems. We then adapted the same pretrained model to three held-out systems whose reaction terms and trajectories were completely excluded from pretraining. Comparisons with equation-specifi supervised surrogates, an independently trained control without the trajectory-dependent predictive latent pathway, and the same encoder–predictor–decoder architecture trained from random initialization were used to distinguish the contribution of predictive pretraining from that of the downstream architecture or a particular comparator. These experiments address two related questions: whether JEPA-style future-state prediction serves as an effective pretraining objective for PDE dynamics, and whether the resulting representation remains useful when the nonlinear reaction operator changes. In this way, RD-JEPA is evaluated as a reusable representation-learning approach rather than only as an equation-specific forecasting architecture.

## Results

## Predictive latent pretraining targets reusable dynamics

We investigated whether predictive pretraining across multiple reaction operators could yield a dynamical representation that remains useful even when only a small number of trajectories are available from a target system. We considered multi-component reaction–diffusion fields governed by

$$
\partial _ { t } \mathbf { u } ( \mathbf { x } , t ) = \mathbf { D } \nabla ^ { 2 } \mathbf { u } ( \mathbf { x } , t ) + \mathbf { R } ( \mathbf { u } ( \mathbf { x } , t ) ; \pmb { \theta } ) ,\tag{1}
$$

where D is the diffusion matrix and R is the nonlinear reaction operator with parameters θ . The principal transfer experiment withheld the complete target systems: neither their reaction operators nor any trajectories generated from them were used during pretraining. Thus, the target-system experiments test transfer beyond variations in coefficients or initial conditions within a fixed equation family.

RD-JEPA separates self-supervised representation learning from supervised full-field forecasting (Fig. 1). During pretraining, an online encoder maps four observed fields to a latent representation of the trajectory context, while an exponential-moving average target encoder represents a future field. A lead-time-conditioned predictor estimates the future target representation without reconstructing the corresponding field. The predictor contains two structured latent pathways: a diffusion-inspired pathway based on nearest-neighbor interactions and a reaction-inspired pathway based on pointwise nonlinear updates. These pathways provide architectural inductive biases rather than numerical discretizations of the physical diffusion and reaction operators. During downstream adaptation, the target encoder is removed, the pretrained online encoder is frozen, and the predictor is fine-tuned jointly with a newly initialized full-field decoder.

A single RD-JEPA checkpoint was pretrained on Gray–Scott, FitzHugh–Nagumo, Brusselator, complex Ginzburg–Landau, and Schnakenberg trajectories. We first evaluated reuse on these five source systems using $K \in \{ 5 , 1 0 , 2 0 \}$ complete adaptation trajectories. We then adapted the same checkpoint to Lambda–Omega, Barkley, and Oregonator using $K \in \{ 1 , 5 , 1 0 \}$ targetsystem trajectories. All downstream models received four consecutive states and directly predicted the five future states at horizons $h = 1 , \ldots , 5 ,$ , where h is measured in stored-frame intervals rather than through autoregressive rollout. Each trained model was evaluated on the same 300 held-out test trajectories for the corresponding equation. Support-set construction, pre processing, comparison-model configurations, optimization settings, and downstream objectives are detailed in Supplementary Note 1–3 and Supplementary Table S1–S5.

For each equation and adaptation budget, we evaluated three prespecified adaptation-set selections. Within each selection, all models received the same adaptation trajectories and were evaluated on the same test trajectories. Across the three selections, the predictive-pretraining checkpoint, model-training randomness, and test data were fixed; only the selected adaptation trajectories changed. The reported error bars therefore quantify sensitivity to adaptation-set selection rather than variability across pretraining runs, optimization seeds, or test trajectories. For the source systems, the 300-trajectory test set contained 100 in-distribution trajectories, 100 coefficient-out-of-distribution trajectories, and 100 initial-condition-out-of-distribution trajectories.

We report the relative discrete $\ell ^ { 2 }$ field error and the mean absolute spatial first-difference error. The first measures the global discrepancy between the predicted and reference fields, whereas the second measures mismatch in their local grid-scale spatial variation. Values shown in the main figures are the mean and sample standard deviation across the three adaptation-set-level means. Because no formal hypothesis tests were performed, the comparisons below refer to the ordering of the reported numerical means. Split-resolved source-system summaries are reported in Supplementary Table S12 and the corresponding run-level values are provided in Supplementary Data 1.

![](images/6b7ce98df7bd4f35d3499463c8f253f5e587c93b4d4e7e84cb20f8e19d0ef77f.jpg)  
Figure 1. a, During pretraining, a context encoder represents four observed fields and an exponential-moving-average target encoder represents a future field. A lead-time-conditioned predictor estimates the target representation from context; the loss is evaluated in latent space without reconstructing the future field. b, For downstream forecasting, the source-pretrained encoder is frozen and the predictor is adapted jointly with a dense decoder using K target-system trajectories. The same checkpoint is evaluated on source equations and on governing equations excluded from pretraining, and it produces direct forecasts at $h = 1 , \ldots , 5 .$

## Predictive-latent pretraining improves source-system forecasting with limited adaptation data

We first evaluated whether the pretrained representation improved forecasting on the five systems that contributed trajectories to pretraining. For each source equation and adaptation budget, RD-JEPA was compared with an equation-specific Fourier neural operator (FNO) trained from random initialization and an independently trained no-predictive-latent control. The control removes the JEPA encoder–predictor pathway but retains the full-field decoder, the full-resolution pathway from the latest observed state, and forecast-horizon conditioning. Because all five equations were represented during pretraining, this experiment evaluates data-efficient reuse on source systems rather than transfer to a previously unseen reaction operator.

At the smallest adaptation budget, $K = 5$ , and the longest horizon, h = 5, RD-JEPA had the smallest reported mean relative discrete $\ell ^ { 2 }$ field error on each of the five equations and on their equal-weight average (Fig. 2c). The numerical ordering relative to both FNO and the no-predictive-latent control indicates that the source-pretrained representation remained useful when only five equation-specific trajectories were available for adaptation.

Across all adaptation budgets $K \in \{ 5 , 1 0 , 2 0 \}$ and forecast horizons $h \in \{ 1 , . . . , 5 \}$ , the mean forecasting error increased with the horizon for all three models, while the RD-JEPA mean decreased as additional adaptation trajectories were provided $( \mathrm { F i g . ~ } 3 \mathrm { a , b } )$ . RD-JEPA had the smallest mean relative discrete $\ell ^ { 2 }$ error across the five source systems in every evaluated budget–horizon configuration. Its paired percentage reduction relative to FNO was positive throughout the corresponding K-by-h grid (Fig. 3c). These results show that the observed source-system advantage was not limited to a single adaptation budget or forecast horizon.

Complete equation-, adaptation-budget-, and forecast-horizon-resolved results are reported in Supplementary Note 4. Relative discrete $\ell ^ { 2 }$ field errors are given in Supplementary Tables S6–S8, mean absolute spatial first-difference errors in Supplementary Tables S9–S11, and split-resolved results in Supplementary Table S12. The three run-level values underlying these summaries are provided in Supplementary Data 1.

The representative forecasts in Fig. 2 illustrate how the aggregate error measurements appear in the predicted fields. For FitzHugh–Nagumo, the largest pointwise errors occur near rapidly evolving interfaces. The additional Gray–Scott, complex Ginzburg–Landau, Schnakenberg, and Brusselator examples show the corresponding spatial structures at $K = 5$ and $h = 5$ These examples are illustrative; the quantitative conclusions are based on all 300 test trajectories for each equation.

## Predictive-latent representations transfer to unseen reaction laws

We next tested whether reuse survived a change in the governing reaction law rather than only variation within a source equation. Starting from the same source-pretrained checkpoint, we adapted RD-JEPA to Lambda–Omega, Barkley and Oregonator using $K \in \{ 1 , 5 , 1 0 \}$ target-system trajectories. The reaction terms and trajectories of all three equations were excluded from pretraining. For each equation and value of $K ,$ three paired runs shared prespecified support sets across models and used the same 300 test trajectories.

RD-JEPA had the lowest mean relative $L ^ { 2 }$ error among RD-JEPA, FNO and the no-predictive-latent control for every evaluated target equation, adaptation budget and forecast horizon (Fig. 5a). This ordering was already present after adaptation with one trajectory and persisted through h = 5. At K = 1, horizon-averaged relative $\bar { L } ^ { 2 }$ and gradient $L ^ { 1 }$ errors were lower for RD-JEPA than for both comparators (Fig. 5b,c). Because none of the target reaction laws or trajectories contributed to pretraining, these results support equation-level transfer of the learned representation rather than reuse confined to sourceequation parameter regimes.

The independently trained no-predictive-latent control removes the trajectory-specific encoder–predictor pathway while retaining the physical-context decoder route and lead-time conditioning. RD-JEPA retained lower mean field and gradient errors under this intervention, supporting the contribution of the complete predictive-latent representation pathway to few-trajectory transfer (Supplementary Note 5 and Supplementary Table S13).

Representative one-trajectory forecasts connect the aggregate errors to the propagation of fronts, waves and interfaces in the three target systems (Fig. 4). Together with the test-set summaries, these examples show how a representation pretrained on five source systems supports forecasts after adaptation to three previously unseen reaction laws.

## Cross-equation transfer persists across model classes and controlled training schemes

To determine whether the transfer result depended on FNO as the supervised comparator, we evaluated four additional equation specific surrogates on the three target equations: $\mathrm { L N O ^ { 4 1 } }$ $\mathrm { R e V i T } ^ { 4 2 }$ , R $\mathrm { i e s z N O } ^ { 4 3 }$ and $\mathrm { C N e x t U - N e t } ^ { 2 4 }$ . Together with $\mathrm { F N O ^ { 1 1 } }$ , these models span spectral and transform-domain neural operators, an equivariant transformer and a convolutional encoder–decoder. Table 1 reports horizon-averaged relative $L ^ { 2 }$ and gradient $L ^ { 1 }$ errors for $K \in \{ 1 , 5 , 1 0 \}$ across the three paired support-set selections. RD-JEPA alone reused a representation pretrained on the five source equations.

RD-JEPA had the lowest mean on both metrics in all nine equation–budget settings (Table 1). For Lambda–Omega at $K = 1 0$ , its mean relative $L ^ { 2 }$ error was essentially indistinguishable from that of RieszNO, supporting comparable performance in this setting. In the other eight settings, RD-JEPA had a lower mean relative $L ^ { 2 }$ error than the strongest external supervised baseline. Persistence across spectral, transform-domain, equivariant and convolutional surrogates makes the transfer result unlikely to be an artefact of selecting FNO as the comparator. The three run-level values underlying each entry are provided in Supplementary Data 3.

An architecture-matched control tested whether architecture alone could reproduce the transfer result when the encoder– predictor–decoder was trained end to end from random initialization. RD-JEPA had lower mean relative $L ^ { 2 }$ and gradient $L ^ { 1 }$ errors in all nine target-equation–budget settings (Supplementary Note 6 and Supplementary Table S14). Under matched support sets, downstream objectives and update budgets, source pretraining therefore provided transferable information that the same architecture did not recover from the limited target data alone.

We finally tested adaptation beyond the periodic boundary conditions used during pretraining. For Barkley dynamics with homogeneous Neumann, Robin and Dirichlet conditions, RD-JEPA had the lowest mean relative $L ^ { 2 }$ and gradient $L ^ { 1 }$ errors at $K = 1$ , 5 and 10 (Supplementary Table S15). The result extends the observed transfer to a combined shift in boundary conditions and diffusion discretization within Barkley at the evaluated domain, resolution and simulation protocol.

## Discussion

Our results show that predictive-latent pretraining can transform trajectories from several reaction–diffusion equations into a reusable dynamical representation that remains effective after adaptation to previously unseen reaction laws. Reuse on the source equations first established data efficiency, whereas transfer to Lambda–Omega, Barkley and Oregonator showed that the representation was not confined to source-equation parameter regimes. The mean-error ordering persisted across adaptation budgets, forecast horizons and specialized surrogate architectures, indicating that the transferable signal was not specific to one downstream comparator.

![](images/8952062bfc1ccf1b4d9eab8bd15280efa0cbd2f5758a602a3b34eda9b0f74fef.jpg)

b Qualitative comparison across PDEs (K = 5, h = 5)  
![](images/0025102ff22ab63c6d1b8eeef155fa5ed84e84b12ab89b18d1069b3aec071925.jpg)

![](images/c5fef67f6dce2b4a0d705f5df2fc9cad920496693da69eb0ecc4322d5554b22b.jpg)  
Figure 2. Few-trajectory forecasting on the source reaction–diffusion systems. a, Direct forecasts from the prespecified representative downstream run for a FitzHugh–Nagumo test trajectory after adaptation with K = 5 trajectories. The two field components, u and v, are shown at forecast horizons $h = 1 , 3$ and 5. For each component and horizon, columns show the reference field, the RD-JEPA prediction and the pointwise absolute error. Absolute-error colour limits are rescaled independently at each horizon, so magnitudes should be read from the corresponding colour bars rather than compared by colour intensity. The displayed trajectory was selected randomly from the held-out test set. b, Cross-equation qualitative comparison from the same representative run at K = 5 and $h = 5$ for Gray–Scott, Ginzburg–Landau, Schnakenberg and Brusselator. Columns show the reference solution, RD-JEPA, the no-predictive-latent control and an equation-specific FNO trained from scratch using the same adaptation budget. Fields within each row share a common colour scale. c, Mean relative $L ^ { 2 }$ error at $K = 5$ and $h = 5 ,$ reported separately for each source equation and as an equal-weight mean across the five equations (Overall). Each run-level equation score is averaged over the same 300 held-out test trajectories. Markers and error bars show the mean and sample standard deviation across three paired support-selection runs, respectively. The Overall score is first averaged across equations within each run. Lower values indicate better performance, and the horizontal axis is logarithmic.

This result extends cross-system PDE pretraining beyond objectives that predict or reconstruct physical fields directly. Multiple Physics Pretraining, Poseidon and masked function-space pretraining learn reusable models or representations through field-level prediction or reconstruction<sup>27,</sup> <sup>28,</sup> <sup>33</sup>, whereas RD-JEPA predicts future representations from trajectory context. By moving the predictive target from field space to representation space, RD-JEPA tests a different locus for shared dynamical information, following the JEPA principle developed for visual representation learning<sup>39,</sup> <sup>40</sup>. Physical JEPA studies have also considered parameter inference and physics-informed surrogate pretraining<sup>36,</sup> <sup>37</sup>. The present work identifies representation-space prediction as a viable pretraining target for shared learning across reaction laws followed by few-trajectory full-field forecasting. Direct comparisons among predictive, reconstructive and contrastive objectives offer a natural route for characterizing their relative transfer properties.

![](images/97e352d0ef59d24aa0c4b6d8539f548a19033332f9aac4c2da454ec7b1949d5b.jpg)  
c Cross-condition summary

b Few-shot scaling across PDEs  
![](images/758a9c09854f44fcb1c713ee47c12e50538b51c465b6c5e02773d50d338070b7.jpg)

![](images/a696fc27cbb9ab848d812b245bc9b19fb9ff1921414a8174cf13fee0c5d4d500.jpg)

![](images/9df5b4f5e58b22d57798a33a2112ea818c65b60dd5a33291aa16aca49a8a517c.jpg)  
Figure 3. Dependence of source-system forecasting accuracy on adaptation budget and forecast horizon. a, Mean relative $L ^ { 2 }$ error as a function of forecast horizon $h \in \{ 1 , \ldots , 5 \}$ for adaptation budgets $K = 5 ,$ , 10 and 20. RD-JEPA is compared with the no-predictive-latent control (orange) and an equation-specific FNO (purple). Each run-level value is an equal-weight mean across the five source equations after averaging over 300 fixed test trajectories per equation. Points and error bars show the mean and sample standard deviation across three paired support-selection runs. The vertical axis is logarithmic. b, Equation-level few-trajectory scaling at $h = 5$ . Relative $L ^ { 2 }$ error is plotted against K for each source equation and for the equal-weight mean across equations. Points and error bars again show the mean and sample standard deviation across the three runs. Both axes are logarithmic. c, Summary across all adaptation budgets and forecast horizons. The upper heatmap reports the across-run mean relative $L ^ { 2 }$ error of RD-JEPA. The lower heatmap reports its percentage reduction relative to FNO, 100 $\scriptstyle \left( 1 - { \frac { \mathrm { e r r } _ { \mathrm { R D - J E P A } } } { \mathrm { e r r } _ { \mathrm { F N O } } } } \right)$ . Positive values indicate lower error for RD-JEPA. Equation- and split-resolved across-run summaries are reported in Supplementary Tables S6–S12, with individual run-level values in Supplementary Data 1.

RD-JEPA also embeds a problem-specific hypothesis through complementary diffusion-like and reaction-like latent pathways. This decomposition serves as an inductive bias rather than an exact operator splitting and requires no equation identifier, coefficient or symbolic governing equation. The no-predictive-latent and architecture-matched scratch controls evaluate the complete source-pretrained pathway, whose transfer benefit persists under both comparisons. Component-wise analyses could further resolve how the two predictor pathways and the pretraining objective interact.

The present scope covers deterministic, two-component reaction–diffusion systems at one two-dimensional resolution and five directly predicted horizons. The non-periodic Barkley experiment evaluates a coupled shift in boundary conditions and diffusion discretization. RD-JEPA operates through few-trajectory target adaptation rather than zero-shot inference. Variability estimates capture support-set selection for one predictive-pretraining checkpoint and one fixed model-training random-numbergenerator state. Longer rollouts, additional resolutions and repeated pretraining runs will be needed to characterize stability and variability beyond this setting.

B  
![](images/e85b4ce721adef793dbe457b40d9a1b78942a51c08d1b272184ff086db9b183b.jpg)

![](images/78d067ba990f69e82487772586f2589e017686b142f79b4aa7fcd2e9d7e345c4.jpg)  
Figure 4. Representative one-trajectory forecasts on governing equations excluded from pretraining. a, Ground-truth and RD-JEPA forecasts of the u field for Lambda–Omega, Barkley and Oregonator after adaptation with one target trajectory, shown at forecast horizons $h = 1 , 3$ and 5. Ground truth and prediction share a common colour scale within each equation. The dashed line in each $h = 5$ ground-truth field marks the spatial transect used in panel b. b, Values of the u field along the indicated transects at $h = 5 ,$ plotted against normalized spatial coordinate. Black solid lines denote ground truth, blue solid lines RD-JEPA and orange dashed lines the no-predictive-latent control. Aggregate three-run results over all 300 held-out test trajectories are reported in Fig. 5 and Supplementary Table S13.

Within these limits, future-state prediction in representation space provides a viable route to domain-specific reusable dynamics models. A simulation corpus from several related equations can be amortized into one representation and repeatedly specialized when target-system trajectories are scarce. In this operational sense, RD-JEPA constitutes a predictive latent world model and a step towards foundation modelling for reaction–diffusion dynamics.

B  
![](images/1415026f291082880296f4874ac39b2582aedc833a699f67616bb15487ed003b.jpg)  
Figure 5. Few-trajectory forecasting on governing equations excluded from pretraining. a, Mean relative $L ^ { 2 }$ error versus forecast horizon $h = 1 , \ldots , 5$ for Lambda–Omega, Barkley and Oregonator (columns) after adaptation with $K = 1 , 5$ and 10 trajectories (rows). RD-JEPA (blue) is compared with the no-predictive-latent control (orange) and FNO (purple). Each run-level value is averaged over the same 300 held-out test trajectories. Points and error bars show the mean and sample standard deviation across three paired support-selection runs; vertical axes are logarithmic. b, Percentage reduction in horizon-averaged relative $L ^ { 2 }$ error at $K = 1$ , computed within each matched run as 100 $\begin{array} { r } { \left( 1 - \frac { \mathrm { e r r } _ { \mathrm { R D - J E P A } } } { \mathrm { e r r } _ { \mathrm { b a s e l i n e } } } \right) } \end{array}$ , relative to FNO (purple) and the control (orange). Bars and error bars show the mean and sample standard deviation across the three paired run-level reductions. c, Corresponding paired reduction in horizon-averaged gradient $L ^ { 1 }$ error. Positive values indicate lower error for RD-JEPA. The horizon-averaged full-model–control comparison is reported in Supplementary Table S13, and the horizon-resolved run-level values underlying the figure are provided in Supplementary Data 2.

## Methods

## Reaction–diffusion datasets and experimental design

We considered two-field reaction–diffusion systems of the form

$$
\partial _ { t } \mathbf { u } = \mathbf { D } \nabla ^ { 2 } \mathbf { u } + \mathbf { R } ( \mathbf { u } ; \pmb { \theta } ) , \qquad \mathbf { u } = ( u , \nu ) ^ { \top } .\tag{2}
$$

Predictive pretraining used Gray–Scott, FitzHugh–Nagumo, Brusselator, complex Ginzburg–Landau and Schnakenberg dynamics. Lambda–Omega, Barkley and Oregonator were held out at the governing-equation level: their reaction terms and trajectories were excluded from pretraining. Each primary-benchmark trajectory contained 40 two-channel fields on a final 128 × 128 periodic grid. The governing equations, coefficient and initial-condition distributions, numerical solvers and temporal sampling intervals are provided in Supplementary Note 1 and Supplementary Table S1.

For each source equation, we generated 1,000 source-pool trajectories used for predictive pretraining and downstream support selection, 50 validation trajectories and 300 test trajectories. The test set comprised 100 in-distribution trajectories, 100 trajectories from held-out coefficient regimes and 100 trajectories from held-out initial-condition families. Each equation excluded from pretraining used an independently simulated pool of 1,000 trajectories for downstream adaptation and a fixed set of 300 additional test trajectories. All partitions were trajectory-disjoint, and every temporal window inherited the split of it parent trajectory.

Source-equation adaptation used $K \in \{ 5 , 1 0 , 2 0 \}$ trajectories, whereas equation-level transfer used $K \in \{ 1 , 5 , 1 0 \}$ . Here, K counts complete trajectories rather than temporal windows. We evaluated three prespecified support-set selections. Within each selection, support sets were nested across K and shared by all methods, and each adapted or from-scratch model was evaluated on the same 300 test trajectories. The predictive-pretraining checkpoint and model-training random-number-generator state were fixed across selections, so only the selected support trajectories varied. A fixed channel-wise affine transformation, estimated without test data, was applied consistently across methods. Support-selection seeds, the fixed training seeds and preprocessing are reported in Supplementary Note 3.

Table 1. Forecasting accuracy on governing equations excluded from pretraining. K denotes the number of independent target-system trajectories used for adaptation. Each run-level score is obtained by first averaging each trajectory’s error over $h = 1 , \ldots , 5$ and then averaging across the same 300 held-out test trajectories. For every model, values are the mean ± sample standard deviation across three paired support-selection runs. Values are multiplied by 100, and lower values are better. Boldface marks the lowest numerical mean for each equation and adaptation budget; no formal hypothesis tests were performed. The paired horizon-averaged comparison between RD-JEPA and the no-predictive-latent control is reported in Supplementary Table S13.
<table><tr><td></td><td></td><td colspan="3">Relative  $L ^ { 2 } \downarrow ( \times 1 0 ^ { - 2 } )$ </td><td colspan="3">Gradient  $L ^ { 1 } \downarrow ( \times 1 0 ^ { - 2 } )$ </td></tr><tr><td>PDE</td><td>Model</td><td>K = 1</td><td> $K = 5$ </td><td> $K = 1 0$ </td><td> $K = 1$ </td><td> $K = 5$ </td><td> $K = 1 0$ </td></tr><tr><td>Lambda-Omega</td><td>RD-JEPA</td><td> ${ \pm } 2 . 2 4 \pm 2 . 5 8$ </td><td> ${ \bf 9 . 8 5 \pm 0 . 7 9 }$ </td><td> ${ \pm 1 . 3 8 \pm 1 . 4 6 }$ </td><td> $\mathbf { 0 . 6 8 \pm 0 . 1 2 }$ </td><td> ${ \bf 0 . 5 2 \pm 0 . 0 5 }$ </td><td> ${ \bf 0 . 4 5 \pm 0 . 0 8 }$ </td></tr><tr><td></td><td>FNO</td><td> $5 0 . 5 3 \pm 1 3 . 0 6$ </td><td> $2 5 . 4 8 \pm 1 . 0 9$ </td><td> $2 0 . 2 9 \pm 1 . 5 9$ </td><td> $3 . 8 7 \pm 1 . 0 2$ </td><td> $1 . 8 9 \pm 0 . 0 7$ </td><td> $1 . 5 0 \pm 0 . 0 8$ </td></tr><tr><td></td><td>LNO</td><td> $3 4 . 5 6 \pm 8 . 1 5$ </td><td> $2 1 . 0 7 \pm 1 . 2 6$ </td><td> $1 6 . 6 5 \pm 2 . 1 6$ </td><td> $2 . 5 1 \pm 0 . 5 5$ </td><td> $1 . 5 7 \pm 0 . 1 1$ </td><td> $1 . 3 7 \pm 0 . 1 6$ </td></tr><tr><td></td><td>ReViT</td><td> $2 2 . 6 4 \pm 4 . 1 1$ </td><td> $1 5 . 3 5 \pm 0 . 7 2$ </td><td> $1 3 . 6 1 \pm 1 . 3 7$ </td><td> $2 . 8 5 \pm 0 . 2 2$ </td><td> $3 . 2 4 \pm 0 . 1 4$ </td><td> $3 . 0 4 \pm 0 . 1 3$ </td></tr><tr><td></td><td> $\mathrm { R i e s z N O }$ </td><td> $1 9 . 2 0 \pm 4 . 7 2$ </td><td> $1 3 . 4 1 \pm 0 . 8 8$ </td><td> $8 . 4 7 \pm 1 . 1 4$ </td><td> $1 . 2 6 \pm 0 . 2 8$ </td><td> $0 . 8 8 \pm 0 . 0 6$ </td><td> $0 . 4 7 \pm 0 . 0 6$ </td></tr><tr><td></td><td>CNextU-Net</td><td> $2 5 . 5 2 \pm 6 . 0 3$ </td><td> $2 1 . 9 4 \pm 1 . 3 2$ </td><td> $1 6 . 5 4 \pm 2 . 1 6$ </td><td> $1 . 7 2 \pm 0 . 3 8$ </td><td> $2 . 0 2 \pm 0 . 1 4$ </td><td> $1 . 9 2 \pm 0 . 2 2$ </td></tr><tr><td>Barkley</td><td>RD-JEPA</td><td> ${ \bf 1 0 . 5 3 \pm 2 . 5 5 }$ </td><td> ${ \bf 6 . 9 0 \pm 0 . 2 4 }$ </td><td> ${ \pm } 0 . 2 3 \pm 0 . 3 2$ </td><td> ${ \bf 1 . 0 9 \pm 0 . 2 7 }$ </td><td> ${ \bf 0 . 7 4 \pm 0 . 0 4 }$ </td><td> ${ \bf 0 . 5 8 \pm 0 . 0 4 }$ </td></tr><tr><td></td><td>FNO</td><td> $2 9 . 8 6 \pm 2 . 5 2$ </td><td> $1 6 . 1 3 \pm 0 . 7 5$ </td><td> $1 3 . 1 5 \pm 0 . 3 3$ </td><td> $3 . 1 1 \pm 0 . 2 6$ </td><td> $1 . 6 6 \pm 0 . 0 7$ </td><td> $1 . 3 7 \pm 0 . 0 4$ </td></tr><tr><td></td><td>LNO</td><td> $3 4 . 5 8 \pm 3 . 1 5$ </td><td> $1 7 . 6 5 \pm 0 . 7 8$ </td><td> $1 6 . 3 9 \pm 0 . 6 2$ </td><td> $1 . 9 7 \pm 0 . 2 8$ </td><td> $1 . 5 9 \pm 0 . 0 7$ </td><td> $1 . 5 9 \pm 0 . 0 8$ </td></tr><tr><td></td><td>ReViT</td><td> $2 2 . 0 3 \pm 1 . 0 3$ </td><td> $1 4 . 7 2 \pm 0 . 3 5$ </td><td> $1 1 . 0 8 \pm 0 . 2 8$ </td><td> $2 . 2 4 \pm 0 . 1 2$ </td><td> $1 . 5 0 \pm 0 . 0 3$ </td><td> $1 . 1 5 \pm 0 . 0 3$ </td></tr><tr><td></td><td>RieszNO</td><td> $4 6 . 7 5 \pm 4 . 0 7$ </td><td> $2 4 . 5 9 \pm 1 . 0 5$ </td><td> $2 1 . 9 9 \pm 0 . 6 9$ </td><td> $4 . 0 9 \pm 0 . 3 8$ </td><td> $3 . 0 9 \pm 0 . 1 3$ </td><td> $3 . 0 0 \pm 0 . 1 0$ </td></tr><tr><td></td><td>CNextU-Net</td><td> $3 0 . 8 5 \pm 3 . 1 0$ </td><td> $1 7 . 4 0 \pm 0 . 7 3$ </td><td> $1 3 . 8 9 \pm 0 . 5 6$ </td><td> $3 . 9 7 \pm 0 . 4 2$ </td><td> $2 . 1 6 \pm 0 . 1 0$ </td><td> $1 . 8 7 \pm 0 . 0 7$ </td></tr><tr><td>Oregonator</td><td>RD-JEPA</td><td> ${ \bf 1 1 . 3 6 \pm 1 . 3 4 }$ </td><td> $\mathbf { 4 . 8 3 \pm 0 . 7 9 }$ </td><td> $\mathbf { 4 . 0 5 \pm 0 . 4 7 }$ </td><td> ${ \bf 0 . 4 7 \pm 0 . 0 5 }$ </td><td> $\mathbf { 0 . 2 6 \pm 0 . 0 6 }$ </td><td> ${ \bf 0 . 2 2 \pm 0 . 0 2 }$ </td></tr><tr><td></td><td>FNO</td><td> $4 6 . 1 7 \pm 7 . 0 7$ </td><td> $1 8 . 7 9 \pm 4 . 4 2$ </td><td> $1 3 . 7 6 \pm 2 . 2 1$ </td><td> $2 . 3 2 \pm 0 . 2 7$ </td><td> $1 . 0 4 \pm 0 . 1 5$ </td><td> $0 . 7 3 \pm 0 . 0 8$ </td></tr><tr><td></td><td>LNO</td><td> $2 9 . 4 1 \pm 3 . 4 2$ </td><td> $2 0 . 6 6 \pm 3 . 8 7$ </td><td> $1 3 . 3 9 \pm 1 . 6 5$ </td><td> $1 . 2 9 { \pm } 0 . 1 3$ </td><td> $0 . 9 3 \pm 0 . 1 4$ </td><td> $0 . 6 9 \pm 0 . 0 7$ </td></tr><tr><td></td><td>ReViT</td><td> $2 6 . 7 3 \pm 1 . 9 6$ </td><td> $2 0 . 4 9 \pm 1 . 1 0$ </td><td> $2 0 . 0 3 \pm 1 . 3 8$ </td><td> $2 . 6 4 \pm 0 . 0 7$ </td><td> $2 . 0 1 \pm 0 . 0 4$ </td><td> $2 . 0 1 \pm 0 . 0 3$ </td></tr><tr><td></td><td> $\mathrm { R i e s z N O }$ </td><td> $2 7 . 4 2 \pm 3 . 2 8$ </td><td> $1 9 . 1 4 \pm 2 . 9 1$ </td><td> $1 6 . 5 0 \pm 1 . 9 0$ </td><td> $1 . 3 5 { \pm } 0 . 1 6$ </td><td> $1 . 0 0 \pm 0 . 1 7$ </td><td> $0 . 8 6 \pm 0 . 0 9$ </td></tr><tr><td></td><td> $\mathrm { C N e x t U - N e t }$ </td><td> $2 6 . 0 1 \pm 2 . 8 3$ </td><td> $1 6 . 7 6 \pm 3 . 3 1$ </td><td> $1 3 . 8 3 \pm 1 . 6 8$ </td><td> $2 . 0 8 \pm 0 . 2 0$ </td><td> $1 . 1 1 \pm 0 . 1 9$ </td><td> $0 . 9 0 \pm 0 . 1 1$ </td></tr></table>

A separate boundary-condition stress test used Barkley dynamics with homogeneous Neumann, Robin or Dirichlet conditions. Each boundary-specific dataset contained 20 adaptation trajectories, five validation trajectories and 300 test trajectories; $K \in \{ 1 , 5 , 1 0 \}$ trajectories were selected for adaptation. These trajectories did not enter predictive pretraining. Common index-level randomization aligned coefficient and initial-condition draws before boundary-specific quality screening, so the three datasets represent matched generating distributions rather than exactly paired trajectories. Within each boundary condition, all methods received identical support trajectories and were evaluated on the same test trajectories. Dataset construction is detailed in Supplementary Note 1.

## Joint-embedding predictive pretraining

RD-JEPA follows the joint-embedding predictive formulation used for image and video representation learning<sup>39,</sup> <sup>40</sup>. Given four consecutive states $\mathbf { X } _ { t } ^ { c } = \left( \mathbf { x } _ { t - 3 } , \ldots , \mathbf { x } _ { t } \right)$ , an online encoder $E _ { \theta }$ produced 256 spatial context tokens of width 512. An exponential-moving-average target encoder $E _ { \xi }$ represented the target field $\mathbf { X } _ { t + h } .$ , and a predictor $P _ { \phi }$ estimated its target-patch representations from the context, patch position and forecast horizon h, measured in stored-frame intervals. The target branch received no gradients and followed the online encoder through an exponential moving average. One model was shared across the five source equations and received no equation identity, coefficient vector or symbolic governing equation.

Each channel was embedded with non-overlapping $8 \times 8$ patches, yielding a fused $1 6 \times 1 6$ token field. Temporal crossattention aggregated the four context frames, after which a multiscale U-shaped operator encoder combined global, spectral and local spatial mixing. The latent predictor alternated target self-attention and context cross-attention with two reaction–diffusionaligned updates. For context token $\mathbf { z } _ { c , i }$ , the diffusion-like feature was

$$
\Delta _ { \mathrm { l a t } } \mathbf { z } _ { c , i } = \frac { 1 } { 4 } \sum _ { j \in \mathcal { N } ( i ) } \mathbf { z } _ { c , j } - \mathbf { z } _ { c , i } ,\tag{3}
$$

where $\mathcal { N } ( i )$ contains the four periodic nearest neighbours in the primary benchmark. A pointwise pathway supplied a reactionlike update, and both pathways were conditioned on a continuous embedding of h. For the non-periodic Barkley stress test, the predictor retained this pretrained periodic neighbour map and received no boundary-condition label or explicit target-domain boundary operator. Complete block dimensions, positional encodings, conditioning operations and parameter counts are given in Supplementary Note 2 and Supplementary Table S3.

Training batches were balanced across the five source equations. Pretraining sampled future targets up to eight stored-frame intervals ahead and combined future-block, future-tube and same-time masks. The online encoder and predictor minimized the representation-space objective

$$
\mathcal { L } _ { \mathrm { J E P A } } = \frac { 1 } { B M D } \sum _ { b = 1 } ^ { B } \sum _ { m = 1 } ^ { M } \left. \widetilde { \widehat { \mathbf { z } } } _ { b , m } - \mathrm { s g } \left( \widetilde { \mathbf { z } } _ { b , m } \right) \right. _ { 2 } ^ { 2 } ,\tag{4}
$$

where B is the batch size, $M = 2 5 6 , D = 5 1 2 ,$ , sg denotes stop-gradient and the tildes denote the feature transformation defined in Supplementary Note 2. No field decoder was attached during pretraining. We trained for 200,000 optimization steps; the masking distribution and remaining optimization settings are listed in Supplementary Tables S2 and S4.

## Few-trajectory adaptation and comparison models

For downstream forecasting, the online encoder was frozen and the target encoder was discarded. The pretrained predictor was fine-tuned jointly with a newly initialized dense decoder. The predictor was queried independently at $h \in \{ 1 , \ldots , 5 \}$ , and the decoder fused its $1 6 \times 1 6$ latent field with multiscale features from the latest context state. The model therefore predicted each requested horizon directly rather than through autoregressive rollout. One predictor–decoder pair was adapted for each equation, value of K and support-set selection.

RD-JEPA and the no-predictive-latent control used a supervised objective combining pointwise, relative-field, spatialgradient and Fourier-magnitude errors, with uniform weighting over the five horizons. Each primary-benchmark adaptation used 5,000 optimization steps. All comparison models and controls used the same supervised objective and loss weights as RD-JEPA. Optimizer settings, loss weights and the complete decoder specification are reported in Supplementary Notes 2 and 3 and Supplementary Tables S3–S5.

The no-predictive-latent control was trained independently. It removed the JEPA encoder–predictor pathway, retained the same last-frame decoder pathway and supplied only a spatially constant lead-time embedding to the latent interface. The control therefore contained no trajectory-specific predictive latent and evaluates the contribution of the complete sample-dependent predictive-latent pathway under the downstream forecasting protocol. The supervised comparisons additionally included FNO, LNO, RieszNO, ReViT and CNextU-Net. Each baseline received the same support trajectories, four-frame context and five direct target horizons as RD-JEPA.

The architecture-matched scratch control retained the complete encoder–predictor–decoder architecture but trained it end to end from random initialization on the three equations excluded from pretraining. It used the same support trajectories, direct horizons, downstream objective, 5,000-update budget and fixed model-training random-number-generator state as RD-JEPA, but no pretrained tensors, exponential-moving-average target branch or JEPA objective. The final-update checkpoint was evaluated without validation- or test-based checkpoint selection. This architecture-matched comparison evaluates source-pretrained adaptation against end-to-end training from random initialization under identical support sets, forecast horizons, downstream objectives and update budgets.

For the non-periodic Barkley stress test, all models used 5,000 optimization steps. Method-specific objectives, optimization budgets and architecture details are given in Supplementary Notes 3 and 6 and Supplementary Table S5.

## Evaluation and statistical aggregation

All errors were computed on the original field scale. The principal metrics were relative field error and first-order spatial-gradient error,

$$
E _ { \mathrm { r e l } L ^ { 2 } } = \frac { \| \widehat { \mathbf { x } } - \mathbf { x } \| _ { 2 } } { \| \mathbf { x } \| _ { 2 } + 1 0 ^ { - 8 } } , \qquad E _ { \nabla L ^ { 1 } } = \mathbf { M A E } ( \delta _ { x } \widehat { \mathbf { x } } , \delta _ { x } \mathbf { x } ) + \mathbf { M A E } ( \delta _ { y } \widehat { \mathbf { x } } , \delta _ { y } \mathbf { x } ) .\tag{5}
$$

Here, $\delta _ { x }$ and $\delta _ { y }$ denote adjacent first differences along the two spatial axes, without a wrap-around term. The mean absolute errors are averaged over both field channels and all valid neighbouring grid pairs. The same metric definition is used for the periodic and non-periodic datasets. Mean-squared error was retained as a secondary diagnostic for the primary source- and target-equation benchmarks and is supplied in Supplementary Data 1–3.

Every valid temporal window was evaluated. With 40 frames, a four-frame context and a maximum horizon of five, each trajectory contributed 32 windows. Errors were first averaged over windows within each trajectory and then with equal weight over the 300 test trajectories. Horizon-averaged scores assigned equal weight to $h = 1 , \ldots , 5$ . Each displayed value was computed separately for each support-set selection and then summarized as the mean and sample standard deviation across the three run-level means. Thus, $n = 3$ denotes support-set selections, not test trajectories or independent pretraining runs. The same 300 test trajectories were reused across selections, and the predictive-pretraining checkpoint and model-training random-number-generator state were fixed. The reported standard deviation therefore quantifies sensitivity to support-set selection. No formal hypothesis tests were performed. Complete aggregation definitions and source-data mappings are provided in Supplementary Note 3.

## Acknowledgments

This work was partially supported by the National Natural Science Foundation of China (72495131, 82441027), Guangdong Provincial Key Laboratory of Mathematical Foundations for Artificial Intelligence (2023B1212010001), Shenzhen Stability Science Program, and the Shenzhen Science and Technology Program under grant no. JCYJ20250604141043020.

## Author Contributions Statement

C.S. and M.Y. conceived the study. C.S. developed the method, implemented the models, generated the datasets, performed the experiments and prepared the figures. C.S. and M.Y. analysed the results and wrote the manuscript. M.Y. supervised the project.

## Data availability

The source data underlying the quantitative figures and tables are organized as Supplementary Data 1–5 and will be deposited in a public archival repository. For each quantitative display, the deposit will contain the run-level values and their reported across run mean and sample standard deviation. Run-level aggregation follows the averaging unit stated in the corresponding caption: 300 trajectories for pooled equation results, 100 trajectories for each split-resolved source result and 300 boundary-specific trajectories for the Barkley stress test. The deposit will also include the numerical arrays underlying the qualitative fields and transects, prespecified dataset splits, support-set indices and plotting inputs. Simulation and evaluation code, parameter files and recorded random seeds will permit regeneration of the trajectory-level benchmark. The persistent repository identifier will be added before publication.

## Code availability

Code for simulation, predictive pretraining, downstream adaptation, baseline training, evaluation and figure generation will be made available to editors and reviewers through an anonymous repository and archived publicly upon publication. The release will include environment specifications, configuration files and instructions linking each reported result to its source-data file. A persistent version identifier will be added before publication.

## Competing interests

The authors declare no competing interests.

## References

1. Turing, A. M. The chemical basis of morphogenesis. Philos. Transactions Royal Soc. London. Ser. B, Biol. Sci. 237, 37–72 (1952).

2. Kondo, S. & Miura, T. Reaction-diffusion model as a framework for understanding biological pattern formation. science 329, 1616–1620 (2010).

3. Milinkovitch, M. C., Jahanbakhsh, E. & Zakany, S. The unreasonable effectiveness of reaction diffusion in vertebrate skin color patterning. Annu. Rev. Cell Dev. Biol. 39, 145–174 (2023).

4. Raspopovic, J., Marcon, L., Russo, L. & Sharpe, J. Digit patterning is controlled by a Bmp-Sox9-Wnt turing network modulated by morphogen gradients. Science 345, 566–570 (2014).

5. Sheth, R. et al. Hox genes regulate digit patterning by controlling the wavelength of a turing-type mechanism. Science 338, 1476–1480 (2012).

6. Scholes, N. S., Schnoerr, D., Isalan, M. & Stumpf, M. P. A comprehensive network atlas reveals that turing patterns are common but not robust. Cell systems 9, 243–257 (2019).

7. Landge, A. N., Jordan, B. M., Diego, X. & Müller, P. Pattern formation mechanisms of self-organizing reaction-diffusion systems. Dev. biology 460, 2–11 (2020).

8. Fuseya, Y., Katsuno, H., Behnia, K. & Kapitulnik, A. Nanoscale Turing patterns in a bismuth monolayer. Nat. Phys. 17, 1031–1036 (2021).

9. Raissi, M., Perdikaris, P. & Karniadakis, G. E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. J. Comput. physics 378, 686–707 (2019).

10. Karniadakis, G. E. et al. Physics-informed machine learning. Nat. Rev. Phys. 3, 422–440 (2021).

11. Li, Z. et al. Fourier neural operator for parametric partial differential equations. In International Conference on Learning Representations (2021).

12. Lu, L., Jin, P., Pang, G., Zhang, Z. & Karniadakis, G. E. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nat. Mach. Intell. 3, 218–229 (2021).

13. Kovachki, N. et al. Neural operator: Learning maps between function spaces with applications to PDEs. J. Mach. Learn. Res. 24, 1–97 (2023).

14. Li, Z. et al. Physics-informed neural operator for learning partial differential equations. ACM/IMS J. Data Sci. 1, 1–27 (2024).

15. Sanchez-Gonzalez, A. et al. Learning to simulate complex physics with graph networks. In Proceedings of the 37th International Conference on Machine Learning, 8459–8468 (Proceedings of Machine Learning Research, 2020).

16. Brandstetter, J., Worrall, D. E. & Welling, M. Message passing neural PDE solvers. In International Conference on Learning Representations (2022).

17. Gupta, J. K. & Brandstetter, J. Towards multi-spatiotemporal-scale generalized PDE modeling. Transactions on Mach Learn. Res. (2023).

18. Lippe, P., Veeling, B. S., Perdikaris, P., Turner, R. E. & Brandstetter, J. PDE-refiner: Achieving accurate long rollouts with neural PDE solvers. In Thirty-seventh Conference on Neural Information Processing Systems (2023).

19. Cao, S. Choose a transformer: Fourier or galerkin. In Beygelzimer, A., Dauphin, Y., Liang, P. & Vaughan, J. W. (eds.) Advances in Neural Information Processing Systems (2021).

20. Hao, Z. et al. GNOT: A general neural operator transformer for operator learning. In Proceedings ofthe 40th International Conference on Machine Learning, 12556–12569 (PMLR, 2023).

21. Wu, H., Luo, H., Wang, H., Wang, J. & Long, M. Transolver: A fast transformer solver for PDEs on general geometries. In Forty-first International Conference on Machine Learning (2024).

22. Holzschuh, B., Liu, Q., Kohl, G. & Thuerey, N. PDE-transformer: Efficient and versatile transformers for physics simulations. In Proceedings ofthe 42nd International Conference on Machine Learning (2025).

23. Takamoto, M. et al. PDEBench: An extensive benchmark for scientific machine learning. In Advances in Neural Information Processing Systems, vol. 35, 1596–1611 (2022).

24. Ohana, R. et al. The well: a large-scale collection of diverse physics simulations for machine learning. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track (2024).

25. Finn, C., Abbeel, P. & Levine, S. Model-agnostic meta-learning for fast adaptation of deep networks. In International conference on machine learning, 1126–1135 (PMLR, 2017).

26. Psaros, A. F., Kawaguchi, K. & Karniadakis, G. E. Meta-learning PINN loss functions. J. computational physics 458, 111121 (2022).

27. McCabe, M. et al. Multiple physics pretraining for spatiotemporal surrogate models. In Advances in Neural Information Processing Systems, vol. 37, 119301–119335 (2024).

28. Herde, M. et al. Poseidon: Efficient foundation models for PDEs. In Advances in Neural Information Processing Systems, vol. 37, 72525–72624 (2024).

29. Sun, J., Liu, Y., Zhang, Z. & Schaeffer, H. Towards a foundation model for partial differential equations: Multioperator learning and extrapolation. Phys. Rev. E 111, 035304 (2025).

30. Yang, L., Liu, S., Meng, T. & Osher, S. J. In-context operator learning with data prompts for differential equation problems. Proc. Natl. Acad. Sci. 120, e2310142120 (2023).

31. Alkin, B. et al. Universal physics transformers: A framework for efficiently scaling neural operators. In The Thirty-eighth Annual Conference on Neural Information Processing Systems (2024).

32. Subramanian, S. et al. Towards foundation models for scientific machine learning: Characterizing scaling and transfer behavior. In Thirty-seventh Conference on Neural Information Processing Systems (2023).

33. Rahman, M. A. et al. Pretraining codomain attention neural operators for solving multiphysics PDEs. In Advances in Neural Information Processing Systems, vol. 37, 104035–104064 (2024).

34. Chen, W. et al. Data-efficient operator learning via unsupervised pretraining and in-context learning. In Advances in Neural Information Processing Systems, vol. 37, 6213–6245 (2024).

35. Lorsung, C., Li, Z. & Barati Farimani, A. Physics informed token transformer for solving partial differential equations. Mach. Learn. Sci. Technol. 5, 015032 (2024).

36. Qu, H. et al. Representation learning for spatiotemporal physical systems. arXiv preprint arXiv:2603.13227 (2026). ICLR 2026 Workshop on AI and PDE, 2603.13227.

37. Yee, B. & Koh, P. PI-JEPA: Label-free surrogate pretraining for coupled multiphysics simulation via operator-split latent prediction. arXiv preprint arXiv:2604.01349 (2026).

38. LeCun, Y. et al. A path towards autonomous machine intelligence. Open Rev. 62, 1–62 (2022).

39. Assran, M. et al. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 15619–15629 (2023).

40. Bardes, A. et al. Revisiting feature prediction for learning visual representations from video. Transactions on Mach. Learn. Res. (2024).

41. Cao, Q., Goswami, S. & Karniadakis, G. E. Laplace neural operator for solving differential equations. Nat. Mach. Intell. 6, 631–640 (2024).

42. Wei, H., List, B. & Thuerey, N. ReViT: Rotational-equivariant vision transformers for neural PDE solvers. In Forty-third International Conference on Machine Learning (2026).

43. Liu, S., Yang, X. & Chen, Y. Riesz neural operator for solving partial differential equations. In International Conference on Learning Representations (2026).

44. Pearson, J. E. Complex patterns in a simple system. Science 261, 189–192 (1993).

45. FitzHugh, R. Impulses and physiological states in theoretical models of nerve membrane. Biophys. J. 1, 445–466 (1961).

46. Nagumo, J., Arimoto, S. & Yoshizawa, S. An active pulse transmission line simulating nerve axon. Proc. IRE 50, 2061–2070 (1962).

47. Prigogine, I. & Lefever, R. Symmetry breaking instabilities in dissipative systems. ii. The J. Chem. Phys. 48, 1695–1700 (1968).

48. Aranson, I. S. & Kramer, L. The world of the complex ginzburg–landau equation. Rev. Mod. Phys. 74, 99–143 (2002).

49. Schnakenberg, J. Simple chemical reaction systems with limit cycle behaviour. J. Theor. Biol. 81, 389–400 (1979).

50. Sherratt, J. A. On the evolution of periodic plane waves in reaction–diffusion systems of λ–ω type. SIAM J. on Appl. Math. 54, 1374–1385 (1994).

51. Barkley, D. A model for fast computer simulation of waves in excitable media. Phys. D: Nonlinear Phenom. 49, 61–70 (1991).

52. Field, R. J., Körös, E. & Noyes, R. M. Oscillations in chemical systems. ii. thorough analysis of temporal oscillation in the bromate–cerium–malonic acid system. J. Am. Chem. Soc. 94, 8649–8664 (1972).

53. Field, R. J. & Noyes, R. M. Oscillations in chemical systems. iv. limit cycle behavior in a model of a real chemical reaction. The J. Chem. Phys. 60, 1877–1884 (1974).

Supplementary Information for

# RD-JEPA: Predictive latent pretraining for few-trajectory transfer across reaction–diffusion equations

Chenhao Si<sup>1</sup>, Ming Yan<sup>1,∗</sup>

<sup>1</sup>School of Data Science, The Chinese University of Hong Kong, Shenzhen, Shenzhen, China

<sup>∗</sup>Corresponding author: yanming@cuhk.edu.cn

## Contents

## References

Supplementary Note 1: Reaction–diffusion systems and dataset construction 16   
S1.1 Benchmark composition, evaluation regimes, and notation 16   
S1.2 Source systems used for predictive pretraining 16   
S1.3 Held-out systems for equation-level transfer 17   
S1.4 Numerical integration and trajectory screening 18   
S1.5 Barkley benchmark with non-periodic boundary conditions 18   
Supplementary Note 2: RD-JEPA architecture and predictive latent learning 20   
S2.1 Overall two-stage RD-JEPA formulation 20   
S2.2 Online encoder $E _ { \theta } \colon$ spatiotemporal tokenization and multiscale spatial encoding 20   
S2.3 Target sampling and EMA target encoding 22   
S2.4 Lead-time-conditioned predictor with diffusion-inspired and reaction-inspired branches 23   
S2.5 Predictive latent objective and pretraining . 24   
S2.6 Dense forecasting and adaptation from few trajectories 25   
S2.7 Architecture and optimization summary 26   
Supplementary Note 3: Experimental protocols, baselines and statistics 28   
S3.1 Forecasting tasks, support selection and evaluation 28   
S3.2 Comparison models and controls 28   
S3.3 Training objectives and optimization settings 29   
S3.4 Evaluation metrics and statistical aggregation 29   
Supplementary Note 4: Complete forecasting results on source systems 31   
S4.1 Evaluation settings and error averaging 31   
S4.2 Complete relative-field-error results 31   
S4.3 Errors in spatial first differences 31   
S4.4 Split-resolved forecasting results 31   
Supplementary Note 5: Paired predictive-latent control on held-out governing equations 35   
S5.1 Evaluation coverage, data mapping and interpretation 35   
S5.2 Horizon-averaged paired comparison 35   
Supplementary Note 6: Matched control and boundary-condition stress test 36   
S6.1 Architecture-matched training without predictive pretraining 36   
S6.2 Few-trajectory adaptation under non-periodic boundary conditions 36

## Supplementary Note 1: Reaction–diffusion systems and dataset construction

## S1.1 Benchmark composition, evaluation regimes, and notation

We constructed a benchmark comprising eight two-dimensional, two-component reaction–diffusion systems. Trajectories from Gray–Scott, FitzHugh–Nagumo, Brusselator, complex Ginzburg–Landau, and Schnakenberg were used for predictive pretraining. Lambda–Omega, Barkley, and Oregonator were reserved for downstream transfer; neither their nonlinear reaction operators nor any trajectories generated by them were included in pretraining

All systems are written in the form

$$
\partial _ { t } \mathbf { u } ( \mathbf { x } , t ) = \mathbf { D } \nabla ^ { 2 } \mathbf { u } ( \mathbf { x } , t ) + \mathbf { R } ( \mathbf { u } ( \mathbf { x } , t ) ; \pmb \theta ) , \qquad \mathbf { u } = ( u , \nu ) ^ { \top } ,\tag{S1}
$$

where $\mathbf { D } = \mathrm { d i a g } ( D _ { u } , D _ { \nu } )$ is the diffusion matrix and R is the nonlinear reaction operator with parameters θ . Each trajectory in the primary benchmark consists of 40 stored two-channel fields represented on a $1 2 8 \times 1 2 8$ periodic grid and is stored as $\mathbf { X } ^ { ( i ) } \in \overset { \cdot } { \mathbb { R } } ^ { 4 0 \times 1 2 \mathbf { \bar { 8 } } \times 1 2 8 \times 2 }$ . The coefficient and initial-condition distributions were chosen to generate a range of spatial and temporal behaviors, including spots, fronts, oscillations, phase defects, and spiral waves.

For each source system, evaluation was performed under three test regimes: in-distribution, coefficient-out-of-distribution, and initial-condition-out-of-distribution. These source-system evaluations are distinct from the held-out-system transfer experiment, in which the complete Lambda–Omega, Barkley, and Oregonator systems were not included in pretraining.

For each source system, we generated a pool of 1,000 trajectories for predictive pretraining and downstream support-set selection, together with 50 validation trajectories and 300 test trajectories. The test set contained 100 trajectories from each of the three evaluation regimes. The 1,000-trajectory source pool, validation set, and test set were mutually disjoint at the trajectory level. The support trajectories used for source-system adaptation were selected from the source pool used during predictive pretraining. For each held-out system, a separate pool of 1,000 trajectories was used for support-set selection, and 300 additional trajectories formed the fixed test set. Every temporal window inherited its parent trajectory’s partition.

The forecast horizon h is measured in stored-frame intervals. Because the internal integration step and frame-saving interval are system-dependent, the same value of h does not necessarily correspond to the same physical elapsed time across systems.

## S1.2 Source systems used for predictive pretraining

Unless otherwise stated, each scalar model parameter specified by an interval below was sampled independently from the uniform distribution on that interval.

Gray–Scott. We considered the Gray–Scott system

$$
\begin{array} { r l } & { \partial _ { \tau } u = D _ { u } \nabla ^ { 2 } u - u \nu ^ { 2 } + f ( 1 - u ) , } \\ & { \partial _ { \tau } \nu = D _ { \nu } \nabla ^ { 2 } \nu + u \nu ^ { 2 } - ( f + k ) \nu . } \end{array}\tag{S2}
$$

(S3)

This system supports spot, stripe, maze, and moving-pattern $\mathrm { r e g i m e s } ^ { 4 4 }$ . The implementation used the rescaled time variable $t = \tau / 1 0 0 0$ , with all coefficients transformed consistently. We first selected a parameter center $( f _ { 0 } , k _ { 0 } )$ from

$$
( 0 . 0 0 8 , 0 . 0 4 6 ) , ( 0 . 0 2 0 , 0 . 0 5 6 ) , ( 0 . 0 4 0 , 0 . 0 6 0 ) , ( 0 . 0 2 9 , 0 . 0 5 7 ) , ( 0 . 0 5 8 , 0 . 0 6 5 ) .
$$

Then both parameters were perturbed independently by up to 6%. The reference diffusion coefficients $D _ { u } = 2 \times 1 0 ^ { - 5 }$ and $D _ { \nu } = 1 0 ^ { - 5 }$ were perturbed independently by up to 5%. Initial conditions consisted of smooth, low-frequency Fourier fields, localized Gaussian perturbations, and their mixtures. Their phases, centers, spatial scales, and amplitudes were randomized.

FitzHugh–Nagumo. We generated excitable dynamics using the FitzHugh–Nagumo system<sup>45,</sup> <sup>46</sup>

$$
\begin{array} { r l } & { \partial _ { t } u = \gamma _ { u } \nabla ^ { 2 } u + u - u ^ { 3 } - \nu + \alpha , } \\ & { \partial _ { t } \nu = \gamma _ { \nu } \nabla ^ { 2 } \nu + \beta ( u - \nu ) , } \end{array}\tag{S4}
$$

(S5)

with $\gamma _ { u } = 1 , \gamma _ { \nu } = 1 0 0 , \alpha \sim \mathcal { U } ( 0 . 0 0 5 , 0 . 0 1 8 )$ , and $\beta \sim \mathcal { U } ( 0 . 2 2 , 0 . 3 5 )$ . The initial u- and v-fields were generated independently as Gaussian random fields, with their standard deviations sampled from [0.04,0.06].

Brusselator. We simulated the Brusselator system<sup>47</sup>

$$
\begin{array} { r l } & { \partial _ { t } u = D _ { u } \nabla ^ { 2 } u + \rho \left[ A - ( B + 1 ) u + u ^ { 2 } \nu \right] , } \\ & { \partial _ { t } \nu = D _ { \nu } \nabla ^ { 2 } \nu + \rho \left[ B u - u ^ { 2 } \nu \right] , } \end{array}\tag{S6}
$$

(S7)

with $A \in [ 0 . 7 , 1 . 1 ] , B \in [ 2 . 0 , 2 . 6 ] , D _ { u } \in [ 1 0 ^ { - 4 } , 3 \times 1 0 ^ { - 4 } ] , D _ { \nu } \in [ 5 \times 1 0 ^ { - 4 } , 1 . 2 \times 1 0 ^ { - 3 } ]$ , and $\rho \in [ 8 , 1 2 ]$ . For a given pair $( A , B )$ the spatially homogeneous equilibrium is $\left( u _ { * } , \nu _ { * } \right) = \left( A , \frac { B } { A } \right)$ . Initial conditions were smooth perturbations of the equilibrium. Each perturbation combined a low-pass random field, a weak sinusoidal component, and a localized Gaussian component.

Complex Ginzburg–Landau. Let $\psi = u + \mathrm { i } \nu$ be a complex-valued field. We considered the complex Ginzburg–Landau equation<sup>48</sup>

$$
\partial _ { t } \psi = \varepsilon \nabla ^ { 2 } \psi + \nu \psi - \big ( \gamma _ { r } + \mathrm { i } \gamma _ { i } \big ) | \psi | ^ { 2 } \psi .\tag{S8}
$$

Equivalently, its real and imaginary components satisfy

$$
\begin{array} { r l } & { \partial _ { t } u = \varepsilon \nabla ^ { 2 } u + \nu u - ( u ^ { 2 } + \nu ^ { 2 } ) ( \gamma _ { \mathrm { r } } u - \gamma _ { \mathrm { i } } \nu ) , } \\ & { \partial _ { t } \nu = \varepsilon \nabla ^ { 2 } \nu + \nu \nu - ( u ^ { 2 } + \nu ^ { 2 } ) ( \gamma _ { \mathrm { r } } \nu + \gamma _ { \mathrm { i } } u ) . } \end{array}\tag{S9}
$$

(S10)

We fixed $\nu = \gamma _ { \mathrm { r } } = 1$ . The pair $( \varepsilon , \gamma _ { i } )$ was sampled from one of five parameter regimes with different phase-rotation behavior. Across the five regimes, the parameter values covered $\pmb { \varepsilon } \in [ 0 . 0 0 2 , 0 . 0 1 0 ] , \gamma _ { i } \in [ 0 . 5 , 9 ]$ . Initial conditions consisted of randomized vortex seeds, noisy single-vortex states, and phase-disk configurations. The initial-condition-out-of-distribution split used noisier states containing multiple vortices.

Schnakenberg. We used the Schnakenberg system<sup>49</sup>

$$
\begin{array} { r l } & { \partial _ { t } u = D _ { u } \nabla ^ { 2 } u + \rho ( a - u + u ^ { 2 } \nu ) , } \\ & { \partial _ { t } \nu = D _ { \nu } \nabla ^ { 2 } \nu + \rho ( b - u ^ { 2 } \nu ) , } \end{array}\tag{S11}
$$

(S12)

with $a \in [ 0 . 0 3 , 0 . 0 6 ] , b \in [ 0 . 8 4 , 0 . 8 8 ] , D _ { u } \in [ 1 0 ^ { - 4 } , 3 \times 1 0 ^ { - 4 } ] , D _ { \nu } \in [ 5 \times 1 0 ^ { - 4 } , 1 . 2 \times 1 0 ^ { - 3 } ]$ and $\rho \in [ 8 , 1 2 ]$ . For given a and $^ { b , }$ the spatially homogeneous equilibrium is $\begin{array} { r } { ( u _ { * } , \nu _ { * } ) = \left( a + b , \frac { b } { ( a + b ) ^ { 2 } } \right) } \end{array}$ . Initial conditions were smooth perturbations of this equilibrium and used the same low-pass random, sinusoidal, and localized components as those used for the Brusselator.

## S1.3 Held-out systems for equation-level transfer

The three systems below were excluded entirely from predictive pretraining and were used only for downstream adaptation and evaluation.

Lambda–Omega. Letting $r ^ { 2 } = u ^ { 2 } + \nu ^ { 2 }$ . We considered the Lambda–Omega system<sup>50</sup>

$$
\begin{array} { r l } & { \partial _ { t } u = D _ { u } \nabla ^ { 2 } u + \rho \left[ \alpha ( 1 - r ^ { 2 } ) u - ( \omega _ { 0 } - \beta r ^ { 2 } ) \nu \right] , } \\ & { \partial _ { t } \nu = D _ { \nu } \nabla ^ { 2 } \nu + \rho \left[ ( \omega _ { 0 } - \beta r ^ { 2 } ) u + \alpha ( 1 - r ^ { 2 } ) \nu \right] . } \end{array}\tag{S13}
$$

(S14)

The coefficients were sampled from $\alpha \in [ 0 . 7 5 , 1 . 2 5 ] , \omega _ { 0 } \in [ 0 . 8 5 , 1 . 2 0 ] , \beta \in [ 0 . 2 0 , 0 . 7 0 ] , \rho \in [ 0 . 8 5 , 1 . 2 0 ]$ and $D _ { u } , D _ { \nu } \in [ 2 . 5 \times$ $1 0 ^ { - 4 } , 7 . 5 \times 1 0 ^ { - 4 } ]$ . Initial conditions were drawn from three families: spiral-defect states, target waves, and oblique phase waves. Their centers, orientations, spatial frequencies, and smooth perturbations were randomized.

Barkley. We considered the Barkley system

$$
\begin{array} { l } { \displaystyle \partial _ { t } u = D _ { u } \nabla ^ { 2 } u + \frac { 1 } { \varepsilon } u ( 1 - u ) \left( u - \frac { \nu + b } { a } \right) , } \\ { \displaystyle \partial _ { t } \nu = D _ { \nu } \nabla ^ { 2 } \nu + u - \nu . } \end{array}\tag{S15}
$$

(S16)

We used the singular diffusive setting $D _ { \nu } = 0 .$ , so that only the activator field $u \mathrm { d i f f u s e s } ^ { 5 1 }$ . The remaining coefficients were sampled from $a \in [ 0 . 7 2 , 0 . 7 8 ] , b \in [ 0 . 0 1 5 , 0 . 0 2 8 ] , \varepsilon \in [ 0 . 0 1 8 , 0 . 0 2 4 ] { \mathrm { ~ a n d ~ } } D _ { u } \in [ 0 . 9 0 , 1 . 1 0 ]$ . Initial conditions were broken-wave configurations formed from intersecting excited and refractory fronts, with randomized orientation, displacement, width, and smooth-noise perturbations.

Oregonator. We considered the reduced two-variable Oregonator system

$$
\begin{array} { l } { \displaystyle \partial _ { t } u = D _ { u } \nabla ^ { 2 } u + \frac { \rho } { \varepsilon } \left[ u ( 1 - u ) - f \nu \frac { u - q } { u + q } \right] , } \\ { \displaystyle \partial _ { t } \nu = D _ { \nu } \nabla ^ { 2 } \nu + \rho ( u - \nu ) , } \end{array}\tag{S17}
$$

(S18)

which is a reduced activator–inhibitor model of Belousov–Zhabotinsky-type chemistry<sup>52,</sup> <sup>53</sup>. We sampled $\pmb { \varepsilon } \in [ 0 . 0 4 0 , 0 . 0 8 0 ]$ $f \in [ 1 . 1 0 , 1 . 5 5 ] , q \in [ 0 . 0 0 1 5 , 0 . 0 0 4 5 ] , D _ { u } \in [ 4 \times 1 0 ^ { - 5 } , 1 . 6 \times 1 0 ^ { - 4 } ] , D _ { \nu } \in [ 1 0 ^ { - 5 } , 8 \times 1 0 ^ { - 5 } ]$ and $\rho \in [ 0 . 7 0 , 1 . 4 0 ]$ . Initial conditions were excitation blobs, ring-shaped target waves, or broken wavefronts superimposed on the positive spatially homogeneous equilibrium, together with weak low-pass noise.

## S1.4 Numerical integration and trajectory screening

All simulations in the primary benchmark used periodic boundary conditions. For Gray–Scott, Brusselator, complex Ginzburg– Landau, Schnakenberg, Lambda–Omega, and Oregonator, the Laplacian was discretized using second-order periodic central finite differences, and the resulting semi-discrete systems were advanced in time using explicit Euler. FitzHugh–Nagumo used a fourth-order periodic finite-difference approximation of the Laplacian together with classical fourth-order Runge–Kutta integration. Barkley was integrated using Strang splitting: the reaction substep was advanced by the explicit midpoint method, whereas the diffusion substep was integrated exactly in Fourier space. Barkley trajectories were simulated on a $2 5 6 \times 2 5 6$ grid and subsequently average-pooled to $1 2 8 \times 1 2 8$ . The numerical and temporal settings are summarized in Table S1.

Table S1. Numerical discretization and temporal sampling in the primary periodic benchmark. Each retained trajectory contains 40 saved two-channel fields.
<table><tr><td>System</td><td>Domain and grid</td><td>Integrator and spatial discretization</td><td>Internal ∆t</td><td>Saved window</td></tr><tr><td>Gray-Scott</td><td> $[ - 1 , 1 ) ^ { 2 } , 1 2 8 \times 1 2 8$ </td><td>Explicit Euler; second-order periodic central differences</td><td> $1 0 ^ { - 4 }$ </td><td> $t \in [ 0 . 0 5 , 1 . 0 0 ]$ </td></tr><tr><td>FitzHugh-Nagumo</td><td>Periodic  $1 2 8 \times 1 2 8$  lattice,  $\Delta x = 1$ </td><td>RK4; fourth-order periodic finite difference Laplacian</td><td> $2 \times 1 0 ^ { - 3 }$ </td><td> $\Delta t _ { \mathrm { s a v e } } = 1$ </td></tr><tr><td>Brusselator</td><td> $[ - 1 , 1 ) ^ { 2 } , 1 2 8 \times 1 2 8$ </td><td>Explicit Euler; second-order periodic central differences</td><td> $1 0 ^ { - 3 }$ </td><td> $t \in [ 2 , 4 ]$ </td></tr><tr><td>Complex Ginzburg- Landau</td><td> $[ - 1 , 1 ) ^ { 2 } , 1 2 8 \times 1 2 8$ </td><td>Explicit Euler; second-order periodic central differences</td><td> $1 0 ^ { - 3 }$ </td><td> $t \in [ 2 , 1 0 ]$ </td></tr><tr><td>Schnakenberg</td><td> $[ - 1 , 1 ) ^ { 2 } , 1 2 8 \times 1 2 8$ </td><td>Explicit Euler; second-order periodic central differences</td><td> $1 0 ^ { - 3 }$ </td><td> $t \in [ 2 , 4 ]$ </td></tr><tr><td>Lambda-Omega</td><td> $[ - 1 , 1 ) ^ { 2 } , 1 2 8 \times 1 2 8$ </td><td>Explicit Euler; second-order periodic central differences</td><td> $2 \times 1 0 ^ { - 3 }$ </td><td> $t \in [ 3 , 1 2 ]$ </td></tr><tr><td>Barkley</td><td>Periodic square of side length  $1 5 0 ; 2 5 6 \times 2 5 6 $   $1 2 8 \times 1 2 8$ </td><td>Strang splitting; explicit-midpoint reaction and exact spectral diffusion</td><td> $1 0 ^ { - 2 }$ </td><td> $t \in [ 5 , 1 5 ]$ </td></tr><tr><td>Oregonator</td><td> $[ - 1 , 1 ) ^ { 2 } , 1 2 8 \times 1 2 8$ </td><td>Explicit Euler; second-order periodic central differences</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td> $t \in [ 0 . 2 , 0 . 8 ]$ </td></tr></table>

After numerical integration, system-specific quality screening was applied before a trajectory was included in the benchmark. Simulations were excluded if they contained non-finite values, exhibited amplitude divergence, showed negligible spatial or temporal variation, or contained spatial structures that were not adequately resolved on the simulation grid. Gray–Scott, Brusselator, and Schnakenberg were screened using temporal-change criteria; complex Ginzburg–Landau was screened using spatial-variance and total-variation criteria; Oregonator was screened using temporal-change, concentration, and resolved wavelength criteria; and Lambda–Omega was screened using a multi-frame change criterion. Barkley trajectories were screened for numerical validity.

## S1.5 Barkley benchmark with non-periodic boundary conditions

To assess adaptation beyond the periodic domains used for predictive pretraining, we generated three additional Barkley datasets on the same square domain of side length 150 as in the primary Barkley benchmark. Let Ω denote this domain and let $\partial _ { n } u = \nabla u \cdot \mathbf { n }$ denote the outward normal derivative on ∂Ω. For each dataset, the activator field u satisfied one of the following homogeneous boundary conditions:

$$
\left. \begin{array} { r l } { \partial _ { n } u = 0 } & { { } \quad \mathrm { ( N e u m a n n ) } , } \\ { \partial _ { n } u + \kappa u = 0 , } & { { } \kappa = 0 . 1 5 \quad \mathrm { ( R o b i n ) } , } \\ { u = 0 } & { { } \quad \mathrm { ( D i r i c h l e t ) } , } \end{array} \right\} \qquad \mathrm { o n } \ \partial \Omega .\tag{S19}
$$

Because $D _ { \nu } = 0 .$ , the inhibitor equation contains no spatial-diffusion term, and no spatial boundary operator is required for v. Each boundary-specific dataset comprised 20 adaptation trajectories, five validation trajectories, and 300 test trajectories. Every trajectory contained 40 stored two-channel fields. Simulations were performed on a cell-centered $2 5 6 \times 2 5 6$ grid, and the stored fields were subsequently average-pooled to $1 2 8 \times 1 2 8$ . We used $\Delta t = 0 . 0 1$ and retained 40 fields over the interval $t \in [ 5 , 1 5 ]$ . The coefficient ranges were identical to those of the primary Barkley benchmark.

For all three boundary conditions, the diffusion operator was approximated using a second-order five-point finite-difference Laplacian. Time integration used Strang splitting, with the reaction and diffusion subproblems advanced by the explicit midpoint method. The boundary conditions were imposed through second-order ghost-cell relations. Initial conditions were compact broken waves with a quiescent collar adjacent to the boundary and were therefore compatible with all three homogeneous boundary conditions.

Before boundary-specific quality screening, common index-level randomization was used to align the coefficient draws and initial-condition parameters across the three boundary conditions. Because screening was performed separately for each condition, the retained datasets represent matched generating distributions but are not paired trajectory-by-trajectory. None of these trajectories was used during predictive pretraining. Within each boundary condition and support-set selection, all models used the same support trajectories and were evaluated on the same fixed test trajectories.

## Supplementary Note 2: RD-JEPA architecture and predictive latent learning

## S2.1 Overall two-stage RD-JEPA formulation

A joint-embedding predictive architecture (JEPA) learns to predict the latent representation of a target observation from an observed context<sup>39,</sup> <sup>40</sup>. An encoder maps observations to learned feature vectors, referred to here as latent tokens. RD-JEPA applies this principle to reaction-diffusion trajectories in two stages. During predictive latent pretraining, an online encoder and a predictor are trained to predict target tokens at selected spatial patches and time offsets. During downstream adaptation, the online encoder is frozen, while the predictor is fine-tuned jointly with a decoder that produces future-field predictions from the predicted tokens and the latest observed field.

Predictive latent pretraining. Let $\boldsymbol { x } _ { t } \in \mathbb { R } ^ { H \times W \times C }$ denote the discretized reaction-diffusion field at saved time index t, expressed in the standardized coordinates. In all reported experiments, $H = W = 1 2 8$ and $C = 2$ . The model receives a context window of four consecutive saved fields,

$$
X _ { t } ^ { c } = \left( x _ { t - c + 1 } , \ldots , x _ { t } \right) = \left( x _ { t - 3 } , x _ { t - 2 } , x _ { t - 1 } , x _ { t } \right) \in \mathbb { R } ^ { c \times H \times W \times C } , \qquad c = 4 .\tag{S20}
$$

The online encoder $E _ { \theta }$ maps the context window to a spatial grid of latent tokens,

$$
Z _ { t } ^ { c } = E _ { \theta } ( X _ { t } ^ { c } ) = \left( z _ { t , 1 } ^ { c } , \ldots , z _ { t , S } ^ { c } \right) \in \mathbb { R } ^ { S \times d _ { \mathrm { l a t } } } , \qquad \mathrm { w h e r e ~ } S = 1 6 \times 1 6 = 2 5 6 , \quad d _ { \mathrm { l a t } } = 5 1 2 .\tag{S21}
$$

Each token $z _ { t , i } ^ { c } \in \mathbb { R } ^ { d _ { \mathrm { l a t } } }$ is associated with spatial patch i and aggregates spatial and temporal information from the context window. The encoder architecture is described in Section S2.2.

The target encoder $E _ { \xi }$ provides reference representations by processing each complete target field separately:

$$
Z _ { t + h } ^ { \mathrm { t a r } } = E _ { \xi } \left( x _ { t + h } \right) \in \mathbb { R } ^ { S \times d _ { \mathrm { l a t } } } .\tag{S22}
$$

Its parameters $\xi$ are updated via an exponential moving average (EMA) of the corresponding online encoder parameters. The target-encoder construction and EMA update are detailed in Section S2.3.

For each pretraining sample, let $\mathcal { I } = ( ( i _ { m } , h _ { m } ) ) _ { m = 1 } ^ { M }$ be the ordered query list, where $i _ { m } \in \{ 1 , . . . , S \}$ identifies a spatial patch and $h _ { m }$ specifies a time offset. The predictor $P _ { \phi }$ processes the context tokens and all queries jointly, returning one predicted token per query:

$$
\widehat { z } _ { m } = \left[ P _ { \phi } ( Z _ { t } ^ { c } , \mathcal { S } ) \right] _ { m } , \qquad z _ { m } ^ { \mathrm { t a r } } = \left[ Z _ { t + h _ { m } } ^ { \mathrm { t a r } } \right] _ { i _ { m } } , \qquad m = 1 , \ldots , M .\tag{S23}
$$

Query selection and the predictor architecture are described in Sections S2.3 and S2.4, respectively. The online encoder and predictor are trained using the loss in Section S2.5, which compares the normalized predicted and target tokens. A single RD-JEPA model is pretrained jointly on all five source systems.

Dense downstream forecasting. During downstream adaptation, the online encoder parameters θ are held fixed, while the predictor $P _ { \phi }$ and decoder $D _ { \psi }$ are trained jointly by updating φ and ψ. For a requested forecast offset $h ,$ the complete query list $\mathcal { I } _ { h } ^ { \mathrm { f u l l } } = \left( ( i , h ) \right) _ { i = 1 } ^ { S }$ requests a latent token at every spatial patch. The predictor, therefore, produces a complete latent grid,

$$
\widehat { Z } _ { t + h } = P _ { \phi } ( Z _ { t } ^ { c } , \mathcal { I } _ { h } ^ { \mathrm { f u l l } } ) \in \mathbb { R } ^ { S \times d _ { \mathrm { l a t } } } .\tag{S24}
$$

The decoder combines this predicted latent grid with the latest context field to produce

$$
\begin{array} { r } { \widehat { x } _ { t + h } = D _ { \psi } \left( \widehat { Z } _ { t + h } , x _ { t } \right) \in \mathbb { R } ^ { H \times W \times C } . } \end{array}\tag{S25}
$$

The field $x _ { t }$ is processed through a convolutional pathway at the original $1 2 8 \times 1 2 8$ spatial resolution. We evaluate this construction for $h \in \{ 1 , 2 , 3 , 4 , 5 \}$ . All five future fields are predicted directly from the same observed context, rather than b recursively feeding predicted fields back into the model.

## S2.2 Online encoder $E _ { \theta }$ : spatiotemporal tokenization and multiscale spatial encoding

The online encoder $E _ { \theta }$ maps the four-frame context $X _ { t } ^ { c }$ to the spatial latent grid $Z _ { t } ^ { c }$ in three stages. It first tokenizes each frame independently, then aggregates the four tokens at each spatial position, and finally exchanges information across spatial positions through a multiscale encoder.

Per-frame tokenization. For each context frame $r \in \left\{ t - 3 , t - 2 , t - 1 , t \right\}$ , let $P _ { r , i } = { \mathrm { p a t c h } } _ { i } ( x _ { r } ) \in \mathbb { R } ^ { 8 \times 8 \times 2 }$ denote the i-th non-overlapping spatial patch, containing both scalar components of the reaction-diffusion field. Partitioning each $1 2 8 \times 1 2 8$ field into these patches gives a $1 6 \times 1 6$ token grid.

The two scalar components of each patch are projected separately, and the resulting feature vectors are averaged. Since both components are present in every patch, the combined operation can be represented as $\Pi ( P _ { r , i } ) + e$ , where $\Pi : \overline { { \mathbb { R } } } ^ { 8 \times 8 \times 2 }  \mathbb { R } ^ { d _ { \mathrm { l a t } } }$ is a structured linear map and $e \in \mathbb { R } ^ { d _ { \mathrm { l a } } }$ t collects the constant terms, including the averaged channel-identity embeddings. A residual two-layer mixing map then produces the patch token:

$$
y _ { r , i } = \Pi ( P _ { r , i } ) + e + W _ { 2 } \sigma _ { \mathrm { G E L U } } ( W _ { 1 } \ L \mathbf { N } ( \Pi ( P _ { r , i } ) + e ) + b _ { 1 } ) + b _ { 2 } .\tag{S26}
$$

Here, $d _ { \mathrm { l a t } } = 5 1 2 , W _ { \mathrm { l } } , W _ { \mathrm { 2 } } \in \mathbb { R } ^ { d _ { \mathrm { l a t } } \times d _ { \mathrm { l a t } } }$ , and $b _ { 1 } , b _ { 2 } \in \mathbb { R } ^ { d _ { \mathrm { l a t } } }$ . The projection, offset, and mixing parameters are learned during pretraining and shared across spatial positions, context frames, samples, and source systems. The activation is $\sigma _ { \mathrm { G E L U } } ( s ) = s \Phi ( s )$ , where Φ is the standard normal cumulative distribution function. Layer normalization, denoted by LN, acts over the latent feature dimension, with $1 0 ^ { - 5 }$ added to the variance before taking its square root.

Spatial positions and relative frame indices are encoded by fixed sinusoidal embeddings. For an even integer D, define

$$
\begin{array} { r } {  f _ { D } ( a ) = [ ( \sin ( a 1 0 0 0 0 ^ { - 2 j / D } ) ) _ { j = 0 } ^ { D / 2 - 1 } , ( \cos ( a 1 0 0 0 0 ^ { - 2 j / D } ) ) _ { j = 0 } ^ { D / 2 - 1 } ] ^ { \top } \in \mathbb { R } ^ { D } , } \end{array}
$$

with all sine components followed by all cosine components. Let $( a _ { i } , b _ { i } ) \in \{ 0 , \ldots , 1 5 \} ^ { 2 }$ be the row and column indices of patch $i ,$ and let $\ell ( r ) = r - ( t - 3 ) \in \{ 0 , 1 , 2 , 3 \}$ be the position of frame r within the context window. The positional embeddings are

$$
e _ { i } ^ { \mathrm { s p a c e } } = \left[ f _ { { d _ { \mathrm { l a t } } } / 2 } ( a _ { i } ) \right] , \qquad e _ { \ell ( r ) } ^ { \mathrm { t i m e } } = f _ { { d _ { \mathrm { l a t } } } } ( \ell ( r ) ) ,\tag{S27}
$$

both in $\mathbb { R } ^ { d _ { \mathrm { l a t } } }$ . The spatial embedding concatenates row and column embeddings, whereas the temporal embedding encodes the frame’s relative position within the four-frame context.

During predictive pretraining, patch tokens may be masked after tokenization and before temporal aggregation. For the same-time target task $( h = 0 )$ , the tokens $y _ { t , i }$ at the requested target patches in the latest context field $x _ { t }$ are masked. The target-selection procedure is described in Section ${ \bf S } 2 . 3$

Independent of the target type, each training sample receives additional random context masking with probability 0.35. For such a sample, a single masking probability $p$ is drawn uniformly from [0.05,0.25]. Conditional on $p ,$ each patch token in each of the four context frames is independently masked with probability p. Thus, 0.35 controls whether a sample receives additional masking, whereas $p$ controls the masking probability within that sample.

Let $\mathcal { M }$ be the union of the same-time target mask and the additional random context mask, represented as pairs $( r , i )$ of frame and patch indices. Each selected token is replaced by a learned mask vector $e _ { \mathrm { e n c } } ^ { \mathrm { m a s k } } \in \mathbb { R } ^ { d _ { \mathrm { l a t } } }$ . The positional embeddings and a learned encoder embedding $e _ { \mathrm { e n c } } \in \mathbb { R } ^ { d _ { \mathrm { l a t } } }$ are then added to obtain the input to temporal aggregation:

$$
\widetilde { y } _ { r , i } = \left\{ \begin{array} { l l } { e _ { \mathrm { e n c } } ^ { \mathrm { m a s k } } , } & { ( r , i ) \in \mathcal { M } , } \\ { y _ { r , i } , } & { ( r , i ) \not \in \mathcal { M } } \end{array} \right. + e _ { i } ^ { \mathrm { s p a c e } } + e _ { \ell ( r ) } ^ { \mathrm { t i m e } } + e _ { \mathrm { e n c } } .\tag{S28}
$$

Both $e _ { \mathrm { e n c } } ^ { \mathrm { m a s k } }$ and $e _ { \mathrm { e n c } }$ are shared across spatial positions, context frames, samples, and source systems. Masked tokens retain their spatial and temporal embeddings, so their locations and frame indices remain available to the encoder. The target encoder receives complete, unmasked target fields during pretraining. No context masking is applied during downstream forecasting.

Temporal aggregation. At each spatial position i, the four embedded context tokens $\widetilde { y } _ { t - 3 , i } , \ldots , \widetilde { y } _ { t , i }$ are aggregated into a single vector $\bar { z } _ { t , i } \in \mathbb { R } ^ { d _ { \mathrm { l a t } } }$ . A query is initialized as $\widetilde { y _ { t , i } } + b _ { \mathrm { t e m p } }$ , where $b _ { \mathrm { t e m p } } \in \mathbb { R } ^ { d _ { \mathrm { l a t } } }$ is learned, and is updated by three successive cross-attention blocks. In every block, the current query attends to the same four input tokens. These tokens supply the keys and values and are not themselves updated.

Each block applies an eight-head cross-attention update followed by a two-layer feedforward update, both with residual connections. The feedforward map uses GELU activation and intermediate dimension $2 d _ { \mathrm { l a t } }$ . The three blocks have separate parameters, each shared across spatial positions.

The final queries form the temporally aggregated grid

$$
\overline { { Z } } _ { t } ^ { c } = \left[ \bar { z } _ { t , 1 } , \ldots , \bar { z } _ { t , S } \right] ^ { \mathsf { T } } \in \mathbb { R } ^ { S \times d _ { \mathrm { l a t } } } , \qquad S = 2 5 6 .\tag{S29}
$$

Temporal aggregation therefore reduces the four tokens at each patch position to one while preserving the $1 6 \times 1 6$ spatial grid.

Multiscale spatial encoding. The temporally aggregated grid $\overline { { Z } } _ { t } ^ { c }$ is processed by a five-stage U-shaped spatial encoder with a resolution sequence

$$
1 6 \times 1 6 \longrightarrow 8 \times 8 \longrightarrow 4 \times 4 \longrightarrow 8 \times 8 \longrightarrow 1 6 \times 1 6 .\tag{S30}
$$

The stages contain $( 2 , 3 , 6 , 3 , 2 )$ operator blocks, respectively, and retain feature width $d _ { \mathrm { l a t } } = 5 1 2$ throughout. On the descending path, ${ \mathrm { ~ a ~ } } 3 \times 3$ convolution with stride 2 reduces the spatial resolution between successive stages. On the ascending path, bilinear interpolation followed by a $3 \times 3$ convolution restores the resolution. At each ascending stage, the upsampled features are added to a learned projection of the descending-stage output at the same resolution, forming a U-shaped skip connection.

Each operator block applies three successive sub-operations to its input, each with a residual connection.

(i) Spatial self-attention. An eight-head self-attention layer exchanges information across all tokens at the current resolution.

(ii) Spectral–local mixing. Two branches process the same input in parallel. The spectral branch applies a two-dimensional Fourier transform to the spatial grid of each feature component independently, multiplies a selected subset of Fourier coefficients (at most eight per spatial direction, subject to the resolution of the current stage) by learned complex weights, transforms the result back to the spatial domain, mapped back to $d _ { \mathrm { l a t } }$ components by a learned affine map. The local branch applies a depthwise $3 \times 3$ convolution (independently for each feature component), followed by GELU and a $1 \times 1$ convolution. The outputs of the two branches are concatenated along the feature dimension and mapped back to $d _ { \mathrm { l a t } }$ components by a learned affine map.

(iii) Position-wise MLP. The same two learned affine maps, separated by GELU and with intermediate width $4 d _ { \mathrm { l a t } } .$ , are applied at every spatial position.

Before each of the three operations, layer normalization is applied separately to each token. The normalized components are then scaled and shifted using values computed from the shared encoder embedding $e _ { \mathrm { e n c } }$ by a learned nonlinear map. Each normalization layer has its own map; within that layer, the same scaling and shifting are applied at all spatial positions.

After the final stage, layer normalization is applied separately to the 512 feature components of each token, giving

$$
\begin{array} { r } { Z _ { t } ^ { c } = \left[ z _ { t , 1 } ^ { c } , \ldots , z _ { t , S } ^ { c } \right] ^ { \mathsf { T } } \in \mathbb { R } ^ { S \times d _ { \mathrm { l a t } } } , \qquad S = 2 5 6 . } \end{array}\tag{S31}
$$

Each output token remains associated with a spatial patch but incorporates information from the four context frames and other spatial positions. The resulting latent grid is supplied to the lead-time-conditioned predictor described in Section S2.4.

## S2.3 Target sampling and EMA target encoding

Spatial target region. For each pretraining sample, a source system is selected uniformly from the five source systems. Rectangular target regions are then sampled on the $1 6 \times 1 6$ latent grid: the number of rectangles is drawn uniformly from $\{ 1 , \ldots , 4 \}$ ; each rectangle’s height and width are drawn independently and uniformly from $\{ 2 , \ldots , 6 \}$ patches; and its upper-left corner is placed uniformly among all valid positions inside the grid. The union of all covered patch indices defines the spatial target region ${ \mathcal { R } } \subseteq \{ 1 , \ldots , S \}$ , with overlapping positions counted once.

Target type and query list. One target type is selected according to Table S2, yielding a set of time offsets ${ \mathcal { H } } .$ . The candidate set ${ \mathcal { C } } = { \mathcal { R } } \times { \mathcal { H } }$ collects all patch–offset pairs at which the predictor must match the target encoder’s output. Because $\lvert \mathcal { C } \rvert$ varies across samples, a fixed number $M = 2 5 6$ of pairs is subsampled: if $| { \mathcal { C } } | \geq M$ , pairs are drawn without replacement; otherwise with replacement. This gives the ordered query list

$$
\mathcal { S } = \left( \left( i _ { m } , h _ { m } \right) \right) _ { m = 1 } ^ { M } , \qquad M = 2 5 6 .\tag{S32}
$$

$M = 2 5 6$ is the number of loss terms per training step; the number of masked patches is determined separately by $| { \mathcal { R } } |$ . Repeated pairs in $\mathcal { I }$ occupy separate positions and contribute independently to the loss.

For the same-time target type $( \mathcal { H } = \{ 0 \} )$ , the patches in $\mathcal { R }$ are masked in the latest context frame $x _ { t }$ of the online encoder, as described in Section S2.2. For future target types $( h > 0 )$ , no such masking is needed: future frames are not part of the online encoder input, so no information needs to be withheld.

Target encoding. For each distinct offset h appearing in ${ \mathcal { I } } .$ , the target encoder $E _ { \xi }$ processes the complete, unmasked field $x _ { t + h }$ as a separate single-frame input. The target encoder is a separate copy of the online encoder, with its own parameters $\xi$ and the same parameterized modules for tokenization, temporal aggregation, and multiscale spatial encoding. The two encoders differ in temporal input length: the online encoder processes four frames, whereas the target encoder processes one. At each spatial position, the three temporal cross-attention blocks therefore attend to a single target-frame token. This change in sequence length does not alter the parameter dimensions, so the two encoders have matching parameter tensors.

The target frame is assigned relative temporal index 0 and uses $e _ { 0 } ^ { \mathrm { t i m e } }$ for every prediction offset h. This temporal index identifies the frame’s position within the single-frame encoder input; it does not encode the prediction offset. The offset h is supplied to the latent predictor, not to the target encoder.

Table S2. Target selection during predictive pretraining. All three target types use the same spatial-region sampling procedure to obtain ${ \mathcal { R } } .$ The selected offset set $\mathcal { H }$ defines the candidate set ${ \mathcal { C } } = { \mathcal { R } } \times { \mathcal { H } }$ , from which $M = 2 5 6$ query pairs are subsampled for loss computation.
<table><tr><td>Target type</td><td>Probability</td><td>Selection of time offsets</td></tr><tr><td>Future block</td><td>0.50</td><td> $\mathcal { H } = \{ h \}$  , where h is sampled uniformly from  $\{ 1 , \ldots , 8 \}$   $\mathcal { H }$ </td></tr><tr><td>Future tube</td><td>0.40</td><td>contains four distinct offsets sampled uniformly without replacement from  $\{ 1 , \ldots , 8 \}$  . The same spatial region  $\mathcal { R }$  is used at each selected offset.</td></tr><tr><td>Same time</td><td>0.10</td><td> $\mathcal { H } = \{ 0 \}$  , targeting the latest context frame  $x _ { t } .$  The patches in  $\mathcal { R }$  are masked in the online encoder input.</td></tr></table>

All patches remain visible throughout the target-encoder forward pass. The requested target vectors are selected from the resulting latent grids according to Equation (S23). Thus, target sampling determines which output tokens enter the predictive latent loss, rather than which input patches are visible to the target encoder. The selected target vectors are treated as constants when differentiating this loss; no gradients are propagated through $E _ { \xi }$

EMA update. The target encoder is initialized from the online encoder. Let s index pretraining steps. At step s, the target encoder with parameters $\xi _ { s }$ provides the reference representations. After the optimizer updates the online-encoder parameters from $\theta _ { s }$ to $\theta _ { s + 1 }$ , the target-encoder parameters are updated by

$$
\xi _ { 0 } = \theta _ { 0 } , \qquad \xi _ { s + 1 } = m _ { s } \xi _ { s } + ( 1 - m _ { s } ) \theta _ { s + 1 } .\tag{S33}
$$

The averaging coefficient $m _ { s }$ follows the cosine schedule. The target encoder is updated only through this EMA rule, not by the optimizer.

## S2.4 Lead-time-conditioned predictor with diffusion-inspired and reaction-inspired branches

The latent predictor $P _ { \phi }$ receives the context representation $Z _ { t } ^ { c } \in \mathbb { R } ^ { S \times d _ { \mathrm { l a t } } }$ and the ordered query list $\mathcal { I } = \left( ( i _ { m } , h _ { m } ) \right) _ { m = 1 } ^ { M }$ . For each requested pair $\left( i _ { m } , h _ { m } \right)$ , it predicts a latent vector $\widehat { z } _ { m } \in \mathbb { R } ^ { d _ { \mathrm { l a } } }$ associated with spatial patch $i _ { m }$ at offset $h _ { m }$ . The predictor initializes a query vector for each requested pair, using spatial and time-offset embeddings. Eight successive blocks then exchange information among the queries, incorporate the encoded context, and apply diffusion-inspired and reaction-inspired updates.

Encoding the requested time offset. For a requested offset $h \in \{ 0 , \ldots , 8 \}$ , define $\tau = h / 8$ and construct

$$
\begin{array} { r } { \gamma ( h ) = \left[ \left( \sin ( 2 \pi 2 ^ { j } \tau ) \right) _ { j = 0 } ^ { 1 5 } , \left( \cos ( 2 \pi 2 ^ { j } \tau ) \right) _ { j = 0 } ^ { 1 5 } , \tau , \tau ^ { 2 } \right] ^ { \top } \in \mathbb { R } ^ { 3 4 } . } \end{array}\tag{S34}
$$

A shared two-layer map produces the time-offset embedding

$$
\begin{array} { r } { e _ { h } = \widetilde { W } _ { 2 } \sigma _ { \mathrm { G E L U } } \Big ( \widetilde { W } _ { 1 } \mathbf { L N } ( \gamma ( h ) ) + \widetilde { b } _ { 1 } \Big ) + \widetilde { b } _ { 2 } \in \mathbb { R } ^ { d _ { \mathrm { l a t } } } , } \end{array}\tag{S35}
$$

where $\widetilde { W } _ { 1 } \in \mathbb { R } ^ { 2 d _ { \mathrm { l a t } } \times 3 4 } , \widetilde { W } _ { 2 } \in \mathbb { R } ^ { d _ { \mathrm { l a t } } \times 2 d _ { \mathrm { l a t } } } , \widetilde { b } _ { 1 } \in \mathbb { R } ^ { 2 d _ { \mathrm { l a t } } }$ , and $\widetilde { b } _ { 2 } \in \mathbb { R } ^ { d _ { \mathrm { l a t } } }$ . Here, layer normalization is applied to the 34-component input feature vector. All parameters of this map are shared across requested offsets.

Query initialization. For each requested pair $\left( i _ { m } , h _ { m } \right)$ , the initial query is

$$
\begin{array} { r } { q _ { m } ^ { ( 0 ) } = \mathrm { L N } \Bigl ( e _ { i _ { m } } ^ { \mathrm { g r i d } } + e _ { i _ { m } } ^ { \mathrm { s p a c e } } + e _ { h _ { m } } + e _ { \mathrm { p r e d } } \Bigr ) . } \end{array}\tag{S36}
$$

The vector $e _ { i } ^ { \mathrm { g r i d } } \in \mathbb { R } ^ { d _ { \mathrm { l a t } } }$ is a learned embedding specific to position i on the $1 6 \times 1 6$ latent grid, whereas $e _ { i } ^ { \mathrm { s p a c e } }$ is the fixed sinusoidal spatial embedding defined in Section S2.2. The learned vector $e _ { \mathrm { p r e d } } \in \mathbb { R } ^ { d _ { \mathrm { l a t } } }$ is shared across samples, source systems, spatial positions, and requested offsets. These initial queries encode the requested locations and offsets; context information enters through the attention and diffusion-inspired operations described below.

The initial queries $q _ { m } ^ { ( 0 ) }$ pass through eight successive blocks, each performing the complete sequence of attention, diffusion, reaction, and feedforward updates. Block $\ell \in \{ 0 , \ldots , 7 \}$ takes $q _ { m } ^ { ( \ell ) }$ as input and produces $\overset { \vartriangle } { q _ { m } ^ { ( \ell + 1 ) } }$

Query and context attention. Each block first applies self-attention among the M queries, allowing requests at different spatial positions and time offsets to exchange information. It then applies context cross-attention: the current query vectors provide the queries, and the $S = 2 5 6$ context tokens in $Z _ { t } ^ { c }$ provide the keys and values. Thus, each requested vector can incorporate information from all spatial positions in the encoded context. Both attention modules use eight heads of dimension 64. Each head uses its own learned query, key, and value projections. The eight head outputs are concatenated and mapped back to $\mathbb { R } ^ { d _ { \mathrm { l a t } } }$ by a learned output projection. Both attention modules use pre-normalization and residual connections.

Diffusion-inspired and reaction-inspired updates. For spatial patch i, define the nearest-neighbor difference of the encoded context tokens b

$$
\Delta _ { \mathrm { l a t } } z _ { t , i } ^ { c } = \frac { 1 } { 4 } \sum _ { j \in N ( i ) } \left( z _ { t , j } ^ { c } - z _ { t , i } ^ { c } \right) ,\tag{S37}
$$

where $N ( i )$ contains the four nearest neighbors of patch i on the periodic $1 6 \times 1 6$ latent grid.

For the m-th requested pair, let $\tau _ { m } = h _ { m } / 8$ . The diffusion-inspired branch in block ℓ produces

$$
\begin{array} { r } { d _ { m } ^ { ( \ell ) } = F _ { \mathrm { d i f f } } ^ { ( \ell ) } \left( \left[ \boldsymbol z _ { t , i _ { m } } ^ { c } , \Delta _ { \mathrm { l a t } } \boldsymbol z _ { t , i _ { m } } ^ { c } , \boldsymbol e _ { h _ { m } } , \tau _ { m } \right] \right) , } \end{array}\tag{S38}
$$

Let $\bar { q } _ { m } ^ { ( \ell ) }$ denote the query after the self-attention and context cross-attention updates in block ℓ. The diffusion correction $g _ { \mathrm { d } } ^ { ( \ell ) } d _ { m } ^ { ( \ell ) }$ is first added to this query. The reaction-inspired branch then produces

$$
r _ { m } ^ { ( \ell ) } = F _ { \mathrm { r e a c t } } ^ { ( \ell ) } \biggl ( \left[ \mathrm { L N } \Bigl ( \bar { q } _ { m } ^ { ( \ell ) } + g _ { \mathrm { d } } ^ { ( \ell ) } d _ { m } ^ { ( \ell ) } \Bigr ) , e _ { h _ { m } } \right] \biggr ) .\tag{S39}
$$

Here, brackets denote concatenation into a single input vector. After both corrections, the query is

$$
u _ { m } ^ { ( \ell ) } = \bar { q } _ { m } ^ { ( \ell ) } + g _ { \mathrm { d } } ^ { ( \ell ) } d _ { m } ^ { ( \ell ) } + g _ { \mathrm { r } } ^ { ( \ell ) } r _ { m } ^ { ( \ell ) } .\tag{S40}
$$

A residual feedforward update then completes the block:

$$
\begin{array} { r } { { q } _ { m } ^ { ( \ell + 1 ) } = { u } _ { m } ^ { ( \ell ) } + { F } _ { \mathrm { o u t } } ^ { ( \ell ) } \left( \mathrm { L N } \left( u _ { m } ^ { ( \ell ) } \right) \right) , \qquad m = 1 , \dots , M . } \end{array}\tag{S41}
$$

Each of $F _ { \mathrm { d i f f } } ^ { ( \ell ) } , F _ { \mathrm { r e a c t } } ^ { ( \ell ) }$ , and $F _ { \mathrm { o u t } } ^ { ( \ell ) }$ uses two affine maps separated by GELU, with output dimension $d _ { \mathrm { l a t } }$ . Their input dimensions are $3 d _ { \mathrm { l a t } } + 1 , \widehat { 2 d } _ { \mathrm { l a t } }$ , and $d _ { \mathrm { l a t } }$ , respectively, and their intermediate dimensions are $2 d _ { \mathrm { l a t } } , 4 d _ { \mathrm { l a t } }$ , and $4 d _ { \mathrm { l a t } }$

Both $F _ { \mathrm { d i f f } } ^ { ( \ell ) }$ and $F _ { \mathrm { r e a c t } } ^ { ( \ell ) }$ begin with layer normalization of their complete concatenated inputs. The reaction branch also normalizes the query before concatenating it with $e _ { h _ { m } }$ , as shown explicitly in Equation (S39). Its two normalization layers therefore act on vectors of dimensions $d _ { \mathrm { l a t } }$ and $2 d _ { \mathrm { l a t } }$ , respectively, and have separate parameters.

The scalar weights $g _ { \mathrm { d } } ^ { ( \ell ) }$ and $g _ { \mathrm { r } } ^ { ( \ell ) }$ are shared across queries within each block, are unconstrained, and are initialized to 0.25. All eight blocks have separate attention, normalization, MLP, and scalar-weight parameters.

After the eighth block, the predictor returns

$$
\begin{array} { r } { P _ { \phi } \left( Z _ { t } ^ { c } , \mathcal { I } \right) = \left[ \widehat { z } _ { 1 } , \ldots , \widehat { z } _ { M } \right] ^ { \mathsf { T } } \in \mathbb { R } ^ { M \times d _ { \mathrm { l a t } } } , \qquad \widehat { z } _ { m } = \mathbf { L N } \left( q _ { m } ^ { ( 8 ) } \right) . } \end{array}\tag{S42}
$$

The predicted and target vectors are normalized when computing the predictive latent objective described in Section S2.5.

The two structured branches operate on learned latent representations rather than physical field values. They introduce neighborhood-based spatial differences and query-wise nonlinear updates, but are not numerical discretizations of the physical diffusion and reaction operators. In particular, the reaction-inspired map acts separately on each query, although that query already contains information from other positions and offsets through attention. The same periodic neighbor sets $N ( i )$ are used in all experiments, including the Barkley experiments with non-periodic physical boundary conditions.

## S2.5 Predictive latent objective and pretraining

Predictive latent objective. Before computing the loss, each predicted and target vector is normalized independently across its latent feature components by subtracting its mean and dividing by the square root of its variance plus $1 0 ^ { - 5 }$ . For a batch of B samples, each containing $M = 2 5 6$ requested patch-offset pairs, the predictive latent objective is

$$
\mathcal { L } _ { \mathrm { J E P A } } = \frac { 1 } { B M d _ { \mathrm { l a t } } } \sum _ { b = 1 } ^ { B } \sum _ { m = 1 } ^ { M } \left| \left| L N ( \widehat { z } _ { b , m } ) - L N ( z _ { b , m } ^ { \mathrm { t a r } } ) \right| \right| _ { 2 } ^ { 2 } .\tag{S43}
$$

Here, $\widehat { z } _ { b , m }$ and $z _ { b , m } ^ { \mathrm { t a r } }$ are the predicted and target vectors for the m-th query in sample b. The loss averages the squared differences over samples, query positions, and latent feature components. The index m runs over positions in the ordered query list, so repeated patch–offset pairs contribute separately.

Optimization and EMA updates. Gradients of ${ \mathcal { L } } _ { \mathrm { J E P A } }$ propagate through the normalization of the predicted vectors and update the online-encoder parameters θ and predictor parameters $\phi$ . The target vectors are treated as constants when differentiating the loss; the target encoder receives neither gradient nor optimizer updates. After each optimizer step, the target-encoder parameters $\xi$ are updated by the EMA rule in Equation (S33), using the newly updated online-encoder parameters θ.

Predictive pretraining runs for 200,000 optimization iterations. The EMA coefficient $m _ { s }$ increases from 0.996 to 0.99995 according to a cosine schedule over these iterations. The complete architecture and optimization settings are summarized in Tables S3 and S4.

## S2.6 Dense forecasting and adaptation from few trajectories

Prediction on the full grid. During downstream adaptation, the target encoder is discarded, and the pretrained online encoder parameters θ are held fixed. The pretrained predictor parameters φ are fine-tuned jointly with the decoder parameters ψ, which are initialized randomly. Context masking is disabled.

For a forecast offset $h ,$ the predictor receives the complete query list $\mathcal { I } _ { h } ^ { \mathrm { f u l l } }$ defined in Section S2.1 and produces

$$
\widehat { Z } _ { t + h } = P _ { \phi } \left( E _ { \theta } ( X _ { t } ^ { c } ) , \mathcal { I } _ { h } ^ { \mathrm { f u l l } } \right) \in \mathbb { R } ^ { S \times d _ { \mathrm { l a t } } } .\tag{S44}
$$

The decoder combines this predicted latent grid with the latest context field to produce

$$
\widehat { x } _ { t + h } = D _ { \psi } \Big ( \widehat { Z } _ { t + h } , x _ { t } \Big ) \in \mathbb { R } ^ { H \times W \times C } , \qquad h \in \{ 1 , 2 , 3 , 4 , 5 \} .\tag{S45}
$$

For each offset, the predictor processes all $S = 2 5 6$ spatial queries jointly. Different offsets are evaluated using separate query lists, with the same encoder, predictor, and decoder parameters. All five future fields are predicted directly from the same four-frame context; predicted fields are not fed back into the model.

The decoder is a U-shaped convolutional network. It combines the predicted $1 6 \times 1 6$ latent grid with a multiscale feature pyramid extracted from $x _ { t }$ and reconstructs a two-channel field on the original 128 × 128 grid.

Features extracted from the latest context field. The decoder constructs four feature arrays from $x _ { t } \mathrm { : }$

<table><tr><td>array</td><td> $F _ { 0 }$ </td><td> $F _ { 1 }$ </td><td> $F _ { 2 }$ </td><td> $F _ { 3 }$ </td></tr><tr><td>spatial resolution</td><td>128 × 128 64 × 64 32 × 32 16 × 16 .</td><td></td><td></td><td></td></tr><tr><td>feature width</td><td>96</td><td>192</td><td>384</td><td>384</td></tr></table>

A residual convolutional block maps the two-channel input field to $F _ { 0 }$ . For $k = 1 , 2 , 3$ , a 4 × 4 convolution with stride 2 and zero padding of width 1, followed by a residual convolutional block, maps $F _ { k - 1 } : 0 ~ F _ { k }$ . This feature pyramid depends only on $x _ { t }$ and is shared across the five forecast offsets.

Each residual block contains a branch with two successive $3 \times 3$ convolutions, each followed by group normalization and the SiLU activation. The output of this branch is added to the block input. When the input and output feature widths differ, a $1 \times 1$ convolution is applied to the skip path before the addition.

Group normalization divides the feature channels into eight groups. For each sample, normalization statistics are computed over the channels and spatial positions within each group, followed by learned per-channel scales and shifts. The SiLU activation is $\begin{array} { r } { \mathrm { S i L U } ( s ) = \frac { s } { 1 + \exp ( - s ) } } \end{array}$ . Unless otherwise stated, all $3 \times 3$ decoder convolutions use stride 1 and zero padding of width 1, preserving the spatial resolution.

Recovering the field on the original grid. Each predicted token in $\widehat { Z } _ { t + h }$ is layer-normalized across its feature components and mapped from $d _ { \mathrm { l a t } } = 5 1 2$ to 384 components by a learned affine map shared across spatial positions and forecast offsets. The resulting $1 6 \times 1 6$ feature array is concatenated with $F _ { 3 }$ along the feature dimension. A residual block maps these 768-component features to 384 components.

Three successive decoding stages then double the spatial resolution by bilinear interpolation. At resolutions $3 2 \times 3 2$ $6 4 \times 6 4 .$ , and $1 2 8 \times 1 2 8$ , the interpolated decoder features are concatenated with $F _ { 2 } , F _ { 1 }$ , and $F _ { 0 }$ , respectively. The concatenated feature widths are 768, 576, and 288; residual blocks reduce them to 384, 192, and $^ { 9 6 , }$ respectively.

A final $3 \times 3$ convolution preserves the width 96 and is followed by group normalization and SiLU. A 1 × 1 convolution then produces the two components of the predicted future field in standardized coordinates.

Standardized and original field coordinates. For each system, the mean $\mu _ { j }$ and standard deviation $\sigma _ { j }$ of physical component $j \in \{ 1 , 2 \}$ are computed over all trajectories, time frames, and spatial positions in its pretraining dataset. The same statistics are used to standardize context fields and training targets. The standardization and inverse transformation are

$$
\sigma _ { j } ^ { * } = \operatorname* { m a x } ( \sigma _ { j } , 1 0 ^ { - 6 } ) . \qquad x _ { t , j } = \frac { x _ { t , j } ^ { \mathrm { o r i g } } - \mu _ { j } } { \sigma _ { j } ^ { * } } , \qquad \widehat { x } _ { t + h , j } ^ { \mathrm { o r i g } } = \sigma _ { j } ^ { * } \widehat { x } _ { t + h , j } + \mu _ { j }\tag{S46}
$$

The superscript orig denotes the original field scale. The same constants are used for context fields and training targets. Training losses are computed on standardized fields, whereas reported forecasting errors are computed on the original scale.

Adaptation protocol. A separate predictor-decoder pair is adapted for each system, trajectory budget K, and support-set selection, where the support set consists of the K training trajectories available for adaptation. For source systems included in pretraining, $K \in \{ 5 , 1 0 , 2 0 \}$ . For systems excluded from pretraining, $K \in \{ 1 , 5 , 1 0 \}$ . Each RD-JEPA adaptation run in the primary benchmark uses 5,000 optimization steps.

Downstream objective. The downstream objective gives equal weight to the five forecast offsets:

$$
\mathcal { L } _ { \mathrm { d o w n } } = \frac { 1 } { 5 } \sum _ { h = 1 } ^ { 5 } \left( \mathcal { L } _ { \mathrm { M S E } } ^ { ( h ) } + 0 . 5 0 \mathcal { L } _ { \mathrm { r e l } } ^ { ( h ) } + 0 . 1 0 \mathcal { L } _ { \nabla } ^ { ( h ) } + 0 . 0 5 \mathcal { L } _ { \mathrm { F F T } } ^ { ( h ) } \right) .\tag{S47}
$$

For a fixed offset, let $\boldsymbol { u } = ( u _ { b } ) _ { b = 1 } ^ { B }$ and $\widehat { \boldsymbol { u } } = ( \widehat { \boldsymbol { u } } _ { b } ) _ { b = 1 } ^ { B }$ denote a batch of standardized target and predicted fields, respectively, with $u _ { b } , \widehat { u } _ { b } \in \mathbb { R } ^ { H \times W \times C }$ . Suppressing the offset superscript, the four loss terms are

$$
\mathcal { L } _ { \mathrm { M S E } } = \left. ( \widehat { u } - u ) ^ { 2 } \right. ,\tag{S48}
$$

$$
\mathcal { L } _ { \mathrm { r e l } } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \frac { { \lVert \widehat { u } _ { b } - u _ { b } \rVert } _ { 2 } } { \operatorname* { m a x } ( { \lVert u _ { b } \rVert } _ { 2 } , 1 0 ^ { - 8 } ) } ,\tag{S49}
$$

$$
\begin{array} { r } { \mathcal { L } _ { \nabla } = \langle | \delta _ { x } \widehat { u } - \delta _ { x } u | \rangle + \langle | \delta _ { y } \widehat { u } - \delta _ { y } u | \rangle , } \end{array}\tag{S50}
$$

$$
\mathcal { L } _ { \mathrm { F F T } } = \left. \left| \log ( 1 + \left| \mathcal { F } ( \widehat { u } ) \right| ) - \log ( 1 + \left| \mathcal { F } ( u ) \right| ) \right| \right. .\tag{S51}
$$

Here, ⟨·⟩ denotes the mean over all entries of its argument, including the batch and both field components. Squares and absolute values are applied entrywise. The norm $\| \cdot \| _ { 2 }$ is the Euclidean norm over all spatial values and both components of one sample.

The operators $\delta _ { x }$ and $\delta _ { y }$ take adjacent first differences along the two spatial directions. They do not divide by the grid spacing and do not include wrap-around differences at the boundaries. The directional averages in Equation (S50) are computed separately and then added.

The transform $\mathcal { F }$ is a two-dimensional real-input Fourier transform applied independently to each field component, with coefficient normalization $1 / { \sqrt { H W } }$ . For $H = W = 1 2 8$ , it returns $1 2 8 \times 6 5$ complex coefficients, retaining the nonnegative frequencies in the second spatial direction. All returned Fourier magnitudes receive equal weight in Equation (S51); no additional multiplicity weights are applied to account for omitted conjugate frequencies.

The no-predictive-latent control uses the same downstream objective. Objectives and optimization settings for the external supervised baselines are specified separately in Supplementary Section S3.3 and Table S4.

## S2.7 Architecture and optimization summary

Table S3 summarizes the RD-JEPA architecture and parameter counts. Table S4 lists the optimization settings for predictive pretraining and downstream adaptation.

Table S3. RD-JEPA input and output dimensions and parameter counts. Shapes omit the batch dimension. The decoder output corresponds to one forecast offset. Parameter counts are reported in millions (M). Totals are computed before rounding the displayed component counts. The pretraining total includes the EMA target encoder.
<table><tr><td>Component</td><td>Input</td><td>Output</td></tr><tr><td>Online encoder</td><td> $4 \times 1 2 8 \times 1 2 8 \times 2$ </td><td> $2 5 6 \times 5 1 2$ </td></tr><tr><td>Target encoder</td><td> $1 \times 1 2 8 \times 1 2 8 \times 2$ </td><td> $2 5 6 \times 5 1 2$ </td></tr><tr><td>Latent predictor</td><td> $2 5 6 \times 5 1 2$  context tokens and 256 query pairs (i,h)</td><td> $2 5 6 \times 5 1 2$  predicted tokens</td></tr><tr><td>Forecast decoder</td><td> $2 5 6 \times 5 1 2$  predicted tokens and the latest  $1 2 8 \times 1 2 8 \times 2$  context field</td><td> $1 2 8 \times 1 2 8 \times 2$ </td></tr><tr><td colspan="3">Parameter count</td></tr><tr><td colspan="2">Online encoder parameters</td><td></td></tr><tr><td colspan="2">EMA target encoder parameters</td><td>110.48M</td></tr><tr><td colspan="2">Latent predictor parameters</td><td>110.48M 76.37M</td></tr><tr><td colspan="2">Forecast decoder parameters</td><td>20.54M</td></tr><tr><td colspan="2">Total pretraining parameters</td><td>297.33M: online encoder, predictor, and EMA target encoder</td></tr><tr><td colspan="2">Trainable parameters during pretraining</td><td>186.85M: online encoder and predictor</td></tr><tr><td colspan="2">Trainable parameters during adaptation</td><td>96.90M: predictor and decoder</td></tr></table>

Table S4. Optimization settings for predictive pretraining and downstream adaptation. Each training iteration uses one batch. The downstream settings apply to RD-JEPA adaptation runs in the primary benchmark, for both source systems and systems excluded from pretraining.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Shared settings</td><td></td></tr><tr><td>Optimizer</td><td>AdamW,  $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 )$ </td></tr><tr><td>Batch size</td><td>4</td></tr><tr><td>Gradient clipping</td><td>Global norm 1.0</td></tr><tr><td>Predictive pretraining</td><td></td></tr><tr><td>Trainable modules</td><td>Online encoder and predictor</td></tr><tr><td>Encoder updates</td><td>Online encoder: optimizer; target encoder: EMA</td></tr><tr><td>Training iterations</td><td>200,000</td></tr><tr><td>Learning rate</td><td> $7 \times 1 0 ^ { - 5 }$  peak;  $1 0 ^ { - 6 }$  minimum</td></tr><tr><td>Learning rate schedule</td><td>Linear warmup over the first 5,000 iterations, followed by cosine decay</td></tr><tr><td>Weight decay</td><td>0.05</td></tr><tr><td>Downstream adaptation</td><td></td></tr><tr><td>Trainable modules</td><td>Predictor and decoder</td></tr><tr><td>Encoder updates</td><td>Online encoder: frozen; target encoder: not used</td></tr><tr><td>Training iterations</td><td>5,000</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 5 }$  predictor;  $2 \times 1 0 ^ { - 4 }$  decoder</td></tr><tr><td>Learning rate schedule</td><td>Constant learning rates</td></tr><tr><td>Weight decay</td><td> $1 0 ^ { - 4 }$ </td></tr></table>

## Supplementary Note 3: Experimental protocols, baselines and statistics

## S3.1 Forecasting tasks, support selection and evaluation

Each forecasting task is defined by four consecutive observed two-channel fields and five target fields at lead times $h \in \{ 1 , \ldots , 5 \}$ measured in stored-frame intervals. Predictions are obtained directly, without using earlier predicted fields as inputs. The nopredictive-latent control uses only the last of the four observed fields and the requested lead time, as described in Supplementary Section S3.2. For each system, the selected K training trajectories form the support set. We use $K \in \{ 5 , 1 0 , 2 0 \}$ for the five systems included in pretraining and $K \in \{ 1 , 5 , 1 0 \}$ for Lambda-Omega, Barkley, and Oregonator, which were excluded from pretraining. A separate model is trained for each system, value of K, and support-set selection.

Three prespecified random seeds, 777, 778, and 779, are used to permute the trajectory indices in each primary-benchmark adaptation pool. For each seed and value of K, the first K trajectories in the permutation form the support set; thus, each smaller support set is contained in the larger sets obtained with the same seed. All methods use the same selected trajectories for a given system, seed, and value of K. For each method, system, and value of K, the three runs vary only the support-set selection; the model-training seed remains fixed at 777. All RD-JEPA runs start from the same pretrained parameter values. The data partitions and test trajectories remain fixed across runs.

For each system and physical field component, all methods use the same fixed affine transformation, consisting of a scaling and a shift, as defined in Equation (S46). Its coefficients are estimated without using test trajectories. Errors are computed from predictions and reference fields on the original field scale.

For each source system, evaluation uses 300 test trajectories: 100 in-distribution, 100 coefficient-out-of-distribution, and 100 initial-condition-out-of-distribution trajectories. For each system excluded from pretraining, all runs use the same 300 trajectories from that system’s in-distribution test set. Every valid temporal window is evaluated. With 40 stored frames, four context frames, and a maximum lead time of five, each trajectory contributes $4 0 - 4 - 5 + 1 = 3 2$ windows. All methods, including the no-predictive-latent control, are evaluated on these same windows at the five prescribed lead times. In both primary benchmarks, each model is trained for 5,000 parameter updates.

In the boundary-condition experiment, support sets containing $K \in \{ 1 , 5 , 1 0 \}$ trajectories are selected separately for each boundary condition from a pool of 20 adaptation trajectories. Three distinct selections are obtained using random seeds 777, 778, and 780. Seed 779 is excluded because, under the fixed selection procedure, it selects the same K = 1 support trajectory as seed 778. The model-training seed remains fixed at 777. Within each boundary condition and support-set selection, all models use the same selected training trajectories and are evaluated on the same fixed set of 300 test trajectories. The number of training updates used by each model in this boundary-condition experiment is reported in Supplementary Note 6.

## S3.2 Comparison models and controls

The comparison includes spectral neural operators, a rotationally equivariant transformer, a convolutional encoder–decoder, and two controls. Each supervised baseline maps four two-channel context fields directly to five future fields and is trained separately for every equation, value of K, and support-set selection.

Fourier Neural Operator.<sup>11</sup> FNO uses six Fourier layers of width 192 with 16 × 16 retained modes, spatial coordinates, and a joint five-horizon output head.

Laplace Neural Operator.<sup>41</sup> LNO uses width 48, one pole-residue layer, and four retained pole modes along each spatial direction.

Riesz Neural Operator.<sup>43</sup> RieszNO uses width 36, one spectral layer, 8 × 8 retained modes, spatial coordinates, and a 128-dimensional readout.

ReViT.<sup>42</sup> The ReViT configuration uses feature widths ranging from 192 to 768, stage depths (2,4, 8,4,2), 16 attention heads, window size 8, and a direct five-horizon decoder. Local spatial gradients define the reference vectors used to construct invariant tokens.

ConvNeXt U-Net.<sup>24</sup> CNextU-Net uses four encoder–decoder scales, two ConvNeXt blocks per scale, one bottleneck block and widths 42 → 84 → 168 → 336 → 672.

No-predictive-latent control. This model is trained independently with the JEPA encoder–predictor pathway removed. It retains the dense decoder and full-resolution last-frame pathway used by RD-JEPA, receives a continuous lead-time embedding and does not construct a sample-specific predictive latent grid. The comparison evaluates the contribution of the complete sample-dependent predictive-latent pathway.

Table S5. Downstream optimization settings for the primary benchmark. The external-baseline column applies to FNO, LNO, RieszNO, ReViT and CNextU-Net.
<table><tr><td>Setting</td><td>RD-JEPA</td><td>No predictive latent</td><td>External baselines</td></tr><tr><td>Training updates</td><td>5,000</td><td>5,000</td><td>5,000</td></tr><tr><td>Batch size</td><td>4</td><td>4</td><td></td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 5 } \mathrm { p r e d i c t o r } ; 2 \times 1 0 ^ { - 4 }$  decoder</td><td> $1 0 ^ { - 4 }$  conditioner;  $2 \times 1 0 ^ { - 4 } \mathrm { d e } .$  coder</td><td> $2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Weight decay</td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Adam betas</td><td>(0.9,0.95)</td><td>(0.9,0.95)</td><td>(0.9,0.95)</td></tr><tr><td>Gradient clipping</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Weight of  ${ \mathcal L } _ { \mathrm { r e l } }$ </td><td>0.50</td><td>0.50</td><td>0.50</td></tr><tr><td>Weight of  $\mathcal { L } _ { \nabla }$ </td><td>0.10</td><td>0.10</td><td>0.10</td></tr><tr><td>Weight of  ${ \mathcal { L } } _ { \mathrm { F F T } }$ </td><td>0.05</td><td>0.05</td><td>0.05</td></tr><tr><td>Support selection and fixed training seed</td><td colspan="3">Primary benchmark support-selection seeds: 777, 778 and 779; model-training seed: 777 for every support selection; one fixed RD-JEPA predictive-pretraining checkpoint (pretraining seed 1234)</td></tr></table>

Architecture-matched model trained from scratch. To compare source-pretrained adaptation with end-to-end supervised training from scratch, this control uses the same encoder-predictor-decoder architecture as RD-JEPA but initializes all three modules randomly. It is trained on Lambda–Omega, Barkley, and Oregonator, the three systems excluded from pretraining. Unlike RD-JEPA adaptation, which freezes the online encoder, this control jointly updates the encoder, predictor, and decoder.

The control uses the same support trajectories, direct forecast horizons, downstream objective, update budget, and fixed model-training random-number-generator state as RD-JEPA. No pretrained tensor, EMA target branch, or JEPA objective is used during this training. The model is trained for 5,000 updates with AdamW, using learning rates of $1 0 ^ { - 4 }$ for the encoder and predictor and $2 \times 1 0 ^ { - 4 }$ for the decoder. The final update checkpoint is evaluated; neither validation nor test results are used to select the checkpoint.

## S3.3 Training objectives and optimization settings

All models, including RD-JEPA, the no-predictive-latent control, the architecture-matched model trained from random initialization, and the five comparison models, use the loss in eq. (S47). The weights of $\mathcal { L } _ { \mathrm { M S E } } , \mathcal { L } _ { \mathrm { r e l } } , \mathcal { L } _ { \nabla }$ and ${ \mathcal { L } } _ { \mathrm { F F T } }$ are 1, 0.50, 0.10 and 0.05, respectively, for every model.

The losses at the five forecast lead times receive equal weight. In RD-JEPA, the pretrained predictor and the newly initialized decoder use different learning rates. The five comparison models are trained from random initialization using the same optimizer settings. These settings are listed in table S5; the settings for the architecture-matched model trained from random initialization are given in S3.2.

All RD-JEPA runs start from the same pretrained parameter values, obtained with pretraining seed 1234. For each model, the random-number generator used for training is initialized with seed 777 at the start of every run. All models use the same selected training trajectories within each support-set selection. Across the three selections, the support trajectories change while the pretraining and model-training seeds remain fixed. The reported variation therefore measures sensitivity to support-set selection; it does not assess variability across independently chosen training or pretraining seeds.

For every model in the primary benchmark, evaluation uses the parameter values obtained after 5,000 training updates. Training does not use early stopping, and validation results are not used to select the parameter values evaluated.

## S3.4 Evaluation metrics and statistical aggregation

All errors are evaluated in the original physical field units. For prediction $\widehat { \mathbf { x } } _ { i } ^ { ( h ) }$ and reference field $\mathbf { x } _ { i } ^ { ( h ) }$ , the relative field error is

$$
E _ { \mathrm { r e l } L ^ { 2 } , i } ^ { ( h ) } = \frac { \left\| \widehat { \mathbf { x } } _ { i } ^ { ( h ) } - \mathbf { x } _ { i } ^ { ( h ) } \right\| _ { 2 } } { \left\| \mathbf { x } _ { i } ^ { ( h ) } \right\| _ { 2 } + \varepsilon } , \qquad \varepsilon = 1 0 ^ { - 8 } ,\tag{S52}
$$

and the spatial-gradient error is

$$
\begin{array} { r } { E _ { \nabla L ^ { 1 } , i } ^ { ( h ) } = \mathbf { M A E } \Big ( \delta _ { x } \widehat { \mathbf { x } } _ { i } ^ { ( h ) } , \delta _ { x } \mathbf { x } _ { i } ^ { ( h ) } \Big ) + \mathbf { M A E } \Big ( \delta _ { y } \widehat { \mathbf { x } } _ { i } ^ { ( h ) } , \delta _ { y } \mathbf { x } _ { i } ^ { ( h ) } \Big ) . } \end{array}\tag{S53}
$$

Here, $\delta _ { x }$ and $\delta _ { y }$ are adjacent first differences along the two grid axes, without a wrap-around term. Each mean absolute error is averaged over both field channels and all valid neighbouring grid pairs. The same definition is used for periodic and

non-periodic datasets. Relative $L ^ { 2 }$ and gradient $L ^ { 1 }$ are the principal reported metrics. MSE is retained as a secondary diagnostic in Supplementary Data 1–3.

For support-selection run r, errors are first averaged over all valid windows within each test trajectory. Let $e _ { p , m , K , h , i , w } ^ { ( r ) }$ denote the error for window w of trajectory i. The trajectory-level value is

$$
E _ { p , m , K , h , i } ^ { ( r ) } = \frac { 1 } { W _ { i } } \sum _ { w = 1 } ^ { W _ { i } } e _ { p , m , K , h , i , w } ^ { ( r ) } , \qquad W _ { i } = 3 2 .\tag{S54}
$$

The run-level mean for equation p, model m, trajectory budget K and horizon h is

$$
\overline { { E } } _ { p , m , K , h } ^ { ( r ) } = \frac { 1 } { N _ { p } } \sum _ { i = 1 } ^ { N _ { p } } E _ { p , m , K , h , i } ^ { ( r ) } , \qquad N _ { p } = 3 0 0 .\tag{S55}
$$

The horizon-averaged value for each repeat is

$$
\overline { { E } } _ { p , m , K } ^ { ( r ) } = \frac { 1 } { 5 } \sum _ { h = 1 } ^ { 5 } \overline { { E } } _ { p , m , K , h } ^ { ( r ) } .\tag{S56}
$$

For every source- and held-out-equation analysis, $R = 3$ denotes the three support-set selections. The final mean and sample standard deviation are computed across the R run-level means,

$$
\mu _ { p , m , K , h } = \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \overline { { E } } _ { p , m , K , h } ^ { ( r ) } , \qquad s _ { p , m , K , h } = \sqrt { \frac { 1 } { R - 1 } \sum _ { r = 1 } ^ { R } \left( \overline { { E } } _ { p , m , K , h } ^ { ( r ) } - \mu _ { p , m , K , h } \right) ^ { 2 } } .\tag{S57}
$$

Thus, $n = 3$ denotes the three support-set selections. Each run-level value summarizes the same fixed set of 300 test trajectories. The reported sample standard deviation is computed exclusively across the three run-level means and therefore measures support-set sensitivity; heterogeneity across test trajectories is not used as the uncertainty bar. For the split-resolved source analysis, $N _ { p } = 1 0 0$ within each test regime before averaging equations and horizons.

Percentage reductions are paired by support-set selection. For baseline b and metric E, the reduction in run r is

$$
Q _ { p , b , K } ^ { ( r ) } = 1 0 0 \left( 1 - \frac { \overline { { E } } _ { p , \mathrm { R D - J E P A } , K } ^ { ( r ) } } { \overline { { E } } _ { p , b , K } ^ { ( r ) } } \right) ,\tag{S58}
$$

and displayed centres and error bars are the mean and sample standard deviation of the three paired values $Q _ { p , b , K } ^ { ( r ) }$ . This is a mean of matched run-level reductions, not a ratio formed after pooling the runs. Supplementary Data 1 contains the source-equation run-level values, Supplementary Data 2 contains the held-out-equation horizon-resolved run-level values, and Supplementary Data 3 contains the broad-comparison horizon-averaged run-level values. Supplementary Data 4 contains the architecture-matched scratch run-level values, and Supplementary Data 5 contains the boundary-condition stress-test run-level values.

## Supplementary Note 4: Complete forecasting results on source systems

## S4.1 Evaluation settings and error averaging

We compare RD-JEPA, the no-predictive-latent control and FNO on the five source systems, using $K \in \{ 5 , 1 0 , 2 0 \}$ training trajectories and forecast lead times $h = 1 , \ldots , 5 .$ . For each support-set selection, errors are first averaged over valid windows within each trajectory and then over the same 300 test trajectories, comprising 100 trajectories from each of the three test regimes. The reported means and sample standard deviations are computed across the three support-set selections. Metric definitions and averaging procedures are given in Supplementary Note 3, and the values for individual selections are provided in Supplementary Data 1.

After combining the three test regimes, RD-JEPA has the lowest mean error among the three models on both reported metrics for all $5 \times 3 \times 5 = 7 5$ combinations of source system, support-set size and forecast lead time.

## S4.2 Complete relative-field-error results

The relative $L ^ { 2 }$ errors defined in eq. (S52) are reported for K = 5, 10 and 20 in tables S6 to S8, respectively. Each table gives the error at each forecast lead time and its average over $h = 1 , \ldots , 5$

Table S6. Complete relative $L ^ { 2 }$ results on the five source systems at K = 5. Entries are 100× the error and report the mean ± sample s.d. across three paired support-set selections. Within each run, errors are first averaged over all valid windows within each trajectory and then over the same 300 test trajectories (100 from each test regime). The Avg. column averages $h = 1 , \ldots , 5$ within each run before across-run aggregation. Lower values are better. Boldface marks the lowest numerical mean within each equation and horizon; no formal hypothesis tests were performed.
<table><tr><td>PDE</td><td>Model</td><td>h= 1</td><td>h = 2</td><td>h= 3</td><td>h=4</td><td>h= 5</td><td>Avg.</td></tr><tr><td rowspan="3">Gray-Scott</td><td>RD-JEPA</td><td> ${ \bf 3 . 5 6 \pm 1 . 2 1 }$ </td><td> $\mathbf { 4 . 6 6 \pm 0 . 9 7 }$ </td><td> ${ \pm } 0 . 7 4 \pm 0 . 8 4$ </td><td> ${ \bf 6 . 8 3 \pm 0 . 7 5 }$ </td><td> ${ \bf 7 . 9 9 \pm 0 . 6 6 }$ </td><td> ${ \pm } \mathbf { 0 . 7 6 } { \pm } \mathbf { 0 . 8 7 }$ </td></tr><tr><td>No predictive latent</td><td> $6 . 5 4 \pm 1 . 6 8$ </td><td> $1 0 . 1 3 \pm 2 . 3 8$ </td><td> $1 3 . 4 0 \pm 2 . 9 7$ </td><td> $1 6 . 3 5 \pm 3 . 1 9$ </td><td> $1 9 . 0 0 \pm 2 . 8 5$ </td><td> $1 3 . 0 8 \pm 2 . 5 8$ </td></tr><tr><td>FNO</td><td> $6 . 1 0 \pm 0 . 4 9$ </td><td> $7 . 3 0 \pm 0 . 2 2$ </td><td> $9 . 3 6 \pm 0 . 5 6$ </td><td> $1 1 . 5 5 \pm 0 . 9 3$ </td><td> $1 3 . 7 1 \pm 1 . 2 9$ </td><td> $9 . 6 0 \pm 0 . 4 6$ </td></tr><tr><td rowspan="3">FitzHugh-Nagumo</td><td>RD-JEPA</td><td> ${ \bf 4 . 1 1 \pm 1 . 3 8 }$ </td><td> ${ \bf 4 . 5 9 \pm 1 . 3 4 }$ </td><td> ${ \pm } \mathrm { { \bf { 5 } } } . 3 6 { \pm } 1 . 2 0 $ </td><td> ${ \bf 6 . 4 3 \pm 1 . 0 8 }$ </td><td> ${ \bf 7 . 7 5 \pm 1 . 0 2 }$ </td><td> ${ \pm } . 6 5 \pm 1 . 1 9$ </td></tr><tr><td>No predictive latent</td><td> $1 2 . 0 0 \pm 3 . 3 0$ </td><td> $1 3 . 3 1 \pm 3 . 4 8$ </td><td> $1 5 . 8 6 \pm 3 . 8 4$ </td><td> $1 9 . 0 9 \pm 3 . 9 4$ </td><td> $2 3 . 0 0 \pm 4 . 2 6$ </td><td> $1 6 . 6 5 \pm 3 . 7 4$ </td></tr><tr><td>FNO</td><td> $2 1 . 9 9 \pm 0 . 6 5$ </td><td> $2 1 . 4 9 \pm 0 . 8 8$ </td><td> $2 1 . 7 9 \pm 1 . 1 0$ </td><td> $2 2 . 6 4 \pm 1 . 3 4$ </td><td> $2 4 . 0 8 \pm 1 . 6 0 $ </td><td> $2 2 . 4 0 \pm 1 . 1 1$ </td></tr><tr><td rowspan="3">Brusselator</td><td>RD-JEPA</td><td> ${ \bf 3 . 7 9 \pm 0 . 4 3 }$ </td><td> $\mathbf { 4 . 8 5 \pm 0 . 8 3 }$ </td><td> ${ \bf 6 . 1 8 \pm 1 . 2 3 }$ </td><td> ${ \bf 7 . 6 1 \pm 1 . 7 2 }$ </td><td> ${ \bf 9 . 9 0 \pm 2 . 5 8 }$ </td><td> ${ \bf 6 . 4 7 \pm 1 . 3 1 }$ </td></tr><tr><td>No predictive latent</td><td> $9 . 5 2 \pm 1 . 1 1$ </td><td> $1 5 . 7 5 \pm 1 . 6 5$ </td><td> $2 1 . 9 4 \pm 2 . 4 8$ </td><td> $2 8 . 3 8 \pm 3 . 6 6$ </td><td> $3 4 . 4 0 \pm 5 . 5 5$ </td><td> $2 2 . 0 0 \pm 2 . 8 5$ </td></tr><tr><td>FNO</td><td> $9 . 0 8 \pm 1 . 1 6$ </td><td> $1 2 . 6 5 \pm 1 . 1 1$ </td><td> $1 7 . 3 9 \pm 1 . 0 2$ </td><td> $2 1 . 2 1 \pm 1 . 2 7$ </td><td> $2 3 . 8 7 \pm 1 . 8 7$ </td><td> $1 6 . 8 4 \pm 0 . 8 6$ </td></tr><tr><td rowspan="3">Ginzburg-Landau</td><td>RD-JEPA</td><td> ${ \bf 9 . 3 8 \pm 4 . 5 8 }$ </td><td> ${ \bf 1 0 . 9 4 \pm 4 . 0 3 }$ </td><td> ${ \pm } \pm \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { \delta } \mathbf { 1 } 2 . \mathbf { 9 0 } \pm \mathbf { 3 . 8 2 }$ </td><td> $\pm 3 . 8 3 \pm 3 . 8 2$ </td><td> ${ \bf 1 6 . 8 8 \pm 3 . 8 0 }$ </td><td> ${ \pm } \pm \mathbf { \delta . 9 9 } \pm \mathbf { 3 . 9 9 }$ </td></tr><tr><td>No predictive latent</td><td> $2 1 . 7 3 \pm 1 6 . 1 4$ </td><td> $3 6 . 6 3 \pm 2 5 . 3 4$ </td><td> $4 9 . 0 9 \pm 3 0 . 3 1$ </td><td> $5 8 . 9 9 \pm 3 1 . 6 1$ </td><td> $6 7 . 5 2 \pm 3 0 . 6 0$ </td><td> $4 6 . 7 9 \pm 2 6 . 7 4$ </td></tr><tr><td>FNO</td><td> $2 5 . 6 2 \pm 3 . 5 3$ </td><td> $2 9 . 6 7 \pm 4 . 9 6$ </td><td> $3 8 . 3 9 \pm 8 . 4 4$ </td><td> $4 8 . 5 0 \pm 1 2 . 0 8$ </td><td> $5 8 . 4 0 \pm 1 5 . 0 3$ </td><td> $4 0 . 1 2 \pm 8 . 6 9$ </td></tr><tr><td rowspan="3">Schnakenberg</td><td>RD-JEPA</td><td> $\pm . 0 3 \pm \mathbf { 0 . 4 } 2$ </td><td> $\pm . 5 2 \pm \mathbf { 0 . 4 7 }$ </td><td> $\mathbf { 3 . 0 8 \pm 0 . 6 4 }$ </td><td> $\mathbf { 3 . 5 5 \pm 0 . 8 1 }$ </td><td> ${ \bf 4 . 3 5 \pm 1 . 0 8 }$ </td><td> ${ \bf 3 . 1 1 \pm 0 . 6 8 }$ </td></tr><tr><td>No predictive latent</td><td> $4 . 3 3 \pm 0 . 7 7$ </td><td> $7 . 0 8 \pm 0 . 6 2$ </td><td> $1 0 . 1 2 \pm 0 . 6 7$ </td><td> $1 3 . 2 6 \pm 0 . 6 2$ </td><td> $1 6 . 6 5 \pm 0 . 7 3$ </td><td> $1 0 . 2 9 \pm 0 . 6 7$ </td></tr><tr><td>FNO</td><td> $7 . 1 0 \pm 0 . 2 0$ </td><td> $8 . 6 2 \pm 0 . 4 5$ </td><td> $1 0 . 7 3 \pm 0 . 5 6$ </td><td> $1 2 . 7 2 \pm 0 . 5 8$ </td><td> $1 4 . 5 2 \pm 0 . 5 3$ </td><td> $1 0 . 7 4 \pm 0 . 4 5$ </td></tr></table>

## S4.3 Errors in spatial first differences

The mean absolute spatial first-difference errors defined in eq. (S53) are reported for K = 5, 10 and 20 in tables S9 to S11, respectively. Each table gives the error at each forecast lead time and its average over $h = 1 , \ldots , 5 .$

## S4.4 Split-resolved forecasting results

The main source-system analysis pools the in-distribution, coefficient-OOD and initial-condition-OOD trajectories. Table S12 reports each split separately after equal-weight averaging over the five equations and five horizons. The three run-level values underlying every split summary are available in Supplementary Data 1.

Table S7. Complete relative $L ^ { 2 }$ results on the five source systems at $K = 1 0 .$ . Entries are 100× the error and report the mean ± sample s.d. across three paired support-set selections. Within each run, errors are first averaged over all valid windows within each trajectory and then over the same 300 test trajectories (100 from each test regime). The Avg. column averages $h = 1 , \ldots , 5$ within each run before across-run aggregation. Lower values are better. Boldface marks the lowest numerical mean within each equation and horizon; no formal hypothesis tests were performed.
<table><tr><td>PDE</td><td>Model</td><td>h=1</td><td> $h = 2$ </td><td> $h = 3$ </td><td> $h = 4$ </td><td> $h = 5$ </td><td>Avg.</td></tr><tr><td rowspan="3">Gray-Scott</td><td>RD-JEPA</td><td> $\mathbf { 2 . 0 8 \pm 0 . 8 8 }$ </td><td> $\pm . 9 6 \pm 1 . 0 7$ </td><td> ${ \bf 3 . 9 1 \pm 1 . 2 5 }$ </td><td> ${ \bf 4 . 8 5 \pm 1 . 3 8 }$ </td><td> ${ \pm } \mathrm { { \bf { 5 . 8 4 \pm 1 . 3 9 } } }$ </td><td> ${ \bf 3 . 9 3 \pm 1 . 1 9 }$ </td></tr><tr><td>No predictive latent</td><td> $4 . 8 9 \pm 2 . 1 3$ </td><td> $7 . 9 5 \pm 3 . 7 6$ </td><td> $1 0 . 9 6 \pm 4 . 8 9$ </td><td> $1 3 . 8 4 \pm 5 . 6 6$ </td><td> $1 6 . 2 4 \pm 5 . 9 4$ </td><td> $1 0 . 7 8 \pm 4 . 4 7$ </td></tr><tr><td>FNO</td><td> $4 . 7 4 \pm 0 . 8 6$ </td><td> $5 . 7 7 \pm 1 . 0 4$ </td><td> $7 . 4 0 \pm 1 . 1 7$ </td><td> $9 . 1 3 \pm 1 . 2 1$ </td><td> $1 0 . 9 0 \pm 1 . 2 0$ </td><td> $7 . 5 9 \pm 1 . 0 9$ </td></tr><tr><td rowspan="3">FitzHugh-Nagumo</td><td>RD-JEPA</td><td> $\mathbf { 2 . 6 4 \pm 0 . 5 6 }$ </td><td> ${ \bf 3 . 3 7 \pm 0 . 5 9 }$ </td><td> $\mathbf { 4 . 2 3 \pm 0 . 8 0 }$ </td><td> ${ \pm } \mathbf { { 1 . 2 7 \pm 1 . 0 4 } }$ </td><td> ${ \bf 6 . 4 9 \pm 1 . 1 7 }$ </td><td> $\mathbf { 4 . 4 0 \pm 0 . 8 3 }$ </td></tr><tr><td>No predictive latent</td><td> $7 . 3 9 \pm 1 . 7 8$ </td><td> $9 . 1 9 \pm 1 . 8 1$ </td><td> $1 1 . 8 5 \pm 2 . 2 7$ </td><td> $1 4 . 9 3 \pm 3 . 0 6$ </td><td> $1 8 . 4 0 \pm 3 . 7 0$ </td><td> $1 2 . 3 5 \pm 2 . 5 0$ </td></tr><tr><td>FNO</td><td> $1 7 . 0 7 \pm 2 . 4 2$ </td><td> $1 6 . 6 5 \pm 2 . 2 1$ </td><td> $1 6 . 7 8 \pm 2 . 0 0$ </td><td> $1 7 . 3 3 \pm 1 . 7 8$ </td><td> $1 8 . 4 6 \pm 1 . 5 8$ </td><td> $1 7 . 2 6 \pm 1 . 9 9$ </td></tr><tr><td rowspan="3">Brusselator</td><td>RD-JEPA</td><td> $\mathbf { 2 . 3 3 \pm 0 . 1 5 }$ </td><td> ${ \bf 3 . 1 4 \pm 0 . 5 2 }$ </td><td> ${ \bf 3 . 9 8 \pm 0 . 7 5 }$ </td><td> $\mathbf { 4 . 7 7 \pm 0 . 9 3 }$ </td><td> ${ \bf 6 . 1 9 \pm 1 . 0 0 }$ </td><td> $\mathbf { 4 . 0 8 \pm 0 . 6 7 }$ </td></tr><tr><td>No predictive latent</td><td> $3 . 1 9 \pm 0 . 1 4$ </td><td> $5 . 8 4 \pm 0 . 0 6$ </td><td> $8 . 4 1 \pm 0 . 2 4$ </td><td> $1 0 . 8 4 \pm 0 . 4 1$ </td><td> $1 3 . 2 0 \pm 0 . 4 3$ </td><td> $8 . 2 9 \pm 0 . 2 0$ </td></tr><tr><td>FNO</td><td> $6 . 4 6 \pm 0 . 3 1$ </td><td> $9 . 0 6 \pm 0 . 1 6$ </td><td> $1 2 . 1 5 \pm 0 . 1 8$ </td><td> $1 4 . 7 9 \pm 0 . 3 1$ </td><td> $1 7 . 0 5 \pm 0 . 2 2$ </td><td> $1 1 . 9 0 { \pm } 0 . 1 5$ </td></tr><tr><td rowspan="3">Ginzburg-Landau</td><td>RD-JEPA</td><td> $\mathbf { 4 . 8 7 \pm 0 . 2 9 }$ </td><td> ${ \bf 6 . 4 2 \pm 0 . 2 9 }$ </td><td> $\mathbf { 8 . 2 2 \pm 0 . 4 4 }$ </td><td> ${ \bf 1 0 . 0 2 \pm 0 . 5 8 }$ </td><td> ${ \bf 1 1 . 9 9 \pm 0 . 8 2 }$ </td><td> ${ \bf 8 . 3 0 \pm 0 . 4 8 }$ </td></tr><tr><td>No predictive latent</td><td> $1 1 . 1 0 \pm 4 . 9 3$ </td><td> $2 0 . 5 4 \pm 1 0 . 2 8$ </td><td> $2 8 . 6 9 \pm 1 3 . 7 3$ </td><td> $3 6 . 3 9 \pm 1 6 . 2 2$ </td><td> $4 4 . 6 6 \pm 1 9 . 1 0$ </td><td> $2 8 . 2 8 \pm 1 2 . 8 5$ </td></tr><tr><td>FNO</td><td> $1 8 . 5 2 \pm 2 . 2 9$ </td><td> $2 0 . 4 3 \pm 1 . 6 1$ </td><td> $2 6 . 5 0 \pm 3 . 2 6$ </td><td>34.12±4.84</td><td> $4 1 . 9 9 \pm 6 . 1 8$ </td><td> $2 8 . 3 1 \pm 3 . 2 5$ </td></tr><tr><td rowspan="3">Schnakenberg</td><td>RD-JEPA</td><td> $\mathbf { 1 . 6 5 \pm 0 . 0 3 }$ </td><td> $\pm { \bf 0 . 0 2 \pm 0 . 0 1 }$ </td><td> $\mathbf { 2 . 3 6 \pm 0 . 0 5 }$ </td><td> $\mathbf { 2 . 6 0 \pm 0 . 1 0 }$ </td><td> $\mathbf { \ } 2 . 9 7 \pm \mathbf { 0 . } 3 2$ </td><td> $\mathbf { 2 . 3 2 \pm 0 . 0 9 }$ </td></tr><tr><td>No predictive latent</td><td> $2 . 9 2 \pm 0 . 5 0$ </td><td> $4 . 7 6 \pm 0 . 2 9$ </td><td> $6 . 9 0 \pm 0 . 4 1$ </td><td> $9 . 1 5 \pm 0 . 4 5$ </td><td> $1 1 . 5 1 \pm 0 . 5 0$ </td><td> $7 . 0 5 \pm 0 . 3 5$ </td></tr><tr><td>FNO</td><td> $4 . 6 1 \pm 0 . 5 0$ </td><td> $6 . 2 6 \pm 0 . 5 9$ </td><td> $8 . 3 4 \pm 0 . 6 4$ </td><td> $1 0 . 3 6 \pm 0 . 6 0$ </td><td> $1 2 . 1 5 \pm 0 . 5 6$ </td><td> $8 . 3 4 \pm 0 . 5 8$ </td></tr></table>

Table S8. Complete relative $L ^ { 2 }$ results on the five source systems at $K = 2 0 .$ . Entries are $1 0 0 \times$ the error and report the mean ± sample s.d. across three paired support-set selections. Within each run, errors are first averaged over all valid windows within each trajectory and then over the same 300 test trajectories (100 from each test regime). The Avg. column averages $h = 1 , \ldots , 5$ within each run before across-run aggregation. Lower values are better. Boldface marks the lowest numerical mean within each equation and horizon; no formal hypothesis tests were performed.
<table><tr><td>PDE</td><td>Model</td><td>h= 1</td><td>h = 2</td><td> $h = 3$ </td><td>h= 4</td><td>h= 5</td><td>Avg.</td></tr><tr><td rowspan="3">Gray-Scott</td><td>RD-JEPA</td><td> ${ \bf 1 . 4 2 \pm 0 . 0 4 }$ </td><td> ${ \bf 1 . 8 7 \pm 0 . 0 9 }$ </td><td> $\mathbf { 2 . 4 8 \pm 0 . 1 8 }$ </td><td> ${ \bf 3 . 1 9 \pm 0 . 2 6 }$ </td><td> $\mathbf { 4 . 0 0 \pm 0 . 3 6 }$ </td><td> $\mathbf { 2 . 5 9 \pm 0 . 1 7 }$ </td></tr><tr><td>No predictive latent</td><td> $2 . 4 3 \pm 0 . 2 4$ </td><td> $3 . 9 7 \pm 0 . 4 9$ </td><td> $5 . 5 3 \pm 0 . 7 5$ </td><td> $7 . 1 9 \pm 1 . 0 0$ </td><td> $8 . 9 8 \pm 1 . 1 5$ </td><td> $5 . 6 2 \pm 0 . 7 1$ </td></tr><tr><td>FNO</td><td> $3 . 3 6 \pm 0 . 5 3$ </td><td> $4 . 2 4 \pm 0 . 6 4$ </td><td> $5 . 6 5 \pm 0 . 7 2$ </td><td> $7 . 2 5 \pm 0 . 8 1$ </td><td> $8 . 9 1 \pm 0 . 8 7$ </td><td> $5 . 8 8 \pm 0 . 7 1$ </td></tr><tr><td rowspan="3">FitzHugh-Nagumo</td><td>RD-JEPA</td><td> $\pm . 2 4 \pm \mathbf { 0 . 3 2 }$ </td><td> $\mathbf { 2 . 5 5 \pm 0 . 0 8 }$ </td><td> ${ \bf 3 . 1 2 \pm 0 . 1 1 }$ </td><td> $\mathbf { 3 . 9 0 \pm 0 . 1 3 }$ </td><td> ${ \bf 4 . 8 4 \pm 0 . 1 5 }$ </td><td> ${ \bf 3 . 3 3 \pm 0 . 1 1 }$ </td></tr><tr><td>No predictive latent</td><td> $7 . 6 4 \pm 0 . 4 0$ </td><td> $8 . 7 4 \pm 0 . 3 4$ </td><td> $1 1 . 1 6 \pm 0 . 6 1$ </td><td> $1 3 . 9 6 \pm 0 . 7 2$ </td><td> $1 7 . 2 0 \pm 0 . 5 8$ </td><td> $1 1 . 7 4 \pm 0 . 5 0$ </td></tr><tr><td>FNO</td><td> $1 3 . 9 5 \pm 1 . 4 3$ </td><td> $1 3 . 6 1 \pm 1 . 2 3$ </td><td> $1 3 . 7 8 \pm 0 . 9 9$ </td><td> $1 4 . 5 2 \pm 0 . 9 1$ </td><td> $1 5 . 6 1 \pm 0 . 7 9$ </td><td> $1 4 . 2 9 \pm 1 . 0 7$ </td></tr><tr><td rowspan="3">Brusselator</td><td>RD-JEPA</td><td> $\mathbf { 2 . 0 1 \pm 0 . 3 4 }$ </td><td> $\mathbf { 2 . 3 5 \pm 0 . 3 8 }$ </td><td> $\mathbf { 2 . 6 7 \pm 0 . 2 7 }$ </td><td> ${ \bf 3 . 0 7 \pm 0 . 1 3 }$ </td><td> $\mathbf { 4 . 2 1 \pm 0 . 2 4 }$ </td><td> $\pm . 8 6 \pm \mathbf { 0 . 2 6 }$ </td></tr><tr><td>No predictive latent</td><td> $3 . 0 0 \pm 0 . 1 2$ </td><td> $4 . 9 3 \pm 0 . 1 5$ </td><td> $7 . 0 9 \pm 0 . 2 3$ </td><td> $9 . 2 8 \pm 0 . 3 0$ </td><td> $1 1 . 5 6 \pm 0 . 4 0$ </td><td> $7 . 1 7 \pm 0 . 2 3$ </td></tr><tr><td>FNO</td><td> $4 . 6 5 \pm 0 . 1 1$ </td><td> $6 . 5 5 \pm 0 . 0 9$ </td><td> $8 . 7 8 \pm 0 . 1 9$ </td><td> $1 0 . 8 6 \pm 0 . 1 6$ </td><td>12.70±0.13</td><td> $8 . 7 1 \pm 0 . 1 2$ </td></tr><tr><td rowspan="3">Ginzburg-Landau</td><td>RD-JEPA</td><td> $\mathbf { 4 . 0 8 \pm 0 . 4 7 }$ </td><td> ${ \pm } . 3 2 \pm 0 . 4 3$ </td><td> ${ \bf 6 . 8 0 \pm 0 . 4 5 }$ </td><td> ${ \bf 8 . 3 9 \pm 0 . 5 1 }$ </td><td> ${ \bf 1 0 . 2 4 \pm 0 . 5 8 }$ </td><td> ${ \bf 6 . 9 7 \pm 0 . 4 8 }$ </td></tr><tr><td>No predictive latent</td><td> $1 0 . 8 1 \pm 1 . 9 8$ </td><td> $1 6 . 1 7 \pm 2 . 5 5$ </td><td> $2 0 . 3 4 \pm 2 . 8 3$ </td><td> $2 4 . 9 5 \pm 3 . 6 6$ </td><td> $3 0 . 2 8 \pm 4 . 4 4$ </td><td> $2 0 . 5 1 \pm 2 . 9 3$ </td></tr><tr><td>FNO</td><td> $1 2 . 2 6 \pm 1 . 4 3$ </td><td> $1 3 . 2 8 \pm 1 . 2 0$ </td><td> $1 6 . 6 1 \pm 1 . 3 1$ </td><td> $2 0 . 9 9 \pm 1 . 5 6$ </td><td> $2 5 . 8 7 \pm 1 . 8 6$ </td><td> $1 7 . 8 0 \pm 1 . 4 6$ </td></tr><tr><td rowspan="3">Schnakenberg</td><td>RD-JEPA</td><td> ${ \bf 1 . 3 7 \pm 0 . 2 7 }$ </td><td> ${ \bf 1 . 6 0 \pm 0 . 2 6 }$ </td><td> ${ \bf 1 . 8 4 \pm 0 . 3 1 }$ </td><td> $\mathbf { 2 . 1 3 \pm 0 . 3 8 }$ </td><td> ${ \bf 2 . 5 9 \pm 0 . 2 5 }$ </td><td> ${ \bf 1 . 9 1 \pm 0 . 2 9 }$ </td></tr><tr><td>No predictive latent</td><td> $2 . 2 1 \pm 0 . 2 0$ </td><td> $3 . 6 7 \pm 0 . 3 3$ </td><td> $5 . 2 7 \pm 0 . 3 4$ </td><td> $6 . 9 4 \pm 0 . 3 5$ </td><td> $8 . 7 2 \pm 0 . 4 2$ </td><td> $5 . 3 6 \pm 0 . 3 2$ </td></tr><tr><td>FNO</td><td> $3 . 1 3 \pm 0 . 3 0$ </td><td> $4 . 3 3 \pm 0 . 4 1$ </td><td> $6 . 0 0 \pm 0 . 5 0$ </td><td> $7 . 7 4 \pm 0 . 5 3$ </td><td> $9 . 4 3 \pm 0 . 4 9$ </td><td> $6 . 1 2 \pm 0 . 4 4$ </td></tr></table>

Table S9. Complete gradient $L ^ { 1 }$ results on the five source systems at $K = 5 .$ Entries are $1 0 0 \times$ the error and report the mean ± sample s.d. across three paired support-set selections. Within each run, errors are first averaged over all valid windows within each trajectory and then over the same 300 test trajectories (100 from each test regime). The Avg. column averages $h = 1 , \ldots , 5$ within each run before across-run aggregation. Lower values are better. Boldface marks the lowest numerical mean within each equation and horizon; no formal hypothesis tests were performed.
<table><tr><td>PDE</td><td>Model</td><td>h= 1</td><td>h=2</td><td>h=3</td><td>h=4</td><td> $h = 5$ </td><td>Avg.</td></tr><tr><td>Gray-Scott</td><td>RD-JEPA</td><td> $\mathbf { 0 . 3 5 \pm 0 . 1 0 }$ </td><td> $\mathbf { 0 . 5 3 \pm 0 . 1 2 }$ </td><td> ${ \bf 0 . 7 0 \pm 0 . 1 2 }$ </td><td> ${ \bf 0 . 8 6 \pm 0 . 1 1 }$ </td><td> ${ \bf 1 . 0 2 \pm 0 . 0 9 }$ </td><td> ${ \bf 0 . 6 9 \pm 0 . 1 1 }$ </td></tr><tr><td></td><td>No predictive latent</td><td> $0 . 7 2 \pm 0 . 2 0$ </td><td> $1 . 2 1 \pm 0 . 3 2$ </td><td> $1 . 6 3 \pm 0 . 3 9$ </td><td> $2 . 0 0 \pm 0 . 4 1$ </td><td> $2 . 3 2 \pm 0 . 3 6$ </td><td> $1 . 5 7 \pm 0 . 3 3$ </td></tr><tr><td></td><td>FNO</td><td> $0 . 6 9 \pm 0 . 0 2$ </td><td> $0 . 8 8 \pm 0 . 0 3$ </td><td> $1 . 1 6 \pm 0 . 0 5$ </td><td> $1 . 4 4 \pm 0 . 0 6$ </td><td> $1 . 7 2 \pm 0 . 0 6$ </td><td> $1 . 1 8 \pm 0 . 0 4$ </td></tr><tr><td></td><td>RD-JEPA</td><td> ${ \bf 0 . 1 6 \pm 0 . 0 4 }$ </td><td> ${ \bf 0 . 1 9 \pm 0 . 0 4 }$ </td><td> $\mathbf { 0 . 2 3 \pm 0 . 0 4 }$ </td><td> $\mathbf { 0 . 2 8 \pm 0 . 0 3 }$ </td><td> ${ \bf 0 . 3 4 \pm 0 . 0 3 }$ </td><td> ${ \bf 0 . 2 4 \pm 0 . 0 4 }$ </td></tr><tr><td>FitzHugh-Nagumo</td><td>No predictive latent</td><td> $0 . 4 4 \pm 0 . 1 1$ </td><td> $0 . 5 3 \pm 0 . 1 2$ </td><td> $0 . 6 5 \pm 0 . 1 3$ </td><td> $0 . 8 0 \pm 0 . 1 4$ </td><td> $0 . 9 6 \pm 0 . 1 6$ </td><td> $0 . 6 8 \pm 0 . 1 3$ </td></tr><tr><td></td><td>FNO</td><td> $1 . 1 6 \pm 0 . 0 5$ </td><td> $1 . 1 8 \pm 0 . 0 5$ </td><td> $1 . 2 3 \pm 0 . 0 6$ </td><td> $1 . 2 9 \pm 0 . 0 6$ </td><td> $1 . 3 8 \pm 0 . 0 7$ </td><td> $1 . 2 5 \pm 0 . 0 6$ </td></tr><tr><td></td><td>RD-JEPA</td><td> ${ \bf 1 . 3 1 \pm 0 . 1 6 }$ </td><td> ${ \bf 1 . 7 3 \pm 0 . 1 3 }$ </td><td> $\mathbf { 2 . 2 7 \pm 0 . 2 6 }$ </td><td> $\mathbf { 2 . 8 2 \pm 0 . 3 9 }$ </td><td> ${ \bf 3 . 5 8 \pm 0 . 5 5 }$ </td><td> $\mathbf { 2 . 3 4 \pm 0 . 2 5 }$ </td></tr><tr><td>Brusselator</td><td>No predictive latent</td><td> $2 . 9 8 \pm 0 . 2 6$ </td><td> $5 . 0 2 \pm 0 . 1 6$ </td><td> $7 . 1 4 \pm 0 . 2 2$ </td><td> $9 . 1 5 \pm 0 . 3 3$ </td><td> $1 0 . 9 1 \pm 0 . 5 4$ </td><td> $7 . 0 4 \pm 0 . 2 0$ </td></tr><tr><td></td><td>FNO</td><td> $3 . 5 2 \pm 0 . 3 6$ </td><td> $4 . 5 1 \pm 0 . 4 0$ </td><td> $5 . 7 7 \pm 0 . 3 2$ </td><td> $6 . 7 4 \pm 0 . 1 7$ </td><td> $7 . 5 4 \pm 0 . 1 5$ </td><td> $5 . 6 2 \pm 0 . 2 7$ </td></tr><tr><td></td><td>RD-JEPA</td><td> ${ \bf 0 . 3 2 \pm 0 . 1 1 }$ </td><td> ${ \bf 0 . 3 7 \pm 0 . 0 8 }$ </td><td> ${ \bf 0 . 4 4 \pm 0 . 0 7 }$ </td><td> $\mathbf { 0 . 5 3 \pm 0 . 0 7 }$ </td><td> ${ \bf 0 . 6 2 \pm 0 . 0 6 }$ </td><td> $\mathbf { 0 . 4 5 \pm 0 . 0 8 }$ </td></tr><tr><td>Ginzburg-Landau</td><td>No predictive latent</td><td> $0 . 6 2 \pm 0 . 4 2$ </td><td> $1 . 0 5 \pm 0 . 6 7$ </td><td> $1 . 4 5 \pm 0 . 8 5$ </td><td> $1 . 8 1 \pm 0 . 9 3$ </td><td> $2 . 1 5 \pm 0 . 9 5$ </td><td> $1 . 4 2 \pm 0 . 7 6$ </td></tr><tr><td></td><td>FNO</td><td> $1 . 0 1 \pm 0 . 0 7$ </td><td> $1 . 1 0 \pm 0 . 1 0$ </td><td> $1 . 3 5 \pm 0 . 1 9$ </td><td> $1 . 6 6 \pm 0 . 3 0$ </td><td> $1 . 9 9 \pm 0 . 3 9$ </td><td> $1 . 4 2 \pm 0 . 2 0$ </td></tr><tr><td></td><td>RD-JEPA</td><td> ${ \bf 0 . 3 1 \pm 0 . 0 7 }$ </td><td> $\mathbf { 0 . 3 6 \pm 0 . 0 6 }$ </td><td> ${ \bf 0 . 4 4 \pm 0 . 0 9 }$ </td><td> ${ \bf 0 . 5 1 \pm 0 . 1 1 }$ </td><td> ${ \bf 0 . 6 4 \pm 0 . 1 6 }$ </td><td> $\mathbf { 0 . 4 5 \pm 0 . 1 0 }$ </td></tr><tr><td>Schnakenberg</td><td>No predictive latent</td><td> $0 . 6 0 \pm 0 . 1 0$ </td><td> $0 . 9 2 \pm 0 . 0 8$ </td><td> $1 . 2 9 \pm 0 . 0 8$ </td><td> $1 . 7 0 \pm 0 . 0 7$ </td><td> $2 . 1 4 \pm 0 . 0 8$ </td><td> $1 . 3 3 \pm 0 . 0 8$ </td></tr><tr><td></td><td>FNO</td><td> $1 . 2 4 \pm 0 . 0 2$ </td><td> $1 . 4 0 \pm 0 . 0 3$ </td><td> $1 . 6 3 \pm 0 . 0 4$ </td><td> $1 . 8 5 \pm 0 . 0 3$ </td><td> $2 . 0 7 \pm 0 . 0 2$ </td><td> $1 . 6 4 \pm 0 . 0 3$ </td></tr></table>

Table S10. Complete gradient $L ^ { 1 }$ results on the five source systems at $K = 1 0 .$ . Entries are 100× the error and report the mean ± sample s.d. across three paired support-set selections. Within each run, errors are first averaged over all valid windows within each trajectory and then over the same 300 test trajectories (100 from each test regime). The Avg. column averages $h = 1 , \ldots , 5$ within each run before across-run aggregation. Lower values are better. Boldface marks the lowest numerical mean within each equation and horizon; no formal hypothesis tests were performed.
<table><tr><td>PDE</td><td>Model</td><td>h=1</td><td>h=2</td><td>h=3</td><td>h=4</td><td>h=5</td><td>Avg.</td></tr><tr><td rowspan="3">Gray-Scott</td><td>RD-JEPA</td><td> ${ \bf 0 . 2 1 \pm 0 . 0 7 }$ </td><td> $\mathbf { 0 . 3 5 \pm 0 . 1 1 }$ </td><td> ${ \bf 0 . 4 9 \pm 0 . 1 4 }$ </td><td> $\mathbf { 0 . 6 3 \pm 0 . 1 7 }$ </td><td> ${ \bf 0 . 7 7 \pm 0 . 1 7 }$ </td><td> ${ \bf 0 . 4 9 \pm 0 . 1 3 }$ </td></tr><tr><td>No predictive latent</td><td> $0 . 5 2 \pm 0 . 2 5$ </td><td> $0 . 9 3 \pm 0 . 4 7$ </td><td> $1 . 3 3 \pm 0 . 6 2$ </td><td> $1 . 6 8 \pm 0 . 7 2$ </td><td> $1 . 9 9 \pm 0 . 7 4$ </td><td> $1 . 2 9 \pm 0 . 5 6$ </td></tr><tr><td>FNO</td><td> $0 . 5 3 \pm 0 . 0 3$ </td><td> $0 . 7 0 \pm 0 . 0 8$ </td><td> $0 . 9 6 \pm 0 . 1 0$ </td><td> $1 . 2 3 \pm 0 . 1 2$ </td><td> $1 . 5 0 \pm 0 . 1 2$ </td><td> $0 . 9 8 \pm 0 . 0 9$ </td></tr><tr><td rowspan="3">FitzHugh-Nagumo</td><td>RD-JEPA</td><td> ${ \bf 0 . 1 } 2 \pm { \bf 0 . 0 } 2$ </td><td> $\mathbf { 0 . 1 5 \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 1 9 \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 2 4 \pm 0 . 0 3 }$ </td><td> ${ \bf 0 . 2 9 } \pm { \bf 0 . 0 3 }$ </td><td> ${ \bf 0 . 2 0 \pm 0 . 0 3 }$ </td></tr><tr><td>No predictive latent</td><td> $0 . 3 3 \pm 0 . 0 3$ </td><td> $0 . 4 1 \pm 0 . 0 4$ </td><td> $0 . 5 3 \pm 0 . 0 6$ </td><td> $0 . 6 6 \pm 0 . 0 8$ </td><td> $0 . 8 0 \pm 0 . 1 0$ </td><td> $0 . 5 5 \pm 0 . 0 6$ </td></tr><tr><td>FNO</td><td> $0 . 8 6 \pm 0 . 0 4$ </td><td> $0 . 8 7 \pm 0 . 0 3$ </td><td> $0 . 9 0 \pm 0 . 0 2$ </td><td> $0 . 9 5 \pm 0 . 0 2$ </td><td> $1 . 0 2 \pm 0 . 0 2$ </td><td> $0 . 9 2 \pm 0 . 0 2$ </td></tr><tr><td rowspan="3">Brusselator</td><td>RD-JEPA</td><td> $\mathbf { 0 . 9 6 \pm 0 . 0 6 }$ </td><td> ${ \bf 1 . 2 9 \pm 0 . 0 7 }$ </td><td> ${ \bf 1 . 6 4 \pm 0 . 1 3 }$ </td><td> ${ \bf 1 . 9 8 \pm 0 . 1 4 }$ </td><td> $\mathbf { 2 . 5 1 \pm 0 . 1 3 }$ </td><td> ${ \bf 1 . 6 7 \pm 0 . 0 9 }$ </td></tr><tr><td>No predictive latent</td><td> $1 . 2 2 \pm 0 . 1 4$ </td><td> $2 . 1 1 \pm 0 . 1 5$ </td><td> $2 . 9 8 \pm 0 . 1 6$ </td><td> $3 . 8 0 \pm 0 . 1 8$ </td><td> $4 . 5 7 \pm 0 . 1 3$ </td><td> $2 . 9 3 \pm 0 . 1 4$ </td></tr><tr><td>FNO</td><td> $2 . 6 7 \pm 0 . 2 3$ </td><td> $3 . 5 2 \pm 0 . 2 1$ </td><td> $4 . 4 8 \pm 0 . 2 1$ </td><td> $5 . 3 2 \pm 0 . 2 2$ </td><td> $6 . 0 8 \pm 0 . 2 3$ </td><td> $4 . 4 1 \pm 0 . 2 2$ </td></tr><tr><td rowspan="3">Ginzburg-Landau</td><td>RD-JEPA</td><td> ${ \bf 0 . 2 0 \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 2 6 \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 3 4 \pm 0 . 0 1 }$ </td><td> ${ \bf 0 . 4 2 \pm 0 . 0 2 }$ </td><td> $\mathbf { 0 . 5 1 \pm 0 . 0 3 }$ </td><td> $\mathbf { 0 . 3 5 \pm 0 . 0 2 }$ </td></tr><tr><td>No predictive latent</td><td> $0 . 3 8 \pm 0 . 1 4$ </td><td> $0 . 6 8 \pm 0 . 3 0$ </td><td> $0 . 9 5 \pm 0 . 4 1$ </td><td> $1 . 2 3 \pm 0 . 5 0$ </td><td> $1 . 5 6 \pm 0 . 6 2$ </td><td> $0 . 9 6 \pm 0 . 3 9$ </td></tr><tr><td>FNO</td><td> $0 . 8 2 \pm 0 . 0 7$ </td><td> $0 . 8 5 \pm 0 . 0 5$ </td><td> $1 . 0 3 \pm 0 . 0 9$ </td><td> $1 . 2 9 \pm 0 . 1 5$ </td><td> $1 . 5 7 \pm 0 . 2 1$ </td><td> $1 . 1 1 \pm 0 . 1 0$ </td></tr><tr><td rowspan="3">Schnakenberg</td><td>RD-JEPA</td><td> $\mathbf { 0 . 2 5 \pm 0 . 0 0 }$ </td><td> $\mathbf { 0 . 2 8 \pm 0 . 0 0 }$ </td><td> $\mathbf { 0 . 3 2 \pm 0 . 0 0 }$ </td><td> $\mathbf { 0 . 3 6 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 4 2 \pm 0 . 0 3 }$ </td><td> $\mathbf { 0 . 3 2 \pm 0 . 0 0 }$ </td></tr><tr><td>No predictive latent</td><td> $0 . 4 0 \pm 0 . 0 8$ </td><td> $0 . 6 3 \pm 0 . 0 5$ </td><td> $0 . 9 0 \pm 0 . 0 6$ </td><td> $1 . 1 8 \pm 0 . 0 6$ </td><td> $1 . 4 9 \pm 0 . 0 6$ </td><td> $0 . 9 2 \pm 0 . 0 5$ </td></tr><tr><td>FNO</td><td> $0 . 8 3 \pm 0 . 0 9$ </td><td> $1 . 0 0 \pm 0 . 0 9$ </td><td> $1 . 2 3 \pm 0 . 0 8$ </td><td> $1 . 4 6 \pm 0 . 0 8$ </td><td> $1 . 6 7 \pm 0 . 0 7$ </td><td> $1 . 2 4 \pm 0 . 0 8$ </td></tr></table>

Table S11. Complete gradient $L ^ { 1 }$ results on the five source systems at K = 20. Entries are 100× the error and report the mean ± sample s.d. across three paired support-set selections. Within each run, errors are first averaged over all valid windows within each trajectory and then over the same 300 test trajectories (100 from each test regime). The Avg. column averages $h = 1 , \ldots , 5$ within each run before across-run aggregation. Lower values are better. Boldface marks the lowest numerical mean within each equation and horizon; no formal hypothesis tests were performed.
<table><tr><td>PDE</td><td>Model</td><td>h= 1</td><td>h = 2</td><td>h= 3</td><td>h= 4</td><td> $h = 5$ </td><td>Avg.</td></tr><tr><td rowspan="3">Gray-Scott</td><td>RD-JEPA</td><td> ${ \bf 0 . 1 6 \pm 0 . 0 3 }$ </td><td> $\mathbf { 0 . 2 3 \pm 0 . 0 3 }$ </td><td> $\mathbf { 0 . 3 2 \pm 0 . 0 4 }$ </td><td> ${ \bf 0 . 4 1 \pm 0 . 0 5 }$ </td><td> $\mathbf { 0 . 5 2 \pm 0 . 0 6 }$ </td><td> $\mathbf { 0 . 3 3 \pm 0 . 0 4 }$ </td></tr><tr><td>No predictive latent</td><td> $0 . 2 9 \pm 0 . 0 2$ </td><td> $0 . 4 8 \pm 0 . 0 7$ </td><td> $0 . 6 9 \pm 0 . 1 0$ </td><td> $0 . 9 2 \pm 0 . 1 3$ </td><td> $1 . 1 8 \pm 0 . 1 6$ </td><td> $0 . 7 1 \pm 0 . 1 0$ </td></tr><tr><td>FNO</td><td> $0 . 4 0 \pm 0 . 0 2$ </td><td> $0 . 5 3 \pm 0 . 0 4$ </td><td> $0 . 7 6 \pm 0 . 0 7$ </td><td> $1 . 0 1 \pm 0 . 1 0$ </td><td> $1 . 2 7 \pm 0 . 1 1$ </td><td> $0 . 8 0 \pm 0 . 0 7$ </td></tr><tr><td rowspan="3">FitzHugh-Nagumo</td><td>RD-JEPA</td><td> $\mathbf { 0 . 1 0 \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 1 2 \pm 0 . 0 0 }$ </td><td> $\mathbf { 0 . 1 5 \pm 0 . 0 1 }$ </td><td> ${ \bf 0 . 1 9 \pm 0 . 0 1 }$ </td><td> ${ \bf 0 . 2 4 \pm 0 . 0 1 }$ </td><td> ${ \bf 0 . 1 6 \pm 0 . 0 1 }$ </td></tr><tr><td>No predictive latent</td><td> $0 . 3 5 \pm 0 . 0 2$ </td><td> $0 . 4 1 \pm 0 . 0 1$ </td><td> $0 . 5 1 \pm 0 . 0 2$ </td><td> $0 . 6 3 \pm 0 . 0 3$ </td><td> $0 . 7 6 \pm 0 . 0 3$ </td><td> $0 . 5 3 \pm 0 . 0 2$ </td></tr><tr><td>FNO</td><td> $0 . 6 7 \pm 0 . 0 3$ </td><td> $0 . 6 7 \pm 0 . 0 3$ </td><td> $0 . 7 0 \pm 0 . 0 3$ </td><td> $0 . 7 4 \pm 0 . 0 3$ </td><td> $0 . 8 0 \pm 0 . 0 3$ </td><td> $0 . 7 2 \pm 0 . 0 3$ </td></tr><tr><td rowspan="3">Brusselator</td><td>RD-JEPA</td><td> $\mathbf { 0 . 8 0 \pm 0 . 0 4 }$ </td><td> ${ \bf 1 . 0 1 \pm 0 . 1 1 }$ </td><td> ${ \bf 1 . 2 0 \pm 0 . 0 7 }$ </td><td> ${ \bf 1 . 4 0 \pm 0 . 0 2 }$ </td><td> ${ \bf 1 . 8 5 \pm 0 . 0 5 }$ </td><td> ${ \bf 1 . 2 5 \pm 0 . 0 5 }$ </td></tr><tr><td>No predictive latent</td><td> $1 . 1 0 \pm 0 . 0 4$ </td><td> $1 . 8 4 \pm 0 . 0 3$ </td><td> $2 . 5 8 \pm 0 . 0 6$ </td><td> $3 . 3 1 \pm 0 . 0 7$ </td><td> $4 . 0 7 \pm 0 . 0 8$ </td><td> $2 . 5 8 \pm 0 . 0 4$ </td></tr><tr><td>FNO</td><td> $2 . 0 9 \pm 0 . 1 1$ </td><td> $2 . 6 9 \pm 0 . 1 5$ </td><td> $3 . 4 2 \pm 0 . 1 9$ </td><td> $4 . 1 2 \pm 0 . 1 8$ </td><td> $4 . 7 6 \pm 0 . 1 7$ </td><td> $3 . 4 1 \pm 0 . 1 6$ </td></tr><tr><td rowspan="3">Ginzburg-Landau</td><td>RD-JEPA</td><td> ${ \bf 0 . 1 6 \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . } 2 2 \pm \mathbf { 0 . 0 2 }$ </td><td> $\mathbf { 0 . 2 8 \pm 0 . 0 3 }$ </td><td> $\mathbf { 0 . 3 5 \pm 0 . 0 4 }$ </td><td> ${ \bf 0 . 4 4 \pm 0 . 0 5 }$ </td><td> ${ \bf 0 . 2 9 } \pm { \bf 0 . 0 3 }$ </td></tr><tr><td>No predictive latent</td><td> $0 . 3 6 \pm 0 . 0 7$ </td><td> $0 . 5 3 \pm 0 . 0 8$ </td><td> $0 . 6 9 \pm 0 . 0 8$ </td><td> $0 . 8 7 \pm 0 . 1 0$ </td><td> $1 . 0 9 \pm 0 . 1 1$ </td><td> $0 . 7 1 \pm 0 . 0 8$ </td></tr><tr><td>FNO</td><td> $0 . 5 8 \pm 0 . 0 5$ </td><td> $0 . 6 0 \pm 0 . 0 4$ </td><td> $0 . 7 1 \pm 0 . 0 5$ </td><td> $0 . 8 8 \pm 0 . 0 6$ </td><td> $1 . 0 7 \pm 0 . 0 7$ </td><td> $0 . 7 7 \pm 0 . 0 6$ </td></tr><tr><td rowspan="3">Schnakenberg</td><td>RD-JEPA</td><td> ${ \bf 0 . 2 2 \pm 0 . 0 4 }$ </td><td> $\mathbf { 0 . 2 3 \pm 0 . 0 4 }$ </td><td> ${ \bf 0 . 2 6 \pm 0 . 0 4 }$ </td><td> ${ \bf 0 . 3 1 \pm 0 . 0 5 }$ </td><td> ${ \bf 0 . 3 9 \pm 0 . 0 4 }$ </td><td> $\mathbf { 0 . 2 8 \pm 0 . 0 4 }$ </td></tr><tr><td>No predictive latent</td><td> $0 . 3 0 \pm 0 . 0 2$ </td><td> $0 . 4 8 \pm 0 . 0 2$ </td><td> $0 . 6 9 \pm 0 . 0 2$ </td><td> $0 . 9 1 \pm 0 . 0 2$ </td><td> $1 . 1 4 \pm 0 . 0 3$ </td><td> $0 . 7 0 \pm 0 . 0 2$ </td></tr><tr><td>FNO</td><td> $0 . 6 0 \pm 0 . 0 4$ </td><td> $0 . 7 2 \pm 0 . 0 5$ </td><td> $0 . 9 0 \pm 0 . 0 6$ </td><td> $1 . 1 1 \pm 0 . 0 6$ </td><td> $1 . 3 2 \pm 0 . 0 6$ </td><td> $0 . 9 3 \pm 0 . 0 5$ </td></tr></table>

Table S12. Split-resolved source-system forecasting. Each entry reports the mean ± sample s.d. across three paired support-set selections. Within each run, errors are averaged over the indicated 100-trajectory split and then given equal weight across the five source equations and five forecast horizons. Values are multiplied by 100; lower values are better. Boldface marks the lowest numerical mean within each split and adaptation budget; no formal hypothesis tests were performed.
<table><tr><td></td><td></td><td colspan="3">Relative  $L ^ { 2 }$ </td><td colspan="3">Gradient  $L ^ { 1 }$ </td></tr><tr><td>Model</td><td>K</td><td>ID</td><td> $\mathbf { C o e f f . . O O D }$ </td><td>IC-OOD</td><td>ID</td><td> $\mathbf { C o e f f . . O O D }$ </td><td>IC-OOD</td></tr><tr><td></td><td>5</td><td> ${ \bf 6 . 6 1 \pm 1 . 0 6 }$ </td><td> ${ \bf 6 . 8 9 \pm 1 . 1 4 }$ </td><td> ${ \bf 6 . 8 8 \pm 1 . 2 3 }$ </td><td> ${ \bf 0 . 7 9 } \pm { \bf 0 . 1 0 }$ </td><td> $\mathbf { 0 . 8 5 \pm 0 . 1 0 }$ </td><td> $\mathbf { 0 . 8 6 \pm 0 . 0 9 }$ </td></tr><tr><td>RD-JEPA</td><td>10</td><td> ${ \bf 4 . 6 9 \pm 0 . 3 6 }$ </td><td> ${ \bf 4 . 8 7 \pm 0 . 4 5 }$ </td><td> ${ \bf 4 . 8 4 \pm 0 . 3 7 }$ </td><td> $\mathbf { 0 . 6 0 \pm 0 . 0 7 }$ </td><td> ${ \bf 0 . 6 4 \pm 0 . 0 9 }$ </td><td> $\mathbf { 0 . 6 5 \pm 0 . 0 7 }$ </td></tr><tr><td></td><td>20</td><td> ${ \bf 3 . 4 4 \pm 0 . 1 0 }$ </td><td> ${ \bf 3 . 5 2 \pm 0 . 1 1 }$ </td><td> ${ \bf 3 . 6 4 \pm 0 . 1 1 }$ </td><td> $\mathbf { 0 . 4 5 \pm 0 . 0 2 }$ </td><td> $\mathbf { 0 . 4 6 \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 4 8 \pm 0 . 0 1 }$ </td></tr><tr><td></td><td>5</td><td> $2 0 . 6 8 \pm 4 . 5 5$ </td><td> $2 1 . 7 3 \pm 5 . 3 4$ </td><td> $2 2 . 8 8 \pm 6 . 0 4$ </td><td> $2 . 2 9 \pm 0 . 1 5$ </td><td> $2 . 4 1 \pm 0 . 1 4$ </td><td> $2 . 5 1 \pm 0 . 1 9$ </td></tr><tr><td>No predictive latent</td><td>10</td><td> $1 2 . 4 8 \pm 1 . 9 6$ </td><td> $1 3 . 2 8 \pm 2 . 8 1$ </td><td> $1 4 . 2 8 \pm 4 . 7 2$ </td><td> $1 . 2 7 \pm 0 . 1 4$ </td><td> $1 . 3 4 \pm 0 . 2 1$ </td><td> $1 . 3 8 \pm 0 . 2 1$ </td></tr><tr><td></td><td>20</td><td> $9 . 1 1 \pm 0 . 0 7$ </td><td> $9 . 6 5 \pm 0 . 2 3$ </td><td> $1 1 . 4 8 \pm { 1 . 6 3 }$ </td><td> $1 . 0 0 { \pm } 0 . 0 0$ </td><td> $1 . 0 2 \pm 0 . 0 5$ </td><td> $1 . 1 2 \pm 0 . 0 6$ </td></tr><tr><td></td><td>5</td><td> $1 9 . 9 8 \pm 1 . 2 2$ </td><td> $1 9 . 8 2 \pm 1 . 5 4$ </td><td> $2 0 . 0 2 \pm 1 . 4 9$ </td><td> $2 . 1 8 \pm 0 . 0 6$ </td><td> $2 . 1 9 \pm 0 . 0 7$ </td><td> $2 . 2 9 \pm 0 . 0 6$ </td></tr><tr><td>FNO</td><td>10</td><td> $1 4 . 5 2 \pm 0 . 4 0$ </td><td> $1 4 . 6 2 \pm 0 . 4 7$ </td><td> $1 4 . 9 0 \pm 0 . 7 7$ </td><td> $1 . 6 9 \pm 0 . 0 6$ </td><td> $1 . 7 0 \pm 0 . 0 7$ </td><td> $1 . 8 1 \pm 0 . 0 7$ </td></tr><tr><td></td><td>20</td><td> $1 0 . 3 5 \pm 0 . 3 9$ </td><td> $1 0 . 5 1 \pm 0 . 2 3$ </td><td> $1 0 . 8 3 \pm 0 . 2 4$ </td><td> $1 . 2 9 \pm 0 . 0 2$ </td><td> $1 . 2 9 \pm 0 . 0 3$ </td><td> $1 . 3 9 \pm 0 . 0 2$ </td></tr></table>

## Supplementary Note 5: Paired predictive-latent control on held-out governing equations

## S5.1 Evaluation coverage, data mapping and interpretation

The equation-level transfer evaluation covers Lambda–Omega, Barkley and Oregonator, $K \in \{ 1 , 5 , 1 0 \}$ and direct forecast horizons $h = 1 , \ldots , 5 .$ . Figure 5 reports the horizon-resolved comparison among RD-JEPA, the independently trained nopredictive-latent control and $\mathrm { F N O ; }$ the three run-level values underlying those curves are supplied as Supplementary Data 2. The broader horizon-averaged comparison with FNO, LNO, RieszNO, ReViT and CNextU-Net is already reported in main-text Table 1, with its run-level values in Supplementary Data 3. To avoid duplicating those main-text display items, this note reports only the additional paired horizon-averaged comparison between RD-JEPA and the no-predictive-latent control.

Within each support-set selection, the two models receive the same adaptation trajectories and are evaluated on the same 300 test trajectories. The control is retrained independently: it removes the trajectory-specific JEPA encoder–predictor pathway, retains the dense decoder and physical-context pathway, and supplies only spatially shared lead-time conditioning at the latent interface. It is therefore not an inference-time zeroing intervention. The comparison evaluates the contribution of the complete predictive-latent representation pathway under matched support sets and test trajectories.

## S5.2 Horizon-averaged paired comparison

For each equation, model, adaptation budget and support-set selection, errors are first averaged over all valid windows within each trajectory, then over the same 300 test trajectories and finally with equal weight over $h = 1 , \ldots , 5 .$ . The displayed error entries are the mean and sample standard deviation across the three resulting run-level values. Percentage reductions are computed within each paired support-set selection using Eq. (S58) and only then summarized across the three paired reductions. Positive reductions indicate lower error for RD-JEPA. Both the error summaries and paired reductions can be recomputed from the matched horizon-resolved run-level errors supplied as Supplementary Data 2.

Table S13. Paired comparison with the no-predictive-latent control on governing equations excluded from pretraining. Error entries are 100× the error and report the mean ± sample s.d. across three paired support-set selections. Within each selection, errors are first averaged over the same 300 test trajectories and then with equal weight over the five direct forecast horizons. Reduction entries are percentages computed within each paired selection as $1 0 0 ( 1 - E _ { \mathrm { R D - J E P A } } / E _ { \mathrm { c o n t r o l } } )$ and then summarized across the three paired values. Positive reductions favour RD-JEPA. Lower error values are better; boldface marks the lower numerical mean in each paired comparison; no formal hypothesis tests were performed.
<table><tr><td rowspan="2">PDE</td><td rowspan="2">K</td><td colspan="3">Relative  $L ^ { 2 }$ </td><td colspan="3">Gradient  $L ^ { 1 }$ </td></tr><tr><td>RD-JEPA</td><td>No predictive latent</td><td>Reduction (%)</td><td>RD-JEPA</td><td>No predictive latent</td><td>Reduction (%)</td></tr><tr><td rowspan="3">Lambda-Omega</td><td>1</td><td> ${ \pm 2 . 2 4 \pm 2 . 5 8 }$ </td><td> $2 3 . 0 6 \pm 5 . 2 8$ </td><td> $4 6 . 7 8 \pm 1 . 2 2$ </td><td> $\mathbf { 0 . 6 8 \pm 0 . 1 2 }$ </td><td> $1 . 2 4 \pm 0 . 2 8$ </td><td> $4 5 . 1 1 \pm 2 . 1 7$ </td></tr><tr><td>5</td><td> ${ \bf9 . 8 5 \pm 0 . 7 9 }$ </td><td> $1 8 . 7 3 \pm 1 . 3 2$ </td><td> $4 7 . 3 8 \pm 3 . 4 2$ </td><td> $\mathbf { 0 . 5 2 \pm 0 . 0 5 }$ </td><td> $0 . 9 7 \pm 0 . 0 5$ </td><td> $4 6 . 3 9 \pm 4 . 6 9$ </td></tr><tr><td>10</td><td> ${ \pm 1 . 3 8 \pm 1 . 4 6 }$ </td><td> $1 7 . 8 4 \pm 1 . 6 8$ </td><td> $5 3 . 1 6 \pm 5 . 4 6$ </td><td> $\mathbf { 0 . 4 5 \pm 0 . 0 8 }$ </td><td> $0 . 9 4 \pm 0 . 1 0$ </td><td> $5 2 . 0 1 \pm 4 . 7 6$ </td></tr><tr><td rowspan="3">Barkley</td><td>1</td><td> ${ \bf 1 0 . 5 3 \pm 2 . 5 5 }$ </td><td> $1 8 . 9 7 \pm 4 . 0 4$ </td><td> $4 4 . 7 2 \pm 3 . 0 4$ </td><td> ${ \bf 1 . 0 9 \pm 0 . 2 7 }$ </td><td> $1 . 9 7 \pm 0 . 4 2$ </td><td> $4 4 . 6 0 \pm 2 . 6 4$ </td></tr><tr><td>5</td><td> ${ \bf 6 . 9 0 \pm 0 . 2 4 }$ </td><td> $1 2 . 9 7 \pm 0 . 8 5$ </td><td> $4 6 . 7 6 \pm 1 . 7 6$ </td><td> ${ \bf 0 . 7 4 \pm 0 . 0 4 }$ </td><td> $1 . 3 7 \pm 0 . 1 2$ </td><td> $4 6 . 3 5 \pm 2 . 4 6$ </td></tr><tr><td>10</td><td> ${ \pm } 0 . 2 3 \pm 0 . 3 2$ </td><td> $1 0 . 0 6 \pm 0 . 6 7$ </td><td> $4 7 . 9 5 \pm 1 . 0 1$ </td><td> $\mathbf { 0 . 5 8 \pm 0 . 0 4 }$ </td><td> $1 . 1 0 \pm 0 . 0 5$ </td><td> $4 7 . 3 5 \pm 0 . 7 4$ </td></tr><tr><td rowspan="3">Oregonator</td><td>1</td><td> ${ \bf 1 1 . 3 6 \pm 1 . 3 4 }$ </td><td> $2 2 . 1 9 \pm 3 . 9 3$ </td><td> $4 8 . 1 6 \pm 8 . 0 2$ </td><td> ${ \bf 0 . 4 7 \pm 0 . 0 5 }$ </td><td> $1 . 0 1 \pm 0 . 1 2$ </td><td> $5 3 . 4 6 \pm 3 . 7 7$ </td></tr><tr><td>5</td><td> $\mathbf { 4 . 8 3 \pm 0 . 7 9 }$ </td><td> $9 . 8 6 \pm 2 . 4 5$ </td><td> $5 0 . 3 2 \pm 4 . 8 6$ </td><td> $\mathbf { 0 . 2 6 \pm 0 . 0 6 }$ </td><td> $0 . 5 3 \pm 0 . 1 6$ </td><td> $5 0 . 5 0 \pm 2 . 9 4$ </td></tr><tr><td>10</td><td> $\mathbf { 4 . 0 5 \pm 0 . 4 7 }$ </td><td> $8 . 0 6 \pm 1 . 0 4$ </td><td> $4 9 . 6 3 \pm 0 . 7 6$ </td><td> $\mathbf { 0 . } 2 2 \pm \mathbf { 0 . 0 2 }$ </td><td> $0 . 4 5 \pm 0 . 0 3$ </td><td> $5 2 . 1 1 \pm 0 . 5 8$ </td></tr></table>

## Supplementary Note 6: Matched control and boundary-condition stress test

## S6.1 Architecture-matched training without predictive pretraining

The architecture-matched from-scratch control is evaluated only on the three governing equations excluded from pretraining. For every equation and value of K, RD-JEPA and the control receive the same support trajectories and are evaluated on the same 300 test trajectories. Table S14 reports horizon-averaged results.

Across all three target equations and all three adaptation budgets, RD-JEPA has lower mean relative $L ^ { 2 }$ and gradient $L ^ { 1 }$ errors than the architecture-matched scratch model (Supplementary Table S14). Under matched support sets and update budgets, this ordering shows that source pretraining supplies reusable information that the same architecture does not recover from the limited target trajectories alone. The three run-level values underlying each entry are provided in Supplementary Data 4.

Table S14. Architecture-matched control trained without predictive pretraining on governing equations excluded from pretraining. Each entry is 100× the error and reports the mean ± sample s.d. across three paired support-set selections. For each selection, errors are first averaged over the same 300 test trajectories and then over the five direct forecast horizons. The model-training random-number-generator state is fixed, and both models use the same support trajectories within each selection. Lower values are better. Boldface marks the lower numerical mean in each paired comparison; no formal hypothesis tests were performed.
<table><tr><td></td><td></td><td colspan="3">Relative  $L ^ { 2 } \downarrow ( \times 1 0 ^ { - 2 } )$ </td><td colspan="3">Gradient  $L ^ { 1 } \downarrow ( \times 1 0 ^ { - 2 } )$ </td></tr><tr><td>PDE</td><td>Model</td><td>K =1</td><td>K=5</td><td>K = 10</td><td>K =1</td><td>K=5</td><td>K = 10</td></tr><tr><td rowspan="2">Lambda-Omega</td><td>RD-JEPA</td><td>12.24±2.58</td><td>9.85 ± 0.79</td><td>8.38±1.46</td><td>0.68 ± 0.12</td><td>0.52± 0.05</td><td>0.45± 0.08</td></tr><tr><td>Architecture-matched scratch</td><td>23.42±4.73</td><td>20.14±0.91</td><td>19.11 ±1.79</td><td>1.27±0.26</td><td>1.05±0.04</td><td>1.00 ± 0.12</td></tr><tr><td rowspan="2">Barkley</td><td>RD-JEPA</td><td>10.53±2.55</td><td>6.90 ± 0.24</td><td>5.23±0.32</td><td>1.09 ± 0.27</td><td>0.74±0.04</td><td>0.58 ± 0.04</td></tr><tr><td>Architecture-matched scratch</td><td>19.23±3.48</td><td>13.03±0.48</td><td>10.01 ± 1.12</td><td>2.01 ± 0.38</td><td>1.39± 0.06</td><td>1.10±0.12</td></tr><tr><td rowspan="2">Oregonator</td><td>RD-JEPA</td><td>11.36±1.34</td><td>4.83±0.79</td><td>4.05 ±0.47</td><td>0.47 ± 0.05</td><td>0.26 ± 0.06</td><td>0.22 ± 0.02</td></tr><tr><td>Architecture-matched scratch</td><td>22.12±2.82</td><td>9.43±1.89</td><td>7.37±1.00</td><td>0.90±0.08</td><td>0.51±0.12</td><td>0.39±0.03</td></tr></table>

## S6.2 Few-trajectory adaptation under non-periodic boundary conditions

We next assess whether a representation pretrained exclusively on periodic systems remains useful after adaptation to Barkley dynamics with homogeneous Neumann, Robin or Dirichlet boundary conditions. RD-JEPA, the no-predictive-latent control and FNO are trained separately for each boundary condition and $K \in \{ 1 , 5 , 1 0 \}$ . All models use the same support trajectories within each of the three support-set selections and are evaluated on the same 300 boundary-specific test trajectories. The RD-JEPA predictor retains its pretrained periodic latent-neighbour map, and no model receives an explicit boundary-condition label.

RD-JEPA has the lowest mean relative $L ^ { 2 }$ and gradient $L ^ { 1 }$ error in all nine boundary-condition–budget settings (Supplementary Table S15). The ordering holds for homogeneous Neumann, Robin and Dirichlet conditions and for K = 1, 5 and 10. These results extend the observed transfer to a combined boundary-condition and discretization shift within Barkley at the evaluated equation, domain, resolution and simulator. The three run-level values underlying each entry are provided in Supplementary Data 5.

Table S15. Few-trajectory Barkley forecasting under non-periodic boundary conditions. Each entry is 100× the error and reports the mean ± sample s.d. across three paired support-set selections. For each selection, errors are first averaged over the same 300 test trajectories and then over the five direct forecast horizons. The model-training random-number-generator state is fixed, and all models use the same support trajectories within each selection. Lower values are better. Boldface marks the lowest mean for each boundary condition, metric and adaptation budget.
<table><tr><td rowspan="2">Boundary condition</td><td rowspan="2">Model</td><td colspan="3">Relative  $L ^ { 2 } \downarrow ( \times 1 0 ^ { - 2 } )$ </td><td colspan="3">Gradient  $L ^ { 1 } \downarrow ( \times 1 0 ^ { - 2 } )$ </td></tr><tr><td> $K = 1$ </td><td> $K = 5$ </td><td> $K = 1 0$ </td><td> $K = 1$ </td><td> $K = 5$ </td><td> $K = 1 0$ </td></tr><tr><td rowspan="3">Neumann</td><td>RD-JEPA</td><td> ${ \bf 1 0 . 7 9 \pm 1 . 4 2 }$ </td><td> ${ \bf 7 . 3 3 \pm 1 . 3 0 }$ </td><td> ${ \pm } \mathbf { 0 . 4 9 } \pm \mathbf { 0 . 4 6 }$ </td><td> $\mathbf { 0 . 6 8 \pm 0 . 1 0 }$ </td><td> $\mathbf { 0 . 4 8 \pm 0 . 0 8 }$ </td><td> $\mathbf { 0 . 3 8 \pm 0 . 0 3 }$ </td></tr><tr><td>No predictive latent</td><td> $2 6 . 8 8 \pm 5 . 5 8$ </td><td> $1 7 . 7 8 \pm 3 . 2 4$ </td><td> $1 0 . 8 3 \pm 0 . 9 1$ </td><td> $1 . 6 9 \pm 0 . 3 3$ </td><td> $1 . 1 6 \pm 0 . 1 8$ </td><td> $0 . 7 5 \pm 0 . 0 9$ </td></tr><tr><td>FNO</td><td> $3 0 . 8 9 \pm 1 . 5 0$ </td><td> $1 9 . 1 5 \pm 2 . 3 2$ </td><td> $1 5 . 0 2 \pm 0 . 7 6$ </td><td> $2 . 0 2 \pm 0 . 0 9$ </td><td> $1 . 2 1 \pm 0 . 1 3$ </td><td> $0 . 9 7 \pm 0 . 0 4$ </td></tr><tr><td rowspan="3">Robin</td><td>RD-JEPA</td><td> ${ \bf 1 1 . 1 3 \pm 1 . 3 5 }$ </td><td> ${ \bf 7 . 2 1 \pm 1 . 2 6 }$ </td><td> ${ \bf 5 . 5 7 \pm 0 . 3 9 }$ </td><td> ${ \bf 0 . 6 9 \pm 0 . 1 0 }$ </td><td> ${ \bf 0 . 4 7 \pm 0 . 0 8 }$ </td><td> ${ \bf 0 . 3 9 \pm 0 . 0 2 }$ </td></tr><tr><td>No predictive latent</td><td> $2 5 . 6 1 \pm 4 . 4 6$ </td><td> $1 7 . 9 1 \pm 3 . 1 3$ </td><td> $1 0 . 7 9 \pm 2 . 6 7$ </td><td> $1 . 6 1 \pm 0 . 2 8$ </td><td> $1 . 1 8 \pm 0 . 1 7$ </td><td> $0 . 7 6 \pm 0 . 1 9$ </td></tr><tr><td>FNO</td><td> $3 0 . 6 1 \pm 1 . 3 1$ </td><td> $1 8 . 8 8 \pm 2 . 2 4$ </td><td> $1 4 . 9 5 \pm 0 . 6 6$ </td><td> $2 . 0 1 \pm 0 . 0 6$ </td><td> $1 . 2 0 { \pm } 0 . 1 3$ </td><td> $0 . 9 6 \pm 0 . 0 3$ </td></tr><tr><td rowspan="3">Dirichlet</td><td>RD-JEPA</td><td> ${ \bf 1 1 . 1 7 \pm 1 . 7 6 }$ </td><td> ${ \bf 7 . 4 8 \pm 1 . 9 5 }$ </td><td> ${ \pm } 0 . 2 6 \pm { \bf 0 . 2 1 }$ </td><td> ${ \bf 0 . 7 0 \pm 0 . 1 0 }$ </td><td> ${ \bf 0 . 4 9 \pm 0 . 1 2 }$ </td><td> $\mathbf { 0 . 3 6 \pm 0 . 0 0 }$ </td></tr><tr><td>No predictive latent</td><td> $2 8 . 0 4 \pm 2 . 9 1$ </td><td> $1 8 . 3 4 \pm 3 . 5 5$ </td><td> $1 1 . 1 4 \pm 1 . 0 3$ </td><td> $1 . 7 3 \pm 0 . 1 9$ </td><td> $1 . 2 1 \pm 0 . 2 2$ </td><td> $0 . 7 8 \pm 0 . 0 8$ </td></tr><tr><td>FNO</td><td> $3 1 . 7 6 \pm 1 . 0 0$ </td><td> $1 9 . 5 0 \pm 2 . 4 1$ </td><td> $1 5 . 2 9 \pm 0 . 9 2$ </td><td> $2 . 0 4 \pm 0 . 0 7$ </td><td> $1 . 2 1 \pm 0 . 1 3$ </td><td> $0 . 9 7 \pm 0 . 0 4$ </td></tr></table>