# Text-guided flow matching enables sample-eficient crystal structure generation

Wentao Li<sup>1</sup>

<sup>1</sup>Tsinghua University

## Abstract

Crystal generators can now propose periodic structures, but their control interfaces remain poorly matched to the mixed descriptors used in materials design. Text provides a compact way to combine composition, symmetry, prototype and property cues, yet it has not been clear whether such information can steer flow-based crystal generation. Here we introduce TFMat, a text-conditioned flow-matching framework that uses structured materials language as a semantic prior for a CrystalFlow generator. Across Perov-5, Carbon-24 and MP-20 crystal structure prediction benchmarks, TFMat improves one-candidate match rates over CrystalFlow and reaches a 92.04% MP-20 match rate with 20 candidates; in de novo generation, it improves element-count and density distribution alignment while retaining coarse property consistency in composition-selected outputs. These results position structured text as an inspectable control layer for translating human-readable materials intent into candidate crystals for downstream simulation and validation.

## Introduction

Generative modeling has changed the computational search for crystalline materials from enumerating known prototypes to proposing new periodic structures under chemical and functional constraints. The opportunity is large because stable crystals occupy a sparse subset of an enormous space of compositions, lattices and atomic arrangements.<sup>1</sup> High-throughput materials databases and community benchmark tools now provide the infrastructure for data-driven crystal design, <sup>2–10</sup> while graph neural networks and large-scale discovery pipelines have shown that learned atomic environments and interatomic representations can support accurate structure-property modeling. <sup>11–18</sup> The most useful generative model must therefore do more than populate a benchmark distribution: it must respond to the kinds of constraints that define an actual materials question. Such control becomes especially important when candidate structures must be ranked, relaxed and interpreted across downstream computations. The remaining challenge is not simply to generate valid crystals, but to make generation controllable in the language of materials design.

Recent crystal generative models have addressed this challenge mainly by strengthening the geometric model. Conditional graph generators and physics-guided adversarial models established that chemical and symmetry constraints can improve validity,<sup>19–21</sup> and difusion or flow-based methods then made it possible to jointly model periodic coordinates, lattices and atom types.<sup>22–27</sup> Other approaches introduced autoregressive crystal strings, language-model base distributions and large-scale property-conditioned difusion. <sup>28–30</sup> Together these methods have clarified how to respect periodic geometry, but their control variables are often chosen for algorithmic convenience rather than for expressiveness. This gap is methodological as much as architectural, because realistic constraints rarely arrive as isolated labels. In parallel, materials natural-language processing has shown that chemical entities, synthesis recipes and property relations can be extracted from text using parsers, word embeddings and scientific language models. <sup>31–38</sup> These advances suggest that crystallographic metadata and language representations could become a shared control layer between human hypotheses, databases and generative models.

This control layer is still underdeveloped. Most crystal generators are unconditional, conditioned only on composition, or driven by a small number of scalar labels. Such inputs are convenient for benchmarks, but they compress materials intent into a narrow representation: a scientist may know a formula, prototype family, space-group tendency, stability range and electronic-property expectation simultaneously, and these descriptors are often mixed across structured database fields and written descriptions. Text-guided difusion models such as TGDMat and Chemeleon have recently shown that prompt representations can influence periodic materials generation,<sup>39,40</sup> and FlowLLM has explored language models as base distributions for flow-based generation.<sup>29</sup> However, it remains unclear whether a compact, frozen materials-language prior can improve a continuous crystal flow without replacing its geometric backbone. This is a more stringent question than adding another conditioning channel: if language merely repeats formula labels, its apparent benefit may reflect reconstruction from highly informative metadata rather than useful semantic control. The distinction matters for closed-loop and agentic materials workflows, where autonomous systems must translate research intent into candidates, pass them to simulation or synthesis modules, and keep the decision trail legible to researchers.<sup>41–43</sup> A useful text interface should therefore improve sample-eficient structure recovery now, expose where prompt information is most helpful, and remain honest about the boundary between controlled generation and open-ended discovery.

Here we introduce TFMat, a text-guided extension of CrystalFlow designed to test that premise directly. TFMat keeps the flow-matching crystal generator as the central geometric model and adds a condition vector obtained from frozen MatSciBERT embeddings of structured materials prompts. The prompts are database-derived rather than free-form instructions; they include formula or composition information, symmetry descriptors and selected scalar properties, but not lattice vectors or fractional coordinates. This design isolates the contribution of the text prior more cleanly than replacing the entire generator, while keeping the task close to metadata-rich settings encountered in materials databases. We evaluate TFMat in composition-fixed crystal structure prediction (CSP) and text-conditioned de novo generation (DNG). In CSP, TFMat improves one-sample match rates over CrystalFlow on Perov-5, Carbon-24 and MP-20, with the largest gains on the constrained perovskite and carbon-network benchmarks, and the MP-20 gain persists under a 20-candidate evaluation budget. In DNG, TFMat does not dominate every validity metric, but it improves element-count and density distribution alignment and retains coarse formation-energy and band-gap cues in a composition-selected diagnostic subset. These results support a bounded conclusion: structured language descriptors can bias a flow-based crystal generator towards more relevant regions of crystal space, providing a practical interface for future materials discovery pipelines.

## Results

## Structured prompts define two generation regimes

We first ask whether text conditioning helps under benchmark conditions that separate structural recovery from open-ended generation. The evaluation uses three standard crystal-generation datasets with increasing chemical and geometric diversity. <sup>22,23,39</sup> Perov-5 is a constrained five-atom perovskite benchmark and mainly tests recovery of a well-defined prototype family. Carbon-24 contains larger carbon unit cells and tests covalent-network generation in a chemically simple but geometrically demanding setting. MP-20 is a broad Materials Project-derived benchmark of inorganic crystals with up to 20 atoms per unit cell, and is therefore the most diverse dataset considered here.

The same conditioning idea is evaluated in two regimes. In CSP, atom types and composition are fixed while the model predicts lattice parameters and fractional coordinates; this directly tests whether text improves recovery of a known target structure. In DNG, atom types, lattice and fractional coordinates are generated jointly after sampling the number of atoms, so the output is a text-conditioned crystal population rather than a single reconstruction. Throughout, TFMat receives structured prompts derived from database metadata, including formula, symmetry descriptors and selected property values where available. These prompts make the control signal reproducible, but they also define the scope of the claim: TFMat is evaluated as a structured text-conditioned generator, not as an unrestricted natural-language discovery system.

## Text conditioning improves one-sample crystal structure prediction

The clearest test of sample eficiency is the one-sample CSP regime, where each test composition receives only one generated candidate. If the text condition is useful, it should increase the chance that this single trajectory enters the correct structural basin. Table 1 shows that TFMat does so consistently. Relative to CrystalFlow, the match rate increases from 53.69% to 91.33% on Perov-5, from 15.02% to 46.40% on Carbon-24 and from 67.65% to 77.80% on MP-20.

This improvement is not confined to the easiest benchmark. Among the methods shown in Table 1, TFMat gives the highest one-sample match rate on all three datasets. The Carbon-24 result is particularly informative because the chemistry is fixed but the covalent networks are geometrically diverse; TFMat raises match rate by 31.38 percentage points and lowers RMSE from 0.3095 to 0.1353. On MP-20, TFMat improves structural retrieval even though its RMSE for matched structures is not the lowest. Representative matched structures are shown in Fig. 1. The result therefore points to a specific role for text: it helps select plausible basins of crystal space, while final geometric refinement remains a separate challenge.

Table 1: Single-sample CSP comparison across Perov-5, Carbon-24 and MP-20. Best values are bold and second-best values are underlined. Published baseline values are from TGDMat.<sup>39</sup> Match rates are percentages; lower RMSE is better.
<table><tr><td>Method</td><td>Perov-5 Match Rate ↑</td><td>Perov-5 RMSE ↓</td><td>Carbon-24 Match Rate ↑</td><td>Carbon-24 RMSE ↓</td><td>MP-20 Match Rate ↑</td><td>MP-20 RMSE ↓</td></tr><tr><td>P-cG-SchNet</td><td>48.22</td><td>0.4179</td><td>17.29</td><td>0.3846</td><td>15.39</td><td>0.3762</td></tr><tr><td>CDVAE</td><td>45.31</td><td>0.1138</td><td>17.09</td><td>0.2969</td><td>33.90</td><td>0.1045</td></tr><tr><td>DiffCSP</td><td>52.02</td><td>0.0760</td><td>17.54</td><td>0.2759</td><td>51.49</td><td>0.0631</td></tr><tr><td>TGDMat (Short)</td><td>56.54</td><td>0.0583</td><td>24.13</td><td>0.2424</td><td>52.22</td><td>0.0597</td></tr><tr><td>TGDMat (Long)</td><td>90.46</td><td>0.0203</td><td>44.63</td><td>0.2266</td><td>55.15</td><td>0.0572</td></tr><tr><td>CrystalFlow</td><td>53.69</td><td>0.0953</td><td>15.02</td><td>0.3095</td><td>67.65</td><td>0.0791</td></tr><tr><td>TFMat (Ours)</td><td>91.33</td><td>0.0449</td><td>46.40</td><td>0.1353</td><td>77.80</td><td>0.0868</td></tr></table>

![](images/0e1bea6dfaec84c167591104277be19bf75448f2ba2cd2f60026ac97c1f77801.jpg)  
Figure 1: Representative crystal structure predictions from TFMat. Each pair shows a ground-truth crystal (left) and a TFMat prediction (right). Atom species are distinguished by color and unit cells are outlined in black.

## Repeated sampling confirms the MP-20 retrieval gain

Repeated sampling tests whether the one-sample gain persists when the model is allowed to propose multiple candidates. In the 20-candidate MP-20 evaluation, a structure is counted as matched if at least one candidate matches the reference. TFMat again gives the highest match rate among the compared methods, reaching 92.04% compared with 85.38% for CrystalFlow and 82.02% for TGDMat (Long) (Table 2).

The RMSE ranking defines the boundary of this claim. TGDMat (Short) reports the lowest RMSE at 0.0443, followed by TGDMat (Long) at 0.0483 and DifCSP at 0.0492. TFMat therefore contributes most strongly to retrieval of the correct basin, rather than to the final geometric refinement of matched candidates. This distinction is practically important: in screening workflows, a candidate in the right basin can be passed to relaxation, whereas relaxation cannot rescue a sample that never reaches a relevant structural family.

Table 2: MP-20 CSP comparison with 20 candidates. Best values are bold and second-best values are underlined. Published baseline values are from TGDMat. <sup>39</sup> Match rates are percentages.
<table><tr><td>Method</td><td>MP-20 Match Rate ↑</td><td>MP-20 RMSE ↓</td></tr><tr><td>P-cG-SchNet</td><td>32.64</td><td>0.3018</td></tr><tr><td>CDVAE</td><td>66.95</td><td>0.1026</td></tr><tr><td>DiffCSP</td><td>77.93</td><td>0.0492</td></tr><tr><td>TGDMat (Short)</td><td>80.97</td><td>0.0443</td></tr><tr><td>TGDMat (Long)</td><td>82.02</td><td>0.0483</td></tr><tr><td>CrystalFlow</td><td>85.38</td><td>0.0614</td></tr><tr><td>TFMat (Ours)</td><td>92.04</td><td>0.0665</td></tr></table>

## Text-conditioned de novo generation reshapes generated populations

The DNG task is harder because the model must generate composition, lattice and coordinates jointly. Here the question changes from exact recovery to whether structured text can shape the generated population. TFMat’s main efect in this setting is distributional rather than a uniform increase in every validity metric.

Relative to CrystalFlow, TFMat reduces element-count EMD from 0.2489 to 0.1990 and density EMD from 0.1701 to 0.0794, while retaining 99.06% structural validity and 99.67% coverage precision (Table 3). Representative generations are shown in Fig. 2. These changes indicate that the language condition biases population-level statistics without degrading geometric plausibility. The remaining bottleneck is atom-type generation, as reflected by weaker compositional validity than the unconditional baseline and the strongest published text-guided difusion result.

Table 3: MP-20 text-conditioned de novo generation. Values are percentages except for EMD metrics, where lower is better.
<table><tr><td>Model</td><td>Comp. valid ↑</td><td>Struct. valid ↑</td><td>COV-R ↑</td><td>COV-P ↑</td><td># Elem. EMD ↓</td><td>Density EMD ↓</td></tr><tr><td>CrystalFlow</td><td>83.21</td><td>98.74</td><td>99.45</td><td>99.76</td><td>0.2489</td><td>0.1701</td></tr><tr><td>TFMat (Ours)</td><td>81.32</td><td>99.06</td><td>98.91</td><td>99.67</td><td>0.1990</td><td>0.0794</td></tr></table>

![](images/a2753ef2a59b2d112d1865cce8e98ea360fcb20cc702c8a29dbc4e575586dc79.jpg)  
$Z r _ { 2 } { \mathsf { V C o } } _ { 3 }$ is Frank-Kasper � Phase-derived structured and crystallizes in the trigonal R-3m space group. Its formation energy per atom is -0.3281. Its band gap is 0.0. Its space group number is 166. Its $\mathsf { E } _ { \mathsf { h u l l } }$ is 0.0.

![](images/aa5a3a1c9fa19a7b442292f22fefbd9c5a53d08241b8a14ef3263cb9a1e56a52.jpg)  
Tb<sub>2</sub>HgTl is Heusler structured and crystallizes in the cubic Fm-3m space group. Its formation energy per atom is -0.4198. Its band gap is 0.0. Its space group number is 225. Its $\mathsf { E } _ { \mathsf { h u l l } }$ is 0.0.

![](images/ea8e79f8e0c3000c4f2a851a69d0a749c0a7b484fb628c47ccc5f3c7580a9e74.jpg)  
TmIn<sub>2</sub>Sn is beta Cu<sub>3</sub>Ti-derived structured and crystallizes in the tetragonal P4/mmm space group. Its formation energy per atom is -0.3662. Its band gap is 0.0. Its space group number is 123. Its $\mathsf { E } _ { \mathsf { h u l l } }$ is 0.0151.

![](images/0a8f980c1198c459554639f78c87e01ca6142406b414e5c7e27dc02ca3510f54.jpg)  
Yb<sub>2</sub>TlIn is Heusler structured and crystallizes in the cubic Fm-3m space group. Its formation energy per atom is -0.4857. Its band gap is 0.0. Its space group number is 225. Its $\mathsf { E } _ { \mathsf { h u l l } }$ is 0.0067.

![](images/8f40e4a0fc0eed44f81cb38e90cfac33841ade24d1cbebc5caa6702ba11e01fb.jpg)

![](images/158fd9d207a9b993e56afa73e0bc294a60ba8394514a68d7fd667d1fec1104d9.jpg)  
Pr<sub>3</sub>In is Uranium Silicide structured and crystallizes in the cubic Pm-3m space group. Its formation energy per atom is -0.2732. Its band gap is 0.0. Its space group number is 221. Its $\mathsf { E } _ { \mathsf { h u l l } }$ is 0.0.

CsBr is Halite, Rock Salt structured and crystallizes in the cubic Fm-3m space group. Its formation energy per atom is -2.0529. Its band gap is 4.4247. Its space group number is 225. Its $\mathsf { E } _ { \mathsf { h u l l } }$ is 0.0.

Figure 2: Representative text-conditioned de novo generations. Each pair shows a reference MP-20 crystal (left) and a TFMat-generated crystal (right) conditioned on the corresponding structured prompt.

## Surrogate property diagnostics support composition-selected consistency

The MP-20 prompts also contain formation-energy and band-gap cues, so we examined whether these coarse signals survive generation when composition is reasonably consistent. This is a surrogate diagnostic rather than a population-wide property benchmark. A composition-selected subset of 2,000 TFMat DNG candidates was evaluated with CGCNN regressors trained on MP-20 formation energy and band gap.<sup>12</sup> The selection procedure, element-match distribution and outlier rule are reported in Supplementary Note 4.

Within this selected subset, CGCNN predictions were available for 1,994 structures. Formation-energy sign agreement is 95.94% (1,913/1,994). After removing 41 numerical outliers with absolute formation-energy error above 10 eV $\mathrm { a t o m } ^ { - 1 }$ , the cleaned formation-energy MAE is 0.183 eV $\mathrm { a t o m } ^ { - 1 }$ , the median absolute error is 0.114 eV atom<sup>−1</sup> and Pearson $r = 0 . 9 6 2$ (Fig. 3). Band-gap zero/nonzero type agreement is 83.65%, with MAE 0.419 eV, median absolute error 0.018 eV and Pearson $r = 0 . 8 1 7$ . These values support prompt-level consistency for the composition-selected subset, especially for formation-energy scale. They do not establish DFT stability, synthesizability or unfiltered population-wide property control.

![](images/b1a3d93d01a8bcf70b8d639542398ea9fce96d85a4e58a9d5e9d1c0f98d39fd7.jpg)

b  
![](images/ddb2e35e635f683d2fa9de4b7002562c87961d9bebf146296ccd5fba335a24c6.jpg)

![](images/893da1ccbfeff0fb5cf2c9aef5ca793ee23b52ea89aa3d17a4179c5ca46146f4.jpg)

d  
![](images/1e5419fe2d50d63a1b625fb489f9096a857fbcae200418f7a1656f087cfc271c.jpg)  
Figure 3: Surrogate property diagnostics for the composition-selected subset. Kernel density estimates and scatter panels compare reference prompt values with CGCNN-predicted properties for selected TFMat outputs.

## Graph-embedding diagnostics support the population shift

The population-level picture is consistent with this interpretation. We embedded MP-20 test crystals and generated crystals with a pretrained graph encoder, then projected each generated set together with the MP-20 test set using t-SNE (Fig. 4). The unconditional CrystalFlow baseline and TFMat both cover broad regions of the test distribution, as expected from their shared geometric backbone. The text-conditioned run nevertheless shows a modest improvement in the per-panel histogram diagnostic, with Jensen-Shannon divergence decreasing from 0.6367 to 0.6200. A separate shared-coordinate diagnostic gives the same direction of change, with coverage recall increasing from 0.453 to 0.473, manifold precision increasing from 0.500 to 0.512 and shared t-SNE Jensen-Shannon divergence decreasing from 0.435 to 0.429 (Supplementary Table S7).

The visualization is not a definitive metric because t-SNE distances are projection-dependent and the two displayed panels are projected separately. Its value is instead as a consistency check on the DNG statistics. TFMat changes the generated population most clearly through marginal distribution alignment, not through a large jump in validity, and the generated points remain spread across multiple clusters. This reduces the likelihood that the improved EMD values arise from a visually obvious collapse to a single prototype.

![](images/31caf3060c4e5bf778ba007080c6789c146480e721a0a447e241f836a313fab9.jpg)

![](images/690f5cb2b5f73c06966572444deb8e821eccc14c4f96d3a960ad24539246aaf1.jpg)  
Figure 4: t-SNE diagnostic for unconditional and text-conditioned MP-20 generation. Left: unconditional DNG baseline. Right: text-conditioned DNG. Each panel contains 9,046 MP-20 test structures and 10,000 generated structures and uses its own t-SNE projection.

## Discussion

TFMat supports a bounded conclusion: structured materials text can steer a flow-matching crystal generator towards more useful regions of crystal space. The strongest evidence is the one-sample CSP result, where the text-conditioned model improves match rate across Perov-5, Carbon-24 and MP-20 while retaining CrystalFlow as the central geometric generator. This is the regime in which semantic priors should be most visible, because the model has only one chance to choose a plausible structural basin.

The comparison with TGDMat clarifies what kind of progress this represents. TGDMat remains a strong text-guided difusion baseline, particularly for RMSE and MP-20 DNG compositional validity. TFMat is therefore not a universal replacement for difusion-based text guidance. Its contribution is complementary: a compact language condition can be coupled to a continuous flow-matching generator and can improve sample-eficient structural retrieval. The MP-20 pattern, where match rate improves more clearly than RMSE, suggests that the current model is better at reaching relevant basins than at completing final geometric refinement.

The DNG and diagnostic results extend, but also limit, this claim. Improvements in element-count and density EMDs, a modest shift in graph-embedding overlap, and composition-selected formation-energy and band-gap consistency all point in the same direction: structured text changes the generated distribution in ways that are not captured by validity alone. At the same time, compositional validity trails the best published text-guided difusion result, and the property analysis depends on CGCNN surrogates and a selected subset. These observations make atom-type generation, exact prompt adherence and unfiltered property consistency the most important technical bottlenecks.

The present prompts also define an important scope boundary. They are structured database-derived descriptions that include formula or composition-level information, space-group descriptors and scalar property values. The results therefore demonstrate controlled metadata-conditioned generation, not independent discovery from unconstrained natural language. The MP-20 prompt-control experiments support a promptcontent efect, but broader field ablations, unseen-prototype tests and stronger out-of-distribution prompt checks are needed to distinguish semantic generalization from reconstruction using highly informative prompts. Density-functional relaxation and unfiltered-population analysis will also be required before making claims about thermodynamic stability of newly generated materials.<sup>44,45</sup>

Within these limits, the result suggests a practical design principle for future crystal generators. Text should not be treated only as annotation attached after generation; when embedded and fused into a geometric flow, structured materials descriptions can become an actionable prior over the search trajectory. This connects the benchmark result to larger closed-loop discovery workflows, where human hypotheses, database records, language-model planners and autonomous platforms need a shared representation of intent. TFMat is one step in that direction: it improves a current crystal-generation task while preserving an interface that can be inspected, edited and reused by both researchers and automated materials-science systems.

## Methods

## Datasets and text conditions

We evaluated TFMat on Perov-5, Carbon-24 and MP-20, following the benchmark splits used by CDVAE, DifCSP and TGDMat. <sup>22,23,39</sup> In CSP, the atom types and composition are fixed and the model predicts lattice and fractional coordinates. In DNG, the model generates atom types, lattice and fractional coordinates jointly after sampling the number of atoms from the empirical training distribution.

Text conditions were generated from structured metadata fields associated with each crystal record. For MP-20, these prompts include formula-level identity, crystal-system or space-group information, formation energy, band gap and energy above hull when available. MatSciBERT embeddings were precomputed with mean pooling before training. The model therefore receives a structured semantic condition, not raw coordinates.

## Flow-matching formulation for CSP

TFMat follows the flow-matching view of generative modeling, in which a neural network learns a velocity field that transports samples from a simple source distribution to the data distribution. <sup>46–49</sup> This deterministic transport perspective is complementary to denoising difusion and score-based generative modeling. <sup>50–52</sup> For a crystal with � atoms and fixed atom types $\mathbf { a } = \left( a _ { 1 } , \ldots , a _ { N } \right)$ , the generated state is

$$
\begin{array} { r } { { \bf x } = ( { \bf k } , { \bf F } ) , } \end{array}\tag{1}
$$

where $\textbf { k } \in \ \mathbb { R } ^ { 6 }$ is a six-dimensional lattice-polar representation and $\mathbf { F } \in \lbrack 0 , 1 ) ^ { N \times 3 }$ contains fractional coordinates. The target crystal is $\mathbf { x } _ { 1 } = ( \mathbf { k } _ { 1 } , \mathbf { F } _ { 1 } )$ . We sample a source state

$$
{ \bf k } _ { 0 } \sim { \cal N } ( { \bf 0 } , \sigma _ { k } ^ { 2 } { \bf I } ) , \qquad { \bf F } _ { 0 } \sim \mathcal { U } ( [ 0 , 1 ) ^ { N \times 3 } ) .\tag{2}
$$

Fractional-coordinate displacement is wrapped with the minimum-image convention,

$$
\Delta \mathbf { F } = \left( \left( \mathbf { F } _ { 1 } - \mathbf { F } _ { 0 } - 0 . 5 \right) \mathrm { m o d } \ 1 \right) - 0 . 5 .\tag{3}
$$

For $t \sim \mathcal { U } ( 0 , 1 )$ , the interpolation is

$$
{ \bf k } _ { t } = { \bf k } _ { 0 } + t ( { \bf k } _ { 1 } - { \bf k } _ { 0 } ) , \qquad { \bf F } _ { t } = { \bf F } _ { 0 } + t \Delta { \bf F } .\tag{4}
$$

The target velocities are

$$
\begin{array} { r } { \mathbf { v } _ { k } ^ { \star } = \mathbf { k } _ { 1 } - \mathbf { k } _ { 0 } , \qquad \mathbf { v } _ { F } ^ { \star } = \Delta \mathbf { F } . } \end{array}\tag{5}
$$

The network predicts $( \hat { \mathbf { v } } _ { k } , \hat { \mathbf { v } } _ { F } )$ from $( \mathbf { k } _ { t } , \mathbf { F } _ { t } , \mathbf { a } , t , \mathbf { c } )$ , where c is the projected text condition. The training loss is

$$
\mathcal { L } _ { \mathrm { C S P } } = \lambda _ { k } \Vert \hat { \mathbf { v } } _ { k } - \mathbf { v } _ { k } ^ { \star } \Vert _ { 2 } ^ { 2 } + \lambda _ { F } \sum _ { i = 1 } ^ { N } \Vert \hat { \mathbf { v } } _ { F , i } - \mathbf { v } _ { F , i } ^ { \star } \Vert _ { 2 } ^ { 2 } .\tag{6}
$$

## Text-conditioned sampling

Sampling starts from the source state and integrates the learned velocity field with Euler updates. With � integration steps and $\Delta t = 1 / T$

$$
{ \bf k } _ { t + \Delta t } = { \bf k } _ { t } + \Delta t \hat { \bf v } _ { k } , \qquad { \bf F } _ { t + \Delta t } = \left( { \bf F } _ { t } + \Delta t \hat { \bf v } _ { F } \right) \bmod 1 .\tag{7}
$$

For text-conditioned models, the default conditional sampling factor is 1.0. Thus, the main reported TFMat runs should be interpreted as text-conditioned flow sampling. Additional guidance-factor sweeps vary the interpolation between conditional and unconditional decoder calls, in the broad spirit of classifier-free guidance,<sup>53</sup> but the present manuscript does not claim an isolated guidance contribution.

## Extension to de novo generation

For DNG, the state includes a relaxed atom-type representation,

$$
\mathbf { x } = ( \mathbf { Q } , \mathbf { k } , \mathbf { F } ) ,\tag{8}
$$

where $\mathbf { Q } \in \mathbb { R } ^ { N \times d _ { t } }$ is decoded to discrete atom types at the end of sampling. The trajectory is

$$
{ \bf Q } _ { t } = { \bf Q } _ { 0 } + t ( { \bf Q } _ { 1 } - { \bf Q } _ { 0 } ) , \quad { \bf k } _ { t } = { \bf k } _ { 0 } + t ( { \bf k } _ { 1 } - { \bf k } _ { 0 } ) , \quad { \bf F } _ { t } = { \bf F } _ { 0 } + t \Delta { \bf F } .\tag{9}
$$

The model predicts the joint velocity field $( \hat { \mathbf { v } } _ { Q } , \hat { \mathbf { v } } _ { k } , \hat { \mathbf { v } } _ { F } )$ , and the DNG training loss is

$$
\mathcal { L } _ { \mathrm { D N G } } = \lambda _ { Q } \Vert \hat { \mathbf { v } } _ { Q } - \mathbf { v } _ { Q } ^ { \star } \Vert _ { 2 } ^ { 2 } + \lambda _ { k } \Vert \hat { \mathbf { v } } _ { k } - \mathbf { v } _ { k } ^ { \star } \Vert _ { 2 } ^ { 2 } + \lambda _ { F } \sum _ { i = 1 } ^ { N } \Vert \hat { \mathbf { v } } _ { F , i } - \mathbf { v } _ { F , i } ^ { \star } \Vert _ { 2 } ^ { 2 } .\tag{10}
$$

## Evaluation metrics

CSP match rate and RMSE were computed with the pymatgen StructureMatcher<sup>7</sup> using a site tolerance of 0.5, an angle tolerance of $1 0 ^ { \circ }$ and a lattice tolerance of 0.3. In the 20-candidate evaluation, a test case is counted as matched if at least one generated candidate matches the reference. DNG metrics follow the CDVAE/DifCSP/TGDMat protocol: compositional validity, structural validity, coverage recall (COV-R), coverage precision (COV-P), and Earth mover’s distances for property or distribution statistics.<sup>22,23,39</sup> All percentages in tables are reported on a 0–100 scale.

## Visualization and property diagnostics

The CSP and DNG qualitative panels were assembled from ASE renderings. The t-SNE diagnostics use pretrained graph-encoder embeddings of MP-20 test and generated structures. Each displayed t-SNE panel contains 9,046 test structures and 10,000 generated structures; a separate shared-coordinate t-SNE analysis is reported in the Supplementary Information. The property diagnostic used CGCNN regressors trained on MP-20 formation energy and band gap. The diagnostic subset was selected by ranking generated structures by element-composition match score; it is therefore a targeted consistency analysis rather than an unbiased estimate over all generated crystals.

## Data availability

The Perov-5, Carbon-24 and MP-20 benchmark datasets are publicly available through the CDVAE benchmark resources.<sup>22</sup> The derived text annotations, precomputed MatSciBERT embeddings, model checkpoints, generated structures and evaluation outputs are available from the TFMat dataset archive at https:// huggingface.co/datasets/littlepeachs/TFMat.

## Code availability

The TFMat implementation and the scripts required to reproduce the reported generation, evaluation and figures are available at https://github.com/littlepeachs/TFMat.

## Competing interests

The authors declare no competing interests.

## References

[1] Oganov, A. R., Pickard, C. J., Zhu, Q. & Needs, R. J. Structure prediction drives materials discovery. Nature Reviews Materials 4, 331–348 (2019). https://doi.org/10.1038/s41578-019-0101-8

[2] Jain, A. et al. Commentary: The Materials Project: a materials genome approach to accelerating materials innovation. APL Materials 1, 011002 (2013). https://doi.org/10.1063/1.4812323

[3] Curtarolo, S. et al. AFLOW: An automatic framework for high-throughput materials discovery. Computational Materials Science 58, 218–226 (2012). https://doi.org/10.1016/j.commatsci.2012. 02.005

[4] Kirklin, S. et al. The Open Quantum Materials Database (OQMD): assessing the accuracy of DFT formation energies. npj Computational Materials 1, 15010 (2015). https://doi.org/10.1038/ npjcompumats.2015.10

[5] Draxl, C. & Schefler, M. The NOMAD laboratory: from data sharing to artificial intelligence. Journal ofPhysics: Materials 2, 036001 (2019). https://doi.org/10.1088/2515-7639/ab13bb

[6] Choudhary, K. et al. The joint automated repository for various integrated simulations (JARVIS) for data-driven materials design. npj Computational Materials 6, 173 (2020). https://doi.org/10. 1038/s41524-020-00440-1

[7] Ong, S. P. et al. Python Materials Genomics (pymatgen): a robust, open-source Python library for materials analysis. Computational Materials Science 68, 314–319 (2013). https://doi.org/10. 1016/j.commatsci.2012.10.028

[8] Ward, L., Agrawal, A., Choudhary, A. & Wolverton, C. A general-purpose machine learning framework for predicting properties of inorganic materials. npj Computational Materials 2, 16028 (2016). https: //doi.org/10.1038/npjcompumats.2016.28

[9] Ward, L. et al. Matminer: An open source toolkit for materials data mining. Computational Materials Science 152, 60–69 (2018). https://doi.org/10.1016/j.commatsci.2018.05.018

[10] Dunn, A. et al. Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm. npj Computational Materials 6, 138 (2020). https://doi.org/ 10.1038/s41524-020-00406-3

[11] Butler, K. T., Davies, D. W., Cartwright, H., Isayev, O. & Walsh, A. Machine learning for molecular and materials science. Nature 559, 547–555 (2018). https://doi.org/10.1038/s41586-018-0337-2

[12] Xie, T. & Grossman, J. C. Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties. Physical Review Letters 120, 145301 (2018). https://doi.org/10. 1103/PhysRevLett.120.145301

[13] Schutt, K. T. et al. SchNet: A deep learning architecture for molecules and materials. The Journal of Chemical Physics 148, 241722 (2018). https://doi.org/10.1063/1.5019779

[14] Chen, C., Ye, W., Zuo, Y., Zheng, C. & Ong, S. P. Graph networks as a universal machine learning framework for molecules and crystals. Chemistry ofMaterials 31, 3564–3572 (2019). https://doi. org/10.1021/acs.chemmater.9b01294

[15] Choudhary, K. & DeCost, B. Atomistic line graph neural network for improved materials property predictions. npj Computational Materials 7, 185 (2021). https://doi.org/10.1038/ s41524-021-00650-1

[16] Chen, C. & Ong, S. P. A universal graph deep learning interatomic potential for the periodic table. Nature Computational Science 2, 718–728 (2022). https://doi.org/10.1038/s43588-022-00349-3

[17] Deng, B. et al. CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling. Nature Machine Intelligence 5, 1031–1041 (2023). https://doi.org/10.1038/ s42256-023-00716-3

[18] Merchant, A. et al. Scaling deep learning for materials discovery. Nature 624, 80–85 (2023). https: //doi.org/10.1038/s41586-023-06735-9

[19] Gebauer, N. W. A., Gastegger, M., Hessmann, S. S. P., Muller, K.-R. & Schutt, K. T. Inverse design of 3D molecular structures with conditional generative neural networks. Nature Communications 13, 973 (2022). https://doi.org/10.1038/s41467-022-28526-y

[20] Kim, S., Noh, J., Gu, G. H., Aspuru-Guzik, A. & Jung, Y. Generative adversarial networks for crystal structure prediction. ACS Central Science 6, 1412–1420 (2020). https://doi.org/10.1021/ acscentsci.0c00426

[21] Zhao, Y. et al. Physics guided deep learning for generative design of crystal materials with symmetry constraints. npj Computational Materials 9, 104 (2023). https://doi.org/10.1038/ s41524-023-00987-9

[22] Xie, T., Fu, X., Ganea, O.-E., Barzilay, R. & Jaakkola, T. Crystal difusion variational autoencoder for periodic material generation. International Conference on Learning Representations (2022). https: //openreview.net/forum?id=03RLpj-tc\_

[23] Jiao, R. et al. Crystal structure prediction by joint equivariant difusion. Advances in Neural Information Processing Systems 36 (2023). https://openreview.net/forum?id=DNdN26m2Jk

[24] Luo, Y., Liu, C. & Ji, S. Towards symmetry-aware generation of periodic materials. Advances in Neural Information Processing Systems 36 (2023). https://openreview.net/forum?id=Jkc74vn1aZ

[25] Luo, X. et al. Deep learning generative model for crystal structure prediction. npj Computational Materials 10, 254 (2024). https://doi.org/10.1038/s41524-024-01443-y

[26] Miller, B. K., Chen, R. T. Q., Sriram, A. & Wood, B. M. FlowMM: generating materials with Riemannian flow matching. International Conference on Machine Learning (2024). https://openreview.net/ forum?id=W4pB7VbzZI

[27] Luo, X., Wang, Z., Wang, Q. et al. CrystalFlow: a flow-based generative model for crystalline materials. Nature Communications 16, 9267 (2025). https://doi.org/10.1038/s41467-025-64364-4

[28] Antunes, L. M., Butler, K. T. & Grau-Crespo, R. Crystal structure generation with autoregressive large language modeling. Nature Communications 15, 10570 (2024). https://doi.org/10.1038/ s41467-024-54639-7

[29] Sriram, A., Miller, B. K., Chen, R. T. Q. & Wood, B. M. FlowLLM: flow matching for material generation with large language models as base distributions. Advances in Neural Information Processing Systems (2024). https://openreview.net/forum?id=0bFXbEMz8e

[30] Zeni, C. et al. A generative model for inorganic materials design. Nature 639, 624–632 (2025). https://doi.org/10.1038/s41586-025-08628-5

[31] Swain, M. C. & Cole, J. M. ChemDataExtractor: a toolkit for automated extraction of chemical information from the scientific literature. Journal ofChemical Information and Modeling 56, 1894–1904 (2016). https://doi.org/10.1021/acs.jcim.6b00207

[32] Kononova, O. et al. Text-mined dataset of inorganic materials synthesis recipes. Scientific Data 6, 203 (2019). https://doi.org/10.1038/s41597-019-0224-1

[33] Tshitoyan, V. et al. Unsupervised word embeddings capture latent knowledge from materials science literature. Nature 571, 95–98 (2019). https://doi.org/10.1038/s41586-019-1335-8

[34] Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. BERT: pre-training of deep bidirectional transformers for language understanding. Proceedings ofNAACL-HLT 4171–4186 (2019). https://doi.org/10. 18653/v1/N19-1423

[35] Beltagy, I., Lo, K. & Cohan, A. SciBERT: a pretrained language model for scientific text. Proceedings of EMNLP-IJCNLP 3615–3620 (2019). https://doi.org/10.18653/v1/D19-1371

[36] Gupta, T., Zaki, M., Krishnan, N. M. A. & Mausam. MatSciBERT: a materials domain language model for text mining and information extraction. npj Computational Materials 8, 102 (2022). https: //doi.org/10.1038/s41524-022-00784-w

[37] Shetty, P. et al. A general-purpose material property data extraction pipeline from large polymer corpora using natural language processing. npj Computational Materials 9, 52 (2023). https://doi.org/10. 1038/s41524-023-01003-w

[38] Song, Y., Jain, A., Ceder, G. & Ong, S. P. MatSci-NLP: evaluating scientific language models on materials science language tasks using text-to-schema modeling. Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (2023). https://aclanthology.org/ 2023.acl-long.609/

[39] Das, K., Khastagir, S., Goyal, P., Lee, S.-C., Bhattacharjee, S. & Ganguly, N. Periodic materials generation using text-guided joint difusion model. International Conference on Learning Representations (2025). https://proceedings.iclr.cc/paper\_files/paper/2025/file/ 24e4e3234178a836b70e0aa48827e0ff-Paper-Conference.pdf

[40] Park, H. S., Onwuli, A. & Walsh, A. Exploration of crystal chemical space using text-guided generative artificial intelligence. Nature Communications 16, 4379 (2025). https://doi.org/10.1038/ s41467-025-59636-y

[41] MacLeod, B. P. et al. Self-driving laboratory for accelerated discovery of thin-film materials. Science Advances 6, eaaz8867 (2020). https://doi.org/10.1126/sciadv.aaz8867

[42] Szymanski, N. J. et al. An autonomous laboratory for the accelerated synthesis of novel materials. Nature 624, 86–91 (2023). https://doi.org/10.1038/s41586-023-06734-w

[43] Boiko, D. A., MacKnight, R., Kline, B. & Gomes, G. Autonomous chemical research with large language models. Nature 624, 570–578 (2023). https://doi.org/10.1038/s41586-023-06792-0

[44] Hohenberg, P. & Kohn, W. Inhomogeneous electron gas. Physical Review 136, B864–B871 (1964). https://doi.org/10.1103/PhysRev.136.B864

[45] Kohn, W. & Sham, L. J. Self-consistent equations including exchange and correlation efects. Physical Review 140, A1133–A1138 (1965). https://doi.org/10.1103/PhysRev.140.A1133

[46] Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M. & Le, M. Flow matching for generative modeling. International Conference on Learning Representations (2023). https://openreview.net/forum? id=PqvMRDCJT9t

[47] Liu, X., Gong, C. & Liu, Q. Flow straight and fast: learning to generate and transfer data with rectified flow. International Conference on Learning Representations (2023). https://openreview.net/ forum?id=XVjTT1nw5z

[48] Albergo, M. S. & Vanden-Eijnden, E. Building normalizing flows with stochastic interpolants. International Conference on Learning Representations (2023). https://openreview.net/forum?id= li7qeBbCR1t

[49] Tong, A. et al. Improving and generalizing flow-based generative models with minibatch optimal transport. Transactions on Machine Learning Research (2024). https://openreview.net/forum? id=CD9Snc73AW

[50] Ho, J., Jain, A. & Abbeel, P. Denoising difusion probabilistic models. Advances in Neural Information Processing Systems 33, 6840–6851 (2020). https://proceedings.neurips.cc/paper/2020/ hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html

[51] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S. & Poole, B. Score-based generative modeling through stochastic diferential equations. International Conference on Learning Representations (2021). https://openreview.net/forum?id=PxTIG12RRHS

[52] Karras, T., Aittala, M., Aila, T. & Laine, S. Elucidating the design space of difusion-based generative models. Advances in Neural Information Processing Systems 35, 26565–26577 (2022). https:// openreview.net/forum?id=k7FuTOWMOc7

[53] Ho, J. & Salimans, T. Classifier-free difusion guidance. NeurIPS Workshop on Deep Generative Models and Downstream Applications (2021). https://openreview.net/forum?id=qw8AKxfYbI

## Supplementary Information

## Supplementary Note 1: Dataset description

The benchmark tasks use the standard Perov-5, Carbon-24 and MP-20 crystal-generation splits adopted by prior crystal generative models. Perov-5 is a chemically constrained perovskite benchmark, Carbon-24 tests covalent carbon-network generation with larger unit cells, and MP-20 is the broadest benchmark used here, covering inorganic Materials Project structures with up to 20 atoms per unit cell. Table S1 reports the split sizes used for evaluation.

The two evaluation regimes use the same data resources but impose diferent generation constraints. In CSP, atom types and composition are fixed and the model predicts lattice parameters and fractional coordinates. In DNG, the model generates atom types, lattice and fractional coordinates jointly after sampling the number of atoms from the empirical training distribution. The MP-20 t-SNE diagnostic uses 9,046 held-out structures, which is why the visualization count difers from the 8,000-sample CSP evaluation split.

Table S1: Benchmark split sizes and conditioning fields. The MP-20 t-SNE diagnostic uses 9,046 held-out structures.
<table><tr><td>Dataset</td><td>Train</td><td>Validation</td><td>Test</td><td>Text fields used for conditioning</td></tr><tr><td>Perov-5</td><td>11,356</td><td>3,787</td><td>3,785</td><td>Formula, composition, prototype and symmetry-derived descriptors where available</td></tr><tr><td>Carbon-24</td><td>6,090</td><td>2,032</td><td>2,030</td><td>Formula, carbon framework descriptors and symmetry-derived descrip- tors where available</td></tr><tr><td>MP-20</td><td>27,131</td><td>9,046</td><td>8,000</td><td>Formula, crystal system or space group, formation energy, band gap and energy above hull where available</td></tr></table>

## Supplementary Note 2: Model and method details

TFMat keeps the CrystalFlow flow-matching backbone as the central generator and adds a structured semantic condition derived from database text. This design makes the comparison with CrystalFlow interpretable: the main architectural change is the conditioning pathway rather than a replacement of the geometric transport model. The condition vector is injected into the network as a fixed materials-language representation projected into the model conditioning space.

The text conditions used in this work are generated from structured database fields rather than manually written prose. This choice makes the conditioning channel reproducible and keeps the comparison with CrystalFlow interpretable. Each prompt is assembled from fields that a materials scientist would plausibly use when specifying a target: formula or composition, prototype or structural family when available, crystal system or space-group descriptors, and selected scalar property values. The prompts intentionally do not contain fractional coordinates or lattice vectors.

Each structured prompt is paired with a MatSciBERT representation. Embeddings are computed with a frozen MatSciBERT encoder followed by mean pooling over token representations. The resulting vector is projected into the TFMat conditioning space by a small multilayer perceptron before being fused with the flow-matching backbone. Freezing the text encoder reduces the number of trainable parameters and makes the experiment closer to a controlled test of whether existing materials-language representations can act as useful priors for crystal generation.

Table S2 summarizes the metrics used across CSP and DNG. The match-rate and RMSE metrics are reference-based and therefore apply most directly to CSP, where a target structure is known for each composition. The DNG metrics are population-level diagnostics: they measure validity, coverage and distributional similarity, but they do not by themselves establish exact prompt recovery, thermodynamic stability or synthesizability.

Table S2: Metric definitions and interpretation. The table separates reference-based CSP metrics from population-level DNG diagnostics.
<table><tr><td>Metric</td><td>What it measures</td><td>Main limitation</td></tr><tr><td>CSP match rate</td><td>Fraction of test cases for which at least one generated candidate matches the reference under StructureMatcher thresholds</td><td>Sensitive to matcher thresholds and to the number of generated candidates</td></tr><tr><td>CSP RMSE</td><td>Geometric error for matched structures after structure matching</td><td>Only meaningful for matched structures; it does not penalize unmatched failures directly</td></tr><tr><td>Compositional validity</td><td>Fraction of generated compositions passing charge or composition-validity checks in the benchmark protocol</td><td>Does not guarantee exact formula recovery from a text prompt</td></tr><tr><td>Structural validity</td><td>Fraction of generated structures passing basic geometric validity checks</td><td>Does not guarantee stability after DFT relaxation</td></tr><tr><td>COV-R</td><td>Coverage recall of the generated distribution relative to the test set</td><td>Depends on fingerprint choice and distance threshold</td></tr><tr><td>COV-P</td><td>Coverage precision of generated structures relative to the test distribution</td><td>Can remain high even when some prompt-level attributes are not exactly recovered</td></tr><tr><td>EMD statistics</td><td>Earth mover&#x27;s distance between generated and reference distributions for selected</td><td>Summarizes marginal distributions and may miss conditional mismatches at the</td></tr><tr><td>t-SNE JSD</td><td>scalar or count features Histogram divergence between low-dimensional embedding distributions</td><td>individual-prompt level Projection-dependent and therefore used only as a qualitative diagnostic</td></tr></table>

This distinction explains why the manuscript reports several diagnostics rather than a single score. CSP match rate tests whether text conditioning improves the chance of entering the correct structural basin. DNG validity and EMD metrics test whether the generated population is plausible and distributionally aligned. The composition-selected CGCNN diagnostic asks a narrower question: among candidates that already agree well with the target element set, do generated structures preserve coarse property cues from the prompt? The answer is positive for formation-energy sign and moderate for band-gap type, but this remains a surrogate-model diagnostic.

## Supplementary Note 3: CSP experiments and prompt-control analyses

The main text emphasizes the one-sample regime because it is the most direct test of sample eficiency. Table S3 reports both one-sample and 20-sample comparisons between CrystalFlow and TFMat. The gain is largest when the model has only one chance to sample a structure, especially on Perov-5 and Carbon-24. Under 20 evaluations, TFMat remains strongest on MP-20, while Carbon-24 and Perov-5 show a tradeof in which TFMat gives lower RMSE but a slightly lower match rate than CrystalFlow.

Table S3: Full CSP results for CrystalFlow and TFMat. Match rates are percentages. Δ match is TFMat minus CrystalFlow in percentage points.
<table><tr><td>Dataset</td><td>Evaluations</td><td>CrystalFlow match</td><td>CrystalFlow RMSE</td><td>TFMat match</td><td>TFMat RMSE</td><td>∆ match</td></tr><tr><td>Perov-5</td><td>1</td><td>53.69</td><td>0.0953</td><td>91.33</td><td>0.0449</td><td>+37.65</td></tr><tr><td>Carbon-24</td><td>1</td><td>15.02</td><td>0.3095</td><td>46.40</td><td>0.1353</td><td>+31.38</td></tr><tr><td>MP-20</td><td>1</td><td>67.65</td><td>0.0791</td><td>77.80</td><td>0.0868</td><td>+10.15</td></tr><tr><td>Perov-5</td><td>20</td><td>100.00</td><td>0.0416</td><td>96.62</td><td>0.0349</td><td>-3.38</td></tr><tr><td>Carbon-24</td><td>20</td><td>89.21</td><td>0.2208</td><td>85.02</td><td>0.1561</td><td>-4.19</td></tr><tr><td>MP-20</td><td>20</td><td>85.38</td><td>0.0614</td><td>92.04</td><td>0.0665</td><td>+6.66</td></tr></table>

The prompt conditions used for CSP are structured rather than free-form. Table S4 lists the fields and their intended roles. The structured-prompt design is best viewed as a controlled materials-metadata interface: it is more reproducible than natural prose, but it is weaker than coordinate conditioning because it does not provide fractional coordinates or lattice vectors.

Table S4: Prompt fields used as structured text conditions. Field availability varies across datasets and records. The table describes the intended semantic role of each field rather than a guarantee that every record contains every field.
<table><tr><td>Prompt field</td><td>Semantic role</td><td>Important limitation</td></tr><tr><td>Formula or composition</td><td>Defines the chemical identity or composition-level target and gives the</td><td>Highly informative for CSP; shuffled-prompt and field-ablation</td></tr><tr><td>Prototype or structural family</td><td>model a strong chemical prior Provides human-readable structural context when available</td><td>controls are needed to quantify leakage Not uniformly available across all records</td></tr><tr><td>Crystal system or space group</td><td>Encodes coarse symmetry information that may constrain lattice and site patterns</td><td>Exact symmetry recovery is not guaranteed by population-level DNG metrics</td></tr><tr><td>Formation energy</td><td>Provides a coarse stability cue for property-aware generation</td><td>Used as a scalar text value, not as a DFT relaxation target during sampling</td></tr><tr><td>Band gap</td><td>Provides an electronic-type cue, especially zero/nonzero gap information</td><td>Band-gap prediction is intrinsically noisier than formation-energy prediction in the CGCNN diagnostic</td></tr><tr><td>Energy above hull</td><td>Provides a metastability-related descriptor when available</td><td>Missing values and database uncertainty limit its use as a strict generation target</td></tr></table>

Prompt-control experiments on MP-20 test whether the improvement depends on prompt-content alignment rather than only on extra conditioning parameters (Table S5). A milder property-shufle control keeps formula and symmetry fields fixed while replacing scalar property values from chemically related donor prompts; it retains most of the full-text gain, consistent with formula and symmetry being the dominant CSP cues. A partial embedding perturbation replaces 20% of prompt embeddings by randomly shufled embeddings and performs close to the no-text baseline, showing that embedding-level corruption is less interpretable and more disruptive than field-level prompt edits.

Table S5: MP-20 CSP prompt-control experiments. All rows use 100 ODE steps, coordinate annealing and one generated candidate per test case. The full-text, property-shufle and partial-shufle rows use the text-conditioned CSP model; no-text uses the CrystalFlow baseline.
<table><tr><td>Condition</td><td>Prompt modification</td><td>Match rate ↑</td><td>RMSE↓</td></tr><tr><td>Full structured prompt</td><td>None</td><td>0.771875</td><td>0.087776</td></tr><tr><td>Property-shuffle control</td><td>Formula and symmetry retained; scalar property fields shuffled</td><td>0.743500</td><td>0.088601</td></tr><tr><td>Partial-shuffle control</td><td>20% of prompt embeddings replaced by shuffled embeddings</td><td>0.655750</td><td>0.092532</td></tr><tr><td>No-text baseline</td><td>No text condition</td><td>0.669750</td><td>0.077648</td></tr></table>

## Supplementary Note 4: De novo generation experiments and hyperparameter selection

The best TFMat DNG result does not arise from simply increasing the number of ODE steps. We compared four targeted sampling configurations, varying integration depth, coordinate-annealing slope and guidance factor while keeping the evaluation pipeline fixed (Table S6). The best balance is obtained by the 100-step configuration with coordinate-annealing slope 10 and the default text-conditioned sampling factor. This setting gives the best number-of-elements EMD and density EMD while retaining high structural validity and coverage precision.

The sweep suggests that the text-conditioned flow benefits from a moderate integration depth paired with a sharper coordinate schedule. Increasing the number of ODE steps to 200 slightly improves coverage recall but worsens density alignment. Fixed guidance factors above or below the default also fail to dominate. For this reason, the main MP-20 DNG result uses the 100-step, anneal-slope-10 configuration.

Table S6: Targeted TFMat sampling sweep on MP-20 DNG. Best values are bold and second-best values are underlined. The default conditional sampling factor is 1.0.
<table><tr><td></td><td></td><td></td><td>ODE steps Anneal slope Guide factor Comp. validity ↑ Struct. validity ↑ COV-R ↑</td><td></td><td></td><td>COV-P↑</td><td></td><td># Element EMD ↓ Density EMD ρ ↓</td></tr><tr><td>100</td><td>5</td><td>0.8</td><td>80.34</td><td>99.35</td><td>99.50</td><td>99.51</td><td>0.3178</td><td>0.1397</td></tr><tr><td>100</td><td>5</td><td>1.2</td><td>81.03</td><td>97.55</td><td>99.43</td><td>99.42</td><td>0.2428</td><td>0.2102</td></tr><tr><td>100</td><td>10</td><td>1.0</td><td>81.32</td><td>99.06</td><td>98.91</td><td>99.67</td><td>0.1990</td><td>0.0794</td></tr><tr><td>200</td><td>5</td><td>1.0</td><td>81.11</td><td>98.30</td><td>99.55</td><td>99.27</td><td>0.2208</td><td>0.2659</td></tr></table>

The displayed t-SNE panels in Fig. 4 are per-panel projections: each generated set is embedded together with the same MP-20 test set and then projected independently. This makes the plots suitable for visual inspection of overlap within each panel, but it means that point coordinates should not be compared directly between the two panels. To provide a more quantitative cross-model check, Table S7 reports both the per-pane Jensen-Shannon divergence shown in the figure and a separate shared-coordinate t-SNE diagnostic.

Table S7: MP-20 graph-embedding t-SNE diagnostics. The per-panel JSD values correspond to the two panels in Fig. 4. The shared-coordinate metrics are computed from a single projection containing the test set and both generated sets
<table><tr><td>Model</td><td>Test count</td><td>Generated count</td><td>Per-panel JSD ↓</td><td>Shared COV-R ↑</td><td>Shared COV-P ↑</td><td>Shared JSD ↓</td></tr><tr><td>Unconditional DNG baseline</td><td>9,046</td><td>10,000</td><td>0.636697</td><td>0.453350</td><td>0.4997</td><td>0.435103</td></tr><tr><td>Text-guided DNG</td><td>9,046</td><td>10,000</td><td>0.619971</td><td>0.472806</td><td>0.5124</td><td>0.428576</td></tr></table>

The property diagnostic in the main text uses a selected subset rather than the full generated population. The 2,000 structures were selected from the first 5,000 TFMat DNG candidates by element-composition match score, so the result measures the most composition-consistent subset of the generated population. Within this selected subset, 1,350 structures perfectly match the target element set and no structure has zero element overlap (Table S8). The average element-match score is 0.9075.

Table S8: Element-composition match distribution for the top 2,000 generated structures. Structures are ranked by elementcomposition match score before the CGCNN property diagnostic is performed.
<table><tr><td>Match category</td><td>Count</td><td>Percentage</td><td>Score range</td></tr><tr><td>Complete match</td><td>1,350</td><td>67.50%</td><td>[1.0]</td></tr><tr><td>Partial match</td><td>650</td><td>32.50%</td><td>(0.0, 1.0)</td></tr><tr><td>No match</td><td>0</td><td>0.00%</td><td>[0.0]</td></tr><tr><td>Overall average</td><td>2,000</td><td>100.00%</td><td>0.9075</td></tr></table>

CGCNN predictions were available for 1,994 selected structures. The surrogate validation errors are summarized in Table S9. We removed 41 numerical formation-energy outliers with absolute error above 10 eV atom<sup>−1</sup> before computing the cleaned formation-energy scatter statistics reported in Fig. 3. No outlier removal was applied to the band-gap diagnostic. These choices should be treated as diagnostic filtering rather than as a benchmark protocol.

Table S9: CGCNN surrogate models used in the property diagnostic. Validation errors document the reliability of the surrogate models rather than replacing first-principles validation.
<table><tr><td>Target property</td><td>Best validation MAE</td><td>Best epoch</td><td>Diagnostic use</td></tr><tr><td>Formation energy</td><td>0.0391 eV atom−1</td><td>84/100</td><td>Sign agreement and cleaned regression statistics</td></tr><tr><td>Band gap</td><td>0.3747 eV</td><td>6/100</td><td>Zero/nonzero agreement and regression statistics</td></tr></table>

The large diference between formation-energy and band-gap validation errors is consistent with the main-text diagnostic: formation energy shows stronger correlation and lower median error, whereas band gap is more reliable as a coarse electronic-type indicator than as a precise regression target. This is why the main text emphasizes formation-energy sign and band-gap zero/nonzero agreement rather than claiming quantitative electronic-structure accuracy.

A separate diagnostic run with 1,000 generated samples, text embeddings, 100 ODE steps, coordinateannealing slope 5 and a conditional sampling factor of 1.0 is not used as the main DNG result because its formula, space-group and crystal-system matching rates are very low: 2.10%, 2.30% and 0.00%, respectively. It is retained here because it defines an important interpretation boundary: strong population-level validity and coverage metrics do not guarantee exact recovery of the prompt formula or symmetry labels in a small diagnostic sample. This motivates the composition-selected property analysis above and the cautionary language in the Discussion.