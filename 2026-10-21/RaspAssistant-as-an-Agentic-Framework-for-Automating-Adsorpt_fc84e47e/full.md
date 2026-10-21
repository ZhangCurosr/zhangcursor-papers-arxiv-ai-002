# RaspAssistant as an Agentic Framework for Automating Adsorption Simulation Research in Porous Materials

Marko Petkovi´c<sup>1,3</sup>, Susanne Gooijer<sup>1</sup>, Jos´e Manuel Vicent-Luna<sup>1</sup>, Vlado Menkovski<sup>2,3</sup>, and Sof´ıa Calero<sup>∗1,3</sup>

<sup>1</sup>Department of Applied Physics and Science Education, Eindhoven University of Technology, Eindhoven, The Netherlands

<sup>2</sup>Department of Mathematics and Computer Science, Eindhoven University of Technology, Eindhoven, The Netherlands

<sup>3</sup>Eindhoven Artificial Intelligence Systems Institute (EAISI), Eindhoven University of Technology, Eindhoven, The Netherlands

## Abstract

Porous-material adsorption studies require coordinated structure preparation, force-field selection, simulation, validation, and interpretation, making end-to-end automation dificult. We test whether RaspAssistant, a hierarchical multi-agent framework, can carry an adsorption study from force-field preparation through validation and downstream separation analysis. Across twelve multi-agent workflow-reproduction runs spanning four adsorption and separation benchmarks, overall scores ranged from 8.50 to 10.00 on a rubric evaluating workflow correctness, simulation validity, and analysis consistency. In a separate eight-case force-field extraction benchmark, the mean weighted score was 0.967, combining parameter agreement and RASPA readiness. In open-ended studies, the system diagnosed and refined an acetylene force field and, after validating a CO<sub>2</sub> force field against MFI experiments, screened 197 pure-silica zeolites that passed automated input checks. These results demonstrate autonomous workflow execution after task specification and input provision: RaspAssistant coordinates, validates, adapts, and documents extended adsorption-simulation workflows while exposing model sensitivity and methodological limitations for expert review.

## 1 Introduction

Molecular simulations connect atomistic structure to macroscopic performance, reveal mechanisms often dificult to probe experimentally, and evaluate candidate materials before synthesis and characterization. For porous materials, adsorption and separation depend sensitively on framework topology, pore chemistry, and host–guest interactions. Packages such as RASPA<sup>1;2</sup> evaluate adsorption, difusion, and mixture behaviour in nanoporous materials, while computation-ready databases such as CoRE MOF<sup>3;4</sup> support systematic screening <sup>5</sup>.

Despite mature simulation tools, reliable porous-materials studies remain complex and expertise-intensive. Researchers must prepare computation-ready structures, choose supercells and simulation conditions, assign or adapt framework and adsorbate force fields, and decide whether transferable models such as UFF4MOF<sup>6;7</sup>, DREIDING<sup>8</sup>, or TraPPE<sup>9</sup> need systemspecific refinement. MOF adsorption predictions can depend strongly on force-field selection <sup>10</sup>, while experimental variability and occasional outliers complicate validation <sup>11;12</sup>.

An adsorption and separation study extends beyond a single simulation, coupling purecomponent isotherms, experimental validation, mixture predictions using Ideal Adsorbed Solution Theory (IAST) or direct simulations, and breakthrough analysis. Tools support isotherm fitting, mixture prediction, and process simulation <sup>13–15</sup>. Complete automation also requires coordinating these stages and revising models or protocols when validation exposes limitations.

These coordination needs motivate agentic scientific systems, which combine large language models (LLMs) with planning, memory, tool use, and iterative reasoning <sup>16</sup>. Such systems can translate research objectives into tool-based actions and use the results to guide subsequent steps. Applications span experimental chemistry <sup>17;18</sup>, quantum chemistry <sup>19;20</sup>, biomolecular simulation<sup>21</sup>, atomistic materials modeling <sup>22–33</sup>, and porous-material property prediction and design<sup>34;35</sup>.

We introduce RaspAssistant, a multi-agent framework (Figure 1) for literature-informed force-field construction, input preparation, simulation, post-processing, and analysis across adsorption and separation workflows. We test whether it can reproduce a complete workflow without human scientific intervention after task specification and input preparation. Building on our setup-focused framework <sup>36</sup>, agents use complete simulation and force-field files, general file-manipulation tools, and execution tools to revise setup and task sequences as evidence accumulates. We evaluate four published workflows, literature-based force-field extraction, and two open-ended studies of model refinement and validation-based zeolite screening. The contribution is continuity across force-field selection or adaptation, experimental validation, pure-component and mixture simulation, and process-level analysis. Further context is in the Supporting Information (SI).

![](images/5051db63fd4f0c147bc5279bcdf11193a99b533569c466a21ea92987eb9d52bb.jpg)  
Figure 1: Agent architecture of RaspAssistant. Orange nodes denote higher-level agents responsible for planning, coordination, and verification, while blue nodes denote lower-level ReAct agents that alternate reasoning with tool actions. The red dashed subsystem supports literature search, force field extraction, adaptation, and parameter file generation. The blue dashed subsystem supports simulation setup, execution, analysis, plotting, and force field fitting for workflow reproduction. Shared databases, examples, memories, and tools occupy the bottom layer. Agent roles are listed in Table S5.

## 2 Results

## 2.1 Workflow Reproduction

The benchmark comprises four published studies: (1) $\mathrm { { C O } _ { 2 } / N _ { 2 } }$ separation in $\mathrm { T A M O F { - } 1 ^ { 3 7 } }$ , (2) $\mathrm { C _ { 2 } H _ { 2 } / C O _ { 2 } }$ and $\mathrm { C _ { 2 } H _ { 2 } / C H _ { 4 } }$ separation in open-metal-site MOFs<sup>38</sup>, (3) metal-center efects in M-$\mathrm { M O F } { - } 7 4 ^ { 3 9 }$ , and $\mathrm { ( 4 ) C _ { 3 } \mathrm { - } C _ { 6 } }$ olefin/parafin separation in cation-exchanged zeolites <sup>40</sup>. Literaturederived force fields were supplied for these runs. Each study had three multi-agent runs scored for workflow correctness, simulation validity, and analysis consistency (methods in the SI), plus one single-agent baseline reported separately and excluded from multi-agent averages. Table 1 lists run-level scores and approximate resource use.

Table 1: Scores and approximate resource use for the twelve multi-agent workflow-reproduction runs.
<table><tr><td>Benchmark</td><td>Run</td><td>Overall (/10)</td><td>Elapsed (d)</td><td>Tokens (M)</td><td>Cost (US$)</td></tr><tr><td rowspan="3">TAMOF-1</td><td>1</td><td>10.00</td><td>8.8</td><td>12</td><td>15</td></tr><tr><td>2</td><td>9.60</td><td>2.4</td><td>12</td><td>14</td></tr><tr><td>3</td><td>9.60</td><td>8.8</td><td>7</td><td>8</td></tr><tr><td rowspan="3">Acetylene</td><td>1</td><td>9.05</td><td>60.0*</td><td>21</td><td>17</td></tr><tr><td>2</td><td>9.60</td><td>5.0</td><td>14</td><td>15</td></tr><tr><td>3</td><td>8.73</td><td>36.3</td><td>11</td><td>11</td></tr><tr><td rowspan="3">M-MOF-74</td><td>1</td><td>8.58</td><td>2.5</td><td>25</td><td>25</td></tr><tr><td>2</td><td>9.60</td><td>2.9</td><td>22</td><td>22</td></tr><tr><td>3</td><td>8.50</td><td>1.2</td><td>9</td><td>10</td></tr><tr><td rowspan="3">Zeolites</td><td>1</td><td>8.95</td><td>4.1</td><td>23</td><td>19</td></tr><tr><td>2</td><td>8.68</td><td>9.0</td><td>13</td><td>13</td></tr><tr><td>3</td><td>8.70</td><td>18.9</td><td>22</td><td>12</td></tr></table>

<sup>\*</sup>Approximate managed campaign duration, rather than an autonomous makespan for all requested work. The agent repeatedly requested batches of 20–30 nominally identical, high-cycle simulations per condition. To limit redundant resource use, only one representative from each batch was executed.

Across twelve multi-agent runs, the average workflow, simulation-validity, and analysisconsistency scores were 9.36, 8.92, and 9.10, respectively, giving an overall average of 9.13. Deductions concerned normalization errors, limited convergence evidence, or interpretation. Some afected outputs supported only qualitative conclusions or could not support quantitative comparisons. Scores aggregate workflow execution, validity checks, and interpretation. They are not percentages of quantitatively reliable predictions.

The four single-agent baselines scored 6.78–9.05. The comparison is descriptive: each benchmark has one baseline, using GPT-5.4 throughout, and architectures were not matched for model mix, tokens, or inference cost.

## 2.2 Force-Field Extraction

Eight literature-derived cases were evaluated using parameter-set intersection over union (IoU) and RASPA readiness, the fraction of generated force-field files executable without correction. The metrics were combined into $S _ { \mathrm { F F } }$ and averaged over repeated extractions (methods in the SI). Table 2 summarizes performance.

The weighting gives a mean extraction score of 0.967 across eight cases (range 0.864–1.000). Mean parameter IoU is 0.983, indicating close agreement with reference sets. No hallucinated parameters were observed. Deviations mainly reflect omissions during extraction or final-file writing. In a few cases, such as EPM2, irregular source formatting caused incorrect parameter associations. Most issues would be detected by initial simulation logs warning of missing atom or interaction definitions.

Table 2: Force-field extraction performance and weighted score by source.
<table><tr><td>Method</td><td>IoU</td><td>Readiness</td><td> $S _ { \mathrm { F F } }$ </td><td>Cost ($)</td></tr><tr><td> $\overline { { \mathrm { C O _ { 2 } \ E P M 2 ^ { 4 1 } } } }$ </td><td>0.95</td><td>100%</td><td>0.970</td><td>0.19</td></tr><tr><td> $\mathrm { C O _ { 2 } \ i n \ z e o l i t e s ^ { 4 2 } }$ </td><td>1.00</td><td>88%</td><td>0.952</td><td>0.23</td></tr><tr><td> $\mathrm { C O _ { 2 } \ i n \ z e o l i t e s ^ { 4 3 } }$ </td><td>0.99</td><td>92%</td><td>0.962</td><td>0.34</td></tr><tr><td> $\mathrm { C O _ { 2 } } , \mathrm { N _ { 2 } } , \mathrm { O _ { 2 } } , \mathrm { A r }$  in zeolites44</td><td>0.98</td><td>100%</td><td>0.988</td><td>0.24</td></tr><tr><td> $\mathrm { T r a P P E  – z e o ^ { 4 5 } }$ </td><td>1.00</td><td>100%</td><td>1.000</td><td>0.15</td></tr><tr><td> $\mathrm { C } _ { x } \mathrm { H } _ { x } , \mathrm { C O } _ { 2 } , \mathrm { N } _ { 2 } , \mathrm { O } _ { 2 }$  in zeolites46</td><td>1.00</td><td>100%</td><td>1.000</td><td>0.65</td></tr><tr><td> $\mathrm { T A M O F { - } 1 ^ { 3 7 } }$ </td><td>0.94</td><td>75%</td><td>0.864</td><td>0.27</td></tr><tr><td> $\mathrm { U F F + D R E I D I N G \ ( M O F s ) ^ { 8 ; 4 7 } }$ </td><td>1.00</td><td>100%</td><td>1.000</td><td>1.55</td></tr></table>

Lower readiness arose from incorrect declared atom or interaction counts despite accurate interaction definitions. These bookkeeping errors make the files structurally invalid but recoverable by post-processing. Force-field extraction costs remain low (Table 2), increasing when parameters span multiple sources or many entries.

## 2.3 Case Study 1: $\mathbf { C _ { 2 } H _ { 2 } } / \mathbf { C O _ { 2 } }$ and $\mathbf { C _ { 2 } H _ { 2 } } / \mathbf { C H _ { 4 } }$ Separation in Open-Metal-Site MOFs

The first case study tested whether RaspAssistant could identify and correct force-field limitations before analysing acetylene separation in MOFs (Figure 2). Initial validation against experimental isotherms revealed substantial disagreement for $\mathrm { C _ { 2 } H _ { 2 } }$ in Fe-MOF-74, while other adsorption results were suficiently consistent to proceed. The discrepancy persisted after increasing the unit-cell size and Monte Carlo cycles, so RaspAssistant focused on refining the interactions between acetylene and the open Fe sites.

Using the fitting harness, RaspAssistant evaluated 45 force-field versions over 15 iterations, with six simulations per version (270 simulations), in approximately 5 days. Initially, the system attempted to correct the force field by fitting only the $\mathrm { F e { - } C _ { \mathrm { C _ { 2 } H _ { 2 } } } }$ cross interaction, using the experimental isotherm and the $\mathrm { { C _ { 2 } H _ { 2 } } }$ binding geometry as fitting targets. This produced partial improvement, but was not suficient to reproduce the adsorption behaviour. The system subsequently expanded the fitting space to include framework oxygen interactions with the acetylene carbon atom, and later introduced separate interactions for diferent oxygen environments. This led to a final force field that reproduced the $\mathrm { { C _ { 2 } H _ { 2 } } }$ adsorption isotherm with accuracy comparable to the force field reported in the reference study, as shown in Figure 3.

![](images/e0b38a4f7848f850d40fe95a8d0c66b104da850af4b8f3b096160c51bb7124ba.jpg)  
Figure 2: Schematic overview of Case Study 1. Here, baseline validation means validation of the supplied initial force field. The workflow includes detection of the poor $\mathrm { { C _ { 2 } H _ { 2 } } }$ isotherm in Fe-MOF-74, convergence checks, force field fitting, validation of the fitted model, and subsequent mixture adsorption and adsorption site analysis.

![](images/b4c165dc42bbc1bf71430e8acc808ed7ead609882c6f57a6e97e70189647be1a.jpg)  
Figure 3: $\mathrm { C _ { 2 } H _ { 2 } }$ adsorption isotherms in Fe-MOF-74. “Baseline simulation” uses the initial force field. “Revalidation simulation” uses the fitted force field. The plot was generated by RaspAssistant. The isotherm from Ref.<sup>38</sup> was added manually for comparison.

The $\mathrm { C _ { 2 } H _ { 2 } }$ isotherm obtained with the force field fitted by RaspAssistant closely follows the literature-force-field curve over the investigated range. To further assess whether the fitted model captures the local interaction with the open metal site, we compare the predicted $\mathrm { { C _ { 2 } H _ { 2 } } }$ binding geometry. Force-field fitting reduced the arithmetic mean of the two Fe–C contacts from 3.12 to 2.63 <sup>˚</sup>A, bringing it close to the literature-force-field value of 2.60 <sup>˚</sup>A and the experimental mean of 2.50 <sup>˚</sup>A (Table 3).

Table 3: Comparison of the arithmetic mean of the two Fe–C contacts for $\mathrm { { C _ { 2 } H _ { 2 } } }$ adsorbed in Fe-MOF-74.
<table><tr><td>Model</td><td>Mean Fe-C distance (Å)</td></tr><tr><td>Initial force field</td><td>3.12</td></tr><tr><td>RaspAssistant fitted force field</td><td>2.63</td></tr><tr><td>Literature force field</td><td>2.60</td></tr><tr><td>Experiment</td><td>2.50</td></tr></table>

The final fitted parameters difer substantially from the literature model (Table 4), despite similar isotherms: RaspAssistant concentrates the correction in a stronger $\mathrm { F e - C _ { C _ { 2 } H _ { 2 } } }$ interaction with weaker oxygen contributions, whereas the literature correction is distributed over several framework atom types.

Variation in the auxiliary $\mathrm { C O _ { 2 } }$ guardrail minimizations may have caused the fitting loop to reject candidates that were otherwise moving the $\mathrm { { C _ { 2 } H _ { 2 } } }$ description in a promising direction.

The RaspAssistant optimization used the adsorption isotherm and binding geometry as fitting targets, whereas the literature model was also validated against adsorption-heat data <sup>38</sup>. Experimental adsorption heats were unavailable during fitting. The subsequent $Q _ { \mathrm { s t } }$ comparison therefore provides an independent energetic assessment. The fitted model captures the lowloading energy scale and the qualitative decrease toward saturation, although the decrease is sharper than in experiment. The literature model follows the low-loading data more closely but also deviates over part of the high-loading region (Figure S4). This broad agreement outside the fitting targets is encouraging, while showing how adsorption-heat data could further constrain future fitting campaigns.

Table 4: Comparison of fitted Lennard–Jones parameters for acetylene–framework interactions in Fe-MOF-74. Literature parameters are taken from Ref. <sup>38</sup>.
<table><tr><td rowspan="2">Interaction Pair Atom Types</td><td rowspan="2"></td><td colspan="2">Agent</td><td colspan="2">Literature</td></tr><tr><td>∈ (K)</td><td> $\overline { { \sigma \ ( \mathrm { A } ) } }$ </td><td>∈ (K)</td><td> $\overline { { \sigma \ ( \mathrm { A } ) } }$ </td></tr><tr><td>Metal site</td><td> $\overline { { \mathrm { F e } \mathrm { ~ - ~ } \mathrm { C } _ { \mathrm { { C } _ { 2 } \mathrm { { H } _ { 2 } } } } } }$ </td><td>680.0</td><td>2.355</td><td>38.92</td><td>2.878</td></tr><tr><td rowspan="3">Framework oxygen</td><td> $\overline { { \mathrm { { O 1 - C } _ { C _ { 2 } H _ { 2 } } } } }$ </td><td>10.0</td><td>2.900</td><td>79.23</td><td>2.870</td></tr><tr><td> $\mathrm { O 2 \mathrm { ~ - ~ } C _ { C _ { 2 } H _ { 2 } } }$ </td><td>20.0</td><td>3.050</td><td>79.23</td><td>2.870</td></tr><tr><td> $\mathrm { O 3 \mathrm { ~ - ~ } C _ { C _ { 2 } H _ { 2 } } }$ </td><td>20.0</td><td>3.050</td><td>79.23</td><td>2.870</td></tr><tr><td>Framework carbon</td><td> $\overline { { \mathrm { C 1 / 2 / 3 / 4 - C _ { C _ { 2 } H _ { 2 } } } } }$ </td><td></td><td></td><td>78.96</td><td>3.055</td></tr><tr><td>Framework hydrogen</td><td> $\overline { { \mathrm { ~ H ~ } - \mathrm { ~ C _ { C _ { 2 } H _ { 2 } } } } }$ </td><td></td><td></td><td>31.57</td><td>2.792</td></tr></table>

Dashes indicate no fitted cross-interaction override. The initial force field and mixing rules remain active for these pairs.

After fitting, RaspAssistant simulated mixtures and calculated $\mathrm { C _ { 2 } H _ { 2 } / C O _ { 2 } }$ and $\mathrm { C _ { 2 } H _ { 2 } / C H _ { 4 } }$ selectivities. These generally agree with literature trends for Fe-MOF-74 and IRMOF-1 (Figure 4). Deviations are largest when the less strongly adsorbed component has very low uptake, making selectivity sensitive to small absolute loading diferences.

$$
\begin{array} { r } { \begin{array} { l l l } { \overline { { \phantom { - } } } \mathtt { F e } \mathtt { M O F - 7 4 } } & { \overline { { \phantom { - } } } \overline { { \bullet } } \overline { { \phantom { - } } } \mathtt { R a s p A s s i s t a n t } } \\ { \overline { { \phantom { - } } } \mathtt { I R M O F - 1 } } & { \overline { { \phantom { - } } } \overline { { \circ - \cdots } } } & { \mathsf { L u n a - T r i g u e r o \ e t a l } . } \end{array} } \end{array}
$$

![](images/1d254f7bf75fad659dd2cbfc6e2d6fb27c829394325a1e1598fa4379548963bd.jpg)

(b) C₂H₂ / CO₂  
![](images/13f398e20192162a79107c7b51ce0ee0ea45f66b65def86bfcbce9fe98484843.jpg)  
Figure 4: Comparison of $\mathrm { C _ { 2 } H _ { 2 } / C H _ { 4 } }$ (a) and $\mathrm { C _ { 2 } H _ { 2 } / C O _ { 2 } }$ (b) mixture selectivities from the fitted Fe-MOF-74 and retained $\mathrm { I R M O F { - } 1 }$ models with $\mathrm { R e f . ^ { 3 8 } }$ . RaspAssistant generated the plot. Reference data were added manually.

RaspAssistant then interpreted the adsorption sites through nearest Fe distance distributions (Figure 5). $\mathrm { { C _ { 2 } H _ { 2 } } }$ is strongly localized near the open Fe sites, whereas $\mathrm { C O _ { 2 } }$ and $\mathrm { C H _ { 4 } }$ have broader distributions farther into the pore space, consistent with the physical picture from the reference study’s spatial histograms.

Case Study 1 required approximately 10 wall-clock days, 50 million tokens, and US\$51.6 in API-equivalent cost.

![](images/701bb4e7342201fc779d6f22f2125e167e83089db22b81cbb5633bedf668b6bd.jpg)  
Figure 5: Normalized minimum periodic interaction-site-to-Fe distance distributions in Fe-MOF-74 for equimolar $\mathrm { C _ { 2 } H _ { 2 } / C H _ { 4 } }$ and $\mathrm { C _ { 2 } H _ { 2 } / C O _ { 2 } }$ mixtures at 298 K and 1000 kPa. The sharp short distance peak for $\mathrm { { C _ { 2 } H _ { 2 } } }$ indicates preferential localization near the open Fe sites, whereas $\mathrm { { C O } _ { 2 } }$ and $\mathrm { C H _ { 4 } }$ are mainly distributed farther away in the pore space.

## 2.4 Case Study 2: High-Throughput Screening of Pure-Silica Zeolites for $\mathbf { C O } _ { 2 }$ Adsorption

We next tested whether experimental validation could guide force-field selection before applying a common screening protocol across pure-silica zeolites (Figure 6). Given MFI experimental isotherms and pure-silica zeolite structures, RaspAssistant prepared three $\mathrm { C O _ { 2 } }$ force fields: the Garcia-Sanchez model<sup>42</sup>, its framework model combined with TraPPE $\mathrm { C O _ { 2 } } ^ { 4 2 ; 4 8 }$ , and TraPPE $\mathrm { C O _ { 2 } }$ combined with a Vujic-derived silica-zeolite framework model <sup>44</sup>. Simulated MFI isotherms were compared with experiment at 253, 273, and 303 K.

![](images/c7f2e8de947f2f295a82893e5d243cb22da64ca733ba515cdc9a3a2498e8c5b0.jpg)  
Figure 6: Schematic overview of the main steps taken by RaspAssistant in Case Study 2. The workflow includes preparation of literature-based force-field candidates, validation against MFI $\mathrm { C O _ { 2 } }$ adsorption experiments, selection of the best-performing force field, 298 K screening of the pure-silica zeolite dataset, and post-processing through ranking and descriptor analysis.

The TraPPE CO /Vujic-derived silica model best matched MFI experiments overall, with a weighted mean absolute error (MAE) of 0.72 molecules per unit cell, compared with 0.866 for Garcia-Sanchez and 0.872 for Garcia-Sanchez/TraPPE-Potof. At 303 K (Figure 7), the Vujicderived candidate <sup>44</sup> follows experiment more closely over most of the pressure range. The other models underpredict more strongly, while the selected model slightly overpredicts at higher pressures. RaspAssistant selected it for screening, treating results as relative rankings under a consistent protocol rather than quantitatively exact predictions.

$$
\begin{array} { r l } { \bullet \bullet \ \mathsf { E x p e r i m e n t } \ } & { \bullet \bullet \ \mathsf { T r a P P E } / \mathsf { G a r c i a } \mathsf { - S a n c h e z } } \\ { \bullet \bullet \ \mathsf { G a r c i a - S a n c h e z } } & { \bullet \bullet \ \mathsf { T r a P P E } / \mathsf { V u j i c ~ s i l i c a } } \end{array}
$$

![](images/c8ea4944cd1a7e495cc4811d8fa94c51dec9a6927b4f77eb4ff74332ab0eca8b.jpg)

![](images/15ee7603144ad60de9fb776733a9c6f680841a3792a827d1268445a096d6363f.jpg)  
Figure 7: Validation of the candidate force fields against the provided $\mathrm { C O _ { 2 } }$ adsorption isotherm in MFI at 303 K. The TraPPE Silica 2016 candidate combines TraPPE $\mathrm { C O _ { 2 } }$ with a Vujicderived silica-zeolite model and gives the best agreement. Open markers flag block-drift outliers. These points are retained in the error calculations (see the SI).

RaspAssistant completed $2 , 9 5 5 \mathrm { C O _ { 2 } }$ adsorption calculations at 298 K: 15 pressures for each of 197 structures passing automated input checks. Outputs were checked for missing or nonphysical values afecting ranking. Isotherms provided interpolated uptake at 1, 10, and 100 kPa and working capacity q(100 kPa) − q(1 kPa) (Figure 8). Rankings changed modestly with pressure: JBW led at 1 kPa, while GIS led at 10 and 100 kPa and in working capacity. SIV, MON, PHI, AFY, and YUG were also among the strongest candidates at higher pressure or for working capacity.

![](images/d6ecb81bbc57fcc0d985ce116cf4ed1e914f464c1c98967fd65315c221e468a7.jpg)

![](images/0b81e18491610ae413decc6a14fd086ef3fbabcfbb3d38c8dd8d575f593d9f6c.jpg)

![](images/87377468ec106e4eebd34c206ffcc859f00c5042f69f42544e970480006b699d.jpg)

(d)  
![](images/1a10b238e204d0781b85d3a9b87822ed9bcbfdbfbc6a703269c200fa8e9fcee1.jpg)  
Figure 8: Distribution of interpolated $\mathrm { C O _ { 2 } }$ uptake at 1, 10, and 100 $\mathrm { k P a }$ and of the derived 1–100 kPa working capacity in Case Study 2.

Framework density showed a weak positive association with simulated $\mathrm { C O _ { 2 } }$ uptake at 1 kPa (Spearman $\rho = 0 . 2 8 4$ , Figure 9). GIS, JBW, SIV, ATN, and YUG were positive residual outliers. The retained descriptors lacked pore size, accessible volume, and channel connectivity, so these residuals cannot be assigned to a specific pore-scale mechanism.

Outliers: framework density vs uptake at 1 kPa $( \mathsf { C O } _ { 2 }$ , 298 K)  
![](images/a2b8bbf121713faf0e6bd7647c1ab2c418a7279e4f88bed3fc5436bc035447fb.jpg)  
Figure 9: Relationship between framework density and simulated $\mathrm { C O _ { 2 } }$ uptake at 1 kPa for the screened pure-silica zeolites. Highlighted topologies correspond to positive outliers relative to the overall linear trend, indicating unusually high uptake for their framework density.

Case Study 2 required approximately 5 wall-clock days, 23 million tokens, and US\$24.5 in API-equivalent cost.

## 3 Discussion

RaspAssistant connects literature-derived interaction models, experimental validation, atomistic simulation, and separation analysis within a coordinated research workflow. Its contribution extends beyond preparing or executing individual simulations to deciding how a study should proceed as results become available. The open-ended studies demonstrate this capability through refinement of an acetylene force field and validation-guided selection of a $\mathrm { C O _ { 2 } }$ model before zeolite screening. These decisions connect the adequacy of the molecular description to the interpretation of downstream predictions.

A consequential design choice is the combination of high-level scientific planning with direct access to complete simulation and force-field files. The Supervisor coordinates specialist agents while retaining the research objective, and the specialists can revise inputs, perform calculations, and return analyses that inform subsequent planning. This gives the framework flexibility to change both model details and the sequence of calculations within a study. In the acetylene case, the fitting procedure expanded the interaction model in response to the observed adsorption behaviour, then returned the fitted model to mixture and site analysis. The fitting loop itself follows a controlled sequence of parameter selection, simulation, analysis, and continuation decisions. Flexible study planning and structured model optimization therefore serve complementary roles.

The evaluations test diferent parts of this research process. Published-workflow reproduction examines coordination across established adsorption and separation methods, while literature-based extraction examines reconstruction of the interaction models needed for simulation. The open-ended cases then test whether intermediate results can guide the choice of subsequent calculations. Using the same framework across these tasks supports a broader role for agentic tools in connecting model construction, validation, and application. The descriptive architectural baseline does not isolate which individual design choices account for the observed performance.

The scientific value of this integration is clearest when several types of evidence are considered together. In acetylene adsorption, similar isotherms from substantially diferent fitted and literature parameter sets show why a single fitting target cannot fully establish model adequacy. Experimental heats unavailable during fitting provided an independent energetic assessment, complementing the isotherm, binding geometry, mixture, and site analyses. In zeolite screening, validation in MFI guided model selection before applying a common protocol across the structure set. Pressure-dependent rankings then connected candidate selection to operating conditions. Such rankings identify candidates for further investigation, with quantitative transferability beyond the reference system remaining a separate question.

Further development can build on this integration by connecting screening more closely to physical interpretation. Pore geometry descriptors from tools such as Zeo++ <sup>49</sup> could relate adsorption trends to accessible volume and channel structure, while systematic uncertainty analysis could connect confidence in molecular predictions to subsequent mixture and process comparisons. Shared workflow records also provide a basis for expert review of how model choices and intermediate results lead to a reported conclusion. The resulting division of work allows agents to coordinate and document extended computational studies after task specification and input provision, with researchers defining the objectives and assessing the physical models and scientific conclusions.

## 4 Methods

RaspAssistant receives a natural-language objective and a folder of structures, experimental data, and available force fields. Each benchmark run began without benchmark-specific memory from earlier runs. Computational methods and agent configurations are in the SI.

A Supervisor agent decomposes the research objective and coordinates specialists for literature retrieval, force-field preparation, simulation setup and execution, and analysis. Higher-level agents plan, execute, verify, and revise their subtasks. Specialists can read and modify complete files, execute calculations, and analyse outputs. Molecular simulations used RASPA 2.0.47 through a local simulation interface, with RUPTURA supporting mixture and breakthrough calculations. RASPA outputs are converted to structured JSON for analysis, and shared per-run memory and stored scientific files retain context across delegated tasks. The force-field fitting controller organizes successive parameter-selection, simulation, and analysis steps. Agents made scientific decisions during workflow execution. Benchmark execution procedures and human involvement are detailed in the SI.

One human assessor scored workflow-reproduction runs for workflow correctness, simulation validity, and analysis consistency with weights of 40%, 40%, and 20%. Force-field extraction combines exact parameter-set agreement (60%) and direct RASPA readiness (40%). Scoring prioritized completed simulation outputs and efective settings over final reports. Downstream results were assessed in light of upstream validity. Full methods, configurations, benchmark definitions, and scoring procedures are in the SI.

## Data availability

The supporting data are available on Zenodo at 10.5281/zenodo.23462699. The package contains chronological JSON and HTML workflow records, final reports and evaluation records, source data for figures and tables, and case-study force-field definitions and numerical summaries. An artifact manifest links the records to the scientific files.

## Code availability

The RaspAssistant source code and production prompt library are not publicly released. The frozen source code, prompts, and configuration used for the experiments reported here are available confidentially to editors and reviewers upon request. Manuscript-specific analysis files and source data are included in the supporting data package described in the Data Availability statement.

## Supporting Information

Computational methods and agent configurations, detailed benchmark results, workflowreproduction prompts and evaluation criteria, supporting workflow diagrams and adsorption analyses, and force-field fitting procedure and independent energetic validation (PDF).

## Author contributions

M.P. conceived the study, developed the framework, carried out the simulations and analyses, and drafted the manuscript. S.G. and J.M.V.L. contributed to the design of input files for the diferent use cases. V.M. and S.C. supervised the research. Funding was obtained by S.C. and M.P. All authors reviewed the manuscript.

## Competing interests

The authors declare no competing interests.

## Acknowledgment

OpenAI Codex was used to assist with editing the manuscript in October 2026.

## References

[1] Dubbeldam, D.; Calero, S.; Ellis, D. E.; Snurr, R. Q. RASPA: molecular simulation software for adsorption and difusion in flexible nanoporous materials. Molecular Simulation 2016, 42, 81–101.

[2] Ran, Y. A.; Sharma, S.; Balestra, S. R. G.; Li, Z.; Calero, S.; Vlugt, T. J. H.; Snurr, R. Q.; Dubbeldam, D. RASPA3: A Monte Carlo code for computing adsorption and difusion in nanoporous materials and thermodynamics properties of fluids. The Journal of Chemical Physics 2024, 161, 114106, DOI: 10.1063/5.0226249.

[3] Chung, Y. G.; Haldoupis, E.; Bucior, B. J.; Haranczyk, M.; Lee, S.; Zhang, H.; Vogiatzis, K. D.; Milisavljevic, M.; Ling, S.; Camp, J. S. et al. Advances, updates, and analytics for the computation-ready, experimental metal–organic framework database: CoRE MOF 2019. Journal of Chemical & Engineering Data 2019, 64, 5985–5998.

[4] Zhao, G.; Brabson, L. M.; Chheda, S.; Huang, J.; Kim, H. et al. CoRE MOF DB: A curated experimental metal-organic framework database with machine-learned properties for integrated material-process screening. Matter 2025, 8, 102140, DOI: 10.1016/j.matt.2025.102140.

[5] Ongari, D.; Talirz, L.; Smit, B. Too many materials and too many applications: an experimental problem waiting for a computational solution. ACS central science 2020, 6, 1890–1900.

[6] Addicoat, M. A.; Vankova, N.; Akter, I. F.; Heine, T. Extension of the universal force field to metal–organic frameworks. Journal of chemical theory and computation 2014, 10, 880–891.

[7] Coupry, D. E.; Addicoat, M. A.; Heine, T. Extension of the universal force field for metal– organic frameworks. Journal of Chemical Theory and Computation 2016, 12, 5215–5225.

[8] Mayo, S. L.; Olafson, B. D.; Goddard, W. A. DREIDING: a generic force field for molecular simulations. Journal of Physical chemistry 1990, 94, 8897–8909.

[9] Martin, M. G.; Siepmann, J. I. Transferable potentials for phase equilibria. 1. United-atom description of n-alkanes. The Journal of Physical Chemistry B 1998, 102, 2569–2577.

[10] McCready, C.; Sladekova, K.; Conroy, S.; Gomes, J. R.; Fletcher, A. J.; Jorge, M. Quantifying the uncertainty of force field selection on adsorption predictions in MOFs. Journal of Chemical Theory and Computation 2024, 20, 4869–4884.

[11] Park, J.; Howe, J. D.; Sholl, D. S. How reproducible are isotherm measurements in metal– organic frameworks? Chemistry of Materials 2017, 29, 10487–10495.

[12] Bingel, L. W.; Walton, K. S.; Sholl, D. S. Experimentally verified alkane adsorption isotherms in nanoporous materials from literature meta-analysis. Journal of Chemical & Engineering Data 2022, 67, 1757–1764.

[13] Simon, C. M.; Smit, B.; Haranczyk, M. pyIAST: Ideal adsorbed solution theory (IAST) Python package. Computer Physics Communications 2016, 200, 364–380.

[14] Sharma, S.; Balestra, S. R.; Baur, R.; Agarwal, U.; Zuidema, E.; Rigutto, M. S.; Calero, S.; Vlugt, T. J.; Dubbeldam, D. RUPTURA: simulation code for breakthrough, ideal adsorption solution theory computations, and fitting of isotherm models. Molecular Simulation 2023, 49, 893–953.

[15] Hassan, M.; Yoon, S.; Chen, Y.; Kim, P.; Yun, H.; Kwon, H. T.; Bae, Y.-S.; Yoo, C.-Y.; Koh, D.-Y.; Hong, C.-S. et al. AIM: A user-friendly GUI workflow program for isotherm fitting, mixture prediction, isosteric heat of adsorption estimation, and breakthrough simulation. Computer Physics Communications 2026, 319, 109944, DOI: 10.1016/j.cpc.2025.109944.

[16] Dral, P. O.; Nawaz, H.; Ullah, A. Science Done on a Machine by a Machine: AI Agents in Computational Chemistry. arXiv preprint arXiv:2608.18508 2026, DOI: 10.48550/arXiv.2608.18508.

[17] Boiko, D. A.; MacKnight, R.; Kline, B.; Gomes, G. Autonomous chemical research with large language models. Nature 2023, 624, 570–578, DOI: 10.1038/s41586-023-06792-0.

[18] Bran, A. M.; Cox, S.; Schilter, O.; Baldassari, C.; White, A. D.; Schwaller, P. Augmenting large language models with chemistry tools. Nature Machine Intelligence 2024, 6, 525–535, DOI: 10.1038/s42256-024-00832-8.

[19] Zou, Y.; Cheng, A. H.; Aldossary, A.; Bai, J.; Leong, S. X.; Campos-Gonzalez-Angulo, J. A.; Choi, C.; Ser, C. T.; Tom, G.; Wang, A. et al. El Agente: An autonomous agent for quantum chemistry. Matter 2025, 8, 102263, DOI: 10.1016/j.matt.2025.102263.

[20] Pham, T. D.; Tanikanti, A.; Ke¸celi, M. ChemGraph as an agentic framework for computational chemistry workflows. Communications Chemistry 2026, 9, 33, DOI: 10.1038/s42004-025-01776-9.

[21] Campbell, Q.; Cox, S.; Medina, J.; Watterson, B.; White, A. D. MDCrow: automating molecular dynamics workflows with large language models. Machine Learning: Science and Technology 2026, 7, 025037, DOI: 10.1088/2632-2153/ae4b07.

[22] Wang, Z.; Huang, H.; Zhao, H.; Xu, C.; Zhu, S.; Janssen, J.; Viswanathan, V. DREAMS: Density functional theory based research engine for agentic materials simulation. arXiv preprint arXiv:2507.14267 2025,

[23] Liu, J.; Zhu, T.; Ye, C.; Fang, Z.; Weng, H.; Wu, Q. VASPilot: MCP-facilitated multi-agent intelligence for autonomous VASP simulations. Chinese Physics B 2025, 34, 117106, DOI: 10.1088/1674-1056/ae0681.

[24] Chaudhari, A.; Ock, J.; Barati Farimani, A. Modular large language model agents for multi-task computational materials science. Communications Materials 2026, 7, 131, DOI: 10.1038/s43246-025-00994-x.

[25] Soleymanibrojeni, M.; Aydin, R.; Guedes-Sobrinho, D.; Dias, A. C.; Piotrowski, M. J. et al. GENIUS: an agentic AI framework for autonomous design and execution of simulation protocols. Communications Materials 2026, 7, 115, DOI: 10.1038/s43246-026-01167-0.

[26] Vriza, A.; Kornu, U.; Koneru, A.; Chan, H.; Sankaranarayanan, S. K. R. S. Multi-agentic AI framework for end-to-end atomistic simulations. Digital Discovery 2026, 5, 440–452, DOI: 10.1039/D5DD00435G.

[27] Rothfarb, S.; Davis, M. C.; Matanovic, I.; Li, B.; Holby, E. F.; Kort-Kamp, W. J. Hierarchical Multi-agent Large Language Model Reasoning for Autonomous Heterogeneous Catalyst Discovery. npj Computational Materials 2026, 12, 309, DOI: 10.1038/s41524-026-02139-1.

[28] Liu, G.; Yang, S.; Zhong, Y. Masgent: an AI-assisted materials simulation agent. Digital Discovery 2026, 5, 2151–2171, DOI: 10.1039/D6DD00043F.

[29] Somasundaram, R.; Habib, A.; Dang, K.; Shivakumar, S.; Hill, R. G.; Wimmer, G.; Mishra, A.; Pachalieva, A.; Lui, A.; Viswanathan, H. et al. An Agentic Orchestration of Atomistic Simulations. arXiv preprint arXiv:2607.22596 2026, DOI: 10.48550/arXiv.2607.22596.

[30] Huang, H.; Li, Y.; Xu, A.; Liu, B.; Wang, C.; Tang, K.; Yang, N.; Liu, S.; Liu, H.; Zhang, P. et al. ALKEMIE Agent: an autonomous platform for computational materials design. arXiv preprint arXiv:2608.15776 2026, DOI: 10.48550/arXiv.2608.15776.

[31] Yang, F.; Evans, J. D. QUASAR: A universal autonomous system for atomistic simulation and a benchmark of its capabilities. Journal of Chemical Information and Modeling 2026, 66, 5911–5918, DOI: 10.1021/acs.jcim.6c00295.

[32] Ghafarollahi, A.; Buehler, M. J. Automating alloy design and discovery with physics-aware multimodal multiagent AI. Proceedings of the National Academy of Sciences 2025, 122, e2414074122, DOI: 10.1073/pnas.2414074122.

[33] Shi, Z.; Xin, C.; Huo, T.; Jiang, Y.; Wu, B.; Chen, X.; Qin, W.; Ma, X.; Huang, G.; Wang, Z.; Jing, X. A fine-tuned large language model based molecular dynamics agent for code generation to obtain material thermodynamic parameters. Scientific Reports 2025, 15, 10295, DOI: 10.1038/s41598-025-92337-6.

[34] Kang, Y.; Kim, J. ChatMOF: an artificial intelligence system for predicting and generating metal-organic frameworks using large language models. Nature Communications 2024, 15, 4705, DOI: 10.1038/s41467-024-48998-4.

[35] Lee, J.; Bae, T.; Kim, J. SimMOF: AI agent for Automated MOF Simulations. arXiv preprint arXiv:2603.29152 2026, DOI: 10.48550/arXiv.2603.29152.

[36] Petkovi´c, M.; Menkovski, V.; Calero, S. Towards Fully Automated Molecular Simulations: Multi-Agent Framework for Simulation Setup and Force Field Extraction. arXiv preprint arXiv:2509.10210 2025,

[37] Gooijer, S.; Capelo-Avil´es, S.; Sharma, S.; Giancola, S.; Gal´an-Mascaros, J.; Vlugt, T.; Dubbeldam, D.; Vicent-Luna, J.; Calero, S. TAMOF-1 for capture and separation of the main flue gas components. Journal of Materials Chemistry A 2025, 13, 16879–16892, DOI: 10.1039/D5TA01362C.

[38] Luna-Triguero, A.; Vicent-Luna, J.; Madero-Castro, R.; G´omez-Alvarez, P.; Calero, S.<sup>´</sup> Acetylene storage and separation using metal–organic frameworks with open metal sites. ACS applied materials & interfaces 2019, 11, 31499–31507, DOI: 10.1021/acsami.9b09010.

[39] Wasik, D. O.; Vicent-Luna, J. M.; Luna-Triguero, A.; Dubbeldam, D.; Vlugt, T. J.; Calero, S. The impact of metal centers in the M-MOF-74 series on carbon dioxide and hydrogen separation. Separation and Purification Technology 2024, 339, 126539, DOI: 10.1016/j.seppur.2024.126539.

[40] Luna-Triguero, A.; S lawek, A.; S´anchez-de Armas, R.; Guti´errez-Sevillano, J. J.; Ania, C. O.; Parra, J. B.; Vicent-Luna, J. M.; Calero, S. π-Complexation for olefin/parafin separation using aluminosilicates. Chemical Engineering Journal 2020, 380, 122482, DOI: 10.1016/j.cej.2019.122482.

[41] Harris, J. G.; Yung, K. H. Carbon dioxide’s liquid-vapor coexistence curve and critical properties as predicted by a simple molecular model. The Journal of Physical Chemistry 1995, 99, 12021–12024.

[42] Garcia-Sanchez, A.; Ania, C. O.; Parra, J. B.; Dubbeldam, D.; Vlugt, T. J.; Krishna, R.; Calero, S. Transferable force field for carbon dioxide adsorption in zeolites. The Journal of Physical Chemistry C 2009, 113, 8814–8820.

[43] Fang, H.; Kamakoti, P.; Ravikovitch, P. I.; Aronson, M.; Paur, C.; Sholl, D. S. First principles derived, transferable force fields for CO 2 adsorption in Na-exchanged cationic zeolites. Physical Chemistry Chemical Physics 2013, 15, 12882–12894.

[44] Vuji´c, B.; Lyubartsev, A. P. Transferable force-field for modelling of CO2, N2, O2 and Ar in all silica and Na+ exchanged zeolites. Modelling and simulation in Materials science and Engineering 2016, 24, 045002.

[45] Bai, P.; Tsapatsis, M.; Siepmann, J. I. TraPPE-zeo: Transferable potentials for phase equilibria force field for all-silica zeolites. The Journal of Physical Chemistry C 2013, 117, 24375–24387.

[46] Findley, J. M.; Boulfelfel, S. E.; Fang, H.; Muraro, G.; Ravikovitch, P. I.; Sholl, D. S. A transferable force field for predicting adsorption and difusion of hydrocarbons and small molecules in silica zeolites with coupled-cluster accuracy. The Journal of Physical Chemistry C 2021, 125, 8418–8429.

[47] Rapp´e, A. K.; Casewit, C. J.; Colwell, K. S.; Goddard III, W. A.; Skif, W. M. UFF, a full periodic table force field for molecular mechanics and molecular dynamics simulations. Journal of the American chemical society 1992, 114, 10024–10035.

[48] Potof, J. J.; Siepmann, J. I. Vapor–liquid equilibria of mixtures containing alkanes, carbon dioxide, and nitrogen. AIChE journal 2001, 47, 1676–1682.

[49] Willems, T. F.; Rycroft, C. H.; Kazi, M.; Meza, J. C.; Haranczyk, M. Algorithms and tools for high-throughput geometry-based analysis of crystalline porous materials. Microporous and Mesoporous Materials 2012, 149, 134–141.

# Supporting Information for

# RaspAssistant as an Agentic Framework for Automating Adsorption Simulation Research in Porous Materials

Marko Petkovi´c<sup>1,3</sup>, Susanne Gooijer<sup>1</sup>, Jos´e Manuel Vicent-Luna<sup>1</sup>, Vlado Menkovski<sup>2,3</sup>, and Sof´ıa Calero<sup>∗1,3</sup>

<sup>1</sup>Department of Applied Physics and Science Education, Eindhoven University of Technology, Eindhoven, The Netherlands

<sup>2</sup>Department of Mathematics and Computer Science, Eindhoven University of Technology, Eindhoven, The Netherlands

<sup>3</sup>Eindhoven Artificial Intelligence Systems Institute (EAISI), Eindhoven University of Technology, Eindhoven, The Netherlands

Reader guide. Expanded research context is in Section 1. Run-level benchmark results are in Section 2. Section 3 describes the framework. Detailed scoring, original task prompts, and scientific criteria are in Sections 5.1, 5.2, and 5.3. Fitting and energetic validation are in Sections 5.5 and 5.6. Prompt headings identify the study number used in the results.

## 1 Additional Research Context

## 1.1 Additional Context on Agentic Scientific Systems

Recent advances in artificial intelligence (AI) have led to the emergence of agentic scientific systems, in which large language models (LLMs) are combined with planning, memory, tool use, and iterative reasoning to execute complex scientific tasks <sup>1</sup>. In experimental science, Chem-Crow<sup>2</sup> and Coscientist<sup>3</sup> demonstrate how LLM-based agents can integrate literature knowledge, planning, and tool use, while autonomous materials laboratories <sup>4–6</sup> illustrate the broader closed-loop integration of computation and experiment. These developments suggest a broader shift from AI systems that assist with isolated subtasks toward systems that can coordinate multi-step scientific workflows.

A similar trend is emerging in computational chemistry and materials simulation. The El Agente family <sup>7–10</sup> has shown that multi-agent architectures can translate natural-language instructions into executable quantum-chemistry, molecular-editing, and quantum-simulation workflows. Systems including DREAMS<sup>11</sup>, VASPilot<sup>12</sup>, MASTER<sup>13</sup>, MatSciAgent<sup>14</sup>, Masgent<sup>15</sup>, and an end-to-end multi-agent framework for LAMMPS workflows<sup>16</sup> extend this approach to tasks such as simulation setup, convergence testing, job submission, error recovery, structure generation, data retrieval, and molecular dynamics. More recent systems support autonomous interatomic-potential selection and validation through URSA <sup>17</sup>, traceable multimethod materials workflows through ALKEMIE Agent <sup>18</sup>, and adaptive planning and execution across several simulation methods through QUASAR<sup>19</sup>. AtomAgents combines multi-agent reasoning with physics-based alloy design<sup>20</sup>, while MDAgent generates, executes, and iteratively reviews LAMMPS scripts for thermodynamic calculations <sup>21</sup>. These implementations expose simulation control at diferent levels. For example, ChemGraph <sup>22</sup> coordinates computational chemistry through structured interfaces to tools such as the Atomic Simulation Environment (ASE), whereas GENIUS<sup>23</sup> generates and revises Quantum ESPRESSO input files within a knowledge-graph-guided finite-state workflow. Benchmarking of LLM-generated LAMMPS inputs<sup>24</sup> further illustrates the challenge of combining low-level control with scientifically valid setup. Thus, flexibility must be assessed with respect to the decisions exposed by a particular interface and workflow rather than treated as a binary property of an agentic framework.

Within porous materials, domain-specific agentic automation is developing rapidly, as surveyed by Yu and co-workers <sup>25</sup>. SimMOF<sup>26</sup> connects natural-language queries to $\mathrm { Z e o { + } { + } ^ { 2 7 } }$ RASPA<sup>28</sup>, VASP<sup>29</sup>, and LAMMPS<sup>30</sup>, and demonstrates geometric analysis, adsorption, difusion, binding-energy analysis, and methane-uptake screening. SimMOF generates simulation inputs using LLM reasoning and retrieved literature, and can revise inputs and rerun calculations during error recovery<sup>26</sup>. Our previous multi-agent framework<sup>31</sup> demonstrated literatureinformed force-field extraction and automated RASPA setup. Other recent MOF studies combine multi-agent evaluation with molecular simulation and process modelling for CO -capture screening<sup>32</sup>, or literature-derived rules, GCMC, and machine learning in a human-supervised radon-capture workflow<sup>33</sup>. Agent-assisted sorption campaigns have also combined deterministic GCMC–MD execution with selective LLM review in clay–water desorption <sup>34</sup>. These studies demonstrate connected simulation workflows, input preparation, and screening. Among the systems discussed here, we found no reported evaluation reproducing complete published adsorption and separation studies across the combined sequence of force-field selection or adaptation, experimental validation, pure-component and mixture simulations, and downstream IAST or breakthrough analysis.

## 2 Detailed Workflow-Reproduction Results

## 2.1 Study 1: $\mathbf { C O } _ { 2 } / \mathbf { N } _ { 2 }$ Separation in TAMOF-1.

Table S1: Multi-agent RaspAssistant performance and the single-agent baseline on Study 1.
<table><tr><td>Run</td><td>Workflow</td><td>Simulations</td><td>Analysis</td><td>Overall</td></tr><tr><td>Run 1</td><td>10.00</td><td>10.00</td><td>10.00</td><td>10.00</td></tr><tr><td>Run 2</td><td>10.00</td><td>9.00</td><td>10.00</td><td>9.60</td></tr><tr><td>Run 3</td><td>10.00</td><td>9.25</td><td>9.50</td><td>9.60</td></tr><tr><td>Multi-agent average</td><td>10.00</td><td>9.42</td><td>9.83</td><td>9.73</td></tr><tr><td>Single-agent baseline</td><td>9.13</td><td>9.13</td><td>8.75</td><td>9.05</td></tr></table>

As summarized in Table S1, the three multi-agent runs completed the requested purecomponent, mixture, ideal adsorbed solution theory (IAST), and breakthrough workflow. Run 1 received full credit. Runs 2 and 3 lost points mainly for limited formal convergence evidence. Run 2 also used a slightly reduced cutof, and Run 3 made one overgeneralized temperaturetrend statement. Breakthrough disagreement with experiment was not penalized. The singleagent baseline scored 9.05: its final pure, mixture, and breakthrough outputs were complete, with deductions for the absent IAST–GCMC check, partial breakthrough-validation coverage, provisional breakthrough fits, and limited qualification of selectivities when $\mathrm { N _ { 2 } }$ loading was small.

## 2.2 Study 2: $\mathbf { C _ { 2 } H _ { 2 } } / \mathbf { C O _ { 2 } }$ and $\mathbf { C _ { 2 } H _ { 2 } } / \mathbf { C H _ { 4 } }$ Separation in Open-Metal-Site MOFs.

Table S2: Multi-agent RaspAssistant performance and the single-agent baseline on Study 2.
<table><tr><td>Run</td><td>Workflow</td><td>Simulations</td><td>Analysis</td><td>Overall</td></tr><tr><td>Run 1</td><td>9.50</td><td>8.13</td><td>10.00</td><td>9.05</td></tr><tr><td>Run 2</td><td>10.00</td><td>9.00</td><td>10.00</td><td>9.60</td></tr><tr><td>Run 3</td><td>8.88</td><td>8.56</td><td>8.75</td><td>8.73</td></tr><tr><td>Multi-agent average</td><td>9.46</td><td>8.56</td><td>9.58</td><td>9.13</td></tr><tr><td>Single-agent baseline</td><td>7.50</td><td>8.56</td><td>8.13</td><td>8.05</td></tr></table>

As summarized in Table S2, the three multi-agent runs completed the principal purecomponent, mixture, binding-geometry, and adsorption-site workflow. Run 1 lost points for partial heat coverage, unresolved missing-interaction warnings, limited independent convergence evidence, and a small high-pressure fugacity allowance. Its repeated replica requests were treated as a resource-planning opportunity rather than an accuracy failure. Run 2 lost points for limited convergence evidence and the same fugacity allowance. Run 3 transparently excluded Fe-MOF-74 from unsupported mixture predictions after its $\mathrm { { C _ { 2 } H _ { 2 } } }$ interaction could not be validated, but lost points for sparse Fe-MOF-74 validation and heat coverage, incomplete mixture coverage, limited convergence evidence, and a site-analysis choice based on non-matching pressure endpoints. The negative binding-geometry result and numerical disagreement with experiment were not penalized. The single-agent baseline scored 8.05, with deductions for omitted Fe-MOF-74 result families, unavailable heat validation, and an indirect site analysis.

## 2.3 Study 3: Metal Center Efects in M-MOF-74.

Table S3: Multi-agent RaspAssistant performance and the single-agent baseline on Study 3.
<table><tr><td>Run</td><td>Workflow</td><td>Simulations</td><td>Analysis</td><td>Overall</td></tr><tr><td>Run 1</td><td>8.69</td><td>8.75</td><td>8.00</td><td>8.58</td></tr><tr><td>Run 2</td><td>9.50</td><td>9.75</td><td>9.50</td><td>9.60</td></tr><tr><td>Run 3</td><td>8.50</td><td>8.50</td><td>8.50</td><td>8.50</td></tr><tr><td>Multi-agent average</td><td>8.90</td><td>9.00</td><td>8.67</td><td>8.89</td></tr><tr><td>Single-agent baseline</td><td>6.88</td><td>7.25</td><td>5.63</td><td>6.78</td></tr></table>

As summarized in Table S3, the three multi-agent runs completed pure-component validation, 298 K fitting, mixture checks, breakthrough, and adsorption-site analysis, and used direct mixture GCMC to qualify unstable IAST predictions. Run 1 lost points mainly because its $\mathrm { H _ { 2 } }$ and heat validation used the wrong metal-site normalization, with smaller deductions for convergence evidence and site interpretation. Run 2 lost points for incomplete $\mathrm { H _ { 2 } }$ heat validation, a small high-pressure assumption allowance, and an overly fine breakthrough ordering among close cases. Run 3 used the wrong $\mathrm { H _ { 2 } / h e a t }$ basis and received smaller deductions for convergence evidence and reliance on qualified breakthrough rankings. These normalization issues did not invalidate independently normalized mixture results. The single-agent baseline scored 6.78 because it lacked a valid 298 K $\mathrm { H _ { 2 } }$ process basis, complete all-feed and breakthrough validation, and full-metal-site geometry analysis.

## 2.4 Study 4: Olefin/Parafin Separation with Zeolites.

Table S4: Multi-agent RaspAssistant performance and the single-agent baseline on Study 4.
<table><tr><td>Run</td><td>Workflow</td><td>Simulations</td><td>Analysis</td><td>Overall</td></tr><tr><td>Run 1</td><td>9.50</td><td>8.50</td><td>8.75</td><td>8.95</td></tr><tr><td>Run 2</td><td>9.00</td><td>8.31</td><td>8.75</td><td>8.68</td></tr><tr><td>Run 3</td><td>8.75</td><td>9.25</td><td>7.50</td><td>8.70</td></tr><tr><td>Multi-agent average</td><td>9.08</td><td>8.69</td><td>8.33</td><td>8.78</td></tr><tr><td>Single-agent baseline</td><td>9.38</td><td>8.63</td><td>5.00</td><td>8.20</td></tr></table>

As summarized in Table S4, Run 1 completed all principal calculation families and retained detailed simulation and analysis records. Its main deduction was propagation of unity rather than final Rosenbluth factors for flexible $\mathrm { C _ { 4 } \mathrm { - C _ { 6 } } }$ adsorbates: $\mathrm { C _ { 3 } }$ remained valid, the $\mathrm { C _ { 4 } / C _ { 5 } }$ results were retained only for qualitative interpretation, and $\mathrm { C } _ { 6 }$ was not quantitatively interpretable. Smaller deductions reflected limited convergence evidence and incomplete synthesis of available rankings. Run 2 completed the validation, 288-mixture, and five-map grids and correctly applied 298 K Rosenbluth factors to mixtures. It lost points for unity factors in the $\mathrm { C } _ { 6 }$ isobars, an undersized ITQ-29 cell, omitted cation-position restarts, and overprecise small-denominator selectivities. Run 3 completed all requested simulation families, including 832 direct-mixture calculations with appropriate periodic cells and correctly propagated 298 K Rosenbluth factors. Its deductions were localized to the $\mathrm { C } _ { 6 }$ validation series, which used unity rather than the calculated temperature-dependent factors and was not compared directly with experiment, together with limited convergence evidence and overprecise reporting of some small-denominator selectivities. The single-agent baseline scored 8.20. After an autonomous repair completed its grid, deductions remained for the undersized ITQ-29 cell and insuficient qualification of small-denominator and site-mechanism claims.

## 3 Computational Methods

## 3.1 RaspAssistant

## 3.1.1 Overview

RaspAssistant is a hierarchical multi-agent framework designed to automate porous materials simulations, from literature-informed force field construction to simulation execution and analysis. As illustrated in Figure 1, the system integrates two main functional components: (i) a simulation pipeline responsible for setting up, executing, and analyzing molecular simulations, and (ii) a force-field extraction pipeline that retrieves and structures information from scientific publications. These components are tightly coupled, enabling the system to incorporate literature-derived knowledge directly into simulation workflows.

Agents receive a natural-language description of the research objective, together with a structured folder containing relevant data, including experimental measurements, structural information (e.g., CIF files), and force field definitions when available. This design allows the framework to reproduce published workflows as closely as possible, while reducing computational cost and ensuring controlled evaluation. The primary output of the system is a structured report addressing the requested task (e.g., gas separation performance), although evaluation focuses on the full workflow, including planning, simulation validity, and analysis.

## 3.1.2 System Architecture

The framework follows a hierarchical organization, with a Supervisor Agent responsible for highlevel scientific planning and coordination. Based on the input task and available resources, the Supervisor Agent decomposes the problem into a sequence of subtasks, determines the required simulations, and delegates responsibilities to specialized agents. It can also invoke a File agent for managing directory structures and organizing intermediate and final results.

The simulation pipeline consists of several agents. The Setup agent generates simulation inputs by combining information from the task description, the RASPA manual, and prior experience stored in memory. It is supported by auxiliary agents, including an Example agent that retrieves the location of relevant simulation examples from the RASPA manual and a Python agent for automated file manipulation and template generation.

A distinction is made between simulation setup and execution depending on the complexity of the underlying software. For RASPA simulations, input generation is non-trivial, and responsibilities are split between a Setup agent, which prepares simulation inputs, and a dedicated RASPA agent, which executes simulations and performs initial diagnostics (e.g., detection of runtime errors or incomplete jobs). In contrast, for RUPTURA simulations, setup and execution are suficiently lightweight to be handled by a single agent. In addition to running RUPTURA tasks, the RUPTURA agent performs basic validation of results, including inspection of isotherm fits and derived quantities such as IAST predictions and breakthrough curves.

In addition, an Analysis agent performs in-depth interpretation of simulation outputs, including post-processing, consistency checks, and visualization. RASPA outputs can contain thousands of lines of unstructured text. After each simulation, they are converted to JSON to facilitate information extraction. The Analysis agent is further supported by the Python and Plotting agents, enabling flexible analysis workflows and generation of publication-ready figures.

Finally, the Force Field Fitting agent is responsible for diagnosing discrepancies between simulations and reference data and for developing or refining force-field parameters. The fitting harness is entered through a dedicated agent tool call and implemented as a state-machine controller. Each optimization attempt is structured into four phases: parameter selection, simulation, analysis, and a continue-or-terminate decision. The controller injects phase-specific instructions, delegates setup and execution to specialist agents, preserves every iteration separately, and limits the number of attempts. When the loop ends, it compacts the completed trajectory into a concise summary for subsequent reasoning.

In parallel, the force-field extraction pipeline enables the system to retrieve and process information from scientific publications. A Literature agent coordinates this process, delegating tasks to a Search agent that identifies relevant papers (using Semantic Scholar <sup>35</sup>) and a Reader agent that parses and interprets their content. Retrieved documents are automatically converted into structured representations using PyMuPDF4LLM <sup>36</sup>, enabling targeted extraction of key information such as force field parameters, simulation settings, and experimental data. A dedicated FF Writer agent translates extracted parameters into simulation-ready force field definitions, while an FF Adapter agent modifies or combines force fields to match specific use cases (e.g., adapting atom labels or combining parameter sets). While primarily used for force field construction, this pipeline can also extract other relevant information from the literature to support downstream simulation and analysis tasks.

## 3.1.3 Agent Coordination and Reasoning

RaspAssistant employs two distinct reasoning paradigms, depending on the complexity of the agent’s role. High-level agents, such as the Supervisor, Setup, Analysis, and Literature agents, follow an iterative plan–execute–verify–replan strategy. These agents first construct a plan based on the task and available resources, execute subtasks sequentially, and explicitly verify the outcome of each step. Verification is performed using model-based self-evaluation, where the agent assesses whether the produced output satisfies the intended objective. If necessary, the plan is updated to account for errors or newly available information.

In contrast, lower-level agents follow the ReAct paradigm <sup>37</sup>, in which they iteratively select actions based on the current state and available tools. This approach is well-suited for more structured tasks such as file manipulation, simulation execution, or example retrieval.

The agents use largely general-purpose reasoning strategies and tool interfaces. Outside the phase-controlled fitting protocol (Section 5.5), agents combine high-level planning with flexible tool use rather than following a fixed end-to-end study workflow. Because they can create and revise complete input files rather than select only from schema-defined parameters, the architecture can accommodate input variants and task sequences that were not explicitly anticipated when the workflow was designed. It can also extend to previously unseen simulation types when the required executable, documentation, and examples are available. This design naturally extends to how failures are handled: instead of relying on predefined error-handling routines, agents diagnose issues based on the current state and iteratively adjust their strategy, for example by modifying inputs, selecting alternative tools, or revising intermediate steps. These capabilities support adaptation in the presence of incomplete information or unexpected outcomes.

## 3.1.4 Tools, Memory, and Execution Environment

Agents interact with their environment through a set of tools bound to the underlying language models. These tools provide capabilities such as file system interaction (e.g., reading, writing, and listing files), execution of simulations, viewing images, and code execution. This abstraction enables agents to operate on real simulation workflows while maintaining a consistent interface.

The system is executed on a local machine, where simulation software is accessed through a wrapped interface that mimics command-line execution. Molecular simulations were executed with RASPA 2.0.47. To reduce computational cost during evaluation, previously computed results can be reused when simulation inputs are identical, allowing the benchmark to avoid repeating equivalent calculations without changing the inputs or scientific evidence available to the agents. For batches of nominally identical long simulations, execution was limited to one representative per batch to reduce redundant computation. One run resumed after correction of an external RUPTURA configuration error, without additional scientific guidance. During workflow execution, human actions were limited to benchmark acceleration, resource management, and infrastructure maintenance, excluding scientific decisions or interpretation.

Elapsed time is reported as campaign wall-clock duration. Token use is the sum of input and output tokens. The table reports approximate token totals and approximate API-equivalent costs under the model prices used during execution.

RaspAssistant incorporates both persistent and transient memory mechanisms. Each agent can access a local memory stored as structured markdown files, which captures relevant past experiences such as simulation setups, parameter choices, and observed issues. Persistent memory was disabled between benchmark invocations, so each run began without benchmark-specific information from earlier runs. A shared per-run memory was initialized separately for each invocation to track the current workflow and facilitate coordination between agents.

A detailed overview of the agents, their roles, and capabilities is provided in Table S5. The table lists configured model families. Values in parentheses give case-study configurations where they difer from workflow reproduction. Literature-agent assignments are taken from the frozen reviewer configurations. Exact model identifiers recorded in saved runs are supplied in the supporting data package. A separate GPT-5 mini routing agent passes requests to the scientific Supervisor. PEV denotes plan–execute–verify–replan, and R/W/P/V denote file reading, file writing, Python execution, and vision.

The supporting data package includes benchmark prompts, agent and model configurations, software versions, stopping conditions, and simulation-reuse rules. Its artifact manifest links workflow records to the corresponding simulation inputs, outputs, and derived analysis files.

Table S5: Agent architectures, configured model families, and capabilities. Parentheses indicate case-study configurations where they difer from workflow reproduction.
<table><tr><td>Agent</td><td>Type</td><td>Model</td><td>R</td><td>W</td><td>P</td><td>V</td><td>Other Tools</td></tr><tr><td>Supervisor</td><td>PEV</td><td>GPT-5.2</td><td>x</td><td>X</td><td>X</td><td>X</td><td>Agent delegation, fitting-loop en-</td></tr><tr><td>Setup</td><td>PEV</td><td>GPT-5.2</td><td>√</td><td>L</td><td></td><td></td><td>try</td></tr><tr><td>Example</td><td>ReAct</td><td>GPT-5 mini</td><td>V</td><td>x</td><td>X X</td><td>x X</td><td>RASPA manual RASPA examples</td></tr><tr><td>File</td><td>ReAct</td><td>GPT-5 mini</td><td>√</td><td>√</td><td></td><td></td><td></td></tr><tr><td>Analysis</td><td>PEV</td><td>GPT-5.4</td><td>V</td><td>√</td><td>V X</td><td>X √</td><td></td></tr><tr><td>Python</td><td>ReAct</td><td>GPT-5.2</td><td>V</td><td>√</td><td>V</td><td>X</td><td>Output file parsing</td></tr><tr><td>Plotting</td><td>ReAct</td><td>GPT-5.2 (5.1)</td><td>√</td><td>V</td><td>√</td><td>√</td><td>Output file parsing</td></tr><tr><td>RASPA</td><td>ReAct</td><td>GPT-5.1</td><td>√</td><td>X</td><td>X</td><td>x</td><td>RASPA</td></tr><tr><td>RUPTURA</td><td>ReAct</td><td>GPT-5.1</td><td>√</td><td>√</td><td>X</td><td>√</td><td>RUPTURA</td></tr><tr><td>FF Fitter</td><td>PEV</td><td>GPT-5.2</td><td>√</td><td>7</td><td>√</td><td>5</td><td>Output file parsing, FF fitting</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>tool</td></tr><tr><td>Literature Search</td><td>PEV</td><td>GPT-5 mini</td><td>V</td><td>X</td><td>x</td><td>x</td><td></td></tr><tr><td>Reader</td><td>ReAct</td><td>GPT-5 mini</td><td>V</td><td>X</td><td>X</td><td>x</td><td>Semantic Scholar</td></tr><tr><td></td><td>ReAct</td><td>GPT-5.2</td><td>√</td><td>√</td><td>X</td><td>V</td><td>Paper reading</td></tr><tr><td>FF Writer</td><td>ReAct</td><td>GPT-5.2</td><td>√</td><td>√</td><td>V</td><td>X</td><td>RASPA manual</td></tr><tr><td>FF Adapter</td><td>ReAct</td><td>GPT-5.2</td><td>√</td><td>L</td><td>X</td><td>x</td><td>RASPA manual</td></tr></table>

## 3.2 Workflow Reproduction Evaluation

We evaluate RaspAssistant on workflow reproduction tasks using four representative case studies drawn from adsorption and separation studies in porous materials. These benchmarks are designed to assess the ability of the system to reproduce complete computational workflows from published studies, including simulation setup, execution, and analysis. In all cases, literaturederived force field parameters are provided as input, ensuring that the evaluation focuses on workflow automation and scientific reasoning rather than force field construction or fitting. Each case corresponds to a full end-to-end simulation pipeline derived from the literature, enabling controlled evaluation of agentic performance in realistic scientific scenarios.

Each benchmark provides (i) a natural-language task description, (ii) structural information (CIF files), (iii) experimental adsorption data where available, and (iv) pre-defined force field parameters extracted from the literature. The objective is to reproduce the computational workflow of the corresponding study as faithfully as possible, while allowing for minor variations in implementation.

As an architectural baseline, each benchmark was also executed once by a single GPT-5.4 agent. It received the same scientific request, input data, file workspace, simulation backends, tracking mechanisms, and the deduplicated union of the primitive tools available to the specialized simulation agents, but no delegation or handof tools. The single agent maintained a dependency-ordered task list in one history. At an approximate occupancy of 70% of its nominal 200,000-token context, older messages were replaced by a structured checkpoint while the original request and 12 most recent messages were retained, and the removed messages were archived. Baseline outputs were scored with the same rubric and evidence hierarchy as the multi-agent invocations. Because the multi-agent system uses a heterogeneous model mix, this is a descriptive architectural baseline rather than a model-, token-, or cost-matched ablation.

## 3.2.1 Benchmark Cases

Study 1: $\mathbf { C O } _ { 2 } / \mathbf { N } _ { 2 }$ Separation in TAMOF-1<sup>38</sup>. This case considers adsorption and separation of $\mathrm { C O _ { 2 } }$ and $\mathrm { N _ { 2 } }$ in the TAMOF-1 metal–organic framework. The workflow includes singlecomponent adsorption isotherms, validation against experimental data, ideal adsorbed solution theory (IAST) calculations, mixture adsorption simulations, and breakthrough curve analysis. This benchmark evaluates the ability of the system to reproduce a complete adsorption-toseparation pipeline, including both equilibrium and dynamic simulations.

Study 2: $\mathbf { C _ { 2 } H _ { 2 } / C O _ { 2 } }$ and $\mathbf { C } _ { 2 } \mathbf { H } _ { 2 } / \mathbf { C } \mathbf { H } _ { 4 }$ Separation in Open-Metal-Site $\mathbf { M O F s ^ { 3 9 } }$ This case focuses on adsorption and separation of acetylene, methane, and carbon dioxide in metal–organic frameworks with open metal sites. The workflow includes pure-component adsorption isotherms, validation against experimental data, thermodynamic and energetic analysis, mixture adsorption simulations, and selectivity evaluation. In addition, adsorption site analysis is performed to interpret molecular-scale binding mechanisms governing separation performance.

Study 3: Metal Center Efects in $\mathbf { M } \mathbf { - } \mathbf { M } \mathbf { O } \mathbf { F } \mathbf { - } 7 4 ^ { 4 0 }$ . This case investigates $\mathrm { { C O } _ { 2 } / H _ { 2 } }$ separation across the M-MOF-74 series with diferent metal centers. The workflow includes adsorption isotherm calculations and binding geometry analysis, followed by validation against experimental isotherms and adsorption geometries. Ideal adsorbed solution theory (IAST) and mixture grand canonical Monte Carlo simulations are then used to evaluate separation performance, followed by breakthrough simulations. The final objective is to rank metal centers according to their separation eficiency.

Study 4: Olefin/Parafin Separation in Cation-Exchanged Zeolites<sup>41</sup>. This case investigates olefin/parafin separation in LTA- and FAU-type zeolites containing Na and/or Ca charge-balancing cations. The adsorbates considered are $\mathrm { C _ { 3 } \mathrm { - } C _ { 6 } }$ hydrocarbons, allowing the efect of chain length on separation performance to be assessed. The workflow includes calculation of Rosenbluth weights for flexible hydrocarbons, validation against pure-component propane/propene adsorption isotherms and hexane/hexene isobars, and mixture adsorption simulations to determine separation selectivities. In addition, adsorption site analysis is performed to examine how cation identity, zeolite topology, and hydrocarbon chain length influence molecular binding and separation behaviour. The final objective is to identify structure–composition trends that govern olefin/parafin separation in cation-exchanged zeolites.

## 3.2.2 Evaluation Metrics

We score each workflow-reproduction run for workflow correctness, simulation validity, and analysis consistency. Table S6 summarizes the evaluation dimensions and weights. Detailed metric definitions and benchmark-specific workflow components are provided in Sections 5.1 and 5.3, respectively.

Table S6: Workflow-reproduction evaluation dimensions and weights.
<table><tr><td>Dimension</td><td>Definition</td><td>Weight</td></tr><tr><td>Workflow correctness</td><td>Completion of the expected workflow steps in the appro- priate order and using appropriate methodological choices.</td><td>40%</td></tr><tr><td>Simulation validity</td><td>Physical and computational validity of the simulations, in- cluding parameter selection and convergence.</td><td>40%</td></tr><tr><td>Analysis consistency</td><td>Internal support and calibration of the scientific interpre- tation, including uncertainty and model limitations.</td><td>20%</td></tr></table>

## 3.2.3 Workflow Scoring Procedure

Each workflow-reproduction run is evaluated for workflow correctness, simulation validity, and analysis consistency. Each dimension is reported on a 0–10 scale. The overall score uses the 40/40/20 weighting in Equation S1.

Workflow correctness. Each benchmark is decomposed into weighted steps derived from the task specification and corresponding literature workflow. Steps receive full, partial, or zero credit according to their coverage and methodological appropriateness. Step weights $w _ { j }$ and awarded fractions $c _ { j }$ enter Equation S2. The benchmark-specific steps and scoring anchors are provided in Sections 5.3 and 5.1.1, respectively.

Simulation validity. This metric assesses whether the simulations are physically meaningful and computationally sound. It considers the simulated systems and state points, force fields and electrostatics, periodic-cell and cutof choices, sampling, units, output integrity, and the validity of downstream methods. Model–experiment disagreement is not itself penalized. A systematic issue is scored once according to its scope, and only dependent result families are afected.

Analysis consistency. This metric assesses whether the conclusions are internally correct, supported by the run’s usable evidence, calibrated to model sensitivity and uncertainty, and based on an appropriate hierarchy of evidence. Agreement with the reference study and exact material rankings are not required. A cautious conclusion that the available results are inconclusive can therefore receive full credit, whereas unsupported precision or propagation of an invalid upstream model is penalized.

Sections 5.1.1 and 5.3 specify the scoring anchors, result-family validity labels, non-doublepenalization rules, and benchmark-specific workflow criteria.

The retained simulation inputs, completed outputs, derived analyses, and reports were scored by one human assessor using this rubric. Scores did not depend on numerical agreement with the reference publication.

## 3.3 Force-Field Extraction Evaluation

In addition to workflow reproduction, we evaluate the ability of RaspAssistant to extract and construct simulation-ready force fields from scientific literature. For this benchmark, we select eight representative force-field extraction cases spanning diferent classes of porous materials and interaction models. The goal is to assess whether the system can correctly identify, extract, and format force field parameters into RASPA-compatible input files.

We score force-field extraction for parameter correctness and RASPA readiness. Table S7 summarizes the evaluation dimensions and weights. Section 5.4 defines the metrics and Table S14 lists the benchmark cases.

Table S7: Force-field extraction evaluation dimensions and weights.
<table><tr><td>Dimension</td><td>Definition</td><td>Weight</td></tr><tr><td>Parameter correctness</td><td>Correctness and completeness of the extracted parameters relative to the reference parameter set.</td><td>60%</td></tr><tr><td>RASPA readiness</td><td>Fraction of generated force-field files that execute in RASPA without correction.</td><td>40%</td></tr></table>

Readiness was assessed by using the generated files in RASPA without manual correction. Parser errors, missing definitions, and inconsistent declared entry counts were treated as nonexecutable.

## 3.4 Case Studies

In addition to the structured benchmark tasks, we evaluate the system on two less prescriptive case studies. Whereas the benchmark tasks are designed to test whether RaspAssistant can reproduce predefined workflows with relatively explicit instructions, these case studies assess its ability to make useful decisions in a more open-ended setting. We therefore focus not only on the final numerical results, but also on the choices made by the system, including force field selection, validation strategy, simulation design, and interpretation of the resulting data. The two case studies are closely related to the workflow-reproduction benchmarks, but require the system to decide how to adapt the available information to a less fully specified research objective.

## 3.4.1 Case Study 1: $\mathbf { C _ { 2 } H _ { 2 } } / \mathbf { C O _ { 2 } }$ and $\mathbf { C } _ { 2 } \mathbf { H } _ { 2 } / \mathbf { C } \mathbf { H } _ { 4 }$ Separation in Open-Metal-Site MOFs

This case study examines whether RaspAssistant can identify and address limitations in a generic force field when applied to adsorption in metal–organic frameworks with open metal sites<sup>39</sup>. Instead of providing the final fitted force field from the reference study, we provide the initial force field used as the starting point in that work, consisting of UFF and Dreiding parameters for the framework and adsorbate definitions taken from prior literature. The system must therefore recognize that standard mixing rules are insuficient to describe the strong interaction between acetylene and open metal centres, diagnose the origin of the discrepancy with reference data, and propose an appropriate fitting strategy.

To keep the computational cost manageable, the study is restricted to two representative systems: IRMOF-1, which does not contain open metal sites, and Fe-MOF-74, which contains coordinatively unsaturated metal centres. This allows the system to distinguish between cases where the generic force field is expected to be adequate and cases where additional framework– adsorbate interactions are required. The evaluation focuses on whether RaspAssistant can identify the need for force field refinement, perform a controlled fitting procedure, and justify the resulting parameters in terms of adsorption behaviour and molecular interactions.

## 3.4.2 Case Study 2: High-Throughput Screening of Pure-Silica Zeolites for $\mathbf { C O } _ { 2 }$ Adsorption

This case study evaluates whether RaspAssistant can perform an end-to-end screening workflow for $\mathrm { C O _ { 2 } }$ adsorption in zeolites. The supplied dataset comprised 198 structures from the IZA Database of Zeolite Structures <sup>42;43</sup>. Of these, 197 passed automated input checks and were retained for screening. Experimental $\mathrm { C O _ { 2 } }$ adsorption isotherms in pure-silica MFI at several temperatures were also supplied<sup>44</sup>. Per-temperature mean absolute errors use linear-pressure interpolation of simulations at experimental points within the shared pressure domain. The aggregate MAE weights each temperature by its number of overlapping experimental points. Open simulation markers identify interquartile-range outliers in the archived block-drift diagnostic (block drift iqr outlier). These flagged points remain in the interpolation and error calculations. The task is to select a suitable force field for $\mathrm { C O _ { 2 } }$ adsorption in pure-silica zeolites, validate it against the available MFI experimental data, and then apply the selected force field to the accepted zeolite structures.

Unlike the workflow-reproduction benchmarks, the system is not given a fully specified simulation protocol. It must decide which candidate force fields to consider, how to compare simulated and experimental isotherms, and which criteria should be used to select the most reliable model. After validation, the chosen force field should be used to compute adsorption properties across the zeolite dataset under consistent simulation conditions. The resulting predictions are then analyzed to identify promising zeolite topologies and to relate adsorption performance to structural features such as pore size, topology, and adsorption capacity. This case study therefore tests the ability of RaspAssistant to combine literature-based force field selection, experimental validation, large-scale simulation setup, and post-processing into a complete materials screening workflow.

## 4 Additional Interpretation and Research Context

The force-field extraction and case study results further demonstrate the usefulness of this approach. The literature agents extracted simulation-ready force field parameters with nearperfect agreement to the reference data, showing that agentic systems can help address a common practical bottleneck in classical molecular simulation. In the open-metal-site MOF case study, RaspAssistant identified the failure of a generic force field, carried out a targeted fitting procedure, and used the resulting model for downstream mixture adsorption and site analysis. In the zeolite screening case study, the system selected a literature-based $\mathrm { C O _ { 2 } }$ force field through comparison with experimental MFI data and applied it consistently to a larger set of pure silica zeolites. Together, these examples show that the framework can support both reproduction of published workflows and less prescriptive simulation studies that require validation-based decision-making.

Together, these results position RaspAssistant as a system for autonomous execution of complete adsorption-simulation workflows after task specification and input provision. Its contribution lies in carrying validation, adaptation, and documentation across coupled simulation stages while retaining expert-defined criteria for physical adequacy.

Expanded research context is provided in Section 1. Here, flexibility refers to the range of scientific decisions the agents can make and revise across an adsorption study, beyond generating input files. Among the systems discussed in Section 1, we found no reported evaluation spanning force-field selection or adaptation, experimental validation, pure-component and mixture simulations, and process-level analysis through reproduction of complete published porous-materials adsorption and separation studies.

The final fitted parameters are compared with the literature force field in Table 4. Although the final adsorption isotherm is very similar to the literature result, the fitted parameters difer substantially. In the literature model, the correction is distributed across several framework atom types, including framework oxygen, carbon, hydrogen, and the metal site. In contrast, the force field obtained by RaspAssistant concentrates the correction primarily in a much stronger $\mathrm { F e { - } C _ { C _ { 2 } H _ { 2 } } }$ interaction, with additional weaker oxygen contributions. This indicates that the system was able to find an efective parameterization for the target adsorption behaviour, while following a diferent parameterization strategy from the literature force field.

After fitting the $\mathrm { { C _ { 2 } H _ { 2 } } }$ interactions, RaspAssistant proceeded with the remaining adsorption and separation analysis. The fitted force field was used to run mixture adsorption simulations and calculate $\mathrm { C _ { 2 } H _ { 2 } / C O _ { 2 } }$ and $\mathrm { C _ { 2 } H _ { 2 } / C H _ { 4 } }$ selectivities. As shown in Figure 4, the resulting selectivities are in overall good agreement with the values reported in the reference study, reproducing both the magnitude and the pressure dependence for Fe-MOF-74 and IRMOF-1. Some deviations are observed, particularly in pressure regions where the uptake of the less strongly adsorbed component, $\mathrm { C H _ { 4 } }$ or $\mathrm { C O _ { 2 } }$ , is very small. In these cases, the selectivity is highly sensitive to small absolute diferences in loading, so modest discrepancies in mixture uptake can lead to larger relative diferences in the selectivity. Overall, this comparison shows that the fitted force field is not only able to reproduce the pure-component adsorption behaviour used during fitting, but also remains suitable for the downstream mixture separation analysis.

Finally, RaspAssistant performed an adsorption site analysis to interpret the molecular origin of the observed separation behaviour. The reference study visualized adsorption sites using spatial histograms, whereas RaspAssistant quantified the same efect through nearest Fe distance distributions. These distributions convey the same physical picture: $\mathrm { { C _ { 2 } H _ { 2 } } }$ is strongly localized near the open metal sites, while $\mathrm { C O _ { 2 } }$ and $\mathrm { C H _ { 4 } }$ are predominantly located farther away in the pore space. As shown in Figure 5, $\mathrm { { C _ { 2 } H _ { 2 } } }$ exhibits a sharp peak at short Fe distances in both mixtures, indicating preferential binding near the Fe sites. In contrast, $\mathrm { C O _ { 2 } }$ and $\mathrm { C H _ { 4 } }$ show broader distributions at larger distances, consistent with weaker localization near the open metal sites. Together with the isotherm and selectivity results, this analysis shows that RaspAssistant was able to proceed beyond force field fitting and mixture simulations to provide a molecular interpretation of the predicted separation behaviour.

RaspAssistant also examined relationships between the screening results and simple structural descriptors. Figure 9 shows the relationship between framework density and simulated $\mathrm { C O _ { 2 } }$ uptake at 1 kPa. Framework density showed a weak positive association with uptake at 1 kPa (Spearman $\rho = 0 . 2 8 4 )$ , while GIS, JBW, SIV, ATN, and YUG were positive residual outliers relative to this trend. Because the retained descriptor set did not include pore size, accessible volume, or channel-connectivity measures, these residuals cannot be assigned to a specific pore-scale mechanism.

The descriptor analysis also revealed a limitation of the workflow. RaspAssistant could use simple crystallographic and density-based quantities, but did not have direct access to more informative geometric pore descriptors such as accessible pore volume, pore-limiting diameter, largest included sphere, or helium void fraction. As a result, the system could identify densitynormalized outliers, but could not fully explain them in terms of pore size or accessible volume. A natural extension would be to connect the workflow to a geometric analysis package such as $\mathrm { Z e o + + ^ { 2 7 } }$ . This would give the agents direct access to pore descriptors during post-processing, allowing future screening workflows to relate adsorption performance more directly to channel geometry, pore accessibility, and confinement efects.

The agentic systems discussed above cover computational chemistry, materials simulation, and screening $7 - 1 4 ; 1 9 - 2 1 ; 2 6 ; 3 4$ . RaspAssistant is evaluated on the continuity of an adsorption study from model selection through process-level interpretation. Direct numerical comparison is not appropriate because these systems address diferent tasks and use diferent evaluation criteria. The present results instead demonstrate that a hierarchical agent architecture can preserve scientific context across several coupled tools and stages of a porous-materials workflow.

The benchmark also highlights experiment-cost awareness as an opportunity for further development. In Acetylene Run 1, the system adopted a conservative strategy of requesting 20–30 long replicas for individual conditions without access to a prospective estimate of queue size or computational cost. We limited execution to one copy per launch because equilibrated copies were expected to provide little additional information. This example suggests a practical route to more eficient automation: future versions could provide planning agents with predicted queue size and simulation cost, configurable budgets, and explicit uncertainty objectives before launching replicas.

One direction for future development is to broaden the set of domain-specific tools available to the agents. Integrating geometric characterization packages such as $\mathrm { Z e o { + } { + } ^ { 2 7 } }$ would provide direct access to descriptors including accessible pore volume, pore-limiting diameter, and largest included sphere, while charge-assignment and charge-equilibration methods could extend automated structure preparation. Interfaces to electronic-structure codes, force-field parameterization and validation utilities, and systematic convergence and uncertainty-analysis tools could further support model construction and verification. More broadly, the modular architecture provides a route for incorporating these and other capabilities as required by new materials classes and simulation workflows. These additions would strengthen reproducibility and resource awareness while preserving the role of expert judgement in defining acceptable

physical approximations.

## 4.1 Additional Workflow and Force-Field Extraction Details

Adsorption and separation studies typically couple several simulation stages. A typical workflow may include pure-component adsorption isotherms, validation against experimental data, mixture-performance evaluation using Ideal Adsorbed Solution Theory (IAST) <sup>45–47</sup> or direct mixture simulations, and process-level analysis through breakthrough simulations <sup>48;49</sup>. Tools such as pyIAST<sup>50</sup>, RUPTURA<sup>48</sup>, and $\mathrm { A I M ^ { 4 9 } }$ have lowered the barrier for individual stages of this workflow, but they do not by themselves automate the complete process from literaturegrounded model construction to validated scientific interpretation.

Executable rates are slightly lower despite the high IoU scores. These failures are not caused by incorrect parameter values or syntax errors, but by inconsistencies in global bookkeeping. RASPA force field files require the total number of interactions or atoms to be declared at the top of the file, consistent with the number of entries defined below. The observed failures arise when this count is incorrect, even though the interaction definitions themselves are accurate. These files cannot be executed until the declared entry counts are corrected.

The cost of force-field extraction remains low across all benchmarks. Higher costs occur when parameters are distributed across multiple sources $( \mathrm { C } _ { x } \mathrm { H } _ { x } , \mathrm { C O } _ { 2 } , \mathrm { N } _ { 2 } , \mathrm { O } _ { 2 }$ in zeolites), requiring additional retrieval steps, or when the force field contains a large number of parameters, as in the combined MOF case.

## 4.2 Open-Ended Case Studies

Unlike the workflow-reproduction benchmarks, these studies required higher-level modelling decisions rather than reproduction of a prescribed target workflow. Case Study 1 required the system to diagnose and refine an inadequate acetylene interaction model, whereas Case Study 2 required selection and experimental validation of a $\mathrm { C O _ { 2 } }$ force field before large-scale zeolite screening.

## 4.3 Supporting Data Package

The data package supporting this study is available on Zenodo and contains chronological workflow records in machine-readable JSON and self-contained HTML. Each record presents the original task, the top-level scientific interaction and its tool calls and responses, the instructions supplied for scientific delegations, and the outcome of each delegated task. For every workflowreproduction invocation, the package also includes the final report, the scientific files cited in that report, and the corresponding evaluation record. The package additionally provides the source data underlying the reported figures and tables, the final force-field definitions and numerical result summaries for the two open-ended case studies, and representative simulation inputs with source-access instructions for external structure files. An artifact manifest links these records to the corresponding scientific files.

## 5 Evaluation Metrics

Workflow reproduction and force-field extraction were assessed using the following rubric. Workflow-reproduction scores are reported on a 0–10 scale, whereas force-field extraction metrics are reported on a 0–1 scale unless stated otherwise.

## 5.1 Workflow-Reproduction Evaluation

Each workflow-reproduction run is evaluated for workflow correctness, simulation validity, and analysis consistency. The three dimensions contribute 40, 40, and 20 raw points, respectively.

Dimension scores are reported on a 0–10 scale, and the weighted overall score is

$$
S _ { \mathrm { o v e r a l l } } = 0 . 4 0 S _ { \mathrm { w o r k f l o w } } + 0 . 4 0 S _ { \mathrm { v a l i d i t y } } + 0 . 2 0 S _ { \mathrm { a n a l y s i s } } .\tag{S1}
$$

## 5.1.1 General Scoring Rules

The assessment is based on simulation inputs, completed raw outputs, derived data, plots, and the final report. Completed raw simulation output and efective settings reported by the engine take precedence. Next are simulation inputs, structures, force-field files, and source-data metadata, followed by derived tables, scripts, plots, and analysis records. The final scientific report has the lowest priority. Efective simulation behaviour is assessed rather than the literal presence of a keyword in an input file.

Numerical agreement with experiment or with the reference publication is not itself rewarded or penalized. The supplied force fields and process models can be sensitive or systematically inaccurate, including for breakthrough calculations. The evaluation instead asks whether the comparison was performed on a consistent basis and whether disagreement was identified and interpreted with appropriate caution. Exact material rankings are likewise not required when the diferences are within uncertainty or depend on the mixture model.

Workflow correctness. Each benchmark is decomposed into weighted atomic steps. A step normally receives full credit $( c _ { j } = 1 )$ , partial credit $( c _ { j } = 0 . 5 )$ , or zero credit $( c _ { j } = 0 )$ . Values of 0.25 or 0.75 are permitted only for a documented borderline case. The score is

$$
S _ { \mathrm { w o r k f l o w } } = 1 0 \frac { \sum _ { j } w _ { j } c _ { j } } { \sum _ { j } w _ { j } } ,\tag{S2}
$$

where $w _ { j }$ is the weight of workflow step j. Full credit requires the requested coverage and a suitable method. Partial credit represents a meaningful but incomplete or methodologically qualified attempt. Zero credit represents an absent step or one performed on the wrong central system or conditions.

Simulation validity. Simulation validity is divided into the seven criteria in Table S8. Each criterion is awarded a fraction $v _ { k }$ using the same full/partial/zero anchors, and the 40-point validity contribution is $\begin{array} { r } { P _ { \mathrm { v a l i d i t y } } = \sum _ { k } u _ { k } v _ { k } } \end{array}$ , where $u _ { k }$ is the listed point value. Model–experiment disagreement is not a validity error. Examples of validity errors are inconsistent units, wrong state points, ignored charges, inappropriate simulation settings, incomplete jobs, and use of a failed upstream model without qualification.

Table S8: Simulation-validity criteria used for every workflow-reproduction benchmark.
<table><tr><td>ID</td><td>Full-credit requirement</td><td>Points</td></tr><tr><td>V1</td><td>Correct structures, species, temperatures, pressures, and compositions for the re- quested result families.</td><td>6</td></tr><tr><td>V2</td><td>Supplied force fields and charge convention are used effectively. Required specialized interactions are retained. Effective electrostatics are confirmed from output.</td><td>7</td></tr><tr><td>V3</td><td>Appropriate supercell replication and cutoff settings are used for the benchmark.</td><td>6</td></tr><tr><td></td><td>V4 Equilibration and production are credible, and convergence, uncertainty, or an equiv- alent stability check is demonstrated.</td><td>6</td></tr><tr><td>V5</td><td>Pressure, loading, heat, geometry, and normalization conventions are reconciled and comparisons are restricted to overlapping domains.</td><td>7</td></tr><tr><td></td><td>V6 Inputs, completed outputs, warnings, derived values, and report claims are traceable.</td><td>4</td></tr><tr><td></td><td>V7 Fits, IAST or direct-mixture calculations, site analyses, and breakthrough dependen- cies are checked before quantitative use.</td><td>4</td></tr></table>

Nominal replicas count as independent samples only when their seeds or outputs demonstrate independence. For V5, using an ideal fugacity coeficient at 2.5–4 MPa is not a hard failure because the benchmark tasks do not prescribe an equation of state. A small deduction is appropriate only when the assumption is silent and exact high-pressure magnitudes or rankings are emphasized.

Analysis consistency. The 20-point analysis contribution is the sum of the four criteria in Table S9. Interpretation is graded against the run’s own scientifically usable evidence. A wellsupported conclusion that the available results are inconclusive or model-sensitive can receive full credit.

Table S9: Analysis-consistency criteria used for every workflow-reproduction benchmark.
<table><tr><td>ID</td><td>Full-credit requirement</td><td>Points</td></tr><tr><td>I1</td><td>Species, equations, signs, and reported values are internally correct. Claims follow the run&#x27;s usable outputs. A conclusion of “inconclusive&quot; is accepted</td><td>5</td></tr><tr><td>I2</td><td>when supported.</td><td>5</td></tr><tr><td>I3</td><td>Small denominators, force-field sensitivity, conflicting models, and sampling limita- tions are recognized.</td><td>5</td></tr><tr><td>I4</td><td>Experimental comparison, pure fits, IAST, direct mixtures, breakthrough, and site evidence are weighted according to their reliability. Mechanistic claims have direct support.</td><td>5</td></tr></table>

Result-family status and non-double-penalization. Each major result family is labelled valid, qualified, or invalid. A valid family supports the claims made. A qualified family supports only restricted or qualitative claims with a stated limitation. An invalid family is not usable for its claimed purpose. A systematic issue is deducted once according to its scope. Dependent analyses inherit the status of the faulty input, but do not each incur a repeated numerical penalty. For example, a complete pure-isotherm grid can receive workflow credit despite an invalid unit conversion. The conversion is penalized under validity, and full analysis credit remains possible when the limitation is identified and no unsupported claim is made. Conversely, an IAST–GCMC disagreement is not itself a failure, but ignoring it when making precise breakthrough claims reduces analysis consistency.

Completion is established from retained inputs and completed raw outputs. Summary labels or status fields are not suficient when requested result families are absent, while a documented repair can establish completion when it produces and verifies the missing outputs.

Single-agent baseline. Each workflow-reproduction benchmark was additionally run once with a single GPT-5.4 agent. The baseline received the same task, input data, workspace, simulation interfaces, tracking and memory mechanisms, and the deduplicated union of the primitive tools available to the specialized simulation agents, but it had no delegation or handof tools. All actions and their results were therefore retained in a single persistent conversation history. At an approximate occupancy of 70% of a nominal 200,000-token context window, and after completed tasks, older messages were replaced by a structured checkpoint while retaining the original request and 12 recent messages. Removed messages were archived verbatim. Failed attempts remain part of the completion record even when later autonomous repair completed the scientific grid. The same 40/40/20 rubric, evidence hierarchy, result-family status rules, and non-double-penalization procedure were used for the baseline and multi-agent runs. The comparison is not model-matched and is interpreted descriptively.

## 5.2 Workflow-Reproduction Task Prompts

The following natural-language prompts were supplied to the Supervisor Agent for the four workflow-reproduction benchmarks. Original spelling, notation, and task wording are retained.

## 5.2.1 Study 2: C2H2/CO2 and C2H2/CH4 separation

We want to investigate C2H2/CO2 and C2H2/CH4 separation using different MOF materials (Cu-BTC, Fe-MOF-74, IRMOF-1 and PCN-16).

We want to understand which material perfoms best at these separations, and how the performance changes with pressure.

Data:

\- input\_data/structures contains CIF files for Cu-BTC, Fe-MOF-74, IRMOF-1 and PCN-16. - input\_data/exp\_data/isotherms contains experimental pure component isotherms for CO2, C2H2 and CH4 in these materials at various temperatures.

\- input\_data/exp\_data/isosteric\_heat contains experimental isosteric heat of adsorption data for CO2, C2H2 and CH4 at various temperatures.

\- input\_data/exp\_data/binding\_geometry contains experimental binding geometry data for CO2 and C2H2 in Fe-MOF-74 at 298K.

\- input\_data/forcefield contains all necesarry force field files per material, including any necessary pseudo atom files and mixing rules.

Use this data to validate the results from simulations to understand the quality of the results.

Task: Measure the separation performance of the various MOF materials for C2H2/CO2 and C2H2/CH4 separation. Specifically, we want to:

\- Validate the simulation methods and results against the experimental data. This includes:

\- isotherms for pure CO2, C2H2 and CH4 at various temperatures for the different materials

\- isosteric heat of adsorption for CO2, C2H2 and CH4 at various temperatures for the different materials

\- binding geometry for CO2 and C2H2 at 298K

\- Investigate and compare pure component adsorption behaviour for CO2, C2H2 and CH4 at 318K for the different materials.

How do host-guest and guest-guest interactions influence the pure component adsorption behaviour of these molecules in the different materials, and how does this relate to the separation performance?

\- Measure and compare selectivity (S) for an equimolar mixture of C2H2/CO2 and C2H2/CH4 at 298K using mixture isotherms. We want to know how this metric compares the different materials at various pressures.

S = (x\_CO2/ x\_N2) / (y\_CO2/y\_N2), where x is the mole fraction in the adsorbed phase and y is the mole fraction in the gas phase. The pressure range should cover the same range as experimental data.

\- For the best MOF, analyze the adsorption sites for the equimolar mixture of C2H2/CO2 and C2H2/CH4 at 298K and 10 bar, and understand how this relates to the separation performance.

The pressures at which you evaluate should cover the same range as the experimental data (you do not need to use the same pressures).

Any simulation and results folders you create should be placed in the sims folder.

Output:

At the end you should create a report on the following:

\- The selectivity of the different materials for C2H2/CO2 and C2H2/CH4 separation at 298K, and how this metric changes with pressure.

\- The predicted adsorption sites for CO2 and C2H2 in an equimolar mixture for the best MOF, and how this relates to the separation performance.

\- A comparison of the simulation results with the experimental data, including any insights on the quality of the simulations and potential reasons for discrepancies. - Paths to any files you created with the results, including isotherm data, fitted isotherm parameters, IAST predictions, and adsorption site data and plots.

Reading note: Original request notation is retained. The mixture species and evaluation criteria are specified in Section 5.3.2.

## 5.2.2 Study 3: M-MOF-74 CO2/H2 separation

We want to investigate CO2/H2 separation for M-MOF-74 with various metal centres (Co, Cu, Fe, Mn, Ni, Zn).

Specifically, we want to know how the perfromance of different metal centres affects the separation performance for CO2/H2 separation.

Data:

\- input\_data/structures contains CIF files for M-MOF-74 for different metal centres (Co, Cu, Fe, Mn, Ni, Zn). Use the charges from the CIF files.

\- input\_data/exp\_data/isotherms contains experimental pure component isotherms for CO2 and H2 in M-MOF-74, at 77K for H2 and 298K for CO2. Different units are used for CO2 and H2.

\- input\_data/exp\_data/isosteric\_heat contains experimental isosteric heat of adsorption data for CO2 and H2 in M-MOF-74, at 77K for H2 and 298K for CO2. Different units are used for CO2 and H2.

\- input\_data/exp\_data/binding\_geometry contains experimental binding geometry data for CO2 and H2 in M-MOF-74 at 298K.

\- input\_data/forcefield contains all necesarry force field files and adsorbate definitions for CO2 and H2.

Use this data to validate the results from simulations to understand the quality of the results.

Task: Measure the separation performance of M-MOF-74 for CO2/H2 separation and rank the different metal centres based on their performance. Specifically, we want to:

\- Validate the simulation methods and results against the experimental data. This includes:

\- isotherms for pure CO2 and H2 at 298K for CO2 and 77K for H2

\- isosteric heat of adsorption for CO2 and H2 at 298K for CO2 and 77K for H2

\- binding geometry for CO2 and H2 at 298K

\- Determine which metal centre performs best for CO2/H2 separation, at 298K, 2.5MPa, column void fraction of 0.4, initial velocity 0.006791 and a length of 0.065m. Show how different CO2 mixture ratios (0.1/0.9, 0.2/0.8, 0.3/0.7, 0.4/0.6, 0.5/0.5 and 0.9/0.1 CO2/H2) affect the separation performance.

\- Investigate where adsorption occurs (adsorption sites) in an equimolar CO2/H2 mixture at 4000kPa for different metal centres, and how this relates to the separation performance.

Notes:

The pressures at which you evaluate should cover the same range as the experimental data (you do not need to use the same pressures).

Any simulation and results folders you create should be placed in the sims folder.

Output:

At the end you should create a markdown report on the following:

\- The separation performance of M-MOF-74 for CO2/H2 separation at the specified conditions, and how this changes with different metal centres and mixture ratios. - The predicted adsorption sites for CO2 and H2 in an equimolar mixture for different metal centres, and how this relates to the separation performance.

\- A comparison of the simulation results with the experimental data, including any insights on the quality of the simulations and potential reasons for discrepancies. - Paths to any files you created with the results, including isotherm data, fitted isotherm parameters, IAST predictions, and adsorption site data and plots.

## 5.2.3 Study 1: TAMOF-1 CO2/N2 separation

We want to investigate CO2/N2 separation for TAMOF-1.

Specifically, we want to know at which conditions (pressure, temperature, mixture ratio) TAMOF-1 performs best for separating CO2 from N2.

Data:

\- input\_data contains the CIF file for TAMOF-1 (Use the charges from the CIF file). Helium void fraction: 0.427

\- input\_data/exp\_data/isotherms contains experimental pure component isotherms for CO2 and N2 in TAMOF-1 at 298K, 323K, and 353K.

\- input\_data/exp\_data/breakthrough/CO2 contains experimental breakthroughs for single component CO2 in He carrier gas at various conditions (see readme for details).

\- input\_data/exp\_data/breakthrough/CO2\_N2 contains experimental breakthroughs for a mixture of CO2 and N2 in He carrier gas at a single condition (see readme for details).

\- input\_data/forcefield contains all necesarry force field files and adsorbate definitions for CO2 and N2.

Use this data to validate the results from simulations to understand the quality of the results.

Task: Measure the separation performance of TAMOF-1 for CO2/N2 separation using molecular simulations, at 298K, 323K, and 353K. Specifically, we want to:

\- Validate the simulation methods and results against the experimental data. This includes:

\- isotherms for pure CO2 and N2 at 298K, 323K, and 353K

\- single component breakthroughs for CO2 at various conditions

\- mixture breakthrough for CO2/N2 at a single condition

\- measure selectivity (S) and trade off selectivity and uptake (TSN). We want to know how these metrics change with pressure, temperature, and mixture ratio.

S = (x\_CO2/ x\_N2) / (y\_CO2/y\_N2), where x is the mole fraction in the adsorbed phase and y is the mole fraction in the gas phase.

TSN = ln(S) \* q\_CO2, where q\_CO2 is the uptake of CO2 in mol/kg, and S is the selectivity for theose conditions.

For S and TSN, the mixture ratios should be 0.1/0.9, 0.3/0.7, and 0.5/0.5 and 0.7/0.3 (CO2/N2). The pressure range should cover the same range as experimental data.

\- predict breakthrough curves for 10

Conditions for breakthrough curve predictions: 1 bar, 298K, 0.005 m/s, column length 5.8 cm and column void fraction 0.58.

## Notes:

The pressures at which you evaluate should cover the same range as the experimental data (you do not need to use the same pressures).

Any simulation and results folders you create should be placed in the sims folder.

## Output:

At the end you should create a report on the following:

\- The selectivity and TSN of TAMOF-1 for CO2/N2 separation at the specified conditions, and how these metrics change with pressure, temperature, and mixture ratio.

\- The predicted breakthrough curves for 10

\- A comparison of the simulation results with the experimental data, including any insights on the quality of the simulations and potential reasons for discrepancies. - Paths to any files you created with the results, including isotherm data, fitted isotherm parameters, IAST predictions, and breakthrough curve data and plots.

Reading note: The original request contains an incomplete feed-list line. The four prediction feeds used in evaluation are specified in Section 5.3.1.

## 5.2.4 Study 4: Olefin/parafin separation in zeolites

We want to investigate olefin/paraffin separation using different zeolite materials. We want to understand which material perfoms best at these separations, and how chain length affects the performance.

Data:

\- input\_data/structures contains CIF files for zeolites with LTA topolgy (CaA, LTA5A, ITQ-29) and FAU topology (FAU\_Si, FAU-2, NaY, NaX, CaX). For zeolites with cations, restart files with their positions are also included.

\- input\_data/exp\_data/isotherms contains experimental pure component isotherms for propane and propene for LTA5A and NaY.

\- input\_data/exp\_data/isobars contains experimental pure component isobars for hexane and 1-hexene for LTA5A and NaY.

\- input\_data/forcefield contains all necesarry force field files per material, including any necessary pseudo atom files and mixing rules.

Use this data to validate the results from simulations to understand the quality of the results.

Task: Measure the separation performance of the various zeolites for olefin/paraffin separation for C3-C6 hydrocarbons.:

\- Validate the simulation methods and results against the experimental data. This includes:

\- isotherms for pure propane and propene for LTA5A and NaY at 298K.

\- isobars for pure hexane and 1-hexene for LTA5A and NaY at different pressures.

\- Investigate the influence of framework and type/amount of cations:

\- Calculate average occupancy maps (histograms of adsorption positions) for the pure component adsorption behaviour of propene at 298K, 100kPA for zeolites of the FAU topology.

\- For all frameworks, measure and compare selectivity (S) for an equimolar mixture of olefins and paraffins for C3-C6 hydrocarbons at 298K and 100kPa, and investigate the influence of the chain length.

S = (x\_olefin/ x\_paraffin) / (y\_olefin/y\_paraffin), where x is the mole fraction in the adsorbed phase and y is the mole fraction in the gas phase. The pressure range should cover the same range as experimental data.

## Notes:

The pressures at which you evaluate should cover the same range as the experimental data (you do not need to use the same pressures).

Any simulation and results folders you create should be placed in the sims folder.

## Output:

At the end you should create a report on the following:

\- The selectivity of the different materials for olefin/paraffin separation for C3-C6 hydrocarbons at 298K, and how this metric changes with pressure.

\- The influence of type and amount of cations on the adsorption behaviour of propene at 298K, 100kPA for zeolites of the FAU topology, using average occupancy maps. - The influence of chain length on the separation performance by comparing selectivity for C3-C6 hydrocarbons at 298K, 100kPA.

\- A comparison of the simulation results with the experimental data, including any insights on the quality of the simulations and potential reasons for discrepancies. - Paths to any files you created with the results, including isotherm data, fitted isotherm parameters, IAST predictions, and adsorption site data and plots.

## 5.3 Benchmark-Specific Scientific Criteria

## 5.3.1 TAMOF-1 $\mathbf { C O } _ { 2 } / \mathbf { N } _ { 2 }$ separation

Table S10 defines the 40-point TAMOF-1 workflow. The adsorption helium void fraction is 0.427, whereas the predicted breakthrough column void fraction is 0.58. Experimental He-carrier breakthrough cases must be distinguished from the requested binary $\mathrm { C O _ { 2 } / N _ { 2 } }$ predictions. Direct mixture GCMC may be representative rather than exhaustive when IAST supplies the complete feed–temperature–pressure grid. Accurate breakthrough timing is not required. The score reflects whether the supplied experiments were compared correctly and whether disagreement was acknowledged.

Table S10: Workflow criteria for TAMOF-1 $\mathrm { C O _ { 2 } / N _ { 2 } }$ separation<sup>38</sup>.
<table><tr><td>ID</td><td>Expected workflow step</td><td>Points</td></tr><tr><td></td><td>TAM-W1 Inspect the structure, charges, experimental metadata, and distinct adsorption and column void fractions.</td><td>3</td></tr><tr><td>TAM-W2 Run pure</td><td> $\mathrm { C O _ { 2 } }$  and  $\mathrm { N _ { 2 } }$  GCMC at 298, 323, and 353 K over the experimental pressure range.</td><td>6</td></tr><tr><td></td><td>TAM-W3 Compare unit-matched pure simulations with experiment over the overlapping domains.</td><td>5</td></tr><tr><td></td><td>TAM-W4 Calculate selectivity and TSN over all four feeds, three temperatures, and the pressure range.</td><td>7</td></tr><tr><td>TAM-W5</td><td>Use IAST and representative direct mixture GCMC to assess mixture-model reliability.</td><td>4</td></tr><tr><td></td><td>TAM-W6 Compare breakthrough with the supplied pure-  $\mathrm { C O _ { 2 } } /$  He and  $\mathrm { C O _ { 2 } / N _ { 2 } / H e }$  periments.</td><td>6</td></tr><tr><td>TAM-W7 Predict 10, 15, 20, and 25%</td><td>&gt;0  $\mathrm { C O _ { 2 } }$  in  $\mathrm { N _ { 2 } }$  at 1 bar, 298 K, 0.005 5  $\mathrm { ~ m ~ s ~ } ^ { - 1 }$  , 5.8 cm, and column void fraction 0.58.</td><td>5</td></tr><tr><td></td><td>TAM-W8 Synthesize the results, limitations, and evidence paths in an auditable report.</td><td>4</td></tr></table>

The use of direct-mixture checks reflects known limitations of IAST in heterogeneous adsorbents<sup>47</sup>. Interpretation should distinguish selectivity from TSN, recognize instability when $\mathrm { N _ { 2 } }$ uptake is very small, and downgrade IAST or breakthrough conclusions when direct mixture

GCMC or experimental comparison exposes limitations. No exact selectivity or breakthrough agreement is prescribed.

![](images/a4fa5f479fae51b8f4eda38ae1d77dc72a33f39766edb7755a7346b4f57354d8.jpg)  
Figure S1: TAMOF-1 diagnostic from Study 1, multi-agent Run 3, comparing IAST and directmixture GCMC selectivities for the benchmark based on Ref. <sup>38</sup>. Deviations from the parity line motivate treating IAST-derived selectivities and downstream breakthrough predictions as qualified rather than requiring numerical agreement.

## 5.3.2 $\mathbf { C _ { 2 } H _ { 2 } } / \mathbf { C O _ { 2 } }$ and $\mathbf { C } _ { 2 } \mathbf { H } _ { 2 } / \mathbf { C } \mathbf { H } _ { 4 }$ separation

Table S11 defines the 40-point open-metal-site-MOF workflow. The benchmark covers Cu-BTC, Fe-MOF-74, IRMOF-1, and PCN-16 with their supplied material-specific force fields. Native experimental temperatures are used for validation, 318 K for the pure-component mechanistic study, and 298 K for mixture and site analyses. The supplied task prompt uses generic $\mathrm { C O _ { 2 } / N _ { 2 } }$ labels in its selectivity expression. For this benchmark, the expression applies to $\mathrm { C _ { 2 } H _ { 2 } / C O _ { 2 } }$ and $\mathrm { C _ { 2 } H _ { 2 } / C H _ { 4 } }$

Table S11: Workflow criteria for $\mathrm { C _ { 2 } H _ { 2 } / C O _ { 2 } }$ and $\mathrm { C _ { 2 } H _ { 2 } / C H _ { 4 } }$ separation<sup>39</sup>.
<table><tr><td>ID</td><td>Expected workflow step</td><td>Points</td></tr><tr><td></td><td>ACE-W1 Inventory the four structures, material-specific force fields, and experimental states.</td><td>3</td></tr><tr><td>ACE-W2 Validate pure</td><td> $\mathrm { C _ { 2 } H _ { 2 } , C O _ { 2 } }$  , and  $\mathrm { C H _ { 4 } }$  isotherms at their native experimental tem- 1 peratures.</td><td>6</td></tr><tr><td></td><td>ACE-W3 Validate adsorption heats using compatible loading bases and overlapping do- mains.</td><td>4</td></tr><tr><td>ACE-W4 Validate Fe-MOF-74</td><td> $\mathrm { C O _ { 2 } }$  and  $\mathrm { { C _ { 2 } H _ { 2 } } }$  binding geometries at 298 K.</td><td>3</td></tr><tr><td></td><td>ACE-W5 Perform the 318 K pure-component study with host-guest and guest-guest ev- idence.</td><td>6</td></tr><tr><td></td><td>ACE-W6 Run equimolar mixture GCMC for both gas pairs at 298 K over the pressure range.</td><td>8</td></tr><tr><td></td><td>ACE-W7 Select a candidate transparently and analyze both 298 K, 10 bar mixture tra- jectories or adsorption sites.</td><td>6</td></tr><tr><td></td><td>ACE-W8 Synthesize the comparison, uncertainty, limitations, and evidence paths.</td><td>4</td></tr></table>

Site analysis must use periodic distances, and open-metal-site or host–guest causal claims require trajectory, occupancy, geometry, or energy evidence. Replicas are required only if replicabased uncertainty is claimed. Identical replicas count as one efective sample. No particular winning material is prescribed, and large low-pressure selectivities should be qualified when the weak-component loading is statistically indistinguishable from zero.

![](images/631a837daf03cd4b53b87d35e3e89bdddb4cea30ed1d7c1bdb2d0bbd24bc47a5.jpg)  
Figure S2: Pure-component validation overview from Study 2, multi-agent Run 2, for the four MOF materials and supplied experimental reference data <sup>39</sup>. Material-dependent disagreement is not itself a scoring failure.

## 5.3.3 Metal-center efects in M-MOF-74

Table S12 defines the 40-point M-MOF-74 workflow across Co, Cu, Fe, Mn, $\mathrm { N i } ,$ and $\mathrm { Z n } .$ . The authoritative experimental conventions use bar for $\mathrm { C O _ { 2 } }$ isotherm pressure and mol $\mathrm { k g ^ { - 1 } }$ for loading at 298 K. $\mathrm { H _ { 2 } }$ isotherms use kPa for pressure and mol per metal site for loading at 77 K. For adsorption heats, $\mathrm { C O _ { 2 } }$ loading is expressed as $( N _ { \mathrm { C O _ { 2 } } } / N _ { M } ) \times 1 0 0 \%$ and $\mathrm { H _ { 2 } }$ loading as mol per metal site. Heats are in kJ mol<sup>−1</sup>. $\mathrm { { C O } _ { 2 } }$ binding uses the closest O–metal distance, whereas $\mathrm { H _ { 2 } }$ binding uses the H<sub>2</sub> centre-of-mass–metal distance. Both geometries are evaluated at 298 K.

Table S12: Workflow criteria for metal-center efects in M-MOF-74 <sup>40</sup>.
<table><tr><td>ID</td><td>Expected workflow step</td><td>Points</td></tr><tr><td>M74-W1</td><td>Audit all six materials and the experimental units and definitions.</td><td>3</td></tr><tr><td>M74-W2</td><td>Validate  $\mathrm { C O _ { 2 } }$  at 298 K and  $\mathrm { H _ { 2 } }$  at 77 K for all six metals.</td><td>6</td></tr><tr><td>M74-W3</td><td>Validate  $\mathrm { C O _ { 2 } }$  and  $\mathrm { H _ { 2 } }$  adsorption heats using the correct loading axes.</td><td>4</td></tr><tr><td>M74-W4</td><td>Validate  $\mathrm { C O _ { 2 } }$  and  $\mathrm { H _ { 2 } }$  binding geometries at 298 K.</td><td>4</td></tr><tr><td>M74-W5</td><td>Generate and audit 298 K pure  $\mathrm { C O _ { 2 } }$  and  $\mathrm { H _ { 2 } }$  data and fits for process calculations.</td><td>4</td></tr><tr><td>M74-W6</td><td>Compare IAST with representative direct mixture GCMC and judge reliability.</td><td>5</td></tr><tr><td>M74-W7</td><td>Evaluate all six feeds at 298 K, 2.5 MPa, void fraction 0.4, velocity 0.006791 m s−1, and length 0.065 m.</td><td>7</td></tr><tr><td>M74-W8</td><td>Analyze an equimolar 4 MPa mixture for both species in all six materials.</td><td>4</td></tr><tr><td>M74-W9</td><td>Synthesize rankings, limitations, and evidence paths.</td><td>3</td></tr></table>

$\mathrm { H _ { 2 } }$ data at 298 K must be generated for the 298 K IAST and process calculations rather than reusing the 77 K validation isotherm. Interpretation should allow tiers or ties, recognize instability caused by very small $\mathrm { H _ { 2 } }$ loadings, and downgrade IAST-derived breakthrough rankings when direct mixture GCMC disagrees. No exact metal order is prescribed.

![](images/e68d18647b9bcc88810660aa04276e5a586687b7bbf172860140e566c41ebfe1.jpg)  
Figure S3: $\mathrm { C O _ { 2 } }$ isotherms at 298 K from Study 3, multi-agent Run 2, for the six M-MOF-74 materials. Black lines denote simulations and colored circles the experimental reference data<sup>40</sup>. Loadings are in mol $\mathrm { k g ^ { - 1 } }$ . The comparison illustrates both successful reproduction of broad adsorption trends and material-dependent deviations. Numerical agreement is not itself a scoring criterion.

## 5.3.4 Olefin/parafin separation in cation-exchanged zeolites

The zeolite workflow comprises calculation of Rosenbluth weights, validation of pure propane and propene isotherms and hexane and hexene isobars, mixture adsorption and selectivity, and adsorption-site analysis across cation identities, topologies, and chain lengths <sup>41</sup>. Table S13 gives the atomic workflow decomposition. For flexible hydrocarbons, full simulation-validity credit requires the final averaged Widom/Rosenbluth factor at the matching temperature to be propagated into the corresponding GCMC input. Completion of the auxiliary calculation alone is not suficient when an incorrect value is used downstream.

Table S13: Atomic workflow criteria for the olefin/parafin zeolite benchmark.
<table><tr><td>ID</td><td>Expected workflow step</td><td>Points</td></tr><tr><td></td><td>ZEO-W1 Audit structures, cation and restart data, force fields, experimental states, units, and pressure coverage.</td><td>3</td></tr><tr><td>ZEO-W2</td><td>Calculate the  $\mathrm { C _ { 3 } \mathrm { - C _ { 6 } } }$  Rosenbluth prerequisites at 298 K and the  $\mathrm { C _ { 6 } }$  values across the isobar temperatures.</td><td>4</td></tr><tr><td>ZEO-W3</td><td>Simulate and compare propane and propene isotherms for LTA5A and NaY at 298 K.</td><td>5</td></tr><tr><td>ZEO-W4</td><td>Simulate and compare hexane and 1-hexene isobars for LTA5A and NaY at the experimental pressures.</td><td>5</td></tr><tr><td>ZEO-W5</td><td>Run equimolar direct-mixture GCMC for all eight frameworks and four chain lengths at 298 K over the pressure range, including 100 kPa.</td><td>9</td></tr><tr><td>ZEO-W6</td><td>Calculate and audit selectivity versus pressure and at 100 kPa, including denominator-sensitive points.</td><td>5</td></tr><tr><td>ZEO-W7</td><td>Generate and compare 298 K, 100 kPa propene occupancy maps for the five FAU materials.</td><td>5</td></tr><tr><td></td><td>ZEO-W8 Synthesize material, pressure, chain-length, cation, validation, and limitation findings in an auditable report.</td><td>4</td></tr><tr><td colspan="2">Total</td><td>40</td></tr></table>

The supplied cation-position restarts and component-order conventions should be retained where applicable, and supercell replication must be appropriate for the selected cutof. Exact material or chain-length rankings are not prescribed. Selectivities based on very small parafin loadings should be treated as denominator-sensitive, and occupancy-map comparisons should distinguish within-framework localization from absolute density. Claims assigning adsorption changes to cation identity or position require direct spatial, energetic, or otherwise controlled evidence. Numerical disagreement with the experimental isotherms or isobars is not itself penalized.

## 5.4 Force-Field Extraction Evaluation

Force-field extraction is evaluated for parameter correctness and RASPA readiness. The combined score is

$$
S _ { \mathrm { F F } } = 0 . 6 0 S _ { \mathrm { p a r a m e t e r s } } + 0 . 4 0 S _ { \mathrm { r e a d i n e s s } } .\tag{S3}
$$

The score is calculated separately for each repeated extraction and then averaged across repetitions for each case. The benchmark-level score is the arithmetic mean of the eight case scores.

Parameter correctness. The extracted parameter set is compared with a reference set derived from the original publications. A parameter is any independent numerical quantity that influences the simulation: Lennard–Jones σ and ϵ are counted separately, as are atomic charges where applicable, molecular bond lengths, and other intramolecular or bonded terms. A parameter is considered correct only when its identifier, interaction form, value, and units exactly match the reference definition. No numerical tolerance is applied. Performance is quantified using an intersection-over-union score,

$$
S _ { \mathrm { p a r a m e t e r s } } = \frac { | P _ { \mathrm { c o r r e c t } } | } { | P _ { \mathrm { r e f e r e n c e } } \cup P _ { \mathrm { p r e d i c t e d } } | } ,\tag{S4}
$$

where $P _ { \mathrm { c o r r e c t } }$ is the set of correctly extracted parameters. Missing, hallucinated, and incorrectly valued parameters reduce the score.

RASPA readiness. This metric measures whether the generated force-field files can be used directly in RASPA without manual correction:

$$
S _ { \mathrm { r e a d i n e s s } } = { \frac { N _ { \mathrm { e x e c u t a b l e } } } { N _ { \mathrm { g e n e r a t e d } } } } .\tag{S5}
$$

A generated file is considered non-executable if it produces a RASPA error, violates formatting or bookkeeping requirements, or requires manual editing before a simulation can run.

Table S14 lists the eight force-field extraction cases included in the benchmark.

Table S14: Scope of the force-field extraction benchmark.
<table><tr><td>Case</td><td>Source</td></tr><tr><td>EPM2 CO2</td><td>Harris and Yung 51</td></tr><tr><td>CO2 in zeolites</td><td>Garcia-Sanchez et al.44</td></tr><tr><td>CO2 in zeolites</td><td>Fang et al.52</td></tr><tr><td> $\mathrm { C O _ { 2 } , N _ { 2 } , O _ { 2 } , }$  and Ar in zeolites</td><td>Vujic et al.53</td></tr><tr><td>TraPPE-zeo</td><td>Bai et al.54</td></tr><tr><td>Hydrocarbons and permanent gases in zeolites</td><td>Findley and Sholl55</td></tr><tr><td>TAMOF-1</td><td>Gooijer et al.38</td></tr><tr><td>UFF and DREIDING for MOFs</td><td>Rappe et al. and Mayo et al.56;57</td></tr></table>

## 5.5 Force-Field Fitting Harness

Force-field optimization was controlled through two calls:

enter fitting loop(strategy, objective,   
stop criteria, max attempts)   
complete fitting phase(decision)

The controller enforced the sequence parameter selection → simulation → analysis → continuation or termination, recorded the attempt number and declared scientific criteria, and preserved each iteration separately. The allowed attempt limit was 1–50. The $\mathrm { F e - M O F  – 7 4 / C _ { 2 } H _ { 2 } }$ campaign used 15 attempts with three candidates per attempt, giving 45 evaluated variants.

Candidate generation followed an adaptive, hypothesis-driven local search. The initial stage varied the explicit $\mathrm { F e - C _ { C 2 H 2 } }$ Lennard–Jones cross interaction over $\epsilon = 1 2 0 { - } 1 2 0 0 \mathrm { ~ K ~ }$ and $\sigma =$ 2.30–3.20 <sup>˚</sup>A. Subsequent stages introduced explicit $\mathrm { O 1 / O 2 / O 3 { - C } _ { C 2 H 2 } }$ terms, first as a grouped interaction and then by oxygen type, over the approximate ranges $\epsilon = 1 0 { - } 5 0$ K and $\sigma =$ 2.80–3.60 <sup>˚</sup>A. Three candidates were selected near the most promising feasible region after each attempt, using the observed sensitivity of adsorption to ϵ and binding geometry to σ.

Candidates were compared by multi-criterion, guardrail-constrained selection rather than a single scalar loss. The criteria comprised Fe–C binding-distance deviations from experimental values of 2.48 and 2.52 <sup>˚</sup>A at 298 K, logarithmic loading errors at 0.055 and 1.696 kPa at 318 K, and the 318 K Widom Henry coeficient. Widom adsorption energy was monitored as a diagnostic and did not determine candidate acceptance or rejection. $\mathrm { F e - M O F - 7 4 / C O _ { 2 } }$ binding geometry and loading at 0.1 kPa served as guardrails, and candidates that lost the open-metal-site $\mathrm { { C O } _ { 2 } }$ pose were rejected. The loop terminated when both Fe–C distances were within 0.10 <sup>˚</sup>A of experiment, both $\mathrm { C _ { 2 } H _ { 2 } }$ loadings were within a factor of two of experiment, and the $\mathrm { C O _ { 2 } }$ guardrails were retained, or when the 15-attempt limit was reached.

## 5.6 Independent Energetic Validation

Experimental adsorption-heat data were unavailable to the fitting agent and were not used as optimization targets. The fit used the $\mathrm { { C _ { 2 } H _ { 2 } } }$ isotherm and binding geometry. Predicted Widom adsorption energy was monitored without guiding candidate selection or comparison with experimental heats. The literature model was also validated against adsorption-heat data <sup>39</sup>. Figure S4 compares the final fitted model with experimental heats after fitting, providing an independent energetic assessment. The fitted model reproduces the broad low-loading energy scale and the decrease toward saturation. The decrease occurs more sharply than in experiment, while the literature force field follows the low-loading measurements more closely but likewise departs from the experimental curve over part of the high-loading range. Including adsorptionheat observables in future fitting campaigns could further constrain interaction models that otherwise reproduce similar adsorption isotherms and binding geometries.

C2H2 adsorption heat in Fe-MOF-74 at 318 K  
![](images/52e0653d9e49dd40f5e3e813500b26d67264b79acfa0857811b8336d1d5ba91b.jpg)  
Figure S4: Post-fitting energetic assessment for $\mathrm { { C _ { 2 } H _ { 2 } } }$ in Fe-MOF-74 at 318 K. The experimental curve is from Bloch et $\mathrm { a l . ^ { 5 8 } }$ . The literature-force-field points were digitized from Ref. <sup>39</sup>. Experimental heats were unavailable to the fitting agent. RaspAssistant results use the final fitted force field. The zero-loading Widom prediction was monitored without an experimental heat reference or use in selection. Error bars show the reported simulation uncertainties and are smaller than the symbols where not visible.

## References

[1] Dral, P. O.; Nawaz, H.; Ullah, A. Science Done on a Machine by a Machine: AI Agents in Computational Chemistry. arXiv preprint arXiv:2608.18508 2026, DOI: 10.48550/arXiv.2608.18508.

[2] Bran, A. M.; Cox, S.; Schilter, O.; Baldassari, C.; White, A. D.; Schwaller, P. Augmenting large language models with chemistry tools. Nature Machine Intelligence 2024, 6, 525–535, DOI: 10.1038/s42256-024-00832-8.

[3] Boiko, D. A.; MacKnight, R.; Kline, B.; Gomes, G. Autonomous chemical research with large language models. Nature 2023, 624, 570–578, DOI: 10.1038/s41586-023-06792-0.

[4] Burger, B.; Mafettone, P. M.; Gusev, V. V.; Aitchison, C. M.; Bai, Y.; Wang, X.; Li, X.; Alston, B. M.; Li, B.; Clowes, R. et al. A mobile robotic chemist. Nature 2020, 583, 237–241.

[5] Abolhasani, M.; Kumacheva, E. The rise of self-driving labs in chemical and materials sciences. Nature Synthesis 2023, 2, 483–492.

[6] Szymanski, N. J.; Rendy, B.; Fei, Y.; Kumar, R. E.; He, T.; Milsted, D.; McDermott, M. J.; Gallant, M.; Cubuk, E. D.; Merchant, A. et al. An autonomous laboratory for the accelerated synthesis of inorganic materials. Nature 2023, 624, 86–91, DOI: 10.1038/s41586-023-06734-w.

[7] Zou, Y.; Cheng, A. H.; Aldossary, A.; Bai, J.; Leong, S. X.; Campos-Gonzalez-Angulo, J. A.; Choi, C.; Ser, C. T.; Tom, G.; Wang, A. et al. El Agente: An autonomous agent for quantum chemistry. Matter 2025, 8, 102263, DOI: 10.1016/j.matt.2025.102263.

[8] Gustin, I.; Calder´on, L. M.; P´erez-S´anchez, J. B.; Gonthier, J. F.; Nakamura, Y.; Panicker, K.; Ramprasad, M.; Zhang, Z.; Zou, Y.; Bernales, V. et al. El Agente Cuantico: Automating quantum simulations. arXiv preprint arXiv:2512.18847 2025, DOI: 10.48550/arXiv.2512.18847.

[9] Choi, C.; Zou, Y.; M¨uller, M.; Hao, H.; Kang, Y.; P´erez-S´anchez, J. B.; Gustin, I.; Xu, H.; Wang, A.; Vakili, M. G. et al. El agente estructural: An artificially intelligent molecular editor. arXiv preprint arXiv:2602.04849 2026,

[10] P´erez-S´anchez, J. B.; Zou, Y.; Campos-Gonzalez-Angulo, J. A.; M¨uller, M.; Gustin, I.; Wang, A.; Hao, H.; Ko, T. W.; Choi, C.; Isbrandt, E. S. et al. El Agente Quntur: A research collaborator agent for quantum chemistry. arXiv preprint arXiv:2602.04850 2026,

[11] Wang, Z.; Huang, H.; Zhao, H.; Xu, C.; Zhu, S.; Janssen, J.; Viswanathan, V. DREAMS: Density functional theory based research engine for agentic materials simulation. arXiv preprint arXiv:2507.14267 2025,

[12] Liu, J.; Zhu, T.; Ye, C.; Fang, Z.; Weng, H.; Wu, Q. VASPilot: MCP-facilitated multi-agent intelligence for autonomous VASP simulations. Chinese Physics B 2025, 34, 117106, DOI: 10.1088/1674-1056/ae0681.

[13] Rothfarb, S.; Davis, M. C.; Matanovic, I.; Li, B.; Holby, E. F.; Kort-Kamp, W. J. Hierarchical Multi-agent Large Language Model Reasoning for Autonomous Heterogeneous Catalyst Discovery. npj Computational Materials 2026, 12, 309, DOI: 10.1038/s41524-026-02139-1.

[14] Chaudhari, A.; Ock, J.; Barati Farimani, A. Modular large language model agents for multi-task computational materials science. Communications Materials 2026, 7, 131, DOI: 10.1038/s43246-025-00994-x.

[15] Liu, G.; Yang, S.; Zhong, Y. Masgent: an AI-assisted materials simulation agent. Digital Discovery 2026, 5, 2151–2171, DOI: 10.1039/D6DD00043F.

[16] Vriza, A.; Kornu, U.; Koneru, A.; Chan, H.; Sankaranarayanan, S. K. R. S. Multi-agentic AI framework for end-to-end atomistic simulations. Digital Discovery 2026, 5, 440–452, DOI: 10.1039/D5DD00435G.

[17] Somasundaram, R.; Habib, A.; Dang, K.; Shivakumar, S.; Hill, R. G.; Wimmer, G.; Mishra, A.; Pachalieva, A.; Lui, A.; Viswanathan, H. et al. An Agentic Orchestration of Atomistic Simulations. arXiv preprint arXiv:2607.22596 2026, DOI: 10.48550/arXiv.2607.22596.

[18] Huang, H.; Li, Y.; Xu, A.; Liu, B.; Wang, C.; Tang, K.; Yang, N.; Liu, S.; Liu, H.; Zhang, P. et al. ALKEMIE Agent: an autonomous platform for computational materials design. arXiv preprint arXiv:2608.15776 2026, DOI: 10.48550/arXiv.2608.15776.

[19] Yang, F.; Evans, J. D. QUASAR: A universal autonomous system for atomistic simulation and a benchmark of its capabilities. Journal of Chemical Information and Modeling 2026, 66, 5911–5918, DOI: 10.1021/acs.jcim.6c00295.

[20] Ghafarollahi, A.; Buehler, M. J. Automating alloy design and discovery with physics-aware multimodal multiagent AI. Proceedings of the National Academy of Sciences 2025, 122, e2414074122, DOI: 10.1073/pnas.2414074122.

[21] Shi, Z.; Xin, C.; Huo, T.; Jiang, Y.; Wu, B.; Chen, X.; Qin, W.; Ma, X.; Huang, G.; Wang, Z.; Jing, X. A fine-tuned large language model based molecular dynamics agent for code generation to obtain material thermodynamic parameters. Scientific Reports 2025, 15, 10295, DOI: 10.1038/s41598-025-92337-6.

[22] Pham, T. D.; Tanikanti, A.; Ke¸celi, M. ChemGraph as an agentic framework for computational chemistry workflows. Communications Chemistry 2026, 9, 33, DOI: 10.1038/s42004-025-01776-9.

[23] Soleymanibrojeni, M.; Aydin, R.; Guedes-Sobrinho, D.; Dias, A. C.; Piotrowski, M. J. et al. GENIUS: an agentic AI framework for autonomous design and execution of simulation protocols. Communications Materials 2026, 7, 115, DOI: 10.1038/s43246-026-01167-0.

[24] Holbrook, E.; Verduzco, J. C.; Strachan, A. Evaluating LLM-generated code for domainspecific languages: Molecular dynamics with LAMMPS. Computational Materials Science 2026, 272, 114839, DOI: 10.1016/j.commatsci.2026.114839.

[25] Yu, J.; Jiang, Z.; He, D. AI agents for MOFs and COFs discovery. AI Agent 2026, 2, 18, DOI: 10.20517/aiagent.2026.33.

[26] Lee, J.; Bae, T.; Kim, J. SimMOF: AI agent for Automated MOF Simulations. arXiv preprint arXiv:2603.29152 2026, DOI: 10.48550/arXiv.2603.29152.

[27] Willems, T. F.; Rycroft, C. H.; Kazi, M.; Meza, J. C.; Haranczyk, M. Algorithms and tools for high-throughput geometry-based analysis of crystalline porous materials. Microporous and Mesoporous Materials 2012, 149, 134–141.

[28] Dubbeldam, D.; Calero, S.; Ellis, D. E.; Snurr, R. Q. RASPA: molecular simulation software for adsorption and difusion in flexible nanoporous materials. Molecular Simulation 2016, 42, 81–101.

[29] Kresse, G.; Hafner, J. Ab initio molecular dynamics for liquid metals. Physical review B 1993, 47, 558–561, DOI: 10.1103/PhysRevB.47.558.

[30] Thompson, A. P.; Aktulga, H. M.; Berger, R.; Bolintineanu, D. S.; Brown, W. M.; Crozier, P. S.; In’t Veld, P. J.; Kohlmeyer, A.; Moore, S. G.; Nguyen, T. D. et al. LAMMPSa flexible simulation tool for particle-based materials modeling at the atomic, meso, and continuum scales. Computer physics communications 2022, 271, 108171.

[31] Petkovi´c, M.; Menkovski, V.; Calero, S. Towards Fully Automated Molecular Simulations: Multi-Agent Framework for Simulation Setup and Force Field Extraction. arXiv preprint arXiv:2509.10210 2025,

[32] Li, Y.; Zhao, J.; Cao, X.; Li, S. A large language model-based multi-agent methodology for intelligent materials screening: A case study on MOFs for CO2 capture. Separation and Purification Technology 2026, 394, 137462, DOI: 10.1016/j.seppur.2026.137462.

[33] Zhang, J.; Liang, B.; Li, T.; Li, Y. et al. AI agent-driven discovery of metal-organic frameworks for radon capture: Integrating machine learning and molecular simulation. Chemical Engineering Science 2026, 335, 124406, DOI: 10.1016/j.ces.2026.124406.

[34] Wang, Y.; Yin, Z.-Y.; Tang, Z.; Chu, X. Agent-MD: Selective LLM Intervention with Event-Driven Escalation for Stateful GCMC–MD Campaigns. arXiv preprint arXiv:2608.07637 2026, DOI: 10.48550/arXiv.2608.07637.

[35] Ammar, W.; Groeneveld, D.; Bhagavatula, C.; Beltagy, I.; Crawford, M.; Downey, D.; Dunkelberger, J.; Elgohary, A.; Feldman, S.; Ha, V. et al. Construction of the literature graph in semantic scholar. Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 3 (Industry Papers). 2018; pp 84–91, DOI: 10.18653/v1/N18-3011.

[36] Artifex Software, Inc. PyMuPDF4LLM. Software available at https://github.com/ pymupdf/pymupdf4llm, 2024; Accessed September 3, 2026.

[37] Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan, K. R.; Cao, Y. ReAct: Synergizing Reasoning and Acting in Language Models. The Eleventh International Conference on Learning Representations. 2023.

[38] Gooijer, S.; Capelo-Avil´es, S.; Sharma, S.; Giancola, S.; Gal´an-Mascaros, J.; Vlugt, T.; Dubbeldam, D.; Vicent-Luna, J.; Calero, S. TAMOF-1 for capture and separation of the main flue gas components. Journal of Materials Chemistry A 2025, 13, 16879–16892, DOI: 10.1039/D5TA01362C.

[39] Luna-Triguero, A.; Vicent-Luna, J.; Madero-Castro, R.; G´omez-Alvarez, P.; Calero, S.<sup>´</sup> Acetylene storage and separation using metal–organic frameworks with open metal sites. ACS applied materials & interfaces 2019, 11, 31499–31507, DOI: 10.1021/acsami.9b09010.

[40] Wasik, D. O.; Vicent-Luna, J. M.; Luna-Triguero, A.; Dubbeldam, D.; Vlugt, T. J.; Calero, S. The impact of metal centers in the M-MOF-74 series on carbon dioxide and hydrogen separation. Separation and Purification Technology 2024, 339, 126539, DOI: 10.1016/j.seppur.2024.126539.

[41] Luna-Triguero, A.; S lawek, A.; S´anchez-de Armas, R.; Guti´errez-Sevillano, J. J.; Ania, C. O.; Parra, J. B.; Vicent-Luna, J. M.; Calero, S. π-Complexation for olefin/parafin separation using aluminosilicates. Chemical Engineering Journal 2020, 380, 122482, DOI: 10.1016/j.cej.2019.122482.

[42] Baerlocher, C.; McCusker, L. B.; Olson, D. H. Atlas of zeolite framework types, 6th ed.; Elsevier: Amsterdam, 2007.

[43] Baerlocher, C.; Brouwer, D.; Marler, B.; McCusker, L. B. Database of Zeolite Structures. Structure Commission of the International Zeolite Association, https://www. iza-structure.org/databases/, Accessed 7 October 2026.

[44] Garcia-Sanchez, A.; Ania, C. O.; Parra, J. B.; Dubbeldam, D.; Vlugt, T. J.; Krishna, R.; Calero, S. Transferable force field for carbon dioxide adsorption in zeolites. The Journal of Physical Chemistry C 2009, 113, 8814–8820.

[45] Myers, A. L.; Prausnitz, J. M. Thermodynamics of mixed-gas adsorption. AIChE Journal 1965, 11, 121–127, DOI: 10.1002/aic.690110125.

[46] Walton, K. S.; Sholl, D. S. Predicting multicomponent adsorption: 50 years of the ideal adsorbed solution theory. AIChE Journal 2015, 61, 2757–2762.

[47] Krishna, R.; van Baten, J. M. How reliable is the ideal adsorbed solution theory for the estimation of mixture separation selectivities in microporous crystalline adsorbents? ACS omega 2021, 6, 15499–15513.

[48] Sharma, S.; Balestra, S. R.; Baur, R.; Agarwal, U.; Zuidema, E.; Rigutto, M. S.; Calero, S.; Vlugt, T. J.; Dubbeldam, D. RUPTURA: simulation code for breakthrough, ideal adsorption solution theory computations, and fitting of isotherm models. Molecular Simulation 2023, 49, 893–953.

[49] Hassan, M.; Yoon, S.; Chen, Y.; Kim, P.; Yun, H.; Kwon, H. T.; Bae, Y.-S.; Yoo, C.-Y.; Koh, D.-Y.; Hong, C.-S. et al. AIM: A user-friendly GUI workflow program for isotherm fitting, mixture prediction, isosteric heat of adsorption estimation, and breakthrough simulation. Computer Physics Communications 2026, 319, 109944, DOI: 10.1016/j.cpc.2025.109944.

[50] Simon, C. M.; Smit, B.; Haranczyk, M. pyIAST: Ideal adsorbed solution theory (IAST) Python package. Computer Physics Communications 2016, 200, 364–380.

[51] Harris, J. G.; Yung, K. H. Carbon dioxide’s liquid-vapor coexistence curve and critical properties as predicted by a simple molecular model. The Journal of Physical Chemistry 1995, 99, 12021–12024.

[52] Fang, H.; Kamakoti, P.; Ravikovitch, P. I.; Aronson, M.; Paur, C.; Sholl, D. S. First principles derived, transferable force fields for CO 2 adsorption in Na-exchanged cationic zeolites. Physical Chemistry Chemical Physics 2013, 15, 12882–12894.

[53] Vuji´c, B.; Lyubartsev, A. P. Transferable force-field for modelling of CO2, N2, O2 and Ar in all silica and Na+ exchanged zeolites. Modelling and simulation in Materials science and Engineering 2016, 24, 045002.

[54] Bai, P.; Tsapatsis, M.; Siepmann, J. I. TraPPE-zeo: Transferable potentials for phase equilibria force field for all-silica zeolites. The Journal of Physical Chemistry C 2013, 117, 24375–24387.

[55] Findley, J. M.; Boulfelfel, S. E.; Fang, H.; Muraro, G.; Ravikovitch, P. I.; Sholl, D. S. A transferable force field for predicting adsorption and difusion of hydrocarbons and small molecules in silica zeolites with coupled-cluster accuracy. The Journal of Physical Chemistry C 2021, 125, 8418–8429.

[56] Rapp´e, A. K.; Casewit, C. J.; Colwell, K. S.; Goddard III, W. A.; Skif, W. M. UFF, a full periodic table force field for molecular mechanics and molecular dynamics simulations. Journal of the American chemical society 1992, 114, 10024–10035.

[57] Mayo, S. L.; Olafson, B. D.; Goddard, W. A. DREIDING: a generic force field for molecular simulations. Journal of Physical chemistry 1990, 94, 8897–8909.

[58] Bloch, E. D.; Queen, W. L.; Krishna, R.; Zadrozny, J. M.; Brown, C. M.; Long, J. R. Hydrocarbon separations in a metal-organic framework with open iron(II) coordination sites. Science 2012, 335, 1606–1610, DOI: 10.1126/science.1217544.