# Representation Bias, Correction Transfer, and Resolution Sensitivity in Three-Dimensional Mitochondrial Morphometry

Farouk Ganiyu Adewumi and Timothy Oladunni

Abstract— Quantitative imaging pipelines can produce precise but systematically different measurements of the same object. We present an empirical reliability assessment of three-dimensional mitochondrial morphometry that connects representation bias, a controlled processing intervention, correction transfer, and resolution sensitivity. Using 2,720 development objects from the 3D Mitochondria Shape Library for Optical Microscopy, we find that occupancy-derived volumes exceed reference mesh volumes by 3.665% on average despite an intraclass correlation coefficient of 0.994. Boundary analysis identifies an outward label displacement of 0.00304 normalized units. In a controlled label-pipeline reimplementation, removing the depth offset reduces volume error in all 55 analyzed objects by a mean of 1.57 percentage points, approximately 45% of mean reproduced inflation; the source of the remainder is not isolated. A frozen regression using occupancy-derived features reduces median absolute percentage error from 3.481% to 0.664% in 2,728 previously unused objects from the same resource. However, its calibrated error bound covers only 92.1% overall and 49.2% in a low-occupancy subgroup, demonstrating that accuracy and uncertainty transfer must be evaluated separately. In 550 rat-cortex objects from the MitoEM resource, coarsening in-plane spacing from 8 to 24 nanometers changes median surface area by minus 10.60% and sphericity by plus 11.76%, despite a rank correlation of 0.994. These results provide quantitative checks for distinguishing processing-induced descriptor changes from candidate biological differences, without establishing biological invariance or cross-source correction transfer.

Index Terms— Conformal calibration, electron microscopy, measurement reliability, mitochondrial morphometry, occupancy representations, representation bias, resolution sensitivity.

## I. INTRODUCTION

ITOCHONDRIAL morphology is a regulated phenocellular environment [1]. Its relationship with bioenergetics is bidirectional: structural perturbations can accompany changes in respiration, and metabolic interventions can reshape the mitochondrial network [2]. A morphological difference therefore has to be interpreted at two levels. An apparent change in volume, surface area, or sphericity can reflect altered organelle structure, but it can also arise from segmentation, meshing, sampling, or resolution.

Studies that pair structure with function make the biological question more tractable: electron microscopy (EM) morphology with cytochrome c oxidase histochemistry in the same tissue [3], live-cell network imaging with biophysical modeling in pancreatic beta cells [4], single-organelle morphology with fluorescent functional probes [5], and shape phenotyping with metabolic measurements [6]. Functional readouts carry assay-dependent uncertainty [7]; the structural side of such studies needs the same scrutiny.

For computational imaging, the practical problem is deciding whether a descriptor remains comparable after a change in representation or processing: a high correlation does not establish object-level agreement, and a low average prediction error does not establish that an uncertainty bound remains reliable. We address this problem through paired measurements of the same annotated objects, connecting a representation audit to a controlled intervention in a reimplemented labeling pipeline and then evaluating a frozen correction and its error bound separately. A complementary experiment tests descriptor sensitivity to regridding in EM annotations. The reference meshes are computational comparators, not independent measurements of biological truth.

## A. Contributions and Scope

The contribution is an empirical characterization of linked measurement failure modes, with reproducible evidence and practical checks for quantitative imaging pipelines. The four contributions are:

1) Systematic representation bias. In 2,720 development objects, occupancy-derived volume has a mean bias of +3.665% despite an intraclass correlation coefficient of 0.994; boundary and repeatability analyses separate a displaced label surface from sampling variability.

2) A controlled test of a label-processing parameter. In 55 non-pilot objects, removing the depth offset in a pipeline reimplementation reduces volume error in every object, by 1.57 percentage points on average, leaving the residual bias unresolved.

3) Separate evaluation of correction and uncertainty transfer. A frozen, mesh-free correction reduces median absolute percentage error from 3.481% to 0.664% in

2,728 new objects, but its error-bound coverage is only   
92.1% overall and 49.2% in a low-occupancy subgroup.

4) Absolute descriptor sensitivity despite stable ranks. Regridding 550 rat-cortex objects to 24 nm in-plane spacing changes median surface area by −10.60% and sphericity by +11.76% while preserving sphericity ranks $( \rho = 0 . 9 9 4 )$

The novelty lies in the linked experimental evidence, not a new regression algorithm or agreement statistic. Correction transfer is evaluated within one resource, and no linked functional measurements are available to test biological invariance.

## II. RELATED WORK

## A. Automated phenotyping tools

MiNA measures mitochondrial network skeletons [8]. Mito-Graph reconstructs three-dimensional (3D) surfaces and estimates volume [9]. Nellie extends automated organelle analysis to segmentation, tracking, and hierarchical features in 2D and 3D live-cell microscopy [10].

MitoClass assigns network morphologies to fragmented, intermediate, or elongated categories with a convolutional neural network [11]. MitoTex provides texture-based characterization [12]. These tools produce descriptors from a chosen representation; the agreement of such descriptors across representations of the same object is the question studied here.

## B. Shape representations and public resources

The 3D Mitochondria Shape Library for Optical Microscopy (3DMSL) provides meshes, point clouds, implicit occupancy data, and simulated optical microscopy images for more than 27,000 mitochondrial shapes [13]. The upstream focused ion beam scanning electron microscopy (FIB-SEM) resource is archived as EMPIAR-10791 [14]. Its originating study investigated liver subcellular architecture and metabolic homeostasis [15]. The 3DMSL repository identifies obese (ob/ob) mouse liver as the source [16]; its occupancy labels follow the Occupancy Networks preprocessing [17], in which query points are labeled against a watertight surface produced by depth-map fusion [18]. MitoEM provides large-scale instance-labeled EM volumes [19], and the subsequent challenge analysis documents the importance of annotation quality and evaluation design [20]. Derived representations of one annotation allow controlled processing studies in which object identity is fixed.

## C. The measurement gap

Table I places this study relative to representative approaches; it compares scope and evidence, not benchmark performance. The structure–function studies motivate paired measurements but do not establish a universal mapping from shape to function. The question here is whether a descriptor remains comparable after representation conversion, and whether a correction and its uncertainty bound transfer together.

## III. MEASUREMENT FRAMEWORK

Let $S _ { i }$ denote the underlying structure of object i and let $T _ { r }$ denote processing into representation r. A descriptor is observed as $m _ { i r } = g _ { r } ( T _ { r } ( S _ { i } ) )$ ), so $m _ { i r }$ and $m _ { i s }$ may differ even when the object is unchanged. Our experiments hold object identity fixed and vary representation or processing. The raw 3DMSL mesh serves as the computational reference; it is not treated as an error-free biological measurement.

For mesh volume $V _ { m }$ and occupancy volume $V _ { o } .$ the signed percentage error is

$$
e = 1 0 0 \left( \frac { V _ { o } } { V _ { m } } - 1 \right) .\tag{1}
$$

We report mean signed error, median absolute percentage error (MdAPE), and root mean square percentage error (RMSE). Spearman rank correlation assesses ordering. Agreement is summarized with the two-way, absolute-agreement, singlemeasure intraclass correlation coefficient, ICC(A,1) [21], interpreted following [22]. ICC depends on between-object variability relative to disagreement; a wide size range can yield a high ICC despite practically important object-level errors. We therefore also report Bland–Altman bias and limits of agreement [23] and the percentage error, $1 0 0 \times 1 . 9 6 \mathrm { S D } _ { \Delta } / \bar { M }$ where $\begin{array} { r } { \bar { M } = n ^ { - 1 } \sum _ { i } ( V _ { m i } + V _ { o i } ) / 2 \ [ 2 4 ] } \end{array}$

For an approximately uniform outward boundary displacement t that is small relative to local geometric dimensions, the first-order volume change and the implied normalized displacement are

$$
\Delta V \approx A _ { m } t , \qquad t _ { \mathrm { n o r m } } \approx { \frac { V _ { o } - V _ { m } } { A _ { m } s } } ,\tag{2}
$$

where $A _ { m }$ is mesh area and s is the stored normalization scale. For constant normalized displacement, the corresponding prediction is

$$
e \approx 1 0 0 t _ { \mathrm { n o r m } } { \frac { A _ { m } s } { V _ { m } } } = 1 0 0 t _ { \mathrm { n o r m } } { \frac { A _ { m } } { V _ { m } ^ { 2 / 3 } } } { \frac { s } { V _ { m } ^ { 1 / 3 } } } .\tag{3}
$$

Thus, dimensionless shape complexity $A _ { m } / V _ { m } ^ { 2 / 3 }$ alone omits a potentially variable scale-to-size factor. We report its association with error descriptively and additionally test $A _ { m } s / V _ { m }$ as a mechanistic predictor.

## IV. DATASETS AND EXPERIMENTAL DESIGN

## A. Dataset provenance and units of analysis

We analyzed two public resources. From 3DMSL [25] we used the development archive 10 24553 27272.zip (IDs 24553–27272; 2,720 objects) and the frozen validation archive 1 1 2729.zip (IDs 1–2728; 2,728 objects, none excluded, no ID overlap). From MitoEM-R [19] we used the rat-cortex validation slices 400–499 at $3 0 \times 8 \times 8$ nm (1,203 instance IDs, 550 eligible complete objects); training slices 0–29 served only as a technical pilot. The two 3DMSL ZIP files are batches within one resource, not independent biological cohorts. Their IDs are disjoint, and no raw-mesh file is duplicated across batches, but source-animal independence is not established. EMPIAR-10791 is the upstream source of 3DMSL, not a third analyzed dataset.

TABLE I  
REPRESENTATIVE APPROACHES AND THE MEASUREMENT QUESTION ADDRESSED BY THIS STUDY.
<table><tr><td>Study or resource</td><td>Approach and endpoint</td><td>Relationship to this study</td></tr><tr><td>Faitg et al. [3], 2020</td><td>EM morphology with cytochrome c oxidase labeling; structure-function assessment</td><td>Motivates paired biological measurements</td></tr><tr><td>MitoEM [19], 2020; [20], 2023</td><td>Large-scale 3D EM instance segmentation and its evaluation</td><td>Supplies real-EM labels; we measure descriptor sensitivity</td></tr><tr><td>Tseng et al. [4], 2024</td><td>Live-cell network imaging and biophysical modeling; metabolic coupling</td><td>Biological coupling; we characterize measurement prerequisites</td></tr><tr><td>Nellie [10], 2025</td><td>Automated 2D/3D organelle segmentation; morphology and Produces descriptors whose sensitivity we quantify motility features</td><td></td></tr><tr><td>Ding et al. [5], 2025</td><td>Fluorescent probes with AI organelle analysis; joint functional biomarkers</td><td>Multimodal evidence absent from our datasets</td></tr><tr><td>3DMSL [13], 2026</td><td>Paired 3D shapes and synthetic microscopy resource</td><td>Enables comparison of representations of one object</td></tr><tr><td>Present study</td><td>Bias audit, boundary mechanism, frozen correction, subgroup and resolution tests</td><td>Quantifies representation sensitivity; no functional inference</td></tr></table>

E4 uses both 3DMSL batches and E8 reuses the E5 objects, so the subset sizes in Table II should not be summed. An initial 40-object MitoEM analysis and an initial 400-object imaging analysis were superseded by the 550-object and 300-object analyses and are not counted as additional evidence.

## B. Experiment-to-claim mapping

The experiment labels in Table II organize completed analyses; they do not imply prospective registration. E8 and the supplementary analyses were added after the primary results (Supplementary Material, Section S4). Boundary and distribution-shift diagnostics are exploratory analyses that explain or qualify the primary results.

## C. Structural processing and quality control

The 3DMSL audit verifies the mesh, point-cloud, and occupancy files for each object. Point-cloud and occupancy coordinates are transformed with $\mathbf { x } _ { \mathrm { r a w } } = s \mathbf { x } _ { \mathrm { n o r m } } + \mathbf { l o c } .$ , and bitpacked occupancy labels are unpacked before processing. Each archive stores 100,000 uniformly distributed query points per object in a normalized box of ±0.55 (padding 0.1), with labels computed against the watertight fused mesh [16]. The data article reports 10,000 points [13]; the released files and build script contain 100,000. The implemented occupancy volume estimate is the fraction of inside-labeled points multiplied by the product of the observed query-coordinate spans in raw units. This empirical-box convention (not the nominal width 1.1s) is retained for consistency with the saved correction. Mesh volume is the absolute signed volume; mesh area and axis-aligned extents are recorded separately.

Load failures are hard exclusions. Four conditions are advisory flags: nonwatertight meshes; occupancy fractions below 0.02 (the low-occupancy flag); incomplete rendered views; and a raw, untrimmed point-cloud extent that differs from the mesh extent by more than 10% along any single axis. The flags were defined before any correction model was fitted. Descriptor comparisons of point-cloud extent use the 1st-to-99th-percentile trimmed extent, which suppresses isolated outliers but also removes genuine extremal points. Both 3DMSL batches contained only watertight meshes.

## D. Repeatability estimation

A simple random sample of 400 numerically ordered development IDs is selected without replacement with NumPy seed 0. The same generator subsequently permutes each object’s stored queries into ten disjoint folds. The standard deviation of the ten fold-level volume estimates, divided by ${ \sqrt { 1 0 } } ,$ , estimates the standard deviation (SD) of the full-sample estimate under the independent-query model. The pairwise repeatability coefficient is $1 . 9 6 \sqrt { 2 } \approx 2 . 7 7$ times this SD, expressed relative to the estimate. It describes differences between two independent numerical estimates and excludes repeat scanning, segmentation, and biological variability. A convergence analysis evaluates subsets from 1,000 to 100,000 queries.

## E. Correction and calibration

The internal split is stratified by mesh-volume decile (seed 2026, numerically ordered IDs; saved split identifiers retained). Training data alone determine the regression coefficients and feature standardization, calibration data the residual bounds, and test data the internal performance summaries. Because earlier whole-batch exploration may have informed feature selection, the internal test is validation after exploration, not prospective confirmation.

Let $B _ { q }$ be the product of coordinate spans over all query points in raw units, $p$ the fraction labeled inside, and $B _ { \mathrm { i n } }$ the bounding-box volume of only the inside-labeled points. The volume estimate is $V _ { o } = p B _ { q }$ . The five predictors are $V _ { o } / B _ { \mathrm { i n } } , \ p ,$ log $V _ { o } ,$ , log $B _ { \mathrm { i n } }$ , and the ratio of maximum to minimum coordinate span among the inside-labeled points. They describe bounding-box fill, sampling occupancy, scale, and elongation without requiring the reference mesh. Ordinary least squares, a transparent baseline, predicts the signed percentage error, predictions are clipped to [−50, 100]%, and the corrected volume is

$$
V _ { \mathrm { c o r r e c t e d } } = \frac { V _ { o } } { 1 + \widehat { e } / 1 0 0 } .\tag{4}
$$

The global correction substitutes the mean training-set percentage error for ${ \widehat { e } } ,$ and the uncorrected baseline retains $V _ { o }$ . Mesh

TABLE II  
EXPERIMENT-TO-CLAIM MAP.
<table><tr><td>ID</td><td>Experiment</td><td>Sample and comparison</td><td>Supported claim</td></tr><tr><td>E1</td><td>Integrity and representation agreement</td><td>2,720 development objects; mesh, occupancy, point cloud</td><td>Systematic disagreement despite high agreement</td></tr><tr><td>E2</td><td>Occupancy sampling repeatability</td><td>400 objects; 10 disjoint query folds; convergence</td><td>Numerical precision is separate from offset</td></tr><tr><td>E3</td><td>Internal correction and calibration</td><td>1,630 train, 540 calibration, 550 test objects</td><td>Internal performance after exploration</td></tr><tr><td>E4</td><td>Frozen transfer and shift diagnostics</td><td>2,728 new objects; original model and bounds</td><td>Within-dataset transfer; coverage failures</td></tr><tr><td>E5</td><td>Label boundary characterization</td><td>60 objects; distance-label logistic fits</td><td>Outward label-boundary offset</td></tr><tr><td>E6</td><td>Native MitoEM representation comparison</td><td>550 objects from one rat-cortex slab</td><td>Mask-to-mesh and sampling sensitivity on EM</td></tr><tr><td>E7</td><td>In-plane resolution sensitivity</td><td>Same 550 objects; native, 16, 24 nm grids</td><td>Absolute shifts with rank stability</td></tr><tr><td>E8</td><td>Controlled label-pipeline reimplementation</td><td>55 E5 objects; depth offsets 0, 0.75, 1.5 voxels</td><td>Fidelity; effect of removing the offset</td></tr></table>

values are used as development targets and for evaluation but are not predictors at deployment.

For each method, calibration uses the absolute correctedvolume percentage error

$$
a _ { i } = \left| 1 0 0 \left( \frac { \widehat { V } _ { i } } { V _ { m i } } - 1 \right) \right| .\tag{5}
$$

For the occupancy-only correction, the signed residual is $( e _ { i } -$ $\widehat { e } _ { i } ) / ( 1 + \widehat { e } _ { i } / 1 0 0 )$ , rather than $e _ { i } - { \widehat { e } } _ { i }$ . The saved bound $q$ is the $\lceil ( n _ { \mathrm { c a l } } + 1 ) 0 . 9 5$ ⌉th ordered score: the 514th of 540 calibration scores here. For $q < 1 0 0 \%$ , inversion gives a reference-volume prediction interval

$$
\left[ \frac { \widehat { V } } { 1 + q / 1 0 0 } , \frac { \widehat { V } } { 1 - q / 1 0 0 } \right] .\tag{6}
$$

The split-conformal construction [26] provides marginal coverage under the required exchangeability and fitting conditions, not conditional coverage for each subgroup [27]. Batch shift can violate exchangeability [28]. We report empirical coverage at the saved bound.

Frozen evaluation preserves the development feature means and scales, coefficients, clipping rule, global offset, and calibrated bounds. Model artifacts were hashed before evaluation and rechecked afterward, and no second-batch outcome was used to update the correction. Subsequent second-batch regressions are exploratory descriptions of error associations, not refits. Algorithm 1 summarizes the correction, calibration, and frozen-transfer procedure.

## F. Boundary and subgroup diagnostics

The boundary analysis uses 60 development objects sampled without replacement with seed 790; their recorded IDs are reused in the exact-distance refit. That refit uses querysampling seed $j$ for zero-based object index $j .$ It analyzes the occupancy queries near the raw mesh in normalized coordinates. Points with an approximate distance below 0.02 (computed from the 32 nearest triangle centroids) are candidates, and up to 3,000 per object are retained. Their distances are then computed exactly, as the point-to-triangle distance over every triangle that could lie closer than the approximate value, with the inside/outside sign given by the generalized winding number [29]. The labels are fitted with the logistic model

Algorithm 1 Mesh-free occupancy volume correction with a   
split-conformal error bound (development, then frozen trans  
fer).   
Require: Development objects D with query points x<sub>norm</sub>, labels,   
scale s, offset loc, and reference mesh volume $V _ { m } ;$ new batch   
$\mathcal { D } ^ { \prime }$ without meshes; coverage level $1 - \alpha = 0 . 9 5$   
1: function FEATURES(object)   
2: x<sub>raw</sub> ← s x<sub>norm</sub> + loc ▷ unpacked labels   
3: $B _ { q } \gets 1$ product of query-coordinate spans; $p $ inside fraction   
4: $V _ { o } ^ { ' }  \hat { p } B _ { q } ; B _ { \mathrm { i n } } $ bounding-box volume of inside points   
5: return $\begin{array} { r } { \mathbf { f } \doteq [ V _ { o } / B _ { \mathrm { i n } } , p , } \end{array}$ log ${ \breve { V } } _ { o } ,$ log $B _ { \mathrm { i n } } ,$ span ratio], $V _ { o }$   
6: for each $i \in \mathcal { D }$ do   
7: $( \mathbf { f } _ { i } , V _ { o i } ) \gets \mathrm { F E A T U R E S } ( i ) ; e _ { i } \gets 1 0 0 ( V _ { o i } / V _ { m i } - 1 )$   
8: Split $\mathcal { D }$ by mesh-volume decile (seed 2026) into train, calibration,   
test   
9: Standardize f and fit OLS eb(f) on train only   
10: for each i in calibration do   
11: $\widehat { e } _ { i } \gets \mathrm { c l i p } ( \widehat { e } ( \mathbf { f } _ { i } ) , - 5 0 , 1 0 0 ) ; \widehat { V } _ { i } \gets V _ { o i } / ( 1 + \widehat { e } _ { i } / 1 0 0 )$   
12: $a _ { i }  | 1 0 0 ( \widehat { V } _ { i } / V _ { m i } - 1 ) |$ ▷ conformity score   
13: $q $ the $\lceil ( n _ { \mathrm { c a l } } + 1 ) ( 1 - \alpha )$ ⌉th smallest $a _ { i } \qquad \triangleright \ q = 2 . 6 6 1 \%$   
14: Freeze and hash scaler, coefficients, clipping rule, and q   
15: for each $j \in \mathcal { D } ^ { \prime }$ do $\triangleright$ no refitting, no mesh   
16: $( \mathbf { f } _ { j } , \bar { V _ { o j } } ) \gets \mathrm { F E A T U R E S } ( j ) ; \widehat { e } _ { j } \gets \mathrm { c l i p } ( \widehat { e } ( \mathbf { f } _ { j } ) , - 5 0 , \mathsf { \bar { 1 } } 0 0 )$   
17: $\widehat { V _ { j } } \gets \mathsf { \bar { V } } _ { o j } / ( 1 + \widehat { e } _ { j } / 1 0 0 )$   
18: Report $[ \widehat { V } _ { j } / ( 1 + q / 1 0 0 ) , ~ \widehat { V } _ { j } / ( 1 - q / 1 0 0 ) ]$   
19: Evaluate MdAPE, RMSE, and coverage overall and for $p < 0 . 0 2$   
against $V _ { m }$

$$
P ( { \mathrm { i n s i d e } } \mid d ) = \left[ 1 + \exp \left( { \frac { d - t _ { 5 0 } } { w } } \right) \right] ^ { - 1 } ,\tag{7}
$$

where the midpoint $t _ { 5 0 }$ estimates the label-transition displacement and $w$ its width. The model is fitted by maximum likelihood from five starting points. A fit is accepted only if the optimizer reports convergence, the scaled projected gradient is below $1 0 ^ { - 4 }$ , neither parameter lies on a bound, the starts that reach the optimum agree on $t _ { 5 0 }$ to within $1 0 ^ { - 5 }$ , each label class contains at least 20 points, and the classes overlap.

In the documented label pipeline [16], [30], meshes are scaled to largest extent 0.9, and depth maps from 100 views, each shifted 1.5 voxels toward the camera, are fused at $2 5 6 ^ { 3 }$ resolution. E8 tests this pipeline directly.

Second-batch diagnostics compare shape distributions, implied offsets, low-occupancy prevalence, and coverage by shape-complexity decile. A development regression of error on shape complexity predicts the change in mean raw error expected from the change in mean shape complexity; this is an association-based decomposition. The worst-error analysis is post hoc. A zero-intercept model $e \mathrm { ~ = ~ } \beta A _ { m } s / V _ { m }$ , fitted on development objects and applied unchanged to the second batch, serves as a mechanistic diagnostic; it uses mesh information and is not a deployment correction.

## G. Controlled label-pipeline reimplementation

E8 reimplemented the 3DMSL label step in NumPy from the repository code [16], [18], because the original libraries require compiled graphics components. Each mesh is rendered from the same 100 views $( 6 4 0 ~ \times ~ 6 4 0$ depth maps); each depth map is shifted toward the camera by the depth offset and eroded with a $3 \times 3$ minimum filter; and the maps are fused into a truncated signed-distance field at $2 5 6 ^ { 3 }$ with a truncation of 10 voxels. The stored query points are labeled inside where the interpolated field is negative. For 55 of the 60 E5 objects, labels were generated at depth offsets of 0, 0.75, and 1.5 voxels (1.5 is the dataset default) with nothing else changed (paired within object). The primary endpoint is the change in occupancy-volume error from offset 0 to 1.5, with a 5,000-sample paired object bootstrap interval. Fidelity is measured by agreement with the released labels for sampled query points within 0.01 normalized units of the mesh and by the correlation of recreated and released per-object errors. Midpoint changes use objects whose fits pass the E5 acceptance checks at both offsets. For ten objects, a hybrid labeling replaced interpolated labels with winding-number containment in the extracted mesh for query points whose interpolated field magnitude was below $2 / 2 5 6$ , retaining interpolated labels elsewhere, a partial containment check. E8 measures fidelity to the released labels, not identity with the historical build; Supplementary Section S4 records the pilot exclusion and coordinate-convention choice.

## H. Real EM processing

MitoEM-R analysis uses labels on slices 400–499 at 30 × $8 \times 8$ nm spacing in z, y, and x. Eligible objects contain at least 2,000 voxels, span at least three z planes, do not touch slab or image boundaries, and have a padded region of interest (ROI) of at most 2,000,000 voxels. Of 1,203 instance IDs, sequential filtering removed 586 for slab-boundary contact, 61 for image-boundary contact, none for size or z span, and six for ROI size, leaving 550 (45.7%). All 550 were meshed by marching cubes [31] at level 0.5. Mesh volume and area are compared with voxel-count volume and exposed-voxelface area. Surface point sampling uses 10,000 points and five repetitions per object. All representations derive from the same annotation, so this is a processing-portability experiment.

In E7, masks are downsampled in y and x only, by block occupancy of at least one half with ties assigned to foreground. Spacings become $3 0 \times 1 6 \times 1 6$ and $3 0 \times 2 4 \times 2 4$ nm; z is unchanged. The 24 nm in-plane spacing matches the reported 3DMSL mesh-generation spacing; the complete voxel geometry remains $3 0 \times 2 4 \times 2 4$ nm [16]. Ties can occur only for the $2 \times 2$ blocks of the 16 nm grid; the $3 \times 3$ blocks of the 24 nm grid cannot tie. Remeshed volume, surface area, and sphericity, $\pi ^ { 1 / 3 } ( 6 V ) ^ { 2 / 3 } / A$ , are compared with the native mesh. The experiment measures sensitivity to this regridding and meshing procedure, not the full effect of lower-resolution acquisition. Supplementary analysis S3 repeats it over all block phases and tie rules.

## I. Statistical reporting

We use percentile bootstrap intervals [32]. Frozen evaluation uses 1,000 object-level resamples (seed 790); internal validation uses 2,000 resamples (seed 2026 with method-specific offsets). These intervals describe object-level variability and do not capture between-animal uncertainty. The primary uncertainty bound is the saved 2.661% development-calibrated bound, not recalibrated on the second batch. The 2.17% margin is the median pairwise repeatability coefficient from E2 and is retained as a descriptive sensitivity threshold, not a singleestimate accuracy bound; 5%, 6%, and 10% are sensitivity thresholds. None is a biological equivalence margin. Subgroup coverage intervals use the 95% Wilson method as an objectindependence approximation.

## V. RESULTS

## A. High overall agreement coexists with systematic representation bias

All 2,720 development objects were retained. The audit flagged 73 objects for low occupancy fraction, four for a raw point-cloud extent discrepancy above 10% on at least one axis, and two for incomplete rendered images. Volume ICC was 0.9939, while mean signed occupancy error was +3.665% (bootstrap interval $+ 3 . 5 7 2 \% \mathrm { ~ t o ~ } + 3 . 7 5 5 \% )$ and median signed error was +3.052%. Occupancy volume exceeded mesh volume for 2,719 of 2,720 objects.

The ICC largely reflects the 68-fold range of mesh volumes. Within volume deciles, the median ICC was 0.8570 (range 0.747–0.983). Bland–Altman analysis gave a bias of +1,369 cubic mesh units (+4.67% of the mean mesh volume), 95% limits of agreement of $\dot { \mathbf { \tau } } - 3 , 5 6 8 \mathbf { t o } + 6 { , } 3 0 6 .$ a percentage error of 16.47% using the mean of both methods (16.85% using mean mesh volume), and a proportional-bias correlation of 0.82: the absolute discrepancy grows with object size (Supplementary Fig. S2). Occupancy extents were nearly unbiased (−0.31% to −0.03% by axis), whereas trimmed point-cloud extents were smaller than mesh extents by 1.50% to 2.31%, the expected effect of discarding extremal points.

In E2, the median estimated full-sample sampling SD was 0.784% (interquartile range 0.508%–1.088%), giving a median repeatability coefficient of 2.172%. The median signed error in this 400-object subset was +2.930% at 100,000 queries, about 3.7 times the median sampling SD. This supports a systematic component beyond sampling variability. Under the binomial query model, sampling SD grows at low occupancy, reaching 2.21% at $p = 0 . 0 2$

TABLE III  
CORRECTION PERFORMANCE. ERRORS ARE PERCENTAGES RELATIVE TO MESH VOLUME. COVERAGE USES EACH METHOD’S OWN CALIBRATED ABSOLUTE-ERROR BOUND.
<table><tr><td>Method</td></tr><tr><td>Bias MdAPE RMSE Bound  $\mathbf { C o v } .$  Internal test, n = 550</td></tr><tr><td>Uncorrected +3.608 3.035 4.286 8.451 96.4% Global correction -0.040 1.370 2.231 4.632 96.4%</td></tr><tr><td>Occupancy-only -0.032 0.580 1.149 2.661 96.2%</td></tr><tr><td>Frozen second batch,  $n = 2 , 7 2 8$ </td></tr><tr><td>Uncorrected +4.497 3.481 5.743 8.451 90.0% 1.503</td></tr><tr><td>Global correction +0.817 3.542 4.632 90.0% Occupancy-only +0.006 0.664 1.743 2.66192.1%</td></tr><tr><td></td></tr></table>

## B. An outward label-boundary offset is observed

Of the 60 fits, 57 met every acceptance criterion; the other three stopped with an optimizer termination warning (not on a bound; midpoints 0.00263–0.00309, within the accepted range) and are excluded from primary summaries. For the accepted fits, the median label-transition midpoint was $t _ { 5 0 } =$ 0.00304 normalized units outside the mesh (interquartile range 0.00287–0.00315; coefficient of variation 0.09), with a median transition width of 0.00027. The median volume-implied offset was 0.00312, a ratio of 0.98, and the per-object Pearson correlation between the two was 0.58 (Spearman 0.35). One object with a large volume-implied displacement raises the Pearson value (without it, $r = 0 . 3 1 )$ . The 32-centroid approximation returned the exact distance for every selected point. No point more than ten transition widths inside the boundary was labeled outside, and at most 0.1% of points more than ten widths outside were labeled inside, supporting a displaced label transition with few far-from-transition inconsistencies (Fig. 1).

The labels therefore describe a surface slightly outside the released mesh, consistent with the volume bias arising in dataset generation.

## C. The frozen correction transfers with reduced coverage

Table III compares internal and frozen second-batch performance. The occupancy-only correction reduced internal MdAPE from 3.035% to 0.580%. On the 2,728 new objects, it reduced MdAPE from 3.481% to 0.664% and RMSE from 5.743% to 1.743%, reductions of 80.9% and 69.7%. Its second-batch mean signed bias was +0.006% (95% bootstrap interval −0.059% to +0.076%), and the MdAPE interval was 0.636%–0.700%.

A near-zero mean residual does not imply uniformly accurate estimates. Coverage of the occupancy-only bound fell to 92.1% (bootstrap interval 91.1%–93.0%). Corrected secondbatch pass rates at 2.17%, 5%, 6%, and 10% were 88.5%, 98.0%, 98.8%, and 99.7%. Fig. 2 shows the complete absoluteerror distributions, including the tails; the vertical line marks the occupancy-only bound. The uncorrected and global methods have identical coverage because they cover the same objects: the global correction divides every estimate by $k =$ $1 + c / 1 0 0 \ ( c = 3 . 6 5 \%$ , the mean training error), so it covers raw errors from $c - b _ { g } k = - 1 . 1 5 \%$ to $c + b _ { g } k = 8 . 4 5 1 \% ;$ the upper limit equals the uncorrected bound because both calibration quantiles fall on the same calibration object, and no raw error fell below the lower limit.

## D. The label pipeline reproduces the bias, and removing its depth offset removes about half of it

The reimplementation reproduced the released labels closely. At the default offset, a median of 98.0% of sampled near-surface query points received the released label, compared with 91.7% at offset 0 and 95.0% at 0.75. Across objects, recreated and released volume errors were almost perfectly correlated (Pearson $r = 0 . 9 9 9 )$ , although the recreated errors were systematically smaller (median +3.43% versus +3.90%; median per-object difference 0.37 percentage points). In ten objects, replacing near-surface interpolated labels with meshcontainment labels (hybrid labeling) changed volume errors by at most 0.26 percentage points and did not close this gap; complete containment labeling was not tested.

Removing the depth offset lowered the occupancy-volume error in every one of the 55 objects, by a mean of 1.57 percentage points (95% bootstrap interval 1.37–1.77), and the median error rose monotonically from 1.578% to 2.649% and 3.435% across 0, 0.75, and 1.5 voxels (Fig. 3); midpoint fits were accepted for 54 of 55 objects at each offset. The fusedmesh volume error fell by 1.48 points, and in the 53 objects with accepted fits at both offsets the label midpoint moved inward in every object, by a mean of 0.00126. Removing the offset therefore removed about 45% of the reproduced inflation. The source of the remainder was not isolated: depth-map erosion and truncated-distance fusion are candidate contributors, but neither was varied separately. The alternative coordinate convention and inclusion of the five pilot objects gave the same effect (1.59 and 1.58 points).

## E. Shape distribution and low occupancy define the transfer limits

The second batch contained larger and less spherical objects: median sphericity was 0.793 versus 0.8358, median $\bar { \boldsymbol { A } } _ { m } / V _ { m } ^ { 2 / 3 }$ was 6.098 versus 5.786, and median occupancy fraction was 0.1138 versus 0.1437. Low-occupancy prevalence increased from 73/2,720 (2.7%) to 181/2,728 (6.6%).

Mean raw error increased by 0.833 percentage points; the development regression on shape complexity predicted 0.747 points (about 90%), a descriptive decomposition. The offset itself replicated: median volume-implied offsets were 0.00293 (development) and 0.00297 (second batch), with coefficients of variation of 0.28 and 0.27, and the slope of error on $A _ { m } / V _ { m } ^ { 2 / 3 }$ was 1.78 and 2.00 percentage points per unit $( R ^ { 2 } =$ 0.67 and 0.69). The scale-aware predictor $A _ { m } s / V _ { m }$ fitted on development objects (slope 0.310, equivalent to $t _ { \mathrm { n o r m } } =$ 0.00310) explained the second-batch errors without refitting $( R ^ { 2 } = 0 . 8 3 ;$ mean residual −0.06 points).

Table IV identifies much higher failure risk below occupancy 0.02, but the flag does not capture all violations: 124 of 216 second-batch failures (57.4%) occurred outside it. Unflagged coverage decreased from 97.8% internally to

![](images/489c674c25ddfbb7d8da3750580357f95fcee1534d38e474bf8af1b8cd70355f.jpg)

![](images/909a35523cdabebf149baec3e3daa7b96c2377d8a3f0be2dd02e79d0a651cd93.jpg)  
Fig. 1. Exact-distance boundary fits for the 57 objects accepted in the recorded analysis, replayed on the original queries. Left: object-specific fitted transitions (thin lines) and the transition at median fitted parameters (bold); zero marks the raw mesh. Right: label-transition midpoint versus volume-implied displacement, with identity line $( r = \mathbf { 0 . 5 8 } )$ ). Open markers show the three fits that failed the acceptance checks; they are excluded from the correlation.

![](images/b90ecd6e56189aecf13a9393b227903a8fedf0b96a42b3b489e5d085c8aee4cd.jpg)  
Fig. 2. Empirical cumulative distributions of absolute percentage error for all 2,728 second-batch objects. Every observation is included. The horizontal axis is linear near zero and logarithmic in the tail. The vertical line is the occupancy-only 2.661% bound; the horizontal dotted line marks 95% coverage.

TABLE IV  
OCCUPANCY-ONLY COVERAGE AT THE SAVED 2.661% BOUND. WILSON INTERVALS ASSUME INDEPENDENT OBJECTS AND DO NOT QUANTIFY BETWEEN-ANIMAL UNCERTAINTY; p IS THE OCCUPANCY FRACTION.
<table><tr><td>Eval.</td><td>Group</td><td>Covered</td><td>Fail.</td><td>MdAPE</td><td>Coverage [95% CI]</td></tr><tr><td>Internal</td><td>All</td><td>529/550</td><td>21</td><td>0.580%</td><td>96.2% [94.2, 97.5]</td></tr><tr><td>Internal</td><td> $p < 0 . 0 2$ </td><td>7/16</td><td>9</td><td>3.011%</td><td>43.8% [23.1, 66.8]</td></tr><tr><td>Internal</td><td> $p \geq 0 . 0 2$ </td><td>522/534</td><td>12</td><td>0.551%</td><td>97.8% [96.1, 98.7]</td></tr><tr><td>Second</td><td>All</td><td>2,512/2,728</td><td>216</td><td>0.664%</td><td>92.1% [91.0, 93.0]</td></tr><tr><td>Second</td><td> $p < 0 . 0 2$ </td><td>89/181</td><td>92</td><td>2.766%</td><td>49.2% [42.0, 56.4]</td></tr><tr><td>Second</td><td> $p \geq 0 . 0 2$ </td><td>2,423/2,547</td><td>124</td><td>0.625%</td><td>95.1% [94.2, 95.9]</td></tr></table>

95.1% in the second batch. Reweighting the internal subgroup coverages to second-batch subgroup proportions gives 94.17% coverage, versus 96.18% internally and 92.08% observed. This attributes 2.01 percentage points of the decline to subgroup composition and a further 2.09 points to within-group changes. Using the audit flag as an applicability restriction is a later interpretation, and the observed 95.1% is not a conditional guarantee. Within the second batch, coverage declined across shape-complexity deciles from 100% to 64.1%, and the highest decile remained under-corrected by a mean of +1.45%. The worst 136 errors had median sphericity 0.557 and occupancy 0.017 (cohort medians: 0.793 and 0.114).

![](images/bd5c71c4c659098d26dc9040fea3f8e4ff4408d4709c4bc7c6255f25d6f7307d.jpg)  
Fig. 3. Controlled label-pipeline test (E8). Occupancy-volume error relative to the raw mesh at depth offsets of 0, 0.75, and 1.5 voxels; thin lines are individual objects and the cross marks the median error of the released labels. The plot includes the five pilot objects; statistics in the text use the other 55.

## F. Real EM processing preserves volume more than surface descriptors

All 550 eligible MitoEM-R objects were meshed. The median mesh-versus-voxel volume difference was −0.19% (5th–95th percentile −0.35% to −0.10%). The median meshversus-voxel-face surface difference was −16.31%, illustrating the dependence of digitized-surface measurements on the area estimator [33]. The median trimmed point-cloud extent difference was −3.70%, whereas raw sampled extents differed by only −0.03%, reproducing on real EM the trimming effect seen in 3DMSL.

TABLE V  
MITOEM-R IN-PLANE REGRIDDING RELATIVE TO NATIVE-GRID MESHES FOR THE SAME 550 OBJECTS (MEDIANS; BRACKETS SHOW 5TH–95TH PERCENTILES). ρ: SPHERICITY RANK CORRELATION.
<table><tr><td>Spacing, nm</td><td>Volume</td><td>Surface area</td><td>Sphericity</td><td>ρ</td></tr><tr><td> $3 0 \times 1 6 \times 1 6$ </td><td>+0.98%</td><td>-5.84%</td><td>+6.90% 0.997</td><td></td></tr><tr><td></td><td>[+0.45, +1.46]</td><td>[-7.68, −4.72]</td><td>[+5.68, +8.69]</td><td></td></tr><tr><td> $3 0 \times 2 4 \times 2 4$ </td><td>-0.08%</td><td>-10.60%</td><td>+11.76% 0.994</td><td></td></tr><tr><td></td><td></td><td>[-0.93, +0.41] [-12.14, -8.67] [+9.24, +13.75]</td><td></td><td></td></tr></table>

All meshes in the original regridding analysis remained watertight. At the original 24 nm in-plane grid phase, volume rank correlation was 1.000 to reported precision, while median sphericity increased by 11.76% (Table V). The near-zero volume change at 24 nm holds across grid alignments, but the 16 nm result does not: across all block phases and tie rules (Supplementary Section S3), median volume changes ranged from −0.22% to +0.01% at 24 nm but from −5.47% to +6.94% at 16 nm. Euler characteristics changed in nine of 550 objects at each spacing, so watertightness does not guarantee topology preservation. These shifts are not correction factors for other tissues or instruments.

## VI. DISCUSSION

## A. Measurement agreement is a prerequisite for biological interpretation

The experiments separate three properties that are easily conflated. An estimator can be numerically precise while targeting a boundary that differs from the reference mesh (sampling SD about 0.8%, systematic offset nearly four times larger); a correction can remove average bias while leaving substantial errors in a morphological subgroup; and object rankings can remain stable while a descriptor’s absolute scale changes. The high ICC, the near-zero corrected mean bias, and the high regridding rank correlations are each informative, but none alone establishes interchangeability.

For users of 3DMSL and similar resources, three recommendations follow. First, occupancy labels describe a surface about 0.003 normalized units outside the released mesh, so derived volumes and models trained on them inherit a bias whose relative size scales with $A _ { m } s / V _ { m }$ . Second, in our reimplementation, zero depth offset removed only about half of this bias; labeling query points directly against the released mesh avoids the fusion step. Third, a mesh-free correction reduces typical error about fivefold, but its uncertainty bound should not be relied on for objects with occupancy fraction below 0.02, and it does not guarantee coverage for other objects either.

## B. Correction and uncertainty transfer are separate results

Observed coverage was near 95% in the unflagged group and about 49% in the flagged group, and it declined within the unflagged group between evaluations, illustrating the distinction between marginal and conditional coverage [27]. Conditional residual scales, Mondrian calibration by occupancy group, or covariate-shift weighting [28] are natural next steps, but they require new development, and the second batch could not serve as an untouched test for methods selected using its outcomes.

One potential application is Homeostatic Invariance, which we propose as a framework in which structural variability may coexist with maintained function within a bounded biological regime. The findings address a measurement prerequisite for testing that proposition, not the proposition itself. Such a test would need aligned functional observations, longitudinal or perturbational designs, and replication across biological units; the present datasets contain no linked respiration, ATP, or membrane-potential measurements.

## VII. LIMITATIONS

The two 3DMSL batches share construction and provenance; disjoint IDs do not establish independent acquisitions or animals. Object-level bootstrap intervals may be optimistic under spatial or biological dependence. The MitoEM-R analysis covers one rat-cortex volume and excludes boundarytruncated, small, shallow, and very large objects. The internal split followed exploratory analysis, and E8 and the supplementary analyses were added after the primary results had been inspected (Supplementary Material, Section S4). E8 is a reimplementation whose correspondence to the historical code has not been confirmed with the dataset authors; it underpredicts the released errors by a median of 0.37 percentage points, labels points from the interpolated field, and does not separate erosion from fusion. No new-source evaluation of the correction was performed. The regridding experiment leaves z unchanged and does not model the point-spread function, segmentation uncertainty, or another instrument’s acquisition. Candidate tolerances are not biological equivalence margins.

## VIII. CONCLUSION

This study establishes four empirical findings for quantitative mitochondrial imaging: high whole-sample agreement can conceal a systematic representation bias; a controlled processing change can reduce that bias without eliminating it; correction accuracy can transfer while calibrated coverage deteriorates; and descriptor rankings can remain stable while absolute measurements shift. The resulting validation requirements are concrete: report signed object-level errors alongside agreement coefficients, test correction accuracy and uncertainty coverage separately on held-out objects, and assess grid alignment and tie rules before interpreting descriptor changes. The reported effect sizes apply to the tested resources and procedures and are not universal correction factors; biological function and cross-source correction transfer remain for future validation.

## DATA, CODE, AND ETHICS

The source datasets, 3DMSL [25] and MitoEM [19], are publicly available. The reproducibility code and supporting materials, including executed notebooks, processed perobject results, split identifiers, and the frozen model, are publicly available at https://github.com/farouze/ mitocondria. This study analyzed only publicly available, previously published datasets. No new animal experiments or human-participant research were conducted.

## REFERENCES

[1] A. M. van der Bliek, Q. Shen, and S. Kawajiri, “Mechanisms of mitochondrial fission and fusion,” Cold Spring Harbor Perspectives in Biology, vol. 5, no. 6, p. a011072, 2013. [Online]. Available: https://doi.org/10.1101/cshperspect.a011072

[2] G. Benard et al., “Mitochondrial bioenergetics and structural network organization,” Journal of Cell Science, vol. 120, pp. 838–848, 2007. [Online]. Available: https://doi.org/10.1242/jcs.03381

[3] J. Faitg, T. Davey, D. M. Turnbull, K. White, and A. E. Vincent, “Mitochondrial morphology and function: two for the price of one!” Journal of Microscopy, vol. 278, pp. 89–106, 2020. [Online]. Available: https://eprints.ncl.ac.uk/266059

[4] W.-W. Tseng et al., “Metabolic regulation of mitochondrial morphologies in pancreatic beta cells: coupling of bioenergetics and mitochondrial dynamics,” Communications Biology, vol. 7, p. 1267, 2024. [Online]. Available: https://doi.org/10.1038/s42003-024-06955-3

[5] Y. Ding et al., “Single Mitochondrion Morphology-Function Relationship Analysis Using Fluorescent Probes and Artificial Intelligence,” Advanced Science, vol. 12, no. 43, p. e09140, 2025. [Online]. Available: https://doi.org/10.1002/advs.202509140

[6] S. Charrasse et al., “Combining mitochondrial morphology and cellular metabolism measurements improves colorectal cancer cell classification,” Advances in Cancer Biology - Metastasis, vol. 16, p. 100175, 2026. [Online]. Available: https://doi.org/10.1016/j.adcanc. 2026.100175

[7] O. Tovar-Ferrero, J. Rubio, A. Zorzano, G. Martinez-Corrales, and M. Liesa, “Measuring mitochondrial membrane potential,” The EMBO Journal, vol. 44, pp. 7334–7345, 2025. [Online]. Available: https://doi.org/10.1038/s44318-025-00632-9

[8] A. J. Valente, L. A. Maddalena, E. L. Robb, F. Moradi, and J. A. Stuart, “A simple ImageJ macro tool for analyzing mitochondrial network morphology in mammalian cell culture,” Acta Histochemica, vol. 119, no. 3, pp. 315–326, 2017. [Online]. Available: https: //doi.org/10.1016/j.acthis.2017.03.001

[9] M. P. Viana, S. Lim, and S. M. Rafelski, “Quantifying mitochondrial content in living cells,” in Methods in Cell Biology. Academic Press, 2015, vol. 125, pp. 77–93. [Online]. Available: https: //doi.org/10.1016/bs.mcb.2014.10.003

[10] A. E. Y. T. Lefebvre et al., “Nellie: automated organelle segmentation, tracking and hierarchical feature extraction in 2D/3D live-cell microscopy,” Nature Methods, vol. 22, pp. 751–763, 2025. [Online]. Available: https://doi.org/10.1038/s41592-025-02612-7

[11] S. Zaghbani, L. Faber, and A. J. Garcia-Saez, “Convolutional Neural Network approach to classify mitochondrial morphologies,” Computational Biology and Chemistry, vol. 118, p. 108477, 2025. [Online]. Available: https://doi.org/10.1016/j.compbiolchem.2025. 108477

[12] A. Kaianathbhatta et al., “MitoTex (Mitochondria Texture Analysis User Interface): Open-Source Framework for Textural Characterization and Classification of Mitochondrial Structures,” International Journal of Molecular Sciences, vol. 27, no. 3, p. 1191, 2026. [Online]. Available: https://doi.org/10.3390/ijms27031191

[13] A. R. Punnakkal, S. S. Jadhav, A. V. Celeste, A. Horsch, K. Agarwal, and D. K. Prasad, “3D Mitochondria Shape Library for Optical Microscopy (3DMSL): A multimodal dataset for deep learning based mitochondrial analysis,” Data in Brief, vol. 65, p. 112507, 2026. [Online]. Available: https://doi.org/10.1016/j.dib.2026.112507

[14] G. Parlakgul and G. S. Hotamisligil, “High resolution 3D imaging of¨ liver subcellular architecture and its link to metabolic function,” https: //doi.org/10.6019/EMPIAR-10791, eMPIAR-10791, dataset, released 2022, accessed Sep. 25, 2026.

[15] G. Parlakgul¨ et al., “Regulation of liver subcellular architecture controls metabolic homeostasis,” Nature, vol. 603, pp. 736–742, 2022. [Online]. Available: https://doi.org/10.1038/s41586-022-04488-5

[16] BioAILab, “3DMSL,” https://github.com/bioailab/3DMSL, official dataset code repository, accessed Sep. 24, 2026.

[17] L. Mescheder, M. Oechsle, M. Niemeyer, S. Nowozin, and A. Geiger, “Occupancy Networks: Learning 3D Reconstruction in Function Space,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 4460–4470. [Online]. Available: https://openaccess.thecvf.com/ content CVPR 2019/html/Mescheder Occupancy Networks Learning 3D Reconstruction in Function Space CVPR 2019 paper.html

[18] D. Stutz and A. Geiger, “Learning 3D Shape Completion Under Weak Supervision,” International Journal of Computer Vision, vol. 128, pp. 1162–1181, 2020. [Online]. Available: https://doi.org/10.1007/ s11263-018-1126-y

[19] D. Wei et al., “MitoEM Dataset: Large-scale 3D Mitochondria Instance Segmentation from EM Images,” in Proc. Med. Image Comput. Comput. Assist. Interv. (MICCAI), ser. Lecture Notes in Computer Science, vol. 12265, 2020, pp. 66–76. [Online]. Available: https://doi.org/10.1007/978-3-030-59722-1 7

[20] D. Franco-Barranco et al., “Current Progress and Challenges in Large-Scale 3D Mitochondria Instance Segmentation,” IEEE Transactions on Medical Imaging, vol. 42, no. 12, pp. 3956–3971, 2023. [Online]. Available: https://doi.org/10.1109/TMI.2023.3320497

[21] K. O. McGraw and S. P. Wong, “Forming inferences about some intraclass correlation coefficients,” Psychological Methods, vol. 1, no. 1, pp. 30–46, 1996. [Online]. Available: https://doi.org/10.1037/ 1082-989X.1.1.30

[22] T. K. Koo and M. Y. Li, “A Guideline of Selecting and Reporting Intraclass Correlation Coefficients for Reliability Research,” Journal of Chiropractic Medicine, vol. 15, no. 2, pp. 155–163, 2016. [Online]. Available: https://doi.org/10.1016/j.jcm.2016.02.012

[23] J. M. Bland and D. G. Altman, “Statistical methods for assessing agreement between two methods of clinical measurement,” The Lancet, vol. 327, no. 8476, pp. 307–310, 1986. [Online]. Available: https://doi.org/10.1016/S0140-6736(86)90837-8

[24] L. A. H. Critchley and J. A. J. H. Critchley, “A meta-analysis of studies using bias and precision statistics to compare cardiac output measurement techniques,” Journal of Clinical Monitoring and Computing, vol. 15, no. 2, pp. 85–91, 1999. [Online]. Available: https://doi.org/10.1023/A:1009982611386

[25] A. R. Punnakkal et al., “3DMSL - 3D Mitochondria Shape Library for Optical Microscopy,” https://doi.org/10.18710/JX6JXF, dataverseNO, dataset, version 2, 2024.

[26] A. N. Angelopoulos and S. Bates, “A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification,” https:// arxiv.org/abs/2107.07511, arXiv:2107.07511, 2021.

[27] V. Vovk, “Conditional validity of inductive conformal predictors,” Machine Learning, vol. 92, no. 2–3, pp. 349–376, 2013. [Online]. Available: https://doi.org/10.1007/s10994-013-5355-6

[28] R. J. Tibshirani, R. F. Barber, E. J. Candes, and A. Ramdas,\` “Conformal Prediction Under Covariate Shift,” in Advances in Neural Information Processing Systems, vol. 32, 2019. [Online]. Available: https://papers.neurips.cc/paper files/paper/2019/ hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html

[29] A. Jacobson, L. Kavan, and O. Sorkine-Hornung, “Robust inside-outside segmentation using generalized winding numbers,” ACM Transactions on Graphics, vol. 32, no. 4, pp. 33:1–33:12, 2013. [Online]. Available: https://igl.ethz.ch/projects/winding-number/

[30] D. Stutz, “mesh-fusion: Depth rendering and volumetric fusion implementation,” https://github.com/davidstutz/mesh-fusion/blob/master/2 fusion.py, source file 2 fusion.py, accessed Sep. 25, 2026.

[31] W. E. Lorensen and H. E. Cline, “Marching cubes: A high resolution 3D surface construction algorithm,” ACM SIGGRAPH Computer Graphics, vol. 21, no. 4, pp. 163–169, 1987. [Online]. Available: https: //www.cs.toronto.edu/<sup>∼</sup>jacobson/seminar/lorenson-and-cline-1987.pdf

[32] B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York, NY, USA: Chapman & Hall, 1993. [Online]. Available: https://www.routledge.com/An-Introduction-to-the-Bootstrap/ Efron-Tibshirani/p/book/9780412042317

[33] J. Lindblad, “Surface area estimation of digitized 3D objects using weighted local configurations,” Image and Vision Computing, vol. 23, no. 2, pp. 111–122, 2005. [Online]. Available: https: //doi.org/10.1016/j.imavis.2004.06.012