# Learning from Shared-Control Overrides: Context-Driven Acceleration Profile Prediction for Personalized Overtaking

Ruizheng Xu<sup>1,2</sup>, Lounis Adouane<sup>1</sup>, Javier Ibanez-Guzm˜ an´ <sup>2</sup>, Clement Zinoune´ <sup>2</sup>

Abstract— Adaptive Cruise Control (ACC) systems are typically calibrated for an average driver, often resulting in a mismatch between vehicle behavior and individual expectations during time-critical maneuvers such as highway overtaking. When the ACC is perceived as too conservative and inconsistent, drivers intervene through throttle overrides, providing implicit feedback on the system’s behavior.

This paper reframes these override actions as human-in-theloop supervisory signals and proposes a data-driven framework for personalized vehicle adaptation, termed Context-driven Personalized ACC (CoP-ACC). Rather than relying solely on end-to-end regression, which tends to over-smooth dynamic responses, we introduce a hybrid pipeline combining: (i) unsupervised hierarchical clustering to extract representative acceleration profiles from override events; (ii) a context classifier that maps pre-maneuver driving conditions to the appropriate profile; and (iii) a residual regressor that refines the selected profile into a smooth, personalized acceleration profile tailored to the immediate context.

Evaluated on real-world public-road data against a withheld forced-ACC baseline, the approach demonstrates high reconstruction fidelity and generates acceleration profiles that tend toward the driver’s expected behavior in potential override contexts. The results highlight the potential of learning from shared-control overrides to enable anticipatory, personalized ACC behavior, reducing manual interventions and improving ride comfort.

## I. INTRODUCTION

Adaptive Cruise Control (ACC) significantly reduces driver workload [1], but its “one-size-fits-all” calibration limits user acceptance during complex highway maneuvers. Notably, drivers commonly keep ACC engaged during overtaking. We define this scenario as an ACC-assisted overtake: the driver manually initiates and steers the lane change, while the ACC remains active and automatically controls longitudinal acceleration to manage speed relative to the overtaken vehicle. Because drivers possess highly individualized driving tolerances, generic ACC systems often execute conservative accelerations that feel delayed or insufficient [2]. This human-machine discrepancy forces the driver to intervene and physically override the pedal to achieve the desired overtake urgency [3].

Recent ACC personalization studies [4] predominantly target steady-state car-following or rely on end-to-end continuous regression. Applied to highly variable driving data where even a single individual driver exhibits significantly different behaviors depending on their immediate situational context [5], these regressors suffer from “regression to the mean,” producing over-smoothed profiles that fail to capture urgency during transient maneuvers like overtaking. Furthermore, end-to-end models lack interpretability for functional safety. Finally, although some studies have begun using driver overrides to adapt gap preferences in car-following [6], none exploit them to learn continuous, context-dependent acceleration profiles for transient maneuvers like overtaking.

To bridge this gap, this paper proposes CoP-ACC, a Context-Driven Personalized ACC framework treating manual interventions as explicit ground-truth labels of driver expectation, formally defined as shared control override events. Targeting the overtaking scenario, we introduce a hybrid machine-learning pipeline to overcome regression limitations. Unsupervised hierarchical clustering extracts discrete acceleration profiles, preserving dynamics. This intent is mapped to pre-maneuver kinematics using supervised classification (Random Forest), ensuring interpretability. Finally, a Context-Conditioned 1D-CNN Decoder generates continuous residual tracking modifications. We introduce a distributional evaluation methodology, proving the framework’s ability to preemptively fulfill individual driver expectations on withheld data.

This paper is organized as follows: Section II reviews related work in ADAS personalization. Section III outlines the proposed methodology. Section IV details the data acquisition and experimental protocol. Section V presents the experiments and results. Section VI concludes the paper and discusses future works.

## II. RELATED WORKS

This section reviews two bodies of work relevant to the proposed framework: personalization techniques for longitudinal vehicle control in Section II-A and the current state of overtaking automation and shared control in Section II-B.

## A. Personalization in Longitudinal Control

User acceptance of Advanced Driver Assistance Systems (ADAS) is heavily dependent on how closely the machine’s behavior aligns with an individual driver’s internal expectations [4]. To address the limitations of rigid traditional control (e.g., MPC, PID), recent literature has shifted toward data-driven longitudinal personalization, aiming to deduce and replicate these individualized preferences.

One primary approach to individual personalization relies on style categorization. Techniques such as unsupervised clustering and Inverse Reinforcement Learning (IRL) are used to extract individualized driving styles from traffic data [7]–[9]. For instance, [7] successfully employs clustering and classification to identify driver styles online, but relies on traditional MPC for longitudinal control. Similarly, [6] use IRL to dynamically update scalar gap preferences. While these methods successfully categorize individual behaviors, two main limitations are: an exclusive focus on steady-state gap maintenance (car-following) and reliance on heuristic physical controllers for output. They do not address the highly transient, aggressive dynamics required for maneuverspecific accelerations, such as overtaking.

Another approach achieves more nuanced individual control by formulating personalization as a continuous trajectory prediction problem. Both works from [10] and [11] utilize methods such as Gaussian Process Regression (GPR) and Gaussian Mixture Regression (GMR) to predict acceleration directly from environmental states. While effective, treating individual personalization exclusively as an end-to-end regression task presents two critical limitations. First, by skipping an explicit style categorization pipeline, these models naturally converge toward overly smoothed average trajectories (“regression to the mean”). Because human driving data embodies both gentle and aggressive reactions, a single continuous regressor allows to smooth out these extremes. Consequently, these models often fail to execute the decisive, high-urgency acceleration peaks unique to individual drivers—the sudden surges in speed strictly required to safely clear a slower vehicle during overtaking. Second, when predicting trajectories using complex, end-to-end models like Artificial Neural Networks [12], the system functions as a “black box”. This lack of interpretability makes it mathematically impossible to attribute exactly which environmental feature triggered a specific acceleration output, posing a significant challenge for safety compliance in modern ADAS.

The proposed CoP-ACC methodology overcomes these limitations by introducing a novel, hybrid machine-learning pipeline designed explicitly for individual driver personalization (cf. Section III). By utilizing unsupervised clustering to extract discrete acceleration profile templates prior to supervised classification and continuous residual regression, the proposed framework prevents over-smoothing and preserves the unique dynamics of each driver based on its situational context. Furthermore, the classification stage employs an interpretable Random Forest whose feature contributions are quantified through SHapley Additive exPlanations (SHAP) [13] analysis, ensuring transparent mapping between premaneuver environmental features and the predicted driver intent. This approach effectively bridges the gap between discrete driving styles and continuous profile prediction, providing a solution specifically tailored for the overtaking maneuver.

## B. Overtaking and Shared Control

Overtaking is a complex, multi-phase maneuver [14] that presents a significant challenge for generalized ADAS. When a generic ACC performs a conservative acceleration that fails to match the driver’s expectations, the human-machine mismatch prompts a shared control override via the pedals [15]. While works like [16] extensively study shared control during overtaking, their focus remains predominantly on lateral steering conflicts and safety assessments rather than learning the personalized longitudinal acceleration style. Additionally, while recent literature has begun treating overrides as implicit feedback, they are generally used to incrementally tune scalar parameters such as gap preferences [6], rather than to learn the continuous acceleration profile expected by the driver during a specific transient maneuver.

Currently, there is a distinct lack of frameworks that extract the full, continuous kinematic trajectory of a shared control override to use as a primary teaching signal for overtaking maneuvers. This paper addresses this gap by exclusively utilizing these real-world shared-control conflicts to drive the aforementioned hybrid personalization pipeline. By treating shared control overrides as ground-truth acceleration profiles rather than simple post-hoc parameter tuners, the proposed framework can preemptively output the precise acceleration curve required to resolve the human-machine conflict during an overtake before it arises.

## III. COP-ACC: CONTEXT-DRIVEN PERSONALIZED ACC FRAMEWORK

This section first describes the ACC operating principle and the shared control mechanism exploited by CoP-ACC, then details the framework architecture, the unsupervised profiling stage, pre-maneuver feature extraction, and the context-driven classification and residual regression pipeline.

## A. ACC Operating Principle

ACC operates in two modes: cruise mode, maintaining a driver-set target speed, and following mode, adjusting speed to preserve a safe gap behind a target vehicle. Driver pedal inputs govern state transitions: pressing the throttle pedal triggers a temporary override allowing manual acceleration, and on pedal release the ACC automatically resumes its target; pressing the brake pedal immediately deactivates the system and returns full control to the driver [17]. We define this driver intervention as a shared control override. During overtaking, the ACC remains engaged in following mode. However, its conservative acceleration often falls short of the driver’s intended acceleration, resulting in a throttle override that constitutes the shared control signal studied in this work.

## B. Framework Overview

We propose a hybrid, context-driven architecture based on machine learning methods (cf Fig. 1). The framework operates in two primary stages: offline unsupervised profiling of driver behavioral expectations, and online supervised classification and regression to predict the appropriate personalized acceleration profile.

As shown in Fig. 1, this architecture is separated into an offline training phase and an online inference phase. During offline training, the pipeline proceeds in three steps: (1) raw acceleration traces from overtaking events are clustered via time-series clustering to identify distinct driver behavioral preferences, producing K base centroid profiles; (2) each event is assigned a cluster label, and a context classifier is trained to predict this label from an 11-dimensional premaneuver kinematic and behavioral feature vector; (3) a residual regressor is trained to predict the individual residual between each event’s actual profile and its assigned centroid, conditioned on both the context features and the predicted cluster label.

![](images/fc43cfc31b868653ceca149bf82fc74cef034ca1f72e3feca9ac32d5c4b22562.jpg)  
Fig. 1. Overview of the proposed CoP-ACC framework: offline unsupervised profiling and online context-driven prediction pipeline.

During online inference, the system receives a new premaneuver context vector and executes the trained pipeline: the classifier predicts the appropriate cluster, the regressor generates the corresponding residual curve, and the final personalized acceleration profile is computed as the sum of the selected base centroid and the predicted residual. The concrete data collection protocol, temporal windowing, and event annotation process are detailed in Section IV.

## C. Unsupervised Acceleration Profile Clustering

The first phase extracts a discrete set of behavioral profiles from the uncorrupted dataset, which strictly comprises naturalistic manual driving and shared control override events. We explicitly exclude passively accepted ACC events, as only active pedal interventions contain the true ground-truth of the driver’s desired acceleration. This is framed as a timeseries clustering problem.

We employ Agglomerative Hierarchical Clustering [18] utilizing Euclidean distance on the set of normalized acceleration profiles $\mathcal { A } = \{ { \bf a } _ { 1 } , \ldots , { \bf a } _ { N } \}$ with Ward’s minimum variance criterion to ensure dense, spherical clusters [19].

To ensure the framework remains scalable across different individual drivers, the number of behavioral profiles (K) is automatically determined via a Silhouette Stability Analysis [20]. This method calculates Silhouette Scores across a range of K, evaluating the performance drop between successive values. The algorithm dynamically selects the first K where the gap to $K + 1$ drops below a predefined stability threshold, indicating diminishing returns in structural distinction. For our subject, optimal separation stabilized at $K = 3$ . The resulting clusters, $C _ { k }$ where $k \in \{ 0 , 1 , 2 \}$ , are defined by their base centroid acceleration profile $\mathbf { a } _ { k } ^ { * }$ (cf. Fig. 4).

## D. Pre-Maneuver Feature Extraction

To predict the expected profile entirely from pre-maneuver conditions [21], we extracted 11 state variables to populate the context vector $\mathbf { x } _ { i } .$ . These features comprise two categories: Kinematic State features including Ego Speed, Relative Speed, Distance to Target, Speed Deficit to Target, Time Headway, and Inverse Time-To-Collision evaluated at $t \ = \ 0 ;$ and Behavioral Style features including standard deviations of steering angle, longitudinal jerk, and throttle, along with maximum lateral jerk and mean target acceleration aggregated over the $[ - 5 s , 0 s ]$ context pre-maneuver window (cf. Fig. 1).

## E. Context-Driven Classification and Residual Regression

With the context vectors $\mathbf { x } _ { i }$ and corresponding cluster labels $y _ { i } ~ \in ~ \{ 0 , 1 , 2 \}$ established, we employ a Random Forest classifier [22] to map the pre-maneuver context to a discrete base profile yˆ. Both classifier and residual regressor use an 80/20 stratified split, and implementation-level model settings are summarized in Fig. 1. Both models are trained on the combined dataset of naturalistic manual driving and shared control override events, allowing them to learn the driver’s underlying preference through both how they naturally execute overtaking maneuver and how they explicitly override the system.

However, mapping to a single rigid centroid only provides an averaged expectation based on the categorical cluster. It fails to capture the fluid continuity of human driving or adjust for inner-cluster variance. Therefore, we introduce a Continuous Profile Prediction strategy via Residual Regression. Recognizing that the driver’s ultimate input is a modification of the base intent, we compute the residual curve as:

$$
\mathbf { r } _ { i } = \mathbf { a } _ { i } - \mathbf { a } _ { y _ { i } } ^ { * }\tag{1}
$$

A context-conditioned 1D-CNN Decoder is trained to predict this continuous residual curve ˆr. The decoder conditions on both the 11-dimensional context vector and the predicted cluster label. The cluster label is represented through a learned embedding. The combined conditioning information is then mapped through fully connected layers and transposed one-dimensional convolutions to generate a fixed-length 100-point residual acceleration profile. The final generated acceleration profile is the summation of the base centroid and the predicted residual. This hybrid approach preserves the discrete behavioral intent while injecting the nuanced, continuous adjustments demanded by the real-time context.

## IV. DATA ACQUISITION AND PROCESSING

This section describes the experimental setup, data collection protocol, and the curation pipeline used to prepare the training dataset.

## A. Experimental Setup

Data were logged directly from an instrumented vehicle’s (Renault Austral cf. Fig 2a) Controller Area Network (CAN) bus with production-level ADAS. Four primary signal categories were identified from the telemetry: Vehicle States characterizing ego and target kinematics (e.g., longitudinal/lateral acceleration, relative speed, distance); Driver Inputs & HMI detailing physical interventions and statuses (e.g., throttle position, steering angle, ACC overrides); Environmental Context defining the tactical driving corridor (e.g., lane positioning, speed limits).

To achieve individual-level ADAS personalization, this study deliberately adopts a single-subject data collection approach for this initial framework, aiming to demonstrate personalization for one specific driver before scaling up. The subject is an eight-year experienced highway commuter with a dynamic driving style. While this provides the isolated signals necessary to establish the methodology, we plan future data collection across diverse driver profiles to validate generalizability.

(a)  
![](images/2165bce42cadcb87b56c305903b4cc490819c4433e6d3d3a428a87274cfac3ce.jpg)

![](images/0e4fb418076d45660faac647a637931d77ceca19bb9d4355f5f987d4980fdf2c.jpg)  
(b)  
Fig. 2. (a) Instrumented vehicle used for data collection. (b) GNSS path of round trips with annotated overtaking events.

The data collection procedure was conducted on a threelane highway in France during low-traffic periods. The driver completed three predefined round trips as shown in Figure 2, each capturing a distinct driving paradigm. In Trip 1 (Forced-ACC Baseline), the vehicle was entirely controlled by the ACC and the driver was strictly forbidden from overriding via the pedals, establishing a reference dataset of the $\mathbf { A C C } \mathbf { \ ' } _ { \mathbf { S } }$ generic acceleration profile. In Trip 2 (Naturalistic Manual), the ACC was completely disabled and the driver operated the vehicle naturally during overtakes. In Trip 3 (Shared-Control), the ACC was active, but the driver freely pressed the accelerator pedal to override the system whenever its longitudinal behavior failed to meet their personal expectations, providing a ground-truth shared control override acceleration profile.

## B. Data Curation and Event Annotation

All raw vehicle signals were first synchronized to a unified 50 Hz timebase. To maintain data integrity, continuous physical signals (e.g., speed) were aligned using linear interpolation, while discrete categorical data (e.g., ACC status) were mapped using nearest-neighbor interpolation.

Following synchronization, overtaking events were manually annotated based on the four-phase overtaking model proposed by [14]. We exclusively annotated the start $( t = 0 )$ and end $( t _ { e n d } )$ times of the second phase (active passing phase), resulting in 223 total identified overtaking events. For each event, we extracted a Context Window $( [ - 5 s , 0 s ] )$ a 5-second pre-maneuver window aggregating kinematic and environmental state variables into the input feature vector x<sub>i</sub>, and an Action Window $( [ 0 s , t _ { e n d } ] )$ , the duration of the active overtaking phase capturing the specific longitudinal acceleration profile a<sub>i</sub> (cf. Fig. 1).

To accommodate varying maneuver durations, action window data points were geometrically normalized to a fixed length for profile clustering and as the target output for sequence regression. In contrast, pre-maneuver context features were extracted as scalars from the raw context window for both intent classification and continuous residual regression.

To strictly isolate actual driver expectations, we excluded all pure ACC events from Trip 1 and events where the driver did not override the ACC in Trip 3. Consequently, the final training dataset comprised $\mathrm { N } = 1 4 0$ valid events of unconstrained manual driving (Trip 2) and shared control overrides (Trip 3).

## V. EXPERIMENTS AND RESULTS

The CoP-ACC framework is evaluated across three sequential dimensions: discrete classification accuracy, continuous profile fidelity, and distributional kinematic analysis.

To evaluate the online prediction capabilities, the Random Forest Classifier was validated utilizing a 5-Fold Stratified Cross-Validation protocol with balanced class weighting. For generating the continuous expected acceleration, the 1D-CNN Decoder was evaluated on the computed residual datasets utilizing data augmentations (e.g., Gaussian Noise, Temporal Warping) to stabilize learning under limited human kinematic data. The resulting personalized profiles were validated through two complementary tests: Pairwise Profile Reconstruction Fidelity and Distributional Kinematic Analysis described in Section V-C and V-D.

## A. When and Why Drivers Override the ACC

Prior to generating personalized profiles, a statistical analysis of the pre-maneuver contexts examines why and when this individual rejects standard machine behavior. By analyzing experimental round trips where interventions were explicitly permitted, we identified sharp geometric boundaries that distinguish genuine ACC acceptance from bottleneck situations where inadequate acceleration prompts a manual override.

![](images/6d7ef16f1edc7e28dfb611727479ec6f88a4a73c1491eae7c1e409d339072391.jpg)

![](images/37d8396120169668000927874bfcdf88fc96a2f923fe1eebdc9134dc1ef5d6d2.jpg)

![](images/e3522b6ac1d6e2f0c3063656b3a04810e8cdf4975ca600e61b4d2b6aa6f1ad30.jpg)  
Fig. 3. Distributional comparison of pre-maneuver kinematic features between ACC-accepted and shared control override events at overtake onset $( t = 0 )$

Fig. 3 compares the distributions of three pre-maneuver features between ACC-accepted and shared control override events. The left column shows that override events concentrate at notably higher speed deficits, meaning the driver intervenes when the ego vehicle is travelling well below the speed limit or its ACC target speed. The center column shows that overrides occur at lower ego speeds, reinforcing that the driver is stuck behind a slower vehicle. The right column reveals a tighter distance distribution shifted toward shorter gaps for override events, indicating the ACC’s response becomes insufficient at close range. Together, these three observations suggest that the driver is more likely to override the standard ACC when constrained close behind a target with a large gap between current and desired speed, justifying the necessity of a context-driven prediction framework.

## B. Acceleration Profile Clustering and Classification

The unsupervised hierarchical clustering algorithm identified three distinct empirical profiles as shown in Fig. 4: Cluster 0 (aggressive, high-intensity acceleration peaking near $\mathrm { 0 . 1 5 ~ m / s ^ { 2 } ) } .$ , Cluster 1 (moderate, medium-intensity peaking near $0 . 0 8 ~ \mathrm { m / s ^ { 2 } } )$ , and Cluster 2 (passive, near-zero flat profile at $0 . 0 1 ~ \mathrm { m / s } ^ { 2 } )$ . Contextual correlations reveal that Clusters 0 and 1 correspond primarily to accelerative overtakes, where the ego vehicle initiates the pass from close behind the target and must actively accelerate to complete it. Conversely, Cluster 2 corresponds to flying overtakes, where the ego vehicle approaches with sufficient speed margin and passes at a comfortable distance, requiring minimal acceleration adjustment [14], [23].

![](images/2a930cb276848e6742db6a679f31606d49144111ee561d9406095ea0aab2a436.jpg)  
Fig. 4. Longitudinal acceleration profiles for the three identified clusters $\bar { ( K = 3 ) }$ . In red the mean, in light gray each individual profile.

As detailed in Table I, the clustering separates humanmachine conflict geometries. Remarkably, 64.4% of all maneuvers assigned to Clusters 0 and 1 (aggressive and moderate acceleration) originated directly from shared control override events, indicating that these high-intensity profiles are primarily triggered when the individual must compensate for the generic ACC’s insufficient acceleration.

TABLE I  
DISTRIBUTION OF DRIVING MODES ACROSS THE DISCRETE CLUSTERING PROFILES
<table><tr><td rowspan=1 colspan=1>Event Driving Mode</td><td rowspan=1 colspan=1>Cluster 0</td><td rowspan=1 colspan=1>Cluster 1</td><td rowspan=1 colspan=1>Cluster 2</td><td rowspan=1 colspan=1>Total</td></tr><tr><td rowspan=1 colspan=1>Naturalistic Manual</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>61</td><td rowspan=1 colspan=1>88</td></tr><tr><td rowspan=1 colspan=1>Shared-Control Override</td><td rowspan=1 colspan=1>29</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>52</td></tr><tr><td rowspan=1 colspan=1>Total</td><td rowspan=1 colspan=1>48</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>140</td></tr></table>

Feature importance analysis evaluated via SHAP in Fig. 5 revealed that kinematic motivation variables, specifically speed deficit and ego speed, dominated the cluster prediction. In contrast, behavioral style variables had negligible predictive influence, indicating that the acceleration profile of this individual driver primarily depends on specific kinematic tolerance thresholds.

![](images/8432f50e4f2d75a46de370458629602755493353dc6f8ef600611066e2e39c62.jpg)  
Fig. 5. SHAP global feature importance for Random Forest cluster classification.

Exploiting these kinematic boundaries, the Random Forest classifier achieved an accuracy of 89.29% on the test set, with macro precision of 85.19%, macro recall of 86.67%, and macro F1-score of 85.65%.

## C. Pairwise Profile Reconstruction Fidelity

This test evaluates whether the predicted profile $\hat { \mathbf { a } } _ { i } = \mathbf { a } _ { \hat { y } } ^ { \ast } +$ ˆr can reconstruct the driver’s actual acceleration ${ \bf a } _ { i }$ at each of the $L = 1 0 0$ normalized sample points. We isolated the shared control override events, fed their contexts into the pipeline (cf Fig. 1), and directly overlaid the predicted output onto the raw human profile. Fidelity was quantified through three complementary metrics:

$$
\mathrm { M e a n \ A b s o l u t e \ E r r o r \ ( M A E ) } = \frac { 1 } { L } \sum _ { l = 1 } ^ { L } \left| a _ { i , l } - \hat { a } _ { i , l } \right|\tag{2}
$$

MAE measures the average point-wise longitudinal acceleration error $\mathrm { ( m / s ^ { 2 } ) }$ . Dynamic Time Warping (DTW) distance [24] quantifies shape similarity under temporal misalignment by accumulating point-wise acceleration differences along the normalized profiles $( \mathrm { m } / \mathrm { s } ^ { 2 } , )$ . Pearson Correlation Coefficient $r ( \mathbf { a } _ { i } , \hat { \mathbf { a } } _ { i } )$ [25] is unitless and evaluates temporal trend alignment between the two profiles.

Integrating the output of the classifier with the conditional residual regressor yields a continuous profile parameterization. The Context-Conditioned 1D-CNN Decoder improved over the centroid-only baseline, reducing Mean Squared Error (MSE) from 0.001308 $\mathrm { { m } / \mathrm { { s } ^ { 2 } } }$ to 0.001088 $\mathrm { m / s ^ { \bar { 2 } } }$ and MAE from 0.02522 $\mathrm { { m } / \mathrm { { s } ^ { 2 } } }$ to 0.02398 $\mathrm { { m } / \mathrm { { s } ^ { 2 } } }$

![](images/eb549ceae36094ffc05940ef4f0b5b2fbda8234d689c2b684c9a4ef042ddad7f.jpg)  
Fig. 6. Context-matched comparison of generic ACC, shared control override, and predicted acceleration profiles with corresponding kinematic metrics.

We extract three diverse pairs of events with similar driving contexts from forced ACC events and shared control override events. As shown in Fig. 6, the first row shows the contextual similarity between each pair. The second row reveals the central result: given an ACC baseline context, the predicted profile clearly departs from the conservative, delayed ACC acceleration and converges toward the shared control override profile observed in the matching context. This indicates that the CoP-ACC framework can produce, from a context where the driver had no control, an acceleration profile close to what the driver would have naturally demanded. The third row further shows that the predicted profiles achieve nearly identical AUC to the human override while generating significantly lower jerk, indicating a smoother execution of the same driving intent.

For the shared control override test subset, the integrated pipeline reconstructed the driver’s acceleration trend with an MAE of $0 . 0 3 0 6 ~ \mathrm { { m / s } ^ { 2 } }$ , an average DTW Distance of 2.00, and an average Pearson Correlation Coefficient of $r = 0 . 8 0 4$ These metrics indicate that the predicted acceleration remains close to the human pedal input while preserving the main temporal shape of the driver’s intended response.

## D. Distributional Validation of Personalization

While pairwise fidelity evaluates isolated events, a secondary validation analyzes the resulting kinematic distribution on withheld forced-ACC contexts to confirm system generalization. To achieve this, we structured an independent evaluation comparing three distinct groups of acceleration profiles: Actual Overrides, i.e., the raw profiles physically executed by the human driver during shared control override events; Machine Baseline, i.e., the raw profiles executed by the generic ACC during “potential override” events extracted from Trip 1 (where the driver was forbidden from overriding), restricted to contexts that mathematically match the thresholds of actual override events; and Predicted, i.e., the synthetic profiles generated by our framework when conditioned on the “potential override” contexts of Machine Baseline.

To quantify system performance, we extracted core integral and differential kinematic features from each trajectory: Area Under the Curve (AUC) of Total Velocity Gained, Peak Longitudinal Acceleration, and Maximum Jerk.

Applying the continuous prediction pipeline to all “potential override” events illustrates the main advantage of our system: it shifts the $\mathrm { c a r } ^ { \prime } \mathrm { s }$ acceleration toward the driver’s observed preference while keeping the ride smooth. As demonstrated in the distributional boxplots of Fig. 7, evaluating the generated profiles column-by-column reveals the mechanism of this personalization.

![](images/6fe874185bb2f72794b18aabb2ab821452373dd99d6ce59ab0b68a4eb055276d.jpg)  
Fig. 7. Distributional comparison of kinematic features (AUC, Peak Acceleration, Maximum Jerk) across machine baseline, actual overrides, and predicted profiles.

First, examining the AUC of Total Velocity Gained reveals that the predicted profiles capture the driver’s macroobjective. The predictive framework elevates the AUC distribution out of the congested, slow baseline of the generic ACC to closely match the broader, higher-velocity distribution of the human expectation, indicating that the identified model initiates the early acceleration demanded by the driver.

Second, examining the Peak Acceleration and Maximum Jerk reveals a comfort-oriented effect. The predicted profiles exhibit a lower overall peak acceleration compared to the human override. This occurs because the predictive framework anticipates the maneuver and initiates acceleration earlier than the generic ACC (as shown in AUC), reducing the need for the late, hard acceleration that the driver executes during intervention. Importantly, the predicted profiles maintain a notably higher average peak acceleration than the sluggish machine baseline, indicating that the system preserves the required agility. Finally, the predicted profiles achieve a maximum jerk distribution that is lower than both the human override and the baseline ACC profiles, suggesting a smoother driving experience.

## VI. CONCLUSION AND FUTURE WORK

This work demonstrates the feasibility of reframing manual pedal interventions as shared-control override signals, enabling a human-in-the-loop learning paradigm for personalized ACC adaptation. The proposed CoP-ACC hybrid framework extracts distinct acceleration profiles reflecting driver urgency and maps pre-maneuver contexts to smooth, personalized acceleration profiles through a Random Forest context classifier coupled with a 1D-CNN Decoder. Counterfactual evaluation indicates the intended behavioral shift: when conditioned on baseline ACC contexts associated with potential intervention, the predicted profiles tend toward the acceleration patterns observed during real override events.

All data were collected from production vehicles operating on public roads, grounding the results in real-world driving behavior. Despite being trained on a constrained single-driver dataset (N = 140), CoP-ACC achieves low reconstruction error, strong temporal correlation with actual driver inputs, and distributional alignment with human override behavior. These results indicate that meaningful behavioral adaptation can emerge even at limited scale. Nevertheless, generalization across heterogeneous populations remains to be validated. Current feature set excludes road geometry (e.g., curvature), and the dataset required augmentation to stabilize learning, both aspects motivating broader real-world data collection.

A closed-loop simulator study has been initiated to evaluate behavioral adaptation under interactive conditions. Future work will focus on translating predicted acceleration profiles into adaptive ACC parameters (e.g., time gap, response dynamics) using learning-based optimization approaches, while preserving the existing ACC architecture and safety barriers. Scalable deployment will also require automatic event detection, and validating CoP-ACC with diverse drivers in instrumented vehicles. The ultimate objective is anticipatory ACC behavior that reduces manual overrides while preserving driver comfort and intent alignment.

## ACKNOWLEDGMENTS

This work was supported in part by Renault Group and ANRT through a CIFRE (Convention Industrielle de Formation par la Recherche) and in part by the French Government, through the CPER RITMEA, Hauts-de-France Region. This work has been also partially supported by ROBOTEX 2.0, funded by the French program France 2030.

## REFERENCES

[1] F. M. Ali and N. H. Abbas, “Adaptive Cruise Control System: A Literature Survey,” Journal of Engineering, vol. 30, no. 9, pp. 239– 272, 2024.

[2] Y. Wang, Z. Wang, K. Han, P. Tiwari, and D. B. Work, “Personalized Adaptive Cruise Control via Gaussian Process Regression,” in IEEE International Intelligent Transportation Systems Conference (ITSC), 2021, pp. 1496–1502.

[3] Z. Ma and Y. Zhang, “Drivers trust, acceptance, and takeover behaviors in fully automated vehicles: Effects of automated driving styles and driver’s driving styles,” Accident Analysis & Prevention, vol. 159, p. 106238, 2021.

[4] M. Hasenjager, M. Heckmann, and H. Wersing, “A Survey of Personal-¨ ization for Advanced Driver Assistance Systems,” IEEE Transactions on Intelligent Vehicles, vol. 5, no. 2, pp. 335–344, 2020.

[5] X. Liao, Z. Zhao, M. J. Barth, A. Abdelraouf, R. Gupta, K. Han, J. Ma, and G. Wu, “A Review of Personalization in Driving Behavior: Dataset, Modeling, and Validation,” IEEE Transactions on Intelligent Vehicles, vol. 10, no. 2, pp. 1241–1262, Feb. 2025.

[6] Z. Zhao, X. Liao, A. Abdelraouf, K. Han, R. Gupta, M. J. Barth, and G. Wu, “Real-Time Learning of Driving Gap Preference for Personalized Adaptive Cruise Control,” in IEEE International Conference on Systems, Man, and Cybernetics (SMC), Oct. 2023, pp. 4675–4682, iSSN: 2577-1655.

[7] B. Gao, K. Cai, T. Qu, Y. Hu, and H. Chen, “Personalized Adaptive Cruise Control Based on Online Driving Style Recognition Technology and Model Predictive Control,” IEEE Transactions on Vehicular Technology, vol. 69, no. 11, pp. 12 482–12 496, 2020.

[8] S. Sheng, E. Pakdamanian, K. Han, Z. Wang, and L. Feng, “A Study on Learning and Simulating Personalized Car-Following Driving Style,” in IEEE 25th International Conference on Intelligent Transportation Systems (ITSC), 2022, pp. 1208–1215.

[9] Z. Zhao, Z. Wang, K. Han, R. Gupta, P. Tiwari, G. Wu, and M. J. Barth, “Personalized Car Following for Autonomous Driving with Inverse Reinforcement Learning,” in International Conference on Robotics and Automation (ICRA), 2022, pp. 2891–2897.

[10] Y. Wang, Z. Wang, K. Han, P. Tiwari, and D. B. Work, “Gaussian Process-Based Personalized Adaptive Cruise Control,” IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 11, pp. 21 178–21 189, 2022.

[11] S. Lefevre, A. Carvalho, and F. Borrelli, “A Learning-Based Frame-\` work for Velocity Control in Autonomous Driving,” IEEE Transactions on Automation Science and Engineering, vol. 13, no. 1, pp. 32–42, 2016.

[12] D. Nava, G. Panzani, P. Zampieri, and S. M. Savaresi, “A personalized Adaptive Cruise Control driving style characterization based on a learning approach,” in IEEE Intelligent Transportation Systems Conference (ITSC), 2019, pp. 2901–2906.

[13] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in Proceedings of the 31st International Conference on Neural Information Processing Systems, 2017, pp. 4768–4777.

[14] M. Dozza, R. Schindler, G. Bianchi-Piccinini, and J. Karlsson, “How do drivers overtake cyclists?” Accident Analysis & Prevention, vol. 88, pp. 29–36, 2016.

[15] M. Marcano, F. Tango, J. Sarabia, S. Chiesa, J. Perez, and S. D ´ ´ıaz, “Can Shared Control Improve Overtaking Performance? Combining Human and Automation Strengths for a Safer Maneuver,” Sensors, vol. 22, no. 23, 2022.

[16] M. Marcano, S. D´ıaz, J. Perez, and E. Irigoyen, “A Review of Shared ´ Control for Automated Vehicles: Theory and Applications,” IEEE Transactions on Human-Machine Systems, vol. 50, no. 6, pp. 475– 491, 2020.

[17] L. Yu and R. Wang, “Researches on Adaptive Cruise Control system: A state of the art review,” Proceedings of the Institution of Mechanical Engineers, Part D: Journal of Automobile Engineering, vol. 236, no. 2-3, pp. 211–240, 2022.

[18] G. N. Lance and W. T. Williams, “A General Theory of Classificatory Sorting Strategies: 1. Hierarchical Systems,” The Computer Journal, vol. 9, no. 4, pp. 373–380, 1967.

[19] J. H. Ward Jr., “Hierarchical Grouping to Optimize an Objective Function,” Journal of the American Statistical Association, vol. 58, no. 301, pp. 236–244, 1963.

[20] P. J. Rousseeuw, “Silhouettes: A graphical aid to the interpretation and validation of cluster analysis,” Journal of Computational and Applied Mathematics, vol. 20, pp. 53–65, 1987.

[21] S. Bouhsissin, N. Sael, and F. Benabbou, “Driver Behavior Classification: A Systematic Literature Review,” IEEE Access, vol. 11, pp. 14 128–14 153, 2023.

[22] L. Breiman, “Random Forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.

[23] A. Rasch, C.-N. Boda, P. Thalya, T. Aderum, A. Knauss, and M. Dozza, “How do oncoming traffic and cyclist lane position influence cyclist overtaking by drivers?” Accident Analysis & Prevention, vol. 142, p. 105569, 2020.

[24] H. Sakoe and S. Chiba, “Dynamic programming algorithm optimization for spoken word recognition,” IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 26, no. 1, pp. 43–49, 1978.

[25] J. Benesty, J. Chen, Y. Huang, and I. Cohen, “Pearson Correlation Coefficient,” in Noise Reduction in Speech Processing, 2009, pp. 1–4.