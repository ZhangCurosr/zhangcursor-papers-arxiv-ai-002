# Reconstruction-Based Diagnosis through Gödel-Regularized Virtual Observability

Oliver Niggemann Jonas Ehrhardt Alexander Diedrich

Chair of Computer Science in Mechanical Engineering, Helmut Schmidt University, Hamburg, Germany (e-mail: oliver.niggemann@hsu.hamburg).

## Abstract

Not every component in a technical system can be equipped with sensors. Hence, when a fault occurs, available sensors often detect it only after its efects have propagated. This makes the isolation of faults dificult. We approach this problem by introducing virtual sensors. For each combination of component and fault type, we let external knowledge describe the fault-sensitive variables. We introduce a novel neural observer that is only trained on normal data. For fault isolation, it uses available sensor measurements where it is possible and reconstructs missing variables as virtual sensors to predict normal behavior. It detects deviations between current values and nominal predictions and computes local symptoms, which are then compared by onset time. The result is a set of root causes describing the faulty behavior. We use Gödel logic to encode physical knowledge and guide training. We evaluate our approach on synthetic networks, the Tennessee Eastman Process, and datasets of two modular production plants.

Keywords: Deep learning based FDI, Anomaly detection, Virtual sensors, Neuro-symbolic Artificial Intelligence.

## 1 Introduction

Fault diagnosis supports the safe and reliable operation of technical processes. But not all behaviors within a system can be observed at all times. A fault may hence appear at an unobserved location and its efects then propagate through material, energy, and control connections before a sensor can detect them. Noise and delays can change the order of symptoms. Model-based methods address this with residual and structural models (Isermann, 2005; Krysander et al., 2008), while data-driven methods learn normal or faulty behavior from observations (Qin, 2012). Both must infer the cause from indirect and delayed evidence.

Suppose that every possible root cause had a faultsensitive local sensor that reacts before all propagated symptoms. The first persistent local symptom would then identify the root cause, and fault isolation would reduce to comparing symptom onset times. However, no real-world plants ofer such complete coverage, and learned estimates remain uncertain. In this article we address how machine learning can close the observation gaps without hiding this uncertainty.

We propose Reconstruction-Based Diagnosis through Virtual Observability (ReDiVO) which employs virtual sensors to make the whole system observable. We use available a-priori knowledge to determine the components within the system and associate these with known sensor locations. We then identify locations that contain components but have no direct sensors monitoring the specific components. For instance, if some pump performance is not measured, we reconstruct it from related process measurements like downstream flow and control signals. Measured and virtual sensors together create virtual observability, this means that every fault candidate has at least one observed or virtual root-cause variable. We propose a neural observer which compares the current value of each variable with a nominal prediction learned from normal data, both as uncertainty ranges. A persistent separation of these ranges is a local symptom. We compare the onset times of these symptoms and return a set of candidate diagnoses.

We pursue a hybrid approach fusing data-driven modeling of normal behavior with a-priori knowledge. Without reference values, normal data alone may not determine a reconstruction uniquely. A shifted or rescaled estimate of the pump performance, for example, can predict the measured flows equally well. We therefore add physical knowledge about topology, qualitative dependencies, balances, and operating constraints, expressed as rules in Gödel logic. During training, rule violations penalize the reconstructions and nominal predictions. For example, a higher pump performance should not reduce downstream flow when all other influences stay fixed. Such a rule favors plausible estimates without forcing the pump to stay nominal.

We address three research questions. RQ1: Can a virtual sensor reconstruct an unobserved component variable well enough to provide a useful local symptom? RQ2: Which types of prior knowledge in Gödel logic improve this reconstruction? RQ3: When does the first local symptom on a common clock isolate the root cause?

We make four contributions. First, we formulate diagnosis under incomplete observability as virtual observability of component-fault candidates. Second, we develop a neural observer that learns reconstruction and open-loop prediction from normal data only. Third, we introduce a training-only regularizer in Gödel logic that encodes topology, qualitative relations, and physical constraints without excluding the fault efects of interest. Fourth, we turn calibrated current and nominal value ranges into persistent symptoms and use their onset intervals to return a unique candidate, an ambiguity group, or no diagnosis. We evaluate ReDiVO on synthetic component networks, the Extended Tennessee Eastman Process (Bathelt et al., 2015; Reinartz et al., 2021), HAI-CPPS, and Twin-Flow (Hranisavljevic et al., 2026b).

## 2 Related Work

Structural diagnosis isolates faults through overdetermined model relations (Krysander et al., 2008), and diagnosability determines whether observations distinguish a fault within bounded time (Sampath et al., 1995). Setbased methods compare measurements with all values allowed by uncertain models (Harirchi and Ozay, 2018). We adopt this range-based view for learned virtual sensors and symptom onset times, but use empirical bounds from normal data that do not need to hold under faults. On the data-driven side, subsystem-wise anomaly detectors can feed consistency-based diagnosis (Steude et al., 2024), and causal methods reason over monitored variables with anomaly times or Granger causality (Assaad et al., 2023; Han et al., 2025). We address the complementary case in which a fault-relevant variable is not monitored and must first be reconstructed as a virtual sensor.

Learned observers estimate unmeasured variables from measured histories (Revach et al., 2022), and trained networks can reconstruct missing inputs by optimization (Ludwig et al., 2025). SMT reasoning over critical pairs infers measurements that make faults distinguishable, but requires an SMT system model (Diedrich et al., 2024a). These methods neither assign variables to componentfault candidates nor compare symptom onset times. To guide learning, diferentiable fuzzy logic turns symbolic knowledge into training losses (Badreddine et al., 2022; van Krieken et al., 2022), also for fault diagnosis (Hogea et al., 2024), and system descriptions for diagnosis can be learned from data (Diedrich et al., 2024b; Moddemann et al., 2024). We combine both research directions: apriori system knowledge constrains reconstructions and nominal predictions during training but is not used as online evidence.

## 3 Reconstruction-Based Diagnosis through Virtual Observability

## 3.1 Diagnosis Problem and Required Conditions

Let C be the finite set of physical components. For each component $c \in { \mathcal { C } }$ , the finite set $\boldsymbol { A } _ { c }$ contains the considered fault types. A diagnosis candidate $r = \langle c , a \rangle$ is a hypothesis about the root cause of a fault: component c has fault type a. The candidate set $\mathcal { R }$ contains all hypotheses that diagnosis distinguishes:

$$
\mathcal { R } = \{ \langle c , a \rangle \ : | \ : c \in \mathcal { C } , a \in \mathcal { A } _ { c } \} .\tag{1}
$$

Each process variable has an index $j ,$ and $v _ { j } ( t )$ denotes its value at sample t. A variable is observed if a sensor measures it online and unobserved otherwise. Before training, domain experts or plant documentation provide a-priori system knowledge about components, faultsensitive variables, and reconstruction dependencies. For each diagnosis candidate $r \in \mathcal { R }$ , this knowledge specifies a nonempty set $B _ { r }$ of fault-sensitive root-cause variables. Each of them is observed or reconstructed as a virtual sensor, that is, a learned estimate of an unobserved variable. A local symptom of r is a persistent deviation in a variable of $B _ { r }$ . ReDiVO achieves virtual observability if every candidate has at least one observed or reconstructed root-cause variable.

Learning uses only normal time series recorded during fault-free operation. A-priori knowledge identifies the predictor variables used as inputs for reconstruction. Ofline reference values, where available, supervise reconstruction during training but are not online inputs. For example, a temporary sensor may record the pump speed during training-data collection even if that measurement is unavailable during deployment. Online, ReDiVO receives the measurements and those commands or setpoints that do not reveal the fault. It returns either one candidate, an ambiguity group (Section 3.4), or no diagnosis.

## 3.2 Learning Virtual Observability

ReDiVO compares two estimates of all process variables $\left( \mathrm { F i g . ~ 1 } \right)$ . The current system trajectory $\widehat { \mathbf { v } } ^ { \mathrm { c u r r e n t } } ( t )$ estimates what the plant does at sample t. It takes the measurement of observed variables (Case 1) and reconstructs unobserved variables (Case 2). The learned nominal trajectory $\widehat { \mathbf { v } } ^ { \mathrm { n o m i n a l } } ( t \mid t - \delta t )$ estimates what the plant does under normal behavior. It starts $\delta t$ samples earlier and runs a long-short-term memory (LSTM) network on its own predictions, without new measurements. A fault can separate both trajectories (Section 3.4).

For the current system trajectory, each unobserved variable j is reconstructed from predictors identified by the a-priori system knowledge (Case 2 in Fig. 1). Predictors may include internal process quantities. Their values form the input vector $\dot { \mathbf { v } } _ {  j } ^ { \mathrm { i n p u t } } ( t )$ . An observed predictor k contributes its measurement, and an unobserved one its one-step prediction $\widetilde { v _ { k } } \left( t \right)$ , which uses information only up to $t - 1$ . The target j is never its own predictor. Both choices prevent circular reconstruction. The current system trajectory is

$$
\begin{array} { r l } & { \quad ( \mathbf { h } _ { t } , \mathbf { c } _ { t } ) = \mathrm { L S T M } _ { \theta } \big ( \widehat { \mathbf { v } } ^ { \mathrm { c u r r e n t } } ( t - 1 ) , \mathbf { h } _ { t - 1 } , \mathbf { c } _ { t - 1 } \big ) , } \\ & { \qquad \widehat { v _ { j } } ^ { - } ( t ) = d _ { \theta , j } ( \mathbf { h } _ { t } ) , } \\ & { \widehat { v } _ { j } ^ { \mathrm { c u r r e n t } } ( t ) = v _ { j } ^ { \mathrm { m e a s } } ( t ) , \quad \mathrm { i f ~ } j \mathrm { ~ i s ~ o b s e r v e d , } } \\ & { \widehat { v } _ { j } ^ { \mathrm { c u r r e n t } } ( t ) = a _ { \theta , j } \big ( \mathbf { v } _ {  j } ^ { \mathrm { i n p u t } } ( t ) \big ) } \\ & { \qquad + n _ { \theta , j } \big ( \mathbf { v } _ {  j } ^ { \mathrm { i n p u t } } ( t ) , \widehat { v _ { j } } ^ { - } ( t ) , \mathbf { h } _ { t } \big ) , \quad \mathrm { o t h e r w i s e . } } \end{array}\tag{2}
$$

We chose an LSTM for processing windows of a fixed history length. Its hidden state $\mathbf { h } _ { t }$ and memory state $\mathbf { c } _ { t }$ summarize the past complete vectors, which contain measurements and reconstructions. Output head $d _ { \theta , j }$ turns this summary into the one-step prediction of variable $j .$

![](images/1082782ae1e031c0e4010ea7c36d8046ef76cefa3606745e953ab44ca2214061.jpg)  
Figure 1: Neural observer and diagnosis. The current trajectory uses measurements where available and virtual sensors otherwise. The nominal trajectory rolls the same network forward from $t - \delta t$ without measurements. Both losses apply only during training on normal data.

An unobserved variable is the sum of an afine estimate $a _ { \theta , j } ( \mathbf { x } ) = \mathbf { w } _ { j } ^ { \top } \mathbf { x } + \gamma _ { j }$ and a nonlinear correction by the one-hidden-layer network $n _ { \theta , j }$ . All trainable weights form θ.

The learned nominal trajectory starts δt samples before the current time t and predicts the entire interval up to t using the learned normal model, without corrections from process measurements. This open-loop reuses the same LSTM and heads:

$$
\begin{array} { r } { \widehat { \mathbf { v } } ^ { \mathrm { n o m i n a l } } ( t \mid t - \delta t ) = D _ { \theta } ^ { ( \delta t ) } \left( \widehat { \mathbf { v } } ^ { \mathrm { c u r r e n t } } ( t - \delta t ) , \right. \qquad } \\ { \left. \qquad \mathbf { h } _ { t - \delta t } , \mathbf { c } _ { t - \delta t } , \mathbf { u } _ { t - \delta t + 1 : t } \right) . } \end{array}\tag{3}
$$

Operator $D _ { \theta } ^ { ( \delta t ) }$ applies the model δt times to its own outputs. Commands or setpoints $\mathbf { u } _ { t - \delta t + 1 : t }$ may enter if they do not reveal the fault. The start at $t - \delta t$ can already be fault-afected, so the nominal trajectory is no guaranteed fault-free reference. The diagnosis horizon δt can difer from the history length and the training horizon.

The network is trained on normal data with

$$
{ \mathcal { L } } _ { \mathrm { b a s e } } = \lambda _ { \mathrm { c u r r e n t } } { \mathcal { L } } _ { \mathrm { c u r r e n t } } + \lambda _ { \mathrm { n o m i n a l } } { \mathcal { L } } _ { \mathrm { n o m i n a l } } .\tag{4}
$$

$\mathcal { L } _ { \mathrm { c u r r e n t } }$ compares reconstructions with ofline targets and is dropped if none exist. $\mathcal { L } _ { \mathrm { n o m i n a l } }$ compares the rollout with normal data. In recursive training, the model feeds its predictions back as inputs and learns from errors across several steps. The loss covers every step of a fixed training horizon, and gradients pass through all steps. One-step training is a faster approximation that uses only the next prediction. Diagnosis always uses the full rollout over δt. The weights $\lambda _ { \mathrm { c u r r e n t } } , \lambda _ { \mathrm { n o m i n a l } } \geq 0$ balance both terms. After training, the weights are frozen.

## 3.3 Using A Priori Knowledge to Improve Virtual Observability

Several behavior models may fit normal data equally well yet disagree outside the observed operating range. This ambiguity is known as underspecification (D’Amour et al., 2022). For a virtual pump-speed sensor, matching measured flows alone may not determine the physical speed uniquely. Supplied physical and system knowledge adds an inductive bias: training favors reconstructions consistent with known plant relationships. Appropriate knowledge can reduce the training data required and improve extrapolation to operating conditions not covered by those data (Oneto et al., 2025). These benefits depend on the rules and are not guaranteed.

Table 1 lists the main types of knowledge. The knowledge is compiled as rules over predicates, that is, elementary relations between process variables such as “the tank level rises”. During training, the selected rules $\kappa _ { \mathrm { t r a i n } }$ are evaluated on both trajectories.Gödel logic assigns each predicate p a truth degree between zero and one. Conjunction takes the minimum, disjunction the maximum, and implication follows Gödel semantics:

$$
\begin{array} { c } { { \tau ( p ) = \left( 1 + \rho _ { p } / \sigma _ { p } \right) ^ { - 1 } , } } \\ { { \tau ( p \Rightarrow _ { G } q ) = \displaystyle \left\{ 1 , \begin{array} { l l } { { \begin{array} { r l r } { { \tau ( p ) \leq \tau ( q ) , } } & { } \\ { { \tau ( q ) , } } & { { \mathrm { o t h e r w i s e . } } } \end{array} } } \end{array} \right. } } \end{array}\tag{5}
$$

Here, $\rho _ { p } \geq 0$ measures the violation of $p ,$ in physical units for measurement equations, and $\sigma _ { p } > 0$ is its engineering tolerance.

Each rule is evaluated separately at each sample of a training sequence. These rule evaluations enter training as a penalty:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { b a s e } } + \beta \sum _ { g \in \mathcal { G } } \bar { \alpha } _ { g } \frac { 1 } { | \mathcal { K } _ { g } | } \sum _ { k \in \mathcal { K } _ { g } } w _ { k } \bar { \ell } _ { k } .\tag{6}
$$

For rule $k , \mathit { \bar { \ell } } _ { k }$ is the mean shortfall max $\{ 0 , m - \tau ( k ) \}$ below a required truth degree $m \in ( 0 , 1 ]$ over all usable rule evaluations, including satisfied ones. Set G contains the rule groups with usable rules, and $\mathcal { K } _ { g } \subseteq \mathcal { K } _ { \mathrm { t r a i n } }$ the rules of group g. Averaging within each group keeps groups with many rules from dominating. The weights $\beta \ge 0 , \bar { \alpha } _ { g }$ , and $w _ { k } > 0$ are fixed in advance. With $\beta = 0$ Eq. (6) reduces to Eq. (4).

A rule evaluation is usable if the required history and operating conditions are available at that sample. Other evaluations are left out rather than counted as satisfied.

Table 1: A priori knowledge for training, with process-industry examples. Experiments use the applicable rule groups.
<table><tr><td>Prior knowledge</td><td>Example relation</td></tr><tr><td>Topology</td><td>Connected variables have compatible normal reconstruction errors.</td></tr><tr><td>Qualitative relations</td><td>Variables change in the same or opposite directions under specified conditions.</td></tr><tr><td>Temporal relations</td><td>Responses follow within a specified delay.</td></tr><tr><td>Physical flow</td><td>Signed flows balance and mass flow agrees with volume flow through density.</td></tr><tr><td>Sensor physics</td><td>Modes and state dynamics Modes are exclusive. Filling implies a nondecreasing tank level. Measurement equations relate states and outputs, such as level and pressure.</td></tr></table>

## 3.4 Time-Synchronous Virtual-Observability Diagnosis

ReDiVO compares local symptom-onset times to identify the root cause. After training and calibration, Algorithm 1 applies three diagnosis steps at scheduled evaluation times.

1. Detect anomalies by comparing value ranges (Algorithm 1, line 4). To account for noise and estimation errors, each measured or reconstructed process value and its nominal prediction receive uncertainty ranges:

$$
\begin{array} { r l } & { C _ { j } ( t ) = \big [ \widehat { v } _ { j } ^ { \mathrm { c u r r e n t } } ( t ) \pm E _ { j } ^ { \mathrm { C } } \big ] , } \\ & { N _ { j } ( t ) = \big [ \widehat { v } _ { j } ^ { \mathrm { n o m i n a l } } ( t \mid t - \delta t ) \pm E _ { j } ^ { \mathrm { N } } \big ] , } \end{array}\tag{7}
$$

with $[ x \pm E ] = [ x - E , x + E ]$ . Bound $E _ { i } ^ { \mathrm { C } } \geq 0$ covers measurement or reconstruction error, and $\check { E } _ { j } ^ { \mathrm { N } } \geq 0$ covers prediction error over $\delta t .$ Overlapping ranges give no evidence of a fault. Separated ranges give the normalized gap

$$
U _ { j } ( t ) = \frac { 1 } { b _ { j } } \operatorname* { m a x } \{ 0 , \underline { { C } } _ { j } ( t ) - \overline { { N } } _ { j } ( t ) , \underline { { N } } _ { j } ( t ) - \overline { { C } } _ { j } ( t ) \} ,\tag{8}
$$

where underlines and overlines denote lower and upper limits. The scale $b _ { j } > 0$ makes gaps comparable across variables.

2. Confirm persistent symptoms. Single exceedances can be outliers. Variable $j \in B _ { r }$ therefore shows a symptom of candidate r only if $U _ { j } ( t )$ exceeds the threshold $\eta _ { r , j } > 0$ at P consecutive evaluation times (Algorithm 1, line 6). The timestamp of the first one is the onset time $\widehat { t } _ { r , j }$ , and the Pth confirms the symp tom. A candidate with at least one confirmed symptom is symptomatic.

3. Compare symptom-onset times (Algorithm 1, lines 7–8). Each onset is evaluated on the range $[ \widehat { t } _ { r , j } \pm$ $\left. { \zeta _ { r , j } } \right]$ , with $\zeta _ { r , j } ~ \geq ~ 0$ in physical time. The candidate interval $\mathcal { T } _ { r } = [ t _ { r } ^ { - } , t _ { r } ^ { + } ]$ takes the smallest lower and the smallest upper limit over the confirmed symptoms in $B _ { r }$ . A non-symptomatic candidate q receives $t _ { q } ^ { - } = + \infty$ for the comparison only, which is no evidence of health. Candidate r is the unique diagnosis if it is certainly earlier than every competitor:

$$
t _ { r } ^ { + } < t _ { q } ^ { - } \qquad \mathrm { f o r ~ e v e r y ~ } q \in \mathcal { R } \setminus \{ r \} .\tag{9}
$$

Otherwise, if symptoms exist, the result is an ambiguity group. It contains every symptomatic candidate whose lower limit does not exceed the smallest upper limit, including touching intervals. For a unique diagnosis, the same rule yields only r. The Potentially correct metric in Section 4 uses this group. Without symptoms, ReDiVO returns no diagnosis.

Algorithm 1 ReDiVO training and diagnosis   
Require: Normal training, validation, and calibration data,   
predictors, rules $\kappa _ { \mathrm { t r a i n } } ,$ root-cause variables $B _ { r } ,$ horizons   
D, persistence P   
Ensure: Unique candidate, ambiguity group, or no diagnosis   
Training on normal data   
1: Train θ with $\operatorname { E q . }$ (6) and select the epoch   
2: Freeze $\theta ,$ select $\delta t \in { \mathcal { D } } ,$ , calibrate bounds and thresholds   
Diagnosis of one episode   
3: for each evaluation time t do   
4: Compute both trajectories and anomaly gaps $U _ { j } ( t )$   
with Eqs. (2)–(3), (7)–(8) (Step 1)   
5: end for   
6: Confirm persistent symptoms and record onset times   
(Step 2)   
7: Form candidate onset-time intervals I (Step 3)   
8: return r satisfying Eq. (9), else the ambiguity group, or   
no diagnosis without symptomatic candidates (Step 3)

## 4 Evaluation

Virtual sensor reconstruction, the efect of prior knowledge, and onset-based fault isolation are evaluated on artificial data and three diagnosis datasets: the Extended Tennessee Eastman Process (TEP) (Bathelt et al., 2015; Reinartz et al., 2021), HAI-CPPS v2.1, and the TwinFlow dataset Hranisavljevic et al. (2026b,a).

Under the selection procedure of Section 3.2, reconstruction loss on normal validation data selects the training epoch, with prediction loss resolving ties. After freezing the weights, the diagnosis horizon minimizes the mean absolute current–nominal diference at the forecast endpoint over the predefined horizons. Separate normal recordings calibrate error bounds and thresholds. All choices are fixed before fault testing. The HAI study uses the modified procedure described in Section 4.3.

Correct counts true-cause inclusion in the earliest persistent symptom-time group, including ties. Potentially correct counts use Section 3.4’s possible-cause group. Uniquely correct counts require only the true cause in that group. Hit@3 counts inclusion among the first three active candidates. Cases without symptoms count as failures. Alphabetical ties convey no fault-type evidence. Random references select from all candidates while matching the method’s output sizes, including empty outputs.

Table 2: Linear networks, eight seeds. Setup: component count or combined stress at 30 components. Mask: hidden outputs per component. Full: value, direction, and delay rules. Rates in percent over 192 healthy and 240 fault recordings per condition (960 faults at 120 components).
<table><tr><td colspan="5">Unique Healthy</td></tr><tr><td>Setup</td><td></td><td>Rules Mask diagnosis</td><td></td><td>Detection alarms</td></tr><tr><td>30</td><td>None</td><td>2</td><td>99.17</td><td>100.00 13.02</td></tr><tr><td>30</td><td>Full</td><td>2</td><td>100.00 100.00</td><td>11.46</td></tr><tr><td>30</td><td>None</td><td>4</td><td>99.58 100.00 100.00</td><td>9.38</td></tr><tr><td>30</td><td>Full</td><td>4</td><td>100.00</td><td>11.46</td></tr><tr><td>120</td><td>None</td><td>2</td><td>96.77 97.71</td><td>100.00 43.75</td></tr><tr><td>120</td><td>Full</td><td>2</td><td>100.00</td><td>38.54</td></tr><tr><td>Stress None</td><td></td><td>4</td><td>47.08</td><td>55.00 7.81</td></tr><tr><td>Stress Value</td><td></td><td>4 65.83</td><td>72.50</td><td>6.25</td></tr><tr><td>Stress Full</td><td></td><td>4 59.58</td><td>65.00</td><td>7.81</td></tr></table>

In the result tables, detection counts fault recordings with at least one persistent diagnostic symptom, whereas healthy alarms counts fault-free recordings with such a symptom. Detection alone does not establish the right cause or an alarm after the fault begins. Reconstruction errors use the targets and units stated in each caption.

## 4.1 Artificial Data

Two artificial studies test reconstruction, masking, and prior knowledge, without claiming industrial realism. Their results remain separate.

## 4.1.1 Linear Component Networks

Each component has two hidden states, six redundant outputs, linear discrete-time dynamics, and delayed connections. With four outputs hidden, the two remaining independent measurements still constrain both states. Eight graph seeds and 24 conditions yield 912 models, with separate normal data for training, selection, and calibration.

Isolation remains high under masking and scaling (Table 2), but healthy alarms increase with system size. Afine reconstruction is more accurate on these linear data.

Under combined masking, noise, delays, coupling, and weak faults, value rules improve isolation from 47.08% to 65.83%. The 95% gain interval is 13.75–23.33 percentage points from 20,000 paired seed-bootstrap resamples, but the Holm-adjusted probability is 0.703 across 90 comparisons. Only 1.60% of faulty time points have confirmed symptoms, distinguishing recording-level diagnosis from continuous detection.

## 4.1.2 Random Hybrid Automata

Eight randomly parameterized six-component systems switch between three modes with afine or nonlinear ordinary diferential equations (ODEs). Both internal states remain hidden, with two or four of six measurements masked. Actuator-efectiveness loss and additional outflow occur as steps or ramps near mode switches or in locally stationary intervals.

Table 3: Hybrid automata. Rates in percent. Mask: hidden measurements per component. Both states are reconstructed. Rules afect training only.
<table><tr><td>Rules</td><td>Unique Mask diagnosis</td><td>Detection</td><td>Healthy alarms</td></tr><tr><td>None</td><td>2 72.40</td><td>83.20</td><td>11.72</td></tr><tr><td>Structure</td><td>2 70.96</td><td>82.16</td><td>13.67</td></tr><tr><td>Mode</td><td>2 52.73</td><td>61.33</td><td>12.50</td></tr><tr><td>Both</td><td>2 56.51</td><td>64.97</td><td>10.94</td></tr><tr><td>None</td><td>4 64.97</td><td>82.29</td><td>13.28</td></tr><tr><td>Structure</td><td>4 62.24</td><td>80.99</td><td>10.55</td></tr><tr><td>Mode</td><td>4 68.36</td><td>84.77</td><td>18.36</td></tr><tr><td>Both</td><td>4 63.54</td><td>79.69</td><td>12.89</td></tr></table>

Per system, 24 normal recordings train the models, eight select them, and 64 calibrate them. Tests use 16 healthy recordings and 48 fault cases, each lasting 200 seconds at 0.2-second sampling. Two model initializations are averaged within each independent system.

Structure rules align reconstruction errors across connections. Mode rules encode mode-dependent measurement equations and mode exclusivity. Each group retains weight 0.5 when used alone or together. Without rules, virtual sensors support correct isolation in 72.40% and 64.97% of cases under the two masks (Table 3).

On healthy test recordings with four measurements masked, combined rules reduce current-state mean absolute error (MAE) from 0.003910 to 0.003345, but increase nominal-forecast MAE from 0.032515 to 0.045318, in normalized ODE-state units. The diagnosis change is −1.43 percentage points, with a descriptive 95% interval of [−8.72, 8.33] from 5,000 paired system-bootstrap resamples stratified by dynamics family. Structure alone does not improve reconstruction. Mode-dependent ridge reconstruction is more accurate, but no ridge diagnosis pipeline was tested.

## 4.2 Extended TEP

Extended TEP (Bathelt et al., 2015; Reinartz et al., 2021) provides 412 fault cases from four modes. The 24 runs combine four rule settings, two masks, and three seeds. Windows span 44 samples.

Table 4 excludes 32 cases post hoc: condenser-valve sticking (IDV15) and stripper steam-transfer variation (IDV16), whose sensors never activate persistently for unresolved reasons. All 26 candidates remain. Including those cases gives 59.71% Correct, 74.92% Potential, and 64.40% Hit@3 without rules or masking.

Masking hides recycle flow, stripper underflow, and condenser cooling flow. Earliest groups average 2.04– 2.25 candidates without rules, possible groups 8.01–8.05. Inclusion does not imply unique isolation.

Full rules change unmasked Correct by −0.35 percentage points, with a 95% interval of [−1.58, 0.96] from 2,000 paired case-bootstrap resamples without multiplicity correction. Six qualitative runs failed gradient verification, and their replay remained incomplete. Tests use known fault windows, not continuous monitoring.

Table 4: TEP means (%), three model seeds and 380 fault cases. Full combines topology and qualitative rules. Mask counts additionally hidden channels. Correct allows ties in the earliest symptom group and is not unique isolation. Detection counts a persistent symptom anywhere in a fault window. Healthy-recording alarm rates were not evaluated (n.e.). Random matches the no-rule output sizes and has no detector.
<table><tr><td colspan="6"></td><td rowspan="2">Healthy alarms</td></tr><tr><td>Gödel rules</td><td>Mask</td><td>Correct</td><td>Potential</td><td>Hit@3</td><td>Detection</td></tr><tr><td>None</td><td>0</td><td>64.74</td><td>81.23</td><td>69.82</td><td>93.86</td><td>n.e.</td></tr><tr><td>Full</td><td>0</td><td>64.39</td><td>79.65</td><td>67.72</td><td>93.68</td><td>n.e.</td></tr><tr><td>Random</td><td>0</td><td>7.86</td><td>30.79</td><td>10.20</td><td></td><td></td></tr><tr><td>None</td><td>3</td><td>66.84</td><td>80.88</td><td>69.65</td><td>93.77</td><td>n.e.</td></tr><tr><td>Full</td><td>3</td><td>66.32</td><td>79.82</td><td>68.86</td><td>93.33</td><td>n.e.</td></tr><tr><td>Random</td><td>3</td><td>8.65</td><td>30.94</td><td>10.30</td><td></td><td>一</td></tr></table>

Table 5: Results for ReDiVO on the HAI-CPPS v2.1 benchmark. All rows use Gödel regularization $( \beta = 0 . 0 1 )$ , state specific calibration, and separate regression heads. Healthy alarms count one healthy episode. Reconstruction and forecast MAE use normal-validation pump speeds in pump rpm, weighting all pumps and operating phases equally.
<table><tr><td>Dataset</td><td>Training</td><td>Correct pump</td><td>Detection</td><td>Healthy alarms</td><td>Reconstruction MAE</td><td>Forecast MAE</td></tr><tr><td>ds5</td><td>Recursive</td><td>4/4</td><td>4/4</td><td>0/1</td><td>7.644</td><td>9.745</td></tr><tr><td>ds5</td><td>One-step</td><td>4/4</td><td>4/4</td><td>0/1</td><td>9.314</td><td>12.728</td></tr><tr><td>ds6</td><td>Recursive</td><td>6/6</td><td>6/6</td><td>0/1</td><td>6.160</td><td>9.859</td></tr><tr><td>ds6</td><td>One-step</td><td>6/6</td><td>6/6</td><td>0/1</td><td>9.811</td><td>16.307</td></tr><tr><td>ds7</td><td>Recursive</td><td>3/4</td><td>3/4</td><td>0/1</td><td>7.440</td><td>10.784</td></tr><tr><td>ds7</td><td>One-step</td><td>2/4</td><td>4/4</td><td>1/1</td><td>8.543</td><td>22.046</td></tr><tr><td>ds8</td><td>Recursive</td><td>6/6</td><td>6/6</td><td>1/1</td><td>6.597</td><td>10.412</td></tr><tr><td>ds8</td><td>One-step</td><td>5/6</td><td>5/6</td><td>0/1</td><td>9.155</td><td>18.008</td></tr><tr><td>ds9</td><td>Recursive</td><td>8/8</td><td>8/8</td><td>1/1</td><td>5.459</td><td>9.653</td></tr><tr><td>ds9</td><td>One-step</td><td>6/8</td><td>8/8</td><td>1/1</td><td>8.650</td><td>18.033</td></tr><tr><td>ds10</td><td>Recursive</td><td>2/2</td><td>2/2</td><td>0/1</td><td>5.725</td><td>10.178</td></tr><tr><td>ds10</td><td>One-step</td><td>2/2</td><td>2/2</td><td>0/1</td><td>7.502</td><td>13.374</td></tr><tr><td>All</td><td>Recursive</td><td>29/30</td><td>29/30</td><td>2/6</td><td>6.287</td><td>10.066</td></tr><tr><td>All</td><td>One-step</td><td>25/30</td><td>29/30</td><td>2/6</td><td>8.735</td><td>16.562</td></tr></table>

## 4.3 HAI-CPPS Version 2.1

The HAI-CPPS dataset is a simulation of a modular industrial process plant. We focus on datasets ds5 to ds10, using all 30 supplied faulty recordings and one healthy test recording per dataset. Specifically, we focus on the pumps within each module, as they provide a good evaluation testbed for ReDiVO’s virtual sensor reconstruction. The model estimates efective pump speeds from other measurements, without observing these speeds directly.

We evaluate ReDiVO with Gödel regularization during training, a state dependent calibration, and additional separate regression heads. The masked pump speed from the dataset provides the ofline targets and validation references.

Diagnosis. Recursive training identifies the correct pump in 29/30 faulty recordings. All 29 decisions identify a unique pump, the remaining recording produces no diagnosis. One-step training identifies the correct pump in 25/30 cases. Both versions raise a pump alarm in 29/30 faulty recordings and give false alarms in two of the six healthy recordings (Table 5).

Prognosis and reconstruction. Recursive training predicts healthy validation behavior more accurately. The forecast MAE falls from 16.562 to 10.066 rpm (39.2%), and reconstruction MAE from 8.735 to 6.287 rpm for the pump speed.

## 4.4 TwinFlow

TwinFlow is a material-handling simulation with conveyors, robots, mobile transport vehicles, and workstations. Faults include motor failures, emergency stops, and conveyor speed-calibration errors (Hranisavljevic et al., 2026b). Here, we focus on conveyor speed-calibration faults, which afect the accuracy of the reported conveyor speed.

Across all 15 speed-calibration faults in the TwinFlow dataset, ReDiVO identifies the correct faulty conveyor in 12 cases (80%), while considering the full set of 1,444 possible component–fault combinations. This means that the diagnosis narrows the problem down to the right conveyor, although several explanations for its abnormal

behavior may remain.

Finding the faulty component and identifying its exact fault type are diferent tasks. ReDiVO determines both uniquely in four of the 15 events, as TwinFlow’s fault types do overlap in the characterization of symptoms. E.g. a speed calibration fault has the same symptoms as a motor fault.

## 4.5 Ablation Studies: Temporal Resolution and Fault Types

Each artificial ablation uses six independent graph seeds.

Temporal resolution and clock ofsets. Sampling every four rather than one second retained every strong fault in the earliest group without rules. Correction learned from normal synchronization pulses restored ordering disrupted by clock ofsets. This tests symptom timestamps, not asynchronous observer inputs or short transients.

Fault types. Strong additive faults were reliably localized. Noise and masking impaired weaker gain and decay faults. Duplicated fault labels remained ambiguous. Sensor-identity, sum, diference, and direction rules improved some reconstructions but suppressed weak symptoms at weight one. Better reconstruction did not ensure better diagnosis.

False alarms limit continuous monitoring. Paired differences are not significant after Holm correction.

## 4.6 Discussion of RQ1–RQ3

RQ1: Can virtual sensors provide useful local symptoms? Yes, under the tested observation conditions. Both artificial studies reconstruct hidden states even when four of six process measurements are unavailable. Isolation remains nearly complete in the linear tests, and the reconstructed states enable useful diagnosis in the hybrid tests. In HAI, virtual pump-speed sensors identify the faulty pump in 29 of 30 fault recordings with recursive training, compared with 25 of 30 with one-step training. Recursive training also improves normal-validation reconstruction and forecasting. Thus, root-cause variables need not all be measured directly when the remaining measurements suficiently constrain them.

RQ2: Which prior knowledge helps? Helpful rules depend on the relationships and observation conditions. Adding more rules is not automatically beneficial. Value equalities and known sign reversals reduce reconstruction error in the linear tests and improve diagnosis in the dificult condition, although the diagnosis gain is not significant after campaign-wide correction. Direction and delay rules provide no comparable diagnosis gain in that condition.

In the hybrid tests, mode-dependent measurement rules improve current-state reconstruction, whereas structure compatibility alone does not. Combined rules improve reconstruction under stronger masking but worsen nominal forecasting, without a reliable diagnosis gain. With two measurements masked, combined rules leave more faults without a diagnosis than training without rules. Lower reconstruction error alone is therefore insuficient for choosing the rules. TEP likewise shows no consistent diagnosis gain: qualitative rules were nearly inactive, while active topology rules did not improve isolation. HAI uses Gödel rules in both training variants, so this comparison does not isolate their benefit.

RQ3: When does the earliest symptom identify the cause? The method provides useful isolation when the local fault produces a persistent symptom before competing symptoms. Without rules, it uniquely diagnoses 72.40% and 64.97% of fault recordings in the sixcomponent hybrid tests under the two masks. Additionaloutflow faults are isolated more reliably than actuatorefectiveness losses, which more often fail to activate a persistent local symptom. The failure audit also links early false symptoms to nominal-forecast errors around normal mode switches, even when current-state reconstruction is accurate. Synchronization alone cannot remove these prediction and detection errors.

The linear tests retain high isolation under masking and moderate disturbances, and clock-ofset correction restores symptom ordering in the temporal ablation. In HAI, recursive training uniquely identifies the faulty pump in 29 of 30 fault recordings. TEP includes the true cause in the earliest symptom group in about two thirds of cases, substantially above size-matched random selection, but ties can retain several causes. The remaining priorities are accurate nominal forecasts, reliable local symptom detection, and temporal separation between competing symptoms.

## 5 Conclusion

ReDiVO connects reconstruction of named root-cause variables with time-based fault isolation. Measurements and virtual sensors form the current system trajectory. Comparison with a learned nominal trajectory provides persistent local symptoms. Their onset times distinguish a unique cause or a group of possible causes. Gödel regularization incorporates physical and system knowledge during training only.

RQ1 is supported by useful virtual-sensor symptoms in both artificial studies and the HAI pump cases, including four hidden measurements out of six in the artificial systems. Direct measurement of every root-cause variable is not necessary under these observation conditions. RQ2 has a conditional answer. Specific value relationships reduce reconstruction error in the reported linear comparisons. Mode-dependent rules improve reconstruction in the hybrid systems, but combining them with structure rules can worsen nominal forecasting and diagnosis. TEP shows no consistent additional diagnosis gain. Priors must therefore be assessed for their efect on both trajectories and the resulting symptoms. For RQ3, onsetbased diagnosis achieves high isolation in the linear tests, correctly isolates most faults in the six-component hybrid tests without rules, and selects the faulty pump in 29 of 30 HAI fault recordings with recursive training, compared with 25 of 30 with one-step training. TEP places the true cause in the earliest symptom group substantially more often than random selection. Missed actuator symptoms and early false symptoms remain important limits of the hybrid results, despite useful reconstruction.

Future work should improve nominal forecasts around mode switches and calibrate false alarms using longer normal recordings. Accumulating small residual deviations may reveal faults missed by consecutive-threshold detection, but requires normal-data calibration and checks for shifted symptom-onset times. Further ablations should identify which Gödel rules help which variables and fault mechanisms, and at what weights. Timing studies should vary clock errors, sampling intervals, and sensor-response delays to quantify the precision needed to distinguish competing symptom onsets. Independent fault realizations should test whether these findings generalize.

Code and data availability. Algorithm code and generators for the artificial datasets are provided at https: //github.com/ollini42/research-artifacts/tre e/d989de6e4737cf836c54fa2846b5e023cff2aace/goe del-diagnosis-2026. Extended TEP data are available at https://doi.org/10.11583/DTU.13385936.v1. HAI-CPPS benchmark data (Ehrhardt et al., 2025) are available at https://doi.org/10.21227/5ewb-cn40, with v2.1 generation code at https://github.com/j -ehrhardt/hai-cpps-benchmark/tree/v2.1. Twinflow data are available at https://zenodo.org/rec ords/22943653, with the generation code at https: //github.com/imb-hsu/twinflow.

## Disclaimer

During the preparation of this work the authors used OpenAI’s ChatGPT and Codex in order to translate texts, discuss concepts and fine-tune the readability of the manuscript and to support the development and review of research code. After using this tool/service, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## References

Assaad, C.K., Ez-Zejjari, I., and Zan, L. (2023). Root cause identification for collective anomalies in time series given an acyclic summary causal graph with loops. In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research, 8395–8404. PMLR. URL https://proceedings.mlr.press/v2 06/assaad23a.html.

Badreddine, S., d’Avila Garcez, A., Serafini, L., and Spranger, M. (2022). Logic tensor networks. Artificial Intelligence, 303, 103649. doi:10.1016/j.artint.2021.10 3649.

Bathelt, A., Ricker, N.L., and Jelali, M. (2015). Revision of the tennessee eastman process model. IFAC-PapersOnLine, 48(8), 309–314. doi:10.1016/j.ifacol.201 5.08.199.

D’Amour, A., Heller, K., Moldovan, D., et al. (2022). Underspecification presents challenges for credibility in modern machine learning. Journal of Machine Learning Research, 23(226), 1–61. URL https://jmlr.org/p apers/v23/20-1335.html.

Diedrich, A., Heesch, R., Bozzano, M., Ludwig, B., Cimatti, A., and Niggemann, O. (2024a). Inferring sensor placement using critical pairs and satisfiability modulo theory. In 35th International Conference on Principles of Diagnosis and Resilient Systems (DX 2024), volume 125 of OASIcs, 9:1–9:19. doi: 10.4230/OASIcs.DX.2024.9.

Diedrich, A., Moddemann, L., and Niggemann, O. (2024b). Learning system descriptions for cyberphysical systems. IFAC-PapersOnLine, 58(4), 628–633. doi:10.1016/j.ifacol.2024.07.289.

Ehrhardt, J., Moddemann, L., Diedrich, A., and Niggemann, O. (2025). HAI-CPPS: The Hamburg AI benchmark for cyber-physical production sytems. IEEE Dataport. doi:10.21227/5ewb-cn40.

Han, X., Absar, S., Zhang, L., and Yuan, S. (2025). Root cause analysis of anomalies in multivariate time series through Granger causal discovery. In The Thirteenth International Conference on Learning Representations. URL https://openreview.net/forum?id=k38Th3x4 d9.

Harirchi, F. and Ozay, N. (2018). Guaranteed modelbased fault detection in cyber-physical systems: A model invalidation approach. Automatica, 93, 476–488. doi:10.1016/j.automatica.2018.03.040.

Hogea, E., Onchiş, D.M., Yan, R., and Zhou, Z. (2024). LogicLSTM: Logically-driven long short-term memory model for fault diagnosis in gearboxes. Journal of Manufacturing Systems, 77, 892–902. doi:10.1016/j.jm sy.2024.10.003.

Hranisavljevic, N., Diedrich, A., Moddemann, L., Schaeffer, D., Marek, F., Pill, I., and Niggemann, O. (2026a). Hsu twinflow: A living benchmark for evaluating anomaly detection and diagnosis in cppss using digitaltwin-generated data. doi:10.5281/ZENODO.22943652. URL https://zenodo.org/doi/10.5281/zenodo.22 943652.

Hranisavljevic, N., Diedrich, A., Moddemann, L., Schaefer, D., Marek, F., Pill, I., and Niggemann, O. (2026b). HSU TwinFlow: A Living Benchmark for Evaluating Anomaly Detection and Diagnosis Methods in Cyber-Physical Production Systems Using Digital-Twin-Generated Data. In 37th International Conference on Principles of Diagnosis and Resilient Systems (DX 2026), volume 148, 7:1–7:14.

Isermann, R. (2005). Model-based fault-detection and diagnosis: Status and applications. Annual Reviews in Control, 29(1), 71–85. doi:10.1016/j.arcontrol.2004.12. 002.

Krysander, M., Åslund, J., and Nyberg, M. (2008). An eficient algorithm for finding minimal overconstrained subsystems for model-based diagnosis. IEEE Trans. Systems, Man, and Cybernetics – Part A, 38(1), 197– 206.

Ludwig, B., Ehrhardt, J., and Niggemann, O. (2025). Creating virtual sensors using neural networks. In Proc. IEEE 30th International Conference on Emerging Technologies and Factory Automation, 1–8. doi:10.110 9/ETFA65518.2025.11205672.

Moddemann, L., Steude, H.S., Diedrich, A., Pill, I., and Niggemann, O. (2024). Extracting knowledge using machine learning for anomaly detection and root-cause diagnosis. In 2024 IEEE 29th International Conference on Emerging Technologies and Factory Automation (ETFA). IEEE. doi:10.1109/ETFA61755.2024.1071064 7.

Oneto, L., Ridella, S., and Anguita, D. (2025). Informed machine learning: Excess risk and generalization. Neurocomputing, 646, 130521. doi:10.1016/j.neucom.2025. 130521.

Qin, S.J. (2012). Survey on data-driven industrial process monitoring and diagnosis. Annual Reviews in Control, 36(2), 220–234.

Reinartz, C.C., Kulahci, M., and Ravn, O. (2021). An extended tennessee eastman simulation dataset for fault-detection and decision support systems. Computers & Chemical Engineering, 149, 107281. doi: 10.1016/j.compchemeng.2021.107281.

Revach, G., Shlezinger, N., Ni, X., López Escoriza, A., van Sloun, R.J.G., and Eldar, Y.C. (2022). Kalman-Net: Neural network aided kalman filtering for partially known dynamics. IEEE Transactions on Signal Processing, 70, 1532–1547. doi:10.1109/TSP.2022.3158588.

Sampath, M., Sengupta, R., Lafortune, S., Sinnamohideen, K., and Teneketzis, D. (1995). Diagnosability of discrete-event systems. IEEE Transactions on Automatic Control, 40(9), 1555–1575. doi:10.1109/9.412626.

Steude, H.S., Moddemann, L., Diedrich, A., Ehrhardt, J., and Niggemann, O. (2024). Diagnosis driven anomaly detection for cyber-physical systems. IFAC-PapersOnLine, 58(4), 13–18. doi:10.1016/j.ifacol.2024. 07.186.

van Krieken, E., Acar, E., and van Harmelen, F. (2022). Analyzing diferentiable fuzzy logic operators. Artificial Intelligence, 302, 103602.