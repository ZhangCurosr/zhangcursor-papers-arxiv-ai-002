# Toward Fast Human Reaction Modeling from High-Frequency Digital Interaction Data

Emanuel Dopater<sup>∗</sup>, Daniel Gavlas<sup>∗</sup>, Lukáš Jochymek<sup>∗</sup>, Kannan Paul<sup>∗</sup>, Miloslav Szczypka<sup>∗</sup>, David Vojácekˇ <sup>∗</sup>, Josef Žácekˇ <sup>∗</sup>, Jan Plucar <sup>†</sup>, Radek Svoboda <sup>†</sup>, Ivan Zelinka <sup>†</sup>

<sup>∗</sup>Mesocort AI

Tovární 589, 709 00 Ostrava 9, Czechia

{emanuel.dopater, lukas.jochymek, kannan, miloslav.szczypka, david.vojacek, josef.zacek}@mesocort.ai

<sup>†</sup>Faculty of Electrical Engineering and Computer Science

VŠB–Technical University of Ostrava

17. listopadu 2172/15, 70833 Ostrava, Czechia

{jan.plucar, radek.svoboda, ivan.zelinka}@vsb.cz

Abstract—We ask whether limbic-like behavioral signatures, fast salience-driven responses preceding deliberation, can be detected and modeled from high-resolution Counter-Strike 2 telemetry. As our first work in this area, the paper is primarily exploratory: it brings several ideas together into a single framework for studying rapid reactions, response latency, and decision-making under pressure, and some parts are necessarily preliminary. The central contribution is an Ego-Knowledge representation, a leak-reduced reconstruction of what each player could plausibly perceive rather than omniscient replay truth, together with a millisecond-precision visibility engine, a reactioncentered extraction pipeline, and an intracranially motivated 60– 120 ms operational research window. We derive coarse fightor-flight labels for a mixed dataset of visibility-triggered and damage-triggered events and train two telemetry-native models using information available at or before the trigger: EMBER (gradient boosting) and ORBIT (a custom transformer). Backdamage events provide a constrained setting for reaction-time analysis and manual validation of the automatic labels. EMBER reaches 73.5% and ORBIT 71.2% on the full validation set against a 51.5% majority-class baseline; both generalize to heldout players. The strongest large language model baseline reaches 62.7% on 1,000 predictions drawn from the same validation set; on those same trials EMBER (73.6%) and ORBIT (70.9%) outperform it (McNemar, $p = 1 . 3 \times 1 0 ^ { - 9 }$ and $p = 1 . 5 \times 1 0 ^ { - 6 } ) .$ and the best LLM accuracy did not improve across a generation of newer models. Both models run at several orders of magnitude lower inference time and cost than the LLM baselines, although not compute-matched. A short post-anchor window turns prediction into early response recognition, with accuracy rising monotonically to 85.15%; we propose this curve as a measurable handle on the fast-to-deliberative control transition, a hypothesis that the present, roughly uniform per-tick gains support but do not establish. Neuroscience vocabulary is hypothesis-generating rather than mechanistic evidence; physiological ground truth, control distributions, and the reverse-inference constraint remain open, and a roadmap toward auditory reconstruction, largerscale data, and telemetry-native foundation models closes the paper.

Index Terms—behavior modeling, reaction time, affective computing, esports analytics, game telemetry, ego-knowledge, stress response modeling

## I. INTRODUCTION

Human behavior under pressure is not governed exclusively by slow, deliberative reasoning. Contemporary neurophysiology suggests that rapid, evolutionarily older systems can initiate orienting and action before full cortical appraisal is complete, and coarse sensory detection, salience evaluation, and motor preparation may occur within approximately 60– 120 ms with limited conscious involvement. We treat this interval as an operational research window rather than a settled neuroscience category; its strongest direct support is intracranial evidence of rapid fear-selective amygdala responses in humans, approximately 74 ms for visible and 88 ms for invisible fearful faces [1], [2]. The Mesocort project asks whether traces of such fast, salience-driven, limbic-like control can be detected and modeled from high-resolution Counter-Strike 2 (CS2) gameplay telemetry [3]–[5].

Dual-pathway theories distinguish a slower cortical “high road” from a faster subcortical “low road,” but the framing is debated: Pessoa and Adolphs argue for “many roads” rather than a single privileged shortcut, and recent discussion emphasizes distributed relevance evaluation over a one-route model [3], [6], [7]. Converging intracranial, primate, and tractography evidence nonetheless keeps a fast subcortical contribution plausible [1], [2], [8], [9]. In motor neuroscience, the StartReact phenomenon shows that, under some conditions, prepared motor programs can be released at unusually short latencies through brainstem-mediated pathways; it serves here only as an analogy, since many CS2 responses are clearly more complex than a preloaded laboratory movement and should not be read as direct StartReact equivalents [10]–[12]. These perspectives motivate the hypothesis that at least some actions in high-pressure digital environments may emerge from anticipation, rapid salience confirmation, and fast motor execution rather than from fully deliberative reasoning alone.

CS2 combines millisecond-level reaction demands, incomplete information, repeated threat-like stimuli, and rich behavioral telemetry. It thus serves as a naturalistic laboratory that preserves the complexity of real decision-making under pressure at resolutions rarely available experimentally; unlike classical reaction-time experiments isolating simple sensorimotor responses, it captures the continuous interaction of reflexive reactions, expectation, tactical reasoning, and strategic adaptation [13]–[15].

Replay data contains the full ground truth of the match, whereas the player does not; omniscient access to the world state makes standard AI systems poor models of human cognition. The project’s core methodological principle is therefore an Ego-Knowledge layer approximating what the player could realistically see, hear, infer, and believe at each moment.

On top of this layer, first-contact events, visual or auditory moments anchoring time zero for reaction latency, support analysis of rapid post-stimulus behavior (crosshair correction, flick onset, counter-strafing, crouching, escape movement) as candidate indicators of a fast reactive layer, distinct from slower tactical actions (peeking, holding, retreating). This yields a hierarchy of behavioral targets (H1, micro-intent and reflexive motor response; H2, tactical choice; H3, strategic plan shifts), with the longer-term aim of inferring not only the action but the control mode behind it; the present paper addresses the action.

Modeling proceeds incrementally, from baseline models that validate feature quality and leakage risk toward sequential models over Ego-Knowledge history evaluated on unseen players, maps, and regions; if successful, this would offer a framework for studying pressure-sensitive behavior in ecologically valid digital environments [13], [16].

Scope of the neuroscience. The neuroscience reviewed in the next two sections is the motivation for this work, not one of its results. It shaped three design choices: which events we extract (abrupt, high-salience contacts), where we look (the first moments after stimulus onset), and the vocabulary of the outcome labels. The contributions and results themselves are behavioral and computational: a representation of what a player could perceive, a pipeline that extracts reaction moments, and models that predict a labeled outcome. No result in this paper localizes a response to a brain structure or identifies the control process that produced it, and the system latency of the recordings means that the 60–120 ms window cannot be resolved on single events (Section X-A). We use “limbic-like” descriptively, for fast salience-driven defensive responding, without presupposing a unitary limbic system, a concept that has long been questioned [17]. Section X states what the present data cannot show, and Section XIII lists the experiments that would test the framing.

## A. Contribution Framing

We also introduce two complementary fight-or-flight classifiers: ORBIT, a custom transformer for temporally ordered Ego-Knowledge sequences, and EMBER, a gradient-boosted, non-sequential model for lower-cost prediction from aggregated game-state features. Both are evaluated against commercial and open-source large language model baselines, and we also analyze performance as the input extends into the postanchor window, separating pre-reaction prediction from early response recognition. Although EMBER performs on par with or slightly above ORBIT in this constrained Phase 1 setting, the transformer is the primary direction because the longerterm objective is a single model serving multiple behavioral use cases under substantially larger, more temporally structured data.

The empirical anchor is a deliberately constrained class of high-salience back-damage events. Their stimulus-response structure is clear: the attack arrives from outside the current focus of attention, and the response reads as an immediate defensive, evasive, or engagement-oriented reaction, less ambiguous than complex tactical situations.

The primary contributions of this work are as follows:

1) A conceptual framework for studying fast pressuresensitive behavior in a high-resolution digital environment, with a modeling roadmap toward player-specific, multi timescale behavior prediction by a single telemetry-native model serving multiple behavioral tasks at larger data scale.

2) A leak-reduced Ego-Knowledge reconstruction pipeline for approximating player-specific subjective information and belief state.

3) A custom millisecond-precision visibility engine that computes the visible fraction of a target from an observer’s exact viewpoint via 3D depth-map segmentation, enabling a precise, retroactive stimulus-onset anchor not available from standard line-of-sight flags.

4) A reaction-centered dataset focused on interpretable backdamage events, where the player is attacked from behind and must react immediately to an unexpected threat.

5) A predictive modeling framework for fight/flight outcome classification, combining coarse behavioral labeling with two complementary models, a custom transformer (OR-BIT) and a gradient-boosted model (EMBER), and comparative evaluation against commercial and open-source LLM baselines.

6) A manual gold-standard validation of the automatic labeling procedure (majority voting over seven expert annotators, with quantified consensus coverage and label agreement), and an initial empirical validation that backdamage events preserve a structured short-latency response regime in the source telemetry, with a reference-frame analysis reconciling the measurement floor, the proposed 60–120 ms window, and the engine/network system delay separating demo-measured latency from true reaction time.

## II. STATE OF THE ART

Recent research suggests that virtual threat processing is not cognitively isolated from real-world threat processing: rather than a separate “digital” pathway, the nervous system appears to recruit conserved threat-related computations spanning sensory detection, integrative limbic and hypothalamic processing, and survival-oriented output structures such as the periaqueductal gray, shaped in games by amygdala and prefrontal interaction. In virtual reality, oscillatory correlates of threat imminence have been reported during first-person navigation, and early threat perception appears partially dissociable from later cognitive control [5], [16], [18]–[20].

A useful organizing framework is the Predatory Imminence Continuum, which distinguishes pre-encounter, post-encounter, and circa-strike defensive modes, from cautious scanning and anticipatory vigilance through freezing and risk assessment to explosive fight-or-flight when threat becomes proximal and unavoidable; central-amygdala circuitry has been proposed as a candidate switching mechanism [18], [21]–[24]. For gameplay analytics this is an analogy rather than a literal equivalence (a player peeking a corner is not an animal in post-encounter defense), but it provides a useful computational vocabulary for high-pressure behavioral regimes, especially alongside the emerging VR evidence [19], [24].

Digital games are nonetheless not perfect equivalents of real-world threat. Stressful 2D gaming environments produce measurable autonomic and psychological stress responses but also involve somatic decoupling: threat-related arousal occurs while the downstream motor consequences of biological fightor-flight are constrained by the sedentary, interface-mediated task. Mesocort therefore does not claim direct measurement of literal biological fight-or-flight behavior, but seeks limbic-like behavioral signatures or low-latency salience-driven response patterns that emerge under pressure in a high-resolution digital setting [13], [14], [25].

In-game avoidance, withdrawal, aggression, pacing, and risk-taking may reflect broader tendencies such as coping style, neuroticism, or behavioral inhibition, although fast behavior in highly practiced players may instead reflect automaticity and overlearned sensorimotor routines [26], [27]; the defensible implication is not personality inference from isolated episodes but that repeated player behavior may contain stable personal signatures.

fMRI studies of practiced FPS players report active suppression or deactivation of amygdala responses during gameplay, with increased engagement of dorsal anterior cingulate and prefrontal regions associated with executive control, suggesting that with extensive practice threat processing becomes increasingly cortically regulated rather than subcortically driven [28]–[30]. The population studied here, experienced competitive players with thousands of hours of play, may therefore be the population least likely to exhibit tonic limbic engagement during routine gameplay. This does not invalidate the search for fast salience-driven responses, since tonic suppression is compatible with transient phasic reactivation during genuine surprise events such as unexpected back-shots, but any “limbic-like” signal in experienced players is likely to be event-locked, brief, and substantially attenuated relative to naive subjects. We treat this literature as a boundary condition rather than a contradiction: it narrows the expected signal to specific high-salience surprise events and argues against reading broad gameplay behavior as limbic.

Repeated exposure to digital stressors may also lead to habituation or desensitization, and the absence of real biological cost changes risk calibration relative to physical reality, arguing against strong claims of one-to-one translation between virtual and physical behavior. The most robust framing is that game telemetry offers an ecologically rich environment for studying pressure-sensitive behavioral organization, while broader transfer claims must remain probabilistic, constrained, and empirically validated [7], [13], [25].

A second tradition bears directly on the control-mode question. Computational accounts of decision making distinguish model-free control, which selects actions from cached values learned through experience, from model-based control, which evaluates actions against an internal model of the environment, and treat behavior as arbitrating between them according to their relative uncertainty [31], [32]. Two properties of that literature constrain the present work. First, model-free control is itself predictive: a temporal-difference value function is a prediction, so speed does not imply the absence of prediction. Second, model-based control becomes fast and automatic with training, and intermediate representations such as the successor representation can produce model-based-looking behavior through model-free machinery at low computational cost [33]. Latency therefore cannot by itself identify which controller produced a response. We consequently name the reaction-time bands in this paper by latency rather than by mechanism, and treat the mapping from band to control mode as a hypothesis the design is built to test rather than as an assumption it encodes.

Relatedly, work on large-scale human gameplay data has established the practice of estimating an upper bound on predictable behavior by fitting a high-capacity neural network and treating its performance as a ceiling against which interpretable cognitive models are compared [34]. We do not yet report such a bound. Section VIII-F reports only the reliability of our labels, whose Fleiss’s kappa of 0.38 indicates that a substantial part of the residual error may be attributable to label noise rather than to unmodeled behavioral structure. We identify the predictability ceiling as the primary missing quantity for interpreting the accuracies reported here.

The behavioral signal studied here is a high-frequency multivariate time series: each tick contributes a fixed-width feature vector, and the task maps a window of recent ticks to nearfuture behavior. Transformer forecasters such as Informer [35], Autoformer [36], and PatchTST [37] are strong on longhorizon benchmarks, but the quadratic cost of self-attention and its weak inductive bias for ordered numerical data have drawn criticism, most pointedly the finding that a simple linear model can outperform them [38]. State-space models, which scale linearly with sequence length but compress history into a fixed-size state and must retrofit cross-channel interaction [39]–[43], and pretrained time-series foundation models [44]–[46] are natural later directions. Because the relevant context spans hundreds of densely sampled ticks and the per-tick feature vector is strongly multivariate, we adopt a custom attention-based architecture (Section VI-D).

## III. NEUROPHYSIOLOGICAL MOTIVATION FOR THE FAST REACTION WINDOW

Mesocort hypothesizes a fast reaction window of approximately 60–120 ms, treated as an operational research interval rather than a canonical neuroscience construct. The lower end is anchored by intracranial evidence of fear-selective amygdala responses at approximately 74 ms for visible and approximately 88 ms for invisible fearful faces in humans [1], [2]; the upper end loosely marks where cortically elaborated processing becomes increasingly plausible. It is thus not a reflex interval but a zone in which rapid salience detection, anticipatory motor readiness, and fast descending control may converge [3], [5].

One basis is the dual-pathway account of perception, described as a slower cortical “high road” and a faster, coarser “low road.” Conscious recognition depends on the geniculostriate pathway (lateral geniculate nucleus, primary visual cortex), whereas a faster subcortical route via the superior colliculus, pulvinar, and amygdala has been proposed to detect coarse, potentially threatening patterns fast enough to bias behavior before detailed analysis concludes. This account remains debated, and the paper treats the “low road” as one plausible mechanistic contributor rather than as settled doctrine [1]–[3], [6], [8], [9].

The amygdala is thus best understood as an early relevanceevaluation node rather than a passive recipient of interpreted percepts, and its rapid responses are consistent with threatening cues influencing defensive preparation at very short latencies. Some of the earliest post-contact adjustments in gameplay may therefore reflect fast salience-driven processing rather than only deliberate tactical reasoning [2], [5], [8].

Fast response also depends on motor release: the corticospinal tract is more associated with fine, fractionated voluntary control, whereas the reticulospinal system is primarily associated with posture, coordination, and rapid whole-body action (with some fine-motor contributions) [47]. The StartReact literature suggests that prepared motor programs can be released at unusually short latencies through brainstem-mediated mechanisms, especially under anticipatory readiness [10]–[12]. Here this is only a partial analogy: a CS2 player reacting to a back-shot is not necessarily executing a prepared motor program and may require directional inference and action selection.

Individuated finger movements are not susceptible to StartReact, being predominantly corticospinally mediated [11], but CS2 play involves bimanual coordinated control closer to the coordinated-grasp condition in which StartReact is triggered [48], and mouse aiming engages proximal musculature with well-established reticulospinal contributions [49]. CS2 motor responses are therefore best described not as reticulospinally dominated but as a mixture of corticospinal and reticulospinal contributions whose weighting likely depends on the action (fine crosshair adjustment versus gross emergency reorientation) and on anticipatory preparation.

Very short latencies are therefore not automatically stimulus-driven. Between roughly 40 and 60 ms lies a gray zone in which sensory registration may begin but conscious discrimination and large-scale motor implementation are not yet plausible, so responses that appear “too fast” likely involve a mixture of prediction, anticipation, motor preloading, or rapid confirmation. Predictive coding reinforces this: in practiced, temporally structured environments such as esports, expectations may pre-activate sensorimotor pathways and reduce effective latency once sensory confirmation arrives, so exceptionally fast actions in Counter-Strike 2 should not be interpreted as “pure reflexes” but as the product of a coupled system of anticipatory preparation, rapid salience detection, and optimized motor execution [15], [50].

For Mesocort, low-latency events after first contact may thus carry information about the player’s control mode, and the 60–120 ms regime is especially interesting as the boundary between deliberate and pre-deliberative control. This justifies no claim of direct measurement of a brain structure or literal limbic discharge, but it does justify treating them as candidate markers of a fast reactive layer of behavior: neurophysiologically plausible, computationally measurable, and distinct from slower tactical and strategic planning [5], [11], [15]. In the present data the window itself is not resolvable: the recordings carry a system delay of the same order as the window (Section X-A), so it serves here as the motivation for the latency analysis rather than as a quantity we measure.

## IV. SYSTEM OVERVIEW AND EGO-KNOWLEDGE RECONSTRUCTION

The system has four layers (Fig. 1): capture acquires gameplay traces and game state from CS2 demo files, and reconstruction transforms them into a player-centric representation of what could plausibly be perceived at each moment. Event mapping extracts first-contact windows and downstream actions, and prediction uses the resulting sequence for shorthorizon behavior modeling.

## A. Why Ego-Knowledge Is Necessary

A replay contains the full ground truth; the player sees only a restricted, uncertain subset shaped by line of sight, field of view, sound cues, prior observations, map knowledge, and short-term inference. A model trained on omniscient replay truth may predict well yet remain cognitively unrealistic. Mesocort therefore centers on an Ego-Knowledge layer approximating the subjective information state, not the objective replay truth.

This distinguishes Mesocort from environment world mod els such as JEPA and V-JEPA 2 [51], [52], which learn environment dynamics but do not model the gap between the objective world state and the partial information available to a specific human agent; a world model trained on CS2 replays would risk omniscient leakage. Ego-Knowledge does not reconstruct the full world state but filters it through what the player could plausibly perceive or infer (Section IX).

![](images/072dd8d8a7a71d6d900ec91e1d21a774c7226b3df485d7194376324b30a1fba5.jpg)  
Fig. 1: High-level architecture of the Mesocort pipeline. Raw CS2 demo telemetry is transformed into reconstructed game state, player-centric Ego-Knowledge, reaction windows, and fight/flight labels used for behavior modeling, prediction, and evaluation.

## B. Reconstruction Principles

A leak-reduced pipeline converts demo files into a pertick timeline of player positions, view angles, utility usage, health states, and combat events. Line-of-sight and field-ofview calculations estimate what was visually available at each moment; auditory estimation is planned, fusing both into a belief state of where the player could plausibly believe enemies to be. This remains a physically motivated approximation rather than a validated proxy for full perception: it does not quantify attentional narrowing, display-dependent detectability, or other observer-side factors, and reconstruction fidelity is an open validation target. The current layer is handengineered; the target architecture is a learned agent-centric encoder, analogous to I-JEPA [51], with the visibility engine as a training-time data filter, not a feature extractor.

## C. Behavioral Hierarchy

H1 captures micro-intent and reflexive motor response: abrupt crosshair correction, movement onset, weapon fire, or rapid reorientation after first contact. H2 captures tactical choices: peeking, holding, retreating, rotating, or deploying utility. H3 captures strategic plan shifts: delayed team rotation or map-wide repositioning. A sub-100 ms motor onset and a multi-second strategic response do not share a control process. The mapping to threat-imminence theory is analogical, not literal, but bridges established neuroscience frameworks [23], [24]: pre-encounter states (rotating to a new area, no enemies detected, high uncertainty) map to strategic control (H3); postencounter states (enemy detected or suspected via sound, communication, or partial visibility) to tactical control (H2); and circa-strike states (direct engagement, being shot at, surprise contact) to reactive control (H1).

## D. Definition of Fight/Flight Labels

Fight and flight are operational gameplay labels, not direct references to the classical fight-or-flight response described by Cannon [53]: a coordinated acute stress reaction in which sympathetic and adrenomedullary activation mobilize the organism for defensive action [53]–[55]. Our use is narrower and purely behavioral: the labels imply neither a literal survival response nor direct observation of sympathetic, hormonal, limbic, or other neurophysiological mechanisms, only their measurable outcomes in high-frequency gameplay data.

Labels are defined around reaction moments in which the analyzed player is damaged from outside the current field of view, most characteristically from behind, a scenario chosen for its relatively clear stimulus-response structure: a sudden, unexpected threat answered under time pressure with limited information on the attacker’s position.

A reaction is labeled fight when immediate post-damage behavior indicates re-engagement with the threat (rapid rotation toward the attacker, crosshair correction, visual search, aiming, or fire toward the source of damage) and flight when it indicates disengagement (movement toward cover, retreat from the exposed position, repositioning away from the attacker, or evasive movement without an immediate attempt to locate or engage the opponent). These are the dominant observable alternatives after the triggering event: coarse outcome classes summarizing the short-horizon response, not exhaustive descriptions of intention. Fight often contains H1 components; flight often adds H2-level repositioning.

## V. DATA AND DATA PREPARATION

The data foundation is a medallion architecture with three layers, stored as Parquet-backed Delta Lake tables: Bronze preserves raw telemetry in immutable form, Silver consolidates it for analysis, and Gold prepares model-ready features. The atomic Bronze unit is a tick, one sampled snapshot of the complete game state; CS2 demos are recorded at 64 ticks per second, approximately 15.625 ms per sample. Bronze separates into tick-level data, a per-tick state vector for every player (position, velocity, team, health, armor, inventory, view orientation, pressed buttons, and round-level context), and event-level data, a sparse stream of discrete combat, bomblifecycle, and meta events. Silver removes invalid or incomplete matches, applies consistency checks against noise and leakage, adds match summaries, and derives features aligning sparse events with the continuous game state. Gold produces the wide feature vectors consumed by training: numerical features are scaled according to their distribution (min-max, standard, power, and logarithmic transformations) and categorical and discrete spatial variables are one-hot encoded or indexed for embedding layers.

## A. Raw Data Source: CS2 Demo Files

CS2 stores every match as a binary demo file, a compact tick-by-tick log of the game state, which we decode with a custom high-performance C++ parser into the per-tick data used here. All data come from server-side GOTV demos, which record all players from the server’s authoritative view, the source of the server-versus-client timing gap discussed in Section X-A.

## B. Dataset features

Each input vector describes the reconstructed game state at a single tick from the analyzed player’s perspective; the model receives a sequence of N consecutive such vectors. Features fall into seven semantic groups (Table I). Player features describe position, view direction, health, equipment, movement, and objective-related state; positions of the player, teammates, and enemies are represented both as continuous coordinates and as discretized map-square identifiers used as embedding indices. Aiming and visibility features describe crosshair movement, angular and Euclidean distances to enemies, enemy visibility, and recent observation history, some of which summarize short-term history (e.g., ticks since an enemy was last visible). Player input features encode button states and weapon firing; damage and kill features summarize recent combat interactions and their temporal distance from the current tick. Teammate features cover distance, visibility, and position; enemy features add alive state, line-of-sight relation, crouching, firing, visibility ratio, crosshair distance, and radial velocity. In total, each tick is represented by 154 input features; the complete list with preprocessing methods is given in Appendix A, Table X.

## C. Source Domain and Sampling Strategy

For Phase 1 of Mesocort modeling, we constructed a reaction-centered dataset from Counter-Strike 2 demo recordings, selecting 50 players with the most stable game history and collecting their most recent demos from approximately the last 6 months: an initial pool of approximately 35,000 games, about 700 per player. The rank composition was a compromise between partial uniformity and the natural rank distribution of the broader game population.

## D. Reaction-Moment Scenario

The dataset is built around the reaction moment: a temporally localized situation in which a player must respond to a salient, combat-relevant stimulus under time pressure. The prototypical case is a one-on-one confrontation in which the victim sustains damage from an attacker outside the current field of view, most characteristically a shot from behind. This imposes a clean stimulus-response structure: the threat arrives abruptly, the player is not visually prepared, and the immediate behavior can be interpreted as a defensive, evasive, or retaliatory reaction to a single well-defined onset, with minimal confounding from teammates, crossfire, and competing threats. In Phase 1, candidate moments were triggered by an enemy becoming visible or by the player sustaining damage; prior internal analysis indicated similar reaction-time distributions across the two, so both were included in a shared dataset. The damage-from-behind subset is treated as the cleanest interpretable case and anchors the reaction-time analysis.

## E. Leak Reduction and Filtering Logic

A central design goal was to reduce hidden confounds that could distort the interpretation of reaction latency. Because a full auditory engine was not yet implemented in Phase 1, the extraction procedure explicitly attempted to mitigate soundrelated ambiguity. Reactions to thrown utility were excluded, and later iterations added rules removing cases such as flashed players or active shootouts, which proved to heavily influence the player’s ability to react. These constraints keep the extracted moments interpretable with respect to immediate visual or direct-damage triggers. The network and demorecording delays that shift the effective stimulus-onset anchor are discussed in Section X-A.

## F. Dataset Composition and Splits

For the present experiments we intentionally use only a selected subset of our broader data estate: Phase 1 prioritizes interpretability, leakage reduction, and controlled validation over scale, and the subset is a research sample drawn from a substantially larger and continuously growing platform.

As summarized in Table II, the experimental corpus comprises 32,205 unique demos, 27,356 for training and 4,849 for validation. With the filters above, extraction isolated 271,541 reaction moments, approximately 8 per demo, across 119,149 unique players; the 50-player core subset contained 28,332 moments, roughly 567 per player. Triggers were strongly dominated by visual exposure: 251,427 moments (92.6%) were triggered by an enemy becoming visible and 20,114 (7.4%) by damage received. A semi-automated rule-based procedure then assigned a coarse outcome offight (the decision to engage an opponent in a duel) or flight (the decision to retreat) from relevant features in the 2 seconds following the trigger. In the current version of the dataset, 131,628 reactions (48.5%) are labeled fight and 139,913 (51.5%) flight; the pipeline also stores a confidence-oriented score indicating how strongly a reaction supports its assigned class.

## G. Custom Visibility Engine for Millisecond-Precision Analysis

For analysis at the scale of fractions of a second, we developed a custom visibility engine, since the required signal is not natively available in standard game outputs. It computes the visible fraction of an enemy model from the observer’s point of view for every frame using 3D depth-map simulations of the full game geometry: for each tick, a depth map is rendered from the observer’s exact viewpoint, the target player’s model is rendered into the same scene as a silhouette mask, and pixel-level segmentation marks each silhouette pixel visible (green in Fig. 2) where the model is closer than the nearest occluding surface, or obstructed (red) where geometry intervenes. The ratio of visible to total silhouette pixels gives a per-frame visibility percentage, which identifies the exact moment visual detection begins. For visibility-triggered moments this percentage defines the stimulus-onset anchor from which latency is measured, so any imprecision propagates directly into the behavioral analysis; for damage-triggered moments such as back-damage events, the onset is anchored to the damage tick. Running retroactively on recorded demos, it applies at scale without real-time rendering. To our knowledge no comparable system exists in the public esports analytics literature, where visibility is typically approximated by coarse line-of-sight checks or server-side flags lacking the spatial resolution needed for sub-100 ms analysis.

TABLE I: Summary of input feature groups used by the model.
<table><tr><td>Feature group</td><td>No.</td><td>Description</td></tr><tr><td>Round and map context</td><td>5</td><td>Round-level and map-level state variables.</td></tr><tr><td>Player position and state</td><td>21</td><td>Player position, view direction, equipment, movement, and status variables.</td></tr><tr><td>Aiming and visibility</td><td>13</td><td>Crosshair movement, enemy visibility, and enemy proximity features.</td></tr><tr><td>Player input actions</td><td>12</td><td>Button states and weapon-firing indicators.</td></tr><tr><td>Damage and kill events</td><td>14</td><td>Tick-level and round-level damage and kill history features.</td></tr><tr><td>Teammate features</td><td>24</td><td>Position and visibility features for four teammates.</td></tr><tr><td>Enemy features</td><td>65</td><td>State, position, visibility, distance, and movement features for five enemies.</td></tr><tr><td>Total</td><td>154</td><td>All input features computed for the current game tick.</td></tr></table>

TABLE II: Phase 1 reaction dataset summary.
<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Core players selected for Phase 1</td><td>50</td></tr><tr><td>Average demos collected per core player</td><td>690</td></tr><tr><td>Initial pool of matches</td><td>~35,000</td></tr><tr><td>Time span covered up to</td><td>~7 months</td></tr><tr><td>Filtered demos in extracted dataset</td><td>32,205</td></tr><tr><td>Total reaction moments</td><td>271,541</td></tr><tr><td>Average reaction moments per demo with</td><td>~8</td></tr><tr><td>filters Unique players observed overall</td><td>119,149</td></tr><tr><td>Reaction moments in 50-player core sub- set</td><td>28,332</td></tr><tr><td>Average reaction moments per core</td><td>~567</td></tr><tr><td>player Visibility-triggered moments</td><td>251,427 (92.6%)</td></tr><tr><td>Damage-triggered moments</td><td>20,114 (7.4%)</td></tr><tr><td>Fight labeled reactions</td><td>131,628 (48.5%)</td></tr><tr><td>Flight labeled reactions</td><td>139,913 (51.5%)</td></tr></table>

![](images/712f9bc375c0ee4e15021ff6502a30c2eae91017e96cf01e2d6b5e3a00a4446f.jpg)  
Fig. 2: Visibility engine processing flow: on the left, the scene’s depth map showing the silhouette of the target player; on the right, the pixel segmentation result, where green marks the visible portion and red the portion obstructed by geometry (e.g., a wall), from which the per-frame visibility percentage is computed.

## H. Reaction-Time Distributions

Latency between stimulus onset and reaction onset is measured by an onset-detection algorithm that considers three observed signals (movement on the map, rotation of the player, and shooting) and searches their differences for a sudden change of predefined magnitude; if no plausible candidate is found, the situation is dropped. Reactions cannot be paired with concrete stimuli in 100% of cases, but steps were taken to minimize the influence of other stimuli; the resulting error was found statistically insignificant on a randomly chosen subset and without emphatic bias. Manual labels on a subset of the same scenario moments seem likely to follow a similar distribution to the automatic ones, the most notable difference being more low reaction times (below 150 ms) in the automatic source. Distributions also vary among annotators, so reaction time is not unambiguous from the game data alone, although within a larger dataset these uncertainties are statistically diminished. The back-damage reaction-time histogram is reported in Section VIII-A.

## VI. PROPOSED METHODOLOGY

Figure 3 summarizes the methodology. Gold-layer data and extracted fight-or-flight moments are combined into modelspecific inputs; two settings follow from the temporal information given to the model. In Pre-Reaction Prediction, models classify the player’s later response from information available at or before the anchor (trigger) moment; LLM baselines, EMBER, and ORBIT are evaluated here. In Early Response Recognition, ORBIT and EMBER receive windows that may include a short post-anchor segment, observing part of the reaction as it unfolds, with results reported on a validation split.

## A. Fight/Flight Label Generation

1) Automatic Labeling: The dataset contains first-contact events that led to a reactionary decision. A proprietary rulebased process labels the post-stimulus reaction by estimating the player’s intention: to engage the threat (fight) or to retreat (flight).

The process considers player-specific signals expected to reflect intention in a two-second window after the reaction tick. For each signal and several sub-intervals of that window, the change in the signal is independently mapped to a fight-orflight score via quantile-based thresholds. Sub-intervals closer to the reaction carry higher weight, but the window extends to two seconds because a player can start reacting while the animation of a previous action is still finishing. The scores are combined into a weighted aggregate

![](images/eb6b12b715c5ab229197768b670a6f7f21c5f98ab01deff7a0d86c1e3777406b.jpg)  
Fig. 3: Overview of the methodology. Pre-reaction prediction uses only information up to the anchor moment; early response recognition may also use a short post-anchor window.

$$
s c o r e = \sum _ { s \in S } \sum _ { t \in T } w _ { s } w _ { t } x _ { s , t } ,
$$

where S is the set of signals and $T$ the set of intervals; $w _ { t }$ decreases as |t| increases, $w _ { s }$ is set from experience, and both the weights and the set of signals were re-assessed manually on a data subset. The final score is the absolute difference between the fight and flight scores, and the label is the one with the higher score.

The scoring system is thus a deterministic classifier of future player action; higher scores express higher confidence, and low-confidence values were removed from the dataset used here.

2) Manual Labeling: To obtain ground-truth labels for the response of an unexpectedly damaged player, we developed a custom annotation tool for short, isolated one-versus-one situations in which a player received damage from behind without initially seeing the attacker. A total of 200 such situations were selected, each centered on the damage event. The annotation provided a ground-truth reference for the automatic labels and identified inherently ambiguous situations.

Seven annotators with advanced knowledge of CS2 gameplay labeled the situations independently as fight (the player took the duel: turning toward the attacker, aiming, or firing), flight (the player disengaged: moving away, repositioning, or taking cover), or don’t know when the evidence was insufficient. Discrepancies were resolved by majority voting.

The interface (Fig. 4) combines a two-dimensional topdown reconstruction of the game state with synchronized temporal signals of the analyzed player (crosshair movement, pressed buttons, aim-direction changes, velocity), playback and annotation controls, and gameplay highlights for visual confirmation.

![](images/3dc9b870c1e096b5c5e54f127e2d541fd4ca09bf94da913fbb0d0f10ff56633c.jpg)  
Fig. 4: Manual annotation tool. The central view reconstructs the situation in two dimensions; the left panel shows synchronized temporal signals of the analyzed player; the remaining panels hold playback, annotation controls, and gameplay highlights.

## B. Model Task and Input Sampling

Gold-layer data are chronologically ordered tick-level feature vectors, converted into model-specific samples. The task is defined around selected moments rather than all ticks: each labeled situation is centered on an anchor tick, the moment at which the situation is identified, and the label describes the player’s subsequent response as fight or flight, which typically occurs within approximately one second after the anchor.

Let $\mathbf { x } _ { t }$ denote the feature vector at tick t and $t _ { a }$ the anchor tick. A general temporal sample is

$$
{ \bf X } _ { t _ { a } , H , F } = [ { \bf x } _ { t _ { a } - H + 1 } , \ldots , { \bf x } _ { t _ { a } } , \ldots , { \bf x } _ { t _ { a } + F } ] ,
$$

where H is the number of historical ticks up to and including the anchor tick and $F$ the number of future ticks after it;

the sample’s label is the response associated with the anchor situation. A single-tick sample has $H = 1 , F = 0 ;$ a historyonly sequence has $H = N , F = 0$

1) Pre-Reaction Prediction: Here models classify the response from information available at or before the anchor only $( F = 0 )$ : the post-anchor reaction is not observed, and the eventual label must be inferred from the current or preceding game state. For non-sequential models such as XGBoost the input is the anchor-tick vector $\mathbf { x } _ { t _ { a } } \ \mathrm { ~ o r , ~ }$ as in EMBER, a short window of subsampled pre-anchor snapshots (Section VI-E); for sequence models such as transformers it is a history window of N ticks ending at the anchor, $\mathbf { X } _ { t _ { a } , N , 0 } .$ The target is thus consistent across EMBER and ORBIT, while ORBIT can use ordered pre-anchor context.

2) Early Response Recognition: The second setting studies how post-anchor information $( F > 0 )$ influences performance, with future windows restricted to the short reaction interval after the anchor, up to approximately one second (64 ticks). Although automatic labeling uses a longer two-second window to assign a robust outcome, the 64-tick window tests how early the response can be recognized from its initial development, keeping the focus on fast pressure-driven behavior. Because the model sees partial evidence of the actual reaction, results are interpreted as early response recognition rather than prediction from pre-reaction information; the label definition is unchanged and only the temporal extent of the input differs.

## C. Reaction Detection and Onset Extraction

Two time points must be fixed before a reaction moment is analyzable: the stimulus onset, at which the triggering event becomes available to the player, and the reaction onset, at which the player’s behavior first reflects a response; moments in which either is ambiguous are rejected.

For visibility-triggered moments the stimulus onset is the first frame at which the enemy becomes visible (Section V-G); for back-damage events, where the attacker is typically outside the field of view, it is the damage tick, and a large yaw mismatch between view and attacker direction is additionally required so that the stimulus originates outside the current focus of attention.

Reaction onset is the first sudden change in player movement after the stimulus, operationalized for back-shots primarily as a rapid yaw change, assuming the canonical response is a turn toward the threat. It is localized from crosshair kinematics: the reaction begins when the player abruptly decelerates an ongoing movement before re-accelerating toward the threat, visible in the first and second derivatives of mouse speed. This deceleration and reacceleration signature separates stimulusindependent motion from directed corrective movement, although motion already underway from a previous event can blur it.

Confounded moments (the player already in a shootout, engaging another enemy, flashed, or otherwise occupied) are excluded, and latencies are searched between the minimum measurable latency (≈ 46.875 ms) and approximately one second. The lower bound excludes physically implausible or annotation-induced near-zero values but left-censors the distribution: responses faster than the measurable boundary accumulate near it and cannot, on their own, establish a distinct sub-100 ms mechanism. The extracted latencies are therefore descriptive rather than a formal decomposition of reaction-time regimes [56]–[58].

These latencies are measured in demo time, not true reaction time: $t _ { \mathrm { m e a s u r e d } } \approx t _ { \mathrm { t r u e } } + \Delta _ { \mathrm { s t i m } } + \Delta _ { \mathrm { r e s p } } ,$ , where the two strictly positive system delays total roughly 40–80 ms for backdamage events. Measured onsets are thus conservative upper bounds, and only outcomes from approximately 100 ms of measured latency onward are treated as candidate genuine reactions, the earliest range consistent with a true post-stimulus response once the delay is subtracted (Section X-A).

## D. Behavioral Prediction Model (ORBIT)

ORBIT is a transformer-based neural network that classifies a player’s behavioral response from temporally ordered gamestate representations. For a selected anchor tick, its input is a history window of N consecutive tick-level feature vectors ending at the classified moment; N is a configurable sequence length varied in the experiments.

Each tick-level feature vector becomes one input token holding the complete model input for the analyzed player at that tick (player state, spatial information, visibility, aimingrelated features, input actions, and event-derived variables). Continuous and Boolean variables enter as numerical inputs; selected categorical or discrete variables (round number, weapon identifier, aim-direction state, discretized map-square identifiers) pass through embedding layers, and both are concatenated and projected into a fixed-dimensional token.

Global context tokens carry sequence-level information (map, analyzed player, and rank: learned embeddings for map and player, a projection of the normalized rank) and are prepended to the tick-level tokens. Sinusoidal time embeddings retain temporal order, and a Transformer encoder applies selfattention across all tokens. The encoder output at the final tick represents the classified situation and is passed to a classification head that produces the fight-or-flight prediction (Fig. 5). Hyperparameters are given in the experimental setup.

![](images/9fb21a8a045922db87a0ab51e5099e652e20a096f9651c41f873b2435cd583b3.jpg)  
Fig. 5: ORBIT architecture. A sequence of N tick-level tokens, optionally preceded by global context tokens (map, player, rank), is processed by a Transformer encoder; the final representation feeds a classification head that predicts the behavioral response.

The same architecture serves both experimental settings; only the temporal extent of the input differs.

## E. Gradient-Boosted Trees (EMBER)

EMBER (Ensemble Model for Behavioral Event Response) is a gradient-boosted decision-tree model, the second proposed architecture. Unlike ORBIT, it operates on a temporally subsampled representation of the reconstructed Ego-Knowledge feature space, testing how much of the signal structured gamestate features capture without the cost and complexity of a transformer. Gradient boosting was preferred over linear or logistic regression because of the high dimensionality and heterogeneous structure of the feature space; we use XG-Boost [59].

Because the telemetry changes slowly between consecutive ticks, the native 64 samples per second were reduced to a coarser resolution, typically two per second. Each training instance consisted of a sequence of player-centric game-state snapshots spanning a fixed temporal window relative to the anchor tick (a sub-interval of $[ - 5 \mathrm { s } , + 1 \mathrm { s } ] )$ at this reduced time step (typically and most commonly 0.5 s time step on interval $[ - 2 \mathrm { s } , + 0 \mathrm { s } ] )$ ). The model is a binary classifier over fight and flight.

Hyperparameters (Table III) were kept close to library defaults as a simple, reproducible first configuration; tuning was explored but yielded a validation-accuracy range of only approximately 0.3 percentage points, considered negligible at this stage.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>LEARNING_RATE</td><td>0.02</td></tr><tr><td>MAX_DEPTH</td><td>8</td></tr><tr><td>MIN_CHILD_WEIGHT</td><td>5.0</td></tr><tr><td>SUBSAMPLE</td><td>0.8</td></tr><tr><td>COLSAMPLE_BYTREE</td><td>0.8</td></tr><tr><td>REG_LAMBDA</td><td>1.0</td></tr><tr><td>REG_ALPHA</td><td>0.0</td></tr><tr><td>MAX_BIN</td><td>128</td></tr><tr><td>NUM_BOOST_ROUND</td><td>50000</td></tr><tr><td>EARLY STOPPING ROUNDS</td><td>1000</td></tr></table>

TABLE III: Hyperparameter configuration of EMBER

## F. Baseline Models

ORBIT and EMBER (Sections VI-D and VI-E) are compared against two baselines of increasing capability: a majority-class classifier, a trivial lower bound relying solely on the marginal label distribution, and general-purpose large language models, which probe whether off-the-shelf models can solve the task from textual situation descriptions without task-specific architecture or fine-tuning. The baselines share the label definitions, input information, and metrics of OR-BIT/EMBER (Section VII-C). The LLM baselines are scored on a fixed sample of 1,000 validation rows; for like-for-like comparison, EMBER and ORBIT are additionally scored on the same 1,000 rows (Section VIII-C). Jev, a decision model that does not generate text, is scored on the same 1,000 rows as an additional reference point (Section VII-E).

1) Majority-Class Classifier: The majority-class classifier uses no input features and always predicts the most frequent training class, verifying that the learned models improve over the prior class distribution. In the current split the majority class represents 51.5% of the labeled situations, so this baseline achieves an expected accuracy of 51.5% on data with the same class distribution. Since the classes are close to balanced, it is only slightly above random guessing.

2) Language Model Baselines: LLMs provide a strong and reproducible baseline for the same binary Fight versus Flight task without task-specific architecture or fine-tuning, and give practical reference points for performance, cost, and deployment complexity. The situation (features in Table X) is encoded in three text-based representations:

• Narrative: each game-state row is converted into concise natural-language situation lines by deterministic rulebased templates, using only the top-k features of a feature-importance ranking computed on the training split (Fig. 15); numeric values become interpretable phrases, missing values (including −1) are treated as absent, and few-shot prompts hold class-balanced labeled examples followed by the unlabeled query.

• Compact JSON: the same top-k features are serialized as a minified, nested JSON object per example (e.g. enemy\_X); few-shot examples include the target label, the query omits it, and examples are newline-separated for token efficiency.

• Full CSV: all feature columns are serialized as CSV after removing metadata fields (e.g. demo\_id, tick) to prevent information leak; training examples carry a leading label column, then a header row and one data row per example. This is the textual representation closest to ORBIT’s input.

## G. Evaluation Protocol

Accuracy is the main comparative metric, since the dataset is well balanced and neither error type is preferred (metrics are defined in Section VII-C). Because querying the providers over the full validation set is prohibitively expensive, the LLM baselines are scored on a fixed sample of 1,000 validation rows, which includes all 18 gold-standard consensus moments that fall in the validation partition. ORBIT and EMBER are evaluated on the full validation set and, for like-for-like comparison, on the same 1,000 rows; differences between paired predictions on those rows are tested with McNemar’s test. No confidence intervals are reported for the headline accuracies (Section X).

We also tested whether the models learn different players characteristics. In the player-disjoint test, only the 50 core players were kept; 10 formed the test set and the remaining 40 players’ moments were used for training and validation. In the extended player-disjoint test, the test set again comprised 10 core players while training and validation contained all other players (40 core players plus non-core players).

## VII. EXPERIMENTAL SETUP

This section specifies the data splits, hardware, metrics and model configurations behind the results in Section VIII.

## A. Data Splits

In the player-disjoint setup the test set contained 8,730 observations, with 16,609 and 2,993 rows in the train and validation sets respectively. The extended player-disjoint experiment used the same test set, with train and validation sets of 223,169 and 39,642 rows. Because early player-disjoint experiments found validation and test accuracies to be comparable (Section VIII), the remaining experiments used a train/validation split only, without a dedicated test set, comprising 230,547 and 40,994 observations respectively. Table IV summarizes the set sizes.

<table><tr><td>Setup</td><td>Train</td><td>Val.</td><td>Test</td></tr><tr><td>Player-disjoint</td><td>16,609</td><td>2,993</td><td>8,730</td></tr><tr><td>Extended player-disjoint 223,169</td><td></td><td>39,642</td><td>8,730</td></tr><tr><td>Train/validation only</td><td>230,547</td><td>40,994</td><td></td></tr></table>

TABLE IV: Number of rows in train, validation, and test splits.

## B. Hardware and Software

All experiments were carried out on a single workstation with an NVIDIA RTX PRO 6000 Blackwell GPU (96 GB VRAM) and an Intel Xeon Gold 5412U CPU (24 cores, 48 threads), running Ubuntu 24.04 LTS with Python 3.13. ORBIT was implemented in PyTorch 2.11 and EMBER used XGBoost 3.2 [59]. Data preparation and analysis relied on Polars 1.31 and PyArrow 22.0 for columnar processing, the Delta Lake Python bindings (deltalake 1.3) for the medallionarchitecture tables of Section V, and NumPy, pandas, SciPy and scikit-learn for numerical computation and evaluation metrics; experiment tracking used Weights & Biases. The language-model baselines were not run on this workstation: commercial models were accessed through their provider APIs and the open-source LLMs via cloud-hosted inference on OpenRouter (Section VII-E).

## C. Evaluation Metrics

The models are evaluated with standard binary classification metrics, comparing the predicted class of each selected situation (fight or flight) with its ground-truth label. Because the dataset is approximately balanced between the two classes, accuracy is the primary evaluation metric; precision and recall are reported as complementary metrics, with fight as the positive class.

For the LLMs we additionally recorded inference time and cost. Inference time, in milliseconds, was defined as the wallclock duration from sending the request to receiving the final token of the response; cost was derived from response metadata and pricing information in provider documentation. These two metrics are not measured under like-for-like conditions and should not be read as a controlled latency benchmark. For the LLM strategies the recorded time is an end-to-end wallclock duration that includes the full client-to-provider round trip (network latency, provider-side queueing and server-side batching), none of which reflects the model’s raw forwardpass cost, whereas ORBIT is timed as a single-sample local inference on the dedicated hardware described above, without any network component. The resulting difference is therefore one of orders of magnitude and is dominated by deployment topology rather than by model size or computational complexity alone. We report these numbers because they characterize the operational reality of a locally served telemetry-native model versus an API-served general-purpose LLM, not as a compute-matched comparison; a strict compute-matched comparison, running the LLM baselines locally under identical batching and hardware conditions where possible, is left to future work.

TABLE V: ORBIT hyperparameters used in the experiments.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Task</td><td>Binary classification</td></tr><tr><td>Target classes</td><td>fight, flight</td></tr><tr><td>Loss function Token dimension</td><td>Cross-entropy loss</td></tr><tr><td>Transformer layers</td><td>256 2</td></tr><tr><td>Attention heads</td><td>8</td></tr><tr><td>Dropout</td><td></td></tr><tr><td>Sequence-history ticks</td><td>0.3</td></tr><tr><td>Optimizer</td><td>7, 14, 21, 32, 64, 192 AdamW</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td></td><td></td></tr><tr><td>Weight decay</td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Maximum gradient norm</td><td>1.0</td></tr><tr><td>Training batch size</td><td>512</td></tr><tr><td>Validation batch size</td><td>512</td></tr><tr><td>Number of epochs</td><td>50</td></tr></table>

## D. Model Configurations

1) ORBIT: ORBIT, the main sequence-based neural architecture, is described in Section VI-D; only the concrete settings are reported here. The model was trained as a binary classifier for the fight-or-flight task using cross-entropy loss, with the full set of hyperparameters summarized in Table V. Since ORBIT operates on temporal windows of tick-level tokens, the sequence history length was treated as an experimental parameter: windows of 7, 14, 21, 32, 64 and 192 ticks were evaluated, each ending at the anchor tick of the classified situation.

2) EMBER: The EMBER hyperparameters are listed in Table III; any parameter not mentioned there was left at its default. Its main experimental parameter was the time step between subsequent feature values, since, unlike a transformer, XGBoost cannot inherently model temporal sequences and treats the input as a fixed set of feature columns.

## E. Language Model Setup

The LLM baseline suite evaluates multiple inference providers, comparing cloud-hosted, aggregated and local deployment paths under a consistent prompting and scoring protocol. Models employ prompt caching to substantially lower overall costs and use temperature = 0 to reduce additional hidden reasoning in the answer. Sample prompts for each game-state encoding are provided in Appendix D, and the exact serialization of data into LLM input is described in Figure 17. The following providers were used:

• OpenAI cloud API: GPT-5.6 (Terra), accessed through the OpenAI SDK with a medium reasoning effort.

• OpenRouter cloud: a cloud aggregation layer directing requests to upstream open-source model providers (via the OpenAI SDK); the evaluated models are Qwen 3.5 9B and Qwen 3.8 27B, with extra reasoning tokens turned off and context budgeted locally using tiktoken. The complete set of evaluated configurations is given in Table VII.

• Google Gemini cloud API: Gemini 3 Flash and Gemini 3.7 Flash, accessed through the GenAI client with medium thinking effort.

• Ollama local inference: the lightweight local models qwen2.5:0.5b and llama3.2:1b, served through the local endpoint on port 11434 and run sequentially to avoid overwhelming the available GPU; later open-source evaluations moved to OpenRouter, allowing larger and more capable models to be tested through cloud-hosted routes.

The number of few-shot examples in the system prompt was bounded for cost. For the commercial models most runs used one eighth of the maximum example count that fits the model’s context window (denoted 1/8 in Table VII), with single runs at one quarter and three quarters of that maximum as checks; the open-source models used a fixed 32 examples.

As an additional reference point we evaluated Jev (Type-Safe AI), released in September 2026.<sup>2</sup> Its developer describes it as a “System One” model: instead of generating text token by token, it returns typed, probabilistic decisions in a single pass, and it is aimed at real-time loops such as games, robotics, and simulation, which makes it a relevant comparison for a telemetry task. Jev exposes three decision primitives: Choice (a selection among predefined options), Score (a position on an ordinal scale), and Noul (a single yes/no probability). For this binary task, Choice selects between the two labels, Score is used with a two-level scale so that it returns the probability of fight, and Noul answers whether the player will fight. We searched over the situation encodings, the number of labeled examples (from none to 120, bounded by the model’s context limit), and the three primitives, scoring every configuration on the same 1,000 validation rows as the LLM baselines; Table VII reports the best configuration.

## VIII. RESULTS

This section reports the empirical findings, beginning with a descriptive validation of the extracted reaction-time structure (Section VIII-A) and then turning to the predictive results, their cross-model comparison, an explainability analysis, and qualitative case studies. The predictive results are organized around two temporal regimes that share the same ORBIT architecture but differ in what information the model is given relative to the reaction moment. In the Pre-Reaction Prediction setting (Section VIII-B), the model observes only information available up to and including the anchor tick; no part of the response is visible to it, so the task is genuine prediction of the eventual fight-or-flight outcome. This is the stricter regime and serves as a lower bound on what pre-stimulus context alone can reveal.

In the Early Response Recognition setting (Section VIII-D), the input is extended by a short post-anchor window, so the model observes the first moments of the reaction as it begins to unfold. We regard this regime as the operationally decisive one. All analysis is performed ex post on recorded demo files, and the objective is to recognize, as early as possible, which response is already in progress from its earliest kinematic signature: the model is detecting and classifying an event whose onset is already present in the signal rather than estimating a hidden future. The two regimes should therefore be read as complementary stages along the reaction timeline rather than as competing accuracy figures on a single problem.

## A. Descriptive Validation: Reaction-Time Structure of Back-Damage Events

As a first application of the extraction procedure (Section VI-C), and as an initial validation of the dataset, we apply the reaction-detection protocol to back-damage events. The aim is deliberately modest: rather than fitting a multivariate model, we ask whether the extracted latencies form a structured, non-trivial distribution. Back-damage events constitute a natural surprise paradigm in which the threat arrives abruptly, the player is typically not visually prepared, and the canonical response is a rapid orientation correction. If such events yield a stable and interpretable latency distribution after filtering, the telemetry demonstrates one of the project’s central requirements: that it preserves short-timescale behavioral organization rather than only coarse tactical outcomes.

The resulting distribution (Fig. 6) is neither uniform nor randomly dispersed. Three regions are visible: a pronounced concentration of responses near the lower measurable boundary, a broader dominant mass in the few-hundred-millisecond range, and a long, decreasing tail toward slower responses. The mid-latency mass plausibly corresponds to full defensive reorientation episodes, in which the player must detect the threat, update directional belief, initiate a turn, and prepare a follow-up action; the long tail plausibly captures delayed reactions, hesitation, or residual labeling ambiguity. The apparent separation into a faster and a slower band is the structure of primary interest, as it indicates that more than one timing regime is preserved in the data rather than collapsed into a single mode. Fig. 7 compares the latency distributions obtained when the onset is detected from mouse movement (angular velocity), keyboard input (linear velocity), and weapon fire (shot). The three distributions are clearly offset: onsets detected from mouse movement and keyboard presses peak earliest, whereas the shot distribution is markedly right-shifted. This ordering matches the expected temporal structure of a motor reaction, in which a change in aim or movement is the first observable evidence of a response and firing is a later, downstream component. It reinforces our decision to anchor reaction onset on the first sudden kinematic change rather than on weapon fire (Section VI-C), since a fire-based anchor would systematically overestimate the true reaction latency.

The central observation is that the extraction pipeline preserves meaningful variability across fast, intermediate, and slow response bands rather than reducing them to a narrow artifact. This establishes, descriptively, that the source telemetry contains analyzable short-latency structure close to the project’s operational fast-response regime. The interpretation of this distribution relative to the proposed 60–120 ms window, together with the claims it does and does not support, is taken up in Section IX.

![](images/a3febff5437e7235ef07ca157a96f1fd68a8b79502d7a00b3b8fd7089e3dc8c4.jpg)  
Fig. 6: Histogram of reaction times for events in which the player is shot in the back. Base data granularity is 15.625 ms. The distribution provides an initial descriptive view of shortlatency behavioral structure; number of situations is 20,114.

![](images/c186c041f32568070138f18bea3404b192de6de2588e572b9619e91a8fcd5967.jpg)  
Fig. 7: KDE plots comparing reaction times based on different sources of reaction (KDE plot follows the source histogram).

## B. Pre-Reaction Prediction Model Results

1) ORBIT: In this setting the model uses only information available at or before the anchor tick, so the task is to predict the player’s eventual fight-or-flight response before it is observed.

ORBIT was trained with several predefined sequencehistory lengths to evaluate how the amount of pre-anchor temporal context affects classification performance. Each model used the same architecture and training setup, while the number of preceding ticks in the input sequence was varied over 7, 14, 21, 32, 64, and 192 ticks, each window ending at the anchor tick. The resulting validation accuracies are shown in Table VI.

TABLE VI: Validation accuracy of ORBIT models trained with different sequence-history lengths in the pre-reaction prediction setting.
<table><tr><td>Sequence-history ticks</td><td>Accuracy (%)</td></tr><tr><td>7</td><td>71.20</td></tr><tr><td>14</td><td>70.98</td></tr><tr><td>21</td><td>70.96</td></tr><tr><td>32</td><td>70.58</td></tr><tr><td>64</td><td>68.93</td></tr><tr><td>192</td><td>68.32</td></tr></table>

The best validation accuracy, 71.20%, was achieved with the shortest evaluated window of 7 ticks. Performance gradually decreased as the window increased, with the 192-tick model achieving 68.32%. Although this trend suggests that shorter pre-anchor temporal context was more effective in the current experimental setting, the observed differences should not be interpreted as statistically significant without repeated runs or confidence-interval analysis. The validation set was used for this comparison because the history window is a hyperparameter.

The remaining analysis focuses on the 7-tick ORBIT model. Fig. 8 shows its training and validation loss over 50 training epochs. The training loss decreases consistently throughout; the validation loss follows the same trend during the initial epochs, stops improving after approximately epoch 30, and then begins to increase while the training loss continues to fall, indicating overfitting in the later stages of training. The final evaluation therefore used the checkpoint with the lowest validation loss rather than the weights from the last epoch, so that the reported performance corresponds to the model state with the best observed validation generalization.

The selected 7-tick ORBIT model achieved a validation accuracy of 71.20%, outperforming the majority-class baseline and all longer-history ORBIT variants. This suggests that, under the current dataset size and training setup, a short pre-anchor temporal context was sufficient for the model to capture useful information for fight-or-flight prediction before the player’s response was observed. To assess whether the model generalizes beyond players observed during training, the 7-tick model was also evaluated in a player-disjoint setting, in which the test set contained players not present in the training split. It achieved an accuracy of 71.1%, very close to the 71.20% obtained in the standard validation setting, suggesting that the model does not substantially lose predictive performance on unseen players and learns patterns of fight-orflight behavior that generalize beyond player-specific identity.

Train and Validation Loss over Epochs  
![](images/7acbc7abdb5e88e5a372a3ec1ad0ac368a190ff82bf57a36db372016c0a244ff.jpg)  
Fig. 8: Training and validation loss of the selected 7-tick ORBIT model in the pre-reaction prediction setting over 50 epochs. The validation loss reaches its minimum around epoch 30 and then increases while the training loss continues to decrease, indicating overfitting in the later stages of training.<sup>3</sup>

2) EMBER: Initial experiments confirmed that the constructed dataset and the labeling pipeline produce a consistent and learnable signal. A classifier was trained on temporally aggregated Ego-Knowledge features, with input samples sampled at a purposefully higher granularity of history (0.5 s) and incorporating a short history window of up to two seconds prior to the reaction moment. Despite the simplicity of the model and the absence of hyperparameter tuning, training was stable and converged reliably across runs. The best achieved validation accuracy was 73.5%.

The player-disjoint test and its extended version indicate that the model is not learning a player-specific reaction prior and instead generalizes across players; this property can later be used to compare a player’s decision process against a modelderived baseline. The player-disjoint test yielded a validation accuracy (data from the same players as the training set) of 70.7%, while the test accuracy (players never seen in either training or validation) was 71.4%. The extended playerdisjoint test, conducted to check that adding more random players to the training and validation sets does not diminish this property, yielded validation and test accuracies of 73.3% and 73.7%, respectively. The slight increase of test over validation accuracy may reflect lower variability of source players in the test set (more players can introduce more outliers).

3) Language Models: The LLM baselines reach at most 62.7% accuracy over 1,000 validation predictions (Table VII); the configurations fall between 57.3% and 62.7%, and the same model can differ by several points between encodings and example budgets. The best-performing configuration is

Gemini 3.7 Flash with medium thinking and Narrative encoding at 62.7%, followed by GPT-5.6 Terra with Narrative encoding at 61.8% and Gemini 3.7 Flash with Full-Dump encoding and three quarters of the maximum example budget at 61.6%; the latter is the most expensive configuration in the comparison, so a larger example budget did not translate into higher accuracy. The open-source Qwen 3.5 9B and Qwen 3.8 27B models reach 59.0–59.9% at a small fraction of the cost. An earlier evaluation on a previous revision of the dataset, using the model generation available at the time, produced a best LLM accuracy of 63.5%; the LLM ceiling has therefore not moved with a newer generation of models, which is consistent with the limitation lying in the modality rather than in model capability.

In the earlier evaluation, Centaur 70B [62] and Minitaur 8B were included as an additional comparison layer because they are explicitly designed to model human choice behavior. Centaur was trained on Psych-101, a large collection of triallevel human responses from psychological experiments, and therefore represents a relevant baseline for evaluating whether a cognition-oriented foundation model transfers to the fightversus-flight setting. Both models performed poorly on our game-state dataset, reaching only 48.3% and 50.4% accuracy on the previous dataset revision, respectively, and their predictions showed a strong bias toward the Flight class; they were not re-evaluated on the current revision and are therefore not listed in Table VII. A likely explanation is the mismatch between training domain and target representation: Psych-101 consists primarily of natural-language descriptions of controlled psychological experiments and participant choices, whereas our dataset contains high-dimensional, temporally structured game-state observations with 154 numerical and categorical features describing health, movement, visibility, enemy proximity, aim alignment, and recent combat dynamics. These results suggest that cognition-oriented pretraining alone is insufficient for this setting unless the model is adapted to the specific structure, feature semantics, and class balance of the game-behavior dataset.

We additionally examined whether a very small open-source model could be adapted to the task via parameter-efficient fine-tuning: Qwen2.5-0.5B was fine-tuned on our labeled game states using LoRA (rank 8, learning rate $2 \times 1 0 ^ { - 4 } )$ ), framing the problem as a JSON-to-JSON mapping with no class weighting. The result was clearly negative: although the model learned to reproduce the expected JSON output structure, it collapsed to a constant prediction, emitting the Flight class for every input, so precision, recall, and $F _ { 1 }$ score were all effectively 0. We conclude that sub-billion-parameter models of this scale lack the capacity to recover a meaningful decision boundary for this setting.

The best Jev configuration (Score primitive, Narrative encoding restricted to 20 features, 104 labeled examples) reaches 63.4% accuracy on the same 1,000 rows, with precision 61.91% and recall 68.88%, at 0.48 s and \$1.27 per 1,000 predictions; on the same input the Noul and Choice primitives reach 62.3% and 61.9%. Across the configurations tried, accuracy ranges from 47.3% with no examples to 63.4%. Jev therefore sits at the top of the baselines that read a textual description of the situation, 0.7 percentage points above Gemini 3.7 Flash at roughly a quarter of its cost, and well below the models that consume the telemetry directly. It is not part of the paired tests in Section VIII-C; on the same rows EMBER and ORBIT lead it by 10.2 and 7.5 percentage points.

4) Majority-Class Baseline: The majority-class baseline yields accuracy equal to the percentual representation of the most common class, 51.5%, and gives perspective on how much the machine learning models improve on this naive baseline.

## C. Pre-Reaction Prediction Cross-Model Comparison

EMBER reaches 73.5% accuracy and ORBIT 71.2% on the full validation set; the strongest LLM configuration, Gemini 3.7 Flash with medium thinking and Narrative encoding, reaches 62.7% over 1,000 predictions drawn from the same validation set (Table VII). TabPFN provides a substantially stronger non-generative baseline: TabPFN v3.5 reaches 71.5% accuracy with a 25,000-sample context, 72.6% with 100,000, and 73.5% with the full 200,000+ context, matching EMBER on overall accuracy, but providing less balance between Type I and Type II errors. To compare like for like, EMBER and ORBIT were scored on the same 1,000 rows, where they reach 73.6% and 70.9%, respectively. McNemar’s test on the paired predictions gives $p \ = \ 1 . 3 \times 1 0 ^ { - 9 }$ for EMBER and $p = 1 . 5 \times 1 0 ^ { - 6 }$ for ORBIT against the strongest LLM configuration, and $p = 1 . 1 \times 1 0 ^ { - \hat { 2 } 3 }$ and $p = 1 . 3 \times 1 0 ^ { - 1 7 }$ against the majority-class baseline. The margin over the best LLM, roughly eleven percentage points for EMBER and eight for ORBIT, is therefore significant on identical trials under a test that treats the rows as independent (Section X). Additionally, in its full context, TabPFN v3.5 reaches the same 73.5% validation accuracy as EMBER, so the present results do not establish an accuracy advantage for EMBER over strong pretrained tabular models. Two caveats remain: the commercial LLMs were run with a bounded few-shot example budget (Section VII-E), and no confidence intervals are reported (Section X).

Both ORBIT and EMBER also provide substantially better operational efficiency: for ORBIT, inference is approximately 1,412× faster and 10,322× cheaper than Qwen 3.5 9B with Narrative encoding, the fastest and cheapest LLM configuration, corresponding to a 99.99% cost reduction; against the most accurate LLM configuration (1.81 s and \$4.70 per 1,000 predictions versus 0.34 ms and \$0.000062 for ORBIT) the ratios are roughly 5,300× and 76,000×. TabPFN occupies an intermediate operating point: the full-context TabPFN v3.5 reaches 73.5% accuracy at 0.215 s and 0.0429 per 1,000 predictions, making it roughly 8.4× faster and 110× cheaper than the most accurate LLM configuration, while ORBIT remains approximately 630× faster and 690× cheaper than this TabPFN setting. EMBER achieves the greatest decrease in both cost and latency. Jev, which targets low-latency decision loops, is among the faster and cheaper API-served options (0.48 s and \$1.27 per 1,000 predictions), yet ORBIT remains roughly 1,400× faster and 20,000× cheaper. As noted in Section VII-C, this is not a compute-matched comparison: the API-served figures are end-to-end wall-clock measurements including network and queueing time, and the gap is dominated by deployment topology. Results from the initial experiments using locally hosted models were excluded from the comparison due to unstable inference, given the available GPU resources, with an extreme bias toward a single prediction class.

Because the models consume Ego-Knowledge data only up to the tick of the stimulus (visibility or damage event), and the dataset was constructed with a clear attempt to minimize other stimuli affecting the current player action, only player data (position, rotation, and so on) has a truly temporal character in these experiments. The part of the data that might be most influential on the player’s decision (enemy position, for example) is essentially reduced to a single value. It is therefore not necessarily surprising that the boosted-tree architecture outperformed the transformer here. The learning process nevertheless worked well for the transformer as well, so for related future use cases with data of a more temporal character this architecture has a solid base and may later prove more valuable.

Table VII presents the comparison of all models on the chosen factors. Overall, the results suggest that the predictive signal in this task is contained primarily in the structured telemetry rather than in its natural-language representation. While large language models are able to recover part of this signal, they are consistently outperformed by models designed to operate directly on tabular inputs. Notably, EMBER matches the accuracy of the strongest foundation-model configuration while requiring only a fraction of its inference cost and latency, indicating that substantially greater model complexity does not translate into a corresponding performance advantage for this setting.

## D. Early Response Recognition Results

In the second experimental setting, we evaluated ORBIT and EMBER in the Early Response Recognition task defined in Section VI-B2, in which the input sequence may include a short post-anchor window, so that the model observes an increasing portion of the player’s actual reaction.

1) ORBIT: The history window was fixed to 7 ticks, the best configuration in the pre-reaction prediction experiment, and ORBIT models were trained with progressively longer future windows after the anchor tick: 4, 8, 12, and more future ticks, up to 64. Since CS2 demos are sampled at 64 ticks per second, the maximum future window corresponds to approximately one second after the anchor moment, matching the label definition. Accuracy increases consistently as more post-anchor ticks are included (Fig. 9), reaching 85.15% with the full one-second window.

The monotonic improvement is expected, because the label is itself defined by the behavioral response shortly after the anchor moment: as the model receives more of the unfolding reaction, the task gradually shifts from predicting the player’s likely response before it is visible to recognizing it from partial evidence of the behavior itself. These results should therefore be interpreted differently from the pre-reaction results. The experiment does not evaluate whether the model can predict a response before it starts, but how quickly and reliably ORBIT identifies the response once its first part becomes observable, and the consistent increase suggests that even a short postanchor context helps.

TABLE VII: Performance comparison of prediction strategies; the LLM and Jev rows are over 1,000 predictions from the current validation set.
<table><tr><td>Strategy</td><td>Accuracy (%)</td><td>Avg. time (s)</td><td>Total Cost ($)</td><td>Precision (%)</td><td>Recall (%)</td></tr><tr><td>Gemini 3.7 Flash Medium Narrative (1/8)</td><td>62.7</td><td>1.81</td><td>4.70</td><td>59.66</td><td>77.51</td></tr><tr><td>Gemini 3.7 Flash Medium Full-Dump (3/4)</td><td>61.6</td><td>6.31</td><td>60.03</td><td>58.19</td><td>81.33</td></tr><tr><td>Gemini 3 Flash Medium Full-Dump (1/8)</td><td>59.7</td><td>2.97</td><td>14.72</td><td>56.81</td><td>79.52</td></tr><tr><td>Gemini 3 Flash Medium Full-Dump (1/4)</td><td>57.3</td><td>3.28</td><td>15.97</td><td>55.42</td><td>72.89</td></tr><tr><td>Gemini 3 Flash Medium Full-Dump (3/4)</td><td>57.4</td><td>2.10</td><td>40.11</td><td>55.36</td><td>74.70</td></tr><tr><td>Gemini 3 Flash Medium Narrative (3/4)</td><td>60.3</td><td>5.07</td><td>16.98</td><td>57.69</td><td>76.10</td></tr><tr><td>Gemini 3 Flash Medium Compact (3/4)</td><td>61.1</td><td>4.74</td><td>23.64</td><td>58.61</td><td>74.50</td></tr><tr><td>GPT-5.6 Terra Medium Full-Dump (1/8)</td><td>57.7</td><td>3.32</td><td>29.76</td><td>55.71</td><td>73.49</td></tr><tr><td>GPT-5.6 Terra Medium Narrative (1/8)</td><td>61.8</td><td>1.42</td><td>14.21</td><td>59.48</td><td>73.09</td></tr><tr><td>Qwen 3.8 27B Full-Dump (32)</td><td>59.9</td><td>1.13</td><td>4.85</td><td>59.34</td><td>61.85</td></tr><tr><td>Qwen 3.8 27B Narrative (32)</td><td>59.0</td><td>0.92</td><td>2.18</td><td>62.09</td><td>45.38</td></tr><tr><td>Qwen 3.5 9B Full-Dump (32)</td><td>59.4</td><td>0.82</td><td>1.72</td><td>56.32</td><td>82.33</td></tr><tr><td>Qwen 3.5 9B Narrative (32)</td><td>59.0</td><td>0.48</td><td>0.64</td><td>61.52</td><td>47.19</td></tr><tr><td>Jev (Score) Narrative, 20 features (104)</td><td>63.4</td><td>0.48</td><td>1.27</td><td>61.91</td><td>68.88</td></tr><tr><td>TabPFN v2, context 25,000‡</td><td>67.7</td><td>0.028</td><td>0.0057</td><td>62.9</td><td>80.2</td></tr><tr><td>TabPFN v3.5, context 25,000‡</td><td>71.5</td><td>0.0089</td><td>0.0018</td><td>66.9</td><td>80.1</td></tr><tr><td>TabPFN v3.5, context 100,000‡</td><td>72.6</td><td>0.061</td><td>0.0122</td><td>68.7</td><td>78.2</td></tr><tr><td>TabPFN v3.5, full context (200,000+)‡</td><td>73.5</td><td>0.215</td><td>0.0429</td><td>69.6</td><td>79.0</td></tr><tr><td>Majority-class baseline*</td><td>51.5</td><td></td><td></td><td>NaN</td><td>0</td></tr><tr><td>EMBER*</td><td>73.5</td><td>0.00001</td><td>0.000002</td><td>70.97</td><td>76.12</td></tr><tr><td>ORBIT*</td><td>71.2</td><td>0.00034</td><td>0.000062</td><td>67.0</td><td>79.5</td></tr></table>

\*EMBER, ORBIT, and majority-class accuracies are over the full validation set; on the 1,000 rows used for the LLM evaluation they are 73.6%, 70.9%, and 50.2%, respectively (paired tests in the text). The 1,000 rows subset contains 502 flight and 498 fight instances. The value in parentheses gives the few-shot example budget: a fraction denotes the share of the maximum example count that fits the model’s context window, and an integer denotes a fixed number of examples. <sup>‡</sup> For TabPFN, context rows are chosen as a random subset from EMBER and ORBIT training set. Random validation subset contained 52.3% of the majority (flight) class. For full comparability, full context run (all training rows) is evaluated also on all validation rows.

![](images/f6b0c38d8749ad99ccc61e5c1de67366f088c84cd401e6ed2b68d062f145f444.jpg)  
Fig. 9: Validation accuracy of ORBIT in the early response recognition setting as the number of future ticks increases. The history window is fixed to 7 ticks, while the future window is extended up to 64 ticks, corresponding to approximately one second after the anchor moment.

2) EMBER: A similar experiment was carried out with EMBER, except that the whole history was used instead of the 7-tick history and training covered a grid of time steps and future ticks (the latter being the maximum tick difference from the stimulus, not necessarily the number of future ticks in the input). The results (Fig. 10) indicate that EMBER performs better with a higher time step and improves with each forward extension of the time frame. Part of this gain can be attributed to the task shifting from purely predictive to partly classificational; however, the reaction-time histograms in Section VIII-A show that only a very small portion of reactions occurs in the first 4 or even 8 ticks. It is therefore more likely that this result points to the reaction not being fully correlated with intention, as later stimuli can change the final decision from the initial intention.

## E. Explainability

Feature importance was computed with a perturbation-based procedure: each feature was replaced with a neutral baseline value and the resulting change in the predicted-class margin was measured. Fig. 11 shows that, for both predicted fight and predicted flight cases, ORBIT is most sensitive to features describing the player state, spatial context, weapon state, field of view, enemy configuration, and recent combat information, suggesting that predictions are driven by the interaction between the player’s local situation and the geometry of the nearby engagement rather than by a single isolated variable. Fight predictions are more associated with the immediate engagement context, whereas flight predictions also reflect cues related to threat and disengagement, such as recent damage history. Overall, the plots support the intended role of Ego-Knowledge as a compact representation of the local tactical context. Full rankings are given in Fig. 15 (ORBIT, predicted fight) and Fig. 16 (EMBER) in the Appendix.

![](images/eb4a909562e6414e901bfea1a76a293aa8208f75229843d2d0eedf588088b0ae.jpg)  
Fig. 10: Validation accuracy of EMBER in the early response setting as a function of future time steps (maximum difference in ticks between stimulus and the latest moment in the model input) and time step (in ticks). The best accuracy for each future time step is highlighted in red.

These results should still be interpreted as model sensitivity diagnostics rather than causal explanations. Some inputs, such as weapon and spatial identifiers, are represented through learned embeddings, so their importance reflects ORBIT’s sensitivity to the learned representation rather than a direct physical intervention in the match.

In all LLM models, the main determinants of fight situations are crosshair proximity, weapon readiness, and enemy awareness; six recurring factors appear in the models’ reasoning (crosshair distance, weapon equipped, visibility and awareness, movement velocity, temporal cues such as recent firing and weapon switches, and health and team status), and Table XI in the Appendix lists the identified fight and flight patterns together with the error patterns behind prediction mistakes. Larger commercial models (OpenAI and Gemini, and DeepSeek in the earlier evaluation) prioritize tactical advantages and aim alignment, while smaller local models often collapse into repetitive templates or meta-commentaries; a recurring systemic bias is to predict fight whenever an enemy is visible, often failing to recognize tactical retreats or “saving” behaviors.

## F. Label Validation by Majority Voting

A subset of situations was selected for manual labeling as described in Section VI-A2, chosen so that external influences were expected to be limited and the timing of the stimulus was well defined. Each situation was labeled independently by seven annotators with advanced knowledge of CS2 gameplay, who could assign a response label or leave the situation unresolved when the available evidence was insufficient or ambiguous.

A gold-standard subset, containing exclusively damage events, was constructed by majority voting: a situation received a gold-standard label only if more than 50% of annotators agreed on the same response class. Of the 200 manually inspected situations, 165 received a majority-vote label, a consensus coverage of 82.5%. Consensus coverage is the proportion of moments on which a majority was reached; with seven annotators and a binary choice, a majority is near-certain unless annotators abstain, so it principally reflects abstention and is not an agreement statistic. The agreement statistic is Fleiss’s kappa: 0.38 for the entire subset of 200 moments and 0.44 for the gold-standard set, corresponding to fair (0.38) and moderate (0.44) agreement on the conventional Landis and Koch interpretation. Agreement of this level bounds the accuracy achievable by any model trained against these labels, so the reported model accuracies should be read against that ceiling rather than against 100%.

Table VIII summarizes the validation: agreement between the automatic labeling procedure and the gold-standard set is 80.6%. A subset (119 instances) of situations with majorityvote label which also fit the dataset filtering conditions, used as input to EMBER, did not yield significantly different results from the rest of the validation set (74.8%, which given the gold-standard set size is comparable to the whole-validation 73.5%). The gold-standard set is subject to expansion: it comprises damage events only and does not yet include the majority event type, visibility, which poses more challenges for correct filtering.

TABLE VIII: Summary of majority-vote label validation. Agreement with the automatic labeling system is computed on the majority-vote gold-standard subset.
<table><tr><td>Quantity</td><td>Value</td></tr><tr><td>Manually inspected situations</td><td>200</td></tr><tr><td>Annotators per situation</td><td>7</td></tr><tr><td>Majority-vote threshold</td><td>&gt; 50%</td></tr><tr><td>Situations with majority-vote label</td><td>165</td></tr><tr><td>Number of situations used for EMBER validation</td><td>119</td></tr><tr><td>Consensus coverage</td><td>82.5%</td></tr><tr><td>Automatic label agreement with gold-standard set</td><td>80.6%</td></tr><tr><td>EMBER accuracy on gold-standard set</td><td>74.8%</td></tr></table>

## G. Qualitative Case Studies

To complement the quantitative evaluation, we analyze four qualitative case studies from the validation set, two correctly classified and two failures, all from the middle area of the map Mirage. The analyzed player is the orange Terrorist side player and the opponent the blue Counter-Terrorist side player; for the two cases shown, the first image is the anchor moment at which the prediction was made and the second the same situation approximately one second later. These are illustrative examples, not a statistical evaluation.

ORBIT: Top 10 Global Feature Importances for Predicted Fight and Flight Cases (A) Predicted Fight Cases  
![](images/4211f935c535983a460144f415130216dbed18496abf58b1d6ed38fe849e51c0.jpg)

(B) Predicted Flight Cases  
![](images/7f4fa046f0fbe1d16948f319626807bb6b4be6c271e3ed928d8d6187560c37d9.jpg)  
Feature importance (mean absolute class-margin change)

Fig. 11: Global perturbation-based feature importance of ORBIT for samples predicted as fight and flight. Importance is measured as absolute change in predicted-class margin, so the plot should be interpreted as model sensitivity rather than causal evidence.  
![](images/9865e010b5fd7483fbc2ebcc98f28ba995ac12aed320f59cd2475c11873bd61e.jpg)  
(a) Anchor moment.

![](images/95bb6444467358651f6ff6b72d8d0722b6ebc58ce2a6fb6144d89523f93b2aea.jpg)  
(b) Approximately one second later.  
Fig. 12: Correctly classified fight example: the analyzed player spots the opponent on lower middle and commits to the duel.

Fig. 12 shows a correctly classified fight. At the anchor moment (Fig. 12a) the analyzed player spots the opponent on lower middle and, instead of disengaging, takes the duel (Fig. 12b). The example comes from the pistol round, where weapon strength is typically comparable, so committing to the fight is a plausible tactical decision. In the correctly classified flight case (not shown), the analyzed player, holding a lighter sniper rifle, identifies that the window is controlled by an opponent with an AWP on a favorable angle and disengages, as the model predicts.

In the first failure case (not shown), the model predicted fight for a player holding an AWP against an M4 rifle, a favorable long-range matchup, who after missing the first shot hides and reloads rather than re-engaging; the model appears to interpret the weapon advantage and initial engagement as evidence for fight and fails to capture the subsequent disengagement.

Fig. 13 shows the opposite failure: the model predicted flight, but the player continued into the duel. At the anchor moment (Fig. 13a) the analyzed player, with an AK-47, is exposed to an AWP in a window, a risky engagement because the AWP player controls a strong angle and, even after missing the first shot, can often retreat behind cover, reload, and repeek; the player nevertheless commits to the fight (Fig. 13b).

![](images/ed1f951161a6f5aad9c3764c0ec13949dc8bfa2c127a7036f2b93893a551604b.jpg)  
(a) Anchor moment.

![](images/d584280e3844b02def3bf2944c4d7dd30c6c80a5ca4532d9fb9b5f5bb61bfa44.jpg)  
(b) Approximately one second later.  
Fig. 13: Incorrectly classified flight example: the analyzed player, with an AK-47, faces an AWP in a window but continue the duel.

Such cases are difficult because the correct label depends not only on the objective geometry and weapon matchup, but also on the player’s willingness to take a high-risk duel.

Overall, the examples show that ORBIT can correctly identify both engagement and disengagement when the player’s behavior is consistent with the local tactical context. The failure cases, however, suggest that some decisions depend on factors only partially captured by the current representation, such as risk preference, confidence in a weapon matchup, or the intention to re-peek after temporarily taking cover.

## IX. DISCUSSION

The reaction-time result of Section VIII-A should be read conservatively. Its value lies relative to the proposed 60– 120 ms window: because the extraction procedure enforces a minimum measurable latency of approximately 46.875 ms, the accumulation of a non-trivial number of events near that boundary before the distribution rises into slower ranges shows that the data contain events at the boundary between fast anticipatory-confirmatory action and more fully elaborated tactical response, with the qualification developed in Section X-A that the near-floor cluster itself is better explained by motion already underway than by genuine reactions to the shot. This does not isolate literal limbic discharge, nor does it validate a specific brain structure or a one-to-one fight-or-flight mapping. What it validates is more modest: a measurable, interpretable, and highly non-random short-latency response regime with a plausible spread of fast, intermediate, and slow response states, the condition under which Mesocort seeks to differentiate rapid reactive control from slower tactical and strategic behavior.

Several caveats bound this interpretation. The earliest bins may reflect a mixture of anticipatory readiness, partial preorientation, predictive coding, and rapid sensory confirmation rather than a single mechanism; the slower bins may contain genuine freezing or hesitation alongside residual confounds not yet removed by filtering. The right-skewed shape itself is not diagnostic, since skewed and ex-Gaussian-like distributions are common across many reaction-time tasks [56]–[58]; we therefore do not treat the histogram shape as evidence of a unique multi-regime mechanism, and stronger claims would require both a control distribution from expectedcontact events and formal model comparison using mixture or diffusion-style analyses.

The broader contribution of the current phase is therefore primarily methodological. We show that it is feasible to transform omniscient replay data into a player-centric representation, extract interpretable reaction-centered events, and train predictive models that capture regularities in shorthorizon behavior. The neuroscience reviewed here informed the architectural choices (signal selection, model structure, labeling distinctions), but their validation remains empirical: whether the resulting architecture predicts human behavior under pressure. This does not constitute direct measurement of limbic or subcortical processes, nor does it justify strong one-to-one biological claims; it does show that ecologically rich digital environments support measurable, scalable study of pressure-sensitive behavioral organization. Recent EEG findings in esports, showing that the neural state before a match carries predictive information about competitive outcomes, are consistent with this, although our pipeline relies on behavior alone [63], [64].

If the model captures real structure in how humans behave under sudden threat, especially under out-of-player generalization, then applying it to a new domain becomes primarily a data and validation problem rather than a claim that neural pathways are identical across domains. The same modeling structure (ego-centric information reconstruction, threatimminence-structured control-mode inference, and a multitimescale behavioral hierarchy) could in principle be trained on telemetry from driving simulators, surgical training systems, or VR platforms. The transfer question is therefore whether the architecture learns predictive behavioral structure from new data: neuroscience informs the architecture, predictions validate it.

One inferential challenge must be stated explicitly. The claim that fast behavioral responses in CS2 reflect “limbiclike” processing is a form of reverse inference: reasoning from an observed behavior back to a putative neural mechanism [65], [66]. Poldrack showed that such inferences are valid only when the neural structure is selectively engaged by the task, which behavioral data alone cannot establish. No pattern in gameplay telemetry can yet distinguish limbiclike processing from fast cortical processing, trained automaticity, anticipatory motor preparation, or strategic preorientation without concurrent physiological evidence. The neuroscience vocabulary in this paper should therefore be read as a hypothesis-generating framework rather than as direct evidence of a mechanism.

What the behavioral data can establish is the temporal and structural organization of pressure-sensitive responses (when they occur, how they vary across individuals, and what contextual factors predict them), the prerequisite for future studies pairing the Mesocort pipeline with concurrent EEG, pupillometry, eye tracking, or electrodermal measurement. Mesocort’s broader significance is thus to make the “limbiclike” hypothesis computationally testable: a robust expectation model could identify atypical responses and learn playerspecific response profiles under uncertainty.

## X. LIMITATIONS

## A. Latencies in Counter-Strike 2

Reaction latencies are measured from server-side demo files, yet the behavior they describe was produced by a player acting on a client-side view of the match. The server records the game state 64 times per second and the demo preserves this record, whereas each client renders a predicted, networkdelayed, interpolated reconstruction in which the local player is predicted forward to hide input latency while every other entity is interpolated from earlier server updates, leaving the local player roughly one round-trip ahead of the enemies around them [67]. The residual latency measurably degrades player performance [68]–[70]; the peeker’s advantage, governed more by the latency of the player being shot at than by that of the shooter [71], puts a player shot from behind on the disadvantaged side. A demo-measured reaction time therefore contains a purely technical delay.

Two delays bracket the player’s own reaction time. $\Delta _ { \mathrm { s t i m } }$ is the delay before the player can see the event: the update must reach the client, be rendered, and be displayed. For an enemy coming into view there is an extra interpolation term: the demo stores the enemy at server tick $t _ { N } ,$ , but the screen showed an earlier position (around $t _ { N - 2 }$ to $t _ { N - 1 }$ at 64 Hz), so using t as the stimulus time starts the clock before the enemy was visible. For incoming damage, noticed from sound and damage feedback, this term does not apply. $\Delta _ { \mathrm { r e s p } }$ begins after the human reaction and covers mouse polling, the trip back to the server, and rounding to the next recorded tick. Both are strictly positive, so the demo always reports a reaction time longer than the real one: $t _ { \mathrm { m e a s u r e d } } \approx t _ { \mathrm { t r u e } } + \Delta _ { \mathrm { s t i m } } + \Delta _ { \mathrm { r e s p } } .$

Table IX gives order-of-magnitude estimates for typical European competitive conditions (a few hundred frames per second, a 144–240 Hz monitor): roughly 40–80 ms when the trigger is incoming damage and 60–110 ms for an enemy coming into view, the difference being the interpolation term. Sub-tick input timing does not help: the server timestamps inputs to a fraction of a millisecond, but the demo is still written once per 64 Hz tick [72], so the view-angle stream used to find the onset is rounded to the tick. On top of this average offset sits a per-event jitter of tens of milliseconds (tick rounding on both sides, frame-rate changes, drifting ping) that no fixed correction can remove.

TABLE IX: Estimated CS2 system-latency contribution to demo-measured reaction time (typical European competitive conditions). Values are order-of-magnitude estimates, not measurements.
<table><tr><td>Component</td><td>Est. (ms)</td></tr><tr><td> $\Delta _ { \mathrm { s t i m } } \cdot s e r \nu e r \to p l a y e r$ </td><td></td></tr><tr><td> $\mathrm { S e r v e r } \to \mathrm { c l i e n t ~ ( o n e \mathrm { - } w a y ) }$ </td><td>12-18</td></tr><tr><td>Client processing + render pipeline (200–400 FPS)</td><td>6-15</td></tr><tr><td>Display (144–240 Hz scan-out + pixel response)</td><td>5-12</td></tr><tr><td>Interpolation (enemy-in-view trigger only)</td><td>16-31</td></tr><tr><td> $\Delta _ { \mathrm { r e s p } } \colon p l a y e r  d e m o$ </td><td></td></tr><tr><td>Mouse polling + client → server (one-way)</td><td>13-23</td></tr><tr><td>Quantization to the recording tick</td><td>0-16</td></tr><tr><td>Total delay, shot from behind</td><td>≈40-80</td></tr><tr><td>(stimulus without interpolation + response)</td><td></td></tr><tr><td>Total delay, enemy-in-view (stimulus with interpolation + response)</td><td>≈60-110</td></tr></table>

Note: $\Delta _ { \mathrm { s t i m } }$ is the delay before the player can perceive an event the server has recorded; $\Delta _ { \mathrm { r e s p } }$ is the delay before the player’s reaction is recorded back into the demo. Network terms vary widely across players with connection quality (e.g. wired versus wireless) and distance to the server.

These delays change how the back-shot histogram (Fig. 6) should be read. The large group of reactions in the fewhundred-millisecond range survives the correction. The cluster near the measurement floor (≈ 46.875 ms) does not: subtracting a 40–80 ms system delay would put the true reaction time at zero or below, impossible as a response to the shot itself, so these near-floor cases are better explained by movement already underway (anticipation, or leftover mouse motion from an earlier event, Section VI-C) than by genuinely fast reactions. Because the system delay plus jitter is about as large as the 60–120 ms window itself, no single demomeasured latency can show that a reaction fell inside that window. Since the delay grows with ping, which the demo records every tick, filtering to players with low, stable ping would shrink both the average delay and its spread.

Fig. 14 shows the histogram after subtracting the measured ping and the estimated delay bands, exposing the incorrectly included data points. The bands differ from Table IX because each player’s exact ping replaces the approximated value. The distribution is not significantly changed, but the case for improved visibility-triggered event filtering arises again. It is purposefully built from all collected reactions rather than the more heavily filtered model dataset, so the two differ in sample size.

## B. Statement of Limitations

None of the following invalidates the findings, but they shape their interpretation.

Few-shot example budget. For cost reasons the commercial LLMs were mostly run with one eighth of the few-shot examples that fit their context window; the single run at three quarters of the budget (Gemini 3.7 Flash, Full-Dump) did not improve accuracy, but the LLM figures should be read as a bounded rather than exhaustive search over prompting configurations. The paired comparisons are on a 1,000-row sample of the validation set, not the full set. The Jev comparison is bounded by the model’s context limit (at most 120 in-context examples) and is not included in the paired tests.

![](images/2b365cd1c292b6894b26cf8b0bd62f1a2242a8bfc78d1059e52fb274f39a9c39.jpg)  
Fig. 14: Visualization of reaction time transformation based on estimated unmeasured delay and measured ping. Top row = raw reaction time; bottom row = reaction time minus ping and lower band, with shifted upper band. Minimum reaction time filter before adjustment: 46.875 ms (3 ticks)

Model selection and held-out evaluation. Models were selected on validation loss and the headline accuracies are reported on the same validation set; no separate held-out test set was reserved for the main experiments (the player-disjoint experiments are the exception), so the reported figures should be read as an upper bound on generalization.

Interval estimates. The headline accuracies are point estimates without confidence intervals, and no retraining variance across seeds is reported; paired significance tests are given for the headline comparisons only, so differences of a few percentage points between LLM configurations in Table VII should not be treated as established.

Independence of observations. Reaction moments are nested within players and matches, whereas McNemar’s test treats the 1,000 sampled rows as independent. The reported pvalues may therefore overstate the evidence to an extent that has not been quantified; a test clustered by player and match is left to future work.

Label reliability. Inter-annotator agreement is Fleiss’s kappa κ = 0.38 (200 moments) and 0.44 (gold set), fair and moderate on the Landis and Koch scale. This bounds the accuracy achievable by any model trained against these labels, so the reported accuracies should be read against that ceiling rather than against 100%; part of the residual error may reflect genuine ambiguity.

Consensus coverage. The 82.5% consensus coverage is the proportion of moments on which a majority was reached. With seven annotators and a binary choice, a majority is near-certain unless annotators abstain, so it principally reflects abstention and is not an agreement statistic; κ is.

Label circularity. Fight-or-flight labels are coarse operational approximations derived from the same telemetry stream as the features, so classifier performance partly reflects internal consistency. The gold-standard check (80.6% agreement) mitigates but does not remove this, and the gold set covers damage events only.

Temporal resolution and system latency. The nominal 64 Hz tick (15.625 ms) bounds the resolution of every measurement. Measured latencies are upper bounds on true reaction times, inflated by roughly 40–80 ms of system delay plus per-event jitter of tens of milliseconds (Section X-A), and the 46.875 ms floor introduces left-censoring that may inflate apparent clustering at short latencies.

Sequence-history behavior. The degradation with longer history windows in Table VI is not expected theoretically and is most likely a training artifact; no conclusion depends on it.

Absent baselines and controls. The back-shot histogram lacks a control distribution from non-surprise contacts, preventing claims of threat-specificity, and its unimodal versus multimodal explanations are not formally tested by mixture or diffusion-style modeling [56]–[58]. Furthermore, ORBIT has not been compared against feature-ablated naive baselines, limiting how far its loss curves evidence generalization.

Scope. The data come from a single game, a single heavily filtered scenario, and demos selected around a core cohort of 50 players, and may not transfer to other environments without revalidation. Auditory information is not yet fully reconstructed, and the Ego-Knowledge layer has not been validated against perception-oriented ground truth such as eye tracking. There is no physiological ground truth, so references to limbic-like behavior remain inferential (reverse inference) until concurrent neuroimaging or psychophysiological measurement is available [65]. Neuroimaging of experienced FPS players suggests tonic amygdala suppression during sustained gameplay [28], [29], which constrains the expected signal to transient event-locked responses. Event extraction may also be confounded by skill, role, map knowledge, automaticity, or contextual priors not yet represented explicitly [27].

Latency bands and mechanism. Reaction-time bands are named by latency; the mapping from latency to control mode is a hypothesis under test, and no claim is made that a band identifies a mechanism.

Architecture. Hand-engineered features and behavioral cloning are substantially behind the frontier of world-model architectures; the roadmap in Section XIII addresses this through learned encoders and action-conditioned prediction.

## XI. ETHICS AND DATA PROTECTION

Nature of the data. This work is a secondary analysis of recorded gameplay telemetry from Counter-Strike 2 matches, extracted from server-side GOTV demo files. No participants were recruited and no intervention took place. The demo files were obtained from two sources. The primary source is the Rankacy platform, where users upload or record their matches and, under the platform’s Terms of Service and Privacy Policy, consent to the processing of their gameplay data, including derived analysis and use for research and scientific purposes.

A second portion of the dataset was purchased from thirdparty operators who ran their own game servers and generated the demo files on their own infrastructure; these datasets were acquired under contractual terms permitting retention, derived analysis, and publication of derived, pseudonymized results. No demo files were scraped from third-party services without authorization.

Controller and legal basis. The data controller is JMC a.s. (Czech Republic), which operates the Rankacy platform and conducts the research described here under the Mesocort AI name; the applicable terms are published at rankacy.com. Processing for this research relies on Article 6(1)(f) GDPR (legitimate interests), documented in a Legitimate Interests Assessment, alongside the contractual and consent framework of the platform’s Terms of Service, and is carried out with the safeguards for scientific research under Article 89(1) GDPR, principally pseudonymization and data minimization. A Data Protection Impact Assessment under Article 35 GDPR covering behavioral profiling research on gameplay telemetry has been carried out and is maintained by the controller.

Identifiers and pseudonymization. Player identifiers are replaced by sequential labels (player 1, player 2, . . . ) in all published material and in any released dataset. Identifiers are never perturbed, because platform identifiers occupy a contiguous number space in which an altered identifier resolves to a different real account. The identifier-to-label mapping is retained by the controller under restricted access, solely to support scientific validation, correction, and the exercise of data-subject rights; the data are therefore treated as pseudonymized personal data and the GDPR applies in full. Longitudinal behavioral traces also carry a residual reidentification risk, which is a further reason the data are described as pseudonymized rather than anonymous.

Ethics review. No institutional ethics committee review was sought for this work. The study is a secondary analysis of previously recorded gameplay telemetry processed under the contractual and data-protection framework described above; no participants were recruited, no intervention or interaction with individuals took place, and no special categories of personal data under Article 9 GDPR were processed. The research was conducted by a commercial entity outside a mandatory institutional review framework, and oversight was provided through the controller’s internal data-protection process. Future phases involving physiological measurement (EEG, eye tracking) will be conducted with academic partners under their institutional ethics approval.

Transparency. The individuals whose gameplay appears in the dataset were not contacted individually about this study. Users of the Rankacy platform were informed through the platform’s Terms of Service and Privacy Policy at the point of collection that their data may be processed for research and scientific purposes (Article 13 GDPR). For records originating from purchased third-party datasets, individual notification would require disproportionate effort within the meaning of Article 14(5)(b) GDPR; in line with that provision, the safeguards of Article 89(1) are applied, and this publication, together with the information available at rankacy.com, serves as the public information measure.

Intended use and misuse. Because this line of research concerns behavior prediction and behavioral profiling, it requires explicit safeguards around privacy, data governance, fairness, uncertainty reporting, and misuse prevention. A system that models pressure-sensitive or player-specific response tendencies may be valuable for training and simulation, but it also carries risks if used for manipulative personalization, covert profiling, or surveillance. The models described here are developed for research purposes and are not intended for individual-level profiling of identifiable persons.

## XII. CONCLUSION

This work positions Mesocort as an effort to model a fast, pressure-sensitive layer of human behavior from highfrequency digital traces. The strongest near-term claim is not that the system reads the limbic system directly, but that it can detect and organize behavioral proxies of rapid, highpressure responses; the neuroscience reviewed here informed the architecture, and its validation remains empirical.

The two evaluation regimes are complementary. Under strictly pre-reaction information, the fight-or-flight outcome is predictable well above the 51.5% majority-class baseline: EMBER reaches 73.5% and ORBIT 71.2% on the full validation set, and both generalize to held-out players. The strongest LLM baseline reaches 62.7% on 1,000 predictions from the same validation set; on those same trials EMBER (73.6%) and ORBIT (70.9%) outperform it with paired McNemar $p = 1 . 3 \times 1 0 ^ { - 9 }$ and $p = 1 . 5 \times 1 0 ^ { - 6 }$ , respectively, and the best LLM accuracy did not improve across a generation of newer models. Both telemetry-native models predict at several orders of magnitude lower inference time and cost, although that comparison is not compute-matched. In Early Response Recognition, accuracy rises monotonically with added postanchor ticks, reaching 85.15% over the full ≈1 s window. Part of this increase is mechanical, since the label is defined by post-anchor behavior; the informative quantity is the shape of the curve, along which a fast reactive component is expected to dominate early and a slower, deliberative one to assert itself later, sometimes revising the early intention. Marginal gains per added tick are roughly uniform, so this remains a hypothesis rather than a result the data establish.

These results were obtained on a deliberately small slice of the available data (demos selected around a 50-player core cohort and one heavily filtered surprise scenario) and are a Phase 1 proof of feasibility rather than a converged figure. The two models perform comparably, with EMBER slightly ahead: leakage minimization leaves little temporal signal at the anchor, and ORBIT overfits well before the end of training (Figure 8), a symptom of data-hungry attention-based models in this low-data regime [60], [61]. We nonetheless prioritize the transformer direction deliberately: the goal is a single model spanning many behavioral use cases, where shared representations and transfer matter more than singletask accuracy, and that direction depends on larger scale and richer temporal inputs, under which attention-based models are expected to gain. ORBIT’s stable training is read as a base for those settings rather than present superiority; EMBER remains a strong, low-cost model where temporal context is uninformative.

Mesocort complements environment world models and language models by targeting a different layer: not how environments evolve, nor deliberative reasoning and communication, but embodied, individual-specific, pressure-sensitive dynamics learned from telemetry. The LLM comparison, now paired on identical trials, is consistent with native telemetry models capturing variance in the fast-response regime that textbased general-purpose models are structurally poorly suited to access, although it is not compute-matched. The longterm vision, a validated human behavior prior populating world-model simulations with realistic agents, depends first on proving the prediction under out-of-player generalization and on resolving the reverse-inference constraint through physiological measurement.

## XIII. FUTURE WORK

The present study establishes a Phase 1 baseline; the most consequential next steps extend the evidence base, input signal, data scale, and behavioral phenomena captured.

1) Confidence intervals, held-out test set, and further baselines. Reporting confidence intervals for every headline figure, reserving a held-out test set separate from the validation set used for model selection, and adding a pretrained non-LLM tabular baseline such as TabPFN together with a compute-matched, locally served LLM comparison and a full, paired evaluation of decision models such as Jev.

2) Richer sensory reconstruction (auditory channel). Phase 1 reconstructs vision but not sound. Footsteps, gunfire, reloads, and grenade audio are primary anticipatory cues in Counter-Strike 2, and their absence confounds reaction-onset measurement, since a visually attributed response may have been sound-cued earlier. An auditory layer would remove this confound and open the study of anticipatory behavior.

3) Scaling the dataset. Results rest on a constrained set of heavily filtered surprise events, limiting statistical power for control distributions from non-surprise contacts, playerdisjoint generalization, and rare edge cases such as freezing. Larger coverage is a prerequisite for inferential rigor and the foundation-model direction below, and must include the gold-standard subset.

4) Reducing the system-latency confound. Reaction times read from demo files are upper bounds on true latency, inflated by a strictly positive engine-and-network delay of roughly 40–80 ms for back-damage events (Section X-A). Because the demo records each player’s ping every tick, filtering to low, stable connections would shrink the average offset and its per-event jitter without new data.

5) Strengthening the ORBIT evaluation protocol. ORBIT results do not yet report non-learned baselines beyond the majority class, nor a fully documented strict-split strategy. Future iterations will add naive and feature-ablated baselines and an explicitly documented player- and matchdisjoint splitting protocol.

6) A telemetry-native foundation model for Counter-Strike 2. We intend to pretrain a self-supervised foundation model on unlabeled CS2 telemetry at scale, with a learned agent-centric encoder replacing hand-engineered features and the visibility engine as a training-time leak filter, serving as a shared backbone for downstream tasks and as the substrate for upgrading ORBIT from behavioral cloning to an action-conditioned, counterfactual agent world model.

7) Modeling the transition from fast to deliberative control. Dual-route accounts described a subcortical “low road” preceding a cortical “high road” [6], [17], but that binary has since been superseded: LeDoux and Daw recast defensive behavior as a hierarchy of controllers within a single survival circuit [73], and Pessoa and Adolphs argue for distributed relevance evaluation rather than a privileged shortcut [3]. We therefore name our bands by latency rather than by pathway, and treat the mapping from latency to control mode as the hypothesis under test. The target is the handoff between the fast and slower bands, modeled explicitly via, for instance, a switching-state-space formulation gated by threat-imminence features. A first test needs no new recordings: retrain with the fastest correctedlatency band removed from the training set, compare against removing an equal number of slower events, and measure the change on the full validation set, starting with damage-triggered events, where the stimulus anchor is cleanest. If accuracy does not move, the fast band carries no predictive signal of its own and the fast-response framing should be narrowed accordingly.

8) Validation, physiology, and transfer. In parallel we will pursue a larger gold-standard set to reduce label circularity, formal statistical treatment of the reaction-time distribution (control conditions, mixture- and diffusion-style fits), synchronized physiological measurement (EEG, pupillometry, electrodermal) where partnerships allow, to address the reverse-inference constraint, and architectural transfer to at least one non-gaming pressure domain.

## XIV. DATA AND CODE AVAILABILITY

The trained models (EMBER and ORBIT), the extracted model-input data underlying the reported results, and the evaluation protocol are available on reasonable request to qualified research groups for validation and replication purposes, subject to a research data-sharing agreement preserving pseudonymization and prohibiting re-identification. The raw demo corpus, the parsing and Ego-Knowledge reconstruction pipeline, and the visibility engine are proprietary and are not released, as they form part of Mesocort AI’s commercial platform. Data are otherwise available from the corresponding author on reasonable request.

## XV. COMPETING INTERESTS

Several authors are employees, contractors, or shareholders of Mesocort AI, which operates within the JMC a.s. group and is developing commercial applications of the models described in this paper. The authors declare no other competing interests.

## XVI. FUNDING

This research was funded entirely by private capital of the JMC a.s. group; no public grants or external funding were used.

## XVII. ACKNOWLEDGMENTS

We thank Dr. Marcel Binz for his thorough review of this work and for his valuable comments and suggestions.

## APPENDIX

## A. Input features used by the ORBIT model in detail

TABLE X: Description of the input features used by the model. All features are related for the current game tick.
<table><tr><td>Feature</td><td>Description</td><td>Encoding / scaling</td></tr><tr><td colspan="3">Round and map context</td></tr><tr><td>Premier rank</td><td>Player Premier rank.</td><td>Min-max scaler</td></tr><tr><td>Round number</td><td>Current round index.</td><td>Embedding layer</td></tr><tr><td>Map ID</td><td>Identifier of the map being played.</td><td>Embedding layer</td></tr><tr><td>Bomb planted</td><td>Indicates whether the bomb is currently planted.</td><td>Boolean</td></tr><tr><td>Remaining round time</td><td>Remaining time in the current round.</td><td>Standard scaler</td></tr><tr><td colspan="3">Player position and state</td></tr><tr><td>Player square ID</td><td>Discretized player position on the map. The map space is divided into</td><td>Embedding layer</td></tr><tr><td>Square local X</td><td>squares, each represented by a pretrained embedding. Player&#x27;s local X coordinate inside the current square</td><td>Min-max scaler</td></tr><tr><td>Square local Y</td><td>Player&#x27;s local Y coordinate inside the current square.</td><td>Min-max scaler</td></tr><tr><td>Global Z position</td><td>Player&#x27;s vertical position on the map.</td><td>Per-map min-max scaler</td></tr><tr><td>Flash duration</td><td>Duration for which the player is blinded by a flashbang</td><td>Standard scaler</td></tr><tr><td>View offset Z</td><td>Height of the player&#x27;s head or view origin.</td><td>Min-max scaler</td></tr><tr><td>Alive</td><td>Indicates whether the player is alive.</td><td>Boolean</td></tr><tr><td>Health</td><td>Current health of the player.</td><td>Min-max scaler</td></tr><tr><td>Team</td><td>Indicates whether the player belongs to the T or CT team.</td><td>Boolean</td></tr><tr><td>Pitch</td><td>Vertical angle of the player&#x27;s view direction.</td><td>Sine transform</td></tr><tr><td>Yaw cosine</td><td>Cosine component of the player&#x27;s yaw angle.</td><td>Cosine transform</td></tr><tr><td>Yaw sine</td><td>Sine component of the player&#x27;s yaw angle.</td><td>Sine transform</td></tr><tr><td>Weapon ID</td><td>Identifier of the currently equipped weapon.</td><td>Embedding layer</td></tr><tr><td>Two-dimensional velocity</td><td>Player velocity in the horizontal plane, excluding the Z component.</td><td>Min-max scaler</td></tr><tr><td>Field of view</td><td>Player field of view.</td><td>Min-max scaler</td></tr><tr><td>Has defuser</td><td>Indicates whether the player carries a defuse kit.</td><td>Boolean</td></tr><tr><td>Has bomb</td><td>Indicates whether the player carries the bomb.</td><td>Boolean</td></tr><tr><td>Number of team defusers</td><td>Number of defuse kits available in the player&#x27;s team.</td><td>Min-max scaler</td></tr><tr><td>Defusing time</td><td>Time for which the player has been defusing the bomb.</td><td>Standard scaler</td></tr><tr><td>Bomb planting time</td><td>Time for which the player has been planting the bomb.</td><td>Standard scaler</td></tr><tr><td>Team has bomb</td><td>Indicates whether the player&#x27;s team currently controls the bomb.</td><td>Boolean</td></tr><tr><td colspan="3">Aiming and visibility</td></tr><tr><td>Angular velocity</td><td>Change in the player&#x27;s view angle between the previous tick and the current tick. This feature captures the speed of mouse or view movement.</td><td>Power transformer</td></tr><tr><td>Aim direction state</td><td>Discrete state describing the player&#x27;s aim direction.</td><td>Embedding layer</td></tr><tr><td>Aim direction changed</td><td>Indicates whether the player&#x27;s crosshair direction changed from the previous tick.</td><td>Boolean</td></tr><tr><td>Closest enemy to crosshair,</td><td>Euclidean distance between the crosshair direction and the closest enemy.</td><td>Power transformer</td></tr><tr><td>Euclidean distance Closest enemy angular</td><td>Angular distance between the crosshair and the closest enemy.</td><td>Power transformer</td></tr><tr><td>distance Closest enemy Euclidean</td><td>Euclidean distance to the closest enemy.</td><td>Power transformer</td></tr><tr><td>distance Number of visible enemies</td><td>Number of enemies visible to the player.</td><td></td></tr><tr><td>Some enemies visible</td><td>Indicates whether at least one enemy is visible.</td><td>Min-max scaler Boolean</td></tr><tr><td>Some enemies visible,</td><td>Indicates uncertain enemy visibility, for example due to smoke boundaries or</td><td>Boolean</td></tr><tr><td>uncertain</td><td>partial occlusion. Number of ticks for which an enemy has been continuously visible.</td><td></td></tr><tr><td>Enemy visible duration</td><td></td><td>log(1 + x) + standard scaler</td></tr><tr><td></td><td>Number of ticks since an enemy was last visible.</td><td>log(1 + x) + standard scaler</td></tr><tr><td>Has seen enemy in round</td><td>Indicates whether the player has seen an enemy at least once in the current round.</td><td>Boolean</td></tr></table>

TABLE X – continued from previous page
<table><tr><td>Feature</td><td>Description</td><td>Encoding / scaling</td></tr><tr><td>Player in smoke</td><td>Indicates whether the player is currently inside smoke.</td><td>Boolean</td></tr><tr><td colspan="3">Player input actions</td></tr><tr><td>Movement and action buttons</td><td>Indicates whether the player currently holds one of the following input buttons: jump, fire, use, crouch, walk, forward, back, left, or right.</td><td>Boolean</td></tr><tr><td>Weapon fired</td><td>Indicates whether the player fired a weapon in the current tick.</td><td>Boolean</td></tr><tr><td>Ticks since last weapon fire</td><td>Number of ticks since the player last fired a weapon.</td><td>log(1 + x) + standard scaler</td></tr><tr><td>Has fired weapon in round</td><td>Indicates whether the player has fired a weapon in the current round.</td><td>Boolean</td></tr><tr><td colspan="3">Damage and kill events</td></tr><tr><td>Damage given</td><td>Indicates whether the player dealt damage in the current tick.</td><td>Boolean</td></tr><tr><td>Ticks since last damage given</td><td>Number of ticks since the player last dealt damage.</td><td>log(1 + x) + standard</td></tr><tr><td>Has given damage in round</td><td>Indicates whether the player has dealt damage in the current round.</td><td>scaler Boolean</td></tr><tr><td>Damage received</td><td>Indicates whether the player received damage in the current tick</td><td>Boolean</td></tr><tr><td>Ticks since last damage</td><td>Number of ticks since the player last received damage.</td><td>log(1 + x) + standard</td></tr><tr><td>received Has received damage in round</td><td></td><td>scaler Boolean</td></tr><tr><td>Team suffered kill</td><td>Indicates whether the player has received damage in the current round. Indicates whether the player&#x27;s team lost a player in the current tick</td><td>Boolean</td></tr><tr><td>Ticks since last team suffered</td><td>Number of ticks since the player&#x27;s team last lost a player</td><td>log(1 + x) + standard</td></tr><tr><td>kill Has team suffered kill in</td><td></td><td>scaler</td></tr><tr><td>round</td><td>Indicates whether the player&#x27;s team has lost at least one player in the current round.</td><td>Boolean</td></tr><tr><td>Enemy team suffered kill Ticks since last enemy team</td><td>Indicates whether the enemy team lost a player in the current tick. Number of ticks since the enemy team last lost a player.</td><td>Boolean log(1 + x) + standard</td></tr><tr><td>suffered kill Has enemy team suffered kill</td><td></td><td>scaler</td></tr><tr><td>in round</td><td>Indicates whether the enemy team has lost at least one player in the current round.</td><td>Boolean</td></tr><tr><td>Kill given Has given kill in round</td><td>Indicates whether the player killed an enemy in the current tick.</td><td>Boolean</td></tr><tr><td></td><td>Indicates whether the player has killed an enemy in the current round.</td><td>Boolean</td></tr><tr><td colspan="3">Teammate features</td></tr><tr><td>Teammate i distance</td><td>Distance between the player and teammate i, where  $i \in \{ 1 , \ldots , 4 \}$ </td><td>Standard scaler</td></tr><tr><td>Teammate i visible</td><td>Indicates whether teammate i is visible to the player.</td><td>Boolean</td></tr><tr><td>Teammate i square ID</td><td>Discretized map square occupied by teammate ¿.</td><td>Embedding layer</td></tr><tr><td>Teammate i square local X</td><td>Local X coordinate of teammate i inside the corresponding map square.</td><td>Min-max scaler</td></tr><tr><td>Teammate i square local Y</td><td>Local Y coordinate of teammate i inside the corresponding map square.</td><td>Min-max scaler</td></tr><tr><td>Teammate i global Z position</td><td>Vertical position of teammate i on the map.</td><td>Per-map min-max scaler</td></tr><tr><td colspan="3">Enemy features</td></tr><tr><td>Enemy i alive</td><td>Indicates whether enemy i is alive, where  $i \in \{ 1 , \ldots , 5 \}$ </td><td>Boolean</td></tr><tr><td>Enemy i visible</td><td>Indicates whether enemy i is visible to the player.</td><td>Boolean</td></tr><tr><td>Enemy i can see player</td><td>Indicates whether enemy ¿ has line of sight to the current player.</td><td>Boolean</td></tr><tr><td>Enemy i square ID</td><td>Discretized map square occupied by enemy i.</td><td>Embedding layer</td></tr><tr><td>Enemy i square local X</td><td>Local X coordinate of enemy i inside the corresponding map square.</td><td>Min-max scaler</td></tr><tr><td>Enemy i square local Y</td><td>Local Y coordinate of enemy i inside the corresponding map square.</td><td>Min-max scaler</td></tr><tr><td>Enemy i global Z position</td><td>Vertical position of enemy ¿ on the map.</td><td>Per-map min-max scaler</td></tr><tr><td>Enemy i distance</td><td>Distance between the player and enemy ¿.</td><td>Standard scaler</td></tr><tr><td>Enemy i crouching</td><td>Indicates whether enemy i is crouching.</td><td>Boolean</td></tr><tr><td>Enemy i weapon fired</td><td>Indicates whether enemy i fired a weapon in the current tick.</td><td>Boolean</td></tr><tr><td>Enemy i visibility ratio</td><td>Fraction of the enemy model visible to the player. The value decreases when</td><td>Min-max scaler</td></tr><tr><td>Enemy i crosshair distance</td><td>the enemy is partially occluded by an obstacle. Distance between the player&#x27;s crosshair and enemy ¿.</td><td>Power transformer</td></tr><tr><td>Enemy i source radial velocity</td><td>Radial velocity of enemy i relative to the player.</td><td>Standard scaler</td></tr></table>

Table X describes the input of the transformer model (ORBIT). The dataset used by the gradient-boosted-tree architecture (EMBER) is very similar; embedding and scaling are not used for EMBER, and the remaining differences are listed below:

• player, teammate and enemy position is described by standard X, Y, Z values

• view offset Z is not used

• alive column is not used

• field of view is not used

• aim direction state and changed are not used

• some enemies visible and has seen enemy in round are not used

• movement and action buttons are not used

• damage given and received features are not used

• number of enemies and teammates alive are added

• closest teammate distance is added

ORBIT: Global Feature Importances for All Validation Cases  
![](images/70654d1de34039095917c63a8a321e7aaf91fe142418dc76b2c7ed334c523477.jpg)  
Feature importance (mean absolute class-margin change)

Fig. 15: Complete ORBIT global feature-importance ranking. Feature importance is measured as the mean absolute change in the predicted $\mathit { f l g h t }$ or $\# i g h t$ class margin after replacing each feature with a neutral baseline value. Features are ordered by decreasing importance.

![](images/e4c718ff415b56af73813119f9626b73a4c68bb006272890864575490ac8cc2b.jpg)  
Fig. 16: Complete global feature-importance ranking for EMBER. Sorted by gain. EMBER uses a subset of previously laid out feature set.  
Feature importance (mean absolute class-margin change)

![](images/f2d5dae08f65dfc19f13324973603e7b6665d0fd139ca9c863223856314783be.jpg)  
Fig. 17: General serialization pipeline from the source dataset row to LLM input. All benchmarks share row preparation; encodings differ in how features are rendered. All runs attach labeled train examples and one unlabeled query; the mode returns structured JSON with label and reasoning.

## D. LLM prompt examples

Listing 1: Task preamble and label mapping of the LLM prompt using Narrative encoding (excerpt) You are a classifier for Counter-Strike 2 (5v5) situations.

Your task is to predict a binary label from game-state features.   
Use this label mapping exactly:   
- 0 = flight   
- 1 = fight   
Meaning:   
- flight (0): the player is more likely to disengage, retreat, avoid direct confrontation, reposition   
defensively, or delay contact.   
- fight (1): the player is more likely to take or continue engagement, peek, challenge, chase, hold for a   
duel, or commit to combat.   
Constraints:   
- Use only the provided feature values.   
- Do not assume access to external game knowledge beyond the feature names and values.   
- Do not invent missing information.   
- Base the decision on the current snapshot and recent history included in the features.   
Missing / unknown values:

- Usually -1 or -1.0 means unknown or unavailable information, not a real physical value.   
- Treat these values as missing information, not as literal coordinates, distances, or measurements.

The complete prompt templates for the Narrative, Compact JSON and Full-CSV encodings, including the few-shot examples and the serialized query rows, are available from the authors on request.

## E. Recognized patterns in LLM reasoning

TABLE XI: Fight-or-flight patterns in LLM reasoning
<table><tr><td>Pattern</td><td>Description</td></tr><tr><td colspan="2">Fight patterns</td></tr><tr><td>The Stationary Hold</td><td>Remaining stationary with a primary weapon (especially AWP) while the crosshair is aligned with a potential entry point.</td></tr><tr><td>Aggressive Pursuit</td><td>Moving quickly toward an enemy or tracking a retreating enemy with a combat-ready weapon.</td></tr><tr><td>Tactical Ambush Weapon Switching</td><td>Engaging an unaware enemy from a position of advantage, often described as &quot;stalking&quot; or &quot;initiative&quot;. Transitioning from a knife or utility to a primary weapon immediately upon spotting an enemy.</td></tr><tr><td>Flight patterns</td><td></td></tr><tr><td colspan="2"></td></tr><tr><td>Utility/Knife Rotation</td><td>Moving at maximum velocity with a knife or grenade out, interpreted as prioritizing speed for repositioning.</td></tr><tr><td>Defensive Repositioning Survival Preservation</td><td>Moving away from enemy coordinates or turning the view (Yaw) away from the threat to find cover. Retreating due to critically low health or being the last player alive in a 1vX situation.</td></tr><tr><td>Long-Range Avoidance</td><td>Disengaging when the enemy is at extreme distance (&gt; 2000 units) and the current weapon is ineffective.</td></tr><tr><td colspan="2">Error patterns</td></tr><tr><td>Crosshair Proximity Bias</td><td>Models frequently predict fight simply because the crosshair is near an enemy, ignoring that the player is actually retreating or holding a passive angle.</td></tr><tr><td>Misinterpreting Aggressive Movement</td><td>High-speed movement is often labeled as flight (rotation) when it is actually an aggressive push, or fight (push) when it is a tactical retreat.</td></tr><tr><td>Sniper Staticity</td><td>Models consistently assume stationary snipers are &quot;fighting,&quot; failing to recognize when they are &quot;saving&quot; or waiting</td></tr><tr><td>Utility/Knife False Negatives</td><td>to rotate. Models assume holding a knife or grenade always means flight, missing instances where players use speed or utility to initiate a fight.</td></tr></table>

[1] C. Méndez-Bértolo, S. Moratti, R. Toledano, F. López-Sosa, R. Martínez-Álvarez, Y. H. Mah, P. Vuilleumier, A. Gil-Nagel, and B. A. Strange, “A fast pathway for fear in human amygdala,” Nature Neuroscience, vol. 19, no. 8, pp. 1041–1049, 2016.

[2] Y. Wang, L. Luo, G. Chen, G. Luan, X. Wang, Q. Wang, and F. Fang, “Rapid processing of invisible fearful faces in the human amygdala,” Journal of Neuroscience, vol. 43, no. 8, pp. 1405–1413, 2023.

[3] L. Pessoa and R. Adolphs, “Emotion processing and the amygdala: From a ‘low road’ to ‘many roads’ of evaluating biological significance,” Nature Reviews Neuroscience, vol. 11, no. 11, pp. 773–783, 2010.

[4] D. Kahneman, “A perspective on judgment and choice: Mapping bounded rationality,” American Psychologist, vol. 58, no. 9, pp. 697– 720, 2003.

[5] J. McFadyen, M. Mermillod, J. B. Mattingley, V. Halasz, and M. I. Garrido, “A rapid subcortical amygdala route for faces irrespective of spatial frequency and emotion,” Journal of Neuroscience, vol. 37, no. 14, pp. 3864–3874, 2017.

[6] J. E. LeDoux, The Emotional Brain: The Mysterious Underpinnings of Emotional Life. New York: Simon & Schuster, 1996.

[7] L. Pessoa, The Cognitive-Emotional Brain: From Interactions to Integration. Cambridge, MA: MIT Press, 2013.

[8] M. Inagaki, K.-i. Inoue, S. Tanabe, K. Kimura, M. Takada, and I. Fujita, “Rapid processing of threatening faces in the amygdala of nonhuman primates: Subcortical inputs and dual roles,” Cerebral Cortex, vol. 33, no. 3, pp. 895–915, 2023.

[9] R. D. Rafal and K. Koller, “Collothalamic projections to the human amygdala: Hemispheric asymmetry modulates trait anxiety,” Journal of Neurophysiology, vol. 133, no. 4, pp. 1054–1066, 2025.

[10] J. Valls-Solé, J. C. Rothwell, F. Goulart, G. Cossu, and E. Muñoz, “Patterned ballistic movements triggered by a startle in healthy humans,” Journal of Physiology, vol. 516, no. 3, pp. 931–938, 1999.

[11] A. N. Carlsen, R. Chua, J. T. Inglis, D. J. Sanderson, and I. M. Franks, “Differential effects of startle on reaction time for finger and arm movements,” Journal of Neurophysiology, vol. 101, no. 1, pp. 306–314, 2009.

[12] E. N. A. Adjei, K. Wright, J. P. A. Dewald, and J. Yao, “Effects of asymmetrical postural demands on sternocleidomastoid reflex in the StartReact paradigm,” Frontiers in Human Neuroscience, vol. 19, p. 1592691, 2025.

[13] G. Y. Wang, D. Simkute, and I. Griskova-Bulanova, “Neurobiological link between stress and gaming: A scoping review,” Journal of Clinical Medicine, vol. 12, no. 9, p. 3113, 2023.

[14] H. Aliyari, H. Sahraei, M. R. Daliri, B. Minaei-Bidgoli, M. Kazemi, H. Agaei, M. Sahraei et al., “The beneficial or harmful effects of computer game stress on cognitive functions of players,” Basic and Clinical Neuroscience, vol. 9, no. 3, pp. 177–186, 2018.

[15] A. C. Nobre and F. van Ede, “Anticipated moments: Temporal structure in attention,” Nature Reviews Neuroscience, vol. 19, no. 1, pp. 34–48, 2018.

[16] D. V. P. S. Murty, S. Song, S. G. Surampudi, and L. Pessoa, “Threat and reward imminence processing in the human brain,” Journal of Neuroscience, vol. 43, no. 16, pp. 2973–2987, 2023.

[17] J. E. LeDoux, “Emotion circuits in the brain,” Annual Review of Neuroscience, vol. 23, pp. 155–184, 2000.

[18] D. Mobbs, J. L. Marchant, D. Hassabis, B. Seymour, G. Tan, M. Gray, P. Petrovic, R. J. Dolan, and C. D. Frith, “From threat to fear: The neural organization of defensive fear systems in humans,” Journal of Neuroscience, vol. 29, no. 39, pp. 12 236–12 243, 2009.

[19] G. Karpov, M.-H. Lin, D. B. Headley, and T. E. Baker, “Oscillatory correlates of threat imminence during virtual navigation,” Psychophysiology, vol. 61, no. 7, p. e14551, 2024.

[20] J. Lu, S. K. Kemmerer, L. Riecke, and B. de Gelder, “Early threat perception is independent of later cognitive and behavioral control: A virtual reality-EEG-ECG study,” Cerebral Cortex, vol. 33, no. 13, pp. 8748–8758, 2023.

[21] M. S. Fanselow, “The role of learning in threat imminence and defensive behaviors,” Current Opinion in Behavioral Sciences, vol. 24, pp. 44–49, 2018.

[22] ——, “Negative valence systems: Sustained threat and the predatory imminence continuum,” Emerging Topics in Life Sciences, vol. 6, no. 5, pp. 467–477, 2022.

[23] J. M. Moscarello and M. A. Penzo, “The central nucleus of the amygdala and the construction of defensive modes across the threat-imminence continuum,” Nature Neuroscience, vol. 25, no. 8, pp. 999–1008, 2022.

[24] R. Abend et al., “Understanding anxiety symptoms as aberrant defensive responding along the threat imminence continuum,” Neuroscience & Biobehavioral Reviews, vol. 152, p. 105305, 2023.

[25] J. C. K. Wells, P. Michael, M. S. Fewtrell, M. Siervo, and M. Cortina-Borja, “Fright but not fight-or-flight: Violent video games elevated stress markers, but did not impact muscle function, memory recall or food intake, in a randomized trial in healthy young men,” American Journal of Biological Anthropology, vol. 178, no. 3, pp. 476–487, 2022.

[26] C. Laier, E. Wegmann, and M. Brand, “Personality and cognition in gamers: Avoidance expectancies mediate the relationship between maladaptive personality traits and symptoms of internet-gaming disorder,” Frontiers in Psychiatry, vol. 9, p. 304, 2018.

[27] P. M. Fitts and M. I. Posner, Human Performance. Belmont, CA: Brooks/Cole, 1967.

[28] C. Montag, B. Weber, P. Trautner, B. Newport, S. Markett, N. T. Walter, A. Felten, and M. Reuter, “Does excessive play of violent first-personshooter-video-games dampen brain activity in response to emotional stimuli?” Biological Psychology, vol. 89, no. 1, pp. 107–111, 2012.

[29] R. Weber, U. Ritterfeld, and K. Mathiak, “Does playing violent video games induce aggression? Empirical evidence of a functional magnetic resonance imaging study,” Media Psychology, vol. 8, no. 1, pp. 39–60, 2006.

[30] M. Klasen, R. Weber, T. T. J. Kircher, K. A. Mathiak, and K. Mathiak, “Neural contributions to flow experience during video game playing,” Social Cognitive and Affective Neuroscience, vol. 7, no. 4, pp. 485–495, 2012.

[31] N. D. Daw, Y. Niv, and P. Dayan, “Uncertainty-based competition between prefrontal and dorsolateral striatal systems for behavioral control,” Nature Neuroscience, vol. 8, no. 12, pp. 1704–1711, 2005.

[32] N. D. Daw, S. J. Gershman, B. Seymour, P. Dayan, and R. J. Dolan, “Model-based influences on humans’ choices and striatal prediction errors,” Neuron, vol. 69, no. 6, pp. 1204–1215, 2011.

[33] E. M. Russek, I. Momennejad, M. M. Botvinick, S. J. Gershman, and N. D. Daw, “Predictive representations can link model-based reinforcement learning to model-free mechanisms,” PLoS Computational Biology, vol. 13, no. 9, p. e1005768, 2017.

[34] I. Kuperwajs, H. H. Schütt, and W. J. Ma, “Using deep neural networks as a guide for modeling human planning,” Scientific Reports, vol. 13, p. 20269, 2023.

[35] H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang, “Informer: Beyond efficient transformer for long sequence time-series forecasting,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 12, 2021, pp. 11 106–11 115, arXiv:2012.07436.

[36] H. Wu, J. Xu, J. Wang, and M. Long, “Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 34, 2021, pp. 22 419–22 430, arXiv:2106.13008.

[37] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” in International Conference on Learning Representations (ICLR), 2023, arXiv:2211.14730.

[38] A. Zeng, M. Chen, L. Zhang, and Q. Xu, “Are transformers effective for time series forecasting?” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 9, 2023, pp. 11 121–11 128, arXiv:2205.13504.

[39] C. Zeng, Z. Liu, G. Zheng, and L. Kong, “Cmamba: Channel correlation enhanced state space models for multivariate time series forecasting,” arXiv preprint arXiv:2406.05316, 2024.

[40] Z. Liu et al., “Dual-perspective enhanced mamba for multivariate time series forecasting,” Neurocomputing, vol. 618, p. 129218, 2025.

[41] B. Kang et al., “Dc-mamber: A dual channel prediction model based on mamba and linear transformer for multivariate time series forecasting,” arXiv preprint arXiv:2507.04381, 2025.

[42] R. Chen and F. Sun, “Dmamba: Decomposition-enhanced mamba for time series forecasting,” arXiv preprint arXiv:2602.09081, 2026.

[43] Z. Wang, F. Kong, S. Feng, M. Wang, X. Yang, H. Zhao, D. Wang, and Y. Zhang, “Is mamba effective for time series forecasting?” Neurocomputing, vol. 619, p. 129178, 2025, arXiv:2403.11144.

[44] A. F. Ansari, L. Stella, C. Turkmen, X. Zhang, P. Mercado, H. Shen, O. Shchur, S. S. Rangapuram, S. Pineda Arango, S. Kapoor, J. Zschiegner, D. C. Maddix, H. Wang, M. W. Mahoney, K. Torkkola, A. G.

Wilson, M. Bohlke-Schneider, and Y. Wang, “Chronos: Learning the language of time series,” Transactions on Machine Learning Research (TMLR), 2024, arXiv:2403.07815.

[45] G. Woo, C. Liu, A. Kumar, C. Xiong, S. Savarese, and D. Sahoo, “Unified training of universal time series forecasting transformers,” in International Conference on Machine Learning (ICML), 2024, arXiv:2402.02592.

[46] V. Ekambaram, A. Jati, P. Dayama, S. Mukherjee, N. H. Nguyen, W. M. Gifford, C. Reddy, and J. Kalagnanam, “Tiny time mixers (ttms): Fast pre-trained models for enhanced zero/few-shot forecasting of multivariate time series,” Advances in Neural Information Processing Systems (NeurIPS), 2024, arXiv:2401.03955.

[47] C. N. Riddle and S. N. Baker, “Convergence of pyramidal and medial brain stem descending pathways onto macaque cervical spinal interneurons,” Journal of Neurophysiology, vol. 103, no. 5, pp. 2821–2832, 2010.

[48] C. F. Honeycutt, M. Kharouta, and E. J. Perreault, “Evidence for reticulospinal contributions to coordinated finger movements in humans,” Journal of Neurophysiology, vol. 110, no. 7, pp. 1476–1483, 2013.

[49] C. M. Santangelo and A. N. Carlsen, “Startle-triggered responses indicate reticulospinal drive is larger for voluntary shoulder versus finger movements,” Scientific Reports, vol. 13, p. 6532, 2023.

[50] M. Grabenhorst, G. Michalareas, L. T. Maloney, and D. Poeppel, “The anticipation of events in time,” Nature Communications, vol. 10, no. 1, p. 5802, 2019.

[51] Y. LeCun, “A path towards autonomous machine intelligence,” OpenReview preprint, 2022, version 0.9.2.

[52] A. Bardes, Q. Garrido, J. Ponce, X. Chen, M. Rabbat, Y. LeCun, M. Assran, and N. Ballas, “V-JEPA 2: Self-supervised video models enable understanding, prediction, and planning,” 2025, meta AI Research.

[53] W. B. Cannon, Bodily Changes in Pain, Hunger, Fear and Rage: An Account of Recent Researches into the Function of Emotional Excitement. New York, NY and London: D. Appleton and Company, 1915.

[54] L. D. Godoy, M. T. Rossignoli, P. Delfino-Pereira, N. Garcia-Cairasco, and E. H. de Lima Umeoka, “A comprehensive overview on stress neurobiology: Basic concepts and clinical implications,” Frontiers in Behavioral Neuroscience, vol. 12, p. 127, 2018.

[55] D. Purves, G. J. Augustine, D. Fitzpatrick, L. C. Katz, A.-S. LaMantia, J. O. McNamara, and S. M. Williams, Eds., Neuroscience, 2nd ed. Sunderland, MA: Sinauer Associates, 2001.

[56] R. Ratcliff, “Group reaction time distributions and an analysis of distribution statistics,” Psychological Bulletin, vol. 86, no. 3, pp. 446– 461, 1979.

[57] R. Ratcliff and G. McKoon, “The diffusion decision model: Theory and data for two-choice decision tasks,” Neural Computation, vol. 20, no. 4, pp. 873–922, 2008.

[58] R. D. Luce, Response Times: Their Role in Inferring Elementary Mental Organization. New York: Oxford University Press, 1986.

[59] T. Chen and C. Guestrin, “Xgboost: A scalable tree boosting system,” in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD). ACM, 2016, pp. 785–794, arXiv:1603.02754.

[60] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations (ICLR), 2021.

[61] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei, “Scaling laws for neural language models,” arXiv preprint arXiv:2001.08361, 2020.

[62] M. Binz, E. Akata, M. Bethge, F. Brändle, F. Callaway, J. Coda-Forno, P. Dayan, C. Demircan, M. K. Eckstein, N. Élteto, T. L. Griffiths,˝ S. Haridi, A. K. Jagadish, L. Ji-An, A. Kipnis, S. Kumar, T. Ludwig, M. Mathony, M. Mattar, A. Modirshanechi, S. S. Nath, J. C. Peterson, M. Rmus, E. M. Russek, T. Saanum, N. Scharfenberg, J. A. Schubert, L. M. S. Buschoff, N. Singhi, X. Sui, M. Thalmann, F. Theis, V. Truong, V. Udandarao, K. Voudouris, R. Wilson, K. Witte, S. Wu, D. Wulff, H. Xiong, and E. Schulz, “Centaur: a foundation model of human cognition,” 2024. [Online]. Available: https://arxiv.org/abs/2410.20268

[63] S. Minami, K. Watanabe, N. Saijo, and M. Kashino, “Neural oscillation amplitude in the frontal cortex predicts esport results,” iScience, vol. 26, no. 6, p. 106845, 2023.

[64] S. Minami, H. Koyama, K. Watanabe, N. Saijo, and M. Kashino, “Prediction of esports competition outcomes using EEG data from expert players,” Computers in Human Behavior, vol. 160, p. 108351, 2024.

[65] R. A. Poldrack, “Can cognitive processes be inferred from neuroimaging data?” Trends in Cognitive Sciences, vol. 10, no. 2, pp. 59–63, 2006.

[66] ——, “Inferring mental states from neuroimaging data: From reverse inference to large-scale decoding,” Neuron, vol. 72, no. 5, pp. 692–697, 2011.

[67] Y. W. Bernier, “Latency compensating methods in client/server ingame protocol design and optimization,” in Proceedings of the Game Developers Conference (GDC), 2001.

[68] M. Claypool and K. Claypool, “Latency and player actions in online games,” Communications of the ACM, vol. 49, no. 11, pp. 40–45, 2006.

[69] S. Liu, X. Xu, and M. Claypool, “A survey and taxonomy of latency compensation techniques for network computer games,” ACM Computing Surveys, vol. 54, no. 11s, 2022.

[70] X. Xu, S. Liu, and M. Claypool, “The effects of network latency on Counter-Strike: Global Offensive players,” in Proceedings of the 14th International Conference on Quality of Multimedia Experience (QoMEX), 2022.

[71] S. S. Tokey, Z. Chen, C. Mettler, D. Tang, B. Boudaoud, J. Kim, J. Spjut, and M. Claypool, “The effects of network latency on the peeker’s advantage in first-person shooter games,” in Proceedings of the 19th International Conference on the Foundations of Digital Games (FDG), 2024.

[72] Valve Corporation, “Counter-strike 2: Moving beyond tick rate,” Official reveal video, Counter-Strike YouTube channel, 2023, https://www. youtube.com/watch?v=GqhhFl5zgA0, accessed 2026-05-30.

[73] J. E. LeDoux and N. D. Daw, “Surviving threats: Neural circuit and computational implications of a new taxonomy of defensive behaviour,” Nature Reviews Neuroscience, vol. 19, no. 5, pp. 269–282, 2018.