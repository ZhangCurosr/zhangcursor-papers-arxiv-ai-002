# FFASR: Benchmarking Far-Field Automatic Speech Recognition using High-Fidelity Simulated RIRs

Shivam Saini<sup>∗</sup>, Eric Bezzam<sup>†</sup>, Georg Götz<sup>∗</sup>, Alessia Milo<sup>∗</sup>, Steinar Guðjónsson<sup>∗</sup>,

Konstantinos Gkanos<sup>∗</sup>, Finnur Pind<sup>∗</sup>, Daniel Gert Nielsen<sup>∗</sup>

<sup>∗</sup>Treble Technologies, Reykjavík, Iceland

<sup>†</sup>Hugging Face, Paris, France

\*Correspondence: shivam@treble.tech

Abstract—Far-field automatic speech recognition (ASR) degrades under reverberation, noise, and talker motion, yet the benchmarks that drive model selection emphasize close-microphone speech. We present FFASR, a held-out corpus of 15,637 utterances and an open leaderboard spanning nine conditions, each varying a single acoustic factor: anechoic near-field speech, a measured-versus-simulated ofice-lab pair, static farfield mixtures at high/mid/low signal-to-noise ratio (SNR), and moving-talker variants at matched SNR. Dry speech from 15 talkers is convolved with hybrid wave/geometrical-acoustics room impulse responses from 14 furnished rooms; because the speech is newly recorded and the test waveforms are never released, the corpus resists training-data contamination. Across contemporary systems, mean word error rate (WER) rises from 4.4% near-field to 41.3% in the static low-SNR condition; a moving talker adds a small but consistent penalty at matched SNR; and on the ofice-lab pair, measured and simulated WER agree to within about 1.7 pp on average. These results support high-fidelity simulation as a scalable proxy for measured far-field evaluation under the conditions we test.

Index Terms—Far-Field Speech Recognition, ASR, benchmark, leaderboard, moving sources

## I. Introduction

Consumer and enterprise voice interfaces increasingly operate at meter-scale distances from the microphone: smart speakers, conference systems, and wearable assistants capture speech only after room reflections, difraction, and competing noise have reshaped the waveform. The resulting mismatch is well documented, reverberation smears phonetic cues over hundreds of milliseconds while additive noise suppresses low-energy consonants, and has motivated a long line of dedicated evaluations, from the REVERB challenge [1] to the CHiME series [2], [3], AMI [4], and VOiCES [5]. Yet the benchmarks that drive day-to-day model selection remain dominated by close-microphone read speech such as LibriSpeech [6] and its derivatives [7], and modern foundation models are routinely trained on simulated reverberant data [8], [9] without a corresponding standardized far-field test.

Constructing such a test set is a balancing act: it needs physical realism, broad and controllable acoustic coverage, and source material novel enough to resist training-data contamination, and these goals pull against one another. Measured corpora [10]–[12] ofer physical ground truth but cover few geometries and fixed SNR statistics, and once they have been public for several years their audio and transcripts leak into web-scale training crawls, raising contamination risk. Classical simulators based on geometrical acoustics (GA), such as the image-source method [13]–[15], scale arbitrarily but omit difraction, scattering, and lowfrequency modal behavior, exactly the wave phenomena that dominate below the Schroeder frequency of typical rooms. Hybrid engines that couple wave-based solvers [16] with GA above a crossover frequency have narrowed this realism gap [17]–[20], and recent hybrid higher-order Ambisonics RIR datasets [18], [21] demonstrated that such room impulse responses (RIRs) support far-field ASR research at scale. In our own preliminary leaderboard runs, several systems that were almost indistinguishable on near-field speech separated sharply once the same utterances were rendered in rooms, which is what convinced us a dedicated far-field test set was needed. No existing resource combines all of this: a fixed far-field test set whose audio stays private, separate SNR and sourcemotion factors, a measured anchor for the sim-to-real gap, and shared infrastructure that scores diferent model families under identical decoding and normalization rules. FFASR is built to close that gap; Fig. 1 gives a oneview overview and Table I lists the nine conditions. We contribute:

A benchmark and open infrastructure. A ninecondition far-field corpus (Table I) built from newly recorded anechoic speech (889 utterances, 15 talkers) convolved with hybrid wave/GA RIRs from 14 furnished rooms (53–367 m<sup>3</sup>), plus a measured ofice-lab triplet for sim-to-real calibration. A deterministic, seeded pipeline shares transcripts, talkers, and noise draws across conditions, so per-condition WER diferences come only from the acoustic factor being varied. An open leaderboard scores each submission in an isolated GPU sandbox [22] and never releases the test waveforms (Sec. III–IV).

Empirical findings from evaluating contemporary systems, from compact CTC models to large speech LLMs (Sec. V): clean-speech rankings hide large far-field diferences, separating systems that look equivalent on clean speech and triggering insertion-dominated hallucination at low SNR; source motion adds a smaller additional cost at matched SNR; and the measured and simulated ofice-lab renders give similar WER across architectures, suggesting the simulated renders are close enough to measured ones for this evaluation.

![](images/03f1fc9ffd04c60c147719eb087a086c3949b66c8209d8b7d63ac2704243091c.jpg)  
Fig. 1. Data-generation pipeline. Newly recorded anechoic speech is convolved with hybrid wave/GA room impulse responses and seeded noise draws; each mixture is labelled by its post-render SNR.

## II. Related Work

## A. Far-Field Corpora and Challenges

The REVERB challenge [1] established reverberant evaluation with simulated and measured single- and eight-channel data. CHiME-5/6 [2], [3] moved to unsegmented multi-talker dinner parties, while AMI [4] and LibriCSS [23] target meetings and overlapped speech. These corpora combine many degradation factors at once, such as overlap, disfluency, and distant arrays, which makes them excellent integration tests but poor instruments for isolating a single far-field factor; convolution with measured RIRs is controllable but static. FFASR complements these with single-talker, factorized conditions with matched lexical content across all nine splits.

## B. Simulation for Far-Field Speech Processing

Simulated RIRs are a standard tool for training robust ASR [8], [9] and for privacy-preserving on-device development [24]; our concern here is instead their fidelity for evaluation. The hybrid wave/GA rendering in an industrial hybrid wave/GA simulator(Sec. III-B) has been validated against measurement across several tasks [18], [25]–[30]. Most relevant to the present evaluation, [26] investigates the simulation-to-real gap when evaluating audio algorithms on simulated data with varying fidelity rather than measurements, while [29] and [30] report that the added physical fidelity of hybrid RIR training data over image-source augmentation transfers to a downstream ASR metric, reducing median WER on measured test data by up to 38% relative. FFASR builds on this line, using 14 furnished scenes with many source–receiver positions and directional noise sources.

TABLE I  
FFASR evaluation conditions. N is the number of clips in each packed split.
<table><tr><td>Condition</td><td>N</td><td>Description</td></tr><tr><td>Near field</td><td>889</td><td>Dry anechoic speech</td></tr><tr><td>Lab simulated</td><td>2000</td><td>Stage A hybrid render</td></tr><tr><td>Lab measured</td><td>2000</td><td>Stage A measured mono</td></tr><tr><td>High SNR</td><td>1746</td><td>Static, SNR ≥ 14 dB</td></tr><tr><td>Mid SNR</td><td>1938</td><td>Static, 8–12 dB</td></tr><tr><td>Low SNR</td><td>1647</td><td>Static, ≤ 6 dB</td></tr><tr><td>Moving high</td><td>1808</td><td>Trajectory + high SNR</td></tr><tr><td>Moving mid</td><td>1816</td><td>Trajectory + mid SNR</td></tr><tr><td>Moving low</td><td>1793</td><td>Trajectory + low SNR</td></tr></table>

## C. Robust ASR Systems

The systems we evaluate are drawn from across the current architecture space: web-scale weakly supervised encoder–decoders (Whisper [31]) and compressed variants [32], [33]; Conformer [34]/FastConformer [35] encoders with CTC or transducer [36] decoders, including the token-and-duration transducer (TDT) [37] used by Parakeet; Canary [38]; OWSM [39], [40]; speech-aware LLMs (Granite Speech [41]); wav2vec 2.0 [42] CTC baselines via SpeechBrain [43]; and streaming-oriented compact models (Moonshine [44]). We treat these as the systems under test rather than as related methods, and report their behavior in Sec. V.

## D. Leaderboards and Evaluation Methodology

The Open ASR Leaderboard [7] standardized text normalization and RTFx reporting for predominantly closetalk English test sets and documented how normalization choices alone shift WER rankings. FFASR adopts the same normalizer and eficiency metric so that scores are directly comparable, while adding the far-field axes those suites lack. Prior work also documents Whisper-family hallucination on degraded or silent input [45]; our low-SNR conditions elicit this failure mode systematically and expose it as WER exceeding 100% (Sec. V-A).

## III. The FFASR Benchmark

Fig. 1 summarizes the generation flow and Algorithm 1 states it step by step. A few deliberate choices shape the corpus. All source speech is newly recorded and the test waveforms are never distributed, so a model cannot score well simply by having seen the test data in training. Each condition changes a single acoustic factor while holding transcripts, talkers, and seeded noise draws fixed, so a WER gap between two conditions can be attributed to that factor rather than to a change in content. And rendering uses a hybrid wave/GA solver that has been checked against measurement, which keeps the simulated rooms physically grounded rather than idealized. The nine resulting conditions are listed in Table I.

## A. Anechoic Source Speech

Clean speech is recorded in the anechoic chamber at an anonymized university. Fifteen subjects (6 female, 9 male; who gave informed consent for research use of the recordings, who gave informed consent for research use of the recordings) each read 80 English sentences from a fixed prompt list. Fifteen talkers is a compromise: enough to average over individual voice and accent, but small enough that every talker appears in every condition, so no split is confounded by a change in voices. Two talkers are native

Algorithm 1 FFASR scene generation (one mixture)   
Require: one dry utterance $s ,$ a pool of anechoic interferers,   
the RIR collections, and a shared random seed   
1: draw 1–2 interferers $\{ n _ { k } \}$ using the shared seed, so the same   
draw recurs across conditions   
2: pick a room and the target/receiver positions, and load the   
target’s RIR $h _ { \mathrm { s } }$ and one RIR h per interferer   
3: place each source in the room by convolution: target $y _ { \mathrm { s } } =$   
$h _ { \mathrm { s } } * s ,$ background $\begin{array} { r } { y _ { n } = \sum _ { k } h _ { k } * n _ { k } } \end{array}$ {time-varying $h _ { \mathrm { s } } ( t , \tau )$   
if the target moves}   
4: pass every render through the device front-end: a 70 Hz   
high-pass plus added pink self-noise $d ( t )$   
5: form the mixture $y = y _ { \mathrm { s } } + y _ { n } + d ,$ and mark the speech  
active region A of $\dot { y } _ { \mathrm { s } }$ with an ITU-T P.56 detector   
6: measure the SNR of y over A using (2)   
7: if the SNR lands in the High, Mid, or Low band then   
8: keep y and label it with that band   
9: else   
10: discard $y \colon$ it fell in a guard band between bands   
11: end if   
Ensure: mono 16 kHz mixture y with its SNR-band label

English speakers and thirteen are non-native, spanning diferent native-language backgrounds, so the pool reflects the accent diversity of deployed voice interfaces. FFASR isolates the far-field acoustic factor rather than accent: all nine conditions share this single talker pool, so each model’s own near-field score is its clean-speech reference. Accent thus varies across talkers but is held constant across conditions (Sec. VI). Of the 1,200 prompt recordings (15 talkers × 80 prompts), a single annotator screened every recording and discarded 311 (26%) for mispronunciations, disfluencies, truncated prompts, or audible recording defects, leaving 889 utterances. Because no public readspeech test set is reused, leaderboard scores reflect acoustic generalization rather than memorized lexicon or speaker profiles. Each utterance is normalized and rendered at 60 dB SPL free-field level.

## B. Hybrid Room Acoustic Simulation

Let s(t) denote the dry target utterance, $\{ n _ { k } ( t ) \} _ { k = 1 } ^ { K }$ the anechoic interferer signals, and $h _ { \mathrm { s } } , h _ { k }$ the RIRs to the receiver from the target speaker (subscript s) and the k-th noise source. Each scene uses up to $K \leq 2$ interferers, and each static far-field mixture is

$$
y ( t ) = ( h _ { \mathrm { s } } \ast s ) ( t ) + \sum _ { k = 1 } ^ { K } ( h _ { k } \ast n _ { k } ) ( t ) + d ( t ) ,\tag{1}
$$

where ∗ denotes convolution and $d ( t )$ is device self-noise. RIRs are drawn from three precomputed collections:

• Static Sources: 14 scenes spanning living spaces, ofices, meeting rooms, classrooms, and restaurants, with volumes 53–367 m<sup>3</sup>.

• Moving Sources: paired moving variants of the same rooms with time-varying source paths.

• Measured-Simulated pairs: 44 measured RIRs and their simulated counterparts over six receivers in a single ofice, in three configurations that change only the acoustic treatment: bare brick walls(RT ≈ 0.9 s), the same room furnished $( \approx ~ 0 . 6 \mathrm { s } )$ , and the room fitted with wall absorber panels (≈ 0.3 s).

All furnished-room IRs use hybrid simulation in an industrial hybrid wave/GA simulator [17]: a wave-based discontinuous Galerkin finite-element solver [16] up to a 2 kHz crossover and GA above, computed at 32 kHz and stored as 8th-order ambisonics (81 channels) before rendering to mono 16 kHz. The 2 kHz crossover sits below most consonantal energy (2–8 kHz), so the wave solver mainly improves the low- and mid-frequency modal and difraction behavior. The external validations above [18], [29] and the measured anchor (Sec. V-C) confirm that this added fidelity still transfers to ASR-level WER. Receivers are restricted to heights $1 . 0 ~ < ~ z ~ < ~ 2 . 3 \mathrm { m }$ , consistent with tabletop and wall-mounted devices; elevated sources $\left( z > 1 . 9 \mathrm { m } \right)$ model HVAC-like emitters. Each scene places one directive target talker and up to two noise paths: a mixture pool and static-like ambient noises from AID [46].

## C. Noise, SNR Ranges, and Device Chain

Reverberant energy builds up diferently depending on room geometry and absorption, so the source-level ratio set in advance is a poor predictor of the SNR observed at the receiver. We therefore measure SNR after rendering. A fixed device chain is first applied at the listener to emulate the analog front-end of consumer hardware: a 4th-order Butterworth high-pass at 70 Hz and pink device self-noise d(t) at 35–40 dB SPL. The 70 Hz corner approximates the low-frequency rollof of the small transducers in smart speakers and phones and removes sub-band rumble those devices would not capture, so the signal reaching the recognizer matches deployed hardware rather than an idealized full-band render. The SNR is then the energy ratio between the target render $y _ { \mathrm { s } } ~ { = } ~ h _ { \mathrm { s } }$ ∗ s and the full background, namely the interferer render $\begin{array} { r } { y _ { n } = \sum _ { k } h _ { k } * n _ { k } } \end{array}$ plus the device noise $d ( t )$ . Both energy terms are accumulated only over the speech-active region A of the target render $y _ { \mathrm { s } } ,$ where $\mathcal { A }$ is the set of samples lying in frames flagged active by an ITU-T P.56 detector [47], so that silent and inter-word pauses do not contribute:

$$
\mathrm { S N R } = 1 0 \log _ { 1 0 } \frac { \sum _ { t \in \mathcal { A } } y _ { \mathrm { s } } ^ { 2 } ( t ) } { \sum _ { t \in \mathcal { A } } \left( y _ { n } ( t ) + d ( t ) \right) ^ { 2 } } .\tag{2}
$$

Keeping d(t) in the denominator means the label reflects the SNR actually presented to the recognizer, not an idealized speech-to-interferer ratio. Each mixture is assigned to one of three SNR ranges, Low [−2, 6] dB, Mid [8, 12] dB, and $\mathrm { H i g h } \geq 1 4 \mathrm { d B }$ . The bands are chosen to bracket qualitatively diferent listening regimes, adverse, efortful, and comfortable, rather than to trace a fine SNR curve, which keeps the conditions interpretable and the per-condition splits large. The transition bands (6, 8) and (12, 14) dB act as guard bands and are excluded, so labels stay unambiguous under small changes in the render. For reproducibility, a shared integer seed fixes the interferer draw, room, and source/receiver positions so any scene can be regenerated, mixtures are peak-limited below 0 dBFS to prevent clipping before (2) is evaluated, and all renders are written as 16 kHz mono WAV.

## D. Moving Talker Conditions

To simulate a moving talker, the static target RIR is replaced by a time-varying RIR that evolves along a predefined source trajectory, while keeping the interferers fixed. Therefore, the static convolution in (1) becomes a time-varying convolution,

$$
y ( t ) = \sum _ { \tau } h _ { \mathrm { s } } ( t , \tau ) s ( t - \tau ) + \sum _ { k = 1 } ^ { K } ( h _ { k } * n _ { k } ) ( t ) + d ( t ) ,\tag{3}
$$

where $\tau$ is the discrete time lag and $h _ { \mathrm { s } } ( t , \tau )$ is the target impulse response at time t, interpolated along the source trajectory; the interferers stay static. We synthesize $h _ { \mathrm { s } } ( t , \tau )$ with the frequency-domain interpolation method of Mullins and Sampedro Llopis [48]. That work compared three ofline movement-simulation algorithms. Frequency-domain interpolation achieved the most accurate interaural-time-diference reproduction and the highest perceptual (MUSHRA) similarity to real movingsource recordings, outperforming time-domain crossfading and nearest-neighbor switching. We reuse the same talker and interferer signals as in the static scenes. The static target RIR is replaced by a trajectory that traverses the complete moving-IR path over 6 s of speech within a 15 s scene, corresponding to an efective spatial update rate of ≈15.6 Hz.We define Moving High, Mid, and Low using the same SNR ranges as the corresponding static conditions; the labels refer to SNR rather than movement speed.

The moving and static splits are matched by SNR range but not perfectly seed-aligned: because SNR is measured after rendering and motion changes the source–receiver distance within an utterance, a seed in one static range can land in an adjacent range once it moves, so a naive movingminus-static diference mixes the motion efect with a small reshufling across ranges. At the scale of FFASR this is minor: each moving range holds ≈1,800 utterances from the same talker, room, and noise seeds as its static counterpart, the guard bands limit boundary leakage, and the matched splits’ post-render SNR distributions stay closely aligned. The aggregate comparison is therefore stable enough for the leaderboard-level analysis reported here, and the motion penalty is consistently positive across competitive models and SNR ranges (Sec. V).

## E. Dataset Statistics and Splits

Table I lists the packed evaluation splits and Table II summarizes the acoustic statistics of the furnished-room renders. Up to 2000 samples per condition are kept by subsampling that keeps the 15 talkers balanced; the anechoic split keeps all 889 dry utterances. In total 15,637 clips are drawn from a pool of 17,700 renders, and a full ninecondition run transcribes ≈54.9 h of audio. The talkers contribute roughly evenly (per-talker counts 821–992). At the clip level the corpus is about 60% male / 40% female and 86% non-native / 14% native, tracking the 9:6 and 13:2 splits of the talker pool. The renders span the 14 rooms of Sec. III-B and more than 40 receiver positions (seated and standing listeners, wall-, shelf-, and countermounted devices), with post-render SNR roughly balanced across the three ranges. Fig. 2 shows an example scene.

TABLE II  
Acoustic statistics of the FFASR furnished-room renders.
<table><tr><td>Quantity</td><td>Min</td><td>Max</td><td>Mean</td><td>SD</td></tr><tr><td>Room volume (m³)</td><td>17.47</td><td>367.41</td><td>128.52</td><td>77.26</td></tr><tr><td>T20 (s)</td><td>0.19</td><td>1.29</td><td>0.60</td><td>0.3</td></tr><tr><td>Source-receiver dist. (m)</td><td>0.3</td><td>13.8</td><td>3.98</td><td>2.2</td></tr><tr><td>C50 (dB)</td><td>-7.8</td><td>19.61</td><td>5.3</td><td>4.5</td></tr><tr><td>Post-render SNR (dB)</td><td>-1.9</td><td>88.9</td><td>10.5</td><td>6.4</td></tr></table>

![](images/16316372190a3c13c43e96169623eb7eaf2faf68264b0a7865885e6aa900d243.jpg)  
Fig. 2. Example scene from the dataset. Left: the target speech source, the directional transient and ambient noise sources, and the receiver. Right: an example trajectory of the moving speech source.

## IV. Evaluation Protocol and Metrics

## A. Leaderboard Infrastructure

Submissions arrive through a public Hugging Face Space [49]. Each job clones the evaluation harness inside an isolated UV sandbox on an NVIDIA L4 GPU [22], loads the user-specified checkpoint, and writes a versioned JSON artifact to a private bucket; the orchestrating process never imports the inference stack, avoiding dependency collisions across model families. The packed test waveforms are never exposed to submitters, which is the mechanism that resists contamination. For the same reason we do not accept closed, API-only systems, since scoring them would require sending the held-out audio to a third party.

## B. Decoding and Metrics

Every utterance is transcribed at 16 kHz with batch size 1. References and hypotheses pass through the Whisper EnglishTextNormalizer [31] (lowercasing, punctuation removal, contraction expansion) before scoring, following the Open ASR Leaderboard protocol [7]. WER is the length-normalized Levenshtein cost

TABLE III  
Distribution of WER (%) across the 15 leading evaluated systems, summarized per condition. The lower panel reports the two derived effects in percentage points (pp).
<table><tr><td>Condition</td><td>Mean</td><td>Med</td><td>SD</td><td>Min</td><td>Max</td></tr><tr><td>Near field</td><td>4.4</td><td>4.3</td><td>0.4</td><td>3.8</td><td>5.4</td></tr><tr><td>Lab measured</td><td>29.0</td><td>27.8</td><td>6.7</td><td>20.0</td><td>45.0</td></tr><tr><td>Lab simulated</td><td>27.7</td><td>26.9</td><td>6.9</td><td>18.6</td><td>42.6</td></tr><tr><td>Static High SNR</td><td>10.1</td><td>10.2</td><td>2.2</td><td>6.7</td><td>15.1</td></tr><tr><td>Static Mid SNR</td><td>20.5</td><td>20.3</td><td>4.7</td><td>13.9</td><td>29.5</td></tr><tr><td>Static Low SNR</td><td>41.3</td><td>40.2</td><td>8.3</td><td>28.4</td><td>57.0</td></tr><tr><td>Moving High SNR</td><td>11.5</td><td>11.5</td><td>2.3</td><td>8.2</td><td>16.8</td></tr><tr><td>Moving Mid SNR</td><td>23.2</td><td>23.5</td><td>5.1</td><td>15.3</td><td>32.3</td></tr><tr><td>Moving Low SNR</td><td>43.5</td><td>42.4</td><td>8.6</td><td>30.7</td><td>60.0</td></tr><tr><td colspan="6">Derived effects (pp)</td></tr><tr><td>Sim2real Δ (meas-sim)</td><td>1.3</td><td>1.3</td><td>1.2</td><td>-2.5</td><td>2.8</td></tr><tr><td>Motion penalty, High SNR</td><td>1.3</td><td>1.5</td><td>0.5</td><td>0.1</td><td>1.9</td></tr><tr><td>Motion penalty, Mid SNR</td><td>2.7</td><td>2.6</td><td>0.9</td><td>1.2</td><td>4.7</td></tr><tr><td>Motion penalty, Low SNR</td><td>2.2</td><td>2.3</td><td>1.0</td><td>0.1</td><td>3.8</td></tr></table>

$$
\mathrm { W E R } = { \frac { S + D + I } { N } } ,\tag{4}
$$

with substitutions S, deletions $D ,$ insertions $I ,$ and reference length N [50]. WER can exceed 1 when insertions dominate; rather than clipping these values, we report them because they are diagnostic of hallucination [45].

The primary ranking metric is Average WER over four core scenarios: near-field speech plus the three static SNR ranges. This ranking should be read together with the percondition results on the live leaderboard, since changing which conditions enter the average will in general change the ordering. Throughput is reported as RTFx batch size 1 on the reference GPU, matching [7].

## V. Results and Discussion

At the time of submission the leaderboard holds 24 publicly released systems spanning CTC, RNN-T/TDT, encoder–decoder, and speech-LLM architectures, all scored under the same harness, normalizer, and decoding settings. We summarize the field through the percondition distribution in Table III and report what we find across conditions rather than a ranking of named systems, since that ranking will change as new models appear; model identities and their eficiency are shown in Fig. 3. Where we quantify an efect below, we give its mean across the 15 systems together with a 95% interval, mean $\pm 1 . 9 6 \mathrm { S D } / \sqrt { 1 5 } .$ that reflects how tightly the acrosssystem mean is pinned down by our sample of systems (not a per-utterance bootstrap, which would need the seedpaired data deferred to a future release, Sec. VI).

## A. Far-field speech is much harder than near-field

Near-field WER is uniformly low and tightly clustered (mean 4.4%, all systems within 3.8–5.4%), so clean speech barely separates the field. Reverberation and noise change this: mean WER rises to 10.1% at high SNR, 20.5% at mid SNR, and 41.3% at low SNR, and the spread across systems widens from under 2 pp near-field to roughly 30 pp at low SNR. The largest single step is from mid to low SNR (≈21 pp in the mean). We read this as the regime where the noise floor approaches speech energy and masks low-energy consonant cues, though we infer this from the error pattern rather than measure it directly. The degradation is also architecture-dependent. Aggressively compressed encoder–decoders (distilled and “lite” Whisper variants, and the most compact Canary checkpoints) match the rest of the field near-field yet fall furthest behind at low SNR, while large speech-LLM and welltrained transducer systems degrade more gracefully. For the checkpoints we evaluate this suggests that compression trades away reverberation robustness first; we cannot rule out that better-trained small models would behave difer ently. The most extreme case is outright hallucination: at low SNR the smallest Whisper checkpoints exceed 100% WER (whisper-base 117%, whisper-tiny 137%) [31] even though the same checkpoints score below 10% on clean speech [7]. Inspecting these outputs, the errors are insertion-dominated: during noise-only or low-energy segments the model emits repeated phrases or fabricated sentences [45], so the hypothesis runs far longer than the reference and WER passes 100%. Far-field WER therefore separates robust from brittle systems where clean-speech WER cannot.

## B. Source motion adds a consistent penalty

Adding source motion at matched SNR raises WER across the board: the static-to-moving penalty (Table III, lower panel) is positive for essentially every competitive system in every SNR range. The penalty is not uniform with SNR. It is largest at mid SNR (mean +2.7 pp, 95% interval [2.2, 3.2]), smaller at low SNR (+2.2 pp, [1.7, 2.7]), and smallest at high SNR (+1.3 pp, [1.0, 1.6]); all three intervals exclude zero. We had expected the penalty to grow as SNR fell, tracking the overall error rate. Instead the largest average penalty sits at mid SNR. A plausible reading is masking: at low SNR additive noise already dominates the errors and hides the kinematic contribution, while at high SNR there is little accuracy left to lose, leaving the mid-SNR regime, where the within-utterance change in direct-to-reverberant ratio is the main remaining dificulty, as the place a moving talker costs most. We ofer this as an interpretation, not a measured mechanism. Because the moving conditions are released as a beta and are matched by SNR range rather than seed-paired (Sec. III-D), we report these as an aggregate trend rather than a per-model paired estimate.

## C. High-fidelity simulation matches measurement

On the matched ofice-lab pair, where only the rendering engine difers, measured and simulated WER agree closely: across systems the mean absolute diference is 1.7 pp and the mean signed diference is +1.3 pp (95% interval [0.7, 1.9]). The direction is not consistent, measured is marginally harder for most systems but easier for some, so the residual gap behaves like small, roughly zero-mean diferences between two realistic renderings rather than a systematic bias of the simulator. The gap is also modeldependent: robust systems sit within about 1 pp, while brittle ones amplify small statistical diferences between the measured and simulated RIRs into large WER differences (up to +12.7 pp), which is why we report it per system rather than as a single constant. The agreement is not from one easy operating point: the anchor spans three configurations $( \mathrm { R T _ { 6 0 } } \approx 0 . 3 \ – 0 . 9 \mathrm { s } )$ and 44 RIR pairs over six receivers, and it reproduces on a larger model set the downstream-ASR fidelity reported for wave-based/hybrid rendering [26]. Because the anchor is a single ofice lab (Sec. VI), we read this as site-specific evidence: within those bounds, the simulated ofice-lab renders are close enough to the measured ones for this evaluation.

![](images/14f11eb9c89b65bfb387a8c46905d972273c037265335551880f8d4290436b33.jpg)  
Fig. 3. Average WER vs. RTFx on NVIDIA L4 (batch size 1). Eficient CTC/TDT models occupy the upper-left region.

## D. Speed-accuracy trade-of

Fig. 3 plots Average WER against RTFx and shows a clear accuracy–throughput tension. The most accurate systems are large and comparatively slow, anchoring the low-WER but not the high-throughput end of the Pareto frontier. The eficient end is dominated by frame-synchronous transducer and CTC decoders [35], [37], which reach competitive WER at one to two orders of magnitude higher throughput, whereas autoregressive speech-LLMs fall well below real-time. The frontier reads as a deployment guide: transducer/CTC models for latency-bound use, attention-decoder or speech-LLM systems when accuracy is paramount.

## VI. Limitations and Future Work

Mono input. FFASR currently scores single-channel renders even though the underlying RIRs are 8th-order ambisonic. We chose mono because all 24 evaluated systems accept only single-channel input, so a mono harness is what makes them directly comparable; the cost is that beamforming and array front-ends are not yet exercised.

Measured-room coverage. The measured anchor is one ofice, rendered in three configurations spanning $\mathrm { R T _ { 6 0 } } \approx 0 . 3 – 0 . 9 \mathrm { s }$ (absorber panels, furnishing, bare brick). This covers a wide range of reverberation at a single site, so the sim-to-real agreement we report should not be read as a general bound. We used the rooms we had measured pairs for; extending to more measured rooms is the clearest next step.

Speaker and accent coverage. Source material is English read speech from 15 talkers. Accent varies across talkers but is held constant across conditions by design, which is what lets each near-field score serve as a clean reference; the trade-of is that FFASR measures far-field robustness, not accent robustness, and broader demographic coverage would need a larger recording efort.

Read vs. conversational speech. The prompts are read sentences. This keeps lexical content matched across conditions, but omits the disfluency, overlap, and turntaking of spontaneous speech, which corpora such as CHiME [3] and AMI [4] target directly.

Moving-source beta. The moving conditions are a beta. The splits are matched by SNR range rather than seed-paired, and the rendering may still be revised, so we report the motion penalty as an aggregate trend rather than a per-model paired estimate.

Future work. Planned extensions include multichannel and multi-speaker input that uses the existing ambisonic RIRs, moving toward joint ASR–diarization evaluation [3], [23]; more measured rooms; a seed-paired motion metric with confidence intervals; and streaming latency tiers that report word-emission delay alongside RTFx [44].

## VII. Conclusion

FFASR is a far-field ASR benchmark whose source speech is newly recorded and whose test audio stays private, built so that a single acoustic factor changes between conditions. The practical aim of FFASR is to make far-field robustness visible in comparisons that look saturated on near-field speech. Two patterns stand out from the current field. Clean-speech WER has saturated, the systems we evaluate sit within 2 pp of each other nearfield, yet the same systems spread by more than 20 pp once reverberation and noise are added, and the smallest models tip into insertion-dominated hallucination at low SNR. Moreover, on the ofice-lab anchor, measured and simulated WER track each other to within about 1.7 pp across architectures, close enough for simulation to stand in for measurement at that site.

[1] K. Kinoshita, M. Delcroix, S. Gannot, E. A. P. Habets, R. Haeb-Umbach, W. Kellermann, V. Leutnant, R. Maas, T. Nakatani, B. Raj, A. Sehr, and T. Yoshioka, “A summary of the REVERB challenge: State-of-the-art and remaining challenges in reverberant speech processing research,” EURASIP J. Adv. Signal Process., vol. 2016, no. 1, p. 7, 2016.

[2] J. Barker, S. Watanabe, E. Vincent, and J. Trmal, “The fifth ‘CHiME’ speech separation and recognition challenge: Dataset, task and baselines,” in Proc. Interspeech, 2018, pp. 1561–1565.

[3] S. Watanabe, M. Mandel, J. Barker, E. Vincent et al., “CHiME-6 challenge: Tackling multispeaker speech recognition for unsegmented recordings,” in Proc. 6th Int. Workshop on Speech Processing in Everyday Environments (CHiME), 2020, pp. 1–7.

[4] J. Carletta, S. Ashby, S. Bourban, M. Flynn et al., “The AMI meeting corpus: A pre-announcement,” in Proc. Int. Workshop Machine Learning for Multimodal Interaction (MLMI), ser. LNCS, vol. 3869. Springer, 2006, pp. 28–39.

[5] C. Richey, M. A. Barrios, Z. Armstrong, C. Bartels, H. Franco, M. Graciarena, A. Lawson, M. K. Nandwana, A. Staufer, J. van Hout, P. Gamble, J. Hetherly, C. Stephenson, and K. Ni, “Voices obscured in complex environmental settings (VOiCES) corpus,” in Proc. Interspeech, 2018, pp. 1566–1570.

[6] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “LibriSpeech: An ASR corpus based on public domain audio books,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP), 2015, pp. 5206–5210.

[7] V. Srivastav, S. Zheng, E. Bezzam, E. L. Bihan, A. Moumen, and S. Gandhi, “Open asr leaderboard: Towards reproducible and transparent multilingual speech recognition evaluation,” arXiv preprint arXiv:2510.06961, 2025.

[8] T. Ko, V. Peddinti, D. Povey, M. L. Seltzer, and S. Khudanpur, “A study on data augmentation of reverberant speech for robust speech recognition,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP), 2017, pp. 5220–5224.

[9] C. Kim, A. Misra, K. Chin, T. Hughes, A. Narayanan, T. N. Sainath, and M. Bacchiani, “Generation of large-scale simulated utterances in virtual rooms to train deep-neural networks for farfield speech recognition in Google Home,” in Proc. Interspeech, 2017, pp. 379–383.

[10] J. Eaton, N. D. Gaubitch, A. H. Moore, and P. A. Naylor, “Estimation of room acoustic parameters: The ACE challenge,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 24, no. 10, pp. 1681–1693, 2016.

[11] I. Szöke, M. Skácel, L. Mošner, J. Paliesek, and J. Černocký, “Building and evaluation of a real room impulse response dataset,” IEEE J. Sel. Topics Signal Process., vol. 13, no. 4, pp. 863–876, 2019.

[12] E. Hadad, F. Heese, P. Vary, and S. Gannot, “Multichannel audio database in various acoustic environments,” in Proc. Int. Workshop Acoust. Signal Enhancement (IWAENC), 2014, pp. 313–317.

[13] J. B. Allen and D. A. Berkley, “Image method for eficiently simulating small-room acoustics,” J. Acoust. Soc. Am., vol. 65, no. 4, pp. 943–950, 1979.

[14] R. Scheibler, E. Bezzam, and I. Dokmanić, “Pyroomacoustics: A python package for audio room simulation and array processing algorithms,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP), 2018, pp. 351–355.

[15] D. Díaz-Guerra, A. Miguel, and J. R. Beltrán, “gpuRIR: A python library for room impulse response simulation with GPU acceleration,” Multimedia Tools and Applications, vol. 80, no. 4, pp. 5653–5671, 2021.

[16] F. Pind, A. P. Engsig-Karup, C.-H. Jeong, J. S. Hesthaven, M. S. Mejling, and J. Strømann-Andersen, “Time domain room acoustic simulations using the spectral element method,” J. Acoust. Soc. Am., vol. 145, no. 6, pp. 3299–3310, 2019.

[17] Treble Technologies, “Treble software development kit,” https: //www.treble.tech/software-development-kit, 2025.

[18] S. S. Mullins, G. Götz, E. Bezzam, S. Zheng, and D. G. Nielsen, “Treble10: A high-quality dataset for far-field speech recognition, dereverberation, and enhancement,” arXiv preprint arXiv:2510.23141, 2025.

[19] Z. Tang, R. Aralikatti, A. Ratnarajah, and D. Manocha, “GWA: A large high-quality acoustic dataset for audio processing,” in Proc. ACM SIGGRAPH, 2022.

[20] C. Chen, C. Schissler, S. Garg, P. Kobernik, A. Clegg, P. Calamia, D. Batra, P. Robinson, and K. Grauman, “SoundSpaces 2.0: A simulation platform for visual-acoustic learning,” in Advances in Neural Information Processing Systems (NeurIPS), 2022.

[21] S. Saini and J. Peissig, “Hifi-harp: A high-fidelity 7th-order ambisonic room impulse response dataset,” in ICASSP 2026- 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2026, pp. 14 632–14 636.

[22] Hugging Face, “Hugging face jobs documentation,” https:// huggingface.co/docs/huggingface\_hub/en/guides/jobs, 2024.

[23] Z. Chen, T. Yoshioka, L. Lu, T. Zhou, Z. Meng, Y. Luo, J. Wu, X. Xiao, and J. Li, “Continuous speech separation: Dataset and analysis,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP), 2020, pp. 7284–7288.

[24] N. Khan, S. Nisar, M. A. Khan, F. Afghah, Y. A. U. Rehman, and G. Barb, “A novel privacy-preserving framework for lowresource asr using federated self-supervised learning,” IEEE Access, vol. 13, pp. 217 156–217 175, 2025.

[25] A. Milo, J. F. Einarsson, Ú. Einarsson, and F. Pind, “Treble auralizer: a real time web audio engine enabling 3dof auralization of simulated room acoustics designs,” in 2023 Immersive and 3D Audio: from Architecture to Automotive (I3DA). IEEE, 2023, pp. 1–8.

[26] G. Götz, D. G. Nielsen, S. Gudjónsson, and F. Pind, “Roomacoustic simulations as an alternative to measurements for audio-algorithm evaluation,” IEEE Access, vol. 13, pp. 214 000– 214 008, 2025.

[27] F. Pind, J. F. Einarsson, S. Guðjónsson, M. Cosnefroy, J. Pedersen, J. B. Stefánsson, and A. Milo, “A novel wave-based virtual acoustics and spatial audio framework,” in Audio Engineering Society Conference: 2022 AES International Conference on Audio for Virtual and Augmented Reality. Audio Engineering Society, 2022.

[28] P. Puzalowski, J. Pedersen, and C.-H. JEONG, “Optimizing a small room for critical listening with treble,” in INTER-NOISE and NOISE-CON Congress and Conference Proceedings, vol. 270, no. 9. Institute of Noise Control Engineering, 2024, pp. 2317–2328.

[29] G. Götz, A. Milo, S. Guðjónsson, D. G. Nielsen, J. Pedersen, and F. Pind, “Improving multichannel speech enhancement through accurate room-acoustic simulations,” in Proc. Interspeech, 2026.

[30] A. Milo, G. Götz, S. Guðjónsson, D. G. Nielsen, J. Pedersen, and F. Pind, “Training deepfilternet with accurate room acoustic simulations improves single-channel speech enhancement,” in Proc. Int. Workshop Acoust. Signal Enhancement (IWAENC), 2026.

[31] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proc. Int. Conf. Machine Learning (ICML), vol. 202, 2023, pp. 28 492–28 518.

[32] S. Gandhi, P. von Platen, and A. M. Rush, “Distil-whisper: Robust knowledge distillation via large-scale pseudo labelling,” 2023, arXiv:2311.00430.

[33] K. Kamahori, J. Kasai, N. Kojima, and B. Kasikci, “Liteasr: Eficient automatic speech recognition with low-rank approximation,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025, pp. 3430–3442.

[34] A. Gulati, J. Qin, C.-C. Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han, S. Wang, Z. Zhang, Y. Wu, and R. Pang, “Conformer: Convolution-augmented transformer for speech recognition,” in Proc. Interspeech, 2020, pp. 5036–5040.

[35] D. Rekesh, N. R. Koluguri, S. Kriman, S. Majumdar, V. Noroozi, H. Huang, O. Hrinchuk, K. C. Puvvada, A. Kumar, J. Balam, and B. Ginsburg, “Fast Conformer with linearly scalable attention for eficient speech recognition,” in Proc. IEEE Automat. Speech Recognit. Understanding Workshop (ASRU), 2023, pp. 1–8.

[36] A. Graves, “Sequence transduction with recurrent neural networks,” 2012, arXiv:1211.3711.

[37] H. Xu, F. Jia, S. Majumdar, H. Huang, S. Watanabe, and B. Ginsburg, “Eficient sequence transduction by jointly predicting tokens and durations,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, Eds., vol. 202. PMLR, 23–29 Jul 2023, pp. 38 462–38 484. [Online]. Available: https://proceedings.mlr.press/v202/xu23g.html

[38] K. C. Puvvada, P. Żelasko, H. Huang, O. Hrinchuk, N. R. Koluguri, K. Dhawan, S. Majumdar, E. Rastorgueva, Z. Chen, V. Lavrukhin, J. Balam, and B. Ginsburg, “Less is more: Accurate speech recognition & translation without web-scale data,” in Proc. Interspeech, 2024, pp. 3964–3968.

[39] Y. Peng, M. Shakeel, Y. Sudo, W. Chen, J. Tian, C.-J. Lin, and S. Watanabe, “OWSM v4: Improving open whisper-style speech models via data scaling and cleaning,” in Proc. Interspeech, 2025, pp. 2225–2229.

[40] Y. Peng, Y. Sudo, M. Shakeel, and S. Watanabe, “OWSM-CTC: An open encoder-only speech foundation model for speech recognition, translation, and language identification,” in Proc. Annual Meeting Assoc. Comput. Linguistics (ACL), 2024. [Online]. Available: https://aclanthology.org/2024.acl-long.549

[41] G. Saon, A. Dekel, A. Brooks, T. Nagano et al., “Granitespeech: Open-source speech-aware LLMs with strong English ASR capabilities,” 2025, arXiv:2505.08699.

[42] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, 2020, pp. 12 449–12 460.

[43] M. Ravanelli, T. Parcollet, A. Moumen, S. de Langen, C. Subakan, P. Plantinga et al., “Open-source conversational AI with SpeechBrain 1.0,” J. Machine Learning Research, vol. 25, no. 333, pp. 1–11, 2024.

[44] N. Jefries, E. King, M. Kudlur, G. Nicholson, J. Wang, and P. Warden, “Moonshine: Speech recognition for live transcription and voice commands,” 2024, arXiv:2410.15608.

[45] A. Koenecke, A. S. G. Choi, K. X. Mei, H. Schellmann, and M. Sloane, “Careless whisper: Speech-to-text hallucination harms,” in Proc. ACM Conf. Fairness, Accountability, and Transparency (FAccT), 2024.

[46] P. Götz, C. Tuna, A. Walther, and E. A. P. Habets, “AID: Opensource anechoic interferer dataset,” in Proc. Int. Workshop Acoust. Signal Enhancement (IWAENC), 2022, pp. 1–5.

[47] ITU-T, “Objective measurement of active speech level,” International Telecommunication Union, Recommendation ITU-T P.56, 2011.

[48] S. S. Mullins and H. Sampedro Llopis, “Comparing three movement simulation algorithms from discrete impulse responses,” in Proc. AES Int. Conf. on Audio for Virtual and Augmented Reality and Immersive Games. Paris, France: Audio Engineering Society, 2026.

[49] Treble Technologies, “FFASR evaluation leaderboard,” Hugging Face Space, 2026. [Online]. Available: https://huggingface.co/ spaces/treble-technologies/FFASR\_Leaderboard-storage

[50] A. C. Morris, V. Maier, and P. Green, “From WER and RIL to MER and WIL: Improved evaluation measures for connected speech recognition,” in Proc. Interspeech, 2004, pp. 2765–2768.