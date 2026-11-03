# MultiMental-IAS: A Multimodal Driver Mental State Analysis Dataset for Context-Aware Non-IID Evaluation

Baran Can Gül   
Institute of Industrial Automation and   
Software Engineering (IAS),   
University of Stuttgart   
Stuttgart, Germany   
baran-can.guel@ias.uni-stuttgart.de   
Nagendra Pal Singh   
Institute of Industrial Automation and   
Software Engineering (IAS),   
University of Stuttgart   
Stuttgart, Germany   
st191659@stud.uni-stuttgart.de   
Anjana Arvind Naik   
Institute of Industrial Automation and   
Software Engineering (IAS),   
University of Stuttgart   
Stuttgart, Germany   
anjananaik04@gmail.com

## Abstract

Robust driver-state modelling requires synchronized physiological, behavioural, and contextual signals, yet existing open resources typically target a single cognitive state, omit continuous vehicle context, or do not define evaluation protocols that expose intersubject heterogeneity. We present MultiMental-IAS, a multimodal driver mental-state dataset collected in a cockpit-in-the-loop simulation built on CARLA. It contains vehicle telemetry with scenariocontext channels, four-channel EEG and three-channel PPG from a Muse 2 headband, and cabin-facing video from 23 participants, recorded in 148 trials (11.6 hours) under six induced states: Normal, Distracted, Alerted, Stressed, Relaxed, and Drowsy. We document the acquisition pipeline in detail and provide a benchmark with one protocol for all models: chronological within-subject splits, participant-disjoint cross-subject folds, validation-based model selection, and ablations over modalities, scenario-context channels, and signal normalization. Multimodal fusion reaches a macro-F1 of 0.94 for drivers seen during training. For unseen drivers, fusing driving dynamics with per-window normalized EEG/PPG raises macro-F1 from 0.408 (telemetry alone) to 0.485, consistently across all five folds, while the gap between the two protocols quantifies the inter-driver heterogeneity that personalized models must address. MultiMental-IAS thus provides a concrete testbed for non-IID generalization, personalization, and federated learning in driver monitoring.

## Keywords

multimodal dataset; driver monitoring; mental state; EEG; PPG; telemetry; non-IID; cross-subject evaluation; personalization; federated learning

## 1 Introduction

Transitions between automation levels in semi-autonomous and software-defined vehicles require the human driver to maintain or promptly resume cognitive engagement [21]. Fatigue, distraction, stress, and drowsiness each impair reaction time and situational awareness in distinct ways, and their signatures overlap considerably when individual sensor streams are considered in isolation [17]. Combining vehicle dynamics, physiological measurements, and cabin-facing video ofers a more complete picture of driver state, and multimodal learning has consequently attracted considerable attention in automotive human-centred computing [19].

Publicly available datasets, however, still present three recurring limitations. First, most resources focus on a single target state, typically drowsiness or vigilance, and do not capture the broader spectrum of overlapping cognitive conditions relevant to driving safety [13, 29]. Second, few datasets synchronize physiological streams with continuous vehicle context channels such as traffic density and illumination, even though these factors modulate arousal and workload [22]. Third, as summarized in Table 1, existing datasets rarely define evaluation protocols that separate withinsubject from cross-subject performance, which hinders assessment of generalization across users.

This third limitation is particularly consequential in safety-critical deployments. Physiological baselines vary substantially across individuals: resting cortical rhythms, baseline heart-rate variability (HRV), and the mapping between mental state and EEG spectral power difer with age, health status, and experience [18]. Models optimized on pooled cohorts frequently degrade on unseen users, a problem formalized in the federated learning literature as non-IID data [9, 12]. Addressing this gap requires resources that treat inter-subject heterogeneity as a first-class property rather than an uncontrolled nuisance.

We introduce MultiMental-IAS, a cockpit-in-the-loop dataset for six-class driver mental-state analysis. Among the resources in Table 1, it is the only one that combines vehicle telemetry (including scenario context channels), EEG, PPG, and video across six mental states. Our contributions are:

• A four-modality dataset (telemetry, EEG, PPG, and cabinfacing video) of 148 trials from 23 participants across six induced driver mental states, released with the original host timestamps, corrected channel labels, and per-trial metadata.

• A detailed account of the acquisition and preprocessing pipeline, with per-trial metadata (durations, efective sampling rates, scenario settings) that lets users select and preprocess the data for their own analyses.

• A reproducible benchmark with one protocol for all models: a chronological within-subject split with a held-out validation segment, participant-disjoint cross-subject folds, validation-based model selection, and ablations over modalities, scenario-context channels, and signal normalization.

• Reference results showing that multimodal fusion reaches a macro-F1 of 0.94 within subject and that fusing driving dynamics with normalized physiology improves recognition for unseen drivers (0.485 vs. 0.408, better in all five folds).

Table 1: Comparison with representative open datasets in multimodal driver monitoring and physiological afective computing. Modalities: E = EEG/EOG, P = PPG/ECG/EDA/EMG, V = video, T = telemetry. <sup>�</sup>Face video for 22 of the 32 participants.
<table><tr><td>Dataset</td><td>Year</td><td>Subj.</td><td>Modalities</td><td>Primary Labels</td><td>Acquisition setting</td><td>States</td></tr><tr><td>MAHNOB-HCI [20]</td><td>2012</td><td>27</td><td>E, P, V</td><td>Affective response</td><td>Lab (video stimuli)</td><td>5</td></tr><tr><td>DEAP [10]</td><td>2012</td><td>32</td><td>E, P, Va</td><td>Valence/arousal/dominance/liking</td><td>Lab (music-video stimuli)</td><td>cont.</td></tr><tr><td>DROZY [13]</td><td>2016</td><td>14</td><td>E, P, V</td><td>Drowsiness / KSS</td><td>Lab (PVT, sleep deprivation)</td><td>cont.</td></tr><tr><td>SEED-VIG [29]</td><td>2017</td><td>23</td><td>E</td><td>Vigilance / PERCLOS</td><td>Driving simulator</td><td>cont.</td></tr><tr><td>Distracted Driving [22]</td><td>2017</td><td>68</td><td>P, V, T</td><td>Distraction condition</td><td>Driving simulator</td><td>4</td></tr><tr><td>AffectiveROAD [8]</td><td>2018</td><td>10</td><td>P, V</td><td>Stress / attention</td><td>Real-world urban driving</td><td>cont.</td></tr><tr><td>DD-Database [16]</td><td>2023</td><td>10</td><td>E, P</td><td>Drowsiness events</td><td>Driving simulator</td><td>1</td></tr><tr><td>MPDB [23]</td><td>2024</td><td>35</td><td>E, P, T</td><td>Driving behaviour</td><td>Driving simulator</td><td>5</td></tr><tr><td>MultiMental-IAS (ours)</td><td>2026</td><td>23</td><td>E, P, V, T</td><td>Driver mental states</td><td>CARLA simulator + cabin 6</td><td></td></tr></table>

## 2 Related Work

Research on inferring driver cognitive state from physiological and behavioural signals spans three decades, yet gaps in modality coverage, contextual sensing, and cross-subject benchmarking persist [2, 19].

## 2.1 Physiological and Behavioural Biomarkers

EEG spectral dynamics provide the most direct cortical measure of driver state. Alpha-band power (8–13 Hz) is commonly associated with reduced arousal and tends to increase during fatigue, while task-related cognitive load is often accompanied by alpha suppression. Frontal theta activity (4–8 Hz), in contrast, has been linked to sustained mental efort and increases with workload [4, 27].

Cardiac signals ofer a complementary autonomic perspective: RMSSD and the LF/HF ratio of HRV respond to acute stress and are used as markers of autonomic activation [11, 24]. In-cabin video contributes behavioural cues (blink rate, eyelid closure, head pose) that are informative for drowsiness and distraction, while vehicle telemetry captures the consequences of internal state changes in observable control behaviour [1, 22]. Because physiological signals reflect internal changes that can precede visible behavioural deterioration, the modalities are expected to carry complementary information [2]. Consumer-grade dry-electrode headbands such as the Muse 2 have been shown to capture workload-related changes in frontal and temporal band power [15, 28].

## 2.2 Driver Monitoring Datasets

SEED-VIG [29] and DROZY [13] are foundational EEG and physiology benchmarks for vigilance and drowsiness estimation, yet both target a single vigilance or drowsiness dimension. The Distracted Driving dataset [22] provides rich simulator telemetry and peripheral physiology across distraction conditions from 68 par ticipants, but contains no EEG. AfectiveROAD [8] captures stress and attention during real-world urban driving with wearable sensors, but is limited to ten participants and a single continuous label dimension. MAHNOB-HCI [20] and DEAP [10] are influential mul timodal afect datasets, but are grounded in passive media viewing without driving context or vehicle telemetry. More recent simulatorbased resources combine EEG with other physiological signals in driving: DD-Database [16] annotates drowsiness events only, and

MPDB [23] labels five driving manoeuvres rather than mental states. MultiFuser [26] demonstrates multimodal transformer fusion for driver action recognition rather than internal cognitive state.

## 2.3 Non-IID Generalization and Federated Learning

Intra- and inter-subject variability in EEG induces covariate shift that impedes model transferability [18], producing non-IID data distributions under which models trained on pooled cohorts can degrade markedly on unseen users [12]. Federated learning [14] enables collaborative training across heterogeneous clients without sharing raw data. Its feasibility in automotive contexts has been shown for comfort feature prediction and collaborative sensing in software-defined vehicles [5, 6], and for real-time multimodal emotion recognition in embedded settings [7]. None of these eforts addressed mental-state classification from synchronized physiological, behavioural, and vehicle streams across multiple states, which MultiMental-IAS is designed to support.

## 3 The MultiMental-IAS Dataset

## 3.1 Apparatus and Data Acquisition

Data were collected in a cockpit-in-the-loop simulation running CARLA 0.9.13 [3] on a workstation with an Intel Core i9-12900K processor, 64 GB RAM, and an NVIDIA RTX 3090 GPU. The cockpit used a Logitech G27 force-feedback steering wheel and pedal set (Figure 1). Figure 2 shows the acquisition and processing pipeline.

Physiological signals were acquired with a Muse 2 headband, which provides four dry EEG channels (TP9, AF7, AF8, TP10; native rate 256 Hz), an auxiliary channel, and three optical PPG channels (PPG1–PPG3; native rate 64 Hz). The headband streamed over Bluetooth, and the recording host received the streams through the Lab Streaming Layer (LSL) using pylsl. Vehicle telemetry was obtained through the CARLA Python API and comprises position (�, �, �), orientation (yaw, pitch, roll), speed, 3-axis acceleration, and scenario context variables: trafic density, road type, sun altitude, time of day, ambient light, snow, and travelled distance. Cabin-facing video was recorded with an Intel RealSense D455 camera at 640 × 480 pixels; the camera shows the whole driver seat and part of the room behind it.

![](images/3376e447ed6787a145bbc2bd3dbfe347100b520cd9d0a4bf920a36d575fc4dda.jpg)  
Figure 1: MultiMental-IAS data-collection setup: IAS cockpit with Logitech G27 steering wheel and pedals, Muse 2 EEG/PPG headband, and Intel RealSense D455 cabin-facing camera.

Efective sampling rates: The recording logger wrote one row per stream in each acquisition cycle, timestamped with the host wall clock (millisecond resolution). The telemetry, EEG, and PPG logs of a trial therefore share one timestamp grid, and their efective rate is the rate of that cycle rather than the native sensor rate: the median over all trials is 20.3 Hz (range 9.7–30.0 Hz). In the Drowsy condition, which is driven at night, the cycle ran at a median of 11.3 Hz (20.1–21.4 Hz for the other conditions), as rendering the night scene requires more computation. The logs therefore capture the slow components of the physiological signals, such as lowfrequency EEG activity and the pulse waveform. The benchmark resamples all streams to a common band (Section 4.3), and efective rates are listed per trial in metadata.csv.

## 3.2 Participants

Twenty-three participants took part voluntarily (identifiers P11– P33; IDs P1–P7 belong to an earlier pilot that is not part of the release). Participants were aged 19–63 years (�=30.5, ��=9.8, median 27). Seven did not hold a driving licence; six reported corrected vision problems, two low blood pressure, and one diabetes. Participants with active cardiac arrhythmia, a pacemaker, self-reported photosensitive epilepsy, or severe simulator sickness were excluded at screening. Three of the authors took part as participants. All participants gave written informed consent.

## 3.3 Induction Protocol

Each condition was recorded as a block of up to five minutes. States were induced through guided mental recall, behavioural imitation, explicit task instructions, pre-trial preparatory activities, and CARLA scenario settings. Specifically, Normal was recorded under calm driving analogous to a daily commute; Distracted involved simulated secondary-task glances and infotainment interactions; Alerted required sustained hazard anticipation; Stressed combined pre-trial stair-climbing, stressful music, and urgency instructions; Relaxed used soothing music, recall of a peaceful memory, and minimal task demand; and Drowsy recordings were scheduled when participants reported fatigue and included fatigue-related behavioural cues such as slow blinking and head nodding.

Table 2 summarizes the conditions together with the illumination recorded in the context channels. Five conditions were driven in daylight (sun altitude 45<sup>◦</sup>); Drowsy was driven at night in 28 of its 30 trials and at dusk in the other two. Because illumination and trafic settings are part of the induction, these context channels covary with the label by design.

## 4 Data Organization and Preprocessing

## 4.1 Release Structure

The release is organized by participant, condition, and numbered trial ({participant}/{condition}/{trial}/), as shown in Figure 3. Each trial folder contains the telemetry log (data.csv), the PPG log (ppg.csv), the EEG log (eeg.csv), and the cabin video (face.mp4). Every CSV file starts with time\_s, the time since the first sample of the trial computed from the host timestamps, followed by the original timestamp (timestamp\_local) and the channels. A metadata.csv file lists, per trial, the start time, duration, row count, efective rate, largest gap, logged context settings, and video properties of every stream.

The release contains 148 trials (Table 2): 20 participants contributed all six conditions, and three contributed four or five. For several participants, additional recordings were collected in selected conditions to increase the number of samples; these additional trials vary in length, and 13 trials in total are shorter than five minutes (13–222 s). The total logged duration is 11.6 hours, and 146 of the 148 trials include video.

## 4.2 Release Generation

The release is generated directly from the original logs by a provided script (fix\_dataset.py). It derives time\_s from the host timestamps, labels the EEG channels TP9, AF7, AF8, TP10, and AUX, and writes the per-trial metadata. Videos were normalized to a duration of 300 s during post-processing and are mapped linearly onto the logged duration of each trial.

## 4.3 Benchmark Preprocessing

Each CSV stream is interpolated onto a uniform 20 Hz grid using time\_s, low-pass filtered with a zero-phase fourth-order Butterworth filter at 4 Hz, and decimated to 10 Hz. Because every trial, including the Drowsy trials logged at about 11 Hz, then carries the same 0–4 Hz band, the sampling rate can no longer give away the condition. For EEG, this retains the delta band only. Telemetry is represented by speed, three-axis acceleration, yaw rate (from the unwrapped yaw angle), pitch, and roll; the scenario channels (trafic density, sun altitude, ambient light, snow) and the map position are used only in explicitly marked ablations. Videos are decoded at 5 fps, converted to grayscale, and downscaled to 48 × 48 pixels.

![](images/25472236d6d04145e099a5e927f1639c3d8f22a738ccb83a8ae63a6c09515b11.jpg)  
Figure 2: MultiMental-IAS acquisition and processing pipeline: CARLA telemetry, Muse 2 EEG/PPG streams received via LSL and Intel RealSense video are recorded per trial and stored hierarchically; parallel unimodal encoders and a fusion head form the reference model. The LLM-assisted confirmation loop is an optional acquisition aid (Section 4.4).

Table 2: Conditions, logged illumination (from the context channels of all 148 trials), and hypothesized multimodal tendencies (expectations, not measured results).
<table><tr><td>State</td><td>Mental focus</td><td>Illumination (logged)</td><td>Trials</td><td>Hypothesized tendencies</td></tr><tr><td>Normal</td><td>Routine calm driving</td><td>day (22), dusk (1)</td><td>23</td><td>Stable speed, smooth steering, moderate physio- logical baseline</td></tr><tr><td>Distracted</td><td>Split attention</td><td>day (22), dusk (2)</td><td>24</td><td>Speed variability, lane deviation, gaze shifts, de- layed braking</td></tr><tr><td>Alerted</td><td>Sustained high vigilance</td><td>day (20), dusk (2)</td><td>22</td><td>Rapid corrections, heightened posture, sup- pressed frontal alpha</td></tr><tr><td>Stressed</td><td>Urgency and arousal</td><td>day (22), dusk (1)</td><td>23</td><td>Harsh braking, aggressive steering, reduced HRV, elevated sympathetic tone</td></tr><tr><td>Relaxed</td><td>Low arousal but attentive day (24), dusk (2)</td><td></td><td>26</td><td>Smooth low-variance control, elevated HRV, higher relative alpha</td></tr><tr><td>Drowsy</td><td>Low vigilance / fatigue</td><td>night (28), dusk (2)</td><td>30</td><td>Delayed corrections, inconsistent speed, physio- logical slowing</td></tr></table>

## 4.4 LLM-Assisted Human-in-the-Loop Acquisition Tool

The acquisition software includes an optional labelling aid. Every 10 s it samples four camera frames, detects the face with MediaPipe, and computes simple EEG, PPG, and telemetry statistics. It sends the frames and statistics to a locally hosted LLaVA-7B model (Ollama), which returns a label from the six conditions, a confidence score, and a short rationale. When the same label is returned for four consecutive windows (40 s), the operator is asked to accept it or to select a diferent label. The labels used in this paper are the induced condition of each trial, not LLM outputs.

## 5 Benchmark

The benchmark asks two questions: how well can each modality and combination recognize the induced state of a driver whose other recordings were seen during training, and how well does this transfer to drivers who were never seen? A single script (train\_ benchmark.py) implements the protocol for all models.

![](images/7531bda787fae2f34343f55fc8a3d8c8e9b0259a7201aa7594f804098b8751c1.jpg)  
Figure 3: Hierarchical organization: one folder per participant, condition, and trial, each containing data.csv, ppg.csv, eeg.csv, and face.mp4.

## 5.1 Protocol

All models classify 10-s windows with a 5-s stride into the six conditions; each window inherits the label of its trial. Within-subject: in every trial, windows lying entirely in the first 70% of the trial are used for training, those in the 70–85% segment for validation, and those in the last 15% for testing. Windows that cross a boundary are discarded, so no sample appears in two splits (5,548 / 962 / 962 windows). Cross-subject: participants are divided into five disjoint folds; in each fold, three of the training participants are held out for validation, and all windows of the test participants are evaluated (8,047 windows over the five folds). Inputs are z-scored with statistics of the training windows. Hyperparameters are selected on the within-subject validation split (seed 0), and the checkpoint of each run is the one with the best validation macro-F1. The test split is evaluated once per run. We report the mean and standard deviation over three seeds (within-subject) and over the five folds (cross-subject). Chance level for macro-F1 and accuracy is about 1/6.

## 5.2 Models

Sequence encoders: Telemetry, EEG, PPG, and EEG+PPG are encoded by a temporal convolutional network with four blocks (64 channels, kernel size 7, stride 2 after the first block, batch normalization, ReLU), followed by mean and max pooling and a 128-unit embedding. We also evaluate mean-removed (“m.-r.”) physiological inputs, in which each channel’s mean is subtracted per window. This removes recording-specific ofsets, such as the PPG signal level that depends on sensor contact, and forces the model to use signal dynamics.

Video encoder: A 3D CNN with four convolutional blocks (16, 32, 64, 64 filters; $3 ^ { 3 }$ kernels; batch normalization), max pooling, and global average pooling produces a 128-unit embedding from 25-frame clips (2.5 fps) of 48 × 48 grayscale frames, with random horizontal flips during training.

Training: Each encoder has a classification head (64 units) and is trained with AdamW (weight decay $1 0 ^ { - 4 } )$ , class-weighted crossentropy, gradient clipping at 1.0, learning-rate halving on validation plateaus, and early stopping on validation macro-F1. Searched values: learning rate $\{ 1 0 ^ { - 3 } , 3 \cdot 1 0 ^ { - 4 } \}$ and dropout {0.2, 0.5} for the sequence encoders; learning rate $\{ 1 0 ^ { - 3 } , 3 \cdot 1 0 ^ { - 4 } \}$ for the video encoder.

![](images/3f502d18410f01ea61ccc95b9d04494faf560112a4af24ce97940bdbf6df4719.jpg)  
Figure 4: Macro-F1 for drivers seen during training (withinsubject) and unseen drivers (cross-subject). Fusion of driving dynamics with mean-removed EEG/PPG generalizes best to new drivers.

Fusion: The encoders trained in each run are frozen; their embeddings are z-scored and concatenated, and a fusion head (256 and 128 units) is trained on the same split, with dropout, modality dropout ({0, 0.2}), and learning rate selected on the validation split. A missing modality (two trials without video) is represented by a zero embedding. As a training-free reference, we also report late fusion by averaging the encoders’ class probabilities.

## 5.3 Results

Table 3 reports all configurations and Figure 4 contrasts the two protocols for the main ones.

Seen vs. unseen drivers: Physiology and video achieve high withinsubject scores (EEG+PPG 89.7%, video 86.7%), and both benefit from information that is specific to an individual and a session, such as the PPG signal level, which depends on sensor contact, and the driver’s appearance. Such individual characteristics are valuable for personalized monitoring, but they do not carry over to new drivers (EEG+PPG 19.6%, video 30.8% cross-subject). Removing the per-window mean from the physiological signals makes the models rely on signal dynamics, which generalize better (PPG from 15.7% to 31.2% cross-subject).

Scenario context: Adding trafic density, sun altitude, ambient light, and snow raises telemetry from 0.408 to 0.532 cross-subject macro-F1, better in all five folds. Since these channels describe the scenario that accompanies each induced state (e.g., Drowsy is driven at night), we report the main results without them, so that the models are evaluated on the driver’s own signals; the context channels remain available for context-aware modelling.

Table 3: Six-class results (mean ± SD; within-subject: 3 seeds, cross-subject: 5 participant-disjoint folds). “Scenario” = trafic density, sun altitude, ambient light, snow; $" \mathbf { m } . \mathbf { r } . ^ { \prime \prime } =$ per-window mean removed. Bold: main results without scenario channels or recording ofsets.
<table><tr><td rowspan="2">Group</td><td rowspan="2">Input</td><td colspan="2">Within-subject</td><td colspan="2">Cross-subject (unseen drivers)</td></tr><tr><td>Acc. (%)</td><td>Macro-F1</td><td> $\operatorname { A c c . } \left( \% \right)$ </td><td>Macro-F1</td></tr><tr><td rowspan="3">Telemetry</td><td>Dynamics</td><td> $5 2 . 1 \pm 2 . 6$ </td><td> $0 . 5 2 4 \pm 0 . 0 2 7$ </td><td> $4 0 . 8 \pm 5 . 3$ </td><td> $0 . 4 0 8 \pm 0 . 0 5 0$ </td></tr><tr><td>Dynamics + scenario</td><td> $6 4 . 9 \pm 1 . 4$ </td><td> $0 . 6 4 8 \pm 0 . 0 1 3$ </td><td> $5 4 . 0 \pm 7 . 8$ </td><td> $0 . 5 3 2 \pm 0 . 0 5 6$ </td></tr><tr><td>Dynamics + scenario + position</td><td> $6 6 . 3 \pm 0 . 9$ </td><td> $0 . 6 6 1 \pm 0 . 0 0 9$ </td><td> $5 2 . 3 \pm 6 . 9$ </td><td> $0 . 5 1 9 \pm 0 . 0 4 9$ </td></tr><tr><td rowspan="6">Physiology</td><td>EEG</td><td> $6 9 . 3 \pm 1 . 8$ </td><td> $0 . 6 9 2 \pm 0 . 0 1 7$ </td><td> $2 5 . 9 \pm 6 . 0$ </td><td> $0 . 2 5 3 \pm 0 . 0 6 0$ </td></tr><tr><td>PPG</td><td> $8 0 . 7 \pm 0 . 2$ </td><td> $0 . 8 0 4 \pm 0 . 0 0 1$ </td><td> $1 5 . 7 \pm 1 . 4$ </td><td> $0 . 1 4 0 \pm 0 . 0 2 9$ </td></tr><tr><td>EEG + PPG</td><td> $8 9 . 7 \pm 0 . 5$ </td><td> $0 . 8 9 8 \pm 0 . 0 0 5$ </td><td> $1 9 . 6 \pm 4 . 6$ </td><td> $0 . 1 8 4 \pm 0 . 0 3 7$ </td></tr><tr><td>EEG (m.-r.)</td><td> $6 0 . 3 \pm 2 . 2$ </td><td> $0 . 6 0 1 \pm 0 . 0 2 2$ </td><td> $2 8 . 7 \pm 4 . 6$ </td><td> $0 . 2 6 6 \pm 0 . 0 4 4$ </td></tr><tr><td>PPG (m.-r.)</td><td> $4 7 . 0 \pm 1 . 4$ </td><td> $0 . 4 5 5 \pm 0 . 0 0 9$ </td><td> $3 1 . 2 \pm 3 . 8$ </td><td> $0 . 2 7 6 \pm 0 . 0 4 4$ </td></tr><tr><td> $\mathrm { E E G } + \mathrm { P P G } \left( \mathrm { m . - r . } \right)$ </td><td> $5 9 . 5 \pm 0 . 9$ </td><td> $0 . 5 8 7 \pm 0 . 0 1 2$ </td><td> $2 9 . 2 \pm 3 . 0$ </td><td> $0 . 2 6 1 \pm 0 . 0 3 6$ </td></tr><tr><td>Video</td><td>48× 48 grayscale</td><td> $8 6 . 7 \pm 5 . 9$ </td><td> $0 . 8 6 6 \pm 0 . 0 6 0$ </td><td> $3 0 . 8 \pm 4 . 5$ </td><td> $0 . 2 4 1 \pm 0 . 0 7 1$ </td></tr><tr><td rowspan="7">Fusion</td><td>Telemetry + EEG/PPG</td><td> $9 1 . 6 \pm 1 . 0$ </td><td> $0 . 9 1 6 \pm 0 . 0 0 9$ </td><td> $3 4 . 1 \pm 7 . 6$ </td><td> $0 . 3 2 3 \pm 0 . 0 7 9$ </td></tr><tr><td> $\mathbf { T e l e m e t r y } + \mathbf { E E G } / \mathbf { P P G } \left( \mathbf { m . - r . } \right)$ </td><td> ${ \bf 7 6 . 0 \pm 0 . 8 }$ </td><td> $\mathbf { 0 . 7 5 8 \pm 0 . 0 0 8 }$ </td><td> ${ \bf 4 9 . 3 \pm 5 . 1 }$ </td><td> $\mathbf { 0 . 4 8 5 \pm 0 . 0 4 8 }$ </td></tr><tr><td>Telemetry + video</td><td> $9 1 . 5 \pm 2 . 3$ </td><td> $0 . 9 1 5 \pm 0 . 0 2 3$ </td><td> $4 6 . 5 \pm 4 . 9$ </td><td> $0 . 4 4 4 \pm 0 . 0 4 1$ </td></tr><tr><td>EEG/PPG + video</td><td> $9 7 . 9 \pm 0 . 7 $ </td><td> $0 . 9 7 9 \pm 0 . 0 0 8$ </td><td> $3 4 . 7 \pm 6 . 2$ </td><td> $0 . 3 1 2 \pm 0 . 0 6 0$ </td></tr><tr><td> $\mathrm { T e l e m e t r y } + \mathrm { E E G } / \mathrm { P P G } + \mathrm { v i d e o }$ </td><td> $9 7 . 7 \pm 0 . 4$ </td><td> $0 . 9 7 7 \pm 0 . 0 0 4$ </td><td> $4 2 . 4 \pm 1 1 . 1$ </td><td> $0 . 3 9 2 \pm 0 . 1 1 5$ </td></tr><tr><td>Telemetry + EEG/PPG (m.-r.) + video</td><td> ${ \bf 9 4 . 0 \pm 1 . 9 }$ </td><td> $\mathbf { 0 . 9 3 9 \pm 0 . 0 2 0 }$ </td><td> ${ \bf 5 0 . 5 \pm 5 . 0 }$ </td><td> $\mathbf { 0 . 4 8 9 \pm 0 . 0 4 6 }$ </td></tr><tr><td>Telemetry + scenario + EEG/PPG + video</td><td> $9 7 . 7 \pm 0 . 4$ </td><td> $0 . 9 7 8 \pm 0 . 0 0 4$ </td><td> $4 8 . 5 \pm 9 . 4$ </td><td> $0 . 4 4 1 \pm 0 . 0 9 2$ </td></tr></table>

Table 4: Per-class results of telemetry + EEG/PPG (m.-r.) + video on unseen drivers (5 folds pooled; 8,047 windows).
<table><tr><td>Class</td><td>Precision</td><td>Recall</td><td>F1</td><td>Windows</td></tr><tr><td>Alerted</td><td>0.58</td><td>0.58</td><td>0.58</td><td>1,277</td></tr><tr><td>Distracted</td><td>0.54</td><td>0.50</td><td>0.52</td><td>1,361</td></tr><tr><td>Drowsy</td><td>0.67</td><td>0.74</td><td>0.70</td><td>1,451</td></tr><tr><td>Normal</td><td>0.34</td><td>0.32</td><td>0.33</td><td>1,335</td></tr><tr><td>Relaxed</td><td>0.33</td><td>0.25</td><td>0.28</td><td>1,360</td></tr><tr><td>Stressed</td><td>0.51</td><td>0.67</td><td>0.58</td><td>1,263</td></tr></table>

Implications: For evaluation on MultiMental-IAS, we recommend reporting both protocols, reporting scenario-context channels separately, and normalizing physiological inputs per window. The gap between within- and cross-subject scores (0.758 vs. 0.485 for the main model) quantifies the inter-driver heterogeneity that personalization and federated learning [12, 14] aim to close; with one client per participant, temporal misalignment and client latency become additional factors [25].

## 6 Limitations and Future Work

Fusion improves recognition for unseen drivers: Without scenario channels and recording ofsets, fusing driving dynamics with meanremoved EEG/PPG raises cross-subject macro-F1 from 0.408 (telemetry alone) to 0.485, and is better in all five folds (mean gain 7.7 points); within subject, it exceeds the best single input (m.-r. EEG, 0.601) by 15.7 points. Adding video brings no further cross-subject gain (0.489; better in two of five folds). Fusion with non-normalized physiology reaches the highest within-subject scores (0.916) but transfers less well (0.323), which underlines the value ofper-window normalization for cross-driver models. A learned fusion head outperforms probability averaging (late fusion: 0.447 cross-subject for telemetry + m.-r. EEG/PPG).

The MultiMental-IAS recordings were made in a driving simulator with a consumer-grade four-channel EEG headband, a setup that favours analyses of slow physiological dynamics; the per-trial metadata documents the efective sampling rates. The benchmark uses compact models, including low-resolution video clips, so that it can be reproduced on standard hardware; face-cropped, higherresolution video models are a natural extension. Future work includes extending the cohort, full-rate physiological recording, leaveone-subject-out and federated evaluation, and on-road data collection.

Per-class behaviour: Table 4 gives per-class results of the main model on unseen drivers, pooled over folds. Drowsy is recognized best (F1 0.70), followed by Alerted and Stressed (0.58). The lowarousal, low-demand conditions Normal and Relaxed are the most similar pair (0.33 and 0.28) and are mainly confused with each other, which reflects their close physiological and behavioural profiles.

## 7 Conclusion

We presented MultiMental-IAS, a four-modality driver mental-state dataset with telemetry, EEG, PPG, and in-cabin video across six induced states from 23 participants, released with corrected labels, original timestamps, and per-trial metadata. The benchmark shows that multimodal fusion reaches a macro-F1 of 0.94 for seen drivers and that fusing driving dynamics with normalized physiology consistently improves recognition for unseen drivers. The remaining gap between seen and unseen drivers makes MultiMental-IAS a concrete testbed for personalized and federated driver monitoring.

## Data and Code Availability

MultiMental-IAS is archived in DaRUS, the research data repository of the University of Stuttgart, under doi:10.18419/DARUS-XXXX. Raw logs and video are available under a data-use agreement (DUA); metadata, documentation, and the access procedure are public on the DaRUS landing page. The release includes the conversion script (fix\_dataset.py), the benchmark script (train\_benchmark.py) with the exact configuration used here, the per-run results, and the acquisition software.

## Ethics and Responsible Use

MultiMental-IAS contains EEG, PPG, and cabin-facing video, which constitute special categories of personal data under the GDPR. Released files contain no direct identifiers; participants are identified only by pseudonymous IDs, and the key linking IDs to identities is not part of the release. Access to raw physiological logs and video is gated through a DUA that prohibits re-identification attempts, unapproved biometric profiling, and use of derived models in non-consented surveillance, including workplace monitoring and insurance scoring. We encourage users to report participantwise performance before drawing conclusions about deployment readiness.

## References

[1] Yehya Abouelnaga, Hesham Eraqi, and Mohamed Moustafa. 2017. Real-Time Distracted Driver Posture Classification. In ArXiv preprint arXiv:1706.09498.

[2] Tadas Baltrusaitis, Chaitanya Ahuja, and Louis-Philippe Morency. 2019. Multi modal Machine Learning: A Survey and Taxonomy. IEEE Trans. Pattern Anal. Mach. Intell. 41, 2 (Feb. 2019), 423–443. doi:10.1109/TPAMI.2018.2798607

[3] Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. 2017. CARLA: An Open Urban Driving Simulator. In Proceedings of the 1st Annual Conference on Robot Learning (Proceedings of Machine Learning Research, Vol. 78), Sergey Levine, Vincent Vanhoucke, and Ken Goldberg (Eds.). PMLR, 1–16. https://proceedings.mlr.press/v78/dosovitskiy17a.html

[4] Stephan Getzmann, Julian E Reiser, Melanie Karthaus, Georg Rudinger, and Edmund Wascher. 2021. Measuring correlates of mental workload during simulated driving using ceegrid electrodes: a test–retest reliability analysis. Frontiers in neuroergonomics 2 (2021), 729197. doi:10.3389/fnrgo.2021.729197

[5] Baran Can Gül, Neeharika Devarakonda, Nasser Jazdi, and Michael Weyrich. 2024. Personalized Comfort Features in Software-defined Vehicles Using Federated Learning. In 2024 IEEE 29th International Conference on Emerging Technologies and Factory Automation (ETFA). 1–7. doi:10.1109/ETFA61755.2024.10710704

[6] Baran Can Gül, Daniel Dittler, Nasser Jazdi, and Michael Weyrich. 2024. Federated Learning for Comfort Features in Vehicles with Collaborative Sensing: A Review. In 2024 IEEE 29th International Conference on Emerging Technologies and Factory Automation (ETFA). 1–7. doi:10.1109/ETFA61755.2024.10710998

[7] Baran Can Gül, Suraksha Nadig, Stefanos Tziampazis, Nasser Jazdi, and Michael Weyrich. 2025. FedMultiEmo: Real-Time Emotion Recognition via Multimodal Federated Learning. In 2025 5th International Conference on Electrical, Computer, Communications and Mechatronics Engineering (ICECCME). 1–8. doi:10.1109/ ICECCME64568.2025.11277502

[8] Neska El Haouij, Jean-Michel Poggi, Sylvie Sevestre-Ghalila, Raja Ghozi, and Mériem Jaïdane. 2018. AfectiveROAD system and database to assess driver’s attention. In Proceedings ofthe 33rd Annual ACM Symposium on Applied Computing (Pau, France) (SAC ’18). Association for Computing Machinery, New York, NY, USA, 800–803. doi:10.1145/3167132.3167395

[9] Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and Ananda Theertha Suresh. 2020. SCAFFOLD: Stochastic Controlled Averaging for Federated Learning. In Proceedings of the 37th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 119), Hal Daumé III and Aarti Singh (Eds.). PMLR, 5132–5143. https://proceedings.mlr.press/v119/karimireddy20a.html

[10] Sander Koelstra, Christian Muhl, Mohammad Soleymani, Jong-Seok Lee, Ashkan Yazdani, Touradj Ebrahimi, Thierry Pun, Anton Nijholt, and Ioannis Patras. 2012. DEAP: A Database for Emotion Analysis Using Physiological Signals. IEEE Transactions on Afective Computing 3, 1 (2012), 18–31. doi:10.1109/T-AFFC.2011.15

[11] Paul Lehrer, Karenjot Kaur, Agratta Sharma, Khushbu Shah, Robert Huseby, Jay Bhavsar, Phillip Sgobba, and Yingting Zhang. 2020. Heart Rate Variability Biofeedback Improves Emotional and Physical Health and Performance: A Sys tematic Review and Meta Analysis. Appl Psychophysiol Biofeedback 45, 3 (Sept. 2020), 109–129.

[12] Tian Li, Anit Kumar Sahu, Ameet Talwalkar, and Virginia Smith. 2020. Federated Learning: Challenges, Methods, and Future Directions. IEEE Signal Processing Magazine 37, 3 (2020), 50–60. doi:10.1109/MSP.2020.2975749

[13] Quentin Massoz, Thomas Langohr, Clémentine François, and Jacques G. Verly. 2016. The ULg multimodality drowsiness database (called DROZY) and examples ofuse. In 2016 IEEE Winter Conference on Applications ofComputer Vision (WACV). 1–7. doi:10.1109/WACV.2016.7477715

[14] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. 2017. Communication-eficient learning of deep networks from decentralized data. In Artificial intelligence and statistics. Pmlr, 1273–1282.

[15] Dmitry Mikhaylov, Muhammad Saeed, Mohamed Husain Alhosani, and Yasser F. Al Wahedi. 2024. Comparison of EEG Signal Spectral Characteristics Obtained with Consumer- and Research-Grade Devices. Sensors 24, 24 (2024). doi:10.3390/ s24248108

[16] Lorena Orosco, María Agustina Garcés, Gabriel Eduardo Cañadas Fragapane, Carlos Dell’Aquila, Juan Carlos Iturrieta Gimeno, and Eric Laciar Leber. 2023. Drivers Drowsiness Database: A Collection of Physiological Signals during the Use of a Driving Simulator (DD-Database). Dataset, Dryad. doi:10.5061/dryad. 5tb2rbp9c

[17] Maria Rivas-Vidal, Alberto Calvo Cordoba, Cecilia E. García Cena, and Fernando Daniel Farfán. 2026. Combining EEG and eye-tracking for cognitive and physiological states monitoring: a systematic review. Frontiers in Neuroergonomics 6 (2026). doi:10.3389/fnrgo.2025.1736672

[18] Simanto Saha and Mathias Baumert. 2020. Intra- and Inter-subject Variability in EEG-Based Sensorimotor Brain Computer Interface: A Review. Frontiers in Computational Neuroscience 13 (2020), 87. doi:10.3389/fncom.2019.00087

[19] Gulbadan Sikander and Shahzad Anwar. 2019. Driver Fatigue Detection Systems: A Review. IEEE Transactions on Intelligent Transportation Systems 20, 6 (2019), 2339–2352. doi:10.1109/TITS.2018.286849

[20] Mohammad Soleymani, Jeroen Lichtenauer, Thierry Pun, and Maja Pantic. 2012. A Multimodal Database for Afect Recognition and Implicit Tagging. IEEE Transactions on Afective Computing 3, 1 (2012), 42–55. doi:10.1109/T-AFFC.2011. 25

[21] Johannes Stümpfle, Johannes Sigel, Matthias Weiß, Baran Can Gül, Falk Dettinger, Nasser Jazdi, Max Hoßfeld, and Michael Weyrich. 2025. The Software-Defined Vehicle: A Comprehensive Study on Current Trends and Challenges. IEEE Engineering Management Review (2025), 1–15. doi:10.1109/EMR.2025.3636574

[22] Salah Taamneh, Panagiotis Tsiamyrtzis, Malcolm Dcosta, Pradeep Buddharaju, Ashik Khatri, Michael Manser, Thomas Ferris, Robert Wunderlich, and Ioannis Pavlidis. 2017. A Multimodal Dataset for Various Forms of Distracted Driving. Scientific Data 4 (2017), 170110. doi:10.1038/sdata.2017.110

[23] Xiaoming Tao, Dingcheng Gao, Wenqi Zhang, Tianqi Liu, Bing Du, Shanghang Zhang, and Yanjun Qin. 2024. A multimodal physiological dataset for driving behaviour analysis. Sci Data 11, 1 (April 2024), 378.

[24] Julian F Thayer and Esther Sternberg. 2006. Beyond heart rate variability: vagal regulation of allostatic systems. Ann N Y Acad Sci 1088 (Nov. 2006), 361–372.

[25] Stefanos Tziampazis, Baran Can Gül, Nasser Jazdi, and Michael Weyrich. 2025. OracleFed: Latency-Aware Federated Learning via Dynamic Recovery and Causal Aggregation. IEEE Access 13 (2025), 188064–188083. doi:10.1109/ACCESS.2025. 3627490

[26] Ruoyu Wang, Wenqian Wang, Jianjun Gao, Dan Lin, Kim-Hui Yap, and Bingbing Li. 2024. MultiFuser: Multimodal Fusion Transformer for Enhanced Driver Action Recognition. In 2024 IEEE 26th International Workshop on Multimedia Signal Processing (MMSP). 1–6. doi:10.1109/MMSP61759.2024.10743612

[27] Edmund Wascher, Emad Alyan, Melanie Karthaus, Stephan Getzmann, Stefan Arnau, and Julian Elias Reiser. 2023. Tracking drivers’ minds: Continuous evaluation of mental load and cognitive processing in a realistic driving simulator scenario by means of the EEG. Heliyon 9, 7 (2023), e17904. doi:10.1016/j.heliyon.2023.e17904

[28] Limin Zhang and Hong Cui. 2022. Reliability of MUSE 2 and Tobii Pro Nano at capturing mobile application users’ real-time cognitive workload changes. Frontiers in Neuroscience 16 (2022), 1011475.

[29] Wei-Long Zheng and Bao-Liang Lu. 2017. A multimodal approach to estimating vigilance using EEG and forehead EOG. J Neural Eng 14, 2 (Jan. 2017), 026017.