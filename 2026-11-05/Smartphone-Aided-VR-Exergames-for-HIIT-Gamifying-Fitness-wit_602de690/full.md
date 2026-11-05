# Smartphone-Aided VR Exergames for HIIT: Gamifying Fitness with Real-Time Pose Estimation

Abhiram Kothagundu<sup>1</sup>, Pheelip Pritipal Raipure<sup>1</sup>, Abhijit Chandrakant Patil<sup>1</sup>, Himangshu Sarma<sup>1</sup>, Christian Eichhorn<sup>2</sup>, David A. Plecher<sup>2</sup>, and Johanna Pirker<sup>2</sup>

<sup>1</sup> Human Computer Interaction Lab, Indian Institute of Information Technology Sri City, Andhra Pradesh 517646, India

{abhiram.k22, pheelippritipal.r23, abhijitchandrakant.p23, himangshu.sarma}@iiits.in

2 Technical University of Munich, Munich, Germany {christian.eichhorn, plecher, johanna.pirker}@tum.de

Abstract. This paper introduces an innovative virtual reality (VR) exergame system designed to gamify High-Intensity Interval Training (HIIT) workouts through 11 predefined combo moves including squatpunch combinations, bilateral arm strikes, and diagonal stretch combos. Unlike conventional tracking methods that often rely on wearable motion capture suits or limited headset-based sensors, our system uses a standard smartphone camera placed on a tripod to capture full-body movement. This approach ofers a cost-efective and scalable alternative, particularly well-suited to the fast, dynamic motions typical of HIIT routines. The smartphone footage is processed in real time using a pose estimation algorithm, with extracted biomechanical data analyzed by a backend server built on the Model Context Protocol (MCP), Anthropic’s open standard for connecting AI models to structured data sources and tools, and a GPT-4o large language model. The system generates personalized assessments of form accuracy, movement eficiency, exercise intensity, and potential injury risks, tailored specifically to high-impact, time-sensitive exercises. A within-subject study involving 20 participants (ages 18–25) compared experiences in VR and Non-VR HIIT sessions. Results indicated significantly higher enjoyment (Mean = 3.5 vs 2.8, $p = 0 . 0 2 )$ , improved immersion/focus (Mean = 3.2 vs 2.8, p = 0.03), and increased likelihood to recommend (Mean = 3.2 vs 2.7, p = 0.04) under the VR condition. Confidence to continue did not difer significantly between conditions $( p = 0 . 1 5 )$ , while perceived intensity remained comparable. The MCP system showed strong agreement with expert evaluation (87.5%, N = 10 sessions, one certified trainer), supporting its reliability as a feedback tool. Building on prior work in mobile pose estimation, VR exergaming, and LLM-based coaching, this system is the first to unify all three into a single proof-of-concept home-fitness platform. The smartphone-based approach provides consistent, accessible, wearable-free motion monitoring.

Keywords: VR exergame · HIIT · Pose estimation · Gamification · MediaPipe BlazePose · MCP · Mixed Reality

## 1 Introduction

The convergence of virtual reality (VR) and immersive digital technologies has redefined traditional fitness paradigms by introducing interactive, gamified workout experiences. One of the most promising applications lies in the domain of VR-based exergames, which combine physical activity with virtual environments to enhance user motivation, engagement, and adherence to fitness routines. These systems are particularly efective in supporting high-intensity interval training (HIIT), a time-eficient training regimen known for its efectiveness in improving cardiovascular health, metabolic function, and muscular endurance through short bursts of intense activity followed by rest periods [3]. However, implementing HIIT at home remains challenging due to dificulties in maintaining motivation, ensuring proper form, and managing intensity without external supervision or professional feedback.

While existing VR fitness platforms have attempted to address some of these challenges using built-in tracking mechanisms on head-mounted displays (HMDs) like the Meta Quest or HTC Vive, they fall short in tracking the complex full-body dynamics required for HIIT. These HMD-based systems typically capture only head and hand movements, which are insuficient for exercises involving rapid, multi-joint, whole-body actions such as air squats, jumping jacks, and lunges. More advanced tracking technologies like marker-based motion capture or depth sensors ofer better precision but are costly and impractical for home use.

To address these limitations, this paper presents a novel VR exergaming system that integrates smartphone-based pose estimation, server-based biomechanical analytics, and gamified virtual environments. The system uses a tripodmounted smartphone to capture the user’s movements during workouts. The video stream is processed in real time using MediaPipe BlazePose [1,13], extracting key joint coordinates and spatial data to evaluate motion quality. Research has demonstrated that smartphone cameras, when combined with these algorithms, produce pose detection results with suficient accuracy for physiotherapy and fitness applications [10].

This pose information is then transmitted to a backend performance analysis pipeline built on MCP combined with a Large Language Model (LLM) for biomechanical feedback generation. The system interprets the incoming pose data to generate comprehensive, user-friendly performance feedback, evaluating metrics such as movement eficiency, repetition consistency, intensity levels, and deviations from correct form. Gamification plays a pivotal role in this system’s design, turning traditionally repetitive exercises into engaging, narrative-driven challenges [23,19].

This paper presents a completed implementation and user study of a scalable, low-cost VR-based exergaming system that combines smartphone-powered pose estimation, server-based MCP<sup>3</sup> movement analytics, and an immersive Meta Quest 3 dojo environment. We report full results from a within-subjects study $( N = 2 0 )$ comparing the VR system against an instructor-led non-VR HIIT control, including inferential statistics, efect sizes, and automated coaching accuracy validated against a certified expert. The system addresses core limitations in motion tracking and user feedback, demonstrating that home-based HIIT can be made more efective, engaging, and safe without specialist hardware.

## 2 Literature Review

We survey prior work across five areas directly relevant to the proposed system: pose estimation, vision-based exercise monitoring, VR exergaming, gamification, and AI-based coaching.

## 2.1 Pose Estimation for Human Motion Analysis

Bazarevsky et al. introduced BlazePose [1], a lightweight convolutional architecture producing 33-landmark full-body estimates at over 30 FPS on commodity smartphones, serving as the direct technical predecessor to MediaPipe Pose and the backbone of our system. Earlier work by Cao et al. [5] demonstrated that a single RGB camera sufices for real-time multi-person keypoint extraction; surveys across 260+ methods [30] identify mobile on-device inference as the democratization frontier. Validation studies [8] confirmed that BlazePose keypoints support sport-specific biomechanical quantification beyond laboratory conditions.

## 2.2 Vision-Based Exercise and Biomechanics Monitoring

Translating pose estimation into exercise guidance spans geometric heuristics, supervised classifiers, and self-supervised representations. Chen and Yang’s Pose Trainer [7] established rule-based form assessment from OpenPose joint angles without specialist hardware. Chae et al. [6] conducted an RCT validating smartphone-based squat coaching with significant form improvement over a video-only control, making it the closest prior work to our coaching objective absent VR gamification. Riccio [17] demonstrated real-time per-exercise classification and repetition counting from standard camera footage, employing a BiLSTM model over joint angles and raw coordinates to achieve over 99% test accuracy across four common exercises.

At the mobile form-factor, Cai [4] demonstrated that smartphone-captured images sufice for deep classification of squat correctness into multiple error categories. Supanich et al. [24] confirmed these findings in exercise monitoring contexts using single-camera MediaPipe-based pose detection for real-time posture correction, while Parmar et al. [16] proposed self-supervised workout form representations incorporating biomechanical domain constraints.

## 2.3 Virtual Reality Exergaming Systems

A growing body of controlled evidence validates VR as an efective medium for inducing HIIT while improving psychological engagement outcomes. She et al. designed HIITCopter [21], a rowing-machine VR exergame with adaptive dificulty that successfully induced HIIT-level cardiovascular exertion while yielding significantly higher motivation scores than instruction-only conditions. Xu et al. demonstrated an analogous approach with VirusBoxing [29], a VR boxing game structured around HIIT interval mechanics through timed action windows, a design pattern closely related to our reaction-based challenge architecture. Shah et al. [20] found that collaborative VR exergaming significantly elevates motivation and physical exertion compared to individual play, demonstrating sustained engagement across a five-week elderly rehabilitation study.

Comparative trials by Rubio-Arias et al. [18] and Merola et al. [15] consistently find VR exercise physiologically equivalent to traditional training while producing significantly higher enjoyment and lower perceived efort; Merola et al. replicated this pattern specifically in a medical student cohort. Prior reviews of exergaming systems identify personalization, adaptive dificulty, and integration of physiological feedback as important directions for future development [22].

## 2.4 Gamification and Motivation in Digital Fitness

Self-Determination Theory (SDT) [25,9], synthesized across 66 motivation studies, identifies competence, autonomy, and relatedness as reliable predictors of long-term exercise adherence. Mazeas et al. [14] found a significant gamification efect on physical activity (Hedges’ $g \ : = \ : 0 . 4 4 )$ , with goal-setting and progress feedback yielding the strongest contributions. Tu et al. [26] found that social gamification (relatedness-enhancing features) sustains physical activity more effectively than enjoyment-focused mechanics, a finding that motivates our realtime form scoring and social-feedback architecture.

## 2.5 AI-Based Exercise Coaching and Feedback Systems

Liao et al. [12] established a deep spatio-temporal framework for automated quality scoring of rehabilitation exercises from skeletal movement sequences; Bhadane et al. [2] systematically reviewed movement tracking systems for realtime gym exercise monitoring, confirming the viability of deep-learning-based corrective feedback pipelines. Prior research on conversational health coaching systems highlights personalization, trustworthiness, and user engagement as major design challenges for AI-driven coaching platforms [11], also what our MCP architecture targets by grounding LLM generation in session-level joint kinematic measurements.

No existing system simultaneously combines smartphone-based full-body pose estimation, server-based biomechanical analysis via a structured AI middleware architecture, LLM-generated natural language post-workout coaching, and a reaction-based VR HIIT exergame on a standalone headset. The present work directly addresses this intersection, requiring only a single commodity smartphone, unlike multi-camera platforms such as OpenCap [27] that require two devices for full 3D kinematics.

## 3 System Design and Gameplay Architecture

This section outlines the design and structure of the interactive exergame system, which combines reactive movement challenges, real-time pose validation, and immersive audiovisual feedback to deliver a high-intensity, engaging workout experience. The core gameplay revolves around performing predefined combo exercises in response to incoming orbs, detected and validated using smartphonebased pose estimation. Orb speed and combo sequences evolve over time, introducing progressive dificulty to maintain user engagement and physical exertion.

## 3.1 Game Concept

Users enter a 3D virtual dojo, a stylized training space combining calm Zen aesthetics with intense physical activity, rendered on a Meta Quest 3 standalone VR headset. The Meta Quest 3 is required hardware; it renders the immersive environment and its controllers provide haptic feedback. Within this environment, orbs continuously fly toward the player from various directions, posing challenges that require immediate physical counteractions via predefined combo moves (e.g., punches, squats, jumps). Crucially, full-body tracking is not handled by the headset: a standard smartphone on a tripod, positioned 2.5 m in front of the player, captures all lower-body and whole-body movements that the HMD alone cannot observe.

Unlike traditional rhythm-based games, this system is reaction-driven, focusing on spatial cues and unpredictable orb trajectories. Players must rely on timing, reflexes, and form rather than synchronized beats. This unpredictability enhances cognitive engagement while improving physical responsiveness and agility.

A companion smartphone application (implemented in React Native with Expo) serves a dual role in the system. As a sensor, it continuously streams compressed video frames to the backend server at 5 fps via WebSocket, enabling realtime full-body pose detection through MediaPipe. As a user-facing product, it manages player profiles, guides the T-pose calibration process, displays body alignment feedback during sessions, and presents the post-session AI coaching report with an interactive natural-language chat interface, all accessible without removing the VR headset.

## 3.2 Combo System and Pose-Driven Gameplay

Player interaction is governed by a combo system, a series of predefined exercise responses triggered by orb cues. Each combo is recognized by a classifier that uses real-time pose estimation via smartphone-based MediaPipe tracking.

![](images/06d1a0ed7d3d3ac48c546de13e3983f47c247fd88f85e0076b33226276ec53bf.jpg)  
Fig. 1. Orb grid layout and combo mapping. Green orbs represent active targets in each combo.

The smartphone, mounted on a tripod, captures the user’s full-body motion, enabling consistent and afordable tracking in home environments.

## Example Combo Mechanic

• Combo 1 – Lateral Punch: The user drives both arms simultaneously to shoulder-height lateral positions, striking two horizontal orbs at top-left and top-right, training shoulder mobility, explosive power, and bilateral coordi nation.

The remaining 10 combos follow analogous logic across horizontal, diagonal, and vertical axes (Table 1), randomized in order and directional origin to ensure the experience remains physically diverse and cognitively stimulating.

## 3.3 Orb Grid Layout and Combo Mapping

The game environment is structured as a 3×2 spatial grid, forming six distinct orb positions arranged in a rectangular plane (Fig. 1). The six orb positions are:

• Orb\_1: Top-left

• Orb\_4: Middle-right

• Orb\_2: Top-right

• Orb\_5: Bottom-left

• Orb\_3: Middle-left

• Orb\_6: Bottom-right

Based on this layout, 11 unique combos are visually and spatially mapped as shown in Table 1. This layout enforces multiplanar movement (sagittal, frontal, and transverse) and enhances proprioceptive awareness, coordination, and dynamic stability.

Table 1. Orb Position Mapping and Movement Direction
<table><tr><td rowspan=1 colspan=1>Combo</td><td rowspan=1 colspan=1>Orbs</td><td rowspan=1 colspan=1>Direction</td><td rowspan=1 colspan=1>Movement Type</td></tr><tr><td rowspan=1 colspan=1>Combo_1</td><td rowspan=1 colspan=1>Orb_1, Orb_2</td><td rowspan=1 colspan=1>Horizontal</td><td rowspan=11 colspan=1>Lateral punch or side jabTorso twist or mid strikeSide squat-punchUpper diagonal hand stretchUpper diagonal hand stretch[Lower diagonal stretch with squats[Lower diagonal stretch with squatsLeft upper + mid punchRight upper + mid punchLeft lower + mid punchRight lower + mid punch</td></tr><tr><td rowspan=1 colspan=1>Combo_2</td><td rowspan=1 colspan=1>Orb_3, Orb_4</td><td rowspan=1 colspan=1>Horizontal</td></tr><tr><td rowspan=1 colspan=1>Combo_3</td><td rowspan=1 colspan=1>Orb_5, Orb_6</td><td rowspan=1 colspan=1>Horizontal</td></tr><tr><td rowspan=1 colspan=1>Combo_4</td><td rowspan=1 colspan=1>Orb_1, Orb_4</td><td rowspan=1 colspan=1>Diagonal</td></tr><tr><td rowspan=1 colspan=1>Combo_5</td><td rowspan=1 colspan=1>Orb_3, Orb_2</td><td rowspan=1 colspan=1>Diagonal</td></tr><tr><td rowspan=1 colspan=1>Combo_6</td><td rowspan=1 colspan=1>Orb_3, Orb_6</td><td rowspan=1 colspan=1>Diagonal</td></tr><tr><td rowspan=1 colspan=1>Combo_7</td><td rowspan=1 colspan=1>Orb_5, Orb_4</td><td rowspan=1 colspan=1>Diagonal</td><td rowspan=1 colspan=1>Lower diagona</td></tr><tr><td rowspan=1 colspan=1>Combo_8</td><td rowspan=1 colspan=1>Orb_1, Orb_3</td><td rowspan=1 colspan=1>Vertical</td></tr><tr><td rowspan=1 colspan=1>Combo_9</td><td rowspan=1 colspan=1>Orb_4, Orb_2</td><td rowspan=1 colspan=1>Vertical</td></tr><tr><td rowspan=1 colspan=1>Combo_10</td><td rowspan=1 colspan=1>Orb_5, Orb_3</td><td rowspan=1 colspan=1>Vertical</td></tr><tr><td rowspan=1 colspan=1>Combo_11</td><td rowspan=1 colspan=1>Orb_6, Orb_4</td><td rowspan=1 colspan=1>Vertical</td></tr></table>

## 3.4 Combo Design and Motion Classification

The core gameplay consists of 11 distinct combos (predefined multi-joint movement patterns triggered by orb cues, each corresponding to one HIIT exercise repetition), classified using a real-time motion detection system powered by MediaPipe BlazePose. The user’s smartphone, placed on a fixed tripod, serves as the camera input device. Captured pose landmarks are transmitted to a backend MCP server implemented using Anthropic’s oficial MCP Python SDK, where classification occurs in real time via the stdio transport.

The MCP server computes a 13×13 normalized pairwise distance matrix from each frame’s 13 key joints (nose, left/right shoulder, elbow, wrist, hip, knee, and ankle), selected to cover both upper-body and lower-body combos while excluding face and finger landmarks prone to occlusion. This matrix is scored via cosine similarity against precomputed reference matrices for all 11 combos, using a threshold empirically set during development; see Section 5 for expert-agreement evidence of its efectiveness. A sliding temporal window of 2.5 s enforces consistency before confirming a classification. This training-free, server-side approach eliminates the need for a labelled dataset or GPU inference and reduces processing burden on mobile devices. Upon validation, a composite confidence score is stored for scoring, performance tracking, and post-session coaching report generation.

Ofloading classification to a remote server introduces potential latency challenges. During testing, typical latency remained under 300 ms and did not disrupt the experience. To mitigate rare network instability, the system bufers pose frames locally and employs fallback logic to momentarily rely on recent classifications.

## Combos target various fitness domains

• Explosive Power: Fast bilateral punches, dynamic squat-strikes.

• Strength: Squat-punch combinations and diagonal lower-body stretches.

• Agility & Reaction: Cross-body diagonal strikes and timed directional responses.

• Balance & Coordination: Dynamic combo transitions and multiplanar lateral movement.

## 3.5 Immersion and Engagement Mechanics

Several carefully designed mechanics ensure the experience remains physically challenging while psychologically engaging:

• Minimalist Dojo Environment: The 3D space is visually clean and serene, helping users enter a state of psychological flow while maintaining focus on movement tasks.

• Continuous Orb Streams: An endless stream of directional orbs maintains elevated cardiovascular activity and enforces reaction-based decision-making throughout the session.

• High-Energy Background Music: Dynamic audio tracks support user momentum and contribute to rhythmic immersion without requiring exact synchronization to movement.

• Haptic Feedback: Successful orb interceptions trigger tactile feedback via controller vibration, simulating physical contact and enhancing realism and reward.

## 3.6 Interaction Pipeline and Feedback

The system follows a three-phase pipeline illustrated in Fig. 2. During Setup, the headset briefly enters Mixed Reality (MR) passthrough mode, superimposing the live smartphone camera feed within the virtual environment to guide users through T-pose, squat, and arm-extension registrations; once validated, the passthrough fades and the remainder of the session is a fully immersive VR experience (Fig. 3). The Gameplay phase then runs for 5–20 minutes: orbs launch toward the player, the matching combo is performed, and the server classifies the pose via cosine similarity and timestamps the form score for that hit, repeating until the session timer expires (Fig. 4).

At session end, the Feedback phase presents a performance dashboard (combo accuracy, reaction time, form quality score, intensity profile) and a Personalized Session Report powered by $\mathbf { G P T - 4 o ^ { 4 } }$ , with a rule-based template as a safety net. For each combo hit the server computes:

form\_score = 0.5×peak\_conf+0.3×avg\_symmetry+0.2×avg\_alignment (1)

where peak\_conf is the peak cosine-similarity confidence in the 2.5 s window, avg\_symmetry is the bilateral joint-distance ratio, and avg\_alignment is the deviation from the reference pose profile. GPT-4o receives structured per-event metrics and produces a coaching report with Summary, Strengths, Corrections, and Next Goal sections, grounded in measured session data rather than unconstrained generation (validated at 87.5% expert agreement, Section 5). Session data is persisted for longitudinal progress tracking.

![](images/4e065bdf397d390acbebc786406b383da19de56a25c4d8373a4d53a3bf42b198.jpg)  
Fig. 2. Three-phase interaction pipeline. Setup: smartphone calibration via MR passthrough overlay. Gameplay: reaction-based orb-combo loop with pose scoring per hit (repeats until session timer expires). Feedback: GPT-4o coaching report delivered to the Quest browser.

## 4 Methodology

This section describes the technical classification and reporting architecture, participant selection, experimental procedure, and statistical analysis plan.

## 4.1 MCP Classification Logic and Reporting Architecture

The system employs a dual-stream architecture. The smartphone continuously streams compressed video frames to the backend server via a persistent Web-Socket connection; MediaPipe extracts per-frame landmark matrices which are timestamped and bufered. Concurrently, the VR game client sends event-driven HTTP POST requests to the same server at each combo completion, including the event timestamp. The server fuses the two streams purely by wall-clock time, looking back 2.5 seconds in the phone’s landmark bufer to evaluate form at the moment of each orb hit. This architecture is illustrated in Fig. 5, and the MCP (Model Context Protocol) middleware orchestrates both the real-time classifier and the post-session reporting pipeline. This primary MCP server applies several analytical steps:

![](images/15262cb69106e405d6f627349168628947d64c62f02d38f73c93a9e33eafe847.jpg)

![](images/b9967b52bedd9a2e0844c590a23884cdfaff998fa6042153d56267bf859e8518.jpg)

Fig. 3. Setup Phase—player approaching the MR passthrough window in the virtual dojo to position the smartphone camera before session start.  
![](images/591552d56fda677e81bc62175e52f3d84f115d2604cb936622693afe12372444.jpg)  
Fig. 4. Gameplay Phase—player performing combos as orbs approach in the dojo space.

• Real-time Frame Ingestion and Bufering: Each incoming frame’s joint matrix, along with its precise timestamp, is ingested into a rolling bufer. When a combo-hit HTTP event arrives, the current 2.5 s window is frozen and queued for scoring; a fresh window then begins immediately, ensuring uninterrupted processing. (LLM context management is distinct and applies only to the post-session GPT-4o reporting pipeline.)

![](images/cad59115e4bede9fae3b32931768a4a87feaf0627196f5253b1757677454dd27.jpg)  
Fig. 5. Dual-stream system architecture. The smartphone streams continuous video frames via WebSocket; the Meta Quest 3 sends per-combo events via HTTP POST. The FastAPI Gateway fuses both streams by wall-clock timestamp. The Classifier MCP performs training-free pose recognition using a 13×13 distance matrix and cosine similarity; on session end, GPT-4o generates the coaching report delivered as a selfcontained HTML page to the Quest browser.

• Relative Distance Matrix Construction: For each ingested frame, the system calculates a 13×13 matrix representing the normalized spatial relationships between tracked joints, accounting for user anthropometry and perspective.

• Reference Combo Matching: Each frame’s 13×13 relative-distance matrix is compared against a stored reference matrix for the target combo using cosine similarity. No model training is required; the reference matrix is captured once from a correct demonstration and remains fixed. A classification is accepted only when cosine similarity exceeds a threshold across consecutive frames in the 2.5 s sliding window, ensuring temporal consistency.

• Temporal Consistency Checking: Movement patterns are analyzed over a sliding time window to validate classifications and assess rhythmic consistency against expected biomechanical profiles.

• Form Quality Scoring: The primary MCP server computes confidence scores reflecting performance accuracy, factoring in symmetry, alignment, and timing.

Hierarchical Report Generation. Upon session completion, a dedicated Reporting MCP instance queries and retrieves the persisted, timestamped data segments associated with the user’s session. By aggregating and analyzing this complete, chronologically ordered dataset, the Reporting MCP synthesizes a comprehensive performance report including detailed pose evaluations, trend analysis, mobility progression tracking, and personalized feedback. The Reporting MCP also enables an interactive conversational interface, allowing users to query their performance data in natural language and receive personalized recommendations.

Architectural Novelty. Three design choices distinguish this architecture: (1) training-free classification: cosine similarity on compact 13×13 distance matrices eliminates labelled datasets and GPU inference, making the classifier deployable on commodity hardware and extensible to new combos via a single reference pose capture; (2) wall-clock stream fusion: WebSocket landmark frames and HTTP game events are joined purely by timestamp, requiring no shared state between streams; and (3) MCP as a structured context bridge: rather than embedding raw landmark arrays in an LLM prompt, the gateway delivers schemavalidated, per-combo metrics to GPT-4o, producing reproducible coaching text without cloud GPU dependency.

## 4.2 Participants

A total of 20 participants (ages 18–25) participated in this within-subjects design, with each individual experiencing both VR and non-VR conditions. The participants were primarily university students from various academic backgrounds, exhibiting varying levels of physical fitness, gaming experience, and familiarity with virtual reality systems.

Each participant completed a pre-study questionnaire collecting demographic details, prior experience with HIIT and VR, motivation to exercise, and current workout habits. This screening ensured a balanced sample with representation across diferent body types, fitness levels, and exercise preferences.

Ethical Considerations This study was conducted in accordance with the Declaration of Helsinki. All participants provided written informed consent prior to participation, covering the nature of the activities, use of pose-estimation video analysis, and storage of anonymized performance data. Participation was entirely voluntary and participants could withdraw at any time without consequence. No personally identifiable data were retained beyond session completion; pose landmark streams were processed server-side and only aggregate metrics were stored. The study protocol was reviewed and approved by the Institutional Ethics Committee of IIIT Sri City prior to data collection.

## 4.3 Study Procedure

Pre-Study Phase. Before participation, all subjects completed a structured questionnaire capturing fitness profile (physical activity levels, exercise frequency, prior experience with HIIT), technology literacy (experience with VR, motion tracking, and exergames), motivational baseline (adapted from the Behavioral Regulation in Exercise Questionnaire), and exercise style preferences. These responses ensured even distribution of motivation and fitness background across the two conditions.

Each participant completed two workout sessions: one in a traditional Non-VR setting and another using a VR exergame prototype. The VR environment was developed for the Meta Quest 3 headset and incorporated real-time fullbody pose tracking using MediaPipe BlazePose. Exercises such as jumping jacks, squats, and punches were presented as interactive elements (virtual orbs) in a gamified dojo setting.

VR Condition. Participants wore the Meta Quest 3 headset and interacted with dynamic workout sequences in the virtual environment. Full-body motion was captured using the external smartphone camera, and pose landmarks were detected via MediaPipe BlazePose. The phone streamed landmark data to the MCP server continuously throughout the session; the VR game notified the server at each combo completion, triggering a form-quality assessment against the bufered landmark data. Upon session completion, a personalized AI coaching report was generated and delivered to the participant via the Quest browser.

Non-VR Condition. Participants completed the same 10-minute HIIT routine guided by a live instructor demonstration. No immersive or gamified feedback was provided. A certified fitness trainer manually observed and recorded each participant’s form and performance for later comparison.

## 4.4 Post-Study Phase: Evaluation

Upon session completion, participants responded to a post-study survey comprising the following instruments:

• Engagement and Motivation: Adapted Intrinsic Motivation Inventory (IMI), measuring interest/enjoyment, perceived competence, and value/usefulness.

• Perceived Exertion: Borg RPE scale (6–20) to quantify subjective exercise intensity.

• Immersion and Presence: Presence Questionnaire (PQ) for VR users, assessing spatial presence, realism, and involvement.

• Usability and Satisfaction: System Usability Scale (SUS) and custom Likert-scale questions regarding clarity, responsiveness, and willingness to continue.

• Open-Ended Feedback: Participants commented on perceived efectiveness, discomforts, or suggestions for improvements.

In-game metrics for the VR group (combo success rate, average reaction time, and form deviation) were logged for objective cross-validation against selfreported efort and usability ratings.

## 4.5 Analysis Plan

Given the sample size, data were analyzed using:

• Descriptive statistics (mean, SD) to summarize user responses and performance metrics.

Table 2. Mean Post-Study Questionnaire Scores (Scale: 1–5)
<table><tr><td>Metric</td><td>VR Group</td><td>Non-VR Group</td></tr><tr><td>Motivation</td><td>2.8</td><td>2.6</td></tr><tr><td>Enjoyment</td><td>3.5</td><td>2.8</td></tr><tr><td>Focus/Immersion</td><td>3.2</td><td>2.8</td></tr><tr><td>Perceived Intensity</td><td>3.3</td><td>3.4</td></tr><tr><td>Confidence to Continue</td><td>2.9</td><td>2.8</td></tr><tr><td>Recommendation Likelihood</td><td>3.2</td><td>2.7</td></tr><tr><td>Gamified Engagement Score</td><td>3.0</td><td>N/A</td></tr></table>

• Paired t-tests and non-parametric tests (Mann-Whitney U) to compare VR vs. non-VR outcomes.

• Efect sizes (Cohen’s d) to estimate practical significance.

## 5 Statistical Analysis and Results

This section reports quantitative outcomes from the within-subjects study, covering descriptive statistics, inferential tests, and expert-agreement analysis for the MCP coaching pipeline.

## 5.1 Summary Statistics

Post-study feedback was collected across key user experience dimensions using a 5-point Likert scale. Table 2 summarizes the mean scores for both the VR and Non-VR groups (n = 20).

Immersion levels and average enjoyment scores were greater in the VR group, indicating that interactive environments positively influence user satisfaction. Both groups demonstrated modest motivation, but the VR group scored higher on continued use intention and recommendation likelihood, suggesting greater potential for long-term engagement.

## 5.2 Data Analysis

To evaluate exercise performance across both groups, we employed a dual scoring methodology combining expert human evaluation with automated machinebased assessment. For the Non-VR group, a certified fitness trainer manually scored participants using standardized criteria such as posture accuracy, form consistency, range of motion, and overall control. For the VR group, MCPgenerated confidence scores were compared with the expert evaluations to assess the system’s ability to replicate human-level assessment.

The average agreement score across 10 randomly selected VR-condition sessions (one certified trainer, blind to system scores) was 87.5% (using the composite form score defined in Eq. 1), indicating a high level of consistency. The system showed particularly strong alignment with human ratings for joint alignment (90.2%) and body symmetry (88.7%), while scoring slightly lower for reaction timing accuracy (81.3%), likely due to minor latency in smartphone video capture and pose processing. Expert evaluation was not extended to Non-VR sessions, as the system’s coaching reports apply only to the VR condition.

## 5.3 Inferential Statistical Tests

Paired Sample t-Tests. To compare the VR and Non-VR conditions $( N = 2 0 )$ :

• Motivation: No statistically significant diference $( p = 0 . 2 1$ , Cohen’s $d =$ 0.22 – small efect).

• Enjoyment: Significantly higher in VR $( p \ = \ 0 . 0 2 .$ , Cohen’s $d \ : = \ : 0 . 6 5 \ : -$ medium efect). Gamified VR substantially improved how enjoyable the workout felt.

• Focus/Immersion: Significantly higher in VR (p = 0.03, Cohen’s d = 0.58 – medium efect). VR provided a more immersive and engaging environment.

• Recommendation Likelihood: Significantly higher in $\mathrm { V R } \ ( p = 0 . 0 4$ , Cohen’s d = 0.48 – small-medium efect), consistent with higher enjoyment driving willingness to recommend the system.

• Confidence to Continue: No significant diference $( p = 0 . 1 5$ , Cohen’s d = 0.28 – small efect).

Mann-Whitney U Tests (Non-Parametric). Non-parametric Mann-Whitney U tests confirmed the parametric findings: Enjoyment $( U = 4 8 . 0 , p = 0 . 0 2 )$ , Focus/Immersion $( U = 4 4 . 5 , \ p = 0 . 0 1 )$ , and Recommendation Likelihood $( U =$ $5 1 . 5 , p = 0 . 0 4 )$ were significantly higher in VR; Motivation $( U = 8 0 . 5 , p = 0 . 3 2 )$ and Confidence $( U = 7 0 . 0 , p = 0 . 1 9 )$ were not significant, supporting the robustness of results given the sample size.

One-Way ANOVA – Gamification Efect. A one-way ANOVA showed that VR gamification had a significant efect on both enjoyment and recommendation likelihood: $F ( 1 , 1 9 ) = 5 . 9 2 , p < 0 . 0 5$

Pearson Correlation Coeficients.

• Enjoyment & Recommendation Likelihood: $r = 0 . 8 2 , \ p \ < \ 0 . 0 0 1$ (strong positive)

• Gamification & Motivation: $r = 0 . 6 5 , p < 0 . 0 1$ (moderate positive)

• Technical Issues & Consistency: $r = - 0 . 4 7 , p < 0 . 0 5$ (moderate negative)

## 6 Discussion

This pilot study highlights the promise of VR-based HIIT gamification in fostering greater physical and cognitive engagement. Participants using the VR dojo system reported comparable perceived exercise intensity to the non-VR condition (VR = 3.3 vs. Non-VR = 3.4), suggesting no meaningful diference in physical demand, alongside significantly increased immersion, enjoyment, and recommendation likelihood attributed to interactive feedback and virtual visuals.

The statistically significant improvements in enjoyment $( p = 0 . 0 2 , d = 0 . 6 5 )$ and focus/immersion $( p = 0 . 0 3 , d = 0 . 5 8 )$ under the VR condition align with prior controlled evidence. Rubio-Arias et al. [18] reported that VR exercise produces markedly higher enjoyment than equivalent non-VR protocols while maintaining comparable physiological load. Our reaction-based design, where orb trajectories are randomized rather than beat-synchronized, likely amplifies cognitive engagement beyond what rhythm-driven systems such as VirusBoxing [29] achieve, because participants must continuously anticipate direction and combo type. This divergence from music-synchronized paradigms may partially explain the absence of significant diferences in motivation and confidence: one possible interpretation, which remains untested in this single-session design, is that novelty efects from the VR medium dominate short-term engagement [25], while intrinsic behavioral regulation benefits require longer exposure and richer personalization.

The 87.5% agreement observed in our study is consistent with the growing body of evidence demonstrating that smartphone-based markerless motion analysis can provide practically useful biomechanical assessments outside laboratory environments [27]. The lower agreement for reaction timing (81.3%) reflects known smartphone video latency compounded by the online classification round-trip; this gap could be closed by multi-view triangulation as demonstrated in OpenCap [28].

## 7 Limitations and Future Work

Several limitations shape the current findings: self-reported questionnaire data may carry social desirability bias; first-session VR novelty likely inflates enjoyment scores; the absence of biometric wearables limits objective intensity measurement; smartphone placement and lighting afect tracking quality; the university-student cohort (ages 18–25) limits generalizability to older adults and clinical populations; expert validation was conducted on 10 of 20 sessions with one rater; and the cosine similarity classification threshold was set empirically rather than by systematic cross-validation.

These limitations motivate the following future directions: integrating biometric wearables (heart rate, caloric burn) for objective intensity feedback; adding adaptive dificulty via real-time fatigue detection from pose-confidence drops; designing a multiplayer challenge mode to foster social motivation; and conducting longitudinal studies to assess adherence and performance trends over extended use.

## 8 Conclusion

This exploratory study demonstrates that a smartphone-powered VR exergame leveraging real-time pose estimation can enhance the HIIT workout experience by significantly increasing enjoyment $( p \ : = \ : 0 . 0 2 , \ : d \ : = \ : 0 . 6 5 )$ , focus/immersion $( p = 0 . 0 3 , d = 0 . 5 8 )$ , and recommendation likelihood $( p = 0 . 0 4 , d = 0 . 4 8 )$ The proposed system, integrating a standard smartphone camera, pose estimation algorithms, and MCP-driven feedback, proved efective in providing accessible, hands-free, and personalized training insights without relying on complex hardware. Quantitative results from a within-subject study indicate statistically significant improvements in enjoyment and focus under the VR condition, with expert-aligned feedback supporting the reliability of the MCP-based analysis. While preliminary, these findings validate the core design and feasibility of our approach. Future work will focus on expanding personalization, long-term adherence studies, and the inclusion of adaptive and social features to further augment its impact in home-based fitness scenarios.

Acknowledgments. The authors gratefully acknowledge the use of AI writing assistance for improving the clarity of this manuscript. All interpretations, analyses, and conclusions are solely those of the authors.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., Grundmann, M.: BlazePose: On-device real-time body pose tracking. https://arxiv.org/abs/ 2006.10204 (2020)

2. Bhadane, S., Ganguly, S., Karanje, M., Rachaveti, D., Amutha, S., Surendiran, B.: A systematic review of movement tracking for real-time monitoring of physical exercises in the gym. WIREs Data Mining and Knowledge Discovery 15(4), e70057 (2025). https://doi.org/10.1002/widm.70057

3. Boutcher, S.H.: High-intensity intermittent exercise and fat loss. Journal of Obesity 2011, 868305 (2011). https://doi.org/10.1155/2011/868305

4. Cai, H.: Application of intelligent real-time image processing in fitness motion detection under internet of things. The Journal of Supercomputing 78, 7788–7804 (2022). https://doi.org/10.1007/s11227-021-04145-0

5. Cao, Z., Hidalgo, G., Simon, T., Wei, S.E., Sheikh, Y.: OpenPose: Realtime multiperson 2D pose estimation using part afinity fields. IEEE Transactions on Pattern Analysis and Machine Intelligence 43(1), 172–186 (2021). https://doi.org/10.1109/ TPAMI.2019.2929257

6. Chae, H.J., Kim, J.B., Park, G., O’Sullivan, D.M., Seo, J., Park, J.J.: An artificial intelligence exercise coaching mobile app: Development and randomized controlled trial to verify its efectiveness in posture correction. Interactive Journal of Medical Research 12, e37604 (2023). https://doi.org/10.2196/37604

7. Chen, S., Yang, R.: Pose trainer: Correcting exercise posture using pose estimation. https://arxiv.org/abs/2006.11718 (2020)

8. Hulleck, A.A., Menoth Mohan, D., Abdallah, N., El Rich, M., Khalaf, K.: Present and future of gait assessment in clinical practice: Towards the application of novel trends and technologies. Frontiers in Medical Technology 4, 901331 (2022). https: //doi.org/10.3389/fmedt.2022.901331

9. Johnson, D., Deterding, S., Kuhn, K.A., Staneva, A., Stoyanov, S., Hides, L.: Gamification for health and wellbeing: A systematic review of the literature. Internet Interventions 6, 89–106 (2016). https://doi.org/10.1016/j.invent.2016.10.002

10. Kanko, R.M., Laende, E.K., Davis, E.M., Selbie, W.S., Deluzio, K.J.: Concurrent assessment of gait kinematics using marker-based and markerless motion capture. Journal of Biomechanics 127, 110665 (2021). https://doi.org/10.1016/j.jbiomech. 2021.110665

11. Kocaballi, A.B., Berkovsky, S., Quiroz, J.C., Laranjo, L., Tong, H.L., Rezazadegan, D., Briatore, A., Coiera, E.: The personalization of conversational agents in health care: Systematic review. Journal of Medical Internet Research 21(11), e15360 (2019). https://doi.org/10.2196/15360

12. Liao, Y., Vakanski, A., Xian, M.: A deep learning framework for assessing physical rehabilitation exercises. IEEE Transactions on Neural Systems and Rehabilitation Engineering 28(2), 468–477 (2020). https://doi.org/10.1109/TNSRE.2020.2966249

13. Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C.L., Yong, M.G., Lee, J., Chang, W.T., Hua, W., Georg, M., Grundmann, M.: MediaPipe: A framework for building perception pipelines. https://arxiv.org/abs/1906.08172 (2019)

14. Mazeas, A., Duclos, M., Pereira, B., Chalabaev, A.: Evaluating the efectiveness of gamification on physical activity: Systematic review and meta-analysis. Journal of Medical Internet Research 24(1), e26779 (2022). https://doi.org/10.2196/26779

15. Merola, P., Cardoso, M.B., Barreto, G., Chagas, M.C., Farias Oliveira Saunders, L., Saunders, B., Cortozi Berton, D.: Virtual reality high-intensity interval training exergaming compared to traditional high-intensity circuit training among medical students: Pilot crossover study. JMIR Serious Games 13, e63461 (2025). https: //doi.org/10.2196/63461

16. Parmar, P., Morris, B.: Action quality assessment across multiple actions. In: Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV). pp. 1468–1476 (2019). https://doi.org/10.1109/WACV.2019.00161

17. Riccio, R.: Real-time fitness exercise classification and counting from video frames. https://arxiv.org/abs/2411.11548 (2024)

18. Rubio-Arias, J.Á., Verdejo-Herrero, A., Andreu-Caravaca, L., Ramos-Campo, D.J.: Impact of immersive virtual reality games or traditional physical exercise on cardiovascular and autonomic responses, enjoyment and sleep quality: A randomized crossover study. Virtual Reality 28, 64 (2024). https://doi.org/10.1007/ s10055-024-00981-6

19. Sevinçli, M.C., Aydoğmuş, M.E.: Video games as a part of intervention programs based on self-determination theory. Psikiyatride Güncel Yaklaşımlar 14(2), 207– 220 (2022). https://doi.org/10.18863/pgy.974968

20. Shah, S.H.H., Karlsen, A.S.T., Solberg, M., Hameed, I.A.: A social VR-based collaborative exergame for rehabilitation: Codesign, development and user study. Virtual Reality pp. 1–18 (2022). https://doi.org/10.1007/s10055-022-00721-8

21. She, H., Young, J., Wittenberg, O., Poole, P., Mercer, A., Shaw, A., Wünsche, B.C.: HIITCopter: Analysis of the suitability of VR exergaming for high-intensity interval training. In: Proceedings of the 32nd Australian Conference on Human-Computer Interaction. pp. 293–302 (2021). https://doi.org/10.1145/3441000.3441009

22. Skjæret, N., Nawaz, A., Morat, T., Schoene, D., Helbostad, J.L., Vereijken, B.: Exercise and rehabilitation delivered through exergames in older adults: An integrative review of technologies, safety and eficacy. International Journal of Medical Informatics 85(1), 1–16 (2016). https://doi.org/10.1016/j.ijmedinf.2015.10.008

23. Staiano, A.E., Calvert, S.L.: Exergames for physical education courses: Physical, social, and cognitive benefits. Child Development Perspectives 5(2), 93–98 (2011). https://doi.org/10.1111/j.1750-8606.2011.00162.x

24. Supanich, W., Kulkarineetham, S., Sukphokha, P., Wisarnsart, P.: Machine learning-based exercise posture recognition system using MediaPipe pose estimation framework. In: Proceedings of the 9th International Conference on Advanced Computing and Communication Systems (ICACCS). pp. 2003–2007. IEEE (2023). https://doi.org/10.1109/ICACCS57279.2023.10112726

25. Teixeira, P.J., Carraça, E.V., Markland, D., Silva, M.N., Ryan, R.M.: Exercise, physical activity, and self-determination theory: A systematic review. International Journal of Behavioral Nutrition and Physical Activity 9, 78 (2012). https://doi. org/10.1186/1479-5868-9-78

26. Tu, R., Hsieh, P., Feng, W.: Walking for fun or for “likes”? the impacts of diferent gamification orientations of fitness apps on consumers’ physical activities. Sport Management Review 22(5), 682–693 (2019). https://doi.org/10.1016/j.smr.2018. 10.005

27. Uhlrich, S.D., Falisse, A., Kidziński, Ł., Muccini, J., Ko, M., Chaudhari, A.S., Hicks, J.L., Delp, S.L.: OpenCap: Human movement dynamics from smartphone videos. PLOS Computational Biology 19(10), e1011462 (2023). https://doi.org/ 10.1371/journal.pcbi.1011462

28. Wade, L., Needham, L., McGuigan, P., Bilzon, J.: Applications and limitations of current markerless motion capture methods for clinical gait biomechanics. PeerJ 10, e12995 (2022). https://doi.org/10.7717/peerj.12995

29. Xu, W., Liang, H.N., Ma, X., Li, X.: VirusBoxing: A HIIT-based VR boxing game. In: Extended Abstracts of the 2020 Annual Symposium on Computer-Human Interaction in Play. pp. 98–102 (2020). https://doi.org/10.1145/3383668.3419958

30. Zheng, C., Wu, W., Chen, C., Yang, T., Zhu, S., Shen, J., Kehtarnavaz, N., Shah, M.: Deep learning-based human pose estimation: A survey. ACM Computing Surveys 56(1), 1–37 (2023). https://doi.org/10.1145/3603618