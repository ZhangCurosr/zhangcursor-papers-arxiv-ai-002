# Socrates went Nuclear: Comparing Interaction Strategies for AI systems in a Learning Context using Brain Sensing

Alexandre Clin Defarges aclindefarg@student.ethz.ch ETH Zürich

Nataliya Kosmyna nkosmyna@mit.edu MIT Media Lab

Pattie Maes pattie@mit.edu MIT Media Lab

## Abstract

Students are adopting LLMs at scale. This poses a question: does unrestricted AI access bypass the cognitive efort required for learning, or does it streamline knowledge acquisition? This paper reports on a study where we compare three designs for user-AI interaction in a learning context: (1) an unrestricted conversational bot like ChatGPT, (2) a pedagogically constrained bot that guides through hints without giving final answers, which we refer to as the Socratic mode; and (3) a non-conversational adaptive tutoring system that adjusts dificulty in real-time based on the user’s cognitive engagement derived from the brain signals. Fifty study participants were tasked with learning about nuclear safety protocols, a domain chosen for its zero-prior knowledge baseline. The participants progressed through an instructional video, a pre-test, an AI-driven assessment phase, which varied in the three conditions, and an immediate post-test. The nature of the questions centered primarily on factual knowledge acquisition, but it still required participants to have a global understanding of the concepts in order to answer the questions correctly. A Muse headband was used to derive the cognitive engagement of all users in all conditions.

The unrestricted chatbot produced higher learning gains (delta) than both constrained modes (� < .03, � > 0.80), while the adaptive condition generated significantly higher EEG engagement (� = .018). The cluster analysis of chatbot usage and discussion patterns by users showed that most participants in the unrestricted-mode adopted a direct answer-retrieval strategy, while participants in the Socratic-mode initially attempted to reason through the hints before progressively disengaging. Consequently, this also suggests that the success of the unrestricted AI is not an evidence of deeper learning, but rather a result of the immediate post-test evaluation after the training phase.

## CCS Concepts

• Human-centered computing → User studies; Interactive systems and tools; • Applied computing → Computer-assisted instruction.

## Keywords

Large Language Models, Socratic tutoring, Adaptive learning, EEG, Cognitive engagement, Human–Agent Interaction, Nuclear course

## 1 Introduction

The way students study and work on their own has changed drastically since 2022. Across all educational levels, they increasingly turn to general purpose LLMs like ChatGPT to work through assignments and homework. Recent survey data indicate that over 89% of students have used ChatGPT for homework assistance, 53% using it to write essays and 48% for unproctored assessments. A 2026 Pew Research Center study found that 54% of adolescents actively use AI for schoolwork [31].

What this means for learning is not clear. One perspective, based on cognitive load theory and the desirable dificulties framework [8], is that general-purpose LLMs might threaten deep learning because they are built to be helpful and resolve tasks quickly and thus may reduce the productive struggle that builds lasting understand ing [8]. Students seem to sense this themselves: many adolescents (59%) see AI-assisted academic dishonesty as systemic in their institutions, and 72% of college students support the ban on ChatGPT in universities [31].

However, a competing perspective suggests that unrestricted AI access may ofer underappreciated benefits by providing immediate, high-quality explanations on demand. This perspective posits that general purpose LLMs enable eficient knowledge acquisition that can serve as a foundation for deeper understanding. Unrestricted access also lets students explore tangential questions and adapt the LLM’s output to their own level. The empirical evidence for which perspective better predicts actual learning outcomes is very mixed [8].

In response to these concerns, two directions have emerged. The first focuses on Socratic conversational LLMs which are constrained to guide students through questions and hints rather than giving answers. Theoretically, such LLMs have the potential to support deeper cognitive processing. But challenges include potential frustration over a series of questions and longer time requirements. Additionally, without clear guidance, the Socratic chat could remain superficial [17]. Additionally, relying on LLMs too much might also hinder critical thinking [17].

Motivated by recent papers and a growing interest in personalized adaptive learning [34], the second approach focuses on adap tive, non-conversational exercise systems [22]. Here, AI analyzes the performance of the user in real-time and restructures questions based on the success of previous responses of the said user. There is also a possibility to leverage physiological measurements in order to estimate the learner’s engagement or cognitive load and adapt an output of an LLM accordingly. This adaptive approach moves beyond the open-ended dialogue of a typical chatbot.

Although prior work, including personalized feedback mechanisms explored in systems like NeuroChat [23], demonstrates that personalized interaction improves engagement outcomes, the literature lacks a direct empirical comparison between an unrestricted chatbot, a Socratic bot, and an adaptive bot in the context of homework assistance.

We ran a controlled experiment to compare these three approaches: unrestricted, Socratic and adaptive. To handle the confound of heterogeneous prior knowledge, we chose a topic of nuclear safety protocols, currently rare enough that our participants would not demonstrate any pre-existing knowledge. The experiment simulated an independent study session: participants first watched a standardized video, took a baseline pre-test, and were then randomly assigned to one of the three modes. We administered a post-test immediately following the experiment to measure direct knowledge retention. Beyond test scores, we collected neurophysiological recordings using a Muse 2 EEG headband [32] to measure cognitive engagement in all three conditions. Only the adaptive condition used the real time EEG data to modify its content; in the other two conditions, the EEG was recorded passively. This setup allowed us to evaluate both immediate knowledge retention and the use of cognitive engagement to directly influence bot output.

## 2 Related Work

The literature on AI in education is large and fragmented. We organize it around six themes that directly inform our study design: the cognitive cost of unrestricted AI use, the theoretical basis for efortful learning, Socratic and constrained architectures, adaptive tutoring systems, and neurophysiological measurement of engagement.

## 2.1 The Cognitive Cost of Unrestricted AI in Education

Several recent studies suggest that using general LLMs during learning has a measurable cognitive cost. Kosmyna, Hauptmann et al. [1] ran an EEG study with 54 participants across four essay-writing sessions and showed that students relying on ChatGPT showed weaker functional brain connectivity during the task than those using search engines or no tools. Stadler, Bannert and Sailer [3] reached a similar conclusion: in a randomized trial with 91 students learning about nanoparticles, LLM users reported lower cognitive load and produced weaker reasoning. Gerlich [2] argued that delegating mental work to AI tools could decrease critical thinking over time.

Large scale surveys documented a similar efect. A RAND report [6] stated that students themselves think AI is hurting their critical thinking. Fan et al. [4] described a "metacognitive laziness” efect: when learners overrely on AI answers, they lose track of what they know and do not know, leading to a false sense of knowledge. Rizvi et al. [5] described LLMs as a double-edged sword, and Ortiz-Bonnin and Blahopoulou [7] showed that ChatGPT usage is blurring the line between help and cheating.

## 2.2 The Need for Efort

Bjork and Bjork [9] formalized the idea of desirable dificulties: techniques that make learning harder in the short term but improve long term retention. Efort during learning strengthens memory, which is precisely what AI tools can bypass. Wells et al. [10] brought these ideas to the current use of AI and explained that productive struggle is potentially what is missing when using an LLM. They also connected this to Vygotsky’s Zone of Proximal Development (ZPD) [29]: the dificulty needs to be tailored to the learner’s level.

## 2.3 Socratic Architectures and Constrained LLM Design

Since LLMs tend to give direct answers, recent work tries to move them towards Socratic implementations, meaning that they guide students toward the answer through targeted questions. Dinucu-Jianu, Macina et al. [13] used reinforcement learning to reward "quality" teaching (fosters understanding and reasoning) instead of focusing on correct answers alone.

Several systems targeted specific subjects. Xie et al. [14] built STAP, a Socratic programming tutor with adaptive step-by-step support. Liu et al. [15] created PACE, a math tutor that simulates each student’s learning style and aligns responses with the corresponding persona. They also applied the Socratic method to deliver instant feedback and stimulate deeper reasoning. This enabled PACE to adapt to individual learner needs.

Cohn et al. [16] proposed Inquizzitor, an LLM-based agent for science assessment that scores student responses and delivers adaptive feedback in the Zone of Proximal Development (ZPD, [29]), reaching high agreement with human graders $\left( \kappa _ { w } \ge 0 . 7 0 \right)$

## 2.4 Adaptive Learning Systems and Intelligent Tutoring Systems

Vanacore et al. [20] argued that LLMs must be combined with established Intelligent Tutoring Systems (ITS, adaptive software that delivers personalized instruction and feedback [19]) methods to become real tutors rather than generic chatbots. Scarlatos et al. [21] went further by training LLM tutors to track student knowledge directly within the conversation. Work on adaptive dificulty mod els [22] proposed algorithms that adjust problem complexity in order to keep students in the optimal dificulty range.

## 2.5 Neurophysiological Measurements and Neuro-Adaptive Interfaces

Several recent works connect the learner’s cognitive state directly to AI using physiological measurements such as brain signals. Baradari, Kosmyna et al. [23] developed NeuroChat, an AI tutor that uses real-time EEG to derive cognitive engagement of the subjects in order to adjust the LLM dificulty. Wearable EEG devices [24] now make this kind of cognitive tracking possible outside the laboratory [18]. Noel et al. [25] used the same engagement index (beta/(theta + alpha)) to show how diferent teaching methods impacted student’s focus. Beauchemin et al. [26] built a system that adjusts learning speed based on cognitive load. Gkintoni et al. [11] reviewed 103 empirical studies and argued that AI systems combined with neurophysiological monitoring (such as EEG) can dynamically adjust dificulty in real time to keep learners engaged.

Putting this together, unrestricted LLMs can undercut the cognitive efort that contributes to learning. Socratic prompting and adaptive tutoring each address this problem to an extent, but have only been studied separately, and while physiological sensing like

EEG can track engagement in real time, it has rarely been used to compare these fundamentally diferent AI tutoring styles. To the best of our knowledge, no studies have been reported that compare an unrestricted chatbot, a Socratic bot, and an EEG-adaptive exercise system side by side under controlled conditions while measuring both learning and engagement in all three conditions.

## 3 Study Design

To understand how these diferent AI tools actually impact learning and student engagement, we ran a controlled study. This section explains the experimental design, the three AI modes compared, and how brain activity was measured.

## 3.1 The Three AI Interaction modes

• Mode 1 : Baseline, Unrestricted Chatbot : A standard ChatGPT-like bot integrated in a side panel to the right of the problem interface, without any pedagogical restrictions (see Figure 1a). If the student asks for the final answer, the AI gives it directly. EEG engagement is derived and recorded passively and continuously.

• Mode 2 : Socratic, Restricted Chatbot : The exact same interface as in Mode 1 (see Figure 1a), but the side bot is constrained by a Socratic tutoring system prompt inspired by the tutoring prompts of Tack and Piech [12], but rewritten for this study (the full prompt is provided in Appendix A). It never gives the final answer. Instead, it asks guiding questions and provides hints to nudge the student in the right direction. The pedagogical policy goes beyond simply hiding the answer: at every turn, the bot compares the student’s cur rent answer with the reference correction to identify what is missing or wrong, asks one guiding question at a time, consolidates progress by asking the student to restate ideas in their own words, and ends every message with a short directional hint. The theoretical rationale is that hiding the answer while providing the guidance step-by-step keeps the student working just above their current level, in the zone of proximal development (ZPD) [29], while the one-question / one-hint format keeps mental load low so that the efort goes into understanding [36, 37]. EEG engagement is also derived and recorded passively and continuously.

• Mode 3 Adaptive Exercise AI: No Chatbot: Here we do not have a conversational interface. If the student fails (< 80%, denoting a partial or incorrect answer based on the evaluation criteria), the system redirects them through a set of simpler sub-questions. If they also struggle with those, they can generate tailored hints for each one. In addition, this mode actively ingests real-time EEG data to adapt visual feedback and question framing. If the engagement drops below a baseline (calculated for each person), the interface introduces visual stimuli and reformulates the question. Low engagement paired with incorrect answers triggers strong visual feedback (e.g., explosion animations, punchy dramatic questions, etc; see Figure 1b and Figure 2). Correct answers with high engagement get positive reinforcement.

The instructional inputs were identical across the three modes: the same video, ten conceptual topics, and AI always had access to the problem statement and the reference correction. What difered was only how much of that information the system was allowed to reveal. Mode 1 participants, therefore, efectively had easier access to more information, and this was deliberate: an unrestricted chatbot that gives direct answers is exactly what students face when they do their homework with a general-purpose LLM. Likewise, the sub-question redirection is specific to Mode 3 by design: Modes 1 and 2 are about participants thinking through the problem themselves, the AI is there, but the thinking still has to come from them, not from the bot, whereas Mode 3 is diferent: the LLM guides them directly, using sub-questions to lay out the steps toward the answer. Giving those same sub-questions to Modes 1–2 would have mixed the conditions we were trying to compare.

![](images/161ccf19682389b320b79357cc8ca3e08966e4366fe62e56a98ed84b393420ca.jpg)  
(a) Modes 1 and 2: chatbot on the side.  
(b) Mode 3: EEG-driven feedback.

Figure 1: Interface screenshots. (a) Modes 1–2 share the chatbot-on-the-side layout. The screenshot shows Mode 1; Modes 1 and 2 share the exact same interface on purpose (only the system prompt changes), so a Mode 2 screenshot would look identical. (b) In Mode 3, EEG-driven feedback adds a “Focus Required” alert and high-contrast elements when low engagement is detected.  
![](images/e5635b7e1c0d14ee0b20489dbe9bb32952f13e25bf5ea1b88bdf7801df307868.jpg)  
Figure 2: Examples of the EEG-triggered visual feedback (explosion) in Mode 3.

Common evaluation system. For each mode, answers are evaluated in real time using a prompt that includes the problem statement, the student’s answer, and the problem correction. Students must score at least 80% to pass. Otherwise, they can try again as many times as needed and receive brief explanatory feedback. Pilot testing demonstrated that an 80% score reliably indicates conceptual understanding, whereas 100% typically requires exact specific vocabulary.

Conversational AI backend (Modes 1 and 2). Both conversational Modes (mode 1 and 2) use the OpenAI GPT-5.2 model. The chatbot has access to the current problem statement, the student’s answer so far and past trials, and the reference correction. For Mode

2, the system prompt is inspired by the tutoring prompt from Tack and Piech [12], but was rewritten for this study (full text in Appendix A); it enforces Socratic pedagogical constraints: the model never provides the final answer and instead guides the student through subquestions, hints, and targeted challenges.

## 3.2 Hypothesis

Based on prior research, we formulate three hypotheses. (H1) The Socratic method leads to better learning: participants using the Socratic Chatbot (mode 2) will achieve higher learning deltas (Δ = post-test − pre-test) compared to the Unrestricted Chatbot (mode 1), because the deeper cognitive processing required is expected to strengthen retention. The adaptive mode (mode 3) may also produce superior learning delta, though this prediction is more uncertain. (H2) Real-time adaptation maximizes cognitive engagement: participants in the adaptive mode will show higher cognitive engagement (measured by Beta / (Alpha + Theta) [23, 27]) than the conversational conditions, thanks to dynamic dificulty adjustment in the zone of proximal development [29] and real-time EEG-driven visual feedback. (H3) Socratic and adaptive modes enhance perceived engagement: participants in the adaptive (mode 3) and Socratic (mode 2) conditions will report greater subjective engagement and perceived learning, and may also report higher perceived dificulty.

## 3.3 Experimental Procedure

To ensure a fair and consistent comparison across all 3 Modes, every participant went through the exact same seven-step study session.

Experimental Phases: (1) Background questionnaire (with prior knowledge self-assessment); (2) EEG calibration (resting state + Stroop task); (3) Learning phase (10 minutes video); (4) Baseline assessment (pre-test: 10 open-ended questions); (5) AI-assisted study session (randomized mode, 10 open-ended questions with assistance and grading); (6) Final assessment (post-test, 10 open-ended questions); (7) Post-experiment questionnaire.

The experiment took place in a dedicated study room of the research lab on the institute’s campus. Participants faced a wall sitting at an empty table, free from any external distractions like posters, etc. The room is a closed space, without windows. The verbal instruction script at the beginning of the experiment was identical for all participants.

There were no strict time limits, participants received recommended durations ( 10 minutes for the pre and post test, 30 minutes for the study session) with a timer on the screen to help them pace themselves. Time on task was not equal across conditions, due to the nature of the conditions themselves, since a tutor that never provides a direct answer and proceeds by Socratic questioning, will necessarily takes longer. We had to choose between sacrificing equal time or an equal number of questions, and we preferred to hold the number ofquestions constant, both to simulate a real homework situation (where students face a fixed set of exercises rather than a fixed duration of time) and because it makes per-question comparisons more direct.

Mode 3 participants did not receive any hands-on practice phase with the EEG-triggered feedback before training: they only received a short standardized verbal explanation of focus alerts and of how the EEG feedback works.

EEG Protocol and Calibration. The calibration consisted of two phases: first, 30 seconds with eyes open in a resting state (staring at the wall), establishing a low-engagement baseline; second, 30 seconds doing the Stroop Color-Word Test, where colored shapes appear with color words printed in misleading ink colors and participants must pick the word that actually matches the shape’s color. The Stroop task induces high cognitive load [33], providing $E _ { \mathrm { m i n } }$ and $E _ { \mathrm { m a x } }$ values for future normalization. Engagement was also passively measured during the video and the pre-test to identify each participant’s average engagement; together with the calibration data, we calculated the smoothed baseline (via EMA) used for adaptive engagement classification in mode 3.

## 3.4 Participants

A total of 50 participants were recruited: 17 in the Full/unrestricted condition, 17 in Socratic condition, 16 in Adaptive condition. With 50 participants across three conditions, the study is exploratory in nature and detected moderate-to-large efects (Cohen’s � ≥ 0.70). Individuals were aged 18 to 40, recruited locally (� = 27.82, �� = 5.34). Of the 50 participants, 31 were female and 19 male (Full: 11F/6M; Socratic: 8F/9M; Adaptive: 12F/4M). The entire study was conducted in English, and all the materials used in the study were written in English. Participants were recruited from MIT campus and work or study in English : 26/50 were native English speakers (Full: 11/17, Socratic: 6/17, Adaptive: 9/16).

Participants were heavy AI users: 30/50 reported using chatbots daily and 12 weekly. 36/50 reported using them for learning, and the mean self-rated prompting skill was 3.1/5, with similar distributions across the three conditions. Self-reported prior familiarity with the topic of nuclear safety was very low (� = 1.5/10, median = 1, max = 4, no diference across conditions). Interest in the topic itself was not measured.

The study protocol was approved by the IRB of MIT (protocol ID 21070000428). Each participant was randomly assigned to exactly one mode via sequential random allocation: participant 1 to mode 1, participant 2 to mode 2, participant 3 to mode 3, participant 4 to mode 1, and so on.

## 3.5 Materials

We custom-built a web platform hosted on Netlify, using the OpenAI API with GPT-5.2.

Instructional domain. The chosen subject was nuclear safety protocols in a power plants, including defense-in-depth barriers, protection priorities, the chain of responsibility, and analysis of historical failures (Fukushima, Chernobyl). We chose this topic deliberately as a zero-prior-knowledge baseline; the background questionnaire included a self-assessment of prior familiarity to control for any unexpected baseline knowledge.

Knowledge acquisition. 10-minute instructional video with explanatory diagrams, developed from two IAEA publications [28, 30]. Playback control and speed settings were disabled to ensure uniform exposure. The video covers all ten topics of the tests, but at a conceptual level: full credit on the test questions required a degree of details that the video alone rarely provided.

Question design. Three sets of 10 open-ended, short-answer questions (pre-test, training, post-test), for a total of 30 questions inspired by the IAEA publications [28, 30]. Each set covered the same 10 conceptual topics, phrased diferently across the 3 phases, enabling per-topic learning analysis.

Questionnaires. A background questionnaire captured demographics, sleep/alertness/physiological state, prior chatbot experience and use patterns, as well as assessment of prior knowledge on the topic of the study. A post-experiment questionnaire, tailored to each mode (3 slightly diferent versions), shared a common core: perceived learning, problem dificulty, enjoyment, self-reported engagement trajectory, and mode-specific items.

EEG hardware. The Muse 2 headband [32] with 4 active channels (AF7, AF8 on the forehead; TP9, TP10 behind the ears), with Fpz as the reference electrode, mapped to the international 10–20 system. EEG data streams at 256 Hz via Bluetooth and was worn throughout the entire experiment for all 3 Modes.

## 3.6 EEG Signal Processing Pipeline

Raw EEG data is acquired in the browser using muse-js library and processed in real time with rxjs with @neurosity/pipes. The processing pipeline follows established methods from Neurochat paper [23].

The pipeline proceeded as follows: (1) Sample alignment of the raw samples from all 4 channels using zipSamples; (2) Filtering with a band-pass filter (1–30 Hz) and a 60 Hz notch filter to remove power line interference; (3) Epoching into 1-second epochs (250 ms overlapping during calibration, non-overlapping during the ex periment); (4) Spectral decomposition via FFT into Theta (4–8 Hz), Alpha (8–13 Hz), and Beta (13–30 Hz); (5) Engagement computation � = Beta/(Alpha + Theta), introduced by Pope et al. [27], where higher values indicate greater cognitive engagement; (6) Normalization of � to $E _ { \mathrm { n o r m } } \in [ 0 , 1 ]$ using the calibration $E _ { \mathrm { m i n } }$ and $E _ { \mathrm { m a x } } \mathrm { : }$ $E _ { \mathrm { n o r m } } = ( E - E _ { \mathrm { m i n } } ) / ( E _ { \mathrm { m a x } } - E _ { \mathrm { m i n } } )$

For mode 3, the participant’s engagement is classified in real time into three levels (high, normal, low), which directly drives the adaptive visual feedback and question reformulation. The initial baseline is the equally weighted average of the calibration, pre-test, and video phase’s mean engagement, and is gradually updated after each problem using an Exponential Moving Average (EMA) with a small weight, so the baseline drifts slowly while still adapting to natural shifts in cognitive state.

## 3.7 Grading and Evaluation

Real-time grading (assessment phase). During the training phase (for Modes 1, 2, and 3), answers were graded in real-time by GPT-5.2, using a standardized prompt that includes the problem statement, the student’s answer, the correction and the notation grid. The temperature was set to 0 for deterministic outputs, so with the fixed rubric the same answer almost always receives the same score. We observed no participant blocked by a hallucinated grade; the complaint reported by some participants was rather that reaching 100% required specific vocabulary, whereas the 80% pass threshold indicated conceptual understanding. While auto mated grading may not be perfectly accurate in absolute terms, it was applied in the same way across all Modes with the same parameters. We acknowledge, however, that this does not by itself guarantee identical grading behavior across conditions — in particular, GPT may grade GPT-phrased answers more favorably; Section 5.4.5 quantifies this concern and we discuss this further in the Limitations section.

Pre-test and post-test grading. All pre- and post- test responses were graded by GPT-5.2 with the exact same parameters to maintain consistent criteria. The scoring grid was: 90–100 (conceptually complete and correct), 70–89 (mostly correct but not the specific vocabulary), 40–69 (partial understanding, misses major elements); 1–39 (mostly incorrect but not blank); 0 (blank or irrelevant). Temperature was set to 0 so identical answers always receive the same score. During the sessions, a human evaluator informally spot-checked GPT scores (1–2 questions per participant, 60–120 items total) to make sure they were reasonable; however, those checks were never written down as independent scores, so no human–AI inter-rater reliability statistic can be computed retrospectively (see Section 6.3).

## 4 Data Analysis Plan

## 4.1 Primary and Secondary Metrics

To measure the impact of each learning approach, we looked at several diferent metrics:

• Primary Metric : Learning Delta (Δ): The diference between each post-test and pre-test question’s scores (across the 10 conceptual topics). This is the main dependent variable for between-group comparisons.

• Primary Metric : Cognitive Engagement: The normalized EEG Engagement $\left( E _ { \mathrm { n o r m } } \right)$ measured across each experimental phase and per question, compared between modes in the training phase.

• Secondary Metrics:

– comparison of the interactions with the restricted pedagogical bot and the unrestricted Bots

– Number of interactions/attempts required to reach the 80% mastery threshold

– Self-reported engagement and perceived learning (from post-experiment questionnaires).

## 4.2 Statistical Methods

Two statistical approaches were applied to each primary metric: a one-way ANOVA on participants’ average learning Delta and engagement, and a Linear Mixed Model (LMM) on the full perquestion repeated-measures data. The ANOVA tested whether the three mode group means difered significantly, using each participant’s mean score (averaged across the 10 questions) as a single observation (� = 50). The LMM addressed the hierarchical structure of the data (questions clustered within participants, within modes) by modeling all � = 500 per-question observations simul taneously, with modes as a fixed efect and participant as a random intercept:

$$
Y _ { i j } \sim \mathrm { M o d e } _ { j } + ( 1 \mid \mathrm { P a r t i c i p a n t } _ { i } )
$$

where $Y _ { i j }$ is the dependent variable (learning delta $\Delta _ { q }$ or normalized engagement $E _ { \mathrm { n o r m } } )$ for question � of participant � in mode �.

ANOVA computation. When the omnibus �-test is significant $( p < . 0 5 )$ , complementary comparisons are conducted using Welch’s �-tests, with Cohen’s � as the efect size.

LMM computation. The LMM was estimated using REML, with pairwise modes compared via estimated marginal means (we report �-statistic and 95% CI). For learning gains the model ran on 500 change scores; for EEG engagement on all 22,775 cleaned EEG segments recorded during training.

## 5 Results

All 50 participants completed the experiment. Results were compared based on four criteria described above : baseline equivalence, learning outcomes (learning delta), cognitive engagement (EEG), and conversational interaction patterns (discussion clusters).

## 5.1 Baseline Equivalence: Pre-Test Scores

A critical prerequisite for a valid group comparisons is that participants began the experiment with equivalent knowledge levels. The three groups exhibited highly similar mean pre-test scores: Full $( \bar { S } _ { \mathrm { p r e } } = 3 7 . 1 9 , S D = 8 . 3 3 )$ , Socratic $( \bar { S } _ { \mathrm { p r e } } = 3 8 . 4 0 , S D = 8 . 3 0 ) _ { \mathrm { : } }$ and Adaptive $( \bar { S } _ { \mathrm { p r e } } = 3 4 . 4 9 , S D = 8 . 4 9 )$ . A one-way ANOVA confirmed no significant diference between groups $\left( F ( 2 , 4 7 ) = 0 . 9 3 8 , \right.$ $\mathinner { p \mathopen { \left. \vert { \begin{array} { l } { \phantom { + } } \end{array} } \right. } }$ . This shows that the three modes’ groups started with approximately equivalent baseline knowledge of the nuclear safety domain, confirming the validity of the zero-prior-knowledge assumption and ensuring that any post-test diferences reflect the efect of the training mode rather than pre-existing knowledge diferences.

## 5.2 Learning Delta (Δ)

5.2.1 Overall Learning Delta by Mode. The overall mode delta is the mean across all 10 questions and all participants in that group, resulting in $N = 1 7 0$ observations for Full and Socratic (17 participants × 10 questions) and $N = 1 6 0$ for Adaptive (16 participants × 10 questions).

Table 1 presents the aggregate learning delta statistics for each mode.

Table 1: Learning delta (Δ) statistics by Mode.
<table><tr><td>Mode</td><td>N</td><td>Mean  $\Delta$ </td><td>SD</td><td>Median</td></tr><tr><td>Full (unrestricted)</td><td>17</td><td>20.12</td><td>11.85</td><td>25.00</td></tr><tr><td>Socratic</td><td>17</td><td>10.52</td><td>12.02</td><td>6.50</td></tr><tr><td>Adaptive</td><td>16</td><td>11.20</td><td>8.95</td><td>9.55</td></tr></table>

The Full (unrestricted) mode produced the highest overall learning delta $( \bar { \Delta } = 2 0 . 1 2 , S D = 1 1 . 8 5 )$ , compared to the Socratic mode $( \bar { \Delta } \ = \ 1 0 . 5 2 , \ S D \ = \ 1 2 . 0 2 )$ and the Adaptive mode $( \bar { \Delta } \ = \ 1 1 . 2 0 ,$ $S D = 8 . 9 5 ) $ ) conditions. This result is against hypothesis H1, which predicted that the Socratic condition would result in higher learning gains than the unrestricted condition through productive struggle and deeper cognitive processing.

Statistical tests. A one-way ANOVA on the per-participant mean delta revealed a significant main efect of mode $( F ( 2 , 4 7 ) = 3 . 9 4 8 ,$ $p ~ = ~ . 0 2 6 )$ . Pairwise Welch’s �-tests with Cohen’s � efect sizes showed:

• Full vs. Socratic: $t = 2 . 3 4 5 , p = . 0 2 5 , d = 0 . 8 0$ (large efect). Participants with unrestricted access achieved significantly higher deltas than those in the Socratic condition.

• Full vs. Adaptive: $t = 2 . 4 4 8 , p = . 0 2 1 , d = 0 . 8 5$ (large efect). The unrestricted condition also outperformed the adaptive exercise system.

• Socratic vs. Adaptive: $t = - 0 . 1 8 6 , p = . 8 5 4 , d = - 0 . 0 6$ (negligible efect). The modalities 2 and 3 produced statistically indistinguishable learning outcomes.

Complementary LMM analysis. To account for the repeated measures (10 learning delta per participant), we additionally fit a linear mixed model on every deltas (� = 500 rows):

$$
\Delta _ { q } \sim \mathrm { M o d e } + ( 1 \ | \ \mathrm { P a r t i c i p a n t } ) .
$$

The results were consistent with the ANOVA summary (Full: 20.12, Socratic: 10.52, Adaptive: 11.20) :

• Socratic / Full: estimate = −9.60, �� = 3.68, � = −2.61, $\begin{array} { r } { \dot { p } = . 0 0 9 , 9 5 \% \mathrm { C I } \left[ - 1 6 . 8 2 , - 2 . 3 8 \right] . } \end{array}$

• Adaptive / Full: estimate = −8.92, �� = 3.74, � = −2.39, $\begin{array} { r } { { \boldsymbol { p } } = . 0 1 7 , 9 5 \% \mathrm { C I } \left[ - 1 6 . 2 5 , - 1 . 5 9 \right] . } \end{array}$

• Adaptive / Socratic: estimate = 0.68, �� = 3.74, � = 0.18, $\begin{array} { r } { p = . 8 5 5 , 9 5 \% \mathrm { C I } \left[ - 6 . 6 5 , 8 . 0 1 \right] . } \end{array}$

The LMM confirms the same pattern as the ANOVA/Welch analyses : the Full condition significantly outperforms the 2 other conditions, while Socratic and Adaptive do not difer significantly.

5.2.2 Per-Question Learning Delta. Examining the mean delta per question by mode, the Full condition achieved the highest delta on most questions (Q1 to Q7), with particularly strong gains on $Q 4 \left( \Delta = 3 1 . 5 3 \right)$ and Q7 $( \Delta = 2 7 . 7 6 )$ . The Adaptive mode showed a progressive increase in delta across the session (from $\Delta = 3 . 1 6$ on Q1 to Δ = 19.97 on Q9), suggesting that participants benefited increasingly from the adaptive mode as they got to understand the system. The Socratic mode exhibited more variable performance.

## 5.3 EEG Cognitive Engagement

5.3.1 Engagement During the Training Phase. The three groups exhibited comparable engagement during all pre-training phases: calibration (Full: 0.546, Socratic: 0.522, Adaptive: 0.590), video (Full: 0.519, Socratic: 0.477, Adaptive: 0.477), and pre-test (Full: 0.436, Socratic: 0.447, Adaptive: 0.464). A one-way ANOVA on baseline engagement (mean across calibration, video, and pre-test) confirmed no significant group diferences for the pre training phases $( F ( 2 , 4 7 ) = 0 . 3 9 2 , p = . 6 7 8 )$ , validating the between-group comparison on the training phase. Table 2 reports the detailed statistics for the training phase.

Table 2: Mean EEG engagement index $( E _ { \mathbf { n o r m } } )$ during training by mode.
<table><tr><td>Mode</td><td>N</td><td>Mean  $E _ { \mathbf { n o r m } }$ </td><td>SD</td><td>Median</td></tr><tr><td>Full (unrestricted)</td><td>17</td><td>0.389</td><td>0.129</td><td>0.395</td></tr><tr><td>Socratic</td><td>17</td><td>0.460</td><td>0.211</td><td>0.422</td></tr><tr><td>Adaptive</td><td>16</td><td>0.547</td><td>0.215</td><td>0.519</td></tr></table>

Only the Adaptive condition’s mean exceeds the 0.5 baseline threshold, indicating that participants in this condition were, on average, more cognitively engaged during the training phase than during the preceding calibration, video, and pre-test phases.

Statistical tests. A one-way ANOVA on per-participant mean $E _ { \mathrm { n o r m } }$ during training resulted in a marginally significant main efect of mode $( F ( 2 , 4 7 ) = 2 . 9 0 8 , p = . 0 6 4 )$ . Pairwise Welch’s �-tests revealed:

• Full vs. Adaptive: � = −2.545, � = .018. The Adaptive condition produced significantly higher engagement than the unrestricted chatbot.

• Socratic vs. Adaptive: � = −1.170, � = .251. The diference between Socratic and Adaptive engagement did not reach significance.

• Full vs. Socratic: � = −1.192, � = .244. No significant diference between the two conversational modalities.

Complementary LMM analysis. We also fit a mixed model at the EEG epoch level during training (� = 22,775 epochs):

$$
E _ { \mathrm { n o r m } } \sim \mathrm { M o d e } + ( 1 \mid \mathrm { P a r t i c i p a n t } ) .
$$

Estimated means were Full = 0.388, Socratic = 0.460, Adaptive = 0.547. Pairwise contrasts showed:

• Socratic - Full: estimate = 0.0716, �� = 0.0627, � = 1.14, � = .253, 95% CI [−0.051, 0.195].

• Adaptive - Full: estimate = 0.1588, �� = 0.0637, � = 2.49, � = .013, 95% CI [0.034, 0.284].

• Adaptive - Socratic: estimate = 0.0871, �� = 0.0636, � = 1.37, � = .171, 95% CI [−0.038, 0.212].

This complementary measures analysis confirms the ANOVA interpretation: participants in the Adaptive mode have an engagement that is significantly higher than the Full mode, while diferences involving Socratic remain non-significant.

These results support hypothesis H2: the Adaptive condition significantly outperformed the Full condition in engagement and focus during the task.

5.3.2 Per-Question Engagement. The Adaptive and Full mode maintained relatively stable engagement throughout the session. The Socratic mode exhibited its highest engagement in the first half of the session (Q2 to Q5, reaching 0.529 on Q5) followed by a decline toward the final questions (Q10: � = 0.297). This late-session disengagement in the Socratic condition is further analyzed in Section 5.4.

## 5.4 Discussion Strategy Analysis (Mode 1 vs. Mode 2)

5.4.1 Computation Methodology. Although Modes 1 and 2 share the exact same interface and model, the two conditions diverged on every behavioral measure we collected: median training time, number of attempts, copy-pasting, and perceived dificulty. Time on task difered as expected by design, with Socratic sessions taking longer than Full sessions.

To understand how participants interacted with the conversational AI, we developed a rule-based classification system that categorizes each discussion into one of eight behavioral clusters.

This analysis applied only to mode 1 (Full/unrestricted) and mode 2 (Socratic), as mode 3 (Adaptive) has no conversational interface. For each exercise attempt (defined as the sequence of user messages and AI responses before submission), we first detected copy-paste of the exercise statement or of the chatbot’s answer. If the message was not a copy-paste, user messages were matched against pattern-based categories: ask for explanation (e.g. “why”, “explain”), express confusion (“I don’t understand”, “not sure”), ask for definition (“what is/are”, “define”), ask for confirmation (“is this correct”, “am I right”), and propose/refine answer (“I think”, “maybe...?”, “let me try”). Discussions with no chatbot interaction were labeled No discussion and those matching no category as Other. Each classified discussion was linked to the training score for that exercise.

The categories were derived from our pilot sessions and match the classic help-seeking distinction between getting the answer (copy-paste) and trying to gain an understanding (explanation, definition, confusion, proposing an answer) [38, 39]. The rules were applied (copy-paste detection first, then the pattern-based categories), so each discussion received a single dominant label even when a conversation mixed several strategies. The Other category is mostly composed of content questions (e.g., “Which are the barriers in a nuclear system?”), not of of-topic behavior.

5.4.2 Distribution of Discussion categories. Table 3 summarizes the proportional distribution of each discussion category across the two conversational modes:

Table 3: Discussion strategy distribution by mode.
<table><tr><td>Strategy</td><td>Full</td><td>% Socratic</td><td>%</td></tr><tr><td>Copy-paste question Ask for definition</td><td>90 52.9</td><td>37</td><td>21.8</td></tr><tr><td rowspan="3">No discussion Other</td><td>38 22.4</td><td>49</td><td>28.8</td></tr><tr><td>26 15.3</td><td>38</td><td>22.4</td></tr><tr><td>11 6.5</td><td>28</td><td>16.5</td></tr><tr><td rowspan="3">Ask for explanation Express confusion Propose/refine answer</td><td>3 1.8</td><td>3</td><td>1.8</td></tr><tr><td>2 1.2</td><td>11</td><td>6.5</td></tr><tr><td>0 0.0</td><td>4</td><td>2.4</td></tr><tr><td>Ask for confirmation</td><td>0 0.0</td><td>0</td><td>0.0</td></tr><tr><td>Total</td><td>170</td><td></td><td>170</td></tr></table>

5.4.3 Behavioral Observations During the Study. Beyond the automated cluster analysis, direct observation of participants during the experiment sessions revealed distinct behavioral patterns in each conversational mode.

Mode 1 (Full/unrestricted). The majority of Mode 1 participants quickly discovered that the chatbot could provide direct and complete answers to the exercises. Once this was understood, most participants shifted their strategy toward systematic answer extraction: they would copy-paste the exercise prompt, retrieve the chatbot’s response, and transcribe it as their answer. Some participants also used the chatbot to reformulate or simplify the question before requesting the answer. The data clustering confirms this behavior : 52.9% of all exercise interactions in Mode 1 were classified as copy-paste question, and the remaining interactions were mostly ask for definition (22.4%).

Mode 2 (Socratic). In Mode 2, where the chatbot is constrained to never provide the final answer and instead guides through questions, two distinct behavioral profiles emerged among participants. The first and larger group quickly got frustrated with the constraints of the Socratic approach. These participants approached the chatbot with the same mental model as a general-purpose LLM : expecting direct, immediate answers. When the chatbot consistently redirected them by asking guiding questions or providing guidance instead of direct solutions, these participants progressively disengaged from the chatbot, using it less and less as the session progressed. The second, smaller group of participants genuinely engaged with the Socratic dialogue: they asked follow-up questions and explored concepts in depth.

5.4.4 Quantitative Evidence: Chatbot Disengagement Over Time. Mode 2 participants exchanged far more messages overall $( \bar { M } =$ 119.2 total messages per session, $S D = 5 0 . 2 )$ compared to Mode $1 \ ( \bar { M } = 3 7 . 3 , S D = 1 9 . 8 )$ . This is expected: the Socratic chatbot’s refusal to provide direct answers forces participants into longer exchanges to arrive at the solution. However, the per-question trajectory reveals a disengagement pattern in Mode 2 (the mean number of chat messages declines after Q5). In Mode 1, the decline is also present but more moderate, reflecting a natural eficiency gain as participants learned to talk with the chatbot.

This behavioral disengagement in Mode 2 converges with the EEG data: the Socratic mode’s EEG engagement drops to its lowest point at Q10 $( E _ { \mathrm { n o r m } } = 0 . 2 9 7 $ , well below the 0.5 baseline). Together, these two independent measures , behavioral (fewer messages) and neurophysiological (lower engagement), indicate that the majority of Mode 2 participants who expected the chatbot to talk as an answer-providing tool progressively abandoned it when it failed to meet their expectations.

5.4.5 Linguistic Similarity Between Participant Answers andChatbot Text. To quantify how much participants’ submitted answers were contaminated by the chatbot’s phrasing, we compared each final training answer with the chatbot messages of the corresponding exercise. Mean similarity between participant answers and chatbot messages was 0.374 in Mode 1 versus 0.044 in Mode 2 $(  { p } < 1 0 ^ { - 1 8 } )$ and 25.3% of Mode-1 final training answers were nearly identical to the chatbot text, against 0% in Mode 2. This confirms that Mode 1 answers were heavily derived from the chatbot, while Mode 2 answers were formulated by the participants themselves.

## 5.5 Number of Attempts to Reach Mastery

A key feature ofthe training phase is the 80% threshold: participants must achieve a score of at least 80 on each exercise before moving on to the next question. Socratic participants required nearly twice as many attempts $( M = 2 . 8 0 , S D = 0 . 7 3 )$ as Full participants (� = $1 . 4 6 , S D = 0 . 5 9 )$ ; Welch’s � = −5.865, � < .0001, � = −2.01 (very large efect). The comparison excludes Mode 3, where the adaptive condition redirects participants to simpler sub-questions after an incorrect answer. The threshold was almost never reached in the Socratic condition: 58% of the exercises of participants in Socratic Mode, who abandoned the dialogue mid-way, never reached the 80% threshold, meaning those participants ended the exercise without having produced a complete, correct formulation of the answer. We saw no evidence of participants being blocked by a hallucinated grade, but sometime the barrier was the specific vocabulary required for a full credit.

## 5.6 Self-Reported Engagement and Perceived Learning

To complement the objective measures of learning and engagement, the post-questionnaire asked participants to rate their subjective experience on several points. These self-report measures address hypothesis H3.

5.6.1 PerceivedLearning. Participants rated how much they thought they had learned during the session on a 0-10 scale. The three modes produced very similar scores (Full: $M = 6 . 0 6 , S D = 1 . 8 1 ;$ Socratic: $M = 6 . 1 1 , S D = 2 . 0 3 ;$ Adaptive: $M = 5 . 7 5 , S D = 1 . 9 5 )$ . A one-way ANOVA revealed no significant diference between groups $( F ( 2 , 4 7 ) = 0 . 1 6 9 , p = . 8 4 5 )$ . Participants in Full mode, who achieved nearly double the learning delta of the other two conditions, did not perceive themselves as having learned more.

5.6.2 Perceived Dificulty. Participants rated the dificulty of the questions on a 1-10 scale (Full: $M \ : = \ : 6 . 7 5 , \ : S D \ : = \ : 1 . 9 5 ;$ Socratic: $M = 8 . 0 6 , S D = 1 . 5 5 ;$ Adaptive: $M = 7 . 8 8 , S D = 1 . 7 1 ) .$ . The omnibus ANOVA was marginally significant $( F ( 2 , 4 7 ) = 2 . 7 3 4 , p = . 0 7 5 )$ Pairwise comparisons indicated that Socratic participants perceived the exercises as significantly more dificult than Full participants $( t = - 2 . 1 4 3 , \boldsymbol { p } = . 0 4 1 , d = - 0 . 7 4 )$ , while the Adaptive-Full comparison was marginally significant $( t = - 1 . 7 3 6 , p = . 0 9 3 , d = - 0 . 6 1 )$ No significant diference between Socratic and Adaptive $( p = . 7 5 0 )$ These results are consistent with the objective data: the two constrained modes (mode 2 and 3) imposed greater cognitive demands, while the copy-paste strategy in mode 1 made the exercises feel easier.

Perceived dificulty was not correlated with objective engagement: per participant, perceived dificulty and mean EEG engagement during training showed no correlation $( r = . 1 2 6 , p = . 3 8 8 ,$ $n = 4 9 )$ , with the same null pattern within each condition. Perceived dificulty did not correlate with perceived learning either $( r = - . 0 6 )$

5.6.3 Chatbot and System Helpfulness. For the two conversational modes, participants rated the chatbot’s helpfulness on a 0-10 scale. The Full condition rated the chatbot as highly helpful $( \bar { x } = 8 . 6 2 ,$ $S D \ = \ 1 . 3 1 )$ , while the Socratic condition rated it notably lower $( \bar { x } = 5 . 6 7 , S D = 2 . 0 0 ) ; \mathrm { W e l c h } ^ { \prime } s t = 5 . 1 5 4 , p < . 0 0 0 1 , d = 1 . 7 5$ (very large efect). Several Socratic participants explicitly described their frustration: “Itfelt annoying and frustrating” (P035); “Feeling almost crazy” (P029); “I just gave up” (P014).

## 6 Discussion

## 6.1 Interpretation of Results

The central finding is that the unrestricted chatbot (mode 1) produced nearly double the learning delta of the Socratic (mode 2) and Adaptive (mode 3) conditions, going against hypothesis H1. However, the data and analysis of the bot discussions suggests that the success of the unrestricted AI is not evidence of deeper learning, but rather a result of the immediate post-test evaluation after the training phase.

Direct answer retrieval in Mode 1. The discussion cluster analy sis (Table 3) provides an explanation for mode 1’s better learning delta: 52.9% of all exercise interactions were classified as copy-paste question, meaning that participants pasted the exercise statement directly into the chatbot and received the complete answer. Because the post-test was administered immediately after the homework phase, this short-term retention was suficient to produce high scores. The per-question delta supports this interpretation: mode 1 consistently achieved the highest deltas on Q1 to Q7, precisely the questions where participants spent the most energy extracting answers from the chatbot.

Participants “gave up” on the Socratic chatbot. In contrast, mode 2 participants faced a very diferent interaction dynamic. The Socratic chatbot’s design to refuse providing final answers forced participants into extended dialogues. While this is pedagogically sound in principle, especially with respect to supporting a produc tive struggle and deeper understanding, the data reveals that most of the participants did not complete this process and "gave up". Behaviorally, the mean number of messages per question in mode 2 declined after question 5, showing that participants progressively stopped engaging with the chatbot’s guided questioning. Neurophysiologically, EEG engagement in mode 2 dropped to its lowest value at Q10 $( E _ { \mathrm { n o r m } } = 0 . 2 9 7 )$ , well below the 0.5 personal baseline (Full: $E _ { \mathrm { n o r m } } = 0 . 4 5 1$ ; Adaptive: $E _ { \mathrm { n o r m } } = 0 . 4 4 3 )$

Participants who abandoned the Socratic dialogue before reaching a complete understanding were left with only a partial or frag mented answer to retain for the post-test, penalizing performance regardless of any partial understanding developed.

Why the shared video did not provide equal outcomes. The video was complete but not suficient: right after watching it, all groups scored only 34–38/100 on the pre-test (no group diference, � = .399). Post-test performance therefore mostly reflects the training phase. Mode 1 repeatedly showed complete, well-formulated answers (52.9% copy-paste). Socratic participants who quit middialogue never got a complete formulation at all (58% of their responses never reached the threshold): little to retain, whatever partial understanding they had built.

Seen through the lens of cognitive load theory [36, 37], the three conditions distributed mental efort very diferently: Mode 1 let participants finish the task with almost no mental efort, Mode 2 required an efort that most participants eventually gave up on, and Mode 3 kept engagement by splitting the task into sub-questions itself, taking over some of the load.

Immediate testing favors retention of answers over understanding. The post-test followed the assessment session with no delay, so the experiment measures short-term recall rather than deep con ceptual understanding or long-term retention. Mode 1’s complete answers create an optimal pathway for immediate recall, mirroring the documented "illusion of competence,” where exposure to complete solutions results in high immediate performance [9]. Mode 2’s Socratic approach, by contrast, builds understanding through guided discovery, a process that takes time and efort.

Engagement and learning dissociation. An important secondary finding is the dissociation between cognitive engagement (EEG)

and learning outcomes. The Adaptive condition produced significantly higher EEG engagement than Mode 1 $( p = . 0 1 8 )$ , partially supporting H2, yet this did not translate into superior learning deltas $( \bar { \Delta } = 1 1 . 2 0$ for Adaptive vs. 20.12 for Full). Cognitive engagement, while necessary, is not suficient when the assessment rewards the completeness of the final answer rather than the depth of the process: mode 1 participants achieved high deltas with low engagement by transcribing AI answers, while mode 3 participants stayed focused but produced only moderate gains.

A null resultfor perceived learning. Perhaps the most unexpected finding is the absence of any significant diference in perceived learning across modes $( p = . 8 4 5 )$ . Full mode participants, who objectively have a learning delta nearly twice bigger, did not feel they had learned more than Socratic or Adaptive mode participants. Full mode participants may have been aware that their good results were driven by answer retrieval rather than deep understanding (Quote from P031: “A bit like cheating, copy/pasting context, questions and answers”). The frustration-driven disengagement observed in the Socratic condition is a critical challenge for Socratic AI systems: this pedagogical path can only be beneficial if learners persist through the discomfort, which many did not.

## 6.2 Practical Implications

The choice of AI architecture must be aligned with the assessment format and learning timeline. Unrestricted AI access maximizes immediate post-session performance but may favor superficial knowledge; Adaptive approaches may be preferable for deeper understanding but must be carefully designed to sustain engagement and prevent premature abandonment. The Adaptive condition’s significantly higher engagement demonstrates the promise of real-time neurophysiological feedback, but the translation of this engagement into measurable learning gains requires further work, particularly with delayed post-tests.

More specifically, our findings suggest one such design, in which each element is based on a finding of this study: (1) deployment of Socratic dialogue by default, but when EEG shows sustained disengagement below the personal baseline, switch temporarily to Mode-3-style sub-questions : the format that kept engagement highest, and then return to dialogue. (2) When struggle stops being productive (repeated failures, shrinking messages), give progressively more concrete hints instead of indefinite refusal, keeping the learner in the zone of proximal development (ZPD) [29]. Such a system design is a suggestion, and should be evaluated in a formal study with a delayed post-test.

## 6.3 Limitations and Future Work

Grading validity comes first among our study’s limitations: the human checks of GPT scores during the sessions were done by eye to make sure the scores were reasonable, but they were never written down as scores, so we cannot compare human and GPT grading directly, and no inter-rater reliability can be reported; a systematic double-grading of a subset is planned as future work. It is worth noting, however, that the main behavioral results (copypaste rates, disengagement, time on task, EEG) do not come from GPT scores, and they compliment the obrained results.

Our study was restricted to a single subject domain and mostly focused on factual knowledge acquisition, so replication across multiple domains and open-ended problem-solving questions is needed. Due to the Hawthorne efect [35], participants may behave diferently because of the EEG device. The study accounts for shortterm retention only; long-term retention was not measured, and the recommended 30-minute duration may have created implicit time pressure penalizing Mode 2 participants. With 16–17 participants per condition, the sample does not provide the power for subgroup analyses by AI/LLM proficiency use or age, which remain exploratory.

Additional limitations should be noted. Interest in the topic was not measured: nuclear safety was chosen as a zero-prior-knowledge domain, and we cannot rule out that low interest made the task feel like something to complete rather than an occasion to learn. Finally, Mode 3 participants had no hands-on familiarization phase with the EEG-triggered feedback, which likely explains the warm-up efect we observe (Mode 3’s per-question delta climbs from 3.2 on Q1 to 20.0 on Q9 as participants learn the system): future versions of the studies of this nature should include a short practice session.

Two directions for future work emerge: a delayed post-test (one to two weeks after the learning session) to analyze whether Mode 2 and Mode 3’s deeper cognitive engagement translates into superior long-term retention; and replication with larger, more diverse participant pools, additional LLM models, and a human teacher control.

## 7 Conclusion

This study compared three AI modes with 50 participants learning nuclear safety protocols. The unrestricted chatbot produced the highest learning delta $( \bar { \Delta } \ : = \ : 2 0 . 1 2 )$ , outperforming both Socratic $( d = 0 . 8 0 )$ and Adaptive (� = 0.85) conditions. The Adaptive condition produced the highest EEG engagement $( p = . 0 1 8 )$ and was the only condition above the 0.5 personal baseline. More than 50% of mode 1 interactions involved direct answer retrieval, while mode 2 exhibited progressive disengagement. Perceived learning and perceived dificulty are self-reported measures and should be interpreted as such.

## References

[1] N. Kosmyna, E. Hauptmann, Y.T. Yuan, J. Situ, X.-H. Liao, A.V. Beresnitzky, I. Braunstein, and P. Maes. Your Brain on ChatGPT: Accumulation of Cogni tive Debt when Using an AI Assistant for Essay Writing Task. arXiv preprint arXiv:2506.08872, 2025.

[2] M. Gerlich. AI Tools in Society: Impacts on Cognitive Ofloading and the Future of Critical Thinking. Societies, 15(1):6, 2025.

[3] M. Stadler, M. Bannert, and M. Sailer. Cognitive Ease at a Cost: LLMs Reduce Mental Efort but Compromise Depth in Student Scientific Inquiry. Computers in Human Behavior, 160:108386, 2024.

[4] Y. Fan, L. Tang, H. Le, K. Shen, S. Tan, Y. Zhao, Y. Shen, X. Li, and D. Gašević. Beware of Metacognitive Laziness: Efects of Generative Artificial Intelligence on Learning Motivation, Processes, and Performance. British Journal ofEducational Technology, 56(2):489–530, 2025.

[5] E.A. Assefa. The Double-Edged Sword of Artificial Intelligence (AI) in Education: Maximizing Benefits While Mitigating Risks. The Journal of Quality in Education (JoQiE), 14(24), 2024.

[6] H.L. Schwartz and M.K. Diliberti. More Students Use AI for Homework, and More Believe It Harms Critical Thinking. RAND Corporation, Research Report RRA4742-1, 2026.

[7] S. Ortiz-Bonnin and J. Blahopoulou. Chat or Cheat? Academic Dishonesty, Risk Perceptions, and ChatGPT Usage in Higher Education Students. Social Psychology ofEducation, 28(1):Article 113, 2025. doi:10.1007/s11218-025-10080-2.

[8] L. Fesler, J.P. Martinez Claeys, C. Agnew, and S. Loeb. The Evidence Base on AI in K-12: A 2026 Review. Stanford SCALE Initiative, AI Hub for Education, 2026. Available at: https://scale.stanford.edu/sites/default/files/The%20Evidence% 20Base%20on%20AI%20in%20K-12%20Report.pdf

[9] E.L. Bjork and R.A. Bjork. Making Things Hard on Yourself, But in a Good Way: Creating Desirable Dificulties to Enhance Learning. In Psychology and the Real World, pp. 56–64, 2011.

[10] A.C. Kulesa, M. Mission, M. Croft, and M.K. Wells. Productive Struggle: How Artificial Intelligence Is Changing Learning, Efort, and Youth Development in Education. Bellwether, June 2025.

[11] E. Gkintoni, H. Antonopoulou, A. Sortwell, and C. Halkiopoulos. Challenging Cognitive Load Theory: The Role of Educational Neuroscience and Artificial Intelligence in Redefining Learning Eficacy. Brain Sciences, 15(2):203, 2025.

[12] A. Tack and C. Piech. The AI Teacher Test: Measuring the Pedagogical Ability of Blender and GPT-3 in Educational Dialogues. In Proceedings of the 15th International Conference on Educational Data Mining (EDM), pp. 522–529, 2022.

[13] D. Dinucu-Jianu, J. Macina, N. Daheim, I. Hakimi, I. Gurevych, and M. Sachan. From Problem-Solving to Teaching Problem-Solving: Aligning LLMs with Pedagogy using Reinforcement Learning. arXiv preprint arXiv:2505.15607, 2025.

[14] W. Xie, H. Shi, Y. Li, and J. Liu. STAP: A Socratic Tutor for Adaptive Programming with Pedagogical Scafolding. In Proceedings ofthe 2025 2nd International Symposium on Artificial Intelligence for Education (ISAIE ’25), pp. 584–587, 2025. doi:10.1145/3775073.3775165.

[15] B. Liu,J. Zhang, F. Lin, X.Jia, and M. Peng. One Size Doesn’t Fit All: A Personalized Conversational Tutoring Agent for Mathematics Instruction. arXiv preprint arXiv:2502.12633, 2025.

[16] C. Cohn, S. Rayala, N. Srivastava, J.H. Fonteles, S. Jain, X. Luo, D. Mereddy, N. Mohammed, and G. Biswas. A Theory of Adaptive Scafolding for LLM-Based Pedagogical Agents. arXiv preprint arXiv:2508.01503, 2025.

[17] H. Fakour and M. Imani. Socratic Wisdom in the Age of AI: A Comparative Study of ChatGPT and Human Tutors in Enhancing Critical Thinking Skills. Frontiers in Education

[18] N. Kosmyna and E. Hauptmann. NeuroSkill: Proactive Real-Time Agentic System Capable of Modeling Human State of Mind. arXiv preprint arXiv:2603.03212, 2026.

[19] K. VanLEHN. The Relative Efectiveness of Human Tutoring, Intelligent Tutoring Systems, and Other Tutoring Systems. Educational Psychologist, 46(4):197–221, 2011.

[20] K. Vanacore, R.S. Baker, A.H. Closser, and J. Roschelle. The Path to Conversational AI Tutors: Integrating Tutoring Best Practices and Targeted Technologies to Produce Scalable AI Agents. arXiv preprint arXiv:2602.19303, 2026.

[21] A. Scarlatos, N. Liu, J. Lee, R. Baraniuk, and A. Lan. Training LLM-Based Tutors to Improve Student Learning Outcomes in Dialogues. In Artificial Intelligence in Education (AIED 2025), Lecture Notes in Computer Science, vol. 15877, pp. 251– 266. Springer, Cham, 2025. doi:10.1007/978-3-031-98414-3\_18.

[22] A.M. Kassenkhan, M. Mendes, and A. Bekarystankyzy. An Adaptive Task Difficulty Model for Personalized Reading Comprehension in AI-Based Learning Systems. Algorithms, 19(2):100, 2026. doi:10.3390/a19020100.

[23] D. Baradari, N. Kosmyna, O. Petrov, R. Kaplun, and P. Maes. NeuroChat: A Neuroadaptive AI Chatbot for Customizing Learning Experiences. In Proceedings of the 7th ACM Conference on Conversational User Interfaces (CUI ’25), 2025. https://doi.org/10.1145/3719160.3736623.

[24] E. Lekati, G.N. Dimitrakopoulos, K. Lazaros, P. Giannopoulou, A.G. Vrahatis, M.G. Krokidis, P. Vlamos, and S. Doukakis. Wearable EEG Sensor Analysis for Cognitive Profiling in Educational Contexts. Sensors, 25(20):6446, 2025. doi:10.3390/s25206446.

[25] G.P.J.C. Noel, I. Xiao, M. Chaouachi, A. Ilie, J. O’Brien, and S.C. McWatt. Engagement and Cognitive Load of Upper-Year Medical Trainees during Mixed Reality–Enhanced Dissection. Anatomical Sciences Education, 18(11):1278, 2025. doi:10.1002/ase.70126.

[26] N. Beauchemin, P. Charland, A. Karran, J. Boasen, B. Tadson, S. Sénécal, and P.-M. Léger. Enhancing Learning Experiences: EEG-Based Passive BCI System Adapts Learning Speed to Cognitive Load in Real-Time, with Motivation as Catalyst. Frontiers in Human Neuroscience, 18:1416683, 2024.

[27] A.T. Pope, E.H. Bogart, and D.S. Bartolome. Biocybernetic System Evaluates Indices of Operator Engagement in Automated Task. Biological Psychology, 40(1–2):187–195, 1995. doi:10.1016/0301-0511(95)05116-3

[28] International Atomic Energy Agency. Safety of Nuclear Power Plants: Design. IAEA Safety Standards Series No. SSR-2/1, Vienna, 2012. Available at: https: //www-pub.iaea.org/MTCD/Publications/PDF/Pub1534\_web.pdf

[29] L.S. Vygotsky. Mind in Society: The Development of Higher Psychological Processes. Harvard University Press, Cambridge, MA, 1978.

[30] International Atomic Energy Agency. Basic Safety Principles for Nuclear Power Plants, 75-INSAG-3 Rev. 1. INSAG-12, IAEA, Vienna, 1999. Available at: https: //www-pub.iaea.org/MTCD/Publications/PDF/P082\_scr.pd

[31] Pew Research Center. How Teens Use and View AI. 2026. Available at: https: //www.pewresearch.org/internet/2026/02/24/how-teens-use-and-view-ai

[32] InteraXon Inc. Muse 2: Brain Sensing Headband. Available at: https://choosemuse. com/products/muse-2

[33] J.R. Stroop. Studies of Interference in Serial Verbal Reactions. Journal of Experimental Psychology, 18(6):643–662, 1935.

[34] J. Hwang and C. Shin. Toward Personalized Adaptive Learning Using Artificial Intelligence and Physiological Signals. IEEE Access, 2026.

[35] J. McCambridge, J. Witton, and D.R. Elbourne. Systematic Review of the Hawthorne Efect: New Concepts Are Needed to Study Research Participation Efects. Journal ofClinical Epidemiology, 67(3):267–277, 2014.

[36] J. Sweller, J.J.G. van Merriënboer, and F.G.W.C. Paas. Cognitive Architecture and Instructional Design. Educational Psychology Review, 10(3):251–296, 1998.

[37] J. Sweller. Element Interactivity and Intrinsic, Extraneous, and Germane Cogni tive Load. Educational Psychology Review, 22(2):123–138, 2010.

[38] S. Nelson-Le Gall. Help-Seeking: An Understudied Problem-Solving Skill in Children. Developmental Review, 1(3):224–246, 1981.

[39] V. Aleven, E. Stahl, S. Schworm, F. Fischer, and R. Wallace. Help Seeking and Help Design in Interactive Learning Environments. Review ofEducational Research, 73(3):277–320, 2003

## A Socratic Tutor System Prompt

You are an encouraging nuclear technical protocols

training assistant.

You help students solve exercises without ever giving

the solution, even partially.

You have access to:

\- exerciseStatement = the exercise the student must solve

\- studentAnswer = the student's current attempt

\- correction = the reference solution (for your eyes only; NEVER reveal it)

Your goals:

1. Guide the student step-by-step using questions, hints, and reasoning prompts.

2. NEVER reveal any element of the correction, not even small steps.

3. Do not solve the exercise for the student.

4. Ask one question at a time, always based on what

the student answered.

5. Identify misconceptions by comparing studentAnswer

with correction, but only use that to:

\- ask targeted questions

\- propose analogies

\- suggest a path of reflection

6. If the student is wrong or stuck, give a hint, not a solution.

7. Encourage the student to explain their thinking or try a smaller sub-step.

8. When the student makes progress, praise and reinforce.

9. When the student reaches the correct reasoning (even if imperfect), help them consolidate by asking them to restate the idea in their own words.

10. At NO moment should you provide:

\- the correct answer

\- the missing step

\- the final numerical result

\- a direct formula application that reveals the solution.

11. You can give definitions of concepts if the student asks for them.

Always answer in an upbeat, motivating tone, like a real tutor.

At the end of every response, add one short

directional hint (one sentence max) that helps the student know what to check next, without revealing the solution. Prefix it with "Hint:".

Here is the inputs :

\- The exercise statement: {exerciseStatement}

\- The correction: {correction}

\- The trainee's current response: {studentAnswer}