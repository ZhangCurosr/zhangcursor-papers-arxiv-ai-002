# The Importance of Speech Genre for End-of-Turn Detector Training

Rini Sharon<sup>⋆</sup> Kadri Hacioglu<sup>⋆</sup> Jeena Prakash<sup>⋆</sup> Andreas Stolcke<sup>⋆</sup>

<sup>⋆</sup> Uniphore

Abstract—End-of-turn (EoT) detection has been studied mostly through features, models, and modalities, while the speech genre of the training data has drawn far less attention. We isolate this factor. Using one reproducible annotation pipeline, we build a matched pair of corpora, multi-turn human-human conversational and single-turn human-machine, and train the same streaming ConvLSTM EoT detector on each. Across four test sets the conversational model reaches up to +0.45 higher F1, driven by fewer false alarms on the mid-turn pauses and hesitations that are rare in single-turn corpora. This conversational-training effect is not specific to frame-level decoding. Under a matched complete-or-not test it also appears at the utterance level, across two encoders, but only on conversational test data. Testing only on single-turn speech, as most prior work does, therefore hides the benefit of conversational training data. Against open-source baselines, our 1.8M-parameter model attains the highest F1 on every test set.

Index Terms—streaming end-of-turn detection, speech genre, conversational speech, turn-taking, voice activity detection, spoken dialogue systems

## I. INTRODUCTION AND RELATED WORK

Turn-taking is how speakers coordinate who holds the floor and when a turn has ended. In a spoken dialogue system the task comes down to deciding when the user has finished, so the system can respond. Responding too early interrupts the user, and waiting too long makes the exchange feel unnatural [1], [2]. A fixed silence threshold remains a widely used default [1], [3]: it treats every pause the same and cannot tell a mid-turn hesitation from a true turn end [2], [4]. To get past this, prior work has improved end-of-turn (EoT) detection along four directions. Three are well established, namely the features used as turn-taking cues, the modeling approach, and the signal modalities. The fourth and more recent direction is the composition of the training data. We review each in turn. Feature axis. Early systems combined silence-based rules with prosodic cues [5], [6]. Later work showed that richer acoustic features such as pitch, energy, and duration help separate true completions from mid-turn pauses [7], [8], and that higher-level cues beyond acoustics improve prediction further [9].

Modeling axis. Approaches range from voice activity detection (VAD) with fixed pause thresholds [7], [9], to discrete classifiers placed at candidate boundaries [9], to frame-level models that predict upcoming speech activity [7]. Voice activity projection (VAP) uses a transformer to predict joint voice-activity states in a self-supervised way [10], and textonly models learn to predict turn-shift tokens from dialogue corpora [11]. More recent work moves toward semantic endpointing, where the decision depends on what the user said and not only on silence. One such design pairs a lightweight ondevice detector with a stronger server model to tell a true turn end from a mid-turn hesitation [12]. These detectors can run standalone or sit inside ASR and full-duplex spoken language systems [10], [13], [14].

Modality axis. Recent systems combine acoustic encoders with text and visual streams and outperform unimodal baselines on multi-party conversations [15], [16]. Adding gaze, head pose, and facial action units gives further gains in faceto-face settings [16], [17]. Even so, a recent survey reports that 72% of turn-taking studies do not benchmark against prior methods [16], which makes it hard to say whether gains come from the model or from differences in data and evaluation.

Data axis. The training data itself shapes turn-taking behavior, yet it has drawn the least attention of the four axes. Many works train and evaluate on a single corpus with no cross-data comparison [16]. Multilingual VAP transfers poorly across languages when trained on a single language [18], and work extending turn-taking to triadic and multi-party settings points to a similar sensitivity to the training corpus [15], [19]. These difficulties reflect a broader shortage of diverse, real end-ofturn training data, which has prompted recent corpora that mix real and synthetic speech [12].

We focus on this last axis. The training data for EoT detection falls into broad speech genres that differ in the turntaking behavior they contain.

1) Conversational multi-turn speech is human-to-human dialogue, rich in hesitations, backchannels, overlapping speech, and mid-turn pauses [15], [19], [20].

2) Human-to-system dialogue is a user addressing a system that prompts and then waits silently, so turns have cleaner boundaries [21].

3) Command-and-control speech is short single-turn queries with clean onsets and offsets [22].

4) Read or scripted speech is prompted sentences or audiobooks with few turn-taking signatures [16].

5) Mixed real-plus-synthetic speech is human recordings augmented with TTS corpora [23], [24].

What separates these genres is how many within-turn events they expose, and that is exactly what a model can learn from. Conversational data are full of mid-turn pauses, filled pauses, and backchannels that should not trigger an EoT, and a model can learn to hold back at them from their prosodic and contextual cues [7], [25]. The other genres carry far fewer such events. Human-to-system data lack the floor negotiation of human-human dialogue [16], [26], command-style data offer little within-turn material, and synthetic speech may not capture the full range of hesitation cues found in natural speech [16], [27]. The contrast is widened by evidence that speakers adjust their timing and prosody once they know they are addressing a machine [26], [28], [29].

![](images/1810a256e20d1cce86d9e6504f997a1ab6c275edd08fe4296b63c852dd72faba.jpg)  
Fig. 1: Frame-level vs. utterance-level decisions on a turn with two mid-turn pauses. The frame-level detector scores every frame, holds back through both pauses, and fires only at the turn end. The utterance-level classifier is queried only at a pause and commits at the first one it judges complete.

We therefore study two genres that contrast sharply on this within-turn axis: multi-turn human-human speech and singleturn human-machine speech. We pick these because large amounts of real speech can be reliably labelled for both, while the other combinations are either uninformative or hard to obtain. A single utterance with no second speaker offers no turn-taking to learn from, and real multi-turn human-machine dialogue with reliable end-of-turn labels remains scarce, so recent work has had to annotate it from scratch [30]. Restricting the comparison to these two lets us vary one factor, the density of within-turn events, while labelling both genres the same way. This matters because the two are rarely compared under matched conditions: studies that train on conversational data [7], [10] seldom test against single-turn training, and systems built on single-turn or synthetic data [12], [21] seldom test against conversational training. Genre therefore stays underexplored and entangled with differences in architecture, features, and evaluation, so a reported gain cannot be traced to the data itself.

A second gap is about how the end-of-turn decision is produced. A frame-level detector emits a prediction at every frame, so it can hold back through a mid-turn pause and commit only at the turn end [7], [8], [10]. An utterance-level detector instead acts only at a voice-activity pause, classifying the speech up to that pause as a complete or incomplete turn, so it makes a single decision per segment [9], [21]. Figure 1 contrasts the two. Because utterance-level systems are tested almost only on single-turn or clean speech [22], [31] and crossgenre comparison is rare [16], whether conversational training helps them remains untested.

Contributions. To our knowledge, this is the first study to isolate the speech genre of the training data as a controlled variable for end-of-turn detection. Our three contributions measure this effect, bound its scope, and benchmark it against

Algorithm 1 Full-Call EoT Annotation Pipeline   
1: Input. Stereo calls → per-channel mono WAV at 8kHz   
2: VAD + Smoothing. Extract speech regions $R _ { i } = [ t _ { s t a r t } , t _ { e n d } ]$   
per channel, then smooth via gap-filling and noise-pruning (ker  
nel size 3) [35]   
3: Turn Merging. Merge consecutive same-channel $R _ { i } , R _ { i + 1 }$ if gap   
$< ~ 0 . 5 s$ and opponent silent (this reassembles VAD fragments   
into turns, and the EoT is placed in Step 5 only)   
4: Filtering. Discard segments where duration < 0.1 s (noise) OR   
(duration < 0.2 s AND overlaps opponent, a backchannel)   
5: EoT Placement. Mark EoT at $t _ { e n d }$ if opponent speaks after turn   
end or turn is final   
6: Output. Full-call EoT annotations

## prior systems:

1) A controlled comparison of training genre. Using one labelling pipeline, we build conversational and single-turn training sets matched in size and labelling and train the same detector on each, so training genre is the variable under test.

2) The scope of the effect. We rerun this comparison at the utterance level, where most recent detectors operate [21], across conversational and single-turn test genres and two independently pretrained encoders, mapping where the effect holds and where it does not.

3) A benchmark with significance tests. We compare our compact streaming detector against open-source VAD and learned baselines under one protocol with a one-sided tolerance window, a silence-matched utterance test, and significance tests.

## II. METHODOLOGY

This section describes how we build the data, the model that consumes it, the training setup, the datasets we compare on, the baselines, and the metrics.

## A. Data Curation Pipeline

The conversational corpus comes from real human-tohuman call center conversations in US English, spanning banking, insurance, retail, and telecommunications. Recordings are 8 kHz stereo with one channel per speaker, the native rate of telephony, where the channel limits speech to roughly the 300 to 3400 Hz band [32], [33]. The cues that matter for end-ofturn detection sit well below this upper edge, since pitch is usually 80 to 400 Hz and the energy and pause patterns we rely on are low-frequency as well, so 8 kHz preserves them [34]. For baselines that expect 16 kHz input (Silero, SmartTurn, and Whisper), we upsample the audio before inference.

The curation pipeline runs in three stages, turn extraction, silence extension, and frame-level labelling.

Turn extraction. This stage splits each call into single speaker turns and marks the EoT at a turn’s last frame, the point after which the other speaker takes over or the call ends. Algorithm 1 gives the steps.

Silence extension. This stage appends the silence that follows each turn. A sustained silence after the final word is a strong cue that the turn has ended [4], [7]. The window extends up to 1.3 s past the EoT and is VAD-validated to contain at most 10% speech. If the next speaker or the same speaker resumes inside the window, the extension stops at that point. We keep a turn only when it has at least 0.3 s of clean trailing silence, and we cap the window at the file boundary.

Frame-level labelling. This assigns binary labels at 10 ms resolution. Frames before the EoT are labeled 0, and frames from the EoT through the appended silence are labeled 1.

## B. Datasets

Training datasets. The central comparison rests on two 10Kutterance training sets, one conversational and one singleturn. Conv-10K holds 10,000 segments drawn from multi-turn human-to-human call center conversations using the pipeline above. ST-10K holds 10,000 complete single-turn human-tosystem utterances, both real and synthetic, from the SmartTurn v3 training partition [21], drawn as a seeded, stratified random sample that is disjoint from the evaluation utterances. Because each utterance is a complete turn, its end is itself the turn end, so Stage 1 (turn extraction) does not apply. Frame-level labels then come from the same Stage 2 and Stage 3 as Conv-10K. The speech offset of each utterance is taken as its EoT, frames before it are labeled 0 and frames after it 1, and the same silence extension is applied. The labelling rule is thus identical for both genres. We refer to these as conversational and single-turn data throughout, and both models share the same architecture and hyperparameters.

Evaluation datasets. We evaluate on four test sets that span human-human conversation, human-machine dialogue, and single-command interaction, so each training regime is tested both in-domain and out-of-domain. All audio uses the same 8 kHz feature extraction as training.

Internal 5K. 5,000 speaker-separated segments from multiturn customer service calls with human-verified EoT annotations. In-domain for Conv-10K.

CASPER [36]. 2,500 turn segments from an independent open-source corpus of two-party English conversations with hesitations, backchannels, and mid-turn pauses, processed through the same full-call pipeline. It tests whether conversational training carries beyond the source domain.

ST-Complete [21]. 2,500 samples (mean 9.12 s) drawn from the SmartTurn v3 pool of complete turns (turns that have genuinely ended, i.e. with a true EoT). In-domain for ST-10K, which gives a symmetric cross-training comparison. Since SmartTurn ships only binary per-sample labels, frame-level labels come from VAD-based speech detection. SmartTurn reports above roughly 80% accuracy on its own test sets [31], which include real and synthetic audio and both complete and incomplete turns. Our subset keeps only real complete turns, chosen to make the comparison with the conversational and open-source data fair, which accounts for the gap relative to the published SmartTurn numbers.

SLURP [22]. 2,500 short voice commands (mean 2.91 s, test split, seed = 42) with no multi-turn structure. It sits entirely outside the turn detection domain and tests whether an EoT detector carries over to command-and-control speech.

## C. Model Architecture

The model consumes the labelled frames one at a time. Each 10 ms frame is an 85-dimensional vector that joins an 80-dim log-Mel filterbank with 5 prosodic features, namely normalized F0, F0 delta, a voiced flag, a speech flag, and cumulative silence duration.

A Conv1D (kernel size 33, 256 channels) gives 320 ms of lookahead per frame. Its output passes through LayerNorm and a GELU [37] activation, then into a 2-layer unidirectional LSTM (256 hidden units, dropout 0.25 between layers). A 3- layer MLP (256 → 128 → 64 → 1, GELU activations, dropout 0.25) maps to a per-frame EoT logit, and a sigmoid converts this into a per-frame EoT probability. The model has 1.8M trainable parameters. Because the LSTM states carry over across chunks, the model can stream continuously with unbounded left context.

## D. Training Methodology

EoT frames are far rarer than non-EoT frames, so the loss is binary cross-entropy with a positive-class weight $w ^ { + } =$ $N _ { \mathrm { n e g } } / N _ { \mathrm { p o s } } = 5 . 6 $ , equal to the ratio of non-EoT to EoT frames, which stops the model from trivially predicting the majority class. To match streaming inference, each sample is processed chunk by chunk with the LSTM hidden states carried forward, and the loss is masked to each sample’s valid frames.

We optimize with Adam [38] at learning rate $3 \times 1 0 ^ { - 4 }$ and weight decay $1 \times 1 0 ^ { - 4 } .$ , with 2 epochs of linear warmup followed by cosine decay [39] to zero. Gradients are clipped at norm 5.0. Batch size is 8, with samples sorted by length to reduce padding. Training runs up to 30 epochs with early stopping on validation F1 (patience 7).

## E. Baselines

We compare against three paradigms, a signal-based VAD, a neural VAD, and a learned end-of-turn classifier. The two VAD baselines decide from silence alone. Each fires an EoT once the silence after speech grows longer than a threshold, so neither can tell a mid-turn pause from a true turn end. We sweep this threshold from 0 to 1000 ms in 50 ms steps and report the best F1 per baseline per dataset. WebRTC VAD [40] is the signal-based detector at aggressiveness level 3, and Silero VAD v5 [41] is the neural detector with its default threshold of 0.5.

The third baseline, SmartTurn v3 [21], is a learned, utterance-level classifier rather than a silence rule. A Whisper-Tiny encoder [42] with a linear classifier on top (8M parameters in total) labels the speech up to a pause as a complete or incomplete turn, and Silero VAD supplies the pauses (Figure 1). We also evaluate SmartTurn FT, the same model fine-tuned on our conversational data.

To check that any utterance-level effect is not tied to one model, we add two more utterance-level baselines, each a frozen pretrained encoder with only a small classifier trained on top, so the training genre of the classifier is the only variable. The two encoders, Whisper-Tiny [42] and wav2vec2- base [43], are pretrained very differently, one on transcribed speech and one on untranscribed audio, so they encode a segment in independent ways. SmartTurn also builds on Whisper-Tiny, but it updates the whole encoder during training rather than keeping it fixed as we do. With the two SmartTurn baselines, this gives four utterance-level systems across two encoders.

## F. Evaluation Metrics

A positive EoT decision fires when the sigmoid output stays above threshold τ for d = 8 consecutive frames (80 ms at 100 Hz), and the predicted endpoint t<sup>ˆ</sup> is the last frame of this crossing. The tolerance window is one-sided: a detection is correct if it is no more than 50 ms early $( - 5 0 \mathrm { m s } ~ \leq ~ ( \hat { t } ~ -$ $t _ { \mathrm { g t } } )$ , with $t _ { \mathrm { g t } }$ the ground-truth boundary), and earlier triggers are false alarms, since a premature interruption costs more than a modest delay. We impose no late cutoff, as a late but correct endpoint is still correct, and report lateness through the detection latency $\delta = \hat { t } - t _ { \mathrm { g t } }$ . Because a one-sided window could reward a model that simply waits, we read F1 together with δ and credit a high F1 only when latency stays near natural turn gaps.

In the frame-level setting, each system runs over the whole utterance and yields one endpoint t<sup>ˆ</sup>: for our model, the first sustained threshold crossing; for the SmartTurn baselines, the first pause judged to be a turn end. However t<sup>ˆ</sup> is produced, it is scored against the ground-truth boundary with the same tolerance window, so all systems compare on equal terms. Two elements of this protocol are specific to our setting rather than standard practice. The tolerance window is one-sided, so an early fire is always a false alarm instead of being absorbed by a symmetric window, which is the stricter choice for a deployed agent. The utterance-level test is silence-matched, so trailing pause duration carries no label information and the decision has to rest on the speech before the pause.

We report five metrics here. Each endpoint is correct, a false alarm, or a miss, and F1 (the frame-level F1) is the harmonic mean of the resulting precision and recall. FA% is the fraction of utterances whose first trigger is premature; Miss% the fraction with no detection; FA/utt the mean premature triggers per utterance; Med. Lat. the median δ over correct detections.

A second, utterance-level setting scores the pooled classifiers: each receives one segment that ends in a pause and decides whether the turn is complete. Silence length alone could separate a true turn end from a mid-turn pause, so we match the two classes on trailing silence. A complete segment ends 150 ms after the true turn end and an incomplete one at a real mid-turn pause with the same 150 ms taken from earlier in the turn, so the decision must come from the speech before the pause. We score this setting with AUROC, which is threshold-free and imbalance-robust, so it needs no per-set operating point and cannot be inflated by favoring the majority class.

TABLE I: Dataset characteristics (pauses $\geq 1 0 0 \mathrm { m s } .$ , long pauses ≥400 ms, from the pipeline’s VAD segments).
<table><tr><td></td><td>Conv-10K</td><td>ST-10K</td><td>Int-5K</td><td>CASPER</td><td>ST-C</td></tr><tr><td>Mean dur. (s)</td><td>11.85</td><td>7.51</td><td>11.79</td><td>4.99</td><td>9.12</td></tr><tr><td>Median dur. (s)</td><td>10.90</td><td>7.04</td><td>10.80</td><td>3.33</td><td>9.76</td></tr><tr><td>Pauses / utt</td><td>8.37</td><td>5.13</td><td>8.47</td><td>6.60</td><td>6.00</td></tr><tr><td>Long pauses / utt</td><td>2.10</td><td>1.22</td><td>2.12</td><td>1.34</td><td>1.45</td></tr><tr><td>Silence ratio</td><td>0.687</td><td>0.333</td><td>0.688</td><td>0.600</td><td>0.433</td></tr></table>

## III. RESULTS AND DISCUSSION

We now ask whether the genre of the training data changes how a streaming detector finds the end of a turn. Every system is scored at its own best-F1 threshold on each dataset, under the same metrics and tolerance window. The numbers are therefore an upper bound rather than a deployable result, but the same tuning is applied to our model and to every baseline, so the comparison is fair and what it settles is the ranking between systems.

## A. The Two Genres Differ Mainly in Within-Turn Events

Before comparing models, we look at what actually separates the two training genres, because this is what a detector must learn and it predicts where the genres should diverge. Table I reports pause statistics for the training and evaluation sets.

The genres differ in one thing that matters, how many pauses fall inside a turn. Conv-10K carries far more withinturn structure than ST-10K, with roughly 60% more internal pauses per utterance, nearly double the silence ratio, and over 70% more long pauses. These are the ordinary marks of spontaneous speech, the hesitations, breath pauses, and midturn silences, and a detector has to learn to sit through them rather than read them as turn ends. Conv-10K utterances run longer too, so each example gives more context to hold across. The evaluation sets spread along the same axis. Int-5K matches Conv-10K, CASPER keeps the high silence ratio of spontaneous conversation despite its shorter utterances, and ST-Complete falls between the two training distributions. What changes across genres is not what a pause looks like but how many pauses fall within a turn, and that is the one distinction conversational training teaches and single-turn training does not.

## B. Conversational Training Generalises Across Domains

With the genres characterised, the training genre is the only thing we change. We train the same detector on Conv-10K and on ST-10K and test both on all four sets (Table II). Conversational training wins on every test set, in domain and out. Conv-10K gives higher F1 than ST-10K on all four sets, and the margin holds even on SLURP and ST-Complete, the two sets closest to the single-turn data. The advantage is therefore not a training-test domain match; it travels with the training genre. Where, then, does the single-turn model fall behind?

The gap comes from false alarms, not misses. The singleturn model finds the turn end in almost every utterance, as its low miss rate shows, but triggers too early. Trained on complete single-turn utterances with little exposure to midturn pauses, it reads the hesitations and silences of natural speech as turn ends. This is clearest on CASPER, the most spontaneous set, where the conversational model holds false alarms to 6.6% against 44.7% for the single-turn model. ST-10K reaches lower median latency only by firing at the first pause, leaving it two to five times as many premature triggers per utterance. In a live system a premature trigger interrupts the speaker, the costlier error, so cutting false alarms is what matters most in practice.

TABLE II: ConvLSTM trained on conversational (Conv-10K) vs. single-turn (ST-10K) data, scored frame-level on four test sets under the protocol of Sec. II-F. Best F1 per test set in bold. Med. Lat. is the median latency over correct detections.
<table><tr><td>Test</td><td>Train</td><td>F1</td><td>Miss%</td><td>FA%</td><td>FA/utt</td><td>Med. Lat.</td></tr><tr><td rowspan="2">Int-5K</td><td>Conv-10K</td><td>0.894</td><td>0.1</td><td>19.1</td><td>1.22</td><td>190ms</td></tr><tr><td>ST-10K</td><td>0.444</td><td>0.0</td><td>71.5</td><td>5.72</td><td>20ms</td></tr><tr><td rowspan="2">SLURP</td><td>Conv-10K</td><td>0.760</td><td>9.6</td><td>29.1</td><td>0.49</td><td>140ms</td></tr><tr><td>ST-10K</td><td>0.558</td><td>20.2</td><td>41.1</td><td>1.04</td><td>20ms</td></tr><tr><td rowspan="2">CASPER</td><td>Conv-10K</td><td>0.959</td><td>1.3</td><td>6.6</td><td>0.07</td><td>400ms</td></tr><tr><td>ST-10K</td><td>0.694</td><td>2.2</td><td>44.7</td><td>1.88</td><td>260ms</td></tr><tr><td rowspan="2">ST-Comp</td><td>Conv-10K</td><td>0.911</td><td>0.9</td><td>15.4</td><td>0.64</td><td>250ms</td></tr><tr><td>ST-10K</td><td>0.738</td><td>0.8</td><td>40.7</td><td>2.31</td><td>50ms</td></tr></table>

The benefit follows the genre, not the domain. CASPER also rules out a domain-similarity explanation. It is an independent open-source corpus recorded under different conditions from our call-centre data, so it is out of domain for both training sets. The conversational model still wins there by a wide margin, so the gain comes from the conversational genre and not from any resemblance to our own recordings, just as the pause statistics in Section III-A would predict.

The gaps are large and statistically reliable. A paired Mc-Nemar test [44] over the utterances where the models disagree gives $p < 1 0 ^ { - 7 0 }$ on all four sets, with the conversational model right far more often, and a 5,000-sample bootstrap [45] puts every 95% CI on the F1 gap above zero (+0.17 to +0.45). Nor is this an effect of per-model tuning: a shared threshold τ=0.7 leaves every gap at least as large (+0.49, +0.31, +0.32, +0.20 on Int-5K, SLURP, CASPER, ST-Complete).

## C. At the Utterance Level, the Benefit Holds Only on Conversational Speech

We established the benefit of conversational training for a frame-level detector. Most recent systems decide at the utterance level instead, classifying a segment that ends in a pause as complete or not, so the question is whether the benefit survives there. We repeat the comparison in that setting, scoring the two pooled classifiers by AUROC on the matched segments of Section II-F (Figure 2).

On conversational speech, the benefit holds for both encoders. Conversational training raises AUROC substantially on both conversational sets, by +0.21 (Int-5K) and +0.38 (CASPER) for Whisper-Tiny and by +0.37 and +0.29 for wav2vec2-base (Figure 2). On CASPER the single-turn head is near or below chance (AUROC 0.52 and 0.43), so conversational training accounts for almost all of the separation. The two encoders share no pretraining, and CASPER is out of domain for both, so the gain reflects genre rather than domain. On single-turn speech, the benefit is neutral or domainbound. On SLURP, out of domain for both training sets, the two heads are effectively tied (∆AUROC +0.01 and −0.03), so conversational training does not cost single-turn ability. On ST-Complete the single-turn head wins, but that set shares its source with the single-turn training data (SmartTurn v3), so its head is in domain while the conversational head is not; even there the conversational head stays well above chance (AUROC 0.78 and 0.73). Both single-turn sets conclude: the ST-Complete gap reflects domain match, not genre effect.

![](images/73f5c3e034f16830c441dbf2b4e20bf4048aad7d69bea42c1b74e34386c81c5d.jpg)  
Fig. 2: Utterance-level complete-or-not AUROC for two frozen encoders, conversational (Conv) vs single-turn (ST) training, across four test sets. Dotted line: chance.

The gaps are significant. A paired bootstrap over utterances places a 95% confidence interval on each AUROC gap. The four conversational gains are clear of zero, as are the two ST-Complete reversals, while the SLURP gaps sit at zero. So conversational training gives a real and large utterance-level advantage on conversational speech, stays neutral on out-ofdomain single-turn speech, and loses only where the singleturn model is in domain.

Why the test genre matters. This is why the effect has gone unnoticed. Utterance-level classifiers are usually benchmarked on single-turn or command data [21], [22], exactly the regime where the conversational benefit does not surface. It is not that conversational training fails there. A single decision on a clean ending simply does not exercise the within-turn events the two genres differ on. This is consistent with the framelevel result, where the conversational model wins even on the single-turn test sets, because that metric scores the whole streaming recording and rewards holding back through every mid-turn pause (Figure 3), whereas the utterance-level decision sees only one fixed ending. The practical lesson is that the utterance-level benchmark itself decides whether the traininggenre effect is visible, so evaluating only on single-turn or command speech leaves it untested.

TABLE III: Cross-dataset results at each model’s best threshold. Bold = highest F1. Latency in ms; parentheses give the utterance count; FA/utt (Section II-F) is omitted for space.
<table><tr><td></td><td colspan="4">Int-5K</td><td colspan="4">CASPER</td><td colspan="4">ST-Complete</td></tr><tr><td>Model</td><td>F1</td><td>FA%</td><td>Miss%</td><td>Med. Lat.</td><td>F1</td><td>FA%</td><td>Miss%</td><td>Med. Lat.</td><td>F1</td><td>FA%</td><td>Miss%</td><td>Med. Lat.</td></tr><tr><td>ConvLSTM (ours)</td><td>.894</td><td>19.1</td><td>0.1</td><td>190 (4040)</td><td>.959</td><td>6.6</td><td>1.3</td><td>400 (2301)</td><td>.911</td><td>15.4</td><td>0.9</td><td>250 (2092)</td></tr><tr><td>Silero VAD</td><td>.781</td><td>33.2</td><td>2.7</td><td>50 (3205)</td><td>.734</td><td>29.6</td><td>12.4</td><td>12 (1448)</td><td>.673</td><td>13.2</td><td>36.1</td><td>110(1268)</td></tr><tr><td>WebRTC agg3</td><td>.476</td><td>66.7</td><td>2.1</td><td>20(1560)</td><td>.512</td><td>52.8</td><td>12.8</td><td>0(861)</td><td>.420</td><td>41.9</td><td>31.5</td><td>90(665)</td></tr><tr><td>SmartTurn v3</td><td>.523</td><td>28.9</td><td>35.7</td><td>48(1770)</td><td>.390</td><td>22.0</td><td>53.7</td><td>8(606)</td><td>.652</td><td>50.6</td><td>12.3</td><td>150 (928)</td></tr><tr><td>SmartTurn FT</td><td>.525</td><td>64.0</td><td>0.4</td><td>50(1780)</td><td>.644</td><td>44.6</td><td>7.9</td><td>4(1188)</td><td>.694</td><td>34.6</td><td>1.0</td><td>152(1610)</td></tr></table>

![](images/3587dc3e525c9a9c597931747705903418be87d5babebd0564ef5e271f77193c.jpg)  
Fig. 3: Frame-level P(EoT) on a CASPER turn, with the speech waveform overlaid and the true end dashed.

D. Outperforming Open-Source Baselines with Fewer Parameters

The ConvLSTM leads on all three sets at a fraction of SmartTurn’s size. With the genre effect established at both granularities, we benchmark it against open-source systems on the three test sets that carry conversational or multi-turn structure (Table III). SLURP, a single-command set, appears only for our own model, in Table II. It leads in F1 on all three at 1.8M parameters, about 4.4× fewer than the 8M-parameter SmartTurn, and it is the only system to hold false alarms, misses, and latency in check at once.

The two silence VADs fail exactly as their design predicts. On CASPER the flowing dialogue leaves few clean silence gaps, so they miss most turns, while on ST-Complete the frequent mid-speech pauses drive their false alarms up. They reach low latency only by firing at the first silence, so their speed comes at the cost of reliability. The ConvLSTM instead waits about 400 ms on CASPER, the price of holding through mid-turn pauses rather than stopping at them, which sits near the upper end of natural between-turn gaps [2], [4] while keeping both false alarms and misses low.

Reading the speech does not by itself beat the silence rule. Trained on single-turn data, SmartTurn v3 transfers poorly to these conversational endings and trails even Silero on all three sets, missing most turns (Miss% up to 53.7 on CASPER), while fine-tuning only trades the misses for over-firing (FA% up to 64 on Int-5K). A learned endpoint detector thus rises or falls with its training genre, the factor we isolate throughout.

## E. Limitations

A few limitations frame our conclusions. The conversational corpus is English-only telephony while the single-turn data is multilingual and partly synthetic [31], so the advantage may not hold across languages, accents, or styles, and part of the gap could reflect synthetic-speech artifacts rather than genre. Labels for SLURP and ST-Complete are VAD-derived rather than human-annotated, which may favor some model types. The frozen Whisper and wav2vec2 comparators receive 8 kHz audio upsampled to 16 kHz, lowering their absolute scores, though this applies to both genres and leaves the Convminus-ST comparison intact. The two training sets differ in genre and source domain, so part of the gap could be domain rather than genre, though the out-of-domain CASPER result argues against this. Our internal sets are private, but the effect reproduces on public data, since CASPER, the single-turn data, and both frozen comparators are all public. Finally, we study the two genres for which abundant real speech can be reliably labelled today; extending to others such as multi-turn humanmachine dialogue awaits suitable corpora [30].

## IV. CONCLUSION

This paper treats speech genre as a primary experimental variable for end-of-turn detection. Across four test sets, a streaming ConvLSTM trained on conversational humanhuman data steadily beats the same model trained on singleturn human-machine data, with the gains coming from fewer false alarms on the mid-turn pauses and hesitations that single-turn corpora rarely contain. The benefit holds at both decision granularities: a frame-level model captures it directly, and an utterance-level classifier captures it too, but only on conversational test speech. On single-turn benchmarks the two genres tie out of domain and the single-turn model leads only on its own source domain, so such evaluations leave the effect untested rather than contradict it. Against open-source baselines, the proposed model attains the highest F1 on all test sets with far fewer parameters than the learned SmartTurn baseline. These results suggest that who the speaker is talking to shapes the training signal as much as model capacity or feature choice, and that speech genre deserves attention as a first-class design decision for EoT detection. Natural next steps are multilingual settings and mixtures of conversational and single-turn data.

[1] A. Raux and M. Eskenazi, “Flexible turn-taking for spoken dialog systems,” in Proc. Interspeech, 2008, pp. 1329–1332.

[2] S. C. Levinson and F. Torreira, “Timing in turn-taking and its implications for processing models of language,” Frontiers in Psychology, vol. 6, p. 731, 2015.

[3] D. Schlangen and G. Skantze, “A general model of incremental dialogue processing,” in Proc. SIGDIAL, 2011, pp. 1–10.

[4] M. Heldner and J. Edlund, “Pauses, gaps and overlaps in conversations,” Journal of Phonetics, vol. 38, no. 4, pp. 555–568, 2010.

[5] L. Ferrer, E. Shriberg, and A. Stolcke, “Is the speaker done yet? faster and more accurate end-of-utterance detection using prosody,” in Proceedings of the International Conference on Spoken Language Processing (ICSLP), Denver, Colorado, USA, 2002, pp. 2061–2064.

[6] ——, “A prosody-based approach to end-of-utterance detection that does not require speech recognition,” in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), vol. 1, 2003, pp. I–605–I–608.

[7] G. Skantze, “Towards a general, continuous model of turn-taking in spoken dialogue using lstm recurrent neural networks,” in Proceedings of SIGDIAL, Saarbrucken, Germany, 2017, pp. 220–230.¨

[8] M. Roddy, A. Eshghi, and C. Howes, “Investigating speech features for continuous turn-taking prediction using LSTMs,” in Proceedings of Interspeech, 2018.

[9] Z. Aldeneh, D. Dimitriadis, and E. Mower Provost, “Improving end-ofturn detection in spoken dialogues by detecting speaker intentions as a secondary task,” in Proceedings of ICASSP, 2018.

[10] E. Ekstedt and G. Skantze, “Voice Activity Projection: Self-supervised turn-taking prediction in dialogue,” in Proc. Interspeech, 2022, pp. 5125–5129.

[11] ——, “Turngpt: a transformer-based language model for predicting turn-taking in spoken dialog,” in Findings of the Association for Computational Linguistics: EMNLP 2020. Online: Association for Computational Linguistics, Nov. 2020, pp. 2981–2990. [Online]. Available: https://aclanthology.org/2020.findings-emnlp.268

[12] H. Ok, S. Yoo, and J. Lee, “Speculative end-turn detector for efficient speech chatbot assistant,” arXiv preprint arXiv:2503.23439, 2025. [Online]. Available: https://arxiv.org/abs/2503.23439

[13] A. Defossez, L. Mazar´ e, M. Orsini, A. Royer, P. P´ erez, H. J´ egou,´ E. Grave, and N. Zeghidour, “Moshi: a speech-text foundation model for real-time dialogue,” arXiv preprint arXiv:2410.00037, 2024. [Online]. Available: https://arxiv.org/abs/2410.00037

[14] G.-T. Lin et al., “Full-duplex-bench: A benchmark to evaluate fullduplex spoken dialogue models on turn-taking capabilities,” arXiv preprint arXiv:2503.04721, 2025.

[15] J. Lee and J. Deng, “Online multimodal end-of-turn prediction for threeparty conversations,” in Proceedings of ACM ICMI, 2024.

[16] G. Castillo-Lopez, G. de Chalendar, and N. Semmar, “A survey of´ recent advances on turn-taking modeling in spoken dialogue systems,” in Proceedings of the 15th International Workshop on Spoken Dialogue Systems Technology (IWSDS). Bilbao, Spain: Association for Computational Linguistics, 2025, pp. 254–271.

[17] S. O. Russell and N. Harte, “Visual cues enhance predictive turn-taking for two-party human interaction,” in Findings of the Association for Computational Linguistics: ACL 2025. Bangkok, Thailand: Association for Computational Linguistics, 2025. [Online]. Available: https://aclanthology.org/2025.findings-acl.12

[18] K. Inoue, B. Jiang, E. Ekstedt, T. Kawahara, and G. Skantze, “Multilingual turn-taking prediction using voice activity projection,” in Proceedings of LREC-COLING, 2024.

[19] M. Elmers, K. Inoue, D. Lala, and T. Kawahara, “Triadic multiparty voice activity projection for turn-taking in spoken dialogue systems,” arXiv preprint arXiv:2507.07518, 2025. [Online]. Available: https://arxiv.org/abs/2507.07518

[20] C. Cieri, D. Miller, and K. Walker, “The fisher corpus: A resource for the next generations of speech-to-text,” in Proc. LREC, 2004.

[21] Pipecat AI, “Smartturn v3: Semantic turn detection,” https://github.com/ pipecat-ai/smart-turn, 2025.

[22] E. Bastianelli, A. Vanzo, P. Swietojanski, and V. Rieser, “SLURP: A spoken language understanding resource package,” in Proc. EMNLP, 2020.

[23] P. AI, “Smart turn v2: Faster inference and 13 new languages for voice ai,” https://www.daily.co/blog/ smart-turn-v2-faster-inference-and-13-new-languages-for-voice-ai/, 2025, accessed 2026-03-04.

[24] F. AI, “Ultravad: A context-aware, audio-native endpointing model,” https://huggingface.co/fixie-ai/ultraVAD, 2025, accessed 2026-03-04.

[25] A. Gravano and J. Hirschberg, “Turn-taking cues in task-oriented dialogue,” Journal of Pragmatics, vol. 43, no. 11, pp. 2982–2997, 2011.

[26] A. Raux, “Flexible turn-taking for spoken dialog systems,” Ph.D. dissertation, Carnegie Mellon University, Pittsburgh, PA, USA, 2008.

[27] J. Edlund, G. Ambrazaitis et al., “The interplay of vocal effort and hesitation disfluencies,” in Proceedings of Speech Prosody, 2017.

[28] C. Darves and S. Oviatt, “Adaptation of users’ spoken dialogue patterns in a conversational interface,” in Proceedings of ICSLP, Denver, CO, USA, 2002, pp. 1869–1872.

[29] S. Oviatt, R. Coulston, and C. Darves, “Toward adaptive conversational interfaces: Modeling speech convergence with animated personas,” Cognitive Science, vol. 28, no. 3, pp. 329–354, 2004.

[30] Y. Lin, Y. Zheng, M. Zeng, and W. Shi, “Predicting turn-taking and backchannel in human-machine conversations using linguistic, acoustic, and visual signals,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Vienna, Austria: Association for Computational Linguistics, 2025, pp. 15 310–15 322.

[31] Pipecat AI, “Announcing Smart Turn v3, with CPU inference in just 12ms,” https://www.daily.co/blog/ announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/, 2025, accessed 2026-03-04.

[32] International Telecommunication Union, “ITU-T recommendation G.712: Transmission performance characteristics of pulse code modulation channels,” ITU-T, Tech. Rep., 2001.

[33] J. Valin and R. Lefebvre, “Bandwidth extension of narrowband speech for low bit-rate wideband coding,” in Proceedings ofthe IEEE Workshop on Speech Coding, 2000, pp. 130–132.

[34] L. R. Rabiner and R. W. Schafer, Theory and Applications of Digital Speech Processing. Pearson, 2011.

[35] J. Sohn, N. S. Kim, and W. Sung, “A statistical model-based voice activity detection,” IEEE Signal Processing Letters, vol. 6, no. 1, pp. 1–3, 1999.

[36] K. Hlavnickovˇ a, M.´ Zelezn<sup>ˇ</sup> y´ et al., “CASPER: Corpus and annotation of spontaneous speech for speech recognition,” in Proc. Interspeech, 2017, pp. 2646–2650.

[37] D. Hendrycks and K. Gimpel, “Gaussian error linear units (gelus),” arXiv preprint arXiv:1606.08415, 2016. [Online]. Available: https: //arxiv.org/abs/1606.08415

[38] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in Proc. International Conference on Learning Representations (ICLR), 2015. [Online]. Available: https://arxiv.org/abs/1412.6980

[39] I. Loshchilov and F. Hutter, “SGDR: Stochastic gradient descent with warm restarts,” in Proc. International Conference on Learning Representations (ICLR), 2017. [Online]. Available: https://arxiv.org/abs/ 1608.03983

[40] WebRTC Project, “Webrtc: Real-time communication for the web,” https://webrtc.org/, 2011, accessed: 2026-03-04.

[41] Silero Team, “Silero VAD: Pre-trained enterprise-grade voice activity detector,” https://github.com/snakers4/silero-vad, 2021.

[42] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proceedings of the 40th International Conference on Machine Learning (ICML), ser. PMLR, vol. 202, 2023, pp. 28 492–28 518.

[43] A. Baevski, H. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” in Proc. NeurIPS, 2020.

[44] T. G. Dietterich, “Approximate statistical tests for comparing supervised classification learning algorithms,” Neural Computation, vol. 10, no. 7, pp. 1895–1923, 1998.

[45] B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. Chapman and Hall/CRC, 1993.