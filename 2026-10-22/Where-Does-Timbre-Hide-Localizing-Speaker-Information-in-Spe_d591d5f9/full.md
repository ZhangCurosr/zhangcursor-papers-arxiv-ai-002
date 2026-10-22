# Where Does Timbre Hide? Localizing Speaker Information in Speech Signals

Yiquan Zhou , Yixuan Wang , Xiaozhe Qi , Xin Gao , and Shihao Li

Abstract—Timbre similarity is a key evaluation target for generative speech systems, including voice conversion, speakercloning text-to-speech, and zero-shot speech models. When the generated voice does not match the target speaker, the problem often reflects imperfect separation of speaker-related timbre from content, making it important to identify where timbre information resides and which representations still carry it. Using speaker identity as a measurable proxy, we probe frequency bands, phase/magnitude structure, WORLD source/filter decomposition, voiced/unvoiced regions, self-supervised learning (SSL) layers, EnCodec/HuBERT tokens, and a 45-listener test across Mandarin, English, and Japanese speech/singing data. Speaker information is distributed: 1–8 kHz magnitude and vocal-tract spectral-envelope structure dominate perception, above-8 kHz bands remain machine-discriminative, shallow SSL layers expose compact speaker information, and token leakage depends on tokenizer/layer choice. These findings indicate where generation systems should condition on or transform speaker information, and where content, codec, and token pipelines should audit residual speaker leakage.

Index Terms—speaker identity, timbre, speaker verification, voice conversion, self-supervised speech representations, listening test

## I. INTRODUCTION

IMBRE similarity is now a central evaluation target for generative speech systems. Voice conversion (VC), speaker-cloning text-to-speech (TTS), and some zero-shot large speech models must preserve linguistic content while controlling the target speaker’s voice [1]–[4]. This requirement makes speaker-independent content modeling a practical engineering concern rather than only a design choice in representation learning. If speaker-related information remains in a content, codec, or self-supervised learning (SSL) representation, it can leak into generation; if the information needed for the target speaker is not injected or transformed at the right place, the generated speech may remain intelligible but sound like the wrong speaker.

It is therefore important to know where speaker-related timbre information resides and how it is coupled with content and other acoustic factors. Timbre itself is perceptual and difficult to measure directly at scale, so we use speaker identity as a measurable proxy for speaker-related timbre while keeping machine discriminability separate from human similarity judgments. Classical source-filter theory and perceptual studies point to vocal-tract and spectral-envelope structure [5]–[8], while modern speaker verification and SSL models such as ECAPA-TDNN and WavLM provide strong but less interpretable speaker information [9], [10]. Earlier subband speaker-recognition studies also found that low- and midfrequency bands carry substantial speaker information [11]– [13], but they predate high-sample-rate corpora, neural codecs, tokenized speech language models, and recent SSL probing work [14]–[17].

This paper asks where speaker-related timbre information is preserved, transformed, or leaked in both speech signals and learned representations. We systematically compare complementary probes: frequency filtering, phase/magnitude randomization, WORLD source/filter swaps, voiced/unvoiced extraction, WavLM and ECAPA speaker verification, mel classifiers, SSL-layer and PCA probes, EnCodec and HuBERT token histograms, cross-style robustness checks, and a 45- listener similarity test. These probes allow the same speakerrelated information to be viewed from signal, representation, and perceptual sides.

The experiments lead to three findings. First, the combined probes give a structured view of where speaker-related timbre is concentrated. Frequency-band manipulations and listener ratings place the perceptually important speaker information mainly in the 1–8 kHz region, rather than in very high bands. Phase/magnitude edits show that this information is carried mainly by spectral magnitude, not phase. WORLD source/filter swaps further identify the vocal-tract spectral envelope as a stronger carrier than fundamental frequency (F0) alone. Second, representation probes show that speaker-related information remains accessible in shallow SSL layers and can survive in some token streams, so content and codec representations should be audited rather than assumed speaker-free. Third, speaker information measured by a classifier is not the same as perceived timbre similarity. High-frequency bands still contain speaker information that classifiers can use, but this information does not produce strong same-speaker similarity for listeners. This suggests that high-band reconstruction may improve acoustic detail without restoring the perceived speaker similarity that listeners rely on. Together, these findings can guide future VC, TTS, anonymization, codec, and speech language model work in deciding which components need speaker conditioning, transformation, suppression, or leakage checks.

TABLE I  
DATA USED IN THE MAIN CHECKS.
<table><tr><td>Experiment group Data scale</td><td></td><td>What it checks</td></tr><tr><td>AISHELL-1</td><td>20 spk.</td><td>lowpass/highpass, SSL, tokens</td></tr><tr><td>AISHELL-3</td><td>98 spk.</td><td>high-sample-rate bands, SSL/token</td></tr><tr><td>VCTK</td><td>100 spk.</td><td>validation, listening 48 kHz high-band and unvoiced vali-</td></tr><tr><td>ESD/CREMA-D</td><td>20/91 spk.</td><td>dation emotion robustness</td></tr><tr><td>NUS/JVS</td><td>12/100 spk.</td><td>speech-singing robustness</td></tr><tr><td>Self-rec.</td><td>4 spk.</td><td>controlled band-swap and voiced/unvoiced pilot</td></tr></table>

## II. EXPERIMENTAL SETUP

This section defines the experimental setting used in the paper. It contains two parts: the datasets and common evaluation setup, and the design of each experiment.

## A. Datasets and Common Setup

The protocol uses multiple datasets because different probes require different data characteristics (e.g., sampling rate, language, style labels). AISHELL-1 and AISHELL-3 provide Mandarin multi-speaker speech for frequency, SSL, token, and listening checks [18], [19]. VCTK provides 48 kHz English speech for high-band and voiced/unvoiced tests [20]. ESD and CREMA-D test whether speaker information survives emotion changes [21], [22]. NUS-48E and JVS/JVS-MuSiC test speech–singing transfer [23]–[25]. We also use a self-recorded 48 kHz Chinese–English corpus for controlled band-swap and voiced/unvoiced pilot experiments. Table I summarizes the data used by each experiment group.

All experiments use speaker identity as the measurable target, but the metric depends on the probe. WavLM-basesv is the main speaker verification (SV) backend [10]. For signal edits that may depend on the backend, we repeat the phase/magnitude and source/filter tests with a Speech-Brain ECAPA-TDNN VoxCeleb model [9]. Embedding-based tests use cosine similarity between speaker embeddings, and report intra-speaker similarity, inter-speaker similarity, and discriminability, defined as the same-speaker score minus the different-speaker score. Bandwidth curves also report equal error rate (EER). Closed-set feature and token probes report speaker-classification accuracy. The listening test reports mean voice-timbre similarity ratings (i.e., perceived same-speaker similarity) on a 1–5 scale. Because WavLM-base-sv requires 16 kHz input, spectral content above 8 kHz is discarded before embedding extraction. High-sample-rate spectra are therefore evaluated with mel multilayer perceptron (MLP) probes in the frequency-band experiment. Fig. 1 summarizes the full protocol.

## B. Experiment Design

Frequency-band experiment: This experiment tests which spectral regions preserve speaker information. For lowpass and highpass analysis, we apply Butterworth filters to each utterance, extract WavLM-base-sv embeddings, and compute speaker discriminability and EER. Lowpass curves show how speaker information changes as more bandwidth is retained from the bottom up. Highpass curves show how speaker information changes when low and mid frequencies are removed. For bands above 8 kHz, we cannot rely on WavLM-basesv because its 16 kHz input removes this frequency range. We therefore keep AISHELL-3 and VCTK at their original sampling rates, isolate fixed frequency bands, extract log-mel summaries from each band, and train a 2-layer closed-set MLP speaker classifier. We also check shuffled labels and duration/energy baselines to make sure the classifier is not succeeding for trivial reasons. Mel-cosine separability (the cosine-distance gap between same- and different-speaker mel features) provides an additional classifier-free check. To compare machine-detectable information with perception, we also construct AISHELL-3 paired trials from full-band, lowpass, bandpass, highpass, and different-speaker conditions. Listeners rate each pair on a 1–5 voice-timbre similarity scale. The main metrics are discriminability and EER for the SV curves, speaker-classification accuracy for the mel MLP checks, and mean listener rating for the listening conditions.

Phase/magnitude experiment: This experiment tests whether speaker information is carried mainly by spectral magnitude or phase. For each utterance, we compute a shorttime Fourier transform (STFT) and synthesize two variants. The magnitude-kept variant keeps the original magnitude and randomizes phase. The phase-kept variant keeps the original phase and randomizes magnitude. We then compare the speaker discriminability of each variant against that of the unmodified signal. This manipulation directly tests whether speaker information is carried by the spectral envelope or the phase structure.

Source/filter experiment: This experiment tests whether speaker information is better explained by source pitch (F0) or vocal-tract filtering. Using WORLD, we decompose each utterance into fundamental frequency (F0), spectral envelope, and aperiodicity [26]. We then create cross-speaker resynthesized signals by keeping one speaker’s F0 while replacing the spectral envelope, or by keeping the spectral envelope while replacing F0. We judge the two conditions by same-speaker cosine similarity and retention (the fraction of the original same-speaker similarity preserved). If the filter-kept signal preserves more speaker similarity than the F0-kept signal, the retained information is more closely linked to vocal-tract spectral-envelope structure than to pitch alone. We repeat the phase/magnitude and source/filter tests with ECAPA-TDNN to check that the pattern is not specific to WavLM-base-sv.

Self-supervised learning (SSL) representation experiment: This experiment tests where speaker information is exposed in continuous speech features. We extract layer-wise features from WavLM, HuBERT, Wav2Vec 2.0, and Whisperbase, mean-pool each utterance, and train lightweight speaker classifiers. We also apply principal component analysis (PCA) before classification to test whether speaker information re mains accessible after dimensionality reduction. This shows which layers are most speaker-discriminative and whether the information is compact enough to leak through content features when they are reused in downstream systems. The main metric is closed-set speaker-classification accuracy.

![](images/70413863061eab2b7bdbc67ea853914290a5c5636a3db28dceb46eac7484b9b7.jpg)  
Fig. 1. Speaker-information localization protocol. The study combines signal edits, model-based and model-free measurements, representation probes, and human similarity ratings, so speaker information measured by models can be kept separate from perceived speaker-related timbre information.

Token experiment: This experiment tests whether discrete speech units retain speaker information. For EnCodec, we compute histograms from each residual vector quantization (RVQ) layer and from cumulative RVQ groups. For HuBERT, we compute k-means token histograms from multiple SSL layers and codebook sizes, and add a bigram check to capture limited sequential dependencies. These probes test whether speaker leakage depends on tokenizer, layer, codebook size, and RVQ depth. The main metric is closed-set speakerclassification accuracy.

Robustness experiments: These experiments test whether speaker information survives common style and phonation changes. Emotion tests compare same-speaker crossemotion similarity with different-speaker baselines, and include a CREMA-D leave-one-emotion-out validation. Speech– singing tests compare same-speaker cross-modal similarity and train speech-based classifiers that are evaluated on singing. Voiced/unvoiced tests separate active speech into voiced and unvoiced regions and evaluate whether speaker information survives outside pitch-bearing regions. These checks prevent the main claims from relying only on neutral read speech. The main metrics are cosine similarity gaps and speakerclassification accuracy.

## III. RESULTS

## A. Frequency-Band Experiment Results

The frequency-band results show that the strongest speaker information lies in the low/mid spectrum, while higher bands retain weaker but measurable information. In the WavLMbase-sv lowpass curves, discriminability increases as more bandwidth is retained and nearly reaches the full-band score by 7.5 kHz. The highpass curves give the complementary pattern: performance remains stable when only sub-1 kHz content is removed, but drops sharply once the cutoff exceeds 1–2 kHz. These two curves indicate that speaker information is not carried by sub-1 kHz content alone, but by a broader low/midfrequency range. Because WavLM-base-sv is limited by its 16 kHz input, the WavLM curves cannot determine whether bands above 8 kHz still contain speaker information.

Table II fills this gap with the mel MLP checks. Both highsample-rate datasets show the same trend: accuracy decreases with frequency. The 1–8 kHz bands are strongest: VCTK reaches 92.1% at 1–4 kHz and 89.3% at 4–8 kHz, while AISHELL-3 reaches 82.5% and 72.4% in the same bands. Bands above 8 kHz remain well above shuffled labels and duration/energy baselines, so they do contain speaker information that classifiers can use. However, the scores decline with frequency: VCTK falls to 50.1% in the 20–24 kHz band, and AISHELL-3 falls to 42.8% in the 16–20 kHz band.

TABLE II  
BAND MLP CHECK. VALUES ARE SPEAKER-CLASSIFICATION ACCURACY(%); SHUFFLED LABELS STAY NEAR CHANCE AND DURATION/ENERGYBASELINES ARE MUCH LOWER.
<table><tr><td>Dataset/Band</td><td>1-4 4-8 8-12</td><td></td><td>High band</td></tr><tr><td>VCTK acc.</td><td>92.1 89.3</td><td>88.1</td><td>50.1 (20–24)</td></tr><tr><td>VCTK shuf.</td><td>1.2 1.5</td><td>1.1</td><td>0.8</td></tr><tr><td>VCTK dur./eng.</td><td>4.5 5.8</td><td>5.4</td><td>5.4</td></tr><tr><td>AISHELL-3 acc.</td><td>82.5 72.4</td><td>55.0</td><td>42.8 (16–20)</td></tr><tr><td>AISHELL-3 shuf.</td><td>0.9 1.0</td><td>1.0</td><td>0.9</td></tr><tr><td>AISHELL-3 dur./eng.</td><td>6.4 5.6</td><td>5.3</td><td>6.0</td></tr></table>

(a) Lowpass EER  
![](images/cd6462979d156975b5852a824581153cb7fc8e50cbc174ee883fa7be8ae69efb.jpg)

(b) VCTK native-rate MLP (100 spk)  
![](images/6d0ceae1f64c7dfe816d567b3f423306eb23ba18b3ad2ba32cbcb9710c1e1c39.jpg)  
Fig. 2. Bandwidth comparison. SV lowpass curves saturate near 4–8 kHz, while mel probes show weaker but measurable above-8 kHz information.

Mel-cosine separability confirms the same frequency trend, with AISHELL-3 spectral-shape features peaking at 4–8 kHz $( d ^ { \prime } = 1 . 8 6 )$ and decreasing toward the highest tested band $( d ^ { \prime } = 0 . 9 7 )$ . Fig. 2 summarizes the contrast between low/midband dominance and weaker high-band information.

Table III and Fig. 3 report the listening-test results for the same frequency contrast.

The listening results show that classifier-usable highfrequency information is not sufficient for perceived samespeaker timbre similarity. Same-speaker full-band pairs are rated much higher than different-speaker full-band controls (4.44 vs. 2.29, $p = 3 . 6 \times 1 0 ^ { - 2 1 } )$ . Low/mid-frequency samespeaker conditions (Same ${ < } 4 \ \mathrm { \ k H z } ,$ Same 1–4 kHz, Same ${ < } 8$ kHz) are rated much higher than high-frequency-only conditions (Same >8 kHz, Same 12–20 kHz), with a mean paired difference of 2.57 points $( p = 1 . 7 \times 1 0 ^ { - 3 8 } )$ . The $< 8$ kHz versus $> 8$ kHz contrast is especially large (4.45 vs. 1.24, $p = 1 . 3 \times 1 0 ^ { - 3 9 } )$ . Together with the MLP results, this shows that high-frequency bands contain detectable speaker information, but do not provide strong same-speaker similarity for listeners under the present manipulations.

TABLE III  
HUMAN SIMILARITY RATINGS (45 LISTENERS, 1–5 SCALE).
<table><tr><td>Condition</td><td> $\mathrm { M e a n } \pm \mathrm { S D }$ </td><td>95% CI</td></tr><tr><td>Different full</td><td> $2 . 2 9 \pm 0 . 5 3$ </td><td>[2.13, 2.45]</td></tr><tr><td>Same 12–20 kHz</td><td> $1 . 0 7 \pm 0 . 2 8$ </td><td>[0.99, 1.16]</td></tr><tr><td>Same &gt;8 kHz</td><td> $1 . 2 4 \pm 0 . 2 9$ </td><td>[1.15, 1.32]</td></tr><tr><td>Same &lt;4 kHz</td><td> $3 . 8 1 \pm 0 . 5 8$ </td><td>[3.64, 3.99]</td></tr><tr><td>Same 1-4 kHz</td><td> $2 . 9 2 \pm 0 . 6 9$ </td><td>[2.71, 3.13]</td></tr><tr><td>Same &lt;8 kHz</td><td> ${ \bf 4 . 4 5 \pm 0 . 3 5 }$ </td><td>[4.35, 4.56]</td></tr><tr><td>Same full</td><td> $4 . 4 4 \pm 0 . 3 6$ </td><td>[4.34, 4.55]</td></tr></table>

Listening Test (E11, 45 listeners)  
![](images/7c62e34e4601774029d59e07108ab5f4813e5b84991a92beab2bb81d2ba6b70e.jpg)  
Fig. 3. Listening test. Low/mid-frequency same-speaker pairs receive high similarity ratings; high-frequency-only pairs receive low ratings.

## B. Phase/Magnitude Experiment Results

Table IV and Fig. 4 summarize the phase/magnitude and source/filter results across the two SV backends. For the phase/magnitude experiment, magnitude preserves much more speaker information than phase. With WavLM, keeping the original magnitude spectrum while randomizing phase preserves 81.6% of the original discriminability. The phasekept variant loses speaker discriminability entirely: WavLM discriminability drops to -0.199 and ECAPA to 0.003, both effectively at or below chance.

## C. Source/Filter Experiment Results

The source/filter rows in Table IV and Fig. 4 confirm the same pattern in a source-filter decomposition: spectralenvelope information dominates over F0. Keeping the vocaltract spectral envelope preserves 92.4% of full-signal samespeaker similarity with WavLM, compared with 82.0% when only F0 is retained. With ECAPA, the filter-kept condition also retains much more speaker similarity than the F0-kept condition (79.3% vs. 26.2%). Together with the phase/magnitude result in Table IV, this links the main speaker-related information to spectral-envelope structure rather than to phase or F0 alone.

TABLE IV  
PHASE/MAGNITUDE AND SOURCE/FILTER SUMMARY ACROSS TWO SV BACKENDS. RET. DENOTES RETENTION RELATIVE TO THE FULL OR ORIGINAL CONDITION.
<table><tr><td>Backend Condition</td><td>Score</td><td>Ret.</td></tr><tr><td>WavLM magnitude kept WavLM phase kept WavLM filter kept WavLM F0 kept</td><td>disc. 0.195 disc. -0.199 sim. 0.906 sim. 0.804</td><td>81.6% 92.4% 82.0%</td></tr><tr><td>ECAPA ECAPA phase kept</td><td>magnitude kept disc. 0.284</td><td>77.1%</td></tr><tr><td>ECAPA filter kept</td><td>disc. 0.003 sim. 0.601</td><td>0.7% 79.3%</td></tr><tr><td>ECAPA F0 kept</td><td>sim. 0.199</td><td>26.2%</td></tr></table>

![](images/5de31171295831eeb63ae600848327a603ea5b71b05262535dac4a798818e9db.jpg)

![](images/164cfb12044464faabec612776d9ff20caefdecc9fdee84c2373bf9642e8f077.jpg)  
Fig. 4. Second-backend validation. WavLM and ECAPA both favor magnitude/filter information over phase/F0-dominant information.

## D. SSL Representation Experiment Results

Table V summarizes the model-wise layer and PCA results. The representation results show that speaker information is strongest in early to mid layers. On AISHELL-1, WavLM, HuBERT, Wav2Vec 2.0, and Whisper speaker probes all peak in the first few layers (L1–L5), well before the final layer. On AISHELL-3 (not tabulated), WavLM reaches 96.5–97.7% in layers L0–L4 but falls to 75.7–76.5% in L9–L12. PCA results show the same depth effect: AISHELL-3 WavLM L0 remains at 95.8% with 32 dimensions, while L12 falls to 62.7%.

## E. Token Experiment Results

Table VI summarizes the token and codec results. The results are tokenizer- and layer-dependent. With the order-free histogram probe used here, HuBERT k-means tokens preserve more speaker information than the tested 24 kHz EnCodec RVQ histograms. This is not a final ranking of all codec representations. These results show that tokenization objective and layer choice strongly affect how much speaker information remains available.

TABLE V  
SSL REPRESENTATION RESULTS ON AISHELL-1. PCA COLUMNS USE EACH MODEL’S PEAK LAYER.
<table><tr><td>Model</td><td>Peak</td><td>Final</td><td>32d 16d 8d</td></tr><tr><td>WavLM</td><td>L1 99.8</td><td>L12 294.8</td><td>99.8 98.3 85.3</td></tr><tr><td>HuBERT</td><td>L5 99.5 L12</td><td>95.3 99.0</td><td>93.8 69.2</td></tr><tr><td>Wav2Vec2</td><td>L1 99.3</td><td>L12 95.0</td><td>99.0 95.8 82.2</td></tr><tr><td>Whisper</td><td>L3 97.5</td><td>L6 92.0</td><td>96.5 90.2 74.5</td></tr></table>

TABLE VI

TOKEN AND CODEC RESULTS. HUBERT AND ENCODEC VALUES ARE SPEAKER-CLASSIFICATION ACCURACIES (%).
<table><tr><td colspan="3">HuBERT k-means tokens</td></tr><tr><td>Layer</td><td>k=50k=100 k=200 k=500k=1000</td><td></td></tr><tr><td>L1</td><td>60.2 68.5 78.0</td><td>89.0 94.0</td></tr><tr><td>L3</td><td>46.5 65.7 70.7</td><td>84.0 92.7</td></tr><tr><td>L6</td><td>43.0 57.8 75.3</td><td>87.3 96.2</td></tr><tr><td>L9</td><td>37.7 52.0 64.7</td><td>75.2 88.5</td></tr><tr><td>L12</td><td>40.7 57.2 68.3</td><td>74.2 84.2</td></tr><tr><td colspan="3">EnCodec RVQ tokens</td></tr><tr><td>Layer</td><td>Acc. Cum.</td><td colspan="2">Cum. Acc.</td></tr><tr><td>RVQ-0</td><td>53.5</td><td>0</td><td>53.5</td></tr><tr><td>RVQ-1</td><td>65.0</td><td>0-1</td><td>63.7</td></tr><tr><td>RVQ-2</td><td>67.2</td><td>0-2</td><td>68.8</td></tr><tr><td>RVQ-3</td><td>47.0</td><td>0-3</td><td>68.7</td></tr><tr><td>RVQ-4</td><td>45.8</td><td>0-4</td><td>67.8</td></tr><tr><td>RVQ-5</td><td>42.0</td><td>0-5</td><td>68.8</td></tr><tr><td>RVQ-6</td><td>36.0</td><td>0-6</td><td>67.8</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>RVQ-7</td><td>36.7</td><td>0-7</td><td>67.0</td></tr></table>

## F. Robustness Experiment Results

Table VII reports the robustness results across style, modality, and phonation changes. Speaker information remains detectable in all tested settings. In CREMA-D, leave-oneemotion-out speaker classification over 91 actors reaches 89.2% mean accuracy; anger is hardest at 75.4%, still far above the 1.1% chance level. In JVS/JVS-MuSiC, a classifier trained on 10,000 read-speech utterances from 100 speakers reaches 23.7% on 600 singing segments — above the 1.0% chance level but substantially lower than within-mode accuracy. For VCTK voiced/unvoiced validation, unvoicedonly active frames reach 70.9% speaker accuracy, and the same-speaker cosine similarity gap between voiced and unvoiced frames is 0.463 (voiced higher). These checks show that speaker information remains detectable across emotion, singing, and unvoiced speech, although it weakens when style or phonation changes.

## IV. DISCUSSION

The results provide a consistent localization pattern. The strongest perceived speaker-related timbre information is carried by low- and mid-frequency magnitude structure, especially in the 1–8 kHz region. Phase/magnitude edits and

TABLE VII  
ROBUSTNESS DETAILS ACROSS STYLE, MODALITY, AND PHONATION CHANGES.
<table><tr><td>Condition</td><td>Quantitative detail</td></tr><tr><td>ESD emotion</td><td>Same-speaker cross-emotion cosine 0.899 vs. different-speaker same-emotion 0.644;</td></tr><tr><td>CREMA-D</td><td>gap +0.255. Leave-one-emotion-out mean 89.2%; held- out anger 75.4%, neutral 95.1%.</td></tr><tr><td>NUS-48E</td><td>Sing-read same-speaker similarity 0.909; cross-modal discriminability 0.246.</td></tr><tr><td>JVS/JVS-MuSiC</td><td>Speech-trained classifier reaches 23.7% on singing; centroid gap 0.272.</td></tr><tr><td>VCTK V/UV</td><td>Unvoiced-only 70.9±1.2%; voiced-vs.- unvoiced cosine gap 0.463.</td></tr></table>

WORLD source/filter swaps, validated with a second SV backend (ECAPA-TDNN), further point to the vocal-tract spectral envelope rather than phase or F0 alone. This agrees with the source-filter view of speech: the vocal tract shapes formants and broad spectral balance, both of which remain relatively stable across utterances; vocal-tract filtering also applies to unvoiced segments, consistent with the unvoiced speaker-recognition results in Section III-F.

The high-frequency results need a separate interpretation. Bands above 8 kHz remain useful for closed-set classifiers and mel-cosine checks, so they should not be treated as empty. However, high-frequency-only same-speaker pairs receive low listener ratings. This suggests that high-band reconstruction may contribute to acoustic detail (e.g., frication or airiness), but is unlikely to restore the speaker similarity that listeners primarily perceive.

Learned representations echo the signal-level findings. Shallow SSL layers retain more speaker information and keep it in a compact subspace, while deeper layers expose less speaker information under our probes. HuBERT and EnCodec results also show that discrete representations are not automatically speaker-free. The amount of speaker information depends on the source layer, codebook size, and residual vector quantization layer. Therefore, content, codec, and token pipelines should check speaker leakage at the actual representation used by the downstream system, not only in explicit speaker embeddings.

The robustness experiments show that this information is not limited to clean neutral voiced speech. Speaker information remains measurable across emotion, speech–singing changes, and unvoiced segments. The strength decreases under larger domain changes, especially speech-to-singing transfer, but remains above chance. Combined with the signal-level findings, this supports the view that speaker-related spectral-envelope structure can survive changes in prosody, style, and phonation.

These findings provide a practical map for speaker-aware speech systems. Voice conversion and speaker-cloning TTS should control low/mid spectral-envelope information when matching a target speaker, while treating high-band enhancement mainly as acoustic detail unless the downstream task specifically requires closed-set speaker discrimination at high sampling rates. Anonymization and tokenized speech language models should test residual speaker information in shallow SSL features, codec layers, and token streams. A single speaker-similarity score is therefore not enough: evaluation should separate machine speaker discriminability, perceived speaker similarity, and representation-level leakage.

## V. LIMITATIONS AND CONCLUSION

This study has several limitations. Many probes are closedset, so speaker-classification accuracy and SV-style scores should not be interpreted as open-set speaker verification performance. WavLM-base-sv operates at 16 kHz, which limits SV-based frequency analysis above 8 kHz; the high-samplerate mel probes, mel-cosine checks, and listening test reduce this issue but do not replace a high-sample-rate open-set SV backend. The high-frequency listening stimuli are narrowband and less natural than full-band speech, so their low ratings describe perceived voice-timbre similarity under the present filtering conditions. The emotion, singing, and token results would also benefit from more speaker backends, more detailed listener metadata, better-controlled playback conditions, and sequence-aware token probes.

Overall, this paper uses speaker identity as a measurable proxy to study where speaker-related timbre information resides in speech signals and learned representations. First, the strongest perceived speaker-related information is concentrated in low/mid-frequency spectral-magnitude structure — specifically the vocal-tract spectral envelope in the 1–8 kHz region — while phase, F0, and high-frequency-only information is weaker under our probes. Second, speaker information is not confined to explicit speaker embeddings: shallow SSL layers and some token streams still expose recoverable speaker information. Third, machine speaker discriminability is not the same as perceived timbre similarity: high-frequency bands contain speaker information that classifiers can use, but they do not produce strong same-speaker ratings in the listening test. Together, these findings indicate where VC, speakercloning TTS, anonymization, codec, and speech language model pipelines should inject, transform, suppress, or audit speaker-related information.

## REFERENCES

[1] B. Sisman, J. Yamagishi, S. King, and H. Li, “An overview of voice conversion and its challenges: From statistical modeling to deep learning,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 132–157, 2021.

[2] S. Chen, C. Wang, Y. Wu, Z. Zhang, L. Zhou, S. Liu, Z. Chen, Y. Liu, H. Wang, J. Li, L. He, S. Zhao, and F. Wei, “Neural codec language models are zero-shot text to speech synthesizers,” IEEE Transactions on Audio, Speech and Language Processing, vol. 33, pp. 705–718, 2025.

[3] X. Zhang, D. Zhang, S. Li, Y. Zhou, and X. Qiu, “Speechtokenizer: Unified speech tokenizer for speech language models,” in International Conference on Learning Representations, B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun, Eds., vol. 2024, 2024, pp. 31 798– 31 818. [Online]. Available: https://proceedings.iclr.cc/paper files/paper/ 2024/file/86d1ab582afb247ccaa84bec4a7e24f7-Paper-Conference.pdf

[4] Z. Ju, Y. Wang, K. Shen, X. Tan, D. Xin, D. Yang, E. Liu, Y. Leng, K. Song, S. Tang, Z. Wu, T. Qin, X. Li, W. Ye, S. Zhang, J. Bian, L. He, J. Li, and S. Zhao, “NaturalSpeech 3: Zero-shot speech synthesis with factorized codec and diffusion models,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, Eds., vol. 235. PMLR, 21–27 Jul 2024, pp. 22 605–22 623. [Online]. Available: https://proceedings.mlr.press/v235/ju24b.html

[5] G. Fant, Acoustic Theory of Speech Production: With Calculations based on X-Ray Studies of Russian Articulations. Berlin, Boston: De Gruyter Mouton, 1971. [Online]. Available: https://doi.org/10.1515/ 9783110873429

[6] Y. Lavner, I. Gath, and J. Rosenhouse, “The effects of acoustic modifications on the identification of familiar voices speaking isolated vowels,” Speech Communication, vol. 30, no. 1, pp. 9–26, 2000. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S016763939900028X

[7] R. E. Remez, J. M. Fellowes, and P. E. Rubin, “Talker identification based on phonetic information,” Journal of Experimental Psychology: Human Perception and Performance, vol. 23, no. 3, pp. 651–666, 1997.

[8] J. Kreiman and D. Sidtis, Foundations of voice studies: An interdisciplinary approach to voice production and perception. Wiley-Blackwell, 2011.

[9] B. Desplanques, J. Thienpondt, and K. Demuynck, “ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification,” in Interspeech 2020, 2020, pp. 3830–3834.

[10] S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, Z. Chen, J. Li, N. Kanda, T. Yoshioka, X. Xiao, J. Wu, L. Zhou, S. Ren, Y. Qian, Y. Qian, J. Wu, M. Zeng, X. Yu, and F. Wei, “Wavlm: Large-scale self-supervised pretraining for full stack speech processing,” IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1505–1518, 2022.

[11] S. Hayakawa and F. Itakura, “Text-dependent speaker recognition using the information in the higher frequency band,” in Proceedings ofICASSP ’94. IEEE International Conference on Acoustics, Speech and Signal Processing, vol. i, 1994, pp. I/137–I/140 vol.1.

[12] L. Besacier, J. Bonastre, and C. Fredouille, “Localization and selection of speaker-specific information with statistical modeling,” Speech Communication, vol. 31, no. 2-3, pp. 89–106, 2000. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0167639399000709

[13] T. Kinnunen and H. Li, “An overview of text-independent speaker recognition: From features to supervectors,” Speech Communication, vol. 52, no. 1, pp. 12–40, 2010. [Online]. Available: https: //www.sciencedirect.com/science/article/pii/S0167639309001289

[14] S. wen Yang, P.-H. Chi, Y.-S. Chuang, C.-I. J. Lai, K. Lakhotia, Y. Y. Lin, A. T. Liu, J. Shi, X. Chang, G.-T. Lin, T.-H. Huang, W.-C. Tseng, K. tik Lee, D.-R. Liu, Z. Huang, S. Dong, S.-W. Li, S. Watanabe, A. Mohamed, and H. yi Lee, “SUPERB: Speech Processing Universal PERformance Benchmark,” in Interspeech 2021, 2021, pp. 1194–1198.

[15] A. Pasad, J.-C. Chou, and K. Livescu, “Layer-wise analysis of a self-supervised speech representation model,” in 2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), 2021, pp. 914–921.

[16] Y. K. Singla, J. Shah, C. Chen, and R. R. Shah, “What do audio transformers hear? probing their representations for language delivery & structure,” in 2022 IEEE International Conference on Data Mining Workshops (ICDMW), 2022, pp. 910–925.

[17] A. Defossez, J. Copet, G. Synnaeve, and Y. Adi, “High fidelity neural´ audio compression,” Transactions on Machine Learning Research, 2023, featured Certification, Reproducibility Certification. [Online]. Available: https://openreview.net/forum?id=ivCd8z8zR2

[18] H. Bu, J. Du, X. Na, B. Wu, and H. Zheng, “Aishell-1: An opensource mandarin speech corpus and a speech recognition baseline,” in 2017 20th Conference of the Oriental Chapter of the International Coordinating Committee on Speech Databases and Speech I/O Systems and Assessment (O-COCOSDA), 2017, pp. 1–5.

[19] Y. Shi, H. Bu, X. Xu, S. Zhang, and M. Li, “AISHELL-3: A Multi-Speaker Mandarin TTS Corpus,” in Interspeech 2021, 2021, pp. 2756– 2760.

[20] J. Yamagishi, C. Veaux, and K. MacDonald, “Cstr vctk corpus: English multi-speaker corpus for cstr voice cloning toolkit (version 0.92),” 2019, [sound]. [Online]. Available: https://doi.org/10.7488/ds/2645

[21] K. Zhou, B. Sisman, R. Liu, and H. Li, “Emotional voice conversion: Theory, databases and esd,” Speech Communication, vol. 137, pp. 1–18, 2022. [Online]. Available: https://www.sciencedirect.com/science/ article/pii/S0167639321001308

[22] H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur, A. Nenkova, and R. Verma, “Crema-d: Crowd-sourced emotional multimodal actors dataset,” IEEE Transactions on Affective Computing, vol. 5, no. 4, pp. 377–390, 2014.

[23] Z. Duan, H. Fang, B. Li, K. C. Sim, and Y. Wang, “The nus sung and spoken lyrics corpus: A quantitative comparison of singing and speech,” in 2013 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference, 2013, pp. 1–9.

[24] S. Takamichi, K. Mitsui, Y. Saito, T. Koriyama, N. Tanji, and H. Saruwatari, “Jvs corpus: free japanese multi-speaker voice corpus,” arXiv preprint arXiv:1908.06248, 2019.

[25] H. Tamaru, S. Takamichi, N. Tanji, and H. Saruwatari, “JVS-MuSiC: Japanese multispeaker singing-voice corpus,” arXiv preprint arXiv:2001.07044, 2020.

[26] M. Morise, F. Yokomori, and K. Ozawa, “World: A vocoder-based high-quality speech synthesis system for real-time applications,” IEICE TRANSACTIONS on Information and Systems, vol. E99-D, no. 7, pp. 1877–1884, July 2016.