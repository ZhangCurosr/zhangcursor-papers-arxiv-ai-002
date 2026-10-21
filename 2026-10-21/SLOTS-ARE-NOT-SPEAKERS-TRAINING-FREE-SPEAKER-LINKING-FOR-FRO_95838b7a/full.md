# SLOTS ARE NOT SPEAKERS: TRAINING-FREE SPEAKER LINKING FOR FROZEN STREAMING DIARIZERS

Aditya Srinivas Menon, Kumarmanas Nethil, Kavya Manohar

Adalat AI, India

## ABSTRACT

This paper presents a causal, training-free linker that allows a frozen streaming diarizer to track more speakers than it has output slots. Streaming end-to-end diarizers such as Streaming Sortformer emit a fixed number N of concurrent speaker slots. Above N speakers, they keep detecting speech but reuse slots across people, because N bounds how many speakers the model can express at once, not how many it will meet. The linker uses only the host’s slot posteriors and the audio. It keeps the frames where one slot is active, splits each slot where the speaker changes, embeds each piece with an offthe-shelf speaker encoder, and matches it against a bank of speakers seen so far. It only relabels, so missed speech and false alarm are identical to the host’s and every gain is a reduction in speaker confusion. On 1418 sessions from eight corpora, it reduces the 4-slot host’s DER above its cap from 36.8% to 22.6%, and barely moves it below the cap. The same constants transfer to the host’s 8-slot successor and to two streaming EEND hosts from a different model family. On courtroom hearings with up to 23 speakers it reduces cpWER from 82.7% to 24.4%.

Index Terms— Streaming speaker diarization, End-toend neural diarization, Speaker re-identification, Sortformer, Multi-talker speech recognition

## 1. INTRODUCTION

End-to-end neural diarizers emit a fixed number of output channels [1, 2], and their streaming variants fix that number when they are trained [3, 4, 5]. Streaming Sortformer [3, 6] emits N=4 arrival-ordered speaker slots. Production ships the checkpoint that exists, and meetings do not respect N. On a 13-speaker earnings call [7] (Fig. 1) a 4-slot host does not fail loudly. It gets the voice activity regions right, but the 4 slots are reused across people, and the transcript silently merges speakers. Pooled over the 505 sessions in our suite with more than four speakers, its diarization error rate (DER) is 36.8 %, of which 25.3 points are speaker confusion.

The authors describe the model as tracking “the four most dominant speakers in a conversation” [3]. Fig. 1 shows slot 2 being occupied by 5 speakers for over 68 minutes. A slot is not a speaker but a lane, freed when its occupant falls silent and handed to whoever speaks next. The host’s arrival-order speaker cache [3] is not a persistent identity store (Sec. 2), so a speaker who returns after their lane has been reassigned is admitted as new. N is therefore a concurrency cap, not a lifetime cap. EEND-GLA [8] draws the same distinction for block-local attractors, whose total speaker count “can be higher than the limitation” within a block, and resolves it with a jointly trained linker.

The host already frees and reuses slots, but what it lacks is a global speaker identity memory that lives within the session. Identity can therefore be restored outside the model, if the linker is causal, cheap, and does not disturb the host where the host is already right. Pairing a capacity-limited local diarizer with a global linker is an established design [9, 8, 10], and diart [11] does it online. This work adds a global linker to a frozen production host, with nothing trained and no frame of its activity altered. The linker relabels and does nothing else, so missed speech and false alarm are from the host by construction, and every change it makes is speaker confusion alone.

We report four results as our contribution. On 1418 sessions from eight corpora under one scorer, the linker cuts the DER of the 4-slot host [12] from 36.8 to 22.6 above its cap, and moves it by −0.08 below (paired bootstrap CI [−0.41, +0.24]). The same constants, unchanged, lower DER on the host’s 8-slot successor [13] above its higher cap and on two streaming EEND hosts from a different model family [4, 5], on the same sessions. Widening this host’s head to eight slots by fine-tuning [14] does not close the gap: the frozen 4-slot host plus the linker (18.41) beats the widened host plus the same linker (21.42). On four courtroom hearings with up to 23 speakers, where the host predicts four speakers throughout, the linker recovers 20/23, 14/16 and 10/14 identities, and takes cpWER from 82.7 to 24.4.

## 2. TRAINING-FREE LINKER

Observation. On the 4-slot host, pooled over 1418 sessions (213.5 h, eight corpora), missed speech and false alarm barely change with speaker count. Below the cap (913 sessions) they are 8.65 and 2.61, above it (505 sessions) 8.93 and 2.59. Speaker confusion goes from 4.07 to 25.29, and DER with it, from 15.32 to 36.81. Above the cap the host still finds the speech. What it loses is who said it.

Setting. The frozen host emits at 12.5 frames per second and in causal chunks, a posterior tensor of shape [T, N]. At most N concurrent slot activations. The linker consumes that tensor and the audio. It reads no cache, embedding or internal state, so it applies to any diarizer emitting per-channel activity posteriors, and the host is never modified.

Pipeline. (Fig. 1). The linker has four steps and two rules, all causal. Select. For every frame we compute a purity score, the log-odds from the slot posteriors that exactly one slot is active, and keep the purest 60 % of frames in each region. Overlapped and uncertain frames therefore never enter an embedding. Segment. Each slot’s timeline is cut into stretches at speaker change points: a new stretch starts when a region’s embedding has cosine similarity below 0.40 to the running centroid of the current stretch. The stretch is the unit the linker decides on. Embed. The kept frames of a stretch are embedded with TitaNet-large [15] (192-d, 25.3 M parameters, run on CPU) and mean-pooled. A bank holds one centroid per speaker identity. Link. The first stretch on a slot keeps that slot’s identity. A later stretch takes the bank identity with the highest cosine similarity if that similarity is at least 0.60 [11]; moving a slot away from the identity it already carries needs a margin of 0.15. Cannot-link. A stretch may not join an identity that was active on another slot for more than 2 s of the same span: two slots speaking at once are two people [9, 11]. Fold-back. The linker emits a new identity only after 15 s of its speech. It adds no trainable parameters and runs on CPU beside the host at a real-time factor of 0.019.

Invariant and bounds. The linker only relabels, so missed speech and false alarm are the host’s. A new identity is emitted only once it has 15 s of speech or more than ten stretches, so a speaker who never reaches either stays under the slot label. The link threshold of 0.60 is fixed while the bank grows, and we have not tested how often it matches the wrong identity beyond the 23 speakers seen here.

Relation to prior work. A local diarizer with limited capacity paired with a global linker is the EEND-VC design [9]. EEND-GLA [8] trains the linker jointly and has no public checkpoint, and the speaker-tracing buffer [10] carries identity across chunks of a fixed EEND. diart [11] is the nearest ancestor, online and overlap-aware, and inspired our overlap-weighted pooling and incremental centroid assignment. We differ in working on a frozen host’s persistent slot tracks rather than a re-solved 5 s window, in deriving cannot-link from concurrent activity rather than from every local speaker, in adding the relink margin and fold-back, and in never altering a frame of activity, which diart’s assignment can. The host’s arrival-order cache [3] is not an identity store. Its rows separate same- from different-speaker pairs at AUC 0.705 and hold 3.52 s of audio per slot, so the trunk re-derives identity every chunk. Streaming EEND raises the cap by training [4, 5], to 4 and 8 channels in released checkpoints, and Sec. 4.2 measures that move on this host. Speaker caches serve the same end inside speech LLMs [16, 17], and practitioners report the problem<sup>1</sup>.

![](images/a16e9bab9e0a63d65325dd3e34f4bf3c9c9d21defff96444286417d37215e778.jpg)  
Fig. 1. Earnings-21 session 018, 13 speakers. Top: the frozen 4-slot host, regions coloured by dominant reference speaker; Slot 2 alone carries five people. Middle: the linker, 13 of 13 identities. Bottom: the reference. DER 40.0 to 5.4.

## 3. EXPERIMENTAL SETUP

Suite and strata. One suite of 1418 sessions, 213.5 h, from eight corpora: Indic-DiarBench [18] (1164 sessions, 22 Indian languages), Earnings-21 [7], DISPLACE-2024 [19] and DISPLACE-2026 Hindi and Kannada [20], AVA-AVD [21], AMI-SDM [22], and four manually annotated Supreme Court hearings (Sec. 4.3). Strata by reference lifetime speaker count: 913 at 4 or fewer, 505 above 4. Of the 505, 406 are Indic-DiarBench. Only 53 exceed eight speakers, most of them Earnings-21 calls (31), AVA-AVD clips (15) plus three of the four hearings, so the many-speaker evidence rests on three corpora. No system receives the count. Every constant was frozen on disjoint train-role corpora (VoxConverse dev, CallHome, DISPLACE-2024 dev, the AMI-SDM train and the ICSI train fold).

Contamination. Sortformer v2.1’s card [12] lists twelve training corpora without giving splits, among them AMI, ICSI, VoxConverse, AISHELL-4, NOTSOFAR, and NIST SRE 2000, which is the CallHome-English diarization set. Seven of our eight corpora are not on it. AMI is, and the card reports on AMI test SDM, so we mark AMI-SDM indomain. We dropped VoxConverse, AISHELL-4 and ICSI, named in training and absent from the card’s results tables, and NOTSOFAR and CallHome-English, which have no above-cap sessions. Restoring all five moves the below-cap delta from −0.08 to +0.07 and the pooled gain from −6.1 to −5.1. Nemotron-3 [13] trains on DISPLACE-2024 and

Table 1. Single-knob ablations, speaker confusion, 1418 sessions. $\Delta$ is variant minus full linker (positive hurts); bold: 95 % paired session bootstrap CI excludes zero.
<table><tr><td>configuration</td><td> ${ \leq } 4$ </td><td> $\Delta$ </td><td> ${ > } 4$ </td><td> $\Delta$ </td></tr><tr><td>frozen host, no linker full linker</td><td>4.07 3.99</td><td>+0.08</td><td>25.29 11.12</td><td>+14.17</td></tr><tr><td>bank → greedy</td><td>9.24</td><td></td><td></td><td></td></tr><tr><td>- fold-back</td><td>5.34</td><td>+5.25 +1.35</td><td>12.29</td><td>+1.17</td></tr><tr><td>fold-back  $1 5 \mathrm { s } \to 7 \mathrm { s }$ </td><td>4.40</td><td></td><td>11.39</td><td>+0.27</td></tr><tr><td>— clean-frame selection</td><td>4.56</td><td>+0.41</td><td>11.05</td><td>-0.07</td></tr><tr><td>— cannot-link</td><td>4.24</td><td>+0.57</td><td>12.28</td><td>+1.16</td></tr><tr><td></td><td></td><td>+0.25</td><td>11.37</td><td>+0.25</td></tr><tr><td>– relink margin TitaNet-large → small</td><td>3.93 4.31</td><td>-0.06 +0.32</td><td>11.13 16.71</td><td>+0.01 +5.59</td></tr></table>

DISPLACE-2026 Hindi and Kannada, 136 of our sessions; we keep them, and Sec. 4.2 reports Nemotron-3 with and without them.

Scoring and reproducibility. DER is computed with pyannote 4.1 at a 0.25 s collar, overlap scored, no UEM, one optimal speaker mapping per session, with miss, false alarm and confusion reported separately. Pooling is durationweighted. Every interval is a paired session bootstrap with 10 000 resamples. The v2.1 host (117 M) [12] runs at its card’s 30.4 s profile (cache 188, FIFO 40, chunk 340, right context 40, update 300 frames of 80 ms), batch 1, activation threshold 0.5, no post-processing, torch 2.6.0, CUDA 12.4 on A100. The linker runs TitaNet-large on CPU. Rescoring at collar 0 changes no conclusion.

## 4. RESULTS

## 4.1. Lifting the slot cap

Headline. Above the cap the linker takes the frozen host from 36.81 to 22.64 DER (Table 2), a 14.17-point cut delivered entirely as confusion, since miss and false alarm are identical by construction. Below the cap it costs −0.08, nothing measurable: the paired session bootstrap CI is [−0.41, +0.24].

The improvement is not uniform. Earnings-21 [7] (39.6 to 18.1) and the hearings (51.9 to 24.9, Sec. 4.3) carry most of it, while Indic-DiarBench (406 of the 505 above-cap sessions) moves 1.5 over the whole corpus, and within it nearfield audio only 0.11 against 8.1 far-field and 9.8 in the wild. Per corpus the linker improves five of eight and costs at most +0.56 on the other three, all below-cap-only sets with nothing to recover.

Ablations. Table 1 changes one component at a time. Each variant is replayed from the same host pass over the same regions, so it reports confusion alone. Below the cap the host is already right, so the linker’s job is to leave it alone, and the bank and fold-back are what do that: replacing the bank with greedy nearest-centroid assignment adds 5.25 points of confusion there and removing fold-back adds 1.35, while above the cap the same changes cost 1.17, not significant, and 0.27. Above the cap, embeddings matter: the small TitaNet costs 5.59 points and dropping clean-frame selection 1.16. Cannot-link and the relink margin guard rare events, so their pooled effect is small and their above-cap CIs include zero.

Table 2. Pooled DER / speaker confusion, every system on the same 1418 sessions. Bold: the frozen host plus the linker.
<table><tr><td>system</td><td>cap</td><td> ${ \leq } 4$ </td><td> ${ > } 4$ </td><td>all</td></tr><tr><td>frozen host + linker</td><td>4 一</td><td>15.32/4.07 15.24/3.99</td><td>36.81/25.29 22.64/11.13</td><td>24.50/13.14 18.41/7.04</td></tr><tr><td>Ultra (widened)</td><td>8</td><td>17.23/6.26</td><td>35.59/20.76</td><td>25.08/12.46</td></tr><tr><td>+ linker</td><td>一</td><td>17.01/6.04</td><td>27.33/12.51</td><td>21.42/8.80</td></tr><tr><td>Nemotron  $\cdot 3 ^ { \frac { \dagger } { + } }$ </td><td>8</td><td>12.83/1.59</td><td>21.86/8.09</td><td>16.69/4.36</td></tr><tr><td> $+ \operatorname { l i n k e r } ^ { \ddag }$ </td><td>一</td><td>13.40/2.15</td><td>19.74/5.96</td><td>16.11/3.78</td></tr><tr><td>diart</td><td>20</td><td>25.91/13.08</td><td>41.91/29.03</td><td>32.75/19.90</td></tr><tr><td>LS-EEND</td><td>8</td><td>35.52/17.87</td><td>49.42/28.35</td><td>41.46/22.35</td></tr><tr><td>+ linker</td><td>一</td><td>31.94/14.30</td><td>34.36/13.29</td><td>32.98/13.87</td></tr><tr><td>pyannote c-  $1 ^ { \ P }$ </td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathrm { D i a r i Z e n } ^ { \bullet }$ </td><td>一 一</td><td>16.79/6.57 12.61/2.74</td><td>17.72/6.55 16.95/5.63</td><td>17.19/6.56 14.47/3.97</td></tr></table>

Latency: host and Nemotron-3 30.4 s, diart 5 s, LS-EEND 1 s; the linker adds none. <sup>‡</sup> trains on the three DISPLACE corpora. <sup>¶</sup> offline, trains on AMI.

Other streaming hosts. We also ran the linker, with every constant still frozen on Sortformer train-role data, on two streaming EEND checkpoints from a different model family [4, 5]: FS-EEND (cap 4) and LS-EEND (cap 8), both conformer encoders with attractor decoders. On the same 1418 sessions, FS-EEND falls from 55.29 to 44.52 and LS-EEND from 41.46 to 32.98 (Table 2), again entirely as confusion. Two things limit what this shows. These hosts leave channels empty even below their caps, so the linker’s gain there comes from filling empty channels rather than from recycled slots. Their absolute DERs also stay far above this host’s, because every released checkpoint of the family targets telephone or close-talk audio.

## 4.2. Comparison with alternatives

Eight slots, two ways. The alternative to a linker is a host with more slots, and two released checkpoints give us this: Ultra-Sortformer [14], its speaker head widened to eight rows and fine-tuned, and NVIDIA’s own successor, Nemotron-3 [13], retrained with an 8-slot head. Widening the head did not buy accuracy: Ultra scores 25.08 pooled against the frozen host’s 24.50, and the frozen 4-slot host plus the linker (18.41) beats the widened host plus the same linker (21.42), of which 1.25 points are miss and false alarm. Retraining did: with the linker, Nemotron-3 gives the best streaming numbers on the suite, 16.11 pooled and 19.74 above four speakers (Table 2). But eight slots is still a concurrency budget: above eight speakers the linker takes Nemotron-3 from 24.69 to 19.31, and below its cap it pays +0.57, the tax an alwayson linker levies on a host that is already right. Nemotron-3 trains on the three DISPLACE corpora. On the 1282 sessions outside its training list the two hosts are level below the cap (15.32 against 15.48), so its below-cap lead in Table 2 comes from the 136 sessions it trained on; its lead above the cap holds (21.93 against 36.84, and linked 19.78 against 22.58).

Streaming baselines. Table 2 compares, on the same sessions, references and scorer, the linked host with diart [11] (pyannote segmentation-3.0 plus WeSpeaker embeddings, 5 s latency, 20-speaker cap) and LS-EEND [5] (streaming EEND, 8-speaker cap, best released checkpoint). A 12-trial search of diart’s three thresholds on our train-role split returned its shipped defaults, so that row is both tuned and default; LS-EEND runs as released; our constants were frozen on the same train-role split. The linked 4-slot host leads diart by 10.7 (≤4) and 19.3 (>4) points and LS-EEND by 20.3 and 26.8, and the linker lowers LS-EEND itself as well (Sec. 4.1).

Offline systems. DiariZen-md-v2 [23] leads the linked 4-slot host on both strata, and pyannote community-1 leads it above the cap and trails it below. Both see the whole file at once, carry no speaker cap, and train on AMI, so this is not like-for-like; it is the bound on what a frozen streaming host plus a causal linker recovers, and our claim is confined to streaming. Linked Nemotron-3 closes most of that gap (16.11 against DiariZen’s 14.47).

## 4.3. Courtroom hearings and downstream transcription

Reference. Four Indian Supreme Court hearings (three of about five hours, one of 18 minutes, 5 to 23 speakers, Indian English with romanised Hindi) came with transcripts and speaker identities whose segment timestamps are forced alignment pseudo-labels. We re-annotated the timing by hand along each recording’s own timeline, correcting every boundary against the audio rather than inside the segment’s stored window, which comes from the alignment under test and would confirm its own errors; speech the transcript omitted was inserted and spurious segments deleted. One annotator per hearing, two in total; no inter-annotator agreement statistic was computed. To our knowledge this is the first manually time-aligned speaker reference for the domain, and two hearings exceed the cap by a factor of four. The pseudo-reference scores 41.1 % DER against the human labels.

Diarization. The host predicts four speakers on every hearing, 51.9 DER pooled; linked, it recovers 20/23, 14/16 and 10/14 identities at 24.9, against 29.8 for linked Nemotron-3, which predicts eight speakers throughout, and 35.7 for offline DiariZen-md-v2 [23]. On the one 5-speaker hearing the linker is the worst system (30.0 against 25.3 for the host), the below-cap cost made visible.

Table 3. Speaker-attributed transcription, one fixed recogniser per corpus: Parakeet-TDT 0.6B v2 for Earnings-21, an in-house English model for the hearings.
<table><tr><td></td><td colspan="2">Earnings-21</td><td colspan="2">courtroom</td></tr><tr><td>system</td><td>cpWER</td><td>WDER</td><td>cpWER tcpWER</td><td>WDER</td></tr><tr><td>oracle segmentation</td><td>12.3</td><td>1.3</td><td>15.9</td><td>23.7 0.9</td></tr><tr><td>host</td><td>68.2</td><td>36.1</td><td>82.7</td><td>114.5 40.1</td></tr><tr><td>+ linker</td><td>34.2</td><td>14.0</td><td>24.4</td><td>62.0 5.1</td></tr><tr><td>DiariZen-md-v2</td><td>17.7</td><td>4.7</td><td>35.6</td><td>70.6 11.1</td></tr></table>

Transcription. Table 3 asks whether the confusion the linker removes reaches the transcript. Each system’s segments are transcribed by one fixed recogniser per corpus with the speaker label passed through, and an oracle row on the reference segmentation is the diarization-free floor. The linker halves the host’s concatenated-minimum-permutation word error rate (cpWER) on Earnings-21 and takes it from 82.7 to 24.4 on the hearings, with word diarization error rate (WDER) [24] falling in step. Under time-constrained cpWER at a 5 s collar the hearings move from 114.5 to 62.0 against an oracle of 23.7; the residual above the oracle is segment timing, which the linker inherits from the host and does not alter. DiariZen-md-v2 leads on Earnings-21 and trails on the hearings, the same domain split the DER tables show.

## 5. CONCLUSION

We showed that the slot count of a streaming diarizer is a limit on concurrency, not on the number of speakers it can follow. When a slot is reused, the host loses the speaker’s identity, which can be recovered outside the model. A causal linker that reads only the frozen host’s slot posteriors and the audio does this without training and without changing a frame of voice activity, so every change it makes is to speaker confusion. On 1418 sessions from eight corpora it cut DER above the cap from 36.8 to 22.6 and left DER below the cap unchanged, and the same constants transferred to the host’s 8-slot successor and to two streaming EEND hosts from another family. Widening this host’s head to eight slots gained less than adding the linker to it unchanged; a retrained 8-slot host does better, but still needs the linker above eight speakers.

Two limits bound the claim. All results are at the host’s 30.4 s profile [3, 6]; we make no latency claim. And below the cap the linker can over-count: on Indic it emits more speakers than the reference on 43 % of sessions against the host’s 22 %, at no DER cost.

## 6. COMPLIANCE WITH ETHICAL STANDARDS

This study used speech corpora (Sec. 3) released for research under their own licences, and publicly available recordings of Supreme Court hearings for which the authors produced the speaker annotations. No new data were collected from human subjects and no ethical approval was required.

## 7. REFERENCES

[1] Yusuke Fujita, Naoyuki Kanda, Shota Horiguchi, Kenji Nagamatsu, and Shinji Watanabe, “End-to-end neural speaker diarization with permutation-free objectives,” in Proc. Interspeech, 2019, pp. 4300–4304.

[2] Shota Horiguchi, Yusuke Fujita, Shinji Watanabe, Yawen Xue, and Kenji Nagamatsu, “End-to-end speaker diarization for an unknown number of speakers with encoder-decoder based attractors,” in Proc. Interspeech, 2020, pp. 269–273.

[3] Ivan Medennikov, Taejin Park, et al., “Streaming Sortformer: Speaker cache-based online speaker diarization with arrival-time ordering,” in Proc. Interspeech, 2025, pp. 5238–5242.

[4] Di Liang, Nian Shao, and Xiaofei Li, “Frame-wise streaming end-to-end speaker diarization with nonautoregressive self-attention-based attractors,” in Proc. IEEE ICASSP, 2024, pp. 10521–10525.

[5] Di Liang and Xiaofei Li, “LS-EEND: Long-form streaming end-to-end neural diarization with online attractor extraction,” IEEE Trans. Audio, Speech, Lang. Process., vol. 33, pp. 3568–3581, 2025.

[6] Taejin Park, Ivan Medennikov, and et. al., “Sortformer: A novel approach for permutation-resolved speaker supervision in speech-to-text systems,” 2025.

[7] Miguel Del Rio, Natalie Delworth, et al., “Earnings-21: A practical benchmark for ASR in the wild,” in Proc. Interspeech, 2021, pp. 3465–3469.

[8] Shota Horiguchi, Shinji Watanabe, Paola Garc´ıa, Yuki Takashima, and Yohei Kawaguchi, “Online neural diarization of unlimited numbers of speakers using global and local attractors,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 31, pp. 706–720, 2023.

[9] Keisuke Kinoshita, Marc Delcroix, and Naohiro Tawara, “Integrating end-to-end neural and clustering-based diarization: Getting the best of both worlds,” in Proc. IEEE ICASSP, 2021, pp. 7198–7202.

[10] Yawen Xue, Shota Horiguchi, Yusuke Fujita, Shinji Watanabe, Paola Garc´ıa, and Kenji Nagamatsu, “Online end-to-end neural diarization with speaker-tracing buffer,” in Proc. IEEE SLT, 2021.

[11] Juan M. Coria, Herve Bredin, Sahar Ghannay, and So-´ phie Rosset, “Overlap-aware low-latency online speaker diarization based on end-to-end local segmentation,” in Proc. IEEE ASRU, 2021, pp. 1139–1146.

[12] NVIDIA, “diar streaming sortformer 4spk-v2.1,” Hugging Face model card, 2025.

[13] NVIDIA, “Nemotron-3-diarization,” Hugging Face model card, 2026.

[14] MagoAI, “ultra diar streaming sortformer 8spk v1,” Hugging Face model card, 2026.

[15] Nithin Rao Koluguri, Taejin Park, and Boris Ginsburg, “TitaNet: Neural model for speaker representation with 1D depth-wise separable convolutions and global context,” in Proc. IEEE ICASSP, 2022, pp. 8102–8106.

[16] Jing Peng, Ziyi Chen, et al., “G-STAR: End-to-end global speaker-tracking attributed recognition,” arXiv preprint arXiv:2603.10468, 2026.

[17] Mohan Shi, Xiong Xiao, et al., “Train short, infer long: Speech-LLM enables zero-shot streamable joint ASR and diarization on long audio,” arXiv preprint arXiv:2511.16046, 2025.

[18] Deovrat Mehendale, Aditya Mehndiratta, et al., “Indic DiarBench: A multilingual joint diarization and ASR benchmark for Indian languages,” arXiv preprint arXiv:2607.23808, 2026.

[19] Shareef Babu Kalluri, Prachi Singh, et al., “The second DISPLACE challenge,” in Proc. Interspeech, 2024, pp. 1630–1634.

[20] DISPLACE 2026 organisers, “DISPLACE-M,” Challenge website, https://displace2026. github.io, 2026, Accessed 23 Sep 2026.

[21] Eric Zhongcong Xu, Zeyang Song, et al., “AVA-AVD: Audio-visual speaker diarization in the wild,” in Proc. ACM Multimedia, 2022, pp. 3838–3847.

[22] Jean Carletta et al., “The AMI meeting corpus: A preannouncement,” in Proc. MLMI, 2005, pp. 28–39.

[23] Jiangyu Han, Federico Landini, et al., “Leveraging selfsupervised learning for speaker diarization,” in Proc. IEEE ICASSP, 2025, pp. 1–5.

[24] Laurent El Shafey, Hagen Soltau, and Izhak Shafran, “Joint speech recognition and speaker diarization via sequence transduction,” in Proc. Interspeech, 2019, pp. 396–400.