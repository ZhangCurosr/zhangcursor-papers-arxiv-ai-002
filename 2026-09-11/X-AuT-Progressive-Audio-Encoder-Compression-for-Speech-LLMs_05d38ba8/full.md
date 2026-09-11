# X-AuT: Progressive Audio-Encoder Compression for Speech LLMs with Cross-Scale Distillation

Haojun Zhang, Yi Zou<sup>†</sup>, Min Chen, Qize Yu, Lianrui Fan Xini Ding, Hao Li, Shuchang Zhou, Xianming Liu, Shiyu Huang<sup>‡</sup>

XPeng Inc.

{zhanghj15,zouy12,chenm36,yuqz,fanlr1,dingxn2}@xiaopeng.com

{lih87,zhousc6,xianming.liu,huangsy16}@xiaopeng.com

## Abstract

Reducing audio-encoder depth lowers the inference cost of speech large language models, but removing complete blocks perturbs the embeddings consumed by the decoder and can cause deletion and premature end-of-sequence errors. We introduce X-AuT, a progressive framework that selects layer combinations through short behavioral probes and restores the pruned model through representation alignment, cross-scale distillation, scheduled student-policy supervision, and LoRA finetuning. The language-model backbone remains frozen, while attention LoRA adapters and the tied output embedding adapt during distillation. Training uses the highest-agreement tier from a transcript-consistency pipeline, followed by source reweighting during finetuning. On ten public Chinese–English benchmarks, compressing Qwen3-ASR-0.6B from 18 to 16 audio-encoder layers reduces macroaverage error from 5.61% to 5.27%. The 14-layer model reaches 5.75% with 20.7% fewer audio-tower parameters. Under the matched recipe, the 1.7B teacher yields 5.55% mean error, compared with 8.45% for self-distillation, and progressive 18→14 pruning outperforms direct pruning (5.75% vs. 6.73%). These singlerun results establish two practical operating points and show that the accuracy effects vary across benchmarks. Project website: https://xpeng-ai.github. io/x-aut.

## 1 Introduction

Speech large language models typically combine a deep audio encoder, a bridge that maps acoustic features into the language-model embedding space, and an autoregressive text decoder [1–3]. This architecture is accurate and flexible, but the audio encoder must process every input frame and therefore remains important for first-token latency in streaming, mobile, and in-vehicle systems [4, 5]. Reducing encoder depth is attractive because it removes complete Transformer blocks and produces a regular, deployment-friendly model.

Starting from a strong pretrained model is also substantially cheaper than training a compact speech encoder and realigning it with a decoder from scratch. We therefore study post-training depth reduction of the 18-layer audio Transformer (AuT) in Qwen3-ASR-0.6B [1]. Removing layers changes the audio embeddings inserted into the decoder and can trigger premature end-of-sequence (EOS) predictions and large deletion errors. Our recipe freezes the pretrained language-model weights while allowing decoder attention LoRA adapters and, during distillation, the tied output embedding to adapt. We use “frozen decoder backbone” in this sense throughout the paper.

Prior ASR compression work has explored decoder distillation, low-rank encoder compression, weight sparsity, supernet training, and encoder-layer removal [6–10]. For an already-pretrained speech LLM, two practical questions remain: which combinations of layers can be removed and recovered under a fixed training budget, and how should recovery address both hidden-state mismatch and errors induced by the student’s own decoding history?

These questions are coupled. A layer that appears redundant in isolation may become important once another layer is removed, because subsequent blocks receive a shifted representation and the bridge must preserve the decoder interface learned during pretraining. Static importance scores therefore cannot fully predict whether a multi-layer candidate can be recovered within a short budget. Selection and recovery must therefore be evaluated together.

X-AuT addresses these questions through progressive pruning and recovery. Before each pruning hop, short behavioral probes compare candidate layer sets under matched initialization, data, and optimization. The selected student then undergoes representation alignment, distillation with scheduled student-policy contexts, and low-rate LoRA finetuning. A Qwen3-ASR-1.7B teacher supplies cross-scale supervision; grouped layer matching and a learned bottleneck projection accommodate the depth and width differences between teacher and student.

Figure 1 compares the Stage 2 16- and 14-layer models with the unpruned baseline across all ten evaluation sets. Each spoke reports accuracy preservation, $1 0 0 ( 1 - e _ { \mathrm { m o d e l } } ) / ( 1 - e _ { \mathrm { b a s e } } )$ , where e is CER or WER expressed as a fraction. This transformation provides a common visual reference across benchmarks; all quantitative comparisons use the macro-average error and the exact CER/WER values in Tables 2 and 3.

The two contours show different accuracy– efficiency tradeoffs. The 16-layer model remains close to or above the baseline across the suite and reaches 5.27% macro error. Further compression to 14 layers yields 5.75% macro error while reducing audio-tower parameters by 20.7% (186.4M→147.8M), with most of the additional loss concentrated on a few English benchmarks.

![](images/9f24632b01b66e47da263e367c4754228f8160136798928cd11fa5e60b58f0a7.jpg)  
Figure 1: Per-benchmark accuracy preservation of the Stage 2 X-AuT models relative to unpruned Qwen3-ASR-0.6B. The dashed contour marks the baseline (100); higher values indicate better accuracy retention.

The benchmark variation also shows why target depth alone is not enough: layer combinations differ in recoverability, and the pruned encoder must be realigned with the decoder under both teacher-forced and student-generated contexts. Figure 2 connects the main steps, from transcriptconsistency filtering and behavioral probes to pro-

gressive pruning and three-stage recovery. The reported configuration uses class 1 data in all three stages, with source reweighting during Stage 2.

Our main contributions are:

• We introduce a progressive recovery framework that combines behavioral candidate screening, cross-scale hidden-state and logit supervision, scheduled student-policy training, and LoRA finetuning while preserving the pretrained decoder backbone.

• A matched teacher-scale comparison reduces mean error from 8.45% with self-distillation to 5.55% with the 1.7B teacher, demonstrating the practical value of cross-scale supervision in the tested setting.

• Layer-pair probes expose non-additive interactions: the {6, 8} pair combines the two strongest single removals but underperforms {5, 6} by 0.85 pp after matched recovery.

• The 14-layer model reaches 5.75% macro error, compared with 5.61% for the baseline, while removing 20.7% of audio-tower parameters and reducing measured encoder latency by 21.4% on the in-vehicle accelerator and 11.4% on H800.

![](images/f078f9834eba388a2a30095f68a16e82eca0b5bc9ff60af791229f0ed9b11bee.jpg)  
Figure 2: Overview of X-AuT. Transcript-consistency filtering constructs the class 1 training pool, matched short-budget probes select recoverable layer combinations, and two pruning hops reduce the audio tower from 18 to 14 layers. Each pruned student is recovered through representation alignment, cross-scale distillation with teacher-forced and scheduled student-policy contexts, and LoRA finetuning while the decoder backbone remains frozen. Evaluation aggregates CER/WER over ten benchmarks; Secs. 4.2 and 3.4 provide the full configuration.

## 2 Related Work

## 2.1 Speech Architectures and Encoder Compression

Modern speech systems increasingly couple an acoustic encoder with a pretrained language model. Qwen3-ASR [1] inserts bridged audio representations into the Qwen3 decoder input, while SLAM-ASR [2] examines lightweight connectors between speech encoders and LLMs. Qwen2-Audio [11] similarly integrates acoustic representations with a general-purpose language model. Whisper [3] follows an encoder–decoder architecture rather than an LLM-connector design, but remains an important reference for multilingual ASR and subsequent compression work. In each case, the encoder processes the full acoustic sequence, making its depth a direct contributor to computation and latency.

ASR compression has targeted different parts of this architecture. Distil-Whisper [6] primarily reduces the decoder while retaining the Whisper encoder. LiteASR [7] combines low-rank factorization with distillation for encoder matrices, whereas structured sparsity removes weights or attention heads [8]. LayerDrop [12] trains networks to tolerate variable depth, and Dynamic Encoder Size [9] learns a supernet from which multiple encoder depths can be extracted. These approaches either introduce compression during training or reduce computation within existing blocks. Direct removal of complete blocks offers a regular architecture, but it also changes the representations consumed by downstream modules.

Kolluri et al. [10] study Whisper encoder-layer pruning in an LLM-based SLAM-ASR model and recover the pruned network with LoRA [13]. X-AuT instead starts from a pretrained Qwen3-ASR audio tower, removes layers in successive hops, and uses short post-removal probes to compare candidate layer sets under a shared recovery budget. This design treats recoverability as a property of a layer combination rather than an additive score assigned to individual layers.

## 2.2 Distillation and Data Selection

Knowledge distillation can transfer teacher output distributions, intermediate representations, or both. Teacher-forced logit distillation evaluates the teacher and student under gold-prefix contexts, which is efficient but differs from inference once the student conditions on its own predictions. On-policy distillation instead supplies supervision on student-generated histories [14]. Ark-ASR [15] develops data-efficient on-policy distillation for ASR, and ASKD-Whisper [16] adjusts the strength of selfdistillation during training. X-AuT combines the two regimes: teacher-forced supervision remains the default, while scheduled student-policy batches expose the teacher to the student’s decoding context after an initial stabilization period. Rollout filtering returns degenerate batches to the teacher-forced objective.

Training data also shape recovery. Curriculum and data-selection methods rank, order, or filter examples according to estimated learning value or label reliability [17]. Our preprocessing compares the source transcript with hypotheses from two external ASR systems and assigns a transcriptconsistency tier from their pairwise agreement. The reported experiments use the highest-agreement tier throughout recovery and alter source weights during Stage 2 to emphasize target-domain data. The resulting pipeline combines confidence-based filtering with phase-specific resampling without relying on a staged mixture of progressively noisier tiers.

## 3 Method

X-AuT filters training examples by transcript agreement, identifies recoverable layer combinations through short behavioral probes, and restores each progressively pruned model with a three-stage training schedule (Figure 2).

## 3.1 Architecture and Objective

An input waveform x is encoded by an N-layer audio encoder $E _ { \theta }$ and bridge $B _ { \phi }$ into audio embeddings $\mathbf { e } = B _ { \phi } ( E _ { \theta } ( \mathbf { x } ) )$ ). Qwen3-ASR places these embeddings at audio-placeholder positions in the token embedding sequence; the causal language-model decoder $D _ { \psi }$ then predicts transcription tokens through a tied output projection $H _ { \omega }$ . This is input-embedding conditioning rather than a separate decoder cross-attention module.

A pruning operation retains an ordered subset $\mathcal { T } \subset \{ 1 , \ldots , N \}$ and forms $E _ { \theta , \mathcal { Z } }$ from those blocks. We seek a recoverable subset and parameters that minimize aggregate text error rate (TER) under a target depth:

$$
\operatorname* { m i n } _ { \mathbb { Z } , \theta ^ { \prime } , \phi ^ { \prime } , \omega ^ { \prime } } \mathrm { T E R } ( E _ { \theta ^ { \prime } , \mathbb { Z } } , B _ { \phi ^ { \prime } } , D _ { \psi } , H _ { \omega ^ { \prime } } ) \quad \mathrm { s . t . } \quad | \mathbb { Z } | = M < N .\tag{1}
$$

The pretrained weights of $D _ { \psi }$ remain frozen. LoRA parameters attached to its q/k/v/o attention projections are trainable, and $\dot { H } _ { \omega } -$ which shares weights with the token embedding in Qwen3-ASR— is trainable during Stages 0–1 and frozen during Stage 2.

Removing encoder blocks changes the conditioning embeddings presented to the decoder. The recovery objective therefore first aligns intermediate and bridge representations, then adapts token distributions under teacher-forced and student-generated prefixes.

## 3.2 Transcript-Consistency Filtering

The source pool contains heterogeneous supervision. For each reference-bearing utterance, two strong ASR systems produce offline hypotheses. After language-aware normalization, we compute the three pairwise edit rates among the source transcript and the two hypotheses, using CER for Chinese and WER for English. Their maximum, $e _ { \mathrm { m a x } } .$ , measures the largest disagreement within the transcript–hypothesis triplet. Exact agreement, Mandarin homophone agreement, and a consistency vote assign one of the nine tiers in Table 1. Lower tier numbers indicate stronger transcript agreement rather than ground-truth quality.

The reported configuration uses class 1 for both distillation stages. Stage 2 retains the same consistency threshold but reweights sources toward cockpit and other target-domain data, separating confidence-based filtering from phase-specific source sampling.

Table 1: Nine-tier transcript-consistency hierarchy. Threshold bands use $e _ { \mathrm { m a x } }$ , the maximum pairwise edit rate. The tiers measure transcript agreement rather than ground-truth quality.
<table><tr><td>Tier</td><td>Agreement band</td><td>Assignment rule</td></tr><tr><td>1</td><td>Full agreement</td><td>Both model hypotheses exactly match the source transcript.</td></tr><tr><td>2</td><td>Partial exact</td><td>At least one of the three text pairs matches exactly (i.e., at least two candidates agree).</td></tr><tr><td>3</td><td>Homophone</td><td>For Mandarin, at least one source-hypothesis pair has zero pinyin CER.</td></tr><tr><td>4</td><td> $0 < e _ { \mathrm { m a x } } \le 5 \%$ </td><td>The consistency vote passes and  $e _ { \mathrm { m a x } }$  falls in the indicated interval.</td></tr><tr><td>5</td><td> $5 \% < e _ { \mathrm { m a x } } \leq 1 0 \%$ </td><td>Same voting rule with a small transcript discrepancy.</td></tr><tr><td>6 7</td><td> $1 0 \% < e _ { \mathrm { m a x } } \leq 2 0 \%$ </td><td>Same voting rule with a moderate discrepancy.</td></tr><tr><td>8</td><td> $2 0 \% < e _ { \mathrm { m a x } } \leq 3 0 \%$ </td><td>Same voting rule with a relatively large discrepancy.</td></tr><tr><td>9</td><td> $3 0 \% < e \mathrm { m a x } \le 5 0 \%$ </td><td>Same voting rule with a large discrepancy.</td></tr><tr><td></td><td> $\mathrm { F a i l u r e } / e _ { \mathrm { m a x } } > 5 0 \%$ </td><td>The vote fails,  $e _ { \mathrm { m a x } } > 5 0 \%$  or an aligned hypothesis is empty.</td></tr></table>

## 3.3 Behavior-Driven Progressive Pruning

We prune in two hops, 18→16→14. The first hop removes original layers {1, 18}. For the second hop, every candidate starts from the same recovered 16-layer checkpoint and is trained with the same 0.3-epoch LoRA warm-up. We first evaluate each remaining layer as a single removal, then evaluate a fixed set of adjacent and non-adjacent layer pairs. Candidate selection uses the lowest aggregate TER on a fixed five-benchmark development suite (Sec. 4). This procedure is more expensive than a static score but directly measures post-removal behavior under the available recovery budget.

The pair probes are necessary because recovery after removing several layers cannot be predicted reliably from the corresponding single-layer scores. The matched comparison selects {5, 6} for the 16→14 hop; Section 5.4 reports the candidate-level results.

## 3.4 Three-Stage Recovery

Each hop uses the same three-stage recipe. The student is the pruned Qwen3-ASR-0.6B model; the teacher is Qwen3-ASR-1.7B with a 24-layer audio encoder. Teacher parameters are frozen and discarded at inference.

## 3.4.1 Stage 0: Representation Alignment

Stage 0 occupies the first 5% of the distillation epoch and combines intermediate-layer, bridge, logit, and transcript losses:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S 0 } } = \lambda _ { \mathrm { l a y e r } } \mathcal { L } _ { \mathrm { l a y e r } } + \lambda _ { \mathrm { b r i d g e } } \mathcal { L } _ { \mathrm { b r i d g e } } + \lambda _ { \mathrm { l o g i t } } \mathcal { L } _ { \mathrm { l o g i t } } + \lambda _ { \mathrm { c e } } \mathcal { L } _ { \mathrm { c e } } . } \end{array}\tag{2}
$$

Both representation losses sum mean-squared error and cosine distance. Teacher layers are divided uniformly into M ordered groups, and student layer m aligns to the last teacher layer in group m. Because the teacher and student hidden widths are 2048 and 1024, respectively, a learned two-layer MLP with a 256-dimensional bottleneck projects teacher hidden and bridge features into the student space. Logit KD uses temperature-scaled KL divergence under gold prefixes.

The pruned audio encoder and bridge are fully trainable. Decoder base weights remain frozen, while rank-32 LoRA adapters on q/k/v/o projections and the tied output embedding are trained with a separate decoder learning rate.

## 3.4.2 Stage 1: Distillation with Scheduled Student-Policy Contexts

Stage 1 disables intermediate-layer loss and uses bridge alignment, teacher-forced logit KD, and gold-transcript CE:

$$
{ \mathcal { L } } _ { \mathrm { o f f } } = \lambda _ { \mathrm { b r i d g e } } { \mathcal { L } } _ { \mathrm { b r i d g e } } + \lambda _ { \mathrm { l o g i t } } { \mathcal { L } } _ { \mathrm { l o g i t } } + \lambda _ { \mathrm { c e } } { \mathcal { L } } _ { \mathrm { c e } } .\tag{3}
$$

After 20% of Stage 1 has elapsed, every fifth optimizer step is scheduled for student-policy supervision. The student greedily generates a prefix; student and teacher are then evaluated on the same generated context. Their distributions are compared over the union of each model’s top-k support (k = 512).

Gold CE remains an anchor on these scheduled steps. The implementation uses no confidence reweighting (weight\_mode=none).

Rollout safeguards prevent degenerate prefixes from entering the KD loss. Generation enforces min\_new\_tokens= 3, uses a duration-aware maximum capped at 256 tokens, and rejects budgetexhausted, over-long, or repetitive rollouts. If at least half of a batch is rejected, that microbatch falls back to the teacher-forced objective. Thus, approximately 20% is a scheduling target; the realized on-policy fraction can be lower after filtering.

## 3.4.3 Stage 2: LoRA Finetuning

Stage 2 initializes from the best Stage 1 checkpoint and optimizes gold-transcript CE for one epoch. The audio encoder, bridge, and decoder LoRA adapters remain trainable at $5 \times 1 0 ^ { - 6 }$ ; the tied lm\_head/embedding is frozen. The class-1 data index is reweighted toward target-domain sources. This stage contains no teacher loss:

$$
\mathcal { L } _ { \mathrm { S 2 } } = \mathcal { L } _ { \mathrm { c e } } ( \mathbf { y } ^ { \mathrm { g o l d } } , \hat { \mathbf { y } } ) .\tag{4}
$$

## 4 Experimental Setup

## 4.1 Models and Parameter Accounting

The student starts from Qwen3-ASR-0.6B [1], whose audio tower contains 18 Transformer blocks and a convolutional/bridge frontend. Counting tensors in the released checkpoint gives 186.376M audio-tower parameters: 12.758M outside the Transformer stack and 9.645M per block. The 16- and 14-layer students therefore contain 167.085M and 147.794M audio-tower parameters, corresponding to 10.35% and 20.70% reductions. We report these exact counts rather than inferring compression from rounded labels such as 180M/140M. The cross-scale teacher is Qwen3-ASR-1.7B with a 24-layer audio tower and 2048-dimensional hidden states; the student audio hidden width is 1024.

## 4.2 Training Data and Quality Labels

The source pool combines public and proprietary multilingual ASR corpora, including AISHELL-1/4/5 [18–20], CommonVoice [21], Emilia [22], GigaSpeech [23], KeSpeech [24], LibriSpeech [25], WenetSpeech [26], and cockpit-domain speech. The pool exceeds 280k hours before quality filtering. Audio is capped at 40 seconds.

Manifest construction. Each corpus is converted to a unified JSONL manifest containing an utterance identifier, audio reference, source transcript, language and split tags, duration, sampling rate, and channel count. Text normalization includes Unicode NFKC normalization, width conversion, Traditional-to-Simplified conversion for Mandarin, removal of invisible characters and numeric separators, dash canonicalization, and whitespace normalization. Original and normalized text are retained for traceability. Figure 3 summarizes the complete path from corpus ingestion through transcript agreement and quality ranking.

Reference-bearing utterances are decoded offline by Qwen3-ASR-1.7B and Qwen3.5-Omni [27]. Records without an inference result, a valid audio reference, or nonempty supervision are excluded. The remaining records receive the consistency labels described in Sec. 3.2. The reported distillation runs use the class-1 manifest index; the first-hop log contains approximately 299k weighted target records per epoch and the second-hop log approximately 292k. Stage 2 keeps class 1 but changes corpus weights, increasing AISHELL-4/5 and cockpit-query contributions while dropping several weakly matched web-speech sources. These counts describe the realized loader indices rather than the size of the 280k-hour source pool.

## 4.3 Selection and Evaluation Suites

Development selection suite. Behavior probes and checkpoint selection use five fixed development or validation subsets: AISHELL-1 (Mandarin CER), CommonVoice-en (English WER), Fleurs-en (English WER), WenetSpeech-meeting (Mandarin CER), and a proprietary cockpit-query subset (Mandarin CER). Frequent evaluation is capped at 25 utterances per benchmark, and the macro average of the five normalized error rates determines checkpoint selection. The final results are evaluated separately on the full public benchmark suite described below.

![](images/1d647369303b6a582c6507b46d943ddc0b18cae6926f29dfb71d5f5820a70c7d.jpg)  
Figure 3: Four-step data pipeline: corpus unification and normalization, dual-system transcription, pairwise CER/WER and consistency voting, and quality-ranked label selection. The rightmost panel reports the WenetSpeech audit summary; audit percentages describe label-selection behavior and are not used as training weights.

Full public evaluation suite. Final checkpoint results are reported on ten public benchmarks: AISHELL-1 (CER) [18]; Fleurs zh/en (CER/WER) [28]; LibriSpeech test-clean/testother (WER) [25]; THCHS-30 (CER) [29]; Tedlium (WER) [30]; CommonVoice v15 zh/en (CER/WER) [21]; and WenetSpeech-meeting (CER) [26]. The macro mean weights benchmarks equally, not by utterance count. The proprietary selection subset is excluded from all main-result tables.

## 4.4 Optimization and Reporting Protocol

The reported 0.6B runs use 32 accelerators, per-device batch size 8, gradient accumulation 2, and global batch size 512. We use AdamW with $2 \times 1 0 ^ { - 5 }$ for the audio tower and $1 \times 1 0 ^ { - 4 }$ for decoderside LoRA plus the tied output embedding during distillation. Stage 2 uses $5 \times 1 0 ^ { - 6 }$ for all trainable parameters and freezes the tied output embedding. Weight decay is 0.01, gradient clipping is 1.0, and training uses bf16. Stage 0 and Stage 1 occupy 0.05 and 0.95 epoch; Stage 2 runs for one epoch. All configurations use seed 42.

Checkpoints are selected by the lowest observed selection-suite macro TER. Tables report the corresponding single-run checkpoint on the full suite. We did not run repeated seeds or bootstrap utterance-level confidence intervals, so boldface denotes the best observed number in a row and not statistical significance. Full hyperparameters are listed in Appendix A.1.

Efficiency is measured separately on an in-vehicle PPU and an NVIDIA H800 GPU using more than 50 utterances of varying duration. We report descriptive averages from the available benchmark output; run-to-run variability was not retained.

## 5 Results

## 5.1 Main Results

The 16-layer model lowers macro-average error from 5.61% to 5.27%, an absolute change of −0.34 pp and a 6.1% relative error reduction (Table 2). It improves four benchmarks: AISHELL-1, CommonVoice zh/en, and WenetSpeech-meeting. The largest gains occur on CommonVoice zh (−1.83 pp), CommonVoice en (−1.85 pp), and WenetSpeech-meeting (−1.30 pp), while the largest degradation is 0.44 pp on Tedlium. Stage 2 improves all ten entries relative to the Stage 1 checkpoint.

The 14-layer model contains 147.794M audio-tower parameters, 20.70% fewer than the 186.376M baseline. Its macro error is 5.75%, a 0.14-pp increase over the baseline (Table 3). CommonVoice zh improves by 1.59 pp and LibriSpeech test-clean by 0.03 pp, whereas Fleurs-en has the largest

Table 2: Full-suite error rates for the 16-layer model. S1 is the Stage 1 best checkpoint and S2 is the finetuned checkpoint. Mean is the unweighted macro average. Relative mean-error change is $( \bar { e } / \bar { e } _ { \mathrm { b a s e } } - 1 ) \times 1 0 \bar { 0 } \% ;$ ; negative is better. Bold marks the best observed value, not statistical significance.
<table><tr><td>Benchmark AuT parameters</td><td>Base (18L) 186.4M</td><td>Prune-16 S1 167.1M</td><td>Prune-16 S2 167.1M</td></tr><tr><td>AISHELL-1 (CER)</td><td>3.33%</td><td>3.30%</td><td>3.21%</td></tr><tr><td>Fleurs-zh (CER)</td><td>2.80%</td><td>3.35%</td><td>3.28%</td></tr><tr><td>Fleurs-en (WER)</td><td>4.17%</td><td>4.28%</td><td>4.23%</td></tr><tr><td>LibriSpeech test-clean (WER)</td><td>2.48%</td><td>2.73%</td><td>2.65%</td></tr><tr><td>THCHS-30 (CER)</td><td>3.87%</td><td>4.10%</td><td>4.06%</td></tr><tr><td>Tedlium (WER)</td><td>3.35%</td><td>3.92%</td><td>3.79%</td></tr><tr><td>LibriSpeech test-other (WER)</td><td>5.39%</td><td>5.90%</td><td>5.79%</td></tr><tr><td>CommonVoice v15 zh (CER)</td><td>9.95%</td><td>8.56%</td><td>8.12%</td></tr><tr><td>CommonVoice v15 en (WER)</td><td>12.35%</td><td>10.74%</td><td>10.50%</td></tr><tr><td>WenetSpeech-meeting (CER)</td><td>8.36%</td><td>8.62%</td><td>7.06%</td></tr><tr><td>Macro mean (%) Relative mean-error change (%)</td><td>5.61</td><td>5.55</td><td>5.27</td></tr></table>

Table 3: Full-suite error rates for the 14-layer model. The 147.8M audio tower has 20.7% fewer parameters than the 186.4M baseline. Formatting and reporting conventions follow Table 2.
<table><tr><td>Benchmark</td><td>Base (18L)</td><td>Prune-14 S1</td><td>Prune-14 S2</td></tr><tr><td>AuT parameters</td><td>186.4M</td><td>147.8M</td><td>147.8M 3.39%</td></tr><tr><td>AISHELL-1 (CER) Fleurs-zh (CER)</td><td>3.33% 2.80%</td><td>3.52% 3.49%</td><td>3.32%</td></tr><tr><td>Fleurs-en (WER)</td><td>4.17%</td><td>5.24%</td><td>5.10%</td></tr><tr><td>LibriSpeech test-clean (WER)</td><td>2.48%</td><td>3.09%</td><td>2.45%</td></tr><tr><td>THCHS-30 (CER)</td><td>3.87%</td><td>4.23%</td><td>4.17%</td></tr><tr><td>Tedlium (WER)</td><td></td><td></td><td></td></tr><tr><td>LibriSpeech test-other (WER)</td><td>3.35%</td><td>4.07%</td><td>3.95%</td></tr><tr><td>CommonVoice v15 zh (CER)</td><td>5.39%</td><td>7.00%</td><td>5.52%</td></tr><tr><td></td><td>9.95%</td><td>9.54%</td><td>8.36%</td></tr><tr><td>CommonVoice v15 en (WER)</td><td>12.35%</td><td>13.94%</td><td>12.49%</td></tr><tr><td>WenetSpeech-meeting (CER)</td><td>8.36%</td><td>10.71%</td><td>8.78%</td></tr><tr><td>Macro mean (%) Relative mean-error change (%)</td><td>5.61</td><td>6.48</td><td>5.75</td></tr></table>

degradation at 0.93 pp. Relative to the 16-layer model, seven benchmark changes remain within 0.3 pp; CommonVoice en (+1.99 pp) and WenetSpeech-meeting (+1.72 pp) account for most of the macro-average gap. Figure 1 summarizes this benchmark-level variation, and the tables report the corresponding CER/WER values.

The two operating points expose a clear tradeoff. The 16-layer model improves the observed macro average, while the 14-layer model provides a larger structural reduction at a small average cost. Section 6 discusses the uncertainty associated with these single-run comparisons.

## 5.2 Training Trajectories

Stage 0 TER falls from 10.88% at step 500 to 7.80% at step 2500, followed by 8.38% at the first Stage 1 evaluation near step 4000 (Figure 4). Stage 1 reaches its minimum of 5.76% at step 47,000. Starting from that checkpoint, Stage 2 reduces TER by another 0.40 pp and reaches 5.36% at step 28,000.

The trajectory shows rapid early recovery followed by slower optimization in Stage 1 and a further gain from Stage 2. Because the stage transition changes both the loss and optimizer state, the curve describes the recovery process rather than the contribution of an individual loss term.

Prune-16 Recovery: Stage 0 + Stage 1 Distillation  
![](images/36963eb231cf2fc4e2fc34084c5d54d11c44c9a034ce1af2e671833faa06cb1e.jpg)

Prune-16 Recovery: Stage 2 LoRA Finetuning  
![](images/c75f4755ec87e7cb5ce078e4b637e26aaba4146b9f0c7f2ffc72798b471d0e2b.jpg)  
Figure 4: Selection-suite TER during 16-layer recovery. (a) Stage 0/1 distillation, with the transition near step 4000. (b) Stage 2 finetuning from the best Stage 1 checkpoint; the dashed line marks 5.76%. The displayed trends summarize checkpoint evaluations and are not uncertainty estimates.

## 5.3 Teacher Scale

A same-scale control replaces the 1.7B teacher with the student’s unpruned 0.6B model while retaining the Stage 0/1 schedule, data, LoRA configuration, and optimization hyperparameters. The cross-scale case additionally requires the learned 2048→1024 teacher projections. At the Stage 1 best checkpoints, mean full-suite error is 5.55% for the cross-scale teacher and 8.45% for the self-teacher; the cross-scale model is better on all ten benchmarks (Appendix A.3). Relative to the unpruned baseline, the cross-scale checkpoint improves macro error by 1.1%, while the self-teacher checkpoint increases it by 50.6%.

The large gap demonstrates the practical value of the stronger teacher under the implemented recipe. On CommonVoice zh/en, the cross-scale checkpoint also improves over the original 0.6B baseline, suggesting that the larger teacher transfers useful acoustic behavior rather than merely restoring the pruned student to its starting point. The scope of this interpretation is discussed in Section 6.

## 5.4 Behavior-Driven Layer Selection

The single-layer sweep spans 6.29–8.90% TER. L6 is the strongest single removal at 6.29%, followed by L5 at 6.42% and L8 at 6.52%. The selected {5, 6} pair reaches 6.93%, whereas {6, 8} reaches 7.78%. Defining the interaction penalty as pair TER minus the mean TER of its constituent removals gives 0.58 pp for {5, 6} and 1.38 pp for {6, 8}.

The adjacent candidates {5, 6}, {6, 7}, and {8, 9} rank ahead of the tested non-adjacent candidates {6, 8} and {3, 6}, while the adjacent {14, 15} pair performs worst overall. Thus, adjacency alone is not a reliable selection rule, and pair recoverability cannot be inferred from single-layer scores in isolation.

![](images/455e50a7603fa2b2618e2b574a4e6b571157db91c030284980d70d2be7547f5d.jpg)

![](images/3470e2c489203b167a20c072a649542aa45a41731688ba0f57432980c86f2fbd.jpg)  
Lower is better; y-axis begins at 5.8%

Figure 5: Layer-screening results for single- and double-layer removals after the same 0.3-epoch warm-up. Lower TER is better; green marks the candidate selected for progressive pruning.

## 6 Discussion

## 6.1 Interpretation

Taken together, the results suggest that successful depth reduction depends on both representation recovery and retained encoder capacity. Removing layers perturbs the audio embeddings presented to the decoder, as reflected in the early TER trajectory and the EOS diagnostics. Stage 0 directly reduces this mismatch at the intermediate and bridge levels, and Stage 1 extends recovery to token distributions under gold and student-generated prefixes. This procedure is sufficient for the 16-layer model to surpass the baseline macro error, but the remaining gap at 14 layers indicates that alignment cannot fully replace the capacity lost through deeper pruning.

The teacher-scale comparison further shows that the source of the recovery signal matters. Under the matched 16-layer recipe, the 1.7B teacher yields 5.55% mean error, compared with 8.45% for self-distillation from the unpruned 0.6B model. The improvement spans all ten benchmarks and is particularly pronounced on CommonVoice zh/en. We interpret this result as useful transfer from the larger teacher within the present training recipe, while recognizing that the comparison does not separate pruning recovery from gains that the same recipe might provide to an unpruned student.

Layer choice and pruning schedule also affect recoverability. Although L8 and L6 are the two strongest single-layer removals, pruning them together performs worse than the selected {5, 6} pair after matched recovery. The pair probes therefore provide information that cannot be inferred by ranking individual layers alone. Similarly, progressive 18→16→14 pruning reaches 5.75% mean error, whereas direct 18→14 pruning reaches 6.73% under the same nominal budget. These comparisons favor explicit pair evaluation and progressive pruning for the model and candidates studied here.

## 6.2 Limitations

The primary results are single runs with seed 42, without repeated-seed variation, paired utterancelevel confidence intervals, or significance tests. Checkpoint selection also relies on fixed development subsets capped at 25 utterances per benchmark. This design makes frequent evaluation tractable, but it introduces selection noise and leaves small differences, including the 0.14-pp gap between the 14-layer model and the baseline, as descriptive observations.

The study covers one model family and a limited set of pruning candidates. Layer interactions, projection-based alignment, and EOS behavior may differ in Whisper [3], Qwen2-Audio [11], or other speech–language architectures. The data pipeline likewise supports nine consistency classes, whereas the reported runs use class 1 throughout and change only source weights in Stage 2. Experiments across model families, broader layer combinations, and controlled data mixtures would clarify which parts of the recipe generalize beyond the present setting.

Only the audio tower is compressed, so autoregressive decoding limits the reduction in end-toend latency. The efficiency measurements are averages from one retained benchmark output and do not include run-to-run uncertainty. A complete deployment study should repeat the hardware measurements under a fixed software stack and report latency distributions alongside average values.

## 7 Conclusion

X-AuT reduces the Qwen3-ASR-0.6B audio tower from 18 to 14 layers through progressive pruning and cross-scale recovery. The 16-layer model improves macro-average error from 5.61% to 5.27%, while the 14-layer model reaches 5.75% with 20.7% fewer audio-tower parameters. The teacher-scale, pair-selection, and direct-pruning controls show that recovery depends on teacher strength, joint layer selection, and pruning schedule. Because the results come from single runs on one model family, broader models and repeated trials are needed to determine how well this tradeoff generalizes.

## References

[1] Xian Shi, Xiong Wang, Zhifang Guo, Yongqi Wang, Pei Zhang, Xinyu Zhang, Zishan Guo, Hongkun Hao, Yu Xi, Baosong Yang, Jin Xu, Jingren Zhou, and Junyang Lin. Qwen3-asr technical report, 2026.

[2] Ziyang Ma, Guanrou Yang, Yifan Yang, Zhifu Gao, Jiaming Wang, Zhihao Du, Fan Yu, Qian Chen, Siqi Zheng, Shiliang Zhang, and Xie Chen. An embarrassingly simple approach for llm with strong asr capacity, 2024.

[3] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. arXiv preprint arXiv:2212.04356, 2022.

[4] Yanzhang He, Tara N. Sainath, Rohit Prabhavalkar, Ian McGraw, Raziel Alvarez, Ding Zhao, David Rybach, Anjuli Kannan, Yonghui Wu, Ruoming Pang, Qiao Liang, Deepti Bhatia, Yuan Shangguan, Bo Li, Golan Pandey, Khe Chai Sim, Thomas Bagby, Shuo-Yin Chang, Kanishka Rao, and Alexander Gruenstein. Streaming end-to-end speech recognition for mobile devices. In Proc. IEEE ICASSP, pages 6381–6385, 2019.

[5] Song Han, Huizi Mao, and William J. Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. In International Conference on Learning Representations (ICLR), 2016.

[6] Sanchit Gandhi, Patrick von Platen, and Alexander M. Rush. Distil-whisper: Robust knowledge distillation via large-scale pseudo labelling. In arXiv preprint arXiv:2311.00430, 2023.

[7] Keisuke Kamahori, Jungo Kasai, Noriyuki Kojima, and Baris Kasikci. Liteasr: Efficient automatic speech recognition with low-rank approximation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2025.

[8] Prasenjit K Mudi, Anshi Sachan, Dahlia Devapriya, and Sheetal Kalyani. Structured sparsity and weight-adaptive pruning for memory and compute efficient whisper models, 2025.

[9] Jingjing Xu, Eugen Beck, Zijian Yang, and Ralf Schlüter. Dynamic encoder size based on data-driven layer-wise pruning for speech recognition. In Proc. Interspeech, pages 4563–4567, 2024.

[10] Ganesh Pavan Kartikeya Bharadwaj Kolluri, Michael Kampouridis, and Ravi Shekhar. On the role of encoder depth: Pruning whisper and lora fine-tuning in slam-asr. In Proceedings ofthe SPEAKABLE Workshop, LREC, 2026.

[11] Qwen Team. Qwen2-audio technical report, 2024.

[12] Angela Fan, Edouard Grave, and Armand Joulin. Reducing transformer depth on demand with structured dropout. In International Conference on Learning Representations (ICLR), 2020.

[13] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR), 2022.

[14] Mingyang Song and Mao Zheng. A survey of on-policy distillation for large language models, 2026.

[15] Yu Lin, Yiming Wang, Runyuan Cai, and Xiaodong Zeng. Data-efficient on-policy distillation for automatic speech recognition, 2026.

[16] Junseok Lee, Nahun Kim, Sangyong Lee, and Chang-Jae Chun. Askd-whisper: Adaptive self-knowledge distillation for efficient and low-latency automatic speech recognition, 2026.

[17] Simon Rampp, Manuel Milling, Andreas Triantafyllopoulos, and Björn W. Schuller. Does the definition of difficulty matter? scoring functions and their role for curriculum learning, 2024.

[18] Huihui Bu, Juntao Du, Xingyu Na, Yanyan Ji, and Fang Zheng. Aishell-1: An open-source mandarin speech corpus and a speech recognition baseline. In Proc. O-COCOSDA, 2017.

[19] Yihui Fu, Luyao Xu, Yukai Zhai, Yuxiang Wang, Kun Liang, Yongqiang Li, Shiliang Zhang, Yonghong Yan, and Xie Chen. Aishell-4: An open source dataset for speech recognition in multi-party conference scenario, 2021.

[20] Yuhao Shi, Luyao Xu, Shiliang Zhang, and Yonghong Yan. Aishell-5: Multi-domain mandarin speech recognition corpus with labeled data of 520 hours. In Proc. IEEE ICASSP, 2023.

[21] Rosana Ardila, Megan Branson, Kelly Lee, Michael Kohler, Rebekah Schumann, Laure Sterckx, Juan Diez Bayron, Prasanga Karunanayake, Ramón Sanabria, Andre Baas, et al. Common voice: A massively-multilingual speech corpus. In Proc. LREC, 2020.

[22] Haorui He, Zengqiang Shang, Chaoren Wang, Xuan Li, Yicheng Gu, Peiyu Hua, Liwei Liu, Chen Yang, Jiaqi Li, Peiyang Shi, Yuancheng Wang, Kai Chen, and Zhizheng Wu. Emilia: An extensive, multilingual, and diverse speech dataset for large-scale speech generation, 2024.

[23] Guoguo Chen, Shuzhou Chai, Guanbo Wang, Jiayu Du, Wei-Qiang Zhang, Chao Weng, Dan Lu, Daniel Povey, Jan Trmal, Junbo Zhang, Mingjie Jin, Sanjeev Khudanpur, Shinji Wang, Shuai Wu, Yong Yang, Yaqing Wang, Zhuo Yu, and Zeyu Wang. Gigaspeech: An evolving, multi-domain asr corpus with 10,000 hours of transcribed audio. In Proc. Interspeech, 2021.

[24] Zhiyuan Tang, Dong Wang, Xiaoxue Xu, Hongbin Zheng, Yexin Lei, Jiawen Li, Shuai Zhang, Yifan Zhu, Jingjing Meng, Haoran Li, Xuyang Xu, Yuhang Zheng, and Shihao Li. Kespeech: An open source speech dataset of mandarin and its eight subdialects. In Proc. IEEE ASRU, 2021.

[25] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: An asr corpus based on public domain audio books. In Proc. IEEE ICASSP, 2015.

[26] Bin Zhang, Hui Lv, Pengcheng Guo, Qijie Shao, Chao Yang, Lei Xie, Xin Xu, Hui Bu, Xie Chen, Chuang Zeng, Di Wu, and Zhendong Peng. Wenetspeech: A 10,000+ hours multi-domain mandarin corpus for asr. In Proc. IEEE ICASSP, 2022.

[27] Qwen Team. Qwen3.5-omni technical report, 2026.

[28] Alexis Conneau, Ankur Bapna, Yu Zhang, Min Ma, Patrick von Platen, Anton Lozhkov, Colin Cherry, Ye Jia, Clara Rivera, Mihir Kale, Noam Remez, Viktor Glavcev, Sanjay Gopala, Jian Ni, Yu-Hsiang Wu, Po-Ning Hsu, Jason Liu, Amir Sahebi, Paul-Ambroise Duquenne, Mia Chen, Vinh Chau, et al. Fleurs: Few-shot learning evaluation of universal representations of speech. In Proc. IEEE SLT, 2022.

[29] Dong Wang and Xuewei Zhang. Thchs-30: A free chinese speech corpus, 2015.

[30] Anthony Rousseau, Paul Deléglise, and Yannick Estève. Ted-lium: An automatic speech recognition dedicated corpus. In Proc. LREC, 2012.

## Appendix

## A.1 Full Hyperparameter Configuration

Table 4: Main 16-layer hyperparameters. The 14-layer run uses the same recipe and initializes from the recovered 16-layer checkpoint.
<table><tr><td>Parameter</td><td>Stage 0</td><td>Stage 1</td><td>Stage 2</td></tr><tr><td>Fraction / epochs</td><td>0.05 epoch</td><td>0.95 epoch</td><td>1 epoch</td></tr><tr><td>LR (audio tower)</td><td> $2 \times 1 0 ^ { - 5 }$ </td><td> $2 \times 1 0 ^ { - 5 }$ </td><td> $5 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>LR (LoRA / tied head)</td><td> $1 0 ^ { - 4 } / 1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 4 } / 1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 6 } / \mathrm { f r o z e n }$ </td></tr><tr><td>Optimizer / weight decay</td><td>AdamW / 0.01</td><td>AdamW / 0.01</td><td>AdamW / 0.01</td></tr><tr><td>Per-device batch / accumulation</td><td>8/2</td><td>8/2</td><td>8/2</td></tr><tr><td>Global batch size</td><td>512</td><td>512</td><td>512</td></tr><tr><td>LoRA r/α/dropout</td><td>32 / 64 / 0.05</td><td>32 / 64 / 0.05</td><td>32 / 64 / 0.05</td></tr><tr><td>Temperature</td><td>1.5</td><td>1.0</td><td></td></tr><tr><td>λlayer</td><td>1.0</td><td>0.0</td><td></td></tr><tr><td>λbridge</td><td>1.0</td><td>0.5</td><td></td></tr><tr><td> $\lambda _ { \mathrm { l o g i t } }$ </td><td>0.2</td><td>0.1</td><td></td></tr><tr><td> $\lambda _ { \mathrm { c e } }$ </td><td>0.5</td><td>1.0</td><td>1.0</td></tr><tr><td>On-policy start / fraction</td><td></td><td>0.2 / 0.2</td><td></td></tr><tr><td>Union top-k / weight mode</td><td></td><td>512 / none</td><td>一</td></tr><tr><td>Minimum / maximum new tokens</td><td></td><td>3 /256</td><td></td></tr><tr><td>Reject-batch threshold</td><td></td><td>0.5</td><td></td></tr></table>

## A.2 Premature-EOS Safeguards

Immediately after audio-encoder pruning, the model can emit EOS after only a few tokens, producing large deletion errors. Qwen3-ASR does not use a separate decoder cross-attention module: bridge outputs replace audio-placeholder embeddings in the causal decoder input. We hypothesize that pruning shifts these conditioning embeddings, weakening acoustic evidence for the pretrained decoder. We do not directly measure embedding mean/variance or EOS causality, so this account motivates the safeguards rather than constituting a mechanistic proof.

The implementation uses three safeguards. First, during main Stage 0/1 distillation, the tied lm\_head/token embedding is trainable at the decoder-side learning rate $( 1 0 ^ { - 4 } ; ~ 5 ~ \times ~ 1 0 ^ { - 5 }$ in the EOS ablation). Decoder Transformer weights remain frozen apart from LoRA. Second, on-policy generation enforces a fixed min\_new\_tokens= 3; its maximum is duration-aware, min(256, max(128, ⌈12d + 32⌉)), where d is the longest audio duration in the batch. Third, rollout filters reject budget-exhausted, implausibly long, and repetitive sequences. If the rejected fraction reaches 0.5, the scheduled on-policy microbatch falls back to teacher-forced distillation.

![](images/95fb0603b498c795f892e3928122241a8201883fdac9e053f591f1379aeb2ea0.jpg)

![](images/ed310d390bbf256c2adb803f5c9e4e412b6f46e9d3ec7dfa5683a751b1135d9f.jpg)  
Figure 6: Premature-EOS ablation on prune-16. A disables both mitigations, B trains the tied head, C applies minimum-token gating, and D combines them. (a) Development TER. (b) Empty-rollout ratio (dashed) and mean rollout length (solid). All runs start from the same Stage 0 checkpoint.

Table 5: Premature-EOS safeguards on prune-16 in a 2×2 ablation. All configurations share the same Stage 0 checkpoint and Stage 1 recipe. Empty events count 100-step windows with at least one empty rollout; rejections sum budget, length, and repetition filters. TER is the best development-suite observation.
<table><tr><td>Configuration</td><td>lm_head</td><td>min_new</td><td>Empty events</td><td>TER</td><td>Rejections</td></tr><tr><td>A: neither</td><td>frozen</td><td>0</td><td>5</td><td>6.86</td><td>20</td></tr><tr><td>B: lm_head</td><td>trainable</td><td>0</td><td>3</td><td>6.83</td><td>11</td></tr><tr><td>C: gating</td><td>frozen</td><td>3</td><td>0</td><td>6.97</td><td>2113</td></tr><tr><td>D: both</td><td>trainable</td><td>3</td><td>0</td><td>6.75</td><td>2162</td></tr></table>

The no-mitigation condition has five windows with a nonzero empty ratio; the maximum is 0.3% and the mean is 0.018%. Training the tied output embedding reduces this to three events, while min\_new\_tokens= 3 removes empty events in both gating conditions. Gating also converts some premature terminations into longer or degenerate rollouts: configurations C and D trigger 2113 and 2162 filter rejections, compared with 20 and 11 for A and B.

The best observed TER is 6.75% for the combined configuration D. The tied-head-only configuration B reaches 6.83% and shows only 0.04-pp best-to-last drift; A drifts from 6.86% to 7.58%. Mean rollout length is 15.06 tokens for D and 11.05 for A. These single runs support using the tied head and gating together in the main recipe, but the 0.08-pp B–D difference is too small to interpret without uncertainty estimates.

## A.3 Teacher-Scale Comparison

The self-teacher control uses the unpruned 18-layer Qwen3-ASR-0.6B model. The cross-scale condition uses Qwen3-ASR-1.7B and the required 2048→1024 bottleneck projections. Other Stage 0/1 settings are matched, and both models are evaluated at their best development checkpoint before Stage 2.

Table 6: Teacher-scale comparison for the 16-layer Stage 1 student. Schedules, data, and optimization are matched; cross-scale training additionally uses the required hidden-width projections. Values are single-run best-checkpoint results.
<table><tr><td>Benchmark</td><td>Base (18L)</td><td>Self-teacher</td><td>Cross-scale</td></tr><tr><td>AISHELL-1 (CER)</td><td>3.33%</td><td>4.29%</td><td>3.30%</td></tr><tr><td>Fleurs-zh (CER)</td><td>2.80%</td><td>4.17%</td><td>3.35%</td></tr><tr><td>Fleurs-en (WER)</td><td>4.17%</td><td>6.15%</td><td>4.28%</td></tr><tr><td>LibriSpeech test-clean (WER)</td><td>2.48%</td><td>6.00%</td><td>2.73%</td></tr><tr><td>THCHS-30 (CER)</td><td>3.87%</td><td>5.23%</td><td>4.10%</td></tr><tr><td>Tedlium (WER)</td><td>3.35%</td><td>11.28%</td><td>3.92%</td></tr><tr><td>LibriSpeech test-other (WER)</td><td>5.39%</td><td>9.89%</td><td>5.90%</td></tr><tr><td>CommonVoice v15 zh (CER)</td><td>9.95%</td><td>11.71%</td><td>8.56%</td></tr><tr><td>CommonVoice v15 en (WER)</td><td>12.35%</td><td>14.89%</td><td>10.74%</td></tr><tr><td>WenetSpeech-meeting (CER)</td><td>8.36%</td><td>10.85%</td><td>8.62%</td></tr><tr><td>Macro mean (%)</td><td>5.61</td><td>8.45</td><td>5.55</td></tr><tr><td>Relative mean-error change (%)</td><td>一</td><td>+50.6</td><td>-1.1</td></tr></table>

The cross-scale checkpoint is better on every public benchmark and lowers mean error from 8.45% to 5.55%. The largest gaps occur on Tedlium (11.28 vs. 3.92), LibriSpeech test-clean (6.00 vs. 2.73), and LibriSpeech test-other (9.89 vs. 5.90). Relative to the original baseline, the cross-scale model also improves CommonVoice zh/en and AISHELL-1, whereas the self-teacher is worse on all ten benchmarks. This pattern is consistent with useful cross-scale transfer, especially on English and higher-error conditions. Because the comparison has one seed and includes projection modules only when dimensions differ, we do not treat it as proof that the teacher creates capabilities absent from every unpruned student.

Table 7: Pair probes for 16→14 pruning after the same 0.3-epoch warm-up. TER is the macro mean over five development subsets, including proprietary SC. ∆ is relative to {5, 6}. Bold marks the best observed candidate.
<table><tr><td>Candidate</td><td>Type</td><td>TER</td><td>∆</td><td>AISHELL</td><td>CV-en</td><td>Fleurs-en</td><td>Wenet / SC</td></tr><tr><td>{5, 6}</td><td>Adj.</td><td>6.93</td><td>一</td><td>0.63</td><td>16.97</td><td>6.85</td><td>7.30/2.91</td></tr><tr><td>{6,7}</td><td>Adj.</td><td>7.37</td><td>+0.44</td><td>0.63</td><td>16.06</td><td>6.48</td><td>8.47 / 5.23</td></tr><tr><td>{8,9}</td><td>Adj.</td><td>7.75</td><td>+0.82</td><td>0.63</td><td>23.39</td><td>5.37</td><td>6.42 / 2.91</td></tr><tr><td>{6, 8}</td><td>Non-adj.</td><td>7.78</td><td>+0.85</td><td>0.63</td><td>20.64</td><td>6.11</td><td>8.03 / 3.49</td></tr><tr><td>{3, 6}</td><td>Non-adj.</td><td>8.12</td><td>+1.19</td><td>0.63</td><td>17.89</td><td>6.48</td><td>8.03 / 7.56</td></tr><tr><td>{14, 15}</td><td>Adj.</td><td>9.93</td><td>+3.00</td><td>0.95</td><td>23.39</td><td>11.48</td><td>8.61 / 5.23</td></tr></table>

## A.4 Layer-Interaction Details

The single-layer results used to form the interaction comparison are L8=5.97%, L6=6.11%, and L5=6.15%. Thus, the non-adjacent pair {6, 8} contains the two best constituent removals (mean 6.04%) but reaches 7.78% as a pair. The selected adjacent pair {5, 6} has a slightly worse constituent mean (6.13%) but reaches 6.93%. The descriptive interaction penalties are therefore 1.74 and 0.80 pp, respectively.

All three adjacent candidates in Table 7 rank above the two non-adjacent candidates, while {14, 15} confirms that adjacency alone is not sufficient. One possible account is that removing a contiguous sub-block creates one residual-stream discontinuity whereas dispersed removal creates two. This explanation is untested; causal activation analysis and a larger factorial candidate set are needed.

## A.5 Progressive versus Direct Pruning

We compare the reported progressive path with a direct 18→14 run that drops the same original layers {1, 18, 5, 6} and uses the same nominal recovery and data budget.

Table 8: Direct 18→14 pruning versus progressive 18→16→14 pruning. Both remove {1, 18, 5, 6} and use the same recovery/data budget. Single-run best-checkpoint results; bold marks the best observed value.
<table><tr><td>Benchmark</td><td>Base</td><td>Direct 18→14</td><td>Progressive</td></tr><tr><td>AISHELL-1 (CER)</td><td>3.33%</td><td>3.81%</td><td>3.39%</td></tr><tr><td>Fleurs-zh (CER)</td><td>2.80%</td><td>3.76%</td><td>3.32%</td></tr><tr><td>Fleurs-en (WER)</td><td>4.17%</td><td>5.94%</td><td>5.10%</td></tr><tr><td>LibriSpeech test-clean (WER)</td><td>2.48%</td><td>3.42%</td><td>2.45%</td></tr><tr><td>THCHS-30 (CER)</td><td>3.87%</td><td>4.62%</td><td>4.17%</td></tr><tr><td>Tedlium (WER)</td><td>3.35%</td><td>5.10%</td><td>3.95%</td></tr><tr><td>LibriSpeech test-other (WER)</td><td>5.39%</td><td>6.31%</td><td>5.52%</td></tr><tr><td>CommonVoice v15 zh (CER)</td><td>9.95%</td><td>9.79%</td><td>8.36%</td></tr><tr><td>CommonVoice v15 en (WER)</td><td>12.35%</td><td>15.22%</td><td>12.49%</td></tr><tr><td>WenetSpeech-meeting (CER)</td><td>8.36%</td><td>9.29%</td><td>8.78%</td></tr><tr><td>Macro mean (%)</td><td>5.61</td><td>6.73</td><td>5.75</td></tr></table>

Progressive pruning reaches 5.75% mean error, compared with 6.73% for direct pruning, and is better on all ten benchmarks. The largest direct-minus-progressive gaps are CommonVoice en (+2.73 pp), CommonVoice zh (+1.43 pp), and Tedlium (+1.15 pp). This shows that the progressive path is preferable under the tested budget. It does not establish that no alternative schedule or larger budget could improve direct pruning.

## A.6 On-Policy Strategy Comparison

Figure 7 compares three Stage 1 runs on prune-16: a plain off-policy baseline, an off-policy control matched to the extra student forward and loss settings used on scheduled steps, and hybrid on-policy training. The dashboard provides the visual trajectory; numerical comparisons use the archived checkpoint logs.

![](images/2d3a74537e08ac063f9521bd1d2e6f63f98c563055c05994b236d96d3476fc49.jpg)  
Figure 7: Stage 1 strategy dashboard for plain off-policy, matched off-policy, and hybrid on-policy training; lower TER is better. Exact best and final values in the accompanying text are taken from the archived checkpoint logs.

The best observed TER values are 6.98% for plain off-policy, 6.23% for matched off-policy, and 6.41% for hybrid on-policy. Final observations are 7.43%, 6.51%, and 6.99%, respectively. Hybrid training therefore improves over the plain baseline but does not outperform the matched off-policy control in this experiment. We retain the hybrid recipe in the main run because it directly supervises student-generated contexts and works well in the complete pipeline, while recognizing that this ablation does not establish accuracy superiority. More seeds and a sweep over rollout fraction are needed.

## A.7 Inference Efficiency

Table 9: Descriptive inference measurements averaged over more than 50 utterances of varying lengths. No run-to-run variance was retained. Peak device memory includes the full model, so encoder-only pruning changes it modestly.
<table><tr><td>Model</td><td>Encoder (ms)</td><td>End-to-end (ms)</td><td>RTF</td><td>Peak memory (MB)</td></tr><tr><td colspan="5">In-vehicle PPU</td></tr><tr><td>Baseline 18L</td><td>14</td><td>486</td><td>0.0786</td><td>1500.3</td></tr><tr><td>Prune-14</td><td>11</td><td>463</td><td>0.0748</td><td>1434.4</td></tr><tr><td>Relative change</td><td>-21.4%</td><td>-4.7%</td><td>-4.8%</td><td>-4.4%</td></tr><tr><td colspan="5">NVIDIA H800</td></tr><tr><td>Baseline 18L</td><td>88</td><td>1672</td><td>0.256</td><td>1500.3</td></tr><tr><td>Prune-14</td><td>78</td><td>1629</td><td>0.245</td><td>1458.2</td></tr><tr><td>Relative change</td><td>-11.4%</td><td>-2.6%</td><td>-4.3%</td><td>-2.8%</td></tr></table>

The 14-layer model reduces encoder time by 21.4% on the in-vehicle PPU and 11.4% on H800. End-to-end reductions are 4.7% and 2.6%, because autoregressive decoding dominates total time. These measurements support encoder pruning as a localized efficiency improvement, especially when encoder and decoder are pipelined, but not as a large end-to-end speedup by itself.