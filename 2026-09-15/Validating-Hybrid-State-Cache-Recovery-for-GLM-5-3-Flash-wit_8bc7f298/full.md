# Validating Hybrid-State Cache Recovery for GLM-5.3-Flash with vLLM and LMCache

Frank Li

UNSW Sydney ・research@n1a.net

Abstract. External cache transfers can succeed while a hybrid language model resumes from an inconsistent state. We examine the ful 45-layer GLM-5.3-Flash model, using the RedHatAI/GLM-5.3-Flash-NVFP4 quantized checkpoint with vLLM and LMCache unde four-way tensor parallelism. A complete-hit recovery mismatch restored state for the full prompt while the scheduler credited one fewer token. We aligned recovery through strict-prefix lookup and established a numerical comparison using shared computation corrections, matched checkpoint scheduling, and fixed per-rank kernel configurations. In a nine-length serial workload, agreement with the modified recomputation control improved from 34/36 to 36/36 generations, each containing 64 token IDs. A separate instrumented run passed recorded transfer-page, efective-tail, and delayed-save checks. Three additional synthetic templates passed 72 paired 256-token contin uations across two fresh-container runs. A subsequent serial performance study preserved output equality across 120 requests; among the measured trials, CPU reload reduced time to first token by 46–64% and total request time by 1.9–7.0% relative to modified cold recomputation. The contribution is an experimentally validated integration repair applying an existing checkpoint-alignment principle. The evidence is confined to one model revision and controlled configuration; it does not establish general determinism, task-quality equivalence, concurrent-serving gains, or capacity beyond GPU memory.

## 1 Introduction

At a prompt length of 3,584 tokens, four transfer workers reported restoring a cached checkpoint for all 3,584 tokens. The scheduler, however, credited only 3,583 tokens and scheduled one additional input token. The subsequent continuation difered from the no-connector control at generated-token index 11. Transfer had occurred on every tensor-parallel rank; a positive cache-hit counter did not explain whether computation resumed from the correct logical position.

This distinction matters when a cache contains state updated in place as well as attention history. Reducing a token count cannot reconstruct an earlier recurrent state. Existing work already describes the need to align reusable attention prefixes and recurrent checkpoints; we apply that principle to a concrete integration failure rather than propose a new checkpointing algorithm [1].

The failure was dificult to isolate because the output reference also required validation. Earlier paired runs could diverge despite identical recorded weights, and apparently successful byte checks had initially inspected inefective all-zero tail locations. Neither source inspection nor one successful transfer was suficient evidence. The eventual comparison therefore used a modified recomputation control with shared computation changes, a matched checkpoint schedule, and a fixed per-rank Flash Linear Attention (FLA) kernel profile. It did not compare a connector-only patch against an untouched production executable.

This case study addresses three questions: which recovery condition failed in the integration; how a useful numerical control was established; and what the repaired path demonstrably passed. Its contributions are an observed complete-hit state/position mismatch and its repair, a documented construction of the numerical comparison, and a bounded regression combining generatedtoken agreement with transfer and completion checks. The historical investigation contains 53 changing experimental rounds, not 53 independent repetitions of one

treatment. The main evaluation uses the final, explicitly identified comparisons.

## 2 System and comparison design

## 2.1 Execution and cached state

The experiment runs one full GLM-5.3-Flash target model across four GPUs with tensor parallelism. GLM-5.3-Flash combines sparse and linear attention; the evaluated NVFP4 checkpoint is Red Hat’s quantized derivative of the Z.ai model. Our workloads use text inputs only [8], [9]. Throughout this paper, “GLM” abbreviates this evaluated model and checkpoint, not the GLM model family as a whole. vLLM performs model execution and scheduling; LMCache provides the external cache path. The four ranks are parts of one inference engine, not four independent model replicas. The experiment does not implement prefill/decode disaggregation.

Figure 1 summarizes the execution and recovery responsibilities.

The integration handles attention-related cache entries, checkpointed state, and model-specific tail entries. Logical engine groups and physical transfer groups are distinct: the audited registration contains 70 entries per rank across six transfer groups. Some transfer representations are byte views, while the observed tail entries are BF16. Thus neither the model checkpoint’s NVFP4 name nor the FP8 KV configuration describes every state tensor’s precision.

MTP configuration does not imply that every step drafts a token: checkpoint-protection steps can disable drafting. The later content extension records four NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs and their UUIDs. This current inventory does not retroactively establish the exact devices used in earlier rounds. Performance-specific runtime settings are reported in §5.5; the shared computation controls are described below.

![](images/37e7b926439023cd80c695548a74b22190bf24ecd333248924011f44b2b2df43.jpg)  
Logical components: four ranks belong to one engine; hardware interconnects are not depicted.

Figure 1: Evaluated logical components. The four tensor-parallel ranks form one GLM-5.3-Flash engine. Both arms share modified computation controls; only the candidate enables the external CPU-cache path. Arrows describe logical interactions, not hardware topology.  
Table 1: Evaluated configuration
<table><tr><td>Setting</td><td>Evaluated configuration</td></tr><tr><td>Target</td><td>Full 45-layer RedHatAI/GLM-5.3-Flash-NVFP4</td></tr><tr><td>Model revision</td><td>36c184c6cda000a481711306df5adde42f63321a</td></tr><tr><td>Runtime</td><td>vLLM 0.1.dev20051+g487ecf187; LMCache 0.5.4; PyTorch 2.13.0+cu130</td></tr><tr><td>Parallelism</td><td>TP=4, PP=1, DP=1; four GPUs on one host</td></tr><tr><td>Precision</td><td>Configured model dtype: BF16; compressed- tensors quantization; FP8 KV setting</td></tr><tr><td>Execution</td><td>Eager, synchronous scheduling; MTP configured with one speculative token</td></tr><tr><td>Limits</td><td>4 GiB KV budget per rank; maximum model length 32,768; maximum sequences 4; token batch limit 8,192</td></tr><tr><td>Checkpoint interval</td><td>C=1,792 tokens</td></tr><tr><td>Numerical control</td><td>Seven recorded FLA autotuner configurations per rank</td></tr></table>

## 2.2 State position and scheduler position

Let N denote prompt length, C the checkpoint interval, B the logical prefix represented by the restored checkpoint, and p the scheduler’s computed-token count. Prefix lengths are counts; token indices are zero based. A state representing prefix [0,B) supports continuation from B. Crediting p=B therefore aligns the restored state with the start of subsequent computation.

For the fully cached prompts in the final workload, restricting lookup to the prompt without its last token selects B=C floor((N−1)/C). The scheduler then computes the nonempty sufix [B,N). This formula assumes the aligned checkpoint is available, as it was in these reload tests. In a general cache, lookup must select an available checkpoint within the query boundary; the formula is not a guarantee of a hit.

## 2.3 Treatments and observations

The cumulative functional patch changes eleven files across LMCache and vLLM. It includes layout and snapshot handling, save completion, checkpoint behavior, and stable selection/packing changes. The FLA installer and the no-connector scheduling fallback are separate controls. Consequently, successful evaluation is a result for their stated combination; it does not establish that the cumulative patch alone preserves production outputs. The final strict-prefix intervention changes two GLM lookup entry points within this existing configuration.

## 3 Integration failures and repair

## 3.1 Observing efective state and waiting for saves

Early tail checks reported equal all-zero pages. Later examination of the efective locations and strides invalidated that coverage claim: equality at an irrelevant address was not a test of the state consumed by subsequent computation. Correcting the addressing exposed nonzero tail data and transfer discrepancies that the earlier observation had missed. The earlier success interpretation was retained as a corrected failure of observation, not counted toward the final validation.

Save submission also had to be distinguished from completion. The integration was changed to wait on the actual DeviceMessagingFuture before allowing dependent state reuse. The final regression inserted a 500 ms delay into saves and paired the actual waits with the delayed operations. This tests whether the recorded dependency is respected under the injected condition; it does not estimate a production failure probability.

Prompt length N = 3,584; checkpoint interval C = 1,792

![](images/1da0ea15e526df0cc3e50bb5f6346ac866defe655074173878096736acb76c8e.jpg)  
B = p: look up a strict prefix, then resume at the matching position

Figure 2: Checkpoint alignment at N=3,584 and C=1,792. Before repair, the loaded state represents B=3,584 while the scheduler credits p=3,583. Strict-prefix lookup instead restores B=p=1,792 and recomputes the remaining sufix. This is a logical boundary example, not a timing trace.  
Table 2: Treatments and observation conditions
<table><tr><td>Configuration</td><td>Computation and scheduling</td><td>Connector</td><td>Observation conditions</td></tr><tr><td>uration</td><td>Original production config- Original production settings</td><td>Not the evaluated treatment</td><td>Not the output control in the final matrix</td></tr><tr><td>Modified recomputation control</td><td>Shared computation corrections, fixed per- rank FLA profile, matched checkpoint sched- ule</td><td>Disabled</td><td>CPU batch/configuration logging; no GPU tensor probes</td></tr><tr><td>Modified candidate</td><td>Corresponding common corrections and FLA profile; connector checkpoint path</td><td>Enabled; lookup boundary changed between the two matrices</td><td>CPU batch/configuration and transfer logging; no GPU tensor probes</td></tr><tr><td>Instrumented candidate</td><td>Repaired candidate configuration</td><td>Enabled</td><td>GPU byte checks, destination poisoning, tail checks, injected save delay</td></tr></table>

These checks answer separate questions. Comparing source and destination pages checks the audited bytes. Observing nonzero efective tails checks that selected useful state entries are exercised. Waiting for the actual save checks a completion dependency. None alone establishes that the bytes describe the prefix from which the scheduler resumes.

## 3.2 Complete-hit boundary mismatch

The fixed-profile pre-fix matrix exposed failures at N=3,584 and N=5,376. For both lengths, raw transfer events from every rank reported a checkpoint covering the entire prompt, B=N. The scheduler’s computed count and external-hit metric instead reported p=N−1, and the first reload batch scheduled one input token.

The archived connector explains the discrepancy. It retained the aligned lookup result in the request tracker used for snapshot retrieval. A later complete-prompt branch reduced the amount credited to the scheduler by one. Snapshot selection still used the retained full boundary. The observed condition was therefore B=N and p=N−1, rather than a snapshot of the shorter prefix. This mechanism is supported by the actual transfer and batch records, not inferred from the metric alone.

The repair excludes the final known token before lookup at both GLM entry points. Alignment then selects an earlier complete checkpoint for an exact-boundary prompt. Restricting the query before checkpoint selection also keeps selection and its associated lookup bookkeeping at the same boundary; subtracting a count after retrieving full state would leave the original problem intact.

At N=3,584, the repaired path loads B=1,792 and computes a 1,792-token sufix. At N=5,376, it loads B=3,584 and computes the same sufix length. Both previously failing continuations then match their controls. This repair increases recomputation relative to the faulty onetoken schedule at exact boundaries. The incremental latency cost relative to that incorrect path has not been measured; §5.5 instead compares repaired reload with modified cold recomputation.

## 4 Establishing a numerical control

## 4.1 What the diagnostic comparisons established

Temperature zero and a fixed seed did not make the original cross-process comparison a reliable attribution test. Recorded batches difered: an earlier no-connector path processed a 3,583-token prefill, while the connector path split it into 1,792 and 1,791 tokens; decode scheduling difered as well. A matched checkpoint schedule removed this particular discrepancy, but output diferences remained.

A subsequent audit found matching named parameter and bufer storages across arms. That narrowed the investigation without proving that all scratch allocations or execution choices were identical. An actual mHC invocation then showed that the attention-output input already difered between arms, while the other seven captured tensor inputs and scalar arguments agreed. This observation does not demonstrate a same-input mHC defect. Probes taken in separate runs also cannot be joined into one observed layer-by-layer causal trace.

The relevant prior distinction is between repeated execution at one shape and invariance across batching or prefix splits. Fixed seeds do not resolve changes in floatingpoint reduction geometry. He discusses this distinction and concrete kernel controls in an author technical report; those results motivate the local control but do not validate this GLM implementation [2].

## 4.2 Recorded FLA profile and its scope

The experiment recorded actual FLA autotuner choices in a baseline run and pinned the seven observed configurations separately for each rank. The installer constrained each observed tuner to its recorded configuration and cleared its selection cache. Invocation of an unrecorded tuner was rejected. Later runs additionally checked the actual selected configuration after calls.

The profile was a joint intervention on seven configurations per rank. No single-kernel ablation establishes that one tile size, warp count, or kernel alone caused the earlier divergence. The profile is neither a general determinism algorithm nor an established performance optimum.

An older reference from a diferent diagnostic round remained unequal under the fixed profile. Its failed assertion and nonzero process exit were preserved. The selected profile reference was identified before the intervention; agreement with it does not turn the older comparison into a pass. This distinction prevents a changing numerical reference from silently changing the reported acceptance criterion.

## 5 Evaluation

## 5.1 Workload and metrics

The same archived plan is used for the pre-fix, repaired, and instrumented matrices. It contains nine prompt lengths around selected boundaries: 3,583, 3,584, 3,585, 3,587, 3,588, 3,589, 5,375, 5,376, and 5,377. Each case performs five serial operations: generate the target from a cold state, generate two interposer requests, reset the GPU cache while retaining the external cache, and regenerate the target. Each paired arm therefore executes 45 operations, including 36 generations and nine resets.

The inputs comprise 27 distinct token sequences generated from one Chinese password-retrieval and explanation template. Target and interposer variants difer in a document tag near the start; they are not independent application domains. There is no explicit request cache salt, and recorded transfer events have an empty salt. The output criterion compares all 64 generated token IDs per request, with temperature zero, seed 42, and EOS ignored. No recorded output contains the known EOS token IDs. Although the prompt requests a longer explanation, the 64-token cap is a finite continuation test, not evidence of satisfying that instruction or preserving task quality.

We distinguish paired output equality, within-engine cold/reload equality, actual four-rank transfers, and internal transfer observations. There are 36 paired comparisons in each two-arm matrix, not 36 distinct tasks. Preemption counters remain zero. The serial reset/interposer procedure exercises the cache path; it does not demonstrate behavior under sustained memory pressure or concurrent preemption.

Evidence checks separately recompute results from archived records and cross-check recorded observations; they do not provide an independently implemented reference for every low-level checker. Hash agreement establishes byte identity, not correct state addressing. The retrospective page audit verifies recorded comparisons rather than re-reading historical device bufers.

## 5.2 Boundary results without GPU tensor probes

Table 4 retains all nine cases. “First diference” is the zero-based generated-token index in the pre-fix reload comparison; a dash means the complete 64-token continuation agrees. B comes from transferred checkpoint metadata and p from the first reload batch. Each repaired row contains four paired generations: target cold, two interposers, and target reload.

The pre-fix matrix has 34/36 equal generation pairs, with the two diferences confined to the exact-boundary reloads. The repaired matrix has 36/36 equal pairs, and all nine target cold/reload comparisons agree. All 36 control generations also agree between the pre-fix and repaired runs. Across the repaired matrix’s nine reloads and four ranks, the first batch has computed=B and scheduled=N−B. Every reload records positive H2D transfer on all four ranks.

At N=3,584, transferred bytes per rank change from 69,625,856 to 54,809,600; at N=5,376, they change from 84,442,112 to 69,625,856. These metadata corroborate the change in selected checkpoints. These before/after byte counts alone do not establish a latency benefit. Section 5.5 separately measures the repaired path against modified cold recomputation; it is not a performance comparison against the incorrect pre-fix path.

An earlier attempt to execute the matrix inadvertently ran only the five-operation first case because of a harness import entry point. The independent coverage verifier rejected it. It is excluded from the nine-case results. Likewise, the changing diagnostic rounds are not pooled into a statistical reliability estimate.

## 5.3 Instrumented transfer and save regression

A separate repaired candidate executes the complete plan with byte observers, destination poisoning before H2D, and injected save delay. Its 36 generations match the archived repaired-run control, and all nine cold/reload pairs agree. This is an instrumented candidate compared with a previously recorded control, not a new two-arm run.

Table 3: Numerical-control experiments
<table><tr><td>Experiment</td><td>Observation</td><td>Interpretation</td></tr><tr><td>Profile capture</td><td>Recorded per-rank configurations and actual call inputs</td><td>A concrete reference for the next intervention</td></tr><tr><td>Fixed-profile first case</td><td>All four 64-token generations agree between arms and with the selected reference</td><td>Supports the controlled paired comparison</td></tr><tr><td>Fresh-container repeat</td><td>New engines reproduce those first-case compar- isons</td><td>Independent repetition of that bounded test</td></tr><tr><td>Full boundary matrices</td><td>Actual configurations verified in both arms on four ranks</td><td>Numerical control retained during the boundary intervention</td></tr></table>

Table 4: Complete boundary matrix before and after repair
<table><tr><td>N</td><td>B before</td><td>p before</td><td>First difference before B = p after</td><td></td><td>Suffix after</td><td>Equal generations after</td></tr><tr><td>3583</td><td>1792</td><td>1792</td><td>一</td><td>1792</td><td>1791</td><td>4/4</td></tr><tr><td>3584</td><td>3584</td><td>3583</td><td>11</td><td>1792</td><td>1792</td><td>4/4</td></tr><tr><td>3585</td><td>3584</td><td>3584</td><td>一</td><td>3584</td><td>1</td><td>4/4</td></tr><tr><td>3587</td><td>3584</td><td>3584</td><td>一</td><td>3584</td><td>3</td><td>4/4</td></tr><tr><td>3588</td><td>3584</td><td>3584</td><td>一</td><td>3584</td><td>4</td><td>4/4</td></tr><tr><td>3589</td><td>3584</td><td>3584</td><td></td><td>3584</td><td>5</td><td>4/4</td></tr><tr><td>5375</td><td>3584</td><td>3584</td><td>一</td><td>3584</td><td>1791</td><td>4/4</td></tr><tr><td>5376</td><td>5376</td><td>5375</td><td>0</td><td>3584</td><td>1792</td><td>4/4</td></tr><tr><td>5377</td><td>5376</td><td>5376</td><td>一</td><td>5376</td><td>1</td><td>4/4</td></tr></table>

Page rows are repeated internal checks, not 20,928 independent round trips or randomized trials. Nonzero tails demonstrate coverage of the selected entries, not equality of every internal model state. The artificial delay and heavy observation deliberately change execution conditions, so this run is not a serving-performance measurement. Its agreement complements the output matrix without GPU tensor probes; it does not replace that matrix.

## 5.4 Additional prompt templates and longer continuations

A subsequent preregistered extension tests three additional synthetic templates: English ledger arithmetic, Chinese ordered-rule application, and English Python-code reasoning. Each template uses prompt lengths 3,583, 3,584, and 5,376, with the same cold/interposer/reset/reload procedure and 256 generated tokens per new request. The historical first case remains as a 64-token bridge to the previously selected reference. The combined plan contains 30 distinct input sequences and 50 operations per arm: four bridge generations and 36 extended generations, totaling 9,472 output tokens per arm.

Two separate disposable containers each run a fresh no-connector engine and a fresh candidate engine under the existing fixed configuration. The exported token plans are identical. Both pairs pass all 40 continuation comparisons, and the two runs also agree on every baseline and candidate continuation. For all ten reloads per run, fourrank transfer metadata and first-batch records agree on B=C floor((N−1)/C), computed=B, scheduled=N−B, and zero draft tokens in the protected first batch. Actual FLA profiles match the recorded policy; no known EOS token or request preemption is observed. Each run exports 49 files whose SHA hashes are independently verified.

This adds 72 successful 256-token generation pairs and eight repeated bridge pairs across two container-level repetitions. These are repeated synthetic inputs, not 80 independent tasks. A setup attempt was aborted before workload execution after detecting an incorrect served-model identifier in the new harness; it has no model-correctness verdict and is excluded. The prompt contents, continuation lengths, output reference, and acceptance criteria were not changed in response to model results. This extension does not repeat the poisoned byte audit, establish task-answer quality, or measure performance. The complete plan, failed setup record, raw exports, and independent checks are retained in the internal evidence archive.

## 5.5 Paired serial performance

A separately preregistered study measures serial streaming completions on the disposable container’s loopback interface. Two fresh containers run the same modified controls in opposite orders: OFF→ON, then ON→OFF. Each engine runs one excluded warmup and three measured trials per length, N {3583,3584,5376}, with distinct early document identities and 128-token continuations. OFF attempts cold and immediate-repeat requests; ON additionally attempts CPU reload after a GPU-only reset. All 120 requests (90 measured) pass complete output-token checks: each input has the same output across its five request conditions and both containers. The four engines are two paired repetitions, not 120 independent experiments. The workload uses one English ledger task (opening balance 137, receipt 58, dispatch 29) with repeated background text. Its 12 tokenized inputs vary length and early document identity; nine inputs enter the measured sample. They are input variants of one synthetic task, not 12 distinct tasks. Requests use temperature 0, seed 42, and ignore\_eos=True; task-answer quality is not evaluated.

Table 5: Instrumented transfer and save regression
<table><tr><td>Check</td><td>Observed coverage</td><td>Result</td></tr><tr><td>Candidate/control generations</td><td>36 continuations × 64 tokens</td><td>All agree</td></tr><tr><td>D2H page-comparison rows</td><td>17,640</td><td>Zero reported byte mismatches</td></tr><tr><td>H2D page-comparison rows</td><td>3,288</td><td>Zero reported byte mismatches; destinations poisoned</td></tr><tr><td>Combined page-comparison rows</td><td>20,928</td><td>All 70 registered entries covered per rank and direction</td></tr><tr><td>Nonzero tail H2D observations</td><td>4 ranks × 9 reloads × 12 entries = 432</td><td>All observed entries nonzero</td></tr><tr><td>Delayed-save wait pairs</td><td>67 per rank; 268 total</td><td>Actual waits cover the injected 500 ms delay</td></tr><tr><td>H2D volume</td><td>2,447,265,792 bytes across four ranks</td><td>Positive transfers for every reload</td></tr></table>

Table 6: Content and continuation extension across fresh engines
<table><tr><td>Measure</td><td>First pair</td><td>Fresh repeat</td></tr><tr><td>Operations per arm</td><td>50</td><td>50</td></tr><tr><td>Historical 64-token bridge comparisons</td><td>4/4</td><td>4/4</td></tr><tr><td>New 256-token comparisons</td><td>36/36</td><td>36/36</td></tr><tr><td>Output tokens per arm</td><td>9,472</td><td>9,472</td></tr><tr><td>Four-rank reload cases with aligned checkpoint and batch</td><td>10/10</td><td>10/10</td></tr><tr><td>Known EOS occurrences, both arms</td><td>0</td><td>0</td></tr><tr><td>Request preemptions, both arms</td><td>0</td><td>0</td></tr></table>

TTFT measures request start to receipt of the first nonempty token-ID event; total latency ends after the stream’s DONE marker. Decode rate counts tokens after the first event over the first-to-last nonempty-event interval, allowing multiple tokens per event. CPU metadata observers remain enabled; GPU tensor probes, poisoned destinations, artificial save delays, initialization and interrequest pauses are excluded. The GPUs are dedicated to the experiment while other production workloads continue on the shared host. Trials run in fixed ascending order of trial index and prompt length; only engine order is reversed. This balances engine order across two runs but does not randomize request history or eliminate shared-host interference.

The performance runs use four NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs. Both arms use the Marlin MoE backend with custom allreduce disabled. NCCL\_P2P\_DISABLE is set to 1 and OMP\_NUM\_THREADS to 2; expandable\_segments is False. FlashInfer autotuning is disabled while the separately fixed FLA profile remains in force. The candidate’s local LMCache server uses LRU, separate object groups, l1-size-gb=16, and disabled lazy L1 allocation. These settings bound the performance result; no inter-host cache transfer or optimized communication comparison is measured.

Each latency is the median of three measured trials. Ratios are medians of per-input OFF-cold/ON-reload ratios, not ratios of the displayed medians. Across the six run/length groups, TTFT ratios range from 1.85–2.80 and total-latency ratios from 1.019–1.076. ON-cold/OFF-cold TTFT ratios range from 1.01–1.04; this is an end-to-end enabled/disabled comparison under the shared modified controls. It does not isolate transfer latency, observer cost, or the cost of all changes relative to production. Raw perrequest values, ranges and event-level decode rates are retained in the evidence tables.

Across these same six groups, OFF-cold total-latency medians span 5.570–6.104 s and ON-reload medians span 5.369–5.745 s. Post-first-event decode-rate medians span 23.38–24.80 and 23.08–24.76 tokens/s, respectively. These ranges describe the observed time scale across inputs and runs; they are not uncertainty intervals.

Attempted path names do not determine actual paths. Among 18 measured OFF immediate repeats, 12 recompute and 6 reuse a local prefix; among 18 ON immediate repeats, 18 reload from CPU and 0 use local-only reuse. The OFF local reuse occurs only at N=5376 and credits 1792 tokens; its N=3583 and N=3584 repeats recompute. At N=5376 the ON CPU path instead restores 3584 tokens, so those repeat conditions do not perform equal sufix work. Consequently, a complete five-path local-hit comparison is unavailable. This is an observed limitation of the evaluated configuration, not evidence that hybrid models cannot support local reuse. Each of the 18 measured explicit CPU reloads has positive fourrank H2D and an independently verified first-batch prefix B=p: 1792, 1792 and 3584 tokens for the three lengths, leaving 1791, 1792 and 1792 prompt tokens to recompute. Classification retains this actual sufix work. We report

![](images/3d4339ecd7b2e169785aa6ba1721804862f84d8e84b6e4c51e8c5d929453fdc2.jpg)  
Figure 3: Serial latency measurements. Hollow points show three measured requests per run, length, and condition; filled markers show medians, and thin bars show observed min–max ranges, not confidence intervals. R1 runs OFF then ON; R2 reverses engine order. Both panels use the same 54 requests and zero-based axes. OFF is the modified no-connector control. The cold conditions have similar TTFT; total time includes the full 128-token generation.

Table 7: Serial latency under matched controls
<table><tr><td>Run</td><td>N</td><td>OFF cold TTFT (ms)</td><td>ON cold TTFT (ms)</td><td>ON reload TTFT (ms)</td><td>TTFT ratio</td><td>Total ratio</td></tr><tr><td>1</td><td>3,583</td><td>447.3</td><td>451.9</td><td>239.4</td><td>1.87</td><td>1.030</td></tr><tr><td>1</td><td>3,584</td><td>447.6</td><td>458.8</td><td>240.4</td><td>1.86</td><td>1.037</td></tr><tr><td>1</td><td>5,376</td><td>670.9</td><td>686.7</td><td>239.7</td><td>2.80</td><td>1.076</td></tr><tr><td>2</td><td>3,583</td><td>447.3</td><td>455.0</td><td>237.9</td><td>1.88</td><td>1.019</td></tr><tr><td>2</td><td>3,584</td><td>447.2</td><td>462.6</td><td>241.4</td><td>1.85</td><td>1.031</td></tr><tr><td>2</td><td>5,376</td><td>671.2</td><td>691.8</td><td>241.2</td><td>2.78</td><td>1.053</td></tr></table>

no confidence interval from two engine pairs, sustainedconcurrency result, or capacity gain beyond GPU memory.

## 6 Related work and limitations

Hybrid-state reuse is an established problem. Marconi explains why recurrent states updated in place require checkpoints aligned with reusable attention prefixes and studies cache admission and eviction. The Sparse Prefix Caching preprint studies sparse checkpoint materialization and sufix recomputation for hybrid and recurrent serving. Our strict-prefix lookup repair applies this existing requirement to a connector/scheduler disagreement. We introduce neither a new eviction policy nor the principle of replaying a sufix from a checkpoint [1], [3].

An upstream vLLM checkpoint RFC separately discusses prefix identity, non-KV logical state, and immutable payload versus logical equivalence. It is a design proposal, currently closed as not planned, rather than an implemented standard. Its prior discussion limits a novelty claim about a general recovery interface; the contribution here is evidence from a running integration [4].

Numerical-control work is also directly relevant. He distinguishes repeatability from batch invariance. The LLM-42 preprint uses verified speculation and replaces speculative KV state with verified state even when generated tokens agree. These works reinforce why finite token equality should not be promoted to complete state or future-output identity. Our experiment fixes one observed profile and tests finite continuations; it does not implement a general online verification system [2], [5].

LMCache supplies the cache infrastructure used in this study. Its broader cache management and transfer mechanisms are existing platform contributions. The local cumulative patch spans representation, completion, scheduling, and common computation changes; eleven changed files do not constitute eleven independently novel mechanisms. Only the final lookup intervention is isolated by the otherwise controlled before/after matrix. The preceding integration changes were not subjected to a full factorial ablation [6].

External validity is limited by one model revision, one four-rank setup, synthetic prompt templates, finite continuations, serial requests, and a fixed profile selected from an observed baseline. The original matrix uses one template and 64-token outputs; the extension adds three templates and 256-token outputs, with only two freshcontainer repetitions. We have not measured all logits, arbitrary future continuation, cross-TP invariance, concurrent preemption, other models, or default-production equivalence. The complete modified control is essential to interpreting the result.

The serial study measures a CPU-reload latency benefit against the shared modified cold control. Costs of fixed configuration, ordering changes and completion barriers relative to the original production configuration remain unmeasured. Immediate repeats do not consistently provide local-only reuse, limiting a complete five-path comparison. Two engine pairs on one shared host do not establish concurrent SLOs, broad reliability, or capacity beyond GPU memory. No public artifact release or production deployment is claimed.

## 7 Conclusion

The evaluated GLM integration restored a complete checkpoint while crediting the scheduler with a shorter prefix. Actual transfer metadata and batch records exposed that mismatch; restricting lookup to a strict prefix repaired both failing exact-boundary cases under a controlled numerical configuration. The repaired nine-case matrix passes all 36 finite continuation comparisons, and a separate instrumented run passes the specified byte, nonzero-tail, and delayed-save checks. The subsequent content extension also passes two fresh-container comparisons with 256-token continuations. The engineering result is a validated recovery path within these conditions, with a measured serial CPU-reload latency benefit under matched controls; representative concurrency, originalproduction overhead and overflow capacity remain open.

## Acknowledgements

The author acknowledges Research Technology Services, UNSW Sydney, for providing GPU resources on the Katana computational cluster [7] for model quantization. AI assistance. AI tools assisted with the research and preparation of this manuscript. The author takes responsibility for the methods, results, and final text.

## References

[1] Rui Pan et al. Marconi: Prefix Caching for the Era of Hybrid LLMs. MLSys, 2025.

[2] Horace He and Thinking Machines Lab. Defeating Nondeterminism in LLM Inference. Author technical report, 2025.

[3] Mikhail Shirokikh and Sergey Nikolenko. Sparse Prefix Caching for Hybrid and Recurrent LLM Serving. arXiv:2605.05219v1, preprint, 2026.

[4] vLLM checkpoint RFC, issue #40533. Engineering design proposal, 2026; closed as not planned when reviewed on 14 September 2026.

[5] Raja Gond et al. LLM-42: Enabling Determinism in LLM Inference with Verified Speculation. arXiv:2601.17768v2, preprint, 2026.

[6] Yuhan Liu et al. LMCache: An Eficient KV Cache Layer for Enterprise-Scale LLM Inference. arXiv:2510.09665v1, preprint, 2025.

[7] PVC (Research Infrastructure), UNSW Sydney. Katana. UNSW, Sydney, 2010. doi:10.26190/669xa286.

[8] Z.ai. GLM-5.3-Flash. Model card, accessed 14 September 2026.

[9] Red Hat AI. GLM-5.3-Flash-NVFP4. Model card, accessed 14 September 2026.