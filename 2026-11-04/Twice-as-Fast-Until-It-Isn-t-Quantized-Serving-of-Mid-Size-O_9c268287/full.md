# Twice as Fast, Until It Isn’t: Quantized Serving of Mid-Size Open LLMs

SMIT KADVANI, KRITI FAUJDAR, and SPURTHI TALLAM, Independent researcher, United States

A team serving a 9B–20B open-weight language model on vLLM or SGLang must choose a quantization recipe, and needs two numbers: the speedup over BF16 and the accuracy given up. We assemble every public measurement of these numbers that we could verify and find that the record supplies them only in pieces. Only two checkpoints, both INT4 builds of Gemma 3 12B, have both a vLLM timing and an accuracy score, and that score covers a single MMLU subject, one that the AWQ build was calibrated on. The batch-1 timings predate current engine releases, AMD and Intel Gaudi have no timing in this band at all, and FP4 has none against BF16. The timings that do exist follow one rule. Decoding a small batch is limited by memory bandwidth, so 4-bit weights roughly double speed (2.12× for Qwen2.5-14B on an A100), but the gain vanishes once a large batch makes the GPU compute-bound (1.02× for Gemma 3 12B at batch size 256), where FP8 is the recipe that can still help. Many published accuracy deltas, including every HumanEval delta, lie within benchmark sampling noise. We close with a seven-item reporting checklist for future measurements.

CCS Concepts: • Computing methodologies → Natural language processing; Neural networks; • General and reference → Surveys and overviews; Performance.

Additional Key Words and Phrases: large language models, post-training quantization, LLM serving, vLLM, SGLang, FP8, AWQ, GPTQ, reporting standards

## ACM Reference Format:

Smit Kadvani, Kriti Faujdar, and Spurthi Tallam. 2026. Twice as Fast, Until It Isn’t: Quantized Serving of Mid-Size Open LLMs. ACM AI Lett. 1, 1 (October 2026), 7 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

## 1 Introduction

Open-weight models of 9B to 20B parameters fit on one data-center GPU in BF16 and on one consumer GPU once quantized, which makes them a common choice for self-hosted serving. A team serving one on vLLM [15] or SGLang [49] must pick a quantization recipe for its GPU, and needs the speedup over BF16 and the accuracy given up. This letter asks whether the public record answers that question for Qwen2.5-14B, Qwen3-14B, Gemma 3 12B, Phi-4, Mistral NeMo, Ministral 3 14B, and gpt-oss-20b [1, 9, 20, 21, 23, 26, 27]. We kept a timing if its source named the engine, GPU, checkpoint, and workload, and an accuracy score if the same source reported a BF16 score from the same harness. We searched the vLLM, SGLang, AMD ROCm, and Intel Gaudi documentation, Hugging Face model cards from the model developers and from quantization publishers (Red Hat AI, PyTorch, Tencent AngelSlim), benchmark reports, and arXiv, pairing each model name with each recipe (FP8, INT8, AWQ, GPTQ, INT4, NVFP4, MXFP4) and each engine, up to 5 October 2026. An AI coding assistant helped collect the sources and extract the numbers; every number was then checked against its cited source on 5 October 2026.

The record answers it only in pieces, and we make three contributions. We explain the timings that do exist with one memory-versus-compute rule (Section 3). We show that many published accuracy deltas are smaller than the sampling noise of their benchmarks (Section 4). We grade what the evidence supports and propose a reporting checklist to close the gaps (Sections 5 and 6).

Earlier work covers parts of this question. Surveys cover quantization before today’s serving stacks [10] and eficient LLM inference broadly [50], and Zhao et al. benchmark post-training algorithms on quality alone [48]. Kurtic et al. measure both accuracy and vLLM speed for FP8, INT8, and INT4, but only for the Llama 3.1 family at 8B, 70B, and 405B [14]. Shi and Ding measure eleven recipes on TensorRT-LLM for Llama-2 and CodeLlama [44]. InferenceX benchmarks vLLM and SGLang on NVIDIA and AMD GPUs continuously, but lists no model in this band [42].

## 2 Recipes and engine support

A recipe is written W�A�: weights stored in � bits, matrix multiplications run on �-bit activations. Weight-only INT4 (W4A16) dequantizes 4-bit weights into a 16-bit matmul. GPTQ (published as OPTQ) chooses the integers with a one-shot second-order reconstruction [7], AWQ rescales the channels that see large activations [17], and the Marlin kernel runs the dequantized matmul, reporting up to 2.8× end-to-end speedup inside vLLM [8]. W8A8 also quantizes activations, so the matmul itself runs in 8 bits: LLM.int8() routes outlier features to a 16-bit path [5], SmoothQuant migrates them into the weights [47], and FP8 uses the E4M3 format [19]. FP8 activation scales are either fixed afte calibration (static) or computed per token at run time (dynamic). MXFP4 stores 4-bit floats in blocks of 32 that share a power-of-two scale [41]; NVFP4 is NVIDIA’s 4-bit float format, executed natively on Blackwell [43].

Engine support depends on the GPU. vLLM runs FP8 W8A8 on Ada, Hopper, and AMD but not on Ampere, where its Marlin kernels still serve GPTQ, AWQ, and FP8 checkpoints as weight-only [46]. SGLang supports fp8, awq, and mxfp4 on AMD Instinct MI300X and later, and keeps its Marlin kernels CUDA-only [43]; AMD documents FP8 and Triton AWQ for vLLM [2].

## 3 Performance: one rule explains the timings

Table 1 has every in-band timing we found: four engine–GPU groups, each with its own model and metric, so speedups compare within a group only. A memory-versus-compute (roofline) view explains all of them. At batch size 1, each generated token reads every weight once, so decode time is set by bytes moved, and 4-bit weights move a quarter of the bytes of BF16. That is why every INT4 recipe is 1.35–2.12× faster at batch size 1; the gain falls short of 4× because dequantization, attention, and KV-cache reads do not shrink. At batch size 256, one read of the weights serves 256 sequences, the matmuls become compute-bound, and W4A16 still computes in 16 bits, so Gemma 3 12B gains only 1.02×. Long prompts are compute-bound for the same reason: the A100 INT4 gain falls from 2.12× to 1.64× at a 30,720-token prompt, and on the H20, whose metric counts prompt tokens, FP8 pulls ahead of AWQ (1.68× versus 1.42×) because Hopper’s FP8 tensor cores run at twice the BF16 rate [22]. The rule predicts that FP8 keeps a speedup at large batch where INT4 loses it, but no in-band source tests FP8 above batch size 1.

gpt-oss-20b is a diferent case: a mixture-of-experts model with 20.9B total and 3.6B active parameters, post-trained with MXFP4 expert weights [23]. Its only in-band timing compares engines, not recipes: at default settings, vLLM delivers 2.93× the total throughput of SGLang on the same prompts, and the same report raises vLLM to 19,128.79 tokens/s with asynchronous scheduling [11]. No BF16 timing exists, although MXFP4 values are exactly representable in BF16 at realistic block scales, so a BF16 copy would have the same accuracy. For AMD, published FP8 tables stop at Llama 3.1 70B and 405B [18]; Intel’s Gaudi 3 data uses Optimum-Habana [13], and Lee et al. measure FP8 on Gaudi 2 without vLLM or SGLang [16].

## 4 Accuracy: most deltas are within noise

Table 2 covers five of the seven models; we found no quantized accuracy table for Ministral 3 14B and none for INT4 on Qwen2.5-14B-Instruct. No source reports a standard error, so we supply a yardstick. For a score � on � questions, one standard error of a single run is ${ \sqrt { p ( 1 - p ) / n } } ;$ about 3.9 points on HumanEval (� = 164) [3], 0.8–1.3 on GSM8K Manuscript submitted to ACM

Table 1. Public vLLM and SGLang timings in the 9B–20B band. Each group has its own BF16 baseline, and the speedup is relative to it (higher is beter; for latency, BF16 latency divided by recipe latency). Pairs a/b give two setings of the same run. “n.r.” means not reported. <sup>†</sup>Memory comes from a Hugging Face Transformers run in the same source, because vLLM and SGLang pre-allocate GPU memory; units follow each source. <sup>★</sup>Checkpoint size from the model card [23].
<table><tr><td>Recipe</td><td>Speed or latency</td><td>Speedup vs BF16</td><td>Memory†</td></tr><tr><td colspan="4">vLLM 0.6.3, one A100 80GB, Qwen2.5-14B-Instruct, batch size 1, 2048 output tokens;</td></tr><tr><td colspan="4">tokens/s at input length 1/30,720 [25]</td></tr><tr><td>BF16 (baseline)</td><td>46.30 /37.18</td><td>1.00 / 1.00</td><td>28.08 GB</td></tr><tr><td>GPTQ-Int8 (W8A16)</td><td>70.40 / 49.23</td><td>1.52 / 1.32</td><td>16.11 GB</td></tr><tr><td>AWQ (W4A16)</td><td>92.66 / 59.01</td><td>2.00 / 1.59</td><td>9.79 GB</td></tr><tr><td>GPTQ-Int4 (W4A16)</td><td>98.02 / 60.91</td><td>2.12/1.64</td><td>9.94 GB</td></tr><tr><td colspan="4">vLLM benchmark_latency, H100, Gemma 3 12B IT, 256 input and 256 output tokens;</td></tr><tr><td colspan="4">latency in seconds at batch size 1/256 [24]</td></tr><tr><td>BF16 (baseline)</td><td>3.73 /14.07</td><td>1.00 / 1.00</td><td>24.51 GB</td></tr><tr><td>TorchAO INT4 (W4A16)</td><td>2.73 /13.81</td><td>1.37 / 1.02</td><td>10.37 GB</td></tr><tr><td>AWQ-INT4 (W4A16)</td><td>2.76 / 13.93</td><td>1.35 / 1.01</td><td>12.60 GB</td></tr><tr><td colspan="4">SGLang 0.4.6.post1, one H20 96GB, Qwen3-14B, batch size 1, 2048 output tokens;</td></tr><tr><td colspan="4">(prompt + output tokens)/s at input length 1/30,720 [29]</td></tr><tr><td>BF16 (baseline)</td><td>47.10 /525.80</td><td>1.00 / 1.00</td><td>28,402 MB</td></tr><tr><td>FP8 (W8A8)</td><td>97.11 / 880.72</td><td>2.06 / 1.68</td><td>16,012 MB</td></tr><tr><td>AWQ-INT4 (W4A16)</td><td>96.49 / 744.74</td><td>2.05 / 1.42</td><td>9,962 MB</td></tr><tr><td colspan="4">vLLM 0.11.0 and SGLang 0.5.4.post2 at default settings, one H100 SXM, gpt-oss-20b,</td></tr><tr><td>1000 ShareGPT prompts; total tokens/s [11] MXFP4, vLLM</td><td>16,493.75</td><td></td><td>12.8 GiB*</td></tr><tr><td>MXFP4, SGLang</td><td>5,626.99</td><td>n.r. n.r.</td><td>12.8 GiB*</td></tr></table>

(� = 1,319) [4], 0.34–0.39 on full MMLU (� = 14,042) [12], and 2.3 on MMLU philosophy (� = 311), at the BF16 scores in Table 2. The error of a delta also depends on how many answers flip between runs [6], which no source reports.

Against this yardstick, the HumanEval deltas, from −3.05 to +6.10, are −5 to +10 problems and cannot separate recipes. The GSM8K losses of 4.17 (AWQ on Qwen3-14B) and 4.02 (GPTQ on Mistral NeMo) are more than three single-run standard errors and are the clearest losses in the table. On full MMLU, every 8-bit delta is at most 0.77 points, about two standard errors, and INT4 costs 0.43 to 1.43 points; NVFP4 costs 1.63. The Gemma INT4 rows are both small-sample and contaminated, since the AWQ checkpoint was calibrated on the subject it is scored on. Harness settings matter as much as recipes: the two Qwen3-14B groups disagree on GSM8K for the same W4A16 idea (−1.06 versus +3.94), and their BF16 baselines difer by 20.77 points. gpt-oss-20b has no delta; its card reports 85.3 on MMLU at high reasoning efort, a protocol not comparable with the rows above [23].

## 5 What the evidence supports

Table 3 grades each claim. For latency-bound serving at small batch, W4A16 cuts per-token latency by a quarter to a half and costs about one MMLU point; check GSM8K, which loses up to four. For throughput-bound serving at large batch, INT4 is a memory choice, taking a 14B model from about 28 GB to about 10 GB, and FP8 is the recipe the roofline favors, but no public in-band timing confirms it. On Ampere, FP8 checkpoints run only as weight-only, so the roofline predicts gains like GPTQ-Int8’s 1.52×, not INT4’s.

Table 2. Accuracy against BF16. BF16 rows give the source’s score; other rows give the change in points (negative is worse). Each group has its own BF16 baseline because harness setings difer by source; compare deltas within a group. None of these scores comes from a timing run in Table 1. <sup>‡</sup>HumanEval pass@2 on 64 samples. <sup>§</sup>MMLU philosophy subset (311 questions) only; the AWQ checkpoint was calibrated on 30 of its samples.
<table><tr><td>Recipe</td><td>MMLU</td><td>GSM8K</td><td>HumanEval</td><td>C-Eval</td></tr><tr><td colspan="5">Qwen3-14B, AngelSlim, lm-eval, engine n.r. [45]</td></tr><tr><td>BF16</td><td>78.90</td><td>88.40</td><td>55.49</td><td>83.06</td></tr><tr><td>FP8-static</td><td>-0.33</td><td>+1.06</td><td>+1.83</td><td>-0.44</td></tr><tr><td>FP8-dynamic</td><td>+0.02</td><td>-0.08</td><td>-3.05</td><td>-0.82</td></tr><tr><td>INT8-dynamic (W8A8)</td><td>-0.77</td><td>-2.12</td><td>+0.61</td><td>-1.19</td></tr><tr><td>GPTQ-INT4 (W4A16)</td><td>-0.88</td><td>-1.06</td><td>+2.44</td><td>-2.01</td></tr><tr><td>AWQ-INT4 (W4A16)</td><td>-1.22</td><td>-4.17</td><td>+6.10</td><td>-1.04</td></tr><tr><td colspan="5">Qwen3-14B, Red Hat AI, lm-eval on vLLM [38, 40]</td></tr><tr><td>BF16</td><td>76.81</td><td>67.63</td><td>n.r.</td><td>n.r.</td></tr><tr><td>FP8-dynamic</td><td>-0.34</td><td>-1.44</td><td>n.r.</td><td>n.r.</td></tr><tr><td>GPTQ W4A16</td><td>-0.91</td><td>+3.94</td><td>n.r.</td><td>n.r.</td></tr><tr><td colspan="5">Qwen3-14B, Red Hat AI, lm-eval on vLLM [39]</td></tr><tr><td>BF16</td><td>78.86</td><td>88.70</td><td> $9 0 . 7 4 ^ { \frac { 1 } { + } }$ </td><td>n.r.</td></tr><tr><td>NVFP4 (W4A4)</td><td>-1.63</td><td>-0.45</td><td> $- 0 . 8 7 ^ { \ddagger }$ </td><td>n.r.</td></tr><tr><td colspan="5">Qwen2.5-14B-Instruct, Red Hat AI, lm-eval on vLLM [32]</td></tr><tr><td>BF16</td><td>79.87</td><td>83.55</td><td>n.r.</td><td>n.r.</td></tr><tr><td>INT8 W8A8</td><td>-0.12</td><td>+1.97</td><td>n.r.</td><td>n.r.</td></tr><tr><td colspan="5">Gemma 3 12B IT, Red Hat AI, lm-eval on vLLM [33, 34]</td></tr><tr><td>BF16</td><td>72.15</td><td>88.10</td><td>n.r.</td><td>n.r.</td></tr><tr><td>FP8-dynamic</td><td>-0.35</td><td>-0.08</td><td>n.r.</td><td>n.r.</td></tr><tr><td>INT8 W8A8</td><td>-0.50</td><td>-0.38</td><td>n.r.</td><td>n.r.</td></tr><tr><td colspan="5">Gemma 3 12B IT, PyTorch, lm-eval on Transformers [24]</td></tr><tr><td>BF16</td><td>79.10$</td><td>n.r.</td><td>n.r.</td><td>n.r.</td></tr><tr><td>TorchAO INT4 (W4A16)</td><td>-3.548</td><td>n.r.</td><td>n.r.</td><td>n.r.</td></tr><tr><td>AWQ-INT4 (W4A16)</td><td>-2.25$</td><td>n.r.</td><td>n.r.</td><td>n.r.</td></tr><tr><td colspan="5">Phi-4, Red Hat AI, lm-eval on vLLM [35–37]</td></tr><tr><td>BF16</td><td>80.30</td><td>90.07</td><td>n.r.</td><td>n.r.</td></tr><tr><td>FP8-dynamic</td><td>0.00</td><td>+0.60</td><td>n.r.</td><td>n.r.</td></tr><tr><td>INT8 W8A8</td><td>+0.09</td><td>+0.23</td><td>n.r.</td><td>n.r.</td></tr><tr><td>GPTQ W4A16</td><td>-0.43</td><td>-0.38</td><td>n.r.</td><td>n.r.</td></tr><tr><td colspan="5">Mistral NeMo Instruct 2407, Red Hat AI, lm-eval on vLLM [30, 31]</td></tr><tr><td>BF16</td><td>68.35</td><td>74.45</td><td>n.r.</td><td>n.r.</td></tr><tr><td>FP8-static</td><td>+0.15</td><td>-1.44</td><td>n.r.</td><td>n.r.</td></tr><tr><td>GPTQ W4A16</td><td>-1.43</td><td>-4.02</td><td>n.r.</td><td>n.r.</td></tr></table>

Table 3. What the public record supports for 9B–20B models on vLLM and SGLang, and how firmly. Confidence reflects the number of independent sources, models, and GPUs behind each claim.
<table><tr><td>Question</td><td>What the record shows</td><td>Confidence</td></tr><tr><td>Speed at batch size 1</td><td>W4A16 is 1.35-2.12× faster than BF16; FP8 ties AWQ on the H20</td><td>Moderate: 3 sources, 3 GPUs, 2024-25 engines</td></tr><tr><td>Speed at large batch</td><td>W4A16 gains nothing at batch 256; FP8 untested</td><td>Low: 1 model, 1 source</td></tr><tr><td>Speed on long prompts</td><td>FP8 beats AWQ (1.68× vs 1.42×)</td><td>Low: 1 model, 1 GPU (H20)</td></tr><tr><td>8-bit accuracy</td><td>Within 0.77 MMLU points of BF16</td><td>Moderate-high: 5 models, 2 publishers</td></tr><tr><td>INT4 accuracy</td><td>0.43-1.43 MMLU points; up to about 4 GSM8K Moderate: 3 models, 2 points</td><td>publishers</td></tr><tr><td>gpt-oss-20b engine</td><td>vLLM 2.93× SGLang at defaults</td><td>Low: 1 source, untuned</td></tr><tr><td>AMD, Gaudi speed; FP4 speedup</td><td>No in-band measurement</td><td>None</td></tr></table>

## 6 A reporting checklist

Several gaps above trace to details a source omitted rather than to experiments nobody ran. A quantized-serving claim answers the practitioner’s question if it reports: (1) one checkpoint ID and revision, used for both timing and accuracy; (2) engine, version, and every non-default flag; (3) GPU model and count; (4) input and output lengths, batch size or request rate, and whether throughput counts prompt tokens; (5) a BF16 baseline timed on the same stack; (6) harness, version, few-shot count, chat template, and reasoning mode, with � and standard errors; (7) the calibration data, confirmed disjoint from the evaluation sets. No source in this letter meets all seven: none reports standard errors, and only the Gemma card times the checkpoint it scores.

## 7 Limitations

The rows in Table 1 come from diferent experiments and are not one benchmark. Our standard errors assume independent questions and a single run. The Qwen FP8 checkpoint used for timing is quantized in blocks of 128 [28], but its activation scaling is not stated, so it cannot be matched to the static or dynamic accuracy rows. All web sources are vendor or project pages that can change; they were last checked on 5 October 2026.

## References

[1] Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J. Hewett, Mojan Javaheripi, Piero Kaufmann, James R. Lee, Yin Tat Lee, Yuanzhi Li, Weishung Liu, Caio C. T. Mendes, Anh Nguyen, Eric Price, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Xin Wang, Rachel Ward, Yue Wu, Dingli Yu, Cyril Zhang, and Yi Zhang. 2024. Phi-4 Technical Report. arXiv:2412.08905

[2] AMD. 2026. vLLM V1 Performance Optimization. ROCm documentation 7.1.0. Retrieved October 5, 2026 from https://rocm.docs.amd.com/en/docs-7.1.0/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html

[3] Mark Chen et al. 2021. Evaluating Large Language Models Trained on Code. arXiv:2107.03374

[4] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training Verifiers to Solve Math Word Problems. arXiv:2110.14168

[5] Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. 2022. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. In Advances in Neural Information Processing Systems, Vol. 35. Neural Information Processing Systems Foundation, Inc., 30318–30332. doi:10.52202/068431-2198

Manuscript submitted to ACM

[6] Abhinav Dutta, Sanjeev Krishnan, Nipun Kwatra, and Ramachandran Ramjee. 2024. Accuracy is Not All You Need. In Advances in Neural Information Processing Systems, Vol. 37. Neural Information Processing Systems Foundation, Inc., 124347–124390. doi:10.52202/079017-3950

[7] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2023. OPTQ: Accurate Quantization for Generative Pre-trained Transformers. In The Eleventh International Conference on Learning Representations (ICLR 2023). OpenReview.net. Also released as GPTQ, arXiv:2210.17323

[8] Elias Frantar, Roberto L. Castro, Jiale Chen, Torsten Hoefler, and Dan Alistarh. 2025. MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models. In Proceedings ofthe 30th ACM SIGPLAN Annual Symposium on Principles and Practice ofParallel Programming. ACM, New York, NY, USA, 239–251. doi:10.1145/3710848.3710871

[9] Gemma Team. 2025. Gemma 3 Technical Report. arXiv:2503.19786

[10] Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W. Mahoney, and Kurt Keutzer. 2022. A Survey of Quantization Methods for Eficient Neural Network Inference. In Low-Power Computer Vision: Improve the Eficiency of Artificial Intelligence. Chapman and Hall/CRC, Boca Raton, FL, USA, 291–326. doi:10.1201/9781003162810-13

[11] GPUStack. 2026. Optimizing GPT-OSS-20B Throughput on NVIDIA H100 GPUs. GPUStack Inference Performance Lab, documentation v2.2 Retrieved October 5, 2026 from https://docs.gpustack.ai/2.2/performance-lab/gpt-oss-20b/h100/

[12] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring Massive Multitask Language Understanding. In The Ninth International Conference on Learning Representations (ICLR 2021). OpenReview.net.

[13] Intel. 2026. Inference Model Performance Data for Intel Gaudi 3 AI Accelerators. Intel developer documentation. Retrieved October 5, 2026 from https://www.intel.com/content/www/us/en/developer/platform/gaudi/model-performance.htm

[14] Eldar Kurtic, Alexandre Marques, Shubhra Pandit, Mark Kurtz, and Dan Alistarh. 2025. “Give Me BF16 or Give Me Death”? Accuracy-Performance Trade-Ofs in LLM Quantization. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 26872–26886. doi:10.18653/v1/2025.acl-long.1304

[15] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Eficient Memory Management for Large Language Model Serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles. ACM, New York, NY, USA, 611–626. doi:10.1145/3600006.3613165

[16] Joonhyung Lee, Shmulik Markovich-Golan, Daniel Ohayon, Yair Hanani, Gunho Park, Byeongwook Kim, Asaf Karnieli, Uri Livne, Haihao Shen, Tai Huang, Se Jung Kwon, and Dongsoo Lee. 2025. Faster Inference of LLMs using FP8 on the Intel Gaudi. arXiv:2503.09975

[17] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. 2024. AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration. In Proceedings ofMachine Learning and Systems, P. Gibbons, G. Pekhimenko, and C. De Sa (Eds.), Vol. 6. 87–100.

[18] Andy Luo. 2025. Best Practices for Competitive Inference Optimization on AMD Instinct MI300X GPUs. ROCm blogs. Retrieved October 5, 2026 from https://rocm.blogs.amd.com/artificial-intelligence/LLM\_Inference/README.htm

[19] Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey, Richard Grisenthwaite, Sangwon Ha, Alexander Heinecke, Patrick Judd, John Kamalu, Naveen Mellempudi, Stuart Oberman, Mohammad Shoeybi, Michael Siu, and Hao Wu. 2022. FP8 Formats for Deep Learning arXiv:2209.05433

[20] Mistral AI. 2024. Mistral NeMo. Mistral AI news. Retrieved October 5, 2026 from https://mistral.ai/news/mistral-nemo

[21] Mistral AI. 2025. Ministral 3 14B Instruct 2512 BF16 Model Card. Hugging Face model card. Retrieved October 5, 2026 from https://huggingface.co/ mistralai/Ministral-3-14B-Instruct-2512-BF16

[22] NVIDIA. 2026. NVIDIA H100 Tensor Core GPU. NVIDIA data center product page. Retrieved October 5, 2026 from https://www.nvidia.com/en us/data-center/h100/

[23] OpenAI. 2025. gpt-oss-120b & gpt-oss-20b Model Card. arXiv:2508.10925

[24] PyTorch Team. 2025. AWQ-INT4 google/gemma-3-12b-it Model Card. Hugging Face model card. Retrieved October 5, 2026 from https: //huggingface.co/pytorch/gemma-3-12b-it-AWQ-INT4

[25] Qwen Team. 2024. Qwen2.5 Speed Benchmark. Qwen documentation, v2.5. Retrieved October 5, 2026 from https://qwen.readthedocs.io/en/v2.5 benchmark/speed\_benchmark.htm

[26] Qwen Team. 2024. Qwen2.5 Technical Report. arXiv:2412.15115

[27] Qwen Team. 2025. Qwen3 Technical Report. arXiv:2505.09388

[28] Qwen Team. 2025. Qwen/Qwen3-14B-FP8 Model Card. Hugging Face model card. Retrieved October 5, 2026 from https://huggingface.co/Qwen/ Qwen3-14B-FP8

[29] Qwen Team. 2025. Speed Benchmark. Qwen documentation. Retrieved October 5, 2026 from https://qwen.readthedocs.io/en/latest/getting\_started/ speed\_benchmark.html

[30] Red Hat AI. 2024. RedHatAI/Mistral-Nemo-Instruct-2407-FP8 Model Card. Hugging Face model card. Retrieved October 5, 2026 from https: //huggingface.co/RedHatAI/Mistral-Nemo-Instruct-2407-FP8

[31] Red Hat AI. 2024. RedHatAI/Mistral-Nemo-Instruct-2407-quantized.w4a16 Model Card. Hugging Face model card. Retrieved October 5, 2026 from https://huggingface.co/RedHatAI/Mistral-Nemo-Instruct-2407-quantized.w4a16

[32] Red Hat AI. 2024. RedHatAI/Qwen2.5-14B-Instruct-quantized.w8a8 Model Card. Hugging Face model card. Retrieved October 5, 2026 from https://huggingface.co/RedHatAI/Qwen2.5-14B-Instruct-quantized.w8a8

[33] Red Hat AI. 2025. RedHatAI/gemma-3-12b-it-FP8-dynamic Model Card. Hugging Face model card. Retrieved October 5, 2026 from https: //huggingface.co/RedHatAI/gemma-3-12b-it-FP8-dynamic

[34] Red Hat AI. 2025. RedHatAI/gemma-3-12b-it-quantized.w8a8 Model Card. Hugging Face model card. Retrieved October 5, 2026 from https: //huggingface.co/RedHatAI/gemma-3-12b-it-quantized.w8a8

[35] Red Hat AI. 2025. RedHatAI/phi-4-FP8-dynamic Model Card. Hugging Face model card. Retrieved October 5, 2026 from https://huggingface.co RedHatAI/phi-4-FP8-dynamic

[36] Red Hat AI. 2025. RedHatAI/phi-4-quantized.w4a16 Model Card. Hugging Face model card. Retrieved October 5, 2026 from https://huggingface.co RedHatAI/phi-4-quantized.w4a16

[37] Red Hat AI. 2025. RedHatAI/phi-4-quantized.w8a8 Model Card. Hugging Face model card. Retrieved October 5, 2026 from https://huggingface.co RedHatAI/phi-4-quantized.w8a8

[38] Red Hat AI. 2025. RedHatAI/Qwen3-14B-FP8-dynamic Model Card. Hugging Face model card. Retrieved October 5, 2026 from https://huggingface. co/RedHatAI/Qwen3-14B-FP8-dynamic

[39] Red Hat AI. 2025. RedHatAI/Qwen3-14B-NVFP4 Model Card. Hugging Face model card. Retrieved October 5, 2026 from https://huggingface.co RedHatAI/Qwen3-14B-NVFP4

[40] Red Hat AI. 2025. RedHatAI/Qwen3-14B-quantized.w4a16 Model Card. Hugging Face model card. Retrieved October 5, 2026 from https: //huggingface.co/RedHatAI/Qwen3-14B-quantized.w4a16

[41] Bita Darvish Rouhani et al. 2023. Microscaling Data Formats for Deep Learning. arXiv:2310.10537

[42] SemiAnalysis. 2026. InferenceX (formerly InferenceMAX). Open-source inference benchmark. Retrieved October 5, 2026 from https://inferencex semianalysis.com/about

[43] SGLang Project. 2026. Quantization. SGLang documentation. Retrieved October 5, 2026 from https://docs.sglang.io/docs/advanced\_features quantization

[44] Tianyao Shi and Yi Ding. 2025. Systematic Characterization of LLM Quantization: A Performance, Energy, and Quality Perspective. arXiv:2508.16712

[45] Tencent AngelSlim Project Contributors. 2025. AngelSlim/Qwen3-14B\_fp8\_static Model Card. Hugging Face model card. Retrieved October 5, 2026 from https://huggingface.co/AngelSlim/Qwen3-14B\_fp8\_static

[46] vLLM Project. 2026. Quantization. vLLM documentation. Retrieved October 5, 2026 from https://docs.vllm.ai/en/stable/features/quantization

[47] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. 2023. SmoothQuant: Accurate and Eficient Post-Training Quantization for Large Language Models. In Proceedings ofthe 40th International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 202). PMLR, 38087–38099. https://proceedings.mlr.press/v202/xiao23c.html

[48] Jiaqi Zhao, Ming Wang, Miao Zhang, Yuzhang Shang, Xuebo Liu, Yaowei Wang, Min Zhang, and Liqiang Nie. 2025. Benchmarking Post-Training Quantization in LLMs: Comprehensive Taxonomy, Unified Evaluation, and Comparative Analysis. arXiv:2502.13178

[49] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jef Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. 2024. SGLang: Eficient Execution of Structured Language Model Programs. In Advances in Neural Information Processing Systems, Vol. 37. Neural Information Processing Systems Foundation, Inc., 62557–62583. doi:10.52202/079017-2000

[50] Zixuan Zhou, Xuefei Ning, Ke Hong, Tianyu Fu, Jiaming Xu, Shiyao Li, Yuming Lou, Luning Wang, Zhihang Yuan, Xiuhong Li, Shengen Yan, Guohao Dai, Xiao-Ping Zhang, Yuhan Dong, and Yu Wang. 2024. A Survey on Eficient Inference for Large Language Models. arXiv:2404.14294