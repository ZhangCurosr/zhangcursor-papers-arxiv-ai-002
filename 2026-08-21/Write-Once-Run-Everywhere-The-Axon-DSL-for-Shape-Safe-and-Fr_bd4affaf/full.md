# Write Once, Run Everywhere: The Axon DSL for Shape-Safe and Framework-Agnostic LLM Architectures

Jacob Nielsen, Danial Namazifard, Lukas Galke Poech, Peter Schneider-Kamp

University of Southern Denmark Campusvej 55, 5230 Odense, Denmark jacn@imada.sdu.dk

## Abstract

The entire ecosystem of open-source language models effectively relies on a single platform. What if this platform was forced to shut down tomorrow? Implementing and maintaining efficient model definitions and translating them between different training and inference regimes is a resource-heavy task that severely limits model efficiency and portability, hindering both scaling and deployment. Here, we present Axon, a strongly typed domain-specific language with Haskell-like syntax, that enables a write-once, run everywhere paradigm for LLM architectures. By basing collaboration on a language specification rather than a specific framework's vision, Axon fosters open cooperation and empowers researchers to implement highly specialized architectures without giving up optimization infrastructure or accepting deployment lock-in. Axon allows for concise, auditable specifications that can be automatically compiled to standalone implementations for leading frameworks: PyTorch, PyTorch with Triton, JAX, MLX and vLLM. In 467 inference benchmarking experiments on models ranging from 135M to 32B parameters, we demonstrate median speedups of 7% on PyTorch, 12% on PyTorch with Triton, 91% on JAX, and 107% on MLX, compared to the reference implementations from Transformers. When deployed as native vLLM architectures with PagedAttention and KV-cache, Axon models achieve a 58% median speedup over Transformers implementations.

## Introduction

In 1972/73, Dennis Ritchie created the C programming language (Ritchie 1973) which – uniquely at the time – combined the hardware access of low-level programming languages with the portability and syntax of high-level languages. Undoubtedly, the C language has revolutionized how we program computers. The current situation in LLM development is reminiscent of the pre-C era. Some frameworks enable high levels of abstraction, such as the Transformers library (Wolf et al. 2020), while others are closer to hardware, such as CUDA kernels (Nickolls et al. 2008) or FlashAttention implementations (Dao et al. 2022). The de-facto standard implementations of LLM architectures are currently determined by rigid, monolithic platforms, most notably the Hugging Face Hub¹, which prioritize the ease of sharing models and model definitions over efficiency.

![](images/dcb2622ed510137f3d11d07dfd1b41985ce2b1989af2f96806869680a21d3c93.jpg)  
Figure 1: Write-once, run everywhere. Axon DSL compiles axon definitions (. axon) to standalone model definitions for PyTorch, JAX, MLX and vLLM. All that is needed is a standard safetensorscheckpoint.

This discrepancy between architectural transparency and execution efficiency creates a high barrier to porting models between software backends and hardware platforms: porting implementations between backends, such as PyTorch (Ansel et al. 2024), JAX (Bradbury et al. 2018), and MLX (Hannun et al. 2023), is error-prone and non-trivial, creating technical debt (Sculley et al. 2015).

Modeling code is hidden in a vast ecosystem of glue code and pipeline jungles connecting frameworks and hardware backends, freezing a system to the peculiarities of a specific package and making architectural improvements or backend transitions prohibitively expensive. Most current optimization efforts, constrained by this dependency chain, patch existing frameworks – leading to implementation drift (Sculley et al. 2015), where backend-specific optimizations are lost or omitted for compatibility with general framework conventions, and forcing researchers and practitioners to choose between (a) suboptimal performance or (b) redundantly rewriting implementations for every backend and hardware. Exacerbating this issue, the efforts of most practitioners and researchers lie in the hands of a few private companies, leading to undesirable centralization of power (Ahmed, Wahed, and Thompson 2023; Widder, Whittaker, and West 2024).

Here, we present Axon (Figure 1), a compact, strongly typed functional domain-specific language (DSL) that allows for minimal, standalone model definitions, and thereby reduces the dependency chains and abstraction debt (Sculley et al. 2015) that characterize monolithic frameworks: Axon targets model families rather than hand-written modelspecific implementations per backend, per hardware: lexical names, symbolic dimensions, parameter paths, optional configuration defaults, control-flow structure, and precise tensor types. Furthermore, Axon removes a major source of training/serving skew identified by Breck et al. (2017) as one of the most critical yet least-frequently implemented tests for production readiness. Axon further mitigates the fastintegration debt (Ehsani et al. 2026) inherent in rapid LLM adoption and empowers researchers to implement and share specialized architectures without sacrificing optimization infrastructure.

While manual porting between backends often leads to implementation drift – where optimizations for one framework are omitted or lost in another – Axon's write-once, run everywhere paradigm facilitates integrity across the entire model lifecycle. By formalizing the model specification as an auditable and unit-testable artifact, Axon directly supports the ML Test Score criteria for reproducibility and ensures that architectures are inherently ready for diverse production environments (Breck et al. 2017). Moreover, Axon enables a seamless generation of optimized implementations for both training and inference, currently targeting PyTorch, PyTorch with Triton, JAX, MLX, and vLLM, within a single compiler pipeline. Axon's compiler preserves the integrity of the symbolic dimensions and control-flow structures.

In sum, Axon contributes to democratizing highperformance AI, empowering smaller entities to achieve competitive throughput while enabling large-scale operators to optimize hardware utilization and reduce their environmental footprint. Our contributions are as follows:

• A strongly-typed domain-specific language, Axon, for authoring backend-agnostic neural (language) models for cross-backend materializations.

• A compiler pipeline for seamless compilations of optimized standalone implementations for over 204 models spanning 60 families.

• Benchmarks of these checkpoints across five backends displaying improved or at least competitive results in terms of throughput, while maintaining top-1 token parity across backends.

• A deployment path to multiple backends based on a checkpoint and the Axon model definition.

## Related Work

The ecosystem for optimizing deep neural networks spans high-level API abstractions to low-level hardware primitives. We broadly categorize existing work into five layers: (i) deep learning backends, (ii) intermediate representations, (iii) compilers and optimizers, (iv) kernel generation, and (v) specialized inference frameworks.

Deep Learning Backends. At the highest level, we have well known backends such as PyTorch (Ansel et al. 2024), TensorFlow (Abadi et al. 2015), and JAX (Bradbury et al. 2018) providing the tensor operations from which model definitions and their execution are built with the backward pass typically derived automatically. While PyTorch and TensorFlow mainly rely on eager execution, the industry has shifted towards just-in-time (JIT) compilations for performance. More recently, specialized backends, such as Tinygrad (Hotz and contributors 2020) and MLX (Hannun et al 2023), emphasize minimalism and hardware-specific optimizations to reduce overhead often introduced by the generalpurpose frameworks.

Intermediate Representations (IR). To map high level abstractions to hardware-specific code, IR frameworks maintain a structured representation of computational graphs. ONNX (Bai et al. 2019) serves as a cross-platform standard for model exchanges, while MLIR (Multi-level IR) (Lattner et al. 2021) and LLVM (Lattner and Adve 2004) provide modular infrastructure for dialect-specific optimizations. StableHLO (OpenXLA Community 2023) bridges JAX and TensorFlow, lowering high-level operations into a representation that compilers can efficiently optimize. IREE (IREE Authors 2019) extends this with an end-to-end pipeline from MLIR to executable code across diverse devices.

Compilers and Optimizers. Once represented in an IR, compilers optimize the graph through techniques such as operator fusion, memory planning, and constant folding. Accelerated Linear Algebra (XLA) (Sabne 2020) does this for JAX and TensorFlow, while the torchinductor is the primary backend for PyTorch 2.0 (Ansel et al. 2024). Hardware specific optimizers like NVIDIA's TensorRT (NVIDIA 2026) and Intel's OpenVINO (Intel Corporation 2026) maximize throughput for specific chipsets. Apache TVM (Chen et al. 2018) provides general-purpose automated optimization, leveraging machine learning to find efficient operator implementations.

Kernel Generation. The lowest level of optimization happens at the kernel level, where mathematical operations are mapped to hardware threads/warps in a tensorized formulation. Native libraries, such as NVIDIA's CUTLASS (Kerr et al. 2017) and cuDNN (Chetlur et al. 2014), and the crossplatform oneDNN (oneDNN Contributors 2017), provide highly tuned primitives. A fast-growing trend towards higherlevel, programmable kernel generation has followed: Triton (Tillet, Kung, and Cox 2019) compiles high-level Python to efficient GPU kernels, DITRON targets distribution-aware kernels (Zheng et al. 2026), and Mojo² bridges Python's usability with the performance of C++ and CUDA. LLMs now also power this space, e.g. as part of the AutoKernel project³.

Specialized Inference Frameworks. Specialized inference frameworks wrap prior layers of optimization for deployments and serving. vLLM increases serving throughput through PagedAttention, a virtual-memory-inspired KVcache manager (Kwon et al. 2023). For Apple Silicon, emerging projects like MLX-vLLM (Barrios 2025) and Rapid MLX (Chai 2026) adapt these high-performance serving techniques to the MLX ecosystem.

Summary. A common thread across these five layers is that each is specialized to a particular software or hardware stack Backends expose framework-specific model (tensor) APIs. IRs define machine-exchange formats rather than authoring languages, compilers optimize within a single stack, kernel generation targets specific hardware, and inference frameworks serve one runtime. Portability across stacks at any level is achieved by manually re-expressing implementations. Axon occupies a complementary layer above these: a humanauthored, strongly typed specification language with symbolic dimensions, from which a single Graph IR is lowered to standalone native implementations for PyTorch, Tritonaccelerated PyTorch, JAX, MLX, and vLLM. By enforcing one Graph IR contract consumed by every backend, Axon ensures that a definition-level optimization cannot be present in one framework's port yet silently omitted in another.

## The Axon Language

Axon is a compact functional language for describing neural networks independently of any particular implementation or runtime backend, producing standalone model definitions. A more detailed description of the language including Backus-Naur form is provided in the technical supplement.

Values. The first-class domain objects in Axon include tensors, symbolic dimensions, and paths into checkpoint statedicts and configurations. The language is intentionally designed to describe the mathematical structure of the neural network rather than the mechanics of a specific implementation in a particular language or backend.

Types. The tensor types carry shapes comprised of symbolic dimensions. A type such as x : : Tensor [B, S, D] states that a value x is a tensor BxSxD, representing a batch of sequences of hidden-state vectors. Symbolic dimensions are part of the static semantics, participating in type checking, primitive typing rules, and the Graph IR metadata. As a consequence, Axon definitions can express reusable shape relationships without hard-coding checkpoint-specific sizes. Symbolic dimensions can operate as values in expressions, reducing the retrieval of the sequence length of a tensor x from an expression such as x . shape [-2] to simply S.

Syntax. Axon's syntax is Haskell-inspired. Definitions use signatures, arrow types, optional parameters, pipes, and a dostyle statement form. As an example, consider a pre-norm attention sub-layer, in which the same normalized hidden state h is passed as query, key, and value, and @1n indicates that the RMS-norm parameters live under the current scope's 1n key in the checkpoint's state-dict.

```julia
block :: Tensor[B,S,D] -> ?Tensor[B,K] -> Tensor[B,S,D]
block x attn_mask = do
```

h <- NN.rmsnorm@ln x   
a <- Attention.attention h h h attn\_mask

Scoped parameter paths are one of several surface conveniences that will be normalized before flattening during compilation.

Functional semantics. Semantically, Axon definitions are functions over explicit or implicit arguments, configuration lookups, and parameter-path reads. A bind statement names the values produced by an expression. Except for ternary branch expressions, all expressions are eager, i.e., an argument is evaluated before the callee receives it as a value.

Model paths. Model parameters and config values are accessed through values of type Path. Relative path values are denoted using “@": NN.rmsnorm@ln from our example refers to a path scoped by the current lexical context. Absolute paths are denoted using “@@": NN.rmsnorm@@'layers.{i}.attn.ln' refers to checkpoint locations directly, with i being instantiated from the runtime context. Whenever possible, relative paths are normalized into absolute paths by the compiler.

Reusable libraries. Primitive Axon operations, such as the tensor operation \_where or parameter retrieval \_param, are implemented as internal definitions prefixed by an underscore. They are exposed through lightweight wrappers as part of reusable library modules, such as NN, Tensor, Attention,Masking,MoE,Cache,andPositions.In addition to exposing primitive operations, these modules define reusable higher-level generic model-independent operations such as Attention.attention and Cache.prepare.

Semantic core. The full Axon DSL language offers expressiveness through library imports, default arguments, nested expressions, path scoping, operation pipes, and other syntactic sugar. Full Axon DSL can be desugared to a small semantic core where library references, default arguments, and paths are resolved, and nested expressions are flattened and fully typed, at the subexpression level. This semantic core remains valid Axon DSL but has a one-to-one correspondence to Axon's Graph IR.

## The Axon Compiler

The Axon compiler comprises three main phases: (1) desugaring of full Axon DSL into its semantic core; (2) optimization at the Graph IR level; and (3) lowering of the Graph IR to backend-specific code. In the technical supplement, we walk an exemplary small definition through all phases.

Phase 1: Desugaring. The Axon DSL is defined by an EBNF grammar that is operationalized using Lark (Lark Contributors 2026). The parser transforms valid Axon DSL into an abstract syntax tree (Axon AST), which serves as the representation for Phase 1 and can be dumped to Axon DSL and reparsed. Every desugaring stage of Phase 1 receives an Axon AST and outputs an Axon AST, typically guaranteeing additional properties: (a) the resolver stage inlines library definitions imported and referenced, guaranteeing a closure property stating that all non-primitive operations are expressed as definitions in the AST; (b) the flattening stage flattens nested expressions, guaranteeing a flatness property stating that definitions consist of a flat sequence of operations; (c) the type checker stage infers types from type signatures, type ascriptions, and the type rules of primitive operations, guaranteeing a consistently typed Axon AST. The desugaring phase also contains dead code elimination, constant folding, and other straightforward code normalizations.

Phase 2: Graph optimizations. The closed flat typed Axon AST can be viewed as an instance of a static single assignment (SSA) graph (Cytron et al. 1991). In this phase, the Axon compiler converts this semantic core to the Graph IR, which is then iteratively optimized using a number of stages: (a) definition inlining replaces wrapper definitions and other low-complexity definitions by integrating their sequence of operations into their call sites, enabling, for example, the inlining of primitive wrapper definitions; (b) generic graph rewriting replaces known patterns of operations by equivalent patterns of operations based on matching of primitive operational provenance, enabling, for example, the replacement of loops iterating over experts in an MoE by a more efficient stacked-experts formulation; and (c) backend-specific graph rewriting replaces known patterns of operations by backend-specific primitives, enabling, for example, the use of optimized attention implementations and kernels.

Phase 3: Lowering. The optimized Graph IR contains definitions as assignment sequences, calls to definitions, and generic and backend-specific primitive operations. The lowering phase is backend-specific and transpiles the Graph IR into one of the currently targeted backends: PyTorch, Tritonaccelerated PyTorch, MLX, JAX, and vLLM. Each of the separate code generators needs to be able to implement the control flow and the primitive operators represented by the Graph IR. In addition, the code generators also need to implement forward and generate methods. To achieve competitive performance, compilation-oriented backends typically supply scaffolding that ensures static cache allocations.

## Results

We now present a series of results from our Axon-derived models across PyTorch, JAX, MLX and vLLM backends. The results will establish that Axon-derived models yield performance parity, and often outperform implementations in existing frameworks. We further show that Axon-derived models yield near-identical training behavior. All figures can be found in larger size in the technical supplement.

Experimental Setup MLX backend experiments on < 1 B models were executed on Apple Silicon M3 Max, 36GB RAM with MLX 0.32.0. We enabled compilation and allowed a max generation length of 256 tokens. PyTorch, Triton-enhanced PyTorch, JAX and vLLM experiments were conducted on an NVIDIA B200 GPU with 180 GB HBM3, using PyTorch 2.11.0, Triton 3.6.0,JAX0.10.2, and Transformers 5.10.0. dev0. Models that can natively be compiled without errors and all Axon models are compiled. All models ran with 1 generation-warmup step, 3 repetitions, and a max generation length of 128 tokens. During these experiments we exercise the generate () method in the models. Both the Axon-compiled model and the Transformers baseline use torch.compile and equivalent JIT compilations for Axon using the -compi le-axon, -compile-hf, -optimize-graph flags. The primary precision is bfloat16, with float32 fallback for pairs that fail top-1 token parity with the Transformers baseline.

![](images/2d0d24763a3febf7afb2dcbc2c60f3bada260239e7c7fb328eccfc2a9a806bdb.jpg)  
Figure 2: Autoregressive generation performance with decoder-only models with up to 4B parameters. Log-log plot where the diagonal line indicates equal performance. Points above the parity line indicate Axon is faster, color-fill denotes backends and border-line denotes dtype.

Measures On generation benchmarks, throughput is calculated from per-run log output as generated tokens/wall time, where generated tokens is the actual number of tokens produced before the end-of-sequence token, rather than the maximum generation length cap. All comparisons use a runtime ratio $\rho = t _ { \mathrm { A x o n } } / t _ { \mathrm { T r a n s f o r m e r s } }$ where $t _ { \mathrm { A x o n } }$ and $t _ { \mathrm { T r a n s f o r m e r s } }$ are the wall-clock times for the Axon-derived model and the reference (Transformers) implementation under identical inputs, precision, and compilation settings. A value $\rho < 1$ indicates that Axon is faster, while $\rho = 1$ represents parity.

## Sub-4B Models

We benchmark 91 checkpoints spanning 71 Axon model definitions up to 4B parameters, across 26+ architecture families including GPT, Llama, Mistral, Qwen, Gemma, SmolLM, T5, BART, BERT, and Mamba – measured across three backends (PyTorch, Triton, JAX). Of these, 68 are decoder-only language models tested via autoregressive generation (Figure 2), and 23 are encoder-only or encoder-decoder (seq2seq) models tested via forward passes (Table 2 and plots in the technical supplement).

Each autoregressive generation produces at least 8 tokens across the 225 checkpoints (200 BF16, 25 FP32), with a median of 123 tokens, where 91% generated at least 100 tokens. Nine checkpoints failed BF16 top-1 token parity due to precision noise and were re-run in FP32, achieving 100% top-1 token parity across all 303 checkpoints. Forward benchmarks are reported in wall-time (ms), as mean of 3 repeated measurements excl. a warm-up pass, covering 78 checkpoints across 23 encoder/seq2seq families (69 BF16, 9 FP32).

Table 1: Per-backend comparison of Axon's autoregressive generation performance with decoder-only models against PyTorch, Triton, and JAX. $^ { \mathrm { \scriptsize {  } } } \mathrm { A x o n } \leq 1 \times ^ { \mathrm { \scriptsize { , } } } =$ : count (and %) of checkpoints where Axon is faster than Transformers."Axon $> 1 \times \hat { \mathbf { \mu } } =$ count (%) of checkpoints where Axon is slower. Median and mean report runtime ratio (lower means Axon is faster).
<table><tr><td>Backend</td><td>Checkpoints</td><td>Axon ≤ 1×</td><td>Axon &gt; 1×</td><td>Median ratio</td><td>Mean ratio</td></tr><tr><td colspan="6">&lt;4B</td></tr><tr><td>PyTorch</td><td>76</td><td>48 (63%)</td><td>28 (37%)</td><td>0.903</td><td>0.925</td></tr><tr><td>Triton</td><td>75</td><td>60 (80%)</td><td>15 (20%)</td><td>0.843</td><td>0.878</td></tr><tr><td>JAX</td><td>74</td><td>64 (86%)</td><td>10 (14%)</td><td>0.481</td><td>1.122</td></tr><tr><td>&lt;4B total</td><td>225</td><td>172 (76%)</td><td>53 (24%)</td><td>0.804</td><td>0.974</td></tr><tr><td colspan="6">4-32B</td></tr><tr><td>PyTorch</td><td>87</td><td>55 (63%)</td><td>32 (37%)</td><td>0.987</td><td>0.980</td></tr><tr><td>Triton</td><td>87</td><td>67 (77%)</td><td>20 (23%)</td><td>0.924</td><td>0.931</td></tr><tr><td>JAX</td><td>68</td><td>55 (81%)</td><td>13 (19%)</td><td>0.589</td><td>0.981</td></tr><tr><td>4-32B total</td><td>242</td><td>177 (73%)</td><td>65 (27%)</td><td>0.908</td><td>0.963</td></tr></table>

Table 2: Per-backend comparison of Axon's Forward performance with Encoder-Only and Encoder-Decoder Models against PyTorch, Triton, and JAX backends. “Axon $\leq 1 \times \ " = \mathrm { c o u n t }$ (and %) of checkpoints where Axon is faster than Transformers. “Axon $> 1 \times \warrow =$ count (%) of checkpoints where Axon is slower. Median and mean report runtime ratio (lower means Axon is faster).
<table><tr><td>Backend</td><td>Checkpoints</td><td>Axon ≤ 1×</td><td>Axon &gt; 1×</td><td>Median ratio</td><td>Mean ratio</td></tr><tr><td colspan="6">≤4B</td></tr><tr><td>PyTorch</td><td>26</td><td>15 (58%)</td><td>11 (42%)</td><td>0.883</td><td>1.786</td></tr><tr><td>Triton</td><td>26</td><td>9 (35%)</td><td>17 (65%)</td><td>1.332</td><td>2.169</td></tr><tr><td>JAX</td><td>26</td><td>7 (27%)</td><td>19 (73%)</td><td>1.079</td><td>1.260</td></tr><tr><td>≤4B total</td><td>78</td><td>31 (40%)</td><td>47 (60%)</td><td>1.084</td><td>1.738</td></tr><tr><td colspan="6">4-32B</td></tr><tr><td>PyTorch</td><td>20</td><td>13 (65%)</td><td>7 (35%)</td><td>0.990</td><td>1.506</td></tr><tr><td>Triton</td><td>20</td><td>8 (40%)</td><td>12 (60%)</td><td>1.133</td><td>1.546</td></tr><tr><td>JAX</td><td>14</td><td>0 (0%)</td><td>14 (100%)</td><td>3.747</td><td>3.356</td></tr><tr><td>4–32B total</td><td>54</td><td>21 (39%)</td><td>33 (61%)</td><td>1.175</td><td>2.001</td></tr></table>

Autoregressive Axon-derived models are comparable or faster than Transformers for 76% of the 225 checkpoints, with a median speed ratio of 0.804× and a mean of 0.974×—a typical Axon model is 24% faster. We report a per-backend breakdown in Table 1. Most checkpoints run on all three backends; a small number did not complete on every backend, yielding 76 PyTorch, 75 Triton, and 74 JAX generate checkpoints (bloomz-560m generated fewer than 5 tokens and was excluded; BlackMamba-2.8B did not complete on JAX). JAX achieves the best overall generate throughput with a median of 0.481× and 86% at or below parity, though it also has the highest variance with a mean of 1.122×, driven by outliers on small BLOOM and XGLM models. Triton is the most consistent with 80% at or below parity and a mean of 0.878×. PyTorch is competitive but has more outliers above 1 ×. For encoder-only and encoder-decoder models (Table 2), performance is generally weaker, particularly for the T5 and mT 5 encoder-decoder models. The worst forward outliers are t5-small on Triton (9.2×), t5-base on PyTorch (4.1×), and mt5-large on Triton (3.7×). PyTorch is the best forward backend with 58% at or below parity and a median of 0.883×, while Triton and JAX struggle with the short forward pass.

![](images/c321f9577a1bb298521589c5cee2fbdf8f0a5415a337d27373c2a8e55adc8f4c.jpg)  
Figure 3: Benchmarking Autoregressive models between 4B and 32B models. Log-log plot where the diagonal parity line indicates equal performance. Generally, Axon produces comparable or superior throughput on the bigger models.

## Models between 4B and 32B

We benchmark 93 checkpoints spanning 47 Axon model definitions between 4B and 32B parameters, yielding 242 generate checkpoints (205 BF16, 37 FP32; Figure 3, Table 1) and 54 forward-pass checkpoints (Table 2, figure in technical supplement). Most checkpoints run on all three backends; JAX coverage is narrower at 68 generate checkpoints, as some models exceeding 20B parameters did not complete due to memory constraints – we only benchmarked models fitting within a single GPU.

Each generate run produced at least 8 tokens, with a median of 122 tokens, where 94% generated at least 100 tokens. 16 pairs failed BF16 top-1 parity across nine unique checkpoints, of which eleven were fixed by FP32. Flex-code-2x7B-1T (JAX), Flex-news-2x7B-1T (all backends), and Gemma–7b (JAX) did not meet the top-1 token parity criterion for all tested prompts. Forward benchmarks cover 54 checkpoints across 9 seq2seq families, reported in wall-time (ms) as the mean of 3 repeated measurements excluding 1 warm-up pass.

In Figure 3, Axon is at or below Transformers’ latency for 73% of checkpoints with a median speed of 0.908× and a mean of 0.963×. We report a per-backend breakdown for generative models in Table 1. JAX achieves the best generation throughput with a median of 0.589× and 81% at or below parity, though with higher variance than in the ≤4B tier. Triton is the most consistent at 77% at or below parity. PyTorch is competitive with 63% at or below parity. Notably, DeepSeek-MoE (1 6B) runs 7-14× faster across Axon backends, vastly outperforming Transformers' implementation. The worst generate outliers are concentrated on JAX for XGLM models: xg1m−7.5B and xg1m−4.5B.

![](images/0483317828c5e85ce958e69207c413e6210c2fa8f66e4955d504fc3f5fac68a4.jpg)  
Figure 4: vLLM: Axon (vLLM native) vs. Transformers generation throughput on 88 checkpoints. Points above the parity line indicate Axon is faster. 74% fall above.

Table 3: Autoregressive generation comparison of Axon (vLLM native) against Transformers. “Axon $\leq 1 \times \warrow =$ number (and %) of checkpoints where Axon is faster than Transformers. "Axon $> 1 \times \warrow =$ number (and %) of checkpoints where Axon is slower than Transformers. Median and mean report Axon-to-Transformers runtime ratio (lower is faster).
<table><tr><td>Model size</td><td>Checkpoints</td><td>Axon ≤ 1×</td><td>Axon &gt; 1×</td><td>Median ratio</td><td>Mean ratio</td></tr><tr><td>Small (≤4B)</td><td>54</td><td>38 (70%)</td><td>16 (30%)</td><td>0.656</td><td>1.430</td></tr><tr><td>Large (4B–32B)</td><td>34</td><td>27 (79%)</td><td>7 (21%)</td><td>0.609</td><td>2.889</td></tr><tr><td>Total</td><td>88</td><td>65 (74%)</td><td>23 (26%)</td><td>0.631</td><td>1.994</td></tr></table>

Forward performance is weak across all backends, particularly for large encoder-decoder models like $\mathrm { T } 5 - \mathrm { 1 1 } \mathrm { B } ,$ MT5-XXL, t 5gemma where the encoder forward path has high kernel launch overhead. JAX is universally slower on forward (100% above parity, median 3.747×). PyTorch is the best forward backend with 65% at or below parity and a median of 0.990×.

## vLLM

In Figure 4 and Table 3, we report on Axon derived native vLLM models. We test 88 checkpoint-backend pairs covering models from 135 M to 30 B parameters. Axon-compiled vLLM is faster than HF Transformers in 65 of 88 cases (74%), with a median runtime ratio of 0.631. At the median, Axon completes generation in 63% of the time taken by HF Transformers (1.6× speedup).

## MLX

In Figure 5, we report the ratios between Axon and Transformers for the MLX backend (MPS). All 21 models have 100% parity in FP32, but 5 models have mismatches in BF16: bloom-560m, smollm-135m, smollm2-135m, mt 5-smal1, mt 5-base. Axon is at or below Transformers' latency for 94.4% of the 126 checkpoints, with a median runtime ratio of 0.483× and a mean of 0.544× (Table 4). The decoder-only models achieve the best throughput with 98% at or below parity. The fast forward path using MLX's fast module with Metal kernels gives Monad and Gemma3–270M the largest speedups, at roughly 4.5× faster. We see that performance narrows at longer contexts.

![](images/154f6c913abfe89f7269e717da21a6f7223548d0daa0ad35fbc9d4996bb3d243.jpg)  
Figure 5: MLX. Benchmarking conventional HF models against Axon derived standalone model definitions. Axon yields some considerable speed-ups across the board.

Table 4: Breakdown of Axon performance by precision, model type, and sequence length. “Axon $\leq 1 \times \warrow \stackrel { \cdot } { = }$ number (and %) of checkpoints where Axon is faster than Transformers. "Axon $> 1 \times \ '$ = number (and %) of checkpoints where Axon is slower than Transformers. Median and mean report Axon-to-Transformers runtime ratio (lower = Axon is faster).
<table><tr><td>Group</td><td>Points</td><td>Axon ≤ 1×</td><td>Axon &gt; 1×</td><td>Median ratio</td><td>Mean ratio</td></tr><tr><td>BF16</td><td>63</td><td>61 (97%)</td><td>2 (3%)</td><td>0.423</td><td>0.500</td></tr><tr><td>FP32</td><td>63</td><td>58 (92%)</td><td>5 (8%)</td><td>0.503</td><td>0.588</td></tr><tr><td>causal_lm</td><td>84</td><td>82 (98%)</td><td>2 (2%)</td><td>0.390</td><td>0.437</td></tr><tr><td>seq2seq_lm</td><td>42</td><td>37 (88%)</td><td>5 (12%)</td><td>0.574</td><td>0.752</td></tr><tr><td>len=64</td><td>42</td><td>42 (100%)</td><td>0 (0%)</td><td>0.417</td><td>0.448</td></tr><tr><td>len=128</td><td>42</td><td>39 (93%)</td><td>3 (7%)</td><td>0.483</td><td>0.525</td></tr><tr><td>len=256</td><td>42</td><td>38 (90%)</td><td>4 (10%)</td><td>0.494</td><td>0.652</td></tr><tr><td>Total</td><td>126</td><td>119 (95%)</td><td>7 (5%)</td><td>0.483</td><td>0.544</td></tr></table>

## Training with Axon Derived Models.

We conducted experiments on the PyTorch backend, comparing with the Transformer's implementation as a reference. We experiment with Gemma-3 270M on the ArXiv Summarization dataset⁴, because it is a well known task. The experiments are carried out on an NVIDIA RTX A6000 GPU, with a batch size of 4 and a learning rate of 1e — 4. We report the loss curve for 2000 steps in the technical supplement. We compiled both models with torch. compile with 61s and 42s first step time for Axon and Transformers models, respectively, but with average step-time on 120.3ms for Axon against 133.0ms for Transformers, making Axon 9.6% faster. Both models had a final loss of 2.506, with a numerical instability of $3 . 7 4 \times 1 0 ^ { - 4 }$ . The only modification on the Axonderived PyTorch model was enabling requires\_grad.

# Discussion

The C language resolved a tension between hardware access and portability by fixing a single specification that compilers could lower to many machines. Axon targets an analogous situation in LLM development, where architectural transparency and dependency chains are locked inside monolithic frameworks. Across five runtime backends, we have demonstrated the potential of the Axon language under the “write once, run everywhere"-principle: a single specification achieves competitive execution times on over 204 model checkpoints across 60 model families.

Moreover, across 225 autoregressive checkpoints up to 4B and 242 checkpoints between 4B and 32B, Axon-derived models are as fast or faster than the Transformers baseline for 76% and 73% of the checkpoints, with median runtime ratios of 0.804× and 0.908×, respectively. On the MLX backend on Apple Silicon, Axon reaches 94.4% at or below parity in inference latency, median of 0.483× (twice as fast) over 126 checkpoints. These numbers indicate that a single . axon definition is competitive across five software runtimes and two hardware vendors.

We achieve top-1 token parity on almost all models, though some required an FP32 fallback. Axon-derived models generally provide comparable or better performance across backends, though some remain slightly slower than the baseline. We believe future maturing of the compiler, especially the lowering mechanisms, will close this gap. We suspect kernellaunch overhead dominates the relatively short generation time for these slower models. See technical supplement.

Portability and Shape Safety. A single Axon definition yields successful results across four backends from one Graph IR contract. Because all four backends are generated from one shared IR rather than maintained as independent handwritten ports, a definition-level optimization cannot be omitted in one framework's implementation while present in another. Remaining differences are pushed down to backendspecific lowering and kernel execution. The shape safety promised by Axon's strong type system is not directly measured by the throughput benchmarks, but the results are evidence of it: top-1 token parity across backends with distinct tensor libraries and kernel implementations is only possible because symbolic dimensions and the tensor-type contract are preserved end-to-end. From surface Axon through the Graph IR to each backend, a shape mismatch rejected by the typechecker never reaches execution.

Axon for Training. We demonstrate a training scheme for Axon-derived models: a model derived to PyTorch exhibits the same training behavior as the corresponding Transformers implementation. Already on a 270M model the Axon model yielded roughly 10% faster step-times. We expect the impact to be larger on bigger models, and leave this for future work. Axon therefore supports both training and inference from the same Graph IR contract, making it possible to train on one backend and serve on another.

Limitations. This work has several limitations: Although Axon operates on tensor level and facilitates arbitrary neural network architectures, our experiments are currently limited to language models. The main autoregressive and forward benchmarks are done on a single GPU type. MLX results are likewise on a single Apple Silicon configuration. Hardware variance is therefore not characterized. Some checkpoints required the FP32 fallback to meet top-1 token parity, indicating room to improve code generation for specific operations. Forward performance is weak across backends particularly for large encoder-decoder models, calling for future work. We present no ablation isolating the AST and Graph IR optimization passes, making it hard to attribute speed-ups to specific compiler stages. Lastly, we have experimented with using LLM-based coding assistants to convert Transformers model definitions to Axon; we carefully verified functional validity, though the code structure and maintainability of a few of the LLM-generated Axon definitions could be improved. Nevertheless, we regard it as promising that LLMs are able to generate functionally valid code in our newly proposed Axon language.

Practical Impact. LLMs are usually formulated as a mixture of Python modules and classes, checkpoint-specific configuration conventions and tensor state dictionaries. Axon changes that, enabling a platform for open collaboration with easily hackable, inspectable implementations free of dependency chains controlled by a framework's vision. This enables both cross-platform and platform-specific optimization in the search for maximum performance per Watt and per byte of VRAM. We hope this better democratizes and secures the community's contributions than the current state, learning and applying important lessons from the history of computer science and the story of the C language.

## Conclusion

We have introduced the Axon DSL, demonstrating that shape-safe, framework-agnostic compilation from Axon definitions to standalone implementations is a viable path toward open cooperation and greater freedom for researchers and practitioners. Axon achieves better or competitive throughput on most models across PyTorch, Triton-enhanced PyTorch JAX, MLX, and vLLM backends.

Future Work This work opens several directions: (i) natively distributed tensors for finer control over communication, (ii) third-party kernel injection from projects such as Unsloth⁵ and Liger-kernel6 as opt-in graph optimizations, and (iii) additional backends such as Vulkan and CoreML, extending write-once, run-everywhere reach to edge devices.

## References

Abadi, M.; Agarwal, A.; Barham, P.; Brevdo, E.; Chen, Z.;  
Citro, C.; Corrado, G. S.; Davis, A.; Dean, J.; Devin, M.;

⁵https://github.com/unslothai/unsloth

Ghemawat, S.; Goodfellow, I.; Harp, A.; Irving, G.; Isard, M.; Jozefowicz, R.; Jia, Y.; Kaiser, L.; Kudlur, M.; Levenberg,

K.; Tucker, P.; Vanhoucke, V.; Vasudevan, V.; Viégas, F.; Vinyals, O.; Warden, P.; Wattenberg, M.; Wicke, M.; Yu, Y.; and Zheng, X. 2015. TensorFlow, Large-scale machine learning on heterogeneous systems.

Ahmed, N.; Wahed, M.; and Thompson, N. C. 2023. The Growing Influence of Industry in AI Research. Science, 379(6635): 884–886.

Ansel, J.; Yang, E.; He, H.; Gimelshein, N.; Jain, A.; Voznesensky, M.; Bao, B.; Bell, P.; Berard, D.; Burovski, E.; et al. 2024. Pytorch 2: Faster machine learning through dynamic python bytecode transformation and graph compilation. In Proceedings of the 29th ACM international conference on architectural support for programming languages and operating systems, volume 2, 929–947.

Bai, J.; Lu, F.; Zhang, K.; et al. 2019. ONNX: Open Neural Network Exchange. https://github.com/onnx/onnx.

Barrios, W. 2025. vllm-mlx: Apple Silicon MLX Backendfor vLLM. https://github.com/waybarrios/vllm-mlx.

Bradbury, J.; Frostig, R.; Hawkins, P.; Johnson, M. J.; Katariya, Y.; Leary, C.; Maclaurin, D.; Necula, G.; Paszke A.; VanderPlas, J.; Wanderman-Milne, S.; and Zhang, Q. 2018. JAX: composable transformations of Python+NumPy programs.

Breck, E.; Cai, S.; Nielsen, E.; Salib, M.; and Sculley, D. 2017. The ML test score: A rubric for ML production readiness and technical debt reduction. In 2017 IEEE international conference on big data (big data), 1123–1132. IEEE.

Chai, R. 2026. Rapid-MLX: The fastest local AI engine for Apple Silicon. https://github.com/raullenchai/Rapid-MLX.

Chen, T.; Moreau, T.; Jiang, Z.; Zheng, L.; Yan, E.; Shen, H.; Cowan, M.; Wang, L.; Hu, Y.; Ceze, L.; et al. 2018. {TVM}: An automated {End-to-End} optimizing compiler for deep learning. In 13th USENIX symposium on operating systems design and implementation (OSDI 18), 578–594.

Chetlur, S.; Woolley, C.; Vandermersch, P.; Cohen, J.; Tran, J.; Catanzaro, B.; and Shelhamer, E. 2014. cudnn: Efficient primitives for deep learning. arXiv preprint arXiv:1410.0759.

Cytron, R.; Ferrante, J.; Rosen, B. K.; Wegman, M. N.; and Zadeck, F. K. 1991. Efficiently Computing Static Single Assignment Form and the Control Dependence Graph. ACM Transactions on Programming Languages and Systems, 13(4): 451–490.

Dao, T.; Fu, D.; Ermon, S.; Rudra, A.; and Ré, C. 2022. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in neural information processing systems, 35: 16344–16359.

Ehsani, R.; Rawal, S.; Cai, Y.; and Chatterjee, P. 2026. Faster Code, Deeper Debt? A Multivocal Literature Review on Technical Debt and Its Early Signs in LLM-Assisted Software Development. ACM Transactions on Software Engineering and Methodology.

Hannun, A.; Digani, J.; Katharopoulos, A.; and Collobert, R. 2023. MLX: Efficient and flexible machine learning on Apple silicon.

Hotz, G.; and contributors. 2020. tinygrad: A simple and powerful neural network framework.

Intel Corporation. 2026. Intel Distribution of Open-VINO Toolkit. https://www.intel.com/content/www/us/en/ developer/tools/openvino-toolkit/overview.html. Accessed: 2026-07-11.

IREE Authors. 2019. IREE: Intermediate Representation Execution Environment. https://github.com/iree-org/iree. Accessed: 2026-07-20.

Kerr, A.; Merrill, D.; Demouth, J.; Tran, J.; and NVIDIA CUTLASS Contributors. 2017. CUTLASS: CUDA Templates for Linear Algebra Subroutines. https://github.com/ NVIDIA/cutlass. Accessed: 2026-07-20.

Kwon, W.; Li, Z.; Zhuang, S.; Sheng, Y.; Zheng, L.; Yu C. H.; Gonzalez, J.; Zhang, H.; and Stoica, I. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, 611–626.

Lark Contributors. 2026. Lark Documentation. https://larkparser.readthedocs.io/. Accessed: 2026-07-29.

Lattner, C.; and Adve, V. 2004. LLVM: a compilation framework for lifelong program analysis & transformation. In International Symposium on Code Generation and Optimization, 2004. CGO 2004., 75–86.

Lattner, C.; Amini, M.; Bondhugula, U.; Cohen, A.; Davis, A.; Pienaar, J.; Riddle, R.; Shpeisman, T.; Vasilache, N.; and Zinenko, O. 2021. MLIR: Scaling Compiler Infrastructure for Domain Specific Computation. In 2021 IEEE/ACM International Symposium on Code Generation and Optimization (CGO), 2–14.

Nickolls, J.; Buck, I.; Garland, M.; and Skadron, K. 2008. Scalable Parallel Programming with CUDA. ACM Queue, 6(2): 40–53.

NVIDIA. 2026. NVIDIA TensorRT. https://developer. nvidia.com/tensorrt. Version 11.

oneDNN Contributors. 2017. oneDNN: oneAPI Deep Neural Network Library. https://github.com/uxlfoundation/ oneDNN. Accessed: 2026-07-20.

OpenXLA Community. 2023. StableHLO: Backward Compatible ML Compute Opset Inspired by HLO/MHLO. https: //github.com/openxla/stablehlo. Accessed: 2026-07-20.

Ritchie, D. M. 1973. C reference manual. Unpublished memorandum, Bell Telephone Laboratories.

Sabne, A. 2020. XLA : Compiling Machine Learning for Peak Performance.

Sculley, D.; Holt, G.; Golovin, D.; Davydov, E.; Phillips, T.; Ebner, D.; Chaudhary, V.; Young, M.; Crespo, J.-F.; and Den-

nison, D. 2015. Hidden technical debt in machine learning systems. Advances in neural information processing systems 28.

Tillet, P.; Kung, H. T.; and Cox, D. 2019. Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations. In MAPL.

Widder, D. G.; Whittaker, M.; and West, S. M. 2024. Why open'AI systems are actually closed, and why this matters. Nature, 635(8040): 827–833.

Wolf, T.; Debut, L.; Sanh, V.; Chaumond, J.; Delangue, C.; Moi, A.; Cistac, P.; Rault, T.; Louf, R.; Funtowicz, M.; Davison, J.; Shleifer, S.; von Platen, P.; Ma, C.; Jernite, Y.; Plu, J.; Xu, C.; Scao, T. L.; Gugger, S.; Drame, M.; Lhoest, Q.; and Rush, A. M. 2020. Transformers: State-of-the-Art Natural Language Processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 38–45. Online: Association for Computational Linguistics.

Zheng, S.; Zheng, X.; Sun, H.; Hou, Q.; Bao, W.; Li, S.; Duanmu, H.; Fang, J.; Xue, C.; Huang, C.; et al. 2026. DITRON: Distributed Multi-level Tiling Compiler for Parallel Tensor Programs. In Forty-third International Conference on Machine Learning.

## Technical Supplement

## Terminology

Throughout the paper we distinguish three levels of granularity. An Axon module is a single . axon source file with a number of Axon definitions, typically describing a model family or a model for a particular checkpoint, for example generic-1l ama3. axon. A checkpoint is one set of pretrained weights that a definition can be materialized against, for example met a-11 ama/L1 ama-3.2-1B; a single definition typically covers several checkpoints, which it declares through a CHECKPOINTS pragma. A benchmark run is one measurement of one checkpoint on one backend at one precision, and, for the MLX experiments, at one generation length. Because each checkpoint is measured on several backends, the per-backend counts reported in the tables exceed the number of distinct checkpoints.

## Extended Results

This section reproduces the encoder-only and encoderdecoder benchmarks, which are summarized in the main text, and provides full-width versions of the remaining benchmark in Figures 8, 9, 10, 11, 12, and 13.

## The Axon Language

Design principles. The Axon DSL was designed according to five main design principles.

1. Tensors and shapes are rst-class citizens: Neural models and other components of state-of-the-art models resolve around the manipulation of tensors, i.e., potentially multidimensional arrays of values. We implemented tensors and their shapes as first-class citizens through a buildin tensor type and a dimension type, that acts both as symbolic dimension of a tensor shape in the type signature and as a referencable value in the executable part of the definition.

2. Expressive but intuitive syntax: Axon definitions should be expressive enough to be concise and intuitive enough to be readable. To this end, we adopted a Haskell-inspired syntax with do-style command sequences and pipe operators as intuitive surface representations of single-static assignment functional programs over tensors.

3. Clear denotational and operational semantics: A clear denotational and operational semantics allows to determine statically and dynamically what code means and how it will behave. We chose functional definitions as these rule out side effects and avoid hard-to-follow orthogonal and multi-layered abstractions common in many ML frameworks and backends.

4. Backend-agnostic primitive operations: The primitive operations should be backend-agnostic in the sense that most or all of them should be implementable for all the actual and potentially targetted backends. To this end, we avoided the use of primitive operations that only exist in one backend. The naming of the primitive operations does follow PyTorch naming in many instances simply because we assume this to be what most ML developers are at least somewhat acquainted with.

5. Balanced abstraction level: The abstract level should be balanced in the sense that modules remain relatively concise and high-level enough to understand but simultaneously are based on low-level primitive operations that can efficiently be implemented when lowering to backend code. We achieved this to mostly through refactoring similar code in reusable built-in libraries that hide the low-level operations from typical use of the high-level concepts such as attention or positional embeddings.

Implementation principles. The implementation of the Axon DSL follows three implementation principles.

1. Stable reusable IRs: All stages of the first phase of the compiler operate on the same canonical Axon AST, i.e., the representation of parsed, resolved, normalized, flattened, and typed programs differs only by invariants and metadata of the Axon AST rather than having different representations at each stage. Any Axon AST can be serialized as an Axon module.

All stages of the second phase operate on the same canonical Graph IR, i.e., rwrites and optimizations applied at the graph level operate within the semantic closure of the Graph IR. Any Graph IR can be serialized as an Axon module.

2. Stage-local contracts: Each stage owns a narrow contract, allowing validators and round-trip tests to target individual compiler properties. For example, name resolution should not infer tensor types.

As an invariant, each stage should be stable under roundtrips, i.e., serializing the results of a compiler stage, reparsing the resulting Axon module, and running it through the preceding stages should result in an identical Axon AST or Graph IR. This invariant is tested by serializing this IR and comparing it to the original serialization.

3. Limited model-specific code in shared modules: All operations needed to implement a model including highly model-specific quirks should be defined as Axon definitions in the respective Axon module. If an Axon definition can be reused by different modules, it can be promoted to a shared Axon module, either placed locally or as part of the built-in libraries.

## Built-In Libraries

Axon includes built-in libraries, which consist of Axon modules that cover the operations needed by most modern LLM architectures. Authors simply import reusable Axon definitions for standard layers, tensor manipulation, mathematics, masking, positions, caching, activations, state-space opertions, and mixture-of-expert logic. As these built-ins are themselves written in Axon, they are inspectable, type checked, and lowered through the same compiler pipeline as user-authored code. To be precise, the definitions form these built-in Axon modules are resolved and integrated into the main Axon module in the resolve stage such that later stages see a self-contained Axon module.

Standards layers, parameter, and configuration modules. The most central built-in modules are NN, Params, and Config. NN provides parameterized neural-network layers such as embedding, linear, layernorm, and rmsnorm. Params exposes explicit checkpoint reads, while Config provides typed access to configuration values. Listing 1 shows the typical authoring style.

Listing 1: Core built-ins for layers, parameters, and configuration.

import Config   
import NN   
import Params (param)   
embed\_block@tok :: Tensor[B,S] -> Tensor[B,S,D]   
embed\_block ids = do   
hidden\_dim <- Config.dim @@hidden\_size   
x <- NN.embedding@tok ids dim=hidden\_dim   
scale <- param @ln.weight   
x <- NN.rmsnorm@ln x   
return x \* scale

Listing 1 illustrates the division of labor: configuration is read through Config, checkpoint tensors through Params, and backend-agnostic layers through NN.

Tensor and math primitives. Lower-level tensor and scalar functionality lives in Tensor and Math. These modules provide reshaping, slicing, matrix multiplication, broadcasting, comparison, reductions, and elementary scalar functions. Listing 2 shows a small helper built directly from these primitives.

Listing 2: Tensor and math built-ins.

```python
import Math (sqrt)
import Tensor
scale_heads :: Tensor[B,H,S,HD] -> Tensor[B,H,S,HD]
scale_heads x = do
factor <- 1.0 / sqrt HD
y <- Tensor.reshape x shape=[B, H, S, HD]
return y * factor
```

Listing 2 is minimal, but the same modules also provide frequently used operators such as Tensor . transpose,

Tensor.concat,Tensor.slice,Tensor.softmax, Tensor.where, Math.exp, and Math.clamp.

Attention and masking. Transformer-specific definitions are packaged in Attention and Masking. These builtin modules cover head reshaping, grouped-QKV splitting, scaled dot-product attention, additive masking, groupedquery attention, and causal or bidirectional mask construction. Listing 3 shows the standard self-attention pattern.

Listing 3: Attention and masking built-ins.

```python
import Attention
import Masking
self_attend :: Tensor[B,S,D] -> ?Tensor[B,K] ->
→ Tensor[B,S,D]
self_attend x attn_mask = do
q <- Attention.reshape_heads x heads=H
k <- Attention.reshape_heads x heads=H
v <- Attention.reshape_heads x heads=H
keep <- Masking.causal_mask q k padding_mask=attn_mask
y <- Attention.attention q k v keep
return Attention.merge_heads y
```

Listing 3 is representative of authored transformer code: model definitions express the architecture-level flow, while the builtin library packages the tensor mechanics behind typed interfaces.

Positions and cache management. Autoregressive decoding depends on two additional builtin families. Positions computes position ids and rotary or relative-position ingredients, while Cache manages null-safe key-value caching across decode steps. Listing 4 shows the common preparation pattern.

Listing 4: Position and cache built-ins for decoding.

```haskell
import Cache
import Positions
decode_prep :: Tensor[B, S] -> ?Cache[B,H,T,C, DH] -> ?
↔ Tensor[B, K] -> ?Bool ->
(Tensor[B,S], Tensor[B,S], ?Cache[B,H,T,C,DH])
decode_prep ids past attn_mask use_cache = do
work_cache <- Cache.prepare past use_cache S
past_len <- Cache.past_length past
pos <- Positions.position_ids ids attn_mask
→ past_length=past_len
return ids, pos, work_cache
```

Listing 4 captures the intended split: Positions supplies index structure, Cache threads state, and the backend only executes the lowered tensor program.

There are two Cache built-in modules, the default one, which represents the the dynamic cache state as a growing list of pairs of tensors, and a st atic one, which pre-allocates a statically-sized cache. Both Cache built-in modules share the same interface, such that switching simply requires selecting the static overlay. Depending on the target framework, this selection defaults to dynamic vs static for Pytorch vs JAX/MLX but can be overriden.

Activations and specialized modules. Beyond the core layers, Axon ships higher-level reusable libraries for common model families. Activations provides functions such as gelu, silu, and swiglu. MoE provides routing and grouped expert helpers for mixture-of-experts models, and SSM provides Mamba-style state-space building blocks. Listing 5 gives two representative calls.

Listing 5: Activation and specialized built-ins.

import Activations (silu)   
import MoE   
activate x = silu x   
route\_tokens@gate :: Tensor[B,S,D] ->   
→ (Tensor[B,S,TOPK], IdxTensor[B,S,TOPK])   
route\_tokens x =   
MoE.softmax\_topk\_router@gate x top\_k=TOPK

Listing 5 shows the two ends of the spectrum: small reusable nonlinearities and larger architecture-specific helpers. The builtin boundary is therefore not limited to primitive operations; it also captures recurring structural patterns that should remain backend-independent.

Taken together, these modules form Axon's standard authoring surface. They keep model definitions concise and declarative, while still exposing enough structure for the compiler to typecheck, normalize, flatten, optimize, and lower the program through the shared AST and Graph IRs.

Top-level declarations. An Axon module is a sequence of top-level items: definitions, global bindings, imports, exports, pragmas, and type aliases. Definitions may carry signatures, optional parameters, and scoped path parameters.

Listing 6: Top-level declarations, imports, exports, and a scoped definition.

import NN   
import Act (swiglu)   
export block Pair   
type Pair[T] = (T, T)   
hidden\_dim <- 768   
block@attn :: Tensor[B,S,D] -> ?Tensor[B,K] ->   
↔ Tensor [B, S, D]   
block x attn\_mask = do   
h <- NN.rmsnorm@ln x   
a <- Attention.attention h h h attn\_mask   
return x + a

Listing 6 groups the top-level forms that appear in ordinary authored Axon files. The first three lines show namespace import, member import, and export. The type alias introduces a reusable type constructor. The global binding defines a toplevel constant expression. The definition itself combines a signature, an optional argument marked with ?, and a relative parameter scope @attn.

Binding, return, and do blocks. The basic statement form is single-assignment binding with <-. A do block sequences binds and ends in an explicit return or yield.

Listing 7: Bindings and explicit return in a do block.

```erlang
pair_sum :: Tensor[B,D] -> Tensor[B,D] -> (Tensor[B,D],
↔→ Tensor[B,D])
pair_sum x y = do
```

s <- x + y   
d <- x - y   
return s, d

Listing 7 shows the canonical block form. The same surface form also supports inline sequences separated by semicolons:

Listing 8: Inline do sequences.

identity x = do return x   
cache\_step x = do h <- NN.rmsnorm@ln x; return h

Conditionals. Axon provides both statement-level and expression-level conditionals, as well as ternary shorthand. Statement-level conditionals branch between two do suites, while expression-level conditionals and ternaries produce values.

Listing 9: Expression-level and statement-level conditionals.

normalize\_or\_skip x use\_norm =   
if use\_norm then NN.rmsnorm@ln x else x   
mask\_or\_null mask use\_mask =   
use\_mask ? mask : null   
choose\_path x use\_skip = do   
if use\_skip then do   
return x   
else do   
y <- NN.rmsnorm@ln x   
return y

Listing 9 places the three conditional forms side by side. Only the taken branch of a ternary or if/then/else expression is evaluated. This is the one place where Axon is not eager.

Function application and composition. Calls are written in bare-argument style, optionally with keyword arguments. The language also exposes forward piping and monadic bind as first-class composition operators.

Listing 10: Bare calls, pipes, keyword arguments, and monadic bind.

mlp x = NN.linear@up x |> Act.swiglu |> NN.linear@down

```ocaml
mlp_kw x = NN.linear@proj x bias=true
mlp_bind x =
NN.linear@up x >>= \h -> Act.swiglu h
```

Listing 10 shows the main composition idioms. Nested calls, pipes, lambda definitions, and binds are interchangeable surface styles for the same underlying dataflow. Pipes are often the clearest notation for feed-forward sublayers, while >>= binds are convenient when an intermediate value must be named inside an expression.

Loops, carried state, and yielded values. The for form iterates over explicit ranges. It supports optional lexical scoping via @ . . . , explicit steps, and carried values. A loop may appear as a statement or as a binding-producing form.

Listing 11: Ranges, carried values, and scoped for loops.

sum\_to n = do   
total <- for i <- [0..n) carry (total) do   
next <- total + i   
yield next

```lua
return total
```

states <- for@layers i <- [0..L) step=1 carry (cache) do   
layer <- Cache.index cache i   
yield layer

Listing 11 illustrates both looping forms. The first example updates a carried accumulator. The second shows the scoped form for@layers, which binds the loop body under an additional lexical path prefix.

Scoped parameter binding. The scope form introduces a lexical parameter path and binds the result of the nested sequence. This keeps reusable helper definitions independent of hard-coded checkpoint paths.

Listing 12: Scoped parameter binding with scope.

attn\_out <- scope@attn do   
q <- NN.linear@q\_proj x   
k <- NN.linear@k\_proj x   
v <- NN.linear@v\_proj x   
y <- Attention.attention q k v attn\_mask   
return NN.linear@o\_proj y

Listing 12 shows how a local lexical scope rewrites relative parameter paths. Inside the scope, relative paths such as @ qpro j or @o\_pro j resolve against the enclosing @at tn prefix before lowering.

Values, paths, and type ascription. Expressions include names, literals, lists, tuples, parenthesized expressions, arithmetic, comparisons, Boolean connectives, and explicit type ascriptions.

Listing 13: Values, paths, comparisons, and type ascription.

spec ids = do   
dims <- [B, S, D]   
pair <- (ids, ids)   
abs\_path <- @@model.embed\_tokens   
rel\_path <- @mlp.c\_fc   
same <- ids == ids   
typed <- (ids :: Tensor[B,S])   
return dims, pair, abs\_path, rel\_path, same, typed

Listing 13 collects the basic value forms. At the type level, Axon supports optional types, tuple types, list types, tensor types with symbolic dimensions, and symbolic-dimension arithmetic such as Tensor[B,S, H+Hd]. Together these forms cover the full surface language accepted by the grammar while keeping authored model definitions compact and auditable.

Extended Backus-Naur Grammar We provide the Backus-Naur form in Figure 14, for Axon's surface syntax as a readable specification of the language accepted by the compiler.

## Axon Compiler Phases

The Axon compiler comprises three main phases: (1) desugaring of full Axon DSL into its semantic core; (2) optimization at the Graph IR level; and (3) lowering of the Graph IR to backend-specific code. In the technical supplement, we walk an exemplary small definition through all phases.

Phase 1: Desugaring. The Axon DSL is defined by an EBNF grammar that is operationalized using Lark (Lark

Contributors 2026). The parser transforms valid Axon DSL into an abstract syntax tree (Axon AST), which serves as the representation for Phase 1 and can be dumped to Axon DSL and reparsed. Every desugaring stage of Phase 1 receives an Axon AST and outputs an Axon AST, typically guaranteeing additional properties: (a) the resolver stage inlines library definitions imported and referenced, guaranteeing a closure property stating that all non-primitive operations are expressed as definitions in the AST; (b) the flattening stage flattens nested expressions, guaranteeing a flatness property stating that definitions consist of a flat sequence of operations; (c) the type checker stage infers types from type signatures, type ascriptions, and the type rules of primitive operations, guaranteeing a consistently typed Axon AST. The desugaring phase also contains dead code elimination, constant folding, and other straightforward code normalizations.

Phase 2: Graph optimizations. The closed flat typed Axon AST can be viewed as an instance of a static single assignment (SSA) graph. In this phase, the Axon compiler converts this semantic core to the Graph IR, which is then iteratively optimized using a number of stages: (a) definition inlining replaces wrapper definitions and other low-complexity definitions by integrating their sequence of operations into their call sites, enabling, for example, the inlining of primitive wrapper definitions; (b) generic graph rewriting replaces known patterns of operations by equivalent patterns of operations based on matching of primitive operational provenance, enabling, for example, the replacement of loops iterating over experts in an MoE by a more efficient stacked-experts formulation; and (c) backend-specific graph rewriting replaces known patterns of operations by backend-specific primitives, enabling, for example, the use of optimized attention implementations and kernels.

Phase 3: Lowering. The optimized Graph IR contains definitions as assignment sequences, calls to definitions, and generic and backend-specific primitive operations. The lowering phase is backend-specific and transpiles the Graph IR into one of the currently targeted backends: PyTorch, Tritonaccelerated PyTorch, MLX, JAX, and vLLM. Each of the separate code generators needs to be able to implement the control flow and the primitive operators represented by the Graph IR. In addition, the code generators also need to implement forward and generate methods. To achieve competitive performance, compilation-oriented backends typically supply scaffolding that ensures static cache allocations.

## Axon Compiler Stages

Table 5 lists each stage of the three phases of the Axon compiler together with the representation it operates on and the invariant it establishes. As is conventional, the first invariant is monotonicity of structure: later compiler stages should remove ambiguity and add metadata but should not throw away semantic information or re-interpret the program. The second invariant is backend isolation: backend implementations may choose different tensor libraries and primitive implementations, but they must consume the same Graph IR contract. If a backend requires weaker validation or modelfamily-specific dispatch to run, the issue belongs within the

backend implementation.

## Running Example

In this section we walk a small Axon fragment through the main compiler stages. The example is intentionally smaller than a full transformer block, but exercises the same mechanisms: path sugar, defaulted keyword arguments, tensor shapes, nested calls, and typed graph lowering.

Surface Axon. A reusable projection can be written in a concise functional form:

project :: Tensor[B,S,D] -> Tensor[B,S,O]   
project x = do   
y <- Tensor.reshape x shape=[B, S, D]   
|> NN.linear@proj dim=0   
return y

This surface program contains a pipe, callee-attached path sugar @proj, a shape expression, and omitted default arguments such as bias=true from the library definition of NN.linear.

Normalize and elaborate. Normalization removes the pipe syntax and path sugar by turning them into ordinary call structure. Elaboration then fills in the default parameters declared by the callee signature.

project x = do   
y <- NN.linear @proj (Tensor.reshape x shape=[B, S,   
D]) dim=O bias=true   
return y

The point is not that all defaults must be textually present in the surface program. The invariant is that downstream stages see a call surface whose positional, keyword, and path arguments are unambiguous.

Flatten. Flattening preserves eager evaluation order by introducing explicit binds and making paths absolute or templated absolute values:

project :: Tensor[B,S,D] -> Tensor[B,S,O]   
project x = do   
\_v1 <- Tensor.reshape x shape=[B, S, D]   
y <- NN.linear @@proj \_v1 dim=0 bias=true   
return y

The flat program has no pipe expression, no nested call used as an argument, and no callee path sugar. This is the Axon AST form expected by the typechecker and trivally translated into the Graph IR.

Typecheck. Typecheck annotates every expression with type, arity, and tensor-dimension metadata. The same flat program can be read as:

\_v1 <- (Tensor.reshape (x :: Tensor[B,S,D])   
shape=([B, S, D] :: List[Dim]) ::   
↔ Tensor[B,S,D])   
y <- (NN.linear (@@proj :: Path) (\_v1 :: Tensor[B,S,D])   
dim=(O :: Dim) bias=(true :: Bool) ::   
↔ Tensor[B, S, O])

What matters is not the rendered punctuation, but that the shape relation between Tensor[B,S,D] and Tensor [B, S, O] is explicit before lowering.

Graph IR. Lowering turns each flat bind into a typed graph node. A representation of the graph module for pro ject is given in Figure 6, which might be schematically represented as follows:

```julia
module project(x: Tensor[B,S,D]) -> Tensor[B,S,O]
n1 = call Tensor.reshape(
x, shape=[B,S,D]) -> Tensor[B,S,D]
n2 = call NN.linear(
path=@@proj, x=n1, dim=0, bias=true)
-> Tensor[B,S,O]
return n2
```

Graph Optimizations. As already introduced in the description of Phase 2, the graph optimizations rewrite, replace, or join module graphs at the Graph IR level. An exhaustive list would be beyond the scope of this appendix.

Backend. The Torch, Triton, MLX, JAX, and vLLM code generators and their runtime implementations are based on the Graph IR and therefore share the same typed generic primitive operations, path values, and output types, which defines the cross-backend interface at the lower levels of the Axon compiler pipeline. The only differences are backendspecific primitive operations.

## A Complete Model Definition

Figure 7 gives the complete Axon module GPT-2. The entire model, including the layer stack, is expressed in these Axon definitions and the Axon definitions of the built-in modules uses. The PyTorch, Triton, JAX, MLX, and vLLM implementations benchmarked in this paper are all generated from these Axon definitions together with a standard safetensors or PyTorch checkpoint.

## Python Overhead: Axon vs. Transformers

We profile autoregressive generation (max. 128 tokens) on two models: Qwen/Qwen2.5-0.5B (0.67B parameters, 24 1ayers) and PleIAs/Pleias-3b-Preview (3.8B, 22 layers) to investigate the Python dispatch overhead that Axon's codegen eliminates. Both models were selected because they achieved > 1.0× speedup on both the PyTorch and JAX backends in a 91-checkpoint benchmark, naturally representing subjects of interest for further analysis.

All configurations run on the same underlying framework. Axon's torch code generator emits standard Py-Torch code that calls the same torch.\_C.\_nn.linear, scaled\_dot\_product\_attention,andCUDAkernels as HuggingFace's Transformers. Axon's jax code generator emits standard JAX code compiled by XLA. The speedup therefore cannot come from better kernels. Hence, it comes from how the code is structured at the Python layer.

Table 6 reports wall-clock time and GPU idle percentage for each configuration. GPU active time is measured viatorch.profiler (CPU+CUDA activities), which sums self\_device\_time\_total across all CUDA kernel events. GPU idle is wall-clock time minus GPU active time.

For the JAX backend, GPU idle is reported as 0.0%. This is a property of JAX's asynchronous dispatch model. JAX enqueues operations on the accelerator without blocking Python, returning jax.Array futures. Python code runs ahead of the GPU, so the device always has queued work and never idles waiting for dispatch. This is not a measurement artifact, JAX simply has no concept of GPU idle between kernel launches because there is no synchronous gap between them. We verified that wall-clock measurements use jnp.block\_until\_ready () to ensure the GPU has completed all queued work before recording the end time.

Table 5: Axon Compiler Phase and Stage with Invariants
<table><tr><td>Phase</td><td>Representation</td><td>Primary invariant</td></tr><tr><td>Parse</td><td>one-file AST</td><td>Syntactic structure and explicit MAIN pragma insertion</td></tr><tr><td>Load</td><td>loaded AST set</td><td>Imports and builtins located without rewriting semantics</td></tr><tr><td>Materialize</td><td>one-file AST</td><td>Optional checkpoint/config specialization for generic models</td></tr><tr><td>Resolve/validate-closed</td><td>closed AST</td><td>No unresolved imports or names; unreachable definitions pruned from MAIN</td></tr><tr><td>Normalize</td><td>normalized AST</td><td>Call syntax, pipes, path sugar, and zero-arg call/name distinctions</td></tr><tr><td>Elaborate/validate-elaborated elaborated AST</td><td></td><td>made explicit Default arguments filled and call arguments positionalized</td></tr><tr><td>Flatten/validate-flat</td><td>flat AST</td><td>Explicit evaluation order; flat calls and binds accepted by type- check and Graph IR lowering</td></tr><tr><td>Typecheck/validate-typed</td><td>Typed flat AST</td><td>expression types, arities, dimensions, and primitive rules applied</td></tr><tr><td>Optimize-ast</td><td>typed flat AST</td><td>to a fixpoint Optional conservative AST cleanup with retype/validation</td></tr><tr><td>Graph lowering/validation</td><td>Graph IR</td><td>Typed graph modules, multi-output nodes, structured paths, con- straints, and metadata</td></tr><tr><td>Optimize-graph/validation</td><td>Graph IR</td><td>Optional graph cleanup, specialization, backend-neutral rewrites, and opt-in backend intrinsics</td></tr><tr><td>Backend</td><td>generated/runtime code</td><td>Executable tensor program consuming the validated Graph IR contract</td></tr></table>

Table 6: Wall-clock time and GPU idle for autoregressive generation (112 tokens for Qwen 0.5B, 96 for Pleias 3.8B). “" indicates the metric is not directly measurable: JAX has no torch.profiler equivalent for per-kernel timing, and GPU active/idle is a synchronous-execution concept that does not apply to JAX's async dispatch model.
<table><tr><td rowspan="2">Metric</td><td colspan="3">Qwen2.5-0.5B (0.67B)</td><td colspan="3">Pleias-3b-Preview (3.8B)</td></tr><tr><td>Transformers</td><td>Axon-Torch</td><td>Axon-JAX</td><td>Transformers</td><td>Axon-Torch</td><td>Axon-JAX</td></tr><tr><td>Wall-clock (ms)</td><td>1380</td><td>1185</td><td>541</td><td>1091</td><td>1071</td><td>1160</td></tr><tr><td>Speedup vs Transformers</td><td>1.0×</td><td>0.86×</td><td>0.39×</td><td>1.0×</td><td>0.98×</td><td>1.07×</td></tr><tr><td>GPU active (ms)</td><td>373</td><td>370</td><td></td><td>557</td><td>561</td><td></td></tr><tr><td>GPU idle (ms)</td><td>1007</td><td>815</td><td></td><td>534</td><td>510</td><td></td></tr><tr><td>GPU idle %</td><td>73.0%</td><td>68.8%</td><td>0.0%</td><td>48.9%</td><td>47.6%</td><td>0.0%</td></tr><tr><td>CUDA kernels</td><td>129,189</td><td>120,987</td><td></td><td>101,935</td><td>104,123</td><td></td></tr></table>

Table 7 reports total Python function calls and cProfile cumulative time. Python call counts are measured by cP rofi le in a separate profiling pass (after the wall-clock pass) to avoid profiler overhead interference. JAX's cProfile time on Pleias 3.8B (1.16 s) exceeds Tranformer's (0.87 s) despite having 77% fewer calls. The remaining calls include expensive JAX runtime primitives (apply\_primitive for broadcast/ful1 operations) that dominate this model size.

The comparison is fair by construction: Axon's torch code generator emits PyTorch code that calls the same torch.\_C.\_nn.linear, scaled\_dot\_product\_attention, and CUDA kernels as HuggingFace's Transformers. Both produce identical token sequences (100% top-1 parity). The speedup comes entirely from the Python layer above the kernels.

The Price of Transformers. Transformers implements each model as a nested hierarchy of nn . Module subclasses. Every cal1 goes through PyTorch's dispatch chain (\_call\_ → \_wrapped\_call\_impl → \_call\_impl → forward), costing 3 Python function calls per module per step per layer. For Qwen2.5-0.5B (24 layers, \~13 modules/layer), this is 35,616 wrapper calls per generate pass. On top of this, Transformers's generate () loop adds per-step prepare\_inputs\_for\_generation, a LogitsProcessorList chain (float32 cast + clone), and a stopping-criteria check.

Advantages of Axon. Axon's compiler pipeline (resolve → normalize → elaborate → typecheck → lower → graph optimize → emit) flattens the module hierarchy into a single inline function: model. forward () → \_def\_qwen2 ( )

Table 7: Python function call counts and cProfile cumulative time.“"indicates the category does not apply: JAX fuses all per-op dispatches into a single XLA compilation per step, so individual F. 1inear and SDPA calls do not exist as Python-level dispatches.
<table><tr><td rowspan="2">Metric</td><td colspan="3">Qwen2.5-0.5B (0.67B)</td><td colspan="3">Pleias-3b-Preview (3.8B)</td></tr><tr><td>Transformers</td><td>Axon-Torch</td><td>Axon-JAX</td><td>Transformers</td><td>Axon-Torch</td><td>Axon-JAX</td></tr><tr><td>Total Python calls</td><td>455,769</td><td>508,390</td><td>137,805</td><td>359,909</td><td>330,929</td><td>82,210</td></tr><tr><td>Calls vs Transformers</td><td>1.0×</td><td>1.12×</td><td>0.30×</td><td>1.0×</td><td>0.92×</td><td>0.23×</td></tr><tr><td>cProfile time (s)</td><td>1.13</td><td>0.95</td><td>0.54</td><td>0.87</td><td>0.78</td><td>1.16</td></tr><tr><td colspan="7">Per-op dispatch (Python calls per generate pass):</td></tr><tr><td>nn.Module.__call</td><td>35,616</td><td>0</td><td>0</td><td>28,032</td><td>0</td><td>0</td></tr><tr><td>F.linear</td><td>18,928</td><td>18,928</td><td></td><td>14,880</td><td>14,880</td><td></td></tr><tr><td>SDPA</td><td>2,688</td><td>2,688</td><td></td><td>2,112</td><td>2,112</td><td></td></tr><tr><td>rope_apply</td><td>0</td><td>5,376</td><td></td><td>0</td><td>4,224</td><td></td></tr><tr><td>forward (jit dispatch)</td><td></td><td></td><td>112</td><td></td><td></td><td>96</td></tr></table>

![](images/e90eb1e396fa6aa8ab24368a46c9ba8ed1e98c964f3008d944a0b647a4caaf24.jpg)  
Figure 6: Graph IR view of the running example. Blue denotes typed graph values, orange operation nodes, and green structured operands and attributes. Paths, literals, and shape lists are structured operands, not strings that backends must parse.

(one flat function, zero module wrappers).

Weight access changes from attribute chains through nn.Module.\_\_getattr\_\_ to direct dict lookups (param("layers.0.self\_attn.qproj.weight")). The generate loop drops from 6 Python phases to 4 (no logits processor, no prepare\_inputs). Table 7 confirms: the GPU-op call counts (F.linear, SDPA) are identical between Transformers and Axon-Torch, but nn .Module.\_cal1\_drops from 35,616 to 0.

Comparison to JAX. Axon's jax code generator applies the same flattening but jax. jit fuses all per-op Python dispatches into a single XLA compilation per step. The 18,928 F.1inear calls and 2,688 SDPA calls that exist as separate Python dispatches in both Transformers and Axon-Torch become zero Python calls, they are compiled into one fused XLA kernel. Combined with JAX's async dispatch, which lets Python run ahead of the GPU (0% GPU idle), this achieves 2.55× speedup on the 0.5B model. However, the per-step host synchronization for EOS checking (jax . devi ce\_get) scales with GPU compute time, causing JAX's advantage to diminish on larger models.

Why this generalizes. The overhead Transformers Library pays is structural, not modelspecific: nn.Module.\_\_call\_ dispatch scales with num\_layers × modules\_per\_layer. Logits processing happens every step regardless of model. prepare\_inputs\_for\_generation is a Python abstraction every model passes through. Axon's codegen eliminates these for any model. The compiler flattens the hierarchy at compile time, so the generated code never pays the nn .Module dispatch tax. This is why the speedup is consistent across architectures (Qwen, Mistral, Llama, Bloom, Pleias) and scales with decode steps (generate is faster; forward is not, because a single pass amortizes the overhead).

## Full-width versions of the main-text figures.

Figures 8–11 reproduce the remaining benchmark figures from the main text at full page width.

Encoder-only and encoder-decoder models. Figure 12 shows the performance comparison for encoder-only and encoder-decoder models up to 4B. Figure 13 shows the performance comparison for encoder-only and encoder-decoder models between 4B and 32B.

Figure 14 shows a loss curve comparison on Gemma 3 270M between Transformer's implementation and an Axonderived PyTorch model. The important insight here is not the quality of the training but the identical training behaviour with the two curves perfectly overlaid on each other

## Figure 7: Complete Axon definition of GPT-2.

```haskell
{-# CHECKPOINTS ["openai-community/gpt2", "openai-community/gpt2-medium",
"openai-community/gpt2-large","openai-community/gpt2-xl"] #-}
{-# TASK "causal_lm" #-}
import Activations (gelu_new)
import Attention (attention, merge_heads, reshape_heads)
import Masking (Mask, causal_mask_for_input, mask_for_input, mask_length_for_input)
import Positions (position_ids)
import Cache
import Config
import NN
import Tensor
CONTEXT_SIZE <- (Config.dim @@n_positions default=1024 :: Dim)
MODEL_DIM <- (Config.dim @@n_embd default=768 :: Dim)
NUM_LAYERS <- (Config.dim @@n_layer default=12 :: Dim)
NUM_HEADS <- (Config.dim @@n_head default=12:: Dim)
VOCAB SIZE <- (Config.dim @@vocab size default=52057 :: Dim)
gpt2_block :: Tensor[B,S,D] -> Tensor[B,1,S,K] -> ?CacheLayer[B,H,P,CONTEXT_SIZE,DH] ->
→ (Tensor[B,S,D], ?CacheLayer[B,H,K,CONTEXT_SIZE,DH])
gpt2_block x mask past_kv = do
x1 <- NN.layernorm@ln_1 x
a, new_kv <- scope@attn do
q_lin, k_lin, v_lin <- Tensor.chunk (NN.linear@c_attn x1 dim=(3 * MODEL_DIM) bias=true
→ transpose=true) parts=3
q <- reshape_heads q_lin heads=NUM_HEADS
k <- reshape_heads k_lin heads=NUM_HEADS
v <- reshape_heads v_lin heads=NUM_HEADS
k, v, new_kv <- Cache.update past_kv k v
a <- attention q k v mask I> merge_heads |> NN.linear@c_proj dim=MODEL_DIM bias=true
→ transpose=true
return a, new_kv
x|<- x+a
x3 <- NN.layernorm@ln_2 x
m <- scope@mlp do
return NN.linear@c_fc x3 dim=(4 * MODEL_DIM) bias=true transpose=true |> gelu_new |>
NN.linear@c_proj dim=MODEL_DIM bias=true transpose=true
return x + m, new_kv
gpt2 :: Tensor.TokenIds[B,S] -> ?Mask[B,K,CONTEXT_SIZE] -> ?
↔ Cache[B,NUM_HEADS,P,CONTEXT_SIZE,MODEL_DIM / NUM_HEADS] -> ?Bool -> (TenSOr[B,S,V], ?
Cache[B,NUM_HEADS,K,CONTEXT_SIZE,MODEL_DIM / NUM_HEADS])
gpt2 input_ids attn_mask past_kv use_cache = do
tok <- NN.embedding@wte input_ids dim=MODEL_DIM
past_len <- Cache.past_length past_kv
attn_mask <- mask_for_input input_ids attn_mask CONTEXT_SIZE
pos <- position_ids input_ids attn_mask=attn_mask past_length=past_len pad_fill=1 |>
→ NN.embedding@wpe
x <- tok + pos
key_len <- mask_length_for_input input_ids attn_mask past_len
mask <- causal_mask_for_input input_ids key_len padding_mask=attn_mask window=CONTEXT_SIZE
work_kv <- Cache.prepare past_kv use_cache CONTEXT_SIZE
read_kv <- Cache.read past_kv work_kv
x, work_kv <- for@h i <- [0..NUM_LAYERS) carry (x, work_kv) do
past_i <- Cache.index read_kv i
x, new_i <- gpt2_block x mask past_i
work_kv <- Cache.append work_kv new_i
yield x, work_kv
new_length <- past_len + S
work_kv <- Cache.finish work_kv new_length
new_kv <- (use_cache ? work_kv : null)
logits <- NN.layernorm@ln_f x |> NN.linear@wte
return logits, new_kv
```

![](images/65d8208e15811aff55cf494b0e1bcc9bd806b0e1ed333e4d9300e3e6907f10ae.jpg)  
Transformers tok/s  
Figure 8: Autoregressive generation performance with decoder-only models with up to 4B parameters. Log-log plot where the diagonal line indicates equal performance. Points above the parity line indicate Axon is faster, color-fill denotes backends and border-line denotes dtype.

Axon vs. Transformers: Generation Throughput (32B) log-log throughput scatter; above diagonal means Axon is faster (higher tok/s)  
![](images/f64610e858de7b74213765994b8df8f877a18009fd70f06c89db9f6bc7546ebe.jpg)  
Transformers tok/s  
Figure 9: Benchmarking Autoregressive models between 4B and 32B models. Log-log plot where the diagonal parity line indicates equal performance. Generally, Axon produces comparable or superior throughput on the bigger models.

vLLM Native Generation: Axon vs Transformers (HF)  
![](images/bbd5b420827a9946228eca6d9d6f047ddaa0dbbcb47efb65aa3598fdcea1331e.jpg)  
Figure 10: vLLM: Axon (vLLM native) vs. Transformers generation throughput on 88 checkpoints. Points above the parity line indicate Axon is faster. 74% fall above.

HF vs Axon Throughput (tok/s) MLX: Axon vs. Transformers (HuggingFace) on Apple Silicon  
![](images/f6bf877cb2ee55cc59c0be9cda9e8e28ad94894a57ef84a9ab850af088059667.jpg)  
Figure 11: MLX. Benchmarking conventional Transformers models against Axon derived standalone model definitions. Axon yields some considerable speed-ups across the board.

Transformers vs Axon Throughput log-log wall-time scatter; below diagonal means Axon is faster (lower ms)  
![](images/1ee43db25090a32c678947f1d3527d11893ed0dc30a1bc6cc0adbed6ea081fbc.jpg)  
Transformers wall time (ms)  
Figure 12: Benchmarking Encoder-only and Encoder-Decoder up to 4B models. Log-log plot where the diagonal parity line indicates equal performance. Here, we see the majority of the points in the vicinity below the parity line.

Axon vs. Transformers: Forward Pass (32B) log-log wall-time scatter; below diagonal means Axon is faster (lower ms)  
![](images/fa3df6214b6a6bff03364d756c1b07aa8e219ae0bfd848934ee2b3b14d544f1e.jpg)  
Transformers wall time (ms)  
Figure 13: Benchmarking Encoder-only and Encoder-Decoder between 4B and 32B models. Log-log plot where the diagonal parity line indicates equal performance.

Axon vs. Transformers: Gemma-3 270M Training Loss 2000steps | bs=4 | seq=512 |bfloat16 |ccdv/arxiv-summarization  
![](images/c14dc2b96251955fe2134846f3c5641c95909f8f11b648700dccf95d348f7f48.jpg)  
Figure 14: Training of Gemma 3 270M on summarization task for 2000 steps, demonstrating identical training behaviour between conventional Transformers-definition and Axon derived PyTorch model. Note the Axon and Transformers loss curves are on top of each other together.

# Axon Grammar (Current Authoring Form)   
#   
# Indentation-sensitive (Python-style INDENT/DEDENT tokens).   
# "--" starts a comment outside strings.   
# Newlines are significant; \_NL tokens carry indentation context.   
# The NAME terminal is permissive: it matches qualified identifiers   
# including "::", "@", and "." internally (e.g. ns::op, mod@path, ns.mod).   
#   
# Authoritative source: brainsurgery/synapse/axon/parse/grammar.py   
# (Lark LALR grammar with custom indentation post-lexer).   
#   
# Some parser-only constructs (parallel arg\_expr hierarchy, nl\_gap/arg\_ws   
# whitespace gaps for multi-line pipe/ternary/if expressions) are omitted   
# for clarity; see grammar.py for the full Lark grammar.   
program ::= { top\_item NEWLINE } ;   
top\_item ::= definition\_decl   
| global\_binding   
I import\_decl   
| export\_decl   
| pragma   
| type\_alias\_decl   
;   
# -- Module definitions --   
definition\_decl ::= [ signature NEWLINE ] definition ;   
signature ::= mod\_decl "::" signature\_type ;   
signature\_type ::= type\_expr { "->" type\_expr } ;   
definition ::= mod\_decl { def\_param } "=" expr ;   
mod\_decl ::= module\_name { "@" NAME } ;   
module\_name ::= NAME { "." NAME } ;   
def\_param : := NAME   
"?" NAME "=" def\_param\_simple   
| "?" NAME "=" "(" expr ") "   
;   
def\_param\_simple ::= literal | NAME ;   
# -- Global bindings --   
global\_binding ::= module\_name "<-" expr ;   
# -- Imports --   
import\_decl ::= "import" import\_item { ", " import\_item } ;   
import\_item ::= module\_name [ import\_members ] ;   
import\_members ::= " (" [ NAME { ", " NAME } [ ", " ] ] ") "   
| NAME { NAME }   
;   
# -- Exports   
export\_decl ::= "export" export\_members ;   
export\_members ::= " (" [ NAME { ", " NAME } [ ", " ] ] ") "   
| NAME { NAME }   
;   
# -- Pragmas --   
pragma ::= "{-#" NAME pragma\_value "#-}" ;   
pragma\_value ::= tuple\_expr | list\_expr | literal ;

```tcl
# -- Type aliases --
type_alias_decl ::= "type" NAME [ type_alias_params ] "=" type_expr ;
type_alias_params ::= "[" type_alias_param { "," type_alias_param } "]" ;
type_alias_param ::= NAME | ".." NAME ;
# -- Type expressions --
type_expr ::= type_optional ;
type_optional ::= "?" type_optional
|type_tuple
| type_list
type_tensor
type_name
;
type_tuple ::= " (" type_expr ", " type_expr { ", " type_expr } [ ", " ] ") " ;
type_list ::= "List" "[" type_expr "]" ;
type_tensor ::= type_name "[" type_dim_expr { ", " type_dim_expr } "]" ;
type_name : := NAME ;
type_dim_expr ::= type_dim_term { ("+" | "−") type_dim_term } ;
type_dim_term ::= type_dim_ factor { ("" | "/" | "
type_dim_factor ::= NUMBER
".." type_name
type_name
"(" type_dim_expr ")"
# -- Statements --
statement ::= for_statement
for_bind_statement
if_statement
scope_bind_statement
return_statement
yield_statement
bind_statement
for_statement ::= "for" [ for_scope ] NAME "<-" range_expr
for_step ] [ for_carry
( "do" suite | bind_expr )
for_bind_statement ::= target_list "<-" "for" [ for_scope ] NAME "<-" range_expr
for_step ] [ for_carry
( "do" suite i bind_expr )
for_scope ::= "@" scoped_name ;
for_step ::= "step" "=" expr ;
for_carry ::= "carry" "(" target_list ")" ;
if_statement ::= "if" expr "then" "do" suite
"else" "do" suite
;
scope_bind_statement ::= target_list "<-" "scope" scope_ref
{ scope_bind_kwarg } "do" suite
scope_bind_kwarg ::= NAME "=" kwarg_value ;
scope_ref ::= path_lit
"@" ] scoped_name
"@" ] STRING
scoped_name ::= scoped_part { "." scoped_part } ;
scoped_part ::= NAME | TEMPLATE_NAME ;
```

```tcl
return statement ::= "return" expr_ list ;
yield_statement ::= "yield" expr_list ;
bind_statement ::= target_list "<-" expr ;
target_list : := NAME ," NAME };
expr_list ::= expr { "," expr } ;
# -- Suites (statement blocks)
suite ::= inline_suite [ block_suite ]
| block_suite
inline_suite ::= inline_statement { ";" inline_statement } [ ";" ] ;
inline_statement ::= return_statement | yield_statement | bind_statement
block_suite ::= INDENT { statement_line NEWLINE } DEDENT ;
statement_line ::= statement { ";" statement } [ ";" ] ;
-- Range expressions --
[a..b) half-open: from=a，to=b (default)
(a..b] open-closed: from=a+1, to=b+1
# [a..b] closed-closed: from=a, to=b+1
range_expr ::= range_start expr ".." expr range_end ;
range_start ::= " [" | " (" ;
range end ::= "] " | ") " ;
# Expression hierarchy --
# Precedence (low to high):
＃# ascribe >> bind (>>=) >> pipe (I>) >> ternary/if >> or >> and >> cmp
>> add >> mul >> application (bare-call) >> atom
#
# Multi-line variants of if/ternary (with INDENT/DEDENT around branch
values) and multi-line pipe/bind chains (with nl_gap newlines) are
# also accepted; see grammar.py for details.
expr ::= expr_core [ "::" type_expr ] ;
expr_core ::= do_expr | bind_expr ;
do_expr ::= "do" suite ;
bind_expr ::= pipe_expr [ ">>=" lambda_expr ] ;
lambda_expr ::= "\" NAME "->" expr ;
pipe_expr ::= ternary_expr { ">" ternary_expr } ;
ternary_expr ::= if_expr
or_expr "?" tuple_value ":" tuple_value
| or_expr
if_expr ::= "if" expr "then" tuple_value "else" tuple_value
or_expr
;
or_expr ::= and_expr { "or" and_expr } ;
and_expr ::= cmp_expr { "and" cmp_expr } ;
cmp_expr ::= add_expr { CMP_OP add_expr } ;
add_expr ::= mul_expr { ("+" "_") mul_expr } ;
mul_expr ::= app_expr { ("" "/"|"
app_expr ::= NAME bare_arg { bare_arg }
| atom
;
bare_arg ::= NAME "=" arg_expr # keyword argument
| arg_expr # positional argument
# arg_expr mirrors the full expr precedence hierarchy (or/and/cmp/add/mul/
```

# if/ternary/do/ascribe) but excludes bare tuple expressions at the atom   
# level -- use "(expr)" for parenthesized single-value arguments.   
# See grammar.py (arg\_expr through arg\_mul) for the parallel hierarchy.   
kwarg\_value ::= arg\_expr ;   
tuple\_value : := expr { ", " expr } [ ", " ] ;   
atom ::= tuple\_expr   
"(" expr ") "   
list\_expr   
I literal   
NAME   
tuple\_expr ::= " (" expr ", " expr { ", " expr } [ ", " ] ") " ;   
list\_expr ::= " [" [ expr { ", " expr } [ ", " ] ] "]" ;   
# -- Literals --   
literal ::= NUMBER | "true" | "false" | "null" | STRING | path\_lit ;   
path\_lit ::= PATH\_LIT ;   
# -- Terminals --   
NAME ::= /[A-Za−z\_](?:[A-Za−z0−9\_:@]|\.(?!\.))\*/   
# Greedy: matches ns::op, mod@path, Act.swiglu as one token.   
TEMPLATE NAME ::= /\{[A-Za-z\_][A-Za-z0-9\_]\*\}/ ;   
NUMBER ::= /-?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?/ ;   
STRING : := /" ([^"\\] |\\. ) \*" | ′ ([^ \\] |\\. ) \*′/ ;   
PATH LIT : := /@@?   
↔′([^′\\]|\\.)\*′|@@?([A-Za-z\_][A-Za-z0-9\_]\*|[0-9]+)(\.([A-Za-z\_][A-Za-z0-9\_]\*|[0-9]+))\*/   
# @@abs.path @rel.path @'escaped path' @@'abs escaped'   
CMP\_OP ::="=="|"!="|"<="|">=" "<" ">"   
NEWLINE ::= /(\r?\n[ \t]\*)+/ ;   
COMMENT ::= /--[^\n]\*/ ;   
INDENT ::= /\* emitted by indentation post-lexer \*/ ;   
DEDENT ::= /\* emitted by indentation post-lexer \*/ ;