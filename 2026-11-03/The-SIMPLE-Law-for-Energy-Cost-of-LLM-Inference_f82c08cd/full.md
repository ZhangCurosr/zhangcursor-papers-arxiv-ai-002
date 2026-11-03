# The SIMPLE Law for Energy Cost of LLM Inference

Hiari Pizzini Cavagna<sup>a,∗,1</sup>, Andrea Proia<sup>b,1</sup>, Daniele Cesarini<sup>c</sup>, Zeynep Kiziltan<sup>b</sup> and Andrea Bartolini<sup>a</sup>

<sup>a</sup>DEI - University ofBologna, Bologna, Italy <sup>b</sup>DISI - University of Bologna, Bologna, Italy <sup>c</sup>CINECA, Casalecchio di Reno, Italy

## A R T I C L E I N F O

Keywords:   
LLM Energy Consumption   
LLM Inference Modeling   
Energy Prediction   
Token Energy Cost   
Tokenomics   
Token Economy   
Sustainable AI   
Reasoning Energy Cost   
Green AI

## A BS T RA C T

Generative AI is billed per token, yet tokens are not equally expensive. Each token enters the KV-cache and raises the cost of those that follow, so energy grows non-linearly along a sequence whose tokens look identical. No analytical model states what a token costs, and no law names the factors that set it. The most accurate SOTA model is a fit, not a law: it absorbs architecture into fitted coeficients, requiring per-LLM fitting. What is therefore needed is a law expressed in the architecture’s own hyperparameters, one that names which of them govern the cost of a token, explicitly accounts for attention mechanism, and applies to unseen LLMs. We derive closed-form expressions of the computational and memory-access complexity of LLM inference in terms of architectural hyperparameters, covering Multi-Head Attention (MHA) and Grouped-Query Attention (GQA), and compare a family of analytical models of increasing granularity. From these, we obtain the SIMPLE Inference Model to Predict LLMs Energy (SIMPLE), a four-parameter law that separates prefill from decode, and decode into KV-cache cost and next-token computation. Over traces of 13 dense LLMs from six families (OPT, LLaMA, Gemma, Falcon, Qwen2, and Granite) on NVIDIA H100 GPUs, it attains �<sup>2</sup> = 0.969 and 11.97% MAPE, 3.1−3.5× below the SOTA, without per-LLM calibration. The SIMPLE law explains as well as predicts: prefill energy grows quadratically with the hidden dimension, so larger LLMs sufer most from long inputs; and more aggressive GQA KV grouping reduces the per-token energy by up to 3× relative to standard MHA. Increasing the reasoning token budget pushes output length beyond the analytically derived energy minimum, driving superlinear energy growth; although accuracy grows with the budget, energy grows faster, so Intelligence per Joule peaks at an intermediate budget and falls by up to 3.14× across a 4.0× reasoning-budget range. Agentic serving raises energy per request by 13.0× relative to non-agentic ones, driven by the growth of both the prefill and the KV-cache terms with input length; while prefix caching removes most of the prefill cost (−93%), the KV-cache term is unafected, so even a cached agentic request costs 4.8× more. An interactive implementation of SIMPLE is available at ecs-lab/SIMPLE.

## 1. Introduction

Inference, not training, is where most of the energy of a Large Language Model (LLM) is spent: a model is trained once and then served continuously, and inference accounts for up to 90% of its life-cycle energy [15]. Yet for an LLM not yet deployed, the only dependable way to know that cost is to build it, serve it, and measure it, an answer that arrives once the architecture is no longer a choice. The target is also moving: reasoning spends additional tokens to buy accuracy [21], while agentic workflows [7] have pushed median prompt lengths from 750–1050 to 68K tokens [18].

The token is the unit of account of this economy. Inference is priced, billed, budgeted, and benchmarked per token [35], and capacity is planned in tokens per second and tokens per joule. Yet no analytical model states what a token costs, and no law names the factors that set it. A token is an observable of the workload, not a unit of computational work: each token enters the KV-cache and raises the cost of those that follow, so average token energy depends on input length, generation length, context reuse, and how attention is organized. The consequence is already visible in the economics: using a model with a lower listed price per token can incur up to 28× the total cost of using a nominally more expensive one [4]. An entire industry settles in a currency whose production cost cannot be computed.

Energy models exhibit the same gap in analytical form. Prior work fits energy as a function of input and output lengths [38, 43, 40, 24] empirically, without deriving it from the computational kernels, thereby describing a measured regime rather than explaining it. SweetSpot [26], the most accurate analytical formulation reported to date, removes the decisive obstacle by deriving the decode phase in closed form from the computation and memory accesses of decoder-only Transformer inference. SweetSpot, however, is a fit, not a law. Its coeficients are identified on a single LLM and carry that LLM’s architecture implicitly: the number of layers, the hidden dimension, and the organization of attention never appear as terms. The formulation therefore neither names the factors that set the cost of a token nor applies to a model it has not already measured, and later analytical models [36, 11, 22] relax this only in part.

This limitation is felt where decisions are now being made: in energy eficiency. Intelligence per Watt relates delivered capability to the power that produced it [27], and serving is reported in tokens per joule [35], but both are measured on systems that already exist. They rank deployments after the fact and cannot say which architecture would deliver the same capability for less, how many reasoning tokens are worth spending [21], or at what context length serving an agentic request ceases to be afordable [18]. A measured eficiency locates no optimum. What is therefore needed is a law expressed in the architecture’s own hyperparameters, one that explains which of them govern the energy of a token, predicts that energy for an LLM never benchmarked, and thereby turns eficiency from something observed into something derived.

Modern LLMs share the decoder-only Transformer [37] and difer in a handful of hyperparameters and attention architectures: Multi-Head Attention (MHA) has largely given way to Grouped-Query Attention (GQA) [1] and Multihead Latent Attention (MLA) [9], each cutting memory trafic and KV-cache size, and with them the energy per generated token [26]. Furthermore, long generations, 68Ktoken contexts, and mandatory prefix caching push today’s requests precisely into the regime where the KV-cache term dominates: no law of token cost can be blind to how attention is organized.

Together, these considerations motivate the following open research questions:

• RQ1: Is there an analytical model that can accurately predict inference energy across LLMs with diferent architectures, including the attention mechanism?

• RQ2: Can an LLM-agnostic model remain compact and accurate enough to be a practical alternative to LLM-specific models, while generalizing to unseen LLM architectures and workloads?

• RQ3: How do LLM size and architectural hyperparameters influence inference energy consumption?

• RQ4: How does the reasoning token budget afect inference energy and Intelligence per Joule?

• RQ5: How does the energy cost of a single query shift from non-agentic to agentic workloads?

This work answers these with the following contributions:

• We present a taxonomy of state-of-the-art LLM inference energy models, classified by functional form, grounding theory, and ability to predict inference energy across LLMs with diferent architectures, including the attention mechanism. We find that the most accurate model, SweetSpot [26], absorbs architectural information implicitly into its fitted coeficients, requiring per-LLM calibration and making it neither LLM-agnostic nor attention-aware. The SOTA therefore lacks an analytical model capable of accurately predicting the energy consumed by a generic request for an arbitrary LLM.

• To close this gap, we analytically derive closed-form expressions for the computational and memory-access complexity of LLM inference, covering both MHA and GQA mechanisms Building on these, we propose the SIMPLE law, which separately models prefill and decode energy costs, further decomposing the latter into the self-attention KV-cache cost and the next-token computation (i.e., the feed-forward network). Validated against a family of analytically derived models spanning diferent fitting granularities and modeling approaches, SIMPLE achieves the best accuracy-compactness trade-of among the evaluated models, reducing the error of SOTA models by 3.1– 3.5×.

• Applying the SIMPLE law across LLMs of diferent sizes, families, and architectures, we show that hidden dimension � is the most sensitive architectural hyperparameter, as prefill energy scales quadratically with �, while layer depth contributes linearly. Attention architecture matters strongly: more aggressive GQA KV grouping reduces per-token energy by up to 3× relative to standard MHA, making it an impactful architectural choice for reducing inference energy.

• We use the SIMPLE law to analytically derive the output length $\tau _ { o u t }$ that minimizes per-token energy (the sweet spot), showing that increasing the reasoning token budget pushes $\tau _ { o u t }$ beyond this minimum, driving super-linear energy growth. Although accuracy grows with the reasoning budget, energy grows faster, so Intelligence Per Joule peaks at an intermediate budget and falls by up to 3.14× at the lowest and highest budgets across a 4.0× range.

• Using the SIMPLE law, we analyze the energy cost of agentic AI workloads and show that energy per request increases by 13.0× relative to non-agentic ones, driven by the growth of both the prefill and KVcache terms with input length. While prefix caching removes most of the prefill cost (-93%), the KV-cache term, accounting for 86% of the remaining energy, is unafected, so even a cached agentic request still costs 4.8× more than a non-agentic one.

We evaluate our models on a large dataset [26] of TensorRT-LLM [23] traces collected from a set of widely used dense LLMs running on NVIDIA H100. The included models span model sizes and architectures, covering OPT [44], LLaMA [14], Gemma [33], Falcon [2], Qwen2 [41], and Granite [20].

The rest of the paper is organized as follows. After providing background in Section 2, we review related work in Section 3. Then we derive the floating point and memory operations of LLM inference in Section 4, and building on this, we present in Section 5 SIMPLE, together with the family of complexity-derived models used throughout the evaluation. In Section 6 we experimentally validate SIM-PLE, and discuss in Section 7 how model size, architectural hyperparameters, reasoning, and agentic workloads impact energy consumption. Finally, we conclude in Section 8.

## 2. Background

This section introduces the background needed for the analysis in this work. We briefly review the fundamentals of LLM architectures, their computational characteristics, and the metrics used throughout the paper.

LLM Architecture LLMs are built on the Transformer architecture [37], which has become the standard for generative AI. A Transformer block consists of two main components, the Multi-Head Self-Attention Mechanism (MHA) and the Feed-Forward Network (FFN), each wrapped in residual connections, layer normalization, and dropout. In MHA, the hidden representation is projected into several subspaces, corresponding to diferent heads. Each head learns to capture a diferent type of dependency within the sequence, and the head outputs are then combined. This improves the model’s expressivity and its ability to represent diverse relationships, but it also increases the number of operations in proportion to the number of heads. To reduce this overhead, Multi-Query Attention (MQA) was proposed in 2019 [29]. It keeps multiple query heads but shares a single set of Key-Value (KV) projections across all of them, which drastically reduces memory usage and KV-cache size during decoding and makes inference more eficient. More recently, Grouped-Query Attention (GQA) was introduced in 2023 [1] as a generalization of MHA and MQA: keys and values are shared within groups of heads, striking a balance between the eficiency of MQA and the expressivity of MHA. Multi-Head Latent Attention (MLA), introduced with DeepSeek-V2 [9], provides a diferent mechanism to reduce the KV-cache footprint. Rather than sharing KV projections across heads, it compresses keys and values into a single lowrank latent vector. The full per-head keys and values are then recovered from this latent through up-projection matrices, preserving the expressivity of MHA at a fraction of the memory cost. Despite its distinct compression mechanism, MLA can nonetheless be seen as a variant of GQA [5].

LLM Inference Computation In large-scale models, most of the learnable weights, referred to as parameters, reside in the FFN layers, whereas the runtime complexity is dominated by the attention mechanism, owing to its quadratic dependence on the input sequence length. The total number of parameters, typically in the billions, is referred to as model size. LLM inference is typically split into two phases: $p r e f i l l ,$ in which the entire input sequence is processed in parallel to initialize the KV-cache, and decode, in which tokens are generated autoregressively [42]. During prefill, the computational cost is dominated by the quadratic attention term; during decode, thanks to the reuse of cached keys and values, the cost per token grows only linearly with the sequence length. Nevertheless, for long outputs, the decode phase often accounts for most of the total inference time, since tokens must be produced one at a time, which restricts the opportunities for parallelization. On modern GPUs, the throughput of LLM inference is constrained not only by arithmetic performance but also by memory bandwidth. Storing and retrieving the large KVcache during decoding places considerable stress on the memory subsystem. For example, in autoregressive decoding, every generated token requires loading all previous keys and values from memory, which makes attention kernels strongly memory-bound, especially for long contexts.

Metrics and Measurement We define the total energy consumption $E _ { \mathrm { t o t } } ( \tau _ { \mathrm { i n } } , \tau _ { \mathrm { o u t } } )$ , expressed in Joules, as the energy the GPU consumes to process a given request, characterized by an input length $\tau _ { \mathrm { i n } }$ and an output length $\tau _ { \mathrm { o u t } } .$ Normalizing $E _ { \mathrm { t o t } }$ by $\tau _ { \mathrm { o u t } }$ yields the Energy-per-Token $E _ { \mathrm { t o k } } \mathrm { : }$

$$
E _ { \mathrm { t o k } } = \frac { E _ { \mathrm { t o t } } } { \tau _ { \mathrm { o u t } } } \quad \left[ \frac { \mathrm { J o u l e s } } { \mathrm { T o k e n } } \right]
$$

We also define energy eficiency $E _ { \mathrm { e f f } }$ as the reciprocal of $E _ { \mathrm { t o k } } .$ capturing the number of output tokens generated per unit of energy. Finally, to relate energy consumption to the quality of the generated output, we adopt the Intelligence per Joule (IPJ) metric [27]. We compute it as the ratio of task accuracy to the energy required to process a request:

$$
\mathrm { I P J } = \frac { \mathrm { A c c } } { E _ { \mathrm { t o t } } } \quad \left[ \frac { \mathrm { A c c u r a c y \ \% } } { \mathrm { J o u l e } } \right]
$$

where Acc is the accuracy averaged over the evaluated queries and $E _ { \mathrm { t o t } }$ is the average per-request energy, making it a practical metric for realistic deployment scenarios.

## 3. Related Work

Recent work on LLM inference energy falls broadly into two categories: (i) experimental characterization of energy consumption, and (ii) analytical or empirical models for predicting it. We review each in turn, then position our contributions with respect to the identified limitations.

## 3.1. Experimental Characterizations

Several studies have characterized LLM inference energy across diferent workloads and hardware conditions. Samsi et al. [28] benchmark energy for LLaMA models of varying sizes across two NVIDIA GPU generations (V100 and A100), including multi-node and multi-GPU execution. Chung et al. [6] introduce ML.ENERGY, a benchmark suite for measuring energy under realistic serving conditions across diferent tasks, and show that automated configuration recommendations can reduce consumption. Stojkovic et al. [31] study how inference server configurations afect latency, throughput, and energy. Maliakel et al. [19] examine the efect of GPU Dynamic Voltage and Frequency Scaling (DVFS) settings on energy for diferent workloads. Fernandez et al. [13] analyze energy across inference optimization strategies and task types. Wilhelm et al. [39] investigate the accuracy–energy trade-of, focusing on how inference strategies such as chain-of-thought afect it.

Unlike the following works, none of these proposes an energy prediction model.

Table 1  
Comparison of existing energy models for LLM inference.
<table><tr><td>Model</td><td> $E _ { \mathrm { t o t } }$  Form</td><td>Analytical</td><td>LLM-agnostic</td><td>Attention-aware</td></tr><tr><td>Vellaisamy et al., 2026 [38]</td><td>Linear</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Zhan et al., 2026 [43]</td><td>Linear</td><td>X</td><td>×/√</td><td>X</td></tr><tr><td>Delavande et al., 2026 [11]</td><td>Bilinear</td><td>√</td><td>X</td><td>X</td></tr><tr><td>Wilkins et al., 2025 [40]</td><td>Bilinear</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Oviedo et al., 2026 [24]</td><td>Power-law</td><td>×</td><td>X</td><td>X</td></tr><tr><td>SweetSpot, 2026 [26]</td><td>Quadratic</td><td>√</td><td>X</td><td>X</td></tr><tr><td>Vartziotis et al., 2026 [36]</td><td>Quadratic</td><td>√</td><td>√</td><td>X</td></tr><tr><td>Niu et al., 2026 [22]</td><td>-</td><td>√</td><td>X</td><td>√</td></tr><tr><td>SIMPLE</td><td>Quadratic</td><td>√</td><td>√</td><td>√</td></tr></table>

## 3.2. Predictive Models

Recent studies have proposed models to predict the energy consumption of LLM inference. These models focus on the number of input and output tokens $( \tau _ { \mathrm { i n } }$ and $\tau _ { \mathrm { o u t } } )$ , while not explicitly accounting for batch size. Batch size is instead commonly incorporated into models targeting distributed inference [17, 25], whereas it is generally not considered in models for single-GPU inference, as in our setting.

The following paragraphs review the state-of-the-art models, grouped by $E _ { \mathrm { t o t } }$ functional form and further characterized by three criteria that directly reflect the limitations our work addresses. First, whether they are analytically derived from first principles, such as Transformer computational complexity, or are empirical. Second, whether they must be fitted separately for each LLM. We refer to models that require a per-LLM fit as LLM-specific, and to those that do not as LLM-agnostic. Third, whether they are attentionaware, meaning that they explicitly account for diferent attention mechanisms, rather than assuming standard MHA. Table 1 summarizes all reviewed models along these criteria.

Linear Models A common empirical approach is to assign a fixed cost to input and output sequence lengths, yielding a linear model:

$$
E _ { \mathrm { t o t } } \approx \theta _ { 0 } + \theta _ { \mathrm { 1 } } \tau _ { \mathrm { i n } } + \theta _ { 2 } \tau _ { \mathrm { o u t } }
$$

Both Vellaisamy et al. [38] and Zhan et al. [43] propose linear models, though they difer in how they handle inputlength dependence. Vellaisamy et al. focuses solely on decode energy absorbing the prefill cost into a fixed constant, efectively modeling decode energy as linear in $\tau _ { \mathrm { o u t } }$ alone, while Zhan et al. make $\tau _ { \mathrm { i n } }$ an explicit term as in the formulation above. Neither model is derived from first principles or includes architectural hyperparameters, so both must be fitted separately for each LLM and are not attention-aware.

Zhan et al. [43] apply their model to Vision-Language Models and additionally propose an LLM-agnostic formulation that scales with model size � (in billions of parameters):

$$
E _ { \mathrm { t o t } } \approx \theta _ { 0 } + \theta _ { 1 } S + \theta _ { 2 } \tau _ { \mathrm { i n } } + \theta _ { 3 } \tau _ { \mathrm { o u t } } + \theta _ { 4 } S \tau _ { \mathrm { i n } } + \theta _ { 5 } S \tau _ { \mathrm { o u t } }
$$

While this formulation is LLM-agnostic, model size serves as the only architectural proxy, capturing scale but not structural hyperparameters such as the number of layers, hidden dimension, or attention configuration.

Bilinear Models Wilkins et al. [40] and Delavande et al. [11] extend the linear formulation with an interaction term between input and output sequence lengths:

$$
E _ { \mathrm { t o t } } \approx \theta _ { 0 } + \theta _ { 1 } \tau _ { \mathrm { i n } } + \theta _ { 2 } \tau _ { \mathrm { o u t } } + \theta _ { 3 } \tau _ { \mathrm { i n } } \tau _ { \mathrm { o u t } }
$$

Wilkins et al. motivate the interaction term statistically, using ANOVA to confirm that input and output lengths have a significant joint efect on energy, and fit the model empirically via ordinary least squares. Delavande et al. derive the same form analytically from the roofline model: the interaction term reflects the cumulative cost of attending to a growing context at each decode step. Neither model includes architectural hyperparameters, thus both must be fitted separately for each LLM and are not attention-aware.

Power-law Models Oviedo et al. [24] propose a bottomup framework that estimates per-query inference energy from node power draw and token throughput:

$$
E _ { \mathrm { t o t } } = \frac { \mathrm { P U E } } { 3 . 6 } \cdot \frac { P _ { \mathrm { n o d e } } \cdot L _ { \mathrm { e f f } } } { \mathrm { T P S } ( \tau _ { \mathrm { i n } } , \tau _ { \mathrm { o u t } } ) }
$$

where PUE accounts for data-center overhead, $P _ { \mathrm { n o d e } }$ is the measured steady-state node power, $\mathrm { T P S } ( \tau _ { \mathrm { i n } } , \tau _ { \mathrm { o u t } } )$ is fitted empirically to per-model benchmarks as a piecewise loglinear regression, and $L _ { \mathrm { e f f } } \approx \tau _ { \mathrm { o u t } }$ scales energy by output length. Because TPS is linear in log $\tau _ { \mathrm { i n } }$ and log $\tau _ { \mathrm { o u t } } ,$ substituting it into the formula yields a power-law dependence on sequence lengths:

$$
E _ { \mathrm { t o t } } \approx \theta _ { 0 } \tau _ { \mathrm { i n } } ^ { \theta _ { 1 } } \tau _ { \mathrm { o u t } } ^ { \theta _ { 2 } }
$$

The framework includes no architectural hyperparameters; thus the throughput model must be re-fitted for each LLM, making the approach LLM-specific and not attention-aware.

Quadratic Models SweetSpot [26] models inference energy as proportional to FLOPs and data transfers, analytically derived from the transformer computational kernels as a function of $\tau _ { \mathrm { i n } }$ and $\tau _ { \mathrm { o u t } }$ . The decoding phase is handled by expressing the growing KV-cache context as an arithmetic series, which yields a closed-form quadratic formulation:

$$
E _ { \mathrm { t o t } } \approx \theta _ { 0 } + \theta _ { 1 } \tau _ { \mathrm { i n } } + \theta _ { 2 } \tau _ { \mathrm { i n } } ^ { 2 } + \theta _ { 3 } \tau _ { \mathrm { i n } } \tau _ { \mathrm { o u t } } + \theta _ { 4 } \tau _ { \mathrm { o u t } } + \theta _ { 5 } \tau _ { \mathrm { o u t } } ^ { 2 }
$$

The model reduces the MAPE of the bilinear baseline from over 30% to below 2%. However, the fitted coeficients are LLM-specific, requiring an exhaustive measurement dataset for each target model. The approach therefore cannot generalize to unseen architectures or be applied during training. Moreover, the analytical derivation treats all �, $K .$ , and � projection matrices as having the same hidden dimension �, implicitly assuming standard MHA. Since the attention mechanism is not parameterized in the formulation, SweetSpot is not attention-aware.

Following a similar analytical approach, Vartziotis et al. [36] decompose inference energy into compute and memory-access components:

$$
E _ { \mathrm { t o t } } \approx \alpha _ { \mathrm { T C } } C _ { \mathrm { T C } } + e _ { \mathrm { H B M } } B \eta ( N )
$$

where $C _ { \mathrm { T C } }$ denotes the total tensor-core FLOP count for prefill and decode, and � denotes the total HBM trafic. $C _ { \mathrm { T C } }$ and � are derived analytically from the transformer computational graph, following a similar decomposition to [26], though the FLOP count $C _ { \mathrm { T C } }$ relies on the Kaplan training parameter-scaling approximation [16] (� ≈ ���, $K = 6$ FLOPs per parameter per token) rather than an inference kernel-by-kernel tally. The factor $\eta ( N )$ , where � is the number of model hyperparameters, is a dimensionless inefficiency multiplier that accounts for non-ideal HBM bandwidth utilization, while $\alpha _ { \mathrm { T C } }$ and $e _ { \mathrm { H B M } }$ are hardware-specific energy coeficients for tensor-core and HBM operations, expressed in pJ/FLOP and pJ/bit respectively (H100 GPU, FP16/BF16). Because $\alpha _ { \mathrm { T C } }$ and $e _ { \mathrm { H B M } }$ are hardware constants rather than fitted per-model coeficients, and $C _ { \mathrm { T C } }$ and � are expressed in terms of � and structural hyperparameters such as layer count and hidden dimension, the model is LLMagnostic and captures the order-of-magnitude energy and scaling trends observed in authors’ measurements.

However, its accuracy is limited, with a reported mean relative error of approximately 20%. Furthermore, expanding $C _ { \mathrm { T C } }$ and � yields a long closed-form expression in the architectural hyperparameters, which makes the model hard to interpret and limits its usefulness for identifying the hyperparameters that drive energy consumption. Similarly to SweetSpot, the model assumes MHA, so the reduced KVcache trafic of GQA, which is standard in many modern LLMs, is not captured by the model.

To the best of our knowledge, Niu et al. [22] is the only existing work to propose an attention-aware energy model for LLM inference. Their model estimates the energy per output token of a single decode step, expressing decode energy as a linear function of the bytes accessed, considering

RQ1: Is there an analytical model that can accurately predict inference energy across LLMs with diferent architectures, including the attention mechanism?

No. The most accurate model, SweetSpot [26], is LLMspecific: its coeficients must be fitted for each target LLM, so it cannot generalize to unseen architectures. LLM-agnostic alternatives do not explicitly model the attention mechanism; the most analytically grounded of these [36] reports a mean relative error of about 20% and yields hard-to-interpret expressions. The one model that parameterizes the attention mechanism [22] targets the decode phase only and requires measured runtime inputs, making it inapplicable to total inference energy prediction. No prior model is simultaneously interpretable, LLM-agnostic, attention-aware, and applicable to the full inference pipeline. SIMPLE is the first to satisfy all these conditions.

both the model weights and the KV-cache, and the measured decode time. The KV-cache byte term is analytically derived from the model architecture and attention mechanism: it is explicitly parametrized by the number of KV heads per layer $( N _ { k v } )$ , which accounts for MHA and GQA. The decode time, however, must be measured at runtime. The resulting model requires a separate calibration and fit for each LLM and hardware configuration. Since the model covers only the decode phase energy, excluding the prefill, it is not directly applicable to predict the total inference energy $E _ { \mathrm { t o t } }$ for a given query, which is the focus of this work. We therefore do not consider it further here.

## 3.3. SIMPLE Model

The box above summarizes the answer to RQ1. We address the key limitations of previous approaches by extending the derivation of the computational and memoryaccess complexities of SweetSpot [26] to GQA, and using the resulting expressions to derive SIMPLE, a compact LLM-agnostic model that explicitly represents the attention mechanism, builds on SweetSpot’s interpretable quadratic structure to expose the role of individual architectural hyperparameters, and predicts energy across multiple LLM architectures without per-LLM calibration.

Throughout our study, we target the Energy-per-Token $E _ { \mathrm { t o k } } .$ rather than total energy consumption $E _ { \mathrm { t o t } } ,$ since normalizing by $\tau _ { \mathrm { o u t } }$ isolates the cost of generation from the cost of processing the input, exposing the trade-of between amortizing the prefill cost over more output tokens and incurring additional decode cost as $\tau _ { \mathrm { o u t } }$ grows. Moreover, $E _ { \mathrm { t o k } }$ serves as a natural unit for comparison across diferent workloads and deployments, as it corresponds directly to operational metrics such as cost-per-token and tokens-perjoule, matching how inference is commonly priced and provisioned in production.

Table 2  
Decoder-only LLM inference symbols and tensor dimensions.
<table><tr><td>Symbol</td><td>Dimension</td><td>Description</td><td>Symbol</td><td>Dimension</td><td>Description</td></tr><tr><td>Scalars</td><td></td><td></td><td>Prefill Phase</td><td></td><td></td></tr><tr><td> $\tau _ { \mathrm { i n } }$ </td><td> $\mathbb { N } ^ { 1 \times 1 }$ </td><td>Number of input tokens</td><td> $\mathbf { X } _ { P }$ </td><td> $\mathbb { R } ^ { \tau _ { \mathrm { i n } } \times d }$ </td><td>Input</td></tr><tr><td> $\tau _ { \mathrm { { o u t } } }$ </td><td> $\mathbb { N } ^ { 1 \times 1 }$ </td><td>Number of output tokens</td><td> $\mathbf { Q } _ { P } , \mathbf { O } _ { P }$ </td><td> $\mathbb { R } ^ { \tau _ { \mathrm { i n } } \times d }$ </td><td>Query, Output projections</td></tr><tr><td> $n _ { \mathfrak { q } }$ </td><td> $\mathbb { N } ^ { 1 \times 1 }$ </td><td>Number of query heads</td><td> ${ \bf K } _ { P } , { \bf V } _ { P }$ </td><td> $\mathbb { R } ^ { \tau _ { \mathrm { i n } } \times d _ { \mathrm { k v } } }$ </td><td> $\mathsf { K e y } ,$  Value projections</td></tr><tr><td> $n _ { \mathrm { k v } }$ </td><td> $\mathbb { N } ^ { 1 \times 1 }$ </td><td>Number of KV heads</td><td> $\mathbf { S } _ { P }$ </td><td> ${ \mathbb { R } } ^ { \tau _ { \mathrm { i n } } \times \tau _ { \mathrm { i n } } }$ </td><td>Attention scores</td></tr><tr><td> $L$ </td><td> $\mathbb { N } ^ { 1 \times 1 }$ </td><td>Number of layers</td><td> $\mathbf { A } _ { P }$ </td><td> $\mathbb { R } ^ { \tau _ { \mathrm { i n } } \times d }$ </td><td>Attention output</td></tr><tr><td> $d$ </td><td> $\mathbb { N } ^ { 1 \times 1 }$ </td><td>Hidden dimension</td><td> $\mathbf { H } _ { P }$ </td><td> $\mathbb { R } ^ { \tau _ { \mathrm { i n } } \times 4 d }$ </td><td>Expanded activation</td></tr><tr><td> $\begin{array} { r } { R = \frac { n _ { \mathrm { k v } } } { n _ { \mathrm { q } } } } \end{array}$ </td><td> $\mathbb { R } ^ { 1 \times 1 }$ </td><td>KV-to-query head ratio</td><td> ${ \bf Y } _ { P }$ </td><td> $\mathbb { R } ^ { \tau _ { \mathrm { i n } } \times d }$ </td><td>FFN output</td></tr><tr><td> $d _ { \mathsf { k v } } = d \ R$ </td><td> $\mathbb { R } ^ { 1 \times 1 }$ </td><td> $\mathsf { K e y } ,$  Value dimension</td><td></td><td></td><td></td></tr><tr><td colspan="2">Parameters</td><td></td><td>Decode Phase</td><td></td><td></td></tr><tr><td> ${ \bf W } _ { Q }$ </td><td> $\mathbb { R } ^ { d \times d }$ </td><td>Query weight</td><td> $\mathbf { X } _ { D }$ </td><td> $\mathbb { R } ^ { 1 \times d }$ </td><td>Input</td></tr><tr><td> $\mathbf { W } _ { K } , \mathbf { W } _ { V }$ </td><td> $\mathbb { R } ^ { d \times d _ { \mathsf { k v } } }$ </td><td>Key, Value weights</td><td> $\mathbf { Q } _ { D } , \mathbf { O } _ { D }$ </td><td> $\mathbb { R } ^ { 1 \times d }$ </td><td>Query, Output projections</td></tr><tr><td> ${ \bf W } _ { O }$ </td><td> $\mathbb { R } ^ { d \times d }$ </td><td>Output weight</td><td> $\mathbf { K } _ { D } , \mathbf { V } _ { D }$ </td><td> $\mathbb { R } ^ { 1 \times d _ { \mathsf { k v } } }$ </td><td> $\mathsf { K e y } ,$  Value projections</td></tr><tr><td> $\mathbf { W } _ { 1 }$ </td><td> $\mathbb { R } ^ { d \times 4 d }$ </td><td>FFN expansion</td><td> $\mathbf { K } _ { D _ { \mathrm { c a c h e } } } , \mathbf { V } _ { D _ { \mathrm { c a c h e } } }$ </td><td> $\mathbb { R } ^ { ( \tau _ { \mathrm { i n } } + t - 1 ) \times d _ { \mathrm { k v } } }$ </td><td> $\mathsf { K e y } ,$  Value cache</td></tr><tr><td> $\mathbf { W } _ { 2 }$ </td><td> $\mathbb { R } ^ { 4 d \times d }$ </td><td>FFN projection</td><td> $\mathbf { S } _ { D }$ </td><td> $\mathbb { R } ^ { 1 \times ( \tau _ { \mathrm { i n } } + t - 1 ) }$ </td><td>Attention scores</td></tr><tr><td></td><td></td><td></td><td> $\mathbf { A } _ { D }$ </td><td> $\mathbb { R } ^ { 1 \times d }$ </td><td>Attention output</td></tr><tr><td></td><td></td><td></td><td> $\mathbf { H } _ { D }$ </td><td> $\mathbb { R } ^ { 1 \times 4 d }$ </td><td>Expanded activation</td></tr><tr><td></td><td></td><td></td><td> $\mathbf { Y } _ { D }$ </td><td> $\mathbb { R } ^ { 1 \times d }$ </td><td>FFN output</td></tr></table>

## 4. FLOPs and MemOPs Formulation

In this section, we present the analytical derivation of the Floating-Point Operations (FLOPs) and Memory Operations (MemOPs) of LLM inference based on a decoder-only Transformer, distinguishing the prefill and decode phases and assuming no quantization of weights or KV-cache. In the following derivations, we consider a dense LLM and a constant batch size, extending the derivation of SweetSpot [26] to include the GQA mechanism.

For a matrix multiplication between $\textbf { A } \in \ \mathbb { R } ^ { m \times k }$ and $\mathbf { B } \in \mathbb { R } ^ { k \times n }$ , we denote the number of FLOPs and MemOPs as $F$ and � respectively, calculated as:

$$
\begin{array} { l } { F = 2 m n k } \\ { M = m k + k n + m n \quad { \mathrm { ( r e a d ~ } } \mathbf { A } { \mathrm { , ~ r e a d ~ } } \mathbf { B } { \mathrm { , ~ w r i t e ~ r e s u l t ) } } } \end{array}
$$

We neglect bias additions, layer normalization, softmax, and dropout, as their cost is negligible relative to projections and FFN layers. Moreover, we treat the FFN expansion factor as a fixed constant (4×), consistent with the original decoderonly Transformer design and still the dominant convention across dense LLM architectures, even though some models in our evaluation set adopt slightly diferent ratios.

Table 2 describes the symbols we use throughout the paper. In order to model both MHA and GQA, we define the KV-to-query head ratio � as the ratio between the number of KV heads $n _ { \mathrm { k v } }$ and the number of query heads $n _ { \mathrm { q } } .$ We consider the KV projections as weight matrices of shape $d \times d \cdot R = d \times d _ { \mathrm { k v } }$ , reflecting that in GQA only $n _ { \mathrm { k v } } \leq n _ { \mathrm { q } }$ key and value heads are materialized. For MHA, $n _ { \mathrm { k v } } = n _ { \mathrm { q } }$ and all expressions reduce to the standard form.

## 4.1. Prefill FLOPs and MemOPs

During prefill, the full input sequence of length $\tau _ { \mathrm { i n } }$ is processed in parallel.

Attention Projections $Q _ { P }$ projection $X _ { P } W _ { Q } .$

$$
\begin{array} { r l } & { F _ { Q _ { P } } = 2 \tau _ { \mathrm { i n } } d ^ { 2 } } \\ & { M _ { Q _ { P } } = \tau _ { \mathrm { i n } } d + d ^ { 2 } + \tau _ { \mathrm { i n } } d = 2 \tau _ { \mathrm { i n } } d + d ^ { 2 } } \end{array}\tag{1}
$$

� , � projections $X _ { P } W _ { K }$ and $X _ { P } W _ { V } $

$$
\begin{array} { r l } & { F _ { K _ { P } } = F _ { V _ { P } } = 2 \tau _ { \mathrm { i n } } d ^ { 2 } R } \\ & { M _ { K _ { P } } = F _ { V _ { P } } = \tau _ { \mathrm { i n } } d + d ^ { 2 } R + \tau _ { \mathrm { i n } } d R } \end{array}\tag{2}
$$

Output projection $X _ { P } W _ { O } \mathrm { : }$

$$
\begin{array} { c } { { F _ { O _ { P } } = 2 \tau _ { \mathrm { i n } } d ^ { 2 } } } \\ { { { } } } \\ { { M _ { O _ { P } } = 2 \tau _ { \mathrm { i n } } d + d ^ { 2 } } } \end{array}\tag{3}
$$

Total prefill projection complexity:

$$
\begin{array} { r l } & { F _ { P _ { \mathrm { p r o j } } } = 2 \tau _ { \mathrm { i n } } d ^ { 2 } ( 2 + 2 R ) = 4 \tau _ { \mathrm { i n } } d ^ { 2 } ( 1 + R ) } \\ & { M _ { P _ { \mathrm { p r o j } } } = \tau _ { \mathrm { i n } } d ( 6 + 2 R ) + d ^ { 2 } ( 2 + 2 R ) } \end{array}
$$

## Attention Scores Attention scores $Q _ { P } K _ { P } ^ { T } ;$

$$
\begin{array} { r c l } { { } } & { { } } & { { F _ { S _ { P } } = 2 n _ { \mathrm { q } } \tau _ { \mathrm { i n } } ^ { 2 } \frac { d } { n _ { \mathrm { q } } } = 2 \tau _ { \mathrm { i n } } ^ { 2 } d } } \\ { { } } & { { } } & { { } } \\ { { } } & { { } } & { { M _ { S _ { P } } = \tau _ { \mathrm { i n } } d + \tau _ { \mathrm { i n } } d R + \tau _ { \mathrm { i n } } ^ { 2 } n _ { \mathrm { q } } } } \end{array}\tag{4}
$$

Attention output $S _ { P } V _ { P }$ over $n _ { \mathrm { q } }$ heads:

$$
\begin{array} { r l } & { F _ { A _ { P } } = 2 \tau _ { \mathrm { i n } } ^ { 2 } d } \\ & { M _ { A _ { P } } = \tau _ { \mathrm { i n } } ^ { 2 } n _ { \mathrm { q } } + \tau _ { \mathrm { i n } } d R + \tau _ { \mathrm { i n } } d } \end{array}\tag{5}
$$

Feed-Forward Network FFN output $A _ { P } W _ { 1 }$ and $H _ { P } W _ { 2 } { \mathrm { : } }$

$$
\begin{array} { r l } & { F _ { Y _ { P } } = 2 \times 2 \tau _ { \mathrm { i n } } d \cdot 4 d = 1 6 \tau _ { \mathrm { i n } } d ^ { 2 } } \\ & { M _ { Y _ { P } } = \tau _ { \mathrm { i n } } d + 4 d ^ { 2 } + 4 \tau _ { \mathrm { i n } } d + 4 \tau _ { \mathrm { i n } } d + 4 d ^ { 2 } + \tau _ { \mathrm { i n } } d } \\ & { \qquad = 1 0 \tau _ { \mathrm { i n } } d + 8 d ^ { 2 } } \end{array}\tag{6}
$$

Total OPs With � layers, the total prefill FLOPs is:

$$
\begin{array} { r l } & { F _ { \mathrm { p r e f i l l } } = L \Big [ F _ { P _ { \mathrm { p r o j } } } + F _ { S _ { P } } + F _ { A _ { P } } + F _ { Y _ { P } } \Big ] } \\ & { \qquad = L \big [ 4 \tau _ { \mathrm { i n } } d ^ { 2 } ( 5 + R ) + 4 \tau _ { \mathrm { i n } } ^ { 2 } d \big ] \quad } \end{array}\tag{7}
$$

The total prefill MemOPs is:

$$
\begin{array} { r l } & { M _ { \mathrm { p r e f i l l } } = L \Big [ M _ { P _ { \mathrm { p r o j } } } + M _ { S _ { P } } + M _ { A _ { P } } + M _ { Y _ { P } } \Big ] } \\ & { \qquad = L \left[ \tau _ { \mathrm { i n } } d ( 1 8 + 4 R ) + d ^ { 2 } ( 1 0 + 2 R ) + 2 \tau _ { \mathrm { i n } } ^ { 2 } n _ { \mathrm { q } } \right] } \end{array}\tag{8}
$$

## 4.2. Decode FLOPs and MemOPs

At each decode step $t ~ \in ~ [ 1 , \dots , \tau _ { \mathrm { o u t } } ] .$ , one new token is generated autoregressively. The KV-cache at step � holds $( \tau _ { \mathrm { i n } } + t - 1 )$ tokens: the $\tau _ { \mathrm { i n } }$ input tokens and � − 1 previously generated tokens.

Attention Projections $Q _ { D }$ projection $X _ { D } W _ { Q }$

$$
\begin{array} { c } { { F _ { Q _ { D } } = 2 d ^ { 2 } } } \\ { { M _ { Q _ { D } } = d + d ^ { 2 } + d = 2 d + d ^ { 2 } } } \end{array}\tag{9}
$$

$K _ { D } , V _ { D }$ projection $X _ { D } W _ { K }$ and $X _ { D } W _ { V }$

$$
\begin{array} { c } { { F _ { K _ { D } } = F _ { V _ { D } } = 2 d ^ { 2 } R } } \\ { {  } } \\ { { M _ { K _ { D } } = F _ { V _ { D } } = d + d ^ { 2 } R + d R } } \end{array}\tag{10}
$$

Output projection $X _ { D } W _ { O }$

$$
\begin{array} { c } { { F _ { O _ { D } } = 2 d ^ { 2 } } } \\ { { { } } } \\ { { M _ { O _ { D } } = 2 d + d ^ { 2 } } } \end{array}\tag{11}
$$

Total decode projection complexity:

$$
\begin{array} { r l } & { F _ { D _ { \mathrm { p r o j } } } = 2 d ^ { 2 } \left( 2 + 2 R \right) = 4 d ^ { 2 } ( 1 + R ) } \\ & { M _ { D _ { \mathrm { p r o j } } } = d \left( 6 + 2 R \right) + d ^ { 2 } \left( 2 + 2 R \right) } \end{array}
$$

Attention Scores Attention scores $Q _ { D } K _ { D _ { \mathrm { c a c h e } } } ^ { T }$

$$
\begin{array} { r l } & { F _ { S _ { D } } ( t ) = 2 ( \tau _ { \mathrm { i n } } + t - 1 ) d } \\ & { M _ { S _ { D } } ( t ) = d + ( \tau _ { \mathrm { i n } } + t - 1 ) d R + ( \tau _ { \mathrm { i n } } + t - 1 ) n _ { \mathrm { q } } } \end{array}\tag{12}
$$

Attention output $S _ { D } V _ { D _ { \mathrm { c a c h e } } }$ over $n _ { \mathrm { q } }$ heads:

$$
\begin{array} { r l } & { F _ { A _ { D } } ( t ) = 2 ( \tau _ { \mathrm { i n } } + t - 1 ) d } \\ & { M _ { A _ { D } } ( t ) = ( \tau _ { \mathrm { i n } } + t - 1 ) n _ { \mathrm { q } } + ( \tau _ { \mathrm { i n } } + t - 1 ) d R + d } \end{array}\tag{13}
$$

Feed-Forward Network FFN output $A _ { D } W _ { 1 }$ and $H _ { D } W _ { 2 } { \mathrm { : } }$

$$
\begin{array} { c } { { F _ { Y _ { D } } ( t ) = 1 6 d ^ { 2 } } } \\ { { M _ { Y _ { D } } ( t ) = d + 4 d ^ { 2 } + 4 d + 4 d + 4 d ^ { 2 } + d } } \\ { { = 1 0 d + 8 d ^ { 2 } } } \end{array}\tag{14}
$$

Total OPs Using the closed-form sum:

$$
\sum _ { t = 1 } ^ { \tau _ { \mathrm { o u t } } } t = \frac { \tau _ { \mathrm { o u t } } ( \tau _ { \mathrm { o u t } } + 1 ) } { 2 }
$$

and summing over $\tau _ { \mathrm { o u t } }$ generated tokens and � layers, we obtain the total decode FLOP count:

$$
\begin{array} { l } { { \displaystyle F _ { \mathrm { d e c o d e } } = L \sum _ { t = 1 } ^ { \tau _ { \mathrm { o u t } } } \left[ F _ { D _ { \mathrm { p r o j } } } + F _ { S _ { D } } ( t ) + F _ { A _ { D } } ( t ) + F _ { Y _ { D } } ( t ) \right] } } \\ { ~ = L \sum _ { t = 1 } ^ { \tau _ { \mathrm { o u t } } } \left[ 4 d ^ { 2 } ( 5 + R ) + 4 d ( \tau _ { \mathrm { i n } } + t - 1 ) \right] } \\ { ~ = L \left[ 4 d ^ { 2 } \tau _ { \mathrm { o u t } } \left( 5 + R \right) + 4 d \left( \tau _ { \mathrm { o u t } } \tau _ { \mathrm { i n } } + \frac { \tau _ { \mathrm { o u t } } ( \tau _ { \mathrm { o u t } } - 1 ) } { 2 } \right) \right] } \end{array}\tag{15}
$$

Similarly, we obtain the total decode MemOPs:

$$
\begin{array} { l } { { \displaystyle M _ { \mathrm { d e c o d e } } = L \sum _ { t = 1 } ^ { \tau _ { \mathrm { o u t } } } \left[ M _ { D _ { \mathrm { p r o j } } } + M _ { S _ { D } } + M _ { A _ { D } } + M _ { Y _ { D } } \right] } } \\ { ~ } \\ { { \displaystyle ~ = L \left[ \tau _ { \mathrm { o u t } } d ( 1 8 + 2 R ) + \tau _ { \mathrm { o u t } } d ^ { 2 } ( 1 0 + 2 R ) \right. } } \\ { { \displaystyle ~ \left. + 2 \left( n _ { \mathrm { q } } + d R \right) \left( \tau _ { \mathrm { o u t } } \tau _ { \mathrm { i n } } + \frac { \tau _ { \mathrm { o u t } } ( \tau _ { \mathrm { o u t } } - 1 ) } { 2 } \right) \right] } } \end{array}\tag{16}
$$

## 5. SIMPLE Derivation

In this section, we present the formulation of SIMPLE, the proposed analytical model for predicting the energy eficiency of LLM inference, for a generic LLM architecture $f ( d , L , n _ { \mathrm { q } } , n _ { \mathrm { k v } } )$ and the input/output sequence $f ( \tau _ { \mathrm { i n } } , \tau _ { \mathrm { o u t } } )$ The model is built on the complexity analysis introduced in the previous section, more precisely on Equations (1–14). These equations ofer diferent valid groupings of their inner terms according to the number of fitting coeficients, each implying a diferent interpretation for each coeficient.

A first approach we propose is to assign a meaning to each coeficient in terms of Joule/FLOPs or Joule/MemOPs. We refer to these models as OPs-centric. A second orthogonal approach is to drop the interpretation but preserve only the relational one with respect to $f ( \tau _ { \mathrm { i n } } , \tau _ { \mathrm { o u t } } )$ . We refer to these models as IO-centric. In these formulations, we include the LLM architectural hyperparameters $( d , L , n _ { \mathrm { q } } , n _ { \mathrm { k v } } )$ in order to obtain LLM-agnostic models. While the OPscentric models serve as an exploratory baseline, SIMPLE is ultimately derived within the IO-centric family.

The proposed models are summarized in Table 3. As direct comparisons, we include the $E _ { \mathrm { t o k } }$ formulations of SweetSpot [26] and the two LLM-agnostic models from related work: Vartziotis et al. [36], which can be classified as OPs-centric as it explicitly assigns meaning to its terms, and Zhan et al. [43], which falls among the IO-centric models.

## 5.1. OPs-centric Approach

We formulate three models that assign a meaning to each coeficient in terms of Joule/FLOPs and Joule/MemOPs at diferent levels of granularity.

Summary of the proposed energy models and selected baselines for predicting the energy per generated token, $E _ { \mathrm { t o k } }$ . SweetSpot is included for direct comparison. For the LLM-agnostic SweetSpot model, $\phi _ { 0 } , \ldots , \phi _ { 5 }$ denote architecture-dependent analytical coeficients, whose explicit expressions are given in Equation 20.
<table><tr><td>Approach</td><td>Model</td><td># coeff.</td><td>Energy per token,  $E _ { \mathbf { t o k } }$ </td><td></td></tr><tr><td rowspan="5">OPs-centric</td><td>Vartziotis et al. [36]</td><td>6</td><td> $\frac { \alpha _ { \mathsf { T C } } C _ { \mathsf { T C } } } { \tau _ { \mathsf { o u t } } } + \frac { e _ { \mathsf { H B M } } B \eta ( N ) } { \tau _ { \mathsf { o u t } } }$ </td><td></td></tr><tr><td>FLOPs-MemOPs</td><td>3</td><td> $\theta _ { 0 } + \theta _ { 1 } \frac { F _ { \mathrm { p r e f i l l } } + F _ { \mathrm { d e c o d e } } } { \tau _ { \mathrm { o u t } } } + \theta _ { 2 } \frac { M _ { \mathrm { p r e f i l l } } + M _ { \mathrm { d e c o d e } } } { \tau _ { \mathrm { o u t } } }$ </td><td></td></tr><tr><td>Prefill-Decode</td><td>5</td><td> $\theta _ { 0 } + \theta _ { 1 } \frac { F _ { \mathrm { p r e f i l } } | } { \tau _ { \mathrm { o u t } } } + \theta _ { 2 } \frac { F _ { \mathrm { d e c o d e } } } { \tau _ { \mathrm { o u t } } } + \theta _ { 3 } \frac { M _ { \mathrm { p r e f i l } } | } { \tau _ { \mathrm { o u t } } } + \theta _ { 4 } \frac { M _ { \mathrm { d e c o d e } } } { \tau _ { \mathrm { o u t } } }$ </td><td></td></tr><tr><td>Fine-Grained</td><td>25</td><td></td><td> $\begin{array} { r l r } { \theta _ { 0 } + \sum _ { i = 1 } ^ { 2 4 } \theta _ { i } \frac { L C _ { i } } { \tau _ { \mathrm { o u t } } } } & { { } } & { C _ { i } \in \left\{ F _ { Q _ { P } } , \dots , \sum _ { t = 1 } ^ { \tau _ { \mathrm { o u t } } } M _ { A _ { D } } ( t ) \right\} } \end{array}$ </td></tr><tr><td>Zhan et al. [43]</td><td>5</td><td> $\theta _ { 0 } + \theta _ { 1 } S \frac { 1 } { \tau _ { \mathrm { o u t } } } + \theta _ { 2 } \frac { \tau _ { \mathrm { i n } } } { \tau _ { \mathrm { o u t } } } + \theta _ { 3 } S \frac { \tau _ { \mathrm { i n } } } { \tau _ { \mathrm { o u t } } } + \theta _ { 4 } S$ </td><td></td></tr><tr><td rowspan="3">IO-centric</td><td>SweetSpot [26]</td><td>6</td><td> $\theta _ { 0 } + \theta _ { 1 } \frac { \tau _ { \mathrm { i n } } ^ { 2 } } { \tau _ { \mathrm { o u t } } } + \theta _ { 2 } \tau _ { \mathrm { i n } } + \theta _ { 3 } \frac { \tau _ { \mathrm { i n } } } { \tau _ { \mathrm { o u t } } } + \theta _ { 4 } \tau _ { \mathrm { o u t } } + \frac { \theta _ { 5 } } { \tau _ { \mathrm { o u t } } }$ </td><td></td></tr><tr><td>LLM-agnostic SweetSpot</td><td>7</td><td> $\theta _ { 0 } + \theta _ { 1 } \frac { \tau _ { \mathrm { i n } } ^ { 2 } } { \tau _ { \mathrm { o u t } } } \phi _ { 0 } + \theta _ { 2 } \tau _ { \mathrm { i n } } \phi _ { 1 } + \theta _ { 3 } \frac { \tau _ { \mathrm { i n } } } { \tau _ { \mathrm { o u t } } } \phi _ { 2 } + \theta _ { 4 } \tau _ { \mathrm { o u t } } \phi _ { 3 } + \frac { \theta _ { 5 } } { \tau _ { \mathrm { o u t } } } \phi _ { 4 } + \theta _ { 6 } \phi _ { 5 }$ </td><td></td></tr><tr><td>SIMPLE</td><td>4</td><td> $\theta _ { 0 } + \theta _ { 1 } d ^ { 2 } L \ \frac { \tau _ { \mathrm { i n } } } { \tau _ { \mathrm { o u t } } } + \theta _ { 2 } d L R \Big ( \tau _ { \mathrm { i n } } + \frac { \tau _ { \mathrm { o u t } } } { 2 } \Big ) + \theta _ { 3 } d L$ </td><td></td></tr></table>

FLOPs-MemOPs The first formulation assigns one coefficient to Equations (7, 15) and another to Equations (8, 16):

$$
E _ { \mathrm { t o k } } = \theta _ { 0 } + \theta _ { 1 } \frac { F _ { \mathrm { p r e f i l l } } + F _ { \mathrm { d e c o d e } } } { \tau _ { \mathrm { o u t } } } + \theta _ { 2 } \frac { M _ { \mathrm { p r e f i l l } } + M _ { \mathrm { d e c o d e } } } { \tau _ { \mathrm { o u t } } }\tag{17}
$$

The model assumes that these two components contribute independently to the total inference energy, assigning a separate cost to each, regardless of whether the operations are performed in the prefill or the decode phase.

Prefill-Decode The second formulation diferentiate these energy costs by prefill and decode phases, and assigns individual coeficients to Equations (7, 15, 8, 16), allowing the model to capture potentially diferent operating conditions and energy costs of the two phases:

$$
E _ { \mathrm { t o k } } = \theta _ { 0 } + \theta _ { 1 } \frac { F _ { \mathrm { p r e f i l l } } } { \tau _ { \mathrm { o u t } } } + \theta _ { 2 } \frac { F _ { \mathrm { d e c o d e } } } { \tau _ { \mathrm { o u t } } } + \theta _ { 3 } \frac { M _ { \mathrm { p r e f i l l } } } { \tau _ { \mathrm { o u t } } } + \theta _ { 4 } \frac { M _ { \mathrm { d e c o d e } } } { \tau _ { \mathrm { o u t } } }\tag{18}
$$

Fine-Grained The third formulation maximizes the fitting granularity assigning a coeficient to each intermediate term, but perishing the physical meaning of Joule/FLOPs and Joule/MemOPs:

$$
E _ { \mathrm { t o k } } = \theta _ { 0 } + \sum _ { i = 1 } ^ { 2 4 } \theta _ { i } \frac { L C _ { i } } { \tau _ { \mathrm { o u t } } } ,\tag{19}
$$

where:

$$
C _ { i } \in \left\{ F _ { Q _ { P } } , M _ { Q _ { P } } , \ldots \sum _ { t = 1 } ^ { \tau _ { \mathrm { o u t } } } F _ { A _ { D } } ( t ) , \sum _ { t = 1 } ^ { \tau _ { \mathrm { o u t } } } M _ { A _ { D } } ( t ) \right\}
$$

denotes the complexity of each intermediate term across prefill and decode from Equations (1–6, 9–14).

## 5.2. IO-centric Approach

The previous models explore diferent ways of incorporating the analytical FLOPs and MemOPs into the energy prediction model. While these formulations explicitly account for the underlying computational cost, they also introduce a growing number of terms and fitting coeficients. The IO-centric approach instead groups the complexity terms according to their dependency on $f ( \tau _ { \mathrm { i n } } , \tau _ { \mathrm { o u t } } )$ , simplifying or discarding those of negligible contribution.

In the following paragraphs, we first introduce LLMagnostic SweetSpot, a natural extension of SweetSpot, [26] used as a comparison baseline, then present SIMPLE.

LLM-agnostic SweetSpot The SweetSpot model [26] captures quadratic dependencies on $f ( \tau _ { \mathrm { i n } } , \tau _ { \mathrm { o u t } } )$ , but is fitted for individual LLMs. All architectural information is implicitly absorbed into the learned coeficients, making it impossible to generalize across diferent LLM architectures. A natural extension is therefore to preserve SweetSpot’s functional dependencies on $f ( \tau _ { \mathrm { i n } } , \tau _ { \mathrm { o u t } } )$ while making the coeficients explicitly dependent on the LLM architecture $f ( d , L , n _ { \mathrm { q } } , n _ { \mathrm { k v } } )$ . To this end, we group the complexity terms from Equations (7, 8, 15, 16) to match the corresponding terms of the SweetSpot formulation reported in Table 3.

$$
\begin{array} { l } { { E _ { \mathrm { t o k } } = \theta _ { 0 } + \theta _ { 1 } \displaystyle \frac { \tau _ { \mathrm { i n } } ^ { 2 } } { \tau _ { \mathrm { o u t } } } \ { L \left( { 4 d + 2 n _ { \mathrm { q } } } \right) } + \theta _ { 2 } \tau _ { \mathrm { i n } } { L \left( { 4 d + 2 d R + 2 n _ { \mathrm { q } } } \right) } } } \\ { { { } ~ + \theta _ { 3 } \displaystyle \frac { \tau _ { \mathrm { i n } } } { \tau _ { \mathrm { o u t } } } { L \left[ { d ^ { 2 } \left( { 4 R + 2 0 } \right) + d \left( { 4 R + 1 8 } \right) } \right] } } } \\ { { { } ~ + \theta _ { 4 } \tau _ { \mathrm { o u t } } { L \left( { 2 d + d R + n _ { \mathrm { q } } } \right) } + \displaystyle \frac { \theta _ { 5 } } { \tau _ { \mathrm { o u t } } } { L d ^ { 2 } \left( { 2 R + 1 0 } \right) } } } \\ { { { } ~ + \theta _ { 6 } { L \left[ { d ^ { 2 } \left( { 6 R + 3 0 } \right) + d \left( { 2 R + 1 8 } \right) - 2 d - \left( { d R + n _ { \mathrm { q } } } \right) } \right] } } } \end{array}\tag{20}
$$

The resulting formulation preserves the SweetSpot dependencies while explicitly incorporating the architectural complexity of the target LLM, thereby enabling generalization across multiple LLM architectures. However, despite sharing a similar number of fitting coeficients and functional dependencies with SweetSpot, this formulation involves a considerable number of terms, similarly to the previously derived Fine-Grained model, since it groups complexity terms without assessing their individual significance or exploring optimal grouping strategies. As a consequence, it provides limited insights into how each architectural parameter and inference phase contributes to energy consumption.

SIMPLE To obtain a compact, interpretable, and generalizable model, we systematically evaluated diferent simplification and term-aggregation strategies, starting from Equations (7, 8, 15, 16). In particular, we adopted three complementary methods to refine the model. First, we performed term ablations, removing inner terms with low statistical significance. Second, we simplified the resulting expressions by discarding lower-order terms. Third, we explored different grouping strategies for the remaining terms. Based on these refinement methods, we derived the following $E _ { \mathrm { t o t } }$ formulation of SIMPLE:

$$
E _ { \mathrm { t o t } } = \underbrace { \theta _ { 1 } d ^ { 2 } L \tau _ { \mathrm { i n } } } _ { \mathrm { P r e f i l l } } + \underbrace { \overbrace { \theta _ { 2 } d L R \Big ( \tau _ { \mathrm { i n } } \tau _ { \mathrm { o u t } } + \frac { \tau _ { \mathrm { o u t } } ^ { 2 } } { 2 } \Big ) } ^ { \mathrm { K V - c a c h e } } + \overbrace { ( \theta _ { 0 } + \theta _ { 3 } d L ) \tau _ { \mathrm { o u t } } } ^ { \mathrm { N e x t - t o k e n } } } _ { \mathrm { D e c o d e } } ,
$$

where the total energy consumption combines the cost of the Prefill and Decode phases.

The Prefill term accounts for processing the input prompt and is obtained by refining Equations (7, 8). It scales linearly with the input length $\tau _ { \mathrm { i n } }$ and the number of layers �, and quadratically with the hidden dimension �. The Decode component covers the generation of the output tokens and is obtained by refining Equations (15, 16). It comprises two terms. The KV-cache term captures the cost of reading the KV-cache, which grows as the context lengthens with each generated token. The Next-token term represents the cost that accumulates at a constant rate per generated token. Both terms scale linearly with � and �, while the KV-cache term additionally scales with the KV-to-query head ratio �.

The fitting coeficients $\theta _ { i }$ weigh each term independently, assigning a distinct cost to each phase and separating the KV-cache cost from the Next-token cost. Dividing by the number of generated tokens $\tau _ { \mathrm { o u t } }$ yields the per-token formulation $E _ { \mathrm { t o k } }$ of SIMPLE:

$$
E _ { \mathrm { t o k } } = \underbrace { \overbrace { \underbrace { \theta _ { 1 } d ^ { 2 } L \ { \frac { \tau _ { \mathrm { i n } } } { \tau _ { \mathrm { o u t } } } } } _ { \mathrm { P r e f i l ~ A m o r t i z a t i o n } } } ^ { \mathrm { K V - c a c h e } } } _ { \mathrm { P r e f i l l ~ A m o r t i z a t i o n } } + \underbrace { \overbrace { \left( { \theta _ { 2 } d L R \left( \tau _ { \mathrm { i n } } + { \frac { \tau _ { \mathrm { o u t } } } { 2 } } \right) } \right) } ^ { \mathrm { K V - t o k e n } } + \overbrace { \left( { \theta _ { 0 } + \theta _ { 3 } d L } \right) } ^ { \mathrm { N e x t - t o k e n } } } _ { \mathrm { D e c o d e ~ A c c u m u l a t i o n } }\tag{21}
$$

The Prefill term becomes a Prefill Amortization term, reflecting how the fixed cost of processing the prompt is spread over the generated tokens. The Decode term becomes a Decode Accumulation term, in which the KV-cache cost grows linearly with the context length, and the Next-token cost reduces to a constant per generated token.

## 6. Experimental Validation

This section details the experimental study conducted to validate SIMPLE. Section 6.1 describes the dataset <sup>1</sup> introduced in [26] and reused in our experiments, summarizing its key observations on energy consumption, namely the relevance of the attention mechanism and the non-linear relationship with input and output lengths. Section 6.2 presents our results <sup>2</sup>.

## 6.1. Dataset Description

The dataset contains energy measurements of TensorRT-LLM inference traces collected from a diverse set of widely used dense LLMs running on NVIDIA H100 accelerators. The included models span a range of LLM sizes and architectures, including MHA and on GQA with diferent KV grouping ratios, using a standard bf16 format for weights and KV-cache. The full list of models is given in Table 4.

The dataset comprises synthetic requests processed using trtllm-bench configured in throughput mode, with request concurrency set to 1000 and a maximum batch size of 1024. The maximum sequence length and the maximum number of batched tokens are both set to 8192. The experiments were conducted using one GPU per run on a server equipped with an Intel Xeon Platinum 8480+ CPU and four NVIDIA H100 Tensor Core GPUs connected via NVLink. Input and output lengths independently span 64– 4096 tokens in powers of two, yielding 49 combinations with 1000 requests each. GPU power was sampled every 500 ms using pynvml [8]. The per-query energy was obtained by multiplying the average GPU power over 1000-request by the total measurement time, and dividing by the number of requests. Since neither the baseline models nor the SIMPLE model explicitly accounts for batching, we use these fixed settings, and fit energy per individual request.

Figure 1 reports the average tokens per joule computed on the 49 diferent workload configurations across the different LLMs in the dataset. Smaller models and GQA-based architectures generally achieved higher eficiency (Figure 1): Llama 3.2 1B reached 55.91 tok/J, compared with 5.12 tok/J for Falcon-RW 7B. Among similarly sized models, Qwen2 1.5B (GQA R=1/7) achieved 49.06 tok/J versus 20.66 tok/J for OPT 1.3B (MHA R=1). This underlines the importance of modeling the attention architecture alongside other LLM architectural hyperparameters. Model size alone is not enough to characterize inference energy consumption.

Table 4  
LLM test set grouped into size-based families (1B–9B parameters), with architectural hyperparameters for each model.
<table><tr><td>Name</td><td>Size (B)</td><td>Attention</td><td> $\mathbf { n } _ { \mathsf { q } }$ </td><td>R</td><td>d</td><td>L</td></tr><tr><td>Llama 3.2</td><td>1</td><td>GQA</td><td>32</td><td> $1 / 4$ </td><td>2048</td><td>16</td></tr><tr><td>OPT</td><td>1.3</td><td>MHA</td><td>32</td><td>1</td><td>2048</td><td>24</td></tr><tr><td>Qwen 2</td><td>1.5</td><td>GQA</td><td>12</td><td> $1 / 6$ </td><td>1536</td><td>28</td></tr><tr><td>Gemma 2</td><td>2</td><td>GQA</td><td>8</td><td> $_ { 1 / 2 }$ </td><td>2304</td><td>26</td></tr><tr><td>OPT</td><td>2.7</td><td>MHA</td><td>32</td><td>1</td><td>2560</td><td>32</td></tr><tr><td>Llama 3.2</td><td>3</td><td>GQA</td><td>24</td><td> $1 / 3$ </td><td>3072</td><td>28</td></tr><tr><td>Granite</td><td>3</td><td>MHA</td><td>32</td><td>1</td><td>2560</td><td>32</td></tr><tr><td>OPT</td><td>6.7</td><td>MHA</td><td>32</td><td>1</td><td>4096</td><td>32</td></tr><tr><td>Qwen 2</td><td>7</td><td>GQA</td><td>28</td><td> $1 / 7$ </td><td>3584</td><td>28</td></tr><tr><td>Falcon-RW</td><td>7</td><td>MHA</td><td>64</td><td>1</td><td>4096</td><td>36</td></tr><tr><td>Granite</td><td>8</td><td>GQA</td><td>32</td><td> $1 / 4$ </td><td>4096</td><td>36</td></tr><tr><td>Llama 3.1</td><td>8</td><td>GQA</td><td>32</td><td> $1 / 4$ </td><td>4096</td><td>32</td></tr><tr><td>Gemma 2</td><td>9</td><td>GQA</td><td>16</td><td> $1 / 2$ </td><td>3584</td><td>42</td></tr></table>

![](images/08b622edfac7b1233f39c6ba4d6ec5bcbe7f49a849e58cf0ee0ef9920a7ce211.jpg)  
Figure 1: Average energy eficiency across the evaluated LLMs, highlighting the adopted attention architecture.

Figure 2 reports an aggregated average of the tokens per joule computed on the diferent models across the 49 diferent workload configurations in the dataset. Eficiency generally peaked at short inputs $( \tau _ { \mathrm { i n } } = 6 4 )$ and intermediate outputs $( \tau _ { \mathrm { o u t } } \in [ 1 2 8 , 2 5 6 ] )$ , and was lowest for long inputs and short outputs. Across models, the best configuration was $( 3 3 . 4 1 \pm 4 . 9 0 ) \times$ more eficient than the worst. These results showed a model-dependent energy-eficiency “sweet spot" associated with model size, attention architecture, and sequence lengths. These findings provide the basis for the extended analysis presented here.

## 6.2. Experimental Results

In this section, we evaluate the accuracy of the proposed SIMPLE model in predicting the energy per token against that of the SOTA models and the other OPs-centric and IO-centric models reported in Table 3. All models have been fitted using Weighted Least Squares (WLS), since the variance of the measured token energy increases with the number of processed tokens, indicating heteroscedasticity.

![](images/5300982c9af7a3c676850a6b7538abd9d1914744abfb72c55ddc0be6779c019f.jpg)  
Figure 2: Aggregated energy eficiency across the dataset as a function of input and output sequence length.

Table 5 reports the results of this comparison across three evaluation protocols: (i) Leave-One-Out Cross Validation (LOOCV), which evaluates generalization to previously unseen LLM architectures by withholding the corresponding architectural hyperparameters (�, $L , n _ { \mathrm { q } } , n _ { \mathrm { k v } } )$ during fitting; (ii) �-fold cross validation, which evaluates model stability and robustness using 10-fold cross validation with 5 repetitions over individual $( \boldsymbol { t } _ { \mathrm { i n } } , \boldsymbol { t } _ { \mathrm { o u t } } )$ observations randomly shufled across the complete dataset, ensuring each fold contains samples from all 13 architectures, in both the training and test sets, and (iii) full-dataset fit, used to generate the final fitting coeficients for energy estimation. In the table, columns correspond to the fitted models and rows report the accuracy results under each protocol. Figure 3 shows the values reported in Table 5, except for SweetSpot, whose scale would otherwise compress the remaining variants, at the expense of readability. We then use the full-dataset fit coeficients to compare SIMPLE against LLM-specific models on the same dataset. For each model and validation protocol, we report the coeficient of determination $( R ^ { 2 } )$ and the Mean Absolute Percentage Error (MAPE). All three protocols show similar comparative results.

Cross Validation When comparing LOOCV results, SOTA approaches of Zhan et al. [43] and Vartziotis et al. [36] achieve $R ^ { 2 }$ values of 0.456 and 0.645, respectively. The proposed LLM-agnostic SweetSpot outperforms SOTA models, reaching an $R ^ { 2 }$ of 0.780, showing that extending the previously proposed SweetSpot formulation with LLM architectural hyperparameters makes it LLM-agnostic, able to predict across LLMs without separate per-model fits, improving MAPE from 69.444 to 30.390. Despite being more accurate than SOTA models, its accuracy is far from optimal. Among OPs-centric models, Fine-Grained is the most accurate, reaching $R ^ { 2 } ~ = ~ 0 . 9 4 5$ , while FLOPs-MemOPs and Prefill-Decode obtain substantially lower $R ^ { 2 }$ of 0.551 and 0.522, respectively. This shows that considering only one coeficient per operation (FLOPs and MemOPs) or per inference phase lowers the attainable accuracy, which drops by 2.7× in MAPE compared to Fine-Grained.

Table 5  
Comparison of energy models across three evaluation protocols: LOOCV, �-fold Cross Validation (CV), and the full-dataset fit, using WLS $\left( w = 1 / y ^ { 2 } \right)$ . For cross-validation protocols, we report mean $R ^ { 2 }$ and MAPE, ± standard deviation across folds.
<table><tr><td rowspan="2"></td><td colspan="4">OPs-centric</td><td colspan="4">10-centric</td></tr><tr><td>Vartziotis et al. [36]</td><td>FLOPs MemOPs</td><td>Prefill Decode</td><td>Fine Grained</td><td> $Z \mathsf { h a n } \ \mathsf { e t } \ \mathsf { a l } .$  [43]</td><td>SweetSpot [26]</td><td>LLM-agnostic SweetSpot</td><td>SIMPLE</td></tr><tr><td># coefficients</td><td>6</td><td>3</td><td>5</td><td>25</td><td>5</td><td>6</td><td>7</td><td>4</td></tr><tr><td>LOOCV</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">R2</td><td>0.645</td><td>0.551</td><td>0.522</td><td>0.945</td><td>0.456</td><td>-0.119</td><td>0.780</td><td>0.944</td></tr><tr><td>(±0.153)</td><td>(±0.348)</td><td>(±0.385)</td><td>(±0.046)</td><td>(±0.387)</td><td>(±0.858)</td><td>(±0.175)</td><td>(±0.057)</td></tr><tr><td>37.743</td><td>40.392</td><td>37.525</td><td>14.208</td><td>44.962</td><td>69.444</td><td>30.390</td><td>13.252</td></tr><tr><td>MAPE (%)</td><td>(±14.357)</td><td>(±10.793)</td><td>(±14.091)</td><td>(±6.826)</td><td>(±13.935)</td><td>(±23.455)</td><td>(±9.393)</td><td>(±5.298)</td></tr><tr><td>k-fold CV</td><td></td><td></td><td></td><td></td><td>0.444</td><td>-0.024</td><td></td><td></td></tr><tr><td rowspan="3"> $R ^ { 2 }$  MAPE (%)</td><td>0.679</td><td>0.505</td><td>0.513</td><td>0.967</td><td></td><td></td><td>0.762</td><td>0.965</td></tr><tr><td>(±0.131)</td><td>(±0.227)</td><td>(±0.215) 34.929</td><td>(±0.013)</td><td>(±0.245) 41.707</td><td>(±0.151) 63.269</td><td>(±0.108) 27.694</td><td>(±0.015)</td></tr><tr><td>37.734 (±2.703)</td><td>39.510 (±3.245)</td><td>(±3.690)</td><td>10.895 (±1.051)</td><td>(±3.539)</td><td>(±2.607)</td><td>(±2.590)</td><td>12.038 (±0.896)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Full-dataset Fit R²</td><td>0.725</td><td>0.581</td><td>0.585</td><td>0.970</td><td>0.530</td><td>0.033</td><td>0.799</td><td>0.969</td></tr><tr><td></td><td>37.740</td><td>39.384</td><td>34.748</td><td>10.658</td><td>41.438</td><td>62.665</td><td>27.443</td><td>11.974</td></tr><tr><td>MAPE (%)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/0772feab6d11881e1c42fdef6c9bf40975459353aad6a1d789160fd11f422d1b.jpg)

![](images/d7c2c2341cbe4e44c0da162b4b9410fe24d29e4fd7d3060d2390d56e1982f005.jpg)  
Figure 3: Graphical comparison of the results in Table 5. Error bars denote standard deviation across folds.

The proposed SIMPLE model, which groups fine-grained terms into an interpretable formulation consisting of the terms corresponding to Prefill, KV-cache and Next-token costs, reduces the fitting coeficients by 6.25× and achieves the highest accuracy, with an $R ^ { 2 }$ of 0.944 and a MAPE of 13.252%, compared to 0.945 and 14.208% for the Fine-Grained model, while compared to the SOTA models, SIMPLE achieves a MAPE of 2.8–3.4× lower.

The �-fold results confirm the trends observed with LOOCV, with only minor diferences due to the diferent nature of the validation task. Together, these results show that SIMPLE generalizes well to unseen LLM architectures (LOOCV) and interpolates efectively, accurately predicting unseen workload configurations of known LLMs (�-fold).

Finally, on the full dataset, SIMPLE reaches an $R ^ { 2 }$ of 0.969 and a MAPE of 11.974, confirming the small accuracy gap with Fine-Grained observed under cross validation. The resulting coeficients are used throughout the paper and are reported in Table 9 in Appendix A.

To validate our OPs-centric approach, which aims to give a physical meaning to the fitted coeficients, we also report the coeficients of the OPs-centric $F L O P s  – M e m O P s$ model. The estimated energy per FLOP $( 1 . 8 6 \times 1 0 ^ { - 1 2 } \mathrm { J } )$ and per memory operation $( 4 . 3 1 7 \times 1 0 ^ { - 1 2 } \mathrm { J } )$ are on the order of picoJoules, consistent with the measured energy per operation of Tensor Cores and HBM in NVIDIA GPUs [3].

Comparison with LLM-specific Models Table 6 compares the full-dataset fit of SIMPLE with SOTA LLMspecific models [38, 43, 11, 40, 24, 26], which are fitted separately for each LLM; for these we report the mean and standard deviation of $R ^ { 2 }$ and MAPE across LLMs in the dataset. The previously proposed SweetSpot [26] model achieves the highest accuracy among SOTA formulations, with a nearperfect in-sample fit (mean $R ^ { 2 } ~ = ~ 1 . 0 0 0$ , mean MAPE = 1.841%). This sets an upper bound on the accuracy attainable by the proposed LLM-agnostic model. The architectureaware terms of SIMPLE allow it to remain LLM-agnostic: a

Comparison of $R ^ { 2 }$ and MAPE between SIMPLE and LLM-specific baseline models, all fitted with WLS $\left( w = 1 / y ^ { 2 } \right)$ on $E _ { \mathrm { t o k } }$ SIMPLE is fitted once across all LLMs. As LLM-specific models are fitted separately for each LLM, we report their mean and standard deviation across LLMs.
<table><tr><td>Model</td><td> $E _ { \mathbf { t o t } }$  Form</td><td> $E _ { \mathbf { t o k } }$  Formulation</td><td> ${ \bf R } ^ { 2 }$ </td><td>MAPE (%)</td></tr><tr><td>Vellaisamy et al. [38] Zhan et al. [43]</td><td>Linear</td><td> $\theta _ { 0 } + \theta _ { 1 } \frac { \tau _ { \mathrm { i n } } } { \tau _ { \mathrm { o u t } } }$ </td><td>0.604 (±0.339)</td><td>38.199 (±10.729)</td></tr><tr><td>Delavande et al. [11] Wilkins et al. [40]</td><td>Bilinear</td><td> $\theta _ { 0 } + \theta _ { 1 } \frac { \tau _ { \mathrm { i n } } } { \tau _ { \mathrm { o u t } } } + \theta _ { 2 } \tau _ { \mathrm { i n } }$ </td><td>0.886 (±0.065)</td><td>20.477 (±4.890)</td></tr><tr><td>Oviedo et al. [24]</td><td>Power-law</td><td> $\theta _ { 0 } \tau _ { \mathrm { i n } } ^ { \theta _ { 1 } } \tau _ { \mathrm { o u t } } ^ { \theta _ { 2 } }$ </td><td>0.723 (±0.099)</td><td>63.234 (±24.888)</td></tr><tr><td>SweetSpot [26]</td><td>Quadratic</td><td> $\theta _ { 0 } + \theta _ { 1 } \frac { \tau _ { \mathrm { i n } } ^ { 2 } } { \tau _ { \mathrm { o u t } } } + \theta _ { 2 } \tau _ { \mathrm { i n } } + \theta _ { 3 } \frac { \tau _ { \mathrm { i n } } } { \tau _ { \mathrm { o u t } } } + \theta _ { 4 } \tau _ { \mathrm { o u t } } + \frac { \theta _ { 5 } } { \tau _ { \mathrm { o u t } } }$ </td><td>1.000 (±0.000)</td><td>1.841 (±0.692)</td></tr><tr><td>SIMPLE</td><td>Quadratic</td><td> $\theta _ { 0 } + \theta _ { 1 } d ^ { 2 } L \ \frac { \tau _ { \mathrm { i n } } } { \tau _ { \mathrm { o u t } } } + \theta _ { 2 } d L R \Big ( \tau _ { \mathrm { i n } } + \frac { \tau _ { \mathrm { o u t } } } { 2 } \Big ) + \theta _ { 3 } d L$ </td><td>0.969</td><td>11.974</td></tr></table>

RQ2: Can an LLM-agnostic model remain compact and accurate enough to be a practical alternative to LLM-specific models, while generalizing to unseen LLM architectures and workloads?

SIMPLE shows that a small set of architecture-aware terms captures inference energy without per-LLM fitting. Despite using 6.25× fewer coeficients than Fine-Grained, it matches its $R ^ { 2 }$ on unseen architectures and workloads with comparable MAPE. On the full dataset, it achieves $R ^ { 2 } = 0 . 9 6 9$ and $\mathbf { M A P E = } 1 1 . 9 7 \%$ , the best accuracy-compactness trade-of among the evaluated models. Relative to the most accurate LLM-specific model SweetSpot [26], SIMPLE trades accuracy for generalization (MAPE 11.97% vs. 1.84%), while outperforming all other LLM-specific baselines, making it a practical LLM-agnostic alternative.

single set of coeficients covers all architectures, with no permodel fitting. This generality comes at a cost in accuracy, with MAPE increasing from 1.841% to 11.974%, while $R ^ { 2 }$ decreases only slightly, from 1.000 to 0.969.

Nevertheless, SIMPLE outperforms all other LLM-specific baselines in both $R ^ { 2 }$ and MAPE. Its MAPE is 41.5% lower than that of the bilinear models [11, 40], and 3.2× and 5.3× lower than those of the linear [38, 43] and power-law [24] models, resp. The box above summarizes the answer to RQ2.

## 7. Use Cases of the SIMPLE Law

The accuracy and generalizability of SIMPLE, demonstrated in Section 6, establish its analytical formulation beyond a predictive model: its architecture-aware terms allow it to be applied as a law, from which structural insights on LLM inference energy can be derived. In this section, we apply SIMPLE as such to explain a family of notable tradeofs in LLM inference: (i) how LLM size and architecture $f ( d , L , R )$ impact $E _ { \mathrm { t o k } }$ (Section 7.1), (ii) how input/output sequence $f ( \tau _ { \mathrm { i n } } , \tau _ { \mathrm { o u t } } )$ yields an energy minimum in $E _ { \mathrm { t o k } }$ and how this minimum shapes $E _ { \mathrm { t o t } }$ , and, through it, the Intelligence per Joule attained under increasing reasoning token budgets (Section 7.2), and (iii) how $E _ { \mathrm { t o t } }$ of a single query shifts from non-agentic to agentic AI workloads (Section 7.3). We use the SIMPLE law coeficients fitted on the full dataset (coeficients � in Table 9).

To aid readability, we recall here the three contributions of the SIMPLE law to $E _ { \mathrm { t o k } }$ , color-highlighted consistently with the figures:

![](images/2fc229e5242704bf16484efcbea31c376b7fc7a7869c46ce1d88258580240160.jpg)

## 7.1. Impact of LLM Size and Architecture on Energy

Figure 4 shows the contribution of each SIMPLE law term to the predicted $E _ { \mathrm { t o k } }$ as a function of the individual key architectural hyperparameters: the hidden dimension �, the number of layers $L ,$ and the KV-to-query head ratio $R = n _ { \mathrm { k v } } / n _ { \mathrm { q } }$ . We set $\tau _ { \mathrm { i n } } = \tau _ { \mathrm { o u t } } = 5 1 2$ and use Llama 3.1 8B architectural hyperparameters as a reference.

In Figure 4 (a), the Prefill Amortization term grows quadratically with $d ,$ while the Decode Accumulation terms grow only linearly: the KV-cache term is proportional to �, and the Next-token term is afine in � due to the constant ofset $\theta _ { 0 } .$ . This quadratic dependence dominates the overall scaling, producing a rapid increase in $E _ { \mathrm { t o k } }$ for large hidden dimensions and marking � as one of the most sensitive LLM architectural design hyperparameter. Moreover, it suggests that the energy penalty of LLMs with a larger � is amplified in prefill-dominated scenarios where long contexts are paired with short outputs.

The number of layers �, shown in Figure 4 (b), multiplies the Prefill Amortization and KV-cache terms, and the $\theta _ { 3 } d L$ component of the Next-token term. Only the constant ofset $\theta _ { 0 }$ is independent of $L ,$ so $E _ { \mathrm { t o k } }$ grows proportionally with model depth. Finally, Figure 4 (c) shows the energy impact of the attention head configuration. The Decode Accumulation KV-cache is the only term that depends on �. It scales linearly with $\tau _ { \mathrm { i n } }$ and $\tau _ { \mathrm { o u t } } ,$ so a smaller � (i.e., a larger $1 / R \ = \ n _ { \mathrm { q } } / n _ { \mathrm { k v } } ,$ corresponding to more aggressive GQA grouping) reduces the KV-cache cost during decoding, making LLM architectures with higher GQA grouping more eficient on long output sequences. In the given example, $E _ { \mathrm { t o k } }$ is ≈60 mJ/tok under $R = 1 / 7$ , compared to ≈150 mJ/tok under standard MHA (� = 1), a reduction of approximately 3×. How do these dependencies impact the token energy consumption across (i) LLMs of the same family with growing size and (ii) LLMs of the same size but diferent families?

![](images/4a2f825b04bf75aa363a9d34deb4d355874e662825826b1d1a40c49ced7c045d.jpg)

![](images/4a8df7bcdedb2064293def20eb2ae971dd547e4e772c9d7c82508d51ad054ab1.jpg)

![](images/eb7e0749ef00d77b5902bf4648a80a7431192529c15eb2395a0a456b07376653.jpg)  
Prefill Amortization Decode Accum. KV-cache Decode Accum. Next-token Total $E _ { \mathrm { t o k } }$  
Figure 4: Term-wise decomposition of the predicted $E _ { \mathrm { t o k } }$ under the SIMPLE law, varying hidden dimension � (a), number of layers � (b), and KV head ratio � (c), using Llama 3.1 8B architectural hyperparameters, with $\tau _ { \mathrm { i n } } = \tau _ { \mathrm { o u t } } = 5 1 2$ . Each band represents the additive contribution of one term of the SIMPLE law.

![](images/73dd658422d25966ecc1b396924c245df40e5568fc35fa2989ad41a114980198.jpg)  
Figure 5: Term-wise decomposition of the predicted $E _ { \mathrm { t o k } }$ under the SIMPLE law, using Llama 3.2 1B, Qwen2 1.5B, OPT 1.3B, and Llama 3.1 8B, with $\tau _ { \mathrm { i n } } = \tau _ { \mathrm { o u t } } = 5 1 2$

Figure 5 shows the contribution of each SIMPLE term to the predicted $E _ { \mathrm { t o k } }$ at $\tau _ { \mathrm { i n } } = \tau _ { \mathrm { o u t } } = 5 1 2$ , using Llama 3.2 1B as the reference and comparing it against three other models: Qwen2 1.5B and OPT 1.3B, which are similar in size, and Llama 3.1 8B, which belongs to the same family but is larger. Qwen2 1.5B shows an $E _ { \mathrm { t o k } }$ similar to Llama 3.2 1B (15 vs. 14 mJ/tok) but a diferent breakdown: its smaller � (1536 vs. 2048) and � (1/6 vs. 1/4), partially ofset by a larger �

## RQ3: How do LLM size and architectural hyperparameters influence inference energy consumption?

SIMPLE law shows that hidden dimension � is the most sensitive architectural hyperparameter, with prefill energy scaling quadratically with �, while layer depth � contributes linearly. Attention architecture matters strongly: more aggressive GQA KV grouping (1∕� = 7) reduces $E _ { \mathrm { t o k } }$ by up to 3× relative to standard MHA $( R \ = \ 1 )$ identifying it as an impactful architectural choice for reducing inference energy. Applied across LLMs of diferent sizes and families, it explains how the energy cost changes and which term accounts for it. These results confirm that architectural hyperparameters are key determinants of inference energy eficiency, capturing efects that model size alone cannot explain.

(28 vs. 16), reduce the KV-cache share from 48% to 39%; the combined efect of its smaller � and larger � keeps the Prefill Amortization nearly unchanged, while increasing the Nexttoken share from 35% to 46%. At longer sequences, these structural diferences favor Qwen2 1.5B, whose lower KVcache coeficient increasingly dominates, making it more energy-eficient than Llama 3.2 1B. A starker contrast is seen with OPT 1.3B, which has the same � as Llama 3.2 1B but a larger � (24 vs. 16) and uses MHA (� = 1) instead of GQA (� = 1∕4). As a result, the KV-cache share grows from 48% to 78%, increasing the KV-cache energy from 6 mJ/tok to 40 mJ/tok, resulting in an overall $E _ { \mathrm { t o k } }$ increase from 14 mJ/tok to 51 mJ/tok. Comparing Llama 3.2 1B with the larger Llama 3.1 8B, the Prefill Amortization share increases from 17% to 26% while the KV-cache share decreases from 48% to 38%: as both � (4096 vs. 2048) and � (32 vs. 16) double, Prefill Amortization grows 8× while the KV-cache term grows only 4×, due to the quadratic contribution of �, despite the same �.

These results show that the SIMPLE law explains the energy diferences between LLMs (Figure 1) as well as their expected scalability to longer input/output sequences through the architectural hyperparameters that drive each inference phase, which model size alone cannot capture. The box above summarizes the answer to RQ3.

![](images/88d3fe26ce8492c574cbd1e994170f089625d0e324fd84e28e124770f0e5edf6.jpg)  
(a) Qwen3-1.7B  
Architectural hyperparameters: � = 2048, � = 28, � = 8∕16.

![](images/fbe830a68566cc2049b3ff7081d005e73c868aea4b8fff7946d72a9761f3d4ae.jpg)  
(b) DeepSeek-R1-Distill-Llama-8B  
Architectural hyperparameters: � = 4096, � = 32, � = 8∕32.  
Figure 6: Estimated $E _ { \mathrm { t o k } }$ contour maps generated by the SIMPLE law of Qwen3-1.7B (a) and DeepSeek-R1-Distill-Llama-8B (b).

![](images/e283f5527bd8dc32e3db6e4bfc4239894c367a7b057394c04e2e6bb5b65e079e.jpg)  
Figure 7: $E _ { \mathrm { t o k } }$ additive components of DeepSeek-R1-Distill-Llama-8B predicted by the SIMPLE law, with $\tau _ { \mathrm { i n } } = 6 4$

## 7.2. From Energy Minimum to Reasoning Eficiency

Throughout, Qwen3-1.7B and DeepSeek-R1-Distill-Lla ma-8B to derive the $E _ { \mathrm { t o k } }$ minimum analytically and to map $E _ { \mathrm { t o k } }$ over a wide input/output token range and connect $E _ { \mathrm { t o t } }$ to Intelligence per Joule.

$E _ { t o k }$ and Input/Output Sequence Figure 7 shows the predicted $E _ { \mathrm { t o k } }$ for DeepSeek-R1-Distill-Llama-8B as a function of $\tau _ { \mathrm { o u t } } ,$ with $\tau _ { \mathrm { i n } }$ fixed at an arbitrary length of 64. The colored regions correspond to the additive components of the SIMPLE law (Prefill Amortization, and Decode Accu mulation, comprising separate KV-cache and Next-token regions), with the black curve giving the total. The Prefill Amortization energy cost decreases as $1 / \tau _ { \mathrm { o u t } } ,$ so its $E _ { \mathrm { t o k } }$ decreases hyperbolically as more output tokens are generated. In contrast, the Decode Accumulation cost of the KVcache term increases linearly with $\tau _ { \mathrm { o u t } } .$ , reflecting the accumulation of KV-cache entries with each new output token.

The remaining cost of the Next-token term is independent of the output length and contributes a constant energy. The balance between the decreasing prefill contribution and the increasing decode contribution produces a minimum in the $E _ { \mathrm { t o k } }$ curve, corresponding to the sweet spot [26] of inference energy eficiency. The location of this optimum can be obtained analytically from the SIMPLE law by diferentiating $E _ { \mathrm { t o k } }$ with respect to $\tau _ { \mathrm { o u t } }$ and setting the derivative to zero. Since � and � are strictly positive by definition, the common factor $d L > 0$ can be divided out, yielding:

$$
\frac { \partial E _ { \mathrm { t o k } } } { \partial \tau _ { \mathrm { o u t } } } = - \theta _ { 1 } d \frac { \tau _ { \mathrm { i n } } } { \tau _ { \mathrm { o u t } } ^ { 2 } } + \theta _ { 2 } \frac { R } { 2 } = 0
$$

Solving for $\tau _ { \mathrm { o u t } }$ yields the optimal output sequence length:

$$
\tau _ { \mathrm { o u t } } ^ { * } = \sqrt { 2 \frac { \theta _ { 1 } } { \theta _ { 2 } } \frac { 1 } { R } d \tau _ { \mathrm { i n } } } = \sqrt { 2 \frac { \theta _ { 1 } } { \theta _ { 2 } } \frac { n _ { \mathrm { q } } } { n _ { \mathrm { k v } } } d \tau _ { \mathrm { i n } } }\tag{22}
$$

This result gives a closed-form expression for the sweet spot, i.e., the output sequence length that minimizes the energy per generated token. In particular, the optimal output length grows proportionally to $\sqrt { \tau _ { \mathrm { i n } } d }$ , and is inversely proportional to the KV-to-query head ratio ${ \sqrt { R } } \ = \ { \sqrt { n _ { \mathrm { k v } } / n _ { \mathrm { q } } } } .$ Consequently, larger input lengths require longer generations to amortize the prefill cost, while architectures with a larger � reach the energy optimum with shorter outputs.

Figure 6 shows the $E _ { \mathrm { t o k } }$ landscape generated by the SIM-PLE law, over the input token range [16, 65536] and output token range [4, 16384], using Qwen3-1.7B and DeepSeek-R1-Distill-Llama-8B architectural hyperparameters. These spans cover the LLM workloads used in conversation and coding [32], agentic [18] and reasoning [21, 30]. Both predicted landscapes closely mirror the non-linear eficiency pattern empirically observed in the dataset of [26] (Figure 2), capturing both the prefill-amortization and the decodedominated regimes within a single surface. For a fixed $\tau _ { \mathrm { i n } } ,$ the surface exhibits the expected amortization efect, with $E _ { \mathrm { t o k } }$ reaching a minimum before rising again as the linear decode cost dominates. For the shortest input $( \tau _ { \mathrm { i n } } ~ = ~ 1 6 )$ this minimum lies at $\tau _ { \mathrm { o u t } } ~ \approx ~ 6 4$ tokens for Qwen3-1.7B and $\tau _ { \mathrm { o u t } } ~ \approx ~ 1 2 8$ for DeepSeek-R1-Distill-Llama-8B, and corresponds to the most energy-eficient point in the token space, at ≈12 and ≈30 mJ/tok, respectively. As $\tau _ { \mathrm { i n } }$ increases, the contours elongate along the output axis, reflecting the growing weight of the prefill term relative to the linear decoding contribution. The efect is strongest for short outputs, where the prefill cost is amortized over only a handful of tokens: the least energy-eficient point in the token space is located at $\tau _ { \mathrm { i n } } = 6 5 \mathrm { K }$ and $\tau _ { \mathrm { o u t } } = 4 _ { \mathrm { : } }$ , where $E _ { \mathrm { t o k } }$ reaches ≈66 J/tok for Qwen3-1.7B and ≈293 J/tok for DeepSeek-R1-Distill-Llama-8B.

Accuracy on AIME-24 dataset from [30] and SIMPLE estimation of $E _ { \mathrm { t o k } }$ and $E _ { \mathrm { t o t } }$ for Qwen3-1.7B and DeepSeek-R1- Distill-Llama-8B as a function of reasoning token budgets. Ratios (in parentheses) are relative to the minimum budget, corresponding to $\tau _ { \mathrm { o u t } } ~ = ~ 2 0 4 8$ . Intelligence per Joule (IPJ) is computed as accuracy (%) divided by $E _ { \mathrm { t o t } } \ ( \mathrm { J } )$ . For both models, the mean $\tau _ { \mathrm { i n } }$ of the dataset is used $( \tau _ { \mathrm { i n } } = 1 0 5 )$
<table><tr><td>Reasoning [tok]</td><td>Acc. [%]</td><td> $E _ { \mathrm { t o k } }$   $[ \boldsymbol { \mathsf { m } } \mathsf { J } / \boldsymbol { \mathsf { t o } } \boldsymbol { \mathsf { k } } ]$ </td><td> $E _ { \mathrm { t o t } }$  [J]</td><td>IPJ [%/kJ]</td></tr><tr><td>Qwen3-1.7B</td><td></td><td></td><td></td><td></td></tr><tr><td>2048 (1.0×)</td><td>3.0 (1.00×)</td><td>43.6 (1.00×)</td><td>89.3 (1.00×)</td><td>33.7 (1.00×)</td></tr><tr><td>4063 (2.0×)</td><td>14.0 (4.67×)</td><td>73.6 (1.69×)</td><td>298.9 (3.35×)</td><td>46.8 (1.39×)</td></tr><tr><td>6978 (3.4x)</td><td>27.0 (9.00×)</td><td>117.0 (2.68×)</td><td>816.5 (9.14×)</td><td>33.1 (0.98×)</td></tr><tr><td>9510 (4.6×)</td><td>38.0 (12.67×)</td><td>154.8 (3.55×)</td><td>1471.9 (16.48×)</td><td>25.8 (0.77×)</td></tr><tr><td>11896 (5.8×)</td><td>46.0 (15.33×)</td><td>190.4 (4.37×)</td><td>2264.5 (25.36×)</td><td>20.3 (0.60×)</td></tr><tr><td colspan="5">DeepSeek-R1-Distill-Llama-8B</td></tr><tr><td>2048 (1.0×)</td><td>2.0 (1.00×)</td><td>64.0 (1.00×)</td><td>131.1 (1.00×)</td><td>15.3 (1.00×)</td></tr><tr><td>3515 (1.7×)</td><td>15.0 (7.50×)</td><td>88.7 (1.39×)</td><td> $3 1 1 . 6 \ ( 2 . 3 8 \times )$ </td><td>48.1 (3.15×)</td></tr><tr><td>5086 (2.5×)</td><td>28.0 (14.00×)</td><td>115.3 (1.80×)</td><td> $5 8 6 . 3 \ ( 4 . 4 7 \times )$ </td><td>47.8 (3.13×)</td></tr><tr><td>7330 (3.6×)</td><td>41.0 (20.50×)</td><td>153.4 (2.40×)</td><td> $1 1 2 4 . 6 \ ( 8 . 5 8 \times )$ </td><td>36.4 (2.39×)</td></tr><tr><td>8145 (4.0×)</td><td>44.0 (22.00×)</td><td>167.3 (2.61×)</td><td> $1 3 6 2 . 6 \ ( 1 0 . 3 9 \times )$ </td><td>32.3 (2.11×)</td></tr></table>

Reasoning Cost and Intelligence per Joule Reasoning (or thinking) is widely referred to as the process of generating intermediate tokens before producing the final response, thereby leveraging additional test-time computation to improve accuracy [21, 30]. Throughout this section, we refer to the total number of generated tokens, including explicit reasoning tokens, as the reasoning token budget. Since SIMPLE already models all generated tokens as $\tau _ { \mathrm { o u t } } .$ increasing the reasoning token budget directly increases $\tau _ { \mathrm { o u t } }$ and hence $E _ { \mathrm { t o t } }$ . To assess how this energy cost translates into reasoning eficiency, we combine the $E _ { \mathrm { t o t } }$ predicted by SIM-PLE with task accuracy, measured as Intelligence per Joule $\mathrm { ( I P J = a c c u r a c y \ / } E _ { \mathrm { t o t } } \mathrm { ) }$ . We draw on results from [30], which reported the accuracy of a set of SOTA LLMs (including Qwen3-1.7B [34] and DeepSeek-R1-Distill-Llama-8B [10]) on widely used benchmarks while varying reasoning budget.

From the results of [30], we report in Table 7 the accuracy as a function of the reasoning token budget for Qwen3- 1.7B and DeepSeek-R1-Distill-Llama-8B on the AIME-24 dataset [45], for which both models report results across a comparable range of reasoning token budgets. In the same table, we also report the estimated $E _ { \mathrm { t o k } }$ and $E _ { \mathrm { t o t } }$ using the

![](images/a41ca436661933bb7617df315107390cd644d04ab1911c11ac613ab49517803d.jpg)  
Figure 8: Graphical comparison of the accuracy (a), $E _ { \mathrm { t o t } }$ (b) and Intelligence per Joule (c) results reported in Table 7.

SIMPLE law, at the corresponding reasoning token budget, together with their ratios with respect to the minimum considered reasoning token budget, $\tau _ { \mathrm { o u t } } = 2 0 4 8$ . As input length, we use AIME-24 mean prompt length $( \tau _ { \mathrm { i n } } = 1 0 5 )$

Figure 8 shows that accuracy increases consistently with the reasoning token budget $\tau _ { \mathrm { o u t } }$ for both models, as does $E _ { \mathrm { t o t } } .$ For Qwen3-1.7B, accuracy rises from 3.0% to 46.0% as $\tau _ { \mathrm { o u t } }$ grows from 2048 up to 11896 tokens, corresponding to a reasoning budget 5.8× the minimum. Over the same range, $E _ { \mathrm { t o k } }$ rises from 43.6 mJ/tok up to 190.4 mJ/tok, increasing $E _ { \mathrm { t o t } }$ by 25.4× and lowering IPJ to 0.60× of its initial value. DeepSeek-R1-Distill-Llama-8B shows a similar trend: accuracy improves from 2.0% to 44.0% as $\tau _ { \mathrm { o u t } }$ grows from 2048 to 8145 tokens, increasing $E _ { \mathrm { t o t } }$ by 10.4×. Unlike Qwen3- 1.7B, this causes an IPJ increase of 2.11×.

Despite these opposite trends at the two reasoning extremes, both models show an IPJ peak neither at the minimum nor the maximum reasoning budget, but at an intermediate level, as visible in Figure 8(c). This arises because all reported budgets lie well past the $E _ { \mathrm { t o k } }$ minimum predicted by SIMPLE $( \tau _ { \mathrm { o u t } } ^ { * } \approx 1 6 5$ tokens for Qwen3-1.7B and ≈ 331 tokens for DeepSeek-R1-Distill-Llama-8B), so $E _ { \mathrm { t o t } }$ grows super-linearly as $E _ { \mathrm { t o k } }$ and $\tau _ { \mathrm { o u t } }$ increase together, while the marginal accuracy gain per additional token decreases. Concretely, at the IPJ peak, the accuracy gain exceeds the $E _ { \mathrm { t o t } }$ increase for both models; for Qwen3-1.7B (4063 tokens), accuracy grows by 4.67× against $E _ { \mathrm { t o t } }$ by 3.35×, yielding an IPJ gain of 1.39×; for DeepSeek-R1-Distill-Llama-8B (3515 tokens), 7.5× against 2.38×, yielding an IPJ gain of 3.15×.

This application of the SIMPLE law provides a quantitative framework for identifying the reasoning budget that

RQ4: How does the reasoning token budget afect inference energy and Intelligence per Joule? The SIMPLE law analytically derives the output length $\tau _ { \mathrm { o u t } } ^ { * }$ that minimizes per-token energy (the sweet spot) showing that practical reasoning token budgets lie well beyond this minimum, driving super-linear growth in $E _ { \mathrm { t o t } }$ . Although accuracy grows with the reasoning bud-$\mathrm { g e t . }$ energy grows faster: Intelligence per Joule peaks at an intermediate budget and falls by up to 3.14× at the lowest and highest budgets across a 4.0× range.

maximizes inference energy, measured as IPJ. The box above summarizes the answer to RQ4.

## 7.3. Energy Cost in Agentic Workloads

So far, SOTA works [11, 24, 12] have focused on evaluating the environmental impact of conversational AI. Agentic AI workloads are changing this landscape, and no existing work proposes a theoretical framework that can capture and explain the resulting transformation in energy cost, as they lack a way of accounting for prompt/prefix caching. Liu et al. [18] present an extensive analysis of the GitHub Copilot [7] coding agent production workload. Unlike conversational $\mathrm { A I } ,$ characterized by queries with a median prompt length $( \tau _ { \mathrm { i n } } )$ of 750–1050 tokens and a median completion length $( \tau _ { \mathrm { o u t } } )$ of 80–105, the reported agentic workload follows a long-tail distribution with a median context length $( \tau _ { \mathrm { i n } } )$ of 68K tokens and a median completion length of 247 tokens. Moreover, they report a median prefix KV-cache of 63K tokens, which corresponds to a hit rate of ≈ 93%. This means that in the average query only 5K input tokens (out of 68K) are newly computed in the prefill stage.

We apply the SIMPLE law with the following adaptation. Let $\tau _ { \mathrm { c } } ~ \leq ~ \tau _ { \mathrm { i n } }$ be the number of input tokens served from the cache. Only the uncached sufix needs to be prefilled, so the Prefill Amortization term in the SIMPLE law ( $\theta _ { 1 } d ^ { 2 } L \tau _ { \mathrm { i n } } / \tau _ { \mathrm { o u t } } )$ scales with $\tau _ { \mathrm { i n } } - \tau _ { \mathrm { c } }$ instead of $\tau _ { \mathrm { i n } } .$

Table 8 quantifies this efect for Llama 3.1 8B on the median GitHub Copilot request, comparing three scenarios. In the non-agentic scenario, for which we consider only the non-cached input $( \tau _ { \mathrm { i n } } ~ = ~ 5 \mathrm { K } , \tau _ { \mathrm { c } } ~ = ~ 0 )$ , the request costs 138.1 J, with Prefill (88.8 J) as the dominant term and the KV-cache contribution (43.2 J) playing a secondary role. Moving to the agentic scenario $( \tau _ { \mathrm { i n } } = 6 8 \mathrm { K } , \tau _ { \mathrm { c } } = 0 )$ , both context-dependent terms grow approximately proportionally to $\tau _ { \mathrm { i n } } \mathrm { : }$ Prefill increases by 13.6× and KV-cache by 13.3×, while the Next-token term is unchanged, bringing $E _ { \mathrm { t o t } }$ to 1787.7 J (13.0×). Prefix caching $( \tau _ { \mathrm { i n } } = 6 8 \mathrm { K } , \tau _ { \mathrm { c } } = 6 3 \mathrm { K } )$ is highly efective on the Prefill term in this setting: only the 5K uncached tokens are processed, reducing Prefill contribution back to 88.8 J (−93%) and reducing $E _ { \mathrm { t o t } }$ by 2.7× (668.7 J). However, the agentic cached request still consumes 4.8× more energy than the non-agentic one. This is because the KV-cache term is unafected by prefix caching: the full 68K context must still be read and attended to at every decoding step, accounting for 86% of the remaining energy.

SIMPLE law estimation of the per-request energy, decomposed by contributions, for Llama-3.1-8B on the median GitHub Copilot coding-agent request from [18] $\left( \tau _ { \mathrm { o u t } } \ = \ 2 4 7 \right)$ . In parentheses, the ratio w.r.t. the non-agentic case. $\Delta E _ { \mathrm { t o t } }$ is the energy cost of one additional token $\left( \tau _ { \mathrm { o u t } } = 2 4 8 \right)$
<table><tr><td>Term [j]</td><td>Non-Agentic  $\tau _ { \mathrm { i n } } { = } 5 \mathsf { K } , \tau _ { \mathrm { c } } { = } 0 ,$ </td><td> $\scriptstyle { \tau _ { \mathrm { i n } } = 6 8 \mathsf { K } , \tau _ { \mathrm { c } } = 0 }$ </td><td>Agentic Agentic Cached  $\tau _ { \mathrm { i n } } { = } 6 8 \mathsf { K } , \tau _ { \mathrm { c } } { = } 6 3 \mathsf { K }$ </td></tr><tr><td>Prefill</td><td>88.8 (1.0×)</td><td>1207.8 (13.6×)</td><td>88.8 (1.0×)</td></tr><tr><td>KV-cache</td><td>43.2 (1.0×)</td><td>573.8 (13.3×)</td><td>573.8 (13.3×)</td></tr><tr><td>Next-token</td><td>6.1 (1.0×)</td><td>6.1 (1.0×)</td><td>6.1 (1.0×)</td></tr><tr><td> $E _ { \mathrm { t o k } } ~ [ \mathrm { J / t o k } ]$ </td><td>0.56 (1.0×)</td><td>7.24 (12.9×)</td><td>2.71 (4.8×)</td></tr><tr><td>Total  $E _ { \mathrm { t o t } }$ </td><td>138.1 (1.0×)</td><td>1787.7 (13.0×)</td><td>668.7 (4.8×)</td></tr><tr><td> $\Delta E _ { \mathrm { t o t } } ~ ( + 1 ~ \mathrm { t o k } )$ </td><td>0.20 (1.0×)</td><td>2.35 (11.8×)</td><td>2.35 (11.8×)</td></tr></table>

## RQ5: How does the energy cost of a single query shift from non-agentic to agentic workloads?

Moving from a non-agentic $( \tau _ { \mathrm { i n } } = 5 \mathrm { K } )$ to an agentic $( \tau _ { \mathrm { i n } } ~ = ~ 6 8 \mathrm { K } )$ request raises the energy per request by 13.0×, as both the prefill and KV-cache terms grow with $\tau _ { \mathrm { i n } }$ . Prefix caching removes most of the prefill cost (−93%, 2.7× lower $E _ { \mathrm { t o t } } )$ , but the KV-cache term, which accounts for 86% of the remaining energy, is unafected: a cached agentic request still costs 4.8× more than a non-agentic one. Likewise, each additional generated token costs 11.8× more, with or without caching.

The last row reports the energy cost of generating one additional token $( \tau _ { \mathrm { o u t } } = 2 4 8 )$ . For the non-agentic request, it is 0.20 J. In the agentic scenario, it rises to 2.35 J (11.8×), driven by KV-cache and Next-token energy growth, while the Prefill energy is unchanged. Since caching only afects Prefill, the cost of the additional generated token is identical in the agentic scenario with or without prefix caching. The box above summarizes the answer to RQ5.

## 8. Conclusions

This paper presented the SIMPLE law, an architectureaware closed-form expression for predicting the energy cost of dense decoder-only LLM inference, derived from the computational and memory-access complexity of the prefill and decode phases. SIMPLE expresses energy consumption with a compact and interpretable formulation, which explains prefill and decode energy costs as a function of LLM architectural hyperparameters and input/output sequence lengths. By explicitly modeling the attention mechanism, SIMPLE closes a gap in the literature, as the first analytical energy law able to generalize across LLM architectures. With only four parameters fitted once across all LLMs, it reduces the error of previous SOTA models by 3.1– 3.5×, reaching a MAPE of 11.97%.

Beyond predictive accuracy, the SIMPLE law quantitatively explains three of the most prominent efects in modern LLM inference: how LLM size and architecture afect energy cost, how the reasoning token budget drives super-linear energy growth and shapes the Intelligence per Joule, and how agentic AI transforms energy consumption. Taken together, these results establish the SIMPLE law as an analytical foundation for sustainable LLM inference.

Fitted coeficients of the SIMPLE Law $\left( R ^ { 2 } = 0 . 9 6 9 \right.$ , MAPE $= 1 1 . 9 7 4 \%$ and of the OPs-centric FLOPs-MemOPs model $( R ^ { 2 } = 0 . 5 8 \dot { 1 }$ , MAPE = 39.384%).
<table><tr><td>Coeff.</td><td>Estimate</td><td>Std. Err.</td><td>t</td><td> $P { > } | t |$ </td></tr><tr><td colspan="5">SIMPLE</td></tr><tr><td> $\theta _ { 0 }$ </td><td> $- 1 . 8 6 6 \times 1 0 ^ { - 3 }$ </td><td>0.000</td><td>-4.412</td><td> $< 0 . 0 0 1$ </td></tr><tr><td> $\theta _ { 1 }$ </td><td> $3 . 3 0 8 \times 1 0 ^ { - 1 1 }$ </td><td> $6 . 8 0 \times 1 0 ^ { - 1 3 }$ </td><td>48.631</td><td> $< 0 . 0 0 1$ </td></tr><tr><td> $\theta _ { 2 }$ </td><td> $1 . 0 4 1 \times 1 0 ^ { - 9 }$ </td><td> $1 . 2 4 \times 1 0 ^ { - 1 1 }$ </td><td>84.004</td><td> $< 0 . 0 0 1$ </td></tr><tr><td> $\theta _ { 3 }$ </td><td> $2 . 0 2 0 \times 1 0 ^ { - 7 }$ </td><td> $8 . 0 1 \times 1 0 ^ { - 9 }$ </td><td>25.208</td><td>&lt; 0.001</td></tr><tr><td colspan="5">FLOPs-MemOPs</td></tr><tr><td> $\theta _ { 0 }$ </td><td> $7 . 0 5 7 \times 1 0 ^ { - 3 }$ </td><td> $9 . 3 3 \times 1 0 ^ { - 4 }$ </td><td>7.566</td><td>&lt; 0.001</td></tr><tr><td> $\theta _ { 1 }$ </td><td> $1 . 8 5 6 \times 1 0 ^ { - 1 2 }$ </td><td> $9 . 1 9 \times 1 0 ^ { - 1 4 }$ </td><td>20.196</td><td> $< 0 . 0 0 1$ </td></tr><tr><td> $\theta _ { 2 }$ </td><td> $4 . 3 1 7 \times 1 0 ^ { - 1 2 }$ </td><td> $5 . 9 8 \times 1 0 ^ { - 1 3 }$ </td><td>7.221</td><td> $< 0 . 0 0 1$ </td></tr></table>

Our approach is subject to a few limitations, which we group into three categories. First, our evaluation is restricted to dense, single-GPU LLMs and does not cover mixture-ofexperts models. We also restrict our evaluation to MHA and GQA attention mechanisms; validation on more recent attention variants, LLM sizes, and architectures is left for future work. Second, the SIMPLE law does not explicitly model batch size or the number of concurrent requests as hyperparameters of the formulation. We plan to explicitly incorporate such additional workload parameters to extend the law. Finally, the reference dataset was collected using TensorRT-LLM on a single H100 GPU. Therefore, the SIMPLE law captures the energy behavior of single-device inference, and does not account for the additional costs of multi-GPU deployments, such as inter-device communication, tensor or pipeline parallelism, and distributed KV-cache management. As such, the fitted coeficients are specific to this hardwareframework pair. How the coeficients would change using other accelerators, multi-GPU deployments, and other inference frameworks remains to be explored in future work.

## Acknowledgements

The research has been partially funded by EuroHPC JU SEANERGYS (g.a. 101177590), DARE (g.a. 101143421) and Spoke “FutureHPC & BigData” of the ICSC.

## A. Coeficients and Predictions

Table 9 reports the values of SIMPLE and the OPscentric FLOPs-MemOPs models coeficients �, fitted over the full dataset, together with their standard errors, �-statistics, and �-values, obtained using WLS $( w = 1 / y ^ { 2 } )$

## References

[1] Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebron, F., Sanghai, S., 2023. GQA: Training generalized multi-query transformer models from multi-head checkpoints, in: Bouamor, H., Pino, J., Bali, K. (Eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Association for Computational Linguistics, Singapore. pp. 4895–4901. URL: https://aclanthology.org/2023.emnlp-main.298/, doi:10.18653/ v1/2023.emnlp-main.298.

[2] Almazrouei, E., Alobeidli, H., Alshamsi, A., Cappelli, A., Cojocaru, R., Debbah, M., Étienne Gofinet, Hesslow, D., et al., 2023. The falcon series of open language models. URL: https://arxiv.org/abs/2311. 16867, arXiv:2311.16867.

[3] Antepara, O., Zhao, Z., Austin, B., Ding, N., Oliker, L., Wright, N.J., Williams, S., 2025. Benchmark-driven models for energy analysis and attribution of gpu-accelerated supercomputing, in: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, Association for Computing Machinery, New York, NY, USA. p. 888–904. URL: https://doi.org/ 10.1145/3712285.3759815, doi:10.1145/3712285.3759815.

[4] Chen, L., Zhang, C., He, Y., Stoica, I., Zaharia, M., Zou, J., 2026. The price reversal phenomenon: When cheaper reasoning models cost more. URL: https://arxiv.org/abs/2603.23971, arXiv:2603.23971.

[5] Chen, Y., Wu, Y., Song, C., Thai, Z.L., Shen, X., Han, X., Liu, Z., Sun, M., 2025. Cost-optimal grouped-query attention for long-context modeling, in: Christodoulopoulos, C., Chakraborty, T., Rose, C., Peng, V. (Eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, Association for Computational Linguistics, Suzhou, China. pp. 5360–5376. URL: https://aclanthology.org/2025.emnlp-main.272/, doi:10.18653/ v1/2025.emnlp-main.272.

[6] Chung, J.W., Ma, J.J., Wu, R., Liu, J., Kweon, O.J., Xia, Y., Wu, Z., Chowdhury, M., 2025. The ml.energy benchmark: Toward automated inference energy measurement and optimization, in: Belgrave, D., Zhang, C., Lin, H., Pascanu, R., Koniusz, P., Ghassemi, M., Chen, N. (Eds.), Advances in Neural Information Processing Systems, Curran Associates, Inc. URL: https://proceedings.neurips.cc/paper\_files/ paper/2025/file/9dc510e3d7b0b3b2a58ffed7a3ad6b0f-Paper-Datasets\_ and\_Benchmarks\_Track.pdf, doi:10.52202/085713-3652.

[7] Copilot, G., 2025. Github copilot. URL: https://github.com/ features/copilot.

[8] Corporation, N., 2025. Python bindings for nvidia management library (pynvml). https://pypi.org/project/nvidia-ml-py3/. Accessed: 2025-09-12.

[9] DeepSeek-AI, 2024. Deepseek-v2: A strong, economical, and eficient mixture-of-experts language model. URL: https://arxiv.org/ abs/2405.04434, arXiv:2405.04434.

[10] DeepSeek-AI, 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. URL: https://arxiv.org/abs/ 2501.12948, arXiv:2501.12948.

[11] Delavande, J., Pierrard, R., Luccioni, S., 2026. Small talk, big impact: The energy cost of thanking ai. URL: https://arxiv.org/abs/2601. 22357, arXiv:2601.22357.

[12] Elsworth, C., Huang, K., Patterson, D., Schneider, I., Sedivy, R., Goodman, S., Townsend, B., Ranganathan, P., Dean, J., Vahdat, A., et al., 2025. Measuring the environmental impact of delivering ai at google scale. arXiv preprint arXiv:2508.15734 .

[13] Fernandez, J., Na, C., Tiwari, V., Bisk, Y., Luccioni, S., Strubell, E., 2025. Energy considerations of large language model inference and eficiency optimizations, in: Che, W., Nabende, J., Shutova, E., Pilehvar, M.T. (Eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Association for Computational Linguistics, Vienna, Austria. pp. 32556–32569. URL: https://aclanthology.org/2025.acl-long.1563/, doi:10.18653/v1/2025.acl-long.1563.

[14] Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., et al., 2024. The llama 3 herd of models. URL: https://arxiv.org/abs/2407.21783, arXiv:2407.21783.

[15] Jegham, N., Abdelatti, M., Elmoubarki, L., Hendawi, A., 2025. How hungry is ai? benchmarking energy, water, and carbon footprint of llm inference. URL: https://arxiv.org/abs/2505.09598, arXiv:2505.09598.

[16] Kaplan, J., McCandlish, S., Henighan, T., Brown, T.B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., Amodei, D., 2020. Scaling laws for neural language models. URL: https://arxiv.org/abs/2001. 08361, arXiv:2001.08361.

[17] Liakopoulos, D., Sinha, P., Hu, T., Lee, M., Yadwadkar, N.J., 2025. Maveriq: Fingerprint-guided extrapolation and fragmentation-aware layering for intent-based llm serving, in: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, Association for Computing Machinery, New York, NY, USA. p. 1676–1696. URL: https://doi.org/10.1145/ 3712285.3759867, doi:10.1145/3712285.3759867.

[18] Liu, B., Qiu, H., Íñigo Goiri, Fonseca, R., Bianchini, R., Choukse, E., 2026. Agentic coding in the wild: Characterizing github copilot traces at production scale. URL: https://arxiv.org/abs/2608.00101, arXiv:2608.00101.

[19] Maliakel, P.J., Ilager, S., Brandic, I., 2026. Characterizing llm inference energy-performance tradeofs across workloads and gpu scaling, in: 2026 IEEE 26th International Symposium on Cluster, Cloud and Internet Computing (CCGrid), pp. 33–43. doi:10.1109/ CCGrid68966.2026.00013.

[20] Mishra, M., Stallone, M., Zhang, G., Shen, Y., Prasad, A., Soria, A.M., Merler, M., Selvam, P., et al., 2024. Granite code models: A family of open foundation models for code intelligence. URL: https://arxiv.org/abs/2405.04324, arXiv:2405.04324.

[21] Muennighof, N., Yang, Z., Shi, W., Li, X.L., Fei-Fei, L., Hajishirzi, H., Zettlemoyer, L., Liang, P., Candès, E., Hashimoto, T., 2025. s1: Simple test-time scaling, in: Christodoulopoulos, C., Chakraborty, T., Rose, C., Peng, V. (Eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, Association for Computational Linguistics, Suzhou, China. pp. 20275–20321. URL: https://aclanthology.org/2025.emnlp-main. 1025/, doi:10.18653/v1/2025.emnlp-main.1025.

[22] Niu, C., Sencer, B., Zhang, W., Chen, Y., 2026. The token-energy law of llm decode: Active bytes and realized time account for joules per token. URL: https://arxiv.org/abs/2610.28576, arXiv:2610.28576.

[23] NVIDIA, 2023. TensorRT-LLM. https://github.com/NVIDIA/ TensorRT-LLM. Version X.Y.Z, accessed October 2026.

[24] Oviedo, F., Kazhamiaka, F., Choukse, E., Kim, A., Luers, A., Nakagawa, M., Bianchini, R., Lavista Ferres, J.M., 2026. Energy use of ai inference, eficiency pathways, and test-time scaling. Joule 10, 102430. URL: http://dx.doi.org/10.1016/j.joule.2026.102430, doi:10.1016/j.joule.2026.102430.

[25] Palladino, V., Palermo, G., Papka, M.E., Lan, Z., 2026. Energylens: Interpretable closed-form energy models for multimodal llm inference serving. URL: https://arxiv.org/abs/2605.10556, arXiv:2605.10556.

[26] Pizzini Cavagna, H., Proia, A., Madella, G., Esposito, G.B., Antici, F., Cesarini, D., Kiziltan, Z., Bartolini, A., 2026. Sweetspot: An analytical model for predicting energy eficiency of llm inference, in: Proceedings of the 17th ACM/SPEC International Conference on Performance Engineering, Association for Computing Machinery, New York, NY, USA. p. 83–95. URL: https://doi.org/10.1145/ 3777884.3797011, doi:10.1145/3777884.3797011.

[27] Saad-Falcon, J., Narayan, A., Akengin, H.O., Grifin, J.W., Shandilya, H., Lafuente, A.G., Goel, M., Joseph, R., Natarajan, S., Guha, E.K., Zhu, S., Athiwaratkun, B., Hennessy, J., Mirhoseini, A., Ré, C., 2026. Intelligence per watt: Measuring intelligence eficiency of local ai. URL: https://arxiv.org/abs/2511.07885, arXiv:2511.07885.

[28] Samsi, S., Zhao, D., McDonald, J., Li, B., Michaleas, A., Jones, M., Bergeron, W., Kepner, J., Tiwari, D., Gadepally, V., 2023. From words to watts: Benchmarking the energy costs of large language model inference, in: 2023 IEEE High Performance Extreme Computing Conference (HPEC), pp. 1–9. doi:10.1109/HPEC58863.2023.10363447.

[29] Shazeer, N., 2019. Fast transformer decoding: One write-head is all you need. CoRR abs/1911.02150, 0. URL: http://arxiv.org/abs/

1911.02150, arXiv:1911.02150.

[30] Shi, D., Asi, A., Li, K., Yuan, X., Pan, L., Lee, W., Xiao, W., 2026. Swireasoning: Switch-thinking in latent and explicit for paretosuperior reasoning llms. URL: https://arxiv.org/abs/2510.05069, arXiv:2510.05069.

[31] Stojkovic, J., Choukse, E., Zhang, C., Goiri, I., Torrellas, J., 2024. Towards greener llms: Bringing energy-eficiency to the forefront of llm inference, in: EMC2 at ASPLOS. URL: https://arxiv.org/abs/ 2403.20306.

[32] Stojkovic, J., Zhang, C., Goiri, I., Torrellas, J., Choukse, E., 2025. Dynamollm: Designing llm inference clusters for performance and energy eficiency, in: 2025 IEEE International Symposium on High Performance Computer Architecture (HPCA), pp. 1348–1362. doi:10. 1109/HPCA61900.2025.00102.

[33] Team, G., Mesnard, T., Hardin, C., Dadashi, R., Bhupatiraju, S., Pathak, S., Sifre, L., Rivière, M., et al., 2024. Gemma: Open models based on gemini research and technology. URL: https://arxiv.org/ abs/2403.08295, arXiv:2403.08295.

[34] Team, Q., 2025. Qwen3 technical report. URL: https://arxiv.org/ abs/2505.09388, arXiv:2505.09388.

[35] Tschand, A., Idgunji, S., Rajan, A.T.R., Kanter, D., Reddi, V.J., 5555. MLPerf Power: Addressing the AI Power Wall . IEEE Micro , 1–9URL: https://doi.ieeecomputersociety.org/10.1109/MM. 2026.3700126, doi:10.1109/MM.2026.3700126.

[36] Vartziotis, T., Kosteli, R., Vartziotis, E., Dasoulas, G., Keckeisen, M., Skianis, K., Kotsopoulos, S., Dominici, F., 2026. From tokens to watthours: Analytical energy estimation for llm inference on modern gpus. URL: https://arxiv.org/abs/2607.26571, arXiv:2607.26571.

[37] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I., 2017. Attention is all you need, in: Proceedings of the 31st International Conference on Neural Information Processing Systems, Curran Associates Inc., Red Hook, NY, USA. p. 6000–6010.

[38] Vellaisamy, P., Lam, V., Blanton, S., Shen, J.P., 2026. Characterization of request and token energy costs for llm inference workloads on gpu platforms. URL: https://arxiv.org/abs/2608.28044, arXiv:2608.28044.

[39] Wilhelm, P., Wittkopp, T., Kao, O., 2025. Beyond test-time compute strategies: Advocating energy-per-token in llm inference, in: Proceedings of the 5th Workshop on Machine Learning and Systems, Association for Computing Machinery, New York, NY, USA. p. 208–215. URL: https://doi.org/10.1145/3721146.3721953, doi:10. 1145/3721146.3721953.

[40] Wilkins, G., Keshav, S., Mortier, R., 2025. Ofline energy-optimal llm serving: Workload-based energy models for llm inference on heterogeneous systems. SIGENERGY Energy Inform. Rev. 4, 113–119. URL: https://doi.org/10.1145/3727200.3727217, doi:10. 1145/3727200.3727217.

[41] Yang, A., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C., Li, C., Li, C., et al., 2024. Qwen2 technical report. URL: https://arxiv.org/ abs/2407.10671, arXiv:2407.10671.

[42] Yuan, Z., Shang, Y., Zhou, Y., Dong, Z., Zhou, Z., Xue, C., Wu, B., Li, Z., Gu, Q., Lee, Y.J., Yan, Y., Chen, B., Sun, G., Keutzer, K., 2024. Llm inference unveiled: Survey and roofline model insights. URL: https://arxiv.org/abs/2402.16363, arXiv:2402.16363.

[43] Zhan, J., Shen, H., Guo, M., Huang, Z., He, T., 2026. Seeing is free, speaking is not: Uncovering the true energy bottleneck in edge vlm inference. URL: https://arxiv.org/abs/2607.09520, arXiv:2607.09520.

[44] Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M., et al., 2022. Opt: Open pre-trained transformer language models. URL: https://arxiv.org/abs/2205.01068, arXiv:2205.01068.

[45] Zhang, Y., Math-AI, T., 2024. American invitational mathematics examination (aime) 2024.