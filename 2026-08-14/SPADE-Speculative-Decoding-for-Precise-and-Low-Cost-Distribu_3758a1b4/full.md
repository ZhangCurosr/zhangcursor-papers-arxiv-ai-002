# SPADE: Speculative Decoding for Precise and Low-Cost Distributed Edge–Cloud Inference

Divya Jyoti Bajpai, Kishan Kumar Upadhyay and Manjesh Kumar Hanawal

MLiONS, Department of IEOR, IIT Bombay

Mumbai, Maharashtra-400076, India

{divyajyoti.bajpai, 24n0453, mhanawal}@iitb.ac.in

Abstract—Large Language Models (LLMs) have achieved remarkable success in natural language understanding and generation, but their deployment is constrained by high computational demands. Deploying smaller LLMs directly on the edge can circumvent this, but with degraded accuracy. Deploying smaller cloud-based big LLMs preserves performance, but at the cost of expensive per-token computation. We present a distributed inference framework, SPADE, that integrates speculative decoding (SD) across edge and cloud. A compact draft model deployed on the edge generates candidate tokens rapidly, and a large verifier model on the cloud validates these tokens in parallel. Accepted tokens are retained, while only rejections trigger verifier correction, substantially reducing the number of cloud queries. Our plug-and-play design shifts the bulk of computation to the edge, significantly lowers inference time and cloud cost, and preserves the accuracy of the big model without any retraining requirement. Our approach demonstrates a practical path toward scalable, cost-efficient, and accurate deployment of LLMs in real-world environments. Experimental results across multiple Natural Language Processing tasks using SpecBench and CNN/Dailymail datasets demonstrate that SPADE reduces the cloud model calls by 76% with zero loss in accuracy as compared to the full model.

Index Terms—Speculative Decoding, Distributed inference.

## I. INTRODUCTION

Large Language Models (LLMs) have achieved remarkable breakthroughs in understanding and generating language. Yet, this surge in model scale comes with a hidden cost: larger models demand vast computational resources and memory, which are rarely available on mobile or edge devices. As a result, deploying state-of-the-art models outside of highperformance cloud servers becomes a major bottleneck.

Existing literature has explored multiple strategies to alleviate this burden. Techniques such as model pruning [1], weight quantization [2], and knowledge distillation [3] reduce the size and computational load of models, making them feasible for smaller devices. However, they often come at the cost of reduced accuracy, as simplifying the network limits its ability to capture complex patterns. Even purpose-built, smaller variants of large models, while more efficient, consistently underperform compared to their full-scale counterparts.

One alternative is to run the full model on the cloud. Hosting large models on powerful cloud servers can restore top-tier performance, enabling the use of models that are too large to run on mobile or edge devices. However, cloud-based inference introduces a key challenge: the computational cost grows with the duration for which the resources are utilized. As LLMs generate output in an autoregressive manner, they require sustained computation to process each input, which can result in longer inference times and increased costs.

These constraints highlight a critical challenge: how can we achieve high accuracy without incurring large inference times on the cloud? This motivates the need for distributed inference between edge and cloud devices, where a smaller version of the LLM is deployed on the edge while the fullscale model is hosted in the cloud. Inference fully on the edge can reduce the accuracy, and inference fully on the cloud will increase the cost to a greater extent. The central question that remains is how to further reduce the reliance on the cloud by utilizing the resources available at the edge device, while still maintaining the accuracy of the full model.

Recently, Speculative Decoding (SD) has emerged as a powerful technique for accelerating LLM inference while preserving the accuracy of the full model. The approach relies on a two-model setup: a smaller, faster draft model that proposes speculative tokens and a larger, more accurate verifier model that evaluates them in parallel. Verified or accepted tokens are retained, while the first rejected token is replaced with the verifier’s predictions, after which the draft model continues generation conditioned on the corrected context. This iterative process cuts the autoregressive structure of LLMs to significantly reduce the expensive verifier calls—for instance, a 50-token sequence that would normally require 50 full model call can often be completed with only a few model calls—thus providing substantial efficiency gains without sacrificing output quality.

We propose SPADE, a distributed inference framework across edge and cloud devices, addressing the dual challenges of efficiency and accuracy in large-scale LLM deployment. In this framework, the smaller model deployed on the edge acts as a draft generator, producing a sequence of speculative (draft) tokens locally. These draft sequences are then transmitted to the larger, full-scale model hosted on the cloud, which serves as a verifier to evaluate and correct the discrepancies in a single pass. By generating the draft tokens at the edge and only verifying them in the cloud, our approach significantly reduces the computational burden on the cloud, decreases inference latency, and minimizes the number of expensive model calls on the cloud, while maintaining the output fidelity of the full-scale model without any training requirements. Our method is detailed in Fig. 1 (details in Sec. III-C).

![](images/420bfd0dcfc18acf8e101479132fe08661287630fd6ab815d0ee930f8a6dba63.jpg)  
Fig. 1. A lightweight edge model generates tokens autoregressively, which are sent to a cloud model for parallel verification and consistency checking. Accepted tokens are kept, rejected ones are replaced, and generation continues from the updated context.

The importance and usefulness of this method lie in its ability to enable scalable and resource-efficient deployment of large LLMs in real-world scenarios. It allows the edge device to handle most of the generative computation, reducing dependency on the cloud and improving the cloud computation cost. At the same time, the cloud ensures accuracy of the full model, so there is no compromise in output quality with minimal resource utilization of the cloud. This combination makes the deployment of large LLMs feasible in settings where purely cloud-based or purely edge-based solutions would be inefficient or impractical. In summary, our key contributions are as follows:

• Distributed Speculative Decoding Framework: We introduce a novel edge-cloud distributed inference setup that leverages speculative decoding to efficiently split computation between a smaller edge model and a fullscale cloud model.

• Reduce Cloud Computation: Our method significantly reduces the number of calls to the cloud model by using as the edge model to generate the fraft sequences. This lowers the inference time and resource usage without compromising accuracy.

• Maintain Full-Scale Accuracy: The framework ensures that the final outputs are equivalent to those of the full cloud model, guaranteeing fidelity while benefiting from edge-side computation. This is achieved without any additional training requirements.

• Empirical evaluations: We provide empirical results on various NLP tasks using SpecBench and CNN/DailyMail datasets, where the cloud computation time was reduced by 76% with no loss in performance.

## II. RELATED WORKS

We review prior works on distributed inference and speculative decoding related to our method.

Distributed Inference: Recent works leverage heterogeneous resources across edge and cloud devices to distribute computation. 1) Layer-splitting: Neurosurgeon [4] partitions DNNs by executing initial layers on the edge and remaining layers on the cloud. 2) Encoder-side training: Methods improve edge encoders for efficient transmission, e.g., head-network distillation [5]. 3) Early-exit classifiers: Intermediate classifiers enable fast inference by exiting early for easy samples [6]. SplitEE [7] and I-SplitEE [8] optimize splitting and prediction. 4) Complexity-aware routing: DIMEE [9] routes samples based on estimated complexity, but relies on datasetspecific heuristics.

Speculative Decoding: Speculative Decoding (SD) [10] accelerates autoregressive models using a lightweight draft model and a larger verifier for parallel validation. Self-Speculative Decoding: LayerSkip [11] reuse early layers of the large model as the draft.

Distributed inference methods face key trade-offs: (i) lenient routing reduces performance, (ii) strict routing increases cost, and (iii) Complexity estimation lacks generalization.

Our Approach: Our method differs as follows:

• We apply speculative decoding to distributed inference, with a draft model on the edge and verifier on the cloud.

• We achieve zero performance loss relative to the large model as proven in [12].

• Our framework generalizes across tasks and avoids dataset-specific heuristics.

• SPADE is fully plug-and-play and requires no retraining.

## III. PROBLEM SETUP

We begin by introducing the general framework of autoregressive decoding, speculative decoding (SD), followed by our adaptation of SD for distributed inference.

## A. Autoregressive Decoding in LLMs

LLMs typically adopt the transformer architecture, consisting of an embedding layer, L stacked transformer blocks, and a final language modeling head. Let the input prompt be denoted by $y _ { 1 : t } = ( y _ { 1 } , y _ { 2 } , . . . , y _ { t } )$ . The embedding layer maps input tokens to embeddings x<sub>0</sub>. At each layer $l \in \{ 1 , \ldots , L \}$ the hidden representation evolves as $x _ { l + 1 } ^ { t } = x _ { l } ^ { t } + f _ { l } ( x _ { l } ^ { t } )$ where $f _ { l } ( \cdot )$ denotes the transformation at layer l (self-attention and feedforward). The final representation at the L-th layer, $\boldsymbol { x } _ { L } ^ { t } ,$ is projected to logits via the language modeling head, $o _ { L } ^ { t } \ = \ g ( x _ { L } ^ { t } )$ . Autoregressive decoding estimates the conditional distribution of the next token as

$$
P ( y _ { t + 1 } \mid y _ { 1 : t } ) = \mathrm { s o f t m a x } ( o _ { L } ^ { t } ) .
$$

The next token $y _ { t + 1 }$ is then selected either via greedy decoding (argmax) or sampling for diversity.

A key limitation of this procedure is its computational cost: generating each new token (word) requires a full forward pass of the model over the entire context. Consequently, inference latency is dominated by model depth and memory requirements, leading to inefficiencies in large-scale models.

## B. Speculative Decoding (SD)

To mitigate the inefficiency of standard autoregressive decoding, SD employs a two-model pipeline that combines a smaller, faster model (draft model) with the full-scale model (verifier). The procedure consists of two stages:

1) Drafting Stage: The draft model autoregressively generates a block of d candidate tokens,

$$
y _ { t + 1 : t + d } = ( y _ { t + 1 } , y _ { t + 2 } , \ldots , y _ { t + d } ) ,
$$

conditioned on the prefix $y _ { 1 : t }$

2) Verification Stage: In a single forward pass, the verifier evaluates the block $y _ { t + 1 : t + d }$ against its probability distribution. Tokens consistent with the verifier are accepted directly. The first token $y _ { j }$ that is rejected is replaced by sampling from an adjusted distribution (explained next), and the draft model resumes generation from the updated prefix $( y _ { 1 } , \dots , y _ { j } )$ where $j$ is the index of the token where the first rejection happens.

This procedure ensures that the generated sequence is statistically identical [12] to what would have been produced by the verifier alone, while substantially reducing the number of expensive verifier calls. By verifying tokens in blocks rather than individually, SD achieves significant speedups in practice with minimal loss of performance.

## C. Speculative Decoding for Distributed Inference

We utilize speculative decoding in a distributed inference framework that leverages both edge and cloud resources.

Draft Model $\mathcal { M } _ { q }$ on Edge: A compact LLM $\mathcal { M } _ { q }$ is deployed on the edge device to perform fast inference under constrained memory and bandwidth. Model selection at the edge is governed by available resources such as GPU memory, CPU throughput, and system bandwidth. The draft model must be sufficiently lightweight to generate speculative sequences without overwhelming the device.

Verifier Model $\mathcal { M } _ { p }$ on Cloud: The cloud hosts a larger, high-accuracy model $\mathcal { M } _ { p }$ that validates the candidate tokens proposed by $\mathcal { M } _ { q }$ . The computational demands of $\mathcal { M } _ { p }$ make it unsuitable for edge deployment but ideal for cloud platforms, which can scale elastically with demand. Since cloud services often charge per model call, minimizing verifier invocations directly reduces inference cost.

Verification Criterion: Let $p ( x )$ denote the target distribution from ${ \mathcal { M } } _ { p } ,$ and $q ( x )$ the draft distribution from $\mathcal { M } _ { q }$ . A candidate $x \sim q ( x )$ is accepted with probability

$$
\alpha ( x ) = \operatorname* { m i n } \left( 1 , { \frac { p ( x ) } { q ( x ) } } \right) .
$$

If rejected, a replacement token is sampled from the adjusted distribution

$$
p ^ { \prime } ( x ) = \operatorname { n o r m } \big ( \operatorname* { m a x } ( 0 , p ( x ) - q ( x ) ) \big ) ,
$$

which corrects the bias introduced by the draft distribution.   
This guarantees equivalence to verifier-only decoding [12].   
The approach is depicted pictorially in Fig 1.

Pipeline: A pseudo-code of our method is given in Algorithm 1. The procedure is summarized as follows: starting from a user prompt, the edge model $\mathcal { M } _ { q }$ generates a block of $d$ draft tokens. These tokens are sent to the cloud, where $\mathcal { M } _ { p }$ verifies them in parallel using a single forward pass. All tokens accepted under $\alpha ( x )$ are appended to the output. The first rejected token is replaced via sampling from $p ^ { \prime } ( x )$ , and the remaining draft tokens are discarded. The context is then updated with the accepted and corrected tokens, and the draft model resumes autoregressive generation for the next block until the end-of-sentence < eos > is generated.

This design guarantees that at least one token is appended to the sequence in each iteration, thereby ensuring progress. By batching verification in the cloud while exploiting low-latency drafting on the edge, the framework balances efficiency, accuracy, and cost in distributed inference.

Algorithm 1 Speculative Decoding for Distributed Inference   
1: Input: Prompt $y _ { 1 : m : }$ , Draft model $\mathcal { M } _ { q }$ (edge), Verifier   
model $\mathcal { M } _ { p }$ (cloud)   
2: Initialize: $Y \gets [ ]$ ▷ Generated sequence   
3: while $<$ eos > token not predicted do   
4: Edge: Generate d speculative tokens   
5: for $i \in [ 1 , d ]$ do   
6: $y _ { m + i }  \mathcal { M } _ { q } ( y _ { 1 : m + i - 1 } ) \sim q ( \cdot )$   
7: end for   
8: Send $y _ { m + 1 : m + d }$ to cloud for verification   
9: Cloud: In parallel, $[ y _ { m + i } ^ { * }  \mathcal { M } _ { p } ( y _ { 1 : m + i - 1 } ) \sim p ( \cdot )$   
for $i \in [ 1 , d + 1 ] ]$ in a single forward pass.   
10: Veri $\mathrm { f y } ( y _ { m + i } , y _ { m + i } ^ { * } )$ in parallel.   
11: Accept the longest verified draft $y _ { m : m + k }$   
12: Sample $\tilde { y } _ { m + k + 1 } \gets n o r m ( m a x ( 0 , p - q )$   
13: Correct the rejected draft $y _ { m + k + 1 } = \tilde { y } _ { m + k + 1 } .$   
14: Append tokens $y _ { m : m + k + 1 }$ to Y   
15: Update context: m ← m + k + 1, y<sub>1:m+k+1</sub>   
16: end while   
17: Return: Y

Draft token length: The number of draft tokens, denoted as d, generated per edge call is a key control parameter in speculative decoding, governing the trade-off between computation and communication in the edge–cloud pipeline. A small d increases verification frequency and synchronization overhead, while a large d raises token rejection likelihood and draft computation cost, reducing overall speedup.

TABLE I  
MEAN PERFORMANCE AND EFFICIENCY METRICS ON THE SPEC-BENCH DATASET, COMPARING Target, Draft, AND Speculative MODELS ACROSS NLP TASKS. HIGHER (↑) IS BETTER PERFORMANCE, LOWER (↓) INDICATES GREATER EFFICIENCY.
<table><tr><td>Task / Metric</td><td>Target Model</td><td>Draft Model</td><td>Our Model (SPADE)</td></tr><tr><td colspan="4">Task Scores (↑)</td></tr><tr><td>Multi-turn Conversation</td><td>3.62</td><td>2.67</td><td>3.52</td></tr><tr><td>Translation</td><td>4.80</td><td>3.85</td><td>4.71</td></tr><tr><td>Summarization</td><td>4.61</td><td>4.41</td><td>4.55</td></tr><tr><td>Question Answering</td><td>4.25</td><td>3.08</td><td>4.15</td></tr><tr><td>Mathematical Reasoning</td><td>4.88</td><td>3.19</td><td>4.68</td></tr><tr><td>Retrieval-Augmented Generation</td><td>4.56</td><td>3.13</td><td>4.68</td></tr><tr><td>Overall Score (↑)</td><td>4.45</td><td>3.39</td><td>4.38</td></tr><tr><td colspan="4">Efficiency Metrics</td></tr><tr><td>Mean Target Model Calls (↓)</td><td>133.25</td><td>0.00</td><td>30.16</td></tr><tr><td>Average Throughput (tokens/s) (↑)</td><td>2.43</td><td>3.91</td><td>3.25</td></tr><tr><td>Cloud runtime (↓)</td><td>1.00×</td><td></td><td>0.23×</td></tr></table>

TABLE II

PERFORMANCE COMPARISON ON THE CNN / DAILYMAIL SUMMARIZATION DATASET. HIGHER VALUES (↑) INDICATE BETTER GENERATION QUALITY.
<table><tr><td>Metric</td><td>Target Model</td><td>Draft Model</td><td>Our Model (SPADE)</td></tr><tr><td>BLEU-1 (↑)</td><td>23.76</td><td>22.33</td><td>23.39</td></tr><tr><td>BLEU-4 (↑)</td><td>07.57</td><td>06.49</td><td>06.98</td></tr><tr><td>ROUGE-1F1 (↑)</td><td>38.38</td><td>36.05</td><td>37.99</td></tr><tr><td>ROUGE-LF1 (↑)</td><td>24.32</td><td>22.49</td><td>23.92</td></tr><tr><td>CIDEr-D (↑)</td><td>02.50</td><td>01.15</td><td>03.19</td></tr><tr><td colspan="4">Efficiency Metrics</td></tr><tr><td>Mean Target Model Calls (↓)</td><td>127.30</td><td>0.00</td><td>30.79</td></tr><tr><td>Average Throughput (tokens/s) (↑)</td><td>1.21</td><td>2.82</td><td>1.95</td></tr><tr><td>Cloud runtime (↓)</td><td>1.00×</td><td></td><td>0.24×</td></tr></table>

From a systems perspective, d balances edge computation and cloud communication: smaller values make the system communication-bound, whereas larger values risk redundant local computation. Thus, d is treated as a hyperparameter, selected empirically by monitoring acceptance rates on an initial validation subset (typically ∼ 10 samples). The objective is to maximize throughput while maintaining a stable acceptance rate under realistic latency and compute constraints.

## IV. EXPERIMENTS

Datasets: We use publicly available datasets, including CNN/DailyMail [13] for summarization, and Spec-Bench [10], a benchmark designed to evaluate speculative decoding methods. Spec-Bench consists of six subtasks—multi-turn conversation, summarization, translation, retrieval-augmented generation, question answering, and mathematical reasoning—allowing fair comparison of both speed and performance across diverse NLP settings.

Setup: Our framework follows an edge–cloud design, where a small draft model operates at the edge and a larger verification model runs on the cloud. The edge setup uses an NVIDIA RTX 3080 GPU (12 GB RAM), providing a balance between computational capability and deployability for highend edge devices. The cloud setup employs an NVIDIA RTX A6000 GPU (48 GB RAM), representing typical industrialscale infrastructure for large model inference.

Models: To instantiate this setup, we use LLaMA-3.2-1B as the draft model at the edge and LLaMA-3.1-8B [14] as the target verification model on the cloud. This configuration reflects a realistic distributed scenario in which a lightweight model performs draft generation, while the larger model validates outputs to ensure correctness during verification.

Metrics: For the CNN/DailyMail dataset, we evaluate generated summaries using standard metrics. BLEU-1 and BLEU-4 assess n-gram precision, ROUGE-1 and ROUGE-L measure unigram recall and sequence-level overlap, and CIDEr-D evaluates semantic relevance using TF-IDF–weighted n-grams. All metric scores are normalized to a 0–100 scale for consistency.

For the Spec-Bench dataset, performance is evaluated using Gemini-2.5-Flash-Lite [15] as an automatic judge, which assigns each generated response a score on a 1-5

![](images/f91f56524c273c4e994f7f06d3d61298c5540dbd849dc58fdbf60c4a23feb656.jpg)  
Fig. 2. Performance trade-off of our method, illustrating the variation in target model calls when the hyperparameter d (speculative token generation before one verifier call) increases.

Likert scale. Responses are assessed across six dimensions, correctness, instruction-following, completeness, clarity and coherence, conciseness, and factuality, with higher scores indicating better overall performance.

Baseline: We compare against two baselines: (1) the Target Model, a large cloud model representing the upper bound in output quality, and (2) the Draft Model, a lightweight edge model providing low-latency outputs. (3) Our method SPADE bridges these extremes using speculative decoding, where the target model selectively verifies draft outputs.

## V. RESULTS AND ANALYSIS

In Table I, we present results of our method on the SpecBench dataset, alongside two baselines: the Full Model (fully deployed on the cloud) and the Draft Model (operating solely at the edge). Evaluation uses an LLM-asa-judge framework across multiple qualitative dimensions, correctness, instruction-following, completeness, clarity and coherence, conciseness, and factuality, with average scores reported as the overall performance metric.

In addition to performance, we evaluate system efficiency using three indicators: average throughput (tokens per second), average number of cloud model calls, and average cloud runtime reduction. Our method achieves performance comparable to the full model while reducing cloud model calls by 77.4%, yielding significant efficiency gains with minimal quality degradation.

Table II reports results on the CNN/DailyMail dataset using standard metrics (BLEU-1, BLEU-4, ROUGE-1, ROUGE-L, CIDEr-D) to measure linguistic quality and relevance, along with the same efficiency metrics. The results show that our method closely matches the accuracy and fluency of the full model, while reducing cloud calls by 76% and substantially lowering computational cost.

Overall, these findings demonstrate that our approach maintains near cloud-level performance while significantly improving efficiency. Consistent results across SpecBench and CNN/DailyMail highlight its robustness and generalizability, supporting the use of selective, hybrid edge–cloud inference.

Analysis on the value of d: In Figure 2, we analyze the effect of the number of draft tokens (d) on target model calls for CNN/DailyMail. Increasing d consistently reduces target model invocations, as more draft tokens decrease verification frequency. This effect is amplified by strong alignment between the draft and target models, leading to higher token acceptance rates. Poor alignment, however, would reduce acceptance and increase target calls.

## VI. CONCLUSION

We propose a distributed inference framework, SPADE, that leverages speculative decoding in a dual-model setup, with a lightweight draft model at the edge and a larger verification model on the cloud. By using the verifier only for parallel verification rather than autoregressive generation, SPADE reduces costly cloud invocations. Experiments across multiple NLP tasks show that SPADE significantly lowers cloud model calls while maintaining performance close to the full model. These results establish SPADE as a practical, scalable solution for latency-aware distributed inference, effectively balancing efficiency without any loss in performance.

## REFERENCES

[1] P. Michel, O. Levy, and G. Neubig, “Are sixteen heads really better than one?” Advances in neural information processing systems, vol. 32, 2019.

[2] S. Kim, A. Gholami, Z. Yao, M. W. Mahoney, and K. Keutzer, “I-bert: Integer-only bert quantization,” in International conference on machine learning. PMLR, 2021, pp. 5506–5518.

[3] X. Jiao, Y. Yin, L. Shang, X. Jiang, X. Chen, L. Li, F. Wang, and Q. Liu, “Tinybert: Distilling bert for natural language understanding,” arXiv preprint arXiv:1909.10351, 2019.

[4] Y. Kang, J. Hauswald, C. Gao, A. Rovinski et al., “Neurosurgeon: Collaborative intelligence between the cloud and mobile edge,” in ACM Computer Architecture News, vol. 45, 2017, pp. 615–629.

[5] Y. Matsubara and M. Levorato, “Neural compression and filtering for edge-assisted real-time object detection in challenged networks,” in 2020 25th International Conference on Pattern Recognition (ICPR). IEEE, 2021, pp. 2272–2279.

[6] D. J. Bajpai and M. K. Hanawal, “CeeBERT: Cross-domain inference in early exit BERT,” in Findings of the Association for Computational Linguistics: ACL 2024, L.-W. Ku, A. Martins, and V. Srikumar, Eds. Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 1736–1748. [Online]. Available: https: //aclanthology.org/2024.findings-acl.101/

[7] D. J. Bajpai, V. K. Trivedi, S. L. Yadav, and M. K. Hanawal, “Splitee: Early exit in deep neural networks with split computing,” in Proceedings of the Third International Conference on AI-ML Systems, ser. AIMLSystems ’23. New York, NY, USA: Association for Computing Machinery, 2024. [Online]. Available: https://doi.org/10.1145/3639856.3639873

[8] D. J. Bajpai, A. Jaiswal, and M. K. Hanawal, “I-splitee: Image classification in split computing dnns with early exits,” in ICC 2024 - IEEE International Conference on Communications, 2024, pp. 2658–2663.

[9] D. J. Bajpai and M. K. Hanawal, “Distributed inference on mobile edge and cloud: An early exit based clustering approach,” in ICC 2025 - IEEE International Conference on Communications, 2025, pp. 5425–5430.

[10] H. Xia, Z. Yang, Q. Dong, P. Wang, Y. Li, T. Ge, T. Liu, W. Li, and Z. Sui, “Unlocking efficiency in large language model inference: A comprehensive survey of speculative decoding,” arXiv preprint arXiv:2401.07851, 2024.

[11] M. Elhoushi, A. Shrivastava, D. Liskovich, B. Hosmer, B. Wasti, L. Lai, A. Mahmoud, B. Acun, S. Agarwal, A. Roman et al., “Layer skip: Enabling early exit inference and self-speculative decoding,” arXiv preprint arXiv:2404.16710, 2024.

[12] Y. Leviathan, M. Kalman, and Y. Matias, “Fast inference from transformers via speculative decoding,” in International Conference on Machine Learning. PMLR, 2023, pp. 19 274–19 286.

[13] A. See, P. J. Liu, and C. D. Manning, “Get to the point: Summarization with pointer-generator networks,” arXiv preprint arXiv:1704.04368, 2017.

[14] A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Yang, A. Fan et al., “The llama 3 herd of models,” arXiv e-prints, pp. arXiv–2407, 2024.

[15] G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen et al., “Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities,” arXiv preprint arXiv:2507.06261, 2025.