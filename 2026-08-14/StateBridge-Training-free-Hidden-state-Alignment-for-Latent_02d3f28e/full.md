# StateBridge: Training-free Hidden-state Alignment for Latent Communication in LLM Multi-Agent Systems

Yanwen Peng, Delvin Ce Zhang, Xi Wang, Nikolaos Aletras

School of Computer Science, University of Sheffield

Sheffield, United Kingdom

{ypeng86,delvin.ce.zhang,xi.wang,n.aletras}@sheffield.ac.uk

## Abstract

Large language model based multi-agent systems usually communicate in text, i.e., using discrete tokens. However, text introduces a discrete bottleneck. Converting the sender’s continuous hidden states into discrete tokens discards information that token identities alone cannot capture. Recent work proposes latent communication as an alternative, where agents transmit hidden representations directly without converting them to text. However, existing latent methods either inject working memory layer by layer across the transformers, or require trained projectors that limit portability. We propose StateBridge, a training-free latent communication approach that aligns the sender’s final-layer hidden states to the receiver’s input space via a closed-form orthogonal transformation. Lightweight norm calibration and vocabulary anchoring ensure compatibility with the pretrained input distribution. The aligned states are prepended to the input of the receiver agent as a continuous prefix. We evaluate StateBridge on math reasoning, code generation, and question answering with four models from two families. StateBridge achieves the best or tied-best score on 22 out of 26 model–task pairs, consistently outperforming the strongest baseline.<sup>1</sup>

## 1 Introduction

Modern Multi-Agent (MA) systems consist of Large Language Model (LLM) agents that collaborate for tackling complex tasks requiring planning, critique, and verification (Wu et al., 2024; Li et al., 2023). Yet their performance hinges on the communication channel between agents (Guo et al., 2024; Yan et al., 2026). Natural language remains the dominant medium, but text introduces a discrete bottleneck. The sender must transform its continuous internal state into tokens before transmission. This compression discards continuous information from the latent representation that discrete tokens alone do not capture (Zou et al., 2026; Du et al., 2026), impairing inter-agent coordination (Cemri et al., 2025). The receiver maps this token sequence back to vectors through its own embedding layer, recovering only token identities rather than the sender’s original hidden states. This results in information loss and motivates a fundamental question: can off-the-shelf LLM agents bypass natural language and effectively communicate through continuous representations?

An intuitive strategy is to transmit hidden states directly. This is viable when all agents share the same pretrained LLM and thus the same hidden dimensionality (Zheng et al., 2025; Zou et al., 2026). However, matching dimensionality alone is not sufficient. Pretrained LLMs are trained to read token embeddings at the input layer, whereas decoder hidden states occupy a different region of the representation space. Passing hidden states directly therefore produces latent vectors with the right dimensionality to receivers but lacks the geometric alignment required for the model to interpret them, causing semantic mismatch during communication.

Prior work addresses this mismatch in two ways. Key-value (KV)-cache transfer methods (Fu et al., 2026; Zou et al., 2026) avoid the input-layer mismatch by injecting internal states across all transformer layers. However, they transfer internal processing states rather than a compact representation of the complete message, which may potentially cause information loss. Learned latent-communication methods (Du et al., 2026; Zheng et al., 2025) bridge the gap through trained projectors, but the projectors tie the method to the specific model and tasks they were trained on and require re-training for different models. To the best of our knowledge, it has yet to be explored whether this mismatch can be resolved through alignment alone, without training or architectural modification.

![](images/35a1cdc7e5d666aec9c9c12c319617b41ea3462811e02783f6eceb328615c243.jpg)  
Figure 1: Overview of StateBridge. Standard text communication (a) transforms the sender’s message to discrete token identities via sampling, which causes information loss, whereas StateBridge (b) extracts message hidden states from the sender, aligns them with the input embedding space, and prepends the resulting prefix to the receiver’s prompt.

We propose StateBridge, a training-free communication interface for MA systems (illustrated in Figure 1). We extract hidden states from the sender’s generated message and align them with the receiver’s input embedding space through a closed-form transformation. The aligned states are then injected as a continuous prefix for the receiver agent. Unlike text, it preserves a richer continuous representation from the sender. Compared to raw hidden-state transfer, it also makes such representation compatible with the receiver’s input space. The central idea is that latent communication helps only when the transmitted states are both informative and compatible with the receiver’s input space.

We evaluate StateBridge in a four-agent sequential pipeline on mathematical reasoning, code generation, and question answering benchmarks. Across four models spanning two families, StateBridge consistently outperforms text-based communication and KV-cache transfer methods. It achieves the best or tied-best score on 22 out of 26 model–task pairs without model updates and larger gains on challenging benchmarks. By operating only at the input embedding layer, StateBridge also applies more broadly across model families than methods that inject states across all transformer layers. A case study further confirms that the aligned prefix carries semantic information beyond the suffix tokens alone.

Contributions. (1) We show that closed-form alignment alone resolves the representation mismatch between final-layer hidden states and input embeddings, enabling training-free and effective latent communication between off-the-shelf LLM agents. (2) We propose StateBridge, combining Procrustes alignment, norm calibration, and vocabulary anchoring into a closed-form alignment interface with no learnable parameters. (3) Across four models from two families, StateBridge outperforms both existing text-based and KV-cache transfer baselines. Ablations confirm that each component contributes, with geometry preservation providing the largest effect.

## 2 Related work

LLM MA systems. LLM MA systems extend classical multi-agent coordination (Park et al., 2023; Yang et al., 2024) to modern language model settings, enabling autonomous agents to collaborate on reasoning, planning, and problem-solving (Tao et al., 2024; Wang et al., 2025; Zhao et al., 2025; Fang et al., 2025). Early methods such as AutoGen (Wu et al., 2024) and CAMEL (Li et al., 2023) coordinate multiple LLMs through explicit dialogue or role assignment. Subsequent work introduces structured communication protocols (Chen et al., 2025; Yan et al., 2026) and agent specialization (Mieczkowski et al., 2025; Fourney et al., 2024). These systems have been applied to math and science reasoning (Liang et al., 2024; Yue et al., 2024), open-domain question answering (Fourney et al., 2024; Wu et al., 2025), and GUI interaction (Zhang et al., 2025a; Ye et al., 2025). However, most existing methods communicate through text, which discards hidden state information (Du et al., 2026), introduces redundancy for linguistic coherence (Zhang et al., 2025b), and can impair inter-agent coordination (Cemri et al., 2025). On the contrary, agents in our model communicate in the latent space, mitigating these problems.

Latent communication in MA systems. Recent work explores latent-space communication as an alternative to text. Early work such as CIPHER (Pham et al., 2024) transmits weighted average of vocabulary embeddings to avoid token sampling, but still projects onto the discrete vocabulary space rather than preserving hidden states. KV-cache methods such as Cache-to-Cache (Fu et al., 2026) and LatentMAS (Zou et al., 2026), which uses a trainingfree linear alignment to enable latent reasoning, transfer working memory across agents, but transmit internal processing state rather than communication content. Embeddingbased methods such as Interlat (Du et al., 2026) and ThoughtComm (Zheng et al., 2025) transmit hidden states directly, but require trained modules, limiting their applicability to the specific model and projector they were trained on. In contrast, our StateBridge transmits generated-message hidden states and enables effective message passing through training-free alignment to the receiver’s input embedding space, offering effective content communication and wide generalizability.

## 3 Method

## 3.1 Problem statement

A homogeneous MA system is defined by all agents sharing the same pretrained LLM M (Zhang et al., 2024; Zou et al., 2026). M has vocabulary size V and embedding dimension d, with the token embedding matrix defined as ${ \mathbf { W } } _ { \mathrm { e m b } } \in \dot { \mathbb { R } } ^ { V \times d }$ . We abstract the communication channels within a MA system consisting of two primary entities: a sender $A _ { s }$ and a receiver A<sub>r</sub>. A message generated by the sender $A _ { s }$ is characterized by (1) the final-layer hidden states (S) and (2) the embeddings of the corresponding decoded tokens in the shared $\mathbf { W } _ { \mathrm { e m b } } ,$ which we term the reference embeddings R.

Sending only R reduces the message to discrete text, discarding continuous information that token identities alone do not capture (see detailed explanation in Appendix A.1). Conversely, sending raw S preserves more information, but it is not aligned with the receiver’s input embedding space, as the receiver is pretrained to consume token embeddings rather than last-layer hidden states. Our goal is to produce an aligned prefix S<sup>¯</sup> that satisfies two requirements: (1) it preserves the pairwise geometric relationships among the sender’s hidden states; and (2) it is compatible with the receiver’s input embedding space, enabling its process without modification. StateBridge satisfies these two requirements as follows.

## 3.2 Message extraction

In autoregressive models, subsequent hidden states aggregate information from the preceding context. To leverage this, we retain only the final K hidden states of the tokens of the generated message $( \mathrm { e . g . } , K = 6 4 )$ . These states are indexed from 1 to K and denoted by

$$
\begin{array} { r } { \pmb { \mathsf { S } } = \left( \pmb { \mathsf { s } } _ { 1 } , \ldots , \pmb { \mathsf { s } } _ { K } \right) ^ { \top } \in \mathbb { R } ^ { K \times d } , } \end{array}\tag{1}
$$

where $\mathbf { s } _ { i } \in \mathbb { R } ^ { d }$ is the final-layer hidden state at position i. For models like Qwen3 (Yang et al., 2025) that generate intermediate reasoning (e.g., chain-of-thought delimited by ⟨think⟩ and ⟨/think⟩ tokens before the response), we discard hidden states from this section and retain the segment intended for the receiver. Let the resulting message tokens of the receiver be $\mathbf { y } = \left( y _ { 1 } , \ldots , y _ { K } \right)$ . Looking up these tokens in the token embedding matrix gives

$$
\mathbf { R } = \mathbf { W } _ { \mathrm { e m b } } [ \mathbf { y } ] \in \mathbb { R } ^ { K \times d } .\tag{2}
$$

The reference embeddings R serve as the optimization target for aligning the continuous message. The discrete tokens y are not transmitted between agents.

## 3.3 Alignment interface

The message hidden states S and the reference embeddings R describe the same generated message but occupy different regions of the representation space. The transformation must bridge this gap (Cao et al., 2025). Since geometric proximity in hidden-state space encodes semantic similarity (Ethayarajh, 2019), the alignment transformation should therefore preserve pairwise geometric relationships rather than minimize pointwise reconstruction error. StateBridge achieves this through a three-step alignment interface illustrated in Figure 2: Procrustes alignment, norm calibration, and vocabulary anchoring.

Procrustes alignment. We treat the message hidden states S and the reference embeddings R as two point clouds in $\mathbb { R } ^ { d }$ . Both describe the same generated message, but in different coordinate systems. S carries the sender’s richer internal representation, while R lies in the space the receiver can read. We align S to R using the orthogonal Procrustes method (Schonemann¨ , 1966), which finds the rotation that best maps S onto R while preserving pairwise distances and angles.

![](images/6f195e7a46a98df58070b7fbe2d7b4272fe891295d05c6cad7edf43b952510dc.jpg)  
Figure 2: The StateBridge alignment interface transforms message hidden states S into the aligned prefix $\bar { \bf S }$ for receiver using reference embeddings R.

We first remove the global offset of each set by centering:

$$
\mathbf { S } _ { c } = \mathbf { S } - \mathbf { 1 } _ { K } { \pmb \mu } _ { S } ^ { \top } , \qquad \mathbf { R } _ { c } = \mathbf { R } - \mathbf { 1 } _ { K } { \pmb \mu } _ { R } ^ { \top } ,\tag{3}
$$

where $\mathbf { 1 } _ { K } \in \mathbb { R } ^ { K }$ is the all-ones vector, $\begin{array} { r } { \pmb { \mu } _ { S } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \mathbf { s } _ { i } } \end{array}$ is the row mean of S, and $\pmb { \mu } _ { R }$ is defined analogously for R. Centering restricts the alignment to the relative structure of the points rather than their absolute position difference in space.

We then whiten both sets:

$$
\pmb { \Sigma } _ { S } = \frac { 1 } { K } \pmb { S } _ { c } ^ { \top } \pmb { S } _ { c } + \lambda \mathbf { I } , \qquad \pmb { \Sigma } _ { R } = \frac { 1 } { K } \pmb { \mathrm { R } } _ { c } ^ { \top } \pmb { \mathrm { R } } _ { c } + \lambda \mathbf { I } ,\tag{4}
$$

$$
{ \bf S } _ { w } = { \bf S } _ { c } { \bf \Sigma } _ { S } ^ { - 1 / 2 } , \qquad { \bf R } _ { w } = { \bf R } _ { c } { \bf \Sigma } _ { R } ^ { - 1 / 2 } ,\tag{5}
$$

where $\lambda > 0$ is a small regularization constant, and I is the identity matrix. Whitening rescales the principal directions of each set so that a few high-variance directions do not dominate the alignment. Centering and whitening together remove the global offset and equalize the principal directions. The remaining mismatch between the two point clouds is then a rotation.

We then solve an orthogonal Procrustes problem:

$$
\mathbf { Q } ^ { * } = \arg \operatorname* { m i n } _ { \mathbf { Q } } \| \mathbf { S } _ { w } \mathbf { Q } - \mathbf { R } _ { w } \| _ { F } ^ { 2 } \quad \mathrm { s . t . } \quad \mathbf { Q } ^ { \top } \mathbf { Q } = \mathbf { I } ,\tag{6}
$$

where $\| \cdot \| _ { F }$ denotes the Frobenius norm. This objective seeks the rotation that best aligns the whitened sender states to the whitened reference embeddings without introducing arbitrary shearing. The solution is closed-form. Let $\mathbf { S } _ { w } ^ { \top } \mathbf { R } _ { w } = \mathbf { U } \mathbf { D } \Breve { \mathbf { V } } ^ { \top }$ be the singular value decomposition (SVD) (Golub & Van Loan, 2013) of the cross-correlation matrix. The optimal rotation is then $\mathbf { Q } ^ { * } = \mathbf { U } \mathbf { V } ^ { \top }$ . Since $\mathbf { Q } ^ { * }$ is orthogonal, it preserves pairwise distances and angles in the whitened space. We provide a formal proof in Appendix A.3.

Finally, we restore the overall scale pattern and location of the reference embeddings:

$$
\tilde { \mathbf { S } } = \mathbf { S } _ { w } \mathbf { Q } ^ { * } \pmb { \Sigma } _ { R } ^ { 1 / 2 } + \mathbf { 1 } _ { K } \pmb { \mu } _ { R } ^ { \top } .\tag{7}
$$

This Procrustes alignment ensures that the resulting matrix $\tilde { \bf S }$ remains a continuous representation, derived from the sender’s hidden states, while being closer to the input geometry required by the receiver for effective communication.

Norm calibration and vocabulary anchoring. The Procrustes alignment step matches the orientation of the message to the input embedding space, but two practical issues remain: (1) Final-layer hidden states have much larger norms than token embeddings. On Qwen3-4B (Yang et al., 2025), the average hidden-state norm is approximately 140× that of the input embeddings. This gap arises from the accumulation of residual updates across layers and the next-token prediction objective. The norm gap persists after alignment and changes how strongly the aligned vectors interact with the receiver’s attention layers. (2) Aligned vectors $\tilde { \bf S }$ could still lie far from regions occupied by actual vocabulary embeddings. We address these issues by applying a two-step adjustment to each aligned vector.

First, we calibrate each aligned vector to the typical norm of the vocabulary embeddings. Let $\begin{array} { r } { { \bar { n } } = \frac { 1 } { V } \sum _ { v = 1 } ^ { V } \| \mathbf { W } _ { \mathrm { e m b } } [ v ] \| _ { 2 } } \end{array}$ be the average $\ell _ { 2 }$ norm of the vocabulary embeddings. For each row $\tilde { \bf { s } } _ { i }$ of $\tilde { \mathbf { S } } ,$ , we compute

$$
\hat { \bf s } _ { i } = \tilde { \bf s } _ { i } \cdot \frac { \bar { n } } { \lVert \tilde { \bf s } _ { i } \rVert _ { 2 } } .\tag{8}
$$

This step places the prefix on a comparable norm scale to the ordinary input token embeddings. Without it, the larger prefix norms would dominate the dot-product attention scores, causing the receiver to attend to the prefix regardless of semantic relevance.

Second, we move each calibrated vector slightly toward its nearest vocabulary embedding under cosine similarity:

$$
v _ { i } ^ { * } = \arg \operatorname* { m a x } _ { v \in \{ 1 , \dots , V \} } \frac { \hat { \mathbf { s } } _ { i } ^ { \top } \mathbf { W } _ { \mathrm { e m b } } [ v ] } { \| \hat { \mathbf { s } } _ { i } \| _ { 2 } \| \mathbf { W } _ { \mathrm { e m b } } [ v ] \| _ { 2 } } , \qquad \bar { \mathbf { s } } _ { i } = ( 1 - \alpha ) \hat { \mathbf { s } } _ { i } + \alpha \mathbf { W } _ { \mathrm { e m b } } [ v _ { i } ^ { * } ] ,\tag{9}
$$

where $\alpha \in [ 0 , 1 ]$ . This vocabulary anchoring step does not snap the message hidden states to discrete tokens. Instead, it keeps the representation continuous while moving it closer to regions of the input embedding space encountered during pretraining. While Procrustes alignment preserves the geometric structure of the message hidden states, these two steps ensure that the prefix is also compatible with the receiver’s input embedding space. Stacking all rows $\bar { \bf { s } } _ { i }$ gives the final aligned prefix $\bar { \pmb S } \in \mathbb { R } ^ { K \times d }$

## 3.4 Prefix injection

Let $\mathbf { P } \in \mathbb { R } ^ { N \times d }$ denote the receiver’s prompt embeddings, where N is the number of prompt tokens. StateBridge prepends the aligned prefix to the prompt, forming the combined input $\pmb { \mathrm { X } } = [ \bar { \pmb { \bar { S } } } ; \mathbf { P } ] \in \mathbb { R } ^ { ( \check { K } + \hat { N } ) \times \hat { d } }$ . StateBridge injects the aligned prefix directly at the embedding layer, bypassing tokenization. We assign position indices sequentially over the concatenated input, so the prefix occupies positions 0 through $K { - } 1$ , and the prompt follows from position K onward. The receiver’s attention and position-encoding mechanisms therefore treat the prefix identically to ordinary token embeddings, requiring no special handling. The receiver processes X with the standard language modeling layers, without any architectural modification. StateBridge therefore changes only the communication interface between agents. The sender transmits an aligned continuous prefix instead of text, while the underlying LLM remains unchanged.

## 3.5 Computational cost

Time complexity. The alignment is dominated by two operations: the whitening eigendecomposition, which costs $\check { O } ( d ^ { 3 } )$ , and the vocabulary anchoring search, which costs $O ( K V d )$ Both computed once per batch. The remaining operations (centering, rotation, norm calibration) are $\bar { O } ( K d ^ { 2 } )$ or lower. For comparison, let T denote the number of generated tokens and L the number of transformer layers. A single autoregressive pass costs $O ( T L d ^ { 2 } )$ . For typical configurations $( T \geq 2 5 6 , L = 3 2 )$ , alignment is cheaper than one generation pass.

Space complexity. Naively storing every layer’s hidden states during generation costs $\overrightharpoon { O ( T L d ) }$ . We instead register a lightweight callback (forward hook) on the final transformer layer. It records only that layer’s output at each decoding step and discards the rest, reducing the extraction cost to $O ( T d )$ . Only the last K states are retained, giving $O ( K d )$ The alignment adds $O ( d ^ { 2 } )$ for covariance and SVD factors. Both are negligible relative to the model parameters. By contrast, KV-cache transfer methods store $O ( \breve { T } L \breve { d } )$ per agent (Fu et al., 2026; Zou et al., 2026), incurring substantially larger memory overhead.

## 4 Experimental setup

We follow Zou et al. (2026) and use a standard four-agent pipeline across reasoning, question answering, and code generation tasks.

Tasks and datasets. We use eight benchmarks from three task categories: mathematical reasoning (GSM8K (Cobbe et al., 2021), AIME24 (Maxwell-Jia, 2024), and AIME25 (Zhang & Math-AI, 2025)), question answering (GPQA-Diamond (Rein et al., 2024), ARC-Challenge (Clark et al., 2018), and MedQA (Jin et al., 2021)), and code generation (MBPP+ and HumanEval+ (Liu et al., 2023)). This selection covers tasks that differ in reasoning depth, knowledge demand, and output precision. Detailed dataset descriptions are provided in Appendix B.1.

Models. We test two model families: Qwen3 (4B, 8B, and 32B) (Yang et al., 2025) and OLMo3-7B-Think (Team OLMo et al., 2026). Within each run, all agents share the same model weights. We evaluate all models on five core benchmarks. Following Zou et al. (2026), we additionally report AIME24, AIME25, and GPQA-Diamond for the stronger Qwen3-8B and Qwen3-32B models.

Baselines. We compare StateBridge against three baselines: (i) Single, standard single-agent generation; (ii) TextMAS, text-based sequential communication (Zhang et al., 2024); and (iii) LatentMAS (Zou et al., 2026), which injects the sender’s KV cache into the receiver. All methods share the same four-agents (Planner, Critic, Refiner, Judger), identical hyperparameters, and the same evaluation protocol. The only difference across prompts is the communication modality description. Appendix B.2 describes the agent pipeline and Appendix C lists the full prompt templates.

Implementation details. We set $K = 6 4 , \alpha = 0 . 3$ , and $\lambda = 1 0 ^ { - 3 }$ . Following Zou et al. (2026), all agents use temperature 0.6 and top-p 0.95. Maximum output lengths range from $2 { , } 0 4 8$ to 20,000 tokens depending on the task (Appendix B.3). All experiments were run on 2 NVIDIA A100-80G GPUs. The evaluation protocol for each task category is described in Appendix B.4.

## 5 Results

Table 1 summarizes the results across MA communication methods, models, and tasks. We report accuracy for question answering and mathematical reasoning and pass@1 (Chen et al., 2021) for code generation. StateBridge achieves the best average score in all four model settings, improving over the best baseline by 2.4 to 2.9 points. It also wins or ties on 22 out of 26 model–task pairs. These gains hold across both model families.

<table><tr><td></td><td colspan="5">Qwen3-4B</td><td colspan="5">OLMo3-7B-Think</td></tr><tr><td>Task</td><td>Single</td><td>Text</td><td>Latent</td><td>SB</td><td>Δ</td><td>Single</td><td>Text</td><td>Latent</td><td>SB</td><td>Δ</td></tr><tr><td>ARC-C</td><td>89.2</td><td>90.0</td><td>92.3</td><td>93.7</td><td>↑1.4</td><td>84.6</td><td>82.2</td><td>78.3</td><td>89.6</td><td>↑5.0</td></tr><tr><td>MedQA</td><td>47.7</td><td>65.3</td><td>66.3</td><td>70.3</td><td>↑4.0</td><td>50.7</td><td>54.0</td><td>44.3</td><td>59.0</td><td>↑5.0</td></tr><tr><td>GSM8K</td><td>82.4</td><td>89.8</td><td>88.2</td><td>89.8</td><td>0.0</td><td>85.7</td><td>85.7</td><td>67.5</td><td>86.5</td><td>↑0.8</td></tr><tr><td>MBPP+</td><td>63.5</td><td>69.8</td><td>73.5</td><td>75.9</td><td>↑2.4</td><td>66.7</td><td>66.9</td><td>42.3</td><td>69.1</td><td>↑2.2</td></tr><tr><td>HumanEval+</td><td>75.0</td><td>79.7</td><td>79.9</td><td>82.3</td><td>↑2.4</td><td>71.3</td><td>80.5</td><td>43.3</td><td>79.3</td><td>↓1.2</td></tr><tr><td>Avg.</td><td>71.6</td><td>78.9</td><td>80.0</td><td>82.4</td><td>↑2.4</td><td>71.8</td><td>73.9</td><td>55.1</td><td>76.7</td><td>↑2.8</td></tr><tr><td></td><td colspan="5">Qwen3-8B</td><td colspan="5">Qwen3-32B</td></tr><tr><td>Task</td><td>Single</td><td>Text</td><td>Latent</td><td>SB</td><td>Δ</td><td>Single</td><td>Text</td><td>Latent</td><td>SB</td><td>Δ</td></tr><tr><td>ARC-C</td><td>91.0</td><td>94.6</td><td>94.4</td><td>95.1</td><td>↑0.5</td><td>95.5</td><td>93.9</td><td>95.7</td><td>94.5</td><td>↓1.2</td></tr><tr><td>MedQA</td><td>53.0</td><td>75.0</td><td>75.3</td><td>78.3</td><td>↑3.0</td><td>86.0</td><td>77.0</td><td>84.3</td><td>87.3</td><td>↑1.3</td></tr><tr><td>GSM8K</td><td>81.1</td><td>92.3</td><td>93.8</td><td>91.2</td><td>↓2.6</td><td>91.1</td><td>92.4</td><td>92.7</td><td>90.2</td><td>↓2.5</td></tr><tr><td>MBPP+</td><td>64.8</td><td>69.5</td><td>74.6</td><td>76.9</td><td>↑2.3</td><td>74.3</td><td>75.1</td><td>74.1</td><td>76.2</td><td>↑1.1</td></tr><tr><td>HumanEval+</td><td>74.4</td><td>80.5</td><td>80.5</td><td>83.6</td><td>↑3.1</td><td>78.7</td><td>84.8</td><td>84.2</td><td>85.4</td><td>↑0.6</td></tr><tr><td>AIME24</td><td>50.0</td><td>53.3</td><td>56.7</td><td>63.3</td><td>↑6.6</td><td>70.0</td><td>73.3</td><td>66.7</td><td>76.7</td><td>↑3.4</td></tr><tr><td>AIME25</td><td>46.7</td><td>53.3</td><td>53.3</td><td>53.3</td><td>0.0</td><td>66.7</td><td>70.0</td><td>56.7</td><td>73.3</td><td>↑3.3</td></tr><tr><td>GPQA</td><td>39.9</td><td>43.4</td><td>45.5</td><td>52.5</td><td>↑7.0</td><td>52.5</td><td>58.3</td><td>57.1</td><td>64.1</td><td>↑5.8</td></tr><tr><td>Avg.</td><td>62.6</td><td>70.2</td><td>71.8</td><td>74.3</td><td>↑2.5</td><td>76.9</td><td>78.1</td><td>76.4</td><td>81.0</td><td>↑2.9</td></tr></table>

Table 1: Main results across model families. Accuracy (%) on QA and math tasks, pass@1 (%) on code tasks. Bold: best or tied-best per task. ∆: StateBridge vs. the best baseline (green = gain, red = loss). Text = TextMAS, Latent = LatentMAS, SB = StateBridge (Ours).

Switching from Qwen3 to OLMo3 tests whether each latent method is sensitive to the LLM architecture. On OLMo3-7B-Think, LatentMAS averages only 55.1%, far below the 73.9% of TextMAS. StateBridge reaches 76.7%. Because the agent pipeline, prompts, and decoding settings are otherwise unchanged, this gap points to the communication channel. KV-cache transfer injects states across each transformer layers. If the layer structure differs between LLMs, compatibility breaks. StateBridge operates only at the input embedding layer and the use of such a uniform interface ensures a greater applicability across model families.

StateBridge helps most on the more challenging benchmarks, including MedQA, GPQA, AIME24/25, and the code generation tasks. StateBridge’s message hidden states encode distributional information beyond the sampled tokens, including confidence signals and traces of alternative reasoning paths. Text-based communication collapses this to a single token sequence, leading to information loss and incomplete message communication.

Not all tasks show gains. On GSM8K for Qwen3 models, StateBridge trails the best baseline, likely because continuous prefixes affect output formatting under exact-match evaluation (Li & Liang, 2021; Tam et al., 2024).

## 6 Analysis

## 6.1 Ablation study

Table 2 isolates the contribution of each StateBridge component.

Orthogonal vs. linear alignment. A natural alternative to Procrustes is to fit a linear map via ridge regression, minimizing $| \mathsf { S } \boldsymbol { \mathsf { W } } - \boldsymbol { \mathsf { R } } | ^ { 2 } + \lambda | \boldsymbol { \mathsf { W } } | ^ { 2 }$ . This map is not constrained to preserve distances or angles, so it achieves low pointwise reconstruction error. However, it distorts the pairwise geometry among the sender states. Replacing Procrustes with ridge regression results to average performance drops from 82.4% to 74.9%, with especially larger drops on MBPP+ and HumanEval+. Since pairwise geometry encodes semantic similarity, distorting it discards the richer information that latent communication carries over text. The effect is strongest on code generation, which demands structurally precise output. Orthogonal alignment preserves these semantic similarity and maintains the precision of the receiver’s generation. Appendices A.2 and A.3 provide a geometric explanation for this gap.

<table><tr><td>Variant</td><td>ARC-C</td><td>MedQA</td><td>GSM8K</td><td>MBPP+</td><td>HE+</td><td> $\operatorname { A v g } .$ </td><td>Δ</td></tr><tr><td>Full StateBridge</td><td>93.7</td><td>70.3</td><td>89.8</td><td>75.9</td><td>82.3</td><td>82.4</td><td>一</td></tr><tr><td>Ridge Regr.</td><td>93.0</td><td>68.0</td><td>88.6</td><td>61.5</td><td>63.6</td><td>74.9</td><td>↓7.5</td></tr><tr><td>Random Noise</td><td>32.2</td><td>30.7</td><td>84.5</td><td>48.2</td><td>48.2</td><td>48.8</td><td>↓33.6</td></tr><tr><td>w/o Norm Calib.</td><td>90.9</td><td>68.0</td><td>87.1</td><td>71.7</td><td>79.9</td><td>79.5</td><td>↓2.9</td></tr><tr><td>w/o Vocab. Anchor.</td><td>91.7</td><td>66.7</td><td>88.9</td><td>73.0</td><td>80.5</td><td>80.2</td><td>↓2.2</td></tr></table>

Table 2: Ablation on Qwen3-4B. Each row removes or replaces one component. HE+: HumanEval+.
<table><tr><td></td><td colspan="4">Prefix length K</td><td colspan="6">Anchoring coefficient α</td></tr><tr><td>Task</td><td>16</td><td>32</td><td>64</td><td>128</td><td>0.0</td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.6</td></tr><tr><td>ARC-C</td><td>93.2</td><td>93.3</td><td>93.7</td><td>91.7</td><td>91.7</td><td>93.5</td><td>93.2</td><td>93.7</td><td>92.5</td><td>93.3</td></tr><tr><td>MedQA</td><td>68.0</td><td>68.7</td><td>70.3</td><td>68.7</td><td>66.7</td><td>64.0</td><td>71.3</td><td>70.3</td><td>69.7</td><td>66.3</td></tr><tr><td>GSM8K</td><td>87.6</td><td>87.9</td><td>89.8</td><td>88.6</td><td>88.9</td><td>89.1</td><td>88.7</td><td>89.8</td><td>88.0</td><td>89.0</td></tr><tr><td>MBPP+</td><td>74.6</td><td>75.9</td><td>75.9</td><td>74.9</td><td>73.0</td><td>73.3</td><td>69.3</td><td>75.9</td><td>72.2</td><td>72.0</td></tr><tr><td>HumanEval+</td><td>82.3</td><td>81.7</td><td>82.3</td><td>79.3</td><td>80.5</td><td>78.1</td><td>81.7</td><td>82.3</td><td>83.5</td><td>82.9</td></tr></table>

Table 3: Sensitivity to prefix length K and anchoring coefficient α on Qwen3-4B. Moderate values work best. Default values (K=64, α=0.3) are shaded.

Compatibility steps. The two compatibility steps also matter. Removing norm calibration lowers the average to 79.5%, and removing vocabulary anchoring lowers it to 80.2%. This suggests that alignment alone is not enough. The prefix also needs the right norm scale and must stay close to embedding regions seen during pretraining. Replacing the aligned prefix with random noise drops the average to 48.8%. This rules out the possibility that gains come from simply prepending extra continuous vectors.

## 6.2 Hyperparameter sensitivity

For all experiments, we use the best hyperparameters (K=64, α=0.3) on MedQA obtained with Qwen3-4B and apply them without modification to all other datasets and models. Table 3 reports the sensitivity to prefix length K and anchoring coefficient α. Increasing K from 16 to 64 usually improves performance, suggesting that very short messages loss useful information. However, K=128 hurts on most tasks. We attribute this to the alignment procedure. StateBridge fits a single global rotation to all K states. When K is small, retained states come from the message end and tend to have similar norms and geometry. Larger K includes earlier states that may differ in both. The global rotation then has to compromise across a more heterogeneous point set, potentially lowering alignment quality.

The anchoring coefficient α controls a trade-off between informativeness and compatibility. A smaller α preserves more of the aligned hidden state but keeps the prefix further from the receiver’s input embedding space. A larger α moves the prefix closer to actual reference embeddings but discards more continuous information. α=0.3 strikes a consistent balance across all benchmarks, so we adopt it as a fixed default.

## 6.3 Alignment visualization

Figure 3 shows the gap between message hidden states and reference embeddings, and how StateBridge closes it. States are collected from 300 MedQA queries. Before alignment, the message hidden states occupy a different region from the reference embeddings. After alignment, the prefix moves much closer to the input embedding space. The same pattern appears for both Qwen3-4B and Qwen3-8B, suggesting that the effect is stable across model scales.

## 6.4 Case study: plan recovery from aligned prefixes

Table 4 shows a controlled reconstruction example. We ask the critic to restate the planner’s original plan using only the aligned prefix received through StateBridge, without access to the original text. We then vary the prefix length K. At K=16, the visible suffix tokens form only a short fragment. Yet the critic recovers the diagnosis, the key clinical features, and the exclusion of alternative diagnoses. The restatement even includes koilonychia, a more precise medical term for “flat nails.” This term does not appear in the visible suffix tokens. At K=64, the recovered plan becomes more detailed, and the critic additionally identifies differential diagnoses. At K=128, the critic copies only a truncated, incomplete plan. The aligned prefix no longer carries continuous information beyond the suffix tokens, consistent with the reduced alignment quality discussed in Section 6.2.

![](images/027ff96b8f381f7029d9d88f5822162cb0e6b579942e85f7c4afd9fa3a57b0dd.jpg)  
Figure 3: Principal component analysis (PCA) density (left) and PCA scatter of the embedding space (right) for Qwen3-4B and Qwen3-8B. Orange: message hidden states $\mathbf { s } _ { i } \mathbf { ; }$ blue: reference embeddings r<sub>i</sub>; green: aligned prefix s¯<sub>i</sub>.

<table><tr><td>Original Plan (185 tokens)</td><td colspan="3">1. Analyze clinical features (fatigue, dysphagia, weight loss, esophageal web, flat nails). 2. Evaluate lab results (low hemoglobin, normal WBC/ESR). 3. Correlate: Plummer-Vinson syndrome linked to anemia, web, dysphagia; other diagnoses less likely. 4. Confirm diagnosis. ..</td></tr><tr><td></td><td>K=16</td><td>K=64</td><td>K=128</td></tr><tr><td>Suffix Tokens</td><td>...al web, and systemic symptoms aligns most strongly with Plummer- Vinson.</td><td>.. . Plummer-Vinson is linked to iron de- ficiency, esophageal webs, and oral can- didiasis, aligning with the patient&#x27;s clin- ical picture.</td><td>... Low hemoglobin with normal WBC and ESR supports chronic anemia. ... Iron deficiency in Plummer-Vinson is due to strictures or blood loss, aligning with the patient&#x27;s symptoms.</td></tr><tr><td>Critic Recovery</td><td>... the most likely diagnosis based on the clinical features (anemia, dysphagia, koilonychia, upper esophageal web) ... The differential diagnoses (e.g., esophageal cancer, achalasia) are less consistent with the presented findings.</td><td>The plan ... noting its association with iron deficiency, esophageal webs, and oral candidiasis, align- ing with (fatigue, weight loss, flat nails, and esophageal web). It also differentiates this from esophageal cancer or achalasia . ..</td><td>... features like the web or specific malignancy indicators. Confirm as- sociations: Iron deficiency anemia in Plummer-Vinson is often due to stric- tures or blood loss, aligning with the patient&#x27;s symptoms.</td></tr></table>

Table 4: Case study on a MedQA example (Qwen3-4B). Suffix Tokens shows the last K decoded tokens; Critic Recovery shows the critic’s restatement from the aligned prefix alone. Bold highlights information absent from the visible suffix.

## 7 Conclusion

We introduced StateBridge, a training-free communication interface that aligns sender hidden states to the receiver’s input embedding space. It requires no training and no architectural modification. Across four model settings and eight benchmarks, StateBridge consistently outperforms both text-based and KV-cache baselines. The gains are largest on harder tasks, where a text handoff is most likely to discard useful information. Together with the ablations and case study, these results point to a key insight. The challenge in latent communication is not only sending richer states. It is making those states readable to the next agent. More broadly, the results show that communication interface design matters in homogeneous MA systems. Our current setting assumes a shared model and a fixed prefix length. Future work could extend the method to heterogeneous models, choose the prefix length adaptively, and combine training-free alignment with learned communication modules.

## References

Mingzi Cao, Xi Wang, and Nikolaos Aletras. Progressive depth up-scaling via optimal transport. arXiv preprint arXiv:2508.08011, 2025.

Mert Cemri, Melissa Z Pan, Shuyi Yang, Lakshya A Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, et al. Why do multi-agent llm systems fail? In Advances in Neural Information Processing Systems (NeurIPS 2025, Datasets and Benchmarks Track), 2025.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Weize Chen, Jiarui Yuan, Chen Qian, Cheng Yang, Zhiyuan Liu, and Maosong Sun. Optima: Optimizing effectiveness and efficiency for llm-based multi-agent system. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 11534–11557, 2025.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Zhuoyun Du, Runze Wang, Huiyu Bai, Zouying Cao, Xiaoyong Zhu, Yu Cheng, Bo Zheng, Wei Chen, and Haochao Ying. Enabling agents to communicate entirely in latent space. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 27106–27129, 2026.

Kawin Ethayarajh. How contextual are contextualized word representations? comparing the geometry of BERT, ELMo, and GPT-2 embeddings. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 55–65. Association for Computational Linguistics, 2019. doi: 10.18653/v1/D19-1006.

Jinyuan Fang, Yanwen Peng, Xi Zhang, Yingxu Wang, Xinhao Yi, Guibin Zhang, Yi Xu, Bin Wu, Siwei Liu, Zihao Li, Zhaochun Ren, Nikos Aletras, Xi Wang, Han Zhou, and Zaiqiao Meng. A comprehensive survey of self-evolving ai agents: A new paradigm bridging foundation models and lifelong agentic systems. arXiv preprint arXiv:2508.07407, 2025.

Adam Fourney, Gagan Bansal, Hussein Mozannar, et al. Magentic-one: A generalist multiagent system for solving complex tasks. arXiv preprint arXiv:2411.04468, 2024.

Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang, and Yu Wang. Cache-to-cache: Direct semantic communication between large language models. In The Fourteenth International Conference on Learning Representations, 2026.

Gene H. Golub and Charles F. Van Loan. Matrix Computations. Johns Hopkins University Press, 4th edition, 2013.

Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V. Chawla, Olaf Wiest, and Xiangliang Zhang. Large language model based multi-agents: A survey of progress and challenges. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence (IJCAI 2024), 2024.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. Applied Sciences, 11(14):6421, 2021. doi: 10.3390/app11146421.

Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: communicative agents for mind exploration of large language model society. In Proceedings ofthe 37th International Conference on Neural Information Processing Systems, NIPS ’23, 2023.

Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 4582–4597. Association for Computational Linguistics, 2021. doi: 10.18653/v1/2021. acl-long.353.

Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and Zhaopeng Tu. Encouraging divergent thinking in large language models through multi-agent debate. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 17889–17904, 2024.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. Advances in Neural Information Processing Systems, 36:21558–21572, 2023.

Maxwell-Jia. AIME 2024 dataset. https://huggingface.co/datasets/Maxwell-Jia/AIME 2024, 2024.

Elizabeth Mieczkowski, Ruaridh Mon-Williams, Neil Bramley, Christopher G Lucas, Natalia Velez, and Thomas L Griffiths. Predicting multi-agent specialization via task parallelizability. arXiv preprint arXiv:2503.15703, 2025.

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pp. 1–22, 2023.

Chau Pham, Boyi Liu, Yingxiang Yang, Zhengyu Chen, Tianyi Liu, Jianbo Yuan, Bryan A. Plummer, Zhaoran Wang, and Hongxia Yang. Let models speak ciphers: Multiagent debate through embeddings. In The Twelfth International Conference on Learning Representations, 2024.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level googleproof q&a benchmark. In First Conference on Language Modeling, 2024.

Peter H. Schonemann. A generalized solution of the orthogonal Procrustes problem. ¨ Psychometrika, 31(1):1–10, 1966. doi: 10.1007/BF02289451.

Zhi Rui Tam, Cheng-Kuang Wu, Yi-Lin Tsai, Chieh-Yen Lin, Hung-yi Lee, and Yun-Nung Chen. Let me speak freely? a study on the impact of format restrictions on large language model performance. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, pp. 1218–1236. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.emnlp-industry.91.

Wei Tao, Yucheng Zhou, Yanlin Wang, Wenqiang Zhang, Hongyu Zhang, and Yu Cheng. Magis: Llm-based multi-agent framework for github issue resolution. Advances in Neural Information Processing Systems, 37:51963–51993, 2024.

Team OLMo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, et al. Olmo 3. arXiv preprint arXiv:2512.13961, 2026.

Zhao Wang, Sota Moriyama, Wei-Yao Wang, Briti Gangopadhyay, and Shingo Takamatsu. Talk structurally, act hierarchically: A collaborative framework for llm multi-agent systems. arXiv preprint arXiv:2502.11098, 2025.

Feijie Wu, Zitao Li, Fei Wei, Yaliang Li, Bolin Ding, and Jing Gao. Talk to right specialists: Iterative routing in multi-agent systems for question answering. arXiv preprint arXiv:2501.07813, 2025.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via multi-agent conversations. In First Conference on Language Modeling, 2024.

Bingyu Yan, Zhibo Zhou, Litian Zhang, Lian Zhang, Ziyi Zhou, Dezhuang Miao, Zhoujun Li, Chaozhuo Li, and Xiaoming Zhang. Beyond self-talk: A communication-centric survey of llm-based multi-agent systems. arXiv preprint arXiv:2502.14321, 2026.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Yingxuan Yang, Qiuying Peng, Jun Wang, Ying Wen, and Weinan Zhang. Llm-based multi agent systems: Techniques and business perspectives. arXiv preprint arXiv:2411.14033, 2024.

Jiabo Ye, Xi Zhang, Haiyang Xu, et al. Mobile-agent-v3: Fundamental agents for gui automation. arXiv preprint arXiv:2508.15144, 2025.

Ling Yue, Sixue Xing, Jintai Chen, and Tianfan Fu. Clinicalagent: Clinical trial multi-agent system with large language model-based reasoning. In Proceedings of the 15th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics, pp. 1–10, 2024.

Chaoyun Zhang, Shilin He, Jiaxu Qian, et al. Large language model-brained gui agents: A survey. Transactions on Machine Learning Research, 2025a.

Guibin Zhang, Yanwei Yue, Zhixun Li, Sukwon Yun, Guancheng Wan, Kun Wang, Dawei Cheng, Jeffrey Xu Yu, and Tianlong Chen. Cut the crap: An economical communication pipeline for llm-based multi-agent systems. In The Thirteenth International Conference on Learning Representations, 2025b.

Yifan Zhang and Team Math-AI. AIME 2025 dataset. https://huggingface.co/datasets/ math-ai/aime25, 2025.

Yusen Zhang, Ruoxi Sun, Yanfei Chen, Tomas Pfister, Rui Zhang, and Sercan Arik. Chain of agents: Large language models collaborating on long-context tasks. Advances in Neural Information Processing Systems, 37:132208–132237, 2024.

Wanjia Zhao, Mert Yuksekgonul, Shirley Wu, and James Zou. Sirius: Self-improving multiagent systems via bootstrapped reasoning. In Advances in Neural Information Processing Systems, 2025.

Yujia Zheng, Zhuokai Zhao, Zijian Li, Yaqi Xie, Mingze Gao, Lizhu Zhang, and Kun Zhang. Thought communication in multiagent collaboration. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Jiaru Zou, Ruizhong Qiu, Gaotang Li, Xiyuan Yang, Katherine Tieu, Pan Lu, Ke Shen, Hanghang Tong, Yejin Choi, Jingrui He, James Zou, Mengdi Wang, and Ling Yang. Latent collaboration in multi-agent systems. In Forty-third International Conference on Machine Learning, 2026.

## A Theoretical analysis

In this section, we analyze the geometric properties of two alignment strategies used in experiments: orthogonal Procrustes and ridge regression. We first formalize the information bottleneck in text communication, then contrast the two approaches at the representation level.

## A.1 Information loss in text communication

Standard text communication maps the sender’s hidden states $\pmb { S } \in \mathbb { R } ^ { K \times d }$ to a discrete token sequence $\mathbf { y } \in \{ 1 , \ldots , V \} ^ { K }$ via sampling. This mapping is many-to-one: distinct hidden-state matrices that produce the same token sequence become indistinguishable to the receiver. We treat S and y as random variables induced by the data distribution and decoding randomness. Since $\mathbf { \dot { W } } _ { \mathrm { e m b } } [ \mathbf { y } ]$ is a deterministic function of $\mathbf { y } ,$ the data processing inequality bounds the mutual information that the inter-agent channel carries:

$$
I ( \mathbf { S } ; \mathbf { W } _ { \mathrm { e m b } } [ \mathbf { y } ] ) \leq I ( \mathbf { S } ; \mathbf { y } ) \leq H ( \mathbf { y } ) \leq K \log _ { 2 } V ,\tag{10}
$$

where the first inequality becomes an equality under injective embedding lookup. When communication is restricted to the retained K sampled tokens, the channel carries at most $K \log _ { 2 }$ V bits; for $V \approx 1 . 5 \times 1 0 ^ { 5 }$ and $K = 6 4$ , this gives roughly 1101 bits, regardless of how much information the $K \times d$ continuous hidden states contain. Continuous transmission is not subject to the same combinatorial bound, although its effective capacity depends on numerical precision and the receiver’s ability to interpret the representation.

## A.2 Confinement to the token-embedding span under ridge alignment

We now show that replacing Procrustes with an unconstrained linear map confines the transmitted message to the span of the discrete token embeddings, regardless of the regularization strength. Consider ridge regression alignment, which solves

$$
\mathbf { W } _ { \mathrm { r i d g e } } = \mathop { \arg \operatorname* { m i n } } _ { \mathbf { W } \in \mathbb { R } ^ { d \times d } } \| \mathbf { S } \mathbf { W } - \mathbf { R } \| _ { F } ^ { 2 } + \gamma \| \mathbf { W } \| _ { F } ^ { 2 } ,\tag{11}
$$

with closed-form solution

$$
\mathbf { W } _ { \mathrm { r i d g e } } = ( \mathbf { S } ^ { \top } \mathbf { S } + \gamma \mathbf { I } ) ^ { - 1 } \mathbf { S } ^ { \top } \mathbf { R } .\tag{12}
$$

Proposition A.1 (Span Confinement). The ridge-aligned prefix $\mathbf { S } _ { \mathrm { r i d g e } } = \mathbf { S } \mathbf { W } _ { \mathrm { r i d g e } }$ satisfies

$$
\begin{array} { r } { { \bf S } _ { \mathrm { r i d g e } } = { \bf H } _ { \gamma } { \bf R } , } \end{array}\tag{13}
$$

where $\mathbf { H } _ { \gamma } = ( \mathbf { S } \mathbf { S } ^ { \intercal } + \gamma \mathbf { I } ) ^ { - 1 } \mathbf { S } \mathbf { S } ^ { \intercal } \in \mathbb { R } ^ { K \times K }$ . Consequently, every row of $\mathbf { S } _ { \mathrm { r i d g e } }$ is a linear combination of the rows of R, and the aligned prefix is confined to span $\left( \mathbf { r } _ { 1 } , \ldots , \mathbf { r } _ { K } \right)$

Proof. Applying the push-through identity $\mathbf { A } ( \mathbf { A } ^ { \top } \mathbf { A } + \gamma \mathbf { I } ) ^ { - 1 } = ( \mathbf { A } \mathbf { A } ^ { \top } + \gamma \mathbf { I } ) ^ { - 1 } \mathbf { A }$ with $\mathbf { A } = \mathbf { S } { \mathrm { : } }$

$$
\begin{array} { r } { { \pmb S } _ { \mathrm { r i d g e } } = { \pmb S } ( { \pmb S } ^ { \top } { \pmb S } + \gamma { \bf I } ) ^ { - 1 } { \pmb S } ^ { \top } { \pmb R } = ( { \pmb S } { \pmb S } ^ { \top } + \gamma { \bf I } ) ^ { - 1 } { \pmb S } { \pmb S } ^ { \top } { \pmb R } = { \bf H } _ { \gamma } { \bf R } . } \end{array}\tag{14}
$$

Since ${ \bf H } _ { \gamma } \in \mathbb { R } ^ { K \times K }$ , row i of $\mathbf { S } _ { \mathrm { r i d g e } }$ equals $\Sigma _ { j = 1 } ^ { K } ( \mathbf { H } _ { \gamma } ) _ { i j } \mathbf { r } _ { j } \in \mathsf { s p a n } ( \mathbf { r } _ { 1 } , \ldots , \mathbf { r } _ { K } )$

Corollary A.2 (Limiting Behavior). When S has full row rank (which holds generically for $K \leq d$ and rows in general position), $\mathbf { H } _ { \gamma }  \mathbf { I } _ { K }$ as $\gamma  \hat { 0 } , s o \ \pmb { \mathrm { S } } _ { \mathrm { r i d g e } }  \pmb { \mathrm { R } }$ . In the opposite limit, ${ \bf H } _ { \gamma }  { \bf 0 }$ as $\gamma \to \infty ,$ , so the prefix vanishes entirely.

Because R contains the embeddings of the K sampled text tokens, its row space is at most K-dimensional and is determined entirely by the discrete token identities. Proposition A.1 shows that the ridge-aligned prefix is confined to this span for all $\gamma ,$ while Corollary A.2 establishes the two limiting regimes: text reconstruction $( \gamma  0 )$ and silence $( \gamma \to \infty )$ . For intermediate $\gamma ,$ the S-dependent coefficients in $\mathbf { H } _ { \gamma }$ still encode continuous variation within span(R), but directions outside this subspace remain inaccessible.

## A.3 Geometric preservation under Procrustes alignment

In contrast, the orthogonal Procrustes alignment preserves the internal geometry of the sender’s states in the whitened space.

Proposition A.3 (Isometry of Procrustes Alignment). Let ${ \bf Q } ^ { \ast } \in \mathbb { R } ^ { d \times d }$ be the orthogonal Procrustes solution. For any two rows $i , j$ of the whitened sender states $\mathbf { s } _ { w } .$

$$
\lVert ( \mathbf { S } _ { w } \mathbf { Q } ^ { * } ) _ { i } - ( \mathbf { S } _ { w } \mathbf { Q } ^ { * } ) _ { j } \rVert _ { 2 } = \lVert ( \mathbf { S } _ { w } ) _ { i } - ( \mathbf { S } _ { w } ) _ { j } \rVert _ { 2 } ,\tag{15}
$$

$$
\langle ( \mathsf { \pmb { S } } _ { w } \mathbf { Q } ^ { \ast } ) _ { i } , ( \mathsf { \pmb { S } } _ { w } \mathbf { Q } ^ { \ast } ) _ { j } \rangle = \langle ( \mathsf { \pmb { S } } _ { w } ) _ { i } , ( \mathsf { \pmb { S } } _ { w } ) _ { j } \rangle .\tag{16}
$$

Hence the sequence Gram matrix is preserved: $( \mathbf { S } _ { w } \mathbf { Q } ^ { * } ) ( \mathbf { S } _ { w } \mathbf { Q } ^ { * } ) ^ { \top } = \mathbf { S } _ { w } \mathbf { S } _ { w } ^ { \top } ,$

Proof. Since $( \mathbf { Q } ^ { * } ) ^ { \top } \mathbf { Q } ^ { * } = \mathbf { I } .$ , right-multiplying any matrix by $\mathbf { Q } ^ { * }$ preserves its Gram matrix: $( \mathbf { S } _ { w } \mathbf { Q } ^ { * } ) ( \mathbf { S } _ { w } \mathbf { Q } ^ { * } ) ^ { \top } \mathbf { \Phi } = \mathbf { S } _ { w } \mathbf { Q } ^ { * } ( \bar { \mathbf { Q } ^ { * } } ) ^ { \top } \mathbf { S } _ { w } ^ { \top } \mathbf { \Phi } = \mathbf { \bar { S } } _ { w } \bar { \mathbf { S } } _ { w } ^ { \top }$ . Since the $( i , j )$ entry of the Gram matrix equals $\langle ( \mathbf { S } _ { w } ) _ { i } , \langle \mathbf { S } _ { w } ) _ { j } \rangle$ , inner products between rows are preserved. Distance preservation follows because $\| \mathbf { a } - \mathbf { b } \| _ { 2 } ^ { 2 } = \| \mathbf { a } \| _ { 2 } ^ { 2 } - 2 \langle \mathbf { a } , \mathbf { b } \rangle + \| \mathbf { b } \| _ { 2 } ^ { 2 }$ depends only on inner products.

The isometry guarantee established above applies to the orthogonal rotation step in whitened coordinates; the subsequent de-whitening, norm calibration, and vocabulary anchoring steps modify the final prefix further. Within this scope, the key property is that Procrustes preserves the pairwise Euclidean structure of the whitened sender states: tokens that were geometrically close (or distant) in the sender’s internal representation remain so after the alignment step. By contrast, an unconstrained linear map such as ridge regression arbitrarily rescales and shears these relationships, even when the pointwise reconstruction error is low.

The combination of Propositions A.1 and A.3 provides a geometric rationale for the ablation results in Section 6: ridge regression distorts the pairwise structure of the sender’s representation during alignment, while the orthogonal Procrustes step preserves it. This distinction is especially relevant for tasks such as code generation, where the receiver must produce structurally precise output and is therefore sensitive to geometric distortion in the input representation. Together, these results suggest that the orthogonal constraint is more than a regularization choice: it ensures that the alignment step preserves the internal geometric relationships of the whitened sender representation.

## B Additional experimental details

## B.1 Dataset descriptions

We provide brief descriptions of all evaluation benchmarks, grouped by the three categories introduced in Section 4.

Mathematical Reasoning.

• GSM8K (Cobbe et al., 2021) contains 8.5K grade-school math word problems that require multi-step numerical reasoning. Each problem must be decomposed into structured arithmetic steps, making it a standard testbed for chain-of-thought reasoning.

• AIME24 (Maxwell-Jia, 2024) consists of 30 competition-level problems from the 2024 American Invitational Mathematics Examination. Problems span algebra, geometry, number theory, and combinatorics, and require precise numeric answers.

• AIME25 (Zhang & Math-AI, 2025) provides 30 additional problems from the 2025 AIME exam. Compared with AIME24, this set includes more multi-phase derivations and intricate combinatorial constructions, offering a complementary test of mathematical reasoning.

## Knowledge-Intensive QA.

• GPQA-Diamond (Rein et al., 2024) is the most difficult split of the GPQA benchmark, featuring 198 graduate-level multiple-choice questions written by domain experts in physics, biology, and chemistry. The dataset emphasizes conceptual depth and cross-disciplinary reasoning.

• ARC-Challenge (Clark et al., 2018) includes the most difficult items from the AI2 Reasoning Challenge. These questions require multi-hop reasoning and systematic elimination of distractors, making performance on this benchmark a useful indicator of multi-step reasoning.

• MedQA (Jin et al., 2021) contains real medical licensing exam questions that assess biomedical knowledge, clinical reasoning, and diagnostic decision-making. Problems require integrating textual context with domain-specific medical understanding.

## Code Generation.

• MBPP+ (Liu et al., 2023) extends the original MBPP benchmark with broader input coverage, additional hidden test cases, and stricter execution-based evaluation. Each problem requires generating a self-contained Python function that satisfies a comprehensive unit-test suite.

• HumanEval+ (Liu et al., 2023) augments HumanEval with denser, more challenging test suites, increasing the rigor of functional correctness evaluation. The benchmark emphasizes generalization beyond prompt examples and tests a model’s ability to produce semantically precise, executable Python code.

## B.2 Agent pipeline

Following Zou et al. (2026), we instantiate a sequential four-agent pipeline: (i) Planner decomposes the problem and outlines a solution strategy; (ii) Critic identifies potential errors or gaps in the plan; (iii) Refiner incorporates feedback and produces an improved solution; and (iv) Judger synthesizes the preceding outputs and generates the final answer. All methods share nearly identical prompts; the only difference is the description of the communication modality (“text format” for TextMAS, “latent KV representation format” for LatentMAS, “embedding representation format” for StateBridge).

## B.3 Implementation details

We implement all methods in Python using PyTorch<sup>2</sup> and HuggingFace Transformers<sup>3</sup>. We use each model’s official chat template and special tokens for prompt formatting. For StateBridge, we extract final-layer hidden states from the sender’s generation pass via a forward hook on the last transformer layer and compute the Procrustes alignment (centering, whitening, SVD, and reconstruction) on GPU. We then concatenate the aligned prefix with the receiver’s prompt embeddings and pass it through the standard forward method via inputs embeds, with no modification to the model architecture or attention mask.

Following Zou et al. (2026), we set the maximum output length to 2,048 tokens for GSM8K and ARC-C, 4,096 tokens for MBPP+ and HumanEval+, 8,192 tokens for MedQA and GPQA, and 20,000 tokens for AIME24/25.

## B.4 Evaluation protocol

We evaluate all methods using task-specific protocols.

Multiple-choice QA (GPQA-Diamond, MedQA, ARC-Challenge). We extract the model’s final answer string and compare it via exact match to the ground-truth answer letter.

Mathematical reasoning (GSM8K, AIME24, AIME25). We extract the final predicted answer, parse both prediction and ground truth into numbers, and mark the sample as correct only if the two values match. Predictions that fail numeric parsing are counted as incorrect.

Code generation (MBPP+, HumanEval+). We extract the predicted code from the model’s output, append the ground-truth unit tests provided by the benchmark, and execute the

combined script in a sandboxed environment with a 10-second timeout. A sample is counted as correct if and only if all tests pass without runtime errors.

For all non-coding benchmarks, answer extraction applies text normalization (lowercasing, trimming whitespace, and removing extraneous punctuation) before matching.

## C Prompt templates

The following prompt templates are shown for StateBridge. The Planner, Critic, and Refiner prompts are shared across all task categories; only the Judger prompt varies to specify the answer format.

![](images/dc219a706c948ba18688a6538ef50ef57abb0cf1ca38c9eec3d585d1519307a8.jpg)

## StateBridge Prompts for Multiple-Choice Tasks (ARC-C / GPQA / MedQA)

<table><tr><td>System Prompt for All Agents: You are Qwen/Olmo. You are a helpful assistant.</td></tr><tr><td>Prompt for Planner Agent: You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.</td></tr><tr><td>Question: {question} Your outlined plan should be concise with a few bullet points for each step. Do not produce the final answēr. Now output your plan to solve the question below:</td></tr><tr><td>Prompt for Critic Agent: The following is the message from previous Agent (provided in embedding format): [Aligned continuous prefix is inserted here]</td></tr><tr><td>Question: {question} You are a Ċritic Agent to evaluate the correctness of the input plan for the given question and provide helpful feedback for improving the plan. The plan information is provided in embedding representation format. Review the plan and question and output: (1) original plan contents (2) constructive feedback on the original plan.</td></tr><tr><td>Format your response as follows: Originaí Plan: [Ċopy the provided Planner Agent&#x27;s plan here] Feedback: [Your detailed feedback to improve the plan here] Now, output your response below:</td></tr><tr><td>Prompt for Refiner Agent: The following is the message from previous Agent (provided in embedding format): [Aligned continuous prefix is inserted here]</td></tr><tr><td>Question: {question} You are a Reiner Agent to provide a refined step-by-step plan for solving the given question. You are provided with: (1) embedding-format information: a previous plan with feedback (2)</td></tr><tr><td>text-format information: the input question you need to solve. Based on the input, write a refined and improved plan to solve the question. Make sure your output plan is correct and concise. Now, output your refined plan below:</td></tr><tr><td>Prompt for Judger Agent: The following is the message from previous Agent (provided in embedding format): [Aligned continuous prefix is inserted here]</td></tr><tr><td>Target Question: {question} You are a helpful assistant. You are provided with embedding information for reference and a target question to solve. The embedding information might contain irrelevant contents.</td></tr><tr><td>Ignore it if it is not helpful for solving the target question. You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.</td></tr></table>

## StateBridge Prompts for Code Generation Tasks (MBPP+ / HumanEval+)

<table><tr><td>StateBridge Frompts for Code Generation Tasks</td></tr><tr><td>System Prompt for All Agents: You are Qwen/Olmo. You are a helpful assistant.</td></tr><tr><td>Prompt for Planner Agent: You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to</td></tr><tr><td>solve the question. Question: {question}</td></tr><tr><td>Your outlined plan should be concise with a few bullet points for each step. Do not produce the final answer. Now output your plan to solve the question below: Prompt for Critic Agent:</td></tr><tr><td>The following is the message from previous Agent (provided in embedding format): [Aligned continuous prefix is inserted here]</td></tr><tr><td>Question: {question} You are a Ċritic Agent to evaluate the correctness of the input plan for the given question</td></tr><tr><td>and provide helpful feedback for improving the plan. The plan information is provided in embedding representation format. Review the plan and question and output: (1) original plan contents (2) constructive feedback on the original plan.</td></tr><tr><td>Format your response as follows: Original Plan: [Copy the provided Planner Agent&#x27;s plan here]</td></tr><tr><td>Feedback: [Your detailed feedback to improve the plan here] Now, output your response below:</td></tr><tr><td>Prompt for Refiner Agent: The following is the message from previous Agent (provided in embedding format):</td></tr><tr><td>[Aligned continuous prefix is inserted here]</td></tr><tr><td>Question: {question}</td></tr><tr><td>You are a Reiner Agent to provide a refined step-by-step plan for solving the given question. You are provided with: (1) embedding-format information: a previous plan with feedback (2) text-format information: the input question you need to solve.</td></tr><tr><td>Based on the input, write a refined ând improved plan to solve the question. Make sure your output plan is correct and concise.</td></tr><tr><td>Now, output your refined plan below: Prompt for Judger Agent:</td></tr><tr><td>The following is the message from previous Agent (provided in embedding format):</td></tr><tr><td>[Aligned continuous prefix is inserted here]</td></tr><tr><td>Target Question: {question} You are a helpful assistant. You are provided with embedding information for reference and a target question to solve. The embedding information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.</td></tr></table>