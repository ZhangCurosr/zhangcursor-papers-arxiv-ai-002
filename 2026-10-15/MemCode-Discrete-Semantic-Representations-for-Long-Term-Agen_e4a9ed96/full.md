# MemCode: Discrete Semantic Representations for Long-Term Agent Memory

Xin Li<sup>1</sup> Liang Hu<sup>1</sup> Duoqian Miao<sup>1</sup> Ming Peng<sup>1</sup> Qi Zhang<sup>1,2∗</sup> <sup>1</sup>Tongji University <sup>2</sup>WorldMind

{lx cs, lianghu, dqmiao, pengming, zhangqi cs}@tongji.edu.cn

## Abstract

Long-term memory remains a fundamental bottleneck for large language model (LLM) agents across sessions. Existing systems typically adopt a write-thenretrieve paradigm, storing context-dependent dialogue fragments and retrieving them via nearest-neighbor search in continuous embedding space. This design yields unstable access units and systematically misses logically related but lexically distant evidence. We propose MemCode, a long-term agent memory framework that couples retrieval-oriented writing with discrete associative retrieval. MemCode rewrites dialogue into self-contained atomic memory units (AMUs), maps AMUs into a learnable codebook with isomorphic semantic quantization (ISQ), and uses codebook co-occurrence topology as a retrieval path complementary to vector search. For long-term memory stores, MemCode separates online atomic writes from asynchronous offline consolidation. On LongMemEval-S and LoCoMo with GPT-4o-mini and Qwen3-30B-A3B-Instruct-2507 backbones, MemCode achieves the best overall accuracy while keeping per-dialogue memory construction within roughly 40∼130 seconds, making it 3∼21× faster than LightMem and over an order of magnitude faster than A-MEM and MemoryOS.

## 1 Introduction

Large language model (LLM) agents are rapidly evolving from single-turn systems into persistent assistants that interact with users over extended sessions [1, 2]. However, neither model parameters nor context windows provide a satisfactory substrate for this need. Parameters are largely static after training, while long contexts incur quadratic attention costs [3] and exhibit degraded utilization as sequences grow [4]. It fundamentally elevates the role of agent memory that, distinct from both parameters and context, must accumulate, organize, and recall information across sessions to support personalization, consistency, and long-horizon reasoning.

As a result, external memory has emerged as a central component for agent systems. Most memory systems follow a retrieval-augmented generation (RAG) paradigm [5, 6]: past interactions are written into storage, retrieved at query time, and re-injected into the model context. In this setting, two design choices are tightly coupled: 1) what to write, and 2) how to retrieve it. Prior systems address these choices primarily from memory persistence. Mem0 [7], LangMem [8], and A-MEM [9] improve writing and organization through Create, Read, Update, and Delete (CRUD) operations, typed memories, and evolvable note networks; MemoryOS [10] and LightMem [11] improve scalability through scheduling and compression. However, we argue that the key bottleneck for RAG-style long-term memory is not storage scale alone, but jointly modeling what to write and how to retrieve it, as illustrated in Figure 1.

This coupling becomes sharper as conversations grow. A memory item may be easy to store but expensive to use if it lacks the necessary context, such as missing speakers or times. Conversely, a compressed summary might answer broad questions but can erase the factual hooks needed for multi-hop recall. Our goal is to create independently retrievable memory units that maintain pointers to raw dialogue, avoiding irreversible abstraction bottlenecks.

![](images/81164eef07894f9d9f229afc31a33ce64957b305f57b2d7dca9336b6f65ed13c.jpg)  
Figure 1: Traditional memory pipelines store raw dialogue fragments and retrieve them primarily by surface similarity. Ours rewrites dialogue into atomic memory units and retrieves them through complementary vector and codebook paths.

Specifically, the writing pipeline is retrieval-oriented memory modeling, focusing on creating retrieval-accessible targets that remain understandable and indexable after separation from the orig inal dialogue. Context-dependent fragments are weak storage units; instead, atomic facts are desirable. Long-term recall is usually cued by events and relations, not exact wording. In addition, the retrieval issue revolves around access representation. Most systems use continuous vectors for memory, retrieving content via top-K nearest-neighbor searches [12], as illustrated by the upper pipeline in Figure 1. This works for close memories in surface semantics but inherits the geometric bias of continuous representations: logically related but lexically distant content may be missed, consistent with evidence that dense retrievers can be brittle when retrieval conditions shift across tasks [13]. In contrast, human memory distinguishes itself through pattern separation, allowing the hippocampus to encode similar continuous inputs into discrete traces for associative recall of partial cues [14, 15]. Systems that just mimic human multi-level memory without discrete anchors may still misalign architecture and retrieval.

Our proposed MemCode retains the RAG-style pipeline while adding a discrete representation layer alongside continuous vectors, utilized throughout writing, retrieval, and long-term maintenance. During writing, we employ LLM-based ingestion and abstraction to transform dialogue into selfcontained AMUs with source pointers. Each AMU serves as a minimal semantic unit that can be independently retrieved. Independent semantic quantization (ISQ) anchors each AMU to a discrete symbolic state within a learnable codebook. During retrieval, global codebook co-occurrence offers an alternative access path for logically related yet geometrically distant memories, while source pointers enhance context reconstruction. MemCode also separates online writing from offline consolidation, allowing clustering and synthesizing AMUs without disrupting real-time retrieval.

Contributions. Our contributions are fourfold. (1) We propose MemCode, a unified write–index– retrieve–consolidate framework that addresses the write–retrieve mismatch in long-term agent memory by organizing memory around atomic facts and discrete representations. (2) We introduce AMUs as the retrieval unit of this framework, turning raw dialogue into self-contained and source-grounded facts. (3) We pair AMUs with ISQ and codebook topology, giving each fact both a continuous vector interface and a discrete associative path for evidence beyond local embedding neighborhoods. (4) On LongMemEval-S, MemCode reaches 78.40%/74.60% Overall with GPT-4o-mini/Qwen3 after asynchronous consolidation; on LoCoMo it reaches 79.55%/76.88%, while keeping construction runtime roughly 3∼21× faster than LightMem and over an order of magnitude faster than A-MEM and MemoryOS.

## 2 Related Work

## 2.1 RAG-Based Long-Term Memory for Agents

Most long-term memory schemes for agents are built on RAG, which combines external stores with generative models to reduce knowledge cutoffs and hallucination; in cross-session memory, the same pattern writes, indexes, and invokes historical interactions [16]. Gao et al. [17] identify retrieval precision and context utilization as central limits that long-term memory inherits.

Prior work has explored diverse ways to write and organize memories. Mem0 [7] maintains a dynamic fact store through LLM-driven CRUD operations. A-MEM [9] organizes memories as an evolvable note network with explicit association edges, while LangMem [8] separates semantic, episodic, and procedural memory. These methods move beyond simple caches, but most machinery still sits above a continuous vector index. This leaves two gaps: 1) pronouns and relative-time expressions can make memory units hard to retrieve once detached from the original dialogue; 2) nearest-neighbor access is poorly matched to memories that are logically related but distant in surface semantics. Other systems focus on storage management and lifecycle maintenance. MemoryOS [10] divides memory into short-, medium-, and long-term levels with heat-driven eviction, LightMem [11] lowers token cost through offline compression, and MEMORY-VQ [18] uses VQ-VAE to compress precomputed token representations for memory-augmented models. These methods improve storage use and throughput, but operate mainly at scheduling, compression, and eviction layers rather than retrieval-oriented memory organization. A long-term memory store can still grow redundant fragments with overlapping semantics, weakening later retrieval [19].

## 2.2 VQ Codebooks as Discrete Memory Anchors

In machine learning, vector-quantized variational autoencoders (VQ-VAE) [20] provide a standard discretization tool. VQ-VAE maps continuous encodings to discrete embeddings in a finite codebook through a nearest-neighbor quantizer and uses the straight-through estimator (STE) for the non-differentiable quantization step. A finite codebook encourages distinguishable semantic primitives and has been widely used in image generation and speech coding [21]. Discrete codebooks can therefore serve as compact anchors for high-dimensional continuous representations. This codebook view is also compatible with neurocognitive memory intuition. Yassa and Stark [14] describe how the hippocampal dentate gyrus separates similar inputs into sparse traces, while the hippocampal CA3 subfield supports pattern completion from partial cues [15]. For agent memory, the relevant lesson is narrower: discrete anchors can separate similar facts and support associative retrieval beyond geometric nearest neighbors. Dialogue memory, however, requires a retrieval-oriented use of VQ rather than direct lossy compression: because AMUs contain timestamps, entity identifiers, and state descriptions as factual anchors, MemCode applies stride-1 ISQ to preserve token-level alignment and uses the resulting discrete codes as associative retrieval keys.

## 3 Methodology

## 3.1 System Overview

Figure 2 summarizes MemCode. Ingestion canonicalizes dialogue into self-contained facts (Section 3.2). Dual-path indexing maps each fact to a dense vector and an ISQ codebook fingerprint (Section 3.3). Retrieval uses the dense vector for local matching, then uses codebook co-occurrence to add associated evidence and reconstruct source context (Section 3.4). Consolidation clusters, merges, and deduplicates memories offline (Section 3.5).

## 3.2 Stream-Based Atomic Abstraction Framework

Interaction chunking & context injection. The writing stage preserves streaming order. We represent the raw stream as $\mathcal { S } = \langle m _ { 1 } , \ldots , m _ { N } \rangle$ , where $m _ { i } = ( r _ { i } , t _ { i } , c _ { i } )$ contains role, timestamp, and text. Each compound message $c _ { i }$ is split into atomic sentences s<sub>i</sub> [22], and messages are grouped into true interaction turns $\bar { \boldsymbol { B } } ^ { \prime } = \{ B _ { 1 } , \bar { \cdot } \cdot \cdot , B _ { K _ { \mathrm { t u r n } } } \}$ . If $\mathcal { T } _ { k }$ is the message-index set of turn k, then $B _ { k } = \langle m _ { i } : i \in \mathcal { I } _ { k } \rangle$ in chronological order. Because strict turn segmentation can break coreference,

![](images/985044feee91b3ecfab72276db271b525b743d7b35ee01ffc07d15b22e098990.jpg)  
Figure 2: MemCode architecture. Starting from raw dialogue streams, Ingestion & Abstraction preserves source-grounded base memory and rewrites interactions into self-contained atomic memory units (AMUs). Dual-Path Indexing assigns each AMU both a continuous embedding and a discrete codebook fingerprint. During the Retrieval Loop, vector search provides high-precision matches, codebook topology adds associative evidence, and source pointers reconstruct richer context for answer generation. Memory Evolution then asynchronously consolidates long-term facts through clustering, deduplication, temporal refinement, and logic-oriented merging.

MemCode processes the k-th turn with a suffix of $B _ { k - 1 }$ as read-only background:

$$
\begin{array} { r } { \mathcal { H } _ { k } = \mathrm { S u f f i x } ( B _ { k - 1 } , \omega ) , \qquad \mathcal { X } _ { k } = \underbrace { \mathcal { H } _ { k } } _ { \mathrm { B a c k g r o u n d C o n t e x t } } \oplus \underbrace { B _ { k } } _ { \mathrm { A c t i v e T a r g e t } } . } \end{array}\tag{1}
$$

Here, $\mathcal { H } _ { k }$ contains the last $\omega$ sentences from the previous turn, with $\mathcal { H } _ { 1 } = \varnothing$ for the first turn; ω is the extraction-background budget, and ⊕ denotes concatenation. The background resolves local coreference but is excluded from fact extraction.

Semantic canonicalization & atomic extraction. Given $\mathcal { X } _ { k } .$ , the abstraction module rewrites the active turn $B _ { k }$ into self-contained facts $\mathcal { F } _ { k }$ that remain retrievable after being detached from the dialogue. We model this as a generative mapping conditioned on the read-only background and metadata extracted from $\mathcal { X } _ { k }$

$$
\mathcal { F } _ { k } = \Phi ( B _ { k } \mid \mathcal { H } _ { k } , T _ { a n c h o r } , \mathcal { R } _ { r o l e s } )\tag{2}
$$

Here, Φ is the LLM abstraction engine. $T _ { a n c h o r }$ is derived from message timestamps in $\mathcal { X } _ { k }$ and anchors relative time, while $\mathcal { R } _ { r o l e s }$ is the role–speaker map extracted from message roles and used with background to resolve pronouns [23]. Each output AMU $u \in \mathcal { F } _ { k }$ is denoted as:

$$
u = ( c _ { f a c t } , ~ t _ { a b s } , ~ \Lambda _ { s r c } )\tag{3}
$$

where $c _ { f a c t }$ is the canonicalized fact generated from $B _ { k } , t _ { a b s }$ is the absolute timestamp obtained by resolving relative time against $T _ { a n c h o r }$ , and $\Lambda _ { s r c }$ points to original spans in $B _ { k }$ and, when needed for grounding, $\mathcal { H } _ { k }$ . During retrieval, these pointers recover nearby utterances when an atomic fact matches the query but is too terse for answer generation.

## 3.3 Fine-Grained Discrete Semantic Indexing

Isomorphic semantic quantization. To build an associative discrete index, MemCode follows VQ VAE and treats the codebook as a latent state basis [24]. For AMU text $x = c _ { f a c t }$ , ISQ obtains encoder states $H = ( h _ { 1 } , \ldots , h _ { L } )$ from a T5-base encoder fine-tuned on DailyDialog [25]. Here, stride denotes the projection step over token positions. A learned stride-1 projection maps each state to $z _ { t } = W _ { p } h _ { t } + b _ { p }$ , giving $Z = ( z _ { 1 } , \dots , z _ { L } )$ and preserving $L _ { \mathrm { o u t } } = L$ so timestamps, numbers, and entity tokens are not dropped. We then introduce a learnable Euclidean codebook $\mathcal { E } = \{ e _ { j } \} _ { j = 1 } ^ { N _ { c } }$ where $e _ { j }$ is the $j \cdot$ -th code vector and $N _ { c }$ is the codebook size. Each projected vector $z _ { t }$ is assigned to its nearest code:

$$
\kappa _ { t } = \underset { j \in \{ 1 , \dots , N _ { c } \} } { \mathrm { a r g m i n } } \| z _ { t } - e _ { j } \| _ { 2 } , \qquad \bar { z } _ { t } = e _ { \kappa _ { t } } .\tag{4}
$$

Here, $\kappa _ { t }$ is the discrete code id and $\bar { z } _ { t }$ is the quantized vector used in place of $z _ { t }$ by the decoder.   
Backpropagation uses the straight-through estimator (STE) [26].

Conversational codebook learning & semantic fingerprinting. The codebook is trained with predictive coding rather than input reconstruction: for each context–response pair, the model predicts the response from the quantized context representation [27], preserving reasoning cues while filtering shallow lexical noise. The loss is:

$$
\mathcal { L } _ { t o t a l } = - \sum _ { r = 1 } ^ { | y | } \log p _ { \theta } ( y _ { r } \mid y _ { < r } , \bar { Z } ) + \lambda _ { \mathrm { v q } } \sum _ { t = 1 } ^ { L } \ell _ { \mathrm { v q } } ( z _ { t } , \bar { z } _ { t } ) .\tag{5}
$$

Here, y is the target response, θ denotes the decoder parameters, $\bar { Z } = ( \bar { z } _ { 1 } , \dots , \bar { z } _ { L } )$ is the quantized context sequence, $\ell _ { \mathrm { v q } } \stackrel { \cdot } { = } \| \mathrm { s g } [ z _ { t } ] - \bar { z } _ { t } \| _ { 2 } ^ { 2 } + \beta \| z _ { t } - \mathrm { s g } [ \bar { z _ { t } } ] \| _ { 2 } ^ { 2 } , \mathrm { s g } [ \cdot ]$ stops gradients, β is the commitment weight, and $\lambda _ { \mathrm { v q } }$ balances generation and quantization. The codebook E is updated by EMA [28]; low-frequency codewords are reinitialized from current-batch encoder outputs. At inference time, adjacent repeats in $\kappa _ { 1 : L }$ are removed to form a compact fingerprint:

$$
S _ { f p } ( u ) = \langle \kappa _ { t } \ : \ 1 \leq t \leq L , t = 1 \mathrm { o r } \kappa _ { t } \neq \kappa _ { t - 1 } \rangle .\tag{6}
$$

The sequence $S _ { f p } ( u )$ is the discrete fingerprint of AMU u. Its unique-code set $U _ { f p } ( u ) = \{ a :$ $a \in S _ { f p } ( u ) \}$ is used for topology construction, so repeated code ids are counted once per AMU when building co-occurrence statistics. Quantization does not replace the stored AMU text: the original $c _ { f a c t }$ remains the evidence presented to the answer generator, while $S _ { f p } ( u )$ serves only as an auxiliary retrieval key.

## 3.4 Topology-Aware Dual-Path Retrieval

Implicit topology & associative inference. The discrete fingerprints define an implicit topology through codebook co-occurrence, complementing the geometric locality of dense vectors. Let ${ \mathcal { F } } =$ $\textstyle \bigcup _ { k = 1 } ^ { K _ { \mathrm { t u r n } } } { \mathcal { F } } _ { k }$ be the global AMU store and $N _ { u } = | \mathcal { F } |$ . We keep a global matrix M over code ids:

$$
\mathcal { M } _ { a b } = \sum _ { u \in \mathcal { F } } \mathbf { 1 } [ a \in U _ { f p } ( u ) ] \mathbf { 1 } [ b \in U _ { f p } ( u ) ] \mathbf { 1 } [ a \neq b ] , \qquad n ( a ) = \sum _ { u \in \mathcal { F } } \mathbf { 1 } [ a \in U _ { f p } ( u ) ] .\tag{7}
$$

Thus $\mathcal { M } _ { a b }$ counts how often distinct ids a and b co-occur in the same AMU fingerprint, while $n ( a )$ counts how many AMUs contain source id $^ { a ; }$ diagonal self-transitions are masked [29]. The one-hop transition probability is:

$$
P ( b \mid a ) = \frac { \mathcal { M } _ { a b } } { \sum _ { b ^ { \prime } \neq a } \mathcal { M } _ { a b ^ { \prime } } + \epsilon } .\tag{8}
$$

Here, ϵ is a small smoothing constant for codes with no outgoing co-occurrence. During inference, the query is projected and deduplicated by the same ISQ module, giving query code set $Q _ { f p }$ . Treating ${ \bar { P } } ( b \mid a )$ as the entry $P _ { a b }$ of the code-to-code transition matrix, MemCode scores activation from a source id $a \in Q _ { f p }$ to an associated target id b across up to $H _ { r }$ topology hops:

$$
S _ { \mathrm { c o d e } } ( b \mid a ) = \underbrace { \log \left( \frac { N _ { u } + 1 } { n ( a ) + 1 } \right) } _ { \mathrm { s o u r c e ~ I D F } } \cdot \operatorname* { m a x } _ { 1 \leq h \leq H _ { r } } \underbrace { [ P ^ { h } ] _ { a b } } _ { h \mathrm { - h o p ~ t r a n s i t i o n } } \cdot \alpha ^ { h }\tag{9}
$$

where $\alpha \in ( 0 , 1 )$ controls decay during neighbor expansion and $H _ { r }$ is the fixed retrieval expansion depth. We use $\dot { H } _ { r } = 1$ in all experiments, preventing recursive topology propagation from amplifying noisy associations. The source-side IDF downweights frequent query anchors [30], while the transition term activates codes reached by repeated co-occurrence.

Hybrid evidence retrieval & context reconstruction. The final retrieval stage merges the two evidence paths through structured supplementation. The dense path (Path A) retrieves $K _ { v }$ atomic facts by nearest-neighbor search over their dense vectors, forming $\mathcal { R } _ { v e c } .$ The codebook path (Path B) scores each AMU by the best code in its fingerprint:

$$
S _ { \mathrm { a m u } } ( u ) = \operatorname* { m a x } _ { \stackrel { a \in Q _ { f p } } { b \in U _ { f p } ( u ) } } S _ { \mathrm { c o d e } } ( b \mid a ) .\tag{10}
$$

It then selects $K _ { c }$ supplementary facts from ${ \mathcal F } ,$ excluding Path A memories:

$$
\begin{array} { r l } & { \mathcal { R } _ { t o p o } = \mathrm { T o p - K } _ { c } ( \mathcal { F } \setminus \mathcal { R } _ { v e c } \mathrm { ~ ; ~ r a n k e d ~ b y ~ } S _ { \mathrm { a m u } } ( u ) \downarrow ) , } \\ & { \mathcal { R } _ { h y b r i d } = \mathcal { R } _ { v e c } \cup \mathcal { R } _ { t o p o } . } \end{array}\tag{11}
$$

Here, $K _ { v }$ and $K _ { c }$ are the fixed dense-path and codebook-path evidence budgets, and ↓ denotes descending order by score. Greedy topology selection adds evidence associated in codebook space but absent from dense candidates. MemCode then follows $\Lambda _ { s r c }$ back to original spans and orders them chronologically. The merge is intentionally asymmetric: Path A provides high-precision neighbors, while Path B only supplements it. This keeps popular associations from overwhelming the answer context and makes the ablation interpretable under a fixed $K _ { v } + K _ { c }$ evidence budget. In Appendix, Figures 5–7 further show learned codes form a concentrated but non-collapsed topology, supporting asymmetric use of Path B.

## 3.5 Asynchronous Memory Consolidation

Online atomic writes can accumulate duplicate facts, outdated states, and retrieval interference. MemCode handles this with an asynchronous consolidation engine outside the online path. It backfills dense vectors, filters invalid entries, and applies hierarchical agglomerative clustering (HAC) based on cosine distance [31]:

$$
\mathcal { G } = \mathrm { H A C } \bigg ( \mathcal { F } ; ~ d ( u _ { i } , u _ { j } ) = 1 - \frac { \mathbf { v } _ { i } \cdot \mathbf { v } _ { j } } { \| \mathbf { v } _ { i } \| \| \mathbf { v } _ { j } \| } , ~ d _ { \operatorname* { m a x } } = 1 - \tau \bigg ) .\tag{12}
$$

Here, $\mathbf { v } _ { i }$ is the dense vector of AMU $u _ { i }$ produced by the same encoder used by Path A, τ is the configured minimum cosine similarity for grouping, and G is the candidate cluster set. Large clusters are split by time. Before entering the LLM, a gate skips count-sensitive clusters, restricts updatesensitive clusters, and avoids cross-speaker mixing. Passing clusters are assigned to AGGREGATE, MERGE, or UPDATE, with fewer new facts than archived facts. After write-back, Pointer Forwarding marks old facts as archived and records their new fact indices, forming a two-level evidence chain with the original $\Lambda _ { s r c }$ . At retrieval, old facts are replaced by new ones. If the temporal span of a cluster $g \in { \mathcal { G } }$ exceeds the minute level, the new fact is prefixed with [Time Span: $T _ { s t a r t }$ to $T _ { e n d } ]$ , where $\begin{array} { r } { T _ { s t a r t } = \operatorname* { m i n } _ { u \in g } t _ { a b s } ( u ) } \end{array}$ and $T _ { e n d } = \operatorname* { m a x } _ { u \in g } t _ { a b s } ( u )$ . The archive policy combines consolidation mode, number of parent facts, time span, and speaker source to decide whether to replace or retain a fact, reducing the risk of irreversible information loss.

## 4 Experiments

## 4.1 Experimental Setting

Datasets and protocol. We evaluate long-term agent memory on LoCoMo [32] and LongMemEval S [33]. LongMemEval-S stresses dynamic multi-session retrieval and updating, while LoCoMo covers multi-hop, temporal, open-domain, and single-hop questions. We use an incremental protocol that mimics deployment: the system ingests conversations turn by turn in chronological order and cannot access the full history in advance. Baselines and implementation. Baselines include Full Text, Naive RAG, LangMem, A-MEM, MemoryOS, Mem0, and LightMem; LangMem is reported on LongMemEval-S and omitted from the LoCoMo table. Full Text provides the controlled full-history self-attention alternative under the same answer backbone. All methods use GPT-4omini [34] and Qwen3-30B-A3B-Instruct-2507 [35]. MemCode’s discrete representation uses a VQ-VAE initialized from T5-base [36] and trained on DailyDialog with a single 32GB GPU; its continuous representation uses 768-dimensional bge-base-en-v1.5 embeddings [37]. We use only the official DailyDialog training split (11,118 dialogues; 64,935 context–response pairs) to learn reusable local conversational patterns. The codebook is frozen before memory evaluation; no benchmark histories, questions, answers, labels, or retrieval traces enter its optimization. Online ingestion and consolidation are constrained to a consumer RTX 3050 Ti GPU. Metrics. For task performance, we follow the common LLM-as-a-Judge protocol [38]: GPT-4o-mini compares each output with the reference answer or key topic points, and we report overall and category accuracy. For efficiency, we report average runtime, Calls, and Total Tokens for memory construction only; after retrieval, all methods use a single LLM generation call, while vector retrieval and codebook inverted lookup are computationally negligible and depend little on external API conditions. In long-term agent settings, the bottleneck is therefore the construction of structured historical memory rather than millisecondlevel differences in a single query. Appendix Tables 5–7 report the compute environments, offline VQ-VAE training settings, and dataset-specific retrieval budgets; Appendix F reports the prompts.

Table 1: Accuracy and memory-construction cost on LongMemEval-S. We report Overall/category ACC (↑), runtime per dialogue (↓), Total Tokens, and Calls. Asyn. denotes asynchronous consolidation. Bold and underline denote the best and 2<sup>nd</sup>-best results within each backbone.
<table><tr><td>Method</td><td>Runtime ↓ (s)</td><td>Total Tokens ↓ (k)</td><td>Calls ↓</td><td>Overall ↑ ACC (%)</td><td>Temporal (n = 133)</td><td>Multi-Session (n = 133)</td><td>Know.-Upd. (n = 78)</td><td>Single-User (n = 70)</td><td>Single-Asst. (n = 56)</td><td>Single-Pref. (n = 30)</td></tr><tr><td colspan="9">S GPT-4o-mini</td><td></td></tr><tr><td>Full Text</td><td></td><td></td><td></td><td>56.80</td><td>31.58</td><td>45.45</td><td>76.92</td><td>87.14</td><td>89.29</td><td>36.67</td></tr><tr><td>Naive RAG</td><td></td><td></td><td></td><td>61.00</td><td>39.85</td><td>48.48</td><td>67.95</td><td>90.00</td><td>98.21</td><td>53.33</td></tr><tr><td>LangMem</td><td>2293.70</td><td>1102.16</td><td>520.62</td><td>37.20</td><td>15.79</td><td>20.30</td><td>66.67</td><td>60.00</td><td>46.43</td><td>60.00</td></tr><tr><td>A-MEM</td><td>5132.06</td><td>1605.81</td><td>986.55</td><td>62.60</td><td>47.36</td><td>48.87</td><td>64.11</td><td>92.86</td><td>96.43</td><td>46.67</td></tr><tr><td>MemoryOS</td><td>8030.04</td><td>2991.75</td><td>2938.41</td><td>44.80</td><td>32.33</td><td>31.06</td><td>48.72</td><td>80.00</td><td>64.29</td><td>30.00</td></tr><tr><td>Mem0</td><td>4248.49</td><td>1152.62</td><td>811.57</td><td>53.61</td><td>40.15</td><td>46.21</td><td>70.12</td><td>81.43</td><td>41.07</td><td>60.00</td></tr><tr><td>LightMem</td><td>283.76</td><td>28.25</td><td>18.43</td><td>68.64</td><td>67.18</td><td>71.74</td><td>83.12</td><td>87.14</td><td>32.14</td><td>68.18</td></tr><tr><td>MemCode MemCode + Asyn.</td><td>74.05</td><td>318.41</td><td>248.84</td><td>77.20</td><td>69.92</td><td>65.41</td><td>82.05</td><td>92.86</td><td>92.86</td><td>83.33</td></tr><tr><td></td><td>82.98</td><td>328.05</td><td>274.37</td><td>78.40</td><td>67.67</td><td>69.17</td><td>83.33</td><td>95.71</td><td>91.07</td><td>90.00</td></tr><tr><td colspan="9">立 Qwen3-30B-A3B-Instruct-2507</td></tr><tr><td></td><td colspan="3"></td><td>54.80</td><td>33.08</td><td>35.61</td><td>76.92</td><td>82.86</td><td>87.50</td><td>50.00</td></tr><tr><td>Full Text Naive RAG</td><td></td><td></td><td></td><td>60.80</td><td>36.84</td><td>47.73</td><td>65.38</td><td>91.43</td><td>98.21</td><td>70.00</td></tr><tr><td>LangMem</td><td>3237.16</td><td>1430.02</td><td>495.12</td><td>50.80</td><td>37.60</td><td>38.35</td><td>67.95</td><td>78.57</td><td>42.86</td><td>70.00</td></tr><tr><td>A-MEM</td><td>5367.51</td><td>1864.93</td><td>989.30</td><td>65.20</td><td>51.88</td><td>51.12</td><td>76.93</td><td>90.00</td><td>96.43</td><td>40.00</td></tr><tr><td>MemoryOS</td><td>8721.78</td><td>2944.97</td><td>2922.28</td><td>49.60</td><td>28.57</td><td>36.84</td><td>61.54</td><td>72.86</td><td>92.86</td><td>33.33</td></tr><tr><td>Mem0</td><td>2239.94</td><td>1001.90</td><td>722.76</td><td>39.51</td><td>41.94</td><td>28.13</td><td>28.57</td><td>55.32</td><td>26.09</td><td>81.82</td></tr><tr><td>LightMem</td><td>357.13</td><td>25.15</td><td>16.54</td><td>61.95</td><td>54.20</td><td>51.91</td><td>66.67</td><td>80.00</td><td>31.25</td><td>80.00</td></tr><tr><td>MemCode</td><td>78.05</td><td>356.41</td><td>248.84</td><td>73.80</td><td>57.89</td><td>64.66</td><td>78.21</td><td>94.29</td><td>92.86</td><td>90.00</td></tr><tr><td>MemCode + Asyn.</td><td>95.02</td><td>363.59</td><td>265.94</td><td>74.60</td><td>60.90</td><td>66.17</td><td>78.21</td><td>91.43</td><td>92.86</td><td>90.00</td></tr></table>

## 4.2 Main Results

Tables 1 and 2 show that MemCode improves the accuracy–efficiency trade-off across both backbones and benchmarks. It achieves the highest overall accuracy while reducing construction runtime to roughly 1/21–1/3 of LightMem’s runtime and 1–2 orders of magnitude below A-MEM and MemoryOS. The category breakdown shows where the gains arise: atomic writing improves precise localization and speaker-aware facts, the codebook path supports structured temporal and compositional retrieval, and consolidation partially restores long-range continuity lost by atomic segmentation. Appendix Tables 10–12 further show that backbone choice changes the memory store itself: Table 11 shows that Qwen3 writes denser AMU/codebook traces, while Tables 10 and 12 connect this density to better open-ended coverage but more distractors for exact temporal and multi-hop evidence. LongMemEval-S. This benchmark contains roughly 30–40 sessions and about 115K tokens, with an emphasis on retrieval and updating in dynamic memory. Under GPT-4o-mini, Mem-Code reaches 77.20% Overall, 8.56 points above LightMem; asynchronous consolidation raises it to 78.40%. The same trend appears under Qwen3 (73.80% → 74.60%). Category results clarify the source of the gains. Absolute timestamps turn some temporal questions into constrained singlehop retrieval, which helps Temporal and Single-User. LightMem remains stronger on Multi-Session (71.74% vs. 65.41%), where paragraph compression preserves cross-session continuity; AGGRE-GATE partially recovers it to 69.17%. MemCode is much stronger on Single-Assistant (92.86% vs. 32.14%), indicating the value of speaker binding.

In efficiency, MemCode takes about 74 seconds per dialogue and about 83 seconds with consolidation. Its Calls and Tokens exceed LightMem’s, but the runtime is lower because atomic extraction is parallel, whereas batch compression is sequential. Tokens are closer to API cost; Runtime is closer to deployment latency. LoCoMo. LoCoMo averages 304.9 dialogue turns and 9209.2 tokens. MemCode reaches 78.12%/76.36% Overall under GPT-4o-mini/Qwen3, and asynchronous consolidation improves this to 79.55%/76.88%. Single tasks improve over LightMem (87.04% vs. 77.41%), supporting atomic facts and spatiotemporal resolution in single-hop localization. On

Table 2: Category-level accuracy and memory-construction cost on LoCoMo. Asyn. denotes asynchronous consolidation. Bold and underline denote the best and $2 ^ { n d }$ -best results on each backbone.
<table><tr><td>Method</td><td>Runtime ↓ (s)</td><td>Total Tokens ↓ (k)</td><td>Calls ↓</td><td>Overall ↑ (ACC %)</td><td>Multi  $( n = 2 8 2 )$ </td><td>Open  $( n \doteq 9 6 )$ </td><td>Single  $( n = \mathsf { 8 4 1 } )$ </td><td>Temp  $( n = 3 2 1 )$ </td></tr><tr><td colspan="9">S GPT-4o-mini</td></tr><tr><td>Full Text</td><td></td><td></td><td></td><td>73.83</td><td>68.79</td><td>56.25</td><td>86.56</td><td>50.16</td></tr><tr><td>Naive RAG</td><td></td><td></td><td></td><td>63.64</td><td>55.32</td><td>47.92</td><td>70.99</td><td>56.39</td></tr><tr><td>A-MEM</td><td>6060.73</td><td>1149.43</td><td>1175.47</td><td>64.16</td><td>56.03</td><td>31.25</td><td>72.06</td><td>60.44</td></tr><tr><td>MemoryOS (eval)</td><td>2422.05</td><td>287.00</td><td>553.45</td><td>58.25</td><td>56.74</td><td>45.83</td><td>67.06</td><td>40.19</td></tr><tr><td>MemoryOS (pypi)</td><td>3332.59</td><td>526.48</td><td>1016.06</td><td>54.87</td><td>52.13</td><td>43.75</td><td>63.97</td><td>36.76</td></tr><tr><td>Mem0</td><td>4432.87</td><td>1693.39</td><td>1602.20</td><td>61.69</td><td>56.38</td><td>43.75</td><td>66.47</td><td>59.19</td></tr><tr><td>LightMem</td><td>848.49</td><td>99.76</td><td>41.65</td><td>71.95</td><td>62.41</td><td>44.79</td><td>77.41</td><td>74.14</td></tr><tr><td>MemCode</td><td>40.70</td><td>189.32</td><td>287.10</td><td>78.12</td><td>64.89</td><td>43.75</td><td>87.04</td><td>76.64</td></tr><tr><td> $M e m C o d e + A s y n .$ </td><td>109.40</td><td>266.59</td><td>439.60</td><td>79.55</td><td>69.15</td><td>51.04</td><td>87.87</td><td>75.39</td></tr><tr><td colspan="9">Qwen3-30B-A3B-Instruct-2507</td></tr><tr><td></td><td></td><td>一</td><td></td><td>74.87</td><td>69.86</td><td>57.29</td><td>87.40</td><td>51.71</td></tr><tr><td>Full Text Naive RAG</td><td></td><td></td><td></td><td>66.95</td><td>62.41</td><td>57.29</td><td>76.81</td><td>47.98</td></tr><tr><td>A-MEM</td><td>5543.90</td><td>1626.80</td><td>1175.40</td><td>56.10</td><td>57.45</td><td>43.75</td><td>67.90</td><td>27.73</td></tr><tr><td>MemoryOS (eval)</td><td>1269.70</td><td>361.51</td><td>414.70</td><td>61.04</td><td>62.77</td><td>51.04</td><td>72.29</td><td>33.02</td></tr><tr><td>MemoryOS (pypi)</td><td>1982.20</td><td>666.35</td><td>1004.60</td><td>51.30</td><td>52.48</td><td>40.62</td><td>61.59</td><td>26.48</td></tr><tr><td>Mem0</td><td>4540.70</td><td>1799.40</td><td>1614.50</td><td>43.31</td><td>42.91</td><td>46.88</td><td>46.37</td><td>34.58</td></tr><tr><td>LightMem</td><td>815.70</td><td>99.87</td><td>29.10</td><td>71.36</td><td>70.57</td><td>60.42</td><td>79.19</td><td>54.83</td></tr><tr><td>MemCode</td><td>41.14</td><td>206.10</td><td>287.10</td><td>76.36</td><td>59.22</td><td>52.08</td><td>88.11</td><td>67.91</td></tr><tr><td> $M e m C o d e + A s y n .$ </td><td>129.69</td><td>339.81</td><td>517.50</td><td>76.88</td><td>66.31</td><td>48.96</td><td>88.59</td><td>63.86</td></tr></table>

Multi tasks, codebook topology under GPT-4o-mini exceeds LightMem and Naive RAG (64.89% vs. 62.41%/55.32%), consistent with discrete retrieval recovering implicit cross-segment dependencies [39]. Full Text and LightMem remain competitive on Multi as a complete or compressed narrative context preserves logical connections; retrieval methods remain limited by recall completeness.

Open tasks expose a boundary of abstraction: mental states, implicit emotions, and motivations are difficult to fully propositionalize, so Full Text retains an advantage. Consolidation raises Open to 51.04%, but the synthesized material still comes from extracted facts, leaving limited headroom. In efficiency, MemCode needs only 40.70 seconds on LoCoMo, about one twentieth of LightMem’s runtime; with consolidation, 109.40 seconds remains below all baselines, so the offline stage retain the same parallelism advantage. Boundary of consolidation. Consolidation synthesizes fragmented facts into denser memories, improving LoCoMo Multi by 4.26 and 7.09 points. UPDATE also reduces interference by archiving outdated states. The trade-off is temporal granularity: multiple time points may be compressed into one entry, causing a slight Temporal drop. Since extraction already normalizes absolute time, temporal results after consolidation still outperform or match most baselines. Qwen3 shows larger Multi gains but also a clearer Temporal drop, indicating that different backbones use compressed memories differently. This frames consolidation as an exchange between denser global memory and finer temporal resolution, not a uniformly monotonic gain. Appendix Table 9 shows that the LoCoMo GPT-4o-mini result is stable across repeated runs.

Taken together, the two benchmarks separate two regimes. When answers are grounded in explicit entities, dates, speaker bindings, or linked facts, MemCode’s AMUs and discrete topology outperform raw retrieval and compressed-memory baselines. When answers depend on implicit affect or broad narrative continuity, Full Text and LightMem remain competitive; consolidation narrows this gap by restoring continuity at a small temporal-granularity cost. The ablation below tests the retrieval path directly, while the appendix adds repeated-run stability (Table 9), dense-embedding sensitivity (Table 15), retrieval novelty (Figure 4), and codebook-space structure (Figures 5–7).

## 4.3 Ablation Study

We run ablations on LoCoMo with GPT-4o-mini. Figure 3(a) shows that doubling vector Top-K adds only 1.4%, whereas dual-path retrieval improves Multi-Hop by 5.3 points. The limiting factor is therefore not candidate count, but continuous neighborhoods missing logical associations in the discrete topology. Figure 3(b) shows that coarser stride hurts Temporal and Multi-Hop while barely affecting Single-Hop, confirming that timestamps and entity IDs are core memory tokens rather than implementation details.

Figure 3(c) shows that removing atomic-fact abstraction drops Overall to 53.8% and Multi-Hop by 31.2%, indicating that context-dependent writes are weak retrieved evidence. Removing raw context costs only 3.2%, meaning that AMUs carry the main retrievable semantics while source reconstruction mainly improves evidence completeness. This confirms that the quality of the write unit is the primary driver of retrieval success, and highlights the importance of precise temporal and speaker grounding. Figure 3(d) shows that topology expansion provides long-range activation, while IDF suppresses high-frequency anchors; removing either component reduces Multi-Hop, and the full model benefits from combining recall expansion with noise control. Together, these results illustrate how structured retrieval paths complement dense embeddings to handle compositional and temporally distributed evidence.

![](images/b1a3fc50dc41c49227815354a415aef2a3dd544e6a79d0818849a7a14ead95b3.jpg)

![](images/f8052205996dee0a3e2bf80dfde4f48651bfbece0b77866eac9c037e81bb386d.jpg)

![](images/e61a84ddc1fade2d2ce6d70b37b53cf0f3a81cd4d520783b1056a40a70d0a737.jpg)

![](images/9d2ab878b4102b812ab6039e1444a76741a67d3e9c1db93d77503b59485a93c8.jpg)  
Figure 3: Ablation on LoCoMo (GPT-4o-mini): retrieval path, stride, ingestion/context reconstruction, and topology/IDF weighting.

Table 3 separates the codebook path from consolidation under a fixed representation and retrieval budget. The codebook path adds 5.32 points without consolidation and 3.90 points with it; consolidation adds 5.67 points for dense-only retrieval and 4.26 points for dual-path retrieval. The −1.42-point interaction indicates partial overlap, while each component retains a positive marginal contribution.

These ablations rule out a simple retrieval-budget explanation. More vector candidates mostly add near-duplicates from the same local embedding neighborhood, while the codebook path changes the access relation by following repeated co-occurrence among discrete anchors. The stride result gives the same message from the representation side: textual memory is dense in names, dates, quantities, and speaker states, so coarse compression removes exactly the tokens that later make temporal and compositional questions answerable. MemCode therefore gains by aligning the write unit, discrete index, and evidence reconstruction path rather than by tuning one isolated retriever. This indicates that future improvements in long-term memory should focus on enhancing the semantic structure of the stored units and the associative topology rather than merely increasing candidate counts or vector dimensions.

Table 4(a) separates retrieval structure from evidence volume. Increasing dense retrieval from 20 to 40 adds 1.43 points, while BM25, dense–sparse fusion, and HyDE remain below MemCode under the shared maximum budget.

Table 3: Component attribution on LoCoMo Multi-Hop (GPT-4o-mini, n = 282). All settings use the same base AMU store and a total evidence budget of 40. Counts of correct answers are shown in parentheses.
<table><tr><td>Retrieval path</td><td>Without consolidation</td><td>With consolidation</td></tr><tr><td>Dense only</td><td>59.57 (168)</td><td>65.25 (184)</td></tr><tr><td>Dense + codebook</td><td>64.89 (183)</td><td>69.15 (195)</td></tr></table>

Table 4: Fixed-budget retrieval controls on LoCoMo (GPT-4o-mini). Panel (a) uses the same frozen 5,911-AMU store and at most 40 evidence items. Panel (b) isolates the second path under a 20+20 budget; ∆ is codebook minus BM25, C/B-only counts paired wins, and BM25 inj. reports additional base facts/time buckets beyond the vector stage.  
(a) Overall same-store controls
<table><tr><td>Method</td><td>Evidence budget</td><td>Overall accuracy (%)</td></tr><tr><td>Dense-20</td><td>20</td><td>73.18</td></tr><tr><td>Dense-40</td><td>40</td><td>74.61</td></tr><tr><td>Dense-20 + BM25-20</td><td>20+20</td><td>76.75</td></tr><tr><td>Dense-sparse hybrid + reranker</td><td>40</td><td>76.36</td></tr><tr><td>HyDE + reranker [40]</td><td>40</td><td>76.69</td></tr><tr><td>Dense-20 + codebook-20 (MemCode)</td><td>20+20</td><td>78.12</td></tr></table>

<table><tr><td colspan="7">(b) Codebook versus BM25 by question category</td></tr><tr><td>Category</td><td>n</td><td>Codebook</td><td>BM25</td><td>∆</td><td>C-only</td><td>B-only</td><td>BM25 inj.</td></tr><tr><td>Overall</td><td>1540</td><td>78.12</td><td>76.75</td><td>+1.36</td><td>105</td><td>84</td><td>18.58/10.96</td></tr><tr><td>Multi-Hop</td><td>282</td><td>64.89</td><td>60.64</td><td>+4.26</td><td>33</td><td>21</td><td>18.50/10.65</td></tr><tr><td>Temporal</td><td>321</td><td>76.64</td><td>70.41</td><td>+6.23</td><td>35</td><td>15</td><td>18.66/11.34</td></tr><tr><td>Open</td><td>96</td><td>43.75</td><td>50.00</td><td>-6.25</td><td>5</td><td>11</td><td>18.59/10.78</td></tr><tr><td>Single-Hop</td><td>841</td><td>87.04</td><td>87.63</td><td>-0.59</td><td>32</td><td>37</td><td>18.58/10.93</td></tr></table>

Panel (b) then changes only the second path. BM25 injects many additional facts and time buckets, but it does not reproduce the codebook gain on Temporal and Multi-Hop questions. Sparse lexical expansion helps when surface terms already point to relevant context, whereas the codebook path helps when evidence must be joined across time, entities, and repeated discrete anchors. Appendix Tables 13 and 14 provide the trace-level category and explicit-graph cost details.

## 5 Conclusion

Continuous nearest-neighbor retrieval leaves systematic gaps in structured long-term memory queries. Our proposed MemCode keeps the RAG-style paradigm but couples atomic abstraction with semantic quantization, topology-aware dual-path retrieval, and asynchronous consolidation. This gives facts stable, discrete anchors and exposes associations beyond vector nearest neighbors. Across LongMemEval-S and LoCoMo, MemCode achieves the best overall accuracy under two backbones while preserving construction efficiency. The component and BM25 replacement ablations further clarify the mechanism: extra lexical candidates or larger vector budgets do not recover the same temporal and multi-hop gains unless retrieval follows the discrete co-occurrence structure written during memory construction. This keeps the claim narrow: the advantage comes from matching memory writing, indexing, and reconstruction, not from increasing context volume at query time. The remaining gaps in open questions reflect a clear limitation of AMU-based abstraction: MemCode preserves explicit entities, times, speakers, and factual links well, but it may mis implicit motivations, emotions, and pragmatic cues that are better preserved in the raw dialogue.

## Acknowledgments

This research is supported by the National Key Research and Development Program of China (Grant No.2025CSJGG01800/YDZX20263100004012), the National Natural Science Foundation of China (No. 62406225), the Shanghai Municipal Commission of Economy and Informatization under the AI for Science ”Hundred Teams, Hundred Projects” Program (No. 2025-GZL-RGZN-BTBX-02020), and the Fundamental Research Funds for the Central Universities.

## References

[1] Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran Wang, Changhao Jiang, Yicheng Zou, Xiangyang Liu, Zhangyue Yin, Shihan Dou, Rongxiang Weng, Wensen Cheng, Qi Zhang, Wenjuan Qin, Yongyan Zheng, Xipeng Qiu, Xuanjing Huang, and Tao Gui. The rise and potential of large language model based agents: A survey. Science China Information Sciences, 68(2):121101, 2025. doi: 10. 1007/s11432-024-4222-0. URL https://doi.org/10.1007/s11432-024-4222-0.

[2] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Ji-Rong Wen. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18 (6):186345, 2024. doi: 10.1007/s11704-024-40231-1. URL https://doi.org/10.1007/ s11704-024-40231-1.

[3] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS), 2017. URL https://arxiv.org/abs/1706.03762.

[4] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12:157–173, 2024. doi: 10.1162/tacl a 00638. URL https://direct.mit.edu/tacl/article/doi/10.1162/tacl\_a\_00638/ 119630/Lost-in-the-Middle-How-Language-Models-Use-Long.

[5] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨ aschel, Sebastian Riedel, and¨ Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems (NeurIPS), 2020. URL https://arxiv. org/abs/2005.11401.

[6] Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models for open domain question answering. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics (EACL), pages 874–880, 2021. doi: 10.18653/v1/2021.eacl-main.74. URL https://aclanthology.org/2021.eacl-main. 74/.

[7] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready AI agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025. URL https://arxiv.org/abs/2504.19413.

[8] LangChain. LangMem. https://langchain-ai.github.io/langmem/, 2025. Software documentation and GitHub repository; accessed 2026-04-25.

[9] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-MEM: Agentic memory for LLM agents. In Advances in Neural Information Processing Systems (NeurIPS), 2025. URL https://openreview.net/forum?id=FiM0M8gcct.

[10] Jiazheng Kang, Mingming Ji, Zhe Zhao, and Ting Bai. Memory OS of AI agent. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 25961–25970, 2025. doi: 10.18653/v1/2025.emnlp-main.1318. URL https: //aclanthology.org/2025.emnlp-main.1318/.

[11] Jizhan Fang, Xinle Deng, Haoming Xu, Ziyan Jiang, Yuqi Tang, Ziwen Xu, Shumin Deng, Yunzhi Yao, Mengru Wang, Shuofei Qiao, Huajun Chen, and Ningyu Zhang. Light Mem: Lightweight and efficient memory-augmented generation. In International Conference on Learning Representations (ICLR), 2026. URL https://openreview.net/forum?id= dyJ0GWpjJB.

[12] Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language

Processing (EMNLP-IJCNLP), pages 3982–3992, 2019. doi: 10.18653/v1/D19-1410. URL https://aclanthology.org/D19-1410/.

[13] Nandan Thakur, Nils Reimers, Andreas Ruckl¨ e, Abhishek Srivastava, and Iryna Gurevych.´ BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Proceedings ofthe Neural Information Processing Systems Track on Datasets and Benchmarks, 2021. URL https://openreview.net/forum?id=wCu6T5xFjeJ.

[14] Michael A. Yassa and Craig E. L. Stark. Pattern separation in the hippocampus. Trends in Neurosciences, 34(10):515–525, 2011. doi: 10.1016/j.tins.2011.06.006.

[15] Susumu Tonegawa, Xu Liu, Steve Ramirez, and Roger Redondo. Memory engram cells have come of age. Neuron, 87(5):918–931, 2015. doi: 10.1016/j.neuron.2015.08.002.

[16] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov,˘ Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769–6781, 2020. doi: 10.18653/v1/2020.emnlp-main.550. URL https: //aclanthology.org/2020.emnlp-main.550/.

[17] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2023. URL https://arxiv.org/abs/2312. 10997.

[18] Yury Zemlyanskiy, Michiel de Jong, Luke Vilnis, Santiago Ontan˜on, William Cohen, Sumit´ Sanghai, and Joshua Ainslie. MEMORY-VQ: Compression for tractable Internet-scale memory. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pages 737–744, Mexico City, Mexico, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-short.64. URL https://aclanthology.org/2024. naacl-short.64/.

[19] Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. MemGPT: Towards LLMs as operating systems. arXiv preprint arXiv:2310.08560, 2023. doi: 10.48550/arXiv.2310.08560. URL https://arxiv.org/abs/ 2310.08560.

[20] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representa-¨ tion learning. In Advances in Neural Information Processing Systems (NeurIPS), 2017. URL https://arxiv.org/abs/1711.00937.

[21] Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. SoundStream: An end-to-end neural audio codec. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 30:495–507, 2022. doi: 10.1109/TASLP.2021.3129994. URL https://doi.org/10.1109/TASLP.2021.3129994.

[22] Tibor Kiss and Jan Strunk. Unsupervised multilingual sentence boundary detection. Computational Linguistics, 32(4):485–525, 2006. doi: 10.1162/coli.2006.32.4.485.

[23] Kenton Lee, Luheng He, Mike Lewis, and Luke Zettlemoyer. End-to-end neural coreference resolution. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 188–197, 2017. doi: 10.18653/v1/D17-1018. URL https:// aclanthology.org/D17-1018/.

[24] Alexandre Defossez, Jade Copet, Gabriel Synnaeve, and Yossi Adi. High fidelity neural au-´ dio compression. arXiv preprint arXiv:2210.13438, 2022. URL https://arxiv.org/abs/ 2210.13438.

[25] Yanran Li, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu. DailyDialog: A manually labelled multi-turn dialogue dataset. In Proceedings of the 8th International Joint Conference on Natural Language Processing (IJCNLP), pages 986–995, 2017. URL https: //aclanthology.org/I17-1099/.

[26] Yoshua Bengio, Nicholas Leonard, and Aaron Courville. Estimating or propagating gradi-´ ents through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013. URL https://arxiv.org/abs/1308.3432.

[27] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. URL https://arxiv.org/abs/ 1807.03748.

[28] Ali Razavi, Aaron van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images¨ with VQ-VAE-2. In Advances in Neural Information Processing Systems (NeurIPS), 2019. URL https://arxiv.org/abs/1906.00446.

[29] Omer Levy and Yoav Goldberg. Neural word embedding as implicit matrix factorization. In Advances in Neural Information Processing Systems (NeurIPS), 2014. URL https://proceedings.neurips.cc/paper/ 5477-neural-word-embedding-as-implicit-matrix-factorization.pdf.

[30] Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4):333–389, 2009. doi: 10.1561/ 1500000019. URL https://www.nowpublishers.com/article/Details/INR-019.

[31] Joe H. Ward, Jr. Hierarchical grouping to optimize an objective function. Journal of the American Statistical Association, 58(301):236–244, 1963. doi: 10.1080/01621459.1963.10500845.

[32] Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of LLM agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13851–13870, 2024. doi: 10.18653/v1/2024.acl-long.747. URL https://aclanthology.org/2024.acl-long.747/.

[33] Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. Long-MemEval: Benchmarking chat assistants on long-term interactive memory. In International Conference on Learning Representations (ICLR), 2025. URL https://openreview.net/ forum?id=pZiyCaVuti.

[34] OpenAI. GPT-4o mini: Advancing cost-efficient intelligence. https://openai.com/ index/gpt-4o-mini-advancing-cost-efficient-intelligence/, 2024. Official release; accessed 2026-04-25.

[35] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. doi: 10.48550/arXiv.2505.09388. URL https://arxiv.org/abs/2505.09388.

[36] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020. URL https://jmlr.org/papers/v21/20-074.html.

[37] Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. C-pack: Packed resources for general chinese embeddings. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), pages 641–649, 2024. doi: 10.1145/3626772.3657878. URL https://arxiv.org/ abs/2309.07597.

[38] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-bench and chatbot arena. In Advances in Neural Information Processing Systems (NeurIPS), 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash/ 91f18a1287b398d378ef22505bf41832-Abstract-Datasets\_and\_Benchmarks.html.

[39] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. MuSiQue: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 10:539–554, 2022. doi: 10.1162/tacl a 00475. URL https://direct.mit.edu/tacl/article/doi/10.1162/tacl\_a\_00475/ 110996/MuSiQue-Multihop-Questions-via-Single-hop-Question.

[40] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. Precise zero-shot dense retrieval without relevance labels. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1762–1777. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.acl-long.99. URL https://aclanthology.org/2023.acl-long.99/.

[41] Bernal Jimenez Guti´ errez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From RAG to´ memory: Non-parametric continual learning for large language models. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 21497–21515. PMLR, 2025. URL https://proceedings.mlr. press/v267/gutierrez25a.html.

[42] Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. β-VAE: Learning basic visual concepts with a constrained variational framework. In International Conference on Learning Representations (ICLR), 2017. URL https://openreview.net/forum?id=Sy2fzU9gl.

## A Limitations

MemCode targets retrieval-intensive long-term conversational memory, not fully deployed agents with planning, tool use, and interactive adaptation. Its LLM-based AMU writing can miss or distort implicit motivations, emotions, and pragmatic cues; these cases explain why full-context methods remain competitive on LoCoMo Open and bound our claims to structured memory retrieval. Our HippoRAG2 comparison also measures one-shot graph construction over complete LoCoMo histo ries; repeated graph maintenance in a growing online memory is outside the present evaluation.

## B Implementation Settings

MemCode uses a lightweight local memory stack. Dialogue and atomic-memory metadata are stored in SQLite, including raw utterances, extracted AMUs, codebook IDs, and code-to-AMU links. Dense retrieval uses a FAISS L2 index over 768-dimensional bge-base-en-v1.5 embeddings [37], while the discrete path uses a 1024-entry VQ codebook and maintains code frequency and code cooccurrence statistics in the metadata store. The online stage writes AMUs asynchronously, assigns both dense vectors and discrete code indices, and reconstructs evidence at query time by combining vector retrieval with topology-aware codebook expansion. The compute environments for memory runtime and offline ISQ training are summarized in Table 5.

Table 5: Compute environments used by MemCode. Offline ISQ codebook training is performed once and is not included in the per-dialogue memory-runtime measurements in Tables 1 and 2.
<table><tr><td>Stage</td><td>Use</td><td>Hardware</td></tr><tr><td>Memory runtime</td><td>Ingestion, retrieval, consoli- dation, and benchmark eval- uation</td><td>12th Gen Intel Core i5-12500H @ 3.10GHz; NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GB; 16 GB RAM</td></tr><tr><td>Offline ISQ training</td><td>VQ-VAE codebook learning before memory evaluation</td><td>12 vCPU Intel Xeon Platinum 8352V @ 2.10GHz; one 32 GB vGPU; 90 GB RAM</td></tr></table>

Beyond the compute environments, reproducibility also depends on the fixed codebook-training configuration and the dataset-specific retrieval budgets. Table 6 summarizes the ISQ training setup, and Table 7 summarizes the dataset-specific ingestion and retrieval settings.

Table 6: Offline VQ-VAE training and ISQ codebook settings. These settings are fixed for all memory-runtime experiments.
<table><tr><td>Block</td><td>Item</td><td>Setting</td></tr><tr><td>Data</td><td>Training pairs</td><td>DailyDialog context-response pairs; the context contains the two previous utterances</td></tr><tr><td>Data</td><td>Sequence length</td><td>Source and target sequences are truncated or padded to 128 tokens</td></tr><tr><td>Model</td><td>Base encoder</td><td>T5-base encoder; code dimension equals the T5 hidden size</td></tr><tr><td>Model</td><td>Projection and codebook</td><td>Stride-1 Conv1d projection, kernel size 3, padding 1; 1024 Euclidean code vectors</td></tr><tr><td>Optimization</td><td>Optimizer and schedule</td><td>AdamW, weight decay 0.01; batch size 16; gradient accumulation 2; 10 epochs; cosine schedule with 10% warmup; gradient clipping at 1.0</td></tr><tr><td>Optimization</td><td>Learning rates</td><td> $1 0 ^ { - 4 }$  for T5 parameters;  $1 0 ^ { - 3 }$  for VQ, projection, and decoder-side parameters</td></tr><tr><td>Objective</td><td>Quantization loss</td><td>Generation cross-entropy plus VQ loss; commit- ment cost 0.10; VQ loss weight 1.0; entropy usage</td></tr><tr><td>Codebook</td><td>EMA and anti-collapse</td><td>regularizer weight 0.01 EMA decay 0.99; codebook initialized from 4096 valid encoder states; codes with EMA cluster size</td></tr><tr><td>Inference</td><td>Fingerprint construction</td><td>below 1.0 are revived every 20 optimizer steps Adjacent duplicate code ids are removed be- fore building AMU fingerprints and co-occurrence topology</td></tr></table>

Table 7: Dataset-specific ingestion and evidence-reconstruction budgets.
<table><tr><td>Dataset</td><td>Ingestion unit</td><td>Extraction background</td><td>Dense path</td><td>Codebook path</td><td>Final evidence budget</td></tr><tr><td>LoCoMo</td><td>A-B dialogue turn</td><td>10 prior sentences</td><td>20 AMUs</td><td>20 AMUs</td><td> $4 0 \mathrm { A M U s } + 2 5$  raw dialogue turns</td></tr><tr><td>LongMemEval-S</td><td>User-Assistant turn</td><td>5 prior sentences</td><td>20 AMUs</td><td>20 AMUs</td><td>40 AMUs + 10 raw dialogue turns</td></tr></table>

The extraction background controls how much local context is available when the system rewrites dialogue into self-contained AMUs, while the final evidence budget controls how much retrieved material is shown to the answer generator. LoCoMo uses a larger raw-context budget because image captions, speaker roles, and local conversational references often disambiguate short facts. LongMemEval-S has longer sessions and more topic drift, so its smaller raw-context budget keeps the final prompt closer to the retrieved AMUs and reduces unrelated historical evidence.

## C Hyperparameter Sensitivity and Run Variability

## C.1 Codebook Retrieval Hyperparameters

We also test the sensitivity of the codebook path to three retrieval hyperparameters on LoCoMo: the number of expanded neighbors K, the neighbor-weight decay coefficient α, and the IDF threshold τ . The default configuration in the main paper is K=3, α=0.3, and $\tau _ { \mathrm { I D F } } { = } 0 . 5$ , with an overall accuracy of 78.12%.

Table 8: Additional hyperparameter ablations for the codebook retrieval path on LoCoMo (GPT-4omini). ∆ denotes the signed percentage-point difference relative to the deployed default configuration $( K { = } 3 , \alpha { = } 0 . 3 , \tau _ { \mathrm { I D F } } { = } 0 . \dot { 5 } )$ with 78.12% overall accuracy.
<table><tr><td>Hyperparameter</td><td>Value</td><td>Accuracy (%)</td><td>∆ vs. default</td></tr><tr><td rowspan="5">K</td><td>1</td><td>77.40</td><td>-0.72</td></tr><tr><td>3 (default)</td><td>78.12</td><td>+0.00</td></tr><tr><td>5</td><td>77.47</td><td>-0.65</td></tr><tr><td>7</td><td>77.47</td><td>-0.65</td></tr><tr><td>9</td><td>76.49</td><td>-1.63</td></tr><tr><td rowspan="5">α</td><td>0.1</td><td>77.79</td><td>-0.33</td></tr><tr><td>0.3 (default)</td><td>78.12</td><td>+0.00</td></tr><tr><td>0.5</td><td>77.34</td><td>-0.78</td></tr><tr><td>0.7</td><td>77.53</td><td>-0.59</td></tr><tr><td>0.1</td><td>77.27</td><td>-0.85</td></tr><tr><td rowspan="4">TIDF</td><td>0.3</td><td></td><td></td></tr><tr><td></td><td>77.34</td><td>-0.78</td></tr><tr><td>0.5 (default)</td><td>78.12</td><td>+0.00</td></tr><tr><td>0.7 0.9</td><td>77.53 77.40</td><td>-0.59 -0.72</td></tr></table>

Table 8 shows modest variation across the tested range, with the default configuration consistently ahead of the alternatives. The result suggests a balance among neighbor expansion, propagation strength, and noise suppression. K is the most sensitive parameter: increasing the number of expanded neighbors to 9 drops accuracy to 76.49%, indicating that excessive expansion adds noise. Changes in α and $\tau _ { \mathrm { I D F } }$ are smoother. The codebook path is robust to these two parameters, although moderate propagation strength and filtering give the most stable retrieval.

## C.2 Repeated-run Variability

Although all LLM calls use temperature-0 decoding, the full hosted-LLM pipeline is not perfectly deterministic in practice. Small residual differences can arise from service-side execution, longprompt boundary cases in memory writing and answer generation, and automatic judge decisions near the correctness boundary. We repeated the main results of LoCoMo using the GPT-4o-mini model six times, using the same retrieval configuration as Tables 1 and 2. The run IDs in Table 9 are renumbered chronologically from 1 to 6; they are not tied to internal experiment-folder suffixes.

Table 9: Repeated runs for the main results of LoCoMo on GPT-4o-mini setting under temperature-0 decoding. Category columns report accuracy. The main-table result corresponds to Run 6, which is close to the center of the repeated-run distribution and is not the best run.
<table><tr><td>Run</td><td>Correct / Total</td><td>Overall</td><td>Multi-Hop</td><td>Temporal</td><td>Open</td><td>Single-Hop</td></tr><tr><td>1</td><td>1200 / 1540</td><td>77.92</td><td>64.54</td><td>77.88</td><td>48.96</td><td>85.73</td></tr><tr><td>2</td><td>1199 / 1540</td><td>77.86</td><td>64.89</td><td>76.95</td><td>50.00</td><td>85.73</td></tr><tr><td>3</td><td>1210 / 1540</td><td>78.57</td><td>62.77</td><td>78.19</td><td>48.96</td><td>87.40</td></tr><tr><td>4</td><td>1207 / 1540</td><td>78.38</td><td>65.25</td><td>76.32</td><td>47.92</td><td>87.04</td></tr><tr><td>5</td><td>1212 / 1540</td><td>78.70</td><td>65.60</td><td>77.26</td><td>47.92</td><td>87.16</td></tr><tr><td>6 (main table)</td><td>1203 / 1540</td><td>78.12</td><td>64.89</td><td>76.64</td><td>43.75</td><td>87.04</td></tr><tr><td>Mean ± Std.</td><td>一</td><td> $7 8 . 2 6 \pm 0 . 3 5$ </td><td> $6 4 . 6 6 \pm 0 . 9 9$ </td><td> $7 7 . 2 1 \pm 0 . 7 2$ </td><td> $4 7 . 9 2 \pm 2 . 1 9$ </td><td> $8 6 . 6 8 \pm 0 . 7 5$ </td></tr><tr><td>Range</td><td></td><td> $7 7 . 8 6 - 7 8 . 7 0 $ </td><td> $6 2 . 7 7 - 6 5 . 6 0$ </td><td> $7 6 . 3 2 \mathrm { - } 7 8 . 1 9$ </td><td> $4 3 . 7 5 \substack { - 5 0 . 0 0 }$ </td><td> $8 5 . 7 3 \substack { - 8 7 . 4 0 }$ </td></tr></table>

Table 9 shows that the main results of GPT-4o-mini on LoCoMo are stable across repeated runs: the full overall range is 0.84 percentage points, corresponding to 13 questions out of 1540. The reported 78.12% is slightly below the six-run mean of 78.26% and is therefore a representative run rather than a best-case selection. Category-level variation is largest on Open because it contains only 96 questions; by contrast, the larger Multi-Hop, Temporal, and Single-Hop groups remain within narrow ranges.

## D Backbone Sensitivity

The main results show a consistent but counter-intuitive pattern: GPT-4o-mini is stronger than Qwen3-30B-A3B-Instruct-2507 in several MemCode settings, even though Qwen3 is a newer and generally stronger backbone. This gap is not simply a raw model-capability result. In MemCode, the backbone participates in both memory writing and answer generation, so a more verbose or more expansive model can change the density and noise profile of the memory store itself. To isolate this effect, the automatic judge is held fixed across the two backbones.

Table 10: Backbone-level performance and generation behavior for MemCode without consolidation. Category columns report accuracy. Avg. Ans. reports average answer length in words.
<table><tr><td>Benchmark</td><td>Backbone</td><td>Overall</td><td>Temporal</td><td>Multi</td><td>Open</td><td>Single</td><td>Know.-Upd.</td><td>Single-Asst.</td><td>Avg. Ans.</td></tr><tr><td>LoCoMo LoCoMo</td><td>GPT-4o-mini</td><td>78.12</td><td>76.64</td><td>64.89</td><td>43.75</td><td>87.04</td><td></td><td></td><td>73.12</td></tr><tr><td></td><td>Qwen3</td><td>76.36</td><td>67.91</td><td>59.22</td><td>52.08</td><td>88.11</td><td></td><td></td><td>128.88</td></tr><tr><td>LongMemEval-S</td><td>GPT-4o-mini</td><td>77.20</td><td>69.92</td><td>65.41</td><td></td><td>92.86</td><td>82.05</td><td>92.86</td><td>75.52</td></tr><tr><td>LongMemEval-S</td><td>Qwen3</td><td>73.80</td><td>57.89</td><td>64.66</td><td></td><td>94.29</td><td>78.21</td><td>92.86</td><td>146.37</td></tr></table>

Table 10 shows that the gap is concentrated in precision-sensitive categories. On LoCoMo, GPT-4omini is 8.73 points higher on Temporal and 5.67 points higher on Multi, while Qwen3 is higher on Open and slightly higher on Single. On LongMemEval-S, the largest gap is again Temporal: 69.92% for GPT-4o-mini versus 57.89% for Qwen3. This pattern suggests that Qwen3 is not uniformly weaker. Rather, it is less stable under a fixed evidence budget when the question requires exact dates, counts, or entity binding.

Table 11: Backbone-induced memory-store characteristics. Values are averaged over dialogue-level memory stores. Qwen3 writes more AMUs and activates more codebook assignments, which increases coverage but also raises the density of potential distractors.
<table><tr><td>Benchmark</td><td>Backbone</td><td>AMUs / dialog</td><td>Active codes</td><td>Code assignments</td><td>Max code freq.</td><td>Avg. AMU words</td></tr><tr><td>LoCoMo</td><td>GPT-4o-mini</td><td>591.1</td><td>216.9</td><td>3678.6</td><td>294.0</td><td>16.48</td></tr><tr><td>LoCoMo</td><td>Qwen3</td><td>940.4</td><td>266.1</td><td>6150.2</td><td>461.4</td><td>15.92</td></tr><tr><td>LongMemEval-S</td><td>GPT-4o-mini</td><td>1553.4</td><td>420.4</td><td>9941.1</td><td>798.0</td><td>21.41</td></tr><tr><td>LongMemEval-S</td><td>Qwen3</td><td>2130.3</td><td>450.2</td><td>14532.2</td><td>1356.1</td><td>21.80</td></tr></table>

The memory statistics in Table 11 explain why the stronger backbone does not automatically yield higher end-to-end memory accuracy. Qwen3 produces substantially more AMUs: 59.1% more on LoCoMo and 37.1% more on LongMemEval-S. It also produces many more code assignments, increasing by 67.2% and 46.2% respectively. This denser write behavior can improve coverage for open-ended questions, but it also creates more near-duplicate facts and more competing codebook activations. Since the answer prompt still keeps a fixed number of AMUs, higher memory density can make the selected evidence less clean.

Table 12: Paired question-level disagreements between GPT-4o-mini and Qwen3 under MemCode. GPT-only means GPT-4o-mini answers correctly while Qwen3 does not; Qwen-only is the reverse.
<table><tr><td>Benchmark</td><td>Shared n</td><td>Both correct</td><td>Both wrong</td><td>GPT-only</td><td>Qwen-only</td><td>Largest GPT-only source</td></tr><tr><td>LoCoMo</td><td>1540</td><td>1055</td><td>216</td><td>148</td><td>121</td><td>Temporal: 55 vs. 27</td></tr><tr><td>LongMemEval-S</td><td>500</td><td>322</td><td>67</td><td>64</td><td>47</td><td>Temporal: 28 vs. 12</td></tr></table>

The paired comparison in Table 12 supports the same interpretation. GPT-4o-mini has more unique wins on both benchmarks, and the largest asymmetry comes from temporal reasoning. Manual inspection of the disagreement patterns shows recurring Qwen3 failure modes: answering with the total amount rather than the per-item amount, selecting a nearby but wrong event date, overexplaining abstention questions, or following a plausible distractor when multiple related facts are retrieved. These errors are consistent with a memory-system interaction: a more expansive writer and generator improves recall coverage, but it also increases distractor density and makes exact evidence selection harder. The backbone gap therefore reflects how the model interacts with the external memory pipeline, not a simple ranking of model capability in isolation.

## E Further Retrieval and Codebook Analysis

This appendix examines retrieval-path behavior through same-budget controls, dense-embedding sensitivity, question-level novelty analysis, case studies, global codebook usage, and a local visualization of semantic purity.

## E.1 Codebook Path vs. a BM25 Keyword Second Path

The main paper summarizes a second-path replacement ablation in Table 4. Here we provide the full trace-level version. We replace MemCode’s codebook path with vanilla BM25 keyword retrieval under the same “20+20” retrieval budget: Path A remains dense vector retrieval, while Path B changes from discrete codebook retrieval to BM25.

Table 13: Full details for the second-path replacement ablation on LoCoMo (GPT-4o-mini). ∆ denotes codebook minus BM25. “C-only” counts cases where the codebook path answers correctly while BM25 does not, and “B-only” denotes the reverse. “New Base” and “New Time” report the average number of additional base facts and time buckets injected by the BM25 stage beyond the vector stage.
<table><tr><td rowspan="2">Category</td><td rowspan="2">n</td><td colspan="3">Accuracy</td><td colspan="2">Paired Wins</td><td colspan="2">BM25 Injection</td></tr><tr><td>Codebook</td><td>BM25</td><td>Δ</td><td>C-only</td><td>B-only</td><td>New Base</td><td>New Time</td></tr><tr><td>Overall</td><td>1540</td><td>78.12</td><td>76.75</td><td>+1.36</td><td>105</td><td>84</td><td>18.58</td><td>10.96</td></tr><tr><td>Multi-Hop</td><td>282</td><td>64.89</td><td>60.64</td><td>+4.26</td><td>33</td><td>21</td><td>18.50</td><td>10.65</td></tr><tr><td>Temporal Reasoning</td><td>321</td><td>76.64</td><td>70.41</td><td>+6.23</td><td>35</td><td>15</td><td>18.66</td><td>11.34</td></tr><tr><td>Open Domain</td><td>96</td><td>43.75</td><td>50.00</td><td>-6.25</td><td>5</td><td>11</td><td>18.59</td><td>10.78</td></tr><tr><td>Single-Hop / Basic</td><td>841</td><td>87.04</td><td>87.63</td><td>-0.59</td><td>32</td><td>37</td><td>18.58</td><td>10.93</td></tr></table>

The codebook path is not directly replaceable by a lexical second path. Overall accuracy rises from 76.75% with BM25 to 78.12% with the codebook path, or 21 more correct answers. The advantage is concentrated in the categories closest to structured memory retrieval: Temporal Reasoning improves by 6.23 percentage points and Multi-Hop by 4.26 points. BM25 is 6.25 points higher on Open Domain, but that category contains only 96 questions; on Single-Hop / Basic, BM25 has only a 0.59- point edge. The distribution matches the inductive biases of the two paths. BM25 favors surface term overlap, whereas the codebook path helps when evidence has stronger temporal structure or higher compositionality.

The paired comparison over 1540 shared question-level traces supports the category-level result. There are 105 samples where the codebook path is correct and BM25 is wrong, compared with 84 in the reverse direction. The asymmetry again comes from the more complex categories. On Temporal Reasoning, the codebook path has 35 “codebook-only wins” versus 15 “BM25-only wins”; on Multi-Hop, the counts are 33 versus 21. BM25 shows only local advantages on Single-Hop / Basic (37 versus 32) and Open Domain (11 versus 5).

Retrieval traces also show that BM25 is not empty expansion: relative to the vector stage, it injects about 18.6 additional new base facts and about 11 new time buckets per question on average. Still, this lexical expansion does not close the gap on temporal and compositional questions. The codebook path helps by adding candidates that better match structured memory retrieval, rather than by merely expanding the candidate set.

These results should not be read as showing that the codebook path is uniformly better than BM25 across all question types. Under the same dual-path budget, BM25 is a reasonable sparse-retrieval control, but it does not reproduce the gain pattern of the codebook path, especially its relative advantage on temporal and multi-hop questions.

## E.2 Same-Budget Retrieval Controls and an Explicit-Graph Baseline

As a supplementary explicit-graph control, we evaluate Full HippoRAG2 [41] on the same frozen 5,911-AMU store with at most 40 evidence items per query. It answers 1,200 of 1,540 questions correctly, compared with 1,203 for MemCode; the 0.20-point difference is not statistically significant in the paired analysis. Thus, an explicit graph can recover comparable associations, while Table 14 isolates the additional construction and query profile.

Table 14: Additional retrieval-system cost beyond the shared AMU construction stage. Query latency is averaged over LoCoMo questions.
<table><tr><td>Metric</td><td>MemCode</td><td>Full HippoRAG2@40</td></tr><tr><td>Additional model calls</td><td>0</td><td>12,779</td></tr><tr><td>Additional tokens</td><td>0</td><td>8.47M</td></tr><tr><td>Explicit graph construction</td><td>None</td><td>34.49 min</td></tr><tr><td>Graph size</td><td>None</td><td>17,275 nodes / 109,779 edges</td></tr><tr><td>Query latency</td><td>0.060 s</td><td>2.864 s</td></tr></table>

The explicit graph baseline is evaluated favorably with a one-shot build over each complete dialogue history. Even in this static setting, its additional construction calls and 47.8× query latency contrast with MemCode’s use of code assignments already produced during AMU indexing. This comparison supports an efficiency claim, rather than a claim that graph retrieval is intrinsically less accurate.

## E.3 Sensitivity to the Dense Embedding Model

We further test whether MemCode’s LoCoMo result depends on the particular dense embedding backbone used by the vector path. The main experiments use bge-base-en-v1.5. As a smaller and widely used alternative, we replace the dense encoder with all-MiniLM-L6-v2 while keeping the GPT-4o-mini writer/reader, the 1024-entry codebook, and the same “20+20” retrieval budget fixed. Table 15 reports paired question-level results over the same 1540 LoCoMo questions.

Table 15: Embedding-model sensitivity on LoCoMo (GPT-4o-mini). The bge-base-en-v1.5 row corresponds to the main MemCode setting. ∆ is the signed percentage-point difference of all-MiniLM-L6-v2 minus bge-base-en-v1.5. “bge-only” and “MiniLM-only” count paired questionlevel wins.
<table><tr><td>Category</td><td>n</td><td>bge Acc.</td><td>MiniLM Acc.</td><td>∆</td><td>bge-only</td><td>MiniLM-only</td><td>Shared correct</td></tr><tr><td>Overall</td><td>1540</td><td>78.12</td><td>76.23</td><td>-1.88</td><td>90</td><td>61</td><td>1113</td></tr><tr><td>Multi-Hop</td><td>282</td><td>64.89</td><td>62.77</td><td>-2.13</td><td>23</td><td>17</td><td>160</td></tr><tr><td>Temporal Reasoning</td><td>321</td><td>76.64</td><td>73.83</td><td>-2.80</td><td>20</td><td>11</td><td>226</td></tr><tr><td>Open Domain</td><td>96</td><td>43.75</td><td>43.75</td><td>+0.00</td><td>4</td><td>4</td><td>38</td></tr><tr><td>Single-Hop / Basic</td><td>841</td><td>87.04</td><td>85.37</td><td>-1.66</td><td>43</td><td>29</td><td>689</td></tr></table>

The smaller encoder preserves the qualitative result but slightly weakens retrieval accuracy. Overall accuracy decreases from 78.12% to 76.23%, and the paired comparison gives 90 bge-only wins versus 61 MiniLM-only wins. The drop is concentrated in the categories that require tighter evidence binding: Temporal Reasoning falls by 2.80 points and Multi-Hop by 2.13 points. Open Domain is unchanged, which is consistent with the earlier BM25 control: open-ended questions are less dominated by exact nearest-neighbor geometry and more affected by whether the final prompt contains broadly relevant context.

The memory-store statistics are nearly unchanged by embedding replacement. Averaged over the 10 LoCoMo dialogue stores, bge-base-en-v1.5 writes 591.1 AMUs, activates 216.9 codes, and produces 3678.6 code assignments per dialogue; all-MiniLM-L6-v2 writes 588.3 AMUs, activates 215.5 codes, and produces 3651.9 assignments. The performance gap therefore does not come from a different memory volume. It more likely reflects the precision of dense evidence selection before codebook expansion. This supports a conservative interpretation: MemCode does not rely on bge-base-en-v1.5 to function, but stronger dense embeddings improve the exact-evidence categories most sensitive to retrieval precision.

## E.4 Dual-Path Retrieval: Novelty Analysis and Case Studies

We measure what the codebook second path adds by computing the proportion of facts independently hit by the codebook path but not covered by the vector path. Figure 4 shows that the added content is closer to facts that the vector path structurally misses than to candidates the vector path could already find at lower ranks. In particular, 69.3% of facts added by the codebook path are still absent from vector top-100, while simply doubling vector Top-K brings only a 1.4% marginal gain. The difference between the two paths is therefore not retrieval depth but representation: the vector path favors fragments with similar surface semantics, while the codebook path uses co-occurrence topology to activate memories that are logically related but more distant on the surface.

![](images/63f8b25c8454c32ab7a904ef67f7f68a9538c1d9c9a8bf988c6a605f77b60f09.jpg)  
Figure 4: Dual-path retrieval: novelty analysis and case studies.

The three cases in Figure 4(C) show the same complementarity in concrete tasks. The failures follow different patterns: in the counting question, the vector path misses another independent entity (Coco); in the location question, it misses a low-frequency proper noun (Whispering Falls); in the shared-attribute question, it misses parallel facts scattered across dialogue fragments. Entity omis sion, suppression of low-frequency proper nouns, and cross-fragment attribute dispersion all follow from geometric nearest-neighbor bias. Codebook topology compensates for them by modeling cooccurrence relations. Together with the BM25 control, this novelty evidence supports a more specific claim: the codebook path recovers structured evidence that both vector and lexical paths often miss.

## E.5 Additional Codebook Space Analysis

We also analyze the codebook space learned in the main experiments. The analysis uses metadata statistics from 10 dialogue buckets in the LoCoMo main run and focuses on sparsity, cross-session stability, global concentration, and co-occurrence topology. The 1024-dimensional codebook activates 216.9 codes on average within a single dialogue (21.18%). The union of 10 dialogues covers 525 codes (51.27%), while only 61 codes (5.96%) appear in every dialogue. The learned discrete space therefore does not collapse into a flat structure where all codes are used uniformly. It forms a backbone-plus-tail pattern with a stable core and conversation-specific tail.

![](images/4e8b9e636573de8244979a8ffd02773c2be73c81eb5b0790b5b936fdab6bc9f0.jpg)  
Figure 5: Global codebook usage concentration. The cumulative coverage curve over ranked codes shows the top-10 codes already explain 39.23% of all assignments and the top-100 explain 91.53%.

Figure 5 shows concentrated rather than uniform code usage: the top-10 codes explain 39.23% of all AMU assignments, the top-20 explain 54.41%, and the top-100 explain 91.53%. This concentration is not trivial collapse, as the activated vocabulary extends well beyond a tiny fixed set of symbols.

![](images/10bbabb2a32c5bc5211c8a5970f01c80f4fd193c12c965f723bda71c26542510.jpg)

![](images/7264eb045aaf3b108b47c41ec33ee7e6b80c05d804b714eefdf7462de09bc3ce.jpg)  
Figure 6: Conversation-level sparsity and shared hub codes. Left: active-code counts per conversation, showing strong per-dialogue sparsity and a small shared core. Right: frequency distribution of the top-15 hub codes across all 10 conversations; all 15 appear in every conversation, with Code 806 being the dominant hub.

Figure 6 complements the aggregate coverage view by showing both sides of the activation pattern. Each dialogue occupies only a small local subset of the 1024-code vocabulary, while the top hub codes recur across all 10 conversations, with Code 806 standing out as the most frequent anchor. Together, Figures 5 and 6 support a shared backbone + conversation-specific tail regime, which is the setting in which a global co-occurrence topology can support associative retrieval beyond nearest-neighbor matching.

Figure 7 checks two properties of the learned discrete space. First, all conversations lie in a highentropy (≈ 5.8–6.4 bits), low-HHI (≈ 0.022–0.034) region, with average entropy 6.00 and average HHI 0.029. This pattern is inconsistent with codebook collapse and supports the entropy regularization and dead-code revival mechanism used in ISQ training. Second, the co-occurrence heatmap is structured: Code 806 acts as a clear hub, the strongest pair (242, 806) reaches weight 1067, and several sub-clusters are visibly denser than the background. Together, these observations support the claim that the codebook path provides a structured semantic topology rather than a bag of arbitrary discrete tokens.

![](images/7fae8bba3a6161ca06d6c5630ea5ba1af1746ead52fa6231750e4915ed72a2b6.jpg)

![](images/51d30200124a04653b03c3b64c0b6cba6360fb378407cfa517b7ef51b3182d00.jpg)  
Figure 7: Codebook stability and co-occurrence topology. Left: entropy–HHI scatter across conversations; all points lie in a high-entropy, low-HHI regime, indicating that the codebook does not collapse to a few trivial symbols. Right: co-occurrence heatmap of representative high frequency codes; the strongest pair is (242, 806) with weight 1067, and visible sub-clusters such as (242, 525, 526) indicate structured semantic topology.

![](images/8fc1beff00244b0cce05e5cae53419367fd77f32c3cac4439ae2f3e652d6bfe1.jpg)

![](images/954490d054adaefe17f8a65880eccda8b392bdded9ada7a6872998a673d86537.jpg)

![](images/479c5467b4052ac94b95abcf96d585275673d235d33a410f938b029e0f806c03.jpg)  
(c) Multi-Cluster Semantic Decoupling Matrix  
Figure 8: Visualization of discrete semantic disentanglement. (a) shows the continuous latent manifold from the T5-base encoder; (b) shows topological tightening after commitment loss; (c) shows the disentanglement matrix with exclusive activation of top-K codebook IDs by semantic clusters.

## E.6 Visualization of Discrete Semantic Disentanglement

To complement the aggregate codebook statistics, we inspect the geometry of the learned latent space. Figure 8(a) shows that the continuous latent manifold produced by the T5-base encoder remains diffuse, with soft semantic boundaries before quantization. After the commitment constraint is imposed, Figure 8(b) shows the latent space contracting into tighter and more separated discrete semantic clusters.

We probe semantic purity by analyzing the most frequently activated Top-K codebook indices on the validation set. Using Statistical Significance Lift to map input tokens to these high-frequency indices, the disentanglement matrix in Figure 8(c) has a near-diagonal sparse structure, indicating separated discrete representations [42]. For example, Code 442 is selectively activated by medical terms such as “prescription” and “medicine,” while Codes 712 and 715 consistently anchor accommodation- and transportation-related contexts, respectively, with little overlap. This visualization is consistent with the broader appendix evidence: the codebook is not collapsed, and its discrete anchors are semantically meaningful.

## F Prompt Templates

We include the core prompt templates used for memory writing and automatic evaluation. The templates are shown in normalized form with variable placeholders; dataset-specific values such as timestamps, speakers, retrieved evidence, rubrics, and reference answers are substituted at runtime.

## F.1 Atomic Memory Writing

Atomic Fact Extraction with Background Context   
You are an expert Memoryist. Extract dense, stand-alone facts from the conversation.   
Rules   
1. Filter Noise: Ignore greetings, short reactions, and vague chitchat.   
2. Resolve Identity: Replace pronouns (I/you/he) with actual speaker names.   
3. Time Rewriting: Rewrite relative time terms (e.g., “yesterday”, “last week”) into exact dates/-   
times. Use the timestamp header above the text as the anchor.   
4. Merge: Combine fragmented sentences into one complete fact.   
5. Context: If a line is marked as [Background], use it for understanding but do not extract   
facts from it.   
Example   
--- Timestamp: 2023-05-10 14:30 ---   
[User]: I just arrived.   
Output Fact: ‘‘User arrived at 14:30 on 2023-05-10.’’   
Input   
{context}   
Output strictly JSON   
[   
{"fact": "SpeakerName action...",   
"source indices": [1, 2]}   
]

You are an expert Memoryist. Extract dense, stand-alone facts from the conversation. Rules

```jsonl
1. Filter Noise: Ignore greetings, short reactions, and vague chitchat.
2. Resolve Identity: Replace pronouns (I/you/he) with actual speaker names.
3. Time Rewriting: Rewrite relative time terms into exact dates/times using the timestamp
header above the text as the anchor.
4. Merge: Combine fragmented sentences into one complete fact.
Input
{context}
Output strictly JSON
[
{"fact": "SpeakerName action...",
"source indices": [1, 2]}
]
```

The background-aware version is used when the ingestion chunk contains carry-over context from previous turns. This keeps pronoun and temporal resolution stable without allowing old context to create duplicate AMUs. The background-free version is used when the chunk itself already contains the target evidence.

## F.2 LLM-as-Judge

## Standard Tasks

I will give you a question, a correct answer, and a response from a model. Please answer yes if the   
response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct   
answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the   
response only contains a subset of the information required by the answer, answer no.   
Fields   
Reference Time: {query date} (optional)   
Question: {question}   
Correct Answer: {correct answer}   
Model Response: {model response}   
Decision   
Is the model response correct? Answer yes or no only.

## Temporal Reasoning Tasks

You are an intelligent judge evaluating the correctness of an AI’s answer regarding time and dates.   
Fields   
Reference Time: {query date} (optional)   
Question: {question}   
Ground Truth: {correct answer}   
Model Answer: {model response}   
Evaluation Rules   
1. Date formats are equivalent when they refer to the same date.   
2. If the ground truth is a simple time and the model answer contains that time in a full sentence,   
mark it correct.   
3. If the ground truth provides multiple valid options, the model answer is correct if it matches   
any one of them.   
4. For duration questions, allow inclusive/exclusive counting ambiguity and explicit ±1 unit dif  
ferences.

I will give you a question, a rubric for the desired personalized response, and a response from a model.   
Please answer yes if the response satisfies the desired response. The model does not need to reflect all   
points in the rubric; the response is correct as long as it recalls and uses the user’s personal information   
correctly.   
Fields   
Reference Time: {query date} (optional)   
Question: {question}   
Rubric: {rubric}   
Model Response: {model response}   
Decision   
Is the model response correct? Answer yes or no only.

5. If the answer is relative to the reference time and matches the ground-truth logic, mark it   
correct.   
Decision   
Does the model answer factually match the ground truth or one of its valid options? Respond   
with only “yes” or “no”.

## Knowledge Update Tasks

I will give you a question, a correct answer, and a response from a model. Please answer yes if the re  
sponse contains the correct answer. Otherwise, answer no. If the response contains previous information   
along with an updated answer, the response should be considered correct as long as the updated answer   
is the required answer.   
Fields   
Reference Time: {query date} (optional)   
Question: {question}   
Correct Answer: {correct answer}   
Model Response: {model response}   
Decision   
Is the model response correct? Answer yes or no only.

## Preference Tasks

## Abstention Tasks

I will give you an unanswerable question, an explanation, and a response from a model. Please an  
swer yes if the model correctly identifies the question as unanswerable. The model could say that the   
information is incomplete, or that some information is given but the asked information is not.   
Fields   
Reference Time: {query date} (optional)   
Question: {question}   
Explanation: {explanation}   
Model Response: {model response}   
Decision   
Does the model correctly identify the question as unanswerable? Answer yes or no only.

## G Responsible Use and Existing Assets

Broader impacts. MemCode is intended to make long-term agent memory more reliable, auditable, and efficient. Better memory systems can improve personalized assistance, reduce repeated user effort across sessions, and make retrieved evidence easier to inspect through source pointers and structured memory records. The same capability also raises risks when deployed on real user histories: a persistent memory store may retain sensitive personal information, propagate extraction mistakes into future personalization, or retrieve outdated facts after user preferences or circumstances change. These risks are especially relevant because MemCode targets cross-session memory rather than isolated question answering.

Responsible deployment should therefore treat the memory store as user data rather than as ordinary model cache. Practical safeguards include explicit user consent for memory creation, uservisible inspection and deletion controls, retention limits, access control for stored memories and source dialogue, and deployment-time filters for sensitive categories. MemCode’s source point ers, archived-fact records, and conservative consolidation gates provide auditability and reduce irreversible rewriting, but they do not replace product-level privacy controls or jurisdiction-specific compliance review.

Existing assets and licenses. Table 16 summarizes the main third-party assets used in the experiments. We cite the original papers or release pages for these assets and use them according to their accompanying licenses or terms of use. We do not redistribute third-party datasets, model weights, or API outputs as a new dataset; any released code or supplemental material should instruct users to obtain third-party assets from their original sources and follow the corresponding license, non-commercial restriction, or service terms.

Table 16: Existing datasets, models, APIs, and software assets used by MemCode. License and terms information follows the public release pages associated with the cited assets.
<table><tr><td>Asset</td><td>Role in this paper</td><td>Citation/source</td><td>License or terms</td></tr><tr><td>LoCoMo</td><td>Long-term conversational- memory benchmark for QA</td><td>Maharana et al. [32]</td><td>CC BY-NC 4.0 release license</td></tr><tr><td>LongMemEval-S</td><td>evaluation Dynamic multi-session mem- ory benchmark for QA evalu-</td><td>Wu et al. [33]</td><td>MIT-licensed public re- lease</td></tr><tr><td>DailyDialog</td><td>ation Conversation data used to train the discrete codebook compo-</td><td>Li et al. [25]</td><td>CC BY-NC-SA 4.0 re- lease license</td></tr><tr><td>T5-base</td><td>nent Encoder initialization for iso- morphic semantic quantization</td><td>Raffel et al. [36]</td><td>Apache-2.0 release li- cense</td></tr><tr><td>bge-base-en-v1.5</td><td>Dense embedding backbone for vector retrieval</td><td>Xiao et al. [37]</td><td>MIT public model- release terms</td></tr><tr><td>GPT-4o-mini API</td><td>Memory writing, answer gen- eration, and LLM-as-a-judge</td><td>OpenAI [34]</td><td>OpenAI service terms</td></tr><tr><td>Qwen3-30B-A3B-</td><td>evaluation Open-weight LLM backbone</td><td>Yang et al. [35]</td><td>Apache-2.0 model re-</td></tr><tr><td>Instruct-2507 Mem0</td><td>in the main evaluation Baseline long-term memory</td><td>Chhikara et al. [7]</td><td>lease Apache-2.0 public</td></tr><tr><td>LangMem</td><td>system Baseline long-term memory</td><td>LangChain [8]</td><td>repository license MIT public repository</td></tr><tr><td>A-MEM</td><td>system Baseline agentic memory sys-</td><td>Xu et al. [9]</td><td>license MIT public repository</td></tr><tr><td>MemoryOS</td><td>tem Baseline hierarchical memory</td><td>Kang et al. [10]</td><td>license Apache-2.0</td></tr><tr><td></td><td>system</td><td></td><td>public repository license</td></tr><tr><td>LightMem</td><td>Baseline lightweight memory system</td><td>Fang et al. [11]</td><td>MIT public repository license</td></tr></table>