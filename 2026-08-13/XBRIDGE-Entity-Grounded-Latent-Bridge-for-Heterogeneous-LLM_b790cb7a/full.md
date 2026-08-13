# XBRIDGE: Entity-Grounded Latent Bridge for Heterogeneous LLM Communication

Wooseong Yang<sup>1</sup> Wei-Chieh Huang<sup>1</sup> Weizhi Zhang<sup>1</sup> Yu Wang<sup>2</sup> Philip S. Yu<sup>1,†</sup> Junhyun Lee<sup>3,4,†</sup>

<sup>1</sup>University of Illinois Chicago <sup>2</sup>Capital One AI Foundations <sup>3</sup>Hankuk University of Foreign Studies <sup>4</sup>Noah’s Farm <sup>†</sup>Corresponding authors

## Abstract

Heterogeneous multi-agent LLM systems, where agents are powered by different model families, can outperform homogeneous configurations by reducing redundant reasoning patterns. Yet existing communication protocols either operate through text, discarding the sender’s internal representations, or require architectural homogeneity for latent-level transfer. We identify the entity grounding problem in cross-architecture communication: cross-attention bridges that transfer continuous representations across different LLM families suffer from rare-token compression collapse, where entity identity is lost in the continuous bottleneck (bridge-only F1 ∼30%). We propose XBRIDGE, a decode-free communication protocol that addresses this through two mechanisms. Lexical Anchor Mapping (LAM) maps the sender’s original context tokens to the receiver’s vocabulary, providing discrete entity anchors. A Latent Enrichment Bridge (LEB) lets the receiver query the sender’s hidden states for contextual enrichment. The entity anchors ground the bridge’s contextual signals to specific entities through the receiver’s own self-attention. Across three model families (Llama, Qwen, and Mistral), seven benchmarks, and both communication directions, XBRIDGE outperforms text-based communication on all seven tasks for each model pair while achieving 11× lower latency, and in a same-architecture setting it also exceeds a KV-sharing baseline on six of seven tasks. LEB requires only 264M trainable parameters (3.8% of the receiver), is trained on a small balanced sample set, and adds negligible inference overhead. Our implementation is available at https://github.com/WooseongYang/XBridge.

## 1 Introduction

Multi-agent systems (MAS) can improve the reasoning capabilities of large language models (LLMs) by distributing computation across multiple inference passes [1].

Recent findings show that when interacting agents share identical parameters and inductive biases, their interactions may degenerate into redundant majority voting rather than surfacing complementary reasoning [2, 3]. A natural way to mitigate this redundancy is to use heterogeneous ensembles: configurations comprising distinct model families with diverse tokenizers, training corpora, architectures, and alignment procedures. Yet this representational diversity introduces a severe communication bottleneck: agents must exchange information across structurally incompatible latent spaces.

Existing communication paradigms face an efficiency-fidelity trade-off. Text-based methods, such as natural language communication (NLComm) [4], are architecture-agnostic but incur autoregressive decoding latency and compress the sender’s internal state into a textual summary. Latent-level methods avoid sender-side decoding, but existing approaches such as KV cache sharing [5] and hidden-state injection [6] typically assume architectural compatibility, including aligned hidden dimensions, attention layouts, positional encodings, or tokenizers. Learned continuous projections offer a possible route across model families [7]. However, we find that continuous projections alone degrade in a specific and practically important way: they can transfer contextual information without reliably preserving the entities to which that context refers.

![](images/c9c77661d32ac3f7c53c8ab3d8171b64e20b765c9ed6e0e45a3b6077e0a139aa.jpg)

![](images/5a4ccf3e05394c241cd5cadb199c492d823c56e7ed4f34e1f8c94e270f80d17b.jpg)  
Figure 1: XBRIDGE performance and latency (Llama-3.1-8B → Qwen2.5-7B). (a) F1 scores across seven benchmarks. NoComm: receiver only (no communication). NLComm: heterogeneous textbased communication [4]. (b) Per-sample inference latency on H200 GPU. Speedup labels indicate XBRIDGE vs. NLComm.

We define this degradation as the entity grounding problem. In heterogeneous LLM communication, a continuous bridge can transfer contextual information from the sender, but it struggles to preserve the discrete entity identities, such as names, numbers, or rare tokens, to which that context refers. We use entity fidelity to refer to preserving the correct lexical identity, and entity grounding to refer to binding the sender’s contextual signal to the corresponding receiver-native lexical anchor. In bridge-only experiments, the receiver recovers aspects of the context but often loses the exact identities of the participating entities. Homogeneous communication settings are less exposed to this problem because shared tokenizers and representational spaces provide implicit entity anchors. Heterogeneous communication therefore creates a demanding setting in which continuous contextual transfer and exact discrete entity fidelity must be maintained simultaneously.

To address this bottleneck, we propose XBRIDGE, a decode-free communication protocol that mitigates the entity grounding problem via dual channels. First, Lexical Anchor Mapping (LAM) deterministically translates the sender’s original context tokens into the receiver’s vocabulary, providing discrete entity anchors without autoregressive decoding. Second, a Latent Enrichment Bridge (LEB) uses gated cross-attention to let the receiver query the sender’s hidden states, supplying continuous contextual information that can be bound to these lexical anchors. By decoupling the preservation of discrete entity identity (the who and what) from the transfer of continuous contextual knowledge (the how and why), XBRIDGE enables heterogeneous models to communicate without requiring textual decoding or direct latent-space compatibility. The bridge is lightweight, requiring only 264M trainable parameters (3.8% of the receiver), training on a small balanced sample set in under 10 minutes, and adding negligible inference overhead.

In summary, our core contributions are:

• We formulate the entity grounding problem in heterogeneous LLM communication and demonstrate rare-token compression collapse, a failure mode in which continuous bridges preserve contextual signals but lose discrete entity identity.

• We propose XBRIDGE, a decode-free dual-channel protocol that combines Lexical Anchor Mapping for receiver-native entity anchors with Latent Enrichment Bridge for sender-side contextual enrichment. To our knowledge, this is the first protocol to pair an explicit discrete-anchor channel with latent transfer for heterogeneous communication.

• We show that XBRIDGE improves heterogeneous communication across three model families and seven benchmarks, achieving 11× lower latency than text-based communication, and that in a same-architecture setting it exceeds KV-cache sharing on 6 of 7 tasks.

![](images/b150a5d770de6ce1ca6a24b324a50d79351ca7e0fc814abc5ffa9208a3842bb2.jpg)

![](images/6d00ae49adc7055cf81c98debaf9ce01bea466f411418aaac3f778055d12b9d2.jpg)  
Figure 2: Entity fidelity across five communication methods on HotpotQA (Llama→Qwen, 196 samples). NoComm receives no sender signal. NLComm [4] relays text summaries within the same family (homo) or across families (hetero). KVComm [5] shares KV caches within the same architecture. (a) Cosine similarity between generation and answer entity representations. Only XBRIDGE achieves positive alignment. (b) Mean rank of the correct answer token (lower is better). All methods degrade rank beyond the no-communication baseline, while XBRIDGE improves it.

## 2 The Entity Grounding Problem

Heterogeneous communication requires satisfying three requirements. First, the receiver must recover receiver-native lexical identities for entities that appear in the sender context. Second, it must use contextual information encoded in the sender’s hidden states. Third, it must consume this information without treating sender activations as native receiver activations. We refer to these requirements as entityfidelity, entity grounding, and representation compatibility, respectively.

We formalize heterogeneous LLM communication as information transfer between models with incompatible tokenizers, vocabularies, and hidden spaces. Let

$$
M _ { i } = ( \tau _ { i } , V _ { i } , E _ { i } , F _ { i } , G _ { i } ) , \qquad i \in \{ S , R \} ,\tag{1}
$$

denote a sender $M _ { S }$ and receiver $M _ { R } ,$ , where $\tau _ { i }$ is the tokenizer, $V _ { i }$ the vocabulary, $E _ { i }$ the embedding map, $F _ { i }$ the transformer, and $G _ { i }$ the language-model head. The models are heterogeneous when

$$
V _ { S } \ne V _ { R } , \qquad d _ { S } \ne d _ { R } , \qquad F _ { S } \ne F _ { R } ,\tag{2}
$$

where architectural mismatch may include differences in depth, attention layout, normalization, positional encoding, tokenizer design, or training distribution.

The sender observes a context C, while the receiver observes a question $Q$ and must generate an answer A. The sender performs a single forward pass:

$$
c _ { S } = \tau _ { S } ( C ) = ( c _ { 1 } , \dots , c _ { T _ { C } } ) , \qquad H _ { S } = F _ { S } ( E _ { S } ( c _ { S } ) ) \in \mathbb { R } ^ { T _ { C } \times d _ { S } } .\tag{3}
$$

Unless otherwise stated, $H _ { S }$ denotes the sender’s last-layer hidden states. A communication protocol produces a sender-side message $m _ { S } = \pi _ { S } ( c _ { S } , H _ { S } )$ , which the receiver consumes to generate $p _ { \pi } ( A$ $\mathbf { \dot { \boldsymbol { C } } } , \boldsymbol { Q } ) = p _ { R } ( A \mid \boldsymbol { Q } ; \Gamma _ { \theta } ( \boldsymbol { Q } , \boldsymbol { \dot { m _ { S } } } ) )$ . Here, $\Gamma _ { \theta }$ denotes the receiver-side mechanism for consuming the message. Text protocols communicate strings or token sequences, latent protocols communicate continuous states, and hybrid protocols communicate both $\bar { m } _ { S } = ( m _ { \mathrm { d i s c } } ( \bar { C } ) , m _ { \mathrm { c o n t } } ( C ) )$

## 2.1 Entity Fidelity

Let E(C) denote the set of entity mentions in $C ,$ , including names, numbers, dates, and other rare or semantically specific spans. For an entity $e \in { \mathcal { E } } ( C )$ , define its receiver-native lexical realization as

$$
a _ { R } ( e ) = \tau _ { R } ( e ) \in V _ { R } ^ { * } ,\tag{4}
$$

where $V _ { R } ^ { * }$ is the set of finite token sequences over the receiver vocabulary. We call $a _ { R } ( e )$ an entity anchor. Entity fidelity is the requirement that the receiver assigns high probability to the correct receiver-native anchor.

For a gold answer entity $e ^ { \star }$ , we measure entity fidelity by the receiver-side rank

$$
\operatorname { r a n k } _ { \pi } ( e ^ { \star } ; C , Q ) = \operatorname { r a n k } _ { v \in V _ { R } } \big [ \ell _ { R } ( v \mid Q ; \Gamma _ { \theta } ( Q , m _ { S } ) ) \big ] _ { v \in a _ { R } ( e ^ { \star } ) } ,\tag{5}
$$

where $\ell _ { R }$ is the receiver logit induced by $G _ { R } .$ . For multi-token entities, we use the first answer token as a diagnostic proxy. A protocol has high entity fidelity when this rank is low.

Entity fidelity can also be characterized interventionally. Let $C ^ { e  e ^ { \prime } }$ denote the context obtained by replacing every occurrence of e with another entity $\check { e ^ { \prime } }$ of the same semantic type. A protocol is entity-sensitive if

$$
\hat { \cal A } _ { \pi } ( C , Q ) = e \quad \Longrightarrow \quad \hat { \cal A } _ { \pi } ( C ^ { e  e ^ { \prime } } , Q ) = e ^ { \prime } .\tag{6}
$$

This condition separates entity identity from relational structure. If the relation is preserved but the answer does not follow the substituted entity, the protocol does not preserve entity fidelity.

## 2.2 Entity Grounding Failure

Entity fidelity concerns whether the receiver outputs the correct entity. Entity grounding concerns whether the sender’s contextual signal is bound to the correct receiver-native entity anchor. A protocol may transfer contextual information while still leaving that information ungrounded to the correct entity in $V _ { R }$ . A continuous bridge can be written abstractly as $r _ { \theta } ^ { ( \ell ) } = \bar { B _ { \theta } ^ { ( \ell ) } } ( H _ { S } , h _ { R } ^ { ( \ell ) } ) \in \mathbb { R } ^ { T _ { R } \times d _ { R } }$ where $h _ { R } ^ { ( \ell ) } \in \mathbb { R } ^ { T _ { R } \times d _ { R } }$ is the receiver hidden-state sequence at layer ℓ with the receiver sequence length $T _ { R }$ . This abstraction covers projections, adapters, hidden-state fusion, and cross-attention modules. Such a bridge may transmit contextual information encoded in $H _ { S }$ , but by itself it does not expose receiver-native anchors $a _ { R } ( e )$

The resulting failure is an identifiability problem. Consider two contexts $C _ { e }$ and $C _ { e ^ { \prime } }$ that differ primarily by replacing entity e with another entity $e ^ { \prime }$ while preserving the same relational structure. A continuous-only protocol is vulnerable when $\bar { d } \big ( m _ { \mathrm { c o n t } } ( \bar { C _ { e } } ) , m _ { \mathrm { c o n t } } \bar { ( C _ { e ^ { \prime } } ) } \big ) \leq \epsilon ,$ while the required receiver-native anchors remain distinct: $a _ { R } ( e ) \neq a _ { R } ( e ^ { \prime } )$ . In this regime, the receiver may recover the context but struggles to determine which discrete receiver-vocabulary entity instantiates the answer. We refer to this as rare-token compression collapse: continuous representations preserve contextual semantics but lose reliable access to rare or specific lexical identities.

## 2.3 Representation Mismatch

Even if entity identity is preserved, sender activations cannot be directly treated as receiver activations. Let $\mu _ { R } ^ { ( \ell ) }$ denote the distribution of token-level hidden states encountered by receiver layer ℓ. For a projection $P : \mathbb { R } ^ { d _ { S } }  \mathbb { R } ^ { d _ { R } }$ , dimension matching only ensures that $P H _ { S }$ lies in $\mathbb { R } ^ { d _ { R } }$ . It does not imply that projected sender states match the receiver’s native hidden-state distribution: $P H _ { S } \ \nsim$ $\mu _ { R } ^ { ( \bar { \ell } ) }$ in general, where $\mu _ { R } ^ { ( \ell ) }$ denotes the distribution of hidden states at the ℓ-th layer of the receiver model. Directly injecting $\mathrm { \Delta } P H _ { S }$ into the receiver residual stream or key-value cache can therefore introduce out-of-distribution activations.

A representation-compatible protocol should instead let the receiver consume sender information through its own hidden state:

$$
h _ { R } ^ { \prime ( \ell ) } = h _ { R } ^ { ( \ell ) } + g _ { \theta } ^ { ( \ell ) } ( h _ { R } ^ { ( \ell ) } , H _ { S } ) ,\tag{7}
$$

where $g _ { \theta } ^ { ( \ell ) }$ is a learned receiver-conditioned retrieval operator. This formulation preserves the receiver’s native computation while allowing access to sender-side contextual information.

## 3 XBRIDGE: Entity-Grounded Latent Communication

The analysis in Section 2 identifies two challenges for heterogeneous architecture communication: continuous bridges struggle to preserve discrete entity identity (Section 2.2), and sender activations lie outside the receiver’s representation space (Section 2.3). We therefore decompose heterogeneous communication into two complementary mechanisms: Lexical Anchor Mapping (LAM, Section 3.2) provides discrete entity anchors, and a Latent Enrichment Bridge (LEB, Section 3.3) provides contextual enrichment. Both signals are produced by a single sender forward pass with no autoregressive decoding; sender and receiver remain frozen throughout.

## 3.1 XBRIDGE Overview

We adopt the asymmetric communication setting formalized in Section 2, following the protocol established by Shi et al. [5], in which the sender observes a context and the receiver observes a question: the sender $M _ { S }$ processes a context document C through a single forward pass, producing last-layer hidden states $H _ { S } ~ \in ~ \mathbb { R } ^ { T _ { C } \times d _ { S } }$ and context token sequence $c _ { S } = ( c _ { 1 } , \dots , c _ { T _ { C } } )$ . The receiver $M _ { R }$ holds only the question Q and must generate an answer A from whatever signal the sender communicates. XBRIDGE transmits both $\bar { H _ { S } }$ and $c _ { S } \colon$ LAM (Section 3.2) uses $c _ { S } .$ , and LEB (Section 3.3) uses $H _ { S }$

![](images/17ae172d92f78a5adef39413f7632c91108c9e5affd12d465f53147fa3361024.jpg)  
Figure 3: XBRIDGE overview. A single sender forward pass produces hidden states $H _ { S }$ and context tokens $c _ { S }$ . LAM maps $c _ { S }$ to receiver embeddings (entity grounding). LEB lets the receiver query $H _ { S }$ at four layers (contextual enrichment). The receiver generates the answer conditioned on both signals.

## 3.2 Lexical Anchor Mapping (LAM) for Discrete Grounding

LAM addresses the entity grounding failure (Section 2.2) by placing the sender’s original context tokens, including entity mentions, as receiver-native lexical inputs. These lexical anchors provide the discrete basis that LEB requires to ground its contextual enrichment to specific entities (Section 3.4).

Cross-vocabulary mapping. Given the sender’s context tokens $c _ { S } = ( c _ { 1 } , \dots , c _ { T _ { C } } )$ obtained by tokenizing C with $M _ { S } { \bf \ ' } _ { \bf S }$ tokenizer, each token $c _ { i } \in \mathcal { V } _ { S }$ is mapped to the receiver’s vocabulary $\nu _ { R }$ via a deterministic, training-free mapping function $\phi : \mathcal { V } _ { S } \ :  \ : \mathcal { V } _ { R } ^ { * }$ , precomputed offline once per sender–receiver tokenizer pair. For tokens whose surface string appears in both vocabularies, ϕ performs a direct ID-to-ID lookup. For the remainder, a string fallback decodes the token to its surface string and re-tokenizes it with the receiver’s tokenizer, producing one or more receiver tokens expanded in place. The mapping is lossless and adds negligible latency (<1 ms); detailed statistics and examples are in Appendix A.1.

Receiver-native lexical embeddings. The mapped token IDs are converted to receiver embeddings $e _ { \mathrm { c t x } } ~ = ~ ( E _ { R } [ \phi ( c _ { 1 } ) ] , \dots , E _ { R } [ \phi ( c _ { T _ { C } } ) ] )$ via the receiver’s frozen embedding matrix $E _ { R } ,$ , omitting expansion from re-tokenization for clarity. These embeddings are prepended to the receiver’s question input, following the input prepend convention used in soft-token communication [8]:

$$
x _ { R } = [ e _ { \mathrm { c t x } } \ ; \ E _ { R } ( Q ) ]\tag{8}
$$

By using $E _ { R }$ as a lookup table rather than training a separate projection, transmitted tokens are guaranteed to lie in the same space as the receiver’s own input tokens, ensuring that pretrained self-attention patterns apply directly to the transferred context. Unlike prior methods that require the

sender to autoregressively decode a summary [4, 8], LAM transfers the full context without decoding, eliminating both latency and information loss from fixed-length compression.

## 3.3 Latent Enrichment Bridge (LEB) for Context Adaptation

LAM preserves entity identity but transmits only the sender’s raw tokens, not the sender’s contextual understanding of those tokens. The sender’s last-layer hidden states $H _ { S }$ encode information integrated across all context positions through the sender’s full transformer stack, information that raw token embeddings cannot convey. We use the last layer because it produces the most globally-integrated representations, as confirmed by a layer sweep in Appendix B.1. However, $H _ { S }$ lies in the sender’s representation space, which the receiver has never encountered during training (Section 2.3). Directly injecting these representations, whether by prepending hidden states or replacing KV caches, places out-of-distribution vectors into the receiver’s processing pipeline and degrades performance. Crossattention provides a natural solution: rather than injecting sender representations into the receiver, the receiver queries the sender in its own native space. This receiver-driven mechanism has three stages, illustrated in the bridge panel of Figure 3.

Projections. We insert M=4 gated cross-attention modules into the receiver, one at each of layers $\ell \in \{ 6 , 1 3 , 2 0 , 2 7 \}$ out of 28, following the insertion density of Flamingo [9]. We find four modules optimal; fewer provide insufficient capacity while more overfit with limited training data (Appendix B.1). Each module has its own learnable parameters (∼66M each, ∼264M total, 3.8% of the receiver). Both models remain frozen; only the bridge modules are trained. At each insertion layer $\ell ,$ the receiver’s hidden state $h _ { R } ^ { ( \ell ) } \in \mathbb { R } ^ { d _ { R } }$ is projected into a query, while the sender’s hidden states $H _ { S }$ are projected into keys and values:

$$
Q ^ { ( \ell ) } = W _ { Q } ^ { ( \ell ) } \cdot \mathrm { L N } _ { R } ^ { ( \ell ) } ( h _ { R } ^ { ( \ell ) } ) , \quad K ^ { ( \ell ) } = W _ { K } ^ { ( \ell ) } \cdot \mathrm { L N } _ { S } ^ { ( \ell ) } ( H _ { S } ) , \quad V ^ { ( \ell ) } = W _ { V } ^ { ( \ell ) } \cdot \mathrm { L N } _ { S } ^ { ( \ell ) } ( H _ { S } )\tag{9}
$$

where $W _ { Q } ^ { ( \ell ) } \in \mathbb { R } ^ { d _ { R } \times d _ { R } } , W _ { K } ^ { ( \ell ) } , W _ { V } ^ { ( \ell ) } \in \mathbb { R } ^ { d _ { R } \times d _ { S } }$ , and $\mathrm { L N } _ { R } ^ { ( \ell ) } , \mathrm { L N } _ { S } ^ { ( \ell ) }$ are separate layer normalizations for the receiver and sender. The projection matrices simultaneously handle the dimensional mismatch $( d _ { S }  d _ { R } )$ , serving as the learned translation between representation spaces. All four modules query the same $H _ { S }$ , but each has its own parameters, enabling different layers to extract different aspects of the sender’s representation.

Cross-attention. The receiver’s query selects relevant sender positions through scaled dot-product attention:

$$
A ^ { ( \ell ) } = \operatorname { s o f t m a x } \left( Q ^ { ( \ell ) } K ^ { ( \ell ) \top } / \sqrt { d _ { k } } \right) \cdot V ^ { ( \ell ) }\tag{10}
$$

This is the key advantage over injection-based alternatives: the receiver decides what information to retrieve based on its current processing state at layer ℓ, rather than receiving a fixed representation regardless of what it needs.

Gated residual. The cross-attention output is added to the receiver’s hidden state via a gated residual connection, preserving the receiver’s pretrained representations while incorporating bridge information:

$$
{ h _ { R } ^ { \prime ( \ell ) } = h _ { R } ^ { ( \ell ) } + \operatorname { t a n h } ( \alpha ^ { ( \ell ) } ) \cdot A ^ { ( \ell ) } }\tag{11}
$$

where $h _ { R } ^ { \prime ( \ell ) }$ is the updated hidden state that continues through the receiver’s subsequent layers. We initialize $\alpha ^ { ( \ell ) } = 1 . 0 ( \operatorname { t a n h } ( 1 . 0 ) \approx 0 . 7 6 )$ so that the bridge contributes immediately, unlike Flamingo’s zero initialization which requires the model to discover the bridge signal during training. With our limited training budget, warm initialization converges faster (Appendix B.1).

The bridge is trained with standard next-token prediction loss on the answer tokens:

$$
\mathcal { L } ( \theta ) = - \sum _ { t = 1 } ^ { | A | } \log P _ { R } ( a _ { t } \mid a _ { < t } , e _ { \mathrm { c t x } } , Q ; \theta _ { \mathrm { b r i d g e } } )\tag{12}
$$

where $e _ { \mathrm { c t x } }$ is computed via LAM (Section 3.2). Since the sender is frozen, its outputs $( H _ { S }$ and context token IDs) are precomputed and cached, reducing training to under 10 minutes on a single GPU on a balanced set of 587 samples drawn across the seven task domains. Once trained, the bridge is fixed and applied to all tasks at inference time without further adaptation (Section 4). Training data composition is analyzed in Appendix B.2. The bridge adapts to different model pairs by adjusting only the projection dimensions; we train separate instances for each pair using the same protocol.

Table 1: Heterogeneous communication results (F1 %). The best score per task within each model pair is bolded and the Improv. row shows the gain of XBRIDGE over the second-best method, both excluding FullComm.
<table><tr><td rowspan=1 colspan=1>Method         Countries Tipsheets HotpotQA QASPER MuSiQue MFldQA 2Wiki   Avg</td></tr><tr><td rowspan=1 colspan=1>Ms: Llama-3.1-8B-Instruct; MR: Qwen2.5-7B-Instruct</td></tr><tr><td rowspan=1 colspan=1>NoComm         0.0      66.5      24.6      7.1      10.6      17.3      19.3    20.8</td></tr><tr><td rowspan=1 colspan=1>FullComm       50.8     98.5     78.4     31.3     31.5     48.2     47.6    55.2</td></tr><tr><td rowspan=1 colspan=1>NLCommhetero    22.6     96.3      68.5      33.7     32.5     24.8     14.2    41.8</td></tr><tr><td rowspan=1 colspan=1>XBRIDGE        72.5     99.8     78.8     47.5     48.2     44.2     51.1    63.2</td></tr><tr><td rowspan=1 colspan=1>Improv.        +49.9    +3.5    +10.3    +13.8    +15.7    +19.4    +36.9   +21.4</td></tr><tr><td rowspan=1 colspan=1>Ms: Qwen2.5-7B-Instruct; MR: Llama-3.1-8B-Instruct</td></tr><tr><td rowspan=1 colspan=1>NoComm         0.0      11.0     31.4      3.8      2.6      11.2      12.9     10.4</td></tr><tr><td rowspan=1 colspan=1>FullComm        67.0     95.0     82.9      36.7      41.3      53.3     51.8    61.2</td></tr><tr><td rowspan=1 colspan=1>NLCommhetero    56.2     94.8      76.8      34.1     33.5     21.7      16.3    47.6</td></tr><tr><td rowspan=1 colspan=1>XBRIDGE        62.0     99.0     80.6     46.6     62.8     40.3     42.1    61.9</td></tr><tr><td rowspan=1 colspan=1>Improv.        +5.8     +4.2     +3.8     +12.5    +29.3    +18.6    +25.8   +14.3</td></tr><tr><td rowspan=1 colspan=1>Ms: Mistral-7B-Instruct-v0.3; MR: Qwen2.5-7B-Instruct</td></tr><tr><td rowspan=1 colspan=1>NoComm         0.0      66.5     24.6      7.1      10.6      17.3      19.3    20.8</td></tr><tr><td rowspan=1 colspan=1>FullComm        50.8     98.5     78.4     31.3     31.5     48.2     47.6    55.2</td></tr><tr><td rowspan=1 colspan=1>NLCommhetero    43.6     96.0      66.1      31.8     28.3      27.0      15.9    44.1</td></tr><tr><td rowspan=1 colspan=1>XBRIDGE        70.0     100.0     84.9     50.6     65.0     35.8     48.7    65.0</td></tr><tr><td rowspan=1 colspan=1>Improv.        +26.4    +4.0    +18.8    +18.8    +36.7     +8.8    +32.8   +20.9</td></tr></table>

## 3.4 Entity-Grounded Bridge Integration

The two mechanisms are integrated through the receiver’s own processing, without any explicit fusion layer. LAM places entity mentions (e.g., “Christopher Nolan”) in the receiver’s input as discrete embeddings. As these propagate through the receiver’s layers via self-attention, they shape the receiver’s hidden states $h _ { R } ^ { ( \ell ) }$ at every depth, embedding entity information into the representations that LEB modules use as queries. This is the grounding mechanism: when LEB at layer ℓ computes $Q ^ { ( \ell ) } = W _ { Q } ^ { ( \ell ) } \cdot \mathrm { L N } _ { R } ^ { ( \ell ) } ( h _ { R } ^ { ( \ell ) } )$ , the query already encodes which entities are present and where they appear, enabling the cross-attention to focus on the sender positions that carry relevant contextual information about those entities. Without LAM, $h _ { R } ^ { ( \ell ) }$ reflects only the question, producing queries too vague to extract entity-specific signals from the sender, so the LEB conveys contextual signals while entity identity is lost in the continuous bottleneck, which is why LEB-only performance drops to ∼30% F1 (Section 4.3). Without LEB, the receiver has the sender’s context tokens but lacks the sender’s contextual understanding of their relationships. Together, LEB’s contextual enrichment is anchored to specific entities rather than floating in continuous space. We verify this complementarity quantitatively in Section 4.3.

## 4 Experiments

## 4.1 Experimental Setup

Models. We evaluate communication across three model families to test cross-architecture generality: Llama-3.1-8B-Instruct [10] (Llama), Qwen2.5-7B-Instruct [11] (Qwen), and Mistral-7B-Instruct-v0.3 (Mistral). These families differ in architecture (hidden dimension, layer count, attention structure), tokenizer design, and training data, ensuring that improvements are not an artifact of a single architecture pairing. We evaluate three heterogeneous pairs: Llama→Qwen, Qwen→Llama, and Mistral→Qwen. We also verify that the framework generalizes to same-architecture settings (Appendix C.1). To test whether larger senders produce richer representations for LEB, we additionally evaluate sender size scaling with Qwen2.5-{1.5B, 7B, 14B}-Instruct→Llama.

Tasks and protocol. We evaluate on seven benchmarks: HotpotQA [12], MuSiQue [13], QASPER [14], 2WikiMultihopQA (2Wiki) and MultiFieldQA-en (MFldQA) from LongBench [15], Countries and Tipsheets [5]. The sender receives the context document only and the receiver receives the question only, the asymmetric protocol established by Shi et al. [5], which we apply identically to every method compared here. For NLComm baselines, the sender generates a 128-token summary via greedy decoding. We report F1 score. Task details are in Appendix E.

Table 2: Entity perturbation experiment on 100 HotpotQA samples. Each sample’s answer entity is replaced with an unrelated entity (e.g., “Christopher Nolan” → “Bong Joon Ho”). LAM and LEB inputs are independently swapped to test which mechanism determines entity identity. Output is classified by which entity appears in the generated answer.
<table><tr><td>Condition</td><td>LAM</td><td>LEB</td><td>→ original</td><td>→ swapped</td><td>→ neither</td></tr><tr><td>A: Original</td><td>original</td><td>original</td><td>45</td><td>0</td><td>55</td></tr><tr><td>B: Both swapped</td><td>swapped</td><td>swapped</td><td>4</td><td>37</td><td>59</td></tr><tr><td>C: LAM swapped only</td><td>swapped</td><td>original</td><td>5</td><td>37</td><td>58</td></tr><tr><td>D: LEB swapped only</td><td>original</td><td>swapped</td><td>45</td><td>0</td><td>55</td></tr></table>

Compared methods. We compare against natural language communication (NLComm) [4], where the sender generates a text summary and relays it to the receiver. We distinguish NLComm<sub>hetero</sub> (cross-family) and N $\mathrm { . C o m m _ { h o m o } }$ (same family). We also report two single-model reference points measured with the receiver alone. NoComm, where the receiver sees only the question, is a lower reference indicating what the receiver achieves without communication. FullComm, where the receiver reads the full context directly, removes the communication problem rather than solving it, and serves as an upper reference in the sense of the Skyline used by Shi et al. [5]. Neither is a competing communication method, and both are excluded from the best-per-task comparison. Same-architecture methods, including $\mathrm { N L C o m m } _ { \mathrm { h o m o } }$ and KVComm [5] which shares KV caches between models of the same family, are compared in Appendix C.1.

## 4.2 Main Results

Table 1 presents heterogeneous communication results across all three model pairs and seven benchmarks. XBRIDGE outperforms $\mathrm { N L C o m m } _ { \mathrm { h e t e r o } }$ on all 7 tasks across all three model pairs, with average gains of +21.4 pp (Llama→Qwen), +14.3 pp (Qwen→Llama), and +20.9 pp (Mistral→Qwen). The improvements are largest on tasks requiring multi-hop reasoning or world knowledge (Countries, 2WikiMQA, MuSiQue), where LEB provides contextual enrichment that text summaries cannot convey. On average, XBRIDGE surpasses even FullComm across all three model pairs (63.2 vs. 55.2 for Llama→Qwen, 61.9 vs. 61.2 for Qwen→Llama, 65.0 vs. 55.2 for Mistral→Qwen). Since LAM places the full mapped context in the receiver’s input, XBRIDGE sees what FullComm sees and adds the sender’s hidden states through LEB, so the comparison is not one of indirect against direct access. The same-architecture setting isolates this: with Qwen2.5-7B sending to itself, LAM reduces to an identity mapping, so the receiver’s input matches FullComm’s and only LEB differs, and $\mathrm { X B R I D G E _ { h o m o } }$ still leads by 7.9 pp on average (Appendix C.1). With identical models and identical input, the gain comes from LEB carrying the sender’s integrated representation that the receiver would otherwise recompute. The optimal sender varies by task: Mistral→Qwen achieves the highest scores on HotpotQA, QASPER, and MuSiQue, while Llama→Qwen leads on MFldQA and 2WikiMQA. This complementarity suggests that selecting the sender based on task characteristics can further improve performance. In the same-architecture setting, XBRIDGE also exceeds KVComm [5] on 6 of 7 tasks (Appendix C.1).

## 4.3 Ablation: Both Anchors and Enrichment Are Necessary

The gains reported above rely on both LAM and LEB working together. To understand how much each contributes, we remove them individually on HotpotQA (Table 3). XBRIDGE without LAM $( \mathrm { X B R I D G E \mathrm { _ { - L A M } } } )$ drops to 30.3% F1, close to NoComm: LEB receives continuous signals but without entity anchors these signals are not bound to specific entities, confirming the entity grounding failure identified in Section 2.2. XBRIDGE without LEB $( \mathrm { X B R I D G E _ { \mathrm { - L E B } } } )$ reaches 56.5%, providing entity grounding but no contextual enrichment beyond what the tokens themselves contain. The full XBRIDGE achieves 78.8%, exceeding the FullComm. The +22.3 pp gain from adding LEB to LAM contrasts with the modest +5.7 pp from LEB-only over NoComm, demonstrating that grounding is the enabling condition for LEB effectiveness. LEB contribution scales with task reasoning complexity:

![](images/d3bcd3d7ee01de02652678765078eaced74480f1c386338ecfbd5cac8dee5d65.jpg)

![](images/3b5342d3dcb1f78248ff187aeb6a33dc912b715e0aa12079a77ab8d7bc4dabfd.jpg)

![](images/6614e887b7de963746a24a38de51095e8d9c77709b99104c764cb771e36bc646.jpg)  
Figure 4: Bridge mechanism analysis on HotpotQA (Llama→Qwen, 196 samples). (a) Probability assigned to the correct answer token. (b) Relative attention weight on answer positions. NLComm attends most heavily yet predicts worst (cf. Figure 2b), revealing an attention-prediction disconnect. (c) Per-sample entity cosine similarity comparing NoComm and XBRIDGE. Points above y=x indicate bridge improvement.

Table 3: Entity grounding ablation on HotpotQA, Llama→Qwen (F1 %).
<table><tr><td>Config.</td><td>F1</td></tr><tr><td>NoComm</td><td>24.6 +5.7</td></tr><tr><td>XBRIDGE-LAM</td><td>30.3</td></tr><tr><td>XBRIDGE-LEB 56.5</td><td>+31.9</td></tr><tr><td>XBRIDGE</td><td>78.8 +54.2</td></tr></table>

Table 5: Inference latency per sample on HotpotQA.
<table><tr><td>Sender</td><td>Avg</td></tr><tr><td>Qwen2.5-1.5B</td><td>59.5</td></tr><tr><td>Qwen2.5-7B</td><td>61.9</td></tr><tr><td>Qwen2.5-14B</td><td>62.1</td></tr></table>

<table><tr><td>Method</td><td>s/sample</td><td>Rel. Latency</td></tr><tr><td>NoComm</td><td>0.09</td><td>1.0×</td></tr><tr><td>FuliComm</td><td>0.13</td><td>1.4×</td></tr><tr><td>NLCommhomo</td><td>1.70</td><td>18.9×</td></tr><tr><td>NLCommhetero</td><td>1.70</td><td>18.9×</td></tr><tr><td>KVComm</td><td>0.13</td><td>1.4×</td></tr><tr><td>XBRIDGE</td><td>0.15</td><td>1.7×</td></tr></table>

from +35.8 pp on factual inference requiring world knowledge (Countries) to +2.0 pp on structured extraction (Tipsheets), with per-task breakdown in Appendix C.2.

## 4.4 Bridge Mechanism and Role Separation

The ablation confirms that both mechanisms are necessary. A deeper question follows: does each carry a distinct type of information, or do they redundantly encode the same signal? To test this, we run an entity perturbation experiment on 100 HotpotQA samples, replacing every occurrence of the answer entity in the context with an unrelated entity of the same type (e.g., “Christopher Nolan” → “Bong Joon Ho”), then independently control which version each mechanism receives (Table 2).

Swapping LEB alone does not change the output entity: conditions A and D produce identical outputs (45 original, 0 swapped, 55 neither). The “neither” category contains outputs that mention neither the original nor the substituted entity, such as a date, another entity, or an unrelated span; we therefore interpret the experiment by comparing the original-versus-swapped direction among target-entity outputs. Under this interpretation, swapping LAM shifts the output toward the substituted entity regardless of which LEB version is used; conditions B and C produce nearly identical counts (37 swapped, 4–5 original, 58–59 neither). These results indicate a clear role separation: LAM supplies which entities appear in the output, while LEB supplies how the receiver reasons about them.

Figure 4 examines LEB’s effect on the receiver’s internal state across five communication conditions on 196 HotpotQA samples. NLComm methods attend most heavily to answer positions (2.0–2.7× above average) yet achieve the worst token rank because paraphrased text replaces exact entity mentions. XBRIDGE is the only method that produces positive cosine alignment between generation and answer entity representations, translating to correct-token probability of 0.752 vs. 0.635–0.662 for all other conditions (94% of individual samples, Appendix D.1).

## 4.5 Beyond Accuracy: Latency, Composability, and Sender Scaling

XBRIDGE achieves 11× lower latency than NLComm (0.15 s vs. 1.70 s per sample on HotpotQA) by eliminating autoregressive decoding: the sender performs a single forward pass (∼0.05 s) with negligible bridge overhead (<0.01 s), as reported in Table 5. The modular bridge design also enables zero-shot composability: two independently trained bridges (Llama and Mistral as senders) combine at inference without retraining, achieving 70.4% F1 on HotpotQA compared to 67.0% for singlesender full context and 56.8% for dual NLComm (Appendix C.3). Finally, LAM is sender-size invariant since it depends only on the vocabulary mapping, not the sender’s capacity: even a 1.5B sender achieves 59.5%, approaching FullComm, while the 7B and 14B senders surpass it (Table 4).

## 5 Conclusion

In this work, we identified the entity grounding problem in heterogeneous LLM communication, where continuous bridges lose entity identity in the latent bottleneck. We proposed XBRIDGE, a decode-free protocol that addresses this by combining LAM with LEB, decomposing communication into discrete entity anchors and continuous contextual enrichment. Extensive experiments across three model families, seven benchmarks, and both communication directions demonstrated that XBRIDGE outperforms NLComm on every task with 11× lower latency. We highlight the complementarity of the two mechanisms: LAM determines which entities appear in the output, while LEB determines how the receiver reasons about them, as confirmed by the entity perturbation experiment. Our work shows that heterogeneous LLMs can communicate at the latent level while preserving entity fidelity, and we see this as an early step toward cross-architecture multi-agent collaboration.

## Limitations

XBridge is trained for a single sender to receiver direction, so one bridge carries information one way over one exchange. A two-way setup would add a second bridge for the reverse direction, and since each bridge is lightweight and decode-free, the cost per exchange would stay low, but multi-turn latent dialogue, where the two models alternate and each turn conditions on what the other sent, is beyond what we evaluate here. The same holds for the number of agents: we have tested composition of two independently trained bridges, leaving open how bridge signals interact when more agents participate. Extending the protocol along both directions, bidirectional exchange and many-agent composition, is what we see as the path toward a cross-architecture communication protocol suited to full multi-agent systems.

## References

[1] Jiaru Zou, Xiyuan Yang, Ruizhong Qiu, Gaotang Li, Katherine Tieu, Pan Lu, Ke Shen, Hanghang Tong, Yejin Choi, Jingrui He, et al. Latent collaboration in multi-agent systems. arXiv preprint arXiv:2511.20639, 2025.

[2] Tianyu Hu, Zhen Tan, Song Wang, Huaizhi Qu, and Tianlong Chen. Multi-agent debate for LLM judges with adaptive stability detection. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id= Vusd1Hw2D9.

[3] Hyeong Kyu Choi, Jerry Zhu, and Sharon Li. Debate or vote: Which yields better decisions in multi-agent large language models? In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=iUjGNJzrF1.

[4] Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. In Forty-first international conference on machine learning, 2024.

[5] Xiangyu Shi, Marco Chiesa, Gerald Q. Maguire Jr., and Dejan Kostic. KVComm: Enabling efficient LLM communication through selective KV sharing. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum? id=F7rUng23nw.

[6] Vignav Ramesh and Kenneth Li. Communicating activations between language model agents. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=W6RPXUUFic.

[7] Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang, and Yu Wang. Cache-to-cache: Direct semantic communication between large language models. In The

Fourteenth International Conference on Learning Representations, 2026. URL https:// openreview.net/forum?id=LeatkxrBCi.

[8] Chau Pham, Boyi Liu, Yingxiang Yang, Zhengyu Chen, Tianyi Liu, Jianbo Yuan, Bryan A. Plummer, Zhaoran Wang, and Hongxia Yang. Let models speak ciphers: Multiagent debate through embeddings. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=sehRvaIPQQ.

[9] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35: 23716–23736, 2022.

[10] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

[11] Qwen Team. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

[12] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings ofthe 2018 conference on empirical methods in natural language processing, pages 2369–2380, 2018.

[13] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop questions via single-hop question composition. Transactions ofthe Associationfor Computational Linguistics, 10:539–554, 2022.

[14] Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A Smith, and Matt Gardner. A dataset of information-seeking questions and answers anchored in research papers. In Proceedings ofthe 2021 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 4599–4610, 2021.

[15] Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al. Longbench: A bilingual, multitask benchmark for long context understanding. In Proceedings ofthe 62nd annual meeting ofthe associationfor computational linguistics (volume 1: Long papers), pages 3119–3137, 2024.

[16] Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V. Chawla, Olaf Wiest, and Xiangliang Zhang. Large language model based multi-agents: a survey of progress and challenges. In Proceedings ofthe Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI ’24, 2024. ISBN 978-1-956792-04-1. doi: 10.24963/ijcai.2024/890. URL https://doi.org/10.24963/ijcai.2024/890.

[17] Khanh-Tung Tran, Dung Dao, Minh-Duong Nguyen, Quoc-Viet Pham, Barry O’Sullivan, and Hoang D Nguyen. Multi-agent collaboration mechanisms: A survey of llms. arXiv preprint arXiv:2501.06322, 2025.

[18] Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and Zhaopeng Tu. Encouraging divergent thinking in large language models through multiagent debate. In Proceedings of the 2024 conference on empirical methods in natural language processing, pages 17889–17904, 2024.

[19] Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for" mind" exploration of large language model society. Advances in neural information processing systems, 36:51991–52008, 2023.

[20] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via multi-agent conversations. In First conference on language modeling, 2024.

[21] Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, et al. Chatdev: Communicative agents for software development. In Proceedings of the 62nd annual meeting of the association for computational linguistics (volume 1: Long papers), pages 15174–15186, 2024.

## A LAM Details

## A.1 Cross-Vocabulary Mapping

Table 6 summarizes the vocabulary overlap and token transfer statistics for the two sender–receiver pairs used in our experiments.

Table 6: Cross-vocabulary mapping statistics on HotpotQA (100 validation samples). Direct-mapped tokens use ID-to-ID lookup; the remainder use string-based fallback.
<table><tr><td></td><td>Llama-3.1 → Qwen2.5</td><td>Mistral-7B → Qwen2.5</td></tr><tr><td>Sender vocab size</td><td>128,256</td><td>32,768</td></tr><tr><td>Receiver vocab size</td><td>151,665</td><td>151,665</td></tr><tr><td>Shared tokens</td><td>109,566 (85.4%)</td><td>10,566 (32.2%)</td></tr><tr><td>Direct-mapped</td><td>97.1%</td><td>87.8%</td></tr><tr><td>String fallback</td><td>2.9%</td><td>12.2%</td></tr><tr><td>Dominant mismatch type</td><td>Multi-digit numbers</td><td>Words + proper nouns</td></tr></table>

Table 7 shows representative examples of tokenization mismatches and how the string-based fallback resolves them.

Table 7: Examples of cross-vocabulary mapping between different tokenizer pairs. Direct lookup preserves the token as-is; string fallback re-tokenizes the surface string, producing one-to-many expansions.
<table><tr><td>Source text</td><td>Sender tokens</td><td>Receiver tokens</td><td>Type</td></tr><tr><td colspan="4">Llama-3.1 → Qwen2.5</td></tr><tr><td> $\mathrm { \ " { t h e } } ^ { , \mathrm { * } }$  “transformer&quot;  $^ { 6 6 } 1 9 9 ^ { , 9 }$ </td><td>[the] (1) [transform, er] (2) [199](1)</td><td>[the] (1) [transform, er] (2) [1, 9, 9] (3)</td><td>Direct Direct Fallback</td></tr><tr><td> $\mathrm { ^ { * } C O V I D { - } 1 9 ^ { \circ } }$  Mistral-7B → Qwen2.5</td><td>[COVID, -, 19] (3)</td><td> $\left[ \mathrm { C O V I D } , \textrm { - } , 1 , 9 \right] \left( 4 \right)$ </td><td>Mixed</td></tr><tr><td colspan="4"></td></tr><tr><td> $\mathrm { \ " { o d i r e c t e d } \mathrm { \cdots } }$ </td><td>[directed] (1)</td><td>[direct, ed] (2)</td><td>Fallback</td></tr><tr><td> $\mathrm { \displaystyle { \tilde { \hbar } A n g e l e s } ^ { \prime } }$ </td><td>[Angeles] (1)</td><td>[An, ge, les] (3)</td><td>Fallback</td></tr><tr><td> $^ { \circ } \mathrm { c e n t u r y } ^ { \prime \prime }$ </td><td>[century] (1)</td><td>[cent, ury] (2)</td><td>Fallback</td></tr></table>

The mismatch patterns reflect fundamental differences in tokenizer design. Llama-3.1 uses a 128K BPE vocabulary that overlaps substantially with Qwen2.5’s 152K vocabulary, resulting in mismatches confined almost entirely to multi-digit numbers. Mistral-7B uses a smaller 32K SentencePiece vocabulary, causing common English words and proper nouns to fall back to string-based re-tokenization. In both cases, the fallback is lossless: the surface string content is fully preserved, with only positional expansion as a side effect.

## A.2 Grounding Format Independence

A critical question is whether LEB’s effectiveness depends on the specific grounding format (embed ding prepend vs. text prompt) or merely on the presence of entity anchors.

LEB consistently improves all grounding formats (+12.5 to +22.3 pp), demonstrating that entity presence, not format, is what enables LEB. Whether entity anchors arrive as prepended embeddings, as text within a chat template, or as argmax-decoded tokens, LEB extracts contextual enrichment from the sender’s $H _ { S }$ via cross-attention regardless of how those anchors enter the receiver.

The smaller $\Delta$ for NLComm (+12.5 pp) reflects its higher base: NLComm text already contains sender reasoning, leaving less room for LEB improvement. Notably, NLComm with LEB (81.0%) surpasses LAM with LEB (78.8%), because the sender’s autoregressive decoding performs reasoning compression that benefits the receiver. However, LAM remains the preferred grounding format because it eliminates decoding latency entirely (14× faster than NLComm) while achieving comparable performance with LEB.

Table 8: Grounding format comparison on HotpotQA, Llama→Qwen (F1 %). Each grounding format is evaluated with and without LEB to isolate the bridge contribution (∆).
<table><tr><td>Grounding format</td><td>Without LEB</td><td>With LEB</td><td>LEB∆</td></tr><tr><td>LAM</td><td>56.5</td><td>78.8</td><td>+22.3</td></tr><tr><td>NLComm text (chat prompt)</td><td>68.5</td><td>81.0</td><td>+12.5</td></tr><tr><td>Argmax token map</td><td>44.5</td><td>65.5</td><td>+21.0</td></tr></table>

Table 9: Bridge architecture ablations on HotpotQA, Llama→Qwen (F1 %). All variants use balanced training (587 samples).
<table><tr><td>Ablation</td><td>Variant</td><td>F1</td></tr><tr><td rowspan="3">Modules</td><td>1 module (66M)</td><td>66.3</td></tr><tr><td>2 modules (132M)</td><td>74.6</td></tr><tr><td>4 modules (264M) 7 modules (462M)</td><td>78.8 75.8</td></tr><tr><td>Sender layer</td><td>Layer 8 Layer 16 Layer 24 Layer 31 (last)</td><td>64.6 66.7 70.2 78.8</td></tr><tr><td>Gate init</td><td>α = 0.0 (Flamingo) α = 1.0 (ours)</td><td>72.7 78.8</td></tr><tr><td>Loss</td><td>NTP only (ours) NTP + contrastive</td><td>78.8 78.7</td></tr></table>

## B LEB Design

## B.1 Bridge Architecture Ablations

We ablate bridge design decisions on HotpotQA (Llama→Qwen). All variants use balanced training (587 samples).

Number of bridge modules. We vary the number of cross-attention modules inserted into the receiver (Qwen2.5-7B, 28 layers). 4 modules (264M parameters) is optimal. 7 modules (462M) overfits due to the higher parameter-to-data ratio with balanced training. 2 modules (132M) provides insufficient capacity for cross-architecture translation. This follows Flamingo’s [9] finding that sparser insertion is preferred for larger models.

Sender layer selection. Bridge performance increases monotonically from early to final sender layers, with layers before the midpoint falling below the no-bridge baseline. The last layer is optimal because it integrates information across all context positions through the sender’s full attention stack, producing the most globally-aware representations.

Gate initialization. We initialize the tanh gate at $\alpha = 1 . 0 ( \operatorname { t a n h } ( 1 . 0 ) \approx 0 . 7 6 )$ , differing from Flamingo’s zero initialization. With α = 0, the bridge is initially invisible to the receiver, requiring training to “discover” the bridge signal. With α = 1.0, the bridge immediately contributes crossarchitecture context. In our setting with limited training data (587 samples), warm initialization converges faster and achieves higher final performance.

Auxiliary losses. Adding a contrastive loss (encouraging the bridge to distinguish correct sender context from random contexts) does not improve performance beyond NTP loss alone. The NTP objective already implicitly requires the bridge to extract useful information from the sender’s hidden states.

Table 10: Effect of training data composition on LEB performance (Llama→Qwen, F1 %).
<table><tr><td>Bridge training</td><td>HotpotQA</td><td>Tipsheets</td><td>Countries</td><td>QASPER</td><td>Avg (7 tasks)</td></tr><tr><td>HotpotQA only (20K)</td><td>76.1</td><td>88.3</td><td>27.7</td><td>36.2</td><td>49.5</td></tr><tr><td>Unbalanced (42K)</td><td>77.1</td><td>100.0</td><td>66.0</td><td>45.3</td><td>59.3</td></tr><tr><td>Balanced (587)</td><td>78.8</td><td>99.8</td><td>72.5</td><td>47.5</td><td>63.2</td></tr></table>

Table 11: Same-architecture comparison (F1 %). XBRIDGE uses Qwen2.5-7B as both sender and receiver with a separately trained bridge. FullComm = receiver reads full context directly (identical to Table 1). Bold = best per task (excluding FullComm). Improv. = improvement of XBRIDGE<sub>homo</sub> over the second-best method (excluding FullComm).
<table><tr><td>Method</td><td colspan="6">Countries Tipsheets HotpotQA QASPER MuSiQue MFldQA 2Wiki Avg</td></tr><tr><td colspan="7">Ms: Qwen2.5-7B-Instruct; MR: Qwen2.5-7B-Instruct</td></tr><tr><td>FullComm</td><td>50.8</td><td>78.4</td><td>31.3</td><td>31.5</td><td>48.2</td><td>47.6</td><td>55.2</td></tr><tr><td>NLCommhomo</td><td>37.2</td><td>98.5 96.0 64.9</td><td>30.8</td><td>29.1</td><td>23.9</td><td>18.2</td><td>42.9</td></tr><tr><td>KVComm</td><td>35.8</td><td>99.0</td><td>65.3 35.5</td><td>33.1</td><td>48.3</td><td>40.9</td><td>51.1</td></tr><tr><td>XBRIDGEhomo</td><td>72.7</td><td>99.8 87.1</td><td>49.0</td><td>38.7</td><td>45.6</td><td>48.9</td><td>63.1</td></tr><tr><td>Improv.</td><td>+35.5</td><td>+0.8</td><td>+21.8 +13.5</td><td>+5.6</td><td>-2.7</td><td>+8.0</td><td>+12.0</td></tr></table>

## B.2 Bridge Training Data Composition

The composition of bridge training data significantly affects per-task performance.

HotpotQA-only bridge (20K samples) is inferior even on its own training domain (76.1% vs. 78.8% for the balanced bridge), and catastrophically degrades on unseen task types: Countries drops to 27.7% and QASPER to 36.2%. Overfitting to a single task’s context-length distribution and reasoning pattern prevents the bridge from generalizing.

Unbalanced universal bridge (42K samples, 94% HotpotQA+MuSiQue) improves over the singletask bridge across all tasks but remains below the balanced bridge on average (59.3% vs. 63.2%). The heavy skew toward multi-hop QA patterns limits generalization to dissimilar tasks such as Countries and QASPER.

Balanced universal bridge (587 samples, ∼100 per task) achieves the best performance on every metric despite using 30× less training data than the unbalanced variant. The balanced bridge learns task-appropriate gating: it remains near-closed for near-saturated factual tasks (Tipsheets 99.8%) while actively contributing on complex reasoning tasks (HotpotQA 78.8%, Countries 72.5%). The average across all 7 tasks (63.2%) is the highest among all bridge variants.

Implication for sender size scaling. The balanced training set (587 samples) is sufficient for 7B senders but becomes a bottleneck for larger models. The 14B sender (hidden dim 5120) produces richer representations that require more training data to learn effective projections. Within this data budget, the 14B bridge plateaus at 62.1%, matching the 7B sender (61.9%) despite its larger capacity. We expect that scaling the balanced training set would unlock the 14B sender’s advantage, but leave this to future work.

## C Additional Results

## C.1 Homogeneous Comparison

To verify that XBRIDGE generalizes beyond heterogeneous settings, we train a same-architecture bridge using Qwen2.5-7B as both sender and receiver. LAM reduces to an identity mapping in this setting, so the comparison isolates LEB’s contribution on top of full context reading.

Table 12: Bridge contribution per task (Llama→Qwen, F1 %). XBRIDGE $_ { \mathrm { \cdot L E B } } = \mathrm { L A M }$ only. $\Delta =$ XBRIDGE − XBRIDGE<sub>-LEB</sub>.
<table><tr><td>Task</td><td>Type</td><td> $\mathbf { X B R I D G E _ { \mathrm { - L E B } } }$ </td><td>XBRIDGE</td><td> $\Delta$ </td></tr><tr><td>Countries</td><td>factual inference</td><td>36.7</td><td>72.5</td><td>+35.8</td></tr><tr><td>HotpotQA</td><td>multi-hop QA</td><td>56.5</td><td>78.8</td><td>+22.3</td></tr><tr><td>2WikiMQA</td><td>multi-hop QA</td><td>29.5</td><td>51.1</td><td>+21.6</td></tr><tr><td>MuSiQue</td><td>compositional QA</td><td>28.4</td><td>48.2</td><td>+19.8</td></tr><tr><td>QASPER</td><td>scientific QA</td><td>33.0</td><td>47.5</td><td>+14.5</td></tr><tr><td>MFldQA</td><td>long-doc retrieval</td><td>37.3</td><td>44.2</td><td>+7.0</td></tr><tr><td>Tipsheets</td><td>structured extract.</td><td>97.8</td><td>99.8</td><td>+2.0</td></tr></table>

Table 13: Multi-agent communication on HotpotQA, Llama→Qwen (F1 %). Two senders each process one supporting document; the receiver integrates both.
<table><tr><td>Configuration F1</td></tr><tr><td>Single sender, full context (Llama) 67.0</td></tr><tr><td>Dual NLComm (Llama + Mistral) 56.8</td></tr><tr><td>Dual XBRIDGE (Llama + Mistral) 70.4</td></tr></table>

Table 11 compares $\mathrm { X B R I D G E _ { h o m o } }$ against $\mathrm { N L C o m m } _ { \mathrm { h o m o } }$ and KVComm [5]. $\mathrm { X B R I D G E _ { h o m o } }$ outperforms NL $\mathrm { . C o m m _ { h o m o } }$ on all 7 tasks and KVComm on 6 of 7 tasks, with an average improvement of +12.0 pp over the stronger baseline. Notably, XBRIDGE<sub>homo</sub> surpasses even FullComm on 6 of 7 tasks (+7.9 pp on average), demonstrating that LEB adds value beyond what the receiver extracts from reading the context alone. The sole exception is MultiFieldQA, consistent with the pattern observed in heterogeneous settings (Section 4.3).

## C.2 Bridge Contribution per Task

Table 12 breaks down LEB’s contribution across all seven tasks for the Llama→Qwen pair. LEB contribution scales with the reasoning complexity of the task.

On Countries, the context states a person’s location at a landmark (e.g., “Eve is at the Uppland Runic Inscription 896”) and the question asks which country the person is in, but the answer (“Sweden”) appears nowhere in the context. Only LEB can transmit the sender’s implicit landmark-to-country inference, yielding the largest contribution (+35.8 pp). On multi-hop tasks (HotpotQA, 2WikiMQA, MuSiQue), LEB transmits reasoning chains integrated across documents (+19 to +22 pp).

On MultiFieldQA, the question asks for a specific fact (e.g., “What is the advantage of decorrelating the data before running the PLS algorithm $1 \hat { ? } \hat { > } )$ and the answer is a single extractable sentence located deep within a ∼7K-token document. No cross-entity reasoning is needed, and LEB adds only $+ 7 . 0 \mathrm { p p }$ . This is also the sole task where KVComm [5] outperforms XBRIDGE (48.3 vs. 44.2): same-architecture KV sharing preserves position-level attention patterns that directly encode answer location, an advantage specific to retrieval tasks that disappears when reasoning is required (e.g., Hotpot $2 \mathrm { A } + 1 3 . 5 \mathrm { p p }$ , MuSiQue +15.1 pp over KVComm).

On Tipsheets, the answer is directly extractable from structured text, leaving minimal room for LEB improvement (+2.0 pp).

## C.3 Multi-Agent Communication

We test composability on HotpotQA, where each question requires reasoning over two supporting documents. We assign each document to a separate sender, Llama for the first and Mistral for the second, and let the receiver (Qwen) integrate both via their respective bridges. Each bridge was trained independently for its own model pair; no joint training or adaptation is performed.

Dual XBRIDGE achieves 70.4% F1 (Table 13), outperforming both the single-sender full-context baseline (67.0%, +3.4 pp) and dual NLComm (56.8%, +13.6 pp). The single-sender baseline gives one Llama sender both documents, so the dual setup’s improvement demonstrates that distributing documents across specialized senders is more effective than compressing everything through a single sender. This zero-shot composability, combining independently trained bridges without additional training, suggests that the framework scales naturally to multi-party agent settings.

## D Additional Analysis

## D.1 Per-Sample Consistency

The bridge mechanism analysis in Section 4.4 reports aggregate metrics across 196 HotpotQA samples (Llama→Qwen). Here we verify that these improvements are consistent at the individual sample level rather than driven by a few outliers.

Figure 4(c) plots the mean-centered cosine similarity between the generation representation and the answer entity representation for each sample, comparing NoComm and XBRIDGE. Points above the $y = x$ line indicate samples where LEB improves alignment. Of 196 samples, 184 (94%) fall above $y = x \left( p \approx 0 \right.$ , paired t-test). The population mean shifts from µ = −0.053 (NoComm) to $\mu = + 0 . 0 9 1$ (XBRIDGE), a sign change indicating that without LEB the receiver’s generation state moves away from the answer entity, while with LEB it moves toward it.

The 12 samples (6%) where LEB does not improve alignment tend to involve short, unambiguous contexts where LAM alone already provides sufficient grounding. Even in these cases, LEB does not substantially degrade performance, indicating that the gating mechanism (Equation 11) successfully attenuates the bridge signal when it is not needed.

## D.2 Entity Perturbation Examples

Table 14 shows representative examples from the entity perturbation experiment (Section 4.4). In each case, swapping LEB input alone does not change the output entity, while swapping LAM changes it entirely.

Table 14: Representative entity perturbation examples on HotpotQA (Llama→Qwen). The answer entity in the context is replaced with an unrelated entity of the same type. Condition A: both original. Condition D: LEB swapped only. Condition C: LAM swapped only.
<table><tr><td>Question</td><td>Original</td><td>Swapped</td><td>A</td><td>D</td><td>C</td></tr><tr><td>Which port city lies ~25 km north of the Lingnan Fine Arts Museum?</td><td>Keelung</td><td>Majuro</td><td>Keelung</td><td>Keelung</td><td>Majuro</td></tr><tr><td>Who did Neleus of Scepsis study un- der in addition to Theophrastus?</td><td>Aristotle</td><td>Moroni</td><td>Aristotle</td><td>Aristotle</td><td>Moroni</td></tr><tr><td>Who will Billy Howle be seen oppo- site in the upcoming British drama film directed by Dominic Cooke?</td><td>Saoirse Ronan</td><td>Kenji Mizoguchi</td><td>Saoirse Ronan</td><td>Saoirse Ronan</td><td>Kenji Mizoguchi</td></tr></table>

In all three examples, conditions A and D produce identical outputs, confirming that LEB does not influence entity selection. Condition C shifts the output to the swapped entity every time. LEB adapts its contextual enrichment accordingly: contextual signals such as “port city 25 km north of the museum” or “studied under in the Peripatetic school” bind to whichever entity LAM provides.

## E Task Details

Table 15 summarizes the seven benchmarks used in our evaluation. Training and evaluation splits are strictly separated with zero overlap.

## F Related Work

Text-based multi-agent communication. LLMs are increasingly deployed in multi-agent systems where agents collaborate to solve tasks beyond individual capability [16, 17]. Multi-agent debate improves factuality by having models critique each other’s reasoning [4, 18], while frameworks such as CAMEL [19], AutoGen [20], and ChatDev [21] coordinate agents through structured natural language exchanges. Natural language communication (NLComm) [4] generalizes this pattern: one model generates a text summary and relays it to another. These text-based protocols are architectureagnostic but require autoregressive decoding at every communication step, incurring high latency. They also discard the sender’s internal representations, compressing context into a short summary that loses both the sender’s contextual understanding and relational structure among entities.

Table 15: Task details. Context length statistics are measured across evaluation samples using the Llama-3.1 tokenizer.
<table><tr><td>Task</td><td>Type</td><td>Eval N</td><td>Median tok</td><td>Mean tok</td><td>Source</td></tr><tr><td>HotpotQA</td><td>multi-hop QA</td><td>200</td><td>74</td><td>80</td><td>Yang et al. [12]</td></tr><tr><td>MuSiQue</td><td>compositional QA</td><td>200</td><td>~1K*</td><td>~1.2K*</td><td>Trivedi et al. [13]</td></tr><tr><td>QASPER</td><td>scientific QA</td><td>200</td><td>4,118</td><td>4,514</td><td>Dasigi et al. [14]</td></tr><tr><td>2WikiMQA</td><td>multi-hop QA</td><td>100</td><td>6,532</td><td>7,267</td><td>LongBench [15]</td></tr><tr><td>MultiFieldQA-en</td><td>long-doc retrieval</td><td>75</td><td>6,979</td><td>6,878</td><td>LongBench [15]</td></tr><tr><td>Countries</td><td>factual inference</td><td>100</td><td>11</td><td>11</td><td>Shi et al. [5]</td></tr><tr><td>Tipsheets</td><td>structured extract.</td><td>200</td><td>63</td><td>62</td><td>Shi et al. [5]</td></tr></table>

<sup>∗</sup>MuSiQue statistics shown are for supporting paragraphs only; full context with distractors is approximately 4–5× longer. All evaluations use the full context.

Latent multi-agent communication. To avoid the cost and information loss of text-based communication, recent work transmits internal model representations directly. CIPHER [8] communicates via soft tokens across models but requires a shared tokenizer. AC [6] shows that activation-space interventions transfer between LLMs of the same family. C2C [7] projects and fuses KV caches across model families through learned linear projections, achieving cross-architecture transfer but operating entirely in continuous space. KVComm [5] shares selectively chosen KV cache layers, achieving near-FullComm performance with reduced communication cost, but assumes matching hidden dimensions, attention heads, and positional encodings. These methods differ not only in what they transmit but in what they assume the two models share. C2C fuses caches position-wise and therefore has the sharer and receiver process the same context, and LatentMAS [1] concatenates layer-wise KV caches across agents under a common architecture. Both are symmetric in the sense that the receiver already holds the context. XBridge operates in the asymmetric setting of KVComm, where the sender observes a context the receiver does not, and it is this asymmetry that gives rise to the entity grounding problem: once the receiver can no longer see the entities directly, a discrete channel aligned to its own vocabulary becomes necessary rather than supplementary. Existing latent protocols therefore either assume architectural homogeneity, require a shared tokenizer, or place the two models on the same context, and none preserves discrete entity identity across different vocabularies. XBridge addresses this gap by enabling latent communication across different LLM architectures while preserving entity fidelity.