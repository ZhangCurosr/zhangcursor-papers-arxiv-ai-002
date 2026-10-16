# MEMORILLA: LATENT SEMANTIC MEMORY FOR LLMS

Adibvafa Fallahpour<sup>1,2,5,6,∗,†</sup> Parsa Idehpour<sup>3,6,∗,†</sup> Vignesh Kothapalli<sup>1,∗,†</sup> Nikita Mounier<sup>3</sup> Shirley Wu<sup>1</sup> Shayan Pardis<sup>4</sup> Jure Leskovec<sup>1,†</sup>

<sup>1</sup>Stanford University <sup>2</sup>University of Toronto <sup>3</sup>University of Pennsylvania <sup>4</sup>MIT <sup>5</sup>Vector Institute <sup>6</sup>Arc Institute

Website: https://memory.stanford.edu/

## ABSTRACT

Memory enables large language models (LLMs) to use information beyond their parameters, such as past user conversations, long documents, or an agent’s earlier actions. For a frozen LLM, access to this accumulated knowledge is limited to what a memory system places in its context window. However, gathering and combining relevant information needed for each query while keeping the model’s context short remains challenging for existing systems. Here we introduce MEMORILLA, a memory system that gathers information from an entire document collection into a fixed number of query-dependent ‘memory tokens’ for a frozen LLM. MEMORILLA encodes documents (such as user chats, book chapters) once into reusable embeddings, then learns to retrieve and place query-specific information in the LLM’s input embedding space. Using only 16 memory tokens, MEMORILLA achieves the best average performance against baselines including Mem0 and MemoRAG across benchmarks spanning personalization, long-document and knowledge-intensive question answering. Notably, on FactKG, it reaches 84.33% accuracy versus 80.43% with full-context inference, while using 108× fewer tokens to represent the source documents. Furthermore, MEMORILLA can facilitate continual learning in agentic systems where the document count (the agent’s past turns) increases throughout the trajectory. In a TextWorld environment, we show that MEMORILLA reduces the agent’s tendency to wander between rooms it has already explored, and improves its average success rate from 18.6% with full-history in-context learning to 24.2%. Overall, MEMORILLA aids frozen LLMs in accumulating and using knowledge effectively across long-context interactions and takes a step towards building continually learning AI systems.

## 1 INTRODUCTION

Memory is central to how large language models (LLMs) are used. Personal assistants must recall a user’s history across months of conversations, question-answering systems must reason over documents far longer than the model’s context window, and agents must act on everything they have observed. None of this information exists in the model’s parameters, and a frozen LLM can use it only through its context (Hu et al., 2025). A memory system bridges this gap by encoding accumulated information, storing it persistently, and retrieving what each query needs into the context (Wu et al., 2025). The central challenge for such a system is to give a frozen LLM access to an entire, growing collection of knowledge while keeping its context short (An et al., 2025; Du et al., 2025).

Existing memory systems address these issues only in part. Agentic memory systems such as MemGPT (Packer et al., 2023), Mem0 (Chhikara et al., 2025) and A-MEM (Xu et al., 2025) organize past interactions into summaries or structured records, and MemoRAG (Qian et al., 2025) uses a compressed global memory to guide retrieval, but each ultimately places selected text in the context and inherits the limits of selection. Latent compression methods such as ICAE (Ge et al., 2024), IC-Former (Wang et al., 2024a), AutoCompressors (Chevalier et al., 2023) and ComMer (Zeldes et al., 2025) represent documents as a few soft tokens in the LLM embedding space, but they form these tokens without the query, so the same tokens must serve every question. xRAG (Cheng et al., 2024) incorporates query relevance but only by compressing a single retrieved document. Methods that store model-specific hidden or key–value states, such as LongMem (Wang et al., 2023), MemoryLLM (Wang et al., 2024b), M+ (Wang et al., 2025), and Cartridges (Eyuboglu et al., 2026), tie the memory to one model and require training the LLM, a side network, or a separate cache for each corpus. What is missing is a memory system that will retrieve query-relevant information from document embeddings directly into the input embedding space of a frozen LLM. Benefits of such an approach would be that the frozen LLM can retain all of its capabilities, and rely on only a few query relevant latent embeddings to answer questions or solve tasks instead of processing long textual contexts in a slow and lossy manner (Liu et al., 2024). In essence, improving the speed and quality of LLM responses and also the decision making in LLM agents based on past experiences.

![](images/a7f821132cbbfb0b0c6d80e55dca907a1830da3159dd28f9b31e7469e569c221.jpg)  
Figure 1: Performance versus context length. Mean score over eight benchmarks against the geometric mean of document tokens given to a frozen Qwen3-8B decoder. MEMORILLA lies on the Pareto frontier (dashed) with the highest score at 16 memory tokens, 394× fewer than full context.

Present work. Here we introduce MEMORILLA, a memory system that learns to retrieve queryrelevant information from document embeddings directly into the input embedding space of a frozen LLM (Figure 2). First, a frozen embedding model encodes each document, such as a user conversation or a book chapter, once into a persistent memory store. Next, for each query, a trainable memory module attends to every embedding in the persistent memory store and performs a query conditioned memory readout for the LLM. Unlike RAG which retrieves a subset of documents as textual tokens, MEMORILLA learns to recall what matters from the entire collection for a query, and expresses it as K = 16 latent memory tokens for a frozen LLM, and does not require the documents to enter the context as text. Because the store uses the same embeddings as a dense retrieval index, encoding adds no cost over retrieval, and with 512 documents on one H100, MEMORILLA has 27× lower end-to-end latency than full context answer generation (0.74s against 19.9s, Appendix F).

Our benchmarks mirror three settings in which deployed LLMs need memory (Figures 1 and 3). In personalization, an assistant must adapt to preferences a user rarely states directly, whether answering questions that depend on traits revealed across past conversations (PersonaMem-v2, PersonalizationV4) or writing in the style of the user’s earlier text (LaMP-4, LaMP-7). Because a user’s preferences rarely share wording with the query, similarity-based retrieval struggles to find and combine relevant information. Notably, on LaMP-4, MEMORILLA reaches 18.15 ROUGE-L against 14.21 for full context while using 474× fewer document tokens. In knowledge-intensive question answering, such as checking a claim against a knowledge base, MEMORILLA outperforms full context on FactKG (84.33% against 80.43%) with 108× fewer document tokens. In long-document understanding, the answer is buried in thousands of tokens of text and often phrased differently from the question, so a system must find and connect relevant content across the entire document. On

![](images/42adf8dc6484a28642c7289ca364dd869c1527a1e04ffd81b1b152054a50eb0b.jpg)  
Figure 2: MEMORILLA architecture. A frozen encoder embeds source documents (User Chats) once into a persistent memory store. For each Query, K query-conditioned slots attend over the store and are projected into the frozen LLM’s embedding space as a memory readout.

TriviaQA, whose questions were written independently of their source pages, MEMORILLA reaches the highest exact match (57.00) with 743× fewer document tokens than full context.

Beyond answering questions, MEMORILLA allows agents to learn continually from their own experi ence. In this work, we consider the TextWorld environment (Côté et al., 2018) where an agent must explore rooms, find and use objects, and complete a multi-step goal within 50 turns, which requires remembering where it has been and what it has already tried. After every turn, MEMORILLA adds that turn to its document store and draws on this experience at the next turn. With our approach, the agent completes 37.5% of held-out games in at least one of three attempts, against 25.0% for the same agent given its full history in context. Overall, MEMORILLA improves the agent’s ability to escape navigation loops and execute action sequences that are crucial to reaching the target.

## 2 MEMORILLA

A persistent latent memory must preserve a document collection across queries while recalling different information for each query. MEMORILLA meets these requirements through three operations: encoding, storage, and retrieval. Given N document chunks $\mathcal { D } = \{ d _ { 1 } , \ldots , d _ { N } \}$ , a frozen encoder produces a query-independent representation that is stored once. For instance, D can be a collection of N user-LLM conversations, chapters of a lengthy book, or health records. For each query q, a trainable memory module reads this store and returns K query-conditioned soft memory tokens, which a frozen LLM consumes together with the query text. Figure 2 summarizes the architecture.

## 2.1 ENCODING AND PERSISTENT STORAGE

A frozen document encoder $f _ { \mathrm { e n c } } ^ { d }$ maps each chunk $d _ { i }$ to an embedding $\mathbf { z } _ { i } \in \mathbb { R } ^ { h _ { \epsilon } }$ <sup>e</sup>. Stacking these embeddings produces the query-independent document store

$$
\begin{array} { r } { \mathbf { Z } _ { \mathcal { D } } = [ f _ { \mathrm { e n c } } ^ { d } ( d _ { 1 } ) , \dots , f _ { \mathrm { e n c } } ^ { d } ( d _ { N } ) ] \in \mathbb { R } ^ { N \times h _ { e } } . } \end{array}\tag{1}
$$

Because $\mathbf { Z } _ { \mathcal { D } }$ does not depend on a query, it is computed once and persisted in memory or on secondary storage. Every subsequent query reuses the same matrix, and only added or changed documents must be re-encoded. A frozen query encoder $f _ { \mathrm { e n c } } ^ { q }$ then maps q to $\mathbf { z } _ { q } = f _ { \mathrm { e n c } } ^ { q } ( q ) \in \mathbb { R } ^ { \breve { h _ { e } } }$ . Unlike $\mathbf { Z } _ { \mathcal { D } } , \mathbf { z } _ { q }$ is computed only for the current query and is not stored as long-term memory.

## 2.2 QUERY-CONDITIONED RETRIEVAL $( f _ { \mathrm { M E M } } )$

In the next phase, a trainable memory module $f _ { \mathrm { m e m } }$ uses $\mathbf { z } _ { q }$ to select information from $\mathbf { Z } _ { \mathcal { D } } \mathbf { : }$

$$
f _ { \mathrm { m e m } } : ( \mathbf { Z } _ { \mathcal { D } } , \mathbf { z } _ { q } ) \mapsto \mathbf { M } , \mathbf { M } \in \mathbb { R } ^ { K \times h _ { L } } ,\tag{2}
$$

where $h _ { L }$ is the input-embedding dimension of the frozen LLM and $\mathbf { M } = [ \mathbf { m } _ { 1 } ; \dots ; \mathbf { m } _ { K } ]$ is the output memory matrix. Each row $\mathbf { \bar { m } } _ { k } \in \mathbb { R } ^ { h _ { L } }$ is one soft memory token in the LLM embedding space. The module reads all N stored document embeddings but always contributes exactly K memory tokens to the LLM context, and allows significant context compression (over 4000×, see Table 11).

Stage 1: Memory slot initialization. The module initializes K memory slots with learned vectors $\mathbf { Q } _ { 0 } \in \mathbb { R } ^ { K \times h _ { e } }$ that are shared across collections and queries. Distinct initial values define a fixed output budget while allowing the slots to capture different information.

Stage 2: Query conditioning. The slots attend to the query embedding:

$$
{ \bf Q } _ { c } = \mathrm { L a y e r N o r m } ( { \bf Q } _ { 0 } + \mathrm { C r o s s A t t n } ( Q { = } { \bf Q } _ { 0 } , K { = } { \bf z } _ { q } , V { = } { \bf z } _ { q } ) ) .\tag{3}
$$

Because the query is represented by one vector, it provides a shared update to all slots; the residual connection preserves the differences among $\mathbf { Q } _ { 0 }$ . The resulting $\mathbf { Q } _ { c }$ is therefore query-conditioned while retaining distinct slot identities.

Stage 3: Document retrieval. The conditioned slots attend to every document embedding through $L _ { \mathrm { r e f } }$ cross-attention blocks, following the latent-array design of Perceiver (Jaegle et al., 2021). Let $\overline { { \mathbf { Z } } } _ { \mathcal { D } } = \mathop { \mathrm { L a y e r N o r m } } ( \mathbf { Z } _ { \mathcal { D } } )$ and $\mathbf { R } _ { 0 } = \mathbf { Q } .$ <sub>c</sub>. For block ℓ,

$$
\begin{array} { r l } & { \widetilde { \mathbf { R } } _ { \ell } = \mathbf { R } _ { \ell - 1 } + \mathrm { C r o s s A t t n } \big ( Q { = } \mathrm { L a y e r N o r m } ( \mathbf { R } _ { \ell - 1 } ) , K { = } \overline { { \mathbf { Z } } } _ { \mathcal { D } } , V { = } \overline { { \mathbf { Z } } } _ { \mathcal { D } } \big ) , } \\ & { \mathbf { R } _ { \ell } = \widetilde { \mathbf { R } } _ { \ell } + \mathrm { F F N } \Big ( \mathrm { L a y e r N o r m } ( \widetilde { \mathbf { R } } _ { \ell } ) \Big ) , } \end{array}\tag{4}
$$

where FFN is a two-layer feed-forward network with GELU activations (Hendrycks & Gimpel, 2016). Each slot can attend to the full collection, while the query-conditioned slots determine which information is retained. The refinement blocks therefore implement a continuous, query-conditioned read over all stored documents rather than a discrete top-k selection.

Stage 4: Projection to the LLM embedding space. A learned projection maps the refined slots from the encoder dimension $h _ { e }$ to the LLM input-embedding dimension $h _ { L }$

$$
\begin{array} { r } { \mathbf { M } = \mathrm { L a y e r N o r m } ( \mathbf { R } _ { L _ { \mathrm { r e f } } } ) \mathbf { W } _ { \mathrm { p r o j } } , \qquad \mathbf { W } _ { \mathrm { p r o j } } \in \mathbb { R } ^ { h _ { e } \times h _ { L } } . } \end{array}\tag{5}
$$

The resulting rows $\mathbf { m } _ { 1 } , \ldots , \mathbf { m } _ { K }$ are the soft memory tokens passed to the frozen LLM.

## 2.3 MEMORY INJECTION INTO THE FROZEN LLM

Finally, the LLM input-embedding layer maps the text query to $\mathbf { P } _ { q } .$ , giving

$$
\begin{array} { r } { \mathbf { X } _ { \mathrm { L L M } } = [ \underbrace { \mathbf { m } _ { 1 } , \dots , \mathbf { m } _ { K } } _ { \mathrm { s o f t m e m o r y t o k e n s } } ; \mathbf { P } _ { q } ] . } \end{array}\tag{6}
$$

Because $\mathbf { m } _ { k } \in \mathbb { R } ^ { h _ { L } }$ has the same dimension as a text token embedding, the LLM processes memory and text in the same input sequence. No LLM layer or attention mechanism is changed. Rotary position embeddings are applied over the combined sequence in the standard way (Su et al., 2021). The frozen LLM can therefore attend to the recalled memory through its existing self-attention layers.

## 2.4 SUPERVISED TRAINING OBJECTIVE

The frozen LLM generates $\mathbf { Y _ { \mathrm { o u t } } } = ( y _ { 1 } , \dots , y _ { L } )$ autoregressively from $\mathbf { X } _ { \mathrm { L L M } }$ , and we minimize the negative log-likelihood of the target answer tokens:

$$
\mathcal { L } _ { \mathrm { s u p } } = - \sum _ { i = 1 } ^ { L } \log P ( y _ { i } \mid \mathbf { y } _ { < i } , \mathbf { X } _ { \mathrm { L L M } } ; \theta _ { \mathrm { m e m } } ) ,\tag{7}
$$

where $\theta _ { \mathrm { { m e m } } }$ includes the memory module and its projection to the LLM embedding space. The loss is applied only to answer positions, not to the prompt or memory positions.

## 2.5 MEMORILLA WITH QWEN3

Based on the formalization above, we instantiate MEMORILLA with $f _ { \mathrm { e n c } } ^ { d }$ and $f _ { \mathrm { e n c } } ^ { q }$ as a frozen Qwen3- Embedding-4B model. Unless stated otherwise, we fix $K = 1 6$ throughout the paper and use a Qwen3-8B model as the frozen LLM. This gives us 194,123,776 trainable parameters.

## 2.6 TRAINING STAGES

We use a three-stage curriculum that moves from general content preservation to diverse language tasks and then to the evaluation domains. See Appendix B.2 for training hyperparameters.

Stage 1 trains the memory module on 1.98 million English Wikipedia paraphrase reconstruction examples. Each passage is divided into document chunks, and the module must preserve enough information in its fixed memory budget for the frozen LLM to generate a paraphrase. This stage provides a general initialization before the module encounters task-specific supervision.

Stage 2 continues training on 1.34 million examples drawn from ten datasets covering reading comprehension, open-domain question answering, and summarization. The variation in source structure and answer format teaches the module to recall different information from the same type of stored representation. Table 1 lists the component datasets and their sizes.

Stage 3 fine-tunes the resulting module on a unified mixture of the training splits from all eight evaluation benchmarks. One multi-task checkpoint is used directly for FactKG, TriviaQA, NarrativeQA, PubMedQA, LaMP-4, and LaMP-7. PersonaMem-v2 and PersonalizationV4 receive additional benchmark-specific fine-tuning because their validation losses required more epochs to plateau, while the other benchmarks plateaued or began to overfit earlier.

## 3 EXPERIMENTS

## 3.1 BASELINES

We compare MEMORILLA against RAG approaches based on cosine similarity, Mem0 (Chhikara et al., 2025), and MemoRAG (Qian et al., 2025). xRAG (Cheng et al., 2024) and ComMer (Zeldes et al., 2025) (via its reported results, Table 12) compress the context into latent input embeddings for the LLM. Finally, we also compare against LLMLingua-2 (Pan et al., 2024), which is a task-agnostic text based prompt compression method. All baselines we run use a frozen Qwen3-Embedding-4B encoder for document embeddings and a frozen Qwen3-8B decoder for generation. Mem0 and MemoRAG additionally use a frozen Qwen3-4B-Instruct-2507 to build their memories, matching the 4B scale of the encoder with which MEMORILLA reads the same collections (Appendix C).

## 3.2 BENCHMARKS

We evaluate MEMORILLA on eight supervised benchmarks spanning (a) personalization, (b) longdocument QA, and (c) knowledge-intensive QA tasks that demand memory. For personalization, we consider PersonaMem-v2 (PMv2) (Jiang et al., 2025), LaMP-4, LaMP-7 (Salemi et al., 2023) and our own PersonalizationV4 (PV4) benchmarks as they require integrating information scattered across a user’s prior chats or writing history to form a coherent user representation. For long-document QA, we choose NarrativeQA (Kociskˇ y et al.\` , 2018) and TriviaQA (Joshi et al., 2017) to test the method’s ability to compress long documents in their entirety into a usable representation for downstream QA. Finally, for knowledge-intensive QA, we consider FactKG (Kim et al., 2023) and PubMedQA (Jin et al., 2019) to test factual grounding from structured or semi-structured evidence of varying quantity and format. Table 10 (Appendix D.2) reports how many document tokens each method gives the LLM per query across all the benchmarks.

## 3.3 EVALUATION

We use accuracy for classification tasks, exact match and token-level F1 (lowercased, whitespacetokenized) for short-form QA, and ROUGE, BLEU, F1, and METEOR for open-ended text generation. PMv2 and PV4 are evaluated on their short-answer questions. The model’s generation is embedded with the frozen encoder and scored as correct if the gold answer is its cosine nearest neighbor among the question’s predefined answer choices, which we report as short-answer accuracy. All supervised LLM evaluations use a decoding temperature of 0. Our serving stack enforces a 32,768-token decoder context and enables YaRN whenever an assembled full-context input exceeds this limit (Appendix D).

![](images/19686174a7f8926ce3331181281d7862df589c7eb6d641cb988e10c36161ed1f.jpg)  
(a) FactKG (Acc)

![](images/d935676ac56bef5a0944ae28e30e216c50f2b9b438537023443cb76cca6bc46b.jpg)  
(b) TriviaQA (EM)

● MemoRAG ★ MEMORILLA

![](images/5a566423f73b3c7277187bd8ca120c911e7ea2c24cb35d68741f3218138f29e5.jpg)  
(c) PMv2 (Acc)

![](images/76650b9ba2a1dbff81669e759baaa3f95abb6ae1e89169cd0ccde5cddf15d8e8.jpg)  
(d) LaMP-7 (ROUGE-L)

![](images/b83245b631471c59d2083260a77a79efd471209c764e5e9f80099dda8909a27c.jpg)  
(e) LaMP-4 (ROUGE-L)

![](images/9a5187b364912642f07725f581690ecb26d9625206323ae04f9a6d4a3a918f59.jpg)  
(f) NQA (ROUGE-L)

![](images/c0860470ed82cfce48a4fe9dac345e31dfc706799bedbb25e3037149ac9979d3.jpg)  
(g) PV4 (Acc)

![](images/97e29c1c4d1781f596ae9c960a9f1b38eb60cbd91ab15f57f8f1f380d2e07192.jpg)  
(h) PubMedQA (Acc)  
Figure 3: Per-benchmark performance versus context length. Primary-metric score against mean document tokens per query, with a frozen Qwen3-8B decoder. MEMORILLA lies on the Pareto frontier (dashed) of every benchmark, leads on five and is second on NarrativeQA and PV4.

## 4 QUANTITATIVE RESULTS

MEMORILLA outperforms all baselines on average while using 394×, 113× and 49× fewer document tokens than full context, RAG-50 and MemoRAG in geometric mean (Figure 1). It also lies on the Pareto frontier of every benchmark, while the compact baselines xRAG, LLMLingua-2 and Mem0 average only 41.92, 40.75 and 38.02 against 49.68. Figure 3 breaks this down by benchmark, where MEMORILLA leads on five and and is second on NarrativeQA and PV4.

Relational evidence. On FactKG (Figure 3a), MEMORILLA reaches 84.33% against 80.43% for full context, 80.38% for MemoRAG and 78.98% for xRAG, with 108× and 36× fewer tokens than the first two. Because full context receives every document, the margin cannot stem from discarded evidence, and Section 5 shows that MEMORILLA’s margin over RAG-50 grows with the number of entities and widens overall with the number of composed facts.

Long-document and personalization QA. On TriviaQA (Figure 3b), MEMORILLA reaches 57.00 EM against 56.60 for MemoRAG and 52.60 for full context with 19× and 743× fewer document tokens. On PMv2 short-answer (Figure 3c), MEMORILLA reaches 44.60% against 30.90% for xRAG, the strongest baseline. Every other baseline stays within about three points of closed-book (25.64%), including full context, which uses 1,974× more document tokens than MEMORILLA. On LaMP-7, every baseline except xRAG (46.84) stays within one ROUGE-L point of the closed-book model, whereas MEMORILLA improves it from 43.13 to 48.37. Finally, MEMORILLA achieves 18.15 ROUGE-L on LaMP-4 (Figure 3e) with 474× fewer tokens than the full context baseline.

Where memory does not help. On the remaining three benchmarks (Figures 3f–3h), a baseline leads, but only by supplying the LLM with 18 to 690 times more document tokens. MemoRAG leads NarrativeQA in ROUGE-1 and ROUGE-L (21.80 and 20.51 versus 19.68 and 19.31) with 50× more document tokens, consistent with its access to verbatim passages, and PV4 (67.88% versus 66.65%) with 690× more. On PubMedQA, whose four abstracts total only 324 tokens, LLMLingua-2 reaches 69.00% and four other baselines 67.00%, versus 59.00% for MEMORILLA.

Full context is not an upper bound. Full context trails MEMORILLA on every benchmark except PubMedQA and on 20 of 21 metrics in Table 11, and RAG-50 exceeds it on 15, so supplying the entire collection hurts more often than it helps.

![](images/7df35488d67e03e6d06f66ad2b9b3d5de945284cae0c9c0f753142c4978395f3.jpg)  
(a) NarrativeQA by question type

![](images/131b68bafc11295c5fa12cc014383dd9e0c96b9c0ab68e9512fa3c8cb880d786.jpg)  
(b) FactKG by entity count

![](images/c461f8a759172e989b1ee908e7b529a1c2d2e4880f3ae2d412f3ee5304b22a94.jpg)  
(c) FactKG by composition

Figure 4: Per-question breakdowns against RAG-50. (a) NarrativeQA question types are the question’s first word. Conceptual questions (‘Why’, ‘How’, ‘What’) favor MEMORILLA whereas entity-recall questions (‘Which’, ‘Who’) favor RAG. (b) MEMORILLA’s advantage grows with entity count, indicating better capture of multi-entity relational patterns. (c) RAG-50 degrades as more facts are composed, while MEMORILLA remains stable across all depths.  
![](images/bcbeaeb71d93798fd8413fc392ce2cff058c2e03c7549fd8bd49c33847c19b5f.jpg)  
Figure 5: Per-question complementarity between MEMORILLA and RAG-50. Share of questions answered by each method alone and by at least one of them (union), an oracle upper bound for a hybrid system. The union exceeds either method on every benchmark. <sup>†</sup>A question counts as answered when its ROUGE-L is at least max(0.20, m), where m is MEMORILLA’s median.

## 5 WHAT DOES MEMORILLA LEARN?

Conceptual understanding vs. factual recall. Figure 4a breaks down NarrativeQA performance by question type. MEMORILLA outperforms RAG-50 by +4.91 ROUGE-L on “why” questions and +7.02 on “how,” both requiring reasoning about narrative structure and causality. Conversely, RAG-50 leads by +3.45 on “who” and by +2.93 on “which,” where retrieval can locate the entity name verbatim. One explanation is that compression preserves global structure for conceptual reasoning but loses the exact surface tokens that retrieval excels at pinpointing (although question type also varies with answer length and entity frequency). The same trade-off would explain why MemoRAG, which reads selected passages, leads our ROUGE scores on this benchmark while trailing on F1. Figure 6 illustrates the complementary strength in personalization, where MEMORILLA uses semantically relevant medical history despite no lexical overlap with the query.

Multi-entity relational reasoning. On FactKG, MEMORILLA’s advantage over RAG-50 grows monotonically with entity count, from +4.23pp at 1 entity to +7.59pp at 4+ (Figure 4b). RAG-50 drops from 78.73 to 51.75 as the number of composed facts increases from 1 to 4, while MEMORILLA remains in the 80–89 range (Figure 4c). Compression encodes relational structure holistically, whereas retrieval degrades when the reasoning chain spans multiple documents. Because 73.5% of FactKG claims have more than 50 chunks, part of this gap reflects evidence RAG-50 never retrieves, but full context, which reads every chunk, still trails MEMORILLA overall (80.43 against 84.33).

MEMORILLA and RAG solve different questions. On FactKG, we observed that 15.9% of questions are answered by MEMORILLA alone against 10.0% by RAG-50 alone, and on PMv2 short-answer the split is 27.1% against 9.2%. In Figure 5, we show that the union of MEMORILLA and RAG (which is an oracle upper bound), could lead to substantial gains in performance than either approach alone, which motivates the design of hybrid retrieval and latent-memory systems as potential future work.

![](images/790610459ae97242b9fc1fee3c482521d464a51a1802082d4765e903e089e433.jpg)  
Figure 6: Personalization Case Study. On PersonaMem-v2, MEMORILLA personalizes a generic query by leveraging the user’s remembered iron-deficiency anemia history, despite no lexical overlap between the question and the relevant conversation history.

![](images/45b9ce1a68e0c59a019603542158a46f4c9fb87b6510e242c623221179cf303d.jpg)  
Figure 7: Reinforcement-learned memory on TextWorld. (a) A frozen encoder embeds the trajectory, the trainable memory module compresses it to K=8 latents, and the frozen LLM agent acts; sparse rewards over G=16 rollouts drive a GRPO update to the memory module alone. (b) On a held-out game, ICL loops between explored rooms while (c) MEMORILLA reaches the goal (⋆).

What neither method solves. We also examine the questions that both MEMORILLA and RAG-50 answer incorrectly. On PV4, many of these require linking facts that each appear somewhere in the user’s history, which suggests that the remaining errors come from combining information rather than from failing to find it. On FactKG, both methods are biased toward judging claims false. True claims make up 89.5% of the claims both methods get wrong but only 28.7% of those both get right, which points to a miscalibrated decision threshold rather than missing evidence.

## 6 CONTINUAL LEARNING WITH MEMORILLA

Beyond the supervised learning settings where the document collection can be static, we showcase the potential of MEMORILLA in scenarios where an LLM agent has to continuously learn and adapt from its environment. One such environment is TextWorld (Côté et al., 2018), a suite of partially observable text games in which an agent explores up to 65 rooms, manipulates objects and pursues multi-step goals for at most 50 turns, with reward arriving mainly on a win. After each turn, MEMORILLA stores the completed turn as a new document and condenses the growing document embedding store into K = 8 memory tokens placed before the prompt (Figure 7 and Appendix G).

## 6.1 EXPERIMENTAL SETUP

The agent is a frozen Qwen3-4B-Instruct-2507 decoder paired with the frozen Qwen3-Embedding-4B encoder. Only the memory module is trained, from random initialization and with GRPO on environment reward (see Appendix G.1). We compare against two baselines. Full-history in-context learning (ICL) gives the decoder the complete interaction history without memory, and an untrained memory module isolates the contribution of RL. Unlike in the supervised setting, the memory supplements the text rather than replacing it. Every condition keeps the full history under the same input budget (16,000 tokens in training, 4,096 in evaluation), so any improvement reflects what the memory learns to extract, not access to additional information. In the medium and hard games, the history grows from 800 tokens at the first turn to about 3,200 and then levels off, since the median rollout ends at turn 33 and only 5 of 40 rollouts reach turn 50.

## 6.2 RESULTS

Task completion. Training MEMORILLA with GRPO improves avg@3 from 0.186 to 0.242 and pass@3 from 0.250 to 0.375 relative to full-history ICL (Figure 8). Untrained memory instead falls below ICL on both metrics, avg@3 (0.153) and pass@3 (0.225), so the gain comes from learning rather than from the added tokens. The trained memory acts as a compact state that keeps the frozen agent oriented over long trajectories.

MEMORILLA reduces agent loops. The largest behavioral change is termination rather than solve rate. ICL exhausts its 4,096-token budget on 71% of rollouts against 58% for MEMORILLA despite both reading the same trajectory text. Trajectory inspection shows that trained memory helps the agent escape navigation loops and execute keyed action sequences, such as unlocking and then opening a container. Figure 7 illustrates this, with ICL cycling between explored rooms and never reaching the goal, while MEMORILLA advances to the target room. The re- maining failures concentrate in the longest games, whose quests approach or exceed the 50-turn limit and require backtracking across many rooms.

![](images/a941f531bb509e9c910c905b429daaed7445222d8d55c6e8baaf6581da36bed5.jpg)  
Figure 8: MEMORILLA on TextWorld with a frozen Qwen3-4B-Instruct-2507 decoder. Terminal episodes end through the game or the turn limit before the history exceeds the 4,096-token evaluation budget.

## 7 DISCUSSION AND CONCLUSION

MEMORILLA separates persistent knowledge storage from the context required for each query. New documents can be added without updating the LLM, while the memory module learns to extract relevant information through a fixed number of latent tokens. This design supports assistants that draw on growing histories without proportionally increasing the decoder’s context. Our analyses suggest that the learned representations capture relationships across documents and recover personal context beyond lexical similarity. In TextWorld, reinforcement learning improves a frozen agent’s decisions even when the interaction history is already available as text. Together, these findings support learned memory as a means of improving how LLMs use accumulated information, with the memory interface optimized through either answer supervision or environment reward.

Conclusion. In this work, we introduced MEMORILLA, a latent semantic memory system that gives frozen LLMs access to query-dependent information from large corpora as ‘soft’ memory tokens. MEMORILLA demonstrates that a frozen LLM can use accumulated knowledge through a learned, query-dependent interface, rather than relying only on retrieved text or an expanding prompt. MEMORILLA leads retrieval, compression and agent memory baselines on average at 20 to 4,212× fewer document tokens across personalization, long-document QA, and knowledge-intensive tasks. Additionally, MEMORILLA also enables continual learning in agentic RL, where training only the memory module teaches an agent a new environment without overwriting prior knowledge.

Limitations and Future Work. MEMORILLA can lose fine-grained factual details, likely because the document encoder was never trained for memory compression. Future work exploring combinations of latent memory and text retrieval could combine the global structure that memory captures with the exact entities that retrieval recovers. Furthermore, encoders trained specifically for compression could close the remaining gap on fine-grained details. Our experiments pair a Qwen encoder with a Qwen decoder, and testing how well the memory mechanism transfers across model families is also an interesting future area of research. Our evaluation covers personalization, long-document QA and knowledge-intensive QA, so MEMORILLA may not carry over directly to domains such as coding. Nonetheless, for personalization tasks, because the persistent memory store holds only embeddings, a multimodal encoder could extend the same interface to images, audio and video, giving assistants a memory of what a user has seen and heard. Since each user’s memory is a separate store, it can be kept on device or deleted on request, which eases privacy concerns. Finally, because LLM decoding dominates latency, the efficiency gains show up mainly for large collections. Hence one can explore extensions of MEMORILLA where memory modules could share a single persistent store and inject memory tokens across LLMs of varying sizes.

## AI USE STATEMENT

We used coding agents to implement and run experiments, and minimally used generative AI tools to polish the writing of the paper. As part of our method, GPT-5 generated the PersonalizationV4 personas, GPT-5.1 generated the question categories, chat topics and questions, and GPT-5-mini generated the dialogues (Appendix D.1). We did not use generative AI tools for any other part of this work. The authors reviewed all AI-assisted work and verified the code and experimental results produced with coding agents before reporting them. We take responsibility for the final content of this work, including text, claims and artifacts produced with the aid of generative AI.

## ETHICS STATEMENT

MEMORILLA stores and recalls information from user conversations, which raises privacy concerns when deployed on real user data. Because the memory store is external to the LLM, a user’s embeddings can be removed without retraining, but stored embeddings can still encode sensitive information and deployments should obtain user consent and protect them accordingly. PersonalizationV4 consists of synthetic users based on LLM-generated personas and contains no data from real users. All persona profiles follow a single machine template.

## REPRODUCIBILITY STATEMENT

Section 2 describes the MEMORILLA architecture, and Sections 2.6 and 2.4 describe the training curriculum and objective. Appendix B.1 lists the training data, and Appendix B.2 reports the model configuration, training hyperparameters, document chunking and evaluation splits. Appendix C describes the implementation of each baseline, and Appendix D describes the benchmarks, the PersonalizationV4 data pipeline and the context budget of each method. Appendix G gives the full TextWorld RL setup. All supervised evaluations use greedy decoding and a fixed random seed. We are releasing the code, the training and evaluation data with precomputed embeddings, and the PersonalizationV4 data on our project website.

## ACKNOWLEDGMENTS

We thank Ben Geist, Charilaos Kanatsoulis, Jonas De Schouwer, and Michael Bereket for discussions and feedback on our manuscript, Shayan Talaei for discussions about memory, Victor Li for initial brainstorming of the project, and Olivia Markham for initial experimentation. This work used Marlowe (Kapfer et al., 2025), Stanford University’s GPU-based Computational Instrument, supported by Stanford HAI and Stanford Research Computing. We also gratefully acknowledge the support of NSF under Nos. CCF-1918940 (Expeditions), DMS-2327709 (IHBEM), IIS-2403318 (III); NIH under No. 1U24NS146314-01, Gates Foundation, Stanford Data Applications Initiative, Wu Tsai Neurosciences Institute, Stanford Institute for Human-Centered AI, Chan Zuckerberg Initiative, Amazon, Genentech, SAP, and SCBX. The content is solely the responsibility of the authors and does not necessarily represent the official views of the funding entities.

## REFERENCES

Chenxin An, Jun Zhang, Ming Zhong, Lei Li, Shansan Gong, Yao Luo, Jingjing Xu, and Lingpeng Kong. Why does the effective context length of LLMs fall short? In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=eoln5WgrPx.

Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang. Ms marco: A human generated machine reading comprehension dataset. 2016. doi: 10.48550/arXiv.1611.09268. URL https://arxiv.org/abs/1611. 09268.

Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=8GjSf9Rh7Z.

Shiyi Cao, Sumanth Hegde, Dacheng Li, Tyler Griggs, Shu Liu, Eric Tang, Jiayi Pan, Xingyao Wang, Akshay Malik, Graham Neubig, Kourosh Hakhamaneshi, Richard Liaw, Philipp Moritz, Matei Zaharia, Joseph E. Gonzalez, and Ion Stoica. Skyrl-v0: Train real-world long-horizon agents via reinforcement learning, 2025.

Yinpeng Chen, DeLesley Hutchins, Aren Jansen, Andrey Zhmoginov, David Racz, and Jesper Sparre Andersen. MELODI: Exploring memory compression for long contexts. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview. net/forum?id=TvGPP8i18S.

Yulong Chen, Yang Liu, Liang Chen, and Yue Zhang. Dialogsum: A real-life scenario dialogue summarization dataset. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pp. 5062–5074. Association for Computational Linguistics, 2021. doi: 10.18653/v1/2021. findings-acl.449. URL https://aclanthology.org/2021.findings-acl.449.

Xin Cheng, Xun Wang, Xingxing Zhang, Tao Ge, Si-Qing Chen, Furu Wei, Huishuai Zhang, and Dongyan Zhao. xrag: Extreme context compression for retrieval-augmented generation with one token. Advances in Neural Information Processing Systems, 37:109487–109516, 2024.

Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. Adapting language models to compress contexts. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 3829–3846, 2023.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory. 2025. doi: 10.48550/arXiv.2504. 19413. URL https://arxiv.org/abs/2504.19413.

Marc-Alexandre Côté, Ákos Kádár, Xingdi Yuan, Ben Kybartas, Tavian Barnes, Emery Fine, James Moore, Matthew Hausknecht, Layla El Asri, Mahmoud Adada, Wendy Tay, and Adam Trischler. Textworld: A learning environment for text-based games. In Computer Games Workshop at IJCAI, 2018. URL https://arxiv.org/abs/1806.11532.

Yufeng Du, Minyang Tian, Srikanth Ronanki, Subendhu Rongali, Sravan Bodapati, Aram Galstyan, Azton Wells, Roy Schwartz, Eliu A Huerta, and Hao Peng. Context length alone hurts llm performance despite perfect retrieval. arXiv preprint arXiv:2510.05381, 2025.

Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 2368–2378. Association for Computational Linguistics, 2019. doi: 10.18653/v1/N19-1246. URL https: //aclanthology.org/N19-1246.

Sabri Eyuboglu, Ryan Saul Ehrlich, Simran Arora, Neel Guha, Dylan Zinsley, Emily Ruoyu Liu, Atri Rudra, James Zou, Azalia Mirhoseini, and Christopher Re. Cartridges: Lightweight and general-purpose long context representations via self-study. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum? id=0k5w8O0SNg.

Tao Ge, Hu Jing, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. In-context autoencoder for context compression in a large language model. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=uREj4ZuGJE.

Bogdan Gliwa, Iwona Mochol, Maciej Biesek, and Aleksander Wawer. Samsum corpus: A humanannotated dialogue dataset for abstractive summarization. In Proceedings ofthe 2nd Workshop on New Frontiers in Summarization, pp. 70–79. Association for Computational Linguistics, 2019. doi: 10.18653/v1/D19-5409. URL https://aclanthology.org/D19-5409.

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. Realm: Retrievalaugmented language model pre-training. 2020. doi: 10.48550/arXiv.2002.08909. URL https: //arxiv.org/abs/2002.08909.

Jie He, Richard He Bai, Sinead Williamson, Jeff Z Pan, Navdeep Jaitly, and Yizhe Zhang. Clara: Bridging retrieval and generation with continuous latent reasoning. arXiv preprint arXiv:2511.18659, 2025.

Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. URL https: //arxiv.org/abs/2106.09685.

Yuyang Hu, Shichun Liu, Yanwei Yue, Guibin Zhang, Boyang Liu, Fangyi Zhu, Jiahang Lin, Honglin Guo, Shihan Dou, Zhiheng Xi, et al. Memory in the age of ai agents. arXiv preprint arXiv:2512.13564, 2025.

Bernd Huber, Ghazal Fazelnia, Andreas Damianou, Sebastian Peleato, Maksym Lefarov, Praveen Ravichandran, Marco De Nadai, Mounia Lalmas-Roellke, and Paul Bennett. Embedding-to-prefix: Continual personalization with large language models. In AI That Keeps Up: NeurIPS 2025 Workshop on Continual and Compatible Foundation Model Updates, 2025.

Andrew Jaegle, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals, and Joao Carreira. Perceiver: General perception with iterative attention. 2021. doi: 10.48550/arXiv.2103.03206. URL https://arxiv.org/abs/2103.03206.

Pegah Jandaghi, XiangHai Sheng, Xinyi Bai, Jay Pujara, and Hakim Sidahmed. Faithful personabased conversational dataset generation with large language models, 2023. URL https:// arxiv.org/abs/2312.10007.

Bowen Jiang, Yuan Yuan, Maohao Shen, Zhuoqun Hao, Zhangchen Xu, Zichen Chen, Ziyi Liu, Anvesh Rao Vijjini, Jiashu He, Hanchao Yu, Radha Poovendran, Gregory Wornell, Lyle Ungar, Dan Roth, Sihao Chen, and Camillo Jose Taylor. Personamem-v2: Towards personalized intelligence via learning implicit user personas and agentic memory. 2025. doi: 10.48550/arXiv.2512.06688. URL https://arxiv.org/abs/2512.06688.

Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W. Cohen, and Xinghua Lu. Pubmedqa: A dataset for biomedical research question answering, 2019. URL https://arxiv.org/abs/ 1909.06146.

Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. 2017. doi: 10.48550/arXiv.1705.03551. URL https://arxiv.org/abs/1705.03551.

Jikun Kang, Wenqi Wu, Filippos Christianos, Alex J Chan, Fraser Greenlee, George Thomas, Marvin Purtorab, and Andy Toulis. Lm2: Large memory models. arXiv preprint arXiv:2502.06049, 2025.

Craig Kapfer, Kurt Stine, Balasubramanian Narasimhan, Christopher Mentzel, and Emmanuel Candès. Marlowe: Stanford’s gpu-based computational instrument, 2025. URL https://doi.org/ 10.5281/zenodo.14751899.

Jiho Kim, Sungjin Park, Yeonsu Kwon, Yohan Jo, James Thorne, and Edward Choi. Factkg: Fact verification via reasoning on knowledge graphs. 2023. doi: 10.48550/arXiv.2305.06590. URL https://arxiv.org/abs/2305.06590.

Tomáš Kociskˇ y, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, Gábor Melis,\` and Edward Grefenstette. The narrativeqa reading comprehension challenge. Transactions ofthe Associationfor Computational Linguistics, 6:317–328, 2018.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. 2020. doi: 10.48550/arXiv. 2005.11401. URL https://arxiv.org/abs/2005.11401.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models, 2023. URL https:// arxiv.org/abs/2301.12597.

Zongqian Li, Yixuan Su, and Nigel Collier. 500xcompressor: Generalized prompt compression for large language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 25081–25091, 2025.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the associationfor computational linguistics, 12:157–173, 2024.

Maxime Louis, Hervé Déjean, and Stéphane Clinchant. Pisco: Pretty simple compression for retrieval augmented generation. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pp. 15506–15521, 2025.

Jesse Mu, Xiang Li, and Noah Goodman. Learning to compress prompts with gist tokens. Advances in Neural Information Processing Systems, 36:19327–19352, 2023.

Lin Ning, Luyang Liu, Jiaxing Wu, Neo Wu, Devora Berlowitz, Sushant Prakash, Bradley Green, Shawn O’Banion, and Jun Xie. User-llm: Efficient llm contextualization with user embeddings. In Companion Proceedings of the ACM on Web Conference 2025, pp. 1219–1223, 2025.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G Patil, Ion Stoica, and Joseph E Gonzalez. Memgpt: Towards llms as operating systems. arXiv preprint arXiv:2310.08560, 2023.

Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia, Xufang Luo, Jue Zhang, Qingwei Lin, Victor Rühle, Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu, and Dongmei Zhang. Llmlingua-2: Data distillation for efficient and faithful task-agnostic prompt compression, 2024. URL https://arxiv.org/abs/2403.12968.

Hongjin Qian, Zheng Liu, Peitian Zhang, Kelong Mao, Defu Lian, Zhicheng Dou, and Tiejun Huang. Memorag: Boosting long context processing with global memory-enhanced retrieval augmentation. In Proceedings of the ACM on Web Conference 2025, pp. 2366–2377, 2025.

Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don’t know: Unanswerable questions for squad. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, pp. 784–789. Association for Computational Linguistics, 2018. doi: 10.18653/v1/ P18-2124. URL https://aclanthology.org/P18-2124.

Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. In-context retrieval-augmented language models. Transactions ofthe Association for Computational Linguistics, 11:1316–1331, 2023.

David Rau, Shuai Wang, Hervé Déjean, and Stéphane Clinchant. Context embeddings for efficient answer generation in rag. arXiv preprint arXiv:2407.09252, 2024.

Siva Reddy, Danqi Chen, and Christopher D. Manning. Coqa: A conversational question answering challenge. Transactions ofthe Associationfor Computational Linguistics, 7:249–266, 2019. doi: 10.1162/tacl\_a\_00266. URL https://direct.mit.edu/tacl/article/43511.

Anna Rogers, Olga Kovaleva, Matthew Downey, and Anna Rumshisky. Getting closer to ai complete question answering: A set of prerequisite real tasks. Proceedings of the AAAI Conference on Artificial Intelligence, 34(05):8722–8731, 2020. doi: 10.1609/aaai.v34i05.6398. URL https: //ojs.aaai.org/index.php/AAAI/article/view/6398.

Alireza Salemi, Sheshera Mysore, Michael Bendersky, and Hamed Zamani. Lamp: When large language models meet personalization. 2023. doi: 10.48550/arXiv.2304.11406. URL https: //arxiv.org/abs/2304.11406.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. URL https://arxiv.org/ abs/1707.06347.

Abigail See, Peter J. Liu, and Christopher D. Manning. Get to the point: Summarization with pointer-generator networks. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, pp. 1073–1083. Association for Computational Linguistics, 2017. doi: 10.18653/v1/P17-1099. URL https://aclanthology.org/P17-1099.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. URL https://arxiv.org/abs/2402.03300.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. 2021. doi: 10.48550/arXiv.2104.09864. URL https://arxiv.org/abs/2104.09864.

Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng Yan, Jianfeng Gao, and Furu Wei. Augmenting language models with long-term memory. Advances in neural information processing systems, 36:74530–74543, 2023.

Xiangfeng Wang, Zaiyi Chen, Tong Xu, Zheyong Xie, Yongyi He, and Enhong Chen. In-context former: Lightning-fast compressing context for large language model. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 2445–2460, 2024a.

Yu Wang, Yifan Gao, Xiusi Chen, Haoming Jiang, Shiyang Li, Jingfeng Yang, Qingyu Yin, Zheng Li, Xian Li, Bing Yin, et al. Memoryllm: Towards self-updatable large language models. arXiv preprint arXiv:2402.04624, 2024b.

Yu Wang, Dmitry Krotov, Yuanzhe Hu, Yifan Gao, Wangchunshu Zhou, Julian McAuley, Dan Gutfreund, Rogerio Feris, and Zexue He. M+: Extending memoryLLM with scalable longterm memory. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=OcqbkROe8J.

Yaxiong Wu, Sheng Liang, Chen Zhang, Yichao Wang, Yongyue Zhang, Huifeng Guo, Ruiming Tang, and Yong Liu. From human memory to ai memory: A survey on memory mechanisms in the era of llms. arXiv preprint arXiv:2504.15965, 2025.

Zijun Wu, Yongchang Hao, and Lili Mou. Tokmem: One-token procedural memory for large language models. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=RWjEf9PdiJ.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for llm agents. arXiv preprint arXiv:2502.12110, 2025.

Yoel Zeldes, Amir Zait, Ilia Labzovsky, Danny Karmon, and Efrat Farkash. Commer: a framework for compressing and merging user data for personalization. arXiv preprint arXiv:2501.03276, 2025.

Peitian Zhang, Zheng Liu, Shitao Xiao, Ninglu Shao, Qiwei Ye, and Zhicheng Dou. Long context compression with activation beacon. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=1eQT9OzfNQ.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models with long-term memory. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pp. 19724–19731, 2024.

## A RELATED WORK

Textual memory. Textual memory keeps source records readable and retrieves relevant records at inference. RAG (Lewis et al., 2020) and REALM (Guu et al., 2020) retrieve passages from a document index. Systems built for longer histories retain the same basic interface while organizing the records differently. MemGPT (Packer et al., 2023) moves information between context tiers, MemoryBank (Zhong et al., 2024) stores conversations and summaries, and Mem0 (Chhikara et al., 2025) and A-MEM (Xu et al., 2025) maintain structured records across sessions. These systems preserve provenance and exact details, but the LLM can reason only over material that retrieval places in its context. Evidence outside the selected set remains unavailable, and widening that set yields limited returns. In-Context RALM (Ram et al., 2023) finds that most question-answering gains come from the first one or two retrieved documents. MemoRAG (Qian et al., 2025) uses compressed global memory to generate retrieval clues, but its final generator still reads selected passages. Textual memory is therefore effective when evidence can be localized, while signals distributed across a collection remain constrained by document selection. MEMORILLA instead computes a query-conditioned latent state from the full collection, complementing retrieval when exact evidence is needed.

Latent compression. Latent compression replaces replayed text with compact continuous state. ICAE (Ge et al., 2024) and IC-Former (Wang et al., 2024a) encode a long input into reusable vectors consumed by frozen target LLMs, while AutoCompressors (Chevalier et al., 2023) fine-tune an LLM to produce and consume recursive summary vectors. These representations can be cached because they are computed without a query, but they cannot change what they emphasize for different queries. Gist tokens (Mu et al., 2023) and 500xCompressor (Li et al., 2025) make the same tradeoff using layer-specific key–value states rather than standard input embeddings. xRAG (Cheng et al., 2024) makes the representation query relevant by first selecting a document through retrieval and mapping its embedding to one LLM token, which retains document selection as the access mechanism. A second line improves the alignment between compression and generation. PISCO (Louis et al., 2025) and COCOM (Rau et al., 2024) jointly adapt their compressors and generators, while CLaRa (He et al., 2025) learns reusable document vectors before training a query reasoner with the generator. These methods gain a learned latent interface but require an adapted generator. Other methods obtain reuse within a narrower memory domain. TokMem (Wu et al., 2026) optimizes a separate token for each procedure, while ComMer (Zeldes et al., 2025), USER-LLM (Ning et al., 2025), and Embedding to-Prefix (Huber et al., 2025) construct latent state for user histories. MEMORILLA instead accepts unseen document collections without further optimization and converts an LLM-independent store into new query-conditioned input embeddings for a frozen LLM.

Model memory. Model memory stores information in architecture-specific state. MemoryLLM (Wang et al., 2024b) maintains a self-updating latent pool, M+ (Wang et al., 2025) retrieves from long-term storage using the current hidden state, and LongMem (Wang et al., 2023) retrieves cached key–value states through a trained side network. Their shared internal representation supports query-conditioned recall, but also couples the memory and its reader to a particular model. Other methods preserve state during model execution rather than across an independent document collection. LM2 (Kang et al., 2025) updates a gated memory matrix during autoregressive inference. Titans (Behrouz et al., 2025) combines fixed task-level parameters with a contextual neural memory updated while processing a sequence. MELODI (Chen et al., 2025) and Activation Beacon (Zhang et al., 2025) compress state across context windows and transformer layers to extend long-sequence processing. Cartridges (Eyuboglu et al., 2026) makes model-specific key–value caches reusable and composable, but requires offline optimization for each new corpus. These approaches demonstrate the value of persistent latent state, while their architecture-specific interfaces limit reuse across models and collections. MEMORILLA instead keeps memory external and supplies it to a frozen LLM as input embeddings.

## B ADDITIONAL DETAILS ON TRAINING METHODOLOGY

## B.1 TRAINING DATASETS

Table 1 details the training data used across the first two stages of the MEMORILLA pipeline. The pretraining corpus consists of approximately 2M paraphrase reconstruction examples derived from

English Wikipedia articles. The finetuning mixture aggregates 1.34M queries across 10 sources spanning reading comprehension, open-domain QA, and summarization, exposing the memory module to varied compression demands across different answer types and document structures. PubMedQA contributes only its 900-question training split, which is disjoint from the 100 test questions used for evaluation.

Table 1: Training datasets for MEMORILLA. The pretraining corpus is derived from English Wikipedia. The finetuning mixture spans reading comprehension, open-domain question answering, and summarization tasks across 10 sources. Sizes are numbers of training queries.
<table><tr><td>Stage</td><td>Source</td><td>Category</td><td>Num. Queries</td></tr><tr><td>Pretraining</td><td>EnWiki Articles</td><td>Paraphrase</td><td>1,979,809</td></tr><tr><td>Finetuning</td><td>SQuAD v2 (Rajpurkar et al., 2018)</td><td>Reading Comprehension</td><td>86,821</td></tr><tr><td></td><td>DROP (Dua et al., 2019)</td><td>Reading Comprehension</td><td>77,400</td></tr><tr><td></td><td>CoQA (Reddy et al., 2019)</td><td>Reading Comprehension</td><td>108,647</td></tr><tr><td></td><td>PubMedQA (Jin et al., 2019)</td><td>Reading Comprehension</td><td>900</td></tr><tr><td></td><td>QuAIL (Rogers et al., 2020)</td><td>Reading Comprehension</td><td>10,246</td></tr><tr><td></td><td>PwC (Ge et al., 2024)</td><td>Reading Comprehension</td><td>241,564</td></tr><tr><td></td><td>CNN/DailyMail (See et al., 2017)</td><td>Summarization</td><td>287,113</td></tr><tr><td></td><td>SamSum (Gliwa et al., 2019)</td><td>Summarization</td><td>14,731</td></tr><tr><td></td><td>DialogSum (Chen et al., 2021)</td><td>Summarization</td><td>12,460</td></tr><tr><td></td><td>MS MARCO (Bajaj et al., 2016)</td><td>Open-domain QA</td><td>504,111</td></tr><tr><td></td><td>Finetuning total</td><td></td><td>1,343,993</td></tr></table>

## B.2 TRAINING CONFIGURATION

All MEMORILLA runs use frozen encoder and decoder backbones. Only the memory module is trainable unless otherwise noted. Supervised training stages use bf16 mixed precision on 4 NVIDIA H100 80GB GPUs and random seed 23. Document embeddings are precomputed and cached to amortize encoding cost. Table 2 lists how each benchmark’s collection is split into the chunks $d _ { i }$ that the encoder embeds. The supervised objective follows Section 2.4, with loss computed only over answer tokens. A single memory module is trained jointly across all downstream tasks for up to 5 epochs. On PV4 and PMv2 we continue for additional epochs because both were still improving on validation when the other benchmarks had converged. Train and test users are disjoint on both, so one module serves every user and enrolling a new one requires only embedding their chats.

Table 2: Document chunks per benchmark. Each chunk $d _ { i }$ is embedded once by the frozen encoder. Token counts use the Qwen3 tokenizer.
<table><tr><td>Benchmark</td><td>Chunk unit</td><td>Tokens per chunk (mean / median)</td></tr><tr><td>PMv2</td><td>One conversation session</td><td>280 / 262</td></tr><tr><td>PV4</td><td>One chat dialogue</td><td>252 / 224</td></tr><tr><td>PubMedQA</td><td>One abstract section</td><td>84 / 78</td></tr><tr><td>TriviaQA</td><td>Evidence passage, split sentence-greedily at 128 tokens</td><td>70 / 77</td></tr><tr><td>NarrativeQA</td><td>Story paragraph, split on sentences above ~256 tokens</td><td>94 / 35</td></tr><tr><td>FactKG</td><td>One knowledge-graph triple</td><td>13 / 13</td></tr><tr><td>LaMP-4</td><td>One profile item</td><td>47 / 44</td></tr><tr><td>LaMP-7</td><td>One profile tweet</td><td>25 / 25</td></tr></table>

## C ADDITIONAL DETAILS ON BASELINES

All baselines use the frozen Qwen3-8B as the decoder and the frozen Qwen3-Embedding-4B as the document encoder, and every answer is generated greedily. Baselines that insert text share one prompt template per benchmark and differ only in the context placed before the question.

Closed-book. The decoder receives only the query, which shows how much of each benchmark it answers from parametric knowledge.

RAG and Full Context. RAG-k ranks the chunks of a collection by the cosine similarity between their embeddings and the query embedding, and places the top k chunks in the prompt as text, most similar first and separated by blank lines. Full Context places every chunk in its stored order. Inputs longer than the enforced 32,768-token limit use YaRN with a scaling factor of 4, which extends the window to 131,072 tokens and applies to part of PMv2, TriviaQA and NarrativeQA and to all of PV4 (Table 10). For the 10.8% of evaluated NarrativeQA questions whose books exceed this window, the earliest chunks are dropped.

Table 3: Model configuration for MEMORILLA. The memory module maps document representations into a fixed set of learned memory tokens that are inserted into the frozen decoder context.
<table><tr><td>Component</td><td>Setting</td><td>Value</td></tr><tr><td>Memory module</td><td>Pooling mode</td><td>Adaptive cross-attention with softmax over documents</td></tr><tr><td></td><td>Memory slots</td><td>K=16</td></tr><tr><td></td><td>Cross-attention layers</td><td>2 layers, slots ← documents</td></tr><tr><td></td><td>Attention heads</td><td>8</td></tr><tr><td></td><td>Encoder-side dimension</td><td>2560</td></tr><tr><td></td><td>Decoder-side dimension</td><td>2560 for 4B decoder, 4096 for 8B decoder</td></tr><tr><td></td><td>Slot initialization</td><td>Xavier uniform</td></tr><tr><td></td><td>Projection</td><td>Linear encoder-to-decoder projection</td></tr><tr><td></td><td>Regularization</td><td>Dropout 0.1 with residual cross-attention</td></tr><tr><td>Frozen backbones</td><td>Document encoder</td><td>Qwen3-Embedding-4B, mean-pooled per document</td></tr><tr><td></td><td>4B decoder</td><td>Qwen3-4B-Instruct-2507</td></tr><tr><td></td><td>8B decoder</td><td>Qwen3-8B</td></tr><tr><td></td><td>Trainable parameters</td><td>Memory module only; 190,190,080 for 4B and 194,123,776 for 8B</td></tr><tr><td>Generation</td><td>Decoding</td><td>Greedy decoding with T=0, top-p=1.0, top-k=−1</td></tr><tr><td></td><td>Short-answer limit</td><td>8 tokens for classification, 64 tokens for QA and fact-verification tasks</td></tr><tr><td></td><td>Long-form limit</td><td>128 tokens for LaMP-4 and LaMP-7; 512 tokens for EnWiki reconstruction</td></tr></table>

Table 4: Training stages for MEMORILLA. Stage 1 pretrains memory on EnWiki paraphrase reconstruction, Stage 2 warm-starts on the mixed supervision corpus in Table 1, and Stage 3 performs downstream multi-task fine-tuning.
<table><tr><td>Setting</td><td>Stage 1 EnWiki pretrain</td><td>Stage 2 Multi-task pretrain</td><td>Stage 3 Downstream fine-tune</td></tr><tr><td>Initialization</td><td>Random</td><td>Warm-start from Stage 1</td><td>Warm-start from Stage 2</td></tr><tr><td>Data</td><td>EnWiki paraphrase reconstruc- tion</td><td>Mixed supervision corpus</td><td>Unified downstream mixture</td></tr><tr><td>Tasks</td><td>Paraphrase reconstruction</td><td>Reading comprehension, open- domain QA, summarization</td><td>PV4, PMv2, FactKG, TriviaQA, NQA, PubMedQA, LaMP-4, LaMP-7</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>LR scheduler</td><td>Cosine annealing with warmup</td><td>Cosine annealing with warmup</td><td>Cosine annealing with warmup</td></tr><tr><td>Peak learning rate</td><td>3×10−4 1×10⁻2</td><td>3×10−4</td><td>1×10⁻4</td></tr><tr><td>Weight decay</td><td></td><td>1×10⁻²</td><td>1×10⁻2</td></tr><tr><td>Warmup ratio</td><td>0.03</td><td>0.03</td><td>0.1</td></tr><tr><td>Max decoder length</td><td>1024</td><td>1024</td><td>2048 2</td></tr><tr><td>Per-device batch size</td><td>8 for 4B, 4 for 8B</td><td>8 for 4B, 4 for 8B</td><td>8</td></tr><tr><td>Gradient accumulation</td><td>4 for 4B, 8 for 8B</td><td>4 for 4B, 8 for 8B</td><td></td></tr><tr><td>Effective batch size</td><td>128</td><td>128</td><td>64</td></tr><tr><td>Devices</td><td>4 GPUs</td><td>4 GPUs</td><td>4 GPUs</td></tr><tr><td>Epochs</td><td>1</td><td>1</td><td>5</td></tr><tr><td>Checkpoint selection</td><td>Final checkpoint</td><td>Final checkpoint</td><td>One shared multi-task checkpoint; best by validation loss only for PV4/PMv2 specialization</td></tr><tr><td>Validation interval</td><td></td><td></td><td>0.25 epoch</td></tr><tr><td>Random seed</td><td>23</td><td>23</td><td>23</td></tr></table>

LLMLingua-2. LLMLingua-2 (Pan et al., 2024) is a hard prompt compression method that deletes tokens from the context rather than replacing it with latent vectors. We use the released XLM-RoBERTa-large token classifier. Because the classifier never sees the question, each collection is compressed once and reused for every query, similar in principle to MEMORILLA. We concatenate the documents of the Full Context input with blank-line separators and compress them as one block, retaining newlines and question marks. The question, answer-format instruction, system prompt and chat template are copied verbatim, so the prompt differs from Full Context only in the compressed documents. We disable the option to discard whole documents, since document selection is retrieval and is already covered by RAG-k. We set absolute budgets of 32, 64, 128 and 350 tokens rather than a compression rate, which the method treats only as a soft target, and report the best budget for each metric.

Table 5: Downstream training and evaluation details. Stage 3 trains on a unified downstream mixture. PV4 and PMv2 receive additional benchmark-specific fine-tuning, while all other benchmarks use the multi-task checkpoint directly.
<table><tr><td>Item</td><td>Configuration</td></tr><tr><td>Stage 1 corpus</td><td>1.98M EnWiki paraphrase reconstruction examples, chunked at approximately 128 tokens with mean 1.73 documents per row</td></tr><tr><td>Stage 2 corpus</td><td>Mixed supervision over SQuAD v2, DROP, CoQA, PubMedQA, QuAIL, PwC, CNN/DailyMail, SAMSum, DialogSum, and MS MARCO</td></tr><tr><td>Stage 3 corpus</td><td>Unified concatenation of PV4, PMv2, FactKG, TriviaQA, NarrativeQA, PubMedQA, LaMP-4, and LaMP-7</td></tr><tr><td>PV4 and PMv2 specialization</td><td>After multi-task fine-tuning, PV4 and PMv2 are each further fine-tuned for 5 epochs on the corresponding benchmark training split using the same recipe</td></tr><tr><td>Other downstream tasks</td><td>FactKG, TriviaQA, NarrativeQA, PubMedQA, LaMP-4, and LaMP-7 use the multi-task checkpoint directly with no additional task-specific training</td></tr><tr><td>Evaluation split</td><td>Official test split for every task except LaMP-4 and LaMP-7, which use their development splits; TriviaQA, which uses a deterministic seed-23 validation subsample with n=500; and PubMedQA, which uses the 100-example test portion of a 900/100 split derived from PQA-L</td></tr></table>

xRAG. xRAG (Cheng et al., 2024) maps the embedding of the top retrieved chunk to a single soft token in the input space of the decoder through a trained bridge. We reimplement the bridge in our codebase as a layer normalization followed by a two-layer MLP with a GELU activation, which maps the 2,560-dimensional chunk embedding to the 4,096-dimensional decoder input space with 27,276,288 trainable parameters. The chunk is retrieved by cosine similarity with the shared encoder. Only the bridge is trained, from scratch on each benchmark’s training split with the same answer-token loss as MEMORILLA. We train for 5 epochs with AdamW at a learning rate of 10<sup>−4</sup>, a weight decay of 0.01, a warmup ratio of 0.1, an effective batch size of 16, a maximum sequence length of 8,192 tokens and seed 23.

ComMer. ComMer (Zeldes et al., 2025) compresses user-history documents into latent representations with a learned compressor, merges them and passes them to a frozen LLM for personalized generation. We compare against its reported LaMP-4 and LaMP-7 results in Table 12.

Mem0 and MemoRAG. Mem0 (Chhikara et al., 2025) extracts and consolidates memories at write time, while MemoRAG (Qian et al., 2025) forms clue queries from a global memory to guide retrieval. We run the official implementations with the settings in Table 6. Both use a frozen Qwen3-4B-Instruct-2507 to write their memory, as the fact extractor of Mem0 and as the memory model of MemoRAG. This matches the 4B scale of the Qwen3-Embedding-4B encoder with which MEMORILLA reads the same collections, so every method processes the documents with a 4B model and answers with the same frozen Qwen3-8B decoder. MemoRAG runs in its supported long-context memory mode, in which the memory model reads the uncompressed collection rather than a compressed version of it. Its retrieval uses each benchmark’s own chunks and the shared encoder, so the clue mechanism is its only difference from RAG-k. We sweep the retrieval depth of Mem0 over k ∈ {10, 30, 50} and of MemoRAG over its native setting and caps of 5 and 50 chunks, and report each at its best depth per metric.

## D ADDITIONAL DETAILS ON BENCHMARKS

Personalization. These benchmarks require combining information spread across a user’s past chats or writing into a picture of the user that no single document provides.

• PersonaMem-v2 (PMv2, n=5,000) (Jiang et al., 2025). Each user has a long history of conversations with an assistant about everyday tasks, and each question asks for a response that fits the user’s preferences. Most preferences are never stated outright but surface indirectly, for instance in an email the user asks to have polished, so the model must infer them from scattered cues across many sessions rather than look them up.

• PersonalizationV4 (PV4, n=4,237). Our synthetic benchmark gives each user 200 two-turn conversations that reflect a persona without stating its traits (Appendix D.1). Each question places the user in a new scenario and requires combining at least two persona facts, which may appear in different conversations. A hard subset (n=629) keeps the questions that the untrained Qwen3-4B-Instruct-2507 decoder answers incorrectly without documents or with the top retrieved conversation.

Table 6: Mem0 and MemoRAG configuration. Rows below the rule are shared with the RAG-k baselines, so both methods differ from RAG only in the context given to the decoder.
<table><tr><td>Setting</td><td>Mem0</td><td>MemoRAG</td></tr><tr><td>Implementation</td><td>mem0 ai 0.1.118 with its stock prompts and ADD/UP- Official repository with its prompts unchanged DATE/DELETE/NONE loop, unmodified</td><td></td></tr><tr><td>Memory-writing model</td><td>Frozen Qwen3-4B-Instruct-2507 extracts facts and con- Frozen Qwen3-4B-Instruct-2507 reads the collection as solidates memories</td><td>its memory</td></tr><tr><td>Memory-model decoding Write path</td><td>Greedy, up to 2,048 tokens One add () per document, with each extracted fact rec-</td><td>Greedy, up to 128 tokens per clue call Collection read once per entity and cached, front-</td></tr><tr><td></td><td>onciled against its 5 nearest stored memories</td><td>truncated only where it exceeds the context window (NarrativeQA)</td></tr><tr><td>Memory scope</td><td>rativeQA) or question (others)</td><td>One Qdrant collection per user (PMv2, PV4), story (Nar- One memory per user, story or question</td></tr><tr><td>Retrieval query</td><td>The question</td><td>Clue spans and surrogate questions longer than 3 words, plus the question</td></tr><tr><td>Retrieval</td><td>Top-k memories by cosine similarity</td><td>Top 3 chunks per query, unioned, deduplicated and kept in corpus order</td></tr><tr><td>Depths swept</td><td>k ∈ {10, 30, 50} memories</td><td>Native (about 10 to 20 chunks), and capped at 5 and 50 by maximum similarity over queries</td></tr><tr><td>Memory-model answer</td><td>Not applicable</td><td>Not used (use_memory_answer=False)</td></tr><tr><td>Retriever Generator</td><td colspan="2">Qwen3-Embedding-4B with 2,560 dimensions and no instruction prefix, over each benchmark&#x27;s chunks (Table 2) Frozen Qwen3-8B with the system prompt, chat template, answer-format instruction and maximum new tokens of</td></tr><tr><td></td><td colspan="2">RAG-k</td></tr><tr><td>Decoding</td><td colspan="2">Greedy, with T=0, top-p=1.0 and top-k=−1</td></tr></table>

• LaMP-4 (n=1,925), LaMP-7 (n=1,500) (Salemi et al., 2023). LaMP-4 asks the model to write a headline for a new news article in the style of the author’s earlier articles and headlines, and LaMP-7 asks it to rewrite a tweet in the style of the user’s earlier tweets. The challenge is stylistic rather than factual, since style emerges from the whole writing history rather than a single item.

Long-Document QA. This axis tests whether memory can compress entire long documents into a usable representation for downstream QA.

• NarrativeQA (NQA, n=10,557) (Kociskˇ y et al.\` , 2018). Questions concern full books and movie scripts and were written from plot summaries without access to the stories, so they target events, motives and relations described across the narrative rather than sentences that can be matched locally. Answers are free-form text, which rewards understanding the plot over copying a passage.

• TriviaQA (n=500 from the 17,944-question validation split) (Joshi et al., 2017). Questions written by trivia enthusiasts are paired with Wikipedia and web documents collected afterwards. Since the questions were not written against these documents, their wording and structure often differ from the passages that answer them, and an answer may require combining facts across sentences.

Knowledge-Intensive QA. These benchmarks test factual grounding from structured or semistructured evidence of varying quantity and format.

• FactKG (n=9,041) (Kim et al., 2023). Each claim must be labeled as supported or refuted by facts from the DBpedia knowledge graph, through one-hop lookups, conjunctions of several facts, existence checks, multi-hop chains or negations. A multi-hop claim such as “AIDAstella was built by a company in Papenburg” requires chaining two facts, neither of which verifies the claim on its own, and a negated claim requires confirming that a relation is absent.

• PubMedQA (n=100) (Jin et al., 2019). Each question is a biomedical research question based on an article title, answered yes, no or maybe from the article’s abstract with its conclusion removed. The model must infer the conclusion from the reported methods and results, often by interpreting quantitative findings. The entire source fits easily in the context, which makes PubMedQA a contrast case where condensing the source offers little benefit.

Table 7 summarizes the benchmark datasets evaluated in this work. All datasets use standard train/validation/test splits. For the personalization benchmarks (PV4, PMv2, LaMP-4, LaMP-7), training and evaluation users are strictly disjoint to prevent leakage of user-specific information. We evaluate LaMP-4 and LaMP-7 on their development splits and TriviaQA on the validation split. For PubMedQA we use a 900/100 train/test split derived from the PQA-L source. Table 10 reports how many document tokens each method places in the decoder per query. The serving stack enforces a 32,768-token decoder limit and enables YaRN for every full-context example that exceeds it, which covers part of PMv2, TriviaQA and NarrativeQA and all of PV4. Every PubMedQA collection and 96.7% of LaMP-7 collections hold at most 50 chunks, so RAG-50 reads the same chunks as Full Context on them.

Table 7: Benchmark datasets overview. Each row corresponds to an evaluation benchmark. Each data point consists of a textual query paired with a collection of source documents from which the model must produce an answer or generation. µ¯ docs/q and µ¯ tok/q report the mean number of source documents and total source tokens per query in the test split.
<table><tr><td>Dataset</td><td>Task</td><td>µ docs/q</td><td>µ tok/q</td><td>Train</td><td>Val</td><td>Test</td></tr><tr><td>PersonalizationV4 (PV4) (Ours)</td><td>Personalization SA</td><td>200</td><td>50,476</td><td>15,058</td><td>1,755</td><td>4,237</td></tr><tr><td>PersonaMem-v2 (PMv2) (Jiang et al., 2025)</td><td>Personalization SA</td><td>116</td><td>31,618</td><td>18,521</td><td>2,059</td><td>5,000</td></tr><tr><td>NarrativeQA (NQA) (Kočiskò et al., 2018)</td><td>Long-document generative QA</td><td>1,420</td><td>76,225</td><td>32,747</td><td>3,461</td><td>10,557</td></tr><tr><td>FactKG (Kim et al., 2023)</td><td>KG fact verification</td><td>129</td><td>1,665</td><td>86,367</td><td>13,266</td><td>9,041</td></tr><tr><td>TriviaQA (Joshi et al., 2017)</td><td>Open-domain factoid QA</td><td>164</td><td>11,861</td><td>138,384</td><td>17,944</td><td></td></tr><tr><td>PubMedQA (Jin et al., 2019)</td><td>Biomedical QA</td><td>4</td><td>324</td><td>900</td><td></td><td>100</td></tr><tr><td>LaMP-4 (Salemi et al., 2023)</td><td>Personalized headline gen.</td><td>172</td><td>7,638</td><td>12,527</td><td></td><td>1,925</td></tr><tr><td>LaMP-7 (Salemi et al., 2023)</td><td>Personalized tweet paraphrase</td><td>18</td><td>465</td><td>10,437</td><td></td><td>1,500</td></tr></table>

## D.1 PERSONALIZATIONV4 DATA PIPELINE

PersonalizationV4 is a synthetic benchmark for evaluating multi-document user modeling. Each user has one LLM-generated persona profile seeded from the Faithful Persona-based Conversational Dataset (Jandaghi et al., 2023). The resulting profiles are 10k to 22k characters long (median 15.5k) and describe demographics, personality, work, hobbies, communication style and how preferences change over time. All of the user’s data is derived from this profile through the five steps below, and Table 8 traces one held-out user through each of them.

1. Question categories. GPT-5.1 reads the persona and extracts 10 persona-specific categories that questions can test.

2. Chat topics. GPT-5.1 writes exactly 200 one-sentence scenarios per user in which the user asks the assistant for help, covering every part of the persona.

3. Dialogues. GPT-5-mini expands each topic into one two-turn dialogue with one user turn and one assistant turn. The user embodies the persona implicitly without stating its traits. Exactly 200 dialogues are generated per user and all form the collection that the memory module reads; there is no truncation step.

4. Questions. For each category, GPT-5.1 is asked to write 15 questions from the persona alone. It writes 14 in 1,234 of the 1,490 released category files and 13 to 15 in all but one, which has 5. Each question has a reference answer and four distractors. The instructions in Table 9 require every question to combine at least two persona facts in a new scenario and every distractor to be a plausible preference that matches the reference answer in length, structure and specificity. This gives 133 to 145 questions per user (mean 141.3).

5. Randomization. The reference answer’s position among its distractors is shuffled with a fixed seed of 42.

Questions are posed without their candidate answers, and the model answers in free text. The reference answer and distractors are used only by the short-answer scorer, which counts a response as correct when its nearest candidate under the Qwen3-Embedding-4B encoder is the reference answer. We release 149 users: 119 training users and 30 held-out evaluation users, none of whom appear in training. One training user whose generated persona was a refusal is withheld from the release. Every question for these held-out users is evaluated: 139 to 143 per user and 4,237 in total, all in the test split. The PersonalizationV4 dataset is linked on our project website. The hard subset (n=629) is defined from the untrained Qwen3-4B-Instruct-2507 decoder’s multiple-choice answers and contains questions it answers incorrectly without documents, with the top-1 retrieved document, or in both settings; 299 of the 629 questions are answered incorrectly in both settings.

Table 8: One held-out PersonalizationV4 user through the pipeline. Excerpts are verbatim apart from abridgement. The question draws on base-building, which appears only in the dialogue shown, and on the Iron Heart project, which appears in 18 other dialogues, so answering requires combining documents. The distractors are never shown to the model and serve only as scoring candidates.
<table><tr><td>Persona</td><td>Name Maya Patel. Age 34. Location Austin, Texas, USA. Occupation Certified Welder &amp; Part-Time CNC Operator at a regional aerospace supplier. . .. Maya grew up in a multicultural household where her Indian-American parents emphasized discipline, education, and the importance of physical well-being. After completing a two-year technical diploma in welding technology,</td></tr><tr><td>Categories</td><td>Aerospace welding expertise &amp; mental rehearsal habits. Gaming genre preferences &amp; playstyle strategies. Morning mango ritual &amp; mobility routine. ... (3 of 10)</td></tr><tr><td>Chat topic</td><td>Maya asks for a list of cooperative building-and-craft games she can play online with classmates after welding class.</td></tr><tr><td>User turn</td><td>. .. looking for coop building-and-craft games I can jump into with the welding class after tonight&#x27;s session. Need stuff that&#x27;s accessible for mixed skill levels, supports voice chat, and has meaningful base-building or automation</td></tr><tr><td>Assistant turn</td><td>mechanics. Toss me a short list with platform notes and why each one fits. (seven games with platform notes, from Minecraft to Raft) ... Each supports voice via Discord/Steam, scales with mixed experience, and offers clear building goals for group sessions.</td></tr><tr><td>Question</td><td>In a new open-world crafting RPG, players can specialize in one mastery for unique bonuses. Considering Maya&#x27;s welding expertise, love for base-building, and ergonomic focus from the Iron Heart project, which specialization best aligns with her likely in-game identity?</td></tr><tr><td>Reference answer</td><td>Master Builder, optimizing structural stability, workstation placement, and production comfort, because she prefers engineering-style layouts that echo her fabrication and ergonomic design background.</td></tr><tr><td>Distractors</td><td>Arcane Historian, optimizing lore discovery and dialogue outcomes, because she prefers uncovering story fragments over experimenting with materials and structural layouts. Beast Whisperer, optimizing pet behavior and breeding trees, because she prefers managing biological companions over designing durable and efficient physical constructs. Shadow Scavenger, optimizing stealth routes and loot efficiency, because she prefers sneaking past threats over methodically constructing and upgrading comprehensive bases.</td></tr></table>

Table 9: Question-generation instructions. Verbatim excerpts from the prompt given to GPT-5.1 for each category, which also contains the full persona profile, the target category and the number of questions to write.
<table><tr><td>Questions</td><td>Each question must require synthesizing information or understanding nuance from the persona. Questions must require combining two or more facts from the persona profile. Avoid questions that can be answered by recalling a single fact.</td></tr><tr><td>Distractors</td><td>The question stem must not hint at characteristics of the answer. Every choice (A-E) must be a reasonable, professional, or logical preference for a human. Never use obviously wrong, absurd, or incompetent distractors. Ensure distractors include industry-standard best practices that most people in the persona&#x27;s role would choose, but</td></tr></table>

## D.2 CONTEXT BUDGET PER QUERY

Table 10 reports the number of document-derived tokens each method places in the decoder for one query. We exclude the question, the answer-format instruction, the system prompt and the chat template, which are identical across methods. For each text baseline, the count is the length of its prompt minus the length of the closed-book prompt for the same question, measured with the Qwen3-8B tokenizer and averaged over the evaluated questions. Closed-book receives no document tokens, MEMORILLA contributes its 16 memory tokens and xRAG a single token. Figure 1 places each method at the geometric mean of these counts over the eight benchmarks, and the compression column of Table 11 divides the Full Context count by the 16 tokens of MEMORILLA.

Table 10: Document tokens per query. Mean number of document-derived tokens in the decoder input, excluding the question and all shared instructions. Mem0, MemoRAG and LLMLingua-2 are counted at the setting reported for each benchmark’s primary metric in Table 11, and truncated inputs are counted as read. Fits 32K reports whether every Full Context prompt fits the enforced 32,768-token limit, and Partial means that YaRN is enabled for the remaining examples. <sup>†</sup> marks benchmarks where RAG-50 reads the same chunks as Full Context, for every PubMedQA question and 96.7% of LaMP-7 questions.
<table><tr><td>Benchmark</td><td>Docs/q</td><td>Fits 32K</td><td>RAG 1</td><td>RAG 5</td><td>RAG 50</td><td>Full Context</td><td>LLM Lingua-2</td><td>xRAG</td><td>Mem0</td><td>Memo RAG</td><td>MEMORILLA</td></tr><tr><td>PMv2</td><td>116</td><td>Partial</td><td>125</td><td>652</td><td>9,009</td><td>31,587</td><td>250</td><td>1</td><td>665</td><td>9,254</td><td>16</td></tr><tr><td>PV4</td><td>200</td><td>No</td><td>201</td><td>1,021</td><td>10,832</td><td>50,455</td><td>180</td><td>1</td><td>978</td><td>11,046</td><td>16</td></tr><tr><td>PubMedQA</td><td>4</td><td>Yes</td><td>69</td><td>321</td><td>324†</td><td>324</td><td>290</td><td>1</td><td>8</td><td>324</td><td>16</td></tr><tr><td>NarrativeQA</td><td>1,420</td><td>Partial</td><td>69</td><td>354</td><td>3,612</td><td>67,385</td><td>596</td><td>1</td><td>493</td><td>800</td><td>16</td></tr><tr><td>TriviaQA</td><td>164</td><td>Partial</td><td>51</td><td>226</td><td>2,357</td><td>11,888</td><td>243</td><td>1</td><td>269</td><td>303</td><td>16</td></tr><tr><td>FactKG</td><td>129</td><td>Yes</td><td>13</td><td>64</td><td>568</td><td>1,727</td><td>279</td><td>1</td><td>300</td><td>573</td><td>16</td></tr><tr><td>LaMP-4</td><td>172</td><td>Yes</td><td>45</td><td>227</td><td>1,715</td><td>7,580</td><td>227</td><td>1</td><td>251</td><td>230</td><td>16</td></tr><tr><td>LaMP-7</td><td>18</td><td>Yes</td><td>25</td><td>125</td><td>435†</td><td>459</td><td>299</td><td>1</td><td>81</td><td>135</td><td>16</td></tr><tr><td>Geometric mean</td><td></td><td></td><td>55</td><td>273</td><td>1,808</td><td>6,301</td><td>278</td><td>1</td><td>212</td><td>784</td><td>16</td></tr></table>

## E ADDITIONAL RESULTS AND ABLATIONS

Table 11 reports the complete 8B results underlying the main quantitative comparison in Figure 1.   
Table 12 compares MEMORILLA with ComMer on LaMP-4 and LaMP-7.

Table 11: Main results across eight benchmarks. All methods use a frozen Qwen3-Embedding-4B encoder and a frozen Qwen3-8B decoder; Mem0 and MemoRAG also use a frozen Qwen3-4B-Instruct-2507 to build their memories. RAG-k retrieves the top-k chunks by cosine similarity and Full Context reads the full collection. MEMORILLA uses K=16 memory tokens. Best result per row is bold, second best is underlined. The Compression column reports how many times more document tokens Full Context gives the decoder than the 16 memory tokens of MEMORILLA (Table 10). LLMLingua-2 was not evaluated on the PV4 hard subset.
<table><tr><td>Task</td><td>Metric</td><td>Closed book</td><td>RAG 1</td><td>RAG 5</td><td>RAG 50</td><td>Full Context</td><td>LLM Lingua-2</td><td></td><td>xRAG Mem0</td><td>Memo RAG</td><td>MEMORILLA</td><td>Compression</td></tr><tr><td>PMv2</td><td>Acc (SA)</td><td>25.64</td><td>26.10</td><td>25.92</td><td>26.72</td><td>27.92</td><td>26.92</td><td>30.90</td><td>27.48</td><td>28.76</td><td>44.60</td><td>1,974×</td></tr><tr><td>PV4</td><td>Acc (SA)</td><td>58.30</td><td>62.21</td><td>64.98</td><td>65.78</td><td>59.12</td><td>52.14</td><td>62.64</td><td>65.07</td><td>67.88</td><td>66.65</td><td>3,153×</td></tr><tr><td></td><td>Acc (SA, Hard)</td><td>31.96</td><td>38.00</td><td>39.27</td><td>41.97</td><td>36.57</td><td></td><td>34.98</td><td>44.20</td><td>45.95</td><td>41.97</td><td></td></tr><tr><td>PubMedQA</td><td>Acc</td><td>21.00</td><td>34.00</td><td>67.00</td><td>67.00</td><td>67.00</td><td>69.00</td><td>36.00</td><td>24.00</td><td>67.00</td><td>59.00</td><td>20×</td></tr><tr><td>NarrativeQA</td><td>BLEU-1</td><td>7.09</td><td>7.36</td><td>7.85</td><td>8.29</td><td>5.89</td><td>6.87</td><td>16.86</td><td>7.77</td><td>9.14</td><td>16.36</td><td>4,212×</td></tr><tr><td></td><td>ROUGE-1</td><td>11.54</td><td>13.32</td><td>15.60</td><td>18.45</td><td>19.12</td><td>7.26</td><td>17.33</td><td>14.24</td><td>21.80</td><td>19.68</td><td></td></tr><tr><td></td><td>ROUGE-L</td><td>10.71</td><td>12.44</td><td>14.59</td><td>17.23</td><td>17.99</td><td>7.18</td><td>16.92</td><td>13.80</td><td>20.51</td><td>19.31</td><td></td></tr><tr><td></td><td>F1-Score</td><td>9.87</td><td>10.51</td><td>11.64</td><td>13.27</td><td>13.50</td><td>5.76</td><td>15.22</td><td>11.58</td><td>15.03</td><td>16.56</td><td></td></tr><tr><td>TriviaQA</td><td>EM</td><td>36.00</td><td>51.80</td><td>55.20</td><td>55.60</td><td>52.60</td><td>42.40</td><td>47.00</td><td>44.60</td><td>56.60</td><td>57.00</td><td>743×</td></tr><tr><td></td><td>BLEU-1</td><td>29.12</td><td>40.29</td><td>39.09</td><td>43.07</td><td>35.56</td><td>29.28</td><td>46.54</td><td>29.09</td><td>45.00</td><td>58.66</td><td></td></tr><tr><td></td><td>BLEU-4</td><td>3.74</td><td>8.45</td><td>7.54</td><td>10.94</td><td>9.15</td><td>4.30</td><td>11.31</td><td>5.54</td><td>10.35</td><td>14.79</td><td></td></tr><tr><td></td><td>F1-Score</td><td>37.68</td><td>52.06</td><td>54.99</td><td>58.05</td><td>56.49</td><td>39.74</td><td>50.37</td><td>42.74</td><td>57.55</td><td>60.60</td><td></td></tr><tr><td>FactKG</td><td>Acc</td><td>65.97</td><td>73.93</td><td>76.95</td><td>78.39</td><td>80.43</td><td>71.65</td><td>78.98</td><td>72.79</td><td>80.38</td><td>84.33</td><td>108×</td></tr><tr><td>LaMP-4</td><td>ROUGE-1</td><td>14.24</td><td>16.15</td><td>16.98</td><td>16.98</td><td>16.21</td><td>14.98</td><td>18.12</td><td>14.72</td><td>17.08</td><td>20.30</td><td>474×</td></tr><tr><td></td><td>ROUGE-L</td><td>12.24</td><td>14.31</td><td>14.94</td><td>15.09</td><td>14.21</td><td>12.98</td><td>16.10</td><td>12.70</td><td>15.02</td><td>18.15</td><td></td></tr><tr><td></td><td>F1-Score</td><td>8.98</td><td>11.21</td><td>12.15</td><td>12.15</td><td>11.32</td><td>8.71</td><td>13.46</td><td>9.05</td><td>12.41</td><td>15.54</td><td></td></tr><tr><td></td><td>METEOR</td><td>7.13</td><td>9.78</td><td>10.95</td><td>10.95</td><td>9.85</td><td>6.99</td><td>10.11</td><td>7.28</td><td>11.21</td><td>11.41</td><td></td></tr><tr><td>LaMP-7</td><td>ROUGE-1</td><td>48.73</td><td>49.80</td><td>49.80</td><td>49.65</td><td>49.57</td><td>49.61</td><td>52.78</td><td>49.58</td><td>49.92</td><td>54.40</td><td>29×</td></tr><tr><td></td><td>ROUGE-L</td><td>43.13</td><td>44.11</td><td>43.81</td><td>43.77</td><td>43.71</td><td>43.73</td><td>46.84</td><td>43.68</td><td>44.00</td><td>48.37</td><td></td></tr><tr><td></td><td>F1-Score</td><td>36.10</td><td>37.28</td><td>37.28</td><td>37.26</td><td>37.14</td><td>37.17</td><td>42.99</td><td>36.87</td><td>37.55</td><td>44.96</td><td></td></tr><tr><td></td><td>METEOR</td><td>32.31</td><td>33.70</td><td>33.78</td><td>33.73</td><td>33.55</td><td>33.74</td><td>38.35</td><td>33.30</td><td>34.12</td><td>40.37</td><td></td></tr></table>

## E.1 MEMORY ARCHITECTURE

We port the BLIP-2 bridging module (Li et al., 2023) to document collections and report it as a Q-Former architecture ablation. Table 13 compares methods for constructing 16 memory slots from

N document embeddings on PMv2. Each method is trained from scratch using the hyperparameters in Table 4.

Global mean-pool. This baseline averages all document embeddings into one collection vector and copies it into all 16 slots.

Segmented mean-pool. This baseline partitions the ordered sequence of document embeddings into 16 adjacent groups of documents and averages each group to form one slot. The groups contain complete documents rather than subdivisions of individual documents.

Segmented with matched parameters. This method begins with the 16 slots produced by segmented mean-pooling and increases capacity to match MEMORILLA without learned queries or query conditioning.

Cosine top-16 with matched parameters. This method selects the 16 document embeddings with the highest cosine similarity to the question and uses them as the memory slots. Its slot processor matches the parameter budget of MEMORILLA.

Documents only. This Q-Former variant self-attends over its learned latent slots and then crossattends to all document embeddings without access to the question.

Q-Former. This method self-attends over the learned latent slots and question embedding before the slots cross-attend to all document embeddings.

MEMORILLA. Our method first conditions the learned slots on the question embedding and then uses the conditioned slots to cross-attend to all document embeddings. It does not use self-attention.

Query-independent pooling is weakest and parameters do not fix it, since segmented pooling at matched budget reaches 31.10 on short-answer against 44.04. The Q-Former, which also conditions on the question, nearly matches MEMORILLA on PMv2 (43.34 against 44.04). On FactKG, MEMORILLA also outperforms the Q-Former (82.56 against 80.48), despite the Q-Former using 268,856,320 parameters. The documents-only variant reaches 78.72 on FactKG with 268,856,320 parameters. Replacing every document embedding with Gaussian noise changes the documents-only result by only +0.04 while lowering MEMORILLA by 5.93 and the Q-Former by 5.11. The gain therefore comes from conditioning the slots before they read rather than from either interaction on its own.

## E.2 MEMORY CAPACITY AT TRAINING TIME

Figure 9 compares models trained from scratch at K=16 and K=128 under otherwise identical settings. All runs use the Qwen3-4B-Instruct-2507 decoder and the same hyperparameters reported in Table 4, with K as the only changed hyperparameter. Between K=16 and K=128, four of eight benchmarks move by less than a point at eight times the capacity and PMv2 and PV4 drop, so we set K=16 throughout. All runs are trained from scratch, so this concerns capacity rather than the deployed checkpoint.

## E.3 MEMORY TOKEN BUDGETS FOR INFERENCE

Figure 10 varies the number of memory tokens K at inference time without retraining. Both personalization benchmarks gain most from the last slots. PMv2 stays near its closed-book level through K=4 (26.58 against 25.64) and climbs to 44.60 at K=16, while PV4 starts near its closedbook level (57.66 against 58.30), dips at K=2 and 4, and climbs to 66.65. Neither shows saturation, which reflects slots dropped from a checkpoint trained at K=16 rather than headroom above it, since training at K=128 does not help. FactKG, TriviaQA and NarrativeQA rise with every doubling. PubMedQA and LaMP-4 remain roughly flat at low K and improve only from K=8, and LaMP-7 only from K=16. Across all three regimes, the required memory capacity tracks the degree of cross-document integration each task demands.

Table 12: Comparison with ComMer on personalized generation, against ComMer’s reported results. ComMer uses Gemma-2b while MEMORILLA uses Qwen3.
<table><tr><td>Method</td><td>Tok.</td><td>LaMP-4 ROUGE-L</td><td>LaMP-7 ROUGE-L</td></tr><tr><td>ComMer</td><td>128</td><td>17.50</td><td>45.00</td></tr><tr><td>MEMORILLA (4B)</td><td>16</td><td>17.72</td><td>47.04</td></tr><tr><td>MEMORILLA (8B)</td><td>16</td><td>18.15</td><td>48.37</td></tr></table>

Table 13: Memory architecture ablation. All methods use the frozen Qwen3-Embedding-4B encoder and Qwen3-4B-Instruct-2507 decoder and share the training recipe, prompts, decoding and scorer. They are trained from scratch, so values are not comparable to Table 11. The Q-Former variants use more parameters than MEMORILLA.
<table><tr><td>Method</td><td>Query aware</td><td>Params</td><td>PMv2 SA</td></tr><tr><td>Global mean-pool</td><td>no</td><td>6,566,400</td><td>28.94</td></tr><tr><td>Segmented mean-pool</td><td>no</td><td>6,566,400</td><td>29.68</td></tr><tr><td>Segmented, parameter-matched</td><td>no</td><td>190,149,120</td><td>31.10</td></tr><tr><td>Cosine top-16, parameter-matched</td><td>yes (hard)</td><td>190,149,120</td><td>33.86</td></tr><tr><td>Documents only</td><td>no</td><td>268,856,320</td><td>34.72</td></tr><tr><td>Q-Former</td><td>yes (soft)</td><td>268,856,320</td><td>43.34</td></tr><tr><td>MEMORILLA</td><td>yes (soft)</td><td>190,190,080</td><td>44.04</td></tr></table>

![](images/540583d043b959545098de50078d72ed37801ef66c278321b462c8f8518febce.jpg)  
Figure 9: Memory-token capacity. All models use a Qwen3-4B-Instruct-2507 decoder and are trained from scratch under identical settings, differing only in the number of memory tokens K (K=16 is the setting used throughout).

## E.4 FROZEN VS. LORA TRAINING

To separate the memory interface from generic adaptation, we train rank-130 LoRA adapters (Hu et al., 2021) on the Qwen3-4B-Instruct-2507 decoder with 268.4M parameters, comparable to the 194.1M parameters of the memory module paired with the 8B decoder. The adapters use the same training rows as MEMORILLA’s per-task fine-tuning and receive no documents in context. Figure 11 shows that MEMORILLA leads on every primary metric, although the PubMedQA and LaMP-7 gaps are within noise. Together with the noise test above, where MEMORILLA loses 5.93 on FactKG without real documents, this indicates that its gains rely on the stored evidence rather than on adaptation alone. Freezing both backbones keeps the store portable and the decoder untouched, at the cost of an encoder never optimized for compression.

## E.5 ROLE OF ENCODER SIZE

Figure 12 swaps the frozen document encoder from Qwen3-Embedding-4B to Qwen3-Embedding-8B, keeping the Qwen3-8B decoder, memory module and recipe fixed and training from scratch. Every score moves by at most 1.3 except PubMedQA, where the +3.00 gain corresponds to three of 100 questions, so a larger encoder from the same family has minimal effects on performance.

![](images/f33bde74248568bb59160ffe3fc4d26b83b7dd215a20ead947b1462376e08c5e.jpg)  
(a) FactKG (Acc)

![](images/fb39172154172ce5e0990d9ffb241b7acd0408b8adbe5e435c84d2f3ce1190cb.jpg)  
(b) TriviaQA (EM)

![](images/da68827c30819ec3bcf7e8591a4fdabd53be3898df7338a81fafdb73221d437b.jpg)  
(c) PMv2 (Acc)

![](images/ce77b69ee138443295ead27473696159f2d389ced84ba2b7227a53f2a4917644.jpg)  
(d) LaMP-7 (ROUGE-L)

![](images/7188ff6a2c9fdd2cf3c57fc8b5af36c00741f1ede33145657570eee4f3d25b12.jpg)  
(e) LaMP-4 (ROUGE-L)

![](images/4619047c2be17a1612246f6351c55ce459b806b05beb62886ee55accf6e0f7b2.jpg)  
(f) NQA (ROUGE-L)

![](images/f8264cfacffb94ad5fbfb3b397504cea31903e722a214039755dab347a2575f8.jpg)  
(g) PV4 (Acc)

![](images/59a480b49b14e20710b17e7008c6b10245be3eae0972d2189841f27a17323b2a.jpg)  
(h) PubMedQA (Acc)  
Figure 10: Memory token budget at inference. Performance of MEMORILLA with a frozen Qwen3- 8B decoder using the first K memory tokens of the checkpoints trained with the full 16 tokens budget.

## F COMPUTATIONAL ANALYSIS OF MEMORILLA

Consider a collection of N documents serving Q queries. MEMORILLA pays the encoder once on the write path, producing N embeddings that persist and are reused by every later query. Dense retrieval builds the same index, so this cost is shared rather than added. The two methods differ on the read path. RAG places the selected text in the decoder, so its prefill grows with retrieval depth, while MEMORILLA runs one cross-attention read over the stored embeddings and always hands the decoder K=16 tokens. Table 14 reports measurements on one H100-80GB at batch size 1 with a Qwen3-8B decoder. Growing the collection by 512× decreases the memory-module forward time by 0.06 ms and MEMORILLA end-to-end latency by 1.9 ms, while full-context decoding grows by 19,184 ms and exceeds the enforced 32,768-token limit, requiring YaRN. Transfer stays under 0.04% of query latency because a 200 document collection is 1 MB, and the store is 7,266× smaller than the corresponding KV cache on PV4 (50,455 document tokens at 144 KiB per token for Qwen3-8B in bf16, 6.9 GiB, against 200 embeddings of 2,560 bf16 values). Absolute latency is dominated by decoding rather than by memory, so the benefit is not a faster single query but a serving cost that stays flat as collections grow and amortizes across the Q queries that read them.

![](images/ba43f5cb5f21aa9de12fb1cb4db08d90daeb7c12223bc4c95962110f15335e85.jpg)  
Figure 11: Memory versus comparable-budget adaptation. MEMORILLA with an 8B decoder and 194.1M trainable parameters against a no-document rank-130 LoRA adapter with a 4B decoder and 268.4M trainable parameters.

![](images/601412b651a2041e90c3ab7e86efc063dbdeb8ab1c244ca94eaf3e945fa2129d.jpg)  
Figure 12: Encoder size. The memory module is trained from scratch with a frozen Qwen3-8B decoder and either encoder; all other settings are fixed.

Table 14: Serving cost as the collection grows. One NVIDIA H100-80GB, batch size 1, bf16, greedy, max\_new\_tokens=32, 250-token documents; median of 5 timings after 2 warmups. The write path is paid once per document and is shared with dense retrieval; read path and decoder cost are paid per query. Full Context at N=512 is 128,064 context tokens, past the enforced 32,768-token limit, so it uses YaRN (factor 4). Timings use HF transformers generate without a paged KV cache, so Full Context latency depends on the serving stack.
<table><tr><td>Path</td><td>Quantity</td><td>N=1</td><td>N=512</td><td>Growth</td></tr><tr><td rowspan="2">Write</td><td>Document encoding</td><td>22.92 ms</td><td>2,588.84 ms</td><td>linear in N</td></tr><tr><td>Storage per document</td><td>5KB</td><td>5KB</td><td>flat</td></tr><tr><td rowspan="2">Read</td><td>Memory module forward</td><td>2.01 ms</td><td>1.96 ms</td><td>-0.06 ms</td></tr><tr><td>CPU to GPU transfer</td><td>0.014 ms</td><td>0.060 ms</td><td>+0.046 ms</td></tr><tr><td rowspan="3">Decoder</td><td>Context tokens</td><td>16</td><td>16</td><td>flat</td></tr><tr><td>MEMORILLA end to end</td><td>740 ms</td><td>739 ms</td><td>-1.9 ms</td></tr><tr><td>Full Context end to end</td><td>749 ms</td><td>19,932 ms</td><td>+19,184 ms</td></tr></table>

## G CONTINUAL LEARNING WITH MEMORILLA: ADDITIONAL DETAILS

TextWorld games require an agent to maintain state across a partially observable trajectory, navigate a map, manipulate objects, and execute ordered action sequences. Observations are room descriptions and action outcomes, actions are short text commands, and games span up to 65 rooms. From the second turn on, MEMORILLA embeds each completed turn as one document, holding its observation truncated to 200 characters and its action (74 tokens on average), reads these documents with the current observation as the query, and places the resulting K=8 memory vectors before the prompt. The 20 extreme, huge, long and mega test games instead embed one document per five completed turns with full observations, keep the last four documents, and use the mean document embedding as the query. Figure 13 shows a prompt and the documents compressed into the memory vectors, and Section 6.1 reports how the trajectory grows across a rollout.

We train on 800 prompts and evaluate on held-out validation and test splits of 40 games each (30 distinct, since the five medium and five hard games appear twice). Games come in six tiers (medium, hard, long, mega, huge, extreme) with 15–65 rooms and 6–74-step quests; the 800 training prompts are 202 medium, 558 hard and 10 from each longer tier. Rollouts sample up to 50 environment turns with at most 128 new tokens per turn, and evaluation samples three completions per held-out game at temperature 0.6 using the same SkyRL (Cao et al., 2025) generation harness for every condition. The random-memory baseline uses the MEMORILLA architecture with untrained weights, so it adds eight latents to the same prompt without having learned what to place in them. Table 15 lists the full configuration.

Decoder input when choosing the turn-6 action ([P] marks the eight memory positions)   
[P][P][P][P][P][P][P][P]<|im\_start|>system   
You play TextWorld efficiently.   
Use memory summaries when useful and avoid repeating failed actions.   
Return one final action command in the format [ACTION: <command>].<|im\_end|>   
<|im\_start|>user   
What is your next action?<|im\_end|>   
<|im\_start|>user   
<ASCII title banner, objective and first room description><|im\_end|>   
<|im\_start|>assistant   
... [ACTION: go south]<|im\_end|>   
<|im\_start|>user   
-= Laundromat =- You’ve entered a laundromat. ...<|im\_end|>   
... turns 2-5: assistant actions and user observations ...   
<|im\_start|>user   
-= Canteen =- You arrive in a canteen. ...<|im\_end|>   
<|im\_start|>assistant

![](images/534bdcc91d3ff613bb8d95923270aced843d8022455cd6fcd22106cea760a362.jpg)  
Query: the full turn-6 observation (“-= Canteen =- You arrive in a canteen. A typical one. . . . ”). Output: [ACTION: go east]  
Figure 13: An example TextWorld prompt. A winning 12-turn rollout on a held-out medium game, at the choice of the turn-6 action; training prompts have the same format. The eight memory vectors sit before the system prompt and are recomputed every turn from one document per completed turn, with the current observation as the query, while the full trajectory stays in the prompt. Line breaks inside the documents are added for width.

## G.1 REINFORCEMENT-LEARNED MEMORY OBJECTIVE

For agentic settings, we train the memory module from environment rewards while keeping the encoder and decoder frozen (Figure 7). At each TextWorld turn, the interaction trajectory is embedded as a document collection, $f _ { \mathrm { m e m } }$ maps it to memory vectors, and the frozen decoder samples the next action. We optimize only $\theta _ { \mathrm { { m e m } } }$ with Group Relative Policy Optimization (GRPO) (Shao et al., 2024), a critic-free variant of PPO (Schulman et al., 2017). For each prompt, a group of G rollouts receives scalar rewards $R _ { i }$ from the environment, and advantages are standardized within the group:

$$
A _ { i } = { \frac { R _ { i } - { \mathrm { m e a n } } _ { j \in G } ( R _ { j } ) } { \operatorname { s t d } _ { j \in G } ( R _ { j } ) + \epsilon } } .\tag{8}
$$

The memory policy is then updated with the clipped PPO surrogate over generated action tokens,

$$
\mathcal { L } _ { \mathrm { R L } } ( \theta _ { \mathrm { m e m } } ) = - \mathbb { E } \left[ \operatorname* { m i n } \left( r _ { i } A _ { i } , ~ \mathrm { c l i p } ( r _ { i } , 1 - \epsilon _ { c } , 1 + \epsilon _ { c } ) A _ { i } \right) \right] , \quad r _ { i } = \frac { \pi _ { \theta _ { \mathrm { m e m } } } ( a _ { i } \mid \tau _ { i } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( a _ { i } \mid \tau _ { i } ) } .\tag{9}
$$

The reward of each turn is the TextWorld game score after it, without shaping, and a rollout’s return $R _ { i }$ sums these over its turns. Most medium and hard games have a single one-point quest and end when it is solved, so they return 1 for a win and 0 otherwise; games in the four longer tiers (long, mega, huge and extreme) have one to three quests and also score as each is completed. The injection mechanism is unchanged from the supervised setting, while the memory module adds one self-attention layer over the trajectory documents before its cross-attention blocks. Memory here supplements the trajectory rather than replacing it, distinguishing it from the compression results in Section 4. At turn $t ,$ the decoder input follows Eq. (6) with the query replaced by the trajectory $\tau ^ { ( t ) }$ which contains the game description and all prior observations and actions:

$$
\mathbf { X } _ { \mathrm { L L M } } ^ { ( t ) } = [ \mathbf { m } _ { 1 } ^ { ( t ) } , \ldots , \mathbf { m } _ { K } ^ { ( t ) } ; \mathbf { P } _ { \tau ^ { ( t ) } } ] ,\tag{10}
$$

where the memory tokens $\mathbf { m } _ { k } ^ { ( t ) }$ are recomputed at every turn after the first from the completed turns of $\tau ^ { ( t ) }$ , with the current game observation as the query $q$ (the mean document embedding in the longer games).

Table 15: TextWorld RL configuration. The memory module is the only trainable component, and the full textual history is retained in every condition for a fair comparison.
<table><tr><td>Component</td><td>Setting</td><td>Value</td></tr><tr><td>Environment</td><td>Task suite</td><td>TextWorld held-out games up to 65 rooms</td></tr><tr><td></td><td>Splits</td><td>800 train prompts; 40-prompt validation and test splits (30 games each)</td></tr><tr><td></td><td>Agent state</td><td>Initial description plus prior observations/actions</td></tr><tr><td></td><td>Reward</td><td>TextWorld game score; no shaping reward</td></tr><tr><td>Model interface</td><td>Decoder</td><td>Frozen Qwen3-4B-Instruct-2507</td></tr><tr><td></td><td>Trajectory encoder</td><td>Frozen Qwen3-Embedding-4B</td></tr><tr><td></td><td>Memory budget</td><td>K=8 latent vectors inserted into decoder prompt</td></tr><tr><td></td><td>Memory architecture</td><td>8 heads; 1 self-attention layer; 2 cross-attention layers</td></tr><tr><td></td><td>Context parity</td><td>Same input budget for every condition: 16,000 tokens in training, 4,096 in evaluation; full history kept for ICL</td></tr><tr><td>Optimization</td><td></td><td>Memory module only; decoder and encoder frozen</td></tr><tr><td></td><td>Trainable parameters</td><td>Random (Xavier-uniform slots); output projection  $\mathcal { N } ( 0 , 0 . 0 1 )$ </td></tr><tr><td></td><td>Initialization</td><td>not loaded from the supervised checkpoint</td></tr><tr><td></td><td>RL algorithm</td><td>GRPO; group size G=16; normalized group advantages</td></tr><tr><td></td><td>Policy objective Optimizer</td><td>PPO clip 0.2; token-mean loss; no critic/KL/entropy</td></tr><tr><td></td><td></td><td>AdamW; learning rate  $5 \times 1 0 ^ { - 4 } ;$  no warmup</td></tr><tr><td>Rollouts</td><td>Training batch</td><td>4 prompts/step; 16 sampled rollouts/prompt</td></tr><tr><td></td><td>Horizon Action generation</td><td>Up to 50 environment turns/rollout</td></tr><tr><td></td><td></td><td>Up to 128 new tokens/turn with  $\mathrm { S k y R L }$ </td></tr><tr><td>Evaluation</td><td>Sampling</td><td>3 samples/held-out game at temperature 0.6</td></tr><tr><td></td><td>Metrics</td><td>avg@3 mean solve rate; pass@3 any-sample solve rate</td></tr></table>