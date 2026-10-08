# QCATS: Query Context-Aware Transformer Slicing for Efficient Predictive Query Processing

Yueying Li<sup>1</sup>, Zhongle Xie<sup>1,2,</sup> <sup>\*</sup>, Ke Chen<sup>1,2</sup>, Lidan Shou<sup>1,2</sup>

<sup>1</sup>The State Key Laboratory of Blockchain and Data Security, Zhejiang University; <sup>2</sup>Hangzhou High-Tech Zone (Binjiang) Institute of Blockchain and Data Security {zjuyueyingli, xiezl, chenk, should}@zju.edu.cn

Abstract—In-database predictive query processing increasingly applies Transformer-based models within relational pipelines. However, existing in-database inference typically exposes only tuple-level model inputs to the inference runtime, leaving relational predicates and metadata statistics invisible to neural execution planning. In this paper, we propose QCATS, a query contextaware transformer slicing framework that enables efficient sparse inference inside database systems. QCATS executes at query granularity: instead of routing individual tokens or tuples during inference, it uses query predicates and metadata statistics to preselect context-aligned FFN slices before model execution. The framework comprises offline expert construction and lightweight query-level routing that dynamically selects experts during execution. QCATS further introduces system optimizations, including asynchronous CPU–GPU pipelines and routing-aware batching. Experiments on four predictive-query workloads with BERT-base and Qwen-0.6B show that QCATS achieves up to 4.42× latency reduction while preserving prediction accuracy comparable to dense baselines.

Index Terms—In-database Learning, Model Slicing, Query Processing

## I. INTRODUCTION

Modern analytical databases integrate Transformer-based predictive operators into SQL pipelines to process unstructured text alongside relational data. Known as in-database predictive querying [1], [2], this paradigm supports semantic analytics through cloud inference APIs or UDFs [3]–[5]. These queries typically produce short outputs, such as labels or scores, for many SQL-selected tuples, making accumulated tuple-level inference, rather than autoregressive decoding, a dominant cost of query execution.

Opportunities to reduce this cost arise from the structure of predictive queries: inference operates on subsets selected by SQL predicates. These predicates, together with subset statistics, define a semantic space that often differs from the global data distribution, which we refer to as query context. For Transformer models, different query contexts induce distinct activation patterns in feed-forward networks (FFNs) [6]. As illustrated in Fig. 1, sentiment analysis over food reviews activates a different subset of neurons from the same task on clothing or electronics reviews, suggesting that only a fraction of model capacity is required for many query workloads. We further empirically verify this phenomenon in Fig. 2 using eight queries, details are provided in the appendix. For each query, we feed its selected text tuples into the fine-tuned BERT model and compute the average activation of each FFN neuron, producing one row of the heatmap. Each column represents an FFN neuron, and darker colors indicate higher average activation. A small subset of neurons remains highly activated across the queries and is therefore placed on the left. Beyond this shared subset, highly activated regions appear at different neuron positions for different query contexts, indicating that distinct queries activate different subsets of neurons. This observation motivates selecting context-aligned FFN slices according to the query context.

![](images/ffbac0ce999af78fe512235a2310989825a8ece76bcc38370465754985196368.jpg)  
Fig. 1: FFN Activation Varies Across Query Contexts.

![](images/7f3edee0c1b134f7d8ae5577601eb9026ba6cd9df390d7b22731656743735d25.jpg)  
Fig. 2: Average FFN neuron activations for text tuples selected by eight queries (Q1–Q8).

However, existing database systems typically execute inference through black-box user-defined functions (UDFs) [3]– [5], transmitting only raw text data to the model while discarding relational constraints and metadata statistics. Even when the inference engine applies generic batching or kernellevel acceleration, these optimizations are unaware of the relational context that determines which tuples are semantically similar. Consequently, models operate in a context-agnostic manner and execute dense inference for every input tuple. This mismatch leads to substantial waste in computation and memory resources in predictive query processing. That is, the computation repeatedly activates the full model despite the query-level locality of predictive workloads.

From a systems perspective, the inefficiency motivates conditional execution, where only computation relevant to the current query context is activated during inference. Analogous to predicate pushdown [7]–[9] and column pruning [10], [11], which avoid irrelevant data movement and attribute access, query-aware neural execution should avoid irrelevant parameter access and activation computation. Unfortunately, existing neural inference pipelines lack mechanisms to translate query semantics into deterministic and database-friendly conditional model execution.

Sparse activation techniques like Mixture-of-Experts (MoE) [12], [13] partially address this issue by enabling conditional computation within neural networks. However, they rely on token-level, layer-wise routing during inference, which introduces control-flow divergence and dynamic data redistribution overheads, degrading GPU utilization [14], [15]. In database workloads, this overhead is amplified because tuples selected by the same query are naturally processed in large batches; token-level routing breaks this homogeneity and prevents the executor from constructing a single static inference path for the batch. Furthermore, it introduces additional buffering overhead for fragmented tokens across experts, causing inflated activation memory and unpredictable resource demands.

Despite extensive research on sparse inference and expert routing, existing approaches [16]–[18] are primarily designed for standalone model serving and do not exploit the semantic structure exposed by database queries. To our knowledge, no prior system leverages query-level context to enable deterministic, database-integrated sparse execution of Transformer models. This gap prevents current solutions from achieving both high inference efficiency and tight integration with relational execution pipelines.

Beyond model-level limitations, predictive query processing also introduces system-level challenges. CPU-side preprocessing, including tokenization and data preparation, can become a bottleneck when GPU inference is accelerated on-the-fly. Meanwhile, long-tailed text length distributions [19] cause excessive padding overhead under conventional batching, while synchronous execution pipelines provide insufficient CPU– GPU overlap, leading to pipeline stalls and underutilized hardware resources.

In this work, we propose Query Context-Aware Transformer Slicing (QCATS), a framework that rethinks sparse inference for Transformer-based models from a database-first perspective. Instead of performing routing inside the model, QCATS shifts routing decisions to the query level. By leveraging the query context, QCATS pre-determines the subset of model parameters that will be activated before inference begins. This design enables static, deterministic routing, avoids runtime control-flow divergence, and aligns sparse execution with the granularity of database workloads. In this sense, QCATS treats model inference as a query physical operator whose execution path can be specialized using optimizer-visible relational context before tuple-level inference begins.

QCATS consists of two tightly integrated components. First, it transforms a fine-tuned dense model into a library of context-aligned expert slices by exploiting FFN activation sparsity, enabling sparse inference without retraining models from scratch. Second, it introduces a lightweight contextaware router that maps query contexts directly to expert combinations. By decoupling routing from execution, QCATS ensures contiguous memory access via homogeneous batching, thereby avoiding the extra buffering overhead and reducing peak activation memory. To further address system-level bottlenecks, we design an asynchronous CPU–GPU execution pipeline with hierarchical bucketing that masks preprocessing overhead and reduces padding-induced waste. We summarize our main contributions as follows:

• We formulate query context-aware transformer slicing for in-database predictive query processing and propose QCATS, which uses query predicates and metadata statistics to specialize neural inference execution paths.

• We introduce a parameter decomposition strategy that partitions FFN layers into shared and semantic components, enabling efficient sparse inference from a fine-tuned dense Transformer without constructing separate expert networks from scratch.

• We design a lightweight query-level routing mechanism that maps query contexts to sparse expert combinations, shifting conditional computation from token-level dynamic routing to query-level static routing.

• We integrate QCATS into a database-oriented execution pipeline with routing-aware batching and asynchronous CPU–GPU execution. Experiments on four workloads and two Transformer backbones show significant latency and memory reductions while preserving prediction accuracy.

The rest of the paper is organized as follows. §II defines predictive queries and formulates query context-aware transformer slicing. §III presents QCATS and §IV introduces database-oriented execution optimizations. §V reports the experimental evaluation. §VI reviews related work, and §VII concludes the paper.

## II. FOUNDATIONS

In this section, we establish the foundations required to reason about query-aware model slicing in database systems.

## A. Predictive Query

We consider predictive queries in analytical databases, which integrate relational filtering with model inference.

Let a relation T consist of text attributes ${ \mathcal { X } } _ { \mathrm { t e x t } }$ and structured metadata $\mathcal { X } _ { \mathrm { m e t a } } .$ The metadata describes the context of each text record, such as product category, brand, or user profile in e-commerce scenarios. Crucially, our framework is broadly applicable to any scenario where the model input is shaped by

(c) With QCATS (Ours)  
(b) With MoE Model  
![](images/b40132e991d0520c00faa799b599cb4cd85264f83135fe6c5e745ac63da2be74.jpg)  
Fig. 3: Predictive Query Processing in Database.  
Fig. 4: Overview of QCATS.

SQL evaluation over the metadata. Given relational constraints Φ over $\chi _ { \mathrm { m e t a } } ,$ a predictive query Q first obtains an intermediate subset $\mathcal { T } _ { \Phi ( \mathcal { X } _ { \mathrm { m e t a } } ) }$ through standard relational operators, including WHERE, JOIN, and GROUP BY, and then applies a Transformer-based model M to Xtext.

Formally, we define the predictive query $Q$ as an extended projection over the metadata-constrained subset:

$$
\mathcal { T } _ { \mathrm { s c o r e d } } = \pi _ { * , \mathcal { M } ( \mathcal { X } _ { \mathrm { t e x t } } )  Y } ( \mathcal { T } _ { \Phi ( \mathcal { X } _ { \mathrm { m e t a } } ) } )\tag{1}
$$

where $\mathcal { T } _ { \Phi ( \mathcal { X } _ { \mathrm { m e t a } } ) }$ denotes the intermediate subset materialized by the database engine under metadata constraints Φ, and M evaluates ${ \mathcal { X } } _ { \mathrm { t e x t } }$ to append a prediction column Y.

We define query context $\mathcal { C } _ { Q }$ as a composite representation:

$$
\mathcal { C } _ { Q } = \left( \Phi , S ( \mathcal { T } _ { \Phi } ) \right)\tag{2}
$$

where Φ denotes explicit relational constraints, and $S ( \mathcal T _ { \Phi } )$ captures statistics over the post-constraint subset $\mathcal { T } _ { \Phi }$ . The context $\mathcal { C } _ { Q }$ specifies the semantic scope of model inference and can be materialized into dense features via feature engineering or lightweight encoders, as seen in §III-B. Note that existing UDF-based inference processes tuples independently and does not expose such relational context to the inference pipeline.

## B. Core Idea and Goal

Transformer feed-forward networks (FFNs) dominate model parameters and inference cost [13], [20]. Yet, their activations are highly sparse: prior studies show that retaining a modest subset of neurons can preserve most output magnitude and predictive performance [21]–[23]. This redundancy motivates sparse FFN execution. Conditional computation mechanisms such as Mixture-of-Experts (MoE) exploit this redundancy by replacing dense FFNs with expert networks $\boldsymbol { E } _ { 1 } , \ldots , \boldsymbol { E } _ { N }$ controlled by a routing function $G ( \cdot )$ . Formally, for an input hidden state x, the output is:

$$
\mathbf { y } = \sum _ { i = 1 } ^ { N } G ( \mathbf { x } ) \cdot E _ { i } ( \mathbf { x } ) .\tag{3}
$$

Existing MoE architectures are primarily designed for largescale pre-training and rely on token-level, layer-wise routing, in which routing decisions are computed dynamically for individual tokens. While effective for scaling model capacity, this fine-grained routing mechanism introduces non-trivial dynamic scheduling and dispatching overhead and fails to exploit the semantic coherence exposed by database query execution.

As illustrated in Fig. 3, our key insight is that predictive queries naturally enable query-level conditional execution. Different from token-level conditional computation in common MoE design (Fig. 3b), which adapts execution paths independently for each input, query-level conditional execution (Fig. 3c) leverages the semantic consistency of query-selected subsets and applies a unified execution configuration across all tuples in $\mathcal { T } _ { \Phi }$

Under this paradigm, the query context acts as a highlevel routing control signal for model inference activation. Routing signal can be computed before inference, enabling static execution paths, homogeneous batch construction, and tighter integration with relational execution pipelines.

Goal. We focus on predictive queries that apply Transformer-based models independently to many tuples. Our objective is to reduce the accumulated inference cost over $\mathcal { T } _ { \Phi }$ rather than improving autoregressive generation and servingspecific mechanisms of the Transformer itself.

We formalize our task as Query Context-Aware Transformer Slicing problem. Let θ denote the parameter set of a Transformer model M. Given a predictive query Q with constraints Φ, our goal is to generate a sliced model $\mathcal { M } _ { \Phi }$ that selects a sparse parameter subset $\theta _ { \Phi } \subset \theta$ for efficient query execution. The sliced model should satisfy two goals:

• Execution Efficiency (G1). The sliced model should reduce inference latency and peak memory footprint when integrated into the database query pipeline.

• Accuracy Preservation (G2). The activated slice should preserve the predictive performance of the full model on the target subset $\mathcal { T } _ { \Phi }$ within a small tolerance margin.

## III. QCATS

We propose Query Context-Aware Transformer Slicing (QCATS), a framework that shifts model inference from dense execution to query-adaptive sparse execution. Unlike traditional MoE with token-level routing, QCATS adopts querylevel routing to pre-activate semantic parameter subsets, reducing computation and VRAM usage while preserving predictive accuracy. As shown in Fig. 4, QCATS consists of two stages: offline expert construction and query-time dynamic assembly. Note that the offline construction is performed once while the online path performs only lightweight routing and sliced inference.

Stage I: Offline Sparsity-Based Expert Construction.

QCATS exploits FFN activation sparsity to extract sparse expert slices from a fine-tuned dense Transformer. These slices form an expert library E that supports conditional execution while preserving the model’s semantic capability, as elaborated on §III-A.

Stage II: Query-Time Context-Aware Inference. At query time, QCATS extracts relational constraints and metadata statistics as query context, maps them to expert weights through a lightweight router, and executes inference using the selected slices. The inference pipeline, as detailed in §III-B consists of Query Processing, Query Context Featurization, Context-Aware Expert Routing, and Sliced Model Execution.

## A. Stage I: Expert Construction

QCATS begins with a one-off structural reconfiguration step performed during system initialization to prepare the dense transformer model for query-level conditional execution. By analyzing activation patterns in FFNs, we decompose the original model into a library of physically isolated Expert Slices as illustrated in Fig. 5. This offline transformation converts monolithic FFN layers into modular computation units that can be selectively activated at query time. The resulting Expert Library provides a stable execution substrate for query-level routing and selective parameter activation, enabling efficient, deterministic, and database-friendly sparse inference.

To construct specialized experts, QCATS first identifies distinct semantic regions from the text data $X _ { \mathrm { t e x t } }$ . Directly using all data samples introduces noise due to overlapping topics and ambiguous semantic boundaries, which can blur the specialization signals required for expert construction. Instead, we choose to extract prototypical samples. Specifically, we employ a lightweight embedding model (FastText [24]) to extract embeddings and apply K-Means clustering to partition the data into N semantic groups $\mathcal { D } _ { 1 } , \dots , \mathcal { D } _ { N } \ ^ { 1 }$ . For each group, we retain only the samples closest to the corresponding cluster centroid to reduce noise. These representative samples provide stable and discriminative activation patterns for subsequent neuron analysis, ensuring that each expert slice is anchored to a well-separated semantic domain.

Using these samples, we profile the internal activation behavior of the fine-tuned dense model. We perform forward passes on data from each semantic group $\mathcal { D } _ { n }$ and collect neuron activation statistics across FFN layers. This profiling step maps semantic variations in the input space to structured patterns in the neuron activation space, producing the activation signals required to identify different types of neurons.

Neurons in FFN layers exhibit heterogeneous functional roles [6]: some encode general linguistic patterns that are broadly useful across domains, while others respond selectively to specific semantic regions. To preserve both generalization capability and domain specialization, QCATS explicitly separates these two types of neurons. Concretely, we partition FFN neurons into a shared set $U _ { \mathrm { s h a r e d } }$ , which captures universally useful computation, and multiple semantic sets $U _ { \mathrm { u n i q u e } } ^ { ( n ) } ,$ each corresponding to a distinct semantic group $\mathcal { D } _ { n } .$

First, to capture general knowledge shared across different domains, we aim to identify neurons that remain consistently active regardless of the specific semantic group. To quantify the universal utility, we calculate General Importance Score $S _ { j } ^ { \mathrm { g l o b a l } }$

$$
S _ { j } ^ { \mathrm { g l o b a l } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } A _ { j } ^ { ( n ) }\tag{4}
$$

where $A _ { j } ^ { ( n ) }$ denotes the average activation magnitude of neuron $j$ in semantic group n, and N is the total number of groups. Neurons with the highest scores are then selected to form the shared set $U _ { \mathrm { s h a r e d } }$ , ensuring the retention of fundamental model capabilities while providing a stable foundation for subsequent semantic specialization.

Subsequently, to capture semantic knowledge, we need to identify neurons that exhibit high specificity. Such neurons are highly active in a specific semantic group but suppressed in other groups. To quantify this contrastive property, we define Semantic Score $R _ { j , n }$

$$
R _ { j , n } = \frac { A _ { j } ^ { ( n ) } } { \frac { 1 } { N - 1 } \sum _ { m \neq n } A _ { j } ^ { ( m ) } + \epsilon }\tag{5}
$$

where the denominator represents the background noise level from other groups. By filtering for neurons with top-ranked $R _ { j , n }$ values, we construct the semantic set $U _ { \mathrm { u n i q u e } } ^ { ( n ) }$ for each semantic group, ensuring that each expert specializes in its corresponding group.

Based on the selected neuron sets, we physically reconstruct the original dense parameter matrix W into N isolated Expert Slices, namely the tensor set $\{ \mathbf { W } ^ { ( 1 ) } , \ldots , \mathbf { W } ^ { ( N ) } \}$ . For standard Transformer FFN layers composed of up projection $\mathbf { W } _ { u p } ,$ gate projection $\mathbf { W } _ { g a t e } ,$ and down projection $\mathbf { W } _ { d o w n }$ , we apply neuron selection consistently across all three matrices to ensure structural integrity. Specifically, the n-th expert slice $\mathbf { W } ^ { ( n ) }$ is composed of the union of the shared set and the corresponding semantic set, i.e., $U _ { \mathrm { s h a r e d } } \cup U _ { \mathrm { u n i q u e } } ^ { ( n ) }$ . This design allows each expert slice to independently retain both general linguistic capabilities and semantic expertise.

![](images/544c7726e1876af2a871c0e33f22a51435cba89c25c144f5ed335672b7f16e85.jpg)  
Fig. 5: Illustration of Expert Construction.

## B. Stage II: Context-Aware Inference

At query time, QCATS extracts the query context $\mathcal { C } _ { Q }$ from each incoming query. By matching it against the N preestablished semantic groups, the system routes the query to the corresponding expert for efficient inference. Specifically, we represent $\mathcal { C } _ { Q }$ as a dense context embedding $e _ { \mathrm { c t x } }$ through feature engineering and a lightweight encoder, then use it to drive expert routing. This process comprises the following four key steps:

1) Query Processing: Raw predictive queries often interleave lightweight relational operators (e.g., JOIN, GROUP BY) with computationally expensive neural UDFs. Because the data fed into the neural model is strictly pre-filtered by these SQL operators, QCATS must first parse the query to identify these filtering boundaries. We formally extract these relational filtering signals as explicit constraints, denoted as Φ. This query parsing and semantic-aware logical rewriting serve two main purposes: minimizing the data subset fed into the inference stage and providing highly selective semantic context Φ for downstream routing.

A major challenge in extracting a clean Φ arises when neural UDFs are embedded within predicates. In database query processing, the query parser translates an input SQL into a logical query plan represented as an Abstract Syntax Tree (AST), where relational predicates and UDF invocations coexist as distinct operator nodes. A conventional optimizer may treat neural UDF predicates similarly to scalar predicates or lack sufficient cost information to delay them appropriately. To prevent this, QCATS introduces a deferred evaluation rule. As shown in Fig. 6a, it rewrites the AST by moving the UDF predicate (σ) upward, past lightweight scalar constraints and JOINs, delaying its execution. This deferral naturally isolates all preceding relational operators into the explicit constraints Φ. Consequently, the UDF acts as a final semantic filter evaluated on a reduced subset, while Φ provides a precise semantic context for routing. Crucially, features are always extracted from the subset that the UDF actually processes, so the routing input remains consistent with the true inference domain regardless of how the rewrite reshapes the data.

For queries involving logical partitioning (e.g., GROUP BY), QCATS abandons standard tuple-at-a-time inference (Fig. 6b). Instead, the execution engine physically partitions intermediate tuples into dense semantic subsets prior to model inference. The context-aware router then treats each pre-aggregated subgroup as an independent batch, activating distinct sliced models. To balance context-awareness with overhead, QCATS ranks the pre-aggregated groups by their cardinalities in the query-filtered subset, assigns dedicated expert slices only to the few dominant groups, and merges the remaining tail groups into a shared routing path.

Ultimately, this parsing and rewriting phase outputs a refined logical plan alongside the clearly isolated explicit constraints Φ. These elements define the exact semantic boundaries of the target data, providing a precise, noise-free foundation for the subsequent featurization phase.

2) Query Context Featurization: Discrete SQL constraints and structured metadata are semantically heterogeneous and cannot be directly processed by a neural router. To bridge the gap between relational structures and continuous embeddings, QCATS performs a two-phase preparation procedure: Feature Selection, which identifies informative metadata attributes, and Context Representation, which transforms them into unified vector representations suitable for downstream routing.

One-time Feature Selection. Database tables often contain redundant or weakly informative metadata that contributes little to semantic understanding. Feeding all columns to the router introduces noise and computational overhead. To address this issue, we design a dual-view scoring mechanism that identifies a compact subset of highly discriminative attributes. Importantly, this process is executed as a one-off offline task and incurs no overhead during query-time inference.

To quantify the global discriminative power of each column, we train a lightweight classifier using metadata attributes as input features. Specifically, we measure the global relevance score of each column using Gini importance derived from a Random Forest model [25] trained to predict the context groups $\left\{ \mathcal { D } _ { n } \right\}$ built in §III-A.

In ambiguity zones, textual features alone are insufficient. We must identify columns that distinguish samples where text fails to do so.

To this end, we construct a KNN [26] index over the training set of the target downstream task using text embeddings, and retrieve each sample’s k nearest neighbors. Here, the downstream task labels are the ground-truth labels provided in that training set (e.g., sentiment labels in a sentiment analysis dataset of product reviews). A sample pair is marked as a conflict pair if the two samples are close in embedding space yet carry different labels from this training set, indicating that textual similarity alone cannot reliably discriminate them. We then compute the average value discrepancy of each metadata column across all identified conflict pairs, and adopt this discrepancy as the local disambiguation score, prioritizing columns that differ systematically across such pairs and therefore may help resolve ambiguities left by text alone. Finally, we select the Top-M columns based on a weighted average of these scores, defining them as the effective context columns.

Schema-Agnostic Featurization. To ensure the router generalizes across diverse database tables, we require a standardized input format that abstracts away schema differences. We introduce an input featurization mechanism that maps diverse columns into a unified embedding space. Numerical values are processed via discretization-based encoding, while categorical fields utilize frequency-based encoding. The function $\mathcal F ( \cdot )$ generates two dense feature sets that enable the model to infer explicit intent against global data distributions without relying on fixed position rules:

![](images/84ebca03a7c8b64ee7e47843d513f7dbab1ee5b08f0d1c7a132438d8e3fd8d0c.jpg)  
(a) Query Rewriting  
(b) Grouped Query Processing  
Fig. 6: Specialized Query Processing in QCATS.  
Fig. 7: Architecture of Context-Aware Router.

(a) Implicit Metadata Features $\mathbf { F } _ { \mathrm { m e t a } } \colon$ This feature set captures the statistical characteristics of the selected context columns under the current dataset distribution. It encodes both column semantics and distributional profiles as:

$$
\mathbf { F } _ { \mathrm { m e t a } } = \{ \mathbf { e } _ { \mathrm { c o l } } ^ { ( m ) } , \mathbf { e } _ { \mathrm { s t a t } } ^ { ( m ) } \} _ { m = 1 } ^ { M }\tag{6}
$$

where M denotes the number of selected context columns. For the m-th column, ${ \bf e } _ { \mathrm { c o l } } ^ { ( m ) }$ represents the column name embedding, and ${ \mathbf e } _ { \mathrm { s t a t } } ^ { ( m ) }$ encodes its statistical distribution properties.

(b) Explicit Predicate Features $\mathbf { F } _ { \mathrm { q u e r y } } .$ : This feature set captures the user’s specific filtering intent expressed in SQL predicates. To minimize noise from irrelevant attributes, we retain only predicates involving the selected context columns. If the query contains J such predicates, the representation is constructed as:

$$
\mathbf { F } _ { \mathrm { q u e r y } } = \{ \mathbf { e } _ { \mathrm { c o l } } ^ { ( j ) } , \mathbf { e } _ { \mathrm { v a l } } ^ { ( j ) } \} _ { j = 1 } ^ { J }\tag{7}
$$

where J is the number of valid predicates. For the j-th predicate, $\mathbf { e } _ { \mathrm { c o l } } ^ { ( j ) }$ and ${ \bf e } _ { \mathrm { v a l } } ^ { ( j ) }$ denote the embeddings of the target column and filtering value, respectively. Note that if there exist no relevant predicates, this feature set is empty, i.e., $J = 0 ,$

3) Context-Aware Expert Routing: To transform the query context features into specific expert configurations, we design a customized Context-Aware Router. Unlike traditional MoE architectures that perform token-level, layer-wise routing inside the model, our router operates as a decoupled and lightweight look-ahead module that determines expert activation prior to inference. Our design is inspired by hybrid attention architectures [27], but is specifically adapted to effectively process the non-sequential and structured nature of database content, as depicted in Fig. 7.

Since database columns are permutation-invariant, standard positional encodings are unsuitable as they impose artificial order. To preserve the native structure of tabular inputs, we discard absolute positioning and introduce Structural Type

Embedding (STE), enabling the router to capture relational semantics without spurious sequential dependencies.

Building on this structured input, we employ a specialized Transformer model interaction to distill semantic context. First, to capture global correlations among columns (e.g., the joint distribution between Brand and Price), we utilize a 2-layer Transformer encoder to process the implicit metadata features $\mathbf { F } _ { \mathrm { m e t a } } ,$ generating a contextually consistent global representation $\mathbf { H } _ { \mathrm { m e t a } }$

Next, to filter out redundant signals and focus solely on the query-specific intent, we modify a 2-layer Transformer decoder by repurposing the Multi-Head Cross-Attention (MHCA) mechanism. We initialize the query state $\mathbf { C } ^ { ( 0 ) }$ using the explicit predicate features $\mathbf { F } _ { \mathrm { q u e r y } }$ , or employ a learnable global intent embedding $\mathbf { e _ { g l o b a l } }$ to autonomously discern salient metadata when predicates are absent. With the encoded global metadata $\mathbf { H } _ { \mathrm { m e t a } }$ serving as the key and value, the decoder dynamically attends to metadata attributes that are most relevant to the current query context:

$$
\mathbf { C } ^ { ( l ) } = \mathrm { L a y e r N o r m } ( \mathbf { C } ^ { ( l - 1 ) } + \mathbf { M H C A } ( \underbrace { \mathbf { C } ^ { ( l - 1 ) } } _ { \mathrm { Q u e r y } } , \underbrace { \mathbf { H } _ { \mathrm { m e t a } } } _ { \mathrm { K e y } } , \underbrace { \mathbf { H } _ { \mathrm { m e t a } } } _ { \mathrm { V a l u e } } ) )\tag{8}
$$

where $\mathbf { C } ^ { ( l ) }$ denotes the context representation at decoder layer l.

The final stage transforms the variable-length semantic context into a discrete, efficient execution plan. We first apply Masked Mean Pooling on the decoder output to produce a fixed-length context embedding $\mathbf { e } _ { \mathrm { c t x } }$ . To explicitly bound inference cost, we then employ a Top-K Gating Network, which projects $\mathbf { e } _ { \mathrm { c t x } }$ through an MLP and activates only the Top-K most relevant expert slices:

$$
\mathbf { g } = \mathrm { T o p - K } \big ( \mathrm { S o f t m a x } \big ( \mathrm { M L P } ( \mathbf { e } _ { \mathrm { c t x } } ) \big ) \big )\tag{9}
$$

where $\mathbf { g } ~ \in ~ \mathbb { R } ^ { N }$ (with only K non-zero entries) represents the sparse gating vector containing the routing weights for the selected experts. The hyperparameter K explicitly governs the trade-off between model capacity and execution efficiency: a smaller K prioritizes computational sparsity, while a larger K preserves richer semantic nuances. This design effectively translates semantic understanding into a deterministic, hardware-friendly execution path, eliminating redundant parameter computations.

4) Sliced Model Execution: In the inference phase, once the context-aware router generates routing decisions, the system directly retrieves the activated Top-K expert tensors $\{ \dot { \mathbf { W } } ^ { ( k ) }$ $k \in \mathcal { E } _ { \mathrm { a c t i v e } } \}$ from the Expert Library. The output y of the sliced Transformer FFN layer is computed as:

$$
\mathbf { y } = \sum _ { k \in \mathcal { E } _ { \mathrm { a c t i v c } } } \mathbf { g } _ { k } \cdot E _ { k } ( \mathbf { x } )\tag{10}
$$

where x denotes the input hidden state and $\mathbf { g } _ { k }$ is the expert weight.

As mentioned previously, unlike token-level MoE execution, our sliced model execution is performed at query granularity and uses pre-constructed expert slices, avoiding dynamic control flow and runtime expert construction. To be specific, the design yields two key efficiency benefits. First, QCATS reduces computation associated with inactive expert slices by restricting FFN matrix multiplications to the selected parameter subset. Second, peak VRAM reduction is achieved because expert slices have substantially smaller dimensionality than the original dense FFN. Consequently, the size of intermediate activations decreases proportionally, eliminating the need to allocate large GPU memory buffers for the full-width network and thereby enabling memory-efficient deployment.

Discussion. The practicality of QCATS is demonstrated through its robust design and seamless system integration. Its architecture provides adaptive semantic coverage: queries with strong semantic signals are routed to dominant specialized experts, while weakly constrained queries benefit from weighted multi-expert collaboration. The shared neuron set $U _ { \mathrm { s h a r e d } }$ preserves fundamental linguistic capabilities, ensuring a stable performance floor across diverse query contexts.

It is worth noting that QCATS complements generalpurpose model-serving platforms and low-level execution optimizations. While vLLM [28] and SGLang [29] optimize autoregressive serving through KV-cache management and continuous batching, QCATS uses query context to reduce short-output inference costs across database tuples. QCATS also supports attention acceleration methods, such as FlashAttention [30], FastMoE [14], and quantization [31]–[33]. Decoupling the expert library from the router enables efficient execution under stable data distributions. As data and workloads gradually evolve, incremental fine-tuning of the lightweight router can support long-term deployment in dynamic OLAP scenarios without rebuilding the expert library.

## C. Decoupled Training Strategy

To operationalize QCATS, we must optimize two distinct components: the Context-Aware Router ψ and the Expert Library E. Since the router is initialized from scratch, while experts are physically derived from a dense backbone, naive joint optimization may lead to unstable training dynamics. Specifically, the router requires specialized experts to learn discriminative routing policies, whereas expert specialization depends on stable and consistent routing decisions.

To address this problem, we adopt a decoupled two-phase training strategy. Phase I establishes a reliable routing prior through teacher-guided semantic alignment, while Phase II leverages the fixed routing policy to drive physical expert specialization. This separation stabilizes optimization and enables effective coordination between routing and expert learning.

1) Phase I: Router Semantic Alignment via Activation Distillation: In this phase, we train Context-Aware Router by constructing supervised routing targets derived from the internal activation behavior of the fine-tuned dense model. While Section III-A determines the physical composition of expert slices, the semantic correspondence between query contexts and experts remains undefined. We bridge this gap by treating the dense model as a teacher and using its FFN activations as oracle signals to align the expert with the query context.

Given a query-selected subset $\mathcal { D } _ { q } .$ , we run the dense model and aggregate activation magnitudes over the unique neurons of each expert:

$$
s _ { \mathrm { r a w } } ^ { ( n ) } = \frac { 1 } { | \mathcal { D } _ { q } | } \sum _ { x \in \mathcal { D } _ { q } } \sum _ { j \in U _ { \mathrm { u n i q u e } } ^ { ( n ) } } | h _ { j } ( x ) |\tag{11}
$$

where $h _ { j } ( x )$ is the post-activation output of neuron $j .$ Note that shared neurons are excluded because they are active across most inputs and provide weak routing signals.

Even when focusing on unique neurons, disparities in absolute activation magnitudes persist across different experts. To cope with this issue, we propose a calibration strategy that acts as a weighted balance between the relative improvement and the raw activation magnitude:

$$
s _ { \mathrm { c a l } } ^ { ( n ) } = \frac { s _ { \mathrm { r a w } } ^ { ( n ) } - \mu _ { n } } { \mu _ { n } + \epsilon } + \eta \cdot s _ { \mathrm { r a w } } ^ { ( n ) }\tag{12}
$$

where $\mu _ { n }$ denotes the average activation magnitude of expert n across the training corpus and $\eta$ is a coefficient controlling the weight of the original signal. This combination preserves the importance of strong raw signals while emphasizing contextspecific spikes through the relative term.

$$
\mathbf { g } _ { n } ^ { * } = { \left\{ \begin{array} { l l } { \mathrm { N o r m a l i z e } ( { \mathrm { S o f t m a x } } ( \mathbf { s } _ { \mathrm { c a l } } / \tau ) ) _ { n } } & { { \mathrm { i f ~ } } n \in { \mathcal { E } } ^ { * } } \\ { 0 } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }\tag{13}
$$

where $\mathbf { g } _ { n } ^ { * }$ denotes the ground truth routing weight for the n-th expert, $\tau$ is a temperature hyperparameter controlling the sharpness of the probability distribution. This formulation ensures that the ground truth labels not only identify the most suitable experts $\mathcal { E } ^ { * }$ but also accurately reflect their relative importance as determined by the teacher model.

2) Phase II: Context-Driven Joint Expert Training: In the second phase, the context-aware router ψ is frozen and serves as a fixed coordinator that determines data-to-expert assignment. The optimization objective shifts to the physical differentiation and joint training of the expert pool E. By decoupling router learning from expert optimization, we establish a stable training environment in which each expert is forced to specialize in the semantic subspaces identified during Phase I.

a) Router-Guided Physical Specialization: To transform the dense FFN layers into a collection of specialized experts, we decompose the original model into structurally isolated expert modules $\{ E _ { 1 } , \ldots , E _ { N } \}$ . During forward propagation, expert execution strictly follows the routing decisions produced by the frozen router. Only experts selected by the active index set $\mathcal { E } _ { \mathrm { a c t i v e } }$ participate in computation and parameter updates:

$$
\mathbf { y } = \sum _ { k \in \mathcal { E } _ { \mathrm { a c t i v e } } } \mathbf { g } _ { k } \cdot E _ { k } ( \mathbf { x } ; \boldsymbol { \theta } _ { k } )\tag{14}
$$

where x and $\mathbf { y }$ denote the input and output hidden states, respectively, and $\mathbf { g } _ { k }$ represents the fixed routing weight assigned to expert $E _ { k }$ . Under this selective activation scheme, gradients are restricted to the parameters $\theta _ { k }$ of active experts, enforcing strong functional separation and driving each expert to adapt exclusively to its designated context.

Recent studies on MoE refactoring [6] show that routing patterns exhibit strong correlation across consecutive Transformer layers, rendering independent layer-wise routing largely redundant. Following this observation, we adopt a global routing consistency strategy, where the single routing vector g generated by $\psi$ is broadcast to all FFN layers. This constructs vertically aligned expert pathways that preserve semantic coherence throughout the network depth and enable consistent processing of specialized representations.

b) Distillation-Enhanced Expert Specialization (Optional): Directly transitioning from dense to sparse execution may introduce temporary performance degradation. To accelerate expert adaptation and preserve the teacher model’s representational capacity, we incorporate the original dense model as a persistent reference and apply dual-granularity distillation [34].

Specifically, we introduce two auxiliary objectives:

1) Hidden State Alignment. We minimize the mean squared error between student (S) and teacher (T) hidden states across all Transformer layers. An adaptive normalization function N( $\mathcal { N } ( \cdot )$ is applied to account for architectural differences (e.g., Pre-Norm vs. Post-Norm) and improve gradient stability:

$$
\mathcal { L } _ { \mathrm { h i d } } = \sum _ { l = 1 } ^ { L } \mathbf { M S E } ( \mathcal { N } ( \mathbf { H } _ { S } ^ { ( l ) } ) , \mathcal { N } ( \mathbf { H } _ { T } ^ { ( l ) } ) )\tag{15}
$$

where L denotes the total number of layers, and $\mathbf { H } _ { S } ^ { ( l ) }$ and $\mathbf { H } _ { T } ^ { ( l ) }$ represent the corresponding hidden states.

2) Logit Alignment. We apply symmetric KL divergence to align output probability distributions, encouraging the sparse experts to preserve the predictive behavior of the dense teacher.

The overall optimization objective is defined as:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { t a s k } } + \alpha \left( \mathcal { L } _ { \mathrm { h i d } } + \mathcal { L } _ { \mathrm { l o g i t } } \right)\tag{16}
$$

where $\mathcal { L } _ { \mathrm { t a s k } }$ denotes the primary task loss (e.g., the task in the predictive query), and α controls the strength of distillation regularization. This auxiliary supervision stabilizes sparse

training and enables QCATS to retain dense-model accuracy under constrained execution budgets.

## IV. SYSTEM-LEVEL OPTIMIZATION

In large-scale query scenarios, batching data from multiple queries is standard practice to maintain high GPU utilization. However, the diverse query contexts within mixed batches trigger divergent expert activation paths, which may fragment GPU parallelism and thereby degrade hardware efficiency. Simultaneously, database text data typically exhibits a longtailed length distribution, which makes static batching inefficient due to excessive zero padding. Furthermore, as QCATS significantly boosts GPU inference efficiency, traditional synchronous execution pipelines could become increasingly CPUbound, causing preprocessing latency to emerge as a major bottleneck for overall system performance.

To address the aforementioned inefficiencies, we propose a system-level optimization framework that adopts an asynchronous producer-consumer pipeline architecture. As illustrated in Fig. 8, we decouple the inference lifecycle into two separate modules: The CPU acts as the producer, responsible for preparing routing-aware batched inputs, while the GPU serves as the consumer, dedicated to performing compute-intensive sparse inference computations. A threadsafe prefetch buffer connects the two stages, allowing data preparation and model execution to proceed concurrently and overlap in time.

The core of the producer module, as shown in the left part in Fig. 8, is hierarchical bucketing. It reorganizes query data through a two-level process to jointly optimize execution-path consistency and hardware efficiency. The first level focuses on aligning model execution behavior by grouping inputs according to routing decisions, while the second level targets memory and computation efficiency by minimizing padding overhead through length-aware batching.

At the first level, QCATS leverages the context-aware router to precompute expert assignments before batch construction. Text data are grouped based on their routing outcomes, such that all texts within the same group activate an identical group of experts during inference. This routing-aware grouping ensures execution-path homogeneity within each batch and eliminates control-flow divergence on GPUs.

At the second level, text data within each routing group is further reorganized based on its length. Text entries with similar lengths are placed into the same bucket to reduce padding overhead. To maintain stable GPU utilization across varying input sizes, the system dynamically adjusts batch sizes according to the representative sequence length of each bucket and the available GPU resource budget. Subsequently, the bucketed text entries are tokenized and collated into batches, which are then enqueued into a prefetch buffer for immediate processing by the consumer module in the GPU.

In summary, the two-level reorganization enables QCATS to construct routing-consistent and length-homogeneous batches that maximize sparse execution efficiency while minimizing wasted computation caused by padding. Combined with the asynchronous producer–consumer pipeline, hierarchical bucketing effectively overlaps CPU-side preprocessing with GPU inference, eliminates pipeline stalls, and ensures that system throughput is primarily bounded by model execution performance rather than data preparation overhead.

![](images/0b45d3b2aeb67e5092d71a3b2b083d95153aeb5290dfabec152279427f159274.jpg)  
Fig. 8: QCATS Processing with Pipeline.

## V. EXPERIMENTS

In this section, we conduct extensive experiments to evaluate QCATS against the design objectives mentioned in §II-B. The evaluation mainly focuses on the following Research Questions (RQs):

• RQ1 (System Efficiency): For G1, how much improvement does QCATS achieve in terms of end-to-end latency and peak GPU memory footprint?

• RQ2 (System Effectiveness): For G2, can QCATS reproduce the predictive accuracy of the original fine-tuned dense baselines regarding the reduction in activated parameters?

• RQ3 (Ablation & Breakdown): What is the contribution of each component to the goals? Specifically, how does the latency breakdown demonstrate the effectiveness of our proposed optimization strategies? Meanwhile, what is the contribution of each system component to accuracy preservation?

• RQ4 (Sensitivity & Robustness): How sensitive is the model performance to key hyperparameters, and how is the neuron activation pattern aligned with the query context? In addition, does QCATS maintain robustness when transferring to similar domains for data analysis?

## A. Experimental Setup

1) Datasets and Tasks: To evaluate QCATS across diverse domains, we use four real-world datasets summarized in Table I.

• Amazon Review (E-commerce) [35] contains user reviews and item metadata with 17 structured columns. The task is to predict product ratings from user reviews.

• Kickstarter (Crowdfunding Finance) [36] tracks crowdfunding campaigns and includes 7 structured metadata columns. The task is to predict campaign funding status from the project name and blurb.

• OSHA (Industrial Safety) [37] records industrial accidents with 12 structured metadata columns. The task is to classify injury severity from accident abstracts.

TABLE I: Summary of dataset statistics and task definitions.
<table><tr><td>Dataset</td><td>Total Size</td><td>Context Columns</td></tr><tr><td>Amazon</td><td>1,908,006</td><td>[main_category, sub_category, price]</td></tr><tr><td>Kickstarter</td><td>202,835</td><td>[main_category, sub_category, goal]</td></tr><tr><td>OSHA</td><td>42,645</td><td>[occupation, accident_type, age]</td></tr><tr><td>CVE</td><td>90,847</td><td>[cwe, privileges_required, product_count]</td></tr></table>

• CVE (Cybersecurity) [38] describes software vulnerabilities and contains 11 structured metadata columns. The task is to predict CVSS severity from vulnerability descriptions.

2) Query Workload Generation: Public datasets rarely include query logs combining text inputs, structured metadata, and predictive UDFs, so we randomly generate 50 SQL templates with one to three predicates each for the datasets from the aforementioned widely-used benchmarks. Context-column predicates use categorical equality or numerical ranges, combined only with AND; context columns may also appear in GROUP BY clauses. Non-context predicates are randomly sampled from LIKE, IN, equality, and range conditions, optionally connected by OR. Templates may include valid joins sampled from predefined joinable table pairs. These templates represent common analytical prediction queries involving category/range filtering, metadata joins, and grouped prediction aggregation.

3) Baselines and Implementation: To validate generality and effectiveness, we evaluate QCATS on BERT and Qwen architectures, and compare three groups of methods:

1) Dense Baselines: Original unpruned BERT-base [39] and Qwen-base [40] models.

2) MoE Baselines: MoEfication [23], which partitions FFN neurons into logical experts without changing weights, and MoEBERT [34], which converts a fine-tuned model into a physical MoE through adaptation and distillation.

3) QCATS (Ours): QCATS applied to BERT and Qwen, denoted as QCATS (BERT) and QCATS (Qwen).

All sparse methods activate 25% of FFN parameters during inference, while dense baselines use all parameters. Since no standard MoEfication implementation exists for Qwen, we compare Qwen-base directly with QCATS (Qwen). QCATS is implemented in Apache Spark [41] 4.0.1 with PyTorch 2.5 and CUDA 11.8, using BERT-base (110M parameters) and Qwen-0.6B on an Intel Xeon Gold 5218R CPU and one NVIDIA A5000 GPU (24GB).

Note that we use compact models to reflect the resource constraints of predictive database systems and to isolate the benefits of query-aware sparsification. Unless otherwise stated, QCATS uses the N4K1 configuration (N = 4, K = 1), with the hidden dimension split between shared and unique neurons. The router uses the top M = 3 attributes, a 512-hidden Transformer with 4 heads, and a 3-layer 256-dimensional MLP. During training, Phase I optimizes the router with lr = 5e − 4, η = 0.02, and τ = 1.0, while Phase II fine-tunes the experts with α = 5.0, and learning rates of 3e−5 for BERT and 1e−5 for Qwen, respectively. Additional hyperparameters are reported in the appendix.

![](images/95dafced795e1226ca60c591f2c1ca8049d8e427aae207e3577d462066fda172.jpg)  
(a) CVE (BERT)

![](images/4fb934c6f8b4315ab6f57a6dc4e6c067aa9f36d3ff08eeb534ff78aa994a44ca.jpg)  
(b) Kickstarter (BERT)

![](images/daf89be200cbdfadbbe0b082ad3e12b526e1971dce881e02d6c0a7ac74e04a1a.jpg)  
(c) CVE (Qwen)

![](images/0d85a3e58817b16841193c105c2409e118ddda475097c1a9d90b7537d1c8ccc1.jpg)  
(d) Kickstarter (Qwen)  
Fig. 9: Peak Memory usage comparison.

Offline Construction Cost. With 100k training examples, activation profiling, clustering, and expert-slice construction take a few minutes. The expert fine-tuning dominates offline cost, taking approximately 30 minutes for BERT and 2 hours for Qwen, or 1.5 and 6 hours with distillation, respectively.

4) Evaluation Settings and Metrics: We evaluate efficiency by executing approximately 600 SQL queries per dataset, derived from the 50 query templates, retrieving 1 million rows in total. We report end-to-end latency and peak GPU memory usage, with the average batch size aligned across methods. We evaluate effectiveness using accuracy and weighted F1-score on the aggregated test set formed by merging query results from all workloads.

## B. System Efficiency (RQ1)

To answer RQ1, we assess the system-level efficiency in end-to-end response time and peak memory.

1) End-to-End Latency: Under the efficiency evaluation setting (§V-A4) on the 1-million-row workload, Table II reports the end-to-end latency of all methods. All baselines and QCATS w/o pipeline use serial execution, whereas QCATS adopts the proposed pipelined architecture. The results show that QCATS achieves speedup by using query-level routing to effectively reduce execution overhead; on top of this, the pipeline further amplifies the gain by overlapping CPU preprocessing with GPU inference. In contrast, MoEfication only introduces logical sparsity, which does not necessarily translate into efficient physical execution on GPUs.

2) Peak GPU Memory (VRAM) Consumption: As shown in Fig. 9, we evaluate peak GPU memory usage under the efficiency evaluation setting (§V-A4). Given that the static weights of BERT and Qwen occupy only about 220 MB and 1.2 GB, respectively, the substantial reduction in peak memory usage achieved by QCATS indicates that its memory savings mainly come from reduced activation memory. QCATS allocates memory only to the activated expert subsets and reduces padding through bucketing, thereby effectively compressing the runtime memory footprint.

![](images/e8dc683fa188bc35756aac5405924afef98bfc71330606bce6a27fa99fa796ae.jpg)  
(a) Amazon (BERT)

![](images/ce60242077b08da44ad0cded0f71e75c2f1cf85669dd56af83d611c7e3aabb48.jpg)

![](images/49d231262fc9ff050ce0602414d767b980c564a340a20b8be6c592279c4244f4.jpg)  
(c) Amazon (Qwen)

(b) OSHA (BERT)  
![](images/3522cae7d1918d96888d46934981f954ddcaf76fbc21e0f0f760de374447f059.jpg)  
(d) OSHA (Qwen)  
Fig. 10: Latency Breakdown.

## C. System Effectiveness (RQ2)

To answer RQ2, we evaluate the prediction robustness of QCATS under the effectiveness evaluation setting (§V-A4) using accuracy and weighted F1-score. Table III presents the comparative performance of all models across four datasets. It demonstrates that QCATS preserves task accuracy while activating only a subset of FFN parameters.

QCATS consistently rivals or outperforms dense baselines and matches the performance of MoEBERT, despite using a much simpler routing strategy. Our results show that for database-oriented tasks, understanding the global context of the entire query is sufficient to select the right experts. This implies that the heavy computational cost of routing every token is redundant, as QCATS delivers the same level of accuracy with a much more efficient, query-level approach.

## D. Ablation & Breakdown (RQ3)

This section answers RQ3 by evaluating the specific impact of key components. We perform a latency breakdown to evaluate the impact of the sparsity mechanism of QCATS and the hierarchical bucketing. Then, we conduct an ablation study to isolate the roles of the routing mechanism and knowledge distillation in maintaining model accuracy.

1) Latency Breakdown and Bottleneck Analysis: To analyze the sources of QCATS’s performance gains in database scenarios, we executed a query that retrieves 100,000 rows and decomposed the total response time into four main components: database overhead, router computation (specific to QCATS), data preprocessing, and model inference. As shown in Fig. 10, the QCATS router (green) introduces only negligible overhead. During model inference, QCATS not only reduces latency by lowering FLOPs through sparsification, but also avoids the runtime overhead of fine-grained dynamic routing in MoE baselines through query-level global routing. Meanwhile, by reducing padding through hierarchical bucketing, QCATS w/o pipeline outperforms QCATS w/o all opt in both preprocessing and inference. Finally, the latency breakdown further confirms the necessity of pipelining for lightweight models such as BERT: once GPU inference latency is significantly reduced, CPU preprocessing remains a non-negligible bottleneck.

TABLE II: End-to-end execution time (s) comparison.
<table><tr><td rowspan="2">Dataset</td><td colspan="5">BERT-base Backbone</td><td colspan="3">Qwen-base Backbone</td></tr><tr><td>BERT-base</td><td>MoEfication</td><td>MoEBERT</td><td>QCATS w/o pipeline</td><td>QCATS</td><td>Qwen-base</td><td>QCATS w/o pipeline</td><td>QCATS</td></tr><tr><td>Amazon</td><td>279.05</td><td>299.32 (0.93×)</td><td>236.16 (1.18×)</td><td>164.02 (1.70×)</td><td>98.69 (2.83×)</td><td>1499.22</td><td>517.49 (2.90×)</td><td>463.30 (3.24×)</td></tr><tr><td>Kickstarter</td><td>352.94</td><td>376.14 (0.94×)</td><td>295.33 (1.20×)</td><td>172.89 (2.04×)</td><td>129.13 (2.73×)</td><td>1739.19</td><td>874.38 (1.99×)</td><td>780.19 (2.23×)</td></tr><tr><td>OSHA</td><td>785.40</td><td>847.95 (0.93×)</td><td>636.40 (1.23×)</td><td>307.46 (2.55×)</td><td>230.08 (3.41×)</td><td>5241.14</td><td> $2 2 3 1 . 1 2 \ ( 2 . 3 5 \times )$ </td><td>1981.82 (2.64×)</td></tr><tr><td>CVE</td><td>798.29</td><td>862.21 (0.93×)</td><td>653.82 (1.22×)</td><td>231.51 (3.45×)</td><td>180.51 (4.42×)</td><td>4521.26</td><td> $1 5 6 4 . 1 7 \ ( 2 . 8 9 \times )$ </td><td>1418.08 (3.19×)</td></tr></table>

TABLE III: Comparison of Accuracy and Weighted F1-Score. Boldface indicates the best performance.
<table><tr><td>Model</td><td>Amazon (Acc / F1)</td><td>Kickstarter (Acc / F1)</td><td>OSHA (Acc / F1)</td><td>CVE (Acc / F1)</td></tr><tr><td>BERT-base</td><td>89.7  / 88.6</td><td>81.4 / 81.0</td><td>93.0 / 92.9</td><td>77.1 / 77.0</td></tr><tr><td>MoEfication</td><td>89.5 / 88.3</td><td>80.8 / 80.4</td><td>92.7  / 92.4</td><td>76.7  / 76.4</td></tr><tr><td>MoEBERT</td><td>89.9 / 88.7</td><td>81.6 / 81.3</td><td>93.0 / 92.9</td><td>78.1 / 78.0</td></tr><tr><td>QCATS (BERT)</td><td>89.6 / 88.5</td><td>81.8 / 81.4</td><td>93.0 / 92.9</td><td>77.9 / 77.8</td></tr><tr><td>Qwen-base</td><td>89.2 / 87.5</td><td>80.1 / 80.0</td><td>92.2 / 91.9</td><td>74.1 / 73.7</td></tr><tr><td>QCATS (Qwen)</td><td>89.5 / 87.5</td><td>81.2 / 81.0</td><td>92.8 / 92.6</td><td>76.6 / 76.2</td></tr></table>

![](images/02733a18d94be39c82da2a83836534e00f5704e91841fb57cfd124a8ca77a311.jpg)  
(a) BERT Arch

![](images/bb668122b099888c571c5fd5eb7ef295e422dffa114f62a4b9578b35c051eb45.jpg)  
(b) Qwen Arch  
Fig. 11: Ablation study.

2) Context-Aware Router and Knowledge Distillation: To quantify the independent contributions of knowledge distillation (KD) and dynamic routing, we compare four ablation variants under the effectiveness evaluation setting (Fig. 11). The results show that QCATS variants with routing (red/orange) consistently achieve performance comparable to the dense baseline, while the w/o distill variant even surpasses the teacher model on the Qwen-CVE task, indicating that KD is not the decisive factor behind the performance gain. In contrast, removing routing leads to a clear performance drop, showing that the routing mechanism is critical for expert specialization.

## E. Sensitivity & Robustness (RQ4)

To answer RQ4, we study sensitivity to expert granularity and context columns, evaluate diverse SQL workloads and test transferability to a similar domain.

1) Expert granularity: Following the effectiveness evaluation setting, we investigated expert granularity (N vs. K)

![](images/fbaf47d261cc3e86cc8f205bf60533b00a8e70b4f13b53a86dbd606b593652b1.jpg)  
(a) Impact of expert granularity (N and K).

![](images/a00fd4281d2aabdb21fbb5bf7f97a0556cf9b7c9c4b614a009b1aa0bf7ccdb6e.jpg)  
(b) Impact of context column.  
Fig. 12: Hyperparameter experiments.

TABLE IV: Transferability to Similar Domains. Boldface indicates the best performance.
<table><tr><td>Model</td><td>Source Domain (Amazon) (Acc / F1)</td><td>Target Domain (Flipkart) (Acc / F1)</td></tr><tr><td>BERT-base</td><td>89.7  / 88.6</td><td>88.7  / 86.5</td></tr><tr><td>QCATS (BERT)</td><td>89.6 / 88.5</td><td>88.7 / 86.6</td></tr><tr><td>Qwen-base</td><td>89.2 / 87.5</td><td>88.6 / 85.6</td></tr><tr><td>QCATS (Qwen)</td><td>89.5 / 87.5</td><td>88.8 / 85.6</td></tr></table>

on the Kickstarter dataset while maintaining a constant 25% activation ratio. Fig. 12a reveals high performance stability, indicating that accuracy depends primarily on parameter retention rather than slicing size. However, the latency for a query retrieving 100,000 rows, highlights the cost of finer granularity. Therefore, we adopt N4K1 as the default to maximize computational density and throughput without compromising predictive performance.

2) Context Columns: Under the effectiveness evaluation setting, we analyze the sensitivity of the retained context column count M on the Kickstarter dataset. Fig. 12b illustrates this trade-off between weighted F1-score and average routing latency across varying M. As shown, a small M fails to capture the global data distribution, weakening context alignment and routing accuracy. Conversely, a large M introduces redundant attributes; this increases preprocessing overhead.

3) QCATS under diverse SQL workloads: To evaluate the adaptability of QCATS across diverse SQL workloads, we analyze four representative queries on the Amazon dataset using the Qwen backbone (Fig. 13). As shown in Fig. 14, QCATS consistently reduces end-to-end latency across all scenarios while exerting only minimal impact on prediction accuracy. Notably, even for Q4, where the UDF appears in the filter predicate, QCATS still achieves substantial latency reduction by significantly decreasing the amount of data fed into the model through query rewriting.

<table><tr><td>SELECT Sentiment(R.reviewText) FROM Reviews R JOIN Metadata M ON R.asin = M.asin WHERE M.main_category = &#x27;Grocery &amp; Gourmet Food&#x27; AND R.reviewName IS NOT NULL</td><td>SELECT M.sub category, AVG(Sentiment(R.reviewText)) FROM Reviews R JOIN Metadata M ON R.asin = M.asin WHERE M.price &gt; 25.72 AND M.price &lt;= 45 AND (R.unixReviewTime&lt;= 1521158400 OR M.brand LIKE &#x27;Ave%&#x27;); GROUP BY M.main_category</td><td>SELECT Sentiment(R.reviewText) FROM Reviews R WHERE R.verified != &#x27;True&#x27; AND R.vote=0</td><td>SELECT M.brand FROM Reviews R JOIN Metadata M ON R.asin = M.asin WHERE M.sub category=&#x27;Outdoor Recreation&#x27; AND R.verified=&#x27;True&#x27; AND Sentiment(R.reviewText) = 2</td></tr></table>

(a) Q1: Normal  
(b) Q2: Aggregation  
(c) Q3: No Context  
(d) Q4: UDF Predicate  
Fig. 13: Predictive queries in the experiment.

![](images/f5f40c5d6126ebb0a76b5321db6fee1304201c88b378494ec801a0e1a3303df2.jpg)

![](images/106f04483c795fd20f7e9b45d99c827b53ebf861cccc2f837a258a9213d3fbbc.jpg)  
(a) Weighted F1-score.  
(b) End-to-end latency.  
Fig. 14: Performance evaluation of QCATS across diverse SQL workloads.

4) Case Study: Transferability to Similar Domains: To assess generalization, we directly transfer models trained on Amazon to the Flipkart dataset [42]–[44] in a zero-shot manner, using Product Category and Product Price as routing contexts (following effectiveness evaluation setting). As shown in Table IV, QCATS achieves performance comparable to dense baselines in this similar-domain setting, indicating that its query-based slicing mechanism preserves both transferability and core model capabilities.

## VI. RELATED WORK

In-database ML and predictive query processing have long focused on integrating model inference into SQL execution pipelines. Early approaches [16], [17] and modern Cloud APIs [3], [4] predominantly rely on UDFs to invoke models, but they treat inference as a black-box task, discarding rich query-time context. To enable deeper optimization, some frameworks [7], [45] decompose models into basic relational or tensor operators to perform fine-grained execution. However, these white-box optimizations are largely confined to traditional ML or neural networks and do not scale to the massive parameter space of Transformers. In contrast, our work bridges this gap: we move beyond external data movement or scheduling by coupling database analytics with model internals. By leveraging query context, we enable the deterministic customization of the model inference path itself for Transformer-based models.

Database-aware model serving has been explored to improve inference flexibility, but existing approaches differ fundamentally from QCATS in optimization granularity and system objectives. ARM-Net [46] adapts feature interactions at the instance level to improve prediction accuracy, without targeting inference efficiency or execution-path optimization. Model Slicing [47] supports elastic inference cost via width-wise slicing driven by resource budgets, but does not exploit query semantics or workload locality. LEADS [48] also exploits database context, but selects among replicated experts or multiple conventional models primarily to improve prediction quality for structured-data analytics. Like the preceding approaches, it does not specifically address Transformer-based text inference. QCATS instead uses query context to deterministically select FFN neuron slices within a single Transformer backbone. This targets text prediction while reducing inference cost and GPU memory usage and preserving densemodel accuracy. Furthermore, all the aforementioned methods do not specifically address the inference characteristics of Transformer-based text prediction, while QCATS uses query context to deterministically select model parameters, thereby improving inference efficiency while preserving accuracy.

Efficient Transformer inference and MoE architecture [12], [13] scales model capacity while maintaining manageable inference costs. Recent efforts in MoE-fication [6], [23], [34], [49] have extended this concept by refactoring pre-trained dense models into expert-based structures. However, standard MoEs rely on token-level, layer-wise routing, which treats input data as independent instances and ignores high-level semantic contexts. While recent works attempt to optimize routing via decoupling [6], [50] or speculation [51], [52], they remain constrained by micro-level token dependencies, limiting optimization in data-intensive database environments. In contrast, QCATS is tailored for in-database analytics, where data is accessed via structured queries rather than random batches. We exploit the query context to implement query-level global routing. By leveraging these database-specific hints to deterministically pre-select expert slices, we eliminate runtime dispatching overhead and enable efficient, homogeneous batching that is unattainable in context-agnostic systems.

## VII. CONCLUSION

We presented QCATS, a query context-aware transformer slicing framework that enables efficient sparse inference for indatabase predictive queries. By shifting conditional execution from token level to query level, QCATS aligns sparse neural execution with database semantics. Extensive experimental results demonstrate that QCATS significantly improves inference throughput and reduces memory consumption while preserving predictive accuracy compared to dense baselines. This suggests that query-aware execution provides a practical foundation for bridging Transformer-based model inference with high-performance database systems.

We believe this work opens new opportunities for designing database-native inference engines that treat neural execution as a first-class query operator rather than an external blackbox service. Building on this direction, we will evaluate QCATS on larger Transformer backbones and more complex, realistic workloads under distribution shifts and heterogeneous domains. We will also investigate adaptation and cross-domain transferability of expert libraries and routing policies, while interactions with generative-model serving systems remain separate future work.

## VIII. ACKNOWLEDGMENT

This work was supported by the Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (JYB2025XDXM103), the National Natural Science Foundation of China (Grant No. 62602571), and the Leading Talent of Technological Innovation Program (No. 2023R5214) of Zhejiang Province.

We disclose that GPT was used only to assist with English polishing, grammar improvement, and the writing of some utility code. The research questions, overall design, implementation process, experiments, analysis of results, and conclusions were developed entirely by the authors.

## REFERENCES

[1] Q. Lin, S. Wu, J. Zhao, J. Dai, F. Li, and G. Chen, “A comparative study of in-database inference approaches,” in 2022 IEEE 38th International Conference on Data Engineering (ICDE). IEEE, 2022, pp. 1794–1807.

[2] Y. Peng, Z. Xie, K. Chen, G. Chen, and L. Shou, “Towards automatic and efficient prediction query processing in analytical database,” in 2025 IEEE 41st International Conference on Data Engineering (ICDE). IEEE, 2025, pp. 2253–2266.

[3] Databricks, “Model serving with Databricks,” https://docs.databricks. com/en/machine-learning/model-serving/index.html, 2024, accessed: 2026-01-20.

[4] Google Cloud, “Model inference overview,” https://cloud.google.com/ bigquery/docs/inference-overview, 2024, accessed: 2026-01-20.

[5] Alibaba Cloud, “PolarDB for AI,” https://www.alibabacloud.com/ help/en/polardb/polardb-for-mysql/user-guide/polardb-for-ai/, 2023, accessed: 2026-01-20.

[6] R. Cai, Y. Ro, G.-W. Kim, P. Wang, B. Ehteshami Bejnordi, A. Akella, Z. Wang et al., “Read-me: Refactorizing llms as router-decoupled mixture of experts with system co-design,” Advances in Neural Information Processing Systems, vol. 37, pp. 116 126–116 148, 2024.

[7] K. Park, K. Saur, D. Banda, R. Sen, M. Interlandi, and K. Karanasos, “End-to-end optimization of machine learning prediction queries,” in Proceedings of the 2022 International Conference on Management of Data, 2022, pp. 587–601.

[8] C. Yan, Y. Lin, and Y. He, “Predicate pushdown for data science pipelines,” Proceedings of the ACM on Management of Data, vol. 1, no. 2, pp. 1–28, 2023.

[9] Y. Guo, G. Li, R. Hu, and Y. Wang, “In-database query optimization on sql with ml predicates,” The VLDB Journal, vol. 34, no. 1, p. 12, 2025.

[10] D. J. Abadi, S. R. Madden, and N. Hachem, “Column-stores vs. rowstores: how different are they really?” in Proceedings of the 2008 ACM SIGMOD international conference on Management of data, 2008, pp. 967–980.

[11] M. A. Jose and F. G. Cozman, “A multilingual translator to sql with database schema pruning to improve self-attention,” International Journal ofInformation Technology, vol. 15, no. 6, pp. 3015–3023, 2023.

[12] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen, “Gshard: Scaling giant models with conditional computation and automatic sharding,” arXiv preprint arXiv:2006.16668, 2020.

[13] W. Fedus, B. Zoph, and N. Shazeer, “Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity,” Journal of Machine Learning Research, vol. 23, no. 120, pp. 1–39, 2022.

[14] J. He, J. Qiu, A. Zeng, Z. Yang, J. Zhai, and J. Tang, “Fastmoe: A fast mixture-of-expert training system,” arXiv preprint arXiv:2103.13262, 2021.

[15] T. Gale, D. Narayanan, C. Young, and M. Zaharia, “Megablocks: Efficient sparse training with mixture-of-experts,” Proceedings of Machine Learning and Systems, vol. 5, pp. 288–304, 2023.

[16] J. M. Hellerstein, C. Re, F. Schoppmann, D. Z. Wang, E. Fratkin,´ A. Gorajek, K. S. Ng, C. Welton, X. Feng, K. Li et al., “The madlib analytics library: or mad skills, the sql,” Proceedings of the VLDB Endowment, vol. 5, no. 12, pp. 1700–1711, 2012.

[17] M. Boehm, I. Antonov, S. Baunsgaard, M. Dokter, R. E. Ginthoer, K. Innerebner, F. Klezin, S. Lindstaedt, A. Phani, B. Rath et al., “Systemds: A declarative machine learning system for the end-to-end data science lifecycle,” in 10th Conference on Innovative Data Systems Research, 2020.

[18] M. Jasny, T. Ziegler, T. Kraska, U. Roehm, and C. Binnig, “Db4ml-an inmemory database kernel with machine learning support,” in Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data, 2020, pp. 159–173.

[19] W. Zheng, M. Xu, S. Song, and K. Ye, “Bucketserve: Bucket-based dynamic batching for smart and efficient llm inference serving,” in 2025 IEEE International Conferences on Internet of Things (iThings) IEEE Green Computing & Communications (GreenCom) IEEE Cyber, Physical & Social Computing (CPSCom) and IEEE Smart Data (SmartData) and IEEE Congress on Cybermatics (Cybermatics). IEEE, 2025, pp. 103–111.

[20] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[21] Z. Li, C. You, S. Bhojanapalli, D. Li, A. S. Rawat, S. J. Reddi, K. Ye, F. Chern, F. Yu, R. Guo et al., “The lazy neuron phenomenon: On emergence of activation sparsity in transformers,” in The Eleventh International Conference on Learning Representations.

[22] Z. Liu, J. Wang, T. Dao, T. Zhou, B. Yuan, Z. Song, A. Shrivastava, C. Zhang, Y. Tian, C. Re et al., “Deja vu: Contextual sparsity for efficient llms at inference time,” in International Conference on Machine Learning. PMLR, 2023, pp. 22 137–22 176.

[23] Z. Zhang, Y. Lin, Z. Liu, P. Li, M. Sun, and J. Zhou, “Moefication: Transformer feed-forward layers are mixtures of experts,” in Findings of the Association for Computational Linguistics: ACL 2022, 2022, pp. 877–890.

[24] T. Mikolov, E. Grave, P. Bojanowski, C. Puhrsch, and A. Joulin, “Advances in pre-training distributed word representations,” in Proceedings of the eleventh international conference on language resources and evaluation (LREC 2018), 2018.

[25] L. Breiman, “Random forests,” Machine learning, vol. 45, no. 1, pp. 5–32, 2001.

[26] T. Cover and P. Hart, “Nearest neighbor pattern classification,” IEEE transactions on information theory, vol. 13, no. 1, pp. 21–27, 1967.

[27] P. Li, W. Wei, R. Zhu, B. Ding, J. Zhou, and H. Lu, “Alece: An attention-based learned cardinality estimator for spj queries on dynamic workloads,” Proceedings of the VLDB Endowment, vol. 17, no. 2, pp. 197–210, 2023.

[28] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, and I. Stoica, “Efficient memory management for large language model serving with pagedattention,” in Proceedings of the 29th symposium on operating systems principles, 2023, pp. 611–626.

[29] L. Zheng, L. Yin, Z. Xie, C. Sun, J. Huang, C. H. Yu, S. Cao, C. Kozyrakis, I. Stoica, J. E. Gonzalez et al., “Sglang: Efficient execution of structured language model programs,” Advances in neural information processing systems, vol. 37, pp. 62 557–62 583, 2024.

[30] T. Dao, D. Fu, S. Ermon, A. Rudra, and C. Re, “Flashattention: Fast and´ memory-efficient exact attention with io-awareness,” Advances in neural information processing systems, vol. 35, pp. 16 344–16 359, 2022.

[31] E. Frantar, S. Ashkboos, T. Hoefler, and D.-A. Alistarh, “Optq: Accurate post-training quantization for generative pre-trained transformers,” in 11th International Conference on Learning Representations, 2023.

[32] J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang, C. Gan, and S. Han, “Awq: Activation-aware weight quantization for on-device llm compression and acceleration,” Proceedings of machine learning and systems, vol. 6, pp. 87–100, 2024.

[33] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han, “Smoothquant: Accurate and efficient post-training quantization for large

language models,” in International conference on machine learning. PMLR, 2023, pp. 38 087–38 099.

[34] S. Zuo, Q. Zhang, C. Liang, P. He, T. Zhao, and W. Chen, “Moebert: from bert to mixture-of-experts via importance-guided adaptation,” in Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2022, pp. 1610–1623.

[35] Y. Hou, J. Li, Z. He, A. Yan, X. Chen, and J. McAuley, “Bridging language and items for retrieval and recommendation,” arXiv preprint arXiv:2403.03952, 2024.

[36] Y. Kantharia, “Kickstarter campaigns dataset 2.0,” https://www.kaggle. com/datasets/yashkantharia/kickstarter-campaigns-dataset-20, 2020, data originally scraped from webrobots.io. Accessed: 2026-01-26.

[37] Z. Ou, D. Li, Z. Tan, W. Li, H. Liu, and S. Song, “Building safer sites: A large-scale multi-level dataset for construction safety benchmark,” in Proceedings of the 34th ACM International Conference on Information and Knowledge Management, 2025, pp. 6508–6512.

[38] F. Manzoni, “Vulnerability management datasets,” https://www.kaggle. com/datasets/francescomanzoni/vulnerability-management-datasets, 2025, accessed: 2026-01-26.

[39] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 conference of the North American chapter of the associationfor computational linguistics: human language technologies, volume 1 (long and short papers), 2019, pp. 4171–4186.

[40] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.

[41] The Apache Software Foundation, “Apache Spark,” https://spark.apache. org, 2024, accessed: 2026-01-20.

[42] T. MSS, “Water bottle review dataset (flipkart),” https://www.kaggle. com/datasets/tharunmss/water-bottle-dataset-flipkart, 2025, accessed: 2026-01-26.

[43] Guhan, “Speaker scrapped data with sentiments (flipkart),” https://www.kaggle.com/datasets/guhan0912/ speaker-scrapped-data-with-sentiments-flipkart, 2025, accessed: 2026-01-26.

[44] sublimE, “Laptop reviews dataset (flipkart),” https://www.kaggle.com/ datasets/gitadityamaddali/flipkart-laptop-reviews, 2024, accessed: 2026- 01-26.

[45] Q. Lin, S. Wu, J. Zhao, J. Dai, M. Shi, G. Chen, and F. Li, “Smartlite: A dbms-based serving system for dnn inference in resource-constrained environments,” Proceedings of the VLDB Endowment, vol. 17, no. 3, pp. 278–291, 2023.

[46] S. Cai, K. Zheng, G. Chen, H. Jagadish, B. C. Ooi, and M. Zhang, “Arm-net: Adaptive relation modeling network for structured data,” in Proceedings of the 2021 International Conference on Management of Data, 2021, pp. 207–220.

[47] S. Cai, G. Chen, B. C. Ooi, and J. Gao, “Model slicing for supporting complex analytics with elastic inference cost and resource constraints,” Proceedings of the VLDB Endowment, vol. 13, no. 2, pp. 86–99, 2019.

[48] L. Zeng, N. Xing, S. Cai, G. Chen, B. C. Ooi, J. Pei, and Y. Wu, “Powering in-database dynamic model slicing for structured data analytics,” Proceedings of the VLDB Endowment, vol. 17, no. 13, pp. 4813–4826, 2024.

[49] Z. Zhang, C. Xiao, Q. Qin, Y. Lin, Z. Zeng, X. Han, Z. Liu, R. Xie, M. Sun, and J. Zhou, “Exploring the benefit of activation sparsity in pre-training,” in Proceedings of the 41st International Conference on Machine Learning, 2024, pp. 60 040–60 056.

[50] R. Hwang, J. Wei, S. Cao, C. Hwang, X. Tang, T. Cao, and M. Yang, “Pre-gated moe: An algorithm-system co-design for fast and scalable mixture-of-expert inference,” in 2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA). IEEE, 2024, pp. 1018–1031.

[51] Z. Du, S. Li, Y. Wu, X. Jiang, J. Sun, Q. Zheng, Y. Wu, A. Li, H. H. Li, and Y. Chen, “Sida: Sparsity-inspired data-aware serving for efficient and scalable large mixture-of-experts models,” Proceedings of Machine Learning and Systems, vol. 6, pp. 224–238, 2024.

[52] X. He, S. Zhang, Y. Wang, H. Yin, Z. Zeng, S. Shi, Z. Tang, X. Chu, I. Tsang, and O. Y. Soon, “Expertflow: Optimized expert activation and token allocation for efficient mixture-of-experts inference,” arXiv preprint arXiv:2410.17954, 2024.

## APPENDIX

We attach the analysis and implementation details of QCATS in this appendix.

## QUERY CONTEXT ANALYSIS

To evaluate the neuron activation across diverse query contexts, we constructed a benchmark of 16 predictive queries based on the Kickstarter dataset. We first selected 8 seed queries (Q1-Q8) from the workload and applied minor perturbations—such as adding, deleting, or modifying filtering conditions—to generate a corresponding set of variants (Q9- Q16).

TABLE V: Filter Predicates for the Queries.  
ID Filter Predicates P   
Q1 main\_category = ‘technology’ AND   
sub\_category = ‘Gadgets’   
Q2 sub\_category = ‘Video Games’   
Q3 main\_category = ‘music’ AND 1000 < goal   
< 2000   
Q4 main\_category = ‘publishing’ AND   
country = ‘GB’ AND 20 < duration < 30   
Q5 main\_category = ‘film & video’ AND   
sub\_category = ‘Drama’   
Q6 sub\_category = ‘Drinks’   
Q7 main\_category = ‘fashion’ AND goal <   
2000   
Q8 main\_category = ‘art’   
Q9 main\_category = ‘technology’ AND   
sub\_category = ‘Gadgets’ AND city =   
‘Hong Kong’   
Q10 sub\_category = ‘Video Games’ AND   
country = ‘FR’   
Q11 main\_category = ‘music’ AND goal < 2000   
AND duration > 30   
Q12 main\_category = ‘publishing’ AND   
country = ‘CA’   
Q13 sub\_category = ‘Drama’ AND city =   
‘London’   
Q14 sub\_category = ‘Vegan   
Q15 main\_category = ‘fashion’ AND goal >   
2000   
Q16 sub\_category = ‘Sculpture’ AND currency   
= ‘GBP’

All predictive queries follow a unified structure, formally defined as:

SELECT MODEL(name\_blurb)

FROM Kickstarter

(17)

WHERE P;

where P represents the set of variable predicates detailed in Table V.

Given that Q9-Q16 are derivatives of Q1-Q8, their result sets often exhibit inclusion relationships (e.g., a modified query’s results being a subset of the original). To decouple the analysis of neuron activation patterns from inherent data overlap, we randomly sampled 200 distinct tuples from the result set of each query rather than using the full retrieval sets. We then monitored the activation states of a fine-tuned dense model on these samples. Aligned with our experimental sparsity setting—which maintains a 25% activation rate—we focused our analysis exclusively on the top 768 neurons (the top 25% of the post-activation hidden layer) ranked by average magnitude.

![](images/767a6401021f8a7b803bc6d1d9a77574e42ca22dfbb012c5eaf58d38bee6598e.jpg)  
Fig. 15: Cross-Query Top-768 Neuron Overlap Matrix. This heatmap visualizes the pairwise overlap percentage of the top-768 most active neurons across 16 distinct query contexts $( Q 1 -$ Q16). Darker blue indicates higher overlap (similarity), while lighter colors indicate distinct activation patterns. Diagonal elements are 100% by definition.

The pairwise overlap rates of these top-ranked neurons are visualized in Fig. 15. We observe strong context locality: there is significant divergence between unrelated groups (e.g., between the Q1-Q8 block and unrelated queries in Q9-Q16), whereas semantically related pairs (e.g., Q1 & Q9, Q2 & Q10) exhibit high similarity. This empirical evidence supports our core hypothesis: different query contexts induce distinct activation patterns in Transformer feed-forward networks (FFNs).

## HYPERPARAMETER SETTINGS

In Stage I: Expert Construction, we first employ the KNN algorithm to partition the dataset into $N = 4$ semantic groups. Subsequently, we sample $\mathcal { D } _ { n } = 5 0 0$ representative instances from each group to perform activation profiling. This sample size is chosen to mitigate statistical noise while ensuring robust neuron distinction, thereby maintaining high profiling accuracy with minimal offline computational overhead.

Based on this grouping, we adopt an N4K1 configuration, where the feed-forward network is partitioned into $N = 4$ experts, and $K = 1$ expert is dynamically activated during inference. This results in a fixed activation rate of 25%, consistent with settings established in prior work [1], [2]. We prioritize the $N = 4 , K = 1$ configuration because finergrained splitting $( \mathrm { e . g . } , N = 8 , K = 2 { \mathrm { ~ o r ~ } } N = 1 6 , K = 4 )$ while potentially allowing better collaboration among semantic experts, incurs significant inference latency and introduces load balancing challenges while yielding only limited accuracy gains.

Furthermore, we designate 50% of the neurons as shared neurons $U _ { \mathrm { s h a r e d } }$ and the remaining 50% as context-specific neurons. This split is motivated by the observation in Fig. 15, which indicates that the overlap of highly activated neurons across varying query contexts is approximately 50%.

For One-time Feature Selection in Stage II, we train a Random Forest classifier on a subset of 30,000 samples to evaluate metadata importance, selecting the top $M \ = \ 3$ columns as effective context attributes. Empirically, we find that selecting the top three features is sufficient for the datasets used in this work, although the optimal M may vary across different datasets.

In Context-Aware Expert Routing, the router is implemented as a Transformer block with a hidden dimension of 512 and 4 attention heads, followed by a 3-layer MLP with a dimension of 256.

During Phase I: Router Semantic Alignment of Decoupled Training Strategy, we generate ground-truth labels using a calibration coefficient $\eta \ : = \ : 0 . 0 2$ and a softmax temperature $\tau = 1 . 0$ . We selected $\eta = 0 . 0 2$ based on observations from the training set: this value ensures that the distribution of selected expert indices aligns well with the ranking of expert scores, while simultaneously preventing load imbalance caused by the under-utilization of lower-scoring experts.

Finally, during Phase II: Context-Driven Joint Expert Training of Decoupled Training Strategy, we set the distillation loss weight to $\alpha = 5 . 0$ , consistent with [1].

## REFERENCES

[1] S. Zuo, Q. Zhang, C. Liang, P. He, T. Zhao, and W. Chen, “Moebert: from bert to mixture-of-experts via importance-guided adaptation,” in Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2022, pp. 1610–1623.

[2] Z. Zhang, Y. Lin, Z. Liu, P. Li, M. Sun, and J. Zhou, “Moefication: Transformer feed-forward layers are mixtures of experts,” in Findings of the Association for Computational Linguistics: ACL 2022, 2022, pp. 877–890.