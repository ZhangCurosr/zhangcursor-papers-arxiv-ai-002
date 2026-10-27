# UniGFM-RAG: Unify Graph Foundation Models and Large Language Models for End-to-End Retrieval and Reasoning

Zicheng Zhao<sup>∗</sup>, Linhao Luo, Zongji Yu, Junnan Dong, Yanchao Tan, Shirui Pan<sup>†</sup>, Senior Member, IEEE, and Chen Gong<sup>†</sup>, Senior Member, IEEE

Abstract—Large Language Models (LLMs) have shown remarkable capabilities across tasks, but they still struggle with complex reasoning that requires integrating scattered knowledge. Graph Retrieval-Augmented Generation (GraphRAG) addresses this challenge by organizing fragment knowledge as graphs, while recent Graph Foundation Models (GFMs) further improve GraphRAG with graph-native retrieval and reasoning capabilities. However, existing GraphRAG methods usually adopt decoupled retrieve-then-generate pipelines, which create objective misalignment between retrieval and reasoning, and training GFM retrievers typically depends on expensive human annotations. To address these limitations, we propose UniGFM-RAG, a unified end-to-end GraphRAG framework that tightly couples a GFM retriever with an LLM generator. Given a query, the GFM retriever first reasons over the graph to score and retrieve relevant document nodes. These node embeddings are then passed through a fully differentiable graph bridge, which projects them into soft-weighted graph prefixes according to retrieval scores and combines them with raw document texts to form graph-augmented prompts. In this way, structural information is injected directly into LLM generation, while the final answer loss can be back-propagated through the bridge to jointly optimize retrieval and reasoning. To stabilize training and reduce reliance on costly annotations, we further introduce a semantic-guided distillation module that provides auxiliary supervision using pseudo-labels derived from query-document semantic similarities. Extensive experiments on three multi-hop QA datasets and three out-of-domain GraphRAG benchmarks show that UniGFM-RAG consistently outperforms strong baseline methods and generalizes well, which demonstrates the effectiveness of endto-end GraphRAG with tightly integrated GFM retrieval and LLM reasoning.

Index Terms—Large Language Models, Graph Retrieval-Augmented Generation, Graph Foundation Models

## I. INTRODUCTION

Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of natural language processing tasks [1]–[5]. However, they still struggle with knowledge-intensive reasoning, where the evidence is often distributed across multiple documents [6]. Graphs [7]–[10] offer a natural and flexible way of modeling structural information, which makes them particularly suitable for capturing the dependencies within knowledge for complex reasoning. Motivated by this, Graph Retrieval-Augmented Generation (GraphRAG) [11]–[14] has emerged as a promising paradigm for enhancing LLM reasoning with graph-structured knowledge. As illustrated in the upper part of Figure 1, a typical GraphRAG system first constructs a graph from the external corpus, such as a knowledge graph [15] or a document graph [16], where nodes correspond to documents or entities, and edges encode their relationships. It then employs a graph retriever to navigate the graph and identify relevant nodes, whose associated information is provided to an LLM generator to produce the final answer.

![](images/7c496618d8c197ca09deb2ec0e32341ea939644edb4d6f12123745ded7491d08.jpg)  
Fig. 1: Comparison between the current two-stage GraphRAG and the proposed UniGFM-RAG. Existing GraphRAG methods usually employ a decoupled retrieve-then-generate pipeline, which often results in objective misalignment. In contrast, UniGFM-RAG introduces a unified end-to-end GraphRAG framework that tightly couples a Graph Foundation Model (GFM) retriever with a Large Language Model (LLM) generator.

Early graph retrievers relied on the internal capabilities of LLMs to traverse graphs using heuristic search [17], global planning [18], or multi-step agentic reasoning [19], [20]. However, these approaches often suffer from inefficiency and limited structural understanding due to the unstructured nature of LLMs. Recently, Graph Foundation Models (GFMs) [21]– [24] have demonstrated strong graph reasoning and generalized abilities, which inspires a new wave of applying them to GraphRAG [25]. These methods utilize graph neural network (GNN)-based GFMs as native graph retrievers to capture intricate dependencies among nodes, which significantly enhances retrieval performance. Despite the promising results, training GFMs requires large-scale, high-quality relevance labels, which are labor-intensive and difficult to obtain at scale (Limitation 1).

Beyond the challenge of data acquisition, the architecture of existing GraphRAG frameworks poses another major bottleneck. Most GraphRAG methods adopt a decoupled, two-stage “retrieve-then-generate” pipeline [25]–[27]. In this paradigm, the retriever and generator are optimized separately for different objectives: the retriever is trained to maximize retrieval performance rather than improve final answer accuracy. This leads to significant objective misalignment, where information considered highly relevant by the retriever may be suboptimal or redundant for the reasoning process of the generator (Limitation 2). Furthermore, because document retrieval is a discrete operation, it prevents continuous gradient backpropagation from the generator to the retriever, which makes joint training even more difficult. Therefore, it is crucial to tightly integrate the GFM retriever and LLM generator into a unified framework that allows end-to-end optimization.

Beyond enabling joint optimization, a unified GFM-LLM framework can further improve the ability of LLM to leverage graph structure during reasoning [28]. Prior work has attempted to inject graph structural signals into LLMs through GNN-based graph encoders and graph-aware prompting [29]– [31]. Although effective for standalone graph tasks such as anomaly detection [32], node classification and link prediction [33], these methods assume a pre-defined graph as input and are not designed for retrieval-augmented settings. As a result, they neither address how to retrieve query-relevant information from external graph nor capture the interaction between graph retrieval and answer generation (Limitation 3), which is essential in GraphRAG.

In response to these challenges, we propose UniGFM-RAG, a unified end-to-end GraphRAG framework that tightly couples a GFM retriever with an LLM generator for joint retrieval and reasoning. As shown in the bottom of Figure 1, UniGFM-RAG fosters a reciprocal loop: the structural insights of GFM enhance LLM reasoning, while LLM reasoning refines graph retrieval. Given a query, the GFM retriever first reasons over the graph to score and retrieve top-k relevant document nodes. Instead of directly feeding the retrieved documents into the LLM, we introduce a novel graph bridge module to connect the GFM and LLM through a fully differentiable architecture. Specifically, the node embeddings of retrieved documents are extracted from the GFM and projected into “graph prefixes” that encode their structural roles and inter-document dependencies, and are then soft-weighted by the corresponding retrieval scores. These graph prefixes are concatenated with the raw document texts to form graphaugmented prompts for the LLM. In this way, the structural knowledge captured by the GFM is injected directly into the generation process, which enables the LLM to reason not only over the retrieved contents but also over their graph-structured dependencies (Addressing Limitation 3).

More importantly, the graph bridge remains fully differentiable, which allows gradients from the final answer generation loss to flow back through the graph prefixes and soft weights to refine both the LLM generator and GFM retriever. This directly aligns retrieval with downstream reasoning objectives and enables joint optimization of the two modules (Addressing Limitation 2). To further stabilize training, we introduce a semantic-guided distillation module that provides auxiliary supervision for the GFM retriever using pseudolabels derived from query-document semantic similarities. This module helps the retriever learn robust topological navigation from semantic priors without requiring expensive fine-grained human annotations (Addressing Limitation 1).

The core contributions of this work are summarized as follows:

• We propose UniGFM-RAG, a unified end-to-end GraphRAG framework that unifies a GFM retriever with an LLM generator, which enables joint optimization of retrieval and reasoning instead of the conventional decoupled retrieve-then-generate pipeline.

• We design a differentiable graph bridge that transforms retrieved document nodes into soft-weighted graph prefixes and graph-augmented prompts, which injects graph structure into the LLM while allowing answer supervision to back-propagate to the GFM retriever.

• We introduce a semantic-guided distillation module that stabilizes GFM training with pseudo-labels derived from query-document semantic similarity, which reduces reliance on costly fine-grained human annotations.

• Extensive experiments on three multi-hop QA datasets and three out-of-domain GraphRAG benchmarks show that UniGFM-RAG consistently outperforms strong baselines and generalizes effectively across domains, which demonstrate the effectiveness of our unified framework.

## II. RELATED WORK

In this section, we review some representative prior works for Graph Retrieval Augmented Generation (GraphRAG) and Graph Foundation Models (GFMs).

## A. Graph Retrieval Augmented Generation

Graph retrieval-augmented generation has emerged as a promising paradigm that retrieves query-relevant graph elements (e.g., node, triples, paths, or subgraphs) from a preconstructed graph database (e.g., knowledge graphs [34], [35]). By explicitly modeling the inherent logical relationships among isolated pieces of knowledge, these methods significantly enhance the retrieval process beyond standard document retrieval. Recent approaches further integrate textual documents into knowledge graphs to facilitate more structured and semantically rich retrieval. For example, GraphRAG [15] employs community detection algorithms to summarize and organize documents, while LightRAG [36] introduces a duallevel retrieval mechanism that combines structural graph retrieval with semantic vector retrieval to efficiently locate entities and their relations. Similarly, HippoRAG [17] utilizes the Personalized PageRank (PPR) algorithm to identify salient graph entities and subsequently maps them to relevant documents. However, constructing these graphs is often computationally expensive and token-intensive, and the process can introduce noise, which limits the practical deployment of these methods in real-world scenarios. Concurrently, efforts to incorporate Graph Neural Networks (GNNs) into graphenhanced RAG [37], [38] have yielded expressive results, largely due to the multi-hop graph reasoning capabilities of GNNs over incomplete graphs [39]–[41]. Nevertheless, these methods still struggle to generalize across diverse domains due to the absence of a unified graph foundation model.

## B. Graph Foundation Models

Graph Foundation Models (GFMs) leverage large-scale pretraining across massive datasets to create unified models with strong generalizability across various downstream tasks [21], [22], [42]. A primary challenge lies in defining graph tokens capable of capturing invariance across diverse and heterogeneous graph data [43]. For example, ULTRA [44] designs four fundamental relational interactions in knowledge graphs (KGs) to create a lightweight GFM with 0.2M parameters for link prediction. OpenGraph [45] develops a graph tokenizer to convert graphs into a unified node token representation, which enables transformer-like GFMs for tasks such as link prediction and node classification. GFT [46] introduces a transferable tree vocabulary to conduct a GFM that demonstrates effectiveness across diverse domains in graph learning. Despite these successful efforts, most methods primarily focus on conventional graph-related tasks and Transformer-like GFMs [29], [47] often struggle to scale to large graphs.

To combine the power of GFM and LLM, GraphLLM [30] integrates graph learning models with LLM to equip LLM with the ability to reason on graph data. GraphGPT [29] also incorporates a text-graph grounding component to establish connections between text and graph structures for LLMs. GraphTranslator [48] bridges GFM and LLM with a translator, aiming to transfer the node representations into tokens, based on which LLM can perform node classification and link prediction. Beyond conventional graph learning tasks, GFM-RAG [25] adopts a GFM-based retriever for complex question answering. However, this method relies heavily on the groundtruth query-document relevance labels for training the retriever and employs a disjoint two-stage pipeline. This inherently limits end-to-end optimization and often leads to sub-optimal reasoning performance. Therefore, designing a joint framework that seamlessly integrates a GFM-based retriever with a generator remains an open and critical question.

TABLE I: Glossary of commonly used notations.
<table><tr><td>Notation</td><td>Description</td></tr><tr><td colspan="2">Knowledge Graph &amp; Query</td></tr><tr><td>G</td><td>The heterogeneous knowledge graph</td></tr><tr><td> $v \in \mathcal V$ </td><td>A generic node v in the node set V</td></tr><tr><td> $r \in \mathcal { R }$ </td><td>A relation type r in the relation set R</td></tr><tr><td> $( v , r , v ^ { \prime } ) \in \mathcal { E }$ </td><td>An edge  $( \boldsymbol { v } , \boldsymbol { r } , \boldsymbol { v } ^ { \prime } )$  in the edge set E</td></tr><tr><td> $t _ { v } \in \mathcal { T }$ </td><td>The node type of v, belonging to the type set T</td></tr><tr><td> $q , a$ </td><td>The user query and the final answer</td></tr><tr><td> $d \in \mathcal { D }$ </td><td>A document d within the document subset  $\mathcal { D } \subset \mathcal { V }$ </td></tr><tr><td colspan="2">Representations</td></tr><tr><td> $D _ { g n n }$   $\bar { D _ { l l m } }$ </td><td>The hidden dimensions of the GNN embeddings The hidden dimensions of the LLM embeddings</td></tr><tr><td> $L$   $\mathbf { h } _ { q } , \mathbf { h } _ { d }$ </td><td>The total number of GNN layers Initial text embeddings of the query q and document d</td></tr><tr><td> $\mathbf { h } _ { v } ^ { ( l ) }$ </td><td>The embedding of node v at the l-th GNN layer</td></tr><tr><td> $\mathbf { m } _ { v } ^ { ( l ) }$ </td><td>The aggregated message received by node v at the l-th layer</td></tr><tr><td> $\mathbf { h } _ { d } ^ { ( L ) }$ </td><td></td></tr><tr><td> $\tilde { \mathbf { h } } _ { d }$ </td><td>The final graph embedding of document node</td></tr><tr><td></td><td>The structural prefix token projected from the graph embedding</td></tr><tr><td> $\mathbf { V } _ { d }$ </td><td>Dense token embedding sequence of document d</td></tr><tr><td> $\hat { \mathbf { V } } _ { d }$ </td><td>The soft-weighted, graph-augmented prompt representation for d</td></tr><tr><td colspan="2">Retrieval &amp; Distillation</td></tr><tr><td> $\mathcal { D } _ { k }$ </td><td>The subset of the top-k retrieved document nodes</td></tr><tr><td> $s _ { d }$ </td><td>The predicted discrete relevance score for document d</td></tr><tr><td> $w _ { d }$ </td><td>The continuous soft weight for document d,  $w _ { d } \in ( 0 , 1 )$ </td></tr><tr><td>Pt</td><td>The semantic pseudo-label distribution generated by the teacher</td></tr><tr><td>pθ</td><td>The predicted retrieval distribution by the GFM student</td></tr><tr><td colspan="2">Optimization</td></tr><tr><td> $\lambda$ </td><td>Hyperparameter balancing the generation and distillation losses</td></tr><tr><td> $\theta , \phi$ </td><td>Trainable parameters of the GFM retriever and LLM generator</td></tr><tr><td> $\mathcal { L } _ { 1 }$ </td><td>The negative log-likelihood (NLL) generation loss of the LLM</td></tr><tr><td> $\mathcal { L } _ { 2 }$ </td><td>The semantic-guided KL divergence distillation loss</td></tr><tr><td> $\mathcal { L }$ </td><td>The overall optimization objective</td></tr></table>

## III. APPROACH

The overall framework of UniGFM-RAG is illustrated in Figure 2, which consists of three core components: (a) a graph foundation model retriever to effectively reason over the structured knowledge for document retrieval, (b) a graph bridge module to connect the retriever and generator into a unified framework, and (c) a graph-conditioned LLM generator that effectively incorporates graph information into the LLM generation process and enables end-to-end training. In the following sections, we will detail the formulation and design of each module.

GraphRAG Overview. To contextualize our proposed framework, we begin by formally defining the general GraphRAG process. Given a set of documents $\mathcal { D } \ : = \ : \{ d _ { 1 } , d _ { 2 } , \cdot \cdot \cdot , d _ { n } \}$ following common GraphRAG approaches [15], [25], we employ LLMs to extract structural knowledge and construct a knowledge graph $\mathcal { G } = ( \mathcal { V } , \mathcal { E } , \mathcal { R } , \mathcal { T } )$ . Here, V and R denote the sets of nodes and relations, respectively, and ${ \mathcal { E } } = \{ ( v , r , v ^ { \prime } ) \subseteq $ $\mathcal { V } \times \mathcal { R } \times \mathcal { V } \}$ represents the set of edges that model the relationship among knowledge. Additionally, T specifies the node types (e.g., entities, documents).

Given the constructed graph G and a user query $q ,$ standard GraphRAG paradigms first retrieve query-relevant information and then leverage an LLM to reason over the retrieved context and generate the final answer a. Formally, a graph retriever $R _ { \theta }$ identifies the top-k query-relevant document nodes $\mathcal { D } _ { k } \subset \mathcal { D }$ according to the predicted probability $p _ { \theta } ( \mathcal { D } _ { k } \mid q , \mathcal { G } , \mathcal { D } )$ , and a generator $G _ { \phi }$ then produces the final answer a conditioned on both the query and the retrieved documents, which can be formulated as:

![](images/ff7447304c10f3ee0d629e2cc257092977c26bf7b0f41159dffa222befedeb7a.jpg)  
Fig. 2: The overall framework of UniGFM-RAG, which comprises of three core components: (a) a graph foundation model retriever, which reasons over the structural knowledge for accurate document retrieval; (b) a graph bridge module, which seamlessly connects the retriever and the generator into a unified framework; (c) a graph-conditioned LLM generator, which incorporates the structural information to guide the final answer generation. In addition, we introduce a semantic-guided distillation module to provide auxiliary supervision for training GFMs.

$$
\mathcal { G } = \mathrm { G r a p h C o n s t r u c t o r } \left( \mathcal { D } \right) ,\tag{1}
$$

$$
p _ { \Theta } ( a \mid q , \mathcal { G } , \mathcal { D } ) = \sum _ { \mathcal { D } _ { k } \subseteq \mathcal { D } } p _ { \phi } ( a \mid q , \mathcal { D } _ { k } ) p _ { \theta } ( \mathcal { D } _ { k } \mid q , \mathcal { G } , \mathcal { D } ) ,\tag{2}
$$

where $\Theta = \{ \theta , \phi \}$ denote the combined parameters of the retriever and generator.

## A. Graph Foundation Model Retriever

Graph foundation models (GFMs) [21], [22] have demonstrated strong capabilities in capturing complex topological and semantic relationships within graph-structured data, which makes them ideal candidates for graph-based retrieval tasks. In UniGFM-RAG, we leverage a GFM as the backbone of our retriever to effectively reason over the knowledge graph and identify query-relevant document nodes.

Specifically, we adopt a query-dependent Graph Neural Network (GNN) architecture [25], [44] as the core of our GFM retriever. This design allows the model to explicitly incorporate both relational semantics and rich semantic features in node text into the message-passing process, which enables queryaware graph reasoning.

Text Encoding. Given a graph G, we first utilize a pretrained text embedding model (e.g., Qwen3-Embedding [49]) to respectively encode the textual content associated with each node $v \in \mathcal V$ and each relation $r \in \mathcal { R }$ into initial d-dimensional node embeddings $\mathbf { h } _ { v } \in \mathbb { R } ^ { d }$ and relation embeddings $\mathbf { h } _ { r } \in \mathbb { R } ^ { d }$

which can be formulated as:

$$
\mathbf { h } _ { v } = \mathrm { T e x t E m b e d d i n g } \left( \mathrm { T e x t } ( v ) \right) , v \in \mathcal { V } ,\tag{3}
$$

$$
\mathbf { h } _ { r } = { \mathrm { T e x t E m b e d d i n g } } \left( { \mathrm { T e x t } } ( r ) \right) , r \in { \mathcal { R } } ,\tag{4}
$$

where Text(·) denotes the textual content associated with the nodes and relations. These semantic embeddings preserve the original textual knowledge, which provides a robust initialization for the subsequent GNN reasoning. Similarly, for a given user query q, we employ the same text embedding model to extract its dense representation $\mathbf { h } _ { q } \in \mathbb { R } ^ { d }$ , formulated as:

$$
\mathbf { h } _ { q } = { \mathrm { T e x t E m b e d d i n g } } \left( q \right) .\tag{5}
$$

By sharing the same encoding strategy, the query and graph elements are projected into the same latent semantic space. This approach not only enables generalization across different graphs and queries but also facilitates effective interaction between the query and the graph during GNN reasoning.

Query-dependent GNN Reasoning. After obtaining the initial embeddings, we apply an L-layer query-dependent GNN to jointly reason over text semantics and graph topology. The GNN updates the node representations iteratively by aggregating information from their neighbors, which allows the model to capture intricate dependencies among nodes and produce query-aware relevance scores for document retrieval. Formally, the GNN updates are defined as follows:

$$
\mathbf h _ { v } ^ { ( 0 ) } = \mathrm { I n i t } \left( \mathbf h _ { v } , 1 _ { v \in \mathcal V _ { q } } \mathbf h _ { q } \right) , \quad v \in \mathcal V ,\tag{6}
$$

$$
\mathbf { m } _ { v } ^ { ( l ) } = \mathrm { A g g } \left( \left\{ \mathrm { M s g } \left( \mathbf { h } _ { v } ^ { ( l - 1 ) } , \mathbf { h } _ { r } ^ { ( l ) } , \mathbf { h } _ { v ^ { \prime } } ^ { ( l - 1 ) } \right) \mid ( v , r , v ^ { \prime } ) \in \mathcal { E } \right\} \right) ,\tag{7}
$$

$$
\mathbf { h } _ { v } ^ { ( l ) } = \mathrm { U p d a t e } \left( \mathbf { h } _ { v } ^ { ( l - 1 ) } , \mathbf { m } _ { v } ^ { ( l ) } \right) , \quad l \in [ 1 , L ] ,\tag{8}
$$

where $\mathbf { h } _ { v } ^ { ( l ) }$ denotes the embedding of node v at the l-th GNN layer. The Init operator initializes the node embeddings by a single-layer MLP, which fuses the original node embedding $\mathbf { h } _ { v }$ and the query embedding $\mathbf { h } _ { q }$ if the node v belongs to the set of query-related nodes $\mathcal { V } _ { q } \ ( \mathrm { i . e . , } 1 _ { v \in \mathcal { V } _ { q } } )$ . Query-related node set $\mathcal { V } _ { q }$ is constructed by semantic retrieval. Specifically, we encode the query and all graph nodes by an embedding model (e.g., Qwen3-Embedding-0.6B), compute their cosine similarities, and select the top-k most similar entity and document nodes as query-related nodes.

During the message-passing phase at each layer l, the Msg function employs DistMult [50] to compute the incoming messages from neighbors based on target node embedding $\mathbf { h } _ { v } ^ { ( l - 1 ) } ,$ , the source node embedding $\mathbf { h } _ { v ^ { \prime } } ^ { ( \lfloor - 1 \rfloor ) }$ and the relation embedding $\mathbf { h } _ { r } ^ { ( l ) }$ . These messages are subsequently pooled by the $\mathtt { A g g }$ function (e.g., sum aggregation). Finally, the Update function computes the target node embedding $\mathbf { h } _ { v } ^ { ( l ) }$ by combining the aggregated messages $\mathbf { m } _ { v } ^ { ( l ) }$ with its previous representation $\mathbf { h } _ { v } ^ { ( l - \top ) }$ with another MLP. Concurrently, relation embeddings $\mathbf { h } _ { r } ^ { ( l ) }$ are updated with a layer-specific MLP, defined as $\mathbf { h } _ { r } ^ { ( l ) ^ { \smile } } = \operatorname { M L P } ( \mathbf { h } _ { r } ^ { ( l ^ { \bot } - 1 ) } $ ).

Document Retrieval. By encoding the textual content of graph nodes and relations into dense embeddings and applying L-layer of GNN, the GFM retriever can capture intricate dependencies among nodes and produce query-aware relevance scores for document retrieval. The final query-aware node representation $\mathbf { h } _ { v } ^ { ( L ) }$ is then used to compute a relevance score $s _ { v }$ for document retrieval as follows:

$$
s _ { v } = \mathrm { P r e d i c t o r } \left( \mathbf { h } _ { v } ^ { ( L ) } , \mathbf { h } _ { v } , \mathbf { h } _ { q } \right) ,\tag{9}
$$

where $\mathbf { h } _ { v }$ and $\mathbf { h } _ { q }$ denote the original text embeddings of node v and query $q ,$ respectively, and the predictor is implemented as a two-layer MLP to produce a relevance score $s _ { v }$ for each document node.

The top-k document nodes with the highest relevance scores are selected as the retrieved documents for subsequent generation. Formally, the retrieved document set $\mathcal { D } _ { k }$ is defined as:

$$
\mathcal { D } _ { k } = \mathrm { T o p } - k ( s _ { d } ) , d \in \mathcal { D } .\tag{10}
$$

However, this hard selection operation is inherently nondifferentiable, which prevents end-to-end optimization between the retriever and the generator and prevents the abundant structural information encoded by the GFM from being effectively utilized by the LLM generator.

## B. Graph Bridge Module

To construct a differentiable bridge between the GFM retriever and the LLM generator, we introduce a novel graph bridge module that transforms the retrieved document nodes into soft-weighted graph prefixes and seamlessly integrates them into the LLM generation process. The graph bridge serves as a continuous interface that allows the structural information from the GFM to be directly injected into the LLM while enabling gradients from the final answer loss to back-propagate to the GFM retriever for joint optimization.

We first relax the discrete retrieval scores $s _ { d }$ into continuous soft-weights $w _ { d }$ by passing them through a Sigmoid function, which preserves the independent relevance of each document without forcing competitive normalization. This can be formulated as:

$$
w _ { d } = \frac { 1 } { 1 + \exp ( - s _ { d } ) } , d \in \mathcal { D } _ { k } .\tag{11}
$$

These differentiable soft-weights explicitly quantify the importance of each retrieved document and allow the retriever to learn which documents maximize the generator’s generation performance.

To inject structural information, we reuse the node representations of retrieved documents $\{ \mathbf { h } _ { d } ^ { ( L ) } \mid d \in \mathcal { D } _ { k } \}$ obtained from the final GNN layer in GFM retriever, which is defined in Eq. (8). These representations encapsulate the rich topological context of the document d within the graph, which will be transformed into the graph prefixes to guide the generation process of LLM.

A two-layer MLP (i.e., Projector) is first adopted as an alignment projector to map the structural representation into the semantic space of the LLM. Crucially, to allow the gradient to flow back to the retriever, we scale the projected representations by the differentiable soft-weighted $w _ { d }$ derived via Eq. (11), which is formulated as follows:

$$
\tilde { \mathbf { h } } _ { d } = w _ { d } \cdot \mathrm { P r o j e c t o r } \left( \mathbf { h } _ { d } ^ { ( L ) } \right) , d \in \mathcal { D } _ { k } .\tag{12}
$$

This soft-weighted structural representation $\tilde { \mathbf { h } } _ { d } \in \mathbb { R } ^ { D _ { l l m } }$ with $D _ { l l m }$ dimensions serves as a special graph prefix, which explicitly encodes the structural information and relevance of the retrieved document to guide the LLM generation.

## C. Graph-conditioned LLM Generator

By integrating the graph prefixes into the LLM generation process, we enable the LLM to effectively leverage the graph structure during reasoning, which significantly enhances its performance on complex tasks that require reasoning over multiple knowledge.

Graph-augmented Prompt Construction: For each retrieved document $d \in \mathcal { D } _ { k }$ , we first extract its textual content and tokenize it into a sequence of tokens. These tokens are then mapped to their dense embedding sequence $\mathbf { V } _ { d } \in \mathbb { R } ^ { | d | \times D _ { l l m } }$ using the LLM’s input embedding layer, where |d| represents the sequence length of the document and $D _ { l l m }$ denotes the hidden dimension of the LLM. This process captures the raw semantic information of the retrieved documents, which serves as the primary context for generation.

Then, we concatenate the graph prefix $\tilde { \mathbf { h } } _ { d }$ with the document embedding sequence $\mathbf { V } _ { d }$ to form a graph-injected representation ${ \hat { \mathbf { V } } } _ { d }$ for each retrieved document, which is formulated as follows:

$$
\begin{array} { r } { \hat { { \bf V } } _ { d } = \left[ \tilde { \bf h } _ { d } \parallel { \bf V } _ { d } \right] , d \in { \mathcal { D } } _ { k } , \hat { \bf V } _ { d } \in { \mathbb { R } } ^ { ( | d | + 1 ) \times D _ { l l m } } , } \end{array}\tag{13}
$$

where $\ " \Vert \mathbf { \cdots }$ denotes sequence concatenation.

Finally, the formulated representations $\hat { \mathbf { V } } _ { d }$ of all k retrieved documents are concatenated with the initial user query sequence to form the graph-augmented prompt for the LLM generator. The detailed prompt is illustrated in Figure 6 in

Appendix VII-C, where the <graph\_prefix> denotes the injected graph prefix $\tilde { \mathbf { h } } _ { d }$

Graph-conditioned Generation: Conditioned on this graphaugmented prompt, the LLM generator auto-regressively decodes the target answer $a = \{ a _ { 1 } , a _ { 2 } , \ldots , a _ { | a | } \}$ . The probability of generating the entire answer sequence is factorized as:

$$
p _ { \phi } ( a \mid q , \mathcal { D } _ { k } ) = \prod _ { t = 1 } ^ { | a | } p _ { \phi } ( a _ { t } \mid a _ { < t } , q , \hat { \mathbf { V } } _ { \mathcal { D } _ { k } } ) ,\tag{14}
$$

where $\hat { \mathbf { V } } _ { \mathcal { D } _ { k } } = \{ \hat { \mathbf { V } } _ { d } \} _ { d \in \mathcal { D } _ { k } }$

End-to-end Joint Optimization: During the training phase, the primary objective for the LLM generator is to minimize the standard negative log-likelihood (NLL) loss, denoted as $\mathcal { L } _ { 1 }$

$$
\mathcal { L } _ { 1 } = - \sum _ { t = 1 } ^ { | a | } \log p _ { \phi } \big ( a _ { t } \mid a _ { < t } , q , \hat { \mathbf { V } } _ { \mathcal { D } _ { k } } \big ) .\tag{15}
$$

The applying graph prefix ensures that the original language modeling capabilities of LLM remain undisturbed, while still establishing a fully differentiable bridge, which allows gradients from the LLM loss to back-propagate to the GFM retriever’s parameters θ through the prefix $\ddot { \mathbf { h } } _ { d }$

Building upon the chain rule, the gradient flow from each retrieved document explicitly splits into dual pathways as follows:

$$
\frac { \partial \mathcal { L } _ { 1 } } { \partial \theta } = \underbrace { \frac { \partial \mathcal { L } _ { 1 } } { \partial \tilde { \mathbf { h } } _ { d } } \frac { \partial \tilde { \mathbf { h } } _ { d } } { \partial w _ { d } } \frac { \partial w _ { d } } { \partial \theta } } _ { \mathrm { S o f t - w e i g h t i n g ~ P a t h w a y } } + \underbrace { \frac { \partial \mathcal { L } _ { 1 } } { \partial \tilde { \mathbf { h } } _ { d } } \frac { \partial \tilde { \mathbf { h } } _ { d } } { \partial \mathbf { h } _ { d } ^ { ( L ) } } \frac { \partial \mathbf { h } _ { d } ^ { ( L ) } } { \partial \theta } } _ { \mathrm { S t r u c t u r a l ~ P a t h w a y } } , d \in \mathcal { D } _ { k } ,\tag{16}
$$

where the two terms on the right-hand side denote the backpropagated gradient through the continuous weight $w _ { d }$ and the structured embedding $\mathbf { h } _ { d } ^ { ( \breve { L } ) }$ , respectively.

The soft-weighting mechanism offers two main advantages for joint optimization. During the forward pass, $w _ { d }$ acts as a continuous gate that scales the structural embedding and directs the attention of LLM to highly relevant documents while filtering out noise. In the backward pass, this pathway enables effective credit assignment to the retriever by explicitly measuring how changes in retrieval scores $s _ { d }$ affect the final generation loss $\mathcal { L } _ { 1 }$ . As a result, the retriever learns which documents best enhance generator performance, promoting synergistic optimization between retrieval and generation.

## D. Semantic-guided Distillation

In real-world applications, obtaining fine-grained, groundtruth annotations for query-document relevance is prohibitively expensive and often infeasible. To overcome this strict absence of direct supervision, we draw inspiration from knowledge distillation [51]. Specifically, we propose to train GFM through weak supervision by using the abundant unlabeled nodes on the graph. The pre-trained text embedding models [52] exhibit robust semantic reasoning capabilities, which allow them to effectively capture the underlying relevance between the query and documents. Therefore, we utilize a frozen, pre-trained text encoder as a teacher to generate semantic pseudo-labels for all document nodes D. The pseudo-label probability for a document d is formulated by applying a Softmax function over the inner products of their semantic embeddings, which can be formulated as

$$
p _ { t } ( d \mid q , \mathcal { D } ) = \frac { \exp ( \mathbf { h } _ { d } ^ { \top } \mathbf { h } _ { q } / \tau ) } { \sum _ { d ^ { \prime } \in \mathcal { D } } \exp ( \mathbf { h } _ { d ^ { \prime } } ^ { \top } \mathbf { h } _ { q } / \tau ) } ,\tag{17}
$$

where ${ \mathbf { h } } _ { q }$ and $\mathbf { h } _ { d }$ denote the original text embeddings of query q and the document $d ,$ respectively. The temperature parameter τ controls the smoothness of the probability distribution. A smaller temperature sharpens the distribution and assigns larger probabilities to more relevant documents, which helps provide high-confidence supervision signals during distillation.

Similarly, the student GFM retriever generates its predicted retrieval distribution $p _ { \theta } ( d \mid q , \mathcal { G } )$ by applying the identical operation Softmax over its predicted relevance scores $s _ { v } ,$ which is defined as Eq. (9). Finally, we optimize the GFM retriever by minimizing the Kullback-Leibler (KL) divergence between its predicted retrieval distribution and the semantically constructed pseudo-label distribution, which can be formulated as:

$$
\mathcal { L } _ { 2 } = \mathbb { E } _ { d \sim p _ { t } ( d | q , \mathcal { D } ) } \left[ \log \frac { p _ { t } ( d \mid q , \mathcal { D } ) } { p _ { \theta } ( d \mid q , \mathcal { G } ) } \right] .\tag{18}
$$

## E. Training Strategy and Inference

To unify the semantic reasoning of the LLM generator and the structural retrieval of the GFM retriever, we train the entire UniGFM-RAG framework fully from end-to-end. The overall optimization objective is formulated as a linear combination of the generation loss $\mathcal { L } _ { 1 }$ and the distillation loss $\mathcal { L } _ { 2 } \colon$

$$
\mathcal { L } = \mathcal { L } _ { 1 } + \lambda \mathcal { L } _ { 2 } ,\tag{19}
$$

where $\lambda$ is a tunable hyperparameter that balances the primary generation task and auxiliary retrieval mechanism.

Through this joint optimization paradigm, these two modules synergize effectively: (1) $\mathcal { L } _ { 2 }$ provides robust prior guidance for the retriever via semantic pseudo-labels, resolving the scarcity of annotated labels. (2) Concurrently, $\mathcal { L } _ { 1 }$ offers taskspecific gradients through our differentiable soft-weighting bridge, which enables fine-tuning the retriever to fetch document nodes that directly maximize the final answering accuracy.

## IV. EXPERIMENTS

In this section, we conduct extensive experiments to answer the following research questions: RQ1: Can UniGFM-RAG achieve the state-of-the-art performance in complex reasoning? RQ2: Can UniGFM-RAG effectively generalize across diverse domains and data? RQ3: How do the key components of UniGFM-RAG contribute to overall performance? RQ4: How sensitive is UniGFM-RAG to different hyper-parameter settings? RQ5: How does the choice of backbone model impact the effectiveness of UniGFM-RAG? RQ6: How efficient is UniGFM-RAG in terms of inference?

## A. Experimental Setup

Datasets. To comprehensively evaluate UniGFM-RAG, we conduct experiments on three widely adopted multi-hop QA datasets, including HotpotQA [53], MuSiQue [54], and 2Wiki-MultiHopQA (2Wiki) [55]. For the joint training phase, we utilize the training data curated by GFM-RAG [25], where each dataset split consists of 1000 questions along with their corresponding documents. During the evaluation phase, we strictly follow the testing settings established by HippoRAG [17], GFM-RAG [25] for a fair comparison. We first assess the in-domain effectiveness of our proposed method UniGFM-RAG on test datasets of the aforementioned three multi-hop QA benchmarks. Furthermore, to assess the zeroshot generalization capabilities of our framework across diverse domains, we employ three out-of-domain GraphRAG benchmarks: G-bench (Novel) [56], G-bench (Medical) [56], and G-bench (CS) [57]. These benchmarks rigorously test the complex reasoning skills of the model over medical, novel, and computer science (CS) knowledge without any domainspecific fine-tuning. The detailed statistics of the evaluation datasets are summarized in Table II.

Baseline Methods. To rigorously evaluate the superiority of UniGFM-RAG, we compare it against a diverse set of strong baseline methods, which are broadly categorized into three groups. (1) Pure LLM Methods: This category directly employs the pure LLM for answer generation without relying on external retrieval, including zero-shot (ZS) GPT-4o-mini [58] and Qwen3-8B [2] as well as a fine-tuned (FT) Qwen3- 8B on the training data. (2) Traditional RAG Methods: This category represents traditional sparse and dense retrieval models that rely solely on lexical or semantic matching without structural awareness, including BM25 [59], ColBERTv2 [60], and Qwen3-Emb (e.g., Qwen3-Embedding-0.6B) [49]. (3) Graph-enhanced Methods: This group consists of recent advancements that incorporate structural knowledge or graph reasoning into the retrieval and generation pipeline, including RAPTOR [61], GraphRAG (MS) [15], LightRAG [36], HippoRAG [17], SubgraphRAG [62], G-retriever [37], and GFM-RAG [25]. Detailed descriptions of these baseline methods are provided in Appendix VII-B. Crucially, since our realistic problem setting assumes no access to ground-truth labels for query-document relevance, we meticulously re-run the experiments for SubgraphRAG, G-retriever, and GFM-RAG under identical condition to ensure a fair comparison. Meanwhile, all baseline methods employ gpt-4o-mini as reasoning models, with the exception of Qwen3-8B (ZS) and Qwen3-8B (FT). Further, due to computational constraints, the experiments for RQ4, RQ5, and RQ6 were performed on a sampled subset of the original training datasets.

Metrics. We assess the performance of UniGFM-RAG across two primary dimensions: final QA reasoning performance and intermediate retrieval performance. For QA reasoning performance, we report Exact Match (EM) and F1 scores, aligning with established practices in prior works [17], [25]. Conversely, for the G-bench datasets, we report standard Accuracy (ACC) to maintain strict consistency with their original benchmarks [56], [57]. Regarding retrieval performance, we adopt document recall@2 (R@2) and recall@5 (R@5) for multi-hop QA and evidence recall (Recall) for G-bench datasets as evaluation metrics.

Implementation Details. We train UniGFM-RAG using the large-scale corpus curated from GFM-RAG, which consists of 60,000 query samples and 737,310 documents. For the architecture, the GFM retriever is implemented as a 6-layer query-dependent GNN with a hidden dimension of 1024. During message passing, we employ a DistMult function combined with sum aggregation, while the relation update mechanism is parameterized by a 2-layer MLP. The initial text representations are encoded using Qwen3-Embedding-0.6B. The total parameter of the retriever is 34M. The LLM generator is instantiated with the Qwen3-8B model.

TABLE II: Statistics of the evaluation datasets.
<table><tr><td>Dataset</td><td># Query</td><td># Document</td></tr><tr><td>HotpotQA [53]</td><td>1,000</td><td>9,221</td></tr><tr><td>MuSiQue [54]</td><td>1,000</td><td>6,119</td></tr><tr><td>2Wiki [55]</td><td>1,000</td><td>11,656</td></tr><tr><td>G-bench (Novel) [56]</td><td>2,010</td><td>461</td></tr><tr><td>G-bench (Medical) [56]</td><td>2,062</td><td>2,406</td></tr><tr><td>G-bench (CS) [57]</td><td>1,018</td><td>24,534</td></tr></table>

Our proposed UniGFM-RAG is trained with 8 NVIDIA H200 GPUs (141G) for 10 epochs with a per-device batch size 1. We optimize the framework using AdamW with differential learning rates to accommodate the varying convergence dynamics: 1e-4 for the randomly initialized GFM retriever and 2e-5 for the LLM generator. Finally, the loss-balancing hyperparameter λ is set to 1. We set k = 5 for multi-hop QA datasets and k = 10 for G-bench datasets. We set the teacher temperature to $\tau _ { t } = 0 . 0 5$ to sharpen the teacher distribution and provide high-confidence relevance supervision, while the student temperature is set to $\tau _ { s } = 1 . 0$ during training.

## B. Main Results (RQ1)

QA Results. As shown in Table III, UniGFM-RAG outperforms all baseline methods across three multi-hop question answering datasets, demonstrating its effectiveness in reasoning over graph-structured knowledge for complex reasoning. Pure LLM methods (e.g., GPT-4o-mini), even after finetuning (Qwen3-8B (FT)), perform poorly on these tasks due to deficiency in inherent knowledge and complex reasoning. Traditional RAG methods (e.g, BM25, Qwen3-Emb) mitigate knowledge gap by incorporating external knowledge, which generally outperform pure LLMs. However, they still struggle with multi-hop QA due to their inability to capture complex structural relationships. Graph-enhanced methods (e.g., HippoRAG, GFM-RAG) show improved performance by effectively leveraging graph structures. However, these methods either incur significant API cost (e.g, HippoRAG) or rely on high-quality query-document labels to train the retriever (e.g., GFM-RAG). Moreover, these two-stage methods lead to misalignment between the retriever and generator, resulting in suboptimal QA performance as the retrieved information may not be perfectly optimized for the downstream reasoning.

UniGFM-RAG overcomes these limitations with a unified end-to-end framework that tightly integrates the GFM retriever and LLM generator, aligning their objectives to directly optimize for the final QA performance. By modeling complex relationships in the retrieved knowledge graph and ensuring that the retrieved information is highly relevant for reasoning, UniGFM-RAG significantly outperforms all baselines and achieves the state-of-the-art across all evaluated datasets.

TABLE III: QA reasoning performance comparison on multihop datasets.
<table><tr><td rowspan="2">Method</td><td colspan="2">HotpotQA</td><td colspan="2">MuSiQue</td><td colspan="2">2Wiki</td></tr><tr><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td colspan="7">Pure LLM Methods</td></tr><tr><td>GPT-4o-mini (ZS) [58]</td><td>28.6</td><td>41.0</td><td>11.2</td><td>36.3</td><td>30.2</td><td>36.3</td></tr><tr><td>Qwen3-8B (ZS) [2]</td><td>15.1</td><td>23.7</td><td>1.3</td><td>8.1</td><td>23.2</td><td>27.9</td></tr><tr><td>Qwen3-8B (FT) [2]</td><td>19.1</td><td>27.0</td><td>2.7</td><td>10.7</td><td>25.8</td><td>29.0</td></tr><tr><td colspan="7">Traditional RAG Methods</td></tr><tr><td>BM25 [59]</td><td>52.0</td><td>63.4</td><td>20.3</td><td>28.8</td><td>47.9</td><td>51.2</td></tr><tr><td>ColBERTv2 [60]</td><td>43.4</td><td>57.7</td><td>15.5</td><td>26.4</td><td>33.4</td><td>43.3</td></tr><tr><td>Qwen3-Emb [49]</td><td>46.9</td><td>60.4</td><td>24.0</td><td>35.8</td><td>49.5</td><td>57.1</td></tr><tr><td colspan="7">Graph-enhanced Methods</td></tr><tr><td>RAPTOR [61]</td><td>50.6</td><td>64.7</td><td>27.7</td><td>39.2</td><td>39.7</td><td>48.4</td></tr><tr><td>GraphRAG (MS) [15]</td><td>51.4</td><td>67.6</td><td>27.0</td><td>42.0</td><td>34.7</td><td>61.0</td></tr><tr><td>LightRAG [36]</td><td>9.9</td><td>20.2</td><td>2.0</td><td>9.3</td><td>2.5</td><td>12.1</td></tr><tr><td>HippoRAG [17]</td><td>46.3</td><td>60.0</td><td>24.0</td><td>35.9</td><td>59.4</td><td>67.3</td></tr><tr><td>SubgraphRAG [62]</td><td>44.5</td><td>57.0</td><td>25.1</td><td>35.7</td><td>62.7</td><td>69.0</td></tr><tr><td>G-retriever [37]</td><td>41.4</td><td>53.4</td><td>23.6</td><td>34.3</td><td>33.5</td><td>39.6</td></tr><tr><td>GFM-RAG [25]</td><td>46.3</td><td>60.2</td><td>23.5</td><td>34.8</td><td>60.0</td><td>66.8</td></tr><tr><td>UniGFM-RAG</td><td>65.6</td><td>77.2</td><td>31.1</td><td>44.0</td><td>69.7</td><td>73.9</td></tr></table>

TABLE IV: Retrieval performance comparison on multi-hop QA datasets. Recall@k (R@k) is used as the evaluation metric.
<table><tr><td rowspan="2">Method</td><td colspan="2">HotpotQA</td><td colspan="2">MuSiQue</td><td colspan="2">2Wiki</td></tr><tr><td>R@2</td><td>R@5</td><td>R@2</td><td>R@5</td><td>R@2</td><td>R@5</td></tr><tr><td colspan="7">Traditional RAG Methods</td></tr><tr><td>BM25 [59]</td><td>55.4</td><td>72.2</td><td>32.3</td><td>41.2</td><td>51.8</td><td>61.9</td></tr><tr><td>ColBERTv2 [60]</td><td>64.7</td><td>79.3</td><td>37.9</td><td>49.2</td><td>59.2</td><td>68.2</td></tr><tr><td>Qwen3-Emb [49]</td><td>67.5</td><td>80.0</td><td>41.9</td><td>55.7</td><td>63.6</td><td>70.0</td></tr><tr><td colspan="7">Graph-enhanced Methods</td></tr><tr><td>RAPTOR [61]</td><td>58.1</td><td>71.2</td><td>35.7</td><td>45.3</td><td>46.3</td><td>53.8</td></tr><tr><td>GraphRAG (MS) [15]</td><td>58.3</td><td>76.6</td><td>35.4</td><td>49.3</td><td>61.6</td><td>77.3</td></tr><tr><td>LightRAG [36]</td><td>38.8</td><td>54.7</td><td>24.8</td><td>34.7</td><td>45.1</td><td>59.1</td></tr><tr><td>HippoRAG [17]</td><td>60.1</td><td>78.5</td><td>41.2</td><td>53.2</td><td>68.4</td><td>87.0</td></tr><tr><td>SubgraphRAG [62]</td><td>57.6</td><td>71.7</td><td>37.7</td><td>47.6</td><td>70.2</td><td>85.3</td></tr><tr><td>G-retriever [37]</td><td>51.8</td><td>63.8</td><td>35.4</td><td>43.5</td><td>61.0</td><td>66.6</td></tr><tr><td>GFM-RAG [25]</td><td>56.1</td><td>76.4</td><td>37.0</td><td>51.2</td><td>63.5</td><td>82.4</td></tr><tr><td>UniGFM-RAG</td><td>68.7</td><td>92.4</td><td>45.5</td><td>67.3</td><td>65.7</td><td>81.7</td></tr></table>

Retrieval Results. As shown in Table IV, UniGFM-RAG achieves the best retrieval results on the MuSiQue and HotpotQA datasets, while maintaining highly competitive performance on the 2Wiki dataset. Although advanced embeddingbased methods (e.g., Qwen3-Emb) yield strong results by effectively capturing semantic similarity, they consistently fall short of graph-enhanced approaches when handling complex, multi-hop queries. Notably, despite being distilled from the Qwen3-Emb model, UniGFM-RAG significantly outperforms its semantic base model across all evaluated datasets. This substantial performance gap highlights a dual advantage in our approach. First, by utilizing the GFM to model topological relationships, UniGFM-RAG successfully overcomes the inherent limitations of the semantic-only retrieval, proving that structural awareness is essential for complex reasoning. Second, our framework addresses the limitations of existing graph-based retrievers (e.g., GFM-RAG). Although GFM-RAG also adopts a GFM, UniGFM-RAG still surpasses it by aligning the answer objectives directly into the retrieval phase. This demonstrates the superior efficacy of our endto-end training paradigm over disjointed two-stage pipelines, ensuring that the retrieved information are perfectly optimized for the downstream reasoning task without relying on any human labeled ground-truth.

TABLE V: Performance comparison on the G-bench dataset [56], [57]. QA reasoning is measured by Accuracy (ACC), and retrieval is measured by Recall.
<table><tr><td rowspan="2">Method</td><td colspan="3">G-bench (ACC)</td><td colspan="2">G-bench (Recall)</td></tr><tr><td>Novel</td><td>Medical</td><td>CS</td><td>Novel</td><td>Medical</td></tr><tr><td colspan="6">Pure LLM Methods</td></tr><tr><td>GPT-4o-mini (ZS) [58]</td><td>51.4</td><td>67.1</td><td>70.7</td><td></td><td></td></tr><tr><td>Qwen3-8B (ZS) [2]</td><td>31.2</td><td>38.5</td><td>63.5</td><td>1</td><td></td></tr><tr><td>Qwen3-8B (FT) [2]</td><td>42.2</td><td>39.4</td><td>24.9</td><td>-</td><td></td></tr><tr><td colspan="6">Traditional RAG Methods</td></tr><tr><td>BM25 [59]</td><td>56.5</td><td>68.7</td><td>71.7</td><td>82.1</td><td>87.9</td></tr><tr><td>ColBERTv2 [60]</td><td>56.2</td><td>71.8</td><td>71.9</td><td>82.4</td><td>89.5</td></tr><tr><td>Qwen3-Emb [49]</td><td>46.0</td><td>40.5</td><td>73.4</td><td>86.4</td><td>95.2</td></tr><tr><td colspan="6">Graph-enhanced Methods</td></tr><tr><td>RAPTOR [61]</td><td>43.2</td><td>57.1</td><td>73.6</td><td>66.1</td><td>84.2</td></tr><tr><td>GraphRAG (MS) [15]</td><td>50.9</td><td>45.2</td><td>72.5</td><td>67.4</td><td>56.4</td></tr><tr><td>LightRAG [36]</td><td>45.1</td><td>63.9</td><td>71.2</td><td>79.6</td><td>82.6</td></tr><tr><td>HippoRAG [17]</td><td>44.8</td><td>59.1</td><td>72.6</td><td>81.2</td><td>84.0</td></tr><tr><td>GFM-RAG [25]</td><td>56.7</td><td>71.9</td><td>72.6</td><td>83.5</td><td>89.6</td></tr><tr><td>UniGFM-RAG</td><td>60.0</td><td>72.9</td><td>73.9</td><td>87.8</td><td>92.6</td></tr></table>

## C. Generalization across Diverse Domains (RQ2)

To evaluate the generalizability of UniGFM-RAG, we conduct experiments on G-bench datasets in diverse domains, including novel, medical and computer science (CS). As shown in Table V, UniGFM-RAG consistently achieves toptier performance in most metrics, which demonstrate strong robustness and cross-domain adaptability. Such superior generalization stems from the effective fusion of Graph Foundation Models and Large Language Models, which jointly leverage transferable graph representation ability and powerful reasoning and generalization capabilities. This synergy allows UniGFM-RAG to perform reliably across distinct domains, which highlights its strong domain-agnostic nature.

## D. Ablation Study (RQ3)

In this section, we conduct ablation studies to validate the effectiveness of each core component in UniGFM-RAG: (1) the LLM generator objective $( L _ { 1 } )$ , (2) the semantic guided distillation loss $\left( L _ { 2 } \right)$ , and (3) graph foundation model (GFM) retriever to the overall performance. The detailed results are summarized in Figure 3.

From the table, the experimental results clearly demonstrate the necessity of each component in UniGFM-RAG. First, removing the generator objective (w/o $\mathcal { L } _ { 1 } )$ yields a noticeable performance drop across all datasets, which confirms its essential role in ensuring high-quality answer generation. Secondly, excluding the semantic-guided distillation loss (w/o $\mathcal { L } _ { 2 } )$ leads to severe degradation. This highlights that, in the absence of ground-truth document retrieval labels, our pseudolabel supervision is vital for effectively training GFM retriever. Crucially, we observe a catastrophic performance drop when removing the GFM retriever entirely (w/o GFM). This underscores the absolute necessity of reasoning capability over the graph-structured knowledge for resolving complex reasoning.

![](images/748e1b3de5e01c27dfd47c845ce6aaa02b71b2ab06e5e8a1f62165f03881ca75.jpg)  
Fig. 3: Ablation studies of UniGFM-RAG.

![](images/90357501aba73aa594337c412af5aed9eb7f66e27c862b0635ab8b804443278e.jpg)  
(a)

![](images/2e8398247305dd08780f3dd58b828cc43d28dac878e434961f2a8f8ee234cfd9.jpg)  
(b)  
Fig. 4: Parameter analysis on hyperparameter on k and λ.

## E. Parameter Analysis (RQ4)

To investigate the robustness of UniGFM-RAG, we analyze the impact of critical hyper-parameters, the number of the retrieved document $k \in \{ 2 , 5 , 8 , 1 0 , 1 5 , 2 0 \}$ and the distillation loss weight $\lambda ~ \in ~ \{ 0 . 0 0 1 , 0 . 0 1 , 0 . 1 , 1 . 0 , 1 0 \}$ . As illustrated in Figure 4(a), while overall performance improves as k increases, UniGFM-RAG already achieves highly competitive results at $k = 5$ . Beyond this point, the marginal performance gains diminish rapidly, whereas the computational overhead grows significantly. Consequently, to strike an optimal tradeoff between model effectiveness and computational efficiency, we adopt k = 5 as the default setting. Additionally, Figure 4(b) demonstrates that the framework remains largely stable regarding λ. UniGFM-RAG exhibits minor performance fluctuations under varying λ values. Based on these observations, we set $\lambda = 1 . 0$ as the optimal default setting for our experiments.

## F. Different Backbone Models (RQ5)

To verify the model-agnostic nature of UniGFM-RAG, we evaluate it with multiple LLM backbones of different families and parameter scales. As shown in Table VI, replacing the default generator with Qwen3-4B, Llama-3.1-8B, or Llama-3.2-3B still yields consistently strong performance across all three benchmarks. In particular, all backbone variants substantially outperform GFM-RAG on HotpotQA, and also achieve clear gains on the more challenging MuSiQue benchmark, indicating that the effectiveness of UniGFM-RAG is not tied to a specific generator. On 2Wiki, the performance remains highly competitive, with the lightweight Llama-3.2-3B variant even achieving the best EM score among all compared backbones. These results show that the gains of UniGFM-RAG mainly come from the framework itself, namely the tight integration of the GFM retriever and the LLM generator, rather than from relying on one particular backbone model.

TABLE VI: Different backbone models in UniGFM-RAG.
<table><tr><td rowspan="2">Variant</td><td colspan="2">HotpotQA</td><td colspan="2">MuSiQue</td><td colspan="2">2Wiki</td></tr><tr><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td>GFM-RAG</td><td>46.3</td><td>59.5</td><td>23.5</td><td>34.8</td><td>60.0</td><td>66.8</td></tr><tr><td colspan="7">UniGFM-RAG</td></tr><tr><td>Qwen3-8B</td><td>63.4</td><td>75.7</td><td>31.1</td><td>44.0</td><td>61.7</td><td>67.1</td></tr><tr><td>Qwen3-4B</td><td>61.2</td><td>73.2</td><td>26.7</td><td>37.5</td><td>59.3</td><td>64.67</td></tr><tr><td>Llama-3.1-8B</td><td>59.5</td><td>72.3</td><td>29.4</td><td>39.2</td><td>59.0</td><td>63.5</td></tr><tr><td>Llama-3.2-3B</td><td>58.0</td><td>70.8</td><td>28.3</td><td>39.9</td><td>62.4</td><td>67.0</td></tr></table>

## G. Efficiency Analysis (RQ6)

To evaluate the inference efficiency of UniGFM-RAG, we conduct experiments on G-bench (CS) to compare both inference cost (seconds per sample) and reasoning accuracy against various baseline methods. As shown in Figure 5, UniGFM-RAG achieves the highest reasoning accuracy while also incurring the lowest inference time per sample. This advantage mainly comes from its end-to-end graph-aware reasoning design. Existing graph-based RAG methods usually involve additional online reasoning steps. For example, ToG requires multi-round graph exploration and iterative LLM prompting, while HippoRAG performs multi-hop graph retrieval with Personalized PageRank, which also introduces iterative graph propagation during inference. GraphRAG and LightRAG also rely on external LLMs to summarize, reason over, or generate responses from retrieved graph contexts, which increases the online inference cost. In contrast, UniGFM-RAG performs retrieval and reasoning through a unified GFM, which avoids repeated external LLM invocations. Although graph encoding and message passing introduce additional computation, these operations are highly parallelizable on GPUs and are more efficient than multi-step LLM prompting during inference. Therefore, UniGFM-RAG achieves a favorable balance between effectiveness and efficiency for reasoning over graph-structured knowledge.

## H. Retrieval Case Study

To further illustrate the effectiveness of UniGFM-RAG, we present a qualitative retrieval case study. As shown in Table VII, both UniGFM-RAG and GFM-RAG successfully retrieve the initial relevant document for “Rip Curl Pro Portugal”, which mentions John John Florence. However, GFM-RAG fails to retrieve the subsequent relevant information (e.g., date of birth) for “John John Florence”, which directly derives the final answer. In contrast, UniGFM-RAG successfully identifies the critical supporting document for “John John Florence”. This demonstrates that, with the guidance of the LLM generator, our GFM retriever effectively navigates

TABLE VII: Case study comparing the retrieval and reasoning performance of UniGFM-RAG with GFM-RAG on the HotpotQA dataset.
<table><tr><td>Question Gold Answer Supporting Docs.</td><td colspan="2">What American professional Hawaiian surfer born 18 October 1992 won the Rip Curl Pro Portugal? John John Florence</td></tr><tr><td colspan="3">[“MEO Rip Curl Pro Portugal”, “John John Florence&quot;]</td></tr><tr><td></td><td>GFM-RAG (Baseline Method)</td><td></td></tr><tr><td>Retrieved Docs. Predicted Answer</td><td>[“MEO Rip Curl Pro Portugal&quot;, &quot;Reef Hawaiian Pro&quot;]</td><td>(Recall@2=0.5) X Incorrect</td></tr><tr><td colspan="3">document does not contain information about John John Florence</td></tr><tr><td>Retrieved Docs. Predicted Answer</td><td>UniGFM-RAG (Ours) [“MEO Rip Curl Pro Portugal&quot;, “John John Florence&quot;] John John Florence</td><td>(Recall@2=1.0) √ Correct</td></tr></table>

TABLE VIII: Case study comparing the retrieval and reasoning performance of UniGFM-RAG with GFM-RAG on the HotpotQA dataset.
<table><tr><td rowspan=1 colspan=1>QuestionGold AnswerSupporting Docs.</td><td rowspan=1 colspan=1>Which season did Brian Billick coach the Baltimore Ravens though in 2001?6th season[“2001 Baltimore Ravens season&quot;, “Brian Billick&quot;]</td></tr><tr><td rowspan=1 colspan=2>GFM-RAG (Baseline Method)</td></tr><tr><td rowspan=1 colspan=1>Retrieved Docs.Predicted Answer</td><td rowspan=1 colspan=1>[“2015 Baltimore Ravens season&quot;, “2011 Baltimore Ravens season&quot;, “Brian Billick&quot;, “2000 Baltimore Ravensseason&quot;, “2001 Baltimore Ravens season&quot;]                                               (Recall@5=1.0)2001 Baltimore Ravens season                                                             X Incorrect</td></tr><tr><td rowspan=1 colspan=2>UniGFM-RAG (Ours)</td></tr><tr><td rowspan=1 colspan=1>Retrieved Docs.Predicted Answer</td><td rowspan=1 colspan=1>[“2001 Baltimore Ravens season&quot;(0.42), “Brian Billick&quot;(0.38), “2000 Baltimore Ravens season&quot;(0.34),“2015 Baltimore Ravens season&quot; (0.19), “2001 St. Louis Rams season&quot; (0.03)]             (Recall@5=1.0)6th season                                                                                 √ Correct</td></tr></table>

![](images/b615a780199a1b5a4453f47496f5ecb9117a4bb779bfea41b4f8f1a39245f320.jpg)  
Fig. 5: Efficiency and performance comparison on G-bench (CS).  
complex graph structures to locate critical multi-hop evidence for complex reasoning.

## I. QA Case Study

Furthermore, we present a case study on answer generation to validate the critical impact of document ordering driven by our soft-weighting mechanism. As shown in Table VIII, both UniGFM-RAG and GFM-RAG successfully retrieve all supporting documents within their top-5 results (Recall@5=1.0). However, UniGFM-RAG places the relevant documents at the top of the ranked list, whereas GFM-RAG ranks them 3rd and 5th. Consequently, UniGFM-RAG derives the correct answer, while GFM-RAG fails, likely as it suffers from the wellknown “lost in the middle” phenomenon [63]. Ultimately, this demonstrates that introducing the soft-weighting mechanism explicitly reveals the importance of each retrieved document, guiding the generator to focus on the most critical evidence.

## V. CONCLUSION

In this paper, we present UniGFM-RAG, a unified, endto-end GraphRAG framework that seamlessly couples a graph foundation model (GFM) retriever with a large language model (LLM) generator. By leveraging GFM, we effectively reason over structural knowledge to conduct accurate document retrieval. Through the proposed graph bridge module, we connect the retriever and generator via a fully differentiable architecture. This integration fosters a reciprocal learning loop: the structural insights from GFM enhance the reasoning capabilities of LLM, while the gradient feedback from the LLM refines the graph retrieval process. Furthermore, by employing soft-weighting mechanism, we enable the LLM to dynamically focus on the most critical documents within the graph-augmented prompt. Extensive experiments on six complex reasoning benchmarks demonstrate that UniGFM-RAG consistently outperforms existing baseline methods, which establishes new state-of-the-art performance. Additional analyses confirm its exceptional inference efficiency and strong generalizability. Ultimately, UniGFM-RAG offers a promising paradigm for unifying GFMs and LLMs for knowledgeintensive applications. In future work, we plan to extend UniGFM-RAG in two directions. First, we will explore multimodal knowledge injection, which enables the topology-aware reasoning process to incorporate heterogeneous information beyond pure text. Second, we aim to adapt the unified endto-end framework to specialized and high-stakes domains that require both structural evidence and reliable generation.

## REFERENCES

[1] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.

[2] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.

[3] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan et al., “Deepseek-v3 technical report,” arXiv preprint arXiv:2412.19437, 2024.

[4] H. Ding, Z. Fan, I. Guehring, G. Gupta, W. Ha, J. Huan, L. Liu, B. Omidvar-Tehrani, S. Wang, and H. Zhou, “Reasoning and planning with large language models in code development,” in KDD, 2024, p. 6480–6490.

[5] B. Du, P. Li, S. Zhang, H. Liu, and Y. Yu, “Trust-aware human–robot collaborative decision-making with llm-generated data,” IEEE TCDS, vol. 18, no. 3, pp. 763–778, 2026.

[6] Z. Li, X. Chen, H. Yu, H. Lin, Y. Lu, Q. Tang, F. Huang, X. Han, L. Sun, and Y. Li, “Structrag: Boosting knowledge intensive reasoning of llms via inference-time hybrid information structurization,” in ICLR, 2025.

[7] S. Pan, L. Luo, Y. Wang, C. Chen, J. Wang, and X. Wu, “Unifying large language models and knowledge graphs: A roadmap,” IEEE TKDE, vol. 36, no. 7, pp. 3580–3599, 2024.

[8] X. Sun, H. Cheng, J. Li, B. Liu, and J. Guan, “All in one: Multi-task prompting for graph neural networks,” in KDD, 2023, pp. 2120–2131.

[9] L. Yang, J. Zheng, H. Wang, Z. Liu, Z. Huang, S. Hong, W. Zhang, and B. Cui, “Individual and structural graph information bottlenecks for out-of-distribution generalization,” IEEE TKDE, vol. 36, p. 682–693, 2024.

[10] L. Jiang, J. Ren, Z. Zhou, Y. Qu, Z. Zheng, H. Lu, and M. Wu, “Graph reinforcement learning-based reachability map for generalized mobile manipulation,” IEEE TCDS, vol. 18, no. 2, pp. 438–451, 2026.

[11] B. Peng, Y. Zhu, Y. Liu, X. Bo, H. Shi, C. Hong, Y. Zhang, and S. Tang, “Graph retrieval-augmented generation: A survey,” TOIS, vol. 44, no. 2, pp. 1–52, 2025.

[12] Q. Zhang, S. Chen, Y. Bei, Z. Yuan, H. Zhou, Z. Hong, H. Chen, Y. Xiao, C. Zhou, J. Dong et al., “A survey of graph retrievalaugmented generation for customized large language models,” arXiv preprint arXiv:2501.13958, 2025.

[13] Y. Huang, S. Zhang, and X. Xiao, “Ket-rag: A cost-efficient multigranular indexing framework for graph-rag,” in KDD, 2025, p. 1003–1012.

[14] Z. Zhao, L. Luo, J. Dong, H. Luo, X. Li, S. Pan, and C. Gong, “Hyperreco: Retrieving and connecting evidence with hypergraph neural networks for llm multi-hop reasoning,” arXiv preprint arXiv:2609.32327, 2026.

[15] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt, D. Metropolitansky, R. O. Ness, and J. Larson, “From local to global: A graph rag approach to query-focused summarization,” arXiv preprint arXiv:2404.16130, 2024.

[16] Y. Wang, N. Lipka, R. A. Rossi, A. Siu, R. Zhang, and T. Derr, “Knowledge graph prompting for multi-document question answering,” in AAAI, vol. 38, no. 17, 2024, pp. 19 206–19 214.

[17] B. Jimenez Gutierrez, Y. Shu, Y. Gu, M. Yasunaga, and Y. Su, “Hipporag: Neurobiologically inspired long-term memory for large language models,” NeurIPS, vol. 37, pp. 59 532–59 569, 2024.

[18] Y. Li, D. Song, Y. Tian, H. Wang, C. Zhou, and S. Zhang, “A framework of knowledge graph-enhanced large language model based on global planning,” IEEE TKDE, vol. 38, no. 2, pp. 736–748, 2025.

[19] S. Ma, C. Xu, X. Jiang, M. Li, H. Qu, C. Yang, J. Mao, and J. Guo, “Think-on-graph 2.0: Deep and faithful large language model reasoning with knowledge-guided retrieval augmented generation,” in ICLR, 2025.

[20] S. Wei, Z. Zhao, X. Feng, and H. Yu, “A cooperative multiagent reinforcement learning method based on reward decomposition for scalable agent environments,” IEEE TCDS, vol. 18, no. 2, pp. 398–410, 2026.

[21] H. Mao, Z. Chen, W. Tang, J. Zhao, Y. Ma, T. Zhao, N. Shah, M. Galkin, and J. Tang, “Position: Graph foundation models are already here,” in ICML, 2024.

[22] J. Liu, C. Yang, Z. Lu, J. Chen, Y. Li, M. Zhang, T. Bai, Y. Fang, L. Sun, P. S. Yu et al., “Graph foundation models: Concepts, opportunities and challenges,” IEEE TPAMI, vol. 47, no. 6, pp. 5023–5044, 2025.

[23] J. Yang, L. Yang, Z. Guo, J. Gao, J. Wu, T. Chai, H. Huang, C. Yang, and C. Shi, “Benchmarking graph foundation models,” in KDD, 2025, p. 5866–5875.

[24] Y. Fang, H. Miao, Y. Liang, L. Deng, Y. Cui, X. Zeng, Y. Xia, Y. Zhao, T. B. Pedersen, C. S. Jensen et al., “Unraveling spatio-temporal foundation models via the pipeline lens: A comprehensive review,” IEEE TKDE, vol. 38, pp. 2040–2063, 2026.

[25] L. Luo, Z. Zhao, G. Haffari, D. Phung, C. Gong, and S. Pan, “Gfmrag: Graph foundation model for retrieval augmented generation,” in NeurIPS, 2025.

[26] Y. Yu, W. Ping, Z. Liu, B. Wang, J. You, C. Zhang, M. Shoeybi, and B. Catanzaro, “Rankrag: Unifying context ranking with retrievalaugmented generation in llms,” NeurIPS, vol. 37, pp. 121 156–121 184, 2024.

[27] J. Dong, B. Fatemi, B. Perozzi, L. F. Yang, and A. Tsitsulin, “Don’t forget to connect! improving rag with graph-based reranking,” arXiv preprint arXiv:2405.18414, 2024.

[28] B. Jin, G. Liu, C. Han, M. Jiang, H. Ji, and J. Han, “Large language models on graphs: A comprehensive survey,” IEEE TKDE, vol. 36, no. 12, pp. 8622–8642, 2024.

[29] J. Tang, Y. Yang, W. Wei, L. Shi, L. Su, S. Cheng, D. Yin, and C. Huang, “Graphgpt: Graph instruction tuning for large language models,” in SIGIR, 2024, pp. 491–500.

[30] Z. Chai, T. Zhang, L. Wu, K. Han, X. Hu, X. Huang, and Y. Yang, “Graphllm: Boosting graph reasoning ability of large language model,” IEEE TBD, 2025.

[31] N. Chen, Y. Li, J. Tang, and J. Li, “Graphwiz: An instruction-following language model for graph computational problems,” in KDD, 2024, pp. 353–364.

[32] H. Qiao, C. Niu, L. Chen, and G. Pang, “Anomalygfm: Graph foundation model for zero/few-shot anomaly detection,” in KDD, 2025, pp. 2326– 2337.

[33] R. Chen, T. Zhao, A. Jaiswal, N. Shah, and Z. Wang, “Llaga: large language and graph assistant,” in ICML, 2024.

[34] L. LUO, Y.-F. Li, G. Haffari, and S. Pan, “Reasoning on graphs: Faithful and interpretable large language model reasoning,” in ICLR, 2024.

[35] L. Luo, Z. Zhao, G. Haffari, Y.-F. Li, C. Gong, and S. Pan, “Graphconstrained reasoning: Faithful reasoning on knowledge graphs with large language models,” in ICML, 2025.

[36] Z. Guo, L. Xia, Y. Yu, T. Ao, and C. Huang, “Lightrag: Simple and fast retrieval-augmented generation,” EMNLP, 2025.

[37] X. He, Y. Tian, Y. Sun, N. Chawla, T. Laurent, Y. LeCun, X. Bresson, and B. Hooi, “G-retriever: Retrieval-augmented generation for textual graph understanding and question answering,” NeurIPS, vol. 37, pp. 132 876–132 907, 2024.

[38] C. Mavromatis and G. Karypis, “Gnn-rag: Graph neural retrieval for efficient large language model reasoning on knowledge graphs,” in ACL, 2025, pp. 16 682–16 699.

[39] M. Yasunaga, H. Ren, A. Bosselut, P. Liang, and J. Leskovec, “Qa-gnn: Reasoning with language models and knowledge graphs for question answering,” in NAACL, 2021, pp. 535–546.

[40] X. Ren, J. Tang, D. Yin, N. Chawla, and C. Huang, “A survey of large language models for graphs,” in KDD, 2024, pp. 6616–6626.

[41] T. He, Z. Chen, L. Liao, Y. Cao, Y. Liu, W. Tang, X. Mao, K. Lv, M. Liu, and B. Qin, “Subgraph-centric multi-agent reinforcement learning for multi-hop knowledge graph reasoning,” IEEE TKDE, vol. 38, no. 2, pp. 1319–1333, 2025.

[42] Y. Li, Y. Wang, J. Tang, H. Chang, Y. Ren, and J. Li, “Advancing graph foundation models: A data-centric perspective,” in KDD, 2025, pp. 1635–1646.

[43] X. Jiang, T. Jia, Y. Fang, C. Shi, Z. Lin, and H. Wang, “Pre-training on large-scale heterogeneous graph,” in KDD, 2021, pp. 756–766.

[44] M. Galkin, X. Yuan, H. Mostafa, J. Tang, and Z. Zhu, “Towards foundation models for knowledge graph reasoning,” in ICLR, 2024.

[45] L. Xia, B. Kao, and C. Huang, “Opengraph: Towards open graph foundation models,” arXiv preprint arXiv:2403.01121, 2024.

[46] Z. Wang, Z. Zhang, N. V. Chawla, C. Zhang, and Y. Ye, “Gft: graph foundation model with transferable tree vocabulary,” in NeurIPS, 2024.

[47] J. Tang, Y. Yang, W. Wei, L. Shi, L. Xia, D. Yin, and C. Huang, “Higpt: Heterogeneous graph language model,” in KDD, 2024, pp. 2842–2853.

[48] M. Zhang, M. Sun, P. Wang, S. Fan, Y. Mo, X. Xu, H. Liu, C. Yang, and C. Shi, “Graphtranslator: Aligning graph model to large language model for open-ended tasks,” in WWW, 2024, pp. 1003–1014.

[49] Y. Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang, P. Xie, A. Yang, D. Liu, J. Lin et al., “Qwen3 embedding: Advancing text embedding and reranking through foundation models,” arXiv preprint arXiv:2506.05176, 2025.

[50] B. Yang, S. W.-t. Yih, X. He, J. Gao, and L. Deng, “Embedding entities and relations for learning and inference in knowledge bases,” in ICLR, 2015.

[51] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv preprint arXiv:1503.02531, 2015.

[52] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” in NAACL, 2019, pp. 4171–4186.

[53] Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. Cohen, R. Salakhutdinov, and C. D. Manning, “Hotpotqa: A dataset for diverse, explainable multi-hop question answering,” in EMNLP, 2018, pp. 2369–2380.

[54] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal, “Musique: Multihop questions via single-hop question composition,” TACL, vol. 10, pp. 539–554, 2022.

[55] X. Ho, A.-K. D. Nguyen, S. Sugawara, and A. Aizawa, “Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps,” in COLING, 2020, pp. 6609–6625.

[56] Z. Xiang, C. Wu, Q. Zhang, S. Chen, Z. Hong, X. Huang, and J. Su, “When to use graphs in rag: A comprehensive analysis for graph retrieval-augmented generation,” arXiv preprint arXiv:2506.05690, 2025.

[57] Y. Xiao, J. Dong, C. Zhou, S. Dong, Q.-w. Zhang, D. Yin, X. Sun, and X. Huang, “Graphrag-bench: Challenging domain-specific reasoning for evaluating graph retrieval-augmented generation,” arXiv preprint arXiv:2506.02404, 2025.

[58] OpenAI, “Hello gpt-4o,” 2024. [Online]. Available: https://openai.com/ index/hello-gpt-4o/

[59] S. E. Robertson and S. Walker, “Some simple effective approximations to the 2-poisson model for probabilistic weighted retrieval,” in SIGIR, 1994, pp. 232–241.

[60] K. Santhanam, O. Khattab, J. Saad-Falcon, C. Potts, and M. Zaharia, “Colbertv2: Effective and efficient retrieval via lightweight late interaction,” in NAACL, 2022, pp. 3715–3734.

[61] P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, and C. D. Manning, “Raptor: Recursive abstractive processing for tree-organized retrieval,” in ICLR, 2024.

[62] M. Li, S. Miao, and P. Li, “Simple is effective: The roles of graphs and large language models in knowledge-graph-based retrieval-augmented generation,” in ICLR, 2025.

[63] G. A. Baker, A. Raut, S. Shaier, L. E. Hunter, and K. von der Wense, “Lost in the middle, and in-between: Enhancing language models’ ability to reason over long contexts in multi-hop qa,” arXiv preprint arXiv:2412.10079, 2024.

VI. BIOGRAPHY SECTION

![](images/6fd535b2b071094c3c1d93307c591ab2a3735f3b1ee18781df7fbd5d9a908b3a.jpg)  
Zicheng Zhao is currently pursuing the Ph.D. degree in computer science and technology. He is currently a visiting student with the School of Automation and Intelligent Sensing, Shanghai Jiao Tong University. His research interests include knowledge graphs, large language models, and graph neural networks.

![](images/ef721bcb94b1a3337ce901a70fb777db6f2365de782e0b342680c2eb3e27becc.jpg)

Linhao Luo is a Postdoctoral Research Fellow with the Department of Data Science & AI at Monash University, Australia. He obtained his PhD degree in computer science at Monash University and received his B.S. degree from Harbin Institute of Technology, China. He has published over 30 research papers in top-tier journals and conferences, including the IEEE Transactions on Knowledge and Data Engineering (TKDE), Knowledge and Information Systems, Knowledge-Based Systems, ICLR, ICML, NeurIPS, ACL, EMNLP, AAAI, IJCAI, WSDM,

![](images/185f5396c8e3b4e27cf50f814a15a2e0fea1d179abed6679dbdee244bd82a20b.jpg)

Zongji Yu will receive the bachelor’s degree from the School of Control Science and Engineering, Shandong University in Jinan, China. Since 2026, he has been pursuing a doctoral degree at Shanghai Jiao Tong University in Shanghai, China. His research interests include continual learning, reinforcement learning, and AI safety.

![](images/70a6d86350780711a90d82ec61f835e668ce737f463e3691cacec4aba7aade83.jpg)

Junnan Dong is now a senior researcher at Tencent Youtu Lab. He obtained his Ph.D. at the Hong Kong Polytechnic University. He focuses on Large Language Models (LLMs), Diffusion Language Models (DLMs), Retrieval Augmented Generation (RAG) and Agents for advanced domain-specific reasoning. He has published more than 20 papers on top-tier conferences, including ICLR, ICML, NeurIPS, ACL, WWW, KDD, SIGIR, AAAI and WSDM. He was also nominated as the ’Best Research Postgraduate Student’ by both the Faculty and Department of

Computing at PolyU. He has been continuously serving as a program committee member for ICLR, NeurIPS, ICML, ICLR, ACL, AAAI and WWW, etc.

![](images/7a00e87ef18179d3b1b94049e7baf0db981e0b463fc7353f46eb77ad248d1d93.jpg)

Yanchao Tan (Member, IEEE) is currently an Associate Professor in the College of Computer and Data Science, Fujian Key Laboratory of Network Computing and Intelligent Information Processing, Fuzhou University. She obtained her PhD degree in the College of Computer Science, Zhejiang University, China. Her research interests include data mining, healthcare, graph mining, and recommender system.

![](images/176d317565fe7f3dde2250b1ba542c3116385f7e119d0b66fc2d813f26f273fd.jpg)

Shirui Pan (Senior Member, IEEE) received the PhD degree in computer science from the University of Technology Sydney (UTS), Ultimo, NSW, Australia. He is a professor with the School of Information and Communication Technology, Griffith University, Australia. His research interests include data mining and machine learning. His research received the 2024 CIS IEEE TNNLS Oustanding Paper Award and the 2020 IEEE ICDM Best Student Paper Award. He is recognized as one of the AI 2000 AAAI/IJCAI Most Influential Scholars in Australia.

He was the awardee for the 2024 IEEE ICDM Tao Li Award. He is an ARC Future fellow and a fellow of Queensland Academy of Arts and Sciences (FQA).

![](images/b6cf10d6229572dce85860f3833e09286eebd9a0e9a410496aff8ceca202e4ae.jpg)

Chen Gong (Senior Member, IEEE) received his dual doctoral degree from Shanghai Jiao Tong University (SJTU) and University of Technology Sydney (UTS), respectively. Currently, he is a full professor of Shanghai Jiao Tong University. His research interests mainly include machine learning, data mining, and learning-based vision problems. He has published more than 130 technical papers at prominent journals and conferences such as IEEE T-PAMI, JMLR, IJCV, IEEE T-NNLS, IEEE T-IP, ICML, NeurIPS, ICLR, CVPR, ICCV, ECCV,

AAAI, IJCAI, ICDM, etc. He serves as the associate editor for IEEE T-CSVT, Neural Networks, and NePL, and also the Area Chair or Senior PC member of several top-tier conferences such as AAAI, IJCAI, ICML, ICLR, ECML-PKDD, AISTATS, ICDM, ACM MM, etc. He won the ICDM Best Student Paper Runner-Up Award, the second prize of Natural Science Award of the Chinese Institute of Electronics, “Excellent Doctorial Dissertation Award” of Chinese Association for Artificial Intelligence, “Wu Wen-Jun AI Excellent Youth Scholar Award”, and the Scientific Fund for Distinguished Young Scholars of Jiangsu Province. He was also selected as the “Global Top Chinese Young Scholars in AI” released by Baidu, and “World’s Top 2% Scientists” released by Stanford University.

## VII. APPENDIX

## A. Statistics for Training Datasets

The training data contains 60 KGs with over 14M triples constructed from 700k documents extracted from the training set. The statistics are shown in Table IX.

TABLE IX: Statistics of the query-doc pairs and KGs used for training.
<table><tr><td>Dataset</td><td>#Q-doc Pair</td><td>#Document</td><td>#KG</td><td>#Entity</td><td>#Relation</td><td>#Triple</td></tr><tr><td>HotpotQA</td><td>20,000</td><td>204,822</td><td>20</td><td>1,930,362</td><td>967,218</td><td>6,393,342</td></tr><tr><td>MuSiQue</td><td>20,000</td><td>410,380</td><td>20</td><td>1,544,966</td><td>900,338</td><td>4,848,715</td></tr><tr><td>2Wiki</td><td>20,000</td><td>122,108</td><td>20</td><td>916,907</td><td>372,554</td><td>2,883,006</td></tr><tr><td>Total</td><td>60,000</td><td>737,310</td><td>60</td><td>4,392,235</td><td>2,240,110</td><td>14,125,063</td></tr></table>

## B. Detailed Descriptions of Baseline Methods.

To rigorously evaluate the effectiveness of UniGFM-RAG, we compare it with a diverse set of strong baseline methods, which are broadly categorized into three groups. (1) Pure LLM Methods: zero-shot (ZS) GPT-4o-mini [58] and Qwen3-8B [2] as well as a fine-tuned (FT) Qwen3-8B on the training data. (2) Traditional RAG Methods: BM25 [59], ColBERTv2 [60], and Qwen3-Emb (e.g., Qwen3-Embedding-0.6B) [49]. (3) Graphenhanced Methods: RAPTOR [61], GraphRAG (MS) [15], LightRAG [36], HippoRAG [17], SubgraphRAG [62], Gretriever [37], and GFM-RAG [25].

Pure LLM Methods directly employ the LLMs for answer generation without relying on external retrieval.

• GPT-4o-mini [58]: A strong proprietary LLM used in the zero-shot setting, where the model directly generates answers without external retrieval.

• Qwen3-8B [2]: An open-source LLM used in both zeroshot and fine-tuned settings to evaluate the capability of pure LLM-based answer generation.

Traditional RAG Methods represent traditional sparse and dense retrieval models that rely solely on lexical or semantic matching without structural awareness.

• BM25 [59]: A traditional probabilistic retrieval model that ranks documents according to exact term matching and term-frequency statistics.

• ColBERTv2 [60]: A dense retrieval model that improves retrieval quality and efficiency through late interaction, denoised training, and residual compression.

• Qwen3-Emb [49]: A dense embedding retriever based on Qwen3-Embedding-0.6B, which retrieves documents according to semantic similarity between the query and candidate documents.

Graph-enhanced Methods consist of recent approaches that incorporate structural knowledge or graph reasoning into the retrieval and generation pipeline.

• RAPTOR [61]: A hierarchical retrieval framework that recursively clusters, embeds, and summarizes text chunks at different levels to support multi-level retrieval.

• GraphRAG (MS) [15]: A graph-based RAG framework that constructs a corpus-level graph, applies hierarchical community detection, and generates community summaries to support LLM-based reasoning.

![](images/29aba203ada14ccd0a28e1963fe16ca9934228640a7f36d69278d580ade90dc2.jpg)  
Fig. 6: The graph-augmented prompt for LLM Reasoning.

• LightRAG [36]: A graph-enhanced RAG framework that incorporates graph structures into text indexing and performs dual-level retrieval to capture both fine-grained entity information and high-level relational knowledge.

• HippoRAG [17]: A training-free graph retriever that performs multi-hop retrieval over a document-based knowledge graph using Personalized PageRank.

• SubgraphRAG [62]: A retrieval framework that extracts dynamically sized subgraphs through lightweight scoring mechanisms, enabling efficient graph-based retrieval without task-specific fine-tuning.

• G-retriever [37]: A graph-based RAG method that formulates graph retrieval as a Prize-Collecting Steiner Tree optimization problem to extract relevant knowledge from large textual graphs beyond the context window of standard LLMs.

• GFM-RAG [25]: A graph foundation model based RAG framework that leverages structural reasoning and supports zero-shot retrieval on unseen datasets.

## C. Detailed Prompt for Training

To ensure reproducibility and clarify how the generator utilizes the retrieved graph-aware context during training, we provide the detailed LLM reasoning prompt in Fig. 6. The prompt contains the retrieved documents preceded by graph-aware prefixes, which encode the structural information produced by UniGFM-RAG. The LLM is then required to answer the question based only on the provided context and output the final answer without additional explanations. This design ensures that the generation process is conditioned on both textual evidence and graph-derived structural signals.

## D. Fine-grained Analysis by Reasoning Depth

To further evaluate the reasoning capability of UniGFM-RAG under different levels of complexity, we provide a fine-grained analysis by grouping questions according to their required reasoning depth, which include 2-hop, 3-hop, and 4-hop cases. We report both EM and F1 scores to examine whether UniGFM-RAG can maintain its advantage when more complex multi-hop reasoning paths are required.

![](images/a73ed5ebd4aac2def02b6dfbaaf55c52605327281abd7e01e5ecd006664cca8f.jpg)

![](images/e41d50ba1b1093fe032613c29175d74b9c7ede5a53a7c6b39311ed26df414868.jpg)  
Fig. 7: Fine-grained comparison of UniGFM-RAG and GFM-RAG under different reasoning depths. EM and F1 scores are reported for 2-hop, 3-hop, and 4-hop questions across datasets.

As shown in Figure 7, UniGFM-RAG consistently outperforms GFM-RAG across all datasets and reasoning depths. This demonstrates that the proposed unified framework is not only effective for relatively simple 2-hop questions, but also remains robust when the reasoning process involves longer and more complex evidence chains. More importantly, the advantage of UniGFM-RAG is still clear in high-depth scenarios. For example, on the challenging MuSiQue dataset, UniGFM-RAG substantially improves the F1 score over GFM-RAG on both 3-hop questions, from 28.0 to 41.5, and 4-hop questions, from 19.0 to 26.0. These results indicate that the end-to-end joint optimization in UniGFM-RAG helps reduce error propagation across reasoning steps and better captures task-relevant topological reasoning paths.

## E. Complex Reasoning Case Study

To further examine the multi-hop reasoning capability of UniGFM-RAG, we provide qualitative case studies on 3-hop and 4-hop questions from the challenging MuSiQue dataset. As shown in Tables X and XI, UniGFM-RAG exhibits a stronger ability to retrieve complete evidence chains than the decoupled baseline GFM-RAG, especially when the required reasoning path becomes longer and more complex.

For the 3-hop case in Table X, GFM-RAG suffers from evidence loss and retrieves only one of the three supporting documents, which results in a Recall@5 of $1 / 3$ and an incorrect prediction. In contrast, UniGFM-RAG successfully retrieves all three supporting documents and obtains a Recall@5 of 1.0, which provides sufficient evidence for predicting the correct answer, “Oriole Records”.

For the 4-hop case in Table XI, the reasoning challenge becomes more severe because the model needs to follow a longer evidence chain. GFM-RAG retrieves mostly irrelevant contexts and only recalls one of the four supporting documents, which leads to a Recall@5 of $1 / 4$ and an incorrect answer. By contrast, UniGFM-RAG accurately traces the complete four-step reasoning path, retrieves all required supporting documents with a Recall@5 of 1.0, and predicts the correct answer, “Paraguay”. These qualitative results show that the unified end-to-end design of UniGFM-RAG can better preserve evidence across extended topological reasoning paths, help reduce error propagation, and support more reliable answer generation in complex multi-hop scenarios.

## F. Token Cost for Graph Construction

Token consumption during knowledge graph construction is an important factor for evaluating the deployment cost of graph-enhanced RAG methods. Since UniGFM-RAG adopts the same knowledge graph construction pipeline as GFM-RAG, it does not introduce additional token consumption beyond this pipeline during the indexing stage. Table XII reports the token consumption required for index and knowledge graph construction per 1,000 documents.

As shown in the table, UniGFM-RAG requires approximately 4.8M tokens per 1,000 documents for graph construction, which is lower than LightRAG and GraphRAG. This lower construction cost makes UniGFM-RAG more suitable for large-scale graph-enhanced RAG scenarios, especially when the corpus size grows and the upfront indexing cost becomes a major deployment consideration.

## G. Knowledge Graph Construction Prompt

To improve the reproducibility of the knowledge graph construction process, we provide the OpenIE prompt used for extracting RDF-style triples in Fig. 8. UniGFM-RAG adopts the KG construction pipeline used in HippoRAG, where named entities are first extracted from each passage and then used to condition relation extraction. The prompt design follows Fig. 9 in the HippoRAG appendix [17].

## H. Discussion

This section further discusses two practical issues of UniGFM-RAG that are important for real-world deployment and robust training. We first analyze the scalability of UniGFM-RAG when applied to large-scale graph-structured knowledge sources, which focuses on parallel computation and subgraph-level reasoning. We then discuss the potential noise introduced by similarity-based weak supervision and explain how the proposed training framework can alleviate its negative impact through task-level feedback from the generation objective.

Scalability to Large-scale Graphs. We discuss the scalability of UniGFM-RAG to large-scale graphs. First, the graph encoding and message-passing operations in UniGFM-RAG are naturally parallelizable and can be partitioned across multiple GPUs. Distributed message-passing strategies, such as those adopted in G-reasoner, can also be incorporated to further scale the computation to larger graphs. Second, in practical RAG scenarios, it is often unnecessary to reason over the entire graph. UniGFM-RAG can be combined with subgraph extraction, where a query-relevant local subgraph is first retrieved and then processed by the graph foundation model. This strategy reduces memory consumption and inference cost while preserving task-relevant structural information. Therefore, although UniGFM-RAG introduces additional graph-oriented components, its parallelizable computation and compatibility with subgraph-level reasoning make it scalable to large graph-structured knowledge sources.

TABLE X: Case study comparing the retrieval and reasoning performance of UniGFM-RAG with GFM-RAG on the MuSiQue dataset (3-hop).
<table><tr><td>Question</td><td>What UK label was bought by the broadcast company that, along with ABC and the original network of Undercovers, is one of the major broadcasters based in New York?</td></tr><tr><td>Gold Answer Supporting Docs.</td><td>Oriole Records [“New York City_5”, “Undercovers (TV series)&quot;, “Sony Music_4&quot;]</td></tr><tr><td colspan="4">GFM-RAG (Baseline Method)</td></tr><tr><td>Retrieved Docs.</td><td colspan="2">[“Undercovers (TV series)&quot;, &quot;Neil Reid&quot;, &quot;999 ABC Broken Hill&quot;, &quot;Julie (TV series)&quot;, &quot;The Night Shift (season 3)&quot;]</td></tr><tr><td>Predicted Answer</td><td colspan="2">ITV</td></tr><tr><td colspan="3">UniGFM-RAG (Ours)</td></tr><tr><td>Retrieved Docs.</td><td colspan="2">[“Undercovers (TV series)&quot;, “New York City_5&quot;, “Sony Music_4&quot;, &quot;Roxy Recordings&quot;, &quot;New York Giants Radio Network&quot;] Oriole Records</td></tr><tr><td colspan="3">ABLE XI: Case study comparing the retrieval and reasoning performance of Uni GFM-RAG with GFM-RAG on the MuSiQ</td></tr><tr><td>Question</td><td colspan="2">A country&#x27;s military branch, the equivalent of which in the US helped the Old West&#x27;s local police, was unprepared for the invasion of the country that developed the ZH-29. The unprepared country was the only communist country to have an embassy where?</td></tr><tr><td>Gold Answer</td><td colspan="2">Paraguay [“Police”, “ZH-29”, “Josip Broz Tito_1”, “Josip Broz Tito_3”]</td></tr><tr><td colspan="3">Supporting Docs. GFM-RAG (Baseline Method)</td></tr><tr><td colspan="2">Retrieved Docs. [“International Women&#x27;s Day&quot;, “ZH-29&quot;, “Czech Republic&quot;, &quot;International Day of the Girl Child&quot;, “1973 oil</td></tr><tr><td>Predicted Answer</td><td>crisis_1&quot;] (recall@5=1/4) United States X Incorrect</td></tr><tr><td colspan="2">UniGFM-RAG (Ours)</td></tr><tr><td>Retrieved Docs. Predicted Answer</td><td>[ZH-29&quot;, “Josip Broz Tito_1&quot;, &quot;Warsaw Pact&quot;, &quot;Josip Broz Tito_3&quot;, &quot;Police&quot;] Paraguay</td></tr></table>

TABLE XII: Token consumption for index and knowledge graph construction per 1,000 documents.
<table><tr><td>Method</td><td>Token Consumption</td></tr><tr><td>UniGFM-RAG</td><td>4.8M</td></tr><tr><td>LightRAG</td><td>5.5M</td></tr><tr><td>GraphRAG</td><td>7.6M</td></tr></table>

Impact of Noisy Weak Supervision. UniGFM-RAG uses similarity-based signals as weak supervision because groundtruth supporting documents or reasoning paths are often unavailable in real-world multi-hop QA scenarios. Such weak supervision may inevitably contain noise, since semantically similar documents may not always be useful for answer generation. In our framework, the distillation loss $\mathcal { L } _ { 2 }$ transfers the similarity-based relevance signal to the retriever and improves the recall of potentially relevant evidence. The ablation results in Figure 3 show that removing $\mathcal { L } _ { 2 }$ leads to consistent performance degradation across datasets, which confirms the necessity of this weak relevance supervision. Meanwhile, the end-to-end training framework provides task-level feedback from the generation objective to adjust document importance. As shown in Eq. (16), gradients from the generation loss can be propagated to retrieved documents through both the softweighting and structural pathways, which helps the model emphasize answer-supporting documents and reduce the influence of irrelevant ones. These results suggest that UniGFM-RAG can benefit from similarity-based weak supervision while reducing the negative impact of noisy weak supervision through

![](images/d82a8d985c6c619da8eca83ad27db049123070c8c22b2017918f67735ddfc36c.jpg)  
Fig. 8: Open information extraction prompt for knowledge graph construction.

end-to-end optimization.