# QueryFlow: Multi-Hop Graph Retrieval with Adaptive Query Evolution

Manzong Huang Key Laboratory of Knowledge   
Engineering with Big Data (the   
Ministry of Education of China)   
School of Computer Science and Information Engineering   
Hefei University of Technology Hefei, China   
manzonghuang@mail.hfut.edu.cn Chenyang Bu<sup>∗</sup>   
Key Laboratory of Knowledge   
Engineering with Big Data (the   
Ministry of Education of China)   
School of Computer Science and Information Engineering   
Hefei University of Technology Hefei, China chenyangbu@hfut.edu.cn   
Jiapu Wang   
Nanjing University of Science and   
Technology   
Nanjing, China   
jiapu.wang@njust.edu.cn Yi He   
Department of Data Science   
College of William and Mary   
Williamsburg, Virginia, USA yihe@wm.edu Xindong Wu<sup>∗</sup>   
Key Laboratory of Knowledge   
Engineering with Big Data (the   
Ministry of Education of China)   
School of Computer Science and Information Engineering   
Hefei University of Technology Hefei, China xwu@hfut.edu.cn

## Abstract

Multi-hop retrieval-augmented generation (RAG) supports Web information seeking and fact verification by recovering complementary evidence across related pages within a limited retrieval budget. Scoring evidence independently against the original question can miss supporting units whose relevance emerges through intermediate facts. Iterative retrieval incorporates these facts into subsequent queries, while graph retrieval explicitly models relations among evidence units. Yet fixed-query graph retrievers update entity representations without adapting the query condition to accumulated relational context. We view multi-hop evidence recovery as the coupled evolution of evidence representations and query conditions, with accumulated context informing both what the retriever represents and how it searches for complementary evidence. QueryFlow realizes this perspective by recurrently integrating query-conditioned summaries of graph context into a shared query state, which controls boundary inputs and relation features for subsequent propagation. Across four representative multi-hop question answering (QA) datasets, QueryFlow achieves macro Recall@10 (R@10), Full Recall@10 (FR@10), and LLM-Acc gains of 4.98, 8.40, and 8.0 percentage points over representative baselines, respectively.

## CCS Concepts

• Information systems → Information retrieval; • Computing methodologies → Knowledge representation and reasoning.

Keywords GraphRAG, graph neural networks, graph-based retrieval, multihop question answering, web content mining

ACM Reference Format: Manzong Huang, Chenyang Bu, Yi He, Jiapu Wang, and Xindong Wu. 2027.   
QueryFlow: Multi-Hop Graph Retrieval with Adaptive Query Evolution.   
In Proceedings of The ACM Web Conference 2027 (WWW 2027 submission).   
ACM, New York, NY, USA, 14 pages.

## 1 Introduction

Retrieval-augmented generation (RAG) grounds language-model answers in external evidence [28]. In Web content mining, it supports complex information seeking and fact verification across related pages [23, 46]. Multi-hop questions make this evidence requirement more demanding because their answers depend on linked facts distributed across pages about related entities [16, 47, 58]. Answering these questions requires retrieving evidence units, such as documents, passages, or chunks, that jointly support the necessary reasoning steps. A collection of individually relevant units can still omit a fact needed to complete the reasoning chain. Multihop retrieval is fundamentally an evidence-set recovery problem, with the goal of recovering the complete supporting set within a given retrieval budget. By improving multi-hop question answering over interconnected Web content, our work aligns with the Web Mining, Multimedia and Multilingual Content Analysis track.

Recovering a complete supporting set requires more than matching individual evidence units to the original question. Single-round dense retrievers learn semantic matches between questions and candidate evidence [25], but independent scoring can miss complementary units whose relevance becomes apparent only after intermediate facts are found. Iterative retrieval addresses this limitation by conditioning subsequent searches on retrieved passages or intermediate reasoning [48, 56]. In these approaches, evidence dependencies are conveyed through retrieval history, leaving en tity relations across documents implicit. Graph-based RAG makes these connections explicit through entities and relations [7, 61], while neural graph retrievers learn to integrate connected evidence through query-conditioned message passing [31, 35]. A central question in learned graph retrieval is how accumulated relational context should guide subsequent evidence acquisition. Fixed-query designs update entity representations while retaining the original query condition [35]. As illustrated in Figure 1, identifying the author of The Hobbit shifts the remaining information need toward that person’s birthplace, while a fixed query can continue to favor book-related evidence. These designs accumulate relational context in entity representations without an explicit feedback path for that context to revise the query condition governing subsequent propagation.

![](images/e02f8d13badc4ce93b5acbaafcfd3543e5e12ac8b1b2822ac3e047a5197b0034.jpg)  
Figure 1: A fixed query state continues to favor evidence directly related to the original question. Updating the state with accumulated context guides retrieval toward complementary evidence, helping complete the support set. �<sub>1</sub> illustrates the updated state’s meaning.

The query state should evolve with the evidence representations it guides. Intermediate entities and relations can clarify what remains to be retrieved, so accumulated graph context should inform both the representation of evidence and the condition used to explore further evidence. Coupling these updates brings relational propagation and query adaptation into the same retrieval process, allowing newly established context to guide the recovery of missing support.

We introduce QueryFlow, a graph retrieval framework that couples recurrent query evolution with evidence propagation to support multi-hop question answering. It maintains a shared query state alongside entity representations that accumulate relational context through message passing. Between successive propagation layers, query-conditioned attention summarizes all entity representations, making context distributed across the graph available to the query update. A gated recurrent transition integrates this summary with the preceding query state to revise the retrieval condition. Boundary feedback injects the updated state at question-linked entities, while relation feedback modulates the features used to transform neighboring messages. These interfaces change both the context injected into the graph and how evidence propagates across its relations, closing the feedback path from accumulated evidence to subsequent retrieval. Finally, entity scores are aggregated into a document ranking that supplies evidence for answer generation.

On four representative multi-hop QA datasets, QueryFlow achieves macro R@10, FR@10, and LLM-Acc gains of 4.98, 8.40, and 8.0 percentage points over representative baselines, respectively. Improvements in evidence coverage and complete recovery are accompanied by higher downstream QA reasoning accuracy, supporting the value of complete evidence retrieval for multi-hop question answering.

Our contributions are threefold.

• We frame multi-hop graph retrieval as coupled evolution of evidence representations and query conditions, treating accumulated relational context as feedback for subsequent evidence acquisition.

• We propose QueryFlow, which couples recurrent query updates with graph propagation through boundary and relation feedback.

• Experiments on four multi-hop QA benchmarks show gains over representative baselines in evidence recovery and answer accuracy.

## 2 Related Work

Relevance-based retrieval. RAG integrates external evidence into language-model training or generation [3, 14, 21, 28]. Dense representations and token interactions improve matching between questions and evidence [19, 25, 27, 39]. Precomputed passage representations support eficient search, while token-level interactions retain fine-grained matching signals within each candidate passage. Better query representations, evidence fusion, and hierarchical indexing further improve evidence access and use [10, 20, 40, 42]. In single-round matching, retrieved evidence does not update the scoring condition, so complementary support with weak direct relevance can be missed. QueryFlow complements relevance scoring with feedback from accumulated relational context.

Iterative and adaptive retrieval. Iterative retrievers use retrieved passages, condensed facts, or rewritten queries to guide later searches [36, 38, 59]. MDR re-encodes the question with earlier passages [56], while Baleen condenses retrieved passages into facts for later hops [26]. Intermediate generation and reasoning also supply feedback [41, 48], while adaptive approaches control when and how to retrieve based on generation state or question dificulty [2, 22, 24, 43]. In retrieval loops driven by passages or reasoning, feedback passes through text history, leaving entity relations across documents implicit. QueryFlow brings this adaptation into graph propagation, using intermediate entity context to update the shared retrieval condition.

Graph-based retrieval. Knowledge graph research also studies graph completion, which predicts missing entities or relations from graph structure. Completion methods learn multi-concept and fine-grained entity–relation representations [52, 53], or adapt how entity, relation, and temporal information is combined to predict missing facts [49, 54, 55]. Broader graph representation learning combines local and global signals for node classification, while attributed graph clustering uses collaborative graph augmentation and contrastive learning [51, 60]. These tasks learn graph representations or predict missing facts, whereas graph retrieval ranks source documents to recover evidence supporting a question. Graph indices and difusion associate evidence through explicit links [7, 11– 13, 61], while subgraph selection and path exploration construct structured context [15, 17, 30, 33, 34, 45]. Learned relational reasoning supports query-conditioned propagation [8, 50, 62, 63, 67], and neural retrievers learn to aggregate or select connected evidence [31, 37, 44]. Learned graph retrievers difer in how relational context conditions retrieval. GFM-RAG propagates entity relevance under a fixed query and maps it to documents [35], whereas D-RAG updates instructions from graph context for knowledge graph question answering [9, 65, 66]. QueryFlow studies this interaction for complete document evidence recovery, using a shared evolving query state to control boundary inputs and relation features.

## 3 Method

Overview. QueryFlow organizes multi-hop evidence retrieval as a feedback process between entity representations and a shared query state (Figure 2). Entity propagation aggregates relational context, while recurrent query evolution converts that context into the condition for subsequent propagation. The evolving query state controls both boundary inputs and relation features, allowing accumulated context to influence how further evidence is explored. Final entity scores are aggregated into a document ranking for answer generation.

## 3.1 Evidence Graph and Query Grounding

Retrieval task and graph interface. Given a question � and a document corpus $D ,$ the objective is to recover the set of supporting documents $D ^ { \ast } ( x ) \subseteq D$ within a retrieval budget �. The first � documents in the output ranking form $\widehat { D } _ { k } ( x )$ . We represent the corpus by an entity graph $G = \left( V , E , \mathcal { R } \right) \left[ 3 5 \right]$ , where $V$ contains entities and � contains edges labeled by relation types in R. An incidence matrix $A \in \{ 0 , 1 \} ^ { | \check { V } | \times | D | }$ records associations between entities and documents, with $A _ { i j } = 1$ when entity � occurs in document �. Message passing accumulates context at entity nodes; the incidence matrix subsequently transfers entity relevance to documents.

Query and entity initialization. A frozen text encoder produces question and relation embeddings $e _ { x }$ and $e _ { r } .$ . Learned linear projections $\phi _ { q }$ and $\phi _ { r }$ map them to a common �-dimensional space. Let $m _ { x } \in \{ 0 , 1 \} ^ { | V | }$ identify the entities linked to �. The initial query, relation features, and entity states are

$$
\begin{array} { r l r } { q _ { 0 } = \phi _ { q } ( e _ { x } ) , } & { { } } & { R ^ { 0 } ( r ) = \phi _ { r } ( e _ { r } ) , } \\ { { } } & { { } } & { H _ { 0 } ( i ) = m _ { x } ( i ) q _ { 0 } . } \end{array}\tag{1}
$$

Thus entities linked to the question start from $q _ { 0 } ,$ while other entities start from zero. At layer �, the shared query state $q _ { l - 1 } \in \mathbb { R } ^ { d }$ controls the update to entity representations $H _ { l } \in \mathbb { R } ^ { | V | \times \bar { d } }$ . The entity representations store context specific to each entity, while the query supplies a shared condition for propagation. Graph topology, document associations, and the grounding mask remain fixed as these representations evolve.

## 3.2 Query-Conditioned Evidence Propagation

Each layer propagates entity context using boundary inputs and relational messages [63]. The shared query state $q _ { l - 1 }$ controls both the representations injected at question-linked entities and the relation features that transform neighboring messages.

Boundary feedback. The boundary input at the original grounding locations is

$$
B _ { l } ( i ) = m _ { x } ( i ) q _ { l - 1 } .\tag{2}
$$

Its support is fixed by $m _ { x } ,$ while its content changes with the query state. At later layers, this input reintroduces the context encoded by earlier query updates, maintaining a connection between question grounding and evolving relational information.

Relation feedback. The same state generates a gate for each feature of the base relation embeddings,

$$
\begin{array} { r } { \gamma _ { l - 1 } = \sigma ( W _ { r } q _ { l - 1 } + b _ { r } ) , } \\ { \bar { R } _ { l } ( r ) = \psi _ { l } \bigl ( R ^ { 0 } ( r ) \odot \gamma _ { l - 1 } \bigr ) . } \end{array}\tag{3}
$$

Here � is the sigmoid function, ⊙ denotes elementwise multiplication, and �<sub>�</sub> is a layer-specific two-layer MLP with an intervening ReLU. The gate is shared across relation types, each retaining its own base embedding $R ^ { 0 } ( r )$ . It modulates dimensions of the relation features according to the current query condition before the layer-specific transformation.

Relational message passing. Let $N ( i )$ contain pairs of source entities and relations $( j , r )$ with an edge from � to �. DistMult messages [57] and sum aggregation combine neighboring states with the boundary to give

$$
u _ { l , i } = B _ { l } ( i ) + \sum _ { ( j , r ) \in N ( i ) } H _ { l - 1 } ( j ) \odot \bar { R } _ { l } ( r ) .\tag{4}
$$

The next entity representation is obtained by transforming the concatenated previous state and aggregate, followed by a residual connection:

$$
\begin{array} { r } { \begin{array} { c } { t _ { l , i } = W _ { l } [ H _ { l - 1 } ( i ) ; u _ { l , i } ] + c _ { l } , } \\ { H _ { l } ( i ) = H _ { l - 1 } ( i ) + \mathrm { R e L U } ( \mathrm { L N } _ { l } ( t _ { l , i } ) ) . } \end{array} } \end{array}\tag{5}
$$

Here [ ; ] denotes concatenation, and $\mathrm { L N } _ { l }$ is layer-specific normalization. Boundary feedback changes the representation injected at the starting entities, while relation feedback changes the features used to carry messages along edges. For $l < L ,$ , �<sub>�</sub> supplies the next layer’s node input and the context for query evolution.

## 3.3 Graph-Conditioned Query Evolution

Query evolution summarizes $H _ { l }$ to update the shared state �<sub>�</sub>. For $l < L ,$ , this state conditions the next propagation layer. The readout ablation additionally applies the same update at $l = L \mathrm { t o } \left( H _ { L } , q _ { L - 1 } \right)$ obtaining the terminal state $q _ { L }$ without further graph propagation.

Query-conditioned graph summary. We use the preceding query state to attend to the new entity representations with weights

$$
a _ { l , i } = \frac { \exp \left( H _ { l } ( i ) ^ { \top } q _ { l - 1 } / \sqrt { d } \right) } { \sum _ { j \in V } \exp \left( H _ { l } ( j ) ^ { \top } q _ { l - 1 } / \sqrt { d } \right) } .\tag{6}
$$

The normalized graph summary is

$$
z _ { l } = \mathrm { L N } _ { \mathrm { s u m } } \left( \sum _ { i \in V } a _ { l , i } H _ { l } ( i ) \right) .\tag{7}
$$

![](images/812c37699dd371d66335152e93b91c6a62874e9484a1ce5b5be36254ad627966.jpg)  
Figure 2: Overview of QueryFlow for multi-hop evidence retrieval from interconnected document collections. A question is grounded in an entity graph whose nodes are linked to source documents. Graph propagation and recurrent query updates alternate across layers; the updated query state conditions the boundary inputs and relation features of the next layer. Final entity representations are scored together with the original query, and the scores of all entities are transformed by softplus and aggregated through associations between entities and documents using reciprocal document frequency weights. The documents with the highest scores provide evidence for answer generation. The terminal $q _ { L }$ update is used only for the readout ablation.

The softmax ranges over all entities in � . Consequently, �<sub>�</sub> depends jointly on the current query and the graph representations, emphasizing context compatible with the retrieval condition. The summary provides an input of fixed width to the controller regardless of graph size.

Gated recurrent transition. A gated recurrent unit (GRU) [6] integrates $z _ { l }$ with the preceding query state. An additional gate regulates the displacement in each feature dimension toward the candidate:

$$
\begin{array} { r l } & { \widetilde { q } _ { l } = \mathrm { G R U C e l l } ( z _ { l } , q _ { l - 1 } ) , } \\ & { { } g _ { l } = \sigma \big ( W _ { g } [ q _ { l - 1 } ; z _ { l } ] + b _ { g } \big ) , } \\ & { \Delta _ { l } = g _ { l } \odot ( \widetilde { q } _ { l } - q _ { l - 1 } ) . } \end{array}\tag{8}
$$

The candidate $\widetilde { q _ { l } }$ combines previous query information with new graph context, while �<sub>�</sub> controls how much of their diference enters the update. $\mathrm { A }$ scalar $\alpha \in [ 0 , 1 ]$ scales this increment before query normalization:

$$
q _ { l } = \left\{ \begin{array} { l l } { q _ { l - 1 } , } & { \alpha = 0 , } \\ { \mathrm { L N } _ { q } ( q _ { l - 1 } + \alpha \Delta _ { l } ) , } & { 0 < \alpha \leq 1 . } \end{array} \right.\tag{9}
$$

We use $\alpha = 1$ for recurrent propagation. Setting $\alpha = 0$ retains $q _ { l } =$ �<sub>0</sub> by bypassing the transition and its normalization, providing a fixed-query intervention within the same model. The recurrent cell, transition gate, and relation gate share parameters across layers; $\mathrm { L N } _ { \mathrm { s u m } }$ and $\mathrm { L N } _ { q }$ have separate parameters, each shared across layers. For $l < L$ , the updated state controls the next boundary and relation features through Eqs. (2) and (3), completing the feedback from graph context to subsequent propagation.

## 3.4 Document Readout and Learning

Entity relevance scoring. After � propagation layers, a shared two-layer MLP with an intervening ReLU scores each entity using

its final representation and the initial query:

$$
s _ { i } = \mathrm { M L P } \big ( \left[ H _ { L } ( i ) ; q _ { 0 } \right] \big ) .\tag{10}
$$

The default model uses $q _ { 0 }$ in both training and retrieval to anchor scores to the original question. The readout ablation replaces $q _ { 0 }$ with $q _ { L }$ in $\operatorname { E q } .$ (10), keeping $H _ { L }$ unchanged. The entity-level objective trains the query controller through its influence on subsequent graph propagation.

Entity-to-document readout. We convert each entity logit $s _ { i }$ into a positive contribution weight using

$$
v _ { i } = { \mathrm { s o f t p l u s } } ( s _ { i } ) = \log ( 1 + \exp ( s _ { i } ) ) .\tag{11}
$$

Let $\begin{array} { r } { f _ { i } = \sum _ { j = 1 } ^ { | D | } A _ { i j } } \end{array}$ denote the document frequency of entity �. We assign reciprocal frequency weights [35],

$$
\begin{array} { r l r } & { } & { w _ { i } ^ { \mathrm { d o c } } = \left\{ \begin{array} { l l } { f _ { i } ^ { - 1 } , } & { f _ { i } > 0 , } \\ { 0 , } & { f _ { i } = 0 , } \end{array} \right. } \\ & { } & { d _ { j } = \displaystyle \sum _ { i \in V } A _ { i j } \ w _ { i } ^ { \mathrm { d o c } } v _ { i } . } \end{array}\tag{12}
$$

The readout aggregates contributions from all entities associated with each document. For entities that occur in many documents, reciprocal frequency weighting reduces their contribution to each document. We rank documents by $d _ { j }$ and return the top � as $\widehat { D } _ { k } ( x )$

Entity-level learning. Let $y _ { i } \in \{ 0 , 1 \}$ be an entity supporting label and $P = \{ i : y _ { i } = 1 \}$ the positive set. With $p _ { i } = \sigma ( s _ { i } )$ , the listwise loss that assigns normalized score mass to each supporting entity is

$$
\begin{array} { c } { \displaystyle \pi _ { i } = \frac { \displaystyle p _ { i } } { \displaystyle \sum _ { j \in V } p _ { j } + \varepsilon } , } \\ { \displaystyle \mathcal { L } _ { \mathrm { L i s t C E } } = - \frac { \displaystyle 1 } { \displaystyle | P | } \displaystyle \sum _ { i \in P } \log ( \pi _ { i } + \varepsilon ) . } \end{array}\tag{13}
$$

Here � is a numerical stabilizer. We combine the listwise objective with a weighted binary cross-entropy term that classifies individual entities and emphasizes negatives with high scores to obtain

$$
\mathcal { L } _ { x } = 0 . 3 \mathcal { L } _ { \mathrm { B C E } } + 0 . 7 \mathcal { L } _ { \mathrm { L i s t C E } } .\tag{14}
$$

Gradients from entity scores jointly train graph propagation and the recurrent controller. Aggregation ofentity scores into document scores and document ranking are applied at inference. Appendix B details the training objective and retrieval algorithm.

## 4 Experiments

Our experiments connect answer quality to the recovery of supporting evidence. We first compare generation and retrieval performance, then examine query evolution and its feedback interfaces through controlled interventions. Eficiency, retrieval budgets, and paired recovery analyses characterize the costs and conditions under which these gains occur.

## 4.1 Experimental Setup

Datasets. We evaluate on four multi-hop QA benchmarks, namely 2WikiMultiHopQA [16], HotpotQA [58], MuSiQue [47], and ConcurrentQA [1]. Their annotated supporting documents define the target evidence sets. Dataset and evaluation details are provided in Appendix A.1. Model and training settings appear in Appendix A.2.

Evaluation metrics. For answer generation, we report LLM-Acc, the fraction ofanswers accepted by an LLM judge, and Contain, the fraction of generated answers containing the normalized gold answer string. For retrieval budget �, document recall and full recall are

$$
\begin{array} { r l r } & { } & { \mathrm { R } @ { k ( x ) } = \frac { \displaystyle \left. \widehat { D } _ { k } ( x ) \cap D ^ { * } ( x ) \right. } { \displaystyle \left. D ^ { * } ( x ) \right. } , } \\ & { } & { \mathrm { F R } @ { k ( x ) } = \mathbb { I } [ D ^ { * } ( x ) \subseteq \widehat { D } _ { k } ( x ) ] , } \end{array}\tag{15}
$$

where I is the indicator function. Both metrics are averaged over questions with nonempty support sets; FR@� measures complete recovery of the supporting set. We report results for each dataset and an unweighted macro average at � = 10 and � = 30 (Tables 2 and 7).

Retrieval comparisons. We compare R@� and FR@� only for methods whose final retrieval output is a ranked list of documents. Methods returning only graph structures (e.g., ToG [45]) or text chunks without a document ranking are evaluated through downstream answer quality instead.

Baselines. Our retrieval baselines cover hierarchical retrieval with RAPTOR [40], query expansion with HyDE [10], and lateinteraction retrieval with ColBERTv2 [39]. The six graph retrieval baselines are G-Retriever [15], LightRAG [11], GraphRAG [7], DALK [29], HippoRAG 2 [13], and GFM-RAG [35]. Appendix A provides retrieval results for each dataset.

Answer generation. We use Qwen3.5-9B for answer generation across all methods. Following LinearRAG [64], we report LLM-Acc and Contain using the definitions above. Table 1 compares the retrieval methods used to supply evidence, including RAPTOR, HyDE, ColBERTv2, G-Retriever, LightRAG, GraphRAG, DALK, GFM-RAG, LinearRAG, Dense RAG, FastToG [32], Think-on-Graph (ToG) [45], PathRAG [5], and HippoRAG 2 [13]. The table groups these methods by retrieval mechanism.

## 4.2 Main Results

Table 1 compares downstream answer quality across query-based, iterative and structure-aware, and graph-based retrieval methods on four benchmarks. These results address the following research question.

RQ1. How does QueryFlow compare with existing retrieval and reasoning methods on downstream answer generation?

Answer quality improves over representative baselines. QueryFlow reaches macro LLM-Acc of 0.696 and Contain of 0.642. Relative to LinearRAG, these scores increase by 8.0 and 6.9 percentage points, respectively. The gains over GFM-RAG are 11.6 and 5.6 points. Both comparisons show improvements in judged correctness and inclusion of the gold answer.

The answer gains extend across all four benchmarks. Compared with LinearRAG, LLM-Acc increases by 15.0 points on 2Wiki-MultiHopQA, 3.4 on HotpotQA, and 7.0 on both MuSiQue and ConcurrentQA. On MuSiQue, LLM-Acc rises from 0.369 to 0.439 while Contain improves from 0.332 to 0.375. QueryFlow improves both metrics on every dataset relative to LinearRAG and GFM-RAG, showing that the macro gains extend across all four collections.

Semantic correctness and answer inclusion improve together. Contain gains range from 4.3 to 12.5 points over LinearRAG and from 3.8 to 7.5 points over GFM-RAG. For multi-hop questions, intermediate facts identify the entity or relation to which the final answer must refer. QueryFlow feeds this context into subsequent propagation to guide retrieval of complementary support. The gains in complete recovery examined in RQ2 are consistent with this explanation of the answer gains.

To examine the evidence available for answer generation, we compare coverage of supporting documents and complete recovery of evidence sets at $k = 1 0$ and � = 30. Table 2 and Figure 3 summarize the results for the following question.

RQ2. How does QueryFlow compare with existing retrieval methods on multi-hop evidence retrieval benchmarks?

Larger gains in complete recovery. QueryFlow improves macro FR@10 and FR@30 over GFM-RAG by 8.40 and 9.63 percentage points, reaching 70.38% and 77.78%, respectively (Table 2). Support coverage also improves, with R@10 of 81.93% and R@30 of 87.12%, gains of4.98 and 5.70 points. The larger margins in full recall locate the main benefit in recovering every document required by a question. Retrieving the last missing supporting document turns partial coverage into complete support, so this objective favors complementary evidence. QueryFlow lets intermediate entity context revise the condition for later propagation, providing a mechanism for directing retrieval toward such evidence.

Consistent gains across datasets. Across datasets at � = 10, R@10 gains over GFM-RAG range from 4.57 to 5.28 points. The FR@10 gains are 12.60 points on 2WikiMultiHopQA, 8.11 on HotpotQA, 5.99 on MuSiQue, and 6.90 on ConcurrentQA (Table 7). Each dataset exhibits a larger gain in full recall than in average support coverage. On MuSiQue, the margin in full recall grows from 5.99 points at $k = 1 0$ to 8.88 at $k = 3 0$ , extending the improvement to the larger retrieval budget.

Complete recovery distinguishes complementary support from average coverage. On MuSiQue, HippoRAG 2 has higher R@10 than QueryFlow (64.42% versus 61.60%), yet QueryFlow has higher FR@10 (37.82% versus 31.11%). The same ordering holds at � = 30. Average coverage rewards each recovered document, whereas full recall requires all supporting documents to occur to gether. This contrast shows why coverage alone cannot characterize evidence-set recovery and why the two metrics provide complementary views of the retrieval objective.

a Macro scores (%)  
Table 1: Answer generation across four datasets. Scores range from 0 to 1. Baselines are grouped by retrieval mechanism and ordered within each group by average LLM-Acc. Averages are unweighted. Best in bold, second underlined.
<table><tr><td rowspan="2">Category</td><td rowspan="2">Method</td><td colspan="2">2WikiMultiHopQA</td><td colspan="2">HotpotQA</td><td colspan="2">MuSiQue</td><td colspan="2">ConcurrentQA</td><td colspan="2">Average</td></tr><tr><td>LLM-Acc</td><td>Contain</td><td>LLM-Acc</td><td>Contain</td><td>LLM-Acc</td><td>Contain</td><td>LLM-Acc</td><td>Contain</td><td>LLM-Acc</td><td>Contain</td></tr><tr><td rowspan="3">Query-based retrieval</td><td>Dense RAG</td><td>0.235</td><td>0.228</td><td>0.592</td><td>0.517</td><td>0.241</td><td>0.185</td><td>0.331</td><td>0.381</td><td>0.350</td><td>0.328</td></tr><tr><td>HyDE</td><td>0.273</td><td>0.266</td><td>0.654</td><td>0.568</td><td>0.288</td><td>0.231</td><td>0.374</td><td>0.415</td><td>0.397</td><td>0.370</td></tr><tr><td>ColBERTv2</td><td>0.313</td><td>0.306</td><td>0.624</td><td>0.538</td><td>0.237</td><td>0.187</td><td>0.464</td><td>0.486</td><td>0.410</td><td>0.379</td></tr><tr><td rowspan="4">Iterative and structure-aware retrieval</td><td>FastToG</td><td>0.326</td><td>0.331</td><td>0.385</td><td>0.327</td><td>0.077</td><td>0.068</td><td>0.233</td><td>0.269</td><td>0.255</td><td>0.249</td></tr><tr><td>RAPTOR</td><td>0.264</td><td>0.269</td><td>0.393</td><td>0.346</td><td>0.102</td><td>0.076</td><td>0.308</td><td>0.364</td><td>0.267</td><td>0.264</td></tr><tr><td>ToG</td><td>0.418</td><td>0.419</td><td>0.317</td><td>0.282</td><td>0.111</td><td>0.083</td><td>0.221</td><td>0.196</td><td>0.267</td><td>0.245</td></tr><tr><td>PathRAG</td><td>0.443</td><td>0.426</td><td>0.465</td><td>0.390</td><td>0.163</td><td>0.125</td><td>0.203</td><td>0.185</td><td>0.319</td><td>0.282</td></tr><tr><td rowspan="8">Graph-based retrieval</td><td>G-Retriever</td><td>0.083</td><td>0.099</td><td>0.322</td><td>0.284</td><td>0.106</td><td>0.077</td><td>0.216</td><td>0.299</td><td>0.182</td><td>0.190</td></tr><tr><td>LightRAG</td><td>0.188</td><td>0.188</td><td>0.535</td><td>0.453</td><td>0.232</td><td>0.160</td><td>0.293</td><td>0.339</td><td>0.312</td><td>0.285</td></tr><tr><td>GraphRAG</td><td>0.358</td><td>0.357</td><td>0.484</td><td>0.437</td><td>0.140</td><td>0.111</td><td>0.357</td><td>0.381</td><td>0.335</td><td>0.322</td></tr><tr><td>DALK</td><td>0.384</td><td>0.363</td><td>0.463</td><td>0.356</td><td>0.213</td><td>0.140</td><td>0.338</td><td>0.305</td><td>0.350</td><td>0.291</td></tr><tr><td>HippoRAG 2</td><td>0.447</td><td>0.580</td><td>0.701</td><td>0.644</td><td>0.291</td><td>0.283</td><td>0.485</td><td>0.549</td><td>0.481</td><td>0.514</td></tr><tr><td>GFM-RAG</td><td>0.668</td><td>0.744</td><td>0.718</td><td>0.657</td><td>0.320</td><td>0.318</td><td>0.613</td><td>0.625</td><td>0.580</td><td>0.586</td></tr><tr><td>LinearRAG</td><td>0.679</td><td>0.694</td><td>0.752</td><td>0.647</td><td>0.369</td><td>0.332</td><td>0.662</td><td>0.617</td><td>0.616</td><td>0.573</td></tr><tr><td>QueryFlow</td><td>0.829</td><td>0.819</td><td>0.786</td><td>0.695</td><td>0.439</td><td>0.375</td><td>0.732</td><td>0.678</td><td>0.696</td><td>0.642</td></tr></table>

Table 2: Macro retrieval scores (%), ordered by FR@30. We compare R@� and FR@� for methods with document-level outputs, obtained directly or by mapping retrieved graph or hierarchical units to source documents. Results for each dataset appear in Table 7. Best in bold, second underlined.
<table><tr><td>Method</td><td>R@10</td><td>FR@10</td><td>R@30</td><td>FR@30</td></tr><tr><td>G-Retriever</td><td>36.22</td><td>16.72</td><td>38.58</td><td>17.75</td></tr><tr><td>DALK</td><td>57.09</td><td>29.38</td><td>60.63</td><td>33.09</td></tr><tr><td>RAPTOR</td><td>56.53</td><td>27.84</td><td>68.19</td><td>43.84</td></tr><tr><td>LightRAG</td><td>61.66</td><td>37.28</td><td>69.67</td><td>47.70</td></tr><tr><td>GraphRAG</td><td>65.72</td><td>41.58</td><td>74.65</td><td>54.27</td></tr><tr><td>HyDE</td><td>66.52</td><td>43.17</td><td>75.16</td><td>54.73</td></tr><tr><td>ColBERTv2</td><td>71.20</td><td>46.91</td><td>78.63</td><td>58.37</td></tr><tr><td>HippoRAG 2</td><td>74.29</td><td>51.38</td><td>80.95</td><td>61.59</td></tr><tr><td>GFM-RAG</td><td>76.95</td><td>61.98</td><td>81.41</td><td>68.15</td></tr><tr><td>QueryFlow</td><td>81.93</td><td>70.38</td><td>87.12</td><td>77.78</td></tr></table>

Closer agreement between coverage and completeness. Figure 3 shows that QueryFlow’s gap between support coverage and complete recovery decreases from 11.55 points at � = 10 to 9.34 at � = 30. GFM-RAG records gaps of 14.97 and 13.26 points, and HippoRAG 2 22.91 and 19.36 points, respectively. This diference captures the average recall contributed by questions whose support sets remain incomplete. Together with higher values of both metrics, these smaller gaps show that QueryFlow combines broader support coverage with more frequent complete recovery at both budgets.

![](images/95a871337e44e492452ccefb7e864104c7aadfa2817bf3c7982508bfcf7aaaa0.jpg)  
Figure 3: Macro retrieval comparison, ordered by R@10. (a) Scores from Table 2; darker cells indicate higher values. (b) Diferences between the displayed R@� and FR@� at � = 10 and � = 30, with bars sharing a scale from 0 to 30 percentage points.

## 4.3 Query Evolution Analysis

We vary the query update strength � ∈ {0, 0.25, 0.5, 0.75, 1} on all four benchmarks while retaining the learned model parameters. Figure 4 shows macro gains over propagation with a fixed query and the efects of enabling query evolution on each dataset.

RQ3. How does adaptive query evolution afect retrieval performance?

a Query evolution  
![](images/7df1e07ba00a894d5869eeee89ebca84f6d4c19f862da67d87e22d8596108798.jpg)

b α=1 vs α=0 by dataset  
![](images/dafef8b439fa45b6377eb3d1f15e6d264281c84339f8d9336212e9dc23fc31d0.jpg)  
Figure 4: Efect of query evolution. (a) Macro recall and full recall gains over $\alpha = 0$ at $k = 1 0$ and $k = 3 0 .$ . (b) FR@10 gains from $\alpha = 0$ to $\alpha = 1$ by dataset and their macro average with equal weights.

Query evolution improves complete recovery across all four benchmarks. $\operatorname { A t } \alpha = 1 ;$ , macro R@10 and FR@10 reach 81.93% and 70.38%, gains of 5.27 and 8.59 percentage points over $\alpha = 0 .$ Figure 4(b) shows positive FR@10 gains on every dataset, ranging from 2.50 points on 2WikiMultiHopQA to 15.51 on ConcurrentQA. With the anchored $q _ { 0 }$ scorer and document readout fixed, query feedback afects document ranking through the propagated entity representations. The larger gain in full recall connects this feedback to recovering complementary support, beyond improving average coverage alone.

Stronger updates consistently improve recovery across the tested settings. From $\alpha = 0 . 2 5$ to $\alpha = 1 .$ , macro R@10 increases from 79.16% to 81.93%, while FR@10 rises from 65.60% to 70.38% (Figure 4(a)). The same upward trend holds at $k = 3 0$ , where full updates gain 3.76 points in recall and 6.45 in full recall over $\alpha = 0 .$ Increasing � gives the graph-derived update a larger role in the query state that controls subsequent boundary inputs and relation features. Continued improvement up to $\alpha = 1$ supports using accumulated relational context throughout propagation.

## 4.4 Ablation Study

Table 3 examines the feedback interfaces and the choice between initial-query $q _ { 0 }$ and terminal-query �<sub>�</sub> readout, retaining the fixedquery variant from RQ3 for reference. Appendix C examines entity aggregation alternatives.

RQ4. How do boundary feedback, relation feedback, and the readout query contribute to retrieval performance?

Both interfaces contribute, with a larger efect from boundary feedback. Removing boundary feedback lowers FR@10 and

Table 3: Macro ablation scores (%). Best in bold.
<table><tr><td>Variant</td><td>R@10</td><td>FR@10</td><td>R@30</td><td>FR@30</td></tr><tr><td>Full model</td><td>81.93</td><td>70.38</td><td>87.12</td><td>77.78</td></tr><tr><td>w/o query evolution  $( \alpha = 0 )$ </td><td>76.66</td><td>61.78</td><td>83.36</td><td>71.33</td></tr><tr><td>w/o boundary feedback</td><td>79.04</td><td>65.65</td><td>85.05</td><td>74.24</td></tr><tr><td>w/o relation feedback</td><td>79.83</td><td>66.94</td><td>85.62</td><td>75.20</td></tr><tr><td>Readout with  $q _ { L }$ </td><td>80.65</td><td>68.56</td><td>86.13</td><td>76.76</td></tr></table>

FR@30 by 4.73 and 3.54 points, compared with 3.44 and 2.58 for relation feedback. The former refreshes query information at questionlinked entities; the latter controls neighboring-message transformations. Their complementary roles support using both interfaces.

Initial-query readout improves early ranking. The $q _ { 0 }$ readout achieves R@10 and FR@10 of 81.93% and 70.38%, improving over $q _ { L }$ by 1.28 and 1.82 percentage points (Table 3). $\mathrm { A t } \ k = 3 0$ , the corresponding scores are 87.12% and 77.78%, with gains of 0.99 and 1.02 points. The larger full-recall gain at � = 10 supports anchoring final relevance scores to the original question while evolving queries guide context accumulation.

## 4.5 Evidence Recovery Analysis

Figure 5 compares QueryFlow and GFM-RAG on 4,596 common evaluable questions using the shared support targets defined in Section 4.1. We report FR@10 by support set size and count paired changes between incomplete and complete recovery.

RQ5. How do gains in complete recovery of evidence sets vary with support set size?

![](images/cc2d0f50a653ed0f702b5e7e2e643ff13403d189f527019b537f116ea1572d65.jpg)

b Paired outcomes · net +348  
![](images/e646764f7f9d1d8288ca5529dc063d9ff257b6f5c10b4a953cb3fffd4da15542.jpg)  
Figure 5: Evidence recovery at $k = 1 0 .$ (a) FR@10 by support set size; labels show gains (pp), and � is group size. (b) Paired outcomes. The 434 gains minus 86 losses yield 348 net recoveries.

The largest absolute gain occurs for support sets with four documents. Figure 5(a) shows that QueryFlow raises FR@10 from 32.16% to 50.25% for support sets with four documents, a gain of

18.09 percentage points. The gains for sets with two and three documents are 6.38 and 9.62 points, respectively, extending the benefit to questions requiring several supporting documents. Larger support sets require more complementary documents to appear together, and a single missing document prevents full recovery. The larger margin in this group is consistent with feedback from graph context helping connect evidence distributed across multiple entities and documents. Support set size describes evidence requirements across the pooled benchmarks and also reflects their diferent dataset composition.

Paired comparisons yield 348 additional complete recoveries. QueryFlow completes 434 support sets missed by GFM-RAG, with 86 transitions in the reverse direction, for a net gain of 348 questions (Figure 5(b)). FR@10 weighted by question count rises from 62.90% to 70.47%, a gain of 7.57 percentage points. Net gains occur on every benchmark, comprising 121 questions on 2Wiki-MultiHopQA, 75 on HotpotQA, 56 on MuSiQue, and 96 on ConcurrentQA. Each additional recovery makes an entire support set available within the same document budget. The paired outcomes thus identify additional questions with complete support across all four collections.

Additional recoveries preserve most previously complete support sets. Of the 1,705 questions incomplete under GFM-RAG, QueryFlow completes 434, or 25.5%. It also retains complete recovery for 2,805 of the 2,891 questions completed by GFM-RAG, or 97.0%. These rates pool questions across datasets; the main tables weight datasets equally. These conditional rates separate the recov ery of missing support from the preservation of existing coverage. Together, they show that the net gain combines additional complete sets with substantial overlap in successful questions, rather than merely exchanging which questions are recovered.

## 4.6 Retrieval Budget Analysis

Figure 6 compares complete evidence recovery as the retrieval budget increases from 3 to 30 documents across the four benchmarks. The curves address the following question.

RQ6. How does retrieval budget afect complete recovery of evidence sets across methods?

QueryFlow reaches comparable recovery levels with fewer returned documents. QueryFlow reaches 62.63% FR@5, compared with GFM-RAG’s 61.98% FR@10; its 70.38% FR@10 also exceeds GFM-RAG’s 68.15% FR@30. At matched cutofs, the margin over GFM-RAG grows from 4.37 points at � = 3 to 8.40 at � = 10 and 9.63 at $k = 3 0$ . QueryFlow also scores above HippoRAG 2, ColBERTv2, HyDE, and LightRAG throughout the tested range. These comparisons show that complete support sets appear earlier in QueryFlow’s ranking. Aggregating relevance over connected entities can promote complementary documents together, making complete evidence available under smaller document budgets.

Larger budgets recover additional complete support sets. QueryFlow’s macro FR@� increases from 51.99% at � = 3 to 70.38% at � = 10 and 77.78% at � = 30. Expanding the budget from 10 to 20 adds 4.87 percentage points, followed by another 2.54 points from 20 to 30. The gap between R@� and FR@� also narrows from 13.85 points at � = 5 to 9.34 at � = 30. Under a fixed ranking, a question becomes fully recovered when its last required document enters the returned prefix. Average gains per added document decline over the tested intervals in Figure 6(b), characterizing the tradeof between retrieval budget and evidence completeness. The cutof controls evidence passed to the generator without changing the ranking or repeating graph propagation. The paired and budget analyses address complementary aspects of evidence recovery. Paired outcomes show which questions gain or lose complete support at a fixed cutof, whereas budget curves show how early complete support appears in a ranking. Together, they connect the aggregate retrieval improvements to the availability of a complete supporting set for individual questions.

a Complete recovery  
![](images/ccbe604906646bcdfc14911de2221eeeeb0eeff5cc79aea60486faa6edb5f869.jpg)

b Marginal gain · QueryFlow  
![](images/31ffff395046020c3ffc196fe66ca7d68d7a2971a3bc86c31cbf32d2e4b05542.jpg)  
Figure 6: Retrieval budget analysis. (a) Macro FR@� for methods that rank documents. (b) QueryFlow’s marginal FR gain, ΔFR/Δ�, in percentage points per added document.

## 4.7 Eficiency Analysis

Table 4 compares QueryFlow with existing methods in indexing cost, online time, and token consumption on 2WikiMultiHopQA. Together with answer quality in Table 1, these results address the following question.

RQ7. How does QueryFlow compare with existing methods in eficiency, token cost, and answer quality?

Eficient online retrieval and answer generation. QueryFlow records 0.580 ks for retrieval and answer generation, compared with 5.96 ks for GFM-RAG, 1.36 ks for HippoRAG 2, and 1.19 ks for ToG (Table 4). The corresponding reductions in recorded time are 90.3%, 57.3%, and 51.2%, respectively. Query evolution refines the retrieval condition through tensor operations on the existing graph, keeping successive feedback steps within graph computation.

Query adaptation adds no API-token cost. QueryFlow uses 3.59M tokens for retrieval and reasoning, compared with GFM-RAG’s 5.44M, a reduction of 34.1%. With comparable indexing costs of about 11.2M tokens, QueryFlow’s estimated total is 14.8M versus 16.6M for GFM-RAG, a reduction of 10.8%. The graph retriever itself consumes no API tokens, so updating the latent query state adds no intermediate query-generation token cost. This allows adaptive retrieval without repeated textual reasoning calls during graph propagation.

Table 4: Eficiency on 2WikiMultiHopQA. Online time and tokens include query processing, retrieval, and answer generation.
<table><tr><td rowspan="2">Method</td><td colspan="2">Index/Build</td><td colspan="2">Retrieval/Reasoning</td></tr><tr><td>Time (ks)</td><td>Tokens (M)</td><td>Time (ks)</td><td>Tokens (M)</td></tr><tr><td>GFM-RAG</td><td>1.80</td><td>11.2</td><td>5.96</td><td>5.44</td></tr><tr><td>HippoRAG 2</td><td>2.08</td><td>11.1</td><td>1.36</td><td>8.53</td></tr><tr><td>ToG</td><td>2.11</td><td>5.15</td><td>1.19</td><td>12.7</td></tr><tr><td>FastToG</td><td>0.128</td><td>6.44</td><td>0.00820</td><td>1.25</td></tr><tr><td>QueryFlow</td><td>1.80</td><td>11.2</td><td>0.580</td><td>3.59</td></tr></table>

Answer quality improves alongside resource eficiency. Table 1 reports LLM-Acc of 0.829 for QueryFlow and 0.668 for GFM-RAG on 2WikiMultiHopQA, a gain of 16.1 percentage points. The macro gain over GFM-RAG is 11.6 points. These answer results accompany lower recorded time and token costs relative to GFM-RAG in Table 4. QueryFlow also achieves higher LLM-Acc than FastToG (0.829 versus 0.326), with FastToG ofering a lower recorded online time and token cost.

## 5 Conclusion

We presented QueryFlow, a graph retrieval framework that supports multi-hop question answering through coupled evidence propagation and query evolution. Accumulated graph context updates a shared query state, which controls subsequent boundary inputs and relation features. Across four multi-hop QA benchmarks, QueryFlow improves macro R@10, FR@10, and LLM-Acc over GFM-RAG by 4.98, 8.40, and 11.6 percentage points, respectively. At a budget of five documents, it also exceeds this baseline’s full recall at ten documents. Macro LLM-Acc also improves by 8.0 points over LinearRAG. Query-state interventions and interface ablations sup port the contribution of feedback within QueryFlow. The results connect evolving query conditions to more complete evidence sets and stronger downstream answers.

## References

[1] Simran Arora, Patrick Lewis, Angela Fan, Jacob Kahn, and Christopher Ré. 2023. Reasoning over Public and Private Data in Retrieval-Based Systems. Transactions ofthe Association for Computational Linguistics 11 (2023), 902–921.

[2] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. In The Twelfth International Conference on Learning Representations.

[3] Sebastian Borgeaud, Arthur Mensch, Jordan Hofmann, Trevor Cai, Eliza Rutherford, Katie Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Safron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geofrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack W. Rae, Erich Elsen, and Laurent Sifre. 2022. Improving Language Models by Retrieving from Trillions of Tokens. In Proceedings of the 39th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 162). PMLR, 2206–2240.

[4] Jaime Carbonell and Jade Goldstein. 1998. The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries. In Proceedings of the 21st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval. Association for Computing Machinery, 335–336.

[5] Boyu Chen, Zirui Guo, Zidan Yang, Yuluo Chen, Junze Chen, Zhenghao Liu, Chuan Shi, and Cheng Yang. 2026. PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths. Proceedings ofthe AAAIConference on Artificial Intelligence 40, 36 (2026), 30183–30191.

[6] Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, 1724–1734.

[7] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. 2024. From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv:2404.16130

[8] Mikhail Galkin, Xinyu Yuan, Hesham Mostafa, Jian Tang, and Zhaocheng Zhu. 2024. Towards Foundation Models for Knowledge Graph Reasoning. In The Twelfth International Conference on Learning Representations.

[9] Guangze Gao, Zixuan Li, Chunfeng Yuan, Jiawei Li, Jianzhuo Wu, Yuehao Zhang, Xiaolong Jin, Bing Li, and Weiming Hu. 2025. D-RAG: Diferentiable Retrieval-Augmented Generation for Knowledge Graph Question Answering. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 35398–35417.

[10] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. Precise Zero-Shot Dense Retrieval without Relevance Labels. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 1762–1777.

[11] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2025. LightRAG: Simple and Fast Retrieval-Augmented Generation. In Findings ofthe Association for Computational Linguistics: EMNLP 2025. Association for Computational Linguistics, 10746–10761.

[12] Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. 2024. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. In Advances in Neural Information Processing Systems, Vol. 37. Curran Associates, Inc., 59532–59569.

[13] Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. 2025. From RAG to Memory: Non-Parametric Continual Learning for Large Language Models. In Proceedings of the 42nd International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 267). PMLR, 21497–21515.

[14] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020. REALM: Retrieval-Augmented Language Model Pre-Training. In Proceedings ofthe 37th International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 119). PMLR, 3929–3938.

[15] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V. Chawla, Thomas Laurent, Yann LeCun, Xavier Bresson, and Bryan Hooi. 2024. G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering. In Advances in Neural Information Processing Systems, Vol. 37. Curran Associates, Inc., 132876–132907.

[16] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps. In Proceedings ofthe 28th International Conference on Computational Linguistics. International Committee on Computational Linguistics, 6609–6625.

[17] Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling, and Liang Zhao. 2025. GRAG: Graph Retrieval-Augmented Generation. In Findings ofthe Association for Computational Linguistics: NAACL 2025. Association for Computational Linguistics, 4145–4157.

[18] Manzong Huang, Chenyang Bu, Yi He, Xingrui Zhuo, and Xindong Wu. 2026. Relink: Constructing Query-Driven Evidence Graph On-the-Fly for GraphRAG.

Proceedings of the AAAI Conference on Artificial Intelligence 40, 37 (2026), 31202– 31210.

[19] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised Dense Information Retrieval with Contrastive Learning. Transactions on Machine Learning Research (2022).

[20] Gautier Izacard and Edouard Grave. 2021. Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering. In Proceedings ofthe 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume. Association for Computational Linguistics, 874–880.

[21] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. 2023. Atlas: Few-shot Learning with Retrieval Augmented Language Models. Journal ofMachine Learning Research 24, 251 (2023), 1–43.

[22] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong Park. 2024. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). Association for Computational Linguistics, 7036–7050.

[23] Yichen Jiang, Shikha Bordia, Zheng Zhong, Charles Dognin, Maneesh Singh, and Mohit Bansal. 2020. HoVer: A Dataset for Many-Hop Fact Extraction And Claim Verification. In Findings ofthe Association for Computational Linguistics: EMNLP 2020. Association for Computational Linguistics, 3441–3460.

[24] Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active Retrieval Augmented Generation. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 7969–7992.

[25] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, 6769–6781.

[26] Omar Khattab, Christopher Potts, and Matei Zaharia. 2021. Baleen: Robust Multi-Hop Reasoning at Scale via Condensed Retrieval. In Advances in Neural Information Processing Systems, Vol. 34. Curran Associates, Inc., 27670–27682.

[27] Omar Khattab and Matei Zaharia. 2020. ColBERT: Eficient and Efective Passage Search via Contextualized Late Interaction over BERT. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. Association for Computing Machinery, 39–48.

[28] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems, Vol. 33. Curran Associates, Inc., 9459–9474.

[29] Dawei Li, Shu Yang, Zhen Tan, Jae Young Baik, Sukwon Yun, Joseph Lee, Aaron Chacko, Bojian Hou, Duy Duong-Tran, Ying Ding, Huan Liu, Li Shen, and Tian long Chen. 2024. DALK: Dynamic Co-Augmentation of LLMs and KG to answer Alzheimer’s Disease Questions with Scientific Literature. In Findings ofthe Association for Computational Linguistics: EMNLP 2024. Association for Computational Linguistics, 2187–2205.

[30] Mufei Li, Siqi Miao, and Pan Li. 2025. Simple is Efective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation. In The Thirteenth International Conference on Learning Representations.

[31] Zijian Li, Qingyan Guo, Jiawei Shao, Lei Song, Jiang Bian, Jun Zhang, and Rui Wang. 2025. Graph Neural Network Enhanced Retrieval for Question Answering of Large Language Models. In Proceedings ofthe 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). Association for Computational Linguistics, 6612–6633.

[32] Xujian Liang and Zhaoquan Gu. 2025. Fast Think-on-Graph: Wider, Deeper and Faster Reasoning of Large Language Model on Knowledge Graph. Proceedings of the AAAI Conference on Artificial Intelligence 39, 23 (2025), 24558–24566.

[33] Haoran Luo, Haihong E, Guanting Chen, Qika Lin, Yikai Guo, Fangzhi Xu, Zemin Kuang, Meina Song, Xiaobao Wu, Yifan Zhu, and Anh Tuan Luu. 2026. Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning. In Proceedings of the 43rd International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 306). PMLR, 82220–82240.

[34] Linhao Luo, Yuan-Fang Li, Gholamreza Hafari, and Shirui Pan. 2024. Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning. In The Twelfth International Conference on Learning Representations.

[35] Linhao Luo, Zicheng Zhao, Gholamreza Hafari, Chen Gong, Dinh Phung, and Shirui Pan. 2025. GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation. In Advances in Neural Information Processing Systems, Vol. 38. Curran Associates, Inc., 36371–36405.

[36] Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. 2023. Query Rewriting in Retrieval-Augmented Large Language Models. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 5303–5315.

[37] Costas Mavromatis and George Karypis. 2025. GNN-RAG: Graph Neural Retrieval for Eficient Large Language Model Reasoning on Knowledge Graphs. In Findings ofthe Association for Computational Linguistics: ACL 2025. Association for Computational Linguistics, 16682–16699.

[38] Peng Qi, Xiaowen Lin, Leo Mehr, Zijian Wang, and Christopher D. Manning. 2019. Answering Complex Open-domain Questions Through Iterative Query Generation. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). Association for Computational Linguistics, 2590–2602.

[39] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Mate Zaharia. 2022. ColBERTv2: Efective and Eficient Retrieval via Lightweight Late Interaction. In Proceedings ofthe 2022 Conference ofthe North American Chapte ofthe Association for Computational Linguistics: Human Language Technologies. Association for Computational Linguistics, 3715–3734.

[40] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D. Manning. 2024. RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. In The Twelfth International Conference on Learning Representations. OpenReview.net.

[41] Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen. 2023. Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy. In Findings of the Association for Computational Linguistics: EMNLP 2023. Association for Computational Linguistics, 9248–9274.

[42] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2024. REPLUG: Retrieval-Augmented Black-Box Language Models. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). Association for Computational Linguistics, 8371–8384.

[43] Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu, and Yiqun Liu. 2024. DRAGIN: Dynamic Retrieval Augmented Generation based on the Real-time Information Needs of Large Language Models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 12991–13013.

[44] Haitian Sun, Tania Bedrax-Weiss, and William W. Cohen. 2019. PullNet: Open Domain Question Answering with Iterative Retrieval on Knowledge Bases and Text. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). Association for Computational Linguistics, 2380– 2390.

[45] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel M. Ni, Heung-Yeung Shum, and Jian Guo. 2024. Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph. In The Twelfth International Conference on Learning Representations.

[46] Alon Talmor and Jonathan Berant. 2018. The Web as a Knowledge-Base for Answering Complex Questions. In Proceedings of the 2018 Conference of the North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers). Association for Computational Linguistics, 641–651.

[47] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. MuSiQue: Multihop Questions via Single-hop Question Composition. Transactions ofthe Association for Computational Linguistics 10 (2022), 539–554.

[48] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. In Proceedings ofthe 61st Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 10014–10037.

[49] Jiapu Wang, Zheng Cui, Boyue Wang, Shirui Pan, Junbin Gao, Baocai Yin, and Wen Gao. 2024. IME: Integrating Multi-curvature Shared and Specific Embedding for Temporal Knowledge Graph Completion. In Proceedings of the ACM Web Conference 2024. Association for Computing Machinery, 1954–1962. doi:10.1145/ 3589334.3645361

[50] Jiapu Wang, Kai Sun, Linhao Luo, Wei Wei, Yongli Hu, Alan Liew, Shirui Pan, and Baocai Yin. 2024. Large Language Models-guided Dynamic Adaptation for Temporal Knowledge Graph Reasoning. In Advances in Neural Information Processing Systems, Vol. 37. Neural Information Processing Systems Foundation, 8384–8410. doi:10.52202/079017-0269

[51] Jinlu Wang, Yanfeng Sun, Jiapu Wang, Junbin Gao, Shaofan Wang, Baocai Yin, and Jipeng Guo. 2026. ComGRL: Comprehensive Graph Representation Learning from Local to Global Bridged by Mixup. IEEE Transactions on Computational Social Systems 13, 3 (2026), 2839–2852. doi:10.1109/TCSS.2026.3665450

[52] Jiapu Wang, Boyue Wang, Junbin Gao, Simin Hu, Yongli Hu, and Baocai Yin. 2024. Multi-Level Interaction Based Knowledge Graph Completion. IEEE/ACM

Transactions on Audio, Speech, and Language Processing 32 (2024), 386–396. doi:10. 1109/TASLP.2023.3331121

[53] Jiapu Wang, Boyue Wang, Junbin Gao, Yongli Hu, and Baocai Yin. 2023. Multi-Concept Representation Learning for Knowledge Graph Completion. ACM Transactions on Knowledge Discovery from Data 17, 1 (2023), 1–19. doi:10.1145/ 3533017

[54] Jiapu Wang, Boyue Wang, Junbin Gao, Xiaoyan Li, Yongli Hu, and Baocai Yin. 2023. TDN: Triplet Distributor Network for Knowledge Graph Completion. IEEE Transactions on Knowledge and Data Engineering 35, 12 (2023), 13002–13014. doi:10.1109/TKDE.2023.3272568

[55] Jiapu Wang, Boyue Wang, Junbin Gao, Xiaoyan Li, Yongli Hu, and Baocai Yin. 2024. QDN: A Quadruplet Distributor Network for Temporal Knowledge Graph Completion. IEEE Transactions on Neural Networks and Learning Systems 35, 10 (2024), 14018–14030. doi:10.1109/TNNLS.2023.3274230

[56] Wenhan Xiong, Xiang Lorraine Li, Srinivasan Iyer, Jingfei Du, Patrick Lewis, William Wang, Yashar Mehdad, Wen-tau Yih, Sebastian Riedel, Douwe Kiela, and Barlas Oğuz. 2021. Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval. In International Conference on Learning Representations.

[57] Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. 2015. Embedding Entities and Relations for Learning and Inference in Knowledge Bases. In International Conference on Learning Representations.

[58] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2369–2380.

[59] Jiahao Zhang, Haiyang Zhang, Dongmei Zhang, Yong Liu, and Shen Huang. 2024. End-to-End Beam Retrieval for Multi-Hop Question Answering. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). Association for Computational Linguistics, 1718–1731.

[60] Tianxiang Zhao, Youqing Wang, Jinlu Wang, Jiapu Wang, Mingliang Cui, Junbin Gao, and Jipeng Guo. 2025. Hybrid-Collaborative Augmentation and Contrastive Sample Adaptive-Diferential Awareness for Robust Attributed Graph Clustering. In Advances in Neural Information Processing Systems, Vol. 38. Neural Information Processing Systems Foundation, 128567–128592. doi:10.52202/085713-3867

[61] Xiangrong Zhu, Yuexiang Xie, Yi Liu, Yaliang Li, and Wei Hu. 2025. Knowledge Graph-Guided Retrieval Augmented Generation. In Proceedings of the 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). Association for Computational Linguistics, 8912–8924.

[62] Zhaocheng Zhu, Xinyu Yuan, Mikhail Galkin, Sophie Xhonneux, Ming Zhang, Maxime Gazeau, and Jian Tang. 2023. A\*Net: A Scalable Path-based Reasoning Approach for Knowledge Graphs. In Advances in Neural Information Processing Systems, Vol. 36. Curran Associates, Inc., 59323–59336.

[63] Zhaocheng Zhu, Zuobai Zhang, Louis-Pascal Xhonneux, and Jian Tang. 2021. Neural Bellman-Ford Networks: A General Graph Neural Network Framework for Link Prediction. In Advances in Neural Information Processing Systems, Vol. 34. Curran Associates, Inc., 29476–29490.

[64] Luyao Zhuang, Shengyuan Chen, Yilin Xiao, Huachi Zhou, Yujing Zhang, Hao Chen, Qinggang Zhang, and Xiao Huang. 2026. LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora. In The Fourteenth International Conference on Learning Representations.

[65] Xingrui Zhuo, Shirui Pan, Jiapu Wang, Gongqing Wu, Zan Zhang, Rui Li, Zizhong Wei, and Xindong Wu. 2025. Progressive Prefix-Memory Tuning for Complex Logical Query Answering on Knowledge Graphs. In Proceedings ofthe Thirty-Fourth International Joint Conference on Artificial Intelligence. International Joint Conferences on Artificial Intelligence Organization, 3716–3724. doi:10.24963/ ijcai.2025/413

[66] Xingrui Zhuo, Jiapu Wang, Gongqing Wu, Shirui Pan, and Xindong Wu. 2025. Efective Instruction Parsing Plugin for Complex Logical Query Answering on Knowledge Graphs. In Proceedings ofthe ACM on Web Conference 2025. ACM, 4780–4792. doi:10.1145/3696410.3714794

[67] Xingrui Zhuo, Jiapu Wang, Gongqing Wu, Zhongyuan Wang, Jichen Zhang, Shirui Pan, and Xindong Wu. 2026. Knowledge Reasoning Language Model: Unifying Knowledge and Language for Inductive Knowledge Graph Reasoning. In International Conference on Learning Representations. https://proceedings.iclr.cc/paper\_files/paper/2026/file/ c6e9f9d49cc052879efd19da0c3e1900-Paper-Conference.pd

Table 5: Evaluation input records and retrieval corpus sizes.
<table><tr><td>Dataset</td><td>Input records</td><td>Documents</td></tr><tr><td>2WikiMultiHopQA</td><td>1,000</td><td>9,594</td></tr><tr><td>HotpotQA</td><td>1,000</td><td>9,811</td></tr><tr><td>MuSiQue</td><td>1,003</td><td>11,666</td></tr><tr><td>ConcurrentQA</td><td>1,600</td><td>2,840</td></tr></table>

## A Additional Experimental Details

## A.1 Experimental Setup

2WikiMultiHopQA. This benchmark combines Wikidata relations with Wikipedia text to construct questions with explicit multihop reasoning paths [16]. These annotations specify the intermediate relations linking a question to its answer, providing a structured basis for assessing evidence-set recovery.

HotpotQA. Questions draw on multiple Wikipedia documents and include comparisons between entities [58]. Sentence-level sup porting facts identify the evidence needed for an answer. We map these annotations to source documents to define the target evidence sets.

MuSiQue. MuSiQue composes single-hop questions into connected two- to four-hop questions [47]. Intermediate answers link successive steps, reducing opportunities for disconnected reasoning and making subsequent evidence needs depend on earlier discoveries. We use the answerable examples.

ConcurrentQA. ConcurrentQA requires evidence from distinct public and private data distributions [1]. We use the supplied documents as a shared retrieval collection to evaluate evidence recovery across heterogeneous sources with the same document-ranking interface.

Evaluation subsets. Following GFM-RAG [35], LinearRAG [64], and Relink [18], we use fixed evaluation subsets. Matching Lin earRAG’s reference question lists to source records yields 1,000 questions each for 2WikiMultiHopQA and HotpotQA, with HotpotQA drawn from distractor validation. For MuSiQue, we match the reference list to records in the answerable validation split. HotpotQA and MuSiQue are shufled with seed 42 before matching; list membership determines selection. ConcurrentQA uses all 1,600 validation questions. Input sets are shared across methods. Retrieval metrics are averaged over questions with nonempty support sets (Section 4.1).

Retrieval corpora. Supporting and distractor documents from the selected examples form each benchmark’s retrieval corpus. Table 5 reports selected input records and corpus sizes after preprocessing. Each question is evaluated against this collection, with � limiting the number of returned documents. Original support annotations define the target evidence sets.

Training data and supervision. We use 20,000 examples and their associated content from each of 2WikiMultiHopQA, HotpotQA, and MuSiQue, totaling 60,000 training examples. Entity labels $y _ { i }$ are derived from these training datasets. The graph interface is defined in Section 3.1. Extraction, normalization, linking, grounding, and document-to-entity label construction are detailed in the supplementary code.

## A.2 Implementation Details

Model architecture. The frozen all-mpnet-base-v2 encoder supplies precomputed 768-dimensional question and relation embeddings, linearly projected to $d = 5 1 2$ . The propagation operator uses the NBFNet formulation [63]. The retriever uses � = 6 layers with 512 hidden units, DistMult messages, sum aggregation, residual connections, layer normalization, and ReLU. The query controller also has 512-dimensional states. The full model enables both feedback interfaces with $\alpha = 1$ , attention temperature 1, and query-update dropout 0.

Optimization and hardware. We train the retriever and query controller from random initialization on a single NVIDIA A100 using AdamW, learning rate $1 0 ^ { - 5 }$ , batch size one, gradient norm clipping at 1.0, and seed 1024. Activation checkpointing reduces propagation memory; the text encoder remains frozen.

Answer generation and judgment. All methods generate answers with Qwen3.5-9B. Following LinearRAG [64], LLM-Acc records LLM judge acceptance; Contain checks for the normalized gold answer. Prompts, context handling, decoding, judge configuration, and output parsing are detailed in the supplementary code.

## A.3 Additional Results

Tables 6 and 7 report additional answer metrics and retrieval scores by dataset. Macro averages weight datasets equally.

Table 6: Additional answer generation metrics for QueryFlow (%). Macro is the unweighted mean across datasets.

<table><tr><td>Dataset</td><td>EM</td><td>F1 Precision Recall Contain LLM-Acc</td><td></td><td></td><td></td><td></td></tr><tr><td>2WikiMultiHopQA 65.3%75.4%</td><td></td><td></td><td>73.9%</td><td>83.3%</td><td>81.9%</td><td>82.9%</td></tr><tr><td>HotpotQA</td><td>56.7% 69.6%</td><td></td><td>71.0%</td><td>77.0%</td><td>69.5%</td><td>78.6%</td></tr><tr><td>MuSiQue</td><td>24.5% 35.1%</td><td></td><td>35.3%</td><td>49.0%</td><td>37.5%</td><td>43.9%</td></tr><tr><td>ConcurrentQA</td><td>55.7%63.5%</td><td></td><td>63.5%</td><td>72.5%</td><td>67.8%</td><td>73.2%</td></tr><tr><td>Macro</td><td>50.5% 60.9%</td><td></td><td>60.9%</td><td>70.4%</td><td>64.2%</td><td>69.6%</td></tr></table>

## B Training Objective and Retrieval Algorithm

## B.1 Document Evidence Readout

Default entity scoring uses �<sub>0</sub> during both training and retrieval. Intermediate query updates influence document ranking through the final entity representations �<sub>�</sub>.

Aggregation over all entities. Softplus preserves entity ordering and yields positive contributions v. With $\begin{array} { r } { f _ { i } = \sum _ { j } A _ { i j } } \end{array}$ and $w _ { i } ^ { \mathrm { d o c } } =$ $1 / f _ { i }$ for $f _ { i } > 0$ (zero otherwise), the document scores are

$$
\begin{array} { r } { \mathbf { d } = { \cal A } ^ { \top } \left( { \mathbf { w } } ^ { \mathrm { d o c } } \odot { \mathbf { v } } \right) . } \end{array}\tag{16}
$$

Here $A \in \{ 0 , 1 \} ^ { | V | \times | D | }$ and $\mathbf { d } \in \mathbb { R } ^ { | D | }$ . All entities contribute; the top � scores define $\widehat { D } _ { k } ( x )$ . Paired propagation interventions keep the scorer and readout fixed.

Table 7: Retrieval scores for each dataset $( \% ) ,$ ordered by macro FR@30. We compare R@� and FR@� for methods with document-level outputs, obtained directly or by mapping retrieved graph or hierarchical units to source documents. Macro is the unweighted mean across datasets. Best in bold, second underlined.
<table><tr><td></td><td colspan="4">2WikiMultiHopQA</td><td colspan="4">HotpotQA</td><td colspan="4">MuSiQue</td><td colspan="4">ConcurrentQA</td><td colspan="4">Macro</td></tr><tr><td>Method</td><td>R@10 FR@10</td><td></td><td>R@30</td><td>FR@30</td><td>R@10</td><td>FR@10</td><td>R@30</td><td>FR@30</td><td>R@10</td><td>FR@10</td><td>R@30</td><td>FR@30</td><td>R@10</td><td>FR@10</td><td>R@30</td><td>FR@30</td><td>R@10</td><td>FR@10 R@30</td><td>FR@30</td><td></td></tr><tr><td>G-Retriever</td><td>32.10</td><td>11.28</td><td>34.37</td><td>12.08</td><td>44.85</td><td>23.51</td><td>46.57</td><td>24.12</td><td>25.80</td><td>11.60</td><td>27.72</td><td>12.10</td><td>42.14</td><td>20.50</td><td>45.66</td><td>22.70</td><td>36.22</td><td>16.72</td><td>38.58</td><td>17.75</td></tr><tr><td>DALK</td><td>63.68</td><td>28.30</td><td>63.95</td><td>28.80</td><td>63.78</td><td>40.54</td><td>67.40</td><td>45.57</td><td>40.44</td><td>10.87</td><td>44.35</td><td>14.08</td><td>60.47</td><td>37.79</td><td>66.81</td><td>43.91</td><td>57.09</td><td>29.38</td><td>60.63</td><td>33.09</td></tr><tr><td>RAPTOR</td><td>59.22</td><td>27.10</td><td>80.30</td><td>61.70</td><td>65.80</td><td>40.70</td><td>65.80</td><td>40.70</td><td>40.87</td><td>8.87</td><td>66.43</td><td>38.29</td><td>60.22</td><td>34.69</td><td>60.22</td><td>34.69</td><td>56.53</td><td>27.84</td><td>68.19</td><td>43.84</td></tr><tr><td>LightRAG</td><td>52.20</td><td>24.50</td><td>59.75</td><td>31.80</td><td>82.35</td><td>67.60</td><td>88.90</td><td>79.30</td><td>57.40</td><td>24.23</td><td>69.42</td><td>39.38</td><td>54.69</td><td>32.81</td><td>60.59</td><td>40.31</td><td>61.66</td><td>37.28</td><td>69.67</td><td>47.70</td></tr><tr><td>GraphRAG</td><td>70.62</td><td>39.90</td><td>74.35</td><td>46.50</td><td>77.60</td><td>60.80</td><td>86.70</td><td>76.90</td><td>49.20</td><td>14.56</td><td>60.10</td><td>26.12</td><td>65.44</td><td>51.06</td><td>77.44</td><td>67.56</td><td>65.72</td><td>41.58</td><td>74.65</td><td>54.27</td></tr><tr><td>HyDE</td><td>58.52</td><td>30.80</td><td>66.22</td><td>38.80</td><td>86.05</td><td>73.80</td><td>91.75</td><td>84.60</td><td>63.17</td><td>31.01</td><td>74.53</td><td>46.46</td><td>58.34</td><td>37.06</td><td>68.16</td><td>49.06</td><td>66.52</td><td>43.17</td><td>75.16</td><td>54.73</td></tr><tr><td>ColBERTv2</td><td>70.50</td><td>42.20</td><td>74.98</td><td>49.10</td><td>86.75</td><td>74.30</td><td>92.70</td><td>85.50</td><td>56.60</td><td>22.33</td><td>67.95</td><td>37.19</td><td>70.97</td><td>48.81</td><td>78.91</td><td>61.69</td><td>71.20</td><td>46.91</td><td>78.63</td><td>58.37</td></tr><tr><td>HippoRAG 2</td><td>73.82</td><td>45.59</td><td>77.66</td><td>51.70</td><td>89.32</td><td>79.84</td><td>93.38</td><td>87.36</td><td>64.42</td><td>31.11</td><td>74.51</td><td>45.56</td><td>69.60</td><td>48.97</td><td>78.27</td><td>61.74</td><td>74.29</td><td>51.38</td><td>80.95</td><td>61.59</td></tr><tr><td>GFM-RAG</td><td>88.52</td><td>75.00</td><td>89.72</td><td>77.50</td><td>86.19</td><td>76.68</td><td>89.69</td><td>82.38</td><td>57.04</td><td>31.84</td><td>65.32</td><td>42.32</td><td>76.06</td><td>64.39</td><td>80.91</td><td>70.40</td><td>76.95</td><td>61.98</td><td>81.41</td><td>68.15</td></tr><tr><td>QueryFlow</td><td>93.81</td><td>87.60 95.36</td><td></td><td>90.70</td><td>90.99</td><td>84.78</td><td>94.74</td><td>90.99</td><td>61.60</td><td>37.82</td><td>72.27</td><td>51.20</td><td>81.33</td><td>71.29</td><td>86.09</td><td>78.24</td><td>81.93</td><td>70.38</td><td>87.12</td><td>77.78</td></tr></table>

## B.2 Entity-Level Training Objective

Training masks $y \in \{ 0 , 1 \} ^ { | V | }$ define positives $P = \{ i : y _ { i } = 1 \}$ and negatives $N = \{ i : y _ { i } = 0 \}$ . Binary and listwise losses supervise the graph and query controller.

Entity relevance objective. Let $p _ { i } = \sigma ( s _ { i } )$ , where �<sub>�</sub> is computed using the anchored scorer in Eq. (10). The binary cross-entropy for entity � is

$$
\ell _ { i } = - y _ { i } \log { p _ { i } } - \left( 1 - y _ { i } \right) \log { \left( 1 - p _ { i } \right) } .\tag{17}
$$

Positive weights are uniform; negative weights depend on current logits,

$$
\omega _ { i } = \left\{ \begin{array} { l l } { | P | ^ { - 1 } , } & { i \in P , } \\ { \mathrm { s g } \left( \displaystyle \frac { \exp ( s _ { i } / T ) } { \sum _ { j \in N } \exp ( s _ { j } / T ) } \right) , } & { i \in N . } \end{array} \right.\tag{18}
$$

Here $T = 0 . 2 ,$ and sg stops gradients through the negative weights. Negative entities with higher scores receive greater weight, concentrating the classification loss on negatives that the retriever currently ranks highly. The weighted binary loss is

$$
\mathcal { L } _ { \mathrm { { B C E } } } = \frac { \sum _ { i \in V } \omega _ { i } \ell _ { i } } { \sum _ { i \in V } \omega _ { i } } .\tag{19}
$$

ListCE $( \mathrm { E q . } ( 1 3 ) , \varepsilon = 1 0 ^ { - 5 } )$ normalizes sigmoid probabilities over entities and averages negative log mass over all positives, excluding empty targets. The combined loss in Eq. (14) is averaged over questions in each batch.

Aggregation of entity scores into document scores and document ranking are performed at inference time, outside the entity-level training objective. The query controller is trained through its influence on subsequent graph propagation and final entity scores; the supporting-document sets define the downstream evaluation targets.

## B.3 Retrieval Procedure

Algorithm 1 applies � propagation layers with query updates between successive layers before ranking documents. The default $\alpha = 1$ applies the gated increment before normalization; $\alpha = 0$ fixes the query without changing learned propagation parameters.

At layer �, propagation first computes $H _ { l }$ under $q _ { l - 1 }$ . When $l <$ �, the updated entity states are summarized to form $q _ { l } ,$ which conditions the next layer. The default scorer combines $H _ { L }$ with $q _ { 0 } .$ . The �<sub>�</sub> variant adds the terminal update defined in Section 3.3 and substitutes $q _ { L }$ in the scorer. Graph structure, grounding, and document associations remain fixed.

Algorithm 1 QueryFlow with the default $q _ { 0 }$ readout   
Require: $G , A , m _ { x } , e _ { x } , \{ e _ { r } \} ;$ parameters; $L , \alpha , k$   
Ensure: Ranked top-� documents $\widehat { D } _ { k } ( x )$   
1: $q _ { 0 } \gets \phi _ { q } ( e _ { x } ) ; R ^ { 0 } ( r ) \gets \phi _ { r } ( e _ { r } )$ for all �   
2: $H _ { 0 } ( i ) \gets m _ { x } ( i ) q _ { 0 }$ for all $i \in V$   
3: for $l = 1 , \ldots , L$ do   
4: $B _ { l } ( i )  m _ { x } ( i ) q _ { l - 1 }$   
5: $\gamma _ { l - 1 }  \sigma ( W _ { r } q _ { l - 1 } + b _ { r } )$   
6: $\bar { R } _ { l } ( r ) \gets \psi _ { l } ( R ^ { 0 } ( r ) \odot \gamma _ { l - 1 } )$   
7: Compute $H _ { l }$ using Eqs. (4)–(5)   
8: $\textbf { i f } l < L$ then   
9: Compute $z _ { l }$ using Eqs. (6)–(7)   
10: if $\alpha = 0$ then   
11: $q _ { l } \gets q _ { l - 1 }$   
12: else   
13: Form $\widetilde { q } _ { l } , g _ { l } , \Delta _ { l }$ using Eq. (8)   
14: $q _ { l } \gets \mathrm { L N } _ { q } ( q _ { l - 1 } + \alpha \Delta _ { l } )$   
15: end if   
16: end if   
17: end for   
18: $s _ { i } \gets \mathrm { M L P } ( [ H _ { L } ( i ) ; q _ { 0 } ] )$ for all $i \in V$   
19: �<sub>�</sub> ← softplus(�<sub>�</sub>) for all $i \in V$   
20: $\textstyle f _ { i } \gets \sum _ { j = 1 } ^ { | D | } A _ { i j }$ for all $i \in V$   
21: $w _ { i } ^ { \mathrm { d o c } }  1 / f _ { i } \mathrm { i f } f _ { i } > 0 ,$ , otherwise 0   
22: $\begin{array} { r } { d _ { j } \gets \sum _ { i \in V } A _ { i j } w _ { i } ^ { \mathrm { d o c } } v _ { i } } \end{array}$ for all documents �   
23: return the � highest-scoring documents

## C Readout Analysis

We compare aggregation over all entities with two truncated entity readouts, document-aware entity MMR [4] and relevance-based Top-20 selection. All three use the same propagated entity representations and reciprocal document frequency weighting.

Table 8: Document readout comparison. Macro retrieval scores across four datasets (%).
<table><tr><td>Readout</td><td>R@10</td><td>FR@10</td><td>R@30</td><td>FR@30</td></tr><tr><td>MMR Top-20</td><td>80.48</td><td>67.84</td><td>85.31</td><td>74.72</td></tr><tr><td>Top-20 (no MMR)</td><td>80.57</td><td>67.79</td><td>84.87</td><td>73.90</td></tr><tr><td>All entities (default)</td><td>81.93</td><td>70.38</td><td>87.12</td><td>77.78</td></tr></table>

Aggregation over all entities improves complete evidence recovery. The default readout exceeds both truncated alternatives on all four metrics in Table 8. Relative to MMR Top-20, the gains are 2.54 and 3.07 percentage points in FR@10 and FR@30. Aggregating every entity contribution lets documents receive relevance signals distributed beyond a small selected subset, providing a mechanism for retaining complementary evidence. Within the truncated alternatives, MMR improves R@30 and FR@30 by 0.44 and 0.82 points, with mixed but small diferences at � = 10 (−0.09 R@10 and +0.05 FR@10).

Reciprocal weighting preserves each entity’s total contribution. For $f _ { i } > 0$ , summing $A _ { i j } v _ { i } / f _ { i }$ over documents yields $v _ { i } .$ Summing the complete document score vector gives

$$
\sum _ { j = 1 } ^ { | D | } d _ { j } = \sum _ { i \in V } w _ { i } ^ { \mathrm { d o c } } v _ { i } \sum _ { j = 1 } ^ { | D | } A _ { i j } = \sum _ { i \in V : f _ { i } > 0 } v _ { i } .
$$

Document associations redistribute each entity’s score across its source documents. An entity appearing in one document contributes its full score there; one appearing in several documents divides that contribution equally among them. This property controls how graph relevance transfers to documents without selecting a fixed number of entities. Each document can still accumulate contributions from multiple entities, so its rank depends on their combined relevance and document frequencies. The weighting acts on entity frequency rather than normalizing by document length.

(20)