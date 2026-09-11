# MOSAIC: Query-Aware Exploration Policy Adaptation for GraphRAG

Technical Report

EunKyeong Lee, Kyeong-Jin Oh, Jinwon Kim, Hye Woo Lee Minsang Song, Hyeongjun Jang, Junyoung Youn

KT Corporation

September 2026

## Abstract

Graph Retrieval-Augmented Generation (GraphRAG) can connect evidence distributed across a corpus graph, but most systems execute a largely shared exploration procedure for every query. This creates a structural mismatch: a direct fact may require a compact local neighborhood, a comparison requires balanced coverage of multiple targets, and a mediated question may require a deeper path through a weakly query-related connector. We present Mosaic, a training-free framework that formulates GraphRAG retrieval as a per-query control problem. An LLM analyzer translates the evidence requirements implied by a query into a bounded, executable policy over seed selection, graph traversal, cumulative-gain stopping, and evidence selection. The corpus graph, indexes, scoring functions, grounding procedure, and answer generator remain shared. The current analyzer uses six representative considerations—Mediatedness, Output Cardinality, Seed Coverage, Anchoring Need, Identity–Relation Dependence, and Completeness—as composable reasoning signals rather than mutually exclusive query classes or a closed taxonomy.

On GraphRAG-Bench, Mosaic achieves query-weighted Answer Correctness of 76.97 on Medical and 64.33 on Novel, improving over the strongest previously reported overall results by 5.13 and 4.43 points, respectively. On Medical, it reaches 95.1 Evidence Recall and 86.1 Context Relevancy. Controlled comparisons on an identical graph and generator show that no fixed narrow, medium, or wide policy is consistently optimal; Mosaic improves by 9.96 points over the strongest canonical fixed policy. Relative to Fixed Wide, it evaluates 81.9% fewer paths and retains 47.2% fewer evidence items, although its separate analyzer call increases end-to-end latency. Transfer experiments on HotpotQA, MuSiQue, and 2WikiMultiHopQA further show that the policy interface can be applied without benchmark-specific retriever training. These results support analyzer-driven, query-specific exploration as a general and extensible design principle for GraphRAG.

## Technical-report scope

This report consolidates the method, implementation details, controlled analyses, transfer experiments, and qualitative cases in one document. Mosaic is not a six-rule retrieval system: it is an analyzer-driven framework for instantiating query-specific retrieval policies. The six considerations in the current implementation are representative and extensible design signals derived from observed retrieval failures.

## 1 Introduction

Retrieval-Augmented Generation (RAG) grounds language-model outputs in external evidence [9]. GraphRAG extends this idea by representing entities, relations, passages, or communities as a graph and retrieving connected evidence for questions that require relation following, multi-hop reasoning, or synthesis across sources [3, 4]. The graph provides useful structure, but it also creates a control problem: retrieval must decide where to start, how far and broadly to traverse, when to stop, and which paths to preserve.

These decisions are usually configured globally. The query changes similarity scores or initial nodes, yet the main operating limits—seed count, depth, width, stopping behavior, and final evidence budget— remain largely shared. A global policy is attractive because it is simple, but it assumes that heterogeneous questions require comparable evidence structures. That assumption is often false. A narrowly stated fact can be diluted by broad exploration; a comparison can fail if seeds cover only one target; and a mediated question can be unreachable under a shallow beam even when the answer-bearing region exists in the graph.

Mosaic replaces the search for one globally optimal configuration with explicit per-query policy construction. Before retrieval, an analyzer interprets the question and emits bounded controls for the shared pipeline. During retrieval, cumulative structural gain determines whether the current evidence has saturated, and role-aware selection can preserve complementary paths. Answer generation is held fixed: the analyzer changes what evidence is retrieved, not how the final answer is freely generated.

The contribution is therefore not the observation that queries influence retrieval; many prior systems are query-conditioned, and recent systems adapt routes, edges, constraints, or operators. The technical distinction is the representation and target of adaptation:

query −→ explicit multi-stage retrieval policy −→ shared corpus-graph pipeline.

The policy jointly controls seed breadth and formulation, traversal depth and width, anchor and connector behavior, stopping sensitivity, and evidence retention. This interface is training-free, validated before execution, and extensible to new evidence requirements.

Our main findings are:

• No fixed exploration scope is uniformly efective. On Medical, Fixed Medium reaches 67.01% ACC, while both narrower (65.45%) and wider (66.35%) exploration are worse.

• Query-specific control substantially improves retrieval and generation. Mosaic obtains 76.97% overall ACC on Medical and 64.33% on Novel, with Evidence Recall of 95.1% and 90.2%.

• The gain is not explained by exhaustive search. Relative to Fixed Wide, Mosaic evaluates 81.9% fewer paths and retains 47.2% fewer evidence items.

• The analyzer emits diverse and requirement-aligned policies. For example, connector traversal is activated for 65.6% of multi-hop queries but 21.6% of other queries.

• The framework transfers without benchmark-specific retriever training to three standard multi-hop QA benchmarks, while leaving room for stronger answer-format calibration.

## 2 Related Work

RAG combines parametric generation with non-parametric retrieval [9, 8]. GraphRAG systems retrieve entities, relations, paths, passages, or graph communities: Microsoft GraphRAG supports local and global community-oriented search [3]; LightRAG combines entity- and relation-level retrieval [4]; HippoRAG propagates query relevance through personalized PageRank [5]; and PathRAG prunes relational paths using flow-based signals [2]. Learned graph retrievers such as GFM-RAG and G-Reasoner encode textual and structural relevance with pretrained graph models [10, 11]. Adaptive textual RAG methods decide whether or when to retrieve [6, 1, 7]. Closer graph systems adapt the retrieval paradigm, query-side evidence graph, path constraints, or operator composition. EA-GraphRAG routes between dense and graph retrieval [12]; Relink constructs a query-driven evidence graph on the fly [13]; DOTRAG generates query-conditioned constraints for path exploration [14]; and PAGE-RAG composes heterogeneous evidence operators under bounded budgets [15]. Mosaic does not claim that query-aware GraphRAG is itself new. It difers by constructing an explicit stage-wise operating policy inside one shared corpus-graph retriever, jointly controlling seeds, traversal, stopping, and retained evidence without training an additional router or graph retriever.

Table 1: High-level location of query adaptation in representative systems. “Shared” means globally configured at inference time.
<table><tr><td>Method</td><td>ment</td><td>Query-dependent ele- Main adaptation tar- Shared element get</td><td></td><td>Extra training</td></tr><tr><td>GraphRAG / LightRAG</td><td>community relevance or candidates</td><td>Entity, relation, or Initial retrieval mode Expansion limits and</td><td>budgets</td><td>No</td></tr><tr><td>HippoRAG</td><td>tion</td><td>PPR reset distribu- Graph personaliza- Propagation rule and tion</td><td>top-k</td><td>No</td></tr><tr><td>G-Reasoner</td><td>representation</td><td>Learned query-graph Node/evidence scor- Learned inference ar- ing</td><td>chitecture</td><td>Yes</td></tr><tr><td>DOTRAG</td><td>Entity-type straints and path and iterative paths</td><td>con- Admissible subgraph Hop/iteration limits</td><td></td><td>No</td></tr><tr><td>PAGE-RAG</td><td>acceptance Query profile</td><td>tion and evidence- tations</td><td>Operator composi- Operator implemen-</td><td>No</td></tr><tr><td>Mosaic</td><td>Evidence ments</td><td>type budgets require- Joint seed, traver- Graph, indexes, dence policy</td><td>sal, stop, and evi- pipeline, generator</td><td>No</td></tr></table>

## 3 Problem Formulation

Let $G = ( V , E )$ be a corpus graph constructed once from a document collection, and let q be a user query. A conventional graph retriever applies a globally selected configuration π¯:

$$
\hat { y } _ { q } = \mathcal { M } _ { \mathrm { a n s } } ( q , \mathcal { R } ( G , q ; \bar { \pi } ) ) .
$$

The objective of Mosaic is to replace π¯ with a query-specific but bounded policy $\pi _ { q }$ while keeping G, the retrieval implementation $\mathcal { R } _ { : }$ , and the answer model ${ \mathcal { M } } _ { \mathrm { a n s } }$ shared:

$$
\pi _ { \boldsymbol { q } } = \boldsymbol { \mathcal { A } } ( \boldsymbol { q } ) = \left( \pi _ { \boldsymbol { q } } ^ { \mathrm { s e e d } } , \pi _ { \boldsymbol { q } } ^ { \mathrm { t r a v } } , \pi _ { \boldsymbol { q } } ^ { \mathrm { f i l t e r } } \right) .\tag{1}
$$

The policy is generated only from the question. Gold answers, gold evidence, benchmark category labels, and evaluation annotations are not available to the analyzer. All categorical outputs are checked against an allowlist, numerical values are clamped to valid ranges, and invalid or missing fields fall back to conservative defaults.

This formulation separates three concepts that are easily conflated:

1. Query conditioning: the query changes similarity or relevance scores.

2. Query classification: the query is assigned to one of several predefined types, each linked to a fixed pipeline.

3. Query-specific policy construction: the query is translated into composable stage-level controls executed by a common pipeline.

Table 2: Representative evidence-requirement signals and their operational efects. Multiple rows may apply to one query.
<table><tr><td>Signal</td><td>Analyzer question</td><td>Typical fixed-policy failure</td><td>Main controls</td></tr><tr><td>Mediatedness</td><td>Is the answer reachable only A weakly related mediator Depth, connector, anchor, through an unstated interme- is pruned before the answer- stop diate concept?</td><td>bearing region.</td><td></td></tr><tr><td>Output Cardinality</td><td>or must several items be cov- highest-scoring answer item. ered?</td><td>Is one precise value sufficient, Filtering preserves only the Seed/evidence budgets,</td><td>floor, reservation</td></tr><tr><td>Seed Coverage</td><td>Must several targets or facets Full-query seeding concentrates Per-target query mode, be initialized separately?</td><td>on the most salient target.</td><td>seed budget, balancing</td></tr><tr><td>Anchoring Need</td><td>Must exploration remain tied to an exact entity or subtype? ing but semantically similar pression, path mode</td><td>1 Traversal drifts to a neighbor- Anchor strength, hub sup- concept.</td><td></td></tr><tr><td>Identity-Relation Dependence</td><td>Is exact entity identity or rela- A plausible relation path con- Relation-only vs. hybrid tion semantics more decisive?</td><td>tains the wrong entity.</td><td>scoring</td></tr><tr><td>Completeness</td><td>partially missing, or globally</td><td>continues after useful evidence floor, expansion saturates.</td><td>Is current evidence sufficient, A fixed depth stops too early or Low-gain stop, evidence</td></tr></table>

Mosaic implements the third. Diagnostic labels may be logged for analysis or defensive fallback, but they do not select separate retrievers.

## 4 From Retrieval Failures to Policy Signals

We examined retrieval traces in which relevant evidence was present or reachable but the shared policy produced an incorrect answer. The recurring failures did not define six mutually exclusive question types. Instead, they exposed six questions the analyzer should ask when configuring retrieval. Table 2 summarizes the current instantiation.

The six signals are deliberately not a closed taxonomy. Another corpus, graph schema, domain, or pipeline may expose a new failure pattern. The framework can incorporate it by adding a reasoning cue and mapping it to an existing or newly introduced bounded control. The core contribution is the analyzer-to-policy interface, not the number or names of the present cues.

## 4.1 Why controls must be composed

Individual controls are not independent treatments. A comparison may simultaneously require per-target seeds, connector traversal, and a larger evidence budget. A mediated question may require greater depth but weaker anchoring so that an apparently generic connector is not pruned. Strong anchoring can reduce drift, yet the same anchor can block a necessary mediator. Similarly, stopping quality depends on the evidence produced by the seed and traversal stages. Therefore, control-specific counterfactuals are diagnostic rather than an additive decomposition of total performance.

## 5 MOSAIC

## 5.1 End-to-end architecture

![](images/b6659a9420e48a07a3ada0840ed28b06c93eef88526eb37e8e02638f982e85e9.jpg)  
Figure 1: Mosaic maps a query to one validated policy with stage-specific controls. Seed, traversal and stopping, and evidence controls configure a shared GraphRAG pipeline, while answer generation remains fixed.

Given a policy, the complete computation is

$$
S _ { q } = S ( G , q ; \pi _ { q } ^ { \mathrm { s e e d } } ) ,\tag{2}
$$

$$
P _ { q } = \mathcal { T } ( G , S _ { q } ; \pi _ { q } ^ { \mathrm { t r a v } } ) ,\tag{3}
$$

$$
\widetilde E _ { q } = \mathcal { F } ( q , P _ { q } ; \pi _ { q } ^ { \mathrm { f i l t e r } } ) ,\tag{4}
$$

$$
\begin{array} { r } { \widehat { y } _ { q } = \mathcal { M } _ { \mathrm { a n s } } ( q , \mathrm { G r o u n d } ( \widetilde { E } _ { q } ) ) . } \end{array}\tag{5}
$$

## 5.2 Policy space and validation

The analyzer starts from conservative defaults and overrides a control only when the question provides a clear signal. Table 3 reports the paper-run interface.

Table 3: Bounded policy controls.
<table><tr><td>Control</td><td>Valid values</td><td>Default</td><td>Function</td></tr><tr><td>seed_limit</td><td>integer 1-15</td><td>7</td><td>Deduplicated seed budget</td></tr><tr><td>seed_query_mode</td><td>full query, compact keywords, per- full query target keywords</td><td></td><td>Seed-query construction</td></tr><tr><td>selection_mode</td><td>relation-only, hybrid</td><td>relation-only</td><td>Entity/relation contribution</td></tr><tr><td>max_depth</td><td>integer 2–5</td><td>3</td><td>Hard traversal cap</td></tr><tr><td>beam_width</td><td>integer 4–12</td><td>8</td><td>Partial paths retained per depth</td></tr><tr><td>anchor_strength</td><td>real 0-1</td><td>0.0</td><td>Target focus and hub suppres- sion</td></tr><tr><td>use_connector</td><td>Boolean</td><td>false</td><td>Cross-seed connector explo- ration</td></tr><tr><td>stop_regime</td><td>shallow, medium, deep</td><td>shallow</td><td>Low-gain sensitivity</td></tr><tr><td>evidence_count</td><td>integer 5-18</td><td>8</td><td>Final evidence cap</td></tr><tr><td>min_evidence</td><td>integer 1-12</td><td>3</td><td>Evidence preservation floor</td></tr><tr><td>adaptive_final_evidence</td><td>Boolean</td><td>false</td><td>Role-aware path reservation</td></tr></table>

The analyzer also returns target entities, target facets, a short retrieval-risk description, and a rationale for diagnostics. These fields do not bypass policy validation. Unsupported categorical values are rejected, numeric fields are clamped, and missing values revert to defaults. This ensures that the LLM controls a finite interface rather than directly executing arbitrary code or inventing new retrieval operators.

## 5.3 Seed selection

The selected query mode searches fixed entity and relation indexes with the complete question, a compact query, or one query per extracted target. Let $s _ { \mathrm { e n t } } ( v , q _ { e } )$ and $s _ { \mathrm { r e l } } ( v , q _ { e } )$ denote entity- and relation-index scores for node v. Relation-only selection uses

$$
s _ { \mathrm { s e e d } } ( v , q ) = s _ { \mathrm { r e l } } ( v , q _ { e } ) ,\tag{6}
$$

while hybrid selection uses

$$
s _ { \mathrm { s e e d } } ( v , q ) = 0 . 3 s _ { \mathrm { e n t } } ( v , q _ { e } ) + 0 . 7 s _ { \mathrm { r e l } } ( v , q _ { e } ) .\tag{7}
$$

The seed set is $S _ { q } = \mathrm { T o p K } _ { K _ { q } } \{ v : s _ { \mathrm { s e e d } } ( v , q ) \}$ }. In per-target mode, the two best candidates for each target are preserved before node-level deduplication and final truncation. This prevents a comparison from allocating all seeds to its most salient entity.

## 5.4 Policy-conditioned traversal

Let $p = ( v _ { 0 } , e _ { 1 } , v _ { 1 } , \ldots , e _ { h } , v _ { h } )$ be a path with h edges. The structure-aware score combines relation similarity ${ \bar { r } } ,$ target relevance $\bar { t } ,$ specificity ${ \bar { s } } ,$ anchor consistency ${ \bar { a } } ,$ transition alignment ${ \bar { x } } ,$ path coherence ${ \bar { c } } ,$ hub exposure $\bar { h }$ , repetition $\rho _ { \mathrm { r e p } }$ , and question-like labels $\rho _ { \mathrm { q l } }$

$$
s _ { \mathrm { p a t h } } ( p , q ) = w _ { r } \bar { r } + w _ { t } \bar { t } + w _ { s } \bar { s } + w _ { a } \bar { a } + w _ { x } \bar { x } + w _ { c } \bar { c } - w _ { h } \bar { h } - w _ { \mathrm { r e p } } \frac { \rho _ { \mathrm { r e p } } } { h } - w _ { \mathrm { q l } } \rho _ { \mathrm { q l } } .\tag{8}
$$

Default weights are $( 0 . 4 5 , 0 . 1 5 , 0 . 1 5 , 0 . 1 0 , 0 . 1 5 , 0 . 0 5 , 0 . 2 0 , 1 . 0 , 1 . 0 )$ . When $a _ { q } \geq 0 . 4 .$ , target, anchor, transition, and hub terms increase with $\textstyle { a _ { q } , }$ changing a general relation-oriented scorer into a target- and transition-aware scorer. At depth $d ,$ all retained paths are expanded, connector-view candidates are added when enabled, and the best $W _ { q }$ paths are kept. $D _ { q }$ remains the hard cap.

Connector exploration is a second view, not a separate pipeline. Standard beam expansion grows outward from each seed; the connector view searches for paths that join regions initialized by distinct targets. Both views enter the same scoring and pruning pool.

## 5.5 Cumulative-gain stopping

Let $A _ { d - } ^ { N }$ <sub>1</sub> and $A _ { d - } ^ { E }$ <sub>1</sub> be nodes and edges accumulated before depth $d ,$ and $N _ { d } , E _ { d }$ the structure reached at the current depth. Novelty gain is

$$
g _ { d } = \frac { | N _ { d } \setminus A _ { d - 1 } ^ { N } | + | E _ { d } \setminus A _ { d - 1 } ^ { E } | } { \operatorname* { m a x } ( 1 , | A _ { d - 1 } ^ { N } | + | A _ { d - 1 } ^ { E } | ) } .\tag{9}
$$

After minimum depth $d _ { \operatorname* { m i n } } = 2$ , a step is eligible for early stopping when $g _ { d } \leq \epsilon _ { r }$ and cumulative evidence completeness $c _ { d } \geq 0 . 3 5$ . With patience one,

$$
\mathrm { S t o p } ( q , d ) = \mathbb { I } [ d \geq D _ { q } ] \vee \mathbb { I } [ d \geq 2 \wedge g _ { d } \leq \epsilon _ { r _ { q } } \wedge c _ { d } \geq 0 . 3 5 ] .\tag{10}
$$

The paper-run thresholds are 5.00, 0.35, and 0.24 for shallow, medium, and deep regimes. The large shallow threshold intentionally stops at the first completeness-eligible depth; it does not mean novelty is numerically small in an absolute sense.

## 5.6 Evidence selection and source grounding

After traversal, paths are deduplicated and ranked. A shared path filter operates on at most ten candidates. When adaptive reservation is disabled, selection uses the global top paths. When enabled, it first preserves up to four core paths, then reserves up to four additional positions for complementary roles: a connector path, an uncovered target, an uncovered facet, or a previously unseen first-relation label. Remaining positions are filled by path score.

The ranked paths are flattened into deduplicated evidence units. $B _ { q }$ caps retained evidence, while $L _ { q }$ restores high-ranked excluded units when selection falls below the preservation floor. Importantly, the evaluated Mosaic configuration does not add a separate LLM call for post-traversal noise filtering. Noise is controlled through policy-conditioned traversal, path scoring, cumulative-gain stopping, role-aware reservation, and local source-window reranking.

Graph evidence is mapped back to source chunks. Candidate chunks are divided into 1,600-character windows with a stride of 800. A local BAAI/bge-large-en-v1.5 bi-encoder ranks the windows by cosine similarity to the question, and the top 15 windows are passed to GPT-4o-mini at temperature zero. This preserves graph structure during exploration while returning full textual detail for answer generation.

```powershell
Algorithm 1: MOSAIC query-specific graph retrieval
Input: query $q ,$ corpus graph $G ,$ entity and relation indexes. Output: answer $\hat { y } _ { q } ,$ , grounded windows $W _ { q } .$
1. $z _ { q } \gets$ AnalyzeRequirements(q); π ← ValidateAndInstantiate $\left( z _ { q } \right)$
2. $S _ { q } \gets$ RetrieveSeeds $( q , \pi _ { q } ^ { \mathrm { s e e d } } )$ ; initialize $B _ { 0 }  S _ { q }$ and $P _ { q }  \emptyset .$
3. For $d = 1 , \ldots , \pi _ { q } . D _ { \mathrm { m a x } } \colon$ expand $B _ { d - 1 } ;$ rank and prune with $\pi _ { q } ^ { \mathrm { t r a v } } ;$ merge retained paths into $P _ { q } ;$ stop
when d $\geq 2$ and the policy-conditioned stopping test succeeds.
4. $E _ { q }$ ← SelectEvidence $( P _ { q } , q , \pi _ { q } ^ { \mathrm { f i l t e r } } )$ . If $| E _ { q } | < \pi _ { q } . L$ , restore the highest-ranked excluded evidence up
to the floor.
5. $W _ { q } \gets$ GroundToSource $( E _ { q } , q ) ; \hat { y } _ { q }$ ← GenerateAnswer $( q , W _ { q } ) ;$ return $( \hat { y } _ { q } , W _ { q } )$
```

## 6 Experimental Setup

Primary benchmarks. GraphRAG-Bench Medical contains 2,062 questions over 2,406 medicalguideline documents; Novel contains 2,010 questions over 461 literary documents. Both contain Fact Retrieval (FR), Complex Reasoning (CR), Contextual Summarization (CS), and Creative Generation (CG). We use every question and the original corpora.

Metrics. Answer Correctness (ACC) is the primary end-to-end metric because it is defined for all four categories. Overall ACC is query-count weighted, not the unweighted mean of category scores. Retrieval is evaluated with Evidence Recall (ER) and Context Relevancy (CR). Task-specific ROUGE-L, coverage, and faithfulness are reported as secondary metrics.

Implementation. Each corpus graph is constructed once with a LightRAG-style entity–relation extraction pipeline. OpenAI text-embedding-3-large supplies semantic representations. GPT-4o-mini at temperature 0 is used for the analyzer and answer generation. The graph, indexes, retrieval code, and generator are shared by Mosaic and fixed-policy controls. Mosaic requires no additional training or fine-tuning.

Baselines. We compare with reported GraphRAG-Bench systems and with AutoPrunedRetriever and G-Reasoner. Because published baselines were not rerun in our implementation, claims involving them are comparisons to externally reported values. To isolate policy adaptation, we additionally implement Fixed Narrow, Medium, Wide, and a Budget-Matched Fixed policy on the identical graph and generator.

Table 4: Answer Correctness (%) on GraphRAG-Bench. Overall is query-count weighted. Best values are bold.
<table><tr><td></td><td colspan="5">Medical</td><td colspan="5">Novel</td></tr><tr><td>Method</td><td>Overall</td><td>FR</td><td>CR</td><td>CS</td><td>CG</td><td>Overall</td><td>FR</td><td>CR</td><td>CS</td><td>CG</td></tr><tr><td>RAG w/ rerank</td><td>63.04</td><td>64.73</td><td>58.64</td><td>65.75</td><td>60.61</td><td>52.97</td><td>60.92</td><td>42.93</td><td>51.30</td><td>38.26</td></tr><tr><td>HippoRAG2</td><td>64.91</td><td>66.28</td><td>61.98</td><td>63.08</td><td>68.05</td><td>58.41</td><td>60.14</td><td>53.38</td><td>64.10</td><td>48.28</td></tr><tr><td>AutoPruned (LLM)</td><td>65.35</td><td>61.25</td><td>71.59</td><td>70.14</td><td>65.02</td><td>58.34</td><td>45.99</td><td>62.80</td><td>83.10</td><td>62.97</td></tr><tr><td>G-Reasoner</td><td>71.84</td><td>68.84</td><td>75.17</td><td>77.23</td><td>72.04</td><td>59.90</td><td>60.07</td><td>53.92</td><td>71.28</td><td>50.48</td></tr><tr><td>Mosaic</td><td>76.97</td><td>76.05</td><td>76.92</td><td>85.28</td><td>68.78</td><td>64.33</td><td>66.06</td><td>57.21</td><td>73.12</td><td>56.54</td></tr></table>

Table 5: Retrieval performance (%). ER: Evidence Recall; CR: Context Relevancy.
<table><tr><td colspan="3">Medical</td><td colspan="2">Novel</td></tr><tr><td>Method</td><td>ER</td><td>CR</td><td>ER</td><td>CR</td></tr><tr><td>RAPTOR</td><td>84.2</td><td>62.6</td><td>66.1</td><td>58.0</td></tr><tr><td>LightRAG</td><td>82.6</td><td>42.2</td><td>79.6</td><td>35.5</td></tr><tr><td>HippoRAG2</td><td>73.6</td><td>85.3</td><td>66.2</td><td>82.8</td></tr><tr><td>G-Reasoner</td><td>93.8</td><td></td><td>87.7</td><td></td></tr><tr><td>Mosaic</td><td>95.1</td><td>86.1</td><td>90.2</td><td>77.3</td></tr></table>

## 7 Results

## 7.1 Answer correctness

Mosaic obtains 76.97 on Medical and 64.33 on Novel, improving over the strongest previously reported overall results by 5.13 and 4.43 points. On Medical it leads FR, CR, and CS; it does not lead CG. On Novel, it leads overall and FR but trails specialized systems on CR, CS, and CG. We therefore interpret the result as consistent overall accuracy across heterogeneous evidence requirements, not uniform dominance on every task category.

## 7.2 Retrieval quality

Mosaic achieves the highest Evidence Recall on both domains, exceeding G-Reasoner by 1.3 points on Medical and 2.5 points on Novel. It also achieves the highest reported Medical Context Relevancy. Novel reveals a useful trade-of: HippoRAG2 has higher Context Relevancy, but substantially lower Evidence Recall.

## 7.3 Why one fixed policy is insuficient

The relationship between scope and accuracy is non-monotonic. Medium improves on Narrow, but Wide declines despite more seeds, deeper search, a larger beam, and more evidence. Fixed Wide helps CS but hurts FR, CR, and CG relative to Medium. Benchmark category also does not uniquely determine the correct policy: some fact questions require mediated evidence, while some summaries are supported by a compact neighborhood. Mosaic exceeds the strongest canonical fixed policy by 9.96 points, showing that the key benefit is query-level allocation rather than choosing a single larger operating point.

Table 6: Controlled fixed-policy comparison on Medical. All systems share graph and generator.
<table><tr><td>Policy</td><td>Seeds</td><td>Depth</td><td>Width</td><td>Evidence</td><td>FR</td><td>CR</td><td>CS</td><td>CG</td><td>Overall</td></tr><tr><td>Narrow</td><td>3</td><td>2</td><td>4</td><td>1-3</td><td>.620</td><td>.674</td><td>.733</td><td>.686</td><td>.6545</td></tr><tr><td>Medium</td><td>5</td><td>4</td><td>8</td><td>5-8</td><td>.636</td><td>.683</td><td>.758</td><td>.703</td><td>.6701</td></tr><tr><td>Wide</td><td>10</td><td>5</td><td>16</td><td>12-15</td><td>.630</td><td>.672</td><td>.765</td><td>.683</td><td>.6635</td></tr><tr><td>MOSAIC</td><td></td><td>adaptive per query</td><td></td><td></td><td>.761</td><td>.769</td><td>.853</td><td>.688</td><td>.7697</td></tr></table>

![](images/c69c576e0c2a22d153d05609d5bbf7607c3c0a786c31b2b410a20fd052a44b67.jpg)

![](images/a9357f4482104ef28ea29fbd28b861245c03aaf1832b628e81ff6bb4750e061b.jpg)

![](images/f3dde354ae6ec8007572ea0007a01f4c4e30b37f237048b5f647bdee344c3160.jpg)

![](images/265d2f9f4360ff6979d7c91e991e493444d9b7e9f039ffeb43d04e841ce16881.jpg)  
(a) Policy diversity.

![](images/d6799fa7006dd4175d2bd565f1772bb099b6c834bc05be3df79332afc03f4268.jpg)  
(b) Requirement–behavior alignment.  
Figure 2: Analyzer outputs on 2,062 Medical questions. Error bars in (b) are 95% Wilson intervals.

## 7.4 Policy diversity and requirement alignment

Across 2,062 Medical queries, the analyzer produces 19 unique policy combinations. Assigned depth is distributed across 2, 3, and 4 hops for 26%, 43%, and 31% of queries. Seed count ranges from 5 to 12; evidence budgets range from 8 to 16. Coupled connector/two-view exploration is activated for 32%, adaptive evidence selection for 35%, hybrid identity–relation scoring for 19%, and per-target seeding for 19%. This rules out collapse to a single dominant configuration.

Independent post-hoc proxies further show that variation is purposeful. Two-view traversal is activated for 90.5% of multi-target queries versus 30.7% of others; a large evidence budget for 58.4% of multi-answer queries versus 12.1%; connector traversal for 65.6% of multi-hop queries versus 21.6%; shallow traversal for 48.6% of local/direct queries versus 0.8%; and early-stop behavior for 57.2% of closed-form queries versus 10.8%. The proxies use deterministic surface, gold-answer-structure, and benchmark-label rules only for analysis; they are never provided to the analyzer.

## 7.5 Diagnostic counterfactuals

These counterfactuals measure the operational efect of a control while retaining the full analyzer. They should not be summed: the same query can improve through several interacting controls. The strongest single count belongs to output cardinality, consistent with the tendency of high-scoring paths to crowd out sibling evidence needed for list-like answers.

Table 7: Queries improved by the complete policy relative to disabling the corresponding execution control. Counts are diagnostic and non-additive.
<table><tr><td>Consideration</td><td>Improved</td><td>Consideration</td><td>Improved</td></tr><tr><td>Mediatedness</td><td>48</td><td>Output cardinality</td><td>190</td></tr><tr><td>Seed coverage</td><td>55</td><td>Anchoring need</td><td>46</td></tr><tr><td>Identity-relation dependence</td><td>41</td><td>Completeness</td><td>47</td></tr></table>

Table 8: Accuracy and average per-query latency on Medical. ACC uses all 2,062 queries; latency uses the same 197-query subset.
<table><tr><td>Method</td><td>ACC</td><td>Retrieval</td><td>Analyzer</td><td>Generation</td><td>End-to-end</td></tr><tr><td>Fixed Narrow</td><td>65.45</td><td>3.27</td><td></td><td>1.76</td><td>5.03</td></tr><tr><td>Fixed Medium</td><td>67.01</td><td>3.50</td><td></td><td>1.79</td><td>5.29</td></tr><tr><td>Fixed Wide</td><td>66.35</td><td>4.81</td><td></td><td>1.91</td><td>6.72</td></tr><tr><td>Budget-matched Fixed</td><td>72.03</td><td>3.63</td><td></td><td>2.26</td><td>5.89</td></tr><tr><td>MOSAIC</td><td>76.97</td><td>4.67</td><td>2.82</td><td>2.19</td><td>9.68</td></tr></table>

## 7.6 Latency and graph-search efort

Mosaic is not faster end-to-end in the current implementation. Its separate policy-construction call adds 2.82 seconds on average, producing 9.68 seconds total versus 5.29 for Fixed Medium. The relevant eficiency result is diferent: additional LLM computation is used to allocate much less graph search more efectively. The analyzer and seed-keyword extraction are currently separate and unbatched; caching, prompt compression, call fusion, or distillation could reduce control latency without changing the retrieval interface.

## 8 Transfer to Multi-Hop QA

We evaluate the same policy space on 1,000-question subsets of HotpotQA, MuSiQue, and 2WikiMultiHopQA. No benchmark training split or task-specific retriever fine-tuning is used. One or two short answer-format instructions are added for each benchmark.

Mosaic is close to the trained reference on HotpotQA and exceeds it by 1.8 EM on MuSiQue, while a substantial gap remains on 2WikiMultiHopQA. A separate LLM-based semantic correctness evaluation yields 77.9, 51.3, and 75.1, suggesting that lexical EM/F1 penalize some semantically correct but diferently formatted answers. Because the systems were not rerun under a common pipeline and G-Reasoner uses benchmark-specific supervision, these results support transfer but do not establish a state-of-the-art claim.

## 9 Qualitative Analysis

The successful cases show why controls must be combined. More depth alone does not recover the kidney-tumor answer because the larger context can still be dominated by a nearby but incorrect staging interpretation. The failure case is equally important: an expressive controller can overreact. Guardrails constrain the output range, but they cannot guarantee that the evidence requirement itself is inferred correctly.

Table 9: Graph-search efort relative to Fixed Wide.
<table><tr><td>Metric</td><td>Fixed Wide</td><td>MOSAIC</td><td>Reduction</td></tr><tr><td>Evaluated paths</td><td>800</td><td>145</td><td>81.9%</td></tr><tr><td>Reached depth</td><td>5.00</td><td>2.59</td><td>48.2%</td></tr><tr><td>Beam width</td><td>16.0</td><td>8.0</td><td>50.0%</td></tr><tr><td>Selected paths</td><td>16.0</td><td>9.4</td><td>41.3%</td></tr><tr><td>Retained evidence items</td><td>12.35</td><td>6.52</td><td>47.2%</td></tr></table>

Table 10: Transfer results (%). G-Reasoner values are externally reported and use target-benchmark training; Mosaic is not trained on these benchmarks.
<table><tr><td></td><td colspan="2">MOSAIC</td><td colspan="2">G-Reasoner</td><td colspan="2">Difference</td></tr><tr><td>Dataset</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td>HotpotQA</td><td>61.1</td><td>75.3</td><td>61.4</td><td>76.0</td><td>-0.3</td><td>-0.7</td></tr><tr><td>MuSiQue</td><td>40.3</td><td>51.2</td><td>38.5</td><td>52.5</td><td>+1.8</td><td>-1.3</td></tr><tr><td>2WikiMultiHopQA</td><td>62.3</td><td>70.0</td><td>74.9</td><td>82.1</td><td>-12.6</td><td>-12.1</td></tr></table>

## 10 Discussion

## 10.1 What the results establish

The controlled experiments establish that query-specific policy adaptation improves over several globally fixed operating points on a shared graph and generator. Policy logs show that the controller neither collapses to one configuration nor varies arbitrarily: activations align with independent requirement proxies. The graph-search measurements show that gains are not produced by uniformly retrieving more evidence.

## 10.2 What the results do not establish

The experiments do not isolate a causal contribution for every prompt phrase or every low-level scoring term. Controls interact, and the counterfactual counts are not additive. Comparisons to published GraphRAG systems are not component-matched reproductions. Transfer results compare against external G-Reasoner numbers under asymmetric training conditions. Finally, the current latency reflects an unoptimized separate analyzer call.

## 10.3 Extensibility and deployment

The bounded policy schema ofers a practical separation of concerns. Retrieval engineers define safe controls and defaults; the analyzer maps natural-language evidence requirements to those controls; monitoring uses policy logs and retrieval traces. New failure patterns can be addressed without introducing a new end-to-end pipeline for each query type. In production, policies can be cached for recurring query structures, distilled into a smaller controller, or constrained further by domain rules. Because the answer model consumes grounded source windows and remains outside the controller’s free-form action space, the framework also supports clearer auditing of how a query changed retrieval.

Table 11: Representative controlled cases from Medical. Percentages denote ACC.
<table><tr><td rowspan=1 colspan=3>Case                     Question                                 Outcome                      Retrieval explanation</td></tr><tr><td rowspan=1 colspan=3>Bridge recovery        How are multiple tumors in the kid- MosAIC 100%; Narrow  Depth 5, 12 seeds, and</td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>breast-cancer evidence.</td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>Target anchoring retains</td></tr><tr><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>the MGZL diagnostic</td></tr><tr><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>context.</td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>Role-aware reservation</td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>preservesastrocytoma,</td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>oligodendroglioma, and</td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>glioblastoma    instead</td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>of letting grade-related</td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>paths crowd out sibling</td></tr><tr><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>subtype relations.</td></tr><tr><td></td><td rowspan=1 colspan=1>MosAIC 24%; Narrow</td><td rowspan=1 colspan=1>The analyzer incorrectly</td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>predicts       mediation;</td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>depth-5 connector explo-</td></tr><tr><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ration introduces generic</td></tr><tr><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>surgery and care-team</td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>nodes. This is a policy-</td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>calibration failure, not</td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>graph unreachability.</td></tr></table>

## 11 Limitations

First, the analyzer is an LLM and can misclassify evidence structure, as the SLNB case demonstrates. Second, the current signals and valid ranges were developed from observed failures in the evaluated setting; transfer to diferent graph schemas may require additional calibration. Third, graph construction quality bounds retrieval: missing or incorrectly merged entities and relations cannot be repaired solely by a better policy. Fourth, answer quality still depends on source grounding and generation, particularly for open-ended CG questions where Mosaic does not uniformly improve. Fifth, latency and monetary cost are higher than fixed retrieval in the current implementation. Finally, some comparisons rely on externally reported baselines, and a fully component-matched reproduction across all methods remains future work.

## 12 Conclusion

Mosaic treats GraphRAG retrieval as a per-query control problem. An analyzer converts evidence requirements into a bounded policy spanning seed selection, traversal, stopping, and evidence retention, while the corpus graph and answer generator remain shared. The approach improves overall answer correctness and evidence recall on GraphRAG-Bench and uses substantially less graph search than a uniformly wide policy. Its six current considerations are composable implementation signals, not a fixed taxonomy. The broader result is that efective GraphRAG should adapt not merely relevance scores, but

the operating policy of graph exploration itself.

## Artifact and Reproducibility Statement

The evaluated implementation contains proprietary components. For result verification, the authors intend to provide benchmark configurations, dependencies, graph construction and retrieval code required for reproduction, inference scripts, generated outputs, and oficial evaluation commands through a controlled-access repository, subject to organizational approval and applicable benchmark licenses.

## References

[1] A. Asai et al. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. ICLR, 2024.

[2] B. Chen et al. PathRAG: Pruning Graph-Based Retrieval Augmented Generation with Relational Paths. AAAI, 2026.

[3] D. Edge et al. From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv:2404.16130, 2024.

[4] Z. Guo et al. LightRAG: Simple and Fast Retrieval-Augmented Generation. arXiv:2410.05779, 2024.

[5] B. Gutiérrez et al. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. NeurIPS, 2024.

[6] Z. Jiang et al. Active Retrieval Augmented Generation. EMNLP, 2023.

[7] S. Jeong et al. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity. NAACL, 2024.

[8] V. Karpukhin et al. Dense Passage Retrieval for Open-Domain Question Answering. EMNLP, 2020.

[9] P. Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS, 2020.

[10] L. Luo et al. GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation. NeurIPS, 2025.

[11] L. Luo et al. G-Reasoner: Foundation Models for Unified Reasoning over Graph-Structured Knowledge. arXiv preprint, 2025.

[12] S. Dong, Q. Zhang, Y. Xiao, S. Chen, C. Zhou, and X. Huang. Use Graph When It Needs: Eficiently and Adaptively Integrating Retrieval-Augmented Generation with Graphs. arXiv:2602.03578, 2026.

[13] M. Huang, C. Bu, Y. He, X. Zhuo, and X. Wu. Relink: Constructing Query-Driven Evidence Graph On-the-Fly for GraphRAG. arXiv:2601.07192, 2026.

[14] L. Moore, N. Deng, R. Mihalcea, and F. Jahanbakhsh. DOTRAG: Retrieval-Time Reasoning Along Paths. arXiv:2605.18760, 2026.

[15] X. Chen, J. An, J. Guo, and L. Wang. PAGE-RAG: Evidence-Grounded Adaptive Graph Retrieval for Long-Document Question Answering. arXiv:2607.19301, 2026.

[16] Y. Xiao, J. Dong, C. Zhou, S. Dong, Q. Zhang, D. Yin, X. Sun, and X. Huang. GraphRAG-Bench: Challenging Domain-Specific Reasoning for Evaluating Graph Retrieval-Augmented Generation. arXiv:2506.02404, 2025.