# SQLMorph: Query Mutation and Fine-Grained Metrics for Text-to-SQL Evaluation

Mohammadhossein Malekpour, Mohamed Riahi, Maxime Lamothe, Amine Mhedhbi

Polytechnique Montreal´

{mohammadhossein.malekpour, mohamed.riahi, maxime.lamothe, amine.mhedhbi}@polymtl.ca

Abstract—Text-to-SQL systems translate natural language queries into executable SQL, democratizing access to structured data. Despite recent advances driven by large language models (LLMs), evaluation remains a major bottleneck: public benchmarks fail to capture the complexity of enterprise schema, while building private evaluation sets is costly and nondeterministic, making evaluation results difficult to reproduce. To address this issue, we present SQLMorph, a framework for Text-to-SQL evaluation via query mutation. SQLMorph introduces two techniques to automatically generate and expand evaluation sets: Join Query Expansion (JQE), which systematically increases structural complexity through valid join additions, and Textual Query Augmentation (TQA), which generates controlled natural language perturbations to assess robustness to linguistic variation. JQE and TQA create targeted choke points to challenge specific system components. When applied to state-of-the-art systems, JQE increases query coverage and reveals accuracy degradation as the number of joins grows. Meanwhile, TQA shows that linguistic brittleness induced by heavy abbreviation can reduce accuracy by up to 17%.

Beyond evaluation sets, SQLMorph introduces a family of execution-level metrics that address the limitations of current binary measures, such as Execution Accuracy. We define Execution Precision (EXP) and Execution Recall (EXR) to quantify the fraction of correct and recovered results, respectively, and combine them via F1 for unified scoring. Our experiments show that these relaxed metrics enable fine-grained analysis of overand under-prediction, revealing differences across systems that binary metrics obscure. Together, SQLMorph’s query mutation and fine-grained metrics support debugging and better align Text-to-SQL evaluation practices with real-world deployments.

Index Terms—Text-to-SQL, Evaluation, SQL, Natural Language, Metrics, Expansion, Augmentation.

## I. INTRODUCTION

Natural language interfaces to databases aspire to democratize data access. They allow users to express information needs in natural language (NL) and obtain answers directly from structured data. At the core of this vision lies the Textto-SQL task, which translates NL queries into executable SQL. While the task has witnessed remarkable progress [1], the advent of large language models (LLMs) has substantially improved translation accuracy and spurred the development of new techniques. This momentum has, in turn, driven the rapid emergence of benchmarks such as Spider [2] and BIRD [3], and enabled early, controlled production deployments.

Despite this progress, current evaluation practices remain far from capturing the complexity of real-world enterprise workloads. Enterprise queries often span multiple tables, exhibit intricate join structures, and refer to schema elements with domain-specific terminology and abbreviations. In contrast, public benchmarks typically feature short, structurally simple queries defined over intuitive schema names. This leads to an overly optimistic assessment of system capabilities and thereby distorts cross-system comparisons. Recent efforts such as the Beaver benchmark [4] seek to mitigate these limitations by introducing more complex schemas. Nevertheless, progress remains incremental and labor-intensive, advancing only through the development of new benchmarks one at a time.

Evaluating Text-to-SQL systems on private datasets provides a more realistic alternative. Ultimately, the choice of a production Text-to-SQL system should depend on its performance on the target workload. However, constructing high-quality evaluation sets over private data is expensive, requiring substantial manual annotation, domain expertise, and increasingly, LLM-based automation. These processes are nondeterministic and difficult to control, making reliable and reproducible evaluation challenging. As a result, evaluation in both public and private settings remains fragmented, costly, and hard to scale.

Evaluation methodology is further constrained by the metrics used to measure system performance. The dominant standard, Execution Accuracy (EX), assigns a score of 1 when the predicted query’s output relation exactly matches that of the ground-truth query and 0 otherwise. While intuitive, this binary measure collapses a wide spectrum of outcomes into a single judgment, failing to capture partial correctness or provide diagnostic feedback. A query that retrieves nearly all correct tuples but misses one condition is indistinguishable from one that returns an irrelevant result. Similarly, structural differences such as column reordering or the inclusion of irrelevant columns are penalized equally. Consequently, EX provides only a coarse, outcome-based view of correctness. This coarseness leaves evaluation misaligned with real-world requirements, where understanding the degree to which a system errs and identifying which components of a Text-to-SQL system succeed or fail are essential.

These challenges stem from a common limitation: current evaluation practices remain static, relying on fixed benchmark queries in the form of NL–SQL pairs and on coarse binary metrics. To address these challenges, we reconceptualize Textto-SQL evaluation as a dynamic process and introduce SQL-Morph, a new evaluation framework. SQLMorph automatically generates valid query variants to create choke points and stress test system components, and introduces new execution-level metrics that move beyond binary correctness. These metrics are fine-grained and intended for offline development and evaluation to support diagnosis of over- and under-prediction, where ground-truth queries and outputs are available.

## A. Contributions

## • Join Query Expansion (JQE) (Section §IV)

We introduce Join Query Expansion (JQE), an algorithmic technique for increasing SQL complexity by expanding the set of joined tables in a query. JQE performs semantic validation to ensure that each expansion introduces genuine additional reasoning complexity, for example, by requiring joins that cannot be inferred transitively. It further applies diversity-aware pruning to maximize structural coverage across the generated variants. The resulting queries remain executable while becoming substantially more challenging. On BIRD, JQE doubles the average join degree and increases join cyclicity by an order of magnitude, leading to an Execution Accuracy (EX) drop of up to 20% for state-of-the-art systems. Overall, JQE stresses the SQL generation component along the dimension of join-intensive query construction.

## • Textual Query Augmentation (TQA) (Section §V)

We propose Textual Query Augmentation (TQA), a method for modifying schema and natural-language naming to evaluate robustness to textual perturbations. TQA systematically transforms schema elements or NL queries into less natural, abbreviation-heavy forms (e.g., WaterTemperature → WtTp) while preserving semantics and executability. The resulting variants remain semantically faithful yet expose substantial sensitivity to naming naturalness, with an Execution Accuracy (EX) drop of up to 17% across systems. TQA therefore probes linguistic robustness and context understanding, both of which are central to practical Text-to-SQL deployment.

## • Fine-Grained Execution Metrics (Section §VI)

We introduce two fine-grained execution-level metrics, Execution Precision (EXP) and Execution Recall (EXR), to complement the standard Execution Accuracy (EX). EXP measures the fraction of predicted tuples that are correct, while EXR measures the fraction of groundtruth tuples that are recovered. Their harmonic mean, F1, provides a unified summary, while the individual metrics expose over-prediction and under-prediction separately. Together, these metrics provide a more diagnostic view of system behaviour and reveal differences between systems that EX alone obscures.

These contributions establish a framework for evaluation set construction through query mutation andfine-grained metrics. To support reproducibility, we release the SQLMorph framework and all experimental scripts as open-source software.<sup>1</sup>

## II. BACKGROUND

Modern Text-to-SQL systems typically rely on a multistage pipeline [5], [6], [7], [8], [9], as illustrated in Fig. 1. A central design principle of SQLMorph is to create targeted choke points that systematically challenge specific components of this pipeline. Understanding these components is therefore essential to interpreting our evaluation results.

![](images/45bb5b7438e857883d7dca31719edaf5931a13abf49a33a707f19a40a3bea239.jpg)  
Fig. 1. Text-to-SQL pipeline.

1) Retrieval: This stage collects the contextual information needed to translate an NL query into SQL. This context may include schema elements such as tables, columns, and data types, representative database values, external documentation such as column descriptions, query-writing instructions, and demonstration examples [10], [11], [12]. Retrieval methods are commonly divided into two categories: filtering, which narrows the schema and candidate values to those most relevant to the query, and augmentation, which enriches the prompt with additional useful context. Typical augmentation operations include entity extraction and the retrieval of explanatory passages through vector- or keyword-based search [13], [14].

2) Generation: Given the NL query and retrieved context, the generation stage has two main substeps: it first produces one or more candidate SQL queries and then selects a final output. Candidate generation methods include few-shot prompting, intermediate representations such as NatSQL [15], plan decomposition for complex queries, and sampling strategies that vary prompt templates or decoding temperature [16], [17], [18]. The selection step then identifies the final SQL query using techniques such as self-consistency voting, rule-based filtering, and reranking. In practice, generation and selection are often interleaved in an iterative loop that runs for up to k rounds, where feedback from the selection step guides the next round of candidate generation [19].

3) Correction: The correction stage aims to repair remaining errors in the selected SQL query. Common techniques include execution-based feedback, which executes queries to identify issues such as syntax errors or empty results and then revises them accordingly, and model-based feedback, which relies on auxiliary critic models or unit-test-style checks to detect and fix errors. Many systems apply correction iteratively, progressively refining plausible candidates into executable and semantically correct SQL queries [20].

Some systems include additional stages to refine generation through user feedback [21] or to reduce cost via LLM routing [22]. SQLMorph is designed to selectively target such stages through query-based mutations.

## III. EXPERIMENTAL SETUP

This section outlines the experimental setup used to evaluate SQLMorph in Sections IV–VI. All of our experiments aim to reflect a realistic, medium-token-budget setting, with an explicit emphasis on minimizing cost for both generation and potential integration into CI/CD pipelines. We view our parameter defaults as a cost-conscious choice rather than a universally optimal setting, since the appropriate trade-off depends on the user’s evaluation goals and resources.

1) Dataset: Our experiments use the BIRD benchmark [3], a large-scale cross-domain dataset that has become a de facto standard for Text-to-SQL evaluation. BIRD contains 12,751 NL–SQL pairs across 95 databases spanning 37 domains. Each NL query may include evidence, i.e., external text that explains schema attributes, values, or computations, offering richer context for queries within specific domains.

The dataset is divided into three splits: training (9,428 queries over 69 databases), development (dev) (1,534 queries over 11 databases), and hidden test (1,789 queries over 15 databases). We conduct all of our analyses on the dev set.

2) Models: SQLMorph includes a model manager component that supports a range of backends, including OpenAI, Hugging Face, and Ollama, enabling users to pin model versions and, when desired, to run local open-source models for reproducibility and to control model drift. However, for the experiments in this paper, unless otherwise specified, we use GPT-4o and text-embedding-3-small from OpenAI as our models of choice.

3) Metrics: For now, we adopt EX as the primary evaluation metric. EX compares the output relation of a predicted query $\hat { R } _ { i }$ with that of the ground-truth query $R _ { i } ,$ assigning a score of 1 for an exact match and 0 otherwise. The EX score on N queries is computed as follows:

$$
E X = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } \mathbf { 1 } ( R _ { i } = { \hat { R } } _ { i } ) { \mathrm { ~ w h e r e ~ } } \mathbf { 1 } ( R _ { i } , { \hat { R } } _ { i } ) = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f ~ } } R _ { i } = { \hat { R } } _ { i } } \\ { 0 } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. }
$$

While this metric is standard, later sections (§VI) introduce SQLMorph’s fine-grained alternatives that are relaxed to capture partial correctness and aid in error diagnosis.

4) Text-to-SQL Systems: We consider three representative open-source Text-to-SQL systems that differ in architecture as well as their retrieval and correction strategies. Table I summarizes their reported performance on the BIRD leaderboard at the time of their submission. They are described below:

1. DIN-SQL [10] performs schema retrieval to align NL mentions with tables, columns, and values, then classifies queries as easy, non-nested complex, or nested complex to adapt prompting strategies. It employs the intermediate representation NatSQL [15] and applies a final self-correction stage via an LLM invocation.

2. MAC-SQL [11] follows the three-stage pipeline from Section II, while adding a decomposer to break complex NL queries into subproblems, generate subqueries for each during the generation stage, and then merge them.

3. CHESS [13]. It augments the pipeline with an information retriever (IR) that extracts keywords, values, and contextual hints from the NL query and schema. Its candidate generator (CG) repairs syntax errors, while a unit tester (UT) executes candidates and selects the one passing the most validation checks. We adopt $\mathrm { C H E S S } _ { \mathrm { I R + C G + U T } } ,$ the strongest reported configuration.

TABLE I  
PERFORMANCE AS EXECUTION ACCURACY (EX) OF THREE OPEN-SOURCE TEXT-TO-SQL SYSTEMS ON THE BIRD BENCHMARK AT THEIR TIME OF SUBMISSION.
<table><tr><td colspan="3">System Dev</td><td>Test</td></tr><tr><td>Human (upper bound)</td><td></td><td>一</td><td>93.0</td></tr><tr><td>CHESSIR+CG+UT</td><td>[13]</td><td>68.3</td><td>71.2</td></tr><tr><td>MAC-SQL</td><td>[11]</td><td>57.6</td><td>59.6</td></tr><tr><td>DIN-SQL</td><td>[10]</td><td>50.7</td><td>55.9</td></tr></table>

This setup allows us to evaluate the three primary contributions of SQLMorph: Join Query Expansion, Textual Query Augmentation, and Fine-Grained Execution Metrics.

## IV. JOIN QUERY EXPANSION

## A. Overview

Enterprise workloads frequently contain multi-table queries with complex join structures; yet public Text-to-SQL benchmarks underrepresent such patterns. Join Query Expansion (JQE) within SQLMorph addresses this gap by deterministically increasing a query’s structural complexity through the addition of valid joins.

Given an NL–SQL pair, JQE produces one or more expanded variants by adding a new table and valid join conditions. By iterating n times over an evaluation set, JQE bootstraps queries with up to n additional joins. Thus, applying JQE to an evaluation set can generate harder queries from only a small seed subset.

Intuition (running example): Suppose the original query retrieves customer names and total spending:

```sql
SELECT c.name, SUM(p.amount)
FROM Customer c
JOIN Purchase p ON c.id = p.cid
GROUP BY c.name
```

If the schema contains Purchase(pid, cid, sku), Product(sku, price, category), and Customer (id, . . . ), JQE may expand the join query graph (JQG) by adding Product via p.sku = Product.sku, optionally exposing new attributes (e.g., grouping by category) while maintaining the original intent and adding more information to the output. This stresses generation under larger join graphs (more tables, new predicates, and possibly cyclic JQGs) without resorting to free-form LLM generation.

## B. Design Principles

Two principles guide JQE’s design: (i) determinism and limited hallucination: the core expansion primitive is algorithmic and graph-driven rather than generative, and LLMs are used only at the end to regenerate an NL query consistent with the expanded SQL; and (ii) user control: users can constrain generation via diversity-aware pruning and preferences over join topology and coverage. We define coverage in our join expansion context as introducing new join patterns, i.e., JQGs that are non-isomorphic to those in an existing evaluation set. Finally, we ensure that each added join is not redundant, i.e., contributes genuine structural complexity. Specifically, the added join must not be implied through transitivity by the existing joins, and must therefore introduce at least one new explicit join condition in the SQL query.

![](images/7d155270572692c75aaff211f4146d8653a5e2c08973853cd3dfb97f03594f54.jpg)  
Fig. 2. Multi-stage pipeline for join query expansion.

## C. Approach

Our approach to JQE is divided into three parts:

1) Input:

• Schema graph. A graph $G _ { S } = ( V _ { S } , E _ { S } )$ whose vertices are tables and whose edges encode joinability. Each edge $e = ( t _ { i } , t _ { j } ) \in E _ { S }$ carries one or more labels $L ( e )$ , where each $\ell \in L ( e )$ denotes an admissible equi-join predicate $( e . g . , \ t _ { i } . a \ = \ t _ { j } . b )$ derived from PK–FK constraints or other declared key compatibilities (e.g., FK–FK joins).

• Seed query. NL–SQL pair $( Q _ { \mathrm { N L } } , Q _ { \mathrm { S Q L } } )$ with $Q _ { \mathrm { S Q L } } \mathrm { ' s }$ join query graph $G _ { Q } = ( V _ { Q } , E _ { Q } )$ , where $V _ { Q } \subseteq V _ { S }$ are the tables appearing in FROM / JOIN, and $E _ { Q }$ are the explicit join predicates.

• User preferences. Optional parameters that bound and prioritize (ascending/descending) the expansion process: – Join complexity preference: whether to favour expansions that increase the number of join conditions per added table, or to restrict expansions to simpler, single-edge joins.

– Centrality: prioritization of expansions to more central tables in $G _ { S } \ ( e . g .$ ., by degree or betweenness).

– Query budget: the maximum number of expanded NL–SQL pairs to generate.

– Pattern coverage: the maximum number of distinct, non-isomorphic join query graph (JQG) patterns to retain with reference to an input evaluation set.

By default, we prioritize higher complexity (more joins), arbitrary centrality, do not set a query budget, and set a maximum of a single unique JQG.

2) Output: A set of expanded NL–SQL pairs produced by augmenting the seed query with one table $t _ { e } ~ \in ~ V _ { S } \setminus V _ { Q }$ and one or more admissible join predicates from $L ( e )$ . Each resulting variant is retained only if it satisfies the semantic and diversity constraints specified by the user preferences. The corresponding NL query is rewritten to reflect the added table and any newly projected attributes, with the aim of preserving the original query intent.

3) Multi-stage pipeline: Fig. 2 illustrates JQE as a multistage pipeline, and Algorithm 1 specifies its operations. We briefly describe each stage below.

• Stage A. Candidate Table Selection. Using $G _ { S }$ , we enumerate candidate expansion tables, $v _ { e } \in V _ { S } \setminus V _ { Q }$ , that are adjacent to at least one $v _ { t } \in V _ { Q }$ . Each $( v _ { t } , v _ { e } )$ pair yields a potential join-condition expansion.

• Stage B. Join-Condition Combination (JCC) Generation. For each candidate table $v _ { e } ,$ let $\{ j c _ { k } \}$ be the set of join labels on edges between $v _ { e }$ and $\{ v _ { t } \in V _ { Q } \}$ . We consider all non-empty subsets (power set) of $\{ j c _ { k } \}$ as possible expansions for $v _ { e } ,$ then prune combinations that introduce redundant joins (defined below).

• Stage C. Diversity-Aware Pruning. We rank JCCs using user preferences (e.g., by added joins and centrality) and retain up to n queries per unique join pattern, where uniqueness is determined via JQG isomorphism (IS\_UNIQUE\_JQG in Alg. 1). This preserves structural variety while controlling cost.

• Stage D. Expanded SQL Synthesis. For each retained JCC, we programmatically inject the new table and join predicates into Q<sub>SQL</sub> (e.g., by extending FROM/JOIN and ON/WHERE clauses). Optionally, simple predicates or projections may be added.

• Stage E. NL Regeneration. A few-shot LLM prompt rewrites the NL query so that table/column mentions reflect the added join(s), while preserving the original intent. We keep prompts fixed for reproducibility.

In summary, given $G _ { S } .$ , JQE (A) identifies candidate tables for expansion, (B) enumerates all non-empty join-condition combinations (JCCs) for each candidate table, (B<sup>′</sup>) discards JCCs with redundant joins, (C) performs diversity-aware pruning, (D) programmatically synthesizes the expanded SQL query, and (E) regenerates the corresponding NL query via an LLM invocation. We next expand on redundancy checks and diversity-aware pruning.

## D. Semantic Redundancy Check

We remove redundant joins from the JQE-generated results. A join redundancy occurs when a candidate join is implied transitively by existing joins $( e . g . , T _ { 1 } . A = T _ { 2 } . A$ and $T _ { 2 } . A = T _ { 3 } . A$ imply $T _ { 1 } . A = T _ { 3 } . A )$ . Such joins do not increase complexity and rarely appear in manually authored queries. We detect redundancy through HAS REDUNDANCY in Alg. 1 by examining cycles that include the candidate join expansion set, modeled as edges in the JQG; any JCC that forms such a transitive cycle is discarded.

## E. Diversity-Aware Pruning via JQG Isomorphism

We further refine the generated queries by pruning structurally equivalent variants. Two queries are considered structurally equivalent when their JQGs are isomorphic [23]. To allow flexibility, user preferences may specify that up to n queries be retained for each unique JQG, with n=1 by default. We call the retained queries the expansion set and the discarded queries the pruned set; together, they constitute the generated set.

Algorithm 1 EXPAND JOINED TABLES   
1: Input: schema graph $G _ { S } ;$ NL–SQL pair $( Q _ { \mathrm { N L } } , Q _ { \mathrm { S Q L } } ) ;$   
$G _ { S } = ( V _ { S } , E _ { S } ) ;$ ; preferences; EvalSet   
2: CTS $ \emptyset ~ / /$ CTs: candidate tables for expansion   
3: for $v _ { t } \in V _ { Q }$ do ▷ Stage A   
4: for $v _ { e } \in \mathrm { N E I G H B O R S } _ { G _ { S } } ( v _ { t } ) \setminus V _ { Q }$ do   
5: $\mathrm { C T s } \gets \mathrm { C T s } \cup \{ ( v _ { t } , v _ { e } ) \}$   
end for   
7: end for   
8: $\mathrm { J C C S }  \emptyset \mathrm { \ } / /$ join-condition combinations   
9: for $( v _ { t } , v _ { e } ) \in \mathbf { C T S }$ do ▷ Stage B   
10: $\mathcal { I } \gets \mathrm { J O I N \_ L A B E L S } ( v _ { t } , v _ { e } )$   
11: for $S \in \operatorname { P O W E R S E T } ( { \mathcal { I } } ) \setminus \{ \varnothing \}$ do   
12: $\mathbf { i f } \lnot \mathrm { H A S } .$ REDUNDANCY $( G _ { Q } , S )$ then ▷ Stage $\mathbf { C ^ { \prime } }$   
13: $\operatorname { J C C S }  \operatorname { J C C S } \cup \{ ( v _ { e } , S ) \}$   
14: end if   
15: end for   
16: end for   
17: for $( v _ { e } , S ) \in \mathrm { R A N K } ( \mathbf { J C C S } , \mathbf { p r e f s } ) \ \mathbf { d o }$ ▷ Stage C   
18: $Q _ { \mathrm { S Q L } } ^ { \prime }  \mathrm { I N J E C T \_ J O I N S } ( Q _ { \mathrm { S Q L } } , v _ { e } , S )$ ▷ Stage D   
19: if $\mathrm { I \tilde { S } \_ U N I Q U E \_ J Q G } ( Q _ { \mathrm { S Q L } } ^ { \prime } , \mathrm { E v a l S e t } )$ then   
20: $Q _ { \mathrm { N L } } ^ { \prime }  \mathrm { R E W R I T E \_ N L } ( Q _ { \mathrm { S Q L } } ^ { \prime } )$ ▷ Stage E   
21: $\mathtt { E v a l S e t }  \mathtt { E v a l S e t } \bigcup \{ ( Q _ { \mathrm { N L } } ^ { \prime } , Q _ { \mathrm { S Q L } } ^ { \prime } ) \}$   
22: end if   
23: end for

TABLE II  
QUERY COUNT IN THE SETS: BIRD DEV AND JQE OUTPUTS.
<table><tr><td></td><td>BIRD dev</td><td>Generated</td><td>Pruned</td><td>Expansion</td></tr><tr><td>Query Count</td><td>1,534</td><td>6,873</td><td>6,815</td><td>58</td></tr></table>

## F. Experimental Analysis

To evaluate JQE, we follow the experimental setup in Section III and apply JQE to the BIRD dev set with the default user preferences.

1) JQE Output Structural Distribution: JQE generates a total of 6,873 queries (the generated set). Among these, 58 constitute the expansion set, consisting of queries that are non-isomorphic to any query in the BIRD dev set or to any previously retained query, while the remaining 6,815 are pruned because they are isomorphic to a query in either the BIRD dev set or the expansion set already retained. Overall, the generated set is 4.5× larger than BIRD dev, and all queries execute with non-empty results. Table II provides a summary.

Since JQE rewrites NL queries after SQL expansion (Stage E), we perform a lightweight human verification of NL– SQL pair alignment on the full expansion set (58 pairs). Each expanded NL–SQL pair is labelled as one of the following: fully captures the intent (aligned; no edits needed), simply adds context/noise (compatible but underspecified for the added join or columns), or loses the intent (misaligned; would require correction). We find that 74.1% (43/58) fully capture the intent, 13.8% (8/58) simply add context or noise, and 12.1% (7/58) lose the intent and require correction. This indicates that most rewritten NL queries remain faithful, while a small fraction exhibit underspecification or drift and require human or automated verification.

TABLE III  
THE PERCENTAGE OF CYCLIC AND ACYCLIC JOIN QUERY GRAPHS FOR THE ORIGINAL BIRD QUERIES AND THE GENERATED AND EXPANSION SETS.
<table><tr><td>Set</td><td>Acyclic (%)</td><td>Cyclic (%)</td></tr><tr><td>BIRD dev</td><td>99.73</td><td>0.27</td></tr><tr><td>Generated Set</td><td>95.69</td><td>4.31</td></tr><tr><td>Expansion Set</td><td>48.28</td><td>51.72</td></tr></table>

Next, we analyze the resulting structural distribution by comparing the average degrees and cyclicity of the JQGs in the BIRD dev set and the expansion set.

Join Query Graph Node Degree. We measure connectivity using the average node degree of the JQG, $\bar { d } = 2 | E | / | V |$ For BIRD dev, $\bar { d } { = } 0 . 8 2 ;$ for the generated set, ${ \bar { d } } { = } 1 . 3 5 ;$ and for the expansion set, <sup>¯</sup>d=1.80. Higher connectivity is driven by the default preference for greater join complexity, i.e., more join conditions in each expansion.

We further analyze per-query changes in the average node degree by comparing each original JQG $( G _ { \mathrm { o r i g } } )$ with its expanded counterpart $( G _ { \mathrm { e x p } } )$ . Across the 6,873 generated queries, the most common degree increments are $\Delta \bar { d } { = } 0 . 3 3 \ \ : ( 3 , 9 1 0$ cases), ∆ <sup>¯</sup>d=0.17 (1,356 cases), and $\Delta \bar { d } { = } 1 . 0 0 \ ( 1 , 1 6 6 \ \mathrm { c a s e s } )$ These distributions mirror the structure of the input queries: 58.7% of the queries in BIRD dev originally join two tables, so adding a third table with a single join yields ∆ <sup>¯</sup>d=0.33. Expansions that connect a new table to the two existing ones introduce higher connectivity, producing ∆ <sup>¯</sup>d=1.00.

Join Query Graph Cyclicity. Cyclic patterns are rare in BIRD dev (0.27%) but increase to 4.31% in the generated set and to 51.72% in the expansion set of 58 output queries (Table III). By maximizing the number of join conditions and thereby introducing cycles, JQE reflects the cyclic structures present in the schema graphs of several BIRD databases.

Takeaway. JQE substantially increases structural diversity: in our experiment, it is able to generate 6,873 queries with 58 new join patterns, where the JQG average degree rises from 0.82 to 1.35, and cyclicity from 0.27% to 4.31%. These effects are even more pronounced in the expansion set of 58 final queries.

2) Accuracy after JQE: We evaluate CHESS, DIN-SQL, and MAC-SQL (from Section III-4) on: (i) BIRD dev set; (ii) $Q \mathrm { { _ { o r i } } \mathrm { { : } } }$ the 30 original queries that yield the 58 expansions; and (iii) $\begin{array} { r } { Q _ { \mathrm { e x p } } \mathrm { : } } \end{array}$ the 58 expanded queries. Table IV reports the EX of all three systems on each query set.

TABLE IV  
EX OF CHESS, DIN-SQL, AND MAC-SQL ON THE FULL BIRD DEV SET, ON THE 30 ORIGINAL SEED QUERIES THAT PRODUCE JQE EXPANSIONS $( Q _ { o r i } )$ , AND ON THE 58 CORRESPONDING EXPANDED QUERIES $( Q _ { e x p } )$
<table><tr><td>System</td><td> $\mathbf { E X _ { d e v } }$ </td><td> $\mathbf { E X } _ { Q _ { o r i } }$ </td><td> $\mathbf { E X } _ { Q _ { e x p } }$ </td></tr><tr><td>CHESS</td><td>64.86%</td><td>63.33%</td><td>44.83%</td></tr><tr><td>DIN-SQL</td><td>59.11%</td><td>60.00%</td><td>32.76%</td></tr><tr><td>MAC-SQL</td><td>55.90%</td><td>40.00%</td><td>39.66%</td></tr></table>

TABLE V

DISTRIBUTION OF CHANGES IN EX (∆EX) BETWEEN ORIGINAL QUERIES AND THEIR CORRESPONDING EXPANDED QUERIES FOR CHESS, DIN-SQL, AND MAC-SQL.
<table><tr><td>System</td><td>△EX=1</td><td>∆EX=-1</td><td>△EX=0</td></tr><tr><td>CHESS</td><td>8.62%</td><td>18.97%</td><td>72.42%</td></tr><tr><td>DIN-SQL</td><td>5.17%</td><td>24.14%</td><td>70.69%</td></tr><tr><td>MAC-SQL</td><td>17.24%</td><td>8.62%</td><td>74.13%</td></tr></table>

We define ∆EX = −1 when a system changes from correct (EX=1) on an original query $( Q _ { o r i } )$ to incorrect (EX=0) on its expanded version $( Q _ { e x p } ) _ { : }$ , ∆EX = 1 for the reverse, and $\Delta \mathrm { E X } = 0$ for no change. Table V reports the distribution of ∆EX: degradations (∆EX=−1) are more frequent than improvements (∆EX=1) for CHESS and DIN-SQL, with DIN-SQL most affected (24.14%), followed by CHESS (18.97%).

3) Join Complexity vs. Performance: From the generated set, we sample four groups by JQG size: $( | V | , | E | ) \in$ $\{ ( 2 , 1 ) , ( 3 , 2 ) , ( 4 , 3 ) , ( 5 , 4 ) \}$ . For each group, we keep the number of conditions and projections fixed, use the same number of queries per group (120, total 480), and balance databases within each condition/projection combination. Figure 3 shows EX changes across groups for the three systems. Accuracy declines monotonically with the number of joins, reaching ≤15% at higher join counts.

Manual inspection attributes most failures to join-structure changes; a smaller fraction arise from secondary issues such as missing DISTINCT, projection errors, or GROUP BY mismatches. We ran an additional control experiment to better isolate the cause of the degradation. We use DIN-SQL, which is the most straightforward system to instrument because of its simple pipeline, and evaluate each seed query together with its corresponding expanded variant. Concretely, we first run DIN-SQL on the expanded NL query and capture the prompt of the generation stage. We then keep this context fixed, replace only the NL query with the original one, and regenerate the SQL. This experiment aims to disentangle the effect of join expansion (more involved query) from the effect of noise in the prompt by keeping the same context.

We report the transition rates among cases where the expanded NL query fails $( \mathrm { E X _ { e x p } { = } 0 } )$ : Table VI shows what fraction of these failures recover when we swap to the original NL under the same context, i.e., $\mathrm { P r } ( \mathrm { E X } _ { \mathrm { o r g } } { = } 1 \mid \mathrm { E X } _ { \mathrm { e x p } } { = } 0 )$ . This recovery rate is substantial for small join sizes (e.g., 54.90% at $n _ { \mathrm { j o i n } } { = } 1 )$ and decreases as joins increase (down to 24.05% at $n _ { \mathrm { j o i n } } { = } 4 )$ , indicating that once the context is built for higherjoin expansions, even the original NL becomes harder to solve under that fixed context. Despite this decreasing recovery rate, the consistent gains from expanded→original across all groups still show that the additional join reasoning required by the expanded queries is a driver of the observed degradation. Even when degradation is possibly attributable to increased context noise, it is a consequence of queries that require more joins and thus expand to larger schema and evidence.

TABLE VI  
DIN-SQL RECOVERY UNDER FIXED EXPANDED-QUERY CONTEXT. FORCASES WHERE THE EXPANDED QUERY IS INCORRECT $( \mathrm { E X } _ { \mathrm { E X P } } { = } 0 )$ , WEREPLACE THE EXPANDED NL QUERY WITH THE ORIGINAL NL QUERYWHILE KEEPING THE GENERATION-STAGE CONTEXT UNCHANGED, ANDREPORT THE EX TRANSITION RATES BY JOIN COUNT n<sub>JOIN</sub>.
<table><tr><td> $n _ { \mathrm { j o i n } }$ </td><td>0→1 (recover)</td><td>0→0 (still fail)</td><td>Total Qs  $\mathbf { ( E X _ { e x p } { = } 0 ) }$ </td></tr><tr><td>1</td><td>54.90% (28)</td><td>45.10% (23)</td><td>51</td></tr><tr><td>2</td><td>41.54% (27)</td><td>58.46% (38)</td><td>65</td></tr><tr><td>3</td><td>34.67% (26)</td><td>65.33% (49)</td><td>75</td></tr><tr><td>4</td><td>24.05% (19)</td><td>75.95% (60)</td><td>79</td></tr></table>

![](images/e252bea348c6f5275a09c626200663a728dcfdfdbc9d9f7be67c28d7f7b99d27.jpg)  
Fig. 3. Execution accuracy (EX) decreases with the number of joins.

Takeaway. JQE efficiently stresses SQL structure: across CHESS, DIN-SQL, and MAC-SQL, EX drops as join complexity increases, confirming that joinheavy queries remain a key failure mode.

## V. TEXTUAL QUERY AUGMENTATION

## A. Overview

Schema naming conventions exert a strong influence on Text-to-SQL performance. Yet enterprise databases frequently employ domain-specific terminology and abbreviations, whereas public benchmarks predominantly feature more natural names. To alleviate this issue, Textual Query Augmentation (TQA) targets the robustness of Text-to-SQL systems by systematically altering the natural language (NL) query and schema identifiers in evaluation sets while preserving semantics and executability.

Within SQLMorph, TQA serves as a controlled mechanism for testing the retrieval stage, which is known to be sensitive to naming conventions and lexical overlap. Prior studies establish an inverse relationship between naming naturalness and model accuracy, categorizing identifiers into three levels: regular (N1), low (N2), and least (N3) [24]. TQA challenges the retrieval stage by automatically transforming benchmark data into less-natural variants that maintain the same semantics.

![](images/a8018d836b8797cea96090bdbecb90f62ee2df3c91d2be4b148ef7913ed06fd0.jpg)  
Fig. 4. Distribution of naming naturalness across BIRD dev databases.

Figure 4 illustrates the distribution of naming naturalness across the BIRD dev set, in which roughly 80% of schema identifiers have natural (N1) forms. This imbalance motivates the need for augmentation techniques that can generate realistic, enterprise-like naming conditions to challenge retrieval stages and human-in-the-loop modules.

## B. Design Principles

One of TQA’s design principles is to ensure that the observed performance changes arise solely from textual perturbations and not from semantic variation. As such, TQA transforms schema elements and their references within queries and evidence while keeping the SQL executable and replacing an element name in the SQL only when necessary. Values, constants, and clauses are kept fixed to guarantee minimal changes. This ensures that only lexical aspects of the query and not logic or intent, are modified.

## C. Approach

Our approach to TQA is divided into three parts:

1) Input:

• Seed query. NL–SQL pair with accompanying evidence and relevant schema.

• Naturalness classifier. We use the SNAILS classifier [24], which was fine-tuned on google/canine-s, to assign each instance to one of three naturalness levels: regular (N1), low (N2), or least (N3).

• User preferences (optional). De-naturalization aggressiveness (N1→N3 vs. N1→N2), and target references location: NL, schema, or both. By default, we apply the most aggressive setting, targeting N3 in both the NL query and the schema.

TABLE VII  
PERCENTAGE CHANGE IN EX (∆EX) ON THE BIRD DEV SET UNDER THE THREE TQA NATURALNESS SETTINGS FOR CHESS, MAC-SQL, AND DIN-SQL, RELATIVE TO THE ORIGINAL/ORIGINAL (O/O)
<table><tr><td rowspan="2">NL/Schema</td><td colspan="3">∆EX (%)</td></tr><tr><td>CHESS</td><td>MAC-SQL</td><td>DIN-SQL</td></tr><tr><td>L/0</td><td>-7.0</td><td>-11.1</td><td>-9.2</td></tr><tr><td>O/L</td><td>-6.7</td><td>-9.7</td><td>-23.4</td></tr><tr><td>L/L</td><td>-8.8</td><td>-11.7</td><td>-23.9</td></tr></table>

2) Output: A set of minimally changed, less-natural references that remain semantically faithful and executable, potentially yielding (i) modified schemas (via ALTER); (ii) updated ground-truth SQL queries (with rewritten identifiers); and (iii) updated NL queries and evidence (with only schema mentions rewritten). Each item passes execution equivalence checks (EX = 1 relative to the original query).

3) Technique: TQA executes as follows:

• Classify Naturalness. Extract table and column identifiers from the database schema and label each with the SNAILS classifier as N1, N2, or N3, together with a confidence score.

• Decrease Naturalness. Transform N1 and N2 identifiers into N3, or into the user-specified target. Using an LLM invocation with temperature 0 and a fixed seed yields transformations such as WaterTemperature→WtTp.

• Alter Schema. Clone each database and apply table and column renames via SQL ALTER. Validate the rename was successful (e.g., using PRAGMA table\_info()).

• Alter SQL Query. Update the ground-truth SQL query by replacing the old identifier using regex while avoiding SQL keywords, functions, and values. We discard original NL queries for which the system did not obtain EX = 1.

• Alter NL Query. Rewrite the NL query and evidence by substituting only schema mentions with least-natural forms using a fixed few-shot prompt. We avoid changes to values (numbers, dates, entities). By restricting edits to schema mentions, this substitution preserves the original meaning and keeps the NL query semantically consistent.

## D. Experimental Setup and Analysis

To evaluate TQA, we follow the experimental setup in Section III. We evaluate four augmentation settings that rename either the NL side (NL query and evidence), the SQL side (schema definition and SQL query), or both:

• O/O: Original NL + Original SQL

• L/O: Less-natural NL + Original SQL

• O/L: Original NL + Less-natural SQL

• L/L: Less-natural NL + Less-natural SQL

We apply TQA to the BIRD dev set (1,534 NL–SQL query pairs). We evaluate TQA at both the system and the stage levels, and we provide a qualitative example.

a) System-level analysis: We consider only queries for which a system successfully generated a correct output, i.e., queries for which O/O yields EX = 1, and then measure changes under (L/O), (O/L), and (L/L). This leads to initial query counts per system as follows: 998 for CHESS, 857 for MAC-SQL, and 904 for DIN-SQL. Table VII reports ∆EX (%) relative to O/O (negative is a drop). All systems degrade consistently. Schema and NL de-naturalization both reduce EX; DIN-SQL shows the largest degradation, followed by MAC-SQL and CHESS. Thus, TQA can increase benchmark difficulty while preserving semantics.

TABLE VIII  
SCHEMA RETRIEVAL PERFORMANCE ACROSS NATURALNESS SETTINGS. FPR (%): FALSE POSITIVE RATE (LOWER IS BETTER). SLR (%): SCHEMA LINKING RECALL (HIGHER IS BETTER).
<table><tr><td rowspan="2"></td><td colspan="2">Full-Schema</td><td colspan="2">SCSL</td><td colspan="2">TCSL</td></tr><tr><td>NL/Schema FPR</td><td>SLR</td><td>FPR</td><td>SLR</td><td>FPR</td><td>SLR</td></tr><tr><td>0/0</td><td>90.1</td><td>99.5</td><td>29.9</td><td>15.0</td><td>13.7</td><td>27.4</td></tr><tr><td>L/0</td><td>88.0</td><td>30.4</td><td>29.2</td><td>11.7</td><td>14.3</td><td>21.4</td></tr><tr><td>O/L</td><td>90.1</td><td>96.4</td><td>34.1</td><td>9.8</td><td>16.4</td><td>25.4</td></tr><tr><td>L/L</td><td>88.2</td><td>28.2</td><td>33.8</td><td>14.1</td><td>16.1</td><td>20.1</td></tr></table>

Takeaway. TQA can make queries harder while preserving semantics and executability. We find that regardless of the augmentation setting, O/L, L/O, and L/L, all lead to degradation. For production settings in which the schema cannot change, L/O provides a natural setting to challenge system retrieval. This further highlights the importance of query expansion and data expansion techniques in Text-to-SQL systems.

b) Stage-level analysis: To conduct a stage-level analysis, we adopt the schema retrieval formulation of prior work [25] and consider three retrieval approaches: Full-Schema (no filtering; retrieves over the full schema), TCSL (Table-then-Column), and SCSL (Single-Column). We evaluate them across O/O, O/L, L/O, and L/L on the augmented queries. We report two complementary retrieval metrics: FPR (False Positive Rate; lower is better) and SLR (Schema Linking Recall; higher is better). FPR quantifies the proportion of retrieved columns that are irrelevant to the query, thereby measuring retrieval noise. Lower FPR indicates a more precise retriever that introduces less distracting context for downstream generation. SLR, in contrast, measures whether all required columns are retrieved for each query. Results are summarized in Table VIII.

• L/O. Relative to O/O, SLR drops sharply for Full-Schema (up to −69.1%) and more moderately for SCSL and TCSL (−3.3% and −6.0%). FPR decreases moderately, with a slight increase for TCSL. These changes point to worse retrieval quality and help explain the EX reduction.

• O/L. SLR again decreases: −3.1%, −5.2%, and −2.0% for Full-Schema, SCSL, and TCSL, respectively, while FPR rises for SCSL and TCSL (+4.2% and +2.7%).

• L/L. Moving from L/O to L/L, the additional changes are comparatively small: SLR changes by −6.2%, +2.4%, and −1.3%, accompanied by only slight increases in FPR.

c) Qualitative Example: To illustrate the augmentation process and its downstream effects, we show an example from the toxicology database (ID 245) in the BIRD dev set. The original and less-natural versions differ only in schema naming and in its propagation to the NL and SQL, while preserving semantics and execution results. We next present the original and less-natural NL query, the evidence, and the required changes to the relevant schema. We then analyze the behaviour of schema retrieval on this query.

Original NL: What is the average number of bonds the atoms with the element iodine have?

Less-natural NL: What is the average number of bnds the atms with the Elmt iodine have?

## Original Evidence:

atoms with the element iodine refers to element = ‘i’; average = DIVIDE(COUNT(bond id), COUNT(atom id)) where element = ‘i’

## Less-natural Evidence:

atms with the Elmt iodine refers to Elmt = ‘i’; average = DIVIDE(COUNT(b id), COUNT(atmId)) where Elmt = ‘i’

## Original Relevant Schema:

atom.atom\_id, atom.element, connected.atom\_ id, connected.bond\_id

## Less-natural Relevant Schema:

atm.atmId, atm.Elmt, conn.atmId, conn.b\_id

Schema Retrieval. Under the O/L and L/O settings, systems frequently fail to retrieve the correct columns, e.g., by missing atom\_id or connected.atom\_id, which leads to downstream generation and execution errors.

Takeaway. TQA can stress-test retrieval and query augmentation techniques, as well as system-level query reformulation and understanding. TQA can help identify misalignment failures when matching an NL query to a schema, e.g., in the schema linking stage.

## VI. FINE-GRAINED EVALUATION METRICS

## A. Motivation

The standard EX metric is binary: the predicted and expected output relations either match exactly or they do not. As a result, it compresses a wide range of outcomes into a single bit and discards potentially important signal. Queries that recover most of the correct rows but miss a single boundary condition receive the same score as queries that return entirely irrelevant results. Likewise, structural differences such as harmless extra columns are penalized as severely as true semantic errors. Consequently, EX obscures partial correctness, limits error diagnosis, and provides little insight into how a Text-to-SQL system fails, whether through underprediction or over-prediction of rows, columns, or values.

Our proposal: We introduce a family of fine-grained execution-level metrics to quantify how a predicted result deviates from the ground truth, rather than only whether it matches exactly. At the core are two complementary measures: Execution Precision (EXP), which captures the fraction of predicted cells that are correct, and Execution Recall (EXR), which captures the fraction of ground-truth cells that are recovered. Their F1 score provides a compact summary, while separate reporting of EXP and EXR reveals whether errors arise primarily from over-prediction, reflected in lower precision, or under-prediction, reflected in lower recall.

## B. Approach

Given a ground-truth query q and a predicted query ${ \hat { q } } ,$ we execute both to obtain the column-name sets $c o l s ( q )$ , cols(ˆq) and the row multisets rows $( q ) , r o w s ( \hat { q } )$ . If rows(q) and $r o w s ( \hat { q } )$ are identical over their full schemas, then $\mathrm { E X } = 1 $ and we set $\mathrm { E X P } = \mathrm { E X R } = \mathrm { F } 1 = 1$ . Otherwise, we proceed with relaxed matching, which first optionally matches the columns and then the rows and cells. Finally, the evaluator chooses whether to penalize extra predicted columns.

a) Column Matching: We compare only those columns that the evaluator deems matchable; SQLMorph offers three matching regimes:

• Exact-Column (EC). Match by exact column name: $c _ { \cap } = c o l s ( q ) \cap c o l s ( \hat { q } )$

• Semantic-Column (SC). Build a textual descriptor for each column (name plus top-k values), embed the descriptors, restrict to type-compatible pairs and domain values when applicable, and compute a maximum-weight bipartite matching. Keep only pairs above a similarity threshold $( e . g . , \ 0 . 7 )$ . For a fixed embedding model, semantic-column matching is fully deterministic and always yields the same matching; changing the model, however, may change the result. Since matching is performed only over output columns (typically fewer than 20), the cubic assignment cost is negligible in practice and independent of the number of output rows.

• No-Column (NC). Skip column matching entirely; matching then relies only on row and cell comparisons.

b) Row/Cell Matching: Let $c _ { \cap }$ denote the matched column set. We then match $r o w s ( q )$ and rows(ˆq) as follows:

• Exact-Cell (EC). Under EC or SC, project rows(q) and rows(ˆq) onto $c _ { \cap }$ and treat the resulting rows as multisets. Under NC, compare rows directly through their implicitly matched value sets similar to the EX computation. For each unique matched row pattern $r ,$ let $f _ { q } ( r )$ and $f _ { \hat { q } } ( r )$ denote its multiplicities in rows(q) and rows(ˆq), respectively. Then

$$
| \mathrm { M a t c h e d R o w s } | = \sum _ { r } \operatorname* { m i n } ( f _ { q } ( r ) , f _ { \hat { q } } ( r ) ) ,
$$

$$
| \mathrm { M a t c h e d C e l l s } | = | \mathrm { M a t c h e d R o w s } | \cdot | c _ { \cap } | .
$$

Intuition: Credit is assigned only when two rows match exactly over the aligned comparison units, namely matched columns under EC or SC, or implicitly matched values under NC.

• Partial-Cell (PC). Under EC or SC, first project rows(q) and rows(ˆq) onto $_ { c _ { \cap } ; }$ under NC, compare rows through their implicitly matched value sets. Matching then proceeds in two phases:

– Phase 1: exact matching. First, check for exact matches, as in EC. These exact matches are removed, and the remaining unmatched rows are passed to the second phase.

– Phase 2: partial matching. For each remaining $p \in$ rows(ˆq) and ground-truth $g \in r o w s ( q )$ , compute the fraction of equal cells as a similarity score:

$$
\sin ( p , g ) = { \frac { \# \{ i \in c _ { \cap } : p _ { i } = g _ { i } \} } { | c _ { \cap } | } } .
$$

We then greedily select the pair with the highest similarity, add the matching cells $\{ i \in c _ { \cap } : p _ { i } =$ $g _ { i } \}$ to PartialMatchedCells, and remove both rows from further consideration. This process continues until all remaining $r o w s ( \hat { q } )$ have been considered or until no unmatched $r o w s ( q )$ remain. We sort the rows and columns of the outputs of $q$ and $\hat { q }$ before matching so that equivalent relations for the same named attributes produce deterministic matches.

Finally, the total number of matched cells, |MatchedCells|, is |ExactMatchedCells| + |PartialMatchedCells|.

c) Accounting for extra predicted columns: We define $| c e l l s ( q ) | ~ = ~ | r o w s ( q ) | \cdot | c o l s ( q ) |$ as the total number of ground-truth cells and account for extra predicted columns by choosing one of the following options:

• Penalize Extras (PE) predicted columns. $| c e l l s ( \hat { q } ) | =$ $| r o w s ( \hat { q } ) | \cdot | c o l s ( \hat { q } ) |$ . All predicted columns are counted, so extra columns reduce precision.

• Ignore Extras (IE) predicted columns. $| c e l l s ( \hat { q } ) | =$ $\vert r o w s ( \hat { q } ) \vert \cdot \vert c _ { \Omega } \vert$ . Only matched columns are counted, under the assumption that the predicted output already contains the required data and that extra columns may therefore be ignored. This option is unavailable under NC, since no matched column set $c _ { \cap }$ is constructed.

Metrics: The final metrics are defined as follows:

$$
\mathrm { E X P } = | \mathrm { M a t c h e d C e l l s } | / | P C e l l s |
$$

$$
\mathrm { E X R } = | \mathrm { M a t c h e d C e l l s } | / | G C e l l s |
$$

$$
\mathrm { F 1 } = 2 \cdot \mathrm { E X P } \cdot \mathrm { E X R } / ( \mathrm { E X P } + \mathrm { E X R } )
$$

Takeaway. EXP is the fraction of predicted cells that are correct, EXR is the fraction of ground-truth cells that are recovered, and F1 unifies the two. Depending on the evaluation setting, extra predicted columns may either be penalized or ignored.

TABLE IX  
SINGLE-ERROR MUTATIONS AND THEIR EXPECTED EFFECTS ON THE NUMBER OF ROWS (R), COLUMNS (C), VALUES (V), AND EVALUATION METRICS (EXP, EXR). SYMBOLS DENOTE THE EXPECTED DIRECTION OF CHANGE: ↑ INCREASE, ↓ DECREASE, = UNCHANGED, AND × CONTEXT-DEPENDENT.
<table><tr><td>Operator</td><td>Description</td><td>(R, C, V)</td><td>(EXP, EXR)</td></tr><tr><td>projection_drop add_star_wildcard</td><td>Remove one column from the projection list (if &gt; 1 remain) Append * or alias. */table. * if no star exists.</td><td>(=, ↓, =) (=,↑,=)</td><td>(=,↓) (PE ↓; IE =,=)</td></tr><tr><td>distinct_toggle where_predicate_delete</td><td>Remove the DISTINCT keyword. Remove one predicate from a compound condition.</td><td>(×,=,=) (↑,=,=)</td><td>(↓, ×) (, ↑)</td></tr><tr><td>where_condition_flip</td><td>Flip comparison direction  $( =  \neq , >  < , \geq  \leq ) .$ </td><td>(×,=,=)</td><td>(×, ×)</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>where_strengthen</td><td>Make boundary stricter (&lt;→&lt;=, &gt;→&gt;=).</td><td>(↓, =, =)</td><td>(=,↓)</td></tr><tr><td>where_weaken</td><td></td><td></td><td></td></tr><tr><td></td><td>Make boundary looser (&lt;=→&lt;, &gt;=→&gt;).</td><td>(↑,=,=)</td><td>(↓, ↑)</td></tr><tr><td>where_remove</td><td>Remove the entire WHERE clause.</td><td>(↑, =,=)</td><td>(4, ↑)</td></tr><tr><td>having_condition_flip</td><td>Flip aggregate comparison (=↔≠, &gt;↔&lt;, ≥↔≤).</td><td>(×, =, =)</td><td>(×, ×)</td></tr><tr><td>having_remove</td><td>Remove the HAVING clause.</td><td></td><td></td></tr><tr><td>join_break</td><td></td><td>(↑,=,=)</td><td>(4, ↑)</td></tr><tr><td></td><td>Remove the ON condition (cartesian product).</td><td>(↑,=,=)</td><td>(↓, ×)</td></tr><tr><td>join_type_to_left</td><td>Convert non-LEFT joins to LEFT JOIN.</td><td>(↑,=,=)</td><td>(4, ↑)</td></tr><tr><td>limit_increase</td><td>Double the LIMIT and add a random OFFSET.</td><td>(↑,=,=)</td><td>(↓, ↑)</td></tr><tr><td>limit_decrease</td><td>Halve the LIMIT (min 1).</td><td>(↓, =, =)</td><td>(=,1)</td></tr><tr><td>aggregation_swap</td><td>Swap aggregates (SUM↔AVG, MIN↔MAX, COUNT→SUM).</td><td> $( = , = , \times )$ </td><td>(↓,↓)</td></tr></table>

## C. Experimental Analysis

1) Controlled Error Sensitivity: Research Question – Can our fine-grained metrics capture and distinguish the effects of isolated single-error mutants in predicted SQL queries?

To evaluate SQLMorph’s metrics, we construct a benchmark of single-error SQL mutants by applying each mutation to a ground-truth query and evaluating how the resulting change is reflected in the metrics. Each mutant is created by applying exactly one atomic change to a ground-truth query from the BIRD development set; the resulting mutations are summarized in Table IX. In the mutant-generation code, we enforce depth-1 mutations by exhaustively trying each mutation and retaining only those that both modify the AST and produce syntactically valid SQL. When a query contains subqueries, mutations are applied only to the outer query so that each change remains easy to interpret.

Table IX groups the mutants by SQL component and summarizes their expected effects on rows, columns, values, and our metrics. Some mutants mainly change the number of returned rows. For example, increasing the LIMIT or making a WHERE condition less restrictive tends to add rows, which should lower EXP. In contrast, decreasing the LIMIT or making a WHERE condition more restrictive reduces the rows, which should lower EXR. Other mutants affect columns or values. For instance, dropping a projected column should reduce recall, while adding a wildcard mainly hurts EXP when extra predicted columns are penalized. Mutants that alter aggregate values can reduce both EXP and EXR.

Table X summarizes the results. We report ∆EX, ∆EXP, ∆EXR, and ∆F1. Smaller values below 0 indicate a larger penalty or drop from 1. The results show the main weakness of EX. For every mutant, EX drops to 0%. As a result, it cannot distinguish a mild error from a severe one. In contrast, EXP, EXR, and F1 spread the errors across a much wider range and can reveal how a query is incorrect.

Row-count mutants behave as expected. With limit\_ increase, the query returns too many rows, so EXP drops sharply while EXR stays at 1. With limit\_decrease, the query returns too few rows, so EXR drops while EXP remains nearly unchanged. This shows that EXP and EXR clearly separate over-prediction from under-prediction of rows.

TABLE X  
IMPACT OF SINGLE-ERROR MUTATION ON EVALUATION METRICS.
<table><tr><td></td><td colspan="7">Evaluation Metrics</td></tr><tr><td></td><td></td><td colspan="3">EC-EC-IE</td><td colspan="3">SC-EC-IE</td></tr><tr><td>Injected Error</td><td>ΔEX</td><td>∆EXP</td><td>∆EXR</td><td>∆F1</td><td>∆EXP</td><td>∆EXR</td><td>∆F1</td></tr><tr><td>projection_drop</td><td>-100</td><td>0</td><td>-43</td><td>-28</td><td>0</td><td>-43</td><td>-28</td></tr><tr><td>add_star_wildcard</td><td>-100</td><td>-93</td><td>-10</td><td>-87</td><td>-82</td><td>0</td><td>-71</td></tr><tr><td>distinct_toggle</td><td>-100</td><td>-57</td><td>-57</td><td>-57</td><td>-57</td><td>-57</td><td>-57</td></tr><tr><td>where_predicate_delete</td><td>-100</td><td>-85</td><td>-60</td><td>-81</td><td>-85</td><td>-64</td><td>-82</td></tr><tr><td>where_condition_flip</td><td>-100</td><td>-96</td><td>-61</td><td>-94</td><td>-96</td><td>-61</td><td>-94</td></tr><tr><td>where_remove</td><td>-100</td><td>-96</td><td>-54</td><td>-95</td><td>-96</td><td>-54</td><td>-95</td></tr><tr><td>where_strengthen</td><td>-100</td><td>-65</td><td>-60</td><td>-63</td><td>-65</td><td>-60</td><td>-63</td></tr><tr><td>where_weaken</td><td>-100</td><td>-65</td><td>-75</td><td>-73</td><td>-65</td><td>-75</td><td>-73</td></tr><tr><td>having_condition_flip</td><td>-100</td><td>-64</td><td>-100</td><td>-100</td><td>-64</td><td>-100</td><td>-100</td></tr><tr><td>having_remove</td><td>-100</td><td>-61</td><td>0</td><td>-51</td><td>-61</td><td>0</td><td>-51</td></tr><tr><td>join_break</td><td>-100</td><td>-98</td><td>-42</td><td>-98</td><td>-98</td><td>-46</td><td>-98</td></tr><tr><td>join_type_to_left</td><td>-100</td><td>-53</td><td>-50</td><td>-54</td><td>-53</td><td>-50</td><td>-54</td></tr><tr><td>limit_increase</td><td>-100</td><td>-76</td><td>0</td><td>-62</td><td>-76</td><td>0</td><td>-62</td></tr><tr><td>limit_decrease</td><td>-100</td><td>-4</td><td>-61</td><td>-45</td><td>0</td><td>-60</td><td>-43</td></tr><tr><td>aggregation_swap</td><td>-100</td><td>-100</td><td>-100</td><td>-100</td><td>-100</td><td>-100</td><td>-100</td></tr></table>

Schema-related mutants also produce clear patterns. projection\_drop removes output columns, which lowers EXR while leaving EXP unchanged. add\_star\_wildcard adds extra predicted columns, which mainly lowers EXP. Under SC matching, this penalty is smaller, suggesting that SC is more robust to harmless schema noise. distinct\_toggle lowers both EXP and EXR, showing as expected the effect of duplicates on the multiset comparisons of the results.

Filter and grouping mutants show a range of severity. Mutants such as where\_predicate\_delete reduce both EXP and EXR. Bigger changes such as removing the WHERE clause or flipping it causes an even larger drop. For groupingrelated mutants, removing the HAVING clause behaves like over-selection: EXP decreases while EXR stays high. By contrast, flipping a HAVING condition can drive EXR to zero.

-SQL).   
EX: 0.000 vs 0.000   
EXP: 0.393 vs 0.500 (gap: 0.11)   
EXR: 0.982 vs 1.000 (gap: 0.02)   
F1: 0.561 vs 0.667 (gap: 0.11)

Join mutants have their own effects. join\_break, which creates a cartesian product, causes one of the largest EXP drops and also lowers EXR. join\_type\_to\_left produces a smaller and more balanced degradation, which is consistent with a milder semantic change.

Across mutants, SC usually matches or improves on EC, especially when the error introduces extra but semantically related columns. PC helps mainly when predicted rows are close to the correct ones but not identical, such as after join\_break; otherwise, its behaviour is similar to EC.

Takeaway. EXP and EXR provide much more useful diagnostic signal than binary EX. They distinguish over-prediction from under-prediction, separate row and column errors, and better capture cases where outputs are partially correct.

2) System-Level Comparison: Research Question – Do granular metrics reveal finer distinctions in how systems fail when compared to EX?

We conducted a comparative analysis of three Text-to-SQL systems, namely, CHESS, DIN-SQL, and MAC-SQL, on the BIRD dev set. Among the 1, 534 total examples, we identified 410 queries for which all three systems produced EX = 0, representing ∼26.7% of the benchmark. These shared failures pose a particular challenge: EX offers no information for ranking systems or for analyzing their behaviour on failed queries, despite the potential partial correctness of the predictions. Table XI summarizes the results.

The ranking of the systems (CHESS > DIN-SQL > MAC-SQL) remains unchanged under both EX and our fine-grained metrics. However, EXP and EXR reveal meaningful differences within the shared-failure queries: for example, CHESS recovers nearly a quarter of the ground-truth cells, whereas MAC-SQL recovers only about 15%. In addition, precisionrecall asymmetries help explain the nature of these failures: CHESS maintains higher EXR, while DIN-SQL exhibits a more balanced trade-off between EXP and EXP. Thus, although the leaderboard order is stable, our metrics add diagnostic resolution. Fig. 5 illustrates this point by showing the distributions of EXP and EXR over failed queries, highlighting how close each system comes to the correct answer on average even when EX assigns them zero.

3) Qualitative Example: Consider the following NL query: ‘List the names of schools with more than 30 difference in enrollments between K–12 and ages 5–17. Please also give the full street address of the schools.

SC-EC-PE Metrics (DIN-SQL vs MAC-SQL).

This highlights how large the variation can be among failed queries. Next, we report the predicted and golden SQL queries and provide an explanation.

## Ground-truth SQL:

```sql
SELECT T1.School, T1.Street
FROM schools AS T1
JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode
WHERE T2.‘Enrollment (K-12)‘ -
T2.‘Enrollment (Ages 5-17)‘ > 30;
```

## DIN-SQL Predicted SQL:

```sql
SELECT frpm.‘School Name‘, schools.Street,
schools.City, schools.Zip, schools.State
FROM frpm
JOIN schools
ON frpm.CDSCode = schools.CDSCode
WHERE (frpm.‘Enrollment (K-12)‘ -
frpm.‘Enrollment (Ages 5-17)‘) > 30;
```

## MAC-SQL Predicted SQL:

```sql
SELECT T2.School, T2.Street, T2.City, T2.Zip
FROM frpm AS T1
JOIN schools AS T2 ON T1.CDSCode =
T2.CDSCode
WHERE T1.‘Enrollment (K-12)‘ -
T1.‘Enrollment (Ages 5-17)‘ > 30;
```

## Explanation:

• Both systems predict the correct filtering condition for selecting schools with large enrollment differences, but they differ in their projected attributes.

• We would expect both systems to achieve an EXR of 1.0. However, DIN-SQL projects frpm.‘School Name’ instead of School, preventing a perfect match.

• DIN-SQL projects five attributes, whereas MAC-SQL projects four, corresponding to three and two extra attributes, respectively. This additional over-prediction lowers their EXP and, consequently, their F1 scores.

Takeaway: This example shows high EXR but low EXP: both systems recover nearly all expected rows, but are penalized for projecting unnecessary columns. Binary EX treats them as equally wrong, whereas SQLMorph reveals that both are in fact very close to the correct answer.

## VII. RELATED WORK AND CHALLENGES

We contextualize SQLMorph within recent research and emphasize persistent challenges in Text-to-SQL evaluation.

Existing benchmarks such as Spider, BIRD, WikiSQL, KaggleDBQA, and Beaver have driven substantial progress in Text-to-SQL research [2], [3], [26], [27], [28], [4]. However, they remain limited in scale, label fidelity, and compositional diversity. Public datasets also tend to emphasize relatively small schema and shallow join structures, which do not fully reflect the complexity of enterprise workloads [29]. Recent studies show that system accuracy drops sharply as join complexity increases [30]. SQLMorph complements these static benchmarks by generating deterministic query variants that increase both structural and lexical diversity. In particular, JQE introduces additional join predicates, thereby stressing systems along an underrepresented dimension in current benchmarks.

TABLE XI  
SYSTEM-LEVEL COMPARISON OF CHESS, DIN-SQL, AND MAC-SQL ON THE BIRD DEV SET, RESTRICTED TO THE SHARED-FAILED QUERIES WHERE ALL THREE SYSTEMS HAVE EX = 0. FINE-GRAINED METRICS (EXP, EXR, AND F1) UNDER MULTIPLE MATCHING CONFIGURATIONS REVEAL PARTIAL CORRECTNESS AND DIFFERENCES IN FAILURE BEHAVIOUR.
<table><tr><td colspan="2"></td><td colspan="10">Evaluation Metrics (%) – BIRD Shared Failures Subset</td><td colspan="5"></td></tr><tr><td></td><td></td><td colspan="3">EC-EC-IE</td><td colspan="3">EC-PC-IE</td><td colspan="3">SC-EC-IE</td><td colspan="3">SC-PC-IE</td><td colspan="3">NC-PC</td></tr><tr><td>System</td><td>EX</td><td>EXP</td><td>EXR</td><td>F1</td><td>EXP</td><td>EXR</td><td>F1</td><td>EXP</td><td>EXR</td><td>F1</td><td>EXP</td><td>EXR</td><td>F1</td><td>EXP</td><td>EXR</td><td>F1</td></tr><tr><td>CHESS</td><td>0.00</td><td>28.76</td><td>19.66</td><td>18.44</td><td>29.06</td><td>19.85</td><td>18.82</td><td>47.80</td><td>16.62</td><td>15.14</td><td>51.44</td><td>19.11</td><td>17.82</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>DIN-SQL</td><td>0.00</td><td>29.01</td><td>21.12</td><td>18.42</td><td>29.09</td><td>21.25</td><td>18.51</td><td>63.04</td><td>19.54</td><td>17.18</td><td>66.26</td><td>22.05</td><td>19.71</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>MAC-SQL</td><td>0.00</td><td>31.09</td><td>16.07</td><td>14.63</td><td>31.34</td><td>16.30</td><td>14.84</td><td>54.47</td><td>13.18</td><td>11.91</td><td>56.97</td><td>15.10</td><td>13.82</td><td>N/A</td><td>N/A</td><td>N/A</td></tr></table>

![](images/b9a07fecea4152de31ec22951f8702e78d4ccd317cf7175c5f40dc926e564b12.jpg)  
Fig. 5. System-level comparison on the BIRD dev set’s common failed queries on combinations of fine-grained metrics.

A related challenge is that real-world schema differ substantially from those in public datasets. Enterprise databases often contain hundreds of interrelated tables, heterogeneous sources, and cryptic identifiers [4]. Such complexity allows multiple valid SQL queries per NL intent [31], making correctness ambiguous even when intent is preserved [32].

Reproducibility remains another major concern. Progress can be difficult to interpret because results often depend on dataset splits, prompt design, sampling choices, and private evaluation settings [4], [27], [33], [34]. SQLMorph is designed to make evaluation more reproducible by fixing prompts, seeds, and sampling parameters, and by validating generated variants through execution-based checks. It is also suitable for private, in-house evaluation, where schema and workload traces cannot be shared due to privacy, governance, or annotation cost constraints [4], [26]. In this sense, SQLMorph complements recent efforts on enterprise-oriented evaluation such as Beaver and Spider 2.0 [4], [35].

SQLMorph also relates to recent work on schema naming and schema linking. Prior studies have shown that naming quality strongly affects Text-to-SQL performance [24], [26], [36], [25]. For example, SNAILS emphasizes improving schema naturalness to facilitate linking [24]. SQLMorph takes a complementary perspective. TQA intentionally denaturalizes schema identifiers and their mentions in natural language to simulate enterprise-style abbreviations and naming conventions. This allows us to stress-test robustness to lexical mismatch, schema linking, and retrieval stages more broadly.

With respect to metrics, most existing leaderboards still rely on binary Execution Accuracy (EX) [2], [3]. Although EX is simple and widely adopted, it collapses all failures into a single bit and does not indicate whether a system under-predicts rows, over-predicts rows, projects extra columns, or otherwise comes close to the correct answer [31], [37], [32]. SQLMorph introduces Execution Precision (EXP) and Execution Recall (EXR), which separate over-prediction from under-prediction and provide more interpretable error diagnostics. SQLMorph adopts relaxed, execution-grounded metrics with cell-level matching, so that semantically equivalent or partially correct predictions can receive partial credit rather than being counted as complete failures. Our metric design also differs from recent proposals based on continuous similarity scores [38], which can be harder to interpret, and may conflate syntactic and semantic differences.

Recent RL-based and reward-driven training methods for Text-to-SQL often rely on proxy, binary, or composite rewards that do not necessarily align with execution correctness at a fine-grained level [39], [40], [41]. These rewards can reduce sparsity or improve alignment with end-task accuracy, but they do not explicitly decompose execution errors into overprediction and under-prediction in the way that SQLMorph’s EXP and EXR do [39], [41]. As such, SQLMorph’s metrics may be worth exploring as potential training reward signals.

## VIII. CONCLUSION

SQLMorph reframes Text-to-SQL evaluation as a form of controlled stress testing. JQE increases structural complexity through systematic join expansion, while TQA probes robustness to naming variation by de-naturalizing natural-language queries and schema references. Our fine-grained metrics, EXP and EXR, expose over-prediction and under-prediction that binary EX obscures. Together, these components provide a reproducible and diagnostic evaluation framework that better reflects enterprise requirements.

## IX. AI-GENERATED CONTENT ACKNOWLEDGEMENT

We used GenAI tools to correct grammar and suggest synonyms or paraphrases for a few paragraphs.

## REFERENCES

[1] A. Quamar, V. Efthymiou, C. Lei, and F. Ozcan, “Natural language<sup>¨</sup> interfaces to data,” FnT DB, vol. 11, 2022.

[2] T. Yu, R. Zhang, K. Yang, M. Yasunaga, D. Wang, Z. Li, J. Ma, I. Li, Q. Yao, S. Roman, Z. Zhang, and D. Radev, “Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-SQL task,” EMNLP, 2018.

[3] J. Li, B. Hui, G. Qu, J. Yang, B. Li, B. Li, B. Wang, B. Qin, R. Geng, N. Huo et al., “Can llm already serve as a database interface? a big bench for large-scale database grounded text-to-sqls,” NeurIPS, 2024.

[4] P. B. Chen, F. Wenz, Y. Zhang, D. Yang, J. Choi, N. Tatbul, M. Cafarella, C¸ . Demiralp, and M. Stonebraker, “Beaver: an enterprise benchmark for text-to-sql,” CoRR, vol. abs/2409.02038, 2024.

[5] Z. Hong, Z. Yuan, Q. Zhang, H. Chen, J. Dong, F. Huang, and X. Huang, “Next-generation database interfaces: A survey of llm-based text-to-sql,” CoRR, vol. abs/2406.08426, 2024.

[6] B. Li, Y. Luo, C. Chai, G. Li, and N. Tang, “The dawn of natural language to sql: Are we fully ready?” CoRR, vol. abs/2406.01265, 2024.

[7] X. Liu, S. Shen, B. Li, P. Ma, R. Jiang, Y. Luo, Y. Zhang, J. Fan, G. Li, and N. Tang, “A survey of nl2sql with large language models: Where are we, and where are we going?” CoRR, vol. abs/2408.05109, 2024.

[8] W. Zhang, Y. Wang, Y. Song, V. J. Wei, Y. Tian, Y. Qi, J. H. Chan, R. C.-W. Wong, and H. Yang, “Natural language interfaces for tabular data querying and visualization: A survey,” CoRR, vol. abs/2310.17894, 2024.

[9] K. Maamari and A. Mhedhbi, “End-to-end text-to-sql generation within an analytics insight engine,” CoRR, vol. abs/2406.12104, 2024.

[10] M. Pourreza and D. Rafiei, “Din-sql: Decomposed in-context learning of text-to-sql with self-correction,” NeurIPS, 2023.

[11] B. Wang, C. Ren, J. Yang, X. Liang, J. Bai, L. Chai, Z. Yan, Q.-W. Zhang, D. Yin, X. Sun, and Z. Li, “Mac-sql: A multi-agent collaborative framework for text-to-sql,” 2025.

[12] M. Pourreza, H. Li, R. Sun, Y. Chung, S. Talaei, G. T. Kakkar, Y. Gan, A. Saberi, F. Ozcan, and S. O. Arik, “Chase-sql: Multi-path reasoning and preference optimized candidate selection in text-to-sql,” CoRR, 2024.

[13] S. Talaei, M. Pourreza, Y.-C. Chang, A. Mirhoseini, and A. Saberi, “Chess: Contextual harnessing for efficient sql synthesis,” CoRR, vol. abs/2405.16755, 2024.

[14] X. Xie, G. Xu, L. Zhao, and R. Guo, “Opensearch-sql: Enhancing textto-sql with dynamic few-shot and consistency alignment,” CoRR, vol. abs/2502.14913, 2025.

[15] Y. Gan, X. Chen, J. Xie, M. Purver, J. R. Woodward, J. Drake, and Q. Zhang, “Natural sql: Making sql easier to infer from natural language specifications,” EMNLP, 2021.

[16] Y. D. Donder, D. Hommel, A. W. Wen-Yi, D. Mimno, and U. E. S.¨ Jo, “Cheaper, better, faster, stronger: Robust text-to-sql without chainof-thought or fine-tuning,” CoRR, vol. abs/2505.14174, 2025.

[17] D. Lee, C. Park, J. Kim, and H. Park, “MCS-SQL: leveraging multiple prompts and multiple-choice selection for text-to-sql generation,” COLING, 2025.

[18] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. H. Chi, Q. V. Le, and D. Zhou, “Chain-of-thought prompting elicits reasoning in large language models,” NeurIPS, 2022.

[19] Y. Gao, Y. Liu, X. Li, X. Shi, Y. Zhu, Y. Wang, S. Li, W. Li, Y. Hong, Z. Luo, J. Gao, L. Mou, and Y. Li, “A preview of xiyansql: A multi-generator ensemble framework for text-to-sql,” CoRR, vol. abs/2411.08599, 2025.

[20] L. Sheng and S. Xu, “CSC-SQL: corrective self-consistency in text-tosql via reinforcement learning,” CoRR, vol. abs/2505.13271, 2025.

[21] K. Maamari, C. Landy, and A. Mhedhbi, “Genedit: Compounding operators and continuous improvement to tackle text-to-sql in the enterprise,” 2025.

[22] M. Malekpour, N. Shaheen, F. Khomh, and A. Mhedhbi, “Towards optimizing SQL generation via LLM routing,” CoRR, vol. abs/2411.04319, 2024.

[23] J. R. Ullmann, “An algorithm for subgraph isomorphism,” JACM, vol. 23, no. 1, pp. 31–42, 1976.

[24] K. Luoma and A. Kumar, “Snails: Schema naming assessments for improved llm-based sql inference,” SIGMOD, 2025.

[25] K. Maamari, F. Abubaker, D. Jaroslawicz, and A. Mhedhbi, “The death of schema linking? text-to-sql in the age of well-reasoned language models,” CoRR, vol. abs/2408.07702, 2024.

[26] B. Qin, B. Hui, L. Wang, M. Yang, J. Li, B. Li, R. Geng, R. Cao, J. Sun, L. Si, F. Huang, and Y. Li, “A survey on text-to-sql parsing: Concepts, methods, and future directions,” CoRR, 2022.

[27] V. Zhong, C. Xiong, and R. Socher, “Seq2sql: Generating structured queries from natural language using reinforcement learning,” CoRR, 2017.

[28] C.-H. Lee, O. Polozov, and M. Richardson, “KaggleDBQA: Realistic evaluation of text-to-SQL parsers,” Aug. 2021.

[29] A. Mitsopoulou and G. Koutrika, “Analysis of text-to-SQL benchmarks: Limitations, challenges and opportunities,” 2025.

[30] T. Eckmann, M. Urban, J.-M. Bodensohn, and C. Binnig, “HLR-SQL: Human-like reasoning for text-to-SQL,” 2025.

[31] A. Floratou, F. Psallidas, F. Zhao, S. Deep, G. Hagleither, W. Tan, J. Cahoon, R. Alotaibi, J. Henkel, A. Singla, A. v. Grootel, B. Chow, K. Deng, K. Lin, M. Campos, V. Emani, V. Pandit, V. Shnayder, W. Wang, and C. Curino, “NL2SQL is a solved problem... not!” CIDR, 2024.

[32] M. Pourreza and D. Rafiei, “Evaluating cross-domain text-to-SQL models and benchmarks,” 2023.

[33] C. Renggli, I. F. Ilyas, and T. Rekatsinas, “Fundamental challenges in evaluating text2sql solutions and detecting their limitations,” CoRR, vol. abs/242501.18197, 2025.

[34] F. Wenz, O. Bouattour, D. Yang, J. Choi, C. Gregg, N. Tatbul, and C¸ agatay Demiralp, “Benchpress: A human-in-the-loop annota-˘ tion system for rapid text-to-sql benchmark curation,” CoRR, vol. abs/2510.13853, 2025.

[35] F. Lei, J. Chen, Y. Ye, R. Cao, D. Shin, H. Su, Z. Suo, H. Gao, W. Hu, P. Yin, V. Zhong, C. Xiong, R. Sun, Q. Liu, S. Wang, and T. Yu, “Spider 2.0: Evaluating language models on real-world enterprise textto-sql workflows,” CoRR, vol. abs/2411.07763, 2024.

[36] G. Katsogiannis-Meimarakis and G. Koutrika, “A survey on deep learning approaches for text-to-sql,” VLDBJ, 2023.

[37] A. Kumar, P. Nagarkar, P. Nalhe, and S. Vijayakumar, “Deep learning driven natural languages text to sql query conversion: A survey,” CoRR, 2022.

[38] G. Pinna, Y. Perezhohin, L. Manzoni, M. Castelli, and A. De Lorenzo, “Redefining text-to-SQL metrics by incorporating semantic and structural similarity,” Sci. Rep., 2025.

[39] M. Pourreza, S. Talaei, R. Sun, X. Wan, H. Li, A. Mirhoseini, A. Saberi, and S. O. Arik, “Reasoning-sql: Reinforcement learning with sql tailored partial rewards for reasoning-enhanced text-to-sql,” CoRR, 2025.

[40] Z. Yao, G. Sun, L. Borchmann, Z. Shen, M. Deng, B. Zhai, H. Zhang, A. Li, and Y. He, “Arctic-text2sql-r1: Simple rewards, strong reasoning in text-to-sql,” CoRR, 2025.

[41] H. Hao, W. Hu, O. Verkholyak, D. A. Tarzanagh, B. Gutow, S. Didari, M. Faraki, H. Moon, and S. Min, “Paverl-sql: Text-to-sql via partialmatch rewards and verbal reinforcement learning,” CoRR, 2025.