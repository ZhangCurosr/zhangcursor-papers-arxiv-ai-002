# Generated Query Expansion Still Helps Strong Sparse Retrieval: A Controlled Study with SPLADE-v3

Ryan C. Barron<sup>∗</sup>, Cade W. Trotter<sup>†</sup>, Maksim E. Eren<sup>∗</sup>, Kim Ø. Rasmussen<sup>‡</sup>, Liz D. Miller<sup>§</sup> Benjamin J. Migliori<sup>¶</sup> <sup>∗</sup>Computational Intelligence & Modeling, Los Alamos National Laboratory, Los Alamos, New Mexico, USA. <sup>†</sup>Modeling and Observations of Earth Systems, Los Alamos National Laboratory, Los Alamos, New Mexico, USA. <sup>‡</sup>Fluid Dynamics and Solid Mechanics, Los Alamos National Laboratory, Los Alamos, New Mexico, USA. <sup>§</sup>Intelligence & Systems Analysis, Los Alamos National Laboratory, Los Alamos, New Mexico, USA. <sup>¶</sup>Advanced Research in Cyber Systems, Los Alamos National Laboratory, Los Alamos, New Mexico, USA.

Abstract—Scientific queries are often brief, while relevant papers use specialized vocabulary. Generated query expansion can bridge this mismatch, but earlier work suggests that its value shrinks as the underlying retriever becomes stronger. We test the four generated formats of term lists, a pseudo-document, multiple pseudo-references, and corpus-steered text all together with SPLADE-v3 on NFCorpus, TREC-COVID, and SciDocs. Every condition searches the same frozen document index and follows the same query-side integration rule and 256-dimension budget, isolating the effect of the added content. All twelve method-collection comparisons improve aggregate nDCG@10, with best relative gains of 4.81%, 8.92%, and 9.47%. Eleven remain significant after Holm correction. The gain persists in 103 of 114 interpolation settings, including every setting that<sup>[</sup> assigns at least 30% of the mixture weight to the original query. Shuffled-text and non-contextual lexical-bag controls also remain above baseline in all 24 aggregate comparisons, showing that the added vocabulary carries most of the benefit. A corpus-induced typed concept graph, by contrast, produces no consistent gain, and its relation, depth, validation, random, and gating controls do not rescue it. Generated vocabulary can therefore complement a strong learned sparse retriever, provided that the original query remains strongly represented.

Index Terms—scientific information retrieval, query expansion, large language models, learned sparse retrieval, SPLADE, robustness analysis

## I. INTRODUCTION

Scientific search has a persistent vocabulary problem. A user may describe a phenomenon, material, or desired outcome in a few ordinary words, while the relevant literature uses specialized entities, method names, and domain terminology. Query expansion tries to close this gap by adding words that the user did not supply. Classical methods derive those words from judged or initially retrieved documents [1], [2], [3]. Newer methods ask language models to generate keywords, hypothetical documents, or several pseudo-references [4], [5], [6], [7]. The opportunity is clear, but so is the risk: generated text may introduce useful terminology, or it may pull the ranking away from the user’s intent.

This tradeoff is especially sharp for learned sparse retrieval. SPLADE already maps queries and documents to weighted vocabulary terms, including terms inferred from context, while retaining efficient inverted-index search [8], [9]. SPLADEv3 is a strong checkpoint in this family [10]. If its query representation already expands the user’s wording, what can an external generator still add? Prior work makes the answer uncertain: across many settings, generative expansion helps weaker retrievers more and can harm the strongest ones [11].

We answer this question with a controlled comparison. On NFCorpus, TREC-COVID, and SciDocs, we add four forms of generated content to the query side of SPLADE-v3: a short term list, one pseudo-document, four pseudo-references, and corpus-steered text. Every condition searches the same frozen SPLADE-v3 document index, uses the same candidate depth, and obeys the same sparse query budget. We also test HIQE, a structured alternative that selects concepts induced from the corpus, traverses typed relations, and projects the selected concepts into the same SPLADE vocabulary space. This comparison helps distinguish a benefit specific to generated vocabulary from a generic benefit of adding related terms.

The answer is consistent across the three collections. Every generated format improves aggregate nDCG@10, with best relative gains of 4.81% on NFCorpus, 8.92% on TREC-COVID, and 9.47% on SciDocs. The gains persist across a broad range of mixture weights and survive controls that shuffle the generated text or reduce it to a non-contextual lexical bag. The concept graph, however, does not show a comparable pattern. Together, the controls point to a straightforward explanation: the generator contributes useful scientific vocabulary, but that vocabulary works best as a supplement to a strongly preserved original query.

The paper makes three contributions.

1) A controlled strong-retriever test. Four generated formats improve one frozen SPLADE-v3 document index across three scientific-search collections.

2) Evidence about why the gains persist. Weight sweeps, paired tests, and same-content controls show that the benefit is broad and depends more on added vocabulary than on exact generated word order.

3) A structured counterpoint. A corpus-induced concept graph and its relation, depth, random, and gating controls fail to reproduce the generated gains, showing that queryside term addition alone is not sufficient.

## II. RELATED WORK

## A. Generated Query Expansion

Classical query expansion derives new vocabulary from the collection. Rocchio-style feedback moves the query toward judged or pseudo-relevant documents [1]. Relevance models estimate terms from initially retrieved evidence [2], [3]. These methods are corpus-grounded, but they can amplify first-stage errors when the feedback set mixes different interpretations of the query.

Language models provide another source of vocabulary. They can generate keyword lists [4], a single hypothetical passage as in Query2doc [5], a document representation as in HyDE [6], or multiple pseudo-references as in MuGI-style prompting [7]. Corpus-steered methods condition generation on retrieved evidence [12]. Each format covers the information need differently: lists are compact, passages can connect concepts, multiple references can represent several facets, and corpus steering can constrain the vocabulary while inheriting first-stage bias.

These benefits are not guaranteed. Weller et al. find that generative expansion generally helps weaker retrieval systems more than stronger ones and often harms the strongest retrievers [11]. A generator may also reproduce benchmark-specific evidence encountered during training instead of providing a transferable reformulation [13]. We hold the generated texts fixed so that our retrieval experiments measure integration effects rather than generation variability.

## B. Learned Sparse Retrieval and Integration

SPLADE learns contextual lexical expansion while retaining sparse retrieval [8], [9], where SPLADE-v3 is a strong modern baseline in this family [10]. Because its query and document vectors already contain learned expansion, an external reformulator must contribute genuinely complementary evidence. We isolate that contribution by freezing the document side and changing only the query representation.

Original and expanded evidence can be combined in several ways. Reciprocal rank fusion merges ranked lists without requiring a common score scale [14], and Exp4Fuse applies routelevel fusion to language-model expansion for sparse retrieval [15]. QuDAR instead assigns adaptive weights across original and expanded queries and across sparse and dense retrieval [16]. Our study focuses on fixed query-side interpolation because the shared sparse space supports a tightly controlled comparison between generated text and corpus-induced concepts.

## C. Structured Expansion and Evaluation

Knowledge-aware expansion and scientific taxonomy construction motivate explicit concept structure [17], [18]. Typed edges can represent relations among candidate additions, but a graph also adds several possible failure points: phrase extraction, relation induction, query-to-concept mapping, traversal, and projection. Even a well-formed graph may contain edges that are irrelevant to ranking. We therefore compare typed traversal with flat, relation, depth, validation, and random controls.

Finally, aggregate means can hide gains concentrated in a few topics. Following established information-retrieval practice [19], we supplement aggregate effectiveness with paired intervals, corrected p-values, standardized effects, and wins/ties/losses.

## III. STUDY DESIGN AND METHODS

The experiments follow three research questions:

RQ1 Does generated expansion improve a frozen SPLADEv3 retriever?

RQ2 Does any gain survive changes in mixture weight and in the representation of the same generated content?

RQ3 Can corpus-induced concepts, typed relations, or selective gates produce a comparable gain?

Figure 1 summarizes the shared retrieval pipeline.

## A. Shared Frozen-Index Pipeline

Let q be a query and d a document. A frozen SPLADEv3 encoder produces the base query vector $\mathbf { z } _ { q }$ and document vectors $\mathbf { z } _ { d } .$ . We build the document vectors once and reuse them in every matched condition. Only the query vector changes, and every condition scores documents as $s ( d , q ) = \widetilde { \mathbf { z } } _ { q } ^ { \top } \mathbf { z } _ { d }$ . For generated method $m ,$ let $\bar { \mathbf { z } } _ { q }$ and $\bar { \mathbf { a } } _ { m } ( \mathfrak { q } )$ denote row-wise $\ell _ { 1 } \cdot$ normalized base and added SPLADE vectors. We combine them as

$$
\widetilde { \mathbf { z } } _ { q } = \mathrm { B u d g e t } _ { 2 5 6 } \left( \alpha _ { m } \bar { \mathbf { z } } _ { q } + ( 1 - \alpha _ { m } ) \bar { \mathbf { a } } _ { m } ( q ) \right) ,\tag{1}
$$

Here, $\alpha _ { m }$ controls the balance: larger values preserve more of the original query, while smaller values give the expansion more influence. The final vector contains at most 256 active dimensions. We protect the original query within that budget: if the base vector has fewer than 256 dimensions, only additiononly dimensions are pruned. If the budget is exceeded, we retain its 256 highest-weight dimensions. Generated content and graph concepts are encoded separately, so they do not consume the original query’s transformer input length. Thus, the document representation, scorer, candidate depth, query budget, and metric implementation remain fixed across the matched conditions.

The configured original-query weights are $\alpha _ { m } = 0 . 6 5$ for Flat LLM-QE and CSQE and 0.50 for Query2doc and MuGIstyle expansion. Section IV-B tests sensitivity to this choice. Graph projection uses $\lambda _ { \mathrm { p r o j } } ~ = ~ 0 . 7 5$ under the same 256- dimension budget.

![](images/e09d1aa02434e8d9708b8ec1d7d63e4da044c017dcf05f4b3ef8657cda789569.jpg)  
Fig. 1. Controlled study design. The original query, generated expansion, and corpus-induced concept expansion are encoded using the same frozen SPLADE-v3 query model. Added evidence is combined with the protected original-query representation under a fixed 256-dimension sparse budget, and every condition searches the same frozen document index before paired query-level evaluation.

## B. Generated Expansion Conditions

The four generated conditions differ only in the form of the added text. Flat LLM-QE produces up to twelve search terms. Query2doc produces one hypothetical relevant passage [5]. MuGI-style expansion produces four pseudo-references, providing several views of the information need [7]. Corpus-Steered Query Expansion (CSQE) first retrieves evidence, then generates pivotal sentences and knowledge terms conditioned on that evidence [12]. In every case, SPLADE encodes the added text and Eq. 1 combines it with the protected base query.

The run archive preserves the generated texts and the LLM response cache. The cache records the served model identifier gpt-oss-120b, the system and user prompts, request parameters (including temperature 0.0 and the output limit), and raw responses. The retrieval experiments can therefore be rerun from the exact fixed expansions. Independent regeneration should also use the provider and model-revision metadata distributed with the release. The four generated conditions are not matched for model calls, token use, or latency. The study compares retrieval effectiveness rather than generation efficiency.

## C. Interpolation and Same-Content Controls

We sweep $\alpha \in \{ 0 . 1 , 0 . 2 , \ldots , 0 . 9 \}$ and also evaluate the configured 0.65 for Flat LLM-QE and CSQE. At the configured $\alpha _ { m }$ , two controls preserve the generated content while changing its representation. The first deterministically shuffles whitespace tokens before applying the same SPLADE encoder, then the second forms an ℓ -normalized term-frequency bag from the same tokenizer wordpieces, removing contextual SPLADE expansion. MuGI pseudo-references remain separately normalized and averaged. Alternative representations use the same paired tests, with Holm correction across eight comparisons per collection.

## D. Corpus-Induced Structured Comparison

The structured condition extracts one- to three-token TF–IDF keyphrases from scientific titles and abstracts, then records document–concept assignments. Instead of generating text, HIQE expands a query with related concepts induced from the corpus. Typed broader, sibling, and related edges are derived from phrase relations, dense similarity, shared parents, cooccurrence, and corpus support. For the publication run, edge validation is deterministic and grounded in corpus evidence. A separate experimental variant adds LLM-based evidence judgments, but those judgments are not used in the main results.

The query mapper combines dense similarity, lexical overlap, and aliases. It retains at most four seed concepts, traverses at most two edges, and keeps at most twelve expansion concepts. For a path $\pi = ( e _ { 1 } , \ldots , e _ { \ell } )$ from seed $c _ { 0 }$ to concept $c ,$ the path score is

$$
\Omega ( \pi \mid q ) = p ( c _ { 0 } \mid q ) \left( \prod _ { j = 1 } ^ { \ell } \alpha _ { r ( e _ { j } ) } v ( e _ { j } ) \right) \kappa ^ { \mathrm { m a x } ( 0 , \ell - 1 ) } ,\tag{2}
$$

where $v ( e )$ is edge confidence, $\alpha _ { r }$ is a relation prior, and the depth decay is $\kappa = 0 . 7 5$ . Before frozen-SPLADE encoding, the projection text combines the concept name with optional aliases and snippets from representative documents. The resulting graph vector is mixed with the base query using $\lambda _ { \mathrm { p r o j } } = 0 . 7 5$ under the 256-dimension budget. Controls remove validation, equalize relation weights, restrict relation families, reduce traversal depth, flatten concept scores, or substitute random count-matched concepts.

A heuristic gate uses hand-set activation and confidence criteria. A class-balanced logistic-regression gate is trained on two collections and applied to the held-out third collection in a leave-one-dataset-out protocol. We report gate activation, harmful ungated expansions avoided, beneficial ungated expansions missed, and the gap to a per-query oracle that selects the better of unexpanded SPLADE-v3 and ungated graph expansion. The oracle measures the maximum gain obtainable from perfect per-query selection of the two already-computed rankings.

## E. Collections, Baselines, and Statistical Analysis

We evaluate three scientific-search collections distributed through the BEIR benchmark suite: NFCorpus, TREC-COVID, and SciDocs [20], [21], [22]. The three collections contain 3,633, 171,332, and 25,657 documents and 323, 50, and 1,000 test queries, respectively. We report each collection separately because their query counts and relevance structures differ substantially.

The main comparison includes BM25 [23], RM3, TAS-B dense retrieval, BM25+dense reciprocal-rank fusion, Col-BERTv2 late interaction [24], SPLADE-v3, the four generated conditions, and three graph conditions. BM25 provides a lexical baseline and RM3 adds pseudo-relevance feedback. TAS-B is a dense bi-encoder, reciprocal-rank fusion combines the BM25 and dense lists, and ColBERTv2 is a late-interaction retriever. SPLADE-v3 is the frozen learned-sparse baseline directly matched to our query-expansion conditions. Candidate depth is 1,000. nDCG@10 is the primary metric. MRR@10, MAP@10, Precision@10, Recall@10, Recall@100, and candidate recall are secondary aggregate measures. nDCG@10 measures the quality of the top-10 ranking. MRR@10 emphasizes how early the first relevant result appears. MAP@10 summarizes precision across relevant results in the top 10. Recall@100 measures deeper coverage.

From aligned per-query nDCG@10 values, we report mean difference, relative change, Cohen’s paired $d _ { z } ,$ , wins/ties/losses, a 95% interval from 20,000 paired bootstrap resamples, and a two-sided test from 100,000 paired sign randomizations (seed 13). Generated-method p-values use Holm correction across the four methods within each collection. The three ungated graph comparisons use Bonferroni correction across collections. A win or loss is defined by the sign of the per-query nDCG@10 difference, while exact equality is retained as a tie. No manual semantic annotation of graph edges was conducted.

## IV. RESULTS

Each subsection answers one research question first, then presents the evidence supporting that answer.

## A. RQ1: Does Generated Expansion Improve SPLADE-v3?

Answer to RQ1. Yes. All four generated formats improve the matched SPLADE-v3 baseline on all three collections. The best relative nDCG@10 gains are 4.81% on NFCorpus, 8.92% on TREC-COVID, and 9.47% on SciDocs.

Table I shows that CSQE is strongest on NFCorpus and SciDocs, while Query2doc is strongest on TREC-COVID. The corresponding nDCG@10 gains over unexpanded SPLADEv3 are 4.81%, 8.92%, and 9.47%. No single format wins everywhere: CSQE is the weakest generated condition on TREC-COVID, and Query2doc does not lead the other two collections. The effect is therefore shared across generation styles rather than driven by one universally best method.

The broader baseline comparison is less uniform. CSQE exceeds the strongest listed non-SPLADE nDCG@10 baseline by 0.0304 on NFCorpus and 0.0039 on SciDocs; Query2doc exceeds ColBERTv2 by 0.0823 on TREC-COVID. For Recall@100, however, RM3 remains best on NFCorpus (0.3229), while Flat LLM-QE is best on SciDocs (0.3973). Generated expansion reliably improves the matched SPLADE-v3 run, but the best system still depends on the collection and metric.

a) Evidence across metrics.: The strongest generated condition on each collection also improves all six secondary measures in Table II. Notably, Recall@100 rises by 6.20% on NFCorpus, 10.44% on TREC-COVID, and 5.57% on SciDocs, while MAP@10 rises by 5.30%, 6.48%, and 12.58%. The benefit therefore appears in both top-ranked relevance and deeper first-stage coverage.

b) Evidence across queries.: All twelve paired mean differences are positive, and eleven remain significant after Holm correction. The only exception is CSQE on the 50-topic TREC-COVID collection (95% interval [−0.0036, 0.0604], adjusted $p = 0 . 0 9 4 4 )$ . Across the twelve comparisons, paired effect sizes range from approximately $d _ { z } = 0 . 1 4$ to 0.47.

For the best method on each collection, wins outnumber losses among non-tied queries: 99 versus 49 for NFCorpus CSQE, 34 versus 11 for TREC-COVID Query2doc, and 262 versus 127 for SciDocs CSQE (Table III). Many queries remain unchanged, especially on NFCorpus and SciDocs, but the aggregate gains are not driven by only a few large improvements.

## B. RQ2: What Makes the Gain Robust?

Answer to RQ2. The gain does not depend on one favorable interpolation weight or on the exact form of the generated text. It persists when the original query retains at least 30% of the mixture weight and when the generated content is shuffled or reduced to a lexical bag. The most stable source of improvement is therefore the added vocabulary.

a) Weight sensitivity.: Across 114 interpolation settings, 103 beat SPLADE-v3 (Fig. 3). MuGI-style is positive at all 27 evaluated method–collection weights, CSQE at 28/30, Flat LLM-QE at 24/30, and Query2doc at 24/27. Every failure occurs at α = .1 or .2: Flat LLM-QE fails at both values on all three collections, Query2doc at .1 on all three, and CSQE at .1 and .2 on TREC-COVID. Consequently, all 90 settings with $\alpha \geq . 3$ remain above baseline. Generated content can receive substantial weight, but letting it dominate the original query is risky.

b) Representation controls.: Both same-content controls remain above SPLADE-v3 in all 24 aggregate method– collection comparisons. Moreover, 23 of 24 controls do not differ significantly from their contextual counterpart after Holm correction. The only exception is shuffled Query2doc on SciDocs $( \Delta = - . 0 0 2 9 , p _ { \mathrm { H o l m } } = . 0 4 3 4 )$ , which still beats the baseline. Nine alternative representations even score higher in aggregate than the corresponding contextual representation, although none of those increases is significant after correction. Shuffling therefore removes coherent word order, and the lexical-bag control removes contextual SPLADE expansion, yet both retain the positive effect. The common ingredient is the vocabulary supplied by the generator.

TABLE I  
FIRST-STAGE EFFECTIVENESS, WITH NDCG@10 AS THE PRIMARY METRIC, R@100 AS DEEPER COVERAGE, AND BOLD MARKING THE BEST VALUE IN EACH DATASET COLUMN.
<table><tr><td></td><td colspan="2">NFCorpus</td><td colspan="2">TREC-COVID</td><td colspan="2">SciDocs</td></tr><tr><td>Method</td><td>nDCG@10</td><td>R@100</td><td>nDCG@10</td><td>R@100</td><td>nDCG@10</td><td>R@100</td></tr><tr><td>BM25</td><td>0.3231</td><td>0.2457</td><td>0.5696</td><td>0.1091</td><td>0.1490</td><td>0.3477</td></tr><tr><td>RM3</td><td>0.3465</td><td>0.3229</td><td>0.5635</td><td>0.1168</td><td>0.1491</td><td>0.3620</td></tr><tr><td>TAS-B dense</td><td>0.2755</td><td>0.2487</td><td>0.3087</td><td>0.0335</td><td>0.1406</td><td>0.3222</td></tr><tr><td>BM25 + dense RRF</td><td>0.3342</td><td>0.2819</td><td>0.5511</td><td>0.0894</td><td>0.1690</td><td>0.3848</td></tr><tr><td>ColBERTv2</td><td>0.3392</td><td>0.2803</td><td>0.7100</td><td>0.1307</td><td>0.1482</td><td>0.3543</td></tr><tr><td>SPLADE-v3</td><td>0.3596</td><td>0.2971</td><td>0.7275</td><td>0.1388</td><td>0.1579</td><td>0.3709</td></tr><tr><td>Flat LLM-QE</td><td>0.3715</td><td>0.3060</td><td>0.7850</td><td>0.1532</td><td>0.1688</td><td>0.3973</td></tr><tr><td>Query2doc</td><td>0.3724</td><td>0.3102</td><td>0.7924</td><td>0.1533</td><td>0.1673</td><td>0.3923</td></tr><tr><td>MuGI-style</td><td>0.3701</td><td>0.3130</td><td>0.7840</td><td>0.1533</td><td>0.1668</td><td>0.3890</td></tr><tr><td>CSQE</td><td>0.3769</td><td>0.3155</td><td>0.7556</td><td>0.1481</td><td>0.1729</td><td>0.3916</td></tr><tr><td>HiQE, ungated</td><td>0.3609</td><td>0.2961</td><td>0.7149</td><td>0.1379</td><td>0.1585</td><td>0.3717</td></tr><tr><td>HiQE, heuristic gate</td><td>0.3596</td><td>0.2971</td><td>0.7275</td><td>0.1388</td><td>0.1579</td><td>0.3709</td></tr><tr><td>HiQE, learned gate</td><td>0.3603</td><td>0.2958</td><td>0.7149</td><td>0.1379</td><td>0.1579</td><td>0.3709</td></tr></table>

TABLE II

RELATIVE IMPROVEMENT (%) OVER SPLADE-V3 FOR THE STRONGEST GENERATED NDCG@10 CONDITION ON EACH COLLECTION.
<table><tr><td>Dataset</td><td>Method</td><td>nDCG@10</td><td>MRR@10</td><td>MAP@10</td><td>P@10</td><td>R@10</td><td>R@100</td><td>Cand. recall</td></tr><tr><td>NFCorpus</td><td>CSQE</td><td>+4.81</td><td>+5.10</td><td>+5.30</td><td>+4.45</td><td>+1.81</td><td>+6.20</td><td>+6.28</td></tr><tr><td>TREC-COVID</td><td>Query2doc</td><td>+8.92</td><td>+3.74</td><td>+6.48</td><td>+3.91</td><td>+6.99</td><td>+10.44</td><td>+5.11</td></tr><tr><td>SciDocs</td><td>CSQE</td><td>+9.47</td><td>+9.16</td><td>+12.58</td><td>+7.38</td><td>+7.39</td><td>+5.57</td><td>+3.14</td></tr></table>

TABLE III

COMPACT PAIRED PER-QUERY NDCG@10 ANALYSIS COMPARING, FOR EACH COLLECTION, THE STRONGEST AGGREGATE GENERATED CONDITION AND UNGATED HIQE WITH SPLADE-V3.
<table><tr><td>Dataset</td><td>Method</td><td>∆</td><td>95% CI</td><td> $p _ { \mathrm { a d j } }$ </td><td> $d _ { z }$ </td><td>W/T/L</td></tr><tr><td>NFCorpus</td><td>CSQE</td><td>+0.0173</td><td>[+0.0106, +0.0241]</td><td>4e-05</td><td>+0.28</td><td>99/175/49</td></tr><tr><td>NFCorpus</td><td>HIQE, ungated</td><td>+0.0013</td><td>[-0.0007, +0.0034]</td><td>0.661</td><td>+0.07</td><td>40/246/37</td></tr><tr><td>TREC-COVID TREC-COVID</td><td>Query2doc HIQE, ungated</td><td>+0.0649 -0.0126</td><td>[+0.0273, +0.1030] [-0.0259, +0.0007]</td><td>0.00540 0.207</td><td>+0.47 -0.26</td><td>34/5/11 16/12/22</td></tr><tr><td>SciDocs</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SciDocs</td><td>CSQE</td><td>+0.0150</td><td>[+0.0111, +0.0189]</td><td>4e-05</td><td>+0.24</td><td>262/611/127</td></tr><tr><td></td><td>HIQE, ungated</td><td>+0.0006</td><td>[-0.0011, +0.0022]</td><td>1.000</td><td>+0.02</td><td>77/843/80</td></tr></table>

Generated-method rows use Holm adjustment across the four generated methods within that collection. Ungated graph rows use Bonferroni adjustment across the three collection-level graph comparisons.

## C. RQ3: Can Structured Expansion Match the Gain?

Answer to RQ3. No. Ungated HIQE changes nDCG@10 by only +.0013 on NFCorpus, −.0126 on TREC-COVID, and +.0006 on SciDocs. All three paired 95% intervals include zero, and none is significant after Bonferroni correction. The ablations in Table V likewise reveal no consistently useful relation, weighting, validation, or depth choice.

a) Ablation evidence.: Flat equal-weight concepts are worse than the full graph on all three collections. Removing corpus validation causes the largest TREC-COVID degradation, while parent-only traversal has the least negative mean among the named graph variants. More structure is not consistently better: the random count-matched control has a less negative mean than the full graph, and depth two is slightly worse on average than depth one.

The stored hierarchy diagnostics show concept coverage of 100.0% on NFCorpus, 99.71% on TREC-COVID, and above 99.99% on SciDocs. The archive does not include connectedcomponent or broader/narrower cycle counts, so we make no stronger topology claim. The available evidence is nevertheless clear: nearly complete concept coverage does not translate into a ranking gain.

b) Gating does not rescue the graph.: The stored rankings uniquely establish that the learned gate selects graph expansion for all 50 TREC-COVID topics. It therefore preserves the TREC-COVID graph degradation, while a per-query oracle remains 0.0232 nDCG@10 above the learned gate there. On NFCorpus and SciDocs, exact activation counts cannot be recovered for 14 and 13 queries because the base and graph rankings are identical and the archive lacks gate-decision logs. Among the remaining queries, inferred learned-gate activation is 37.9% on NFCorpus and 0% on SciDocs. The heuristic gate selects no graph run on the unambiguous NFCorpus or SciDocs queries and none on TREC-COVID. Neither selector therefore offers a consistent advantage over leaving SPLADE-

![](images/ebf50277c076e43c4f9b5f9071cd9aedfc5e5908a1f25e9072bfa4cb3c76793e.jpg)  
Fig. 2. Paired nDCG@10 differences from SPLADE-v3. Points show mean per-query differences, and bars show paired 95% bootstrap intervals. Filled markers indicate significance after multiple-comparison correction, while hollow markers indicate nonsignificant effects. The graph condition denotes ungated HIQE.

TABLE IV  
INTERPOLATION ROBUSTNESS AND SAME-CONTENT CONTROLS. “POSITIVE α” COUNTS TESTED WEIGHTS ABOVE SPLADE-V3. BOW AND SHUFFLE ARE ABSOLUTE ∆NDCG@10 AT THE CONFIGURED α.
<table><tr><td>Method</td><td>Positive α BOW ∆</td><td>Shuffle ∆</td></tr><tr><td>NFCorpus Flat LLM-QE Query2doc MuGI-style CSQE</td><td>8/10 8/9 9/9 10/10</td><td>+.0094 +.0104 +.0107 +.0141 +.0085 +.0104 +.0180 +.0167</td></tr><tr><td>TREC-COVID Flat LLM-QE Query2doc MuGI-style CSQE</td><td>8/10 8/9 9/9 8/10</td><td>+.0620 +.0604 +.0358 +.0585 +.0469 +.0557 +.0219</td></tr><tr><td>SciDocs</td><td></td><td>+.0238</td></tr><tr><td>Flat LLM-QE</td><td>8/10 +.0129</td><td>+.0107 +.0065†</td></tr><tr><td>Query2doc MuGI-style CSQE</td><td>8/9 9/9 10/10 +.0157</td><td>+.0084 +.0099 +.0112 +.0159</td></tr></table>

<sup>†</sup>Only shuffled Query2doc on SciDocs differs significantly from its contextual counterpart after within-collection Holm correction $\mathsf { \bar { ( } } p = . 0 \dot { 4 } 3 4 )$ . It remains above SPLADE-v3.

v3 unexpanded.

## V. DISCUSSION

The surprising result is that generated expansion still helps after SPLADE-v3 has already performed its own learned lexical expansion. Earlier evidence suggests that generative expansion becomes less useful as the retriever grows stronger [11]; here, four generation formats improve a strong frozen retriever on all three collections. Because the document index and retrieval pipeline are fixed, the difference comes from the query-side evidence.

The controls identify that evidence more precisely. Performance remains positive across a broad range of interpolation weights, so the result is not an artifact of one favorable coefficient. Shuffled text retains the gain, showing that coherent generated word order is not essential. A non-contextual lexical bag also retains the gain, showing that the contextual encoding of the generated passage is not essential either. Across the experiments, the most stable common factor is the added scientific vocabulary: entities, methods, and specialized terms that were absent from the short query.

That vocabulary should complement, not replace, the user’s wording. Every tested configuration with α ≥ .3 beats the baseline, while all failures occur when the original query receives only 10% or 20% of the mixture weight. This pattern supports integration methods that preserve substantial originalquery mass and adapt the expansion weight by query, as in QuDAR [16].

![](images/d33fa518f47ebdb9d03318ae1c89ab878875c0301beb5a4866d009803ccc949e.jpg)

![](images/02afd4d9d81f8b8b4fe32be07c073453bab413f4e931cb23edeef5ab7ee2753d.jpg)

![](images/b8122385684041715f7f58fd13e9377cd6b211cd5b84705c65c7000a571b6669.jpg)  
Fig. 3. Interpolation sensitivity versus SPLADE-v3 as the original-query weight α varies. Stars mark the configured weights. Most method–collection pairs remain above baseline across a broad range, while low α degrades Flat LLM-QE, Query2doc, and TREC-COVID CSQE.

TABLE V  
HIERARCHY ABLATIONS AS ABSOLUTE ∆NDCG@10 RELATIVE TO SPLADE-V3, WITH MEAN ∆ DEFINED AS THE ARITHMETIC MEAN ACROSS THE THREE COLLECTIONS.
<table><tr><td>Variant</td><td>NFCorpus</td><td>TREC-COVID</td><td>SciDocs</td><td>Mean ∆</td></tr><tr><td>Random count-matched</td><td>-.0001</td><td>-.0060</td><td>+.0003</td><td>-.0020</td></tr><tr><td>Flat concepts, equal scores</td><td>-.0048</td><td>-.0223</td><td>-.0002</td><td>-.0091</td></tr><tr><td>No corpus validation</td><td>+.0006</td><td>-.0256</td><td>+.0006</td><td>-.0081</td></tr><tr><td>Equal relation weights</td><td>+.0005</td><td>-.0192</td><td>+.0014</td><td>-.0058</td></tr><tr><td>Parent only</td><td>+.0002</td><td>-.0054</td><td>+.0005</td><td>-.0016</td></tr><tr><td>Child only</td><td>+.0006</td><td>-.0093</td><td>+.0004</td><td>-.0028</td></tr><tr><td>Sibling only</td><td>+.0008</td><td>-.0099</td><td>+.0003</td><td>-.0030</td></tr><tr><td>Depth 1</td><td>+.0015</td><td>-.0129</td><td>+.0010</td><td>-.0035</td></tr><tr><td>Full graph, depth 2</td><td>+.0013</td><td>-.0126</td><td>+.0006</td><td>-.0036</td></tr></table>

The lexical controls also suggest a practical design choice. A generated passage can be used internally as a source of candidate terms; it need not be displayed to the user or treated as a factual answer. Separating retrieval utility from user-facing generation reduces the importance of the passage’s prose quality while keeping attention on the terms that affect ranking.

The graph comparison shows that adding related terms is not sufficient by itself. Corpus support, typed relations, deeper traversal, and relation-specific weighting do not reproduce the generated gains in this pipeline. Future structured systems may need retrieval-supervised concept selection, phrase-preserving representations, and gates trained directly for per-query ranking impact.

## A. Limitations

The study covers one SPLADE-v3 configuration and three English BEIR collections. Other retrievers, languages, domains, and interactive settings may produce different gains. The public benchmarks may also have appeared in the generator’s training data; we do not use a documented-cutoff generator, analyze corpus overlap, or include a paraphrased-query control [13].

The primary methods use fixed interpolation weights rather than a shared development-set tuning protocol. The sweep demonstrates a broad positive region, but it is not a substitute for prospective tuning. Finally, the graph is built automatically and validated with corpus evidence rather than manually labeled edges. Its negative results may reflect graph construction, graphguided integration, or both.

## VI. CONCLUSION

Generated query expansion can improve a strong learned sparse retriever even when that retriever already performs contextual lexical expansion. Across three scientific collections, all four generated formats improve SPLADE-v3, and the gains survive wide changes in mixture weight, token order, and contextual representation. The tested concept graph does not show the same benefit. The clearest interpretation is that generation contributes useful scientific vocabulary—but it works as an addition to a strongly preserved original query, not as a replacement for it.

## ACKNOWLEDGMENT

This manuscript has been approved for unlimited release and has been assigned LA-UR-26-28002. The funding for this paper was provided by Los Alamos National Laboratory (LANL). LANL is operated by Triad National Security, LLC, for the National Nuclear Security Administration of the U.S. Department of Energy (Contract No. 89233218CNA000001).

## REFERENCES

[1] J. J. Rocchio, “Relevance feedback in information retrieval,” in The SMART Retrieval System: Experiments in Automatic Document Processing (G. Salton, ed.), pp. 313–323, Prentice-Hall, 1971.

[2] V. Lavrenko and W. B. Croft, “Relevance-based language models,” in Proceedings of the 24th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 120–127, 2001.

[3] N. Abdul-Jaleel, J. Allan, W. B. Croft, F. Diaz, L. Larkey, X. Li, M. D. Smucker, and C. Wade, “Umass at trec 2004: Novelty and hard,” in Proceedings of the Thirteenth Text REtrieval Conference, 2004.

[4] R. Jagerman, H. Zhuang, Z. Qin, X. Wang, and M. Bendersky, “Query expansion by prompting large language models,” arXiv preprint arXiv:2305.03653, 2023.

[5] L. Wang, N. Yang, and F. Wei, “Query2doc: Query expansion with large language models,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, (Singapore), pp. 9414–9423, Association for Computational Linguistics, Dec. 2023.

[6] L. Gao, X. Ma, J. Lin, and J. Callan, “Precise zero-shot dense retrieval without relevance labels,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, pp. 1762–1777, 2023.

[7] L. Zhang, Y. Wu, Q. Yang, and J.-Y. Nie, “Exploring the best practices of query expansion with large language models,” in Findings of the Association for Computational Linguistics: EMNLP 2024, (Miami, Florida, USA), pp. 1872–1883, Association for Computational Linguistics, Nov. 2024.

[8] T. Formal, B. Piwowarski, and S. Clinchant, “Splade: Sparse lexical and expansion model for first stage ranking,” in Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2021. arXiv:2107.05720.

[9] T. Formal, C. Lassance, B. Piwowarski, and S. Clinchant, “Splade v2: Sparse lexical and expansion model for information retrieval,” arXiv preprint arXiv:2109.10086, 2021.

[10] C. Lassance, H. Déjean, T. Formal, and S. Clinchant, “Splade-v3: New baselines for splade,” arXiv preprint arXiv:2403.06789, 2024.

[11] O. Weller, K. Lo, D. Wadden, D. Lawrie, B. Van Durme, A. Cohan, and L. Soldaini, “When do generative query and document expansions fail? a comprehensive study across methods, retrievers, and datasets,” in Findings of the Association for Computational Linguistics: EACL 2024, (St. Julian’s, Malta), pp. 1987–2003, Association for Computational Linguistics, Mar. 2024.

[12] Y. Lei, Y. Cao, T. Zhou, T. Shen, and A. Yates, “Corpus-steered query expansion with large language models,” in Proceedings of the 18th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics (Volume 2: Short Papers), (St. Julian’s, Malta), pp. 393–401, Association for Computational Linguistics, Mar. 2024.

[13] Y. Yoon, J. Jung, S. Yoon, and K. Park, “Hypothetical documents or knowledge leakage? rethinking LLM-based query expansion,” in Findings of the Association for Computational Linguistics: ACL 2025, pp. 19170– 19187, Association for Computational Linguistics, 2025.

[14] G. V. Cormack, C. L. A. Clarke, and S. Buettcher, “Reciprocal rank fusion outperforms condorcet and individual rank learning methods,” in Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 758–759, 2009.

[15] L. Liu and M. Zhang, “Exp4Fuse: A rank fusion framework for enhanced sparse retrieval using large language model-based query expansion,” in Findings of the Association for Computational Linguistics: ACL 2025, pp. 163–173, Association for Computational Linguistics, 2025.

[16] J. Kim, S. Yoon, X.-B. Le, Y. Nam, D. Kim, H. Song, and J.-G. Lee, “QuDAR: Query-wise dual-perspective adaptive retrieval,” in Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), (San Diego, California, United States), pp. 38662–38679, Association for Computational Linguistics, July 2026.

[17] Y. Xia, J. Wu, S. Kim, T. Yu, R. A. Rossi, H. Wang, and J. McAuley, “Knowledge-aware query expansion with large language models for textual and relational retrieval,” in Proceedings of the 2025 Conference of the Nations ofthe Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, (Albuquerque, New Mexico), pp. 4275–4286, Association for Computational Linguistics, Apr. 2025.

[18] P. Kargupta, N. Zhang, Y. Zhang, R. Zhang, P. Mitra, and J. Han, “TaxoAdapt: Aligning LLM-based multidimensional taxonomy construction to evolving research corpora,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, (Vienna, Austria), pp. 29834–29850, Association for Computational Linguistics, July 2025.

[19] M. D. Smucker, J. Allan, and B. Carterette, “A comparison of statistical significance tests for information retrieval evaluation,” in Proceedings of the Sixteenth ACM Conference on Information and Knowledge Management, pp. 623–632, ACM, 2007.

[20] N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, and I. Gurevych, “Beir: A heterogeneous benchmark for zero-shot evaluation of information retrieval models,” in Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, 2021.

[21] V. Boteva, D. Gholipour, A. Sokolov, and S. Riezler, “A full-text learning to rank dataset for medical information retrieval,” in Advances in Information Retrieval, pp. 716–722, Springer, 2016.

[22] E. M. Voorhees, T. Alam, S. Bedrick, D. Demner-Fushman, W. R. Hersh, K. Lo, K. Roberts, I. Soboroff, and L. L. Wang, “Trec-covid: Constructing a pandemic information retrieval test collection,” SIGIR Forum, vol. 54, no. 1, pp. 1–12, 2021.

[23] S. Robertson and H. Zaragoza, “The probabilistic relevance framework: Bm25 and beyond,” Foundations and Trends in Information Retrieval, vol. 3, no. 4, pp. 333–389, 2009.

[24] K. Santhanam, O. Khattab, J. Saad-Falcon, C. Potts, and M. Zaharia, “Colbertv2: Effective and efficient retrieval via lightweight late interaction,” in Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 3715–3734, 2022.