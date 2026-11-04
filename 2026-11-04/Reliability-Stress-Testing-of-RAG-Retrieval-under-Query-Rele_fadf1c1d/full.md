# Reliability Stress Testing of RAG Retrieval under Query-Relevant Corpus Poisoning

Janet Sung

Kainan Zhou

Gangzhen Qian

Google LLC

Google LLC

Lu Yi

Hang Xiao

Google LLC

Mountain View, USA

Fortinet, Inc.

Mountain View, USA

Google LLC

Mountain View, USA

Sunnyvale, USA

Mountain View, USA

janetsung@google.com zhoumark@google.com irisqian@google.com hangxiao.pisces@gmail.com annabelyi@google.com

Abstract—External text can affect a retrieval-augmented generation (RAG) system only after the retriever admits it to the model context. We stress this admission step while varying document content, attacker query information, per-target poison budget, and contamination scope. Across 623 Benchmarking Information Retrieval (BEIR) queries from SciFact and NFCorpus, five query-targeted documents per query reduce classical-retriever nDCG@10 by 40.2–61.5% in isolated-target stress. Full-query anchoring explains most Best Matching 25 (BM25) degradation, whereas appended text also shifts latent semantic analysis (LSA) and Hybrid rankings. A pretrained BGE-small validation reaches 100% target exposure for query-only P = 5 on both datasets; adding the injection payload changes SciFact nDCG only from 0.269 to 0.271 and lowers NFCorpus exposure slightly to 98.8%, indicating that query anchoring remains the dominant retrievalside factor for this dense model. Query length materially changes the interpretation of partial-information attacks: on 162 long SciFact queries, two random terms cover 19.9% of eligible terms and still yield 39.5% Hybrid target exposure. Multitarget contamination adds cross-target interference. Overlapbased admission can remove full-query poisons but may sharply reduce clean retrieval utility, and maximal marginal relevance (MMR) leaves exposure unchanged under the tested configuration. The study measures retrieval-layer failures, not downstream instruction following.

Index Terms—retrieval-augmented generation, reliability, prompt injection, corpus poisoning, information retrieval, admission control

## I. INTRODUCTION

A retrieval-augmented generation (RAG) system selects evidence from an external corpus before passing it to a language model [1]. Some indexed documents may be controlled by third parties. In corpus poisoning, an attacker inserts adversarial material into these retrievable sources; a document resembling the user’s query may enter the top-k context and displace legitimate evidence before any malicious instruction is executed.

For example, an attacker can publish a page repeating terms from a scientific question and append misleading text. Although unable to change the query or retriever, the attacker can influence which evidence is admitted. We measure that retrieval boundary, not downstream instruction following.

Prior work studies knowledge corruption, retrieval jamming, optimized triggers, and indirect prompt injection [2]–[5]. We isolate retrieval-stage effects that are difficult to distinguish when document content, query knowledge, poison budget, and contamination scope change together.

We organize the evaluation around three research questions: RQ1. What makes a poisoned document retrievable?

RQ2. How does retrieval failure scale with attacker query information and contamination scope?

RQ3. What does admission control trade for lower exposure? Public Benchmarking Information Retrieval (BEIR) labels and Benchmark for Indirect Prompt Injection Attacks (BIPIA) payload text support matched content controls, isolated and multitarget contamination, graded query-information conditions, and post-filter measurements.

We measure ranking quality, poison exposure, retained evidence, safe retrieval reliability, and latency. Query-information analysis accounts for effective query length: two or four terms disclose different fractions of short and long queries. The experiments distinguish query anchoring, cross-target contamination, and the evidence cost of admission rules.

## II. RELATED WORK

## A. Indirect Injection

BIPIA benchmarks indirect prompt injection and associated defenses [5]; InjecAgent extends indirect injection to toolintegrated agents [6]. Those studies evaluate downstream instruction following. Here, BIPIA payloads are used as standardized appended content, and the measured endpoint is retrieval admission.

## B. Retrieval Poisoning and Jamming

PoisonedRAG jointly optimizes retrieval and generation objectives [2], and AgentPoison targets agent memory and RAG knowledge bases with optimized triggers [7]. Machine Against the RAG shows that retrieval disruption need not rely on instruction semantics [3]. Chang et al. characterize a retrieval barrier and optimize triggers across modern embedding models [4]. Here, query information, per-target budget, and contamination scope are varied directly so their retrieval effects can be measured separately.

## C. Reliability and Admission Defenses

ReliabilityRAG uses document-reliability signals and provides stronger corruption defenses and robustness guarantees [8]. We use an interpretable overlap rule to measure the trade-off between lower poison exposure and loss of legitimate evidence; maximal marginal relevance (MMR) [13] provides a diversity-reranking comparison. The measurement distinction is that the closest prior work primarily studies downstream attack effects or optimized retrieval disruption, whereas our evaluation makes query information and contamination scope explicit stress axes and reports poison exposure together with retained evidence and ranking utility.

![](images/6d707d24a992ca523d307c15d84fccab5cce39405cedecccae6e1db04b526c66.jpg)  
Fig. 1. System model and retrieval measurement boundary. Attacker-controlled documents enter the corpus; retrieval and optional admission determine the final top-k set. Downstream instruction following is not evaluated.

## III. METHODOLOGY AND SYSTEM MODEL

Figure 1 shows the system model: attacker-controlled documents enter a fixed corpus, retrieval ranks the combined collection, and optional reranking or filtering determines the admitted candidates. We independently vary content, query knowledge, and contamination scope, then measure exposure and utility. Downstream generation is excluded.

## A. Threat Model and Stress Axes

Let D be a clean corpus, q a benchmark query, $\mathcal { R } _ { q }$ its judged relevant documents, and $\mathbf { \boldsymbol { A } } _ { q } ( \boldsymbol { P } )$ a set of $P$ query-targeted poisoned documents. For a set of contaminated targets ${ \mathcal { G } } ,$ the poison corpus is

$$
{ \mathcal { A } } = \bigcup _ { q \in { \mathcal { G } } } { \mathcal { A } } _ { q } ( P ) , \qquad { \mathcal { T } } _ { k } ( q ) = \operatorname { T o p K } ( q , { \mathcal { D } } \cup { \mathcal { A } } ) .\tag{1}
$$

The attacker can publish indexed content but cannot modify the benchmark query, clean documents, relevance labels, retriever parameters, or $k .$ We vary three stress axes. Per-target budget is $P \in \{ 1 , 3 , 5 \}$ . Contamination scope ranges from one target at a time to all evaluation targets, and is reported with both the number of poison targets/documents and global corpus fraction. Query information ranges from no query terms, through two or four target-query terms, to the full query. Eligible terms are unique lowercase alphanumeric tokens of length at least three, after stop-word removal and restriction to the clean BM25 vocabulary; if fewer than the requested two or four terms remain, all eligible terms are used. High-inverse-documentfrequency (IDF) partial anchors are an informed/oracle-like condition because IDF is computed from the clean corpus; deterministic random-query-term anchors use the same term budget without corpus-statistics knowledge. We report the actual fraction of eligible query terms revealed so that a fixed term budget is not mistaken for a fixed information fraction. These conditions are stress assumptions, not estimates of real-world attack frequency. Thus, query information and contamination scope are controlled directly rather than left implicit in a single attack-strength setting.

## B. Retrieval Reliability Metrics

Target exposure is

$$
E _ { q } = \mathbf { 1 } [ \mathcal { T } _ { 1 0 } ( q ) \cap \mathcal { A } _ { q } ( P ) \neq \emptyset ] ,\tag{2}
$$

and TARGETEXPOSURE@10 is its empirical mean. ANYPOI-SONEXPOSURE@10 counts any poisoned document, including one targeted to a different query; we also report target and any-poison counts in the top ten. Evidence availability is $H _ { q } = \mathbf { 1 } [ T _ { 1 0 } ( q ) \cap \mathcal { R } _ { q } \neq \emptyset ]$ ]. Safe retrieval reliability is

$$
S _ { q } = \mathbf { 1 } [ H _ { q } = 1 \land T _ { 1 0 } ( q ) \cap \mathcal { A } = \emptyset ] .\tag{3}
$$

Standard nDCG@10 and Recall@10 measure ranking utility. $S _ { q }$ is a strict component-level conjunction—evidence present and zero poison admitted—rather than a deployment safety probability. We therefore also decompose attacked queries into four observable states: evidence+poison, evidence+no-poison, no-evidence+poison, and no-evidence+no-poison. Together, these metrics distinguish poison admission, evidence loss, crosstarget interference, and ranking degradation.

## IV. EXPERIMENTAL DESIGN

We instantiate the system model using public retrieval benchmarks, matched content controls, and isolated-target and shared-corpus stress. Clean-corpus retrieval statistics stay fixed so the comparisons isolate the effect of added documents.

## A. Datasets and Retrieval

We use the BEIR test splits of SciFact (5,183 documents, 300 judged queries) and NFCorpus (3,633 documents, 323 judged queries) [9]. Retrieval uses Best Matching 25 (BM25) [10], 64- dimensional latent semantic analysis (LSA) [11], and reciprocalrank fusion (Hybrid) [12]. Hybrid uses RRF constant 60, fuses the top 100 BM25 and LSA candidates, and retains 50 candidates before the final top-10 stage. For external validation, we also evaluate the pretrained 33M-parameter

BAAI/bge-small-en-v1.5 bi-encoder [14] with normalized embeddings and cosine similarity, without task-specific training.

For isolated-target experiments, LSA is fitted once on the clean corpus and injected documents are transformed into that frozen space. BM25 likewise uses vocabulary, IDF, and average document-length statistics fitted to the clean corpus; poison documents are scored under those frozen statistics. Multi-target causal controls use the same clean-corpus statistics, so differences reflect added documents rather than retraining the representation.

## B. Content Controls and Stress Regimes

For each target query and $P \in \{ 1 , 3 , 5 \}$ , query-only poisons contain the target query without appended text; query+payload poisons append a BIPIA payload; fixed-neutral controls append deterministic benign filler of matched length. Payloads are sampled deterministically with replacement from four BIPIA text-attack categories (Base Encoding, Reverse Text, Emoji Substitution, and Language Translation) using seed 2026 and a query-specific hash. A stronger corpus-derived neutral condition uses five deterministic non-relevant clean-corpus snippets per target: each source has zero BM25 query score, is excluded from judged-relevant documents, and is truncated to the payload token length. This control estimates perturbation from appended corpus-like text without claiming perfect lexical, topical, or semantic matching to the payload.

In isolated-target stress, each query is evaluated against the clean corpus plus only its own $P$ poisons. At $P = 5$ , five targeted documents represent 0.10% of SciFact and 0.14% of NFCorpus. In multi-target stress, every evaluation query contributes P poisons to the same corpus; at $P = 5 ,$ , this yields 1,500 SciFact poisons (22.45%) and 1,615 NFCorpus poisons (30.77%). For the attacker-information axis at isolated $P = 5 ,$ payload-only provides no query terms; two- and four-term conditions use either deterministic random eligible query terms or the highest-IDF eligible terms; full-query stress includes the complete query. Random-term selection uses a fixed queryspecific SHA-256-derived seed; all term-budget conditions reuse the same per-query payload construction.

## C. Admission Controls, Statistics, and Latency

MMR reranks the same 50-candidate pool with $\lambda = 0 . 7 ,$ chooses the highest-relevance candidate first, and then trades normalized relevance against LSA-vector redundancy. Admission control uses IDF-weighted unique-query-term coverage $C ( q , d )$ and rejects candidates with $C ( q , d ) \ \geq \ \tau$ , where $\tau \ \in \ \{ 0 . 2 5 , 0 . 4 0 , 0 . 5 5 , 0 . 7 0 , 0 . 8 5 , 0 . 9 5 \}$ . Filtering scans the ranked 50-candidate list and continues downward until ten candidates pass or the pool is exhausted; shortages are reported explicitly. For clean, four-term, and full-query conditions, each threshold is evaluated on final top-10 nDCG, Recall, evidence availability, safe retrieval reliability, target exposure, and the fraction of queries with fewer than ten admitted candidates. We report the full frontier rather than selecting a test-set-optimal threshold.

TABLE I  
PAIRED ∆ NDCG@10 AT ISOLATED $P = 5$ WITH 95% BOOTSTRAP CIS.
<table><tr><td>Data</td><td>Retr.</td><td>∆ vs. query-only</td><td>∆ vs. corpus-neutral</td></tr><tr><td>SciFact</td><td>BM25</td><td> $+ . 0 0 0 0 [ + . 0 0 0 0 , + . 0 0 0 0 ]$ </td><td> $+ . 0 0 0 0 [ + . 0 0 0 0 , + . 0 0 0 0 ]$ </td></tr><tr><td>SciFact</td><td>LSA</td><td> $+ . 0 0 1 4 [ + . 0 0 0 0 , + . 0 0 3 8 ]$ </td><td> $- . 0 1 1 7 [ - . 0 1 5 9 , - . 0 0 8 0 ]$ </td></tr><tr><td>SciFact</td><td>Hybrid</td><td>+.0003[−.0002,+.0010]</td><td>-.0108[-.0174, -.0051]</td></tr><tr><td>NFCorpus</td><td>BM25</td><td> $+ . 0 0 3 9 [ + . 0 0 1 5 , + . 0 0 6 9 ]$ </td><td>-.0004[−.0013, +.0000]</td></tr><tr><td>NFCorpus</td><td>LSA</td><td> $+ . 0 4 5 0 [ + . 0 3 2 9 , + . 0 5 8 6 ]$ </td><td> $- . 0 3 7 0 [ - . 0 4 5 8 , - . 0 2 9 2 ]$ </td></tr><tr><td>NFCorpus</td><td>Hybrid</td><td> $+ . 0 3 8 2 \dot { [ + . 0 2 7 8 , + . 0 4 9 4 ] }$ </td><td> $- . 0 3 6 7 \dot { [ - . 0 4 4 5 , - . 0 2 9 2 ] }$ </td></tr></table>

Metrics are computed per query. Nonparametric bootstrap 95% confidence intervals use queries as the resampling unit; paired intervals compare query+payload with query-only and with corpus-derived neutral content. Latency uses 500 repeated $P = 5$ measurements per retriever/defense pair on an AMD EPYC 9V74 allocation with five visible CPU cores under Python 3.13.5, reporting absolute median and P95 milliseconds.

## V. RESULTS

We first examine content controls for RQ1, then attacker information and contamination scope for RQ2, and finally admission trade-offs for RQ3. Each analysis measures the retrieved candidates rather than a generated answer.

## A. RQ1: What makes a poisoned document retrievable?

Figure 2 compares isolated-target $P = 5$ query-only, fixedneutral, corpus-derived-neutral, and payload conditions. On SciFact, query-only and query+payload nDCG are identical for BM25 and differ by only +0.0014 for LSA and +0.0003 for Hybrid. On NFCorpus, payload-minus-query-only differences are +0.0039 for BM25, +0.0450 for LSA, and +0.0382 for Hybrid, indicating stronger sensitivity of semantic representations to appended text.

The corpus-derived neutral control separates query anchoring from appended-content identity more carefully. For BM25, payload-minus-neutral nDCG is 0 on SciFact and −0.0004 on NFCorpus (95% CI $[ - 0 . 0 0 1 3 , 0 . 0 0 0 0 ] \rangle$ ). For LSA/Hybrid, the corresponding differences are $- 0 . 0 1 1 7 / - 0 . 0 1 0 8$ on SciFact and $- 0 . 0 3 7 0 / - 0 . 0 3 6 7$ on NFCorpus, with paired intervals excluding zero. NFCorpus target exposure is also 13.3 percentage points higher for LSA and 3.4 points higher for Hybrid under payload than corpus-neutral text. On BM25, the query anchor accounts for nearly all of the observed degradation. LSA and Hybrid are more sensitive to the appended text, especially on NFCorpus. Because the neutral additions also change these rankings, the experiment supports an appended-content effect rather than a payload-specific retrieval mechanism.

Table I gives paired bootstrap intervals for the same comparisons. SciFact Hybrid changes by only +0.0003 relative to query-only (95% CI $[ - 0 . 0 0 0 2 , 0 . 0 0 1 0 ] )$ , whereas NFCorpus LSA and Hybrid shift by +0.0450 and +0.0382. Against corpus-derived neutral text, the LSA/Hybrid differences are negative on both datasets and their intervals exclude zero, showing that the identity of the appended text can change semantic rankings even with the same query anchor.

![](images/681298534f55bf8cf20e79a7a5173a60e284507a6f3dc144c420406945b71bee.jpg)  
Fig. 2. Isolated-target $P = 5$ content controls for nDCG@10 and target exposure. Table I reports the paired confidence intervals.

TABLE II  
CORE RETRIEVAL AND RELIABILITY METRICS UNDER CLEAN, ISOLATED $P = 5 ,$ AND MULTI-TARGET P = 5 QUERY+PAYLOAD STRESS.
<table><tr><td>Data</td><td>Retr.</td><td>Condition</td><td>nDCG</td><td>Recall</td><td>TargetExp</td><td> $\mathbf { A n y E x p }$ </td><td>Evid.</td><td>SafeR</td><td>#P</td><td>Poison%</td></tr><tr><td rowspan="9">SciFact</td><td>BM25</td><td>Clean</td><td>.662</td><td>.781</td><td>.000</td><td>.000</td><td>.800</td><td>.800</td><td>.00</td><td>.00</td></tr><tr><td>BM25</td><td>Isolated  $P = 5$ </td><td>.255</td><td>.718</td><td>1.000</td><td>1.000</td><td>.740</td><td>.000</td><td>5.00</td><td>.10</td></tr><tr><td>BM25</td><td>Multi  $P = 5$ </td><td>.188</td><td>.529</td><td>1.000</td><td>1.000</td><td>.550</td><td>.000</td><td>6.22</td><td>22.45</td></tr><tr><td>LSA</td><td>Clean</td><td>.295</td><td>.442</td><td>.000</td><td>.000</td><td>.453</td><td>.453</td><td>.00</td><td>.00</td></tr><tr><td>LSA</td><td>Isolated  $P = 5$ </td><td>.119</td><td>.347</td><td>1.000</td><td>1.000</td><td>.367</td><td>.000</td><td>4.97</td><td>.10</td></tr><tr><td>LSA</td><td>Multi  $P = 5$ </td><td>.085</td><td>.244</td><td>1.000</td><td>1.000</td><td>.257</td><td>.000</td><td>7.16</td><td>22.45</td></tr><tr><td>Hybrid</td><td>Clean</td><td>.465</td><td>.649</td><td>.000</td><td>.000</td><td>.663</td><td>.663</td><td>.00</td><td>.00</td></tr><tr><td>Hybrid</td><td>Isolated  $P = 5$ </td><td>.183</td><td>.530</td><td>1.000</td><td>1.000</td><td>.547</td><td>.000</td><td>5.00</td><td>.10</td></tr><tr><td>Hybrid</td><td>Multi  $P = 5$ </td><td>.138</td><td>.398</td><td>1.000</td><td>1.000</td><td>.410</td><td>.000</td><td>6.34</td><td>22.45</td></tr><tr><td rowspan="9">NFCorpus</td><td>BM25</td><td>Clean</td><td>.308</td><td>.146</td><td>.000</td><td>.000</td><td>.678</td><td>.678</td><td>.00</td><td>.00</td></tr><tr><td>BM25</td><td>Isolated  $P = 5$ </td><td>.181</td><td>.118</td><td>.876</td><td>.876</td><td>.619</td><td>.037</td><td>4.28</td><td>.14</td></tr><tr><td>BM25</td><td>Multi  $P = 5$ </td><td>.153</td><td>.108</td><td>.972</td><td>.972</td><td>.548</td><td>.022</td><td>5.49</td><td>30.77</td></tr><tr><td>LSA</td><td>Clean</td><td>.221</td><td>.109</td><td>.000</td><td>.000</td><td>.563</td><td>.563</td><td>.00</td><td>.00</td></tr><tr><td>LSA</td><td>Isolated  $P = 5$ </td><td>.131</td><td>.090</td><td>.824</td><td>.824</td><td>.498</td><td>.071</td><td>3.64</td><td>.14</td></tr><tr><td>LSA</td><td>Multi  $P = 5$ </td><td>.091</td><td>.066</td><td>.805</td><td>.966</td><td>.328</td><td>.022</td><td>7.23</td><td>30.77</td></tr><tr><td>Hybrid</td><td>Clean</td><td>.267</td><td>.129</td><td>.000</td><td>.000</td><td>.622</td><td>.622</td><td>.00</td><td>.00</td></tr><tr><td>Hybrid</td><td>Isolated  $P = 5$ </td><td>.160</td><td>.099</td><td>.910</td><td>.910</td><td>.551</td><td>.006</td><td>4.22</td><td>.14</td></tr><tr><td>Hybrid</td><td>Multi  $P = 5$ </td><td>.138</td><td>.091</td><td>1.000</td><td>1.000</td><td>.520</td><td>.000</td><td>6.22</td><td>30.77</td></tr></table>

B. RQ2: How does failure scale with attacker information and contamination scope?

Table II shows that contamination scope changes the failure mode, not merely its magnitude. Under isolated $P = 5$ , five query-targeted documents constitute only 0.10–0.14% of the corpus, yet Hybrid nDCG falls from 0.465 to 0.183 on SciFact and from 0.267 to 0.160 on NFCorpus; SAFERETRIEVAL-RELIABILITY@10 falls from 0.663 to 0 and from 0.622 to 0.006, respectively. The small global fractions need to be read together with the strong per-query targeting: this condition measures concentrated query-aware contamination, not random corruption at the same corpus fraction.

Table III closes the $P \in \{ 1 , 3 , 5 \}$ budget sweep for the Hybrid isolated setting. On SciFact, target exposure is already 1.0 at $P = 1$ , while additional poisons mainly displace evidence and reduce nDCG. On NFCorpus, exposure changes less than ranking quality as $P$ grows: nDCG falls from 0.230 at $P = 1$ to 0.160 at $P = 5$ , while exposure moves from 0.889 to 0.910.

TABLE III  
HYBRID ISOLATED FULL-QUERY STRESS AS THE PER-TARGET BUDGET P INCREASES.
<table><tr><td>Data</td><td> $P$ </td><td>nDCG</td><td>TargetExp</td><td>Evid.</td><td>SafeR</td></tr><tr><td rowspan="3">SciFact</td><td>1</td><td>.328</td><td>1.000</td><td>.647</td><td>.000</td></tr><tr><td>3</td><td>.232</td><td>1.000</td><td>.593</td><td>.000</td></tr><tr><td>5</td><td>.183</td><td>1.000</td><td>.547</td><td>.000</td></tr><tr><td rowspan="3">NFCorpus</td><td>1</td><td>.230</td><td>.889</td><td>.607</td><td>.025</td></tr><tr><td>3</td><td>.186</td><td>.907</td><td>.573</td><td>.009</td></tr><tr><td>5</td><td>.160</td><td>.910</td><td>.551</td><td>.006</td></tr></table>

Table IV clarifies the strict SafeR definition. SciFact SafeR is zero because every attacked query admits poison, even though 54.7% still retain judged evidence. NFCorpus likewise retains evidence alongside poison for 54.5% of queries; only 0.6% satisfy the stricter evidence-plus-zero-poison condition. SafeR therefore summarizes a component-level conjunction, not “probability the RAG system answers incorrectly.”

Under multi-target $P = 5 ,$ all targets contribute poisons simultaneously, creating 22.45% and 30.77% corpus fractions. Cross-target interference becomes measurable. For NFCorpus LSA, target exposure is 0.805 while any-poison exposure is 0.966 and the mean top-10 poison count is 7.23. Target exposure alone misses a substantial part of the saturation regime. Reporting per-target P, contaminated-target count, corpus fraction, and any-poison exposure distinguishes concentrated targeting from broad corpus stress.

TABLE IV  
HYBRID ISOLATED $P = 5$ FOUR-STATE DECOMPOSITION (FRACTIONS OF QUERIES).
<table><tr><td>Data</td><td>Evid.+Poison</td><td>Evid.+NoPoison</td><td>NoEvid.+Poison</td><td>NoEvid.+NoPoison</td></tr><tr><td>SciFact</td><td>.547</td><td>.000</td><td>.453</td><td>.000</td></tr><tr><td>NFCorpus</td><td>.545</td><td>.006</td><td>.365</td><td>.084</td></tr></table>

![](images/852296c53af001c4e0a946ce7a95be44aec3b6616f3d9e1a27272f56086d8fb1.jpg)  
Fig. 3. Hybrid target exposure versus attacker query information at isolated $P = 5$

Attacker query information forms a second stress axis, but a fixed term count is not a fixed information fraction. Figure 3 reports the unconditioned Hybrid exposure curves; Table V supplies the missing query-length context. NFCorpus has only 2.48 eligible terms on average (median 2): 57.6% of its queries are fully covered by the two-term budget and 90.7% by the four term budget. Its high 2/4-term exposure therefore should not be read as evidence that a small fraction of query information is always sufficient.

The long-query subset gives a cleaner partial-information test. On 162 SciFact queries with at least eight eligible terms, two random terms reveal 19.9% of eligible terms on average yet still produce 39.5% target exposure; four terms reveal 39.7% and produce 89.5% exposure. High-IDF anchors remain an informed/oracle-like condition, not a necessary attacker capability. These results support sensitivity to partial query information on sufficiently long queries while exposing the short-query confound on NFCorpus.

## C. Pretrained dense-retriever validation

We repeat the isolated P = 5 clean/query-only/query+payload comparison with BAAI/bge-small-en-v1.5. Query-only contamination reaches 100% target exposure on both datasets (Table VI). Adding the payload leaves SciFact essentially unchanged and is less disruptive on NFCorpus, where exposure falls slightly to 0.988. This compact replication extends high query-targeted admission to a pretrained dense retriever while again separating retrieval relevance from malicious payload semantics.

TABLE V  
RANDOM-TERM INFORMATION COVERAGE AND HYBRID EXPOSURE AT ISOLATED $P = 5 . \ { } ^ {  } { \mathrm { L o N G } } ^ { \prime \prime }$ MEANS AT LEAST EIGHT ELIGIBLE TERMS.
<table><tr><td>Data/subset</td><td>n</td><td>Mean elig.</td><td>Cov2</td><td>Exp2</td><td> $\mathrm { C o v _ { 4 } \ / \ E x p _ { 4 } }$ </td></tr><tr><td>SciFact all</td><td>300</td><td>8.33</td><td>.278</td><td>.507</td><td>.553 /  .940</td></tr><tr><td>SciFact long</td><td>162</td><td>10.63</td><td>.199</td><td>.395</td><td>.397  /  .895</td></tr><tr><td>NFCorpus all</td><td>323</td><td>2.48</td><td>.805</td><td>.848</td><td>.974 /  .901</td></tr></table>

TABLE VI

BGE-SMALL ISOLATED $P = 5$ VALIDATION. EACH ATTACKED CELL IS NDCG@10 / TARGET EXPOSURE@10.
<table><tr><td>Data</td><td>Clean nDCG</td><td>Query only</td><td>Query+payload</td></tr><tr><td>SciFact</td><td>.713</td><td> $. 2 6 9 \mathrm { ~ / ~ } 1 . 0 0 0$ </td><td>.271 / 1.000</td></tr><tr><td>NFCorpus</td><td>.345</td><td>.137 / 1.000</td><td>.205 / .988</td></tr></table>

![](images/1ff78bb738316e0756162e9598b7ee169646ef24d96528a5dc2abed89c8ab8fa.jpg)  
Fig. 4. Hybrid four-term overlap-admission frontier for exposure and safe retrieval reliability.

D. RQ3: What does admission control trade for lower exposure?

Figure 4 summarizes the overlap threshold security–utility frontier in final retrieval quality. For full-query Hybrid stress, $\tau = 0 . 9 5$ removes target exposure on both datasets. This is an attack-matched rule: a document that repeats nearly all query terms is expected to have coverage near one, so full-query rejection at a 0.95 threshold is not itself evidence of a broadly effective detector. The informative result is the collateral cost. On SciFact, attack and clean nDCG under that threshold are both 0.440, close to the unfiltered clean 0.465. On NFCorpus, however, attack nDCG is 0.138 and clean nDCG is 0.134, far below the unfiltered clean 0.267; 3.4–4.0% of queries also retain fewer than ten candidates. The four-term condition is more difficult to filter selectively: at $\tau = 0 . 4 0$ , target exposure falls to 5.3% on SciFact and 0.3% on NFCorpus, but clean nDCG collapses to 0.082 and 0.065. ${ \mathrm { A t ~ } } \tau \ = \ 0 . 9 5$ , clean nDCG recovers to 0.440/0.134 while four-term exposure rises to 78.0%/22.3%. Across the sweep, stronger lexical rejection reduces exposure but also removes legitimate evidence, and the trade-off differs sharply between the two datasets.

MMR closes a different question. Under isolated full-query $P = 5$ Hybrid stress, target exposure remains 1.000 on SciFact and 0.910 on NFCorpus while nDCG changes from 0.183 to 0.234 and from 0.160 to 0.164; SAFERETRIEVALRELIABIL-ITY@10 remains 0 and 0.006. MMR therefore changes ranking utility without acting as an admission barrier here, while adding about 1.5 ms to median Hybrid latency on both datasets.

## VI. DISCUSSION

## A. Document Content

Under full-query stress, BM25 is dominated by the query anchor, whereas LSA and Hybrid also respond to the appended text. The neutral controls show that this semantic-ranking shift is not specific to malicious instructions. The BGE-small replication extends the high-exposure result to a pretrained dense retriever, but its query+payload condition is no more damaging than query-only on SciFact and is less damaging on NFCorpus. This reinforces the distinction between queryrelevant retrieval manipulation and downstream maliciousinstruction semantics.

## B. Stress Intensity

Isolated $P = 5$ combines a small global poison fraction with five documents written specifically for each evaluated query. Multi-target $P = 5$ instead creates a broad saturation condition in which poison written for other targets also enters top-k. Query-information budgets require the same care: on short NFCorpus queries, two or four eligible terms often reveal nearly the whole query, whereas the long SciFact subset shows substantial exposure at 19.9–39.7% mean coverage. A single poison rate or term count therefore hides materially different retrieval conditions.

## C. Admission Policy

Lower exposure does not necessarily improve final retrieval if the same rule removes judged evidence or leaves too few candidates. The overlap sweep has no stable cross-dataset operating point: thresholds that strongly suppress poison impose large clean-task costs on NFCorpus. Attack-side exposure and clean-task utility need to be evaluated together.

## VII. LIMITATIONS

The study ends at retrieval, so exposure is not equivalent to downstream attack success. Query-aware stressors are transparent rather than stealthy or optimized, and isolated low-fraction experiments still grant strong per-target targeting. Corpus-derived neutral snippets improve content matching but do not perfectly match payload term frequency, topic, or semantics; the analysis attributes effects to query anchoring versus appended-content identity rather than to malicious meaning itself. Fixed two/four-term budgets are especially difficult to interpret on short queries, so the long-query analysis is reported separately. High-IDF conditions additionally assume access to clean-corpus statistics. Pretrained validation is limited to one compact BGE model and the three core clean/queryonly/query+payload conditions; larger embedding models, alternative dense retrievers, and production vector-search stacks remain outside scope. BEIR qrels are incomplete, so evidence availability and SAFERETRIEVALRELIABILITY@10 inherit judgment incompleteness. Threshold curves are measurement frontiers rather than deployment recommendations.

## VIII. CONCLUSION

The experiments provide direct answers to the three research questions. RQ1: Query anchoring explains most BM25 degradation; appended content can also shift LSA and Hybrid rankings without establishing a malicious-instruction-specific retrieval effect. The pretrained BGE-small check reaches 100% target exposure for query-only poisoning on both datasets, with no additional payload-specific penalty in the tested conditions. RQ2: Two randomly selected terms yield 39.5% Hybrid exposure on 162 long SciFact queries, whereas fixed term budgets can approximate full-query knowledge on short NFCorpus queries; broad contamination additionally causes cross-target interference. RQ3: Overlap admission suppresses full-query exposure but sacrifices clean ranking utility at datasetdependent rates, and maximal marginal relevance (MMR) leaves exposure unchanged under the tested configuration. These are retrieval-stage findings, not downstream attacksuccess estimates.

## REFERENCES

[1] P. Lewis et al., “Retrieval-Augmented Generation for Knowledge Intensive NLP Tasks,” in NeurIPS, vol. 33, 2020.

[2] W. Zou, R. Geng, B. Wang, and J. Jia, “PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models,” in USENIX Security, 2025, pp. 3827–3844.

[3] A. Shafran, R. Schuster, and V. Shmatikov, “Machine Against the RAG: Jamming Retrieval-Augmented Generation with Blocker Documents,” in USENIX Security, 2025, pp. 3787–3806.

[4] H. Chang, E. Bao, X. Luo, and T. Yu, “Overcoming the Retrieval Barrier: Indirect Prompt Injection in the Wild for LLM Systems,” in USENIX Security, 2026, pp. 1567–1586.

[5] J. Yi et al., “Benchmarking and Defending against Indirect Prompt Injection Attacks on Large Language Models,” in Proc. 31st ACM SIGKDD Conf., 2025, pp. 1809–1820, doi: 10.1145/3690624.3709179.

[6] Q. Zhan, Z. Liang, Z. Ying, and D. Kang, “InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents,” in Findings of ACL, 2024, pp. 10471–10506.

[7] Z. Chen et al., “AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases,” in NeurIPS, vol. 37, 2024.

[8] Z. Shen, B. Imana, T. Wu, C. Xiang, P. Mittal, and A. Korolova, “ReliabilityRAG: Effective and Provably Robust Defense for RAG-based Web-Search,” in NeurIPS, vol. 38, 2025.

[9] N. Thakur et al., “BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models,” in NeurIPS Datasets and Benchmarks, 2021.

[10] S. Robertson and H. Zaragoza, “The Probabilistic Relevance Framework: BM25 and Beyond,” Found. Trends Inf. Retr., vol. 3, no. 4, pp. 333–389, 2009.

[11] S. Deerwester et al., “Indexing by Latent Semantic Analysis,” J. Amer. Soc. Inf. Sci., vol. 41, no. 6, pp. 391–407, 1990.

[12] G. V. Cormack, C. L. A. Clarke, and S. Buettcher, “Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods,” in SIGIR, 2009, pp. 758–759.

[13] J. Carbonell and J. Goldstein, “The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries,” in SIGIR, 1998, pp. 335–336.

[14] S. Xiao, Z. Liu, P. Zhang, and N. Muennighoff, “C-Pack: Packaged Resources To Advance General Chinese Embedding,” arXiv:2309.07597, 2023.