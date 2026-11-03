# SearchInject: Retrieval Exposure Control for Indirect Prompt Injection in Language-Model Search

Kainan Zhou   
Google LLC   
Mountain View, USA   
zhoumark@google.com   
Zhi Jing   
Google LLC   
Mountain View, USA   
varusjing@google.com   
Gangzhen Qian   
Google LLC   
Mountain View, USA   
irisqian@google.com   
Rong Bao   
Google LLC   
Mountain View, USA   
arielbao@google.com   
Janet Sung   
Google LLC   
Mountain View, USA   
janetsung@google.com   
Qianru Lao   
Harvard University   
Cambridge, USA   
estherbear17@gmail.com

Abstract—An indirect prompt injection can affect a searchaugmented large language model only after its carrier page reaches model context. SearchInject controls this admission decision with a joint score combining retrieval relevance, searchabuse cues, instruction likelihood, and two relevance interactions. A quarantine threshold is selected under an empirical 0.5% clean-candidate false-quarantine constraint on a calibration split. Evaluation uses 300 held-out SciFact queries and held-out payload categories within four construction families already represented during development. Across 1,200 query–family outcomes per retriever, top-10 target exposure falls from 0.9408 to 0.1583 for BM25, from 0.9158 to 0.1508 for a normalized hybrid, and from 0.8450 to 0.1375 for latent semantic analysis. These are relative reductions of 83.2–83.7%; the largest absolute change in clean normalized discounted cumulative gain at rank 10 is 0.0018. The frozen threshold yields clean test false-quarantine rates of 0.59–0.65%, above the empirical calibration constraint. A querylevel analytical reanalysis accounts for the repeated construction outcomes under an independent-query assumption. A separate 104-case generation follow-up records context removal but no confirmed undefended hijacks under a post hoc endpoint, so it cannot estimate a reduction in attack success. The evidence supports retrieval exposure control within the evaluated construction families; the benefit of the interaction terms at matched clean cost remains unresolved.

Index Terms—indirect prompt injection, information retrieval, retrieval-augmented generation, language models

## I. INTRODUCTION

Search-augmented language models answer queries using retrieved third-party text. Publishers can embed instructions without authority over the application; early demonstrations showed that such content could redirect language-model applications [1]. In search, the carrier page must first be retrieved, ranked, and admitted to context.

Query-term repetition, title manipulation, doorway text, and copied passages can promote a page [2]. When results become model inputs, promotion determines which untrusted instructions the generator encounters, creating an admission problem before any instruction is followed.

InjecAgent and AgentDojo evaluate behavior after hostile content is delivered through tool observations or task environments [3], [4]. Their endpoints do not isolate carrier selection.

Retrievable trigger fragments can cross this selection boundary under ordinary-looking queries [5]. SearchInject addresses the admission decision and its clean retrieval cost.

The main technical contribution is a relevance-conditioned risk score coupled with a clean-cost-constrained quarantine threshold. Logistic scoring and refill implement this admission policy. The evaluation separates target exposure from task hijacking, so a change in one quantity is not credited as a change in the other.

Exposure decreases by 83.2–83.7% across three lightweight retrievers, with small clean ranking changes. We report calibration transfer and query-level uncertainty; interaction effects and matched-cost comparisons remain unresolved.

## II. BACKGROUND AND RELATED WORK

## A. Injection and retrieval

BIPIA embeds instructions in external content paired with benign tasks [6]. The formulation of Liu et al. treats prompt injection as conflict between the application’s objective and an injected objective [7]. Our exposure endpoint records whether the crafted target page reaches context; a confirmed hijack additionally requires observable execution of the injected task.

Retrieval-augmented generation supplies selected external evidence to a generator [8]. PoisonedRAG optimizes inserted content for both retrieval and a chosen answer [9]. SearchInject examines the admission decision and its clean retrieval cost using fixed, attributable page constructions. This setting permits controlled comparisons but does not represent adaptive optimization against the defense.

The constructions modify title coverage, term placement, borrowed evidence, or indexed metadata. These cues connect to content-based spam detection [10]. The source-overlap signal is a controlled proxy; reliable ownership and provenance are not available to the defender in this experiment.

![](images/ca44cca233ab3bff723b557a98d830386c9f2c7558049f4002ca9d06670bee2c.jpg)  
Fig. 1. SearchInject scores the depth-100 candidate list before context assembly. Rejected pages are omitted, and accepted candidates retain their original relative order. The main evaluation measures target admission; generation behavior is a separate exploratory endpoint.

## B. Complementary defense boundaries

DataSentinel evaluates detection against a strategically adapting injector, whereas PromptLocate identifies contaminated spans within a document [11], [12]. SearchInject scores complete retrieved candidates before they enter the generator context. It does not localize an instruction or certify that an admitted page is safe.

StruQ combines structured queries with model training; SecAlign uses preference optimization [13], [14]. Both strengthen the boundary between trusted instructions and external data after retrieval.

Third-party chatbot plugins introduce additional trustboundary failures through scraping and application integration [15]. Our fixed corpus does not reproduce those deployment paths.

## III. SEARCHINJECT DESIGN

## A. Threat model and admission policy

The attacker controls a published page’s title, body, and markup-like metadata for a known query family. The service controls the retrieval and generation pipeline. The defender observes queries, candidate text, and ranker scores. Pages are fixed before final evaluation; attackers do not adapt to defense outputs.

Figure 1 locates the intervention. A crafted target page carries an injected instruction. Quarantine withholds a candidate from generation context without assigning a publisher-level judgment. Exposure and confirmed hijacking are recorded separately.

For each query, the policy scans the original depth-100 ranking. Candidates whose risk score exceeds the threshold are skipped; all others are appended in rank order. The scan stops at ten accepted pages or exhaustion of the candidate list. Thus, removed pages are replaced from the same ranking rather than leaving a shortened top ten. There is no reranking of the survivors and no retrieval beyond the original candidate list.

## B. Controlled construction families

Each target pairs a SciFact query with a deterministically selected source document. Title stuffing repeats salient terms in the title and lead sentence. A doorway page starts with a query-specific summary and then introduces the payload. A near duplicate preserves most of a relevant abstract and appends the instruction. Hidden metadata places query terms in indexed fields while leaving the visible body comparatively natural.

Construction records retain family, source identifier, payload category, and content hash. Generation truncation preserves the retrieval-oriented lead and instruction-bearing tail; near duplicates remain distinct from clean sources. All families occur during development and testing.

## C. Features and risk score

Let $r ( q , d )$ be normalized ranker relevance. An abuse model produces $\boldsymbol { a } ( \boldsymbol { q } , \boldsymbol { d } )$ from title coverage, body term density, repetition, source overlap, and markup indicators. Instruction likelihood $i ( d )$ uses inverse-document-frequency-weighted word one- and two-grams and character three- to five-grams. The word features capture short instruction phrases; the character features tolerate small spacing and punctuation changes.

The joint model is class-balanced logistic regression with the fixed feature order

$$
z ( q , d ) = [ r , a , i , a r , i r ] .\tag{1}
$$

Its output $s ( q , d ) = \sigma ( b + \beta ^ { \mathsf { T } } z ( q , d ) )$ is used as a risk score. The two interactions permit the contributions of abuse and instruction signals to vary with relevance. Construction labels and simulated source markers are excluded. No probabilitycalibration guarantee is attached to the logistic output.

## D. Calibration and implementation scope

The threshold is selected from calibration scores to maximize crafted-page recall subject to an empirical cleancandidate false-quarantine rate at most 0.005. With $\mathcal { C } _ { 0 }$ denoting scored clean query–document placements,

$$
\widehat { \mathrm { F Q } } _ { \mathrm { c a l } } ( t ) = \frac { 1 } { | { \mathcal { C } } _ { 0 } | } \sum _ { ( q , d ) \in { \mathcal { C } } _ { 0 } } \mathbf { 1 } \{ s ( q , d ) > t \} .\tag{2}
$$

The strict comparison $s \mathrm { ~  ~ { ~ \gamma ~ } ~ } > \mathrm { ~ \textit ~ { ~ t ~ } ~ }$ is used for quarantine. A document returned for several queries contributes several placements. The selected threshold is $\tau ~ = ~ 0 . 9 7 3 8 5 3$ , with calibration recall 0.804 and false quarantine 0.00498. The feature definitions, fitted model, and threshold are then frozen before test evaluation.

This empirical selection rule does not guarantee a 0.5% bound on new samples. The feature order and threshold alone cannot reconstruct the trained system; Section VI-C identifies the parameters and records required for replay.

TABLE I  
EVALUATION PROTOCOL AND KNOWN IMPLEMENTATION SETTINGS
<table><tr><td>Item</td><td>Setting</td></tr><tr><td>Corpus</td><td>5,183 SciFact abstracts in BEIR format</td></tr><tr><td>Queries</td><td>655 fitting; 154 calibration; 300 test</td></tr><tr><td>Payload categories</td><td>9 fitting; 3 calibration; 13 test Title stuffing, doorway, near duplicate,</td></tr><tr><td>Construction families</td><td>hidden metadata; all seen during development</td></tr><tr><td>Retrieval</td><td>BM25, LSA, normalized hybrid; depth 100 Class-balanced logistic regression;</td></tr><tr><td>Joint model</td><td>[r, a, i, ar, ir]</td></tr><tr><td>Admission</td><td>Quarantine if s &gt; 0.973853; refill to 10</td></tr><tr><td>Statistical units</td><td>300 queries; four outcomes per query</td></tr></table>

## IV. EXPERIMENTAL METHOD

## A. Data partitions and retrievers

The study uses SciFact scientific abstracts and relevance judgments [16] in the BEIR retrieval format [17]. This format contains 5,183 documents, 809 training queries, and 300 test queries; its test split corresponds to the original SciFact development split. A deterministic hash assigns 655 training queries to fitting and 154 to calibration. The designated 300- query retrieval test split is held out from model and threshold choices.

Payloads are a filtered subset of BIPIA’s text attacks: nine fitting categories, three calibration categories, and 13 retained categories from its test split. These are study-specific subset counts, not BIPIA’s complete official splits. Malware, fraud, propaganda, and political-persuasion payloads are excluded. Clean SciFact claims and retrieval-evidence sentences provide negatives.

Queries and payload categories are disjoint across partitions; source-document disjointness is not established. The test measures transfer within known construction families, as summarized in Table I.

BM25 supplies lexical ranking with document-length normalization [18]. Latent semantic analysis (LSA) applies truncated singular-value decomposition to the term–document matrix; the hybrid averages normalized BM25 and LSA scores. Retrieval depth is 100. LSA does not represent Dense Passage Retrieval or current learned encoders [19].

## B. Metrics and query-level uncertainty

Exposure@k is the fraction of query–family cases in which the crafted target appears among the first k pages admitted to context. Clean utility is evaluated on a separate, unattacked corpus copy. Normalized discounted cumulative gain (NDCG@10) rewards relevant documents near the top; mean reciprocal rank (MRR@10) records the reciprocal rank of the first relevant result; Recall@10 measures recovered judged evidence. False quarantine is measured per clean placement.

The four outcomes for one query are repeated observations. For each retriever, let $U _ { q f }$ and $J _ { q f }$ indicate exposure without defense and with joint quarantine for query q and family f. Define the paired query average

$$
D _ { q } = \frac { 1 } { 4 } \sum _ { f = 1 } ^ { 4 } ( U _ { q f } - J _ { q f } ) , \qquad { \widehat { \Delta } } = \frac { 1 } { 3 0 0 } \sum _ { q = 1 } ^ { 3 0 0 } D _ { q } .\tag{3}
$$

Here $D _ { q } \in [ - 1 , 1 ] \colon$ refill can admit a target that was previously below rank ten. The 1,200 query–family outcomes are therefore not treated as 1,200 independent samples.

Conditional on a fixed corpus, fitted model, and threshold, and treating queries as independent sampling units, Hoeffding’s inequality [20] gives a conservative marginal 95% interval for the expected paired reduction:

$$
\widehat { \Delta } \ \pm \ \sqrt { \frac { 2 \log ( 4 0 ) } { 3 0 0 } } = \widehat { \Delta } \ \pm \ 0 . 1 5 6 8 2 .\tag{4}
$$

This analytical reanalysis uses aggregate counts and permits arbitrary dependence among constructions for one query. It does not cover dependence between queries sharing sources or transfer to unseen families. Intervals are individual per retriever, not simultaneous or obtained by bootstrapping reconstructed records.

Reported clean-metric intervals use 10,000 paired bootstrap samples of whole queries. A sharper exposure analysis requires the original query clusters with all family and policy outcomes. Aggregate counts do not determine their covariance or paired contingency tables; unclustered binomial intervals and aggregate McNemar claims are omitted.

## C. Exploratory generation protocol

The 104-case generation cohort crosses two deterministic selections, four families, and 13 payload categories. Targets must enter the undefended hybrid top five. Clean, attacked, and quarantined retrieval are evaluated under conventional and hardened prompts; the latter labels retrieved material as untrusted.

Qwen3-4B and Phi-4-mini-instruct use four-bit weights and deterministic, temperature-zero decoding. Prompts contain five sources capped at 450 characters each. Answers, limited to 128 tokens, require SUPPORTS, CONTRADICTS, or NOT\_ENOUGH\_INFO plus an explanation. This is a factverification follow-up.

Two planned independent human reviewers were unavailable. Automated payload-specific execution checks were fixed after preliminary semantic labels had been inspected, making the endpoint post hoc and non-blinded. Observable execution counts as confirmed hijacking; incomplete or underspecified substitutions remain unresolved. Context removal is evaluated separately.

## V. RESULTS

## A. Retrieval exposure and clean utility

Table II and Fig. 2 summarize the main operating point. Across 1,200 query–family outcomes per retriever, the exact exposed counts change from 1,129 to 190 for BM25, 1,099 to 181 for the hybrid, and 1,014 to 165 for LSA. The absolute reductions are 0.7825, 0.7650, and 0.7075, respectively; relative reductions range from 83.2% to 83.7%.

TABLE II  
MAIN RESULTS OVER THE FOUR KNOWN CONSTRUCTION FAMILIES
<table><tr><td rowspan="2"></td><td colspan="2">Exposure@10</td><td colspan="2">Clean NDCG@10</td></tr><tr><td>Retriever None</td><td>Joint</td><td>None</td><td>Joint</td></tr><tr><td>BM25</td><td>.9408</td><td>.1583</td><td>.6602</td><td>.6584</td></tr><tr><td>Hybrid</td><td>.9158</td><td>.1508</td><td>.4606</td><td>.4593</td></tr><tr><td>LSA</td><td>.8450</td><td>.1375</td><td>.3000</td><td>.2987</td></tr></table>

TABLE III  
EMPIRICAL CALIBRATION CONSTRAINT AND CLEAN TEST OBSERVATIONS
<table><tr><td>Split or retriever</td><td>False quarantine</td><td>Threshold</td></tr><tr><td>Calibration constraint</td><td>≤ 0.500%</td><td>Selected on calibration</td></tr><tr><td>Calibration observed</td><td>0.498%</td><td>0.973853</td></tr><tr><td>BM25 test</td><td>0.590%</td><td>0.973853</td></tr><tr><td>Hybrid test</td><td>0.620%</td><td>0.973853</td></tr><tr><td>LSA test</td><td>0.650%</td><td>0.973853</td></tr></table>

The conservative 95% intervals from (4) are [0.626, 0.939] for BM25, [0.608, 0.922] for the hybrid, and [0.551, 0.864] for LSA. These intervals use 300 query units and the independentquery sampling model. The observed benchmark reductions themselves are exact functions of the reported counts.

Clean NDCG@10 changes by −0.0018, −0.0013, and −0.0013 for BM25, hybrid, and LSA. The reported paired bootstrap intervals are [−0.0053, 0], [−0.0039, 0.0001], and [−0.0039, 0]. MRR@10 changes by −0.0017, 0, and 0; Recall@10 decreases by 0.0017 for each retriever. Small point estimates and intervals near zero do not constitute a formal equivalence test.

## B. Calibration transfer and threshold interpretation

Table III separates calibration selection from test observations. The selected rule attains 0.498% on calibration placements. Frozen-threshold test rates exceed the 0.5% constraint by 0.09, 0.12, and 0.15 percentage points. Sampling and distribution differences may contribute, but their effects cannot be separated from these rates. Calibration placement counts are unreported, precluding a sampling-error estimate.

Only one calibrated threshold is reported, so sensitivity cannot be recovered from these aggregates. Lower thresholds increase candidate rejection on a fixed score list, but refill can promote surviving targets. A sensitivity analysis must rerun admission and refill and measure both exposure and clean utility.

## C. Construction families and component comparisons

Table IV shows reductions in every construction–retriever pair. Doorway pages have the largest remaining exposure for each retriever, approximately 0.21–0.23. Near duplicates retain exposure under all three retrievers, with lower post-quarantine values for LSA than for BM25 or the hybrid. Copied relevant evidence is a plausible source of difficulty, but these aggregate outcomes do not isolate that mechanism.

TABLE IV  
EXPOSURE@10 BY CONSTRUCTION FAMILY
<table><tr><td>Family</td><td>BM25</td><td>Hybrid</td><td>LSA</td></tr><tr><td>Title stuffing</td><td>1.000/.083</td><td>1.000/.083</td><td>.983/.080</td></tr><tr><td>Doorway</td><td>1.000/.230</td><td>.987/.233</td><td>.847/.207</td></tr><tr><td>Near duplicate</td><td>.763/.197</td><td>.680/.163</td><td>.590/.133</td></tr><tr><td>Hidden metadata</td><td>1.000/.123</td><td>.997/.123</td><td>.960/.130</td></tr></table>

Each cell gives no-defense/joint-quarantine exposure.

TABLE V  
COMPONENT POLICIES AT THEIR RECORDED OPERATING POINTS
<table><tr><td>Policy</td><td>Exposure@10</td><td>Clean ∆NDCG@10</td></tr><tr><td>Abuse only</td><td>.1436</td><td>-.1113</td></tr><tr><td>Instruction only</td><td>.4169</td><td>-.0071</td></tr><tr><td>Joint quarantine</td><td>.1489</td><td>-.0014</td></tr></table>

Means across retrievers. Policies are not matched at a common false-quarantine rate; interaction effects are not isolated.

The component results in Table V compare recorded operating points. Abuse-only screening attains exposure 0.1436 but reduces clean NDCG by 0.1113. Instruction-only screening has a smaller clean loss while leaving exposure at 0.4169. Joint quarantine combines exposure 0.1489 with a reported mean clean change of −0.0014. The latter is the recorded aggregate; rounded per-retriever entries need not reproduce its last digit.

These are complete-policy comparisons. A matched $[ r , a , i ]$ model was not evaluated, and single-signal policies were not matched at a common false-quarantine rate. The observed trade-off cannot therefore be attributed to the interactions. A direct ablation must refit both feature sets on identical data and select separate thresholds under the same calibration constraint.

Continuous reranking is sensitive to the retriever. For the hybrid, exposure is 0.083 with a clean NDCG decrease of 0.062; under BM25, exposure remains 0.878. The two-stage penalty changes exposure by only 0.1–0.5 percentage points relative to quarantine while incurring a larger clean loss. These outcomes motivate direct quarantine as the primary reported policy, but they are also unmatched operating-point comparisons.

The instruction detector’s held-out-category recall is 0.754, with calibration false quarantine 0.00076 at its reported operating point. Under hybrid retrieval, question-like sentiment requests reach post-quarantine exposure 0.500, followed by conversational requests at 0.476, research assistance at 0.346, task automation at 0.298, and business intelligence at 0.269. Smooth transitions from benign prose are also a challenge in TopicAttack [21]. Adding the simulated source marker gives perfect calibration recall, but that marker encodes benchmark provenance and is excluded from the primary model.

![](images/fd72f05cee8053988023f13a4f4149707f5c38deb76ada45a0281ac53b6bb819.jpg)

![](images/b1344fb104ed4b7eee1641061963eabff6acf732c31f75a237ff4ad600ec925d.jpg)  
Fig. 2. Exposure and clean retrieval utility at the recorded operating point. Panel (a) shows the large decrease in target admission; panel (b) shows the small measured clean NDCG change. Bars reproduce Table II; they do not depict an operating-point sweep. The query-level analytical uncertainty for exposur reduction is reported in the text.

TABLE VI  
EXPLORATORY GENERATION OBSERVATIONS BY MODEL AND PROMPT
<table><tr><td>Model</td><td>Prompt</td><td>None</td><td>Joint</td><td>Clean correct</td></tr><tr><td>Qwen3-4B</td><td>Conventional</td><td>0/4</td><td>0/4</td><td>36/90</td></tr><tr><td>Qwen3-4B</td><td>Hardened</td><td>0/4</td><td>0/4</td><td>34/90</td></tr><tr><td>Phi-4-mini</td><td>Conventional</td><td>0/4</td><td>0/4</td><td>47/90</td></tr><tr><td>Phi-4-mini</td><td>Hardened</td><td>0/4</td><td>0/4</td><td>47/90</td></tr></table>

None and Joint report confirmed/unresolved judgments among 104 attacked cases in each cell. Clean accuracy uses 90 distinct queries. Counts do not estimate a defense effect on attack success.

## D. Exploratory downstream observations

All 104 selected targets enter the undefended top five. After quarantine, 21 remain and 83 contexts no longer contain the target instruction. This is a retrieval observation and does not depend on whether an output is judged to execute the instruction.

Table VI reports behavioral observations. Fourteen case pairs share queries, leaving 90 distinct clean queries. Across two models and two prompts, 4(90 + 104 + 104) = 1, 192 responses cover clean, attacked, and filtered conditions. Expanding clean answers to associated cases produces $4 \times 3 \times$ 104 = 1, 248 judgments.

The endpoint confirms no hijacks, assigns failure to 1,200 judgments, and leaves 48 unresolved. Four underspecified monoalphabetic-substitution cases recur across model– prompt–condition cells. Both attacked conditions retain zero confirmed and four unresolved judgments per 104-case cell. Carrier removal does not resolve behavioral ambiguity or establish zero attack success.

Five provisional positives lacked the requested Caesar transformation; two misspelled claim labels lacked the requested automation output. Clean accuracy is 40.0%/37.8% for Qwen3-4B under conventional/hardened prompts and 52.2% for Phi-4-mini under either prompt. The post hoc endpoint, repeated queries, and absence of confirmed undefended hijacks preclude an attack-success reduction estimate.

## VI. DISCUSSION AND REPRODUCIBILITY

## A. Interpretation and remaining validation

The measured benefit is a lower rate of target admission at small observed clean ranking cost. All page constructions were fixed before final evaluation. Adaptive attacks can exploit knowledge of a decision rule, and recent evaluations show that static attack suites can overstate protection [22]. Neither a construction-level holdout nor a defense-aware attack was evaluated here.

A matched-cost component experiment should compare [r, a, i] with [r, a, i, ar, ir] under identical splits and fitting procedures. A prespecified grid of calibration constraints would support threshold sensitivity, reporting test exposure, false quarantine, NDCG, and refill behavior without test-set retuning. Those comparisons remain unreported.

SciFact omits rendered pages, links, freshness, publisher reputation, and client-side content. Dense retrievers and independent construction families are needed for broader transfer. Dependence between queries sharing sources would require higher-level clustering from the original manifests.

## B. Layered deployment and residual errors

Question-like instructions and copied evidence resemble legitimate content. Provenance or duplicate-cluster information may help, but remains untested. Source-level monitoring is also needed because average placement cost can hide repeated false quarantine of one publisher.

AttriGuard checks causal support for tool invocations, and Task Shield checks alignment with the authorized task [23], [24]. Admitted candidates still require such downstream controls.

Blocker documents target answer availability, while opinion poisoning targets response stance [25], [26]. These harms are outside our task-displacement endpoint.

## C. Reproducibility status

The reported run contains 31,500 query-level rows and 315,000 top-10 rows in 105 scenario–retriever–policy groups, each with 300 queries. Its accounting checks covered duplicate identifiers, absent or invalid scores, source joins, and target exposure in clean scenarios. These counts do not substitute for the underlying records.

Exact replay requires input hashes and folds, target manifests, normalization parameters, vectorizer vocabulary and tokenization, fitted coefficients, ranker settings, software versions, raw scores, and rankings before and after refill. Generation additionally requires model and quantization revisions, prompts, truncation code, responses, and endpoint labels. Calibration placement counts are also needed.

The accompanying sources recompile the manuscript, regenerate figures from reported aggregates, and reproduce the analytical bounds. Original candidate records and trained artifacts are unavailable in this revision. The package therefore cannot replay the experiments or supply the missing component and threshold analyses; no public anonymous experimental archive is claimed.

## VII. CONCLUSION

SearchInject applies a relevance-conditioned admission rule before retrieved pages enter model context. Across held-out queries and payload categories within four known construction families, the recorded policy reduces exposure by 83.2–83.7%, while the largest absolute clean NDCG@10 change is 0.0018. The 0.5% calibration constraint is empirical: observed clean test false quarantine is 0.59–0.65%. Query-level analytical bounds account for repeated construction outcomes under an independent-query assumption.

Interaction effects, matched-cost comparisons, and downstream attack-success reduction remain unestablished. Resolving them requires original candidate records, additional calibrated comparisons, and a prospectively defined generation endpoint with confirmed undefended attacks.

## REFERENCES

[1] K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and M. Fritz, “Not what you’ve signed up for: Compromising real-world LLMintegrated applications with indirect prompt injection,” in Proc. 16th ACM Workshop on Artificial Intelligence and Security, 2023, pp. 79–90.

[2] Z. Gyöngyi and H. Garcia-Molina, “Web spam taxonomy,” in Proc. 1st Int. Workshop on Adversarial Information Retrieval on the Web, 2005, pp. 39–47.

[3] Q. Zhan, Z. Liang, Z. Ying, and D. Kang, “InjecAgent: Benchmarking indirect prompt injections in tool-integrated large language model agents,” in Findings of ACL, 2024, pp. 10471–10506.

[4] E. Debenedetti, J. Zhang, M. Balunovic, L. Beurer-Kellner, M. Fischer, and F. Tramèr, “AgentDojo: A dynamic environment to evaluate prompt injection attacks and defenses for LLM agents,” in Advances in Neural Information Processing Systems, vol. 37, 2024.

[5] H. Chang, E. Bao, X. Luo, and T. Yu, “Overcoming the retrieval barrier: Indirect prompt injection in the wild for LLM systems,” in Proc. 35th USENIX Security Symp., 2026, pp. 1567–1586.

[6] J. Yi, Y. Xie, B. Zhu, E. Kiciman, G. Sun, X. Xie, and F. Wu, “Benchmarking and defending against indirect prompt injection attacks on large language models,” in Proc. 31st ACM SIGKDD Conf. on Knowledge Discovery and Data Mining, vol. 1, 2025, pp. 1809–1820.

[7] Y. Liu, Y. Jia, R. Geng, J. Jia, and N. Z. Gong, “Formalizing and benchmarking prompt injection attacks and defenses,” in Proc. 33rd USENIX Security Symp., 2024, pp. 1831–1847.

[8] P. Lewis et al., “Retrieval-augmented generation for knowledge-intensive NLP tasks,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 9459–9474.

[9] W. Zou, R. Geng, B. Wang, and J. Jia, “PoisonedRAG: Knowledge corruption attacks to retrieval-augmented generation of large language models,” in Proc. 34th USENIX Security Symp., 2025, pp. 3827–3844.

[10] A. Ntoulas, M. Najork, M. S. Manasse, and D. Fetterly, “Detecting spam web pages through content analysis,” in Proc. 15th Int. Conf. on World Wide Web, 2006, pp. 83–92.

[11] Y. Liu, Y. Jia, J. Jia, D. Song, and N. Z. Gong, “DataSentinel: A gametheoretic detection of prompt injection attacks,” in Proc. IEEE Symp. on Security and Privacy, 2025, pp. 2190–2208.

[12] Y. Jia, Y. Liu, Z. Shao, J. Jia, and N. Z. Gong, “PromptLocate: Localizing prompt injection attacks,” in Proc. IEEE Symp. on Security and Privacy, 2026, pp. 4243–4261.

[13] S. Chen, J. Piet, C. Sitawarin, and D. Wagner, “StruQ: Defending against prompt injection with structured queries,” in Proc. 34th USENIX Security Symp., 2025, pp. 2383–2400.

[14] S. Chen, A. Zharmagambetov, S. Mahloujifar, K. Chaudhuri, D. Wagner, and C. Guo, “SecAlign: Defending against prompt injection with preference optimization,” in Proc. ACM SIGSAC Conf. on Computer and Communications Security, 2025, pp. 2833–2847.

[15] Y. Kaya, A. Landerer, S. Pletinckx, M. Zimmermann, C. Kruegel, and G. Vigna, “When AI meets the web: Prompt injection risks in third-party AI chatbot plugins,” in Proc. IEEE Symp. on Security and Privacy, 2026.

[16] D. Wadden, S. Lin, K. Lo, L. L. Wang, M. van Zuylen, A. Cohan, and H. Hajishirzi, “Fact or fiction: Verifying scientific claims,” in Proc. EMNLP, 2020, pp. 7534–7550.

[17] N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, and I. Gurevych, “BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models,” in Proc. Neural Information Processing Systems Track on Datasets and Benchmarks, vol. 1, 2021.

[18] S. E. Robertson and H. Zaragoza, “The probabilistic relevance framework: BM25 and beyond,” Foundations and Trends in Information Retrieval, vol. 3, no. 4, pp. 333–389, 2009.

[19] V. Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov, D.˘ Chen, and W.-T. Yih, “Dense passage retrieval for open-domain question answering,” in Proc. EMNLP, 2020, pp. 6769–6781.

[20] W. Hoeffding, “Probability inequalities for sums of bounded random variables,” Journal of the American Statistical Association, vol. 58, no. 301, pp. 13–30, 1963, doi: 10.1080/01621459.1963.10500830.

[21] Y. Chen, H. Li, Y. Li, Y. Liu, Y. Song, and B. Hooi, “TopicAttack: An indirect prompt injection attack via topic transition,” in Proc. EMNLP, 2025, pp. 7327–7345.

[22] M. Nasr et al., “The attacker moves second: Stronger adaptive attacks bypass defenses against LLM jailbreaks and prompt injections,” in Proc 35th USENIX Security Symp., 2026, pp. 1467–1486.

[23] Y. He, H. Zhu, Y. Li, S. Shao, H. Yao, Z. Liu, and Z. Qin, “AttriGuard: Defeating indirect prompt injection in LLM agents via causal attribution of tool invocations,” in Proc. 35th USENIX Security Symp., 2026, pp. 1547–1566.

[24] F. Jia, T. Wu, X. Qin, and A. Squicciarini, “The task shield: Enforcing task alignment to defend against indirect prompt injection in LLM agents,” in Proc. ACL, 2025, pp. 29680–29697.

[25] A. Shafran, R. Schuster, and V. Shmatikov, “Machine against the RAG: Jamming retrieval-augmented generation with blocker documents,” in Proc. 34th USENIX Security Symp., 2025, pp. 3787–3806.

[26] Y. Gong, Z. Chen, J. Liu, M. Chen, F. Yu, W. Lu, X. Wang, and X. Liu, “Topic-FlipRAG: Topic-orientated adversarial opinion manipulation attacks to retrieval-augmented generation models,” in Proc. 34th USENIX Security Symp., 2025, pp. 3807–3826.