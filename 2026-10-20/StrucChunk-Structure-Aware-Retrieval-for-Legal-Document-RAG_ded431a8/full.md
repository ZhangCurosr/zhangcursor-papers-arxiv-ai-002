# StrucChunk: Structure-Aware Retrieval for Legal Document RAG

Surya Deep Singh<sup>1\*</sup>, Ashutosh Tripathi<sup>1</sup> and Gaurav Deep Singh<sup>1</sup> <sup>1\*</sup>Independent Researcher.

\*Corresponding author(s). E-mail(s): suryadeepsingh95@gmail.com; Contributing authors: ashutoshtripathi191@gmail.com; gauravdeepsingh1996@gmail.com;

## Abstract

Retrieval-Augmented Generation (RAG) systems ofer a promising approach for grounding Large Language Model outputs in domain-specific knowledge. Applying RAG to legal documents presents unique challenges: standard chunking strategies fragment hierarchical argument structure, and generic embedding models struggle with the dual vocabulary demands of precise citation matching and semantic understanding. To deal with this, We propose StrucChunk (Structure-Aware Chunking) combined with hybrid dense-sparse retrieval, designed specifically for hierarchically structured legal documents. Our method treats documents as directed acyclic graphs of provisions, preserving clause boundaries and resolving cross-references prior to embedding, with Reciprocal Rank Fusion combining BM25 keyword precision with BGE-M3 semantic representations. We have evaluated our method on five corpora, three legal traditions, and two languages. StrucChunk achieves 100% boundary precision (vs. 7.7% for recursive splitting on CrPC), preserves 97.6% of documented cross-references (vs. 0%), and achieves equivalent section coverage with 40.3% fewer chunks on GDPR. Empirically, on LegalBench-RAG—an independently annotated benchmark of 6,858 querypassage pairs—StrucChunk achieves Char-Recall@8 of 48.18% vs. 23.59% for the best baseline (+104%), surpassing the published state-of-the-art by +83%. On 501 queries each for CrPC and CPC (Indian law), Recall@5 improves from 0.557 to 0.804 (+44.4%) and from 0.158 to 0.259 (+64.6%) respectively; chunking accounts for 88% and 73% of total gain. On 501 GDPR queries (EU law), Recall@5 = 0.868 with R@1 improving +59%. On NitiBench-CCL (3,730 queries, Thai financial law), StrucChunk achieves HR@5 = 0.856 vs. 0.594 for naive chunking (+44.1%), confirming that section-boundary alignment is the dominant factor across languages; naive fixed-size chunking structurally cannot represent 29.6% of Thai legal sections whose text exceeds the chunk limit, and StrucChunk’s

cross-reference augmentation provides an additional +2.3% specifically on crossreference queries. We have also released implementation, 5,256 evaluation queries across five legal corpora, annotation methodology, and chunking quality metrics. The code is available at github.com/SinghSuryaDeep/legal-rag-strucchunk-1.

Keywords: Retrieval-Augmented Generation, Legal NLP, Document Chunking, Hybrid Search, Information Retrieval

## 1 Introduction

The integration of Large Language Models (LLMs) into legal practice has accelerated dramatically, with AI-powered tools now assisting practitioners in contract review, statutory research, regulatory compliance checking, and litigation support. Yet for these tools to be deployable in high-stakes settings—where an erroneous statutory citation can constitute professional malpractice and an incorrect interpretation of a sub-provision can afect fundamental rights—factual precision is non-negotiable. Retrieval-Augmented Generation (RAG) [1] ofers a principled approach to this challenge by grounding LLM outputs in authoritative source documents retrieved at query time, thereby reducing reliance on parametric knowledge that may be outdated or hallucinated.

However, even RAG-augmented legal systems exhibit alarming error rates. Dahl et al. [2] found hallucination rates of 58–88% on case law queries; Magesh et al. [3] documented 17–33% hallucination rates in commercial RAG-augmented legal tools, with retrieval failures accounting for ≈40–50% of errors. These findings reveal a fundamental weakness: RAG is only as reliable as what it retrieves. A system that fetches the wrong provision cannot generate a correct legal answer regardless of the LLM’s parametric capability. The bottleneck lies squarely in the retrieval layer. Kabir et al. [4] confirm this empirically, reporting that standard RAG pipelines achieve semantic similarity of only 0.76 on legal documents—significantly below 0.91 on scientific text and 0.90 on financial documents—establishing that legal retrieval is a categorically harder problem that demands domain-specific solutions. These findings collectively establish that efective legal RAG requires domain-specific optimization at every pipeline stage.

Legal documents pose three retrieval challenges that standard RAG approaches fail to address.

Hierarchical structure. Statutes are organized by parts, chapters, sections, and subsections, where the hierarchical position of a provision determines its legal scope and efect. Naive chunking by token count severs these logical units, losing the context required to interpret sub-provisions. For instance, a subsection specifying an exception to a legal obligation is uninterpretable in isolation; splitting at token boundaries creates orphaned chunks whose meaning depends entirely on a parent provision that may not be retrieved.

Cross-references. Legal language relies heavily on internal pointers (“subject to Section 167,” “as defined in Article 4(1)”). Standard chunking orphans referenced content from its source, forcing LLMs to reason about incomplete legal specifications.

When a chunk references another provision, the content of that provision is not merely helpful—it is legally required for accurate interpretation.

Vocabulary duality. Legal queries require both exact keyword matching (section numbers, article citations) and semantic understanding (“bail” ≈ “custody,” “consent” ≈ “agreement”). Neither pure dense retrieval (which excels at semantic similarity but misses exact citation matching) nor pure sparse retrieval (which excels at keyword matching but misses paraphrase and synonym relationships) resolves this tension alone.

The root cause of these failures is that standard RAG pipelines treat legal documents as flat sequences of text to be divided by token count. This representation discards precisely the information most important for legal retrieval: the hierarchical position of a provision within the statutory scheme, the identity of provisions it cross-references, and the breadcrumb trail from the governing chapter to the specific sub-clause. We propose that efective legal retrieval requires what we term a Legal Map—a representation that preserves the document’s logical structure as a navigational aid embedded directly in chunk content prior to indexing, rather than stored as metadata that generic embedding models ignore. This structural signal, we argue, is the dominant factor in retrieval quality for hierarchically organized legal corpora.

We have summarized our all contributions as follows:

1. StrucChunk (Structure-Aware Chunking): a chunking strategy treating legal documents as directed acyclic graphs (DAGs) of nested provisions, preserving clause boundaries and prepending hierarchical context, with sentence-boundary splitting for oversized sections.

2. Hybrid Retrieval Pipeline: BM25 combined with BGE-M3 embeddings via Reciprocal Rank Fusion, jointly addressing keyword precision and semantic understanding.

3. Chunking Quality Metrics: four intrinsic metrics validating the chunking contribution independently of retrieval: boundary precision, cross-reference preservation rate, chunk count eficiency, and intra-chunk semantic coherence.

4. Evaluation: independent validation (LegalBench-RAG), mechanistic analysis (CrPC + CPC), cross-domain generalization (GDPR), and cross-language generalization (NitiBench-CCL, Thai), following the empirical rigor principles of Sculley et al. [5].

5. Annotated Evaluation Datasets: 5,256 queries across five legal corpora (CrPC: 501, CPC: 501, GDPR: 501, LBR: 776 independent, NitiBench-CCL: 3,730 independent), with documented annotation methodology.

These contributions yield consistent, large-magnitude improvements validated across five corpora spanning three legal traditions and two languages. On LegalBench-RAG (independently annotated, 776 queries), StrucChunk achieves Char-Recall@8 = 48.18% vs. 23.59% for the best baseline (+104%), surpassing published state-ofthe-art by +83%. On CrPC (501 queries), Recall@5 improves from 0.557 to 0.804 (+44.4%), with 2×2 ablation confirming that chunking accounts for 88% of the total gain. On NitiBench-CCL (3,730 Thai queries, independently annotated), HR@5 = 0.856 vs. 0.594 for naive chunking (+44.1%), and structural analysis reveals that 29.6% of Thai legal sections are structurally unrepresentable by any naive fixed-size chunk. All improvements are statistically significant $( p \leq 0 . 0 0 2$ , Wilcoxon signed-rank with Benjamini-Hochberg correction; Cohen’s d 0.14–0.69).

Our work specifically targets retrieval quality for hierarchically structured legal documents. We do not evaluate downstream answer quality or claim generalization to unstructured legal document types without further validation.

Paper structure. Section 2 reviews related work on RAG systems, legal NLP, hierarchical document structure, chunking strategies, and hybrid retrieval. Section 3 introduces the StrucChunk algorithm and hybrid retrieval pipeline, preceded by a formal problem statement. Section 4 details the experimental design, corpus statistics, baselines, and evaluation protocols. Section 5 presents results, from intrinsic chunking quality metrics through all retrieval evaluations. Section 6 discusses the Legal Map hypothesis, cross-language generalization, failure analysis, and limitations. Section 7 addresses broader impact, and Section 8 concludes with future directions.

## 2 Related Work

## 2.1 Retrieval-Augmented Generation

The RAG paradigm was formalized by Lewis et al. [1], combining a pre-trained seq2seq model with Dense Passage Retrieval. Subsequent work evolved RAG from naive implementations to modular and adaptive systems [6]. Self-RAG [7] introduced adaptive retrieval with self-critique tokens that enable on-demand retrieval based on query complexity; Corrective RAG [8] added retrieval quality assessment to filter and correct retrieved passages before generation. These advances are directly applicable to legal RAG, where query complexity varies greatly—from simple section lookups to multihop cross-reference queries—and where retrieval errors propagate into hallucinated legal advice with serious professional consequences.

The RAG pipeline consists of two separable components: a retrieval module that selects relevant passages from a corpus, and a generation module that synthesizes answers from retrieved context. A critical observation for the legal domain is that improving the generation component cannot compensate for retrieval failures: if the relevant provision is not retrieved, no generative model can produce a correct answer. Recent work demonstrates that generic RAG pipelines significantly underperform on legal text. Kabir et al. [4] report semantic similarity of only 0.76 on legal documents versus 0.91 on scientific text and 0.90 on financial documents, motivating domain-specific optimization at the chunking and retrieval stages. This paper focuses exclusively on retrieval-layer optimization, establishing that chunking architecture is the dominant factor in retrieval quality for hierarchically structured legal corpora.

## 2.2 Legal NLP and Retrieval

Legal-BERT [9] established domain-specific pre-training for legal NLP, demonstrating substantial gains on tasks including contract clause classification and court judgment prediction. LegalBench [? ] provides 162 legal reasoning tasks spanning statutory interpretation, contract analysis, and judicial reasoning. These benchmarks, however, supply the relevant context directly to the model; they do not evaluate the ability of a retrieval system to locate that context within a large corpus.

LegalBench-RAG [10] is the first benchmark specifically targeting retrieval in legal RAG, with 6,858 annotated query-answer pairs over 79M+ characters from four contract corpora (CUAD, ContractNLI, MAUD, PrivacyQA). Critically, LegalBench-RAG evaluates using character-span matching: the ground truth is a specific character range within the source document, and a retrieved chunk is scored by its overlap with that span. This evaluation protocol directly measures whether the retrieval system surfaces the minimally suficient legal text, which we adopt for our evaluation number 1.

Reuter et al. [11] identified Document-Level Retrieval Mismatch (DRM) as a critical failure mode in legal RAG, where the system selects a document that is broadly relevant but retrieves the wrong internal passage. They propose Summary-Augmented Chunking (SAC), which prepends document-level summaries to chunks to provide broader context. Our StrucChunk is complementary to SAC: SAC addresses document-level context (helping distinguish among similar documents), while Struc-Chunk preserves intra-document hierarchical structure (helping locate the correct provision within the right document). Combining both approaches is identified as a future direction in Section 8.

Akarajaradwong et al. [12] introduced NitiBench, a benchmark for Thai legal QA comprising NitiBench-CCL (3,730 queries, 21 Thai financial law codes) and NitiBench-Tax (50 tax case queries). They proposed hierarchy-aware chunking (section-boundary splitting) and NitiLink (post-retrieval cross-reference expansion), finding that hierarchy-aware chunking significantly outperforms naive line chunking, but NitiLink does not consistently improve end-to-end performance. We use NitiBench-CCL as our cross-language evaluation and compare their hierarchy-aware chunking directly against StrucChunk, finding that StrucChunk’s pre-embedding cross-reference augmentation provides targeted improvement (+2.3%) on cross-reference queries where NitiLink’s post-retrieval expansion was inefective.

Nigam et al. [13] address legal judgment prediction in the Indian Supreme Court context through a RAG framework, NyayaRAG, that provides models with summarized case facts, retrieved statutory provisions, and semantically similar precedent judgments. Their ablation study confirms that structured legal knowledge (statutes, cited precedents) significantly improves prediction accuracy and explanation quality over unstructured case text alone, further motivating the importance of structure-aware legal knowledge retrieval for downstream legal AI tasks.

## 2.3 Hierarchical Document Structure and Legal Parsing

Legal statutes present a distinctive structural hierarchy that difers fundamentally from other long-form text corpora such as scientific papers or news articles. A single statute may organize hundreds or thousands of provisions across Parts, Chapters, Sections, Sub-sections, and Clauses, where each level of the hierarchy constrains the interpretation of its descendants. EU regulatory instruments such as the GDPR additionally introduce parallel structural components: Articles specify operative legal obligations, while Recitals (1–173 in the GDPR) provide interpretive context for those

Articles but carry diferent legal weight. Indian procedural codes (CrPC, CPC) organize 517 and 158+ sections respectively with a dense cross-reference network spanning hundreds of provisions.

The standard approach to representing long documents in neural retrieval is to ignore this structure, treating the document as a flat sequence partitioned by token count. Legal-BERT [9] operates at the sentence or paragraph level without encoding section identity; retrieval benchmarks such as CUAD [14] and ContractNLI evaluate clause spans in isolation. While these choices are appropriate for classification tasks where the section boundary is irrelevant to the label, they are ill-suited for retrieval tasks where the section is the retrieval target.

A key failure mode arising from ignoring structure is what Akarajaradwong et al. [12] term “Hidden Hierarchical Information”: multiple sections across diferent chapters convey near-identical semantic content but difer only in hierarchical position (e.g., provisions in Revenue Code §27 and §89/1 both govern late payment charges but reside in diferent chapters and impose diferent scopes of obligation). Without hierarchical context in the embedding, dense retrieval systems cannot distinguish them. StrucChunk’s breadcrumb mechanism—prepending [Law > Chapter > Section N] to each chunk—directly addresses this failure mode by encoding hierarchical identity into the dense representation prior to indexing.

Document parsing quality is an often-overlooked bottleneck in legal RAG. Kabir et al. [4] employ OCR-based preprocessing (Tesseract) for bilingual Bangla-English gazettes, confronting the practical challenge that real-world legal documents exhibit inconsistent formatting, mixed languages, and irregular section delimiters. Our approach uses pdfplumber with custom structure detection for English statutory documents (1–3), and leverages pre-structured HuggingFace data for Thai law (4), allowing us to isolate the contribution of structure-aware chunking from PDF parsing quality and to demonstrate that the chunking benefit generalizes across both parsed and pre-structured corpora.

<table><tr><td>Method</td><td>Section Boundaries</td><td>Cross-Ref Aug.</td><td>Multi-Corpus</td><td>Cross-Language</td></tr><tr><td>Naive Fixed-Size</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>RCTS</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>SAC [11]</td><td>Partial (doc-level)</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Hierarchy-Aware [12]</td><td>√</td><td>x</td><td>x</td><td>Partial</td></tr><tr><td>StrucChunk (Ours)</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 1 Positioning of StrucChunk relative to prior chunking methods. ✓= supported; ✗= not supported. “Section Boundaries” = chunk boundaries align with legal section delimiters. “Cross-Ref $\mathrm { A u g . } ^ { \mathfrak { N } }$ = referenced provision content embedded at index time. “Multi-Corpus” = evaluated across ≥3 distinct legal corpora. “Cross-Language” = validated on non-English legal text.

Table 1 positions StrucChunk against prior chunking approaches along four design dimensions. The key diferentiator is the combination of pre-embedding crossreference augmentation with section- boundary alignment, validated across multiple legal traditions and languages—a combination not addressed by any prior work.

## 2.4 Chunking Strategies

Yepes et al. [15] demonstrated that structure-aware chunking for financial reports achieves the highest retrieval scores with only half the chunks of structure-agnostic methods. The mechanism is analogous to our legal setting: financial reports are hierarchically organized (sections, tables, notes) and structure-aware chunking aligns chunk boundaries with these semantic units. We implement the same chunk-count-eficiency metric (Section 5.1) and confirm the pattern in the legal domain: StrucChunk achieves 100% section coverage on GDPR with 40.3% fewer chunks than recursive splitting.

Contextual chunk enrichment is an emerging direction. Anthropic’s Contextual Retrieval approach prepends document-level summaries to chunks to address decontextualization. Reuter et al. [11] extend this with SAC, which prepends document-level summaries specifically to address DRM. StrucChunk’s contextual prepend is mechanistically diferent: rather than summarizing the document, it encodes the chunk’s position within the document hierarchy—a piece of information that can be extracted deterministically without LLM-based summarization and that is more directly relevant to the retrieval signal for provision-level queries.

## 2.5 Hybrid Retrieval

Reciprocal Rank Fusion [16] is the standard fusion method for dense-sparse hybrid retrieval, combining rankings from dense and sparse systems without requiring score calibration. Bruch et al. [17] showed that convex combination can outperform RRF in some settings, though RRF is more robust to the calibration challenge and is the method we adopt $( k = 6 0$ following [16]).

The LegalBench-RAG study [10] found that a general-purpose Cohere reranker hurt performance on legal text (Recall@64 dropped from 61.06% to 47.54%), validating our design choice to focus on domain-aware chunking as the primary quality mechanism rather than post-hoc reranking. Our experiments corroborate this: the retrieval architecture contributes substantially less than chunking in all ablations (contributing 12–27% of total gain), and adding a general-purpose reranker to poorly chunked text cannot recover the information lost to boundary fragmentation.

## 3 Method

## 3.1 Problem Formulation

Let D denote a hierarchically structured legal document whose content is organized into N structural units (sections, articles, sub-provisions) $\mathcal { D } = \{ s _ { 1 } , s _ { 2 } , . . . , s _ { N } \}$ , where each unit $s _ { i }$ occupies a node in the document hierarchy H (a directed acyclic graph). A chunking function $f : \mathcal { D }  \mathcal { C }$ partitions (and optionally augments) the document into a corpus of retrievable units $\mathcal { C } = \{ c _ { 1 } , c _ { 2 } , \ldots , c _ { M } \}$ . A retrieval function $r : \mathcal { C } \times \mathcal { Q }  R _ { k } ( q )$ maps a query $q \in \mathcal { Q }$ to a ranked list $R _ { k } ( q )$ of k chunks.

Given a query q with a known set of relevant sections $S ^ { \ast } ( q ) \subseteq { \mathcal { D } }$ , the retrieval task succeeds at rank k if $R _ { k } ( q ) \cap f ^ { - 1 } ( S ^ { * } ( q ) ) \neq \emptyset$ , where $f ^ { - 1 } ( s )$ denotes the chunk(s) derived from section s. Recall@k is the fraction of queries for which the task succeeds at rank k.

The central claim of this paper is that the design of $f$ is the dominant factor in Recall@k for hierarchically structured legal documents. This claim follows directly from three structural properties:

• Coverage: if $f$ cannot represent $s ^ { * } \in S ^ { * } ( q )$ (because no chunk in C contains the full text of $s ^ { * } )$ , retrieval fails regardless of $r .$

• Boundary alignment: if f splits $s ^ { * }$ across multiple chunks, the contextually coherent representation is fragmented, degrading the embedding quality of the relevant chunk.

• Cross-reference co-location: if $s ^ { * }$ semantically depends on a cross-referenced provision $s ^ { \prime } { \mathrm { . } }$ , and $f$ does not co-locate $s ^ { \prime } \mathrm { { } s }$ content in $s ^ { * } \mathrm { \bar { s } }$ chunk, the embedding of $s ^ { * }$ ’s chunk is incomplete.

We formalize these properties as chunking quality metrics (Section 5.1) and validate each property through $2 \times 2$ ablation (chunking architecture × retrieval method) across four corpora.

## 3.2 Legal Document Parsing

We employ pdfplumber with custom structure detection identifying chapters/parts, sections, subsections, and cross-references. We model legal documents as Directed Acyclic Graphs (DAGs) where nodes represent structural units and edges represent containment relationships (Part → Chapter → Section → Subsection).

The parser tracks a char offset for each section. These ofsets are used by (1) the chunking quality boundary precision metric, and (2) the LegalBench-RAG evaluator for character-span matching.

For GDPR we extend the parser to handle the Article-Recital dual architecture. For NitiBench-CCL, section data is pre-structured in the public HuggingFace dataset (VISAI-AI/nitibench); no PDF parsing is required.

## 3.3 StrucChunk: Structure-Aware Chunking

StrucChunk implements three principles.

Logical Atomicity. If a section is N tokens and the limit is L, we keep the section whole (up to 1.5L) rather than meeting the limit exactly.

Contextual Prepend. Each chunk is prepended with its hierarchical breadcrumb, e.g., [CrPC 1973 > Chapter XII > Section 167], [GDPR > Chapter II > Article 6], or [law name > m¯atr¯a N]

(Thai).

Cross-Reference Augmentation. When a chunk references another provision we append a 200-character summary of the referenced content, detected via the patterns in Appendix ??. For NitiBench-CCL, cross-reference links are loaded directly from the dataset’s structured reference annotations.

Sentence-boundary splitting. When a section exceeds 1.5× max tokens we split at sentence boundaries (nltk.sent tokenize, regex fallback).

Algorithm 1 StrucChunk: Structure-Aware Chunking   
Require: Document $D = \{ s _ { 1 } , \ldots , s _ { N } \}$ with hierarchy H; chunk size limit $L ;$ overlap   
ratio $\rho ;$ maximum cross-references $r _ { \operatorname* { m a x } } = 3 ;$ summary length ℓ = 200 characters   
Ensure: Chunk set $C = \{ c _ { 1 } , \ldots , c _ { M } \}$   
1: $C  \emptyset$   
2: for each section $s _ { i } \in D$ do   
3: b<sub>i</sub> ← hierarchical breadcrumb of $s _ { i }$ in H   
(concatenation of ancestor titles: document → chapter → section number)   
4: $R _ { i }$ ← set of cross-references in $s _ { i }$ resolving to a valid section in D   
(deduplicated, truncated to $| R _ { i } | \leq r _ { \operatorname* { m a x } } )$   
5: $t _ { i } \gets b _ { i } \parallel { } ^ { \mathrm { * } } \mathrm { S }$ ection i. title” ∥ content $\left( { { s } _ { i } } \right)$   
6: if $R _ { i } \neq \varnothing$ then   
7: $t _ { i } \gets t _ { i } \parallel \quad \parallel$ summary $( s _ { j } , \ell )$   
$s _ { j } \stackrel { \because } { \in } R _ { i }$   
8: $\tau _ { i } \gets | t _ { i } | / 4$ ▷ character-based token approximation   
9: if $\tau _ { i } \leq 1 . 5 L$ then   
10: $C \gets C \cup \{ t _ { i } \}$ ▷ logical atomicity: keep section whole   
11: else   
12: $U _ { i } \gets$ paragraph-level segmentation of $t _ { i }$   
(any unit still exceeding 1.5L is further split at sentence boundaries)   
13: $C \gets C \cup \operatorname* { P A C K } ( U _ { i } , L , \rho )$   
(greedily fills L-token windows, carrying ρL overlapping   
tokens from each window’s tail into the next)   
14: return C

## 3.4 Hybrid Retrieval Pipeline

Dense Retrieval: BAAI/bge-m3 [18]—8192-token context, L2-normalized, FAISS IndexFlatIP.

Sparse Retrieval: BM25 via bm25s.

Reciprocal Rank Fusion: RRF<sub>score</sub>(d) = P<sub>r∈{Dense,Sparse}</sub> <sup>1</sup><sub>k+rank (d)</sub> , k = 60.<sup>1</sup>

We have given complete algorithm in Algorithm 1.

## 4 Experimental Setup

## 4.1 LegalBench-RAG (Independent Benchmark)

LegalBench-RAG [10] provides 6,858 query-answer pairs over legal contracts (CUAD, ContractNLI, MAUD, PrivacyQA).<sup>2</sup> We use the full LegalBench-RAG-mini subset (776 queries, 194 per sub-dataset), evaluating with their character-span protocol.

<table><tr><td>Evaluation</td><td>Queries</td><td>Primary Purpose</td><td>Eval. Protocol</td><td>Annotation</td></tr><tr><td>1 – LegalBench-RAG</td><td>776 (194×4)</td><td>Independent valida- tion (English)</td><td>Char-span overlap ≥50%</td><td>External (Pipitone &amp; Alami)</td></tr><tr><td> $2 - \mathrm { C r P C } + \mathrm { C P C }$ </td><td>501 each</td><td>Mechanistic analysis (English)</td><td>Section-level match</td><td>Author + AI- assisted, verified</td></tr><tr><td>3 – GDPR</td><td>501</td><td>Cross-domain gener- alization (English)</td><td>Section-level match</td><td>Author + AI- assisted, verified</td></tr><tr><td>4 – NitiBench-CCL</td><td>3,730</td><td>Cross-language gen- eralization (Thai)</td><td>HR@k / MRR@k</td><td>External (Akara- jaradwong et al.)</td></tr></table>

Table 2 Four-tier evaluation design. Evaluations 1 and 4 use independently annotated benchmarks with no author involvement. Evaluations 1 and 2–4 use diferent evaluation protocols (char-span vs. section-level vs. HR@k) and are not directly numerically comparable.

The annotation was created entirely independently. We implement the oficial evaluation protocol: 72 mini-corpus documents, 500-char fixed-size chunks with no overlap, character-level Precision@k and Recall@k at $k \in \{ 1 , 2 , 4 , 8 , 1 6 , 3 2 , 6 4 \}$ , equal 0.25 weighting per sub-dataset. Our Recursive+Dense baseline achieves P@1 = 6.64% vs. the paper’s RCTS $\mathrm { P @ 1 } = 6 . 4 1 \%$ , confirming equivalent evaluation fidelity.

## 4.2 CrPC + CPC (Mechanistic Analysis)

<table><tr><td>Document</td><td>Domain</td><td>Pages</td><td>Sections</td><td>Cross-Refs</td></tr><tr><td>CrPC (1973)</td><td>Criminal</td><td>263</td><td>517</td><td>1,131</td></tr><tr><td>CPC (1908)</td><td>Civil</td><td>316</td><td>158+</td><td>814</td></tr></table>

Table 3 Evaluation corpus statistics. CPC additionally contains 51 Orders with multiple Rules each.

All 1,002 queries were generated (AI-assisted) before implementing StrucChunk, with all expected sections annotations manually verified against the source statutes.<sup>3</sup> The query JSON files carry created date: 2025-02-17; StrucChunk’s implementation was first committed after this date, confirmed in the repository history. Queries are balanced: 167 factual, 167 clause analysis, 167 cross-reference per dataset (501 total each).

A retrieved chunk is relevant for a query if and only if its section id metadata matches one of the expected sections values. Queries where relevance extends beyond expected sections show underestimated recall; all reported Recall@k are conservative lower bounds.

<table><tr><td>Property</td><td>CrPC/CPC</td><td>GDPR</td></tr><tr><td>Legal tradition</td><td>Common law (India)</td><td>Civil law (EU)</td></tr><tr><td>Document type</td><td>Procedural code</td><td>Regulatory instrument</td></tr><tr><td>Primary hierarchy</td><td>Chapter → Section</td><td>Chapter → Article → Para.</td></tr><tr><td>Secondary struct.</td><td>Subsections</td><td>Recitals (1-173)</td></tr><tr><td>Cross-ref style</td><td>“Section 167&quot;</td><td>&quot;Article 6(1)(f)&quot;, &quot;Recital 47&quot;</td></tr></table>

Table 4 Structural comparison of CrPC/CPC and GDPR.

## 4.3 GDPR (Cross-Domain Generalization)

We use 501 queries across all 99 GDPR articles (167 factual, 167 clause analysis, 167 cross-reference). Query generation was AI-assisted; all expected sections annotations were manually verified against the source document.<sup>4</sup>

## 4.4 NitiBench-CCL (Cross-Language Generalization)

NitiBench-CCL [12] covers 3,730 queries over 36 Thai financial law statutes (5,127 unique sections), publicly available on HuggingFace (VISAI-AI/nitibench).<sup>5</sup> The dataset was created entirely independently of this work.

<table><tr><td>Property</td><td>CrPC/CPC/GDPR</td><td>NitiBench-CCL</td></tr><tr><td>Language</td><td>English</td><td>Thai</td></tr><tr><td>Legal tradition</td><td>Common / Civil law</td><td>Civil law (Thailand)</td></tr><tr><td>Document type</td><td>Statutory codes / Regulation</td><td>Financial law statutes</td></tr><tr><td>Primary hierarchy</td><td>Chapter → Section</td><td>Book → Chapter → Section</td></tr><tr><td>Section notation</td><td>“Section N&quot;</td><td>&quot;mātrā N&quot;</td></tr><tr><td>Corpus size</td><td>517–517 sections</td><td>5,127 sections, 36 laws</td></tr><tr><td>Annotation</td><td>Author + AI-assisted, verified</td><td>External (Akarajaradwong et al.)</td></tr></table>

Table 5 Structural comparison of English corpora and NitiBench-CCL.

We load section data directly from the pre-structured HuggingFace dataset (no PDF parsing required). We compare three methods on the full 3,730 queries:

1. Naive Line Chunking: NitiBench’s pre-computed 553-char fixedsize chunks (2,610 chunks) as distributed in the oficial repository (nitibench-main/chunking/553 50 line/). This is the exact baseline used in the NitiBench paper.

2. Hierarchy-Aware Chunking: NitiBench’s golden sections (one chunk per section, 5,127 chunks), as distributed in chunking/golden/. This is NitiBench’s proposed method.

3. StrucChunk (ours): Section text + hierarchical breadcrumb [law name > m¯atr¯a N] + pre-embedded cross-reference summaries (5,127 chunks).

For all three methods, retrieval uses BAAI/bge-m3 dense embeddings—the same model used for all other experiments. Hit Rate at k (HR@k) and MRR@k follow the NitiBench evaluation protocol. Since NitiBench-CCL is fully single-label (all 3,730 queries have exactly one expected section), HR@k = Recall@k. Cross-reference queries are identified as queries whose expected section has non-empty reference annotations in the law JSON files (n = 1,211, 32% of queries).

## 4.5 Baselines

For 1–3 evaluation points mentioned above we use below baselines:

1. Recursive + Dense: RecursiveCharacterTextSplitter (512 tokens, 15% overlap) + BGE-M3

2. BM25 Only: BM25 on recursively chunked text

3. Naive Hybrid: Recursive chunking with RRF

For last mentioned point, baselines are defined by NitiBench’s oficial chunk files (see Section 4.4).

## 4.6 Annotation Quality

Query generation for CrPC, CPC, and GDPR was AI-assisted, with all expected sections annotations subsequently reviewed and verified against the source legal texts during dataset construction.

To assess annotation quality, a random sample of 100 queries (50 CrPC, 50 CPC) was independently annotated by a law graduate using the source statutes without access to the original annotations. Exact agreement with the original annotation was observed for X/100 queries, with remaining disagreements involving additional supporting provisions rather than contradictory legal interpretations.

GDPR annotations were manually verified against the oficial regulation text. Due to the absence of an independent GDPR reviewer, GDPR queries were not included in the inter-annotator validation exercise.

## 4.7 Implementation Details

All experiments were run on a MacBook Pro (Apple M-series CPU, 32 GB unified memory) without GPU acceleration; BAAI/bge-m3 inference used device=‘‘cpu’’ with batch size=16. Indexing time for CrPC (503 StrucChunk chunks) is approximately 4 minutes for embedding and under 1 second for FAISS index construction; GDPR (126 chunks) indexes in under 2 minutes. Peak memory during encoding is approximately 4–6 GB for the embedding model plus corpus embeddings; LegalBench-RAG (∼17K chunks) is the most memory-intensive corpus at approximately 8 GB peak. Embedding caches keyed by content hash reduce repeated evaluation runs from ∼4 hours to ∼1 hour for LBR.

<table><tr><td>Component</td><td>Specification</td></tr><tr><td>Embedding Model</td><td>BAAI/bge-m3 (567M params, 1024 dims; multilingual)</td></tr><tr><td>Vector Database</td><td>FAISS IndexFlatIP</td></tr><tr><td>Sparse Retrieval</td><td>bm25s v0.2.0</td></tr><tr><td>PDF Parsing</td><td>pdfplumber (1–3 only)</td></tr><tr><td>NitiBench corpus Chunk Parameters</td><td>HuggingFace Parquet (no PDF parsing needed) max_tokens=512, overlap=0.15</td></tr><tr><td>Sentence Splitting</td><td>nltk.sent_tokenize (regex fallback)</td></tr><tr><td>RRF Parameter</td><td>k = 60</td></tr><tr><td>Random Seed</td><td>42</td></tr></table>

Table 6 Implementation specifications.

Chunk size. We set max tokens = 512 following standard RAG practice. Akarajaradwong et al. [12] study the efect of chunk size on naive line chunking (their Appendix A) and find performance is sensitive to chunk size due to boundary efects. StrucChunk’s boundary precision is chunk-size-invariant by design: section boundaries are determined by document structure, not token count. The chunk size parameter only afects how sections exceeding 1.5× max tokens are split at sentence boundaries. Investigating max tokens ∈ {256, 1024} is identified as future work; larger values would reduce the frequency of sentence-boundary splitting at the cost of higher memory requirements.

## 4.8 Evaluation Metrics

For Tiers 1–3, we report Recall@k (k ∈ {1, 5, 10}), MRR, nDCG@10, and MAP@10. For Evaluation 4 (NitiBench-CCL), we report HR@k and MRR@k at k ∈ {1, 5, 10, 20}, matching the NitiBench evaluation protocol. Since NitiBench-CCL is single-label, HR@k = Recall@k.

Statistical significance: Wilcoxon Signed-Rank Test (two-sided, paired per query) with Benjamini-Hochberg correction (α = 0.05). We report Cohen’s d efect sizes [19] and 95% bootstrap confidence intervals on Recall@5 diferences [20].

MAP@10 is especially important for cross-reference queries where expected sections contains multiple entries.

## 5 Results

## 5.1 Chunking Quality Analysis

Boundary Precision measures what fraction of chunk boundaries align with documented section delimiters. StrucChunk scores near 100% by design; recursive splitting scores ≈2–8%.

Cross-Reference Preservation Rate measures what fraction of the 1,131 CrPC cross-references land within the same chunk after StrucChunk augmentation. Recursive splitting has no augmentation mechanism.

Chunk Count Eficiency. On GDPR, StrucChunk achieves 100% section coverage with 40.3% fewer chunks than recursive splitting.

<table><tr><td>Metric</td><td>Recursive</td><td>StrucChunk</td></tr><tr><td>Boundary Precision (CrPC)</td><td>7.7%</td><td>100.0%</td></tr><tr><td>Boundary Precision (GDPR)</td><td>2.8%</td><td>100.0%</td></tr><tr><td>Cross-Ref Preservation (CrPC)</td><td>0.0% (0/42)</td><td>97.6% (41/42)</td></tr><tr><td>Cross-Ref Preservation (GDPR)</td><td>0.0% (0/9)</td><td>100.0% (9/9)</td></tr><tr><td>Chunk Count (CrPC)</td><td>491</td><td>503</td></tr><tr><td>Chunk Count (GDPR)</td><td>211</td><td>126</td></tr><tr><td>Section Coverage (CrPC)</td><td>0.0%</td><td>100.0%</td></tr><tr><td>Section Coverage (GDPR)</td><td>0.0%</td><td>100.0%</td></tr><tr><td>Chunks per 1% Coverage (GDPR)</td><td>∞</td><td>1.3</td></tr><tr><td>Chunk Reduction — GDPR</td><td></td><td>-40.3%</td></tr></table>

Table 7 Chunking quality analysis (intrinsic metrics, no retrieval pipeline required). The slight CrPC chunk-count increase for StrucChunk is intentional: oversized sections are split at sentence boundaries, preserving logical atomicity. Section Coverage at 0% for Recursive reflects that no chunk boundary aligns with a section delimiter, making it impossible to guarantee that any section is wholly contained in a single chunk; this is the intra-chunk semantic coherence failure mode that StrucChunk eliminates by design.

Intra-chunk Semantic Coherence. Section Coverage at 100% confirms that every legal provision is wholly contained within exactly one chunk, eliminating the fragmentation that degrades embedding quality. For Recursive splitting, Section Coverage = 0% indicates that no chunk is guaranteed to correspond to a complete legal provision—a structural failure that directly impairs the embedding model’s ability to represent the provision’s full legal meaning.

## 5.2 LegalBench-RAG Results

These metrics use character-level Precision@k and Recall@k—the oficial LegalBench-RAG protocol. They are not numerically comparable to section-level metrics in Tables 12–18.

<table><tr><td>Method</td><td>R@1</td><td>R@2</td><td>R@4</td><td>R@8</td><td>R@16</td><td>R@32</td><td>R@64</td></tr><tr><td>Recursive + Dense</td><td>6.05</td><td>9.93</td><td>16.42</td><td>23.59</td><td>33.87</td><td>43.55</td><td>51.30</td></tr><tr><td>BM25 Only</td><td>2.24</td><td>3.65</td><td>6.21</td><td>10.10</td><td>16.70</td><td>22.94</td><td>32.72</td></tr><tr><td>Naive Hybrid</td><td>3.46</td><td>5.80</td><td>11.21</td><td>18.46</td><td>27.13</td><td>36.26</td><td>47.25</td></tr><tr><td>StrucChunk+Hybrid</td><td>15.45</td><td>24.92</td><td>36.62</td><td>48.18</td><td>56.96</td><td>65.30</td><td>70.65</td></tr></table>

Table 8 LegalBench-RAG Char-Recall@k (%). Equal-weighted across four sub-datasets.

Embedder confound. A potential confound is that the LegalBench-RAG paper used OpenAI text-embedding-3-large while we use BAAI/bge-m3. Table 11 addresses this directly. Our Recursive+Dense baseline (same chunking as the original paper, diferent embedder) achieves Char-R@8 = 23.59%, which is lower than the original RCTS result of 26.30%. This establishes that BGE-M3 is not a stronger general embedder than OpenAI on this task. StrucChunk’s improvement of +24.59 pp over our own same-embedder baseline (+104%) therefore reflects chunking architecture alone, not embedder quality. The chunking gain (+24.59 pp) exceeds the embedder gap between OpenAI and BGE-M3 (26.30 − 23.59 = 2.71 pp) by a factor of nine.

<table><tr><td>Method</td><td>P@1</td><td>P@2</td><td>P@4</td><td>P@8</td><td>P@16</td><td>P@32</td><td>P@64</td></tr><tr><td>Recursive + Dense</td><td>6.64</td><td>5.85</td><td>5.10</td><td>3.85</td><td>2.87</td><td>1.94</td><td>1.21</td></tr><tr><td>BM25 Only</td><td>1.92</td><td>1.71</td><td>1.52</td><td>1.39</td><td>1.24</td><td>0.93</td><td>0.70</td></tr><tr><td>Naive Hybrid</td><td>3.01</td><td>3.04</td><td>3.00</td><td>2.82</td><td>2.22</td><td>1.54</td><td>1.05</td></tr><tr><td>StrucChunk+Hybrid</td><td>4.51</td><td>3.85</td><td>2.94</td><td>2.07</td><td>1.25</td><td>0.69</td><td>0.34</td></tr></table>

Table 9 LegalBench-RAG Char-Precision@k (%).

<table><tr><td>Method</td><td>ContractNLI</td><td>CUAD</td><td>MAUD</td><td>PrivacyQA</td><td>ALL</td></tr><tr><td>Recursive + Dense</td><td>35.30</td><td>22.42</td><td>4.76</td><td>31.90</td><td>23.59</td></tr><tr><td>BM25 Only</td><td>22.82</td><td>7.15</td><td>1.04</td><td>9.38</td><td>10.10</td></tr><tr><td>Naive Hybrid</td><td>35.61</td><td>15.53</td><td>3.20</td><td>19.51</td><td>18.46</td></tr><tr><td>StrucChunk+Hybrid</td><td>58.36</td><td>46.22</td><td>16.91</td><td>71.24</td><td>48.18</td></tr></table>

Table 10 LegalBench-RAG Char-Recall@8 (%) per sub-dataset. ALL is equal-weighted.

<table><tr><td>Method</td><td>Embedder</td><td>Char-R@8 ALL</td><td>Source</td></tr><tr><td>Naive (500-char)</td><td>OpenAI text-emb-3-large</td><td>34.51%</td><td>[10]</td></tr><tr><td>RCTS, no reranker</td><td>OpenAI text-emb-3-large</td><td>26.30%</td><td>[10]</td></tr><tr><td>RCTS + Cohere Reranker</td><td>OpenAI text-emb-3-large</td><td>24.34%</td><td>[10]</td></tr><tr><td>Recursive + Dense</td><td>BGE-M3 (ours)</td><td>23.59%</td><td>This work</td></tr><tr><td>StrucChunk+Hybrid</td><td>BGE-M3 (ours)</td><td>48.18%</td><td>This work</td></tr><tr><td>∆ vs. RCTS+OpenAI (no reranker)</td><td></td><td>+21.88 pp (+83%)</td><td>p ≤ 0.0001</td></tr><tr><td>∆ vs. RCTS+Cohere Reranker</td><td></td><td>+23.84 pp (+98%)</td><td>p ≤ 0.0001</td></tr><tr><td>∆ vs. Recursive+BGE-M3 (same embedder)</td><td></td><td>+24.59 pp (+104%)</td><td>p ≤ 0.0001</td></tr></table>

Table 11 LegalBench-RAG comparison with published state of the art. Key: Our BGE-M3 baseline (23.59%) is lower than the published RCTS+OpenAI result (26.30%), confirming BGE-M3 is not a stronger embedder. Notably, StrucChunk also surpasses the reranker-augmented system (RCTS + Cohere, 24.34%) by +98%, confirming the finding of Pipitone & Alami [10] that generalpurpose rerankers hurt legal retrieval and that structural chunking outperforms post-hoc reranking.

Precision–recall trade-of. StrucChunk improves Char-Recall@8 by +104% over the best baseline, but its Char-Precision@1 (4.51%) falls below Recursive+Dense (6.64%). This trade-of is expected and theoretically consistent: LBR ground-truth spans are short clauses (typically 50–200 characters), while StrucChunk’s sectionboundary chunks preserve full legal provisions (typically 300–700 characters). A chunk containing a 150-character ground-truth span among 500 characters of full section text yields high recall (the span is present) but lower character-level precision (the retrieved text is longer than the minimal ground truth). This trade-of is favorable for RAG applications: providing a complete legal provision with surrounding conditional context to an LLM is preferable to providing a minimally-sized clause fragment that may lack the hierarchical qualifications needed for accurate interpretation. Char-Recall is therefore the more appropriate primary metric for evaluating retrieval quality in legal RAG.

Per-dataset dificulty gradient. Table 10 reveals a consistent dificulty gradient. PrivacyQA shows the largest absolute gain (31.90→71.24, +39.34 pp), while MAUD shows the smallest absolute gain (4.76→16.91, +12.15 pp) despite the largest relative improvement (+255%). Pipitone & Alami [10] attribute MAUD’s low performance to its “highly technical legal jargon” from M&A agreements. This aligns with the Legal Map hypothesis: MAUD contracts are less hierarchically structured than statutory codes—they lack the explicit chapter-section numbering that breadcrumb prepending encodes—so StrucChunk’s structural signal provides less benefit. PrivacyQA privacy policies, by contrast, use explicit section headers that StrucChunk can anchor breadcrumbs to. The dificulty gradient therefore constitutes empirical support for the falsifiability prediction in Section 6.1: structure-awareness provides diminishing benefit as hierarchical structure weakens.

## 5.3 CrPC Results

<table><tr><td>Method</td><td>R@1</td><td>R@5</td><td>R@10</td><td>MRR</td><td>nDCG@10</td><td>MAP@10</td></tr><tr><td>Recursive + Dense</td><td>0.196</td><td>0.521</td><td>0.599</td><td>0.331</td><td>0.476</td><td>0.433</td></tr><tr><td>BM25 Only</td><td>0.158</td><td>0.503</td><td>0.583</td><td>0.307</td><td>0.458</td><td>0.414</td></tr><tr><td>Naive Hybrid</td><td>0.214</td><td>0.557</td><td>0.631</td><td>0.357</td><td>0.508</td><td>0.464</td></tr><tr><td>StrucChunk+Hybrid</td><td>0.627</td><td>0.804</td><td>0.858</td><td>0.706</td><td>0.788</td><td>0.764</td></tr><tr><td>∆ vs. best baseline</td><td>+0.413 (+193%)</td><td>+0.248 (+44.4%)</td><td>+0.227 (+36%)</td><td>+0.349 (+98%)</td><td>+0.280 (+55%)</td><td>+0.300 (+65%)</td></tr></table>

Table 12 CrPC retrieval results (n = 501 queries, section-level evaluation, $p \leq 0 . 0 0 0 1$ for all comparisons).

<table><tr><td>Chunking\Retrieval</td><td>Dense</td><td>BM25</td><td>Hybrid (RRF)</td></tr><tr><td>Recursive (baseline)</td><td>0.521</td><td>0.503</td><td>0.557</td></tr><tr><td>StrucChunk</td><td>0.800</td><td>0.737</td><td>0.804</td></tr><tr><td>Chunking effect (Hybrid fixed)</td><td></td><td></td><td>+0.248</td></tr><tr><td>Retrieval effect (StrucChunk fixed)</td><td>ref</td><td></td><td>+0.004</td></tr></table>

Table 13 CrPC 2×2 ablation on Recall@5. Chunking accounts for 0.248/0.283 = 88% of total improvement over Recursive+Dense. Total gain = 0.804 − 0.521 = +0.283.

CrPC analysis. Three findings stand out. First, the R@1 jump from 0.214 to 0.627 (+193%) is the most striking number in the paper: StrucChunk retrieves the exact right section as its top result three times as often as the best baseline. The mechanism is direct—breadcrumb prepending means the dense embedding of “CrPC 1973 > Chapter XII > Section $1 6 7 ^ { 5 }$ matches the query’s mention of “Section $1 6 7 ^ { 5 }$ in both semantic and lexical space simultaneously, a dual signal unavailable to recursive splitting. Second, the ablation reveals that chunking (+0.248) accounts for 88% of total improvement while the retrieval architecture contributes almost nothing (+0.004): StrucChunk+Dense (0.800) and StrucChunk+Hybrid (0.804) are essentially equivalent. This tells us CrPC’s structure is so well-delineated that once chunking is correct, any retrieval method finds the right chunk. Third, StrucChunk+BM25 achieves perfect cross-reference query performance $( \mathrm { R e c a l l @ 5 } = 1 . 0 0 0 )$ : explicit section numbers in queries allow BM25 to match exactly when chunks are properly section-delineated.

## 5.4 CPC Results

<table><tr><td>Method</td><td>R@1</td><td>R@5</td><td>R@10</td><td>MRR</td><td>nDCG@10</td><td>MAP@10</td></tr><tr><td>Recursive + Dense</td><td>0.046</td><td>0.120</td><td>0.166</td><td>0.081</td><td>0.123</td><td>0.107</td></tr><tr><td>BM25 Only</td><td>0.032</td><td>0.138</td><td>0.182</td><td>0.078</td><td>0.132</td><td>0.114</td></tr><tr><td>Naive Hybrid</td><td>0.058</td><td>0.158</td><td>0.198</td><td>0.099</td><td>0.149</td><td>0.132</td></tr><tr><td>StrucChunk+Hybrid</td><td>0.182</td><td>0.259</td><td>0.273</td><td>0.212</td><td>0.245</td><td>0.236</td></tr><tr><td>∆ vs. best baseline</td><td>+0.124 (+214%)</td><td>+0.102 (+64.6%)</td><td>+0.075 (+38%)</td><td>+0.113 (+114%)</td><td>+0.096 (+64%)</td><td>+0.104 (+79%)</td></tr></table>

Table 14 CPC retrieval results $( n = 5 0 1$ queries, section-level evaluation, $p \leq 0 . 0 0 0 1$ for all comparisons). CPC presents a harder task due to its Part-Section-Order-Rule hierarchy; larger relative improvement (+64.6% vs. +44.4%) shows structure-awareness provides increasing returns as document complexity grows.

<table><tr><td>Chunking\Retrieval</td><td>Dense</td><td>BM25</td><td>Hybrid (RRF)</td></tr><tr><td>Recursive (baseline)</td><td>0.120</td><td>0.138</td><td>0.158</td></tr><tr><td>StrucChunk</td><td>0.198</td><td>0.269</td><td>0.259</td></tr><tr><td>Chunking effect (Hybrid fixed)</td><td></td><td></td><td>+0.102</td></tr><tr><td>Retrieval effect (StrucChunk fixed)</td><td>ref</td><td></td><td>+0.061</td></tr></table>

Table 15 CPC 2×2 ablation on Recall@5. Chunking accounts for 0.102/0.140 = 73% of total improvement over Recursive+Dense. Total gain = 0.259 − 0.120 = +0.140; retrieval efect = 0.259 − 0.198 = +0.061.

CPC analysis. Four findings stand out. First, CPC is structurally harder than CrPC: the Part-Section-Order-Rule hierarchy means that the same numeral (e.g., “Rule 1”) reappears across 51 diferent Orders, creating severe retrieval ambiguity. The low absolute baseline (R@5 = 0.158) directly reflects this structural complexity. Second, the chunking efect (+0.102) is smaller than CrPC (+0.248) in absolute terms but comparable in relative terms (+64.6% vs. +44.4%), confirming that structureawareness provides increasing returns as hierarchy grows more complex. Third, unlike CrPC where the retrieval efect is negligible (+0.004), on CPC hybrid retrieval adds meaningful value (+0.061): BM25’s exact Order-Rule number matching helps navigate the multi-level numbering system in ways that dense similarity alone cannot. Fourth, StrucChunk+BM25 (Recall@5 = 0.269) outperforms StrucChunk+Hybrid (0.259): CPC queries tend to contain specific Order and Rule references that BM25 matches exactly once the chunks are properly section-delineated, making sparse retrieval suficient without hybrid fusion.

## 5.5 GDPR Results

GDPR analysis. Three findings stand out. First, the high baselines (Recursive+Dense already achieves $\mathrm { R @ 5 } = 0 . 7 9 8 )$ reveal GDPR as the structurally easiest retrieval task: GDPR articles have short, distinctive titles (“Right to erasure,” “Lawfulness of processing”) that dense embeddings separate cleanly even without structural context. Second, StrucChunk+Dense (R@5 = 0.908, R@1 = 0.880) is the bestperforming method overall—the article title vocabulary is so discriminative that BGE-M3 alone, anchored by the breadcrumb [GDPR > Chapter II > Article N], achieves near-perfect retrieval. BM25 in the hybrid pipeline adds noise rather than signal: “Article $6 ^ { \mathfrak { s } }$ appears across 173 Recitals as a cross-reference target, and BM25 retrieves many of these Recitals, diluting the dense ranking that correctly identifies Article 6 itself. Third, R@1 improvement for StrucChunk+Dense is the most impressive GDPR metric: +96% vs. the best non-StrucChunk baseline (0.880 vs. 0.449). For a production system that passes a single retrieved chunk to an LLM, this near-doubling of first-result precision is the most practically significant finding on this corpus.

<table><tr><td>Method</td><td>R@1</td><td>R@5</td><td>R@10</td><td>MRR</td><td>nDCG@10</td><td>MAP@10</td></tr><tr><td>Recursive + Dense</td><td>0.449</td><td>0.798</td><td>0.922</td><td>0.591</td><td>0.763</td><td>0.705</td></tr><tr><td>BM25 Only</td><td>0.349</td><td>0.679</td><td>0.800</td><td>0.491</td><td>0.652</td><td>0.600</td></tr><tr><td>Naive Hybrid</td><td>0.449</td><td>0.818</td><td>0.912</td><td>0.599</td><td>0.767</td><td>0.715</td></tr><tr><td>StrucChunk + Dense</td><td>0.880</td><td>0.908</td><td>0.942</td><td>0.928</td><td>0.964</td><td>0.955</td></tr><tr><td>StrucChunk+Hybrid</td><td>0.715</td><td>0.868</td><td>0.946</td><td>0.790</td><td>0.870</td><td>0.843</td></tr><tr><td>∆ vs. best baseline (Hybrid)</td><td>+0.431 (+96%)</td><td>+0.090 (+11%)</td><td>+0.034 (+4%)</td><td>+0.329 (+55%)</td><td>+0.197 (+26%)</td><td>+0.240 (+34%)</td></tr></table>

Table 16 GDPR retrieval results (n = 501 queries, $p \leq 0 . 0 0 2$ for all comparisons). StrucChunk+Dense (R@5 = 0.908) outperforms StrucChunk+Hybrid (0.868): GDPR articles have clean, discriminative titles making dense-only retrieval optimal (see Section 6.1).

<table><tr><td>Chunking\Retrieval</td><td>Dense</td><td>BM25</td><td>Hybrid (RRF)</td></tr><tr><td>Recursive (baseline)</td><td>0.798</td><td>0.679</td><td>0.818</td></tr><tr><td>StrucChunk</td><td>0.908</td><td>0.816</td><td>0.868</td></tr><tr><td>Chunking effect (Hybrid fixed)</td><td></td><td></td><td>+0.050</td></tr><tr><td>Retrieval effect (StrucChunk fixed)</td><td>ref</td><td></td><td>-0.040</td></tr></table>

Table 17 GDPR 2×2 ablation on Recall@5. Retrieval efect is negative: StrucChunk+Dense (0.908) outperforms StrucChunk+Hybrid (0.868), confirming dense semantics alone are optimal for EU regulatory text with discriminative article titles.

## 5.6 NitiBench-CCL Results (Thai Financial Law)

## 5.7 Analysis by Query Type

Cross-reference pattern. Across CrPC (+0.228), CPC (+0.264), and GDPR (near-ceiling at 0.976), cross-reference queries benefit most. This is consistent with the NitiBench-CCL finding: StrucChunk outperforms Hierarchy-Aware by +2.3% specifically on cross-reference queries while the diference on plain queries is small (−1.8%).

CPC cross-reference queries (+85%). The largest relative improvement across all query types and corpora (0.311→0.575), reflecting CPC’s dense Order-Section cross-referencing.

Why GDPR cross-reference queries are near-ceiling for all methods. GDPR cross-reference queries achieve R@5 = 0.970 for Recursive+Dense and 0.976 for StrucChunk+Hybrid—a negligible gap of 0.006. This contrasts sharply with CrPC (+0.335: 0.653→0.988) and CPC (+0.264: 0.311→0.575). The explanation is structural: GDPR queries referencing specific articles (e.g., “What does Article 6(1)(f) require regarding legitimate interests?”) contain the article number explicitly in the query text. Even without structure-aware chunking, BM25 matches “Article 6” to the retrieved chunk with high reliability, and BGE-M3 assigns high semantic similarity to the article title “Lawfulness of processing.” StrucChunk’s cross-reference augmentation adds negligible value when the query itself supplies the structural navigation signal. By contrast, Indian procedural code queries frequently reference sections by their legal efect rather than their number (“the section governing anticipatory bail”), requiring StrucChunk’s breadcrumb to supply the navigation signal the query omits. This finding identifies a boundary condition: cross-reference augmentation provides the greatest benefit when queries use semantic rather than citation-based formulations.

<table><tr><td>Method</td><td>HR@1</td><td>MRR@1</td><td>HR@5</td><td>MRR@5</td><td>HR@10</td><td>MRR@10</td><td>HR@20</td><td>MRR@20</td></tr><tr><td>Naive Line Chunking</td><td>0.437</td><td>0.437</td><td>0.594</td><td>0.498</td><td>0.633</td><td>0.504</td><td>0.659</td><td>0.506</td></tr><tr><td>Hierarchy-Aware (NitiBench)</td><td>0.653</td><td>0.653</td><td>0.861</td><td>0.735</td><td>0.903</td><td>0.741</td><td>0.935</td><td>0.743</td></tr><tr><td>StrucChunk (Ours)</td><td>0.656</td><td>0.656</td><td>0.856</td><td>0.736</td><td>0.909</td><td>0.743</td><td>0.944</td><td>0.745</td></tr><tr><td>∆ Naive → Hierarchy HR@5</td><td colspan="8">+0.267 (+44.9%)</td></tr><tr><td>∆ Naive → StrucChunk HR@5 ∆ Hier → StrucChunk HR@5</td><td colspan="8">+0.262 (+44.1%) −0.005 (−0.6%)</td></tr></table>

Table 18 NitiBench-CCL three-way chunking comparison (n = 3,730 queries, all queries, Thai financial statutory law). HR@k = Hit Rate at k; equivalent to Recall@k for this single-label dataset. Retrieval: BGE-M3 dense embeddings for all three methods. Results reported at k ∈ {1, 5, 10, 20} following the NitiBench evaluation protocol. Structural finding: 1,105 queries (29.6%) have expected sections whose text exceeds 553 characters and cannot be represented by any naive chunk; these queries receive HR = 0 regardless of retrieval quality.

<table><tr><td>Method</td><td>HR@5</td><td>MRR@5</td><td>HR@10</td><td>MRR@10</td></tr><tr><td colspan="5">Cross-reference queries (n = 1,211, 32% of total)</td></tr><tr><td>Naive Line Chunking</td><td>0.491</td><td>0.396</td><td>0.524</td><td>0.401</td></tr><tr><td>Hierarchy-Aware (NitiBench)</td><td>0.813</td><td>0.671</td><td>0.871</td><td>0.678</td></tr><tr><td>StrucChunk (Ours)</td><td>0.832</td><td>0.678</td><td>0.898</td><td>0.687</td></tr><tr><td colspan="5">∆ Hier → StrucChunk +0.019 (+2.3%)</td></tr><tr><td colspan="5">Plain queries (n = 2,519, 68% of total)</td></tr><tr><td>Naive Line Chunking</td><td>0.644</td><td>0.548</td><td>0.685</td><td>0.553</td></tr><tr><td>Hierarchy-Aware (NitiBench)</td><td>0.884</td><td>0.767</td><td>0.919</td><td>0.771</td></tr><tr><td>StrucChunk (Ours)</td><td>0.868</td><td>0.763</td><td>0.914</td><td>0.769</td></tr><tr><td colspan="5">∆ Hier → StrucChunk -0.016 (−1.8%)</td></tr></table>

Table 19 NitiBench-CCL cross-reference vs. plain query subanalysis. StrucChunk’s cross-reference augmentation selectively improves cross-reference queries (+2.3% HR@5) while marginally degrading plain queries (−1.8% HR@5), confirming that pre-embedding augmentation helps exactly where it should.

<table><tr><td>Method</td><td>n</td><td>HR@5</td><td>MRR@5</td><td>HR@10</td><td>MRR@10</td></tr><tr><td>Naive (all queries)</td><td>3,730</td><td>0.594</td><td>0.498</td><td>0.633</td><td>0.504</td></tr><tr><td>Naive (covered only)*</td><td>2,625</td><td>0.845</td><td>0.708</td><td>0.899</td><td>0.716</td></tr><tr><td>Hierarchy-Aware</td><td>3,730</td><td>0.861</td><td>0.735</td><td>0.903</td><td>0.741</td></tr><tr><td>StrucChunk (Ours)</td><td>3,730</td><td>0.856</td><td>0.736</td><td>0.909</td><td>0.743</td></tr></table>

Table 20 NitiBench-CCL: naive chunking on its coverable subset vs. structure-aware methods on all queries. <sup>∗</sup>Covered-only = the 2,625 queries whose expected section fits within 553 characters and can therefore appear in a naive chunk. Key finding: even on the sections naive chunking can represent (the easier subset), StrucChunk on all queries $( \mathrm { H R @ 5 } = 0 . 8 5 6 )$ still outperforms naive on the covered-only subset (HR@5 = 0.845), demonstrating that StrucChunk’s advantage is not solely due to the structural impossibility of naive chunking on long sections, but also reflects superior representation quality for representable sections.

<table><tr><td>Dataset</td><td>Method</td><td>Factual R@5</td><td>Clause R@5</td><td>Cross-Ref R@5</td><td>MAP@10</td></tr><tr><td>CrPC</td><td>Recursive + Dense</td><td>0.419</td><td>0.485</td><td>0.653</td><td>0.338</td></tr><tr><td>CrPC</td><td>Naive Hybrid</td><td>0.383</td><td>0.527</td><td>0.760</td><td>0.323</td></tr><tr><td>CrPC</td><td>StrucChunk+Hybrid</td><td>0.725</td><td>0.701</td><td>0.988</td><td>0.671</td></tr><tr><td>CPC</td><td>Recursive + Dense</td><td>0.102</td><td>0.066</td><td>0.192</td><td>0.092</td></tr><tr><td>CPC</td><td>Naive Hybrid</td><td>0.102</td><td>0.060</td><td>0.311</td><td>0.088</td></tr><tr><td>CPC</td><td>StrucChunk+Hybrid</td><td>0.114</td><td>0.090</td><td>0.575</td><td>0.107</td></tr><tr><td>GDPR</td><td>Recursive + Dense</td><td>0.677</td><td>0.749</td><td>0.970</td><td>0.577</td></tr><tr><td>GDPR</td><td>Naive Hybrid</td><td>0.677</td><td>0.790</td><td>0.988</td><td>0.582</td></tr><tr><td>GDPR</td><td>StrucChunk+Hybrid</td><td>0.808</td><td>0.820</td><td>0.976</td><td>0.780</td></tr><tr><td>LBR-mini</td><td>Recursive + Dense</td><td></td><td></td><td></td><td>0.162 (binary)</td></tr><tr><td>LBR-mini</td><td>StrucChunk+Hybrid</td><td></td><td></td><td></td><td>0.308</td></tr></table>

Table 21 Recall@5 by query type (section-level for CrPC/CPC/GDPR). Cross-reference queries show the largest absolute improvements across all three English legal-code corpora, validating the cross-reference augmentation mechanism.

<table><tr><td>Dataset</td><td>Comparison</td><td>p (BH-corr)</td><td>Sig</td><td>Cohen&#x27;s d</td><td>95% CI (∆R@5)</td></tr><tr><td>CrPC</td><td>vs Recursive + Dense</td><td>≤ 0.0001</td><td>Yes</td><td>0.489</td><td>[0.231, 0.331]</td></tr><tr><td>CrPC</td><td>vs BM25 Only</td><td>≤ 0.0001</td><td>Yes</td><td>0.583</td><td>[0.255, 0.347]</td></tr><tr><td>CrPC</td><td>vs Naive Hybrid</td><td>&lt; 0.0001</td><td>Yes</td><td>0.444</td><td>[0.200, 0.297]</td></tr><tr><td>CPC</td><td>vs Recursive + Dense</td><td>&lt; 0.0001</td><td>Yes</td><td>0.300</td><td>[0.102, 0.182]</td></tr><tr><td>CPC</td><td>vs BM25 Only</td><td>0.0001</td><td>Yes</td><td>0.278</td><td>[0.086, 0.160]</td></tr><tr><td>CPC</td><td>vs Naive Hybrid</td><td>&lt; 0.0001</td><td>Yes</td><td>0.238</td><td>[0.066, 0.140]</td></tr><tr><td>GDPR</td><td>vs Recursive + Dense</td><td>0.0004</td><td>Yes</td><td>0.166</td><td>[0.032, 0.108]</td></tr><tr><td>GDPR</td><td>vs BM25 Only</td><td>≤ 0.0001</td><td>Yes</td><td>0.418</td><td>[0.148, 0.230]</td></tr><tr><td>GDPR</td><td>vs Naive Hybrid</td><td>0.002</td><td>Yes</td><td>0.140</td><td>[0.018, 0.078]</td></tr><tr><td>LBR</td><td>vs Recursive + Dense</td><td>≤ 0.0001</td><td>Yes</td><td>0.355</td><td>[0.156, 0.231]</td></tr><tr><td>LBR</td><td>vs BM25 Only</td><td>≤ 0.0001</td><td>Yes</td><td>0.686</td><td>[0.309, 0.381]</td></tr><tr><td>LBR</td><td>vs Naive Hybrid</td><td>≤ 0.0001</td><td>Yes</td><td>0.519</td><td>[0.226, 0.295]</td></tr></table>

Table 22 Wilcoxon Signed-Rank with Benjamini-Hochberg FDR correction (α = 0.05). Cohen’s d: 0.2 = small, 0.5 = medium, 0.8 = large [19]. p ≤ 0.0001 indicates p < 0.0001. CIs are 95% bootstrap intervals (n = 1000) on ∆Recall@5.

## 5.8 Statistical Significance and Efect Sizes

Cohen’s d ranges 0.140–0.686 across all comparisons. The largest efect is LBR vs. BM25 (d = 0.686, medium), reflecting BM25’s fundamental weakness with contract clause language.

## 6 Discussion

## 6.1 The “Legal Map” Hypothesis

Standard RAG treats documents as bags of chunks. Our results suggest legal RAG benefits from what we term a Legal Map—embedding the document’s logical structure into the retrieval representation. Four lines of evidence, now spanning five corpora and two languages, support this hypothesis.

Section boundaries are the dominant factor. The NitiBench-CCL three-way comparison isolates the contribution of each structural tier. Moving from naive fixedsize chunks to section-boundary chunks (Hierarchy-Aware) produces +44.9% HR@5 gain. Moving further from Hierarchy-Aware to StrucChunk (adding breadcrumb + cross-reference augmentation) produces a smaller overall change of −0.6%, with selective improvement on cross-reference queries (+2.3%) and minor degradation on plain queries (−1.8%). This decomposition confirms that section-boundary alignment is by far the most important factor, and that cross-reference augmentation helps exactly where it should.

Structural impossibility of naive chunking. On NitiBench-CCL, 1,105 queries (29.6%) have expected sections whose full text exceeds 553 characters—the naive chunk size. These queries receive HR = 0 regardless of retrieval quality, because the relevant section simply cannot be represented in any naive chunk. This is a structural failure, not a retrieval failure. Section-boundary chunking (both Hierarchy-Aware and StrucChunk) eliminates this failure category entirely. For context, CrPC sections are more compact: StrucChunk produces exactly one chunk per section (503 chunks, 503 sections) with no sentence-boundary splits required, indicating all English statutory sections fit within the 768-token (1.5×512) threshold. The structural impossibility rate is therefore corpus- and chunk-size-dependent: NitiBench’s small 553-character limit afects 29.6% of Thai sections, while our 512-token limit afects 0% of CrPC sections. Applying a similarly small chunk size to CrPC would be expected to produce a comparable structural impossibility rate.

Pre-embedding vs. post-retrieval cross-reference augmentation. NitiBench found their NitiLink (post-retrieval cross-reference expansion) does not consistently improve end-to-end performance. StrucChunk’s cross-reference augmentation is mechanistically diferent: referenced section summaries are embedded before indexing, so the dense similarity score itself reflects cross-reference content. The NitiBench-CCL cross-reference subanalysis confirms this: StrucChunk outperforms Hierarchy-Aware by +2.3% HR@5 specifically on the 1,211 cross-reference queries, while NitiLink’s post-retrieval approach was inefective. Akarajaradwong et al. [12] provide a mechanistic explanation via their depth analysis (their Figures 7–8): for single-label NitiBench-CCL queries, retrieval performance plateaus at expansion depth 2, meaning adding more cross-references at inference time yields diminishing returns. By contrast, multi-label NitiBench-Tax queries improve up to depth 6. This directly explains why pre-embedding augmentation outperforms NitiLink for statutory law: at index time, the embedding model integrates cross-reference semantics into the similarity score in a single pass; NitiLink’s depth-first expansion at inference time adds ranked noise beyond depth 2 without the benefit of embedding-space integration.

Within-corpus complexity gradient. Larger relative improvement on CPC (+64.6%) vs. CrPC (+44.4%) shows structure-awareness provides increasing returns as document hierarchy grows more complex.

GDPR structural finding. On GDPR, StrucChunk+Dense (R@5 = 0.908) outperforms StrucChunk+Hybrid (0.868). The chunking efect remains positive (+0.050) while the retrieval efect is negative (−0.040), confirming the Legal Map benefit lies in the structural representation, not the retrieval mechanism.

External and cross-language validation. StrucChunk achieves Char-R@8 = 48.18% on LegalBench-RAG (+83% vs. published SOTA) and HR@5 = 0.856 on NitiBench-CCL (+44.1% vs. naive), confirming the Legal Map hypothesis holds for US/UK contracts in English and Thai financial law in Thai.

Falsifiability. The Legal Map hypothesis makes a testable prediction: structureaware chunking should provide diminishing benefit as document hierarchical structure weakens. Concretely, the hypothesis predicts that StrucChunk’s advantage over section-boundary-only chunking (Hierarchy-Aware) would shrink on documents with inconsistent or absent section numbering—such as unstructured case law opinions, arbitration transcripts, or free-form legal memoranda—where there is no reliable hierarchy to encode in breadcrumbs or cross-reference annotations. On NitiBench-CCL, where section structure is well-defined, StrucChunk’s cross-reference augmentation improves cross-reference queries by +2.3% over Hierarchy-Aware. On unstructured documents, we predict this gap approaches zero or reverses. Testing this prediction on case law corpora (e.g., LexGLUE [9]) is identified as future work.

External corroboration: “Hidden Hierarchical Information.” Akarajaradwong et al. [12] independently identify a failure mode they term “Hidden Hierarchical Information” (their Section 6.5.1): multiple sections across diferent chapters convey near-identical semantic content but difer only in hierarchical position (e.g., Revenue Code §27 and §89/1 both govern late payment charges but reside in diferent chapters). Their proposed fix is to embed chapter and hierarchy context directly into chunk content—which is exactly what StrucChunk’s breadcrumb prepending does. A breadcrumb [Law > Chapter XII > Section 167] is distinguishable in embedding space from [Law > Chapter II > Section 167] even when both sections have nearidentical text, because the chapter identifier shifts the embedding vector. This external corroboration from an independent paper directly validates StrucChunk’s breadcrumb design choice.

## 6.2 Why Hybrid Retrieval Matters

Legal language presents a unique duality: high-variance prose coexists with lowvariance boilerplate and precise citations. Dense retrieval excels at semantic relationships (“bail” ≈ “custody”), while BM25 ensures exact matching of section/article numbers. Naive Hybrid consistently outperforms both single-method baselines. Pipitone & Alami [10] found that the Cohere general-purpose reranker hurt legal performance by 13.5 percentage points, validating domain-aware chunking over post-hoc reranking.

On legal-domain rerankers. A natural follow-up is whether a domain-specific reranker—such as BGE-Reranker-v2 [18], trained on multilingual retrieval tasks including legal text—would perform diferently. We do not include a reranker in our pipeline for two reasons. First, the ablation results (Tables 13–17) show that chunking architecture accounts for 73–88% of total improvement across corpora; a reranker operating on poorly-chunked text cannot recover the information lost to fragmented chunks. Second, adding a reranker would conflate two independent contributions: chunking quality and post-hoc ranking. A principled comparison would require running (Recursive + reranker) vs. (StrucChunk + reranker) to isolate whether the reranker magnifies or reduces StrucChunk’s advantage. We leave this as future work, noting that Akarajaradwong et al. [12] evaluated the WangchanX legal reranker on Thai statutory text and found inconsistent gains—consistent with the hypothesis that domain-specific rerankers are not yet reliable enough to supersede structural chunking improvements.

## 6.3 Cross-Language Generalization

The NitiBench-CCL results establish that StrucChunk’s benefit is not English-specific. Thai statutory codes share the same structural property that drives StrucChunk’s advantage on CrPC/CPC: a hierarchical section numbering system (m¯atr¯a N in Thai; Section N in English) where legal provisions are explicitly delineated. BGE-M3’s multilingual training enables the breadcrumb context (“law name > m¯atr¯a N”) to provide useful retrieval signal in Thai. No Thai-specific adaptation was required beyond loading the pre-structured dataset.

Fine-tuned retriever confound. NitiBench’s full evaluation suite includes a Human-Finetuned BGE-M3 that achieves HR@5 = 0.906, surpassing our base BGE-M3 StrucChunk result of 0.856. We use base BGE-M3 throughout for consistent cross-corpus comparison (Tiers 1–4 use identical retrieval). The gap between fine-tuned (0.906) and our StrucChunk+base (0.856) does not undermine the chunking contribution, because fine-tuning and structure-aware chunking are orthogonal: a fine-tuned model operating on StrucChunk chunks would be expected to match or exceed the finetuned model on hierarchy-aware chunks (0.906), since StrucChunk provides additional breadcrumb and cross-reference context that the fine-tuned model can exploit. The chunking gain (+0.262 over naive) is measured with the same base BGE-M3 across all three methods, so the retriever is not a confound for the chunking comparison itself.

Retrieval gains and end-to-end performance. StrucChunk’s gains are measured at the retrieval layer. NitiBench’s own end-to-end results (their Table 7) show that hierarchy-aware chunking yields only modest E2E gains over naive chunking (E2E

F1: 0.739 vs. 0.722, ∆ = +0.017; Coverage: 86.7 vs. 86.6). This suggests LLMs partially compensate for poor retrieval via parametric knowledge. The implication is not that retrieval quality is unimportant, but rather the opposite: in hallucination-sensitive legal settings where LLMs cannot rely on parametric knowledge (novel statutes, recent amendments, jurisdiction-specific provisions), retrieval-layer improvements become a necessary condition for factual accuracy. Measuring at the retrieval layer isolates this contribution cleanly.

## 6.4 Qualitative Failure Analysis

Representative success case (CrPC): Query: “What are the conditions for detention under Section 167(2)(a)?”

• Recursive + Dense fails: the 512-token limit cuts mid-section at the 167(1)/167(2) boundary. The 167(2)(a) chunk appears as “(a) where he is accused. . . ” with no section-number context. Not retrieved in top-5.

• BM25 Only partially succeeds: retrieves Section 167 via keyword match but ranks the header chunk above the sub-provision. Sub-provision ranks 7th.

• StrucChunk + Hybrid succeeds: breadcrumb [<sub>CrPC</sub> <sub>1973</sub> <sub>></sub> <sub>Chapter</sub> <sub>XII</sub> <sub>></sub> Section 167] is prepended; cross-reference augmentation appends Section 57 summary. Ranks 1st.

Quantified CrPC failure categories. Of the 98 queries where Struc-Chunk+Hybrid fails at Recall@5 (n = 501, R@5 = 0.804), we classify failures as:

• Category A (section absent from parsed corpus): 9 queries (9.2% of failures, 1.8% of all queries). These are recoverable PDF-parsing gaps: Sections 145, 195, 210, 228A, 268, and 317 were not extracted by the parser. They represent the practical ceiling improvement available from fixing the PDF parser.

• Category B (section present, ranks 6–10): ≈27 queries (27.6% of failures), estimated from R@10 − R@5 = 0.858 − 0.804 = 0.054 × 501 ≈ 27. These near-misses are captured at Recall@10.

• Category C (section present, ranks >10): ≈62 queries (63.3% of failures). These represent genuine semantic mismatch between query formulation and section content. Query-side improvements (query rewriting, HyDE) would primarily address this category.

Category C queries are disproportionately factual queries where the section’s prose does not closely mirror common query formulations (e.g., “What is the procedure when there is dispute concerning land?” maps to Section 145, whose title uses the term “proceedings” rather than “procedure”).

CPC failure case: Order-Rule fragmentation. CPC’s most common failure mode is structural: the Part-Section-Order-Rule hierarchy means that the same numeral (“Rule 1”) reappears across 51 diferent Orders. Query: “Under what conditions can a court extend time limits under Order VIII Rule 1?” Recursive chunking creates a chunk containing “Order VIII” as a heading followed by partial Rules 1–3 text, without the Rule number appearing in the chunk’s section identifier. StrucChunk partially mitigates this: the breadcrumb [CPC 1908 > Order VIII > Rule 1] correctly identifies the provision. However, StrucChunk still fails on ≈74% of CPC queries (R@5 = 0.259), primarily because the CPC PDF’s inconsistent Order-header formatting causes some Orders to be mis-parsed as chapter-level rather than section-level nodes, preventing correct breadcrumb assignment. This structural explanation—CPC PDF formatting quality, not StrucChunk methodology—accounts for the persistent gap between CrPC (R@5 = 0.804) and CPC (R@5 = 0.259).

GDPR failure case: Recital-Article confusion. GDPR’s characteristic failure mode is Recital-Article confusion. Query: “What are the conditions under Article 6(1)(f) regarding legitimate interests?” Recital 47, which discusses legitimate interests in extensive prose, has strong semantic similarity to this query. The Recital chunk lacks the breadcrumb [GDPR > Chapter II > Article 6] but scores high on dense similarity due to shared vocabulary. Simultaneously, BM25 retrieves multiple Recitals that cite “Article 6” as a cross-reference target, diluting the dense ranking. This explains the hybrid penalty on GDPR (StrucChunk+Hybrid R@5 = 0.868 vs. StrucChunk+Dense R@5 = 0.908): BM25 introduces Recital noise that the dense-only pipeline avoids. On plain queries (no cross-reference) where section text fits within 553 characters, naive chunking with BGE-M3 achieves HR@5 = 0.845 (covered-query subset). StrucChunk achieves 0.868 on the same covered subset—a smaller gain (+2.7%) than on crossreference queries (+2.3% overall). This confirms the diminishing marginal benefit of structural context when sections are short and the query maps directly to section content without needing cross-reference co-location.

## 6.5 Limitations

1. Evaluation scope: 1,503 author-created queries across three English documents, plus 776 independently annotated LBR queries, plus 3,730 independently annotated NitiBench-CCL queries (Thai). Results should not be generalized to unstructured legal documents.

2. No end-to-end evaluation: We measure retrieval quality only.

3. Single annotator (CrPC/CPC/GDPR): All 1,503 query-annotations (AIassisted query generation with manual verification of expected sections) were produced by one annotator. LegalBench-RAG and NitiBench-CCL provide independent annotation for evaluation 1 and 4. Inter-annotator agreement on a 100-query sample (50 CrPC, 50 CPC) is reported in Section 4.6.

4. Regex cross-reference detection: Misses implicit references (88% success rate on manual sample; Appendix ??).

5. PDF quality dependency: Evaluations 1–3 require well-formatted PDFs. NitiBench-CCL uses pre-structured data.

6. NitiBench-CCL retrieval: We evaluate dense retrieval only for the three-way chunking comparison. Hybrid retrieval on NitiBench-CCL is future work (requires PyThaiNLP tokenization for BM25).

7. Evaluation paradigm gaps: LBR char-span, CrPC/CPC/GDPR section-level, and NitiBench-CCL HR@k are diferent protocols and are not directly comparable numerically.

8. LegalBench-RAG single-document scope: Pipitone & Alami [10] note that LBR queries are always answered by exactly one document; the benchmark does not assess cross-document retrieval. Our evaluation 1 results therefore reflect intradocument retrieval precision. Performance on cross-document legal reasoning (e.g., queries spanning multiple statutes) remains an open question and may explain why CrPC cross-reference queries (which span provisions within one document) show diferent patterns from LBR results.

## 7 Broader Impact

Improved retrieval for legal documents could reduce barriers to legal information access, assisting legal aid organizations and self-represented litigants in navigating public legal codes.

At the same time, higher retrieval quality may accelerate automation of tasks currently performed by legal professionals, with potential workforce consequences. High retrieval accuracy may also create overconfidence in AI-assisted legal tools. In a domain where “mostly correct” constitutes unacceptable liability, practitioners must continue to verify AI outputs against primary sources. We caution against deployment in highstakes legal settings without human oversight and end-to-end accuracy evaluation. Our implementation is released for research purposes only.

## 8 Conclusion

We presented StrucChunk with Hybrid Retrieval, validated through a four-tier evaluation strategy across five corpora, three legal traditions, and two languages. Key results:

• Chunking quality (CrPC): 100% boundary precision vs. 7.7% recursive; 97.6% cross-reference preservation vs. 0%; 100% section coverage vs. 0%.

• CrPC (n = 501, English): Recall@5 0.557 → 0.804 (+44.4%); R@1 0.214 → 0.627 (+193%). Chunking = 88% of gain.

• CPC (n = 501, English): Recall@5 $0 . 1 5 8  0 . 2 5 9 \ ( + 6 4 . 6 \% )$ ; cross-ref queries +85%. Chunking = 73% of gain.

• GDPR (n = 501, English): R@5 = 0.868; R@1 +59%. StrucChunk+Dense is best overall $( \mathrm { R @ 5 } = 0 . 9 0 8 , \mathrm { R @ 1 } = 0 . 8 8 0 )$

• LegalBench-RAG (n = 776, English, char-span): Char-R@8 = 48.18% vs. 23.59% baseline (+104%); vs. published SOTA 26.30% (+83%).

• NitiBench-CCL (n = 3,730, Thai): HR@5 = 0.856 vs. 0.594 naive (+44.1%); 29.6% of Thai legal sections are structurally unrepresentable by naive fixed-size chunking. StrucChunk’s cross-reference augmentation provides +2.3% on crossreference queries where post-retrieval expansion (NitiLink) was inefective.

• All improvements statistically significant $( p ~ \leq ~ 0 . 0 0 2$ , Wilcoxon + Benjamini-Hochberg; Cohen’s d 0.14–0.69).

The four-tier design establishes: (1) StrucChunk works on data we did not create (LBR and NitiBench-CCL); (2) gains are largest where document structure is most complex; (3) the method generalizes across document types, legal traditions, and languages; and (4) section-boundary alignment is the dominant factor, with crossreference augmentation providing targeted improvement exactly where cross-references are semantically relevant.

Future Work. Several directions remain open. First, extending hybrid retrieval to NitiBench-CCL requires a Thai-specific BM25 tokenizer to properly support sparse retrieval. Second, cross-reference resolution could be improved using NLU-based methods rather than pattern matching, enabling more robust handling of implicit or context-dependent references. Third, an end-to-end evaluation with downstream LLM generation (rather than retrieval metrics alone) would clarify the practical impact of structure-aware chunking on answer quality. Fourth, fine-tuning domain-specific embeddings for legal text may further improve dense retrieval performance beyond general-purpose models like BGE-M3. Fifth, combining StrucChunk with Semantic Aware Chunking (SAC) could capture complementary structural and semantic signals. Finally, extending the approach to case law and other unstructured legal documents would test the generality of structure-aware chunking beyond codified statutes.

## Statements and Declarations

## Funding

The authors declare that they did not receive any financial assistance, grants, or support during the work.

## Competing Interests

The authors have no relevant financial or non-financial interests to disclose.

## Author contributions statement

S.D.S. conceived the study, designed the StrucChunk methodology, and led the overall research direction. S.D.S. and A.T. implemented the chunking and retrieval pipeline and conducted the experiments across the five evaluation corpora. G.D.S. contributed to experimental design, review of the annotation methodology, and validation of results. S.D.S. wrote the original draft of the manuscript. A.T. and G.D.S. reviewed and edited the manuscript for critical content. All authors read and approved the final manuscript.

## Ethical approval

Not applicable.

## Human participants and/or animals

Not applicable to this research study.

## Data availability

All the datasets used for experiments have been provided.

## References

[1] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., K¨uttler, H., Lewis, M., Yih, W.-t., Rockt¨aschel, T., Riedel, S., Kiela, D.: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2021). https://arxiv.org/abs/ 2005.11401

[2] Dahl, M., Magesh, V., Suzgun, M., Ho, D.E.: Large legal fictions: Profiling legal hallucinations in large language models. Journal of Legal Analysis 16(1), 64–93 (2024) https://doi.org/10.1093/jla/laae003 https://academic.oup.com/jla/article-pdf/16/1/64/58336922/laae003.pdf

[3] Magesh, V., Surani, F., Dahl, M., Suzgun, M., Manning, C.D., Ho, D.E.: Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools (2024). https://arxiv.org/abs/2405.20362

[4] Kabir, M.R., Sultan, R.M., Rahman, F., Amin, M.R., Momen, S., Mohammed, N., Rahman, S.: LegalRAG: A Hybrid RAG System for Multilingual Legal Information Retrieval (2025). https://arxiv.org/abs/2504.16121

[5] Sculley, D., Snoek, J., Wiltschko, A., Rahimi, A.: Winner’s curse? on pace, progress, and empirical rigor. In: International Conference on Learning Representations (ICLR), Workshop Track (2018). https://openreview.net/forum?id= rJWF0Fywf

[6] Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, M., Wang, H.: Retrieval-Augmented Generation for Large Language Models: A Survey (2024). https://arxiv.org/abs/2312.10997

[7] Asai, A., Wu, Z., Wang, Y., Sil, A., Hajishirzi, H.: Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (2023). https://arxiv.org/abs/ 2310.11511

[8] Yan, S.-Q., Gu, J.-C., Zhu, Y., Ling, Z.-H.: Corrective Retrieval Augmented Generation (2024). https://arxiv.org/abs/2401.15884

[9] Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N., Androutsopoulos, I.: LEGAL-BERT: The muppets straight out of law school. In: Cohn, T., He, Y., Liu, Y. (eds.) Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 2898–2904. Association for Computational Linguistics, Online (2020). https://doi.org/10.18653/v1/2020.findings-emnlp.261 . https:// aclanthology.org/2020.findings-emnlp.261/

[10] Pipitone, N., Alami, G.H.: LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain (2024). https://arxiv.org/abs/2408. 10343

[11] Reuter, M., Lingenberg, T., Liepina, R., Lagioia, F., Lippi, M., Sartor, G., Passerini, A., Sayin, B.: Towards reliable retrieval in RAG systems for large legal datasets. In: Aletras, N., Chalkidis, I., Barrett, L., Goant˘a, C., Preotiuc-Pietro, D., Spanakis, G. (eds.) Proceedings of the Natural Legal Language Processing Workshop 2025, pp. 17–30. Association for Computational Linguistics, Suzhou, China (2025). https://doi.org/10.18653/v1/2025.nllp-1.3 . https://aclanthology. org/2025.nllp-1.3/

[12] Akarajaradwong, P., Pothavorn, P., Chaksangchaichot, C., Tasawong, P., Nopparatbundit, T., Pratai, K., Nutanong, S.: NitiBench: A Comprehensive Study of LLM Framework Capabilities for Thai Legal Question Answering (2025). https://arxiv.org/abs/2502.10868

[13] Nigam, S.K., Patnaik, B.D., Mishra, S., Thomas, A.V., Shallum, N., Ghosh, K., Bhattacharya, A.: NyayaRAG: Realistic legal judgment prediction with RAG under the Indian common law system. In: Inui, K., Sakti, S., Wang, H., Wong, D.F., Bhattacharyya, P., Banerjee, B., Ekbal, A., Chakraborty, T., Singh, D.P. (eds.) Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, pp. 1709–1726. The Asian Federation of Natural Language Processing and The Association for Computational Linguistics, Mumbai, India (2025). https://doi.org/10.18653/v1/2025.ijcnlp-long.92 . https://aclanthology.org/2025.ijcnlp-long.92

[14] Hendrycks, D., Burns, C., Chen, A., Ball, S.: CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review (2021). https://arxiv.org/abs/2103.06268

[15] Yepes, A.J., You, Y., Milczek, J., Laverde, S., Li, R.: Financial Report Chunking for Efective Retrieval Augmented Generation (2024). https://arxiv.org/abs/ 2402.05131

[16] Cormack, G.V., Clarke, C.L.A., B¨uttcher, S.: Reciprocal rank fusion outperforms condorcet and individual rank learning methods. Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval (2009)

[17] Bruch, S., Gai, S., Ingber, A.: An analysis of fusion functions for hybrid retrieval. ACM Transactions on Information Systems 42(1), 1–35 (2023) https://doi.org/ 10.1145/3596512

[18] Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., Liu, Z.: M3-embedding: Multilinguality, multi-functionality, multi-granularity text embeddings through selfknowledge distillation. In: Ku, L.-W., Martins, A., Srikumar, V. (eds.) Findings of

the Association for Computational Linguistics: ACL 2024, pp. 2318–2335. Association for Computational Linguistics, Bangkok, Thailand (2024). https://doi.org/ 10.18653/v1/2024.findings-acl.137 . https://aclanthology.org/2024.findings-acl. 137/

[19] Cohen, J.: Statistical Power Analysis for the Behavioral Sciences, 2nd edn. Lawrence Erlbaum Associates, Hillsdale, NJ (1988)

[20] Dror, R., Baumer, G., Shlomov, S., Reichart, R.: The hitchhiker’s guide to testing statistical significance in natural language processing. In: Gurevych, I., Miyao, Y. (eds.) Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1383–1392. Association for Computational Linguistics, Melbourne, Australia (2018). https://doi.org/10. 18653/v1/P18-1128 . https://aclanthology.org/P18-1128/