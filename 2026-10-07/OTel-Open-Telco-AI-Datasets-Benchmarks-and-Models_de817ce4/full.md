# OTel: Open Telco AI Datasets, Benchmarks, and Models

Farbod Tavakkoli<sup>1∗</sup> Gregory Diamos<sup>2</sup> Kenneth Church<sup>2</sup> David Kanter<sup>3</sup> Mark Austin<sup>1</sup> Imtiaz Karim<sup>4</sup> Mirza Masfiqur Rahman<sup>5</sup> Merouane Abdelkader Debbah<sup>6</sup> Zeinab Nezami<sup>7</sup> Ali Maatouk<sup>8</sup> Leandros Tassiulas<sup>8</sup> Rex Ying<sup>8</sup> Nick Sorros<sup>9</sup> Louis Powell<sup>10</sup> Nikolaos Vasiloglou<sup>2</sup> Ashish Vaswani<sup>11</sup> Somanshu Singla<sup>11</sup> Adarsh Chaluvaraju<sup>11</sup>

<sup>1</sup>AT&T Chief Data Office <sup>2</sup>RelationalAI <sup>3</sup>MLCommons <sup>4</sup>The University of Texas at Dallas <sup>5</sup>Purdue University <sup>6</sup>Khalifa University <sup>7</sup>University of Leeds <sup>8</sup>Yale University <sup>9</sup>Mantis NLP <sup>10</sup>GSMA <sup>11</sup>Essential AI

## Abstract

We present Open Telco (OTel), an open telecom AI resource that releases derived telecom datasets for retrieval, reranking, instruction tuning, and safety/abstention, together with 30 full-parameter post-trained baselines spanning 10 embedding models, 3 rerankers, and 17 language models. The community has already engaged substantially with the resource: as of May 3, 2026, the released models have been downloaded over 16 million times and the project has received 157+ pieces of media coverage worldwide. Building on prior open telecom datasets and benchmarks, OTel provides documented telecom data sources, held-out evaluation partitions, trained embedding models, rerankers, context-grounded LLMs, and safety/abstention data in one unified resource. Each baseline starts from an open-weight model and is post-trained on OTel-derived data using an open training recipe, then evaluated on held-out OTel evaluation partitions. OTel post-training improves performance across all three model families: embedding retrieval reaches 93.1% NDCG@10, reranking reaches 0.947 MRR@10, and language-model correctness reaches 87.8%. We release OTel as a reproducible starting point and invite the community to expand the data, improve embedding and reranking models, and build stronger context-grounded telecom LLMs.

## 1 Introduction

Open Telco (OTel)<sup>23</sup> is an open telecom AI resource designed to help researchers and practitioners train, evaluate, and improve retrieval, reranking, context-grounded generation, and abstention for telecommunications. The resource combines derived telecom datasets, held-out evaluation partitions, and 30 released full-parameter post-trained baselines across embedding models, rerankers, and language models. Our goal is to provide a reproducible starting point for the community rather than a final answer: the released baselines are intended to make it easier for others to compare methods, identify weaknesses, and improve telecom retrieval, reranking, context-grounded generation, and abstention.

The paper makes five points:

1. Resource. OTel is a new open telecom AI resource that builds on prior telecom benchmarks and documents data sources, tasks, and derived dataset formats.

2. Community use. The resource has already seen broad community engagement, with over 16 million model downloads and 157+ media mentions as of May 3, 2026.

3. Reference baselines. OTel releases Hugging Face reference baselines and invites the community to improve on them.

4. Headline performance. The released baselines reach 93.1% NDCG@10 for retrieval, 0.947 MRR@10 for reranking, and 87.8% LLM correctness.

5. Fine-tuning and scale. The results show that OTel fine-tuning improves performance across model families and that larger models generally raise the upper envelope.

There has been considerable recent work on telecom benchmarks and datasets [Karim et al., 2023]. TeleQnA evaluates telecom question answering and standards understanding [Maatouk et al., 2023]. ORAN-Bench and srsRAN-Bench evaluate O-RAN specifications and open-source 5G code understanding [Gajjar and Shah, 2024]. Additional benchmarks target 3GPP group classification, telecom table reasoning, telecom mathematical reasoning, and 5G root-cause analysis [Zou et al., 2024, Ezzakri et al., 2026, Colle et al., 2025, Sana et al., 2025]. The GSMA Open Telco AI Leaderboard [GSMA, 2025] brings these efforts into a shared benchmark view for telecom AI.

OTel builds on this prior work by contributing a unified open resource for training and evaluating telecom RAG components across multiple telecom domains and model families. We use OTel source corpus to refer to the publicly available telecom documents used as inputs, including 3GPP specifications, GSMA documents, O-RAN documents, RFCs, whitepapers, academic papers, and webderived telecom material. We use OTel derived datasetfamily to refer to the cleaned and structured examples released as OTel-Embedding, OTel-Reranker, OTel-LLM, and OTel-Safety. We use OTel evaluation partitions to refer to held-out splits from these derived datasets, and OTel model family to refer to the 30 released full-parameter post-trained baselines.

This paper presents OTel as an Evaluations & Datasets contribution organized around three assets:

1. A derived OTel dataset family built from heterogeneous telecom sources and cleaned from roughly 1.1M raw training points to a higher-quality final corpus, released as OTel-Embedding, OTel-Reranker, OTel-LLM, and OTel-Safety.

2. A family of 30 released full-parameter post-trained OTel baselines following the RAG pipeline, comprising 10 embedding models, 3 rerankers, and 17 LLMs. The released model naming convention is OTel-Embedding-{model size} for embedding models, OTel-Reranker-{model size} for rerankers, and OTel-LLM-{model size}-{IT or Reasoning} for LLMs; auxiliary safety variants use OTel-LLM-{model size}-Safety and are discussed in the appendix.

3. A reproducible evaluation setup for telecom RAG components, using held-out OTel evaluation partitions and model-family-specific metrics, namely LLM-as-judge correctness for LLMs, NDCG@10 for embeddings, and MRR@10 for rerankers.

As of May 3, 2026, the released OTel models have been downloaded over 16 million times and the project has received 157+ pieces of media coverage, providing an early signal of community demand for open telecom AI resources. We release OTel to support continued work on telecom retrieval, reranking, context-grounded generation, and safety, and to invite the community to improve on the baselines reported here.

## 2 Related Work and Benchmark Context

Telecom benchmarking has expanded rapidly during the last two years and now spans several complementary tasks. TeleQnA measures broad telecom knowledge and standards understanding [Maatouk et al., 2023]. ORAN-Bench and srsRAN-Bench probe O-RAN and open-source 5G implementation knowledge [Gajjar and Shah, 2024]. 3GPP-TSG tests working-group classification [Zou et al., 2024]. TeleTables focuses on reasoning over technical tables [Ezzakri et al., 2026]. TeleMath measures domain-specific mathematical reasoning [Colle et al., 2025]. TeleLogs targets 5G root-cause analysis [Sana et al., 2025]. The GSMA Open Telco AI Leaderboard [GSMA, 2025] is an important step because it consolidates these benchmark tasks into a unified evaluation view for telecom AI.

Viewed as part of the broader dataset and post-training ecosystem, OTel connects retrieval data, reranking supervision, instruction-tuning data, safety/abstention data, and released post-trained baselines that are often studied separately.
<table><tr><td rowspan="2">Data sources &amp; tasks</td><td colspan="3">Prior Work</td><td>This Work</td></tr><tr><td>Tele-Data</td><td>TEmbed</td><td>GSMA Leaderboard</td><td>OTel</td></tr><tr><td>Telecom scope</td><td>Telecom QA</td><td>Telecom data</td><td>Benchmark tasks</td><td>Multi-domain telecom data</td></tr><tr><td>Retrieval chunks</td><td>×</td><td>X</td><td>Eval-only</td><td>√ explicit chunks</td></tr><tr><td>Reranker labels</td><td>X</td><td>X</td><td>Eval-only</td><td>√relevance labels</td></tr><tr><td>LLM instruction data</td><td>√ QA-oriented</td><td>√QA-oriented</td><td>Eval-only</td><td>√ grounded prompts</td></tr><tr><td>Safety</td><td>X</td><td>X</td><td>X</td><td>√abstention data</td></tr><tr><td>Held-out eval partitions</td><td>Splits</td><td>Splits</td><td>Benchmark entries</td><td>√ OTel partitions</td></tr><tr><td>Post-trained baselines</td><td>X</td><td>X</td><td>X</td><td>√30 baselines</td></tr></table>

Table 1: Data sources and tasks covered by prior open telecom resources and OTel. Retrieval chunks indicate explicit positive/negative retrieval chunks. The comparison shows how OTel builds on prior work by covering the retrieval–reranking–generation workflow and safety/abstention data; it is not intended to diminish earlier datasets or leaderboards.

Table 1 summarizes the role of OTel in this ecosystem. Earlier open datasets such as Tele-Data and TEmbed helped make telecom training data more accessible, while the GSMA leaderboard provides a shared benchmark interface for several telecom tasks. OTel is complementary to these efforts: it contributes aligned open training resources and full-parameter post-trained OTel baselines for studying retrieval, reranking, and context-grounded generation systematically. This positioning is similar in spirit to domain evaluation ecosystems in medicine [Jin et al., 2021], law [Guha et al., 2023], and finance [Xie et al., 2024], where shared resources and baselines make it easier for the community to compare methods and improve over time.

## 3 The OTel Resource

Following the terminology introduced above, the OTel source corpus consists of public telecom documents and contributor-provided telecom examples, while the released OTel artifacts are derived, cleaned, and structured training and evaluation examples. The raw public source documents are not themselves our release contribution; rather, our contribution is the curated OTel derived dataset family produced from these sources for retrieval, reranking, context-grounded generation, and safety.

The OTel resource was curated by more than 100 domain experts from industry and academia. The source corpus spans six source categories: GSMA permanent reference documents, 3GPP specifications, O-RAN documents, RFCs, telecom-specific topical material such as eSIM and roaming, and industry whitepapers and telecom academic papers.

The incoming data had two distinct formats. Yale contributed roughly 680K question-answer-source triples from telecom papers, standards, Wikipedia, and web-derived telecom pages, which OTel converted into retrieval-ready supervision through enrichment and cleaning. These triples did not include positive or negative retrieval passages, and source documents can be hundreds of thousands of tokens. The remaining contributors provided roughly 420K structured examples with explicit question, positive chunks, negative chunks, answer, and source fields.

To convert the Yale data into retrieval-ready supervision, we used a six-stage enrichment pipeline. First, an ETL stage grouped QA pairs by source document and joined them to the full document text. Second, we created candidate passages via sliding windows and semantic chunking. Third, a retrieval pipeline mined hard negatives both within the source document and across other documents, followed by reranker rescoring. Fourth, we used fact-grounded checks over decomposed answer claims rather than only answer similarity. Fifth, we selected the minimal sufficient context through greedy minimization over top-1, top-3, and all-passage contexts. Sixth, we formatted the verified results into training structures suitable for Multiple Negatives Ranking Loss and related retrieval objectives.

<table><tr><td>Contributor</td><td>Source domain</td><td>Samples</td></tr><tr><td>Yale University</td><td>arXiv telecom papers, 3GPP standards, telecom Wikipedia articles, and telecom-related Common Crawl pages</td><td>681,172</td></tr><tr><td>NetoAI</td><td>RFC series</td><td>100,751</td></tr><tr><td>GSMA</td><td>PRDs, Discover portal material, and mixed telecom documents</td><td>158,006</td></tr><tr><td>Khalifa University</td><td>Industry whitepapers</td><td>62,000</td></tr><tr><td>University of Leeds</td><td>O-RAN specifications across working groups 1–2, 4–10</td><td>58,565</td></tr><tr><td>University of Texas Dallas and Purdue</td><td>O-RAN documents across working groups</td><td>42,000</td></tr><tr><td>University</td><td></td><td></td></tr></table>

Table 2: Contributor summary for the raw OTel corpus.

After enrichment, the Yale data was merged with the structured submissions from the remaining contributors, producing roughly 1.1M training points before aggressive filtering. We then performed a shard-based data-quality experiment in which separate full-parameter embedding models were trained on independent shards. Performance varied substantially, from 71.1% Acc@1 on the best shard to 3.9% Acc@1 on the noisiest shard, confirming that the raw data required significant cleaning.

The final cleaning pipeline applied four filters: heuristic filtering, reranker-based semantic filtering, embedding-based semantic filtering, and deduplication. For the Yale subset, this reduced 680K examples to 220,334. For the remaining contributors, the same process reduced 420K examples to 106,433. The final retained corpus therefore contains 326,767 higher-confidence examples.

<table><tr><td>Dataset split</td><td>Raw size</td><td>Final size</td></tr><tr><td>Yale enrichment path Other contributors</td><td>680,000 420,000</td><td>220,334 106,433</td></tr><tr><td>Total</td><td>~1,100,000</td><td>326,767</td></tr></table>

Table 3: Attrition through the OTel cleaning pipeline.

The retained corpus is released in four model-family-specific formats ordered by the RAG workflow: OTel-Embedding, OTel-Reranker, OTel-LLM, and OTel-Safety. The first three support the 30 released baseline models; OTel-Safety is released separately and supports the appendix-only abstention variants. Because the sources are heterogeneous, release documentation describes provenance, redistribution assumptions, and source-specific constraints at the asset level.

## 4 OTel Post-Trained Baselines

The OTel model family is intended as a reproducible baseline suite for the community rather than a final word on telecom modeling. The release contains 30 full-parameter post-trained OTel baselines following the RAG workflow: 10 embedding models, 3 rerankers, and 17 language models. These baselines establish starting points for retrieval, reranking, and context-grounded generation; show the effect of OTel post-training across model sizes and architectures; and provide reference models for the community to beat, adapt, and extend.

Each released baseline starts from an open base checkpoint and is then full-parameter post-trained on the matching OTel dataset. The 10 embedding models use OTel-Embedding, the 3 rerankers use OTel-Reranker, and the 17 benchmarked LLMs use OTel-LLM. OTel-Safety is reserved for two abstention-focused auxiliary models, OTel-LLM-8.3B-Safety and OTel-LLM-12B-Safety, which are discussed in the appendix and are not counted in the 30-model baseline release.

<table><tr><td>RAG role</td><td>Dataset on Hugging Face</td><td>Task</td><td>Key fields</td></tr><tr><td>Retrieval</td><td>OTel-Embedding</td><td>Retrieve relevant telecom passages from an anchor/query.</td><td>anchor, positive, negative_1- negative_5</td></tr><tr><td>Reranking</td><td>OTel-Reranker</td><td>Score query-passage relevance for cross-encoder reranking.</td><td>sentence_0, sentence_1, label</td></tr><tr><td>Generation</td><td>OTel-LLM</td><td>Generate grounded telecom answers from retrieved context.</td><td>prompt, completion, abstention, chunk-count metadata</td></tr><tr><td>Safety</td><td>OTel-Safety</td><td>Abstain when appropriate.</td><td>Schema-compatible with OTel-LLM; includes abstention and chunk-count metadata</td></tr></table>

Table 4: Released OTel derived dataset family, ordered by the retrieval-reranking-generation workflow and its safety extension.

<table><tr><td>RAG role</td><td>Dataset</td><td>Shortened real example</td></tr><tr><td>Retrieval</td><td>OTel-Embedding</td><td>Retrieval example: a query about F1 Measurement ID Coordination is paired with the relevant F1-log passage and five nearby hard negatives.</td></tr><tr><td>Reranking</td><td>OTel-Reranker</td><td>Reranking example: an O-RAN Fronthaul Gateway query-passage pair about split 7-2 to 8 is marked relevant with 1abe1=1.0.</td></tr><tr><td>Generation</td><td>OTel-LLM</td><td>Generation example: an MBSFN-mode question is answered from retrieved context about TDD MBSFN Information.</td></tr><tr><td>Safety</td><td>OTel-Safety</td><td>Safety example: an SCP trust-domain question has off-topic retrieved contexts, so the completion abstains with abstention=true.</td></tr></table>

Table 5: Compact examples drawn from released OTel rows and shortened for readability. Full rows and regeneration commands are documented in the release materials.

We use the term full-parameter post-trained OTel baselines deliberately: these are not frozen encoders, prompt-only wrappers, or loosely assembled checkpoints, but a controlled family of post-trained telecom models built with a shared recipe.

A representative example is OTel-LLM-8.3B-IT. We initialize from the open rnj-1-instruct checkpoint [Essential AI, 2025] and then full-parameter post-train on OTel-LLM for three epochs, producing a telecom-specialized instruction model. The same pattern is repeated across the released OTel baseline family: an open base model is selected, the family-appropriate OTel dataset is loaded, and the full parameter set is updated under a common configuration template.

The shared training recipe uses AdamW in 8-bit form, cosine decay with warmup, seed 42, a maximum sequence length of 1500 tokens, BF16 precision, Flash Attention 2, gradient checkpointing, and Fully Sharded Data Parallel training. LLMs and embedding models train for three epochs; rerankers train for two. Full base-model mappings and hyperparameter tables are deferred to the appendix to keep the main paper focused on the E&D contribution rather than a file-by-file training inventory.

## 5 Evaluation Protocol

OTel evaluates three roles in the telecom RAG stack: embeddings for retrieval, rerankers for passage ordering, and LLMs for context-grounded answer generation. The evaluation protocol is designed to be reproducible and consistent with the training resources released alongside the baselines.

The LLM evaluation is scoped to context-grounded answer generation: models receive retrieved telecom context and are judged on whether their answers are correct relative to that context and the

<table><tr><td>Family</td><td>Count</td><td>Primary dataset</td></tr><tr><td>Embeddings</td><td>10</td><td>OTel-Embedding</td></tr><tr><td>Rerankers</td><td>3</td><td>OTel-Reranker</td></tr><tr><td>LLMs</td><td>17</td><td>OTel-LLM</td></tr></table>

Table 6: The 30 released full-parameter post-trained OTel baselines, ordered by the RAG workflow.

reference answer. These results should not be interpreted as unrestricted context-free QA performance;   
appendix-only GSMA leaderboard experiments cover that separate setting.

In this paper, safety refers to learning when retrieved context is insufficient or off-topic and the correct behavior is to abstain rather than answer.

Held-out evaluation splits. For LLMs and embeddings, the default split is 90% train and 10% eval with seed 42. For rerankers, the default split is 95% train and 5% eval with seed 42. Unless a stricter external benchmark split is added before submission, all main-paper results are reported on these held-out OTel evaluation partitions.

Metrics. LLMs are evaluated by LLM-as-judge correctness on context-grounded answer generation, using GPT-4o mini and Claude Sonnet 3.5 as judge models to determine whether a generated answer is correct given the provided context and reference answer; the reported correctness is the average score across the two judges. Embeddings are evaluated by NDCG@10, because retrieval quality depends on ranking the most relevant telecom passages at the top of the returned list. Rerankers are evaluated by MRR@10, because reranking quality is governed by how quickly the first truly relevant passage is promoted near the top. Where available, gains over the models without OTel fine-tuning are reported within the same model family.

Reporting policy. The main comparison is across the full-parameter post-trained OTel baselines within each family. All results include base-model performance before OTel fine-tuning and standard errors via bootstrap resampling (n=1000). Auxiliary experiments that are useful but not part of the core 30-model release, including the two abstention-focused safety variants, the TeleLogs classification head, the GSMA non-abstention QnA variant, and GPU utilization analysis, are reported in the appendix.

Collaborator-led stress tests. To probe behavior beyond the aggregate held-out splits, collaborators also ran auxiliary retrieval stress tests. University of Texas at Dallas and Purdue evaluated O-RAN retrieval on fixed-chunk question pools, while the University of Leeds examined retrieval behavior across 3GPP, GSMA PRD, industry whitepapers, O-RAN, RFCs, and telecom academic papers. These analyses are not reported as main quantitative benchmark tables because the protocols were not yet standardized across all domains, but they provide useful diagnostic evidence: O-RAN retrieval appears comparatively strong, whereas academic-paper and GSMA PRD examples remain weaker and need further curation.

## 6 Results

All results are reported on held-out OTel evaluation partitions. Standard errors are computed via bootstrap resampling (1,000 iterations). Each table includes performance without fine-tuning, performance with OTel fine-tuning, and the absolute gain from OTel fine-tuning.

Language-model baselines. Table 7 reports LLM-as-judge correctness for the 17 released instruction-tuned and reasoning LLM baselines. OTel fine-tuning yields consistent gains across all model sizes, ranging from +3.3 to +9.4 percentage points. OTel-LLM-27B-IT achieves the highest correctness at 87.8%, while OTel-LLM-8.3B-IT is the strongest mid-size baseline at 79.0%, outperforming other models in its weight class. Larger models generally raise the upper envelope, although architecture and training family still matter. OTel-LLM-1.2B-IT provides a competitive low-latency option at 73.8%. Two abstention-focused safety variants are discussed separately in the appendix and are not counted in the initial 30-model release.

<table><tr><td>OTel model</td><td>Base model</td><td>Params (B)</td><td>Without fine-tuning</td><td>With OTel fine-tuning</td><td>∆(pp)</td></tr><tr><td>OTe1-LLM-270M-IT</td><td>gemma-3-270m-it</td><td>0.27</td><td>22.2</td><td>30.7 ± 1.4</td><td>+8.5</td></tr><tr><td>OTel-LLM-0.6B-IT</td><td>Qwen3-0.6B</td><td>0.60</td><td>49.0</td><td>58.4 ± 1.1</td><td>+9.4</td></tr><tr><td>OTel-LLM-1B-IT</td><td>gemma-3-1b-it</td><td>1.00</td><td>48.3</td><td>56.8 ± 1.0</td><td>+8.5</td></tr><tr><td>OTel-LLM-1.2B-IT</td><td>LFM2.5-1.2B-Instruct</td><td>1.20</td><td>66.4</td><td> $7 3 . 8 \pm 0 . 8$ </td><td>+7.4</td></tr><tr><td>OTel-LLM-1.7B-IT</td><td>Qwen3-1.7B</td><td>1.70</td><td>52.8</td><td> ${ \bf 6 0 . 8 \pm 0 . 9 }$ </td><td>+8.0</td></tr><tr><td>OTel-LLM-3B-IT</td><td>Mistral-3-3B</td><td>3.00</td><td>56.9</td><td> ${ \bf 6 3 . 9 \pm 0 . 9 }$ </td><td>+7.0</td></tr><tr><td>OTel-LLM-4B-IT</td><td>gemma-3-4b-it</td><td>4.00</td><td>66.2</td><td> $7 2 . 7 \pm 0 . 8$ </td><td>+6.5</td></tr><tr><td>OTel-LLM-7B-IT</td><td>OLMo-3-7B</td><td>7.00</td><td>57.4</td><td> ${ \bf 6 2 . 9 \pm 0 . 9 }$ </td><td>+5.5</td></tr><tr><td>OTe1-LLM-8.2B-IT</td><td>Qwen3-8B</td><td>8.20</td><td>60.4</td><td> ${ \bf 6 5 . 9 \pm 0 . 8 }$ </td><td>+5.5</td></tr><tr><td>OTe1-LLM-8.3B-IT</td><td>RNJ-1-Instruct</td><td>8.30</td><td>72.5</td><td> ${ \bf 7 9 . 0 \pm 0 . 7 }$ </td><td>+6.5</td></tr><tr><td>OTel-LLM-12B-IT</td><td>gemma-3-12b-it</td><td>12.00</td><td>78.3</td><td> ${ \pm 0 . 6 \pm 0 . 6 }$ </td><td>+4.5</td></tr><tr><td>OTel-LLM-14B-IT</td><td>Qwen3-14B</td><td>14.00</td><td>60.7</td><td> ${ \bf 6 5 . 7 \pm 0 . 8 }$ </td><td>+5.0</td></tr><tr><td>OTel-LLM-20B-IT</td><td>GPT-OSS-20B</td><td>20.00</td><td>61.4</td><td> ${ \bf 6 5 . 9 \pm 0 . 8 }$ </td><td>+4.5</td></tr><tr><td>OTel-LLM-20B-Reasoning</td><td>GPT-OSS-20B</td><td>20.00</td><td>65.2</td><td> ${ \bf 7 1 . 1 \pm 0 . 8 }$ </td><td>+5.9</td></tr><tr><td>OTel-LLM-24B-IT</td><td>LFM2-24B-A2B</td><td>24.00</td><td>75.0</td><td> ${ \bf 7 9 . 0 \pm 0 . 7 }$ </td><td>+4.0</td></tr><tr><td>OTe1-LLM-27B-IT</td><td>gemma-3-27b-it</td><td>27.00</td><td>84.5</td><td> ${ \bf 8 7 . 8 \pm 0 . 5 }$ </td><td>+3.3</td></tr><tr><td>OTel-LLM-32B-IT</td><td>OLMo-3-32B</td><td>32.00</td><td>66.8</td><td> ${ \bf 7 0 . 8 \pm 0 . 8 }$ </td><td>+4.0</td></tr></table>

Table 7: LLM-as-judge correctness (%) on held-out OTel evaluation splits. Correctness is the average score from GPT-4o mini and Claude Sonnet 3.5. OTel fine-tuning improves correctness across all language-model baselines, and larger models generally raise the upper envelope while architecture still matters. ∆ is the absolute percentage-point gain. Standard errors via bootstrap (n=1000).

Embedding baselines. Table 8 reports NDCG@10 for the 10 embedding baselines. OTel finetuning on telecom-specific retrieval supervision produces large gains across all model sizes, with improvements ranging from +9.2 to +59.7 percentage points over the base models. Even the smallest model (0.022B parameters) reaches 83.8% NDCG@10 after OTel fine-tuning, while the largest (8.000B) achieves 93.1%.
<table><tr><td>OTel model</td><td>Base model</td><td>Params (B)</td><td>Without fine-tuning</td><td>With OTel fine-tuning</td><td>∆(pp)</td></tr><tr><td>OTel-Embedding-22M</td><td>all-MiniLM-L6-v2</td><td>0.022</td><td>24.1</td><td> ${ \pm } 3 . 8 \pm 0 . 8$ </td><td>+59.7</td></tr><tr><td>OTel-Embedding-33M</td><td>bge-small-en-v1.5</td><td>0.033</td><td>31.4</td><td> ${ \bf 8 6 . 3 \pm 0 . 7 }$ </td><td>+54.9</td></tr><tr><td> $0 \mathrm { T e l - E m b e d d i n g - } 3 4 \mathrm { M }$ </td><td>all-MiniLM-L12-v2</td><td>0.034</td><td>29.2</td><td> ${ \pm } 4 . 6 \pm 0 . 8$ </td><td>+55.4</td></tr><tr><td> $0 \mathrm { T e l - E m b e d d i n g - } 1 0 9 \mathrm { M }$ </td><td>all-mpnet-base-v2</td><td>0.109</td><td>38.5</td><td> ${ \bf 8 7 . 2 \pm 0 . 7 }$ </td><td>+48.7</td></tr><tr><td> $\mathtt { 0 T e l - E m b e d d i n g - 3 0 0 M }$ </td><td>Gemma3-Embedding-300M</td><td>0.300</td><td>72.3</td><td> ${ \bf 9 0 . 4 \pm 0 . 6 }$ </td><td>+18.1</td></tr><tr><td> $0 \mathrm { T e l - E m b e d d i n g } - 3 3 5 \mathrm { M }$ </td><td>bge-large-en-v1.5</td><td>0.335</td><td>51.7</td><td> ${ \bf 8 9 . 2 \pm 0 . 6 }$ </td><td>+37.5</td></tr><tr><td> $0 \mathrm { T e l - E m b e d d i n g - 5 6 8 M }$ </td><td>bge-m3</td><td>0.568</td><td>57.2</td><td> ${ \bf 8 9 . 6 \pm 0 . 6 }$ </td><td>+32.4</td></tr><tr><td> $\mathtt { 0 T e l - E m b e d d i n g - 6 0 0 M }$ </td><td> $\mathrm { \bar { Q w e n 3 - E m b e d d i n g - 0 . 6 B } }$ </td><td>0.600</td><td>79.7</td><td> ${ \bf 9 0 . 0 \pm 0 . 5 }$ </td><td>+10.3</td></tr><tr><td> $\mathbf { 0 T e l - E m b e d d i n g - 4 B }$ </td><td>Qwen3-Embedding-4B</td><td>4.000</td><td>82.5</td><td> ${ \bf 9 1 . 7 \pm 0 . 5 }$ </td><td>+9.2</td></tr><tr><td> $\mathsf { O T e l - E m b e d d i n g { - } 8 B }$ </td><td>Qwen3-Embedding-8B</td><td>8.000</td><td>83.9</td><td> ${ \bf 9 3 . 1 \pm 0 . 4 }$ </td><td>+9.2</td></tr></table>

Table 8: NDCG@10 (%) on held-out OTel evaluation splits. OTel fine-tuning improves telecom retrieval quality across embedding sizes. ∆ is the absolute NDCG@10 percentage-point gain. Standard errors via bootstrap (n=1000).

Reranker baselines. Table 9 reports MRR@10 for the three reranker baselines. All three models achieve MRR@10 between 0.938 and 0.947 after OTel fine-tuning, up from 0.346–0.417 for the base models, a gain of 0.530–0.592 in absolute MRR@10.

End-to-end interpretation. Taken together, the retrieval, reranking, and generation baselines allow practitioners to choose a telecom RAG stack based on the deployment constraint that matters most. The base-model comparisons in Tables 7–9 confirm that OTel fine-tuning consistently improves domain performance across all three model families and all parameter scales. Larger models generally improve the upper envelope, while architecture and training family still influence the final ranking. Smaller models remain useful for latency-sensitive deployments, and mid-size models provide strong cost-performance trade-offs.

<table><tr><td>OTel model</td><td>Params (B)</td><td>Without fine-tuning</td><td>With OTel fine-tuning</td><td>∆</td></tr><tr><td></td><td>0.60</td><td>0.346</td><td> ${ \bf 0 . 9 3 8 \pm 0 . 0 0 7 }$ </td><td>+0.592</td></tr><tr><td> $\mathtt { O T e l - R e r a n k e r - 0 . 6 B }$   $\mathtt { O T e l - R e r a n k e r - 4 B }$ </td><td>4.00</td><td>0.407</td><td> ${ \bf 0 . 9 4 3 \pm 0 . 0 0 6 }$ </td><td>+0.536</td></tr><tr><td>OTel-Reranker-8B</td><td>8.00</td><td>0.417</td><td> ${ \bf 0 . 9 4 7 \pm 0 . 0 0 5 }$ </td><td>+0.530</td></tr></table>

Table 9: MRR@10 on held-out OTel evaluation splits. OTel fine-tuning improves reranking quality for all three reranker baselines. $\Delta$ is the absolute MRR@10 gain. Standard errors via bootstrap (n=1000).

![](images/5c1328e266f0acaa5c83440df4e7953e7fd7c082343beab90bda2057ecc7e462.jpg)  
Figure 1: OTel fine-tuning improves language-model correctness and embedding retrieval quality; larger LLMs generally raise the correctness upper envelope.

## 7 Limitations, Broader Impact, and Responsible Release

OTel is intentionally domain-specific. The reported results should not be generalized outside telecommunications, and even within telecom the current resource remains English-only and primarily text-centric. The OTel LLMs are designed for context-grounded RAG, which means they are not optimized for unrestricted open-ended QA. The main-paper evaluation also relies on held-out splits drawn from the released OTel resource family rather than a fully independent external benchmark suite, so future work should expand the amount of external testing and cross-benchmark transfer analysis.

The collaborator-led retrieval stress tests also indicate that aggregate scores hide important subdomain variation. O-RAN retrieval appears comparatively strong, while academic-paper and GSMA PRD examples remain weaker areas. Future releases should expand and re-clean the academic-paper subset, re-curate GSMA PRD examples with harder negatives and larger evaluation pools, and make per-subdomain reporting a first-class part of the OTel evaluation protocol.

Source coverage is broad but still imperfect. Some telecom subdomains are better represented than others, and the source material mixes standards, reference documents, whitepapers, academic papers, and web-derived data. That heterogeneity is valuable for realism, but it creates nontrivial provenance and redistribution work. For this reason, release documentation should describe source classes, intended use, and source-specific constraints rather than oversimplify the upstream licensing story.

The resource has clear positive value: it reduces barriers to open telecom experimentation, enables reproducible comparison across retrievers, rerankers, and generators, and offers a public starting point for telecom-specific model development. The main risks are misuse of domain-specialized models to generate misleading telecom content and over-trust in models outside their intended RAG setting. Therefore, abstention-oriented safety tuning, model cards, dataset cards, intended-use statements, and asset-level metadata are critical for responsible release.

## 8 Conclusion

OTel delivers the resource promised at the start of the paper: documented telecom data sources and tasks, a derived dataset family, held-out evaluation partitions, and Hugging Face reference baselines spanning retrieval, reranking, and context-grounded generation. The resource has already seen substantial community use, with more than 16 million model downloads and 157+ pieces of media coverage worldwide, and the released baselines provide reference points for the community to beat, adapt, and extend.

The results show strong headline performance: 93.1% NDCG@10 for retrieval, 0.947 MRR@10 for reranking, and 87.8% LLM correctness. OTel fine-tuning improves performance across embedding models, rerankers, and LLMs, and model scale helps raise the upper envelope while architecture still matters. The goal is not to claim that telecom is “solved,” but to provide an open and reproducible starting point for the community to expand the data, improve retrieval and reranking, build stronger context-grounded telecom LLMs, and extend OTel through multilingual, multimodal, per-subdomain, and external benchmark evaluations.

## Acknowledgment

We thank Faik Kerem Ors (Purdue University), Mashroor Hasan Bhuiyan (The University of Texas at Dallas), Roderic Paulk, Jorden Terrazas, and Kostikey Mustakas (AT&T); Molham Aref (RelationalAI); Enrique Molero (GSMA); Lina Bariah, Esraa Fahmy, and Bohao Wang (Khalifa University); Syed Ali Raza Zaidi, Maryam Hafeez, and Shehr Bano (University of Leeds); Alexander Finn, Andy Allred, Andrey Ivannikov, Antti-Ville Suni, Mark van Heeswijk, and Kumaran Siva (AMD); Matt Upson (Mantis NLP); and Vignesh Ethiraj and Ashwath David (NetoAI) for their contributions to the OTel project.

## References

Vincenzo Colle, Mohamed Sana, Nicola Piovesan, Antonio De Domenico, Fadhel Ayed, and Merouane Debbah. TeleMath: A benchmark for large language models in telecom mathematical problem solving. arXiv preprint arXiv:2506.10674, 2025.

Essential AI. Announcing Rnj-1: Building instruments of intelligence. https://essential.ai/research/ rnj-1, 2025.

Anas Ezzakri, Nicola Piovesan, Mohamed Sana, Antonio De Domenico, Fadhel Ayed, and Haozhe Zhang. TeleTables: A benchmark for large language models in telecom table interpretation. arXiv preprint arXiv:2601.04202, 2026.

Pranshav Gajjar and Vijay K. Shah. ORAN-Bench-13K: An open source benchmark for assessing LLMs in open radio access networks. arXiv preprint arXiv:2407.06245, 2024.

Neel Guha et al. LegalBench: A collaboratively built benchmark for measuring legal reasoning in large language models. In Advances in Neural Information Processing Systems 36 (Datasets and Benchmarks Track), 2023.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient have? A large-scale open domain question answering dataset from medical exams. Applied Sciences, 11(14):6421, 2021.

Imtiaz Karim, Kazi Samin Mubasshir, Mirza Masfiqur Rahman, and Elisa Bertino. SPEC5G: A dataset for 5G cellular network protocol analysis. In Findings ofthe Associationfor Computational Linguistics: IJCNLP-AACL 2023, pages 20–38, 2023. doi: 10.18653/v1/2023.findings-ijcnlp.3. URL https://aclanthology. org/2023.findings-ijcnlp.3/.

Ali Maatouk, Fadhel Ayed, Nicola Piovesan, Antonio De Domenico, Merouane Debbah, and Zhi-Quan Luo. TeleQnA: A benchmark dataset to assess large language models telecommunications knowledge. arXiv preprint arXiv:2310.15051, 2023.

Mohamed Sana, Nicola Piovesan, Antonio De Domenico, Yibin Kang, Haozhe Zhang, Merouane Debbah, and Fadhel Ayed. Reasoning language models for root cause analysis in 5G wireless networks. arXiv preprint arXiv:2507.21974, 2025.

Qianqian Xie et al. FinBen: A holistic financial benchmark for large language models. In Advances in Neural Information Processing Systems 37 (Datasets and Benchmarks Track), 2024.

Hang Zou, Qiyang Zhao, Yu Tian, Lina Bariah, Faouzi Bader, Thierry Lestable, and Merouane Debbah. TelecomGPT: A framework to build telecom-specific large language models. arXiv preprint arXiv:2407.09424, 2024.

## A Full Baseline Roster

## A.1 Language models

<table><tr><td>OTel model</td><td>Parameters</td><td>Base model</td></tr><tr><td>OTe1-LLM-270M-IT</td><td>270M</td><td>gemma-3-270m-it</td></tr><tr><td>OTe1-LLM-O.6B-IT</td><td>0.6B</td><td>Qwen3-0.6B</td></tr><tr><td>OTel-LLM-1B-IT</td><td>1B</td><td>gemma-3-1b-it</td></tr><tr><td>OTel-LLM-1.2B-IT</td><td>1.2B</td><td>LFM2.5-1.2B-Instruct</td></tr><tr><td>OTel-LLM-1.7B-IT</td><td>1.7B</td><td>Qwen3-1.7B</td></tr><tr><td>OTe1-LLM-3B-IT</td><td>3B</td><td>Mistral-3-3B</td></tr><tr><td>OTel-LLM-4B-IT</td><td>4B</td><td>gemma-3-4b-it</td></tr><tr><td>OTel-LLM-7B-IT</td><td>7B</td><td>OLMo-3-7B</td></tr><tr><td>OTe1-LLM-8.2B-IT</td><td>8.2B</td><td>Qwen3-8B</td></tr><tr><td>OTe1-LLM-8.3B-IT</td><td>8.3B</td><td>RNJ-1-Instruct</td></tr><tr><td>OTel-LLM-12B-IT</td><td>12B</td><td>gemma-3-12b-it</td></tr><tr><td>OTel-LLM-14B-IT</td><td>14B</td><td>Qwen3-14B</td></tr><tr><td>OTe1-LLM-20B-IT</td><td>20B</td><td>GPT-OSS-20B</td></tr><tr><td>OTel-LLM-20B-Reasoning</td><td>20B</td><td>GPT-OSS-20B</td></tr><tr><td>OTel-LLM-24B-IT</td><td>24B</td><td>LFM2-24B-A2B</td></tr><tr><td>OTe1-LLM-27B-IT</td><td>27B</td><td></td></tr><tr><td>OTe1-LLM-32B-IT</td><td>32B</td><td>gemma-3-27b-it OLMo-3-32B</td></tr></table>

Table 10: Language-model subset of the 30 released full-parameter post-trained OTel baselines.

## A.2 Embedding models

<table><tr><td>OTel model</td><td>Parameters</td><td>Base model</td></tr><tr><td>OTel-Embedding-22M</td><td>22M</td><td>al1-MiniLM-L6-v2</td></tr><tr><td>OTel-Embedding-33M</td><td>33M</td><td>BAAI/bge-small-en-v1.5</td></tr><tr><td>OTel-Embedding-34M</td><td>34M</td><td>al1-MiniLM-L12-v2</td></tr><tr><td>OTel-Embedding-109M</td><td>109M</td><td>all-mpnet-base-v2</td></tr><tr><td>OTel-Embedding-300M</td><td>300M</td><td>Gemma3-Embedding-300M</td></tr><tr><td>OTel-Embedding-335M</td><td>335M</td><td>BAAI/bge-large-en-v1.5</td></tr><tr><td>OTel-Embedding-568M</td><td>568M</td><td>BAAI/bge-m3</td></tr><tr><td>OTel-Embedding-600M</td><td>600M</td><td>Qwen3-Embedding-0.6B</td></tr><tr><td>OTel-Embedding-4B</td><td>4B</td><td>Qwen3-Embedding-4B</td></tr><tr><td>OTel-Embedding-8B</td><td>8B</td><td>Qwen3-Embedding-8B</td></tr></table>

Table 11: Embedding subset of the 30 released full-parameter post-trained OTel baselines.

## A.3 Reranker models

<table><tr><td>OTel model</td><td>Parameters</td><td>Base model</td></tr><tr><td>OTel-Reranker-0.6B</td><td>0.6B</td><td>Qwen3-0.6B</td></tr><tr><td>OTel-Reranker-4B</td><td>4B</td><td>Qwen3-4B</td></tr><tr><td>OTel-Reranker-8B</td><td>8B</td><td>Qwen3-8B</td></tr></table>

Table 12: Reranker subset of the 30 released full-parameter post-trained OTel baselines.

## B Training Datasets

The four released OTel datasets support the 30-model initial release plus appendix-only auxiliary safety analyses.   
In particular, OTel-Safety is used for the two abstention-focused variants discussed in Appendix H.

<table><tr><td>Model type</td><td>Dataset</td></tr><tr><td>LLM instruction tuning</td><td>OTel-LLM</td></tr><tr><td>LLM safety tuning</td><td>OTel-Safety</td></tr><tr><td>Embedding</td><td>OTel-Embedding</td></tr><tr><td>Reranker</td><td>OTel-Reranker</td></tr></table>

Table 13: Dataset mapping for the OTel training releases.

## C Training Hyperparameters and Optimization

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Optimizer</td><td>AdamW (8-bit, bitsandbytes)</td></tr><tr><td>Learning-rate schedule</td><td>Cosine decay with warmup</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Warmup steps</td><td>100</td></tr><tr><td>Per-device batch size</td><td>8-128</td></tr><tr><td>Gradient accumulation</td><td>4-32</td></tr><tr><td>Random seed</td><td>42</td></tr><tr><td>Maximum sequence length</td><td>1500</td></tr><tr><td>Training epochs (LLM, embedding)</td><td>3</td></tr><tr><td>Training epochs (reranker)</td><td>2</td></tr></table>

Table 14: Shared optimization template for the full-parameter post-trained OTel baselines.

## D Memory, Precision, and Distributed Training

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Numerical precision</td><td>BF16</td></tr><tr><td>Attention implementation</td><td>Flash Attention 2</td></tr><tr><td>Gradient checkpointing</td><td>Enabled</td></tr><tr><td>Distributed training</td><td>Fully Sharded Data Parallel</td></tr></table>

Table 15: Precision and distributed training configuration.

## E Compute Resources

<table><tr><td>Hardware</td><td>Count</td></tr><tr><td>AMD MI300X</td><td>32</td></tr><tr><td>AMD MI325X</td><td>64</td></tr><tr><td>AMD MI355X</td><td>32</td></tr><tr><td>NVIDIA A100</td><td>8</td></tr><tr><td>NVIDIA H100</td><td>32</td></tr></table>

Table 16: GPU pools used across OTel training runs.

## F Dataset Splits and Logging

<table><tr><td>Model type</td><td>Train</td><td>Eval</td><td>Seed</td></tr><tr><td>LLM</td><td>90%</td><td>10%</td><td>42</td></tr><tr><td>Embedding</td><td>90%</td><td>10%</td><td>42</td></tr><tr><td>Reranker</td><td>95%</td><td>5%</td><td>42</td></tr></table>

Table 17: Held-out split policy used in the current OTel evaluation setup.

<table><tr><td>Logging/configuration item</td><td>Value</td></tr><tr><td>Evaluation strategy</td><td>Per epoch</td></tr><tr><td>Save strategy</td><td>Per epoch</td></tr><tr><td>Logging interval</td><td>Every 50 steps</td></tr><tr><td>Experiment tracking</td><td>TensorBoard</td></tr><tr><td>LLM judge models</td><td>GPT-4o mini and Claude Sonnet 3.5</td></tr></table>

Table 18: Evaluation cadence and experiment logging defaults.

## G Loss Computation

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Loss masking</td><td>Completion tokens only</td></tr><tr><td>Label smoothing</td><td>None</td></tr><tr><td>Padding side</td><td>Right</td></tr></table>

Table 19: Shared loss-side configuration.

## H Additional Experiments Outside the Core 30-Model Release

The following experiments are useful auxiliary validations but are not part of the core 30-model OTel release described in the main paper.

Appendix-only safety variants. Two full-parameter post-trained abstention-focused variants, OTel-LLM-8.3B-Safety and OTel-LLM-12B-Safety, were trained on OTel-Safety. They were not included in the initial 30-model release, so they are discussed here as auxiliary models rather than counted baseline results.

TeleLogs classification. A classification head added to the rnj-1 base path reaches roughly 99% accuracy on the TeleLogs 5G root-cause-analysis benchmark [Sana et al., 2025]. The resulting classification model weights and training code are publicly released through the OTel Hugging Face and GitHub repositories.

GSMA QnA comparison. One non-abstention QnA variant, OTel-LLM-8.3B-QnA, was trained after the initial release for direct comparison on context-free telecom QA benchmarks. This model starts from the same base checkpoint (RNJ-1-Instruct) as OTel-LLM-8.3B-IT, but follows a separate training path: continued pretraining on telecom documents, followed by RL and supervised fine-tuning on a broad telecom QnA mixture that includes, but is not limited to, OTel-LLM. This auxiliary variant is separate from OTel-Safety and is designed to answer direct questions without requiring RAG context, making it compatible with the GSMA Open Telco AI Leaderboard evaluation protocol. It is intentionally separated from the main paper because it is not part of the core RAG-oriented OTel baseline family.

<table><tr><td>Model</td><td>AVG ↓</td><td>3GPP-TSG</td><td>ORANBench</td><td>srsRANBench</td><td>TeleLogs</td><td>TeleMath</td><td>TeleQnA</td><td>TeleTables</td></tr><tr><td>OTel-LLM-8.3B-QnA</td><td></td><td>86.0 81.4±0.9</td><td>94.1±0.6</td><td>89.7±0.8</td><td>96.3±0.6</td><td>87.4±1.5</td><td>91.2±0.3</td><td>61.8±2.2</td></tr><tr><td>2 gemini-3.1-pro-preview</td><td></td><td>75.6 70.0±4.6</td><td>86.0±2.8</td><td>84.7±3.0</td><td>82.0±3.9</td><td>73.0±4.5</td><td>85.2±1.1</td><td>48.0±5.0</td></tr><tr><td>3 gemini-3-pro-preview</td><td></td><td>74.7 66.0±4.8</td><td>83.3±3.1</td><td>81.3±3.2</td><td>86.0±3.5</td><td>78.0±4.2</td><td>82.9±1.2</td><td>45.0±5.0</td></tr><tr><td>4 = LTM</td><td>73.6</td><td>68.4±1.0</td><td>82.0±0.9</td><td>83.1±0.9</td><td>73.4±1.3</td><td>81.5±1.6</td><td>81.9±0.4</td><td>44.7±1.8</td></tr><tr><td>5 A claude-opus-4.6</td><td>73.3</td><td>66.0±4.8</td><td>90.0 ±2.5</td><td>84.7±3.0</td><td>70.0±4.6</td><td>75.0±4.3</td><td>84.4±1.1</td><td>43.0±5.0</td></tr><tr><td>6 gpt-5</td><td>71.9</td><td>58.0±5.0</td><td>86.0±2.8</td><td>81.3±3.2</td><td>78.0±4.2</td><td>79.0±4.1</td><td>83.8±1.2</td><td>37.0±4.9</td></tr><tr><td>7 o gemini-3-flash-preview</td><td>70.4</td><td>63.0±4.9</td><td>89.3±2.5</td><td>83.3±3.1</td><td>46.0±5.0</td><td>80.0±4.0</td><td>83.2±1.2</td><td>48.0±5.0</td></tr><tr><td>8 A claude-opus-4.5</td><td>69.6</td><td>64.0±4.8</td><td>89.3±2.5</td><td>82.7 ±3.1</td><td>56.0±5.0</td><td>72.0±4.5</td><td>84.5±1.1</td><td>39.0±4.9</td></tr><tr><td>9 kimi-k2.5</td><td>69.4</td><td>57.0±5.0</td><td>82.7±3.1</td><td>84.7±3.0</td><td>60.0±4.9</td><td>76.0±4.3</td><td>83.6±1.2</td><td>42.0±5.0</td></tr><tr><td>10 o3</td><td>69.4</td><td>59.0±4.9</td><td>84.7±3.0</td><td>78.7±3.4</td><td>72.0±4.5</td><td>74.0±4.4</td><td>83.4±1.2</td><td>34.0±4.8</td></tr></table>

Figure 2: Auxiliary GSMA Open Telco AI Leaderboard comparison for the non-abstention OTel-LLM-8.3B-QnA variant.

GPU utilization. Large-scale training runs reached 94.2% utilization across 256 AMD MI325X GPUs, as summarized in the accompanying utilization plot. We treat this as infrastructure evidence rather than a central E&D claim.

## I Asset Documentation and Release Metadata

The release package includes dataset cards, model cards, Croissant metadata files (with Responsible AI extensions), and reproduction instructions for the four datasets and 30 released full-parameter post-trained OTel baselines. All assets are publicly available on Hugging Face and GitHub. Each dataset repository includes a croissant.json file with validated Croissant 1.1 metadata and all required RAI fields (rai:dataLimitations, rai:dataBiases, rai:personalSensitiveInformation, rai:dataUseCases, rai:dataSocialImpact, rai:hasSyntheticData, and prov:wasGeneratedBy).

Public dataset and model cards should cite this paper as published at NeurIPS 2026 (Evaluations & Datasets Track).
<table><tr><td>Asset class</td><td>Release unit</td><td>Documentation</td><td>Metadata</td></tr><tr><td>Dataset</td><td>OTel-LLM</td><td>Dataset card</td><td>Croissant + RAI</td></tr><tr><td>Dataset</td><td>OTel-Safety</td><td>Dataset card</td><td>Croissant + RAI</td></tr><tr><td>Dataset</td><td>OTel-Embedding</td><td>Dataset card</td><td>Croissant + RAI</td></tr><tr><td>Dataset</td><td>OTel-Reranker</td><td>Dataset card</td><td>Croissant + RAI</td></tr><tr><td>Model family</td><td>30 released OTel baselines</td><td>Model cards</td><td>Reproduction instructions</td></tr><tr><td>Code</td><td>Training/evaluation scripts</td><td>README + commands</td><td>Environment specification</td></tr></table>

Table 20: Release artifacts and their accompanying documentation.

Source provenance and licensing. All source documents used in the OTel dataset are publicly available and free to download. The OTel datasets release only derived QA pairs, not the raw source documents, under the Apache-2.0 license.

![](images/a2072d1602ed5e387e860a356fcffe7209616d12317dcd1a836e16da07ff28d4.jpg)  
Figure 3: Auxiliary GPU utilization analysis kept in the appendix rather than the main paper.

<table><tr><td>Source class</td><td>Sources</td><td>Availability</td><td>OTel release</td></tr><tr><td>Standards body</td><td>3GPP specifications</td><td>Public</td><td>Derived QA pairs, Apache- 2.0</td></tr><tr><td>Consortium documents</td><td>GSMA PRDs, Discover portal</td><td>Public</td><td>Derived QA pairs, Apache- 2.0</td></tr><tr><td>O-RAN documentation</td><td>O-RAN Alliance specifications</td><td>Public</td><td>Derived QA pairs, Apache- 2.0</td></tr><tr><td>Open technical corpora</td><td>IETF RFC series</td><td>Public</td><td>Derived QA pairs, Apache- 2.0</td></tr><tr><td>Academic papers</td><td>arXiv telecom papers</td><td>Public (CC-BY)</td><td>Derived QA pairs, Apache-</td></tr><tr><td>Web-derived</td><td>Common Crawl telecom pages</td><td>Public</td><td>2.0 Derived QA pairs, Apache-</td></tr><tr><td>Encyclopedia</td><td>Wikipedia telecom articles</td><td>Public (CC-BY-</td><td>2.0 Derived QA pairs, Apache-</td></tr><tr><td>Industry whitepapers</td><td>Telecom vendor whitepapers</td><td>SA) Public</td><td>2.0 Derived QA pairs, Apache- 2.0</td></tr></table>

Table 21: Source provenance and redistribution status for the OTel dataset family.

## J Worldwide Media Coverage of the Open Telco AI Project

The project has received worldwide media coverage across telecom trade press, general technology outlets, regional business press, and syndicated pickups. The maintained source-of-truth list is provided in the supplemental appendix file OTel\_Media\_Coverage\_List.md, which enumerates the currently tracked outlets, regions, dates, headlines, and links supporting the media-coverage count cited in the main paper.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

## Answer: [Yes]

Justification: The abstract and introduction state that the contribution is a telecom evaluation resource consisting of curated datasets, 30 released full-parameter post-trained OTel baselines, and a reproducible evaluation setup; see Sections 1–2.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Section 7 discusses telecom-only scope, English-only coverage, text-centric evaluation, RAG-specific design choices, and licensing and redistribution constraints.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A]

Justification: The paper does not present new theoretical results or proofs.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Sections 3–6 and Appendices A–I describe dataset construction, train/eval splits, model families, optimization settings, compute resources, and release artifacts needed to reproduce the reported experiments.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

## Answer: [Yes]

Justification: The paper describes release of OTel datasets, final model weights, evaluation code, reproduction scripts, and accompanying cards/metadata; see Appendix I.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: Section 5 gives the held-out split policy and metrics, while Appendices C–G provide optimization, precision, distributed training, compute, loss masking, and logging details.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: All result tables (Tables 3–5) report standard errors computed via bootstrap resampling (n=10). The standard errors capture variability across bootstrap samples of the held-out evaluation partitions.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes]

Justification: Appendix E summarizes the GPU pools used across AMD and NVIDIA hardware, and Sections 4–5 explain the common training configuration and evaluation cadence.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

## Answer: [Yes]

Justification: The work uses public technical sources, documents limitations and misuse risks, and discusses responsible release considerations in Section 7 and Appendix I.

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

## Answer: [Yes]

Justification: Section 7 discusses both the potential benefits of open telecom evaluation resources and the risks of misuse or misleading generated telecom content.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

## Answer: [Yes]

Justification: The paper describes abstention-focused safety training, intended-use restrictions, and planned asset documentation and release metadata in Sections 4 and 7 and Appendix I.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

## Answer: [Yes]

Justification: Appendix I provides a source-by-source provenance and licensing table. All source documents are publicly available. The OTel datasets release only derived QA pairs under Apache-2.0, not raw source documents. Base models are credited with citations and their licenses are respected.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: The paper introduces four dataset releases and 30 released full-parameter post-trained OTel baselines. Each dataset includes a dataset card, a validated Croissant 1.1 metadata file with all required Responsible AI fields, and reproduction documentation; see Appendix I.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

## Answer: [N/A]

Justification: The paper does not involve crowdsourcing experiments or research with human subjects.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

## Answer: [N/A]

Justification: The paper does not involve human subjects research.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

## Answer: [Yes]

Justification: Section 3 describes LLM-assisted answer recreation and fact-grounded verification during data preparation, which is a non-standard methodological component of the resource construction pipeline.