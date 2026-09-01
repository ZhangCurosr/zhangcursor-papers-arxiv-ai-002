# When Does Bigger Help? A Controlled Study of LLM Scale for Ontology Learning

Hamed Babaei Giglou<sup>1</sup>, Sören Auer<sup>1,2</sup> and Jennifer D’Souza<sup>1</sup>

<sup>1</sup>TIB Leibniz Information Centre for Science and Technology, Hannover, Germany

<sup>2</sup>L3S Research Center, Leibniz University ofHannover, Hannover, Germany

## Abstract

The efect of Large Language Model (LLM) scale on ontology learning (OL) performance remains insuficiently characterized. We present a controlled evaluation of 13 models spanning dense and Mixture-of-Experts variants from the Qwen3.5 and Qwen3.6 lineages, together with proprietary GPT release variants, using the OntoLearner retrieval-augmented generation pipeline. All models are evaluated with the same embedding model, retrieval configuration, prompt templates, decoding settings, datasets, and metrics on term typing, taxonomy discovery, and non-taxonomic relationship extraction across four biomedical and materials science and engineering ontologies. Within the dense Qwen3.5 lineage, increasing parameter count primarily improves precision rather than recall, with the largest gains occurring between 9B and 27B parameters. However, the efect of scale is neither monotonic nor uniform across tasks and domains. Dense 27B models outperform substantially larger sparse models on term typing, whereas larger Mixture-of-Experts models achieve the strongest open-weight results on taxonomy discovery. Non-taxonomic relationship extraction remains dificult across model scales, particularly for the Materials Data Science ontology. Performance diferences across matched Qwen variants and proprietary GPT releases further indicate that architecture and model lineage can outweigh nominal parameter count. These findings show that model size alone is an insuficient selection criterion for OL and provide empirical guidance for reproducible LLM-assisted ontology engineering.

## Keywords

Large Language Models, Ontology Learning, OntoLearner, Retrieval-Augmented Generation, Benchmarking

<sup>•</sup> URL: https://ontolearner.readthedocs.io/

<sup>•</sup> GitHub: https://github.com/sciknoworg/OntoLearner

<sup>•</sup> PyPi: https://pypi.org/project/OntoLearner/

<sup>•</sup> Huggingface: https://huggingface.co/collections/SciKnowOrg/ontolearner-benchmarking

<sup>•</sup> License: MIT License

<sup>•</sup> Experimental Resources: https://doi.org/10.5281/zenodo.21719027

## 1. Introduction

Scientific research is becoming increasingly data- and automation-intensive, particularly in biomedicine and materials science, where high-content screening, automated experimentation, and self-driving laboratories generate growing volumes of heterogeneous data, experimental procedures, and domain terminology [1, 2]. For these outputs to be reused across laboratories, databases, and computational systems, they must be represented in forms that are not only digitally accessible but also findable, interoperable, and machine-actionable [3]. Structured knowledge representations are therefore essential for semantic search, cross-source integration, reproducible analysis, scientific reasoning, and AI-assisted discovery [4, 5, 6]. Ontologies provide the formal conceptual layer needed to organize domain entities and relationships consistently across heterogeneous sources. However, constructing and maintaining high-quality ontologies remains a labor-intensive and expert-driven process, limiting the scalability, adaptability, and reproducibility of ontology engineering [7].

Recent advances in Large Language Models (LLMs) present new opportunities for automating ontology learning (OL) tasks [8]. Given lexical terms and ontology-grounded context, LLMs can support assigning terms to semantic types, identifying taxonomic relationships between concepts, and predicting non-taxonomic relations. Their ability to perform these tasks with little or no taskspecific training makes LLM-based OL a promising approach for reducing the manual efort involved in ontology construction, extension, and maintenance, and for supporting the scalable development of knowledge graphs across scientific domains. The first LLMs4OL study established an empirical foundation for investigating LLMs in OL by evaluating a broad selection of model families across term typing, taxonomy discovery, and non-taxonomic relation extraction [8]. Its primary objective was to determine whether general-purpose and domain-specific language models could perform these tasks at all. Accordingly, it compared models with diferent architectures, training objectives, domain specializations, and parameter counts, including BERT [9], BART [10], Flan-T5 [11], BLOOM [12], LLaMA [13], GPT [14], and PubMedBERT [15]. Comparisons between selected Flan-T5 and BLOOM variants provided initial indications that larger models could sometimes perform better, while taskspecific fine-tuning showed that smaller models could also outperform substantially larger ones. Model size was therefore not isolated from model family, architecture, training strategy, or domain adaptation.

The LLMs4OL Challenges, organized at the International Semantic Web Conference (ISWC), subsequently broadened this investigation through shared tasks, datasets, evaluation phases, and metrics for community-developed OL systems [16, 17]. In the 2024 challenge, participants explored fine-tuning and knowledge-enhanced prompt-tuning [18, 19], prompt-tuning across multiple LLMs [20], retrievalaugmented generation [21], hybrid LLM–rule-based processing [22], and conventional machine-learning methods combined with LLMs [23]. The 2025 challenge expanded this methodological range through systems combining prompt engineering and embeddings [24], RAG, few-shot prompting, embedding ensembles, and lightweight fine-tuning [25], clustering and retrieval-augmented extraction [26], data augmentation and candidate filtering [27], and multi-model deliberation [28]. Some submissions also compared diferently sized models [29, 30], but these comparisons generally varied model family, architecture, adaptation strategy, or system configuration alongside size. They were consequently suited to identifying efective OL systems, but not to estimating the independent efect of parameter scale. This leaves a fundamental question unanswered:

This gap reflects a broader and still unresolved debate in foundation-model research concerning whether larger LLMs are systematically more capable. Early neural scaling-law studies identified smooth power-law reductions in pretraining loss as model parameters, training data, and computational resources increased [31]. Subsequent compute-optimal analyses, however, demonstrated that parameter count alone is an incomplete indicator of capability: a smaller model trained on substantially more data can outperform a much larger but under-trained model [32]. Model scale must therefore be understood in relation to training data, compute, and architecture rather than through parameter count alone. The relationship becomes even less predictable at the downstream-task level. Some capabilities have been reported to appear only after models exceed particular scale thresholds [33], whereas other work has shown that apparently discontinuous emergence may partly result from the choice of evaluation metric [34]. Inverse and U-shaped scaling have also been observed, whereby performance can plateau, deteriorate, or change direction as models grow [35, 36]. A recent re-examination similarly found smooth and predictable scaling in only a minority of downstream-task settings [37]. Improvements in pretraining objectives or aggregate benchmarks therefore cannot be assumed to transfer monotonically to specialized knowledge-engineering tasks.

This uncertainty is especially consequential for OL because its constituent tasks impose diferent semantic and structural demands. Term typing requires calibrated selection among candidate ontology classes, taxonomy discovery requires reasoning over hierarchical relations, and non-taxonomic relationship extraction requires distinguishing heterogeneous domain relations. Increasing scale may therefore afect precision, recall, and relational reasoning diferently across tasks and domains. Furthermore, “model size” is not entirely unambiguous across architectures. Dense models activate their full parameter set, whereas Mixture-of-Experts (MoE) models separate total model capacity from the smaller number of parameters activated for an individual input [38]. Architecture, training composition, instruction tuning, and alignment may consequently mediate or outweigh the efect of nominal parameter count.

Although existing OL studies indicate that model choice and capacity matter, they do not isolate these factors through a suficiently broad within-family scale sweep under fixed experimental conditions. We therefore investigate the following research questions:

RQ1: Within-family scaling. How does increasing parameter count within a common dense model family afect precision, recall, and F1 performance in $\mathrm { O L } ,$ and are the resulting improvements monotonic?

RQ2: Task and domain sensitivity. How does the efect of model scale vary across OL tasks and across scientific domains with diferent conceptual structures?

RQ3: Architecture and model lineage. How do dense versus MoE architectures and diferences across Qwen and GPT release lineages afect OL performance beyond nominal parameter count?

We address these questions through a controlled benchmark study using OntoLearner [39], a reproducible ecosystem for LLM-based OL. We evaluate 13 models spanning the Qwen3.5 [40], Qwen3.6 [41, 42], and proprietary GPT lineages. The Qwen3.5 models enable a within-family densemodel scale sweep from 0.8B to 27B parameters, complemented by larger sparse MoE variants with 35B total and 3B active parameters and 122B total and 10B active parameters. The Qwen3.6 models permit matched-scale comparisons with Qwen3.5 across a 27B dense model and a 35B-A3B MoE model, while GPT-5.5 [43] and the GPT-5.6 Luna [44], Terra [45], and Sol [46] variants provide a comparison across proprietary model releases. All models are evaluated on term typing, taxonomy discovery, and non-taxonomic relationship extraction over biomedical and materials science and engineering ontologies. By holding the embedding model, retrieval configuration, retrieved context, prompt templates, decoding conditions, datasets, and evaluation metrics constant, we examine the efects of parameter scale, dense versus sparse architecture, model release, task structure, and domain characteristics. The Results section is organized according to the three research questions and provides empirical guidance for model selection in scalable and reproducible ontology engineering.

## 2. Preliminaries

Following the OL formalization introduced in [8] and adopted by the LLMs4OL Challenge, we consider three fundamental OL tasks. Let � denote a lexical term extracted from text and � denote a type (i.e., an ontology class). These tasks constitute the basis of our experimental evaluation.

Term Typing. Term typing aims to assign the most appropriate ontology type to a lexical term. Formally,

$$
O L _ { \mathrm { t e r m  t y p i n g } } ( L ) : = ( L , T )\tag{1}
$$

where the input is a lexical term $L ,$ and the output is its corresponding ontology type �.

Taxonomy Discovery. Taxonomy discovery identifies hierarchical (is-a) relationships between ontology types. Given two lexical terms, their corresponding ontology types are inferred, and the taxonomic relationship between them is determined. Formally,

$$
O L _ { \mathrm { t a x o n o m y d i s c o v e r y } } ( L _ { a } , L _ { b } ) : = ( T _ { a } , T _ { b } )\tag{2}
$$

where $T _ { a }$ and $T _ { b }$ denote the ontology types associated with lexical terms $L _ { a }$ and $L _ { b ; }$ , respectively.

Non-Taxonomic Relationship Extraction. This task identifies semantic relations other than is-a between ontology types. Given a head lexical term $h ,$ a relation $^ { r , }$ and a tail lexical term $t ,$ the objective is to infer the semantic relation between their corresponding ontology types. Formally,

$$
O L _ { \mathrm { n o n - t a x o n o m i c ~ r e } } ( h , r , t ) : = ( T _ { h } , r , T _ { t } ) ,\tag{3}
$$

where $T _ { h }$ and $T _ { t }$ are the ontology types corresponding to the head and tail lexical terms, respectively.

## 3. Related Work

OL has long been recognized as a bottleneck-relieving alternative to manual ontology engineering, and several recent surveys examine how this bottleneck manifests in specific application domains. Ivanova and Terzieva [47] systematically review OL techniques for intelligent e-learning environments, mapping classical linguistic, statistical, and logic-based OL methods onto educational data sources and highlighting that LLM-based OL for education remains comparatively under-studied despite the maturity ofgeneral-purpose OL research. Their review echoes a broader pattern in the literature: domain adaptation of OL techniques is frequently treated as a secondary concern relative to the development of the underlying extraction methods themselves.

LLM-Based Ontology Generation Pipelines. A growing body of work investigates ontology construction directly from natural language using LLMs. Lippolis et al. [48] introduce two prompting techniques, Memoryless CQbyCQ and Ontogenia, for generating OWL drafts from competency questions and user stories, finding that reasoning-oriented models (e.g., o1-preview) can approach or exceed novice human ontology engineers on structural correctness, while open-weight models like Llama produce more superfluous elements and critical modeling pitfalls. In a related study, Lippolis et al. [49] assess the domain-generalizability of two reasoning-capable LLMs (DeepSeek and o1-preview) across six ontology-engineering projects and find remarkably consistent performance across domains, suggesting that ontology-generation ability is not narrowly tied to domain representation in pre-training data. Teplyi and Dosyn [50] propose a four-stage prompt-guided agent (schema discovery, instance extraction, self-repair validation, ontology alignment) that iteratively enforces domain-range consistency, achieving fact-recall competitive with specialized knowledge graph extraction pipelines such as KGGen while additionally producing schema-consistent output. Their comparison across GPT-4.1-mini, LLaMA-3.3- 70b, and Grok-3-mini shows that larger models tend to hypothesize broader schemas that remain partly unpopulated, whereas smaller models produce more compact but fully instantiated ontologies, an early signal of the scale-dependent trade-ofs that our work investigates more systematically. Moreover, Fathallah et al. [51] extend this line of work with NeOn-GPT, a NeOn-methodology-grounded pipeline incorporating an explicit ontology-reuse stage and automated verification (RDFLib, HermiT/Pellet, OOPS!). Evaluated across four heterogeneous domains using GPT-4o, Mistral, Llama-4, and DeepSeek, they report that all models default to shallow, single-inheritance hierarchies regardless of architecture or parameter count, a “systematic generative limitation” rather than a model-specific one, and that entitylevel semantic alignment consistently exceeds relation-level alignment. Zhang et al. [52] similarly combine LLM-driven extraction (via ChatGPT and Llama) with graph-database backends (OLIVE), emphasizing iterative human-in-the-loop correction of hallucinated relations validated through OOPS! and Protégé reasoning.

Fine-Tuning versus Prompting and Model Comparisons. Several works directly compare model families or adaptation strategies for OL, providing precedent for our size-focused investigation. Doumanas et al. [53] fine-tune GPT-4 and Mistral 7B on curated ontology-engineering textbook corpora, finding that GPT-4 achieves higher precision and syntactic fidelity at greater computational cost, while Mistral 7B is cheaper and faster but degrades unpredictably across successive fine-tuning rounds, underscoring that model family and scale interact with training strategy in ways that are not yet well characterized. Liu et al. [54] compare pre-trained prompting, in-context learning, and fine-tuning for casting domain term and relation extraction, finding fine-tuning superior in F1 but noting the substantial annotation cost required to enable it. Cappelli and Di Marzo Serugendo [55] instead rely on a single frontier model (ChatGPT-4o) guided by a structured “guiding table” of prompts, reporting high logical consistency but persistently flat, low-attribute-richness taxonomies—again suggesting that prompting strategy alone cannot fully compensate for structural limitations that may be tied to model capacity. None of these studies, however, isolate model size as an independent variable while holding domain, prompting strategy, and evaluation protocol constant.

Comparison of representative LLM-based OL studies. ✓/✗ indicate whether a size was explicitly varied or controlled for.
<table><tr><td>Study</td><td>Model(s)</td><td>Approach</td><td>Domain(s)</td><td>Cross- Domain</td><td>Scale Controlled</td><td>Key Finding</td></tr><tr><td>Lippolis et al. [48]</td><td>o1-preview, GPT-4, Llama</td><td>Prompt (CQbyCQ,</td><td>10 mixed on- tologies</td><td>√</td><td>x</td><td>Reasoning models rival novice engineers; Llama yields more critical pitfalls</td></tr><tr><td>Lippolis et al. [49]</td><td>DeepSeek, o1-preview</td><td>Ontogenia) Prompt</td><td>6 domains</td><td>√</td><td>x</td><td>Performance is consistent across domains, not tied to pretraining coverage</td></tr><tr><td>Teplyi &amp; Dosyn [50]</td><td>GPT-4.1-mini, LLaMA- 3.3-70B, Grok-3-mini</td><td>Agentic + self- repair</td><td>General text</td><td>x</td><td>Partial†</td><td>Larger models hypothesize broader, unpop- ulated schemas; smaller models produce</td></tr><tr><td>Fathallah et al. [51]</td><td>GPT-40, Mistral, Llama-4, DeepSeek</td><td>Prompt + reuse</td><td>4 domains</td><td>√</td><td>x</td><td>compact, fully instantiated ones All models default to shallow hierarchies</td></tr><tr><td>Zhang et al. [52]</td><td>ChatGPT, Llama</td><td>Prompt + HITL</td><td>General (KG)</td><td>x</td><td>x</td><td>regardless of scale or architecture Human-in-the-loop correction still re- quired to fix hallucinated relations</td></tr><tr><td>Doumanas et al. [53]</td><td>GPT-4, Mistral 7B</td><td>Fine-tune</td><td>OE textbooks</td><td>x</td><td>Partial†</td><td>Fine-tuning benefit is model-family- dependent; Mistral degrades across</td></tr><tr><td>Liu et al. [54]</td><td>GPT-4.1-mini</td><td>Prompt vs. ICL vs. FT</td><td>Casting</td><td>x</td><td>x</td><td>rounds Fine-tuning yields best F1 but at substan- tial annotation cost</td></tr><tr><td>Cappelli &amp; Seru- gendo [55]</td><td>ChatGPT-4o</td><td>Prompt (guiding table)</td><td>1 domain</td><td>x</td><td>x</td><td>High logical consistency but flat, low- attribute-richness taxonomies</td></tr><tr><td>da Cruz et al. [56]</td><td>LLaMA-3.2-3B</td><td>RAG</td><td>General</td><td>x</td><td>x</td><td>DB-derived ontologies rival GraphRAG at lower LLM inference cost</td></tr><tr><td>Lippolis et al. [57]</td><td>GPT-5, o1-preview</td><td>RAG</td><td>General</td><td>x</td><td>X</td><td>Extension proposals need only minor ex- pert revision before reuse</td></tr><tr><td>Cherian et al. [58]</td><td>unspecified</td><td>OL for FHIR</td><td>Healthcare</td><td>x</td><td>x</td><td>OL quality directly affects downstream in- teroperability outcomes</td></tr></table>

<sup>†</sup>Compares multiple models/families but not multiple sizes within a single family under a controlled protocol.

OL for Knowledge Graphs, Retrieval, and Specialized Domains. A parallel thread examines OL as an upstream component of larger systems rather than an end in itself. da Cruz et al. [56] show that ontology-guided knowledge graphs—particularly those derived from relational-database schemas—achieve retrieval performance competitive with GraphRAG at substantially lower LLMinference cost than text-derived alternatives, while Lippolis et al. [57] target the complementary problem of requirement-driven ontology extension rather than generation from scratch, showing via expert review that RAG-grounded extension proposals require only minor revision before production use. A similar topic was in the middle of interest to the LLMs4OL challenge community this year <sup>1</sup>. Domain-specific deployments further illustrate OL’s practical stakes: an ICMHI ’25 study [58] applies OL models to generate custom FHIR extensions for healthcare data interoperability, underscoring how OL quality directly afects downstream system usability in high-stakes settings.

As Table 1 makes clear, existing work has compared diferent LLM families (GPT vs. Mistral vs. Llama vs. DeepSeek) and, in a few cases, informally noted size-related trends as a byproduct of family comparisons [50, 53], but no study isolates model scale as an independent variable within a single family while holding domain, prompting strategy, and evaluation protocol fixed. Likewise, cross-domain evaluation is common, but it is rarely paired with a scale sweep: studies that vary domain [49, 51] do not vary size, and the one study that varies size across a handful of models [50] does not control for domain or family. This leaves open a basic practical question for anyone using LLM-based OL: does scaling within a modelfamily reliably improve OL quality, and is the marginal gain worth the added computational cost? We address this gap by using the LLMs4OL task formalization (term typing, taxonomy discovery, non-taxonomic relation extraction) together with the OntoLearner benchmarking ecosystem [39] to run a controlled scale sweep across the most recent models from Qwen and GPT model families, evaluated consistently on both biomedical and materials-science ontologies.

## 4. Methodology

To evaluate how model parameter scale impacts performance across core OL tasks defined in section 2, we adopt a Retrieval-Augmented Generation (RAG) framework adapted from the OntoLearner infrastructure [39] <sup>2</sup>. Our pipeline decouples context acquisition from generation by pairing a static dense retriever with the target LLM. This setup isolates the generation capabilities of each LLM scale variant under uniform retrieval conditions. The overall architecture consists of three sequential modules: (1) Semantic Vector Indexing, (2) Similarity-Based Retrieval, and (3) Task-Specific LLM Prompting and Generation.

Semantic Vector Indexing. Before running inference, we preprocess the underlying target ontology � into a searchable vector index. The representations are encoded into dense numerical vectors using a frozen feature extractor $E ( \cdot )$ . Specifically, we deploy Qwen3-Embedding-4B [59] <sup>3</sup> across all experimental setups to project all target ontological entities into a shared high-dimensional vector space: $\mathbf { v } _ { i } = E ( \operatorname { t e x t } ( c _ { i } ) ) \in \mathbb { R } ^ { d }$ . The resulting embeddings are stored in a dense vector database indexed for fast nearest-neighbor lookups using cosine similarity.

Similarity-Based Retrieval. During inference, an input candidate query � (e.g., a candidate lexical term � or a candidate pair of terms) is converted into an embedding using the same feature extractor $\mathbf { q } = E ( \mathop { \mathrm { t e x t } } ( q ) )$ ). Rather than relying on iterative multi-hop expansion or retriever-fine-tuning, we employ a single-pass dense retrieval strategy. The system computes the cosine similarity between the query vector q and all entity vectors $\mathbf { v } _ { i }$ in the pre-indexed ontology base. We retrieve the top-� nearest ontological entities $\mathcal { C } _ { q } = \{ c _ { 1 } , c _ { 2 } , \ldots , c _ { k } \}$ that yield the highest alignment scores Sim $\begin{array} { r } { ( \mathbf q , \mathbf v _ { i } ) = \frac { \mathbf q \cdot \mathbf v _ { i } } { \| \mathbf q \| _ { 2 } \| \mathbf v _ { i } \| _ { 2 } } } \end{array}$ For all experiments in this study, we fix $k = 1 0$ . The textual metadata associated with these top-� entities forms the ground-truth reference context injected into the downstream prompt.

Task-Specific LLM Generation. The final phase constructs a structured, task-specific instruction prompt combining three distinct elements: (1) a system role specification defining the target OL task rules, (2) the retrieved top-� background ontological entities $\mathcal { C } _ { q }$ serving as candidate grounded choices, and (3) the specific target input (e.g., lexical term � for Term Typing or term pair $( L _ { a } , L _ { b } )$ for Taxonomy Discovery). The complete prompt is passed to the target LLM under zero-shot conditions with identical decoding hyperparameters across model sizes. To guarantee reliable evaluation, the model is constrained to generate structured output conforming to the expected formal schema for each task:

<sup>•</sup> Term Typing: The LLM selects the single most accurate ontological class � from the candidate set $\mathcal { C } _ { q }$ that encapsulates the lexical term $L .$

<sup>•</sup> Taxonomy Discovery: The LLM determines whether a direct subsumption relationship (is-a) exists between the concepts associated with $L _ { a }$ and $L _ { b }$ within the retrieved context.

• Non-Taxonomic Relationship Extraction: Given head term $h ,$ candidate relation $^ { r , }$ and tail term $t ,$ the LLM predicts whether the relation holds based on the provided domain context.

Table 2  
LLMs evaluated in this work from the Qwen3.5, Qwen3.6, and GPT families.
<table><tr><td>Model</td><td>Family</td><td>Architecture</td><td>Scale</td><td>Availability</td><td>Model Page</td></tr><tr><td>Qwen3.5-0.8B</td><td>Qwen3.5</td><td>Dense</td><td>0.8B</td><td>Open</td><td>https://huggingface.co/Qwen/Qwen3.5-0.8B</td></tr><tr><td>Qwen3.5-2B</td><td>Qwen3.5</td><td>Dense</td><td>2B</td><td>Open</td><td>https://huggingface.co/Qwen/Qwen3.5-2B</td></tr><tr><td>Qwen3.5-4B</td><td>Qwen3.5</td><td>Dense</td><td>4B</td><td>Open</td><td>https://huggingface.co/Qwen/Qwen3.5-4B</td></tr><tr><td>Qwen3.5-9B</td><td>Qwen3.5</td><td>Dense</td><td>9B</td><td>Open</td><td>https://huggingface.co/Qwen/Qwen3.5-9B</td></tr><tr><td>Qwen3.5-27B</td><td>Qwen3.5</td><td>Dense</td><td>27B</td><td>Open</td><td>https://huggingface.co/Qwen/Qwen3.5-27B</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>Qwen3.5</td><td>MoE</td><td>35B (3B active)</td><td>Open</td><td>https://huggingface.co/Qwen/Qwen3.5-35B-A3B</td></tr><tr><td>Qwen3.5-122B-A10B</td><td>Qwen3.5</td><td>MoE</td><td>122B (10B active)</td><td>Open</td><td>https://huggingface.co/Qwen/Qwen3.5-122B-A10B</td></tr><tr><td>Qwen3.6-27B</td><td>Qwen3.6</td><td>Dense</td><td>27B</td><td>Open</td><td>https://huggingface.co/Qwen/Qwen3.6-27B</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>Qwen3.6</td><td>MoE</td><td>35B (3B active)</td><td>Open</td><td>https://huggingface.co/Qwen/Qwen3.6-35B-A3B</td></tr><tr><td>GPT-5.5</td><td>GPT</td><td>Proprietary</td><td></td><td>Closed</td><td>https://developers.openai.com/api/docs/models/gpt-5.5</td></tr><tr><td>GPT-5.6-Luna GPT-5.6-Terra</td><td>GPT GPT</td><td>Proprietary</td><td></td><td>Closed</td><td>https://developers.openai.com/api/docs/models/gpt-5.6-luna</td></tr><tr><td>GPT-5.6-Sol</td><td>GPT</td><td>Proprietary</td><td></td><td>Closed</td><td>https://developers.openai.com/api/docs/models/gpt-5.6-terra</td></tr><tr><td></td><td></td><td>Proprietary</td><td></td><td>Closed</td><td>https://developers.openai.com/api/docs/models/gpt-5.6-sol</td></tr></table>

Table 3

Selected ontological datasets ground truth statistics. The Terms No. is the number of terms extracted from the given ontology, whereas the Types No. is the number of classes, and Relations No. is the number of non-is-a relations.
<table><tr><td>Domain</td><td>Ontology</td><td>Terms No.</td><td>Types No.</td><td>Relations No.</td><td>Term Typing</td><td>Taxonomy Discovery</td><td>Non-Taxonomic RE</td></tr><tr><td>Material Science and Engineering</td><td>MatWerk [60]</td><td>29</td><td>313</td><td>2</td><td>29</td><td>369</td><td>12</td></tr><tr><td rowspan="2">Medical</td><td>MDS [61]</td><td>-</td><td>297</td><td>3</td><td>-</td><td>351</td><td>128</td></tr><tr><td>OBI [62]</td><td>53</td><td>4,934</td><td>3</td><td>53</td><td>2,368</td><td></td></tr><tr><td></td><td>MFOEM [63, 64]</td><td>19</td><td>621</td><td>2</td><td>19</td><td>837</td><td>20</td></tr></table>

By keeping the embedding model, vector index, top-� parameter (� = 10), and prompt templates strictly constant across all runs, any observed variation in task performance directly reflects the impact of model parameter scale and underlying architecture.

## 5. Experimental Setup

Large Language Models. We evaluate a diverse set of open- and closed-source LLMs spanning multiple parameter scales and architectural designs. The benchmark includes dense and Mixture-of-Experts (MoE) variants from the Qwen3.5 and Qwen3.6 families, as well as frontier OpenAI GPT models, enabling a systematic investigation of how model size and architecture influence OL performance. Table 2 summarizes the evaluated models and their key characteristics, where all are supported by OntoLearner for reproducible experimentation and benchmarking.

Datasets. For experimental purposes, we select four established ontologies from two scientific domains: materials science and engineering and the biomedical domain. These domains are selected to evaluate LLM-scale efects across heterogeneous scientific areas with diferent conceptual structures and levels of complexity. The selected ontologies include MatWerk [60] and MDS [61] from the materials science domain, and OBI [62] and MFOEM [63, 64] from the biomedical domain. For each ontology, we used OntoLearner benchmark collections hosted in Hugging Face <sup>4</sup> to construct evaluation tasks covering term typing, taxonomy discovery, and non-taxonomic relationship extraction. Table 3 summarizes the ground truth statistics of the selected datasets. Moreover, to manage experiments, we used the OntoLearner train\_test\_split strategy to preserve only 20% of the ontological data for OBI experiments.

## 6. Results

The experimental results for medical ontologies are presented in Table 4, and for material science and engineering domain ontologies are represented in Table 5 for three core LLMs4OL paradigm tasks using precision, recall, and F1-scores. Overall, the empirical evaluation reveals three insights: (1) increasing

## Table 4

Results on the medical ontologies.
<table><tr><td rowspan="3">LLMs</td><td colspan="9">MFOEM</td><td colspan="6">OBI</td></tr><tr><td colspan="2">Term Typing</td><td colspan="2"></td><td colspan="2">Taxonomy Discovery</td><td colspan="2">Non-Taxonomic RE</td><td colspan="2"></td><td colspan="2">Term Typing</td><td colspan="2">Taxonomy Discovery</td></tr><tr><td>Prec</td><td>Rec</td><td>F1</td><td>Prec</td><td>Rec</td><td>F1</td><td>Prec</td><td>Rec</td><td>F1</td><td>Prec</td><td>Rec</td><td>F1</td><td>Prec Rec</td><td>F1</td></tr><tr><td>Qwen3.5-0.8B</td><td>25.00</td><td>100.00</td><td>40.00</td><td>4.84</td><td>61.82</td><td>8.98</td><td>3.91</td><td>75.00</td><td>7.43</td><td>9.06</td><td>90.57</td><td>16.47</td><td>3.29</td><td>52.24</td><td>6.19</td></tr><tr><td>Qwen3.5-2B</td><td>25.00</td><td>100.00</td><td>40.00</td><td>4.84</td><td>61.82</td><td>8.98</td><td>3.89</td><td>75.00</td><td>7.39</td><td>9.06</td><td>90.57</td><td>16.47</td><td>3.29</td><td>52.24</td><td>6.19</td></tr><tr><td>Qwen3.5-4B</td><td>25.33</td><td>100.00</td><td>40.43</td><td>4.84</td><td>61.82</td><td>8.98</td><td>3.76</td><td>70.00</td><td>7.14</td><td>9.28</td><td>90.57</td><td>16.84</td><td>3.29</td><td>52.24</td><td>6.19</td></tr><tr><td>Qwen3.5-9B</td><td>27.27</td><td>94.74</td><td>42.35</td><td>4.93</td><td>61.82</td><td>9.12</td><td>5.19</td><td>75.00</td><td>9.71</td><td>11.03</td><td>90.57</td><td>19.67</td><td>3.53</td><td>52.24</td><td>6.61</td></tr><tr><td>Qwen3.5-27B</td><td>41.67</td><td>52.63</td><td>46.51</td><td>8.69</td><td>60.61</td><td>15.20</td><td>6.07</td><td>75.00</td><td>11.24</td><td>18.75</td><td>90.57</td><td>31.07</td><td>6.53</td><td>51.79</td><td>11.59</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>28.33</td><td>89.47</td><td>43.04</td><td>11.38</td><td>59.85</td><td>19.12</td><td>3.97</td><td>75.00</td><td>7.54</td><td>18.68</td><td>90.57</td><td>30.97</td><td>10.41</td><td>50.78</td><td>17.28</td></tr><tr><td>Qwen3.5-122B-A10B</td><td>26.56</td><td>89.47</td><td>40.96</td><td>15.38</td><td>59.24</td><td>24.41</td><td>4.23</td><td>75.00</td><td>8.00</td><td>16.44</td><td>90.57</td><td>27.83</td><td>14.16</td><td>50.03</td><td>22.07</td></tr><tr><td>Qwen3.6-27B</td><td>35.56</td><td>84.21</td><td>50.00</td><td>11.07</td><td>60.61</td><td>18.73</td><td>4.41</td><td>75.00</td><td>8.33</td><td>13.30</td><td>90.57</td><td>23.19</td><td>9.92</td><td>51.54</td><td>16.63</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>26.76</td><td>100.00</td><td>42.22</td><td>9.16</td><td>59.09</td><td>15.86</td><td>3.95</td><td>75.00</td><td>7.50</td><td>9.56</td><td>90.57</td><td>17.30</td><td>7.58</td><td>50.88</td><td>13.20</td></tr><tr><td>GPT-5.5</td><td>70.00</td><td>73.68</td><td>71.79</td><td>33.89</td><td>55.30</td><td>42.03</td><td>9.52</td><td>70.00</td><td>16.77</td><td>70.31</td><td>84.91</td><td>76.92</td><td>28.90</td><td>48.01</td><td>36.08</td></tr><tr><td>GPT-5.6-Luna</td><td>100.00</td><td>5.26</td><td>10.00</td><td>35.22</td><td>48.94</td><td>40.96</td><td>10.13</td><td>40.00</td><td>16.16</td><td>52.70</td><td>73.58</td><td>61.42</td><td>27.91</td><td>44.53</td><td>34.32</td></tr><tr><td>GPT-5.6-Terra</td><td>71.43</td><td>26.32</td><td>38.46</td><td>31.41</td><td>53.48</td><td>39.57</td><td>8.48</td><td>70.00</td><td>15.14</td><td>51.11</td><td>86.79</td><td>64.34</td><td>30.06</td><td>45.09</td><td>36.07</td></tr><tr><td>GPT-5.6-Sol</td><td>58.33</td><td>36.84</td><td>45.16</td><td>27.47</td><td>55.61</td><td>36.77</td><td>8.22</td><td>60.00</td><td>14.46</td><td>71.21</td><td>88.68</td><td>78.99</td><td>26.41</td><td>48.31</td><td>34.15</td></tr></table>

## Table 5

Results on the material science and engineering ontologies.
<table><tr><td rowspan="3">LLMs</td><td colspan="8">MatWerk</td><td colspan="6">MDS</td></tr><tr><td colspan="2">Term Typing</td><td colspan="2"></td><td colspan="2">Taxonomy Discovery</td><td colspan="2">Non-Taxonomic RE</td><td colspan="2">Prec</td><td colspan="2">Taxonomy Discovery</td><td colspan="2">Non-Taxonomic RE</td></tr><tr><td>Prec</td><td>Rec</td><td>F1</td><td>Prec</td><td>Rec</td><td>F1</td><td>Prec</td><td>Rec F1</td><td></td><td>Rec</td><td>F1</td><td>Prec</td><td>Rec</td><td>F1</td></tr><tr><td>Qwen3.5-0.8B</td><td>14.29</td><td>100.00</td><td>25.00</td><td>4.25</td><td>60.66</td><td>7.94 7.89</td><td>3.60</td><td>81.82 6.90</td><td>1.36</td><td>18.95</td><td>2.54</td><td>0.62</td><td>27.34</td><td>1.22</td></tr><tr><td>Qwen3.5-2B</td><td>14.29</td><td>100.00</td><td>25.00</td><td>4.22</td><td>60.33</td><td></td><td>3.60</td><td>81.82 6.90</td><td>1.39</td><td>19.28</td><td>2.59</td><td>0.61</td><td>26.56</td><td>1.19</td></tr><tr><td>Qwen3.5-4B</td><td>14.65</td><td>100.00</td><td>25.55</td><td>4.22</td><td>60.33 7.89</td><td></td><td>3.35</td><td>72.73 6.40</td><td>1.39</td><td>19.28</td><td>2.59</td><td>0.62</td><td>26.56</td><td>1.20</td></tr><tr><td>Qwen3.5-9B</td><td>22.22</td><td>96.55</td><td>36.13</td><td>4.52</td><td>60.33 8.40</td><td>5.03</td><td>81.82</td><td>9.47</td><td>1.42</td><td>19.28</td><td>2.64</td><td>0.51</td><td>21.09</td><td>0.99</td></tr><tr><td>Qwen3.5-27B</td><td>43.86</td><td>86.21</td><td>58.14</td><td>6.76</td><td>59.34</td><td>12.13</td><td>6.62</td><td>81.82 12.24</td><td>3.24</td><td>18.63</td><td>5.52</td><td>0.56</td><td>14.06</td><td>1.07</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>20.28</td><td>100.00</td><td>33.72</td><td>11.34</td><td>58.03</td><td>18.97</td><td>3.63</td><td>81.82 6.95</td><td>2.77</td><td>16.99</td><td>4.77</td><td>0.61</td><td>26.56</td><td>1.20</td></tr><tr><td>Qwen3.5-122B-A10B</td><td>24.78</td><td>96.55</td><td>39.44</td><td>14.50</td><td>56.07 23.05</td><td></td><td>3.90 81.82</td><td>7.44</td><td>5.65</td><td>13.73</td><td>8.01</td><td>0.65</td><td>26.56</td><td>1.27</td></tr><tr><td>Qwen3.6-27B</td><td>23.01</td><td>89.66</td><td>36.62</td><td>9.48</td><td>60.33 16.38</td><td>4.15</td><td>81.82</td><td>7.89</td><td>3.60</td><td>16.99</td><td>5.95</td><td>0.61</td><td>26.56</td><td>1.19</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>15.76</td><td>100.00</td><td>27.23</td><td>6.16</td><td>58.03</td><td>11.14</td><td>3.63</td><td>81.82 6.95</td><td>2.79</td><td>15.69</td><td>4.74</td><td>0.63</td><td>26.56</td><td>1.23</td></tr><tr><td>GPT-5.5</td><td>70.59</td><td>82.76</td><td>76.19</td><td>29.70</td><td>48.20 36.75</td><td></td><td>8.25</td><td>72.73 14.81</td><td>16.88</td><td>12.75</td><td>14.53</td><td>2.80</td><td>7.81</td><td>4.12</td></tr><tr><td>GPT-5.6-Luna</td><td>73.68</td><td>48.28</td><td>58.33</td><td>33.88</td><td>47.21 39.45</td><td>13.33</td><td>36.36</td><td>19.51</td><td>20.56</td><td>7.19</td><td>10.65</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>GPT-5.6-Terra</td><td>69.23</td><td>62.07</td><td>65.45</td><td>30.95</td><td>43.93 36.31</td><td></td><td>7.63 81.82</td><td>13.95</td><td>18.64</td><td>7.19</td><td>10.38</td><td>1.08</td><td>7.81</td><td>1.90</td></tr><tr><td>GPT-5.6-Sol</td><td>76.92</td><td>68.97</td><td>72.73</td><td>24.23</td><td>51.80 33.02</td><td>7.92</td><td>72.73</td><td>14.29</td><td>13.17</td><td>12.09</td><td>12.61</td><td>1.08</td><td>7.03</td><td>1.87</td></tr></table>

parameter scale in open-weight models primarily enhances the decision boundary rather than recall; (2) dense models at medium scale (e.g., 27B) often outperform much larger MoE models on the term typing task; and (3) domain complexity creates severe performance bottlenecks, particularly in the non-taxonomic relation extraction task on specialized domains.

Model Scale Dynamics in Open-Weight Families. Evaluating the Qwen parameter progression (0.8B to 122B) demonstrates that scaling model capacity difers across tasks and parameter sizes. More technically:

• Precision Deficits in Small Models (< 9B). Small-scale open-weight models (Qwen3.5-0.8B, 2B, and 4B) exhibit a persistent pattern in performance. On Term Typing across MFOEM, OBI, and MatWerk, these models achieve near-perfect or complete recall (90.57%–100.00%) but with a poor precision (9.06%–25.33%). In taxonomy discovery and non-taxonomic relation extraction, precision collapses further to 1.36%–4.84%. This behavior indicates that the Qwen family of models with fewer than 9B parameters lacks the decision boundary calibration required to reject negative candidate classes provided within the RAG.

<sup>•</sup> The 27B Parameter Turning Point. The transition from 9B to 27B parameters marks the primary performance inflection point within the open-weight model family. On MatWerk Term Typing, Qwen3.5-27B achieves an F1-score of 58.14%, representing a +22.01 percentage point gain over Qwen3.5-9B (36.13%). This improvement is driven by a doubling in precision (from 22.22% to 43.86%) while maintaining high recall (86.21%). A similar precision-driven gain is observed on OBI Term Typing (31.07% vs. 19.67% F1).

• Dense vs. Mixture-of-Experts Trade-ofs. A comparison between dense and sparse MoE architectures reveals clear task-dependent trade-ofs:

– Term Typing: Dense models demonstrate superior active parameter eficiency. The dense Qwen3.5-27B consistently outperforms the larger Qwen3.5-122B-A10B MoE model on term typing across MFOEM (46.51% vs. 40.96% F1) and MatWerk (58.14% vs. 39.44% F1). High active parameter density appears essential for the term typing task.

– Taxonomy Discovery: Conversely, MoE architectures excel at hierarchical relational reasoning. On MFOEM and MatWerk, Qwen3.5-122B-A10B achieves the highest F1-scores among all open-weight models (24.41% and 23.05%, respectively), outperforming its dense 27B counterpart (15.20% and 12.13%). The expanded total capacity of MoE models provides the enhanced capability necessary to evaluate subsumption relationships across paired concepts.

• Version-Specific Alignment Anomaly (Qwen 3.5 vs. 3.6). Comparing release versions reveals an unexpected performance regression in Qwen3.6-27B relative to Qwen3.5-27B on MatWerk Term Typing (36.62% vs. 58.14% F1). This drop suggests that architectural modifications or alignment fine-tuning introduced in version 3.6 altered the model’s sensitivity to specialized engineering terminologies, reinforcing that parameter scale alone does not guarantee monotonic performance improvements across distinct model releases.

Frontier Model Performance and the Alignment Penalty. Proprietary frontier models (GPT-5.5 and GPT-5.6 variants) maintain a substantial performance margin over open-weight baselines, achieving peak F1-scores of 71.79% on MFOEM term typing and 76.19% on MatWerk term typing. However, cross-model comparison within the GPT family highlights significant alignment-induced variations.

<sup>•</sup> GPT-5.5 Superiority. Across nearly all evaluated tasks and domains, GPT-5.5 outperforms its newer GPT-5.6 successors (Luna, Terra, and Sol). It provides a well-calibrated trade-of between precision and recall (e.g., 70.00% precision and 73.68% recall on MFOEM Term Typing), demonstrating robust generalization across domain shifts.

<sup>•</sup> Over-Conservatism in GPT-5.6 Lineage. The GPT-5.6 family demonstrates an over-conservative bias, prioritizing precision over recall. This dynamic is most visible in GPT-5.6-Luna:

– On MFOEM term typing, GPT-5.6-Luna yields 100% precision but collapses to 5.26% recall, resulting in a low F1-score of 10%.

– On MDS Non-Taxonomic Relation Extraction, GPT-5.6-Luna fails entirely (0% Precision, Recall, and F1), declining to output positive relationship assertions.

This behavior suggests that the 5.6 release rendered certain model variants highly prone to false negatives when faced with domain-specific knowledge.

Scale Sensitivity Across Task Complexity and Domains. To study how model size afects performance scaling, we examine how task complexity and domain abstraction influence the benefits of increasing parameter capacity (shown by the scaling curves in Figure 1).

• Task Complexity Changes the Required Scale. The parameter size needed to achieve useful performance increases with task structural complexity (Term Typing ≺ Taxonomy Discovery ≺ Non-Taxonomic RE):

– Low-Floor Task (Term Typing): This task only requires selecting the correct candidate from a top-� list. As a result, scaling provides strong improvements at smaller model sizes (0.8B → 27B achieves a +33.14 F1 gain on MatWerk).

– Moderate-Floor Task (Taxonomy Discovery): This task requires reasoning over relationships between multiple concepts. Small models (< 9B) reach a performance plateau below 9% F1, while noticeable improvements appear only at 27B+ and frontier-scale models (36%–42% F1).

![](images/63e70c4a42d32ddb34e7371e14364bf6f6808676ebf51d0a4fefa13ecdb0709e.jpg)

![](images/94df2c9c0007f8bc5e6ddf46727be5da16bf2b66b69b7a440a692e37f92e9efe.jpg)

![](images/332c5bef1b79cd5d33341b6058c9eba50b028b39758f8c9327b152f9ebe7666f.jpg)

![](images/1f0117a0175f11100e3a6e27d5403b1d64d5fef200369ea8f50f3fb07e7b1e95.jpg)  
Figure 1: F1-score performance across model scale and release lineages for the evaluated biomedical (MFOEM, OBI) and materials science and engineering (MatWerk, MDS) ontologies. Qwen parameter scale is plotted on a logarithmic scale $( 1 0 ^ { 0 }$ to $1 0 ^ { 2 }$ billion parameters), highlighting architectural transitions (dense vs. sparse MoE and Qwen 3.5 vs. 3.6), alongside performance trajectories across the GPT release sequence (GPT-5.5 to GPT-5.6 variants).

– High-Floor Task (Non-Taxonomic RE): Detecting arbitrary semantic relations $( h , r , t )$ is a highly complex task. Increasing parameters for open-weight models provides limited improvements, reaching only 12.24% F1 for Qwen3.5-27B on MatWerk. Even frontier-scale models achieve only modest F1 scores (16.77%–19.51%), showing that increasing model size alone is not enough to solve multi-relational semantic spaces under zero-shot RAG settings.

• Domain Abstraction Creates Scale Resistance in MDS. Comparing diferent domains shows that the level of abstraction afects how much model scaling can improve performance. Scaling consistently improves results in biomedical domains (MFOEM, OBI) and material synthesis (MatWerk). However, the highly abstract Materials Data Science (MDS) domain shows strong scale resistance. For MDS Non-Taxonomic Relation Extraction, increasing Qwen model size from 0.8B to 122B provides almost no improvement (1.22% vs. 1.27% F1), while frontier GPT models achieve only 4.12% F1. Highly abstract and non-standardized conceptual schemas create dificulties that cannot be solved by parameter scaling alone.

## 7. Conclusion

In this work, we conducted a systematic study of how LLM scale influences OL performance across diferent tasks and scientific domains. Using a unified RAG-based evaluation framework, we benchmarked multiple model sizes from the Qwen and GPT families on LLMs4OL paradigm tasks. Our results show that increasing model size does not always lead to proportional improvements. Larger models generally improve precision and decision-making ability, but the benefits strongly depend on task complexity and domain characteristics. Term typing benefits most from scaling, while taxonomy discovery requires larger capacity for relational reasoning. In contrast, non-taxonomic relation extraction remains challenging even for frontier-scale models, particularly in highly abstract domains such as Materials Data Science. We also observe that architectural choices, such as dense versus MoE models, and model alignment strategies can significantly afect performance beyond parameter count alone. These findings highlight that model scale is only one factor determining OL capability. Efective LLM-based OL requires considering task structure, domain abstraction, and model architecture when selecting an appropriate model. Our OntoLearner infrastructure provides empirical guidance for choosing LLMs for ontology engineering and establishes a foundation for future research on scalable and reproducible OL systems.

## Acknowledgments

This work is supported by the NFDI4DataScience initiative (DFG, German Research Foundation, Grant ID: 460234259).

## Declaration on Generative AI

In preparing this manuscript, generative AI tools, specifically ChatGPT, were used solely for grammar checking, spelling checks, and readability of some sentences. All suggested changes were carefully reviewed and adapted by the authors to ensure accuracy and appropriateness. The scientific content, research design, analysis, and conclusions were developed and verified exclusively by the authors without AI involvement. The use of ChatGPT was limited to enhancing the presentation of the work.

## References

[1] C. Bock, P. Datlinger, F. Chardon, M. A. Coelho, M. B. Dong, K. A. Lawson, T. Lu, L. Maroc, T. M. Norman, B. Song, et al., High-content crispr screening, Nature Reviews Methods Primers 2 (2022) 8.

[2] G. Tom, S. P. Schmid, S. G. Baird, Y. Cao, K. Darvish, H. Hao, S. Lo, S. Pablo-García, E. M. Rajaonson, M. Skreta, et al., Self-driving laboratories for chemistry and materials science, Chemical Reviews 124 (2024) 9633–9732.

[3] M. D. Wilkinson, M. Dumontier, I. J. Aalbersberg, G. Appleton, M. Axton, A. Baak, N. Blomberg, J.-W. Boiten, L. B. da Silva Santos, P. E. Bourne, et al., The fair guiding principles for scientific data management and stewardship, Scientific data 3 (2016) 1–9.

[4] S. Auer, J. D’Souza, K. E. Farfar, M. Y. Jaradeh, A. Jiomekong, O. Karras, A. Oelen, L. Snyder, M. Stocker, L. Vogt, Open research knowledge graph: A large-scale neuro-symbolic knowledge organization system, in: Handbook on Neurosymbolic AI and Knowledge Graphs, IOS Press, 2025, pp. 385–420.

[5] H. Wang, T. Fu, Y. Du, W. Gao, K. Huang, Z. Liu, P. Chandak, S. Liu, P. Van Katwyk, A. Deac, et al., Scientific discovery in the age of artificial intelligence, Nature 620 (2023) 47–60.

[6] S. Eger, Y. Cao, J. D’Souza, A. Geiger, C. Greisinger, S. Gross, Y. Hou, B. Krenn, A. Lauscher, Y. Li, et al., Transforming science with large language models: A survey on ai-assisted scientific discovery, experimentation, content generation, and evaluation, arXiv preprint arXiv:2502.05151 (2025).

[7] E. F. Kendall, D. L. McGuinness, Ontology engineering, Morgan & Claypool Publishers, 2019.

[8] H. B. Giglou, J. D’Souza, S. Auer, Llms4ol: Large language models for ontology learning, Lecture Notes in Computer Science (2023) 408–427.

[9] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, Bert: Pre-training of deep bidirectional transformers for language understanding, 2019. arXiv:1810.04805.

[10] M. Lewis, Y. Liu, N. Goyal, M. Ghazvininejad, A. Mohamed, O. Levy, V. Stoyanov, L. Zettlemoyer, Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension, 2019. arXiv:1910.13461.

[11] S. Longpre, L. Hou, T. Vu, A. Webson, H. W. Chung, Y. Tay, D. Zhou, Q. V. Le, B. Zoph, J. Wei, et al., The flan collection: Designing data and methods for efective instruction tuning, arXiv preprint arXiv:2301.13688 (2023).

[12] B. Workshop, :, T. L. Scao, A. Fan, et al., Bloom: A 176b-parameter open-access multilingual language model, 2023. URL: https://arxiv.org/abs/2211.05100. arXiv:2211.05100.

[13] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al., Llama: Open and eficient foundation language models, arXiv preprint arXiv:2302.13971 (2023).

[14] R. Dale, Gpt-3: What’s it good for?, Natural Language Engineering 27 (2021) 113–118.

[15] Y. Gu, R. Tinn, H. Cheng, M. Lucas, N. Usuyama, X. Liu, T. Naumann, J. Gao, H. Poon, Domainspecific language model pretraining for biomedical natural language processing, ACM Transactions on Computing for Healthcare (HEALTH) 3 (2021) 1–23.

[16] H. Babaei Giglou, J. D’Souza, S. Auer, Llms4ol 2024 overview: The 1st large language models for ontology learning challenge, Open Conference Proceedings 4 (2024) 3–16. doi:10.52825/ocp. v4i.2473.

[17] H. Babaei Giglou, J. D’Souza, N. Mihindukulasooriya, S. Auer, Llms4ol 2025 overview: The 2nd large language models for ontology learning challenge, Open Conference Proceedings 6 (2025). doi:10.52825/ocp.v6i.2913.

[18] Y. Peng, Y. Mou, B. Zhu, S. Sowe, S. Decker, Rwth-dbis at llms4ol 2024 tasks a and b, Open Conference Proceedings 4 (2024).

[19] S. Hashemi, M. Karimi Manesh, M. Shamsfard, Skh-nlp at llms4ol 2024 task b: Taxonomy discovery in ontologies using bert and llama 3, Open Conference Proceedings 4 (2024).

[20] T. Phuttaamart, N. Kertkeidkachorn, A. Trongratsameethong, The ghost at llms4ol 2024 task a: Prompt-tuning-based large language models for term typing, Open Conference Proceedings 4 (2024).

[21] M. Sanaei, F. Azizi, H. Babaei Giglou, Phoenixes at llms4ol 2024 tasks a, b, and c: Retrieval augmented generation for ontology learning, Open Conference Proceedings 4 (2024).

[22] C. Ymele, A. Jiomekong, Combining rules to large language model for ontology learning, Open Conference Proceedings 4 (2024).

[23] P. Kumar Goyal, S. Singh, U. Shanker Tiwari, silp\_nlp at llms4ol 2024 tasks a, b, and c: Ontology learning through prompts with llms, Open Conference Proceedings 4 (2024).

[24] R. Rahnamoun, M. Shamsfard, Sbu-nlp at llms4ol 2025 tasks a, b, and c: Stage-wise ontology construction through llms without any training procedure, Open Conference Proceedings (2025).

[25] A. Beliaeva, T. Rahmatullaev, Alexbek at llms4ol 2025 tasks a, b, and c: Heterogeneous llm methods for ontology learning (few-shot prompting, ensemble typing, and attention-based taxonomies), Open Conference Proceedings (2025).

[26] P. Goyal, S. Singh, U. S. Tiwary, silp\_nlp at llms4ol 2025 tasks a, b, c, and d: Clustering-based ontology learning using llms, Open Conference Proceedings (2025).

[27] I.-A. Latipov, M. Holenderski, N. Meratnia, Iris at llms4ol 2025 tasks b, c, and d: Enhancing ontology learning through data enrichment and type filtering, Open Conference Proceedings (2025).

[28] P. Wiangnak, T. Prabhong, T. Phuttaamart, N. Kertkeidkachorn, K. Shirai, The dream-llms at llms4ol 2025 task b: A deliberation-based reasoning ensemble approach with multiple large language models for term typing in low-resource domains, Open Conference Proceedings (2025).

[29] A. E. Fridouni, M. Sanaei, Phoenixes at llms4ol 2025 task a: Ontology learning with large language

models reasoning, Open Conference Proceedings (2025).

[30] R. Roche, K. Gray, J. Murdock, D. C. Crowder, Ellmo at llms4ol 2025 tasks a and d: Llm-based term, type, and relationship extraction, Open Conference Proceedings (2025).

[31] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, D. Amodei, Scaling laws for neural language models, arXiv preprint arXiv:2001.08361 (2020).

[32] J. Hofmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. d. L. Casas, L. A. Hendricks, J. Welbl, A. Clark, et al., Training compute-optimal large language models, arXiv preprint arXiv:2203.15556 (2022).

[33] J. Wei, Y. Tay, R. Bommasani, C. Rafel, B. Zoph, S. Borgeaud, D. Yogatama, M. Bosma, D. Zhou, D. Metzler, et al., Emergent abilities of large language models, arXiv preprint arXiv:2206.07682 (2022).

[34] R. Schaefer, B. Miranda, S. Koyejo, Are emergent abilities of llms a mirage?, Proc. NeurIPS 23 (2023).

[35] I. R. McKenzie, A. Lyzhov, M. M. Pieler, A. Parrish, A. Mueller, A. Prabhu, E. McLean, X. Shen, J. Cavanagh, A. G. Gritsevskiy, D. Kaufman, A. T. Kirtland, Z. Zhou, Y. Zhang, S. Huang, D. Wurgaft, M. Weiss, A. Ross, G. Recchia, A. Liu, J. Liu, T. Tseng, T. Korbak, N. Kim, S. R. Bowman, E. Perez, Inverse scaling: When bigger isn’t better, Transactions on Machine Learning Research (2023). URL: https://openreview.net/forum?id=DwgRm72GQF, featured Certification.

[36] J. Wei, N. Kim, Y. Tay, Q. Le, Inverse scaling can become u-shaped, in: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 15580–15591.

[37] N. Lourie, M. Y. Hu, K. Cho, Scaling laws are unreliable for downstream tasks: A reality check, in: C. Christodoulopoulos, T. Chakraborty, C. Rose, V. Peng (Eds.), Findings of the Association for Computational Linguistics: EMNLP 2025, Association for Computational Linguistics, Suzhou, China, 2025, pp. 16167–16180. URL: https://aclanthology.org/2025.findings-emnlp.877/. doi:10. 18653/v1/2025.findings-emnlp.877.

[38] W. Fedus, B. Zoph, N. Shazeer, Switch transformers: Scaling to trillion parameter models with simple and eficient sparsity, Journal of Machine Learning Research 23 (2022) 1–39.

[39] H. B. Giglou, J. D’Souza, A. Aioanei, N. Mihindukulasooriya, S. Auer, Ontolearner: A modular python library for ontology learning with large language models, 2026. URL: https://arxiv.org/abs/ 2607.01977. arXiv:2607.01977.

[40] Qwen Team, Qwen3.5: Towards native multimodal agents, 2026. URL: https://qwen.ai/blog?id= qwen3.5.

[41] Qwen Team, Qwen3.6-27B: Flagship-level coding in a 27B dense model, 2026. URL: https://qwen. ai/blog?id=qwen3.6-27b.

[42] Qwen Team, Qwen3.6-35B-A3B: Agentic coding power, now open to all, 2026. URL: https://qwen. ai/blog?id=qwen3.6-35b-a3b.

[43] OpenAI, Gpt-5.5 model, 2026. URL: https://developers.openai.com/api/docs/models/gpt-5.5.

[44] OpenAI, Gpt-5.6 luna model, 2026. URL: https://developers.openai.com/api/docs/models/gpt-5. 6-luna.

[45] OpenAI, Gpt-5.6 terra model, 2026. URL: https://developers.openai.com/api/docs/models/gpt-5. 6-terra.

[46] OpenAI, Gpt-5.6 sol model, 2026. URL: https://developers.openai.com/api/docs/models/gpt-5.6-sol.

[47] T. Ivanova, V. Terzieva, Ontology learning in educational systems, Information 17 (2026). URL: https://www.mdpi.com/2078-2489/17/2/147. doi:10.3390/info17020147.

[48] A. S. Lippolis, M. J. Saeedizade, R. Keskisärkkä, S. Zuppiroli, M. Ceriani, A. Gangemi, E. Blomqvist, A. G. Nuzzolese, Ontology generation using large language models, in: European Semantic Web Conference, Springer, 2025, pp. 321–341.

[49] A. S. Lippolis, M. J. Saeedizade, R. Keskisarkka, A. Gangemi, E. Blomqvist, A. G. Nuzzolese, Assessing the capability of large language models for domain-specific ontology generation, arXiv preprint arXiv:2504.17402 (2025).

[50] Y. Teplyi, D. Dosyn, Prompt-guided llm agent for end-to-end ontology learning, Computer Engineering 22 (2025) 35–47.

[51] N. Fathallah, A. Das, S. De Giorgis, A. Poltronieri, P. Haase, L. Kovriguina, A. Meroño-Peñuela, E. Simperl, S. Staab, A. Algergawy, Extended neon-gpt: Advancing llm-powered ontology learning through ontology reuse and automated verification, Semantic Web 17 (2026) 22104968261453138.

[52] Y. Zhang, A. S. Dalal, C. Martin, S. R. Gadusu, H. K. McGinty, Olive: Ontology learning with integrated vector embeddings, Applied Ontology 20 (2025) 36–53.

[53] D. Doumanas, A. Soularidis, D. Spiliotopoulos, C. Vassilakis, K. Kotis, Fine-tuning large language models for ontology engineering: A comparative analysis of gpt-4 and mistral, Applied Sciences 15 (2025). URL: https://www.mdpi.com/2076-3417/15/4/2146. doi:10.3390/app15042146.

[54] X. Liu, Z. Li, M. He, Z. Ma, X. Wu, G. Yilmaz, Y. Xia, B. Li, H. Tan, J. Y. H. Fuh, et al., From prompt to graph: Comparing llm-based information extraction strategies in domain-specific ontology development, arXiv preprint arXiv:2602.00699 (2026).

[55] M. A. Cappelli, G. Di Marzo Serugendo, Methodological exploration of ontology generation with a dedicated large language model, Electronics 14 (2025) 2863.

[56] T. da Cruz, B. Tavares, F. Belo, Ontology learning and knowledge graph construction: A comparison of approaches and their impact on rag performance, arXiv preprint arXiv:2511.05991 (2025).

[57] A. S. Lippolis, M. J. Saeedizade, S. Schmid, S. Blattner, R. Keskisärkkä, A. Gangemi, E. Blomqvist, A. G. Nuzzolese, Ontoextend: A framework for requirement-driven and scalable ontology extension with llms, arXiv preprint arXiv:2607.17963 (2026).

[58] T. Cherian, W. Y. C. Wang, A. Zahid, Custom fhir extensions with ontology learning models for improving data integration and interoperability in healthcare management systems, in: Proceedings of the 2025 9th International Conference on Medical and Health Informatics, 2025, pp. 132–136.

[59] Y. Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang, P. Xie, A. Yang, D. Liu, J. Lin, F. Huang, J. Zhou, Qwen3 embedding: Advancing text embedding and reranking through foundation models, arXiv preprint arXiv:2506.05176 (2025).

[60] H. Beygi Nasrabadi, E. Norouzi, K. Hubaiev, J. Waitelonis, H. Sack, Nfdi matwerk ontology (mwo): A bfo-compliant ontology for research data management in materials science and engineering, Advanced Engineering Materials (2025) e202502331.

[61] B. P. Rajamohan, A. C. H. Bradley, V. D. Tran, J. E. Gordon, H. W. Caldwell, R. Mehdi, G. Ponon, Q. D. Tran, O. Dernek, J. Kaltenbaugh, et al., Materials data science ontology (mds-onto): Unifying domain knowledge in materials and applied data science, Scientific Data 12 (2025) 628.

[62] A. Bandrowski, R. Brinkman, M. Brochhausen, M. H. Brush, B. Bug, M. C. Chibucos, K. Clancy, M. Courtot, D. Derom, M. Dumontier, et al., The ontology for biomedical investigations, PloS one 11 (2016) e0154556.

[63] J. Hastings, W. Ceusters, B. Smith, K. Mulligan, Dispositions and processes in the emotion ontology (2011).

[64] J. Hastings, W. Ceusters, K. Mulligan, B. Smith, Annotating afective neuroscience data with the emotion ontology (2012).