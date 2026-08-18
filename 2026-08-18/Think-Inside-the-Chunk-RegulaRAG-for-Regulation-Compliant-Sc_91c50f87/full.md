# Think Inside the Chunk: RegulaRAG for Regulation-Compliant Scenario Generation using LLMs: A Case Study of UN Regulation No. 152

Vahid Zolfaghari, IEEE, Nenad Petrovic, Andre Schamschurko, and Alois Knoll, ´ Fellow, IEEE

Abstract—Generating regulation-compliant test scenarios is essential for validating safety-critical automotive systems, yet Large Language Models (LLMs) struggle to ground outputs in long, hierarchical standards. We present RegulaRAG, a Retrieval-Augmented Generation (RAG) pipeline that couples SmartChunking, reference-aware enrichment of paragraphs and tables via graph traversal, with Smart Retrieve & Rerank over these enriched units. To test our system, we evaluate on a manually curated dataset covering all scenarios in UN Regulation No. 152 (AEBS). Our study comprises: (i) a three-step progressive search that identifies near-optimal retrieval parameters without exhaustive grid search; (ii) head-to-head comparisons against five baseline RAG systems; and (iii) a robustness stress test that scales the source corpus with distractor content. Outputs are evaluated using a customized penalized scoring metric. Across all experiments, RegulaRAG achieves the highest average Meta-Score (82.99), outperforming the next-best system by 43% (NoRAG: 57.94), while operating at 14k–25k tokens per query versus up to 500k for graph-centric baselines. It maintains strong performance, remaining stable even as the number of regulatory sources grows, whereas competing RAG systems degrade sharply in both quality and robustness.

Index Terms—Automotive software, Large Language Models, Retrieval Augmented Generation, Standard compliance

## I. INTRODUCTION

L <sup>ARGE</sup> <sup>Language</sup> <sup>Models</sup> <sup>(LLMs)</sup> <sup>such</sup> <sup>as</sup> <sup>GPT-4o</sup> demonstrate impressive capabilities in generating humanlike responses across various domains. However, LLMs face challenges such as hallucinations, outdated knowledge, and difficulty in recalling rare or highly specific information [1], [2]. To overcome these limitations, Retrieval-Augmented Generation (RAG) has emerged as a promising approach by integrating external documents during inference to enhance accuracy, traceability, and grounding [3].

In the context of automotive software development, particularly for Advanced Driver Assistance Systems (ADAS) and Autonomous Driving Systems (ADS), generating precise, regulation-compliant test scenarios is critical. Traditional manual processes for interpreting and validating against regulatory standards like UN Regulation No. 152 are time-consuming, error-prone, and costly. LLMs have potential to automate this effort, but without RAG, they often struggle to identify the correct numerical parameters, differentiate test conditions (e.g., laden vs. unladen), and manage long documents exceeding context limits. Furthermore, directly inputting the full regulation to an LLM results in high token usage and associated costs. Although recent large language models (LLMs) such as Claude provide extended context windows of up to 100k tokens [4], simply feeding full-length automotive documents (e.g., requirements or specifications) into the prompt is rarely feasible due to confidentiality and intellectual property concerns. Moreover, long-context models often struggle with effective utilization of all input tokens, suffering from issues such as context dilution or the so-called ”lost in the middle” effect [5].

To address these challenges, we introduce RegulaRAG, a smart two-stage Retrieval-Augmented Generation pipeline tailored for regulation-compliant scenario generation. RegulaRAG employs a novel SmartChunking strategy to preprocess PDF-based standards, identify hierarchical paragraph structures and nested references, and enrich document chunks via a graph-based traversal. This ensures that retrieved content includes all semantically linked paragraphs and tables, while maintaining a compact token footprint. Our Smart Retrieve and Rerank module performs query-aware retrieval over these reference-enriched chunks, allowing the LLM to generate accurate, traceable, and standards-compliant scenarios. This method outperforms traditional retrieval pipelines that operate on uninformed or semantically shallow chunks. An example of such traditional methods is provided in [6], where a simple Retrieve-and-Rerank strategy was applied for question answering over automotive documents to support hardware and software design workflows. The core methodological novelty of RegulaRAG lies in two algorithmic contributions: (i) reference-aware chunk enrichment via BFS graph traversal over the regulation’s explicit cross-reference structure, and (ii) retrieval and reranking against these enriched representations to surface dispersed regulatory evidence in a single step. Bounded BFS depth, chunk deduplication, and metadata compression are engineering optimizations that improve practical scalability on long regulations without altering the underlying algorithm; they are reported as such in Section III.

The remainder of this paper is structured as follows: Section II reviews related work on LLMs for compliance and scenario generation. Section III presents the RegulaRAG system and the dataset. Section IV describes the experimental setup. Section V presents the experiments and results. Section VI concludes the

paper.

## II. LITERATURE REVIEW

The challenge of processing complex PDF documents, a common format for technical specifications in the automotive industry, has been addressed by several researchers. For example, [7] propose a method that addresses key challenges in automotive document processing, including multi-column layouts and technical specifications. Following is the main challenges in processing these documents:

1) Tables: Automotive standards often contain critical tables with specifications, test results, or compliance data. These may span multiple pages in the PDF and get split into separate tables after conversion, while in fact they must be interpreted as one [7]. An example of these split tables can be seen in Fig. 5.

2) Domain-specific terminology: Regulations use highly technical and legal jargon, abbreviations, and regionspecific terms. Since RAG relies on similarity between queries and chunks, such variation often causes it to miss relevant content.

3) Lexical and syntactic variation: The same requirement can appear in different formulations, typically as long, complex sentences with passive voice and nested clauses. This makes retrieval and interpretation challenging [8] .

4) Nested and interlinked references: Requirements are frequently split across multiple clauses and annexes that point to one another. An example of these interlinked references is illustrated in Figure 1. Standard RAG systems often retrieve isolated chunks, missing the linked context.

Recent research has increasingly focused on leveraging Large Language Models (LLMs) for compliance verification across regulated domains. Sun et al. [9] propose a compliance checking framework that integrates Retrieval-Augmented Generation (RAG) with an “eventic graph” to align business process descriptions with regulatory rules. Their method demonstrates the benefits of combining structured knowledge with retrieval-based augmentation, but it primarily produces compliance verdicts rather than executable artifacts. Bolton et al. [10] introduce DRAFT, a document-retrieval-augmented fine-tuning approach for safety-critical software assessments. By fine-tuning LLMs with both standards and distractors, their method improves factual correctness and evidence handling. However, the output remains focused on assessment text and explanations rather than concrete test cases. LLMs and RAG are also used for detecting inconsistencies, contradictions, and conflicts in regulatory documents. Kumar and Roussinov [11] explore the use of GPT-4 for these purposes. While effective for semantic analysis at the document level, the method does not extend toward structured instantiation of compliance requirements. A complementary line of work seeks to translate regulatory clauses into executable rules. Li et al. [12] present Compliance-to-Code, which maps financial regulations into programmatic logic to automate compliance checks. Although this bridges textual requirements and computational enforcement, it remains abstract and disconnected from domainspecific simulation semantics.

Agarwal et al. [13] propose a multi-agent knowledge graph framework for regulatory Quesion and Answer, where extracted triplets are combined with RAG-based retrieval to improve factual correctness and traceability. This enhances transparency in compliance reasoning but is not tailored to generating domain-grounded artifacts such as test scenarios.

Compared to these approaches, our work targets a different level of compliance verification. Rather than producing static classifications, narrative explanations, or direct code translations, RegulaRAG operationalizes regulatory text into simulation-ready test scenarios. By combining SmartChunking, reference-aware enrichment, and selective retrieval, our method dynamically reconstructs scenario definitions from regulatory clauses, tables, and cross-references in a traceable manner.

A more recent line of work targets structure-aware retrieval over complex documents. Xu et al. [14] propose RDR2, a Retrieve–Document Route–Read pipeline in which an LLMbased router navigates a document structure tree at inference time to select or expand the most relevant heading nodes before generation. RDR2 shows strong gains on general QA benchmarks (TriviaQA, HotpotQA, ASQA), but its structure prior is a heading hierarchy and does not address table reconstruction across page boundaries, typed legal crossreferences, or normative dependencies between provisions. Yu et al. [15] propose TableRAG, which addresses the loss of table integrity in standard chunking by storing tables in a relational database and enabling SQL-driven retrieval alongside text-based search. TableRAG is the closest prior work on the table-preservation problem, yet it targets document QA over Wikipedia-style sources and does not handle regulatory cross-references or downstream scenario synthesis. Closer to the normative domain, Oliveira et al. [16] build a knowledge graph over Portuguese legal resolutions with typed nodes (Document, Article, Paragraph) and edges that explicitly model amendment and revocation relations (Contain, Modify, Revoke). Their system expands retrieved seed nodes to graph neighbours but discards table and image content and does not produce structured scenario artefacts. RegulaRAG differs from all three in targeting regulation-specific scenario generation: it reconstructs fragmented PDF tables via a rule-based heuristic, performs BFS reference-closure enrichment over the regulation’s explicit cross-reference graph, and produces compliance-grounded scenario text rather than short factual answers.

Unlike knowledge-graph-based approaches, RegulaRAG does not perform general entity recognition or relation extraction. This is a deliberate design choice grounded in the properties of regulatory text: legal language in standards such as UN Regulation No. 152 relies on heavily nested clause structures, conditional syntax, and domain-specific crossreferences that make reliable open-domain entity detection and edge extraction challenging. Rather than building a generalpurpose KG, RegulaRAG exploits the regulation’s explicit cross-reference structure (numbered paragraph headings and direct table pointers already embedded in the document) as a lightweight, structure-preserving proxy for knowledge linkage. This avoids the LLM-heavy NER, triple extraction, and graphexpansion passes that drive the high token costs observed in KG-based pipelines (see Section V-B for a direct comparison).

![](images/3f93eff72ad46353749dce9d125fe33cc8a20115258b42b84800269c01db306a.jpg)  
Fig. 1. Example of a reference and table chain in UN Regulation No. 152.

While we evaluate RegulaRAG on UN Regulation No. 152 [17]—which specifies the performance, testing procedures, and safety requirements for Automated Emergency Braking Systems (AEBS) in passenger vehicles—the pipeline is not specific to AEBS or to Regulation 152. In principle, the same workflow can be applied to other UN regulations that share similar structural properties (hierarchical paragraphs, annexes, and interlinked references) and can potentially be adapted to other languages with very little modification. Applying RegulaRAG to a different UN regulation primarily requires updating the prompts, as the core algorithms for chunk enrichment, de-duplication, and retrieval remain universal. This shifts compliance validation from purely textual analysis to the generation of executable test scenarios, bridging the gap between regulatory interpretation and practical system validation in safety-critical testing workflows.

In the technical standards domain, CompliAT [18] presents a structured approach for compliance checking of Assistive Technology (AT) products against relevant standards. It focuses on verifying terminological consistency with standarddefined terms, correctly classifying products, and validating product specifications against applicable requirements. However, unlike these works, which primarily aim at classification or static compliance checks, our work dynamically generates test scenarios based on regulatory documents to validate system behavior through simulation-based execution.

A parallel line of research applies LLMs to the automated generation of driving scenarios for ADS testing. Chat2Scenario [19] extracts concrete simulation scenarios from naturalistic datasets using GPT-4, while TARGET [20] translates textual traffic rules into a formal DSL for test generation. LeGEND [21] derives scenarios from accident reports to enhance diversity and realism, and LEADE [22] leverages multimodal prompting on traffic videos to reconstruct functional scenarios where ADS may fail. OmniTester [23] combines multimodal LLMs with user prompts to produce diverse and critical scenarios. While these works focus on creating challenging or diverse test cases to stress ADS in simulation, our objective differs: rather than inventing novel or corner-case scenarios, RegulaRAG systematically extracts regulation-compliant scenarios directly from formal standards such as UN Regulation No. 152, thereby prioritizing completeness, traceability, and compliance over diversity alone. Because our method generates regulation-faithful scenarios in natural language, they can serve as inputs to downstream code-generation systems that set up simulations; for example, Lebioda et al. [24] show that LLMs can translate abstract requirements from automotive regulations into executable CARLA configuration code.

In summary, while all these works demonstrate the power of LLMs in test scenario generation, our approach is unique in targeting the structured extraction of test scenarios from regulatory standards for compliance verification—bridging the gap between scenario generation and formal safety requirements in a traceable and automated manner.

## III. SYSTEM DESCRIPTION

The main goal of the RegulaRAG pipeline is to extract and structure the most relevant parts of lengthy regulation documents so that Large Language Models (LLMs) can generate regulation-compliant test scenarios. An overview of the full RegulaRAG pipeline can be seen in Figure 2. The pipeline operates in three phases:

a) Phase 1: Extraction.: We use the Mineru tool [25] to convert the regulation document from PDF into Markdown format and then normalize the extracted text to make it machineusable. Concretely, we convert legal paragraph numbers (e.g., 5.2.1.4) into canonical Markdown headers, repair extraction artifacts (line breaks, hyphenation), and standardize tables while preserving the original numbering as stable IDs. The resulting structure keeps chunk boundaries aligned with the regulation’s paragraph hierarchy, makes cross-references detectable for reference-aware enrichment, and provides cleaner inputs for retrieval and re-ranking.

b) Phase 2: Chunking (SmartChunking).: Given the structured Markdown produced in Phase 1, SmartChunking first segments the text into semantically coherent base chunks using LangChain’s RecursiveCharacterTextSplitter<sup>1</sup>, then builds a reference-closure for each chunk by resolving crossreferenced headings and tables via BFS, so every chunk is selfcontained with respect to the regulation’s internal dependencies. We separate the core algorithmic steps from engineering optimizations below.

Following is the core algorithm for SmartChunking:

1) Detect references (heading and table references). For each chunk, we record three types of metadata using rule-based patterns over the normalized text:

a) Defined Headings: Headings explicitly defined within the chunk using Markdown syntax (lines starting with #).

b) Referenced Headings: Identifiers of headings (e.g., “2.3”) mentioned in the chunk but defined elsewhere. These are resolved by locating the defining chunk and linking it.

c) Referenced Tables: References to HTMLformatted tables identified using heuristics (e.g., “the following table”). Chunks mentioning such phrases are linked to the chunk containing the actual table.

Heading references are resolved by locating the chunk that defines the target heading ID. For table references, we apply a rule-based heuristic: any paragraph containing a trigger phrase such as “the following table” is treated as an explicit pointer to subsequent table content. We then collect the consecutive <table> HTML blocks that immediately follow the referencing paragraph in Mineru’s Markdown/HTML output and group them as a single logical table entry. This grouping reconstructs PDF tables that were split across pages and converted into multiple adjacent HTML fragments. Each resolved reference is mapped to its target stable regulation ID.

2) Build a document reference graph. We construct a directed graph where nodes represent heading/table IDs (and their defining chunks), and edges represent explicit references from one node to another. This graph encodes the regulation’s cross-reference structure.

3) Compute transitive reference closure via BFS. After metadata extraction, we expand each chunk’s context by traversing the reference graph via Breadth-First Search (BFS), recursively collecting all referenced headings/tables including nested chains (e.g., a chunk referencing 6.6, which in turn references 6.3.2, as depicted in Figure 1). This produces a reference-closure set of evidence associated with the base chunk. During this stage, we apply inter-chunk deduplication to eliminate redundant content. For instance, if a chunk references two subparagraphs defined within the same chunk, we ensure that the defining chunk is not added twice. This deduplication ensures efficient context expansion and prevents excessive token usage.

4) Create enriched chunk representations (base + closure). We form an enriched representation for each chunk by concatenating the base chunk with the text of its reference-closure items (headings/tables). These enriched representations are used as retrieval units so that evidence dispersed across paragraphs and annexes can be retrieved together.

To keep SmartChunking practical on long regulations, we apply several Engineering optimizations that do not change the underlying algorithm: (i) bounded BFS (we cap BFS expansion to a maximum of 50 nodes per chunk, i.e., max\_expand= 50, to prevent pathological expansion in densely cross-referenced regulations); (ii) de-duplication (we remove repeated referenced items both within a chunk’s closure and across overlapping closures to avoid redundant context); and (iii) metadata compression (we store references as integer IDs/pointers instead of copying full referenced text during preprocessing, materializing text only when assembling retrieval contexts). We report these optimizations explicitly as implementation choices aimed at improving runtime and token efficiency.

c) Phase 3: Retrieval and Generation.: Smart Retrieve and Rerank differs from standard top-k retrieval in that similarity is computed between the query and each chunk’s enriched representation (base text concatenated with its referenceclosure), so that a query for a test condition can match a chunk through its referenced table or sub-paragraph even when the base text alone would score poorly; the retrieved base chunks and their closure items are then assembled into the final LLM context. In the third phase, we embed all enriched chunk representations with sentence-transformers/all-mpnet-base-v2<sup>2</sup> from the sentence-transformers library (v3.4.1), which maps each sentence or paragraph to a 768-dimensional dense vector. This model is based on MPNet and is widely used for semantic search and dense retrieval due to its strong performance on sentence-level similarity benchmarks. We selected this model for retrieval because it provides robust semantic matching performance while maintaining moderate computational cost, which is important for large regulatory corpora.

To compute similarity between the query and each enriched chunk, we use the cosine similarity function from the scikitlearn library.<sup>3</sup> Based on these scores, we retrieve the top-k most relevant chunks.

After ranking enriched chunks and selecting the top-k candidates, we construct the final context provided to the LLM through a structured assembly procedure (see Fig. 2). This step consists of three stages:

1) Reference Expansion. For each selected base chunk, we retrieve its reference-closure set (headings and tables obtained via BFS in Phase 2). This ensures that all semantically linked regulatory elements required to instantiate a scenario are included.

2) Intra- and Inter-Chunk De-duplication. Because multiple selected chunks may reference the same paragraph or table, we remove duplicate entries using their stable regulation IDs. This guarantees that each referenced element appears exactly once in the final context and avoids unnecessary token overhead.

3) Canonical Ordering. The resulting set of base chunks and referenced items is sorted according to the regulation’s original hierarchical numbering (e.g., 5.2.1 ≺ 5.2.1.1 ≺ 5.2.2). Tables inherit the position of their defining paragraph ID. This ordering reconstructs the logical flow of the regulation rather than the arbitrary similarity-based retrieval order. The sorted chunks are then concatenated in this canonical order to form the final context window returned to the LLM.

This ordering step is crucial because similarity-based retrieval alone may return relevant fragments in a non-coherent order. By restoring canonical regulation order and merging referenced tables with their defining clauses, RegulaRAG provides the LLM with a logically structured evidence block, reducing hallucinations and improving numerical consistency in generated scenarios.

Algorithm 1 formalises the complete context assembly procedure.

Algorithm 1: Reference-Aware Context Assembly   
(Smart Retrieve & Rerank)   
Input: Query q; enriched chunk set $\overline { { \mathcal { C } _ { E } } }$ where each chunk c   
carries base text and reference-closure R(c); retrieval   
encoder enc; retrieval breadth k.   
Output: Ordered, de-duplicated context window W.   
E<sub>q</sub> ← enc(q) ; // embed query   
foreach enriched chunk $c \in { \mathcal { C } } _ { E }$ do   
E<sub>c</sub> ← enc(c.enriched text);   
s<sub>c</sub> ← cosine similarity(E<sub>q</sub>, E<sub>c</sub>);   
T ← argtop { s<sub>c</sub> : c ∈ C<sub>E</sub>} ; // select top-k   
chunks   
// Step 1 | Reference expansion   
T<sub>exp</sub> ← T ;   
foreach base chunk $c \in \tau$ do   
T<sub>exp</sub> ← T<sub>exp</sub> ∪ R(c)   
// Step 2 | De-duplication   
T<sub>uniq</sub> ← deduplicate(T<sub>exp</sub>);   
// Step 3 | Canonical ordering ;   
W ← sort(T<sub>uniq</sub>, key = regulation id);   
return W;

## A. Dataset description

To evaluate the capabilities of Large Language Models (LLMs) in generating test scenarios from regulatory standards, we constructed a dataset derived from UN Regulation No. 152. This dataset consists of manually extracted and categorized test scenarios relevant to Advanced Emergency Braking Systems (AEBS). The dataset is summarized in table I. We release the dataset publicly on Hugging Face at vahidzol $\pm / \updownarrow \mathrm { n n \_ } 1 5 2$

TABLE I  
SUMMARY OF TEST SCENARIOS BY CATEGORY AND CONDITION
<table><tr><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Condition</td><td rowspan=1 colspan=1>Num Scenarios</td></tr><tr><td rowspan=1 colspan=1>CtoStC</td><td rowspan=1 colspan=1>unladenladen</td><td rowspan=1 colspan=1>1212</td></tr><tr><td rowspan=1 colspan=1>CtoMoC</td><td rowspan=1 colspan=1>unladenladen</td><td rowspan=1 colspan=1>87</td></tr><tr><td rowspan=1 colspan=1>CtoP</td><td rowspan=1 colspan=1>MassRunMaxMass</td><td rowspan=1 colspan=1>1010</td></tr></table>

The scenarios are divided into three primary categories, each representing a distinct type of test situation:

• Car-to-Stationary-Car (CtoStC): These scenarios assess the AEBS functionality when a vehicle approaches a stationary car. The tests include variations in vehicle load (laden/unladen) and different approach speeds (e.g., 20 km/h, 42 km/h, 60 km/h).

• Car-to-Moving-Car (CtoMoC): This category focuses on test cases where a vehicle interacts with another moving vehicle, considering relative speeds and conditions specified in the standard.

• Car-to-Pedestrian (CtoP): These scenarios examine the effectiveness of AEBS in detecting and mitigating collisions with pedestrians crossing the road at predefined speeds and trajectories.

Each scenario entry in the dataset contains a unique identifier, test title, and a detailed textual description outlining the specific test conditions.

a) Dataset construction and verification.: To construct the ground-truth dataset, we systematically reviewed UN Regulation No. 152 and identified the three scenario categories defined in the standard: CtoStC, CtoMoC, and CtoP. For each category, we first extracted the general test conditions, which are distributed across multiple sections of the regulation. For example, the CtoStC conditions include the requirement from Section 5.2.1.4 that “the subject vehicle shall approach the lead vehicle in a straight line for at least 2 s before the functional part of the test commences.” We then combined these general conditions with the scenario-specific numerical requirements for each test case. Based on this combined information, we created a template scenario for each category, then instantiated each template by varying the relevant parameters according to the values specified in the regulation. For CtoStC, for instance, we varied approach speed using the values listed in the “Maximum Relative Impact Speed (km/h) for M1 vehicles” table and repeated the process separately for laden and unladen vehicle conditions; the corresponding post-condition requirements were filled from the applicable regulation values for the stationary target case. The same procedure was followed for CtoMoC and CtoP. After the initial extraction and construction, all resulting scenarios were reviewed for consistency with the regulation text and crosschecked against the original source sections and tables to confirm accuracy.

Phase 1: Extraction  
![](images/a808a4d3d008bbfb4b23f66888ae024943cab7248f36747e155a5acb035783c6.jpg)  
Phase 2: Chunking  
Phase 3: Retrieve and Generation

## Fig. 2. RegulaRAG pipeline

## B. Evaluation system

To evaluate the performance of different Large Language Models (LLMs), we employ a structured scoring system based on the dataset we have prepared. The generated test scenarios from each LLM are compared against our manually curated ground truth dataset to assess their accuracy and completeness.

The scoring process begins with the RAG system, which retrieves relevant context from UN Regulation No. 152 to generate test scenarios based on an initial prompt. The output is then directly compared to the manually constructed ground truth scenarios.

For evaluation of generated scenarios, we use a different embedding model, namely sentence-transformers/all-MiniLM-L6-v2<sup>4</sup> from the same sentence-transformers library (v3.4.1), which produces 384-dimensional embeddings. This model differs from the retrieval model and is computationally lighter, making it well-suited for large-scale pairwise similarity scoring between generated and ground-truth scenarios. We use this model to create vector embeddings for both the generated and ground-truth scenarios.

The similarity measurement is conducted using cosine similarity. We chose it as it is widely adopted in semantic text comparison tasks due to its efficiency, scale-invariance, and robustness to variations in text length [26].

One key challenge in LLM-generated scenarios is the incorrect assignment of numerical values from the regulation text. While LLMs tend to generate the general structure and wording of the test scenarios correctly, they often fail to extract and apply numerical values accurately. For example, a scenario requires the subject vehicle to travel at 20 km/h (with a tolerance of +0/-2 km/h) before braking. This speed value should be derived from the regulation statement: ”Tests shall be conducted with a vehicle travelling at 20, 42, and 60 km/h (with a tolerance of +0/-2 km/h).” However, LLMs sometimes generate incorrect speed values that are not supported by the regulation text. Similarly, the post-condition requirement, which dictates that the ”AEBS must decrease the vehicle speed to 0 km/h”, should be extracted from the Maximum Relative Impact Speed (km/h) for M1 vehicles table (see Figure 5). LLMs often fail to distinguish between laden and unladen conditions, leading to incorrect post-condition values in the generated scenarios.

However, to ensure a more nuanced evaluation, we incorporate a penalization mechanism that accounts for critical differences between the generated and reference scenarios. This penalization ensures that minor lexical variations are tolerated while significant discrepancies, such as omitted or incorrect details, are appropriately reflected in the final evaluation. A related scoring challenge is that cosine similarity scores tend to be very high, often close to 1.0, due to the fact that the LLM correctly reproduces the scenario template but fails to assign proper numerical values. To address this, we implemented a regex-based matching approach that extracts all numerical values along with the terms laden and unladen from both the generated and ground truth scenarios. The extracted values are then compared using string matching. While cosine similarity might yield a score around 0.95, we introduce a penalty weight λ = 0.2 applied once per discrepancy identified in the numerical values or laden/unladen conditions. This adjustment ensures that LLMs producing incorrect numerical values receive a final score below 0.9 (θ in Algorithm 2 ), which we classify as not equivalent to the ground truth. This refined approach enhances the robustness of our scoring mechanism, ensuring that the most critical numerical details are accurately assessed and penalized when incorrect.

To make scenario matching sensitive to critical compliance attributes that cosine similarity alone often misses (notably numeric parameters and loading conditions), we extract two structured key sets from each scenario text: (i) numeric values and (ii) load/condition terms. Formally, we define two regularexpression families:

Numeric pattern set $\mathcal { R } _ { \# }$ . We extract signed integers and decimals (supporting both decimal dots and commas) using:

$$
\mathcal { R } _ { \# } : \quad \left( \sf { 2 } < \sf { 1 } \big \langle { \sf w } \big \rangle \mathbb { I } + \big \langle - \big \rangle \mathbb { 2 } \big \langle \sf { d } + \left( \sf { 2 } : \left[ \tau , \tau \right] \big \langle \sf { d } + \right) \mathbb { 2 } . \right.
$$

Prior to extraction, we normalize the text by (a) converting decimal commas to decimal dots $( \mathbf { e . g . , 0 , 2 }  0 . 2 )$ , and (b) splitting fused unit strings (e.g., $\mathrm { } ^ { 6 6 } 2 \mathrm { s } ^ { 3 9 }  \mathrm { } ^ { 6 6 } 2 \mathrm { s } ^ { 3 9 } )$

Load/condition term set $\mathcal { R } _ { \ell } .$ We extract regulation-relevant categorical flags as case-insensitive keywords and de-duplicate them:

R<sub>ℓ</sub>: \b(?:unladen|laden|MassRun|MaxMass|   
Maximum\s+mass|Mass\s+in\s+running   
\s+order|stationary|moving|stopped)\b

Given a scenario text x, the extraction function returns $K ( x ) = \bigl ( N ( x ) , L ( x ) \bigr )$ , where $N ( x )$ is the list of numbers extracted by $\mathcal { R } _ { \# }$ after normalization/masking, and $L ( x )$ is the set of categorical terms extracted by $\mathcal { R } _ { \ell } .$ . We then compute a discrepancy score $\Delta$ between two scenarios by combining: (i) a tolerance-based one-to-one matching F1 over numeric lists (with absolute tolerance $\varepsilon _ { \mathrm { a b s } }$ and relative tolerance $\varepsilon _ { \mathrm { r e l } } )$ and (ii) Jaccard distance over the categorical term sets. This discrepancy is converted into a penalized score. Concretely, for a candidate pair of generated scenario h and ground-truth scenario g:

$$
s ( h , g ) \ = \ s _ { \mathrm { c o s } } ( E _ { h } , E _ { g } ) \ - \ \lambda \Delta \big ( K ( g ) , K ( h ) \big )\tag{1}
$$

where $s _ { \mathrm { c o s } }$ is the cosine similarity between sentence embeddings $E _ { h }$ and $E _ { g } , \lambda$ is the penalty weight (default $\lambda \ =$ 0.2), and $\Delta$ is the combined numeric/categorical discrepancy defined above. The complete matching and F1 computation procedure is formalised in Algorithm 2.

a) Metric generalizability.: The evaluation framework comprises two layers with different degrees of domain specificity. The penalized F1 metric (Eq. 1) is structurally regulation-agnostic: the cosine-similarity backbone, tolerancebased numeric matching, and Jaccard-distance categorical comparison can in principle be applied to any domain that produces structured textual artefacts with verifiable numerical parameters (e.g., pharmaceutical dosage specifications, financial compliance documents, or aviation checklists). The domain-specific elements are confined to the two regex families: $\mathcal { R } _ { \# }$ captures generic numeric expressions and requires minimal adaptation across domains, whereas $\mathcal { R } _ { \ell }$ encodes UN-152-specific load-condition flags (laden, unladen, MassRun, etc.) that must be redefined to match the categorical vocabulary of another standard. The penalty weight λ is a tunable hyperparameter, not a regulation-specific constant. Similarly, the

Algorithm 2: Penalized Scoring of Generated Scenar  
ios   
Input: Ground-truth CSV G; Generated CSV H; threshold   
θ; penalty weight λ (default $\lambda = 0 . 2 ) ;$ regex sets $\mathcal { R } _ { \# }$   
(numbers), R<sub>ℓ</sub> (load terms).   
Output: Precision, Recall, F1; counts TP/FP/FN.   
Load all rows from G and H; keep Unique ID and text;   
Initialize $\mathrm { T P } \gets 0 , \mathrm { F P } \gets 0 , \mathrm { F N } \gets 0 ;$   
Mark all $h \in \mathcal H$ as unmatched;   
foreach ground-truth scenario $g \in { \mathcal { G } }$ do   
$E _ { g } \gets \mathrm { e n c o d e } ( g . t e x t ) ;$   
$\dot { K _ { g } }$ ← extract(g.text; $\mathcal { R } _ { \# } \cup \mathcal { R } _ { \ell } )$ ; // numbers +   
load flags   
$s ^ { \star }  \infty , \bar { h ^ { \star } }  \infty ;$   
foreach generated scenario $h \in \mathcal H$ do   
$E _ { h } \gets \mathrm { e n c o d e } ( h . t e x t ) ;$   
${ K } _ { h } \gets \mathrm { e x t r a c t } ( h . t e x t ; \mathcal { R } _ { \# } \cup \mathcal { R } _ { \ell } ) ;$   
$s _ { \mathrm { c o s } } $ cosine similarity $( E _ { h } , E _ { g } ) ;$   
$\Delta  \mathrm { d i f f } ( K _ { g } , K _ { h } ) \ ;$ $/ /$ count   
discrepancies   
$s \gets s _ { \mathrm { c o s } } - \lambda \Delta ~ ; ~ / /$ apply penalty: λ per   
discrepancy   
if $s > s ^ { \star }$ then   
$s ^ { \star }  s ;$   
$h ^ { \star }  h ;$   
if $s ^ { \star } \geq \theta$ then   
Mark $h ^ { \star }$ as matched;   
$\mathrm { T P \gets T P + 1 ; }$   
else   
FN ← FN + 1;   
$\mathrm { F P }  | \{ h \in \mathscr { H } : h$ matched}| − TP;   
Precision $ \frac { \mathrm { T P } } { \mathrm { T P + F P } } ;$   
Recall $\scriptstyle \longleftarrow \frac { \mathrm { T } \bar { \mathrm { P } } } { \mathrm { T } \mathrm { P } + \mathrm { F N } } ;$   
F1 ← 2·Precision·Recall   
Precision+Recall

Meta-Score $( { \bar { F } } _ { 1 } - \sigma )$ is fully domain-agnostic: it summarises mean performance and cross-condition stability using any compatible base metric and is applicable whenever a system is evaluated across multiple structured task categories. In summary, adapting the metric to a new regulatory domain requires (i) replacing $\mathcal { R } _ { \ell }$ with domain-relevant categorical keywords, and (ii) re-tuning λ and the matching threshold θ on a representative sample; the rest of the framework transfers without modification.

b) Compliance-oriented metrics.: Automotive regulations do not define a universal compliance-accuracy metric; compliance is usually assessed through regulation-specific pass/fail criteria, tolerances, and traceability. In our evaluation, these aspects are encoded in the manually constructed ground truth scenarios from UN Regulation No. 152. Thus, the reported F1-score serves as a proxy for scenario-level compliance, while the penalized scoring explicitly checks critical numerical values $( \mathcal { R _ { \# } } )$ and loading conditions $( \mathcal { R } _ { \ell } )$ . Dedicated metrics such as Requirement Coverage, Parameter Accuracy, and Traceability Coverage would provide additional insight and are left as future work, informed by recent advances in LLM-based requirements traceability [27]–[29].

## IV. EXPERIMENTAL SETUP

## A. Language Models

We report results with three representative LLMs: (i) gpt-4o-2024-11-20 accessed via the OpenAI API<sup>5</sup>; (ii) Llama 3.3 70B (“llama-3.3-70b-versatile”) served through the Groq API<sup>6</sup>; and (iii) DeepSeek-chat (API model identifier: deepseek-chat) invoked via the vendor’s direct API<sup>7</sup>. RegulaRAG operates upstream of generation: it determines which parts of the regulation reach the prompt, and the language model is an interchangeable component of the pipeline rather than part of the contribution. The three models were therefore selected to span distinct deployment settings — a proprietary API model, an openly available model, and a low-cost API model — rather than to identify a best-performing generator, and pinned model snapshots (e.g., gpt-4o-2024-11-20) were preferred so that runs remain reproducible. Because every RAG system is evaluated under the same generator with identical prompts, scenario definitions, and decoding settings (Section V-A2), the comparison isolates the retrieval strategy, and a different or newer generator can be substituted without any change to the pipeline. DeepSeek offers both a reasoning model (DeepSeek-reasoner) and a chat model; We adopt the chat variant because, in our setting, it achieves competitive accuracy while being over 12× cheaper than the reasoning model (\$0.14 vs. \$1.74 per million input tokens on a cache miss), making it the practical choice for large-scale pipeline execution. Note that the DeepSeek API does not expose a pinned checkpoint version for the deepseek-chat endpoint; the underlying model may be updated by the provider without a version change in the model string. All models are run with low temperature $( \tau = 0 . 1 )$ to reduce output variance; decoding settings are held fixed across methods. Retrieval hyperparameters (top k and chunk size) are shared across all LLMs and determined via grid search, as described in Section V-A1.

## B. Evaluation Protocol

We report Precision, Recall, and F1 at the scenario level, using F1 as the primary metric. Each failing run is automatically retried up to two additional times and marked ok upon the first successful completion. For corpus-scaling experiments (Section V-B), we additionally record end-to-end runtime. Scenario matching uses a cosine similarity threshold of $\theta = 0 . 9 \mathrm { : }$ : a generated scenario is counted as a true positive only if its penalized similarity to the best-matching groundtruth scenario meets or exceeds this value. Numeric values within scenarios are compared using an absolute tolerance $\varepsilon _ { \mathrm { a b s } } = 0 . 0 5$ and a relative tolerance $\varepsilon _ { \mathrm { r e l } } = 0 . 0 5 ~ ( 5 \% )$ . Due to resource constraints, each (RAG system, LLM, scenariotype) configuration was evaluated in a single run; trial-level dispersion statistics across repeated executions are therefore not reported. This limitation is explicitly acknowledged in Section VI.

## User Prompt for Car-to-Moving-Car (Cto-MoC)

You are an autonomous driving engineer specializing in AEBS testing for M1 category vehicles. You must create test scenarios following UN Regulation No. 152 exactly as specified, with particular attention to: - Vehicle positioning and approach conditions - Precise speed requirements including tolerances - Time To Collision (TTC) specifications - Loading conditions (laden/unladen) - Clear post-condition requirements

Fig. 3. System prompt for Car-to-Moving-Car (CtoMoC)

## C. Compute and Reproducibility

Experiments were executed on a Linux workstation with an NVIDIA GeForce GTX 1080 Ti (11 GB GDDR5X), an Intel<sup>®</sup> Core™ i7-3770 CPU @ 3.40 GHz, and 32 GB RAM. The software stack uses Python 3.10.12. Random seeds (where applicable), prompt templates, and decoding parameters are fixed across conditions. The ground-truth scenario dataset is publicly available on Hugging Face (vahidzolf/un\_152; see Section III-A). Code, prompt templates, and evaluation CSVs necessary to reproduce the results are publicly available in the project repository<sup>8</sup>; API keys are required for commercial models.

Two distinct sentence-transformer models from the sentence-transformers library (v3.4.1) are used at separate pipeline stages:

• Retrieval: all-mpnet-base-v2 (768-dim) — embeds enriched chunks and queries for similarity-based top-k selection in Phase 3.

• Evaluation: all-MiniLM-L6-v2 (384-dim) — computes pairwise cosine similarity between generated and groundtruth scenarios in the penalized scoring metric.

## D. Prompting System

To obtain consistent outputs from all language models (LLMs), we used identical prompts across all models without applying any advanced prompting techniques. A single system prompt was used to describe the general task, and three user prompts were designed, each targeting a specific type of scenario: Car-to-Stationary-Car, Car-to-Moving-Car, and Carto-Pedestrian.

These user prompts requested the generation of scenarios according to the UN Regulation No. 152, each including a single example to act as a structural template. You can find the complete user prompts in Appendix A. We deliberately avoided including instructions on how to derive the scenarios in prompts, to keep the LLMs task-focused and to support automated evaluation against a ground truth dataset. The example format ensured that all outputs followed a consistent structure, enabling effective comparison and scoring.

## V. RESULTS AND EXPERIMENTS

## A. Experiment Design

To thoroughly evaluate our system, a staged experimental design was adapted. The search space is large because multiple pipeline components interact; therefore, we decompose the problem into controllable factors and explore them systematically rather than exhaustively. Accordingly, the following evaluation dimensions was defined, each capturing a distinct aspect of the RAG pipeline that must be scrutinized to understand performance, robustness, and scalability:

1) RAG system We evaluated six RAG systems: RegulaRAG (our method), R&R+RCS, NoRAG, OpenAI-RAG, HippoRAG, and Hybrid.

2) Scenario family Car-to-Stationary-Car (CtoStC), Car-to-Moving-Car (CtoMoC), and Car-to-Pedestrian (CtoP). Section V-A2 reports the corresponding results of this and previous item.

3) Retrieval breadth (top k) top k ∈ {10, 15, 20, 25, 30}.

4) Chunking granularity (chunk size)

To assess the impact of chunk granularity on RAG performance, the chunk size varied across a range of values. Section V-A1 reports the corresponding results.

5) Source size (corpus scale) To assess robustness under semantically similar distractors, we progressively expand the retrieval corpus using eight UN regulations in the automotive software domain, including UN Regulation No. 152. This setup enables a systematic evaluation of the scalability, strengths, and weaknesses of our method relative to the strongest competing RAG baseline. section V-B discusses the result of this experiment.

6) Language model (LLM) GPT-4o was used as a widely adopted commercial model, and include DeepSeek-Chat and Llama 3.3 to probe cost/latency and open-weight generalization.

An exhaustive grid over all factors would be computationally prohibitive and confounds effects across dimensions. Instead, we adopt a progressive design that (i) first identifies stable hyperparameters for retrieval by performing grid search, (ii) then compares RAG systems under matched conditions (to isolate retrieval effects), and (iii) finally stresses the systems with larger corpora (to assess scaling and distractor robustness). This ordering controls nuisance variability and yields conclusions that are both fair and reproducible. Following is the sequence of experiments:

1) Hyperparameter selection via grid search: We estimate robust retrieval hyperparameters by performing a full grid sweep on the CtoStC scenario using UN-152 as the sole source. We evaluate:

top k ∈ {10, 15, 20, 25, 30}

chunk size ∈ {800, 1000, 1400, 1600, 2000, 3000, 4000} for our RegulaRAG method when using GPT-4o for the final scenario generation. Figure 4 presents F1 as a heatmap over the (chunk size, top k) grid. This pattern is consistent with RAGGED [30], which finds that readers exhibiting an improve-then-plateau response to retrieval depth — a trend GPT-4o follows alongside GPT-3.5-turbo — incur substantially smaller performance loss when operating away from their individually optimal top k than noise-sensitive readers (e.g., LLaMA, Claude); this may partly explain why top k=30 remained effective across the eight-regulation scalability experiment (Section V-B). The resulting response surface shows broad plateaus with a few sharp maxima. This plot reveals that, despite a few outliers in the heatmap, the overall performance trend stabilizes around (top k=30, chunk size=2000). At this configuration the F1-score reaches its peak before gradually degrading for larger values, indicating that it represents a near-optimal balance between retrieval breadth and chunk granularity. To ensure a fair comparison, all retrieval-based baselines (R&R+RCS, OpenAI-RAG, HippoRAG, and Hybrid) were subsequently evaluated using the same top k=30 value identified here.

2) RAG system comparison.: With (top k<sup>∗</sup>, chunk size<sup>∗</sup>) fixed, we evaluated six Retrieval-Augmented Generation (RAG) pipelines across all three scenario families (CtoStC, CtoMoC, and CtoP) on UN–152:

• RegulaRAG — our method: SmartChunking with reference-aware enrichment + smart retrieval: Smart Retrieve and Rerank which is integrated into our RegulaRAG pipeline. It compares the enriched chunks (including referenced content) to the query, then gathers the corresponding original chunks and appends the reference chunks after deduplication.

R&R+RCS (rr rcs) — baseline: RecursiveCharacter-Splitter (RCS) + simple retrieval: this method uses the allmpnet-base-v2<sup>9</sup> model, the same model as the one we used for our regulaRAG method, to embed the chunks and queries. The standard version performs retrieval directly over the original chunks and returns the top-k chunks based on similarity.

• OpenAI-RAG (openai rag) — baseline: RCS + OpenAI embeddings + Chroma: In this method, we use OpenAI’s text-embedding-ada-002 model<sup>10</sup> to generate chunk embeddings, which are stored in a Chroma vector database. Retrieval is performed using LangChain’s vector store-backed retriever with the similarity score threshold search type and parameters search kwargs={”k”: 30, ”score threshold”: 0.1}.

• HippoRAG — baseline: As a state-of-the-art knowledge graph-based RAG method, HippoRAG [31] is included to benchmark our retrieval pipeline against recent graphbased systems. It has shown competitive performance in recent benchmarks.

• NoRAG — reference baseline: Direct LLM generation without any retrieval context. The model receives only the user prompt and scenario-format example, with no regulatory document chunks provided. Included to measure the scenario-generation capability embedded in model weights alone, quantifying the marginal benefit of the retrieval pipeline.

• Hybrid — baseline: A hybrid retrieval pipeline that combines sparse keyword-based (BM25) retrieval with dense semantic (embedding-based) retrieval over RCSgenerated chunks, merging the two ranked lists before context assembly.

<table><tr><td colspan="2"></td><td colspan="5">RegulaRAG · GPT-4o — F1 heatmap by chunk_size × top_k</td></tr><tr><td></td><td>10</td><td>15</td><td>top_k 20</td><td>25</td><td>30</td><td>-90</td></tr><tr><td>800</td><td>22.22</td><td>62.86</td><td>22.22</td><td>22.22</td><td>22.22</td><td></td></tr><tr><td>1000</td><td>45.16</td><td>22.22</td><td>76.92</td><td>76.92</td><td>76.92</td><td>-80 -70</td></tr><tr><td>1400</td><td>22.22</td><td>22.22</td><td>22.22</td><td>22.22</td><td>22.22</td><td></td></tr><tr><td>chunize 1600 2000</td><td>76.92</td><td>88.37</td><td>54.55</td><td>76.92</td><td>70.27</td><td>60 060 F -50</td></tr><tr><td></td><td>22.22</td><td>62.86</td><td>22.22</td><td>22.22</td><td>93.33</td><td>-40</td></tr><tr><td>3000</td><td>22.22</td><td>70.27</td><td>76.92</td><td>76.92</td><td>54.55</td><td></td></tr><tr><td>3500</td><td>62.86</td><td>54.55</td><td>54.55</td><td>70.27</td><td>54.55</td><td>-30</td></tr><tr><td>4000</td><td>70.27</td><td>70.27</td><td>70.27</td><td>22.22</td><td>54.55</td><td></td></tr></table>

Fig. 4. Hyperparameter Selection

This design isolates the contribution of the chunking and retrieval strategy by keeping the scenario definitions, prompts, and model-specific decoding settings constant. We report F1 score as the primary metric and track precision/recall for diagnostic completeness. The goal is to establish whether RegulaRAG’s reference-aware enrichment translates into consistent gains across distinct task conditions. The results are presented in Table II. ↑ arrows are for metrics where higher is better and ↓ arrows are for metrics where lower is better. To capture both the average performance and the stability of each RAG system, we introduce a Meta-Score, computed from the F1-scores of the three evaluated LLMs (GPT-4o, DeepSeek-chat, and LLaMA-3.3). While Algorithm 2 described the penalized evaluation of individual scenarios which yields the F1-score, our goal here is to obtain a metric that also reflects the variance of LLM performance and summarizes how consistently a given RAG–LLM pipeline behaves across different scenario categories. The Meta-Score therefore provides a more holistic indicator of system-level robustness.

Let $\bar { F } _ { 1 }$ denote the mean of the three F1 values for 3 types of scenarios for a given RAG system, and let σ denote their standard deviation. The Meta-Score is then defined as:

$$
\begin{array} { r } { \mathrm { M e t a S c o r e } = \bar { F } _ { 1 } - \sigma . } \end{array}
$$

We propose the Meta-Score as a robustness-adjusted aggregation designed for this study; it is not a standard benchmark metric in RAG evaluation. Its formulation is inspired by the classical mean–variance tradeoff principle [32] and the onestandard-error model-selection heuristic of [33], both of which penalize high-variance estimators by combining mean performance with a dispersion penalty. We employ the coefficient of variation (CV) as a normalized measure of relative variability, a well-established statistical tool for comparing dispersion across scales.

This formulation rewards RAG systems with high average F1 performance while penalizing those with high variability across the three LLMs, thereby reflecting both accuracy and robustness. Negative values in Table II arise naturally from the definition of the Meta-Score.When a RAG system produces highly inconsistent results across the three scenario categories like scoring near 0 on one task while performing moderately on others, the variance term becomes large and dominates the average performance, yielding a negative Meta-Score.

Table II reveals three clear patterns. First, RegulaRAG consistently achieves the strongest overall performance and stability across all language models. Across GPT-4o, DeepSeekchat, and LLaMA-3.3, RegulaRAG attains the highest average Meta-Score of 82.99, clearly outperforming all competing methods. In comparison, the second-best system (NoRAG) achieves an average Meta-Score of 57.94, followed by Hybrid (56.71) and HippoRAG (55.75).

This corresponds to an improvement of approximately 43%. At the individual model level, RegulaRAG maintains consistently high Meta-Scores across all LLMs (80.99 for GPT-4o, 87.90 for DeepSeek-chat, and 80.08 for LLaMA-3.3), whereas competing methods exhibit substantial variability and, in some cases, severe degradation (e.g., negative Meta-Score for DeepSeek-chat under R&R+RCS).

The baseline comparison also provides partial ablation evidence for RegulaRAG’s core components . R&R+RCS shares the same embedding model (all-mpnet-base-v2) and the same RecursiveCharacterSplitter chunking as RegulaRAG but omits reference-aware BFS enrichment and smart reranking; the large Meta-Score gap between RegulaRAG and R&R+RCS (82.99 vs. 8.88, averaged across LLMs) therefore isolates the joint contribution of those two components. NoRAG, which supplies the LLM with no retrieval context at all, establishes a lower bound on scenario-generation capability embedded in model weights alone (average Meta-Score 57.94); comparing it with RegulaRAG quantifies the marginal value of the entire retrieval pipeline. A fully factorial component-level ablation—isolating BFS reference-closure depth, deduplication, and canonical reordering individually—remains as future work.

A manual inspection of retrieved contexts shows that the presence and correct ordering of the necessary table chunks is the primary determinant of high F1. Simply retrieving a relevant table slice is insufficient when the original PDF table is split across pages (Fig. 5) and subsequently appears as multiple HTML tables (Fig. 6). To address this, RegulaRAG uses a rule-based heuristic for table detection and reconstruction. Specifically, when a paragraph contains cues such as “the following table,” we treat this as an explicit pointer to subsequent table content. In the Markdown/HTML output produced by Mineru, tables are marked by <table> tags. We therefore collect the consecutive <table> blocks that immediately follow the referring paragraph and attach them to that paragraph as referenced table metadata. This heuristic is particularly useful for PDF tables that are sliced across pages and converted into multiple adjacent HTML table fragments: by grouping consecutive table tags and preserving their original order, the system reconstructs the logical table content before retrieval and prompt assembly.

RegulaRAG’s reference-aware chunk reordering ensures these fragments are placed adjacently and in the correct logical order, improving recall and reducing hallucinated numerical substitutions.

Second, the three scenario categories differ markedly in difficulty. CtoP remains the easiest because it is anchored in a self-contained regulatory section (section 5.2.2. Car to pedestrian scenario in [17] ) with a dedicated table. In contrast, CtoStC and CtoMoC require integrating information dispersed across multiple paragraphs and shared tables, making them more sensitive to retrieval errors.

Third, the language models respond differently to retrieval conditions. DeepSeek-chat performs competitively under RegulaRAG but becomes highly unstable under weaker retrieval methods, as reflected by large variance and several negative Meta-Scores for R&R+RCS and OpenAI-RAG. LLaMA-3.3 shows consistent improvement under RegulaRAG but struggles substantially with baselines, especially in scenarios requiring precise table interpretation and paragraph-level aggregation. GPT-4o displays the most stable behavior under RegulaRAG, yet—even for GPT-4o—the baseline RAG systems exhibit large variability across tasks.

Another reason that we got better results comparing to the baseline methods is that, thanks to our mechanism for untangling the chain of references distributed across different chunks, the input query is then compared against the entire chain of referenced content along with the chunk’s original text, thereby increasing the likelihood of selecting the most relevant chunks. For example, when the input query pertains to a Car-to-Pedestrian scenario in UN Regulation No. 152, a standard chunking and retrieval method may select paragraph 5.2.2, but miss the “Maximum Impact Speed (km/h) for M1 vehicles” table, which contains the required speed values. This happens because the table shares no explicit keywords or semantic similarity with the query. In contrast, our method detects that paragraph 5.2.2 contains phrases like “the following table,” prompting a search for and attachment of that table to the chunk. This increases the likelihood of retrieving the chunk and provides the LLM with all necessary data to generate complete scenario variants.

a) Qualitative example.: To illustrate the end-to-end pipeline, consider a CtoStC scenario from UN Regulation No. 152. Figure 1 shows how the relevant conditions are distributed across cross-linked sections: the test speed values reside in a table, and the approach constraint is stated in other Section. Figure 5 shows how the speed table is split across two PDF pages and emerges as separate HTML fragments after conversion. RegulaRAG’s SmartChunking detects the “the following table” cue, attaches the consecutive HTML table fragments, and add it as metadata to the original chunk. The Smart Retrieve and Rerank module then retrieves the enriched chunk,and with this assembled context, the LLM generates the scenario. Table III shows the expected output (ground-truth entry Test\_CtoStC\_unladen\_10 from the published dataset, Section III-A). Without reference-aware enrichment the speed table is not retrieved, and the LLM either omits the speed value or hallucinates one which is the primary failure mode described in Section V-C.

## TABLE III

GROUND-TRUTH SCENARIO TEST\_CTOSTC\_UNLADEN\_10 — EXPECTED LLM OUTPUT FOR A CTOSTC TEST AT 10 KM/H (UNLADEN).

<table><tr><td rowspan=1 colspan=1>Two vehicles of Category M1 AA saloon shall be positioned.The subject vehicle that performs the braking and the leadvehicle.</td></tr><tr><td rowspan=1 colspan=1>Both vehicles should face in the same direction of travel.</td></tr><tr><td rowspan=1 colspan=1>The subject vehicle shall approach the lead vehicle in astraight line for at least 2 s before the functional part ofthe test commences.</td></tr><tr><td rowspan=1 colspan=1>The subject vehicle should travel at the speed of 10 km/h(with a tolerance of +0/—2 km/h) when the vehicle brakes.</td></tr><tr><td rowspan=1 colspan=1>The functional part of the test shall start at a distancecorresponding to a Time To Collision (TTC) of at least4 seconds from the target.</td></tr><tr><td rowspan=1 colspan=1>The subject vehicle is unladen.</td></tr><tr><td rowspan=1 colspan=1>The lead vehicle is stationary.</td></tr><tr><td rowspan=1 colspan=1>Post-condition requirements:</td></tr><tr><td rowspan=1 colspan=1>When the system is activated, the AEBS shall decrease thespeed to 0 km/h.</td></tr></table>

## B. input document scalability

To assess robustness under growing evidence, we gradually expand the retrieval corpus from a single regulation (152) to bundles containing eight documents. The evaluation pool contains eight UN regulations of varying length and structural complexity:

• No. 10: Electromagnetic compatibility [34]

• No. 79: Steering equipment [35]

• No. 130: Lane Departure Warning System (LDWS) [36]

• No. 140: Electronic Stability Control (ESC) [37]

• No. 152: AEBS (primary target corpus) [38]

TABLE II  
PERFORMANCE, STABILITY, AND META-SCORE ACROSS RAG SYSTEMS AND MODELS
<table><tr><td>RAG System</td><td>Model</td><td>CtoStC (↑)</td><td>CtoMoC (↑)</td><td>CtoP (↑)</td><td>Mean (↑)</td><td>stdev (↓)</td><td>Meta-Score (↑)</td><td>Avg. 3 LLMs (↑)</td></tr><tr><td rowspan="3">RegulaRAG</td><td>GPT-40</td><td>76.92</td><td>96.55</td><td>100.00</td><td>91.16</td><td>10.16</td><td>80.99</td><td>82.99</td></tr><tr><td>DeepSeek-chat</td><td>85.71</td><td>96.55</td><td>97.44</td><td>93.23</td><td>5.33</td><td>87.90</td><td></td></tr><tr><td>Llama-3.3</td><td>76.92</td><td>96.55</td><td>91.89</td><td>88.45</td><td>8.37</td><td>80.08</td><td></td></tr><tr><td rowspan="3">R&amp;R+RCS</td><td>GPT-40</td><td>8.00</td><td>88.89</td><td>91.89</td><td>62.93</td><td>38.86</td><td>24.07</td><td>8.88</td></tr><tr><td>DeepSeek-chat</td><td>0.00</td><td>0.00</td><td>97.44</td><td>32.48</td><td>45.93</td><td>-13.45</td><td></td></tr><tr><td>Llama-3.3</td><td>76.92</td><td>0.00</td><td>91.89</td><td>56.27</td><td>40.26</td><td>16.01</td><td></td></tr><tr><td rowspan="3">NoRAG</td><td>GPT-40</td><td>22.22</td><td>96.55</td><td>100.00</td><td>72.92</td><td>35.88</td><td>37.04</td><td>57.94</td></tr><tr><td>DeepSeek-chat</td><td>73.68</td><td>96.55</td><td>97.44</td><td>89.22</td><td>11.00</td><td>78.23</td><td></td></tr><tr><td>Llama-3.3</td><td>50.00</td><td>96.55</td><td>91.89</td><td>79.48</td><td>20.93</td><td>58.55</td><td></td></tr><tr><td rowspan="3">HippoRAG</td><td>GPT-40</td><td>62.86</td><td>96.55</td><td>97.44</td><td>85.62</td><td>16.10</td><td>69.52</td><td>55.75</td></tr><tr><td>DeepSeek-chat</td><td>90.91</td><td>80.00</td><td>97.44</td><td>89.45</td><td>7.19</td><td>82.26</td><td></td></tr><tr><td>Llama-3.3</td><td>73.68</td><td>0.00</td><td>91.89</td><td>55.19</td><td>39.73</td><td>15.46</td><td></td></tr><tr><td rowspan="3">OpenAI-RAG</td><td>GPT-40</td><td>62.86</td><td>94.00</td><td>99.15</td><td>85.33</td><td>16.03</td><td>69.30</td><td>43.53</td></tr><tr><td>DeepSeek-chat</td><td>30.30</td><td>0.00</td><td>97.44</td><td>42.58</td><td>40.72</td><td>1.86</td><td></td></tr><tr><td>Llama-3.3</td><td>75.79</td><td>55.37</td><td>91.89</td><td>74.35</td><td>14.94</td><td>59.41</td><td></td></tr><tr><td rowspan="3">Hybrid</td><td>GPT-40</td><td>54.55</td><td>96.55</td><td>97.44</td><td>82.85</td><td>20.01</td><td>62.83</td><td>56.71</td></tr><tr><td>DeepSeek-chat</td><td>88.37</td><td>96.55</td><td>97.44</td><td>94.12</td><td>4.08</td><td>90.04</td><td></td></tr><tr><td>Llama-3.3</td><td>85.71</td><td>0.00</td><td>91.89</td><td>59.20</td><td>41.94</td><td>17.26</td><td></td></tr></table>

Maximum relative Impact Speed (km/h) for M, vehicles
<table><tr><td rowspan=2 colspan=1>Relative Speed(km/h)</td><td rowspan=1 colspan=2>Stationary</td><td rowspan=1 colspan=2>Moving</td></tr><tr><td rowspan=1 colspan=1>Laden</td><td rowspan=1 colspan=1>Unladen</td><td rowspan=1 colspan=1>Laden</td><td rowspan=1 colspan=1>Unladen</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>0,00</td><td rowspan=1 colspan=1>0,00</td><td rowspan=1 colspan=1>0,00</td><td rowspan=1 colspan=1>0,00</td></tr><tr><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>0,00</td><td rowspan=1 colspan=1>0,00</td><td rowspan=1 colspan=1>0,00</td><td rowspan=1 colspan=1>0,00</td></tr><tr><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>0,00</td><td rowspan=1 colspan=1>0,00</td><td rowspan=1 colspan=1>0,00</td><td rowspan=1 colspan=1>0,00</td></tr><tr><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>0,00</td><td rowspan=1 colspan=1>0,00</td><td rowspan=1 colspan=1>0,00</td><td rowspan=1 colspan=1>0,00</td></tr><tr><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>0,00</td><td rowspan=1 colspan=1>0,00</td><td rowspan=1 colspan=1>0,00</td><td rowspan=1 colspan=1>0,00</td></tr><tr><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>0,00</td><td rowspan=1 colspan=1>0,00</td><td rowspan=1 colspan=1>0,00</td><td rowspan=1 colspan=1>0,00</td></tr><tr><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>0,00</td><td rowspan=1 colspan=1>0,00</td><td rowspan=1 colspan=1>0,00</td><td rowspan=1 colspan=1>0,00</td></tr><tr><td rowspan=1 colspan=1>42</td><td rowspan=1 colspan=1>10,00</td><td rowspan=1 colspan=1>0,00</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>0,00</td></tr></table>

30.10.2020

Official Journal of the European Union
<table><tr><td rowspan=2 colspan=1>Relative Speed(km/h)</td><td rowspan=1 colspan=2>Stationary</td><td rowspan=1 colspan=2>Moving</td></tr><tr><td rowspan=1 colspan=1>Laden</td><td rowspan=1 colspan=1>Unladen</td><td rowspan=1 colspan=1>Laden</td><td rowspan=1 colspan=1>Unladen</td></tr><tr><td rowspan=1 colspan=1>45</td><td rowspan=1 colspan=1>15,00</td><td rowspan=1 colspan=1>15,00</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>25,00</td><td rowspan=1 colspan=1>25,00</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>55</td><td rowspan=1 colspan=1>30,00</td><td rowspan=1 colspan=1>30,00</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ⅱ</td></tr><tr><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>35,00</td><td rowspan=1 colspan=1>35,00</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr></table>

Fig. 5. Example of split table

Maximum relative Impact Speed \$(\bf k m/h)\$ for \$\mathbf{M\_{1}}\$ vehicles

<html><body><table><tr><td rowspan="2">Relative Speed (km/h)</td><td colspan="2">Stationary</td><td colspan="2">Moving</td></tr><tr><td>Laden</td><td>Unladen</td>≤td>Laden</td><td>Unladen</td>...≤/tr></table></bodv>≤/html>

<html><body><table><tr><td rowspan="2">Relative Speed (km/h)</td><td colspan="2">Stationary</td><td colspan="2">Moving</td></tr><tr><td>Laden</td><td>Unladen</td><td>Laden</td><td>Unladen</td></tr><tr>...</tr></table></body></html>

## Fig. 6. Converted table (HTML) after PDF split

• No. 155: Cybersecurity and CSMS [39]

• No. 156: Software update and SUMS [40]

• No. 13-H: Braking (passenger cars) [41]

We fix $L L M ^ { * } = \mathrm { G P T } \mathrm { } { = } \mathrm { 4 } 0 , t o p _ { k } ^ { * } = 3 0$ , and chunk $s i z e ^ { * } =$ 3000, varying only the corpus size. This experiment isolates the effects of semantic clutter and cross-document interactions

on retrieval stability.

Alongside REGULARAG, we evaluate HIPPORAG, a knowledge-graph-driven system that performs LLM-based NER, triple extraction, and graph-structured retrieval. Comparing them under identical scaling conditions highlights contrasting computational behaviors. We include HippoRAG because it trails RegulaRAG across all categories in Table II.

Figures 7, 8, and 9 report F1-score, end-to-end runtime, and OpenAI token usage as the corpus grows.

a) Runtime stability.: Across all corpus sizes, REGU-LARAG maintains a stable runtime (47–59 s), indicating that its enrichment and reranking pipeline scales gracefully. The enrichment module has been optimized by removing deep copies, avoiding storage of enriched text, compressing metadata into integer references, and bounding BFS expansion. As shown in Figure 8, these optimizations enable RegulaRAG to maintain stable runtime even as the corpus grows. In contrast, HIPPORAG runtime grows steeply, exceeding 200 s on seven or more regulations, driven by its LLM-heavy OpenIE and graph-construction passes.

b) Token usage scaling.: The two models consume substantially different amounts of tokens. REGULARAG remains compact, with token usage fluctuating slightly around 14k–25k tokens. By contrast, HIPPORAG grows near-exponentially, from 34k tokens on a single document to nearly 500k tokens on the full set. This growth is driven by multiple LLMdependent stages—NER, triple extraction, graph expansion, reranking, and auxiliary calls.

c) Retrieval quality.: HIPPORAG delivers consistently strong accuracy (97–100% F1), reflecting the benefits of explicit graph-structured reasoning. REGULARAG, optimized for paragraph-level reference reconstruction, achieves high accuracy on smaller inputs but declines as the corpus becomes increasingly heterogeneous (down to 82.35%). This represents a natural trade-off: RegulaRAG prioritizes computational efficiency and highly targeted retrieval, whereas HippoRAG prioritizes robustness and semantic coverage, at the cost of substantially higher token usage and runtime.

![](images/9226946404f3160403bb36fb05a85b4b36fe4717795b566bcb4a71daf4226dc6.jpg)  
Fig. 7. F1-score scaling of REGULARAG and HIPPORAG as corpus size increases.

![](images/25b53625898c67316dc10e1d83153387c94a679394e2094ee9184bdb1612a261.jpg)  
Fig. 8. Runtime scaling for REGULARAG and HIPPORAG.

![](images/4dda3d04bd1dcfb1db35d73493cfc1cbab0b0cddbc2faaaa6dbcb895d16ddb35.jpg)  
Fig. 9. Token usage comparison across increasing corpus sizes.

## C. Failure Mode Analysis

Despite RegulaRAG’s strong overall performance, manual inspection of generated scenarios reveals three recurring failure patterns that affect all evaluated RAG systems to varying degrees.

(1) Incorrect numerical value assignment. LLMs tend to reproduce the structural template of a scenario correctly but fail to extract and apply numerical values accurately from the regulation text. For example, a CtoStC scenario requires the subject vehicle to travel at 20 km/h (tolerance +0/−2 km/h) before braking, derived from the regulation statement “Tests shall be conducted with a vehicle travelling at 20, 42, and 60 km/h.” LLMs sometimes substitute values not present in the regulation, particularly for speeds that must be read from the Maximum Relative Impact Speed table (Fig. 5).

(2) Loading condition confusion. LLMs frequently fail to distinguish laden from unladen conditions, generating incorrect post-condition values. The correct post-condition (e.g., the AEBS must reduce vehicle speed to 0 km/h) depends on loading state and must be read from the regulation table; confusing the two conditions yields numerically incorrect scenarios despite plausible wording.

(3) Cosine similarity inflation. Because LLMs reproduce the scenario template structure faithfully, cosine similarity between generated and ground-truth scenarios is often close to 1.0 even when numerical content is wrong. This inflates raw similarity scores and motivates the penalized metric (Algorithm 2), which explicitly penalises numeric and categorical discrepancies to surface these otherwise hidden errors.

RegulaRAG directly addresses failure modes (1) and (2) by ensuring the relevant table chunk is present in the retrieved context through reference-aware enrichment, thereby providing the LLM with the correct numerical grounding. Exact numeric hallucination that occurs even when the correct table is supplied—where the LLM misreads or ignores explicit table values—is a well-known problem across regulated-domain LLM applications and remains outside the scope of this work; mitigating it through prompting strategies or constrained decoding is a planned direction for future research.

## VI. CONCLUSION AND FUTURE WORK

In this paper, we introduced RegulaRAG, a two-stage Retrieval-Augmented Generation (RAG) pipeline designed to generate regulation-compliant test scenarios from complex technical standards such as UN Regulation No. 152. By integrating a SmartChunking strategy with reference-aware enrichment and a semantic retrieval mechanism, RegulaRAG offers a favorable balance between accuracy, runtime, and token usage. Across multiple scenario types and language models, RegulaRAG achieves the highest average Meta-Score (82.99, 43% above the next-best baseline NoRAG at 57.94), indicating strong and stable performance while operating at substantially lower context cost (14k–25k tokens per query) than graph-centric alternatives such as HippoRAG (up to 500k tokens on an eight-document corpus).

Future work will focus on further improving RegulaRAG, including refinements to chunk enrichment, scoring, and retrieval ranking. Finally, we aim to generalize the pipeline to additional regulatory documents and domains, thereby broadening its applicability in compliance-critical settings. We further envision integrating the generated scenarios directly with automotive simulation environments (e.g., CARLA, Car-Maker) and real testbench infrastructure.

## Limitations

Several limitations of the present work merit acknowledgment. First, RegulaRAG was developed and evaluated exclusively on UN Regulation No. 152. The core pipeline — SmartChunking, BFS reference-closure enrichment, and semantic retrieval — is regulation-agnostic and in principle applicable to any domain whose standards share similar structural properties: hierarchical numbered paragraphs, explicit cross-references, and tabular specifications. Domains such as pharmaceutical (e.g., ICH guidelines), aviation (e.g., DO-178C), or financial compliance standards share these structural characteristics and are plausible candidates for adaptation . Two bounded adaptation steps are required: (i) the categorical regex family $\mathcal { R } _ { \ell } ,$ which encodes UN-152 load-condition keywords (laden, unladen, MassRun), must be replaced with domain-relevant vocabulary; the cosine-similarity backbone, numeric matching, penalty weight λ, and Meta-Score formula are domain-agnostic and transfer without modification (see Section III-B for a full breakdown); and (ii) the retrieval hyperparameters (top k=30, chunk size=2000) were tuned via grid search on the CtoStC scenario family using UN-152 alone; regulations with substantially different document density, cross-reference depth, or chunk length distributions may require re-tuning of these parameters before deployment. This transfer risk may be partially bounded for top k specifically: prior work on reader noise-sensitivity [30] suggests readers with an improve-then-plateau retrieval-depth response — a pattern GPT-4o follows alongside GPT-3.5-turbo — degrade less when top k is not re-tuned for a new corpus than noisesensitive readers; this bound does not extend to chunk size, which is not examined in that work, and re-tuning both parameters remains advisable. Second, the same subject-matter experts contributed to both dataset construction and qualitative evaluation, which introduces a potential consistency bias; role separation or an independent validation cohort would strengthen future evaluations. Third, RegulaRAG performs reference-aware chunk enrichment and reordering but does not construct or reason over a full knowledge graph (see Section II for a detailed discussion): the legal language of regulatory standards — nested clauses, conditional syntax, and domainspecific terminology — makes reliable general-purpose entity and relation extraction challenging, and KG-based pipelines incur substantially higher token and runtime costs as corpus size grows (Section V-B). Systems that build explicit entity– relation graphs may capture deeper cross-reference semantics at the cost of this additional complexity.

Finally, owing to resource constraints, each configuration was evaluated in a single run without repeated-trial statistics. While the baseline comparison provides partial ablation evidence—R&R+RCS isolates the contribution of BFS referenceaware enrichment and smart reranking, and NoRAG isolates the entire retrieval pipeline—a component-level ablation isolating BFS reference-closure depth, deduplication, and canonical chunk reordering individually was not conducted; these remain open empirical questions for future work.

## USER PROMPTS FOR SCENARIO GENERATION

In addition to the system prompt provided in Listing IV-D, the language model receives a scenario-specific user prompt. These prompts instruct the model to generate AEBS test scenarios in a structured format that mirrors the examples defined in UN Regulation No. 152. The three user prompts corresponding to the Car-to-Stationary-Car (CtoStC), Car-to-Moving-Car (CtoMoC), and Car-to-Pedestrian (CtoP) scenario families are listed below.

## User Prompt for Car-to-Stationary-Car (CtoStC) CONTEXT: {rag\_context}

Based on the context provided (from UN Regulation No. 152), generate all test scenarios related to Car to stationary Car scenario (CtoStC) tests for M1 vehicles. The technical service is strictly required to test any speed, therefore a complete list of all test cases (with the speed range defined in regulation tables) must be produced. The system shall be active at least within the vehicle speed range specified by Maximum Relative Impact Speed (km/h) for M1 vehicles.

Follow this exact format for every test scenario: 1- ”Write the scenario title (e.g., Test\_CtoStC\_unladen\_42) as the first line.” 2- ”Immediately after the title, write the full scenario description.” 3- ”After each complete test case, add exactly one separator line: ,9

Formatting rules: - Use the provided example structure strictly. - Do not add explanations, notes, or extra text. - Only output the requested test scenarios and separators. - Do not include index numbers or any other identifiers.

Example: {example\_test\_case}

[Start generation now.]

## User Prompt for Car-to-Moving-Car (CtoMoC) CONTEXT: {rag\_context}

Based on the context provided (from UN Regulation No. 152), generate all test scenarios related to Car to moving Car scenario (CtoMoC) tests for M1 vehicles. The technical service is strictly required to test any speed, therefore all speeds defined in the regulation tables must be included. The system shall be active at least within the vehicle speed range specified by the Maximum Relative Impact Speed (km/h) for M1 vehicles.

Follow this exact format for every test scenario: 1- ”Write the scenario title (e.g., Test\_CtoMoC\_unladen\_10) as the first line.” 2- ”Immediately after the title, write the full scenario description.” 3- ”After each complete test case, add exactly one separator line: ，

Formatting rules: - Use the provided example structure strictly. - Do not add explanations, notes, or extra text. - Only output the requested test scenarios and separators. - Do not include index numbers or any other identifiers.

EXAMPLE: {example\_test\_case}

[Start generation now.]

User Prompt for Car-to-Pedestrian (CtoP)

## CONTEXT: {rag\_context}

Based on the context provided (from UN Regulation No. 152), generate all test scenarios related to Car to pedestrian scenario (CtoP) tests for M1 vehicles. The technical service is strictly required to test any speed; therefore, disregard speeds mentioned in the context and generate the complete set of speeds defined in the regulation tables. The system shall be active at least within the vehicle speed range specified by the Maximum Impact Speed (km/h) for M1 vehicles.

Follow this exact format for every test scenario: 1- ”Write the scenario title (e.g., Test\_CtoP\_MassRun\_20) as the first line.” 2- ”Immediately after the title, write the full scenario description, using the exact wording of the provided example, adapting only the relevant parameters.” 3- ”After each complete test case, add exactly one separator line: — ,,

Formatting rules: - Use the provided example structure strictly. - Do not add explanations, notes, or extra text. - Only output the requested test scenarios and separators. - Do not include index numbers or any other identifiers.

Example: {example\_test\_case}

[Start generation now.]

## ACKNOWLEDGMENT

This research was funded by the Federal Ministry of Research, Technology and Space (BMFTR) as part of the CeCaS project, FKZ: 16ME0800K.

## REFERENCES

[1] N. Kandpal, H. Deng, A. Roberts, E. Wallace, and C. Raffel, “Large Language Models Struggle to Learn Long-Tail Knowledge,” https: //arxiv.org/abs/2211.08411, Jul. 2023, arXiv preprint arXiv:2211.08411.

[2] A. Mallen, A. Asai, V. Zhong, R. Das, D. Khashabi, and H. Hajishirzi, “When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories,” Jul. 2023.

[3] K. Lee, M.-W. Chang, and K. Toutanova, “Latent Retrieval for Weakly Supervised Open Domain Question Answering,” https://arxiv.org/abs/ 1906.00300, Jun. 2019, arXiv preprint arXiv:1906.00300.

[4] Anthropic, “Introducing 100K context windows,” https://www.anthropic. com/news/100k-context-windows, 2023, accessed: Sep. 28, 2025.

[5] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang, “Lost in the Middle: How Language Models Use Long Contexts,” Nov. 2023.

[6] V. Zolfaghari, N. Petrovic, F. Pan, K. Lebioda, and A. Knoll, “Adopting RAG for LLM-Aided Future Vehicle Design,” https://arxiv.org/abs/2411. 09590, Nov. 2024, arXiv preprint arXiv:2411.09590.

[7] F. Liu, Z. Kang, and X. Han, “Optimizing RAG techniques for automotive industry PDF chatbots: A case study with locally deployed Ollama models,” arXiv preprint arXiv:2408.05933, 2024. [Online]. Available: https://arxiv.org/abs/2408.05933

[8] F. Niu, R. Pan, L. C. Briand, H. Hu, and K. Koravadi, “TVR: Automotive System Requirement Traceability Validation and Recovery Through Retrieval-Augmented Generation,” http://arxiv.org/abs/2504. 15427, 2025, arXiv preprint arXiv:2504.15427.

[9] J. Sun, Z. Luo, and Y. Li, “A Compliance Checking Framework Based on Retrieval Augmented Generation,” in Proceedings of the 31st International Conference on Computational Linguistics. Abu Dhabi, UAE: Association for Computational Linguistics, Jan. 2025, pp. 2603– 2615. [Online]. Available: https://aclanthology.org/2025.coling-main. 178/

[10] R. Bolton, M. Sheikhfathollahi, S. Parkinson, V. Vulovic, G. Bamford, D. Basher, and H. Parkinson, “Document Retrieval Augmented Fine-Tuning (DRAFT) for safety-critical software assessments,” http://arxiv. org/abs/2505.01307, 2025, arXiv preprint arXiv:2505.01307.

[11] B. Kumar and D. Roussinov, “NLP-based Regulatory Compliance – Using GPT 4.0 to Decode Regulatory Documents,” http://arxiv.org/abs/ 2412.20602, 2024, arXiv preprint arXiv:2412.20602.

[12] S. Li, J. Chen, R. Yao, X. Hu, P. Zhou, W. Qiu, S. Zhang, C. Dong, Z. Li, Q. Xie, and Z. Yuan, “Compliance-to-Code: Enhancing Financial Compliance Checking via Code Generation,” http://arxiv.org/abs/2505. 19804, 2025, arXiv preprint arXiv:2505.19804.

[13] B. Agarwal, H. S. Jomraj, S. Kaplunov, J. Krolick, and V. Rojkova, “RAGulating Compliance: A Multi-Agent Knowledge Graph for Regulatory QA,” http://arxiv.org/abs/2508.09893, 2025, arXiv preprint arXiv:2508.09893.

[14] L. Xu, C. Feng, K. Zhang, L. Zhengyong, W. Xu, and F. Meng, “Equipping retrieval-augmented large language models with document structure awareness,” in Findings of the Association for Computational Linguistics: EMNLP 2025. Association for Computational Linguistics, 2025, pp. 24 608–24 631.

[15] X. Yu, P. Jian, and C. Chen, “TableRAG: A retrieval augmented generation framework for heterogeneous document reasoning,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2025, pp. 14 063–14 082.

[16] V. T. d. Oliveira, D. O. d. Silva, M. d. A. Souza, M. R. Lima, S. S. T. d. Oliveira, and T. C. Rosa, “Retrieval-augmented generation and knowledge graphs in Portuguese-language legal documents,” in Proceedings of the 17th International Conference on Computational Processing of Portuguese (PROPOR 2026), vol. 1, 2026, pp. 1–10.

[17] Publications Office of the European Union, “UN regulation no 152 — uniform provisions concerning the approval of motor vehicles with regard to the advanced emergency braking system (AEBS) for M1 and N1 vehicles,” https://op.europa.eu/en/publication-detail/ -/publication/fc2d3589-1a7c-11eb-b57e-01aa75ed71a1, Oct. 2020, accessed: Oct. 1, 2024.

[18] C. Arora, J. Grundy, L. Puli, and N. Layton, “Towards Standards-Compliant Assistive Technology Product Specifications via LLMs,” Apr. 2024.

[19] Y. Zhao, W. Xiao, T. Mihalj, J. Hu, and A. Eichberger, “Chat2Scenario: Scenario Extraction From Dataset Through Utilization of Large Language Model,” in 2024 IEEE Intelligent Vehicles Symposium (IV), Jun. 2024, pp. 559–566.

[20] Y. Deng, J. Yao, Z. Tu, X. Zheng, M. Zhang, and T. Zhang, “TAR-GET: Automated Scenario Generation from Traffic Rules for Testing Autonomous Vehicles,” Oct. 2023.

[21] S. Tang, Z. Zhang, J. Zhou, L. Lei, Y. Zhou, and Y. Xue, “LeGEND: A Top-Down Approach to Scenario Generation of Autonomous Driving Systems Assisted by Large Language Models,” Sep. 2024.

[22] H. Tian, X. Han, Y. Zhou, G. Wu, A. Guo, M. Cheng, S. Li, J. Wei, and T. Zhang, “LMM-enhanced Safety-Critical Scenario Generation for Autonomous Driving System Testing From Non-Accident Traffic Videos,” Jan. 2025.

[23] Q. Lu, X. Wang, Y. Jiang, G. Zhao, M. Ma, and S. Feng, “Multimodal Large Language Model Driven Scenario Testing for Autonomous Vehicles,” Sep. 2024.

[24] K. Lebioda, N. Petrovic, F. Pan, V. Zolfaghari, A. Schamschurko, and A. Knoll, “Are requirements really all you need? using LLMs to generate configuration code: A case study in automotive simulations,” IEEE Access, vol. 13, pp. 145 115–145 126, 2025. [Online]. Available: https://ieeexplore.ieee.org/abstract/document/11122468

[25] B. Wang, C. Xu, X. Zhao, L. Ouyang, F. Wu, Z. Zhao, R. Xu, K. Liu, Y. Qu, F. Shang, B. Zhang, L. Wei, Z. Sui, W. Li, B. Shi, Y. Qiao, D. Lin, and C. He, “MinerU: An Open-Source Solution for Precise Document Content Extraction,” http://arxiv.org/abs/2409.18839, 2024, arXiv preprint arXiv:2409.18839.

[26] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese BERT-networks,” in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, 2019, pp. 3982–3992. [Online]. Available: https://arxiv.org/abs/1908.10084

[27] N. Alturayeif, I. Ahmad, and J. Hassine, “TraceLLM: Leveraging large language models with prompt engineering for enhanced requirements traceability,” Requirements Engineering, 2026, arXiv:2602.01253.

[28] R. Etezadi, S. Abualhaija, C. Arora, and L. Briand, “Classifier or prompt: A case study on legal requirements traceability,” Empirical Software Engineering, 2026, arXiv:2502.04916.

[29] O. Folorunsho and H. Reza, “AI-driven test case generation from natural language requirements: A survey of techniques and research gaps,” https: //arxiv.org/abs/2606.06563, 2026.

[30] J. Hsia, A. Shaikh, Z. Wang, and G. Neubig, “RAGGED: Towards Informed Design of Scalable and Stable RAG Systems,” https://arxiv.org/abs/2403.09040, 2025, proceedings of the 42nd International Conference on Machine Learning (ICML), PMLR 267.

[31] B. J. Gutierrez, Y. Shu, W. Qi, S. Zhou, and Y. Su, “From RAG to Mem-´ ory: Non-Parametric Continual Learning for Large Language Models,” http://arxiv.org/abs/2502.14802, 2025, arXiv preprint arXiv:2502.14802.

[32] H. Markowitz, “Portfolio selection,” The Journal of Finance, vol. 7, no. 1, pp. 77–91, 1952.

[33] T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning: Data Mining, Inference, and Prediction, 2nd ed. Springer, 2009.

[34] “Regulation no 10 of the economic commission for europe of the united nations (UN/ECE) — uniform provisions concerning the approval of vehicles with regard to electromagnetic compatibility,” https://eur-lex. europa.eu/eli/reg/2012/10/oj/eng, United Nations Economic Commission for Europe, 2012, accessed: Sep. 29, 2025.

[35] “Regulation no 79 of the economic commission for europe of the united nations (UN/ECE) — uniform provisions concerning the approval of vehicles with regard to steering equipment,” https://eur-lex.europa.eu/ eli/reg/2008/79(2)/oj/eng, United Nations Economic Commission for Europe, 2008, accessed: Sep. 29, 2025.

[36] “Regulation no 130 of the economic commission for europe of the united nations (UN/ECE) — uniform provisions concerning the approval of motor vehicles with regard to the lane departure warning system (LDWS),” https://eur-lex.europa.eu/eli/reg/2014/130/oj/eng, United Nations Economic Commission for Europe, 2014, accessed: Sep. 29, 2025.

[37] “Regulation no 140 of the economic commission for europe of the united nations (UN/ECE) — uniform provisions concerning the approval of passenger cars with regard to electronic stability control (ESC) systems,” https://eur-lex.europa.eu/eli/reg/2018/1592/oj/eng, United Nations Economic Commission for Europe, 2018, accessed: Sep. 29, 2025.

[38] “UN regulation no 152 — uniform provisions concerning the approval of motor vehicles with regard to the advanced emergency braking system (AEBS) for M1 and N1 vehicles,” https://eur-lex.europa.eu/eli/reg/2020/ 1597/oj/eng, United Nations Economic Commission for Europe, 2020, accessed: Sep. 29, 2025.

[39] “UN regulation no 155 — uniform provisions concerning the approval of vehicles with regards to cybersecurity and cybersecurity management system,” https://eur-lex.europa.eu/eli/reg/2021/387/oj/eng, United Nations Economic Commission for Europe, 2021, accessed: Sep. 29, 2025.

[40] “UN regulation no 156 — uniform provisions concerning the approval of vehicles with regards to software update and software updates management system,” https://eur-lex.europa.eu/eli/reg/2021/388/oj/eng, United Nations Economic Commission for Europe, 2021, accessed: Sep. 29, 2025.

[41] “UN regulation no 13-h — uniform provisions concerning the approval of passenger cars with regard to braking,” https://eur-lex.europa.eu/eli/ reg/2023/401/oj/eng, United Nations Economic Commission for Europe, 2023, accessed: Sep. 29, 2025.

![](images/b304beed0f92924b4bd538fb3f6554e6129b9e484d0277f9e2df7e6ee157f351.jpg)

VAHID ZOLFAGHARI received the M.Sc. degree in computer science from the Amirkabir University of Technology. He is currently pursuing the Ph.D. degree with the Technical University of Munich. He is part of the Central Car Server (CeCaS) Project. Previously, he was a Senior Test Engineer in network security and later a Research Assistant in Italy, contributing to projects in IoT security, robotic systems, and neurorobotics. He has co-authored several publications on generative AI and automotive system design. His research focuses on the application of

large language models and retrieval-augmented generation systems for automotive software engineering.

![](images/486e2a611034aa83c5ab929d9d606ed0fdc8f439cd5093ebff1d6898e2a9a631.jpg)

Nenad Petrovic was born in Pirot, Serbia, in 1992. He received the bachelor’ degree in computer science and informatics from the Facultyof Electronic Engineering, University of Nis, Nis, Serbia, in 2015, the Laurea Magistrale degree in computer engineering from the Politecnico di Milano, Milan, Italy, and the Ph.D. degree in computing and electrical engineering from the Faculty of Electronic Engineering. He was a Teaching Assistant. He is currently a Postdoctoral Researcher and a Scientific Coordinator/Supervisor with the Chair of Robotics, Artificial

Intelligence and Real-Time Systems, Technical University of Munich (TUM), focused on automotive-related projects in area of GenAI-driven software development. He is the author or co-author of more than 200 scientific publications. His main areas of interests include model-driven software engineering, semantic technology, the Internet of Things (IoT), and generative artificial intelligence (GenAI).

![](images/a7158551eb62ee6cb59c35bdef435de50934e171d1fc2c6013bfe87769a487a6.jpg)

ANDRE SCHAMSCHURKO <sup>´</sup> received the Dipl.Inf. degree in computer science from TU Dresden, in 2023. He is currently pursuing the Ph.D. degree with the Chair of Robotics, Artificial Intelligence and Real-Time Systems, Technical University of Munich. The primary focus of his research is on natural language processing and generative AI models, with a particular interests in reducing their hallucinations.

![](images/8f1003f593c620dc844def5c72c6435c340f421a0477cab1a3f2637f3a705bbc.jpg)

ALOIS KNOLL (Fellow, IEEE) received the M.Sc. degree in electrical/communications engineering from the University of Stuttgart, Stuttgart, Germany, in 1985, and the Ph.D. degree (summa cum laude) in computer science from the Technical University of Berlin (TU Berlin), Berlin, Germany, in 1988. He was on the Faculty of the Computer Science Department, TU Berlin, until 1993. He joined the University of Bielefeld, Bielefeld, Germany, as a Full Professor, where he was the Director of the Technical Informatics Research Group, until 2001.

Since 2001, he has been a Professor with the Department of Informatics, Technical University of Munich (TUM), Munich, Germany. He was on the board of directors of the Central Institute of Medical Technology at TUM (IMETUM). From 2004 to 2006, he was the Executive Director of the Institute of Computer Science, TUM. From 2007 to 2009, he was a member of the EU’s highest advisory board on information technology, ISTAG, the Information Society Technology Advisory Group, and its subgroup on future and emerging technologies (FET). In this capacity, he was actively involved in developing the concept of EU’s FET flagship projects. His research interests include cognitive, medical, and sensor-based robotics; multi-agent systems; data fusion; adaptive systems; multimedia information retrieval; modeldriven development of embedded systems, with applications to automotive software and electric transportation; and simulation systems for robotics and traffic.