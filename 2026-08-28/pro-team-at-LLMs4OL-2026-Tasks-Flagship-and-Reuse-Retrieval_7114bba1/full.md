# pro-team at LLMs4OL 2026 Tasks Flagship and Reuse: Retrieval-Augmented Generation and Vocabulary-Constrained Filtering for Ontology Learning

Shivam Mishra<sup>1</sup> , Dhannu Ram Meena<sup>1</sup> , Muneendra Ojha<sup>1</sup> , Krishna Pratap Singh<sup>1</sup> , and Kuldeep Singh<sup>1</sup>

<sup>1</sup>AIMS Lab, Indian Institute of Information Technology Allahabad, Prayagraj, UP, India

\*Correspondence: Shivam Mishra, m.shivam0704@gmail.com

Abstract: Ontology learning from text remains challenging despite significant progress in Large Language Models (LLMs), which can hallucinate domain terms, produce inconsistent formats, and favor hierarchical over associative relations. In the LLMs4OL 2026 Challenge, we address both the End-to-End Flagship Task (Task A) and Ontology Extension Reuse Task (Task B) using an offline retrieval-augmented few-shot prompting pipeline. Our system employs Qwen2.5-14B-Instruct with all-MiniLM-L6-v2 for demonstration retrieval, selecting the top-5 examples for Task A and top-2 for Task B. A left-truncated context-windowing strategy preserves task instructions within long prompts. For Task B, generated triples undergo deterministic vocabulary-constrained filtering, retaining triples when at least one endpoint belongs to the sample’s closed term/type vocabulary and removing duplicates of the initial ontology. The approach achieves Semantic Graph Similarity of 0.8692, Term-Typing F1 of 0.9200, and Taxonomy Discovery F1 of 0.8540 on Task B, while Task A achieves 0.7416 Semantic Graph Similarity. However, no non-taxonomic relations are extracted, highlighting limitations of closed, taxonomy-oriented relation vocabularies.

Keywords: Ontology Learning, Large Language Models, Retrieval-Augmented Generation (RAG), Vocabulary-Constrained Filtering, Knowledge Representation.

## 1 Introduction

Formal ontologies provide machine-readable representations of domain knowledge by defining concepts, semantic relationships, and logical constraints that enable knowledge integration, interoperability, and automated reasoning across heterogeneous information systems [1]. They serve as the semantic backbone of numerous intelligent applications, including semantic search, recommendation systems, knowledge graphs, and digital libraries [2]. However, constructing and maintaining ontologies remains a labor-intensive process that requires substantial domain expertise and continuous refinement as domain knowledge evolves [3]. With the rapid growth of scientific literature and unstructured textual resources, manual ontology engineering has become increasingly difficult to scale, motivating the development of automated Ontology Learning (OL) techniques [2]. Ontology Learning aims to automatically or semi-automatically construct ontologies by extracting concepts, semantic relations, taxonomies, and other ontology components from textual data [3]. Traditional ontology learning methods relied on rule-based systems, statistical learning, and conventional natural language processing techniques for identifying ontology elements [3]. Although these methods achieved encouraging results in controlled environments, they often struggle with linguistic ambiguity, domain-specific terminology, and morphologically complex language structures, limiting their applicability across diverse domains [4].Recent advances in LLMs have significantly transformed ontology engineering by enabling zero-shot and few-shot reasoning over natural language without extensive task-specific feature engineering [1]. Modern instruction-tuned LLMs can identify concepts, semantic relations, ontology mappings, and logical structures directly from textual documents, making them promising tools for automating ontology development [2]. Nevertheless, recent surveys emphasize that effectively integrating LLMs into ontology engineering workflows remains an active research challenge because issues related to evaluation, reproducibility, and structured ontology construction have not yet been fully resolved [1].

To promote systematic evaluation of LLM-based ontology learning methods, the Large Language Models for Ontology Learning (LLMs4OL) paradigm introduced a standardized benchmark covering multiple ontology engineering tasks [5], [6]. The second edition of the challenge evaluated systems across four complementary tasks: Text-to-Ontology Extraction, Term Typing, Taxonomy Discovery, and Non-Taxonomic Relation Extraction [7]. The third edition, which this paper addresses, restructures the benchmark around three integrated tracks that combine these primitives: the Flagship Task (Task A), requiring end-to-end construction of a primitive ontology from raw text; the Reuse Task (Task B), requiring incremental extension of a partial ontology; and the Taxonomy Task (Task C), requiring cross-domain taxonomy induction [8]. Together, these tasks evaluate the complete ontology learning pipeline, from identifying ontology concepts in raw text to assigning semantic categories and discovering hierarchical and associative relationships among ontology entities. Among the ontology primitives these tasks target, accurate term extraction and semantic type assignment constitute the foundation of the ontology construction process, since they directly influence the quality of downstream taxonomy induction and relation extraction. Errors introduced during these early stages propagate to subsequent ontology learning tasks, reducing the overall consistency and usability of the generated ontology. Therefore, improving performance on the Flagship (Task A) and Reuse (Task B) tracks addressed in this paper remains essential for building reliable automated ontology learning systems.

Despite the remarkable reasoning capability of modern LLMs, several challenges continue to limit their effectiveness in ontology learning. Generative models frequently produce hallucinated or lexically inconsistent outputs that do not belong to the predefined ontology vocabulary, thereby reducing semantic consistency and increasing postprocessing complexity [7]. Existing studies also show that while LLMs perform well on hierarchical taxonomy prediction, they remain considerably less effective at identifying complex semantic and non-taxonomic relationships that require deeper contextual understanding [9]. Furthermore, retrieval-based prompting often requires long contextual prompts containing multiple demonstrations, increasing the likelihood of contextwindow truncation and loss of important task instructions during inference [7].

To overcome the identified challenges,we proposes an offline Retrieval-Augmented Generation (RAG) framework for ontology learning that combines dense semantic retrieval with instruction-guided LLM inference. The retrieval module dynamically selects semantically similar training examples to provide high-quality contextual guidance during inference while maintaining a completely offline pipeline suitable for resourceconstrained environments [7]. We further employ a left-truncated context management strategy to preserve task instructions when processing long prompts [7]. Finally, for tasks with closed-world assumptions, we introduce a deterministic vocabularyconstrained filtering step that strictly evaluates generated outputs against the predefined ontology vocabulary using exact lexical matching, dropping any invalid triples. Similar embedding-based semantic neighborhood approaches have demonstrated that constraining predictions to known semantic spaces significantly improves ontology classification performance [10].The primary contributions of this work are summarized as follows:

• We propose an offline Retrieval-Augmented Generation framework for ontology learning that combines dense semantic retrieval with efficient prompt construction.

• We introduce a vocabulary-constrained filtering algorithm for the Reuse Task that restricts generated outputs to the canonical ontology vocabulary, eliminating outof-vocabulary hallucinations by dropping invalid triples.

• We evaluate the proposed framework on the official LLMs4OL benchmark datasets for Task A (End-to-End Flagship) and Task B (Ontology Extension Reuse) using exact-match, fuzzy-match, and semantic similarity metrics.

• We provide a detailed error analysis to identify the remaining challenges in LLMbased ontology learning and discuss directions for future research.

The remainder of this paper is organized as follows. Section 2 reviews existing research on ontology learning and LLM-based ontology engineering. Section 3 presents the proposed methodology. Section 4 describes the experimental setup and reports the evaluation results. Section 5 discusses the findings and limitations of the proposed approach. Finally, Section 6 concludes the paper.

## 2 Related Work

Ontology Learning (OL) has evolved considerably over the past two decades, progressing from rule-based information extraction techniques to modern Large Language Model (LLM)-driven semantic reasoning. This section reviews the major developments in ontology learning, discusses recent advances in LLM-based ontology engineering, and positions the proposed Retrieval-Augmented Generation (RAG) framework within the existing literature.

## 2.1 Classical Ontology Learning

Traditional ontology learning focused on automatically extracting ontology elements such as concepts, taxonomies, semantic relations, and axioms from unstructured textual resources. Early approaches primarily relied on rule-based systems, statistical learning, and conventional natural language processing techniques, including tokenization, pos tagging, and NER, for ontology construction. These methods significantly reduced manual effort but depended heavily on handcrafted linguistic rules and domainspecific feature engineering, limiting their adaptability across domains [3].

The challenges become even more pronounced when processing morphologically rich languages. Existing studies have shown that the limited availability of mature NLP tools, together with complex grammatical structures, substantially reduces the effectiveness of traditional ontology extraction systems and often requires explicit linguistic knowledge for accurate semantic interpretation [4].

Although classical methods remain computationally efficient and interpretable, their limited contextual understanding prevents them from capturing implicit semantic relationships, motivating the adoption of deep learning and, more recently, large language models for ontology learning.

## 2.2 Large Language Models for Ontology Engineering

Recent advances in LLMs have substantially changed the ontology engineering landscape by enabling direct reasoning over natural language without manual feature engineering. Recent studies have shown that LLMs can effectively support ontology construction through zero-shot and few-shot reasoning while reducing the amount of manual annotation required [3].

A comprehensive review of recent research further demonstrates that LLMs are now being applied across multiple stages of the ontology engineering lifecycle, including ontology generation, alignment, evaluation, maintenance, and documentation. The review also highlights the absence of standardized evaluation protocols and reproducible experimental workflows, indicating that LLM-based ontology engineering remains an active research area [1].

Experimental evaluations on scholarly ontology generation have further shown that several instruction-tuned open-source and proprietary LLMs achieve strong zero-shot performance for semantic relation identification while requiring relatively modest computational resources. These findings demonstrate the increasing capability of modern LLMs to support automated ontology construction across specialized knowledge domains [2].

## 2.3 LLMs4OL Benchmark and Retrieval-Augmented Approaches

To facilitate systematic evaluation of ontology learning systems, the LLMs4OL challenge introduced standardized benchmark datasets covering multiple ontology engineering tasks across several application domains [11]. The second edition of the challenge evaluated systems on four complementary tasks, namely Text-to-Ontology Extraction, Term Typing, Taxonomy Discovery, and Non-Taxonomic Relation Extraction [12], and the third edition consolidates these primitives into the integrated Flagship, Reuse, and Taxonomy tracks addressed in this work, thereby encouraging the development of more robust and scalable end-to-end ontology learning systems.

Recent retrieval-augmented approaches combine dense semantic retrieval with fewshot prompting to improve ontology learning. Rather than relying solely on the internal knowledge of language models, these methods dynamically retrieve semantically similar training examples to guide inference, resulting in improved performance on term extraction and semantic typing tasks [7].

Another line of research combines semantic clustering with LLM inference by grouping semantically related concepts using transformer-based embeddings before prompt generation. This strategy reduces semantic ambiguity and improves prediction quality across multiple ontology learning tasks, highlighting the growing importance of retrieval and semantic representation learning in modern ontology learning pipelines [13].

## 2.4 Challenges in Ontology Learning

Despite the impressive capabilities of modern LLMs, several challenges continue to limit fully automated ontology construction. Recent investigations on ontology axiom identification indicate that LLMs perform considerably better on hierarchical subclass prediction than on more expressive ontology axioms such as domain, range, and disjointness constraints. These findings suggest that higher-level semantic reasoning remains substantially more difficult than taxonomy extraction [9].

Another important challenge is the lexical inconsistency introduced by generative models during ontology prediction. Recent studies have demonstrated that combining transformer-based semantic embeddings with classical k-Nearest Neighbors classification improves ontology term typing by constraining predictions within semantically meaningful neighborhoods. Such embedding-based neighborhood classification consistently improves prediction accuracy compared with relying solely on direct generative outputs [10].

## 2.5 Research Gap

The existing literature demonstrates significant progress in applying LLMs to ontology learning; however, several limitations remain. Classical ontology learning methods lack deep contextual understanding, whereas purely generative LLM-based systems often produce hallucinated or lexically inconsistent ontology terms [3]. Retrieval-augmented approaches improve contextual grounding but primarily focus on prompt engineering and semantic retrieval without explicitly enforcing ontology vocabulary consistency during generation [7]. Similarly, embedding-based classification methods improve semantic prediction but are generally designed for individual tasks rather than an integrated ontology learning pipeline [10].

Motivated by these limitations, our work combines dense Retrieval-Augmented Generation with a deterministic vocabulary-constrained filtering strategy that restricts model outputs to a predefined ontology vocabulary where applicable. By integrating retrievalbased contextual guidance with strict exact-match filtering, the proposed framework aims to improve both semantic accuracy and prediction consistency for the LLMs4OL Tasks A and B.

## 3 Methodology

This section presents the proposed Offline Retrieval-Augmented Ontology Learning Framework developed for the LLMs4OL 2026 Flagship (Task A) and Reuse (Task B) challenges. The framework combines dense semantic retrieval, instruction-guided

![](images/8ade0d09243ec8c23e4e70577f6550f07f0e9d4191a2cea182182957e2765850.jpg)

Figure 1. Architecture ofthe proposed Offline Retrieval-Augmented Ontology Learning Framework. The framework retrieves semantically similar examples using MiniLM embeddings, constructs retrieval-augmented prompts for Qwen2.5-14B-Instruct, and refines the generated ontology triples for Task B through vocabulary-constrained filtering before producing challengecompliant JSON outputs.

large language model (LLM) inference, and deterministic vocabulary-aware postprocessing to automatically construct or extend ontologies from unstructured text.

Instead of fine-tuning the language model, semantically similar training examples are dynamically retrieved from the training corpus and incorporated into the inference prompt. For the Reuse Task (Task B), the generated ontology triples are subsequently refined using a deterministic vocabulary-constrained filter before being converted into challenge-compliant JSON outputs. Figure 1 summarizes the overall workflow of the proposed framework.

## 3.1 Problem Formulation

Let

$$
D = \{ d _ { 1 } , d _ { 2 } , \ldots , d _ { n } \} ,\tag{1}
$$

where $D$ denotes a collection of input documents containing unstructured natural language text.

For the Flagship Task (Task $\pmb { \mathsf { A } } )$ , the objective is to construct a primitive ontology represented as

$$
O = \{ ( h , r , t ) \} ,\tag{2}
$$

where $h , r ,$ and t denote the head entity, ontology relation, and tail entity, respectively.

For the Reuse Task (Task B), an existing ontology $O _ { \mathrm { b a s e } }$ is provided together with new textual evidence. The objective is to infer only the ontology triples required to extend the existing ontology, rather than reconstructing it from scratch. Formally,

$$
O _ { \mathrm { f i n a l } } = O _ { \mathrm { b a s e } } \cup O _ { \mathrm { n e w } } ,\tag{3}
$$

where $O _ { \mathrm { n e w } }$ denotes the newly inferred ontology triples extracted from the input document. Consequently, the proposed framework supports both ontology construction and ontology extension within a unified inference pipeline.

## 3.2 Dense Semantic Retrieval

To provide contextual guidance during inference, semantically similar ontology learning examples are retrieved from the training corpus using dense semantic representations. Each training document is encoded with the sentence-transformers/all-MiniLM-L6-v2 model to obtain a fixed-length embedding. Let

$$
e _ { i } = f ( d _ { i } ) ,\tag{4}
$$

where $f ( \cdot )$ denotes the sentence embedding model and $e _ { i }$ represents the embedding of the $i ^ { \mathrm { { t h } } }$ training document. Similarly, the embedding of the input document $q$ is computed as

$$
e _ { q } = f ( q ) .\tag{5}
$$

The semantic similarity between the input document and each indexed example is computed using cosine similarity:

$$
\mathsf { s i m } ( q , d _ { i } ) = \frac { e _ { q } \cdot e _ { i } } { \left\| e _ { q } \right\| \left\| e _ { i } \right\| } .\tag{6}
$$

Training documents are ranked according to their similarity scores, and the most relevant examples are selected as in-context demonstrations for prompt construction, with the number of retrieved demonstrations tuned separately per task: top-5 for Task A and top-2 for Task B. Task A retrieves more demonstrations because the flagship setting requires the model to jointly reason across term extraction, typing, taxonomy, and relation extraction from raw text with no provided vocabulary, so broader in-context coverage was found to help; Task B provides an explicit candidate vocabulary (terms and types) and a partial ontology as additional grounding context, so fewer demonstrations were needed while also keeping prompts shorter and reducing truncation risk. Unlike static few-shot prompting, this retrieval strategy dynamically adapts the demonstrations to each input document, providing task-specific guidance without modifying the parameters of the underlying language model.

## 3.3 Prompt Construction and Ontology Generation

The retrieved examples are incorporated into a structured prompt for ontology generation using Qwen2.5-14B-Instruct. The system prompts explicitly guide the model’s behavior. For Task A, the prompt instructs the model: ”You are an expert ontology engineer. Extract ALL ontology triples from the passage. A triple is [subject, relation, object]... Copy each subject and object EXACTLY as it appears in the text.” It also maps natural language phrases directly to a closed set of relations (e.g., ”subclass of” maps to ”is-a”). For Task B, the prompt enforces strict hierarchy constraints: ”Predict only DIRECT parent relationships — connect a term to its MOST SPECIFIC correct type.”

Generation utilizes Qwen2.5-14B-Instruct in 16-bit precision with greedy decoding (sampling disabled) for deterministic outputs, a batch size of 10, and task-specific output budgets: 900 maximum new tokens for Task A (to accommodate the larger number of triples and prevent truncation on the 10 longest samples) and 320 maximum new tokens for Task B.

Although Qwen2.5-14B-Instruct supports a default context window of 32,768 tokens, we empirically restricted the maximum input length to 3,072 tokens. This threshold was selected to bound the Key-Value (KV) cache memory footprint and cap logits scaling spikes during batched inference on 80GB VRAM hardware. To ensure critical instructions were never lost, we employed a left-truncation strategy. If a dynamically constructed prompt exceeded 3,072 tokens, the oldest retrieved in-context demonstrations were dropped first, guaranteeing that the system instructions and the target query at the end of the prompt remained completely intact.

The generated response is parsed into ontology triples using a three-stage fallback parser: direct JSON parsing, regex extraction of the first well-formed array, and lineby-line regex recovery of quoted triples as a last resort. This improves robustness to minor formatting deviations (e.g. the model wrapping its JSON output in prose) without inferring or altering any content beyond what the model generated. Duplicate triples are removed, along with trivial reflexive self-loops.

## 3.4 Hybrid Extraction and Vocabulary-Constrained Filtering (Task B)

Task B provides each sample with an explicit closed vocabulary of candidate terms and types, together with a partial ontology. To respect this closed-world assumption and maximize performance, we implement a two-stage post-processing pipeline for Task B.First, to mitigate the inherent limitations of generative LLMs—particularly regarding omitted classifications—we supplement the LLM predictions with deterministic pattern-matching heuristics. We employ a proximity-based term-typing heuristic that scans the document for sentence-level co-occurrences of predefined terms and types, assigning an instance-of relation when the lexical distance between them falls within a high-confidence threshold. The triples extracted via this heuristic are merged with the generative LLM outputs to serve as a critical recall booster.Second, the merged triples are passed through a deterministic vocabulary-constrained filter. A candidate triple $[ h , r , t ]$ is retained only if at least one endpoint (h or t) is a member of the sample’s known vocabulary—the union of the provided terms, the provided types, and all entities already appearing in the sample’s initial ontology triples. Triples that duplicate a relationship already present in the initial ontology are discarded. This is a strict exact-match membership filter; a triple whose endpoints do not exactly match a known vocabulary entry is dropped outright rather than lexically repaired.No equivalent vocabulary constraint is applied to Task A, since the flagship setting provides no closed candidate vocabulary to constrain against.

## 3.5 Output Generation

After the generation and filtering stages, the validated ontology triples are converted into the official JSON format required by the LLMs4OL evaluation framework. For the Flagship Task (Task A), the framework outputs the generated primitive ontology triples for each input document. For the Reuse Task (Task B), only the newly inferred ontology triples are produced to extend the provided base ontology while preserving its original structure. The resulting JSON files strictly follow the submission schema specified by the challenge organizers, enabling direct evaluation without additional processing.

## 4 Experiments and Results

In this section, we present the empirical results of our proposed Retrieval-Augmented Generation (RAG) framework across the End-to-End Flagship (Task A) and Ontology Extension (Task B) challenges. We report performance using the official challenge evaluation script, which calculates task-oriented metrics alongside Graph Similarity (comprising Edge F1, Neighborhood Similarity, and Taxonomy Similarity). To evaluate the generated ontologies, the script computes three match variants: Exact Match enforces strict string equality, requiring the predicted head, relation, and tail lexical forms to perfectly mirror the gold standard. Fuzzy Match relaxes this lexical constraint by utilizing RapidFuzz string alignment, penalizing minor morphological variations or token mismatches based on a calculated Levenshtein distance ratio. Finally, Semantic Match relies on continuous embedding similarity, accepting predictions that are conceptually equivalent to the gold standard even if they differ entirely in surface-level string representation.

## 4.1 Results on Task B: Ontology Extension (Reuse)

The Ontology Extension task evaluates the system’s ability to incrementally expand a partial ontology given a constrained set of valid terms and types. Our implementation, driven by the vocabulary-constrained filtering algorithm, achieved exceptionally high structural compliance.

The results in Table 1 demonstrate the critical importance of deterministic postprocessing. Because the generative outputs were strictly filtered against the predefined canonical vocabulary, the delta between the Exact Match Graph Similarity (0.8480) and

Table 1. Evaluation Metrics for Task B (Reuse Task): Graph Similarity
<table><tr><td rowspan=1 colspan=1>Metric Category</td><td rowspan=1 colspan=1>Edge F1</td><td rowspan=1 colspan=1>NeighborhoodSimilarity</td><td rowspan=1 colspan=1>TaxonomySimilarity</td><td rowspan=1 colspan=1>GraphSimilarity</td></tr><tr><td rowspan=1 colspan=1>Exact Match</td><td rowspan=1 colspan=1>0.8776</td><td rowspan=1 colspan=1>0.8500</td><td rowspan=1 colspan=1>0.8165</td><td rowspan=1 colspan=1>0.8480</td></tr><tr><td rowspan=1 colspan=1>Fuzzy Match</td><td rowspan=1 colspan=1>0.8945</td><td rowspan=1 colspan=1>0.8690</td><td rowspan=1 colspan=1>0.8332</td><td rowspan=1 colspan=1>0.8656</td></tr><tr><td rowspan=1 colspan=1>Semantic Match</td><td rowspan=1 colspan=1>0.8988</td><td rowspan=1 colspan=1>0.8727</td><td rowspan=1 colspan=1>0.8359</td><td rowspan=1 colspan=1>0.8692</td></tr></table>

Table 2. Task-Oriented Metrics for Task B (Reuse Task)
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>Recall</td><td rowspan=1 colspan=1>F1-Score</td></tr><tr><td rowspan=1 colspan=1>Term-Typing</td><td rowspan=1 colspan=1>0.9324</td><td rowspan=1 colspan=1>0.9079</td><td rowspan=1 colspan=1>0.9200</td></tr><tr><td rowspan=1 colspan=1>Taxonomy-Discovery</td><td rowspan=1 colspan=1>0.9369</td><td rowspan=1 colspan=1>0.7846</td><td rowspan=1 colspan=1>0.8540</td></tr><tr><td rowspan=1 colspan=1>Non-Taxonomic-RE</td><td rowspan=1 colspan=1>0.0000</td><td rowspan=1 colspan=1>0.0000</td><td rowspan=1 colspan=1>0.0000</td></tr></table>

Semantic Match Graph Similarity (0.8692) is remarkably narrow. This confirms that the model successfully avoided out-of-vocabulary hallucinations. The framework excelled particularly in Term-Typing, achieving a robust 0.9200 F1-score, suggesting that context-augmented prompting coupled with strict vocabulary-constrained filtering contributes positively to closed-world classification. We note, however, that the absence of strict ablation studies limits our ability to precisely quantify the isolated performance gains provided by the RAG module versus the rule-based extraction and filtering components.

## 4.2 Results on Task A: End-to-End Flagship Task

The Flagship task is fundamentally more complex, requiring the model to extract and structure the primitive ontology without predefined vocabulary constraints.

Table 3. Evaluation Metrics for Task A (Flagship Task): Graph Similarity
<table><tr><td>Metric Category</td><td>Edge F1</td><td>Neighborhood Similarity</td><td>Taxonomy Similarity</td><td>Graph Similarity</td></tr><tr><td>Exact Match</td><td>0.5891</td><td>0.5152</td><td>0.5017</td><td>0.5353</td></tr><tr><td>Fuzzy Match</td><td>0.7365</td><td>0.6765</td><td>0.6538</td><td>0.6889</td></tr><tr><td>Semantic Match</td><td>0.7854</td><td>0.7319</td><td>0.7076</td><td>0.7416</td></tr></table>

Table 4. Task-Oriented Metrics for Task A (Flagship Task)
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>Recall</td><td rowspan=1 colspan=1>F1-Score</td></tr><tr><td rowspan=1 colspan=1>Term-Typing</td><td rowspan=1 colspan=1>0.5952</td><td rowspan=1 colspan=1>0.4245</td><td rowspan=1 colspan=1>0.4956</td></tr><tr><td rowspan=1 colspan=1>Taxonomy-Discovery</td><td rowspan=1 colspan=1>0.6161</td><td rowspan=1 colspan=1>0.5708</td><td rowspan=1 colspan=1>0.5926</td></tr><tr><td rowspan=1 colspan=1>Non-Taxonomic-RE</td><td rowspan=1 colspan=1>0.0000</td><td rowspan=1 colspan=1>0.0000</td><td rowspan=1 colspan=1>0.0000</td></tr></table>

In the unconstrained setting of Task A, we observe a significant divergence between exact and semantic metrics. The Exact Match Graph Similarity drops to 0.5353, while the Semantic Match Graph Similarity reaches a much stronger 0.7416. This 20-point difference highlights a known limitation of unconstrained generative models: although they successfully capture the underlying semantic concepts, as reflected by the high semantic similarity score, they often fail to reproduce the exact lexical forms required by strict exact-match evaluation protocols.

## 4.3 In-Depth Analysis: The Non-Taxonomic Extraction Bottleneck

A review of the evaluation logs across both Task A and Task B shows that the F1-score for Non-Taxonomic Relation Extraction is exactly 0.0000 in both tracks. Two factors contribute to this result. First, our relation-mapping instructions defined a closed set of six strictly taxonomic and typing relation strings (is-a, instance-of, disjoint with, equivalent class, part of, has part). This design choice was empirical: early exploration of the training data revealed that hierarchical relations overwhelmingly dominated the dataset. To prevent the LLM from hallucinating diverse, unstandardized predicates, we restricted its generation space. By forcing the LLM to map all textual relationships to this strict set (e.g., standardizing ”type of” and ”category of” to ”is-a”), we implicitly handled relational redundancies directly during the generation phase, which is why we did not employ a post-processing predicate filter to catch impossible co-occurring relations. However, under this strict prompt design, the model had no mechanism by which to emit a non-taxonomic relation label. Second, the extent to which this constrains performance depends on how prevalent genuinely non-taxonomic relations are in the gold data itself for these two tasks. Without a full accounting of the gold relation-type distribution, we cannot fully separate our prompt-design limitation from a property of the benchmark data. We report the 0.0 score as an observed result and flag the closed relation vocabulary as the most direct, verifiable explanation available from our pipeline.

## 5 Conclusion

In this work, we presented an offline Retrieval-Augmented Generation (RAG) framework utilizing Qwen2.5-14B-Instruct and sentence-transformers/all-MiniLM-L6-v2 for the LLMs4OL 2026 Challenge. By combining left-side context window management with deterministic vocabulary-constrained filtering for Task B, our approach effectively prevented out-of-vocabulary hallucinations from entering the final generated outputs and stabilized overall predictions. Our system achieved strong results on Task B (Ontology Extension), securing an Overall Semantic Graph Similarity of 0.8692, a Term-Typing F1-score of 0.9200, and a Taxonomy Discovery F1-score of 0.8540, while also demonstrating the effectiveness of retrieval-augmented prompting in unconstrained scenarios by achieving an Overall Semantic Graph Similarity of 0.7416 on Task A (End-to-End Flagship). However, error analysis revealed a major bottleneck: a 0.0000 F1-score in Non-Taxonomic Relation Extraction across both tasks, highlighting both a limitation of our current prompt design—which defines only a closed taxonomic/typing relation vocabulary—and an open question about how prevalent non-taxonomic relations are in the benchmark data itself. Moving forward, future work will focus on hybrid neuro-symbolic integration, schema-guided constrained decoding, and ontology-aware relation extraction techniques to improve non-taxonomic relation prediction. Overall, the proposed framework demonstrates that combining retrieval augmentation with deterministic vocabulary alignment provides a practical and effective approach for ontology learning using large language models, while also identifying key challenges that motivate future research in hybrid ontology construction methods.

## Limitations

While the proposed framework achieved strong results, several limitations remain. First, the individual performance contributions of dynamic retrieval, prompt configuration, and hybrid rule-based extraction were not isolated through formal ablation studies, making it difficult to definitively quantify each component’s distinct impact. Second, our post-processing focused entirely on entity vocabulary filtering rather than explicit semantic predicate filtering. Finally, critical hyperparameters—such as the 3,072-token truncation limit and the varying in-context demonstration counts (top-5 vs. top-2)—were set empirically based on hardware constraints and preliminary dev-set evaluations, rather than through exhaustive grid search optimization.

## Data availability statement

The dataset used in this study was provided by the task organizers as part of the LLMs4OL 2026 challenge.

## Underlying and related material

The datasets analyzed during the current study are available in the official LLMs4OL 2026 Challenge repository. The code used to generate the submitted results is available at: https://github.com/shivam07041/team-pro-integration\_OntoLearner\_Submission.

## Funding

This research was supported by the Startup Research Grant provided by the Department of Information Technology, Indian Institute of Information Technology Allahabad.

## Author contributions

Shivam Mishra: Conceptualization, Methodology, Software, Writing. Dhannu Ram Meena: Software, Investigation. Muneendra Ojha: Supervision, Writing – review & editing. Krishna Pratap Singh: Supervision, Writing – review & editing. Kuldeep Singh: Resources, Investigation.

## Competing interests

The authors declare no competing interests.

## Acknowledgment

The authors sincerely thank Dr. Sumit Singh and Arindam Ghosh for assistance during this research work.

## Declaration on Generative AI

Generative AI tools were used to assist with manuscript drafting and language refinement. The authors reviewed and validated all AI-assisted content and take full responsibility for the accuracy, originality, and integrity of the final manuscript.

## References

[1] J. Li, D. Garijo, and M. Poveda-Villalon, “Large language models for ontology engineering: ´ A systematic literature review,” Semantic Web, vol. 17, no. 4, p. 22 104 968 261 465 514, 2026.

[2] T. Aggarwal, A. Salatino, F. Osborne, and E. Motta, “Large language models for scholarly ontology generation: An extensive analysis in the engineering field,” Information Processing & Management, vol. 63, no. 1, p. 104 262, 2026.

[3] O. Perera and J. Liu, “Exploring large language models for ontology learning,” 2024.

[4] H. Terdalkar and A. Bhattacharya, “Framework for question-answering in sanskrit through automated construction of knowledge graphs,” in Proceedings of the 6th International Sanskrit Computational Linguistics Symposium, 2019, pp. 97–116.

[5] H. Babaei Giglou, J. D’Souza, and S. Auer, “Llms4ol: Large language models for ontology learning,” in The Semantic Web – ISWC 2023, T. R. Payne et al., Eds., Cham: Springer Nature Switzerland, 2023, pp. 408–427, ISBN: 978-3-031-47240-4.

[6] H. B. Giglou, J. D’Souza, and S. Auer, “Llms4ol 2024 overview: The 1st large language models for ontology learning challenge,” Open Conference Proceedings, vol. 4, pp. 3– 16, Oct. 2024. DOI: 10.52825/ocp.v4i.2473 [Online]. Available: https://www.tibop.org/ojs/index.php/ocp/article/view/2473

[7] H. B. Giglou, J. D’Souza, N. Mihindukulasooriya, and S. Auer, “Llms4ol 2025 overview: The 2nd large language models for ontology learning challenge,” Open Conference Proceedings, vol. 6, Oct. 2025. DOI: 10.52825/ocp.v6i.2913 [Online]. Available: https: //www.tib-op.org/ojs/index.php/ocp/article/view/2913

[8] H. B. Giglou, J. D’Souza, and S. Auer, “Llms4ol 2026 overview: The 3rd large language models for ontology learning challenge,” Open Conference Proceedings, 2026.

[9] R. M. Bakker, D. L. Di Scala, M. H. de Boer, and S. A. Raaijmakers, “Ontology learning with llms: A benchmark study on axiom identification,” arXiv preprint arXiv:2512.05594, 2025.

[10] C. Yimmark and T. Racharak, “T-grec at llms4ol 2025 task b: A report on term-typing task of obi dataset using llm with k-nearest neighbors,” in Open Conference Proceedings, vol. 6, 2025.

[11] H. B. Giglou, J. D’Souza, S. Sadruddin, and S. Auer, “Llms4ol 2024 datasets: Toward ontology learning with large language models,” in Open Conference Proceedings, vol. 4, 2024, pp. 17–30.

[12] A. Beliaeva and T. Rahmatullaev, “Heterogeneous llm methods for ontology learning (few-shot prompting, ensemble typing, and attention-based taxonomies),” arXiv preprint arXiv:2508.19428, 2025.

[13] P. K. Goyal, S. Singh, and U. S. Tiwary, “Silp nlp at llms4ol 2025 tasks a, b, c, and d: Clustering-based ontology learning using llms,” in Open Conference Proceedings, vol. 6, 2025.