# The zbMATH Open Knowledge Graph: Tracing Centuries of Mathematical Research

Yuni Susanti and Moritz Schubotz

FIZ Karlsruhe, Berlin, Germany {yuni.susanti, moritz.schubotz}@fiz-karlsruhe.de

Abstract. We present zbMATH Open Knowledge Graph, a large-scale RDF knowledge graph (KG) covering more than 250 years of mathematical scholarship. Unlike existing scholarly knowledge graphs that primarily capture bibliographic metadata and citation structures, the zbMATH Open KG integrates expert-curated semantic content, including reviews, keywords, subject classifications, software references, and disambiguated authorship. This combination of domain-specific representation of mathematical knowledge and extensive temporal coverage supports analyses that require fine-grained exploration of mathematical concepts, research fields, and scholarly relationships over time. The resulting graph comprises 34 million entities and 168 million RDF triples represented using established Semantic Web vocabularies, supporting interoperability and FAIR data principles. We further demonstrate its capabilities through query-driven historically grounded scholarly exploration use cases, illustrating how the knowledge graph can surface relationships and patterns that may be dificult to identify from bibliographic and citation information alone. The zbMATH Open KG provides an open semantic infrastructure for studying the development of mathematical knowledge and tracing scholarly connections across centuries of scholarship.

Resource type: Dataset/Knowledge Graph   
License: CC-BY-SA 4.0   
DOI: https://doi.org/10.5281/zenodo.21497975   
URL: https://github.com/zbMATHOpen/zbmath-open-kg

Keywords: knowledge graphs · scholarly data · mathematics

## 1 Introduction

Mathematics is one of the oldest continuously evolving scientific disciplines, with a documented scholarly record spanning over many centuries. Beyond formal results, its literature captures the long-term conceptual, methodological, and intellectual development of mathematical knowledge. Despite the availability of millions of digitized publications, the semantic structure and historical development of mathematical knowledge remain dificult to analyze computationally. Existing platforms primarily support keyword-based search and citation-based discovery, ofering limited support for investigating how mathematical concepts emerge, how ideas migrate across subfields, and how intellectual lineages are formed across generations. [9,10,13].

In this work, we present zbMATH Open Knowledge Graph (zbMATH Open KG), a large-scale and historically comprehensive RDF-based knowledge graph of mathematical publications and scholarly knowledge. Constructed from the zb-MATH Open platform [34], the longest-running abstracting and reviewing service in mathematics, the resource builds upon more than 150 years of curated mathematical scholarship, covering over four million publications spanning 250 years [34]. Unlike existing scholarly knowledge graphs that primarily represent bibliographic metadata and citation networks [28,17,1,2,4], zbMATH Open KG integrates expert-curated semantic knowledge, including:

– expert-written reviews providing contextual interpretations of publications,

– disambiguated identities of scholars, including authors and reviewers,

– expert-curated controlled mathematical keywords,

– expert-assigned Mathematics Subject Classification (MSC) [3] codes, a domainspecific taxonomy that has provided a stable conceptual organization of mathematics across decades,

– software references associated with mathematical publications.

To support interoperability within Semantic Web ecosystem, we construct the knowledge graph using established vocabularies and ontologies. The resulting graph comprises more than 34 million entities and 168 million RDF triples and is designed to align with FAIR [33] principles. Beyond data publication, the zbMATH Open KG supports distinct use cases of scholarly knowledge discovery we collectively term historically grounded scholarly exploration: the analysis of long-term intellectual relationships and scholarly development beyond citation networks and bibliographic metadata. By combining expert-curated semantics with centuries of scholarly records, the zbMATH Open KG establishes a foundation for studying the development of mathematical knowledge and tracing intellectual relationships across centuries of scholarly development.

## Contribution. Our main contributions can be summarized as follows:

1. We construct zbMATH Open KG, a large-scale RDF knowledge graph spanning over 250 years of mathematical scholarship grounded in expert-curated semantic contents. The resource is publicly released and made accessible<sup>1</sup> through the following:

(a) We provide the full RDF data dumps via Zenodo for long-term accessibility and preservation. The dump is updated periodically, approximately twice per year, subject to the availability of upstream data sources.

(b) We deploy the knowledge graph in a triple store and provide a public SPARQL endpoint for direct querying.

(c) We enable URI resolution for the core entities (e.g., publications, scholars, software references), supporting linked data access.

<sup>1</sup> See Page 1 for the DOI and URL; SPARQL endpoint is available on GitHub.

(d) We release the full source code for the knowledge graph construction to support reproducibility, customization and future extensions.

2. We design the resource according to FAIR principles and Semantic Web standards. The OWL ontology is published with VoID, DCAT and PROV-O metadata description to support machine-readable discovery and reuse.

3. We evaluate the resource at three complementary levels: serialization validity, structural consistency, and competency-question coverage. Beyond these validations, we demonstrate the practical utility of the resource through query-driven historically grounded scholarly exploration use cases.

## 2 Related Work

Knowledge graphs have become an important paradigm for representing and interlinking scholarly information. Large-scale initiatives such as the Microsoft Academic Graph (MAG) [31], its Linked Data transformation Microsoft Academic Knowledge Graph (MAKG) [16], and its successor OpenAlex [28] provide extensive coverage of scholarly metadata and citation networks. Semantic representations such as SemOpenAlex [17] further improve interoperability with Semantic Web technologies, while domain-specific resources such as LPWC [19] focuses on machine learning publications, and SemRepo [29] extend the LPWC scholarly knowledge graphs by linking publications with software and research artifacts. Similarly, established scholarly databases including arXiv [1], DBLP [2], and Semantic Scholar [4] provide broad bibliographic coverage. However, these resources primarily represent publication-level metadata and citation relationships, with limited expert-curated, high-quality semantic knowledge and longterm conceptual development within a scientific domain.

These projects cover a wide spectrum of scholarly disciplines; however their representation and coverage of mathematical publication and knowledge remains limited. Several initiatives have been undertaken to capture the structure and semantics of mathematical knowledge. For instance, the OntoMath ontology [14] models mathematical concepts and relations for semantic search, while projects such as Formal Abstracts [8], Lean [5], and Coq [20] focus on formalizing mathematical statements and machine-verifiable knowledge. MMLKG provides a thesaurus for describing mathematical definitions, statements, and proofs based on the Mizar Mathematical Library [32]. These eforts provide deep representations of mathematical content but primarily target formal reasoning. More recently, AutoMathKG [6] integrates paper with theorems and proofs; however, its construction relies heavily on semi-automatic extraction and LLM-assisted processing and focused to enhance LLM mathematical reasoning. MaRDI knowledge graph [30], based on Wikidata data modeling, integrates mathematical research data from multiple sources, ofering broad coverage of mathematical research artifacts. However, their structure is tightly coupled to Wikidata’s internal model, relying on Wikibase identifiers and property models rather than standard Semantic Web vocabularies [30].

The zbMATH Open KG complements these eforts by providing an RDF-native representation of mathematically curated scholarly knowledge. Unlike Wikidatabased approaches such as MaRDI, the zbMATH Open KG is built on established Semantic Web vocabularies and ontologies and designed to align to its standards and FAIR principles. Additionally, the explicit modeling of expert-curated semantic contents enables analyses requiring fine-grained understanding of mathematical concepts, fields, and intellectual relationships.

## 3 The zbMATH Open Knowledge Graph

In this section, we describe the construction process of the zbMATH Open KG (§3.1), followed by ontology and schema design (§3.2) and key statistics (§3.3). Finally, we evaluate the graph to assess its validity and coverage (§3.4).

## 3.1 Knowledge Graph Construction

The development of the zbMATH Open KG follows a knowledge engineering methodology largely guided by METHONTOLOGY [23], complemented by ontology engineering practices from SAMOD [25] and LOT [27]. The construction process consists of the following stages:

1. Requirements Specification. We first defined the scope of the knowledge graph by identifying the core entities, relationships, and intended usage scenarios. Competency Questions (CQs) were formulated and checked by human experts to capture representative scholarly information needs, including historical semantic exploration. We share the full CQs in our repository.

2. Knowledge Acquisition. The raw dataset was acquired from the zbMATH Open platform via its oficial API<sup>2</sup>. The acquired metadata served as the primary source for ontology development and instance generation.

3. Conceptualization. Based on the acquired data, we iteratively developed the conceptual model and structural requirement of the knowledge graph. Core classes such as publications, authors, and reviews were identified together with their semantic relationships, including the required properties and datatypes. The conceptual model was repeatedly refined as additional metadata characteristics and competency questions were analyzed.

4. Vocabulary Integration. We maximized interoperability through the reuse of established Semantic Web vocabularies. For instance, scholarly entities are primarily modeled using schema.org [21], citation relations using cito [26], and concept organization using skos [24]. Domain-specific concepts (e.g., mathematical subject classification and keywords) are introduced within the zbMATH namespace. See the detail of ontology and schema design in 3.2.

5. Implementation. The pipeline automatically converts the raw data into RDF triples. To facilitate customization (e.g., custom KG subsets generation), we share the full source code in our repository.

6. Evaluation and Maintenance. The generated RDF knowledge graph is evaluated at three levels to assess its validity and coverage: serialization validity, structural consistency, and competency-question coverage. See the detailed evaluation in 3.4. Maintenance is planned through periodic updates with versioned release, supported by the pipeline that enables regeneration from the zbMATH Open snapshots as new releases become available.

## 3.2 Ontology and Schema Design

Figure 1 illustrates the schema design of the zbMATH Open KG, including an overview of the entity types, the object properties, and the data type properties. To support interoperability with the Linked Data ecosystem, the model follows the best practice of semantic reuse strategy by adopting established Semantic Web vocabularies wherever possible and extending them with zbMATH Open - specific classes and persistent identifiers for mathematical resources.

![](images/1d3dc8a83693cce7f755c9c7e6c053552105caab44c6c731d24c90a362d48ed5.jpg)  
Fig. 1: Schema overview of the zbMATH Open Knowledge Graph.<sup>3</sup>

Scholarly Entity Representation. The core of the model represents mathematical scholarly publications and their contextual relationships. Publication entities are modeled using schema.org terms (schema:) [21]. This representation captures not only the main bibliographic records (schema:ScholarlyArticle, schema:Person for representing scholars i.e., authors and reviewers) but also the broader scholarly ecosystem surrounding a scholarly article, including reviews, periodicals, publishers, and associated mathematical software.

Bibliographic and Citation Modeling. Bibliographic metadata including article titles, publication dates, languages, identifiers (e.g., DOI), url, publishers, and document types are additionally represented using Dublin Core Terms (dcterms:) [11], following established practices in digital library data modeling. Citation relationships are represented using the Citation Typing Ontology (cito:) [26] for explicit modeling of publication-level citation networks.

Mathematical Semantic Representation. One of the key features of the zbMATH Open KG is the explicit representation of mathematical semantics. The organization of the Mathematics Subject Classification (MSC) codes [3] and expertcurated controlled keywords are modelled using SKOS (skos:) [24], while retaining persistent zbMATH Open -specific identifiers. The hierarchical structure of the MSC taxonomy is modeled using skos:broader and skos:narrower, preserving the long-established disciplinary organization of mathematics.

Interoperability and Extensibility. Additionally, RDF Schema (rdfs:) is used to define classes, hierarchies, and labels; XML Schema datatypes (xsd:) provide standardized datatypes for literal values. The ontology defines zbMATH Openspecific classes and persistent IRIs for mathematical resources while maintaining interoperability with the Linked Data ecosystem via established Semantic Web vocabularies. The model is intentionally lightweight and extensible, allowing future integration of additional mathematical resources, such as datasets, formalized results, or mathematical objects (e.g., formulae). The complete ontology specification is publicly available alongside the released data and machinereadable metadata descriptions.

## 3.3 Key Statistics

Table 1 illustrates the scale, coverage, and structural properties of the constructed zbMATH Open KG. With over 34.5 million entities and nearly 170 million relationships spanning 39 distinct predicate types, it represents one of the most comprehensive structured resources in mathematics. On average, each entity is connected to nearly ten others, reflecting a well-linked but non-redundant structure. The graph density of $1 . 4 5 \times 1 0 ^ { - 7 }$ reflects the expected sparsity typical of heterogeneous scholarly RDF graphs, where the number of observed relationships remains small compared to all possible pairwise connections.

Importantly, 15.4 million RDF entity resources (over 80%) are explicitly typed, providing semantic annotations that improve interpretability and support more precise queries over bibliographic records. Although zbMATH, the original platform, was established in 1931 [34], the knowledge graph captures a temporal span of more than 250 years through the inclusion of publications dating back to 1763. This extensive temporal coverage supports analyses of scholarly production and the development of entities and relationships over time.

Table 1: Quantitative metrics of the zbMATH Open Knowledge Graph
<table><tr><td>Metric</td><td>Description</td><td>Value</td></tr><tr><td>Number of Entities</td><td>All distinct nodes (subjects/objects in triples) 34,099,406</td><td></td></tr><tr><td>Number of RDF Resources</td><td>All distinct nodes (subjects/objects in triples) 18,995,799</td><td></td></tr><tr><td></td><td>Number of Typed Resources Resources with : type declaration</td><td>15,392,393</td></tr><tr><td>Type Coverage (%)</td><td>% of resources with a defined type</td><td>81.03%</td></tr><tr><td>Number of Edges</td><td>Total number of triples</td><td>168,670,821</td></tr><tr><td>Number of Relation Types</td><td>Unique predicate types used in edges</td><td>39</td></tr><tr><td>Average Degree</td><td>Mean number of edges per entity</td><td>9.89</td></tr><tr><td>Graph Density</td><td>Ratio of actual to possible edges</td><td> $1 . 4 5 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Temporal Coverage</td><td>Time span covered by temporal data</td><td>1763-2026</td></tr></table>

Table 2 summarizes the distribution of core entities within zbMATH Open KG. The entity distribution reflects the scholarly focus and expert-curated nature of the KG. At its center are more than 4 million articles, which form the backbone of the graph. These publications are connected to a curated network of over 1.1 million disambiguated person entities [12], including more than 1.12 million authors and nearly 14 thousand reviewers, providing a curated representation of scholarly contributors and supporting reliable attribution of research contributions.

The scholarly articles are complemented by 3.10 million expert-written reviews, which provide contextual interpretations and assessments of mathematical publications beyond bibliographic metadata. Furthermore, the presence of 3.01 million controlled keywords and 6,734 mathematics subject classification entities demonstrates the integration of domain-specific semantic annotations that support fine-grained organization and retrieval of mathematical knowledge. The ecosystem is further represented through thousands of organizations i.e., publishers (3,524) and periodicals (e.g., journals) (7,350), as well as over 30,000 references to mathematical software linked to publication, capturing important contextual elements of mathematical research. The resource contains more than 10.5 million identifiers, including nearly 2.5 million DOI references, which facilitate interoperability with external scholarly resources and persistent identification of research outputs.

Table 2: Core entities in zbMATH Open Knowledge Graph
<table><tr><td colspan="2">Entity # Instances</td></tr><tr><td>Scholarly articles</td><td>4,070,393</td></tr><tr><td>Person</td><td>1,125,474</td></tr><tr><td>—Authors</td><td>1,125,238</td></tr><tr><td>—Reviewers</td><td>13,942</td></tr><tr><td>Subjects (MSC)</td><td>6,734</td></tr><tr><td>Keywords</td><td>3,008,881</td></tr><tr><td>Organization</td><td>3,524</td></tr><tr><td>Periodical</td><td>7,350</td></tr><tr><td>Reviews</td><td>3,098,767</td></tr><tr><td>Software</td><td>30,857</td></tr><tr><td>Identifiers</td><td>10,522,258</td></tr><tr><td>-DOI</td><td>2,446,976</td></tr></table>

Table 3: Summary of the structural consistency check results.
<table><tr><td>Check</td><td>Validation criterion</td><td>Violations</td></tr><tr><td></td><td>Required properties Scholarly articles contain required properties:</td><td>61,380</td></tr><tr><td></td><td>Title (dcterms:title)</td><td>3</td></tr><tr><td></td><td>Author (dcterms:creator, schema:author)</td><td>61,377</td></tr><tr><td></td><td>Publication date (schema:datePublished)</td><td>0</td></tr><tr><td></td><td>Document type (dcterms:type)</td><td>0</td></tr><tr><td></td><td>Datatype consistency Date values use the expected RDF datatypes</td><td>0</td></tr><tr><td>Author references</td><td>Article authors are represented as schema:Person</td><td>0</td></tr><tr><td>Document type</td><td>Article types belong to the defined document-type</td><td>0</td></tr><tr><td>MSC references</td><td>vocabulary</td><td>0</td></tr><tr><td>Keyword references</td><td>Article subjects resolve to defined MSC concepts Article keywords resolve to defined keyword con-</td><td>0</td></tr><tr><td>Review references</td><td>cepts Reviewed articles and reviewers resolve to resources</td><td>0</td></tr><tr><td>SKOS hierarchy</td><td>of the expected classes skos:broader and skos:narrower relations are</td><td>0</td></tr></table>

## 3.4 Evaluation and Validation

The generated knowledge graph was validated at three complementary levels<sup>4</sup>: (i) syntactic validity of the RDF serialization, (ii) structural consistency, and (iii) CQ-based evaluation, described in detail in the following.

First, the syntactic validity of the generated RDF serialization was verified using both Apache Jena and RDFLib Pyton library [7], and was subsequently loaded into the OpenLink Virtuoso and Apache Jena Fuseki triple stores [15] to verify successful ingestion and to support the SPARQL-based validation. All RDF files were successfully parsed and loaded without syntax or ingestion errors, confirming that the generated resource was syntactically valid.

Second, the consistency check was performed to assess compliance with the structural requirements defined during the knowledge graph construction. These checks targeted various aspects of structural consistency, such as missing required properties, invalid or unexpected datatypes, and inconsistent class assignments, among others. For each requirement, a SPARQL query was executed over the complete graph to retrieve resources violating the corresponding constraint; the number of returned resources was used as the violation count. Representative results from the checks performed are summarized in Table 3.

The structural check identified 61,380 missing-property violations (3 missing titles, 61,377 missing authors). Manual inspection confirmed that these violations originate from the incomplete records in the source API, where the title or author information is not provided (“None”). The issue is reported to zbMATH platform and is expected to be addressed through a future correction of the API. All other structural consistency checks resulted in zero violations.

<sup>4</sup> All CQs, SPARQL queries, and codes for evaluation are available in GitHub (Page 1).

Table 4: Summary of the CQ-based evaluation results.
<table><tr><td>CQ category</td><td>#CQs</td><td>Full</td><td>Partial</td><td>Not</td><td>Coverage</td></tr><tr><td>Core Bibliographic and Retrieval</td><td>24</td><td>23 (95.8%)</td><td>0 (0%)</td><td>1 (4.2%)</td><td>95.8%</td></tr><tr><td>Scholarly History and Relationships</td><td>21</td><td></td><td>16 (76.2%) 4 (19.0%) 1 (4.8%)</td><td></td><td>95.2%</td></tr></table>

Third, the validity and coverage of the knowledge graph were assessed using CQs, which were defined during development to formalize the information requirements the resource is intended to support. A total of 45 CQs were specified, covering the principal retrieval and relational capabilities of the KG, including bibliographic retrieval, scholarly relationships, and temporal and historical information. We organized the CQs into (i) Core Bibliographic and Retrieval and (ii) Mathematical Scholarly History and Relationships. For each CQ, a SPARQL query was executed against the graph, and the result was manually compared with the expected answer. Table 4 summarizes the evaluation results.

The CQ-based evaluation demonstrates broad competency coverage, with 95.6% of evaluated cases being at least partially-supported and 86.7% fullysupported. The partially-supported CQs primarily concern information available through external resources, e.g., Wikidata links that can potentially be used to obtain institutional or country information of the scholar identities. Although such links are available on the platform, they are not currently exposed via the API and thus cannot be incorporated into zbMATH Open KG. This limitation can be addressed by exposing these links, which would enable analyses of scholarly relationships across institutions and other biographical information.

The not-supported CQs concern information not available in the source, e.g., programming language of mathematical software. While zbMATH Open KG links software to publications, it does not currently provide suficient software-specific metadata to analyze trends in mathematical research infrastructure. Such information would require explicit linkage to external software knowledge graphs (e.g., SemRepo [29]). Overall, the evaluation provides an assessment of the current coverage of the resource and to identify areas for future extension.

## 4 Applications and Use Cases

To demonstrate the utility of the knowledge graph, we present several use case examples we collectively term historically grounded scholarly exploration. These use cases illustrate how the distinctive content of the knowledge graph, particularly its extensive temporal coverage, subject classifications based on the MSC taxonomy, expert-curated keywords, and reviewer-author relationships, supports the exploration of mathematical knowledge across generations of scholarship. Importantly, these use cases are not intended to establish historical influence; rather, they demonstrate how the KG can surface relationships and patterns for further scholarly and historical investigation. Each use case is operationalized through SPARQL queries, available in our repository.

## 4.1 Precursor Discovery: Potential Connections Beyond Citation

One potential capability of the zbMATH Open KG is to surface historically earlier contributions that may be foundational and semantically relevant to later developments but are not connected through explicit citation. Identifying such potential “precursor ” connections can help reconstruct a more faithful scholarly lineage and provide a richer view of mathematical knowledge development.

![](images/defb41e631340d1206957f2f1ebfbac3997c98f3b90b623ddb47a87e4e4dfb85.jpg)  
Fig. 2: Potential connections identified through shared concepts and subject classifications.

As an illustrative example of this use case, we queried zbMATH Open KG for the 2020 paper “Modular d<sub>0</sub>-algebras” and identified the 1987 paper “On ideal and congruence lattices of BCK-semilattices” as potential precursor connection. Although separated by 33 years, both works share subject classification codes (MSC:03G25, MSC:06A12) and the keyword BCK-algebra. The earlier work investigates structural properties of BCK-semilattices, while the later work investigates d<sub>0</sub>-algebras in a related mathematical setting. Although the two publications are not explicitly connected by citation, their shared subject classifications and keywords, which were expert-curated, reveal a conceptual connection that may otherwise be dificult to identify through citation-based analysis. The knowledge graph surfaces the 1987 publication as a candidate for further scholarly and historical investigation. Other examples of such cases are illustrated in Figure 2.

## 4.2 Cross-Field Conceptual Continuity via MSC-Prefixes Bridging

The zbMATH Open KG enables exploration of how mathematical concepts recur across research fields through the hierarchical structure of the subject classification i.e., the MSC code. MSC codes provide a relatively stable taxonomy of mathematical subjects [3] and, in the zbMATH Open KG, are assigned to the publication by mathematical experts. Such cross-field conceptual continuity can be dificult to identify from citation networks, as emerging fields may reinterpret established concepts within new frameworks without directly citing relevant earlier work. The zbMATH Open KG supports the systematic identification of recurring concepts across diferent mathematical fields and periods by leveraging MSC code hierarchies with controlled keywords and temporal information.

![](images/8634f492fd62fd88e045944cbc116d904e1694c5db032d122aca881e5bec0d63.jpg)  
Fig. 3: Conceptual Continuity: Algebraic Topology → Homological Algebra.

An illustrative example derived from zbMATH Open KG examines the distribution of the concept spectral sequences across two MSC areas: Algebraic Topology (MSC:55) and Homological Algebra (MSC:18), as illustrated in Figure 3. Within Algebraic Topology, publications associated with the keyword occur in subfields such as homology and cohomology theories (55Nxx) and fiber spaces (55Rxx), where spectral sequences constitute an important methodological theme. The same keyword also occurs in later publications classified under Homological Algebra (18Gxx), illustrating its recurrence in a diferent disciplinary context. For example, the 2021 work “Gabriel-Zisman cohomology and spectral sequences” is associated with the keyword and classified within Homological Algebra.

Using the keyword spectral sequences together with temporal information and MSC classification codes, we identified a total of 463 early–later publication pairs spanning these two MSC areas. These pairs reveal recurring semantic associations across mathematical fields and time periods and provide candidates for further scholarly investigation of cross-field conceptual development.

## 4.3 Conceptual Resurgence: Tracking the Re-emergence of Ideas

Scientific concepts and ideas do not always develop along continuous trajectories. Some may receive sustained attention, decline in prominence, and later reappear in new theoretical or applied contexts. Identifying such patterns can provide historians with candidates for investigating intellectual cycles and help scientists discover areas experiencing renewed activity. The extensive temporal coverage of the zbMATH Open KG supports this exploration by combining temporal information with curated subject classifications and keywords, allowing patterns of emergence, decline, and resurgence to be examined systematically.

![](images/a9039eec70b2f81f88f2923fe780dc723d4a49fc84f123316328b48aa78c2f3d.jpg)  
Fig. 4: Temporal dynamics of mathematical subject areas: publication activity in Probability (MSC 60\*) and Logic (MSC 03\*) by decade.

We queried zbMATH Open KG to illustrate this use case by examining publication activity in Probability (MSC:60) and Logic (MSC:03). Figure 4 presents the results aggregated by decade. Probability shows a sustained increase in publication activity throughout the twentieth century, with more pronounced growth after the 1950s. Representative works associated with this area include Kolmogorov’s 1933 Grundbegrife der Wahrscheinlichkeitsrechnung”. Logic exhibits a marked increase during the 1930s (representative publication: Gödel’s incompleteness theorems), followed by a period of comparatively lower activity and renewed growth from the 1970s onward. Publications in this later period include works associated with fuzzy logic, such as Zadeh’s 1965 Fuzzy Sets”, as well as developments connecting logic with computer science and formal methods. These patterns illustrate how the knowledge graph can make non-linear research dynamics visible at the subject-field level, identifying periods of changing research activity that can serve as starting points for further investigation of the concepts, methods, and publications associated with these transitions.

## 4.4 Potential Intellectual Connections via Scholarly Engagement

The zbMATH Open KG incorporates mathematicians-curated reviews of publications, linking such reviewers to the publications they assessed and their authors through reviewer–author relationships. These connections provide an additional perspective for exploring potential intellectual continuities between a scholar’s earlier scholarly engagement as a reviewer and their subsequent research.

As an example, we queried the zbMATH Open KG for a specific scholar, Hans L. Bodlaender. We identified a potential intellectual connection between his early reviews (2000 or earlier) of graph-theoretic publications (MSC 05C\*) and his later authored work (2000–2025) in the field of computer science (MSC 68\*). Keywords such as treewidth, dynamic programming, and graph algorithm recur across the reviewed and authored publications, indicating topical overlap between research he encountered through reviewing in early years and topics appearing in his later authored publications. These overlaps identify candidate connections that can be further investigated using bibliographic and historical evidence. This example illustrates how zbMATH Open KG can complement publication- and citation-based analysis by incorporating human scholarly engagement into the exploration of potential research continuities and possible pathways of scholarly knowledge transmission.

![](images/87de5d32c95d05bd7aad220877fcf54e91238c697c08ea17c71d2e225e5eb11f.jpg)  
Fig. 5: Potential intellectual continuity identified through author–reviewer relationships: the case of Hans L. Bodlaender.

## 4.5 Other Potential Applications

The zbMATH Open KG provides an open semantic infrastructure for mathematical scholarship that can support applications beyond bibliographic search and citation network-based analysis. Other potential applications include:

Historically Aware Scholarly Information Retrieval. The zbMATH Open KG supports scholarly retrieval that accounts for the historical development of mathematical knowledge over periods of time, supported by its extensive temporal coverage combined with expert-curated semantic information. It can potentially identify semantically related works despite changes in terminology, disciplinary boundaries, or the absence of direct citation links. This provides a basis for historically informed literature discovery that complements conventional keywordand citation-based scholarly search.

Knowledge Development Analytics. The explicit representation of concepts through controlled keywords, subject domains through MSC codes, and expert-curated

reviews, software references, disambiguated scholars, and their relationships, combined with extensive temporal coverage, provides a basis for quantitative analyses of mathematical knowledge over extended periods. Researchers can investigate patterns of emergence, specialization, cross-field continuity, resurgence, and changing research activity, as well as long-term trends in the development of mathematical disciplines over more than 250 years.

Semantic Infrastructure for AI System. The zbMATH Open KG can serve as a semantic backbone for AI-assisted scholarly applications, such as graph-based scientific question answering and recommendation systems [18,22], retrievalaugmented generation, and agentic literature exploration. Particularly, explicit semantic relationships via the hierarchical MSC codes and controlled keywords provide structured sources of context that complement bibliographic metadata and unstructured text, while expert-curated reviews add broader contextual interpretations of publications beyond its original text. This supports more grounded and explainable AI-assisted exploration of mathematical literature.

## 5 FAIR Compliance and Sustainability

Findable Persistent URIs are assigned to all entities, with URI resolution enabled for the core entities (e.g., publications, scholars, software references). RDF dump releases are versioned and accompanied by machine-readable metadata describing the scope, schema, and statistics of the corresponding release. Dataset-level descriptions (VoID, DCAT, PROV-O) support machine-readable discovery and indexing within the Linked Data ecosystem.

Accessible The complete RDF data are distributed as downloadable dumps through Zenodo, providing a persistent distribution channel and long-term preservation of released snapshots. The RDF representation can be loaded into standard RDF triple stores; the pipeline has been tested with Virtuoso and Apache Jena. A public SPARQL endpoint is provided to support direct querying of the published knowledge graph in addition to bulk access through RDF dumps.

Interoperable The KG uses W3C Semantic Web standards and systematically reuses established vocabularies. Where available, external identifiers such as DOIs and other external links and identifiers are retained, facilitating linking with external scholarly resources and knowledge graphs.

Reusable The resource is accompanied by its OWL ontology and data-level description, RDF dumps, documentation, and source code. Stable namespaces and identifiers support the reuse of entities across releases; versioned RDF snapshots enable reproducible experiments against a fixed resource state. Licensing and attribution information for the released data and accompanying artifacts is provided with each distribution. The released knowledge graph and all accompanying artifacts and source code are distributed under CC-BY-SA 4.0.

Sustainability and Maintenance The knowledge graph construction process is designed as a reproducible pipeline over the continuously maintained zbMATH Open platform. Periodic regeneration from updated source data, versioned releases, persistent entity identifiers, and public repository hosting support the continued evolution of the knowledge graph while allowing individual releases to remain independently reproducible. The ontology is maintained separately from individual data snapshots, enabling future extensions without requiring changes to the underlying identifier scheme. RDF dump releases are updated periodically, approximately twice per year subject to the availability of the underlying data and resource and infrastructure constraint.

## 6 Conclusion

We presented zbMATH Open Knowledge Graph, a large scale RDF knowledge graph that integrates more than 250 years of mathematical scholarship grounded in expert-curated semantic information. Its extensive temporal coverage and domain-specific semantic structure provide a reusable foundation for computational exploration of mathematical knowledge beyond conventional bibliographic and citation-based analysis. The zbMATH Open KG is openly available and designed to align with FAIR principles and Semantic Web standards.

We demonstrated the utility of the knowledge graph through historically grounded scholarly exploration use cases, illustrating how it can identify patterns and candidate relationships that may be dificult to uncover in conventional citation networks and provide starting points for further scholarly and historical investigation. Future directions including, but not limited to:

(i) Extending the zbMATH Open KG through links to external knowledge graphs and related resources e.g., linkage to SemRepo[29] could enrich the current mathematical software references, enabling the study of the development and adoption of mathematical software and infrastructure over time.

(ii) Investigating how KG-based context from zbMATH Open KG complements textual and embedding-based methods in AI-assisted retrieval and exploration of the mathematical literature.

## Supplementary Material: See Page 1.

Acknowledgment This work was supported by the LUMEN project, funded by the European Union’s Horizon Europe Research and Innovation Programme (Grant: 101187940, and by the DFG‘s MaRDI project (Grant: 460135501). The views and opinions expressed are those of the authors only and do not necessarily reflect those of the funding organizations or the granting authorities.

Disclosure of Interests. The authors have no competing interests to declare.

Declaration of use of Generative AI. We utilized generative AI to support the writing process throughout this work. All AI-generated content was carefully reviewed, edited, and approved by the authors, who assume full responsibility of the final manuscript.

## References

1. arXiv: e-print archive. https://arxiv.org, accessed: 2025-09-15

2. Dblp computer science bibliography. https://dblp.org, accessed: 2025-09-15

3. American Mathematical Society: Mathematics subject classification 2020. https: //mathscinet.ams.org/mathscinet/msc/msc2020.html (2020), accessed: 2025-09-14

4. Ammar, W., Groeneveld, D., Bhagavatula, C., et al.: Construction of the literature graph in semantic scholar. Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP) pp. 84–94 (2018)

5. Avigad, J., Carneiro, M., Hudon, G.: The Lean mathematical library. Proceedings of the 9th International Conference on Interactive Theorem Proving (ITP 2018) pp. 367–370 (2018)

6. Bian, R., Geng, Y., Yang, Z., Cheng, B.: Automathkg: The automated mathematical knowledge graph based on llm and vector database. Computational Intelligence 41(4) (2025). https://doi.org/10.1111/coin.70096, http://dx.doi.org/10.1111/coin. 70096

7. Boettiger, C., Ooms, J., pdatascience, Leinweber, K., Hester, J.: ropensci/rdflib: v0.2.3 (Jan 2020). https://doi.org/10.5281/zenodo.3604372, https://doi.org/ 10.5281/zenodo.3604372

8. Buzzard, K., Carneiro, M., Massot, P.: Formal abstracts: Integrating formal and informal mathematics. In: Proceedings of the 7th ACM SIGPLAN International Conference on Certified Programs and Proofs (CPP 2018). pp. 191–203 (2018)

9. Chebukov, D.E., Izaak, A.D., Misyurina, O.G., Pupyrev, Y.A., Zhizhchenko, A.B.: Math-Net.Ru as a Digital Archive of the Russian Mathematical Knowledge from the XIX Century to Today, pp. 344–348. Springer Berlin Heidelberg (2013). https://doi.org/10.1007/978-3-642-39320-4\_26, http://dx.doi.org/ 10.1007/978-3-642-39320-4\_26

10. Council, N.R.: Developing a 21st Century Global Library for Mathematics Research. The National Academies Press (2014). https://doi.org/10.17226/18619

11. DCMI: Dublin core metadata element set, version 1.1. https://www.dublincore. org/specifications/dublin-core/dces/ (2012), accessed: 2025-09-14

12. Deb, M., Beckenbach, I., Petrera, M., Ehsani, D., Fuhrmann, M., Hao, Y., Teschke, O., Schubotz, M.: An overview of zbmath open digital library. In: Proceedings of the 24th ACM/IEEE Joint Conference on Digital Libraries. p. 1–5. JCDL ’24, ACM (Dec 2024). https://doi.org/10.1145/3677389.3702597, http://dx.doi.org/10.1145/ 3677389.3702597

13. Elizarov, A., Kirillovich, A., Lipachev, E., Nevzorova, O.: Ontomath digital ecosystem: Ontologies, mathematical knowledge analytics and management (2017), https: //arxiv.org/abs/1702.05112

14. Elizarov, A., Kirillovich, A., Nevzorova, O., Yagudin, R.: Ontomathpro ontology: A linked data hub for mathematics. In: CEUR Workshop Proceedings. vol. 1018, pp. 100–107 (2013)

15. Erling, O., Mikhailov, I.: RDF Support in the Virtuoso DBMS, pp. 7–24. Springer Berlin Heidelberg, Berlin, Heidelberg (2009), https://doi.org/10.1007/ 978-3-642-02184-8\_2

16. Färber, M.: The microsoft academic knowledge graph: A linked data source with 8 billion triples of scholarly data. In: Ghidini, C., Hartig, O., Maleshkova, M., Svátek, V., Cruz, I., Hogan, A., Song, J., Lefrançois, M., Gandon, F. (eds.) The Semantic Web – ISWC 2019. pp. 113–129. Springer International Publishing (2019)

17. Färber, M., Lamprecht, D., Krause, J., Aung, L., Haase, P.: Semopenalex: A linked open data version of openalex. In: The Semantic Web – ISWC 2023. pp. 94–112. Springer Nature Switzerland (2023)

18. Färber, M., Lamprecht, D., Susanti, Y.: Bridging rdf knowledge graphs with graph neural networks for semantically-rich recommender systems. In: Database Systems for Advanced Applications. pp. 425–437. Springer Nature Singapore, Singapore (2026)

19. Färber, M., Lamprecht, D.: Linked papers with code: The latest in machine learning as an rdf knowledge graph (2023), https://arxiv.org/abs/2310.20475

20. Gonthier, G., Asperti, A., Avigad, J., Bertot, Y., Cohen, C., Garillot, F., Le Roux, S., Mahboubi, A., O’Connor, R., Tassi, E., Théry, L.: Mathematical components. Inria Research Report (RR-6455) (2008)

21. Guha, R.V., Brickley, D., Macbeth, S.: Schema.org: Evolution of structured data on the web. Communications of the ACM 59(2), 44–51 (2016). https://doi.org/10. 1145/2844544

22. Kelber, F., Jobst, M., Susanti, Y., Färber, M.: Do we need bigger models for science? task-aware retrieval with small language models. In: Proceedings of Natural Scientific Language Processing (NSLP) @ LREC 2026. pp. 108–118. European Language Resources Association (ELRA), Palma, Mallorca, Spain (May 2026). https://doi.org/10.63317/2cuutd9tnnhp

23. Lopez, M., Gomez-Perez, A., Sierra, J., Sierra, A.: Building a chemical ontology using methontology and the ontology design environment. IEEE Intelligent Systems and their Applications 14(1), 37–46 (1999). https://doi.org/10.1109/5254.747904

24. Miles, A., Bechhofer, S.: Skos simple knowledge organization system reference. Tech. Rep. W3C Recommendation, W3C (2009), https://www.w3.org/TR/ skos-reference/

25. Peroni, S.: A simplified agile methodology for ontology development. In: OWL: Experiences and Directions – Reasoner Evaluation: 13th International Workshop, OWLED 2016, and 5th International Workshop, ORE 2016, Bologna, Italy, November 20, 2016, Revised Selected Papers. p. 55–69. Springer-Verlag, Berlin, Heidelberg (2016). https://doi.org/10.1007/978-3-319-54627-8\_5, https://doi.org/10. 1007/978-3-319-54627-8\_5

26. Peroni, S., Shotton, D.: The citation typing ontology (cito) and its use for semantic enhancement of scientific documents. Journal of Web Semantics 17, 39–43 (2012). https://doi.org/10.1016/j.websem.2012.08.001

27. Poveda-Villalón, M., Fernández-Izquierdo, A., Fernández-López, M., García-Castro, R.: Lot: An industrial oriented ontology engineering framework. Engineering Applications of Artificial Intelligence 111, 104755 (2022). https://doi.org/ https://doi.org/10.1016/j.engappai.2022.104755, https://www.sciencedirect.com/ science/article/pii/S0952197622000525

28. Priem, J., Piwowar, H., Orr, R.: Openalex: A fully open index of scholarly works, authors, venues, institutions, and concepts. https://openalex.org/ (2022), accessed: 2025-08-21

29. Rafay, A., Susanti, Y., Lamprecht, D., Färber, M.: Semrepo: A knowledge graph for research software and its scholarly ecosystem (2026), https://arxiv.org/abs/ 2605.13310

30. Schubotz, M., Ferrer, E., Stegmüller, J., Mietchen, D., Teschke, O., Pusch, L., Conrad, T.O.: Bravo mardi: A wikibase powered knowledge graph on mathematics (2023), https://arxiv.org/abs/2309.11484

31. Sinha, A., Shen, Z., Song, Y., Ma, H., Eide, D., Hsu, B.J.P., Wang, K.: Microsoft academic graph: When experts are not enough. Quantitative Science Studies 1(1), 396–413 (2020)

32. Tomaszuk, D., Szeremeta, Ł., Korniłowicz, A.: Mmlkg: Knowledge graph for mathematical definitions, statements and proofs. Scientific Data 10, 791 (2023). https://doi.org/10.1038/s41597-023-02681-3

33. Wilkinson, M.D., Dumontier, M., Aalbersberg, I.J., Appleton, G., Axton, M., Baak, A., Blomberg, N., Boiten, J.W., Bonino da Silva Santos, L., Bourne, P.E., Bouwman, J., Brookes, A.J., Clark, T., Crosas, M., Dillo, I., Dumon, O., Edmunds, S., Evelo, C.T., Finkers, R., Gonzalez-Beltran, A., Gray, A.J.G., Groth, P., Goble, C., Grethe, J.S., Heringa, J., ’t Hoen , P.A.C., Hooft, R., Kuhn, T., Kok, R., Kok, J., Lusher, S.J., Martone, M.E., Mons, A., Packer, A.L., Persson, B., Rocca-Serra, P., Roos, M., van Schaik, R., Sansone, S.A., Schultes, E., Sengstag, T., Slater, T., Strawn, G., Swertz, M.A., Thompson, M., van der Lei, J., van Mulligen, E., Velterop, J., Waagmeester, A., Wittenburg, P., Wolstencroft, K., Zhao, J., Mons, B.: The fair guiding principles for scientific data management and stewardship. Scientific Data 3, 160018 (2016). https://doi.org/10.1038/sdata.2016.18

34. About zbmath. https://zbmath.org/about/, accessed: 2025-09-12