# Selective Retrieval Routing for Multi-Violation Knowledge Graph Repair

Tung-Wei Lin UC Berkeley Berkeley, USA

Gabe Fierro Colorado School of Mines Golden, USA

Han Li Tianzhen Hong Lawrence Berkeley National Laboratory Berkeley, USA

Pierluigi Nuzzo   
Alberto Sangiovanni-  
Vincentelli   
UC Berkeley   
Berkeley, USA

## ABSTRACT

Knowledge graphs often contain missing or inconsistent facts that undermine the applications built on them. Schema validation de tects such violations but does not identify which of several valid repairs recovers the intended graph. External corpora can supply the evidence needed to identify the intended repair, but retrieving from them is costly and not always successful. We propose a framework for selective retrieval in iterative graph repair. An LLM repair agent explores and edits the graph, while a router controls its access to a researcher agent that retrieves external evidence. We extend a violation-injection framework to generate cases with multiple compatible violations and known reference repairs. The agent’s repair outcomes on these cases provide supervision for training the deployment-specific router. On two building-automation datasets, researcher access improves repairs for some violations and degrades them for others, with efects difering across deployments. The router uses fewer repair-time tokens than always enabling access in all six evaluation suites. It achieves higher repair � than always disabling access in five of six suites.

## PVLDB Reference Format:

Tung-Wei Lin, Gabe Fierro, Han Li, Tianzhen Hong, Pierluigi Nuzzo, and Alberto Sangiovanni-Vincentelli. Selective Retrieval Routing for Multi-Violation Knowledge Graph Repair. PVLDB, 20(1): XXX-XXX, 2027. doi:XX.XX/XXX.XX

## 1 INTRODUCTION

Knowledge graphs (KGs) are an established abstraction for organizing heterogeneous data around entities and their relationships. Public knowledge graphs such as DBpedia, Wikidata, and YAGO support information retrieval and other knowledge-intensive appli cations [12, 54, 57]. In domain-specific operational settings, building ontologies such as Brick describe equipment, sensors, spaces, and their relationships so that analytics can be deployed across otherwise heterogeneous buildings [13, 24]. Across these uses, the value of the graph depends on its quality: missing, stale, or inconsistent facts can cause queries and downstream applications to operate on an incorrect representation of the domain.

Such defects are dificult to avoid: KGs are typically assembled by integrating databases, tables, text, and other heterogeneous sources, often through imperfect automated extraction [39, 40]. KGs can also drift out of date as the represented domain and its source artifacts evolve. Graph constraint formalisms such as graph-generating dependencies [53], SHACL [2], and PG-Schema [7] provide declarative specifications of the conditions that a valid graph must satisfy. Validating a graph against these constraints identifies violations, but does not describe how to fix them.

Graph repair involves two problems: identifying repairs that restore schema validity, and selecting which of these valid repairs recovers the intended graph. Symbolic reasoning and cost-based optimization techniques can compute valid repairs [6, 8, 17], using minimality to choose among them. This limits how much the graph changes, but does not establish which repair accurately represents the domain: the intended repair may require more edits than the smallest one. Human-validated values and curated master data can resolve this ambiguity [22, 23], but are limited by existing coverage and require human efort to maintain. Recent methods also use large language models (LLMs) to inform candidate repairs, prompting models with graph context to propose corrections using knowledge acquired during pretraining [9, 56]. Evidence can also be obtained directly from documents describing the modeled domain. LLM-assisted graph completion and maintenance systems use retrieved passages to verify proposed triples or suggest replacements for outdated facts [27, 63]. This provides a way to draw on existing knowledge corpora when choosing among schema-valid repairs. However, the corpus is not guaranteed to contain the necessary facts, and querying it is not reliably successful. Recent work on retrieval-augmented generation (RAG) shows that retrieval’s benefit varies across tasks [11], and that retrieved passages can mislead a model that would otherwise answer correctly [38].

We study an LLM graph-repair agent that explores the graph and synthesizes updates, with optional access to a researcher agent that answers the repair agent’s questions by retrieving from the external corpus; when access is enabled, the repair agent chooses whether to query the researcher and what to ask.

Our results show that the benefit of researcher access is heterogeneous: it helps strongly for some kinds ofviolations, only marginally for others, and for some kinds it can actively hurt, leading the agent toward worse repairs than it would have produced on its own. This heterogeneity depends on the relationship among the constraints, the violation, the graph, and the deployment’s retrieval pipeline. When retrieval fails to improve a repair, the tokens spent querying the corpus add cost without improving quality. Dynamically enabling researcher access for some violations and disabling it for others can therefore improve the tradeof between repair quality and token cost.

This raises the question: when should an LLM graph-repair agent be given access to external knowledge? Hand-coding this decision requires knowing which violations benefit from retrieving from the particular deployment’s corpus; the answer can change when either the schema or the corpus changes.

We address this question by training a router that predicts, for each group of related violations, whether researcher access will improve the repair. Since measuring repair quality requires knowing the intended repair, we construct test cases with known ground truth by introducing controlled sets of violations into a valid graph, varying the targeted predicates to obtain diverse cases. We use these cases both to characterize researcher utility and to train the router. We show how controllable multi-violation generation enables learning a deployment-specific access policy from a schema, a valid reference graph, and an accompanying corpus. SHAP analysis [37] of the learned router shows that the same predicate can increase the predicted benefit of access in one deployment and decrease it in another, illustrating how the learned predictions depend on the deployment.

We evaluate on two building-automation datasets with expressive SHACL constraints, instance graphs, and supporting document corpora (Sec. 3.2). Our contributions are as follows:

• Heterogeneous benefit of retrieval: We show empirically that the efect of researcher access varies across violation types, improving some repairs and worsening others, even though the repair agent chooses whether to query the researcher.

• Multi-violation test generation: We extend the framework of [35] to generate test cases containing multiple compatible violations with known ground-truth repairs, varying the targeted predicates to produce diverse cases for evaluating repair agents and access policies.

• Learned access routing: We train a router on repair outcomes from generated cases to decide, for each group of related violations, whether to give the repair agent access to the researcher. In five of six evaluation suites, the resulting policy achieves higher repair $F _ { 1 }$ than never granting access while using fewer repair-time tokens than always granting access.

## 2 RELATED WORK

Existing constraint-based repair methods search the space of possi ble repairs using cost-based optimization [17], probabilistic inference [51], and symbolic reasoning [6]. A common repair criterion is minimality [8], which limits changes to the data but does not establish whether the resulting repair accurately represents the domain: a minimal repair may introduce incorrect information or remove correct information [6, 22]. Semi-automated methods instead rely on additional evidence, such as curated master data with user-validated attributes [22], or user-validated values and entity matches [23]. Because these methods use trusted evidence to identify correct repairs, their coverage depends on how complete that evidence is. Our researcher agent serves a similar role, but consults an unstructured, multi-modal document corpus. The corpus is not guaranteed to contain the necessary facts, and querying it is neither free nor reliably successful.

Another line of work learns repair patterns from previous corrections. Edit histories can be used to mine correction rules [44] or train a neural network [45]. LLMs ofer a similar alternative: they have been applied to data wrangling [41] and graph repair by treating the model’s parametric knowledge as the evidence source [9, 56]. These approaches depend on the availability of edit histories or relevant knowledge in the model; deployment-specific facts may instead require consulting external artifacts. Related work on knowledge graph completion uses graph structure and parametric knowledge to propose candidate facts [60, 63]. Document-based verification and maintenance use external evidence to check proposed facts or suggest replacements for outdated ones [27, 63].

Using external evidence also introduces a decision about when to retrieve it. Selective RAG methods use signals about the model’s knowledge, such as entity popularity [38], or assess the need for retrieval during generation [11, 31]. Adaptive-RAG learns to select among no retrieval, single-step retrieval, and iterative retrieval based on query complexity, using automatically collected labels derived from model outcomes and dataset characteristics [30]. Our work similarly learns from observed outcomes. In our setting, controlled graph corruption supplies reference repairs, and the router learns from outcomes along researcher-enabled and researcherdisabled repair trajectories. The learned policy controls access for each violation cluster within an iterative graph-repair process.

Our decision concerns whether to make a retrieval tool available to the repair agent, which, when access is enabled, chooses whether to query it and what to ask. Tool-selection methods retrieve relevant tools from large catalogs [49]; here, the available tool is fixed, and the question is whether access to it improves a particular repair.

Learning and evaluating this policy requires repair cases with ground truth. Controlled error injection provides a way to generate such cases at scale; for example, BART [10] introduces constraint violations into clean relational data, with controls over error frequency, detectability, and repairability. We reuse the derivation of individual violation-inducing edits from the graph violationgeneration framework of [35]. Our extension samples and composes compatible expressions into multi-violation cases with targeted predicate coverage. The resulting cases support both the study of retrieval utility and training the access policy.

## 3 BACKGROUND AND PROBLEM DEFINITION

We build directly on the abstract syntax introduced in [5] and the violation-generation framework in [35].

## 3.1 Graph Repair Task

We adopt the common graph model [5, 35], where the instance graph is a set of triples relating nodes by predicates or assigning values to node keys. This common graph model generalizes both RDF [1] and property graphs [7]. A schema S is a set of constraints (sel, �): the selector sel picks the nodes in the instance graph to which the constraint applies, each of which must conform to shape �. A violation occurs when a node selected by sel does not conform to �. An instance graph with at least one violation is invalid; otherwise, it is valid. We instantiate constraints in SHACL [2] and target RDF graphs for concreteness, but the formulation is language-agnostic and applies equally to ShEx [3] and PG-Schema [5].

![](images/174721888c647aace5c5574f0fff32d1a0782aedf26e9d73b3928ac68cb44915.jpg)  
Figure 1: The illustrated shape requires each HVAC zone to reach exactly one air-handling unit through one or more cnx edges (cnx<sup>+</sup>). Adding one edge between the existing outlet and inlet connection points satisfies this requirement. However, the equipment schedule identifies an intermediate variable air volume (VAV) box on the supply path. Recovering this documented path requires adding the VAV box and its connection points, yielding a larger (but correct) repair.

The atomic edit operations on an instance graph are the addition and removal of triples. A repair is a sequence of such operations that transforms an invalid graph into a valid one. For RDF graphs, repairs can be expressed as SPARQL updates [4].

Given an invalid instance graph and schema, the graph repair task is to produce a sequence of edits that recovers the intended graph. Multiple repairs may restore schema validity, and the intended repair may require more edits than the smallest one or be one of several equally small alternatives. In cyber-physical knowledge graphs [13, 15, 26, 50], the intended graph represents a particular real-world system, so choosing among repairs may require evidence not represented in the graph or schema.

For evaluation, we use the instance graphs supplied by the dataset authors as reference graphs and assume their factual correctness. A test case is produced by injecting violations into a reference graph: applying recorded edit operations that make selected nodes violate constraints in S. Reversing these edits recovers the reference graph and provides a ground-truth repair for the test case. Other edits may also restore schema validity; we measure repair accuracy by agreement with the reference edits and report schema validity separately (Sec. 5). Sec. 6 shows how to generate test cases with targeted predicates.

## 3.2 Study Setting and Datasets

We instantiate the graph repair task using two building-automation datasets modeled with ASHRAE Standard 223P [48]: the Pritoni dataset [47] and the IBAL dataset [46].

We selected these datasets to satisfy two requirements. First, evaluating a retrieval-augmented system requires the LLM to not simply rely on its parametric memory. Benchmark contamination is well-documented in LLM evaluation [14, 52]; if an agent has already memorized a widely referenced knowledge base like DBpedia [12] or Wikidata [57], it bypasses the need for external retrieval en tirely, confounding the evaluations. Because building-automation metadata is highly specialized, and because the Pritoni corpus is manually created by us, these datasets let us study retrieval of deployment-specific facts that are less likely to be available in the model’s parametric knowledge.

Second, we require three artifacts describing the same system: a schema, a valid reference graph, and an external corpus that grounds the graph’s facts. To our knowledge, no existing benchmark provides all three. Graph-repair benchmarks provide graphs and constraints but lack supporting documents [25]; graph-construction benchmarks supply schemas and text but lack a reference instance graph to repair [40]; and question-answering corpora lack the graph and schema [30, 38]. We therefore assemble these artifacts around the Pritoni and IBAL instance graphs, constructing a corpus for Pritoni and using actual deployment documents for IBAL.

Both the Pritoni and IBAL datasets are validated against the 223P schema, which models cyber-physical building systems by defining an ontology to capture the physical and logical relationships between equipment and sensors. For example, the observes predicate links a sensor to its measured quantities (e.g., a room temperature). Similarly, the hasConnectionPoint predicate links a piece of equipment to a port through which it exchanges a medium with other equipment (e.g., the supply-air outlet of an air-handling unit, joined by a duct to the inlet of a room).

Besides defining an ontology, 223P enforces 1,782 constraints on these relationships. For instance, a minCount constraint triggers a violation if a sensor node lacks observes relationships. Represented as an RDF graph, the 223P schema comprises 8,248 triples (expanding to 96,906 triples after resolving external imports). The majority of these triples define the underlying ontology and class hierarchies. We now detail the instance graph and external corpus of each dataset.

Pritoni Dataset. The Pritoni dataset models a single-story prototype building with five rooms spanning five distinct space types [47]. Its instance graph contains 1,707 triples over 384 entities. Because the original release provided only the instance graph and the 223P schema, we manually constructed an accompanying external corpus to simulate real-world documentation: one network point list (a table of building-automation data points and their descriptions), two equipment schedules, four schematics (drawings and floor plans), and an 800-word HVAC sequence-of-operation document. To reduce the risk of data leakage, this corpus was authored prior to violation injection and without reference to the agents’ prompts or tools, limiting opportunities to tailor the documents to the sampled corruptions or agent configuration.

IBAL Dataset. The IBAL dataset models a laboratory at the National Institute of Standards and Technology (NIST). Unlike Pritoni, IBAL is a real-world deployment and includes authentic engineering drawings and schedules. Its instance graph is roughly 3.5× larger than Pritoni’s, containing 6,016 triples over 1,357 entities. The accompanying external corpus comprises one network point list, 14 equipment schedules, and 16 engineering schematics.

We generate test cases from each dataset using the violationinjection framework of Sec. 6. These repair cases are newly generated for this study, reducing the risk of evaluating on instances memorized during LLM pretraining. The repair system must recover the reference graph, with the accompanying corpus available as an additional source of evidence. We next describe the repair system that operates on these test cases.

## 4 MULTI-AGENT REPAIR SYSTEM

In our system, an LLM repair agent explores the instance graph and schema, synthesizes candidate updates, and uses validation feedback to revise them. When the schema permits several valid repairs, the agent may need external evidence to recover the intended one.

Consider a graph that contains an air-handling unit and a zone in a heating, ventilation, and air-conditioning (HVAC) system, but lacks the supply path connecting them (Fig. 1). Adding a single cnx edge between the existing outlet and inlet connection points satisfies the illustrated shape with one triple addition, yielding a minimal repair. However, the accompanying equipment schedule identifies an intermediate variable air volume (VAV) box on the supply path. Recovering the documented topology requires introducing the VAV box and its connection points, assigning their types, and adding the connecting edges. Methods that minimize the number of edits would favor the one-edge repair over this larger reconstruction, illustrating why minimality alone does not identify the intended graph and why external evidence may be needed.

The repair agent can obtain evidence by consulting an optional researcher agent, which searches the external corpus and answers the repair agent’s questions. When access is enabled, the repair agent chooses whether to query the researcher and what to ask, so having access does not mean that every repair involves retrieval. Our router decides whether to enable this access for each violation cluster, a group of related violations that are repaired together. We use the open-weight model Gemma-4-31B [55] for all LLM calls.

## 4.1 System Overview

Our repair system addresses two practical challenges. First, realworld graphs may contain more violations than an LLM agent can reason about within its efective context window [36]. Second, fix ing violations in isolation may yield more incorrect repairs than resolving interacting violations jointly [20]. Thus, to satisfy LLM context limits while preserving interacting violation context, we group violations into clusters to resolve together. Finally, because a repair can ripple outward to resolve or create violations elsewhere, we revalidate the entire graph and recompute these clusters iteratively.

As shown in Fig. 2, during each iteration, we first validate the instance graph against the schema to identify violations. We then group related violations in the same cluster. The system selects one cluster, synthesizes a repair for it, and discards the rest since updating the graph renders the remaining clusters stale. After applying the repair to the full graph, we revalidate the updated graph to yield a fresh set of violations, initializing the next iteration. When a repair attempt makes no progress, its violations are temporarily excluded from clustering and become eligible again after another repair makes progress. The loop stops when the graph is valid or when it reaches five consecutive iterations without progress.

![](images/a80b3acd011c8361f8d99f78be8e203c27b2e590aa5468a20e9c1ce76023a323.jpg)

Figure 2: Overview of our iterative graph repair system, showing how violation clusters are provided to a repair agent, which optionally invokes a researcher agent to aid in generating a valid repair.  
![](images/2f0ee6843060bc7918cdf9c379b75a57d723bd91e3bb7eb6ebe280430a8417ad.jpg)  
Figure 3: Iterative repair pipeline with agentic graph editing and SHACL validation. Graph exploration tools and the researcher agent also update memory through the tool-call history; these updates are omitted for clarity.

Our implementation has three components: a validator, a clustering algorithm, and a multi-agent repair component.

Recent systems like AutoGen [62] and MetaGPT [29] demonstrate that delegating tasks across specialized agents reduces tool confusion and context pollution compared to monolithic agents. Motivated by this, our system centers on a repair agent that explores and edits the graph and can optionally be equipped with a researcher agent that consults an external corpus. We evaluate the repair system under two prompt configurations. In the OFF configuration, the repair agent operates without access to the researcher and must rely solely on the graph and schema. In the ON configuration, the researcher is enabled, allowing the repair agent to ask the researcher natural-language questions whose answers are obtained from the external corpus. Sec. 5 formalizes the corresponding experimental comparison.

Following Voyager [58], we implement each agent as an iterative loop over structured memory rather than as a multi-turn conversation. At each step, we construct a fresh prompt containing the current state and a trace of the agent’s prior tool calls, then ask the LLM to select the next action. We use two mechanisms to keep this prompt tractable during long reasoning chains. First, every four iterations, a separate LLM summarizes the tool-call history and replaces the execution logs in memory. Second, just before the thirty-iteration limit, we append a “wrap-up” instruction in the message to stop it from exploring and instruct it to commit to its best final action.

## 4.2 Validator

We validate the instance graph using the SHACL validation engine pySHACL [18], yielding a set of violations. Each violation is augmented with two context payloads that will be used by the repair agent: a schema subset, consisting of the failed constraint and its dependencies, and an instance subgraph, containing the triples touched during validation of the failed constraint. The clustering algorithm (Sec. 4.3) uses the instance subgraph to group violations, while the repair agent uses the schema subsets for local revalidation.

## 4.3 Clustering and Batching

We group violations into disjoint clusters to reuse shared context while limiting the context size with a budget �. Initially, each violation forms its own cluster. Merging two clusters takes the union of their instance subgraphs and the union of their schema subsets. Triples appearing in both clusters are included only once in the merged context, reducing the space needed to consider the violations together.

Formally, the clustering is based on two criteria: a size budget and subgraph overlap. Because the context provided to the repair agent includes both the instance graph and schema information, the size of a cluster is defined as the combined size of its underlying unioned instance subgraph and schema subset. We measure this size in nodes and set � = 20 as a heuristic for these datasets; no individual violation exceeds this budget in our experiments. We measure overlap using the Jaccard similarity of the instancesubgraph node sets, favoring clusters that concern the same graph region. This increases the possibility of afecting multiple violations through a repair to the shared region. Restricting the definition of overlap exclusively to the instance graph prevents clustering spatially distant violations merely because they violate the same schema constraints.

Algorithm. Algorithm 1 uses a greedy agglomerative heuristic. Every violation starts as its own cluster. We repeatedly merge the pair with the highest positive overlap among those whose combined size stays within �. We stop when no such merge remains, then pass the cluster containing the most violations to the repair agent. This procedure favors context sharing without claiming to find the largest feasible cluster.

## 4.4 Repair Agent

As shown in Fig. 3, the repair agent reads one cluster of violations and reasons about how to resolve them. In both ON and OFF configurations, the repair agent has access to the first three tools below (two graph exploration tools and one graph edit tool). In the ON configuration, it additionally has access to the researcher. Since both instance graph and SHACL schema are represented as RDF graphs, the agent uses the tools to explore both the instance graph and schema. Any attempt to edit the schema is rejected.

Algorithm 1 Constrained Agglomerative Clustering   
Input: Violations � , size budget �   
Output: A single cluster to pass to the repair agent   
1: $C \gets \{ \{ v \} : v \in V \}$ ⊲ each violation starts in its own cluster   
2: loop   
3: $\mathcal { P }  \{ ( c , c ^ { \prime } ) \in C \times C : c \neq c ^ { \prime } , $ size(� ∪ �<sup>′</sup>) ≤ � }   
4: if P = ∅ or max<sub>(</sub> ′<sub>)∈P</sub> overlap(�, �<sup>′</sup>) = 0 then   
5: break   
6: end if   
7: $( c ^ { \star } , c ^ { \prime \star } ) \gets$ arg max overlap(�, �<sup>′</sup>)   
(�,�<sup>′</sup> ) ∈ P   
8: $C \gets ( C \setminus \{ c ^ { \star } , c ^ { \prime \star } \} ) \cup \{ c ^ { \star } \cup c ^ { \prime \star } \}$ ⊲ merge best admissible   
pair   
9: end loop   
10: return arg max |�|   
�∈ C

(1) Neighbor Traversal: Given a target node, return its immediate neighbors (i.e., the nodes pointing to or from it), optionally filtered by type, keyword, or predicate. This is used when the agent already knows the target node of interest.

(2) Keyword Search: Given a keyword, return nodes with matching labels, optionally filtered by type. This is used when the agent has not yet located a node within the graph, for example, to match an entity mentioned by the researcher to its exact counterpart in the instance graph.

(3) Graph Edit: The agent produces a repair as a SPARQL update, which is applied to the instance subgraph of the cluster and then automatically revalidated against the schema subset of the cluster. If the edited subgraph satisfies this schema subset, the loop exits and the repair is applied to the full graph; otherwise the agent continues, informed by the new validation outcome.

(4) Researcher Agent (only for ON configuration): Query a researcher that consults the external corpus. Provisioning this tool makes retrieval available but does not force its use: the repair agent independently decides whether and when to invoke it. In the OFF configuration, the tool is omitted entirely. For example, given a node Channel\_78, the repair agent might ask the researcher: “What property does Channel\_78 observe? Is it a temperature sensor?”

## 4.5 Researcher Agent

As shown in Fig. 4, the researcher answers questions by searching an external corpus with multiple modalities: text, tables, and images (drawings, schematics, floor plans). It has the following tools:

(1) Term Search: Given an entity or keyword, the researcher queries a pre-computed index to locate relevant files. This index is constructed once prior to any retrieval by prompting an LLM to extract entities from images and by aggregating cell values from tables, enabling the agent to eficiently find where specific terms are mentioned.

(2) Modality-specific readers: Once relevant files are identified, the researcher reads them in detail using a text reader, an image reader, or a SQL interface for tables, as appropriate.

![](images/53393a310afbc559ae5b3767564751403c30bc68365ebd3f932ccfac61cbe90d.jpg)  
Figure 4: Iterative research pipeline with external corpus.

(3) Report: Concludes the research and returns the findings in natural language, e.g., it may answer: “Channel\_78 (measurement\_id: ahu2\_heat\_rtd) observes temperature; it is a temperature sensor measuring the temperature downstream ofthe electric heater in AHU2.”

The researcher returns its report to the repair agent, which combines the report with its graph exploration to synthesize edits for the current cluster.

Beyond mitigating context bloat, this multi-agent architecture isolates the researcher as a modular tool, allowing us to control the repair agent’s access to the external corpus. In the ON configuration, the repair agent’s prompt includes the researcher’s input-format description and usage instructions, making the researcher available as a tool. In the OFF configuration, these components are omitted, restricting the repair agent to graph exploration and editing. This toggle defines the ON and OFF conditions evaluated in Sec. 5 and the binary action space of the routing policy in Sec. 7, which selects a prompt configuration for each violation cluster.

## 5 MOTIVATING STUDY: RETRIEVAL IS HETEROGENEOUS

The ON/OFF toggle determines whether the repair agent can consult the researcher. We can therefore measure how access to the researcher changes repair outcomes. Our results show that the utility of retrieval from an external corpus is not uniform across violations. We organize the study around two factors: the repair’s informa tion requirement and the retrieval pipeline’s eficacy. These factors guide our predicate groupings and help interpret diferences across deployments. We therefore learn a deployment-specific routing pol icy from observed retrieval benefits, regardless of their underlying cause.

Information Requirement. Information requirement distinguishes repairs inferable from the invalid graph and schema (internal) from those requiring external evidence (external), following Paulheim’s taxonomy on graph refinement methods [43]. The predicates involved provide an observable proxy for this requirement because they indicate what kind of information must be recovered. For example, repairing a missing observes relationship requires identifying what a sensor measures—a physical fact that a network point list may supply. Recovering a missing hasConnectionPoint relationship may instead require identifying an existing connectionpoint node through label matching and schema-based type filtering, using techniques similar to those in link discovery [42]. These difer ences motivate grouping predicates by their expected information requirements and testing whether retrieval utility varies across the resulting workloads. The actual requirement also depends on the context available in the graph.

Pipeline Eficacy. Pipeline eficacy concerns whether the corpus and researcher together can supply useful evidence to the repair agent. The kind of relationship expressed by a predicate also suggests how that evidence may be documented and recovered. For example, a network point list may explicitly identify what a sensor measures, while recovering a connection shown in a schematic may require interpreting visual conventions and inferring the relationships they express. Documents may describe physical equipment and connections without naming ontology constructs such as ConnectionPoints; using this evidence for repair also requires mapping it to the schema’s representation. Eficacy therefore depends on whether the deployment’s documents contain suficient evidence and whether the researcher can retrieve and interpret it. We treat the corpus and researcher as a single pipeline whose output is the report available to the repair agent; Sec. 10 discusses the challenge of separating their contributions. Diferences in documentation and evidence recovery may therefore yield diferent retrieval benefits even for the same predicate, motivating comparisons across both predicate mixtures and deployments.

Retrieval utility depends on both what information a repair requires and whether the retrieval pipeline can supply useful evidence. A repair may require facts absent from the graph, yet gain little from retrieval if the researcher cannot recover them. Conversely, external evidence may help the agent even when the repair is inferable from the graph and schema. These factors motivate studying retrieval utility empirically for each deployment.

## 5.1 Instantiation in 223P

To study heterogeneity in retrieval utility in 223P, we group predicates into three broad classes based on where the information needed for repair is likely to be found: evidential, terminological, and representational. We made this grouping by inspecting the 223P schema and considering what information is typically available in engineering documents. It provides a broad characterization of 223P rather than an exact taxonomy. Other ontologies may call for diferent groupings, and retrieval utilities may vary within a class across deployments.

Evidential predicates describe relationships that are commonly documented in engineering artifacts, including point lists, equipment schedules, floor plans, P&IDs, and sequences of operation. Examples include observes, hasPhysicalLocation, actuates, and contains. For example, an equipment schedule or floor plan may identify a sensor’s location.

Terminological predicates are used to represent engineering concepts using the vocabulary defined by 223P and related ontologies. Examples include hasProperty, actuatedByProperty, hasDomainSpace, and QUDT’s hasQuantityKind. Engineering documents often express these concepts through abbreviations, project-specific labels, or ordinary engineering language, which must be mapped to the terms available in the ontology.

Representational predicates are used to construct relationships through structures required by the ontology. Engineering documents may describe a relationship without using the intermediate entities and links through which 223P represents it. The connection entities and points in Fig. 1, linked by predicates such as cnx, connectsThrough, and hasConnectionPoint, illustrate this structure. These intermediate entities are similar to surrogate keys in a relational database: they are identifiers created by the data model to connect records. Repairing violations involving these predicates often requires constructing the intended structure from the schema and the surrounding graph.

We use these classes as a diagnostic lens for constructing test suites and examining how retrieval utility varies across predicate mixtures and deployments. They represent broad expectations about where repair information may reside.

## 5.2 Measuring the Efect

To measure the variation in retrieval utility, we use the violation injection framework to be introduced in Sec. 6 to introduce violations into the valid Pritoni and IBAL instance graphs. For each dataset, we generate three test suites, each predominantly targeting predicates from the evidential, terminological, or representational class. Each combination of dataset and predicate mixture is a suite of 80 test cases. A test case is also characterized by a parameter � (the injection budget of the number of perturbations) and we distribute the suite over eight levels, $K \in \{ 2 , 5 , 1 0 , 1 5 , 2 0 , 2 5 , 3 0 , 3 5 \}$ generating ten test cases at each level. Sweeping � lets us examine retrieval utility across repair tasks of varying size and complexity.

Given a test case and its ground-truth repair, we evaluate the repair system along three dimensions:

(1) � Score: We compare the set of edit operations proposed by the repair system with the ground-truth edit set. The resulting $F _ { 1 }$ score captures the precision and recall of the edits needed to reverse the injected perturbation.

(2) Schema Validity: We record whether the graph produced by the repair system is valid, i.e., whether the edited graph has no remaining violations. We report this metric alongside $F _ { 1 }$ because multiple repairs can satisfy the schema.

(3) Token Cost: We measure repair-time token consumption, including all LLM calls for external research, summarization, and repair generation.

Every generated test case is evaluated under both ON and OFF configurations. In the ON condition, the researcher is available to the repair agent for every cluster; in the OFF condition, it is unavailable. The test case and all non-researcher components— including the repair loop and graph tools—are held fixed. Thus, the paired diference measures the efect of making the researcher available. Comparing these diferences across predicate mixtures and datasets characterizes variation in researcher utility.

The predicate mixtures provide a coarse way to vary the expected information requirement of retrieval utility. Evidential mixtures are likely to require external facts, whereas representational mixtures require mapping domain information onto 223P graph structures. However, the predicate class does not fully determine retrieval utility, which varies by individual predicate and deployment. For example, strong naming conventions in the graph may allow the repair agent to resolve an evidential violation through graph exploration; extracting implicit structure from point-label naming conventions is well established in building systems [16, 28, 32, 59].

Moreover, even when external evidence is needed, retrieval helps only if the deployment’s corpus contains the relevant information and the researcher can extract it. Retrieval utilities may therefore difer among predicates in the same class and across deployments.

To quantify this heterogeneity, we compute the paired diference for each test case. This is defined as the subtraction between conditions. For $F _ { 1 }$ score, this is $\Delta F _ { 1 } = F _ { 1 , \mathrm { O N } } - F _ { 1 , \mathrm { O F F } }$ . We report the median paired diference along with a 95% bootstrap confidence interval over the test cases (Fig. 5). Specifically, we resample 5,000 times with replacement from the paired diferences pooled across �, with each resample the same size as the original set of paired cases. Token counts include every language model call: graph exploration, research, summarization, and repair synthesis.

To account for the eight injection-budget levels, we test whether the paired ON–OFF $F _ { 1 }$ diferences tend to difer from zero, accounting for � when calculating the �-value for each dataset and predicate mixture (a �-stratified Wilcoxon signed-rank test [61]).

Before comparing $F _ { 1 }$ scores, we note that across the 960 ON/OFF runs (80 test cases × 3 mixtures × 2 datasets × 2 configurations), only 6 fail to produce a schema-valid graph, a less than 1% failure rate. In the following, we detail three observations regarding retrieval utility (Fig. 5):

Predicate mixtures reveal diferences in retrieval utility. On IBAL, enabling the researcher yields a median $\Delta F _ { 1 }$ of +0.357 for the evidential suite, with 77 of 80 cases improving. The terminological suite has a smaller median gain of +0.043, with both positive and negative case-level diferences. On Pritoni, the median gains are +0.080 (evidential), +0.051 (terminological), and +0.035 (representational). The evidential suites have the largest median gains in both deployments, consistent with their emphasis on relationships commonly documented in engineering artifacts. This pattern supports using predicates as observable proxies for diferences in retrieval utility. Pritoni’s smaller evidential gain may reflect the context supplied by descriptive graph labels, which can help the repair agent infer physical facts through graph exploration.

The efect of researcher access also depends on the deployment. On IBAL’s representational suite, the median $\Delta F _ { 1 }$ is −0.126, with 67 of 80 cases getting worse. On Pritoni’s representational suite, the median gain is +0.035. Thus, researcher access can help or hurt within the same predicate class. Graph context, corpus coverage, and the researcher’s ability to retrieve evidence ofer possible explanations for this contrast. The sign change motivates learning how predicates relate to retrieval utility within each deployment.

One observed repair trajectory illustrates a possible failure mechanism: a damper (D14) is missing a hasConnectionPoint predicate to its ConnectionPoint node (Damper\_14\_OCP). In the OFF condition, the repair agent uses a keyword search for “Damper $1 4 ^ { \dprime }$ to identify the node Damper\_14\_OCP. However, in the ON condition, the repair agent queries the researcher agent, which returns building descriptions (“the damper’s inlet is a return-air duct and the medium is air”). Following this report, the repair agent explores the graph using keywords “return air duct”, “air”, and “RA”. When these searches fail to locate Damper\_14\_OCP, the repair agent erroneously creates a new ConnectionPoint node to connect to the damper.

Researcher availability has uneven returns. Normalizing median $\Delta F _ { 1 }$ by the additional tokens consumed (Fig. 5) highlights an ineficiency. On IBAL, researcher availability yields an $F _ { 1 }$ improvement of +0.36 per 100k additional tokens for evidential violations, compared to just +0.02 for terminological and −0.04 for representational violations. Similarly, on Pritoni, the return for representational violations is positive but small, at +0.007 per 100k additional tokens. Although the repair agent can choose whether to call the researcher under ON, it does not reliably determine when retrieval is worthwhile. A global ON policy therefore permits costly retrieval in regimes where its benefit is negligible or negative, whereas a global OFF policy forgoes retrieval where it is valuable. Because these regimes vary across predicates and deployments, researcher availability should be decided selectively.

![](images/44cdbac71e647ac7cebae31cba3f9116ec99e6796bf4ac13e14852500f5e3c7c.jpg)  
Figure 5: Impact of retrieval on repair accuracy (Δ� ) and token consumption across datasets and predicate mixtures. Left: Paired diferences $( \Delta F _ { 1 } = F _ { 1 , \mathrm { O N } } - F _ { 1 , \mathrm { O F F } } )$ per test case, showing medians (⋄) and 95% bootstrap confidence intervals (bars). Reported two-sided �-values: on IBAL, Evidential $( p < 0 . 0 0 1 )$ , Terminological $\left( \mathinner { p \mathopen { \left/ { \vphantom { \left( p \mathrm { 1 6 } \mathstrut \right.} }  \kern - delimiterspace \right)} } 0 . 0 0 1 6  \right.$ , and Representational $( p < 0 . 0 0 1 )$ ; on Pritoni, Evidential $( p < 0 . 0 0 1 )$ Terminological $( p = 0 . 0 3 1 )$ , and Representational $( p = 0 . 0 0 3 6 )$ . Right: Median extra tokens consumed by enabling retrieval (bar plots with 95% CIs). For context, the median token consumption of the OFF condition: For IBAL, Evidential: 356k, Terminological: 245k, Representational: 296k. For Pritoni, Evidential: 355k, Terminological: 379k, Representational: 426k. Normalized token eficiency $( \Delta F _ { 1 } / 1 0 0 \mathbf { k }$ extra tokens): IBAL yields +0.352 (Evidential), +0.020 (Terminological), and −0.036 (Representational), whereas Pritoni yields +0.023 (Evidential), +0.013 (Terminological), and +0.007 (Representational).

## 5.3 From Heterogeneity to Learned Routing

These results motivate learning a deployment-specific routing policy. The predicate classes organize our study of heterogeneity, but their aggregate efects difer across deployments: researcher access helps the representational suite on Pritoni and hurts it on IBAL. We therefore use observed repair outcomes to learn access decisions within each deployment.

Efective training requires a representative set of test cases that cover the predicates the router will encounter at test time. Our earlier experiments intentionally skewed the injected violations to reveal heterogeneity, but training a general router requires broad coverage. We use inverse-frequency weighting across the full predicate set to encourage this coverage (Sec. 6.2); if a deployment’s empirical violation distribution is known, that specific distribution can inform the sampling weights instead. Next, Sec. 6 introduces the sampler used to generate test cases, and Sec. 7 presents the learned routing policy.

## 6 CONTROLLABLE TEST GENERATION

Our experiments require test cases with diferent predicate distributions. The suites in Sec. 5 concentrate violations in the evidential, terminological, or representational predicate classes to expose retrieval heterogeneity. Router training instead requires broad coverage across predicates so that common predicates do not dominate the training data. Therefore, we extend the violation-generation framework of [35] with a sampler that controls which perturbations are selected and composes multiple compatible perturbations into each test case.

Recall from Sec. 3 that a test case is constructed by injecting violations. Each injected violation is represented by a set of graph edits that forces a node to break a constraint; reversing those perturbations provides the ground-truth repair. This section focuses on selecting and composing the perturbations.

![](images/38e6dc91ac591bdc90e54b39b8a37cea2eeddb41363d22037377f6e79f90ef74.jpg)  
Figure 6: A valid instance graph example and its nested schema constraint. The point of ahu\_1 has a description whose length is at least three, so ahu\_1 satisfies the constraint.

![](images/b17a91186dfdb98f59e287f024e28d65e8c9e88b3b6a896ed014d6a290c0c04b.jpg)  
Figure 7: The expansion graph of violating the constraint � in Fig. 6. Arrows represent alternative violation strategies.

## 6.1 Constructing Test Cases

We first illustrate how candidate perturbations are derived from a valid graph and its schema. Fig. 6 shows an AHU node ahu\_1 with a point RAT\_Sensor. The schema requires every AHU to have at least one hasPoint value conforming to NamedPoint; in turn, a NamedPoint must have a description whose length is at least three. The graph satisfies this constraint because RAT\_Sensor has the description "RAT\_01".

Following [35], an expansion graph represents alternative ways ofmaking a selected node violate a constraint. It is a search structure constructed from the schema and instance graph, not a subgraph of the instance graph.

Fig. 7 shows the expansion graph for making ahu\_1 violate the constraint � in Fig. 6. We write $\mathrm { V I O } ( x , \varphi )$ for the task of making node � violate constraint �. One way to accomplish this is to remove the hasPoint edge. Alternatively, we can make RAT\_Sensor fail the NamedPoint requirement, denoted VIO(RAT\_Sensor,NamedPoint), which is rewritten as VIO(RAT\_Sensor, $\exists ^ { \geq 1 }$ description. (len ≥ 3)). This second alternative can in turn be achieved by removing the description edge or making the $^ { * } \mathsf { R A T } _ { - } \theta \mathsf { 1 } ^ { * }$ string violate len ≥ 3, which is done by replacing “RAT\_01” with a string shorter than three characters, such as “RA”. Details of how to construct the expansion graph are in [35].

The traversal of the expansion graph from the root node to a leaf node terminates in a normalized violation expression: the edit operation that injects the requested violation. In this example, the three leaves yield expressions that delete the hasPoint edge, delete the description edge, or replace $" { \sf R A T } _ { - } \theta 1 "$ with a string such as "RA". Choosing any of the leaf nodes will achieve the top level task of making ahu\_1 violate the constraint �. Reversing the selected operation provides the corresponding ground-truth repair.

Conceptually, we define the sampling universe U as the set of distinct normalized violation expressions represented by the leaves of the expansion graph. The coverage result of [35] establishes that U contains every normalized violation expression admitted by the formalism. Given a budget �, our objective is to select a conflictfree subset $S \subseteq { \mathcal { U } }$ with $| S | = K .$ . A single expression may trigger multiple validation results. Two expressions conflict when their operations impose mutually incompatible violations on the same node. For example, a node � cannot simultaneously be perturbed to violate both $x \leq 1$ and $x > 0 .$ . We therefore admit only subsets whose expressions can be applied together.

The expansion graph in Fig. 7 is small enough to show in full, but expansion graphs generally have many branches since they enumerate all violation strategies using the instance graph and schema. Our implementation therefore does not materialize the full expansion graph to enumerate U before sampling. Instead, it traverses the expansion graph incrementally, branches on the fly, and emits an expression upon reaching a leaf. In addition, diferent traversal paths may yield the same normalized expression, so the sampler deduplicates and processes this stream as leaves are discovered.

## 6.2 Controlling Predicate Distribution

We treat the incrementally emitted expressions as a data stream and maintain a deduplicated reservoir of � candidate expressions, from which test cases are composed. The reservoir determines the available candidates; the final predicate distribution also depends on which candidates can be composed together.

We influence candidate selection through a positive weight $\omega _ { p }$ assigned to each predicate �. We say an expression targets a predicate when that predicate is part of the edit to induce a violation. In Fig. 7, removing the hasPoint edge targets hasPoint, whereas removing the description or shortening its value targets description. Although all three expressions make ahu\_1 violate the same top-level constraint, they do so through diferent predicates. An expression can also target multiple predicates. The weight �(�) of an expression combines the weights of the predicates it targets, the geometric mean when there are several. Increasing $\omega _ { p }$ therefore makes expressions targeting � more likely to be retained in the reservoir. For the suites in Sec. 5, we assign larger weights to predicates in the targeted class. For router training, we estimate predicate frequencies and use inverse-frequency weights to encourage broad coverage of targeted predicates in the reservoir.

The stream of normalized violation expressions has unknown length and is produced incrementally. We thus use the weighted reservoir sampling method of Efraimidis and Spirakis [21], which converts each expression’s weight into a sort key. When an expression � is emitted, it receives the key

$$
\ker ( v ) = { \frac { - \ln h ( v ) } { w ( v ) } } , \qquad h ( v ) = \operatorname { h a s h } ( v ) \in ( 0 , 1 ] .\tag{1}
$$

The reservoir retains the � distinct expressions with the smallest keys. The hash assigns each expression a stable pseudorandom value, so repeated occurrences receive the same key and are counted only once. Dividing by �(�) makes a larger-weight expression more likely to have a small key and thus to remain in the reservoir.

At any point, the reservoir represents a weighted sample of the expressions discovered so far. Thus, case construction can begin without waiting for the complete expansion graph to be traversed. If traversal completes, the reservoir is sampled from the full universe U; otherwise, it represents the explored portion of that universe.

## 6.3 Composing Test Cases

The final step composes a test case from the size-� reservoir, whose candidates are selected using the predicate weights $\omega _ { p }$ . Given a budget �, we select up to � expressions, admitting an expression only if its edits do not conflict with those already selected. Specifically, we scan the reservoir and select an expression only if its associated violation requirements are compatible with those of the expressions already selected. We check these requirements pairwise, treating requirements on the same node as conflicting when they demand incompatible violations. Selection stops when � expressions have been chosen or the reservoir is exhausted.

We then apply the selected edits to the original graph to obtain the test case, and record their reversal as the ground-truth repair. Separating sampling from composition also lets us draw many test cases from a single traversal of the expansion graph.

## 7 LEARNING TO ROUTE

The variation in researcher utility across violations and deployments motivates learning when to enable access. We use the testcase generator of Sec. 6 to construct training cases with known reference repairs, and learn a routing policy from the observed repair outcomes.

The router operates within the iterative repair loop (Fig. 2). Immediately after the clustering algorithm groups a set of violations, the router determines the prompt configuration for the repair agent. It makes a binary choice: provide the agent with the researcherenabled prompt, or the researcher-disabled prompt.

The router takes as input a multi-hot vector of predicate names reported for the cluster’s violations (sh:resultPath in SHACL); for nested constraints, we do not traverse to collect predicates from referenced shapes. The predicate grouping labels used to organize the motivating study are not included in this feature vector.

We train two XGBoost models [19]: one predicts the $F _ { 1 }$ score of the cluster’s repair with the researcher enabled, and the other with it disabled. This two-model design is known as a T-learner in the causal inference literature [33]; here, we use it as a predictive routing rule. At test time, the router enables the researcher when the predicted ON score exceeds the predicted OFF score. We choose XGBoost because it (a) supports the sparse, binary features and (b) enables SHAP-based interpretation of the predictions (Sec. 9).

We train a separate router for each dataset, sampling training test cases across the full predicate set using the inverse-frequency weighting from Sec. 6.2. We then run each test case twice: once with the researcher always enabled, and once with it always disabled. For each cluster encountered in a trajectory, we extract its feature vector and use the recorded injected perturbations to identify its reference repair edits and compute its training $F _ { 1 }$ . Additional edits that do not match the reference repair count as false positives. The enabled trajectories supply training examples for the ON model, and the disabled trajectories supply examples for the OFF model.

Although ON and OFF are run on the same test case, their repair trajectories can produce diferent clusters. Each model learns from the clusters encountered under its corresponding policy. We evaluate the resulting router using final test-case repair quality and token consumption (Sec. 8).

## 8 EVALUATION OF ROUTING POLICIES

We compare the T-learner router (T-learner) against four baselines, holding the repair loop and graph tools fixed.

• OFF and ON: The baselines established in Sec. 5, representing always disabled and enabled retrieval, respectively. OFF serves as the reference point for repair accuracy and cost.

• Random: Enables the researcher randomly with a probability set to the T-learner’s provision rate for the same dataset and predicate mixture (Table 1). This gives six enabling probabilities, one for each evaluation suite. The baseline compares learned selection with random access at a matched enabling probability. Because the policies can produce diferent repair trajectories and cluster counts, they need not enable the same number of clusters or consume the same number of tokens.

• LLM Router: A zero-shot alternative to learning the routing rule. We prompt an LLM with the same features given to T-learner, the predicates involved in the cluster, and ask whether providing the researcher will improve the repair.

We evaluate all five policies on the six evaluation suites detailed in Sec. 5. Their predicate mixtures define the evaluation workloads; the T-learner’s reported provision rates are measured from its decisions on these workloads. The T-learner is trained on a disjoint set of generated test cases sampled across the full predicate set under inverse-frequency weighting, keeping evaluation cases out of router training. We use the evaluation protocol of Sec. 5, including its significance-testing procedure, and report the median paired diference (with a 95% bootstrap confidence interval) in repair quality (Δ� ) and repair-time token consumption (Δtokens) against the OFF baseline for each test case (Fig. 8). The only modification is that the token counts of LLM Router include its own inference cost.

Across all 2,400 runs (80 test cases × 3 mixtures × 2 datasets × 5 policies), only 17 fail to produce a schema-valid graph, a failure rate below 1%; 6 of these are the ON/OFF failures reported in Sec. 5. All five policies produce repairs at a high rate of schema validity, so the comparisons below concern which edits were made.

T-learner improves $F _ { 1 }$ over OFF while using fewer tokens than ON. In all six evaluation suites, the T-learner uses fewer tokens than ON, as shown in Fig. 8, where the T-learner lies consistently to the left of ON. In five of them, the T-learner also improves $F _ { 1 }$ relative to OFF. The comparisons reported in Table 1 indicate higher $F _ { 1 }$ than ON in four of the six suites, with no statistically significant diference detected in the two evidential suites (IBAL 42/23/15, Pritoni 39/27/14).

Selective access on the evidential suites. On the two evidential suites, the T-learner enables the researcher for only 53% of IBAL clusters and 62% of Pritoni clusters. The comparisons detect no statistically significant $F _ { 1 }$ diference from ON.

![](images/180e70608b0651e441d7aa160b5c85c886f930771e9e3471ada0bb03e272fb39.jpg)  
Figure 8: Repair quality–token cost trade-of across datasets and predicate mixtures. Each point shows the change in $F _ { 1 }$ relative to OFF $( \Delta F _ { 1 } = F _ { \mathrm { 1 , p o l i c y } } - F _ { \mathrm { 1 , O F F } } )$ versus the additional token cost. The OFF $F _ { 1 }$ scores and token costs, as reported in Fig. 5, are 0.488, 0.618, 0.639, 356k, 245k, and 296k for IBAL evidential, terminological, and representational mixtures, respectively, and 0.663, 0.627, 0.595, 355k, 379k, and 426k for the corresponding Pritoni mixtures.

The same router is used across all evaluation suites for a dataset. Its provision rate varies with the clusters encountered in each workload; on IBAL, the rates range from 37% to 74% (Table 1). Sec. 9 examines how individual predicates contribute to its predictions.

The LLM Router almost always enables researcher access. With the evaluated prompt and model, the LLM router enables the researcher on 99–100% of clusters across every suite in Table 1. It therefore provides little selectivity while adding an LLM call per cluster. In Fig. 8, its token consumption is at or slightly above ON in almost all panels.

The T-learner outperforms Random. Random enables researcher access with the same probability as the T-learner for each dataset and predicate mixture. The T-learner achieves higher repair $F _ { 1 }$ in all six evaluation suites (Table 1). These results support using cluster predicate features to select researcher access.

Selective routing mitigates harmful retrieval. The router improves repair $F _ { 1 }$ over OFF in five of six suites. In the remaining IBAL representational suite, where ON degrades repair accuracy, the router lowers its provision rate to 37% and achieves higher $F _ { 1 }$ with fewer tokens than ON (Fig. 8). Selective access therefore limits the downside of harmful retrieval, although the policy remains less accurate and more expensive than OFF.

The current routing rule enables the researcher whenever $\hat { F } _ { 1 , \mathrm { O N } } > \hat { F } _ { 1 , \mathrm { O F F } }$ , regardless of the magnitude of the predicted gain; for example, a gain of $1 0 ^ { - 3 }$ would yield the same access decision as a gain of 0.4. Furthermore, the rule considers only repair accuracy, ignoring the additional token cost. Future work will consider extensions that require a minimum predicted gain to justify researcher access or incorporate token cost and prediction uncertainty into the decision.

Clustering also determines the granularity of researcher access. Larger clusters may combine violations with diferent evidence needs under a single access decision. Future work will examine how clustering and routing interact, including whether adapting cluster boundaries to predicted retrieval benefit can improve repair accuracy and token eficiency.

<table><tr><td rowspan="2"></td><td rowspan="2">Dataset Mixture</td><td colspan="3">T-learner vs. (W/L/T)</td><td colspan="2">Provision Rate (%)</td></tr><tr><td>ON</td><td>Random</td><td>LLM</td><td>LLM</td><td>T-learner</td></tr><tr><td>IBAL</td><td>Evid.</td><td> $4 2 / 2 3 / 1 5 ^ { \mathrm { n s } }$ </td><td> $6 6 / 5 / 9 ^ { ^ { \star \star \star } }$ </td><td> $4 6 / 1 5 / { 1 9 } ^ { \star \star }$ </td><td>99</td><td>53</td></tr><tr><td>IBAL</td><td>Termin.</td><td> $5 7 / 1 7 / 6 ^ { ^ { \star \star \star } }$ </td><td> $5 3 / 2 1 / 6 ^ { \star \star }$ </td><td> $4 8 / 2 1 / { 1 1 } ^ { \star \star }$ </td><td>99</td><td>74</td></tr><tr><td>IBAL</td><td>Repr.</td><td> $6 1 / 1 2 / 7 ^ { \star \star }$ </td><td> $4 4 / 2 7 / 9 ^ { ^ { \star } }$ </td><td> $5 9 / 1 5 / 6 ^ { ^ { \star \star \star } }$ </td><td>99</td><td>37</td></tr><tr><td>Pritoni</td><td>Evid.</td><td> $3 9 / 2 7 / 1 4 ^ { \mathrm { n s } }$ </td><td> $4 6 / 2 5 / 9 ^ { ^ { \star \star } }$ </td><td> $4 3 / 2 0 / 1 7 ^ { \star \star }$ </td><td>99</td><td>62</td></tr><tr><td>Pritoni</td><td>Termin.</td><td> $5 1 / 2 1 / 8 ^ { ^ { \star \star \star } }$ </td><td> $5 6 / 1 6 / 8 ^ { ^ { \star \star \star } }$ </td><td> $5 6 / 1 5 / 9 ^ { ^ { \star \star \star } }$ </td><td>100</td><td>57</td></tr><tr><td>Pritoni</td><td>Repr.</td><td> $6 1 / 1 7 / 2 ^ { \star \star }$  ★</td><td> $6 8 / 9 / 3 ^ { ^ { \star \star } }$ </td><td> $6 2 / 1 4 / 4 ^ { ^ { \star \star } }$ </td><td>99</td><td>52</td></tr></table>

Table 1: Repair quality and provision rate across datasets and predicate mixtures. W/L/T counts test cases in which the T-learner attains higher / lower / equal $F _ { 1 }$ than the competitor; reported significance markers are $^ { * * * } p < . 0 0 1 , ^ { * * } p < . 0 1 , ^ { * } p < . 0 5$ and ns = not significant.

## 9 INTERPRETING THE T-LEARNER

Sec. 8 shows that the T-learner selectively enables researcher access across the evaluation suites. Fig. 9 examines how individual predicates contribute to the router’s predicted gain, $\hat { F } _ { 1 , \mathrm { O N } } - \hat { F } _ { 1 , \mathrm { O F F } } ,$ using SHAP [37]. We compute contributions separately for the ON and OFF models and subtract them; positive contributions raise the predicted gain relative to its base value, while negative contributions lower it. Each bar shows the mean contribution across clusters from that deployment’s training mixtures in which the predicate is present. Whiskers show 95% percentile bootstrap intervals for that mean, based on 5,000 resamples. The router enables researcher access when the total predicted gain is positive.

The learned contributions difer among predicates, including those within the same class. For example, on IBAL, observes has a positive mean contribution to predicted gain, whereas hasMedium has a negative one, although both belong to the evidential class. The same predicate can also contribute diferently across deployments: the mean contribution of hasMedium is positive on Pritoni. These patterns illustrate the finer distinctions learned from repair outcomes within each deployment.

![](images/5e4f3a54a17ddb9d7ae40194e77c7d752f2d12fc6daee07bc3f695846998c0bf.jpg)  
Figure 9: Predicate-level SHAP values for the T-learner’s predicted retrieval gain. The plot’s “helps” and “hurts” labels refer to contributions that raise or lower predicted retrieval gain relative to the base value. The reported base values are +0.1602 for IBAL and +0.0527 for Pritoni.

SHAP decomposes each cluster’s predicted gain into a base value plus the contributions of its predicate features. On IBAL, this base value is +0.1602, roughly three times that on Pritoni (+0.0527), indicating a higher baseline predicted benefit from researcher access. The predicate contributions must therefore ofset a larger positive baseline before the IBAL router disables access. This helps interpret how access can remain enabled even when individual predicates, such as hasConnectionPoint, contribute negatively to predicted retrieval gain. Each access decision depends on the combined con tributions for an individual cluster, while Fig. 9 summarizes mean contributions across clusters.

Overall, these patterns help interpret how the router selects researcher access using individual predicate features. The evaluation suites measure the repair quality and token consumption of ap plying the same learned policy across diferent predicate mixtures within each deployment.

## 10 LIMITATIONS & DISCUSSION

Our evaluation is limited to two building-automation deployments using the 223P schema. This scope reflects a benchmark constraint: as discussed in Sec. 3.2, few datasets provide a schema, a valid instance graph, and an external corpus simultaneously. Widely used knowledge bases also risk confounding retrieval evaluation through contamination of the LLM’s parametric memory. The contrast between Pritoni and IBAL demonstrates that retrieval utility varies even within one domain and schema, but it does not establish gen eralization to other domains. Evaluating such settings will require additional benchmarks with the same three aligned artifacts.

Training the router requires a trusted reference graph and an aligned document corpus for the deployment. Its applicability therefore depends on the availability and coverage of that reference. An existing graph may contain a trusted, schema-valid subgraph from which our multi-violation generation framework (Sec. 6) can construct training cases with known reference repairs. Running these cases with researcher access enabled and disabled supplies outcomes for training the router. Repeating this procedure could sup port periodic retraining as the graph, corpus, or retrieval pipeline changes. Preparing the corpus index, generating and labeling cases, and training the router also incur costs beyond repair-time token consumption.

Furthermore, our analysis is also limited in its ability to attribute retrieval failures. We treat the external corpus and the researcher as a unified retrieval pipeline because the repair agent observes only the researcher’s final report. When this pipeline fails to supply a required fact, we cannot distinguish among the following scenarios:

(1) The fact is explicitly present in the external corpus, but the researcher fails to retrieve or extract it.

(2) The fact is not explicitly stated but can be inferred by combining evidence across documents, and the researcher fails to make this inference.

(3) Neither the fact nor suficient evidence from which to infer it is present in the external corpus.

These scenarios are observationally equivalent to the repair agent because each results in a report that does not provide the required information. Disentangling corpus coverage from researcher eficacy would require ground-truth annotations. Constructing this ground truth is beyond the scope of this paper.

Finally, the T-learner router is trained ofline and held fixed so it does not adapt to deployment drift such as new document additions to the corpus or changes to the retrieval pipeline. Online adaptation with feedback using methods such as contextual bandits [34] remains future work.

## 11 CONCLUSION

This paper studies when an LLM-based graph repair agent should be given access to an external retrieval system. Our experiments show that the efect of researcher access difers across repairs and corpora: it can improve repair accuracy, increase token consumption without improving accuracy, or even lead to less accurate repairs. This variation motivates learning a deployment-specific access policy.

We therefore learn a deployment-specific router from automatically generated repair cases. Our test-generation framework composes multiple compatible constraint violations while retaining programmatic ground-truth repairs and controlling predicate distribution. Using outcomes collected with the researcher enabled and disabled, the router learns predicate-level diferences in retrieval utility and selectively provisions researcher access for each violation cluster. Across six evaluation suites on two building-automation deployments, the router uses fewer repair-time tokens than always enabling access in all six suites. It improves repair $F _ { 1 }$ over always enabling access in four suites and over always disabling access in five. These results demonstrate that learned, selective access to external retrieval can improve repair accuracy while reducing token consumption.

## REFERENCES

[1] 2025. Resource Description Framework (RDF). https://www.w3.org/RDF/. Accessed: 2025-05-05.

[2] 2025. Shapes Constraint Language (SHACL). https://www.w3.org/TR/shacl/. Accessed: 2025-05-05.

[3] 2025. ShEx - Shape Expressions. https://shex.io/. Accessed: 2025-05-05.

[4] 2025. Simple Protocol and RDF Query Language, (SPARQL). https://www.w3. org/TR/sparql11-query/. Accessed: 2025-05-05.

[5] Shqiponja Ahmetaj, Iovka Boneva, Jan Hidders, Katja Hose, Maxime Jakubowski, Jose Emilio Labra Gayo, Wim Martens, Fabio Mogavero, Filip Murlak, Cem Okulmus, et al. 2025. Common foundations for SHACL, ShEx, and PG-Schema. In Proceedings ofthe ACM on Web Conference 2025. 8–21.

[6] Shqiponja Ahmetaj, Robert David, Axel Polleres, and Mantas Šimkus. 2022. Repairing SHACL constraint violations using answer set programming. In International Semantic Web Conference. Springer, 375–391.

[7] Renzo Angles, Angela Bonifati, Stefania Dumbrava, George Fletcher, Alastair Green, Jan Hidders, Bei Li, Leonid Libkin, Victor Marsault, Wim Martens, et al. 2023. Pg-schema: Schemas for property graphs. Proceedings of the ACM on Management ofData 1, 2 (2023), 1–25.

[8] Marcelo Arenas, Leopoldo Bertossi, and Jan Chomicki. 1999. Consistent query answers in inconsistent databases. In Proceedings ofthe eighteenth ACM SIGMOD-SIGACT-SIGART symposium on Principles of database systems. 68–79.

[9] Hiba Arnaout, Trung-Kien Tran, Daria Stepanova, Mohamed Hassan Gad-Elrab, Simon Razniewski, and Gerhard Weikum. 2022. Utilizing language model probes for knowledge graph repair. In Wiki Workshop 2022.

[10] Patricia C Arocena, Boris Glavic, Giansalvatore Mecca, Renée J Miller, Paolo Papotti, and Donatello Santoro. 2015. Messing up with BART: error generation for evaluating data-cleaning algorithms. Proceedings ofthe VLDB Endowment 9, 2 (2015), 36–47.

[11] Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and Hannaneh Hajishirzi. 2024. Self-rag: Learning to retrieve, generate, and critique through self-reflection. In International conference on learning representations, Vol. 2024. 9112–9141.

[12] Sören Auer, Christian Bizer, Georgi Kobilarov, Jens Lehmann, Richard Cyga niak, and Zachary Ives. 2007. Dbpedia: A nucleus for a web of open data. In international semantic web conference. Springer, 722–735.

[13] Bharathan Balaji, Arka Bhattacharya, Gabriel Fierro, Jingkun Gao, Joshua Gluck, Dezhi Hong, Aslak Johansen, Jason Koh, Joern Ploennigs, Yuvraj Agarwal, et al. 2016. Brick: Towards a unified metadata schema for buildings. In Proceedings of the 3rd ACM International Conference on Systems for Energy-Eficient Built Environments. 41–50.

[14] Simone Balloccu, Patrícia Schmidtová, Mateusz Lango, and Ondřej Dušek. 2024. Leak, cheat, repeat: Data contamination and evaluation malpractices in closed source LLMs. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers). 67–93.

[15] Harry Bergmann, Cory Mosiman, Avijit Saha, Selam Haile, William Livingood, Steven Bushby, Gabe Fierro, Joel Bender, Michael Poplawski, Jessica Grander son, and Marco Pritoni. 2020. Semantic Interoperability to Enable Smart, Grid-Interactive Eficient Buildings. Technical Report. Lawrence Berkeley Nationa Laboratory. https://www.osti.gov/biblio/1735554

[16] Arka A Bhattacharya, Dezhi Hong, David Culler, Jorge Ortiz, Kamin Whitehouse, and Eugene Wu. 2015. Automated metadata construction to support portable building applications. In Proceedings of the 2nd ACM International Conference on Embedded Systems for Energy-Eficient Built Environments. 3–12.

[17] Philip Bohannon, Wenfei Fan, Michael Flaster, and Rajeev Rastogi. 2005. A cost-based model and efective heuristic for repairing constraints by value modification. In Proceedings ofthe 2005 ACM SIGMOD international conference on Management ofdata. 143–154.

[18] Nicholas Car. 2024. pySHACL. https://github.com/RDFLib/pySHACL. https: //doi.org/10.5281/zenodo.4750840 For all versions/latest version.

[19] Tianqi Chen and Carlos Guestrin. 2016. Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining. 785–794.

[20] Xu Chu, Ihab F Ilyas, and Paolo Papotti. 2013. Holistic data cleaning: Putting viola tions into context. In 2013 IEEE 29th International Conference on Data Engineering (ICDE). IEEE, 458–469.

[21] Pavlos S Efraimidis and Paul G Spirakis. 2006. Weighted random sampling with a reservoir. Information processing letters 97, 5 (2006), 181–185.

[22] Wenfei Fan, Jianzhong Li, Shuai Ma, Nan Tang, and Wenyuan Yu. 2010. Towards certain fixes with editing rules and master data. Proceedings ofthe VLDB Endowment (PVLDB) 3, 1 (2010), 173–184.

[23] Wenfei Fan, Ping Lu, Chao Tian, and Jingren Zhou. 2019. Deducing certain fixes to graphs. Proceedings of the VLDB Endowment 12, 7 (2019), 752–765.

[24] Gabe Fierro, Marco Pritoni, Moustafa Abdelbaky, Daniel Lengyel, John Leyden, Anand Prakash, Pranav Gupta, Paul Raftery, Therese Pefer, Greg Thomson, and David E. Culler. 2019. Mortar: An Open Testbed for Portable Building Analytics. ACM Trans. Sen. Netw. 16, 1, Article 7 (Dec. 2019), 31 pages. https: //doi.org/10.1145/3366375

[25] Jose Emilio Labra Gayo, Holger Knublauch, and Dimitris Kontokostas. 2025. Data Shapes Test Suite. https://w3c.github.io/data-shapes/data-shapes-test-suite/. Accessed: 2025-04-23.

[26] Karl Hammar, Erik Oskar Wallin, Per Karlberg, and David Halleberg. [n.d.]. The RealEstateCore Ontology. 16.

[27] Shahenda Hatem, Ghada Khoriba, Mohamed H. Gad-Elrab, and Mohamed ElHelw. 2024. Up To Date: Automatic Updating Knowledge Graphs Using LLMs. Procedia Computer Science 244 (2024), 327–334. https://doi.org/10.1016/j.procs.2024.10.206

[28] Dezhi Hong, Hongning Wang, Jorge Ortiz, and Kamin Whitehouse. 2015. The Building Adapter: Towards Quickly Applying Building Analytics at Scale. In Proceedings ofthe 2nd ACM International Conference on Embedded Systems for Energy-Eficient Built Environments. 123–132. https://doi.org/10.1145/2821650. 2821657

[29] Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Steven Yau, Zijuan Lin, Liyang Zhou, et al. 2024. MetaGPT: Meta programming for a multi-agent collaborative framework. In International Conference on Learning Representations, Vol. 2024. 23247–23275.

[30] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park. 2024. Adaptive-rag: Learning to adapt retrieval-augmented large language models through question complexity. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 7036–7050.

[31] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active retrieval augmented generation. In Proceedings ofthe 2023 conference on empirical methods in natural language processing. 7969–7992.

[32] Jason Koh, Bharathan Balaji, Dhiman Sengupta, Julian McAuley, Rajesh Gupta, and Yuvraj Agarwal. 2018. Scrabble: Transferrable Semi-Automated Semantic Metadata Normalization Using Intermediate Representation. In Proceedings of the 5th Conference on Systems for Built Environments. 11–20. https://doi.org/10. 1145/3276774.3276795

[33] Sören R Künzel, Jasjeet S Sekhon, Peter J Bickel, and Bin Yu. 2019. Metalearners for estimating heterogeneous treatment efects using machine learning. Proceedings of the national academy of sciences 116, 10 (2019), 4156–4165.

[34] Lihong Li, Wei Chu, John Langford, and Robert E Schapire. 2010. A contextualbandit approach to personalized news article recommendation. In Proceedings of the 19th international conference on World wide web. 661–670.

[35] Tung-Wei Lin, Gabe Fierro, Han Li, Tianzhen Hong, Pierluigi Nuzzo, and Alberto Sangiovanni-Vinentelli. 2025. Systematic Evaluation of Knowledge Graph Repair with Large Language Models. arXiv preprint arXiv:2507.22419 (2025).

[36] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions of the association for computational linguistics 12 (2024), 157–173.

[37] Scott M Lundberg and Su-In Lee. 2017. A unified approach to interpreting model predictions. Advances in neural information processing systems 30 (2017).

[38] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. When not to trust language models: Investigating efectiveness of parametric and non-parametric memories. In Proceedings ofthe 61st annual meeting ofthe association for computational linguistics (volume 1: Long papers). 9802–9822.

[39] Igor Melnyk, Pierre Dognin, and Payel Das. 2022. Knowledge graph generation from text. arXiv preprint arXiv:2211.10511 (2022).

[40] Nandana Mihindukulasooriya, Sanju Tiwari, Carlos F Enguix, and Kusum Lata. 2023. Text2kgbench: A benchmark for ontology-driven knowledge graph generation from text. In International Semantic Web Conference. Springer, 247–265.

[41] Avanika Narayan, Ines Chami, Laurel Orr, Simran Arora, and Christopher Ré. 2022. Can foundation models wrangle your data? arXiv preprint arXiv:2205.09911 (2022).

[42] Axel-Cyrille Ngonga Ngomo, Mohamed Ahmed Sherif, Kleanthi Georgala, Mofeed Mohamed Hassan, Kevin Dreßler, Klaus Lyko, Daniel Obraczka, and Tommaso Soru. 2021. LIMES: A framework for link discovery on the semantic web. KI-Künstliche Intelligenz 35, 3 (2021), 413–423.

[43] Heiko Paulheim. 2016. Knowledge graph refinement: A survey of approaches and evaluation methods. Semantic web 8, 3 (2016), 489–508.

[44] Thomas Pellissier Tanon, Camille Bourgaux, and Fabian Suchanek. 2019. Learning how to correct a knowledge base from the edit history. In The World Wide Web Conference. 1465–1475.

[45] Thomas Pellissier Tanon and Fabian Suchanek. 2021. Neural knowledge base repairs. In The Semantic Web: 18th International Conference, ESWC 2021, Virtual Event, June 6–10, 2021, Proceedings 18. Springer, 287–303.

[46] Amanda J Pertzborn, Daniel A Veronica, and Glen A Glaeser. 2024. Intelligent Building Agents Laboratory. (2024).

[47] Marco Pritoni, Drew Paine, Gabriel Fierro, Cory Mosiman, Michael Poplawski, Avijit Saha, Joel Bender, and Jessica Granderson. 2021. Metadata schemas and on tologies for building energy applications: A critical review and use case analysis. Energies 14, 7 (2021), 2024.

[48] Pritoni, Marco; Wetter, Michael; Paul, Lazlo; Prakash, Anand; Huang, Weiping; Bushby, Steven; Delgoshaei, Parastoo; Poplawski, Michael; Saha, Avijit; Fierro, Gabe; Steen, Matt; Bender, Joel; Ehrlich, Paul. 2024. Digital and Interoperable: The future of building automation is on the horizon. What’s in it for me? Technical Report. Lawrence Berkeley National Laboratory. https://doi.org/10.20357/B76W3X

[49] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. 2024. Toolllm: Facilitating large language models to master 16000+ real-world apis. In International Conference on Learning Representations, Vol. 2024. 9695–9717.

[50] Mads Holten Rasmussen, Maxime Lefrançois, Georg Ferdinand Schneider, and Pieter Pauwels. 2020. BOT: The Building Topology Ontology of the W3C Linked Building Data Group. Semantic Web 12, 1 (2020), 143–161. https://doi.org/10. 3233/SW-200385

[51] Theodoros Rekatsinas, Xu Chu, Ihab F Ilyas, and Christopher Ré. 2017. Holoclean: Holistic data repairs with probabilistic inference. arXiv preprint arXiv:1702.00820 (2017).

[52] Oscar Sainz, Jon Campos, Iker García-Ferrero, Julen Etxaniz, Oier Lopez de Lacalle, and Eneko Agirre. 2023. NLP evaluation in trouble: On the need to measure LLM data contamination for each benchmark. In Findings ofthe Association for Computational Linguistics: EMNLP 2023. 10776–10787.

[53] Larissa Capobianco Shimomura, George Fletcher, and Nikolay Yakovets. 2020. GGDs: Graph Generating Dependencies. In Proceedings ofthe 29th ACM International Conference on Information and Knowledge Management. Association for Computing Machinery, 2217–2220. https://doi.org/10.1145/3340531.3412149

[54] Fabian M. Suchanek, Mehwish Alam, Thomas Bonald, Lihu Chen, Pierre-Henri Paris, and Jules Soria. 2024. YAGO 4.5: A Large and Clean Knowledge Base with a Rich Taxonomy. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, Washington DC USA, 131–140. https://doi.org/10.1145/3626772.3657876

[55] Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Cărbune, Michelle Casbon,

et al. 2026. Gemma 4 technical report. arXiv preprint arXiv:2607.02770 (2026).

[56] Hrishikesh Terdalkar, Angela Bonifati, and Andrea Mauri. 2025. Graph Repairs with Large Language Models: An Empirical Study. In Proceedings ofthe 8th Joint Workshop on Graph Data Management Experiences & Systems (GRADES) and Network Data Analytics (NDA). 1–10.

[57] Denny Vrandečić and Markus Krötzsch. 2014. Wikidata: A Free Collaborative Knowledgebase. Commun. ACM 57, 10 (Sept. 2014), 78–85. https://doi.org/10. 1145/2629489

[58] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291 (2023).

[59] David Waterworth, Subbu Sethuvenkatraman, and Quan Z. Sheng. 2021. Advancing Smart Building Readiness: Automated Metadata Extraction Using Neural Language Processing Methods. Advances in Applied Energy 3 (2021), 100041. https://doi.org/10.1016/j.adapen.2021.100041

[60] Yanbin Wei, Qiushi Huang, James Tin Yau Kwok, and Yu Zhang. 2026. KICG-PTv2: Large Language Model with Knowledge in Context for Knowledge Graph Completion. IEEE Transactions on Knowledge and Data Engineering 38, 7 (2026), 4464–4476. https://doi.org/10.1109/TKDE.2026.3674507

[61] Robert F Woolson. 2007. Wilcoxon signed-rank test. Wiley encyclopedia ofclinical trials (2007), 1–3.

[62] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. 2024. Autogen: Enabling next-gen LLM applications via multi-agent conversations. In First conference on language modeling.

[63] Tong Zhou, Yubo Chen, Kang Liu, and Jun Zhao. 2024. CogMG: Collaborative Augmentation Between Large Language Model and Knowledge Graph. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 3: System Demonstrations). Association for Computational Linguistics, Bangkok, Thailand, 365–373. https://doi.org/10.18653/v1/2024.acl-demos.35