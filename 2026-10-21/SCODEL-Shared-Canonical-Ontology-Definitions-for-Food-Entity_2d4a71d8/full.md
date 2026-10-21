# SCODEL: Shared Canonical Ontology Definitions for Food Entity Linking and Ontology-Grounded Recipe Knowledge Graphs

Nuhu Ibrahim, Rishi Ravikumar, Ayuba Dauda, Fuqi Xu,

Robert Stevens, Riza Batista-Navarro

Department of Computer Science, The University of Manchester, United Kingdom Correspondence: nuhu.ibrahim@manchester.ac.uk

## Abstract

Identifying recipe entities and linking them to ontology concepts is fundamental for semantic recipe understanding. However, food mentions in recipes are often ambiguous, contextdependent, and lexically diverse, making ontology entity linking (EL) challenging. We propose SCODEL, a streamlined, model-agnostic approach based on a central idea: contextual food mentions and ontology concepts that refer to the same entity should learn the same canonical ontology-oriented definition. SCODEL uses these shared definitions as a semantic normalisation layer, providing complementary semantic information to existing EL models without requiring a new linking architecture. We further introduce ReciLinkD, the first benchmark for contextual food ontology entity linking in recipes. Experiments show that SCODEL consistently improves three modular EL models (ReFinED, GSM-EL, and BLINK): SCODEL×BLINK achieves the highest Hit@1 on seen concepts (88.33%), while SCODEL×GSM-EL achieves the highest Hit@1 on unseen concepts (83.15%), outperforming the strongest large language model (LLM) based EL baseline in each setting. Experiments on a biomedical ontology benchmark further demonstrate that SCODEL generalises beyond the food domain. Finally, we use SCODEL to ground food mentions at scale and construct an ontology-grounded recipe knowledge graph, demonstrating its utility for ontology-aware recipe retrieval and corpus level food analysis.

## 1 Introduction

Entity linking (EL) is a fundamental task in natural language processing (NLP) that grounds textual mentions to canonical concepts in structured knowledge sources such as ontologies and knowledge graphs. By linking surface forms to structured concepts, EL enriches textual data with formal semantics, enabling interoperability, semantic reasoning, and knowledge integration across heterogeneous datasets (Rao et al., 2012; Shen et al., 2014). Consequently, EL has become an essential component of biomedical and scientific NLP, supporting downstream applications including information extraction, knowledge graph construction, and semantic search.

Despite its importance, ontology entity linking remains relatively underexplored in the food domain. Food mentions may differ substantially from ontology labels despite referring to the same food concept (e.g., garbanzo beans vs. chickpea), require recipe context to resolve underspecified mentions (e.g., stock referring to chicken stock or vegetable stock), or occur at different levels of specificity in the ontology hierarchy (e.g., chilli pepper vs. jalapeno pepper). Existing approaches are typically limited by small annotated datasets, restricted ontology coverage, or memorisation-based methods that struggle to generalise to previously unseen concepts (Pappu et al., 2017; Chong and Lim, 2018; Popovski et al., 2019; Feher et al., 2023; Gjorgjevikj et al., 2025).

At the same time, complementary resources have established a rich ecosystem for computational food research. FoodOn (Dooley et al., 2018) provides a structured ontology of food concepts, while Recipe1M (Marın et al., 2021) and RecipeNLG (Bien et al.´ , 2020) provide large-scale recipe corpora. ReciFine (Ibrahim et al., 2026a) further provides fine-grained food entity annotations over recipe text. However, the absence of robust contextual entity linking prevents effective integration between these resources, limiting semantic recipe understanding and downstream analyses such as nutritional reasoning, dietary inference, ingredient composition analysis, and food analytics through large-scale recipe knowledge graph construction.

Recent advances in entity linking have improved zero-shot generalisation through dense representations and pretrained language models (Moreno et al., 2017; Hou et al., 2020; Parravicini et al., 2019; Logeswaran et al., 2019; Yao et al., 2020; Wu et al., 2020; Ayoola et al., 2022). Nevertheless, these approaches primarily operate on lexical mentions or contextual embeddings, leaving a persistent semantic gap between informal recipe language and structured ontology concepts. This challenge is especially pronounced in recipes, where ingredient mentions vary widely and depend heavily on cooking context.

To address these challenges, we propose SCODEL, a method that learns shared canonical ontology-oriented definitions for both contextual recipe food mentions and ontology concepts, projecting them into a shared semantic definition space before downstream entity linking. Rather than replacing existing entity linking systems, SCODEL serves as a model-agnostic semantic normalisation layer that can be integrated with modular entity linking methods to improve supervised and zeroshot linking performance. We further introduce ReciLinkD, the first benchmark for contextual recipe ontology entity linking, covering food entities as they appear in recipes. Together, SCODEL and ReciLinkD enable recipe-level ontology grounding, facilitating richer semantic understanding and analysis of large-scale recipe collections. All code, datasets, and trained model artefacts associated with this work are released publicly under CC BY-NC 4.0<sup>1</sup>.

Our contributions are summarised as follows:

• We propose SCODEL, a streamlined, modelagnostic approach that learns shared canonical ontology-oriented definitions as a semantic normalisation layer for ontology entity linking.

• We introduce ReciLinkD, the first contextual recipe ontology entity-linking benchmark, with 3,979 manually annotated pairs, expanded to 15,559 manually verified pairs using SCODEL.

• We demonstrate that SCODEL consistently improves diverse modular entity linking frameworks across both supervised and zeroshot settings.

• We provide extensive ablation studies analysing SCODEL’s key design choices, demonstrate its generalisability beyond food ontology on the GSCplus dataset with the Human Phenotype Ontology (Luo et al., 2021), and showcase the practical applicability of food entity linking through a downstream recipe knowledge graph construction application.

## 2 Related Work

## 2.1 Entity Linking

Early EL approaches relied on lexical matching and handcrafted features (Rao et al., 2012; Shen et al., 2014), while more recent methods employ dense neural representations, graph reasoning, and pretrained language models to improve semantic matching and candidate ranking (Parravicini et al., 2019; Hou et al., 2020). Modern EL frameworks, including ReFinED (Ayoola et al., 2022), BLINK (Laskar et al., 2022), and GSM-EL (Cheng et al., 2024), have demonstrated strong performance through scalable retrieval and ranking architectures, enabling effective supervised and zeroshot entity linking.

LLMs have recently been incorporated into EL pipelines to further improve semantic understanding. LLMAEL (Xin et al., 2025) augments mention representations by prompting an LLM to generate additional contextual information before candidate ranking, while LLM4EL (Xu et al., 2026) generates semantic descriptions for both food mentions and ontology concepts to facilitate ontologyaware matching in zero-shot settings. Similarly, Vollmers et al. (2025) use LLM-generated contextual augmentation to improve entity disambiguation. SCODEL is most closely related to LLM4EL, as both approaches generate definitions for the food mention and ontology concept sides. However, LLM4EL generates food mention and ontology concept definitions separately through zero-shot prompting, without learning a common definition target across the two views, and performs entity linking through embedding-based semantic similarity between the generated definitions. In contrast, SCODEL learns shared canonical definition generation from aligned food mention and ontology concept pairs: the contextual food mention and the metadata of its corresponding ontology concept are independently learned to generate a similar canonical ontology-oriented definition. Thus, the distinction is not the use of generated definitions itself, but the explicit learning of a shared canonical target that semantically aligns the two views before entity linking, as demonstrated in Appendix B.2.

Unlike LLM4EL’s direct similarity-based linking, SCODEL integrates the learned canonicalisation with diverse downstream EL models. Furthermore, while LLM4EL was developed for biomedical entity linking, SCODEL extends this definition-based approach to the challenging domain of food ontology entity linking.

## 2.2 Recipe and Food Entity Linking

Compared with biomedical and general-domain EL, food ontology entity linking remains relatively underexplored. Existing work has focused on ontology alignment, food concept mapping, or structured food knowledge extraction (Pappu et al., 2017; Chong and Lim, 2018; Popovski et al., 2019; Ibrahim et al., 2026b). More recent studies have explored food EL using small annotated datasets or LLM-based approaches (Feher et al., 2023; Gjorgjevikj et al., 2025). However, these methods are generally limited to conventional food names, narrow ontology coverage, or memorisation-based linking, and rarely consider context-dependent recipe food mentions or generalisation to previously unseen ontology concepts.

## 3 Food EL Task Formulation

Given a recipe food mention m and its surrounding context $x _ { m } .$ , where $x _ { m }$ denotes the recipe context in which m occurs, the entity linking task is to identify the corresponding concept in a structured food ontology. Let $\mathcal { C } _ { \mathrm { F o o d } }$ denote the set of food concepts in the ontology. An entity linking model assigns a compatibility score

$$
s ( c \mid m , x _ { m } )
$$

to each candidate concept $c \in \mathcal { C } _ { \mathrm { F o o d } }$ . The predicted concept is

$$
\hat { c } = \underset { c \in \mathcal { C } _ { \mathrm { F o o d } } } { \arg \operatorname* { m a x } } s ( c \mid m , x _ { m } ) .
$$

Zero-shot Entity Linking Let $\mathcal { C } _ { \mathrm { F o o d } } ^ { \mathrm { t r a i n } }$ denote the FoodOn concepts represented in the training data. We define a test instance as zero-shot when its gold ontology concept c is completely held out from training. Specifically, neither the food mentions and ontology concepts associated with c nor their canonical definitions are used to train the SCODEL definition generator or the downstream entity linking model:

$$
c \notin { \mathcal { C } } _ { \mathrm { F o o d } } ^ { \mathrm { t r a i n } } .
$$

At inference, the held-out ontology concept and its metadata remain available in the candidate ontology for matching. Thus, zero-shot evaluation assesses whether SCODEL and the downstream entity linker can generalise to unseen ontology concepts without having received concept-specific supervision during either shared canonical definition learning or downstream entity linker training.

## 4 Methodology

We propose SCODEL, a semantic enhancement method for ontology entity linking. Rather than relying only on the original food mentions and ontology concepts, SCODEL augments both representations with ontology-oriented definitions generated by a language model trained to produce shared canonical definitions. These definitions provide complementary semantic signal that reduces the gap between informal recipe language and structured ontology descriptions while remaining fully compatible with existing entity linking architectures.

As illustrated in Figure 1, SCODEL consists of three stages: shared canonical definition learning, canonical definition generation, and entity linking integration.

## 4.1 Shared Canonical Definition Learning

Recipe food mentions and ontology concepts describe the same culinary concepts using fundamentally different linguistic styles. Recipe food mentions are short contextual expressions extracted from recipes, e.g., parm used in the context sprinkle freshly grated parm over the pasta before serving, whereas ontology concepts are represented using structured lexical and taxonomic information, e.g., parmesan cheese (FOODON\_03303675). Directly comparing these heterogeneous representations often results in semantic mismatch. To bridge this gap, SCODEL learns to generate a shared canonical ontology-oriented definition that provides a unified semantic description for both representations.

To supervise this process, we manually construct a collection of canonical ontology-oriented definitions. For every linked recipe food mention and ontology concept pair, annotators produce a single ontology-oriented definition describing the underlying culinary concept. Formally, for the ith aligned pair, the shared canonical definition is

$$
d _ { i } ^ { * } = { \mathcal { A } } \left( m _ { i } , x _ { m _ { i } } , c _ { i } ^ { \mathrm { m e t a } } \right) ,
$$

where A denotes the manual annotation process, $m _ { i }$ is the recipe food mention, $x _ { m _ { i } }$ is its surrounding context, and $c _ { i } ^ { \mathrm { m e t a } }$ is the structured representation of its linked ontology concept. The concept representation is constructed as

![](images/e3e75a18bbed37fc0540dd0d380f7c88fac0f31acd9732419b8c41b02baf6a96.jpg)  
Figure 1: Overview of the proposed SCODEL methodology.

$$
\begin{array} { r } { c _ { i } ^ { \mathrm { m e t a } } = \mathrm { c o n c a t } ( \mathrm { \ l a b e l } ( c _ { i } ) , \mathrm { s y n o n y m s } ( c _ { i } ) , } \\ { \mathrm { p a r e n t s } ( c _ { i } ) , \mathrm { c h i l d r e n } ( c _ { i } ) ) . } \end{array}
$$

Thus, $d _ { i } ^ { * }$ is not specific to either the recipe or ontology representation; rather, it is a single canonical definition grounded jointly in both and serves as their shared supervision target.

SCODEL trains a shared definition generation language model $G _ { \theta }$ to generate the canonical definition independently from either representation:

$$
G _ { \theta } ( m _ { i } , x _ { m _ { i } } )  d _ { i } ^ { * } , \qquad G _ { \theta } ( c _ { i } ^ { \mathrm { m e t a } } )  d _ { i } ^ { * } .
$$

$G _ { \theta }$ is jointly optimised over a training set of N aligned manually annotated food mention and ontology concept pairs using the learning objective

$$
\mathcal { L } = - \sum _ { i = 1 } ^ { N } \Big ( \log P _ { \theta } ( d _ { i } ^ { * } \mid m _ { i } , x _ { m _ { i } } ) + \log P _ { \theta } ( d _ { i } ^ { * } \mid c _ { i } ^ { \mathrm { m e t a } } ) \Big ) .
$$

By optimising both food mentions and ontology concepts towards the same manually curated target, SCODEL learns to produce semantically consistent, ontology-oriented definitions from the two heterogeneous input representations.

## 4.2 Canonical Definition Generation

During inference, the shared language model $G _ { \theta }$ generates canonical ontology-oriented definitions for recipe food mentions and ontology concepts. Since definitions are generated directly from recipe food mentions or ontology concept metadata, SCODEL naturally supports previously unseen recipe food mentions or ontology concepts without requiring additional manual annotation.

## 4.3 Entity Linking Integration

SCODEL is model-agnostic and does not replace existing entity linking architectures. Instead, the generated canonical definitions are incorporated as complementary semantic representations alongside the original entity linking inputs.

For recipe food mentions, the entity linking model receives the original food mention and its surrounding context, augmented with the generated canonical definition. Similarly, each ontology concept is represented by its original ontology metadata and augmented with the corresponding generated canonical definition.

The augmented representations are then supplied to existing entity linking models for candidate retrieval and ranking without modifying their underlying architectures.

## 5 Our ReciLinkD Dataset

We introduce ReciLinkD, a gold-standard food entity linking dataset aligning 3,979 recipe-derived food mentions with FoodOn ontology concepts.

## 5.1 Manual Annotation Process

We perform manual entity linking. For each selected recipe-sourced food mention, annotators are presented with its surface form and representative contextual examples from recipes. Annotators are then asked to select the most semantically appropriate FoodOn concept. The annotation workload is distributed across four annotators with domain familiarity. The annotation guide is provided in Appendix O.

To assess annotation reliability, 300 randomly selected food mentions were independently annotated by all four annotators. Across all six annotator pairs, the mean pairwise F1-score was 89.67%, and the mean Cohen’s κ was 0.7253, indicating strong agreement. The full pairwise results are in Appendix C.

## 5.2 ReciLinkD Dataset Descriptive Metrics

FoodOn Ontology: FoodOn (Dooley et al., 2018) is a structured ontology covering a broad range of food-related concepts, including food materials, food processing and preparation, biological source organisms, packaging, and geographic locations. As our entity linking task focuses specifically on food mentions in recipes, we restrict the candidate ontology space to food material concepts. After normalisation and deduplication, this yields 1,014 food-related concepts. The extraction procedure is described in Appendix A.

ReciFine Dataset: ReciFine (Ibrahim et al., 2026a) provides fine-grained food annotations extracted from approximately 2.2M recipes using methods described in Ibrahim et al. (2026b). From this corpus, we select the 4,500 most frequent food mentions and the contexts in which they appear in recipes, forming the recipe-sourced food mention set for manual linking. Frequency-based selection ensures broad coverage of commonly occurring culinary concepts.

After manual annotation, 3,979 of the 4,500 selected food mentions were successfully mapped to 979 FoodOn concepts. Unlinked food mentions primarily consisted of context-dependent referring expressions, such as it in the context mix the flour, butter, and sugar until a crumbly mixture forms, then spread it over the fruit, where the referent is identifiable from the full recipe context to be a dough but the food mention itself, i.e., it, does not explicitly name the food entity. These were excluded to maintain a gold standard centred on explicit food mentions. For the remaining 35 of the 1,014 candidate FoodOn concepts, no suitable food mention was identified among the selected ReciFine mentions; these concepts are therefore not represented in ReciLinkD. Further details on the ReciLinkD dataset statistics are provided in Appendix A (Table 7).

## 5.3 Manual Canonical Definition Generation

To construct training data for SCODEL, two annotators manually created a single ontology-oriented canonical definition for each aligned recipe food mention and ontology concept pair.

To establish a consistent annotation procedure, both annotators independently generated canonical definitions for 50 randomly selected aligned pairs. They compared their definitions for consistency in semantic interpretation, terminology, and level of abstraction and detail, rather than exact textual agreement. Substantive disagreements were discussed with an independent assessor and resolved by consensus, with the resulting conventions incorporated into the annotation guidelines. The remaining pairs were shared between the annotators and annotated using the agreed guidelines. Full guidelines are provided in Appendix P.

## 6 Experimental Setup

## 6.1 Canonical Definition Generation Models

We evaluate three open-weight LLMs as backbone models for learning and generating shared canonical definitions: Llama-3-8B, Mistral-7B, and Qwen2.5-7B. Each model is trained on the manually aligned canonical definitions described in Section 5.3 and subsequently generates canonical definitions independently.

## 6.2 Entity Linking Models

SCODEL is model-agnostic and is evaluated with three SOTA entity linking models: ReFinED, GSM-EL, and BLINK. Because these models were originally developed for general-domain entity linking, primarily using encyclopedic knowledge bases such as Wikipedia and Wikidata rather than specialised food ontologies such as FoodOn, we made minor adaptations where necessary. Appendix J provides further details on the modifications made to each modular entity linking model.

## 6.3 Baselines

We compare SCODEL against two categories of baselines: (1) LLM-based entity linking frameworks, including LLMAEL, LLM4EL and LLM

<table><tr><td rowspan="2">Method</td><td colspan="4">Supervised (Seen)</td><td colspan="4">Zero-Shot (Unseen)</td><td>AVG</td></tr><tr><td>Hit@1</td><td>Hit@5</td><td>MRR</td><td>H-Acc</td><td>Hit@1</td><td>Hit@5</td><td>MRR</td><td>H-Acc</td><td>Hit@1</td></tr><tr><td>LLM4EL</td><td>63.29</td><td>88.33</td><td>73.90</td><td>66.89</td><td>63.72</td><td>90.21</td><td>74.92</td><td>71.05</td><td>63.51</td></tr><tr><td>LLMAEL</td><td>85.25</td><td>94.51</td><td>89.61</td><td>86.28</td><td>80.71</td><td>94.70</td><td>86.47</td><td>81.52</td><td>82.98</td></tr><tr><td>LLM Rerank@10</td><td>82.68</td><td>89.88</td><td>85.97</td><td>83.70</td><td>83.02</td><td>92.80</td><td>87.19</td><td>83.83</td><td>82.85</td></tr><tr><td>LLM Rerank@50</td><td>81.13</td><td>88.51</td><td>84.52</td><td>81.65</td><td>80.43</td><td>90.35</td><td>84.87</td><td>81.39</td><td>80.78</td></tr><tr><td>Lexical Matching</td><td>52.66</td><td>73.93</td><td>61.62</td><td>54.15</td><td>56.79</td><td>75.27</td><td>65.81</td><td>57.99</td><td>54.73</td></tr><tr><td>ReFinED</td><td>81.82</td><td>89.71</td><td>85.42</td><td>82.68</td><td>78.40</td><td>92.66</td><td>84.38</td><td>79.62</td><td>80.11</td></tr><tr><td>GSM-EL</td><td>76.84</td><td>90.56</td><td>82.62</td><td>78.38</td><td>82.33</td><td>93.75</td><td>87.48</td><td>84.78</td><td>79.59</td></tr><tr><td>BLINK</td><td>87.47</td><td>93.65</td><td>90.31</td><td>88.85</td><td>71.19</td><td>87.09</td><td>77.90</td><td>74.72</td><td>79.33</td></tr><tr><td>SCODEL × ReFinED</td><td>85.08</td><td>93.31</td><td>89.11</td><td>85.59</td><td>82.07</td><td>93.75</td><td>87.38</td><td>83.02</td><td>83.58</td></tr><tr><td>SCODEL × GSM-EL</td><td>80.78</td><td>92.62</td><td>86.24</td><td>82.67</td><td>83.15</td><td>94.83</td><td>88.38</td><td>85.59</td><td>81.97</td></tr><tr><td>SCODEL × BLINK</td><td>88.33</td><td>94.68</td><td>90.95</td><td>88.85</td><td>75.00</td><td>88.72</td><td>81.32</td><td>79.21</td><td>81.67</td></tr></table>

Table 1: Main entity linking results on the ReciLinkD benchmark under seen and unseen settings. SCODEL uses manually generated canonical definitions as supervision and Mistral-7B, the best-performing individual definitionlearning backbone (Table 5). LLM4EL uses Mistral-7B for both definition generation and embedding generation, LLMAEL uses Mistral-7B for context augmentation and ReFinED for entity linking (Appendix B.4: Table 11), and LLM Rerank@10 and Rerank@50 use Mistral-7B for reranking the ReFinED predictions. Best results are bold, second-best results are underlined, and AVG denotes the mean Hit@1 across both settings.

Rerank, that augment entity linking using LLMgenerated semantic representations; and (2) standalone entity linking models, namely ReFinED, GSM-EL, and BLINK, evaluated without SCODEL augmentation. Lexical matching is also evaluated as a simple non-neural baseline. Appendix I and Appendix J provide further details on the generic EL models and their implementation settings, and describe modular EL model-specific adaptations. The LLM re-ranking and lexical matching procedures are detailed in Appendices K and L, respectively, while dataset splits are described in Appendix M.

## 6.4 Evaluation

We report Hit@1, Hit@5, Mean Reciprocal Rank (MRR), and hierarchy-aware accuracy (H-Acc). Appendix N provides detailed definitions of all evaluation metrics.

## 7 Results and Discussion

Table 1 presents the main entity linking results. Across all modular entity linking models, SCODEL consistently improves performance over their original implementations, demonstrating that shared canonical definition learning is complementary to existing EL architectures. SCODEL achieves the best overall average performance when combined with ReFinED (83.58% Hit@1), while SCODEL×BLINK attains the highest supervised performance (88.33% Hit@1, 90.95% MRR). More importantly, SCODEL×GSM-EL achieves the strongest zero-shot results (83.15% Hit@1, 88.38% MRR). LLM4EL performs comparatively lower than the other EL methods, which may reflect differences in their linking mechanisms: LLM4EL directly matches generated mentions and concept definitions through embedding similarity, whereas the other methods use trained entity-linking and ranking architectures. This suggests that, on ReciLinkD, definition-based representations are more effective when integrated with a trained EL model than when used for direct similarity-based matching. The slightly higher zero-shot performance of some models may partly reflect lexical differences between the splits, as exact food mention and ontology concept label matches are more frequent in the zero-shot set than in the supervised set (18.61% vs. 11.84%; Appendix M: Table 25). Overall, these results demonstrate that learning shared canonical definitions improves semantic alignment between recipe food mentions and ontology concepts, supporting generalisation to unseen concepts while remaining complementary to different entity linking architectures.

## 7.1 Ablation Studies

We conduct ablation studies to examine the key design choices underlying SCODEL and to determine where its performance gains originate. We focus on the contribution of shared canonical definition learning and the effect of different sources of canonical supervision, and provide additional analyses of model configuration, semantic alignment, and other experimental choices in Appendix B.

7.1.1 SCODEL Cross-Domain Generalisation To assess generalisability, we apply SCODEL without modification to the GSCplus biomedical entity linking benchmark. As shown in Table 2, SCODEL consistently improves ReFinED and achieves the best overall performance, demonstrating that the proposed shared canonical definition learning method generalises beyond food ontology entity linking. The shared canonical definitions for GSCplus are learned and generated using Llama-3-8B to maintain consistency with the backbone model used by (Xu et al., 2026).

<table><tr><td>Method</td><td>Hit@1</td><td>Hit@5</td><td>MRR</td></tr><tr><td>ReFinED Baseline</td><td>62.75</td><td>77.36</td><td>68.96</td></tr><tr><td>LLM4EL</td><td>63.00</td><td>81.10</td><td></td></tr><tr><td>LLMAEL</td><td>69.97</td><td>85.73</td><td>77.08</td></tr><tr><td>SCODEL</td><td>72.45</td><td>87.87</td><td>79.35</td></tr></table>

Table 2: Results on the GSCplus dataset with the Human Phenotype Ontology (Luo et al., 2021) using ReFinED as the downstream entity linking model for LLMAEL and SCODEL. LLM4EL results are taken directly from Xu et al. (2026), which does not report MRR; all other results are obtained using our implementation on the same benchmark. Best results are bold and second-best results are underlined.

## 7.1.2 Effect of Shared Definition Learning

Table 3 evaluates the contribution of shared definition learning. Mention Definition Learning Only trains the definition generator using definitions of recipe food mentions and their contexts alone; Ontology Definition Learning Only uses definitions of ontology metadata alone, while SCODEL jointly learns a shared canonical definition space from both representations. Although Ontology Definition Learning Only also improves entity linking over the ReFinED baseline, SCODEL achieves the best performance, demonstrating that jointly learning from recipe and ontology representations produces more effective semantic definitions for downstream entity linking.

<table><tr><td>Training Strategy</td><td>Hit@1</td><td>MRR</td></tr><tr><td>ReFinED Baseline</td><td>81.82</td><td>85.42</td></tr><tr><td>Mention Def. Learning Only</td><td>81.65</td><td>86.85</td></tr><tr><td rowspan="2">Ontology Def. Learning Only SCODEL</td><td>82.68</td><td>86.68</td></tr><tr><td>85.08</td><td>89.11</td></tr></table>

Table 3: Effect of different definition learning strategies using ReFinED as the downstream entity linking model. Best results are bold and second-best results are underlined.

## 7.1.3 Frozen Definition Generators

Table 4 compares SCODEL with frozen definition generators that did not learn food mentions and ontology concepts shared definitions. While frozen LLM-generated definitions provide modest improvements over the ReFinED baseline, SCODEL achieves higher Hit@1 and MRR, showing that learning shared canonical definitions is more effective than using off-the-shelf LLM-generated definitions without adaptation.

<table><tr><td>Definition Generator</td><td>Hit@1</td><td>MRR</td></tr><tr><td>ReFinED Baseline</td><td>81.82</td><td>85.42</td></tr><tr><td>Frozen Mistral-7B</td><td>82.16</td><td>87.46</td></tr><tr><td>Frozen GPT-5.1-2025-11-13</td><td>82.50</td><td>86.93</td></tr><tr><td>SCODEL (Mistral-7B)</td><td>85.08</td><td>89.11</td></tr></table>

Table 4: Comparison of SCODEL with frozen definition generators that did not learn shared definitions. ReFinED is used as the downstream EL model. Best results are bold and second-best results are underlined.

## 7.1.4 Definition Learning Backbone

Table 5 compares comparable-size backbone LLMs (7–8B parameters) for SCODEL. Hard voting selects the most predicted ontology concept, whereas soft voting selects the ontology concept with the highest aggregated confidence score. Among the individual models, Mistral-7B achieves the highest Hit@1 and is therefore selected as the definitionlearning backbone for the main experiments. Although ensembling further improves performance by combining predictions from multiple definitionlearning backbones, we use a single Mistral-7B backbone in the main experiments to maintain a consistent single-model configuration across downstream EL models. All model sizes are trained using the same SCODEL training setup.

<table><tr><td>Method</td><td>Def. Generator</td><td>Hit@1</td><td>MRR</td></tr><tr><td rowspan="3">Single</td><td>Llama-3-8B</td><td>82.33</td><td>86.43</td></tr><tr><td>Mistral-7B</td><td>85.08</td><td>89.11</td></tr><tr><td>Qwen2.5-7B</td><td>83.88</td><td>87.94</td></tr><tr><td rowspan="2">Ensemble</td><td>Hard Voting</td><td>85.42</td><td>90.16</td></tr><tr><td>Soft Voting</td><td>86.11</td><td>90.59</td></tr></table>

Table 5: Comparison of backbone models for SCODEL shared definition learning and generation. ReFinED is used as the downstream entity linking model.

## 7.2 Error Analysis

Error analysis of the SCODEL entity linker (Re-FinED) shows that most mistakes arise from confusion between semantically related concepts at different levels of the ontology hierarchy (Table 6). Less frequently, the model predicts conceptually related but incorrect concepts, such as sweet cherry instead of tomato. These errors suggest that SCODEL captures the underlying semantics well but occasionally struggles to distinguish the appropriate level of abstraction.

## 8 Knowledge Graph Applications

8.1 Scaling Food Entity Linking to ReciFine To evaluate SCODEL beyond ReciLinkD, we apply SCODEL × ReFinED, the most consistent configuration across seen and unseen concepts, to 20,000 food mentions from the larger ReciFine corpus and manually evaluate the predicted FoodOn links. Overall, 57.90% of the links were judged correct and suitable for use as gold-standard pairs, 36.13% were judged correct but unsuitable as standalone food mentions, and 10.97% were judged incorrect. This provides a large-scale assessment of generalisation beyond ReciLinkD while substantially expanding our manually verified food mention and ontology concept pairs from 3,979 to 15,559. Appendix D provides further details on food mention selection, manual evaluation, and error categories.

<table><tr><td rowspan=1 colspan=1>Mention with Context</td><td rowspan=1 colspan=1>Gold Concept</td><td rowspan=1 colspan=1>Predicted Concept</td></tr><tr><td rowspan=1 colspan=1>Add the dairy milk to the sauce and stir until smooth.</td><td rowspan=1 colspan=1>animal milk (FOODON_02010013)</td><td rowspan=1 colspan=1>cow milk (FOODON_03307455)</td></tr><tr><td rowspan=1 colspan=1>Toss the salad with fresh snap peas before serving.</td><td rowspan=1 colspan=1>pea (FOODON_03301300)</td><td rowspan=1 colspan=1>sugar snap pea plant (FOODON_03412467)</td></tr><tr><td rowspan=1 colspan=1>Garnish with sliced cherry tomatoes before serving.</td><td rowspan=1 colspan=1>tomato (FOODON_03301453)</td><td rowspan=1 colspan=1>sweet cherry (FOODON_03311457)</td></tr><tr><td rowspan=1 colspan=1>Roast the baby potatoes until golden and tender.</td><td rowspan=1 colspan=1>potato (FOODON_03301449)</td><td rowspan=1 colspan=1>red bliss potato (FOODON_03307121)</td></tr></table>

Table 6: Examples of common entity linking errors made by the SCODEL (Mistral-7B) x ReFinED model.

## 8.2 Ontology-Grounded Recipe Knowledge Graph

Using the 3,979 gold-standard links from ReciLinkD and the new 11,580 manually verified and automatically linked pairs, we extend ontology grounding across the 2.2M recipes in ReciFine by connecting their food mentions to FoodOn concepts and their ontology hierarchy. The resulting knowledge graph contains over 2M grounded recipes linked to at least one FoodOn concept, 19,504 grounded lexically unique food mentions, and 3,188 direct and inherited grounded FoodOn concepts, enabling recipes to be queried and analysed through shared ontology concepts and higherlevel semantic categories rather than surface forms alone. Appendix E provides further details on graph construction and statistics.

## 8.3 Ontology-Aware Recipe Retrieval and Corpus Analysis

Across 15 crafted food-category queries, ontologyaware retrieval substantially outperforms lexical and semantic retrieval, improving macro-F1 by 25.42 percentage points over semantic retrieval and 33.76 percentage points over lexical retrieval (Appendix F.1: Table 19). The advantage is particularly pronounced for conceptually broad queries: for example, winter squash achieves 74.51% F1 with ontology-aware retrieval, compared with 0.00% and 20.83% for lexical and semantic retrieval, respectively, allowing recipes containing more specific descendant foods to be recovered without explicitly mentioning the queried category. Importantly, these results are obtained while fewer than 10% of ReciFine food mentions are currently grounded in the graph, suggesting substantial scope for increased coverage as further food mentions are linked. Beyond retrieval, ontology propagation transforms individual food mention-ontology concept links into corpus-level semantic evidence across recipes (Appendix F.2: Table 20). It also enables quantification of relationships between food categories: for example, 74.74% of recipes containing plant food products also contain animal food products (1.3M recipes), while 68.69% of recipes containing vegetables also contain fruit (Appendix F.3: Table 21). Appendix F provides further results, qualitative retrieval examples, and details of the human evaluation and LLM-as-a-judge methodologies used to assess the retrieval results and corpus-level analysis.

## 9 Conclusion

In this work, we introduced SCODEL, a streamlined, model-agnostic method for ontology entity linking that learns a shared canonical definition space for contextual recipe food mentions and ontology concepts before downstream entity linking. By jointly learning shared, ontology-oriented definitions from both representations, SCODEL provides a semantic normalisation layer that moves beyond surface-form matching and better handles ambiguous, context-dependent food mentions. Extensive experiments show consistent improvements over strong modular entity linking baselines in both supervised and zero-shot settings, and demonstrate that the method generalises beyond the food domain to biomedical ontologies. Furthermore, by enabling robust integration of recipe corpora with FoodOn, SCODEL facilitates downstream applications such as ontology-grounded recipe knowledge graph construction and semantic food analysis. More broadly, the proposed method provides a general and modular approach to ontology entity linking in domains where continual growth of ontology concepts presents significant challenges.

## Limitations

SCODEL is designed to exploit the semantic information available in structured ontologies, and its effectiveness can therefore depend on the quality and completeness of ontology metadata. FoodOn provides rich information including concept labels, definitions, synonyms, and hierarchical relations, whereas ontologies with substantially sparser metadata may provide less information from which to generate informative canonical definitions. Nevertheless, the evaluation on GSCplus with the Human Phenotype Ontology demonstrates that the approach can transfer beyond the food domain and FoodOn.

ReciLinkD focuses on 1,014 food material concepts extracted from selected FoodOn branches rather than the complete FoodOn ontology. This restriction provides a focused candidate space for recipe food entity linking, but concepts outside this space cannot occur as gold entities, and the benchmark should therefore not be interpreted as covering the complete FoodOn ontology, as nonfood-related concepts are not covered. A restricted candidate space may also reduce entity linking difficulty by removing unrelated concepts. However, evaluation on GSCplus shows that the approach transfers to ontologies with larger candidate spaces, reaching 17,655 Ontology concepts.

The main SCODEL experiments use manually constructed canonical definitions as training supervision. While this introduces an initial annotation requirement, canonical definitions can also be generated automatically using LLMs, with our experiments showing that LLM-generated definitions provide an effective alternative to manual supervision.

## References

Tom Ayoola, Shubhi Tyagi, Joseph Fisher, Christos Christodoulopoulos, and Andrea Pierleoni. 2022. Re-FinED: An Efficient Zero-shot-capable Approach to End-to-End Entity Linking. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Hu man Language Technologies: Industry Track, pages 209–220.

Michał Bien, Michał Gilski, Martyna Maciejewska, Wo-´ jciech Taisner, Dawid Wisniewski, and Agnieszka Lawrynowicz. 2020. RecipeNLG: A Cooking Recipes Dataset for Semi-Structured Text Generation. In Proceedings ofthe 13th International Conference on Natural Language Generation, pages 22–28.

Xueqi Cheng, Yuanzheng Wang, Yixing Fan, Jiafeng

Guo, Ruqing Zhang, and Keping Bi. 2024. GSM-EL: A Generalizable Symbol-Manipulation Approach for Entity Linking. IEEE Transactions on Knowledge and Data Engineering, 37(3):1213–1226.

Wen-Haw Chong and Ee-Peng Lim. 2018. Implicit Linking of Food Entities in Social Media. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 169–185. Springer.

Damion M Dooley, Emma J Griffiths, Gurinder S Gosal, Pier L Buttigieg, Robert Hoehndorf, Matthew C Lange, Lynn M Schriml, Fiona SL Brinkman, and William WL Hsiao. 2018. FoodOn: a harmonized food ontology to increase global food traceability, quality control and data integration. npj Science of Food, 2(1):23.

Darius Feher, Faridz Ibrahim, Zhuyan Cheng, Viktor Schlegel, Tom Maidment, James Bagshaw, and Riza Batista-Navarro. 2023. Few-shot Entity Linking of Food Names. Information Processing & Management, 60(5):103463.

Ana Gjorgjevikj, Matej Martinc, Gjorgjina Cenikj, Sašo Džeroski, Barbara Koroušic Seljak, and Tome Ef-´ timov. 2025. FoodSEM: Large Language Model Specialized in Food Named-Entity Linking. In International Conference on Discovery Science, pages 395–410. Springer.

Feng Hou, Ruili Wang, Jun He, and Yi Zhou. 2020. Improving Entity Linking through Semantic Reinforced Entity Embeddings. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 6843–6848.

Nuhu Ibrahim, Rishi Ravikumar, Robert Stevens, and Riza Theresa Batista-Navarro. 2026a. ReciFine: Finely Annotated Recipe Dataset for Controllable Recipe Generation. In Findings of the Association for Computational Linguistics: EACL 2026, pages 4058–4074.

Nuhu Ibrahim, Robert Stevens, and Riza Theresa Batista-Navarro. 2026b. Knowledge Augmentation Enhances Token Classification for Recipe Understanding. In Proceedings of the 19th Conference of the European Chapter ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 2776–2788.

Matthew A Jaro. 1989. Advances in Record-Linkage Methodology as Applied to Matching the 1985 Census of Tampa, Florida. Journal ofthe American Statistical Association, 84(406):414–420.

Md Tahmid Rahman Laskar, Cheng Chen, Aliaksandr Martsinovich, Jonathan Johnston, Xue-Yong Fu, Shashi Bhushan Tn, and Simon Corston-Oliver. 2022. BLINK with Elasticsearch for Efficient Entity Linking in Business Conversations. In Proceedings of the 2022 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies: Industry Track, pages 344–352.

VI Lcvenshtcin. 1966. Binary Codes Capable of Correcting Deletions, Insertions, and Reversals. In Soviet Physics-Doklady, volume 10, pages 707–710.

Lajanugen Logeswaran, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, Jacob Devlin, and Honglak Lee. 2019. Zero-Shot Entity Linking by Reading Entity Descriptions. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 3449–3460.

Ling Luo, Shankai Yan, Po-Ting Lai, Daniel Veltri, Andrew Oler, Sandhya Xirasagar, Rajarshi Ghosh, Morgan Similuk, Peter N Robinson, and Zhiyong Lu. 2021. PhenoTagger: a hybrid method for phenotype concept recognition using human phenotype ontology. Bioinformatics, 37(13):1884–1890.

Javier Marın, Aritro Biswas, Ferda Ofli, Nicholas Hynes, Amaia Salvador, Yusuf Aytar, Ingmar Weber, and Antonio Torralba. 2021. Recipe1M+: A Dataset for Learning Cross-Modal Embeddings for Cooking Recipes and Food Images. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(1):187– 203.

Jose G Moreno, Romaric Besançon, Romain Beaumont, Eva D’hondt, Anne-Laure Ligozat, Sophie Rosset, Xavier Tannier, and Brigitte Grau. 2017. Combining Word and Entity Embeddings for Entity Linking. In European Semantic Web Conference, pages 337–352. Springer.

Aasish Pappu, Roi Blanco, Yashar Mehdad, Amanda Stent, and Kapil Thadani. 2017. Lightweight Multilingual Entity Extraction and Linking. In Proceedings ofthe Tenth ACM International Conference on Web Search and Data Mining, pages 365–374.

Alberto Parravicini, Rhicheek Patra, Davide B Bartolini, and Marco D Santambrogio. 2019. Fast and Accurate Entity Linking via Graph Embedding. In Proceedings of the 2nd Joint International Workshop on Graph Data Management Experiences & Systems (GRADES) and Network Data Analytics (NDA), pages 1–9.

Gorjan Popovski, Barbara Korousic-Seljak, and Tome Eftimov. 2019. FoodOntoMap: Linking Food Concepts across Different Food Ontologies. In KEOD, pages 195–202.

Delip Rao, Paul McNamee, and Mark Dredze. 2012. Entity Linking: Finding Extracted Entities in a Knowledge Base. In Multi-source, Multilingual Information Extraction and Summarization, pages 93–115. Springer.

Wei Shen, Jianyong Wang, and Jiawei Han. 2014. Entity Linking with a Knowledge Base: Issues, Techniques, and Solutions. IEEE Transactions on Knowledge and Data Engineering, 27(2):443–460.

Daniel Vollmers, Hamada Zahera, Diego Moussallem, and Axel-Cyrille Ngonga Ngomo. 2025. Contextual Augmentation for Entity Linking using Large

Language Models. In Proceedings ofthe 31st International Conference on Computational Linguistics, pages 8535–8545.

Ledell Wu, Fabio Petroni, Martin Josifoski, Sebastian Riedel, and Luke Zettlemoyer. 2020. Scalable Zeroshot Entity Linking with Dense Entity Retrieval. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pages 6397–6407.

Amy Xin, Yunjia Qi, Zijun Yao, Fangwei Zhu, Kaisheng Zeng, Bin Xu, Lei Hou, and Juanzi Li. 2025. LL-MAEL: Large Language Models are Good Context Augmenters for Entity Linking. In Proceedings ofthe 34th ACM International Conference on Information and Knowledge Management, pages 3550–3559.

Fuqi Xu, Goran Nenadic, and Robert Stevens. 2026. LLM4EL: A Retrieval-Augmented Framework for Entity Linking with Hierarchical and External Knowledge Integration. Research Square.

Zonghai Yao, Liangliang Cao, and Huapu Pan. 2020. Zero-shot Entity Linking with Efficient Long Range Sequence Modeling. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 2517–2522.

## A Extracting Food Related Concepts from FoodOn Ontology

Algorithm 1 describes the procedure for extracting relevant food concepts from the FoodOn ontology. We select four root categories: fruit (PO\_0009001), seed (PO\_0009010), food (raw) (FOODON\_03311126) and cultural food (FOODON\_00001796)). These roots restrict our selected concepts to FoodOn branches corresponding to food materials relevant to recipe entity linking. For each root, we retrieve all descendant concepts and identify leaf nodes within the resulting subtree. To capture both fine-grained and more general concepts, we include each leaf node together with its immediate parent class. We use a leaf-plus-parent extraction strategy to balance concept specificity and coverage. Leaf nodes provide the fine-grained food concepts required for precise entity linking, while their immediate parents retain broader concepts that may also correspond directly to food mentions in recipes. Finally, we normalise and deduplicate the resulting concept set to produce the candidate ontology set used for annotation. Table 7 summarises key statistics of the annotated ReciLinkD dataset.

Algorithm 1 Steps for Extracting Food Concepts   
from FoodOn Ontology   
Require: FoodOn ontology O   
Ensure: Candidate concept set C   
1: R ← {PO\_0009001, PO\_0009010,   
FOODON\_03311126, FOODON\_00001796}   
2: C ← ∅   
3: for each parent r ∈ R do   
4: T ← Descendants(r) ∪ {r}   
5: L ← {c ∈ T | Children<sub>T</sub>(c) = ∅} {leaf   
nodes}   
6: for each ℓ ∈ L do   
7: C ← C ∪ {ℓ}   
8: C ← C ∪ Parents(ℓ)   
9: end for   
10: end for   
11: Normalise and deduplicate C   
12: return C   
Metric Value   
Unique manually selected food mentions 4,500   
Unique manually linked food mentions 3,979   
Unique unlinked food mentions 521   
Unique FoodOn food concepts 979   
Rows with synonyms 2,023   
Rows with superclasses 2,970   
Rows with subclasses 1,971   
Rows with direct parents 1,649   
Rows with direct children 1,151   
Avg. synonyms per row 0.82   
Avg. parent concepts per row 1.37  
Table 7: Descriptive statistics for ReciLinkD.

## B Supplementary Ablation Studies

## B.1 Choosing Canonical Definition Source

Table 8 compares different sources for constructing the shared canonical training definitions. Humanannotated definitions achieve the best overall performance, indicating that carefully aligned canonical definitions provide the strongest supervision for SCODEL. Nevertheless, canonical definitions generated by both proprietary and open-weight LLMs consistently outperform the ReFinED baseline, suggesting that automatically generated shared definitions offer an effective and scalable alternative when manual annotation is impractical for large ontologies.

## B.2 Semantic Alignment Analysis

We examine whether shared canonical definition learning directly improves semantic alignment between recipe mentions and ontology concepts by evaluating definition-based concept retrieval before and after SCODEL training. In both settings, definitions are generated independently for the contextual mention and ontology metadata, embedded using the same all-mpnet-base-v2 model, and ranked using cosine similarity over the same 979 candidate concepts. The baseline uses the original Mistral-7B in a zero-shot setting for definition generation as in LLM4EL (Xu et al., 2026), whereas SCODEL uses Mistral-7B after shared canonical definition learning.

<table><tr><td>Training Definition Source</td><td>Hit@1</td><td>MRR</td></tr><tr><td>ReFinED Baseline</td><td>81.82</td><td>85.42</td></tr><tr><td>OpenAI GPT-5.1-2025-11-13</td><td>84.91</td><td>89.06</td></tr><tr><td>Claude Sonnet 5</td><td>84.05 83.36</td><td>88.28</td></tr><tr><td>Qwen3-235B</td><td></td><td>88.07</td></tr><tr><td>Human Annotated</td><td>85.08</td><td>89.11</td></tr></table>

Table 8: Effect of different sources used for constructing the canonical definitions. All experiments use Mistral-7B as the shared definition learner and generator and ReFinED as the downstream entity linking model. Best results are bold and second-best results are underlined.

We additionally measure Gold Sim., the cosine similarity between the food mention definition and its gold ontology concept definition and Global Margin, the difference between the gold similarity and the mean similarity to non-matching ontology concepts.

<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>Mistral-7B</td><td rowspan=1 colspan=1>SCODEL</td></tr><tr><td rowspan=1 colspan=1>Hit@1</td><td rowspan=1 colspan=1>27.62</td><td rowspan=1 colspan=1>46.31</td></tr><tr><td rowspan=1 colspan=1>Hit@5</td><td rowspan=1 colspan=1>51.63</td><td rowspan=1 colspan=1>70.84</td></tr><tr><td rowspan=1 colspan=1>Hit@10</td><td rowspan=1 colspan=1>64.67</td><td rowspan=1 colspan=1>78.73</td></tr><tr><td rowspan=1 colspan=1>MRR</td><td rowspan=1 colspan=1>39.92</td><td rowspan=1 colspan=1>57.72</td></tr><tr><td rowspan=1 colspan=1>Gold Sim.</td><td rowspan=1 colspan=1>0.6771</td><td rowspan=1 colspan=1>0.8312</td></tr><tr><td rowspan=1 colspan=1>Global Margin</td><td rowspan=1 colspan=1>0.3148</td><td rowspan=1 colspan=1>0.3595</td></tr></table>

Table 9: Semantic alignment before and after SCODEL shared canonical definition learning. Mistral-7B denotes zero-shot definition generation using the original model as in LLM4EL (Xu et al., 2026). Best results are bold.

SCODEL improves retrieval across all ranking metrics, increasing Hit@1 from 27.62% to 46.31% and MRR from 39.92% to 57.72%. Gold similarity increases from 0.6771 to 0.8312, while the global margin increases from 0.3148 to 0.3595. These results provide direct evidence that shared canonical definition learning improves semantic alignment between recipe mention definitions and their corresponding ontology concept definitions.

## B.3 Choosing Definition Learning Backbone Size

Model size influences the quality of the learned canonical definitions, as shown in Table 10. Performance generally improves with increasing model size, although the 8B model slightly underperforms the 3B variant. The 70B model achieves the best overall results, suggesting that larger language models learn more effective shared canonical definitions for downstream entity linking.

<table><tr><td rowspan=1 colspan=1>Model Family</td><td rowspan=1 colspan=1>Parameters</td><td rowspan=1 colspan=1>Hit@1</td><td rowspan=1 colspan=1>MRR</td></tr><tr><td rowspan=1 colspan=1>LLaMA-3</td><td rowspan=1 colspan=1>1B</td><td rowspan=1 colspan=1>81.82</td><td rowspan=1 colspan=1>85.60</td></tr><tr><td rowspan=1 colspan=1>LLaMA-3</td><td rowspan=1 colspan=1>3B</td><td rowspan=1 colspan=1>83.53</td><td rowspan=1 colspan=1>87.12</td></tr><tr><td rowspan=1 colspan=1>LLaMA-3</td><td rowspan=1 colspan=1>8B</td><td rowspan=1 colspan=1>82.33</td><td rowspan=1 colspan=1>86.43</td></tr><tr><td rowspan=1 colspan=1>LLaMA-3</td><td rowspan=1 colspan=1>70B</td><td rowspan=1 colspan=1>84.91</td><td rowspan=1 colspan=1>88.72</td></tr></table>

Table 10: Effect of model size on SCODEL shared definition learning and generation. ReFinED is used as the downstream entity linking model. Best results are bold and second-best results are underlined.

## B.4 LLMAEL Across Entity Linking Models

We assess LLMAEL’s effectiveness across different entity linking architectures by applying it to three downstream entity linking models: ReFinED, GSM-EL, and XMR. Mistral was retained as the language model across all experiments to ensure that differences in performance primarily reflect the choice of entity linking model.

Table 11 shows that LLMAEL performs best when ReFinED is used as the downstream entity linking model. The ReFinED configuration obtains a Hit@1 of 85.25%, Hit@5 of 94.51%, and MRR of 89.61%, outperforming the corresponding GSM-EL and XMR configurations.

<table><tr><td rowspan=1 colspan=1>Entity Linking Model</td><td rowspan=1 colspan=1>Hit@1</td><td rowspan=1 colspan=1>Hit@5</td><td rowspan=1 colspan=1>MRR</td></tr><tr><td rowspan=1 colspan=1>ReFinED</td><td rowspan=1 colspan=1>85.25</td><td rowspan=1 colspan=1>94.51</td><td rowspan=1 colspan=1>89.61</td></tr><tr><td rowspan=1 colspan=1>GSM-EL</td><td rowspan=1 colspan=1>81.81</td><td rowspan=1 colspan=1>93.65</td><td rowspan=1 colspan=1>87.06</td></tr><tr><td rowspan=1 colspan=1>BLINK</td><td rowspan=1 colspan=1>85.22</td><td rowspan=1 colspan=1>93.82</td><td rowspan=1 colspan=1>89.60</td></tr></table>

Table 11: Performance of LLMAEL across different downstream entity linking models. Mistral-7B is used as the language model across all configurations. Best results are in bold.

## C ReciLinkD Inter-Annotator Agreement

To assess the reliability of the ReciLinkD annotations, 300 randomly selected recipe food mentions were independently annotated by all four annotators. Agreement was evaluated across all six annotator pairs using pairwise F1-score and Cohen’s κ. Table 12 reports the results for each annotator pair.

The pairwise F1-scores range from 85.49% to 92.43%, with a mean of 89.67%. Cohen’s κ ranges from 0.7178 to 0.7404, with a mean of 0.7253. These results indicate consistently strong agreement across annotator pairs.

<table><tr><td rowspan=1 colspan=1>Annotator Pair</td><td rowspan=1 colspan=1>F1 (%)</td><td rowspan=1 colspan=1>Cohen&#x27;s κ</td></tr><tr><td rowspan=1 colspan=1>A-B</td><td rowspan=1 colspan=1>85.49</td><td rowspan=1 colspan=1>0.7178</td></tr><tr><td rowspan=1 colspan=1>A-C</td><td rowspan=1 colspan=1>90.56</td><td rowspan=1 colspan=1>0.7254</td></tr><tr><td rowspan=1 colspan=1>A-D</td><td rowspan=1 colspan=1>91.31</td><td rowspan=1 colspan=1>0.7404</td></tr><tr><td rowspan=1 colspan=1>B-C</td><td rowspan=1 colspan=1>89.53</td><td rowspan=1 colspan=1>0.7219</td></tr><tr><td rowspan=1 colspan=1>B-D</td><td rowspan=1 colspan=1>88.68</td><td rowspan=1 colspan=1>0.7203</td></tr><tr><td rowspan=1 colspan=1>C-D</td><td rowspan=1 colspan=1>92.43</td><td rowspan=1 colspan=1>0.7258</td></tr><tr><td rowspan=1 colspan=1>Mean</td><td rowspan=1 colspan=1>89.67</td><td rowspan=1 colspan=1>0.7253</td></tr></table>

Table 12: Pairwise inter-annotator agreement for the 300 ReciLinkD food mentions independently annotated by all four annotators.

## D Large-Scale Entity Linking to ReciFine

We extend ReciLinkD to a substantially larger set of food mentions from ReciFine (Ibrahim et al., 2026a), extracted using the methods in Ibrahim et al. (2026b), to evaluate SCODEL beyond the benchmark and build a larger manually verified resource for ontology grounding. The process selects additional ReciFine food mentions, links them to FoodOn using SCODEL × ReFinED, manually verifies the predictions, and combines the retained mappings with ReciLinkD.

## D.1 Selection and Linking

ReciFine contains food mentions extracted from instructions of approximately 2.2M recipes. We rank these food mentions by the frequency of their appearance in recipes and select the 20,000 most frequent food mentions not already covered by ReciLinkD. Frequency-based selection prioritises mappings with greater potential corpus coverage, since a single verified mapping can ground multiple occurrences of the same food mention across recipes in ReciFine.

## D.2 Manual Verification

All 20,000 predicted food mention and ontology concept pairs are manually reviewed. Each prediction is assigned one of three outcomes: Correct and retained, where the expression is a valid food mention and the predicted FoodOn concept represents the intended entity; Incorrect FoodOn prediction, where the expression is a valid or invalid food mention and is linked to an incorrect FoodOn concept; or Correct unsuitable food mention, where the predicted concept is contextually appropriate, but the extracted expression itself is unsuitable as a standalone food name.

The final category separates entity-linking errors from food-mention extraction incompatibilities. For example, in mix the flour, butter, and sugar until a crumbly mixtureforms, then spread it over the fruit, the referent of it can be identified from the recipe context as the dough, although the extracted mention it does not explicitly name the food entity. The entity-linking prediction may therefore be correct, but retaining it as a lexical alias for the corresponding FoodOn concept would introduce noise and inconsistencies when the resource is subsequently propagated across ReciFine.

Table 13 reports the manual evaluation results.
<table><tr><td rowspan=1 colspan=1>Manual Evaluation Outcome</td><td rowspan=1 colspan=1>Count</td><td rowspan=1 colspan=1>Perc.</td></tr><tr><td rowspan=1 colspan=1>Correct and retained</td><td rowspan=1 colspan=1>11,580</td><td rowspan=1 colspan=1>57.90%</td></tr><tr><td rowspan=1 colspan=1>Correct unsuitable food mention</td><td rowspan=1 colspan=1>7,226</td><td rowspan=1 colspan=1>36.13%</td></tr><tr><td rowspan=1 colspan=1>Incorrect FoodOn prediction</td><td rowspan=1 colspan=1>2,194</td><td rowspan=1 colspan=1>10.97%</td></tr></table>

Table 13: Manual evaluation of SCODEL × ReFinED predictions for 20,000 additional ReciFine food mentions.

Among valid food mentions, SCODEL × Re-FinED correctly links 18,806 to the appropriate FoodOn concept. After excluding expressions unsuitable for subsequent lexical grounding, 11,580 additional manually verified food mention and ontology concept mappings are retained.

## D.3 Expanded Linking Resource

The new retained ReciFine mappings are combined with the original ReciLinkD annotations to form an expanded manually verified food mention and ontology concept resource. Table 14 reports its resulting coverage.

<table><tr><td rowspan=1 colspan=1>Source</td><td rowspan=1 colspan=1>Mention-Concept Pairs</td></tr><tr><td rowspan=1 colspan=1>ReciLinkD</td><td rowspan=1 colspan=1>3,979</td></tr><tr><td rowspan=1 colspan=1>ReciLinkD Expansion</td><td rowspan=1 colspan=1>11,580</td></tr><tr><td rowspan=1 colspan=1>Combined resource</td><td rowspan=1 colspan=1>15,559</td></tr></table>

Table 14: Composition of the expanded manually verified food mention and ontology concept linking resource.

## E Ontology-Grounded Recipe Knowledge Graph

The expanded linking resource provides verified correspondences between ReciFine food mention surface forms and FoodOn concepts. We use these mappings to ground food mentions across the wider ReciFine corpus and integrate the resulting recipe entities with the FoodOn hierarchy.

## E.1 Vocabulary Expansion and ReciFine Grounding

To propagate the manually verified food mention and FoodOn concept mappings efficiently, we construct a concept-preserving lexical vocabulary for each represented FoodOn concept. The vocabulary combines manually verified food mentions with the corresponding FoodOn concept label, synonyms, and their conservative variants such as singular and plural forms.

For example, a verified mapping from chickpeas to a concept with the preferred label chickpea and synonym garbanzo bean allows these equivalent lexical forms, together with conservative plural variants such as garbanzo beans, to ground corresponding food mentions in ReciFine to the same FoodOn concept.

The expansion remains concept-preserving: labels and synonyms are used only as aliases for the FoodOn concept to which they belong. Parent and child labels are not treated as lexical aliases, since ontological relatedness does not imply lexical equivalence. Where the same normalised form is associated with multiple FoodOn concepts, the ambiguity is retained rather than randomly assigning the expression to one concept.

Importantly, this propagation stage does not introduce new SCODEL predictions. Instead, it expands the coverage of manually verified mappings using lexical forms associated with their established FoodOn concepts, allowing additional ReciFine food mentions to be grounded correctly despite differences in surface form or spelling. Table 15 shows the resulting vocabulary expansion.

<table><tr><td rowspan=1 colspan=1>Vocabulary Expansion</td><td rowspan=1 colspan=1>Count</td></tr><tr><td rowspan=1 colspan=1>Original verified food mention names</td><td rowspan=1 colspan=1>15,559</td></tr><tr><td rowspan=1 colspan=1>FoodOn concept labels</td><td rowspan=1 colspan=1>979</td></tr><tr><td rowspan=1 colspan=1>Names from FoodOn concept synonyms</td><td rowspan=1 colspan=1>253</td></tr><tr><td rowspan=1 colspan=1>Singular/plural name variants</td><td rowspan=1 colspan=1>16,738</td></tr><tr><td rowspan=1 colspan=1>Final expanded vocabulary</td><td rowspan=1 colspan=1>33,529</td></tr></table>

Table 15: Expansion of the verified food-name vocabulary used to ground additional ReciFine mentions.

We use the resulting vocabulary to ground the food mentions previously extracted from the ReciFine recipes. We match each extracted food mention against the vocabulary and, when we find a valid match, assign the corresponding FoodOn concept. This propagates the manually verified mappings across ReciFine, producing ontologygrounded food mentions that we then use to construct the recipe knowledge graph. Table 16 summarises the resulting corpus coverage.

## E.2 FoodOn Hierarchy Integration

Lexical grounding identifies the FoodOn concept corresponding to a recipe food mention, while the FoodOn hierarchy lets these concepts be interpreted through broader semantic categories. For each directly grounded concept, we therefore retain its FoodOn identifier and taxonomic relationships.

<table><tr><td rowspan=1 colspan=1>Statistic</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>ReciFine recipes considered</td><td rowspan=1 colspan=1>2,206,253</td></tr><tr><td rowspan=1 colspan=1>Unique food mentions matched to vocabulary</td><td rowspan=1 colspan=1>19,504</td></tr><tr><td rowspan=1 colspan=1>Recipes grounded to food mentions</td><td rowspan=1 colspan=1>2,062,103</td></tr></table>

Table 16: Coverage obtained by propagating the expanded verified food mention and ontology concept vocabulary across ReciFine.

If a recipe contains a mention linked to concept $c ,$ and c is a descendant of concept a, the recipe can also be interpreted and retrieved as containing an instance of a. This distinction between direct grounding and inherited category membership is maintained throughout the graph: the former records the concept explicitly identified from a recipe mention, whereas the latter is inferred through FoodOn’s ontology structure.

The resulting graph contains three principal node types: recipe nodes, food mention nodes, and FoodOn concept nodes. Its principal structure is

$$
r ~ { \xrightarrow { \mathsf { c o n t a i n s } } } \ m ~ { \xrightarrow { \mathrm { 1 i n k e d } _ { - } \mathrm { t o } } } \ c ~ { \xrightarrow { \mathrm { 1 } s _ { - } \mathsf { a } ^ { * } } } \ a ,
$$

where r denotes a recipe, m a grounded food mention, c its directly linked FoodOn concept, and a an inherited higher-level FoodOn concept.

## E.3 Knowledge Graph Statistics

Table 17 reports the resulting graph statistics. We distinguish directly grounded FoodOn concepts from ontology concepts introduced through hierarchy expansion to avoid conflating concepts observed in recipes with those included only as semantic context.

<table><tr><td rowspan=1 colspan=1>Knowledge Graph Statistic</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>ReciFine recipes considered</td><td rowspan=1 colspan=1>2,206,253</td></tr><tr><td rowspan=1 colspan=1>Grounded recipe nodes</td><td rowspan=1 colspan=1>2,062,103</td></tr><tr><td rowspan=1 colspan=1>Grounded food mention nodes</td><td rowspan=1 colspan=1>14,177,860</td></tr><tr><td rowspan=1 colspan=1>Directly grounded FoodOn concepts</td><td rowspan=1 colspan=1>979</td></tr><tr><td rowspan=1 colspan=1>Inherited FoodOn concepts</td><td rowspan=1 colspan=1>2,209</td></tr><tr><td rowspan=1 colspan=1>Recipe → mention edges</td><td rowspan=1 colspan=1>14,177,860</td></tr><tr><td rowspan=1 colspan=1>Food mention → concept edges</td><td rowspan=1 colspan=1>14,177,860</td></tr><tr><td rowspan=1 colspan=1>Direct ontology hierarchy edges</td><td rowspan=1 colspan=1>4,190</td></tr><tr><td rowspan=1 colspan=1>Derived ontology closure edges</td><td rowspan=1 colspan=1>17,555</td></tr><tr><td rowspan=1 colspan=1>Total grounded principal graph nodes</td><td rowspan=1 colspan=1>16,243,151</td></tr><tr><td rowspan=1 colspan=1>Total grounded principal graph edges</td><td rowspan=1 colspan=1>28,359,910</td></tr></table>

Table 17: Statistics of the expanded ontology-grounded ReciFine knowledge graph. Of the 2,206,253 ReciFine recipes considered, 2,062,103 contain at least one grounded food mention and are represented as recipe nodes. Derived transitive ontology closure edges are reported separately and excluded from the principal graph-edge total.

## F Ontology-Aware Retrieval and Corpus Analysis

We evaluate the practical utility of the ontologygrounded representation through recipe retrieval and corpus-level semantic analysis. The former examines whether FoodOn hierarchy information improves retrieval beyond lexical and semantic search. Retrieval relevance is assessed using an LLM-as-ajudge, with two human annotators independently evaluating a sample to measure inter-annotator and human–LLM agreement. The latter examines foodcategory prevalence and co-occurrence across ReciFine using previously verified food mention mappings to gold FoodOn concepts, allowing corpuslevel analysis without additional manual annotation.

## F.1 Ontology-Aware Recipe Retrieval

Ontology-aware retrieval identifies the corresponding FoodOn concept and retrieves recipes containing either that concept or any of its descendants. Thus, a query for fruit, for example, can retrieve recipes grounded to specific descendant concepts such as apple, banana, strawberry, and blueberry even when the lexical formfruit does not occur in the recipe.

The retrieval search space consists of 2,000 recipes selected from the approximately 2.2M recipes in ReciFine using proportionate stratified random sampling. To construct a corpusrepresentative evaluation collection, we stratify recipes by the number of food mentions grounded to verified FoodOn concepts and by recipe instruction length. We sample approximately proportional numbers of recipes from each stratum, using a fixed random seed for reproducibility. This ensures that the evaluation includes recipes with varying levels of ontology coverage (including recipes with no food mentions grounded to FoodOn concepts) and textual complexity. The same fixed 2,000-recipe collection is used to evaluate lexical, semantic, and ontology-aware retrieval.

For lexical retrieval, we normalise the query and then retrieve recipes that contain a food mention that directly matches the query. For example, a query for fruit retrieves recipes that explicitly contain the mentionfruit, but not recipes that contain only apple or banana.

For semantic retrieval, we encode the query and extracted food mentions independently using the pretrained all-mpnet-base-v2 sentencetransformer model and compute their cosine similarity. Recipes containing food mentions whose cosine similarity to the query is at least a fixed threshold of 0.70 are retrieved. This allows semantic retrieval to identify related food mentions without requiring an exact lexical match or using ontology information.

For ontology-aware retrieval, we map the query to its corresponding FoodOn concept and retrieve recipes grounded to either that concept or any of its descendants. Thus, a query for fruit can retrieve recipes grounded to specific descendant concepts such as apple, banana, strawberry, and blueberry, even when the word fruit does not occur in the recipe. Unlike semantic retrieval, this approach determines category membership from explicit relationships in the FoodOn hierarchy rather than embedding similarity.

We evaluate the three approaches independently across the same 15 hand-selected higher-level foodcategory queries. To enable evaluation at scale, retrieved recipes are automatically assessed using a GPT-5.6-Sol model. Given a food-category query and the food mentions extracted from a recipe, the model determines whether the recipe contains at least one food mention belonging to the queried category. This produces a binary relevance judgement for each evaluated query and recipe pair. Figure 2 illustrates the prompt used to determine whether the recipe contains food mentions belonging to the queried category.

To assess the reliability of these automatic judgements, we conduct a separate human agreement study on 200 randomly sampled query and recipe pairs. Two human annotators independently assign binary relevance labels to each pair using the same criterion as the automatic annotator. We measure inter-annotator agreement between the two human annotators and agreement between the GPT-5.6-Sol annotations and each human annotator. The resulting agreement scores are reported in Table 18. Human annotators show substantial agreement (κ = 0.76), while GPT-5.6-Sol also achieves substantial agreement with both annotators (κ = 0.64 and 0.61).

For each retrieval approach and query, we report precision, recall, and F1-score based on the automatic relevance judgements in Table 19. A recipe is considered relevant when its food entities satisfy the semantic category represented by the query.

## F.2 Food-Category Prevalence

The Ontology hierarchy represented by the knowledge graph supports corpus-level analysis that

You are evaluating whether a recipe contains a food belonging to a specified food category.

Food category: {CATEGORY\_QUERY}

Recipe food mentions: {FOOD\_MENTIONS}

Recipe Directions: {RECIPE\_DIRECTIONS}

Determine whether at least one of the food mentions in the recipe belongs to the specified food category

Consider the semantic meaning of the category rather than requiring an exact lexical match. A food may belong to the category even when the category name does not explicitly occur in the recipe.

Return only one label:

YES – the recipe contains at least one food belonging to the category.

NO – the recipe does not contain any food belonging to the category.

Figure 2: Prompt template for determining whether the recipe contains food mentions belonging to the queried category.
<table><tr><td rowspan=1 colspan=1>Annotator Pair</td><td rowspan=1 colspan=1>Agreement</td><td rowspan=1 colspan=1>Cohen&#x27;s κ</td></tr><tr><td rowspan=1 colspan=1>Human 1 – Human 2</td><td rowspan=1 colspan=1>88.95%</td><td rowspan=1 colspan=1>0.76</td></tr><tr><td rowspan=1 colspan=1>Human 1 – GPT-5.6-Sol</td><td rowspan=1 colspan=1>82.00%</td><td rowspan=1 colspan=1>0.64</td></tr><tr><td rowspan=1 colspan=1>Human 2 – GPT-5.6-Sol</td><td rowspan=1 colspan=1>80.50%</td><td rowspan=1 colspan=1>0.61</td></tr></table>

Table 18: Agreement on 200 randomly sampled query and recipe relevance judgements. Agreement denotes the percentage of identical binary relevance labels, while Cohen’s kappa (κ) accounts for chance agreement.

<table><tr><td rowspan="2">Query</td><td colspan="3">Lexical</td><td colspan="3">Semantic</td><td colspan="3">Ontology</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>Fruit</td><td>86.67</td><td>4.34</td><td>8.26</td><td>91.26</td><td>10.46</td><td>18.76</td><td>71.77</td><td>76.08 73.87</td><td></td></tr><tr><td>Vegetable</td><td>85.11</td><td>3.45</td><td>6.64</td><td>96.19</td><td>8.72</td><td>15.99</td><td>82.06</td><td>18.57</td><td>30.28</td></tr><tr><td>Legume</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>60.37</td><td>52.11</td><td>55.93</td></tr><tr><td>Meat</td><td>80.25</td><td>20.93</td><td>33.21</td><td>85.64</td><td>26.89</td><td>40.93</td><td>94.76</td><td>34.94</td><td>51.06</td></tr><tr><td>Pork</td><td>91.49</td><td>21.83</td><td>35.25</td><td>91.49</td><td>21.83</td><td>35.25</td><td>88.24</td><td>15.23</td><td>25.97</td></tr><tr><td>Chicken</td><td>97.64</td><td>71.13</td><td>82.31</td><td>97.70</td><td>58.42</td><td>73.12</td><td>96.84</td><td>63.23</td><td>76.51</td></tr><tr><td>Seafood</td><td>100.00</td><td>1.38</td><td>2.72</td><td>95.16</td><td>40.69</td><td>57.00</td><td>97.06</td><td>22.76</td><td>36.87</td></tr><tr><td>Crustacean</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>94.12</td><td>49.23</td><td>64.65</td></tr><tr><td>Pepper</td><td>40.70</td><td>51.98</td><td>45.65</td><td>43.83</td><td>60.95</td><td>50.99</td><td>80.34</td><td>37.73</td><td>51.35</td></tr><tr><td>Winter squash</td><td>0.00</td><td>0.00</td><td>0.00</td><td>29.41</td><td>16.13</td><td>20.83</td><td>95.00</td><td>61.29</td><td>74.51</td></tr><tr><td>Pome fruit</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>98.28</td><td>61.96 76.00</td><td></td></tr><tr><td>Drupe fruit</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>41.38</td><td>17.96</td><td>25.04</td></tr><tr><td>Aggregate fruit</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>90.00</td><td>54.55</td><td>67.92</td></tr><tr><td>Cereal grain</td><td>0.00</td><td>0.00</td><td>0.00</td><td>84.21</td><td>1.58</td><td>3.09</td><td>92.42</td><td>6.01</td><td>11.29</td></tr><tr><td>Nut</td><td>100.00</td><td>2.13</td><td>4.17</td><td>45.54</td><td>19.57</td><td>27.38</td><td>100.00</td><td>1.70</td><td>3.35</td></tr><tr><td>Average</td><td>45.46</td><td>11.81</td><td>14.55</td><td>50.70</td><td>17.68</td><td>22.89</td><td>85.51</td><td>38.22</td><td>48.31</td></tr></table>

Table 19: Comparison of lexical, semantic, and ontology-aware recipe retrieval across 15 higher-level food-category queries. P, R, and F1 denote precision, recall, and F1-score (%), respectively, over the 2,000- recipe ReciFine evaluation sample. The Average row reports averaged performance across the 15 queries. Best results are in bold.

would be difficult to perform reliably using individual food names alone. We use the FoodOn hierarchy to group directly grounded food concepts under broader food categories and then calculate how many recipes contain each category. A recipe is counted for a category when it contains at least one grounded FoodOn concept corresponding to that category or one of its descendants.

For example, recipes grounded to specific fruits such as apple, banana, or strawberry can all be counted under the broader fruit category. The same process lets us determine the prevalence of broader categories such as vegetables, fish, dairy products, and animal- and plant-derived foods across ReciFine without manually specifying every food in each category, as shown in Table 20.

<table><tr><td rowspan=1 colspan=1>FoodOn Category</td><td rowspan=1 colspan=1>Recipes</td><td rowspan=1 colspan=1>Prevalence</td></tr><tr><td rowspan=1 colspan=1>Plant food product</td><td rowspan=1 colspan=1>1,744,295</td><td rowspan=1 colspan=1>84.59%</td></tr><tr><td rowspan=1 colspan=1>Animal food product</td><td rowspan=1 colspan=1>1,510,986</td><td rowspan=1 colspan=1>73.27%</td></tr><tr><td rowspan=1 colspan=1>Vegetable food product</td><td rowspan=1 colspan=1>1,297,374</td><td rowspan=1 colspan=1>62.92%</td></tr><tr><td rowspan=1 colspan=1>Dairy food product</td><td rowspan=1 colspan=1>1,046,656</td><td rowspan=1 colspan=1>50.76%</td></tr><tr><td rowspan=1 colspan=1>Fruit</td><td rowspan=1 colspan=1>978,283</td><td rowspan=1 colspan=1>47.44%</td></tr><tr><td rowspan=1 colspan=1>Condiment food product</td><td rowspan=1 colspan=1>939,353</td><td rowspan=1 colspan=1>45.55%</td></tr><tr><td rowspan=1 colspan=1>Spice or herb</td><td rowspan=1 colspan=1>742,511</td><td rowspan=1 colspan=1>36.01%</td></tr><tr><td rowspan=1 colspan=1>Sweetener food product</td><td rowspan=1 colspan=1>690,735</td><td rowspan=1 colspan=1>33.50%</td></tr><tr><td rowspan=1 colspan=1>Plant seed or nut food product</td><td rowspan=1 colspan=1>622,813</td><td rowspan=1 colspan=1>30.20%</td></tr><tr><td rowspan=1 colspan=1>Lipid food product</td><td rowspan=1 colspan=1>575,016</td><td rowspan=1 colspan=1>27.88%</td></tr><tr><td rowspan=1 colspan=1>Avian animal food product</td><td rowspan=1 colspan=1>551,230</td><td rowspan=1 colspan=1>26.73%</td></tr><tr><td rowspan=1 colspan=1>Egg food product</td><td rowspan=1 colspan=1>508,606</td><td rowspan=1 colspan=1>24.66%</td></tr><tr><td rowspan=1 colspan=1>Mammalian meat food product</td><td rowspan=1 colspan=1>503,119</td><td rowspan=1 colspan=1>24.40%</td></tr><tr><td rowspan=1 colspan=1>Processed food product</td><td rowspan=1 colspan=1>479,192</td><td rowspan=1 colspan=1>23.24%</td></tr><tr><td rowspan=1 colspan=1>Gravy or sauce</td><td rowspan=1 colspan=1>389,712</td><td rowspan=1 colspan=1>18.90%</td></tr><tr><td rowspan=1 colspan=1>Cereal grain food product</td><td rowspan=1 colspan=1>280,506</td><td rowspan=1 colspan=1>13.60%</td></tr><tr><td rowspan=1 colspan=1>Citrus fruit food product</td><td rowspan=1 colspan=1>264,798</td><td rowspan=1 colspan=1>12.84%</td></tr><tr><td rowspan=1 colspan=1>Herb food product</td><td rowspan=1 colspan=1>257,504</td><td rowspan=1 colspan=1>12.49%</td></tr><tr><td rowspan=1 colspan=1>Food (fermented)</td><td rowspan=1 colspan=1>257,377</td><td rowspan=1 colspan=1>12.48%</td></tr><tr><td rowspan=1 colspan=1>Legume food product</td><td rowspan=1 colspan=1>228,828</td><td rowspan=1 colspan=1>11.10%</td></tr></table>

Table 20: Recipe-level prevalence of 20 selected higherlevel FoodOn categories obtained through ontology hierarchy propagation on the 2,062,103 grounded recipes. Categories were selected for culinary interpretability, diversity, and to reduce redundancy between closely related ontology concepts.

This analysis demonstrates how ontologygrounded entity linking enables individual food mentions to be analysed at broader semantic levels. The resulting category-level information can support further analyses of dietary composition and nutritional patterns and, when combined with external environmental data, the environmental characteristics of recipe ingredients.

## F.3 Food-Category Co-occurrence

Ontology grounding also lets us examine how higher-level food categories co-occur across recipes. For each pair of categories, we count the number of recipes containing both categories and calculate the proportion of recipes containing the first category that also contain the second. For example, this shows how often recipes containing fish also contain vegetables, or how often plant- and animal-derived foods appear in the same recipe, as shown in Table 21.

<table><tr><td rowspan=1 colspan=1>Category A</td><td rowspan=1 colspan=1>Category B</td><td rowspan=1 colspan=1>Recipes</td><td rowspan=1 colspan=1>Proportion</td></tr><tr><td rowspan=1 colspan=1>Plant food product</td><td rowspan=1 colspan=1>Animal food product</td><td rowspan=1 colspan=1>1,303,624</td><td rowspan=1 colspan=1>74.74%</td></tr><tr><td rowspan=1 colspan=1>Animal food product</td><td rowspan=1 colspan=1>Vegetable food product</td><td rowspan=1 colspan=1>969,055</td><td rowspan=1 colspan=1>64.13%</td></tr><tr><td rowspan=1 colspan=1>Plant food product</td><td rowspan=1 colspan=1>Dairy food product</td><td rowspan=1 colspan=1>917,684</td><td rowspan=1 colspan=1>52.61%</td></tr><tr><td rowspan=1 colspan=1>Vegetable food product</td><td rowspan=1 colspan=1>Fruit</td><td rowspan=1 colspan=1>891,116</td><td rowspan=1 colspan=1>68.69%</td></tr><tr><td rowspan=1 colspan=1>Plant food product</td><td rowspan=1 colspan=1>Condiment food product</td><td rowspan=1 colspan=1>831,885</td><td rowspan=1 colspan=1>47.69%</td></tr><tr><td rowspan=1 colspan=1>Animal food product</td><td rowspan=1 colspan=1>Condiment food product</td><td rowspan=1 colspan=1>746,545</td><td rowspan=1 colspan=1>49.41%</td></tr><tr><td rowspan=1 colspan=1>Animal food product</td><td rowspan=1 colspan=1>Fruit</td><td rowspan=1 colspan=1>718,790</td><td rowspan=1 colspan=1>47.57%</td></tr><tr><td rowspan=1 colspan=1>Condiment food product</td><td rowspan=1 colspan=1>Sweetener food product</td><td rowspan=1 colspan=1>690,735</td><td rowspan=1 colspan=1>73.53%</td></tr><tr><td rowspan=1 colspan=1>Vegetable food product</td><td rowspan=1 colspan=1>Dairy food product</td><td rowspan=1 colspan=1>668,829</td><td rowspan=1 colspan=1>51.55%</td></tr><tr><td rowspan=1 colspan=1>Vegetable food product</td><td rowspan=1 colspan=1>Spice or herb</td><td rowspan=1 colspan=1>616,146</td><td rowspan=1 colspan=1>47.49%</td></tr><tr><td rowspan=1 colspan=1>Vegetable food product</td><td rowspan=1 colspan=1>Condiment food product</td><td rowspan=1 colspan=1>588,925</td><td rowspan=1 colspan=1>45.39%</td></tr><tr><td rowspan=1 colspan=1>Animal food product</td><td rowspan=1 colspan=1>Spice or herb</td><td rowspan=1 colspan=1>579,969</td><td rowspan=1 colspan=1>38.38%</td></tr><tr><td rowspan=1 colspan=1>Animal food product</td><td rowspan=1 colspan=1>Sweetener food product</td><td rowspan=1 colspan=1>552,911</td><td rowspan=1 colspan=1>36.59%</td></tr><tr><td rowspan=1 colspan=1>Dairy food product</td><td rowspan=1 colspan=1>Condiment food product</td><td rowspan=1 colspan=1>520,590</td><td rowspan=1 colspan=1>49.74%</td></tr><tr><td rowspan=1 colspan=1>Dairy food product</td><td rowspan=1 colspan=1>Fruit</td><td rowspan=1 colspan=1>508,850</td><td rowspan=1 colspan=1>48.62%</td></tr><tr><td rowspan=1 colspan=1>Fruit</td><td rowspan=1 colspan=1>Condiment food product</td><td rowspan=1 colspan=1>502,324</td><td rowspan=1 colspan=1>51.35%</td></tr><tr><td rowspan=1 colspan=1>Fruit</td><td rowspan=1 colspan=1>Spice or herb</td><td rowspan=1 colspan=1>460,548</td><td rowspan=1 colspan=1>47.08%</td></tr><tr><td rowspan=1 colspan=1>Dairy food product</td><td rowspan=1 colspan=1>Sweetener food product</td><td rowspan=1 colspan=1>396,768</td><td rowspan=1 colspan=1>37.91%</td></tr><tr><td rowspan=1 colspan=1>Dairy food product</td><td rowspan=1 colspan=1>Spice or herb</td><td rowspan=1 colspan=1>388,056</td><td rowspan=1 colspan=1>37.08%</td></tr><tr><td rowspan=1 colspan=1>Fruit</td><td rowspan=1 colspan=1>Sweetener food product</td><td rowspan=1 colspan=1>375,448</td><td rowspan=1 colspan=1>38.38%</td></tr></table>

Table 21: Selected co-occurrence relationships between higher-level FoodOn categories. Proportion denotes the percentage of recipes containing Category A that also contain Category B.

## G Prompt for Shared Food Mention and Ontology Concept Pair Definition Generation

Following the manual definition generation process described in Appendix P, we also generate shared canonical definitions using both open-source and proprietary LLMs to investigate whether automatically generated definitions can effectively be used in place of manually curated ones. Figure 3 illustrates the prompt used to generate these ontologyoriented canonical definitions, which are subsequently compared against manually authored definitions in our experiments.

![](images/c1b49b7fd352d2969b871c0049e8168a3cf4874beac5167d4d4f6201191d1873.jpg)  
Figure 3: Prompt template for generating shared food mention and ontology concept pair definitions.

## H Prompt for Shared Canonical Definition Generation Learning

We design a set of prompts for shared canonical definition generation learning. The objective is to learn a common ontology-oriented definition that can be generated independently from either (i) contextual recipe food mentions or (ii) FoodOn ontology concepts, allowing both representations to converge to the same semantic description. The same prompt templates are used during both training and inference, ensuring consistency between learning and inference. Consequently, the model generates semantically equivalent canonical definitions regardless of whether the input is an informal recipe mention or a formal ontology concept.

## H.1 Prompt for Recipe Food Mention Definition Learning and Generation

Figure 4 shows the prompt used to learn and generate shared canonical definitions from contextual recipe food mentions. The prompt encourages the model to produce an ontology-oriented definition that describes the underlying food concept rather than the food mention’s surface form, enabling semantic alignment with the corresponding FoodOn concept.

![](images/9c92a643e92581465cc8661ac710724fb32a447138d4963491cf05f5f9187475.jpg)  
Figure 4: Prompt template used for shared canonical definition learning and generation from contextual food mentions in SCODEL, and for definition generation from contextual food mentions in LLM4EL.

## H.2 Prompt for FoodOn Concept Definition Learning and Generation

Figure 5 shows the prompt used to learn and generate shared canonical definitions from FoodOn ontology concepts. Given ontology metadata, including concept labels, synonyms, superclasses, and subclasses, the prompt encourages the model to produce the same ontology-oriented canonical definition that would be generated from the corresponding recipe food mention. Together with the recipe food mention prompt (Figure 4), this enables the model to learn a shared canonical definition space across recipe and ontology representations.

## I Model Descriptions

We group the models and experimental configurations according to five roles within the SCODEL evaluation pipeline: (i) paired canonical definition generation, which constructs shared definitions from both of recipe food mention and FoodOn concept pairs; (ii) shared canonical definition learning, in which SCODEL learns to independently generate the same canonical definition from either the contextual mention or ontology concept metadata; (iii) canonical definition generation after learning, where the trained SCODEL model generates definitions independently from the two input views at inference; (iv) modular entity linking models that incorporate the generated definitions for downstream entity linking; and (v) LLM-based entity linking baselines used for comparison.

![](images/e78f17c09054e3de8079bbacc26c732ca421f17fb4c66dbe87f009fa31b251e0.jpg)  
Figure 5: Prompt template used for shared canonical definition learning and generation from FoodOn concept metadata in SCODEL, and for definition generation from FoodOn concept metadata in LLM4EL.

## I.1 Canonical Definition Models

I.1.1 Paired Canonical Definition Generation The main SCODEL experiments use manually authored canonical definitions as supervision for shared canonical definition learning. To evaluate whether this supervision can instead be constructed automatically, we additionally use GPT-5.1-2025- 11-13, Claude Sonnet 5, and Qwen3-235B to generate a single canonical definition from each aligned recipe mention and FoodOn concept pair. These automatically generated definitions provide alternative training targets for SCODEL and are compared with manually authored canonical definitions. The generation configuration is reported in Table 22.

## I.1.2 Shared Canonical Definition Learning

We use LLaMA-3-8B, Mistral-7B, and Qwen2.5- 7B as SCODEL definition-learning backbones. Each model learns to independently generate the same canonical ontology-oriented definition from either a contextual recipe food mention or the metadata of its corresponding FoodOn concept. Depending on the experiment, the shared target is either manually authored or generated automatically using the paired definition generation procedure described above (Appendix I.1.1). The training configuration is reported in Table 23.

## I.1.3 Canonical Definition Generation after Learning

After learning shared definition generation, we apply the SCODEL model independently to the two input views: the contextual recipe food mention and the FoodOn concept metadata. This produces mention-side and concept-side canonical definitions, respectively, which are then incorporated into the downstream entity linking models. The generation configuration is reported in Table 24.

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Models</td><td>GPT-5.1-2025-11-13, Claude Sonnet 5, Qwen3-235B</td></tr><tr><td>Input</td><td>Aligned mention-concept pair</td></tr><tr><td>Output</td><td>Shared canonical definition</td></tr><tr><td>Parameter updates</td><td>None</td></tr></table>

Table 22: Configuration for automatic paired canonical definition generation used to construct alternative SCODEL supervision.

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Base models</td><td>LLaMA-3-8B, Mistral-7B, Qwen2.5-7B</td></tr><tr><td>Finetuning method</td><td>LoRA (Unsloth)</td></tr><tr><td>LoRA rank (r)</td><td>16</td></tr><tr><td>LoRA alpha</td><td>16</td></tr><tr><td>LoRA dropout</td><td>0</td></tr><tr><td>LoRA bias</td><td>None</td></tr><tr><td>LoRA target modules</td><td>q, k, v, o, gate, up, down projections</td></tr><tr><td>4-bit loading</td><td>Yes</td></tr><tr><td>Epochs</td><td>1</td></tr><tr><td>Per-device batch size</td><td>4</td></tr><tr><td>Gradient accumulation</td><td>8</td></tr><tr><td>Effective batch size</td><td>32</td></tr><tr><td>Warmup steps</td><td>5</td></tr><tr><td>Maximum sequence length</td><td>2048</td></tr><tr><td>Optimizer</td><td>AdamW 8-bit</td></tr><tr><td>Gradient checkpointing</td><td>Yes</td></tr><tr><td>Precision</td><td>BF16 if supported; otherwise FP16</td></tr><tr><td>Training seed</td><td>3407</td></tr></table>

Table 23: Training configuration for SCODEL shared canonical definition learning.

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Decoding</td><td>Sampling</td></tr><tr><td>Temperature</td><td>0.7</td></tr><tr><td>Top-p</td><td>0.9</td></tr><tr><td>Maximum new tokens</td><td>1024</td></tr></table>

Table 24: Inference configuration for canonical definition generation after SCODEL training.

## I.2 Modular Entity Linking Models

SCODEL is model-agnostic and is evaluated using three state-of-the-art entity linking models: Re-FinED, GSM-EL and BLINK.

These models were originally designed for general-domain entity linking and therefore required minor adaptations to operate in the recipe ontology entity linking setting (Appendix J). Wherever possible, the original retrieval and ranking architectures were preserved, allowing the impact of SCODEL to be evaluated independently of the downstream entity linking model.

## I.3 LLM-based Entity Linking Baselines

We compare SCODEL against two recent LLMbased entity linking methods: LLM4EL and LL-MAEL.

LLM4EL generates semantic definitions independently for mentions and ontology concepts and performs entity linking by comparing their resulting embedding representations. Unlike SCODEL, these definitions are generated independently in a zero-shot setting and are not learned against a shared canonical definition target. In our implementation, we use Mistral-7B to generate both food mention and ontology concept definitions, then embed the definitions for similarity-based entity linking. We use the prompt templates in Figures 4 and 5 to generate the corresponding mention- and concept-side definitions, respectively.

LLMAEL augments the food mention with LLM-generated contextual information before entity linking, without augmenting the corresponding ontology concept. We use Mistral-7B for context augmentation and ReFinED as the downstream entity linker. The prompt used to generate the foodmention augmentation is provided in Figure 6.

## J Adaptation of Existing Modular Entity Linking Models

ReFinED, GSM-EL, and BLINK were originally developed for general-domain entity linking, primarily against large encyclopedic knowledge bases such as Wikipedia and Wikidata, rather than specialised food ontologies such as FoodOn. To enable fair comparison on ontology-based food entity linking, we retained each model’s original architecture while replacing the underlying knowledge base with FoodOn. Gold food mentions were provided by the benchmark, allowing the mention detection stage to be bypassed for all methods. Candidate entities were constructed directly from FoodOn concepts and their associated ontology metadata (labels, synonyms, hierarchical relations, and canonical definitions where applicable). Apart from these task-specific adaptations, the original

You are an expert in food ontology and culinary knowledge.

Given a food mention and its surrounding recipe context, generate a concise description that captures the semantic meaning of the mention within the current context. The description should help distinguish the food entity from other similar food concepts and provide useful semantic information for downstream entity linking.

The generated description should:

• be factual and contextually grounded;

• be specific enough to distinguish the entity from similar food concepts;

• avoid introducing unsupported information;

• not predict ontology identifiers or ontology concept names;

• contain no more than 60 words.

Return the output as valid JSON in the following   
format:   
{   
"entity": "<food mention>",   
"description": "<generated description>"   
Input   
Mention: ackee   
Context: Drain ackee, stir into the pan   
taking care not to crush it or break it   
apart too much.   
Output   
{   
"entity": "..."   
"description": "   
}  
Figure 6: Prompt template used for generating LL MAEL food mention augmentation description.

retrieval and ranking pipelines of each model remained unchanged.

## J.1 ReFinED Adaptation

The original Wikipedia entity inventory was replaced with the FoodOn ontology. Candidate retrieval was performed over FoodOn concepts using the original dense retrieval framework, while the cross-encoder reranker retained the original Re-FinED architecture. For SCODEL experiments, generated canonical definitions were concatenated with the food mention representation and the ontology concept representation during both retrieval and reranking.

## J.2 GSM-EL Adaptation

The GSM-EL graph was reconstructed using the FoodOn ontology hierarchy instead of Wikipedia hyperlinks. Nodes corresponded to FoodOn concepts, and edges were derived from ontology parent and child relationships. The original graph-aware retrieval and message-passing architecture were preserved. Candidate representations consisted of ontology labels, synonyms, and hierarchical information. For SCODEL variants, generated canonical definitions are appended to both food mention and ontology concept representations.

## J.3 BLINK Adaptation

BLINK’s bi-encoder and cross-encoder architectures were retained without modification. The entity index was rebuilt using FoodOn concepts, replacing the original Wikipedia entity catalogue. Ontology concept embeddings were generated from labels together with available ontology metadata, e.g., synonyms, subclass, superclass, etc. For SCODEL variants, canonical definitions generated for both contextual food mentions and ontology concepts were incorporated into the textual representations before encoding. Meanwhile, candidate retrieval and cross-encoder reranking followed the standard BLINK pipeline.

## K LLM Reranking Entity Linking Implementation

To investigate whether a large language model can improve entity ranking beyond neural retrieval models, we introduce an optional LLM-based reranking stage. Instead of searching the complete ontology, the LLM receives the top-K candidate concepts returned by the underlying entity linker $( K \in \{ 1 0 , 5 0 \} )$ . Each candidate concept includes its ontology label and ontology identifier, along with the contextual food mention extracted from the input text.

The LLM is instructed to select the ontology concept that best matches the food mention in context and to return the identifiers of the most likely candidates ranked from most to least appropriate. This reranking stage does not modify candidate generation; it only reorders the candidates retrieved by the underlying entity linking model. Figure 7 shows the prompt used for the LLM-reranking experiments.

## L Lexical Matching Entity Linking Implementation

To provide a simple non-neural baseline, we implement a lexical entity linking method that ranks FoodOn concepts using only the surface form of the recipe food mention and the lexical forms associated with each candidate concept. Each candidate is represented by its preferred FoodOn concept name together with its available synonyms.

Before matching, we normalise the food mention and candidate ontology concept lexical forms by lowercasing, replacing common separators with spaces, removing punctuation, and collapsing whitespace. Lexical similarity is computed using normalised Levenshtein-based string similarity (Lcvenshtcin, 1966), scaled to [0, 1]. For a food mention m and candidate ontology concept c with lexical forms L(c), the concept score is:

![](images/8a185a9becb2b1c4581bf484028c19453f9d6430435c7bca21d85b5ff0f245f6.jpg)  
Figure 7: Prompt template used for LLM-reranking entity linking experiments.

$$
s ( m , c ) = \operatorname* { m a x } _ { l \in L ( c ) } \sin ( m , l ) ,
$$

where sim denotes the normalised lexical similarity. Candidate concepts are ranked in descending order of this score.

## M Dataset Splits and Evaluation Setup

We construct a training set and two nonoverlapping evaluation sets from the 3,979 entitylinked instances in ReciLinkD: a seen-concept test set for supervised evaluation and an unseenconcept test set for zero-shot evaluation. The splitting procedure operates at both the food mention and ontology concept levels to distinguish generalisation to new surface forms of known ontology concepts from generalisation to entirely unseen ontology concepts. Table 25 summarises the resulting splits.

## M.1 Seen-Concept (Supervised) Evaluation

The seen-concept test set evaluates generalisation to previously unseen recipe food mentions of ontology concepts observed during training. After reserving the zero-shot ontology concepts described below (Appendix M.2), we randomly sampled approximately 18% of the remaining instances for this evaluation. Only concepts with at least two instances are eligible for the test set, and at least one instance of every evaluated concept is retained in training.

<table><tr><td rowspan=1 colspan=1>Statistic</td><td rowspan=1 colspan=1>Train</td><td rowspan=1 colspan=1>Seen</td><td rowspan=1 colspan=1>Unseen</td></tr><tr><td rowspan=1 colspan=1>Food mentions</td><td rowspan=1 colspan=1>2,660</td><td rowspan=1 colspan=1>583</td><td rowspan=1 colspan=1>736</td></tr><tr><td rowspan=1 colspan=1>Unique ontology concepts</td><td rowspan=1 colspan=1>803</td><td rowspan=1 colspan=1>255</td><td rowspan=1 colspan=1>176</td></tr><tr><td rowspan=1 colspan=1>Mean mention-concept pair Jaro similarity</td><td rowspan=1 colspan=1>0.788</td><td rowspan=1 colspan=1>0.758</td><td rowspan=1 colspan=1>0.794</td></tr><tr><td rowspan=1 colspan=1>Exact pair match (%)</td><td rowspan=1 colspan=1>18.98</td><td rowspan=1 colspan=1>11.84</td><td rowspan=1 colspan=1>18.61</td></tr><tr><td rowspan=1 colspan=1>Mean food mention token length</td><td rowspan=1 colspan=1>1.81</td><td rowspan=1 colspan=1>1.83</td><td rowspan=1 colspan=1>1.79</td></tr></table>

Table 25: Dataset split statistics and characteristics. Jaro similarity (Jaro, 1989) is computed between each recipe food mention and the label of its gold FoodOn concept after text normalisation; higher values indicate greater lexical similarity.

The resulting test set contains 583 instances covering 255 FoodOn concepts. All 255 concepts are represented in the training set, while none of the food mention surface forms or exact food mention and ontology concept pairs occurs in training. Thus, the model has previously observed the target ontology concept but not the specific recipe food mention presented at evaluation. This distinction matters because food mentions referring to the same FoodOn concept may use spelling variations, synonyms, alternative surface forms, or expressions in different languages. For example, the FoodOn concept adzuki bean (FOODON\_03315361) may be associated with alternative surface forms such as aduki bean, azuki bean, red bean, and red mung bean.

## M.2 Unseen-Concept (Zero-shot) Evaluation

The zero-shot evaluation measures generalisation to FoodOn concepts for which no aligned supervision is provided during training. We first randomly hold out approximately 18% of the 979 FoodOn concepts represented in ReciLinkD. All instances (food mention and ontology concept pairs) associated with a selected concept are assigned exclusively to the zero-shot test set.

This produces 176 held-out ontology concepts represented by 736 recipe food mentions. None of these 176 ontology concepts occurs in the training set for SCODEL shared canonical definition learning or downstream EL model training. At inference, the held-out ontology concepts and their metadata are restored to the full candidate inventory, requiring the SCODEL model to generate canonical definitions for the food mention and ontology concept pair, and for the downstream entity linking model to identify the ontology concepts for which no initial training examples or canonicaldefinition learning supervision were observed.

## M.3 Split Integrity and Difficulty

We additionally verify the separation between the training and evaluation sets. The seen-concept test set shares all 255 of its target ontology concepts with the training set but does not share any food mention surface form with it. In contrast, the zeroshot test set shares neither target ontology concepts nor food mention surface forms with the training set. In addition, the two evaluation sets are also concept-disjoint.

The seen and zero-shot sets exhibit broadly comparable textual characteristics: food mentions average 1.83 and 1.79 tokens, respectively, while their recipe contexts average 16.24 and 16.39 tokens, respectively.

## N Evaluation Metrics

We evaluate entity linking performance using ranking and ontology-aware metrics. Hit@1 is the primary entity linking accuracy measure, while the remaining metrics provide complementary information about ranking quality and hierarchical proximity.

## N.1 Hit@1

Hit@1 measures the proportion of test instances for which the highest-ranked predicted ontology concept exactly matches the gold concept:

$$
\mathrm { H i t @ 1 } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } [ \hat { c } _ { i } = c _ { i } ] ,
$$

where $c _ { i }$ and $\hat { c } _ { i }$ denote the gold and top-ranked predicted concepts, respectively.

## N.2 Hit@5

Hit@5 measures the proportion of test instances for which the gold ontology concept occurs within the five highest-ranked predictions:

$$
\mathrm { H i t @ 5 } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } [ r _ { i } \leq 5 ] ,
$$

where $r _ { i }$ is the rank of the gold concept.

## N.3 Mean Reciprocal Rank (MRR)

MRR measures ranking quality based on the position of the gold concept:

$$
\mathrm { M R R } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { r _ { i } } ,
$$

where the reciprocal rank is defined as zero when the gold ontology concept is absent from the candidate list.

## N.4 Hierarchy-aware Accuracy (H-Acc)

H-Acc complements exact-match evaluation by assigning partial credit when the top-ranked prediction is hierarchically related to the gold FoodOn concept:

$$
\mathrm { H - A c c } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } h ( \hat { c } _ { i } , c _ { i } ) ,
$$

where

$$
h ( \hat { c } _ { i } , c _ { i } ) = \left\{ \begin{array} { l l } { 1 , } & { \hat { c } _ { i } = c _ { i } , } \\ { 0 . 7 5 , } & { \hat { c } _ { i } \mathrm { ~ i s ~ a ~ d i r e c t ~ p a r e n t ~ o r ~ c h i l d ~ o f ~ } c _ { i } , } \\ { 0 . 5 0 , } & { \hat { c } _ { i } \mathrm { ~ i s ~ a n o t h e r ~ a n c e s t o r ~ o r ~ d e s c e n d a n t ~ o f ~ } c _ { i } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

The metric distinguishes direct from non-direct hierarchical relations but does not otherwise decay with path length or impose a maximum hierarchy distance; consequently, multiple paths between two concepts do not affect the score.

## O Guideline for Manual Recipe Food Mention Linking

The annotation task links a food mention in a recipe to the most appropriate concept from a predefined subset of the FoodOn ontology. Annotators are provided with the food mention, its surrounding recipe context, and a candidate set of 1,014 FoodOn concepts. The task is to select the ontology concept whose meaning best matches the food mention in context.

## O.1 Information Provided

For each annotation instance, annotators are presented with:

• The food mention to be linked.

• The recipe context in which the food mention appears.

• A list of 1,014 candidate FoodOn concepts, each accompanied by its concept label and definition.

## O.2 Annotation Procedure

For each annotation instance, annotators should:

1. Carefully read the food mention and its surrounding recipe context.

2. Compare the meaning of the mention with the candidate FoodOn concepts.

3. Select the single ontology concept whose meaning most closely matches the food mention.

If no suitable concept exists among the candidate concepts, the annotation should be left blank.

Annotators were instructed to follow the guidelines below:

• Base decisions on semantic meaning rather than lexical similarity alone.

• Use the surrounding recipe context to resolve ambiguous food mentions.

• Select the closest and most specific ontology concept whenever possible.

• Leave the annotation blank only if none of the 1,014 candidate concepts adequately represent the food mention.

• When annotators are unfamiliar with a food item, they may consult reliable online resources (e.g., FoodOn, Wikipedia, or other reputable food references) before deciding.

## P Guideline for Manual Shared Definition Generation

For each annotated food mention and ontology concept pair, annotators were additionally asked to write a shared canonical definition describing the underlying food concept represented by both the contextual recipe mention and the corresponding FoodOn concept.

Annotators were instructed to follow the guidelines below:

• Write a single ontology-style definition of the underlying food concept rather than describing the recipe food mention or ontology concept label independently.

• Keep the definition concise, with a maximum length of 60 words.

• Describe intrinsic food properties where appropriate, such as type, composition, origin, biological source, or common culinary characteristics.

• Avoid copying or referring directly to the recipe context, FoodOn labels, ontology metadata, or synonyms.

• Produce definitions that could apply equally to both the recipe mention and the corresponding FoodOn concept.

• When annotators were unfamiliar with a food item, the ontology concept, or the context, they were encouraged to consult reliable online resources before writing the definition.