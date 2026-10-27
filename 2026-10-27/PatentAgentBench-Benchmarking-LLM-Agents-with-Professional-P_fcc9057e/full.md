# PatentAgentBench: Benchmarking LLM Agents with Professional Patent Task Suites

§ PatentAgentBench PatentAgentBench

Yijie Hu, Wei Li, Junjie Zhou, Zhiqi Jiang, Yi Jin Licong Xu, Xinkai Sun, Kai Duan<sup>‡</sup>, Lisha Zhang<sup>†,‡</sup>

AI Eval & Alignment, PatSnap Ltd. Co.

<sup>‡</sup>project lead <sup>†</sup>corresponding author

## Abstract

Patent practice lies at the intersection of technical, legal, and linguistic work, making it challenging yet crucial to technological innovation. Recently, rapid advances in large language models (LLMs) and agentic systems have demonstrated the potential to complete realistic business workflows and even patent works. This raises a question: can current AI systems perform patent work reliably? Existing benchmarks cover a limited range of patent tasks and simplify evidence acquisition by providing reference materials or candidate documents. In this paper, we introduce PatentAgentBench, a benchmark comprising 510 instances across eight task suites: novelty search, freedom-to-operate (FTO) search, design FTO, invalidity search, claim charting, patent drafting, office-action response, and patent translation. Curated from real patent records and professional materials, the benchmark combines task-specific metrics with expert-reviewed annotations and rubrics. Retrieval tasks such novelty search, FTO, etc., require agents to search and compare patent collections without a supplied candidate pool. The drafting and OAR tasks assess the quality of agents’ work given technical disclosure and office action notice. We evaluate seven LLMs across four agent harnesses on PatentAgentBench, finding that current systems still struggle to identify novelty, assess risk, and understand practitioner intent throughout patent workflows. The strongest model–harness pair achieves an average score of 52.8, leaving substantial room for improvement across professional patent tasks. Our analysis demonstrates that current agents struggle to retrieve relevant evidence and produce substantively sound patent documents. Although patent drafts generally follow the expected structure, weaknesses persist in claim organization, case understanding, and legal argumentation. These findings highlight the gap between generating plausible documents and delivering reliable patent work.

![](images/41445db1796aaf9a166ce3649d604428baa26a4d358e9442d627163b9c5f7577.jpg)  
Figure 1: Top twelve model–harness pairs ranked by mean score across the eight task suites.

## Contents

Introduction 4   
2 Related Works   
2.1 LLM Agents   
2.2 Patent Practice Benchmarks   
3 PatentAgentBench   
3.2 Task Taxonomy   
4 Experiments <sup>9</sup><sub>9</sub>   
4.1 Experimental Settings and Metrics   
4.2 Experimental Results 9   
4.3 In-depth Analysis 10   
5 Case Study 12   
6 Conclusion 14   
7 Limitations 14   
A Experimental Setting Details 18   
A.1 Agent Harnesses 18   
A.2 Models and Execution Protocol 18   
A.3 Tools and Judge Model 18   
B Expert Team Profile 19   
C Dataset Details 19   
D Evaluation Metric Definitions 20   
D.1 Notation and Aggregation   
D.2 Novelty Search   
D.3 Patent FTO   
D.4 Design FTO   
D.5 Invalidity Search   
D.6 Claim Charting   
D.7 Patent Drafting   
D.8 Office Action Response   
D.9 Patent Translation   
E Task Completion Rates 24   
F Detailed Task Results   
F.1 Novelty Search   
F.2 Patent FTO   
F.3 Design FTO   
F.4 Invalidity Search   
F.5 Claim Charting   
F.6 Patent Drafting   
F.7 Office-Action Response   
F.8 Patent Translation: ZH–EN   
F.9 Patent Translation: EN–ZH   
G Qualitative Case Comparisons   
G.1 Novelty Search   
G.2 Patent FTO   
G.3 Design FTO 40   
G.4 Invalidity Search 41   
G.5 Claim Charting 42   
G.6 Patent Drafting 43   
G.7 Office-Action Response 44   
H Execution Failure Cases 45

I Task Prompts

## 1 Introduction

Patents play a vital role in protecting inventions, facilitating technology transfer, and providing legal certainty. Patent practice consists of a series of important tasks. For example, a technical idea must be first assessed through novelty search, evaluated in freedom-tooperate analyses or invalidity proceedings (Krestel et al., 2021), then developed into a patent draft, followed by examination and rebuttal before final publication (Srinivas et al., 2024). Patent practice is therefore both knowledge and labor-intensive, requiring coordinated technical, legal, and linguistic expertise, while remaining essential to protecting innovation and managing technology-related risks.

Recently, the advancements in Large Language Models (LLM) and agentic systems (Liu et al., 2024; Singh et al., 2025; Comanici et al., 2025; Zeng et al., 2026; Team et al., 2026; Xu et al., 2026) have demonstrated that agentic systems (Luo et al., 2025) possess long-horizon working abilities, delivering high-value tasks such as coding and legal tasks (Yang et al., 2024; Merrill et al., 2026; Harvey AI, 2026). These promising advancements demonstrate the potential to accelerate or even automate patent work. This raises a question: Can prevalent LLM agents reliably perform patent-related tasks?

Previous efforts that evaluate patent-related capabilities are mainly through examination questions, classification, and output evaluation (Wang et al., 2024; Lee et al., 2024; Knappich et al., 2025; Jiang et al., 2025; Wang et al., 2025; Jang et al., 2025), as summarized in Table 1. These benchmarks provide valuable measures of patent knowledge, reasoning, and selected generation capabilities for LLMs, yet leave two gaps in the evaluation of professional patent work. First, coverage of professional tasks remains limited. For example, IPBench (Wang et al., 2025) covers diverse IP topics but primarily uses multiple-choice and classification tasks, with generation limited to abstracts and dependent claims. PANORAMA (Lim et al., 2026) focuses on prior-art selection, evidence localization, and patentability assessment within examination. Broader tasks, including FTO search, invalidity search, and patent translation, remain unexplored. Second, the evidence acquisition procedure is often over-simplified by supplying case materials or pre-constructed candidate references. This alleviates the need for systems to formulate queries, retrieve documents, and refine their search as new evidence emerges. Consequently, the performance under these settings does not establish whether a system can independently obtain and use the evidence needed to complete a professional patent task.

In this paper, we introduce PatentAgentBench, a benchmark for evaluating LLM agents on eight professional patent tasks: novelty search, freedom-to-operate (FTO) search, design FTO, invalidity search, claim charting, patent drafting, office-action response (OAR), and patent translation. We illustrate some of the representative tasks in Fig. 2, covering evidence discovery, claim-level analysis, and professional document production across patent practice. The retrieval-related task suites require agents to formulate queries, search patent collections, and identify relevant documents without a supplied candidate pool. We construct the benchmark through a four-stage process that combines expert guidance with automated data collection, case reconstruction, and final validation. Cases are constructed from actual patent records, official proceedings, and professional work materials. We identify the task inputs and reference outcomes within each source record, then reconstruct the inputs available to an agent while withholding reference answers. Domain experts are involved to validate the completeness of the materials and annotate task-specific reference evidence or scoring criteria. Each instance specifies the permitted evidence, required output, and evaluation protocol, with reference evidence linked to its source. Evaluation combines retrieval and extraction metrics with expert-reviewed rubrics to assess evidence coverage, factual support, and the quality of the final work product.

We evaluate seven prevalent LLMs across four agent harnesses, comparing models under a fixed harness and harnesses with a fixed model. The strongest pairing achieves an average score of 52.8, leaving substantial room for improvement across professional patent tasks. Detailed analysis further reveal gaps beyond the aggregate scores: under a fixed harness such as Claude Code, all seven models recover less than one-third of annotated references in task such as invalidity search, and majority of them perform poor on evidence alignment than on feature extraction. Document-level analysis reveals a gap between completing a document and ensuring its substantive quality. Patent drafts follow the expected structure but lack meaningful layers of dependent-claim protection. In office-action response, high completion and compliance scores coexist with weaknesses in understanding the case, choosing a response strategy, and developing well-supported arguments. These results reveal that current LLM agents still cannot perform these patent-related tasks reliably, especially in evidence acquisition, evidence-grounded reasoning, and deliver high quality patent documents. We summarize our contributions as follows:

![](images/243bdad79f1611ec8bc842205dcf490cc85ad5ff033ca6ef8ce6c4bda31a118f.jpg)  
Figure 2: Representative tasks of PatentAgentBench. The search-related tasks, such as novelty search, ask the agent to retrieve evidence and results without the given references. Drafting and OAR tasks ask the agent to do research based on the given material, then deliver reliable patent drafting or rebuttal.

• We introduce PatentAgentBench, a benchmark covering eight professional patent tasks, with cases constructed from actual patent records, official proceedings, and professional work materials.

• We propose task-specific evaluation metrics that combine reference-based metrics with expert-reviewed rubrics, assessing evidence retrieval, factual support, and the substantive quality of patent work products.

• We evaluate seven LLMs across four agent harnesses and analyze performance from both perspectives. Our results reveal task-dependent model–harness interactions and persistent gaps in evidence coverage, evidence alignment, and substantive document quality.

## 2 Related Works

## 2.1 LLM Agents

Research on LLM-based agents has progressed from general formulations of reasoning and action to autonomous cooperation, multi-model coordination, multi-agent frameworks, and generalist orchestration (Yao et al., 2022; Tran et al., 2025). Early work characterizes agents as systems that perceive their environment, make decisions, and take actions (Shinn et al., 2023). CAMEL (Li et al., 2023) enables autonomous cooperation through role-playing, while

Table 1: Coverage of professional patent tasks in existing benchmarks.
<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">Evaluated Tasks</td><td rowspan="2">Novelty Search</td><td rowspan="2">Patent FTO</td><td rowspan="2">Design FTO</td><td rowspan="2">Invalidity Search</td><td rowspan="2">Claim Charting</td><td rowspan="2">Patent Drafting</td><td rowspan="2">OA Response</td><td rowspan="2">Patent Translatior</td></tr><tr><td></td></tr><tr><td>IPEval (2024)</td><td>IP, legal knowledge QA</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>PatentEdits (2024)</td><td>Claim-sentence edit prediction</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>PEDANTIC (2025)</td><td>Claim identification</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Patent-CE (2025)</td><td>Patent claim quality assessment</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>IPBench (2025)</td><td>Patent search and drafting</td><td>J</td><td>x</td><td>x</td><td>x</td><td>x</td><td>√</td><td>x</td><td>x</td></tr><tr><td>PANORAMA (2026)</td><td>Novelty search and claim charting</td><td>1</td><td>x</td><td>x</td><td>x</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>PILOT-Bench (2025)</td><td>Legal reasoning and outcome prediction</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>PatRe (2026)</td><td>Office action and response generation</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>√</td><td>x</td></tr><tr><td>PatentAgentBench</td><td></td><td>J</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

HuggingGPT (Shen et al., 2023) uses an LLM to plan tasks and coordinate external AI models across multiple modalities. AutoGen (Wu et al., 2024) extends this direction by providing a general framework for interactions among LLMs, humans, and tools. More recent systems move toward generalist orchestration: Magentic-One (Fourney et al., 2024) coordinates specialized agents operating over browsers, files, and code, while SuperAgent (Yao et al., 2025) routes user requests to tool-equipped agents or generated workflows and selects models according to task complexity. Together, these studies trace a shift from individual LLM agents to coordinated systems that complete complex workflows through a unified interface. PatentAgentBench evaluates the final professional work products of agentic systems across eight representative patent-practice task suites.

## 2.2 Patent Practice Benchmarks

Prior research has studied patent analysis through classification, retrieval, quality assessment, and text generation (Shomee et al., 2025). IPEval (Wang et al., 2024) evaluates IPrelated knowledge and legal reasoning using bilingual questions drawn from patent-agent examinations. IPBench further expands the scope to 20 tasks (Wang et al., 2025). However, most IPBench instances remain closed-set questions or classification problems, patent generation tasks are limited to patent abstracts and dependent claims. PatentEdits (Lee et al., 2024) asks model to predict whether claim will be kept, edited, or deleted during prosecution without revisions. PEDANTIC (Knappich et al., 2025) evaluates claim definiteness and the identification of rejection reasons. Patent-CE (Jiang et al., 2025) evaluates the quality of candidate claims against reference claims. PILOT-Bench (Jang et al., 2025) formulates PTAB appeal reasoning as independent classification tasks for issues, authorities, and outcomes, examining the legal reasoning capabilities. More recent work moves closer to prosecution practice. PANORAMA (Lim et al., 2026) evaluates prior-art selection, evidence localization, and patentability classification as separate closed-pool tasks. PatRe (Wang et al., 2026) generates office actions and applicant responses from prosecution records, although its retrieval-simulated setting supplies a small candidate pool containing oracle references. In contrast, PatentAgentBench evaluates complete, evidence-grounded work products across nine representative patent-practice task suites, covering invention intake, patentability analysis, clearance, validity analysis, evidence mapping, drafting, prosecution, and multilingual practice under a unified evaluation framework.

## 3 PatentAgentBench

We first demonstrate the general data curation pipeline in Section 3.1, each sub-task follows the similar data curation process with small variations, we will introduce them separately in the section in Section 3.2.

![](images/afefc99b4d38bdfd7d855c4fd26e33c62264c1085132af0b3891e703ed04c15b.jpg)  
Figure 3: Four-stage data curation pipeline.

## 3.1 Data Curation

PatentAgentBench follows a four-stage curation pipeline that combines automated processing with expert review, as illustrated in Fig. 3. First, we define each task’s input, expected deliverable item, task evidence, and evaluation criteria based on professional patent practice. Second, experts establish inclusion and exclusion criteria to guide rule-based and LLMassisted screening of patent records, official proceedings, and professional work materials. Our expert team comprises 31 professionals: five senior patent examiners, fifteen senior patent agents, three dual-qualified lawyers, and eight senior IPR specialists. Their detailed backgrounds are reported in Appendix B. Experts review candidate cases for relevance, completeness, and suitability. Third, we parse and normalize the selected materials, reconstruct task inputs, and derive reference evidence or scoring criteria from the source records. Automated pre-annotation supports this process, while experts correct labels and review rubrics for factual support and professional relevance. Each instance is annotated by one expert and reviewed by a second expert. Finally, each case is packaged with its input, evidence conditions, output requirements, and evaluation anchors. Before final validation, we check that reference evidence is traceable to its source and that task inputs do not expose the answers.

## 3.2 Task Taxonomy

PatentAgentBench covers eight task suites spanning evidence discovery, feature-level comparison, and document preparation. Novelty search, patent FTO, design FTO, and invalidity search require agents to retrieve relevant evidence. Claim charting requires them to map technical features to supplied evidence.Patent drafting, office action response, and translation require them to produce professional documents. Table 2 summarizes the dataset statistics, see more details in Appendix 5. We describe the purpose, task specification, and data construction of each suite below.

Novelty Search. Novelty search identifies prior art relevant to assessing whether a technical solution is new. The agent receives a technical specification and returns a ranked list of prior patents. We construct the ground truth from patent families examined by multiple offices. For each family, we collect the X references cited by examiners for the target patent and its family mem-

Table 2: Statistics of PatentAgentBench across eight task suites.
<table><tr><td>Subset</td><td>Size</td><td>Jurisdiction</td><td>Metrics</td></tr><tr><td>Novelty Search</td><td>80</td><td>CN/US/EP/WO</td><td>Hit Rate/Recall</td></tr><tr><td>FTO</td><td>60</td><td>CN/US/EP</td><td>Recall</td></tr><tr><td>Design FTO</td><td>80</td><td>CN/EU</td><td>Recall</td></tr><tr><td>Invalidity Search</td><td>80</td><td>CN/US</td><td>Recall</td></tr><tr><td>Claim Charting</td><td>30</td><td>CN/US/WO</td><td>F1</td></tr><tr><td>OAR</td><td>40</td><td>CN/US/EP</td><td>57 Rubrics</td></tr><tr><td>Patent Drafting</td><td>60</td><td>CN/US/EP</td><td>977 Rubrics</td></tr><tr><td>Translation</td><td>80</td><td>CN/US/EP/WO1</td><td>BLEU/METEOR/ROUGE</td></tr><tr><td>Total</td><td>510</td><td></td><td></td></tr></table>

## bers,then merge them into a

deduplicated reference set. The input contains only the technical disclosure, with citation information excluded to prevent leakage. The 80 cases cover Chinese and English and are balanced across IPC main classes and patent offices.

Patent FTO. Freedom-to-operate (FTO) search identifies claims that may block the implementation of a technical solution or product by systematically searching granted, valid patents in a target jurisdiction. The agent receives a technical description and the target jurisdiction, and returns a screened list of potentially blocking patents. We construct cases from court records with established infringement findings: the accused product’s technical features form the input, and the patents held to be infringed form the reference set. The cases cover CN, US, and EP jurisdictions. This suite requires agents to consider claim scope and legal status when identifying infringement risks, rather than relying on technical similarity alone.

Design FTO. Design FTO identifies design patents that may pose infringement risks to a product’s appearance. The agent receives a target image and jurisdiction, and returns a ranked list of design patents with similarity scores and risk levels. We construct 80 cases from final invalidation decisions issued by the relevant offices, using the invalidated design patents and evidence images from these cases. The cases span 26 first-level Locarno classes. The task requires agents to extract the target design features, compare them with retrieved design images, and assess infringement risks under the rules of the target jurisdiction. We evaluate retrieval coverage against the annotated reference patents.

Invalidity Search. Invalidity search retrieves prior art that can support a challenge to granted patent claims. The agent receives the target patent’s application and publication numbers, filing date, and earliest priority date. It retrieves and analyzes the patent, then returns prior-art references with the relevant novelty or inventive-step grounds, affected claims, and brief explanations. We construct cases from official invalidation decisions and final written decisions in inter partes review. We retain one representative granted patent per case and deduplicate cases at the application level. Alongside the cited references, we preserve the statutory provisions, reasoning, evidence combinations, and affected claims, so that retrieved evidence can be checked against the grounds established in the decision.

Claim Charting. Claim charting maps individual technical features to supporting evidence for novelty assessment, FTO analysis, and invalidity proceedings. Given a target text and comparison documents, the agent must identify the target’s technical features, determine whether each is disclosed, and cite the corresponding evidence with an explanation. We construct cases from Chinese and English patents and technical texts covering IPC sections A, B, C, G, and H. External patent experts annotate feature-level disclosure judgments, word-level evidence mappings, and explanations. We cross-check and standardize these annotations so that each judgment is traceable to specific evidence.

Patent Drafting. Patent drafting turns an invention disclosure into an initial patent application. The agent receives disclosure materials and produces a complete draft, including claims and a specification. We construct cases from real patents, anonymize confidential information, and ask patent experts to review the disclosure materials and develop evaluation rubrics. The cases cover CN, US, and EP jurisdictions and IPC sections A–H. General rubrics assess language quality, document structure, claim coverage, and consistency between the claims and specification. Case-specific rubrics assess the technical features, relationships, and constraints in each disclosure. Together, they evaluate whether the draft clearly describes the invention, preserves its essential features, and supports the claimed scope without introducing unsupported content.

Office Action Response. An office action sets out an examiner’s objections to a patent application. Responding requires addressing these objections through technical and legal arguments and, where appropriate, amendments. The agent receives the office action, jurisdiction, and available application materials and cited references, and produces a formal response. We collect 40 cases from CN, US, and EP patent examinations. Patent experts review the case materials and response requirements, including the objections raised, evidence needed to support an argument, and disclosure supporting an amendment. Evaluation uses 57 rubric criteria covering case understanding, response strategy, amendment quality, substantive argumentation, completeness, and factual and citation accuracy. Each criterion specifies its applicability and scoring levels.

Patent Translation. Patent translation supports filing and examination across languages while preserving technical meaning and claim scope. We include 80 document-level translation cases with 881 word, character, sentence and paragraph-level items, supporting the category-specific evaluation. The agent translates Chinese patent text into English or English text into Chinese without adding, omitting, or altering technical content. We organize translation pairs at character, word, sentence, paragraph, and document levels, covering titles, abstracts, claims, and descriptions. Each instance is labeled by text granularity and evaluation dimension, with reference translations or target expressions for terminology, special characters, and patent writing conventions.

## 4 Experiments

## 4.1 Experimental Settings and Metrics

We evaluate seven LLMs through four agent harnesses, comparing models within each harness and harnesses for each model. The models are GPT-5.6-Sol, Qwen3.8-Max, Kimi K3, DeepSeek-V4-Pro, MiniMax-M3, GLM-5.3, and Claude Opus 5. The harnesses are Codex, Claude Code, Openclaw, and Hermes. Agents can call a commercial patent search MCP service to access to the patent database. We use GPT 6 sol as the LLM judge for rubricbased evaluation. More detailed Harness descriptions and versions, model names, and the execution protocol are in Appendix A. Claude Opus 5 was not evaluated with Codex because the model was incompatible with the protocol supported by Codex configuration.

Novelty search uses Hit@100, the proportion of cases with at least one annotated reference retrieved among the top 100 results. Patent FTO, design FTO, and invalidity search use recall to measure coverage of annotated references. Claim charting uses F1 score. Patent drafting and OAR use weighted rubric-based scores. OAR is assessed using 57 criteria covering basic quality, amendment quality, and substantive argumentation. Translation uses category-specific metrics, averaged across the two translation directions. All primary task scores in Table 3 are expressed on a 0–100 scale, and AVG is their mean score. Appendix D defines the matching rules, formulas, rubric weights, and aggregation for each task.

## 4.2 Experimental Results

We demonstrate the results in Table 3. Claude Opus 5 with Hermes achieves the highest average score of 52.8, followed by the same model with Claude Code at 50.3. If the harness is held constant, such as Hermes, Claude Opus 5 outperforms GPT-5.6-Sol on invalidity search, claim charting, and drafting, while GPT performs better on novelty search and OAR. A shared weakness is Design FTO, where recall remains below 4 across every evaluated configuration. We find that the ability to extract visual features and search for similar designs is not well supported by the current LLMs and harnesses. High recall on textual FTO does not extend to this visual search setting, making design-related evidence discovery a persistent gap.

The results show that neither model rankings nor harness advantages transfer consistently across configurations and tasks. GLM-5.3 outperforms GPT-5.6-Sol under Codex, with average scores of 48.1 and 43.8, but the ordering reverses under OpenClaw, where GPT scores 50.0 and GLM scores 29.5. Model comparisons therefore depend on the harness used to run them. Harness advantages are also task-specific: switching Opus from Claude Code to Hermes raises invalidity-search recall from 28.7% to 55.0%, a gain of 26.3 percentage points, while leaving FTO, claim-charting, and drafting scores comparatively similar. This gain establishes a strong pairing for invalidity search, rather than a general Hermes advantage across patent tasks. GPT-5.6-Sol performs comparably with OpenClaw and Claude Code, with average scores of 50.0 and 49.3, respectively, while Hermes yields its highest OAR score. Consequently, the highest-ranked configuration overall is not necessarily the strongest choice for an individual workflow. Task-level comparisons are necessary to identify the appropriate model–harness configuration for a given patent task. More detailed experimental results and analysis are provided in Appendix F.

Table 3: Main results on PatentAgentBench. Scores are on a 0–100 scale. Higher is better. AVG is the mean task score, with translation averaged over both directions. Within each harness, darker and lighter blue cells mark the highest and second-highest positive scores, respectively. Ties at the displayed precision share the same color.
<table><tr><td rowspan=1 colspan=10>NoveltyPatentDesignInvalidity Claim  Patent Office Action  PatentHarness    Model                                                                                    AVGSearch FTO  FTO   Search ChartingDrafting  Response  Translation</td></tr><tr><td rowspan=1 colspan=1>Codex      GPT-5.6-Sol</td><td rowspan=1 colspan=2>27.5   54.2</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>26.4</td><td rowspan=1 colspan=1>53.3</td><td rowspan=1 colspan=2>84.7      34.8</td><td rowspan=1 colspan=1>69.0</td><td rowspan=1 colspan=1>43.8</td></tr><tr><td rowspan=1 colspan=1>Qwen3.8-Max</td><td rowspan=1 colspan=2>51.3   55.5</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>31.2</td><td rowspan=1 colspan=1>54.9</td><td rowspan=1 colspan=1>31.7</td><td rowspan=1 colspan=1>12.4</td><td rowspan=1 colspan=1>69.1</td><td rowspan=1 colspan=1>38.3</td></tr><tr><td rowspan=1 colspan=1>Kimi K3</td><td rowspan=1 colspan=2>13.8   5.7</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>4.6</td><td rowspan=1 colspan=1>48.2</td><td rowspan=1 colspan=1>67.7</td><td rowspan=1 colspan=1>6.5</td><td rowspan=1 colspan=1>70.5</td><td rowspan=1 colspan=1>27.1</td></tr><tr><td rowspan=1 colspan=1>DeepSeek-V4-Pro</td><td rowspan=1 colspan=2>10.0   43.2</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>10.3</td><td rowspan=1 colspan=1>9.3</td><td rowspan=1 colspan=1>58.2</td><td rowspan=1 colspan=1>6.4</td><td rowspan=1 colspan=1>70.5</td><td rowspan=1 colspan=1>26.0</td></tr><tr><td rowspan=1 colspan=1>MiniMax-M3</td><td rowspan=1 colspan=2>15.0   8.3</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.8</td><td rowspan=1 colspan=1>30.3</td><td rowspan=1 colspan=1>17.9</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>67.9</td><td rowspan=1 colspan=1>17.5</td></tr><tr><td rowspan=1 colspan=1>GLM-5.3</td><td rowspan=1 colspan=1>86.3</td><td rowspan=1 colspan=1>74.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>33.8</td><td rowspan=1 colspan=1>24.8</td><td rowspan=1 colspan=2>74.5      21.0</td><td rowspan=1 colspan=1>70.1</td><td rowspan=1 colspan=1>48.1</td></tr><tr><td rowspan=1 colspan=1>Claude CodeGPT-5.6-Sol</td><td rowspan=1 colspan=1>38.8</td><td rowspan=1 colspan=1>79.4</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>31.8</td><td rowspan=1 colspan=1>54.1</td><td rowspan=1 colspan=1>84.0</td><td rowspan=1 colspan=1>38.0</td><td rowspan=1 colspan=1>68.5</td><td rowspan=1 colspan=1>49.3</td></tr><tr><td rowspan=1 colspan=1>Claude Opus 5</td><td rowspan=1 colspan=1>52.5</td><td rowspan=1 colspan=1>86.5</td><td rowspan=1 colspan=1>2.7</td><td rowspan=1 colspan=1>28.7</td><td rowspan=1 colspan=1>54.0</td><td rowspan=1 colspan=1>82.8</td><td rowspan=1 colspan=1>25.1</td><td rowspan=1 colspan=1>70.2</td><td rowspan=1 colspan=1>50.3</td></tr><tr><td rowspan=1 colspan=1>Qwen3.8-Max</td><td rowspan=1 colspan=1>32.5</td><td rowspan=1 colspan=1>86.2</td><td rowspan=1 colspan=1>3.8</td><td rowspan=1 colspan=1>21.0</td><td rowspan=1 colspan=1>50.7</td><td rowspan=1 colspan=1>80.8</td><td rowspan=1 colspan=1>16.4</td><td rowspan=1 colspan=1>70.2</td><td rowspan=1 colspan=1>45.2</td></tr><tr><td rowspan=1 colspan=1>Kimi K3</td><td rowspan=1 colspan=1>33.8</td><td rowspan=1 colspan=1>83.2</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>25.8</td><td rowspan=1 colspan=1>31.8</td><td rowspan=1 colspan=1>82.9</td><td rowspan=1 colspan=1>12.2</td><td rowspan=1 colspan=1>71.3</td><td rowspan=1 colspan=1>42.6</td></tr><tr><td rowspan=1 colspan=1>DeepSeek-V4-Pro</td><td rowspan=1 colspan=1>37.5</td><td rowspan=1 colspan=1>78.2</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>24.9</td><td rowspan=1 colspan=1>40.8</td><td rowspan=1 colspan=1>43.0</td><td rowspan=1 colspan=1>22.3</td><td rowspan=1 colspan=1>70.7</td><td rowspan=1 colspan=1>39.7</td></tr><tr><td rowspan=1 colspan=1>MiniMax-M3</td><td rowspan=1 colspan=1>27.5</td><td rowspan=1 colspan=1>72.5</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>12.4</td><td rowspan=1 colspan=1>21.1</td><td rowspan=1 colspan=1>72.4</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>67.3</td><td rowspan=1 colspan=1>34.2</td></tr><tr><td rowspan=1 colspan=1>GLM-5.3</td><td rowspan=1 colspan=1>50.0</td><td rowspan=1 colspan=1>83.9</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>32.4</td><td rowspan=1 colspan=1>5.5</td><td rowspan=1 colspan=1>82.3</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>71.4</td><td rowspan=1 colspan=1>40.7</td></tr><tr><td rowspan=1 colspan=1>Hermes     GPT-5.6-Sol</td><td rowspan=1 colspan=1>62.5</td><td rowspan=1 colspan=1>62.7</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>10.3</td><td rowspan=1 colspan=1>12.7</td><td rowspan=1 colspan=1>56.7</td><td rowspan=1 colspan=1>54.8</td><td rowspan=1 colspan=1>69.1</td><td rowspan=1 colspan=1>41.1</td></tr><tr><td rowspan=1 colspan=1>Claude Opus 5</td><td rowspan=1 colspan=1>52.5</td><td rowspan=1 colspan=1>86.4</td><td rowspan=1 colspan=1>1.9</td><td rowspan=1 colspan=1>55.0</td><td rowspan=1 colspan=1>55.1</td><td rowspan=1 colspan=1>81.0</td><td rowspan=1 colspan=1>20.4</td><td rowspan=1 colspan=1>70.3</td><td rowspan=1 colspan=1>52.8</td></tr><tr><td rowspan=1 colspan=1>Qwen3.8-Max</td><td rowspan=1 colspan=1>17.5</td><td rowspan=1 colspan=1>80.5</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>20.5</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>42.7</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>69.3</td><td rowspan=1 colspan=1>28.9</td></tr><tr><td rowspan=1 colspan=1>Kimi K3</td><td rowspan=1 colspan=1>36.3</td><td rowspan=1 colspan=1>82.7</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>34.5</td><td rowspan=1 colspan=1>3.2</td><td rowspan=1 colspan=1>73.2</td><td rowspan=1 colspan=1>20.4</td><td rowspan=1 colspan=1>69.7</td><td rowspan=1 colspan=1>40.0</td></tr><tr><td rowspan=1 colspan=1>DeepSeek-V4-Pro</td><td rowspan=1 colspan=1>22.5</td><td rowspan=1 colspan=1>61.2</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>29.9</td><td rowspan=1 colspan=1>8.3</td><td rowspan=1 colspan=1>67.2</td><td rowspan=1 colspan=1>4.7</td><td rowspan=1 colspan=1>70.3</td><td rowspan=1 colspan=1>33.0</td></tr><tr><td rowspan=1 colspan=1>MiniMax-M3</td><td rowspan=1 colspan=1>11.3</td><td rowspan=1 colspan=1>40.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>6.0</td><td rowspan=1 colspan=1>8.4</td><td rowspan=1 colspan=1>58.1</td><td rowspan=1 colspan=1>10.2</td><td rowspan=1 colspan=1>67.7</td><td rowspan=1 colspan=1>25.2</td></tr><tr><td rowspan=1 colspan=1>GLM-5.3</td><td rowspan=1 colspan=1>43.8</td><td rowspan=1 colspan=1>86.2</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>40.1</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>77.4</td><td rowspan=1 colspan=1>17.0</td><td rowspan=1 colspan=1>69.7</td><td rowspan=1 colspan=1>41.8</td></tr><tr><td rowspan=1 colspan=1>OpenClaw  GPT-5.6-Sol</td><td rowspan=1 colspan=1>46.3</td><td rowspan=1 colspan=1>82.2</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>24.8</td><td rowspan=1 colspan=1>54.7</td><td rowspan=1 colspan=1>80.5</td><td rowspan=1 colspan=1>42.7</td><td rowspan=1 colspan=1>68.6</td><td rowspan=1 colspan=1>50.0</td></tr><tr><td rowspan=1 colspan=1>Claude Opus 5</td><td rowspan=1 colspan=1>5.0</td><td rowspan=1 colspan=1>76.8</td><td rowspan=1 colspan=1>1.3</td><td rowspan=1 colspan=1>7.7</td><td rowspan=1 colspan=1>16.3</td><td rowspan=1 colspan=1>71.7</td><td rowspan=1 colspan=1>1.1</td><td rowspan=1 colspan=1>70.0</td><td rowspan=1 colspan=1>31.2</td></tr><tr><td rowspan=1 colspan=1>Qwen3.8-Max</td><td rowspan=1 colspan=1>36.3</td><td rowspan=1 colspan=1>84.2</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>17.3</td><td rowspan=1 colspan=1>3.7</td><td rowspan=1 colspan=1>78.9</td><td rowspan=1 colspan=1>2.9</td><td rowspan=1 colspan=1>68.9</td><td rowspan=1 colspan=1>36.6</td></tr><tr><td rowspan=1 colspan=1>Kimi K3</td><td rowspan=1 colspan=1>40.0</td><td rowspan=1 colspan=1>83.0</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>17.7</td><td rowspan=1 colspan=1>5.8</td><td rowspan=1 colspan=1>58.9</td><td rowspan=1 colspan=1>13.6</td><td rowspan=1 colspan=1>70.3</td><td rowspan=1 colspan=1>36.2</td></tr><tr><td rowspan=1 colspan=1>DeepSeek-V4-Pro</td><td rowspan=1 colspan=1>23.8</td><td rowspan=1 colspan=1>69.9</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>6.5</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>53.2</td><td rowspan=1 colspan=1>24.6</td><td rowspan=1 colspan=1>70.6</td><td rowspan=1 colspan=1>31.1</td></tr><tr><td rowspan=1 colspan=1>MiniMax-M3</td><td rowspan=1 colspan=1>28.8</td><td rowspan=1 colspan=1>77.5</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>10.5</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>63.1</td><td rowspan=1 colspan=1>15.3</td><td rowspan=1 colspan=1>67.0</td><td rowspan=1 colspan=1>32.8</td></tr><tr><td rowspan=1 colspan=1>GLM-5.3</td><td rowspan=1 colspan=1>57.5</td><td rowspan=1 colspan=1>82.5</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>21.3</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>1.4</td><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=1>71.5</td><td rowspan=1 colspan=1>29.5</td></tr></table>

## 4.3 In-depth Analysis

## 4.3.1 Model–Harness Interaction

Finding: Under the same models, Claude Code achieves the strongest overall performance and leads on most tasks. Figure 4a compares model–harness pairs, with harness means calculated over the six models evaluated under all four harnesses. Claude Code achieves the highest mean score of 41.9, followed by OpenClaw at 36.0, Hermes at 35.0, and Codex at 33.5. It also achieves the highest average score for four of the seven evaluated models: Qwen3.8- Max, Kimi K3, DeepSeek-V4-Pro, and MiniMax-M3. GPT-5.6-Sol performs comparably with Claude Code and OpenClaw, while Opus performs best with Hermes and GLM-5.3 performs best with Codex. Claude Code therefore provides the strongest aggregate performance across the common models.

Figure 4b shows that this lead is achieved by five of the eight tasks. OpenClaw has the second-highest average score across the six common models. Claude Code outperforms it by 23.3 points on claim charting and 18.2 points on patent drafting. Together, these two tasks account for 5.2 points of its 5.9-point average-score advantage. Its FTO recall also reaches 80.6, compared with 40.1 for Codex, showing a substantial advantage over that harness in textual risk retrieval. However, Codex leads on claim charting, OpenClaw leads on novelty search, and Hermes leads on OAR. Translation scores remain close across harnesses, and all remain weak on design FTO. Claude Code’s overall lead is thus supported by stronger performance across several workflows, while task-level comparisons remain necessary when selecting a harness.

![](images/482e64f4ab65fc72ce5b58334ae2f0fc74044185afaca3dc09abfdd2af3f5d08.jpg)  
(a) Model-specific harness performance

![](images/decd7412efb9cb0f83376c8636bc3353b373ba061f97c3befb5c08a2d29802e1.jpg)  
(b) Task-specific advantage: six complete models  
Figure 4: (a) Eight-task average score by model and harness. Dashed lines show harness means over the same six models evaluated under all four harnesses, excluding Opus. Individual bars show all evaluated pairs. (b) Task scores averaged over these six models, centered by subtracting the four-harness mean within each task.

## 4.3.2 Prior-Art Search and Evidence Alignment

![](images/ef7c8013229636d78755d4e71236c760d97570ac33a7976ae06db57a20f65c0c.jpg)  
(a) Invalidity-search recall

![](images/348203320a44093a110e6a96275a1b74c91ef6c0920272fef1f71d57b30d1b70.jpg)  
(b) Feature extraction and evidence alignment  
Figure 5: (a) Recall@5 and Recall@All measure coverage of annotated references in the top five results and the full returned list. (b) SP-F1 measures technical feature extraction, and CP-F1 measures featureto-evidence alignment in claim charting.

Finding: Evidence retrieval is one dominant weakness, and evidence alignment is also weaker than feature extraction in majority of the models. Using Claude Code as a fixed harness, invalidity search is limited mainly by missing references rather than their ranking. Figure 5a compares Recall@5, which measures reference coverage within the top five results, with Recall@All, which considers the full returned list. Recall@All remains below one-third for every model. Extending evaluation beyond the top five adds 4.0–10.4 points, leaving at least two-thirds of the annotated references unrecovered. For GPT-5.6-Sol, recall rises from 21.4% to 31.8%, but 68.2% of references are still missing.

With comparison evidence supplied, six of seven models score lower on evidence alignment than on feature extraction in Figure 5b. The gap is 17.0 points for GPT-5.6-Sol and 18.5 for Opus, using the same scored cases for each model’s two metrics. These models identify technical features more accurately than they establish the corresponding evidence matches.

These two analysis therefore locate concrete weaknesses in separate tasks: invalidity search fails mainly to recover the reference set, while claim charting scores lower on feature-toevidence correspondence than on feature identification.

## 4.3.3 Patent Document Quality

Finding: In patent document tasks, most of the models produce well-structured drafts and address response requirements, but struggle to organize dependent claims, understand case facts, and develop well-supported arguments. Under Claude Code, Figure 6a shows that all seven models score 100.0 on specification form and 87.5–98.3 on claim form. They produce the expected document sections and recognizable independent and dependent claims. However, dependent-claim hierarchy scores range from 40.8 to 52.5, the lowest displayed criterion for every model. This criterion assesses whether dependent claims introduce meaningful additional limitations and establish useful layers of narrower protection. GPT-5.6-Sol and Opus both score 52.5 on hierarchy despite claim-form scores of 95.8 and 98.3. The weakness lies in organizing the scope of protection: a formally complete claim set does not necessarily provide useful fallback positions for examination or amendment.

![](images/8717f3d29a5da63de0d7dcd500e7f5d148c6ff59e3cff1c381a008e7fb96f09f.jpg)  
(a) Drafting quality criteria

![](images/29d9e207dfef864bfed623de2498acaa62f1ffc5fd62f9248e952d3907240959.jpg)  
(b) OAR quality dimensions  
Figure 6: Document quality under Claude Code. (a) Scores on six patent-drafting criteria. (b) Scores on four OAR criterion groups, computed within each case using the original weights of applicable criteria and then averaged over applicable scored cases. Amendment scores cover different case sets across models, all scores are on a 0–100 scale.

Figure 6b examines four groups of OAR criteria: completion and compliance, case understanding and strategy, claim amendment, and substantive argumentation. Group scores retain the original criterion weights, exclude inapplicable criteria, and are averaged over applicable scored cases. Completion and compliance is the highest-scoring group for five of the six models with scored responses. GPT-5.6-Sol reaches 85.8 in this group, but scores only 31.6 on understanding and strategy and 35.5 on argumentation; DeepSeek scores 65.2, 16.1, and 18.3, respectively. These results show that completing the response requirements does not ensure an accurate reading of the case or a well-supported argument addressing the examiner’s objections. Amendment quality also remains limited: five models score below 20.0, while Opus scores 50.0 over only four applicable cases, preventing a direct comparison across models. Together, the two tasks reveal a common limitation: models can assemble the expected document, but struggle with the professional judgments that make its claims and arguments effective.

## 5 Case Study

We use two patent-drafting cases to illustrate how model and harness choices affect draft quality. Figure 7 compares models using the same harness and harnesses using the same model, highlighting differences in technical interpretation and claim drafting. Appendix G provides further comparisons across seven task suites.

![](images/e364ae692e7e929d0fef1869d1057bfe85b6f3793156edf6b597b985956face9.jpg)  
(b) Fixed MiniMax-M3 model: OpenClaw versus Hermes.  
Figure 7: Case studies of different model–harness pairs.

With Codex as a fixed harness, MiniMax-M3 fabricates experimental results, whileGPT-5.6-Sol follows the given technical disclosure. In Figure 7a, the task is to draft an application for recycling AlGaN seed wafers. The disclosure describes cutting a thicker middle wafer, regrowing crystal on both sides to cover exposed cracks, and reusing the wafer in the next cycle. It provides process relationships, but no measured example results. GPT-5.6-Sol retains this sequence without claiming that an experiment has been performed. MiniMax-M3 instead supplies cutting pitches of 900 and 500 µm, reports that the wafer did not fracture, and states that a further 200 µm of growth completely covered the cracks. These statements turn the disclosed process into a report of successful manufacturing without supporting measurements. The fidelity rubric scores the two outputs at 1.0 and 0.0, respectively. The error is therefore an unsupported claim of demonstrated performance, not simply additional descriptive detail. A usable draft must preserve the distinction between how a process is intended to work and what the supplied evidence establishes.

With MiniMax-M3 as a fixed model, Hermes makes an optional requirement mandatory, while OpenClaw preserves it as optional. In Figure 7b, the invention confirms a vehicle’s selected driving direction through a brief motor pulse that produces slight forward or backward motion. Visual, sound or tactile cues may supplement this physical feedback, but the disclosure does not require them. Under OpenClaw, MiniMax-M3 places motionbased confirmation in independent method claim 1 and adds the supplementary cues in dependent claim 2. The core method therefore remains applicable without an auxiliary cue. Under Hermes, the same model requires at least one such cue in independent method claim 1, excluding the disclosed implementation that uses vehicle motion alone. The unnecessarylimitation rubric assigns 1.0 to OpenClaw and 0.0 to Hermes. The harness comparison thus exposes a concrete change in claim scope: a supplementary feature becomes a condition of the core method. Preserving the disclosed components is insufficient if the draft changes which components are required.

## 6 Conclusion

We introduce PatentAgentBench, a benchmark for evaluating LLM agents on professional patent tasks spanning evidence retrieval, claim-level analysis, and document preparation. Built from actual patent records and professional materials, it combines task-specific metrics with expert-reviewed annotations and rubrics to assess the evidence and work products required in patent practice. Our evaluation shows that model and harness choices must be considered together, as their advantages vary across tasks. Current agents remain limited in retrieving relevant references and aligning technical features with supporting evidence. Under Claude Code, patent drafts exhibit strong formal organization but weak dependentclaim hierarchies. Office-action responses show weaknesses in case understanding, response strategy, and substantive argumentation, even when completion and compliance scores are high. The case studies further illustrate how unsupported factual additions and changes to disclosed requirements undermine draft quality. These findings show that reliable patent agents require both stronger evidence use and better control over substantive content. PatentAgentBench provides a common framework for evaluating progress toward these requirements.

## 7 Limitations

PatentAgentBench covers 8 professional patent tasks. However, tasks are evaluated separately rather than as a complete patent lifecycle. The evaluation process focus on the final output or delivery of agents rather than the agent working process. In the future, we plan to extend the benchmark to include more tasks and evaluate the complete patent lifecycle.

## References

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Adam Fourney, Gagan Bansal, Hussein Mozannar, Cheng Tan, Eduardo Salinas, Erkang Zhu, Friederike Niedtner, Grace Proebsting, Griffin Bassman, Jack Gerrits, Jacob Alber, Peter Chang, Ricky Loynd, Robert West, Victor Dibia, Ahmed Awadallah, Ece Kamar, Rafah Hosn, and Saleema Amershi. Magentic-one: A generalist multi-agent system for solving complex tasks. arXiv preprint arXiv:2411.04468, 2024.

Harvey AI. Harvey lab: The legal agent benchmark, 2026. URL https://github. com/harveyai/harvey-labs/tree/v1.0. Announcement: https://www.harvey.ai/blog/ introducing-harveys-legal-agent-benchmark.

Yehoon Jang, Chaewon Lee, Hyun-seok Min, and Sungchul Choi. Pilot-bench: A benchmark for legal reasoning in the patent domain with irac-aligned classification tasks. In Proceedings of the Natural Legal Language Processing Workshop 2025, pp. 240–280, 2025.

Lekang Jiang, Pascal A Scherz, and Stefan Goetz. Towards better evaluation for generated patent claims. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3775–3788, 2025.

Valentin Knappich, Annemarie Friedrich, Anna Hatty, and Simon Razniewski. Pedantic: A dataset for¨ the automatic examination of definiteness in patent claims. arXiv preprint arXiv:2505.21342, 2025.

Ralf Krestel, Renukswamy Chikkamath, Christoph Hewel, and Julian Risch. A survey on deep learning for patent analysis. World Patent Information, 65:102035, 2021.

Ryan Lee, Alexander Spangher, and Xuezhe Ma. Patentedits: Framing patent novelty as textual entailment. arXiv preprint arXiv:2411.13477, 2024.

Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for ”mind” exploration of large language model society. In Advances in Neural Information Processing Systems, pp. 51991–52008, 2023.

Hyunseung Lim, Sooyohn Nam, Sungmin Na, Ji Yong Cho, June Yong Yang, Hyungyu Shin, Yoonjoo Lee, Juho Kim, Moontae Lee, and Hwajung Hong. Panorama: A dataset and benchmarks capturing decision trails and rationales in patent examination. Advances in Neural Information Processing Systems, 38, 2026.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Junyu Luo, Weizhi Zhang, Ye Yuan, Yusheng Zhao, Junwei Yang, Yiyang Gu, Bohan Wu, Binqi Chen, Ziyue Qiao, Qingqing Long, et al. Large language model agent: A survey on methodology, applications and challenges. arXiv preprint arXiv:2503.21460, 2025.

Mike Merrill, Alexander Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich, Lin Shi, Jeong Shin, Thomas Walshe, E Kelly Buchanan, et al. Terminal-bench: Benchmarking agents on hard, realistic tasks in command line interfaces. In International Conference on Learning Representations, volume 2026, pp. 40903–40986, 2026.

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. In Advances in Neural Information Processing Systems, 2023.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652, 2023.

Homaira Huda Shomee, Zhu Wang, Sathya N Ravi, and Sourav Medya. A survey on patent analysis: From nlp to multimodal ai. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 8545–8561, 2025.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

Sakhinana Sagar Srinivas, Vijay Sri Vaikunth, and Venkataramana Runkana. Towards automated patent workflows: ai-orchestrated multi-agent framework for intellectual property management and analysis. arXiv preprint arXiv:2409.19006, 2024.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y Charles, et al. Kimi k3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026.

Khanh-Tung Tran, Dung Dao, Minh-Duong Nguyen, Quoc-Viet Pham, Barry O’Sullivan, and Hoang D Nguyen. Multi-agent collaboration mechanisms: A survey of llms. arXiv preprint arXiv:2501.06322, 2025.

Qiyao Wang, Jianguo Huang, Shule Lu, Yuan Lin, Kan Xu, Liang Yang, and Hongfei Lin. Ipeval: A bilingual intellectual property agency consultation evaluation benchmark for large language models. arXiv preprint arXiv:2406.12386, 2024.

Qiyao Wang, Guhong Chen, Hongbo Wang, Huaren Liu, Minghui Zhu, Zhifei Qin, Linwei Li, Yilin Yue, Shiqiang Wang, Jiayan Li, et al. Ipbench: Benchmarking the knowledge of large language models in intellectual property. arXiv preprint arXiv:2504.15524, 2025.

Qiyao Wang, Xinyi Chen, Longze Chen, Hongbo Wang, Hamid Alinejad-Rokny, Yuan Lin, and Min Yang. Patre: A full-stage office action and rebuttal generation benchmark for patent examination. arXiv preprint arXiv:2605.03571, 2026.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W. White, Doug Burger, and Chi Wang. Autogen: Enabling next-gen llm applications via multi-agent conversations. In First Conference on Language Modeling, 2024.

Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, et al. Deepseek-v4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348, 2026.

John Yang, Carlos Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated software engineering. Advances in Neural Information Processing Systems, 37:50528–50652, 2024.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

Yuhang Yao, Haixin Wang, Yibo Chen, Jiawen Wang, Min Chang Jordan Ren, Bosheng Ding, Salman Avestimehr, and Chaoyang He. Toward super agent system with hybrid ai routers. arXiv preprint arXiv:2504.10519, 2025.

Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, et al. Glm-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763, 2026.

## Contribution List

## Project Lead

Lisha Zhang zhanglisha@patsnap.com

Kai Duan duankai@patsnap.com

Responsible for overall project direction, resource coordination, technical guidance, and final review of the technical report.

## Core Contributors

Yijie Hu huyijie@patsnap.com

Wei Li liwei@patsnap.com

Junjie Zhou zhoujunjie@patsnap.com

Zhiqi Jiang jiangzhiqi@patsnap.com

Yi Jin jinyi@patsnap.com

Licong Xu xulicong@patsnap.com

Xinkai Sun sunxinkai@patsnap.com

Responsible for task and experimental design, data collection and annotation, data quality assurance, experimental infrastructure and execution, paper writing, and open-source development.

## A Experimental Setting Details

## A.1 Agent Harnesses

An agent harness manages the interaction between a model, its task context, and tool calls during execution. We evaluate four general-purpose harnesses with the following versions:

• Codex CLI, version 0.151.0, is OpenAI’s terminal-based coding agent. It provides a local interface for carrying out tasks through model interactions and tool calls.

• Claude Code, version 2.1.251, is Anthropic’s agentic coding environment. It accepts natural-language instructions and coordinates actions within a working context.

• OpenClaw, version 2026.8.1, is an open-source personal assistant framework that connects configurable models with an agent runtime and external integrations.

• Hermes Agent, version 0.21.1, is Nous Research’s agent framework with a terminal interface, support for multiple model providers, and tool-calling workflows.

The main table abbreviates Codex CLI as Codex and Hermes Agent as Hermes.

## A.2 Models and Execution Protocol

We evaluate GPT-5.6-Sol, Claude Opus 5, Kimi K3, Qwen3.8-Max, MiniMax-M3, GLM-5.3, and DeepSeek-V4-Pro. A configuration is a model–harness pair. Claude Opus 5 was not evaluated with Codex because the model interface was incompatible with the protocol supported by our Codex configuration. This untested configuration is omitted from the main table rather than scored as zero. Comparisons across all four harnesses use the six models with complete results; comparisons including Opus use only the harnesses for which its results are available.

Each configuration receives the task input and evidence conditions defined in Section 3.2. An initial execution failure permits up to three retries, giving a maximum of four attempts per instance. The task score is computed over the full assigned case set, including cases that remain unsuccessful after retries.

## A.3 Tools and Judge Model

We provide a commercial patent novelty-search service through the Model Context Protocol (MCP), which agents can call to retrieve patent evidence. This MCP service is the only tool supplied by the benchmark; we provide no additional tools. The harness controls how the model invokes the service during task execution.

For evaluation components that require an LLM judge, including rubric-based patent drafting and OAR assessment, we use GPT 6 sol. The judge scores the submitted work against the case materials and evaluation rubrics. Reference matching and the automatic metrics defined below are computed by their corresponding scorers.

Table 4: Professional backgrounds of the expert team.
<table><tr><td>Professional background</td><td>Count</td></tr><tr><td>Senior patent examiner</td><td>5</td></tr><tr><td>Senior patent agent</td><td>15</td></tr><tr><td>Dual-qualified lawyer</td><td>3</td></tr><tr><td>Senior IPR specialist</td><td>8</td></tr><tr><td>Total</td><td>31</td></tr></table>

## B Expert Team Profile

Table 4 summarizes the professional backgrounds of the 31 experts involved in data curation and evaluation design.

## C Dataset Details

Table 5 summarizes the language, jurisdiction, evaluation metrics, and rubric coverage of each task suite. Language denotes the task input or required output; design FTO uses image inputs. The benchmark contains 510 cases across eight suites. Translation contains 80 cases with 881 evaluation items. Metric definitions and aggregation rules are provided in Appendix D.

Table 5: Detailed dataset coverage and evaluation. ZH and EN denote Chinese and English. Jurisdiction codes identify the target jurisdiction or source patent office: CN (China), US (United States), EP (European Patent Office), EU (European Union), and WO (PCT international publication). Translation coverage follows the source patent publication numbers. Rubric counts are criterion entries summed across cases, including repeated common criteria.
<table><tr><td>Task suite</td><td>Cases</td><td>Language</td><td>Jurisdiction</td><td>Evaluation metrics</td><td>Rubric entries</td></tr><tr><td>Novelty Search</td><td>80</td><td>ZH/EN</td><td>CN/US/EP/WO</td><td>Hit@100</td><td>0</td></tr><tr><td>Patent FTO</td><td>60</td><td>ZH/EN</td><td>CN/US/EP</td><td>Recall@All</td><td>0</td></tr><tr><td>Design FTO</td><td>80</td><td>Image input</td><td>CN/EU</td><td>Recall@200</td><td>0</td></tr><tr><td>Invalidity Search</td><td>80</td><td>ZH/EN</td><td>CN/US</td><td>Recall@All</td><td>0</td></tr><tr><td>Claim Charting</td><td>30</td><td>ZH/EN</td><td>CN/US/EP</td><td>CP-F1; supplementary SP-F1</td><td>0</td></tr><tr><td>Patent Drafting</td><td>60</td><td>ZH/EN</td><td>CN/US/EP</td><td>Weighted rubrics; pass rate</td><td>977</td></tr><tr><td>OAR</td><td>40</td><td>ZH/EN</td><td>CN/US/EP</td><td>Weighted rubrics; pass rate</td><td>57</td></tr><tr><td>Translation</td><td>80</td><td>ZH↔EN</td><td>CN/US/EP/WO</td><td>BLEU-1, METEOR, ROUGE-1, SacreBLEU; terminology, character, and writing checks</td><td>0</td></tr><tr><td>Total</td><td>510</td><td></td><td></td><td></td><td></td></tr></table>

Table 6 reports the drafting rubric inventory: 15 common criteria per case and 77 case-specific entries across the suite. OAR uses 57 rubric entries; its scored dimensions, applicability rules, and aggregation are defined below. The drafting inventory counts criterion instances across cases, whereas the OAR count describes the rubric inventory, so these counts are not summed. Pass rates use the expert-defined thresholds reported with each task’s detailed results.

Table 6: Patent-drafting rubric inventory across cases. Common criteria are counted once per case in which they are provided.
<table><tr><td>Task suite</td><td>Common</td><td>Case-specific</td><td>Total</td></tr><tr><td>Patent Drafting</td><td>900</td><td>77</td><td>977</td></tr></table>

## D Evaluation Metric Definitions

## D.1 Notation and Aggregation

Let $N _ { t }$ be the number of assigned cases in task $t ,$ and let $s _ { t i } \in [ 0 , 1 ]$ be the primary score for case i. Let c be the reporting scale factor, with $c = 1 0 0$ . Every task score is reported on a 0–100 scale:

$$
S _ { t } = \frac { c } { N _ { t } } \sum _ { i = 1 } ^ { N _ { t } } s _ { t i } .\tag{1}
$$

For the primary task scores in Table $^ { 3 , }$ we include all assigned cases, with failed cases contributing zero. For drafting and OAR, a case score originally expressed on a 0–100 scale is divided by 100 before applying Eq. 1. Let T be the number of task suites, with $T = 8$ . The overall mean score S<sup>¯</sup>, reported as AVG in the result tables, is

$$
\bar { S } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } S _ { t } .\tag{2}
$$

Translation contributes one task score after averaging its two directions. An unavailable task result is distinct from an observed zero: AVG is not computed when any task result, or either translation direction, is missing.

We distinguish overall task performance from the quality of scored documents. The main results use all assigned cases. Quality means and document-level analyses use only scored outputs. Let $A _ { t }$ be this scored set, $\dot { n _ { t } } = | A _ { t } |$ , and $q _ { t i }$ the document quality score on a 0–100 scale. For $n _ { t } > 0 .$

$$
Q _ { t } = \frac { 1 } { n _ { t } } \sum _ { i \in A _ { t } } q _ { t i } , \qquad S _ { t } = \frac { n _ { t } } { N _ { t } } Q _ { t } .\tag{3}
$$

Here $Q _ { t }$ describes scored documents, whereas $S _ { t }$ includes unscored cases as zero. The equality relates means of the same quality score; individual rubric dimensions are not task totals. Figure 6 and the appendix quality analyses use the scored-output denominator to compare document properties. These analyses describe the observed documents, not the frequency of completing a task. Pass rates use all $N _ { t }$ assigned cases, with unscored cases counted as not passing. When $n _ { t } = 0 .$ , conditional quality is undefined; the appendix displays 0.0 as a placeholder.

For retrieval, let $G _ { i }$ be the deduplicated reference set and $R _ { i } ^ { ( K ) }$ the set of distinct references in the first K returned results. Let $R _ { i } ^ { ( \mathrm { a l l } ) }$ denote the full returned list after matching and deduplication. Matching uses the task-specific identifiers described below.

## D.2 Novelty Search

The primary metric is Hit@K, with cutoff K = 100. A case receives credit if at least one reference patent family is retrieved:

$$
s _ { \mathrm { n o v } , i } = \mathbf { 1 } \Big [ G _ { i } \cap R _ { i } ^ { ( K ) } \neq \emptyset \Big ] .\tag{4}
$$

Publication identifiers are mapped to patent families before matching. The task score is the percentage of cases with a hit, rather than the mean fraction of all reference families recovered. Supplementary Hit@K uses the same definition with a different cutoff; supplementary Recall@K uses $\lvert G _ { i } \cap R _ { i } ^ { ( K ) } \rvert / \lvert G _ { i } \rvert$

## D.3 Patent FTO

Patent FTO uses macro recall over the full returned list:

$$
s _ { \mathrm { f t o } , i } = \frac { | G _ { i } \cap R _ { i } ^ { \mathrm { ( a l l ) } } | } { | G _ { i } | } .\tag{5}
$$

Matching is performed at the patent-application level. Equation 1 gives every case equal weight, regardless of how many reference patents it contains.

## D.4 Design FTO

The primary metric is macro Recall@K, with cutoff K = 200:

$$
s _ { \mathrm { d e s i g n } , i } = \frac { | G _ { i } \cap R _ { i } ^ { ( K ) } | } { | G _ { i } | } .\tag{6}
$$

Only references found within the first 200 results contribute to the numerator. We retain the design-patent matching rules of the reference annotations.

## D.5 Invalidity Search

Invalidity search uses Recall@All:

$$
s _ { \mathrm { i n v } , i } = \frac { | G _ { i } \cap R _ { i } ^ { ( \mathrm { a l l } ) } | } { | G _ { i } | } .\tag{7}
$$

References are matched at the patent-identifier level. The score measures recovery of the reference evidence set; the associated statutory grounds and affected claims remain available for case-level analysis.

## D.6 Claim Charting

The primary metric is CP-F1 for feature-level comparisons. Let $a _ { i } , b _ { i }$ , and $c _ { i }$ be the numbers of correct, unmatched predicted, and missed reference comparison items under the annotated matching rules. Precision and recall are

$$
P _ { i } = { \frac { a _ { i } } { a _ { i } + b _ { i } } } , \qquad R _ { i } = { \frac { a _ { i } } { a _ { i } + c _ { i } } } .\tag{8}
$$

The case score is

$$
s _ { \mathrm { c c } , i } = \left\{ \begin{array} { l l } { \displaystyle \frac { 2 P _ { i } R _ { i } } { P _ { i } + R _ { i } } , } & { P _ { i } + R _ { i } > 0 , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{9}
$$

A precision or recall term with a zero denominator is set to zero. We average case-level F1 scores, rather than pooling comparison items across cases. Supplementary SP-F1 applies the same precision–recall calculation to feature-decomposition items and does not enter AVG separately.

## D.7 Patent Drafting

Let $\mathcal { I } _ { i }$ contain the applicable general and case-specific quality criteria for case i. Criterion j has its prescribed weight $w _ { i j }$ and a judge score $u _ { i j }$ on the rubric’s normalized scale. The case quality score is

$$
s _ { \mathrm { d r a f t } , i } = \frac { \sum _ { j \in \mathcal { T } _ { i } } w _ { i j } u _ { i j } } { \sum _ { j \in \mathcal { T } _ { i } } w _ { i j } } .\tag{10}
$$

General and case-specific criteria share the same weighted sum; their relative contribution depends on the case’s configured weights. Zero-weight eligibility and unity checks do not contribute to this quality score. The task score averages case quality over the full drafting set.

## D.8 Office Action Response

OAR uses rubric-based assessment of the submitted response against the case materials. Applicable criteria receive scores $u _ { i j } \in \{ 0 , 0 . 5 , 1 \}$ . The base weight pools allocate 25% to basic quality, 20% to amendment quality, and 55% to substantive argumentation.

Let $\mathbf { \mathcal { A } } _ { i }$ denote the applicable scored criteria for case i and $w _ { i j }$ their final allocated weights. The normalized case score is

$$
s _ { \mathrm { o a r } , i } = \frac { \sum _ { j \in \mathcal { A } _ { i } } w _ { i j } u _ { i j } } { \sum _ { j \in \mathcal { A } _ { i } } w _ { i j } } .\tag{11}
$$

Let $q _ { i } \in [ 0 , 1 ]$ denote the recorded case quality after normalization. The recorded score is obtained by expressing $s _ { \mathrm { o a r } , i }$ on the 0–100 scale, rounding to one decimal place, and converting back to the unit interval. The primary task score retains all assigned cases, with unscored cases contributing zero. In the detailed results, quality is the mean of $q _ { i }$ over the scored set ${ \mathcal { D } } ,$ and pass rate is

$$
Q = \frac { 1 } { | \mathcal { D } | } \sum _ { i \in \mathcal { D } } q _ { i } , \qquad P = \frac { 1 } { N } \sum _ { i \in \mathcal { D } } \mathbf { 1 } [ q _ { i } \geq \tau ] ,\tag{12}
$$

where $N = 4 0$ is the number of assigned cases and $\tau = 0 . 8$ is the expert-defined usability threshold. Unscored cases do not pass.

For group-level analysis, we organize the criteria into four groups. Completion and compliance covers objection coverage, deliverable completeness and text quality, and jurisdictional compliance. Case understanding and strategy covers understanding the case, choosing a response strategy, logical consistency, and factual and citation accuracy. Amendment covers whether changes address the objections, preserve appropriate claim scope, and satisfy legal and formal requirements. Argumentation covers the substantive response to inventive step, novelty, eligibility, disclosure, clarity and support, and essential-feature objections. Let $\bar { B _ { k } }$ contain the criteria in group k. Define $\mathcal { A } _ { i k } = \dot { \mathcal { A } } _ { i } \cap \mathcal { B } _ { k }$ and let $\mathcal { D } _ { k }$ contain scored cases with positive total weight in that group. The group score is

$$
R _ { k } = \frac { 1 } { | \mathcal { D } _ { k } | } \sum _ { i \in \mathcal { D } _ { k } } \frac { \sum _ { j \in \mathcal { A } _ { i k } } w _ { i j } u _ { i j } } { \sum _ { j \in \mathcal { A } _ { i k } } w _ { i j } } .\tag{13}
$$

The quantities $Q , P ,$ and $R _ { k }$ lie in the unit interval. For presentation, quality and group scores are converted to the 0–100 scale, and pass rates are expressed as percentages. Each applicable case contributes equally after within-case weighting. For criterion-level tables, each criterion is averaged over its own applicable scored cases. Different applicable case sets must be considered when comparing means.

## D.9 Patent Translation

Let $q _ { d i } \in [ 0 , 1 ]$ be the score of instance i in direction $d ,$ where $d = 1$ denotes Chinese-to-English translation and $d = 2$ denotes English-to-Chinese translation. Let $b _ { d i } , \ m _ { d i } , \ r _ { d i } ,$ and $z _ { d i }$ denote its BLEU-1, METEOR, ROUGE-1, and SacreBLEU scores, respectively, each normalized to [0, 1]. Ordinary character-, sentence-, and paragraph-level translation uses

$$
q _ { d i } = \left\{ \frac { b _ { d i } + m _ { d i } } { 2 } , d = 1 , \right.\tag{14}
$$

Document-level translation uses

$$
q _ { d i } = z _ { d i } .\tag{15}
$$

Category-specific checks replace these overlap scores where the instance tests a particular patent requirement. Let $y _ { i }$ be the translated output and $T _ { i }$ its required target terms. Terminology accuracy is

$$
q _ { d i } ^ { \mathrm { t e r m } } = \prod _ { t \in T _ { i } } \mathbf { 1 } [ t \mathrm { o c c u r s } \mathrm { i n } y _ { i } ] .\tag{16}
$$

Terminology consistency additionally checks the required occurrence count $c _ { i t }$ for each annotated term:

$$
q _ { d i } ^ { \mathrm { c o n s } } = \prod _ { t \in T _ { i } } \mathbf { 1 } [ \mathrm { c o u n t } ( t , y _ { i } ) = c _ { i t } ] .\tag{17}
$$

For the required special-character set $H _ { i } ,$ preservation is

$$
q _ { d i } ^ { \mathrm { c h a r } } = \prod _ { h \in { \cal H } _ { i } } { \bf 1 } [ h \operatorname { o c c u r s } \operatorname { i n } y _ { i } ] .\tag{18}
$$

The English-title convention check assigns

$$
q _ { d i } ^ { \mathrm { t i t l e } } = \mathbf { 1 } [ \mathrm { f i r s t } ( y _ { i } ) \not \in \{ \mathrm { a } , \mathrm { a n } , \mathrm { t h e } \} ] ,\tag{19}
$$

with the initial word checked case-insensitively. For a required set of patent-writing expressions $W _ { i } ,$ , the score is

$$
q _ { d i } ^ { \mathrm { n o r m } } = \frac { \sum _ { w \in W _ { i } } \mathbf { 1 } [ w \mathrm { o c c u r s } \mathrm { i n } y _ { i } ] } { | W _ { i } | } .\tag{20}
$$

Each instance uses the rule corresponding to its annotated category. Hallucination and omission checks are auxiliary diagnostics and do not subtract additional points from $q _ { d i }$

Let D be the number of translation directions, with $D = 2 ,$ , and let $N _ { d }$ be the number of instances assigned to direction d. The directional score $S _ { d }$ and the final translation score $S _ { \tau }$ are

$$
{ { S } _ { d } } = \frac { c } { { { N } _ { d } } } \sum _ { i = 1 } ^ { { { N } _ { d } } } { { { q } _ { d i } } } ,\tag{21}
$$

$$
S _ { \tau } = \frac { 1 } { D } \sum _ { d = 1 } ^ { D } S _ { d } .\tag{22}
$$

Execution failures remain in the directional denominators with zero contribution. The two directional scores are averaged only after their respective instance-level aggregation.

## E Task Completion Rates

We measure task completion of whether an agent delivers the requested result under the evaluation setup. A run is successful if it produces the requested task output, regardless of its quality score Success rate is the proportion of recorded tasks completed after the allowed retries. Table 7 report this rate for each task, model, and harness.

Table 7: Task completion rates for each model–harness configuration.
<table><tr><td>Model</td><td>Novelty search</td><td>FTO</td><td>Design FTO</td><td>Invalidity search</td><td>Claim charting</td><td>Drafting</td><td></td><td>OAR Translation</td></tr><tr><td colspan="9">Codex</td></tr><tr><td>GPT-5.6-Sol</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Qwen3.8-Max</td><td>100.0</td><td>100.0</td><td>82.5</td><td>98.8</td><td>96.7</td><td>40.0</td><td>45.0</td><td>100.0</td></tr><tr><td>Kimi K3</td><td>100.0</td><td>100.0</td><td>98.8</td><td>100.0</td><td>90.0</td><td>83.3</td><td>17.5</td><td>100.0</td></tr><tr><td>DeepSeek-V4-Pro</td><td>26.3</td><td>53.3</td><td>7.5</td><td>57.5</td><td>16.7</td><td>73.3</td><td>30.0</td><td>100.0</td></tr><tr><td>MiniMax-M3</td><td>65.0</td><td>96.7</td><td>12.5</td><td>3.8</td><td>66.7</td><td>25.0</td><td>0.0</td><td>100.0</td></tr><tr><td>GLM-5.3</td><td>100.0</td><td>100.0</td><td>7.5</td><td>100.0</td><td>43.3</td><td>93.3</td><td>75.0</td><td>100.0</td></tr><tr><td colspan="9">Claude Code</td></tr><tr><td>GPT-5.6-Sol</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>96.7</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Claude Opus 5</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>93.3</td><td>100.0</td><td>97.5</td><td>100.0</td></tr><tr><td>Qwen3.8-Max</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>97.5</td><td>100.0</td></tr><tr><td>Kimi K3</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>46.7</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>DeepSeek-V4-Pro</td><td>100.0</td><td>100.0</td><td>0.0</td><td>100.0</td><td>83.3</td><td>55.0</td><td>100.0</td><td>100.0</td></tr><tr><td>MiniMax-M3</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>50.0</td><td>100.0</td><td>82.5</td><td>100.0</td></tr><tr><td>GLM-5.3</td><td>100.0</td><td>100.0</td><td>0.0</td><td>100.0</td><td>6.7</td><td>100.0</td><td>0.0</td><td>100.0</td></tr><tr><td colspan="9">Hermes</td></tr><tr><td>GPT-5.6-Sol</td><td>100.0</td><td>100.0</td><td>97.5</td><td>100.0</td><td>16.7</td><td>66.7</td><td>100.0</td><td>100.0</td></tr><tr><td>Claude Opus 5</td><td>100.0</td><td>100.0</td><td>97.5</td><td>100.0</td><td>96.7</td><td>100.0</td><td>90.0</td><td>99.9</td></tr><tr><td>Qwen3.8-Max</td><td>42.5</td><td>96.7</td><td>61.3</td><td>48.8</td><td>0.0</td><td>51.7</td><td>5.0</td><td>100.0</td></tr><tr><td>Kimi K3</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>6.7</td><td>90.0</td><td>85.0</td><td>100.0</td></tr><tr><td>DeepSeek-V4-Pro</td><td>68.8</td><td>76.7</td><td>95.0</td><td>100.0</td><td>13.3</td><td>86.7</td><td>20.0</td><td>100.0</td></tr><tr><td>MiniMax-M3</td><td>91.3</td><td>96.7</td><td>90.0</td><td>91.3</td><td>20.0</td><td>81.7</td><td>67.5</td><td>100.0</td></tr><tr><td>GLM-5.3</td><td>100.0</td><td>100.0</td><td>0.0</td><td>98.8</td><td>0.0</td><td>100.0</td><td>62.5</td><td>100.0</td></tr><tr><td colspan="9">OpenClaw</td></tr><tr><td>GPT-5.6-Sol</td><td>100.0</td><td>98.3</td><td>100.0</td><td>100.0</td><td>100.0</td><td>96.7</td><td>80.0</td><td>100.0</td></tr><tr><td>Claude Opus 5</td><td>22.5</td><td>93.3</td><td>100.0</td><td>31.3</td><td>23.3</td><td>88.3</td><td>5.0</td><td>100.0</td></tr><tr><td>Qwen3.8-Max</td><td>100.0</td><td>100.0</td><td>98.8</td><td>93.8</td><td>6.7</td><td>96.7</td><td>12.5</td><td>100.0</td></tr><tr><td>Kimi K3</td><td>100.0</td><td>100.0</td><td>100.0</td><td>93.8</td><td>6.7</td><td>71.7</td><td>45.0</td><td>100.0</td></tr><tr><td>DeepSeek-V4-Pro</td><td>100.0</td><td>100.0</td><td>76.3</td><td>97.5</td><td>0.0</td><td>66.7</td><td>87.5</td><td>100.0</td></tr><tr><td>MiniMax-M3</td><td>82.5</td><td>100.0</td><td>11.3</td><td>85.0</td><td>0.0</td><td>86.7</td><td>62.5</td><td>100.0</td></tr><tr><td>GLM-5.3</td><td>97.5</td><td>96.7</td><td>42.5</td><td>83.8</td><td>0.0</td><td>1.7</td><td>5.0</td><td>99.9</td></tr></table>

Claim charting exposes a failure to reserve enough output budget for the final deliverable. In our runs, some model–harness combinations continue reasoning until they exhaust the maximum output length, leaving no complete claim chart. Repeated attempts do not consistently resolve this behavior. For example, DeepSeek–OpenClaw produces no evaluable chart across 30 cases, and every stored response reports truncation at the model’s output-token limit. The failure occurs before the substantive quality of the chart can be assessed: the agent spends its budget on reasoning but does not deliver the requested result.

Office-action response exposes a different failure in document handling. Scanned examination documents require agents to extract their contents before drafting. OCR alone does not prevent failure if later tools return image inputs that the model endpoint cannot accept. The case in Appendix H shows an OAR run ending when image crops reach a text-only endpoint. Reliable completion therefore requires both control over intermediate reasoning and compatibility between tool outputs and model inputs.

## F Detailed Task Results

Scores use a 0–100 scale with one decimal place. The untested Codex–Opus configuration is omitted.

## F.1 Novelty Search

Table 8: Novelty-search results over 80 cases. Hit measures whether a case retrieves at least one annotated patent family; recall measures the fraction of annotated families recovered, averaged over cases.
<table><tr><td>Model</td><td>Hit@10</td><td>Recall@10</td><td>Hit@100</td><td>Recall@100</td></tr><tr><td colspan="5">Codex</td></tr><tr><td>GPT-5.6-Sol</td><td>27.5</td><td>9.4</td><td>27.5</td><td>9.4</td></tr><tr><td>Qwen3.8-Max</td><td>35.0</td><td>10.0</td><td>51.3</td><td>21.6</td></tr><tr><td>Kimi K3</td><td>12.5</td><td>5.1</td><td>13.8</td><td>5.7</td></tr><tr><td>DeepSeek-V4-Pro</td><td>7.5</td><td>2.7</td><td>10.0</td><td>3.4</td></tr><tr><td>MiniMax-M3</td><td>15.0</td><td>5.2</td><td>15.0</td><td>5.6</td></tr><tr><td>GLM-5.3</td><td>76.3</td><td>33.6</td><td>86.3</td><td>42.3</td></tr><tr><td colspan="5">Claude Code</td></tr><tr><td>GPT-5.6-Sol</td><td>38.8</td><td>11.6</td><td>38.8</td><td>11.8</td></tr><tr><td>Claude Opus 5</td><td>37.5</td><td>11.3</td><td>52.5</td><td>19.4</td></tr><tr><td>Qwen3.8-Max</td><td>17.5</td><td>4.5</td><td>32.5</td><td>9.6</td></tr><tr><td>Kimi K3</td><td>31.3</td><td>8.9</td><td>33.8</td><td>9.5</td></tr><tr><td>DeepSeek-V4-Pro</td><td>35.0</td><td>10.0</td><td>37.5</td><td>10.6</td></tr><tr><td>MiniMax-M3</td><td>23.8</td><td>6.1</td><td>27.5</td><td>7.7</td></tr><tr><td>GLM-5.3</td><td>41.3</td><td>13.6</td><td>50.0</td><td>19.6</td></tr><tr><td colspan="5">Hermes</td></tr><tr><td>GPT-5.6-Sol</td><td>57.5</td><td>26.1</td><td>62.5</td><td>31.0</td></tr><tr><td>Claude Opus 5</td><td>45.0</td><td>14.4</td><td>52.5</td><td>21.6</td></tr><tr><td>Qwen3.8-Max</td><td>12.5</td><td>3.4</td><td>17.5</td><td>5.2</td></tr><tr><td>Kimi K3</td><td>36.3</td><td>9.6</td><td>36.3</td><td>9.6</td></tr><tr><td>DeepSeek-V4-Pro</td><td>20.0</td><td>5.4</td><td>22.5</td><td>6.2</td></tr><tr><td>MiniMax-M3</td><td>8.8</td><td>2.3</td><td>11.3</td><td>2.7</td></tr><tr><td>GLM-5.3</td><td>36.3</td><td>8.7</td><td>43.8</td><td>12.1</td></tr><tr><td colspan="5">OpenClaw</td></tr><tr><td>GPT-5.6-Sol</td><td>46.3</td><td>13.5</td><td>46.3</td><td>13.5</td></tr><tr><td>Claude Opus 5</td><td>5.0</td><td>1.1</td><td>5.0</td><td>1.1</td></tr><tr><td>Qwen3.8-Max</td><td>33.8</td><td>9.7</td><td>36.3</td><td>10.7</td></tr><tr><td>Kimi K3</td><td>37.5</td><td>10.9</td><td>40.0</td><td>11.6</td></tr><tr><td>DeepSeek-V4-Pro</td><td>22.5</td><td>6.0</td><td>23.8</td><td>6.2</td></tr><tr><td>MiniMax-M3</td><td>23.8</td><td>6.0</td><td>28.8</td><td>7.1</td></tr><tr><td>GLM-5.3</td><td>52.5</td><td>18.0</td><td>57.5</td><td>21.9</td></tr></table>

Novelty search recovers at least one relevant patent family far more often than it recovers the complete reference set. GLM-5.3 with Codex leads every reported metric, but its 86.3 Hit@100 is accompanied by only 42.3 Recall@100. Expanding the cutoff from ten to one hundred raises its recall from 33.6 to 42.3, so some relevant families occur deeper in the results, while most remain unrecovered on average. The preferred harness also depends on the model: GLM scores higher with Codex than with Hermes, whereas GPT-5.6-Sol shows the reverse ordering. Strong case-level hit rates therefore coexist with incomplete evidence coverage, even for the leading configuration.

## F.2 Patent FTO

Table 9: Patent FTO results over 60 cases. High-risk recall retains candidates labeled high risk. References are matched at the application level. Macro recall weights cases equally; micro recall pools reference applications.
<table><tr><td>Model</td><td>Hit</td><td>Recall macro</td><td>Recall micro</td><td>High-risk macro</td><td>High-risk micro</td></tr><tr><td colspan="6">Codex</td></tr><tr><td>GPT-5.6-Sol</td><td>60.0</td><td>54.2</td><td>51.1</td><td>45.8</td><td>39.8</td></tr><tr><td>Qwen3.8-Max</td><td>56.7</td><td>55.5</td><td>48.9</td><td>55.2</td><td>47.7</td></tr><tr><td>Kimi K3</td><td>6.7</td><td>5.7</td><td>5.7</td><td>5.3</td><td>4.6</td></tr><tr><td>DeepSeek-V4-Pro</td><td>45.0</td><td>43.2</td><td>44.3</td><td>39.5</td><td>40.9</td></tr><tr><td>MiniMax-M3</td><td>8.3</td><td>8.3</td><td>6.8</td><td>8.3</td><td>6.8</td></tr><tr><td>GLM-5.3</td><td>76.7</td><td>74.0</td><td>62.5</td><td>69.8</td><td>59.1</td></tr><tr><td colspan="6">Claude Code</td></tr><tr><td>GPT-5.6-Sol</td><td>86.7</td><td>79.4</td><td>67.1</td><td>74.2</td><td>61.4</td></tr><tr><td>Claude Opus 5</td><td>90.0</td><td>86.5</td><td>77.3</td><td>85.3</td><td>73.9</td></tr><tr><td>Qwen3.8-Max</td><td>88.3</td><td>86.2</td><td>76.1</td><td>83.1</td><td>70.5</td></tr><tr><td>Kimi K3</td><td>86.7</td><td>83.2</td><td>75.0</td><td>81.6</td><td>70.5</td></tr><tr><td>DeepSeek-V4-Pro</td><td>80.0</td><td>78.2</td><td>70.5</td><td>76.8</td><td>63.6</td></tr><tr><td>MiniMax-M3</td><td>75.0</td><td>72.5</td><td>65.9</td><td>68.4</td><td>62.5</td></tr><tr><td>GLM-5.3</td><td>86.7</td><td>83.9</td><td>72.7</td><td>82.6</td><td>69.3</td></tr><tr><td colspan="6">Hermes</td></tr><tr><td>GPT-5.6-Sol</td><td>65.0</td><td>62.7</td><td>52.3</td><td>56.7</td><td>45.5</td></tr><tr><td>Claude Opus 5</td><td>90.0</td><td>86.4</td><td>77.3</td><td>85.2</td><td>73.9</td></tr><tr><td>Qwen3.8-Max</td><td>83.3</td><td>80.5</td><td>75.0</td><td>77.1</td><td>68.2</td></tr><tr><td>Kimi K3</td><td>86.7</td><td>82.7</td><td>75.0</td><td>79.0</td><td>67.1</td></tr><tr><td>DeepSeek-V4-Pro</td><td>63.3</td><td>61.2</td><td>56.8</td><td>57.5</td><td>52.3</td></tr><tr><td>MiniMax-M3</td><td>40.0</td><td>40.0</td><td>33.0</td><td>36.7</td><td>30.7</td></tr><tr><td>GLM-5.3</td><td>90.0</td><td>86.2</td><td>76.1</td><td>81.4</td><td>65.9</td></tr><tr><td colspan="6">OpenClaw</td></tr><tr><td>GPT-5.6-Sol</td><td>86.7</td><td>82.2</td><td>72.7</td><td>55.3</td><td>50.0</td></tr><tr><td>Claude Opus 5</td><td>80.0</td><td>76.8</td><td>65.9</td><td>75.1</td><td>64.8</td></tr><tr><td>Qwen3.8-Max</td><td>86.7</td><td>84.2</td><td>73.9</td><td>83.2</td><td>71.6</td></tr><tr><td>Kimi K3</td><td>86.7</td><td>83.0</td><td>70.5</td><td>81.9</td><td>68.2</td></tr><tr><td>DeepSeek-V4-Pro</td><td>73.3</td><td>69.9</td><td>56.8</td><td>69.9</td><td>56.8</td></tr><tr><td>MiniMax-M3</td><td>80.0</td><td>77.5</td><td>69.3</td><td>74.8</td><td>65.9</td></tr><tr><td>GLM-5.3</td><td>86.7</td><td>82.5</td><td>72.7</td><td>81.5</td><td>70.5</td></tr></table>

Risk assignment determines whether retrieved FTO references remain in the high-risk shortlist. With GPT-5.6-Sol fixed, OpenClaw has higher macro recall than Claude Code, at 82.2 versus 79.4, but lower high-risk macro recall, at 55.3 versus 74.2. The retrieval advantage reverses after risk filtering because more annotated references are excluded from OpenClaw’s high-risk subset. Opus with Claude Code preserves substantially more of its retrieved evidence: macro recall changes only from 86.5 to 85.3 after filtering. These results establish two separate requirements for FTO: retrieving the relevant applications and assigning them a risk level that keeps them in the actionable shortlist.

## F.3 Design FTO

Table 10: Design FTO results over 80 cases. PRES@200 is the source evaluator’s ranksensitive retrieval score at cutoff 200; all three metrics use the assigned case set.
<table><tr><td>Model</td><td>Hit@200</td><td>Recall@200</td><td>PRES@200</td></tr><tr><td colspan="4">Codex</td></tr><tr><td>GPT-5.6-Sol</td><td>1.3</td><td>0.4</td><td>0.4</td></tr><tr><td>Qwen3.8-Max</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Kimi K3</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>MiniMax-M3</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>GLM-5.3</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td colspan="4">Claude Code</td></tr><tr><td>GPT-5.6-Sol</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Claude Opus 5</td><td>5.0</td><td>2.7</td><td>2.5</td></tr><tr><td>Qwen3.8-Max</td><td>3.8</td><td>3.8</td><td>3.7</td></tr><tr><td>Kimi K3</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>MiniMax-M3</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>GLM-5.3</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td colspan="4">Hermes</td></tr><tr><td>GPT-5.6-Sol</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Claude Opus 5</td><td>2.5</td><td>1.9</td><td>1.8</td></tr><tr><td>Qwen3.8-Max</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Kimi K3</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>MiniMax-M3</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>GLM-5.3</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td colspan="4">OpenClaw</td></tr><tr><td>GPT-5.6-Sol</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Claude Opus 5</td><td>1.3</td><td>1.3</td><td>1.2</td></tr><tr><td>Qwen3.8-Max</td><td>1.3</td><td>0.4</td><td>0.4</td></tr><tr><td>Kimi K3</td><td>1.3</td><td>0.4</td><td>0.4</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>MiniMax-M3</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>GLM-5.3</td><td>0.0</td><td>0.0</td><td>0.0</td></tr></table>

Reference coverage is uniformly low in design FTO, including for the best model–harness pairing. Qwen3.8-Max with Claude Code leads Recall@200 and PRES@200 at 3.8 and 3.7, while the highest Hit@200 is only 5.0, achieved by Opus with Claude Code. Thus, even the highest hit rate leaves nearly all cases without an annotated design in the returned results. Qwen’s recall falls to 0.0 with Codex and Hermes and to 0.4 with OpenClaw, so changing the harness does not remove this shared shortfall. The central problem is recovering the reference designs; rearranging the retrieved candidates cannot compensate for references absent from the list.

## F.4 Invalidity Search

Table 11: Invalidity-search results over the assigned 80 cases. Core denotes the source evaluator’s designated core references; All denotes the full annotated reference set. The core and full reference sets can have different eligible case counts, so their scores are reported separately rather than subtracted.
<table><tr><td>Model</td><td>Core hit @5</td><td>Core recall @5</td><td>All hit @5</td><td>All recall @5</td><td>Core hit @All</td><td>Core recall @All</td><td>All hit @All</td><td>All recall @All</td></tr><tr><td colspan="9">Codex</td></tr><tr><td>GPT-5.6-Sol</td><td>26.9</td><td>23.5</td><td>32.5</td><td>23.3</td><td>29.5</td><td>26.7</td><td>33.8</td><td>26.4</td></tr><tr><td>Qwen3.8-Max</td><td>30.8</td><td>23.7</td><td>42.5</td><td>22.6</td><td>41.0</td><td>32.5</td><td>53.8</td><td>31.2</td></tr><tr><td>Kimi K3</td><td>5.1</td><td>3.0</td><td>7.5</td><td>3.0</td><td>6.4</td><td>4.3</td><td>8.8</td><td>4.6</td></tr><tr><td>DeepSeek-V4-Pro</td><td>14.1</td><td>10.0</td><td>18.8</td><td>9.5</td><td>18.0</td><td>12.4</td><td>21.3</td><td>10.3</td></tr><tr><td>MiniMax-M3</td><td>1.3</td><td>1.3</td><td>1.3</td><td>0.8</td><td>1.3</td><td>1.3</td><td>1.3</td><td>0.8</td></tr><tr><td>GLM-5.3</td><td>29.5</td><td>22.8</td><td>43.8</td><td>26.2</td><td>39.7</td><td>31.1</td><td>51.3</td><td>33.8</td></tr><tr><td colspan="9">Claude Code</td></tr><tr><td>GPT-5.6-Sol</td><td>30.8</td><td>23.3</td><td>36.3</td><td>21.4</td><td>39.7</td><td>31.0</td><td>48.8</td><td>31.8</td></tr><tr><td>Claude Opus 5</td><td>28.2</td><td>22.2</td><td>38.8</td><td>20.6</td><td>34.6</td><td>27.1</td><td>48.8</td><td>28.7</td></tr><tr><td>Qwen3.8-Max</td><td>14.1</td><td>11.3</td><td>28.8</td><td>15.2</td><td>21.8</td><td>18.2</td><td>35.0</td><td>21.0</td></tr><tr><td>Kimi K3</td><td>24.4</td><td>19.0</td><td>37.5</td><td>19.1</td><td>30.8</td><td>25.4</td><td>45.0</td><td>25.8</td></tr><tr><td>DeepSeek-V4-Pro</td><td>19.2</td><td>13.7</td><td>31.3</td><td>16.7</td><td>26.9</td><td>21.8</td><td>42.5</td><td>24.9</td></tr><tr><td>MiniMax-M3</td><td>10.3</td><td>6.8</td><td>17.5</td><td>8.3</td><td>14.1</td><td>10.5</td><td>22.5</td><td>12.4</td></tr><tr><td>GLM-5.3</td><td>38.5</td><td>30.3</td><td>52.5</td><td>28.3</td><td>42.3</td><td>33.6</td><td>57.5</td><td>32.4</td></tr><tr><td colspan="9">Hermes</td></tr><tr><td>GPT-5.6-Sol</td><td>9.0</td><td>9.0</td><td>11.3</td><td>8.4</td><td>10.3</td><td>9.6</td><td>13.8</td><td>10.3</td></tr><tr><td>Claude Opus 5</td><td>56.4</td><td>53.9</td><td>61.3</td><td>42.2</td><td>60.3</td><td>58.7</td><td>67.5</td><td>55.0</td></tr><tr><td>Qwen3.8-Max</td><td>20.5</td><td>20.5</td><td>25.0</td><td>17.0</td><td>20.5</td><td>20.5</td><td>27.5</td><td>20.5</td></tr><tr><td>Kimi K3</td><td>42.3</td><td>36.8</td><td>50.0</td><td>28.6</td><td>46.2</td><td>40.6</td><td>55.0</td><td>34.5</td></tr><tr><td>DeepSeek-V4-Pro</td><td>33.3</td><td>27.1</td><td>38.8</td><td>23.7</td><td>39.7</td><td>33.3</td><td>45.0</td><td>29.9</td></tr><tr><td>MiniMax-M3</td><td>6.4</td><td>5.8</td><td>8.8</td><td>4.8</td><td>6.4</td><td>5.8</td><td>11.3</td><td>6.0</td></tr><tr><td>GLM-5.3</td><td>41.0</td><td>34.6</td><td>51.3</td><td>32.6</td><td>43.6</td><td>37.6</td><td>57.5</td><td>40.1</td></tr><tr><td colspan="9">OpenClaw</td></tr><tr><td>GPT-5.6-Sol</td><td>24.4</td><td>18.2</td><td>35.0</td><td>18.1</td><td>30.8</td><td>24.4</td><td>41.3</td><td>24.8</td></tr><tr><td>Claude Opus 5</td><td>9.0</td><td>7.7</td><td>10.0</td><td>6.2</td><td>10.3</td><td>9.0</td><td>12.5</td><td>7.7</td></tr><tr><td>Qwen3.8-Max</td><td>20.5</td><td>15.2</td><td>28.8</td><td>14.7</td><td>23.1</td><td>17.7</td><td>31.3</td><td>17.3</td></tr><tr><td>Kimi K3</td><td>19.2</td><td>15.0</td><td>25.0</td><td>14.3</td><td>23.1</td><td>18.2</td><td>31.3</td><td>17.7</td></tr><tr><td>DeepSeek-V4-Pro</td><td>7.7</td><td>6.4</td><td>8.8</td><td>6.5</td><td>7.7</td><td>6.4</td><td>8.8</td><td>6.5</td></tr><tr><td>MiniMax-M3</td><td>9.0</td><td>7.7</td><td>12.5</td><td>7.6</td><td>9.0</td><td>9.0</td><td>16.3</td><td>10.5</td></tr><tr><td>GLM-5.3</td><td>21.8</td><td>17.1</td><td>33.8</td><td>17.0</td><td>24.4</td><td>20.0</td><td>38.8</td><td>21.3</td></tr></table>

Opus with Hermes has a consistent invalidity-search advantage across both reference groups and both cutoffs, but its reference coverage remains incomplete. It leads all eight metrics, with full-reference recall increasing from 42.2 in the top five results to 55.0 over the complete list. The additional results improve coverage, yet 45.0% of annotated references remain unrecovered on average. This harness advantage is specific to the model: Opus reaches only 28.7 full-list recall with Claude Code, whereas GPT-5.6-Sol performs better with Claude Code than with Hermes, at 31.8 versus 10.3. The table therefore establishes a strong Opus–Hermes pairing, rather than a general retrieval advantage for Hermes.

## F.5 Claim Charting

Table 12: Claim-charting results over all 30 assigned cases. SP-F1 measures feature extraction, CP-F1 measures feature-level evidence comparison, and PB-AC is the source evaluator’s patentability-judgment accuracy. All columns use full-set scores rather than scores conditional on completed outputs.
<table><tr><td>Model</td><td>SP-F1</td><td>CP-F1</td><td>PB-AC</td></tr><tr><td colspan="4">Codex</td></tr><tr><td>GPT-5.6-Sol</td><td>73.7</td><td>53.3</td><td>72.9</td></tr><tr><td>Qwen3.8-Max</td><td>67.6</td><td>54.9</td><td>73.1</td></tr><tr><td>Kimi K3</td><td>62.3</td><td>48.2</td><td>69.5</td></tr><tr><td>DeepSeek-V4-Pro</td><td>12.4</td><td>9.3</td><td>13.5</td></tr><tr><td>MiniMax-M3</td><td>40.1</td><td>30.3</td><td>48.3</td></tr><tr><td>GLM-5.3</td><td>30.0</td><td>24.8</td><td>35.9</td></tr><tr><td colspan="4">Claude Code</td></tr><tr><td>GPT-5.6-Sol</td><td>70.6</td><td>54.1</td><td>72.5</td></tr><tr><td>Claude Opus 5</td><td>71.3</td><td>54.0</td><td>72.6</td></tr><tr><td>Qwen3.8-Max</td><td>51.7</td><td>50.7</td><td>72.7</td></tr><tr><td>Kimi K3</td><td>39.5</td><td>31.8</td><td>37.7</td></tr><tr><td>DeepSeek-V4-Pro</td><td>54.0</td><td>40.8</td><td>63.0</td></tr><tr><td>MiniMax-M3</td><td>25.7</td><td>21.1</td><td>34.9</td></tr><tr><td>GLM-5.3</td><td>5.0</td><td>5.5</td><td>6.0</td></tr><tr><td colspan="4">Hermes</td></tr><tr><td>GPT-5.6-Sol</td><td>14.8</td><td>12.7</td><td>15.3</td></tr><tr><td>Claude Opus 5</td><td>73.9</td><td>55.1</td><td>74.4</td></tr><tr><td>Qwen3.8-Max</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Kimi K3</td><td>4.7</td><td>3.2</td><td>5.9</td></tr><tr><td>DeepSeek-V4-Pro</td><td>9.5</td><td>8.3</td><td>11.2</td></tr><tr><td>MiniMax-M3</td><td>13.9</td><td>8.4</td><td>12.9</td></tr><tr><td>GLM-5.3</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td colspan="4">OpenClaw</td></tr><tr><td>GPT-5.6-Sol</td><td>72.9</td><td>54.7</td><td>74.1</td></tr><tr><td>Claude Opus 5</td><td>20.0</td><td>16.3</td><td>20.1</td></tr><tr><td>Qwen3.8-Max</td><td>5.1</td><td>3.7</td><td>5.9</td></tr><tr><td>Kimi K3</td><td>5.0</td><td>5.8</td><td>6.0</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>MiniMax-M3</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>GLM-5.3</td><td>0.0</td><td>0.0</td><td>0.0</td></tr></table>

Accurate feature extraction does not ensure accurate feature-to-evidence alignment. Opus with Hermes leads all three claim-charting metrics, but its CP-F1 reaches only 55.1 compared with an SP-F1 of 73.9. The distinction also changes model rankings within Codex: GPT-5.6- Sol extracts features more accurately than Qwen3.8-Max, at 73.7 versus 67.6, while Qwen achieves higher CP-F1, at 54.9 versus 53.3. The model with the stronger extracted feature set is therefore not the model with the stronger evidence comparison. Evaluation must retain both stages, because an extraction score alone does not establish whether the resulting chart correctly maps those features to the supplied documents.

## F.6 Patent Drafting

Table 13: Drafting quality and pass rate. The expert-defined passing threshold is 80/100. Quality is the mean over successfully delivered drafts. Pass rate is 100 $\left. i : q _ { i } \geq 8 0 \right. | / 6 0 ;$ an unscored case does not pass. The expert-defined threshold identifies practically valuable, usable drafts. We do not report the full-set task total.
<table><tr><td>Model</td><td>Quality</td><td>Pass rate (%)</td></tr><tr><td colspan="3">Codex</td></tr><tr><td>GPT-5.6-Sol</td><td>84.7</td><td>76.7</td></tr><tr><td>Qwen3.8-Max</td><td>79.2</td><td>25.0</td></tr><tr><td>Kimi K3</td><td>81.2</td><td>46.7</td></tr><tr><td>DeepSeek-V4-Pro</td><td>79.4</td><td>35.0</td></tr><tr><td>MiniMax-M3</td><td>71.6</td><td>5.0</td></tr><tr><td>GLM-5.3</td><td>79.9</td><td>51.7</td></tr><tr><td colspan="3">Claude Code</td></tr><tr><td>GPT-5.6-Sol</td><td>84.0</td><td>65.0</td></tr><tr><td>Claude Opus 5 Qwen3.8-Max</td><td>82.8 80.8</td><td>66.7 63.3</td></tr><tr><td>Kimi K3</td><td>82.9</td><td>65.0</td></tr><tr><td>DeepSeek-V4-Pro</td><td>78.1</td><td>25.0</td></tr><tr><td>MiniMax-M3</td><td>72.4</td><td>28.3</td></tr><tr><td>GLM-5.3</td><td>82.3</td><td>66.7</td></tr><tr><td colspan="3">Hermes</td></tr><tr><td>GPT-5.6-Sol</td><td>85.0</td><td>53.3</td></tr><tr><td>Claude Opus 5</td><td>81.0</td><td>53.3</td></tr><tr><td>Qwen3.8-Max</td><td>82.7</td><td>30.0</td></tr><tr><td>Kimi K3</td><td>81.3</td><td>50.0</td></tr><tr><td>DeepSeek-V4-Pro</td><td>77.6</td><td>35.0</td></tr><tr><td>MiniMax-M3</td><td>71.2</td><td>16.7</td></tr><tr><td>GLM-5.3</td><td>77.4</td><td>40.0</td></tr><tr><td colspan="3">OpenClaw</td></tr><tr><td>GPT-5.6-Sol</td><td>83.3</td><td>66.7</td></tr><tr><td>Claude Opus 5</td><td>81.2</td><td>55.0</td></tr><tr><td>Qwen3.8-Max</td><td>81.6</td><td>53.3</td></tr><tr><td>Kimi K3</td><td>82.1</td><td>41.7</td></tr><tr><td>DeepSeek-V4-Pro</td><td>79.9</td><td>35.0</td></tr><tr><td>MiniMax-M3</td><td>72.8</td><td>23.3</td></tr><tr><td>GLM-5.3</td><td>83.6</td><td>1.7</td></tr></table>

GPT-5.6-Sol with Codex produces the highest proportion of drafts meeting the expertdefined usability standard, despite GPT with Hermes having the highest mean quality. Experts set the passing threshold at 80/100 to identify drafts with practical value that meet the benchmark’s requirements for a usable deliverable. GPT reaches a 76.7% pass rate with Codex, compared with 53.3% with Hermes, although their quality means are close at 84.7 and 85.0. The mean describes scored documents, while pass rate measures usable deliverables over all assigned cases. For this task, selecting by mean quality alone chooses a different configuration from selecting by the frequency of delivering a usable draft.

## F.7 Office-Action Response

Table 14: OAR quality and pass rate: quality averages only successfully scored responses, whereas pass rate uses all 40 assigned cases as its denominator, with unscored cases counted as not passing. The expert-defined threshold of 80/100 identifies practically valuable, usable responses.
<table><tr><td>Model</td><td>Quality</td><td>Pass rate (%)</td></tr><tr><td colspan="3">Codex</td></tr><tr><td>GPT-5.6-Sol</td><td>34.8</td><td>0.0</td></tr><tr><td>Qwen3.8-Max</td><td>27.6</td><td>0.0</td></tr><tr><td>Kimi K3</td><td>36.9</td><td>0.0</td></tr><tr><td>DeepSeek-V4-Pro</td><td>21.2</td><td>0.0</td></tr><tr><td>MiniMax-M3</td><td>0.0</td><td>0.0</td></tr><tr><td>GLM-5.3</td><td>28.0</td><td>0.0</td></tr><tr><td colspan="3">Claude Code</td></tr><tr><td>GPT-5.6-Sol</td><td>38.0</td><td>2.5</td></tr><tr><td>Claude Opus 5</td><td>25.7</td><td>0.0</td></tr><tr><td>Qwen3.8-Max</td><td>17.3</td><td>0.0</td></tr><tr><td>Kimi K3</td><td>12.2</td><td>0.0</td></tr><tr><td>DeepSeek-V4-Pro</td><td>22.3</td><td>0.0</td></tr><tr><td>MiniMax-M3</td><td>0.6</td><td>0.0</td></tr><tr><td>GLM-5.3</td><td>0.0</td><td>0.0</td></tr><tr><td colspan="3">Hermes</td></tr><tr><td>GPT-5.6-Sol</td><td>54.8</td><td>22.5</td></tr><tr><td>Claude Opus 5</td><td>22.7</td><td>0.0</td></tr><tr><td>Qwen3.8-Max</td><td>7.3</td><td>0.0</td></tr><tr><td>Kimi K3</td><td>24.0</td><td>2.5</td></tr><tr><td>DeepSeek-V4-Pro</td><td>23.4</td><td>0.0</td></tr><tr><td>MiniMax-M3</td><td>15.1</td><td>0.0</td></tr><tr><td>GLM-5.3</td><td>27.2</td><td>0.0</td></tr><tr><td colspan="3">OpenClaw</td></tr><tr><td>GPT-5.6-Sol</td><td>53.3</td><td>15.0</td></tr><tr><td>Claude Opus 5</td><td>22.2</td><td>0.0</td></tr><tr><td>Qwen3.8-Max</td><td>23.3</td><td>0.0</td></tr><tr><td>Kimi K3</td><td>30.3</td><td>2.5</td></tr><tr><td>DeepSeek-V4-Pro</td><td>28.1</td><td>0.0</td></tr><tr><td>MiniMax-M3</td><td>24.5</td><td>5.0</td></tr><tr><td>GLM-5.3</td><td>29.5</td><td>0.0</td></tr></table>

Usable OAR responses remain uncommon across configurations. GPT-5.6-Sol with Hermes leads mean quality at 54.8 and passes nine of forty cases, giving a pass rate of 22.5%. The same model with OpenClaw passes six cases, while Claude Code passes one. MiniMax with OpenClaw passes two cases, and Kimi passes one case each with Hermes and OpenClaw. Every other configuration has zero passing cases. Even the leading configuration therefore fails to meet the expert-defined usability standard on most assigned cases; high-quality responses occur in individual cases without establishing consistent delivery.

Table 15: OAR basic-quality criteria. G1: case understanding; G2: response strategy; G7: logical consistency; G8: response completeness; G9: deliverable completeness and text quality; G10: jurisdictional compliance; G11: factual and citation accuracy. Each criterion is averaged over its applicable scored cases on a 0–100 scale. Missing outputs and failed grading are excluded.
<table><tr><td>Model</td><td>G1</td><td>G2</td><td>G7</td><td>G8</td><td>G9</td><td>G10</td><td>G11</td></tr><tr><td colspan="8">Codex</td></tr><tr><td>GPT-5.6-Sol</td><td>23.8</td><td>17.5</td><td>91.3</td><td>97.5</td><td>71.3</td><td>87.5</td><td>15.0</td></tr><tr><td>Qwen3.8-Max</td><td>0.0</td><td>33.3</td><td>19.4</td><td>97.2</td><td>8.3</td><td>16.7</td><td>0.0</td></tr><tr><td>Kimi K3</td><td>21.4</td><td>50.0</td><td>42.9</td><td>78.6</td><td>14.3</td><td>50.0</td><td>14.3</td></tr><tr><td>DeepSeek-V4-Pro</td><td>12.5</td><td>16.7</td><td>54.2</td><td>70.8</td><td>29.2</td><td>54.2</td><td>0.0</td></tr><tr><td>MiniMax-M3</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>GLM-5.3</td><td>16.7</td><td>20.0</td><td>55.0</td><td>91.7</td><td>58.3</td><td>65.0</td><td>13.3</td></tr><tr><td colspan="8">Claude Code</td></tr><tr><td>GPT-5.6-Sol</td><td>28.8</td><td>30.0</td><td>86.3</td><td>93.8</td><td>57.5</td><td>90.0</td><td>8.8</td></tr><tr><td>Claude Opus 5</td><td>9.0</td><td>14.1</td><td>41.0</td><td>96.2</td><td>29.5</td><td>30.8</td><td>0.0</td></tr><tr><td>Qwen3.8-Max</td><td>6.6</td><td>14.5</td><td>26.3</td><td>72.4</td><td>2.6</td><td>35.5</td><td>0.0</td></tr><tr><td>Kimi K3</td><td>6.3</td><td>11.3</td><td>26.3</td><td>43.8</td><td>7.5</td><td>18.8</td><td>0.0</td></tr><tr><td>DeepSeek-V4-Pro</td><td>7.5</td><td>20.0</td><td>40.0</td><td>92.5</td><td>20.0</td><td>53.8</td><td>8.8</td></tr><tr><td>MiniMax-M3</td><td>0.0</td><td>0.0</td><td>15.2</td><td>0.0</td><td>0.0</td><td>4.5</td><td>0.0</td></tr><tr><td>GLM-5.3</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td colspan="8">Hermes</td></tr><tr><td>GPT-5.6-Sol</td><td>47.5</td><td>56.3</td><td>88.8</td><td>100.0</td><td>66.3</td><td>93.8</td><td>18.8</td></tr><tr><td>Claude Opus 5</td><td>4.2</td><td>12.5</td><td>30.6</td><td>98.6</td><td>30.6</td><td>26.4</td><td>0.0</td></tr><tr><td>Qwen3.8-Max</td><td>0.0</td><td>0.0</td><td>50.0</td><td>100.0</td><td>0.0</td><td>25.0</td><td>0.0</td></tr><tr><td>Kimi K3</td><td>10.3</td><td>23.5</td><td>38.2</td><td>95.6</td><td>36.8</td><td>47.1</td><td>5.9</td></tr><tr><td>DeepSeek-V4-Pro</td><td>6.3</td><td>25.0</td><td>75.0</td><td>100.0</td><td>37.5</td><td>81.3</td><td>25.0</td></tr><tr><td>MiniMax-M3</td><td>5.6</td><td>11.1</td><td>24.1</td><td>88.9</td><td>14.8</td><td>18.5</td><td>7.4</td></tr><tr><td>GLM-5.3</td><td>8.0</td><td>28.0</td><td>38.0</td><td>94.0</td><td>12.0</td><td>32.0</td><td>8.0</td></tr><tr><td colspan="8">OpenClaw</td></tr><tr><td>GPT-5.6-Sol</td><td>39.1</td><td>46.9</td><td>70.3</td><td>95.3</td><td>32.8</td><td>87.5</td><td>18.8</td></tr><tr><td>Claude Opus 5</td><td>25.0</td><td>0.0</td><td>25.0</td><td>100.0</td><td>25.0</td><td>50.0</td><td>0.0</td></tr><tr><td>Qwen3.8-Max</td><td>0.0</td><td>60.0</td><td>0.0</td><td>100.0</td><td>0.0</td><td>30.0</td><td>10.0</td></tr><tr><td>Kimi K3</td><td>16.7</td><td>38.9</td><td>25.0</td><td>97.2</td><td>19.4</td><td>58.3</td><td>8.3</td></tr><tr><td>DeepSeek-V4-Pro</td><td>18.6</td><td>31.4</td><td>57.1</td><td>95.7</td><td>27.1</td><td>65.7</td><td>5.7</td></tr><tr><td>MiniMax-M3</td><td>10.0</td><td>20.0</td><td>18.0</td><td>92.0</td><td>20.0</td><td>38.0</td><td>0.0</td></tr><tr><td>GLM-5.3</td><td>25.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>0.0</td><td>50.0</td><td>0.0</td></tr></table>

Completeness does not establish a reliable factual or strategic basis. Under Claude Code, GPT-5.6-Sol scores 93.8 on response completeness but only 28.8 on case understanding, 30.0 on response strategy, and 8.8 on factual and citation accuracy. Opus exhibits the same separation, with completeness at 96.2 and factual accuracy at 0.0. Hermes improves GPT’s understanding and strategy scores to 47.5 and 56.3, yet factual accuracy remains 18.8 despite complete objection coverage. These results distinguish responding to the listed objections from interpreting the supplied materials accurately and building a defensible response.

Table 16: OAR amendment criteria. G3: amendments addressing the objections; G4: proportionality and preservation of scope; G5: amendment legality and form. Required but omitted amendments retain zero scores. Each criterion is averaged over its applicable scored cases on a 0–100 scale. Missing outputs and failed grading are excluded.
<table><tr><td>Model</td><td>G3</td><td>G4</td><td>G5</td></tr><tr><td>Codex</td><td></td><td></td><td></td></tr><tr><td>GPT-5.6-Sol</td><td>0.0</td><td>3.8</td><td>3.8</td></tr><tr><td>Qwen3.8-Max</td><td>17.9</td><td>60.7</td><td>21.4</td></tr><tr><td>Kimi K3</td><td>0.0</td><td>50.0</td><td>33.3</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.0</td><td>20.0</td><td>20.0</td></tr><tr><td>MiniMax-M3</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>GLM-5.3</td><td>8.7</td><td>26.1</td><td>17.4</td></tr><tr><td>Claude Code</td><td></td><td></td><td></td></tr><tr><td>GPT-5.6-Sol</td><td>12.0</td><td>24.0</td><td>20.0</td></tr><tr><td>Claude Opus 5</td><td>50.0</td><td>50.0</td><td>50.0</td></tr><tr><td>Qwen3.8-Max</td><td>6.9</td><td>22.4</td><td>6.9</td></tr><tr><td>Kimi K3</td><td>6.1</td><td>6.1</td><td>6.1</td></tr><tr><td>DeepSeek-V4-Pro</td><td>4.3</td><td>39.1</td><td>13.0</td></tr><tr><td>MiniMax-M3</td><td>0.0</td><td>1.6</td><td>0.0</td></tr><tr><td>GLM-5.3</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Hermes</td><td></td><td></td><td></td></tr><tr><td>GPT-5.6-Sol</td><td>43.3</td><td>50.0</td><td>40.0</td></tr><tr><td>Claude Opus 5</td><td>5.6</td><td>16.7</td><td>11.1</td></tr><tr><td>Qwen3.8-Max</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Kimi K3</td><td>8.7</td><td>30.4</td><td>10.9</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.0</td><td>20.0</td><td>0.0</td></tr><tr><td>MiniMax-M3</td><td>5.6</td><td>13.9</td><td>5.6</td></tr><tr><td>GLM-5.3</td><td>11.9</td><td>45.2</td><td>19.0</td></tr><tr><td>OpenClaw</td><td></td><td></td><td></td></tr><tr><td>GPT-5.6-Sol</td><td>34.1</td><td>59.1</td><td>43.2</td></tr><tr><td>Claude Opus 5</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Qwen3.8-Max</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Kimi K3</td><td>28.6</td><td>50.0</td><td>21.4</td></tr><tr><td>DeepSeek-V4-Pro</td><td>11.4</td><td>38.6</td><td>22.7</td></tr><tr><td>MiniMax-M3</td><td>10.5</td><td>36.8</td><td>18.4</td></tr><tr><td>GLM-5.3</td><td>0.0</td><td>0.0</td><td>0.0</td></tr></table>

Amendment quality is limited even when the need for amendment is recognized. Under Claude Code, GPT-5.6-Sol scores 12.0 for matching amendments to the objections, 24.0 for proportionality, and 20.0 for legality and form across 25 applicable cases. Under Hermes, the corresponding means are 43.3, 50.0, and 40.0 across 30 applicable cases. Opus reaches 50.0 on all three criteria under Claude Code, but only four cases are applicable. These results identify a concrete failure in executing suitable amendments, while the differing applicable case sets prevent treating the raw means as a controlled ranking of amendment ability.

Table 17: OAR inventive-step argumentation. G6.1.1: closest prior art and distinguishing features; G6.1.2: technical effects and objective technical problem; G6.1.3: other references and common knowledge; G6.1.4: motivation to combine. Each criterion is averaged over its applicable scored cases on a 0–100 scale. Missing outputs and failed grading are excluded.
<table><tr><td>Model</td><td>G6.1.1</td><td>G6.1.2</td><td>G6.1.3</td><td>G6.1.4</td></tr><tr><td colspan="5">Codex</td></tr><tr><td>GPT-5.6-Sol</td><td>46.3</td><td>25.9</td><td>45.8</td><td>46.3</td></tr><tr><td>Qwen3.8-Max</td><td>29.2</td><td>30.8</td><td>40.0</td><td>25.0</td></tr><tr><td>Kimi K3</td><td>66.7</td><td>33.3</td><td>0.0</td><td>50.0</td></tr><tr><td>DeepSeek-V4-Pro</td><td>40.0</td><td>10.0</td><td>20.0</td><td>15.0</td></tr><tr><td>MiniMax-M3</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>GLM-5.3</td><td>52.5</td><td>27.5</td><td>31.6</td><td>23.7</td></tr><tr><td colspan="5">Claude Code</td></tr><tr><td>GPT-5.6-Sol</td><td>40.7</td><td>27.8</td><td>50.0</td><td>51.9</td></tr><tr><td>Claude Opus 5</td><td>42.3</td><td>20.4</td><td>33.3</td><td>30.8</td></tr><tr><td>Qwen3.8-Max</td><td>24.0</td><td>14.0</td><td>21.7</td><td>8.0</td></tr><tr><td>Kimi K3</td><td>19.2</td><td>7.7</td><td>20.8</td><td>11.5</td></tr><tr><td>DeepSeek-V4-Pro</td><td>33.9</td><td>8.9</td><td>25.0</td><td>11.1</td></tr><tr><td>MiniMax-M3</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>GLM-5.3</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td colspan="5">Hermes</td></tr><tr><td>GPT-5.6-Sol</td><td>57.4</td><td>51.9</td><td>52.1</td><td>53.7</td></tr><tr><td>Claude Opus 5</td><td>27.1</td><td>14.6</td><td>40.0</td><td>20.5</td></tr><tr><td>Qwen3.8-Max</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Kimi K3</td><td>25.0</td><td>25.0</td><td>21.1</td><td>16.7</td></tr><tr><td>DeepSeek-V4-Pro</td><td>62.5</td><td>0.0</td><td>25.0</td><td>0.0</td></tr><tr><td>MiniMax-M3</td><td>26.5</td><td>8.8</td><td>10.7</td><td>9.4</td></tr><tr><td>GLM-5.3</td><td>38.5</td><td>23.1</td><td>13.6</td><td>26.9</td></tr><tr><td colspan="5">OpenClaw</td></tr><tr><td>GPT-5.6-Sol</td><td>52.5</td><td>45.0</td><td>55.9</td><td>50.0</td></tr><tr><td>Claude Opus 5</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Qwen3.8-Max</td><td>0.0</td><td>0.0</td><td>50.0</td><td>0.0</td></tr><tr><td>Kimi K3</td><td>43.8</td><td>12.5</td><td>42.9</td><td>12.5</td></tr><tr><td>DeepSeek-V4-Pro</td><td>40.9</td><td>20.5</td><td>21.1</td><td>14.3</td></tr><tr><td>MiniMax-M3</td><td>40.0</td><td>10.0</td><td>23.1</td><td>14.3</td></tr><tr><td>GLM-5.3</td><td>50.0</td><td>50.0</td><td>50.0</td><td>0.0</td></tr></table>

Connecting technical differences to their effects and the objective technical problem is a specific weakness in the Claude Code results. GPT-5.6-Sol scores 40.7 on prior-art and feature analysis but 27.8 on effects and problem formulation; Opus scores 42.3 and 20.4, respectively. For both models, the latter is the lowest of the four inventive-step criteria. GPT with Hermes scores between 51.9 and 57.4 across these criteria, showing more balanced argumentation. Identifying a distinguishing feature alone therefore does not establish the reasoning needed to defend inventive step.

Table 18: OAR responses to other examination issues. G6.2: novelty; G6.3: eligibility; G6.4: disclosure, enablement, and written description; G6.5: clarity, support, and claim form; G6.6: essential features. Each criterion is averaged over its applicable scored cases on a 0–100 scale. Missing outputs and failed grading are excluded.
<table><tr><td>Model</td><td>G6.2</td><td>G6.3</td><td>G6.4</td><td>G6.5</td><td>G6.6</td></tr><tr><td colspan="6">Codex</td></tr><tr><td>GPT-5.6-Sol</td><td>36.7</td><td>50.0</td><td>16.7</td><td>20.4</td><td>50.0</td></tr><tr><td>Qwen3.8-Max</td><td>38.9</td><td>100.0</td><td>100.0</td><td>26.9</td><td>50.0</td></tr><tr><td>Kimi K3</td><td>75.0</td><td>0.0</td><td>50.0</td><td>25.0</td><td>25.0</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.0</td><td>0.0</td><td>0.0</td><td>6.3</td><td>0.0</td></tr><tr><td>MiniMax-M3</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>GLM-5.3</td><td>31.8</td><td>25.0</td><td>0.0</td><td>28.9</td><td>0.0</td></tr><tr><td colspan="6">Claude Code</td></tr><tr><td>GPT-5.6-Sol</td><td>36.7</td><td>50.0</td><td>50.0</td><td>31.5</td><td>25.0</td></tr><tr><td>Claude Opus 5</td><td>16.7</td><td>25.0</td><td>0.0</td><td>22.2</td><td>0.0</td></tr><tr><td>Qwen3.8-Max</td><td>7.7</td><td>0.0</td><td>0.0</td><td>16.7</td><td>0.0</td></tr><tr><td>Kimi K3</td><td>10.0</td><td>0.0</td><td>0.0</td><td>9.3</td><td>25.0</td></tr><tr><td>DeepSeek-V4-Pro</td><td>18.8</td><td>50.0</td><td>50.0</td><td>18.5</td><td>50.0</td></tr><tr><td>MiniMax-M3</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>GLM-5.3</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td colspan="6">Hermes</td></tr><tr><td>GPT-5.6-Sol</td><td>56.7</td><td>75.0</td><td>37.5</td><td>51.9</td><td>50.0</td></tr><tr><td>Claude Opus 5</td><td>23.3</td><td>75.0</td><td>37.5</td><td>25.0</td><td>25.0</td></tr><tr><td>Qwen3.8-Max</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Kimi K3</td><td>30.8</td><td>50.0</td><td>100.0</td><td>23.9</td><td>0.0</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.0</td><td>0.0</td><td>100.0</td><td>8.3</td><td>0.0</td></tr><tr><td>MiniMax-M3</td><td>15.0</td><td>50.0</td><td>50.0</td><td>19.4</td><td>50.0</td></tr><tr><td>GLM-5.3</td><td>30.0</td><td>100.0</td><td>25.0</td><td>31.0</td><td>25.0</td></tr><tr><td colspan="6">OpenClaw</td></tr><tr><td>GPT-5.6-Sol</td><td>50.0</td><td>50.0</td><td>75.0</td><td>60.0</td><td>75.0</td></tr><tr><td>Claude Opus 5</td><td>0.0</td><td>0.0</td><td>0.0</td><td>25.0</td><td>0.0</td></tr><tr><td>Qwen3.8-Max</td><td>50.0</td><td>0.0</td><td>0.0</td><td>12.5</td><td>50.0</td></tr><tr><td>Kimi K3</td><td>8.3</td><td>0.0</td><td>100.0</td><td>27.3</td><td>50.0</td></tr><tr><td>DeepSeek-V4-Pro</td><td>23.1</td><td>0.0</td><td>0.0</td><td>23.9</td><td>50.0</td></tr><tr><td>MiniMax-M3</td><td>14.3</td><td>50.0</td><td>50.0</td><td>25.0</td><td>100.0</td></tr><tr><td>GLM-5.3</td><td>50.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr></table>

Novelty and clarity objections remain only partially resolved even for GPT-5.6-Sol. Under Claude Code, its novelty score is 36.7 across 15 applicable cases and its clarity/support score is 31.5 across 27; with Hermes, these rise to 56.7 and 51.9 over the same numbers of applicable cases. The results show a harness-dependent change in the quality of substantive responses for these two objection types. They do not establish a difficulty ranking across objection types: eligibility and essential-feature scores for GPT with Hermes each cover only two cases, while clarity/support covers 27.

## F.8 Patent Translation: ZH–EN

Table 19: ZH–EN translation: key quality measures. The suite size counts 80 document-level cases, with 881 items. The metric-specific evaluation sets contain 421 ordinary-text items, 24 document-accuracy items for SacreBLEU, and 205 and 124 terminology items for accuracy and consistency, respectively.
<table><tr><td>Model</td><td>BLEU-1</td><td>Document SacreBLEU</td><td>Term accuracy</td><td>Term consistency</td></tr><tr><td colspan="5">Codex</td></tr><tr><td>GPT-5.6-Sol</td><td>67.5</td><td>63.0</td><td>53.7</td><td>46.8</td></tr><tr><td>Qwen3.8-Max</td><td>69.1</td><td>65.4</td><td>54.6</td><td>50.8</td></tr><tr><td>Kimi K3</td><td>70.5</td><td>67.2</td><td>58.1</td><td>47.6</td></tr><tr><td>DeepSeek-V4-Pro</td><td>70.6</td><td>64.6</td><td>56.6</td><td>53.2</td></tr><tr><td>MiniMax-M3</td><td>69.7</td><td>64.4</td><td>52.7</td><td>46.0</td></tr><tr><td>GLM-5.3</td><td>70.2</td><td>64.2</td><td>57.6</td><td>47.6</td></tr><tr><td colspan="5">Claude Code</td></tr><tr><td>GPT-5.6-Sol</td><td>67.1</td><td>60.8</td><td>54.2</td><td>46.0</td></tr><tr><td>Claude Opus 5</td><td>70.0</td><td>67.3</td><td>55.6</td><td>50.0</td></tr><tr><td>Qwen3.8-Max</td><td>69.8</td><td>64.8</td><td>55.6</td><td>51.6</td></tr><tr><td>Kimi K3</td><td>70.5</td><td>66.8</td><td>58.5</td><td>49.2</td></tr><tr><td>DeepSeek-V4-Pro</td><td>70.8</td><td>67.5</td><td>58.5</td><td>49.2</td></tr><tr><td>MiniMax-M3</td><td>68.6</td><td>65.0</td><td>53.2</td><td>47.6</td></tr><tr><td>GLM-5.3</td><td>70.8</td><td>64.3</td><td>59.0</td><td>50.0</td></tr><tr><td colspan="5">Hermes</td></tr><tr><td>GPT-5.6-Sol</td><td>67.6</td><td>63.5</td><td>53.2</td><td>46.0</td></tr><tr><td>Claude Opus 5</td><td>69.8</td><td>66.5</td><td>57.1</td><td>49.2</td></tr><tr><td>Qwen3.8-Max</td><td>70.3</td><td>59.2</td><td>54.6</td><td>49.2</td></tr><tr><td>Kimi K3</td><td>69.1</td><td>67.2</td><td>56.6</td><td>48.4</td></tr><tr><td>DeepSeek-V4-Pro</td><td>70.7</td><td>65.4</td><td>56.6</td><td>51.6</td></tr><tr><td>MiniMax-M3</td><td>68.6</td><td>64.7</td><td>56.6</td><td>43.6</td></tr><tr><td>GLM-5.3</td><td>69.8</td><td>54.6</td><td>55.1</td><td>49.2</td></tr><tr><td colspan="5">OpenClaw</td></tr><tr><td>GPT-5.6-Sol</td><td>68.1</td><td>63.5</td><td>54.6</td><td>45.2</td></tr><tr><td>Claude Opus 5</td><td>70.4</td><td>66.7</td><td>57.6</td><td>48.4</td></tr><tr><td>Qwen3.8-Max</td><td>69.6</td><td>54.0</td><td>55.6</td><td>48.4</td></tr><tr><td>Kimi K3</td><td>70.1</td><td>50.6</td><td>57.1</td><td>50.0</td></tr><tr><td>DeepSeek-V4-Pro</td><td>71.1</td><td>61.8</td><td>59.5</td><td>53.2</td></tr><tr><td>MiniMax-M3</td><td>68.9</td><td>51.2</td><td>52.7</td><td>47.6</td></tr><tr><td>GLM-5.3</td><td>72.0</td><td>68.5</td><td>57.1</td><td>50.0</td></tr></table>

The strongest Chinese-to-English text-overlap scores do not identify the strongest terminology performance. GLM-5.3 with OpenClaw leads BLEU-1 and document SacreBLEU at 72.0 and 68.5, but DeepSeek-V4-Pro with OpenClaw leads terminology accuracy at 59.5 and shares the highest consistency score of 53.2. Terminology remains incomplete even at these maxima. Harness differences also depend on text length: Qwen3.8-Max has similar BLEU-1 scores with Codex and OpenClaw, at 69.1 and 69.6, but document SacreBLEU changes from 65.4 to 54.0. Ordinary-text overlap therefore misses both document-level differences and the separate requirement to select and reuse patent terminology correctly.

## F.9 Patent Translation: EN–ZH

Table 20: EN–ZH translation: key quality measures. The suite size counts 80 document-level cases with 881 items. The metric-specific evaluation sets contain 421 ordinary-text items, 24 document-accuracy items for SacreBLEU, and 205 and 124 terminology items for accuracy and consistency, respectively.
<table><tr><td>Model</td><td>ROUGE-1</td><td>Document SacreBLEU</td><td>Term accuracy</td><td>Term consistency</td></tr><tr><td colspan="5">Codex</td></tr><tr><td>GPT-5.6-Sol</td><td>76.1</td><td>65.8</td><td>66.3</td><td>75.8</td></tr><tr><td>Qwen3.8-Max</td><td>75.3</td><td>68.2</td><td>64.4</td><td>73.4</td></tr><tr><td>Kimi K3</td><td>77.4</td><td>67.5</td><td>65.4</td><td>75.8</td></tr><tr><td>DeepSeek-V4-Pro</td><td>77.0</td><td>70.0</td><td>66.8</td><td>75.0</td></tr><tr><td>MiniMax-M3</td><td>75.1</td><td>67.0</td><td>63.9</td><td>67.7</td></tr><tr><td>GLM-5.3</td><td>76.4</td><td>66.7</td><td>67.3</td><td>73.4</td></tr><tr><td colspan="5">Claude Code</td></tr><tr><td>GPT-5.6-Sol</td><td>75.3</td><td>65.7</td><td>66.3</td><td>78.2</td></tr><tr><td>Claude Opus 5</td><td>76.8</td><td>65.4</td><td>67.3</td><td>75.8</td></tr><tr><td>Qwen3.8-Max</td><td>77.1</td><td>65.4</td><td>63.9</td><td>75.8</td></tr><tr><td>Kimi K3</td><td>77.2</td><td>68.2</td><td>68.8</td><td>78.2</td></tr><tr><td>DeepSeek-V4-Pro</td><td>76.4</td><td>70.6</td><td>68.8</td><td>75.8</td></tr><tr><td>MiniMax-M3</td><td>74.9</td><td>66.4</td><td>62.9</td><td>64.5</td></tr><tr><td>GLM-5.3</td><td>77.9</td><td>63.9</td><td>69.3</td><td>77.4</td></tr><tr><td colspan="5">Hermes</td></tr><tr><td>GPT-5.6-Sol</td><td>75.7</td><td>65.8</td><td>65.9</td><td>79.0</td></tr><tr><td>Claude Opus 5</td><td>77.9</td><td>66.3</td><td>65.4</td><td>74.2</td></tr><tr><td>Qwen3.8-Max</td><td>76.6</td><td>57.9</td><td>63.4</td><td>75.0</td></tr><tr><td>Kimi K3</td><td>76.3</td><td>69.4</td><td>65.9</td><td>72.6</td></tr><tr><td>DeepSeek-V4-Pro</td><td>76.7</td><td>69.5</td><td>67.3</td><td>74.2</td></tr><tr><td>MiniMax-M3</td><td>75.2</td><td>63.6</td><td>64.4</td><td>70.2</td></tr><tr><td>GLM-5.3</td><td>77.0</td><td>55.9</td><td>66.3</td><td>75.8</td></tr><tr><td colspan="5">OpenClaw</td></tr><tr><td>GPT-5.6-Sol</td><td>74.0</td><td>66.7</td><td>64.4</td><td>79.0</td></tr><tr><td>Claude Opus 5</td><td>77.8</td><td>9.1</td><td>68.8</td><td>77.4</td></tr><tr><td>Qwen3.8-Max</td><td>76.0</td><td>49.7</td><td>62.9</td><td>73.4</td></tr><tr><td>Kimi K3</td><td>77.4</td><td>59.1</td><td>65.9</td><td>77.4</td></tr><tr><td>DeepSeek-V4-Pro</td><td>77.6</td><td>65.1</td><td>66.3</td><td>72.6</td></tr><tr><td>MiniMax-M3</td><td>75.0</td><td>53.7</td><td>62.0</td><td>67.7</td></tr><tr><td>GLM-5.3</td><td>78.1</td><td>55.3</td><td>68.3</td><td>79.8</td></tr></table>

Strong ordinary-text overlap does not ensure strong document translation in English-to-Chinese tasks. Opus with Hermes and OpenClaw has nearly identical ROUGE-1 scores of 77.9 and 77.8, but document SacreBLEU reaches 66.3 with Hermes and only 9.1 with OpenClaw. This contrast establishes a document-level difference that the ordinary-text score does not capture. Terminology accuracy and consistency also separate: GLM-5.3 with OpenClaw scores 79.8 on consistency but 68.3 on accuracy. Reusing a term consistently does not establish that the term was selected correctly. Document translation and terminology thus remain necessary evaluation dimensions alongside ordinary-text overlap.

## G Qualitative Case Comparisons

Each scenario compares models under a fixed harness and harnesses with a fixed model. Panels retain the curated records and original score precision. These selected runs illustrate local differences; full-set statistics are reported separately.

## G.1 Novelty Search

![](images/ee39c83f8ddf3e4581e44c0b4f83ed1b5f9d04bc183a3f2d1577f87e6df83eb6.jpg)  
(b) GLM-5.3: Codex versus Hermes.  
Figure 8: Novelty Search: fixed-harness comparison above and fixed-model comparison below.

Target mismatches cause missed reference families in both comparisons. In the radar-face case, GLM with Codex retrieves all five annotated families, whereas Kimi retrieves none and prioritizes a hand-gesture patent. In the X-ray alignment case, Codex retrieves two of three annotated families, whereas Hermes retrieves none and prioritizes a visible-light Talbot reference. Both contrasts expose a specific mismatch: shared radar or Fourier terminology does not preserve the target object, imaging modality, and measurement procedure.

![](images/40ac2804ac5709a9629d5e9288a5dfce36e1c5e7f92b280e12696519528e3f20.jpg)  
(b) Opus: Claude Code versus Hermes.  
Figure 9: Patent FTO: fixed-harness comparison above and fixed-model comparison below.

More candidates and high-risk labels do not ensure reference coverage. For the tilting fan, Opus with Claude Code ranks the annotated application first and labels it high risk. DeepSeek returns 42 candidates, including six high-risk candidates, but misses that application. For the folding scooter, Claude Code ranks the annotated two-stage locking reference first among 89 candidates. Hermes returns 100 candidates, including 15 high-risk candidates, but misses it. In both comparisons, a longer or explicitly risk-labeled list fails to recover the mechanism-specific reference.

![](images/2a95a52419de91b764c84c7d81550020c000d8cdcd1e4948112e6321a3ee57c7.jpg)  
(b) Opus: Claude Code versus OpenClaw.  
Figure 10: Design FTO: fixed-harness comparison above and fixed-model comparison below.

Visual misclassification prevents retrieval, while correct broad classification remains insufficient. For the metal tool, Qwen retrieves the annotated design at rank seven. GPT instead describes hair-thinning scissors with finger holes absent from the input and misses the reference. For the dome camera, OpenClaw identifies the broad product category but retrieves none of the three annotated designs. Claude Code retrieves one at rank twelve with a medium-risk label. The first contrast identifies a category error; the second establishes that category recognition alone does not recover the relevant appearance details. The latter result is partial coverage, not complete retrieval.

![](images/8eafd916c38c4929184e016c422c3c8de133da414c87632a726f1b87b448194d.jpg)  
(b) GLM-5.3: Hermes versus OpenClaw.  
Figure 11: Invalidity Search: fixed-harness comparison above and fixed-model comparison below.

Broad technical similarity does not replace mechanism-specific prior art. For the leadfree solder, GPT with Codex retrieves both annotated references, while DeepSeek misses both and relies on broader alloy ranges and routine-optimization arguments. For the ski binding, Hermes retrieves both annotated references and identifies the jaw, wall, and return arrangement that constrains release. OpenClaw misses both and asserts coverage using a different flange-and-plate mechanism. These cases distinguish retrieval of the designated technical evidence from a plausible general argument; neither the argument nor a reference hit by itself establishes a legal invalidity conclusion.

![](images/4ac0adbf6454e5f693d26a1bfdad49036034feb2c645f5dc0aa462ed91b02d4c.jpg)  
(b) DeepSeek-V4-Pro: Hermes versus Claude Code.  
Figure 12: Claim Charting: fixed-harness comparison above and fixed-model comparison below.

Incorrect mechanism reconstruction reverses feature-level disclosure judgments. For the tensioner, Opus correctly maps release of the pin to the untensioned spring and free plunger. MiniMax substitutes a gravity-return mechanism and incorrectly marks the feature as absent. For the siphon, Hermes maps the two legs of one U-shaped component to the suction and outlet portions. Claude Code recognizes the legs but adds a separate-partition requirement and marks the feature as absent. In both cases, the negative judgment arises from changing the mechanism or adding a condition beyond the feature being compared.

![](images/b11a29e81d918b9fbdc00a00e058ea7b27ca9fb5d38b5bb1aec850ba2f1127ec.jpg)  
(a) Claude Code: Kimi K3 versus MiniMax-M3.

![](images/cd10478c17f35e011df6da5abdd01d788b2798e21f18da96435db6cce63fbf11.jpg)  
(b) Opus: Hermes versus OpenClaw.  
Figure 13: Patent Drafting: fixed-harness comparison above and fixed-model comparison below.

Claim quality depends on preserving disclosed relationships and optionality. For the dyeing machine, Kimi preserves the immersion, squeezing, and evaporation sequence within one chamber; MiniMax confuses containment with processing order and misplaces components. For the battery undertray, Hermes keeps optional resin impregnation in dependent claims, while OpenClaw makes it mandatory in the independent method claim. The first error changes the disclosed apparatus relationship; the second narrows the claimed method by requiring an optional step. These identified rubric errors explain local quality differences without attributing the entire document-score gap to a single criterion.

![](images/d5e9f5f1b858489d29ad5c41400997364a117dddc35ae05f98bf47ded4fa6355.jpg)  
(a) Claude Code: GPT-5.6-Sol versus Qwen3.8-Max.

![](images/5fac9788705396b950a78af7cab1ba7c98e2565a5ad4fef2b36683a95e26b778.jpg)  
(b) GPT-5.6-Sol: Hermes versus Codex.  
Figure 14: Office-action response: fixed-harness comparison above and fixed-model comparison below. The excerpts show how case interpretation and amendment execution affect the response to a specific objection.

An effective response must address the actual technical issue and implement the required correction. In the medical-use case, the EPO distinguishes treatment of a patient from in vitro treatment of tissue or an organ. Under Claude Code, GPT-5.6-Sol separates the claim formats and receives full credit on the novelty-response criterion. Qwen instead discusses DTPA chelators and different references, leaving the actual D-DT novelty objection unanswered. In the ejector case, holding GPT fixed isolates a difference in amendment execution: Hermes specifies the outlet reference and flow direction in amended claims, whereas Codex relies on the claimed function to dismiss the objection and leaves the clarification conditional. The clarity/support criterion assigns 1.0 to Hermes and 0.0 to Codex. Both comparisons show that a substantive response requires preserving the case facts and translating the objection into a concrete argument or amendment. GPT’s 93.3 total in the first case still includes a failure to supply corresponding description amendments, so full credit on the illustrated criterion does not imply that every response requirement has been met.

![](images/2e47b127a4a2806fa858145c9d6722fe5ff5e8b2a46600ca3283eacdef7a5b07.jpg)  
Figure 15: Claim-charting failure: MiniMax-M3 repeatedly revisits its reasoning, begins the answer late, and leaves the requested JSON unfinished.

MiniMax is asked to compare seven claims and return a complete JSON array. It repeatedly announces that it will finalize the answer, then reopens the analysis. When the run terminates after timeouts and a connection reset, the output remains unfinished within claim 4. The model does not complete the task within the allotted time. Its extended reasoning fails to produce the complete deliverable required for evaluation.

![](images/7a39a61863efcdc0eb1f266c63bfcec2c9c3f7f60e00eb3ba0450c0f003d2637.jpg)  
Figure 16: OAR failure: GLM-5.3 with Claude Code receives image blocks from a tool, and its text-only endpoint rejects the request before the response is completed.

For a five-page scanned CNIPA office action, Claude Code performs OCR and makes the extracted text available. GLM then requests PNG crops to check unclear passages, but its text-only endpoint rejects the returned images. No response is delivered. Although the harness has extracted the document text, the model fails to choose a viable next step and carry the task through to drafting. This case exposes a failure to use available intermediate results to complete the task.

## I Task Prompts

The following panels reproduce the supplied English display versions of the task prompts. They summarize the task instructions, input placeholders, constraints, and output contracts, rather than reproduce the executed templates verbatim. Requirements for Chinese output are retained. Both translation directions are shown separately.

![](images/31b5068faac6d6e01d0061e7e6726e043219298d8e12ba3ee48538e6235531a2.jpg)  
(a) Novelty Search

![](images/9126a9b91ec66c936c429b0c632ed52f4b6dd76d1ac1878b864ea51924b64174.jpg)  
(b) Patent FTO  
Figure 17: English prompt displays for Novelty Search and Patent FTO.

![](images/ceb136c39017ada7022e1c62a6df92cba06e7e0da14b64ae69a0e813dbdc7199.jpg)  
(b) Invalidity Search  
Figure 18: English prompt displays for Design FTO and Invalidity Search.

![](images/1b663315e542ec94c610f95ec7527bf56a62f77f2dc31be4f46b21b611b99766.jpg)  
(b) Office-Action Response  
Figure 19: English prompt displays for Claim Charting and Office-Action Response.

![](images/e76ad4c456ae2091889c1e79d235ba609b0c495b346b064add4fb9d5ed8d217f.jpg)  
(a) Chinese-to-English Translation

![](images/0bdf33c588e7688c6d66dfbb550ecde6a900474025814343db49f1057f42ce3a.jpg)  
(b) English-to-Chinese Translation  
Figure 20: English prompt displays for Chinese-to-English Translation and English-to-Chinese Translation.