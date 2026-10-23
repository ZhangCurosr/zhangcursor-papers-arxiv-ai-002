# TESSA: A Governance-Oriented Custom GPT for SME Guidance under the EU AI Act: Design, Implementation, and Evaluation

v0.9.7

Chalisa Veesommai Sillberg<sup>a,1</sup>, Kai-Kristian Kemell<sup>a,2</sup> Pekka Sillberg<sup>a,3</sup>, Katri Harjuveteläinen<sup>b,4</sup>, Mika Saari<sup>a,5</sup>, Pekka Abrahamsson<sup>a,6</sup>

<sup>a</sup> Tampere University, Faculty of Information Technology and Communication Sciences, Pori, Finland <sup>b</sup> Boutique Legal Consulting, Helsinki, Finland October 22, 2026 October 22, 2026

## Notes:

• This is the author’s version of the work

• This is a pre-print version of an article accepted to the 17th International Conference on Software Business (ICSOB 2026)

• The final version accepted to publishing the Springer Lecture Notes in Business Information Processing (LNBIP)

# TESSA: A Governance-Oriented Custom GPT for SME Guidance under the EU AI Act: Design, Implementation, and Evaluation

Chalisa Veesommai Sillberg<sup>a</sup> Kai-Kristian Kemell<sup>a</sup> Pekka Sillberg<sup>a</sup> Katri Harjuveteläinen<sup>b</sup> Mika Saari<sup>a</sup> Pekka Abrahamsson<sup>a</sup>

<sup>a</sup> Tampere University, Faculty of Information Technology and Communication Sciences, Pori, Finland <sup>b</sup> Boutique Legal Consulting, Helsinki, Finland

chalisa.sillberg,@tuni.fi

## Abstract

The increasing adoption of generative AI across industries created opportunities for innovation while introducing growing AI governance and regulatory challenges for organization. Small and medium-sized enterprises (SMEs) must navigate compliance requirements under the European Union Artificial Intelligence Act (EU AI Act). Although large language models (LLMs) ofer regulatory and educational support, concerns remain regarding governance alignment, reliability, and the practical applicability. To address this challenge, TESSA, a governance-oriented educational artifact, was developed within the OpenAI Custom GPT environment. Using a design science research approach, TESSA was evaluated through a multidimensional framework consisting of governance alignment assessment, human expert review, and consistency testing across 100 benchmark questions in ten thematic categories. The evaluation compared TESSA with a standard ChatGPT baseline using identical benchmark questions and assessment procedures. preliminary LLM-assisted governance alignment reached 90% for TESSA and 83% for the baseline, while independent validation by an AI regulation lawyer produced comparable rates of 71% and 70%, respectively. Human expert review indicated that both systems provided generally strong governance guidance. Consistency testing showed highly stable performance across repeated executions, with average overall consistency score of 0.969 for TESSA and 0.976 for the baseline. The findings suggest that governanceoriented grounding provides structured and practical compliance guidance while maintaining educational quality and response stability. The study contributes (i) a governance-oriented Custom GPT artifact for AI governance education for SMEs, (ii) a human-validated regulatory update mechanism,

and (iii) a multidimensional evaluation framework that can support the evaluation of governance-oriented AI artifact.

## Keywords:

European Union Artificial Intelligence Act (EU AI Act), AI Governance, Compliance Support, Design Science Research (DSR), Custom GPT

## 1 Introduction

The rapid adoption of generative AI has accelerated the integration of AI into products, services, and business processes across software businesses and other industries. Consequently, understanding and complying with AI regulations, particularly the EU AI Act, has become an important challenge. This is especially true for small and medium-sized enterprises (SMEs), which often lack dedicated legal expertise and governance resources while pursuing AI-driven innovation. The EU AI Act introduces a risk-based regulatory framework for AI systems and establishes obligations for providers, deployers, importers, distributors, and other stakeholders [1, 2].

The regulation aims to promote trustworthy AI and protect fundamental rights; however, its complexity creates challenges for SMEs, startups, researchers, and other non-legal stakeholders seeking to understand their responsibilities and compliance obligations.

Existing studies have identified several challenges for SMEs seeking to comply with the EU AI Act, including limited compliance resources, lack of regulatory expertise, uncertainty regarding risk classification and regulatory interpretation, and the need to balance compliance activities with AI system development [3, 4, 5, 6]. These challenges create a gap between regulatory requirements and practical implementation, highlighting the need for accessible governance support tools. This gap motivated the development of TESSA, a governance-oriented educational artifact designed to translate complex regulatory requirements into practical guidance.

Recent advances in large language models (LLMs) and generative AI have created new opportunities for educational, legal, and compliance support applications. LLMs have increasingly served as the foundation for intelligent support systems across multiple domains. In legal and regulatory contexts, systems such as Chatlaw have demonstrated the potential to provide advisory recommendations and improve access to specialized legal knowledge [7], and interactive legal assistants have been developed to support legal information access and non-specialists seeking legal guidance [8]. More broadly, AssistantGPT illustrates how LLM-based assistants can bridge the gap between complex system functionalities and intuitive user interaction [9], while GPTVoiceTasker represents an example of an LLM-based assistant designed to translate natural language requests into actionable task support [10]. However, the reliability of general-purpose LLMs remains a concern in domain-specific applications, motivating the use of domain-specific knowledge integration and grounding mechanisms to improve alignment, reduce hallucinations, and support trustworthy AI-assisted guidance.

Despite growing research on the EU AI Act, AI governance, and LLMbased support systems, limited empirical evidence exists regarding governanceoriented educational artifacts designed to support SME compliance and AI governance learning. To address this gap, we have developed and evaluated TESSA, designed to support EU AI Act education and compliance guidance. The study addresses the following research questions (RQs):

RQ1 To what extent can TESSA support SMEs and other non-legal stakeholders in understanding and applying EU AI Act requirements? and

RQ2 How does governance-oriented grounding influence the governance alignment, educational quality, and consistency of AI-generated compliance guidance compared with baseline ChatGPT?

This study makes the following contributions: (i) a governance-oriented grounding approach for Custom GPT systems supporting AI governance education, (ii) TESSA, a design science research artifact for SME guidance under the EU AI Act, (iii) a multidimensional evaluation framework, and (iv) an empirical comparison with baseline ChatGPT.

The remainder of the paper is organized as follows. Section 2 presents the research methodology, including the TESSA system architecture, governanceoriented grounding strategy, human-validated regulatory update mechanism, and multidimensional evaluation framework. Section 3 reports the evaluation results. Section 4 discusses the findings, implications, and limitations of the study. Finally, Section 5 concludes the paper and outlines future research directions.

## 2 TESSA Design and Research Methodology

## 2.1 Research Design

This study followed the design science research methodology (DSRM) proposed by Pefers et al. [11] to design, develop, and evaluate the TESSA artifact. The research problem addressed the challenge faced by SMEs, start-ups, students, researchers, and other non-legal dedicated resources in understanding and operationalizing the requirements of the EU AI Act. Following the DSRM process, the study identified the practical problem of SMEs struggling with EU AI Act compliance, defined the objective of developing TESSA, designed and developed the TESSA artifact within the OpenAI Custom GPT environment, demonstrated the artifact by applying it to representative EU AI Act compliance scenarios, and evaluated it through governance alignment assessment, human expert review, and consistency evaluation. This paper communicates the initial design and evaluation cycle, providing a foundation for future refinement.

Following the DSR process, the resulting research artifact, TESSA, was designed, implemented, and evaluated to support SMEs and other nonlegal stakeholders in interpreting EU AI Act requirements and translating them into practical governance-oriented actions. The evaluation investigated whether governance-oriented grounding improves the alignment, educational usefulness, and consistency of AI-assisted guidance.

## 2.2 TESSA System Architecture

TESSA was developed within OpenAI Custom GPT using the GPT-5 model available in January 2026. For comparison, the baseline system was the standard ChatGPT interface using the same underlying GPT-5 model without custom instructions, domain-specific knowledge sources, or governanceoriented grounding. TESSA supports users in interpreting EU AI Act requirements and translating them into practical governance-oriented actions. Figure 1 presents the deployed TESSA interface together with its overall system architecture<sup>1</sup>. The evaluated TESSA configuration consisted of four main components:

• System instructions

• Published EU AI Act guidance knowledge base [1]

• Use case AI List risk classification repository

• Human-validated regulatory updates

The architecture combines governance-oriented knowledge sources, a structured grounding process, and governance-focused outputs.

## 2.3 Governance-Oriented Grounding

TESSA was designed using a governance-oriented grounding approach to support users in interpreting and applying the requirements of the EU AI Act. Rather than relying solely on general-purpose language model knowledge, the system combines governance-oriented instructions, domain-specific knowledge resources, risk classification support, and human-validated regulatory updates.

When responding to a query, TESSA identifies the relevant use case, stakeholder role, and potential AI risk classification before generating

![](images/3a3899fa048ebbe6267b31a0296ae9c51e83f1825c215e2da1937fd89bf03a9a.jpg)  
Figure 1: TESSA implementation and system architecture within OpenAI Custom GPT environment.

governance-oriented guidance. The process follows the sequence: Question → Use Case → Role → Risk → Guidance → Output. The generated output is delivered through the structured governance guidance pattern (SGGP), providing role-specific guidance, risk classification information, compliance considerations, plain-language explanations, and practical recommendations. This grounding approach is intended to improve governance alignment, reduce risk of hallucination, and support actionable compliance guidance.

## 2.4 Human-Validated Regulatory Update Mechanism

As AI regulation and governance guidance continue to evolve, maintaining the currency of regulatory information represents a significant challenge for educational AI artifacts. To maintain current and validated regulatory knowledge, TESSA incorporates a human-validated regulatory update mechanism designed to support the continuous monitoring of relevant regulatory developments while maintaining governance oversight.

The update mechanism consists of two interconnected workflows. The first workflow focuses on automated information collection of candidate regulatory updates, including EU AI governance resources and academic databases, through an N8N-based automation workflow. Collected content is processed, merged, and filtered to remove duplicate entries before being stored in a structured review queue. The second workflow supports human review and controlled publication. Candidate updates are evaluated for relevance, accuracy, and consistency with the governance objectives of the system. Human reviewers assess whether the collected information should be approved or excluded from further use. Only validated updates are subsequently incorporated into TESSA through controlled update processes and associated repositories.

![](images/17cf18e86fc1bb4ce965eff490e4e5a8bacd2beec3d795b4b47cfccdc9245318.jpg)  
Figure 2: TESSA evaluation and maintenance framework. The figure combines the a) human-validated regulatory update mechanism and b) multidimensional evaluation framework used to assess the governance alignment, educational quality, and response consistency of TESSA across 100 benchmark scenarios.

Figure 2 a) represents the overall update mechanism. The process combines automated monitoring with human review to maintain the relevance of the regulatory information used by TESSA. Updates are reviewed before being incorporated into the system, providing an additional quality control step.

## 2.5 Multidimensional Evaluation Framework

The evaluation of TESSA followed a multidimensional framework designed to assess governance alignment, educational suitability, reproducibility, and response consistency. The framework was developed to evaluate i) the correctness of the responses and ii) their usefulness for SME-oriented compliance education and governance support. The evaluation consisted of four primary objectives (Obj):

1. To assess whether TESSA responses aligned with the published EU AI Act guidance knowledge base [1] and oficial EU AI Act sources [2].

2. To evaluate the educational suitability and governance quality of the responses.

3. To compare TESSA with baseline ChatGPT using identical benchmark scenarios.

4. To assess response consistency across repeated executions of the same scenarios.

To address these objectives, a multidimensional evaluation framework comprising four complementary dimensions was adopted: traceability validation, governance alignment evaluation, human expert review, and consistency evaluation. As shown in Figure 2 b) the framework consists of two evaluation phases. Phase A focuses on governance alignment through i) traceability validation, ii) governance alignment evaluation, and iii) human expert review. Phase B evaluates response consistency through repeated execution of benchmark scenarios.

The evaluation dimensions were informed by design science research (DSR) evaluation principles [12], trustworthy AI concepts reflected in the NIST AI Risk Management Framework [13], and structured qualitative assessment approaches. Governance alignment, legal alignment, educational clarity, governance suitability, and hallucination risk were assessed using a three-level rating structure adapted from QualSyst [14].

## 2.5.1 Benchmark Design and Traceability Validation

The benchmark scenarios were generated through a GPT-assisted process and subsequently reviewed and curated by the first author. The objective was to create realistic educational, governance, and compliance-related questions that SMEs, startups, students, researchers, and other non-legal stakeholders might ask when seeking guidance on the EU AI Act.

The benchmark was organized into ten categories: (i) AI literacy, (ii) startup and student projects, (iii) risk classification, (iv) compliance duties, (v) prohibited AI practices, (vi) high-risk AI, (vii) limited-risk AI, (viii)

minimal-risk AI, (ix) GPAI and foundation models, and (x) SME perspectives, each containing ten questions, resulting in a total of 100 evaluation scenarios. Following scenario generation, responses were produced using TESSA and baseline ChatGPT. The responses were then evaluated against expected outcomes developed from a published EU AI Act guidance knowledge base [1] and relevant oficial EU AI Act sources [2]. This traceability validation supported Objective 1 by assessing the alignment between responses and the grounding sources.

## 2.5.2 Governance Alignment Evaluation

A governance alignment evaluation was conducted to assess responses aligned with expected outcomes derived from a published EU AI Act guidance knowledge base [1] and oficial EU AI Act sources [2], and served as the benchmark for governance alignment assessment. This evaluation primarily addressed Obj. 1 and the supporting Obj. 3 by comparing TESSA and baseline ChatGPT under identical benchmark conditions.

Responses from 100 benchmark scenarios were compared with the corresponding expected outcome. The evaluation considered governance interpretation, risk classification, compliance recommendations, and consistency with the reference sources. Responses were assessed using the three-level alignment scale (Yes, Partial, No) presented in Table 1. The scoring approach was adapted from the QualSyst framework [14].

The evaluation consisted of two stages. First, an LLM-assisted preliminary review supported the initial classification of the responses. Second, the classifications were independently reviewed and validated by an experienced AI regulation lawyer practitioner with more than five years of professional legal experience in AI and data protection. The lawyer-reviewed assessments were used as the final governance alignment results reported in this study. The evaluation was intended to assess alignment with predefined educational and governance expectations rather than establish definitive legal interpretations. The resulting assessments enabled systematic comparison between TESSA and baseline ChatGPT and formed one component of the multidimensional evaluation framework.

## 2.5.3 Human Expert Review

In addition to benchmark-based evaluation, a human expert review was conducted to assess the educational and governance quality, as well as the legal interpretation of the responses. This evaluation supported Obj. 2 of the evaluation framework by examining whether the responses were suitable for the intended users seeking guidance on the EU AI Act. The review was conducted using a structured assessment framework consisting of five dimensions: (i) legal alignment, (ii) educational clarity, (iii) governance suitability, (iv) hallucination risk, and (v) overall assessment. The detailed assessment criteria are presented in Table 1. Similar to the governance alignment evaluation, qualitative expert assessments were recorded using a three-level rating structure consisting of strong, acceptable, and weak. The framework was designed to assess the factual and regulatory consistency of the responses as well as their usefulness for compliance education and governance support.

Table 1: Multidimensional Evaluation Criteria
<table><tr><td>Category</td><td>Criteria</td><td>Description</td></tr><tr><td colspan="3">1. Ground truth</td></tr><tr><td></td><td>Yes Partial</td><td>The response aligned with the expected outcome. The response was generally aligned but incomplete, lacked important</td></tr><tr><td></td><td></td><td>nuance, or omitted relevant governance considerations. The response was incorrect, misleading, or substantially inconsistent</td></tr><tr><td>No</td><td></td><td>with the expected outcome.</td></tr><tr><td colspan="3">2. Human expert review 2.1 Legal alignment</td></tr><tr><td colspan="3">Strong</td></tr><tr><td></td><td></td><td>Legally consistent and aligned with EU AI Act concepts and official sources.</td></tr><tr><td></td><td></td><td>Acceptable Mostly aligned but contains minor ambiguity or incomplete interpre- tation.</td></tr><tr><td></td><td>Weak</td><td>Legally misleading, incorrect, or significantly inconsistent with official</td></tr><tr><td colspan="3">2.2 Educational clarity</td></tr><tr><td></td><td>Strong</td><td>Clear, accessible, and understandable for SMEs and non-legal users.</td></tr><tr><td></td><td></td><td>Acceptable Generally understandable but may contain unclear or overly complex parts.</td></tr><tr><td></td><td>Weak</td><td>Confusing, poorly structured, or difficult for intended users to under- stand.</td></tr><tr><td colspan="3">2.3 Governance suitability</td></tr><tr><td></td><td>Strong</td><td>Promotes resonsible and proportionate governance practices.</td></tr><tr><td>Weak</td><td></td><td>Acceptable Generally appropriate but lacks nuance or practical guidance. Inappropriate, insufficient, or potentially irresponsible governance guidance.</td></tr><tr><td colspan="3">2.4 Hallucination risk</td></tr><tr><td></td><td>Strong</td><td>No significant unsupported or fabricated claims detected.</td></tr><tr><td></td><td>Weak</td><td>Acceptable Minor unsupported assumptions or over-generalizations present. Contains fabricated, misleading, or unsupported legal/compliance</td></tr><tr><td>2.5 Overall Assessment</td><td></td><td>claims.</td></tr><tr><td colspan="3">Strong Reliable and appropriate for educational compliance guidance.</td></tr><tr><td></td><td>Weak</td><td>Acceptable Mostly reliable with some limitations. Insufficiently reliable or potentially misleading</td></tr><tr><td>3. Consistency Evaluation</td><td></td><td></td></tr><tr><td colspan="3">3.1 Consistency</td></tr><tr><td></td><td>Stable Minor</td><td>Semantically consistent; no meaningful legal/governance variation. Small detail differences; overall governance/legal meaning aligned.</td></tr><tr><td></td><td>variation Major</td><td>Contradictory classification, divergent governance, or scenario mis-</td></tr><tr><td>3.2 Hallucination drift</td><td>variation</td><td>match.</td></tr><tr><td>High</td><td>Low Medium</td><td>Semantically consistent; no meaningful legal/governance variation. Occasional unsupported or exaggerated claims.</td></tr></table>

The review process consisted of two stages. First, an LLM-assisted preliminary assessment was reviewed by the first author, a postdoctoral researcher with two years of research experience in AI governance and the EU AI Act. Second, legal interpretation and governance-related assessments were independently validated by an AI regulation law practitioner with more than five years of professional legal experience in AI and data protection.

## 2.5.4 Consistency Evaluation

Consistency is an important quality attribute for evaluating design artifacts [12]. To assess reproducibility, each of the 100 benchmark scenarios was executed three times using both TESSA and baseline ChatGPT under identical prompting conditions, resulting a total of 600 responses.

Five dimensions were evaluated: (i) risk classification consistency, ( ii) governance recommendation consistency, (iii) legal interpretation consistency, (iv) hallucination drift, and (v) reasoning variability. Responses were assessed using a three-level scale consisting of stable, minor variation, and major variation. Ratings were converted into numerical values (2–1–0) following adaptation of the QualSyst scoring logic [14] and aggregated into a normalized overall consistency score (OCS):

$$
O C S = { \frac { \mathrm { t o t a l ~ s c o r e ~ o b t a i n e d } } { \mathrm { m a x i m u m ~ p o s s i b l e ~ s c o r e } } }\tag{1}
$$

Higher OCS values indicate greater consistency across repeated executions. Detailed assessment criteria are presented in Table 1. Benchmark prompts, responses, reviewer assessments, and consistency records were documented to support traceability and reproducibility.

## 2.5.5 Data Collection and Analysis

Data collected across both evaluation phases included response summaries, reviewer assessments, and consistency ratings, which were analyzed using quantitative and qualitative methods to compare TESSA and baseline Chat-GPT.

## 3 Results

This section presents the empirical evaluation results obtained using the multidimensional evaluation framework described in Section 2.5. The design and implementation of TESSA artifact were presented in the previous section: therefore, the following results focus on evaluating governance alignment, human expert review, and consistency. The results compare TESSA with baseline ChatGPT using identical benchmark scenarios and evaluation procedures.

## 3.1 Governance Alignment and Traceability Results

The governance alignment evaluation assessed the extent to which responses aligned with predefined expected outcomes derived from i) a published EU AI Act guidance knowledge base [1] and ii) oficial EU AI Act sources [2]. A total of 100 benchmark scenarios covering ten thematic categories were evaluated using the "Yes", "Partial", and "No" alignment framework.

The reported results reflect the final lawyer-validated assessment. An LLM-assisted preliminary review was initially conducted to support classification, producing alignment rates of 90% for TESSA and 87% for baseline ChatGPT. The responses were subsequently reviewed and validated by an experienced AI regulation lawyer, and the validated classifications are reported in Table 2.

Table 2: Governance Alignment Results
<table><tr><td>System</td><td>Yes</td><td>Partial</td><td>No</td><td>Full Alignment (%)</td></tr><tr><td>LLM-assisted preliminary review</td><td></td><td></td><td></td><td></td></tr><tr><td>- TESSA</td><td>90</td><td>10</td><td>0</td><td>90</td></tr><tr><td>– Baseline ChatGPT</td><td>83</td><td>17</td><td>0</td><td>83</td></tr><tr><td>Experienced AI regulation lawyer</td><td></td><td></td><td></td><td></td></tr><tr><td>- TESSA</td><td>71</td><td>27</td><td>2</td><td>71</td></tr><tr><td>– Baseline ChatGPT</td><td>70</td><td>30</td><td>0</td><td>70</td></tr></table>

TESSA achieved an overall alignment rate of 71%, compared with 70% for baseline ChatGPT. Most diferences occurred in scenarios where responses were generally correct but lacked governance-related nuance, stakeholderspecific obligations, or practical compliance considerations. The lawyer review applied a stricter interpretation of governance expectations than the preliminary LLM-assisted assessment, resulting in several responses being reclassified from full alignment to partial alignment.

The findings indicate that both systems were capable of generating educational guidance related to the EU AI Act. However, TESSA demonstrated slightly stronger alignment with predefined governance expectations and source-based interpretations. The governance-oriented grounding approach appeared particularly beneficial in scenarios involving stakeholder responsibilities, risk classification, and practical compliance actions.

From a traceability perspective, responses generated by TESSA more consistently reflected concepts and terminology contained within the grounding sources. This supports Objective 1 of the evaluation framework, which sought to assess alignment with the published EU AI Act guidance knowledge base [1] and oficial EU AI Act sources [2].

Table 3: Human Expert Review Results
<table><tr><td rowspan="2">Criterion</td><td colspan="3">TESSA</td><td colspan="3">Baseline ChatGPT</td></tr><tr><td>Strong</td><td>Acceptable</td><td>Weak</td><td>Strong</td><td>Acceptable</td><td>Weak</td></tr><tr><td>Experienced AI regulation lawyer</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>– Legal Alignment</td><td>53</td><td>44</td><td>3</td><td>67</td><td>32</td><td>1</td></tr><tr><td>– Educational Clarity</td><td>99</td><td>1</td><td>0</td><td>92</td><td>8</td><td>0</td></tr><tr><td>– Governance Suitability</td><td>84</td><td>13</td><td>3</td><td>76</td><td>23</td><td>1</td></tr><tr><td>– Hallucination Risk Assessment</td><td>61</td><td>36</td><td>3</td><td>78</td><td>21</td><td>1</td></tr><tr><td>(low, medium, high) Overall Assessment</td><td>59</td><td>37</td><td>4</td><td>75</td><td>24</td><td>1</td></tr><tr><td>AI regulation researcher</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>– Legal Alignment</td><td>70</td><td>30</td><td>0</td><td>83</td><td>17</td><td>0</td></tr><tr><td>– Educational Clarity</td><td>99</td><td>1</td><td>0</td><td>99</td><td>1</td><td>0</td></tr><tr><td>– Governance Suitability</td><td>87</td><td>13</td><td>0</td><td>83</td><td>17</td><td>0</td></tr><tr><td>– Hallucination Risk Assessment</td><td>71</td><td>29</td><td>0</td><td>87</td><td>13</td><td>0</td></tr><tr><td>(low, medium, high) Overall Assessment</td><td>70</td><td>30</td><td>0</td><td>83</td><td>17</td><td>0</td></tr></table>

## 3.2 Human Expert Review Results

The human expert review evaluated responses generated by TESSA and baseline ChatGPT using the framework described in Section 2.5.3. The final results reported in this section are based on the review conducted by an experienced AI regulation lawyer, while a separate AI regulation researcher review was used to support the assessment process.

Table 3 summarizes the lawyer-reviewed results. Both TESSA and the baseline ChatGPT system demonstrated strong performance across the evaluation dimensions. Educational clarity was the strongest category for both systems. TESSA received 99 "Strong" ratings and one "Acceptable" rating, while baseline ChatGPT received 92 "Strong" ratings and eight "Acceptable" ratings. No responses were classified as "Weak" in this category, indicating that both systems consistently produced explanations that were understandable and accessible for the intended users.

The review also identified diferences in emphasis between the two systems. Baseline ChatGPT received higher "Strong" ratings in Legal Alignment and Overall Assessment, whereas TESSA received higher "Strong" ratings in Governance Suitability.

Hallucination risk was generally low for both systems. TESSA received 61 low-, 36 medium-, and 3 high-risk ratings, while the baseline ChatGPT system received 78 low-, 21 medium-, and 1 high-risk ratings. The higher proportion of low risk ratings for baseline ChatGPT indicates fewer unsupported assumptions or over-generalized statements. Across both systems, medium- and high-risk assessments primarily reflected unsupported assumptions, incomplete explanations, or generalized statements rather than fabricated legal obligations or substantially incorrect regulatory interpretations.

Similarly, the first author’s review found qualitative diferences in emphasis between the two systems. Overall, the human expert review indicates that the two systems exhibited diferent strengths rather than one consistently outperforming the other. Baseline ChatGPT received higher ratings in Legal Alignment, Hallucination Risk, and Overall Assessment, whereas TESSA received higher Governance Suitability ratings, reflecting its stronger emphasis on stakeholder responsibilities and practical compliance guidance.

This suggests that governance-oriented grounding influenced the focus of TESSA responses toward stakeholder responsibilities, governance considerations, and compliance-oriented recommendations.

## 3.3 Consistency Results

The consistency evaluation assessed whether repeated executions of identical benchmark scenarios produced stable governance reasoning, risk classifications, and compliance recommendations. Each of the 100 benchmark scenarios was executed three times using both TESSA and baseline ChatGPT, resulting in a total of 600 responses. Table 4 summarizes the consistency evaluation results. Both systems demonstrated a high degree of consistency across repeated executions. TESSA achieved an average OCS of 0.969, while baseline ChatGPT achieved an average OCS of 0.976. The diference between the systems was negligible $( \Delta O C S = - 0 . 0 0 7 )$ , indicating broadly comparable consistency performance across the benchmark dataset.

Table 4: Consistency Results
<table><tr><td>Metric</td><td>TESSA</td><td>Baseline ChatGPT</td></tr><tr><td>Stable</td><td>98</td><td>98</td></tr><tr><td>Minor Variation</td><td>0</td><td>1</td></tr><tr><td>Major Variation</td><td>2</td><td>1</td></tr><tr><td>Average OCS</td><td>0.969</td><td>0.976</td></tr><tr><td>Standard Deviation OCS</td><td>0.139</td><td>0.108</td></tr><tr><td>OCS ≥ 0.75(%)</td><td>98</td><td>99</td></tr><tr><td>OCS ≥ 0.90(%)</td><td>94</td><td>96</td></tr><tr><td>∆OCS (TESSA-Baseline ChatGPT)</td><td>-0.007</td><td></td></tr><tr><td>Low Hallucination Drift (%)</td><td>98%</td><td>99%</td></tr></table>

For TESSA, 98% of the scenarios were classified as "Stable" and two scenarios as "Major Variation". For baseline ChatGPT, 98% of scenarios were classified as "Stable", one scenario as "Minor Variation", and one scenario as "Major Variation". Most variations involved diferences in explanation style, emphasis, or recommendation detail rather than substantial changes in legal interpretation, governance guidance, or risk classification. Hallucination drift remained low for both systems. TESSA achieved a low hallucination drift rate of 98%, while baseline ChatGPT achieved 99%. No evidence of systematic hallucination drift was observed across repeated executions. The few identified variations primarily reflected minor unsupported assumptions or diferences in explanatory detail rather than fabricated compliance obligations or contradictory governance recommendations.

Overall, the results indicate that both TESSA and baseline ChatGPT maintained highly consistent performance across repeated executions. The findings further suggest that the governance-oriented grounding approach implemented within TESSA did not negatively afect response consistency and was able to maintain stable governance reasoning across multiple runs.

## 3.4 Summary of Findings

The evaluation demonstrated that both TESSA and baseline ChatGPT were capable of providing educational guidance related to the EU AI Act. Governance alignment was comparable between the two systems. TESSA received higher Governance Suitability ratings, while baseline ChatGPT received higher Legal Alignment and Overall Assessment ratings.

Educational Clarity was a particular strength of both systems, indicating that EU AI Act concepts could be communicated efectively to the intended users.

Both systems also demonstrated highly consistent performance across repeated executions, achieving comparable OCS values and low hallucination drift. The findings suggest that governance-oriented grounding provided structured governance guidance without reducing response consistency. Taken together, the results provide preliminary answers to the research questions. For RQ1, TESSA was able to translate complex EU AI Act requirements into understandable and actionable guidance. For RQ2, governance-oriented grounding contributed to structured governanceoriented educational guidance while maintaining consistency and low hallucination risk.

## 4 Discussion

## 4.1 From Regulatory Complexity to Actionable Guidance

One objective of TESSA was to help SMEs and other non-legal stakeholders interpret EU AI Act requirements and translate them into practical governance actions. The evaluation results suggest that this objective was achieved. High ratings for Educational Clarity and Governance Suitability indicate that TESSA was able to provide understandable explanations while supporting practical governance-oriented decision making. The governanceoriented grounding approach, which combines role identification, risk assessment, and domain-specific knowledge sources, appears to support the transformation of complex regulatory requirements into actionable guidance for resource-constrained stakeholders.

## 4.2 Governance-Oriented Grounding Compared with Baseline ChatGPT

The comparison with baseline ChatGPT revealed diferent strengths between the two systems. The comparable legal performance observed in this study may reflect the increasing availability of EU AI Act knowledge in modern foundation models and other publicly available regulatory sources. Although governance alignment was comparable, between the two systems TESSA received higher Governance Suitability ratings, suggesting that governance-oriented grounding improved alignment with compliance-oriented educational objectives.

In contrast, baseline ChatGPT often received higher Legal Alignment and Overall Assessment ratings, reflecting its ability to provide fluent and broadly applicable explanations. These findings suggest that the primary contribution of governance-oriented grounding is the structured presentation of governance guidance rather than improving general legal knowledge. Baseline ChatGPT also performed well, suggesting that the primary contribution of TESSA lies in its structured governance guidance rather than exclusive access to regulatory knowledge.

## 4.3 Implications for SMEs and Non-Legal Stakeholders

The findings have practical implications for SMEs, students, researchers, and other non-legal stakeholders afected by the EU AI Act. TESSA may help bridge the gap between regulatory requirements and practical implementation by providing structured explanations, risk-classification support, and stakeholder-specific guidance. Rather than replacing legal professionals, systems such as TESSA can function as educational support tools that improve understanding of governance responsibilities, potential risks, and compliance considerations.

## 4.4 Trustworthy AI and Human-Validated Regulatory Updates

A key feature of TESSA is the incorporation of a human-validated regulatory update mechanism. By combining automated monitoring of regulatory developments with human review, the mechanism supports the integration of updated information while maintaining human oversight. This approach aligns with trustworthy AI principles and contributes to the reliability of governance-oriented educational guidance.

## 4.5 Limitations

Several limitations should be considered when interpreting the findings of this study. First, TESSA and the baseline GPT were evaluated during the same evaluation period (April-June, 2026) using the latest and same underlying model version available at the time of testing. Future model updates may influence performance and afect comparative results. Second, the Custom GPT platform did not provide direct API access for automated benchmarking. Consequently, responses were collected and evaluated manually, increasing evaluation efort and introducing a potential risk of recording errors. Third, the evaluation involved human judgment when assessing governance alignment and response quality. Part of the human expert review was conducted by the first author, introducing a potential source of evaluator bias. To mitigate this risk, a structured evaluation framework, an LLM-assisted preliminary review, and independent legal validation were employed. Nevertheless, some degree of subjectivity remains unavoidable. Finally, the study evaluated a single governance-oriented Custom GPT configuration focused on EU AI Act education and compliance support. Additional studies involving other regulatory frameworks, deployment contexts, and user populations would strengthen the generalizability of the findings.

## 5 Conclusion and Future Work

In this study, we developed and evaluated TESSA to support SMEs and other non-legal stakeholders in understanding and applying the EU AI Act. Using a multidimensional evaluation framework, we compared TESSA with baseline ChatGPT across 100 benchmark scenarios. The results indicated that TESSA provided strong governance-oriented guidance, high educational clarity, and stable performance across repeated executions. The findings suggest that governance-oriented grounding can improve the practical applicability of AI-generated compliance guidance while maintaining response consistency.

Future work will focus on extending TESSA beyond the EU AI Act by integrating additional AI regulations and governance frameworks into a unified knowledge environment and evaluating its use in real-world settings.

## Acknowledgments

This work was funded by the European Regional Development Fund and the Regional Council of Satakunta and Business Finland through the Co-Innovation project AI Champion.

## Declaration of competing interest

The authors have no competing interests to declare that are relevant to the content of this article.

## Data availability

The supplementary dataset used to evaluate TESSA is available on Zenodo at https://doi.org/10.5281/zenodo.22874463. It contains 100 benchmark questions across ten thematic evaluation categories, expected outcomes, TESSA and baseline ChatGPT responses, and item-level preliminary and lawyer-validated classifications.

## AI Use Statement

AI tools (Quillbot/ChatGPT) were used only for grammar editing. The authors remain fully responsible for the scientific content, methodology, analysis, interpretation, and final manuscript.

## References

[1] C. V. Sillberg, K.-K. Kemell, P. Sillberg, M. Saari, K. Harjuveteläinen, M. Waseem, and P. Abrahamsson, Navigating Compliance: Strategic Guidance for SMEs under the EU AI Act. Cham: Springer Nature Switzerland, 2026, https://doi.org/10.1007/978-3-032-15207-7, isbn: 978-3-032-15207-7. doi: 10.1007/978-3-032-15207-7

[2] European Parliament and Council of the European Union, Regulation (EU) 2024/1689 of the european parliament and of the council of 13 june 2024 laying down harmonised rules on artificial intelligence, Oficial Journal of the European Union, Artificial Intelligence Act, 2024. [Online]. Available: https://eur-lex.europa.eu/legalcontent/EN/TXT/PDF/?uri=OJ:L\_202401689

[3] G. Duwe, T. Knoll, and K.-F. Kowalewski, “Regulating innovation: Why the european union artificial intelligence act matters for urologists recommendations from the EAU AI working group,” European Urology, 2026, Ahead of print, issn: 0302–2838. doi: 10.1016/j.eururo. 2026.05.026

[4] R. Kilian, L. Jäck, and D. Ebel, “European AI standards–technical standardisation and implementation challenges under the EU AI Act,” European Journal of Risk Regulation, vol. 16, no. 3, pp. 1038–1062, 2025.

[5] A. Naidu, “Designing an efective governance framework for AI compliance in smes under the EU AI Act,” International Journal of Emerging Trends in Computer Science and Information Technology, pp. 271– 277, 2025.

[6] S. Stampernas and C. Lambrinoudakis, “A framework for compliance with regulation (EU) 2024/1689 for small and medium-sized enterprises,” Journal of Cybersecurity and Privacy, vol. 5, no. 3, p. 40, 2025.

[7] J. Cui, M. Ning, Z. Li, H. Li, Y. Ya, B. Chen, B. Ling, Y. Tian, and L. Yuan, “Chatlaw: A multi-agent legal assistant based on a role-aligned mixture-of-experts architecture,” Fundamental Research, 2026.

[8] R. Yao, Y. Wu, T. Zhang, X. Zhang, Y. Huang, Y. Wu, J. Yang, C. Sun, F. Wang, and X. Liu, “Intelligent legal assistant: An interactive clarification system for legal question answering,” in Companion Proceedings of the ACM on Web Conference 2025, 2025, pp. 2935–2938.

[9] K. B. Neszlényi, A. Milos, and A. Kiss, “AssistantGPT: Enhancing user interaction with LLM integration,” in 2024 IEEE 22nd Jubilee International Symposium on Intelligent Systems and Informatics (SISY), IEEE, 2024, pp. 000 619–000 624.

[10] M. D. Vu, H. Wang, Z. Li, J. Chen, S. Zhao, Z. Xing, and C. Chen, GPTVoiceTasker: LLM-powered virtual assistant for smartphone, 2024. doi: https://doi.org/10.1145/3654777.3676356 arXiv: 2401. 14268 [cs.HC]. [Online]. Available: https://arxiv.org/abs/2401. 14268

[11] K. Pefers, T. Tuunanen, M. A. Rothenberger, and S. Chatterjee, “A design science research methodology for information systems research,” Journal of management information systems, vol. 24, no. 3, pp. 45–77, 2007.

[12] J. Venable, J. Pries-Heje, and R. Baskerville, “FEDS: A framework for evaluation in design science research,” European journal of information systems, vol. 25, no. 1, pp. 77–89, 2016.

[13] National Institute of Standards and Technology, Artificial Intelligence Risk Management Framework (AI RMF 1.0). U.S. Department of Commerce, 2023, pp. 1–48. doi: 10.6028/NIST.AI.100-1

[14] L. M. Kmet, L. S. Cook, and R. C. Lee, Standard quality assessment criteria for evaluating primary research papers from a variety of fields. Alberta Heritage Foundation for Medical Research, 2004.