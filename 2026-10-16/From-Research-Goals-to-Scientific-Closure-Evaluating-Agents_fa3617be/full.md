# From Research Goals to Scientific Closure: Evaluating Agents Through Evidence-Supported Physics Research

Yijia Mao<sup>1,2\*</sup> Song Dai<sup>1,3\*</sup> Tao Chen<sup>2</sup> Xiangyu Lu<sup>2</sup> Linqian Zeng<sup>2</sup> Chenxi Hu<sup>2</sup> Sitong Zhou<sup>2</sup> Dongyang He<sup>2</sup> Xiaodan Mao<sup>2</sup> Linxuan Zhang<sup>2</sup> Tong Zhu Yafu Li<sup>1</sup> Fangchen Yu<sup>1‡</sup> Peng Ye<sup>1‡</sup> Feng He<sup>2</sup> Yu Cheng<sup>4‡</sup> Yun Luo<sup>1†,‡</sup>

<sup>1</sup> Shanghai AI Laboratory <sup>2</sup> Shanghai Jiao Tong University <sup>3</sup> The HongKong University of Science and Technology(Guangzhou) 4 Nanyang Technological University <sup>\*</sup>Equal Contribution, <sup>†</sup>Project Lead, <sup>‡</sup>Corresponding Author

## ABSTRACT

Can AI agents organize scientific investigations that lead to well-supported conclusions? To evaluate this capability, we introduce PhysClosureBench, a benchmark of 50 expert-verified, goal-specified research tasks spanning strong-field, attosecond, and neighboring physics areas. Each task specifies the research objective and essential scientific constraints, while leaving the agent to organize the investigation and determine how to generate and validate the evidence needed to answer the question. A 20-criterion Atomic Score measures local scientific accomplishment, while Research Closure (RC0–RC4) assesses how fully the investigation resolves the specified research question based on whether its indispensable evidence requirements and applicable scope conditions are satisfied. We evaluate seven frontier LLMs, where Claude-Opus-5 achieves the highest mean Atomic Score (6.29/10) but reaches RC4 in only 32.7% of attempts. Across the benchmark, 36.2% of attempts scoring at least 8/10 still fall short of RC4, while all seven models reach RC4 on substantially fewer tasks in all three runs than in at least one. Case-level analysis further illustrates how misinterpretation of generated results and unresolved physical applicability limits can lead to incomplete closure. Together, these results reveal a persistent gap between AI agents’ local scientific accomplishments and their ability to achieve complete research closure.

![](images/7c37e188d1bc32b5a684157fdd48b0ccdbaf725e67275a3ea85ffe28f5b765a5.jpg)  
Figure 1: From a scientific goal to scientific closure. In PhysClosureBench, agents are required to achieve the provided scientific objective within the specified task setting and a resource-bounded computational environment.

## 1 Introduction

“Science is built up offacts, as a house is built ofstones; but an accumulation offacts is no more a science than a heap ofstones is a house.”

— Henri Poincaré.

As language-model agents become more capable of reasoning, writing code, and processing long contexts, they are increasingly applied to scientific research, assisting with literature synthesis, scientific programming, and the reproduction of prior work (Lu et al., 2024; Starace et al., 2025; Wijk et al., 2025; Zheng et al., 2025; Zhang et al., 2024). While these capabilities make them promising scientific assistants, useful assistance alone does not imply the ability to independently organize an investigation toward a specified scientific goal (Zhang et al., 2025). Scientific research requires more than producing plausible results. It requires organizing an investigation that connects a research question to a well-supported scientific conclusion. Doing so involves deciding what evidence to generate and how to interpret and validate it (Bogen & Woodward, 1988; Mayo, 1996). We use scientific closure to describe whether such an investigation has adequately resolved the research question, which raises a central question:

Can agents independently organize an investigation that achieves scientific closure on a specified research question?

Existing benchmarks evaluate important components of scientific capability, including final answers (Rein et al., 2024; Center for AI Safety et al., 2026), executable code and ML-engineering artifacts (Tian et al., 2024; Chan et al., 2025), and research replication or open-ended research-engineering tasks (Starace et al., 2025; Wijk et al., 2025). However, they rarely assess whether an investigation as a whole establishes a well-supported answer to the specified research question. An agent may therefore obtain a correct result or reproduce a reported finding without completing the broader investigation, a distinction that existing task-level metrics seldom capture.

To address this gap, we formulate goal-specified, agent-directed research tasks in physics, where scientific conclusions are tightly tied to observable evidence, competing physical mechanisms, and well-defined validity regimes. Each task specifies a scientific objective and a bounded setting, while leaving the agent to organize the investigation and determine what evidence is needed to solve the problem. We evaluate agents at two complementary levels that distinguish local scientific accomplishment from overall Research Closure. The Atomic Score measures local accomplishment on 20 predefined, task-specific criteria, each worth 0.5 points, that cover formulation, output, identification, mechanism, and validation/scope, while the Research Closure (RC0–RC4) evaluates whether the investigation has established the evidence, interpretation, and scope needed for a complete answer.

We instantiate this task formulation and evaluation protocol in PhysClosureBench, a benchmark of 50 expert-verified tasks centered on attosecond physics and neighboring regimes, including solid-state strong-field physics, relativistic quantum dynamics, and quantum optics (Krausz & Ivanov, 2009; Kruchinin et al., 2018; Frisk Kockum et al., 2019; Cruz-Rodriguez et al., 2024). Evaluating seven frontier models on PhysClosureBench, we characterize their scientific accomplishment and Research Closure. We further examine whether complete closure is achieved across independent runs of the same task. Claude-Opus-5 achieves the highest mean Atomic Score (6.29/10) and RC4 rate (32.7%), whereas Gemini-3.1-Pro obtains the lowest (2.85/10 and 0.7%, respectively). Even for the strongest model, complete closure remains uncommon. Achieving complete closure reliably can be even harder. For Claude-Opus-5, only 18.0% of tasks reach RC4 in all three independent runs. These results reveal a substantial gap between local accomplishment and complete closure, and a further gap between demonstrating the ability to achieve closure and doing so reliably across independent runs.

Our contributions are threefold:

1. We introduce PhysClosureBench, a benchmark of goal-specified physics research tasks that evaluates whether agents can carry investigations through to well-supported answers to specified research questions.

2. We develop an evaluation framework with two complementary perspectives. An Atomic Score measures local scientific accomplishment across five functional dimensions, while Research Closure assesses how fully an investigation satisfies its task-specific scientific requirements.

Table 1: Comparison of representative scientific AI benchmarks. A checkmark indicates direct assessment of a dimension, while “Partial” indicates limited or indirect assessment and a cross indicates no explicit assessment. Scientific closure concerns whether an investigation establishes a well-supported answer to its research question, including the required validation and scope conditions.
<table><tr><td>Benchmark</td><td>Task formulation</td><td>Execution / artifacts</td><td>Research decisions</td><td>Evidence integration</td><td>Scientific closure</td></tr><tr><td>GPQA / HLE</td><td>Expert-level QA</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>CritPt</td><td>Physics reasoning</td><td>√</td><td>X</td><td>X</td><td>X</td></tr><tr><td>SciCode</td><td>Scientific coding</td><td>√</td><td>×</td><td>×</td><td>X</td></tr><tr><td>PaperBench</td><td>Research reproduction</td><td>√</td><td>Partial</td><td>Partial</td><td>X</td></tr><tr><td>RE-Bench</td><td>Research-engineering tasks</td><td>√</td><td>Partial</td><td>Partial</td><td>X</td></tr><tr><td>FrontierScience-R</td><td>Scientific research subproblems</td><td>X</td><td>√</td><td>Partial</td><td>Partial</td></tr><tr><td>PRL-Bench</td><td>Physics research subtasks</td><td>√</td><td>√</td><td>√</td><td>Partial</td></tr><tr><td>PhysClosureBench</td><td>Goal-specified physics research</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

3. We evaluate seven frontier models and find that substantial local accomplishment does not ensure complete closure, nor does achieving closure in one run guarantee the same outcome in other independent runs.

## 2 Related Work

Recent scientific AI benchmarks span expert-level reasoning and increasingly complex research activities. GPQA and Humanity’s Last Exam assess expert-level question answering (Rein et al., 2024; Center for AI Safety et al., 2026). Moving toward research-level scientific problems, CritPt evaluates composite frontier-physics challenges and finer-grained checkpoints with machine-verifiable answers, including code-enabled evaluation (Zhu et al., 2026). FrontierScience’s Research track assesses expert-authored, PhD-level scientific subproblems using granular rubrics (Wang et al., 2026). Beyond scientific reasoning, agents such as The AI Scientist and The AI Scientist-v2 explore broader research workflows involving experiment design, execution, and iterative analysis (Lu et al., 2024; Yamada et al., 2025). Related benchmarks evaluate different aspects of executable research. SciCode focuses on scientific code generation (Tian et al., 2024); ScienceAgentBench assesses generated programs and execution results in data-driven scientific tasks (Chen et al., 2025); MLE-Bench and RE-Bench evaluate machine-learning and research-engineering tasks, while PaperBench examines research reproduction (Chan et al., 2025; Wijk et al., 2025; Starace et al., 2025). More closely related to physics research, PRL-Bench evaluates exploration-oriented, long-horizon theoretical and computational tasks with code execution and rubric-based assessment (Miao et al., 2026). Table 1 summarizes representative differences in their evaluation scope and design.

PhysClosureBench instead makes scientific closure at the level of a goal-specified, agent-directed investigation an explicit evaluation target. It distinguishes local scientific accomplishment from whether the evidence, interpretation, and validation established by the agent are sufficient to resolve the original research question, and examines how consistently complete closure is achieved across independent attempts.

## 3 Benchmark and Evaluation Design

## 3.1 Task Design and Agent-Directed Setting

PhysClosureBench contains 50 goal-specified physics research tasks derived from published studies in authoritative journals in physics such as Physical Review Letters and Physical Review A, spanning strong-field atomic and molecular physics, attosecond science, solid-state and condensed-matter physics, quantum optics, and neighboring quantum light–matter regimes.

Each task specifies the scientific objective and the information necessary to make the problem wellposed, including its physical setting, relevant parameters, constraints, and computational resources. The benchmark fixes what scientific question must be addressed, while leaving the overall investigation strategy open within the task-specific scientific constraints. No complete solution workflow or fixed diagnostic and validation sequence is prescribed. Agents instead determine how to carry out the investigation, including the numerical implementation, observables, comparisons, controls, and validation procedures needed to answer the question.

![](images/81da49f96356236904567175d209f9829b2ea7effb9afd4129509c1bed15fd11.jpg)  
Figure 2: Two complementary views of scientific performance. Atomic Score (0–10) measures local accomplishment across F/O/I/M/V. Research Closure (RC0–RC4) assesses how fully the investigation resolves the research question. A high Atomic Score does not guarantee complete closure.

Table 2: Atomic scoring dimensions.  
Table 3: Research Closure levels.
<table><tr><td>Dimension</td><td>Scientific question</td><td>Level Interpretation</td><td></td></tr><tr><td>F Formulation</td><td>Physical and numerical setup correct?</td><td>RC4</td><td>Complete closure. Required evidence and scope conditions satisfied.</td></tr><tr><td>O Output</td><td>Required observables generated correctly?</td><td>RC3</td><td>Main conclusion supported. Material validation or scope gaps remain.</td></tr><tr><td>I Identification</td><td>Relevant features identified correctly?</td><td>RC2</td><td>Partial investigation. Core evidence present; central mechanism or other key requirement unresolved.</td></tr><tr><td>M Mechanism</td><td>Mechanism supported by evidence?</td><td>RC1</td><td>Limited investigation. Essential primary output or branch missing.</td></tr><tr><td>V Validation/Scope</td><td>Applicable controls and scope justified?</td><td>RC0</td><td>No meaningful closure. Central scientific object invalid or absent.</td></tr></table>

Tasks are executed in isolated, network-disabled computational environments under task-specific CPU and wall-time limits. Within these resource limits, agents decide how computational effort is allocated across planning, exploration, analysis, and validation.

## 3.2 Task Reliability and Verification

We use a multi-stage verification process to ensure that the benchmark tasks are scientifically grounded, executable, and appropriately resourced. First, as mentioned before, every task is derived from a published physics study, providing a traceable basis for the target phenomenon, physical regime, and scientific conclusion. Second, a domain expert designed and supervised the semiautomated pipeline used for task construction and rubric design. Review by other experts checked that the intended scientific questions and evidential requirements were preserved.

Third, the source results underlying every task were manually reproduced before inclusion in the benchmark. We verified that the relevant numerical results, physical trends, and scientific conclusions could be recovered under the specified conditions. These reproductions were also used to calibrate the resource limits. CPU allocations and wall-time limits were determined from the observed reproduction costs, with additional allowance for planning, exploratory calculations, method selection, and unsuccessful intermediate attempts. This process ensures that tasks are feasible under the provided resources without constraining agents to the reference reproduction pathway.

## 3.3 Evaluation Framework

Scientific research performance can be assessed at two complementary levels. One is the extent of scientific work accomplished during the research process, and the other is whether the investigation as a whole reaches a well-supported answer to the research question. Substantial progress on individual research components does not necessarily imply complete closure, and closure may be achieved through different valid research pathways as long as the task-specific scientific requirements are satisfied. Figure 2 summarizes these two complementary perspectives.

1. Atomic scientific accomplishment. The Atomic Score measures local accomplishment across 20 task-specific, predefined half-point criteria, organized into five functional dimensions, including Formulation (F), Output (O), Identification (I), Mechanism (M), and Validation/Scope (V), as summarized in Table 2. The resulting 0–10 score captures the extent of relevant scientific work completed, without requiring that the investigation itself be complete.

2. Research Closure. Research Closure (RC0–RC4) evaluates the overall completion of the investigation relative to the research question. The closure levels, summarized in Table 3, are determined by task-specific scientific requirements rather than by a fixed workflow or threshold on the Atomic Score. The distinction between adjacent RC levels depends on the role of an unresolved requirement in supporting the main scientific conclusion, including its task-specific prerequisites, rather than on its assignment to a particular functional dimension.

Atomic Scores provide fine-grained information about where scientific progress is made, whereas Research Closure assesses whether the investigation meets the requirements for a complete scientific answer. Together, they support both diagnostic analysis of research behavior and consistent comparison across models under a common evaluation protocol.

## 4 Experimental Setup

## 4.1 Models and Evaluation Procedure

We evaluate seven frontier models—Claude-Opus-5 (Anthropic, 2026), GPT-5.6-Sol (OpenAI, 2026), Kimi-K3 (Kimi Team, 2026), DeepSeek-V4.1-Flash (DeepSeek-AI, 2026), Qwen-3.8-Max (Alibaba Group, 2026), GLM-5.3-Flash (Z.ai, 2026), and Gemini-3.1-Pro (Google DeepMind, 2026)—on all 50 tasks, with three independent runs for each model–task pair, yielding 1,050 final scored attempts. The evaluated configurations use xhigh reasoning effort for Claude-Opus-5, GPT-5.6-Sol, and Qwen-3.8-Max, and high reasoning effort for Kimi-K3, DeepSeek-V4.1-Flash, GLM-5.3-Flash, and Gemini-3.1-Pro.

Each attempt runs in a separate OpenCode (Anomaly, 2026) session with its own workspace, initialized with the task specification, resource limits, and common instructions on tool use, research conduct, and submission requirements. Agents can read and modify files in their workspaces and execute scientific code through a custom Bash interface in an isolated, network-disabled container. When a run ends or reaches its wall-time limit, the controller collects the report and supporting artifacts available at that point for grading, including assessable partial results. Appendix A details the environment and protocol.

Incomplete work produced by the agent is included in the evaluation. Unsuccessful calculations, abandoned approaches, and incomplete analyses are scored according to the evidence produced by the agent. Runs that reach the time limit are graded on the scientific work available at that point, with no additional penalty for timeout itself. When an attempt produces no assessable scientific result, it receives zero atomic credit and RC0. Executions invalidated by infrastructure failures are excluded from scoring and rerun to obtain a valid measurement.

All assessable submissions are evaluated using a staged automated protocol with GPT-5.6-Sol(high). Each submission first receives an independent first-pass grade using the fixed evaluation prompt and task-specific rubric, followed by a targeted review with access to the initial assessment. Conditional adjudication is triggered only when a residual material disagreement could affect the Atomic Score, Research Closure, or attempt validity. Evaluation is restricted to evidence produced by the agent.

A Scientific accomplishment  
![](images/15950a09d6c8310ce951697cddd5a9c3df1f40b23e30dfd51d81c543aaf137ad.jpg)

B Research Closure distributions  
![](images/e619612a4967cb8ab9135892785bbc225d74e1cc19d813e5ec361f6016f736cd.jpg)  
Figure 3: Scientific accomplishment and Research Closure across frontier models. (A) Mean Atomic Score and functional completion for Formulation (F), Output (O), Identification (I), and Mechanism (M). Atomic Scores are shown on the shared axis after multiplication by 10; functional completion is the mean attempt-level fraction of applicable criteria satisfied within each dimension. (B) Distribution of Research Closure levels across independent attempts. RC4 denotes complete closure, RC3 indicates that the main conclusion is supported, but material validation or scope gaps remain, and RC0–RC2 indicate different degrees of incomplete investigation.

Submitted artifacts and execution records may be inspected for verification, but the evaluator’s checks cannot substitute for analyses or interpretations missing from the submissions.

Five physics-trained experts reviewed 100 evaluation cards covering all 50 tasks, yielding 200 selected criterion-level judgments. Final human assessments agreed with automated grading in 81.8% of the selected atomic-criterion judgments, with 75.9% agreement on criteria explicitly linked to Research Closure. Appendix B describes the review protocol and the closure-relevant criteria in detail.

## 4.2 Analysis Measures

The individual research attempt is the basic unit of analysis. Unless otherwise specified, model-level results are aggregated over three independent runs of each of the 50 benchmark cases (150 attempts per model). We report the mean Atomic Score and the distribution of Research Closure levels, including the overall RC4 rate, as the primary measures of scientific accomplishment and closure.

For Formulation, Output, Identification, and Mechanism, functional completion is computed within each attempt as the fraction of satisfied criteria among those applicable to that attempt. Modellevel completion is the mean of the attempt-level completion rates, giving equal weight to each research attempt. Attempts with no assessable scientific accomplishment contribute zero completion. Validation/Scope is analyzed separately because its applicability depends more directly on the method used, evidence produced, and resulting conclusion of each attempt.

To characterize consistency of Research Closure across independent runs, we additionally report, for each model, the fraction of the 50 tasks that achieve RC4 in at least one of the three independent runs and the fraction that achieve RC4 in all three runs. Appendix C provides the detailed calculations for the analysis measures, including the treatment of N/A criteria and how results are aggregated.

A Atomic Score and closure Share of all investigations (%)
<table><tr><td colspan="6">0.0 23.3</td></tr><tr><td colspan="3"></td><td colspan="3">11.7</td></tr><tr><td>RC0 RCl</td><td>23.3%</td><td>11.6%</td><td>3.3%</td><td>1.0%</td><td>0.1%</td></tr><tr><td>Resch oasure RC2</td><td>3.3%</td><td>9.6%</td><td>5.8%</td><td>4.6%</td><td>0.5%</td></tr><tr><td>RC3</td><td>0.2%</td><td>1.9%</td><td>5.2%</td><td>7.6%</td><td>5.2%</td></tr><tr><td></td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.4%</td><td>2.1%</td></tr><tr><td>RC4</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.2%</td><td>13.9%</td></tr><tr><td></td><td>0-&lt;2</td><td>2-&lt;4</td><td>4.&lt;6 Atomic Score</td><td>6-&lt;8</td><td>8-10</td></tr></table>

B Unmet requirements at score ≥ 8  
![](images/147e1e946893dd1f4b0068d9146b0233ba792d61ed9d964616658710c01f9f01.jpg)  
Figure 4: Relationship between local scientific accomplishment and Research Closure. (A) Joint distribution of Atomic Score and Research Closure across all 1,050 independent attempts. Each cell reports the percentage of all attempts falling into the corresponding score bin and RC level. The highlighted region marks high-scoring attempts below RC4. (B) Unmet applicable criteria for Formulation (F), Output (O), Identification (I), and Mechanism (M) among the 229 attempts scoring at least 8/10, comparing RC4 (n = 146) with below RC4 (n = 83). Bars show 100% minus the mean attempt-level functional completion within each group. The inset reports the fraction of satisfied applicable Validation/Scope (V) criteria across the full benchmark, rather than within the high-scoring groups; N/A criteria are excluded.

## 5 Results and Discussion

We analyze scientific-agent performance in terms of local scientific accomplishment, Research Closure, the evidence patterns underlying closure, and consistency of Research Closure across independent runs.

## 5.1 Scientific Accomplishment and Research Closure

We first compare local scientific accomplishment with Research Closure across models, as shown in Figure 3. Claude-Opus-5 achieves the highest mean Atomic Score at 6.29/10, followed by GPT-5.6-Sol at 5.83/10 and Kimi-K3 at 5.17/10, while the remaining models score 4.50 or below. The functional profiles further reveal distinct patterns across research capabilities. Formulation has the highest completion rate among F/O/I/M for every model, suggesting that current systems are comparatively reliable at establishing an appropriate scientific setup. For Claude-Opus-5, completion reaches 85.6% in Formulation, compared with 60.1% in Output, 54.6% in Identification, and 52.8% in Mechanism. The relative advantage of stronger models is particularly visible in Output. Claude-Opus-5, GPT-5.6-Sol, and Kimi-K3 achieve 60.1%, 60.5%, and 49.7%, respectively, compared with 37.3% or less for the remaining models. In contrast, Identification and Mechanism remain challenging across the model set, with even the strongest model completing only about half of the applicable criteria in either dimension. Some submissions lack reliable upstream results needed for interpretation; others generate relevant evidence but fail to identify its key features or establish a supported physical explanation.

Research Closure broadly follows the model ranking by Atomic Score, but complete closure remains uncommon even among the leading models. Claude-Opus-5, GPT-5.6-Sol, and Kimi-K3 achieve RC4 in 32.7%, 22.7%, and 19.3% of attempts, respectively. Despite their stronger performance on individual scientific requirements, most of their investigations do not reach complete closure. The full RC distribution also reveals differences that RC4 rates alone cannot capture. DeepSeek-V4.1-Flash reaches RC4 in 10.7% of attempts, more frequently than the three lower-scoring models, yet 50.0% of its attempts remain at RC0. Thus, a comparatively higher complete-closure rate can coexist with a large proportion of investigations that fail to establish the essential scientific basis of the task.

Taken together, the two panels highlight a substantial gap between local scientific accomplishment and complete closure.

Case study: different evidence paths toward scientific closure  
![](images/2c01509b67f80838c63b30a74a32b8dd8b4a7872057b696f1200929865f98e61.jpg)  
Figure 5: Representative evidence and scientific closure. Case 30 compares representative runs from Kimi-K3, GPT-5.6-Sol, and Gemini-3.1-Pro, which reached RC4, RC3, and RC2, respectively. The shaded region illustrates a possible progression from numerical evidence to a supported conclusion; Gemini’s deviation represents the misinterpretation of the computed intensity trends. Paths are schematic, not execution times, Atomic Scores, or intermediate RC assignments. Inset plots schematically illustrate trends in the submitted results. Appendix E documents the research activities and evidence.

## 5.2 High Atomic Scores and Incomplete Research Closure

We next examine how progress on individual scientific requirements relates to and differs from the overall completion of an investigation (Figure 4).

Panel A shows that higher Atomic Scores generally correspond to higher Research Closure levels. This overall agreement is expected, as both measures assess how well an investigation meets its scientific requirements. Yet a high Atomic Score alone cannot establish whether the research question has been resolved. Among 229 attempts scoring at least 8/10, 83 (36.2%) do not reach RC4. Of these, 55 remain at RC2, indicating that substantial local progress can coexist with unresolved requirements for the main scientific conclusion, rather than merely incomplete validation.

Panel B provides a closer look at these remaining gaps. High-scoring attempts below RC4 satisfy nearly all Formulation and Output criteria, but leave 16.0% of Identification and 20.8% of Mechanism criteria unmet. The corresponding fractions among RC4 attempts are only 2.1% and 3.0%. This suggests that producing the required results does not necessarily mean that their scientific implications are established. For example, one investigation generated the requested spectra and separated their contributions to the signal, but failed to connect these results to the underlying physical mechanism. Most of the numerical work was complete, yet a key part of the explanation remained missing.

A different limitation arises when the main mechanism is supported, but the validity of the conclusion remains uncertain. The Validation/Scope (V) results, shown separately in the inset of Panel B, capture this aspect of scientific closure. Across the benchmark, scope justification is satisfied more frequently (86.9%) than required validation (59.6%) or risk-linked validation (47.8%). In another high-scoring investigation, the main mechanism was established, but an applicability concern identified through the agent’s own calculations remained unresolved. The remaining uncertainty concerned the validity of the conclusion, rather than the mechanism itself. Addressing such concerns requires appropriate validation or restricting the conclusion to a regime supported by the evidence.

## 5.3 Evidence Pathways toward Scientific Closure

Figure 5 compares three investigations of the same attoclock problem. How does post-release scattering change an electron’s angular offset as laser intensity and screening length vary? (Bray et al., 2018) All three calculate Coulomb and screened potentials matched to the same binding energy. The task requires interpreting the screening-dependent intensity response and establishing how far the reduced scattering model supports that interpretation.

Gemini-3.1-Pro generates relevant outputs but describes increasing or turning-over screened responses as monotonically decreasing, leaving the main trend and its mechanism unsupported (RC2). GPT-5.6- Sol identifies and explains the response changes, but does not resolve the high-field applicability risk exposed by its own Coulomb barrier diagnostic (RC3). Kimi-K3 corrects an angle-sign inconsistency, explains the competition between exit radius and screening length, and justifies its applicability limits (RC4). Appendix E documents these scientific revisions, numerical results, and applicability limits.

The comparison distinguishes an incorrect interpretation of computed trends from a supported mechanism with unresolved applicability limits.

## 5.4 Research Closure across Repeated Runs

A model may reach RC4 on a task in one run but fall short in another. Figure 6 compares the proportion of tasks reaching RC4 at least once with the proportion reaching RC4 in all three independent runs.

Claude-Opus-5 reaches RC4 at least once on 50.0% of tasks, but on only 18.0% in all three runs. GPT-5.6-Sol and Kimi-K3 each reach RC4 at least once on 40.0% of tasks, but reach RC4 in all three runs on only 10.0% and 4.0% of tasks, respectively. These two measures distinguish a model’s demonstrated ability to reach closure on a task from the reliability of achieving closure across independent runs.

![](images/2349caf24c7482254040396dc07bae50c165e7a0a1ba6468f47488d799d83a1c.jpg)

DeepSeek-V4.1-Flash reaches RC4 at least once on 24.0% of tasks, but on none in all three runs. Requiring RC4 across all three independent investigations is substantially more stringent than achieving it at least once. Even the strongest models can demonstrate that closure is achievable on a task without achieving it reliably across repeated investigations.

Figure 6: Research Closure across repeated runs. For each model, filled markers show the fraction of the 50 tasks reaching RC4 in at least one of three independent runs, while open markers show the fraction reaching RC4 in all three runs.

## 6 Conclusion

We introduced PhysClosureBench, a benchmark for evaluating scientific-agent performance beyond answer correctness or artifact generation. The benchmark combines goal-specified physics research tasks with an evaluation framework that assesses both atomic scientific accomplishment and Research Closure. Across seven frontier models, we found that higher Atomic Scores generally accompanied higher closure levels, yet high-scoring investigations did not always achieve complete closure. Case-level analysis further showed that investigations can fail for qualitatively different reasons, including misinterpreted results, unsupported physical mechanisms, and unresolved applicability limits. Repeated runs revealed a further gap between demonstrating the ability to achieve closure and achieving it reliably. These findings highlight the need to evaluate not only how much scientific work agents accomplish, but also whether their investigations progress from a research question to a well-supported conclusion, and how reliably they achieve complete closure across independent runs.

## Acknowledgments

This work was supported by the Shanghai Artificial Intelligence Laboratory. We are grateful to the authors and open-source communities whose work made this project possible.

## References

Alibaba Group. Alibaba Unveils Qwen3.8-Max: Its Largest and Most Capable Flagship Model to Date, 2026. URL https://www.alibabagroup.com/en-US/ document-2021044032125272064. Official release, 3 August 2026.

Anomaly. OpenCode, 2026. URL https://github.com/anomalyco/opencode/ releases/tag/v1.18.17. Version 1.18.17. Official software release.

Anthropic. Introducing Claude Opus 5, 2026. URL https://www.anthropic.com/news/ claude-opus-5. Official release, 24 July 2026.

James Bogen and James Woodward. Saving the phenomena. The Philosophical Review, 97(3): 303–352, 1988. URL http://www.jstor.org/stable/2185445.

Alexander W. Bray, Sebastian Eckart, and Anatoli S. Kheifets. Keldysh-rutherford model for the attoclock. Phys. Rev. Lett., 121:123201, Sep 2018. doi: 10.1103/PhysRevLett.121.123201. URL https://link.aps.org/doi/10.1103/PhysRevLett.121.123201.

Center for AI Safety, Scale AI, and HLE Contributors Consortium. A benchmark of expert-level academic questions to assess AI capabilities. Nature, 649:1139–1146, 2026. URL https: //doi.org/10.1038/s41586-025-09962-4.

Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, Aleksander Madry, and Lilian Weng. MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 50466–50494, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ 7e3767db483c942b883eb4f8cfb74e31-Paper-Conference.pdf.

Ziru Chen, Shijie Chen, Yuting Ning, Qianheng Zhang, Boshi Wang, Botao Yu, Yifei Li, Zeyi Liao, Chen Wei, Zitong Lu, Vishal Dey, Mingyi Xue, Frazier N. Baker, Benjamin Burns, Daniel Adu-Ampratwum, Xuhui Huang, Xia Ning, Song Gao, Yu Su, and Huan Sun. ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 96934–96990, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ f12b4df26344f3be803c06b555252efe-Paper-Conference.pdf.

Lidice Cruz-Rodriguez, Diptesh Dey, Antonia Freibert, and Philipp Stammer. Quantum phenomena in attosecond science. Nature Reviews Physics, 6:691–704, 2024. URL https://doi.org/ 10.1038/s42254-024-00769-2.

DeepSeek-AI. DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression, 2026. URL https://arxiv.org/abs/2609.19969.

Anton Frisk Kockum, Adam Miranowicz, Simone De Liberato, Salvatore Savasta, and Franco Nori. Ultrastrong coupling between light and matter. Nature Reviews Physics, 1:19–40, 2019. URL https://doi.org/10.1038/s42254-018-0006-2.

Google DeepMind. Gemini 3.1 Pro, 2026. URL https://deepmind.google/models/ model-cards/gemini-3-1-pro/. Official model card, published 19 February 2026.

Kimi Team. Kimi K3: Open Frontier Intelligence, 2026. URL https://arxiv.org/abs/ 2607.24653.

Ferenc Krausz and Misha Ivanov. Attosecond physics. Rev. Mod. Phys., 81:163–234, Feb 2009. doi: 10.1103/RevModPhys.81.163. URL https://link.aps.org/doi/10.1103/ RevModPhys.81.163.

Stanislav Yu. Kruchinin, Ferenc Krausz, and Vladislav S. Yakovlev. Colloquium: Strong-field phenomena in periodic systems. Rev. Mod. Phys., 90:021002, Apr 2018. doi: 10.1103/RevModPhys.90. 021002. URL https://link.aps.org/doi/10.1103/RevModPhys.90.021002.

Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery, 2024. URL https: //arxiv.org/abs/2408.06292.

Deborah G. Mayo. Error and the Growth of Experimental Knowledge. The University of Chicago Press, 1996.

Tingjia Miao, Wenkai Jin, Muhua Zhang, Jinxin Tan, Yuelin Hu, Tu Guo, Jiejun Zhang, Yuhan Wang, Wenbo Li, Yinuo Gao, Shuo Chen, Weiqi Jiang, Yayun Hu, Zixing Lei, Xianghe Pang, Zexi Liu, Yuzhi Zhang, Linfeng Zhang, Kun Chen, Wei Wang, Weinan E, and Siheng Chen. PRL-Bench: A

Comprehensive Benchmark Evaluating LLMs’ Capabilities in Frontier Physics Research, 2026. URL https://arxiv.org/abs/2604.15411.

OpenAI. GPT-5.6 Sol model, 2026. URL https://developers.openai.com/api/docs/ models/gpt-5.6-sol. Official model documentation.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A Graduate-Level Google-Proof Q&A Benchmark. In First Conference on Language Modeling, 2024. URL https://openreview. net/forum?id=Ti67584b98.

Giulio Starace, Oliver Jaffe, Dane Sherburn, James Aung, Jun Shern Chan, Leon Maksin, Rachel Dias, Evan Mays, Benjamin Kinsella, Wyatt Thompson, Johannes Heidecke, Amelia Glaese, and Tejal Patwardhan. PaperBench: Evaluating AI’s Ability to Replicate AI Research. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 56843–56873. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/starace25a.html.

Minyang Tian, Luyu Gao, Shizhuo Dylan Zhang, Xinan Chen, Cunwei Fan, Xuefei Guo, Roland Haas, Pan Ji, Kittithat Krongchon, Yao Li, Shengyan Liu, Di Luo, Yutao Ma, Hao Tong, Kha Trinh, Chenyu Tian, Zihan Wang, Bohao Wu, Yanyu Xiong, Shengzhu Yin, Minhui Zhu, Kilian Lieret, Yanxin Lu, Genglin Liu, Yufeng Du, Tianhua Tao, Ofir Press, Jamie Callan, Eliu Huerta, and Hao Peng. SciCode: A Research Coding Benchmark Curated by Scientists. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 30624–30650. Curran Associates, Inc., 2024. doi: 10.52202/079017-0963. URL https://proceedings.neurips.cc/paper\_files/ paper/2024/file/36850592258c8c41cecdaa3dea5ff7de-Paper-Datasets\_ and\_Benchmarks\_Track.pdf.

Miles Wang, Robi Lin, Kat Hu, Joy Jiao, Neil Chowdhury, Ethan Chang, and Tejal Patwardhan. FrontierScience: Evaluating AI’s Ability to Perform Expert-Level Scientific Tasks, 2026. URL https://arxiv.org/abs/2601.21165.

Hjalmar Wijk, Tao Roa Lin, Joel Becker, Sami Jawhar, Neev Parikh, Thomas Broadley, Lawrence Chan, Michael Chen, Joshua M Clymer, Jai Dhyani, Elena Ericheva, Katharyn Garcia, Brian Goodrich, Nikola Jurkovic, Megan Kinniment, Aron Lajko, Seraphina Nix, Lucas Jun Koba Sato, William Saunders, Maksym Taran, Ben West, and Elizabeth Barnes. RE-Bench: Evaluating Frontier AI R&D Capabilities of Language Model Agents against Human Experts. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 66772–66832. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/wijk25a.html.

Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jeff Clune, and David Ha. The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search, 2025. URL https://arxiv.org/abs/2504.08066.

Z.ai. GLM-5.3-Flash/FlashX, 2026. URL https://docs.z.ai/guides/vlm/glm-5. 3-flash. Official model documentation.

Yanbo Zhang, Sumeer A. Khan, Adnan Mahmud, Huck Yang, Alexander Lavin, Michael Levin, Jeremy Frey, Jared Dunnmon, James Evans, Alan Bundy, Saso Dzeroski, Jesper Tegner, and Hector Zenil. Exploring the role of large language models in the scientific method: from hypothesis to discovery. npj Artificial Intelligence, 1:14, 2025. URL https://doi.org/10.1038/ s44387-025-00019-5.

Yu Zhang, Xiusi Chen, Bowen Jin, Sheng Wang, Shuiwang Ji, Wei Wang, and Jiawei Han. A Comprehensive Survey of Scientific Large Language Models and Their Applications in Scientific Discovery. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 8783–8817, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/ 2024.emnlp-main.498. URL https://aclanthology.org/2024.emnlp-main.498/.

Tianshi Zheng, Zheye Deng, Hong Ting Tsang, Weiqi Wang, Jiaxin Bai, Zihao Wang, and Yangqiu Song. From Automation to Autonomy: A Survey on Large Language Models in Scientific Discovery. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 17733–17750, Suzhou, China, November 2025. Association for Computational

Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.895. URL https: //aclanthology.org/2025.emnlp-main.895/.

Minhui Zhu, Minyang Tian, Xiaocheng Yang, Tianci Zhou, Lifan Yuan, Penghao Zhu, Eli Chertkov, Shengyan Liu, Yufeng Du, Ziming Ji, Indranil Das, Qingzhi Chen, Junyi Cao, Yufeng Du, Jiabin Yu, Peixue Wu, Jinchen He, Yifan Su, Yikun Jiang, Yujie Zhang, Chang Liu, Ze-Min Huang, Weizhen Jia, Yunkai Wang, Farshid Jafarpour, Yong Zhao, Xinan Chen, Jessie Shelton, Aaron W. Young, John Bartolotta, Wenchao Xu, Yue Sun, Anjun Chu, Victor Colussi, Chris Akers, Nathan Brooks, Wenbo Fu, Jinchao Zhao, Marvin Qi, Anqi Mu, Yubo Yang, Allen Zang, Yang Lyu, Peizhi Mai, Christopher Wilson, Xuefei Guo, Juntai Zhou, Daniel Inafuku, Chi Xue, Luyu Gao, Ze Yang, Yaïr Hein, Yonatan Kahn, Kevin Zhou, Di Luo, John Drew Wilson, Jarrod T. Reilly, Dmytro Bandak, Ofir Press, Liang Yang, Xueying Wang, Hao Tong, Nicolas Chia, Eliu Huerta, and Hao Peng. Probing the Critical Point (CritPt) of AI Reasoning: a Frontier Physics Research Benchmark, 2026. URL https://arxiv.org/abs/2509.26574.

## A Benchmark Tasks

## A.1 Benchmark Construction and Quality Control

PhysClosureBench uses a semi-automated construction process with expert guidance. A domain expert with research experience in ultrafast and strong-field physics designed, iteratively calibrated, and supervised the task-construction pipeline. Objectives based on published studies were formulated into bounded research tasks. Structured model-assisted generation and multi-pass checks supported task formulation, rubric construction, and consistency checks against the source paper.

During pipeline development and benchmark construction, review by other domain experts assessed task validity, grading criteria, scientific requirements and their prerequisites, acceptable alternative evidence, and the separation between task instructions given to agents and hidden grading criteria. Additional domain experts provided scientific feedback and checks on representative tasks and benchmark design choices. Successful reference reproduction was required before a task was admitted to the benchmark, confirming that the intended physical behavior could be reproduced within the benchmark setting.

For each case, the task specification, resource limits, and grading rubric were finalized before the model evaluation.

Physics domains covered by the benchmark 50 benchmark tasks  
![](images/cacb41860a2a34d9ae2751839f916931c11a6698fb83b221872a5a77dbc1ab56.jpg)  
Figure 7: Domain distribution of the 50 benchmark tasks.

Table 4: Case-level CPU and wall-time limits for the 50 benchmark tasks.
<table><tr><td>Case</td><td>CPU cores</td><td>Minutes</td><td>Case</td><td>CPU cores</td><td>Minutes</td></tr><tr><td>Case01</td><td>8</td><td>30</td><td>Case26</td><td>16</td><td>90</td></tr><tr><td>Case02</td><td>16</td><td>30</td><td>Case27</td><td>8</td><td>60</td></tr><tr><td>Case03</td><td>32</td><td>240</td><td>Case28</td><td>16</td><td>60</td></tr><tr><td>Case04</td><td>8</td><td>30</td><td>Case29</td><td>32</td><td>90</td></tr><tr><td>Case05</td><td>32</td><td>45</td><td>Case30</td><td>8</td><td>45</td></tr><tr><td>Case06</td><td>8</td><td>45</td><td>Case31</td><td>96</td><td>360</td></tr><tr><td>Case07</td><td>8</td><td>45</td><td>Case32</td><td>96</td><td>720</td></tr><tr><td>Case08</td><td>32</td><td>60</td><td>Case33</td><td>8</td><td>45</td></tr><tr><td>Case09</td><td>16</td><td>30</td><td>Case34</td><td>96</td><td>180</td></tr><tr><td>Case10</td><td>8</td><td>30</td><td>Case35</td><td>96</td><td>120</td></tr><tr><td>Case11</td><td>16</td><td>60</td><td>Case36</td><td>8</td><td>45</td></tr><tr><td>Case12</td><td>16</td><td>60</td><td>Case37</td><td>8</td><td>45</td></tr><tr><td>Case13</td><td>8</td><td>45</td><td>Case38</td><td>16</td><td>90</td></tr><tr><td>Case14</td><td>8</td><td>45</td><td>Case39</td><td>96</td><td>120</td></tr><tr><td>Case15</td><td>8</td><td>60</td><td>Case40</td><td>16</td><td>90</td></tr><tr><td>Case16</td><td>8</td><td>45</td><td>Case41</td><td>96</td><td>180</td></tr><tr><td>Case17</td><td>16</td><td>45</td><td>Case42</td><td>96</td><td>180</td></tr><tr><td>Case18</td><td>8</td><td>45</td><td>Case43</td><td>32</td><td>90</td></tr><tr><td>Case19</td><td>16</td><td>60</td><td>Case44</td><td>8</td><td>45</td></tr><tr><td>Case20</td><td>32</td><td>90</td><td>Case45</td><td>32</td><td>90</td></tr><tr><td>Case21</td><td>8</td><td>45</td><td>Case46</td><td>16</td><td>90</td></tr><tr><td>Case22</td><td>8</td><td>60</td><td>Case47</td><td>8</td><td>45</td></tr><tr><td>Case23</td><td>8</td><td>45</td><td>Case48</td><td>16</td><td>60</td></tr><tr><td>Case24</td><td>8</td><td>60</td><td>Case49</td><td>8</td><td>60</td></tr><tr><td>Case25</td><td>8</td><td>60</td><td>Case50</td><td>8</td><td>45</td></tr></table>

## A.2 Task-Specific Resource Limits

All 50 cases use isolated, network-disabled, single-node environments. Table 4 summarizes the CPU-core allocations and wall-time limits for all 50 cases. These limits define the computational resources available for each investigation.

Resource calibration. Successful reference reproductions were used to calibrate the CPU and wall-time budgets for all 50 cases. The assigned limits were based on observed reproduction costs, with additional allowance for planning, exploratory calculations, method selection, and unsuccessful intermediate attempts. The resulting resource limits were intended to provide sufficient resources for a scientifically viable investigation while maintaining a bounded execution setting.

## A.3 Agent Execution Environment and Examination Protocol

Each attempt was conducted in a separate OpenCode session and workspace. The OpenCode deployment used version 1.18.17. Slurm allocated task-specific computational resources, while scientific code was executed through a Bash interface in an isolated, network-disabled container. Model inference was handled separately through the configured model service. The environment provided numerical Python libraries and compiled-language tools, allowing agents to implement, execute, and analyze their own calculations.

At the start of each run, agents received common instructions and task-specific research materials. EXAM\_INSTRUCTIONS.md provided the common examination rules and submission requirements. CANDIDATE\_ENVIRONMENT.md described the available computational tools. The task/ directory contained the case-specific research specification and resource limits, with accompanying task materials where supplied. These defined the scientific objective, physical setting, parameters, and constraints. The Case 30 specification is reproduced in Appendix G. The retained common initial instruction template was:

## Common initial instruction

You are an isolated research-exam candidate in case {case\_id}, attempt {attempt}.

Read EXAM\_INSTRUCTIONS.md and every supplied file under task/ before deciding on your research route. Complete the research task independently. You may derive equations, write code, run Linux commands, generate data and figures, and revise your approach within the visible time and resource limits.

All final answer materials must be placed under submission/ using the required entry-point filenames. Keep scratch work under scratch/. Do not ask the user or examiner for guidance. Do not use network search, external agents, hidden files, previous submissions, target-paper materials, or unrecorded references. If the calculation cannot be completed, submit the strongest auditable partial result and document the blocker rather than fabricating data.

Before finishing, verify that submission/ contains the required reports, code/data/figure artifacts needed to audit your claims, and honest reference/integrity declarations.

Candidates could query the remaining time and choose their own computational, analytical, and validation procedures within these requirements.

Standard submission files documented the scientific report, derivations, execution history, references, integrity declarations, and links between reported claims and supporting artifacts (Appendix G). Agents organized task-specific code, data, and figures. At completion or the work deadline, the controller collected the available submission for evaluation without allowing further scientific work during collection.

## B Grading Protocol

## B.1 Automated Grading and Scoring Semantics

Tables 2 and 3 summarize the functional dimensions and Research Closure states used throughout the evaluation. Each case has a grading rubric specifying twenty binary half-point atoms together with the task-specific scientific requirements and their prerequisite structure. Conditional criteria are activated according to the method, evidence, or conclusion produced by the candidate.

GPT-5.6-Sol at high reasoning effort serves as the automated evaluator throughout the staged grading protocol. Each assessable submission first receives an independent first-pass grade using the fixed evaluation prompt and case-specific rubric. A targeted review then checks grading decisions affecting the score or RC assessment, the effects of unresolved prerequisites, total score, Research Closure, and attempt validity with access to the first-pass assessment. Conditional adjudication is triggered only when a residual material disagreement could affect the final score, Research Closure, or validity of the attempt. Appendix H provides the common grading rules, first-pass and targeted-review prompts, and the conditional adjudication procedure.

The Atomic Score is the sum of twenty fixed half-point atoms. Some criteria in the case-specific rubric apply only under specified circumstances. If these circumstances do not arise in a scientifically assessable attempt, the corresponding criteria are marked N/A. An N/A criterion retains its fixed 0.5-point contribution to the 10-point score. This convention avoids penalizing an attempt for an inapplicable requirement; it does not imply that the corresponding scientific work was performed. Therefore, the N/A criterion is excluded from both the numerator and denominator of applicablecriterion functional completion. For risk-triggered Validation/Scope criteria, N/A therefore indicates only that the triggering risk did not arise, not that positive validation evidence was produced. A run with no scientifically assessable submission (a terminal-zero outcome) receives score 0 and RC0 directly, before atom-level scoring, and receives no N/A credit.

Table 5 summarizes the allocation of all 1,000 rubric atoms across the five functional dimensions and their variation across the 50 tasks. Each atom is worth 0.5 points, so the number assigned to a dimension determines its share of each task’s 10-point maximum.

Research Closure is determined by the task-specific scientific requirements needed to resolve the research question, rather than by a threshold on the Atomic Score. The rubric links designated atomic criteria to closure requirements, but each requirement is assessed against its specified scientific conditions independently of the credit awarded for the corresponding criterion. Unmet indispensable requirements constrain the attainable RC level according to their scientific role. The resulting RC0– RC4 states follow the semantics summarized in Table 3. Appendix G presents selected criteria from grading rubric of Case 30.

Table 5: Atom allocation across 50 cases. Total atom counts across all cases; Mean and Median are calculated from the 50 per-case atom counts for each dimension; Range gives their minimum and maximum. Mean score weight is the average percentage of each task’s 10-point maximum allocated to a dimension.
<table><tr><td>Dimension</td><td>Total</td><td>Mean</td><td>Median</td><td>Range</td><td>Mean score weight</td></tr><tr><td>F</td><td>164</td><td>3.28</td><td>3</td><td>2-4</td><td>16.4%</td></tr><tr><td>0</td><td>192</td><td>3.84</td><td>4</td><td>2-6</td><td>19.2%</td></tr><tr><td>I</td><td>163</td><td>3.26</td><td>3</td><td>3-4</td><td>16.3%</td></tr><tr><td>M</td><td>369</td><td>7.38</td><td>7</td><td>7-9</td><td>36.9%</td></tr><tr><td>V</td><td>112</td><td>2.24</td><td>2</td><td>1-3</td><td>11.2%</td></tr><tr><td>Total</td><td>1000</td><td>20.00</td><td>20</td><td>20-20</td><td>100.0%</td></tr></table>

## B.2 Human Consistency Check of Selected Grading Decisions

Five physics-trained experts reviewed 100 evaluation cards covering all 50 benchmark cases. For each case, two submissions from different models were selected, with two material rubric criteria examined per submission, yielding 200 requested criterion-level judgments. The review cards contained the candidate evidence, relevant physical context, the selected rubric criteria, and accepted equivalent routes. Criteria were selected to provide broad case coverage and to emphasize scientifically consequential, locally assessable grading decisions rather than to form an equal-probability sample of all rubric atoms.

Of the 200 requested judgments, 192 yielded effective decisions, with 81.8% of the judgments agreeing with the corresponding automated criterion-level decisions. RC-relevant consistency was assessed using atomic criteria linked to essential task-specific requirements, including central scientific objects, indispensable components, principal mechanisms, and material validation. These roles concern scientific requirements whose failure can limit the attainable RC level. The RC-relevant comparisons covered 84 judgments across 72 cards and 44 tasks, with 60 agreements among 79 binary judgments (75.9%). Both rates measure criterion-level agreement with automated atomic judgments; RC assignment additionally accounts for task-specific trigger conditions and evidence dependencies.

## C Statistical Analysis

The analysis comprises 1,050 final scored attempts originating from seven models, 50 cases, and three independent runs per model–case pair. The attempt is the basic analysis unit. Incomplete attempts, including runs that reach the time limit, remain part of the scored population; infrastructure-invalid executions and superseded retries do not create additional observations.

For attempt a and dimension $d \in \{ F , O , I , M \}$ , let $s _ { a d }$ and $n _ { a d }$ denote satisfied and applicable atoms. Functional completion is

$$
x _ { a d } = { \frac { s _ { a d } } { n _ { a d } } } , \qquad \bar { x } _ { G d } = { \frac { 1 } { | G | } } \sum _ { a \in G } x _ { a d } ,
$$

where G denotes the attempts belonging to the model or Research Closure group being summarized. N/A atoms enter neither the numerator nor the denominator. Terminal-zero attempts without atomlevel grading contribute $x _ { a d } = 0$ . The reported group value is the mean of these attempt-level completion rates, so each attempt contributes equally regardless of its number of applicable atoms.

Validation/Scope (V) is summarized separately at the criterion level as the fraction of applicable V criteria that are satisfied within the reported group. Applicability depends on the method used, the evidence produced, and the resulting conclusion. N/A criteria and terminal-zero attempts without atom records do not enter this criterion-level denominator.

Besides, in the joint score–RC distribution of Figure 4A, each cell is a percentage of all 1,050 attempts, not an RC proportion conditional on its score band.

## D Additional Results

## D.1 Detailed Dimension and Research Closure Results

Tables 6–9 report the model-level and run-level values underlying the main-text dimension and Research Closure summaries. F/O/I/M follow the attempt-level convention defined in Appendix C; V is reported separately using its applicable-criterion denominator. The RC tables report all five closure levels. Run-level summaries cover the three independent runs for each model–case pair.

Table 6: Functional completion by model. F/O/I/M values are mean attempt-level completion rates, with terminal-zero attempts contributing zero. V values represent the percentage of applicable V criteria satisfied, excluding N/A criteria and terminal-zero attempts without atom records.
<table><tr><td>Model</td><td>F (%)</td><td>O (%)</td><td>I (%)</td><td>M (%)</td><td>V (%)</td><td>V satisfied/applicable</td></tr><tr><td>Claude Opus 5</td><td>85.6</td><td>60.1</td><td>54.6</td><td>52.8</td><td>84.0</td><td>263/313</td></tr><tr><td>GPT-5.6 Šol</td><td>83.8</td><td>60.5</td><td>47.9</td><td>46.0</td><td>74.2</td><td>241/325</td></tr><tr><td>Kimi K3</td><td>77.4</td><td>49.7</td><td>43.4</td><td>40.9</td><td>68.4</td><td>214/313</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>67.2</td><td>37.3</td><td>35.2</td><td>35.7</td><td>71.3</td><td>214/300</td></tr><tr><td>Qwen3.8-Max</td><td>62.3</td><td>28.7</td><td>24.1</td><td>25.9</td><td>54.7</td><td>156/285</td></tr><tr><td>GLM-5.3-Flash</td><td>56.7</td><td>23.7</td><td>20.3</td><td>23.3</td><td>65.6</td><td>177/270</td></tr><tr><td>Gemini 3.1 Pro</td><td>61.9</td><td>22.7</td><td>13.7</td><td>15.3</td><td>47.8</td><td>121/253</td></tr></table>

Table 7: Functional completion by models in three independent runs. Measures and applicability conventions follow Table 6.
<table><tr><td>Model</td><td>Run</td><td>F</td><td>0</td><td>I</td><td>M</td><td>V</td><td>V satisfied/applicable</td></tr><tr><td>Claude Opus 5</td><td>R1</td><td>88.2</td><td>62.5</td><td>58.3</td><td>56.2</td><td>84.1</td><td>90/107</td></tr><tr><td></td><td>R2</td><td>83.7</td><td>63.6</td><td>56.8</td><td>55.5</td><td>82.5</td><td>85/103</td></tr><tr><td></td><td>R3</td><td>84.8</td><td>54.4</td><td>48.5</td><td>46.9</td><td>85.4</td><td>88/103</td></tr><tr><td>GPT-5.6 Sol</td><td>R1</td><td>86.8</td><td>58.7</td><td>49.3</td><td>48.3</td><td>76.8</td><td>86/112</td></tr><tr><td></td><td>R2</td><td>78.5</td><td>57.0</td><td>39.0</td><td>39.6</td><td>66.3</td><td>67/101</td></tr><tr><td></td><td>R3</td><td>86.2</td><td>65.9</td><td>55.3</td><td>50.0</td><td>78.6</td><td>88/112</td></tr><tr><td>Kimi K3</td><td>R1</td><td>74.8</td><td>43.0</td><td>35.7</td><td>34.0</td><td>59.6</td><td>65/109</td></tr><tr><td></td><td>R2</td><td>81.0</td><td>54.9</td><td>51.8</td><td>46.5</td><td>68.2</td><td>75/110</td></tr><tr><td></td><td>R3</td><td>76.3</td><td>51.3</td><td>42.7</td><td>42.2</td><td>78.7</td><td>74/94</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>R1</td><td>65.0</td><td>36.5</td><td>34.2</td><td>33.9</td><td>68.9</td><td>71/103</td></tr><tr><td></td><td>R2</td><td>71.8</td><td>39.1</td><td>36.8</td><td>38.3</td><td>69.2</td><td>74/107</td></tr><tr><td></td><td>R3</td><td>64.8</td><td>36.2</td><td>34.7</td><td>35.0</td><td>76.7</td><td>69/90</td></tr><tr><td>Qwen3.8-Max</td><td>R1</td><td>55.5</td><td>25.1</td><td>17.8</td><td>17.9</td><td>41.5</td><td>34/82</td></tr><tr><td></td><td>R2</td><td>66.8</td><td>31.4</td><td>28.2</td><td>32.2</td><td>65.0</td><td>65/100</td></tr><tr><td></td><td>R3</td><td>64.7</td><td>29.7</td><td>26.2</td><td>27.5</td><td>55.3</td><td>57/103</td></tr><tr><td>GLM-5.3-Flash</td><td>R1</td><td>58.8</td><td>31.7</td><td>27.2</td><td>28.0</td><td>62.6</td><td>62/99</td></tr><tr><td></td><td>R2</td><td>61.2</td><td>25.9</td><td>21.7</td><td>25.2</td><td>57.9</td><td>55/95</td></tr><tr><td></td><td>R3</td><td>50.2</td><td>13.6</td><td>12.2</td><td>16.6</td><td>78.9</td><td>60/76</td></tr><tr><td>Gemini 3.1 Pro</td><td>R1</td><td>66.5</td><td>22.5</td><td>15.0</td><td>14.2</td><td>36.2</td><td>38/105</td></tr><tr><td></td><td>R2</td><td>60.2</td><td>22.6</td><td>14.5</td><td>17.1</td><td>57.3</td><td>43/75</td></tr><tr><td></td><td>R3</td><td>59.2</td><td>23.2</td><td>11.5</td><td>14.5</td><td>54.8</td><td>40/73</td></tr></table>

F–V entries are percentages; the final column reports the numbers of satisfied and applicable V criteria.

Table 8: Research Closure distribution by model. Cells report count (percentage) over the 150 final scored attempts for each model.
<table><tr><td>Model</td><td>N</td><td>RC0</td><td>RC1</td><td>RC2</td><td>RC3</td><td>RC4</td></tr><tr><td>Claude Opus 5</td><td>150</td><td>24 (16.0%)</td><td>49 (32.7%)</td><td>25 (16.7%)</td><td>3 (2.0%)</td><td>49 (32.7%)</td></tr><tr><td>GPT-5.6 Sol</td><td>150</td><td>28 (18.7%)</td><td>38 (25.3%)</td><td>43 (28.7%)</td><td>7 (4.7%)</td><td>34 (22.7%)</td></tr><tr><td>Kimi K3</td><td>150</td><td>48 (32.0%)</td><td>36 (24.0%)</td><td>32 (21.3%)</td><td>5 (3.3%)</td><td>29 (19.3%)</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>150</td><td>75 (50.0%)</td><td>25 (16.7%)</td><td>30 (20.0%)</td><td>4 (2.7%)</td><td>16 (10.7%)</td></tr><tr><td>Qwen3.8-Max</td><td>150</td><td>76 (50.7%)</td><td>39 (26.0%)</td><td>22 (14.7%)</td><td>2 (1.3%)</td><td>11 (7.3%)</td></tr><tr><td>GLM-5.3-Flash</td><td>150</td><td>87 (58.0%)</td><td>32 (21.3%)</td><td>19 (12.7%)</td><td>4 (2.7%)</td><td>8 (5.3%)</td></tr><tr><td>Gemini 3.1 Pro</td><td>150</td><td>76 (50.7%)</td><td>31 (20.7%)</td><td>41 (27.3%)</td><td>1 (0.7%)</td><td>1 (0.7%)</td></tr></table>

Table 9: Research Closure counts by model and independent run. Each row reports 50 scored attempts.
<table><tr><td>Model</td><td>Run</td><td>N</td><td>RC0</td><td>RC1</td><td>RC2</td><td>RC3</td><td>RC4</td></tr><tr><td>Claude Opus 5</td><td>R1</td><td>50</td><td>8</td><td>14</td><td>7</td><td>1</td><td>20</td></tr><tr><td></td><td>R2</td><td>50</td><td>11</td><td>9</td><td>12</td><td>2</td><td>16</td></tr><tr><td></td><td>R3</td><td>50</td><td>5</td><td>26</td><td>6</td><td>0</td><td>13</td></tr><tr><td>GPT-5.6 Sol</td><td>R1</td><td>50</td><td>7</td><td>13</td><td>14</td><td>5</td><td>11</td></tr><tr><td></td><td>R2</td><td>50</td><td>15</td><td>9</td><td>13</td><td>2</td><td>11</td></tr><tr><td></td><td>R3</td><td>50</td><td>6</td><td>16</td><td>16</td><td>0</td><td>12</td></tr><tr><td>Kimi K3</td><td>R1</td><td>50</td><td>20</td><td>13</td><td>7</td><td>2</td><td>8</td></tr><tr><td></td><td>R2</td><td>50</td><td>14</td><td>10</td><td>15</td><td>1</td><td>10</td></tr><tr><td></td><td>R3</td><td>50</td><td>14</td><td>13</td><td>10</td><td>2</td><td>11</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>R1</td><td>50</td><td>30</td><td>5</td><td>9</td><td>1</td><td>5</td></tr><tr><td></td><td>R2</td><td>50</td><td>22</td><td>12</td><td>6</td><td>3</td><td>7</td></tr><tr><td></td><td>R3</td><td>50</td><td>23</td><td>8</td><td>15</td><td>0</td><td>4</td></tr><tr><td>Qwen3.8-Max</td><td>R1</td><td>50</td><td>27</td><td>14</td><td>6</td><td>0</td><td>3</td></tr><tr><td></td><td>R2</td><td>50</td><td>26</td><td>12</td><td>6</td><td>1</td><td>5</td></tr><tr><td></td><td>R3</td><td>50</td><td>23</td><td>13</td><td>10</td><td>1</td><td>3</td></tr><tr><td>GLM-5.3-Flash</td><td>R1</td><td>50</td><td>28</td><td>8</td><td>8</td><td>2</td><td>4</td></tr><tr><td></td><td>R2</td><td>50</td><td>24</td><td>15</td><td>6</td><td>2</td><td>3</td></tr><tr><td></td><td>R3</td><td>50</td><td>35</td><td>9</td><td>5</td><td>0</td><td>1</td></tr><tr><td>Gemini 3.1 Pro</td><td>R1</td><td>50</td><td>25</td><td>11</td><td>13</td><td>0</td><td>1</td></tr><tr><td></td><td>R2</td><td>50</td><td>23</td><td>13</td><td>14</td><td>0</td><td>0</td></tr><tr><td></td><td>R3</td><td>50</td><td>28</td><td>7</td><td>14</td><td>1</td><td>0</td></tr></table>

Validation/Scope categories. Table 10 reports the detailed definitions for different V categories and their criterion-level satisfaction across the benchmark. N/A criteria and terminal-zero attempts without atom records are excluded from the applicable-criterion denominator.

Table 10: Validation/Scope criteria across the benchmark.
<table><tr><td>V category</td><td>Description</td><td>Satisfied/applicable</td><td>Rate</td></tr><tr><td>Scope justification</td><td>Justify the conditions under which the con- clusions are valid.</td><td>852/981</td><td>86.9%</td></tr><tr><td>Required validation</td><td>Perform validation checks required by the task, regardless of the chosen method.</td><td>93/156</td><td>59.6%</td></tr><tr><td>Risk-linked validation</td><td>Address specific risks arising from the method, results, or conclusions.</td><td>441/922</td><td>47.8%</td></tr></table>

All 26 RC3 attempts retain an unmet applicable V criterion, while 25 also satisfy another V criterion.

## D.2 Atomic-score Distributions across Investigations

Figure 8 complements the aggregate model scores by showing the full distribution of Atomic Scores across each model’s 150 scored attempts. These distributions reflect variation across benchmark tasks as well as across independent runs of the same task.

![](images/34280aa10cb39951572ec7b1d1a4dcb96d132087c2976e572ee805a7a664d96e.jpg)  
Figure 8: Atomic-score distributions across investigations. Kernel density estimates show the distribution of Atomic Scores over all final scored attempts for each model, including terminal-zero outcomes. Dashed lines indicate mean Atomic Scores for each model. The distributions summarize score variation across the full set of benchmark investigations.

## E Representative Evidence Paths

Figure 5 compares the scientific requirements supported by three Case 30 submissions. Table 11 summarizes key research activities; the paragraphs below give the decisive evidence for each closure outcome.

The task asks when post-release central-potential scattering can account for an intensity-dependent attoclock offset, and whether the scattering-based explanation for a Coulomb potential also applies to screened potentials with the same binding energy. Answering the question requires calculating the offset, identifying and explaining this change, and establishing the interpretation’s applicability. Functionally equivalent validity arguments are permitted rather than requiring a unique formula or validation route.

Table 11: Selected Case 30 research activities and key findings. The steps summarize recorded activities, not a complete or strictly ordered execution history. Scores and RC levels refer to final submissions.
<table><tr><td>Investigation</td><td>Selected research steps</td><td>Key finding or unresolved issue</td></tr><tr><td>Kimi K3 10.0/RC4</td><td>Constructed scattering calculations; detected an angle-sign inconsistency; corrected and checked the results; interpreted the screening dependence.</td><td>Supported the interpretation within a physically justified scope.</td></tr><tr><td>GPT-5.6 Sol 9.5/RC3</td><td>Repaired the numerical solver; identified screening-dependent trends; explained their mechanism; recorded a high-field diagnostic.</td><td>Did not resolve the applicability concern revealed by its high-field diagnostic.</td></tr><tr><td>Gemini 3.1 Pro 4.5/RC2</td><td>Revised trajectory propagation and angle convention; generated Coulomb and screened results; summarized the intensity dependence.</td><td>Misidentified increasing or turning-over screened curves as monotonically decreasing.</td></tr></table>

Kimi: correcting the angle and establishing applicability. Kimi initially obtained opposite offset signs from an orbit integral and direct trajectory propagation. Correcting the angle convention reduced their discrepancy to $4 . { \overset { \vartriangle } { 5 } } 3 \times 1 0 ^ { - 7 }$ rad, checking numerical consistency. It interpreted the screening dependence through the relative sizes of the tunnel-exit distance and screening length. Separately, the report restricted physical interpretation to quasistatic, sub-barrier exits and labeled other regimes as formal extensions. Two saved intensities fall within its stated Coulomb window, approximately

$0 . 0 0 6 5 < I < 0 . 0 0 7 8 \mathrm { a . u . }$ ; both retain a pronounced screening contrast. These restrictions support the reduced-model conclusion.

GPT: an unresolved high-field applicability risk. GPT correctly identified and explained the screening-dependent offset changes by balancing a shorter tunnel-exit distance against increasing electron momentum as intensity rises, but a separate issue arose in the high-field regime. At $\begin{array} { r } { E _ { 0 } = 0 . 0 8 \mathrm { a . u . } } \end{array}$ , its saved diagnostic flags a field above the Coulomb-reference barrier-suppression threshold, $E _ { \mathrm { B S I } } = 0 . 0 6 2 5$ a.u. This is not a universal threshold for screened targets. Nevertheless, the final report called the $\gamma \leq 1$ regime more favorable for its weak-field exit approximation. A low Keldysh parameter does not resolve the separate barrier-suppression risk. The mechanism is supported, but the report did not adequately reconcile this diagnostic with its high-field interpretation, yielding RC3.

Gemini: an intensity trend contradicted by its own data. Gemini generated offset results for Coulomb and screened potentials, but incorrectly summarized their intensity dependence. Its report states that the calculated offset angle θ decreases monotonically with increasing laser intensity for all tested potentials. The saved $\lambda = 1 { - } 3$ curves instead rise, while $\lambda = 5$ and 10 rise and then fall. Figure 9 shows their full submitted scans for λ = 2 and 5. Gemini discussed screening suppression and classical timing limits, but this misidentification prevented its evidence from supporting the screening-dependent response and its mechanism, yielding RC2.

![](images/4b59ef4ce981dacb5201a9d6879c236f97eee4fbcf14a1cfb7a236f78d41c46e.jpg)  
Figure 9: Screened offsets contradict Gemini’s decreasing-trend claim. Replotted from the agent’s submitted numerical data. All 100 recorded points per curve are shown, with straight connectors and no fitting or smoothing. The angle θ is the submitted outgoing-trajectory offset in degrees, with $I = 2 E _ { 0 } ^ { 2 } ; \lambda$ is in atomic units.

## F Supplementary Model-Level Results

Table 12 provides a consolidated view of model-level Atomic Scores, attempt-level RC4 rates, and task-level RC4 consistency across the 50 benchmark tasks.

Table 12: Model-level performance and RC4 consistency. Mean and median Atomic Scores and attempt-level RC4 rates are computed over 150 final scored attempts per model. The two RC4 consistency measures are computed over the same 50-task denominator for every model.
<table><tr><td>Model</td><td>Mean score</td><td>Median</td><td>RC4</td><td>≥1 RC4</td><td>All 3 RC4</td></tr><tr><td>Claude Opus 5</td><td>6.29</td><td>6.75</td><td>32.7%</td><td>50.0%</td><td>18.0%</td></tr><tr><td>GPT-5.6 Sol</td><td>5.83</td><td>6.00</td><td>22.7%</td><td>40.0%</td><td>10.0%</td></tr><tr><td>Kimi K3</td><td>5.17</td><td>5.00</td><td>19.3%</td><td>40.0%</td><td>4.0%</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>4.50</td><td>4.00</td><td>10.7%</td><td>24.0%</td><td>0.0%</td></tr><tr><td>Qwen3.8-Max</td><td>3.46</td><td>2.00</td><td>7.3%</td><td>14.0%</td><td>2.0%</td></tr><tr><td>GLM-5.3-Flash</td><td>3.28</td><td>2.25</td><td>5.3%</td><td>12.0%</td><td>0.0%</td></tr><tr><td>Gemini 3.1 Pro</td><td>2.85</td><td>2.50</td><td>0.7%</td><td>2.0%</td><td>0.0%</td></tr></table>

## G Case 30 Task Specification and Rubric Excerpt

This appendix reproduces the Case 30 task specification provided to the evaluated agents. Its formatting has been adapted to LAT<sub>E</sub>X, but the scientific content is unchanged. Selected atomic criteria and case-specific Research Closure requirements are presented afterward.

## G.1 Case 30 Task Specification

## When can a scattering model account for an attoclock offset?

Background. In an attoclock measurement, the maximum of the photoelectron momentum distribution is displaced from the direction expected from the laser vector potential at ionization. That displacement is often discussed as a timing observable, but the outgoing electron also scatters from the residual binding potential.

Research problem. For a short circularly polarized pulse, determine whether the intensity dependence of the offset angle can be accounted for by a post-release central-potential scattering model. Establish how far a conclusion obtained for an unscreened Coulomb target remains valid as the interaction is screened, and state what the resulting evidence does and does not establish about a universal interpretation of the attoclock offset.

Choose the derivation, numerical evidence, comparisons, and presentation needed to make that assessment scientifically convincing. The number and arrangement of figures are for the candidate to decide.

Physical inputs. Work in atomic units. Use a circular driver of angular frequency $\omega = 0 . 0 5 7$ and the intensity convention

$$
I = 2 E _ { 0 } ^ { 2 } , \qquad A _ { 0 } = \frac { E _ { 0 } } { \omega } ,\tag{1}
$$

where $E _ { 0 }$ and $A _ { 0 }$ are the peak electric-field and vector-potential amplitudes. Study a one-electron target with ionization potential $I _ { p } = 0 . 5$ under both

$$
V _ { C } ( r ) = - \frac { 1 } { r }\tag{2}
$$

and the screened family

$$
V _ { \lambda } ( r ) = - \frac { Z _ { \lambda } } { r } e ^ { - r / \lambda } .\tag{3}
$$

For every finite λ, determine $Z _ { \lambda }$ so that the field-free ground-state binding energy remains $I _ { p } = 0 . 5$ . Include the Coulomb limit, the strong-screening case $\lambda = 1$ , and enough intermediate screening lengths of the candidate’s choice to test whether an inference from either endpoint transfers across the family.

Method and scope boundary. Build the reduced model from the standard weak-field Keldysh exit scale, the stated vector-potential momentum scale, and the outgoing half of a central-potential scattering trajectory. Derive and declare the resulting angle convention. Standard Rutherford and screened central-potential scattering theory may be derived or cited. State the approximation level of the screened calculation used.

Choose an intensity domain that resolves the regimes needed for the claim while remaining consistent with the physical assumptions invoked. Numerical grids, root solvers, quadrature, sampling, tolerances, plotting ranges, and reliability tests are candidate choices. They are not grading targets in themselves; their consequences for the physical evidence are.

High-cost TDSE/iSURF propagation and CTMC regeneration are outside the required task. They may be discussed as external tests, but neither is required, and unsupported numerical data from those methods must not be presented as the candidate’s own result.

Submission and resources. Submit one self-contained scientific report and an auditable offline calculation package. Include the task-specific derivation, code, numerical data underlying material claims, a provenance note listing every offline source or software document actually consulted and the method basis for items supplied from general knowledge, and concise environment and run information. No fixed file tree, filename scheme, figure count, or complete exploration history is required.

The accompanying resource contract supplies the binding limits: eight CPU cores, 45 minutes of candidate-active wall time, no network access, a 64-GiB scratch allowance, and a separate ten-minute sealing reserve. CPU usage is recorded as an audit metric rather than imposed as a second hard ceiling.

Reference and integrity policy. The examination is conducted without network access or live literature retrieval. Candidates may rely on existing general scientific knowledge and on any offline materials or software documentation explicitly supplied with the examination. Every consulted source must be recorded. For formulas, methods, constants, or conventions used from general knowledge rather than a consulted source, the method basis, definition, value, units, and convention must be stated sufficiently to make the calculation auditable; citations must not be invented.

The hidden target article and its identifying metadata, figures or captions, supplementary material, author code or data, target-specific reproduction code, solution or scoring material, online solutions, and worked reproductions of the case must not be used. All submitted scientific figures and numerical data must be generated by the candidate’s own implementation of the stated problem.

Common submission requirements. In addition to the case-specific task instructions, candidates received benchmark-wide submission requirements for six top-level entries under submission/: REPORT.md, DERIVATION.md, RUN\_LOG.md, REFERENCES.md, INTEGRITY.md, and ARTIFACTS.json. These entries standardized submission collection, while the organization and naming of task-specific scientific files remained at the candidate’s discretion.

## G.2 Selected atomic criteria

The Case 30 grading rubric contains twenty binary half-point atoms. Table 13 presents representative criteria spanning the five functional dimensions. Research Closure is assessed separately using the case-specific requirements summarized in the following subsection.

Table 13: Representative Case 30 atomic criteria. Criteria are condensed from the full case-specific grading rubric for readability.
<table><tr><td>Atom</td><td>Function</td><td>Representative criterion</td></tr><tr><td>F1</td><td>Formulation</td><td>Relate the outgoing half of the central-potential deflection to the attoclock offset, using consistent exit and field/vector-potential conventions, units, and approximation limits.</td></tr><tr><td>O2</td><td>Output</td><td>Calculate auditable offset-versus-intensity results for Coulomb and screened potentials matched to the same binding energy, including λ = 1 and enough intermediate screening lengths to resolve changes with potential range.</td></tr><tr><td>I2</td><td>Identification</td><td>Identify the approach to Coulomb behavior at long range, changes in intensity dependence at intermediate screening lengths, and the suppression and relative offsets at strong screening supported by the data.</td></tr><tr><td>M2</td><td>Mechanism</td><td>Explain how the electron&#x27;s exit or impact distance, asymptotic kinetic energy, and potential range compete to change the offset response as screening length decreases.</td></tr><tr><td>V1</td><td>Validation/Scope</td><td>When numerical choices, binding-energy calibration, sampling, or curve struc- ture raise a concrete risk to the conclusion, test that risk or restrict the conclu- sion accordingly. If no risk is triggered, record N/A without claiming positive</td></tr><tr><td>V3</td><td>Validation/Scope</td><td>validation evidence. Specify which conclusions the reduced scattering calculation supports, distin- guishing them from unverified real-atom, quantum-dynamical, or universal tunnelling-time interpretations.</td></tr></table>

## G.3 Case-specific Research Closure summary

Research Closure for Case 30 is assessed separately from the atomic total, using the task-specific scientific requirements needed to answer the research problem. The requirements include a consistent offset calculation, comparisons of equally bound Coulomb and screened potentials, an explanation of the screening-dependent intensity response, and justified validation and applicability limits. The following levels illustrate Case 30, rather than a single decision template for all benchmark tasks:

RC0: the central offset calculation or its mapping from outgoing-half scattering is missing or invalid;

RC1: the central offset is established, but the required comparison with screened potentials at the same binding energy is missing or invalid;

RC2: the required calculations exist, but the change in intensity dependence with screening, its mechanism, or another principal physical conclusion is unsupported or incorrect;

RC3: the principal results and mechanism are supported, but a concrete numerical-validation, physical-applicability, or interpretation-scope issue remains unresolved;

RC4: all indispensable scientific requirements and applicable validation/scope obligations are satisfied. An N/A risk-triggered Validation/Scope atom neither blocks RC4 nor constitutes positive validation evidence.

## H Grading Instructions

## H.1 Grading Inputs and Assessment Stages

Each grading stage uses the common grading rules, the complete task-specific rubric, and the following materials, with access to previous assessments determined by the stage.

## Required inputs

• Research task: The task specification, resource limits, and common submission requirements provided to the agent.

• Task-specific grading rubric: The atomic criteria, Research Closure requirements, applica ble validation conditions, scientific dependencies, and accepted equivalent evidence.

• Submitted scientific work: The agent’s research report, derivations, figures, code, numeri cal data, and supporting artifacts.

• Execution records: Records used to verify submitted results and determine whether a run is valid or requires re-execution.

• Reference material: Relevant scientific figures, passages, or validated reproductions used to check submitted results. These materials cannot supply scientific analyses or interpretations missing from the agent’s work.

## Independence and access

The first-pass grader evaluates each submission without access to previous assessments or other submissions. The targeted reviewer can examine the first-pass assessment and the supporting evidence. If further adjudication is needed, the adjudicator receives both earlier assessments and the remaining disputed issues. At no stage may previous grades or aggregate results substitute for scientific evidence from the submission.

## H.2 Shared Grading Instructions

The following shared instructions apply to all three grading stages.

## Shared Grading Instructions

## Scientific evidence and attribution

Use the administered task, resource contract, complete task-specific rubric, and submitted scientific work. Do not change the rubric, repair the submission, or impose a requirement found only in the reference study. Do not use other submissions, prior scores, or aggregate results to determine the grade.

Distinguish four kinds of evidence: a scientific object exists in the submitted data (E0); the agent presents that object as evidence (E1); the agent identifies the scientifically important signal (E2); and the agent explains the physical inference supported by it (E3). These are different scientific acts. An evaluator-generated plot, slice, fit, or diagnostic can verify object existence and correctness, but cannot create omitted presentation, identification, mechanism, or validation evidence. Re-rendering can check an existing agent-authored presentation; it cannot replace one that the agent omitted.

Inspect every agent-authored figure supporting a scientific claim. Inspect code, data, and execution records when needed for a material scientific or validity decision. Assertions, proposed

## Shared Grading Instructions (continued)

calculations, and unsubmitted exploratory results do not establish completed scientific work. Execution records can clarify validity or responsibility for a failure, but do not supply missing scientific reasoning. Separately authorized process evidence remains limited to its explicitly permitted scope.

## Atomic scoring

Each task contains twenty criteria, each worth either 0 or 0.5 points. Atomic Score is their sum on a 0–10 scale. Do not interpolate or invent fractional credit. A criterion is satisfied when its essential scientific requirement is met; an immaterial imperfection alone does not remove credit. A missing or incorrect essential requirement receives no credit. Record a material unresolved issue for review rather than assigning invented partial credit.

Formulation (F), Output (O), Identification (I), Mechanism (M), and Validation/Scope (V) denote scientific functions, not a required sequence of research actions. Apply their task-specific criteria and allocation. Aggregate score, code length, number of figures, and computational effort do not establish functional completion.

Mechanism credit requires the agent’s presented signal and physical reasoning to support the inference. General knowledge alone, an unsupported correct conclusion, or figures connected only by the evaluator’s explanation are insufficient.

For each Output criterion, separately check whether the scientific object is correct and reconstructible, whether the agent’s presentation exposes its relevant structure, and whether the claim stays within what the evidence supports. Record these as object\_valid, presentation\_adequate, and claim\_adequate. Output credit cannot be awarded if any condition is false. Check the principal structure, overall physical pattern, and scientifically important secondary structure; one superficially matching feature does not rescue a materially incorrect result.

## Research Closure

Assess Research Closure (RC0–RC4) using the complete task-specific rubric, including the scientific requirements for each level and their necessary dependencies. Determine RC independently of the Atomic Score, dimension totals, or individual criterion credits. Complete closure requires the agent’s work to satisfy all indispensable scientific requirements, including the evidence, interpretation, and applicable validation and scope conditions needed to support the final conclusion.

Assess each closure condition against its specified scientific facts, independently of the corresponding criterion’s credit. Inadequate presentation or an unsupported claim does not by itself establish that an otherwise valid required object is absent or physically invalid. When several closure failures apply, use the most severe applicable level specified by the rubric. Explain the essential unmet requirement, or why all indispensable requirements are satisfied. RC imposes no score ceiling, floor, multiplier, or preferred distribution.

## Scientific dependencies and equivalent evidence

Apply only scientifically necessary dependencies specified by the task-specific rubric. For a downstream failure, identify the exact failed prerequisite, the dependent requirement, and why the agent’s evidence cannot establish that requirement independently. An unrelated missing part of a composite upstream criterion does not erase independently valid work. Refer a dependency without a necessary scientific premise to targeted review instead of silently changing the rubric. Do not penalize the same defect again as generic incompleteness or validation failure.

Accept scientifically valid alternative methods and representations within the rubric’s equivalence conditions. Where relevant, compare physical identity, units, conventions, support, symmetry, sign, ordering, overall pattern, and component relations. Exact pixels, plotting style, color maps, reference grids, and unspecified absolute scales are not grading targets. If evidence cannot distinguish an acceptable equivalent from a substantive error, record the

## Shared Grading Instructions (continued)

unresolved issue rather than inventing a numerical tolerance. Report an unauthorized rubric requirement or conflict with an acceptable equivalent as a rubric or fairness issue.

## Validation and applicability

Apply conditional criteria only when their task-specific conditions hold. In an assessable attempt, an inapplicable criterion is recorded as not\_applicable (N/A) and retains its fixed 0.5-point contribution. It enters neither numerator nor denominator of applicable-criterion functional completion. An untriggered validation obligation is not positive evidence that validation was performed.

A valid run with no assessable scientific work receives score 0 and RC0 before criterion-level summation. N/A credit cannot create a score for such a run. Grade incomplete submissions and assessable timeouts normally; timeout alone does not make them zero. Infrastructureinvalid executions are excluded from scoring and rerun to obtain a valid measurement, rather than counted as agent failures. Other requests for a new execution require justification under the attempt-validity rules, not an automatic retry because a score is low.

Validation concerns a concrete scientific threat applying to the submitted method, evidence, or conclusion. Accept discriminating evidence or an evidence-based scope restriction that addresses that threat; do not require one preferred check. Generic refinement, residuals, or caveats do not automatically earn or lose validation credit.

Before assigning RC3 for a validation or scope gap, establish all six requirements:

1. Failure mode: identify the concrete failure mode that remains unexcluded.

2. Origin: explain why it arises from the agent’s method, data, or conclusion.

3. Recognizable risk: identify the fair basis in the task or standard scientific reasoning for recognizing it.

4. Affected conclusion: identify the central conclusion that the failure mode threatens.

5. Insufficient evidence: explain why the submitted evidence is insufficient to expose or exclude this failure mode. Recognizing a risk does not resolve it without discriminating evidence or an evidence-based restriction of the threatened conclusion.

6. Equivalent remedies: give examples of functionally equivalent evidence that would resolve the issue, without requiring a unique validation method.

Without these elements, do not assign RC3 by default. Restricting a conclusion to an established domain removes unsupported breadth; it does not repair an incorrect signal, an absent indispensable component, or a missing principal mechanism.

## Verification of important results

Consult reference evidence when a material score or RC decision depends on feature resolution, physical versus numerical fine structure, required components and their relations, formula identity, or representational equivalence. This includes full Output credit dependent on a nonstandard projection, a cropped range, or reconstruction from submitted data. Prefer applicable reference figures or passages, then validated reproductions within their established scope, and source pages if needed. Resolve insufficient or conflicting evidence using the relevant original source; this reading order does not authorize an out-of-scope reference to override applicable scientific evidence.

Before comparison, establish the observable, case and component, units, basis, gauge, normalization, and coordinate or transform conventions, as applicable.

A provisional Atomic Score of at least 9.0 or provisional RC4 requires a focused check of one central output and one principal-mechanism link. Check physical identity, sign, ordering, and overall pattern as applicable. Internal consistency and numerical convergence alone are insufficient. The reviewer verifies that these checks are applicable and support the result, and performs any missing necessary check before acceptance.

Shared Grading Instructions (continued)

Record the submitted evidence, reference location, representation mapping, findings, scientific basis, affected decision, and remaining uncertainty. Distinguish confirmation from a changed judgment. Reference verification checks submitted results; it cannot complete scientific reasoning the agent omitted.

## H.3 Independent First-Pass Grading Prompt

## Independent First-Pass Grading Prompt

## Role and inputs

Grade one submitted investigation independently using the grading inputs (Section H.1), complete task-specific rubric, and shared grading instructions (Section H.2). Inspect the materials read-only; do not manage the examination, repair the submission, compare agents, inspect prior grades, or change the rubric. Use only this attempt’s permissible evidence.

## Scientific decisions

First establish attempt validity and whether assessable scientific work exists. Apply the terminal-zero rule before criterion-level scoring; otherwise grade the available work, including partial work.

For an assessable submission:

1. Decide each of the exact twenty criteria as pass, fail, unresolved, or not\_ applicable, applying the rubric’s scientific and applicability conditions.

2. Record all three Output conditions and complete the rubric’s central scientific checks.

3. Apply the shared evidence-attribution, necessary-dependency, and equivalentevidence rules. Preserve independently valid work without supplying missing presentation, identification, reasoning, or validation.

4. Assess RC independently from criterion credit, using the most severe applicable unmet closure requirement or explaining why all indispensable requirements are satisfied.

5. Record unresolved fairness issues and any justified request for a new execution under the attempt-validity rules.

Inspect every agent-authored figure supporting a scientific claim and conduct the required reference checks, including the central-output and principal-mechanism checks when the high-result trigger applies.

## Consistency checks

Before returning the assessment, verify three conditions:

1. An Output criterion cannot pass when object\_valid, presentation\_ adequate, or claim\_adequate is false.

2. A downstream failure follows an absent or incorrect necessary scientific premise, not an unrelated display defect or missing component. Refer contradictory dependencies to targeted review.

3. RC conditions follow their specified scientific facts independently of criterion credit. Failed presentation or claim adequacy alone does not establish absence or invalidity of an existing valid object or component.

Provide a short evidence-based reason for failed or unresolved criteria and central checks.   
Passing criteria do not require repetitive justifications.

## Independent First-Pass Grading Prompt (continued)

## Output

Return a single structured JSON assessment recording the run’s validity, decisions on all twenty atomic criteria, the required Output checks, and Research Closure with its supporting rationale. Provide brief evidence-based reasons for failed or unresolved criteria. Record the agent-authored figures inspected and any reference checks performed, noting whether those checks support or contradict the submitted results. Include any unresolved issues requiring further review.

This is the first-pass assessment, not the accepted final result. Contradictory Output flags and credit, evaluator-created presentation, unsupported RC, or a missing or inadequate mandatory check prevent unchanged acceptance at review.

## H.4 Targeted Review Prompt

## Targeted Review Prompt

## Role and inputs

Review the first-pass assessment using the grading inputs (Section H.1), complete taskspecific rubric, and shared grading instructions (Section H.2). You may inspect the initial assessment. This is a targeted review, not a routine second independent blind grade. Do not re run calculations, repair the submission, change the rubric, or overwrite the initial assessment. Verify submission identity, attempt validity, and scientific assessability. Apply the terminalzero and partial-work rules before reviewing criterion credit.

## Material checks

Check decisions affecting Atomic Score, RC, fairness, or attempt validity. Verify physicalobject and formula identity; inspection of all claim-supporting figures; scientifically important output structure; evidence attribution; necessary dependencies; accepted equivalents; and the distinction between mechanism and validation. Check task-specific closure conditions and all six elements supporting any RC3 finding.

Independently verify the first-pass consistency checks (Section H.3), including agreement between Output credit and its three conditions, justified necessary dependencies, and RC decisions independent of criterion credit. Preserve independently valid work and reject repeated deductions for the same defect.

For each high-result trigger in the shared instructions, verify that the first pass completed an applicable central-output check and principal-mechanism check. Perform any missing or inadequate required verification before acceptance. Prior grading records may identify a consistency issue but cannot replace the agent’s evidence or establish a target score distribution.

## Review decisions

Start with the first-pass evidence map and disputed criteria. Reinspect code, data, reference material, or the complete rubric only as needed for a material issue; do not routinely repeat uncontested checks.

Record changed decisions, supporting submitted evidence, the applicable rule, relevant verification findings, and remaining uncertainty. Identify which first-pass findings are accepted.

Accept an assessment with no unresolved material issue, recording any corrections. Request focused adjudication only when disagreement could affect criterion decisions, RC, fairness, attempt validity, or whether a new execution is warranted. Do not average grades, impose RC-based score ceilings, or adjust results to an expected distribution.

## H.5 Focused Adjudication Procedure

Adjudication addresses only material disagreements remaining after targeted review, not a routine third full assessment. The adjudicator receives the task, complete rubric, relevant submitted evidence,

first-pass assessment, review decisions, and precise disputed question. The shared grading instructions (Section H.2) apply throughout.

## Decision procedure

1. Identify the dispute and affected requirements. Verify the attempt and evidence identity, the exact disputed criterion or closure condition, and its basis in the task. Retain the attempt’s validity classification unless validity itself is disputed. Identify unauthorized requirements or equivalence conflicts as rubric or fairness issues.

2. Examine the relevant evidence. Inspect the affected submitted figures, artifacts, and permissible reference material. Apply the shared attribution, equivalence, comparison, and high-result verification rules. Complete any missing mandatory check without rerunning calculations, repairing the submission, or supplying missing agent reasoning.

3. Resolve the affected decisions. Apply the shared atomic, applicability, necessarydependency, and RC rules only to decisions affected by the dispute. Preserve unrelated valid work and uncontested decisions. If an acceptable equivalent cannot be distinguished from a substantive error, retain the uncertainty rather than inventing a definite judgment.

4. Record the decision and rationale. State the dispute, whether it is resolved, submitted and reference evidence examined, applicable rule, scientific reasoning, changes from review, and resulting score or RC where determined. Explain any effect on fairness, attempt validity, or a justified request for a new execution.

The targeted reviewer integrates the adjudication with the first-pass and review assessments. Preserve the initial assessment and record changes explicitly; do not average judgments. Prior assessments identify the dispute, not the scientific answer. Other submissions and expected score distributions do not determine the decision.

## H.6 Grading Output Format

The first-pass grader produces a structured assessment recording the validity of the run, decisions on the twenty atomic criteria, checks of key scientific results, and Research Closure with its supporting rationale. The assessment also records relevant evidence, reasons for failed or unresolved decisions, and any issues requiring further review.

The Atomic Score is calculated from the accepted criterion-level decisions using the prescribed half-point and N/A rules. Runs with no assessable scientific work receive zero credit. Targeted review and, when necessary, adjudication document any changes to the initial assessment and the evidence supporting those changes.