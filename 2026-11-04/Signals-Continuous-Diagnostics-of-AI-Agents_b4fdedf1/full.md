# Signals: Continuous Diagnostics of AI Agents

Amey Rambatla Siddhant Loya

Vatsal Goel Anurag Mishra

Sagar Gaur Asif Mohammad

Cisco {ameyr,vatsalgl,gaur,siddhant,anumish,khasmoha}@cisco.com

## Abstract

AI agents can make incorrect decisions or waste resources even when their tools and infrastructure work normally. Their nondeterministic behavior makes these problems dificult to reproduce and diagnose. We developed Signals to identify such problems from agentic traces and suggest concrete remediations. We built the method around a taxonomy of 45 problem types across eight families, with definitions and evidence requirements for each type. The method distinguishes realized failures, risk exposures, and ineficiencies, groups related findings into engineering issues called Signals, and assesses their severity from the supporting evidence. Each Signal includes actions a developer can take and checks for evaluating the change. Further executions can add evidence to an existing Signal or reveal a new problem. We evaluated the method on public agent executions and achieved 91.43% F1 when matching Signals to reference issues, about 18 percentage points higher than a one-pass LLM control that used the same model and taxonomy.

Keywords: AI agents; diagnostic taxonomy; continuous diagnostics; errors and ineficiencies; remediations.

## 1 Introduction

An AI agent can use the wrong tool, overlook required information, repeat unnecessary work, or claim that a task succeeded when it did not. These failures need not produce an exception or a failed service request. The software can run as written while the agent fails to do what the user intended.

Diagnosing this behavior is dificult because an agent’s decisions are not fixed by its code alone. It may interpret information diferently or choose a diferent sequence of actions across runs of the same task. Benchmarks such as τ -bench studied reliability across repeated executions [1]. One successful run therefore provides only one observation, and a remediation needs to be checked beyond the execution that motivated it.

Agentic traces give developers a record of model interactions, retrieved information, tool calls, and intermediate results. They do not, by themselves, explain whether the agent made the right decision. A successful tool call can perform the wrong action, and several valid calls can form an unnecessary loop. To diagnose either case, a developer needs to compare the execution with the task, the available information, and the requirements governing the actions.

We developed Signals to perform this analysis continuously and connect the findings to work a developer can undertake. We started with a taxonomy of 45 problem types across eight families, drawing on research into agent failures, retrieval, tool use, and safety [1, 2, 3, 4, 5]. The definitions distinguish problems that can lead to similar outcomes but require diferent fixes. Invalid tool arguments, an invalid tool result, and an agent ignoring a valid result are three such cases.

We then used the taxonomy to connect individual findings to related issues, supporting evidence, severity, and remediation suggestions. The distinction between a finding and an issue matters: repeated instances of the same integration problem may need one fix, while two problems with the same taxonomy label may need diferent fixes. New executions can strengthen an existing finding, establish a separate issue, or provide evidence about behavior after a change. The taxonomy supplies a consistent vocabulary even as the models and analysis methods used to apply it change.

Our contributions were:

• A taxonomy of agent problems. We defined 45 problem types across eight families, with evidence requirements and boundaries between related types.

• A continuous diagnostic method. We connected findings from agent executions to Signals that describe the issues developers need to address.

• Evidence-based severity. We defined separate factors for realized failures, risk exposures, and ineficiencies, together with a fixed calculation and Signal-level aggregation.

• Concrete remediation suggestions. We connected each diagnosis to corrective actions and checks, using the available evidence to determine how specific the recommendation could be.

## 2 Related Work

We drew on agent benchmarks and failure-diagnosis research to understand which problems a diagnostic system needed to recognize. τ-bench evaluated interactions among an agent, a user, tools, and domain policies; its pass<sup>k</sup> metric measures success across repeated executions of the same task [1]. MAST studied multi-agent failures through annotated execution histories and a failure taxonomy [2]. AgentRx used a cross-domain taxonomy, constraints, and supporting evidence to diagnose failures within an execution [3]. These studies examined the decisions and interactions leading to an outcome, rather than relying only on whether the task succeeded. SWE-bench evaluated whether language models resolved software issues and provided recorded coding-agent trajectories, adding a repository-based setting for studying agent behavior [6].

Other work examined specific requirements that an agent’s output or actions must satisfy. RAGTruth used human annotations to study statements that were unsupported by, or contradicted, retrieved information [4]. AgentDojo evaluated prompt injection attacks and defenses under explicit task and attack conditions [5]. These settings distinguish an incorrect answer from an attempted attack or a successful defense. The execution record must be interpreted against the task and the requirement that applied to the action.

We used these distinctions when defining the problems in our taxonomy. MAST and AgentRx informed our treatment of coordination, verification, and tool-result interpretation. RAGTruth informed the boundary around unsupported claims, while τ -bench and AgentDojo supplied tool-use, policy, and trust-boundary examples. We then connected the problem definitions to the evidence needed for a diagnosis and to concrete remediation actions. Evaluating false findings requires independently reviewed executions with no supported problem; a benchmark’s unsuccessful-task label alone cannot supply that distinction.

## 3 Method

We built Signals to turn agent executions into diagnoses that developers could investigate and act on. We first used the taxonomy in Section 4 to identify problem occurrences and collect their execution evidence. We then grouped related occurrences into Signals, assessed their severity, and suggested remediations. Figure 1 shows this workflow.

We distinguished three units throughout this process. A session is one recorded agent execution. An occurrence is an individual problem instance. A Signal groups related occurrences into an engineering issue, with an explanation, supporting evidence, severity, and concrete remediation actions. A session can contain several problems, while the same issue can recur across sessions.

## 3.1 Finding problems in an execution

We examined each session against its task request and any supplied instructions or tool contracts. The available record could include model inputs and outputs, retrieved information, tool arguments and responses, and the final result. We interpreted events in that sequence: a rejected call followed by recovery is diferent from the same rejection followed by task failure. Repeated retrieval can supply needed evidence or repeat work that adds no benefit.

For each finding, the diagnosis identifies the afected component, ranks candidate problem types, explains what happened and where, and links to the records that support it. An error violates an applicable requirement or prevents the task from being completed correctly. An ineficiency adds avoidable cost or latency without a corresponding quality, safety, or reliability benefit. Neither successful service responses nor repeated steps establish either diagnosis on their own. The task, surrounding execution, and applicable requirements determine whether a problem occurred.

We also separated the problem type from its observed outcome. A realized defect has an observed adverse consequence, incorrect result, or failure. A risk exposure has evidence of a reachable harmful path whose harm was not observed. An ineficiency leaves the outcome acceptable but adds work without a supported benefit. The same type can support diferent outcomes when the evidence warrants them. Missing telemetry, attack-like input, or a long execution is not suficient evidence of a failure or waste.

## 3.2 Generating and updating Signals

We grouped related findings around the issue a developer would need to fix. Repeated instances of an argumentconstruction error, for example, can support one Signal with several cited occurrences. A diferent problem requiring a separate intervention remains a separate Signal. Sharing a taxonomy type alone does not establish a common root cause or remediation.

Further executions can add evidence to an existing issue or establish a new one. When warranted, Signals updates the afected sessions, evidence, severity, and proposed actions. After a remediation, new executions can also provide evidence about recurrence and preserved behavior. One corrected execution, or no findings during an irrelevant period, does not establish that a fix worked. A change still needs independent checks on afected cases and cases that previously succeeded.

## 3.3 Assessing severity

We treated severity separately from problem type and diagnostic confidence. It describes the consequence, reachable exposure, or avoidable waste supported by the evidence. Teams may use that assessment alongside ownership, release plans, and critical workflows when setting remediation priorities.

For a realized defect, the relevant factor is the consequence that occurred. For a risk exposure, it is the credible potential consequence and the reachability of the harmful path. For an ineficiency, it is avoidable work rather than total resource use. Table 1 gives the consequence and waste values. The evidence must support the selected factor: task failure does not establish financial loss, and extra calls do not establish waste when they provide necessary verification or recovery.

![](images/455ea8fde130004eafb6ecc1935d03bc39d5f87f42b5de424c84d8ab8f5ecff6.jpg)  
Figure 1: The Signals diagnostic workflow. We used the taxonomy to diagnose problems in execution records, assess their severity, and suggest remediation actions. Further executions can add evidence to existing Signals or reveal new issues.

Let C denote observed consequence, $C _ { p }$ potential consequence, W avoidable waste, and A afected-scope points. For risk, the reachability multiplier R is 0.4 for privileged access with several uncommon preconditions, 0.6 for authentication or active user interaction, 0.8 for ordinary application behavior, and 1.0 for a public or reliably automatable path. The scores are

$$
s _ { d } = \operatorname* { m i n } ( 1 0 0 , C + A ) ,\tag{1}
$$

$$
s _ { r } = \operatorname* { m i n } ( 1 0 0 , \lceil C _ { p } R \rceil + A ) ,\tag{2}
$$

$$
s _ { i } = \operatorname* { m i n } ( 1 0 0 , W + A ) .\tag{3}
$$

The ceiling rounds the adjusted potential consequence up to the next integer. An unreachable hypothetical path does not establish a risk occurrence; when harm has occurred, that occurrence is assessed as a realized defect.

Afected scope counts distinct afected sessions for each nature. Repeated spans or observations in one session do not increase the count. Table 2 gives the count and percentage bands. The percentage applies only when at least 20 sessions were analyzed, and the higher of the count and percentage bands determines A. Its denominator is the set actually analyzed for that assessment. The calculation records the counts and population alongside the score so that later assessments can be interpreted against their evidence.

A Signal has a score for each nature represented by its occurrences. Its overall severity is $S = \operatorname* { m a x } ( s _ { d } , s _ { r } , s _ { i } )$ using only those represented natures. Summing would mix diferent meanings and depend on the number of natures recorded; averaging could hide a rare severe defect. The display levels are Low for 1–39, High for 40–79, and Critical for 80–100. The largest consequence or waste value is 85 points and the largest scope value is 15, so the current tables cannot exceed 100; the explicit cap guards against later table changes.

The model assesses the semantic factors from the evidence. Code derives scope, looks up the fixed values, performs the arithmetic, and determines the display level.

The explanation identifies the evidence for each factor and the nature that determines the overall score.

For illustration, a task failure afecting five of 30 analyzed sessions contributes 40 consequence points. Both the count of five and the 16.7% afected share select the repeated band, adding seven scope points. The score is $4 0 + 7 = 4 7$ , with a High display level. This is an arithmetic example, separate from the case in Section 3.5. New evidence can change scope or establish a diferent consequence, but a higher count alone does not justify a more severe consequence category.

## 3.4 Suggesting remediations

A remediation specifies an action a developer can take and how to check its efect. Depending on the evidence, the action may concern instructions, context, tool integration, or code. We kept recommendations specific to the available evidence: a tool contract and invalid invocation can establish how an argument should change, but locating the exact code requires implementation context.

The action may prevent the problem or help detect it. Correcting argument construction can prevent an invalid call; adding an evaluator can identify similar calls later. The recommendation needs to make that intended efect clear. Validation then checks that the change addresses the problem while preserving correct behavior. Fewer calls or lower cost count as an improvement only if the agent still completes the task correctly. Signals supplies the diagnosis and concrete actions; developers implement and validate the changes.

## 3.5 Insurance-intake example

In an audited synthetic insurance-intake task, a false conflict led to human review instead of case registration. The agent read a submission and supporting document, compared their policy references, and decided whether to register a case or send it for human review. The submission contained POL123; extraction returned Policy POL123. The second value added a field label to the same identifier. The workflow nevertheless reported a conflict, routed the request to human review, and did not register a case in that attempt.

<table><tr><td>Points</td><td>Consequence for a defect or credible exposure</td><td>Avoidable waste for an inefficiency</td></tr><tr><td>10</td><td>Small, localized adverse effect.</td><td>Small avoidable work bounded to one operation.</td></tr><tr><td>25</td><td>Material but reversible degradation.</td><td>Meaningful but bounded avoidable work.</td></tr><tr><td>40</td><td>Task failure or a materially wrong result.</td><td>Substantial or repeated avoidable work.</td></tr><tr><td>65</td><td>Major supported harm in a high-consequence domain.</td><td>Waste dominates cost or latency, or persists across stages or turns.</td></tr><tr><td>85</td><td>Severe, permanent, or irreversible harm.</td><td>Work continues until intervention or a hard limit and threatens viability.</td></tr></table>

Table 1: Severity factor values. Consequence and waste have separate definitions even when their point values coincide. Potential consequence applies only to an evidenced reachable path.
<table><tr><td>Scope band</td><td>Distinct affected sessions</td><td>Affected share (20+ analyzed sessions)</td><td>Points</td></tr><tr><td>Isolated</td><td>1</td><td>Less than 1%</td><td>0</td></tr><tr><td>Limited</td><td>2-3</td><td>1% to less than 10%</td><td>3</td></tr><tr><td>Repeated</td><td>4-10</td><td>10% to less than 30%</td><td>7</td></tr><tr><td>Broad</td><td>11-100</td><td>30% to less than 60%</td><td>12</td></tr><tr><td>Widespread</td><td>More than 100</td><td>At least 60%</td><td>15</td></tr></table>

Table 2: Afected-scope points. Use the higher band from the count and, when at least 20 sessions were analyzed, the percentage columns.

<table><tr><td colspan="2">Taxonomy prob- Evidence supporting the diagnosis lem</td></tr><tr><td>error</td><td>Factual or logical POL123 and Policy POL123 identify the same policy. The added field label did not support the reported policy_reference</td></tr><tr><td colspan="2">conflict.</td></tr></table>

Table 3: The taxonomy diagnosis in the synthetic insuranceintake case. Generic policy identifiers replace the source values.

Signals reported “Unsupported conflict flags” and “False conflict detection”. Table 3 connects these findings to the taxonomy and their evidence. The underlying Factual or logical error was the comparison of equivalent identifiers as diferent. Human-review routing and missing registration showed its efect; they did not independently establish additional defects or financial harm. Human review remains appropriate for a genuine conflict.

We proposed comparing normalized policy identifiers while preserving real diferences. Checks should cover equivalent forms (POL123 and Policy POL123), diferent references (POL123 and POL999), and missing references. Diferent or missing values should still follow the applicable review policy. The audit did not include an implemented repair. Appendix B gives a simpler payment-tool illustration.

## 4 Taxonomy for Agent Diagnostics

We built the taxonomy to give the algorithm a consistent definition of the problems it should find. A task outcome alone is insuficient: two failed tasks can require diferent fixes, and a completed task can still contain a risk exposure or unnecessary work. The catalog defines 45 problem types across eight families, summarized in Table 4. Appendix A gives every definition. The families organize the catalog; they do not establish a problem’s root cause. Shared definitions also give reviewers a basis for comparing findings across tasks, agent designs, and diagnostic methods.

## 4.1 How the taxonomy was built

To decide which behaviors the catalog needed to distinguish, we reviewed research on instruction following and priority [7, 8, 9, 10]; agent behavior, coordination, and stateful tool use [2, 11, 3, 12, 13, 1, 14, 15]; retrieval, answer support, and citations [16, 4, 17, 18, 19, 20]; and safety, security, refusal, and bias [21, 5, 22, 23, 24, 25, 26, 27]. We also used telemetry and reliability guidance to identify runtime evidence [28, 29, 30, 31, 32], and cost–quality studies to inform our treatment of avoidable resource use [33, 34].

We consolidated these sources by asking what evidence would establish each problem and what action would address it. For each type, we recorded a definition, required evidence, examples, exclusions, and boundaries with neighboring types. We separated a proposed cause from the observed behavior and its consequence. We split similar symptoms when their evidence requirements difered, and merged overlapping descriptions when they described the same observable violation. We did not adopt benchmark categories unchanged.

The resulting catalog includes runtime and platform failures, safety boundaries, and avoidable resource use alongside failures in the agent’s decisions. Each family therefore requires evidence appropriate to the problem: a provider outage, an unauthorized action, and a redundant retrieval cannot be established in the same way. Source frequency does not establish production prevalence. We disclosed the public benchmark families used in this research and did not treat them as unseen evidence. The catalog can be extended as new agent architectures, tools, and use cases produce problems that the current definitions do not capture.

<table><tr><td>Diagnostic family</td><td></td><td>Types Diagnostic focus</td></tr><tr><td>Instruction and specification</td><td>5</td><td>Following the accepted objective, explicit requirements, and permitted scope.</td></tr><tr><td>Knowledge, context, and retrieval</td><td>6</td><td>Obtaining, retaining, and citing the evidence needed for the task.</td></tr><tr><td>Reasoning, decision, and output</td><td></td><td>Producing supported, correct, complete results and truthful completion claims.</td></tr><tr><td>Tool, action, and integration</td><td>8</td><td>Choosing and calling tools, interpreting their results, and respecting intended effects.</td></tr><tr><td>Orchestration, coordination, and state</td><td>5</td><td>Routing work, transferring information and state, making progress, and stop- ping correctly.</td></tr><tr><td>Runtime and platform reliability</td><td>3</td><td>Meeting execution requirements despite provider, capacity, authentication, or configuration failures.</td></tr><tr><td>Safety, security, privacy, and policy</td><td>6</td><td>Respecting trust, authorization, confidentiality, and safe output boundaries.</td></tr><tr><td>Efficiency and resource use</td><td>4</td><td>Avoiding unnecessary actions, retries, retrievals, and context while preserving acceptable outcomes.</td></tr></table>

Table 4: The 45 problem types and their eight diagnostic families.

## 4.2 Evidence and diagnostic boundaries

We defined types around observable requirements so that a diagnosis could be checked against the execution. Invalid tool arguments concern the invocation; an invalid tool result concerns the tool’s response; and ignoring a valid result concerns the agent’s subsequent decision. Similar task outcomes can therefore require diferent evidence and remediation actions.

A diagnosis must establish both the applicable requirement and the behavior that violated it. For example, invalid arguments require the tool contract and the submitted arguments. A false success claim requires an explicit completion claim contradicted by execution records or authoritative state. Missing telemetry alone does not prove that an action never happened, and private or missing reasoning cannot establish a mismatch between a stated decision and an action.

Problem type remains separate from the outcome distinction used in Section 3. A type can describe a realized defect, a supported risk exposure, or an ineficiency when the evidence warrants it. Attack-like text alone does not establish a security failure, and repeated work may be justified by verification, recovery, pagination, or freshness checks. We also distinguished no supported problem from insuficient evidence and from a candidate problem outside the catalog. Several descriptions of one violation do not make it several events; independently supported problems can coexist in the same execution.

## 5 Dataset

We assembled a reference dataset from recorded agent executions in public benchmarks. The sources were AgentProcessBench [35], SWE-rebench / OpenHands [36], TraceElephant [37], MAD / MAST [2], the original τ -bench [1], and recorded SWE-bench trajectories [6]. We also included AgentDojo [5], Who&When [15], TRAIL [38], RootSE [39], and AgentRx [3]. Together, these sources covered coding, tool use, coordination, and security evaluation.

For each session, the detector received the available task, instructions, tool information, actions, and observations. We excluded native benchmark outcomes and reference labels from its inputs. This kept the reference labels separate from the execution evidence used for detection.

LLM judges created and reviewed the reference labels under human oversight. We organized the resulting reference occurrences into reference Signals within evaluation batches so that we could evaluate individual findings and the issues formed from them. Each reference occurrence belonged to exactly one reference Signal; the same issue could appear in several batches.

## 6 Experiments and Results

To test how accurately our algorithm identified errors and ineficiencies, we evaluated its findings against the reference dataset in Section 5. We first measured agreement on individual occurrences and on the Signals formed from them. We then compared the full workflow with a one-pass control to test whether the workflow added value beyond supplying the taxonomy to the model. Separate session-level checks examined detection and agreement across repeated runs.

## 6.1 Evaluation setup

The Signals detector used GPT-6.1 Sol. For all occurrence results, we used the same frozen reference dataset and fixed predictions. Code matched predictions to references and calculated the metrics; we did not call a model judge for either step.

We allowed a predicted occurrence to match a reference occurrence only when they belonged to the same session, had the same known component type and normalized component name, and shared at least one problem-type ID across their three ranked candidates. Name normalization accounted for case, camel-case boundaries, Unicode form, and separators. We then computed a deterministic maximum one-to-one matching, giving each prediction and reference occurrence credit at most once. Unmatched predictions and references remained in the metric denominators. Descriptions, evidence-span overlap, severity, and remediation did not enter the matcher.

At Signal level, we used two checks. First, we measured reference coverage: the fraction of reference Signals with at least one matched occurrence, and the fraction with at least half their occurrences matched. These matches could be spread across several predicted Signals. Second, we matched predicted and reference Signals within the same evaluation batch, requiring the same component type, normalized component name, and at least one shared problem-type ID. A maximum one-to-one matching again gave each Signal credit at most once. This comparison used the component and problem-type tags, without using occurrence IDs, counts, or membership overlap.

These rules measured agreement on the component and problem type. They did not verify the wording of a diagnosis, the cited evidence, severity calibration, or the correctness of a proposed fix. The matching rule could also connect diferent events on the same component, including when only a lower-ranked problem-type alternative matched.

## 6.2 Detection accuracy

Occurrence matching gave 90.01% precision, 90.98% recall, and 90.49% F1. An unmatched prediction could reflect a component or type disagreement, or an extra repeated report; it was not automatically a factually false description.

The matched occurrences covered 92.86% of reference Signals with at least one match and 92.16% with at least half their occurrences matched. The separate Signal comparison gave 92.90% precision, 90.01% recall, and 91.43% F1. Coverage depended on occurrence matching, while this second comparison required agreement on the component and problem-type tags without an occurrenceoverlap requirement.

We also measured detection at the session level, obtaining 97.26% precision and 97.47% recall. This check required only a prediction somewhere in the run, rather than recovery of the correct individual problems.

## 6.3 Comparison with direct Signal generation

To separate the efect of the workflow from that of the model and taxonomy, we compared Signals with a onepass direct-generation control. The control used the same model and complete 45-type taxonomy and generated complete Signals directly from batches of complete sessions. We scored both systems against the same reference Signals, using the same evaluation cohort and deterministic Signal matcher. Table 5 reports the comparison.

<table><tr><td>System</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>One-pass direct generation</td><td></td><td>71.11% 76.46% 73.69%</td><td></td></tr><tr><td>Signals pipeline</td><td></td><td>92.90% 90.01% 91.43%</td><td></td></tr></table>

Table 5: Signal matching with the same model, taxonomy, evaluation cohort, and deterministic matcher.

The Signals pipeline improved precision by 21.79 percentage points, recall by 13.56 points, and F1 by 17.75 points. The larger precision improvement indicated that the workflow substantially reduced unmatched Signal predictions. In this comparison, the workflow contributed beyond giving the taxonomy to the model in a single generation step. The results came from selected benchmarks; correlated tasks and exposure to benchmark families during taxonomy development constrain their generalization.

## 6.4 Robustness and recurrence

In separate fixed robustness runs, we measured agreement on whether a session contained a finding across repeated runs. Agreement was 97.0–98.5%; removing some telemetry reduced it to 74.2%. These were binary session-presence comparisons, separate from the occurrence-matching scores above.

We also examined exploratory lifecycle histories involving returning issues and periods without recurrence. We did not implement and test the proposed remediations, so these histories did not establish whether a fix worked. The runs also did not establish production cost savings or a reduction in diagnosis time.

## 7 Conclusion

We presented Signals as a way to connect a 45-type diagnostic taxonomy to execution evidence, severity assessment, and concrete remediation suggestions. The taxonomy gave us definitions for the problems to identify; the diagnostic workflow connected individual findings to issues a developer could investigate and fix.

On the reference dataset, the Signals pipeline achieved 91.43% Signal-matching F1, compared with 73.69% for a one-pass control using the same model and taxonomy. This comparison showed that the workflow added value beyond direct taxonomy-guided generation. The scores measured agreement on the component and problem type. They did not establish the accuracy of severity assessments or the efectiveness of proposed remediations, which require separate validation.

## References

[1] S. Yao, N. Shinn, P. Razavi, and K. Narasimhan. τ - bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045, 2024. Historical trajectories in the accompanying repository. https://arxi v.org/abs/2406.12045. https://github.com/sierra-researc h/tau-bench/tree/main/historical\_trajectories.

[2] M. Cemri et al. Why Do Multi-Agent LLM Systems Fail? arXiv:2503.13657, 2025. MAST definitions and MAD / MAST-Data release. https://arxiv.org/abs/2503.13657. https://github.com/multi-agent-systems-failure-t axonomy/MAST. https://github.com/multi-agent -systems-failure-taxonomy/MAST/blob/main/tax onomy\_definitions\_examples/definitions.txt. https: //huggingface.co/datasets/mcemri/MAD.

[3] S. Barke, A. Goyal, A. Khare, A. Singh, S. Nath, and C. Bansal. AgentRx: Diagnosing AI Agent Failures from Execution Trajectories. arXiv:2602.02475, 2026. Accompanying benchmark release. https://arxiv.org/abs/2602.02475. https://github.com/microsoft/AgentRx. https: //huggingface.co/datasets/microsoft/AgentRx.

[4] C. Niu et al. RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models. arXiv:2401.00396v2, 2024. Accompanying corpus and documentation. https://arxiv.org/abs/2401.00396v2. https://github.com/ParticleMedia/RAGTruth.

[5] E. Debenedetti et al. AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents. NeurIPS Datasets and Benchmarks, 2024. Recorded runtime traces and results. https: //arxiv.org/abs/2406.13352. https://github.com/ethz-s pylab/agentdojo. https://agentdojo.spylab.ai/results/.

[6] C. E. Jimenez et al. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR, 2024. Recorded trajectories from the oficial experiments repository. https://arxiv.org/abs/2310.06770. https: //github.com/SWE-bench/experiments.

[7] J. Zhou et al. Instruction-Following Evaluation for Large Language Models. arXiv:2311.07911, 2023. https://arxiv. org/abs/2311.07911.

[8] Y. Jiang et al. FollowBench: A Multi-level Fine-grained Constraints Following Benchmark for Large Language Models. ACL, 2024. Oficial research repository. https: //github.com/YJiangcm/FollowBench.

[9] Z. Zhang et al. IHEval: Evaluating Language Models on Following the Instruction Hierarchy. arXiv:2502.08745, 2025. https://arxiv.org/abs/2502.08745.

[10] C. Guo et al. IH-Challenge: A Training Dataset to Improve Instruction Hierarchy on Frontier LLMs.

arXiv:2603.10521, 2026. https://arxiv.org/abs/2603 .10521.

[11] X. Ma et al. Demystifying the Lifecycle of Failures in Platform-Orchestrated Agentic Workflows (AgentFail). arXiv:2509.23735, 2025. https://arxiv.org/abs/2509.237 35.

[12] F. Yan et al. Berkeley Function Calling Leaderboard (BFCL). Oficial benchmark and methodology resources, accessed 27 October 2026. https://gorilla.cs.berkeley.ed u/leaderboard.html.

[13] Jiarui Lu, Thomas Holleis, Yizhe Zhang, Bernhard Aumayer, Feng Nan, Felix Bai, Shuang Ma, Shen Ma, Mengyu Li, Guoli Yin, Zirui Wang, and Ruoming Pang. ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities. arXiv:2408.04682, 2024. https://arxiv.org/abs/2408.046 82. https://github.com/apple/ToolSandbox.

[14] H. Trivedi et al. AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents. Oficial research repository, ACL 2024. https: //github.com/StonyBrookNLP/appworld.

[15] S. Zhang et al. Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems (Who&When). ICML, 2025. Official research repository and dataset resources. https: //github.com/ag2ai/Agents\_Failure\_Attribution.

[16] Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang, Peng Shi, Shuaichen Chang, Cheng Jiayang, Cunxiang Wang, Shichao Sun, Huanyu Li, Zizhao Zhang, Binjie Wang, Jiarong Jiang, Tong He, Zhiguo Wang, Pengfei Liu, Yue Zhang, and Zheng Zhang. RAGChecker: A Finegrained Framework for Diagnosing Retrieval-Augmented Generation. arXiv:2408.08067, 2024. https://arxiv.org/ abs/2408.08067. https://github.com/amazon-science/R AGChecker.

[17] J. Chen, H. Lin, X. Han, and L. Sun. Benchmarking Large Language Models in Retrieval-Augmented Generation. arXiv:2309.01431, 2023. Retrieval-Augmented Generation Benchmark (RGB). https://arxiv.org/abs/2309.01431.

[18] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert. RAGAs: Automated Evaluation of Retrieval Augmented Generation. Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, 2024. https://acla nthology.org/2024.eacl-demo.16/. https://github.com/v ibrantlabsai/ragas/blob/main/docs/concepts/metrics/a vailable\_metrics/faithfulness.md.

[19] Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, Sajal Choudhary, Rongze Daniel Gui, Ziran Will Jiang, Ziyu Jiang, Lingkun Kong, Brian Moran, Jiaqi Wang, Yifan Ethan Xu, An Yan, Chenyu Yang, Eting Yuan, Hanwen Zha, Nan Tang, Lei Chen, Nicolas Schefer, Yue Liu, Nirav Shah, Rakesh Wanga, Anuj Kumar, Wen-tau Yih, and Xin Luna Dong. CRAG: Comprehensive RAG Benchmark. Advances in Neural

Information Processing Systems, Datasets and Benchmarks Track, 2024. https://arxiv.org/abs/2406.04744. https://github.com/facebookresearch/CRAG/.

[20] T. Gao, H. Yen, J. Yu, and D. Chen. Enabling Large Language Models to Generate Text with Citations. EMNLP, 2023. Oficial ALCE repository. https://github.com/pri nceton-nlp/ALCE.

[21] Y. Ruan et al. Identifying the Risks of LM Agents with an LM-Emulated Sandbox (ToolEmu). Oficial research repository, ICLR 2024. https://github.com/ryoungj/Too lEmu.

[22] OWASP GenAI Security Project. Top 10 for Large Language Model Applications, 2025 edition. Security risk and mitigation catalog. https://genai.owasp.org/llm-top-10/.

[23] National Institute of Standards and Technology. Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations. NIST AI 100-2 E2025, 2025. https://csrc.nist.gov/pubs/ai/100/2/e2025/final.

[24] MITRE. Adversarial Threat Landscape for Artificial-Intelligence Systems (ATLAS). Living threat knowledge base, accessed 27 October 2026. https://atlas.mitre.org/.

[25] P. Röttger et al. XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models. NAACL, 2024. https://aclanthology.org/2024.naacl-long. 301/.

[26] Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, and Dan Hendrycks. HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal. arXiv:2402.04249, 2024. https://arxiv.org/abs/ 2402.04249. https://github.com/centerforaisafety/Har mBench.

[27] A. Parrish, A. Chen, N. Nangia, V. Padmakumar, J. Phang, J. Thompson, P. M. Htut, and S. R. Bowman. BBQ: Bias Benchmark for Question Answering. Findings of the Association for Computational Linguistics: ACL 2022, pp. 2086–2105. https://aclanthology.org/2022.findi ngs-acl.165/. https://github.com/nyu-mll/BBQ.

[28] OpenTelemetry. Generative AI semantic conventions: metrics and agent spans. Living specifications, accessed 27 October 2026. https://github.com/open-telemetry/s emantic-conventions/blob/main/docs/gen-ai/gen-ai-m etrics.md. https://github.com/open-telemetry/semanti c-conventions-genai/blob/main/docs/gen-ai/gen-ai-age nt-spans.md.

[29] OpenTelemetry. Collector troubleshooting. Oficial telemetry receipt, export, and data-loss guidance, accessed 27 October 2026. https://opentelemetry.io/docs/ collector/troubleshooting/.

[30] World Wide Web Consortium. Trace Context. W3C Recommendation, 23 November 2021. w3.org/TR/2021/R EC-trace-context-1-20211123.

[31] gRPC project. Status Codes. Oficial runtime and integration error definitions, accessed 27 October 2026. https://grpc.io/docs/guides/status-codes/.

[32] M. Ulrich. Addressing Cascading Failures. In Site Reliability Engineering, chapter 22. Google, 2016. https: //sre.google/sre-book/addressing-cascading-failures/.

[33] L. Chen, M. Zaharia, and J. Zou. FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. Transactions on Machine Learning Research, 2024. https://arxiv.org/abs/2305.05176. https://github.com/stanford-futuredata/FrugalGPT.

[34] Q. J. Hu et al. RouterBench: A Benchmark for Multi-LLM Routing System. arXiv:2403.12031, 2024. https: //arxiv.org/abs/2403.12031.

[35] S. Fan et al. AgentProcessBench: Diagnosing Step-Level Process Quality in Tool-Using Agents. arXiv:2603.14465, 2026. Accompanying dataset release. https://arxiv.org/ abs/2603.14465. https://huggingface.co/datasets/Lula Cola/AgentProcessBench.

[36] I. Badertdinov et al. SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents. arXiv:2505.20411, 2025. Accompanying OpenHands trajectories dataset. https://arxiv.org/abs/2505.20411 https://huggingface.co /datasets/nebius/SWE-rebench-openhands-trajectories.

[37] M. Chen et al. Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems (TraceElephant). arXiv:2604.22708, 2026. Accompanying execution-trace dataset. https://arxiv.org/ abs/2604.22708 https://huggingface.co/datasets/TraceE lephant/TraceElephant.

[38] D. Deshpande, V. P. Gangal, H. Mehta, J. Krishnan, A. Kannappan, and R. Qian. Trace Reasoning and Agentic Issue Localization (TRAIL). arXiv:2505.08638, 2025. Accompanying dataset release. https://www.patronus.ai/re search/trail.

[39] M. Wang et al. TrajAudit: Automated Failure Diagnosis for Agentic Coding Systems. arXiv:2605.26563, 2026. Introduces the RootSE benchmark for repository-level coding-agent trajectories. https://arxiv.org/abs/2605.2 6563 https://huggingface.co/datasets/dengdan1999/Ro otSE.

## A Diagnostic Taxonomy Catalog

This catalog defines the 45 problem types introduced in Section 4. Each diagnosis requires the applicable task, policy, or interface requirement and supporting execution or configuration evidence. The definitions describe observable behavior; they do not establish hidden causes or intent. Realized defects, risk exposures, and ineficiencies are distinguished as described in the main text.

## A.1 Instruction and specification

• Constraint or instruction-priority violation. An action or output violates an applicable explicit constraint, or follows a lower-priority instruction over a conflicting higher-priority instruction. The governing instructions and violating behavior must both be available.

• User intent misinterpreted. The agent adopts a materially diferent objective from the user’s request before execution. The request, accepted objective or plan, and resulting behavior must show the mismatch.

• Required clarification not requested. The agent acts or stops without asking for a required value that the user could supply and policy permits it to request. The missing value must explain an incorrect action or outcome.

• Inappropriate refusal. The agent refuses an allowed, supported request despite suficient information and no policy requiring refusal. Correct safety refusals, access denials, and user cancellations are excluded.

• Goal drift. After correctly accepting the task, the agent pursues a materially diferent objective or expands its scope without authorization. The initial objective, later change, and resulting actions must be visible; authorized user revisions are excluded.

## A.2 Knowledge, context, and retrieval

• Required evidence not retrieved. Retrieval omits source material required by the task even though it exists in the configured corpus and is retrievable. The corpus or relevance reference must establish what was missing.

• Irrelevant or noisy retrieval. Retrieved material is irrelevant to the task and degrades the result or adds avoidable processing cost. A large retrieved context alone does not establish this problem.

• Stale or false retrieved context. The retrieval path supplies content that is outdated or false for the request. A trusted reference or time and version evidence must establish the discrepancy; a risk exposure requires a proven reachable path.

• Required context or memory lost. Information available earlier is absent, truncated, or replaced when the agent later needs it. Earlier records, the later context, and the resulting behavior must show the loss mattered.

• Context window limit exceeded. The assembled input exceeds, or is proven able to exceed, the active model’s context limit before required completion. The model version, limit, and counted input must establish the condition.

• Citation source mismatch or unsupported citation. A cited source does not support its paired claim, or a required citation is missing. Assessment uses the claim, source content, and applicable citation requirement.

## A.3 Reasoning, decision, and output

Unsupported or invented claim. A material claim is not supported by the evidence available to the agent. Establishing that the claim is false is a separate question; incomplete records alone cannot establish lack of support.

• Factual or logical error. An output or decision contradicts established facts, a deterministic calculation, or a governing logical condition. The diagnosis requires a reference that establishes the contradiction.

• Incomplete or prematurely terminated result. The agent returns or stops before satisfying a required part of the accepted task, including a required final answer. Correctly scoped partial results, valid refusals, and user cancellations are excluded.

• Verification absent or incorrect. A required check is skipped, incomplete, or reports a pass inconsistent with a directly checkable condition. Optional review and checks that correctly block bad output are excluded.

• Output schema or format violation. An agent output violates the syntax, fields, or format required by its consumer. The output contract and raw output must establish the violation; tool arguments are a separate type.

• Action and stated decision mismatch. An executed action or delivered artifact contradicts the same component’s visible committed decision while the objective remains unchanged. Authorized plan revisions and missing private reasoning are excluded; a more specific tool or routing violation uses that type.

• False success claim. The agent claims an action or task succeeded, but execution records or authoritative state establish that it did not. Missing records alone do not prove non-completion, and the diagnosis does not infer deceptive intent.

• Success check subverted. An observed modification or check-specific shortcut makes a success check pass while an independently established task requirement remains unmet. Authorized test corrections and legitimate solutions are excluded; intent is not inferred.

## A.4 Tool, action, and integration

• Wrong tool selected. The agent invokes an unsuitable or impermissible tool when a suitable available tool or correct path without a tool exists. The task, available tools, and selection requirement must establish the error.

• Required tool call omitted. A required tool call is absent before the agent acts, answers, or stops, without selecting an impermissible tool instead. The call must be required under the accepted task and current state.

• Invalid tool arguments. A tool invocation violates its schema, required values, types, or domain preconditions. The tool contract, submitted arguments, and validation or rejection establish the problem.

• Tool interface contract mismatch. The advertised tool interface disagrees with the interface actually exposed across an integration boundary. Versioned definitions must establish the mismatch and a resulting failure or reachable invocation path.

• Tool result invalid. A tool reports success but produces a result that violates its structural or semantic contract. A reported failure is a diferent event; a risk exposure requires a proven reachable invalid result.

• Tool result misinterpreted or ignored. The agent receives a valid tool result but its next decision contradicts or fails to use material information from it. The result and subsequent behavior must establish the error.

• Tool execution error not handled. A tool error or invalid result is followed by a separate failure to recover, validate, degrade safely, or report as required. The error alone does not establish this additional problem.

• Unintended tool side efect. A tool changes state beyond or diferently from its authorized intended efect. The intended action, tool execution, and state before and after the call must establish the unexpected change.

## A.5 Orchestration, coordination, and state

• Invalid route, sequence, or decomposition. The workflow chooses an incorrect node, order, branch, or task decomposition for the accepted objective. The expected workflow and actual execution must show that the choice changed execution or outcome.

• Handof information lost or ignored. Information required at a handof is omitted, withheld, or received and ignored by the destination agent or node. Source state, transferred information, and downstream behavior must show the failure.

• Workflow state not propagated or reset. A workflow transition fails to preserve or update shared state needed by a later node. State snapshots and the expected transition must establish the missing, stale, or reset value.

• Workflow loop or deadlock. The workflow repeats a control state without useful progress or waits cyclically, preventing it from reaching completion. The repeated states or wait cycle and progress evidence must establish the problem.

• Termination condition missed. Execution continues without an authorized purpose after an applicable stopping condition has been reached. Necessary reporting, cleanup, verification, and recovery are excluded; a transient error alone is not a stopping condition.

## A.6 Runtime and platform reliability

• Provider or runtime timeout/unavailable. A model provider or workflow runtime exceeds its deadline or is unavailable, preventing execution from meeting its service requirement. A correctly handled timeout without a service impact is excluded.

• Rate or resource exhaustion. Quota, capacity, memory, concurrency, or another resource limit is exhausted or proven reachable under expected load, impairing the agent execution. Metrics, errors, or configuration and load evidence must establish the condition.

• Runtime authentication or configuration failure. The model provider or runtime cannot execute because required credentials, endpoints, deployments, or settings are invalid or missing. A risk exposure requires configuration evidence of a reachable failure before an outage.

## A.7 Safety, security, privacy, and policy

• Untrusted-instruction boundary failure. Untrusted content controls behavior across a required instruction boundary, or configuration proves a reachable takeover path under a stated threat model. Attack-like text alone does not establish the problem.

• Sensitive information disclosure. Sensitive information reaches an unauthorized destination or is unnecessarily placed in a reachable unsafe context. Data classification, authorization, and the actual or reachable information flow must establish the problem.

• Excessive or unauthorized agency. An agent takes or can take actions beyond the approved, authorized, or confirmation-bounded scope. Excessive permissions without an observed harmful action constitute a risk exposure when the reachable path is established.

• Unsafe or disallowed content produced. The agent produces content prohibited by the application’s explicit safety policy in the evaluated context. The policy and output must establish the violation; an ordinary factual error alone is insuficient.

• Policy or access-control bypass. A required authorization, policy, guardrail, or trust check is bypassed, or configuration establishes a reachable bypass. The applicable control, permissions, and action or path must support the diagnosis.

• Insecure model/tool output handling. Untrusted model or tool output reaches an interpreter, renderer, command, or privileged destination without required validation, encoding, or isolation. Evidence must establish an executed efect or reachable dangerous path.

## A.8 Eficiency and resource use

• Repeated or duplicate action. A successful, safe task repeats an equivalent completed action without adding needed evidence, state, safety, or recovery value.

Required verification, pagination, and other justified repetition are excluded.

• Retry thrash. A task succeeds or safely degrades but repeatedly retries a failing operation beyond a defensible retry policy without useful adaptation. Error history, timing, policy, and the acceptable outcome must establish avoidable work.

• Redundant retrieval. A successful, safe task retrieves equivalent evidence already available without a justified freshness, coverage, or verification need. The retrieval sequence and available context must show no material evidence gain.

• Avoidable context bloat. A successful, safe task processes irrelevant or duplicate context that causes avoidable cost or latency. A smaller context that preserves quality, or equivalent evidence, must establish the waste; size alone is insuficient.

## B Payment Tool Illustration

We considered two synthetic sessions requesting a test payment of 100 cents. The supplied create\_payment contract requires amount\_cents. One agent instead sent amount; the tool rejected the call and the task failed. A control session used the required field and succeeded. Table 6 shows the records.

<table><tr><td>Record</td><td>Rejected session</td><td>Control session</td></tr><tr><td>Request</td><td>Payment of 100 cents</td><td>Payment of 100 cents</td></tr><tr><td>Arguments</td><td>amount: 100</td><td>amount_cents: 100</td></tr><tr><td>Tool result</td><td>Required field missing Payment created</td><td></td></tr><tr><td>Task result</td><td>Failed</td><td>Succeeded</td></tr></table>

Table 6: Synthetic executions using the same payment-tool contract.

The problem type was Invalid tool arguments, with a realized defect. The contract, arguments, and rejection response established the error. The remediation was to use amount\_cents while preserving the requested amount. Checks should cover a valid payment, a missing amount, and an invalid value, including verification that rejected requests create no payment. These sessions were handauthored illustrations, not measured detection or repair results.