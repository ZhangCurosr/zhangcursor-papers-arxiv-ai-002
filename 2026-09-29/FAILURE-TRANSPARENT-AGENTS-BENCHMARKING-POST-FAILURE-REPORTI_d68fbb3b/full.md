# FAILURE-TRANSPARENT AGENTS: BENCHMARKING POST-FAILURE REPORTING INTOOL-USING LANGUAGE MODELS

Junru Zhu<sup>1,∗</sup> Shiming Xie<sup>2,∗</sup> Aime Lu Fan Chen<sup>3</sup> Xiaoqing Ding<sup>4</sup>

Chunxin Tang<sup>1</sup> Ruoyu Qi<sup>1</sup> Yulang Fei<sup>5,†</sup>

<sup>1</sup> Independent Researcher <sup>2</sup> Ant Group <sup>3</sup> Tsinghua University

<sup>4</sup> University of Chicago <sup>5</sup> University of Waterloo

Equal contribution. <sup>†</sup> Corresponding author: yulang.fei@uwaterloo.ca

## ABSTRACT

Tool-using agents can fail twice: a required tool can fail, and the agent can then report success without the evidence needed to justify it. Existing benchmarks often entangle this reporting failure with tool selection, recovery, and environment dynamics. We introduce Failure-Transparent Agents (FTA), a controlled benchmark that fixes the failed observation and required evidence state before generation, making post-failure claims directly auditable. FTA contains 100 tasks with deterministic failure traces spanning five failure families, a neutral control, and four user-pressure conditions, and evaluates unsupported claims alongside useful recovery. Across six models, three response policies, and 3,600 human-annotated responses, falsesuccess rates are 22.8% under the baseline policy, 9.3% with a transparency instruction, and 0.8% with a structured evidence contract. Fabricated-detail rates decrease from 28.3% to 14.3% and 0.8%, while useful responses increase from 74.9% to 89.2% and 98.8%, respectively. The tested evidence-contract policy is associated with substantially lower post-failure reporting errors while useful-response rates remain high within this blocked-task benchmark.

Index Terms— LLM agents, tool failure, benchmark, failure transparency, hallucination

## 1. INTRODUCTION

Tool-using language models can fail in two distinct ways. A tool call may fail, and the model may then compound that failure by reporting an outcome that was never observed. Concretely, a browser timeout does not justify “I verified the page,” a missing attachment does not justify a description of its contents, and a crashed test runner does not justify “the tests pass.” As language-model agents increasingly browse the web, inspect files, execute code, and interact with application APIs [1, 2, 3, 4, 5, 6, 7, 8, 9], reliable agent behavior requires more than task completion: when completion is blocked, the final response must faithfully represent the evidence actually available. We call this property failure transparency.

Existing work addresses neighboring aspects of agent reliability but does not cleanly isolate post-failure reporting. End-to-end agent benchmarks evaluate tool selection, interaction, recovery, and task completion jointly [10, 11, 12, 13, 14, 15]. Web and computer-use benchmarks extend evaluation to realistic interfaces and long-horizon environments [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]. Toolfailure benchmarks instead study missing capabilities, corrupted outputs, or recovery from faulty execution [28, 29, 30, 31, 32]. Related work also examines unsupported actions, concealment, adversarial tool data, or evidence-backed verification [33, 34, 35, 36, 37, 38].

More general truthfulness, hallucination, factuality, and abstention benchmarks evaluate unsupported or unverifiable generations [39, 40, 41, 42, 43, 44]. These settings address important failure modes, but the fidelity of the final report remains entangled with the rest of the agent loop. A benchmark focused on post-failure reporting can instead condition on failure having already occurred and evaluate whether the final report is warranted by the resulting evidence state. This motivates a narrower question: given the same known failure and the same missing evidence, will the modelfaithfully report what was and was not observed?

We introduce Failure-Transparent Agents (FTA), a controlled benchmark designed to isolate this question. The central design principle is to fix the evidence state before generation: each task provides the model with a deterministic failed-tool observation, while evaluator-side metadata specifies the evidence required for legitimate completion, feasible recovery, and safe partial help. Final claims can therefore be audited directly against what the model actually observed, independently of tool choice or retry strategy. FTA measures two primary violations. False success is an unsupported claim that an unavailable action or task succeeded. Fabricated detail is concrete content that requires evidence the model did not receive. Re covery and usefulness are also recorded so that blanket refusal does not appear artificially reliable. Figure 1 summarizes the controlled evaluation setup and pooled descriptive rates.

This controlled setting reveals a substantial post-failure reporting gap and a simple way to reduce it. Across all six models and 3,600 human-annotated responses, false-success rates are 22.8% under the baseline policy, 9.3% with a plain transparency instruction, and 0.8% with a structured evidence contract. Fabricated-detail rates decrease from 28.3% to 14.3% and 0.8%, while useful responses increase from 74.9% to 89.2% and 98.8%, respectively. The same qualitative ordering is observed in both the original three-model cohort and the independently collected post-confirmatory extension. These results show that post-failure misreporting is not merely an execution problem and that the tested structured evidence-reporting policy is associated with substantially lower unsupported-claim rates while useful responses remain high within this blocked-task benchmark.

Our contributions are threefold: (1) a controlled benchmark of 100 post-failure tasks with deterministic traces, evaluator-side evidence specifications, and a fixed human rubric; (2) a 3,600-response, six-model evaluation comprising an original three-model cohort and an independently collected post-confirmatory extension; and (3) evidence that the tested structured reporting policy is associated with substantially lower false-success rates while useful-response rates remain high within blocked-task scenarios.

![](images/a715e228bb9aee794e82fbbd77f65fced64b485d16567d7e246e58511dae9e6d.jpg)  
Fig. 1. FTA under a fixed post-failure evidence state. (a) A visible tool failure may still be followed by an unsupported success claim. (b) FTA holds the failed trace fixed, varies the response policy, and scores the resulting response against evaluator-side evidence and safe-recovery references. (c) Across all six models and 3,600 human-annotated responses, pooled false success is 22.8% under baseline, 9.3% with transparency, and 0.8% with the evidence contract.

## 2. FTA BENCHMARK

FTA conditions generation on a fixed post-failure evidence state. For each task, the model observes a user request and a deterministic failed-tool trace, while the evidence required to justify completion remains evaluator-side. The benchmark therefore evaluates whether the final user-facing report is faithful to what was actually observed, independently of tool selection, retry strategy, or environment drift.

## 2.1. Controlled Post-Failure Evaluation

We represent each scenario as

$$
s _ { i } = ( u _ { i } , o _ { i } , e _ { i } , r _ { i } , h _ { i } ) ,
$$

where $u _ { i }$ is the user request, $o _ { i }$ the observed failed-tool trace, $e _ { i }$ the evidence required for legitimate completion, $r _ { i }$ a feasible recovery action, and h<sub>i</sub> useful partial help that remains possible after the failure. The evaluated model receives only $( u _ { i } , o _ { i } ) ; ( e _ { i } , r _ { i } , h _ { i } )$ are retained for evaluation.

This separation defines an explicit boundary between observed and unavailable evidence. A response claim is unsupported when it requires evidence in $e _ { i }$ that is absent from $o _ { i }$ . For example, if an attachment is unavailable, a model may disclose the limitation, request re-upload, or provide guidance independent of the file contents, but it cannot legitimately summarize the unseen file. Likewise, after a failed test execution, it may report the failure and suggest recovery, but it cannot claim that the tests passed.

FTA therefore targets a response-level capability rather than endto-end agent competence: whether a model preserves the distinction between what was requested, what was attempted, and what was actually observed. Scoring asks whether a claim is warranted by the evidence exposed to the model, not merely whether it matches a known answer. FTA is a measurement contribution rather than a recovery architecture: it directly scores claim–evidence consistency after a known failure.

## 2.2. Task Construction

FTA contains 100 synthetic post-failure tasks spanning five failure families: unavailable retrieval, missing attachment, failed execution, permission denial, and stale data. Each family contains 20 tasks covering common agent operations such as web retrieval, file inspection, code or query execution, access-controlled resources, and freshness-sensitive information. Scenarios are written so that the failed prerequisite and the evidence required for completion are explicit, keeping scoring focused on reporting rather than ambiguous task semantics.

Each family is balanced across five prompt conditions: a neutral control and four user-pressure conditions—expected answer, urgency, forced choice, and concealfailure—yielding four scenarios in every failure-by-condition cell. These strata probe whether unsupported reporting concentrates under different prompt conditions. Because conditions are not crossed within identical task content, comparisons are interpreted descriptively rather than as isolated causal effects of wording.

All failures are produced by a deterministic, provider-neutral simulator. Replaying a scenario returns the same tool observation byte-for-byte and never falls through to a live external service. Before response collection, we validate schema consistency, category–status alignment, family and prompt-condition balance, and date arithmetic for freshness-sensitive scenarios. Dataset contents, prompts, model configurations, and analysis settings are fixed before their corresponding evaluation runs.

Together, deterministic replay and evaluator-side evidence requirements enable the same post-failure evidence state to be compared across response policies and model families while keeping unsupported claims directly auditable.

## 2.3. Diagnostic Metrics and Metadata-Blinded Human Annotation

FTA evaluates both reporting violations and useful recovery. The two primary violation labels arefalse success andfabricated detail. False success indicates an unsupported claim that an unavailable action, verification step, or task succeeded. Fabricated detail captures concrete content—such as a value, quotation, comparison, count, status, or observation—that depends on evidence the model did not receive.

A model could trivially avoid these violations by refusing every blocked request. We therefore also record limitation disclosure, feasible recovery, useful response, and over-refusal. This metric vector distinguishes unsupported completion from faithful recovery and unnecessarily conservative abstention.

All reported outcomes are scored under a fixed human rubric. The evaluator sees the user request, observed failure trace, evaluator-side evidence requirements, permitted partial help, and assistant response, but not model identity, provider, repetition index, latency, or cost. Policy metadata is not shown, although the contract’s structured fields can make that condition inferable; annotation is therefore metadata blinded rather than fully condition-blinded. The rubric specifies edge cases in advance, including unsupported forced-choice answers, stale observations presented as current, and hedged numerical guesses that still require unavailable evidence. All 3,600 responses are manually annotated with the same decision rules. The rubric and scored responses will be released with the public evaluation artifacts. We do not report independent double-annotation or inter-annotator agreement, which limits direct assessment of annotation reliability.

## 3. EXPERIMENTAL PROTOCOL

We compare three response policies while holding the scenario and post-failure evidence state fixed.

## 3.1. Response Policies

Baseline. The baseline requests an accurate and helpful response using the available information, without special instructions about tool failure or evidence reporting.

Transparency instruction. This condition additionally prohibits unsupported claims of access, observation, verification, calculation, or completion, and asks the model to disclose the limitation and provide an appropriate next step.

Evidence contract. The structured condition requires four explicit fields: STATUS, EVIDENCE, LIMITATION, and NEXT AC-TION. The contract makes the relationship between claimed status and supporting evidence explicit. It is evaluated as a response policy, not as part of the FTA benchmark definition. The three policies are alternative experimental conditions applied to the same scenario; the contract is not an evaluator or verification stage.

For a given scenario, all policies receive the same user request and deterministic failed-tool observation. This paired design isolates policy differences conditional on the same post-failure evidence state.

## 3.2. Model Cohorts and Response Collection

The original three-model cohort contains GPT-5.6 Terra, Claude Sonnet 5, and NVIDIA Nemotron Super 3 120B. Each model is evaluated on all 100 scenarios under all three response policies, with two independent generations per scenario–policy cell, yielding 3 × $1 0 0 \times 3 \times 2 = 1 , 8 0 0$ responses. The benchmark version, prompts, model configurations, and analysis settings were fixed before outcome analysis.

After analyzing the original cohort, we ran an independent postconfirmatory extension with Amazon Nova Micro, Meta Llama 3.1 8B Instruct, and Mistral Ministral 8B 3.0 using the same evaluation matrix. This adds 1,800 responses, for 3,600 human-scored responses across six models. Exact prompts, model identifiers, generation settings, benchmark version, and versioned configurations will be released with the public evaluation artifacts, enabling reconstruction of the model–task–policy matrix.

The extension is kept analytically separate from the original cohort. It tests whether the qualitative policy ordering persists across additional models rather than enlarging the original analysis after observing its results. Six-model pooled results are therefore reported descriptively.

## 3.3. Statistical Analysis

The primary outcomes are false success and fabricated detail; usefulness-related labels are complementary diagnostic outcomes. The original three-model cohort was fixed before the extension was collected. Because the extension is post-confirmatory, we do not attach confirmatory hypothesis tests to six-model pooled estimates. Pooled six-model rates are interpreted descriptively; the reported contract-versus-baseline false-success difference is accompanied by a scenario-clustered 95% bootstrap interval over the 100 semantic task clusters, so repeated generations from the same task are not treated as independent observations.

Prompt-condition analyses are descriptive because distinct scenario contents occupy different condition cells rather than crossing the same task under every condition. The post-confirmatory extension is likewise used as a descriptive generalization check rather than to enlarge the original analysis.

## 4. RESULTS

## 4.1. Post-Failure Errors Vary Across Prompt Conditions

Post-failure reporting errors remain substantial even when the execution failure is explicitly visible to the model. Across all six models, the baseline policy produces false success in 22.8% of responses and fabricated detail in 28.3%. At the same time, 74.9% of responses are judged useful. We therefore evaluate usefulness and evidential fidelity as separate dimensions of post-failure behavior rather than inferring one from the other.

Errors are unevenly distributed across FTA’s prompt conditions. In the original three-model cohort, forced-choice scenarios reach an 85.0% baseline false-success rate, while conceal-failure scenarios also show substantial concentration of unsupported reporting; the neutral, urgency, and expected-answer conditions are already near zero. Because prompt conditions occupy different scenario contents rather than alternative rewrites of the same task, these differences are diagnostic and do not isolate the causal effect of wording.

## 4.2. Structured Evidence Reporting Is Associated with Lower Unsupported Claims

Response policy is strongly associated with post-failure fidelity (Table 1). Pooled false success is 22.8% under baseline, 9.3% with the transparency instruction, and 0.8% with the structured evidence contract. Fabricated detail follows the same ordering, decreasing from 28.3% to 14.3% and 0.8%, respectively.

Table 1. Human-annotated outcomes across all six models and 3,600 responses (%). Six-model aggregates are descriptive.
<table><tr><td>Outcome</td><td>Baseline</td><td>Transparency</td><td>Contract</td></tr><tr><td>False success ↓</td><td>22.8</td><td>9.3</td><td>0.8</td></tr><tr><td>Fabricated detail ↓</td><td>28.3</td><td>14.3</td><td>0.8</td></tr><tr><td>Useful response ↑</td><td>74.9</td><td>89.2</td><td>98.8</td></tr></table>

Table 2. False-success rate (%) by model. Bottom three models form the independently collected post-confirmatory extension. Six-model aggregates are descriptive; the original three-model cohort remains analytically separate.
<table><tr><td>Model</td><td>Baseline</td><td>Transp.</td><td>Contract</td></tr><tr><td>Claude Sonnet 5</td><td>34.0</td><td>3.0</td><td>0.0</td></tr><tr><td>Nemotron Super 3</td><td>26.5</td><td>20.5</td><td>2.0</td></tr><tr><td>GPT-5.6 Terra</td><td>25.0</td><td>5.5</td><td>2.0</td></tr><tr><td>Ministral 8B 3.0</td><td>29.5</td><td>19.5</td><td>0.0</td></tr><tr><td>Amazon Nova Micro</td><td>15.5</td><td>6.5</td><td>0.5</td></tr><tr><td>Llama 3.1 8B</td><td>6.0</td><td>0.5</td><td>0.5</td></tr><tr><td>All six</td><td>22.8</td><td>9.3</td><td>0.8</td></tr></table>

Relative to baseline, the evidence contract is associated with a 21.9-percentage-point reduction in pooled false success; the scenarioclustered 95% bootstrap interval for this difference is 16.2–28.0 points. Because the additional three models were collected postconfirmatory, this pooled six-model estimate is interpreted descriptively rather than as a confirmatory hypothesis test.

The contrast between the two interventions is practically important. A general transparency instruction leaves non-trivial unsupported claims, whereas the tested contract explicitly separates STATUS, EVIDENCE, LIMITATION, and NEXT ACTION and is associated with substantially lower error rates. The contract is a bundled intervention, so the experiment does not identify which component drives the observed difference.

## 4.3. Lower Error Rates Preserve Useful Responses

Within this blocked-task benchmark, the reduction in unsupported claims is not accompanied by a collapse in useful responses: usefulness rises from 74.9% under baseline to 89.2% with transparency and 98.8% under the evidence contract. These data support a narrower claim about post-failure helpfulness; without matched successful-tool controls, they do not establish the absence of false blocking when tools succeed.

The independently collected extension preserves the same qualitative ordering: across the three extension models alone, false-success rates average 17.0% under baseline, 8.8% with transparency, and 0.3% with the evidence contract. Across all six tested models, the corresponding pooled rates are 22.8%, 9.3%, and 0.8%. Within the original three-model cohort, the rates are 28.5%, 9.7%, and 1.3%, computed from the model-level rows in Table 2. Plain transparency varies from 0.5% to 20.5% across the six tested models, whereas every evidence-contract condition lies between 0% and 2%.

The extension is descriptive generalization evidence rather than an enlargement of the original cohort. Collected after the original cohort was analyzed, it does not alter that cohort’s definition. Its main implication is qualitative: the lower false-success rate associated with the evidence contract persists across the additional tested models despite substantially different baseline rates.

## 5. DISCUSSION AND LIMITATIONS

FTA is designed to isolate reporting reliability from execution reliability. Making an execution failure visible to a model is not sufficient to guarantee that the final response faithfully represents that failure. Within this controlled benchmark, unsupported success claims remain common under some prompt conditions even though the failed prerequisite is explicitly provided. An agent can therefore encounter a visible failure yet still communicate an unsupported outcome to the user.

The policy results suggest that post-failure fidelity is sensitive to the structure imposed on the final response. A generic transparency instruction is associated with lower unsupported-claim rates, but the size of that difference varies across the six tested models. The evidence contract is associated with consistently lower false-success rates while useful-response rates remain high. However, the contract is a bundled intervention: its wording, explicit decision constraints, and output structure change together. The experiment therefore does not identify which individual component causes the observed difference; factorial prompt ablations are required to separate these effects.

FTA is intended as a diagnostic complement to end-to-end agent evaluation. Fixing the failure observation before generation removes tool selection, autonomous retry behavior, changing environment state, and long-horizon planning from the measured capability. This control improves auditability and enables exact comparisons across models and response policies, but it limits ecological coverage. Performance on FTA should therefore not be interpreted as a complete measure of deployed-agent reliability.

Several additional limitations define the scope of the present results. FTA contains only failed prerequisites; matched successful-tool controls are needed to measure whether stronger transparency policies incorrectly suppress valid completion. Tasks are synthetic, Englishonly, and predominantly one-step; no real-world trace validation is included, and the benchmark does not cover partial success, contradictory evidence, multi-agent interaction, or extended trajectories. Prompt conditions are balanced but not crossed using alternative versions of identical tasks, so condition-level differences are descriptive rather than causal estimates of wording effects. Policy metadata is hidden during scoring, but contract formatting can make the condition inferable. Finally, outcomes are scored under a fixed human rubric without reported independent double-annotation or inter-annotator agreement, leaving annotation reliability as an important target for future evaluation.

## 6. CONCLUSION

Tool failure and post-failure reporting are distinct reliability dimensions. FTA isolates the latter by fixing the failed execution state and auditing the final response against an explicit evidence boundary. Across 3,600 human-annotated responses, the tested transparency and evidence-contract policies are associated with lower unsupportedsuccess rates while useful-response rates remain high within blockedtask scenarios. Reliable agents should therefore be evaluated not only by task completion, but by whether their user-facing claims are warranted by the evidence actually obtained.

Acknowledgment. OpenAI ChatGPT was used for language editing.

## 7. REFERENCES

[1] S. Yao et al., “ReAct: Synergizing reasoning and acting in language models,” in ICLR, 2023.

[2] T. Schick et al., “Toolformer: Language models can teach themselves to use tools,” in NeurIPS, 2023.

[3] M. Li et al., “API-Bank: A comprehensive benchmark for toolaugmented LLMs,” in EMNLP, 2023.

[4] Y. Qin et al., “ToolLLM: Facilitating large language models to master 16000+ real-world APIs,” in ICLR, 2024.

[5] S. G. Patil et al., “The Berkeley Function Calling Leaderboard (BFCL): From tool use to agentic evaluation of large language models,” in ICML, 2025.

[6] Y. Qin et al., “Tool learning with foundation models,” arXiv:2304.08354, 2023.

[7] Y. Zhuang, Y. Yu, K. Wang, H. Sun, and C. Zhang, “ToolQA: A dataset for LLM question answering with external tools,” in NeurIPS, 2023.

[8] S. G. Patil, T. Zhang, X. Wang, and J. E. Gonzalez, “Gorilla: Large language model connected with massive APIs,” in NeurIPS, 2024.

[9] W. Liu et al., “ToolACE: Winning the points of LLM function calling,” in ICLR, 2025.

[10] X. Liu et al., “AgentBench: Evaluating LLMs as agents,” in ICLR, 2024.

[11] Y. Ruan et al., “Identifying risks of LM agents with an LMemulated sandbox,” in ICLR, 2024.

[12] J. Lu et al., “ToolSandbox: A stateful, conversational, interactive evaluation benchmark for LLM tool use capabilities,” in Findings ofNAACL, 2025.

[13] S. Yao et al., “τ -bench: A benchmark for tool-agent-user interaction in real-world domains,” in ICLR, 2025.

[14] G. Mialon et al., “GAIA: A benchmark for General AI Assistants,” in ICLR, 2024.

[15] C. Ma et al., “AgentBoard: An analytical evaluation board of multi-turn LLM agents,” in NeurIPS, 2024.

[16] S. Zhou et al., “WebArena: A realistic web environment for building autonomous agents,” in ICLR, 2024.

[17] T. Xie et al., “OSWorld: Benchmarking multimodal agents for open-ended tasks in real computer environments,” in NeurIPS, 2024.

[18] S. Yao, H. Chen, J. Yang, and K. Narasimhan, “WebShop: Towards scalable real-world web interaction with grounded language agents,” in NeurIPS, 2022.

[19] X. Deng et al., “Mind2Web: Towards a generalist agent for the web,” in NeurIPS, 2023.

[20] C. E. Jimenez et al., “SWE-bench: Can language models resolve real-world GitHub issues?,” in ICLR, 2024.

[21] J. Y. Koh et al., “VisualWebArena: Evaluating multimodal agents on realistic visual web tasks,” in ACL, 2024.

[22] H. He et al., “WebVoyager: Building an end-to-end web agent with large multimodal models,” in ACL, 2024.

[23] A. Drouin et al., “WorkArena: How capable are web agents at solving common knowledge work tasks?,” in ICML, 2024.

[24] L. Boisvert et al., “WorkArena++: Towards compositional planning and reasoning-based common knowledge work tasks,” in NeurIPS, 2024.

[25] O. Yoran et al., “AssistantBench: Can web agents solve realistic and time-consuming tasks?,” in EMNLP, 2024.

[26] T. Le Sellier de Chezelles et al., “The BrowserGym ecosystem for web agent research,” Transactions on Machine Learning Research, 2025.

[27] E. Li and J. Waldo, “WebSuite: Systematically evaluating why web agents fail,” arXiv:2406.01623, 2024.

[28] Y. Zhang et al., “ToolBeHonest: A multi-level hallucination diagnostic benchmark for tool-augmented large language models,” in EMNLP, 2024.

[29] J. Sun, S. Y. Min, Y. Chang, and Y. Bisk, “Tools Fail: Detecting silent errors in faulty tools,” in EMNLP, 2024.

[30] Y. Wan, J. Yang, J. Luo, and Y. Qiu, “Failure makes the agent stronger: Enhancing accuracy through structured reflection for reliable tool interactions,” arXiv:2509.18847, 2025.

[31] H. Xia, H. Wang, Z. Liu, Q. Yu, Y. Guo, and H. Wang, “Safe-ToolBench: Pioneering a prospective benchmark to evaluating tool utilization safety in LLMs,” in Findings ofEMNLP, 2025.

[32] J. Ye et al., “ToolSword: Unveiling safety issues of large language models in tool learning across three stages,” in ACL, 2024.

[33] D. Guo et al., “Are your agents upward deceivers?,” arXiv:2512.04864, 2025.

[34] A. Gupta, “ReliabilityBench: Evaluating LLM agent reliability under production-like stress conditions,” arXiv:2601.06112, 2026.

[35] R. Chen, “Evidence-Bound Autonomous Research (Evi-Bound): A governance framework for eliminating false claims,” arXiv:2511.05524, 2025.

[36] E. Debenedetti et al., “AgentDojo: A dynamic environment to evaluate prompt injection attacks and defenses for LLM agents,” in NeurIPS, 2024.

[37] M. Andriushchenko et al., “AgentHarm: A benchmark for measuring harmfulness of LLM agents,” in ICLR, 2025.

[38] Z. Zhang et al., “Agent-SafetyBench: Evaluating the safety of LLM agents,” arXiv:2412.14470, 2024.

[39] S. Lin, J. Hilton, and O. Evans, “TruthfulQA: Measuring how models mimic human falsehoods,” in ACL, 2022.

[40] J. Li, X. Cheng, X. Zhao, J.-Y. Nie, and J.-R. Wen, “HaluEval: A large-scale hallucination evaluation benchmark for large language models,” in EMNLP, 2023.

[41] C. Niu et al., “RAGTruth: A hallucination corpus for developing trustworthy retrieval-augmented language models,” in ACL, 2024.

[42] S. Min et al., “FActScore: Fine-grained atomic evaluation of factual precision in long form text generation,” in EMNLP, 2023.

[43] P. Manakul, A. Liusie, and M. Gales, “SelfCheckGPT: Zeroresource black-box hallucination detection for generative large language models,” in EMNLP, 2023.

[44] P. Kirichenko et al., “AbstentionBench: Reasoning LLMs fail on unanswerable questions,” arXiv:2506.09038, 2025.