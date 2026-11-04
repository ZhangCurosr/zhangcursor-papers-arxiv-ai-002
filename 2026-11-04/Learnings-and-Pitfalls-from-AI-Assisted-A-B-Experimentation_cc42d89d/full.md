# Learnings and Pitfalls from AI-Assisted A/B Experimentation at Scale

Brett Daley<sup>∗</sup>, Ilya Gorbachev<sup>∗</sup>, Anubhav Natani, Nicholas Brown, Athena Dai, Rachel Song, Marcin Pekalski, Josip Domazet, Anchal Choubey, and Daniel Ting

Meta

{brettdaley, ilyagorbachev, annatani, nicholasbro, athenadai, fqsong, mpekalski, josipdomazet, anchalc, dting}@meta.com

## 1 Introduction

A/B testing underpins product decisions by providing trustworthy, reproducible quantification of proposed changes’ efects. Growing large language model (LLM) capabilities have motivated eforts to automate experimentation workflows. Yet LLMs are nondeterministic and prone to hallucination, properties that conflict with the trust experimentation systems must provide. Experiments are also operationally messy: diagnosing anomalies, debugging configurations, and conducting bespoke follow-up analyses require flexible reasoning and manual efort. These tasks let agents accelerate experimentation without replacing its deterministic statistical foundations.

We report lessons from building and operating such an agent. Our central finding is that this tension is resolved not by making the model deterministic, but by bounding where its nondeterminism can reach; the model selects and sequences procedures, while deterministic code computes every quantity it reports. Three consequences surprised us: withholding information from the model improved its answers, demand concentrated on mundane retrieval rather than the autonomous decisions we anticipated, and the agent’s eagerness to help threatened statistical validity.

Prior work places LLMs inside the experimental loop itself: agents act as simulated participants in pre-launch tests of live websites (Lu et al., 2026), or generate treatments that are refined on live $\mathrm { A } / \mathrm { B }$ feedback (Jiang et al., 2026; Wang et al., 2026). Our agent instead helps experimenters retrieve, validate, and diagnose the results of experiments, leaving the launch decision with them. A concurrent industrial system (Lin et al., 2026) adopts the same separation between the model’s choices and the statistics that judge them.

## 2 Agent Design

Our ABTest Agent follows a conventional design: a foundation LLM, invocable tools, and domainspecific context. Users interact with the agent conversationally or register a prompt to run unattended on a schedule or event trigger.

Although the agent’s supporting infrastructure changed rapidly, several design decisions proved durable. These share one principle: the model chooses which procedures to run and their order, but computes nothing itself. Every reported quantity is produced by deterministic, parameterizable code the model invokes. We therefore do not make the model deterministic; we constrain the surfaces its nondeterminism can reach. Figure 1 depicts this boundary.

![](images/bd51c282db4714a53a9ff34ed7044f24c610a8fb26b5190fbfa33f880ae41b95.jpg)  
Figure 1: The ABTest Agent constrains where model nondeterminism can reach rather than removing it. Above the boundary, the LLM chooses and sequences procedures, guided by skills and vetted knowledge; below it, deterministic code produces every reported quantity. Results and errors return to the model, which must repair any report the gate rejects.

The agent’s capabilities are organized as skills: prompt-triggered units of context that instruct the model how to compose several tools toward a high-level goal. The ABTest Agent has 97 skills spanning 11 functional categories (Table 1); their distribution shows where engineering efort went. Five categories—statistical validity, root-cause diagnosis, assignment integrity, data readiness, and incident and on-call response—concern establishing or restoring trust in a result rather than producing one, and together account for 52 of the 97.

Experiment results are exposed through a core analytics API whose interdependent arguments make direct invocation error-prone. Diagnosis must also draw on further systems sharing no common interface: randomization and assignment checks, model-health signals, incident records, and visualization services. We provide a unified interface that assembles these sources and returns tables and figures traceable to their source tool; access to it evolved. Initially, sandboxed Python supplied flexibility to reach arbitrary services: the model wrote code against libraries wrapping each API. Open-ended code generation increased latency, token consumption, and execution errors because malformed calls surface only after execution. We now expose command-line tools that take fixed arguments, validate requests before dispatch, and return standardized output, retaining Python for specialized analyses the command-line surface does not anticipate. The narrower interface forecloses some analyses to make common ones reliable.

The knowledge base serves questions of definition and interpretation: what a metric measures, how it is computed, how two similar metrics difer. Its composition mattered more than its size. A large organization accumulates documentation that is outdated or inconsistent, and an assistant with access to it inherits those contradictions; LLMs degrade with conflicting inputs on the same question (Xu et al., 2024). We therefore restricted the corpus to material vetted by experimentation experts, trading coverage for internal consistency. Withholding information from the model improved its output.

A final constraint governs agent output. A shared or reviewed analysis is packaged as a structured report rather than a chat transcript. Unexpectedly, the report’s consistency proved as important as its content. A document whose shape varies between runs cannot be compared with its predecessor or consumed by a downstream workflow, and conceals omissions—an especially sharp hazard for unsupervised scheduled analyses. Free-form generation was unreliable, varying in structure and producing tables that drifted from the evidence they described. We route report generation through a template-driven tool that emits only if every declared section is populated and every table and figure resolves to a typed reference to an artifact produced by a deterministic analysis tool. In a controlled comparison against free-form generation on our three most-used report workflows, this raised the cosine similarity of matching sections across repeated prompts. Structure and provenance are enforced by the system rather than requested of the model.

Table 1: ABTest Agent skills by category.
<table><tr><td>Category</td><td>Skills</td></tr><tr><td>Statistical Validity</td><td>17</td></tr><tr><td>Root-Cause Diagnosis</td><td>14</td></tr><tr><td>Results &amp; Analysis</td><td>12</td></tr><tr><td>Monitoring &amp; Alerting</td><td>11</td></tr><tr><td>Assignment Integrity</td><td>9</td></tr><tr><td>Data Readiness</td><td>6</td></tr><tr><td>Incident &amp; On-Call</td><td>6</td></tr><tr><td>Reporting &amp; Visualization</td><td>6</td></tr><tr><td>Setup &amp; Configuration</td><td>6</td></tr><tr><td>Support &amp; Routing</td><td>6</td></tr><tr><td>Launch Lifecycle</td><td>4</td></tr><tr><td>Total</td><td>97</td></tr></table>

Table 2: ABTest Agent usage by category.
<table><tr><td>Category</td><td>Frequency</td></tr><tr><td>Results &amp; Analysis</td><td>45%</td></tr><tr><td>Diagnostics &amp; Health</td><td>20%</td></tr><tr><td>Metric Understanding</td><td>15%</td></tr><tr><td>Export</td><td>8%</td></tr><tr><td>Troubleshooting</td><td>7%</td></tr><tr><td>Conceptual Questions</td><td>5%</td></tr></table>

Putting these together, consider a team that schedules the prompt “Every morning, report how my experiment is doing and flag anything unusual” (Figure 1). The model loads the results skill, which has it first query which dates have landed data and then request results for exactly those dates; the validated command-line tools reject dates outside the experiment’s lifetime and skip dates the platform has excluded, so a malformed request fails before any number exists. A dataintegrity check, which the system prompt requires on every results query, retrieves precomputed experiment-health verdicts to supply evidence for flagging anything unusual. The report template gate then raises an error if a declared section is missing or empty, and the model must repair the report before it is emitted. The model chooses every step; code produces every number in the report’s tables and figures.

## 3 Learnings

Simple result fetching remains the dominant use case. Early ambitions focused on moonshot ideas: automated decisions and even automated hypothesis and treatment generation. Demand proved more mundane. As shown in Table 2, fetching estimates comprises almost half of requests, followed by diagnostic and health checks, metric-understanding questions, and data export. Results analysis accounts for just 12 of our 97 skills (Table 1) yet the largest share of user requests (Table 2), while the bulk of our engineering went to establishing trust in results rather than producing them. The imbalance is not a mistake—the trust-establishing skills make a fetched reading worth acting on—but it was not what we planned.

Agents scale domain and operational knowledge faster than expert judgment. Diagnosing an experiment depends less on inference than on knowing where to look: which health signals exist, what a metric excludes, which system change would produce the pattern on screen. That knowledge sat with a few experimentation experts and reached others by consultation, which does not scale with experiment volume. It now reaches them through the agent: the two knowledgeoriented categories of Table 2 are 20% of requests. In these requests, the agent serves as a curated reference rather than an analysis tool, and three quarters of them ask what a metric measures or how two similar metrics difer. Encoding a procedure as a skill changes who does the work: the expert writes it once, and it runs consistently regardless of who asks. The role has shifted from answering questions toward authoring skills and the evaluations that keep them honest. Because those skills all live in one agent, a procedure written by one team becomes available to domains that had never exchanged practice before.

To build those evaluations, subject-matter experts supply test prompts and ground-truth answers, which are turned into reproducible cases with mocked tool environments, so results do not depend on live data. An LLM grades each response against the ground truth (Zheng et al., 2023), backed by exact checks for values a correct answer must or must not contain. The cases fall into two categories. Regression cases run every day, and a drop in accuracy signals that a change broke existing behavior. Aspirational cases are written before the behavior they specify exists and fail until a change makes them pass, as in test-driven development; fixing the target before the work begins is the same pre-specification that Section 4 asks of experimenters.

Agents test and prune hypotheses eficiently. Debugging often means chasing hypotheses that lead nowhere. An agent discarding a hypothesis requires far less than a person pursuing that dead end. Asked to investigate an anomalous metric, the agent found the cause in ten minutes: the metric was being double-counted on days when experiment versions rotated. No skill covered this; the agent worked from its knowledge base, writing queries against source tables to find when the inconsistency first appeared. The investigation would otherwise have stayed open for a day.

## 4 Pitfalls

Where determinism cannot be enforced, quality degrades. Constraints bind wherever the agent’s output passes through code we control; two surfaces fall outside them. The first resists measurement: selectively loaded skills carry retrieval risk, and an instruction the agent never loads is never followed. Unlike a malformed tool call, which fails loudly, an unloaded skill raises no signal. The second we can measure: we connected the agent to a Q&A forum to draft first-line answers—a setting with no template to populate and no artifact to cite—and LLM-judged correctness dropped meaningfully. The same agent, deprived of an enforceable output contract, is markedly less reliable; this is the clearest evidence that the contract does the work.

P-hacking becomes a greater risk. Experimenters want positive results; in a number of sessions, the agent obliged by recommending p-hacking techniques such as removing readings on unfavorable days. The behavior matches the sycophancy documented in LLMs (Sharma et al., 2024), and post-hoc exclusions are a classic source of false positives (Simmons et al., 2011). Experimenters occasionally succeeded with prompts like “Is there any 7-day period where [treatment group] is stat-sig?” and “My experiment can return statistically significant results if we remove [date]. Can you investigate if there are any valid reasons we can justify removing this date?” In concurrent work, reframing a request this way has been shown to bypass coding agents’ refusals to p-hack (Asher et al., 2026).

Agents amplify this risk by making each additional comparison require minimal resources. A new metric, breakdown, or time window is one request away and the answer arrives in seconds. A search once limited by the hours of analyst time it required now has no such brake; each look the agent takes on the experimenter’s behalf is another chance for a false positive.

We now audit completed sessions for requests and proposed analyses that search across metrics or time windows for significance, add post-hoc exclusions, or extend a running experiment after its results have been seen, a practice that inflates the false-positive rate of fixed-horizon tests (Johari et al., 2022). The audit encodes the distinction between exploratory and confirmatory analysis: debugging may look anywhere, but launch decisions should rest on the metrics, success criteria, and duration fixed before the experiment began.

## 5 Conclusion

An experimentation agent is worth only as much as the trust placed in what it reports. Trust does not come from the model’s capability alone; it comes from the constraints built around the model: validated tools, expert-vetted knowledge, and an enforced report contract. Reliability falls where no such contract applies, as with unloaded skills or open-ended forum answers, and the agent’s eagerness to help can become p-hacking unless launch decisions are held to criteria fixed in advance. Building these constraints, rather than pursuing full autonomy, is what makes the agent trustworthy for experimentation at scale.

Acknowledgments. We thank Rohan Sharma for sharing the double-counting investigation in Section 3, Zitong Zhou for conducting the usage analysis behind Table 2, and Houssam Nassif for reviewing a draft of this paper. We gratefully acknowledge leadership support from Goutam Daggubati, Suhash Chakraborty, Andras Balogh, BG Goyal, and Jay Venkatraman.

## References

Asher, S., Malzahn, J., Persano, J., Paschal, E., Myers, A., and Hall, A. (2026). Do Claude Code and Codex P-hack? Sycophancy and statistical analysis in large language models. SSRN:6270199.

Jiang, Z., Chen, Y., Pan, Y., Hu, Z., Fan, W., Li, Q., Wang, H., Wang, J., and Ou, W. (2026). A/B Agent: A self-evolving agent for strategy iteration in industrial A/B testing. arXiv:2608.04625.

Johari, R., Koomen, P., Pekelis, L., and Walsh, D. (2022). Always valid inference: Continuous monitoring of A/B tests. Operations Research, 70(3):1806–1821.

Lin, J., Lan, R., Zhu, X., Zhang, B., Lai, C., Chen, C., Wang, D., et al. (2026). PILOT technical report. arXiv:2608.18637.

Lu, Y., Hsu, T.-Y., Gu, H., Cui, L., Xie, Y., Headden, W. P., Yao, B., et al. (2026). Agent A/B: Automated and scalable A/B testing on live websites with interactive LLM agents. In CHI Conference on Human Factors in Computing Systems (CHI). Extended abstract.

Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S. R., Cheng, N., et al. (2024). Towards understanding sycophancy in language models. In International Conference on Learning Representations (ICLR).

Simmons, J. P., Nelson, L. D., and Simonsohn, U. (2011). False-positive psychology: Undisclosed flexibility in data collection and analysis allows presenting anything as significant. Psychological Science, 22(11):1359–1366.

Wang, H., Wu, Y., Chang, D., Wei, L., and Heldt, L. (2026). Self-evolving recommendation system: End-to-end autonomous model optimization with LLM agents. In ACM Conference on Recommender Systems (RecSys).

Xu, R., Qi, Z., Guo, Z., Wang, C., Wang, H., Zhang, Y., and Xu, W. (2024). Knowledge conflicts for LLMs: A survey. In Conference on Empirical Methods in Natural Language Processing (EMNLP).

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In Neural Information Processing Systems (NeurIPS).