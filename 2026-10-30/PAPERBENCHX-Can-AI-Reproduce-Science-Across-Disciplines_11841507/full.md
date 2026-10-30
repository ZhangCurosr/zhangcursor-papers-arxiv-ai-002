# PAPERBENCHX: Can AI Reproduce Science Across Disciplines?

UniPat AI

## Abstract

Before AI can be trusted to discover new science, it should be able to reproduce what science already knows. Reproduction provides a rigorous checkpoint for an AI scientist: the target claim is known, yet recovering it still requires translating an incomplete scientific description into an executable experiment and producing evidence that can be independently verified. Existing research-reproduction benchmarks largely focus on AI and machine learning, where experiments share relatively homogeneous software stacks. Extending this capability across scientific disciplines introduces a different challenge: agents must operate domain-specific scientific software, make physical and numerical choices, diagnose long-running computations, and distinguish valid evidence from merely plausible outputs. We introduce PAPERBENCHX, a benchmark of 93 expert-constructed paper-reproduction tasks spanning 12 research areas across electromagnetics, photonics, chemistry, materials science, biology, and robotics. Each task provides the source paper, a public task specification, and a domain-native scientific environment involving specialized scientific software. Agents must reconstruct a key experiment, execute and diagnose it, and deliver a runnable workflow that supports the target scientific claim. To evaluate scientific validity rather than surface-level agreement, we replay every submission from scratch and score only regenerated evidence against expert-validated criteria. Across 10 frontier agent configurations, the best configuration fully reproduces only 14% tasks under the 90% threshold. Meanwhile, mean rubric scores range from 27.53% to 59.08%, indicating that substantial partial progress rarely translates into complete scientific reproduction. These results reveal a large gap between paper understanding and reliably producing reproducible scientific evidence.

<sup></sup> Homepage: https://unipat.ai/blog/PaperBenchX

<sup>§</sup> Code: https://github.com/UniPat-AI/PaperBenchX

<sup></sup> Leaderboard: https://unipat.ai/benchmarks/PaperBenchX

![](images/260a051e7e1d72117430283f854ee9aa1da357fd5c7fc77af13a885d0b2ef89e.jpg)  
Figure 1: Frontier agents on PAPERBENCHX. Bars show full scores (> 90%); boxes show partial scores. GPT-6-Astra and GPT-5.6-sol use Codex; all others use Claude Code.

![](images/95c7e92c68ba94fb38ac37e4de97f769e630645a68e896d3f537678678acd5eb.jpg)  
Figure 2: From task package to replay-grounded score. An agent receives the paper, task specification, environment, and starter submission; interacts with domain-native scientific software; and returns a runnable package. Clean replay regenerates the evidence used to score the scientific rubric. The displayed leaves and scores are illustrative.

## 1 Introduction

Reproducibility is one of the defining features of science: a result is meaningful not only because it has been reported, but because the assumptions, procedures, and evidence behind it can be reconstructed and independently examined. For AI systems, this makes reproduction a demanding test of scientific competence. An agent must go beyond understanding a paper’s conclusion to recover how the result was obtained, translate incomplete descriptions into executable procedures, operate specialized tools, diagnose failures, and determine whether the resulting evidence actually supports the claim. Existing reproduction benchmarks largely focus on AI and machine learning, while cross-domain scientific benchmarks typically isolate only parts of the workflow (Starace et al., 2025; Siegel et al., 2024; Geng et al., 2026). What remains underexplored is whether an agent can reproduce published experiments across disciplines, inside the environments native to the underlying science.

This distinction matters because, outside machine learning, the environment is not merely infrastructure. It is part of the scientific problem. Reproducing an antenna result may require constructing and solving an electromagnetic model in HFSS; reproducing a photonics result may depend on geometry, boundary conditions, mode extraction, and convergence behavior in Lumerical or Meep (Oskooi et al., 2010); reproducing a chemistry or materials result may require choosing basis sets, numerical tolerances, or electronic-structure settings; and reproducing a robotics result may require executing and validating behavior inside an embodied simulator. A workflow can run successfully, generate plausible numbers, and still represent the wrong physical system. Cross-disciplinary scientific reproduction therefore demands more than writing code that executes.

We identify three coupled challenges. First, scientific reconstruction: papers are not executable specifications. Agents must translate prose, equations, figures, and domain conventions into a concrete experiment, identify the target observable, and resolve modeling choices that are not fully specified by the task contract. Second, domain-native execution: agents must operate specialized scientific software, select appropriate physical and numerical settings, interpret diagnostics, and determine whether a computation is stable and converged. Third, scientific verification: matching a reported value is not sufficient. The result must arise from a valid experimental workflow, and the evidence supporting it must be independently regenerable. These challenges separate scientific reproduction from either rerunning released code or producing a plausible final answer.

To study this setting, we introduce PAPERBENCHX, a benchmark for paper-level scientific reproduction across disciplines. It contains 93 expert-constructed tasks spanning 12 research areas across electromagnetics, photonics, computational chemistry, computational materials science, computational biology and bioinformatics, and robotics and embodied AI. The tasks run in 10 domain-native scientific computing environments, including Ansys HFSS, Ansys Lumerical, Meep (Oskooi et al., 2010), PySCF/GPU4PySCF (Sun et al., 2026), ABACUS (Zhou et al., 2025), Quantum ESPRESSO (Giannozzi et al., 2009), AutoDock Vina (Eberhardt et al., 2021), Habitat (Savva et al., 2019), SimplerEnv (Li et al., 2025), and AI2-THOR (Kolve et al., 2017). Each task is centered on a verifiable scientific claim from a published paper. The agent receives the paper, a public task specification defining the reproduction scope and deliverables, and the required scientific environment, but does not receive the private reference workflow, reference evidence, or evaluation rubric. Its goal is to reconstruct the experiment and deliver a runnable workflow that regenerates evidence supporting the target claim.

PAPERBENCHX evaluates sustained, end-to-end scientific reproduction through replay-grounded verification: agents must maintain scientific intent, computational state, and evidence provenance while reconstructing experiments, running and diagnosing long-running simulations, and refining their analyses; after submission, each workflow is rerun from scratch in a clean environment, and only regenerated artifacts are scored against expert-validated criteria. This design distinguishes plausible implementations and successful software execution from actual reproduction of the target scientific result. Across 10 frontier agent configurations on the same 93 tasks, the strongest configuration exceeds the 90% fullreproduction threshold on only 13 tasks, while 71 tasks are not fully reproduced by any evaluated configuration; mean rubric scores range from 27.53% to 59.08%. Agents often make substantial progress in modeling and execution but fail at scientific validation, revealing a substantial gap between partial progress and reliably closing the loop from paper understanding to reproducible scientific evidence.

Our contributions are threefold:

• Cross-disciplinary paper-level reproduction. We introduce 93 expert-constructed tasks spanning 12 research areas and 10 scientific computing environments, each centered on a verifiable claim from a published paper.

• Domain-native scientific execution. Rather than reducing scientific problems to a common coding interface, PAPERBENCHX requires agents to operate inside the simulators, software stacks, numerical conventions, and resource constraints required by the underlying research.

• Replay-grounded scientific verification. We rerun every submitted workflow from scratch and score only regenerated evidence against expert-validated scientific criteria, distinguishing plausible outputs from reproducible scientific results.

## 2 Related Work

## 2.1 Machine-Learning Research and Paper Reproduction

MLAgentBench and RE-Bench evaluate long-horizon machine-learning experimentation and research engineering, but their objectives are specified as task-level metric improvement rather than reproduction of published claims (Huang et al., 2023; Wijk et al., 2024). PaperBench and SA-Bench move toward paper-level evaluation through reproduction or semantic-alignment rubrics (Starace et al., 2025; Hu et al., 2026). They establish that a final metric alone cannot measure fidelity to a paper, but remain centered on relatively homogeneous machine-learning software stacks rather than heterogeneous domain-native scientific environments.

## 2.2 Cross-Domain Scientific Benchmarks

Cross-domain benchmarks broaden the scientific scope but generally isolate only part of the reproduction process. SciCode, DiscoveryBench, and ScienceAgentBench evaluate scientific coding, hypothesis discovery, or data analysis, while CORE-Bench recovers results by rerunning released code (Tian et al., 2024; Majumder et al., 2024; Chen et al., 2024; Siegel et al., 2024). DISCOVERYWORLD evaluates interactive discovery in a unified virtual world, and NatureBench constructs paper-derived scientific-ML tasks (Jansen et al., 2024; Wang et al., 2026). ScienceIDE most directly evaluates domain-native scientific execution by placing agents inside production scientific software, but its tasks concern repository-level repair and implementation rather than reproduction of a published experiment (Geng et al., 2026).

PAPERBENCHX targets the intersection left open by these benchmarks. It requires paper-level reproduction of key experiments, executes tasks inside heterogeneous domain-native scientific environments, and uses replay-grounded scientific verification. Submitted workflows must run again from scratch and regenerate evidence that supports the target claim; neither a plausible final answer nor a completed repository is sufficient. Table 1 summarizes these distinctions.

## 3 The PaperBenchX Benchmark

PAPERBENCHX evaluates whether an agent can turn a scientific claim into independently replayable evidence in the domain-native environment required by the underlying research. Figure 2 summarizes the complete lifecycle, from the materials provided to the agent through clean replay and evidence-based scoring. This section defines the task, describes the benchmark composition and construction pipeline, and specifies the evaluation protocol.

## 3.1 Task Definition

Each PAPERBENCHX task is centered on a verifiable scientific claim from a published scientific paper. As shown in Figure 2, the task input has three parts: the paper; a public task specification defining the scientific scope, target conditions and variants, observables, required evidence, and delivery interface; and a scientific computing environment containing the domain-native simulator, dependencies, data, and allocated resources. The environment is part of the task definition rather than incidental infrastructure, because model representations, numerical conventions, execution procedures, and convergence evidence differ across scientific software.

The agent must submit a runnable package containing all source, model-building, simulation, postprocessing, and analysis code needed for reproduction, with reproduce.sh as the canonical entry point. During clean replay, the evaluator refreshes the task container, restores only the submitted package, runs this entry point, and collects the outputs regenerated by the workflow. Those outputs, rather than the agent’s assertions, are scored against a hidden scientific rubric. The private reference workflow and reference artifacts are never exposed to the agent. Reference artifacts are expert-generated simulator logs, raw numerical outputs, convergence records, and derived metrics produced during task construction; they establish that the task is executable and ground its expected evidence and scoring criteria. Values reported in the source paper remain visible, but private tolerances and reference evidence are withheld. The task therefore tests whether an agent can construct a scientifically valid and independently replayable procedure, not whether it can recover a hidden number.

Table 1: Comparison with prior agent benchmarks. Checkmarks indicate that a benchmark explicitly evaluates the named capability.
<table><tr><td>Benchmark</td><td>Research scope</td><td>Primary evaluation target</td><td>Paper-level reproduction</td><td>Domain-native scientific execution</td><td>Replay-grounded scientific verification</td></tr><tr><td colspan="6">Machine-learning research and paper-reproduction benchmarks</td></tr><tr><td>MLAgentBench (Huang et al., 2023)</td><td>Machine learning</td><td>ML experimentation tasks</td><td></td><td></td><td>X</td></tr><tr><td>RE-Bench (Wijk et al., 2024)</td><td>Machine learning</td><td>Research-engineering tasks</td><td></td><td></td><td></td></tr><tr><td>PaperBench (Starace et al., 2025)</td><td>Machine learning</td><td>Paper reproduction</td><td>× × √</td><td>×××</td><td>××</td></tr><tr><td>SA-Bench (Hu et al., 2026)</td><td>Machine learning</td><td>Paper reproduction</td><td>√</td><td>x</td><td></td></tr><tr><td colspan="6">Cross-domain scientific benchmarks</td></tr><tr><td>DiscoveryBench (Majumder et al., 2024)</td><td>6 scientific fields</td><td>Data-driven discovery</td><td></td><td></td><td>x</td></tr><tr><td>ScienceAgentBench (Chen et al., 2024)</td><td>4 scientific fields</td><td>Scientific workflow tasks</td><td></td><td></td><td>×</td></tr><tr><td>CORE-Bench (Siegel et al., 2024)</td><td>3 scientific fields</td><td>Repository-result reproduction</td><td></td><td>×××</td><td>x</td></tr><tr><td>DISCOVERYWORLD (Jansen et al., 2024)</td><td>8 scientific fields</td><td>Interactive scientific discovery</td><td>×××××</td><td></td><td>×</td></tr><tr><td>NatureBench (Wang et al., 2026)</td><td>6 scientific fields</td><td>Scientific experiment tasks</td><td></td><td></td><td></td></tr><tr><td>ScienceIDE (Geng et al., 2026)</td><td>Physics and geoscience</td><td>Repository-level scientific coding</td><td>x</td><td></td><td></td></tr><tr><td>PAPERBENCHX</td><td>12 research areas</td><td>Reproduction of key paper experiments</td><td>√</td><td></td><td>V</td></tr></table>

## 3.2 Task Statistics

Benchmark composition. PAPERBENCHX contains 93 paper-reproduction tasks spanning 12 research areas and 10 domain-native scientific computing environments, with 3,168 scorable criteria. The twelve research areas are radio-frequency antennas and arrays; radio-frequency metasurfaces and scattering; radio-frequency filters and frequency-selective surfaces; integrated photonic devices; photonic crystals and topological photonics; optical metasurfaces, gratings, and plasmonics; optical computing and signal processing; optical resonators, filters, and tunable devices; computational chemistry; computational materials science; computational biology and bioinformatics; and robotics and embodied AI. Figure 3A shows the distribution of tasks across these research areas.

The tasks run in ten domain-native scientific computing environments: Ansys HFSS, Ansys Lumerical, Meep, PySCF/GPU4PySCF, ABACUS, Quantum ESPRESSO, AutoDock Vina, Habitat, SimplerEnv, and AI2-THOR. The benchmark draws on 93 distinct source papers from 44 publication venues, as summarized in Figure 3B. Each task has an execution budget determined by its simulation scale and resource requirements; agent time limits range from 4 h to 72 h, with a median of 7 h. Figure 3C shows the distribution of scorable criteria across the same 93 tasks.

Release and benchmark integrity. We release one representative task from each of the twelve research areas. Nine tasks include complete, redistributable scientific computing environments; the remaining three rely on commercially licensed software, for which we release the task specifications and scoring details but not the corresponding software environments. The remaining tasks are retained as a hidden evaluation set to reduce test-set contamination and preserve the benchmark’s validity for evaluating future agents.

![](images/71e0bc33f4e31ac1af4349bb9496a3ef5eca5ab8da6380ec378253ec99ccbd79.jpg)

![](images/9b396e91b875fcf4de21c3938f7a34f15ba6ce02835b9ee86e98bcd0d577ad8b.jpg)

![](images/6bf4fef61801e3818a8d6398ea0800d4518dbb79ec13cbc44a20de3095620be3.jpg)  
Figure 3: Coverage and evaluation structure of PAPERBENCHX. A, The 93 tasks span 12 research areas in six scientific domains. B, Their 93 source papers span 44 publication venues; bar colors correspond to the 12 research areas in A. C, The task rubrics contain 3,168 scorable criteria (median 38 per task); points are tasks and black ticks mark research-area medians.

![](images/37d59a38065bbe59a9337b935d05f8ac56968cf64e879fa268bc8a56a51bdd9d.jpg)  
Figure 4: Task construction and quality control. Experts turn a published claim into a runnable task by defining its scientific scope, provisioning the domain-native environment, and specifying the public task contract. Reference replay, calibration checks, and independent expert review verify executability, fairness, and evidence requirements before the task is admitted to the benchmark.

## 3.3 Task Construction Pipeline

The pipeline proceeds in two stages (Figure 4): first building a runnable candidate task, and then validating whether it can be admitted to the benchmark.

Stage 1: Build. Steps 1–2: paper and claim selection. A domain expert selects a published paper, identifies a scientifically meaningful claim, and checks that it can be reproduced under the available software, licensing, compute, and time constraints. The expert then defines the scientific scope: the target figure or table, model configuration, experimental conditions and variants, observables, valid methodological alternatives, and evidence required to support the claim. Steps 3–4: environment and public contract. The expert provisions the domain-native scientific environment and writes the public task specification, which states the scope, required outputs, submission location, and replay entry point without disclosing the private rubric or reference solution. Step 5: initial execution check. The expert implements a private reference workflow and executes it in the assembled environment. This first execution check verifies that the simulator can be invoked, the scoped experiment can finish within the resource envelope, and the required reference artifacts can be generated. A task reaches the candidate stage only after this end-to-end workflow runs successfully.

Stage 2: Validate and review. Step 6: rubric construction. The expert converts the evidence requirements into independently scorable rubric leaves covering model construction, parameter settings, execution, convergence, post-processing, and claim validation. Each leaf specifies the evidence needed for credit and its scoring rule, tolerance, and weight. Step 7: clean reference replay. The private reference workflow is rerun from a clean copy of the final container through the same replay interface used for submissions. Unlike the initial check in Step 5, this stricter execution check verifies that the task does not depend on stale state, undeclared files, or hidden setup; it also confirms that the expected artifacts can be regenerated under the final resource limits and consumed by the evaluator. Step 8: calibration checks. Successful, partially successful, and invalid workflows are scored to confirm that the rubric awards supported progress, withholds credit for missing or contradictory evidence, and distinguishes a complete reproduction from a plausible but scientifically invalid result. Step 9: independent expert review. Two independent domain experts inspect the scientific scope, fairness, reproducibility, evidence requirements, and scoring behavior.

Tasks that fail because of their scope, environment, or reference workflow are rebuilt from Stage 1. Failures caused by the rubric, scoring rules, or evaluator return to Step 6. In either case, all affected downstream execution, calibration, and review checks are repeated. Only tasks that pass the full pipeline are frozen and admitted to the benchmark. Across the benchmark, task construction and validation required more than 1,000 hours of doctoral-level expert work, or approximately ten hours per task.

## 3.4 Evaluation

Scientific correctness cannot be established from an agent’s final report. PAPERBENCHX therefore evaluates each submission through clean replay, evidence-based grading, and hierarchical scoring. This evaluation of the agent’s workflow is distinct from the clean reference replay in Step 7 of task construction, which validates the benchmark task itself. Appendix B gives the full protocol.

Clean replay. After delivery, generated outputs and mutable execution state are removed. The evaluator refreshes the task container, restores only the agent’s submitted package, and invokes bash/home/subm ission/reproduce.sh with private evaluation resources inaccessible, outbound network disabled, and task-specific resource limits enforced. Exit status, timing, source hashes, and execution provenance are recorded independently of the agent’s report. Only artifacts regenerated by this replay are admissible evidence; a replay failure is recorded explicitly, and credit remains possible only for rubric requirements supported by admissible regenerated evidence.

Evidence-based grading. Replay outputs—including simulator logs, raw numerical arrays, trajectories, convergence records, derived metrics, and visualizations—are frozen as a read-only evidence set. Every scientific conclusion must be traceable to this set: a convergence claim requires a convergence record, and a reported observable requires both the raw output and the analysis that derives it. Deterministic checks evaluate directly testable properties such as file schema, units, numerical consistency, and task-specific tolerances. Requirements that need domain interpretation are assessed for every evaluated configuration by the same GPT-5.6-sol scientific rubric evaluator, with access to the hidden rubric, reference artifacts, and read-only replay evidence. The agent’s report may locate or explain evidence, but it cannot instruct the evaluator or replace missing or contradictory artifacts.

Rubric stages. Each rubric leaf is assigned to Modeling (scientific models, assumptions, inputs, parameters, and configuration), Execution (domain-native execution, required completion or convergence, artifacts, and provenance), or Validation (post-processing and quantitative support for the target claim). Stage scores normalize earned weight by the total rubric weight within each stage.

Scoring. Let $s _ { i } \in [ 0 , 1 ]$ be the score of rubric leaf i and $w _ { i }$ its effective hierarchical weight. The leaves represent independently assessable scientific requirements, and their effective weights sum to one. The task score is

$$
R = \sum _ { i } w _ { i } s _ { i } , \quad \quad \sum _ { i } w _ { i } = 1 .\tag{1}
$$

We report performance at both the task-completion level and the partial-progress level using two metrics: • Full rate, the fraction of tasks with $R > 0 . 9 . \mathrm { ~ A ~ }$ score of exactly 0.9 does not qualify.

• Partial score, the mean of R across the shared 93-task result set, which captures weighted progress on incomplete long-horizon reproductions.

Every configuration is evaluated on the same task set; experiment-specific fallback and missing-outcome handling is defined in Section 4.1. A missing artifact affects only the rubric leaves that depend on it, leaving unrelated verified evidence eligible for credit.

## 4 Experiments

## 4.1 Experimental Setup

Models and scaffolds. We evaluate ten model–scaffold configurations: GPT-6-Astra (Codex) (OpenAI, 2026b), Fable 5.1 (Anthropic, 2026b), Fable 5 (Anthropic, 2026b), Opus 5 (Anthropic, 2026b), GPT-5.6-sol (Codex) (OpenAI, 2026b), Kimi-K3 (Moonshot AI, 2026), Qwen3.8-Max (Alibaba Cloud, 2026), GLM-5.3 (Zhipu AI, 2026; Alibaba Cloud, 2026), DeepSeek-V4.1-Flash (DeepSeek, 2026), and HY3 (Tencent, 2026). Tasks are packaged as Harbor environments (Laude Institute, 2026). The two GPT models use Codex (OpenAI, 2026a), while the other eight use Claude Code (Anthropic, 2026a). All configurations use the maximum available reasoning setting and receive the same public task materials and shell access, without access to the private reference workflow or rubric.

Evaluation protocol. All configurations are evaluated on the same 93 tasks. When Fable 5 or Fable 5.1 refuses a task, the system automatically falls back to Opus 5: the original refusal is retained for the refusal rate, while the fallback submission determines the task score. Missing outcomes and refusals without a valid fallback count as zero. We report the partial score as the mean rubric score, the full rate as the fraction scoring strictly above 90%, and the refusal rate from the original model responses. All available submissions are clean-replayed and graded by the same GPT-5.6-sol evaluator (OpenAI, 2026b), supplemented by deterministic checks.

## 4.2 Main Results

<table><tr><td colspan="2"></td><td colspan="3">Overall (%)</td><td colspan="4">Stage score (%)</td><td colspan="2">Average effort</td></tr><tr><td>Model</td><td>Harness</td><td>Partial</td><td>Full</td><td></td><td>Ref. Modeling</td><td></td><td>Execution Validation</td><td></td><td>Turns Time (h)</td><td></td></tr><tr><td>GPT-6-Astra</td><td>Codex</td><td>59.08 13.98</td><td></td><td>0.00</td><td></td><td>72.72</td><td>69.35</td><td>52.76</td><td>348.7</td><td>5.96</td></tr><tr><td>Fable 5.1</td><td>Claude Code</td><td>58.17 12.90 11.83</td><td></td><td></td><td></td><td>71.00</td><td>69.92</td><td>51.06</td><td>113.0</td><td>4.65</td></tr><tr><td>Fable 5</td><td>Claude Code</td><td></td><td>57.96 11.83</td><td>13.98</td><td></td><td>72.13</td><td>68.99</td><td>50.39</td><td>112.0</td><td>5.09</td></tr><tr><td>GPT-5.6-sol</td><td>Codex</td><td>55.07</td><td>9.68</td><td>0.00</td><td></td><td>69.85</td><td>71.23</td><td>45.29</td><td>237.8</td><td>3.28</td></tr><tr><td>Opus 5</td><td>Claude Code</td><td>52.52</td><td>7.53</td><td>0.00</td><td></td><td>66.99</td><td>67.16</td><td>47.76</td><td>259.2</td><td>6.52</td></tr><tr><td>Kimi-K3</td><td>Claude Code</td><td>49.24</td><td>7.53</td><td>0.00</td><td></td><td>62.40</td><td>60.70</td><td>42.32</td><td>185.9</td><td>5.75</td></tr><tr><td>Qwen3.8-Max</td><td>Claude Code</td><td>46.76</td><td>5.38</td><td>0.00</td><td></td><td>64.41</td><td>61.02</td><td>41.65</td><td>234.4</td><td>5.76</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>Claude Code</td><td>45.47</td><td>7.53</td><td>0.00</td><td></td><td>59.87</td><td>57.51</td><td>41.13</td><td>372.6</td><td>6.74</td></tr><tr><td>GLM-5.3</td><td>Claude Code</td><td>40.78</td><td>7.53</td><td>0.00</td><td></td><td>51.11</td><td>51.85</td><td>37.06</td><td>290.8</td><td>6.84</td></tr><tr><td>HY3</td><td>Claude Code</td><td>27.53</td><td>2.15</td><td>0.00</td><td></td><td>36.83</td><td>40.59</td><td>19.67</td><td>197.4</td><td>4.94</td></tr><tr><td>Average</td><td>一</td><td>49.26</td><td>8.60</td><td>2.58</td><td></td><td>62.73</td><td>61.83</td><td>42.91</td><td>235.2</td><td>5.55</td></tr></table>

Table 2: Model-level results on the shared set of 93 tasks. Partial, Full, Ref., and all stage scores are reported as percentages; metric definitions and outcome handling follow Section 4.1.

Overall performance. GPT-6-Astra (Codex) leads with a 59.08% partial score and a 13.98% full rate, yet exceeds the 90% threshold on only 13 of 93 tasks. Across models, the macro-average partial score is 49.26%, whereas the average full rate is only 8.60%, showing that substantial partial progress rarely becomes complete reproduction. This gap is concentrated in scientific validation: average Modeling and Execution scores reach 62.73% and 61.83%, respectively, while Validation falls to 42.91%. Current agents can therefore make substantial progress in modeling and execution, but reliably reproducing a paper’s key experimental results and producing a scientific workflow sufficient to verify them remains difficult.

Performance by Research Area. Performance varies substantially across research areas (Figure 5). Mean partial score ranges from 34.05% for RF filters and frequency-selective surfaces to 84.32% for computational materials science. The heatmap also reveals substantial model–domain variation: GPT-

![](images/14b8596b1324783dd3352590cb5eed542867a577165e21dd33c3e2791ec5e0f1.jpg)  
Figure 5: Domain-specific capability profiles. (a) Mean rubric score by model and research area, ordered from hardest to easiest; circles mark Fable refusals scored using the Opus 5 fallback. (b) Domain difficulty, defined as 100 minus the cross-model mean. (c) Overall scores, matching Table 2. Names are abbreviated.

6-Astra leads six of the twelve areas, while Fable 5 and Fable 5.1 each lead three, and no configuration dominates every area.

## 5 Analysis

Partial Progress Rarely Composes into Full Reproduction. Only 80 of 930 outcomes exceed the 90% threshold, and these successes are concentrated on 22 tasks; 71 tasks are never completed by any configuration, and 11 of the remaining 22 are completed by at most two. Figure 6(a) localizes this composition gap. Modeling and Execution have median scores of 73.4% and 73.5%, respectively, whereas Validation has a median of only 40.0%. Even among the 304 outcomes scoring at least 80% on both Modeling and Execution, only 53 (17.4%) exceed the full-reproduction threshold, with a mean Validation score of 58.37%. These results show that current agents can often reconstruct and execute substantial portions of a scientific workflow, yet still fail to compose those partial successes into an end-to-end reproduction that produces credible evidence and validates the paper’s claims.

Interaction Efficiency Varies Substantially Across Agents. Figure 6(b) compares each configuration’s trimmed mean interaction turns with its Partial score and finds only a weak negative correlation (r = −0.14). Fable 5.1 and Fable 5 average only about 113 turns yet achieve Partial scores of 58.17% and 57.96%, respectively, close to the 59.08% attained by GPT-6-Astra with an average of 348.7 turns. By contrast, DeepSeek-V4.1-Flash averages 372.6 turns but scores only 45.47%. Longer trajectories do not necessarily improve reproduction and may instead reflect repeated trial and error, failure recovery, or continued execution of a mis-specified workflow. These results indicate that interaction count alone is insufficient to explain reproduction performance; the accuracy of early-stage decisions may be an important factor shaping both interaction efficiency and final scores.

Upstream Decisions Shape the Value of Execution. We sample 50 trajectories from each of four models and divide tool-mediated time into paper reconstruction, implementation, native execution, diagnosis and tuning, and evidence and replay. Figure 6(c) shows that native execution accounts for 51.8–63.3% of elapsed time, but models differ substantially in how effectively they convert it into valid scientific evidence. Fable 5 allocates 37.1% of its time to paper reconstruction and implementation, the largest upstream share among the four models.

![](images/3371be31465006815b032193de568f2595d2e28a0b1f2c66629a5caa22809b9b.jpg)

![](images/402d52d38ec3b0117ef5cc5d9e0ac22f1d0fe8a90a2344ecdc1071b1c274e0be.jpg)

![](images/be68ddb7006de9dac40c09d32c3f9706a83441ae31daa01f26406ada59016421.jpg)  
Figure 6: Stage performance, interaction length, and workflow allocation. (a) Stage-score distributions over 930 outcomes; thick and thin lines denote the interquartile and 10th–90th percentile ranges, and circles and diamonds denote medians and means. (b) Partial score versus the trimmed average turns in Table 2. (c) Mean tool-mediated time allocation across 200 sampled trajectories (50 per model), with phases coded from timestamped tool actions using fixed rules.

The F-stub UWB MIMO antenna task illustrates this pattern. Fable 5 scores 28.9% in 159 turns, versus 27.4% for GPT-6-Astra in 464 turns and 18.1% for HY3 in 390 turns. It completes three antenna variants and recovers evidence for port reciprocity, radiation patterns, and diversity gain. This case shows that execution alone is insufficient. Sound reconstruction, parameter choices, and workflow design shape whether computation yields verifiable evidence.

Final Outputs Alone Cannot Distinguish Scientific Failure Modes. Failures take two qualitatively different forms. Some workflows execute successfully but support an incorrect conclusion because of errors in modes, ports, parameters, mechanism implementation, or result extraction. On the polarizationinsensitive LNOI edge-coupler task, GPT-6-Astra receives full Execution credit but only 41.9% in Validation. Similarly, several models complete replayable experiments for the eight-element smartphone MIMO antenna but fail to reproduce its reported performance.

Other failures invalidate the evidence itself. Cached results can prevent verifier-owned replay, privileged test-state information can invalidate scientific results, and premature termination can leave no scoreable submission. Final outputs obscure this distinction: one workflow executes but reaches the wrong conclusion, whereas the other establishes no trustworthy result.

Final-output agreement is therefore insufficient. By separately scoring Modeling, Execution, and Validation and checking replay and provenance, PAPERBENCHX distinguishes experimental-construction errors, incorrect conclusions, and unverifiable evidence. End-to-end reproduction requires not only an output, but one produced by a correctly constructed and independently verifiable scientific workflow.

## 6 Conclusion

PAPERBENCHX tests whether agents can turn published scientific claims into executable workflows and independently verifiable evidence. Across 93 tasks in 12 research areas and 10 domain-native environments, we evaluate 10 frontier agent configurations. The best achieves a 59.08% partial score but only a 13.98% full rate; no configuration reaches full reproduction on 71 tasks. Agents often build plausible models and run native simulations, but validation remains their weakest stage, and more interaction does not necessarily improve performance. These results make scientific reproduction an evidence-grounded checkpoint for AI scientists and motivate better diagnosis, provenance, and verification.

## Contribution

Core contributors. Ruoyu Wu<sup>1</sup>, Zengji Tu<sup>1</sup>, Aolong Sun<sup>1</sup>, Tianyi Ma<sup>1</sup>, Liang Chen<sup>1</sup>, Kuan Li<sup>1</sup>

Contributors. Junren Li<sup>1</sup>, Xinbo Xu<sup>1</sup>, Li Fu<sup>1</sup>,Ziyan Zhang<sup>1</sup>, Pinrui Huang<sup>1</sup>, Xiaochen Wang<sup>1</sup>, Chengda Wu<sup>1</sup>, Yutao Zhou<sup>1</sup>, Jin Chen , Shen Yan , Xu Li , Zhongfei Hou , Nan Shu , Chaoyi Huang , Sihang Yuan , Zihan Xu

Advisors. Baobao Chang<sup>2</sup>, Lin Chang<sup>2</sup>, Jianjun Wu<sup>2</sup>

Affiliations. <sup>1</sup>UniPat AI <sup>2</sup>Peking University

## References

Alibaba Cloud. Model studio model catalog. Alibaba Cloud Documentation, 2026. URL https: //help.aliyun.com/zh/model-studio/models. Accessed 2026-09-25.

Anthropic. Claude code overview. Product documentation, 2026a. URL https://docs.anthropic.com/e n/docs/claude-code/overview. Accessed 2026-09-25.

Anthropic. Models overview. Claude API Documentation, 2026b. URL https://platform.claude.com/ docs/en/about-claude/models/overview. Accessed 2026-09-25.

Ziru Chen, Shijie Chen, Yuting Ning, Qianheng Zhang, Boshi Wang, Botao Yu, Yifei Li, Zeyi Liao, Chen Wei, Zitong Lu, Vishal Dey, Mingyi Xue, Frazier N. Baker, Benjamin Burns, Daniel Adu-Ampratwum, Xuhui Huang, Xia Ning, Song Gao, Yu Su, and Huan Sun. ScienceAgentBench: Toward rigorous assessment of language agents for data-driven scientific discovery. arXiv preprint arXiv:2410.05080, 2024. URL https://arxiv.org/abs/2410.05080.

DeepSeek. DeepSeek V4.1 Flash. Official model announcement, 2026. URL https://www.deepseek.com /news/deepseek-v4-1-flash/. Accessed 2026-09-25.

Jerome Eberhardt, Diogo Santos-Martins, Andreas F. Tillack, and Stefano Forli. AutoDock Vina 1.2.0: New docking methods, expanded force field, and python bindings. Journal ofChemical Information and Modeling, 61(8):3891–3898, 2021. doi: 10.1021/acs.jcim.1c00203.

Hejia Geng, Zesen Huang, Haoyang Li, Wenbin Li, Koutian Wu, Zihan Zhou, Yuanbo Pang, Weihao Liu, Zigong Xu, Zhiping Li, Zongzheng Zhang, Chuanfei Dong, Jiankai Sun, Tianzhe Zheng, Fengyu Xie, Yue Ma, Yueheng Shi, Tong Xie, Zonglin Di, Xianrong Liu, Qucheng Gao, Yimin Liu, Jiaming Pan, Sheng Huang, Xiao-Han Ma, Lanqing Yuan, Zhenlin Zhu, Ziang Liu, Ziyang Xu, Junkai Wang, Kangkai Liang, Jiayi Xian, Zehong Zhao, Liuwei Xu, Jingxu Xie, Peijin Zhang, Qiang Gao, Chengyi Xing, Zhe Zhao, Xi Wang, Yaopeng Xing, Xing Meng, Zhenfei Yin, Yingcheng Wu, and Ling Yang. ScienceIDE: Turning world’s scientific codebase into agent learnable environments. arXiv preprint arXiv:2609.19134, 2026. URL https://arxiv.org/abs/2609.19134.

Paolo Giannozzi, Stefano Baroni, Nicola Bonini, Matteo Calandra, Roberto Car, Carlo Cavazzoni, Davide Ceresoli, Guido L. Chiarotti, Matteo Cococcioni, Ismaila Dabo, Andrea Dal Corso, Stefano de Gironcoli, Stefano Fabris, Guido Fratesi, Ralph Gebauer, Uwe Gerstmann, Christos Gougoussis, Anton Kokalj, Michele Lazzeri, Layla Martin-Samos, Nicola Marzari, Francesco Mauri, Riccardo Mazzarello, Stefano Paolini, Alfredo Pasquarello, Lorenzo Paulatto, Carlo Sbraccia, Sandro Scandolo, Gabriele Sclauzero, Ari P. Seitsonen, Alexander Smogunov, Paolo Umari, and Renata M. Wentzcovitch. QUANTUM ESPRESSO: a modular and open-source software project for quantum simulations of materials. Journal ofPhysics: Condensed Matter, 21(39):395502, 2009. doi: 10.1088/0953-8984/21/39/395502.

Xue Hu, Zewei Pan, Zeli Su, Zhou Liu, and Wentao Zhang. SA-Bench: Evaluating semantic alignment in LLM-based paper reproduction. arXiv preprint arXiv:2608.24252, 2026. URL https://arxiv.org/abs/ 2608.24252.

Qian Huang, Jian Vora, Percy Liang, and Jure Leskovec. MLAgentBench: Evaluating language agents on machine learning experimentation. arXiv preprint arXiv:2310.03302, 2023. URL https://arxiv.org/ab s/2310.03302.

Peter Jansen, Marc-Alexandre Cote, Tushar Khot, Erin Bransom, Bhavana Dalvi Mishra, Bodhisattwa Prasad Majumder, Oyvind Tafjord, and Peter Clark. DISCOVERYWORLD: A virtual environment for developing and evaluating automated scientific discovery agents. arXiv preprint arXiv:2406.06769, 2024. URL https://arxiv.org/abs/2406.06769.

Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Matt Deitke, Kiana Ehsani, Daniel Gordon, Yuke Zhu, Aniruddha Kembhavi, Abhinav Gupta, and Ali Farhadi. AI2-THOR: An interactive 3d environment for visual AI. arXiv preprint arXiv:1712.05474, 2017.

Laude Institute. Harbor: A containerized agent evaluation framework. Software repository, 2026. URL https://github.com/laude-institute/harbor. Accessed 2026-09-25.

Xuanlin Li, Kyle Hsu, Jiayuan Gu, Oier Mees, Karl Pertsch, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, Sergey Levine, Jiajun Wu, Chelsea Finn, Hao Su, Quan Vuong, and Ted Xiao. Evaluating real-world robot manipulation policies in simulation. In Pulkit Agrawal, Oliver Kroemer, and Wolfram Burgard (eds.), Proceedings of The 8th Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, pp. 3705–3728. PMLR, 2025.

Bodhisattwa Prasad Majumder, Harshit Surana, Dhruv Agarwal, Bhavana Dalvi Mishra, Abhijeetsingh Meena, Aryan Prakhar, Tirth Vora, Tushar Khot, Ashish Sabharwal, and Peter Clark. DiscoveryBench: Towards data-driven discovery with large language models. arXiv preprint arXiv:2407.01725, 2024. URL https://arxiv.org/abs/2407.01725.

Moonshot AI. Kimi K3. Kimi API Documentation, 2026. URL https://platform.kimi.com/docs/guide /kimi-k3-quickstart. Accessed 2026-09-25.

OpenAI. Codex. Product documentation, 2026a. URL https://developers.openai.com/codex/. Accessed 2026-09-25.

OpenAI. Models. OpenAI API Documentation, 2026b. URL https://developers.openai.com/api/docs /models. Accessed 2026-09-25.

Ardavan F. Oskooi, David Roundy, Mihai Ibanescu, Peter Bermel, J. D. Joannopoulos, and Steven G. Johnson. Meep: A flexible free-software package for electromagnetic simulations by the FDTD method. Computer Physics Communications, 181(3):687–702, 2010. doi: 10.1016/j.cpc.2009.11.008.

Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh, and Dhruv Batra. Habitat: A platform for embodied AI research. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9339–9347, 2019. doi: 10.1109/ICCV.2019.00943.

Zachary S. Siegel, Sayash Kapoor, Nitya Nadgir, Benedikt Stroebl, and Arvind Narayanan. CORE-Bench: Fostering the credibility of published research through a computational reproducibility agent benchmark. arXiv preprint arXiv:2409.11363, 2024. URL https://arxiv.org/abs/2409.11363.

Giulio Starace, Oliver Jaffe, Dane Sherburn, James Aung, Jun Shern Chan, Leon Maksin, Rachel Dias, Evan Mays, Benjamin Kinsella, Wyatt Thompson, Johannes Heidecke, Amelia Glaese, and Tejal Patwardhan. PaperBench: Evaluating AI’s ability to replicate AI research. arXiv preprint arXiv:2504.01848, 2025. URL https://arxiv.org/abs/2504.01848.

Qiming Sun, Matthew R. Hermes, Xiaojie Wu, Huanchen Zhai, Xing Zhang, Abdelrahman M. Ahmed, Juan José Aucar, Oliver J. Backhouse, Samragni Banerjee, Peng Bao, Nikolay A. Bogdanov, Kyle Bystrom, Frédéric Chapoton, Ning-Yuan Chen, Ivan Yu. Chernyshov, Helen S. Clifford, Sander Cohen-Janes, Zhi-Hao Cui, Yann D. Damour, Nike Dattani, Linus Bjarne Dittmer, Sebastian Ehlert, Janus Juul Eriksen, Francesco A. Evangelista, Simon A. Ewing, Ardavan Farahvash, Kevin Focke, Yang Gao, Kevin E. Gasperich, Nathan Gillispie, Jonas Greiner, Matthew R. Hennefarth, Jan Hermann, Christopher Hillenbrand, Joonatan Huhtasalo, Basil Ibrahim, Bhavnesh Jangid, Alireza Nejati Javaremi, Andrew J. Jenkins, Yu Jin, Daniel S. King, Derk Pieter Kooi, Jo S. Kurian, Henrik R. Larsson, Bryan Tak Gwong Lau, Seunghoon Lee, Susi Lehtola, Chenghan Li, Hao Li, Jiachen Li, Rui Li, Shuhang Li, Aleksandr O. Lykhin, Ankit Mahajan, Nastasia Mauger, Pablo Del Mazo-Sevillano, Jonathan Moussa, Kousuke

Nakano, Verena A. Neufeld, Linqing Peng, Hung Q. Pham, Peter Pinski, Pavel Pokhilko, Zhichen Pu, Yubing Qian, Stephen Jon Quiton, Wanja T. Schulze, Thais R. Scott, Aniruddha Seal, James D. Serna, James E. T. Smith, Kori E. Smyser, Terrence Stahl, Chong Sun, Kevin J. Sung, Egor Trushin, Shiv Upadhyay, Ethan A. Vo, Thijs Vogels, Shirong Wang, Tai Wang, Xiao Wang, Xubo Wang, Yuanheng Wang, Mark Williamson, Junjie Yang, Hong-Zhou Ye, Chia-Nan Yeh, Haiyang Yu, Jincheng Yu, Victor Wen-Zhe Yu, Chaoqun Zhang, Dayou Zhang, Yichi Zhang, Zijun Zhao, Zehao Zhou, Andrew J. Zhu, Tianyu Zhu, Timothy C. Berkelbach, Laura Gagliardi, Sandeep Sharma, Alexander Yu. Sokolov, and Garnet Kin-Lic Chan. The python simulations of chemistry framework: 10 years of an open-source quantum chemistry project. The Journal ofChemical Physics, 165(10):102502, 2026. doi: 10.1063/5.0337441.

Tencent. Tencent Hunyuan. Official model documentation, 2026. URL https://hunyuan.tencent.com/. Accessed 2026-09-25.

Minyang Tian, Luyu Gao, Shizhuo Dylan Zhang, Xinan Chen, Cunwei Fan, Xuefei Guo, Roland Haas, Pan Ji, Kittithat Krongchon, Yao Li, Shengyan Liu, Di Luo, Yutao Ma, Hao Tong, Kha Trinh, Chenyu Tian, Zihan Wang, Bohao Wu, Yanyu Xiong, Shengzhu Yin, Minhui Zhu, Kilian Lieret, Yanxin Lu, Genglin Liu, Yufeng Du, Tianhua Tao, Ofir Press, Jamie Callan, Eliu Huerta, and Hao Peng. SciCode: A research coding benchmark curated by scientists. arXiv preprint arXiv:2407.13168, 2024. URL https://arxiv.org/abs/2407.13168.

Yuru Wang, Lejun Cheng, Yuxin Zuo, Sihang Zeng, Bingxiang He, Che Jiang, Junlin Yang, Yuchong Wang, Kaikai Zhao, Weifeng Huang, Kai Tian, Zhenzhao Yuan, Jincheng Zhong, Weizhi Wang, Ning Ding, Bowen Zhou, and Kaiyan Zhang. NatureBench: Can coding agents match the published SOTA of nature-family papers? arXiv preprint arXiv:2606.24530, 2026. URL https://arxiv.org/abs/2606.24530.

Hjalmar Wijk, Tao Lin, Joel Becker, Sami Jawhar, Neev Parikh, Thomas Broadley, Lawrence Chan, Michael Chen, Josh Clymer, Jai Dhyani, Elena Ericheva, Katharyn Garcia, Brian Goodrich, Nikola Jurkovic, Holden Karnofsky, Megan Kinniment, Aron Lajko, Seraphina Nix, Lucas Sato, William Saunders, Maksym Taran, Ben West, and Elizabeth Barnes. RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts. arXiv preprint arXiv:2411.15114, 2024. URL https://arxiv.org/abs/2411.15114.

Zhipu AI. GLM-5.3-Flash/FlashX. Zhipu AI Documentation, 2026. URL https://docs.bigmodel.cn/cn/ guide/models/vlm/glm-5.3-flash. Accessed 2026-09-25.

Weiqing Zhou, Daye Zheng, Qianrui Liu, Denghui Lu, Yu Liu, Peize Lin, Yike Huang, Xingliang Peng, Jie J. Bao, Chun Cai, Zuxin Jin, Jing Wu, Haochong Zhang, Gan Jin, Yuyang Ji, Zhenxiong Shen, Xiaohui Liu, Liang Sun, Yu Cao, Menglin Sun, Jianchuan Liu, Tao Chen, Renxi Liu, Yuanbo Li, Haozhi Han, Xinyuan Liang, Taoni Bao, Zichao Deng, Tao Liu, Nuo Chen, Hongxu Ren, Xiaoyang Zhang, Zhaoqing Liu, Yiwei Fu, Maochang Liu, Zhuoyuan Li, Tongqi Wen, Zechen Tang, Yong Xu, Wenhui Duan, Xiaoyang Wang, Qiangqiang Gu, Fu-Zhi Dai, Qijing Zheng, Yang Zhong, Hongjun Xiang, Xingao Gong, Jin Zhao, Yuzhi Zhang, Qi Ou, Hong Jiang, Shi Liu, Ben Xu, Shenzhen Xu, Xinguo Ren, Lixin He, Linfeng Zhang, and Mohan Chen. ABACUS: An electronic structure analysis package for the AI era. The Journal ofChemical Physics, 163(19):192501, 2025. doi: 10.1063/5.0297563.

## A Task Package, Inventory, and Public Release

## A.1 Anatomy and Information Boundary of a Complete Task

We use the public 06-gmr-grating-fano-meep package to make the task contract concrete. The package is part of the twelve-task release at github.com/UniPat-AI/PaperBenchX. The task asks an agent to reproduce guided-mode resonances in a three-pair multilayer dielectric grating using Meep. The starter submission and scientific environment are agent-facing during a controlled trial, whereas the tests, rubric, judge prompt, and reference artifacts remain in the evaluator environment. All components are available for inspection.

gmr-grating-fano-meep/   
|-- task.toml   
|-- instruction.md   
-- environment/   
|-- Dockerfile   
|-- docker-compose.yam   
|-- init-workspace.sh   
|-- starter\_submission/   
|-- README.md   
|-- reproduce.sh   
\`-- src/   
-- paper/   
|-- paper.md   
|-- paper.pdf   
|-- addendum.md   
\`-- paper\_image/   
tests/   
|-- test.sh   
|-- evaluate.py   
|-- rubric.json   
|-- config/   
|-- judge\_prompt.md   
\`-- public\_contract.md   
|-- refs/   
\`-- src/

Components and evaluation-time visibility. Table A1 separates the materials available to an agent from evaluator-only resources. This boundary applies during evaluation; the complete demonstration package, including its evaluator resources, is now public for audit.

<table><tr><td>Component</td><td>Role</td><td>Agent</td><td>Evaluator</td></tr><tr><td>Paper and addendum</td><td>Scientific scope, measurements, and evidence requirements</td><td></td><td></td></tr><tr><td>Instruction</td><td>Goal, deliverables, and submission interface</td><td></td><td></td></tr><tr><td>Environment</td><td>Simulator, dependencies, and resource envelope</td><td></td><td></td></tr><tr><td>Starter submission</td><td>Editable source tree and replay entry point</td><td></td><td></td></tr><tr><td>Rubric</td><td>Hierarchical, independently scored scientific requirements</td><td>x</td><td></td></tr><tr><td>Judge prompt</td><td>Evidence-inspection and decision rules</td><td>X</td><td></td></tr><tr><td>Reference artifacts</td><td>Expert-generated logs, arrays, and validation records</td><td>x</td><td></td></tr></table>

Table A1: Task components and their visibility during evaluation. Evaluator-only components are included in the public demonstration package after evaluation.

Worked scientific contract. Table A2 summarizes the parts of the GMR task that determine what must be executed and what evidence must be retained. The complete protocol remains authoritative; this summary illustrates its structure rather than replacing it.

From contract to evidence leaves. The hidden rubric used during evaluation converts the public scientific requirements into independent leaves. Representative leaves check whether (i) the executed geometry contains the specified six layers and full-depth air slots (Modeling), (ii) clean replay regenerates genuine Meep runs, raw arrays, logs, and refinement controls (Execution), and (iii) power closure, pole fits, mode agreement, and convergence satisfy the disclosed scientific conditions (Validation). A missing item affects the leaves that depend on it rather than silently imposing a global zero.

<table><tr><td>Field</td><td>Public contract</td></tr><tr><td>Scientific target</td><td>Guided-mode resonances of a three-pair multilayer dielectric grating in Meep</td></tr><tr><td>Cases</td><td>Unetched control plus full-depth air-slot widths of 0.030, 0.050, 0.070, and 0.150 µm</td></tr><tr><td>Required evidence</td><td>Native spectra and logs, matched reference/device fluxes, complex reflected amplitudes resonance fits, independent decay modes, and refinement records</td></tr><tr><td>Replay interface</td><td>No-argument bash/home/submission/reproduce.sh, executed offline</td></tr><tr><td>Compute envelope</td><td>Meep 1.17.1, 8 CPUs, 32 GiB RAM; 7 hours for the agent, 5 hours for native calculation, and 6 hours for replay and review</td></tr><tr><td>Failure reporting</td><td>Missing, dark, overlapping, or non-identifiable features remain unresolved and cannot be replaced with paper values or invented poles</td></tr></table>

Table A2: Condensed public contract for the GMR demonstration task.

Evaluator policy. Only evidence regenerated by clean replay is admissible. The evaluator accepts equivalent documented data layouts and unit conversions, but does not infer missing units, phase, provenance, or scientific quantities from a report. Scientific failures receive zero on the affected leaves; evaluator read or tool failures instead withhold the judgment for repair and re-evaluation. Each leaf is judged against its stated requirement without targeting a model-level score.

Complete public record. The pinned 06-gmr-grating-fano-meep directory contains the full agent instruction, scientific addendum, environment definition, judge prompt, rubric, and reference materials.

## A.2 Complete Benchmark Inventory

Table A3 follows the 93-task result set used for the reported model results. The numeric prefix identifies the research area, and “yes” marks the public demo in that area.

Table A3: Complete 93-task benchmark inventory used in the reported evaluation.
<table><tr><td>ID</td><td>Task</td><td>Public demo</td></tr><tr><td colspan="3">01 RF antennas and arrays</td></tr><tr><td>1-1</td><td>5g-smartphone-eight-element-mimo</td><td></td></tr><tr><td>1-2</td><td>cpw-quad-port-uwb-mimo</td><td>yes</td></tr><tr><td>1-3</td><td>dual-band-wlan-miniaturized-patch</td><td></td></tr><tr><td>1-4</td><td>f-stub-uwb-mimo</td><td></td></tr><tr><td>1-5</td><td>inkjet-flexible-dual-band-cp-patch</td><td></td></tr><tr><td>1-6</td><td>metasurface-broadband-cp-patch</td><td></td></tr><tr><td>1-7</td><td>parasitic-decoupling-two-monopole-mimo</td><td></td></tr><tr><td>1-8</td><td>tm30-high-gain-cp-patch</td><td></td></tr><tr><td>1-9</td><td>uwb-fractal-dgs-antenna</td><td></td></tr><tr><td>1-10</td><td>wideband-neutralization-line-mimo</td><td></td></tr><tr><td colspan="3">02 RF metasurfaces and scattering</td></tr><tr><td>2-1</td><td>anisotropic-coding-diffusion-metasurface</td><td>yes</td></tr><tr><td>2-2</td><td>checkerboard-ebg-rcs-reduction</td><td></td></tr><tr><td>2-3</td><td>eight-circular-sector-wide-angle-absorber</td><td></td></tr><tr><td>2-4</td><td>fish-shaped-dual-function-polarization-converter</td><td></td></tr><tr><td>2-5</td><td>pcm-entropy-ga-coding-metasurface</td><td></td></tr><tr><td>2-6</td><td>uwb-pcm-checkerboard-rcs</td><td></td></tr><tr><td colspan="3">03 RF filters and frequency-selective surfaces</td></tr><tr><td>3-1</td><td>dual-passband-angular-stable-fss</td><td>yes</td></tr><tr><td>3-2</td><td>dual-stopband-miniaturized-fss</td><td></td></tr><tr><td>3-3</td><td>embedded-window-rasorber</td><td></td></tr><tr><td>3-4</td><td>fourth-order-siw-bpf</td><td></td></tr><tr><td>3-5</td><td>quad-mode-wideband-microstrip-filter</td><td></td></tr><tr><td colspan="3">04 Integrated photonic devices</td></tr><tr><td>4-1</td><td>dualmode-tapered-coupler</td><td></td></tr><tr><td>4-2</td><td>eaton-lens-multimode-bend</td><td></td></tr><tr><td>4-3</td><td>heterogeneous-ln-sin-transitions-splitter</td><td></td></tr><tr><td>4-4</td><td>lnoi-multimode-bend</td><td></td></tr><tr><td>4-5</td><td>1noi-pol-insensitive-edge-coupler</td><td></td></tr><tr><td>4-6 4-7</td><td>luneburg-pcw-coupler mdm-sip-neff-mmi-design</td><td></td></tr><tr><td>4-8</td><td>mfe-multimode-crossing-soi</td><td></td></tr><tr><td>4-9</td><td>mmi-beamsplitter-soi</td><td></td></tr><tr><td></td><td></td><td></td></tr></table>

Continued on next page

Table A3 (continued)
<table><tr><td>ID</td><td>Task</td><td></td><td>Public demo</td></tr><tr><td>4-10</td><td>sin-bent-dc-splitter</td><td></td><td></td></tr><tr><td>4-11</td><td>soi-submicron-crossing</td><td></td><td></td></tr><tr><td>4-12</td><td>soi-swg-mmi-splitter</td><td></td><td>yes</td></tr><tr><td>4-13</td><td>swg-anisotropic-phase-shifter</td><td></td><td></td></tr><tr><td>4-14</td><td>swg-mode-exchanger-yjunction</td><td></td><td></td></tr><tr><td>05 Photonic crystals and topological photonics</td><td></td><td></td><td></td></tr><tr><td>5-1</td><td>half-w1-gainp-slow-light</td><td></td><td></td></tr><tr><td>5-2</td><td>hoti-quadrupole-corner</td><td></td><td></td></tr><tr><td>5-3</td><td>klein-tunneling-honeycomb-phc</td><td></td><td></td></tr><tr><td>5-4</td><td></td><td></td><td></td></tr><tr><td>5-5</td><td>1n-pti-bandstructure-edge-state pc-line-defect-self-collimation</td><td></td><td></td></tr><tr><td>5-6</td><td>phc-fano-cavity-waveguide-gain</td><td></td><td></td></tr><tr><td>5-7</td><td>phc-superlens-imaging</td><td></td><td></td></tr><tr><td>5-8</td><td>phc-w1-slowlight-glide</td><td></td><td>yes</td></tr><tr><td>5-9</td><td>pt-bragg-unidirectional-invisibility</td><td></td><td></td></tr><tr><td>5-10</td><td></td><td></td><td></td></tr><tr><td>5-11</td><td>self-collimation-hex-router</td><td></td><td></td></tr><tr><td>5-12</td><td>ssh-photonic-lattice-edge</td><td></td><td></td></tr><tr><td>5-13</td><td>topo-valley-ring-channel-filter</td><td></td><td></td></tr><tr><td>5-14</td><td>valley-bend-domainwall</td><td></td><td></td></tr><tr><td></td><td>zak-interface-1d-phc</td><td></td><td></td></tr><tr><td></td><td>06 Optical metasurfaces, gratings, and plasmonics</td><td></td><td></td></tr><tr><td>6-1</td><td>al-wire-grid-polarizer-pitch-sweep</td><td></td><td></td></tr><tr><td>6-2</td><td>bowtie-aperture-resonance</td><td></td><td>yes</td></tr><tr><td>6-3 6-4</td><td>fano-invisibility-dielectric-rod</td><td></td><td></td></tr><tr><td>6-5</td><td>gmr-grating-fano</td><td></td><td></td></tr><tr><td>6-6</td><td>integrated-metasurface-nanopillar-lookup</td><td></td><td></td></tr><tr><td>6-7</td><td>metagrating-anomalous-deflector</td><td></td><td></td></tr><tr><td></td><td>photonic-nanojet-twin-hook</td><td></td><td></td></tr><tr><td>6-8 6-9</td><td>plasmonic-dimer-hotspot</td><td></td><td></td></tr><tr><td>6-10</td><td></td><td></td><td></td></tr><tr><td></td><td>quasi-bic-metasurface-q silicon-grating-antenna</td><td></td><td></td></tr><tr><td>07 Optical computing and signal processing</td><td></td><td></td><td>yes</td></tr><tr><td>7-1</td><td>brewster-spatial-differentiator</td><td></td><td></td></tr><tr><td>7-2</td><td>donn-i1-iris-classification</td><td></td><td></td></tr><tr><td>7-3 7-4</td><td>gmr-2d-edge-detector</td><td></td><td></td></tr><tr><td>7-5</td><td>optical-slab-spatial-integrator</td><td></td><td></td></tr><tr><td></td><td>phc-xor-logic-gate</td><td></td><td></td></tr><tr><td>7-6</td><td>swg-spatial-differentiator</td><td></td><td></td></tr><tr><td></td><td>08 Optical resonators, filters, and tunable devices</td><td></td><td>yes</td></tr><tr><td>8-1</td><td></td><td></td><td></td></tr><tr><td>8-2</td><td>clc-1d-resonator</td><td></td><td></td></tr><tr><td>8-3</td><td>lnoi-athermal-mwg-filter</td><td></td><td></td></tr><tr><td>8-4</td><td>sin-microring-optofluidic soi-swg-pump-filter</td><td></td><td></td></tr><tr><td>8-5</td><td>vdw-mems-swx-switch</td><td></td><td></td></tr><tr><td>09 Computational chemistry</td><td></td><td></td><td></td></tr><tr><td>9-1</td><td></td><td></td><td></td></tr><tr><td>9-2</td><td>ketene-water-chain ni13c-ni2-kie-discrimination</td><td></td><td></td></tr><tr><td>9-3</td><td>nickel-catellani</td><td></td><td></td></tr><tr><td>9-4</td><td>nitrogen-deletion-dynamics</td><td></td><td>yes</td></tr><tr><td>9-5 9-6</td><td>pimarenyl-bifurcation</td><td></td><td></td></tr><tr><td>9-7</td><td>ptrh-n2-activation</td><td></td><td></td></tr><tr><td>9-8</td><td>singh-sert-conformational-docking</td><td></td><td></td></tr><tr><td></td><td>solvent-bifurcation-dynamics</td><td></td><td></td></tr><tr><td></td><td>10 Computational materials science</td><td></td><td>yes</td></tr><tr><td>10-2</td><td>gong-be-elastic</td><td></td><td></td></tr><tr><td>10-3 10-5</td><td>haastrup-mos2-bands</td><td></td><td></td></tr><tr><td>10-6</td><td>choudhary-sic-properties-pbesol-hse06 tran-al-surface-energy</td><td></td><td></td></tr><tr><td></td><td>11 Computational biology and bioinformatics</td><td></td><td></td></tr><tr><td>11-1</td><td>horlbeck-genetic-interactions</td><td></td><td>yes</td></tr><tr><td>11-2</td><td>machina-metastatic-migration</td><td></td><td></td></tr><tr><td>11-3</td><td>moschandrea-mito-lipid-enterocyte</td><td></td><td></td></tr><tr><td>11-4</td><td>nichenet-ligand-target</td><td></td><td></td></tr><tr><td>11-5</td><td>ooka-wsme-1-folding</td><td></td><td></td></tr><tr><td>12 Robotics and embodied AI</td><td></td><td></td><td></td></tr><tr><td>12-1</td><td>embodiedbench</td><td></td><td>yes</td></tr><tr><td>12-2</td><td>mapgpt-r2r</td><td></td><td></td></tr><tr><td>12-3</td><td>octo</td><td></td><td></td></tr><tr><td>12-4</td><td>poliformer</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>12-6 12-8</td><td>rt1x tracevla</td><td></td><td></td></tr></table>

## A.3 Public Demonstration Tasks

Table A4 lists the exact package identifiers and source publications for the twelve packages in the public GitHub release. Raw identifiers are retained because they map directly to the released task directories at commit 7abb21c4.

Table A4: The twelve public demonstration tasks and their source publications.
<table><tr><td>Area</td><td>Task</td><td>Source publication and DOI</td></tr><tr><td>rays</td><td>Radio-frequency antennas and ar- 01-cpw-quad-port-uwb-mimo</td><td>CPW Fed Compact UWB 4-Element MIMO Antenna with High Isolation; 10.3390/s21082688</td></tr><tr><td>scattering</td><td>etasurface</td><td>Radio-frequency metasurfaces and 02-anisotropic-coding-diffusion-m Broadband diffusion metasurface based on a single anisotropic element and optimized by the Simulated Annealing algorithm; 10.1038/srep23896</td></tr><tr><td>Radio-frequency filters frequency-selective surfaces</td><td>SS</td><td>and 03-dual-passband-angular-stable-f A Dual-Passband Frequency Selective Surface with High Angular Stability and Polarization Insensitivity; 10.3390/mi15060690</td></tr><tr><td>Integrated photonic devices</td><td>eep</td><td>04-swg-anisotropic-phase-shifter-m Ultra-broadband nanophotonic phase shifter based on subwavelength meta- material waveguides; 10.1364/PRJ.373223</td></tr><tr><td>photonics</td><td>Photonic crystals and topological 05-pt-bragg-unidirectional-invisib ility-meep</td><td>Unidirectional Invisibility induced by PT-Symmetric Periodic Structures; 10.1103/PhysRevLett.106.213901</td></tr><tr><td>plasmonics</td><td>Optical metasurfaces, gratings, and 06-gmr-grating-fano-meep</td><td>Controlling Fano resonances in multilayer dielectric gratings towards opti- cal bistable devices; 10.1038/s41598-018-34787-9</td></tr><tr><td>cessing</td><td>r-meep</td><td>Optical computing and signal pro-07-brewster-spatial-differentiato Analog computing by Brewster effect; 10.1364/OL.41.003467</td></tr><tr><td>Optical resonators, filters, and tun- 08-clc-1d-resonator-meep able devices</td><td></td><td>Photonic eigenmodes and transmittance of finite-length 1D cholesteric liquid crystal resonators; 10.1038/s41598-023-43912-2</td></tr><tr><td>Computational chemistry</td><td>09-pimarenyl-bifurcation-v2</td><td>Post-transition State Bifurcations in Reactions That Form Pi- marenyl Cation Diastereomers—Implications for Terpene Biosynthesis;</td></tr><tr><td>Computational materials science</td><td>10-haastrup-2018-mos2-bands</td><td>10.1021/jacs.5c16476 The Computational 2D Materials Database: high-throughput modeling and</td></tr><tr><td></td><td>Computational biology and bioin-11-horlbeck-2018-genetic-interacti</td><td>discovery of atomically thin crystals; 10.1088/2053-1583/aacfc1 Mapping the Genetic Landscapeof Human Cells;</td></tr><tr><td>formatics</td><td>ons</td><td>10.1016/j.cell.2018.06.010</td></tr><tr><td>Robotics and embodied Al</td><td>12-mapgpt-r2r</td><td>MapGPT: Map-Guided Prompting with Adaptive Path Planning for Vision- and-Language Navigation; 10.18653/v1/2024.acl-long.529</td></tr></table>

## A.4 Platforms and Resource Envelopes

Table A5 reports the limits declared by the public task packages. The limits are configuration ceilings rather than measured runtimes. An empty GPU entry means that no GPU count is declared.

<table><tr><td>ID</td><td>Execution stack</td><td>Agent h</td><td>Verifier h</td><td>CPU</td><td>RAM GiB</td><td>Disk GiB</td><td>GPU</td></tr><tr><td>1-2</td><td>Ansys HFSS</td><td>24</td><td>25</td><td>32</td><td>128</td><td>200</td><td></td></tr><tr><td>2-1</td><td>Ansys HFSS</td><td>72</td><td>73</td><td>32</td><td>128</td><td></td><td>200</td></tr><tr><td>3-1</td><td>Ansys HFSS</td><td>48</td><td>49</td><td>32</td><td>128</td><td></td><td>300</td></tr><tr><td>4-13</td><td>Meep</td><td>7</td><td>6</td><td>8</td><td>32</td><td></td><td>50</td></tr><tr><td>5-9</td><td>Meep</td><td>5</td><td>5</td><td>8</td><td>32</td><td>50 50</td><td></td></tr><tr><td>6-4</td><td>Meep</td><td>7</td><td>6</td><td>8</td><td>32</td><td></td><td></td></tr><tr><td>7-1</td><td>Meep</td><td>6</td><td>5</td><td>8</td><td>32</td><td></td><td></td></tr><tr><td>8-1</td><td>Meep</td><td>7</td><td>7</td><td>8</td><td>32</td><td>50</td><td></td></tr><tr><td>9-5</td><td>PySCF/GPU4PySCF</td><td>7</td><td>7</td><td>8</td><td>32</td><td>50 50</td><td>1</td></tr><tr><td>10-3</td><td>Quantum ESPRESSO</td><td>7</td><td>7</td><td>8</td><td>32</td><td></td><td></td></tr><tr><td>11-1</td><td>Python scientific stack</td><td>6</td><td>6</td><td>8</td><td>32</td><td>50</td><td></td></tr><tr><td>12-2</td><td>Local Qwen2.5-VL/vLLM</td><td>18</td><td>7</td><td>8</td><td>32</td><td>100</td><td>1</td></tr></table>

Table A5: Execution stacks and declared resource limits for the twelve public task packages in Table A4

![](images/97b4f68f0d70d46623090a523425cead4eb21f6e4d19449222d766a981407661.jpg)  
Figure 7: Representative outputs from domain-native environments. Panels show solver fields and spectra, numerical analysis outputs, and embodied-task scenes produced by scientific workflows in the benchmark.

## B Evaluation and Verification Protocol

This appendix expands the evaluation protocol in Section 3.4.

## B.1 Scope, Resources, and Trial Records

Evaluation scope. The benchmark contains 93 tasks and 3,168 scored rubric leaves. All 10 configurations are evaluated on the same 93 tasks, with the fallback and missing-outcome policy defined in Section 4.1. The Partial score is the mean task score over this fixed denominator. The Full rate counts tasks with an aggregate score strictly greater than 0.9; ties in reported rankings are resolved by competition rank on unrounded scores.

Resource envelopes. Each task declares CPU, memory, storage, and wall-clock limits appropriate to its scientific workflow. Across the benchmark, agent time limits range from 4 to 72 hours, with a median of 7 hours. Clean-replay and evaluator limits are specified separately from the agent’s development budget. Exact limits for the twelve public demonstration tasks appear in Table A5.

Trial records. Each trial is linked to a task version, agent configuration, submitted package, replay status, and per-leaf judgments. Agent runtime, clean-replay runtime, and grading time are recorded separately. Missing telemetry remains missing rather than being imputed, and infrastructure failures are distinguished from an agent’s failure to produce an executable submission.

## B.2 Expert Annotation and Release Review

For each task, a domain expert maps the target claim to its source passage, figure, or table; fixes the model configuration, experimental conditions, observables, acceptable methodological alternatives, and required evidence; and converts these requirements into independently scorable rubric leaves. The public instruction, addendum, environment, and starter submission must agree on the task scope and replay interface. Private reference workflows, reference artifacts, and grading tolerances are withheld from agents, while values reported in the source paper remain visible.

The private reference workflow is first executed to establish feasibility and then rerun from a clean copy of the final environment through the same entry point used for submissions. Calibration workflows representing successful, partial, and invalid reproductions are scored to check that supported progress receives credit and unsupported claims do not. Two independent domain experts then review the scientific scope, fairness, reproducibility, evidence requirements, and scoring behavior. Changes to the task contract, environment, rubric, or evaluator require the affected downstream checks to be repeated before release.

## B.3 Clean Replay and Evidence Grading

Submission snapshot. The canonical entry point is bash/home/submission/reproduce.sh. After delivery, the verifier records the submitted source and required inputs, removes generated outputs and mutable execution state, and reconstructs the workspace from the submitted package. This prevents cached or pre-existing result files from being treated as evidence from the replay.

Controlled execution. The package is executed with private evaluation resources inaccessible, outbound network access disabled, and task-specific limits enforced. Exit status, timing, source hashes, and available execution provenance are recorded. These records establish how an artifact was produced; they do not by themselves establish that its scientific content is correct.

Frozen evidence. Artifacts regenerated during replay are frozen as a read-only evidence set. Deterministic checks cover directly testable properties such as required paths, parseable formats, units, finite values, numerical consistency, sampling grids, and task-specific tolerances. Simulator logs, raw arrays, trajectories, convergence records, derived metrics, and visualizations are admitted only when they can be associated with the current replay and the relevant task variant.

Leaf judgments and aggregation. Each rubric leaf is graded from its criterion, effective weight, relevant replay artifacts, and deterministic check results. Requirements needing domain interpretation are judged by the same GPT-5.6-sol evaluator used for every reported configuration. Candidate reports may identify evidence but cannot replace absent or contradictory artifacts. With leaf score $s _ { i }$ and effective weight $w _ { i } ,$ the task score is $\begin{array} { r } { R = \sum _ { i } w _ { i } s _ { i } , } \end{array}$ , where $\textstyle \sum _ { i } w _ { i } = 1$ . A missing artifact affects only the leaves that depend on it; replay, infrastructure, and evaluator failures are recorded separately.

## B.4 Regression and Adversarial Checks

Task-specific regression tests implement the calibration step described in Section 3.3. They pair valid controls with deliberately corrupted or incomplete submissions to test whether the evaluator accepts supported evidence, rejects unsupported claims, and confines a failure to the rubric leaves that depend on it. Table A6 summarizes the principal check families used across the released verifiers.

The twelve public task verifiers exercise different parts of this contract, including numerical parsing, native solver evidence, provenance, cached-output removal, and evaluator-owned records. Promptinjection regression cases are included to check whether instructions embedded in a candidate report, log, or result field can alter the rubric or override a deterministic check.

## C Additional Result Diagnostics

<table><tr><td>Check family</td><td>Perturbation and expected behavior</td></tr><tr><td>Contract and parser integrity</td><td>Preserve rubric hashes and complete leaf mappings; reject duplicate keys, nonfi- nite values, and invalid numerical grids while accepting documented equivalent formats.</td></tr><tr><td>Native evidence and variant identity</td><td>Reject incomplete solver logs, mismatched setup names, explicit non-convergence, and copied variant outputs; accept a shared solve only when the task contract permits it and the variants remain identifiable.</td></tr><tr><td>Evidence boundary</td><td>Exclude cached outputs and links outside permitted roots; keep private references and evaluator credentials inaccessible; fail closed when network isolation cannot be established.</td></tr><tr><td>Completion and provenance</td><td>Give no completion credit to an empty submission; require delivery records, input hashes, and preprocessing conventions to agree with the task contract.</td></tr><tr><td>Local failure accounting</td><td>Corrupt one input, metric, or figure panel and verify that only dependent rubric</td></tr><tr><td>Interactive-task contract</td><td>leaves lose credit. Use evaluator-owned episode records and protocol identifiers; do not allow can- didate summaries to replace required trajectory or visual evidence.</td></tr></table>

Table A6: Regression cases used to check evidence admission and local scoring. Valid controls accompany invalid fixtures so that permissible variation is not mistaken for unsupported evidence.

## C.1 Task Completion

Figure 8 resolves the aggregate Full rate at the task level. For each task, it reports the best score attained by any of the ten configurations and the number of configurations exceeding the operational 90% threshold. The figure makes the concentration of complete reproductions explicit: 71 tasks are not completed by any configuration.

![](images/c174177aff0ade6962a18dbacf826d280437b3c80b90a135703423b47b69a9f1.jpg)  
Figure 8: Task-level completion landscape. (a) Maximum Partial score attained on each of the 93 tasks, ordered by research area. (b) Number of configurations exceeding 90% on the same task. The horizontal line marks the operational full-reproduction threshold; 71 tasks have no configuration above it.

## C.2 Cross-Simulator Trace Case Studies

This section expands the trace examples discussed in Section 5. Figure 9 previews the stage-level contrast developed in the three cases.

![](images/ef293ff0a5e1cb8a89e9bff31d881cdc2d191730737e813cdca3e65a05cbd1b4.jpg)

![](images/a62bd93eb009838dea30afa03919350f45517b75e2dc89ecddc9b2bf8da1a438.jpg)

![](images/2a46b26ca065b7db766f84ee6f57b901c928e9642fa605594626cf201b30c4c7.jpg)  
Figure 9: Stage profiles in three trace case studies. Bars report Modeling, Execution, and Validation scores; diamonds report the aggregate task score.

Eight-element smartphone MIMO (Task 1-1). The task requires two independent eight-port HFSS variants and evaluates geometry, solver provenance and convergence, matching, isolation, efficiency, ECC, gain, radiation patterns, and surface currents. GPT-6-Astra scores 43.5% in 187 turns; Opus 5 scores 41.3% in 450 turns. Both replay native HFSS workflows and recover several isolation, ECC, gain, and current criteria. Opus 5 additionally records five and seven adaptive passes, final maximum ∆S values of 0.0063 and 0.0078, and independent meshes exceeding half a million tetrahedra.

Neither submission reproduces the central four-port matching target. Reflection minima and the values at 3.45 GHz miss their required windows, and the baseline and efficiency conditions fail. GPT-6-Astra also represents a critical length as a reference parameter rather than using it in the model. Native execution and agreement on secondary observables therefore do not establish the paper’s impedance and efficiency claims. DeepSeek-V4.1-Flash provides a complementary efficiency example: 1,024 turns yield a score of 15.3%.

F-stub UWB MIMO (Task 1-4). The task evaluates plain, I-stub, and F-stub variants over a 501-point sweep. As reported in the main text, Fable 5 scores 28.9% in 159 turns, compared with 27.4% for GPT-6- Astra in 464 turns and 18.1% for HY3 in 390 turns. Fable 5 completes all three native variants and recovers evidence for port reciprocity, radiation patterns, and diversity gain, but none of the three submissions reproduces the principal S-parameter response. This comparison supports the limited conclusion in Section 5: more interactions do not necessarily improve reproduction when the model or parameterization remains incorrect.

Polarization-insensitive LNOI edge coupler (Task 4-5). This workflow spans 39 Lumerical variants: FDE facet modes, TE/TM Gaussian overlaps, geometry and alignment variants, and EME sweeps over four coupling lengths. Fable 5.1 scores 73.1% in 111 turns, followed by Fable 5 at 72.9% in 212 turns. Fable 5.1, Opus 5, and GPT-6-Astra each receive full Execution credit, confirming that complete replay is achievable in this environment.

The remaining errors are scientific. Fable 5.1 exceeds the allowed dual-tip TE overlap, Fable 5 exceeds the facet-PDL limit, and GPT-6-Astra obtains an EME800 TM coupling of only 0.0474 despite complete replay artifacts. Kimi-K3 has the highest Modeling score (83.6%), but its EME output is NaN. The traces show why intermediate checks matter: one facet solve, polarization and normalization, and one EME case should be validated before the full sweep. NaNs, missing modes, or incompatible matrix dimensions should stop downstream aggregation.

Synthesis. The cases explain why Modeling, Execution, and Validation are reported separately; they are diagnostic examples rather than a cross-task ranking.