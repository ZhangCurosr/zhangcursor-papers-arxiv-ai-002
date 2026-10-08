# RSI-Forge: From Research Papers to Environments for Recursive Self-Improvement

Renxiong Wang1, Darvin Yi1, Abril Herrlein1, Anas Mahmoud1, Advait Gosai1, Lisiman Hua1, MohammadHossein Rezaei1, Xingang Guo1, Anisha Gunjal1, Utkarsh Tyagi1, David J. Lee1, Minglai Yang1, Haris Riaz1, Chenguang Wang1,2, Huaxiu Yao3, Daniel Yue Zhang1, Aakash Sabharwal1, Tong Zhao1, Yunzhong He1

1Scale AI, 2University of California, Santa Cruz, 3University of North Carolina at Chapel Hill

{renxiong.wang,darvin.yi,tong.zhao,yunzhong.he}@scale.com

## Abstract

Environments are the foundation of recursive self-improvement: they provide the problems agents work on and the feedback used to evaluate progress. Yet constructing challenging research environments with reliable evaluation still depends on domain experts, limiting their scale and disciplinary coverage. We introduce RSI-Forge, a multi-agent pipeline that turns published papers into executable environments for self-improvement. Three agents coordinate construction, reproduction, and review to produce tasks with automated evaluators; each paper's method is independently reimplemented to establish a baseline score. We present 210 environments across 18 fields, including 90 reviewed by independent human domain experts. Both experts and agent judges give high ratings to the potential for improving the provided starting solutions and the evaluators' ability to distinguish solution quality, whereas experts are more critical of shortcut resistance, faithfulness to the source paper, and whether a single idea can exhaust a task. To validate their use for repeated improvement, we evaluate four models over 3 successive attempts on 120 environments, with each attempt inheriting prior code and notes while model weights remain fixed. At least one model improves after the first attempt in 84% of environments. Models also outperform the reproduced paper methods in 68 of the 120 environments, demonstrating room for gains beyond these baselines. Transcript analysis identifies work beyond parameter tuning in 95% of these successful attempts. Analysis of the resulting trajectories shows that models scoring lower on these tasks explore less, more often accept gains smaller than the reported standard error, and rely more heavily on tuning to the development set. RSI-Forge provides a scalable approach to constructing research environments for training and evaluating self-improving agents.

## 1 Introduction

Recursive self-improvement (RSI) is a process in which an agent improves its own capabilities and uses those improvements to guide further changes [5]. Such agents could accumulate useful strategies and tackle increasingly difficult problems with less human intervention. Developing this capability requires environments that provide challenging problems and reliable feedback on attempted improvements. These environments supply experience for training and a basis for evaluation, making their availability and diversity central to the development of RSI at scale.

![](images/4d3b43fc9a2c06850ea45517d18de28572d112ff1ed073b4a7ffe958dc77345c.jpg)  
Figure 1. Overview of RSI-Forge and the information available to each agent. Failed checks lead to repair or rejection before an environment is accepted. The output count includes all 210 environments. The performance bars report median scores of final solutions after 3 successive attempts on 120 environments sampled from the main construction run. Scores are normalized so that the starting solution is 0 and the target score is 1 (Eq. (1)).

Constructing suitable RSI environments is particularly difficult for research tasks. They must pose meaningful problems, allow continued improvement, and evaluate solutions automatically while distinguishing genuine gains from measurement noise and scoring shortcuts. Formulating such tasks requires domain knowledge, and implementing reliable evaluators adds a separate engineering challenge. Existing benchmarks draw on repositories and competitions with established tests or graders [3, 12], or rely on tasks and evaluation rubrics written by experts [21, 24]. These approaches provide valuable environments, but their scale and domain coverage remain tied to available evaluation resources and expert effort. This raises a central question: how can we automatically construct RSI environments at scale?

We introduce RSI-Forge, a multi-agent pipeline that constructs executable RSI environments from published papers (Fig. 1). Papers provide concrete research questions, methods, and measured results that can guide environment design. An author agent develops the task, starting solution, and evaluator. A reproducer agent independently implements the paper's method using only the paper and solverfacing materials, establishing the reproduced baseline. A reviewer agent checks that the task can be understood from its instructions and that the evaluator measures the stated objective, requesting repairs when needed. The agents have distinct responsibilities and access to information, allowing reproduction and review to assess the environment separately from the author agent's construction.

Using this pipeline, we construct 210 environments across 18 fields. To assess their quality, three agent judges evaluate the environments using a common rubric, and independent human domain experts review 90 environments within their fields of expertise. The rubric assesses whether tasks pose meaningful research problems, offer opportunities for improvement, and provide reliable evaluation. Experts assign their highest ratings to the potential for improving the starting solutions and the evaluators' ability to distinguish solution quality. They are more critical than agent judges of shortcut resistance, fidelity to the source paper, and whether a single idea can exhaust a task (Fig. 2).

To assess whether these environments support repeated improvement, we evaluate four solver models over 3 successive attempts on 120 of the environments. Each attempt inherits prior code and notes while model weights remain fixed. At least one model improves after the first attempt in 84% of environments, and the reproduced baseline is surpassed in 68 of the 120 environments. Transcript analysis identifies work beyond parameter tuning in 95% of these successful attempts. Analysis of the resulting trajectories further distinguishes how models explore alternatives, assess uncertain gains, and select solutions. Together, these results show that RSI-Forge can automatically construct research environments that support measurable improvement over successive attempts. Our contributions are:

1. An automated RSI environment construction pipeline. We introduce RSI-Forge, which turns published papers into executable environments.

2. A corpus spanning diverse research fields. We construct 210 environments across 18 fields, including 90 independently reviewed by human domain experts.

3. Validation through repeated improvement. We empirically demonstrate gains over successive attempts and beyond reproduced baselines, with trajectory analysis revealing differences in how models explore alternatives and assess changes.

## 2 Related Work

Self-improving agents. Prior work studies improvement at several levels. Self-Refine iteratively revises model outputs using feedback, while Reflexion retains reflections from earlier attempts to inform subsequent decisions [15, 18]. Other approaches optimize prompts, workflows, agent designs, harnesses, or source code [7, 8, 13, 27–30]. These approaches investigate how agents improve their solutions and procedures. Our work addresses the construction of environments that supply challenging problems and measurable feedback for such improvement.

Automated task and environment generation. Research on learning tasks spans curriculum learning, adaptive environment design, and autonomous task selection [2, 4, 11, 23]. Absolute Zero and R-Zero generate tasks as part of a learning process that couples task proposal with solving [9, 31]. AgentSynth synthesizes tasks and trajectories for computer-use agents [25], while EnvScaler and EnvFactory construct executable environments for tool interaction [20, 26]. SPADE further couples environment generation with learning: a model designs executable environments and learns by acting within them [14]. For terminal agents, TermiGen synthesizes verifiable environments through iterative multi-agent refinement [33], while Fan et al. [6] progressively increase environment difficulty to sustain learning signals. RSI-Forge shares the goal of expanding environment supply through automation, focusing on research problems drawn from published papers across diverse fields.

Research and engineering benchmarks. Existing benchmarks obtain tasks and evaluation procedures from several sources. SWE-bench uses repository issues and tests, while MLE-bench adapts machinelearning competitions [3, 12]. Related benchmarks assess machine-learning experimentation and computational reproducibility [10, 19]. PaperBench evaluates paper replication using rubrics developed with the source papers' authors [21]; RE-Bench provides expert-designed research engineering environments [24]; and Terminal-Bench provides curated terminal tasks with human-written solutions and verification tests [16]. RSI-Exam focuses specifically on repeated improvement in executable research tasks [1]. Closer to automated construction from papers, Theiler et al. [22] translate published methods into a shared benchmarking framework for machine health intelligence. RSI-Forge constructs research tasks and evaluators across fields, with independently reproduced baselines for measuring further improvements.

## 3 RSI-Forge

RSI-Forge draws research problems and reference methods from published papers. It converts each selected problem into an executable environment and independently reimplements the paper's method there to establish a measured baseline. A reviewer agent checks that the instructions clearly describe the task and the evaluator measures the intended objective, requesting repairs when needed.

## 3.1 Environment Format

We start by defining an RSI environment as an executable research task with the resources needed to develop and evaluate solutions. Each environment specifies a research objective, the required solution interface, and a compute budget. It provides a runnable starting solution, code for generating task data, and an automated evaluator. We refer to the agent attempting the task as the solver. The solver can modify or replace the starting solution and test its changes on a development set. A separate held-out test set is reserved for grading submitted solutions.

Each environment includes a reproduced baseline: the source paper's method independently reimplemented and evaluated under the environment's task definition and resource limits. Its measured score provides a reference for assessing improvements beyond that method. The source paper, reproduced implementation, and test data are not supplied as solver inputs.

Each environment also defines a target score based on a task-specific performance bound or an idealized solution, such as one with access to information unavailable to the solver. The target serves as a normalization reference and need not be attainable with the information and resources available to the solver. To compare performance across environments, we normalize scores so that the starting solution corresponds to 0 and the target to 1.

## 3.2 Multi-Agent Construction

Construction begins with papers whose research problems can be evaluated automatically and whose methods fit the available compute budget. For each paper, an author agent adapts the research problem into an executable task, specifying the solver's objective, permitted changes, and resource limits. It builds the starting solution, code for generating task data, and evaluator, and prepares the development and held-out test sets. The author agent also implements the paper's method for preliminary checks and measures the starting-solution and target scores.

The proposed environment then undergoes automated checks and review. A reviewer agent, with a context separate from the author agent's, examines the instructions, implementation, and check results to assess whether the task can be understood from the supplied materials and whether the evaluator measures the intended research objective. Initial checks verify that the components run and that the evaluator executes the code submitted through the required interface. Scripted tests also check whether performance differences exceed measured variation across repeated evaluations. The author agent specifies how the test cases differ from the development cases—for example, by reserving different simulation settings for testing—and the reviewer agent checks this distinction.

Once these checks pass, a reproducer agent receives the paper together with the task instructions, development data, and evaluator available to a solver. The author agent's implementation of the method, construction history, the test data, and the target score are withheld. The reproducer agent independently implements the paper's method using the same submission interface and compute budget as solver solutions. Its implementation is evaluated on the held-out test set to establish the reproduced baseline score. The reproduction must outperform the starting solution for the environment to pass this stage, providing direct evidence that improvement over the starting solution is possible.

The reviewer agent then examines the reproduction report, and a final end-to-end test checks that the environment can start, accept a solver's submission, and grade it. Acceptance requires successful automated checks and approval from the reviewer agent at every stage. A failed check or unresolved review finding returns the environment to the author agent for repair, and environments that still fail after the permitted repair attempts are rejected.

## 3.3 Scoring and Verification

Scoring solver submissions. Each environment measures its research objective using a scoring program that is deterministic for fixed evaluation inputs. To enable performance comparison across environments, we normalize a raw test score v using the starting-solution score w and target score $u ,$ both established on the same test set:

$$
b ( v ) = { \frac { v - w } { u - w } } .\tag{1}
$$

This maps the starting solution to 0 and the target to 1, with the reproduced baseline providing a separate comparison point on this scale.

Submission validity. Alongside performance scoring, automated checks assess whether submitted solutions satisfy the required file structure, interfaces, correctness conditions, and resource limits. A model judge agent, distinct from the reviewer agent used during environment construction, also examines the submitted code for violations of task-specific rules. It assigns zero reward to a violating submission regardless of its measured performance.

Preventing reward hacking. Agents may exploit access to evaluation data or tests to increase their scores without improving their solutions [17, 32]. To mitigate such risk, RSI-Forge restricts access to held-out grading materials. Before submitted code runs, the evaluator loads the test data into memory and removes their on-disk copy. The solver can use development evaluations while working, but held-out scores are not returned during solving. These restrictions do not eliminate all opportunities for exploitation: some environments left other grading materials accessible. We report the observed access failures and the sensitivity of solver results to those failures in Section 5 and Appendix C.3.2.

## 4 Building the Corpus

## 4.1 Sourcing

We collect recent preprints separately by domain, then deduplicate the resulting list to broaden disciplinary coverage. Each paper is read in full against six eligibility criteria: a quantitative objective with a stated direction of improvement, a measured baseline, a metric that can be computed by a deterministic script, a method that fits the available compute budget, measurable room for improvement, and a licence permitting redistribution. All source papers are from 2026. These criteria determine which papers enter construction; they do not guarantee that an environment can be built. Appendix A.1 gives additional selection and acceptance details, and Section 4.2 reports the counts.

Table 1. Main-run construction funnel. Counts include environments that passed after repairs. The additional 30 environments built with GPT-5.6-sol are outside this funnel.
<table><tr><td>Stage</td><td>Count</td></tr><tr><td>Source papers screened Papers eligible for construction Construction attempts begun</td><td>10,800 2,117 213</td></tr><tr><td>Passed Design Passed Build</td><td>199</td></tr><tr><td>Passed Independent Reproduction</td><td>198 189</td></tr><tr><td>Passed End-to-End Testing Included in the main corpus</td><td>180 180</td></tr></table>

Table 2. Examples of environments from the main construction run.
<table><tr><td>Field</td><td>Task</td><td>Evaluation objective</td></tr><tr><td>Robotics</td><td>Choose footholds on a terrain map</td><td>Increase distance in six steps</td></tr><tr><td>Life Sciences</td><td>Propose molecules for a binding pocket</td><td>Improve docking scores</td></tr><tr><td>Earth, Climate &amp; Space</td><td>Estimate weather from partial observations</td><td>Reduce state-estimation error</td></tr><tr><td>Statistics &amp; Inference</td><td>Estimate effects without a control series</td><td>Reduce treatment-effect error</td></tr><tr><td>Software Engineering &amp; Formal Methods</td><td>Generate tests within a fixed budget</td><td>Trigger more distinct failures</td></tr><tr><td>Computational Physics</td><td>Infer a plasma boundary from magnetic data Reduce boundary-estimation error</td><td></td></tr></table>

## 4.2 Construction Funnel

The main construction run used Claude Opus 5 for the author agent, reproducer agent, and reviewer agent, with separate sessions and role-specific inputs. Table 1 traces the source papers through eligibility and construction. We stopped after 180 accepted environments; only 213 of the 2,117 eligible papers entered this run, so the counts do not estimate the yield from all eligible papers.

Among accepted environments, 79% required at least one revision after failing a construction check. The funnel therefore reports passage after any permitted repairs. Analysis of repair records identified recurring defects in task instructions and evaluation, summarized in Table 3. Construction used a median of 170 million model tokens per accepted environment. Across construction, 99.6% of input tokens were cache reads. Appendix A.2 gives the repair breakdown and construction costs.

We then replaced the construction model with GPT-5.6-sol and built 30 additional environments using the same queue and checks. This yields 210 environments overall. The two construction sets are unpaired by source paper and field; the second run tests whether the procedure works with another backend, rather than estimating a paired model effect (Appendix C.4).

## 4.3 Corpus Composition and Contents

The full corpus contains 210 environments across 18 research fields. The 180 environments from the main construction run include 6–14 environments per field. Coverage extends beyond machine learning to areas including robotics, computational physics, statistics, and software engineering. Table 2 showcases example tasks and evaluation objectives from six research fields. Table 5 reports the full field distribution.

Table 3. Most common construction defects identified by model-based coding of repair passages. Percentages are shares of repair passages; categories overlap.
<table><tr><td>Issue identified Share of repair passages</td></tr><tr><td>Checks unable to detect the intended failure 66%</td></tr><tr><td>Incorrect or incomplete instructions 63%</td></tr><tr><td>Checks inspecting a different copy of the code 54%</td></tr><tr><td>Score gains without performing the intended work 37%</td></tr></table>

For each environment, we retain the task instructions, runnable code, code for generating task data, starting solution, reproduced baseline implementation, and evaluation configuration. Environments provide a development set for experimentation and a held-out test set for scoring. Construction records document feedback from the reviewer agent and repairs.

The corpus is accompanied by quality assessments and records of agent experiments. These include human and automated ratings with written justifications, agent transcripts, saved programs, experiment logs, and held-out scores. Together, they support inspection of both the constructed environments and the sequence of changes agents make while solving them.

Example environment. In one weather-estimation environment, the solver estimates a state from a small ensemble and noisy observations of half its components. The starting solution uses an ensemble Kalman method using raw sample covariance. The solver can test changes on 160 development instances; grading uses 324 held-out instances from different simulation conditions. The objective is to minimize mean estimation error. Held-out test errors are 1.786 for the starting solution and 1.390 for the independently reproduced paper method. With a target score of 1.156, normalization maps the starting solution to 0 and the reproduced baseline to approximately 0.629 (Eq. (1)).

## 4.4 Quality

Three agent judges rated all 180 main-run environments, and human domain experts rated 90. Both groups used the same nine-dimension rubric, with each dimension rated from 1 to 5. On the expertreviewed subset, the means across environments and rubric dimensions were 4.24 and 4.54 for human experts and the three-judge panel, respectively. Experts rated room above the starting solution (4.81) and discrimination between attempts (4.74) highest. These ratings support the environments' core measurement properties (Fig. 2 and Table 6).

Experts were more critical of saturation, shortcut resistance, and fidelity to the paper. The widest difference concerned whether one idea could exhaust a task: experts assigned 3.09, compared with 3.98 from the agent panel. This comparison also reflects unequal information: some experts cited the solverresult summaries withheld from agent judges. Fidelity disagreements often concerned simplifications made to fit the compute budget. Expert-review concerns are summarized in Appendix B.2.1.

Removing the most generous judge lowers the panel mean to 4.39 without eliminating the largest disagreements. For the additional 30 environments built with GPT-5.6-sol, all three judges give mean ratings within a fifth of a point of their main-run ratings (Appendix C.4). Under the same rubric, agent judges rated our environments higher than manually constructed RSI-Exam tasks on distinguishing solution quality and resisting shortcuts, whereas RSI-Exam received higher ratings on resistance to saturation (Table 7). Review ratings are not used to select or weight solver outcomes.

## Quality ratings on the same 90 environments

O Human experts (one per environment) Agent judges (mean of three)  
![](images/1c46c016d177cda3e57bc4f38f51c9b75fd4a150d81ada511c78800fb129ab7e.jpg)  
Figure 2. Quality ratings from human experts and agent judges on the same 90 environments, with one expert and three judges per environment. Points show means on a 1-5 scale (higher is better). Agent ratings are averaged within environments; rows are ordered by expert mean. Only experts received available solver results. Appendix B.2.2 and Table 6 give the rubric and per-dimension standard deviations.

## 5 Experiments

We use the environments to measure improvement over successive attempts and relative to the reproduced baselines, and analyze how agents explore alternatives, assess changes, and select solutions.

## 5.1 Experimental Setup

Before inspecting scores, we randomly sampled 120 environments from the main construction run, covering all 18 fields. Four solver agents use the same Claude Code CLI harness, with only the model endpoint changed: Claude Opus 5, GPT-5.6-sol, DeepSeek-V4-Flash, and Claude Haiku 4.5.

Each model runs 3 successive sessions per environment, yielding 480 model-environment chains and 360 sessions per model. Sessions begin with fresh contexts. The first receives the starting solution; subsequent sessions inherit the previous session's submitted solution, saved versions, experiment log, and notes. Sessions end when the agent stops or at 60 minutes. Almost all ended when the agent stopped. This budget is separate from each environment's budget for running a candidate method. Models exchange no records in the main experiment, and their weights remain fixed.

We report held-out scores of submitted solutions after each session, using the final submission after session 3 as the primary outcome. We also replay every saved intermediate version on the heldout test set to measure progress within sessions and assess whether agents submit their best saved solutions. Comparisons use normalized scores and within-environment ranks; bootstrap intervals resample environments. Appendix C.1 gives further protocol details, and Table 9 reports saved versions and output tokens.

Table 4. Submitted-solution scores on 120 environments, normalized to the starting solution (0) and target (1). Session 3 confidence intervals use bootstrap resampling of environments. Mean rank averages within-environment ranks at session 3 (1 is best).
<table><tr><td rowspan="2">Model</td><td colspan="3">Median normalized score ↑</td><td colspan="2">Session 3</td></tr><tr><td>Session 1</td><td>Session 2</td><td>Session 3</td><td>95% CI</td><td>Mean rank ↓</td></tr><tr><td>Claude Opus 5</td><td>0.754</td><td>0.791</td><td>0.800</td><td>[0.753, 0.867]</td><td>1.67</td></tr><tr><td>GPT-5.6-sol</td><td>0.673</td><td>0.747</td><td>0.761</td><td>[0.723, 0.824]</td><td>1.73</td></tr><tr><td>DeepSeek-V4-Flash</td><td>0.550</td><td>0.594</td><td>0.647</td><td>[0.586, 0.711]</td><td>2.97</td></tr><tr><td>Claude Haiku 4.5</td><td>0.129</td><td>0.277</td><td>0.332</td><td>[0.203, 0.451]</td><td>3.63</td></tr></table>

<table><tr><td></td><td>Opus 5</td><td>GPT-5.6-sol</td><td>DeepSeek V4-Flash</td><td>Haiku 4.5</td><td>100</td></tr><tr><td>Improvement</td><td>87</td><td>86</td><td>82</td><td>77</td><td rowspan="3">50 0</td></tr><tr><td>Exploration</td><td>98</td><td>81</td><td>77</td><td>49</td></tr><tr><td>Shortcuts</td><td>46</td><td>72</td><td>81</td><td>82</td></tr></table>

Figure 3. Transcript-coded behavior across 360 sessions per model (120 environments, three sessions each). Cell values and color intensity show percentages of sessions assigned to each group by model-based open coding Groups overlap; "improvement" covers developing and checking solutions, without requiring score gains.

## 5.2 Performance and Continued Improvement

Table 4 reports submitted scores after each session. The two strongest models have overlapping medianscore intervals; both outperform DeepSeek-V4-Flash, which outperforms Claude Haiku 4.5. Withinenvironment ranks give the same ordering, and paired bootstrap comparisons separate five of the six model pairs. These results describe performance on this corpus under the common harness. The same solver ordering holds on the 30 environments constructed with GPT-5.6-sol, although the two construction sets are unpaired by source paper and field (Table 12).

Most score improvement occurs in the first session for the three stronger models, with smaller gains in later sessions. With a 0.01 threshold on gains in normalized score, at least one model improves after the first session in 84% of environments. In 68 of the 120 environments, at least one model surpasses the reproduced baseline in one of its 3 sessions. These comparisons establish progress within the constructed tasks and against their reproduced baselines (Appendix C.2)

Final solution selection. Some final submissions score below an earlier saved version from the same agent, even when the better version passes the automated submission checks. Replaying intermediate versions exposes this difference between discovering a solution and selecting it.

Access to held-out material. The transcript coding records access to held-out material or another run's graded record in 22 of 480 chains. We retain these chains in the main results. Excluding them changes each model's median normalized score by at most 0.006 and preserves the ordering (Appendix C.3.2).

## 5.3 A Three-Session Example

The aggregate scores in Table 4 summarize performance across 120 environments. Figure 4 shows one Claude Opus 5 chain in detail: every saved solution graded over three successive sessions and the changes described in the agent's records.

In this environment, a humanoid crossing stepping stones, pillars, stairs, and gaps must choose its next foothold. A submitted solution receives a local elevation map, a six-component state, and a list of admissible locations, and returns the index of one location. The evaluator measures the distance reachable from that foothold within six steps, using held-out terrain generators.

The submitted normalized score rises from 0.15 in session 1 to 0.33 in session 2 and 0.58 in session 3. The second session evaluates each foothold by several of its best continuations, replacing the first session's single-best-continuation rule. In the third session, the agent adds a check that a foothold has sufficient surrounding ground support, removing invalid continuations. With these removed, it returns to the single-best-continuation rule. The trajectory illustrates how an agent can improve by revisiting an inherited design choice.

![](images/8e02e95ae867c48b329c9f8cfe267850e32456fe189d25d88c680855ac6428c3.jpg)  
Figure 4. A three-session Claude Opus 5 chain on a foothold-selection task. Open points are saved solutions graded on the held-out test set; filled points mark final submissions, with normalized scores of 0.15, 0.33, and 0.58. The annotations summarize the changes recorded in each session. The third session adds a ground-support check and reverses the second session's averaging rule.

## 5.4 Transcript Analysis

We analyze the records to characterize how agents work. A model-based coding procedure groups practices into improvement (practices for developing and checking solutions), exploration (tests of whether a change is useful), and shortcuts (score-directed behavior without sufficient evidence of underlying improvement). For the analysis shown in Fig. 3, Claude Opus 5 identifies recurring practices without a predefined category list. Each positive assignment requires supporting quotations from the transcript Humans did not label these trajectories (Appendices C.3 and C.3.2).

Figure 3 summarizes the prevalence of the three behavior groups across models. All four models frequently modify their solutions, but their evidence-gathering practices differ. Exploration is more common for Claude Opus 5 than for Claude Haiku 4.5, while fitting to the development score follows the reverse pattern. The middle models exchange positions on individual practices.

We check these patterns using a separate analysis with a fixed codebook of ten practices. Three additional model coders apply this codebook to a stratified sample of 30 sessions per solver model, obtaining agreement of κ = 0.73 with the original coder and recovering the same contrast in exploration and shortcuts between Claude Opus 5 and Claude Haiku 4.5 (Appendix C.3.2). A further check uses numeric logs from 139 sessions with recoverable scores and standard errors. Gains smaller than the reported standard error account for 73% of recorded accepted improvements for Claude Haiku 4.5, compared with 44% for Claude Opus 5 (Appendix C.3.3).

Among sessions that outperform the reproduced baseline, 95% are coded as involving work beyond parameter tuning; 5% are tuning-only. This coding characterizes the work performed in these tasks without establishing novelty to the source paper's field. Figure 5 reports mechanism frequencies, and Appendix C.2 gives examples of method changes.

## 6 Discussion

Our results show that published papers can be a practical source of executable research environments across diverse disciplines. Most accepted environments required repairs before passing the construction checks (Section 4.2). Recurring defects included incorrect instructions and checks that could miss failures, showing that automated environment construction involves substantial verification and repair.

The quality reviews clarify what this construction process achieves. Experts rated room above the starting solution and the evaluator's ability to distinguish attempts most highly, supporting the properties needed to measure progress. Their lower ratings for resistance to saturation and fidelity to the source paper identify a harder challenge: preserving the depth of a research problem when making it executable under a fixed budget. The disagreement with automated judges suggests that targeted expert review remains valuable for assessing these properties as construction scales.

The solver experiments demonstrate both improvement over successive attempts and gains beyond reproduced baselines. Gains are nevertheless concentrated in the first session for the stronger models Diminishing gains do not by themselves establish task saturation; they may also reflect limits in how agents search for alternatives or use inherited records.

The trajectories also reveal weaknesses in how agents conduct experiments and retain their results. Lower-scoring models explored less and more often accepted gains smaller than the reported standard error. Across models, some final submissions were worse than previously saved versions, showing that finding a better solution does not ensure it is retained. These observations make experimental judgment and final selection concrete targets for future interventions. The saved programs, measurements, and handover records provide a basis for evaluating such interventions; the present associations do not establish which practices cause higher performance.

Limitations. Tasks are adapted to bounded compute, so surpassing a reproduced baseline does not establish an advance on the original research problem. Our sessions measure progress on individual tasks; the study does not isolate the effect of inherited records from extra solving time. No trajectories were used for training; the study does not establish broader gains in research capability. Some environments left held-out grading materials accessible during solver runs. Expert review covers 90 of 180 main-run environments, with one expert each and no human inter-rater estimate; model-generated trajectory labels lack human validation. We do not ablate construction roles, leaving their individual contributions unresolved. Access restrictions limit independent replication of the full corpus (Appendix D).

## 7 Conclusion

We introduced RSI-Forge, a multi-agent pipeline that turns published papers into executable RSI environments with automated evaluators and independently reproduced baselines. The corpus contains 210 environments across 18 fields. Human expert reviews highlight opportunities to improve starting solutions and the evaluators' ability to distinguish solution quality, while identifying challenges in shortcut resistance and fidelity to the source paper. Solver experiments demonstrate gains over successive attempts and beyond reproduced baselines. Trajectories further reveal differences in how agents explore alternatives, assess uncertain gains, and select solutions. RSI-Forge provides a scalable approach to expanding the supply of research environments for developing and evaluating self-improving agents.

## References

[1] Aiming Lab. RSI-Exam: Benchmarking recursive self-improvement through executable research. https://rsi-exam.ai/,2026.

[2] Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In International Conference on Machine Learning, 2009.

[3] Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, et al. MLE-bench: Evaluating machine learning agents on machine learning engineering. In International Conference on Learning Representations, 2025.

[4] Michael Dennis, Natasha Jaques, Eugene Vinitsky, Alexandre Bayen, Stuart Russell, Andrew Critch, and Sergey Levine. Emergent complexity and zero-shot transfer via unsupervised environment design. In Advances in Neural Information Processing Systems, 2020.

[5] Yi Duan, Ying Liu, Zirui Tang, Haodong Chen, Jun Zhou, Yumou Liu, Bangrui Xu, Yukai Wu, Sidi Chen, Yuhan Zhou, Haoyu Wang, Xiaoyou Yu, Shaokun Han, Xuzhou Zhu, Le Zhou, Bolin Lu, Wei Zhou, Jiachen Liu, Nuozhou Fang, Jiaxin Tian, Ruoyu Chen, Yuxuan Li, Kai Zuo, Kaiyan Zhang, Qianyu Yang, Zijie Wang, Jiantao Qiu, Conghui He, Guoliang Li, Bowen Zhou, Zhiyuan Liu, Zhoufutu Wen, Jihua Kang, Xuanhe Zhou, and Fan Wu. The last AI built by humans: Toward genuine recursive self-improvement, 2026.

[6] Zhiyuan Fan, Tinghao Yu, Yuanjun Cai, Jiang Zhou, Jiangtao Guan, Jincheng Liu, Yun Yang, Dingxin Hu, Zhuo Han, Xing Wu, Feng Zhang, and Lilin Wang. Environment evolution for terminal agents, 2026.

[7] Chrisantha Fernando, Dylan Banarse, Henryk Michalewski, Simon Osindero, and Tim Rocktäschel. Promptbreeder: Self-referential self-improvement via prompt evolution, 2023.

[8] Shengran Hu, Cong Lu, and Jeff Clune. Automated design of agentic systems, 2024.

[9] Chengsong Huang, Wenhao Yu, Xiaoyang Wang, Hongming Zhang, Zongxia Li, Ruosen Li, Jiaxin Huang, Haitao Mi, and Dong Yu. R-Zero: Self-evolving reasoning LLM from zero data, 2025.

[10] Qian Huang, Jian Vora, Percy Liang, and Jure Leskovec. MLAgentBench: Evaluating language agents on machine learning experimentation. In International Conference on Machine Learning, 2024.

[11] Minqi Jiang, Edward Grefenstette, and Tim Rocktäschel. Prioritized level replay. In International Conference on Machine Learning, 2021.

[12] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues? In International Conference on Learning Representations, 2024.

[13] Hyunin Lee, Jinglue Xu, Jeffrey Seely, Donghyun Lee, Matei Zaharia, and Yujin Tang. Recursive harness self-improvement, 2026.

[14] Bo Liu, Simon Yu, Yiding Jiang, Ao Qu, Andrew Zhao, Zichen Liu, Junsu Kim, Zijian Zhou, Seungone Kim, Tongzheng Ren, Mickel Liu, Hanfei Yu, Zhaorun Chen, Weiyan Shi, Paul Pu Liang, Luke Zettlemoyer, Yejin Choi, and Natasha Jaques. SPADE: Self-play in adaptive synthetic executable environments, 2026.

[15] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, et al. Self-refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems, 2023.

[16] Mike A. Merrill, Alexander G. Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, et al. Terminal-Bench: Benchmarking agents on hard, realistic tasks in command line interfaces. In International Conference on Learning Representations (ICLR), 2026. URL https://openreview .net/forum?id=a7Qa4CcHak.

[17] Pradyumna Shyama Prasad, Meiri Anto, Leon Eshuijs, Julian Moncarz, Kaustubh Kislay, and Juan J. Vazquez. BAITBENCH: Measuring agent reward hacking with optional shortcuts planted in ML tasks. arXiv preprint arXiv:2608.30724, 2026.

[18] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash/ 1b44b878bb782e6954cd888628510e90-Abstract-Conference.html.

[19] Zachary S. Siegel, Sayash Kapoor, Nitya Nadgir, Benedikt Stroebl, and Arvind Narayanan. CORE-Bench: Fostering the credibility of published research through a computational reproducibility agent benchmark, 2024.

[20] Xiaoshuai Song, Haofei Chang, Guanting Dong, Yutao Zhu, Ji-Rong Wen, and Zhicheng Dou. EnvScaler: Scaling tool-interactive environments for LLM agent via programmatic synthesis. In Findings of the Association for Computational Linguistics: ACL 2026, 2026. doi: 10.18653/v1/2026. findings-acl.407. URL https://aclanthology.org/2026.findings-acl.407/.

[21] Giulio Starace, Oliver Jaffe, Dane Sherburn, James Aung, Jun Shern Chan, et al. PaperBench: Evaluating AI's ability to replicate AI research, 2025.

[22] Raffael Theiler, Ludovico Comito, David Leko, Leandro Von Krannichfeldt, Lev Telyatnikov, and Olga Fink. From paper to benchmark: agentic, framework-based reproduction of under-specified methods in machine health intelligence. arXiv preprint arXiv:2605.28371, 2026.

[23] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, et al. Voyager: An openended embodied agent with large language models, 2023.

[24] Hjalmar Wijk, Tao Lin, Joel Becker, Sami Jawhar, Neev Parikh, et al. RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts, 2024.

[25] Jingxu Xie, Dylan Xu, Xuandong Zhao, and Dawn Song. AgentSynth: Scalable task generation for generalist computer-use agents, 2025.

[26] Minrui Xu, Zilin Wang, Mengyi DENG, Zhiwei Li, Zhicheng Yang, Xiao Zhu, Yinhong Liu, Boyu Zhu, Baiyu Huang, Chao Chen, Heyuan Deng, Fei Mi, Lifeng Shang, Xingshan Zeng, and Zhijiang Guo. EnvFactory: Scaling tool-use agents via executable environments synthesis and robust RL, 2026.

[27] Xunjian Yin, Xinyi Wang, Liangming Pan, Li Lin, Xiaojun Wan, and William Yang Wang. Gödel agent: A self-referential agent framework for recursive self-improvement, 2024.

[28] Eric Zelikman, Eliana Lorch, Lester Mackey, and Adam Tauman Kalai. Self-taught optimizer (STOP): Recursively self-improving code generation. In Conference on Language Modeling (COLM), 2024.

[29] Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, and Jeff Clune. Darwin Gödel machine: Open-ended evolution of self-improving agents, 2025.

[30] Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, and Chenglin Wu. AFlow: Automating agentic workflow generation, 2024.

[31] Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Yang Yue, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, and Gao Huang. Absolute zero: Reinforced self-play reasoning with zero data, 2025.

[32] Ziqian Zhong, Aditi Raghunathan, and Nicholas Carlini. ImpossibleBench: Measuring LLMs' propensity of exploiting test cases. arXiv preprint arXiv:2510.20270, 2025.

[33] Kaijie Zhu, Yuzhou Nie, Yijiang Li, Yiming Huang, Jialian Wu, Jiang Liu, Ximeng Sun, Zhenfei Yin, Lun Wang, Zicheng Liu, Emad Barsoum, William Yang Wang, and Wenbo Guo. TermiGen: High-fidelity environment and robust trajectory synthesis for terminal agents, 2026.

## A Construction Details

## A.1 Selection and Acceptance

The six sourcing criteria are given in Section 4.1. A task must provide a single ranking of solutions; additional quantities may impose fixed constraints, but unconstrained trade-offs between objectives do not meet this requirement. Screening for room to improve does not establish that the eventual target score is attainable. The source paper's reported performance guides construction but does not replace measurement of the reproduced method within the adapted environment.

The roles and information restrictions are described in Section 3. All three construction roles use Claude Opus 5 in separate contexts in the main run; this separation does not establish that their errors are independent. Acceptance requires automated checks and reviewer-agent approval at every stage. Development and test sets must differ along a documented task condition, rather than only by a random partition. Independent reproduction uses the same execution path and container as solver solutions and must outperform the starting solution. The author agent has at most 3 repair attempts per stage before rejection. End-to-end checks cover startup, submission, and grading; they do not establish that all grading materials are inaccessible (Appendix C.3.2).

## A.2 Repairs and Cost

Among the 180 accepted main-run environments, 79% required repair: 53% during design, 45% during build, 21% during independent reproduction, and 20% during end-to-end testing. These stage shares overlap. Main Table 3 summarizes model-coded defects in author-agent repair passages; its percentages describe passages, not environments, and categories overlap. Median construction cost is 170 million model tokens per accepted environment. Median per-environment shares are 82%, 14%, and 3% for author, reviewer, and reproducer, respectively. These separately summarized medians need not sum to 100%.

## B Corpus Composition and Quality

## B.1 Field Coverage

Table 5 reports field counts for the 180 main-run environments; Appendix C.4 describes the additional 30 environments constructed with a second backend.

Table 5. Field composition of the 180 main-run environments.
<table><tr><td>Field</td><td>Count</td><td>Field</td><td>Count</td></tr><tr><td>Robotics</td><td>14</td><td>Security &amp; Cryptography</td><td>10</td></tr><tr><td>Life Sciences</td><td>13</td><td>Neuroscience</td><td>9</td></tr><tr><td>Earth, Climate &amp; Space</td><td>13</td><td>Optimisation &amp; Discrete Algorithms</td><td>9</td></tr><tr><td>Statistics &amp; Inference</td><td>13</td><td>Chemistry &amp; Materials</td><td>9</td></tr><tr><td>Networks &amp; Information Th.</td><td>12</td><td>Graphics &amp; Rendering</td><td>8</td></tr><tr><td>Software Eng. &amp; Formal Methods</td><td>12</td><td>Quantitative Finance</td><td>7</td></tr><tr><td>Signal, Imaging &amp; Control</td><td>11</td><td>Computational Physics</td><td>7</td></tr><tr><td>Computer Systems</td><td>10</td><td>Mechanical &amp; Thermal</td><td>7</td></tr><tr><td>Numerical &amp; Sci. Computing</td><td>10</td><td>Machine Learning</td><td>6</td></tr></table>

## B.2 Review Protocol and Results

Three agent judges rated all 180 main-run environments: Claude Opus 5 under Claude Code, GPT-5.6-sol under Codex at high reasoning effort, and Gemini 3.7 Flash under Antigravity. Each model-harness pair is one judge. Judges read task instructions and evaluators and can run code, with network access for field knowledge; the author agent's construction transcript, held-out test data, and solver-result summaries are withheld.

Domain experts independent of environment construction, each holding a doctorate in the relevant field, reviewed 90 environments, with one expert per environment and a written justification for every rating. All reviews enter the analysis; there is no human inter-rater estimate. Both groups use the same 9-dimension, 1–5 rubric (Table 8) and have network access. Experts receive the source paper after submitting ratings for the eight environment-focused dimensions, then rate faithfulness last. Only experts receive available summaries of solver scores over successive sessions. Neither group receives construction-stage verdicts or reviewer-agent notes. Expert-agent comparisons therefore reflect differences in evidence as well as judgment.

Harness sensitivity. For Gemini 3.7 Flash, 1,620 paired dimension ratings on 180 environments have a mean absolute difference of 0.13 points between Claude Code and Antigravity; 99% differ by at most one point. Mean ratings are 4.88 and 4.82, respectively. This indicates limited sensitivity for this judge under these two harnesses.

Table 6 gives matched expert and agent-panel ratings with standard deviations. Removing Gemini 3.7 Flash, which gives the highest mean on every dimension, lowers the panel mean from 4.54 to 4.39, compared with 4.24 for experts. Resistance to saturation and shortcut resistance remain the two largest positive gaps (+0.69 and +0.42, respectively). Ratings are not used to select or weight solver outcomes.

Table 6. Human-expert and agent-panel ratings on the same 90 environments. Scores are means on a 1–5 scale; SD denotes standard deviation. Agent ratings are averaged within environments; Gap is the panel mean minus the expert mean.
<table><tr><td>Dimension</td><td>Experts</td><td>SD</td><td>Panel</td><td>SD</td><td>Gap</td></tr><tr><td>Research problem</td><td>4.39</td><td>0.84</td><td>4.59</td><td>0.32</td><td>+0.20</td></tr><tr><td>Room above starting solution</td><td>4.81</td><td>0.51</td><td>4.89</td><td>0.31</td><td>+0.08</td></tr><tr><td>Resistance to saturation</td><td>3.09</td><td>0.96</td><td>3.98</td><td>0.84</td><td>+0.89</td></tr><tr><td>Faithfulness to source paper</td><td>3.72</td><td>0.87</td><td>4.22</td><td>0.51</td><td>+0.50</td></tr><tr><td>Metric validity</td><td>4.40</td><td>0.85</td><td>4.49</td><td>0.45</td><td>+0.09</td></tr><tr><td>Single objective</td><td>4.61</td><td>0.90</td><td>4.66</td><td>0.46</td><td>+0.05</td></tr><tr><td>Shortcut resistance</td><td>4.09</td><td>0.97</td><td>4.67</td><td>0.55</td><td>+0.58</td></tr><tr><td>Score discrimination</td><td>4.74</td><td>0.62</td><td>4.65</td><td>0.55</td><td>-0.09</td></tr><tr><td>Development-test distinction</td><td>4.33</td><td>0.93</td><td>4.67</td><td>0.49</td><td>+0.34</td></tr><tr><td>All nine dimensions</td><td>4.24</td><td>0.99</td><td>4.54</td><td>0.58</td><td>+0.30</td></tr></table>

## B.2.1 Expert-Agent Disagreements

The largest agent-panel minus expert gaps concern resistance to saturation (+0.89), shortcut resistance (+0.58), and faithfulness (+0.50). Expert reviews illustrate these concerns.

Resistance to saturation. One expert gave the lowest rating after a supplied search method covered about 69% of the declared improvement band in roughly five CPU-minutes. The expert also cited solver results reaching the target score in the first session, which agent judges did not receive. The disagreement concerns both available evidence and interpretation of the rubric: reaching a normalization target does not rule out further raw gains or establish that the research problem is exhausted.

Shortcut resistance. One expert proposed adapting SQL answers through repeated held-out grading feedback. The intended protocol withholds that feedback and removes test files before submitted code runs (Section 3.3); whether this attack is possible depends on runtime access controls, not task wording alone. Separately, another review found supplied parameters and held-out family definitions that would let a solution branch on those definitions instead of estimating the intended quantity. We treat that exposure as an environment defect. Observed solver access failures are reported in Section 5.2 and Appendix C.3.2.

Faithfulness. An expert assigned 2 against the panel's 4.5: the task retained 50-way zero-shot classification and comparison of audio and text semantics, but replaced real THINGS-EEG recordings, pretrained feature pipelines, and the trimodal BraVL VAE with synthetic low-dimensional arrays and a linear-Gaussian brain model. A bounded-compute adaptation can preserve the broad question while omitting substantive parts of the source method.

Table 7. Ratings from the same three agent judges on 34 manually constructed RSI-Exam tasks and the 90 RSI-Forge environments in Table 6. Gap is RSI-Forge minus RSI-Exam; the 8 dimensions exclude faithfulness because this comparison has no common source-paper reference. The task sets are unpaired, so the differences do not isolate the effect of automated construction.
<table><tr><td>Dimension</td><td>RSI-Exam</td><td>RSI-Forge</td><td>Gap</td></tr><tr><td>Score discrimination</td><td>2.64</td><td>4.65</td><td>+2.01</td></tr><tr><td>Development-test distinction</td><td>3.23</td><td>4.67</td><td>+1.44</td></tr><tr><td>Shortcut resistance</td><td>3.92</td><td>4.67</td><td>+0.75</td></tr><tr><td>Single objective</td><td>4.64</td><td>4.66</td><td>+0.03</td></tr><tr><td>Room above starting solution</td><td>4.87</td><td>4.89</td><td>+0.02</td></tr><tr><td>Research problem</td><td>4.61</td><td>4.59</td><td>-0.02</td></tr><tr><td>Metric validity</td><td>4.57</td><td>4.49</td><td>-0.08</td></tr><tr><td>Resistance to saturation</td><td>4.41</td><td>3.98</td><td>-0.43</td></tr><tr><td>all 8</td><td>4.11</td><td>4.58</td><td>+0.46</td></tr></table>

## B.2.2 Review Rubric

Human experts and agent judges use the same quality-review rubric, separate from construction acceptance checks and submission-validity checks. Table 8 reproduces the descriptions for scores 1, 3, and 5. Terms such as “starter," “ceiling," and “split" are retained from the rubric wording; in the surrounding text we use starting solution, target score, and development/test sets.

Table 8. Descriptions of scores 1, 3, and 5 in the environment-quality rubric, reproduced verbatim. The dimensions are grouped here as in the rating summaries; the human review form presents faithfulness to the source paper last.
<table><tr><td>Axis</td><td>A score of 1</td><td>A score of 3</td><td>A score of 5</td></tr><tr><td>Real problem</td><td>a toy with no research content</td><td>a real problem, narrowly posed</td><td>you would be pleased to see a student work on this</td></tr><tr><td>Headroom</td><td>the starter is already near the ceiling</td><td>a competent effort gains some- thing</td><td>clearly large, and you can name where it comes from</td></tr><tr><td>Not saturable</td><td>one change takes most of the range</td><td>the first change is worth it, more remains</td><td>improvement requires several independent ideas</td></tr><tr><td>Faithful to the paper</td><td>unrelated to what the paper does</td><td>related, but the hard part has been removed</td><td>this is the paper&#x27;s problem, honestly posed</td></tr><tr><td>Metric validity</td><td>measures something else</td><td>a reasonable proxy</td><td>this is the quantity the field would report</td></tr><tr><td>Single direction</td><td>two objectives silently traded</td><td>one quantity, the constraint is loose</td><td>one quantity, one direction, the constraint binds</td></tr><tr><td>Resistant to shortcuts</td><td>you can name a specific trick and say why it scores</td><td>you suspect one exists but cannot name it</td><td>you looked and the scorer closes the routes you tried</td></tr><tr><td>Discriminating</td><td>nothing states it and the metric looks noisy</td><td>stated, or inferable, but not justified</td><td>stated with the room to im- prove, and the room is the larger of the two</td></tr><tr><td>Split is a real axis</td><td>a uniform random partition dressed up as an axis</td><td>an axis, mild shift</td><td>a real axis; doing well on it means something</td></tr></table>

## C Additional Experimental Details

## C.1 Experimental Setup

The protocol in Section 5.1 uses solver containers with network access restricted to the model endpoint. The saved files described there are the only inheritance between sessions: the live workspace, conversational context, and shell history are not carried over. Development-evaluator calls have no fixed count limit. The final submission is the solution left in the designated method directory when the agent stops or the 60-minute session ends; it is inherited regardless of its score. This session limit is separate from each environment's compute budget for running a candidate method.

Every saved version, including abandoned versions, and the final submission are replayed on held-out data. These scores are not returned during solving. A session without a valid measurement remains unscored rather than receiving an imputed score. Table 9 summarizes saved versions and output volume for the main sample.

Starting and target scores. In Eq. (1), the starting-solution score w maps to 0 and the target u to 1; better-than-target scores exceed 1. Construction records identify the data, code, or bound supporting each value and mark provisional values. Targets may use privileged information, an informationtheoretic or optimization bound, exact solutions on tractable instances, or a Bayes rate. They need not be attainable, and normalization does not validate them. If the reproduced baseline is too close to the target, construction checks require revision of the target or task conditions.

Table 9. Activity per session on the 120 main evaluation environments. Versions are saved runnable solutions that were graded; output tokens are in thousands. Means include one standard deviation. Token volume is not equivalent to monetary or computational cost across providers.
<table><tr><td rowspan="2">Model</td><td colspan="2">Versions graded</td><td colspan="2">Output tokens (thousands)</td></tr><tr><td>Median</td><td>Mean</td><td>Median</td><td>Mean</td></tr><tr><td>Claude Opus 5</td><td>4.0</td><td> $3 . 6 \pm 1 . 6$ </td><td>69.3</td><td> $7 2 . 2 \pm 2 2 . 8$ </td></tr><tr><td>GPT-5.6-sol</td><td>2.0</td><td> $2 . 7 \pm 1 . 7$ </td><td>22.1</td><td> $2 2 . 6 \pm 8 . 3$ </td></tr><tr><td>DeepSeek-V4-Flash</td><td>2.0</td><td> $2 . 1 \pm 1 . 1$ </td><td>54.5</td><td> $5 3 . 8 \pm 1 6 . 3$ </td></tr><tr><td>Claude Haiku 4.5</td><td>2.0</td><td> $2 . 7 \pm 1 . 9$ </td><td>30.2</td><td> $2 9 . 8 \pm 8 . 8$ </td></tr></table>

## C.2 Improvements over Reproduced Baselines

At least one session surpasses the reproduced baseline in 68 of 120 environments. The per-model counts are 65 for GPT-5.6-sol, 63 for Claude Opus 5, 54 for DeepSeek-V4-Flash, and 24 for Claude Haiku 4.5. These counts overlap, and an environment can contribute several successful sessions. The median normalized baseline score is 0.76 where no model surpasses it and 0.46 where at least one does.

The separate mechanism codebook in Fig. 5 applies only to these successful sessions. Across them, 95% receive at least one of the 9 labels beyond parameter tuning, 5% receive only the tuning label, and 0% receive neither. These labels include repairing a non-running solution and evaluating alternatives, so they do not all imply a new method. Tuning can coexist with other work: it appears in 34% of successful Claude Opus 5 sessions and 77% of successful Claude Haiku 4.5 sessions. The coding does not establish which activity caused a gain or whether the result advances the source paper's field.

Actual method changes include retaining alternating least squares while replacing a non-differentiable tolerance with a smooth surrogate, and integrating over parameters that the reproduced baseline estimated at a single point.

## C.3 Behavior Coding and Analysis

Three coding procedures serve different purposes: open coding supplies the behavior groups in Fig. 3; a fixed practice codebook checks those patterns; and the separate mechanism codebook characterizes successful sessions (Appendix C.2). All labels are model-generated, without human validation.

## C.3.1 Coding Procedures

Open coding. Claude Opus 5 summarizes each session's changes, evidence, score-directed work, unexpected events, and stopping decision, then groups recurring practices in bounded batches without a predefined category list. Document identifiers are retained through merges, and positive assignments require matching transcript quotations. This yields 62 families mapped to improvement (developing and checking solutions, without requiring score gains), exploration (testing whether a change is useful), and shortcuts (score-directed behavior with insufficient evidence of underlying improvement), plus one ungrouped family for submitting the inherited solution unchanged. A session receives a group label if it has at least one family in that group; labels overlap and need not sum to 100%.

Fixed practices. Claude Opus 5 proposes categories from 24 documents selected with a fixed seed and no supplied category list. The ten-category codebook is then frozen and applied to all available session records (Table 10). A session can receive multiple labels or none; assignments without a matching quotation are discarded. Quotations make decisions traceable without validating the interpretation or the narrated action. Within-model assignment counts explain little variation in output-token counts, but this does not rule out narration-style differences; additional coders and numeric logs provide complementary checks below.

![](images/048309425b62e3f0fc54231c80a3e8bd8ccf7b29d30ba0a3bc03adf8c632e250.jpg)  
Figure 5. Transcript-coded mechanisms as percentages of each model's sessions that surpass the reproduced baseline. The codebook was derived from a sample and fixed before counting. Sessions may receive several labels; nine categories cover work beyond parameter tuning. The labels describe work within the constructed task and do not establish scientific novelty.

## C.3.2 Behavior Patterns and Sensitivity Checks

Shortcut labels and access failures. Common shortcut families include retaining a version because its development score rises and fitting constants to the development set. These can occur within the permitted workflow; the label concerns limited evidence of generalization. Other families cover reading answer keys, grading code, another run's records, or reconstructing tests from descriptions. These are broader than the access-failure criterion: a family may include labels supplied with development data, whereas an access flag requires a record naming held-out material or another run's graded record. The coding cannot exclude unrecorded access.

This narrower criterion flags 22 of 480 model-environment chains. They remain in the main results. Excluding flagged chains gives median final normalized scores of 0.797, 0.761, 0.641, and 0.332 for Claude Opus 5, GPT-5.6-sol, DeepSeek-V4-Flash, and Claude Haiku 4.5, respectively: ordering is unchanged, and no median moves by more than 0.006. Filtering is by chain; a flag for one model does not exclude the other models on that environment. In 4 of the 68 environments where a model surpasses the baseline, a successful session carries an access flag. This overlap does not attribute the gain to access. Behavior-group percentages use the full sample; only final-score sensitivity uses the exclusion.

## Improvement 21 families · 2,649 assignments

use error analysis to guide changes review the previous session's record select configurations by constraint margins measure structural properties of task data verify claims against supporting files inspect environment code to understand the task reuse information from earlier records inspect program behavior before editing test the final submission test inherited claims measure a baseline directly derive quantities instead of fitting guide changes with mathematical derivations retain and re-evaluate saved solutions detect a contaminated workspace adapt parameters to individual instances document results for the next session diagnose problems through direct inspection prepare a runnable submission select candidates by output properties reformulate the solution mathematically

## Shortcuts

14 families · 2,356 assignments keep changes when development scores improve fit constants to the development set act on unverified claims design the method around evaluator logic read packaged task files read the answer key hard-code values from development examples accept gains within measured variability reconstruct a proxy for the test set submit despite acknowledged overfitting read another run's workspace bound performance using hidden information infer hidden properties from task structure assume test conditions without measurement

![](images/a221e809e768a659eee9ade611fe4dff3c2c8914ec325e1ff3b844805ba7da29.jpg)

## Exploration

compare gains with measured variability evaluate on data excluded from fitting confirm gains with another evaluation abandon approaches after negative results verify checks can detect failures   
estimate potential gains before implementing select using a validation subset select using reasoning beyond scores reject gains that do not reflect improvement verify results with an independent method bound potential gains before searching investigate implausible measurements test the evaluation procedure run additional generalization tests override the development-score ranking perturb one factor at a time screen on fewer cases, confirm on more diagnose from parameter-sweep patterns   
select components using intermediate metrics compare gains across individual cases check measured cost against the budget identify limits of development evaluation retain multiple alternatives   
guide changes with diagnostic measurements isolate contributions with ablations revise conclusions after contrary evidence trace anomalies to the evaluation setup

<table><tr><td rowspan=1 colspan=1>5sndOCluude</td><td rowspan=1 colspan=2>V4---shGP-.-0[  Dee-</td><td rowspan=1 colspan=1>45HaikunCluudde]</td></tr><tr><td rowspan=1 colspan=1>41</td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>11</td></tr><tr><td rowspan=1 colspan=1>46</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>&lt;1</td></tr><tr><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>&lt;1</td></tr><tr><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>7</td></tr><tr><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>6</td></tr><tr><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>&lt;1</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>&lt;1</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>&lt;1</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>&lt;1</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>2</td></tr></table>

62 families. Each cell shows the percentage of a model's 360 sessions assigned to that family. Shading is scaled separately within each group.  
Figure 6. Open-coded families grouped as improvement (21), exploration (27), and shortcuts (14). Cells show percentages of each model's 360 sessions assigned a family. Labels overlap. Shading is scaled within each group; printed percentages support comparisons across groups. The ungrouped family for submitting the inherited solution unchanged is omitted.

Table 10. The ten fixed practices, distinct from the open-coded families in Fig. 6 and the successful-session mechanisms in Fig. 5. Labels and operational criteria are paraphrased from the coding scheme; timing requirements and exclusions are retained.
<table><tr><td>Group</td><td>Practice</td><td>Required evidence</td></tr><tr><td>Improvement</td><td>Check inherited results Use the task specification</td><td>Before any edit, rerun the inherited solution to reproduce its recorded score or independently derive its central claim. Justify the submitted change from a task-document fact or metric prop- erty; a parameter-sweep result alone is insufficient. Address failures such as degenerate inputs, timing limits, or fallback</td></tr><tr><td>Exploration</td><td>Bound potential improvement Test beyond development data Compare with error</td><td>behavior without targeting the metric. Before searching, compute an upper or lower performance bound and decompose the remaining gap to allocate the budget. Evaluate a change beyond the supplied development instances using a stress test or test of invariance. Use an explicit error bar when accepting or rejecting a change; a before- and-after comparison alone is insufficient.</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Shortcut</td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td>independent checks</td><td>weakened variant, an incorrect-generator control, or an independent computation.</td></tr><tr><td></td><td>Reject unreliable</td><td>Discount a gain because its evidence is circular, depends on a parameter</td></tr><tr><td></td><td>gains</td><td>fitted to development data, or changes sign across evaluations.</td></tr><tr><td></td><td>Select by</td><td>Run successive sweeps of individual parameters and treat any favorable</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td>alone</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td>development score</td><td>development-score change as an improvement.</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td>Environment failure</td><td></td></tr><tr><td>Neither</td><td></td><td></td></tr><tr><td></td><td></td><td>An environment failure unrelated to the solution method consumes the</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>session.</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr></table>

Additional model coders. Claude Sonnet 5, Gemini 3.7 Flash, and GPT-5.6-sol apply the fixed practice codebook to a stratified sample of 30 sessions per solver model. Agreement with the original coder is κ = 0.73 over the 9 categories used by either coder. On these records, the original coder labels exploration in 97% of Claude Opus 5 sessions and 30% of Claude Haiku 4.5 sessions; additional coders give 93% and 13%. Shortcut rates reverse the contrast: 3% versus 80% originally, and 0% versus 63% with additional coders. This corroborates the contrast without establishing individual-label correctness. These rates use the fixed codebook on the sampled subset, whereas Fig. 3 uses open-coded families on all 360 sessions per model.

## C.3.3 Gains Smaller than the Reported Standard Error

For 139 sessions, evaluator logs provide both scores and standard errors. We classify an accepted gain as smaller than the reported error when the selected version improves on the best previous version by less than the standard error attached to the selected score. This describes selection; it is not a significance test for the difference between correlated scores. Accepted improvements in these recoverable sessions form the denominator, not all sessions or saved versions. The rates are 73% for Claude Haiku 4.5 and 44% for Claude Opus 5. This agrees with their fixed-codebook ordering, while measuring a different behavior: the codebook concerns selection from development-score changes without sufficient supporting evidence.

## C.4 Environments by Backend Model

Replacing Claude Opus 5 with GPT-5.6-sol in the same sourcing queue, checks, and repair procedure produces 30 additional environments, for 210 overall. The comparison uses accepted environments only and does not measure construction yield. Sets are unpaired by source paper and field, so differences can reflect task composition as well as backend choice. All three judges' mean ratings differ by less than 0.2 points (Table 11). Solvers retain the same ordering (Table 12); this extends the observation to a second construction set without isolating a causal backend effect.

Table 11. Agent-judge ratings by construction backend. Cells average over rubric dimensions, then environments; n counts rated environments. For this comparison, Gemini 3.7 Flash uses Claude Code; Appendix B.2 compares its ratings under Claude Code and Antigravity.
<table><tr><td rowspan="2">Agent judge</td><td colspan="2">Claude Opus 5</td><td colspan="2">GPT-5.6-sol</td></tr><tr><td>Mean</td><td>n</td><td>Mean</td><td>n</td></tr><tr><td>Claude Opus 5 (Claude Code)</td><td>4.37</td><td>180</td><td>4.25</td><td>30</td></tr><tr><td>GPT-5.6-sol (Codex)</td><td>4.40</td><td>180</td><td>4.27</td><td>30</td></tr><tr><td>Gemini 3.7 Flash (Claude Code)</td><td>4.88</td><td>180</td><td>4.74</td><td>30</td></tr></table>

Table 12. Median final submitted normalized scores after three sessions by construction backend. The mainbackend column repeats session 3 of Table 4; n counts chains with all three sessions scored. Construction sets are unpaired.
<table><tr><td rowspan="2">Solver model</td><td colspan="2">Claude Opus 5</td><td colspan="2">GPT-5.6-sol</td></tr><tr><td>Median</td><td>n</td><td>Median</td><td>n</td></tr><tr><td>Claude Opus 5</td><td>0.800</td><td>120</td><td>0.784</td><td>30</td></tr><tr><td>GPT-5.6-sol</td><td>0.761</td><td>120</td><td>0.736</td><td>30</td></tr><tr><td>DeepSeek-V4-Flash</td><td>0.647</td><td>120</td><td>0.564</td><td>30</td></tr><tr><td>Claude Haiku 4.5</td><td>0.332</td><td>120</td><td>0.363</td><td>30</td></tr></table>

## D Release and Access

The corpus is available on request. Access is gated because the sealed test split and grading key make grading possible and also make scores obtainable without solving the task. We do not release the sourcing procedure, per-paper verdicts, or rejection reasons, to avoid attaching our assessments of benchmark suitability to individual preprints and their authors.