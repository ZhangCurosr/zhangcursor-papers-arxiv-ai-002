# VCE-Skill: Enhancing Skill Self-Evolution with Version-Change Experience

Jianming Chen<sup>1,2,3,4</sup>, Xuanbin Ye<sup>5</sup>, Yawen Wang<sup>1,2,3,4∗</sup>, Junjie Wang<sup>1,2,3,4∗</sup>, Yuanzhe Hu<sup>1,2,3,4</sup>, Qing Wang<sup>1,2,3,4</sup>, Fanjiang Xu<sup>1,2,3,4∗</sup>

<sup>1</sup> Institute of Software Chinese Academy of Sciences, Beijing, China

<sup>2</sup> Science & Technology on Integrated Information System Laboratory, Beijing, China

<sup>3</sup> State Key Laboratory of Complex System Modeling and Simulation Technology, Beijing, China

<sup>4</sup> University of Chinese Academy of Sciences, Beijing, China

<sup>5</sup> Beijing University of Post and Telecommunications, Beijing, China

## Abstract

Agents increasingly rely on reusable skills to encode task knowledge, tool-use procedures, and validation rules. Existing skill self-evolution methods primarily revise skills using execution trajectories collected from current tasks, leaving the evolution knowledge accumulated in public skill version histories largely untapped. Our pilot study reveals a clear complementarity between the two sources: public skill changes provide reusable evolution priors, whereas trajectories provide evidence grounded in the current task. Motivated by this, we propose VCE-Skill, which distills noisy and implementationspecific public skill changes into reusable, structured versionchange experience and adaptively fuses it with trajectory derived proposals from the base evolver, thereby exploiting external experience while retaining task-specific evidence. Extensive experiments demonstrate that VCE-Skill improves skill self-evolution, increasing mean scores by 3.20–4.98 points; transfer experiments further show that the resulting skills achieve stronger cross-model transfer performance. Our work highlights public skill version changes as a previously underexplored yet efective source of prior knowledge and advances trajectory-driven skill self-evolution.

## 1 Introduction

Large language model agents increasingly rely on external skills as reusable units of task knowledge (Li et al. 2026; Ling, Zhong, and Huang 2026). A skill can package instructions, scripts, resources, and configuration files into a portable bundle that an agent can invoke when solving a class of tasks (Anthropic 2025, 2026a). However, a skill is rarely finished after its first release. As tools change, task distributions shift, and failures accumulate, skill authors revise instructions, patch scripts, add resources, etc. Skill evolution is part of how agent capabilities become reliable over time (Zhao et al. 2024; Wang et al. 2024).

Recent work on self-evolving agents has started to automate this process. A typical skill self-evolution loop executes an agent on tasks, collects trajectories, diagnoses failures, and revises the current skill accordingly (Zhao et al. 2024; Yang et al. 2026). This trajectory-driven paradigm grounds updates in task-specific execution evidence. However, its guidance is bounded by the quality and coverage of the trajectories collected in the current run (Mi et al. 2026). Noisy or low-quality trajectories may make it dificult to derive reliable guidance, while incomplete coverage of failure modes may cause the resulting updates to overfit to the limited observation. Consequently, trajectory-driven evolution may overlook broadly useful update knowledge that is not revealed by current executions (Huang et al. 2026).

A complementary source of evolution knowledge lies in public version histories. Software engineering research has mined historical patches to recover reusable edit and repair patterns (Bader et al. 2019; Koyuncu et al. 2020). Public repositories similarly preserve how developers revise instructions, scripts, and other components during maintenance. These changes may encode update strategies that are not observed in the current task run. Nevertheless, skill version histories have received limited attention as a source of reusable experience for skill self-evolution, and it remains unclear whether they provide knowledge beyond that already produced from trajectories (Xu and Yan 2026; Li et al. 2026).

To validate this premise, we conduct a motivation study that compares matched public and trajectory-derived skill changes, under a unified comparison protocol. The comparison reveals complementary coverage: trajectory-derived changes are highly concentrated in instructions and a narrow set of intents and patterns, whereas public histories cover broader changes. Meanwhile, trajectory-only changes remain, showing that public histories complement rather than replace trajectory-derived evidence. This complementarity does not make historical changes directly usable. Raw difs are fine-grained and repository-specific, and external experience may not always be relevant to the current failure. Exploiting version histories therefore requires addressing two challenges: distilling raw changes into transferable evolution guidance and selectively fusing this guidance with the trajectory-derived proposal(Liu et al. 2024; Shi et al. 2023).

To address these challenges, we propose VCE-Skill, a method using version-change experience (VCE) to enhance Skill self-evolution with two stages. First, Experience Distillation abstracts raw difs into structured update events and progressively generalizes them into update patterns and insights, together forming an experience bank. Second, Optimization with Adaptive Experience Attention selects relevant experience from the bank while a compatible base evolver independently produces a self-proposal from the trajectories. It then fuses the external experience and self-proposal with attention weights. After each optimization based on the fused proposal, adaptation feedback adjusts experience selection and attention allocation for the next iteration.

We evaluate VCE-Skill on 5 benchmarks with 4 LLMs by enhancing 3 skill self-evolution methods and comparing against 2 non-evolutionary baselines. The experiments examine efectiveness, the ablation of VCE-Skill, and the crossmodel transferability of evolved skills. The results show that VCE-Skill consistently improves the base evolvers.

This paper makes the following contributions:

• We conduct the pilot study comparing public changes with trajectory-derived skill changes, revealing their complementary coverage and taking public VCE as an external prior for skill evolution.

• We introduce VCE-Skill, which distills raw version changes into reusable experience and adaptively selects and fuses it with trajectory-derived proposals during iterative skill optimization.

• We conduct extensive experiments, demonstrating the effectiveness and generality of VCE-Skill.

## 2 Background and Related Work

## 2.1 Agent Skills

Following Anthropic’s definition and the Agent Skills specification (Anthropic 2025, 2026a), we view an agent skill as a reusable, filesystem-based capability package that an agent can discover and load on demand for a specialized task. At minimum, a skill is a directory containing a SKILL.md file.

We represent an agent skill as

$$
s = \langle m , \mathcal { C } _ { s } \rangle , \qquad \mathcal { C } _ { s } = \{ c _ { k } \} _ { k = 1 } ^ { K _ { s } } ,\tag{1}
$$

where m denotes the metadata in the YAML frontmatter, including the required name and description, and $\mathcal { C } _ { s }$ denotes the set of components contained in skill s. Each $c _ { k }$ represents a functional component, such as procedural instructions, executable scripts, reference materials, configurations, templates, or other supporting artifacts. The instruction component corresponding to the body of SKILL.md is required, whereas supporting components are optional. During use, the agent first observes m for skill discovery, loads the instruction component when the skill is activated, and accesses other task-relevant components in $\mathcal { C } _ { s }$ as needed.

Prior work studies skill artifacts and libraries from complementary perspectives (Holzbauer et al. 2026; Ling, Zhong, and Huang 2026; Liu et al. 2026b). SkillsBench (Li et al. 2026) evaluates curated and self-generated skills. SkillReducer (Gao et al. 2026) compresses skill descriptions and bodies while preserving behavior. SKILLFOUNDRY (Shen et al. 2026) constructs and iteratively validates skill libraries from heterogeneous scientific resources. Unlike work on evaluation, ecosystem analysis, compression, or resource-toskill construction, we treat adjacent version changes themselves as reusable evolution experience.

## 2.2 Skill Self-Evolution

Skill self-evolution automates the improvement of a skill from execution feedback. A typical loop executes an agent on one or more tasks with the current skill, records the resulting trajectories, diagnoses observed behavior, proposes a candidate update, and evaluates the candidate on held-out tasks or regression cases before accepting it (Shinn et al. 2023; Zhao et al. 2024).

Let $s _ { t }$ be the current skill and $B _ { t }$ the task batch at evolution round t. Execution produces a trajectory set and an evolver generates a candidate skill:

$$
\mathcal { T } _ { t } = \mathrm { E x e c } ( s _ { t } , B _ { t } ) , \qquad \tilde { s } _ { t + 1 } = \mathrm { E v o l v e } ( s _ { t } , \mathcal { T } _ { t } ) .\tag{2}
$$

The candidate is evaluated using validation cases $V _ { t } ,$ to determine the next skill:

$$
s _ { t + 1 } = \operatorname { S e l e c t } ( s _ { t } , { \tilde { s } } _ { t + 1 } ; V _ { t } ) .\tag{3}
$$

The operator selects $\tilde { s } _ { t + 1 }$ only when the candidate satisfies the acceptance criteria on $V _ { t } ;$ otherwise, it retains $s _ { t }$ This recurrence captures the execution–diagnosis–update– validation loop.

Recent methods instantiate trajectory-driven skill selfevolution in diferent ways. EvoSkill (Alzubi et al. 2026) and SkillForge (Liu et al. 2026a) diagnose failed executions to revise skills, while SkillOpt (Yang et al. 2026) and Skills-Coach (Tian et al. 2026) optimize skill artifacts from scored rollouts or task-level evaluations. CoEvoSkills (Zhang et al. 2026) couples multi-file skill generation with a verifier, Skill-Claw (Ma et al. 2026) aggregates interactions across users, SkillOS (Ouyang et al. 2026) learns a repository-curation policy from task streams, and EmbodiSkill (Ju et al. 2026) distinguishes skill defects from execution lapses in embodied trajectories. MetaSkill-Evolve (Wang et al. 2026) further extends this paradigm by recursively improving both task skills and the meta-skills that govern their evolution through two execution-driven timescales. However, their update signals arise mainly from current or accumulated task interactions and evaluations. In contrast, VCE-Skill mines public skill version histories as a previously untapped source of prior knowledge for skill evolution and integrates the distilled experience with trajectory-derived proposals.

## 2.3 Evolution Knowledge from Version Histories

Version histories have long supported reusable edit mining: studies of repetitive revisions, Getafix (Bader et al. 2019), and FixMiner (Koyuncu et al. 2020) recover recurring codechange or repair patterns. However, these approaches model syntax-level code edits and repair contexts (Dilhara et al. 2022; Livshits and Zimmermann 2005), whereas agent skills are heterogeneous, multi-component artifacts whose changes encode semantic intents across instructions, scripts, references, and configurations. Their code-specific representations are therefore not directly applicable to skill evolution. VCE-Skill extends history-based edit mining to agent skills by distilling guidance from cross-repository skill version changes and retrieving relevant experience to complement trajectory-derived proposals.

![](images/3e3e576e1bbf4ef527c659a75f6ef634365efed64a414bdbd5052c24955c32f6.jpg)

(a) BFCL v4 Full  
![](images/5795fa9aa248264ca22f2afdc46ceef3083c44c7e158455e3cce1d1bbab282fc.jpg)  
(b) SearchQA  
Figure 1: Public and trajectory-derived skill changes.

## 3 Motivation Study

Trajectory-based skill evolution derives update guidance from the trajectory collected during the optimization run. However, the resulting guidance is sensitive to trajectory quality: noisy or incomplete trajectories can obscure failure causes and provide limited actionable guidance. Moreover, this evidence is bounded by the failure modes exposed by the sampled rollouts. Consequently, useful update strategies that do not arise in the current executions may remain unobserved. We provide a supplementary analysis of trajectoryquality sensitivity in the Appendix.

Public skill version histories provide potential complementary knowledge. They record skill changes adopted by developers during maintenance, and therefore constitute a practically grounded and rich source of evolution knowledge. However, it remains unclear whether these public changes merely repeat the update strategies that trajectory-based evolution can produce adequately, or contain additional update knowledge. This distinction directly determines the motivation for our method: ifthe two sources exhibit complementary update coverage, version histories can serve as an external prior for trajectory-based evolution.

We therefore ask: Do public skill version histories contain evolution knowledge that is not adequately captured by trajectory-based skill evolution? To answer this question, we compare the actual skill changes produced by the two evolution processes under a unified analysis protocol.

## 3.1 Data and Comparison Protocol

Data Sources. The public source consists of changes between adjacent versions collected from GitHub and ClawHub. The trajectory source consists of changes produced through the evolution of several representative methods: EvoSkill (Alzubi et al. 2026), SkillClaw (Ma et al. 2026), and SkillOpt (Yang et al. 2026). For both sources, we analyze paired skills between changes. We match the two sources within each task domain and construct a comparison corpus containing 400 skill-change units from each source.

Unified Comparison Protocol. We segment each skill pair into skill-change units by grouping edits that implement the same modification intent and pattern. Each unit may involve multiple components but is assigned one intent and one pattern. An LLM judge annotates both sources using the same domain-specific taxonomy. The taxonomy distinguishes where an edit is made (component), why it is introduced (intent), and which reusable modification strategy it follows (pattern). Multiple authors manually verify a sample of the annotations, with all reviewed cases passing the verification (100%). We compare the resulting distributions and identify categories observed in both sources as shared, and those observed in only one source as public-only or trajectory-only. Full details are provided in the Appendix.

## 3.2 Key Observations

Complementary Coverage across Evolution Sources. Figure 1 shows that trajectory-derived changes are highly concentrated: 372 of 400 BFCL units and 379 of 400 SearchQA units afect instructions. In contrast, public changes are distributed more broadly across diferent components. At the semantic level, public changes also introduce intent and pattern absent from the trajectory source. BFCL contains 5 public-only intent categories and 4 public-only pattern families, while SearchQA contains 10 public-only intent categories and 6 public-only patterns.

These results show that public version histories do not merely duplicate the updates produced from online trajectories. Though the two sources share a core set of update categories, public changes extend the scope of evolution beyond what is exposed by current rollouts. Meanwhile, BFCL also contains trajectory-only intents and patterns, indicating that public version histories cannot replace the task-grounded evidence obtained during online evolution.

Design Implication. Overall, public version histories provide evolution knowledge that is not fully covered by trajectory-derived changes, while trajectory evidence remains directly grounded in the target task. This complementarity motivates integrating public version-change experience as an external prior through adaptive fusion.

## 4 VCE-Skill Method

Figure 2 presents VCE-Skill, a framework that transfers experience from historical skill versions to guide the selfevolution of a target skill. The framework consists of two stages. (1) Experience Distillation distills historical version changes from public skill repositories into reusable experience and organizes it into a structured experience bank. (2) Optimization with Adaptive Experience Attention selects task-relevant experience from this bank and adaptively combines it with the current skill’s own evolution proposal to guide iterative skill improvement. The details of the LLM judge used in VCE-Skill are provided in the Appendix.

![](images/55bef3a222b67647abcdf943ae121655e2e64f965afc2df4708a0d9f77363663.jpg)  
Figure 2: Overview of VCE-Skill. The left stage distills the version-change experience from public skill versions. The right stage enhances the skill self-evolution by allocating adaptive attention between external experience and trajectory-derived proposals.

## 4.1 Experience Distillation

Raw version difs encode concrete, repository-specific implementations rather than directly reusable evolution guidance. We therefore abstract version changes into update events and distill them to constructive experience. Figure 3 provides a concrete example of how VCE-Skill converts a pair of skill versions into structured update events and organizes the distilled knowledge.

Change Abstraction. As illustrated in Figure 3, file-level edits between two versions are converted into an update event that separates the concrete raw dif from its component, operation, and intent. Specifically, for each collected skill, VCE-Skill orders its versions chronologically and compares each pair of adjacent versions to obtain a raw dif. It then decomposes each dif into individual changes, each of which is represented as an update event:

$$
\boldsymbol { u } _ { j } = \langle d _ { j } , \{ c _ { j } \} , o _ { j } , i _ { j } \rangle ,\tag{4}
$$

where $d _ { j }$ is the raw difsegment corresponding to the event, containing its specific editions. The field $\{ c _ { j } \}$ identifies the afected components, such as instructions, scripts, etc. The field $o _ { j }$ describes the semantic operation performed by the change, such as adding an input-existence check. The field $i _ { j }$ records the update intent, such as preventing execution when a required input is missing. This structured representation supports the subsequent abstraction of update events into reusable version-change experience.

Experience Generation. As shown in Figure 3, VCE-Skill then transforms update events into reusable knowledge at three levels. First, within each skill’s version history, it groups related events based on their components, operations, and intents, removes repository-specific details, and summarizes each group as an update pattern. Second, it synthesizes the resulting patterns for each skill into evolution insights that characterize recurring update strategies in that skill’s history. Third, it generalizes common evolution insights across skills in the same task domain into domain insights that capture transferable principles for skill optimization.

The resulting entries from all collected skills are aggregated into the experience bank B. It contains multiple entries at each level, spanning diferent skills and task domains. During skill self-optimization, the bank serves as an external knowledge corpus from which VCE-Skill selects experiences relevant to the current task domain and skill state. All LLM-based distillation steps use fixed prompt and structured outputs; the details are provided in supplementary materials.

## 4.2 Optimization with Adaptive Experience Attention

Experience Selection. At iteration $t > 1$ , VCE-Skill constructs a selection context from the target task, the current skill, and the unified feedback from the preceding update:

$$
z _ { t } = \mathrm { C o n t e x t } ( T , S _ { t - 1 } , f _ { t - 1 } ) .\tag{5}
$$

where $\tau$ denotes the target task, $S _ { t - 1 }$ is the current skill, and $f _ { t - 1 }$ is the feedback from the preceding update, which will be introduced later. The Context(·) function organizes them into a structured textual representation $z _ { t }$ that serves as the query for experience selection.

VCE-Skill then uses an LLM-based selector to semantically compare $z _ { t }$ with the entries in $\boldsymbol { B }$ and select at most K experiences:

$$
\begin{array} { r } { \mathcal { X } _ { t } = \mathrm { S e l e c t } _ { \le K } ( \mathcal { B } , z _ { t } ) , \qquad p _ { t } ^ { \mathrm { e x p } } = \mathrm { A g g r e g a t e } ( \mathcal { X } _ { t } ) , } \end{array}\tag{6}
$$

where K is the selection budget and $p _ { t } ^ { \mathrm { e x p } }$ is the compact external experience guidance aggregated from the selected entries. Selection remains conditioned primarily on compatibility with the current task and skill, while $f _ { t - 1 }$ adjusts the selector’s preference for external experience. Positive feedback favors external guidance, whereas negative feedback makes external selection more conservative. This makes selection adaptive across iterations: after the skill is updated, the new skill state and unified feedback change $z _ { t + 1 }$ and therefore the selected experiences.

![](images/b6f0b63441c40798ad2e47ca99e3e6d84fdfda27367d02bac8ab8787ef7bc6b4.jpg)  
Figure 3: Illustration of experience distillation.

Attention Fusion for Skill Optimizer. In parallel, the base evolver uses the current skill and its execution trajectory to produce a self proposal:

$$
p _ { t } ^ { \mathrm { s e l f } } = \mathrm { B a s e E v o l v e r } ( S _ { t - 1 } , \tau _ { t - 1 } ) ,\tag{7}
$$

where $\tau _ { t - 1 }$ is the trajectory produced by executing $S _ { t - 1 } .$ The base evolver identifies potential improvements from the successes, failures, and optimization needs reflected in this trajectory, and proposes edits for producing $S _ { t } .$ . Thus, the self proposal is grounded in the current skill’s own execution, whereas $p _ { t } ^ { \mathrm { e x p } }$ provides reusable guidance distilled from external skill histories.

Adaptive attention assigns source-level weights to external experience guidance and the self proposal:

$$
\lambda _ { \mathrm { e x p } } ^ { ( t ) } + \lambda _ { \mathrm { s e l f } } ^ { ( t ) } = 1 , \qquad 0 \le \lambda _ { \mathrm { e x p } } ^ { ( t ) } , \lambda _ { \mathrm { s e l f } } ^ { ( t ) } \le 1 .\tag{8}
$$

The weights serve as prompt-level reliance instructions for the LLM-based fuser, rather than as probabilities. Both weights are initialized to 0.5. The fuser combines compatible guidance and prioritizes the higher-weight source when the two sources conflict. It produces:

$$
p _ { t } ^ { \mathrm { e n h } } = \mathrm { F u s e } \left( p _ { t } ^ { \mathrm { e x p } } , p _ { t } ^ { \mathrm { s e l f } } ; \lambda _ { \mathrm { e x p } } ^ { ( t ) } , \lambda _ { \mathrm { s e l f } } ^ { ( t ) } \right) ,\tag{9}
$$

where $p _ { t } ^ { \mathrm { e n h } }$ is the experience-enhanced proposal, represented as a structured set of candidate edits.

To preserve source evidence for subsequent feedback, each selected external experience and each self-proposal item is assigned a unique provenance identifier, such as EXP-k or SELF-k, which will be preserved in the fusion. These editlevel provenance tags are later compared with the actual skill update to determine which proposal items were realized and whether the implemented update was primarily supported by external experience or by the self proposal.

The skill optimizer applies the enhanced proposal to the current skill:

$$
S _ { t } = \mathbf { O } ( S _ { t - 1 } , p _ { t } ^ { \mathrm { e n h } } ) .\tag{10}
$$

Depending on the proposal, the LLM-based optimizer revises the skill bundle to produce $S _ { t }$

Adaptation Feedback. The agent then executes $S _ { t } ,$ producing a new trajectory $\tau _ { t }$ for the base evolver in the next iteration. Separately, the completed skill update and its validation produce two intermediate signals: source feedback $f _ { t } ^ { S }$ , which summarizes whether the implemented edits rely more on external experience or on the self proposal, and validation feedback $f _ { t } ^ { \dot { V } }$ , which measures whether the updated skill improves validation performance.

Because the optimizer may realize only part of the enhanced proposal, VCE-Skill compares the actual skill change with the proposal’s provenance-tagged edits:

$$
\Delta S _ { t } = \operatorname { D i f f } ( S _ { t } , S _ { t - 1 } ) ,\tag{11}
$$

Let $\mathcal { C } _ { t } ~ = ~ \{ c _ { t , k } \} _ { k = 1 } ^ { n _ { t } }$ denote the provenance-tagged candidate edits in $p _ { t } ^ { \mathrm { e n h } }$ . The attribution function semantically aligns each candidate edit with the realized changes and assigns an item-level attribution:

$$
a _ { t , k } = \mathrm { A t t r } ( \Delta S _ { t } , c _ { t , k } ) ,\tag{12}
$$

where $a _ { t , k } \in \{ - 1 , 0 , + 1 \}$ for $k = 1 , \dots , n _ { t }$ . We set $a _ { t , k } =$ +1 when an externally supported candidate edit is realized, $a _ { t , k } = - 1$ when a self-supported candidate edit is realized, and $a _ { t , k } = 0$ when the candidate edit is not realized, cannot be reliably matched, or has mixed provenance. The source feedback is the mean attribution over all candidate edits:

$$
f _ { t } ^ { S } = \frac { 1 } { n _ { t } } \sum _ { k = 1 } ^ { n _ { t } } a _ { t , k } , \qquad f _ { t } ^ { S } \in [ - 1 , 1 ] .\tag{13}
$$

Positive values indicate greater realized contribution from external experience, negative values indicate greater contribution from the self proposal, and values near zero indicate balanced, weak, or uncertain attribution.

Second, validation feedback $f _ { t } ^ { V }$ represents the change in skill performance on a fixed validation set $V { : }$

$$
\begin{array} { r } { f _ { t } ^ { V } = \left\{ \begin{array} { l l } { + 1 , } & { \Delta _ { t } ^ { V } > 0 , } \\ { - 1 , } & { \Delta _ { t } ^ { V } \leq 0 , } \end{array} \right. } \end{array}\tag{14}
$$

where $\Delta _ { t } ^ { V }$ denotes the performance change of validation. We combine source attribution and validation performance into a unified source-preference feedback:

$$
f _ { t } = f _ { t } ^ { V } f _ { t } ^ { S } , \qquad f _ { t } \in [ - 1 , 1 ] .\tag{15}
$$

Positive $f _ { t }$ shifts the next iteration toward external experience, negative $f _ { t }$ shifts it toward the self proposal, and values near zero leave the preference nearly unchanged. The same feedback is included in the next selection context and used to update the attention weights:

$$
\begin{array} { r c l } { { \lambda _ { \mathrm { e x p } } ^ { ( t + 1 ) } } } & { { = } } & { { \mathrm { c l i p } \left( \lambda _ { \mathrm { e x p } } ^ { ( t ) } + \eta f _ { t } , 0 . 3 , 0 . 7 \right) , } } \\ { { \lambda _ { \mathrm { s e l f } } ^ { ( t + 1 ) } } } & { { = } } & { { 1 - \lambda _ { \mathrm { e x p } } ^ { ( t + 1 ) } , } } \end{array}\tag{16}
$$

![](images/5921fc03ec93c65ec270a59edbd3f2594f114f9b7e04e9a40d6c155d47f4fc92.jpg)  
Figure 4: The performance of diferent variants of VCE-Skill.

where the adjustment step size is fixed to $\eta ~ = ~ 0 . 1$ , and clipping to [0.3, 0.7] prevents either source from being permanently ignored. When an improving update relies more on one source, the method increases reliance on that source in proportion to $| f _ { t } ^ { S } |$ . When a non-improving update relies more on one source, it shifts reliance toward the other.

The loop is modular with respect to the base evolver. Diferent trajectory-driven self-evolution methods can supply $q _ { t } ^ { \mathrm { s e l f } }$ without changing the experience bank, feedbackconditioned selector, or source-level attention update.

## 5 Experimental Evaluation

## 5.1 Experimental Setup

Benchmarks and Models. We evaluate on five benchmarks that cover complementary forms of agent skill use: SearchQA (Dunn et al. 2017) for open-domain question answering under noisy retrieval; OficeQA (Team 2025) for document, table, and numerical reasoning; ALF-World (Shridhar et al. 2021) for long-horizon embodied task execution; Spreadsheet (Ma et al. 2024) for spreadsheet operations and calculation; and BFCL-v4 (Patil et al. 2025) for function calling and tool selection.

We take four models as agent: Qwen3.5-27B (Qwen Team 2026), GPT-5.2 (OpenAI 2025), DeepSeek-v3.2 (DeepSeek-AI 2025), and claude-sonnet-5 (Anthropic 2026b).

Compared baselines. We compare VCE-Skill against five groups of baselines. No Skill executes the target agent without loading any skill. LLM Skill uses a one-shot skill generated by an LLM without iterative refinement. EvoSkill (Alzubi et al. 2026) evolves skills from failure trajectories. SkillClaw (Ma et al. 2026) represents collective or multi-skill evolution. SkillOpt (Yang et al. 2026) treats the skill as an optimizable textual state and accepts bounded edits through a validation gate. VCE-Skill enhances these iterative base self-evolvers to compare.

Implementation and Evaluation Details. Each experimental configuration is independently run three times, and we report the mean results in the main paper. We conduct sensitivity analyses of the key hyperparameters. To assess reliability, we manually audit all LLM-based judging. The whole results and details, such as the prompts, LLMs, and environments, are provided in the Appendix.

## 5.2 Performance of VCE-Skill.

Table 1 presents paired comparisons between each selfevolution method and its VCE-enhanced variant. The gray rows denote variants augmented with VCE-Skill, while the parenthesized values show their improvements or declines relative to the corresponding non-VCE baseline. Introducing VCE improves every task score of each base evolver. It also increases the average score by 3.20–4.98 points. These results show that the benefit of VCE-Skill is consistent across different skill-evolution frameworks rather than being specific to a particular method.

These improvements arise because VCE-Skill leverages external evolution experience that is complementary to the agent’s own trajectory-derived knowledge. Experience distillation makes this knowledge reusable and actionable, while adaptive experience selection and attention allocation improve its alignment with the current evolution process.

For each benchmark, VCE-Skill incurs approximately 0.5M tokens for the one-time experience distillation and 0.6M tokens for a complete evolution comprising approximately 20 iterations, resulting in a total additional cost of approximately 1.1M tokens. This corresponds to approximately 10% additional token overhead relative to the bestperforming baseline.

## 5.3 Ablation Study.

Variants. We conduct the ablation study and construct two variants of VCE-Skill, each removing one key component. w/o Dist removes experience distillation and directly uses raw version-change records as external evolution knowledge. w/o Atte replaces adaptive experience attention with a fixed equal-weight combination of the distilled external experience and the self proposal generated by base evolver.

Results. Figure 4 reports the average score on all benchmarks of removing diferent components of VCE-Skill, using the best baseline, SkillOpt, as the base evolver. Removing experience distillation causes the largest performance degradation, with w/o Dist even falling below SkillOpt on all four models. Raw version-change records contain implementation details that may not transfer directly to the current skill. Experience distillation abstracts these records into reusable patterns and insights, thereby reducing irrelevant information and providing more actionable guidance. Replacing adaptive attention with fixed equal weights also consistently underperforms VCE-Skill, although w/o Atte still outperforms SkillOpt on three of the four models. This result indicates that distilled external experience is generally useful, but its contribution should not remain constant throughout evolution. As the skill state changes across iterations, the relevance of external experience and the reliability of the self proposal also vary. Adaptive experience attention accounts for these and allocates the two types of knowledge accordingly, enabling more targeted and reliable skill updates.

## 5.4 Transferability of Skills.

This experiment investigates whether skills produced through self-evolution remain efective when transferred across models. We evolve two skills on the same source model using SkillOpt and SkillOpt augmented with VCE-Skill, respectively, and then transfer them to other models for evaluation. Table 2 reports their transferred performance when Qwen3.5-27B is used as the source model.

<table><tr><td>Model</td><td>Skill</td><td>SearchQA</td><td>OfficeQA</td><td>ALFWorld</td><td>Spreadsheet</td><td>BFCL-v4</td><td>Avg.</td></tr><tr><td rowspan="6">Qwen3.5-27B</td><td>No Skill</td><td>68.42</td><td>33.79</td><td>52.87</td><td>26.93</td><td>54.75</td><td>47.35</td></tr><tr><td>LLM Skill</td><td>67.29</td><td>33.41</td><td>54.74</td><td>29.48</td><td>55.12</td><td>48.01</td></tr><tr><td>EvoSkill</td><td>67.08</td><td>34.13</td><td>57.46</td><td>39.45</td><td>58.45</td><td>51.31</td></tr><tr><td>EvoSkill+VCE</td><td>70.46(+3.38)</td><td>41.79(+7.66)</td><td>59.13(+1.67)</td><td>45.13 (+5.68)</td><td>60.71 (+2.26)</td><td>55.44(+4.13)</td></tr><tr><td>SkillClaw</td><td>69.41</td><td>36.12</td><td>59.01</td><td>42.46</td><td>61.23</td><td>53.65</td></tr><tr><td>SkillClaw+VCE SkillOpt</td><td>72.48(+3.07) 74.27</td><td>41.48(+5.36) 46.12</td><td>62.48 (+3.47)</td><td>46.73(+4.27) 49.08</td><td>63.77 (+2.54)</td><td>57.39 (+3.74)</td></tr><tr><td></td><td>SkillOpt+VCE</td><td>78.86(+4.59)</td><td>59.70 49.42(+3.30)</td><td>63.43(+3.73)</td><td>54.82 52.14(+3.06)</td><td>57.52(+2.70)</td><td>56.80 60.27 (+3.47)</td></tr><tr><td rowspan="6">GPT-5.2</td><td>No Skill</td><td>72.67</td><td>36.57</td><td>69.76</td><td>41.07</td><td>51.13</td><td>54.24</td></tr><tr><td>LLM Skill</td><td>73.75</td><td>43.49</td><td>71.04</td><td>41.84</td><td>53.44</td><td>56.71</td></tr><tr><td>EvoSkill</td><td>76.16</td><td>41.28</td><td>70.45</td><td>43.96</td><td>59.11</td><td>58.19</td></tr><tr><td>EvoSkill+VCE</td><td>80.71 (+4.55)</td><td>48.41 (+7.13)</td><td>74.84(+4.39)</td><td>47.02(+3.06)</td><td>64.85(+5.74)</td><td>63.17(+4.98)</td></tr><tr><td>SkillClaw SkillClaw+VCE</td><td>79.40</td><td>52.09</td><td>76.94</td><td>51.06</td><td>60.42</td><td>63.98</td></tr><tr><td>SkillOpt</td><td>84.14(+4.74)</td><td>55.49 (+3.40)</td><td>80.40(+3.46)</td><td>56.41 (+5.35)</td><td>64.17(+3.75)</td><td>68.12(+4.14)</td></tr><tr><td></td><td>83.11 SkillOpt+VCE 85.86 (+2.75)</td><td>52.16 56.71 (+4.55)</td><td>81.06 84.58 (+3.52)</td><td>54.79 59.60 (+4.81)</td><td>64.16 67.47(+3.31)</td><td>67.06 70.84(+3.78)</td></tr><tr><td rowspan="6">DeepSeek-v3.2</td><td>No Skill</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLM Skill</td><td>70.46</td><td>35.54</td><td>66.24</td><td>29.46</td><td>53.41</td><td>51.02</td></tr><tr><td>EvoSkill</td><td>70.20 72.04</td><td>34.16</td><td>69.67</td><td>30.79</td><td>55.37</td><td>52.04</td></tr><tr><td>EvoSkill+VCE</td><td>74.97 (+2.93)</td><td>38.16</td><td>68.06</td><td>32.72</td><td>56.15</td><td>53.43</td></tr><tr><td>SkillClaw</td><td>73.47</td><td>42.53 (+4.37) 41.06</td><td>71.64(+3.58)</td><td>35.48 (+2.76)</td><td>58.61 (+2.46)</td><td>56.65 (+3.22)</td></tr><tr><td>SkillClaw+VCE</td><td>76.98 (+3.51)</td><td>42.68 (+1.62)</td><td>69.40 72.87 (+3.47)</td><td>34.94 37.48 (+2.54)</td><td>59.27</td><td>55.63</td></tr><tr><td>SkillOpt</td><td>73.82</td><td>45.55</td><td>68.08</td><td>37.78</td><td>64.16(+4.89)</td><td>58.83 (+3.20)</td></tr><tr><td>SkillOpt+VCE</td><td>77.68(+3.86)</td><td>49.01 (+3.46)</td><td>73.70 (+5.62)</td><td>42.02 (+4.24)</td><td>61.36 64.58(+3.22)</td><td>57.32</td></tr><tr><td>No Skill</td><td></td><td></td><td></td><td></td><td></td><td>61.40 (+4.08)</td></tr><tr><td>LLM Skill</td><td>76.40</td><td>51.56</td><td>73.16</td><td>45.06</td><td>57.42</td><td>60.72</td></tr><tr><td rowspan="8">Claude Sonnet 5</td><td></td><td>76.46</td><td>50.77</td><td>74.14</td><td>50.05</td><td>58.76</td><td>62.04</td></tr><tr><td>EvoSkill</td><td>74.83</td><td>52.03</td><td>72.45</td><td>55.16</td><td>59.57</td><td></td></tr><tr><td>EvoSkill+VCE</td><td>78.92 (+4.09)</td><td>56.22 (+4.19)</td><td>77.94 (+5.49)</td><td>59.70(+4.54)</td><td>63.98(+4.41)</td><td>62.81</td></tr><tr><td>SkillClaw</td><td>81.02</td><td>53.21</td><td>78.10</td><td>50.43</td><td>65.93</td><td>67.35 (+4.54) 65.74</td></tr><tr><td>SkillClaw+VCE</td><td>84.56(+3.54)</td><td>56.88 (+3.67)</td><td>82.04(+3.94)</td><td>55.30(+4.87)</td><td>68.37(+2.44)</td><td>69.43 (+3.69)</td></tr><tr><td>SkillOpt</td><td>82.71</td><td>54.17</td><td>83.12</td><td>57.91</td><td>66.06</td><td>68.79</td></tr><tr><td>SkillOpt+VCE</td><td>87.13(+4.42)</td><td>58.44(+4.27)</td><td>85.70 (+2.58)</td><td>62.04(+4.13)</td><td>70.04(+3.98)</td><td>72.67 (+3.88)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 1: Performance comparison on diferent models and benchmarks. For VCE-enhanced variants, red (green) parenthesized values indicate improvements (declines) over their corresponding baseline. Bold indicates the best result.

<table><tr><td>Target Model</td><td>|Source Skill</td><td>|SearchQA OfficeQA ALFWorld Spreadsheet BFCL-v4 Avg.</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">GPT-5.2</td><td>|SkillOpt</td><td>|78.01</td><td>47.47</td><td>79.16</td><td>52.11</td><td>64.10</td><td>64.17</td></tr><tr><td>SkillOpt+VCE</td><td>83.24</td><td>55.34</td><td>86.05</td><td>57.89</td><td>68.54</td><td>70.21 (+6.04)</td></tr><tr><td rowspan="2">DeepSeek-v3.2</td><td>|SkillOpt</td><td>|71.16</td><td>44.94</td><td>69.60</td><td>38.16</td><td>62.10</td><td>57.19</td></tr><tr><td>SkillOpt+VCE</td><td>78.12</td><td>49.53</td><td>75.11</td><td>43.94</td><td>65.48</td><td>62.44(+5.25)</td></tr><tr><td rowspan="2">Claude Sonnet 5</td><td>SkillOpt</td><td>81.10</td><td>53.03</td><td>83.69</td><td>56.01</td><td>66.42</td><td>68.05</td></tr><tr><td>SkillOpt+VCE</td><td>87.37</td><td>58.14</td><td>86.80</td><td>61.73</td><td>71.97</td><td>73.20(+5.15)</td></tr></table>

Table 2: Cross-model transfer performance of skills.

Results. The skill evolved with VCE-Skill outperforms its SkillOpt counterpart. Specifically, VCE-Skill improves the average scores of the three target models by 6.04, 5.25, and 5.15 points, respectively. Notably, this transfer gain exceeds the gain over the available paired non-VCE counterparts in Table 1. It demonstrates that the skills evolved by VCE-Skill possess stronger cross-model transferability.

We attribute it to the more generalizable evolution knowledge incorporated by VCE-Skill. Trajectory-driven evolution primarily learns from source-model trajectories and may therefore retain model-specific behavior patterns. In contrast, VCE-Skill introduces reusable external experience. By suppressing model-specific trajectory details and incorporating generalizable task-solving principles, this external experience makes the evolved skill less dependent on the source model and more reliable when applied to other models.

## 6 Conclusion

In this work, we propose VCE-Skill, which enhances skill self-evolution with public skill versions. Our pilot study shows that public changes and those derived from trajectoryderived self-evolution provide complementary coverage. VCE-Skill distills public skill changes into structured prior experience and adaptively fuses it with trajectory-derived proposals from the base evolver, combining reusable prior knowledge with task-specific execution evidence to guide skill evolution. Experiments across five benchmarks, four LLMs, and three base evolvers demonstrate that VCE-Skill efectively improves skill self-evolution and yields skills with stronger cross-model transferability in the evaluated setting. Overall, our work highlights public skill version histories as an underexplored yet efective source of prior knowledge and provides a practical experience-augmented extension to the trajectory-driven paradigm of skill self-evolution.

## References

Alzubi, S.; Provenzano, N.; Bingham, J.; Chen, W.; and Vu, T. 2026. EvoSkill: Automated Skill Discovery for Multi-Agent Systems. arXiv:2603.02766.

Anthropic. 2025. Introducing Agent Skills.

Anthropic. 2026a. Agent Skills.

Anthropic. 2026b. Introducing Claude Sonnet 5.

Bader, J.; Scott, A.; Pradel, M.; and Chandra, S. 2019. Getafix: learning to fix bugs automatically. Proc. ACM Program. Lang.

DeepSeek-AI. 2025. DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models. arXiv:2512.02556.

Dilhara, M.; Ketkar, A.; Sannidhi, N.; and Dig, D. 2022. Discovering repetitive code changes in python ML systems. In Proceedings of the 44th International Conference on Software Engineering, ICSE ’22, 736–748.

Dunn, M.; Sagun, L.; Higgins, M.; Guney, V. U.; Cirik, V.; and Cho, K. 2017. SearchQA: A New Q&A Dataset Augmented with Context from a Search Engine. arXiv:1704.05179.

Gao, Y.; Li, Z.; Yuan, Y.; Ji, Z.; Ma, P.; and Wang, S. 2026. SkillReducer: Optimizing LLM Agent Skills for Token Eficiency. arXiv:2603.29919.

Holzbauer, F.; Schmidt, D.; Gegenhuber, G. K.; Schrittwieser, S.; and Ullrich, J. 2026. Context Matters: Repository-Aware Security Analysis of the Agent Skill Ecosystem. In Agent Skills ’26 Workshop: ACM Conference on AI and Agentic Systems.

Huang, Z.; Xu, J.; Yang, Y.; Gong, Z.; Yang, Q.; Tian, M.; Wang, X.; Lv, C.; Gao, X.; Dai, Q.; Liu, B.; Qiu, K.; Yang, X.; Chen, D.; Zheng, X.; and Luo, C. 2026. From Raw Experience to Skill Consumption: A Systematic Study of Model-Generated Agent Skills. arXiv:2605.23899.

Ju, R.; Wang, X.; Ding, X.; Yang, Y.; Wu, H.; Jiang, S.; Zhang, Q.; Wen, H.; Li, X.; Wang, W.; Li, K.; Liu, Y.; Dai, H.; Wang, W.; and Cao, T. 2026. EmbodiSkill: Skill-Aware Reflection for Self-Evolving Embodied Agents. arXiv:2605.10332.

Koyuncu, A.; Liu, K.; Bissyandé, T. F.; Kim, D.; Klein, J.; Monperrus, M.; and Le Traon, Y. 2020. FixMiner: Mining relevant fix patterns for automated program repair. Empirical Softw. Engg.

Li, X.; Liu, Y.; Chen, W.; You, B.; Di, Z.; He, Y.; Zheng, S.; Choe, K. W.; Sun, J.; Wang, S.; Tao, C.; Li, B.; Zhao, X.;

Tan, Z.; Shi, C.; Tang, X.; Tankasala, S.; Yuan, B.; Qian, Y.; Tu, J.; Wang, C.; Sun, Y.; Wang, W.; Taylor, A.; Yang, Z.; Guan, C.; Dong, Z.; Zhang, X.; Dillmann, S.; chung Lee, H.; and Song, D. 2026. SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks. arXiv:2602.12670.

Ling, G.; Zhong, S.; and Huang, R. 2026. Agent Skills: A Data-Driven Analysis of Claude Skills for Extending Large Language Model Functionality. arXiv:2602.08004.

Liu, N. F.; Lin, K.; Hewitt, J.; Paranjape, A.; Bevilacqua, M.; Petroni, F.; and Liang, P. 2024. Lost in the Middle: How Language Models Use Long Contexts. Transactions of the Associationfor Computational Linguistics, 12: 157–173.

Liu, X.; Luo, X.; Li, L.; Huang, G.; Liu, J.; and Qiao, H. 2026a. SkillForge: Forging Domain-Specific, Self-Evolving Agent Skills in Cloud Technical Support. In Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval, 4763–4768. ACM.

Liu, Y.; Wang, W.; Feng, R.; Zhang, Y.; Xu, G.; Deng, G.; Li, Y.; and Zhang, L. 2026b. Agent Skills in the Wild: An Empirical Study of Security Vulnerabilities at Scale. arXiv:2601.10338.

Livshits, B.; and Zimmermann, T. 2005. DynaMine: finding common error patterns by mining software revision histories. In Proceedings of the 10th European Software Engineering Conference Held Jointly with 13th ACM SIGSOFT Interna tional Symposium on Foundations of Software Engineering, 296–305.

Ma, Z.; Yang, S.; Ji, Y.; Wang, X.; Wang, Y.; Hu, Y.; Huang, T.; and Chu, X. 2026. SkillClaw: Let Skills Evolve Collectively with Agentic Evolver. arXiv:2604.08377.

Ma, Z.; Zhang, B.; Zhang, J.; Yu, J.; Zhang, X.; Zhang, X.; Luo, S.; Wang, X.; and Tang, J. 2024. SpreadsheetBench: Towards Challenging Real World Spreadsheet Manipulation. In Advances in Neural Information Processing Systems.

Mi, Q.; Ma, Z.; Yang, M.; Li, H.; Wang, Y.; Zhang, H.; and Wang, J. 2026. Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents. In Proceedings of the 43rd International Conference on Machine Learning (ICML 2026). Spotlight.

OpenAI. 2025. Introducing GPT-5.2.

Ouyang, S.; Yan, J.; Chen, Y.; Han, R.; Wang, Z.; Mishra, B. D.; Meng, R.; Li, C.-L.; Jiao, Y.; Zha, K.; Shen, M.; Tirumalashetty, V.; Lee, G.; Han, J.; Pfister, T.; and Lee, C.- Y. 2026. SkillOS: Learning Skill Curation for Self-Evolving Agents. arXiv:2605.06614.

Patil, S. G.; Mao, H.; Yan, F.; Ji, C. C.-J.; Suresh, V.; Stoica, I.; and Gonzalez, J. E. 2025. The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models. In Proceedings of the 42nd International Conference on Machine Learning.

Qwen Team. 2026. Qwen3.5: Towards Native Multimodal Agents.

Shen, S.; Cheng, W.; Ma, M.; Turcan, A.; Zhang, M. J.; and Ma, J. 2026. SKILLFOUNDRY: Building Self-Evolving Agent Skill Libraries from Heterogeneous Scientific Resources. arXiv:2604.03964.

Shi, F.; Chen, X.; Misra, K.; Scales, N.; Dohan, D.; Chi, E. H.; Schärli, N.; and Zhou, D. 2023. Large Language Models Can Be Easily Distracted by Irrelevant Context. In Proceedings of the 40th International Conference on Machine Learning, volume 202, 31210–31227.

Shinn, N.; Cassano, F.; Gopinath, A.; Narasimhan, K.; and Yao, S. 2023. Reflexion: language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems.

Shridhar, M.; Yuan, X.; Côté, M.-A.; Bisk, Y.; Trischler, A.; and Hausknecht, M. 2021. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. In Proceedings of the International Conference on Learning Representations (ICLR).

Team, T. D. A. R. 2025. Introducing OficeQA: A benchmark for end-to-end grounded reasoning.

Tian, Y.; Chen, J.; Zheng, L.; Tao, M.; Zeng, X.; Yin, Z.; Su, H.; and Sun, X. 2026. Skills-Coach: A Self-Evolving Skill Optimizer via Training-Free GRPO. arXiv:2604.27488.

Wang, G.; Xie, Y.; Jiang, Y.; Mandlekar, A.; Xiao, C.; Zhu, Y.; Fan, L.; and Anandkumar, A. 2024. Voyager: An Open-Ended Embodied Agent with Large Language Models. Transactions on Machine Learning Research.

Wang, Z.; Yan, M.; Bi, J.; Yan, S.; Tresp, V.; and Ma, Y. 2026. MetaSkill-Evolve: Recursive Self-Improvement of LLM Agents via Two-Timescale Meta-Skill Evolution. arXiv:2607.05297.

Xu, R.; and Yan, Y. 2026. Agent Skills for Large Language Models: Architecture, Acquisition, Security, and the Path Forward. arXiv:2602.12430.

Yang, Y.; Gong, Z.; Huang, W.; Yang, Q.; Zhou, Z.; Huang, Z.; Li, Y.; Gao, X.; Dai, Q.; Liu, B.; et al. 2026. Skillopt: Executive strategy for self-evolving agent skills. arXiv preprint arXiv:2605.23904.

Zhang, H.; Fan, S.; Zou, H. P.; Chen, Y.; Wang, Z.; Zhou, J.; Li, C.; Huang, W.-C.; Yao, Y.; Zheng, K.; Liu, X.; Li, X.; and Yu, P. S. 2026. CoEvoSkills: Self-Evolving Agent Skills via Co-Evolutionary Verification. arXiv:2604.01687.

Zhao, A.; Huang, D.; Xu, Q.; Lin, M.; Liu, Y.-J.; and Huang, G. 2024. ExpeL: LLM agents are experiential learners. In Proceedings of the Thirty-Eighth AAAI Conference on Artificial Intelligence and Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence and Fourteenth Symposium on Educational Advances in Artificial Intelligence.

# Appendix for VCE-Skill

This supplementary appendix provides the motivation-study protocol, implementation details, reproducibility information, extended analyses, and structured interfaces referenced by the main paper. It is organized around three aspects: the motivation study, the mechanism of VCE-Skill, and the experimental details.

## A Motivation Study Protocol and Supplementary Analyses

## A.1 Matched Source Comparison

The matched study examines how public version histories and trajectory-driven evolution difer in the skill-update knowledge they expose. Both sources use the same skillchange unit and the same component, intent, and pattern taxonomy.

Public source. For BFCL and SearchQA, we collect skills from GitHub and ClawHub whose descriptions are semantically relevant to the benchmark and whose histories contain at least ten version updates. We remove adjacent versions with no substantive skill change. The retained public pool contains 38 skills and 1,266 updates for BFCL, and 21 skills and 1,089 updates for SearchQA.

Trajectory source. We run EvoSkill (Alzubi et al. 2026), SkillClaw (Ma et al. 2026), and SkillOpt (Yang et al. 2026) on BFCL and SearchQA. Each method contributes 400 iterative updates across the two benchmarks, producing 1,200 trajectory-derived updates in total. These updates follow the execution evidence observed by the current agent during skill evolution.

Balanced comparison. From the two source pools, we construct a benchmark-matched corpus containing 400 skillchange units from each source. The shared protocol supports direct comparison of component, intent, and pattern distributions.

## A.2 Skill-Change Units and Annotation Taxonomy

Let $( s ^ { - } , s ^ { + } )$ denote the skill before and after one public or trajectory-derived update, and let $D ( s ^ { - } , s ^ { + } )$ be their filelevel diference. A skill-change unit is the maximal set of edits that implements one semantic intent through one reusable modification pattern. Functionally coupled edits may span several files or components, while edits with distinct intents form separate units. Formatting-only changes, generated files, and dependency-lock churn are removed before segmentation.

Each unit is described by three dimensions. The component records the afected artifact types, the intent records the objective of the update, and the pattern records the reusable modification strategy. Components are multi-label, whereas intent and pattern each receive one primary label.

## A.3 Representative Change Units

Figure 1 contrasts a coordinated public-history update with a trajectory-derived instruction update. The public update changes both user-facing guidance and executable code to maintain service compatibility. The trajectory-derived update adds extraction rules for a failure format observed during SearchQA evolution.

## A.4 Broader Public-History Statistics

Separate from the BFCL/SearchQA matched comparison, we survey 1,904 GitHub skills and 242 ClawHub skills across 12 task domains. The survey contains 36,719 substantive adjacent-version updates from 2,146 public skills. For each update, we identify the afected components and assign its dominant high-level intent.

Figure 2 shows that public evolution routinely changes instructions, scripts, references, and configurations. Instructions are the most frequently modified component in 11 of the 12 domains, with within-domain shares ranging from 58.9% to 82.9% in those domains. Scripts, references, and configurations also occur throughout the domain set, capturing implementation and maintenance knowledge beyond instruction editing.

## A.5 Additional Analyses of Trajectory-Driven Evolution

Trajectory utility. Each trajectory is scored on six dimensions: execution integrity, feedback fidelity, trace observability, root-cause diagnosability, optimization actionability, and cross-task generalizability. Each dimension receives an integer score $r _ { d } \in \{ 0 , 1 , 2 \}$ , giving a total score $\begin{array} { r } { T ( \tau ) = \sum _ { d = 1 } ^ { 6 } r _ { d } \in [ 0 , 1 2 ] } \end{array}$ and normalized utility $U ( \tau ) =$ $T ( \tau ) / 1 2$ . High-utility trajectories satisfy $T ( \tau ) \geq 1 0$ and contain no hard flag, medium-utility trajectories satisfy $7 \leq T ( \tau ) \leq 9$ , and low-utility trajectories satisfy $T ( \tau ) \leq 6$ Any non-infrastructure hard flag assigns the trajectory to the low-utility group. Timeouts, connection errors, API errors, exceptions, empty traces, and missing traces are labeled infra\_invalid before utility classification. For iteration t, trajectory utility is the mean normalized utility of the newly collected trajectories used to produce the update. Update effectiveness is $\Delta \mathrm { V a l } _ { t } = \mathrm { V a l } ( \bar { S _ { t } } ) - \mathrm { V a l } ( S _ { t - 1 } \bar { ) }$ , the validationscore change produced by that iteration.

Figure 3 shows a positive association between trajectory utility and validation improvement. Negative updates are concentrated among low-utility trajectories, while highutility trajectories more often support positive validation changes. The quality of the current rollout therefore directly afects the update signal available to trajectory-driven evolution.

Failure-mode identification. The judge records one primary failure mode, an optional secondary mode, confidence, rationale, and hard flags for each trajectory. The ten canonical OficeQA modes are document retrieval, table localization, row–column alignment, temporal scope, unit/scale/sign, arithmetic aggregation, answer extraction/format, unsupported or hallucinated evidence, tool execution, and other. The normalizer first retains exact canonical labels, then maps markers for answer format, unsupported or hallucinated evidence, missing-document retrieval, table structure or columns, temporal scope, units/scales/signs, arithmetic/- formulas/statistics, and tool execution to the corresponding modes. Unmatched outputs are assigned to the other category.

<table><tr><td>Dimension</td><td>Annotation target</td><td>Decision rule</td></tr><tr><td>Component</td><td>Affected skill artifacts</td><td>Assign all affected artifact types, including instructions, scripts, references, configurations, and templates.</td></tr><tr><td>Intent</td><td>Objective of the update</td><td>Assign the primary outcome pursued by the change.</td></tr><tr><td>Pattern</td><td>Modification strategy</td><td>Abstract the operational mechanism into a repository- independent family.</td></tr><tr><td colspan="3">Selected intent labels and pattern families used in the representative cases.</td></tr><tr><td>Type</td><td>Label</td><td>Definition</td></tr><tr><td>Intent</td><td>INTENT_04: Evidence retrieval qual- ity</td><td>Improve the precision, recall, navigation, or grounding of evi- dence acquisition before reasoning or execution.</td></tr><tr><td>Intent</td><td>INTENT_10: Compatibility and inter- operability</td><td>Align the skill with supported platforms, dependencies, schemas, clients, or execution environments.</td></tr><tr><td>Pattern</td><td>FAMILY_01: Evidence retrieval and navigation</td><td>Modify how evidence is searched, located, traversed, or extracted from information sources.</td></tr><tr><td>Pattern</td><td>FAMILY_08: Executable workflow au- tomation</td><td>Modify executable helpers that automate requests, lifecycle steps, result enrichment, or artifact generation.</td></tr></table>

Table 1: Annotation dimensions and selected taxonomy labels used in the representative skill-change units.

<table><tr><td>Failure-mode coverage</td><td>∆Train ↑</td><td>∆Test ↑</td><td> $\mathrm { G a p } \downarrow$ </td><td>P2F↓</td></tr><tr><td>Low (1–3 modes)</td><td>4.82</td><td>2.54</td><td>2.28</td><td>12.7</td></tr><tr><td>Medium (4–8 modes)</td><td>4.17</td><td>2.81</td><td>1.36</td><td>9.7</td></tr><tr><td>High (9–10 modes)</td><td>4.23</td><td>3.19</td><td>1.04</td><td>5.9</td></tr></table>

Table 2: Training-test gaps and pass-to-fail regressions under diferent failure-mode coverage.

Failure-mode coverage. For a set of trajectories, unique\_modes is the number M of distinct primary failure modes and mode\_coverage is $M / | \mathcal { M } _ { \mathrm { r e f } } |$ , where $\mathcal { M } _ { \mathrm { r e f } }$ is the reference mode set. The normalized entropy of the mode distribution is

$$
H _ { \mathrm { n o r m } } = - \frac { \sum _ { m = 1 } ^ { M } p _ { m } \log p _ { m } } { \log M } ,
$$

with $H _ { \mathrm { n o r m } } = 0$ when $M \leq 1$ . The hq\_diverse condition covers at least four modes and selects at most two trajectories from each mode, while hq\_narrow contains trajectories from only one or two modes. At the run level, we group observed coverage into low (1–3), medium (4–8), and high (9–10) bands. We report the changes in training and test performance, their diference, and pass-to-fail regressions (P2F), defined as tasks that pass before evolution and fail afterward.

Table 2 shows that broader failure-mode coverage improves test gains and reduces regressions. From the lowto high-coverage group, the train–test gap decreases from 2.28 to 1.04 and the mean P2F count decreases from 12.7 to 5.9. These results connect narrow trajectory evidence with overfitting during iterative skill evolution.

The matched study and the supplementary analyses motivate the two information sources used by VCE-Skill. Public histories contribute recurring maintenance and implementation strategies, and current trajectories contribute taskspecific execution evidence. Adaptive selection and fusion combine these signals throughout skill evolution.

![](images/0998f0f835a8113c0e4a2fd7ae31b3804a08dd59b00a05ae085a1adc7b075867.jpg)  
Figure 1: Representative skill-change units from public version histories and trajectory-driven evolution. The public update coordinates instruction and executable-code changes to preserve compatibility with an external service. The trajectory-derived update adds a local extraction rule in response to an observed SearchQA failure.

![](images/84a2e3e49327fddba169e97764b8608b2f7db02e922668d0530c9af0b8c31b14.jpg)  
Figure 2: Skill counts, substantive version updates, and afected-component shares across 12 task domains in the broader public-history survey. Component shares are nonexclusive because one update may afect several artifacts.

![](images/34775f12b15da22399a681e1b4a862ea509ca465121f80305ed646b5e5fbef99.jpg)  
Figure 3: Trajectory utility and validation-score change across skill-evolution iterations.

## B VCE-Skill Implementation Details

## B.1 Ofline Experience-Bank Construction

Algorithms 1 and 2 detail the ofline and online stages described in the main paper. Adjacent versions are used because they preserve the local maintenance context and reduce ambiguity about which edits belong to one update step. The method first abstracts raw difs into events, then progressively removes repository-specific detail at the pattern, skill, and domain levels. The source corpus contains 1,904 GitHub skills and 242 ClawHub skills across 12 task domains, as described in Section A. For each evaluation benchmark, we manually select the five skills most closely aligned with its task and tool-use requirements and distill their adjacentversion changes into a benchmark-specific experience bank.

Grouping and abstraction. Events are grouped when their components, semantic operations, and intents support the same reusable strategy. Duplicate or paraphrased events are merged when they recommend the same operation under the same applicability conditions, while events with diferent conditions or intended outcomes remain separate. An update pattern states transferable operational guidance and cites its supporting event identifiers. Skill-level insights synthesize compatible patterns from one skill, and domain-level insights require support across multiple skills or distinct concerns in the same domain. Conflicting observations yield conditiona guidance when their applicability conditions explain the difference; otherwise, they are omitted. Each abstraction preserves backward links to its supporting events for verification against concrete difs.

<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $u _ { j }$ </td><td>update event  $\langle d _ { j } , \{ c _ { j } \} , o _ { j } , i _ { j } \rangle$  containing a raw diff segment, affected components, semantic op-</td></tr><tr><td> $\boldsymbol { B }$ </td><td>eration, and intent version-change experience bank containing up- date patterns, evolution insights, and domain in-</td></tr><tr><td> $S _ { t }$ </td><td>sights target skill retained by the base evolver after on- line iteration t</td></tr><tr><td> $\tau _ { t }$ </td><td>task-execution trajectory produced with  $S _ { t }$ </td></tr><tr><td> $z _ { t }$ </td><td>selection context containing target task, current skill, and available prior feedback</td></tr><tr><td> $\mathcal { X } _ { t }$ </td><td>at most five experience entries selected from B</td></tr><tr><td> $p _ { t } ^ { \mathrm { e x p } }$ </td><td>aggregated external-experience guidance</td></tr><tr><td> $\bar { p } _ { t } ^ { \mathrm { s e l f } }$ </td><td>trajectory-grounded proposal from the base evolver</td></tr><tr><td> $p _ { t } ^ { \mathrm { e n h } }$ </td><td>provenance-preserving fusion of the two pro- posal sources</td></tr><tr><td> $f _ { t } ^ { S } , f _ { t } ^ { V } , f _ { t }$   $\lambda _ { \mathrm { e x p } } ^ { ( t ) }$ </td><td>source, validation, and unified feedback prompt-level reliance weight assigned to external experience;  $\lambda _ { \mathrm { s e l f } } ^ { ( t ) } = 1 - \bar { \lambda } _ { \mathrm { e x p } } ^ { ( t ) }$ </td></tr></table>

Table 3: Core notation used in the implementation.

Deduplication and quality control. The distillation prompts enforce evidence-linked identifiers at every level and merge entries with equivalent guidance and applicability. They remove repository names, exact paths, version numbers, benchmark answers, and one-of implementation details from reusable guidance. Entries are retained only when they are evidence-grounded, actionable, transferable, and no broader than their supporting changes.

## Distillation Prompt Definitions

Update-event extraction. The extractor receives one dif chunk between adjacent skill versions and returns zero or more atomic semantic events. Each event separates the exact dif evidence, afected components, semantic operation, and supported update intent.

Update-pattern distillation. The input contains one skill name and its structured update events.

Skill-level insight synthesis. The input contains the update patterns distilled from one skill history.

Domain-level insight generalization. The input contains skill-level insights from multiple skills in one task domain.

## B.2 Online Optimization Algorithm

Algorithm 2 gives the complete online recurrence. VCE-Skill changes how the update proposal is constructed and leaves candidate evaluation, retention, checkpointing, and stopping to the native mechanism of the base evolver.

## B.3 Experience Selection and Aggregation

The selector receives the benchmark, current skill, compact entries from the corresponding benchmark-specific bank, and the preceding unified feedback when one is available. It evaluates each entry by task relevance, correspondence to an observable gap in the current skill, next-step actionability, applicability, redundancy, and consistency with the preceding feedback. The selection budget is $\dot { K } = 5 ;$ the output contains zero to five unique bank identifiers together with compact external guidance linked to the selected entries. The selector prefers a small complementary set over redundant restatements and returns an empty selection when no supplied entry is suficiently relevant and actionable. When the selection is empty, fusion is skipped and the base evolver’s self proposal proceeds unchanged.

Algorithm 1: Ofline version-change experience distillation Algorithm 2: Online optimization with adaptive experience   
attention   
Require: Public skills with chronologically ordered versions   
Ensure: Experience bank $\boldsymbol { B }$ Require: Initial skill $S _ { 0 } .$ , target task $\tau ,$ experience bank $B ,$   
1: $B  \emptyset$ and base evolver $\mathcal { E }$   
2: for each retained skill $r$ do Ensure: Skill returned by the native base-evolver loop   
3: order versions $( v _ { 1 } , \ldots , v _ { m _ { r } } )$ chronologically 1: $\lambda _ { \mathrm { e x p } } ^ { ( 1 ) }  0 . 5 , \lambda _ { \mathrm { s e l f } } ^ { ( 1 ) }  \mathrm { \bar { 0 . 5 } }$   
4: ${ { \mathcal { U } } _ { r } } \gets \emptyset$ 2: execute $S _ { 0 }$ to obtain $\tau _ { 0 }$   
5: for $j = 1 \mathrm { t o } m _ { r } - 1$ do 3: for $t = 1$ to the iteration budget do   
6: $D _ { j } \gets \operatorname* { D i f f } ( v _ { j } , v _ { j + 1 } )$ 4: if $t = 1$ then   
7: remove ineligible mechanical changes from $D _ { j }$ 5: $z _ { t } \gets \mathrm { C o n t e x t } ( \mathcal { T } , S _ { t - 1 } )$   
8: segment $D _ { j }$ into semantic change units 6: else   
9: for each change unit g do 7: $z _ { t } \gets \mathrm { C o n t e x t } ( \mathcal T , S _ { t - 1 } , f _ { t - 1 } )$   
10: extract $\boldsymbol { u } = \langle d , \{ c \} , o , i \rangle$ 8: end if   
11: validate schema, labels, and evidence traceability 9: $\mathcal { X } _ { t } \gets \mathrm { S e l e c t } _ { \leq 5 } ( B , z _ { t } )$   
12: ${ \mathcal { U } } _ { r } \gets { \mathcal { U } } _ { r } \cup \{ u \}$ 10: $p _ { t \_ s } ^ { \mathrm { e x p } }  \mathrm { A g g r e g a t e } ( \dot { \mathcal X } _ { t } )$   
13: end for 11: $p _ { t } ^ { \mathrm { { s e l f } } }  \mathcal { E } . \mathrm { { P r o p o s e } } ( S _ { t - 1 } , \tau _ { t - 1 } )$   
14: end for 12: attach unique $\mathrm { E X P - k }$ and SELF-k identifiers   
15: group compatible events in $\mathcal { U } _ { r }$ and distill update pat- 13: $p _ { t } ^ { \mathrm { e n h } }  \mathrm { F u s e } ( p _ { t } ^ { \mathrm { e x p } } , p _ { t } ^ { \mathrm { s e l f } } ; \lambda _ { \mathrm { e x p } } ^ { ( t ) } , \lambda _ { \mathrm { s e l f } } ^ { ( t ) } )$   
terns   
16: synthesize the patterns into skill-level evolution insights 14: $\widehat { S } _ { t } \gets \mathcal { E } . \mathrm { A p p l y } ( S _ { t - 1 } , p _ { t } ^ { \mathrm { e n h } } )$   
17: add validated patterns and insights to B 15: $( S _ { t } , o _ { t } )  \mathcal { E } . \mathrm { R e t a i n } ( S _ { t - 1 } , \widehat { S } _ { t } )$   
18: end for 16: align the tagged edits with Dif $( S _ { t - 1 } , \widehat { S } _ { t } )$   
19: for each task domain represented in B do 17: compute $f _ { t } ^ { \lessgtr } .$ , obtain $f _ { t } ^ { V }$ from $o _ { t } ,$ , and se $\mathbf { \bar { \mathbf { \mathit { f } } } } _ { t } = \mathbf { \mathit { f } } _ { t } ^ { V } \mathbf { \mathbf { \mathit { f } } } _ { t } ^ { S }$   
20: generalize recurring evolution insights into domain in- 18: execute retained $S _ { t }$ to obtain $\tau _ { t }$   
sights 19: $\lambda _ { \mathrm { e x p } } ^ { ( t + 1 ) } \gets \mathrm { c l i p } ( \lambda _ { \mathrm { e x p } } ^ { ( t ) } + 0 . 1 f _ { t } , 0 . 3 , 0 . 7 )$   
21: 22: end for add validated domain insights to B 20: $\lambda _ { \mathrm { s e l f } } ^ { ( t + 1 ) }  1 - \lambda _ { \mathrm { e x p } } ^ { ( t + 1 ) }$   
21: end for   
23: deduplicate entries without removing distinct supporting   
22: return the skill selected by $\mathcal { E }$   
evidence   
24: return B

Role of prior feedback. Task and skill compatibility remain the primary selection criteria. Positive $f _ { t - 1 }$ makes selection more receptive to external guidance, whereas negative feedback requires stronger applicability evidence. The first iteration contains no prior feedback, so selection uses only the benchmark and current skill. The source weights specify relative reliance on external and self-generated proposals.

## B.4 Attention Fusion and Skill Optimization

The source weights are prompt-level reliance instructions. At initialization, $\lambda _ { \mathrm { e x p } } ^ { ( 1 ) } = \lambda _ { \mathrm { s e l f } } ^ { ( 1 ) } = 0 . 5$ . They serve as relative preferences for incompatible recommendations, rather than output quotas or sampling probabilities. Direct trajectory evidence remains authoritative: the fuser preserves useful self edits and uses external guidance to refine, constrain, complement, or replace them when the guidance identifies a concrete conflict or stronger applicable strategy. At equal weights, the same evidence rule resolves conflicts. Duplicate, contradictory, unsupported, non-actionable, and out-of-scope edits are removed.

Every external item receives an identifier EXP-k, and every self-proposal item receives an identifier SELF-k. A fused edit contains all source identifiers that materially support it, and mixed provenance is used only when one edit genuinely combines both sources. The fuser emits replace, append, prepend, or delete operations over the current skill and returns an empty edit list when neither source supports an applicable change. The shared rewriter applies the selected operations to the complete skill document, preserves efective existing guidance and protected blocks, and consolidates overlapping instructions.

Full-skill rewrite. After fusion, the shared skill optimizer receives the current skill and selected revision suggestions and returns the complete rewritten skill document.

## B.5 Provenance Attribution and Adaptation Feedback

Let $\mathcal { C } _ { t } = \{ c _ { t , k } \} _ { k = 1 } ^ { n _ { t } }$ be the provenance-tagged candidate edits. The attribution stage aligns each item with the optimizerproduced candidate dif $\Delta S _ { t } = \mathrm { D i f f } ( \widehat { S } _ { t } , S _ { t - 1 } )$ . It assigns +1 to a realized edit supported by external experience, −1 to a realized edit supported by the self proposal, and 0 when an edit is unrealized, cannot be aligned reliably, or has mixed provenance. We compute the source feedback as follows.

![](images/75eb009b3f95efce38f66e01f1381c1942a28bed78cd6981457a91efdf81e595.jpg)  
Figure 4: System prompt for update-event extraction.

$$
f _ { t } ^ { S } = \frac { 1 } { n _ { t } } \sum _ { k = 1 } ^ { n _ { t } } a _ { t , k } .
$$

The denominator includes all candidate edits, so a proposal that is largely ignored has a source-feedback magnitude near zero.

The native base-evolver outcome supplies binary validation feedback:

$$
f _ { t } ^ { V } = \left\{ \begin{array} { l l } { + 1 , } & { \Delta _ { t } ^ { V } > 0 , } \\ { - 1 , } & { \Delta _ { t } ^ { V } \leq 0 , } \end{array} \right. \qquad f _ { t } = f _ { t } ^ { V } f _ { t } ^ { S } .
$$

Consequently, an improving update increases reliance on the source that contributed more realized edits, while a nonimproving update shifts reliance away from that source. We apply the following attention update.

$$
\lambda _ { \mathrm { e x p } } ^ { ( t + 1 ) } = \mathrm { c l i p } \Big ( \lambda _ { \mathrm { e x p } } ^ { ( t ) } + 0 . 1 f _ { t } , 0 . 3 , 0 . 7 \Big )
$$

keeps both evidence sources active.

Boundary cases. When $n _ { t } \ = \ 0 ,$ source feedback is f<sup>S</sup><sub>t</sub> = 0 and the attention weights remain unchanged. Mixedprovenance edits receive zero source attribution. Partial implementations, ambiguous matches, contradicted changes, no-op changes, wording-only overlap, and unapplied edits are marked unrealized. Attribution is computed on the optimizerproduced candidate dif; the base evolver’s subsequent rejection is represented by negative validation feedback.

## B.6 Base-Evolver Interfaces

VCE-Skill inserts a thin proposal adapter between each base evolver’s native proposal stage and its native skill-update stage. The adapter exposes the current skill and self proposal to the common selector–fuser interface, attaches SELF-k identifiers, and converts the fused result back to the base evolver’s expected format. The rollout collection, diagnosis, candidate evaluation, retention, stopping, and checkpoint logic of the base evolver remain unchanged.

Adapter mappings. For EvoSkill, the adapter wraps the full-skill revision derived from failure trajectories as one replace or append self edit, fuses it with external guidance, materializes a complete SKILL.md, and returns it to the native frontier and retention loop. For SkillClaw, it renders the current and proposed skills, wraps their change as a self edit, materializes the fused proposal, and returns it to the native validation and publication cycle. For SkillOpt, it attaches provenance to the native patch distilled from scored rollouts, fuses it with external edits in the same patch schema, and passes it to the existing ranking, patch-application, and retention stages. EvoSkill and SkillClaw therefore use complete skill documents at the adapter boundary, whereas SkillOpt operates directly on bounded patch edits. Adapting a new base evolver requires proposal serialization, provenance attachment, and result materialization.

## B.7 End-to-End Evolution Case Study

Figure 12 follows one evolution trace from a public version change to feedback-driven attention adaptation. The source update converts recurring incomplete-output incidents into explicit activation cues, completeness checks, and recovery rules. Its update event, pattern, and insight provide reusable guidance for skills that produce structured outputs.

```jsonl
You distill supplied update events from one skill's version history into reusable update patterns.
An update pattern is transferable guidance about how and why to evolve a skill. Group compatible events by
component, operation, and intent, while preserving meaningful applicability boundaries.
Distillation rules:
Every pattern must cite one or more supplied event IDs in `supporting_event_ids`.
Cite only supplied event IDs. Never invent, rewrite, or omit the grounding IDs for a non-empty pattern.
Merge duplicate or paraphrased patterns when they recommend the same operation under the same conditions.
Keep separate patterns when their applicability or intended outcome differs.
Remove repository names, exact paths, version numbers, benchmark-specific answers, and one-off implementation
details.
State constructive, operational guidance that can transfer to another skill.
Reject observations that describe a change but do not provide a reusable evolution decision.
Return an empty list when no supplied event supports a transferable pattern.
Before returning, verify that patterns are grounded, non-duplicate, transferable, actionable, and no broader
than their supporting evidence.
Return JSON only:
"patterns": [
{
"content": "reusable update pattern",
"applicability": "conditions under which this pattern is useful",
"supporting_event_ids": ["EVENT-id"]
}
]
}
```  
Figure 5: System prompt for update-pattern distillation.

You synthesize supplied update patterns from one skill into evolution insights.   
An evolution insight is an actionable strategy for improving similar skills, not a summary of one repository   
change.   
Synthesis rules:   
Prefer insights supported by multiple compatible patterns.   
A single pattern may support an insight only when it clearly expresses a general evolution strategy rather   
than an incidental fix.   
Every insight must cite only supplied pattern IDs in \`supporting\_pattern\_ids\`.   
Preserve applicability boundaries. Do not generalize beyond the situations represented by the supporting   
patterns.   
Merge overlapping insights when one actionable formulation covers both.   
Keep insights separate when they address different failure modes or applicability conditions.   
Exclude benchmark answers, repository details, unsupported causal claims, and purely descriptive observations.   
Return an empty list when no reliable skill-level insight is supported.   
Before returning, verify grounding, applicability, actionability, non-duplication, and that incidental edits   
were not promoted into principles.   
Return JSON only:   
{   
"insights": [   
{   
"content": "skill-level evolution insight",   
"applicability": "when another skill should use this insight",   
"supporting\_pattern\_ids": ["PATTERN-id"]   
}   
]  
Figure 6: System prompt for skill-level evolution-insight synthesis.

The target trajectory completes its analysis but stops before emitting the required function call. The selector retrieves three complementary experiences, while the base evolver proposes four edits covering JSON closure, tool matching, output formatting, and pre-output validation. The fuser preserves both sources in four mixed-provenance edits, and all four edits appear in the accepted Skill update.

Across 507 paired validation examples, the hard score increases from 0.5661 to 0.5740, with 17 wrong-to-right and 13 right-to-wrong transitions. The representative parallelcall example improves from zero matched calls out of two to two matched calls out of two. All realized edits have mixed provenance, giving $f _ { t } ^ { S } = 0 , f _ { t } ^ { V } = + 1$ , and $f _ { t } = 0 ;$ the next-round weights therefore remain $\lambda _ { \mathrm { e x p } } = \lambda _ { \mathrm { s e l f } } = 0 . 5$

You generalize supplied evolution insights from multiple skills in the same task domain.   
A domain insight is an operational optimization principle that transfers across skills or across clearly   
different skill concerns in the domain.   
Generalization rules:   
Every insight must cite only supplied skill-insight IDs in \`supporting\_skill\_insight\_ids\`.   
Require evidence across skills or clearly distinct concerns. Do not rename a single narrow skill insight as a   
domain principle.   
Preserve applicability conditions and operational actions.   
Merge duplicate principles with equivalent actions and applicability.   
When supplied insights conflict, keep conditional guidance only if the differing applicability conditions   
explain the conflict; otherwise omit the unsupported generalization.   
Exclude repository details, dataset instances, benchmark-specific answer knowledge, and unsupported claims.   
Return an empty list when cross-skill evidence is insufficient.   
Before returning, verify cross-skill support, transferability, conflict handling, non-duplication, and valid   
supplied IDs.   
Return JSON only:   
{   
"insights": [   
{   
"content": "domain-level evolution insight",   
"applicability": "when a target skill should use this insight",   
"supporting\_skill\_insight\_ids": ["SKILL-id"]   
}   
]   
}

Figure 7: System prompt for domain-level evolution-insight generalization.  
![](images/663dddab52b1f9e9da650f14aca427045fe3f57730d3c21f92bbec02affac915.jpg)  
Figure 8: System prompt for feedback-conditioned experience selection and aggregation.

Figure 9: System prompt for adaptive-attention fusion.  
You fuse an independently generated self proposal with selected external evolution guidance into candidate   
skill edits.   
The attention weights are relative conflict preferences, not output quotas and not probabilities. They guide   
which source to prefer when both sources make incompatible recommendations. Do not manufacture edits to   
match a numerical ratio.   
Fusion rules:   
Direct trajectory evidence supporting the self proposal remains authoritative. External guidance may refine,   
constrain, complement, or replace a self edit only when semantically justified.   
Preserve useful self edits even when \`lambda\_exp\` is larger, unless relevant external guidance identifies a   
concrete conflict or stronger transferable strategy.   
Use external-only provenance for an edit derived only from external guidance, self-only provenance for an   
unchanged self edit, and mixed provenance only when an edit genuinely combines both sources.   
Copy provenance IDs exactly from the supplied inputs.   
Remove duplicate, contradictory, unsupported, non-actionable, and out-of-scope edits.   
Every output edit must be a valid patch operation using \`replace\`, \`append\`, \`prepend\`, or \`delete\`, with the   
required target and content fields.   
Return an empty edit list rather than inventing an unsupported change.   
Before returning, verify valid patch operations, exact provenance, conflict handling, non-duplication, and that   
the attention weights were used only as relative conflict preferences.   
Return JSON only:   
"reasoning": "brief fusion rationale",   
"edits": [   
{   
"op": "replace|append|prepend|delete",   
"target": "exact target when required",   
"content": "edit content when required",   
"provenance\_ids": ["SELF-1", "EXP-1"]   
}   
]   
}

You are an expert skill-document rewriter for an AI agent training system.   
You will receive:   
1. The current skill document   
2. A selected set of revise\_suggestions distilled from trajectory analysis   
Your job is to rewrite the FULL target skill document so it incorporates the selected suggestions coherently.   
Hard requirements:   
1. Produce a complete standalone skill document, not a patch.   
2. Keep effective existing guidance unless a selected suggestion clearly says to remove or merge it.   
3. Prefer consolidation and clarity over making the document longer.   
4. Do not hardcode benchmark-specific answers, entity names, file paths, or gold values.   
5. Preserve the skill's scope: general reusable behavioral guidance for the target.   
6. Do not modify content inside the protected slow-update block between <!-- SLOW\_UPDATE\_START --> and <!--   
SLOW\_UPDATE\_END --> except to keep it intact.   
7. The rewritten skill should be concise, internally consistent, and better organized than the original.   
Respond ONLY with a valid JSON object:   
"reasoning": "<why this rewrite implements the selected suggestions well>",   
"change\_summary": ["<short change 1>", "<short change 2>"],   
"new\_skill": "<the full rewritten skill document>"  
Figure 10: System prompt for full-skill rewriting.

You semantically match provenance-tagged ranked candidate edits to the actual skill diff produced by the   
optimizer.   
Attribution rules:   
Evaluate each supplied \`candidate\_id\` independently.   
Set \`realized=true\` only when the actual skill diff implements the candidate's core semantic change.   
Exact wording is not required when behavior and intent are equivalent.   
Partial implementations, ambiguous similarity, contradicted changes, no-op changes, wording-only overlap, and   
unselected or unapplied changes are \`realized=false\`.   
Evidence must briefly identify the relevant actual skill diff content.   
If the actual diff provides insufficient evidence, use \`realized=false\`.   
Do not decide whether external or self experience was beneficial and do not assign source scores.   
Deterministic code calculates source feedback from provenance IDs.   
Return one match record for each supplied candidate ID and do not invent IDs.   
Before returning, verify candidate coverage, conservative semantic matching, actual-diff evidence, valid   
supplied IDs, and boolean \`realized\` values.   
Return JSON only:   
{   
"matches": [   
{   
"candidate\_id": "CAND-1",   
"realized": true,   
"evidence": "brief reference to the matching actual change"   
}   
]   
}  
Figure 11: System prompt for semantic realization matching.

![](images/8bfa04d84da7a35ca7199c9b1aac06bbfaa0895648eb4d72168b21984169ac0e.jpg)

Figure 12: End-to-end evolution case. A public version change addressing incomplete structured outputs is distilled into reusable experience. The selected experience and self proposal are fused into four mixed-provenance edits, all of which are realized in the accepted Skill update. The hard score increases from 0.5661 to 0.5740, while mixed provenance yields zero source feedback and retains balanced experience attention.

## C Additional Experimental Results and Analyses

Unless otherwise specified, all supplementary experiments follow the main-paper protocol over SearchQA, OficeQA, ALFWorld, Spreadsheet, and BFCL-v4 with Qwen3.5-27B, GPT-5.2, DeepSeek-v3.2, and Claude Sonnet 5. Each configuration is independently run three times, and paired comparisons use the same benchmark instances, initial skill, evolution budget, and target model. The compared methods are No Skill, LLM Skill, EvoSkill, SkillClaw, SkillOpt, and the VCE-enhanced variants of the three iterative self-evolvers.

We use GPT-5.5 for the LLM-based annotation and analysis in the motivation study. For experience distillation and fusion in VCE-Skill, we use the corresponding target model in each experimental configuration. All large language models are accessed through their APIs. All experiments are conducted on a server equipped with an Intel Core i7-10700 CPU, an NVIDIA TITAN RTX GPU, and 32 GB of RAM.

## C.1 Ablation Results Across Base Self-Evolvers

Experimental setup. We evaluate experience distillation and adaptive experience attention with EvoSkill, SkillClaw, and SkillOpt as the base self-evolvers. For each base evolver, w/o Dist replaces distilled experience with raw versionchange records, while w/o Atte replaces adaptive attention with fixed equal weights for external experience and selfgenerated proposals. Figure 13 reports the average performance across all five benchmarks for four agent models.

Results. The complete VCE-Skill achieves the highest average score for every base evolver and agent model, covering all twelve base-evolver–model combinations. Removing experience distillation produces the larger degradation in most settings, indicating that distilled update patterns and evolution insights transfer more efectively than raw versionchange records. Removing adaptive experience attention also consistently reduces performance relative to the complete method, although fixed equal weighting usually retains part of the improvement over the corresponding base evolver. These trends remain consistent across EvoSkill, SkillClaw, and SkillOpt and across Qwen3.5-27B, GPT-5.2, DeepSeekv3.2, and Claude Sonnet 5.

Conclusion. Experience distillation and adaptive experience attention provide complementary improvements to VCE-Skill. Distillation converts concrete version changes into reusable evolution knowledge, while adaptive attention controls the contribution of external experience during self-evolution. Their consistent gains across three base selfevolvers demonstrate that both components generalize across diferent self-evolution frameworks.

## C.2 Hyperparameter and Attention Sensitivity

Experimental setup. We evaluate the sensitivity of VCE-Skill to the experience-selection budget K, attention step size η, and attention clipping bounds. All experiments use SkillOpt as the base self-evolver and Qwen3.5-27B as the agent model. We vary one hyperparameter at a time while fixing the remaining settings to $K = 5 , \eta = 0 . 1$ , initial

![](images/52e37e78e051829ff9fd566cde7ba364acc88d16eceb710a5a1346244fbd884e.jpg)  
Figure 13: Ablation results averaged over all five benchmarks. Each panel uses EvoSkill, SkillClaw, or SkillOpt as the base self-evolver. w/o Dist replaces distilled experience with raw version changes, and w/o Atte assigns equal weights to external experience and self-generated proposals. Higher is better.

source weights of 0.5/0.5, and clipping bounds of [0.3, 0.7].   
Table 5 reports the mean performance on each benchmark.

Results. The default configuration achieves the highest score on every benchmark for all three hyperparameters. For the selection budget, K = 4 performs within 0.02–0.11 points of K = 5, whereas increasing the budget to K = 7 reduces performance by 1.11–1.88 points. The attention step size has the strongest efect: $\eta = 0 .$ 1 consistently performs best, while η = 0.2 decreases performance by 2.28–3.88 points across the five benchmarks. The clipping bounds are comparatively stable around the default configuration. The intervals [0.4, 0.6] and [0.2, 0.8] remain within 0.29 points of the best result, whereas the wider interval [0.1, 0.9] produces a consistent reduction of 0.71–1.37 points.

Conclusion. The results support $K = 5 , \eta = 0 . 1$ , and [0.3, 0.7] as the default configuration across all five benchmarks. A moderate selection budget provides suficient experience coverage without introducing excessive weakly relevant candidates. The attention step size requires tighter calibration because it directly controls the rate of sourceweight adaptation. Moderately constrained clipping bounds maintain stable attention updates, whereas overly permissive bounds reduce performance. Importantly, every evaluated configuration outperforms the corresponding SkillOpt baseline on all five benchmarks. The hyperparameters affect the magnitude of improvement, and VCE-Skill retains a consistent advantage over SkillOpt throughout the evaluated ranges.

<table><tr><td>Source Model</td><td>Target Model</td><td>Source Skill</td><td>SearchQA</td><td>OfficeQA</td><td>ALFWorld</td><td>Spreadsheet</td><td>BFCL-v4</td><td>Avg.</td><td>Avg. Gain</td></tr><tr><td rowspan="6">Qwen3.5-27B</td><td rowspan="3">GPT-5.2</td><td>SkillOpt</td><td>78.01</td><td>47.47</td><td>79.16</td><td>52.11</td><td>64.10</td><td>64.17</td><td></td></tr><tr><td>SkillOpt+VCE-Skill</td><td>83.24</td><td>55.34</td><td>86.05</td><td>57.89</td><td>68.54</td><td>70.21</td><td>6.04</td></tr><tr><td>SkillOpt</td><td>71.16</td><td>44.94</td><td>69.60</td><td>38.16</td><td>62.10</td><td>57.19</td><td></td></tr><tr><td rowspan="2">DeepSeek-v3.2 Claude Sonnet 5</td><td>SkillOpt+VCE-Skill</td><td>78.12</td><td>49.53</td><td>75.11</td><td>43.94</td><td>65.48</td><td>62.44</td><td>5.25</td></tr><tr><td>SkillOpt</td><td>81.10</td><td>53.03</td><td>83.69</td><td>56.01</td><td>66.42</td><td>68.05</td><td></td></tr><tr><td rowspan="2">Qwen3.5-27B</td><td>SkillOpt+VCE-Skill</td><td>87.37</td><td>58.14</td><td>86.80</td><td>61.73</td><td>71.97</td><td>73.20</td><td>5.15</td></tr><tr><td>SkillOpt</td><td>75.81</td><td>47.66</td><td>61.24</td><td>50.62</td><td>56.36</td><td>58.34</td><td>1</td></tr><tr><td rowspan="6">GPT-5.2</td><td rowspan="3">DeepSeek-v3.2</td><td>SkillOpt+VCE-Skill</td><td>82.04</td><td>53.59</td><td>66.49</td><td>56.46</td><td>60.89</td><td>63.89</td><td>5.56</td></tr><tr><td>SkillOpt</td><td>72.70</td><td>46.48</td><td>71.14</td><td>39.70</td><td>63.64</td><td>58.73</td><td></td></tr><tr><td>SkillOpt+VCE-Skill</td><td>79.74</td><td>51.15</td><td>76.73</td><td>45.56</td><td>67.10</td><td>64.06</td><td>5.32</td></tr><tr><td rowspan="2">Claude Sonnet 5</td><td>SkillOpt</td><td>82.64</td><td>54.57</td><td>85.23</td><td>57.55</td><td>67.96</td><td>69.59</td><td></td></tr><tr><td>SkillOpt+VCE-Skill</td><td>88.99</td><td>59.76</td><td>88.42</td><td>63.35</td><td>73.59</td><td>74.82</td><td>5.23</td></tr><tr><td rowspan="4">DeepSeek-v3.2</td><td rowspan="2">Qwen3.5-27B</td><td>SkillOpt</td><td>74.35</td><td>46.20</td><td>59.78</td><td>49.16</td><td>54.90</td><td>56.88</td><td></td></tr><tr><td>SkillOpt+VCE-Skill</td><td>80.65</td><td>52.21</td><td>65.10</td><td>55.07</td><td>59.51</td><td>62.51</td><td>5.63</td></tr><tr><td rowspan="2">GPT-5.2</td><td>SkillOpt</td><td>78.09</td><td>47.55</td><td>79.24</td><td>52.19</td><td>64.18</td><td>64.25</td><td></td></tr><tr><td>SkillOpt+VCE-Skill</td><td>83.47</td><td>55.57</td><td>86.28</td><td>58.12</td><td>68.77</td><td>70.44</td><td>6.19</td></tr><tr><td rowspan="3"></td><td rowspan="2">Claude Sonnet 5</td><td>SkillOpt</td><td>81.18</td><td>53.11</td><td>83.77</td><td>56.09</td><td>66.50</td><td>68.13</td><td></td></tr><tr><td>SkillOpt+VCE-Skill</td><td>87.60</td><td>58.37</td><td>87.03</td><td>61.96</td><td>72.20</td><td>73.43</td><td>5.30</td></tr><tr><td>Qwen3.5-27B</td><td>SkillOpt</td><td>76.07</td><td>47.92</td><td>61.50</td><td>50.88</td><td>56.62</td><td>58.60</td><td>一</td></tr><tr><td rowspan="5">Claude Sonnet 5</td><td rowspan="2">GPT-5.2</td><td>SkillOpt+VCE-Skill</td><td>82.32</td><td>53.88</td><td>66.77</td><td>56.74</td><td>61.18</td><td>64.18</td><td>5.58</td></tr><tr><td>SkillOpt</td><td>79.81</td><td>49.27</td><td>80.96</td><td>53.91</td><td>65.90</td><td>65.97</td><td></td></tr><tr><td rowspan="2">DeepSeek-v3.2</td><td>SkillOpt+VCE-Skill</td><td>85.14</td><td>57.24</td><td>87.95</td><td>59.79</td><td>70.44</td><td>72.11</td><td>6.14</td></tr><tr><td>SkillOpt</td><td>72.96</td><td>46.74</td><td>71.40</td><td>39.96</td><td>63.90</td><td>58.99</td><td></td></tr><tr><td></td><td>SkillOpt+VCE-Skill</td><td>80.02</td><td>51.43</td><td>77.01</td><td>45.84</td><td>67.38</td><td>64.34</td><td>5.34</td></tr></table>

Table 4: Complete cross-model transfer results over all 12 directed source–target model pairs. Each source-trained skill is directly applied to the target model without further evolution. Avg. reports the mean performance across the five benchmarks, and Avg. Gain reports the mean improvement of SkillOpt+VCE-Skill over SkillOpt computed from the unrounded results. Bold indicates the higher result within each source–target pair.

## C.3 Complete Cross-Model Transfer Results

Experimental setup. We evaluate cross-model transfer over all ordered pairs of Qwen3.5-27B, GPT-5.2, DeepSeekv3.2, and Claude Sonnet 5. For each source model, SkillOpt and SkillOpt+VCE-Skill independently evolve a skill, which is then directly applied to each of the other three target models without further evolution. This design yields 12 directed source–target pairs evaluated on five benchmarks. The average gain is computed as the mean diference between SkillOpt+VCE-Skill and SkillOpt across the five benchmarks.

Results. SkillOpt+VCE-Skill outperforms SkillOpt on every benchmark for all 12 directed source–target pairs, yielding positive improvements in all 60 benchmark-level comparisons. The average gains range from 5.15 to 6.19 points, with an overall mean improvement of5.56 points. Using Qwen3.5- 27B, GPT-5.2, DeepSeek-v3.2, and Claude Sonnet 5 as the source yields average gains of 5.48, 5.37, 5.71, and 5.69 points, respectively. The small variation across source models shows that the transfer improvement is preserved across diferent source-model capabilities. Among the target models, GPT-5.2 receives the largest mean gain of 6.12 points, and every target model improves by at least 5.23 points on average.

Conclusion. The complete transfer results demonstrate that version-change experience improves the cross-model portability of evolved skills. The improvement holds for every source model, target model, and benchmark evaluated in this study. Because the transferred skills receive no target-side evolution, the consistent gains reflect the stronger transferable guidance produced by VCE-Skill.

<table><tr><td>Parameter Value</td><td></td><td>SearchQA OfficeQA ALFWorld Spreadsheet BFCL-v4</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="5">K</td><td>3</td><td>76.24</td><td>47.54</td><td>61.30</td><td>51.39</td><td>55.98</td></tr><tr><td>4</td><td>78.75</td><td>49.34</td><td>63.38</td><td>52.12</td><td>57.42</td></tr><tr><td>5</td><td>78.86</td><td>49.42</td><td>63.43</td><td>52.14</td><td>57.52</td></tr><tr><td>6</td><td>78.03</td><td>48.82</td><td>62.76</td><td>51.59</td><td>57.03</td></tr><tr><td>7</td><td>76.98</td><td>48.07</td><td>61.90</td><td>50.89</td><td>56.41</td></tr><tr><td rowspan="5">η</td><td>0.01</td><td>75.16</td><td>46.96</td><td>61.42</td><td>50.67</td><td>56.34</td></tr><tr><td>0.05</td><td>76.91</td><td>48.02</td><td>61.85</td><td>50.84</td><td>56.37</td></tr><tr><td>0.10</td><td>78.86</td><td>49.42</td><td>63.43</td><td>52.14</td><td>57.52</td></tr><tr><td>0.15</td><td>76.16</td><td>47.48</td><td>61.24</td><td>51.34</td><td>56.93</td></tr><tr><td>0.20</td><td>74.98</td><td>46.83</td><td>60.28</td><td>49.55</td><td>55.24</td></tr><tr><td rowspan="5">Clipping</td><td>[0.4, 0.6]</td><td>78.61</td><td>49.14</td><td>63.23</td><td>51.97</td><td>57.37</td></tr><tr><td>[0.3,0.7]</td><td>78.86</td><td>49.42</td><td>63.43</td><td>52.14</td><td>57.52</td></tr><tr><td>[0.2,0.8]</td><td>78.57</td><td>49.21</td><td>63.41</td><td>52.05</td><td>57.48</td></tr><tr><td>[0.1,0.9]</td><td>77.49</td><td>48.54</td><td>62.62</td><td>51.43</td><td>56.71</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 5: Hyperparameter sensitivity of SkillOpt+VCE-Skill with Qwen3.5-27B. Each row reports mean performance on five benchmarks while varying one hyperparameter and fixing the remaining settings to their defaults. Bold indicates the best score within each hyperparameter group.

## D Human Audit of LLM-based Decisions

Audit Scope and Sampling. We conduct a human audit of six types of LLM-based semantic decisions: Motivationstudy Taxonomy, Skill-Change Coding, Change Abstraction, Experience Generation, Experience Selection, and Source Attribution.

For Motivation-study Taxonomy, we sample 200 category–example instances from the frozen taxonomy and its calibration records. The samples are balanced across the two task domains and stratified across the component, intent, and pattern dimensions. Each instance contains one category definition and one supporting calibration example, allowing annotators to assess both the semantic validity of the category and whether the example supports it.

For Skill-Change Coding, we sample 200 annotated skillchange units. The sample is balanced across the two task domains and two evolution sources, with 50 instances from each domain–source combination. Each instance contains the paired skill states, the complete proposed segmentation as context, and one highlighted skill-change unit with its assigned component, intent, and pattern labels. The audit target is the highlighted unit rather than every other unit in the complete segmentation. Whenever possible, sampled units are drawn from diferent public version pairs or trajectoryevolution iterations to reduce sample dependence. In addition, every category reported as public-only or trajectoryonly is represented by at least one audited unit.

For each of the remaining four decision types, we sample 200 instances from the complete experimental records. The samples are stratified across tasks, target skills, optimization iterations, and base evolvers to avoid over-representing a particular experimental setting. For Experience Generation, the 200 samples consist of 67 update patterns, 67 evolution insights, and 66 domain insights. Whenever possible, instances derived from diferent version pairs or optimization iterations are selected to reduce sample dependence.

Annotation Process. Two annotators familiar with agent skills independently inspect each instance. The generating model, optimization method, and experimental condition are anonymized, while all evidence required for judgment is retained. For Motivation-study Taxonomy, the explicit source identities of the supporting calibration examples are hidden. For Skill-Change Coding, annotators are not informed whether an instance comes from the public or trajectory source.

Each annotator assigns a binary label, Accept or Reject, and provides a brief reason for rejected instances. An output is accepted only when all corresponding criteria specified below are satisfied. Disagreements are resolved by a third annotator.

Before the formal audit, the annotators complete a pilot round of 20 additional instances for each decision type. These pilot instances are excluded from the final results. If the interannotator agreement in a pilot round is below κ = 0.7, the corresponding annotation guidelines are clarified and the pilot round is repeated.

Annotation Criteria. For Motivation-study Taxonomy, annotators inspect a category definition and one of its supporting calibration examples. The instance is accepted if the category represents a coherent evolution concept, is semantically distinguishable from other categories at the same level, is applicable independently of the evolution source, and is correctly supported by the given example.

For Skill-Change Coding, annotators inspect the paired skill states, the complete segmentation as context, and one highlighted skill-change unit. The instance is accepted only if the highlighted unit: (1) groups edits that implement the same primary intent through the same modification pattern; (2) includes all semantically coupled edits required to implement that change; (3) excludes unrelated edits; and (4) assigns component, intent, and pattern labels that are semantically correct and supported by the underlying skill change. Errors in non-highlighted units do not afect the judgment of the current audit instance.

For Change Abstraction, annotators inspect a raw dif and its generated update event. The event is accepted if the raw dif segment corresponds to the actual change and its component, operation, and intent are all semantically correct and directly supported by the dif.

For Experience Generation, annotators inspect an experience-bank entry together with its supporting lowerlevel items. An entry is accepted if it faithfully summarizes its supporting items, conforms to the intended abstraction level, contains no unsupported claims, and provides reusable guidance for skill optimization.

For Experience Selection, annotators inspect the selection context z and one selected experience. The selection is accepted if the experience is relevant to both the current task and the current skill state and provides applicable guidance for the current optimization step.

For Source Attribution, annotators inspect a provenancetagged candidate edit, the realized skill dif, and the predicted attribution. The attribution is accepted if the candidate edit is correctly matched to the realized change and its label follows our definition: +1 for a realized externally supported edit, −1 for a realized self-supported edit, and 0 for an unrealized, uncertain, or mixed-provenance edit.

Table 6: Human audit results for the LLM-based semantic decisions. HAR is computed from the final adjudicated labels, whereas κ measures agreement between the two initial annotators before adjudication.
<table><tr><td>Decision Type</td><td>Samples HAR (%) ↑ κ ↑</td><td></td><td></td></tr><tr><td>Motivation study</td><td></td><td></td><td></td></tr><tr><td>Motivation-study Taxonomy Skill-Change Coding</td><td>200 200</td><td>94.0 100.0</td><td>0.84 0.84</td></tr><tr><td>VCE-Skill method</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Change Abstraction</td><td>200 200</td><td>92.5</td><td>0.82</td></tr><tr><td>Experience Generation</td><td>200</td><td>90.5 86.5</td><td>0.78</td></tr><tr><td>Experience Selection</td><td></td><td></td><td>0.75</td></tr><tr><td>Source Attribution</td><td>200</td><td>90.5</td><td>0.80</td></tr></table>

Metrics. We report two common metrics for all six decision types. First, the Human Acceptance Rate (HAR) is the proportion of outputs accepted after adjudication:

$$
\mathrm { H A R } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } { \bf 1 } \left[ \widetilde { y } _ { i } = \mathrm { A c c e p t } \right] , \qquad N = 2 0 0 ,\tag{17}
$$

where $\widetilde { y } _ { i }$ is the final adjudicated label.

Second, we report Cohen’s κ between the two initial annotators to quantify the consistency of the annotation criteria. HAR is computed from the final adjudicated labels, whereas Cohen’s κ is computed from the two annotators’ independent labels before adjudication.

Audit Results and Summary. As shown in Table 6, the six audited decision types achieve HAR values ranging from 86.5% to 100.0%, with Cohen’s κ values between 0.75 and 0.84. The two decisions used in the motivation study show strong agreement with human assessment. Motivation-study Taxonomy achieves an HAR of 94.0% and a κ of 0.84, while all audited Skill-Change Coding instances are accepted after adjudication, with a pre-adjudication κ of 0.84. These results support the reliability of the taxonomy construction and skillchange annotations underlying the motivation-study findings.

The LLM-based decisions within VCE-Skill also remain consistently aligned with humanjudgments. Change Abstraction, Experience Generation, and Source Attribution achieve HAR values above 90%, indicating that most generated update events and hierarchical experience entries are faithful to their supporting changes and that most provenance assignments are consistent with the realized skill updates. Experience Selection obtains the lowest HAR, at 86.5%, which is consistent with the more context-dependent nature of determining whether an experience is applicable to the current task and skill state. Nevertheless, its κ of 0.75 indicates that the corresponding assessment criteria remain reasonably consistent across annotators. Source Attribution achieves an HAR of 90.5% and a κ of 0.80, providing additional support for its use as the feedback signal in adaptive experience attention.

Overall, the high acceptance rates and consistent interannotator agreement provide empirical evidence that the LLM-based semantic decisions used in both the motivation study and VCE-Skill are suficiently reliable for the analyses and optimization procedure. These results do not imply that the LLM judgments are error-free; rather, they show that the outputs are generally supported by human assessment under the defined audit criteria.