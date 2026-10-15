# ReFormBench: Benchmarking the Stability of Response-Conditioned Format Correction in Large Language Models

Tianxiang Gao<sup>1,2</sup> Yahan Li<sup>3</sup> Chenlu Guo<sup>4</sup> Diana Inkpen<sup>5</sup> Yi Chang<sup>1,6,7</sup> Yuan Wu<sup>1</sup>\*

<sup>1</sup>School of Artificial Intelligence, Jilin University

<sup>2</sup> College of Software, Jilin University

<sup>3</sup>Xingchen AGI Lab, China Telecom Artificial Intelligence Technology (Beijing) Co., Ltd <sup>4</sup>BiliBili Inc. <sup>5</sup>University of Ottawa

<sup>6</sup>Engineering Research Center of Knowledge-Driven Human-Machine Intelligence, Jilin University <sup>7</sup>International Center of Future Science, Jilin University

## Abstract

Reliable use of large language models requires responses to satisfy formatting requirements across changing instructions. Standard compliance scores do not distinguish preserving a model’s observed format from correcting it, or reveal whether that correction holds across variants. We introduce ReForm-Bench, a benchmark that uses each model’s unconstrained response to the same question as its formatting baseline. It links targets by feature and require/forbid mode, evaluating their instantiated requirements across variants. ReFormBench comprises 5,000 difficultyscreened three-constraint test instances derived from 910 source questions and covering 66 formatting features. TCPR measures complete instruction compliance, while EMCR measures individual and joint correction of targets opposed to the annotated baseline state. Across ten models, EMCR ranges from 34.04% to 64.72%, and the highest three-target correction estimate is 50.02%. Compliance and correction scores yield different model rankings and task-level profiles. A prompting experiment on Qwen3.5-9B improves both metrics, and the catalog covers over 95% of extracted instances on four additional models. ReForm Bench makes response-conditioned correction across instruction variants a measurable dimension of format following. The code and data are available at https://anonymous.4open. science/r/ReFormBench/.

## 1 Introduction

Large language models (LLMs) increasingly support interactive applications and automated workflows (Wu et al., 2022; Yao et al., 2023; Shen et al., 2023). Their responses must satisfy both content needs and the formats expected by users and downstream systems (Xia et al., 2024; Dong et al., 2025). An extraction pipeline may require a fixed structure (Geng et al., 2023), a programming interface may expect code without commentary, and a document task may require continuous prose. Such instructions can jointly constrain headings, lists, markup, and response boundaries. Format following therefore involves coordinating requirements within a response and carrying them across different instruction configurations.

![](images/9f8788a3c3993a713b7f91306bcff04f53e6071fa427f2b5d534948bde210651.jpg)  
Figure 1: Illustration of correcting a model’s observed format with an explicit constraint.

Format compliance alone does not identify which requirements change a model’s observed behavior. A prohibition on bullet points demands a feature-state change when the model’s unconstrained answer uses a list, but agrees with the observed state when that answer contains no list. The same requirement can therefore represent correction for one model–question pair and continuation for another (Huang et al., 2026). The response to the same question without explicit formatting requirements supplies an empirical reference for separating these cases.

A second question concerns correction across variants, given the challenges of constraint composition (Wen et al., 2024; Zou et al., 2025). One instruction may combine a list prohibition with a heading requirement; another pairs that prohibition with a prescribed ending. Success in one variant does not establish success in the other. We identify a target by its formatting feature and require or forbid mode, holding the model and source question fixed. Correction stability then means satisfying that target wherever it occurs, across its instantiated requirements and accompanying constraints.

Existing benchmarks assess verifiable instructions (Zhou et al., 2023; Qin et al., 2024), constraint composition (Jiang et al., 2024; Wen et al., 2024), and output-format adherence (Xia et al., 2024; Wang et al., 2025; Lu et al., 2025; Yang et al., 2025b). Inverse IFEval (Zhang et al., 2025) examines departures from common conventions, while Harness-IF (Huang et al., 2026) evaluates rules opposing coding-agent defaults. ReForm-Bench brings a model–question-specific response baseline together with cross-variant target evaluation, distinguishing full instruction compliance from correction relative to observed formatting.

We examine this distinction with shared instruction variants, then test its sensitivity to constraint composition, prompting, and feature-catalog coverage.

Our contributions are threefold:

• We establish a response-conditioned protocol that identifies baseline-opposed feature–mode targets from each model’s unconstrained responses and links them across instruction variants. EMCR measures individual and joint correction; TCPR separately measures full instruction compliance.

• We construct 5,000 satisfiability-checked, three-constraint instructions from 910 questions across ten domains and seven tasks. They draw on 66 formatting features in eight categories, with model-specific baselines and explicit acceptance criteria.

• We show distinct compliance and correction rankings across ten models and a penalty when Qwen3.6-27B addresses constraints jointly. Qwen3.5-9B prompting improves both metrics, and catalog coverage exceeds 95% on four additional models.

<table><tr><td>Benchmark</td><td>FF McC StC PrB</td><td></td></tr><tr><td>IFEval (Zhou et al., 2023) FollowBench (Jiang et al., 2024)</td><td>△ △ ×  $\triangle$   $\bigtriangleup$  △</td><td>X X</td></tr><tr><td>ComplexBench (Wen et al., 2024)</td><td> $\triangle$   $\bigtriangleup$  △</td><td>X</td></tr><tr><td>EIFBench (Zou et al., 2025) FOFO (Xia et al., 2024)</td><td>△ √ X</td><td>X</td></tr><tr><td>SchemaBench (Lu et al., 2025)</td><td>√ √ X</td><td>X</td></tr><tr><td>StructEval (Yang et al., 2025b)</td><td>√ √ X</td><td>X</td></tr><tr><td>Inverse IFEval (Zhang et al., 2025)</td><td>√ √ X</td><td>X</td></tr><tr><td>Harness-IF (Huang et al., 2026)</td><td> $\triangle$   $\bigtriangleup$  ×</td><td>×</td></tr><tr><td>ReFormBench (Ours)  $\checkmark$ </td><td> $\triangle$   $\checkmark$ </td><td>X  $\bigtriangleup$ </td></tr></table>

FF: format focus $( \checkmark )$ primary evaluation focus; $\triangle \colon$ part of a broader instruction taxonomy). McC: multi-constraint coverage $( \checkmark )$ multi-constraint throughout; $\triangle \colon$ mixed single-/multiconstraint instances or levels). Generic task fulfillment is excluded; explicit schema requirements count. StC: stability across compositions $( \checkmark )$ feature-and-mode target evaluation across instruction variants for the same model and source question; $\triangle \colon$ related cross-level or branch-condition consistency protocols). PrB: paired response baseline $( \checkmark )$ a paired unconstrained response identifies model-specific correction targets; $\triangle \colon$ empirical prior control without model–question-specific correction targets). × denotes absence under the corresponding definition.

Table 1: Comparison of benchmark designs.

## 2 Related Work

## 2.1 Instruction- and Format-Following Evaluation

Within the broader evaluation of LLMs (Chang et al., 2024), instruction-following benchmarks examine verifiable requirements, constraint composition, and multi-task workflows. IFEval (Zhou et al., 2023) establishes a framework based on programmatically verifiable instructions. Follow-Bench (Jiang et al., 2024) progressively adds constraints across levels; its Consistent Satisfaction Levels (CSL) metric measures consecutive fully satisfied levels starting at level 1. ComplexBench (Wen et al., 2024) explicitly models constraint composition and introduces a coherent test for Selection that jointly evaluates instructions with shared branches but different selection conditions. EIF-Bench (Zou et al., 2025) extends evaluation to workflows containing multiple tasks and associated constraints.

Format-focused benchmarks examine adherence to specified output structures and presentation requirements. FOFO (Xia et al., 2024) evaluates detailed domain-specific formatting requirements separately from content quality. SchemaBench (Lu et al., 2025) tests JSON generation under structural and field-level constraints, while StructEval (Yang et al., 2025b) covers generation and conversion across renderable and non-renderable formats.

![](images/42635dd01b556e872098f48d948cdbd11c6fbf81a8ce1e6522ab74a70ce721a0.jpg)  
Figure 2: Benchmark construction and evaluation. Source questions support a shared formatting constraint space and model-specific response baselines. The finalized benchmark contains 5,000 three-constraint instruction variants. Quality control covers instruction validity, difficulty screening, and response judging; evaluation measures complete compliance (TCPR) and individual and joint correction across variants (EMCR).

The consistency protocols discussed above aggregate success across instruction levels or selection conditions. ReFormBench evaluates baselineopposed formatting targets in every variant where they occur, holding the model and source question fixed. It measures correction of individual targets and joint correction of co-occurring target sets, with joint success restricted to the selected baseline-opposed targets.

## 2.2 Behavioral Priors and Instruction Following

Inverse IFEval (Zhang et al., 2025) evaluates counterintuitive instruction following across eight categories, including Counter-Conventional Formatting. Its formatting requirements depart from familiar output structures by reversing common behavioral conventions associated with post-training.

Harness-IF (Huang et al., 2026) further investigates the relationship between default behaviors and instruction following in coding agents, including format-related output requirements. It probes behavior with target rules withheld and evaluates compliance with rules labeled as opposing these defaults.

Building on this line of work, ReFormBench uses each model’s unconstrained response to the same source question to identify formatting targets that require correction. It then links feature-andmode targets across variants and evaluates their instantiated requirements together with changing accompanying constraints. Table 1 summarizes the related benchmarks.

## 3 ReFormBench

## 3.1 Evaluation Setup

We pair each model’s unconstrained response with its responses to shared instruction variants of the same question. Baseline annotations identify feature-level targets requiring correction; target links track their satisfaction across variants. TCPR summarizes complete instruction compliance, and EMCR summarizes correction relative to these model-specific baselines. Figure 2 presents the construction and evaluation pipeline.

## 3.2 Source Questions

We assemble public-source questions from Wild-Bench (Lin et al., 2024), ThoughtTrace (Jin et al., 2026), Inverse IFEval (Zhang et al., 2025), ELI5 (Fan et al., 2019), and ExpertQA (Malaviya et al., 2024). These sources cover user requests, planning, behavioral conventions, explanatory questions, and expert tasks. GPT-5.4 removes explicit formatting requirements and self-checks task preservation and constraint removal; a constraint checker screens for residual requirements. The process yields 553 public-source questions (Appendix Table A10 lists their licenses and documented uses). We author 357 further questions, particularly for information extraction, to broaden coverage. The 910 source questions span ten domains and seven task types (Table A9).

## 3.3 Constraint and Instruction Construction

Shared Constraint Space. We construct the shared catalog from formatting features extracted from unconstrained responses across models and source questions. Extraction combines annotation of predefined core features with open-ended identification of additional features. We normalize equivalent descriptions and consolidate implementation variants that share constraint descriptions and acceptance criteria, yielding 66 features across eight categories. Each feature has complementary require and forbid templates with explicit acceptance criteria, forming a shared constraint space for instruction construction and evaluation (Appendix A.7.3).

Model-Specific Behavioral Baselines. For each model–question pair, positive catalog annotations define the feature set $P _ { m , q }$ of its unconstrained response (Appendix A.6). A target is baselineopposed when it forbids a feature in this set or requires one outside it; the remaining targets are baseline-aligned. These groups describe the direction of the requested generic feature state. Every response is assessed against the instantiated requirement, including its counts, positions, and other parameters. This separates feature-state correction from instruction-specific formatting details.

Shared Variants and Recurring Targets. Each instruction combines three constraints from distinct categories. A target’s identity is its feature and require/forbid mode; concrete parameters belong to its occurrences in individual variants. We link occurrences by source question and target identity, then evaluate each against its own acceptance criteria. Difficulty screening retains 5,000 variants after discarding the highest-scoring candidates under mean calibrated compliance across the ten screening models. All evaluated models receive this shared set. Generation, target linking, and satisfiability review are detailed in Appendix A.7.

## 3.4 Metrics

We report two metrics. Three-Constraint Pass Rate (TCPR) measures compliance with all three constraints within an instruction. Equal-Weight Multi-Constraint Correction Rate (EMCR) measures individual and joint correction of baselineopposed targets across relevant variants. For a fixed model, $y = 1$ denotes a pass and y = 0 a failure. Scores are percentages; absolute differences are expressed in percentage points (pp).

## 3.4.1 TCPR

For N instructions, let $y _ { i , j }$ indicate whether the response to instruction i satisfies constraint j. An instruction passes only when all three constraints are satisfied:

$$
T C P R = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \prod _ { j = 1 } ^ { 3 } y _ { i , j }\tag{1}
$$

## 3.4.2 EMCR

For a fixed model and source question $q ,$ let $T _ { q }$ contain the baseline-opposed targets defined in Section 3.3. A target t is identified by its formatting feature and require/forbid mode. Let $V ( \boldsymbol { q } , t )$ contain the retained variants including t, and let $y _ { i , t }$ indicate satisfaction of its instantiated requirement in variant i.

An evaluation unit $( q , S )$ selects K targets from $T _ { q } .$ . Eligibility requires the targets to co-occur in at least one variant and collectively cover at least two variants:

$$
\begin{array} { l l } { \mathcal { U } _ { K } ^ { \mathrm { c o } } = \displaystyle \big \{ ( q , S ) ~ \big | ~ S \subseteq T _ { q } , ~ | S | = K , } \\ & { ~ \displaystyle \bigcap _ { t \in S } V ( q , t ) \neq \emptyset , } \\ & { ~ \displaystyle \bigcup _ { t \in S } V ( q , t ) \Bigg | \geq 2 \big \} . } \end{array}\tag{2}
$$

Success requires every selected target to pass in every variant containing it:

$$
J _ { q , S } = \left\{ { \begin{array} { l l } { 1 , } & { y _ { i , t } = 1 { \mathrm { f o r } } { \mathrm { a l l } } t \in S } \\ { } & { { \mathrm { a n d } } i \in V ( q , t ) , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} } \right.\tag{3}
$$

The correction rate averages successful units at each K level:

$$
R _ { \mathrm { c o } } ^ { ( K ) } = \frac { \displaystyle \sum _ { ( q , S ) \in \mathcal { U } _ { K } ^ { \mathrm { c o } } } J _ { q , S } } { \displaystyle \left| \mathcal { U } _ { K } ^ { \mathrm { c o } } \right| } , K \in \{ 1 , 2 , 3 \}\tag{4}
$$

EMCR gives the three K levels equal weight:

$$
E M C R = \frac { 1 } { 3 } \sum _ { K = 1 } ^ { 3 } R _ { \mathrm { c o } } ^ { ( K ) }\tag{5}
$$

## 3.4.3 Interpreting the Metrics

For example, requiring two named headings and requiring three named headings share the sectionheading/require identity, but each occurrence is judged against its own count and names. Target links group these occurrences for correction evaluation; the outcome of one occurrence does not determine the outcome of another. Every judgment remains tied to the concrete instruction under which the response was generated.

Here, K counts jointly evaluated targets; every instruction still has three constraints. The twovariant requirement evaluates correction beyond a single instruction configuration. At K = 1, the individual target must recur. At $K = 2 , 3 ,$ , the targets must co-occur at least once and their combined coverage must span multiple variants; a member target may occur once. Each level therefore evaluates its own eligible target sets. Joint success includes variants where the other selected targets are absent. Each K level contributes equally to EMCR, regardless of its number of eligible units. Appendix A.1 gives the complete worked calculation.

Reading the two metrics. TCPR and EMCR answer different questions on the same response collection. TCPR asks whether one response meets all three requirements. EMCR asks whether a baselineopposed target, or a selected set of such targets, succeeds everywhere it is required. A variant can fail TCPR because of an unselected constraint while still satisfying the targets examined by an EMCR unit. Conversely, a target set can pass in its shared variant but fail joint correction when a selected target fails in another variant. The evaluation follows target occurrences, rather than averaging instruction scores.

From target links to evaluation units. An illustrative three-variant case makes the distinction concrete. Suppose a model’s baseline has a bullet list and a closing invitation, with no heading annotated. Let a forbid bullet lists, b require headings, and c forbid a closing invitation. All three targets oppose the annotated baseline state. The variants place these targets in different combinations, with occurrence sets

$$
\begin{array} { l } { { V ( q , a ) = \{ v _ { 1 } , v _ { 2 } , v _ { 3 } \} , } } \\ { { V ( q , b ) = \{ v _ { 1 } , v _ { 2 } \} , } } \\ { { V ( q , c ) = \{ v _ { 1 } \} . } } \end{array}\tag{6}
$$

Each variant has three constraints; baseline-aligned requirements fill the remaining slots in v<sub>2</sub> and $v _ { 3 }$ For $K = 1$ , a and b qualify because each recurs. For K = 2, all three pairs qualify: their members share v<sub>1</sub>, and each pair’s union covers at least two variants. The triple also qualifies. The singleton $\{ c \}$ is ineligible, while c can participate in joint units whose combined coverage spans multiple variants.

Now suppose all occurrences pass except b in $v _ { 2 }$ . The unit $\{ a , c \}$ succeeds because both targets pass wherever required. The unit $\{ a , b \}$ fails despite both targets passing in $v _ { 1 }$ , since the linked occurrence of b in v<sub>2</sub> also belongs to its success event. The triple fails for the same reason. This occurrence-based accounting preserves the connection between the source response, the requested feature-state changes, and every relevant output. Appendix A.1 supplies the judgment matrix and completes the calculation at all three K levels.

Model-specific correction profiles. All models receive the same instructions, while each model’s baseline determines its correction targets. EMCR therefore characterizes correction relative to that model’s observed formatting. For example, requiring headings opposes an unheaded baseline but aligns with a headed one. Eligible counts and mean variant coverage accompany each model’s scores (Table A8). Within-model prompting comparisons hold the baseline and target links fixed, so both conditions evaluate the same eligible units. Tasklevel results apply the same construction within each task category.

## 3.5 Response Judging and Calibration

GPT-5.4 Mini and Gemini 3.1 Flash-Lite independently assess each response against its constraints and acceptance criteria, returning binary labels and supporting evidence. For a response– constraint instance $u ,$ their respective judgments are $g _ { u } , m _ { u } \in \{ 0 , 1 \}$

For the main evaluation, human adjudication is restricted to sampled response–constraint instances with conflicting judge labels. We stratify disagreements within each model by direction, feature category, and constraint mode, covering every nonempty stratum. For each sampled instance, we compare both judges’ evidence against the instruction, response, and acceptance criteria and select the better-supported judgment (Appendix A.3.1). Let $\mathcal { R } _ { h }$ contain the reviewed disagreements in stratum $h ,$ with $a _ { u }$ denoting the selected judge’s binary label. The estimated pass rate for stratum h is

$$
\widehat { \pi } _ { h } = \frac { 1 } { | \mathcal { R } _ { h } | } \sum _ { u \in \mathcal { R } _ { h } } a _ { u } .\tag{7}
$$

Let $\textstyle { \mathcal { R } } = \bigcup _ { h } { \mathcal { R } } _ { h }$ and let $h ( u )$ denote the stratum

of $u .$ The calibrated pass probability is

$$
p _ { u } = \left\{ { \begin{array} { l l } { g _ { u } , } & { g _ { u } = m _ { u } , } \\ { a _ { u } , } & { u \in { \mathcal { R } } , } \\ { { \widehat { \pi } } _ { h ( u ) } , } & { { \mathrm { o t h e r w i s e } } . } \end{array} } \right.\tag{8}
$$

Agreed judgments and reviewed disagreements retain binary labels; unreviewed disagreements use the estimated pass rate for their stratum. The probability $p _ { u }$ represents uncertainty in the constraint judgment.

For a fixed model, $p _ { i , j }$ and $p _ { i , t }$ index $p _ { u }$ by constraint position and target identity, respectively. Under a factorization assumption, we treat the constituent pass events as independent and approximate joint success by multiplying their estimated probabilities:

$$
\begin{array} { l } { \displaystyle \widehat { T C P R } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \prod _ { j = 1 } ^ { 3 } p _ { i , j } , } \\ { \displaystyle \widehat { J _ { q , S } } = \prod _ { t \in S } \prod _ { i \in V ( q , t ) } p _ { i , t } . } \end{array}\tag{9}
$$

We obtain E $\widehat { ^ { \prime } M C R }$ by replacing $J _ { q , S }$ with $\widehat { J } _ { q , S }$ in bEqs. 4–5. The constraint-composition and prompting experiments use fixed pooled calibration (Appendix A.3.2). All compliance and correction scores reported in this paper incorporate the corresponding judgment calibration.

## 4 Experiments

## 4.1 Experimental Setup

We evaluate ten models: Claude Opus 4.6 (Anthropic, 2026), DeepSeek-V4-Pro (DeepSeek-AI, 2026), GPT-5.6 Sol (OpenAI, 2026), Qwen3- 235B-A22B-Thinking (Yang et al., 2025a), Gemini 3.5 Flash (Google DeepMind, 2026b), Kimi K3 (Kimi Team, 2026), Qwen3.6-27B (Qwen Team, 2026b), Qwen3.6-35B-A3B (Qwen Team, 2026c), Qwen3.5-9B (Qwen Team, 2026a), and GLM-5.2 (GLM-5-Team, 2026; Z.ai, 2026).

All models use reasoning-enabled configurations. Detailed settings for Qwen3.6-27B, Qwen3- 235B-A22B-Thinking, and Qwen3.5-9B are listed in Table A13 (Appendix A.10).

The main results use stratified calibration, while the constraint-composition and prompting experiments use fixed pooled calibration (Appendix A.3). All conditions within an experiment share the same calibration rule.

## 4.2 Main Results

Scores below are calibrated point estimates. EMCR evaluates each model’s baseline-opposed target sets, with eligible counts and variant coverage reported in Table A8.

Instruction compliance and responseconditioned correction produce different model profiles. Qwen3.6-27B scores slightly below DeepSeek-V4-Pro on TCPR (65.46% versus 66.39%) but above it on EMCR (47.84% versus 44.81%; Table 2). These results distinguish overall compliance on shared instructions from correction on each model’s own baseline-defined targets.

Cross-variant evaluation exposes substantial room for sustained correction. On this difficultyscreened benchmark, EMCR ranges from 34.04% to 64.72%, with five models below 50%. Kimi K3 has the highest estimates at all three K levels: 80.22%, 63.92%, and 50.02% for their respective eligible sets. GPT-5.6 Sol follows at K = 3 with 46.42%. Thus, the highest estimated three-target success rate is approximately one half, even when each individual requirement has an explicit acceptance criterion.

Task-level profiles reveal strengths beyond overall rankings. Four models attain the highest EMCR point estimates across the seven tasks (Figure 3). Qwen3.6-35B-A3B ranks fourth overall but has the highest Decision Analysis estimate, 69.74%. Code Generation has the lowest mean EMCR (45.27%) and the lowest TCPR and EMCR for seven models. Creative Writing shows the widest EMCR range, 41.85 percentage points. Complete task results appear in Appendix A.5; Appendix A.2 further characterizes violations within unsuccessful units.

## 4.3 Constraint Composition Analysis

Using Qwen3.6-27B, we pair each of the 5,000 three-constraint instructions with three separately answered single-constraint versions. We divide the 15,000 target items into 7,959 baseline-opposed and 7,041 baseline-aligned items using the generic feature states defined in Section 3.3. Group membership remains fixed across conditions; each target is judged against its instantiated requirement. Separate and Joint use three responses and one response per instruction group, respectively (Appendix A.8).

Joint responses score below the separateresponse all-pass estimate. Joint all-pass is 64.43%, compared with 70.48% for the mean product of the three separately evaluated pass probabilities, a difference of 6.05 percentage points (Table 3). This paired comparison evaluates satisfying the requirements in one response versus answering them separately.

<table><tr><td colspan="3"></td><td colspan="3">EMCR components</td></tr><tr><td>Model</td><td>TCPR</td><td>EMCR</td><td>K = 1</td><td>K = 2</td><td>K = 3</td></tr><tr><td>Kimi K3</td><td>82.82</td><td>64.72</td><td>80.22</td><td>63.92</td><td>50.02</td></tr><tr><td>GPT-5.6 Sol</td><td>81.99</td><td>62.31</td><td>78.55</td><td>61.95</td><td>46.42</td></tr><tr><td>Gemini 3.5 Flash</td><td>78.08</td><td>60.63</td><td>78.28</td><td>60.20</td><td>43.42</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>77.82</td><td>60.34</td><td>76.83</td><td>60.46</td><td>43.71</td></tr><tr><td>GLM-5.2</td><td>75.76</td><td>55.50</td><td>74.33</td><td>55.09</td><td>37.07</td></tr><tr><td>DeepSeek-V4-Pro</td><td>66.39</td><td>44.81</td><td>65.83</td><td>43.89</td><td>24.72</td></tr><tr><td>Qwen3.6-27B</td><td>65.46</td><td>47.84</td><td>67.04</td><td>45.81</td><td>30.66</td></tr><tr><td>Claude Opus 4.6</td><td>60.09</td><td>42.33</td><td>63.74</td><td>41.24</td><td>22.00</td></tr><tr><td>Qwen3-235B-A22B-Thinking</td><td>56.33</td><td>34.04</td><td>54.76</td><td>31.16</td><td>16.22</td></tr><tr><td>Qwen3.5-9B</td><td>51.81</td><td>40.12</td><td>61.96</td><td>37.86</td><td>20.55</td></tr></table>

Table 2: Format following and sustained correction across variants for ten models (%). EMCR is the equal-weight mean of the three K-level correction rates.

![](images/d5cb906fa9ddcf5c22ba7187e2c924549b0b0195a4071c23a153ee9f3b7c6a30.jpg)  
Figure 3: Task-level format following across ten models and seven task types. TCPR (top) and EMCR (bottom), shown as calibrated point estimates (%). Both panels use the same model order and a shared 0–100 color scale. Values are rounded to one decimal place; task summaries and scores to two decimal places appear in Tables A5 and A6.

<table><tr><td>Measure</td><td>Separate</td><td>Joint ∆ (pp)</td><td></td></tr><tr><td>All-pass (%)</td><td>70.48</td><td>64.43</td><td>-6.05</td></tr><tr><td>Target pass (%)</td><td></td><td></td><td></td></tr><tr><td>Baseline-opposed</td><td>86.80</td><td>84.05</td><td>-2.75</td></tr><tr><td>Baseline-aligned</td><td>91.33</td><td>87.30</td><td>-4.03</td></tr><tr><td>Group gap (pp)</td><td>-4.53</td><td>-3.25</td><td>+1.28</td></tr></table>

Table 3: Separate versus joint constraint satisfaction on Qwen3.6-27B. ∆ is Joint minus Separate (pp); group gap is baseline-opposed minus baseline-aligned target pass. Separate all-pass averages products of calibrated pass probabilities. Differences use unrounded estimates.

Baseline-opposed targets have lower pass rates in both conditions. Their rates trail baselinealigned targets by 4.53 points under Separate and 3.25 points under Joint. Joint lowers target pass by 2.75 and 4.03 points in the two groups, respectively. The group gap narrows by 1.28 points while pass rates fall in both groups, showing that the direction relative to the baseline remains informative under both evaluation conditions.

## 4.4 Prompting Improvements

We test a fixed meta-prompt on Qwen3.5-9B by appending it to each of the same 5,000 instructions. The prompt provides guidance in four areas (Appendix A.8):

• Identify all explicit constraints.

• Plan a minimal output structure that satisfies these constraints.

• Avoid unrequested content.

![](images/97d4756ce18e7b2d455b4914de16b14d64896bda94723cf9ae82fbf6061c0b0c.jpg)  
Figure 4: Prompting configuration improvement for Qwen3.5-9B. Paired dots show TCPR and EMCR, followed by EMCR’s three K-level components. Bars show gains in percentage points, calculated from unrounded estimates.

• Check the response against each constraint and revise any violations before answering.

Both conditions share pooled calibration, baselineopposed targets, and eligible evaluation units.

Figure 4 shows TCPR increasing from 52.76% to 65.20% and EMCR from 40.33% to 46.11%. Correction rates increase at all three K levels, with the largest relative gain at K = 3 (21.28% to 27.51%). With the baseline and eligible units held fixed, this intervention demonstrates improvements in both instruction compliance and cross-variant correction for Qwen3.5-9B.

## 4.5 Catalog Coverage on Additional Models

To assess applicability beyond the construction models, we evaluate the catalog on Gemini 3.1 Pro (Google DeepMind, 2026a), DeepSeek-V3.2 (DeepSeek-AI, 2025), gpt-oss-120b (OpenAI, 2025), and Doubao-Seed-2.1-Pro (ByteDance Seed, 2026). We apply open-ended feature extraction to each model’s unconstrained responses to the same 910 source questions, then map the extracted features to the 66-feature catalog. Evaluation measures the proportion of valid extracted instances represented in the catalog and uses Spearman correlation to compare response-level feature frequencies with the ten-model reference (Appendix A.4).

Coverage exceeds 95% for each additional model and reaches 95.75% pooled, with 65 of 66 catalog features observed across the four models (Table 4). Each model instantiates 62–65 features, showing broad reuse of the shared vocabulary. The Spearman correlation between four-model mean feature frequencies and the ten-model reference is 0.804. Feature count, instance coverage, and frequency ranking provide complementary evidence: the catalog recurs across models, covers most observed formats, and retains a similar ordering of common features. Together, these results support using the catalog as a common reference for response-conditioned evaluation beyond the construction models.

<table><tr><td>Model</td><td>Cov. (%)</td><td>Catalog features</td><td>ρ</td></tr><tr><td>Gemini 3.1 Pro</td><td>95.05</td><td>64/66</td><td>0.737</td></tr><tr><td>DeepSeek-V3.2</td><td>96.03</td><td>64/66</td><td>0.777</td></tr><tr><td>gpt-oss-120b</td><td>96.54</td><td>65/66</td><td>0.805</td></tr><tr><td>Doubao-Seed-2.1-Pro</td><td>95.21</td><td>62/66</td><td>0.776</td></tr><tr><td>Four models pooled</td><td>95.75</td><td>65/66</td><td>0.804</td></tr></table>

Table 4: Catalog coverage on four additional models, each with 910 responses. “Cov.” gives the percentage of valid extracted feature instances covered by the catalog. Catalog features are distinct entries (union for the pooled row). Spearman ρ compares feature frequencies with the ten-model reference; the pooled row uses four-model mean frequencies and aggregated instance counts.

## 5 Conclusion

ReFormBench evaluates whether explicit formatting requirements redirect a model from its observed response pattern and whether correction persists across instruction variants. Shared instructions and model-specific unconstrained baselines identify baseline-opposed feature-and-mode targets. TCPR measures full-instruction compliance, while EMCR measures individual and joint correction, separating success on a particular instruction from stable change in the targeted formatting behavior.

Across ten models, the two measures yield distinct rankings and task profiles; the highest calibrated three-target correction estimate is 50.02% on the difficulty-screened benchmark. Qwen3.6-27B’s joint constraint responses score below the separateresponse all-pass estimate, while Qwen3.5-9B prompting improves both compliance and correction under fixed targets. Catalog coverage exceeds 95% on each of four additional models. Together, these results establish response-conditioned correction as a measurable dimension of format following and provide a basis for testing whether improved instruction compliance also sustains the requested change.

## Limitations

ReFormBench uses one unconstrained response per model–question pair as its formatting baseline. EMCR therefore describes correction relative to that observed response, not the frequency of its format across repeated samples. Because concrete target parameters and eligible units can differ across variants and models, cross-model EMCR values concern model-specific correction populations rather than identical target sets.

The calibrated scores combine two automated judges, human review of sampled disagreements, and a factorized estimate ofjoint pass events. Judge agreements are not independently reviewed, and dependence among pass events is not estimated. Direct adjudication of complete evaluation units would help quantify the sensitivity of the scores to these assumptions.

The experiments cover difficulty-screened, single-turn, three-constraint text formatting. The four additional models are assessed for catalog coverage rather than correction, while the composition and prompting analyses each use one model. In the composition comparison, three separate responses are contrasted with one joint response, so the estimated gap pertains to these response settings. Repeated baselines, matched target sets, and broader correction evaluation with content-quality checks would test how far the findings extend beyond this setting.

## References

Anthropic. 2026. Claude Opus 4.6 system card. System card, Anthropic.

ByteDance Seed. 2026. Seed2.1 model card: Agentic intelligence for productivity. Technical report, ByteDance.

Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, Wei Ye, Yue Zhang, Yi Chang, Philip S. Yu, Qiang Yang, and Xing Xie. 2024. A survey on evaluation of large language models. ACM Transactions on Intelligent Systems and Technology, 15(3):1–45.

DeepSeek-AI. 2025. DeepSeek-V3.2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556.

DeepSeek-AI. 2026. DeepSeek-V4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348.

Yixin Dong, Charlie F. Ruan, Yaxing Cai, Ruihang Lai, Ziyi Xu, Yilong Zhao, and Tianqi Chen. 2025. XGrammar: Flexible and efficient structured generation engine for large language models. In Proceedings ofMachine Learning and Systems, volume 7.

Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. 2019. ELI5: Long form question answering. In Proceedings of the 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 3558–3567, Florence, Italy. Association for Computational Linguistics.

Saibo Geng, Martin Josifoski, Maxime Peyrard, and Robert West. 2023. Grammar-constrained decoding for structured NLP tasks without finetuning. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 10932– 10952, Singapore. Association for Computational Linguistics.

GLM-5-Team. 2026. GLM-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763.

Google DeepMind. 2026a. Gemini 3.1 Pro model card. Model card.

Google DeepMind. 2026b. Gemini 3.5 Flash model card. Model card.

Zining Huang, Haoran Que, Hong Zeng, Ge Zhang, Zuo Wang, Jin Chen, Haodong Wang, Zhongfei Hou, Changxin Pu, Shen Yan, and Wenhao Huang. 2026. Harness-IF: Evaluating instruction following across instruction surfaces in coding agents. arXiv preprint arXiv:2608.11727.

Yuxin Jiang, Yufei Wang, Xingshan Zeng, Wanjun Zhong, Liangyou Li, Fei Mi, Lifeng Shang, Xin Jiang, Qun Liu, and Wei Wang. 2024. Follow-Bench: A multi-level fine-grained constraints following benchmark for large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4667–4688, Bangkok, Thailand. Association for Computational Linguistics.

Chuanyang Jin, Binze Li, Haopeng Xie, Cathy Mengying Fang, Tianjian Li, Shayne Longpre, Hongxiang Gu, Maximillian Chen, and Tianmin Shu. 2026. ThoughtTrace: Understanding user thoughts in real-world LLM interactions. arXiv preprint arXiv:2605.20087.

Kimi Team. 2026. Kimi K3: Open frontier intelligence. arXiv preprint arXiv:2607.24653.

Bill Yuchen Lin, Yuntian Deng, Khyathi Chandu, Faeze Brahman, Abhilasha Ravichander, Valentina Pyatkin, Nouha Dziri, Ronan Le Bras, and Yejin Choi. 2024. WildBench: Benchmarking LLMs with challenging tasks from real users in the wild. arXiv preprint arXiv:2406.04770.

Yaxi Lu, Haolun Li, Xin Cong, Zhong Zhang, Yesai Wu, Yankai Lin, Zhiyuan Liu, Fangming Liu, and Maosong Sun. 2025. Learning to generate structured output with schema reinforcement learning. arXiv preprint arXiv:2502.18878.

Chaitanya Malaviya, Subin Lee, Sihao Chen, Elizabeth Sieber, Mark Yatskar, and Dan Roth. 2024. ExpertQA: Expert-curated questions and attributed answers. In Proceedings of the 2024 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3025–3045, Mexico City, Mexico. Association for Computational Linguistics.

OpenAI. 2025. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925.

OpenAI. 2026. GPT-5.6 system card. System card, OpenAI.

Yiwei Qin, Kaiqiang Song, Yebowen Hu, Wenlin Yao, Sangwoo Cho, Xiaoyang Wang, Xuansheng Wu, Fei Liu, Pengfei Liu, and Dong Yu. 2024. InFoBench: Evaluating instruction following ability in large language models. In Findings of the Association for Computational Linguistics: ACL 2024, pages 13025– 13048. Association for Computational Linguistics.

Qwen Team. 2026a. Qwen3.5: Towards native multimodal agents.

Qwen Team. 2026b. Qwen3.6-27B: Flagship-level coding in a 27B dense model.

Qwen Team. 2026c. Qwen3.6-35B-A3B: Agentic coding power, now open to all.

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. 2023. Hugging-GPT: Solving AI tasks with ChatGPT and its friends in Hugging Face. In Advances in Neural Information Processing Systems, volume 36.

Zhaoyang Wang, Jinqi Jiang, Huichi Zhou, Wenhao Zheng, Xuchao Zhang, Chetan Bansal, and Huaxiu Yao. 2025. Verifiable format control for large language model generations. In Findings ofthe Associationfor Computational Linguistics: NAACL 2025, pages 3499–3513. Association for Computational Linguistics.

Bosi Wen, Pei Ke, Xiaotao Gu, Lindong Wu, Hao Huang, Jinfeng Zhou, Wenchuang Li, Binxin Hu, Wendy Gao, Jiaxin Xu, Yiming Liu, Jie Tang, Hongning Wang, and Minlie Huang. 2024. Benchmarking complex instruction-following with multiple constraints composition. In Advances in Neural Information Processing Systems, volume 37.

Tongshuang Wu, Michael Terry, and Carrie J. Cai. 2022. AI Chains: Transparent and controllable human-AI interaction by chaining large language model prompts. In Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems, pages 1–22. Association for Computing Machinery.

Congying Xia, Chen Xing, Jiangshu Du, Xinyi Yang, Yihao Feng, Ran Xu, Wenpeng Yin, and Caiming Xiong. 2024. FOFO: A benchmark to evaluate LLMs’ format-following capability. arXiv preprint arXiv:2402.18667.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025a. Qwen3 technical report. Preprint, arXiv:2505.09388.

Jialin Yang, Dongfu Jiang, Lipeng He, Sherman Siu, Yuxuan Zhang, Disen Liao, Zhuofeng Li, Huaye Zeng, Yiming Jia, Haozhe Wang, Benjamin Schneider, Chi Ruan, Wentao Ma, Zhiheng Lyu, Yifei Wang, Yi Lu, Quy Duc Do, Ziyan Jiang, Ping Nie, and Wenhu Chen. 2025b. StructEval: Benchmarking LLMs’ capabilities to generate structural outputs. arXiv preprint arXiv:2505.20139.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations.

Z.ai. 2026. GLM-5.2. Official model card. Accessed September 9, 2026.

Qinyan Zhang, Xinping Lei, Ruijie Miao, Yu Fu, Haojie Fan, Le Chang, Jiafan Hou, Dingling Zhang, Zhongfei Hou, Ziqiang Yang, Changxin Pu, Fei Hu, Jingkai Liu, Mengyun Liu, Yang Liu, Xiang Gao, Jiaheng Liu, Tong Yang, Zaiyuan Wang, and 2 others. 2025. Inverse IFEval: Can LLMs unlearn stubborn training conventions to follow real instructions? arXiv preprint arXiv:2509.04292.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911.

Tao Zou, Xinghua Zhang, Haiyang Yu, Minzheng Wang, Fei Huang, and Yongbin Li. 2025. EIFBENCH: Extremely complex instruction following benchmark for large language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 20930–20953, Suzhou, China. Association for Computational Linguistics.

## A Supplementary Methods and Results

Appendix A.1 walks through EMCR from a baseline to joint success. Appendices $\mathsf { A } . 2 \mathrm { - A } . 5$ report violation statistics, judgment calibration, catalog coverage, and complete task results. Appendices A.6–A.8 document baselines, construction, and paired experiments. The remaining sections provide combination analyses, model settings, two response-judging examples, and the five experimental prompts.

## A.1 A Worked EMCR Calculation

This worked example traces EMCR from an illustrative response baseline through target links and variant-level judgments. We use binary judgments to make each step transparent; calibrated probabilities enter the same products in Eq. 9.

Baseline and targets. Consider one model answering one source question. Its baseline is annotated as containing a bullet list and a closing invitation, with no section heading annotated. Three targets therefore oppose its generic feature state: a forbids a bullet list, b requires section headings, and c forbids a closing invitation. Their identities are feature-and-mode pairs; any requested heading names or other concrete parameters are checked in their respective variants.

Three instruction variants. Variant $v _ { 1 }$ contains $a , b , c ;$ v contains a, b and one baseline-aligned constraint; $v _ { 3 }$ contains a and two baseline-aligned constraints. Thus every instruction still has three constraints. The other constraints pass in this illustration. Table A1 gives the selected-target judgments. A dash denotes absence of a target, not a failure.

<table><tr><td>Variant</td><td>a</td><td>b</td><td>C</td><td>Other passes</td></tr><tr><td>v1</td><td>1</td><td>1</td><td>1</td><td></td></tr><tr><td>v2</td><td>1</td><td>0</td><td>一</td><td>1</td></tr><tr><td>v3</td><td>1</td><td>一</td><td>一</td><td>2</td></tr></table>

Table A1: Illustrative target judgments for one model– question pair. Each variant contains three constraints. Other passes count satisfied baseline-aligned constraints.

Coverage and eligible units. The occurrence sets are $V ( \boldsymbol { q } , \boldsymbol { a } ) = \{ \boldsymbol { v } _ { 1 } , \boldsymbol { v } _ { 2 } , \boldsymbol { v } _ { 3 } \} , V ( \boldsymbol { q } , \boldsymbol { b } )$ = $\{ v _ { 1 } , v _ { 2 } \}$ , and $V ( q , c ) = \{ v _ { 1 } \} . \mathrm { A t } \ : K = 1 $ , the eligible sets are a and b because each target recurs. At $K = 2 ,$ , all three pairs are eligible: their members co-occur in $v _ { 1 }$ and each pair covers at least two variants in total. The triple $\{ a , b , c \}$ is also eligible. Target c therefore enters joint units through their combined coverage despite having one occurrence.

Success and aggregation. Target a passes in all three occurrences, whereas b fails in $v _ { 2 }$ . Hence $J _ { q , \{ a \} } = 1$ and ${ \cal J } _ { q , \{ b \} } = 0$ . For the pairs, ${ \cal J } _ { q , \{ a , b \} } =$ $0 , J _ { q , \{ a , c \} } = 1$ , and ${ J } _ { { q , \{ b , c \} } } = 0 ;$ the triple also fails. The passing shared variant $v _ { 1 }$ does not rescue a set containing b, because its occurrence in $v _ { 2 }$ is also evaluated. For this toy collection, $R _ { \mathrm { c o } } ^ { ( 1 ) } = 1 / 2 $ $R _ { \mathrm { c o } } ^ { ( 2 ) } = 1 / 3$ , and $R _ { \mathrm { c o } } ^ { ( 3 ) } = 0$ . Therefore,

$$
E M C R = \frac { 1 } { 3 } \left( \frac { 1 } { 2 } + \frac { 1 } { 3 } + 0 \right) \approx 2 7 . 7 8 \%\tag{10}
$$

By comparison, v<sub>1</sub> and $v _ { 3 }$ satisfy all three constraints, yielding $T C P R = 2 / 3 \approx 6 6 . 6 7 \%$ . The difference follows from the success events: TCPR counts complete instructions, while EMCR follows baseline-opposed targets across their linked occurrences. In the full benchmark, successes and eligible units are pooled across questions within each K level before taking the equal-weight mean.

## A.2 Target Violations within Unsuccessful Evaluation Units

To complement the analysis of sustained correction in Section 4.2, we report the mean number of instruction variants with a target violation within unsuccessful evaluation units at each K level. A variant has a target violation when its response violates at least one selected correction target present in that variant. Conditioning on unsuccessful units characterizes how broadly these violations extend across the relevant variants.

Let $\textstyle I _ { q , S } = \bigcup _ { t \in S } V ( q , t )$ be the variants covered by an eligible unit, and $S _ { i } = \{ t \in S : i \in V ( q , t ) \}$ Under the factorization in Eq. 9, the expected number of violating variants in that unit is

$$
\widehat { W } _ { q , S } = \sum _ { i \in I _ { q , S } } \left( 1 - \prod _ { t \in S _ { i } } p _ { i , t } \right) .\tag{11}
$$

Each variant contributes at most one violation count, even if several selected targets fail. The reported conditional mean is

$$
\widehat { C } ^ { ( K ) } = \frac { \sum _ { ( q , S ) \in { \mathcal U } _ { K } ^ { \mathrm { c o } } } \widehat { W } _ { q , S } } { \sum _ { ( q , S ) \in { \mathcal U } _ { K } ^ { \mathrm { c o } } } \left( 1 - \widehat { J } _ { q , S } \right) } .\tag{12}
$$

This ratio pools expected violation counts and expected unsuccessful units before division. With binary judgments, it reduces to the mean violation count among failed units. Units with larger $\vert I _ { q , S } \vert$ provide more opportunities for violations; Table A8 therefore also reports mean variant coverage. These counts describe failure extent within each model’s eligible population, alongside the correction rates in Table 2.

Correction rate and violation spread. At $K =$ 3, Qwen3.6-27B achieves a higher correction rate than DeepSeek-V4-Pro (30.66% versus 24.72%). Within unsuccessful units, however, it also has more variants containing a selected-target violation on average (3.35 versus 2.41; Appendix A.2, Table A2). The same pattern holds at K = 1, 2 and for Kimi K3 versus GPT-5.6 Sol. These counts describe violation spread within each model’s eligible units, complementing correction success rates. Table A8 reports their unit counts and mean variant coverage.

## A.3 Dual-Judge Adjudication and Selection Probabilities

GPT-5.4 Mini and Gemini 3.1 Flash-Lite independently judge each model response with temperature = 0 and the highest reasoning effort supported by each model. Each judge receives the full instruction, its three constraints, the corresponding acceptance criteria, and the response, with the evaluated model’s identity withheld. Judges assess each constraint separately and provide supporting textual evidence. Each constraint receives a binary judgment, with 1 indicating a pass and 0 otherwise, yielding a raw instruction score from 0 to 3.

## A.3.1 Stratified Adjudication for the Main Results

All manual adjudications were performed by the authors; no external annotators were employed.

Human adjudication during response judging is restricted to sampled response–constraint instances on which GPT-5.4 Mini and Gemini 3.1 Flash-Lite disagree. Each model’s 5,000 instructions yield 15,000 constraint judgments, for a total of 150,000 judgments across ten models. The two judges agree on 131,508 judgments (87.67%). The remaining 18,492 constitute the disagreement population. The main results use 6,654 adjudication records matched to the corresponding instructions, responses, and disagreement strata.

For disagreements, we construct strata within each evaluated model by disagreement direction, feature category, and constraint mode. The two directions distinguish cases in which GPT-5.4 Mini assigns a pass and Gemini 3.1 Flash-Lite assigns a failure from cases with the reverse judgments. Review allocation accounts for stratum size and covers every nonempty stratum. The review set includes matched adjudication records and withinstratum random samples. Each record must match the evaluated instruction, response, and stratum; sampling prioritizes strata with lower sampling fractions. Sample sizes follow a finite-population correction at a 95% confidence level, a conservative pass proportion of 0.5, and a target margin of 3 percentage points. These parameters guide review allocation rather than confidence intervals for TCPR or EMCR. Final matched sample counts are reported in Table A3.

For each sampled response–constraint instance, we manually compare both judges’ evidence against the original instruction, constraint definition, acceptance criteria, and model response. We select the better-supported judgment and record its binary label as the adjudicated outcome. In Table A3, agreement with GPT or Gemini means that the adjudicated label matches that judge’s decision; the percentage in parentheses uses the number of adjudicated disagreements for that model as its denominator. Agreed judgments retain their shared binary label. The main results use adjudicated binary labels for reviewed disagreements and stratum-specific estimated pass probabilities for the remaining disagreements. A stratum estimate averages the selected binary pass labels, rather than the frequency with which a particular judge is selected. Thus, selecting GPT contributes a pass in one disagreement direction and a failure in the other. Section 3.5 defines the corresponding stratum estimates and per-instance correction rule.

## A.3.2 Pooled Calibration for Further Evaluation

For the constraint-composition and prompting experiments, the fixed calibration pool contains 7,125 adjudicated disagreements: 6,689 favor GPT and 436 favor Gemini, yielding pooled weights of 93.88% and 6.12%. With binary judge labels $g , m \in \{ 0 , 1 \}$ , the calibrated pass probability is

<table><tr><td>Model</td><td>K = 1</td><td>K = 2</td><td>K = 3</td></tr><tr><td>Kimi K3</td><td>1.41</td><td>1.61</td><td>1.90</td></tr><tr><td>GPT-5.6 Sol</td><td>1.32</td><td>1.52</td><td>1.74</td></tr><tr><td>Gemini 3.5 Flash</td><td>1.42</td><td>1.68</td><td>1.98</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>1.35</td><td>1.55</td><td>1.91</td></tr><tr><td>GLM-5.2</td><td>1.58</td><td>1.91</td><td>2.31</td></tr><tr><td>DeepSeek-V4-Pro</td><td>1.48</td><td>1.85</td><td>2.41</td></tr><tr><td>Qwen3.6-27B</td><td>2.04</td><td>2.60</td><td>3.35</td></tr><tr><td>Claude Opus 4.6</td><td>1.73</td><td>2.17</td><td>2.70</td></tr><tr><td>Qwen3-235B-A22B-Thinking</td><td>1.83</td><td>2.42</td><td>3.28</td></tr><tr><td>Qwen3.5-9B</td><td>2.29</td><td>3.00</td><td>3.73</td></tr></table>

Table A2: Mean number of instruction variants with a target violation per unsuccessful evaluation unit at each K level.
<table><tr><td>Evaluated model</td><td>Dual-judge agreement</td><td>Disagreements</td><td>Adjudicated samples</td><td>Agrees with GPT count (%)</td><td>Agrees with Gemini count (%)</td></tr><tr><td>Kimi K3</td><td>91.74%</td><td>1,239</td><td>576</td><td>568 (98.61%)</td><td>8 (1.39%)</td></tr><tr><td>GPT-5.6 Sol</td><td>91.28%</td><td>1,308</td><td>588</td><td>491 (83.50%)</td><td>97 (16.50%)</td></tr><tr><td>Gemini 3.5 Flash</td><td>90.56%</td><td>1,416</td><td>609</td><td>563 (92.45%)</td><td>46 (7.55%)</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>90.83%</td><td>1,375</td><td>602</td><td>578 (96.01%)</td><td>24 (3.99%)</td></tr><tr><td>GLM-5.2</td><td>89.71%</td><td>1,544</td><td>632</td><td>598 (94.62%)</td><td>34 (5.38%)</td></tr><tr><td>DeepSeek-V4-Pro</td><td>86.60%</td><td>2,010</td><td>698</td><td>653 (93.55%)</td><td>45 (6.45%)</td></tr><tr><td>Qwen3.6-27B</td><td>85.31%</td><td>2,203</td><td>719</td><td>641 (89.15%)</td><td>78 (10.85%)</td></tr><tr><td>Claude Opus 4.6</td><td>84.23%</td><td>2,365</td><td>736</td><td>689 (93.61%)</td><td>47 (6.39%)</td></tr><tr><td>Qwen3-235B-A22B-Thinking</td><td>85.43%</td><td>2,185</td><td>718</td><td>717 (99.86%)</td><td>1 (0.14%)</td></tr><tr><td>Qwen3.5-9B</td><td>81.02%</td><td>2,847</td><td>776</td><td>750 (96.65%)</td><td>26 (3.35%)</td></tr><tr><td>Total</td><td>87.67%</td><td>18,492</td><td> $6 { , } 6 5 4$ </td><td>6,248 (93.90%)</td><td>406 (6.10%)</td></tr></table>

Table A3: Dual-judge agreement and human adjudication of sampled response–constraint disagreements for the ten models in the main evaluation. Agreement rates use 15,000 constraint judgments per model as the denominator. Adjudication percentages use the corresponding model’s reviewed sample count; the total row pools the sample counts. The fixed pool instead uses 582, 1,000, and 789 records for Kimi K3, GPT-5.6 Sol, and Claude Opus 4.6, respectively; the other seven models have the same adjudication counts.

$$
\widehat { p } ( y = 1 \mid g , m ) = \left\{ \begin{array} { l l } { { g , } } & { { g = m , } } \\ { { { 6 6 8 9 } } } \\ { { { 7 1 2 5 } ^ { g } + { \frac { 4 3 6 } { 7 1 2 5 } } m , } } & { { g \neq m . } } \end{array} \right.\tag{13}
$$

The pooled baseline TCPR is 64.43% for composition and 52.76% for prompting. These differ from Table 2, which uses model-specific strata and the review set in Table A3; auxiliary comparisons apply the fixed pooled calibration to both conditions.

## A.4 Feature-Frequency Correlations across Four Additional Models

We assess coverage using unconstrained responses from four additional models on the same 910 source questions. Open-ended extraction identifies formatting features before mapping them to the shared catalog. A valid instance describes observable response organization or presentation; records classified as not\_format are excluded. The mapping distinguishes exact matches, subtypes, composites, and uncovered features. Exact, subtype, and composite mappings count as covered, whereas uncovered instances remain in the denominator.

Table A4 provides the counts underlying Table 4. A composite instance contributes once to instance coverage even when mapped to several catalog features. The pooled coverage divides the total covered instances by the total valid instances, rather than averaging the four percentages. Inline code spans are a recurring uncovered feature in all four models. Such spans are distinct from the catalog’s block-level code containers.

For frequency comparisons, each mapped catalog feature is counted at most once per response.

<table><tr><td>Additional model</td><td>Extracted</td><td>Excluded</td><td>Valid</td><td>Covered</td><td>Uncovered</td><td>Coverage (%)</td></tr><tr><td>Gemini 3.1 Pro</td><td>9,085</td><td>42</td><td>9,043</td><td>8,595</td><td>448</td><td>95.05</td></tr><tr><td>DeepSeek-V3.2</td><td>8,967</td><td>48</td><td>8,919</td><td>8,565</td><td>354</td><td>96.03</td></tr><tr><td>gpt-oss-120b</td><td>11,601</td><td>24</td><td>11,577</td><td>11,177</td><td>400</td><td>96.54</td></tr><tr><td>Doubao-Seed-2.1-Pro</td><td>10,259</td><td>28</td><td>10,231</td><td>9,741</td><td>490</td><td>95.21</td></tr><tr><td>Pooled</td><td>39,912</td><td>142</td><td>39,770</td><td>38,078</td><td>1,692</td><td>95.75</td></tr></table>

Table A4: Raw counts for catalog instance coverage on four additional models, each evaluated on 910 source questions. Excluded instances are classified as non-format; all mapping-error counts are zero. Covered instances include exact, subtype, and composite mappings. Coverage is covered divided by valid instances.

This deduplication applies separately from instance coverage, which counts extracted feature occurrences. Construction-model annotations follow Appendix A.6; additional models retain their openextraction and mapping records.

Let $F _ { m } ( q )$ be the normalized set of catalog annotations for model m on question $q , \mathcal { Q } _ { \mathrm { c o m m o n } }$ the common set of 910 questions, $\mathcal { M } _ { \mathrm { r e f } }$ the ten construction models, and $\mathcal { M } _ { \mathrm { e x t r a } }$ the four additional models. For each catalog feature $^ { c , }$ we define its response-level annotation frequency and its mean frequency in each model group as

$$
\begin{array} { l } { { \displaystyle f _ { m } ( c ) = \frac { 1 } { 9 1 0 } \sum _ { q \in { \mathcal { Q } } _ { \mathrm { c o m m o n } } } 1 \{ c \in F _ { m } ( q ) \} } , }  \\ { { \displaystyle f _ { \mathrm { r e f } } ( c ) = \frac { 1 } { 1 0 } \sum _ { m \in { \mathcal { M } } _ { \mathrm { r e f } } } f _ { m } ( c ) } , } \\ { { \displaystyle f _ { \mathrm { e x t r a } } ( c ) = \frac { 1 } { 4 } \sum _ { m \in { \mathcal { M } } _ { \mathrm { e x t r a } } } f _ { m } ( c ) } . }  \end{array}\tag{14}
$$

Table 4 compares each additional model’s $f _ { m }$ with $f _ { \mathrm { r e f } } ;$ ; its pooled row compares $f _ { \mathrm { e x t r a } }$ with $f _ { \mathrm { r e f } }$ Spearman’s coefficient treats the 66 features as observations and compares their frequency rankings, assigning average ranks to ties. For the pooled coefficient, we average the four frequency vectors before ranking, rather than averaging their correlation coefficients. This comparison holds the question set, catalog, and response-level counting rules constant. These correlations describe annotated feature rankings under each model group’s extraction and mapping procedure.

## A.5 Complete Results by Task Type

We report TCPR and EMCR for the ten models by the task type of each source question. Task labels are inherited by all instruction variants of a source question, keeping each question and its variants in the same category. The seven task types cover all

910 source questions and 5,000 evaluation instructions. All results use the model-specific stratified calibration and baseline-opposed target definition used in Table 2.

## A.5.1 TCPR and EMCR by Task

The metrics in Section 3.4 are computed within each task, with EMCR weighting the three Klevel correction rates equally. Table A5 summarizes source-question and instruction counts, mean scores across the ten models, and the highest EMCR estimate for each task. Table A6 pairs TCPR and EMCR rows for each model. The mean rows weight all ten models equally, and bold values mark the highest point estimate for each metric and task.

## A.5.2 Task Sample Sizes and Eligible Evaluation Units

All models receive the same instructions, so TCPR denominators are identical within each task. Table A7 lists eligible target-set counts by model, task, and K level, using the model-specific baseline and eligibility conditions in Section 3.4.2.

To recover the overall results in Table 2, TCPR is weighted by the number of instructions in each task. For EMCR, expected successes and eligible unit counts are pooled across tasks within each K level before the three resulting correction rates are averaged with equal weight. The task-level means characterize performance under each task’s question and formatting-constraint configuration in this benchmark.

## A.6 Unconstrained Formatting Baselines and Correction Evaluation Units

The baseline annotation unit is a model–sourcequestion–catalog-feature tuple. Model and question identifiers, response-record identifiers, and text hashes link each annotation to its corresponding unconstrained response. Core-feature mappings and normalized open-ended annotations define each response’s catalog-feature set, with each feature recorded once. These baselines support the main evaluation, task analysis, constraint-composition grouping, and prompting experiment.

<table><tr><td>Task type</td><td>Source questions</td><td>Instruction variants</td><td>Mean TCPR</td><td>Mean EMCR</td><td>Highest EMCR (model: score)</td></tr><tr><td>Knowledge QA</td><td>163</td><td>754</td><td>73.54</td><td>56.59</td><td>Kimi K3: 73.63</td></tr><tr><td>Concept Explanation</td><td>105</td><td>559</td><td>73.22</td><td>56.00</td><td>GPT-5.6 Sol: 71.74</td></tr><tr><td>Information Extraction</td><td>93</td><td>362</td><td>73.00</td><td>59.59</td><td>GPT-5.6 Sol: 71.72</td></tr><tr><td>Decision Analysis</td><td>133</td><td>835</td><td>70.96</td><td>51.87</td><td>Qwen3.6-35B-A3B: 69.74</td></tr><tr><td>Solution Design</td><td>167</td><td>1008</td><td>68.10</td><td>50.23</td><td>Kimi K3: 65.72</td></tr><tr><td>Creative Writing</td><td>133</td><td>848</td><td>66.50</td><td>48.73</td><td>Kimi K3: 65.60</td></tr><tr><td>Code Generation</td><td>116</td><td>634</td><td>64.96</td><td>45.27</td><td>Gemini 3.5 Flash: 55.24</td></tr></table>

Table A5: Format following across seven task types (%). Mean TCPR and mean EMCR assign equal weight to the ten models. The final column gives the highest EMCR point estimate and the corresponding model for each task. The seven tasks contain 910 source questions and 5,000 instruction variants; each variant has three constraints.

Positive mappings connect 34 of the 37 core labels to 26 catalog features based on their definitions. Composite labels map to every catalog feature explicitly included: for example, an overview–elaboration–summary structure maps to both “overview followed by elaboration” and “summary and closure.” General inline enumeration and inline code fall outside the catalog. “Definition first” does not uniquely identify a catalog feature and has no direct automatic mapping.

Across the ten models, 9,100 unconstrained responses yield 91,332 positive catalog-feature annotations. These response-level feature sets define the model-specific formatting baselines used to identify correction targets.

Let $P _ { m , q }$ denote the set of catalog features annotated as present for model m on question q. Membership defines the operational generic feature state: forbid targets are baseline-opposed when their feature belongs to $P _ { m , q } ,$ , and require targets when it does not. Remaining targets are baseline-aligned. The complement of $P _ { m , q }$ denotes features not annotated as present. This feature-state grouping determines eligibility; instantiated acceptance criteria determine response compliance.

Table A8 lists each model’s positive annotations, eligible units, and mean number of covered variants. Eligibility is applied after the common instruction set is fixed. Differences between models therefore reflect their baseline annotations and resulting target sets, while their evaluated instructions remain identical.

The original and improved prompt configurations share the same Qwen3.5-9B unconstrainedresponse baseline and therefore have identical denominators at each K level. The constraintcomposition analysis likewise uses Qwen3.6-27B’s own baseline to distinguish baseline-opposed from baseline-aligned targets. Table A8 reports the overall eligible unit counts, and Table A7 breaks them down by source-question task type; both use the same baseline and target-set definitions.

## A.7 Benchmark Construction Details

Table A9 details the distribution of source questions across the ten domains and seven task types. Each source question contributes to one cell, regardless of its number of instruction variants.

## A.7.1 Source Provenance and Release

Table A10 attributes the 553 selected questions to five public datasets and records their stated licenses or intended uses. Inverse IFEval documents evaluation as an intended use without specifying a data license; ELI5’s repository BSD license covers software rather than the question data. We use these sources for academic research. The full 910- question pool contains 543 English and 367 Chinese questions. The anonymous artifact includes the selected original questions, their rewritten versions, and the resulting benchmark instructions. The applicable license and use terms of each source govern its source-derived content.

## A.7.2 Content Review and Ethical Considerations

We screened the released 910 source questions and 20,000 benchmark and single-constraint instructions with email-address, US/Chinese phonenumber, and targeted harmful-content patterns. No contact-pattern matches occurred. The single item flagged by the harmful-content patterns was retained after contextual review. The screening covered released question and instruction text, not model-generated responses.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Metric</td><td rowspan="2">Knowledge QA</td><td rowspan="2">Concept explanation</td><td rowspan="2">Information extraction</td><td rowspan="2">Decision analysis</td><td rowspan="2">Solution design</td><td rowspan="2">Creative writing</td><td rowspan="2">Code generation</td></tr><tr><td></td></tr><tr><td>Kimi K3</td><td>TCPR</td><td>87.69</td><td>83.43</td><td>84.76</td><td>84.05</td><td>79.95</td><td>83.01</td><td>78.06</td></tr><tr><td rowspan="2">GPT-5.6 Sol</td><td>EMCR</td><td>73.63</td><td>65.36</td><td>64.42</td><td>65.97</td><td>65.72</td><td>65.60</td><td>54.32</td></tr><tr><td>TCPR</td><td>85.92</td><td>85.93</td><td>87.37</td><td>84.91</td><td>80.12</td><td>80.86</td><td>71.40</td></tr><tr><td rowspan="2">Gemini 3.5 Flash</td><td>EMCR</td><td>69.20</td><td>71.74</td><td>71.72</td><td>59.22</td><td>63.87</td><td>60.20</td><td>50.66</td></tr><tr><td>TCPR</td><td>80.55</td><td>78.56</td><td>82.48</td><td>78.91</td><td>75.31</td><td>81.24</td><td>71.32</td></tr><tr><td rowspan="2">Qwen3.6-35B-A3B</td><td>EMCR</td><td>62.23</td><td>66.66</td><td>67.88</td><td>59.38</td><td>56.25</td><td>63.56</td><td>55.24</td></tr><tr><td>TCPR</td><td>79.10</td><td>82.00</td><td>82.74</td><td>80.12</td><td>74.81</td><td>77.66</td><td>71.75</td></tr><tr><td rowspan="2">GLM-5.2</td><td>EMCR</td><td>59.33</td><td>65.13</td><td>67.15</td><td>69.74</td><td>58.76</td><td>59.84</td><td>52.56</td></tr><tr><td>TCPR</td><td>79.24</td><td>81.01</td><td>77.09</td><td>75.65</td><td>77.30</td><td>72.71</td><td>68.01</td></tr><tr><td rowspan="2">DeepSeek-V4-Pro</td><td>EMCR</td><td>56.46</td><td>61.28</td><td>70.01</td><td>54.63</td><td>56.84</td><td>53.65</td><td>47.87</td></tr><tr><td>TCPR</td><td>71.83</td><td>68.95 51.97</td><td>69.03</td><td>70.24</td><td>62.82</td><td>64.67</td><td>59.10</td></tr><tr><td rowspan="2">Qwen3.6-27B</td><td>EMCR</td><td>51.68</td><td></td><td>56.91</td><td>46.72</td><td>45.59</td><td>39.85</td><td>35.54</td></tr><tr><td>TCPR</td><td>68.99</td><td>69.02 44.61</td><td>64.14 48.46</td><td>65.78</td><td>65.21</td><td>59.91</td><td>66.25</td></tr><tr><td rowspan="2">Claude Opus 4.6</td><td>EMCR</td><td>55.71</td><td>62.17</td><td></td><td>49.53</td><td>49.67</td><td>38.71</td><td>50.42</td></tr><tr><td>TCPR</td><td>65.89</td><td>39.32</td><td>66.81</td><td>58.67</td><td>57.02</td><td>59.10</td><td>55.63</td></tr><tr><td>Qwen3-235B-A22B</td><td>EMCR</td><td>49.81</td><td></td><td>58.22</td><td>40.79</td><td>38.15</td><td>42.23</td><td>37.31</td></tr><tr><td>Thinking</td><td>TCPR</td><td>58.12 43.52</td><td>64.95 45.72</td><td>71.03</td><td>58.47</td><td>57.95</td><td>40.85</td><td>53.48</td></tr><tr><td rowspan="3">Qwen3.5-9B</td><td>EMCR</td><td></td><td></td><td>50.12</td><td>32.75</td><td>31.31</td><td>23.75</td><td>31.52</td></tr><tr><td>TCPR</td><td>58.06</td><td>56.16 48.17</td><td>44.50 41.00</td><td>52.79 40.00</td><td>50.48 36.12</td><td>45.02</td><td>54.65</td></tr><tr><td>EMCR</td><td>44.38</td><td></td><td></td><td></td><td></td><td>39.88</td><td>37.25</td></tr><tr><td>Mean across</td><td>TCPR</td><td>73.54</td><td>73.22</td><td>73.00</td><td>70.96</td><td>68.10</td><td>66.50</td><td>64.96</td></tr><tr><td>ten models</td><td>EMCR</td><td>56.59</td><td>56.00</td><td>59.59</td><td>51.87</td><td>50.23</td><td>48.73</td><td>45.27</td></tr></table>

Table A6: Three-Constraint Pass Rate (TCPR, %) and Equal-Weight Multi-Constraint Correction Rate (EMCR, %) for ten models across seven task types. Bold indicates the highest value for each metric within each task type.

Formatting compliance does not imply factual correctness or safety, and these scores should not be used alone for deployment decisions.

## A.7.3 Feature Catalog and Constraint Templates

We obtain unconstrained responses to all 910 source questions from 10 LLMs (Section 4.1). GPT-5.4 annotates a predefined set of core formatting features and uses open-ended extraction to identify additional features. Normalization unifies semantically equivalent feature names and implementation variants that share constraint descriptions and acceptance criteria. The resulting unified catalog contains 66 formatting features across 8 categories. Section 4.5 evaluates catalog coverage on four additional models.

Each catalog feature has two complementary constraint templates: require, which specifies that the feature must appear, and forbid, which specifies that it must be absent. The catalog, templates, and corresponding acceptance criteria define a shared constraint space for constructing instructions and evaluating their responses. Table A12 lists all 66 features, their generic presence definitions, and require/forbid occurrence counts in the retained benchmark. Counts refer to the 15,000 constraint slots, not to observed model behavior. Both templates exist for every feature, although some modes do not occur in the retained set.

Generic presence definitions support baseline annotation. Instantiated constraints add the requested counts, field names, ordering, positions, and other acceptance conditions for a particular instruction. We retain the feature identity and mode together with the concrete requirement and its judge specification. A broad baseline feature can therefore be present even when the unconstrained response would fail a more specific require instruction; current eligibility uses the generic feature state.

## A.7.4 Instruction Generation and Target Links

Candidate construction explores alternative combinations of compatible features from three distinct categories for shared source questions. GPT-5.4 selects suitable features and require or forbid directions, then integrates the three constraints into the question to produce an instruction variant. Crosscategory composition diversifies the requirements while reducing semantic redundancy and potential conflicts within a category.

<table><tr><td>Model</td><td>Knowledge</td><td>Concept QA explanation</td><td>Information extraction</td><td>Decision analysis</td><td>Solution design</td><td>Creative writing</td><td>Code generation</td></tr><tr><td rowspan="3">Kimi K3</td><td>256</td><td>165</td><td>116</td><td>259</td><td>311</td><td>290</td><td>254</td></tr><tr><td>385</td><td>238</td><td>164</td><td>372</td><td>482</td><td>468</td><td>424</td></tr><tr><td>103</td><td>57</td><td>39</td><td>86</td><td>108</td><td>125</td><td>141</td></tr><tr><td rowspan="3">GPT-5.6 Sol</td><td>249</td><td>172</td><td>132</td><td>261</td><td>291</td><td>295</td><td>269</td></tr><tr><td>369</td><td>265</td><td>193</td><td>381</td><td>439</td><td>459</td><td>461</td></tr><tr><td>99</td><td>73</td><td>53</td><td>85</td><td>101</td><td>115</td><td>154</td></tr><tr><td rowspan="3">Gemini 3.5 Flash</td><td>211</td><td>144</td><td>104</td><td>238</td><td>282</td><td>299</td><td>253</td></tr><tr><td>284</td><td>201</td><td>152</td><td>318</td><td>410</td><td>488</td><td>405</td></tr><tr><td>67</td><td>46</td><td>36</td><td>62</td><td>83</td><td>134</td><td>135</td></tr><tr><td rowspan="3">Qwen3.6-35B-A3B</td><td>221</td><td>143</td><td>113</td><td>239</td><td>288</td><td>290</td><td>264</td></tr><tr><td>294</td><td>190</td><td>166</td><td>330</td><td>454</td><td>495</td><td>452</td></tr><tr><td>66 205</td><td>40</td><td>49</td><td>67</td><td>111</td><td>144</td><td>153</td></tr><tr><td rowspan="3">GLM-5.2</td><td></td><td>146</td><td>111</td><td>228</td><td>285</td><td>291</td><td>237 365</td></tr><tr><td>266</td><td>204 43</td><td>144 34</td><td>332 73</td><td>413 97</td><td>464 133</td><td>113</td></tr><tr><td>56 233</td><td>154</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">DeepSeek-V4-Pro</td><td></td><td></td><td>126</td><td>244</td><td>277</td><td>302</td><td>265</td></tr><tr><td>314 71</td><td>221 50</td><td>187 54</td><td>349 79</td><td>407 86</td><td>528 161</td><td>440 141</td></tr><tr><td>229</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">Qwen3.6-27B</td><td></td><td>158</td><td>104</td><td>254</td><td>302</td><td>289</td><td>238</td></tr><tr><td>324</td><td>227</td><td>129</td><td>365</td><td>468</td><td>479</td><td>382</td></tr><tr><td>65</td><td>63</td><td>27</td><td>92</td><td>107</td><td>134</td><td>113</td></tr><tr><td rowspan="3">Claude Opus 4.6</td><td>250</td><td>156</td><td>112</td><td>262</td><td>310</td><td>287</td><td>241</td></tr><tr><td>365</td><td>219</td><td>152</td><td>393</td><td>494</td><td>436</td><td>388</td></tr><tr><td>94</td><td>49</td><td>37</td><td>87</td><td>110</td><td>102</td><td>120</td></tr><tr><td rowspan="3">Qwen3-235B-A22B-Thinking</td><td>221</td><td>144</td><td>103</td><td>226</td><td>267</td><td>292</td><td>214</td></tr><tr><td>279</td><td>189</td><td>137</td><td>338</td><td>414</td><td>494</td><td>307</td></tr><tr><td>59</td><td>47</td><td>28</td><td>77</td><td>90</td><td>129</td><td>82</td></tr><tr><td rowspan="2">Qwen3.5-9B</td><td>205</td><td>139</td><td>100</td><td>217</td><td>271</td><td>282</td><td>254</td></tr><tr><td>276</td><td>187</td><td>128</td><td>294</td><td>396</td><td>460</td><td>434</td></tr><tr><td></td><td>58</td><td>37</td><td>28</td><td>66</td><td>92</td><td>123</td><td>143</td></tr></table>

Table A7: Eligible correction evaluation units by model and task. Read each cell from top to bottom: it lists the numbers of eligible target sets at $K = 1 / K = 2 / K = 3$ , which form the denominators of the corresponding correction rates.

Different combinations can reuse a feature in the same mode, creating recurring targets for a source question. After screening, we group all retained target occurrences by source question, feature, and mode to establish each target’s variant coverage. Each model’s baseline annotations identify which targets require correction relative to its unconstrained response (Appendix A.6). Section 3.4.2 defines eligibility for individual targets and target sets using this coverage.

The 5,000 retained variants contain 6,025 distinct question–feature–mode groups, of which 3,017 occur at least twice (Table A11). A singleoccurrence target is excluded at K = 1, but may participate at K = 2 or K = 3 when the selected targets co-occur and their union covers multiple variants. A question with only one retained variant contributes to TCPR but cannot contribute an eligible EMCR unit.

For example, question 1164 asks for ideas for a mother’s birthday party. Variants H-0008, H-1324, and H-1325 all require a repeated section pattern and a closing next step. H-0008 and H-1325 forbid tables, whereas H-1324 instead forbids visual symbols. Accordingly, the repeated-section target covers all three variants, while the table prohibition covers two. If both are baseline-opposed for a model, their joint unit covers three variants and requires both selected targets to pass wherever each appears.

This linkage holds the source question, feature identity, and mode fixed; concrete parameters can vary. In the example, all three variants request three plans, but H-1325 specifies four fields per plan while the others specify three. Each response is judged against its own instantiated requirement. The resulting grouping evaluates correction across realizations of the same formatting target, rather than requiring identical parameter values across variants.

<table><tr><td>Model</td><td>Positive annotations</td><td> $K = 1$ </td><td> $K = 2$ </td><td> $K = 3$ </td></tr><tr><td>Kimi K3</td><td>8327</td><td>1,651 (3.99)</td><td>2,533 (6.66)</td><td>659 (8.92)</td></tr><tr><td>GPT-5.6 Sol</td><td>7559</td><td>1,669 (4.00)</td><td>2,567 (6.67)</td><td>680 (9.08)</td></tr><tr><td>Gemini 3.5 Flash</td><td>10970</td><td>1,531 (3.91)</td><td>2,258 (6.49)</td><td>563 (8.51)</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>11642</td><td>1,558 (3.90)</td><td>2,381 (6.46)</td><td>630 (8.60)</td></tr><tr><td>GLM-5.2</td><td>10155</td><td>1,503 (3.88)</td><td>2,188 (6.38)</td><td>549 (8.40)</td></tr><tr><td>DeepSeek-V4-Pro</td><td>9052</td><td>1,601 (3.98)</td><td>2,446 (6.67)</td><td>642 (9.18)</td></tr><tr><td>Qwen3.6-27B</td><td>7557</td><td>1,574 (4.06)</td><td>2,374 (6.96)</td><td>601 (9.58)</td></tr><tr><td>Claude Opus 4.6</td><td>7382</td><td>1,618 (4.01)</td><td>2,447 (6.70)</td><td>599 (9.06)</td></tr><tr><td>Qwen3-235B-A22B-Thinking</td><td>7758</td><td>1,467 (4.05)</td><td>2,158 (6.85)</td><td>512 (9.31)</td></tr><tr><td>Qwen3.5-9B</td><td>10930</td><td>1,468 (3.92)</td><td>2,175 (6.59)</td><td>547 (8.93)</td></tr></table>

Table A8: Baseline annotations and correction evaluation units. Each model has 910 unconstrained responses. K-level cells give the eligible unit count followed by mean variant coverage $| I _ { q , S } |$ in parentheses. Each unit count is the denominator of its overall correction rate. Coverage counts a variant once even if several selected targets occur in it.

![](images/0f1b434c8d2a4f8ab3796cd59da685933e6fefbf4e44d730e3162929fb032144.jpg)  
Table A9: Source-question counts by domain and task. Each of the 910 questions is counted once; zeros indicate empty cells. Domains are ordered by total count. Task order and abbreviations follow Figure 3.

## A.7.5 Validation and Difficulty Screening

During generation, GPT-5.4 checks task preservation, inclusion of every constraint, and compatibility of the three constraints. Automated checks validate constraint parameters, metadata mappings, and instruction structure. Each retained record links its source question and final instruction to three feature identifiers, their categories and modes, concrete constraints, and acceptance criteria. These fields support separate checks of instruction composition and response compliance.

<table><tr><td>Dataset</td><td>Questions</td><td>License / use</td></tr><tr><td>WildBench</td><td>217</td><td>CC BY 4.0</td></tr><tr><td>ThoughtTrace</td><td>186</td><td>CC BY 4.0</td></tr><tr><td>Inverse IFEval</td><td>71</td><td>Evaluation use</td></tr><tr><td>ELI5</td><td>33</td><td>Unstated</td></tr><tr><td>ExpertQA</td><td>46</td><td>MIT</td></tr></table>

Table A10: Sources of the 553 retained public-source questions. Licenses are those stated on the linked dataset pages; “Evaluation use” is a documented intended use rather than a license, and “Unstated” denotes no specified data license.

We rank candidate instructions by the mean calibrated probability of satisfying all three constraints across the 10 screening models, using the responsejudging and calibration procedure in Section 3.5 and Appendix A.3. We discard the highest-scoring candidates and retain a fixed set of 5,000 instruction variants, with score ties broken by instruction identifier. All evaluated models receive this same instruction set.

The retained set undergoes independent satisfiability review by Gemini 3.1 Flash-Lite, using a concrete witness answer and requirement-level verification for each instruction. Cases requiring clarification are reassessed after the acceptance criteria or candidate answers are supplemented. All 5,000 retained instructions have a witness answer and verification record. The witness provides a concrete candidate satisfying the joint requirements; requirement-level verification checks it against each constraint. This review concerns the feasibility of the instruction and does not replace scoring the evaluated models’ responses. These checks assess instruction validity, while difficulty screening selects challenging combinations. Figure 2 summarizes the construction and evaluation pipeline.

<table><tr><td>Variants</td><td>Questions</td><td>Target groups</td></tr><tr><td>1</td><td>256</td><td>3,008</td></tr><tr><td>2</td><td>147</td><td>1,139</td></tr><tr><td>3</td><td>98</td><td>683</td></tr><tr><td>4-5</td><td>119</td><td>610</td></tr><tr><td>6-10</td><td>152</td><td>466</td></tr><tr><td>&gt; 10</td><td>138</td><td>119</td></tr><tr><td>Total</td><td>910</td><td>6,025</td></tr></table>

Table A11: Variant multiplicity in the shared instruction set before applying model-specific baseline eligibility. The second column counts questions with that many retained variants; the third counts targets appearing in that many variants. Of 6,025 targets, 3,017 recur. The maximum is 30 variants per question and 24 per target.

## A.8 Paired Composition and Prompting Details

Paired single-constraint evaluation. The three single-constraint versions of each of 5,000 instructions yield 15,000 paired evaluation records covering 14,908 distinct prompt strings. Repeated prompts remain in their respective pairs, preserving correspondence with the original three-constraint instruction. The separate-response estimate multiplies the three calibrated pass probabilities within each pair group before averaging across the 5,000 groups. Specifically, let $p _ { i , j } ^ { \mathrm { s i n g l e } }$ be the calibrated pass probability for constraint j when answered separately, and $p _ { i , j } ^ { \mathrm { j o i n t } }$ its counterpart in the combined instruction. The two estimates are

$$
\begin{array} { c } { { \displaystyle { \widehat P ^ { ( a ) } = \frac { 1 } { 5 , 0 0 0 } \sum _ { i = 1 } ^ { 5 , 0 0 0 } \prod _ { j = 1 } ^ { 3 } p _ { i , j } ^ { ( a ) } , } } } \\ { { \displaystyle { a \in \{ \mathrm { s i n g l e } , \mathrm { j o i n t } \} . } } } \end{array}\tag{15}
$$

The single estimate uses three separate responses per group; the joint estimate evaluates all three constraints on one response. Pairing preserves the task and selected constraints. Both use pooled calibration (Appendix A.3.2).

Prompting configuration. The fixed metaprompt, appended to the user message, asks the model to identify every explicit constraint, construct the minimal output structure satisfying them, avoid unrequested content, and check and revise the response against each constraint before producing the final answer. Both conditions use the fixed pooled calibration in Appendix A.3.2; the shared unconstrained-response baseline and eligible unit counts are specified in Appendix A.6.

## A.9 Difficult Constraint Combinations

These qualitative cases illustrate how local formatting, document structure, and response boundaries interact within a response.

Section headings × Atomic steps × No call to action. Headings, atomic steps, and a closing without a follow-up invitation jointly require separating explanatory content from executable steps while keeping every step within the prescribed action budget. Section introductions can introduce extra instructions, and compressing a workflow beneath fixed headings can merge several actions into one numbered item. In an inventory-management response, a step asks the reader to check a list and then submit an order, although each step permits only one action. The visible heading and numbering structure is preserved, but the action-level requirement is broken.

Repeated sections × Quotation and parenthesis markers × No call to action. Repeated templates, quotation and parenthesis markers, and a declarative closing require distributing a global marker budget across repeated fields while controlling the final field’s rhetorical function. In an organizational-plan response, all three sections retain their required fields, but the requested quotation and parenthesis pairs are absent and the final sentence begins with “Conduct quarterly compliance workshops.” Completing the template therefore does not ensure either marker coverage or an acceptable closing.

Table format × Comparison dimensions × No call to action. A table, a comparison along fixed dimensions, and a non-directive closing require fitting the concluding statement inside a restricted table while retaining both valid syntax and comparable cell content. A closing sentence placed in the final cell can displace the requested comparison or add material beyond its specified dimensions. A mysticism comparison illustrates another failure: it supplies the named columns, object rows, and a descriptive final sentence, but omits the Markdown header-separator row. Recognizable comparison content and an acceptable closing therefore do not establish valid table syntax.

<table><tr><td></td><td>Feature</td><td>Generic presence definition</td><td>Req.</td><td>Forb.</td></tr><tr><td></td><td>Bilingual parallel layout</td><td>Aligns corresponding source and translated text.</td><td>52</td><td>9 2</td></tr><tr><td></td><td>Column alignment</td><td>Organizes content in parallel columns or aligned rows.</td><td></td><td></td></tr><tr><td>ayout</td><td>Dialogue turns</td><td>Separates speakers or dialogue turns onto distinct lines.</td><td>219</td><td>83</td></tr><tr><td></td><td>Line breaks and spacing</td><td>Uses line breaks, blank lines, or compact spacing.</td><td>1,110</td><td>63</td></tr><tr><td>ext and</td><td>Paragraph or sentence count</td><td>Explicitly controls the number of paragraphs or sentences.</td><td>22 3</td><td>0</td></tr><tr><td>Arrhh</td><td>Poetry lines and stanzas</td><td>Groups verse into lines or stanzas.</td><td></td><td>3</td></tr><tr><td></td><td>Single text block</td><td>Restricts the response to one paragraph, sentence, or block.</td><td></td><td></td></tr><tr><td></td><td>Appendix section</td><td>Adds an appendix, supplementary section, or afterword.</td><td>64</td><td>27</td></tr><tr><td></td><td>Document template</td><td>Organizes the response as a reusable outline or template.</td><td></td><td>0</td></tr><tr><td></td><td>Document title</td><td>Places a standaloñe document title at the beginning.</td><td>861</td><td>55</td></tr><tr><td>hierhy</td><td>Front matter</td><td>Places an abstract, metadata, or introductory block before the body.</td><td></td><td>3</td></tr><tr><td></td><td>Multipart framework</td><td>Organizes parallel dimensions into a fixed set of sections.</td><td></td><td>2</td></tr><tr><td></td><td>Repeated section pattern</td><td>Repeats thè same heading or paragraph structure across sections.</td><td>18 513</td><td>14</td></tr><tr><td>A2::t es.</td><td>Section divider</td><td>Separates content with rules, page breaks, or standalone markers.</td><td>126</td><td>14</td></tr><tr><td></td><td>Section headings</td><td>Divides the response using headings or section labels.</td><td>666</td><td>82</td></tr><tr><td></td><td>Title and subtitle</td><td>Combines a main title with a subtitle or explanatory line.</td><td>142</td><td>11</td></tr><tr><td>A:s Ls nd</td><td></td><td></td><td>242</td><td>135</td></tr><tr><td></td><td>Bullet list</td><td>Lists items using unordered bullet markers.</td><td>182</td><td>107</td></tr><tr><td></td><td>Checklist</td><td>Presents items with checkboxes or checkable markers.</td><td>247</td><td>49</td></tr><tr><td></td><td>Nested list</td><td>Uses at least two levels of list items.</td><td></td><td></td></tr><tr><td></td><td>Numbered enumeration</td><td>Enumerates items with numbers, letters, or other ordered labels.</td><td>456</td><td>96</td></tr><tr><td></td><td>Fixed schema</td><td>Fills predefined fields, slots, or columns.</td><td>314</td><td>0</td></tr><tr><td></td><td>Form or record card</td><td>Uses a form, record card, or fillable template.</td><td></td><td>46 3</td></tr><tr><td></td><td>Glossary entries</td><td>Presents terms paired with definitions or explanation fields.</td><td></td><td></td></tr><tr><td>sutd unon A4ba :. el nd</td><td>Key-value fields</td><td>Organizes information into named fields and their values.</td><td></td><td>1</td></tr><tr><td></td><td>Interface panel</td><td>Uses status areas, controls, or system-information panels.</td><td></td><td>0</td></tr><tr><td></td><td>Table or grid</td><td>Aligns information in a table, matrix, or grid.</td><td>312</td><td>176</td></tr><tr><td></td><td>Answer first</td><td>Gives the answer or main conclusion before elaboration.</td><td></td><td>9</td></tr><tr><td></td><td>Call to action</td><td>Ends with a follow-up invitation, next step, or action cue.</td><td>193</td><td>377</td></tr><tr><td>Ar i iey</td><td>Code and explanation</td><td>Organizes code and its explanation into corresponding parts.</td><td>302</td><td>44</td></tr><tr><td></td><td>Formal report</td><td>Uses the overall structure of a report, memo, or proposal.</td><td></td><td>0</td></tr><tr><td></td><td>Framing text</td><td>Wraps the core content in a preface, note, or closing frame.</td><td></td><td>6</td></tr><tr><td></td><td>Letter or email</td><td>Uses a salutation, body, closing, and signature structure.</td><td></td><td>11</td></tr><tr><td>anoso cure</td><td>Opening lead</td><td>Starts with an introduction, background, or scene-setting text.</td><td></td><td>44</td></tr><tr><td></td><td>Overview then detail</td><td>Presents an overview before expanding individual parts.</td><td>359</td><td>41</td></tr><tr><td></td><td>Schedule or itinerary</td><td>Organizes a plan by date, time, stage, or location.</td><td></td><td>0</td></tr><tr><td></td><td>Script or narrative</td><td>Uses scripts, scenes, storyboards, narration, or a finished story.</td><td>227</td><td>60</td></tr><tr><td></td><td>Summary and closure</td><td>Ends with a summary, conclusion, recap, or closing statement.</td><td>255</td><td>241</td></tr><tr><td></td><td>Alternative options</td><td>Offers multiple versions, options, paths, or alternatives.</td><td>36</td><td>28</td></tr><tr><td></td><td>Cause and effect</td><td>Explicitly organizes causes, mechânisms, outcomes, or impacts.</td><td>327</td><td>53</td></tr><tr><td>rlaol olaton</td><td>Classification</td><td>Groups content by type, dimension, or level.</td><td>20</td><td>2</td></tr><tr><td></td><td>Comparison and contrast</td><td>Compares objects, options, advantages, or opposing cases.</td><td>335</td><td>56</td></tr><tr><td></td><td>Conditional branches</td><td>Uses if-then structures or situation-dependent branches.</td><td></td><td>13</td></tr><tr><td></td><td></td><td>Explains a point through examples, cases, analogies, or demonstrations.</td><td>44</td><td></td></tr><tr><td></td><td>Examples and cases</td><td></td><td>317 23</td><td>121</td></tr><tr><td></td><td>Question and answer</td><td>Uses question-answer pairs, FAQs, or self-posed questions.</td><td></td><td>0</td></tr><tr><td>Aa éeon and</td><td>Ranking and priority</td><td>Organizes items by rank, priority, level, or score.</td><td></td><td>18</td></tr><tr><td></td><td>Rationale and evidence</td><td>Explicitly links conclusions to reasons or evidence.</td><td></td><td>4</td></tr><tr><td></td><td>Recommendations</td><td>Organizes the response as advice, action items, or executable measures.</td><td></td><td>15</td></tr><tr><td></td><td>Steps and processes</td><td>Describes task completion through ordered steps or stages.</td><td>387</td><td>72</td></tr><tr><td></td><td>Timeline and stages</td><td>Organizes information by time, stages, or temporal progression.</td><td>332</td><td>41</td></tr><tr><td></td><td>Arrows and separators</td><td>Uses arrows, connectors, or special relation/separation markers.</td><td>243</td><td>98</td></tr><tr><td>anel  es &#x27;utton</td><td>Typographic emphasis</td><td>Emphasizes text with bold, italic, underline, or other font changes.</td><td>250</td><td>183</td></tr><tr><td></td><td>Labels and tags</td><td>Marks content attributes with short labels, status words, or prefixes.</td><td>300</td><td>181</td></tr><tr><td></td><td>Quotes and parentheses</td><td>Highlights terms or additions with quotation marks or parentheses.</td><td>406</td><td>286</td></tr><tr><td></td><td>Reader address</td><td>Addresses the reader directly or uses the second person.</td><td></td><td>37</td></tr><tr><td>ALpasii i.</td><td>Roles and interaction</td><td>Specifies speaking roles, interactive actions, questions, or response cues.</td><td></td><td>2</td></tr><tr><td></td><td>Signature and sign-off</td><td>Adds a signature, attribution, salutation, or sign-off at the end.</td><td></td><td>8</td></tr><tr><td></td><td>Visual symbols</td><td>Uses emoji, icons, checkmarks, stars, or other visual markers</td><td>178</td><td>305</td></tr><tr><td></td><td>Callout or note block</td><td>Separates supplementary information into notices, warnings, or notes.</td><td>227</td><td>75</td></tr><tr><td></td><td>Captioned visual</td><td>Plâces a short caption immediately after visual content.</td><td></td><td>0</td></tr><tr><td></td><td>Citations and links</td><td>Uses quotations, footnotes, references, sources, or links</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>22</td></tr><tr><td></td><td>Code annotations</td><td>Explains code with comments, docstrings, or documentation comments.</td><td>233</td><td>28</td></tr><tr><td></td><td>Code block</td><td>Presents code in fenced or standalone code blocks.</td><td>70</td><td>10</td></tr><tr><td></td><td>Textual diagrams</td><td>Uses flowcharts, trees, ASCII diagrams, or other text-based visuals.</td><td>152</td><td>65</td></tr><tr><td>Ae Tenal anad</td><td>Files and modules</td><td>Separates implementations and explanations by file or module name.</td><td></td><td>1</td></tr><tr><td></td><td>Machine-readable block</td><td>Uses JSON, YAML, XML, CSV, or other machine-readable structures.</td><td></td><td></td></tr><tr><td>emq ceors</td><td>Mathematical notation</td><td>Uses formulas, equations, mathematical symbols, or typesetting.</td><td></td><td>1 39</td></tr></table>

Table A12: Complete formatting catalog of 66 features in eight categories. Names and definitions are English translations of the catalog. Req. and Forb. count require and forbid occurrences in the retained 5,000 instructions; these counts describe instruction composition rather than model-specific correction eligibility.

## A.10 Model Configurations

Table A13 reports the generation settings for Qwen3.6-27B, Qwen3-235B-A22B-Thinking, and Qwen3.5-9B, including sampling parameters, reasoning controls, and explicitly configured token budgets. Parameters not specified in a request are handled by the inference service.

Generations reaching a configured output limit are truncated, including those involving repetitive loops. Returned responses are evaluated using the same constraint-level criteria as other responses.

<table><tr><td>Model</td><td>Temperature</td><td>Top-p</td><td>Top-k</td><td>Reasoning control</td><td>Output limit (tokens)</td><td>Thinking budget (tokens)</td></tr><tr><td>Qwen3.6-27B</td><td>0</td><td>1</td><td>0</td><td>Enabled</td><td>65,536</td><td>32,768</td></tr><tr><td>Qwen3-235B-A22B-Thinking</td><td>0</td><td></td><td></td><td>high</td><td>16,384</td><td></td></tr><tr><td>Qwen3.5-9B</td><td>0</td><td>1</td><td>0</td><td>Enabled</td><td>65,536</td><td>32,768</td></tr></table>

Table A13: Generation settings for Qwen3.6-27B, Qwen3-235B-A22B-Thinking, and Qwen3.5-9B. Each row reports the predominant client-side configuration. high denotes the requested reasoning-effort level; “Enabled” denotes thinking mode with the specified token budget. Blank cells indicate parameters not explicitly set in the request. Top-k = 0 disables top-k filtering. Each input is submitted as one user message without an additional system prompt, and one response is retained per benchmark item.
<table><tr><td>Item</td><td>Setting</td></tr><tr><td>Model scale</td><td>Public model identifiers/cards report Qwen3.6-27B (27B), Qwen3.6-35B-A3B (35B total, 3B active), Qwen3-235B-A22B-Thinking (235B/22B), Qwen3.5-9B (9B), and gpt-oss-120b (116.8B/5.1B).</td></tr><tr><td>Client requests</td><td>One user message per item, no additional system prompt; one final response retained. Explicit Qwen sampling and token settings are in Table A13; unspecified parameters follow the serving implementation.</td></tr><tr><td>Final-response workload</td><td>50,000 main, 9,100 unconstrained-baseline, 3,640 coverage, 15,000 single-constraint, and 5,000 prompting generations (82,740 in total).</td></tr><tr><td>Hardware</td><td>Locally deployed models used RTX 4080 SUPER and RTX PRO 6000 GPUs.</td></tr><tr><td>Software</td><td>The released inference and feature-extraction runners require Python 3.10+ and use only the Python standard library.</td></tr></table>

Table A14: Reproduction settings and final answer-generation workload.

## A.11 Worked Examples and Experimental Prompts

Two examples reproduce complete instructions, responses, and translated judge rationales: one shows agreed pass labels and the other shows a stratum estimate for an unreviewed conflict. The complementary EMCR calculation appears in Appendix A.1. Response-line numbers identify original source lines; visual wrapping adds no source lines. The calibrated probabilities $p _ { j }$ use the recorded adjudications and stratum estimates in Section 3.5.

The prompt figures give English translations of the five Chinese prompts. Examples and labels are translated; variables and placeholders are preserved. JSON whitespace is reformatted for readability, and long prompts have numbered parts.

## Example 1.

Agreement preserves a passing response

Evaluated model: Gemini 3.5 Flash

Instruction ID: H-1055

## Instruction

Work it out from the log dates. Keep it neat: write exactly 3 non-empty lines with no blank lines. Instead of a paragraph, make the first 2 lines bullet points, one per line, using the same unordered bullet on both lines. You may want to give the date first, but do not begin with a summary or the final date; put the inspection date only on the third line. A ship’s log notes that the vessel left Boston on 3 May, reached Halifax on 6 May, departed again on 8 May, and arrived in Liverpool on 18 May. A cargo inspection occurred two days after arrival. On what date did the inspection take place?

## Model Output

<sub>1</sub> \* 3 May: The vessel departed Boston.

<sub>2</sub> \* 18 May: The vessel arrived in Liverpool.

<sub>3</sub> 20 May

## Independent Judgments

GPT-5.4 Mini — Pass (1): The response has exactly three lines and no blank lines, satisfying the requirement.

Gemini 3.1 Flash-Lite — Pass (1): There are three non-empty lines and no blank lines, so the requirement is satisfied.

C2. Bullet list (require): the first two lines use the same unordered bullet, one item per line.

GPT-5.4 Mini — Pass (1): Both of the first two lines begin with the same unordered bullet, \*, with one item on each line.

Gemini 3.1 Flash-Lite — Pass (1): The first two lines use the same unordered bullet format (\*), satisfying the requirement.

C3. Answer position (forbid): no opening summary or final date; the inspection date appears only on line three.

GPT-5.4 Mini — Pass (1): The response does not begin with a summary or the final date; the first bullet gives 3 May. The date 20 May appears only on the third line.

Gemini 3.1 Flash-Lite — Pass (1): The opening contains neither a summary nor the final date, and the inspection date (20 May) appears only on the third line.

## Adjudication and Calibration

Both judges assign a pass to all three constraints. The agreement branch retains each shared binary judgment; no disagreement adjudication is needed for this instruction.

## Instruction-Level Result

(p<sub>1</sub>, p<sub>2</sub>, p<sub>3</sub>) = (1, 1, 1), P<sub>i</sub> = <sup>3</sup><sub>j=1</sub> p<sub>j</sub> = 1.

All three constraints contribute a calibrated pass value of 1. This instruction contributes $\widehat { P } _ { i }$ to the numerator of the TCPR estimate.

Figure 5: Example 1. Agreement preserves a passing response. Complete instruction, model response, independent judgments, and calibrated result.

## Example 2.

An unreviewed conflict receives a stratum estimate

Evaluated model: GPT-5.6 Sol

Instruction ID: H-4161

## Instruction

An invoice lists 75 monitors at \$240 each for a total of \$18,000. The purchase order approved 70 monitors, and the warehouse received 70. Assuming the unit price is correct, what discrepancy should the accounts team investigate? Write it as a quick reconciliation note: use exactly two headings, Facts and Issue, each on its own line; put exactly one numbered line under each heading, using only 1. and 2.; and use exactly one double-quoted phrase and exactly one parenthetical note in the whole answer.

## Model Output

<sub>1</sub> Facts

<sub>2</sub> 1. Invoice bills 75 monitors at \$240 each (\$18,000), while the PO and warehouse receipt both show 70.

<sub>3</sub> Issue

<sub>4</sub> 2. Investigate the "extra 5 monitors" billed, representing a \$1,200 overcharge (75 minus 70).

## Independent Judgments

C1. Headings (require): Facts and Issue, on separate lines and in that order, with no additional heading.

GPT-5.4 Mini — Pass (1): Facts and Issue each occupy a separate line in the correct order, with no additional title.

Gemini 3.1 Flash-Lite — Pass (1): The two headings are Facts and Issue, each on its own line, in the required order, with no extra heading.

C2. Enumeration (require): only lines 1. and 2., directly under Facts and Issue respectively.

GPT-5.4 Mini — Pass (1): The body consists only of two numbered entries, 1. and 2., beneath their corresponding headings; there is no additional numbered or unnumbered body text.

Gemini 3.1 Flash-Lite — Pass (1): The response uses 1. beneath Facts and 2. beneath Issue as its only body content, satisfying the requirement.

C3. Local markers (require): exactly one double-quoted phrase and one parenthetical note in the entire response.

GPT-5.4 Mini — Fail (0): There is one double-quoted phrase, "extra 5 monitors", but two parenthetical notes, (\$18,000) and (75 minus 70), violating the requirement for exactly one.

Gemini 3.1 Flash-Lite — Pass (1): Identifies "extra 5 monitors" and (75 minus 70), counts one quoted phrase and one parenthetical note, and assigns a pass.

## Adjudication and Calibration

C1 and C2 retain their agreed pass labels. C3 was not selected for adjudication in the main evaluation. Its stratum is defined by evaluated model GPT-5.6 Sol, disagreement direction GPT=0/Gemini=1, category A7 (Emphasis, punctuation, and local markers), and require mode. Of the 38 reviewed disagreements in this stratum, 7 have an adjudicated pass label.

$$
p _ { 3 } = \widehat { \pi } _ { h } = \frac { 7 } { 3 8 } \approx 0 . 1 8 4 2 .
$$

## Instruction-Level Result

(p<sub>1</sub>, p<sub>2</sub>, p<sub>3</sub>) = (1, 1, 7/38),

$$
\begin{array} { r } { \widehat { P } _ { i } = \prod _ { j = 1 } ^ { 3 } p _ { j } = \frac { 7 } { 3 8 } \approx 0 . 1 8 4 2 . } \end{array}
$$

The value 7/38 is a stratum-level estimated pass probability, not an adjudicated binary label for this response. This instruction contributes $\widehat { P } _ { i }$ to the numerator of the TCPR estimate.

Figure 6: Example 2. An unreviewed conflict receives a stratum estimate. Complete instruction, model response, independent judgments, and calibrated result.

![](images/60dee253296f3fccdfcaad7bd84c8c620fbf2cf2112addf4fd4ba9a98129d938.jpg)  
Figure 7: Prompt 1: Core-catalog and open feature extraction (part 1 of 3). System prompt for annotating format features in unconstrained responses. English translation.

![](images/6190b8775b4718c7aab51aceb3775171f39812bc5c9fb3cd6e95fbdbca559455.jpg)  
Figure 8: Prompt 1: Core-catalog and open feature extraction (part 2 of 3). System prompt for annotating format features in unconstrained responses. English translation.

Prompt 1: Core-catalog and open feature extraction (part 3 of 3)   
[primary\_format]   
primary\_format is only a coarse summary of the overall appearance; it does not restrict detecting   
multiple features:   
- plain\_prose   
- headings\_and\_sections   
- bullet\_list   
- numbered\_steps   
- table   
- code\_centric   
- mixed   
- other   
Choose headings\_and\_sections when headings dominate the response; choose mixed when several   
formats each occupy substantial space and none dominates. Numbered headings alone do not constitute   
numbered\_steps.   
[Output requirements]   
Return only strictly valid JSON, without explanations, analysis, or Markdown code fences.   
{   
"question\_id": "Copy the input value exactly",   
"primary\_format": "plain\_prose",   
"detected\_feature\_ids": ["feature\_id"],   
"additional\_feature\_ids": ["additional\_feature\_id"],   
"all\_detected\_feature\_ids": ["feature\_id"],   
"feature\_evidence": {   
"feature\_id": ["Verbatim text from candidate\_answer"]   
},   
"feature\_confidence": {"feature\_id": "high"},   
"additional\_features": [   
{   
"feature\_id": "stable\_snake\_case\_id",   
"name": "Chinese name",   
"definition": "A concise, stable, reusable definition in Chinese",   
"category": "rhetorical\_structure",   
"evidence": ["Verbatim text from candidate\_answer"],   
"confidence": "medium"   
}   
],   
"notes": ""   
}   
[Field consistency]   
1. Copy question\_id exactly.   
2. additional\_feature\_ids must exactly match the feature\_id values in additional\_features.   
3. all\_detected\_feature\_ids is the deduplicated union of detected\_feature\_ids and additional\_feature\_ids.   
4. feature\_evidence and feature\_confidence must cover every detected\_feature\_id in detected\_feature\_   
ids.   
5. Write evidence and confidence for additional features only in additional\_features.   
6. No ID array may contain duplicates.   
7. Return an empty array, object, or string when there is no corresponding content.   
8. Do not output fields outside the structure above.  
Figure 9: Prompt 1: Core-catalog and open feature extraction (part 3 of 3). System prompt for annotating format features in unconstrained responses. English translation.

Prompt 2: Open feature extraction with eight categories   
You are a response-format feature discovery annotator. Analyze only the observable presentation of   
candidate\_answer. Do not evaluate content correctness or infer the question, model identity, or writing   
intent.   
Freely extract reusable format features as comprehensively as possible. You are not restricted to existing   
feature names and should name each feature yourself; however, a feature must:   
1. Be explicitly requireable or forbidable;   
2. Be reliably judgeable from the response text alone;   
3. Transfer across topics and models;   
4. Not be a topic, fact, opinion, tone, or a vague judgment that the response is well written.   
Use the following eight categories to aid classification, but do not treat them as a closed catalog:   
- A1: Length, paragraphs, and line layout   
- A2: Headings, hierarchy, and sections   
- A3: Lists, numbering, and enumeration   
- A4: Tables, fields, and structured slots   
- A5: Openings, overviews, and closings   
- A6: Reasoning, comparison, and information relations   
- A7: Emphasis, punctuation, and local markers   
- A8: Code, mathematics, quotations, and embedded blocks   
Rules:   
- Use short, stable English snake\_case for feature\_id; do not split the same concept into synonymous   
duplicates.   
- Give each feature a concise Chinese name, an operational definition, high/medium confidence, and 1-3   
verbatim evidence snippets from candidate\_answer.   
- If a feature spans categories, select only the most appropriate one.   
- Be comprehensive, but do not excessively split one-off content details into format features.   
- Output only one valid JSON object, without Markdown or explanations.   
Output structure:   
{   
"question\_id": "Exactly the same as the input",   
"primary\_format": "plain\_prose|headings\_and\_sections|bullet\_list|numbered\_steps|   
table|code\_centric|mixed|other",   
"features": [   
{   
"feature\_id": "snake\_case\_id",   
"feature\_name": "Chinese name",   
"category\_id": "A1|A2|A3|A4|A5|A6|A7|A8",   
"definition": "Operational definition",   
"evidence": ["Verbatim evidence"],   
"confidence": "high|medium"   
}   
],   
"notes": "May be empty"   
}  
Figure 10: Prompt 2: Open feature extraction with eight categories. System prompt for open feature discovery on additional models. English translation.

![](images/778a19bd0b45e454a40c58184caf7434b5302b4a19c3d6b2598ab14b7b0da660.jpg)  
Figure 11: Prompt 3: Three-constraint format grading (part 1 of 4). Shared grading prompt for GPT-5.4 Mini and Gemini 3.1 Flash-Lite. English translation.

![](images/24520942a3b50bc87bfc4c9d607d4b7c563f8d81d0b8a55eb799777a49747d37.jpg)  
Figure 12: Prompt 3: Three-constraint format grading (part 2 of 4). Shared grading prompt for GPT-5.4 Mini and Gemini 3.1 Flash-Lite. English translation.

![](images/46863dacbaebcfb2801525734c7e3c1229effdf2949d54178ea1c64fbd7bb358.jpg)  
Figure 13: Prompt 3: Three-constraint format grading (part 3 of 4). Shared grading prompt for GPT-5.4 Mini and Gemini 3.1 Flash-Lite. English translation.

Prompt 3: Three-constraint format grading (part 4 of 4)   
Format feature 3:   
- feature\_id: {{constraint\_3\_feature\_id}}   
- feature\_name: {{constraint\_3\_feature\_name}}   
- definition: {{constraint\_3\_definition}}   
- chosen\_mode: {{constraint\_3\_chosen\_mode}}   
- constraint\_text: {{constraint\_3\_constraint\_text}}   
Unscored generic constraint:   
- included: {{generic\_constraint\_included}}   
- type: {{generic\_constraint\_type}}   
- definition: {{generic\_constraint\_definition}}   
- constraint\_text: {{generic\_constraint\_text}}   
<Student Answer>   
{{model\_response}}   
[9. Output format]   
Give a concise rationale for each feature, directly identifying what is satisfied or violated. Follow this   
format strictly, without additional headings, fields, or explanations:   
[Scoring rationale]:   
1. {{constraint\_1\_feature\_name}}: Briefly explain the score of 0 or 1.   
2. {{constraint\_2\_feature\_name}}: Briefly explain the score of 0 or 1.   
3. {{constraint\_3\_feature\_name}}: Briefly explain the score of 0 or 1.   
[Feature scores]: X1 points, X2 points, X3 points   
[Total score]: X points   
[JSON]:   
\`\`\`json   
{"feature\_scores":[   
{"feature\_id":"{{constraint\_1\_feature\_id}}","score":X1},   
{"feature\_id":"{{constraint\_2\_feature\_id}}","score":X2},   
{"feature\_id":"{{constraint\_3\_feature\_id}}","score":X3}   
],"answer\_score":X}   
X1, X2, and X3 can only be 0 or 1. X must equal X1+X2+X3 and can only be 0, 1, 2, or 3. The number,   
order, and feature\_id values of feature\_scores in JSON must exactly match the three input features.  
Figure 14: Prompt 3: Three-constraint format grading (part 4 of 4). Shared grading prompt for GPT-5.4 Mini and Gemini 3.1 Flash-Lite. English translation.

![](images/dba3c667f20888a2489e8543ea9dbe39d6a4bb5470319e7bcc8a752709061d0f.jpg)  
Figure 15: Prompt 4: Single-constraint question reconstruction (part 1 of 3). Reconstruction prompt for preparing the single-constraint ablation questions. English translation.

Prompt 4: Single-constraint question reconstruction (part 2 of 3)   
5. If the original constraint depends on another constraint, first use only information already in the   
original question to supply necessary content objects and make the requirement semantically equivalent   
and self-contained. Do not add format conditions when repairing references.   
6. If the original question embeds necessary content roles in a constraint, such as "item 1 gives the   
judgment and item 2 explains the grounds," preserve the content objects needed by that constraint itself,   
rather than retaining only its numerical shell.   
7. Write standalone\_text as natural sentences in the same language as the task stem. Do not include   
experimental expressions such as "constraint 1," "C1," "format feature," "acceptance criteria," or   
"formats not listed are not required."   
[Version rules]   
1. All three final\_question values must share the same normalized\_task\_requirement verbatim.   
2. Each final\_question must contain its selected standalone\_text verbatim, and must not contain an   
unselected standalone\_text or a synonymous paraphrase.   
3. Apart from the shared separator, do not add version-specific transitions, explanations, politeness   
expressions, or extra task requirements.   
4. Each version must contain exactly one scored constraint. Do not generate C12, C13, C23, or any   
two-constraint version.   
5. generic\_constraint must never enter any of the three versions.   
6. Do not copy original\_final\_question and mechanically delete sentences. Rebuild from normalized T   
and U1/U2/U3.   
7. separator is usually an empty string for Chinese, Japanese, etc., and a single ASCII space for English,   
etc. Use the same separator in all versions.   
[Dependencies and exceptions]   
dependency\_status can only be independent, repaired, or unresolved:   
- independent: the original constraint is already independently understandable;   
- repaired: it becomes independently understandable by supplying only objects or content bindings   
already in the original question;   
- unresolved: its original meaning cannot be preserved without other constraints, or any rewrite would   
alter judgeable conditions.   
Even when unresolved occurs, provide the most conservative best reconstruction, generate all three   
versions, and set manual\_review\_required to true. Flag manual review whenever task fidelity, parameter   
preservation, exclusion of constraints, or self-containedness is uncertain.   
[Output requirements]   
Output only one valid JSON object, without Markdown code fences, explanations outside JSON, or   
reasoning. All keys must be present, with arrays in the fixed order below:   
{   
"parent\_benchmark\_item\_id": "Copy the input ID",   
"language": "Copy the input language",   
"normalized\_task\_requirement": "T",   
"separator": "Empty string or one ASCII space",   
"constraint\_units": [   
{"original\_index":1,   
"feature\_id":"Copy feature\_id of constraint 1",   
"chosen\_mode":"Copy require or forbid",   
"standalone\_text":"U1",   
"dependency\_status":"independent|repaired|unresolved",   
"repair\_note":""},  
Figure 16: Prompt 4: Single-constraint question reconstruction (part 2 of 3). Reconstruction prompt for preparing the single-constraint ablation questions. English translation.

Prompt 4: Single-constraint question reconstruction (part 3 of 3)   
{"original\_index":2,   
"feature\_id":"Copy feature\_id of constraint 2",   
"chosen\_mode":"Copy require or forbid",   
"standalone\_text":"U2",   
"dependency\_status":"independent|repaired|unresolved",   
"repair\_note":""},   
{"original\_index":3,   
"feature\_id":"Copy feature\_id of constraint 3",   
"chosen\_mode":"Copy require or forbid",   
"standalone\_text":"U3",   
"dependency\_status":"independent|repaired|unresolved",   
"repair\_note":""}   
],   
"variants": [   
{"condition":"C1","included\_constraint\_indices":[1],   
"final\_question":"Strictly concatenate T, separator, U1"},   
{"condition":"C2","included\_constraint\_indices":[2],   
"final\_question":"Strictly concatenate T, separator, U2"},   
{"condition":"C3","included\_constraint\_indices":[3],   
"final\_question":"Strictly concatenate T, separator, U3"}   
],   
"self\_check": {   
"task\_preserved": true,   
"all\_constraint\_parameters\_preserved": true,   
"original\_constraint\_text\_verbatim": true,   
"all\_units\_self\_contained": true,   
"each\_variant\_has\_exactly\_one\_constraint": true,   
"excluded\_constraints\_absent": true,   
"generic\_constraint\_absent": true,   
"no\_new\_format\_constraint\_added": true,   
"shared\_wording\_invariant\_satisfied": true   
},   
"manual\_review\_required": false,   
"review\_reasons": []   
}   
[Mandatory checks before output]   
1. ID, language, all three feature\_id values, and chosen\_mode values match the input verbatim.   
2. constraint\_units contains exactly three entries with original\_index 1, 2, and 3 in order; variants is   
ordered exactly C1, C2, C3.   
3. Each child question contains only one constraint. No two-constraint version or generic\_constraint is   
present.   
4. Each final\_question can be reconstructed verbatim from T, separator, and its single corresponding U.   
5. Does each standalone\_text contain the corresponding parent constraint\_text verbatim and   
contiguously? Are all numbers, ranges, fixed strings, punctuation, symbols, heading text, list-item   
counts, paragraph counts, position words, order, and prohibition scope fully preserved?   
6. There are no dangling references, missing content slots, language mixing, new format requirements, or   
answers to the original question.   
7. If any check is uncertain, set the corresponding self\_check field to false, set manual\_review\_   
required=true, and explain in review\_reasons.   
PARENT\_PAYLOAD:   
{{PARENT\_PAYLOAD}}  
Figure 17: Prompt 4: Single-constraint question reconstruction (part 3 of 3). Reconstruction prompt for preparing the single-constraint ablation questions. English translation.

![](images/7940203246219d1da8de7a899065f2d368f8d3fe70fd9cc1bab438067018e4af.jpg)  
Figure 18: Prompt 5: Qwen3.5-9B prompt enhancement. Meta-prompt appended to the question in the same user message for Qwen3.5-9B. English translation.