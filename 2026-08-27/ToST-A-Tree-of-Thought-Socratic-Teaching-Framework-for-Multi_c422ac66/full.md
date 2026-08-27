# ToST: A Tree-of-Thought Socratic Teaching Framework for Multi-Path Guidance and Parallel Thinking

Feng Ling Beijing Normal University

Heng Yu Beijing Normal University

## Abstract

Large Language Models (LLMs) exhibit strong problem-solving abilities, positioning them as promising agents for Socratic teaching to guide students through step-by-step heuristic questioning. However, existing approaches typically adopt a one-problem–one-solution paradigm, restricting the teaching guidance to a single linear reasoning path. This design limits instructional flexibility, weakens error recov ery, and restricts students’ ability to engage in parallel thinking to explore multiple valid solutions. To overcome these, we propose ToST, a Tree-of-Thought Socratic Teaching frame work that explicitly supports multi-path guid ance under a one-problem–multiple-solutions paradigm. ToST employs Parallel Sowing, a parallel-thinking–oriented questioning strategy to encourage students to approach problems from diverse perspectives, and a Multi-Path Adaptive Guidance mechanism to provide more robust and non-linear instructions across alternative solution trajectories. Concurrently, to fill the void in systematically evaluating such non-linear instructional capabilities, we advance the task of multi-path Socratic guid ance by establishing MPSG-Bench, a comprehensive benchmark that includes a dataset of 31K multi-path teaching dialogues and a fivedimensional evaluation framework grounded in the SOLO (Structure of Observed Learning Outcomes) theory to assess parallel-thinking guidance. Experimental results demonstrate that ToST significantly enhances guidance success rates while empowering students to navi gate and explore multiple solution paths more effectively under both automatic and human metrics.

## 1 Introduction

Large language models (LLMs) have become central to personalized AI tutoring (Pedro et al., 2019; Zhai et al., 2021), with recent advancements focusing on expert pedagogical modeling (Wang et al., 2024) and Socratic questioning (Zhang et al., 2024b; Liu et al., 2024). These methods aim to provide coherent, step-by-step scaffolding through instructional dialogues.

Despite these gains, most existing approaches are confined to a one-problem–one-solution (1P1S) paradigm (Gao et al., 2025a), which restricts instructional flexibility and undermines the development of learners’ parallel thinking (Pásztor et al., 2015)—an essential capability for navigating complex, non-linear problems from multiple perspectives (Evans and Swan, 2014). As illustrated in Figure 1(a), traditional Socratic teaching typically anchors guidance to a single solution path, making it hard to recover from ineffective guidance (Chu et al., 2025). These constraints motivate a shift toward a one-problem–multiple-solutions (1PMS) paradigm that explicitly models multiple solution paths to provide more flexible and cognitively enriched guidance.

However, simply prompting LLMs to provide 1PMS-based guidance is challenging, since multipath instruction imposes substantial System 2 (Kahneman, 2011) planning demands: LLMs often struggle to track student progress across concurrent paths or decide when to switch trajectories. This difficulty is compounded by a lack of interpretable mechanisms for evaluating guidance strategies and estimating student cognitive states.

To address these challenges, we propose ToST, a Tree-of-Thought Socratic Teaching framework that instantiates the 1PMS paradigm via a parallel reasoning tree. ToST formalizes Socratic teaching as a tree-structured decision-making process, in which the parallel reasoning tree explicitly supports System 2 reasoning by tracking student progress across multiple concurrent solution paths and modeling pedagogically meaningful inter-path relationships. As illustrated in Figure 1(b), ToST applies Parallel Sowing, a questioning strategy that elicits multiple perspectives on the problem, followed by a multi-path adaptive guidance mechanism that dynamically decides whether to persist the current reasoning path or transition to alternatives when misconceptions arise. During path transitions, teachers are encouraged to leverage reusable steps (the shared steps between different paths) that the student has previously executed correctly. These validated steps serve as familiar scaffolding to bridge transitions toward alternative strategies and encourage the exploration of new methods.

![](images/a713af0a89415f26cf3ad564280f1f9239e41aaf112b99ff35dba378fe9aaa1d.jpg)  
Figure 1: Comparison between traditional Socratic Teaching and the proposed ToST paradigm. Traditional method (left) employs a 1P1S paradigm with linear reasoning, susceptible to stalling from intermediate errors. In contrast, ToST (right) adopts a 1PMS paradigm via parallel thinking, enabling dynamic redirection when paths fail.

To fill the void in systematically evaluating nonlinear instructional capabilities in 1PMS paradigm, we establish MPSG-Bench, a Multi-Path Socratic Guidance benchmark comprising 31k teaching dialogues. We propose a five-dimensional evaluation framework grounded in the Structure of Observed Learning Outcomes (SOLO) taxonomy (Biggs and Collis, 2014), enabling systematic assessment of teachers’ 1PMS guidance and students’ parallelthinking ability. Experimental results on MPSG-Bench demonstrate that our ToST framework significantly improves guidance success rates by 11% over educational LLM baselines and achieves more efficient exploration on multi-solution reasoning trees.

## In summary, our contributions are:

• We introduce the 1PMS paradigm to Socratic teaching and formalize it via a parallel reasoning tree, enabling explicit System 2 reasoning and inter-path pedagogical analysis to provide more adaptive instructional feedback.

• We propose ToST, a Tree-of-Thought Socratic Teaching framework to dynamically navigate concurrent solution paths, improving instructional flexibility and robustness.

• We present MPSG-Bench, a new benchmark for multi-solution Socratic teaching, comprising 31k teaching dialogues and a SOLO-based evaluation framework to quantify structural complexity and parallel thinking across multiple solution paths, with extensive experiments validating the effectiveness of our approach.

## 2 Related Work

## 2.1 LLMs-enhanced Personalize Tutoring

Large language models (LLMs) have been widely adopted for personalized tutoring, supporting adaptive instruction (Kasneci et al., 2023; Chen et al., 2024), student simulation (Zhan et al., 2025; Gao et al., 2025a), and learning path planning (Gao et al., 2025b; Wang et al., 2025). Pedagogical alignment is commonly achieved through expert rules (Wang et al., 2024; Park et al., 2024), educational pretraining (Dan et al., 2024), or reinforcement learning (Dinucu-Jianu et al., 2025), with Socratic prompting further enhancing personalization (Liu et al., 2024; Chang, 2023). However, most approaches rely on single-chain reasoning and adopt a one-problem–one-solution (1P1S) paradigm, limiting the cultivation of oneproblem–multiple-solutions (1PMS) reasoning.

![](images/754ceb9da318deb683961904d9446925eeb6e75d2d3c1a09996f3b16380bc5ce.jpg)  
Figure 2: Parallel Reasoning Tree (PRT) structure. The root defines the problem, with branches representing alternative solution paths weighted by complexity, novelty, and depth. Nodes signify correct intermediate steps leading to final answer.

## 2.2 Foundational Paradigms in Multi-Solution Pedagogical Reasoning

The one-problem–multiple-solutions (1PMS) paradigm promotes flexible problem-solving through the exploration of diverse valid solution paths (Evans and Swan, 2014). This perspective aligns with parallel thinking, which promotes the simultaneous consideration of multiple reasoning strategies (De Bono, 2016). Learning progression in such settings is commonly assessed using the Structure of Observed Learning Outcomes (SOLO) taxonomy (Biggs and Collis, 2014) (introduced in Appendix D), which provides a principled basis for our structure-aware, multi-path instructional framework. Recent work extends linear chainof-thought prompting (Wei et al., 2022) with structured reasoning based on trees (Yao et al., 2023), graphs (Besta et al., 2024), or heuristic and value-based evaluation (Browne et al., 2012). Parallel reasoning approaches aggregate independent solution chains via comparison or consensus (Du et al., 2023; Zheng et al., 2025), but lack explicit hierarchical representations and path-transition modeling. In contrast, ToST explicitly constructs and navigates multiple solution trajectories within a Parallel Reasoning Tree, enabling process-aware multi-path instructional guidance.

## 3 Background and Notation

## 3.1 Parallel Reasoning Tree Construction

The Parallel Reasoning Tree (PRT) represents multiple valid solution paths for a single problem within a unified hierarchical structure. Each rootto-leaf path corresponds to a complete reasoning trajectory, explicitly representing alternative solution strategies in 1PMS settings. As illustrated in Figure 2, each solution path is encoded as a complete reasoning trajectory, with explicit branching to capture divergence. Each path is annotated with three interpretable attributes: complexity, reflecting intrinsic solution difficulty; innovativeness, measuring structural deviation from canonical solutions provided by the official dataset; and depth, defined as the number of intermediate reasoning nodes. These attributes facilitate path-level comparisons and adaptive guidance. The tree construction detail is introduced in Appendix B, including the construction cost analysis in Appendix B.2.

## 3.2 Problem Definition

Let Q denote a problem instance admitting a set of valid solutions $\boldsymbol { S } = \{ S _ { 1 } , \ldots , S _ { k } \}$ with $k \geq 2$ . We formalize the Parallel Reasoning Tree (PRT) as a rooted directed acyclic graph $\mathcal { T } = ( V , P , D , C , I )$ where V comprises the root $v _ { 0 } = \mathcal { Q }$ , intermediate reasoning nodes, and solution leaves. The set P contains all root-to-leaf solution paths, and each path $p \in P$ is associated with a depth $D ( p ) \in \mathbb { N } ,$ complexity $C ( p ) \in \mathbb { R }$ , and innovativeness $I ( p ) \in$ R.

Let $P _ { e } \subseteq P$ denote expert-annotated paths and $P _ { s }$ student-generated paths extracted from instructional interactions, where each student path corresponds to a (possibly partial) subpath of some $p \in P _ { e }$ . Intermediate nodes in $P _ { s }$ are partitioned into correctly solved nodes K and unsolved nodes M, with a subset identified as reusable nodes $R \subseteq K \cup M$ . Given $\tau$ and a student’s partial trajectory τ, the instructional objective is to select guidance actions that maximize learning effectiveness over T under the one-problem–multiplesolutions (1PMS) setting, by jointly optimizing student–expert path alignment and pedagogical factors such as reasoning depth and cognitive load.

## 4 Tree-of-Thought Socratic Teaching

As shown in Figure 3, ToST operates on an expert PRT $\mathcal { T } _ { e } = ( V _ { e } , P _ { e } , D _ { e } , C _ { e } , I _ { e } )$ and an incrementally parsed student tree $\mathcal { T } _ { s } = ( V _ { s } , P _ { s } , D _ { s } , C _ { s } , I _ { s } )$

![](images/a30bae3fa14427c6b6fec0170790e10a160dda7ba3472842688b7f4eed579266.jpg)  
Figure 3: Overview of the Tree-of-Thought Socratic Teaching (ToST) framework. ToST operates on Parallel Reasoning Trees and consists of Parallel Sowing for multi-solution exploration and Multi-Path Adaptive Guidance for personalized Socratic instruction via student–expert tree comparison.

to diagnose reasoning progress across concurrent solution paths. It comprises Parallel Sowing, which initiates multi-path exploration, and Multi-Path Adaptive Guidance (MPAG), which performs treebased diagnosis and guidance selection.

## 4.1 Parallel Sowing

Parallel Sowing initializes the student reasoning tree through teacher-guided multi-path exploration rather than by requiring the student to independently propose several complete solutions. This step forms the initial branches of $\mathcal { T } _ { s }$ , establishing the structural basis for subsequent path-level diagnosis and guidance. Importantly, Parallel Sowing does not assume that a novice learner can fully enumerate complete solutions. The teacher may elicit partial intuitions, observable constraints, decomposition ideas, or candidate methods, each of which can serve as an entry point into a different branch of the PRT.

## 4.2 Multi-Path Adaptive Guidance Strategy

Multi-Path Adaptive Guidance (MPAG) provides path-aware instructional control over $\mathcal { T } _ { s }$ . It comprises two modules: a Tree-based Student Cognitive Manager for progress estimation and error localization, and an Automatic Path Analyzer for guidance decision-making.

## 4.2.1 Error Diagnosis with Tree-based Student Cognitive Manager

As dialogue proceeds, the Tree-based Student Cognitive Manager updates $\mathcal { T } _ { s }$ and performs Path-Match for solution-category alignment and Node-Match for intermediate-step correctness.

For each student path $P _ { s } ^ { ( i ) }$ , let $V _ { P _ { s } ^ { ( i ) } }$ denote the set of nodes along that path. A progress score is computed as

$$
\mathrm { S c o r e } _ { p } ( i ) = \frac { \sum _ { v \in V _ { P _ { s } ^ { ( i ) } } } w _ { d } ( v ) \omega ( v ) s ( v , \mathcal { T } _ { e } ) } { \sum _ { v \in V _ { P _ { s } ^ { ( i ) } } } w _ { d } ( v ) \omega ( v ) }\tag{1}
$$

where $s ( v , \mathcal { T } _ { e } ) \in \{ 0 , 1 \}$ indicates node-level correctness, $w _ { d } ( v )$ prioritizes earlier steps, and ω(v) gives higher weight to reusable nodes:

$$
\omega ( v ) = \left\{ \begin{array} { l l } { \gamma , } & { v \in R , } \\ { \sigma , } & { v \in K \setminus R , } \\ { \delta , } & { v \in M \setminus R , R \neq \emptyset , } \\ { 1 . 0 , } & { v \in M \setminus R , R = \emptyset , } \end{array} \right.\tag{2}
$$

Here, R denotes reusable nodes, K correctly solved nodes, and M unsolved nodes. The coefficients γ, σ, and δ are hyperparameters controlling the relative importance of reusable, correct, and incorrect nodes, respectively, with $\gamma \ge \sigma \ge \delta > 0$

This weighting favors stable and reusable reasoning steps, so the score reflects how much reliable structure can support subsequent guidance.

## 4.2.2 Guidance Decision with Automatic Path Analyzer

The Automatic Path Analyzer selects instructional paths by evaluating candidate expert paths. Inspired by the global Product-of-Experts formulation (Loula et al., 2025), we model each instructional path as being constrained by multiple pedagogically relevant factors, each contributing a potential function to the overall guidance value. Formally, for each candidate path $P _ { t } \in P _ { e }$ , we define a guidance value $H ( P _ { t } )$ that integrates student progress, remaining problem difficulty, instructional investment, and cognitive load. The resulting guidance value is computed as

$$
\begin{array} { r } { H ( P _ { t } ) = \underbrace { \left[ \alpha \mathrm { S c o r e } _ { p } ( t ) + \cfrac { \beta } { 1 + E ( P _ { t } ) } \right] } _ { \mathrm { P a t h U t i i i t y ~ E s t i m a t i o n } } \cdot \underbrace { I ( P _ { t } ) } _ { C ( P _ { t } ) } } \\ { \cdot \underbrace { \left( 1 + \rho \frac { T ( P _ { t } ) } { T _ { \mathrm { m a x } } } \right) } _ { \mathrm { C o n v e r s a t i o n ~ G a i n } } \cdot \underbrace { \frac { 1 } { 1 + \lambda D ( P _ { t } ) } } _ { \mathrm { C o g n i t i v e ~ L o a d } } \quad ( 3 ) } \end{array}\tag{}
$$

where $E ( P _ { t } )$ is remaining difficulty, $I ( P _ { t } )$ and $C ( P _ { t } )$ are innovativeness and complexity, $T ( P _ { t } ) / T _ { \operatorname* { m a x } }$ measures dialogue investment, and $D ( P _ { t } )$ is path depth. Intuitively, $H ( P _ { t } )$ favors paths with reliable progress, manageable remaining difficulty, useful novelty relative to complexity, sufficient dialogue investment, and bounded cognitive load.

The next guidance path is selected using a greedy switching rule:

$$
P _ { \mathrm { n e x t } } = \left\{ \begin{array} { l l } { P _ { \mathrm { c } } , \quad \mathrm { i f } \ H ( P _ { \mathrm { c } } ) + \theta _ { \mathrm { s w i t c h } } \ge H _ { \mathrm { m a x } } ^ { - c } , } \\ { \mathrm { a r g } \displaystyle \operatorname* { m a x } _ { P \in P _ { e } , P \ne P _ { c } } H ( P ) , \quad \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{4}
$$

where $H _ { \operatorname* { m a x } } ^ { - c } = \operatorname* { m a x } _ { P \in P _ { e } , P \neq P _ { c } } H ( P )$ denotes the highest guidance value among alternative paths excluding the current path $P _ { c } .$ The threshold $\theta _ { \mathrm { s w i t c h } } > 0$ controls the sensitivity of switching decisions. It acts as an inertia margin: MPAG switches only when an alternative path is clearly better, which keeps the rule interpretable and reduces oscillation.

At each dialogue round, the Tree-based Student Cognitive Manager and the Automatic Path Analyzer jointly update $\mathcal { T } _ { s }$ and select guidance actions, enabling adaptive 1PMS instruction to promote students’ parallel thinking.

## 5 Benchmark: MPSG-Bench

Effective 1PMS instruction requires benchmarks that capture parallel reasoning beyond final-answer accuracy. Existing educational datasets typically assume a single canonical solution, making them unsuitable for multi-path assessment. To fill this gap, we introduce Multi-Path Socratic Guidance Benchmark (MPSG-Bench): a large-scale benchmark of multi-path problem-solving trajectories and a SOLO-grounded evaluation framework for quantifying a model’s ability of parallel-thinking guidance.

## 5.1 Constructing MPSG-Bench with a Multi-Agent Pipeline

MPSG-Bench contains 31k PRT-grounded teaching dialogues with two components:

Enhanced Multi-Path Problem-Solving Collection. Seed problems from GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021) are expanded with DeepSeek v3.2 into annotated PRTs with approximately five distinct solution paths per problem.

Multi-Path Teaching Dialogue Dataset. Using these PRTs, Student, Teacher, and Expert agents generate Socratic dialogues under six student archetypes (Appendix B).

Together, these datasets facilitates the finetuning of LLMs on structured, non-linear instructional turns and provides a standardized testbed to quantify performance gaps in error recovery and path-consistency—capabilities that traditional single-path datasets are unable to measure.

## 5.2 Parallel Thinking Evaluation System

We propose a five-dimensional evaluation framework that instantiates the 1PMS objective via SOLO theory, to quantify structure-aware reasoning, path-sensitive diagnosis, and guidance efficiency. Unlike prior methods that rely on finalanswer accuracy or heuristic judgments (Dinucu-Jianu et al., 2025; Liu et al., 2024; Yang et al., 2025), our approach enables an interpretable, finegrained assessment of a model’s ability to facilitate parallel thinking.

(1) SOLO Advancement Score (SAS). SOLO Advancement Score (SAS) measures changes in students’ cognitive structure before and after guidance, based on the SOLO taxonomy (Biggs and Collis, 2014). Student reasoning is independently assessed five times using GPT-5, and SAS is computed as the difference between post- and preguidance SOLO levels.

Table 1: Performance comparison. Original student performance (before guidance) is reported as "Raw Students". Acc denotes final problem-solving accuracy, while TreeAcc, $\bar { N } _ { \mathrm { R o u n d } } .$ , and TreeAcc-R are defined in Section 5.2. $\bar { N } _ { \mathrm { M e t h o d } }$ denotes the average number of solution strategies attempted. ToST w/o PS and ToST w/o MPAG denote ablations removing Parallel Sowing, and Multi-Path Adaptive Guidance, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="5">GSM8K</td><td colspan="5">MATH-500</td></tr><tr><td>Acc (%)</td><td>TreeAcc</td><td> $\overline { { N _ { \mathrm { R o u n d } } } }$ </td><td>TreeAcc-R</td><td> $\overline { { N _ { \mathrm { M e t h o d } } } }$ </td><td>Acc (%)</td><td>TreeAcc</td><td> $\underline { { \overline { { N } } _ { \mathrm { R o u n d } } } }$ </td><td>TreeAcc-R</td><td> $\overline { { N _ { \mathrm { M e t h o d } } } }$ </td></tr><tr><td>Raw Students</td><td>48.49</td><td>37.28</td><td></td><td></td><td>1.04</td><td>61.20</td><td>35.82</td><td></td><td></td><td>1.16</td></tr><tr><td>SocraticLM</td><td>84.38</td><td>57.47</td><td>3.33</td><td>17.26</td><td>1.95</td><td>92.60</td><td>49.42</td><td>2.29</td><td>21.58</td><td>1.55</td></tr><tr><td>EduChat-R1-8b</td><td>76.27</td><td>51.93</td><td>3.94</td><td>13.18</td><td>1.62</td><td>84.40</td><td>44.39</td><td>2.42</td><td>18.34</td><td>1.41</td></tr><tr><td>EduChat-R1-32b</td><td>79.45</td><td>54.37</td><td>3.51</td><td>15.49</td><td>1.78</td><td>86.80</td><td>47.18</td><td>2.33</td><td>20.25</td><td>1.58</td></tr><tr><td>TutorRL-7B</td><td>89.61</td><td>60.17</td><td>2.86</td><td>21.04</td><td>1.91</td><td>91.80</td><td>48.99</td><td>2.26</td><td>21.68</td><td>1.52</td></tr><tr><td>DeepSeek V3.2</td><td>89.92</td><td>57.11</td><td>2.26</td><td>25.27</td><td>1.16</td><td>96.00</td><td>50.13</td><td>1.89</td><td>26.52</td><td>1.53</td></tr><tr><td>GPT5</td><td>98.56</td><td>67.05</td><td>1.97</td><td>34.04</td><td>2.06</td><td>98.20</td><td>50.75</td><td>1.75</td><td>29.00</td><td>1.67</td></tr><tr><td>ToST (Ours)</td><td>98.18</td><td>68.75</td><td>1.68</td><td>40.92</td><td>2.39</td><td>99.20</td><td>56.01</td><td>1.53</td><td>36.61</td><td>2.12</td></tr><tr><td>ToST w/o PS</td><td>96.36</td><td>64.65</td><td>1.76</td><td>36.73</td><td>2.04</td><td>97.00</td><td>50.68</td><td>1.71</td><td>29.64</td><td>1.53</td></tr><tr><td>ToST w/o MPAG</td><td>97.19</td><td>66.12</td><td>1.72</td><td>39.02</td><td>2.23</td><td>98.80</td><td>54.88</td><td>1.67</td><td>32.86</td><td>1.96</td></tr><tr><td>Raw w/ PS</td><td>51.40</td><td>42.79</td><td>AIME24</td><td></td><td>2.12</td><td>65.00</td><td>33.53</td><td></td><td></td><td>1.86</td></tr><tr><td rowspan="2">Method</td><td colspan="5"></td><td colspan="5">AIME25</td></tr><tr><td>Acc (%) TreeAcc</td><td></td><td>NRound</td><td>TreeAcc-R</td><td> $\overline { { N _ { \mathrm { M e t h o d } } } }$ </td><td>Acc (%)</td><td>TreeAcc</td><td> $\overline { { N _ { \mathrm { R o u n d } } } }$ </td><td>TreeAcc-R</td><td> $\overline { { N _ { \mathrm { M e t h o d } } } }$ </td></tr><tr><td>Raw Students SocraticLM</td><td>71.94</td><td>46.08</td><td></td><td></td><td>1.13</td><td>31.25</td><td>29.41</td><td></td><td></td><td>1.07</td></tr><tr><td></td><td>90.83</td><td>55.70</td><td>2.25</td><td>24.76</td><td>1.32</td><td>63.13</td><td>53.19</td><td>4.72</td><td>11.27</td><td>1.74</td></tr><tr><td>EduChat-R1-8b</td><td>84.72</td><td>49.72</td><td>2.86</td><td>17.38</td><td>1.23</td><td>46.04</td><td>30.15</td><td>7.29</td><td>4.14</td><td>1.51</td></tr><tr><td>EduChat-R1-32b</td><td>86.11</td><td>51.68</td><td>2.59</td><td>19.95</td><td>1.29</td><td>53.33</td><td>48.71</td><td>5.43</td><td>8.97</td><td>1.64</td></tr><tr><td>TutorRL-7B</td><td>89.72</td><td>55.67</td><td>2.33</td><td>23.89</td><td>1.31</td><td>41.88</td><td>34.49</td><td>6.72</td><td>5.13</td><td>1.53</td></tr><tr><td>DeepSeek V3.2</td><td>96.39</td><td>59.49</td><td>1.65</td><td>36.05</td><td>1.31</td><td>72.50</td><td>53.62</td><td>4.23</td><td>12.68</td><td>1.49</td></tr><tr><td>GPT5</td><td>97.78</td><td>60.32</td><td>1.74</td><td>34.67</td><td>1.29</td><td>73.13</td><td>47.71</td><td>4.18</td><td>11.41</td><td>1.45</td></tr><tr><td>ToST (Ours)</td><td>96.67</td><td>63.39</td><td>1.65</td><td>38.42</td><td>1.73</td><td>81.25</td><td>55.77</td><td>3.89</td><td>14.34</td><td>1.79</td></tr><tr><td>ToST w/o PS</td><td>95.28</td><td>57.28</td><td>1.82</td><td>31.47</td><td>1.62</td><td>79.38</td><td>52.82</td><td>4.03</td><td>13.11</td><td>1.62</td></tr><tr><td>ToST w/o MPAG</td><td>96.39</td><td>61.74</td><td>1.73</td><td>35.69</td><td>1.71</td><td>78.54</td><td>54.19</td><td>3.94</td><td>13.75</td><td>1.73</td></tr><tr><td>Raw w/ PS</td><td>76.11</td><td>52.61</td><td></td><td></td><td>1.67</td><td>35.21</td><td>45.90</td><td></td><td></td><td>1.65</td></tr></table>

(2) Tree Accuracy (TreeAcc). TreeAcc evaluates the correctness of students’ internal reasoning structures within the PRT by jointly considering node-level reasoning quality, path-level methodological alignment, and final-answer correctness: This approach is inspired by CodeBLEU (Ren et al., 2020), which combines abstract syntax trees (AST), data flow, and n-gram similarities to evaluate code generation tasks.

$$
\begin{array} { r l } & { \mathrm { T r e e A c c } = \alpha _ { t } \cdot \mathrm { N o d e A c c } + \beta _ { t } \cdot \mathrm { P a t h A c c } } \\ & { ~ + ~ \gamma _ { t } \cdot \mathrm { F i n a l M a t c h } } \end{array}\tag{5}
$$

where $\alpha _ { t } ~ + ~ \beta _ { t } ~ + ~ \gamma _ { t } ~ = ~ 1$ , NodeAcc = $\begin{array} { r } { \frac { 1 } { | P _ { s } | } \sum _ { p \in P _ { s } } \mathrm { S c o r e } _ { p } ( p ) } \end{array}$ FinalMatch $\mathbb { I } [ \mathrm { l e a f } ( p )$ is correct], PathAcc measures alignment between student and expert solution paths, weighted by innovativeness and complexity, and $\bar { N } _ { \mathrm { R o u n d } }$ denotes the average number of dialogue rounds.

Moreover, as the efficiency in 1PMS settings depends on both reasoning correctness and how quickly guidance facilitates convergence, we define Tree Accuracy per Round (TreeAcc-R) as TreeAcc- $\begin{array} { r } { { \bf \nabla } \cdot { \bf R } \ = \ \frac { \mathrm { T r e e A c c } } { \bar { N } _ { \mathrm { R o u n d } } } } \end{array}$ where $\bar { N } _ { \mathrm { R o u n d } }$ is the average guidance round. This metric reflects the accuracy–efficiency trade-off critical to 1PMS instruction.

(3) Diagnostic Precision (DP). DP evaluates the teacher’s ability to identify misconceptions, recognize latent solution paths, and select appropriate interventions, assessed via LLM-based judgment.

(4) Persistence Gain (PG). PG measures average TreeAcc gains when guidance follows a student’s current path. It evaluates single-path scaffolding and identifies the transition from prestructural to unistructural cognition.

(5) Switching Gain (SG). SG measures the average TreeAcc improvement following path switching. It reflects the teacher’s capacity to promote cross-method expansion and integration, which is critical for advancing students from unistructural to multistructural and relational levels.

Together, these metrics jointly quantify overall parallel thinking development (SAS, TreeAcc-R, DP) and stage-specific instructional effectiveness (PG, SG), enabling comprehensive and interpretable benchmarking of 1PMS teaching systems.

![](images/99b9c3093175d23c11dffc232cfb1e27337139a48c630d1d4aeab5515d4f7318.jpg)  
Figure 4: Comparison of guidance methods with five diagnostic protocol metrics on GSM8k, MATH-500, AIME24, and AIME25.

## 6 Experiments and Results

## 6.1 Experimental Setup

We compare ToST with educational agents (SocraticLM (Liu et al., 2024), TutorRL-7B (Dinucu-Jianu et al., 2025), EduChat (Dan et al., 2024)) and general LLMs (DeepSeek V3.2 (DeepSeek-AI, 2025), GPT-5 (OpenAI, 2025)); for fair educational comparison, ToST uses QWEN2.5-MATH-7B-INSTRUCT (Team et al., 2024) as the teacher model. Experiments cover GSM8K (Cobbe et al., 2021), MATH-500 (Hendrycks et al., 2021), AIME24, and AIME25 (Mathematical Association of America, 2024); implementation and dataset details are in Appendices A and B. Ablations remove Parallel Sowing (PS) via single-chain prompting or disable MPAG by removing explicit path selection; each experiment uses three random seeds. Evaluation has two complementary parts: a large-scale simulated benchmark evaluation using MPSG-Bench protocol metrics (Table 1, Figure 4) and a smallscale human pilot evaluation in Section 6.3.

## 6.2 Main Results and Ablation Study

As shown in Table 1, ToST consistently outperforms existing educational agents and strong general-purpose LLMs across all benchmarks. Under the standard accuracy metric (Acc), ToST surpasses representative single-chain tutoring methods on every dataset, achieving an average improvement of 11% in guidance success rate, while remaining competitive with state-of-the-art general LLMs despite using a substantially smaller 7B teacher model. The advantage of ToST is more pronounced under TreeAcc-R, with gains of up to 20% over the single-chain method SocraticLM on GSM8K, underscoring the effectiveness of multipath reasoning. These improvements are consistent across problem difficulties; notably, on the challenging AIME25 benchmark, ToST outperforms the strongest general-purpose LLM baseline by approximately 8%, indicating strong generalization to complex problem settings. Ablations show that removing PS reduces explored strategies (e.g., 2.12 → 1.53 on MATH-500), while removing MPAG degrades solution quality; PS alone improves raw students by 3.71% but is insufficient. Figure 4 further shows consistent gains over educational baselines across all five internal protocol metrics, especially DP and SAS.

The PRT Parser reliability is confirmed by crossmodel agreement analysis: average PDC and NMR reach 99.23% and 97.69%, respectively, across three LLM backends (see Appendix G). Appendix B.2 quantifies the offline PRT construction overhead in terms of token use. To clarify the intended domain scope of PRT, Appendix C provides case studies in physics, coding, and scientific reasoning.

## 6.3 Human Pilot Evaluation

To further assess whether MPSG-Bench’s structural advantages translate to human judgments and learner experience, we ran a human pilot study comparing blinded expert ratings, students’ subjective responses, and an independent LLM-based judge. Complete rubrics, prompts, interfaces, and additional results are provided in Appendix F.

Track A: Blinded human evaluation. We conduct two blinded small-scale studies on 100 dialogue sessions: an expert review by seven mathematics teaching/tutoring raters and a pilot student study with ten undergraduates interacting with all five systems in counterbalanced order.

Table 2: Student study results (Track A, learner self-report). Participants rated their tutoring session on five dimensions using a 5-point Likert scale (mean $\pm \ \mathrm { s t d } )$ . κ: Fleiss’ inter-rater agreement. Teacher/expert and LLM judge results are in Appendix F. Bold: best per column.
<table><tr><td>System</td><td>Helpfulness↑</td><td>Clarity↑</td><td>Understanding↑</td><td>Willingness↑</td><td>Burden↑</td><td>κ</td></tr><tr><td>SocraticLM</td><td> $\overline { { 3 . 3 9 \pm 0 . 0 1 9 } }$ </td><td> $3 . 4 3 { \pm } 0 . 2 2 2$ </td><td> $3 . 5 6 { \pm } 0 . 0 9 0$ </td><td> $3 . 2 5 { \pm } 0 . 1 9 6$ </td><td> $\overline { { 4 . 3 3 \pm 0 . 0 8 8 } }$ </td><td>0.82</td></tr><tr><td>EduChat-R1-32b</td><td> $3 . 2 1 { \pm } 0 . 0 7 8$ </td><td> $3 . 0 2 { \pm } 0 . 0 6 2$ </td><td> $3 . 6 5 { \pm } 0 . 0 6 5$ </td><td> $3 . 1 4 { \pm } 0 . 0 9 7$ </td><td> $4 . 1 9 { \pm } 0 . 0 7 8$ </td><td>0.75</td></tr><tr><td>TutorRL-7B</td><td> $3 . 7 5 { \pm } 0 . 0 0 2$ </td><td> $4 . 0 2 { \pm } 0 . 2 1 1$ </td><td> $4 . 8 6 { \pm } 0 . 0 1 1$ </td><td> $3 . 7 3 { \pm } 0 . 2 1 6$ </td><td> $4 . 3 7 { \pm } 0 . 0 6 6$ </td><td>0.80</td></tr><tr><td>DeepSeek V3.2</td><td> $4 . 4 2 { \pm } 0 . 1 1 7$ </td><td> $4 . 1 3 { \pm } 0 . 1 1 9$ </td><td> $\mathbf { 4 . 9 2 \pm 0 . 2 2 5 }$ </td><td> $4 . 6 9 { \pm } 0 . 0 7 0 $ </td><td> $4 . 6 8 { \pm } 0 . 1 7 0 $ </td><td>0.68</td></tr><tr><td>ToST (Ours)</td><td> $\pm . 6 6 \pm 0 . 1 2 8$ </td><td> $\pm . 6 8 \pm 0 . 0 7 1$ </td><td> $4 . 8 9 { \pm } 0 . 1 9 1$ </td><td> $\pm . 9 3 { \pm } 0 . 2 0 9$ </td><td> $\pm . 9 5 { \pm } 0 . 2 2 0$ </td><td>0.74</td></tr></table>

Track B: Independent general LLM judge. As an auxiliary trend check, an independent generalpurpose LLM judge rates 300 anonymized samples on helpfulness, clarity, diagnostic correctness, and switch naturalness.

![](images/1935db0269c5d6ab43d8580a3d62fd9a28c9f17102e16f795b5994ac35474f9b.jpg)

Statistics and reporting. We report mean ± standard deviation, agreement statistics, and nonparametric tests, framing the small-scale study as pilot evidence. Table 2 presents learner-facing outcomes; teacher/expert and LLM-judge results are reported in Appendix F.

![](images/53b12d27eec24aaed672fbcaf9400397c0afb282c6f28034e5e9909b307d39bb.jpg)  
Figure 5: Stage promotion results with SocraticLM and ToST. Diagonal entries indicate no cognitive progression; upper-right entries correspond to upward stage transitions. ToST induces substantially more upward transitions (e.g., 73.3% S3→S4); some downward entries (e.g., 17.8% S3→S2) reflect adaptive consolidation under increased task difficulty.

The learner self-reports further qualify the benchmark results. ToST receives the strongest ratings on helpfulness, clarity, willingness to continue exploring, and perceived burden/manageability, indicating that learners found its structured multipath guidance useful, understandable, and well paced. This pattern is consistent with ToST’s design: by preserving partial progress, seeding only a small number of teacher-selected alternatives when needed, and guiding transitions between them, the teacher can support exploration without making the interaction feel harder to manage. The expert review and independent LLM judge in Appendix F provide the same broad signal: ToST ranks first or near first on the dimensions most directly tied to multi-path tutoring control.

## 6.4 Pedagogical Advantages of the ToST Framework

Figure 5 presents a stage-transition analysis over multi-round instructional dialogues on all datasets, based on the SOLO taxonomy introduced in Section 2.2. Here, the previous stage corresponds to the level of cognitive structure exhibited in students’ initial problem-solving attempts, whereas the next stage reflects the level attained following teacher guidance. As shown, ToST consistently promotes upward transitions (e.g., 73.3% S3→S4), substantially exceeding SocraticLM. Compared to SocraticLM, Parallel Sowing in ToST elevates students’ initial reasoning to higher SOLO levels. In some cases, guidance appropriately consolidates multiple paths into a single focused strategy (e.g., 17.8% S3→S2), reflecting adaptive scaffolding under increased task difficulty. These effects stem from ToST’s explicit representation and management of reasoning diversity, which enables comparison across alternatives.

## 7 Conclusion

In this work, we propose ToST, a Tree-of-Thought Socratic Teaching framework that enables flexible, multi-path guidance under the oneproblem–multiple-solutions paradigm. By integrating Parallel Sowing for diverse perspective exploration and Multi-Path Adaptive Guidance for robust instruction, ToST fosters deeper parallel thinking. We also introduce MPSG-Bench, a benchmark with 31K dialogues and a SOLO-based evaluation framework, to systematically assess multi-path teaching. Experiments show ToST significantly improves guidance effectiveness and supports richer exploration of solution paths.

## Limitations

While the ToST framework shows promising performance in guiding LLM-simulated students, several limitations exist.

Firstly, the current experiments and MPSG-Bench benchmark remain focused on mathematical problem-solving (e.g., arithmetic and algebra). Secondly, ToST still depends on the availability of sufficiently reliable PRTs for the target problem domain. Adaptive expansion of tree structure during tutoring is therefore an important direction for reducing upfront construction requirements. Thirdly, ethical concerns, including potential over-reliance on AI-generated guidance and its impact on learners’ autonomy and critical thinking, have not been systematically examined.

From a broader social and ethical perspective, improper deployment of ToST-like systems may risk encouraging passive learning behaviors or reinforcing narrow problem-solving patterns if instructional guidance is overly prescriptive. Ensuring that such systems are used to support, rather than replace, human instruction and learners’ independent reasoning is therefore essential.

Future research will focus on expanding the evaluation to larger cohorts and extending the MPSG-Bench dataset to include other domains. Additionally, efforts to mitigate bias and ensure ethical safeguards will be prioritized.

## Ethical Considerations

While LLMs offer educational potential, they also introduce risks regarding bias, data privacy, and the loss of human oversight. Moreover, their effects on learning outcomes must be continuously and rigorously evaluated to ensure pedagogical effectiveness and to prevent unintended negative consequences. To address these concerns and ensure pedagogical effectiveness, we developed a custom model that prioritizes granular data control. This approach prevents learner information from entering large-scale systems where re-identification risks exist, fostering a secure and trustworthy environment for our intelligent tutoring system.

## References

Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, and 1 others. 2024. Graph of thoughts:

Solving elaborate problems with large language models. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 17682–17690.

John B Biggs and Kevin F Collis. 2014. Evaluating the quality of learning: The SOLO taxonomy (Structure ofthe Observed Learning Outcome). Academic press.

Cameron B Browne, Edward Powley, Daniel Whitehouse, Simon M Lucas, Peter I Cowling, Philipp Rohlfshagen, Stephen Tavener, Diego Perez, Spyridon Samothrakis, and Simon Colton. 2012. A survey of monte carlo tree search methods. IEEE Transactions on Computational Intelligence and AI in games, 4(1):1–43.

Edward Y Chang. 2023. Prompting large language models with the socratic method. In 2023 IEEE 13th annual computing and communication workshop and conference (CCWC), pages 0351–0360. IEEE.

Eason Chen, Jia-En Lee, Jionghao Lin, and Kenneth Koedinger. 2024. Gptutor: Great personalized tutor with large language models for personalized learning content generation. In Proceedings of the Eleventh ACM Conference on Learning@ Scale, pages 539– 541.

Zhendong Chu, Shen Wang, Jian Xie, Tinghui Zhu, Yibo Yan, Jinheng Ye, Aoxiao Zhong, Xuming Hu, Jing Liang, Philip S Yu, and 1 others. 2025. Llm agents for education: Advances and applications. arXiv preprint arXiv:2503.11733.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, and 1 others. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Yuhao Dan, Zhikai Lei, Yiyang Gu, Yong Li, Jianghao Yin, Jiaju Lin, Linhao Ye, Zhiyan Tie, Yougen Zhou, Yilei Wang, and 1 others. 2024. Educhat: A large language model-based conversational agent for intelligent education. In China Conference on Knowledge Graph and Semantic Computing, pages 297–308. Springer.

Edward De Bono. 2016. Parallel thinking. Random House.

DeepSeek-AI. 2025. Deepseek-v3 technical report. Preprint, arXiv:2412.19437.

David Dinucu-Jianu, Jakub Macina, Nico Daheim, Ido Hakimi, Iryna Gurevych, and Mrinmaya Sachan. 2025. From problem-solving to teaching problem-solving: Aligning llms with pedagogy using reinforcement learning. arXiv preprint arXiv:2505.15607.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. 2023. Improving factuality and reasoning in language models through multiagent debate. In Forty-first International Conference on Machine Learning.

Sheila Evans and Malcolm Swan. 2014. Developing students’ strategies for problem solving in mathematics: The role of pre-designed “sample student work”. Educational Designer, 2(7).

Weibo Gao, Qi Liu, Linan Yue, Fangzhou Yao, Rui Lv, Zheng Zhang, Hao Wang, and Zhenya Huang. 2025a. Agent4edu: Generating learner response data by generative agents for intelligent education systems. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, pages 23923–23932.

Xian Gao, Zongyun Zhang, Mingye Xie, Ting Liu, and Yuzhuo Fu. 2025b. Graph of ai ideas: Leveraging knowledge graphs and llms for ai research idea generation. arXiv preprint arXiv:2503.08549.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, and 1 others. 2022. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3.

Daniel Kahneman. 2011. Thinking, fast and slow. Farrar, Straus and Giroux.

Enkelejda Kasneci, Kathrin Seßler, Stefan Küchemann, Maria Bannert, Daryna Dementieva, Frank Fischer, Urs Gasser, Georg Groh, Stephan Günnemann, Eyke Hüllermeier, and 1 others. 2023. Chatgpt for good? on opportunities and challenges of large language models for education. Learning and individual differences, 103:102274.

David A Kolb. 2014. Experiential learning: Experience as the source oflearning and development. FT press.

Jiayu Liu, Zhenya Huang, Tong Xiao, Jing Sha, Jinze Wu, Qi Liu, Shijin Wang, and Enhong Chen. 2024. Socraticlm: Exploring socratic personalized teaching with large language models. Advances in Neural Information Processing Systems, 37:85693–85721.

Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, and 1 others. 2023. The flan collection: Designing data and methods for effective instruction tuning. In International Conference on Machine Learning, pages 22631–22648. PMLR.

João Loula, Benjamin LeBrun, Li Du, Ben Lipkin, Clemente Pasti, Gabriel Grand, Tianyu Liu, Yahya Emara, Marjorie Freedman, Jason Eisner, and 1 others. 2025. Syntactic and semantic control of large language models via sequential monte carlo. In The Thirteenth International Conference on Learning Representations.

Mathematical Association of America. 2024. American invitational mathematics examination (aime) i and ii. Accessed: 2026-01-06.

OpenAI. 2025. Gpt-5. https://openai.com. Large language model accessed via OpenAI API.

Minju Park, Sojung Kim, Seunghyun Lee, Soonwoo Kwon, and Kyuseok Kim. 2024. Empowering personalized learning through a conversation-based tutoring system with student modeling. In Extended Abstracts ofthe CHI Conference on Human Factors in Computing Systems, pages 1–10.

Attila Pásztor, Gyöngyvér Molnár, and Beno Csapó.˝ 2015. Technology-based assessment of creativity in educational context: the case of divergent thinking and its relation to mathematical achievement. Thinking skills and Creativity, 18:32–42.

Francesc Pedro, Miguel Subosa, Axel Rivas, and Paula Valverde. 2019. Artificial intelligence in education: Challenges and opportunities for sustainable development.

Shuo Ren, Daya Guo, Shuai Lu, Long Zhou, Shujie Liu, Duyu Tang, Neel Sundaresan, Ming Zhou, Ambrosio Blanco, and Shuai Ma. 2020. Codebleu: a method for automatic evaluation of code synthesis. arXiv preprint arXiv:2009.10297.

Qwen Team and 1 others. 2024. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2(3).

Rose Wang, Qingyang Zhang, Carly Robinson, Susanna Loeb, and Dorottya Demszky. 2024. Bridging the novice-expert gap via models of decision-making: A case study on remediating math mistakes. In Proceedings ofthe 2024 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 2174–2199.

Tianfu Wang, Yi Zhan, Jianxun Lian, Zhengyu Hu, Nicholas Jing Yuan, Qi Zhang, Xing Xie, and Hui Xiong. 2025. Llm-powered multi-agent framework for goal-oriented learning in intelligent tutoring system. In Companion Proceedings of the ACM on Web Conference 2025, pages 510–519.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, and 1 others. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824– 24837.

Qihao Yang, Yu Yang, Sixu An, Tianyong Hao, and Guandong Xu. 2025. Llm-based collaborative agents with pedagogy-guided interaction modeling for timely instructive feedback generation in taskoriented group discussions. In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25), pages 9972–9980. International Joint Conferences on Artificial Intelligence.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving

with large language models. Advances in neural information processing systems, 36:11809–11822.

Xuesong Zhai, Xiaoyan Chu, Ching Sing Chai, Morris Siu Yung Jong, Andreja Istenic, Michael Spector, Jia-Bao Liu, Jing Yuan, and Yan Li. 2021. A review of artificial intelligence (ai) in education from 2010 to 2020. Complexity, 2021(1):8812542.

Yi Zhan, Qi Liu, Weibo Gao, Zheng Zhang, Tianfu Wang, Shuanghong Shen, Junyu Lu, and Zhenya Huang. 2025. Coderagent: Simulating student behavior for personalized programming learning with large language models. arXiv preprint arXiv:2505.20642.

Boning Zhang, Chengxi Li, and Kai Fan. 2024a. Mario eval: Evaluate your math llm with your math llm–a mathematical dataset evaluation toolkit. Preprint, arXiv:2404.13925.

Liang Zhang, Jionghao Lin, Ziyi Kuang, Sheng Xu, and Xiangen Hu. 2024b. Spl: A socratic playground for learning powered by large language model. In CEUR Workshop Proceedings.

Tong Zheng, Hongming Zhang, Wenhao Yu, Xiaoyang Wang, He Xing, Runpeng Dai, Rui Liu, Huiwen Bao, Chengsong Huang, Heng Huang, and 1 others. 2025. Parallel-r1: Towards parallel thinking via reinforcement learning. In NeurIPS 2025 Workshop on Efficient Reasoning.

## A Implementation Details

All experiments are implemented using Py-Torch 2.8.0 with Python 3.12 on Ubuntu 22.04, and executed with CUDA 12.8 on two virtual GPUs, each equipped with 32 GB memory. QWEN2.5- MATH-7B-INSTRUCT (Team et al., 2024) is instruction-tuned (Longpre et al., 2023) on the MPSG-Bench dataset as the ToST teacher using LoRA (Hu et al., 2022); detailed hyper-parameter settings are reported in Table 3. Student behavior is simulated using GPT-3.5-turbo with six predefined student archetypes; one archetype is randomly sampled per session. Student responses are parsed into PRTs using DeepSeek V3.2.

To assess the effectiveness of ToST, we compare against representative educational tutoring models and strong general LLMs:

SocraticLM (Liu et al., 2024). SocraticLM adopts a Socratic, thought-provoking instructional paradigm to deliver personalized guidance. Its teacher model is trained on the SocraTeach dataset, which is constructed via a Dean–Teacher–Student multi-agent pipeline. In our experiments, we evaluate the officially released checkpoint using the authors’ recommended prompts.

Table 3: Parameters for LoRA Finetuning (left) and ToST (right).
<table><tr><td colspan="2">LoRA Fine-Tuning</td><td colspan="2">ToST</td></tr><tr><td>Parameter</td><td>Value</td><td>Parameter</td><td>Value</td></tr><tr><td>lora rank</td><td>8</td><td>γ</td><td>0.6</td></tr><tr><td>lora alpha</td><td>16</td><td>σ</td><td>0.5</td></tr><tr><td>lora target</td><td>q,k</td><td>δ</td><td>0.1</td></tr><tr><td>lora dropout</td><td>0.1</td><td>α</td><td>0.52</td></tr><tr><td>learning rate</td><td>5e-5</td><td> $\beta$ </td><td>0.39</td></tr><tr><td>weight decay</td><td>0.1</td><td> $\rho$ </td><td>0.05</td></tr><tr><td>train epochs</td><td>3</td><td> $\lambda$ </td><td>0.16</td></tr><tr><td>temperature</td><td>0.95</td><td> $\mu$ </td><td>2</td></tr><tr><td>length penalty</td><td>1.0</td><td> $\theta _ { \mathrm { s w i t c h } }$ </td><td>0.2</td></tr><tr><td>repetition penalty</td><td>1.1</td><td> $\alpha _ { t }$ </td><td>0.4</td></tr><tr><td>trained context length</td><td>6144</td><td> $\beta _ { t }$ </td><td>0.2</td></tr><tr><td>max new tokens</td><td>1024</td><td> $\gamma _ { t }$ </td><td>0.4</td></tr></table>

TutorRL (Dinucu-Jianu et al., 2025). TutorRL is trained via an online reinforcement learning framework that adapts LLMs into tutoring agents through expert-guided student–tutor simulations. We employ the best-performing released checkpoint, which incorporates explicit reasoning tags to support pedagogically structured explanations of student errors.

EduChat (Dan et al., 2024). EduChat is a largescale educational chatbot guided by principles from cognitive psychology and learning sciences. We evaluate both the 8B and 32B versions of the July 2025 released checkpoints, which introduce a “thinking-before-teaching” paradigm to enhance instructional coherence.

General LLMs. We additionally include DeepSeek V3.2 and GPT-5 as strong general baselines. Both models are prompted to act as tutors using standardized instructional prompts, without task-specific fine-tuning. Evaluations are conducted via their official APIs using identical prompting protocols.

For all methods, the maximum number of interaction turns is capped at 10 to ensure fair comparison across models. We use MARIO\_EVAL (Zhang et al., 2024a) to access the correctness of students answer.

## B MPSG-Bench Dataset: Construction and Statistics

This section describes the construction process and statistical characteristics of the proposed dataset within the Multi-Path Socratic Guidance Benchmark (MPSG-Bench). The dataset is designed to support 1PMS instruction and consists of two tightly coupled components: (i) an Enhanced Multi-Path Problem-Solving Collection annotated with explicit Parallel Reasoning Trees (PRTs), and (ii) a Multi-Path Teaching Dialogue Dataset built upon these structured problem representations. Unless otherwise noted, GSM8k and MATH problems are drawn from their official releases, while AIME24 and AIME25 problems are collected from Zheng et al. (Zheng et al., 2025).

## B.1 Enhanced Multi-Path Problem-Solving Collection

The Enhanced Multi-Path Problem-Solving Collection augments existing mathematical problemsolving datasets with explicit PRT annotations. As illustrated in Figure 6, the construction of a PRT for each problem proceeds in three stages. First, Problem-to-Multiple-Solutions (P2MS) generates a natural-language response containing multiple methodologically distinct solution strategies. Second, Text-to-XML (Txt2Xml) extracts individual reasoning paths, annotates them with attributes such as complexity and innovativeness, and organizes them into a structured XML format. Finally, XML-to-Tree (Xml2Tree) parses the XML representation into a hierarchical and operational tree structure that explicitly captures parallel reasoning paths and their intermediate steps.

Table 5 reports statistics of the generated PRTs grouped by source dataset. The results reveal a clear relationship between problem difficulty and tree structure. Relatively simpler datasets (e.g., GSM8k and MATH) admit a broader range of viable solution strategies, resulting in a larger average number of reasoning paths, while their shorter solution chains yield fewer nodes per path. In contrast, more challenging datasets (e.g., AIME25) constrain the solution space, producing fewer alternative paths but substantially deeper trees with a greater number of intermediate reasoning nodes.

## B.2 PRT Construction Cost Analysis

The PRTs used by MPSG-Bench are generated through the automated P2MS–Txt2Xml–Xml2Tree pipeline described above, and online tutoring only parses the learner’s evolving response and applies guidance against the available problem-level structure. Table 4 reports the offline token cost of constructing problem-level PRTs by DeepSeek V3.2.

The results show that PRT construction is a bounded offline preprocessing cost rather than a per-turn tutoring cost. Overall, each problem requires 27,901 input tokens and 5,884 output tokens on average. The split-level variation mainly follows tree structure: AIME25 has the highest token cost because its PRTs are deeper, whereas GSM8k requires fewer tokens because its solution chains are shorter. This suggests that reusable PRT resources are feasible to construct once for a problem set, while also motivating lighter two- or three-path trees and adaptive expansion for domains where full multi-path trees are unnecessarily costly.

Table 4: PRT construction cost analysis.
<table><tr><td>Dataset / Split</td><td># Problems</td><td>Avg. Input Tokens</td><td>Avg. Output Tokens</td></tr><tr><td>GSM8k (Train)</td><td>7,473</td><td>25,761</td><td>5,432</td></tr><tr><td>MATH (Train)</td><td>7,498</td><td>29,447</td><td>6,210</td></tr><tr><td>GSM8k (Test)</td><td>1,319</td><td>25,658</td><td>5,411</td></tr><tr><td>MATH-500</td><td>500</td><td>32,369</td><td>6,826</td></tr><tr><td>AIME24</td><td>360</td><td>30,292</td><td>6,388</td></tr><tr><td>AIME25</td><td>480</td><td>36,786</td><td>7,757</td></tr><tr><td>Overall</td><td>17,630</td><td>27,901</td><td>5,884</td></tr></table>

## B.3 Multi-Path Teaching Dialogue Dataset

Based on the constructed PRTs, we further build a Multi-Path Teaching Dialogue Dataset to support adaptive instructional guidance. Following the multi-agent generation paradigm of SocraticLM (Liu et al., 2024), we employ a pipeline consisting of Student, Teacher, and Expert agents to generate approximately 31k instructional exchanges. The Expert agent refines teacher responses to ensure strict adherence to SOLO taxonomy constraints and enforces alignment with the guidance recommendations produced by the Automatic Path Analyzer in ToST.

Table 6 summarizes the statistics of the Multi-Path Teaching Dialogue Dataset. The dataset is divided into a training set—comprising a Question-Scope-Oriented Single-Round Dialogue Corpus and a Student-Diversity-Oriented Multi-Round Dialogue Corpus—and a held-out Teaching Evaluation Set.

The Question-Scope-Oriented Single-Round Dialogue Corpus focuses on increasing problem diversity. For each problem drawn from GSM8k and MATH (difficulty levels 3–5), we generate a single-round instructional interaction that provides targeted guidance on the student’s current reasoning state. To prevent overfitting to specific learner behaviors, one of six predefined student cognitive archetypes is randomly assigned to each problem instance. This design encourages broad coverage of problem types and solution structures while maintaining controlled interaction length, thereby promoting generalization across diverse problem distributions.

![](images/138aa2708145c197c376fd3ce395514e9bd2cc2cee2bd703f4430f73a287f241.jpg)  
Figure 6: Overview of the Parallel Reasoning Tree (PRT) construction pipeline.

The Student-Diversity-Oriented Multi-Round Dialogue Corpus is designed to increase diversity in student behaviors and instructional strategies. This subset consists of multi-round dialogues constructed on MATH problems of lower difficulty (levels 1–2), which naturally admit extended interaction without excessive solution complexity. For each problem, two distinct student cognitive archetypes are randomly sampled from the same pool of six, and guidance is delivered over multiple dialogue rounds. This setting induces dynamic shifts in reasoning trajectories and instructional focus, enabling the teacher model to learn adaptive path continuation and path switching behaviors in response to evolving student states.

Together, these two corpora provide complementary supervision signals. This design supports the training and evaluation of teacher models capable of delivering adaptive, personalized instruction across diverse student behaviors and reasoning trajectories.

Table 5: Statistics of the Enhanced Multi-Path Problem-Solving Collection in the MPSG-Bench dataset. Problems are grouped by their source datasets. For each subset, we report the number of problems, the average number of reasoning paths per PRT, and the average number of nodes per PRT.
<table><tr><td>Source Dataset</td><td># Problems</td><td>Avg. # Paths</td><td>Avg. # Nodes</td></tr><tr><td>GSM8k (Train)</td><td>7,473</td><td>5.64</td><td>22.57</td></tr><tr><td>MATH (Train)</td><td>7,498</td><td>4.95</td><td>25.80</td></tr><tr><td>GSM8k (Test)</td><td>1,319</td><td>5.48</td><td>22.48</td></tr><tr><td>MATH-500</td><td>500</td><td>4.90</td><td>28.36</td></tr><tr><td>AIME24</td><td>360</td><td>4.76</td><td>26.54</td></tr><tr><td>AIME25</td><td>480</td><td>4.17</td><td>32.23</td></tr><tr><td>Total</td><td>17,630</td><td>5.03</td><td>26.29</td></tr></table>

Table 6: Statistics of the Multi-Path Teaching Dialogue Dataset in the MPSG-Bench dataset. We report the number of underlying problems and the total number of dialogue instances for each subset.

<table><tr><td>Subset</td><td># Problems</td><td># Dialogues</td></tr><tr><td>Question-Scope-Oriented Single- Round Dialogue Corpus</td><td>11.5k</td><td>11.5k</td></tr><tr><td>Student-Diversity-Oriented Multi- Round Dialogue Corpus</td><td>3.5k</td><td>16.5k</td></tr><tr><td>Teaching Evaluation Set</td><td>1.8k</td><td>3.2k</td></tr><tr><td>Total</td><td>16.8k</td><td>31.2k</td></tr></table>

## B.4 Student Archetypes and Dialogue Diversity

Student behaviors are simulated using six archetypes with distinct problem-solving preferences, including algebraic, logical, combinatorial, trial-and-error, equation-oriented, and nonspecialized strategies. These archetypes are grounded in Kolb’s Learning Styles theory (Kolb, 2014), which characterizes learners along two dimensions: abstract conceptualization versus concrete experience, and reflective observation versus active experimentation. While Kolb’s framework defines four general learning styles, it remains coarse-grained for modeling domain-specific reasoning processes.

To better capture the diversity of mathematical problem-solving behaviors, we operationalize and extend these dimensions into six fine-grained archetypes. For example, algebraic and equationoriented students emphasize abstract conceptualization, logical students exhibit reflective and structured reasoning, while trial-and-error and combinatorial students reflect stronger tendencies toward active experimentation. The non-specialized archetype represents learners without a dominant strategic preference.

To promote meaningful guidance interactions, student responses are repeatedly sampled (up to ten attempts) to increase the likelihood of incorrect initial answers. Dialogue diversity is further encouraged by combining single-round and multi-round interactions, randomly sampling student archetypes per problem, and balancing guidance that continues along the current reasoning path with guidance that explicitly triggers path switching. We adopt GPT-3.5-turbo to simulate students, as it exhibits relatively weaker problem-solving performance while maintaining strong role-playing and instructionfollowing capabilities.

Table 7: Consistency of solution tree parsing across APIs. Agreement results on sampled MPSG-Bench stu dent responses demonstrate reliability of PRT parsing.
<table><tr><td>Pairwise Comparison</td><td>PDC (%)</td><td>NMR (%)</td></tr><tr><td>DeepSeek V3.2 vs. GPT-5</td><td>99.48</td><td>98.78</td></tr><tr><td>DeepSeek V3.2 vs. Gemini-2.5-pro</td><td>99.68</td><td>96.92</td></tr><tr><td>GPT-5 vs. Gemini-2.5-pro</td><td>98.52</td><td>97.36</td></tr><tr><td>Average</td><td>99.23</td><td>97.69</td></tr></table>

## C Cross-Domain Case Studies for PRT Generalizability

This section provides conceptual case studies beyond mathematics to clarify the intended scope of PRT-based tutoring. These examples are not benchmark-scale experiments; they illustrate when ToST can plausibly transfer: the task should admit explicit multi-step reasoning, multiple valid strategies or hypotheses, checkable intermediate states, and a shared target answer or evidence-based conclusion.

Physics. For an Atwood-machine problem with two masses connected over a frictionless pulley, a PRT can encode at least four valid paths: a system-level Newton’s second-law solution, individual free-body equations, a work-energy derivation, and a Lagrangian derivation. The root contains the shared givens $( m _ { 1 } = 5 \mathrm { k g } , m _ { 2 } = 3 \mathrm { k g }$ $g = 9 . 8 \mathrm { m } / \mathrm { s } ^ { 2 } )$ , while leaves converge to the same acceleration, $2 . 4 5 \mathrm { m } / \mathrm { s } ^ { 2 }$ . Reusable nodes include quantities such as total mass, net force, acceleration direction, and shared energy terms. Thus, if a learner stalls while solving simultaneous tension equations, MPAG can redirect the learner to the system-level force path while preserving correctly established physical quantities.

Coding. For computing the n-th Fibonacci number, a PRT can represent naive recursion, bottom-up iteration, memoized recursion, matrix exponentiation, and a closed-form formula. Here, paths differ not by final value but by algorithmic structure, complexity, and implementation constraints. Reusable nodes include base cases, recurrence relations, variable states, cache entries, and test cases. Parallel Sowing can elicit whether the learner thinks recursively, iteratively, or through dynamic programming, while path switching can preserve correct base-case reasoning when guiding the learner from an inefficient recursive program to a more efficient iterative or memoized one.

Table 8: Teacher/expert rubric for blinded dialogue review. All dimensions use a 5-point Likert scale.
<table><tr><td>Dimension</td><td>Rating anchors (1 → 5)</td></tr><tr><td>Diagnostic ac- curacy</td><td>Incorrect diagnosis or misses the student&#x27;s actual issue → pre- cisely identifies the student&#x27;s misconception or latent valid</td></tr><tr><td>ralness</td><td>path. Guidance natu- Awkward, abrupt, or overly scripted tutoring language → smooth, conversational, and pedagogically natural tutoring language.</td></tr><tr><td>Preservation of correct progress</td><td>Discards or ignores correct par- tial reasoning → explicitly pre- serves and builds on what the student already got right.</td></tr><tr><td>ness</td><td>Switch natural- Path transition is abrupt or confusing → transition is well-motivated and smoothly bridged through reusable</td></tr><tr><td>fulness</td><td>intermediate steps. Overall help- Guidance is unlikely to help the student progress → guidance is highly useful for moving the student forward.</td></tr><tr><td>Cognitive load appropriate- ness</td><td>Overwhelming or under- informative relative to student state → well-calibrated support with appropriate challenge and pacing.</td></tr></table>

Scientific reasoning. For a population-decline analysis, a PRT can organize competing causal hypotheses, such as habitat loss, invasive predators, disease, and climate-induced resource scarcity. Each path begins from the same observation, e.g., a 60% population drop over five years, and develops a different evidence chain using land-use maps, ship-arrival logs, pathology reports, or precipitation records. The leaves need not be numeric answers; they can be evidence-weighted conclusions or a multi-factor synthesis. Reusable nodes include correctly interpreted data sources and shared causal constraints, enabling the teacher to preserve valid observations while redirecting the learner toward a better-supported hypothesis.

Across these cases, the transferable component is not the mathematical content of MPSG-Bench, but the PRT abstraction: a structured set of alternative reasoning paths with verifiable intermediate states and reusable nodes.

Table 9: Student post-interaction questionnaire. All dimensions use a 5-point Likert scale.
<table><tr><td>Dimension</td><td>Rating anchors (1 → 5)</td></tr><tr><td>fulness</td><td>Perceived help- The tutor was not helpful for solving the problem → the tutor</td></tr><tr><td>Clarity</td><td>was very helpful. The tutor&#x27;s guidance was hard to understand → the tutor&#x27;s</td></tr><tr><td>Understanding support</td><td>guidance was very clear. The interaction did not improve my understanding → the inter- action substantially improved my understanding.</td></tr><tr><td>Willingness to continue</td><td>I would not want to continue with this tutoring style in the fu-</td></tr><tr><td>exploring</td><td>ture → I would like to continue exploring with this tutor. Perceived bur- The interaction felt overly</td></tr><tr><td>den</td><td>stressful or cognitively heavy → the interaction felt well- paced and manageable.</td></tr></table>

## D SOLO Taxonomy for Cognitive Structure Assessment

The Structure of Observed Learning Outcomes taxonomy (SOLO) (Biggs and Collis, 2014) is a hierarchical framework to assess the structural complexity of student understanding. Rather than focusing on surface correctness, SOLO characterizes how well learners organize, integrate, and abstract knowledge. It categorizes learning outcomes into five progressively sophisticated levels:

Prestructural (S1). The learner demonstrates little or no understanding of the problem, often responding with irrelevant or incorrect information.

Unistructural (S2). The learner identifies a single relevant aspect or solution strategy but applies it in a shallow or incomplete manner.

Multistructural (S3). The learner recognizes multiple relevant strategies or facts but treats them independently, without coherent integration.

Relational (S4). The learner integrates multiple solution paths into a coherent reasoning structure, demonstrating meaningful connections among ideas.

Extended Abstract (S5). The learner generalizes and abstracts beyond the immediate problem, applying principles at a theoretical or meta-cognitive level.

In this work, our evaluation focuses on the first four SOLO levels (S1–S4), as extended abstract reasoning (S5) is rarely exhibited in short-form tutoring dialogues and lies beyond the scope of the targeted student proficiency levels.

## E Prompt Templates

This section presents the prompt templates employed in ToST for (i) PRT parsing (Figure 9) and (ii) instructional guidance under two modes: path continuation and path switching (Figure 10).

## F Human Pilot Evaluation Materials

This section provides supplementary materials for the human pilot evaluation introduced in Section 6.3, including rating rubrics, participant instructions, evaluation interface screenshots, and detailed full results for all tracks.

## F.1 Teacher/Expert Rubric

Table 8 presents the full six-dimension rubric used in the blinded expert evaluation (Track A). Each dimension is operationalized as a 5-point Likert scale with explicit behavioral anchors at both endpoints, ensuring that annotators apply consistent standards across dialogue samples from different systems.

## F.2 Student Rubric

Table 9 shows the five-dimension post-interaction questionnaire administered to student participants (Track A). Items assess perceived helpfulness, clarity, understanding support, willingness to continue exploring, and cognitive burden, capturing both the effectiveness and the affective quality of the tutoring experience.

## F.3 Blinding, Annotation Protocol, and Statistics

Expert participants. Seven graduate students and university instructors (4 graduate students in mathematics education or NLP, 3 university-level mathematics instructors) were recruited from top-tier universities. All had prior experience in mathematics tutoring or education research. Each expert independently rated 20 randomly sampled dialogue sessions per system (100 sessions in total).

Student participants. Ten undergraduate students (STEM majors) participated in the pilot student study. A counterbalanced within-subjects design was adopted: each participant completed one tutoring session with each of the five systems in a randomized order, with a five-minute break between sessions to reduce fatigue.

Table 10: Teacher/expert blinded evaluation results (Track A). Expert raters evaluated anonymized tutoring dialogues from five systems on six pedagogical dimensions using a 5-point Likert scale $( \mathrm { m e a n } \pm \mathrm { s t d } )$ . κ: Fleiss inter-rater agreement (substantial >0.6). Student self-report results are in Table 2. Bold: best per column.
<table><tr><td>System</td><td></td><td>Diag. Acc.↑ Guidance Nat.↑ Pres. Correct↑ Switch Nat.↑ Overall Help.↑ Cog. Load↑</td><td></td><td></td><td></td><td></td><td>κ</td></tr><tr><td>SocraticLM</td><td> $\overline { { 4 . 0 8 \pm 0 . 0 4 9 } }$ </td><td> $3 . 2 6 { \pm } 0 . 1 5 3$ </td><td> $2 . 8 5 { \pm } 0 . 0 2 4$ </td><td> $2 . 5 2 { \pm } 0 . 1 0 3$ </td><td> $\overline { { 2 . 2 3 \pm 0 . 0 2 9 } }$ </td><td> $\overline { { 3 . 8 1 \pm 0 . 1 0 7 } }$ </td><td>0.73</td></tr><tr><td>EduChat-R1-32b</td><td> $4 . 3 5 { \pm } 0 . 1 4 8$ </td><td> $3 . 5 1 { \pm } 0 . 1 7 4$ </td><td> $3 . 6 9 { \pm } 0 . 0 2 2$ </td><td> $3 . 3 7 { \pm } 0 . 2 1 6$ </td><td> $2 . 3 4 { \pm } 0 . 0 1 0$ </td><td> $3 . 3 7 { \pm } 0 . 2 4 4$ </td><td>0.76</td></tr><tr><td>TutorRL-7B</td><td> $4 . 8 0 { \pm } 0 . 0 7 1$ </td><td> $3 . 1 7 { \pm } 0 . 0 0 1$ </td><td> $3 . 9 6 { \pm } 0 . 2 3 8$ </td><td> $4 . 1 5 { \pm } 0 . 2 2 8$ </td><td> $3 . 3 5 { \pm } 0 . 2 3 1$ </td><td> $3 . 6 6 { \pm } 0 . 0 4 6$ </td><td>0.74</td></tr><tr><td>DeepSeek V3.2</td><td> $\pm . 9 6 \pm 0 . 1 4 9$ </td><td> $4 . 0 9 { \pm } 0 . 0 4 5$ </td><td> $4 . 5 7 { \pm } 0 . 0 2 6$ </td><td> $4 . 6 1 { \pm } 0 . 0 7 4$ </td><td> $3 . 8 9 { \pm } 0 . 1 9 0 $ </td><td> $\pm . 7 8 { \pm } 0 . 2 1 1$ </td><td>0.70</td></tr><tr><td>ToST (Ours)</td><td> $4 . 9 4 { \pm } 0 . 0 9 0$ </td><td> $\mathbf { 4 . 1 9 } \pm 0 . 2 1 9$ </td><td> $\pm . 7 7 { \pm } 0 . 2 0 3$ </td><td> $\mathbf { 4 . 8 0 \dot { = } 0 . 0 0 8 }$ </td><td> $4 . 4 2 { \pm } 0 . 2 1 8$ </td><td> $4 . 6 6 { \pm } 0 . 1 8 1$ </td><td>0.75</td></tr></table>

Table 11: Independent LLM judge results (Track B) and human–LLM agreement. An independent generalpurpose LLM rated anonymized dialogues on four dimensions (mean ± std). Win Rate: fraction of samples in which the system is preferred over all others. ρ<sub>T</sub> / ρ<sub>S</sub>: Spearman rank correlation with teacher / student scores. Bold: best per column.
<table><tr><td>System</td><td>Helpfulness↑</td><td>Clarity↑</td><td></td><td>Diag. Correct↑ Switch Nat.↑</td><td>Win Rate↑</td><td> $\rho _ { T } \uparrow$ </td><td> $\rho _ { S } \uparrow$ </td></tr><tr><td>SocraticLM</td><td> $\overline { { 3 . 4 5 { \pm 0 . 3 0 7 } } }$ </td><td> $3 . 7 8 \pm 0 . 2 3 2$ </td><td> $\overline { { 3 . 1 1 \pm 0 . 1 3 6 } }$ </td><td> $\overline { { 2 . 8 0 { \pm 0 . 8 2 7 } } }$ </td><td> $\overline { { 0 . 3 4 \pm 0 . 0 3 1 } }$ </td><td>0.82</td><td>0.82</td></tr><tr><td>EduChat-R1-32b</td><td> $3 . 1 9 { \pm } 0 . 2 9 6$ </td><td> $3 . 6 0 { \scriptstyle \pm 0 . 1 6 7 }$ </td><td> $3 . 5 7 { \pm } 0 . 0 1 5$ </td><td> $2 . 9 6 { \pm } 0 . 6 9 3$ </td><td> $0 . 2 4 { \pm } 0 . 0 8 4$ </td><td>0.71</td><td>0.78</td></tr><tr><td>TutorRL-7B</td><td> $4 . 3 9 { \pm } 0 . 2 6 0$ </td><td> $4 . 1 7 { \pm } 0 . 3 0 4$ </td><td> $3 . 6 4 { \pm } 0 . 1 8 8$ </td><td> $3 . 0 0 { \pm } 0 . 8 5 5$ </td><td> $0 . 6 5 { \pm } 0 . 0 9 6$ </td><td>0.84</td><td>0.81</td></tr><tr><td>DeepSeek V3.2</td><td> $4 . 6 2 { \pm } 0 . 0 0 8$ </td><td> $4 . 5 1 { \pm } 0 . 1 6 0$ </td><td> $4 . 6 4 { \pm } 0 . 1 8 6$ </td><td> $3 . 9 1 { \pm } 0 . 8 0 3 $ </td><td> $0 . 7 7 { \pm } 0 . 0 2 1$ </td><td>0.80</td><td>0.86</td></tr><tr><td>ToST (Ours)</td><td> $\pm . 7 0 { \pm } 0 . 1 0 7$ </td><td> $\mathbf { 4 . 6 9 } \pm 0 . 2 6 0$ </td><td> $\pm . 9 \mathbf { 0 } \pm 0 . 1 1 9$ </td><td> $\mathbf { 4 . 1 7 \pm 0 . 9 4 0 }$ </td><td> $\mathbf { 0 . 8 6 { \scriptstyle \pm 0 . 0 6 6 } }$ </td><td>0.74</td><td>0.87</td></tr></table>

Blinding. System identities are hidden during annotation. Dialogue samples are randomized and presented as System A, System B, and (when needed) System C. Annotation unit. Each unit contains the problem statement, optional expert reference solution, student dialogue history, and one tutoring continuation or full tutoring session depending on the study condition. Preference task. When pairwise comparison is enabled, annotators select the more helpful tutoring response and may optionally provide a short free-text reason. Statistics. Likert ratings are reported as mean ± standard deviation; pairwise preferences as win rate; and inter-rater agreement as Fleiss’ κ computed across expert raters. For the student study, κ reflects crossparticipant consistency in subjective ratings rather than annotation agreement, and should be interpreted accordingly. Ethics. Participants receive a short consent form stating the study purpose, expected duration, anonymization policy, and the right to withdraw at any time.

## F.4 Evaluation Interface Screenshots

Figure 7 shows the blinded expert-review interface presented to teacher and teaching-assistant annotators, where anonymized tutoring dialogues from two systems are displayed side-by-side for Likert scoring. Figure 8 shows the learner-facing interface used in the pilot student study, where participants interact with an anonymized tutoring system via a chat-style window before completing the postsession questionnaire. Both interfaces are designed to conceal system identities throughout the evaluation session.

## F.5 Detailed Evaluation Results

Table 2 in the main text reports the Track A learner self-report results across five subjective dimensions. Here, Table 10 provides the complementary blinded teacher/expert evaluation, and Table 11 presents Track B independent LLM judge scores along with human–LLM rank-correlation with teacher and student ratings. Taken together, these results show that ToST consistently achieves the highest or near-highest ratings across human participants and the independent LLM judge, with strong agreement between human raters and the LLM judge $( \rho _ { T } , \rho _ { S } > 0 . 7 $ for all systems).

The expert ratings further qualify the benchmark results. ToST is not uniformly best on every teacher-rated dimension: DeepSeek V3.2 is marginally higher on diagnostic accuracy and cognitive load appropriateness. For cognitive load, this likely reflects the tradeoff between directness and structured exploration. DeepSeek V3.2 tends to give more linear, answer-oriented guidance, which can feel lower-friction to expert raters because the dialogue asks the learner to make fewer pathmanagement decisions. ToST explicitly maintains multi-path state, preserves correct intermediate reasoning, and guides path transitions; these operations introduce more instructional structure and active reasoning demands, so they can be judged slightly less lightweight even when pedagogically useful. However, ToST receives the strongest ratings on guidance naturalness, preservation of correct intermediate reasoning, switch naturalness, and overall helpfulness, which are the dimensions most directly tied to multi-path tutoring control.

## G Parser Agreement Analysis of Parallel Reasoning Tree Parsing

As the PRT forms the foundation of our framework, we evaluate its consistency by cross-validating the parsing process. We sampled one-eighth of the student responses (332 samples) from the MPSG-Bench test set and independently re-parse them using DeepSeek V3.2, GPT-5, and Gemini-2.5-pro with an identical prompt (as shown in Appendix E). Consistency is measured via Path Diversity Consistency (PDC; Jaccard similarity over paths) and Node Match Rate (NMR; embedding-based semantic alignment). As shown in Table 7, average PDC and NMR reach 99.23% and 97.69%, respectively, indicating high parser agreement across model choices. We emphasize that agreement does not guarantee semantic correctness; rather, it suggests relative stability of the parsing pipeline under alternative LLM backends.

![](images/86ddf629a325b5cc284677227fb28e21a3a22ddf96959e6174bcbdb01f795849.jpg)  
Figure 7: Expert-review interface. The blinded rating interface presents anonymized tutoring dialogues side-byside and collects six-dimension Likert scores with optional free-text rationale.

![](images/cf4c8ac2c2411e637fea99533bb6aa5dd9cafcae2c382a786673f140491276c9.jpg)  
Figure 8: Student-study interface. The learner-facing page displays the tutoring dialogue in a chat format and collects five-dimension post-session Likert ratings with optional free-text reflection.

![](images/c3107694b2b32183ea716f752ca0e7543034c8a780bd34e6ddfaa197ee903f04.jpg)  
Figure 9: Prompt template for Parallel Reasoning Tree parsing. {problem\_statement}, {expert\_tree}, {current\_path\_context}, and {response} denote placeholders for the target mathematical problem, the reference solution tree provided by the MPSG-Bench dataset (shown in the standard Solution Tree format), the student’s current problem-solving trajectory, and the raw student response, respectively.

![](images/449668d0ad58873faeca6d0d0f80f7bf3089468e0aa20f9b1744d2de169705b0.jpg)  
Figure 10: Prompt template for instructional guidance under dual pedagogical modes. {problem\_statement}, {analysis}, and {advice} denote placeholders for the target problem, the path-level analysis report and the recommended instructional guidance provided by the Automatic Path Analyzer, respectively.