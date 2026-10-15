# Recovering Reliable Supervision via Critique-Guided On-Policy Distillation

Kaixiang Zhao<sup>1</sup> Tianrun Yu<sup>1</sup> Shangzhe Li<sup>2</sup> Yuxiao Yang<sup>2</sup>

Amanda Hughes<sup>1</sup> Weitong Zhang<sup>2</sup> Taylor W. Killian<sup>1</sup>

<sup>1</sup>Brigham Young University <sup>2</sup>University of North Carolina at Chapel Hill kzhao2@byu.edu tkillian@cs.byu.edu

## Abstract

On-policy distillation (OPD) trains a student model on its own rollouts with token-level supervision from a teacher, assuming that the teacher can reach a correct answer from any prefix the student writes. We call this property the recoverability of the teacher and show it fails on incorrect rollouts: the teacher follows student errors, although it recognizes them. We propose CriticOPD, using error recognition to restore recoverability. The teacher critiques an incorrect rollout, allowing the student to rewrite its solution following the error. The resulting output, including the critique as privileged information, is scored by the teacher while the student uses the revision without the critique for training. On two Qwen3 students and six mathematical reasoning benchmarks, CriticOPD improves OPD, with the largest gains on competition problems. Code is available at github.com/kzhao5/CriticOPD.

## 1 Introduction

On-policy distillation (OPD) trains a student model on its own rollouts with token-level supervision from a teacher (Agarwal et al., 2024; Gu et al., 2024). OPD addresses a weakness of offline distillation. In offline distillation, the student trains on sequences the teacher generates, whereas the student at inference continues from prefixes that it generates itself, so the distribution of prefixes in training differs from the distribution of prefixes that the student visits at inference. We call the distance between these two distributions the occupancy mismatch. Under this mismatch, the errors of the student may be underrepresented in training, and once the student errs at inference, it may enter a prefix that it has rarely seen during training, allowing errors to accumulate (Ross et al., 2011; Agarwal et al., 2024). OPD reduces the occupancy mismatch by training on sequences that the student generates, with the next-token distribution of the teacher at each prefix as supervision. However, the effectiveness of this supervision in correcting errors depends on the teacher’s ability to recover from the student’s prefixes. When the teacher follows the student’s errors rather than correcting them, its next-token distribution may no longer provide a reliable corrective target.

![](images/a5b644b5dcc3375d28c9213d8d2d6e10e7ae719e69f36a5a36f55905c4933b4f.jpg)  
Figure 1: Where the supervision of each method falls after the student errs, and who writes it.

Prior work has found that teacher supervision can become less reliable as the prefix of the student grows longer (Liu et al., 2026b; Zhou et al., 2026; Li et al., 2026). Existing methods respond in two ways (Figure 1). The first reduces supervision. It truncates the later part of long responses (Yang et al., 2026c; Zhang et al., 2026a) or stops rollouts or supervision early (Zhou et al., 2026; Liu et al., 2026a). The second replaces the sequence. It lets the teacher take over a span of the generation (Xu et al., 2026) or regenerate from an earlier context that the teacher judges reliable (Wang et al., 2026a). Both obtain more reliable supervision by restricting the states receiving supervision or modifying training trajectories, and thereby introduce more occupancy mismatch between training and inference than OPD. Reducing supervision leaves untrained the later states that the student reaches at inference, and replacing the sequence trains the student on sequences that the teacher writes and the student rarely generates. These methods largely avoid unreliable teacher supervision rather than improving its corrective value on erroneous student prefixes. In this work, we aim to make teacher supervision on student-generated continuations more reliable while introducing as little additional occupancy mismatch as possible.

![](images/e781428d9682e7a848ffbf359dc15089a776ed5676fc1cb1d6af91d95410c2b6.jpg)  
Figure 2: Overview of CriticOPD. The student rewrites an incorrect rollout from the last error that the teacher critiques. The teacher scores the rewrite with the critique, and the student learns from it without the critique.

We measure teacher reliability through recoverability, a notion from imitation learning (Swamy et al., 2021), which we define as the teacher’s accuracy when continuing from a student prefix. On incorrect rollouts, recoverability declines sharply even on problems the teacher can solve. The teacher often follows the student’s errors, with up to half of its failed continuations ending in the student’s incorrect answer. Yet its ability to identify errors remains largely intact. This reveals an asymmetry: the teacher may fail to continue correctly but still recognize errors.

Motivated by this observation, we propose Critic-OPD. Given an incorrect rollout and the reference answer, the teacher identifies reasoning errors and provides corrective feedback. The student rewrites its solution from the last error using this feedback. The teacher then scores the rewrite with the critique in context, while the student learns without it. The critique serves as privileged information during training (Vapnik and Vashist, 2009; Lopez-Paz et al., 2016), enabling the student to learn corrective behavior without feedback at inference. Unlike teacher takeover, CriticOPD keeps all supervised tokens student-generated while improving the recoverability of the revised trajectories.

## 2 Measuring the Reliability of Teacher Supervision

This section measures the reliability of teacher supervision through recoverability and examines where and why OPD supervision becomes unreliable. Let $h _ { t } = ( x , y _ { < t } )$ denote a student-generated prefix, and let $r ( y ) \in \{ 0 , 1 \}$ indicate whether a complete solution y is correct. We define teacher recoverability as its expected accuracy when continuing from $h _ { t } , \Lambda ^ { T } ( h _ { t } ) = \mathbb { E } _ { y \sim \pi _ { T } ( \cdot | h _ { t } ) } [ r ( y ) ]$ , where $y$ denotes the completed solution. Student recoverability $\Lambda ^ { S } ( h _ { t } )$ is defined analogously using $\pi _ { \theta }$

$\Lambda ^ { T }$ measures the reliability of the teacher’s continuation policy as a corrective target. Under the sampled reverse-KL objective, a student-sampled token $y _ { t }$ receives the advantage

$$
A _ { t } = \log \pi _ { T } ( y _ { t } \mid h _ { t } ) - \log \pi _ { \theta } ( y _ { t } \mid h _ { t } ) ,\tag{1}
$$

which is positive when the teacher assigns y<sub>t</sub> a higher probability than the student. The OPD target at each prefix is therefore $\pi _ { T } ( \cdot \mid h _ { t } )$ . A student that perfectly imitates the teacher from $h _ { t }$ onward would achieve accuracy $\Lambda ^ { T } ( h _ { t } )$ . Thus, low recoverability indicates that the teacher’s continuation policy is an unreliable corrective target, even though individual tokens may still be useful.

We measure both quantities using a Qwen3-1.7B student after 80 steps of OPD, the Qwen3-4B-Instruct-2507 teacher, and DAPO-Math-17k problems. The student generates one rollout for each of 900 problems, from which we randomly select 150 correct and 150 incorrect rollouts. We cut each rollout at the empty prefix and the seven paragraph boundaries nearest to $1 / 8 , \ldots , 7 / 8$ of its length. From each cut, the teacher and student each generate 16 continuations to estimate $\Lambda ^ { T }$ and $\Lambda ^ { S }$

To distinguish prefix effects from problem difficulty, we divide incorrect rollouts by the success rate of the teacher on the original problem, which is $\Lambda ^ { T }$ at the empty prefix estimated from 16 continuations: 85 come from problems with high initial teacher success $( \Lambda ^ { T } \geq 0 . 5 ;$ solid lines in Figure 3(b)), and 65 from problems with low initial success (dashed lines). Appendix B gives details and repeats the analysis with Qwen3-0.6B.

![](images/0b8f43e2d2f3231728c5aa0ed00a60ea49eebb31fcd1429fce18b025340251a8.jpg)  
Figure 3: Teacher $( \Lambda ^ { T } )$ and student $( \Lambda ^ { S } )$ continuation accuracy along (a) correct and (b) incorrect rollouts.

![](images/26c54d020335207e18e9ac6c0eee26726ba24e9b20785fcebe10f1119a9e27b6.jpg)  
Figure 4: (a) Outcomes of teacher continuations. (b) Valid error-flag rate across teacher-recoverability bins.

## 2.1 Recoverability Declines Along Incorrect Rollouts

The teacher retains recoverability on correct rollouts. On correct rollouts (Figure 3(a)), $\Lambda ^ { T }$ stays at or above 0.93, while $\Lambda ^ { S }$ rises from 0.81 to 0.94. Thus, prefix growth alone does not explain the decline in teacher accuracy.

Incorrect rollouts reduce teacher recoverability. On problems with high initial teacher success (solid lines in Figure 3(b)), $\Lambda ^ { T }$ falls from 0.82 to 0.29, and $\Lambda ^ { S }$ from 0.34 to 0.10, consistent with prior observations (Liu et al., 2026b; Zhou et al., 2026; Li et al., 2026). Together with the control in Figure 3(a), this suggests that the decline is associated with erroneous student prefixes rather than merely their length or problem difficulty. On problems with low initial teacher success (dashed lines), $\Lambda ^ { T }$ falls from 0.14 to 0.05, indicating that limited teacher capability is another source of unreliable supervision. Thus, OPD faces two difficulties: erroneous student prefixes can mislead an otherwise capable teacher, while some problems exceed the teacher’s own ability.

More than half of the tokens in incorrect rollouts receive supervision at low-recoverability prefixes. We assign each token to its preceding cut point and group tokens by $\Lambda ^ { T }$ at that cut. Among tokens in incorrect rollouts, 53% correspond to prefixes with $\Lambda ^ { T } < 0 . 2$ , while only 21% correspond to $\Lambda ^ { T } \geq 0 . 8$ . Since incorrect rollouts account for 61% of training tokens, approximately one third of all training tokens receive supervision at prefixes from which the teacher rarely continues correctly.

## 2.2 The Teacher Follows the Error but Still Recognizes It

We next examine why the teacher loses recoverability and whether it can still recognize errors. Both tests use the 85 incorrect rollouts from problems with high initial teacher success in Section 2.1, without modifying the original trajectories.

First, we examine the final answers of teacher continuations from each cut. Second, we divide each nonempty prefix into numbered paragraphs and ask the teacher to identify its errors using our critique prompt (Appendix A). Unlike the critic in Section 3, the teacher receives no reference answer in this test. Qwen3-8B in thinking mode then judges each flagged error with access to the reference answer, labeling it as a new error, a propagated error, a restatement of an earlier error, or not an error. We count the first three categories as valid (Appendix B). This measures the validity of flagged errors rather than error-detection recall.

The teacher often inherits the student’s erroneous conclusions. As the prefix grows longer (Figure 4(a)), the fraction of correct teacher continuations falls from 0.82 to 0.29, while the fraction ending with the student’s incorrect answer rises from 1% to 35%, or from 8% to 50% of failed continuations. Since the teacher rarely produces this answer from the empty prefix, these results suggest that it often follows errors in the student prefix rather than failing solely because it cannot solve the original problem.

Error recognition precision remains stable as recoverability declines. Across five bins of $\Lambda ^ { T }$ (Figure 4(b)), mean recoverability ranges from 0.07 to 0.93, while the fraction of valid error flags remains between 73% and 81%. This fraction is 73% at $\Lambda ^ { T } < 0 . 2$ and 74% at $\Lambda ^ { T } \geq 0 . 8$ . Thus, teacher continuation accuracy declines sharply, but critique precision remains stable.

## 3 CriticOPD: Restoring Recoverability with Critique

Section 2 shows that after the student errs, the teacher may fail to continue correctly but can still recognize the errors. We therefore build training sequences that the student writes from the states that it reaches at inference, which adds little occupancy mismatch, and on which the teacher supervision is reliable. Section 3.1 builds these sequences and measures the recoverability of the teacher on them, and Section 3.2 gives the training objective.

## 3.1 Critique and Rewrite

Trigger. The student generates a rollout y for each problem x with the prompt $\mathcal { P } _ { \mathrm { t r a i n } }$ . Incorrect rollouts and those reaching the length limit without an answer are sent for critique and rewriting. Correct rollouts are trained with standard OPD.

Critique. The teacher serves as the critic. It receives the problem, the reference answer, and the solution of the student in numbered paragraphs, and it lists the errors in the solution (prompts $\mathcal { P } _ { \mathrm { c r i t i c } }$ and $\mathcal { P } _ { \mathrm { c r i t i c - u s e r } }$ in Appendix A.1). Each error contains the index of its paragraph, the erroneous sentence, and a short fix. We combine these errors and the reference answer into the critique c. The reference answer lets the teacher critique problems that it cannot solve. If the critique contains no complete error, the rollout is trained with standard OPD.

Cut point. The student keeps its prefix through the paragraph of the last error in the critique and discards the rest. The discarded part starts at position τ , so the retained prefix is $h _ { \tau } = ( x , y _ { < \tau } )$ This prefix contains all errors in the critique, so the rewrite starts from the erroneous states where teacher recoverability declines (Section 2).

Rewrite. The student appends the critique c to the retained prefix with the prompt $\mathcal { P } _ { \mathrm { f b } }$ and generates a new suffix, which gives the revised rollout y˜. The tokens before τ are unchanged, $\tilde { y } _ { < \tau } = y _ { < \tau } ,$ , and the student samples the suffix with the critique in its context. Unlike teacher takeover, the student writes every token of the training sequence. In Figure 5, the critique identifies two errors, and the student rewrites from the last one to reach the correct answer.

Recoverability on the rewrite. We test whether the teacher is more reliable on the rewrite than on the original rollout. As in Section 2, we estimate recoverability as the fraction of correct answers among eight continuations from each prefix. The test uses the student after 80 steps of OPD and the 161 incorrect rollouts of Appendix B.4. We critique and cut each rollout as CriticOPD does, and the student writes a rewrite with the critique. This student is not trained with CriticOPD, so the rewrite differs from the original rollout only through the critique. Figure 6 compares recoverability after the cut point on the two sequences. On the original rollout, the teacher continues from the prefix of the student, and on the rewrite, the teacher continues with the critique in its context. At 75% of the continuation, the recoverability of the teacher is 0.14 on the original rollout and 0.67 on the rewrite, and the success rate of the student without the critique rises from 0.03 to 0.16.

## 3.2 Training Objective

Standard OPD. For correct rollouts and rollouts without a valid critique, we minimize the reverse KL $\begin{array} { r } { \mathcal { L } _ { \mathrm { O P D } } ( y ) \ = \ \sum _ { t = 1 } ^ { | y | } \mathrm { K L } \big ( \pi _ { \theta } ( \cdot \ | \ h _ { t } ) \| \pi _ { T } ( \cdot \ | } \end{array}$ $h _ { t } ) \big )$ at each student-generated prefix $h _ { t }$ with the sampled-token advantage in Equation (1).

Feedback distillation. For rollouts with a valid critique, we train on the revised sequence y˜. Let $\tilde { h } _ { t } = ( x , \tilde { y } _ { < t } )$ denote its prefix without the critique. The teacher scores the rewrite with c in its context, while the student does not see it:

$$
\mathcal { L } _ { \mathrm { f b } } ( \tilde { y } ; c ) = \sum _ { t = \tau } ^ { \lfloor \tilde { y } \rfloor } \mathrm { K L } \left( \pi _ { \theta } ( \cdot  { \mid } \tilde { h } _ { t } )  { \parallel } \pi _ { T } ( \cdot  { \mid } \tilde { h } _ { t } , c ) \right)\tag{2}
$$

Only the rewritten suffix receives supervision, since the retained prefix contains errors at states where the teacher has low recoverability. As in OPD, each suffix token receives the sampled-token advantage

$$
\begin{array} { r } { A _ { t } ^ { \mathrm { f b } } = \log \pi _ { T } ( \tilde { y } _ { t } \mid \tilde { h } _ { t } , c ) - \log \pi _ { \theta } ( \tilde { y } _ { t } \mid \tilde { h } _ { t } ) . } \end{array}\tag{3}
$$

The suffix is sampled with the critique, whereas the log-probabilities of the student are computed without it, so this update is a surrogate rather than an unbiased gradient of Equation (2).

Overall objective. CriticOPD minimizes $\mathcal { L } _ { \mathrm { o u r s } } =$ $\mathcal { L } _ { \mathrm { O P D } } + \mathcal { L } _ { \mathrm { f b } }$ , where each loss sums over its rollouts in the batch and each rollout enters exactly one term. Compared with OPD, the teacher target changes from π<sub>T</sub> $\mathbf { \psi } ^ { \prime } ( \cdot \mathbf { \ K } \mid h _ { t } )$ to $\pi _ { T } ( \cdot \ | \ \tilde { h } _ { t } , c )$ , under which the teacher is much more recoverable (Section 3.1). The student learns without the critique and needs no feedback at inference. Algorithm 1 in Appendix C.3 summarizes one training step.

![](images/dddc630266361711dca9d4df9fd56a0ee9b0cb76a642ffcb07ddab36e813e9ae.jpg)  
Figure 5: An example of critique-guided rewriting. Left: the original solution, with the critiqued paragraphs marked. Middle: the critique, which quotes each erroneous sentence, gives a fix, and ends with the reference answer. Right: the suffix that the student writes from the last error with the critique.

![](images/5823b606264e482dfb0ed2c81b9f34cfa2e4dfeff60a182e94601919bf8caaae.jpg)  
Figure 6: Recoverability of the teacher and the student on original rollouts (dashed) and critique-guided rewrites (solid), plotted against the position relative to the cut point. On the rewrite, the teacher sees the critique, and the student does not.

## 4 Experiments

The experiments answer three questions. RQ1: does critique-guided rewriting improve on-policy distillation on held-out mathematical reasoning benchmarks, and does it preserve performance outside mathematics? RQ2: which components of CriticOPD account for the improvement? RQ3: where do the gains of CriticOPD come from? Section 4.1 describes the shared setup, and the following subsections answer each question in turn.

## 4.1 Setup

Models and training. We distill Qwen3-1.7B and Qwen3-0.6B, both in non-thinking mode, from the Qwen3-4B-Instruct-2507 teacher on the prompts of DAPO-Math-17k. All methods within a student scale share the same training recipe. CriticOPD uses the teacher as the critic, and the critic lists all errors that it finds in the solution of the student within a limit of 2,048 tokens. Appendix C gives the hyperparameters, and Appendix A lists the prompts.

Baselines. We compare four classes of baselines. The first class contains standard post-training methods. SFT and KD train on solutions of the teacher, GRPO uses an outcome reward without the teacher, and OPD performs standard on-policy distillation.

The second class uses privileged information during training, as CriticOPD does. OPSD-T (Zhao et al., 2026) lets the teacher score student rollouts with the reference answer in its context. Critique-GRPO (Zhang et al., 2026b) lets the student revise incorrect solutions with teacher critiques and optimizes the original and revised solutions with outcome rewards. The third and fourth classes correspond to the two responses to the loss of recoverability that Section 1 describes. The third class reduces supervision, and FastOPD (Zhang et al., 2026a) represents it by truncating long rollouts. The fourth class replaces the sequence, and TRD (Jiang et al., 2026), SKD (Xu et al., 2025), and RelayOPD (Xu et al., 2026) let the teacher rewrite the solution of the student or write into its trajectory.

Evaluation. We evaluate on AIME 2024, AIME 2025, AMC 2023, MATH500, OlympiadBench, and Minerva Math with a budget of 32,768 tokens. We report the mean accuracy over 32 samples per problem on the first three benchmarks and over 8 samples on the others. We train each method with three seeds and report the mean and the standard deviation at the checkpoint that we select under seed 42 (Appendix C).

## 4.2 RQ1: Does critique-guided rewriting improve OPD?

Table 1 reports the main results. CriticOPD improves OPD by 4.57 points (50.08 → 54.65) on Qwen3-1.7B and 4.06 points (34.80 → 38.86) on Qwen3-0.6B. On Qwen3-1.7B, the largest gains occur on AIME 2024, AIME 2025, and AMC 2023 (+6.01, +7.36, and +6.94). CriticOPD achieves the best student accuracy on five of six benchmarks at both scales. On Qwen3-1.7B, it outperforms FastOPD and RelayOPD by 2.10 and 0.97 points, respectively. On Qwen3-0.6B, the corresponding gains are 1.57 and 0.76 points.

Among baselines using privileged information, OPSD-T yields smaller gains over OPD (+0.93 and +1.82 at the two scales), while Critique-GRPO underperforms the initial students. These results suggest that answer-conditioned scoring or critiqueguided revision alone is insufficient to match Critic-OPD. The improvement also extends to OLMo-3 models, where CriticOPD raises the average from 54.69 to 57.23 (Appendix E.1). Outside mathematics, CriticOPD achieves an average of 70.69, comparable to OPD (70.47), with modest declines on code benchmarks (Appendix E.2). Training dynamics and computational costs are provided in Appendices E and D.

## 4.3 RQ2: Which components of CriticOPD matter?

Table 2 changes one component of CriticOPD at a time. The components correspond to three designs of Section 3: the feedback that the student sees during the rewrite, the model that writes the suffix after the cut point, and whether the teacher sees the feedback when it scores the suffix. All variants use the Qwen3-1.7B student with seed 42, so differences below about 0.5 points are comparable to the standard deviation across seeds in Table 1. Appendix E.3 reports all six benchmarks for each variant, and Appendix E.4 varies the length limit of the critique.

Content of the feedback. The feedback of Critic-OPD contains the critique and the reference answer. Removing the reference answer lowers the average by 0.85, removing the critique lowers it by 5.90, and replacing both with generic feedback lowers it by 1.46. With the answer alone, the student falls even below OPD, likely because the rewrite then reaches the given answer without a corrected derivation, so the student learns to state answers rather than the reasoning that leads to them.

![](images/91bccb1616ed90557c1a6f2cfb9f61ac619044ed7c37f4fa600b27687ea7be30.jpg)  
Figure 7: Occupancy mismatch and teacher reliability on the states that each method supervises, and the average accuracy of the Qwen3-1.7B student.

Writer of the suffix. In this variant, the teacher continues from the same cut point with the same feedback, and the student is trained on the suffix of the teacher with the same loss. This variant lowers the average by 0.66.

Scoring of the suffix. When the teacher scores the suffix of the student without the feedback, the average falls by 1.19. In this case, the teacher scores the correction of the student at states after its error, and Section 2 shows that the recoverability of the teacher is low at these states. The critique therefore improves both the suffix that the student writes and the reliability of the teacher that scores it, and Section 4.4 analyzes this effect.

## 4.4 RQ3: Where do the gains of CriticOPD come from?

Section 3.1 shows that the critique substantially raises the recoverability of the teacher on the rewrite. This section answers two questions: whether the supervision of CriticOPD meets both requirements on reliability and occupancy mismatch, and how the more reliable target of the teacher becomes an ability of the student.

Reliability and occupancy mismatch together determine accuracy. Figure 7 places each method by the occupancy mismatch and the teacher reliability on the states that it supervises, and it marks the average accuracy of the Qwen3-1.7B student. The occupancy mismatch is the total variation distance between the token positions of the prefixes that a method supervises and those of the prefixes

Table 1: Results on six mathematical reasoning benchmarks. Entries are means and standard deviations over three seeds, and the subscript after each method gives the selected step. Bold marks the best student at each scale.
<table><tr><td>Method</td><td>AIME24</td><td>AIME25</td><td>AMC23</td><td>MATH500</td><td>OlympiadBench</td><td>Minerva</td><td>Avg</td></tr><tr><td>Teacher</td><td>62.19</td><td>48.12</td><td>85.99</td><td>94.83</td><td> $7 2 . 2 6 $ </td><td>45.54</td><td>68.16</td></tr><tr><td colspan="8">Student: Qwen3-0.6B-Non-Thinking</td></tr><tr><td>Student</td><td>2.08</td><td>1.35</td><td>18.98</td><td>45.52</td><td>16.94</td><td>13.69</td><td>16.43</td></tr><tr><td> $\mathrm { S F T _ { @ 1 3 9 } }$ </td><td> $4 . 6 2 \pm 0 . 4 9$ </td><td> $6 . 8 4 _ { \pm 0 . 1 2 }$ </td><td> $3 0 . 0 8 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $5 8 . 6 0 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $2 7 . 2 4 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $1 5 . 5 9 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $2 3 . 8 3 { \scriptstyle \pm 0 . 0 6 }$ </td></tr><tr><td> $\mathrm { K D } _ { @ 1 3 9 }$ </td><td> $4 . 5 1 _ { \pm 0 . 6 3 }$ </td><td> $6 . 1 5 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $2 8 . 7 5 _ { \pm 0 . 8 0 }$ </td><td> $5 7 . 8 8 _ { \pm 0 . 6 9 }$ </td><td> $2 6 . 6 2 _ { \pm 0 . 5 2 }$ </td><td> $1 5 . 0 9 _ { \pm 0 . 8 8 }$ </td><td> $2 3 . 1 7 _ { \pm 0 . 4 9 }$ </td></tr><tr><td> $\mathrm { G R P O } _ { \ @ 8 0 }$   $\mathrm { C r i t i q u e { \cdot } G R P O _ { @ 4 0 } }$ </td><td> $3 . 8 2 _ { \pm 0 . 7 1 }$ </td><td> $3 . 7 8 \pm 1 . 2 7$ </td><td> $2 6 . 4 6 _ { \pm 0 . 9 2 }$ </td><td> $5 3 . 7 2 _ { \pm 1 . 6 0 }$ </td><td> $2 3 . 4 5 _ { \pm 1 . 1 4 }$ </td><td> $1 5 . 1 2 _ { \pm 0 . 7 1 }$ </td><td> $2 1 . 0 6 _ { \pm 1 . 0 0 }$ </td></tr><tr><td></td><td> $1 . 4 2 _ { \pm 0 . 4 3 }$ </td><td> $0 . 9 7 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $1 6 . 5 9 _ { \pm 0 . 6 1 }$ </td><td> $4 1 . 0 2 _ { \pm 1 . 0 7 }$ </td><td> $1 3 . 9 3 _ { \pm 0 . 4 5 }$ </td><td> $1 1 . 4 7 _ { \pm 0 . 1 5 }$ </td><td> $1 4 . 2 3 _ { \pm 0 . 4 9 }$ </td></tr><tr><td> $\mathrm { O P D } _ { \ @ 4 0 }$ </td><td> $1 0 . 2 4 _ { \pm 1 . 1 7 }$ </td><td> $1 4 . 6 9 _ { \pm 1 . 3 6 }$ </td><td> $4 5 . 1 4 _ { \pm 0 . 6 4 }$ </td><td> $7 2 . 9 3 _ { \pm 0 . 3 2 }$ </td><td> $4 2 . 4 2 _ { \pm 0 . 7 9 }$ </td><td> $2 3 . 3 5 _ { \pm 0 . 2 0 }$ </td><td> $3 4 . 8 0 _ { \pm 0 . 3 0 }$ </td></tr><tr><td> $\mathrm { O P S D  – T _ { @ 8 0 } }$ </td><td> $1 3 . 3 7 _ { \pm 0 . 9 3 }$ </td><td> $1 5 . 0 0 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $4 7 . 4 8 _ { \pm 0 . 2 5 }$ </td><td> $7 5 . 5 2 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $4 4 . 1 0 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $2 4 . 2 6 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $3 6 . 6 2 { \scriptstyle \pm 0 . 1 5 }$ </td></tr><tr><td> $\mathrm { F a s t O P D } _ { @ 4 0 }$ </td><td> $1 3 . 7 2 _ { \pm 0 . 9 5 }$ </td><td> $1 8 . 7 5 _ { \pm 0 . 9 1 }$ </td><td> $4 7 . 4 3 _ { \pm 0 . 9 4 }$ </td><td> $7 5 . 1 8 _ { \pm 0 . 2 9 }$ </td><td> $4 4 . 4 8 _ { \pm 0 . 6 5 }$ </td><td> $2 4 . 1 7 _ { \pm 0 . 6 3 }$ </td><td> $3 7 . 2 9 _ { \pm 0 . 4 6 }$ </td></tr><tr><td> $\mathrm { T R D } _ { @ 1 3 9 }$ </td><td> $3 . 7 5 { \scriptstyle \pm 0 . 3 1 }$ </td><td> $5 . 3 5 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $2 7 . 8 5 { \scriptstyle \pm 0 . 5 1 }$ </td><td> $5 5 . 1 4 _ { \pm 0 . 5 5 }$ </td><td> $2 5 . 7 2 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $1 4 . 8 6 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $2 2 . 1 1 { \scriptstyle \pm 0 . 2 5 }$ </td></tr><tr><td> $\mathrm { S K D } _ { \ @ 4 0 }$ </td><td> $7 . 7 8 { \scriptstyle \pm 0 . 6 3 }$ </td><td> $1 0 . 3 8 _ { \pm 0 . 5 3 }$ </td><td> $3 6 . 1 9 _ { \pm 0 . 4 5 }$ </td><td> $6 5 . 6 0 _ { \pm 0 . 2 1 }$ </td><td> $3 3 . 8 1 _ { \pm 0 . 5 3 }$ </td><td> $1 9 . 4 1 _ { \pm 0 . 5 6 }$ </td><td> $2 8 . 8 6 _ { \pm 0 . 2 4 }$ </td></tr><tr><td> $\mathrm { R e l a y O P D _ { @ 6 0 } }$ </td><td> $1 3 . 9 0 _ { \pm 0 . 5 8 }$ </td><td> $1 7 . 6 4 _ { \pm 0 . 5 7 }$ </td><td> $4 8 . 9 6 _ { \pm 1 . 1 1 }$ </td><td> $7 6 . 4 9 _ { \pm 0 . 3 1 }$ </td><td> $4 6 . 0 1 _ { \pm 0 . 2 5 }$ </td><td> ${ \bf 2 5 . 6 0 _ { \pm 0 . 4 4 } }$ </td><td> $3 8 . 1 0 _ { \pm 0 . 2 2 }$ </td></tr><tr><td> $\mathbf { C r i t i c O P D } _ { \textcircled { \cong } 4 0 }$ </td><td> ${ \bf 1 4 . 8 1 _ { \pm 0 . 7 3 } }$ </td><td> $\mathbf { 1 9 . 4 1 _ { \pm 0 . 4 7 } }$ </td><td> $\pm 0 . 4 4 _ { \pm 0 . 4 8 }$ </td><td> ${ \bf 7 6 . 5 6 _ { \pm 0 . 3 5 } }$ </td><td> ${ \bf 4 6 . 8 1 _ { \pm 0 . 5 1 } }$ </td><td> $2 5 . 1 4 _ { \pm 0 . 2 6 }$ </td><td> ${ \bf 3 8 . 8 6 _ { \pm 0 . 1 5 } }$ </td></tr><tr><td> $\Delta { \ v s } \mathrm { O P D }$ </td><td> $+ 4 . 5 7$ </td><td> $+ 4 . 7 2$ </td><td> $+ 5 . 3 0$ </td><td> $+ 3 . 6 3$ </td><td> $+ 4 . 3 9$ </td><td> $+ 1 . 7 9$ </td><td>+4.06</td></tr></table>

<table><tr><td>Student</td><td>11.98</td><td>9.17</td><td>41.00</td><td>73.22</td><td>39.80</td><td>29.23</td><td>34.07</td></tr><tr><td> $\mathrm { S F T _ { @ 4 0 } }$ </td><td> $2 7 . 7 1 { \scriptstyle \pm 1 . 0 0 }$ </td><td> $2 3 . 3 7 _ { \pm 1 . 0 3 }$ </td><td> $5 8 . 8 2 _ { \pm 0 . 4 0 }$ </td><td> $8 3 . 3 9 _ { \pm 0 . 3 8 }$ </td><td> $5 2 . 5 8 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $3 4 . 5 6 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $4 6 . 7 4 { \scriptstyle \pm 0 . 4 8 }$ </td></tr><tr><td> $\mathrm { K D } _ { @ 1 3 9 }$ </td><td> $2 0 . 5 9 _ { \pm 0 . 4 3 }$ </td><td> $1 7 . 8 5 _ { \pm 2 . 2 9 }$ </td><td> $5 2 . 1 8 _ { \pm 0 . 6 3 }$ </td><td> $7 9 . 2 2 _ { \pm 0 . 2 2 }$ </td><td> $4 8 . 5 0 _ { \pm 0 . 4 6 }$ </td><td> $3 3 . 7 9 _ { \pm 0 . 9 2 }$ </td><td> $4 2 . 0 2 _ { \pm 0 . 4 9 }$ </td></tr><tr><td> $\mathrm { G R P O } _ { \ @ 1 2 0 }$ </td><td> $2 6 . 9 4 _ { \pm 1 . 5 9 } ^ { - }$ </td><td> $2 0 . 8 0 _ { \pm 1 . 2 0 }$ </td><td> $5 6 . 2 2 _ { \pm 0 . 4 3 }$ </td><td> $8 2 . 2 1 _ { \pm 0 . 9 8 }$ </td><td> $5 2 . 1 5 _ { \pm 1 . 0 4 }$ </td><td> $3 3 . 9 5 _ { \pm 0 . 5 3 }$ </td><td> $4 5 . 3 8 _ { \pm 0 . 8 5 }$ </td></tr><tr><td> $\mathrm { C r i t i q u e { \cdot } G R P O _ { @ 4 0 } }$ </td><td> $9 . 8 3 _ { \pm 0 . 5 7 }$ </td><td> $8 . 5 8 _ { \pm 0 . 3 9 }$ </td><td> $3 8 . 2 4 _ { \pm 0 . 2 8 }$ </td><td> $7 0 . 1 1 _ { \pm 0 . 3 4 }$ </td><td> $3 5 . 9 0 _ { \pm 0 . 2 5 }$ </td><td> $2 8 . 4 3 _ { \pm 0 . 3 1 }$ </td><td> $3 1 . 8 5 _ { \pm 0 . 2 5 }$ </td></tr><tr><td> $\mathrm { O P D } _ { \ @ 8 0 }$ </td><td> $3 5 . 1 0 _ { \pm 0 . 4 2 }$ </td><td> $2 2 . 7 8 _ { \pm 0 . 4 2 }$ </td><td> $6 2 . 9 4 _ { \pm 0 . 4 0 }$ </td><td> $8 6 . 2 6 _ { \pm 0 . 4 2 }$ </td><td> $5 6 . 6 5 _ { \pm 0 . 3 6 }$ </td><td> $3 6 . 7 8 _ { \pm 0 . 5 3 }$ </td><td> $5 0 . 0 8 _ { \pm 0 . 0 9 }$ </td></tr><tr><td> $\mathrm { O P S D - T _ { @ 6 0 } }$ </td><td> $3 7 . 0 8 _ { \pm 0 . 7 3 }$ </td><td> $2 3 . 5 1 { \scriptstyle \pm 0 . 5 1 }$ </td><td> $6 3 . 6 4 _ { \pm 0 . 3 0 }$ </td><td> $8 6 . 8 5 _ { \pm 0 . 3 3 }$ </td><td> $5 7 . 4 7 _ { \pm 0 . 1 9 }$ </td><td> $3 7 . 5 3 _ { \pm 0 . 7 9 }$ </td><td> $5 1 . 0 1 { \scriptstyle \pm 0 . 3 9 }$ </td></tr><tr><td> $\mathrm { F a s t O P D _ { @ 6 0 } }$ </td><td> $4 0 . 9 7 _ { \pm 1 . 8 4 }$ </td><td> $2 6 . 2 5 _ { \pm 0 . 5 8 }$ </td><td> $6 6 . 0 8 _ { \pm 0 . 2 4 }$ </td><td> $8 7 . 2 8 _ { \pm 0 . 3 6 }$ </td><td> $5 8 . 0 1 _ { \pm 0 . 2 7 }$ </td><td> $3 6 . 7 3 _ { \pm 0 . 9 9 }$ </td><td> $5 2 . 5 5 _ { \pm 0 . 3 6 }$ </td></tr><tr><td> $\mathrm { T R D } _ { \ @ 4 0 }$ </td><td> $2 1 . 1 5 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $1 8 . 7 2 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $5 0 . 9 5 _ { \pm 0 . 8 5 }$ </td><td> $7 9 . 3 1 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $4 6 . 3 8 { \scriptstyle \pm 0 . 4 9 }$ </td><td> $3 2 . 0 5 _ { \pm 0 . 1 0 }$ </td><td> $4 1 . 4 2 _ { \pm 0 . 1 5 }$ </td></tr><tr><td> $\mathrm { S K D } _ { \ @ 1 2 0 }$ </td><td> $2 6 . 4 2 _ { \pm 1 . 0 0 }$ </td><td> $2 1 . 5 3 { \scriptstyle \pm 1 . 2 2 }$ </td><td> $5 6 . 8 4 _ { \pm 0 . 1 3 }$ </td><td> $8 3 . 0 0 _ { \pm 0 . 5 5 }$ </td><td> $5 2 . 0 7 _ { \pm 0 . 4 7 }$ </td><td> $3 6 . 6 3 _ { \pm 0 . 4 2 }$ </td><td> $4 6 . 0 8 _ { \pm 0 . 5 4 }$ </td></tr><tr><td> $\mathrm { R e l a y O P D } _ { \textcircled { \cong } 4 0 }$ </td><td> $4 0 . 1 0 _ { \pm 1 . 1 5 }$ </td><td> $2 9 . 9 7 _ { \pm 0 . 6 0 }$ </td><td> $6 7 . 3 2 _ { \pm 0 . 5 2 }$ </td><td> $8 8 . 1 0 _ { \pm 0 . 3 5 }$ </td><td> $5 9 . 0 7 _ { \pm 0 . 2 5 }$ </td><td> $3 7 . 5 0 _ { \pm 0 . 1 2 }$ </td><td> $5 3 . 6 8 _ { \pm 0 . 3 9 }$ </td></tr><tr><td>CriticOPD@40</td><td> ${ \bf 4 1 . 1 1 _ { \pm 1 . 0 2 } }$ </td><td> $\mathbf { 3 0 . 1 4 _ { \pm 1 . 0 0 } }$ </td><td> ${ \bf 6 9 . 8 8 _ { \pm 0 . 2 6 } }$ </td><td> ${ \bf 8 8 . 8 7 _ { \pm 0 . 5 0 } }$ </td><td> ${ \bf 6 0 . 4 5 _ { \pm 0 . 5 1 } }$ </td><td> $3 7 . 4 4 _ { \pm 0 . 1 9 }$ </td><td> $\bar { 5 } 4 . 6 \bar { 5 } _ { \pm 0 . 1 0 }$ </td></tr><tr><td>∆ vs OPD</td><td>+6.01</td><td>+7.36</td><td>+6.94</td><td>+2.61</td><td>+3.80</td><td>+0.66</td><td></td></tr></table>

Table 2: Ablations on Qwen3-1.7B with seed 42. $\Delta$ is the change in Avg from CriticOPD, and from CriticOPD without the answer in (b) and (c). AIME averages AIME 2024 and AIME 2025.

RelayOPD. FastOPD does not supervise positions after 8,192 tokens, and RelayOPD ends the rollout after the second takeover, so both remove the later part of the response that the student reaches at inference. The rewrite of CriticOPD instead runs from the cut point to the end of the response. A shorter length limit for OPD also increases the occupancy mismatch, and the accuracy stops improving beyond 8k tokens. CriticOPD combines the highest teacher reliability with a low occupancy mismatch and reaches the highest average accuracy.

<table><tr><td>Variant</td><td>AIME</td><td>Avg</td><td>Δ</td></tr><tr><td>CriticOPD</td><td>34.79</td><td>54.57</td><td></td></tr><tr><td>(a) Content of the feedback</td><td></td><td></td><td></td></tr><tr><td>w/o answer</td><td>34.58</td><td>53.72</td><td>-0.85</td></tr><tr><td>w/o critique</td><td>28.22</td><td>48.67</td><td>-5.90</td></tr><tr><td>critique + answer → generic</td><td>33.85</td><td>53.11</td><td>-1.46</td></tr><tr><td>(b) Writer of the suffix student → teacher</td><td>33.18</td><td>53.06</td><td>-0.66</td></tr><tr><td>(c) Scoring of the suffix</td><td></td><td></td><td></td></tr><tr><td>w/o feedback</td><td>33.02</td><td>52.53</td><td>-1.19</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>OPD</td><td>28.70</td><td></td><td> $5 0 . 1 8 \quad - 4 . 3 9$ </td></tr></table>

The critique gives credit to the correction. For each token of the rewrite, the advantage in Equation (3) decomposes exactly as

$$
\begin{array} { r l } & { { \cal A } _ { t } ^ { \mathrm { f b } } = \underbrace { \log \pi _ { T } ( \tilde { y } _ { t } \mid \tilde { h } _ { t } ) - \log \pi _ { \theta } ( \tilde { y } _ { t } \mid \tilde { h } _ { t } ) } _ { \mathrm { o p D a d v a r a g e } } } \\ & { ~ + \underbrace { \log \frac { \pi _ { T } ( \tilde { y } _ { t } \mid \tilde { h } _ { t } , c ) } { \pi _ { T } ( \tilde { y } _ { t } \mid \tilde { h } _ { t } ) } } _ { r _ { t } } . } \end{array}\tag{4}
$$

that the student reaches at inference. This distance is a lower bound on the true mismatch, and it shows which part of the response the supervision covers. The teacher reliability is the mean recoverability of the teacher on the supervised states (Appendix B.6). CriticOPD has an occupancy mismatch of 0.16, which is slightly above 0.10 for OPD but below 0.21 for FastOPD and 0.35 for

The first term is the advantage of OPD on the same sequence, and the second term $r _ { t }$ is the pointwise mutual information between the token and the critique under the teacher. Feedback distillation is therefore OPD on the rewrite with an additional reward $r _ { t }$ on each token, and through this reward the teacher turns its recognition of errors into tokenlevel credit. To measure this credit, we compare two continuations of 64 tokens at 132 cut points, of which one continues the error and the other corrects it. Under the scoring of OPD, the correction leads by only 0.24 per token (95% CI [0.19, 0.29]), whereas under the scoring of CriticOPD, it leads by 1.47 ([1.36, 1.58]; Figure 8(a)). The first 32 tokens account for 87% of this increase. A teacher without the critique thus barely separates the correction from the error, which is consistent with the drop of 1.19 points when the teacher scores the suffix without the feedback (Section 4.3). The credit from the critique does not depend on which model writes the suffix, so a suffix that the teacher writes receives similar credit, consistent with the smaller drop of 0.66 points in that case.

![](images/ecbc5ae052aa3a57fada1e153f0ad11e7972137ca3bcd192b7a0e8824c4f20b2.jpg)

![](images/bbaad4276101b081a98a61ddb7ab5908f67c9ff0175ff8ff541395193bda80ac.jpg)  
Figure 8: (a) Advantage per token of two continuations of 64 tokens at the cut point, which either continue or correct the error, under the scoring of OPD and of CriticOPD. Numbers give correct minus continue. (b) Success rate of the student of CriticOPD minus that of OPD from the same prefixes on OlympiadBench, with 95% confidence bands.

The student learns to recover from its own errors. We take 150 incorrect and 150 correct rollouts of the initial student on OlympiadBench, which is not part of the training data, and let the students of OPD and CriticOPD continue 16 times from the empty prefix and each cut point. The two students differ by at most 0.015 on the empty prefix and by 0.000 on average on correct prefixes. On incorrect prefixes, the student of CriticOPD leads by 0.023 on average ([0.010, 0.036]) and by about 0.04 at the last two cut points (Figure 8(b)). The gain therefore concentrates on the states after the student errs, which are the states where the supervision of OPD loses its reliability in Section 2. This behavior also lengthens the responses, so part of the gain comes with longer reasoning, but at the same response length the critique still improves the

student (Appendix E.6).

## 5 Related Work

On-policy distillation and its supervision. Onpolicy distillation trains the student on its own sequences with feedback from the teacher (Agarwal et al., 2024; Gu et al., 2024; Ko et al., 2024; Yang et al., 2026b), and its transfer depends on whether the reasoning of the student and the teacher is compatible (Li et al., 2026). As prefixes degrade, the feedback of the teacher can lose its corrective value (Liu et al., 2026a; Xin et al., 2026), and existing methods truncate, select, or reweight the supervision (Zhang et al., 2026a; Yang et al., 2026c; Wang et al., 2026b; Zheng et al., 2026; Lee et al., 2026). These methods choose where to supervise, whereas CriticOPD rewrites the rollout after its last error.

Teacher intervention and trajectory repair. Other methods let the teacher write tokens into the trajectory through interleaved generation, revisions, or takeovers (Xu et al., 2025; Jiang et al., 2026; Xu et al., 2026), or regenerate from an earlier reliable context (Wang et al., 2026a). In CriticOPD, the teacher writes only a critique, which is removed before training, so every supervised token comes from the student.

Privileged context and language feedback. Context distillation trains a model to match itself with a fixed context (Askell et al., 2021; Snell et al., 2022), and OPSD applies it to OPD with the reference answer as the context (Zhao et al., 2026; Yang et al., 2026a). Other work trains on revisions guided by language feedback (Scheurer et al., 2023; Zhang et al., 2026b) or on self-correction in a later turn (Kumar et al., 2025; Qu et al., 2024). In CriticOPD, the teacher writes the critique for each incorrect rollout, and the student needs neither feedback nor an extra turn at inference.

## 6 Conclusion

We show that on-policy distillation relies on the recoverability of the teacher, which fails after the student errs because the teacher follows the errors of the student while still recognizing them. Critic-OPD uses the critique of the teacher to restore recoverability and improves OPD on mathematical reasoning benchmarks at two student scales. The critique changes only the context in which the teacher scores the rewrite, so the student writes every supervised token and needs no feedback at inference. It restores reliable token-level credit.

## Limitations

Reference answers. CriticOPD uses reference answers to decide which rollouts receive a critique and to ground the critic, so it applies directly to tasks with verifiable answers, such as mathematics. Tasks without verifiable answers would need a reward model or a critic that judges correctness on its own, which we leave to future work. For code, unit tests could play the role of the reference answer.

Models and domains. Our experiments distill Qwen3 students of two sizes from one Qwen3 teacher in non-thinking mode, and an OLMo-3 student from an OLMo-3 teacher with one seed (Appendix E.1), all on mathematical problems. The students preserve their performance outside mathematics (Appendix E.2), but students that reason in thinking mode, training domains such as code, and larger teachers and students remain to be tested.

Training cost. Each critiqued rollout needs one more generation of the teacher and one more generation of the student, although CriticOPD reaches its selected step earlier and uses 0.42 times the total training compute of OPD (Appendix D).

Quality of the critique. The teacher serves as the critic, and 73% to 81% of the errors that it lists are real under our judge (Section 2.2). CriticOPD improves OPD with this critic, and a more accurate critic may further improve the rewrite. We also use one critique per rollout, and sampling several critiques could further reduce noise. The diagnosis also relies on a model judge, which may disagree with human annotators in some cases.

## References

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. 2024. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations.

Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Benjamin Mann, Nova DasSarma, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Jackson Kernion, Kamal Ndousse, Catherine Olsson, Dario Amodei, Tom B. Brown, Jack Clark, and 3 others. 2021. A general language assistant as a laboratory for alignment. CoRR, abs/2112.00861.

Mislav Balunovic, Jasper Dekoninck, Ivo Petrov, Nikola´ Jovanovic, and Martin Vechev. 2025.´ MathArena: Evaluating LLMs on uncontaminated math competitions. arXiv preprint arXiv:2505.23281.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. 2024. MiniLLM: Knowledge distillation of large language models. In International Conference on Learning Representations.

Li Jiang, Haoran Xu, Yichuan Ding, and Amy Zhang. 2026. Trajectory-refined distillation. arXiv preprint arXiv:2606.08432.

Jongwoo Ko, Sungnyun Kim, Tianyi Chen, and Se-Young Yun. 2024. DistiLLM: Towards streamlined distillation for large language models. In International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 24872–24895.

Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D. Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, Lei M. Zhang, Kay McKinney, Disha Shrivastava, Cosmin Paduraru, George Tucker, Doina Precup, Feryal M. P. Behbahani, and Aleksandra Faust. 2025. Training language models to self-correct via reinforcement learning. In International Conference on Learning Representations.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with PagedAttention. In Proceedings ofthe 29th Symposium on Operating Systems Principles, pages 611–626. ACM.

Chia-Hsuan Lee, Zelei Cheng, Yu Wang, Renkun Ni, Sambit Sahu, Shi-Xiong Zhang, and William Campbell. 2026. SEAD: Competence-aware on-policy distillation via entropy-guided supervision. arXiv preprint arXiv:2606.28562.

Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan-ang Gao, Wenkai Yang, Zhiyuan Liu, and Ning Ding. 2026. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe. arXiv preprint arXiv:2604.13016.

Kaiyuan Liu, Ziyuan Zhuang, Yang Bai, Bing Wang, Rongxiang Weng, and Jieping Ye. 2026a. Prefix teach, suffix fade: Local teachability collapse in strong-to-weak on-policy distillation. arXiv preprint arXiv:2605.13643.

Yanjiang Liu, Jie Lou, Xinyan Guan, Yuqiu Ji, Hongyu Lin, Ben He, Xianpei Han, Le Sun, Xing Yu, and Yaojie Lu. 2026b. Your teacher can’t help you here: Combating supervision fidelity decay in on-policy distillation. arXiv preprint arXiv:2605.30833.

David Lopez-Paz, Léon Bottou, Bernhard Schölkopf, and Vladimir Vapnik. 2016. Unifying distillation and privileged information. In International Conference on Learning Representations.

Yuxiao Qu, Tianjun Zhang, Naman Garg, and Aviral Kumar. 2024. Recursive introspection: Teaching

language model agents how to self-improve. In Advances in Neural Information Processing Systems.

Stephane Ross, Geoffrey J. Gordon, and Drew Bagnell. 2011. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings ofthe Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings ofMachine Learning Research, pages 627–635.

Jérémy Scheurer, Jon Ander Campos, Tomasz Korbak, Jun Shern Chan, Angelica Chen, Kyunghyun Cho, and Ethan Perez. 2023. Training language models with language feedback at scale. CoRR, abs/2303.16755.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2025. HybridFlow: A flexible and efficient RLHF framework. In Proceedings of the Twentieth European Conference on Computer Systems, pages 1279–1297. ACM.

Charlie Snell, Dan Klein, and Ruiqi Zhong. 2022. Learning by distilling context. CoRR, abs/2209.15189.

Gokul Swamy, Sanjiban Choudhury, J. Andrew Bagnell, and Steven Wu. 2021. Of moments and matching: A game-theoretic framework for closing the imitation gap. In International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 10022–10032.

Vladimir Vapnik and Akshay Vashist. 2009. A new learning paradigm: Learning using privileged information. Neural Networks, 22(5-6):544–557.

Bing Wang, Shaotian Yan, Chen Shen, Kaiyuan Liu, Sinan Fan, Ximing Li, Rui Miao, Xiaosong Yuan, Zhanming Shen, and Jieping Ye. 2026a. Backtracking when it strays: Mitigating dual exposure biases in LLM reasoning distillation. arXiv preprint arXiv:2605.19433.

Yuanyi Wang, Su Lu, Yanggan Gu, Pengkai Wang, Yifan Yang, Zhaoyi Yan, Congkai Xie, Jianmin Wu, and Hongxia Yang. 2026b. Not all disagreement is learnable: Token teachability in on-policy distillation. arXiv preprint arXiv:2605.26844.

Haoran Xin, Anhao Zhao, Ying Sun, Jin Li, Xiaoyu Shen, and Hui Xiong. 2026. Escaping the KL agreement trap in on-policy distillation. arXiv preprint arXiv:2606.09471.

Haolei Xu, Xiaowen Xu, Haiwen Hong, Zixuan Ni, Hongxing Li, Yiwen Qiu, Weiming Lu, and Yongliang Shen. 2026. Pass the baton: Trajectoryrelayed on-policy distillation. arXiv preprint arXiv:2607.26057.

Wenda Xu, Rujun Han, Zifeng Wang, Long T. Le, Dhruv Madeka, Lei Li, William Yang Wang, Rishabh Agarwal, Chen-Yu Lee, and Tomas Pfister. 2025. Speculative knowledge distillation: Bridging the teacherstudent gap through interleaved sampling. In International Conference on Learning Representations.

Yuxiao Yang, Xiaoyun Wang, and Weitong Zhang. 2026a. Ogls-sd: On-policy self-distillation with outcome-guided logit steering for llm reasoning. arXiv preprint arXiv:2605.12400.

Yuxiao Yang, Tianrun Yu, Shangzhe Li, Kaixiang Zhao, Xuchao Zhang, Chetan Bansal, Huaxiu Yao, Taylor W Killian, and Weitong Zhang. 2026b. When eos tokens disagree: Understanding length inflation in onpolicy distillation. arXiv preprint arXiv:2609.20511.

Zhicheng Yang, Zhijiang Guo, Yifan Song, Minrui Xu, Yongxin Wang, Yiwei Wang, Xiaodan Liang, and Jing Tang. 2026c. Prune-OPD: Efficient and reliable on-policy distillation for long-horizon reasoning. arXiv preprint arXiv:2605.07804.

Dongxu Zhang, Zhichao Yang, Sepehr Janghorbani, Jun Han, Andrew Ressler, II, Qian Qian, Gregory D. Lyng, Sanjit Singh Batra, and Robert E. Tillman. 2026a. Fast and effective on-policy distillation from reasoning prefixes. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 25553– 25569.

Xiaoying Zhang, Yipeng Zhang, Hao Sun, Kaituo Feng, Chaochao Lu, Chao Yang, and Helen M. Meng. 2026b. Advancing LLM reasoning with natural language and numerical feedback. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 160575–160632. PMLR.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. 2026. Self-distilled reasoner: On-policy self-distillation for large language models. In Forty-third International Conference on Machine Learning.

Binbin Zheng, Xing Ma, Yiheng Liang, Jingqing Ruan, Xiaoliang Fu, Kepeng Lin, Benchang Zhu, Ke Zeng, and Xunliang Cai. 2026. SCOPE: Signal-calibrated on-policy distillation enhancement with dual-path adaptive weighting. arXiv preprint arXiv:2604.10688.

Ziheng Zhou, Jiaqi Li, Huacong Tang, Ying Nian Wu, and Demetri Terzopoulos. 2026. Less is more: Early stopping rollout for on-policy distillation. arXiv preprint arXiv:2605.27028.

## Appendix

A Prompts 11   
A.1 Prompts of CriticOPD 11   
A.2 Prompts of the Ablations 12   
A.3 Prompts of the Diagnosis 12   
A.4 Prompts of the Baselines 13   
B Details of the Diagnosis 13   
B.1 Rollouts and Cut Points 13   
B.2 Continuations and Success Rates 13   
B.3 Critique and Judgment 14   
B.4 Rewrite Analysis 14   
B.5 Diagnosis with the Qwen3-0.6B   
Student 14   
B.6 Occupancy Mismatch 15   
C Experimental Details 15   
C.1 Training 15   
C.2 Software and Hardware 15   
C.3 Configuration of CriticOPD 16   
C.4 Baselines 17   
C.5 Evaluation 17   
C.6 Checkpoint Selection 17   
D Computational Cost 18   
E Additional Results 18   
E.1 Results with OLMo-3 Models 18   
E.2 Results Outside Mathematics 18   
E.3 Per-Benchmark Ablation Results 19   
E.4 Sensitivity to Hyperparameters 19   
E.5 Training Dynamics 19   
E.6 Generation Length 20

## A Prompts

This appendix lists the prompts verbatim, with placeholders in typewriter font. In training, the student receives each problem of DAPO-Math-17k in the user message $\mathcal { P } _ { \mathrm { t r a i n } }$ , and in evaluation, each problem of the benchmarks in $\mathcal { P } _ { \mathrm { e v a l } }$ . In the other prompts, {student prompt} stands for the complete user message $\mathcal { P } _ { \mathrm { t r a i n } }$ of the problem. The critic is the teacher model. It receives the solution of the student split into paragraphs at blank lines, where headings, separator lines, and short lead-in lines are merged into the next paragraph, and the prompts call these paragraphs segments.

{problem} Please reason step by step, and put your final answer within \boxed{}.

## A.1 Prompts of CriticOPD

The critic of CriticOPD receives the system prompt $\mathcal { P } _ { \mathrm { c r i t i c } }$ , which defines the task and the output format, and the user message $\mathcal { P } _ { \mathrm { c r i t i c - u s e r } }$ , which gives the problem, the reference answer, and the numbered paragraphs of the solution. For a rollout that reaches the length limit, the first paragraph of $\mathcal { P } _ { \mathrm { c r i t i c } }$ is replaced by $\mathcal { P } _ { \mathrm { c r i t i c } } ^ { \mathrm { l e n } }$ . From the reply of the critic, we parse the listed errors and build the feedback, which ends with the reference answer. CriticOPD keeps the rollout up to the end of the paragraph of the last listed error and appends $\mathcal { P } _ { \mathrm { f b } }$ , which contains the feedback. The student continues after ${ \mathcal { P } } _ { \mathrm { f b } } .$ and the teacher scores this continuation with $\mathcal { P } _ { \mathrm { f b } }$ in its context, whereas the training sequence of the student does not contain $\mathcal { P } _ { \mathrm { f b } }$

P<sub>train</sub> — user message of the student in training (DAPO-Math-17k)   
Solve the following math problem step by step. The last line of your response should be of the form Answer: \$Answer (without quotes) where \$Answer is the answer to the problem.   
{problem}   
Remember to put your answer on its own line after $\mathrm { ^ { 6 6 } A n \mathrm { - } }$ swer:”.

## P<sub>critic</sub> — system prompt of the critic

You are a meticulous math teacher. A student has submitted a complete solution whose FINAL ANSWER IS WRONG. You are given the correct answer. Use it as a reference to trace the student’s work forward and find the steps that contain genuine errors.

How to locate the errors:

Read the numbered segments in order. For each one ask: given everything before it, is this step correct? List every segment that contains a genuine error, in the order in which they appear.

Do NOT flag a segment merely because it is: an exploratory attempt the student later abandons or corrects; a longer route than necessary; or sloppy notation that does not change the mathematics. Flag only a step that is actually wrong.

Rules you must follow:

1. List the erroneous segments in order of appearance.

2. For each one, quote verbatim the single sentence that carries the error.

3. For each one, say what is wrong and what to do instead. The student will resume writing after the last listed error, so phrase it as guidance they can act on. Two or three sentences per error.

4. NEVER reveal the final answer. Do not state it, do not write an expression that evaluates to it, do not say what it should be. Describe the method, not the result. 5. If every step looks correct to you and you cannot identify a genuine error, reply with SEGMENT: NONE. Do not guess -- a wrong location is worse than none.

P<sup>diag</sup> — first paragraph of P<sub>critic</sub> in Section 2.2, com  
plete solution   
You are a meticulous math teacher. A student has sub  
mitted a complete solution whose FINAL ANSWER IS   
WRONG. Trace the student’s work forward and find the   
steps that contain genuine errors.

Reply in exactly this format, one block per error:   
ERROR 1   
SEGMENT: <index>   
QUOTE: <the exact wrong sentence, copied verbatim>   
FEEDBACK: <what is wrong and what to do instead>   
ERROR 2   
SEGMENT: <index>   
QUOTE: <...>   
FEEDBACK: <...>

P<sup>len</sup><sub>critic</sub> — first paragraph of P<sub>critic</sub> for a rollout that reaches the length limit

You are a meticulous math teacher. A student’s solution ran on without ever reaching an answer. You are given the correct answer. Use it as a reference to trace the student’s work forward and find the steps that sent the reasoning off track. If the work is still on a valid path and was merely slow, reply SEGMENT: NONE.

P<sub>critic-user</sub> — user message of the critic   
Problem:   
{student prompt}   
Correct answer (reference only -- never reveal it):   
{reference answer}   
Student’s solution, split into numbered segments:   
[0] {segment 0}   
[1] {segment 1}

## P<sub>fb</sub> — text appended to the kept prefix of the rollout

{kept prefix}   
[Teacher feedback] {feedback}   
Using this feedback, continue the solution from here. Put   
your final answer in \boxed{}.

{feedback} in P<sub>fb</sub> when the critic lists k > 1 errors;   
for k = 1, the lines before the last line are only   
{feedback 1}   
Your solution contains {k} errors.   
(1) In “{quote 1}”: {feedback 1}   
(2) In “{quote 2}”: {feedback 2}   
The correct final answer is {reference answer}.  
Each quote is cut to 200 characters.

## A.2 Prompts of the Ablations

All variants in Table 2 use the critic of CriticOPD. The variant without the answer omits the last line of the feedback, and its critic removes every listed error whose feedback contains the reference answer while the solution of the student does not. The variants in (b) and (c) of Table 2 change this variant, and the runs in Appendices E.4 and E.1 use the same feedback. The variant without the critique keeps only the last line of the feedback, which states the reference answer, and the variant with generic feedback replaces the feedback by $\mathcal { P } _ { \mathrm { g e n e r i c } } .$

P<sub>generic</sub> — generic feedback   
Your solution contains errors. Check the previous reason  
ing and redo it.

## A.3 Prompts of the Diagnosis

The diagnosis in Section 2.2 uses the critic of Critic-OPD without the reference answer. The first paragraph of $\mathcal { P } _ { \mathrm { c r i t i c } }$ is replaced by $\mathcal { P } _ { \mathrm { c r i t i c } } ^ { \mathrm { d i a g } }$ for a complete solution and by $\mathcal { P } _ { \mathrm { c r i t i c } } ^ { \mathrm { p a r t } }$ for a truncated solution, and the line with the reference answer is removed from $\mathcal { P } _ { \mathrm { c r i t i c - u s e r } }$ . The judge receives the system prompt $\mathcal { P } _ { \mathrm { j u d g e } }$ and the user message $\mathcal { P } _ { \mathrm { j u d g e - u s e r } }$ , which lists the errors that the critic claims. The generic hint in Appendix B.4 replaces the feedback in $\mathcal { P } _ { \mathrm { f b } }$ by P<sub>hint</sub>.

part critic first paragraph of P<sub>critic</sub> in Section 2.2, truncated solution

You are a meticulous math teacher. A student has written part of a solution, and the solution is not finished yet. Trace the student’s work forward and find the steps that contain genuine errors. If every step so far is correct, reply SEGMENT: NONE.

## P<sub>judge</sub> — system prompt of the judge (Qwen3-8B)

You are an expert mathematics grader. You are given a math problem, its correct answer, a student’s solution split into numbered segments, and a list of errors that a teacher claims to have found in the solution, in order of appearance. For every listed error, decide whether the claim is right.

## Labels:

NEW\_ERROR - the flagged step contains a genuine mistake of its own (a wrong formula, wrong logic, an arithmetic slip, a misread condition, an unjustified claim) that would still be wrong even if every earlier step were correct.

PROPAGATED - the reasoning in the flagged step is valid in itself; it is wrong only because it uses a value, expression or conclusion produced by an earlier mistake.

RESTATEMENT - the flagged step only restates or reports a wrong result that was already reached earlier (for example the final answer line or a summary), with no new reasoning.

NOT\_AN\_ERROR - the flagged step is actually correct.

![](images/91b50dd8573fab3d216706d569fb9ac05d105c49db24eeedbbe2a35b24da8c5e.jpg)

## A.4 Prompts of the Baselines

SFT, KD, GRPO, OPD, FastOPD, SKD, and RelayOPD use $\mathcal { P } _ { \mathrm { t r a i n } }$ . TRD lets the teacher rewrite one solution of the student per prompt with the system prompt $\mathcal { P } _ { \mathrm { T R D } } ^ { \mathrm { s y s } }$ and the user message $\mathcal { P } _ { \mathrm { T R D } }$ OPSD-T gives the student $\mathcal { P } _ { \mathrm { t r a i n } }$ and scores its rollout with a teacher that receives ${ \mathcal { P } } _ { \mathrm { O P S D } }$ . Critique-GRPO critiques an incorrect rollout as CriticOPD does, including the reference answer at the end of the feedback, and the student then answers again in a new conversation with the system prompt $\mathcal { P } _ { \mathrm { C G R P O } } ^ { \mathrm { s y s } }$ and the user message $\mathcal { P } _ { \mathrm { C G R P O } }$

P<sup>sys</sup> TRD system prompt of the TRD rewrite   
Please reason step by step, and put your final answer   
within \boxed{}.   
P<sub>TRD</sub> — user message of the TRD rewrite   
Your task is to rewrite your mathematical solution.   
\*\*Problem:\*\*   
{student prompt}   
\*\*Your Initial Solution:\*\*   
{initial solution}   
\*\*Instructions:\*\*   
1. Preserve the overall structure and reasoning path of   
your original solution   
2. Identify and fix errors in computation or logic   
3. Keep correct intermediate steps and meaningful work   
4. Output ONLY the rewritten solution

## B.1 Rollouts and Cut Points

The diagnosis uses the Qwen3-1.7B student after 80 steps of OPD with seed 42 and the teacher of the main experiments. The student generates one rollout for each of 900 problems from DAPO-Math-17k with temperature 1.0, top-p 1.0, and at most 16,384 tokens, and we grade each rollout with the reward function of training. We randomly select 150 correct and 150 incorrect rollouts. An incorrect rollout must end with a final answer, so we exclude rollouts that reach the length limit. We split each response into paragraphs at blank lines, as Critic-OPD does. The cut points are the empty prefix and the 7 paragraph boundaries that lie closest to the positions $1 / 8 , \ldots , 7 / 8$ of the response.

## B.2 Continuations and Success Rates

From each cut point, the teacher and the student each generate 16 continuations of the prefix of the student within the same assistant turn. Each model uses its own chat template, temperature 1.0, top-p 1.0, and a budget of 16,384 tokens minus the length of the prefix. $\bar { \Lambda } ^ { T } ( h )$ and $\Lambda ^ { S } ( h )$ are the fractions of correct continuations of the teacher and the student from the prefix h. A problem is solvable by the teacher if the teacher succeeds in at least 8 of the 16 continuations from the empty prefix. All bands are 95% bootstrap confidence intervals with 2,000 resamples of rollouts.

## B.3 Critique and Judgment

For Figure 4(b), the teacher critiques the truncated solution at every cut point with $\mathcal { P } _ { \mathrm { c r i t i c } } ^ { \mathrm { p a r t } }$ (Appendix A.3). It decodes greedily for at most 2,048 tokens, and we apply the filters of training. Qwen3- 8B in thinking mode judges every listed error with the reference answer in its context. It labels an error as a new mistake, a mistake that follows from an earlier one, a restatement of an earlier wrong result, or not an error, and we count every label other than the last as real. The judge samples with temperature 0.6, top-p 0.95, and top-k 20 and generates at most 16,384 tokens.

## B.4 Rewrite Analysis

Figure 6 uses the incorrect rollouts in which the success rate of the teacher, estimated from its first 8 continuations, is at least 0.5 at the empty prefix and falls below 0.5 at a later cut point. To enlarge this set, we add the 204 remaining incorrect rollouts from the same 900 problems, which gives 163 such rollouts. The teacher critiques each complete solution with the prompts of CriticOPD, and 161 critiques keep at least one error after filtering. We cut each rollout as CriticOPD does and draw 8 continuations per condition. At the cut point, the teacher continues with and without the critique, and the student continues without the critique, with the critique, and with the generic hint $\mathcal { P } _ { \mathrm { h i n t } }$ . At 25%, 50%, and 75% of a rewrite that the student writes with the critique, the teacher continues with the critique and the student continues without it. At the same positions of the discarded part of the original rollout, both continue without the critique.

## B.5 Diagnosis with the Qwen3-0.6B Student

We repeat the diagnosis with the Qwen3-0.6B student after 40 steps of OPD with seed 42, which is the checkpoint in Table 1, and keep the teacher, the problems, and all other settings. Of the 900 rollouts, 270 are correct, 593 are incorrect, and 37 reach the length limit. Figures 9, 10, and 11 correspond to Figures 3, 4, and 6, and they show the same pattern as the Qwen3-1.7B student. On the 106 incorrect rollouts from problems that the teacher solves, $\Lambda ^ { T }$ falls from 0.89 at the empty prefix to 0.38 at the last cut point, and the fraction of teacher continuations that end with the incorrect answer of the student rises from 0.00 to 0.15. Across the groups of $\Lambda ^ { T }$ , whose means range from 0.05 to 0.92, between 77% and 84% of the errors that the teacher lists are real. For Figure 11, we add the 443 remaining incorrect rollouts as in Appendix B, which gives 304 rollouts in which the success rate of the teacher falls below 0.5, and 295 critiques keep at least one error. After the cut point, the success rate of the teacher falls from 0.45 to 0.14 on the original rollout and stays between 0.70 and 0.75 on the rewrite, and the success rate of the student rises from 0.06 to 0.12 on the rewrite.

![](images/ca0a6b2ed68616fd8f0782cde58a3ca119f5f3bd4321049c93021c1d3b52d047.jpg)

![](images/c41260f8ca05f6eaf945d02ac4e1904eff669cb3a0375caabfdde85c0f1012b7.jpg)  
Figure 9: Expected accuracy of the teacher and the Qwen3-0.6B student from prefixes of the student on (a) correct and (b) incorrect rollouts.

![](images/a595a0b9e3399c857d977b00811a3354f00f302dd9e4003236ac840f8dbb964b.jpg)

![](images/61473bc6091b5d6030126e440e07a68df83aafb23d746c2bf5d81d30c46f5c4f.jpg)  
Figure 10: (a) Outcomes of the continuations of the teacher and (b) fraction of the listed errors that are real, for the Qwen3-0.6B student.

![](images/74ca7267cc77fd995441847a9c4e9cb58e8582db495b00b28f0e9d3aa32c6eed.jpg)  
Figure 11: Success rates of the teacher and the Qwen3- 0.6B student around the cut point.

## B.6 Occupancy Mismatch

The measurement in Section 4.4 uses the first 256 problems in the training order of DAPO-Math-17k with seed 42 and the student of Section 2. For d , the student generates four rollouts per problem with temperature 1.0, top-p 1.0, and at most 32,768 tokens, and every generated token contributes one position. For $d _ { m } ,$ CriticOPD, RelayOPD, and SKD each run one training step on these problems from the same student, and every token that receives a loss contributes one position. For OPD, we take all positions of the original rollouts in the step of CriticOPD, which the student generates with the training limit of 16,384 tokens. FastOPD differs from OPD only in the length limit, so we take the first 8,192 positions of these rollouts, and the first 1,024, 2,048, and 4,096 positions for the shorter limits. SFT and KD train on the same teacher solutions of these problems, and TRD trains on the solutions that the teacher rewrites from rollouts of the student, so every response token of these data contributes one position. We split the positions from 1 to 32,768 into 48 intervals of equal width on a logarithmic scale and merge the intervals whose rounded boundaries coincide, which leaves 45 intervals. We compute the total variation distance between the two position distributions over these intervals. The curves in Figure 12 are kernel density estimates over the logarithm of the position and serve only for display.

Figure 12 shows the position distributions of all methods. A shorter length limit moves the supervision to earlier positions, and the distance grows from 0.10 for OPD to 0.21, 0.46, 0.68, and 0.84 for limits of 8,192, 4,096, 2,048, and 1,024 tokens. RelayOPD and TRD supervise shorter sequences than the student generates, with distances of 0.35 and 0.53. SFT, KD, and SKD stay close to inference in position, with distances of 0.18 and 0.15. At inference, 9.3% of the tokens lie after position 16,384, which no method with a training limit of 16,384 tokens supervises.

The position does not reflect the content of a prefix, so we also compare the hidden states of the student, which are the inputs to its output layer at the prefixes. We sample 20,000 states per method and 30,000 states for inference uniformly over tokens, subtract the mean inference state of each problem, and divide each dimension by its standard deviation over the inference states. A logistic regression separates the states of a method from an equal number of inference states, with five folds that split the problems. If its held-out accuracy is a, then 2a − 1 is a lower bound on the total variation distance between the two state distributions. Two disjoint halves of the inference rollouts give −0.02, which is the noise level. Figure 13 projects the states onto the first two principal components of the inference states. CriticOPD has a distance of 0.05, the same as OPD and lower than FastOPD (0.09), SKD (0.15), and RelayOPD (0.19). SFT and KD (0.29) and TRD (0.50) train on text that the teacher writes or rewrites and have the largest distances in hidden states, although SFT and KD stay close to inference in position.

## C Experimental Details

## C.1 Training

Table 3: Hyperparameters of all methods.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Training</td><td></td></tr><tr><td>Prompts per step</td><td>128</td></tr><tr><td>Rollouts per prompt</td><td>1 (GRPO: 8)</td></tr><tr><td>Optimizer updates per step</td><td>1</td></tr><tr><td>Learning rate</td><td>10−⁶, constant, no warmup</td></tr><tr><td>Maximum prompt length Maximum response length</td><td>2,048</td></tr><tr><td></td><td>16,384 (FastOPD: 8,192)</td></tr><tr><td>Rollout temperature, top-p</td><td>1.0,1.0</td></tr><tr><td>Checkpoint interval GPUs</td><td>20 steps 4 (2 student, 2 teacher)</td></tr><tr><td></td><td></td></tr><tr><td>Critic of CriticOPD</td><td></td></tr><tr><td>Decoding</td><td>greedy</td></tr><tr><td>Maximum critique length</td><td>2,048 all</td></tr><tr><td>Number of listed errors</td><td></td></tr><tr><td>Evaluation</td><td></td></tr><tr><td>Temperature, top-p</td><td>1.0,1.0</td></tr><tr><td>Maximum new tokens</td><td>32,768</td></tr><tr><td>Samples per problem</td><td>32 (AIME, AMC), 8 (others)</td></tr></table>

Table 3 lists the shared hyperparameters. OPD optimizes the sampled reverse-KL objective of Equation (1) with the policy gradient and applies no KL penalty. One epoch over DAPO-Math-17k takes 139 steps.

## C.2 Software and Hardware

We implement all methods on the code of RelayOPD (Xu et al., 2026), which builds on verl 0.9.0.dev (Sheng et al., 2025), and the released code fixes the commit of this code base. The software consists of Python 3.12, PyTorch 2.11 with CUDA 13.0, Transformers 5.15, and Ray 2.57. vLLM 0.21 (Kwon et al., 2023) generates the rollouts and the critiques, and a separate vLLM server computes the log-probabilities of the teacher on the training sequences. We grade mathematical answers with Math-Verify 0.9.0 and code with EvalPlus 0.3.1. Each training run uses one node with four NVIDIA A100-SXM4-80GB GPUs, two for the student and two for the teacher. The seed-42 runs of OPD, FastOPD, and RelayOPD on Qwen3- 1.7B use four GPUs for each model with the same batch size, and GRPO and SFT, which use no teacher during training, use all four GPUs for the student. Runs on Qwen3-0.6B use A100, H100, H200, or B200 GPUs, and each evaluation job uses one GPU.

![](images/73749ab550178fb5521aacd5da8f9d499775557b410538eb370e0d660654296e.jpg)  
Figure 12: Token positions of the prefixes that each method supervises (colored) and of the inference prefixes (gray), with the total variation distance in each title. The dotted line marks the training limit of 16,384 tokens.

![](images/bd88a58a827493eb82fbf76429376642b6f53b6877a7ea6c2b523db9a6172a82.jpg)  
Figure 13: Hidden states of the student on the prefixes that each method supervises, projected onto the first two principal components of the inference states. The contour encloses 95% of the inference states, and each title gives the classifier bound on the total variation distance.

## C.3 Configuration of CriticOPD

Algorithm 1 summarizes one training step of Critic-OPD. The critic is the teacher. If a critique reaches the length limit, we discard its last item, which is incomplete. The feedback lists the remaining items and ends with the reference answer (Appendix A.1). Rollouts that reach the response limit are also sent to the critic, with a prompt that tells the critic the solution is unfinished (Appendix A.1). The rewrite uses the response budget that remains after the kept prefix. We train CriticOPD for 40 steps on Qwen3- 1.7B and for 80 steps on Qwen3-0.6B, and select the step as for all methods (Appendix C.6).

Algorithm 1 CriticOPD Training   
Require: Student $\pi _ { \theta } ,$ teacher $\pi _ { T }$ , batch $\{ ( x , a ) \}$   
1: $\mathcal { L } _ { \mathrm { o u r s } }  0$   
2: for each $( x , a )$ in the batch do   
3: Sample $y \sim \pi _ { \boldsymbol { \theta } } ( \cdot \vert x )$   
4: $\mathbf { i f } \ y$ is incorrect or truncated then   
5: Generate critique c using $\pi _ { T }$ given   
$( x , y , a )$   
6: Discard incomplete error items from c   
7: if a valid critique is available then   
8: $\tau ~ $ first position after the last cri  
tiqued paragraph   
9: $\tilde { y } _ { < \tau } \gets y _ { < \tau }$   
10: Sample $\tilde { y } _ { \ge \tau } \sim \pi _ { \theta } ( \cdot \mid h _ { \tau } , c )$   
11: $\ell \gets \mathcal { L } _ { \mathrm { f b } } ( \tilde { y } ; c )$   
12: else   
13: $\ell \gets \mathcal { L } _ { \mathrm { O P D } } ( y )$   
14: $\mathcal { L } _ { \mathrm { o u r s } } \gets \mathcal { L } _ { \mathrm { o u r s } } + \ell$   
15: Update $\theta$ using ${ \mathcal { L } } _ { \mathrm { o u r s } }$

## C.4 Baselines

SFT trains the student with cross-entropy on one solution of the teacher per prompt, which the teacher samples with temperature 1.0 and which we do not filter by correctness. KD trains on the same solutions with the forward KL divergence on the 128 most probable tokens of the teacher. GRPO uses a binary correctness reward and no teacher. FastOPD is OPD with a maximum response length of 8,192 tokens during training. TRD lets the teacher rewrite one solution of the student per prompt and fix its errors, and it trains the student on the rewrites with the objective of KD. SKD generates rollouts by speculative decoding: the teacher accepts a draft token of the student if the token is among its 5 most probable tokens and otherwise samples a replacement, and the student is trained with the objective of KD. RelayOPD lets the teacher take over for at most 3 paragraphs or 256 tokens when the teacher predicts a reflection token that is outside the 5 most probable tokens of the student. It allows at most two takeovers, ends the rollout after the second one, and applies the OPD objective to all tokens. OPSD-T trains as OPD, except that the teacher scores each rollout with the reference answer after the problem (P in Appendix A.4). The student generates and is trained without the reference answer, and no rollout is critiqued or rewritten. Critique-GRPO follows its official implementation and uses only the outcome reward. The student samples 7 rollouts per prompt. If all of them are incorrect, the critic of CriticOPD critiques each of them with the feedback ending with the reference answer, and the student answers again in a new conversation with $\mathcal { P } _ { \mathrm { C G R P O } }$ . One of these refinements, a correct one if any, joins the group as an eighth response. The advantage is the reward minus the mean reward of the group, without the normalization by the standard deviation that GRPO uses. Tokens of the initial rollouts use the importance ratio without clipping, the tokens of the refinement are weighted by $p / ( p + 0 . 1 )$ , where $p$ is their probability under the student, and the entropy coefficient is 0.001.

## C.5 Evaluation

The benchmarks contain 30 (AIME 2024), 30 (AIME 2025), 83 (AMC 2023), 500 (MATH500), 675 (OlympiadBench), and 272 (Minerva Math) problems. The students generate in non-thinking mode, and Avg is the macro average over the six benchmarks. For each method, we select one step with seed 42 (Appendix C.6), train seeds 43 and 44 to the same step, and report the mean and the sample standard deviation over the three seeds. The teacher and the initial students are evaluated once.

## C.6 Checkpoint Selection

We select checkpoints on a validation set that does not overlap with the training data or the six test benchmarks. We call a problem contaminated if at least half of its 13-grams appear in DAPO-Math-17k or in a test benchmark, or if its normalized text equals that of such a problem. A single shared 13-gram is not enough, because common phrases such as “where m and n are relatively prime positive integers. Find $m + n ^ { \prime \prime }$ already contain more than 13 words. The validation set has two parts. MATH-val starts from the 4,500 test problems of MATH outside MATH500, of which 1,304 are contaminated, 1,284 of them by DAPO-Math-17k, and we sample 500 of the remaining 3,196 problems uniformly with seed 0. Comp-val contains all 153 problems of four competitions from 2025 in Math-Arena (Balunovic et al.´ , 2025), namely BRUMO, CMIMC, HMMT February, and SMT, and none of them is contaminated. We do not use AIME 2022 and 2023, because 54 of their 60 problems appear in DAPO-Math-17k. The validation set uses the prompt $\mathcal { P } _ { \mathrm { e v a l } }$ , the chat template, the stop tokens, the grader, and the sampling parameters of the test benchmarks, with up to 32,768 new tokens, and it draws 4 samples per problem on MATH-val and 8 on Comp-val. The accuracy of each part averages the fraction of correct samples over its problems, and the validation score is the mean of the accuracies of the two parts.

![](images/a009835d6a770499f7772e2cbb9a068f52311be4d7285cb7c397f0288f5b0745.jpg)

![](images/aa8000bd7ce5667a9324cb75ff18b66aeade173e36c800a7c365466a883fe77e.jpg)  
Figure 14: Training cost of the Qwen3-1.7B students. (a) A100 GPU-hours until the selected step, including the generation of offline data. (b) Seconds per training step, averaged over five steps.

The candidates of each method are the checkpoints at every multiple of 20 steps and at the last step of training if it is not a multiple of 20. We select the candidate with the highest validation score, and the earlier step if two candidates tie. For Qwen3-1.7B and Qwen3-0.6B, we select once for each student and method with the checkpoints of seed 42 and use the selected step for all seeds. For the OLMo-3 student, we make no selection and reuse the steps selected for Qwen3-1.7B, with the exceptions that Appendix E.1 describes. The test results do not enter the selection.

## D Computational Cost

Figure 14 compares the training cost of all methods on the Qwen3-1.7B student. We measure the time of every training step under seeds 43 and 44, in which all methods run on four A100 GPUs, and average the two seeds. For SFT, KD, and TRD, we add the time to generate the offline data on

A100 GPUs. The time per step of CriticOPD grows from about 170 seconds over the first 10 steps to about 870 seconds over steps 31 to 40, compared with 160 and 560 seconds for OPD, because the responses of the student grow longer and incorrect rollouts receive a critique and a rewrite. Over the same 40 steps, CriticOPD uses 0.96 times the GPU-hours of OPD. CriticOPD reaches its selected step after 40 steps, so its total cost of 17.2 GPUhours is 0.42 times that of OPD at step 80. GRPO and Critique-GRPO, which sample 8 rollouts per prompt, take about 890 and 820 seconds per step.

## E Additional Results

## E.1 Results with OLMo-3 Models

Table 4 repeats the comparison with models from another family. The student is Olmo-3-7B-Instruct-SFT, and the teacher is Olmo-3.1-32B-Instruct, which shares the tokenizer of the student. All methods use the training recipe and the evaluation protocol of Table 1 with seed 42. Each method is trained to the step selected for Qwen3-1.7B, except that OPD and CriticOPD are both evaluated at step 60, and that we stop KD and SKD early, at steps 80 and 60 instead of 139 and 120. The critique of Critic-OPD here does not end with the reference answer. CriticOPD raises the average of OPD from 54.69 to 57.23. As with the Qwen3 students, the gains are largest on the competition benchmarks, with +5.83 on AIME 2025 and +3.16 on AMC 2023.

## E.2 Results Outside Mathematics

Table 5 reports the results of the Qwen3-1.7B students outside mathematics. We evaluate on

Table 4: Results with OLMo-3 models on six mathematical reasoning benchmarks with seed 42. The subscrip after each method gives the step, and bold marks the best student.
<table><tr><td>Method</td><td>AIME24</td><td>AIME25</td><td>AMC23</td><td>MATH500</td><td>OlympiadBench</td><td>Minerva</td><td>Avg</td></tr><tr><td>Teacher</td><td>73.54</td><td>60.62</td><td>92.24</td><td>96.22</td><td>77.35</td><td>47.43</td><td>74.57</td></tr><tr><td colspan="8">Student: Olmo-3-7B-Instruct-SFT, teacher: Olmo-3.1-32B-Instruct</td></tr><tr><td>Student</td><td>6.15</td><td>4.17</td><td>30.50</td><td>64.80</td><td>30.09</td><td>29.09</td><td>27.47</td></tr><tr><td> $\mathrm { S F T _ { @ 4 0 } }$ </td><td>31.56</td><td>26.15</td><td>59.45</td><td>79.50</td><td>51.91</td><td>31.66</td><td>46.70</td></tr><tr><td> $\mathrm { K D } _ { \ @ 8 0 }$ </td><td>35.42</td><td>30.83</td><td>72.33</td><td>86.83</td><td>59.48</td><td>33.27</td><td>53.03</td></tr><tr><td> $\mathrm { G R P O } _ { \ @ 1 2 0 }$ </td><td>10.00</td><td>7.60</td><td>37.46</td><td>69.85</td><td>36.28</td><td>30.70</td><td>31.98</td></tr><tr><td> $\mathrm { C r i t i q u e { \cdot } G R P O _ { @ 4 0 } }$ </td><td>6.56</td><td>4.90</td><td>31.40</td><td>65.67</td><td>31.15</td><td>29.32</td><td>28.17</td></tr><tr><td> $\mathrm { { O P D } _ { \ @ 6 0 } }$ </td><td>41.04</td><td>31.98</td><td>71.54</td><td>87.58</td><td>60.61</td><td>35.39</td><td>54.69</td></tr><tr><td> $\mathrm { O P S D - T _ { @ 6 0 } }$ </td><td>41.46</td><td>34.90</td><td>72.29</td><td>88.40</td><td>61.80</td><td>34.42</td><td>55.54</td></tr><tr><td> $\mathrm { F a s t O P D _ { @ 6 0 } }$ </td><td>38.23</td><td>33.33</td><td>71.05</td><td>88.05</td><td>60.65</td><td>35.52</td><td>54.47</td></tr><tr><td> $\mathrm { T R D } _ { \ @ 4 0 }$ </td><td>27.50</td><td>24.48</td><td>56.25</td><td>79.55</td><td>50.70</td><td>31.66</td><td>45.02</td></tr><tr><td> $\mathrm { S K D } _ { \ @ 6 0 }$ </td><td>38.82</td><td>29.33</td><td>70.82</td><td>86.61</td><td>59.30</td><td>33.78</td><td>53.11</td></tr><tr><td> $\mathrm { R e l a y O P D } _ { \textcircled { \cong } 4 0 }$ </td><td>32.60</td><td>27.08</td><td>67.47</td><td>84.67</td><td>57.22</td><td>33.55</td><td>50.43</td></tr><tr><td> $\mathbf { C r i t i c O P D } _ { \ @ 6 0 }$ </td><td>43.44</td><td>37.81</td><td>74.70</td><td>89.08</td><td>62.33</td><td>36.02</td><td>57.23</td></tr><tr><td> $\Delta { \ v s } \mathrm { O P D }$ </td><td>+2.40</td><td>+5.83</td><td>+3.16</td><td>+1.50</td><td>+1.72</td><td>+0.63</td><td>+2.54</td></tr></table>

Table 5: Results outside mathematics for Qwen3-1.7B. Code uses EvalPlus pass@1 on the extended tests.
<table><tr><td>Method</td><td>MMLU-Pro</td><td>MMLU</td><td>ARC-C</td><td>OpenBookQA</td><td>HumanEval+</td><td>MBPP+</td><td>Avg</td></tr><tr><td>Teacher</td><td>69.75</td><td>80.09</td><td>94.94</td><td>93.95</td><td>82.00</td><td>74.70</td><td>82.57</td></tr><tr><td>Student</td><td>38.10</td><td>55.24</td><td>77.69</td><td>74.10</td><td>62.50</td><td>56.50</td><td>60.69</td></tr><tr><td>SFT</td><td>50.80</td><td>63.29</td><td>84.41</td><td>79.70</td><td>63.10</td><td>55.40</td><td>66.12</td></tr><tr><td>KD</td><td>50.85</td><td>63.78</td><td>84.28</td><td>79.30</td><td>64.00</td><td>57.10</td><td>66.55</td></tr><tr><td>GRPO</td><td>51.41</td><td>61.49</td><td>81.68</td><td>78.75</td><td>63.70</td><td>58.50</td><td>65.92</td></tr><tr><td>OPD</td><td>56.61</td><td>67.14</td><td>87.07</td><td>83.80</td><td>66.30</td><td>61.90</td><td>70.47</td></tr><tr><td>FastOPD</td><td>56.14</td><td>66.27</td><td>85.52</td><td>83.40</td><td>66.50</td><td>61.20</td><td>69.84</td></tr><tr><td>TRD</td><td>47.44</td><td>61.22</td><td>81.08</td><td>75.45</td><td>62.80</td><td>57.20</td><td>64.20</td></tr><tr><td>SKD</td><td>53.84</td><td>65.87</td><td>85.90</td><td>81.65</td><td>63.70</td><td>59.90</td><td>68.48</td></tr><tr><td>RelayOPD</td><td>56.04</td><td>66.40</td><td>85.90</td><td>83.20</td><td>66.90</td><td>60.40</td><td>69.81</td></tr><tr><td>CriticOPD</td><td>57.31</td><td>67.88</td><td>87.73</td><td>84.90</td><td>65.90</td><td>60.40</td><td>70.69</td></tr></table>

MMLU-Pro, MMLU, ARC-Challenge, Open-BookQA, HumanEval+, and MBPP+ with 4 samples per problem and a budget of 32,768 tokens. For MMLU and MMLU-Pro, we use subsets of 2,000 questions that keep the proportion of each subject. We grade multiple-choice answers by the option letter in the final box and report EvalPlus pass@1 on the extended tests for code.

## E.3 Per-Benchmark Ablation Results

Table 6 reports the six benchmarks for every variant in Table 2, all at step 40 except OPD at step 80.

## E.4 Sensitivity to Hyperparameters

CriticOPD adds two hyperparameters to OPD: the number k of listed errors that the feedback contains, and the length limit of the critique. The rewrite starts after the k-th error. Figure 15 varies each hyperparameter and keeps the other at its default, which uses all listed errors and a limit of 2,048 tokens. The average rises with k, from 51.56 with one error to 53.31 with three and 53.72 with all errors, so the feedback on later errors still helps. The length limit matters until most critiques fit: limits of 512 and 1,024 tokens cut off 79% and 27% of the critiques and give 51.72 and 52.78, whereas limits of 2,048 and 4,096 tokens cut off 6% and 3% and give similar averages (53.72 and 53.31).

![](images/9777add7ee804407e5f1d9ba62472b8369b0292d6e1019af9628d74c6dba6959.jpg)

![](images/8e9c4797cd9b7d658a7f7900fdff7d160e878560e7e6a533ceb989f0b49885e3.jpg)  
Figure 15: Sensitivity of CriticOPD on the Qwen3-1.7B student with seed 42. The star marks the default, and the numbers in (b) give the share of truncated critiques.

## E.5 Training Dynamics

Figure 16 shows the training dynamics of all methods. Panels (a) and (e) report the fraction of training sequences that reach the reference answer. These sequences contain the tokens that the teacher samples in SKD, the takeovers of the teacher in RelayOPD, the refinements in Critique-GRPO, and the rewrites after the critique in CriticOPD, so this fraction is not the accuracy of the student alone. The entropy of Critique-GRPO is not shown, because its loss sums over the tokens and divides by a constant length, so its logged entropy is not a mean over tokens. Panels (d) and (h) report the training loss of the methods that train on offline data, which is the cross-entropy for SFT and the forward KL divergence for KD and TRD. The training logs of seed 42 are missing for SFT, KD, and TRD on Qwen3-0.6B, so these curves average seeds 43 and 44. At both scales, the training sequences of Critic-OPD become longer than those of OPD, which agrees with the longer responses at evaluation (Section 4.4).

(c) Entropy  
Table 6: Per-benchmark results of the ablations on Qwen3-1.7B with seed 42. The rows follow Table 2.
<table><tr><td>Variant</td><td>AIME24</td><td>AIME25</td><td>AMC23</td><td>MATH500</td><td>Olympiad</td><td>Minerva</td><td>Avg</td></tr><tr><td>CriticOPD</td><td>40.52</td><td>29.06</td><td>70.18</td><td>89.40</td><td>60.59</td><td>37.64</td><td>54.57</td></tr><tr><td colspan="8">(a) Content of the feedback</td></tr><tr><td>w/o answer</td><td>41.56</td><td>27.60</td><td>67.43</td><td>88.75</td><td>59.04</td><td>37.91</td><td>53.72</td></tr><tr><td>w/o critique</td><td>34.48</td><td>21.96</td><td>60.33</td><td>85.55</td><td>54.83</td><td>34.89</td><td>48.67</td></tr><tr><td>critique + answer → generic</td><td>40.00</td><td>27.71</td><td>67.39</td><td>88.05</td><td>58.91</td><td>36.63</td><td>53.11</td></tr><tr><td colspan="8">(b) Writer of the suffix</td></tr><tr><td>student → teacher</td><td>38.12</td><td>28.23</td><td>68.45</td><td>87.48</td><td>59.37</td><td>36.72</td><td>53.06</td></tr><tr><td colspan="8">(c) Scoring of the suffix</td></tr><tr><td>w/o feedback</td><td>38.85</td><td>27.19</td><td>67.09</td><td>87.72</td><td>58.41</td><td>35.94</td><td>52.53</td></tr><tr><td>OPD</td><td>34.69</td><td>22.71</td><td>63.33</td><td>86.70</td><td>56.31</td><td>37.32</td><td>50.18</td></tr></table>

GRPO SKD OPSD-T RelayOPD SFT TRD Critique-GRPO OPD FastOPD CriticOPD KD

![](images/1b3c8ffa6e0d48fecf57101e155f45af746f020c63066034dedf690bc28f3192.jpg)

![](images/16e1911c82c84299ba447dcdfdc49e199a94ae3b1f0280fc459ffe454ba7222e.jpg)

![](images/d729d8991c966d2abf9a8aaef5e26e65194e7121e81ed0a6ed1040c3d4397741.jpg)

![](images/4687547731255e401cbebd2dbf895af0774c86f4baa99bd5db923bea288577f8.jpg)

(f) Response length (k tokens)  
![](images/b1847437508ed1fcd19a49341cd03370b6697023cf9e13e61a36966c513f2a98.jpg)

(h) Loss of offline methods  
(g) Entropy  
![](images/36a8a8e8c28b709ae37823d3d23747a8709f21a51af5eee8cf6609631f66aab3.jpg)

![](images/7b5ae6326227b09db1a3bd90c3e6253dd81b53f107d63803d9d6b9c0abefc4ca.jpg)

![](images/76fefacec274d5a55dbc7d6ac3c8d5d480144a61a590798e677f6f35ca15d76a.jpg)  
Figure 16: Training dynamics, averaged over seeds and smoothed over five steps.

## E.6 Generation Length

The student of CriticOPD writes longer responses than the student of OPD, with 10.2k tokens per response on average, compared with 5.3k. Figure 17 evaluates the Qwen3-1.7B students under generation budgets from 4k to 32k tokens. We reuse the responses with the budget of 32,768 tokens and count a response as incorrect if it is longer than the budget. CriticOPD falls below OPD at 4k and 8k tokens and exceeds it at 16k and 32k.

Longer responses alone do not explain the gain. Table 7 compares variants that share the rewriting of CriticOPD and produce responses of about the same median length on AIME. At the 32k generation budget, when the teacher scores the rewrite without the critique, the average is 2.39 points lower than that of CriticOPD with the same feedback, and generic feedback lowers it by 1.78 points. FastOPD also writes responses of a similar length and is 2.10 point lower. At the same response length, the critique in the context of the teacher therefore still improves the student.

![](images/a62f37df61ead861760d60d3ff92ebd353e6db44bb4b05eb68dabada801f62f9.jpg)  
Figure 17: Average over six benchmarks when responses longer than the budget count as incorrect, over three seeds of Qwen3-1.7B.

Table 7: Length-controlled comparison on Qwen3- 1.7B, averaged over three seeds.
<table><tr><td>Method</td><td>AIME length</td><td>Avg@16k</td><td>Avg@32k</td></tr><tr><td>OPD</td><td>8,487</td><td>49.99</td><td>50.08</td></tr><tr><td>FastOPD</td><td>11,910</td><td>50.27</td><td>52.55</td></tr><tr><td>CriticOPD</td><td>12,457</td><td>50.32</td><td>54.65</td></tr><tr><td>scoring w/o feedback</td><td>12,065</td><td>50.62</td><td>52.26</td></tr><tr><td>critique → generic</td><td>12,051</td><td>50.86</td><td>52.87</td></tr></table>