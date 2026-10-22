# RECURSIVE SELF-IMPROVEMENT ON A BUDGET: SCORING CODE EDITS BEFORE THEY RUN

Erfan Darzi<sup>1,2</sup>, Sree Bhargavi Balija<sup>3</sup>, Sheng He<sup>1</sup>

<sup>1</sup>Harvard University <sup>2</sup>Massachusetts Institute of Technology <sup>3</sup>UC San Diego

## ABSTRACT

A coding agent that rewrites its own scaffold has to choose which proposed edit to expand before that edit has been evaluated, and the search then has to return one agent. Each run fits one linear readout on the parent’s steering-split solve rate and eight coordinates of the proposed diff, from 96 scaffold edits labeled by four evaluations of a 16-instance fit split, and then freezes the coefficients. HumanEval+ reserves 48 instances as the steering split and Polyglot reserves 64. That split scores parents and keeps an archive of four nodes. Huxley–Gödel Machine selection draws a parent by Thompson sampling on the steering counts of nodes that have already been evaluated, and it expands the child that parent proposes. Every method then returns the earliest evaluated node whose steering-split solve count is maximal. The tables report the held-out score of that node on the other 100 HumanEval+ instances and the other 145 Polyglot instances, next to the root scaffold and the calibration edit with the highest fit-split label. Twelve runs of GPT-4o return a node at 87.0% on held-out HumanEval+, against 84.2% for the parent-rate ablation. On that split Huxley–Gödel Machine selection returns 85.1%, the root is at 81.2%, and the calibration-best edit is at 82.8%. On the same split Claude Sonnet 4.5 returns 86.6%, beside 86.5% for Huxley–Gödel Machine selection, 85.3% for the parent-rate ablation, and 85.1% for the calibration-best edit.

## 1 INTRODUCTION

A coding agent that rewrites its own scaffold has to rank the edits it proposes before any of them has been evaluated, because the checker runs on one child per keep step and the child that is kept becomes a parent of the edits that follow. The Darwin Gödel Machine adds a child to its archive only after that child has been evaluated, and it samples parents from observed scores (Zhang et al., 2026). The Huxley–Gödel Machine draws a parent by Thompson sampling on the success and failure counts of evaluated nodes and expands the child that parent proposes (Wang et al., 2026). AIDE scores a candidate after the evaluator has run it (Jiang et al., 2025). Surrogate-assisted evolution fits a predictor of fitness from earlier evaluations of similar individuals (Jin, 2011). The readout is that kind of predictor, applied to the diff before the child is expanded. It is a linear score on the parent’s steering-split solve rate and eight coordinates of the diff. The ablation uses the same archive and expands by the parent’s steering rate alone. When the keep steps end, the readout, the parent-rate ablation, and Huxley–Gödel Machine selection each return the earliest evaluated node whose steering-split solve count is maximal. The held-out score of that node is the comparison, together with the root and the calibration edit whose fit-split label is highest. Huxley–Gödel Machine selection draws θ from Bet $\mathsf { \Omega } _ { \mathsf { i } } ( 1 + n _ { s } , 1 + n _ { f } )$ for each evaluated node, using that node’s steering counts, and expands the child of the parent with the largest draw. Every run of every method spends the same number of checker evaluations: the readout and the ablation label 96 calibration edits and then expand 24 children, and Huxley–Gödel Machine selection expands 144 children on Polyglot and 176 on HumanEval+. Claude Sonnet 4.5 and GPT-4o (gpt-4o-2024-08-06) are fit separately (OpenAI, 2024). The held-out splits are 100 HumanEval+ instances and 145 Polyglot instances (Liu et al., 2023; Gauthier, 2024). Table 1 is the held-out score of the returned node for the readout, the parent-rate ablation, and Huxley–Gödel Machine selection. Table 2 is the child the readout and the parent-rate ablation expand at the first keep step, and Figure 1 plots the returned node. Theorem 1 gives the mean of that parent draw, $( 1 + \bar { n _ { s } } ) / ( \bar { 2 } + n _ { s } + \bar { n _ { f } } )$ , on the steering counts of an evaluated node. The proof is in Appendix A.

![](images/ae756fa41f4d1214f490824a5cfbee215026144b34fa7439a2da27b10ba12897.jpg)

![](images/971202f46635a0b9dafc0ec3c12b33c35470d72a0b9f177b23971f482de1ae79.jpg)

![](images/5cf21c1eefad1fd4640af45bfcb8895d6d03bc8a8ac4c98bc3bbe7ee9447df9b.jpg)

![](images/8fc9207b440efbdf2234aaf508bb1d417d8ab131a01315636038a063d9fd81ba.jpg)  
Figure 1: Held-out solve rate of the returned node, twelve runs. Points are means and bars are run standard deviations. For the readout, the parent-rate ablation, and Huxley–Gödel Machine selection, the node is the earliest evaluated node whose steering-split solve count is maximal. The root is the starting scaffold, and calibration best is the edit with the highest fit-split label. On HumanEval+ with GPT-4o the readout is the highest mean. On HumanEval+ with Claude Sonnet 4.5 the readout sits with Huxley–Gödel Machine selection, the parent-rate node, and the calibration-best edit. Table 1 reports these means.

• The readout ranks eight scaffold diffs at each keep step, and every method returns the earliest evaluated node whose steering-split solve count is maximal (Algorithm 1, Figures 2 and 3).

• Huxley–Gödel Machine selection draws a parent from Beta $( 1 + n _ { s } , 1 + n _ { f } )$ on the steering counts of evaluated nodes and expands that parent’s child (Theorem 1, Figure 3).

• On held-out HumanEval+, GPT-4o’s returned node leads the parent-rate ablation, and the same table reports Huxley–Gödel Machine selection, the root, and the calibration-best edit (Table 1, Figure 1).

## 2 METHOD

## 2.1 BIRTH-TIME READOUT

The root is a scaffold: a system prompt and a loop that writes a solution, runs the unit tests, and retries once. A proposal is a unified diff of that scaffold. The readout scores that diff from the parent’s steering-split solve rate and from eight coordinates of the diff, before the diff is expanded. The diff is embedded by text-embedding-3-small as a vector in R<sup>1536</sup>. Each run draws its own projection, with i.i.d. standard normal entries and rows scaled to unit length, and it maps that vector onto the coordinates e. The birth feature of the full readout is $X = ( R _ { \mathrm { p a r } } , e ) \in \mathbb { R } ^ { 9 }$ . Both $R _ { \mathrm { p a r } }$ and the label R are solve rates in [0, 1].

Before the search, each run calibrates its own readout. It proposes 96 scaffold edits, 48 of the root and two from each of 24 first-generation children of the root, and labels each edited child by its mean solve rate over four evaluations of a fixed 16-instance split of the benchmark being run. Each of those 24 parents is evaluated once on the steering split, so $\bar { R } _ { \mathrm { p a r } }$ is measured for every calibration parent. Because those parents do not share one rate, the least-squares slope on $R _ { \mathrm { p a r } }$ is identified, and on these pairs that slope is positive. Each generator is fit separately on each benchmark. The full readout minimizes $\begin{array} { r } { \sum _ { i = 1 } ^ { 9 6 } ( \dot { R _ { i } } - w ^ { \top } X _ { i } - b ) ^ { 2 } } \end{array}$ and scores a proposal by $\widehat { \boldsymbol { v } } ( \boldsymbol { X } ) = \boldsymbol { w } ^ { \intercal } \boldsymbol { X } + \boldsymbol { b }$ . The ablation minimizes the same squared error on $R _ { \mathrm { p a r } }$ alone,

![](images/7f8f8594884b7b91141472765f8bbffb95e6e7a6dde04041678f80345ce85983.jpg)  
Figure 2: One keep step of the full readout. The eight proposals are divided across the archive. The birth feature is $X = ( R _ { \mathrm { p a r } } , e )$ , with e the eight coordinates from the run’s projection. The readout expands the highest score. The archive keeps four nodes by the steering-split solve rate.

$$
\sum _ { i = 1 } ^ { 9 6 } \bigl ( R _ { i } - w _ { r } R _ { \mathrm { p a r } , i } - b _ { r } \bigr ) ^ { 2 } ,\tag{1}
$$

and scores a proposal by $\widehat { v } _ { r } ( R _ { \mathrm { p a r } } ) = w _ { r } R _ { \mathrm { p a r } } + b _ { r }$ . The coefficients stay fixed for the 24 keep steps.

During the search, $R _ { \mathrm { p a r } }$ is the parent’s solve rate on the steering split, 48 HumanEval+ instances or 64 Polyglot instances, disjoint from the 16 fit instances. The readout and the ablation keep the four evaluated nodes with the largest solve rate on that steering split. Huxley–Gödel Machine selection keeps the tree. For each evaluated node it draws θ from Beta $( 1 + n _ { s } , 1 + n _ { f } )$ on the success and failure counts of that node’s clade on the steering split, and it expands the child proposed by the parent with the largest draw. The steering split chooses the parents, the archive, and the returned node. The fit, the archive, and the parent draw read the steering split. The tables report the held-out score of the node that rule returns.

Each keep step of the readout and of the ablation divides eight proposals as evenly as possible across the archive and expands the argmax of that method’s score. On a tie the earliest proposal is expanded. Proposals are decoded at temperature 1.0, so the eight diffs of one keep step differ in what they change and by how much, and a diff that does not apply cleanly is discarded and redrawn. The archive begins as the evaluated root, so the first eight proposals share $R _ { \mathrm { p a r } }$ and the ablation’s scores tie. The weights on e are then what ranks those eight diffs. From the next keep step the archive holds more than one parent. The positive slope $w _ { r }$ orders the ablation’s proposals by the parent’s steering-split rate, and the weights on e can prefer a diff from a lower-scoring parent. The readout and the ablation run for $H = 2 4$ keep steps. Every method evaluates one child on the steering split at each keep step. When those steps end, every method returns the earliest evaluated node whose steering-split solve count is maximal, and the root remains eligible. Figure 2 is one keep step of the full readout.

## 2.2 SEARCH LOOP

Algorithm 1 is the search, and the evaluator on each benchmark is that benchmark’s checker. Figure 3 places Huxley–Gödel Machine selection beside the full readout. The parent draw uses the steering counts of evaluated nodes, and Theorem 1 gives its mean.

![](images/6ffa8e82dfc76e275f0e7f645d53db969f28188a45fcaaabe051025773d2926e.jpg)  
Figure 3: Two ways to choose the next parent. (Left) Huxley–Gödel Machine selection draws θ for each evaluated node from its steering counts and expands the child of the largest draw. $( R i g h t )$ The full readout scores the edit from $X \doteq ( R _ { \mathrm { p a r } } , e ) \ : \flat \ : \forall \ : \hat { v } ( X ) = w ^ { \top } X + b .$

## 2.3 BIRTH-TIME LAW

Huxley–Gödel Machine selection uses the case $\tau = 1$ of the law below. The density, the ratio, and the event of zero clade counts are the objects the theorem uses.

Definition 1 (Beta draw). For $\tau > 0$ and integers $n _ { s } , n _ { f } \geq 0$ , the law of $\theta _ { n _ { s } , n _ { f } } ^ { ( \tau ) }$ is the probability measure on (0, 1) with density $B ( \alpha , \beta ) ^ { - 1 } u ^ { \alpha - 1 } ( 1 - u ) ^ { \beta - 1 }$ , where $\alpha = \tau ( 1 + n _ { s } ) , \beta = \tau ( 1 + n _ { f } )$ and $\begin{array} { r } { B ( \alpha , \beta ) = \int _ { 0 } ^ { 1 } u ^ { \alpha - 1 } ( 1 - u ) ^ { \beta - 1 } d u \in ( 0 , \infty ) } \end{array}$

Definition 2 (Ratio estimator). If $n _ { s } + n _ { f } \ge 1$ , set $\rho ( n _ { s } , n _ { f } ) = n _ { s } / ( n _ { s } + n _ { f } )$ . The expression is not a real number when $n _ { s } = n _ { f } = 0$

Definition 3 (Birth time). A birth time of a node is an event on which that node’s clade counts, in the sense of the subtree rooted at the node, including the node, satisfy $n _ { s } = n _ { f } = 0$ . The code-change feature may still vary.

Assumption 1 (Positive scheduler). $\tau > 0$

Assumption 2 (Birth-time zero counts). The node under discussion is at a birth time in the sense of Definition 3.

Theorem 1. Assume Assumptions 1 and 2. On the event ofAssumption 2, $\theta _ { 0 , 0 } ^ { ( \tau ) }$ has law Beta $. ( \tau , \tau )$ This law does not depend on X. Its mean is m $( 0 , 0 ) = 1 / 2$ , and its variance is $1 / ( 4 ( 2 \tau + 1 ) )$ ). For all integers $n _ { s } , n _ { f } \ge 0$ and every $\tau > 0 , \mathbb { E } [ \theta _ { n _ { s } , n _ { f } } ^ { ( \tau ) } ] = m ( n _ { s } , n _ { f } ) = ( 1 + n _ { s } ) / ( 2 + n _ { s } + n _ { f } )$ . The ratio $\rho ( 0 , 0 )$ is undefined. The case $\tau = 1$ is a Beta(1, 1) draw, uniform on (0, 1), with mean $1 / 2$ and variance 1/12.

The experiments use $\tau = 1$ . Huxley–Gödel Machine selection draws θ from this law for each evaluated node, from the success and failure counts on that node’s steering evaluation, and expand the child of the parent with the largest draw. The mean of the draw is $m ( \bar { n } _ { s } , n _ { f } ) = ( 1 + n _ { s } ) \bar { / } ( 2 +$ $n _ { s } + n _ { f } )$ . The proof is in Appendix A.

## 3 EXPERIMENTS

<table><tr><td>Algorithm 1 Birth-time readout search. The loop stops after H keep steps, with H set so that every method spends the same number of checker evaluations per run:  $H = 2 4$  for the readout and the ablation, and H = 144 on Polyglot or 176 on HumanEval+ for Huxley-Gödel Machine selection.</td></tr><tr><td>Input: Generator G, Claude Sonnet 4.5 (claude-sonnet-4-5-20250929) or GPT-4o (gpt-4o-2024-08-06); checker C; horizon  $H ;$  eight proposals at decoding temperature 1.0; archive width 4. Edits are unified diffs of that scaffold. Each run draws its own projection mapping the embedding onto  $e \in \mathbb { R } ^ { 8 } .$  The full readout uses</td></tr><tr><td> $X = ( R _ { \mathrm { p a r } } , e )$  . The steering split is 48 HumanEval+ instances or 64 Polyglot instances. Output: The evaluated node with the largest steering-split solve count, scored on the held-out split.</td></tr><tr><td>1: On this generator and this benchmark, fit (w, b) on X and  $( w _ { r } , b _ { r } )$  on  $R _ { \mathrm { p a r } }$  alone, using the 96 pairs of Section 2.1. Hold both fits fixed.</td></tr><tr><td>2: Evaluate the root with C on the steering split, set the archive to that root, and set its clade counts from that evaluation.</td></tr><tr><td>3: while fewer than H keep steps have completed do 4: if the method is the full readout or the parent-rate ablation then</td></tr><tr><td>5: Propose eight edits, divided as evenly as possible across the archive. Redraw a diff that does not apply. 6: Score each edit by (X) or by  $\widehat { v } _ { r } ( R _ { \mathrm { p a r } } )$  , according to the method.</td></tr><tr><td>7: Expand the edit with the largest score. On a tie, expand the earliest proposal. 8: Evaluate that child with C on the steering split.</td></tr><tr><td>9: Keep the four evaluated nodes with the largest steering-split solve rate.</td></tr><tr><td>10: else 11: Draw  $\theta \sim \mathrm { B e t a } ( 1 + n _ { s } , 1 + n _ { f } )$  for each evaluated node and expand the child of the</td></tr><tr><td>parent with the largest draw.</td></tr><tr><td>12: Evaluate that child with C on the steering split and add the outcomes to the clade counts of its ancestors.</td></tr></table>

The held-out splits are the 100 HumanEval+ instances and the 145 Polyglot instances that remain after the 16-instance fit split and the steering split (Liu et al., 2023; Chen et al., 2021; Gauthier, 2024). The steering splits contain 48 and 64 instances. Claude Sonnet 4.5 (claude-sonnet-4-5-20250929) and GPT-4o (gpt-4o-2024-08-06) are fit separately and run separately. Each cell is twelve runs. Each run draws its own projection, calibrates its own coefficients on its own 96 edits, and draws its own proposals. Proposals are decoded at temperature 1.0 and the scaffold’s solutions at 0.2. In four-fold cross-validation on each run’s calibration edits, the fitted score ranks the measured labels at Spearman $\rho$ between 0.18 and 0.34 across the four settings. Every run of every method spends the same number of checker evaluations: 9,280 on Polyglot and 8,496 on HumanEval+. A run of the readout or of the ablation spends 6,144 of them labeling the calibration edits, 1,536 on Polyglot or 1,152 on HumanEval+ evaluating the 24 calibration parents on the steering split, 64 or 48 evaluating the root, and the rest on its 24 keep steps. Huxley–Gödel Machine selection evaluates the root and expands 144 children on Polyglot and 176 on HumanEval+, reaching the same totals. The calibration stage is itself a search: the edit with the highest measured label is a candidate agent, and Table 1 reports it as the calibration-best baseline. After the keep steps, the readout, the parent-rate ablation, and Huxley–Gödel Machine selection each return one node, the earliest evaluated node whose steering-split solve count is maximal. Table 1 is the held-out score of that node, of the root, and of the calibration-best edit. Table 2 is the held-out score of the child the readout and the parent-rate ablation expand at step 1. Printed means and run standard deviations follow the aggregation protocol in Appendix B. A paired interval is the 95% Student interval on the twelve run-wise differences, with eleven degrees of freedom. The one-sided permutation probability is the fraction of the 4096 sign flips whose signed sum is at least the observed sum.

At keep step 1 the archive holds only the root, so the eight proposals share one parent rate. The readout ranks them by the coordinates of the diff. The parent-rate scores tie, and the ablation expands the earliest proposal. On held-out Polyglot with Claude Sonnet 4.5 those children score 60.1% and 57.4% (Table 2).

Table 1: Held-out solve rate of the returned node. For the readout, the parent-rate ablation, and Huxley–Gödel Machine selection, the node is the earliest evaluated node whose steering-split solve count is maximal. The root row is the starting scaffold, and calibration best is the edit with the highest measured label among the run’s 96 calibration edits. The standard deviation is across the twelve runs. Bold marks the highest mean and underline marks the second highest.
<table><tr><td rowspan="2">Method</td><td colspan="2">Claude Sonnet 4.5</td><td colspan="2">GPT-40</td></tr><tr><td>HumanEval+↑</td><td>Polyglot ↑</td><td>HumanEval+ ↑</td><td>Polyglot ↑</td></tr><tr><td>Root</td><td> $8 2 . 3 \pm 2 . 2 2$ </td><td> $5 7 . 4 \pm 2 . 8 3$ </td><td> $8 1 . 2 \pm 2 . 2 5$ </td><td> $2 2 . 0 \pm 2 . 4 6$ </td></tr><tr><td>Calibration best</td><td> $8 5 . 1 \pm 3 . 6 8$ </td><td> $6 0 . 6 \pm 4 . 2 6$ </td><td> $8 2 . 8 \pm 2 . 5 3$ </td><td> $2 4 . 2 \pm 2 . 5 7$ </td></tr><tr><td>HGM</td><td> $\underline { { 8 6 . 5 \pm 4 . 6 8 } }$ </td><td> $\underline { { 6 4 . 0 \pm 3 . 8 8 } }$ </td><td> $\underline { { 8 5 . 1 \pm 2 . 4 3 } }$ </td><td> $2 5 . 7 \pm 2 . 8 9$ </td></tr><tr><td> $R _ { \mathrm { p a r } } { \bf - o n l y }$ </td><td> $\overline { { 8 5 . 3 \pm 1 . 9 2 } }$ </td><td> $\overline { { 6 0 . 4 \pm 4 . 4 6 } }$ </td><td> $\overline { { 8 4 . 2 \pm 2 . 6 6 } }$ </td><td> $\overline { { 2 3 . 0 \pm 2 . 9 3 } }$ </td></tr><tr><td>Full readout</td><td> ${ \bf 8 6 . 6 \pm 2 . 7 5 }$ </td><td> ${ \bf 6 5 . 2 \pm 4 . 2 2 }$ </td><td> $\mathbf { 8 7 . 0 \pm 2 . 7 0 }$ </td><td> ${ \bf 2 8 . 4 \pm 4 . 0 4 }$ </td></tr></table>

Table 2: Held-out solve rate of the child expanded at keep step 1, when the archive holds only the root. Entries are the mean ± sample standard deviation over twelve runs. Bold marks the highest mean.
<table><tr><td rowspan="2">Method</td><td colspan="2">Claude Sonnet 4.5</td><td colspan="2">GPT-40</td></tr><tr><td>HumanEval+↑</td><td>Polyglot ↑</td><td>HumanEval+↑</td><td>Polyglot ↑</td></tr><tr><td>Full readout</td><td> ${ \bf 8 3 . 8 \pm 2 . 0 9 }$ </td><td> ${ \bf 6 0 . 1 \pm 4 . 1 5 }$ </td><td> ${ \bf 8 2 . 3 \pm 3 . 0 2 }$ </td><td> ${ \bf 2 5 . 2 \pm 2 . 1 7 }$ </td></tr><tr><td> $R _ { \mathrm { p a r } } \mathrm { - o n l y }$ </td><td> $8 1 . 7 \pm 3 . 3 7$ </td><td> $5 7 . 4 \pm 4 . 0 0$ </td><td> $8 0 . 6 \pm 3 . 7 5$ </td><td> $2 2 . 2 \pm 3 . 7 2$ </td></tr></table>

Table 1 is the held-out score of the node returned after the keep steps. On HumanEval+ with GPT-4o that node scores 87.0%, the parent-rate node 84.2%, Huxley–Gödel Machine selection 85.1%, the calibration-best edit 82.8%, and the root 81.2%. The paired difference against the parent-rate node is 2.83 points, with interval [0.65, 5.01] and permutation probability 42/4096. The paired difference against Huxley–Gödel Machine selection is 1.92 points, with interval [−0.87, 4.70] and permutation probability 368/4096. On the same split Claude Sonnet 4.5 returns 86.6%, Huxley–Gödel Machine selection 86.5%, the parent-rate node 85.3%, and the calibration-best edit 85.1%. The paired difference against the parent-rate node is 1.25 points, interval [−0.83, 3.33], probability 490/4096, and against the calibration-best edit it is 1.50 points, interval [−1.68, 4.68], probability 697/4096. These comparisons are paired within runs. On Polyglot, Claude Sonnet 4.5 returns 65.2% for the readout, 64.0% for Huxley–Gödel Machine selection, 60.4% for parent-rate expansion, 60.6% for the calibration-best edit, and 57.4% for the root. GPT-4o returns 28.4%, 25.7%, 23.0%, 24.2%, and 22.0% on that split. Figure 1 plots the returned node.

## 4 RELATED WORK

Gödel machines accept a self-rewrite only after a proof of higher expected utility (Schmidhuber, 2007). Good (1966) and Clune (2019) discuss ultraintelligent machines and AI-generating algorithms. The Self-Taught Optimizer (STOP), Automated Design of Agentic Systems (ADAS), and the Self-Improving Coding Agent (SICA) edit a scaffold or a codebase; SICA reports 17% to 53% on a random SWE-bench Verified subset (Zelikman et al., 2023; Hu et al., 2024; Robeyns et al., 2025). The Gödel Agent searches a space of agent designs (Yin et al., 2025). With Claude 3.5 Sonnet (new), DGM reports 20.0% to 50.0% on a staged SWE-bench subset (Zhang et al., 2026). Wang et al. report 56.7% on SWE-Verified-60 with a different generator (Wang et al., 2026). In Table 1 that parent draw runs for the full horizon, and the reported node is the earliest evaluated node whose steering-split solve count is maximal. Liu et al. report 78.3% on SWE-Verified-60 and 93.3% on Polyglot with Qwen3.6-35B-A3B (Liu et al., 2026). Lai et al. aggregate descendant outcomes into clade-level Beta beliefs and select nodes by Thompson sampling (Lai et al., 2026). AIDE scores a candidate only after an evaluator runs it, and AIDE<sup>2</sup> keeps a self-rewrite only after a hidden evaluation (Jiang et al., 2025; Srikanth et al., 2026).

## 5 CONCLUSION

The readout ranks scaffold diffs by the parent’s steering-split solve rate and by eight coordinates of the diff, and every method returns the earliest evaluated node whose steering-split solve count is maximal. On held-out HumanEval+ with GPT-4o, that node scores 87.0%, against 84.2% for the same rule after parent-rate expansion. Huxley–Gödel Machine selection returns 85.1% on that split, the root is at 81.2%, and the calibration-best edit is at 82.8%. On the same split Claude Sonnet 4.5 returns 86.6%, beside 86.5% for Huxley–Gödel Machine selection, 85.3% for parent-rate expansion, and 85.1% for the calibration-best edit (Table 1, Figure 1).

## 6 LIMITATIONS

The readout is linear in eight coordinates of one projection per run, and the coefficients stay fixed through the 24 keep steps. The fit uses 96 scaffold edits labeled on 16 instances, and the archive keeps four nodes. The generators are Claude Sonnet 4.5 and GPT-4o (gpt-4o-2024-08-06), on HumanEval+ and Polyglot. The steering split that keeps the archive is disjoint from the held-out split in Tables 1 and 2.

## REPRODUCIBILITY STATEMENT

The proof of Theorem 1 is in Appendix A. Evaluation and aggregation details for the twelve runs are in Appendix B. Appendix C gives the scaffold instruction.

## ETHICS STATEMENT

The search can keep a lineage the checker later rejects. On these benchmarks the immediate score is the automatic checker.

## REFERENCES

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. doi: 10.48550/arxiv.2107.03374. URL https://arxiv.org/ abs/2107.03374.

Jeff Clune. AI-GAs: AI-generating algorithms, an alternate paradigm for producing general artificial intelligence. arXiv preprint arXiv:1905.10985, 2019. doi: 10.48550/arxiv.1905.10985. URL https://arxiv.org/abs/1905.10985.

Paul Gauthier. Aider polyglot leaderboard. https://aider.chat/docs/leaderboards/, 2024.

Irving John Good. Speculations concerning the first ultraintelligent machine. In Advances in Computers, pp. 31–88. 1966. doi: 10.1016/s0065-2458(08)60418-0.

Shengran Hu, Cong Lu, and Jeff Clune. Automated design of agentic systems. arXiv preprint arXiv:2408.08435, 2024. doi: 10.48550/arxiv.2408.08435. URL https://arxiv.org/abs/ 2408.08435.

Zhengyao Jiang, Dominik Schmidt, Dhruv Srikanth, Dixing Xu, Ian Kaplan, Deniss Jacenko, and Yuxiang Wu. AIDE: AI-driven exploration in the space of code. arXiv preprint arXiv:2502.13138, 2025. doi: 10.48550/arxiv.2502.13138. URL https://arxiv.org/abs/2502.13138.

Yaochu Jin. Surrogate-assisted evolutionary computation: Recent advances and future challenges. Swarm and Evolutionary Computation, 1(2):61–70, 2011. doi: 10.1016/j.swevo.2011.05.001.

Kezhao Lai, Yutao Lai, and Hai-Lin Liu. Beyond the node: Clade-level selection for efficient MCTS in automatic heuristic design. arXiv preprint arXiv:2602.00549, 2026. URL https: //arxiv.org/abs/2602.00549.

Changzhi Liu, Yilun Liu, Sikuan Yan, Volker Tresp, and Yunpu Ma. Mendel Gödel machine: Recursive self-improving coding agents via comparative evolution. arXiv preprint arXiv:2608.07645, 2026. doi: 10.48550/arxiv.2608.07645. URL https://arxiv.org/abs/2608.07645.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by ChatGPT really correct? rigorous evaluation of large language models for code generation. arXiv preprint arXiv:2305.01210, 2023. doi: 10.48550/arxiv.2305.01210. URL https://arxiv.org/ abs/2305.01210.

OpenAI. GPT-4o system card, 2024. URL https://arxiv.org/abs/2410.21276.

Maxime Robeyns, Martin Szummer, and Laurence Aitchison. A self-improving coding agent. arXiv preprint arXiv:2504.15228, 2025. doi: 10.48550/arxiv.2504.15228. URL https://arxiv.org/ abs/2504.15228.

Jürgen Schmidhuber. Gödel machines: Fully self-referential optimal universal self-improvers. In Artificial General Intelligence, Cognitive Technologies, pp. 199–226. Springer, 2007. doi: 10. 1007/978-3-540-68677-4\_7.

Dhruv Srikanth, Bingchen Zhao, Dixing Xu, Yuxiang Wu, and Zhengyao Jiang. Recursive selfimprovement of AI research agents. arXiv preprint arXiv:2609.26457, 2026. doi: 10.48550/ arxiv.2609.26457. URL https://arxiv.org/abs/2609.26457.

Wenyi Wang, Piotr Pi˛ekos, Li Nanbo, Firas Laakom, Yimeng Chen, Mateusz Ostaszewski, Mingchen Zhuge, and Jürgen Schmidhuber. Huxley-Gödel machine: Human-level coding agent development by an approximation of the optimal selfimproving machine. In International Conference on Learning Representations, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/ 821d20219c2f14850af1b5220f0ed13f-Abstract-Conference.html. arXiv:2510.21614.

Xunjian Yin, Xinyi Wang, Liangming Pan, Lin Li, Xiaojun Wan, and William Yang Wang. Gödel agent: A self-referential agent framework for recursively self-improvement. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 27890–27913, 2025. doi: 10.18653/v1/2025.acl-long.1354.

Eric Zelikman, Eliana Lorch, Lester Mackey, and Adam Tauman Kalai. Self-taught optimizer (STOP): Recursively self-improving code generation. arXiv preprint arXiv:2310.02304, 2023. doi: 10.48550/arxiv.2310.02304. URL https://arxiv.org/abs/2310.02304.

Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, and Jeff Clune. Darwin Gödel machine: Open-ended evolution of self-improving agents. In International Conference on Learning Representations, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/ aa5f5e6eb6f613ec412f1d948dfa21a5-Abstract-Conference.html. arXiv:2505.22954.

## A PROOFS

The argument below is the proof of Theorem 1.

## A.1 PROOF OF THE BIRTH-TIME LAW

Step A, mean ofa Beta law. Let $\alpha > 0 , \beta > 0$ , and write $\begin{array} { r } { I ( \alpha , \beta ) = \int _ { 0 } ^ { 1 } u ^ { \alpha - 1 } ( 1 - u ) ^ { \beta - 1 } d u \in ( 0 , \infty ) } \end{array}$ The mean of the corresponding Beta law is $I ( \alpha + 1 , \beta ) / I ( \alpha , \beta )$ . Define $\phi ( u ) = u ^ { \alpha } ( 1 - u ) ^ { \beta }$ for $u \in [ 0 , 1 ]$ , with $\phi ( 0 ) = \phi ( 1 ) = 0$ because $\alpha > 0$ and $\beta > 0 .$ . On (0, 1),

$$
\begin{array} { l } { { \phi ^ { \prime } ( u ) = \alpha u ^ { \alpha - 1 } ( 1 - u ) ^ { \beta } - \beta u ^ { \alpha } ( 1 - u ) ^ { \beta - 1 } } } \\ { { \qquad = \alpha u ^ { \alpha - 1 } ( 1 - u ) ^ { \beta - 1 } - ( \alpha + \beta ) u ^ { \alpha } ( 1 - u ) ^ { \beta - 1 } . } } \end{array}
$$

For $0 < \delta < 1 / 2$ , the fundamental theorem on $[ \delta , 1 - \delta ]$ gives $\begin{array} { r } { \int _ { \delta } ^ { 1 - \delta } \phi ^ { \prime } ( u ) d u = \phi ( 1 - \delta ) - \phi ( \delta ) } \end{array}$ The right-hand side tends to 0 as $\delta \downarrow 0 .$ . The two integrands on the expanded right-hand side are nonnegative, so the monotone convergence theorem yields α $\iota ( \alpha , \beta ) - ( \bar { \alpha + \beta } ) I ( \bar { \alpha + 1 } , \beta ) = 0$ , hence $I ( \alpha + \mathsf { \bar { 1 } } , \beta ) / I ( \alpha , \beta ) = { \alpha } / ( \alpha + \beta )$ . Assumption 1 gives $\alpha = \tau ( 1 + n _ { s } ) > 0$ and $\beta = \tau ( \dot { 1 } + n _ { f } ) > 0$

$$
\mathbb { E } \big [ \theta _ { n _ { s } , n _ { f } } ^ { ( \tau ) } \big ] = \frac { \tau ( 1 + n _ { s } ) } { \tau ( 1 + n _ { s } ) + \tau ( 1 + n _ { f } ) } = \frac { 1 + n _ { s } } { 2 + n _ { s } + n _ { f } } .
$$

At $( n _ { s } , n _ { f } ) = ( 0 , 0 )$ the parameters are $( \tau , \tau )$ and the mean is $1 / 2$ . The same ratio at parameters $( \alpha + 1 , \beta )$ yields $\mathbb { E } [ U ^ { 2 } ] = \alpha ( \alpha + 1 ) / ( ( \alpha + \beta ) ( \alpha + \beta + 1 ) )$ . Subtracting the squared mean produces

$$
\operatorname { V a r } ( U ) = { \frac { \alpha \beta } { ( \alpha + \beta ) ^ { 2 } ( \alpha + \beta + 1 ) } } .
$$

At $\alpha = \beta = \tau$ this is $1 / ( 4 ( 2 \tau + 1 ) )$ . For $\tau = 1$ the variance is $1 / 1 2$ . None of the parameters, the mean, or the variance reads X. Under Assumption 2 the counts are $( 0 , 0 )$ , so the sampling law is $\mathrm { B e t a } ( \tau , \tau )$ for every value of the code-change feature. The symbol $\rho ( 0 , 0 )$ would divide by zero and is left undefined.

## B EXPERIMENTAL DETAILS

Each cell of Tables 1 and 2 is twelve runs of Claude Sonnet 4.5 (claude-sonnet-4-5-20250929) or GPT-4o (gpt-4o-2024-08-06). The steering split is 48 HumanEval+ instances or 64 Polyglot instances, and the held-out split is the other 100 or 145 instances. Huxley–Gödel Machine selection draws the parent from $\mathrm { B e t a } ( 1 + n _ { s } , 1 + n _ { f } )$ on the steering counts of evaluated nodes. The calibration-best row is the edit with the highest fit-split label, scored on the held-out split.

A printed accuracy is the one-decimal rounding of 100 times the pooled solve count of the returned node divided by twelve times the split size, and the sample standard deviation is the ordinary standard deviation of the twelve run percentages. For the readout and the parent-rate ablation the candidates are the root and the 24 expanded children. For Huxley–Gödel Machine selection the candidates are the root and the expanded children, 144 on Polyglot and 176 on HumanEval+.

## C PROMPTS AND REPRODUCTIONS

Both generators are asked for a unified diff of the scaffold, at decoding temperature $1 . 0 ;$ a diff that does not apply cleanly is discarded and redrawn. The instruction is: “The scaffold is a system prompt and a loop that writes a solution, runs the unit tests, and retries once. Return a unified diff of that scaffold.”