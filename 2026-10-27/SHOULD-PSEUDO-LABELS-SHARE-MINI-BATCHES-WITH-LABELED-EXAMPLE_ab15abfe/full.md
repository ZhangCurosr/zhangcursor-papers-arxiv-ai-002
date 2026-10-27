# SHOULD PSEUDO-LABELS SHARE MINI-BATCHES WITH LABELED EXAMPLES?SEMI-SUPERVISED CONTINUAL ACTIVE LEARNING WITH LIMITED TRAINING COST

Shiqi Zhang , Tuomas Virtanen

Faculty of Information Technology and Communication Sciences Tampere University, Tampere, Finland

## ABSTRACT

Active learning (AL) reduces annotation by querying informative examples. Continual active learning (CAL) also cuts the training cost, the total number of gradient steps, by training one model across AL rounds with a fixed number of mini-batch steps per round. Pseudolabels add supervision without annotation, but they increase training cost. In CAL, where every round trains on a fixed number of minibatches, whether and how pseudo-labeled examples should share these mini-batches with labeled examples is therefore open. We provide the first formalization of this problem as two decisions after each query, namely (i) which unlabeled examples should enter training with a pseudo-label and (ii) what share of each mini-batch they should take. Our method, semi-supervised continual active learning (SSCAL), works with any acquisition function. It admits pseudo-labels based on distances from unlabeled examples to labeled neighbors, caps their share, and reweights the loss on labeled examples to keep their expected gradient contribution unchanged. On frozen embeddings, its pseudo-labeler updates incrementally without extra model forward passes. Across two audio datasets, two frozen encoders, six acquisition functions and ten seeds, SSCAL ranks first on average among four protocols with equal training steps, answering the question affirmatively.

Index Terms— Audio classification, active learning, continual learning, pseudo-labeling, semi-supervised learning

## 1. INTRODUCTION

Active learning (AL) selects informative examples to label and thereby reduces the annotation that model training needs. In audio, for instance, AL has studied clustering with label propagation [1], confidence sampling [2], and disagreement with diversity [3]. These advances target annotation efficiency. As models grow larger and deeper, the training cost, the total number of gradient steps spent over all AL rounds, becomes a concern of its own. In AL this cost also delays annotation. When the queries depend on the model, the annotator needs to wait for training in every round before labeling the next queries. Conventional AL protocols retrain for a fixed number of epochs in each AL round, so their training steps are proportional to the labeled set, which grows every round, and the cumulative cost over T rounds grows quadratically in T. To avoid this quadratic cost, continual active learning (CAL) [4] continues training the same model across rounds instead of reinitializing it (model continuation) and keeps training on earlier labels alongside the new ones (replay). The simplest replay in CAL combines new labeled examples with uniformly sampled earlier ones in every mini-batch [4]. Uniform replay, which samples every labeled example with equal probability, matches or outperforms costlier sampling and distillation methods in fixed-step continual learning [5].

Semi-supervised learning has long combined the supervised and pseudo-label losses within each mini-batch, with a scheduled weight on the pseudo-label loss [6] or a fixed ratio of unlabeled to labeled examples [7]. Its pseudo-labels come from the predictions of the model being trained [6, 7] or from labeled neighbors in a feature space, by label propagation over a graph of the examples [8, 9, 10] or by nearest-neighbor assignment from labeled support examples [11]. Semi-supervised AL borrows this practice to also train on unlabeled examples, through consistency regularization [12], pseudo-labels [13, 14] or self-training [15]. In both lines of work the labeled set is fixed during training and the training cost is usually unconstrained, so pseudo-labeled examples add computation on top of the labeled ones and crowd out none of them.

Inspired by both lines of work, we study whether pseudo-labels can also help CAL when every round is fixed at K gradient steps on stochastic mini-batches of m examples, which keeps the cumulative cost linear in T. We call this setting fixed-step CAL. Under this constraint, pseudo-labels replace labeled supervision instead of adding to it. A round samples exactly Km examples into its mini-batches, so each pseudo-labeled example among them replaces a labeled one. Labeled and pseudo-labeled examples thus share every fixed-size mini-batch. The displacement worsens once the labeled set grows past Km, after which a round samples fewer examples than those are labeled, and pseudo-labels, which can be incorrect, crowd out supervision that is already insufficient. Whether pseudo-labels are worth admitting, and what share of each mini-batch they may take, thus depends on how accurate and numerous the admitted pseudo-labels are and on how much labeled supervision they displace. Without a training constraint this trade-off does not arise, so the additive gains of semi-supervised learning do not settle it. We therefore examine this mini-batch sharing in AL at an unchanged annotation budget, where labels are scarce, arrive incrementally, and change every round which pseudo-labels the labeled examples support.

We formalize this problem as two decisions taken after each query and define the corresponding protocol. First, pseudo-label admission decides which unlabeled examples enter training with a pseudo-label. Second, mini-batch allocation decides how the minibatches are shared by setting the sampling distribution over labeled and pseudo-labeled examples. We propose semi-supervised continual active learning (SSCAL), a method that works with any acquisition function. It admits a pseudo-label when an unlabeled example lies closer to two labeled positives of a class than to the nearest labeled negative of that class, caps the sampling share of pseudo-labeled examples, and reweights the loss on labeled examples to keep their expected gradient contribution equal to that of uniform replay. Its pseudo-labeler runs incrementally on frozen embeddings without extra model forward passes. Among four protocols with equal training steps, SSCAL attains the highest area under the learning curve in 202 of 240 paired runs and a mean rank of 1.18.

Our contributions are (i) to our knowledge, the first study to bring pseudo-labels into fixed-step CAL, formalized as pseudo-label admission and mini-batch allocation with a corresponding protocol; (ii)

SSCAL, an acquisition-function-agnostic method for both decisions; and (iii) experimental validation that answers the mini-batch sharing question affirmatively.

## 2. PROBLEM AND PROTOCOL

Pool-based AL starts with a pool $\mathcal { D } = \{ x _ { 1 } , . . . , x _ { N } \}$ of N examples $x _ { i }$ , a fixed label space of $\bar { C }$ classes, and an annotator who returns the label $y _ { i }$ of any queried example. All examples are available for training and for computing queries, whereas labels are available only for queried examples. Pseudo-labels may use the acquired labels, the examples themselves and the current model. AL round t starts from the labeled set $L _ { t - 1 }$ , where $L _ { 0 }$ is a random initial labeled set of $n _ { 0 }$ examples, acquires labels for a query set $Q _ { t }$ of b unlabeled examples chosen by the acquisition function, and sets $L _ { t } = L _ { t - 1 } \dot { \cup } Q _ { i }$ and $U _ { t } = \mathcal { D } \setminus L _ { t } .$ . We write $H _ { t } = L _ { t - 1 }$ for the earlier labels, so $L _ { t } = H _ { t } { \dot { \cup } } Q _ { t }$ . The model $f _ { \theta }$ is trained to output class logits, and its parameters and optimizer state carry over rounds. Each round performs K gradient steps, each on a stochastic mini-batch of m examples. Since K and m are fixed, each round samples Km examples. The acquisition function is an external choice held fixed over all rounds, and its queries may depend on the current model.

After each query, a policy pair $\Pi = \left( \pi _ { p } , \pi _ { s } \right)$ makes the two decisions. The admission policy $\pi _ { p }$ selects a set $P _ { t } \subseteq U _ { t }$ of pseudolabeled examples and assigns each a pseudo-label target $\widehat { z } _ { t , i }$ . The allocation policy $\pi _ { s }$ sets the distribution from which mini-batches are sampled. Training uses the union $A _ { t } = H _ { t } { \dot { \cup } } Q _ { t } { \dot { \cup } } P _ { t } = L _ { t } { \dot { \cup } } P _ { t }$ of labeled and pseudo-labeled examples, each with a target $z _ { t , i }$ that is the acquired label $y _ { i }$ on $L _ { t }$ and the pseudo-label $\widehat { z } _ { t , i }$ on $P _ { t }$ Pseudo-labeled examples consume no annotation budget and stay in $U _ { t } .$ , so they remain queryable. To simplify notation, the step index within a round is omitted.

The policy $\pi _ { s }$ sets a distribution $\nu _ { t }$ over $A _ { t }$ , and the $m$ examples of a mini-batch are sampled i.i.d. (with replacement) from $\nu _ { t } ,$ , so an example may appear in one mini-batch more than once. Example i therefore appears $m \nu _ { t , i }$ times in a mini-batch in expectation, that is, it occupies an expected fraction $\nu _ { t , i }$ of every mini-batch, which we call its sampling share. The sampling share $\dot { \alpha } _ { t } ^ { G }$ of a group $G \in \{ H , Q , P \}$ is the expected fraction of a mini-batch that its members occupy together, and the three group shares sum to one because $\nu _ { t }$ is a distribution over the disjoint union $A _ { t }$ of the three groups,

$$
\alpha _ { t } ^ { G } = \sum _ { i \in G _ { t } } \nu _ { t , i } , \qquad \alpha _ { t } ^ { H } + \alpha _ { t } ^ { Q } + \alpha _ { t } ^ { P } = 1 .\tag{1}
$$

Any share given to pseudo-labeled examples is thus taken from labeled examples, which is how the two share every mini-batch.

Admission, targets and sampling shares jointly determine the expected gradient of one step. Let $\ell _ { i } = \ell ( \theta ; x _ { i } , z _ { t , i } )$ be the loss of exam ple i on its target, and let $w _ { t , i }$ be the loss weight of example i, which $\pi _ { s }$ may set to compensate for non-uniform sampling $( w _ { t , i } = 1$ unless stated). For the weighted mini-batch loss $\begin{array} { r } { \widehat { R } _ { t } = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } w _ { t , i _ { j } } \ell _ { i _ { j } } } \end{array}$ over the sampled examples $i _ { 1 } , \dots , i _ { m }$

$$
\mathbb { E } [ \nabla \widehat { R } _ { t } ] = \sum _ { i \in L _ { t } } \nu _ { t , i } w _ { t , i } \nabla \ell _ { i } + \sum _ { i \in P _ { t } } \nu _ { t , i } w _ { t , i } \nabla \ell _ { i } ,\tag{2}
$$

where the first sum is the contribution of labeled examples and the second that of pseudo-labeled examples.

Since the number of AL rounds available in practice is unknown in advance, the objective is the normalized area under the learning curve (AULC) over all T rounds of a run, which averages the task metric over all labeled-set sizes,

$$
J ( \Pi ) = \mathbb { E } _ { \Pi } \left[ \frac { \sum _ { t = 2 } ^ { T } ( n _ { t } - n _ { t - 1 } ) ( M _ { t } + M _ { t - 1 } ) / 2 } { n _ { T } - n _ { 1 } } \right] ,\tag{3}
$$

where $n _ { t } = | L _ { t } | , M _ { t }$ is the task metric after round t on test data held out from the acquisition function and the policies, and the expectation is over the randomness of a run. With label counts, K and m fixed, policies are compared at equal annotation and training steps.

## 3. METHOD

## 3.1. Neighbor-margin admission

Independently of acquisition scores, SSCAL admits pseudo-labels by propagating acquired labels to nearby unlabeled examples in a frozen embedding space, a nearest-neighbor form of label propagation in semi-supervised learning [8, 9, 10]. Pretrained embeddings tend to place examples of the same class close together, so an unlabeled example likely shares the class of labeled examples that lie much closer to it than all other labeled examples. Labeled neighbors are preferred to the predictions of the model being trained [6, 7] because acquired labels are reliable and grow every round, whereas the model is trained for only K steps per round and its errors would feed back into its own training [16]. On $\ell _ { 2 }$ -normalized embeddings, a labeled example is a positive of class c if its acquired label contains c and a negative of class c otherwise. For an unlabeled candidate i, let $\Delta _ { i , c } ^ { + , 2 }$ and $\Delta _ { i , c } ^ { - }$ be the squared Euclidean distances from i to its secondnearest positive and to its nearest negative of class c. SSCAL admits class c as a pseudo-label of candidate i when the second-nearest positive is closer than the nearest negative by a margin set by a threshold $0 < \rho \leq 1$

$$
\begin{array} { c } { r _ { i , c } = \Delta _ { i , c } ^ { + , 2 } / \Delta _ { i , c } ^ { - } , \qquad a _ { i , c } = \mathbf { 1 } \{ r _ { i , c } < \rho ^ { 2 } \} , } \\ { P _ { t } = \{ i \in U _ { t } : \sum _ { c } a _ { i , c } > 0 \} , } \end{array}\tag{4}
$$

where ${ \boldsymbol { r } } _ { i , c }$ is the distance ratio and ${ a } _ { i , c }$ the admission indicator. Only candidates with at least one admitted class enter $P _ { t }$ and thus training, since the others lack the labeled support for a reliable pseudo-label. Requiring two positives means that one mislabeled or atypical neighbor cannot admit a pseudo-label on its own, and the nearest negative bounds how far the positives may lie. Class c is not admitted $( a _ { i , c } = 0 )$ when fewer than two positives or no negative of class c has been labeled, or when the nearest negative is at distance zero. For single-label tasks, at most one class is admitted per candidate, since its second-nearest positive must be closer than the nearest labeled example of every other class. The admitted class is that of the nearest labeled neighbor and is the cross-entropy target $\widehat { z } _ { t , i }$ . For multi-label tasks, the pseudo-label loss $\begin{array} { r } { \ell _ { i } = C ^ { - 1 } \dot { \sum } _ { c } a _ { i , c } ^ { - } \mathrm { B C E } ( f _ { \theta } ( x _ { i } ) _ { c } , 1 ) } \end{array}$ supervises only the admitted classes, each with target one, as the other classes remain unknown. Labeled examples are trained with crossentropy in single-label tasks and with binary cross-entropy averaged over all C classes in multi-label tasks.

Labeled negatives near a candidate tighten admission and thereby bound pseudo-label errors. For class c, suppose every candidate that does not belong to c is at Euclidean distance at least $s > 0$ from every positive of c and within $h > 0$ of some negative of c. Then $\Delta _ { i . c } ^ { + , \tilde { 2 } } \geq s ^ { 2 }$ and $\Delta _ { i , c } ^ { - } \le h ^ { 2 }$ , so $\rho h \leq$ s makes the strict inequality in (4) impossible and no false positive is admitted for c. Because admission uses only labeled examples as references, a newly acquired negative can also revoke an earlier admission.

## 3.2. Mini-batch allocation

In AL the labeled examples are few and sparse relative to the size of the pool, so the pseudo-labels propagated from them are of limited reliability. To keep most of every mini-batch for the more reliable acquired labels, SSCAL caps the sampling share of pseudo-labeled examples at a hyperparameter $\beta \in [ 0 , \bar { 1 } )$ ), reserving at least $1 - \beta$ for labeled examples, and samples uniformly within each group of $A _ { t }$

Table 1. AULC (%, higher is better), mean±s.d. over ten seeds, each averaging six acquisition functions. Parentheses give mean ranks among four fixed-step protocols within paired runs. <sup>∗</sup> marks $p < 0 . 0 0 1$ and ns marks $p = 0 . 4 2$ (two-sided paired Wilcoxon, 60 pairs per setting, unadjusted). The epoch<sup>†</sup> protocol runs 10.5× as many training steps, for reference only, and has no rank (–).
<table><tr><td>Setting</td><td>uniform</td><td>equal</td><td>new-only</td><td>SSCAL</td><td> $\mathrm { e p o c h } ^ { \dagger }$ </td></tr><tr><td>BSD10K / CLAP</td><td> $5 7 . 7 9 { \pm } 0 . 6 7 ( 2 . 0 8 )$ </td><td>56.92±0.53 (2.90)</td><td> $5 1 . 2 0 { \pm } 0 . 1 8 \left( 4 . 0 0 \right)$ </td><td> $\mathbf { 5 9 . 7 9 { \pm 0 . 5 3 ( 1 . 0 2 ) } ^ { \ast } }$ </td><td> $5 9 . 0 1 { \pm } 0 . 4 2 \left( - \right)$ </td></tr><tr><td>BSD10K / PANNs</td><td> $5 0 . 7 0 { \scriptstyle \pm 0 . 5 5 ( 1 . 6 5 ) }$ </td><td> $4 9 . 6 6 { \pm } 0 . 3 9 ( 2 . 7 7 )$ </td><td> $3 9 . 4 0 { \pm } 0 . 3 6 ( 4 . 0 0 )$ </td><td> ${ \bf 5 0 . 8 6 { \pm } 0 . 6 2 { ( 1 . 5 8 ) } ^ { \mathrm { n s } } }$ </td><td> $5 2 . 2 1 { \pm } 0 . 6 5 \left( - \right)$ </td></tr><tr><td>FSD50K / CLAP</td><td> $4 4 . 4 0 { \pm } 0 . 6 0 ( 2 . 2 7 )$ </td><td> $4 3 . 9 4 { \pm } 0 . 5 5 ( 2 . 7 3 )$ </td><td> $3 6 . 1 5 { \pm } 0 . 6 0 \left( 4 . 0 0 \right)$ </td><td> $\mathbf { 4 8 . 9 5 { \pm } 0 . 5 8 ( 1 . 0 0 ) ^ { \ast } }$ </td><td> $5 6 . 1 6 { \pm } 0 . 3 3 \left( - \right)$ </td></tr><tr><td>FSD50K / PANNs</td><td>45.69±0.21 (2.01)</td><td> $4 5 . 4 2 { \pm } 0 . 2 6 ( 2 . 8 8 )$ </td><td> $3 6 . 4 3 { \pm } 0 . 3 8 \ : ( 4 . 0 0 )$ </td><td> $\mathbf { 4 7 . 1 5 { \pm 0 . 1 7 ( 1 . 1 2 ) } ^ { \ast } }$ </td><td>43.46±0.15(-)</td></tr><tr><td>First / 240</td><td>34</td><td>4</td><td>0</td><td>202</td><td></td></tr></table>

Table 2. AULC (%), mean±s.d. over ten seeds, each averaging four settings. The overall row also averages all six acquisition functions within each seed. Values combine BSD10K macro accuracy and FSD50K macro mAP. Parentheses give mean within-run ranks. The epoch<sup>†</sup> protocol runs 10.5× as many training steps, for reference only, and has no rank (–).
<table><tr><td>Acquisition</td><td>uniform</td><td>equal</td><td>new-only</td><td>SSCAL</td><td> $\mathrm { e p o c h } ^ { \dagger }$ </td></tr><tr><td>random</td><td> $5 1 . 4 4 { \pm } 0 . 4 8 ( 1 . 9 8 )$ </td><td>50.84±0.41 (2.90)</td><td> $4 4 . 6 4 { \pm } 0 . 3 4 ( 4 . 0 0 )$ </td><td> $\pm 2 . 6 4 \pm 0 . 5 0 ( 1 . 1 3 )$ </td><td> $5 3 . 0 0 { \scriptstyle \pm 0 . 4 8 \ ( - ) }$ </td></tr><tr><td>entropy</td><td>45.28±0.47 (2.08)</td><td>44.77±0.55 (2.68)</td><td> $3 3 . 1 7 { \pm } 0 . 3 9 ( 4 . 0 0 )$ </td><td> ${ \bf 4 8 . 6 8 { \scriptstyle \pm 0 . 6 5 ( 1 . 2 5 ) } }$ </td><td>50.32±0.39 (-)</td></tr><tr><td>margin</td><td>46.20±0.50 (2.15)</td><td>45.77±0.51 (2.65)</td><td>35.65±0.80 (4.00)</td><td> $\mathbf { 4 9 . 6 4 \pm 0 . 6 3 ( 1 . 2 0 ) }$ </td><td>51.35±0.63 (-)</td></tr><tr><td>Core-set</td><td>49.86±0.39(1.78)</td><td>48.60±0.33 (2.98)</td><td>40.07±0.28 (4.00)</td><td> $\mathbf { 5 1 . 2 3 \bot 0 . 5 3 ( 1 . 2 5 ) }$ </td><td>52.36±0.43 (-)</td></tr><tr><td>TypiClust</td><td> $5 3 . 2 2 { \pm } 0 . 3 3 ( 1 . 8 5 )$ </td><td>52.52±0.37 (3.00)</td><td> $4 6 . 0 0 { \scriptstyle \pm 0 . 2 4 } \left( 4 . 0 0 \right)$ </td><td> $\mathbf { 5 4 . 3 5 { \pm } 0 . 3 3 ( 1 . 1 5 ) }$ </td><td>54.63±0.33 (-)</td></tr><tr><td>BADGE</td><td> $5 1 . 8 6 { \pm } 0 . 4 6 ( 2 . 1 9 )$ </td><td>51.42±0.24 (2.71)</td><td> $4 5 . 2 3 { \pm } 0 . 2 4 ( 4 . 0 0 )$ </td><td> $\mathbf { 5 3 . 5 8 { \overset { \triangledown } { = } } 0 . 2 9 ( 1 . 1 0 ) }$ </td><td>54.60±0.41 (-)</td></tr><tr><td>overall</td><td> $4 9 . 6 4 { \pm } 0 . 2 7 \left( 2 . 0 0 \right)$ </td><td> $4 8 . 9 9 { \pm } 0 . 2 2 \left( 2 . 8 2 \right)$ </td><td> $4 0 . 7 9 { \pm } 0 . 2 6 \left( 4 . 0 0 \right)$ </td><td> $\mathbf { 5 1 . 6 9 { \pm } 0 . 3 2 ( 1 . 1 8 ) }$ </td><td>52.71±0.25(−)</td></tr></table>

![](images/f9c0df58e15fbee5eda4f37f92bb772738ae56fcd8ce3fb06deb79f23eed0a86.jpg)

![](images/a4ab3cf521db44bcad57ff5357567af04b9b0cf85772de0455e9ef08dd813054.jpg)

![](images/cbf15d8a769351cd3982aced14f4bbda4cda9d7c60f4932879a4c074c7a8dbb0.jpg)  
Fig. 1. Aggregated test learning curves for the four settings against the labeled fraction of the training set. Each seed first averages six acquisition functions. Lines and shading show the mean and one standard deviation of these curves across ten seeds. The epoch<sup>†</sup> protocol runs 10.5× as many training steps and is shown for reference only.

$$
\begin{array} { c } { { \alpha _ { t } ^ { P } = \operatorname* { m i n } \Biggl \{ \beta , \frac { \left| P _ { t } \right| } { \left| L _ { t } \right| + \left| P _ { t } \right| } \Biggr \} , } } \\  { \nu _ { t , i } = \Biggl \{ \begin{array} { l l } { { \displaystyle ( 1 - \alpha _ { t } ^ { P } ) / | L _ { t } | , } } & { { i \in L _ { t } , } } \\ { { \displaystyle \alpha _ { t } ^ { P } / | P _ { t } | , } } & { { i \in P _ { t } , } } \end{array} } \end{array}\tag{5}
$$

where the minimum of the two keeps the share of a small $P _ { t }$ at no more than its proportion among all training examples. Since labeled examples are now sampled with total probability $\bar { \bf \Phi } _ { 1 } - \alpha _ { t } ^ { P }$ instead of one, their losses are weighted by $w _ { t , i } \stackrel { - } { = } 1 / ( 1 - \alpha _ { t } ^ { P } )$ . Pseudo-labeled examples have no reference contribution to restore and keep unit weight. Admission, targets and shares stay fixed within a round.

At any parameter value, this weighting preserves the expected gradient contribution of the labeled examples exactly as under uniform replay,

$$
\mathbb { E } [ \nabla \widehat { R } _ { t } ] = \frac { 1 } { \vert L _ { t } \vert } \sum _ { i \in L _ { t } } \nabla \ell _ { i } + \frac { \alpha _ { t } ^ { P } } { \vert P _ { t } \vert } \sum _ { i \in P _ { t } } \nabla \ell _ { i } ,\tag{6}
$$

whose first term is the expected gradient of uniform replay over $L _ { t }$ and whose second term adds the pseudo-label gradient with total weight $\alpha _ { t } ^ { P }$ . Labeled examples are still sampled $1 - \alpha _ { t } ^ { P }$ times as often as under uniform replay, so the K steps can follow a different trajectory. This is the displaced supervision studied here. The cap β and weight $w _ { t , i }$ play the roles of the unlabeled ratio and loss weight in additive semi-supervised learning [6, 7], and here jointly bound the displaced labeled supervision and restore its expected gradient. With empty P<sub>t</sub> or $\beta = 0 ,$ , SSCAL reduces to uniform replay.

## 3.3. Incremental cost

The frozen-embedding setting lets the pseudo-labeler update incrementally as labels are acquired. For every candidate and class it stores three distances, to the two nearest positives and the nearest negative, and each round computes distances only to the b newly labeled examples. A round thus costs $O ( N b ( d + C ) )$ for embedding dimension d, independent of the size of the labeled set. It needs no extra forward pass of $f _ { \theta }$ and leaves the K steps per round untouched. Table 4 reports the measured costs.

## 4. EXPERIMENTS

## 4.1. Setup

BSD10K (23 classes, macro accuracy) [17, 18] and FSD50K (200 classes, macro mean average precision, mAP) [19] represent singlelabel and multi-label audio classification. Fixed train, validation and test splits contain 8,767, 1,098 and 1,091 examples for BSD10K (a stratified 8:1:1 split) and 36,785, 4,169 and 10,230 for FSD50K (the official split excluding 13 clips longer than 30 s). The annotator is simulated by the dataset labels, revealed as examples are queried. Frozen embeddings come from contrastively pretrained CLAP (Transformer, 512 dimensions) [20] and supervised-pretrained PANNs CNN14 (CNN, 2,048 dimensions) [21], and a standard GELU MLP head $d  2 5 6  C$ shared by all protocols is trained with Adam at learning rate $5 \times 1 0 ^ { - 3 }$ and zero weight decay.

A run consists of $T = 4 0 \mathrm { \ A I }$ rounds with $K = 2 0$ steps per round and $b = m = n _ { 0 } = 4 4$ for BSD10K or 184 for FSD50K, that $\mathrm { i s } , 0 . 5 \%$ of the respective training set. These values keep annotation sparse and training short. The labeled set grows from 0.5% to 20.5% of the training set, and the $K m = 2 0 b$ examples sampled per round cover it ten times in the first round and under half in the last. Each of ten paired seeds fixes the same random initial labeled set $L _ { 0 }$ and model initialization for all protocols and acquisition functions. As a reference from the conventional AL setting, epoch makes $E = 1 0$ passes over $L _ { t }$ per round, 10.5× as many steps over a run. Before the first AL round, all protocols train on $\dot { L } _ { 0 } .$ , fixed-step protocols for K steps (an identical paired warm-up, so AULC in (3) starts at the first AL round) and epoch for E passes. The acquisition functions are random, entropy, margin [22], Core-set [23], TypiClust [24] and BADGE [25], spanning uncertainty, representativeness and hybrid methods. For multi-label tasks, entropy sums binary entropies, margin averages the five smallest distances to probability 0.5, and BADGE uses BCE gradient embeddings. Entropy, margin and BADGE query with the current model, so protocols under the same seed and acquisition function may acquire different labels and the comparison covers the complete loop of querying, pseudo-labeling and training.

The baselines are uniform, which samples $L _ { t }$ uniformly, equal, which gives sampling share $1 / 2$ each to $\bar { H _ { t } }$ and $Q _ { t }$ , and new-only, which samples only Q<sub>t</sub>. All use the same K steps of m examples per round as SSCAL. The default $( \rho , \beta ) = ( 1 . 0 , 0 . 4 )$ maximizes mean validation AULC over all settings on the grid $\rho \in \{ 0 . 6 , 0 . 8 , 1 . 0 \}$ $\beta \in \{ 0 . 1 , 0 . 2 , 0 . 4 \}$ (Table 3). The main comparison contains 1,200 runs (3,120 with the full grid).

## 4.2. Performance across settings and acquisition functions

SSCAL significantly improves over uniform in three of the four dataset/encoder settings (Table 1). AULC rises by 2.00, 4.55 and 1.46 points for BSD10K/CLAP, FSD50K/CLAP and FSD50K/PANNs (paired two-sided Wilcoxon, $p \leq 5 \times 1 0 ^ { - 1 0 } )$ , while BSD10K/PANNs gains only 0.17 points $( p = 0 . 4 2 )$ . SSCAL achieves the best mean rank with each acquisition function (Table 2), ten-seed means exceed uniform in 21 of the 24 dataset/encoder/acquisition configurations, and entropy and margin gain most, from 45.28 to 48.68 and from 46.20 to 49.64 AULC.

Gains emerge early and persist across rounds (Fig. 1). The equal and new-only baselines, which change only the sampling shares among labeled examples, reduce mean AULC across the four settings by 0.27–1.03 and 6.59–11.30 points, so the benefit comes from sharing the mini-batches with pseudo-labeled examples. Interestingly, the epoch reference even falls below uniform on FSD50K/PANNs, so extra steps over the full labeled set do not guarantee gains even without the fixed-step constraint.

## 4.3. Admission and sampling sensitivity

To assess hyperparameter sensitivity we run all nine admission and sampling parameter combinations. Every one achieves the best mean rank (Table 3), with mean ranks of 1.08–1.38 and aggregate AULC gains of 0.12–2.04 points. Increasing $\rho$ improves aggregate gains $( \rho = 1$ still requires both positives strictly closer than the nearest negative), and a larger $\beta$ helps at $\rho = 1$ but not consistently under strict admission.

Table 3. Paired ∆AULC versus uniform (points), mean±s.d. across ten seeds after averaging all 24 configurations within each seed. Ranks in parentheses compare seed means against uniform, equal and new-only within configurations (Tables 1–2 use run ranks). Bold marks the best validation AULC.
<table><tr><td> $\rho$ </td><td> $\beta = 0 . 1$ </td><td> $\beta = 0 . 2$ </td><td> $\beta = 0 . 4$ </td></tr><tr><td>0.6</td><td> $+ 0 . 1 5 { \pm } 0 . 1 1 \left( 1 . 2 5 \right)$ </td><td> $+ 0 . 1 6 { \pm } 0 . 1 1 \left( 1 . 3 8 \right)$ </td><td> $+ 0 . 1 2 { \pm } 0 . 1 4 ( 1 . 3 8 )$ </td></tr><tr><td>0.8</td><td> $+ 0 . 4 8 { \pm } 0 . 0 9 \left( 1 . 0 8 \right)$ </td><td> $+ 0 . 6 1 { \pm } 0 . 0 8 \left( 1 . 1 7 \right)$ </td><td> $+ 0 . 7 0 { \pm } 0 . 1 1 \left( 1 . 2 1 \right)$ </td></tr><tr><td>1.0</td><td> $+ 1 . 0 1 { \pm } 0 . 1 2 \left( 1 . 1 3 \right)$ </td><td> $+ 1 . 5 0 { \pm } 0 . 1 5 ( 1 . 2 1 ) $ </td><td> $\mathbf { + 2 . 0 4 { \pm 0 . 1 5 } ( 1 . 1 3 ) }$ </td></tr></table>

Table 4. Default SSCAL, with B and F for BSD10K and FSD50K. Prec. is pooled positive-label precision and Cov. is $| P _ { t } | / | U _ { t } |$ , both in ${ \% } ,$ averaged with $| P _ { t } |$ over AL rounds and runs. ∆Wall is trainingside wall time (training + policy) over initialization and 40 AL rounds in excess of uniform $( 0 . \bar { 9 5 } { - 2 . 2 7 } { \mathrm { s } } )$ , averaged over 60 runs. SSCAL and uniform run 820 steps, epoch<sup>†</sup> 8,610 (10.5×, reference only).
<table><tr><td rowspan="2">Setting</td><td rowspan="2">Prec.</td><td rowspan="2"> $\operatorname { C o v . }$ </td><td rowspan="2"> $| P _ { t } |$ </td><td colspan="2">∆Wall (s)</td></tr><tr><td>SSCAL</td><td>epoch†</td></tr><tr><td>B/CLAP</td><td>83.2</td><td>52.8</td><td>4092</td><td>+1.36</td><td>+7.43</td></tr><tr><td>B /PANNs</td><td>75.4</td><td>48.0</td><td>3730</td><td>+0.94</td><td>+7.91</td></tr><tr><td>F /CLAP</td><td>95.2</td><td>85.7</td><td>28035</td><td>+1.36</td><td>+9.85</td></tr><tr><td>F / PANNs</td><td>87.2</td><td>71.3</td><td>23331</td><td>+0.25</td><td>+15.95</td></tr></table>

Counterintuitively, relaxing admission lowers pseudo-label precision while improving AULC in all four settings. At $\beta = 0 . 4 .$ , raising $\rho$ from 0.6 to 1.0 lowers precision from 97.1/91.0/99.9/99.3% to 83.2/75.4/95.2/87.2% in the order of Table 1, while AULC rises from 57.97/50.69/44.75/45.64 to 59.79/50.86/48.95/47.15. Pseudo-label precision alone thus does not predict the task benefit. At the same cap $\beta = 0 . 4$ , the mean sampling share $\alpha _ { t } ^ { P }$ rises from 0.19–0.38 to 0.40 and broader admission brings more unlabeled inputs into training, so this expanded coverage and larger sampling share may explain the gains.

## 4.4. Diagnostics, cost and scope

At the default parameters, pseudo-label quality depends on how well the representation separates the classes (Table 4). BSD10K/PANNs has the lowest precision (75.4%) and a neutral performance change. Precision is computed offline by replaying each configuration’s own query trajectory, so unacquired labels never enter the protocol.

Over initialization and 40 AL rounds, SSCAL and uniform run 820 steps and epoch runs 8,610. Relative to uniform, SSCAL adds 0.25–1.36 s of training-side wall time and epoch adds 7.43–15.95 s, so the extra time behind SSCAL’s gains stays far below that of the conventional epoch protocol.

## 5. CONCLUSION AND FUTURE DIRECTION

We formalize pseudo-label admission and mini-batch allocation in fixed-step CAL, define its protocol, and answer the sharing question affirmatively. The acquisition-function-agnostic SSCAL method improves 21 of 24 audio configurations and ranks best on average under all six acquisition functions. Future work could extend SSCAL to end-to-end training, refreshing distances as the embeddings change.

## 6. COMPLIANCE WITH ETHICAL STANDARDS

This study is limited to computational analysis of the existing public BSD10K [17] and FSD50K [19] datasets. No new experiments involving human or animal subjects were conducted.

## 7. ACKNOWLEDGEMENTS

This work received funding from the European Union’s Horizon Europe programme under the Marie Skłodowska-Curie grant agreement No. 101116715 (BioacousticAI).

## 8. REFERENCES

[1] S. Zhao, T. Heittola, and T. Virtanen, “Active learning for sound event classification by clustering unlabeled data,” in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017, pp. 751–755.

[2] Y. Wang, A. E. M. Mendez, M. Cartwright, and J. P. Bello, “Active learning for efficient audio annotation and classification with a large amount of unlabeled data,” in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2019.

[3] S. Zhang and T. Virtanen, “Hybrid disagreement-diversity active learning for bioacoustic sound event detection,” in Proc. European Signal Processing Conference (EUSIPCO), 2025, pp. 131–135.

[4] A. Das, G. Bhatt, M. Bhalerao, V. Gao, R. Yang, and J. Bilmes, “Accelerating batch active learning using continual learning techniques,” Transactions on Machine Learning Research, 2023.

[5] A. Prabhu, H. A. Al Kader Hammoud, P. K. Dokania, P. H. S. Torr, S.-N. Lim, B. Ghanem, and A. Bibi, “Computationally budgeted continual learning: What does matter?” in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 3698–3707.

[6] D.-H. Lee, “Pseudo-label: The simple and efficient semisupervised learning method for deep neural networks,” in ICML Workshop on Challenges in Representation Learning, 2013.

[7] K. Sohn, D. Berthelot, N. Carlini, Z. Zhang, H. Zhang, C. A. Raffel, E. D. Cubuk, A. Kurakin, and C.-L. Li, “FixMatch: Simplifying semi-supervised learning with consistency and confidence,” in Advances in Neural Information Processing Systems, vol. 33, 2020.

[8] X. Zhu and Z. Ghahramani, “Learning from labeled and unlabeled data with label propagation,” Carnegie Mellon University, Technical Report CMU-CALD-02-107, 2002.

[9] D. Zhou, O. Bousquet, T. N. Lal, J. Weston, and B. Schölkopf, “Learning with local and global consistency,” in Advances in Neural Information Processing Systems, vol. 16, 2004.

[10] A. Iscen, G. Tolias, Y. Avrithis, and O. Chum, “Label propagation for deep semi-supervised learning,” in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 5070–5079.

[11] M. Assran, M. Caron, I. Misra, P. Bojanowski, A. Joulin, N. Ballas, and M. Rabbat, “Semi-supervised learning of visual features by non-parametrically predicting view assignments with support samples,” in Proc. IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 8443–8452.

[12] M. Gao, Z. Zhang, G. Yu, S. O. Arik, L. S. Davis, and T. Pfister, “Consistency-based semi-supervised active learning: Towards

minimizing labeling cost,” in Proc. European Conference on Computer Vision (ECCV), 2020.

[13] O. Siméoni, M. Budnik, Y. Avrithis, and G. Gravier, “Rethinking deep active learning: Using unlabeled data at model training,” in Proc. International Conference on Pattern Recognition (ICPR), 2020, pp. 1220–1227.

[14] T. Yin, N. Liu, and H. Sun, “Towards cost-effective learning: A synergy of semi-supervised and active learning,” in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 10 163–10 172.

[15] S. Shishkin, D. Hollosi, S. Doclo, and S. Goetze, “Active learning for sound event classification using Monte-Carlo Dropout and PANN embeddings,” in Proc. Detection and Classification of Acoustic Scenes and Events Workshop (DCASE), 2021, pp. 150–154.

[16] E. Arazo, D. Ortego, P. Albert, N. E. O’Connor, and K. McGuinness, “Pseudo-labeling and confirmation bias in deep semisupervised learning,” in Proc. International Joint Conference on Neural Networks (IJCNN), 2020.

[17] P. Anastasopoulou, J. Torrey, X. Serra, and F. Font, “Heterogeneous sound classification with the Broad Sound Taxonomy and Dataset,” in Proc. Detection and Classification ofAcoustic Scenes and Events Workshop (DCASE), 2024.

[18] P. Anastasopoulou, F. A. Dal Rí, X. Serra, and F. Font, “Hierarchical and multimodal learning for heterogeneous sound classification,” in Proc. Detection and Classification ofAcoustic Scenes and Events Workshop (DCASE), 2025.

[19] E. Fonseca, X. Favory, J. Pons, F. Font, and X. Serra, “FSD50K: An open dataset of human-labeled sound events,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, 2022.

[20] Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, and S. Dubnov, “Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation,” in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023.

[21] Q. Kong, Y. Cao, T. Iqbal, Y. Wang, W. Wang, and M. D. Plumbley, “PANNs: Large-scale pretrained audio neural networks for audio pattern recognition,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, 2020.

[22] B. Settles, “Active learning literature survey,” University of Wisconsin–Madison, Computer Sciences Technical Report 1648, 2009.

[23] O. Sener and S. Savarese, “Active learning for convolutional neural networks: A core-set approach,” in Proc. International Conference on Learning Representations (ICLR), 2018.

[24] G. Hacohen, A. Dekel, and D. Weinshall, “Active learning on a budget: Opposite strategies suit high and low budgets,” in Proc. International Conference on Machine Learning (ICML), ser. Proceedings of Machine Learning Research, vol. 162, 2022, pp. 8175–8195.

[25] J. T. Ash, C. Zhang, A. Krishnamurthy, J. Langford, and A. Agarwal, “Deep batch active learning by diverse, uncertain gradient lower bounds,” in Proc. International Conference on Learning Representations (ICLR), 2020.