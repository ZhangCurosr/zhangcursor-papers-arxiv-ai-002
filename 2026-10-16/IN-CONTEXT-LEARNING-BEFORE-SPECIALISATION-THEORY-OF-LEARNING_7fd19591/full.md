# IN-CONTEXT LEARNING BEFORE SPECIALISATION:THEORY OF LEARNING IN A SOLVABLE TOPIC MODEL

Francesco Cagnetta∗ Simplex

Casper L. Christensen Simplex, Astera Institute

## ABSTRACT

Sequence models can generalise in fundamentally distinct ways: extrapolating structures seen in training, or using their prompt to adapt to novel tasks. The prevailing strategy determines how the model behaves beyond its training distribution, controlling potential capabilities and risks. Recent work has characterised many different strategies theoretically, and empirically described the transition between them as the quality and quantity of pretraining data vary. However, there is no learning theory for this competition, hence no fundamental understanding of why one solution might be favoured over another. Here we provide such a theory in a solvable topic model—documents drawn from a finite mixture of latent topics, mirroring the heterogeneity of modern pretraining data. The dataset comes with two distinct solutions: a specialised predictor, that stores the finite topic mixture, and an in-context one, optimal for topics never seen during training. Our theory predicts their emergence at distinct sample complexities: the in-context predictor is available early, while the specialised one emerges only after the training tokens outnumber the parameters of the data-generating process. Trained transformers reproduce both scales and the separating crossover in remarkable agreement with the theory, with no fitted parameters.

![](images/b1db1b7b498abd31cda9dd86682946013187e96293547723eace84889cf334cb.jpg)

![](images/95641e7cf3dae0f890970732280618cb301974fb647db12c3d89165fc4d7d222.jpg)  
Figure 1: In-context learning and specialisation in a random topic model. Left: A random 1topic model consists of D topics, i.e. unigram probabilities $\mathbf { p } _ { z } \in \bar { \mathbf { \Gamma } } \Delta ^ { K - 1 }$ over a vocabulary of size $K .$ Training sequences are obtained by sampling a topic z uniformly, then $T$ tokens i.i.d. from $\mathbf { p } _ { z }$ . Right: Learning curve of a one-layer transformer trained for next-token prediction on a random topic model with $\bar { ( } K , D , T ) = ( 3 2 , \bar { 5 1 } 2 , 1 2 8 )$ . Initially, both In-Distribution (ID, on the D training topics, blue) and Out-Of-Distribution (OOD, on a new topic $\mathbf { p } _ { \star } ,$ , orange) descend with N towards the entropy of the Bayes-optimal OOD predictor (orange dotted): the add-one estimator $r _ { t } ( a \mid X _ { \leq t } ) = ( \# \bar { \{ } i  \leq t \mid X _ { i } = a \} ^ { - } + 1 ) / ( t + K )$ Later, the predictor specialises to the training topics: the ID loss converges to the ID Bayes-optimal floor (blue dotted), while the OOD loss rises as in overfitting. Code: https://github.com/fracagnetta/random-topic-model.

## 1 INTRODUCTION

Large language models generalise in ways that were not anticipated in training. In particular, novel behaviours can be elicited simply via prompting, including few-shot learning of supervised tasks (Brown et al., 2020) or adopting different roles depending on a set of instructions (Shanahan et al., 2023). This In-Context Learning (ICL) mechanism is not captured by the classical learningtheoretic paradigm, where a model is evaluated on fresh samples from same the distribution it was trained on. Nevertheless, ICL is a critical source of both generality and risk (Anwar et al., 2024) in modern AI systems, calling for a theory of what trained models do beyond their training distribution.

An early theoretical attempt framed ICL as implicit Bayesian inference of the latent concept behind the prompt (Xie et al., 2022). This framework has found extensive support: trained transformers behave as Bayes-optimal estimators for many families of tasks (Garg et al., 2022), and how such estimators are represented and acquired during training is understood in simple settings (Nichani et al., 2024; Edelman et al., 2024; Lu et al., 2025; Zhang et al., 2025; Bondaschi et al., 2026). However, other settings admit distinct classes of predictors, and models transition between them as the training conditions vary (Akyurek et al., 2023). Raventos et al. (2023) identified task diversity¨ as the key control parameter: When trained at low diversity, the transformer can only memorise the training tasks, acquiring general ICL abilities only above a certain task-diversity threshold.

The role of task diversity has been explored in synthetic data, revealing that the competition between the two behaviours extends along training time (Singh et al., 2023; Reddy, 2024; Nguyen & Reddy, 2025; Park et al., 2025; Gibson et al., 2026), culminating in a hierarchical Bayesian framework that fits the model behaviour to a weighted mixture of the memorising and generalising predictors (Carroll et al., 2025; Wurgaft et al., 2025). This framework is phenomenological: still missing is a microscopic theory that predicts the favoured solution from data properties alone, and how much data it takes for one solution to give way to the other.

Our Contribution Here we provide this theory in a synthetic dataset containing the minimal required ingredients: a task diversity parameter D, a vocabulary of size K, a document length T and a corpus of N training documents. Task diversity enters as a mixture of D latent topics from which documents are generated (Fig. 1, left). In this setting, we provide:

• A solvable model of strategy competition. We characterise analytically two distinct Bayes-optimal predictors: a specialised one, relying on knowledge of the D topics in the data-generating process, and an in-context one, equivalent to the add-one estimator from statistics under our data model (Sec. 2).

• A first-principles learning-curve theory for specialisation. We derive the specialisation learning curve in a Bayesian learning setting: the excess loss over the asymptotic floor is $D K / ( 2 N T )$ at leading order—a function of the number of training tokens per parameter of the data-generating process (Sec. 3).

• A statistical mechanism for the in-context phase. The in-context predictor is linear in the token counts, and a learner that estimates this map from data acquires it at a sample complexity set by its $K ^ { 2 }$ parameters and by the information that a context carries about the next token: both independent of D. Hence, comparison with the specialisation scale gives a task-diversity threshold above which (see Fig. 1) the in-context solution emerges first, it remains stable in a window that widens with task diversity, then it gives way to the specialised solution (Sec. 4).

• Parameter-free agreement with trained transformers. Across vocabularies, task diversities, sequence lengths and corpus sizes, transformer in-distribution losses collapse as functions of NT/DK, converging to the specialised solution at the predicted rate. Nontrivial ICL performance develops at the predicted, D-independent sample complexity, then vanishes again with specialisation (Sec. 5).

• Separation from classical spectral methods. Reconstructing the topics from low-order moments (Hsu et al., 2012; Anandkumar et al., 2014) requires resolving a Marchenko– Pastur bulk of eigenvalues, which takes $N \sim D ^ { 2 } K$ samples: a worse scaling with task diversity than the Bayes-optimal $N T \sim D K$ followed by trained transformers (Sec. 5.1).

## 1.1 RELATED WORK

Mixtures of tasks as ICL testbeds. Finite task mixtures underlie most theoretical studies of ICL: linear regression (Garg et al., 2022; Akyurek et al., 2023; Raventos et al., 2023; Lu et al., 2025;¨ Zhang et al., 2025; Bordelon et al., 2026; Letey et al., 2026), classification (Reddy, 2024; Nguyen & Reddy, 2025), and mixtures of Markov (Edelman et al., 2024; Park et al., 2025; Bondaschi et al., 2026; Gibson et al., 2026) and hidden Markov (Riechers et al., 2025; Ray et al., 2026) models. The “Balls & Urns” setting of Wurgaft et al. (2025) is precisely the random topic model studied here, and their memorising and generalising predictors are our specialised and in-context ones. Our work complements their phenomenological Bayesian framework: where they assume a scaling law and a low-Kolmogorov-complexity bias, we derive the sample complexities of the two predictors from first principles. Thus, our approach does not require estimating Kolmogorov complexities and fitting parameters, but rather prescribes a simple picture where the functional simplicity of the ICL solution promotes its emergence as long as it leads to a noticeable loss gain with fewer data.

Representations, Belief-state geometry and task vectors. Due to next-token pretraining, transformers linearly represent the belief state over latent structure (Shai et al., 2024). In particular, mixtures of independent components, or non-ergodic sources, give rise to a telescopic geometry, which sparsifies along the context (Ray et al., 2026). The random topic model is the simplest non-ergodic source, and our results predict the sample dependence of the resulting belief geometry. Beyond controlled data structures, ICL has been linked to task vectors encoding the inferred task (Hendel et al., 2023), whose geometry distinguishes dual modes of task inference corresponding to our two predictors (Yan et al., 2026). Overall, these approaches describe how a trained network implements the chosen strategies; ours predicts when the data favours each.

Learning theories of ICL. Most analytical studies of learning focus on the infinite-data and infinite-diversity limit. Lu et al. (2025) is closest to us: they solve linear attention trained on Gaussian regression in a proportional asymptotic limit, obtaining sharp learning curves and a genuine transition with increasing task diversity. However, since their linear model cannot fit the specialised predictors exactly, they did not observe the crossover from in-context to specialised predictor as the number of data increases. Furthermore, our setup can be extended to mixtures of Markov chains (Gibson et al., 2026) and generic hidden Markov models (Riechers et al., 2025).

## 2 DATA MODEL AND OPTIMAL PREDICTORS

This section introduces our synthetic model of mixture data and the two reference predictors representing the specialised and in-context solutions. The solutions are distinguished by their nexttoken-prediction loss profiles. As shown in Fig. 2, trained transformers behave like the specialised predictor at low data diversity, while converging to the in-context one at high diversity. The rest of the paper explains why, and on which scale of the corpus size N each solution is selected.

Random Topic Model A dataset is specified by a vocabulary size $K ,$ a number of topics D and a document length T. Each topic is a probability distribution over the vocabulary, $\mathbf { p } _ { z } ^ { \mathbf { \Upsilon } } \in \Delta ^ { K - 1 }$ with entries $p _ { z } ( a )$ for $a = 1 , \ldots , K .$ , drawn independently and uniformly from the simplex, $\mathbf { p } _ { z } \sim$ $\operatorname { D i r } _ { K } ( { \bf 1 } )$ for $z = \dot { 1 } , \dots , D _ { \mathrm { : } }$ , and then held fixed. We collect the topics as the columns of the topic matrix $\ b { P } = [ \ b { \mathrm { p } } _ { 1 } , \dots , \ b { \mathrm { p } } _ { D } ] \in \mathbb { R } ^ { K \times D }$ . A document is generated by drawing a topic index $Z$ uniformly from $\{ 1 , \ldots , D \}$ , then $T$ tokens $X _ { 1 } , \ldots , X _ { T } \in \{ \bar { 1 , } \ldots , K \}$ independently from $\mathbf { p } _ { Z } \left( \mathrm { F i g . ~ 1 } \right.$ , left). Given the topic, the likelihood of a context $X _ { \leq t } = ( X _ { 1 } , \ldots , X _ { t } )$ depends only on its histogram, the count vector

$$
C ( X \leq t ) \in \mathbb { N } ^ { K } , \qquad C _ { a } ( X \leq t ) : = \# \{ s \leq t : X _ { s } = a \} .\tag{1}
$$

The training corpus consists of N documents, all generated with the same topic matrix $P .$

Data versus topic distribution. Two kinds of averages appear throughout. $\mathbb { E } [ \cdot ]$ denotes the expectation over documents at fixed topic matrix P, i.e. over the topic index Z and the tokens: this is the average that defines the loss of any predictor, including a trained transformer.  denotes the further average over the draw of the topics. Unless stated otherwise, the curves in the figures are averaged over independent realisations of the topic matrix $P$

Next-token prediction, In- and Out-Of-Distribution. We consider two test distributions. Indistribution (ID) documents are generated as the training ones, from one of the D topics of $P .$ Out-of-distribution (OOD) documents are generated from a fresh topic $\mathbf { p } _ { \star } \sim \operatorname { D i r } _ { K } ( \mathbf { 1 } )$ , drawn from the same prior as the training topics but independently, and hence almost surely different from all of them. A next-token predictor f maps each context $X _ { \leq t }$ to a distribution $\mathbf { f } _ { t } ( \bar { X } _ { < t } ) \in \Delta ^ { K - 1 }$ over $X _ { t + 1 }$ . We measure ID performance by the next-token loss at context length t, or t-gram entropy

$$
\mathcal { L } _ { t } [ \mathbf { f } ] : = \mathbb { E } \left[ - \log f _ { t } ( X _ { t + 1 } \mid X _ { \leq t } ) \right] , \qquad t = 0 , \ldots , T - 1 ,\tag{2}
$$

where the expectation is over fresh documents generated from the D training topics. The same expectation over a fresh topic $\mathbf { p } _ { \star }$ defines the OOD loss $\mathcal { L } _ { t } ^ { \star } [ \mathbf { f } ]$ . The usual autoregressive loss $\mathcal { L } _ { \mathrm { A R } }$ is the average of $\mathcal { L } _ { t }$ over positions.

## 2.1 THE SPECIALISED PREDICTOR

A learner that knows the topic matrix P still does not observe the topic $Z$ of a given document, and must infer it from the context. The Bayes-optimal next-token distribution is the posterior mixture of the topics,

$$
\mathbf { q } _ { t } ( X _ { \leq t } ) : = \sum _ { z = 1 } ^ { D } \gamma _ { t } ( z ) \mathbf { p } _ { z } , \qquad \gamma _ { t } ( z ) : = \mathbb { P } \left( Z = z \mid X _ { \leq t } \right) = \frac { \prod _ { a = 1 } ^ { K } p _ { z } ( a ) ^ { C _ { a } } } { \sum _ { y = 1 } ^ { D } \prod _ { a = 1 } ^ { K } p _ { y } ( a ) ^ { C _ { a } } } ,\tag{3}
$$

where the posterior over the topic $\gamma _ { t } ( z )$ depends on the context only through the counts. $\mathbf { q } _ { t }$ is the true conditional distribution of $X _ { t + 1 }$ given the context, hence it minimises $\mathcal { L } _ { t }$ over all predictors (App. B). We call it specialised, as it is built from the D topics realised in training.

The cross-entropy loss splits into an entropy plus a Kullback–Leibler divergence,

$$
\mathcal { L } _ { t } [ \mathbf { q } ] = \frac { 1 } { D } \sum _ { z = 1 } ^ { D } H ( \mathbf { p } _ { z } ) + \mathbb { E } [ D _ { \mathrm { K L } } ( \mathbf { p } _ { Z } \| \mathbf { q } _ { t } ) \rfloor ,\tag{4}
$$

both averaged over the unknown topic. The first term is the loss of an oracle that observes the topic, and no predictor can go below it: it is the entropy rate of the data-generating process. For sufficiently large D, the entropy rate converges to the average over draws of the topic matrix $\overline { { H ( \mathbf { p } ) } }$ which is log $K - 1 + \gamma _ { \mathrm { E } } + O ( 1 / K )$ for large K (App. A), with $\gamma _ { \mathrm { E } } \simeq 0 . 5 7 7 2 1$ Euler’s constant. The second term is the excess loss due to the uncertainty on the topic and, by Eq. 3, it is controlled by the posterior mass on the wrong topics $y \neq z$ . As shown in App. B, the mass on a wrong topic y decays exponentially in t, at a rate given by the divergence $D _ { \mathrm { K L } } ( \mathbf { p } _ { z } \| \mathbf { p } _ { y } )$ . For random topics this divergence is of order one $( \operatorname { A p p } . \operatorname { A } )$ , so that discriminating the true topic from the other $D - 1$ requires a context of length $t \sim \log D \left( \mathbf { A p p . B } \right)$ . This scale is visible in the left panel of Fig. 2.

## 2.2 THE IN-CONTEXT PREDICTOR

Consider now an OOD document, generated from a new topic p that does not belong to the topic matrix P. A learner that knows the prior $\operatorname { D i r } _ { K } ( { \bf 1 } )$ but not p<sub>⋆</sub> must infer from the context the topic itself, rather than its index within a known dictionary. By Dirichlet categorical conjugacy, the posterior over p<sub>⋆</sub> given $X _ { \leq t }$ is $\operatorname { D i r } _ { K } ( C _ { 1 } { + } 1 , \ldots , C _ { K } { + } 1 )$ , and the Bayes-optimal next-token distribution is its mean, the add-one smoothing of the histogram $( { \mathrm { A p p . ~ C } } )$

$$
r _ { t } ( a \mid X _ { \leq t } ) : = \mathbb { P } \left( X _ { t + 1 } = a \mid X _ { \leq t } \right) = \frac { C _ { a } ( X _ { \leq t } ) + 1 } { t + K } .\tag{5}
$$

We call it in-context as it does not depend on the topic matrix P, and thus applies unchanged to documents from seen or unseen topics.

The OOD loss of $\mathbf { r } _ { t }$ decomposes as in Eq. 4, with the unseen topic in place of $\mathbf { p } _ { Z }$

$$
\mathcal { L } _ { t } ^ { \star } [ \mathbf { r } ] = \overline { { H ( \mathbf { p } ) } } + \mathbb { E } \left[ D _ { \mathrm { K L } } \left( \mathbf { p } _ { \star } \parallel \mathbf { r } _ { t } \right) \right] = \overline { { H ( \mathbf { p } ) } } + \frac { K - 1 } { 2 t } + O \left( \frac { \log t } { t ^ { 2 } } \right) ,\tag{6}
$$

where the expansion holds at fixed $K$ for $t \gg K \left( \mathrm { A p p . C } \right)$ . Simply put, the histogram needs to include at least order $K$ tokens to depart from a trivial prediction $1 / \dot { K }$ . Note that, since $\mathbf { p } _ { \star }$ is drawn from the same prior as the topics of $P ,$ the entropy floor is the same for the two predictors, while the approach is significantly different (see Fig. 2, left). Identifying one of D known topics requires log D nats of evidence, which the context delivers at a constant rate per token, hence the exponential convergence of Sec. 2.1. Estimating a new topic is instead a (K 1)-dimensional problem, and the excess in Eq. 6 is the classical half of the number of parameters per observation behaviour.

![](images/16ded084397f7453438cd2dc23a07dff52daaa2c2a5d3c70f123c2d159370437.jpg)  
<sup>1</sup>Figure 2: Next-token loss profiles of the two Bayes-optimal predictors and of trained transformers, for $K = 6 4$ and $T = 1 2 8$ . Left: ID loss $\overline { { \mathcal { L } _ { t } [ \mathbf { q } ] } }$ of the specialised predictor for D from 32 to 8192 (colour), averaged over realisations of the topic matrix, and OOD loss $\mathcal { L } _ { t } ^ { \star } [ \mathbf { r } ]$ of the in-context predictor (black dashed), which does not depend on D. Dotted line: log K. Markers: context length at which each specialised profile has covered half of its descent, growing as log D (grey guide; Sec. 2.1). Middle and right: the same two references at $D = 3 2$ and $D = 8 1 9 2$ , with the ID (blue) and OOD (orange) loss profiles of a one-layer transformer trained on N documents at that D, mean $\pm \ : s . e .$ . over four runs with independent topic matrix, corpus and initialisation, at the checkpoint of minimal validation loss. $\mathbf { A } \mathbf { t } \ D = 3 2$ the transformer matches the specialised predictor in distribution and stays above log K out of distribution; at $D = 8 1 9 2$ both profiles follow the in-context predictor.

## 3 LEARNING-CURVE THEORY OF SPECIALISATION

The training corpus contains only the D topics in the matrix P. Hence, a sufficiently expressive, well-optimised machine learning model must converge to the specialised solution Eq. 3 in the infinite-data and training time limit. But how many training documents does it take to specialise?

Bayesian learning We answer this question with a Bayesian learner that uses the training corpus optimally: It knows the model class, including D and $\dot { K }$ , but not the realised topic matrix $P ,$ and observes a collection $\mathcal { C } _ { N }$ of N length-T training documents, together with the test context. The training corpus informs a posterior $\bar { \pi } ( d P | \mathcal { C } _ { N } )$ over the topic matrix $P ,$ proportional, by Bayes’ rule, to prior $\mu ( d P )$ times the joint probability of the corpus under $P , \mathbb { P } \left\{ \mathcal { C } _ { N } \right\}$ . The Bayes-optimal learner is the posterior mean of the $\bar { P ^ { . } }$ -dependent specialised predictor from Eq. 3,

$$
\hat { \mathbf { q } } _ { t } ( X _ { \leq t } ) \propto \int \mathbf { q } _ { t } ( X _ { \leq t } ; P ) \pi ( d P | \mathcal { C } _ { N } , X _ { \leq t } ) ,\tag{7}
$$

where the posterior also includes the test context $X _ { < t }$ (see App. D for details). Note that, after averaging over the draw of the topics, no predictor built from the same data has a lower loss than $\hat { \mathbf { q } } .$ Its loss is the statistical limit of specialisation from N training documents, hence a natural reference against which a transformer trained on the same corpus can be measured.

Learning curve and mutual information. The learning curve of specialisation is the excess loss of the Bayes-optimal learner over the specialised predictor, averaged over topic matrices,

$$
\varepsilon _ { t } ( N ) : = \overline { { \mathcal { L } _ { t } [ \hat { \mathbf { q } } ] - \mathcal { L } _ { t } [ \mathbf { q } ] } } ,\tag{8}
$$

Averaging over the topics gives it an information-theoretic meaning: $\varepsilon _ { t } ( N )$ is the conditional mutual information between $\scriptstyle \mathbf { \hat { \boldsymbol { P } } }$ and the next token, given corpus and context (App. D),

$$
\varepsilon _ { t } ( N ) = I ( P ; X _ { t + 1 } \mid X _ { \leq t } , \mathcal { C } _ { N } ) \Rightarrow \sum _ { t = 0 } ^ { T - 1 } \varepsilon _ { t } ( N ) = I ( P ; \mathcal { C } _ { N + 1 } ) - I ( P ; \mathcal { C } _ { N } ) ,\tag{9}
$$

where $\mathcal { C } _ { N + 1 }$ adds the test document to the corpus. The excess loss of the learner on the test document equals the information that this document adds about the topic matrix. Thus, the learning curve quantifies how fast the corpus reveals P.

The $D K / N T$ law. The topic matrix has $D ( K { - } 1 )$ free parameters, and a corpus of N documents reveals them at the rate of parametric statistics: under identifiability and regularity, $I ( P ; { \mathcal { C } } _ { N } ) =$ $( D ( K - 1 ) / 2 )$ log $N + O ( { 1 } )$ (Clarke & Barron, 1990; 1994). The increment gives the leading term of the learning curve at fixed D and $K \left( \mathrm { A p p . D } \right)$ ,

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \varepsilon _ { t } ( N ) = \frac { D ( K - 1 ) } { 2 N T } + o \big ( N ^ { - 1 } \big ) .\tag{10}
$$

Specialisation thus requires $N T \gg D K$ , and the learning curve is a function of the number of training tokens per parameter, $x : = N T / ( D K )$ , decaying as $x ^ { - 1 }$ . As shown in Sec. 5, trained transformers specialise on the same sample-complexity scale, and with the same decay. Two remarks. First, identifiability, i.e. that the law of a document determines $P$ up to a relabelling of the topics, requires a sufficiently long document and a generic topic matrix. For $D \leq K , T \geq 3$ suffices: the columns of P are linearly independent almost surely under the topics draw, and the third-order moment of the tokens has a unique decomposition with the topics as factors (Kruskal, 1977; Anandkumar et al., 2014). For larger D the same holds for the lifted topics $\mathbf { p } _ { z } ^ { \otimes { m } }$ , which are generic in the same sense, i.e. linearly independent, as long as $D \le \binom { K + m - 1 } { m }$ , the dimension of the space of symmetric tensors, at the price of longer documents, $T \geq 2 m { \ddot { + } } 1 \ { \mathrm { ( A p p . ~ D ) } }$ . For instance, pairs of tokens cover $K < D \le \dot { K } ( K { + } 1 ) / 2$ with $T \geq 5$ . Secondly, the learning curve is an average over topic matrices, whereas an experiment runs on individual realisations. Comparing the two assumes that the excess loss of a given realisation is close to its average, i.e. typicality of the random topic matrix. An empirical check of this typicality is the spread across independent realisations in Sec. 5.

The same law at fixed, large position. Equation 10 concerns the loss averaged over positions; at a fixed position t the same law follows from a simple picture. Beyond the resolution scale log D of Sec. 2.1, the learner can identify the topic of the test document, and likewise the topics of the training documents if $T \gg$ log D. It is then left with estimating the token distribution of the test topic from the training documents of that topic, about $N / D$ of them, i.e. from $m \approx N T / D$ tokens: this is the problem solved by the in-context predictor of Sec. 2.2, with m in place of t, whose excess $( K - 1 ) / ( 2 m )$ becomes $D ( K - 1 ) / ( 2 N T )$ . The t-gram loss at large t therefore obeys the same law as the average, and App. D makes this rigorous at fixed $P ,$ , under conditions on $t , T$ and $N ; \mathrm { F i g }$ . 10 compares it with the T-gram learning curves of trained transformers.

## 4 WHEN DOES ICL EMERGE?

Simplicity of the in-context predictor. If any learner eventually converges to the specialised predictor, when does ICL emerge? Both predictors of Sec. 2 are functions of the counts, but of very different complexity. The in-context predictor is linear up to normalisation,

$$
\mathbf { r } _ { t } ( X _ { \leq t } ) \propto \mathbf { C } ( X _ { \leq t } ) + \mathbf { 1 } ,\tag{11}
$$

whereas the specialised predictor is exponential in the counts $( \mathrm { E q . } 3 )$ . Then, simplicity bias suggests that the in-context predictor is easier to learn. $\mathbf { r } _ { t }$ represents a point in the family of ${ \dot { K } } ^ { 2 }$ linear maps from counts to next-token probability, whereas random topic models $P$ lie in $D ( K - 1 )$ dimensions. The estimation error follows from the same argument as Sec. 3, in the form appropriate to a fitted parametric family rather than a Bayesian posterior (App. E). Simply put, a learner estimating the ${ \bar { K } } ^ { 2 }$ parameters of the linear map from a corpus of $N T$ tokens pays an excess $K ^ { 2 } / ( 2 N T )$ over the optimal in-context solution, again proportional to the number of parameters per training token.

Estimation penalty versus loss gain. This excess must be compared with the gain of the add-one rule over the uniform predictor, equal to the mean mutual information between the next token and its context (App. E),

$$
G ( K , T ) : = \log K - \frac { 1 } { T } \sum _ { t < T } \mathcal { L } _ { t } ^ { \star } [ \mathbf { r } ] = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } I \left( X _ { t + 1 } ; X _ { \leq t } \right) .\tag{12}
$$

The new solution can only appear once the error due to finite data $K ^ { 2 } / ( 2 N T )$ is smaller than the gain, i.e. for

$$
N \gtrsim N _ { \mathrm { i c l } } : = \frac { K ^ { 2 } } { 2 T G ( K , T ) } ,\tag{13}
$$

independently of the task diversity D.

Task-diversity threshold. By Eq. 10, the specialised solution emerges at $N _ { \mathrm { s p e c } } { = } D K / T$ . Hence, if

$$
D \gg \frac { K } { 2 G ( K , T ) } ,\tag{14}
$$

then the in-context solution emerges before the specialised one, and persists in the window $( N _ { \mathrm { i c l } } , N _ { \mathrm { s p e c } } )$ . The resulting picture is that of Fig. 1, right. In the $( N _ { \mathrm { i c l } } ,  { N _ { \mathrm { s p e c } } } )$ window the learner applies the same rule to seen and unseen topics alike, thus OOD and ID loss decay together towards the OOD optimum $\begin{array} { r } { \mathcal { L } _ { t } ^ { \star } [ \mathbf { r } ] . \ \mathrm { A t } \ N _ { \mathrm { s p e c } } , } \end{array}$ the learner resolves the dictionary: the ID loss descends to the entropy rate, and the OOD loss rises towards $\mathcal { L } _ { t } ^ { \star } [ \mathbf { q } ] { \longrightarrow } 0 0 \mathrm { D }$ loss of the specialised predictor.

## 5 TRANSFORMER EXPERIMENTS

This section collects empirical learning curves of one-block transformers trained with Adam on data generated from random topic models of varying K and D. Experiments with varying T are reported in App. H. Since we are testing the scaling of performance with the number of training tokens, we conducted extensive hyperparameter search, looking for the best configuration of model size, learning rate, batch size and training time at fixed depth and optimiser. Further details of architectures and training algorithms are reported in App. G.

Fig. 3 shows that transformers learn random topic models at the same rate as the Bayesian learner of Sec. 3.

![](images/b56ab2141d34994f5616f479f465c0ab151e3e25b62f7174199c52ff8af37f19.jpg)

![](images/931bc94b3216f53425c4f3617ac3feff57ef8664a6771fce8269da74e46511e4.jpg)

![](images/bb616c03d82369572212d013a2b58f3982d59ac4d843fb029c467f24ba111bd7.jpg)  
<sup>1</sup>Figure 3: Transformers match the Bayes-optimal specialisation learning curve. Left: Scaled ID autoregressive loss $( \mathcal { L } _ { \mathrm { A R } } ( N ) - \mathcal { L } _ { \mathrm { A R } } ( \bar { \infty } ) ) / ( \log K - \bar { \mathcal { L } } _ { \mathrm { A R } } ( \infty ) )$ versus number of training sequences for one-block transformers trained on random topic model data with varying $( K , D ) , T = 1 \bar { 2 } 8$ . The scaled loss is 1 at the uniform predictor $1 / K$ , 0 at the specialised predictor. Middle: Rescaling the x-axis as the number of tokens per parameter of the data-generating process collapses all the curves onto a single functional form. Right: The decay towards 0 (black dashed) shows remarkable agreement with the $x ^ { - 1 }$ rate of the Bayes-optimal learning curve.

Even above the task-diversity threshold Eq. 14, specialisation kicks in as soon as the number of tokens NT surpasses the number of parameters DK. This is clearly shown by Fig. 4: the large-N decay of the ID loss scales with $( N \bar { T } / ( D K ) )$ ), while the early-N decay develops a D-independent behaviour. Correspondingly, the OOD losses at early stopping (dashed lines) show an initial descent, followed by a steep ascent around $N \sim ( D K ) / T$ This result shows that the transience of ICL is inevitable, unless the learning model is regularised.

Fig. 5 shows OOD learning curves in the high-diversity regime: the initial D-independent descent scales like $K ^ { 2 } / G ( K , T )$ as predicted in Eq. 13. Since specialisation occurs at $\bar { N } T \sim D K$ , the in-context solution remains stable for a window of training set sizes linear in D. Fig. 11 of App. H shows the same scaling behaviour for the initial decay of ID learning curves.

![](images/3ba713d2217d5c38c3cce880be8236b92fb947fca5ab8a8c0e1b61b765c64c1c.jpg)  
Figure 4: In-context learning gives way to specialisation at the parametric scale1 $N T \sim D K$ Left: Scaled ID (solid) and OOD (dashed) autoregressive losses, both subtracting the ID loss floor for comparison. Right: Rescaling the x-axis as the number of tokens per parameter of the datagenerating process collapses both the descent of ID losses and the ascent of OOD losses.

![](images/5485ac9ab019afa87af1c356a133bbc14923a13441b82e2a0e61f4812bb2d1e1.jpg)  
Figure 5: Empirical OOD learning curves decay at the predicted sample scale. Left:1 Scaled OOD autoregressive loss $( \mathcal { L } _ { \mathrm { A R } } ^ { \star } ( N ) ^ { - } - \mathcal { L } _ { \mathrm { A R } } ^ { \star } ( \infty ) ) / ( \log K - \mathcal { L } _ { \mathrm { A R } } ^ { \star } ( \infty ) )$ versus number of training tokens NT for one-block transformers trained on random topic model data with varying $( K , D { \bar { ) } }$ at $T = 1 2 8$ (solid), and with $K = 6 4$ $D = 1 0 2 4$ at $T = 6 4$ (dotted) and $T = 2 5 6$ (dashed). These losses are not recorded at early stopping (training step with the lowest ID validation loss), but at the training step with the lowest OOD validation loss. Hence, the OOD losses shown in this figure do not rise at $\dot { N } \sim D K / T$ as in Fig. 4. Right: The early-N departure from the trivial loss log K does not depend on D and collapse after dividing the number of training sequences by $K ^ { 2 } / G ( { \breve { K } } , T )$ , in perfect agreement with the theory of Eq. 13.

## 5.1 COMPARISON WITH SPECTRAL BASELINES

Latent-variable models like the random topic model are classically learnt by spectral methods based on the moments of the tokens (Hsu et al., 2012; Balle et al., 2014; Anandkumar et al., 2014). The first step is recovery of the D-dimensional subspace spanned by the topics from a low-order moment: the pair moment $\begin{array} { r } { M _ { 2 } = \frac { 1 } { D } \sum _ { z } \mathbf p _ { z } \mathbf p _ { z } ^ { \top } } \end{array}$ , of rank D, when $D \leq K$ , and the moment of 2m tokens when $D \leq { \binom { K + m - 1 } { m } }$ , the number of independent products of m topic entries. The topics themselves then follow from the moment with one more token, which is the identifiability step of Sec. 3 (App. F). The subspace is resolved once the empirical moment, estimated from N independent 2m-tuples, is close to the population moment. The corresponding sample size is a lower bound for the method to work.

The error on the empirical moment is of order $( K ^ { m } N ) ^ { - 1 / 2 }$ by the matrix Bernstein inequality (App. F, verified numerically in Fig. 8). The scales of the population moments, instead, follow from random matrix theory, in the limit $D , K \to \infty$ . With $D / K = \alpha < 1$ and $m = 1$ , the bulk of $M _ { 2 } { \bf \bar { s } }$ spectrum converges to a Marchenko–Pastur distribution with eigenvalues of order $1 / ( D K ) \left( \mathrm { F i g . } 6 \right)$ Comparison with the noise gives $N \gtrsim D ^ { 2 } K$ . With $D / ( K ( \bar { K } + 1 ) / 2 ) = \beta < 1$ and $m = 2 ,$ the relevant bulk of $M _ { 4 }$ is of order $1 / ( D \bar { K } ^ { 2 } )$ (Fig. 7). Comparison with the noise gives $N \gtrsim D ^ { 2 } K ^ { 2 }$ This analysis is carried out in a different limit from the rest of the paper, and it counts independent tuples rather than documents of $T$ tokens, so we only compare the dependence on task diversity: quadratic in D for the spectral baseline, against the linear $N \bar { T } \sim D K$ of the Bayes-optimal learner, which the trained transformers follow. Furthermore, in the spectral baseline, the order of K grows with the order of the required moment.

## 6 CONCLUSIONS

We have studied a solvable model of a mixture of D topics, where two strategies compete: indistribution specialisation to the finite mixture, or out-of-distribution generalisation to topics never seen in training. Specialisation requires more training tokens than there are parameters in the datagenerating process—a statistical limit that no learner can beat. The in-context solution is available at a D-independent scale, hence it comes first whenever the number of topics is large enough. Trained transformers follow both scales. It follows that the transience of ICL (Singh et al., 2023) is inevitable as the corpus grows, unless a constraint on parameters or training budget keeps the learner in the in-context window.

The specialised solution was called memorising in previous studies of the same competition (Raven tos et al., 2023; Nguyen & Reddy, 2025; Carroll et al., 2025; Wurgaft et al., 2025). Indeed, its placement after the in-context phase mirrors the training dynamics of diffusion models, where generalisation is acquired before the memorisation of the training set (Favero et al., 2025). We find the label misleading: the specialised predictor stores the D topics of the population, not the N documents of the corpus—memorisation happens in feature space. This is a form of in-distribution generalisation, distinct from the adaptation to new topics of the in-context predictor, and we name it after the specialisation of teacher-student networks, where the hidden units of the student align with those of the teacher (Goldt et al., 2019; Barbier et al., 2025).

Several directions are open. The first is the training dynamics of transformers on the random topic model, which would connect our sample complexities with the phenomenological descrip tion of Wurgaft et al. (2025). More importantly, the dynamics determine which representations are learned in each phase and the corresponding geometry of the residual stream (Shai et al., 2024; Ray et al., 2026). Clarifying the distinction between representations in the two phases could help understand the emergence and role of task vectors (Hendel et al., 2023), including scenarios where multiple solutions coexist (Yan et al., 2026). The second is the competition between task diversity and other structures of the data. The tools developed here can address this problem in general non-ergodic hidden Markov models (Riechers et al., 2025), of which the random topic model is the simplest instance. Such mixtures are natural models of large heterogeneous corpora. Solvable mod els like the one proposed here could help understand how heterogeneity contributes to data statistic and scaling laws (Cagnetta et al., 2026), and how improvements on the individual components of a mixture interact.

## AI USE STATEMENT

In this work, we used generative AI tools for: assistance in the writing of proofs; implementation of numerical methods. We have not used generative AI tools for: Generating synthetic data sets, helping develop theoretical models or conceptual frameworks, formulating mathematical claims, providing critical ingredients for proving mathematical claims, proposing or refining hypotheses, designing feedback on research methodology or experiments, interpreting results, and assisting with translation, cleaning and reformatting dataset, supporting qualitative and thematic data analysis are not applicable to this work. Additionally, we used generative AI tools for: modifying scientific figures, drafting parts of a research paper, identifying relevant literature. We have reviewed all AI-assisted work. We checked LLM-generated proofs; LLM-generated numerical implementations were verified and tested for correctness by 2 authors; We checked LLM-generated descriptions of additional related works. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## REPRODUCIBILITY STATEMENT

The theoretical results are stated with their assumptions and proved in the appendices. Appendices A–C prove the statements of Sec. 2: the averages over the topic prior, the optimality and loss decomposition of the two predictors, and the exact finite-t loss of the in-context predictor, Eq. 28, from which the in-context reference, the gain $G ( K , T )$ and the sample scale $N _ { \mathrm { i c l } }$ used to rescale the axes of the figures are computed. App. D states the identifiability and regularity assumptions behind the $D K / N T$ law, Eq. 10, gives sufficient conditions on (D, K, T) under which they hold almost surely, proves the law and gives its fixed-position version. App. E lists the conditions under which the estimation excess of a fitted family equals its parameter count, behind Eq. 13, and App. F states the joint limit in which the spectral scales hold. Arguments that we do not prove are identified as such where they appear: the concentration of pairwise divergences behind the log D constant, the signal-to-noise criterion and the block approximation of the spectral baseline, and the applicability of the estimation argument to a trained transformer.

The experiments are fully specified in App. G: the data-generating process, with the topic prior, the sizes of the validation and OOD sets and the number of fresh OOD topics; the architecture, initialisation and parameter count; the optimiser, learning-rate schedule and batching; the evalu ation protocol, the definition of early stopping and the two readings of the OOD loss; and the hyperparameter grid with the selection rule. Every point of a learning curve is the mean over four runs with independent topic matrix, corpus and initialisation, with standard errors as error bars, and every run stores its full configuration, including the seeds of the draws, so that any point can be regenerated exactly. The theoretical references in the figures are evaluated from closed-form expressions or by Monte Carlo estimates over topic realisations, as stated in the captions. The code for data generation, training, evaluation and the theoretical curves is available at https://github.com/fracagnetta/random-topic-model.

## REFERENCES

Ekin Akyurek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning¨ algorithm is in-context learning? investigations with linear models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum? id=0g0X4H8yN4I.

Elizabeth S. Allman, Catherine Matias, and John A. Rhodes. Identifiability of parameters in latent structure models with many observed variables. The Annals of Statistics, 37(6A):3099–3132, 2009. URL http://www.jstor.org/stable/25662188.

Animashree Anandkumar, Rong Ge, Daniel Hsu, Sham M. Kakade, and Matus Telgarsky. Tensor decompositions for learning latent variable models. J. Mach. Learn. Res., 15(1):2773–2832, 2014. doi: 10.5555/2627435.2697055. URL https://dl.acm.org/doi/10.5555/2627435. 2697055.

Usman Anwar, Abulhair Saparov, Javier Rando, Daniel Paleka, Miles Turpin, Peter Hase, Ekdeep Singh Lubana, Erik Jenner, Stephen Casper, Oliver Sourbut, Benjamin L. Edelman, Zhaowei Zhang, Mario Gunther, Anton Korinek, Jose Hernandez-Orallo, Lewis Hammond, Eric J¨ Bigelow, Alexander Pan, Lauro Langosco, Tomasz Korbak, Heidi Chenyu Zhang, Ruiqi Zhong, Sean O hEigeartaigh, Gabriel Recchia, Giulio Corsi, Alan Chan, Markus Anderljung, Lilian Edwards, Aleksandar Petrov, Christian Schroeder de Witt, Sumeet Ramesh Motwani, Yoshua Ben gio, Danqi Chen, Philip Torr, Samuel Albanie, Tegan Maharaj, Jakob Nicolaus Foerster, Florian Tramer, He He, Atoosa Kasirzadeh, Yejin Choi, and David Krueger. Foundational challenges\` in assuring alignment and safety of large language models. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id= oVTkOs8Pka. Survey Certification, Expert Certification.

Borja Balle, William L. Hamilton, and Joelle Pineau. Methods of moments for learning stochastic languages: unified presentation and empirical comparison. In Proceedings of the 31st Interna tional Conference on Machine Learning, volume 32, pp. 1386–1394. PMLR, 2014.

Jean Barbier, Francesco Camilli, Minh-Toan Nguyen, Mauro Pastore, and Rudy Skerk. Optimal generalisation and learning transition in extensive-width shallow neural networks near interpolation. arXiv preprint arXiv:2501.18530, 2025.

Marco Bondaschi, Nived Rajaraman, Xiuying Wei, Razvan Pascanu, Caglar Gulcehre, Michael Gastpar, and Ashok Vardhan Makkuva. From markov to laplace: How mamba in-context learns markov chains. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=kmK3WSCOCT.

Blake Bordelon, Mary Letey, and Cengiz Pehlevan. Theory of scaling laws for in-context regression: Depth, width, context and time. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=qA42mWsnbl.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D. Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam Mc-Candlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are fewshot learners. In Advances in Neural Information Processing Systems, volume 33, pp. 1877–1901, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html.

Francesco Cagnetta, Allan Raventos, Surya Ganguli, and Matthieu Wyart. Deriving neural scaling laws from the statistics of natural language. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=VXlniX0C5y.

Liam Carroll, Jesse Hoogland, Matthew Farrugia-Roberts, and Daniel Murfet. Dynamics of transient structure in in-context linear regression transformers. arXiv preprint arXiv:2501.17745, 2025.

Djalil Chafa¨ı and Konstantin Tikhomirov. On the convergence of the extremal eigenvalues of empirical covariance matrices with dependence. Probability Theory and Related Fields, 170:847–889, 2018. doi: 10.1007/s00440-017-0778-9. URL https://arxiv.org/abs/1509.02231.

Bertrand S. Clarke and Andrew R. Barron. Information-theoretic asymptotics of Bayes methods. IEEE Transactions on Information Theory, 36(3):453–471, 1990. doi: 10.1109/ 18.54897. URL https://bertrandsclarke.weebly.com/uploads/2/7/3/6/ 2736392/clarkebarron1990.pdf.

Bertrand S. Clarke and Andrew R. Barron. Jeffreys’ prior is asymptotically least favorable under entropy risk. Journal of Statistical Planning and Inference, 41(1):37–60, 1994. doi: 10.1016/0378-3758(94)90153-8. URL https://doi.org/10.1016/0378-3758(94) 90153-8.

Ezra Edelman, Nikolaos Tsilivis, Benjamin L. Edelman, eran malach, and Surbhi Goel. The evolution of statistical induction heads: In-context learning markov chains. In The Thirtyeighth Annual Conference on Neural Information Processing Systems, 2024. URL https: //openreview.net/forum?id=qaRT6QTIqJ.

Alessandro Favero, Antonio Sclocchi, and Matthieu Wyart. Bigger isn’t always memorizing: Early stopping overparameterized diffusion models. arXiv preprint arXiv:2505.16959, 2025.

Shivam Garg, Dimitris Tsipras, Percy Liang, and Gregory Valiant. What can transformers learn incontext? a case study of simple function classes. In Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=flNZJ2eOet.

Cole Gibson, Wenping Cui, and Gautam Reddy. Distinct mechanisms underlying in-context learning in transformers. arXiv preprint arXiv:2604.12151, 2026.

Sebastian Goldt, Madhu Advani, Andrew M. Saxe, Florent Krzakala, and Lenka Zdeborova. Dy-´ namics of stochastic gradient descent for two-layer neural networks in the teacher-student setup. In Advances in Neural Information Processing Systems, volume 32, 2019.

Roee Hendel, Mor Geva, and Amir Globerson. In-context learning creates task vectors. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023. URL https: //openreview.net/forum?id=QYvFUlF19n.

Daniel Hsu, Sham M. Kakade, and Tong Zhang. A spectral algorithm for learning hidden Markov models. Journal ofComputer and System Sciences, 78(5):1460–1480, 2012.

Joseph B. Kruskal. Three-way arrays: rank and uniqueness of trilinear decompositions, with application to arithmetic complexity and statistics. Linear algebra and its applications, 18(2):95– 138, 1977. doi: 10.1016/0024-3795(77)90069-6. URL https://linkinghub.elsevier. com/retrieve/pii/0024379577900696.

Mary Letey, Jacob A Zavatone-Veth, Yue M. Lu, and Cengiz Pehlevan. Pretrain–test task alignment governs generalization in in-context learning. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id= KZLeg0MQ2r.

Yue M. Lu, Mary Letey, Jacob A. Zavatone-Veth, Anindita Maiti, and Cengiz Pehlevan. Asymptotic theory of in-context learning by linear attention. Proceedings of the National Academy of Sciences, 122(28):e2502599122, 2025. doi: 10.1073/pnas.2502599122. URL https: //www.pnas.org/doi/abs/10.1073/pnas.2502599122.

Alex Nguyen and Gautam Reddy. Differential learning kinetics govern the transition from memorization to generalization during in-context learning. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id= INyi7qUdjZ.

Eshaan Nichani, Alex Damian, and Jason D. Lee. How transformers learn causal structure with gradient descent. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id=jNM4imlHZv.

Alain Pajor and Leonid Pastur. On the limiting empirical measure of eigenvalues of the sum of rank one matrices with log-concave distribution. Studia Mathematica, 195(1):11–29, 2009. doi: 10.4064/sm195-1-2. URL https://arxiv.org/abs/0710.1346.

Core Francisco Park, Ekdeep Singh Lubana, and Hidenori Tanaka. Competition dynamics shape algorithmic phases of in-context learning. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=XgH1wfHSX8.

Allan Raventos, Mansheej Paul, Feng Chen, and Surya Ganguli. Pretraining task diversity and the emergence of non-bayesian in-context learning for regression. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum? id=BtAz4a5xDg.

Kyle J. Ray, Paul M. Riechers, and Adam S. Shai. The geometry of nonergodic composition. Belief Updates (Simplex Blog), 2026. URL https://belief-updates.pub/ nonergodic-geometry/.

Gautam Reddy. The mechanistic basis of data dependence and abrupt learning in an in-context classification task. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=aN4Jf6Cx69.

Paul M Riechers, Henry R Bigelow, Eric A Alt, and Adam Shai. Next-token pretraining implies in-context learning. arXiv preprint arXiv:2505.18373, 2025.

Adam Shai, Paul M. Riechers, Lucas Teixeira, Alexander Gietelink Oldenziel, and Sarah Marzen. Transformers represent belief state geometry in their residual stream. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview. net/forum?id=YIB7REL8UC.

Murray Shanahan, Kyle McDonell, and Laria Reynolds. Role play with large language models. Nature, 623(7987):493–498, 2023.

Aaditya K Singh, Stephanie C.Y. Chan, Ted Moskovitz, Erin Grant, Andrew M Saxe, and Felix Hill. The transient nature of emergent in-context learning in transformers. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview. net/forum?id=Of0GBzow8P.

Joel A. Tropp. User-friendly tail bounds for sums of random matrices. Foundations ofcomputational mathematics, 12(4):389–434, 2012. doi: 10.1007/s10208-011-9099-z. URL https://link. springer.com/content/pdf/10.1007/s10208-011-9099-z.pdf.

Daniel Wurgaft, Ekdeep Singh Lubana, Core Francisco Park, Hidenori Tanaka, Gautam Reddy, and Noah Goodman. In-context learning strategies emerge rationally. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview. net/forum?id=bBUUOQI0N6.

Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=RdJVFCHjUMI.

Hao Yan, Haolin Yang, and Yiqiao Zhong. Task vector geometry underlies dual modes of task inference in transformers. arXiv preprint arXiv:2605.03780, 2026.

Yedi Zhang, Aaditya K Singh, Peter E. Latham, and Andrew M Saxe. Training dynamics of in context learning in linear attention. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=aFNq67ilos.

## A STATISTICS OVER RANDOM TOPIC MODELS

This appendix collects the averages over the topic prior used in the main text. Let $\mathbf { p } , \mathbf { p } ^ { \prime } \sim \operatorname { D i r } _ { K } ( \mathbf { 1 } )$ be independent, i.e. uniform on the simplex $\Delta ^ { \dot { K } - 1 }$ , and let $\psi = \Gamma ^ { \prime } / \Gamma$ denote the digamma function. Each entry is marginally $p _ { a } \sim \mathrm { B e t a } ( 1 , K { - } 1 )$ , with moments

$$
\overline { { p _ { a } ^ { r } } } = \frac { \Gamma ( K ) \Gamma ( r + 1 ) } { \Gamma ( K { + } r ) } , \qquad \overline { { \log p _ { a } } } = \psi ( 1 ) - \psi ( K ) , \qquad \overline { { p _ { a } \log p _ { a } } } = \frac { 1 } { K } \left[ \psi ( 2 ) - \psi ( K { + } 1 ) \right] ,\tag{15}
$$

where the last two follow by differentiating the first with respect to r at $r = 0$ and $r = 1$

Entropy of a topic. By symmetry of the entries,

$$
\overline { { { H ( { \bf p ) } } } } = - K \overline { { { p _ { a } \log p _ { a } } } } = \psi ( K + 1 ) - \psi ( 2 ) = \log K - 1 + \gamma _ { \mathrm { E } } + \frac { 1 } { 2 K } + O \big ( K ^ { - 2 } \big ) ,\tag{16}
$$

using $\psi ( 2 ) = 1 - \gamma _ { \mathrm { E } }$ and $\psi ( K { + } 1 ) = \log K + 1 / ( 2 K ) + O ( K ^ { - 2 } )$ . A random topic thus has entropy $1 - \mathit { \gamma } _ { \mathrm { { E } } } - 1 / ( 2 K ) \approx 0 . 4 2$ nats below the uniform value log K, at leading orders: this gap is the height of the descent of every profile in $\mathrm { F i g . } 2$

Divergence between two topics. Expanding the divergence and using the independence of $\mathbf { p }$ and $\mathbf { p ^ { \prime } }$

$$
\overline { { { D _ { \mathrm { K L } } ( { \bf p } | | { \bf p } ^ { \prime } ) } } } = K \overline { { { p _ { a } \log p _ { a } } } } - K \overline { { { p _ { a } } } } \overline { { { \log p _ { a } ^ { \prime } } } } = [ \psi ( 2 ) - \psi ( K + 1 ) ] - [ \psi ( 1 ) - \psi ( K ) ] = 1 - \frac { 1 } { K } ,\tag{17}
$$

by the recurrence $\psi ( x + 1 ) = \psi ( x ) + 1 / x$ . Two random topics are thus separated by an order-one divergence for every vocabulary size, which sets the rate of topic resolution in App. B.

Bhattacharyya coefficient. The non-asymptotic resolution bound of App. B involves $\begin{array} { r } { \rho ( \mathbf { p } , \mathbf { p } ^ { \prime } ) : = \sum _ { a } \sqrt { p _ { a } p _ { a } ^ { \prime } } } \end{array}$ , whose average follows from Eq. 15 at $r = 1 / 2$

$$
\overline { { \rho ( { \bf p } , { \bf p } ^ { \prime } ) } } = K \left( \overline { { \sqrt { p _ { a } } } } \right) ^ { 2 } = \frac { \pi K } { 4 } \left[ \frac { \Gamma ( K ) } { \Gamma ( K + \frac { 1 } { 2 } ) } \right] ^ { 2 } = \frac { \pi } { 4 } \left( 1 + \frac { 1 } { 4 K } + { \cal O } \big ( K ^ { - 2 } \big ) \right) .\tag{18}
$$

## B THE SPECIALISED PREDICTOR

This appendix proves the statements of Sec. 2.1: the optimality of $\mathbf { q } _ { t } ,$ , the loss decomposition Eq. 4, and the resolution of the topic on the context scale log D. Throughout, the topic matrix $P$ is fixed, with distinct columns, and all expectations are over documents at fixed $P .$

Optimality. Conditional on the context, $X _ { t + 1 }$ is distributed as $\mathbf { q } _ { t } ( X _ { \leq t } ) \colon$ : by construction $q _ { t } ( a \mid$ $\begin{array} { r } { \dot { X _ { < t } } ) = \dot { \sum } _ { z } \mathbb { P } ( Z = z \mid X _ { < t } ) \mathbb { P } ( X _ { t + 1 } = a \mid \dot { Z } = z ) } \end{array}$ , using that $X _ { t + 1 }$ is independent of the context given Z. For any predictor f, therefore,

$$
\begin{array} { r } { \mathbb { E } \left[ - \log f _ { t } ( X _ { t + 1 } \mid X _ { \le t } ) \mid X _ { \le t } \right] = H ( \mathbf { q } _ { t } ) + D _ { \mathrm { K L } } ( \mathbf { q } _ { t } \| \mathbf { f } _ { t } ) \ge H ( \mathbf { q } _ { t } ) , } \end{array}\tag{19}
$$

with equality if and only if $\mathbf { f } _ { t } = \mathbf { q } _ { t }$ . Averaging over contexts, $\mathcal { L } _ { t } [ \mathbf { f } ] \geq \mathcal { L } _ { t } [ \mathbf { q } ]$

Loss decomposition. Conditioning instead on the topic and the context, $X _ { t + 1 } \sim \mathbf { p } _ { Z }$ , and the same identity gives $\mathbb { E } [ - \log q _ { t } ( X _ { t + 1 } \mid X _ { < t } ) \mid Z , X _ { < t } ] \stackrel { . } { = } H ( \mathbf { p } _ { Z } ) + D _ { \mathrm { K L } } ( \mathbf { p } _ { Z } \| \dot { \mathbf { q } } _ { t } ) ;$ averaging over the topic, uniform on $\{ 1 , \ldots , D \}$ , and over the context yields Eq. 4. The floor is the loss of the oracle $\mathbf { f } _ { t } = \mathbf { p } _ { Z }$ , which is not a function of the context and hence not a predictor in the sense of Eq. 2; by the optimality of $\mathbf { q } _ { t }$ , every predictor pays at least the excess $\mathbb { E } [ D _ { \mathrm { K L } } ( \mathbf { p } _ { Z } \Vert \mathbf { q } _ { t } ) ]$ above it.

Excess loss and wrong-topic mass. Write the posterior in terms of the likelihood ratios of the wrong topics against the true one,

$$
R _ { y } ^ { ( t ) } : = \frac { \gamma _ { t } ( y ) } { \gamma _ { t } ( Z ) } = \prod _ { s = 1 } ^ { t } \frac { p _ { y } ( X _ { s } ) } { p _ { Z } ( X _ { s } ) } , \qquad \gamma _ { t } ( Z ) = \Big ( 1 + \sum _ { y \ne Z } R _ { y } ^ { ( t ) } \Big ) ^ { - 1 } .\tag{20}
$$

Isolating the true topic in Eq. 3, $\begin{array} { r } { \mathbf q _ { t } = \gamma _ { t } ( Z ) \mathbf p _ { Z } + \sum _ { y \neq Z } \gamma _ { t } ( y ) \mathbf p _ { y } \geq \gamma _ { t } ( Z ) } \end{array}$ p entrywise, hence

$$
D _ { \mathrm { K L } } ( \mathbf { p } _ { Z } \Vert \mathbf { q } _ { t } ) \leq \sum _ { a = 1 } ^ { K } p _ { Z } ( a ) \log \frac { p _ { Z } ( a ) } { \gamma _ { t } ( Z ) p _ { Z } ( a ) } = - \log \gamma _ { t } ( Z ) = \log \Big ( 1 + \sum _ { y \neq Z } R _ { y } ^ { ( t ) } \Big ) .\tag{21}
$$

Conditional on $Z ,$ the tokens are i.i.d. from $\mathbf { p } _ { Z }$ , so by the strong law of large numbers

$$
\frac { 1 } { t } \log R _ { y } ^ { ( t ) } = \frac { 1 } { t } \sum _ { s = 1 } ^ { t } \log \frac { p _ { y } ( X _ { s } ) } { p _ { Z } ( X _ { s } ) } \xrightarrow [ t \to \infty ] { \mathrm { a . s . } } - D _ { \mathrm { K L } } ( { \bf p } _ { Z } \| { \bf p } _ { y } ) < 0 ,\tag{22}
$$

i.e. each wrong topic is suppressed exponentially in $t ,$ at a rate equal to its divergence from the true topic. This is the mechanism described in the main text; it is asymptotic in t at fixed $P _ { - }$

Resolution on the scale log D. A bound valid at every t follows from a fractional moment of the likelihood ratios. Using log( $\begin{array} { r } { 1 + \sum _ { y } R _ { y } ) \le \sum _ { y } \log ( 1 + R _ { y } ) \le \sum _ { y } \sqrt { R _ { y } } } \end{array}$ in $\operatorname { E q } . 2 1 $ , then the independence of the tokens given $Z ,$

$$
\begin{array} { r } { \mathbb { E } \left[ D _ { \mathrm { K L } } ( \mathbf { p } _ { Z } \| \mathbf { q } _ { t } ) \right] \leq \mathbb { E } \left[ - \log \gamma _ { t } ( Z ) \right] \leq \displaystyle \frac { 1 } { D } \sum _ { z = 1 } ^ { D } \sum _ { y \neq z } \mathbb { E } \left[ \sqrt { R _ { y } ^ { ( t ) } } \Big | Z = z \right] } \\ { = \displaystyle \frac { 1 } { D } \sum _ { z = 1 } ^ { D } \sum _ { y \neq z } \rho _ { z y } ^ { t } \leq ( D - 1 ) \rho _ { \star } ^ { t } , } \end{array}\tag{23}
$$

where $\begin{array} { r } { \rho _ { z y } : = \sum _ { a } \sqrt { p _ { z } ( a ) p _ { y } ( a ) } } \end{array}$ is the Bhattacharyya coefficient of topics z and $y ,$ and $\rho _ { \star } : = \operatorname* { m a x } _ { z \neq y } \rho _ { z y } < 1$ by the Cauchy–Schwarz inequality and the distinctness of the topics. Hence the excess loss of the specialised predictor is below any $\epsilon > 0$ as soon as

$$
t \geq \frac { \log { \left[ ( D - 1 ) / \epsilon \right] } } { \log ( 1 / \rho _ { \star } ) } ,\tag{24}
$$

which is the log D scale of Sec. 2.1, with a rate set by the closest pair of topics. The guaranteed rate $\log ( 1 / \rho _ { z y } )$ is at most half of the almost-sure rate in Eq. 22, since log $\rho _ { z y } \geq - { \textstyle \frac { 1 } { 2 } } D _ { \mathrm { K L } } ( { \bf p } _ { z } \| { \bf p } _ { y } )$ by Jensen’s inequality, and $\rho _ { \star }$ grows with D as the closest pair among D random topics gets closer: Eq. 24 is a statement about the dependence on t at fixed $P ,$ not a sharp constant. For the constant, the typical pair is the relevant one: two random topics have $\overline { { D _ { \mathrm { K L } } ( \mathbf { p } \| \mathbf { p } ^ { \prime } ) } } = 1 - 1 / K$ and $\overline { { \rho ( { \bf p } , { \bf p } ^ { \prime } ) } } \simeq \pi / 4$ (App. A), and if the $D - 1$ divergences from the true topic concentrate near their mean, the odds of a wrong topic in Eq. 21 are approximately $\left( D - 1 \right) e ^ { - \left( \hat { 1 } - 1 / K \right) : }$ <sup>t</sup>, so that the floor is reached for t $\log ( D - 1 )$ . This last estimate is heuristic, as it assumes a concentration of the pairwise divergences that we do not prove; it is consistent with the half-descent points in Fig. 2, which grow linearly in log D.

## C THE IN-CONTEXT PREDICTOR

This appendix proves the statements of Sec. 2.2: the form and optimality of $\mathbf { r } _ { t } ,$ the loss decomposition Eq. 6, and the exact finite-t risk behind its expansion. Throughout, the document is OOD: $\mathbf { p } _ { \star } \sim \operatorname { D i r } _ { K } ( \mathbf { 1 } )$ and, given $\mathbf { p } _ { \star }$ , the tokens are i.i.d. from $\mathbf { p } _ { \star }$

Posterior and add-one rule. The prior density is constant on the simplex and the likelihood of a context is $\prod _ { a } p _ { \star } ( a ) ^ { C _ { a } }$ , so the posterior density is proportional to $\begin{array} { r } { \prod _ { a } p _ { \star } ( a ) ^ { C _ { a } } ; } \end{array}$ : this is $\operatorname { D i r } _ { K } ( C _ { 1 } + 1 , \dots , C _ { K } + 1 )$ , whose density is proportional to $\Pi _ { a } p _ { a } ^ { \alpha _ { a } - 1 }$ with $\alpha _ { a } = C _ { a } + 1$ . Given $\mathbf { p } _ { \star }$ and the context, $X _ { t + 1 } \sim \mathbf { p } _ { \star }$ , and the mean of $\operatorname { D i r } _ { K } ( \alpha )$ is $\alpha _ { a } / \sum _ { b } \alpha _ { b } .$ , so the posterior predictive distribution is

$$
r _ { t } ( a \mid X _ { \leq t } ) = \operatorname { \mathbb { E } } \left[ p _ { \star } ( a ) \mid X _ { \leq t } \right] = { \frac { C _ { a } + 1 } { t + K } } ,\tag{25}
$$

which is Eq. 5.

Optimality. As in App. $\mathbf { B } , \mathbf { r } _ { t }$ is the conditional distribution of $X _ { t + 1 }$ given the context, so for any predictor f

$$
\begin{array} { r } { \mathbb { E } \left[ - \log f _ { t } ( X _ { t + 1 } \mid X _ { \leq t } ) \mid X _ { \leq t } \right] = H ( \mathbf { r } _ { t } ) + D _ { \mathrm { K L } } ( \mathbf { r } _ { t } \| \mathbf { f } _ { t } ) \geq H ( \mathbf { r } _ { t } ) , } \end{array}\tag{26}
$$

with equality if and only if $\mathbf { f } _ { t } = \mathbf { r } _ { t } ;$ averaging over contexts, $\mathcal { L } _ { t } ^ { \star } [ \mathbf { f } ] \geq \mathcal { L } _ { t } ^ { \star } [ \mathbf { r } ]$ . Unlike $\mathbf { q } _ { t }$ , the predictor $\mathbf { r } _ { t }$ does not depend on the topic matrix: on a document generated from any topic p, its loss is a function $R _ { t } ( \mathbf { \bar { p } } ) : = \mathbb { E } [ - \log \bar { r _ { t } } ( X _ { t + 1 } \mid X _ { \leq t } ) \mid \mathbf { p } ]$ of that topic alone. Its ID loss at fixed $P$ is $\begin{array} { r } { \mathcal { L } _ { t } [ \mathbf { r } ] = \frac { 1 } { D } \sum _ { z } R _ { t } ( \mathbf { p } _ { z } ) } \end{array}$ and its OOD loss is $\mathcal { L } _ { t } ^ { \star } [ \mathbf { r } ] = \overline { { R _ { t } ( \mathbf { p } ) } }$ , whence $\overline { { \boldsymbol { \mathcal { L } } _ { t } [ \mathbf { r } ] } } = \mathcal { L } _ { t } ^ { \star } [ \mathbf { r } ]$

Loss decomposition. Conditioning on p<sub>⋆</sub> and the context, $X _ { t + 1 } \sim \mathbf { p } _ { \star }$ , so $\mathbb { E } [ - \log r _ { t } ( X _ { t + 1 } \ )$ $X _ { \le t } ) \mid \mathbf { p } _ { \star } , X _ { \le t } ] = H ( \mathbf { p } _ { \star } ) + D _ { \mathrm { K L } } ( \mathbf { p } _ { \star } \| \mathbf { r } _ { t } ) ;$ ; averaging over p and the context gives the first equality in Eq. 6, with $\mathbb { E } [ H ( \mathbf { p } _ { \star } ) ] = { \overline { { H ( \mathbf { p } ) } } }$ because p<sub>⋆</sub> has the law of the training topics. Moreover, by Eq. 26 at $\mathbf { f } = \mathbf { r } , { \mathcal { L } } _ { t } ^ { \star } [ \mathbf { r } ] { \stackrel { } { = } } H ( X _ { t + 1 } \mid X _ { < t } )$ is a conditional entropy; since the tokens are exchangeable given $ { \mathbf { p } } _ { \star } , H ( X _ { t + 2 } \mid X _ { \leq t } ) = H ( X _ { t + 1 } \mid X _ { \leq t } )$ , and conditioning on one more token shows that $\bar { \mathcal { L } } _ { t } ^ { \star } [ \mathbf { r } ]$ is non-increasing in t.

Exact finite-t risk. Integrating the likelihood against the uniform prior, the count vector of a context is uniform over the $( \overset { t + K - 1 } { K - 1 } )$ weak compositions of t into K parts,

$$
\mathbb { P } \left( C ( X _ { \leq t } ) = \mathbf { n } \right) = { \frac { t ! } { \prod _ { a } n _ { a } ! } } { \frac { \Gamma ( K ) \prod _ { a } \Gamma ( n _ { a } + 1 ) } { \Gamma ( t + K ) } } = { \binom { t + K - 1 } { K - 1 } } ^ { - 1 } , \qquad \sum _ { a } n _ { a } = t .\tag{27}
$$

Hence, by Eq. 25 and the symmetry of the entries, for $K \geq 2$

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { t } ^ { \star } [ \mathbf { r } ] = \log ( t + K ) - \mathbb { E } \left[ \sum _ { a } r _ { t } ( a \mid X _ { \le t } ) \log ( C _ { a } + 1 ) \right] } \\ { \displaystyle \quad = \log ( t + K ) - \frac { K } { ( t + K ) \binom { t + K - 1 } { K - 1 } } \sum _ { n = 0 } ^ { t } ( n + 1 ) \log ( n + 1 ) \binom { t - n + K - 2 } { K - 2 } , } \end{array}\tag{28}
$$

where $\binom { t - n + K - 2 } { K - 2 }$ counts the compositions with $C _ { 1 } = n$ . In particular ${ \mathcal { L } } _ { 0 } ^ { \star } [ \mathbf { r } ] =$ log K and $\mathcal { L } _ { t } ^ { \star } [ \mathbf { r } ] $ $\overline { { H ( \mathbf { p } ) } }$ as $t \to \infty$ . Expanding at large t and fixed K gives the second equality in Eq. 6,

$$
\mathcal { L } _ { t } ^ { \star } [ \mathbf { r } ] = \psi ( K + 1 ) - \psi ( 2 ) + \frac { K - 1 } { 2 t } + O _ { K } \left( \frac { \log t } { t ^ { 2 } } \right) ,\tag{29}
$$

where the logarithm in the remainder comes from the mass that the uniform prior puts near the boundary of the simplex. The leading correction is one half of the number of parameters of a topic, $K - 1$ , per observation, and the expansion requires $t \gg K :$ at $K = 6 4$ and $t = 1 2 8$ , Eq. 28 gives an excess of 0.15 nats over the floor, against 0.25 for the leading term alone; the former is the gap between the dashed line and the floor at the end of the profiles in Fig. 2.

## D THE BAYESIAN LEARNER

This appendix proves the statements of Sec. 3: the form and optimality of the Bayesian learner, the information identity, the $D K / N T$ law and its fixed-position version.

Posterior and predictor. Let $\mathcal { C } _ { N } = ( X _ { 1 : T } ^ { ( n ) } ) _ { n = 1 } ^ { N }$ be the corpus, with count vectors $C ^ { ( n ) }$ , and write $\begin{array} { r } { \ell _ { C } ( \mathbf { p } ) : = \prod _ { a } p ( a ) ^ { C _ { a } } } \end{array}$ for the likelihood of a count vector under a topic. Given a candidate topic matrix $\tilde { P } .$ , integrating out the latent topic of each document gives the likelihood of corpus and context,

$$
L ( \tilde { P } ) : = \prod _ { n = 1 } ^ { N } \left[ \frac { 1 } { D } \sum _ { z = 1 } ^ { D } \ell _ { C ^ { ( n ) } } ( \tilde { { \bf p } } _ { z } ) \right] \times \frac { 1 } { D } \sum _ { z = 1 } ^ { D } \ell _ { C ( X _ { \le t } ) } ( \tilde { { \bf p } } _ { z } ) ,\tag{30}
$$

up to multinomial factors that do not depend on ${ \tilde { P } } .$ The posterior of $P$ is $\pi ( \mathrm { d } \tilde { P } ) \propto L ( \tilde { P } ) \mu ( \mathrm { d } \tilde { P } )$ with $\mu$ the product of D uniform priors on the simplex, and Eq. 7 reads $\begin{array} { r } { \hat { \mathbf { q } } _ { t } = \int \mathbf { q } _ { t , \tilde { P } } ( X _ { \leq t } ) \pi ( \mathrm { d } \tilde { P } ) } \end{array}$ where $\mathbf { q } _ { t , \tilde { P } }$ is the specialised predictor of Eq. 3 computed with ${ \tilde { P } } .$ . By construction $\hat { \mathbf { q } } _ { t }$ is the conditional law of $X _ { t + 1 }$ given $\left( { \mathcal { C } } _ { N } , X _ { \leq t } \right)$ under the joint law of topics, corpus and test document, so, as in $\mathrm { A p p }$ . B, every predictor f built from corpus and context satisfies $\overline { { \mathcal { L } _ { t } [ \mathbf { f } ] } } \geq \overline { { \mathcal { L } _ { t } [ \hat { \mathbf { q } } ] } }$ , with equality if and only if $\mathbf { f } _ { t } = \hat { \mathbf { q } } _ { t }$

Excess and mutual information. At fixed $P ,$ , conditioning on the context, $X _ { t + 1 } \sim \mathbf { q } _ { t }$ and the corpus is independent of the test document, so $\mathcal { L } _ { t } [ \hat { \mathbf { q } } ] - \mathcal { L } _ { t } [ \mathbf { q } ] = \mathbb { E } [ D _ { \mathrm { K L } } ( \mathbf { q } _ { t } \Vert \hat { \mathbf { q } } _ { t } ) ] \geq 0$ . Averaged over $P ,$ , the two losses are the conditional entropies $\bar { H } \bar { ( X _ { t + 1 } ) } \bar { \mid } \bar { X } _ { \leq t } , \bar { \mathcal { C } } _ { N } )$ and $\cdot \bar { H } ( \bar { X } _ { t + 1 } \mid X _ { \leq t } , \bar { \mathcal { C } } _ { N } , P )$ whence

$$
\varepsilon _ { t } ( N ) = I ( P ; X _ { t + 1 } \mid X _ { \leq t } , \mathcal { C } _ { N } ) .\tag{31}
$$

The second entropy does not change if one more document $X _ { 1 : T } ^ { ( N + 1 ) }$ is added to the corpus, since given $P$ the corpus is irrelevant, so $\varepsilon _ { t } ( N ) - \varepsilon _ { t } ( N + 1 ) = I ( X _ { t + 1 } ; X _ { 1 : T } ^ { ( N + 1 ) } \mid X _ { \leq t } , \mathcal { C } _ { N } ) \geq 0 ;$ the learning curve is non-increasing in $N$ . Summing Eq. 31 over positions with the chain rule, and using that the test document has the law of $X _ { 1 : T } ^ { ( N + 1 ) }$ and is independent of the corpus given $P ,$

$$
\sum _ { t = 0 } ^ { T - 1 } \varepsilon _ { t } ( N ) = I \big ( P ; X _ { 1 : T } ^ { ( N + 1 ) } \mid \mathcal { C } _ { N } \big ) = I ( P ; \mathcal { C } _ { N + 1 } ) - I ( P ; \mathcal { C } _ { N } ) ,\tag{32}
$$

which is Eq. 9, the document-level form of the cumulative predictive risk identity of Clarke & Barron (1990).

The DK/NT law. Up to a relabelling of the topics, the law $\begin{array} { r } { Q _ { P } : = \frac { 1 } { D } \sum _ { z } \mathbf p _ { z } ^ { \otimes T } } \end{array}$ of a document has $d = \overset { \vartriangle } { \boldsymbol { D } } ( \boldsymbol { K } - 1 )$ parameters. Assume that $P \mapsto Q _ { P }$ is identifiable up to relabelling and regular: twice continuously differentiable in local coordinates, with positive definite Fisher information, and with a prior density that is positive and continuous. Then, for fixed $D , K , T$ and $N \to \infty$ , Theorem 1 of Clarke & Barron (1994) gives

$$
I ( P ; { \mathcal { C } } _ { N } ) = { \frac { d } { 2 } } \log N + c + o ( 1 ) ,\tag{33}
$$

with a constant c that depends on prior and document model but not on $N .$ . The increment in Eq. 32 does not follow by direct subtraction, since an $o ( 1 )$ term need not have $o ( N ^ { - 1 } )$ increments, but it does follow from monotonicity. Let $\begin{array} { r } { a _ { N } : = \sum _ { t } \varepsilon _ { t } ( N ) } \end{array}$ ), non-increasing in $N$ , and for $\lambda > 1$ let $m = \lfloor \lambda N \rfloor$ and $l = \lfloor N / \lambda \rfloor$ . Summing $a _ { n }$ over $N \leq n <$ m and over $l \leq n < N$ and using monotonicity,

$$
\frac { I ( P ; \mathcal C _ { m } ) - I ( P ; \mathcal C _ { N } ) } { m - N } \leq a _ { N } \leq \frac { I ( P ; \mathcal C _ { N } ) - I ( P ; \mathcal C _ { l } ) } { N - l } ,\tag{34}
$$

where both numerators equal $\textstyle { \frac { d } { 2 } } \log \lambda + o ( 1 )$ by $\operatorname { E q } . 3 3$ . Letting $N  \infty$ and then $\lambda \downarrow 1$ squeezes $N a _ { N }$ between $\frac { d } { 2 } \frac { \log \lambda } { \lambda - 1 }$ and $\frac { d } { 2 } \frac { \lambda \log \lambda } { \lambda - 1 }$ , both of which tend to $d / 2 \colon$ ; dividing by $T$ gives Eq. 10.

Identifiability. For $T \geq 3$ the third-order moment of the tokens, $\begin{array} { r } { M _ { 3 } = \frac { 1 } { D } \sum _ { z } \mathbf { p } _ { z } ^ { \otimes 3 } } \end{array}$ , is a marginal of $Q _ { P }$ P, and by Kruskal’s theorem (Kruskal, 1977; Allman et al., 2009) its decomposition into $D$ rank-one terms is unique up to permutation when $3 \mathrm { r a n k } _ { \mathrm { K } } ( P ) \geq 2 D + 2$ , where the Kruskal rank $\mathrm { r a n k } _ { \mathrm { K } } ( P )$ is the largest k such that any k columns of $P$ are linearly independent. For topics drawn from the simplex, ran $\mathrm { \bar { \mathrm {  ~ \Lambda } } } _ { \mathrm {  ~ \Lambda } } ( P ) = \mathrm { m i n } \mathrm { \bar { \{ } }  K , D \}$ almost surely, so the condition holds for $D \ \leq \ K$ (and up to $D \le 3 K / 2 - 1 )$ , and the topics can be recovered constructively: the second moment $\begin{array} { r } { M _ { 2 } = \bar { \frac { 1 } { D } } \sum _ { z } \mathbf { p } _ { z } \mathbf { p } _ { z } ^ { \top } } \end{array}$ has rank D and whitens $M _ { 3 }$ into an orthogonally decomposable tensor whose factors are the topics (Anandkumar et al., 2014). For larger $D _ { \colon }$ , lift the topics to $\mathbf { v } _ { z } = \mathbf { p } _ { z } ^ { \otimes m }$ , which lie in the space of symmetric tensors, of dimension $s _ { m } ~ = ~ \binom { K + m - 1 } { m }$ , and span it as $\mathbf { p } _ { z }$ ranges over the simplex, since a homogeneous polynomial vanishing on the simplex vanishes identically; some $s _ { m } \times s _ { m }$ minor of $\left[ \mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { D } \right]$ is then a nonzero polynomial in the topic entries, so the lifted topics are linearly independent almost surely whenever $D \le s _ { m }$ . In that case the 2m-token moment $\begin{array} { r } { \frac { 1 } { D } \sum _ { z } \mathbf { v } _ { z } \mathbf { v } _ { z } ^ { \top } } \end{array}$ has rank $D$ , the rank of the Hankel matrix of the token process, and together with the $( 2 m { + } 1 )$ -token moment it determines the law of documents of every length, hence $P$ up to relabelling (Balle et al., 2014) (App. F). Moreover, for $T \geq 2 m + 1$ the Fisher information of a document is nonsingular: a perturbation $\delta P$ that leaves the law of a document unchanged to first order leaves the $( 2 \bar { m } + 1 )$ -token marginal $\begin{array} { r } { \frac { 1 } { D } \sum _ { z } \mathbf { v } _ { z } \otimes \mathbf { v } _ { z } \otimes \mathbf { p } _ { z } } \end{array}$ unchanged, and contracting its variation with $a _ { z } \otimes a _ { z } \otimes \mathrm { I d }$ , where the functionals $a _ { z }$ satisfy $a _ { z } ( \mathbf { v } _ { z ^ { \prime } } ) = \delta _ { z z ^ { \prime } }$ , gives $\delta \mathbf { p } _ { z } +$ $2 a _ { z } ( \delta \mathbf { v } _ { z } ) \mathbf { p } _ { z } = 0$ , whose sum over coordinates forces $a _ { z } ( \delta \mathbf { v } _ { z } ) = 0$ and then $\delta \mathbf { p } _ { z } ~ = ~ 0$ . Hence $D \leq \binom { K + m - 1 } { m }$ and $T \geq 2 m + 1$ suffice almost surely: $D \leq K$ with $T \geq 3 , D \leq K ( K { + } 1 ) / 2$ with $T \geq 5$ , and so on; these conditions are sufficient, not necessary, and they are the only place where the genericity of the topic matrix is used. Regularity additionally excludes the sets, of prior measure zero, where two topics coincide or a topic lies on the boundary of the simplex.

Fixed position: the resolved learner. Reveal to the learner the topic labels of the training documents and of the test document. Conditional on the labels, by the conjugacy of App. C, the Bayesian predictor is the add-one rule on the pooled counts of the test topic,

$$
\hat { q } _ { t } ^ { \mathrm { r e s } } ( a \mid X _ { \leq t } ) = \frac { M _ { Z , a } + C _ { a } + 1 } { N _ { Z } T + t + K } ,\tag{35}
$$

where $N _ { Z }$ is the number of training documents of topic Z and $M _ { Z , a }$ their total count of token a: the in-context predictor of Eq. 5 applied to $m = N _ { Z } \bar { T } + t$ tokens. Averaged over topics, its excess over the entropy rate $\begin{array} { r } { \frac { 1 } { D } \sum _ { z } H ( \mathbf p _ { z } ) } \end{array}$ is therefore $\mathbb { E } [ g _ { K } ( N _ { Z } T + t ) ]$ ], with $g _ { K } ( m ) : = \mathcal { L } _ { m } ^ { \star } [ \mathbf { r } ] - \overline { { H ( \mathbf { p } ) } }$ the excess of the in-context predictor in Eq. 6 and $N _ { Z } \sim \mathrm { B i n } ( N , 1 / D )$ . If $N / D \to \infty$ then $N _ { Z } / ( N / D ) \to 1$ in probability, and if moreover $t D / ( N T ) \to 0$ and log $\dot { T } = o ( \dot { N } / D )$ , so that the event that the test topic is absent from the corpus is negligible on the $1 / ( N T )$ scale, Eq. 6 gives

$$
\varepsilon _ { t } ^ { \mathrm { r e s } } ( N ) : = \mathbb { E } \left[ g _ { K } ( N _ { Z } T + t ) \right] = \frac { D ( K - 1 ) } { 2 N T } \left( 1 + o ( 1 ) \right) .\tag{36}
$$

The resolved learner is not the Bayesian learner, which must also infer the labels and the topic matrix. For a fixed $P$ with distinct topics, and assuming identifiability from a marginal of $k \leq T$ tokens, the two excesses differ by at most

$$
C ( P ) \left( e ^ { - c ( P ) t } + N e ^ { - c ( P ) T } \right) + \exp \big \{ - a ( P ) N + O _ { P } \big ( \log [ N ( T + t ) ] \big ) \big \} ,\tag{37}
$$

with $c ( P ) = \log ( 1 / \rho _ { \star } )$ from Eq. 23 and $a ( P ) > 0$ . The first two terms are the costs of unresolved test and training labels, by the bound of App. B applied to the test context and to each of the N training documents; the last is the posterior mass on topic matrices outside a fixed neighbourhood of $P ,$ exponentially small in N by a likelihood-ratio test on the identifying marginal, against the polynomial prior mass of the neighbourhood. Hence the Bayesian learner obeys Eq. 36 at fixed P whenever t and $T$ grow like A log N with A large enough for Eq. 37 to be $o ( \bar { 1 } / ( N T ) )$ : the fixedposition form of the $D K / N T$ law. The statement is conditional on $P ;$ averaging it over the prior would require uniform control over nearly coinciding topics, for which $c ( P )$ is arbitrarily small.

## E THE IN-CONTEXT SOLUTION

This appendix supports $\mathtt { S e c }$ . 4: the estimation excess of a fitted parametric family, the count-linear family that contains the in-context predictor, and the gain $G ( K , T )$ with its interpretation and expansions.

Estimation excess of a fitted family. Let $\{ f _ { \theta } : \theta \in \Theta \subset \mathbb { R } ^ { d } \}$ be a family of distributions of a document $Y ~ = ~ X _ { 1 : T }$ , with all d coordinates of θ free, and let $\ell ( \theta ; Y ) : = - \log f _ { \theta } ( Y )$ and $\mathcal { R } ( \theta ) : = \mathbb { E } [ \ell ( \theta ; Y ) ]$ be its log loss and its population risk under the true law of Y. Let $\theta _ { \star }$ be an interior minimiser of $\mathcal { R } _ { : }$ , isolated up to finite symmetries, around which $f _ { \theta } ( y ) > 0$ is three times continuously differentiable, and define the risk curvature and the gradient covariance

$$
\mathsf { H } : = \nabla ^ { 2 } \mathcal { R } ( \theta _ { \star } ) \succ 0 , \qquad \mathsf { J } : = \mathbb { E } \left[ \nabla \ell ( \theta _ { \star } ; Y ) \nabla \ell ( \theta _ { \star } ; Y ) ^ { \top } \right] ;\tag{38}
$$

the gradient has zero mean because $\theta _ { \star }$ is a minimiser. Let $\hat { \theta } _ { N }$ be a consistent fit on N independent documents whose empirical gradient is $o _ { \mathbb { P } } ( N ^ { - 1 / 2 } )$ , as for an interior minimiser of the empirical risk $\widehat { \mathcal { R } } _ { N }$ , and let $g _ { N } : = \nabla \widehat { \mathcal { R } } _ { N } ( \theta _ { \star } )$ , which has mean zero and covariance $\mathsf { J } / N$ by independence. b bExpanding the empirical gradient and the population risk around $\theta _ { \star }$ ,

$$
\begin{array} { r } { \widehat { \theta } _ { N } - \theta _ { \star } = - \mathsf { H } ^ { - 1 } g _ { N } + u _ { N } , \qquad \mathcal { R } ( \widehat { \theta } _ { N } ) - \mathcal { R } ( \theta _ { \star } ) = \frac { 1 } { 2 } ( \widehat { \theta } _ { N } - \theta _ { \star } ) ^ { \top } \mathsf { H } ( \widehat { \theta } _ { N } - \theta _ { \star } ) + v _ { N } , } \end{array}\tag{39}
$$

with $u _ { N } ~ = ~ o _ { \mathbb { P } } ( N ^ { - 1 / 2 } )$ and $v _ { N } ~ = ~ o _ { \mathbb { P } } ( N ^ { - 1 } )$ . If the remainders are also small in expectation, $\mathbb { E } \| u _ { N } \| ^ { 2 } = o ( N ^ { \dot { - } 1 } )$ and $\mathbb { E } | v _ { N } | = o ( N ^ { - 1 } )$ , then substituting and using $\mathbb { E } [ g ^ { \top } A g ] = \operatorname { t r } ( A { \dot { \operatorname { C o v } } } g )$ for centred g,

$$
\mathbb { E } \left[ \mathcal { R } ( \hat { \theta } _ { N } ) - \mathcal { R } ( \theta _ { \star } ) \right] = \frac { d _ { \mathrm { e f f } } } { 2 N } + o ( N ^ { - 1 } ) , \qquad d _ { \mathrm { e f f } } : = \mathrm { t r } \left( \mathsf { J } \mathsf { H } ^ { - 1 } \right) .\tag{40}
$$

If the family is well specified, i.e. the true law is $f _ { \theta _ { \star } }$ , differentiating $\textstyle \sum _ { y } f _ { \theta } ( y ) = 1$ twice gives $\mathsf { H } =$ J, the Fisher information of one document, and $d _ { \mathrm { e f f } } = d$ whenever it is nonsingular. The remainder bounds hold, for instance, for an empirical-risk minimiser over a compact set that contains $\theta _ { \star }$ in its interior and on which probabilities are bounded below; they are not automatic for an unrestricted fit, nor for a transformer. Divided by T, Eq. 40 is a per-token excess $d / ( 2 N T )$ . For the topic matrix itself, $d = D ( K - 1 )$ , and the Fisher information is nonsingular almost surely for $D \leq K$ and $T \geq 3$ (the Jacobian of the three-token marginal is injective when the topics are linearly independent), so a fitted dictionary recovers Eq. 10 by an argument on a fitted parameter rather than on the posterior; for the count map below, $d \simeq K ^ { 2 }$

The count-linear family. The predictors that are linear in the counts up to normalisation,

$$
f _ { A , \mathbf { b } } ( a \mid X _ { \leq t } ) : = \frac { \left( A C ( X _ { \leq t } ) + \mathbf { b } \right) _ { a } } { \mathbf { 1 } ^ { \top } \left( A C ( X _ { \leq t } ) + \mathbf { b } \right) } , \qquad A \in \mathbb { R } ^ { K \times K } , \mathbf { b } \in \mathbb { R } ^ { K } ,\tag{41}
$$

contain the in-context predictor at $( A , \mathbf { b } ) \ = \ ( I , \mathbf { 1 } )$ . Counts are bounded by $T ,$ so the family is positive and smooth on a neighbourhood of this point, and it has $d = K ^ { 2 } + \dot { K } - 1$ free parameters, the overall scale of $( A , \mathbf { b } )$ being immaterial. On OOD documents the family is well specified, since the law of an OOD document is the product of its add-one predictors $( { \mathrm { A p p . ~ C } } ) $ , so $\bar { \theta _ { \star } } = ( I , { \bf 1 } )$ and Eq. 40 gives the excess $d / ( 2 N T ) \stackrel { } { \simeq } K ^ { 2 } / ( 2 N T )$ of Sec. 4 for a learner that fits this family. On the training distribution the family is misspecified at finite $D _ { \mathbf { \delta } }$ , its optimum being a fit to $\mathbf { q } _ { t }$ , a ratio of exponentials of the counts. The misspecification vanishes as $D \to \infty \colon$ writing Eq. 3 as a ratio of averages over the dictionary,

$$
q _ { t } ( a \mid X _ { \leq t } ) = \frac { \frac { 1 } { D } \sum _ { z } p _ { z } ( a ) \prod _ { b } p _ { z } ( b ) ^ { C _ { b } } } { \frac { 1 } { D } \sum _ { z } \prod _ { b } p _ { z } ( b ) ^ { C _ { b } } } \xrightarrow [ D  \infty ] { \mathbf { a . s . } } \frac { \overline { { p ( a ) \prod _ { b } p ( b ) ^ { C _ { b } } } } } { \overline { { \prod _ { b } p ( b ) ^ { C _ { b } } } } } = \frac { C _ { a } + 1 } { t + K } = r _ { t } ( a \mid X _ { \leq t } ) ,\tag{42}
$$

by the strong law of large numbers applied to numerator and denominator at fixed K and count vector, and the Dirichlet moment $\overline { { \prod _ { b } p ( b ) ^ { n _ { b } } } } = \Gamma ( K ) \prod _ { b } \Gamma ( n _ { b } + 1 ) / \Gamma ( K + \sum _ { b } n _ { b } )$ , of which Eq. 15 is the one-entry case. The convergence is pointwise in the count vector, and the dictionary size needed for a given accuracy grows with $t ,$ since the weights in Eq. 42 concentrate on a neighbourhood of the empirical frequencies that shrinks with $t .$ Note that the family cannot be reached through the specialised predictor of a small dictionary. $\mathbf { q } _ { t }$ is a convex combination of the D topics, so its Jaco bian with respect to the counts has rank at most $D - 1$ , whereas that of the add-one map is $K - 1 \colon$ a learner whose predictor tracks the in-distribution optimum can realise the count-linear rule only if $D \geq K$

The gain of the in-context predictor. On an OOD document every token is marginally uniform, $\mathbb { P } ( X _ { t + 1 } = a ) = \overline { { p ( a ) } } = 1 / K$ , so $H ( X _ { t + 1 } ) = \log K$ , and $\mathcal { L } _ { t } ^ { \star } [ \mathbf { r } ] = H ( X _ { t + 1 } \mid X _ { \leq t } )$ by App. C; hence

$$
\log K - \mathcal { L } _ { t } ^ { \star } [ \mathbf { r } ] = I ( X _ { t + 1 } ; X _ { \leq t } ) ,\tag{43}
$$

$$
T G ( K , T ) = T \log K - H ( X _ { 1 : T } ) = T \left[ \log K - \overline { { H ( \mathbf { p } ) } } \right] - I ( \mathbf { p } _ { \star } ; X _ { 1 : T } ) ,\tag{44}
$$

by the chain rule, and $H ( X _ { 1 : T } ) = H ( X _ { 1 : T } \mid \mathbf { p } _ { \star } ) + I ( \mathbf { p } _ { \star } ; X _ { 1 : T } )$ with $H ( X _ { 1 : T } \mid \mathbf { p } _ { \star } ) = T { \overline { { H ( \mathbf { p } ) } } }$ The gain per token is thus the entropy gap between the uniform distribution and a random topic, $1 - \bar { \gamma _ { \mathrm { E } } } - \bar { 1 / ( 2 K ) } + O ( K ^ { - 2 } )$ by Eq. 16, minus the information about its own topic that the document leaves unrevealed. Its exact value follows from Eq. 28, and its two regimes are

$$
G ( K , T ) = \log K - { \overline { { H ( { \bf p } ) } } } - { \frac { K - 1 } { 2 } } { \frac { \log T } { T } } + O _ { K } \left( \frac { 1 } { T } \right) \qquad ( T \gg K ) ,
$$

$$
G ( K , T ) = \left( 2 \log 2 - 1 \right) \frac { T - 1 } { 2 K } + { \cal O } \left( \frac { T ^ { 2 } } { K ^ { 2 } } \right)
$$

$$
( K \gg T ) .\tag{45}
$$

The first follows by summing Eq. 29 over positions, which gives $\begin{array} { l } { { I ( { \bf p } _ { \star } ; X _ { 1 : T } ) = \frac { K - 1 } { 2 } } } \end{array}$ log T + $O _ { K } ( 1 ) { : }$ a long context reveals its topic up to the usual parametric cost. The second follows from Eq. 28 at $t \ll K$ : a given token appears once in the context with probability $\mathbb { P } ( C _ { a } = 1 ) = ( K -$ $1 ) \dot { t } / [ ( t + K - 1 ) ( t + K - 2 ) ] \stackrel { . } { \simeq } t / K$ and more than once with probability $\dot { O } ( t ^ { 2 } / K ^ { 2 } )$ , so that $\begin{array} { r } { \mathcal { L } _ { t } ^ { \star } [ \mathbf { r } ] = \log ( t + K ) - \frac { K } { t + K } 2 \log 2 \operatorname { l o g } 2 \operatorname { l p } ( \dot { C } _ { a } = 1 ) + O ( t ^ { 2 } / K ^ { 2 } ) = \log K - ( 2 \log 2 - 1 ) t / K + O ( t ^ { 2 } / K ^ { 2 } ) } \end{array}$ in a short context most tokens are new, and the gain comes from the few repeated ones.

## F THE SPECTRAL BASELINE

This appendix supports Sec. 5.1: the Hankel formulation of spectral learning, the spectra of the pair and four-token moments under the random topic model, the noise of their empirical estimates, and the resulting sample complexities. Its asymptotic statements are in the joint limit $D , K \to \infty$ with $D / K$ , respectively $D / K ^ { \frac { \mathbf { \lambda } } { 2 } }$ , fixed.

Hankel matrix and subspace. For the token process, the Hankel matrix has entries $H ( u , v ) : =$ $\mathbb { P } ( X _ { 1 : | u | + | v | } = u v )$ , indexed by a prefix string u and a suffix string v; since tokens are i.i.d. given the topic, $\begin{array} { r } { H ( u , v ) = \frac { 1 } { D } \sum _ { z } g _ { z } ( u ) g _ { z } ( v ) } \end{array}$ with $\begin{array} { r } { g _ { z } ( u ) = \prod _ { s } p _ { z } ( u _ { s } ) } \end{array}$ , so that rank $H \le D$ (Balle et al., 2014). Spectral learning first recovers the column space of H from a complete sub-block, i.e. one of rank $D _ { : }$ , then the observation operators from the block with one more symbol; for the random topic model the hidden state is constant within a document, the transition matrix is the identity and the observation matrix is $P$ (Hsu et al., 2012). The block on prefixes and suffixes of m tokens is $\begin{array} { r } { M _ { 2 m } = \frac { 1 } { D } \sum _ { z } ( \mathbf { p } _ { z } ^ { \otimes m } ) ( \mathbf { p } _ { z } ^ { \otimes m } ) ^ { \top } } \end{array}$ , whose nonzero spectrum is that of the $D \times D$ Gram matrix $\begin{array} { r } { \frac { 1 } { D } [ ( \mathbf { p } _ { i } ^ { \top } \mathbf { p } _ { j } ) ^ { m } ] _ { i j } } \end{array}$ . The lifted topics $\mathbf { p } _ { z } ^ { \otimes m }$ lie in the space of symmetric tensors, of dimension $s _ { m } ~ = ~ \binom { K + m - 1 } { m }$ , and span it as $\mathbf { p } _ { z }$ ranges over the simplex, since a homogeneous polynomial vanishing on the simplex vanishes identically; some $s _ { m } \times s _ { m }$ minor of the lifted topic matrix is therefore a nonzero polynomial in the topic entries, and rank $M _ { 2 m } = \operatorname* { m i n } \{ D , s _ { m } \}$ almost surely for topics drawn from the simplex. The single-token block $M _ { 2 }$ is thus complete for $D \leq K$ , the two-token block $M _ { 4 }$ for $D \le \dot { K } ( K + 1 ) / 2$ , and so on; the topics follow from $M _ { 3 } , M _ { 5 } , . . . ,$ for $D \leq K$ through the whitening of $M _ { 3 }$ by $M _ { 2 }$ and the decomposition of the resulting orthogonal tensor (Anandkumar et al., 2014), as recalled in App. D.

Spectrum of $M _ { 2 }$ . By Eq. 15, $\overline { { p _ { a } p _ { b } } } = 1 / ( K ( K + 1 ) )$ for $a \neq b$ and $\overline { { p _ { a } ^ { 2 } } } = 2 / ( K ( K + 1 ) )$ , so the mean of $M _ { 2 }$ over the topics is $( I + { \bf 1 1 } ^ { \top } ) / ( K ( K + 1 ) )$ ), with one eigenvalue $1 / K$ along 1 and $1 / ( K ( K + 1 ) )$ on its complement. For a realisation, write $\begin{array} { r } { \mathbf { p } _ { z } = \mathbf { 1 } / K + \pmb { \xi } , } \end{array}$ with $\pmb { \xi } _ { z }$ centred, orthogonal to 1, of covariance $\dot { \overline { { K ( K + 1 ) } } } ( I - { \bf { 1 1 } } ^ { \top } / K )$ and independent across topics; the nonzero spectrum of $\begin{array} { r } { M _ { 2 } = \frac { 1 } { D } P P ^ { \top } } \end{array}$ is that of $\begin{array} { r } { \frac { 1 } { D } \dot { P } ^ { \top } P = \frac { 1 } { D } \big [ \frac { 1 } { K } \mathbf { 1 } \mathbf { 1 } ^ { \top } + \Xi ^ { \top } \Xi \big ] } \end{array}$ with $\Xi = [ \xi _ { 1 } , \ldots , \xi _ { D } ]$ . In the $( K - 1 )$ dimensional complement of 1, the rescaled columns $\sqrt { K ( K + 1 ) } \xi _ { z }$ are independent, isotropic and log-concave, being uniform on a convex set, so that as $D , K \to \infty$ with $D / ( \bar { K } - 1 ) \to \alpha \ \bar { \in } \ ( 0 , 1 )$ the eigenvalues of $\frac { K ( K + 1 ) } { K - 1 } \Xi ^ { \top } \Xi$ follow the Marchenko–Pastur law of ratio α, with extreme eigenvalues converging to the edges $( 1 \pm { \sqrt { \alpha } } ) ^ { 2 }$ (Pajor & Pastur, 2009; Chafa¨ı & Tikhomirov, 2018). The rank-one term removes at most one eigenvalue from this bulk, by interlacing, and produces the outlier $D / K ;$ ; dividing by $D _ { : }$

$$
\lambda _ { 1 } ( M _ { 2 } ) = \frac { 1 } { K } \left( 1 + o _ { \mathbb { P } } ( 1 ) \right) , \qquad \lambda _ { 2 } ( M _ { 2 } ) , \lambda _ { D } ( M _ { 2 } ) = \frac { K - 1 } { D K ( K + 1 ) } \Big ( 1 \pm \sqrt { \frac { D } { K - 1 } } \Big ) ^ { 2 } ( 1 + o _ { \mathbb { P } } ( 1 ) ) ,\tag{46}
$$

and the remaining $K - D$ eigenvalues vanish: one spike at $1 / K$ and a bulk of $D - 1$ eigenvalues at scale $1 / ( D K )$ , as in Fig. 6.

Spectrum of $M _ { 4 } .$ For $K < D \le K ( K + 1 ) / 2$ the complete block is $M _ { 4 }$ , whose nonzero spectrum is that of $\frac { 1 } { D } [ ( \mathbf { p } _ { i } ^ { \top } \mathbf { p } _ { j } ) ^ { 2 } ] _ { i j }$ . Writing $\mathbf { p } ^ { \otimes 2 } \stackrel { \prime \prime } { = } \mathbf { u } ^ { \otimes 2 } + ( \mathbf { \dot { u } } \otimes \pmb { \xi } + \pmb { \xi } \otimes \mathbf { u } ) + \pmb { \xi } ^ { \otimes 2 }$ with $\mathbf { u } = \bar { \mathbf { 1 } } / K$ separates three scales. The mean feature has squared norm $( K + 3 ) / ( K ( K + 1 ) ^ { 2 } ) \simeq K ^ { - 2 }$ and gives one spike at $K ^ { - 2 } { \mathrm { : } }$ the linear features span $K - 1$ directions with covariance eigenvalue $\overline { { 2 } } / ( K ^ { 2 } ( K \overset { \cdot } { + } 1 ) )$ ; the centred quadratic features span the remaining $r _ { 2 } ~ = ~ K ( K - 1 ) \bar { / } 2$ directions, with average covariance eigenvalue $2 K ^ { - 4 } ( 1 \stackrel { \cdot } { + } O ( K ^ { - 1 } ) )$ . Neglecting the coupling between the blocks, a Marchenko–Pastur approximation of each predicts $K - 1$ eigenvalues of order $K ^ { - 3 }$ with edges $\begin{array} { r } { \frac { 2 } { K ^ { 2 } ( K + 1 ) } ( 1 \pm \sqrt { ( K - 1 ) / D } ) ^ { 2 } } \end{array}$ , and $D - K$ eigenvalues of order $1 / ( D K ^ { 2 } )$ , with edges $\begin{array} { r } { \frac { 1 + O ( K ^ { - 1 } ) } { D K ^ { 2 } } ( 1 \pm \sqrt { D / r _ { 2 } } ) ^ { 2 } } \end{array}$ when $D / r _ { 2 } \to \beta \in ( 0 , 1 )$ . This block picture is heuristic, since the quadratic features are neither isotropic nor log-concave and the coupling is neglected; Fig. 7 compares it with the spectrum of one realisation at $D = K ( K + 1 ) / 4$ , where both bulks follow the predicted curves and their Frobenius norms agree with the predictions within $7 \%$ for the linear block and $2 \%$ for the quadratic one, for $K$ from 64 to 192.

![](images/c3c440932e1a96d51990861fb67b7d70a1ed36b618de8292e068afbe869b9691.jpg)

![](images/6640ab6bdb3e269c0aacd83cff08c217a98794f18c07742bdbf89627fef98780.jpg)  
Figure 6: Population spectrum of $M _ { 2 }$ for $D \leq K$ . Left: ranked eigenvalues $\lambda _ { r } D K$ against the scaled rank $r / D$ at $\alpha = D / K = 1 / 2$ , for $K \in \{ 2 5 6 , 5 1 2 , 1 0 2 4 , 2 0 4 8 \}$ and one topic realisation per $( K , D )$ ; dashed: the ranked Marchenko–Pastur law; the spike is the outlier at $\lambda _ { 1 } \bar { D } K \approx D$ . Right: density of the bulk eigenvalues, spike removed, on the $\lambda D K$ scale, against the Marchenko–Pastur density (dashed).

![](images/9c2f658b0a210a2a4fbc3c93e040710d43d6e2bc231d4a437d63e30f37d34f64.jpg)

![](images/f2214f51ede956ca2284a6d5597b30486b34e83f2a4e48910b667736b56abd32.jpg)  
Figure 7: Population spectrum of $M _ { 4 }$ for $K < D \le K ( K + 1 ) / 2$ . Left: ranked eigenvalues of $M _ { 4 }$ for one realisation at $K = 1 9 2$ $D = K ( K + 1 ) / 4 = 9 2 6 4 \colon$ the spike at $K ^ { - 2 }$ (dotted), the $K - 1$ eigenvalues of the linear block and, beyond $r = K$ (dotted), the $D - K$ eigenvalues of the quadratic block, against the Marchenko–Pastur predictions of the two blocks (dashed). Right: Frobenius norm of each block against $K$ , for $D = K ( K + 1 ) / 4$ , with the parameter-free predictions (dashed).

Noise and sample complexity. Let $\widehat { M } _ { 2 }$ be the empirical pair moment from N independent pairs, so that $\widehat { M } _ { 2 } - M _ { 2 } = { \textstyle \frac { 1 } { N } } \sum _ { \mu } Y _ { \mu }$ with $Y _ { \mu } = \mathbf { x } _ { 1 } ^ { ( \mu ) } \mathbf { x } _ { 2 } ^ { ( \mu ) \top } - M _ { 2 }$ centred, $\| Y _ { \mu } \| _ { \mathrm { o p } } \leq 1 + 1 / K$ and $\mathbb { E } [ Y _ { \mu } Y _ { \mu } ^ { \top } ] \ \preceq \ \mathrm { d i a g } ( \bar { \mathbf { p } } )$ , with $\begin{array} { r } { \bar { \bf p } = \frac { 1 } { D } \sum _ { z } \mathbf p _ { z } } \end{array}$ the unigram marginal, of entries close to $1 / K$ . The Pmatrix Bernstein inequality (Tropp, 2012) then gives, with probability at least $1 - \delta$

$$
\| \widehat { M } _ { 2 } - M _ { 2 } \| _ { \mathrm { o p } } \le \sqrt { \frac { 2 \log ( 2 K / \delta ) } { K N } } + \frac { 2 \log ( 2 K / \delta ) } { 3 N } ,\tag{47}
$$

a noise level of order $( N K ) ^ { - 1 / 2 }$ ; the measured error follows $1 . 5 / \sqrt { N K }$ across eleven $( K , D )$ configurations (Fig. 8, left). Treating the noise as a generic perturbation, an eigenvalue λ of $M _ { 2 }$ is resolved once $\lambda \gtrsim 1 . 5 / \sqrt { N K }$ : the spike at $N \sim K$ , and the lower edge of the bulk, $\frac { K - 1 } { D K ( K + 1 ) } ( 1 -$ $\sqrt { \alpha } ) ^ { 2 }$ , at $N ^ { * } = 2 . 2 5 D ^ { 2 } K / ( 1 - \sqrt { \alpha } ) ^ { 4 }$ , i.e. at $N \sim D ^ { 2 } K$ up to an α-dependent constant. This criterion is heuristic, but the scale is confirmed by the eigenvectors: the overlap between the bulk eigenvector subspaces of $\widehat { M } _ { 2 }$ and $M _ { 2 }$ rises from its random value $( D - 1 ) / ( K - 1 )$ to one around $\breve { N / } ( D ^ { 2 } K ) = \dot { O } ( 1 )$ c(Fig. 8, right). For the four-token moment the same bound has a variance proxy of order $K ^ { - 2 }$ and gives a noise level of order $( N K ^ { 2 } ) ^ { - 1 / 2 }$ , so the spike at $K ^ { - 2 }$ is resolved at $\bar { N } \sim K ^ { 2 }$ , the linear block at $N \sim K ^ { 4 }$ and the quadratic block, of scale $\bar { 1 / ( D K ^ { 2 } ) }$ , at $N \sim D ^ { 2 } K ^ { 2 } ;$ extrapolating, the bulk of $M _ { 2 m }$ sits at $1 / ( D K ^ { m } )$ against a noise of $( N K ^ { m } ) ^ { - 1 / 2 }$ , and its resolution takes $N \sim \breve { D } ^ { 2 } K ^ { m }$

![](images/72c38d3cad8ef6cbad53e9d9fb35e44d1df42c068d8b010c5094b063d34d9380.jpg)

![](images/38f5e0fe736220f3fb76837d5263a1fd25526e8b91e5a119cb72880c74a8b3fd.jpg)  
Figure 8: Convergence of the empirical pair moment for<sub>1</sub> $D \ \leq \ K$ . Left: operator-norm error $\| \widehat { M } _ { 2 } - M _ { 2 } \| _ { \mathrm { o p } } ,$ , geometric mean over eight replicas, against NK for eleven $( K , D )$ configcurations, $\alpha = \ddot { D / K } \ \stackrel { \triangledown } { \in } \ \left\{ 1 / 4 , 1 / 2 , 3 / 4 \right\}$ times $K = \{ 1 2 8 , 2 5 6 , 5 1 2 \}$ plus $K = 2 5 6$ with $D \in$ $\{ 1 6 , 3 2 , 6 4 , 1 2 8 , 1 9 2 \}$ ; colour encodes K, line style α; all curves collapse on the dotted $1 . 5 / \sqrt { N K }$ circles mark where each curve crosses its own population $\lambda _ { D }$ , beyond which the lower bulk edge is resolved. Right: overlap $\begin{array} { r } { \frac { 1 } { D - 1 } \lVert U _ { 2 : D } ^ { \top } \widehat { U } _ { 2 : D } \rVert _ { F } ^ { 2 } } \end{array}$ between the bulk eigenvector subspaces of $M _ { 2 }$ and $\widehat { M } _ { 2 }$ , mean one standard deviation over replicas, against $N / ( D ^ { 2 } K ) ;$ ; each curve rises from the crandom-subspace value $( D - 1 ) / ( K - 1 )$ towards one around $\dot { N } / ( D ^ { 2 } K ) = O ( 1 )$ (dashed line at one). One topic realisation per $( \check { K } , D )$ , uniform topic prior.

## G EXPERIMENTAL DETAILS

This appendix specifies the transformer experiments of Sec. 5. The code to reproduce them is available at https://github.com/fracagnetta/random-topic-model.

Data. Every run draws its own topic matrix $P ,$ with D topics from $\operatorname { D i r } _ { K } ( { \bf 1 } )$ and uniform topic weights, and from it N training documents and 32 768 validation documents; a further 32 768 OOD documents are generated from 16 fresh topics, drawn independently of $P$ from the same prior. Each point of a learning curve averages four runs with independent topic matrices, test ID and OOD corpora and initialisation, and error bars are standard errors over the four. Documents carry $T { + 1 }$ tokens: the model reads the first $T$ and is trained to predict tokens 2 to $T { + 1 }$ , so that its loss profile covers context lengths 1 to $T$ , the $T \cdot$ -gram loss is the loss at context length $T ,$ , and the autoregressive loss is the average of the profile over the T positions.

Model. A one-block decoder-only transformer: token embeddings of dimension $d _ { \mathrm { e m b } }$ with no positional embedding, so that the causal mask is the only source of order, as befits documents that are exchangeable given their topic; one causal self-attention head; a ReLU multilayer perceptron with one hidden layer of width $4 d _ { \mathrm { e m b } } ;$ both sub-layers pre-normalised and residual; a final layer normalisation and an untied linear readout to the K logits. Attention and perceptron carry no biases and there is no dropout. The model has $1 2 d _ { \mathrm { e m b } } ^ { 2 } + 2 K \mathsf { \bar { d } } _ { \mathrm { e m b } } + O ( d _ { \mathrm { e m b } } )$ parameters, from $5 \times 1 0 ^ { 4 }$ at $( K , d _ { \mathrm { e m b } } ) = ( 3 2 , \bar { 6 4 } )$ to $3 \times 1 0 ^ { 6 }$ at (128, 512). Weights are initialised as standard normals and each linear map is scaled by the inverse square root of its input dimension, the readout by $1 / d _ { \mathrm { e m b } }$

Training. The loss is the cross-entropy of the next token, averaged over the $T$ positions and the documents of a batch. Batches of b documents are taken cyclically from the corpus and the parameters are updated with Adam, moment parameters (0.9, 0.999) and no weight decay, at a learning rate η raised linearly from zero over the first 8 steps and constant thereafter, for a fixed number of passes over the corpus.

Evaluation and early stopping. The validation autoregressive loss is measured at log-spaced steps throughout training. The checkpoint of lowest validation loss defines early stopping, and its loss is the value that enters the ID learning curves $( \mathrm { F i g . ~ 1 , F i g . ~ 3 , F i g . ~ 9 } )$ . At a coarser log-spaced subset of the steps the full loss profiles on the validation and on the OOD documents are recorded as well. The $T \cdot$ -gram ID loss of a run (Fig. 10) is the minimum over these recorded checkpoints of the loss at context length T; the OOD losses are read either at the recorded checkpoint of lowest validation loss, “at early stopping” (Fig. 1, Fig. 4), or minimised over the recorded checkpoints, “best over training” (Fig. 1, Fig. 5), as stated in each caption.

Hyperparameter selection. The embedding dimension, batch size, learning rate and training budget are selected separately for every $( K , D , \bar { T } , N )$ on the validation loss at early stopping, so that each learning curve reports the best model of the family rather than a fixed architecture, and the model is not the bottleneck of the comparison with the Bayes-optimal learning curves. The grid is $d _ { \mathrm { e m b } }$ from K to 512 in powers of two, $\dot { b ^ { \prime } } \in \{ 1 6 , 3 2 , 6 4 , 1 2 8 \}$ and $\dot { \eta } \in \{ 1 0 ^ { - 2 } , 3 \times 1 0 ^ { - 2 } , 1 0 ^ { - 1 } \}$ ; larger dictionaries require larger $d _ { \mathrm { e m b } }$ , and the selected width grows with D at fixed $K .$ . The number of passes is extended until the early-stopping checkpoint falls well inside the budget, or until a doubling of the budget improves the validation loss by less than the spread over seeds. A candidate combination replaces the incumbent only when its early-stopping validation loss, over four fresh seeds, is lower beyond the standard errors, or tied with a clearly earlier early-stopping step. The combination selected on the ID validation loss serves for every quantity reported at that $( K , D , T , N )$ , including the OOD losses, which play no role in the selection.

## H ADDITIONAL FIGURES

![](images/c0f46cbc8c2de5e5512f17ba9bd6cd981abe6352b27b33ee5078f03313d5e9c9.jpg)

![](images/d8f8faa0ce7b6cf36a8162427024cd7c8f5fb1a1ac6043de7d4b18fba3abfc79.jpg)  
Figure 9: Learning curves at different document lengths collapse on the number of tokens1 per parameter. Left: Scaled ID autoregressive loss $( \mathcal { L } _ { \mathrm { A R } } ( N ) - \mathcal { \bar { L } } _ { \mathrm { A R } } ( \infty ) ) / ( \log K - \mathcal { L } _ { \mathrm { A R } } ( \infty ) )$ versus number of training sequences for one-block transformers trained on random topic model data with $K = 1 2 8 , D = 1 6$ and $\bar { T } = 1 2 8 , 2 5 6 , 5 1 2$ (colour). $\mathcal { L } _ { \mathrm { A R } } ( \infty )$ is the autoregressive loss of the specialised predictor at each T, averaged over 32 realisations of the topic matrix. Right: Against $\bar { N } T / ( D K )$ the three curves coincide: at fixed N, longer documents bring the loss closer to its floor in proportion to the number of tokens they contribute.

![](images/1f3732a2711b3e510f429bd3b88487dbdddab1120f232615be22749aba34a9bf.jpg)

![](images/ce253f0fed2bb38b5f563e808d183093f02c30d836c6e2040b9566f360bf937f.jpg)  
NT /(KD)

![](images/b20f0cf46c17ec7f752d5d8090779a62e4b11bffe71efb471546f2d5544761a4.jpg)  
<sup>1</sup>Figure 10: Transformers match the Bayes-optimal specialisation learning curve at the longest context too. T-gram counterpart of Fig. 3, for the same twelve $( K , D )$ at $T = 1 2 8$ . Left: Scaled ID T-gram loss $( \bar { \mathcal { L } _ { T } } ( N ) - \bar { \mathcal { L } _ { T } } ( \infty ) ) / ( \mathrm { l o \bar { g } } K - \bar { \mathcal { L } _ { T } } ( \infty ) )$ versus number of training sequences, where $\mathcal { L } _ { T } ( \infty )$ is the mean entropy $\begin{array} { r } { \frac { 1 } { D } \sum _ { z } H ( \mathbf { p } _ { z } ) } \end{array}$ of the topics of each run, the large-context floor of the specialised predictor. Middle: Rescaling the x-axis as the number of tokens per parameter of the data-generating process collapses all the curves onto a single functional form, more tightly than for the autoregressive loss. Right: The decay towards 0 (black dashed) follows the $x ^ { - 1 }$ rate of the Bayes-optimal learning curve.

![](images/880953d84668e01a3fe35691a4cf11f6cd629bb80207d4e9dd755c57f722fb70.jpg)

![](images/d1d06d51cd4d57a9d3ecd4d8c5ab75625fa32eba4eb5126963639ffc5690fce7.jpg)  
Figure 11: The ID loss descends to the in-context level at the predicted sample scale. Left:1 ID autoregressive loss at early stopping, scaled as $( \mathcal { L } _ { \mathrm { A R } } ( N ) - \mathcal { L } _ { \mathrm { A R } } ^ { \star } [ \mathbf { r } ] ) / ( \log K - \mathcal { L } _ { \mathrm { A R } } ^ { \star } [ \mathbf { r } ] )$ , where $\mathcal { L } _ { \mathrm { A R } } ^ { \star } [ \mathbf { r } ] = \log K - G ( K , T )$ is the autoregressive loss of the in-context predictor, the same on ID and OOD documents (App. C). This scaled loss is 1 at the uniform prediction, 0 at the in-context solution, and negative at specialisation. Versus number of training tokens NT, for the $( K , D )$ and T of Fig. 5, with the same line styles. Right: When plotted against $N T G ( K , T ) / K ^ { 2 } { = } \dot { N } / ( \overset { . } { 2 } N _ { \mathrm { i c l } } )$ (Eq. 13) the descents to the in-context level coincide for all K, D and $T _ { \ast }$ , with no adjustable constant. The subsequent departure below zero is specialisation.