# Rethinking GA<sup>2</sup>Ms: performance and out-of-distribution generalisation

George Dimitriadis\*, Damian Machlanski, Paul Clarke, Spyridon Samothrakis\*

Abstract—Generalised Additive Models (GAMs) are a cornerstone of interpretable machine learning, offering transparent predictions through structured additive decompositions. Despite advances from classical GAMs to modern neural and splinebased variants, two notable gaps remain. First, comprehensive benchmarking of spline-based GAMs on large-scale tabular learning tasks—the dominant modality in applied settings—is lacking. Second, their robustness under distribution shift, a setting where structured inductive biases should confer advantages, remains unexplored. We propose a multi-resolution, higherorder Composite Generalised Additive Model (CoGA<sup>2</sup>M) and evaluate it on the TALENT benchmark suite. CoGA²M ranks in the top third across all baselines (29 for regression, 32 for classification), establishing competitive performance with stateof-the-art tabular methods. We further extend TALENT with outof-distribution (OOD) splits exhibiting train-test distributional shift and benchmark $\dot { C } o G A ^ { 2 } M$ against LightGBM, a consistently top-performing model on TALENT. Under distribution shift, $C o \dot { G } \dot { A } ^ { 2 } M$ outperforms LightGBM, demonstrating a clear structural advantage. Our results position spline-based GAMs as strong contenders for tabular prediction and reveal a previously unrecognised benefit of additive structure in the OOD regime.

Index Terms—Generalised Additive Models, Out-ofdistribution, tabular data

## I. INTRODUCTION

Generalised Additive Models (GAMs), first proposed by Hastie and Tibshirani [1], have long occupied a distinctive niche in machine learning: they offer interpretable, structured predictions by modelling the response as a sum of smooth functions of individual features. Over the past three decades, a rich family of extensions has emerged, ranging from penalised spline methods [2], [3] to higher-order interaction models $( G A ^ { 2 } M s )$ [4], constrained variants [5], boosting-based approaches such as Explainable Boosting Machines [5], and neural network–based formulations [6]–[9]. A unifying theme across this line of work is the emphasis on explainability, and recent research has demonstrated that interpretability need not come at the cost of predictive performance [9]. This finding is encouraging, yet the empirical evaluations in the GAM literature have largely been confined to standard in-distribution (ID) settings.

The study of out-of-distribution (OOD) generalisation has received considerable attention in other areas of machine learning, with theoretical foundations in domain adaptation and covariate shift [10], and a range of algorithmic families proposed to improve robustness to distribution shift [11]–[15]. On the evaluation side, substantial benchmarking efforts have been directed at image [16], [17], text [18], [19] and general multi-modal settings [20]. However, the vast majority of this research — both methodological and empirical — has concentrated on image and text domains. The tabular data setting, which remains the most prevalent data modality in practice, has received comparatively little attention. The TableShift benchmark [21] represents a notable recent effort, but even it focuses primarily on comparing deep learning architectures and generic robustness methods, rather than examining how the inductive biases of specific model families — such as the smoothness and additivity priors of GAMs — affect OOD behaviour. To our knowledge, there has been no systematic effort to examine the behaviour of GAMs and their variants under OOD conditions — a critical gap given that real-world deployment routinely exposes models to data that diverge from the training distribution. This is a critical obstacle to safe and reliable deployment in high-stakes environments [22], [23].

This gap motivates the present work. The smooth, structured nature of spline-based models suggests they may extrapolate more gracefully than flexible black-box alternatives when faced with diversity shifts — regions of input space unseen during training where the underlying data-generating process remains unchanged. Yet this hypothesis has not been tested empirically. Relatedly, while GA<sup>2</sup>M-family models have been shown to be competitive on standard benchmarks, no existing study places them on an equal footing with state-of-the-art methods across a large-scale tabular evaluation while simultaneously probing their OOD behaviour.

To address both gaps, we make two contributions. First, we introduce a multi-resolution, higher-order, P-Spline, Composite Generalised Additive Model $( C o G A ^ { 2 } M )$ that generates splines at multiple resolutions and arities, combining linear and cubic Hermite basis functions into a single differentiable layer whose parameters — including the regularisation strength — are learned end-to-end via backpropagation. We evaluate $C o G A ^ { 2 } M$ on the full TALENT benchmark [24] comprising 300 tabular datasets, demonstrating that it ranks in the top third of all 30+ models across regression, binary classification, and multi-class classification tasks, with performance comparable to leading methods such as modernNCA, LightGBM, XGBoost, realmlp, tabr and CatBoost. Second, we develop an adversarial OOD dataset generation procedure that carves compact, connected regions out of a dataset’s interior to create diversity-shifted train–test splits. Using an evolutionary strategy (CMA-ES) [25] in an adversarial loop with learner models, we produce progressively harder OOD splits for 40 regression datasets. We then compare $C o G A ^ { 2 } M$ against LightGBM on these OOD splits, finding that $C o G A ^ { 2 } M$ exhibits a statistically significant smaller drop in performance when moving from ID to OOD data (bootstrap $\mathrm { ~ p ~ } = \mathrm { ~ 0 . 0 0 6 ) ~ }$ , suggesting that the smooth, spline-based inductive bias of $G A ^ { 2 } M$ -family models confers a meaningful advantage in diversity-shift OOD scenarios. In this study we focus exclusively on diversity shift.

The remainder of the paper is organised as follows. Section II reviews related work on GAM-family models and OOD generalisation. Section III describes our methods, including the $C o G A ^ { 2 } M$ architecture, the adversarial OOD dataset generation procedure, and the experimental setup. Section IV presents results on both the TALENT benchmark and the OOD comparison. Section V concludes.

## II. BACKGROUND

## A. Generalised Additive Models and their extensions

Generalised Additive Models were introduced by Hastie and Tibshirani [1] with a local scoring algorithm to automatically determine the appropriate number of knots. Penalised Bsplines (P-Splines) [2] introduced efficient roughness penalties for controlling smoothness, while penalised thin plate regression splines with shrinkage (TPSplines) [3] offered a more flexible alternative that avoids manual knot placement. The move to higher-order interactions gave rise to $G A ^ { 2 } M \mathrm { s }$ [4], which augment the additive structure with pairwise feature terms, substantially increasing expressiveness while preserving a degree of interpretability. $G A ^ { 2 } M \mathrm { s }$ introduced the FAST measure to select the most appropriate feature combinations. Further developments include $C G A ^ { 2 } M { + } \mathbf { s }$ [5], which enforce monotonicity constraints on selected features, and Explainable Boosting Machines (EBMs) [5], which use a cyclic gradient boosting procedure to fit GAM-style models at scale. More recently, neural network–based variants have gained traction. Neural Additive Models (NAMs) [6] replace traditional splines with learned neural sub-networks, GAMI-Net [7] introduces structured interaction selection into neural GAMs, Explainable Neural Networks (ExNNs) [8] impose architectural constraints for transparency and interpretability, and Interpretable Generalised Additive Neural Networks (IGANNs) [9] combine neural flexibility with additive structure. Kruschel et al. [9] present an initial evaluation of GAM based models showing that interpretable machine learning models can challenge the conventional performance–interpretability trade-off. A 20 dataset (10 binary classification, 10 regression) benchmark was constructed and used to compare 7 GAM based models, two classical, interpretable models (linear/logistic regression and decision-trees) and 5 black-box models (random forests,

XGBoost [26], CatBoost [27], Multi Layer Perceptron (MLP) and TabNet [28]). The GAM based approached achieved competitive accuracy relative to the black-box alternatives while showing better interpretability, as defined by 6 measures (additivity, sparsity, linearity, smoothness, monotonicity and visualizability).

## B. Out-of-distribution generalisation

The theoretical foundations of OOD generalisation were laid by work on domain adaptation and covariate shift [10], which formalised how discrepancies between training and test distributions affect generalisation bounds. Building on these ideas, several algorithmic families have been proposed to improve robustness to distribution shift. Invariant Risk Minimisation (IRM) [11] seeks to learn feature representations that support an optimal predictor across data drawn from any possible distribution (usually known as training domains, here referred to as environments) which, importantly, supports an underlying, invariant across environments, causal mecha nism. Those include environments not seen during learning. Given training data from an adequate number of environments the method encourages invariance to spurious correlation and optimality in OOD scenarios. Distributionally Robust Optimisation (DRO) [12] takes a complementary approach, minimising worst-case risk over a set of predefined groups or perturbation sets. Again, this method targets datasets where spurious correlations exist in small sub-sets which although averaged over in the training regime can become prominent in the test data. The method relies on heavy regularisation techniques to allow the model to focus its risk minimization on the worse cases of such sub-groups. Risk Extrapolation (REx) [13] extends DRO by extrapolating beyond the convex hull of training risks to achieve robustness against more severe shifts. The concept here is that if a method achieves both an almost minimum risk (kept equal across several domains) then it will be able to extrapolate to unseen and more severe distribution shifts far away from the ones represented by the training do mains. Domain-adversarial methods [14] learn representations that are indistinguishable across domains. Data augmentation strategies such as Mixup [15] and its domain-generalisation variants create virtual data points that are linear interpolations between real points are designed to improve generalisation, model calibration and robustness against adversarial examples. Yet, and despite these methods not targeting OOD data or dealing with multiple domains they also deal better with OOD datasets, especially distribution shifts where the test data are drawn from a separate part of the distribution compared to the train set, simply because by accessing virtual data the model is forced to learn something about regions where real data do not exist. A more extensive account of the current state of the OOD literature can be found in a recent survey [29].

## C. OOD benchmarks and evaluation

On the evaluation side, substantial efforts have been directed at creating standardised benchmarks to assess OOD methods. DomainBed [16] established a unified test-bed for domain generalisation in computer vision, encompassing seven multidomain image datasets and multiple algorithms — and notably demonstrated that carefully implemented Empirical Risk Minimisation (ERM) can match or exceed many specialised OOD methods. WILDS [17] broadened the scope to real-world distribution shifts across diverse data modalities including medical imaging, satellite imagery, and natural language. The OoDBench+ framework [20] proposed a particularly useful taxonomy, distinguishing between two principal types of domain shift: diversity shift, where the test data is drawn from a region of the input space unseen during training while the underlying data-generating process remains unchanged, and correlation shift, where spurious correlations present in the training data are altered or reversed at test time (see Fig. 1 for a visual representation of these ideas). Under a causal framework diversity shift is a generalisation of covariate shift (where the distribution of X changes but not the distribution of the labels given X). Correlation shift on the other hand can be considered as a latent shift [30] since the spurious correlations can be attributed to a correlation of some driving latent variable with both some features and the labels. In the natural language processing domain, benchmarks such as BOSS [18] and GLUE-X [19], have further advanced the field. BOSS leverages Simple Contrastive Sentence Embeddings (SimCSE) to calculate semantic similarity between phrases and is in this way able to evaluate OOD distances between language datasets. It then utilises this metric to construct maximally challenging train–test splits. GLUE-X evaluates language understanding models on a series of language benchmarks chosen for their OOD focus.

## D. Tabular data under distribution shift

Despite this progress, the vast majority of OOD research — both methodological and empirical — has concentrated on image and text domains. The tabular data setting, which remains the most prevalent data modality in practice, has received comparatively little attention. The TableShift benchmark [21] represents a notable recent effort, providing 15 curated binary classification tasks under real-world distribution shifts and revealing that the robustness of tabular models to distribution shift is poorly understood. Yet even TableShift focuses primarily on comparing deep learning architectures and generic robustness methods, rather than examining how the inductive biases of specific model families — such as the smoothness and additivity priors of GAMs — affect OOD behaviour. More broadly, relatively few efforts have focused on rigorously evaluating established tabular algorithms under controlled distribution shifts, and the question of whether interpretable models generalise differently from black-box ones in OOD settings remains largely unanswered.

## III. METHODS

## A. Multi-resolution composite P-Splines

We develop an extension to the P-Splines algorithm [2] in which for each feature, we generate splines with different number of knots (spline resolution) and with different arity (number of features as input). For each resolution and arity

![](images/1b52f09e1cae9ddbaba3515120d142bd386082c58938a58a7ff7628b46c4dc47.jpg)

![](images/e8389f539f1fe254d5c99e4a452466ffd2508c5dfc96bf2b705db60b758a7bb7.jpg)

(a) In-distribution  
![](images/aa952b4450be81c87d0a6eadaed1d8554b9d478439f025ec2ba5a99a81a594a6.jpg)

![](images/88bbe9620d463bb7a8ea2918708fa1c0a5bf27333992ec84d858c4040bd91940.jpg)

(b) Out-of-distribution diversity shift  
![](images/6b29b3c74836f620ae6882776f81062f9e60c0a219fa1251389811928e6f1907.jpg)  
(c) Out-of-distribution correlation shift

Fig. 1: Schematic diagram of in-distribution (a), out-ofdistribution with diversity shift (b) and out-of distribution with correlation shift (c) tabular datasets. In all three sub-figures the data are 2 dimensional (2 features) and the colour indicates the Y values of the examples. The white regions indicate missing data. The left plots show the training data and the right ones the test sets. In (a) and (b) the data are generated by the function $Y = c o s ( X _ { 1 } ) * c o s ( X _ { 2 } )$ . In (a) the test data cover small patches of the whole distribution and are uniformly distributed. The size of the missing patches is not larger than the characteristic length of the latent features of the dataset, the frequencies of the two cosines. In (b) on the other hand the train set’s missing data form a simply connected region whose size is of the same order as the characteristic length of the latent feature. The test data are drawn from a distribution that is diverse (diversity shifted) from the one generating the train set. In (c) the underlying causal structure is $Y = c o s ( X )$ . In the left plot feature C correlates with ${ \textrm { Y } } ( C = Y )$ as only data with that correlation are shown to the model (insert shows data from full range with no C, Y correlation). In the right plot (test set) that correlation is broken since for each value of C there are multiple examples with $Y \neq C .$ . To make the point clear, the green rectangle shows the Y values for a specific value of C. In the train set all Y values are the same (equal to C) while in the test set Y takes a number of values uncorrelated with C. In this case the model will focus on the spurious correlation and will fail to learn the underlying structure. The schematic makes clear the hard to intuit nature of tabular correlation shifts in comparison with other types of data (e.g. images).

we create both a linear and a cubic spline. The results of all the splines acting on their appropriate feature values are then concatenated into the final result of the Splines’ layer. This output is then passed into a Linear layer which combines the Spline’s layer output into either a single regression result or a set of probabilities for classification problems. Both the parameters θ that parametrise each spline and the weights w for the final linear composition are learned.

For arity bigger than one, the generated splines are the product of the equivalent univariate splines for each feature. So for arity two we generate a linear and a cubic spline for each one of the feature pairs as the product of the linear and cubic splines of each individual feature. For arity three, that product would arise from the three univariate splines, and so on. In this work we use up to arity two. We have found that arity one does not do as well in all of the datasets we tried. Arity three and above models are both prohibitively expensive and from initial experiments did not seem to provide any significant improvements.

1) Spline basis functions: For input $x \in [ - 1 , 1 ]$ mapped to a scaled position $\tilde { x } \in [ 0 , K - 1 ]$ where K is the number of knots: $\tilde { x } = c l i p ( \textstyle { \frac { x + 1 } { 2 } } , 0 , 0 . 9 9 9 9 ) \dot { * } ( K - 1 )$

$j = \lfloor \tilde { x } \rfloor$ and $t = \tilde { x } - j$ . The linear spline is:

$f _ { l i n } * ( x ) = ( 1 - t ) \theta _ { j } * + t \theta _ { j + 1 }$ where $\theta _ { j }$ are the learnable knot values. Using the same scaled position x˜, index j and fraction t, we compute the Catmull-Rom style tangents:

$$
\begin{array} { r } { m _ { j } = \left\{ \begin{array} { l l } { \theta _ { j + 1 } - \theta _ { j } } & { \mathrm { i f ~ } j = 0 } \\ { \frac { \theta _ { j + 1 } - \theta _ { j - 1 } } { 2 } } & { \mathrm { o t h e r w i s e } } \end{array} \right. } \end{array}
$$

$$
m _ { j + 1 } = { \left\{ \begin{array} { l l } { \theta _ { j + 1 } - \theta _ { j } } & { { \mathrm { i f ~ } } j + 1 = K - 1 } \\ { { \frac { \theta _ { j + 2 } - \theta _ { j } } { 2 } } } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }
$$

Using those we compute the cubic Hermite basis polynomials:

$$
\begin{array} { r } { f _ { c b } ( x ) = ( 2 t ^ { 3 } - 3 t ^ { 2 } + 1 ) \theta _ { j } + ( t ^ { 3 } - 2 t ^ { 2 } + t ) m _ { j } + } \\ { ( - 2 t ^ { 3 } + 3 t ^ { 2 } ) \theta _ { j + 1 } + ( t ^ { 3 } - t ^ { 2 } ) m _ { j + 1 } } \end{array}
$$

2) Composite layer output: For input vector $\pmb { x } \in \mathbb { R } ^ { N }$ the spline layer output is the concatenation of: Univariate terms (for each feature i and resolution r):

$$
\begin{array} { c } { { \phi _ { i } ^ { r a w } = w _ { i } ^ { r a w } * x _ { i } } } \\ { { \phi _ { i , r } ^ { l i n } = w _ { i , r } ^ { l i n } * f _ { i , r } ^ { l i n } * ( x _ { i } ) } } \\ { { \phi _ { i , r } ^ { c b } = w _ { i , r } ^ { c b } * f _ { i , r } ^ { c b } * ( x _ { i } ) } } \end{array}
$$

Bivariate terms (for each pair $i < j$ and resolution r):

$$
\begin{array} { c } { { \psi _ { i , j } ^ { r a w } = w _ { i , j } ^ { r a w } * x _ { i } * x _ { j } } } \\ { { \ } } \\ { { \psi _ { i , j , r } ^ { l i n } = w _ { i , j , r } ^ { l i n } * f _ { i , r } ^ { l i n } * \left( x _ { i } \right) * f _ { j , r } ^ { l i n } * \left( x _ { j } \right) } } \\ { { \ } } \\ { { \psi _ { i , j , r } ^ { c b } = w _ { i , j , r } ^ { c b } * f _ { i , r } ^ { c b } * \left( x _ { i } \right) * f _ { j , r } ^ { c b } * \left( x _ { j } \right) } } \end{array}
$$

All ϕ and $\psi$ terms are concatenated together for the layer’s output:

$$
S = \bigoplus _ { i = 1 , r = 1 } ^ { N , R } [ \phi _ { i } ^ { r a w } \Vert \phi _ { i , r } ^ { l i n } \Vert \phi _ { i , r } ^ { c b } \Vert ( \bigoplus _ { i < j } ^ { N } [ \psi _ { i , j } ^ { r a w } \Vert \psi _ { i , j , r } ^ { l i n } \Vert \psi _ { i , j , r } ^ { c b } ] ) ]
$$

where $\oplus$ and ∥ denote concatenation. For a visualisation of the spline composition method see Figure 2.

3) Loss function and regularisation: The Spline layer adds to the loss of the learning algorithms as:

$$
\mathcal { L } _ { m r c p s } = \mathcal { L } _ { L 1 } + \mathcal { L } _ { s m o o t h } .
$$

$\mathcal { L } _ { L 1 }$ is an L1 sparsity penalty:

$$
\mathcal { L } _ { L 1 } = \lambda _ { 1 } ( \sum _ { r , i , t y p e } \| \theta _ { i , r } ^ { t y p e } \| _ { 1 } + \sum _ { r , i , t y p e } \| w _ { i , r } ^ { t y p e } \| _ { 1 } )
$$

where type can be raw, lin or cb. $\mathcal { L } _ { s m o o t h }$ is the smoothness (roughness) penalty:

$$
\mathcal { L } _ { s m o o t h } = \frac { 1 } { | s | + \epsilon } \sum _ { i , r , t y p e } \sum _ { k = 0 } ^ { K _ { r } - 3 } ( \theta _ { k + 2 } - 2 \theta _ { k + 1 } + \theta _ { k } ) ^ { 2 }
$$

following the P-Spline formulation [2]. In our case the model learns the regularisation parameter $\lambda _ { 1 }$ through backpropagation instead of choosing it via Generalised Cross Validation (GCV) while we fix the $\frac { 1 } { | s | + \epsilon } \ t _ { 0 } \ 0 .$

4) Extracting splinesfrom the trained model: After a model is trained the splines it uses for inference are parametrised through the θs and ws. In order to explore what a model has learned and how the splines it generates can provide interpretable results, we can combine the appropriate θs and ws to reconstruct and visualise the learned splines. We reconstruct the full set of splines the model generates, as described above. These by themselves though, would not be a very useful visualisation and interpretability tool. We also linearly combine all the resolutions for all spline types (linear and cubic) for each feature separately. This way we construct N splines, each the linear combination of six univariate resolutions - three linear and three cubic plus the raw spline $\phi ^ { r a w } ( x _ { i } ) = w _ { i } ^ { r a w } x _ { i }$ for feature $x _ { i } .$ For the arity 2 spline we construct N-1 bivariate splines each combining the feature with each one of all the other features minus itself $( x _ { i } , x _ { j }$ for $i \neq j )$ . Again for the bivariate case each spline is the linear composition of the corresponding bivariate resolutions for the raw, linear and cubic types (see Fig. 2 for a full description of how spline components are combined).

## B. Full network structure and training

1) Input: Our model is designed to deal with tabular datasets where each feature can be either numerical or categorical (treating ordinal as numerical). Numerical inputs are scaled between -1 and 1 before being passed on to the Spline layer. Since our model is embedded in the TALENT benchmark [24] we deal with categorical data using TALENT’s infrastructure. Specifically TALENT by default uses ordinal encoding, assigning each category a number ranging from -1 to 1 so that all categories are equally spread over this range. This is not optimal and there are better ways to deal with categorical data (like appropriate embeddings) but we wanted to be able to directly compare our results with the ones provided by TALENT for all other models so we did not modify the default categorical data transformation.

![](images/a2c8ec3d0b052672cf13c4551f1e3e5a385b87121e6f2e43004a0a6b4d2620a3.jpg)  
(a) Break down of spline generation from sub-splines  
Fig. 2: Visualisation of the spline composition. (a) For each feature $x _ { i }$ and each of the three resolutions we use in this work (4, 16 and 64 knots) we compute the univariate and bivariate linear splines as well as the cubic Hermite ones. We also compute $f _ { i } ^ { r a w } ( x _ { i } ) = x _ { i }$ . For each input $x _ { i }$ and each resolution r the bivariate splines (linear and cubic) are the product $f _ { i } ( x _ { i } ) * f _ { j } ( x _ { j } )$ $\forall j \ \mathrm { s . t . } \ i < j . ( \mathrm { a } )$ The vector that is the result of the concatenation of all the generated splines acting on their individual features is then linearly combined to produce the layer’s output.

2) Output: The tabular data we deal with can be either a regression problem (continuous output), a binary classification problem (binary output) or a multi classification one (categorical output with more than two categories). In the regression case the continuous output of our Spline layer is passed to a Linear layer with a single output and in the two classification cases the Spline’s output is passed to a Linear layer with number of output neurons the number of classes. Their output is then treated as probabilities and passed to a Softmax layer to pick the highest probability class. In the case of regression, once the data pass through the Spline Layer and the Linear Layer for N epochs and their learning finishes then the output of the Spline Layer is passed through a Ridge Cross Validation (RidgeCV) Layer which selects the best regularization parameter for the Linear Layer’s weights.

## C. TALENT datasets comparisons

In order to compare our algorithm with the ones offered by the TALENT benchmark we created two metrics, a ranking order and a score measuring proportional difference to the

MLP algorithm. The score we used for both metrics was either the $R ^ { 2 }$ for regression datasets or the Accuracy (Acc) for multi or binary classification ones. We run our method, $C o G A ^ { 2 } M$ with 15 (TALENT default) repetitions. The results over the repetitions were in all datasets diverging to less than 0.1% (data not shown). We did no hyperparameter search and run the model with a batch size of 16, a learning rate of 0.001 and with arity 2. We set it to run a maximum of 5000 epochs but with early stopping if the validation score (validation set was 10% of the data) stopped getting better for at least 100 epochs.

For ranking we used the Placket-Luse model [31]. The difference to the MLP results were calculated as relative difference between scores:

$$
\frac { s c o r e _ { C o G A ^ { 2 } M } - s c o r e _ { M L P } } { s c o r e _ { M L P } }
$$

For the regression we used the $R ^ { 2 }$ as the score while for the classifications the model’s accuracy.

## D. Generation of OOD datasets

We use only a subset of the 300 datasets in the TALENT benchmark to generate OOD datasets. The first selection strategy was to use only regression datasets that had only numerical inputs. Using only numerical inputs datasets both alleviated the issue of how exactly to transform the categorical data for our $C o G A ^ { 2 } M$ algorithm and allowed us to create an OOD data generation process that did not have to deal with categorical data. That resulted in a subset of 48 datasets. The second selection had to do with the time each of our OOD generation processes (we used two) took to break up the data. For the first process we managed to create 31 OOD datasets while with the second 39. Those were partially overlapping (29 in common) subsets of the original 48 datasets. The rest of the datasets were too big (in number of samples and features) to be processed within a reasonable time frame.

We focused on creating OOD datasets with diversity shifts. In order to achieve this we needed to generate new datasets with data from a compact, connected subset of the interior (a region) of the dataset’s convex hull assigned to the test data while the rest (100% of the region’s complement) is assigned to training (90%) and validation data (10%). But this region needed to be a feasible region in the sense that the data in it and the data in its complement needed to be drawn from different distributions. Since we could not directly infer the distributions of the newly generated train and test sets we used as a proxy for OOD the drop in the ability of a model to predict the OOD test data when trained on the OOD train set compared to its ability to predict the TALENT ID test data when trained on the ID train set.

In order to create new data cuts where a model would fail to learn the test data we created two versions of an adversarial algorithm. Each had an evolutionary process optimising the feasible region to make the test data in it as hard to predict from the train data in the complement of the feasible region. At the same time, a model learned the newly cut train data trying to maximize its result on the new test data. The two versions of the adversarial algorithm differed by the model learning on the train data. In the first case we used a K Nearest Neighbours Regressor (kNNR) with a hyperparameter search over its distance metric (Manhattan and Euclidean), its weights distribution (uniform and distance) and its k (1, 3, 5, 7, 9 and 11). In the second case we used ν Support Vector machine Regression (νSVR) with $\nu = 5$

For both versions, the OOD generating process was the Covariance Matrix Adaptation (CMA) evolutionary strategy [25] with a population size of 20. For both the kNNR and the νSVR models and for each run of the algorithm we followed the decreasing $R ^ { 2 }$ of the models’ results on the CMA updating train, validation and test datasets. For every drop in the $R ^ { \overset { \mathbf { \sim } } { 2 } }$ we saved that train/val/test data cut. We asked the algorithms to stop at either after 20 such cuts or at 300 epochs (whichever came first). That produced a dataset specific number of different cuts, each one progressively harder to be learned by the kNNR or νSVR models. We run each algorithm on each dataset 5 times concatenating the cuts over all runs and sorting them according to descending $R ^ { 2 }$

## E. OOD datasets comparisons

Once we had OOD cut data for as many datasets as possible from the kNNR (31 datasets) and the νSVR (39 datasets) based processes we compiled a list of datasets with their cuts by choosing the results from one of the two cutting algorithms for each dataset. The selection process was:

1) If an algorithm had not produced an $R ^ { 2 }$ in the last (smallest $R ^ { 2 } )$ cut smaller than 0 then do not consider this algorithm/dataset combination a valid series of cuts.

2) If only one of the two algorithms had produced a valid cuts series, use this.

3) If both algorithms had produced valid cuts then use the one from the kNNR unless

4) the kNNR’s maximum $R ^ { 2 }$ (from the first cut of the series) was smaller than 0 and the νSVR’s first cut $R ^ { 2 }$ was larger than 0, in which case use the νSVR.

The logic behind the 3rd rule was to keep as many of the kNNR cuts as possible since we deemed this to be the algorithm that would bias the follow up comparison between $C o G A ^ { 2 } M$ and LightGBM the least. The logic behind the 4th rule was that if the kNNR algorithm at its first cut (before the evolution algorithm had time to select appropriate regions) had failed to learn the data then it was not providing the CMA a good enough signal for appropriate region selection and then the νSVR algorithm was to be preferred. The final list consisted of 40 datasets (out of the 48 initial ones), 28 were cut with the kNNR and 12 with the νSVR algorithm.

For each dataset and the selected cutting model we grouped the sorted series of cuts into 3 sets, each one having cuts with $R ^ { 2 }$ in the top, middle and bottom quantiles of the $R ^ { 2 }$ range. For each (ID) dataset we defined its OOD dataset as the group of all the cuts of the bottom quantile (the ones with the worst $R ^ { 2 } { \bf s } )$ . The number of such cuts for each dataset finally used for our comparisons is shown in Table I in the column N.

We run the $C o G A ^ { 2 } M$ and the LightGBM models (to compare with a model that does really well on the regression part of the benchmark) on both the ID and the OOD datasets. We then used the generated $R ^ { 2 }$ to exclude some datasets from the final comparison. We would exclude a dataset using the following rules:

1) If both the LightGBM and $C o G A ^ { 2 } M$ algorithms did not achieve an $R ^ { 2 }$ larger than 0 in the OOD data. In this case both algorithms were considered to fail to learn the OOD data. There were 13 datasets like that.

2) If both algorithms achieved an $R ^ { 2 }$ larger than 0.9 in the OOD data. In this case both algorithms are considered to succeed in the OOD data or (most probably) our OOD generation method did not manage to break the dataset appropriately to create an OOD test set. There were 3 datasets in this group.

3) If both algorithms fail to achieve an $R ^ { 2 }$ larger than 0.5 in the ID data. In this case neither algorithm can learn the ID dataset, so there is no point in comparing how well they managed to learn the OOD one. There are 6 datasets in this group 2 of which also exist in the $r _ { O O D } ^ { 2 } < 0$ set.

This left 22 datasets on which we compared $C o G A ^ { 2 } M$ and LightGBM. For each dataset we first calculated the differences between the $R ^ { 2 } { \mathrm { s } }$ of the two algorithms on the ID and the OOD data

$$
\Delta _ { C o G { A ^ { 2 } M } } = r _ { C o G { A ^ { 2 } M } , O O D } ^ { 2 } - r _ { C o G { A ^ { 2 } M } , I D } ^ { 2 }
$$

and

$$
\Delta _ { L i g h t G B M } = r _ { L i g h t G B M , O O D } ^ { 2 } - r _ { L i g h t G B M , I D } ^ { 2 }
$$

and finally the difference of the differences

$$
\Delta \Delta = \Delta _ { C o G A ^ { 2 } M } - \Delta _ { L i g h t G B M }
$$

We then tested whether the $\Delta \Delta _ { d a t a s e t }$ series across the 22 datasets is significantly larger than 0. We did this using both a t-test (one-tailed) and bootstrapping with 10000 draws (in order to mitigate the small number of samples).

## IV. RESULTS

## A. Spline visualisation and interpretability

GAMs and $G A ^ { 2 } M s$ were designed to provide interpretable results. As the models evolved in complexity, significant effort was aimed into keeping the resulting splines as interpretable as possible. From the initial $G A M$ idea [1] with its local scoring algorithm to automatically calculate the smallest appropriate number of knots, to the Penalised-Splines $G A M s$ [2], for more efficient knot calculation, to the higher order arity $G A ^ { 2 } M s$ [4] with the FAST measure to pick the most appropriate feature combinations and the $C G A ^ { 2 } M { + } \mathbf { s }$ [5] enforcing monotonicity on certain features, interpretability has been at the core of GAM development. Here we show that our multiple resolution, P-Spline based $G A ^ { 2 } M$ without monotonicity or any separate (outside the backpropagation algorithm) method for feature selection provides interpretable results. Figures 3 and S3 (Supplementary Materials) show the splines learned by our $C o G A ^ { 2 } M$ model on four datasets from the TALENT benchmark. The airfoil self noise succeeds at learning the data with an $R ^ { 2 }$ on the test set of $0 . 9 2 \pm 0 . 0 0 3 8 .$ the concrete compression strength with an $R ^ { 2 }$ of $0 . 9 3 4 ~ \pm$

0.0063 and the stock with $R ^ { 2 } \operatorname { o f } 0 . 9 8 5 \pm 0 . 0 0 0 4$ . We also show the splines on the fifa dataset where it achieves a lower $R ^ { 2 }$ of $0 . 6 5 1 \pm 0 . 0 0 0 8$ . In all four cases the arity 1 and arity 2 plots show trends that make sense for the features they represent. We have found that interesting insights can be drawn from visually exploring not just the splines themselves but in the case of the 1d splines also the individual resolution sub-splines that compose them. For the case of the airfoil self noise and the fifa datasets we show in Figs. S1 and S2 (Supplementary Materials) all the sub-splines that comprise the final 1d splines of all the features for each dataset.

An important point to note in the graphs, is the fact that large regions of input space are not supported by data, especially in the arity 2 splines. In some cases, the splines seem to interpolate logically between data rich regions (e.g. the length and velocity 1d splines of the airfoil self noise). In other cases (like the release clause eur x height cm 2d spline of the fifa dataset in Fig. 3 or the Superplasticizer 1d spline of the concrete compressive strength in Fig. S3 the splines show structure (large first and second derivatives) in regions where there are no data to support those. Whether this structure is a failure of spline based methods or an often appropriate extrapolation that only such methods can achieve was the question we tried to answer by breaking the datasets into OOD train and test sets (see IV-C).

## B. TALENT benchmark

The TALENT benchmark comes with 300 datasets divided in three categories, regression sets (100 datasets), binary classification sets (120 datasets) and full classification ones (80 datasets). It also comes with the results on all of these datasets for 29 models for regression and 31 for classification. We run the $C o G A ^ { 2 } M$ model on all of the datasets and then used the Packett-Luce metric to rank all model results. $C o G A ^ { 2 } M$ achieves an 11th/29 place in the regression set and 7th/32 places in the binary and the full classification sets (see Figure 4). These results put the $C o G A ^ { 2 } M$ model in the top third of all tested models in the benchmark.

To give a more comprehensive understanding of how the models do over the range of datasets tested we also calculated the percentage change that each model achieved on each dataset compared to the MLP model. In Figure 5 we show over all models the box and whiskers plot with the box showing the quartiles of the results distribution for this model over the category’s datasets, the whiskers extend to $1 . 5 ~ \ast$ InterQuartile Distance (IQR) and the circles are the results that lie beyond this range. As can be seen by the figure, $C o G A ^ { 2 } M$ does on average just better than the MLP but with its distribution shifted to the positive (shown by the larger positive box and more positive individual datasets beyond the whisker range). This behaviour is comparable to the results produced by the best models in the datasets (e.g. tabr, realmlp, catboost, LightGBM, xgboost, modernNCA).

## C. OOD benchmark

As described in III-E we used the $C o G A ^ { 2 } M$ and the LightGBM models’ $R ^ { 2 }$ results from both the ID and the

![](images/daff98446cc31dd8f1a9297ac5bb6389bbc27773f5719488b1202d58c66a55eb.jpg)  
(b) fifa  
Fig. 3: Visualisation of the splines (1d and 2d) generated by the $C o G A ^ { 2 } M$ model on the a) airfoil self noise and b) fifa datasets. The 1d feature plots (labelled with the name of the corresponding feature) show the equivalent 1d spline (red curve). The left axis is the axis of the spline. The right is the axis of the histogram of the number of training data points over the range of the feature (blue histogram in the plot). This allows us to visualise directly the regions of the spline with and without good support. The 2d heatmaps represent the arity 2 spline surfaces. The two inputs for each plot are the features of the 1d plots above and to the right of the 2d plot. So for example in panel A, the bottom left 2d plot would have as feature input the frequency (y axis) and thickness (x-axis) features. The white dots represent the coordinates (over the two features) where data points exist, again allowing us to directly visualise the support points of the surface.

![](images/701968392872098d02fad0aa4e660f552e5e2743105acdf38045b2033b3d886b.jpg)  
Fig. 4: Results of the Packett-Luce score based ranking of the TALENT models plus $C o G A ^ { 2 } M$ for regression, binary classification and full classification datasets.

OOD datasets to calculate the ID - OOD differences for each model and each dataset that was selected to take part in the final comparison. We wanted to ask whether, for a given dataset, the $C o G A ^ { 2 } M$ model would be expected to show a smaller drop in the OOD average result vs its ID result compared to the equivalent drop in the results of the LightGBM model. To do this we first calculated for each dataset the model’s difference between the ID and the average OOD $R ^ { 2 } s \ \Delta _ { C o G A ^ { 2 } M }$ and $\Delta _ { L i g h t G B M }$ and then the between model difference of these differences $\Delta \Delta$ (as described in III-E). In Table I we show these calculated differences and differences of differences (∆∆) for all 22 datasets that we use to compare the two models. We show the actual $R ^ { 2 }$ values for the ID and OOD datasets, as well as which method was used to generate the OOD data, for all 40 models in Table S1 (Supplementary Materials).

We then tested whether these $\Delta \Delta s$ were expected to be drawn from a distribution with an expected value larger than 0. To do this we performed a one-tailed t-test and a bootstrap non-parametric test with 10000 iterations. In the t-test the tstatistic was 2.118 and the one-tailed p-value 0.0232, above the 0.05 value which would indicate that our results were drawn from a larger than 0 mean distribution by chance with a chance higher than 5%. The bootstrap results over 10000 iterations create an observed mean of 0.247 with a one-tailed p-value of 0.006 and 95% confidence intervals of 0.0481 and 0.4877. In Figure 6 we show the distribution of the means of the 10000

<table><tr><td colspan="5">Models&#x27; R2 differences and number of OOD cuts</td></tr><tr><td>Dataset</td><td> $\overline { { C o G A ^ { 2 } M } }$  OOD - ID</td><td>LGBM OOD - ID</td><td> $\Delta _ { C o G A ^ { 2 } M } .$  ∆LGBM</td><td>N</td></tr><tr><td>3D-Estimation- using-RSSI-of-</td><td>-0.164</td><td>-0.075</td><td>-0.089</td><td>13</td></tr><tr><td>WLAN-dataset- complete-1-target Another-Dataset-</td><td>-0.137</td><td>-0.017</td><td>-0.120</td><td>12</td></tr><tr><td>on-used-Fiat-500- (1538-rows)</td><td></td><td></td><td></td><td></td></tr><tr><td>Bias-correction-r Brazilian-houses-</td><td>-0.872 -0.046</td><td>-0.198 -0.164</td><td>-0.674 0.118</td><td>34 11</td></tr><tr><td>reproduced</td><td></td><td></td><td></td><td></td></tr><tr><td>IEEE80211aa- GATS</td><td>-0.397</td><td>-0.470</td><td>0.073</td><td>49</td></tr><tr><td>Laptop-Prices- Dataset</td><td>-0.619</td><td>-2.736</td><td>2.117</td><td>3</td></tr><tr><td>MiamiHousing2016</td><td>-0.755</td><td>-1.219</td><td>0.464</td><td>16</td></tr><tr><td>airfoil-self-noise</td><td>-0.197</td><td>-0.328</td><td>0.131</td><td>4</td></tr><tr><td>bank8FM</td><td>-0.245</td><td>-0.869</td><td>0.623</td><td>24</td></tr><tr><td>concrete- compressive-</td><td>-0.193</td><td>-0.455</td><td>0.262</td><td>19</td></tr><tr><td>strength</td><td></td><td></td><td></td><td></td></tr><tr><td>dataset-sales</td><td>-0.264</td><td>-0.312</td><td>0.048</td><td>1</td></tr><tr><td>delta-elevators</td><td>-0.017</td><td>-0.062</td><td>0.045</td><td>9</td></tr><tr><td>fifa</td><td>-0.324</td><td>-0.458</td><td>0.134</td><td>5</td></tr><tr><td>house-sales-reduced</td><td>-0.708</td><td>-1.147</td><td>0.439</td><td>26</td></tr><tr><td>houses</td><td>-0.245</td><td>-0.269</td><td>0.024</td><td>53</td></tr><tr><td>kin8nm</td><td>-0.506</td><td>-0.360</td><td>-0.146</td><td>15</td></tr><tr><td>pole</td><td>-0.106</td><td>-0.126</td><td>0.020</td><td>53</td></tr><tr><td>puma8NH</td><td>-0.045</td><td>-0.115</td><td>0.070</td><td>9</td></tr><tr><td>qsar-aquatic-</td><td>-0.220</td><td>-0.429</td><td>0.208</td><td>1</td></tr><tr><td>toxicity</td><td></td><td></td><td></td><td></td></tr><tr><td>stock</td><td>-0.162</td><td>-0.470</td><td>0.308</td><td>12</td></tr><tr><td>treasury</td><td>-0.097</td><td>-1.346</td><td>1.249</td><td>19</td></tr><tr><td>wind</td><td>-0.244</td><td>-0.370</td><td>0.126</td><td>10</td></tr></table>

TABLE I: Differences of $R ^ { 2 }$ between ID and OOD data for each dataset for both the $C o G A ^ { 2 } M$ and the LightGBM models and the differences of these differences between the two models. With bold we showcase the differences of the differences where $C o G A ^ { 2 } M$ does better than LightGBM.

bootstrap generated ∆∆ sets.

## V. CONCLUSION

We frame OOD as a statistical problem to develop an intuition on why $C o G A ^ { 2 } M \ \mathrm { i s } ,$ in many OOD scenarios, an appropriate model. Denote a dataset as a matrix $O = ( Y , X )$ with sample data inputs X and outputs Y where $Y = f ( X ) + \epsilon .$ We wish to learn $\{ f ( x ) : x \in \Omega _ { 0 } \}$ over the population domain $\Omega _ { 0 }$ from sampled data over a restricted domain $\Omega _ { s } ~ \subseteq ~ \Omega _ { 0 }$ The n rows of O are taken to be i.i.d realisations from the probability distribution of random vector $( y _ { i } , x _ { i } )$ with density $f _ { Y , X }$ . For finite samples, it is always the case that $\Omega _ { s } \neq \Omega _ { 0 }$ and for ID scenarios $\Omega _ { s }  \Omega _ { 0 }$ as sample size $n  \infty$ . In the OOD case, the sampling process induces a distribution with density $f _ { s } \neq f _ { Y , X }$ arising from a sampling processes $R ( y , x )$ with one of two possible special characteristics. In the first case $R \ = \ I ( y , x \in \ \Omega _ { 1 } )$ for some $\Omega _ { 1 } ~ \subset ~ \Omega _ { 0 } ,$ so that the sampled domain is restricted to a strict subset of the population domain, and $\Omega _ { s } ~  ~ \Omega _ { 1 }$ as $n  \infty$ . In the second case, supp $ { ' } f _ { s } ) =  { \mathrm { s u p p } } ( f _ { Y , X } )$ so that $\Omega _ { s }  \Omega _ { 0 }$ , but $f _ { s } ( y , x ) \neq f _ { Y , X } ( y , x )$ for some $( y , x ) \ \in \ \Omega _ { 0 }$ . That ${ \mathrm { i s } } ,$ the sampling process covers the full domain but with a distorted distribution. Importantly, there exist regions $\mathcal { R } \subseteq \Omega _ { 0 }$ of high utility $U ( y , x )$ for which $f _ { s } ( y , x ) \ll f _ { Y , X } ( y , x )$ , so that these regions are under-represented in the sampled data relative to their importance.

![](images/a0b830a2f466f82ed9a1180994cd86ffd05a48497f610f07b25b5de8d2393a58.jpg)  
(a) Regression

![](images/d78f8e074339734ce169742dd7370b781ffa4bfe7d279862de662b30664e450d.jpg)  
(b) Binary Classification  
(c) Full Classification  
Fig. 5: Comparing all models’ results with that of the MLP. For the regression models the metric used was the $r ^ { 2 }$ while for the two classification types was Accuracy. The y-axis of models for the Full Classification is the same as the one for the Binary Classification.

![](images/674772e2b71e784d17ea17ca6817b2aae7f23f3dd1a0c9baa6e83c4996d453e1.jpg)  
Fig. 6: The histogram of the means of the bootstrap generated $\Delta \Delta$ data showing that the distribution these data are drawn from is expected to have a larger than 0 mean.

Let the learner define (potentially infinite-dimensional) model space $F$ with respect to loss function ${ \mathcal { L } } ,$ and define the sample loss for choice $f \in F$ by $\mathcal { L } ( Y , X \in \Omega _ { s } ; f )$ . Then

$$
\hat { f } = \underset { ( f \in F ) } { \mathrm { a r g m i n } } \ \mathcal { L } ( Y , X \in \Omega _ { s } , f )
$$

is learnt from the sample data satisfying $\hat { f }  f _ { 1 }$ as $n \to \infty$ where

$$
\begin{array} { r } { f _ { 1 } = \underset { ( f \in F ) } { \operatorname { a r g m i n } } \ E [ \mathcal { L } ( y , x \in \Omega _ { 1 } , f ) ] . } \end{array}
$$

If $f _ { 1 } = f _ { 0 }$ such that

$$
\boldsymbol { f } _ { 0 } = \underset { ( \boldsymbol { f } \in \boldsymbol { F } ) } { \mathrm { a r g m i n } } \ \boldsymbol { E } [ \boldsymbol { \mathcal { L } } ( \boldsymbol { y } , \boldsymbol { x } \in \Omega _ { 0 } , \boldsymbol { f } ) ]
$$

then all OOD predictions will be optimally accurate. Arguments from the statistical literature on incomplete/coarse data give

$$
E [ \mathcal { L } ( y , x \in \Omega _ { 1 } , f ) ] = \iint \pi ( y , x ) \mathcal { L } ( y , x \in \Omega _ { 0 } , f ) d v ( y , x ) ,
$$

where $v ( y , x )$ is a dominating probability measure and $\pi ( y , x ) = P r ( R = 1 \mid y , x ) \in [ 0 , 1 ]$ can take deterministic values 1 or 0 (i.e. an individual is either selected or not selected) - case 1 - or even values in (0,1) (some individuals in the range are probabilistically selected) - case 2. However, it cannot be guaranteed that $f _ { 0 }$ is a solution to the above problem because the range of $\mathrm { Y }$ could be restricted or, even more severely, the codomain and domain could be jointly constrained. Correlation shifts occur when this constrained varies between the train and test data sets (see Fig. 1(c)). In this work we concentrate on datasets that do not show such correlation shifts, hence, for our cases, the following assumption is required:

Assumption 1: The sample range only depends on values of the inputs, and does not depend on the values of the output (either purposefully or inadvertently), such that $\pi ( y , x ) = \pi ( x )$ Then

$$
E [ \mathcal { L } ( y , x \in \Omega _ { 1 } , f ) ] = \int \pi ( x ) \int \mathcal { L } ( y , x \in \Omega _ { 0 } , f ) d v ( y , x ) ,
$$

which is a linear combination of input-stratified optimisation problems with no constraints on the codomain and, hence, one that is minimised by $f = f _ { 0 }$ . We note that it can also be minimised by more complex $f ^ { * }$ empirically indistinguishable from $f$ over $\Omega _ { 0 }$ when the population domain $\Omega _ { 0 }$ is a strict subset of the entire domain of $f ,$ but presume that model selection will effectively avoid choosing $f ^ { * }$ over $f .$ . However, a further assumption is still needed:

Assumption 2: $\pi ( x )$ induces ‘sufficient variation’ in $\Omega _ { 1 }$ such that there are no simpler $g \in F$ satisfying $E [ \mathcal { L } ( y , x \in$ $\Omega _ { 1 } ; g ) ] = E [ \mathcal { L } ( y , x \in \Omega _ { 1 } ; f _ { 0 } ) ]$ e.g. cases where $g ( x ) = f _ { 0 } ( x )$ for $x \in \Omega _ { 1 }$ but $g ( x ) \neq f _ { 0 } ( x )$ for $x \in \Omega _ { 0 } \backslash \Omega _ { 1 }$ (g is simpler in the sense that model selection would almost surely choose $g$ over $f _ { 0 }$ as $n \to \infty )$ .

Assumption 2 is required to prevent dangers-ofextrapolation cases where $f$ is highly non-linear over $\Omega _ { 0 }$ but $\Omega _ { 1 }$ contains only regions (possibly narrow ones) over which the behaviour of $f$ is empirically indistinguishable from $\mathrm { e . g . }$ . linear $g ( x )$ . The $C o G A ^ { 2 } M$ setup has the advantage that Assumption 2 is more plausible in the lower (one or two)-dimensional domains of each compositor function than it is for the entire multi-dimensional support of $f .$ So compared with a full-domain alternative, in case 1, $C o G A ^ { 2 } M$ induces an $\Omega _ { 1 }$ closer to $\Omega _ { 0 }$ . In case 2, the distributional distortion induced by $\pi ( x )$ is less severe when projected onto the low-dimensional domain of each compositor function, so the effective sampling density within each compositor domain is closer to the population density.

Several lessons emerge from this work. First, spline-based $G A ^ { 2 } M$ models can compete with state-of-the-art models on standard tabular benchmarks without sacrificing interpretability, provided they are equipped with sufficient expressiveness through multi-arity basis functions. Second, the smooth inductive bias inherent in spline methods appears to be not merely an aesthetic or interpretability choice but a functional one: it offers a measurable advantage under diversity shift, where models must generalise to unseen regions of input space. Third, our adversarial OOD generation procedure revealed that many datasets (12 out of 40) commonly used in tabular benchmarks are surprisingly resistant to diversity shift - both models continued to perform well (with OOD $R ^ { \dot { 2 } } { \bf s }$ larger than 0.5) - suggesting that standard benchmark performance may overstate real-world robustness. Finally, we found that the choice of proxy learner in the adversarial loop (kNNR vs. νSVR) materially affects which datasets can be successfully broken, highlighting that OOD benchmark construction is itself a non-trivial design decision that deserves more attention from the community.

## VI. FUTURE WORK

Several questions have been raised by this work. First, we did not fully explore the importance of the multi-resolution character of our model. Initial experiments showed that breaking up the splines into separate, multi-knot resolution, linear and cubic parts and allowing the backpropagation algorithm to learn their additive weights (with an L1 regularisation) improved performance. This claim though will need to be adequately substantiated in future work with appropriate comparisons between our and GAM based models that do not decompose splines. Second, our $C o G A ^ { 2 } M$ model was run with fixed hyperparameters and a single repetition, without the extensive tuning afforded to other models in the TALENT benchmark; a systematic hyperparameter search — particularly over the set of resolutions, the arity, and the learning rate — could reveal further performance gains. Third, our treatment of categorical features relied on TALENT’s simple linear encoding rather than learned embeddings, which likely underestimates the model’s potential on categorically rich datasets. Initial efforts with a one-hot encoding faced computability problems - mainly due to the exponential growth of the arity 2 components - so efforts to create a more nuanced encoding scheme of the categorical features could potentially increase the model’s results in such datasets. Fourth, our OOD generation procedure was restricted to regression datasets with purely numerical features, leaving open the question of how spline-based models behave under distribution shift in classification tasks and in the presence of categorical variables. Additionally, our OOD generation could be improved by more direct OOD metrics compared to the proxy ability of a model to solve the test data. Finally, an important extension of our work would be to generate datasets not only with diversity but also correlation shifts. The correlation shift setting where spurious feature–target associations change between training and test — remains unexplored and would require careful experimental design to disentangle genuine from spurious relationships. Finally we concentrate only on diversity shifts, leaving the correlation shift scenario for future work, as it would require intensive hyperparameter optimisation to appropriately disentangle spurious from genuine feature–target relationships.

## VII. ACKNOWLEDGMENTS

GD and SS acknowledge the support of Engineering and Physical Sciences Research Council (EPSRC) grant ”UKRI1076 Element-compositor methods for out-ofdistribution machine learning”.

PC was partly funded by UK Economic and Social Research Council award ES/S012486/1 (MiSoC).

DM acknowledges the support of the UKRI AI programme, and the Engineering and Physical Sciences Research Council, for CHAI - Causality in Healthcare AI Hub [grant number EP/Y028856/1].

For the purpose of open access, the authors have applied a Creative Commons Attribution (CC BY) licence to any Author Accepted Manuscript version arising.

We used Claude Opus 4.6 Extended as an editor. The whole manuscript was written manually by the authors and then given to Claude to do small changes in the format of phrases and the choice of certain words. The proposed changes were then re-reviewed by the authors.

## REFERENCES

[1] T. Hastie and R. Tibshirani, “Generalized Additive Models.”

[2] P. H. C. Eilers and B. D. Marx, “Flexible smoothing with B-splines and penalties,” Statistical Science, vol. 11, no. 2, May 1996.

[3] S. N. Wood, “Thin Plate Regression Splines,” Journal of the Royal Statistical Society Series B: Statistical Methodology, vol. 65, no. 1, pp. 95–114, Aug. 2003.

[4] Y. Lou, R. Caruana, J. Gehrke, and G. Hooker, “Accurate intelligible models with pairwise interactions,” in Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, ser. KDD ’13. New York, NY, USA: Association for Computing Machinery, Aug. 2013, pp. 623–631.

[5] A. Watanabe, M. Kuramata, K. Majima, H. Kiyohara, K. Kensho, and K. Nakata, “Constrained Generalized Additive 2 Model With Consideration of High-Order Interactions,” in 2021 International Conference on Electrical, Computer and Energy Technologies (ICECET), Dec. 2021, pp. 1–6.

[6] R. Agarwal, L. Melnick, N. Frosst, X. Zhang, B. Lengerich, R. Caruana, and G. E. Hinton, “Neural additive models: Interpretable machine learning with neural nets,” in Proceedings of the 35th International Conference on Neural Information Processing Systems, ser. NIPS ’21. Red Hook, NY, USA: Curran Associates Inc., Dec. 2021, pp. 4699–4711.

[7] Z. Yang, A. Zhang, and A. Sudjianto, “GAMI-Net: An explainable neural network based on generalized additive models with structured interactions,” Pattern Recognition, vol. 120, p. 108192, Dec. 2021.

[8] ——, “Enhancing Explainability of Neural Networks Through Architecture Constraints,” IEEE Transactions on Neural Networks and Learning Systems, vol. 32, no. 6, pp. 2610–2621, Jun. 2021.

[9] S. Kruschel, N. Hambauer, S. Weinzierl, S. Zilker, M. Kraus, and P. Zschech, “Challenging the Performance-Interpretability Trade-Off: An Evaluation of Interpretable Machine Learning Models,” Business & Information Systems Engineering, Feb. 2025.

[10] S. Ben-David, J. Blitzer, K. Crammer, A. Kulesza, F. Pereira, and J. W. Vaughan, “A theory of learning from different domains,” Machine Learning, vol. 79, no. 1, pp. 151–175, May 2010.

[11] M. Arjovsky, L. Bottou, I. Gulrajani, and D. Lopez-Paz, “Invariant Risk Minimization,” ArXiv, Jul. 2019.

[12] S. Sagawa, P. W. Koh, T. B. Hashimoto, and P. Liang, “Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization,” ArXiv, Nov. 2019.

[13] D. Krueger, E. Caballero, J.-H. Jacobsen, A. Zhang, J. Binas, D. Zhang, R. L. Priol, and A. Courville, “Out-of-Distribution Generalization via Risk Extrapolation (REx),” in Proceedings of the 38th International Conference on Machine Learning. PMLR, Jul. 2021, pp. 5815–5826.

[14] Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, and V. Lempitsky, “Domain-Adversarial Training of Neural Networks,” in Domain Adaptation in Computer Vision Applications, G. Csurka, Ed. Cham: Springer International Publishing, 2017, pp. 189–209.

[15] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, “Mixup: Beyond Empirical Risk Minimization,” in International Conference on Learning Representations, Feb. 2018.

[16] I. Gulrajani and D. Lopez-Paz, “In Search of Lost Domain Generalization,” in International Conference on Learning Representations, 2021.

[17] P. W. Koh, S. Sagawa, H. Marklund, S. M. Xie, M. Zhang, A. Balsubramani, W. Hu, M. Yasunaga, R. L. Phillips, I. Gao, T. Lee, E. David, I. Stavness, W. Guo, B. Earnshaw, I. Haque, S. M. Beery, J. Leskovec, A. Kundaje, E. Pierson, S. Levine, C. Finn, and P. Liang, “WILDS: A Benchmark of in-the-Wild Distribution Shifts,” in Proceedings of the 38th International Conference on Machine Learning. PMLR, Jul. 2021, pp. 5637–5664.

[18] L. Yuan, Y. Chen, G. Cui, H. Gao, F. Zou, X. Cheng, H. Ji, Z. Liu, and M. Sun, “Revisiting Out-of-distribution Robustness in NLP: Benchmarks, Analysis, and LLMs Evaluations,” Advances in Neural Information Processing Systems, vol. 36, pp. 58 478–58 507, Dec. 2023.

[19] L. Yang, S. Zhang, L. Qin, Y. Li, Y. Wang, H. Liu, J. Wang, X. Xie, and Y. Zhang, “GLUE-X: Evaluating Natural Language Understanding Models from an Out-of-Distribution Generalization Perspective,” in Findings of the Association for Computational Linguistics: ACL 2023. Toronto, Canada: Association for Computational Linguistics, 2023, pp. 12 731–12 750.

[20] N. Ye, K. Li, F. Wu, J. Zhou, H. Bai, R. Yu, L. Hong, F. Zhou, Z. Li, J. Zhu, X. Wang, and C. Zhou, “OoDBench+: Quantifying and Understanding Two Dimensions of Out-of-Distribution Generalization,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 3, pp. 2566–2580, Mar. 2026.

[21] J. Gardner, Z. Popovic, and L. Schmidt, “Benchmarking Distribution Shift in Tabular Data with TableShift,” 2023.

[22] A. Subbaswamy and S. Saria, “From development to deployment: Dataset shift, causality, and shift-stable models in health AI,” Biostatistics, vol. 21, no. 2, pp. 345–352, Apr. 2020.

[23] V. Subasri, A. Krishnan, A. Kore, A. Dhalla, D. Pandya, B. Wang, D. Malkin, F. Razak, A. A. Verma, A. Goldenberg, and E. Dolatabadi, “Detecting and Remediating Harmful Data Shifts for the Responsible Deployment of Clinical AI Models,” JAMA Network Open, vol. 8, no. 6, p. e2513685, Jun. 2025.

[24] S.-Y. Liu, H.-R. Cai, Q.-L. Zhou, H.-H. Yin, T. Zhou, J.-P. Jiang, and H.-J. Ye, “Talent: A Tabular Analytics and Learning Toolbox.”

[25] N. Hansen, “The CMA Evolution Strategy: A Tutorial,” Mar. 2023.

[26] T. Chen and C. Guestrin, “XGBoost: A Scalable Tree Boosting System,” Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785–794, Aug. 2016.

[27] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, “CatBoost: Unbiased boosting with categorical features,” in Proceedings

of the 32nd International Conference on Neural Information Processing Systems, ser. NIPS’18. Red Hook, NY, USA: Curran Associates Inc., Dec. 2018, pp. 6639–6649.

[28] S. O. Arik and T. Pfister, “TabNet: Attentive Interpretable Tabular Learn-<sup>¨</sup> ing,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 8, pp. 6679–6687, May 2021.

[29] K. Zhou, Z. Liu, Y. Qiao, T. Xiang, and C. C. Loy, “Domain Generalization: A Survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 04, pp. 4396–4415, Apr. 2023.

[30] K. Tsai, S. R. Pfohl, O. Salaudeen, N. Chiou, M. Kusner, A. D’Amour, S. Koyejo, and A. Gretton, “Proxy Methods for Domain Adaptation,” in Proceedings of The 27th International Conference on Artificial Intelligence and Statistics. PMLR, Apr. 2024, pp. 3961–3969.

[31] L. Maystre and M. Grossglauser, “Fast and Accurate Inference of Plackett– Luce Models,” in Advances in Neural Information Processing Systems, vol. 28. Curran Associates, Inc., 2015.

Rethinking $G A ^ { 2 } M s \mathrm { . }$ : performance and out-of-distribution generalisation. Supplemental materials

![](images/c7e68e84072036d9b68e656179a990890e3155a02dab1ff61453e9b0252e9ea1.jpg)  
Fig. S1. Sub-splines for the 1d splines of the airfoil self noise dataset.

![](images/98af28cbd54a48f91e0063c061a085766ec1b352a137d612024e58967310ad33.jpg)  
Fig. S1. Sub-splines for the 1d splines of the airfoil self noise dataset

(e) Frequency

![](images/1365cbc6b2611b254ad51bc64ac0e048a62f2f0fca08daaa994e62b1d74f41c4.jpg)  
Fig. S1. Sub-splines for the 1d splines of the airfoil self noise dataset. For each feature there are 7 sub-splines linearly composed through learned weights (Coef. at the top of each sub-spline plot) to the final feature spline (bottom left of each group).The red lines are the sub-splines while the blue histogram (left y-axis) shows the data distribution).

![](images/cd52a6124853b0f7e7a1083b0e5c819c8afbad4cf390cf090ef9cd27ce5d84bc.jpg)  
Fig. S2. Sub-splines for the 1d splines of the fifa dataset

![](images/cd9f08ad5c855d70a6b0f4ae4eaf22de41acebc07ed9e5fe8ad2ee8442460c4b.jpg)  
Fig. S2. Sub-splines for the 1d splines of the fifa dataset

![](images/086897fb866d59190471130eedc577022b0a77dadeacf04f0a756392a4c3effe.jpg)  
Fig. S2. Sub-splines for the 1d splines of the fifa dataset. For each feature there are 7 sub-splines linearly composed through learned weights (Coef. at the top of each sub-spline plot) to the final feature spline (bottom left of each group).The red lines are the sub-splines while the blue histogram (left y-axis) shows the data distribution)  
(e) age

![](images/bbc6230f42b3462567f43a162fe9127d6de5d389bf0ac0099914259d66bb8265.jpg)  
(a) stock

![](images/357b113c383b5e8862fc050fea40659d900dfc30e5b8663244b509a98759226f.jpg)  
(b) concrete compressive strength  
Fig. S3. Visualisation of the splines (1d and 2d) generated by the $C o G A ^ { 2 } M$ model on the (a) stock and (b) concrete compressive strength. See Fig. 2 for details.

<table><tr><td colspan="6">Models&#x27;  $\overline { { r ^ { 2 } } }$  for ID and OOD datasets</td></tr><tr><td>Dataset</td><td> $\overline { { C o G A ^ { 2 } M ~ R _ { I D } ^ { 2 } } }$ </td><td> $\overline { { \mathrm { L G B M } ~ R _ { I D } ^ { 2 } } }$ </td><td> $\overline { { C o G A ^ { 2 } M } }$   $R _ { O O D } ^ { 2 }$ </td><td> $\overline { { \mathrm { \mathbf { L G B M } } \ R _ { O O D } ^ { 2 } } }$ </td><td>OOD generator</td></tr><tr><td>3D-Estimation-using-RSSI-of-WLAN-</td><td>0.995</td><td>0.999</td><td>0.831</td><td>0.925</td><td>kNNR</td></tr><tr><td>dataset-complete-1-target Another-Dataset-on-used-Fiat-500-(1538-</td><td>0.853</td><td>0.828</td><td>0.730</td><td>0.813</td><td>kNNR</td></tr><tr><td>rows) Bias-correction-r</td><td>0.924</td><td>0.933</td><td>0.111</td><td>0.748</td><td>νSVR</td></tr><tr><td>Brazilian-houses-reproduced</td><td>0.988</td><td>0.989</td><td>0.944</td><td>0.831</td><td>kNNR</td></tr><tr><td>IEEE80211aa-GATS</td><td>0.994</td><td>0.979</td><td>0.635</td><td>0.554</td><td>kNNR</td></tr><tr><td>Laptop-Prices-Dataset</td><td>0.666</td><td>0.646</td><td>-0.012</td><td>-2.356</td><td>kNNR</td></tr><tr><td>MiamiHousing2016</td><td>0.933</td><td>0.929</td><td>0.199</td><td>-0.256</td><td>νSVR</td></tr><tr><td>airfoil-self-noise</td><td>0.92</td><td>0.938</td><td>0.704</td><td>0.578</td><td>νSVR</td></tr><tr><td>bank8FM</td><td>0.963</td><td>0.956</td><td>0.706</td><td>0.045</td><td>kNNR</td></tr><tr><td>concrete-compressive-strength</td><td>0.934</td><td>0.909</td><td>0.75</td><td>0.478</td><td>kNNR</td></tr><tr><td>dataset-sales</td><td>0.631</td><td>0.524</td><td>0.369</td><td>0.215</td><td>kNNR</td></tr><tr><td>delta-elevators</td><td>0.646</td><td>0.564</td><td>0.63</td><td>0.504</td><td>kNNR</td></tr><tr><td>fifa</td><td>0.651</td><td>0.618</td><td>0.333</td><td>0.168</td><td>kNNR</td></tr><tr><td>house-sales-reduced</td><td>0.862</td><td>0.825</td><td>0.061</td><td>-0.472</td><td>νSVR</td></tr><tr><td>houses</td><td>0.81</td><td>0.839</td><td>0.568</td><td>0.573</td><td>νSVR</td></tr><tr><td>kin8nm</td><td>0.611</td><td>0.824</td><td>0.119</td><td>0.474</td><td>kNNR</td></tr><tr><td>pole</td><td>0.958</td><td>0.989</td><td>0.852</td><td>0.864</td><td>νSVR</td></tr><tr><td>puma8NH</td><td>0.666</td><td>0.606</td><td>0.621</td><td>0.493</td><td>kNNR</td></tr><tr><td>qsar-aquatic-toxicity</td><td>0.526</td><td>0.482</td><td>0.252</td><td>-0.048</td><td>kNNR</td></tr><tr><td>stock</td><td>0.985</td><td>0.98</td><td>0.802</td><td>0.45</td><td>kNNR</td></tr><tr><td>treasury wind</td><td>0.995</td><td>0.992</td><td>0.904</td><td>-0.275</td><td>kNNR</td></tr><tr><td>1000-Cameras-Dataset</td><td>0.81</td><td>0.783</td><td>0.569</td><td>0.418</td><td>kNNR</td></tr><tr><td>3D-Estimation-using-RSSI-of-WLAN-</td><td>0.571</td><td>0.208 0.999</td><td>-0.583 -2.54</td><td>-1.699</td><td>kNNR</td></tr><tr><td>dataset</td><td>0.998</td><td></td><td></td><td>-1.687</td><td>kNNR</td></tr><tr><td>CookbookReviews</td><td>0.069</td><td>-0.009</td><td>-1.22</td><td>-1.011</td><td>kNNR</td></tr><tr><td>Parkinson-Multiple-Sound-Recording</td><td>0.096</td><td>-0.0618</td><td>-0.529</td><td>-0.575</td><td>kNNR</td></tr><tr><td>Wine-Quality-red</td><td>0.305</td><td>0.385</td><td>-0.303</td><td>-0.074</td><td>kNNR</td></tr><tr><td>cpu-act</td><td>0.985</td><td>0.984</td><td>-111.127</td><td>-5.654</td><td>νSVR</td></tr><tr><td>cpu-small</td><td>0.977</td><td>0.979</td><td>-231.819</td><td>-5.47</td><td>νSVR</td></tr><tr><td>debutanizer</td><td>0.757</td><td>0.814</td><td>-1.072</td><td>-0.721</td><td>kNNR</td></tr><tr><td>house-16H-reg</td><td>0.587</td><td>0.634</td><td>-0.215</td><td>-0.088</td><td>νSVR</td></tr><tr><td>house-8L</td><td>0.676</td><td>0.673</td><td>-0.81</td><td>-0.997</td><td>νSVR</td></tr><tr><td>space-ga</td><td>0.701</td><td>0.683</td><td>-0.154</td><td>-0.235</td><td>kNNR</td></tr><tr><td>sulfur</td><td>0.717</td><td>0.825</td><td>-1.32</td><td>-1.301</td><td>kNNR</td></tr><tr><td>CookbookReviews</td><td>0.069</td><td>-0.009</td><td>-1.22</td><td>-1.011</td><td>kNNR</td></tr><tr><td>Goodreads-Computer-Books</td><td>0.148</td><td>-0.292</td><td>0.594</td><td>0.51</td><td>νSVR</td></tr><tr><td>Parkinson-Multiple-Sound-Recording</td><td>0.096</td><td>-0.061</td><td>-0.529</td><td>-0.575</td><td>kNNR</td></tr><tr><td>Wine-Quality-red</td><td>0.305</td><td>0.385</td><td>-0.303</td><td>-0.074</td><td>kNNR</td></tr><tr><td>Wine-Quality-white</td><td>0.371</td><td>0.463</td><td>0.286</td><td>0.311</td><td>kNNR</td></tr><tr><td>wine+quality</td><td>0.304</td><td>0.381</td><td>0.147</td><td>0.252</td><td>kNNR</td></tr><tr><td>analcatdata-supreme</td><td>0.971</td><td>0.979</td><td>0.98</td><td>0.99</td><td>νSVR</td></tr><tr><td>combined-cycle-power-plant weather-izmir</td><td>0.945 0.993</td><td>0.962 0.99</td><td>0.911 0.974</td><td>0.9 0.989</td><td>kNNR kNNR</td></tr></table>

TABLE S1

$R ^ { 2 }$ OF ID AND OOD DATA FOR EACH DATASET WITH OOD DATA FOR BOTH THE CoGA<sup>2</sup>M AND THE LIGHTGBM MODELS. THE TOP SUB PART OF THE TABLE SHOWS THE DATASETS THAT WERE FINALLY USED IN THE COMPARISON BETWEEN CoGA2M AND LIGHTGBM. THE MIDDLE PART SHOWS THE DATASETS THAT DID NOT MAKE IT INTO THE ABOVE LIST BECAUSE BOTH THE CoGA<sup>2</sup>M AND THE LIGHTGBM OOD $R ^ { 2 } { \mathrm { s } }$ WERE SMALLER THAN 0. THE THIRD PART SHOWS DATASETS THAT WERE REMOVED BECAUSE BOTH MODELS DID NOT ACHIEVE AN $R ^ { 2 }$ LARGER THAN 0.5 IN THE ID DATA. FINALLY THE LAST PART SHOWS DATASETS REMOVED BECAUSE BOTH MODELS MANAGED TO ACHIEVE LARGER THAN 0.9 R<sup>2</sup>S IN BOTH THE ID AND OOD DATA