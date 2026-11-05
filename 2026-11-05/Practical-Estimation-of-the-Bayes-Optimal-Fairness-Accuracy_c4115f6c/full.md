# Practical Estimation of the Bayes Optimal Fairness-Accuracy Tradeoff with Soft Labels

Mohit Sharma<sup>1∗</sup> Okan Koç<sup>2</sup> Amit Deshpande<sup>3</sup> Takashi Ishida<sup>4,2</sup>

Gang Niu<sup>2</sup> Rajiv Ratn Shah<sup>5</sup> Masashi Sugiyama<sup>4,2</sup>

<sup>1</sup>IIIT Delhi <sup>2</sup>RIKEN-AIP <sup>3</sup>Microsoft Research <sup>4</sup>The University of Tokyo <sup>5</sup>IIT Kanpur

## Abstract

There is a fundamental limit to the best prediction error of any model for a given data distribution. In classification, the Bayes error quantifies this limit and serves as both a benchmark for trained models and a criterion to detect overfitting. The state-of-the-art Bayes error estimation techniques for binary classification use only soft labels (positive class probabilities) and require no input features or auxiliary training. We extend such notions to constrained classification problems, such as fairness. Fair classification is an important case of constrained or multi-objective classification. It requires a model not only to be accurate but also to yield fair outcomes (equal or near-equal true positive rates) across different demographic groups (e.g., race and gender). This leads to an inherent fairness-accuracy tradeoff or a Pareto frontier that is used to compare fair classifiers against each other. The focus of our work is to estimate the fundamental limit for this fairness-accuracy tradeoff. We derive an explicit expression for the fair Bayes error rate (i.e., best achievable error rate subject to given fairness constraints), and construct consistent, instance-free estimators for it using only soft labels and group information. We provably bound the bias and the sample complexity of our estimators. Empirically, our method produces a robust, low-variance estimate of the optimal fairnessaccuracy tradeoff curve, even when the soft labels are noisy and derived from a classification model on hard labels.

## 1 Introduction

The notion of Bayes optimality in machine learning defines the best achievable performance, given a data distribution, with respect to the 0-1 loss [1, 2, 3, 4]. For binary classification, an observation consist of features $X \in \mathcal { X } \subseteq \mathbb { R } ^ { d }$ , a label $Y \in \mathcal { Y } = \{ 0 , 1 \}$ }, and a sensitive attribute $A \in \mathcal { A } = \{ 0 , 1 \}$ governed by an underlying distribution $( X , Y , A ) \sim { \overset { \cdot } { \mathcal { D } } } . { \overset { \cdot } { \mathrm { A } } }$ group-aware classifier $g : \mathcal { X } \times \mathcal { A }  \mathcal { \dot { \{ 0 , 1 \} } }$ predicts labels using both features and sensitive attributes. A well-known result states that the Bayes optimal classifier $\check { g ^ { * } }$ can be expressed as the following decision rule: $g ^ { * } ( x , a ) = \mathbb { I } \left( \eta ( x , a ) \geq \mathrm { i } / 2 \right)$ [4, 5]. Here, I () denotes the indicator function and $\eta ( x , a ) : = \operatorname* { P r } ( Y = 1 \mid X = x , A = a )$ defines the posterior, which assigns a soft label $c _ { i } : = \eta ( x _ { i } , a _ { i } )$ to an instance. This rule also allows us to estimate the Bayes error, the best achievable performance for a given distribution, without access to any instances [4, 6]: $\beta = \mathbb { E } _ { X , A } \left[ \operatorname* { m i n } \{ C , 1 - C \} \right]$ , where $C : \doteq \eta ( X , A )$ is the random variable corresponding to the soft labels.

This notion can also be extended to multi-objective classification problems, such as those that enforce fairness constraints [7]. With the widespread deployment of machine learning models, it is essential that models are fair and do not produce disproportionate errors and harm to certain sensitive groups in the data [7, 8, 9]. However, applying fairness constraints generally introduces a tradeoff with accuracy, since common fairness constraints are incompatible with the standard notion of accuracy [10, 11]. This tradeoff depends sharply on (i) the underlying data distribution, (ii) the fairness metric (equality of positives or true positives), and (iii) the degree of allowed unfairness tolerance.

![](images/b27dcc0c8a9cf0cf3de9c79562bd1564d2ce71eba08218b58aefb295338b79eb.jpg)

![](images/f258be6ba109ccafbae8952c33cf3a26a474f7797dbac7a563c7f3b8f238d192.jpg)  
Figure 1: Unfairness-error tradeoffs on synthetic data [12] for Demographic Parity (DP) and Equal Opportunity (EOP) constraints. The proposed estimate (green) establishes a lower bound against competing algorithms. Dotted lines denote the unconstrained Bayes Optimal classifier at $\lambda = 0$

Bayes optimal fair classifiers can often be characterized as adjustments of the classification thresholds: $g _ { \mathrm { f a i r } } ^ { * ^ { - } } ( x , a ) = \mathbb { I } \left( \eta ( x , a ) \geq 1 / 2 + t _ { a } ( \delta ) \right)$ where $t _ { a } ( \delta ) > 0$ is a group-dependent adjustment that depends on the fairness metric, the underlying data distribution and the tolerance for unfairness δ. This is precisely where existing works on estimating the Bayes error do not generalize [6, 13, 14]. While Bayes error estimators solely rely on posterior probabilities (soft scores), resulting in a constant threshold of $1 / 2 .$ , this is not the case for cost-sensitive thresholds, such as fairness. The error estimators for these cost-sensitive thresholds also require an estimator for the underlying thresholds.

Therefore, to estimate the optimal fairness-accuracy tradeoff, we first show how we can disentangle the unfairness level δ and the corresponding threshold adjustment $t _ { a } ( \delta )$ by studying when the classifiers become trivial (Lemma 3.8). Using this, we can estimate the threshold and, eventually, the unfairness and error rate at several points along the tradeoff curve by leveraging its monotonicity properties (Lemma C.1). We show that fair classification can be reduced to cost-sensitive thresholding with appropriate λ-dependent thresholds (Lemma C.5), where $\lambda \in \mathbb { R }$ is a parameter that directly controls the correction to the classification threshold and depends on δ. We then derive consistent estimators for the fair classification thresholds, only using soft labels, at any fixed λ (Theorem 4.1).

Once we have consistent estimates of thresholds, we then, as an independent result, generalize the Bayes error estimators in prior works [6, 13, 14] to work with any cost-sensitive risk (Theorem 4.3). Since our thresholds and estimators are both constructed using finite soft labels, we move beyond the scope of prior art to study consistency, bias, and sample complexity of the estimation of the cost-sensitive Bayes risk (Theorem 4.4). These results and rates apply to many cost-sensitive learning scenarios [15, 16] that require distribution-dependent classification thresholds.

We then focus on estimating the optimal fairness-accuracy tradeoff using only soft labels. We first derive the estimated risk at any threshold and show its consistency properties (Theorem 4.5). We then also provide estimators for the expected unfairness under demographic parity and equal opportunity, and derive their consistency and sample complexity properties (Theorem 4.6). All of them culminate in a bisection-search driven algorithm with our estimators (Algorithm 1), inspired by a similar algorithm for the Bayes optimal fair classifiers [5], but only with soft labels and without instances. An example of this bisection algorithm in action is presented in Figure 1 (green curve).

Finally, we estimate the fairness-accuracy tradeoff on real-world datasets. We compare our estimated tradeoff against several recent works on estimating the fairness-accuracy tradeoff [17, 18] and differentiable objectives for fair classification [19], in Figures 1 and 2. We show that our estimated tradeoff lower bounds the tradeoff estimated by these methods. Whenever it does not (for example, in the case of the algorithm proposed in [18]), it is explained by the algorithm’s optimization of a completely different lower bound on the Pareto frontier. We also perform experiments with noisy soft scores generated by a Logistic Regression model on the Adult dataset [20] and find that our method still lower-bounds all other fairness tradeoff algorithms.

Overall, in a similar spirit to prior work [6, 13, 14], where the authors use the Bayes error as a tool to validate performance and check for overfitting, our estimated tradeoff with soft labels establishes a similar precedent: Any current or future methods aiming to compare fair algorithms should always close the gap between their estimated tradeoff and the theoretically best, that we estimate.

Related Work: Our work is inspired by the prior results [6, 13, 14] that characterize the performance limitations in terms of accuracy using the threshold characterization of the Bayes optimal classifier. There are several other works on Bayes error estimation that aim to estimate the Bayes error indirectly [21, 22, 23, 24]. But since most of these methods estimate the Bayes error indirectly, without leveraging soft labels or thresholding rules, generalizing them to constrained optimization settings, such as fairness constraints and cost-sensitive risk, is future work beyond the scope of this work. A major theme of this paper centers around fair machine learning, which has received rightful attention in recent years [7]. Several fairness metrics, such as demographic parity [25] and equal opportunity [26], have been proposed, and corresponding fair Bayes optimal classifiers have been extensively studied in the literature [27, 11]. We make extensive use of threshold characterizations of fair Bayes optimal classifiers [11, 28, 5]. Finally, a major part of our work also concerns deriving consistency and sample-complexity properties of the thresholds and risk and unfairness estimators. Some related estimators have also been studied in prior work [29, 13].

## 2 Notations and Assumptions

Let $\pi _ { a } = \operatorname* { P r } \left( A = a \right) , \pi _ { y a } = \operatorname* { P r } \left( Y = y , A = a \right) , \pi _ { y | a } = \operatorname* { P r } \left( Y = y | A = a \right) , \pi = \operatorname* { P r } \left( Y = 1 \right)$ , and Pr $( Y = y ) = \sum _ { a \in \mathcal { A } } \pi _ { y a }$ . We assume the existence of all groups and classes, i.e., $\pi _ { a } > 0 , \pi _ { y | a } > 0$ for

all $y \in \mathcal { V }$ and for all $a \in { \mathcal { A } }$ . Finally, we let $\xrightarrow { p }$ denote convergence in probability. All probabilities and expectations are taken with respect to the underlying distribution $\mathcal { D }$

Whenever we work with a finite sample, we assume that we are given access to N i.i.d. soft labels of the form $( c _ { i } , a _ { i } )$ , and we will denote our dataset with $D = \{ ( c _ { i } , a _ { i } ) \} _ { i = 1 } ^ { N }$ . Since our approach is model-free, we do not require access to features $x _ { i }$ . We will also partition our dataset $D$ into two parts: $D = D _ { \alpha } \cup D _ { \beta }$ , with $D _ { \alpha } \cap D _ { \beta } = \emptyset$ , where $D _ { \alpha }$ (with $N _ { \alpha }$ samples) will be the data used to estimate the thresholds that we will put on $\eta ( x , a )$ , and $D _ { \beta }$ (with $N _ { \beta }$ samples) will be used to construct the estimators of classification risk and unfairness, with $N \stackrel { . } { = } N _ { \alpha } + N _ { \beta }$ . The splits are chosen independently of the sample values. Furthermore, we will denote with $N _ { \alpha , a }$ (or $N _ { \beta , a } )$ the number of samples from group $A = a$ . Lastly, $i \in [ D ]$ denotes drawing an index from the dataset $D$ . We need two independent sample sets because we estimate the required quantities in two steps. First, using $D _ { \alpha }$ , we estimate the distribution priors and group-dependent thresholds (Theorem 4.1), and next using $D _ { \beta }$ , we estimate the cost-sensitive Bayes risk, the expected error, and unfairness (Theorems $4 . 4 , 4 . 5 ,$ , and 4.6 respectively). We now assume the following about our soft labels:

Assumption 2.1. (No interior atoms / Continuity of the Cumulative Distribution function of η) The mapping $t  F _ { a } ( t ) = \operatorname* { P r } ( \eta ( X , a ) \leq t | A = a ) , \forall a \in \mathcal { A }$ is continuous in $( 0 , 1 )$ . Moreover, $\operatorname* { P r } \left( \eta ( X , a ) \geq 1 / 2 | A = a \right) > 0 { \mathrm { ~ f o r ~ a l l ~ } } a \in { \mathcal { A } } .$

Assumption 2.2. (Lipschitz continuous group CDF) For each $a \in { \mathcal { A } } .$ the group-conditional distribution of $C = \eta ( X , { \bar { a } } ) | A = a$ admits a density $f _ { a }$ on $( 0 , 1 )$ such that sup $\bar { f _ { a } } ( t ) \leq L _ { a } < \infty$ . In $t \in ( 0 , 1 )$ particular, $F _ { a }$ is L<sub>a</sub>-Lipschitz: $| F _ { a } ( u ) - F _ { a } ( v ) | \le L _ { a } \cdot | u - v |$ for all $u , v \in ( 0 , 1 )$

Let $\begin{array} { r } { \hat { \pi } _ { a } = \frac { 1 } { N _ { \alpha } } \sum _ { i \in [ D _ { \alpha } ] } \mathbb { I } \left( a _ { i } = a \right) } \end{array}$ denote our estimator for $\pi _ { a } .$ . By the Weak Law of Large Numbers, $\hat { \pi } _ { a } \overset { p } {  } \pi _ { a } .$ . Furthermore, the bias of the estimator $\hat { \pi } _ { a }$ is zero, which is easy to see via the linearity of expectation. We can derive a similar unbiased estimator for $\pi = \operatorname* { P r } \left( Y = 1 \right)$ as $\begin{array} { r } { \hat { \pi } = \frac { 1 } { N _ { \alpha } } \sum _ { i \in [ D _ { \alpha } ] } c _ { i } } \end{array}$

Similarly, let $\begin{array} { r } { \hat { \pi } _ { 1 a } = \frac { 1 } { N _ { \alpha } } \sum _ { i \in [ D _ { \alpha } ] } \mathbb { I } \left( a _ { i } = a \right) c _ { i } } \end{array}$ denote our estimator for $\pi _ { 1 a }$ . By the Weak Law of Large Numbers, $\hat { \pi } _ { 1 a } \overset { p } {  } \pi _ { 1 a }$ . Furthermore, the bias of the estimator $\hat { \pi } _ { 1 a }$ is zero, which is easy to see via the linearity of expectation. The same analysis holds for the unbiased estimator ${ \hat { \pi } } _ { 1 | a } =$ $\begin{array} { r } { \frac { 1 } { N _ { \alpha , a } } \sum _ { i \in [ D _ { \alpha } ] } \mathbb { I } \left( a _ { i } = a \right) c _ { i } } \end{array}$ and its expected value $\pi _ { 1 | a }$ . Since we will work with inverse functions of $\hat { \pi } _ { a }$ (and other estimators), we also define $\begin{array} { r } { \hat { \pi } _ { a } ^ { + } = \operatorname* { m a x } \{ \hat { \pi } _ { a } , \frac { 1 } { N _ { \alpha } } \} } \end{array}$ , which is still statistically consistent, but a biased estimator (same for $\hat { \pi } _ { 1 a } ^ { + }$ and ${ \hat { \pi } _ { 1 | a } ^ { + } } )$ . Finally, let $\Pi _ { [ 0 , 1 ] } ( t ) : = \operatorname* { m i n } \{ 1 , \operatorname* { m a x } \{ 0 , t \} \}$ denote the truncation operation, which we will frequently use to ensure our estimators are bounded between 0 and 1. All proofs are provided in the Appendix.

## 3 Bayes Optimal Fair Classifiers

We will now describe a well-known result that describes the Bayes optimal classifier for binary classification, when we know the underlying distribution.

Lemma 3.1. [3, 11, 5] The Bayes optimal classifier $g ^ { * }$ for a given distribution can be precisely characterized as: $g ^ { * } ( x , a ) = \mathbb { I } \left( \bar { \eta } ( x , \bar { a } ) \geq { 1 } / { 2 } \right)$

However, as noted in the literature [11, 28, 5], the unconstrained Bayes optimal classifier for a distribution may not satisfy fairness constraints. To study this phenomenon in more detail, we now introduce definitions of the fairness constraints we consider in this work.

Definition 3.2. A classifier g is said to be δ-fair if: (1) Demographic Parity (DP) [25]: $| \mathrm { P r } \left( g ( X , A ) = 1 | A = 1 \right) - \mathrm { P r } \left( g ( X , A ) = 1 | A = 0 \right) | \leq \delta$ , and (2) Equal Opportunity (EOP) [26]: $| \mathrm { { P r } } \left( g ( X , A ) = 1 | Y = 1 , A = 1 \right) - \mathrm { { P r } } \left( g ( X , \dot { A } ) = 1 | \dot { Y } = 1 , A = 0 \right) | \stackrel { - } { \leq } \delta .$

Intuitively, a classifier satisfies demographic parity whenever it classifies approximately the same number of individuals from different groups as positive. However, this does not account for the error incurred within each group. The equal opportunity constraint overcomes this by enforcing approximately equal true positive rates between the two groups. Zeng et al. [5] recently characterized the Bayes optimal δ-fair classifiers whenever the fairness metric can be expressed as a linear function of a classifier g. We introduce their definition below.

Definition 3.3. A fairness disparity measure $\Delta ( g )$ is linear if there exists a $w : \mathcal { X } \times \mathcal { A }  \mathbb { R }$ such that for all classifiers $\begin{array} { r } { g , \Delta ( g ) \stackrel { - } { = } \int _ { A } \int _ { \mathcal X } g ( x , a ) \stackrel { } { w ( x , a ) } \overline { { d \mathbb P ( x , a ) } } } \end{array}$ . Furthermore, it is bilinear if there exist distribution-dependent quantities $s ,$ b such that $w ( x , a ) = s \cdot \eta ( x , a ) + b$

Remark 3.4. Proposition 3.4 in Zeng et al. [5] shows the suitable choices of s and b for the fairness constraints mentioned in Definition 3.2. For demographic parity, $\begin{array} { r } { w _ { D P } ( x , a ) = \frac { ( 2 a - 1 ) } { \pi _ { a } } } \end{array}$ and for equal opportunity, $\begin{array} { r } { w _ { E O P } ( x , a ) = \frac { ( 2 a - 1 ) \eta ( x , a ) } { \pi _ { 1 a } } } \end{array}$

We will now define the risk of a classifier g and its disparity for linear disparity metrics [5].

Definition 3.5. (Equation A.1 and 4.2 from [5]) For a classifier g, we can define the risk $R ( g )$ and disparity $\Delta ( g )$ as: $\begin{array} { r l r } { R ( g ) } & { = } & { \sum _ { a } \pi _ { a } \int _ { \mathcal X } ( 1 - 2 \eta ( x , a ) ) g ( x , a ) d \mathbb P _ { X | A = a } ( x ) + } \end{array}$ a∈A $\begin{array} { r l } & { \displaystyle \sum _ { a \in \mathcal { A } } \pi _ { a } \int _ { \mathcal { X } } \eta ( x , a ) d \mathbb { P } _ { X | A = a } ( x ) \mathrm { , a n d ~ } \Delta ( g ) = \displaystyle \sum _ { a \in \mathcal { A } } \pi _ { a } \int _ { \mathcal { X } } \left[ w ( x , a ) \cdot g ( x , a ) \right] d \mathbb { P } _ { X | A = a } ( x ) . } \end{array}$

Because we are working with binary classification, introducing the class posterior probability allows us to decompose the risk into purely classifier-dependent and distribution-dependent terms. We now introduce the optimal fair classification problem. For at most δ-unfairness, we want to solve the following optimization problem: $g _ { \delta } ~ \in ~ \arg \operatorname* { m i n } R ( g )$ subject to $| \Delta ( g ) | \le \delta$ . For demographic g parity and equal opportunity, we can precisely write down the fair Bayes optimal classifier that solves this optimization problem, where the threshold depends on the distribution.

Lemma 3.6. (Proposition 2.3 in [28], Theorem 4.2 in [5]) Under Assumption 2.1, the Bayes optimal δ-fair classifier takes the following form: $\begin{array} { r } { g _ { \delta } ( x , a ) = \mathbb { I } \left( \eta ( x , a ) \ge \frac { 1 } { 2 } + \frac { \lambda ^ { * } ( \delta ) } { 2 } w ( x , a ) \right) } \end{array}$ , where $\lambda ^ { * } ( \delta )$ is the optimal dual parameter for the optimization program.

Remark 3.7. Using Remark 3.4, we can immediately write the threshold characterizations for demographic parity: $\begin{array} { r } { g _ { \delta } ( x , a ) = \mathbb { I } \left( \eta ( x , a ) \ge \frac { 1 } { 2 } + \frac { ( 2 a - 1 ) \lambda ^ { * } ( \delta ) } { 2 \pi _ { a } } \right) } \end{array}$ and equal opportunity: $g _ { \delta } ( x , a ) =$ $\begin{array} { r } { \mathbb { I } \left( \eta ( x , a ) \ge \frac { 1 } { 2 - \frac { \left( 2 a - 1 \right) \lambda ^ { * } \left( \delta \right) } { \pi _ { 1 a } } } \right) } \end{array}$

Lemma 3.6 implies that for a fixed $\delta ,$ there may exist a $\lambda ^ { * }$ (that depends on whether for a given $\delta ,$ finding a $\lambda ^ { * } ( \delta )$ is feasible), with which we can adjust the unconstrained Bayes optimal threshold of $^ 1 / 2$ . This correction depends on which group $A \ = \ a$ is underprivileged, as well as on the distribution-dependent quantities like $\pi _ { a } , \pi _ { y a }$

Characterizing the precise value of $\lambda ^ { * }$ as a function of δ and distribution-dependent quantities is a challenging problem, and several prior works $( \mathbf { e } . \mathbf { g } . , [ 1 1 , 3 0 , 2 8 , 5 ] )$ treat it as another optimization problem. However, in our approach, we will show how we can obtain a feasible range for λ. By doing so, we will first be able to parametrize the risk and the unfairness in terms of λ, and then use a bisection search to get arbitrarily close to the required level of unfairness $\delta .$ In a similar spirit to the Bayes error estimation work in the literature [6, 13, 14], this will allow us to achieve the theoretically optimal fairness-accuracy tradeoff using only soft labels.

Let $g _ { \lambda } = \mathop { \mathrm { a r g m } } _ { g }$ in $R ( g ) + \lambda \cdot \Delta ( g )$ , for $\lambda \in \mathbb { R }$ denote the primal solution for a fixed λ. We dub this

as the λ-fair classifier, since this can potentially be a $\delta ( \lambda )$ -fair classifier. We can now write down the Bayes optimal λ-fair classifier using the threshold characterization in Lemma 3.6, which will be useful later in this paper. We now prove how distribution-dependent quantities bound λ.

Lemma 3.8. Assuming $w ( . , 0 ) < 0$ and $w ( . , 1 ) > 0$ (Definition 3.3), when $\Delta ( g _ { 0 } ) > 0 , \lambda _ { \mathrm { m a x } } =$ $\frac { 1 } { \operatorname* { i n f } _ { ( x , a ) } | w ( x , a ) | }$ , so that $\lambda \in [ 0 , \lambda _ { \operatorname* { m a x } } ]$ is sufficient to interpolate the entire Pareto frontier, including the λ for exact fairness $\lambda ^ { * } ( 0 )$ . And when $\begin{array} { r } { \Delta ( g _ { 0 } ) < 0 , \lambda _ { \mathrm { m i n } } = \frac { - 1 } { \underset { ( x , a ) } { \operatorname* { i n f } } | w ( x , a ) | } } \end{array}$ , so that $\lambda \in [ \lambda _ { \operatorname* { m i n } } , 0 ]$ , is sufficient to interpolate the entire Pareto frontier.

This condition becomes much simpler for the equal opportunity and the demographic parity fairness metrics: $\lambda _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { a \in \mathcal { A } } \pi _ { 1 a }$ and $\lambda _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { a \in \mathcal { A } } \pi _ { a }$ , respectively. Restricting the possible range of λ allows us to set up a bisection search to estimate the fairness-accuracy tradeoff (Algorithm 1). We now argue that approximating this tradeoff curve pointwise is sufficient for our scenario to obtain a high-probability estimate of the true fairness-accuracy tradeoff.

Proposition 3.9. Ifa classifier g belongs to the Pareto-frontier, then $g = g _ { \lambda } , f o r$ some $\lambda \in \mathbb { R }$

In the Appendix, we show that $g _ { \lambda }$ can be constructed as a threshold classifier of the form $g _ { \lambda } ( x , a ) =$ $\mathbb { I } \left( \eta ( x , a ) \ge \alpha _ { a , \lambda } \right)$ , where the threshold $\alpha _ { a , \lambda }$ is λ and distribution dependent (Remark 3.7).

## 4 Estimating the Unfairness and Error Rates Using Soft Labels

With all the tools in place, we will now derive expressions for the expected and finite-sample estimators. We will first derive consistent estimators of the cost-sensitive thresholds $\alpha _ { a }$ that depend on $\lambda ,$ the unfairness metric of choice, and some distribution-dependent quantities in Theorem 4.1. Once we have that, we will first show a consistent estimator for the cost-sensitive Bayes risk, in a similar spirit to the results in prior work [6, 14] in Theorem 4.3 and then its finite sample version in Theorem 4.4. Since we are interested in estimating the fairness-accuracy tradeoff, we will then derive consistent estimators for the risk of a classifier (Theorem 4.5) and then the unfairness (Theorem 4.6). Finally, we will present an algorithm to construct the tradeoff curve (Algorithm 1 in the Appendix).

## 4.1 Estimating Fair Thresholds using Finite Soft Labels

Unlike prior work [6, 31, 13, 14], where soft labels were enough to estimate the Bayes risk $\beta ,$ in the case of any cost-sensitive risk $\beta _ { C S }$ , unless the cost $\alpha _ { a }$ is a constant, we need to estimate $\alpha _ { a }$ from given soft labels, since $\alpha _ { a }$ would often depend on distribution dependent quantities. This is certainly true for the fairness constraints defined in Lemma 3.6. We now define the estimated thresholds for both DP and EOP fairness constraints (Lemma 3.6 and Remark 3.7). For demographic parity: $\hat { \alpha } _ { a } ^ { + } = \Pi ( \tilde { \alpha } _ { a } ^ { + } )$ , where $\begin{array} { r } { \tilde { \alpha } _ { a } ^ { + } = \frac { 1 } { 2 } + \frac { ( 2 a - 1 ) \lambda } { 2 \hat { \pi } _ { a } ^ { + } } } \end{array}$ , and for equal opportunity: $\hat { \alpha } _ { a } ^ { + } = \Pi ( \tilde { \alpha } _ { a } ^ { + } )$ ), where $\tilde { \alpha } _ { a } ^ { + } =$ $\frac { 1 } { 2 - \frac { ( 2 a - 1 ) \lambda } { \hat { \pi } _ { 1 a } ^ { + } } }$ , where Π is the truncation operation defined earlier, that ensures that our estimated

thresholds stay in the interval [0, 1].

We now show that the estimators of $\alpha _ { a }$ for DP and EOP, with finitely many soft labels, are statistically consistent. Since the specific construction of the estimator requires knowledge of $w ( X , A )$ , we present estimators for demographic parity and equal opportunity, and the proof techniques can be generalized to tackle other cost-sensitive applications.

Theorem 4.1. (Consistency and Sample Complexity for threshold estimation) Given a dataset $D _ { \alpha }$ of clean soft labels $( c _ { i } = \eta ( x _ { i } , a _ { i } ) ,$ and afixed $\bar { \lambda } \in \mathbb { R }$ , consider the empirical estimator $\hat { \alpha } _ { a } ^ { + }$ . Assuming $\begin{array} { r } { 2 - \frac { ( 2 a - 1 ) \lambda } { \pi _ { 1 a } } \ge \tau > 0 } \end{array}$ (denominator bounded away from zero), the following holds:

1. Consistency: $\hat { \alpha } _ { a } ^ { + } \overset { p } { \to } \alpha _ { a }$

2. Bias:

$$
\begin{array} { r } { \bullet D P \colon | \alpha _ { a } - \mathbb { E } _ { D _ { \alpha } } \left[ \widehat { \alpha } _ { a } ^ { + } \right] | \leq \frac { | \lambda | } { \pi _ { a } } \sqrt { \frac { 1 - \pi _ { a } } { \pi _ { a } N _ { \alpha } } } + 2 \exp ( - \frac { \pi _ { a } ^ { 2 } } { 2 } N _ { \alpha } ) , } \end{array}
$$

$\begin{array} { r l r } { E O P ; ~ | \alpha _ { a } - \mathbb { E } _ { D _ { \alpha } } [ \hat { \alpha } _ { a } ^ { + } ] | } & { \le } & { { \frac { 4 | \lambda | } { \tau ^ { 2 } \pi _ { 1 , a } ^ { 2 } } } \sqrt { \frac { \pi _ { 1 a } ( 1 - \pi _ { 1 a } ) } { N _ { \alpha } } } + 2 \exp ( - 2 \epsilon ^ { 2 } N _ { \alpha } ) } \end{array}$ , where $\epsilon \quad \leq$ min $\bigl \{ \frac { \pi _ { 1 a } } { 2 } , \frac { \tau \pi _ { 1 a } ^ { 2 } } { 4 | \lambda | } \bigr \}$

3. Sample Complexity: With $1 - \delta$ probability, Pr $\begin{array} { r } { \left( \left| \hat { \alpha } _ { a } ^ { + } - \alpha _ { a } \right| > \epsilon \right) \leq \delta } \end{array}$ , as long as:

$$
\begin{array} { r } { \bullet \ D P : N _ { \alpha } > \operatorname* { m a x } \{ \frac { 2 } { \pi _ { a } ^ { 2 } } \log \frac { 4 } { \delta } , \frac { \lambda ^ { 2 } } { 2 \epsilon ^ { 2 } \pi _ { a } ^ { 4 } } \log \frac { 4 } { \delta } , \frac { 2 } { \pi _ { a } } \} , } \end{array}
$$

$$
\begin{array} { r } { \bullet { \cal E } O P ; N _ { \alpha } \geq \frac { 1 } { 2 t ^ { 2 } } \log ( \frac { 2 } { \delta } ) f o r t = \operatorname* { m i n } \Big \{ \frac { \pi _ { 1 a } } { 2 } , \frac { \tau \pi _ { 1 a } ^ { 2 } } { 4 | \lambda | } , \frac { \epsilon \tau ^ { 2 } \pi _ { 1 a } ^ { 2 } } { 4 | \lambda | } \Big \} . } \end{array}
$$

Remark 4.2. It is possible to remove the dependency of λ in the sample complexity results by using Lemma 3.8 that tells us about the maximum $\lambda ,$ and study the worst-case sample complexity bounds. However, keeping λ-dependent results here highlights the (indirect) polynomial dependency on the unfairness through the correction factor λ. This also indicates that threshold estimation is non-uniform with respect to the underlying unfairness: the number of samples required may change drastically depending on the degree of correction and the relative group proportions in the distribution.

## 4.2 Bayes Error Estimation for Cost-Sensitive Classification

Prior works [6, 14, 13] study an estimator for the Bayes error of a distribution for binary classification, by leveraging a simple observation: the Bayes error (which we can denote with β) can be written down as $\beta = \mathbb { E } _ { X } \left[ \operatorname* { m i n } ( \eta ( x ) , 1 - \eta ( x ) ) \right.$ ]. They also show that $\hat { \beta } = \textstyle \frac { 1 } { N } \sum _ { i \in [ N ] } \operatorname* { m i n } ( \eta _ { i } , 1 - \eta _ { i } )$ is a consistent

estimator of the Bayes risk. In the previous section, we derived the expression for cost-sensitive Bayes risk and the corresponding Bayes optimal classifier (Lemma C.5). Let $\beta _ { C S }$ denote the Bayes error for a distribution with respect to a fixed cost-sensitive risk, with group-dependent costs $\alpha _ { a } .$ As an independent result, and in a similar spirit to Proposition 3.1 of Ishida et al. [6], we now derive the consistency and sample-complexity properties of an estimator for the cost-sensitive Bayes risk.

Theorem 4.3. Consider the Bayes optimal classifierfor the cost-sensitive risk with group-dependent costs $\alpha _ { a } .$

$$
g _ { \alpha } ^ { \ast } ( x , a ) = \left\{ \begin{array} { l l } { 1 , } & { i f \eta ( x , a ) \geq \alpha _ { a } } \\ { 0 , } & { o t h e r w i s e . } \end{array} \right.
$$

1. Bayes $R i s k .$ : The Bayes error with respect to the cost-sensitive risk can be expressed as $\beta _ { C S } = \mathbb { E } _ { X , A } \left[ \operatorname* { m i n } ( ( 1 - \alpha _ { A } ) \eta ( X , A ) , \alpha _ { A } ( 1 - \eta ( X , A ) ) ) \right]$

2. Consistency: $\hat { \beta } _ { C S } = \operatorname * { m } _ { N _ { \beta } } \sum _ { i \in [ N _ { \beta } ] }$ min $( ( 1 - \alpha _ { i } ) c _ { i } , \alpha _ { i } ( 1 - c _ { i } ) ) \stackrel { p } {  } \beta _ { C S }$

3. Sample Complexity: For any $\delta > 0$ , with probability at least $1 - \delta , | \hat { \beta } _ { C S } - \beta _ { C S } | \leq$ $\begin{array} { r } { \frac { 1 } { 4 } \sqrt { \frac { 1 } { N _ { \beta } } \log ( \frac { 2 } { \delta } ) } } \end{array}$

While this result is geared toward group-aware cost-sensitive risks, it provides a roadmap for deriving results similar to prior work $[ 6 , 1 4 , 1 3 ]$ and for generalizing their estimators for Bayes error estimation in cost-sensitive and complex metric applications [15, 16]. We will now proceed to specialize Theorem 4.3 towards fairness applications and construct finite sample estimators using soft labels.

With the fact $\hat { \alpha } _ { a } ^ { + } \overset { p } { \to } \alpha _ { a }$ established in Theorem 4.1, let $\hat { \beta } _ { C S , \hat { \alpha } _ { a } ^ { + } } = \frac { 1 } { N _ { \beta } } \sum _ { i \in [ D _ { \beta } ] } \operatorname* { m i n } ( ( 1 - \hat { \alpha } _ { a } ^ { + } ) c _ { i } , \hat { \alpha } _ { a } ^ { + } ( 1 -$

$c _ { i } ) )$ denote the empirical estimator of the cost-sensitive Bayes error with the empirical estimation of cost-sensitive thresholds $\alpha _ { a }$ . We now show that $\hat { \beta } _ { C S , \hat { \alpha } } \stackrel { p } {  } \beta _ { C S }$

Theorem 4.4. Suppose the dataset $D _ { \alpha }$ with $N _ { \alpha }$ samples is used to construct a consistent estimator of the cost-sensitive threshold $\alpha _ { a }$ (Theorem 4.1). Then, given a dataset $D _ { \beta }$ with $N _ { \beta }$ samples, and assuming $\begin{array} { r } { 2 - \frac { ( 2 a - 1 ) \lambda } { \pi _ { 1 a } } \ge \tau > 0 } \end{array}$ (denominator bounded away from zero), the following statements are true:

1. Consistency: $\hat { \beta } _ { C S , \hat { \alpha } } = \frac { 1 } { N _ { \beta } } \sum _ { i \in [ D _ { \beta } ] } \operatorname* { m i n } ( ( 1 - \hat { \alpha } _ { a _ { i } } ^ { + } ) c _ { i } , \hat { \alpha } _ { a _ { i } } ^ { + } ( 1 - c _ { i } ) ) \stackrel { p } {  } \beta _ { C S } .$

2. Bias: For a fixed $\delta \in ( 0 , 1 )$

• Demographic Parity: $\begin{array} { r l } & { { \left| { \beta _ { C S } } - { { \mathbb { E } } _ { { D _ { \beta } } , { D _ { \alpha } } } } \left[ { { { \hat { \beta } } _ { C S , \hat { \alpha } } } } \right] \right| } = \mathcal { O } \left( { | \lambda | \cdot \sqrt { \frac { { \log ( 1 / \delta ) } } { { { N _ { \alpha } } } } } } \right) } \end{array}$

• Equal Opportunity: $\begin{array} { r } { | \beta _ { C S } - \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ \hat { \beta } _ { C S , \hat { \alpha } } \right] | = \mathcal { O } \left( \frac { | \lambda | } { \tau ^ { 2 } } \sqrt { \frac { \log ( 1 / \delta ) } { N _ { \alpha } } } \right) } \end{array}$

3. Sample Complexity: With 1 − δ probability, $| \beta _ { C S } - \hat { \beta } _ { C S , \hat { \alpha } } | < \epsilon$ as long as $\begin{array} { r } { N _ { \beta } \ge \frac { 1 } { 8 \epsilon ^ { 2 } } } \end{array}$ log( <sup>4</sup><sub>δ</sub> ) and:

• Demographic Parity: $\begin{array} { r } { N _ { \alpha } = \mathcal { O } \left( \frac { \lambda ^ { 2 } } { \epsilon ^ { 2 } } \log \left( \frac { 1 } { \delta } \right) \right) } \end{array}$

• Equal Opportunity: $\begin{array} { r } { N _ { \alpha } = \mathcal { O } \left( \frac { \lambda ^ { 2 } } { \tau ^ { 4 } \epsilon ^ { 2 } } \log \left( \frac { 1 } { \delta } \right) \right) } \end{array}$

Unlike Bayes error estimation [6, 13, 14] where $\alpha _ { a } = 1 / 2$ , due to distribution and unfairness dependent $\alpha _ { a } ,$ we now have biased estimators and sufficient sample requirement that depends on the unfairness correction λ, and for τ that determines how well our denominator is defined.

## 4.3 Expected Error Estimation

However, when considering the fairness-accuracy tradeoff, we are interested in computing the error rate with respect to the expected 0-1 loss. Recall from Definition 3.5 the definition of expected risk with respect to the 0-1 loss: $R ( g ) = \sum _ { a \in \{ 0 , 1 \} } \pi _ { a } \Big ( \mathbb { E } _ { X \mid A = a } \left[ ( 1 - 2 \eta ( X , a ) ) g ( X , a ) \right] \Big ) + \pi =$ $\mathbb { E } _ { X , A } \left[ ( 1 - 2 C ) \mathbb { I } \left( C \geq \alpha _ { A } \right) \right] + \pi . \mathrm { A }$ finite sample estimator of the above risk can be written down as follows: $\hat { R } ( \hat { g } ) = \textstyle \frac { 1 } { N _ { \beta } } \sum _ { i \in [ D _ { \beta } ] } ( 1 - 2 c _ { i } ) \mathbb { I } \left( c _ { i } \ge \hat { \alpha } _ { a _ { i } } ^ { + } \right) + \hat { \pi }$

Theorem 4.5. Consider $\hat { R } ( \hat { g } )$ , the empirical estimator for the expected risk from Definition 3.5. Then, thefollowing properties hold:

1. Consistency: ${ \hat { R } } ( { \hat { g } } ) \ { \overset { p } { \to } } \ R ( g )$

2. Bias: For a fixed $\delta \in ( 0 , 1 )$

• Demographic Parity: $\begin{array} { r } { | R ( g ) - \mathbb { E } _ { D _ { \alpha } , D _ { \beta } } \left[ \hat { R } ( \hat { g } ) \right] | = O \left( | \lambda | \cdot \operatorname* { m a x } \{ L _ { 0 } , L _ { 1 } \} \cdot \sqrt { \frac { \log \left( 1 / \delta \right) } { N _ { \alpha } } } \right) } \end{array}$

• Equal Opportunity: $\begin{array} { r } { | R ( g ) - \mathbb { E } _ { D _ { \alpha } , D _ { \beta } } \left[ \hat { R } ( \hat { g } ) \right] | = \mathcal { O } \left( \frac { | \lambda | \operatorname* { m a x } \{ L _ { 0 } , L _ { 1 } \} } { \tau ^ { 2 } } \sqrt { \frac { \log ( 1 / \delta ) } { N _ { \alpha } } } \right) } \end{array}$

3. Sample Complexity: With 1 − δ probability, $| R ( g ) - \hat { R } ( \hat { g } ) | < \epsilon$ as long as $\begin{array} { r } { N _ { \beta } \ge \frac { 1 8 } { \epsilon ^ { 2 } } \log \left( \frac { 6 } { \delta } \right) } \end{array}$ and:

• Demographic Parity: $\begin{array} { r } { N _ { \alpha } = \mathcal { O } \left( \frac { \lambda ^ { 2 } \operatorname* { m a x } \{ L _ { 0 } ^ { 2 } , L _ { 1 } ^ { 2 } \} } { \epsilon ^ { 2 } } \log \left( \frac { 1 } { \delta } \right) \right) } \end{array}$

• Equal Opportunity: $\begin{array} { r } { N _ { \alpha } = \mathcal { O } \left( \frac { \lambda ^ { 2 } \operatorname* { m a x } \{ L _ { 0 } ^ { 2 } , L _ { 1 } ^ { 2 } \} } { \epsilon ^ { 2 } \tau ^ { 4 } } \log \left( \frac { 1 } { \delta } \right) \right) } \end{array}$

Note that estimating the expectation using the sample set $D _ { \beta }$ is standard. The difficult part is that our underlying group-dependent thresholds are estimates with respect to $D _ { \alpha }$ . This makes estimating risk under constraints more difficult and introduces nontrivial differences from prior results on risk estimation with finite samples [6, 13, 14].

## 4.4 Unfairness Estimation

We will now estimate unfairness using a finite number of soft labels. Since every fairness metric would require different distribution-dependent quantities, we will consider demographic parity and equal opportunity metrics here. To proceed with estimating the unfairness rates, we will need estimators for the True positive rates (TPR) and Positive rates (PR) later in the paper. We will first tackle TPR, and PR can be tackled similarly. A natural estimator for the TPR is to replace all quantities involved with their sample estimates, as done in Jeong et al. [13]. Let $\rho _ { \mathrm { T P R } , a } ( g _ { \alpha } )$ denote the true positive rate for a threshold classifier $g _ { \alpha } ( x , a ) = \mathbb { I } \left( \eta ( x , a ) \geq \alpha _ { a } \right)$ on group $\dot { A } = a$ , which can be expressed as: $\begin{array} { r } { \rho _ { \mathrm { T P R } , a } ( g _ { \alpha } ) = \operatorname* { P r } \left( g _ { \alpha } ( X , A ) = 1 | Y = 1 , A = a \right) = \frac { \mathbb E _ { X \mid A = a } \left[ \mathbb { I } ( \eta ( X , a ) \geq \alpha _ { a } ) \eta ( X , a ) \right] } { \pi _ { 1 \mid a } } } \end{array}$ , and its sample 1|a estimate: $\begin{array} { r } { \hat { \rho } _ { \mathrm { T P R } , a } \big ( g _ { \hat { \alpha } _ { a } ^ { + } } \big ) = \frac { \frac { 1 } { N _ { \beta , a } } \sum _ { i \in [ D _ { \beta , a } ] } \big ( c _ { i } \big ( \mathbb { I } \big ( c _ { i } \geq \hat { \alpha } _ { a } ^ { + } \big ) \big ) } { \hat { \pi } _ { 1 | a } ^ { + } } } \end{array}$ , where $N _ { \beta , a }$ are the number of samples in $D _ { \beta }$ from group $A = a ,$ and $\hat { \alpha } _ { a } ^ { + }$ is the threshold constructed using finite samples.

Once we have our estimators for the TPR, we can define the equal opportunity difference of a classifier g as $\Delta _ { \mathrm { E O P } } ( g ) \ : = \ : \vert \rho _ { \mathrm { T P R } , 1 } ( g ) \ : - \ : \rho _ { \mathrm { T P R } , 0 } ( g ) \vert$ and its empirical counterpart $\hat { \Delta } _ { \mathrm { E O P } } ( g ) =$ $\vert \hat { \rho } _ { \mathrm { T P R } , 1 } ( g ) - \hat { \rho } _ { \mathrm { T P R } , 0 } ( g ) \vert$ . This defines $\Delta _ { \mathrm { E O P } } ( g _ { \alpha } )$ as the true equal opportunity difference with respect to the true threshold $\alpha _ { a }$ and $\hat { \Delta } _ { \mathrm { E O P } } ( g _ { \hat { \alpha } } )$ as the empirical estimate of equal opportunity with respect to the empirical estimate of the threshold.

If we only care about the positive rate (for demographic parity), we can write: $\rho _ { \mathrm { P R } , a } ( g _ { \alpha } ) \ =$ $\begin{array} { r l r } { \operatorname* { P r } \left( g _ { \alpha } ( X , a ) = 1 | A = a \right) } & { { } = } & { \operatorname { \mathbb { E } } _ { X \mid A = a } \left[ \mathbb { I } \left( \eta ( X , a ) \geq \alpha _ { a } \right) \right] } \end{array}$ , and its finite sample estimator: $\begin{array} { r } { \hat { \rho } _ { \mathrm { P R } , a } ( g _ { \hat { \alpha } _ { a } ^ { + } } ) = \frac { 1 } { N _ { \beta , a } } \sum _ { i \in [ D _ { \beta , a } ] } \mathbb { I } \left( c _ { i } \geq \hat { \alpha } _ { a } ^ { + } \right) } \end{array}$ . Similar to EOP, we can write the DP difference as $\Delta _ { D P } ( g ) = \vert \rho _ { \mathrm { P R } , 1 } ( g ) - \rho _ { \mathrm { P R } , 0 } ( g ) \vert$ . Then, the true DP difference on the true threshold is $\Delta _ { D P } ( g _ { \alpha } )$ and the empirical DP difference on the estimated threshold is $\hat { \Delta } _ { D P } ( g _ { \hat { \alpha } } )$

Theorem 4.6. Suppose a dataset $D _ { \beta }$ is used to construct $\hat { \Delta } ( g _ { \hat { \alpha } _ { \alpha } ^ { + } } )$ , where $\hat { \alpha } _ { a } ^ { + }$ is the estimatorfor the DP or the EOP threshold (Theorem 4.1). Then thefollowing holds:

1. Consistency: $\hat { \Delta } ( g _ { \hat { \alpha } _ { \alpha } ^ { + } } ) \stackrel { p } {  } \Delta ( g _ { \alpha } ) .$

2. Bias: For $\delta \in ( 0 , 1 )$

$$
\begin{array} { r } { \bullet \ D P \colon | \Delta _ { D P } ( g ) - \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ \hat { \Delta } _ { D P } ( \hat { g } ) \right] | = \mathcal { O } \left( \frac { 1 } { \sqrt { N _ { \beta } } } + | \lambda | \cdot \operatorname* { m a x } \{ L _ { 0 } , L _ { 1 } \} \sqrt { \frac { \log ( 1 / \delta _ { a } ) } { N _ { \alpha } } } \right) . } \end{array}
$$

$$
\begin{array} { r } { \bullet \operatorname { \cal E O P : } \left| \Delta _ { \mathit { E O P } } ( g ) - \mathbb { E } _ { \mathit { D } _ { \beta } , \mathit { D } _ { \alpha } } \left[ \hat { \Delta } _ { \mathit { E O P } } ( \hat { g } ) \right] \right| = \mathcal { O } \left( \frac { | \lambda | \cdot \operatorname* { m a x } \left\{ L _ { 0 } , L _ { 1 } \right\} } { \tau ^ { 2 } } \sqrt { \frac { \log \left( 1 / \delta \right) } { N _ { \alpha } } } + \frac { 1 } { \sqrt { N _ { \beta } } } \right) . } \end{array}
$$

3. Sample Complexity: With $1 - \delta$ probability, $| \hat { \Delta } ( \hat { g } ) - \Delta ( g ) | \leq \epsilon ,$ , as long as $N _ { \beta } \ =$ $\begin{array} { r } { \mathcal { O } \left( \frac { 1 } { \epsilon ^ { 2 } } \log \left( \frac { 1 } { \delta } \right) \right) } \end{array}$ and:

$$
\begin{array} { r } { \bullet \ D P : N _ { \alpha } = \mathcal { O } \Big ( \frac { \lambda ^ { 2 } \cdot \operatorname* { m a x } \{ L _ { 0 } ^ { 2 } , L _ { 1 } ^ { 2 } \} } { \epsilon ^ { 2 } } \log \big ( \frac { 1 } { \delta } \big ) \Big ) , } \end{array}
$$

$$
\begin{array} { r } { \bullet \operatorname { \mathcal { E } } O P ; N _ { \alpha } = \mathcal { O } \left( \frac { \lambda ^ { 2 } \operatorname* { m a x } \{ L _ { 0 } ^ { 2 } , L _ { 1 } ^ { 2 } \} } { \epsilon ^ { 2 } \tau ^ { 4 } } \log \left( \frac { 1 } { \delta } \right) \right) . } \end{array}
$$

Just like the bias and sample complexity results in Theorem 4.5, the rates of unfairness estimation depend sharply on the unfairness correction and our smoothness assumptions, apart from standard rates [4]. For readability, we subsume the dependence on prior group and subgroup probabilities $( \pi _ { a }$ and $\pi _ { y a } )$ within the order notation. However, the proofs in the Appendix highlight this dependence explicitly, which is useful to study rare and imbalanced group behaviors. We provide a proof sketch, along with the proofs in the Appendix.

Tradeoff Estimation: We use Algorithm 1 (details in the Appendix) to estimate the Pareto frontier using the estimators derived using soft labels in the previous sections. The tradeoff algorithm appeals to the monotonicity and convexity of the Pareto frontier (Lemma C.1 and Proposition 4.6 in [5]) to set up a bisection search. However, since the guarantees for our estimators in previous sections are only derived for a fixed but arbitrary λ, we below note a couple of approaches to derive high-probability guarantees on the estimated Pareto frontier.

Remark 4.7 (On the consistency of the estimated frontier). Our aim is to estimate the fairnessaccuracy tradeoff, estimated pointwise in λ, rather than the risk at a prescribed unfairness level δ: the guarantees of Theorem 4.1 (and Theorems 4.5, 4.6) are stated for a fixed λ. However, we can provide a high-probability guarantee on the ‘correctness’ of the tradeoff: (1) Fixed grid: If the λ values are fixed on a grid $\{ \bar { \lambda _ { 1 } } , \dotsc , \lambda _ { K } \} \subset [ \lambda _ { \operatorname* { m i n } } , \lambda _ { \operatorname* { m a x } } ] ,$ , the search points are data independent, and a union bound over the K points controls the error; (2) Bisection: In its current state, Algorithm 1 selects each λ from the estimated disparity $\hat { \Delta } ,$ which makes the points data-dependent, and a union bound over them is not directly valid. This can be rectified by studying the uniform-in-λ guarantee, sup<sub>λ</sub> $| \hat { \Delta } ( h _ { \lambda } ) - \Delta ( h _ { \lambda } ) |  0$ . Since $\{ h _ { \lambda } \}$ is a low-complexity, one-parameter monotone threshold family, such a bound may be available via a uniform law of large numbers. Bisection is more query-efficient, but its analysis requires controlling the error uniformly in λ.

Note that Algorithm 1 can not only be used to generate an estimate of the Pareto frontier, but can also be used to approximate the risk of the classifier when we desire disparity at most δ. This is because we can always interpolate the curve between the obtained points to get a sense of the sharpness of the tradeoff, along with the bias and sample-complexity bounds in Theorems 4.5 and 4.6. This can be readily used to enforce fairness and choose between fairness metrics, among many applications.

## 5 Experiments

We now present empirical verification of the estimated tradeoff. We work with different sources of soft labels: a synthetic Gaussian setting where we can analytically compute the soft labels [12], an average of hard labels on the CIFAR-10H dataset [32], directly borrowed from prior work, [6, 14] and noisy soft labels derived from a logistic regression model trained on the Adult dataset [20], a popular dataset for unfairness evaluation [7]. Further details of the experiments are provided in the Appendix.

We compare our computed tradeoff against several algorithms that return a tradeoff. We choose the ‘Exp-Grad’ algorithm [33] due to its theoretical and practical strengths [34, 35], Oxonfair [17], Fairret [19], and the MIFPO algorithm [18]. Oxonfair [17] uses validation data to optimize classification thresholds of a pretrained predictor and supports a variety of fairness metrics. MIFPO (Model-Independent Fairness-Performance Optimization) [18] proposes a discrete optimization method to compute the fairness-accuracy tradeoff using soft scores from calibrated classifiers. Finally, Fairret [19] proposes differentiable regularizers as proxies for any linear-fractional form of fairness metrics [30]. All methods return a tradeoff for both fairness metrics, except MIFPO [18], which returns a tradeoff only for DP. For a fair comparison, we ensure that MIFPO uses the same soft scores we do, rather than recomputing them. Note that our estimated tradeoff curve relies on bisection search and hence concentrates more around the unfairness value of 0, which is our desired target unfairness.

In Figures 1 and 2, we show the results for the Gaussian, CIFAR-10H, and the Adult datasets. Averaged across 5 runs, we see that our estimate of the Pareto frontier consistently lower bounds those of all other methods and exhibits the least variance. A peculiar observation is from the demographic parity results, where MIFPO seems to be a lower bound for the method in almost all cases. This is where we would like to highlight how MIFPO operates. Given a dataset, MIFPO aims to optimize for a lower bound that would almost always lower bound any Pareto frontier obtained on the data (Section 4 in [18]). However, for MIFPO, there are no guarantees that we are optimizing for the Bayes optimal fairness-accuracy tradeoff. The MIFPO algorithm does not optimize the true tradeoff curve and instead relies on convex programming to solve for a proxy objective. Hence, comparing algorithms against MIFPO may always yield a large gap, and such comparisons are not indicative of the true gap.

Finally, note that our method at λ = 0 corresponds to the Bayes error, as indicated by the intersection of the dotted black lines in Figures 1 and 2. Note that our plots also indicate that many fairness tradeoff estimation techniques fail to remain close to the Bayes error when their objectives don’t require fairness-constraining solutions. This might indicate that these algorithms are overfitting, similar to the phenomena observed in prior works on Bayes error estimation [6, 14].

## 6 Conclusion

In this work, we set out to extend Bayes error estimation techniques towards estimating the fairnessaccuracy Pareto frontier. We do so by generalizing the instance-free soft-label estimation techniques from prior work [6, 13, 14] for cost-sensitive risks. Since fairness constraints themselves can be treated as a cost-sensitive learning problem, and given the properties of the fairness-accuracy Pareto front, we can set up consistent estimators for the fair thresholds and for risk and unfairness estimates. Empirically, we can also lower-bound many recent algorithms to obtain the fairness-accuracy frontier.

Our guarantees show that, even with clean soft scores, the problem remains challenging due to strong dependence on the distribution’s priors and the unfairness-dependent term λ. Recent work generalizes Bayes error estimation with soft labels to the case of monotonic noise in clean soft scores [14], and extending our estimators to such noisy setups and beyond is an immediate future work. One can also study other models of weak supervision in the Bayes optimal settings, such as positive-confidence (PC) data [36], and positive-unlabeled (PU) data [37]. Generalizing to these settings has attracted attention [38, 6], and fine-tuning our theoretical framework for them is an interesting future direction. To move towards group-blind and sensitive attribute-unaware estimation, we can either use group-blind optimal characterizations (Proposition E.1 in [28]) or probabilistic predictors of sensitive attributes [39, 40, 41, 42], at the cost of increased sample complexity of the underlying estimators. Finally, our guarantees and approach also serve as a template to approach multi-objective problems where the standard Bayes optimal classifier may not align with the optimal predictor for complex metrics [15, 43, 16] and broader applications [44, 45].

![](images/800f3826cd12a2309cbcb0fde399b35755a9be2a293ad5901434b8de82726f81.jpg)

![](images/bfd70ef852af9c2d364b2ad55816f57334088fcfbb184f6de27db2a93f19f056.jpg)

![](images/76354d5a972a4f118b645fe1e33eca3852af6ba982d2bc19fee8c820b0e274ab.jpg)

![](images/a5c07e7c539fe7fed10d45ce941393c43b4123fdbc242a54a423ae3b6a9edabd.jpg)  
Figure 2: Unfairness-error tradeoffs on Cifar-10H and the Adult Dataset, for Demographic Parity (DP) and Equal Opportunity (EOP) constraints. The proposed estimate (green) establishes a lower bound against competing algorithms, except MIFPO (red), which computes a high-variance data-dependent lower bound different from the Bayes optimal. Dotted lines denote the unconstrained Bayes Optimal classifier at λ = 0.

## Acknowledgments

The authors would like to thank the reviewers for a positive and constructive exchange during the review process. Takashi Ishida was supported by KAKENHI Grant Number 22K17946. Rajiv Ratn Shah would like to thank IIIT Delhi, where he has formerly been an associate professor. Masashi Sugiyama was supported by JST ASPIRE Grant Number JPMJAP25B1.

## References

[1] Thomas Cover and Peter Hart. Nearest neighbor pattern classification. IEEE transactions on information theory, 13(1):21–27, 1967.

[2] Keinosuke Fukunaga. Introduction to statistical pattern recognition. Elsevier, 2013.

[3] Luc Devroye, László Györfi, and Gábor Lugosi. A probabilistic theory of pattern recognition, volume 31. Springer Science & Business Media, 2013.

[4] M Mohri, A Rostamizadeh, and A Talwalkar. Foundations of machine learning second edition. In Google Books. MIT Press, 2018.

[5] Xianli Zeng, Guang Cheng, and Edgar Dobriban. Bayes-optimal fair classification with linear disparity constraints via pre-, in-, and post-processing. arXiv preprint arXiv:2402.02817, 2024.

[6] Takashi Ishida, Ikko Yamane, Nontawat Charoenphakdee, Gang Niu, and Masashi Sugiyama. Is the performance of my deep network too good to be true? a direct approach to estimating the bayes error in binary classification. In The Eleventh International Conference on Learning Representations.

[7] Solon Barocas, Moritz Hardt, and Arvind Narayanan. Fairness and machine learning: Limitations and opportunities. MIT press, 2023.

[8] Julia Angwin, Jeff Larson, Surya Mattu, and Lauren Kirchner. Machine bias. ProPublica, May, 23(2016):139–159, 2016.

[9] Joy Buolamwini and Timnit Gebru. Gender shades: Intersectional accuracy disparities in commercial gender classification. In Conference on fairness, accountability and transparency, pages 77–91. PMLR, 2018.

[10] Geoff Pleiss, Manish Raghavan, Felix Wu, Jon Kleinberg, and Kilian Q Weinberger. On fairness and calibration. Advances in neural information processing systems, 30, 2017.

[11] Aditya Krishna Menon and Robert C Williamson. The cost of fairness in binary classification. In Conference on Fairness, accountability and transparency, pages 107–118. PMLR, 2018.

[12] Xianli Zeng, Edgar Dobriban, and Guang Cheng. Bayes-optimal classifiers under group fairness. arXiv preprint arXiv:2202.09724, 2022.

[13] Minoh Jeong, Martina Cardone, and Alex Dytso. Data-driven estimation of the false positive rate of the bayes binary classifier via soft labels. In 2024 IEEE International Symposium on Information Theory (ISIT), pages 368–373. IEEE, 2024.

[14] R. Ushio, T. Ishida, and M. Sugiyama. Practical estimation of the optimal classification error with soft labels and calibration. In Proceedings of Fourteenth International Conference on Learning Representations (ICLR2026), page to appear, Rio de Janeiro, Brazil, Apr. 23–27 2026.

[15] Oluwasanmi O Koyejo, Nagarajan Natarajan, Pradeep K Ravikumar, and Inderjit S Dhillon. Consistent binary classification with generalized performance metrics. Advances in neural information processing systems, 27, 2014.

[16] Shashank Singh and Justin Khim. Optimal binary classification beyond accuracy. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022.

[17] Eoin Delaney, Zihao Fu, Sandra Wachter, Brent Mittelstadt, and Chris Russell. Oxonfair: A flexible toolkit for algorithmic fairness. Advances in Neural Information Processing Systems, 37:94209–94245, 2024.

[18] Mark Kozdoba, Binyamin Perets, and Shie Mannor. Efficient fairness-performance pareto front computation. arXiv preprint arXiv:2409.17643, 2024.

[19] Maarten Buyl, MaryBeth Defrance, and Tijl De Bie. Fairret: a framework for differentiable fairness regularization terms. arXiv preprint arXiv:2310.17256, 2023.

[20] Moshe Lichman et al. Uci machine learning repository, 2013.

[21] Visar Berisha, Alan Wisler, Alfred O Hero, and Andreas Spanias. Empirically estimable classification bounds based on a nonparametric divergence measure. IEEE Transactions on Signal Processing, 64(3):580–591, 2015.

[22] Morteza Noshad, Li Xu, and Alfred Hero. Learning to benchmark: Determining best achievable misclassification error from training data. arXiv preprint arXiv:1909.07192, 2019.

[23] Ryan Theisen, Huan Wang, Lav R Varshney, Caiming Xiong, and Richard Socher. Evaluating state-of-the-art classification models against bayes optimality. Advances in Neural Information Processing Systems, 34:9367–9377, 2021.

[24] Cedric Renggli, Luka Rimanic, Nora Hollenstein, and Ce Zhang. Evaluating bayes error estimators on real-world datasets with feebee. arXiv preprint arXiv:2108.13034, 2021.

[25] Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Richard Zemel. Fairness through awareness. In Proceedings of the 3rd Innovations in Theoretical Computer Science (ITCS) Conference, page 214–226. Association for Computing Machinery, 2012.

[26] Moritz Hardt, Eric Price, and Nati Srebro. Equality of opportunity in supervised learning. Advances in neural information processing systems, 29:3315–3323, 2016.

[27] Blake Woodworth, Suriya Gunasekar, Mesrob I Ohannessian, and Nathan Srebro. Learning non-discriminatory predictors. In Conference on Learning Theory, pages 1920–1953. PMLR, 2017.

[28] Evgenii Chzhen, Christophe Denis, Mohamed Hebiri, Luca Oneto, and Massimiliano Pontil. Leveraging labeled and unlabeled data for consistent fair binary classification. Advances in Neural Information Processing Systems, 32, 2019.

[29] Drona Khurana, Srinivasan Ravichandran, Sparsh Jain, and Narayanan Unny Edakunni. Statistical guarantees for fairness aware plug-in algorithms. arXiv preprint arXiv:2107.12783, 2021.

[30] L Elisa Celis, Lingxiao Huang, Vijay Keswani, and Nisheeth K Vishnoi. Classification with fairness constraints: A meta-algorithm with provable guarantees. In Proceedings of the conference on fairness, accountability, and transparency, pages 319–328, 2019.

[31] Minoh Jeong, Martina Cardone, and Alex Dytso. Demystifying the optimal performance of multi-class classification. Advances in Neural Information Processing Systems, 36:31638– 31664, 2023.

[32] Joshua C Peterson, Ruairidh M Battleday, Thomas L Griffiths, and Olga Russakovsky. Human uncertainty makes classification more robust. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9617–9626, 2019.

[33] Alekh Agarwal, Alina Beygelzimer, Miroslav Dudík, John Langford, and Hanna Wallach. A reductions approach to fair classification. In International Conference on Machine Learning, pages 60–69. PMLR, 2018.

[34] Mohit Sharma, Amit Deshpande, and Rajiv Ratn Shah. On comparing fair classifiers under data bias. arXiv preprint arXiv:2302.05906, 2023.

[35] Avijit Ghosh, Pablo Kvitca, and Christo Wilson. When fair classification meets noisy protected attributes. In Proceedings of the 2023 AAAI/ACM Conference on AI, Ethics, and Society, pages 679–690, 2023.

[36] Takashi Ishida, Gang Niu, and Masashi Sugiyama. Binary classification from positiveconfidence data. Advances in neural information processing systems, 31, 2018.

[37] Marthinus C Du Plessis, Gang Niu, and Masashi Sugiyama. Analysis of learning from positive and unlabeled data. Advances in neural information processing systems, 27, 2014.

[38] Ziwei Wu and Jingrui He. Fairness-aware model-agnostic positive and unlabeled learning. In Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency, pages 1698–1708, 2022.

[39] Emily Diana, Wesley Gill, Michael Kearns, Krishnaram Kenthapadi, Aaron Roth, and Saeed Sharifi-Malvajerdi. Multiaccurate proxies for downstream fairness. In Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency, pages 1207–1239, 2022.

[40] Flavien Prost, Pranjal Awasthi, Nick Blumm, Aditee Kumthekar, Trevor Potter, Li Wei, Xuezhi Wang, Ed H Chi, Jilin Chen, and Alex Beutel. Measuring model fairness under noisy covariates: A theoretical perspective. In Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society, pages 873–883, 2021.

[41] Beepul Bharti, Paul Yi, and Jeremias Sulam. Estimating and controlling for equalized odds via sensitive attribute predictors. Advances in neural information processing systems, 36:37173– 37192, 2023.

[42] Beepul Bharti, Mary Versa Clemens-Sewall, Paul H Yi, and Jeremias Sulam. Multiaccuracy and multicalibration via proxy groups. arXiv preprint arXiv:2503.02870, 2025.

[43] Andrew Cotter, Heinrich Jiang, Maya Gupta, Serena Wang, Taman Narayan, Seungil You, and Karthik Sridharan. Optimization with non-differentiable constraints with applications to fairness, recall, churn, and other goals. Journal of Machine Learning Research, 20(172):1–59, 2019.

[44] Peter Súkeník and Christoph Lampert. Generalization in multi-objective machine learning. Neural Computing and Applications, 37(30):24669–24683, 2025.

[45] Tobias Wegel, Geelon So, Junhyung Park, and Fanny Yang. On the sample complexity of semi-supervised multi-objective learning. Advances in Neural Information Processing Systems, 38:128421–128476, 2026.

[46] Hal R. Varian. Two problems in the theory of fairness. Journal of Public Economics, 5(3- 4):249–260, None 1976.

[47] Charles Elkan. The foundations of cost-sensitive learning. In International joint conference on artificial intelligence, volume 17, pages 973–978. Lawrence Erlbaum Associates Ltd, 2001.

[48] Clayton Scott, Gilles Blanchard, and Gregory Handy. Classification with asymmetric label noise: Consistency and maximal denoising. In Conference on learning theory, pages 489–511. PMLR, 2013.

[49] Felix Petersen, Debarghya Mukherjee, Yuekai Sun, and Mikhail Yurochkin. Post-processing for individual fairness. Advances in Neural Information Processing Systems, 34, 2021.

## A Notations and Assumptions

Our data domain is defined over ${ \mathcal { X } } \times { \mathcal { Y } } \times { \mathcal { A } }$ , where $\mathcal { X } \subset \mathbb { \backslash } \mathbb { R } ^ { d } , \mathcal { y } \mathbf { \textnormal { = } } \{ 0 , 1 \}$ and $\begin{array} { r l } { A } & { { } = } \end{array}$ $\{ 0 , 1 \}$ . We denote the posterior probability of the positive class, given a sample, as $\eta ( x , a ) =$ $\operatorname { \tilde { P r } } \left( \tilde { Y } = 1 | X = x , A = \mathbf { \bar { \alpha } } a \right)$ We define the soft label for an instance $( x _ { i } , a _ { i } )$ as posterior probability: $c _ { i } = \eta ( x _ { i } , a _ { i } )$ , and let $C = \eta ( X , A )$ denote a random variable. Let $\pi _ { a } = \operatorname* { P r } \left( A = a \right)$ $\pi _ { y a } ~ = ~ \operatorname* { P r } \left( Y = y , A = a \right)$ , and $\pi _ { y | a } = \operatorname* { P r } { ( Y = y | A = a ) }$ . By the law of total probability, Pr $( Y = y ) = \sum _ { a \in \mathcal { A } } \pi _ { y a }$ . We assume the existence of all groups and classes, $\mathrm { i . e . , } \pi _ { a } > 0 , \pi _ { y | a } > 0$ for all $y \in \{ 0 , 1 \}$ and for all $a \in A .$

Whenever we work with finite sample approximations, we assume that we are given access to N i.i.d. soft labels of the form $( c _ { i } , a _ { i } )$ , and we will denote our dataset with $D = \{ ( c _ { i } , a _ { i } ) \} _ { i = 1 } ^ { N }$ . Since our approach is model-free, we do not require access to instances or features $x _ { i }$ . We will also partition our dataset D into two parts: $D = D _ { \alpha } \cup D _ { \beta }$ , with $D _ { \alpha } \cap D _ { \beta } = \emptyset$ , where $D _ { \alpha }$ (with $N _ { \alpha }$ samples) will be the data used to estimate the thresholds, and $D _ { \beta }$ (with $N _ { \beta }$ samples) will be used to construct the estimators of the required expectation, with $N = \dot { N } _ { \alpha } + N _ { \beta }$ . The splits are chosen independently of the sample values. Furthermore, we will denote with $N _ { \alpha , a } \mathrm { ~ ( o r ~ } N _ { \beta , a } )$ the number of samples from group $A = a$ . We now assume the following about our soft labels:

Assumption A.1. (No interior atoms / Continuity of the CDF of η) The mapping $t  F _ { a } ( t ) =$ $\operatorname* { P r } \left( \eta ( { \bar { X } } , a ) \leq t | A = a \right)$ , ∀a ∈ A is continuous in (0, 1). Moreover, $\operatorname* { P r } \left( \eta ( X , a ) \geq 1 / 2 | A = a \right) > 0$ for all $a \in \{ 0 , 1 \}$

For later results, we would also need a bounded density assumption on our group-conditioned soft labels.

Assumption A.2. (Lipschitz continuous group CDF) For each $a \in \{ 0 , 1 \}$ , the group-conditional distribution of $C = \eta ( X , a ) | A = a$ admits a density $f _ { a }$ on (0, 1) such that sup $\mathsf { \bar { f } } _ { a } ( \bar { t } ) \le L _ { a } < \infty$ t∈(0,1)

In particular, $F _ { a }$ is L<sub>a</sub>-Lipschitz: $| F _ { a } ( u ) - F _ { a } ( v ) | \le L _ { a } \cdot | u - v |$ for all $u , v \in ( 0 , 1 )$

Let $\begin{array} { r } { \hat { \pi } _ { a } = \frac { 1 } { N _ { \alpha } } \sum _ { i \in [ D _ { \alpha } ] } \mathbb { I } \left( a _ { i } = a \right) } \end{array}$ denote our estimator for $\pi _ { a } . \mathrm { \bf B y }$ the Weak Law of Large Numbers, $\hat { \pi } _ { a } \overset { p } {  } \pi _ { a }$ . Furthermore, the bias of the estimator $\hat { \pi } _ { a }$ is zero, which is easy to see via the linearity of expectation. We can derive a similar estimator for $\pi = \operatorname* { P r } \left( Y = 1 \right)$ as $\begin{array} { r } { \hat { \pi } \stackrel {  } { = } \frac { 1 } { N _ { \alpha } } \sum _ { i \in [ D _ { \alpha } ] } c _ { i } } \end{array}$

Similarly, let πˆ $\begin{array} { r } { \mathsf { \Omega } _ { 1 a } = \frac { 1 } { N _ { \alpha } } \sum _ { i \in [ D _ { \alpha } ] } \mathbb { I } \left( a _ { i } = a \right) c _ { i } } \end{array}$ denote our estimator for $\pi _ { 1 a }$ . By the Weak Law of Large Numbers, $\hat { \pi } _ { 1 a } \overset { p } {  } \pi _ { 1 a }$ . Furthermore, the bias of the estimator $\hat { \pi } _ { 1 a }$ is zero, which is easy to see via the linearity of expectation. The same analysis works out for the unbiased estimator ${ \hat { \pi } } _ { 1 | a } =$ $\begin{array} { r } { \frac { 1 } { N _ { \alpha , a } } \sum _ { i \in [ D _ { \alpha } ] } \mathbb { I } \left( a _ { i } = a \right) c _ { i } } \end{array}$ and its expected value $\pi _ { 1 | a }$ . Since we will work with functions of $\hat { \pi } _ { a }$ (and other estimators) that may have inverses of $\hat { \pi } _ { a } .$ , we also define another estimator $\begin{array} { r } { \hat { \pi } _ { a } ^ { + } = \operatorname* { m a x } \{ \hat { \pi } _ { a } , \frac { 1 } { N _ { \alpha } } \} } \end{array}$ which is still statistically consistent, but a biased estimator (same for $\hat { \pi } _ { 1 a } ^ { + }$ and ${ \hat { \pi } _ { 1 | a } ^ { + } } )$

We would also require estimators for the True positive rates (TPR) and Positive rates (PR) later in the paper. We will first tackle TPR, and PR can be tackled similarly. A natural estimator for the TPR is to replace all quantities involved with their sample estimates, as done in (author?) [13]. Let $\rho _ { \mathrm { T P R } , a } ( g _ { \alpha } )$ denote the true positive rate for a threshold classifier $g _ { \alpha } ( x , a ) = \mathbb { I } \left( \eta ( x , a ) \geq \alpha _ { a } \right)$ on group $A = a .$

$$
\begin{array} { r l } & { \rho _ { \mathrm { T P R } , a } ( g _ { \alpha } ) = \operatorname* { P r } \left( g _ { \alpha } = 1 | Y = 1 , A = a \right) = \frac { \mathbb { E } _ { X | A = a } \left[ \mathbb { I } \left( \eta ( x , a ) \geq \alpha _ { a } \right) \cdot \operatorname* { P r } \left( Y = 1 | X = x , A = a \right) \right] } { \pi _ { 1 | a } } } \\ & { \qquad = \frac { \mathbb { E } _ { X | A = a } \left[ \mathbb { I } \left( C \geq \alpha _ { a } \right) \cdot C \right] } { \pi _ { 1 | a } } . } \end{array}
$$

We can now define a sample estimate as:

$$
\begin{array} { r } { \hat { \rho } _ { \mathrm { T P R } , a } ( g _ { \hat { \alpha } _ { a } ^ { + } } ) = \frac { \frac { 1 } { N _ { \beta , a } } \sum _ { i \in [ D _ { \beta , a } ] } \left( c _ { i } ( \mathbb { I } \left( c _ { i } \geq \hat { \alpha } _ { a } ^ { + } \right) \right) } { \hat { \pi } _ { 1 | a } ^ { + } } , } \end{array}
$$

where $N _ { \beta , a }$ are the number of samples in $D _ { \beta }$ from group $A = a .$ , and αˆ is the threshold constructed using finite samples or intermediate estimators using finite samples (for example, thresholds for fairness constraints derived in Lemma 3.6).

Once we have our estimators for the TPR, we can define the equal opportunity difference of a classifier g as $\Delta _ { \mathrm { T P R } } ( g ) \ : = \ : \vert \rho _ { \mathrm { T P R } , 1 } ( g ) \ : - \ : \rho _ { \mathrm { T P R } , 0 } ( g ) \vert$ and its empirical counterpart $\hat { \Delta } _ { \mathrm { T P R } } ( g ) =$ $\vert \hat { \rho } _ { \mathrm { T P R } , 1 } ( g ) - \hat { \rho } _ { \mathrm { T P R } , 0 } ( g ) \vert$ . This defines $\Delta _ { \mathrm { T P R } } ( g _ { \alpha } )$ as the true equal opportunity difference with respect to the true threshold $\alpha _ { a }$ and $\hat { \Delta } _ { \mathrm { T P R } } ( g _ { \hat { \alpha } } )$ as the empirical estimate of equal opportunity with respect to the empirical estimate of the threshold.

If we only care about the positive rate (for ensuring demographic parity), we can come up with a natural estimator:

$$
\rho _ { \mathrm { P R } , a } ( g _ { \alpha } ) = \operatorname* { P r } \left( g _ { \alpha } = 1 | A = a \right) = \mathbb { E } _ { X \mid A = a } \left[ \mathbb { I } \left( c \geq \alpha _ { a } \right) \right] ,
$$

and its finite sample estimate:

$$
\hat { \rho } _ { \mathrm { P R } , a } ( g _ { \hat { \alpha } _ { a } ^ { + } } ) = \frac { 1 } { N _ { \beta , a } } \sum _ { i \in [ D _ { \beta , a } ] } \mathbb { I } \left( c _ { i } \geq \hat { \alpha } _ { a } ^ { + } \right) .
$$

Similar to the equal opportunity difference, we can write the demographic parity difference as $\Delta _ { D P } ( g ) = \bar { \rho _ { \mathrm { P R , 1 } } ( g ) } \bar { - } \bar { \rho _ { \mathrm { P R , 0 } } ( g ) }$ |. Then, the true DP difference on the true threshold is $\Delta _ { D P } ( g _ { \alpha } )$ and the empirical DP difference on the estimated threshold is $\hat { \Delta } _ { D P } ( g _ { \hat { \alpha } } )$

## B Helper Results

Lemma B.1. (Finite sample bound for estimation of π) Let $X _ { 1 } , X _ { 2 } , . . . , X _ { N } \in [ 0 , 1 ]$ be N i.i.d. random variables with mean π. Let $\begin{array} { r } { \hat { \pi } = \frac { 1 } { N } \sum _ { i \in [ N ] } X _ { i } } \end{array}$ . For any $\begin{array} { r } { \epsilon > 0 , i f N \geq \frac { 1 } { 2 \epsilon ^ { 2 } } \operatorname { l } \dot { \log } ( \frac { 2 } { \delta } ) } \end{array}$ , then with $1 - \delta$ probability over the choice ofthe data sample, $\hat { \pi } \in ( \pi - \epsilon , \pi + \epsilon )$

Proof. The proof follows directly from the application of Hoeffding’s inequality<sup>2</sup>.

Lemma B.2. (Upper bound on Mean Absolute Error) Consider an estimator πˆ constructed using N samplesfor some quantity π (Lemma B.1). Ifπˆ is an unbiased estimate ofπ, then $\mathbb { E } _ { D _ { N } } \left[ \left| \hat { \pi } - \pi \right| \right] \leq$ $\sqrt { V a r ( \hat { \pi } ) }$

Proof. Let $Z = { \hat { \pi } } - \pi$ . Since πˆ is an unbiased estimate of $\tau , \mathbb { E } _ { D _ { N } } \left[ Z \right] = 0 , V a r ( Z ) = \mathbb { E } _ { D _ { N } } \left[ Z ^ { 2 } \right] =$ $\mathbb { E } _ { D _ { N } } \left\lceil | Z | ^ { 2 } \right\rceil = V a r ( \hat { \pi } )$ . Observe that by Jensen’s inequality, $\mathbb { E } \left[ | Z | ^ { 2 } \right] \geq ( \mathbb { E } \left[ | Z | \right] ) ^ { 2 } \implies \bar { \mathbb { E } } \left[ | Z | \right] \leq$ $\sqrt { \mathbb { E } \left[ | Z | ^ { 2 } \right] }$ , Hence $\mathbb { E } _ { D _ { N } } \left[ | \hat { \pi } - \pi | \right] \leq \sqrt { V a r ( \hat { \pi } ) }$ □

## C Proofs for Section 3

Let g<sub>λ</sub> = argmin $R ( g ) + \lambda \cdot \Delta ( g )$ , for $\lambda \in \mathbb { R }$ denote the primal solution for a fixed λ. We can also g   
dub this the λ-fair classifier, since this can potentially be $\delta ( \lambda )$ -fair classifier. Using this λ, we can also write down the Bayes optimal λ-fair classifier using the threshold characterization in Lemma 3.6, which will be useful later in this paper. We now introduce a key lemma [5] that enables us to study the fairness-accuracy Pareto frontier.

Lemma C.1. (Proposition 4.1 from [5]) As a function of λ, $\Delta ( g _ { \lambda } )$ is monotone non increasing, i.e. $\lambda _ { 1 } < \lambda _ { 2 } \implies \Delta ( g _ { \lambda _ { 1 } } ) \geq \Delta ( g _ { \lambda _ { 2 } } )$ . The risk $R ( g _ { \lambda } )$ is monotone non-increasing on $( - \infty , 0 )$ and monotone non-decreasing on (0, ∞).

Lemma C.1 is incredibly helpful to devise an algorithm to estimate the tradeoff, since it allows us to parametrize the curve with $\lambda ,$ establish the convexity of the tradeoff curve (Proposition 4.6 in [5]), and hence enables the use of bisection search to get close to any level of unfairness required. We will now attempt to bound the range of λ. Along with this information on endpoints and the monotonicity properties of λ from Lemma C.1, this would allow us to set up a bisection search algorithm.

Lemma C.2. (Lemma 3.8 in the main text) Assuming $w ( . , 0 ) < 0$ and $w ( . , 1 ) > 0$ (Definition $3 . 3 ) _ { \cdot }$ when $\begin{array} { r } { D i s ( g _ { 0 } ) > 0 , \lambda _ { \mathrm { m a x } } = \frac { 1 } { \underset { ( x , a ) } { \operatorname* { i n f } } | w ( x , a ) | } } \end{array}$ , so that $\lambda \in [ 0 , \dot { \lambda } _ { \operatorname* { m a x } } ]$ is sufficient to interpolate the entire Paretofrontier, including the λfor exactfairness $\lambda ^ { * } ( 0 )$ . And when $\begin{array} { r } { D i s ( g _ { 0 } ) < 0 , \lambda _ { \mathrm { m i n } } = \frac { - 1 } { \underset { ( x , a ) } { \operatorname* { i n f } } | w ( x , a ) | } , } \end{array}$ so that $\lambda \in [ \lambda _ { \operatorname* { m i n } } , 0 ]$ , is sufficient to interpolate the entire Pareto frontier.

Proof. To construct our bounds on λ, we will try to study the case of exact fairness to obtain the worst possible bounds. Assume WLOG that ${ \cal D } i s \dot { ( } g _ { 0 } ) > 0 .$ . The proof for $D i s ( g _ { 0 } ) < 0$ is similar.

For demographic parity and equal opportunity fairness constraints (Definition 3.3), $v ( . , 0 ) < 0$ and $w ( . , 1 ) > \bar { 0 }$ . Consider the threshold classifier for groups $A = 0$ and $A = 1$ at any λ (Lemma 3.6):

$$
g _ { \lambda } ( x , 0 ) = \mathbb { I } \left( \eta ( x , 0 ) \geq \frac { 1 } { 2 } - \frac { \lambda } { 2 } | w ( x , 0 ) | \right) \quad \mathrm { a n d } \quad g _ { \lambda } ( x , 1 ) = \mathbb { I } \left( \eta ( x , 1 ) \geq \frac { 1 } { 2 } + \frac { \lambda } { 2 } | w ( x , 1 ) | \right) .
$$

Also, by assumption, disparity is positive at $\lambda = 0$ , where disparity is defined as:

$$
D i s ( g _ { \lambda } ) = \pi _ { 1 } \int _ { \chi } | w ( x , 1 ) | \cdot g _ { \lambda } ( x , 1 ) d \mathbb { P } _ { X | A = 1 } ( x ) - \pi _ { 0 } \int _ { \chi } | w ( x , 0 ) | \cdot g _ { \lambda } ( x , 0 ) d \mathbb { P } _ { X | A = 0 } ( x ) > 0 .
$$

From Lemma C.1, we can see that as $\lambda$ increases, the threshold for group $A = 1 .$ , t increases (since $w ( . , 1 ) > 0 )$ and $t _ { 0 }$ decreases (since $w ( . , 0 ) < 0 ) . \mathrm { I f } t _ { 1 } > 1$ , then the $g ( x , 1 ) = 0 { \mathrm { ; } }$ , ∀x, since $\eta ( x , 1 ) \in [ 0 , 1 ]$ . Similarly, if $t _ { 0 } \le 0 , g ( x , 0 ) = 1 , \forall x$

At that point, the disparity $D i s ( g _ { \lambda } )$ at $\lambda = 0$ , which was positive to begin with, becomes negative, and hence we cross over the point of zero unfairness. At such a trivial classification point, $R ( g _ { \lambda } ) > 0$ and $D i s ( g _ { \lambda } ) < 0$ , which is the minimum possible disparity since $D i s ( g ) \in [ - 1 , 1 ]$

It remains to check the value of λ at which the group-aware classifier $g$ becomes trivial for each group. Let such a λ be denoted with $\lambda _ { \mathrm { m a x } }$ . Such a $\lambda _ { \mathrm { m a x } }$ occurs at:

$$
\begin{array} { r l r } & { \mathrm { ( A = 1 ) } \colon \displaystyle \frac 1 2 + \lambda \frac { | w ( x , 1 ) | } 2 > 1 \implies \lambda > \frac 1 { \operatorname* { i n f } _ { x } | w ( x , 1 ) | } \mathrm { , ~ a n d ~ } } & \\ & { \mathrm { ~ } } & { \mathrm { ( A = 0 ) } \colon \displaystyle \frac 1 2 - \lambda \frac { | w ( x , 0 ) | } 2 \leq 0 \implies \lambda \geq \frac 1 { \operatorname* { i n f } _ { x } | w ( x , 0 ) | } } \\ & { \mathrm { O v e r a l l } \colon \implies \lambda > \frac 1 { \operatorname* { i n f } _ { x } | w ( x , a ) | } . } \end{array}
$$

Now, note that due to Lemma C.1, the tradeoff function parametrized by λ is convex, and by increasing λ towards $\lambda _ { \mathrm { m a x } }$ (the disparity point $\operatorname { \mathrm { ~ o f ~ } } - 1 )$ , we would have also crossed the zero disparity point for some $\lambda \in [ 0 , \lambda _ { \operatorname* { m a x } } ]$ . Hence $\lambda \in [ 0 , \lambda _ { \operatorname* { m a x } } ]$ whenever the Bayes optimal classifier has positive disparity $( D i s ( g _ { 0 } ) > 0 )$ ). A similar argument follows for the other case when $D i s ( g _ { 0 } ) < 0$ □

This condition becomes much simpler for the equal opportunity and the demographic parity fairness metrics: $\lambda _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { a \in \mathcal { A } } \pi _ { 1 a }$ and $\lambda _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { a \in \mathcal { A } } \pi _ { a }$ , respectively. Restricting the possible range of λ

allows us to set up a bisection search to estimate the fairness-accuracy tradeoff (Algorithm 1). We now argue that approximating this tradeoff curve pointwise is sufficient for our scenario to obtain a high-probability estimate of the true Pareto frontier.

Definition C.3. Given a risk measure $R ( \cdot )$ and a fairness disparity measure $\Delta ( \cdot )$ , a classifier $g ^ { \prime }$ Pareto-dominates another classifier $^ { g , }$ if $R \dot { ( g ^ { \prime } ) } < R ( g )$ and $| \Delta ( \dot { g } ^ { \prime } ) | \dot { < } | \Delta ( g ) |$ |. The Pareto frontier is the set of those classifiers that are not Pareto-dominated by any other classifier.

Using a well-known characterization of the Pareto frontier by [46], we can equivalently write the above set as $g _ { \lambda } =$ argmin $R ( g ) + \lambda \cdot \Delta ( g )$ . We now show that any point on the Pareto frontier will g

be covered by any curve that we generate by varying the values of λ specified by the range obtained from Lemma 3.8.

Proposition C.4. (Proposition 3.9 in the main text) If a classifier g belongs to the Pareto-frontier, then $g = g _ { \lambda }$ , for some $\lambda \in \mathbb { R }$

Proof. Suppose g belongs to the Pareto-frontier and WLOG assume $\Delta ( g ) \ge 0$ . We know from Proposition 4.6 in [5] that the Pareto front is convex.

Since g lies on the Pareto-frontier, then $( R ( g ) , \Delta ( g ) )$ ) must lie on the boundary of the above convex set. Otherwise, if $( R ( g ) , \Delta ( g ) )$ lies in the interior, there exists a small neighborhood around $( R ( g ) , \Delta ( g ) )$ that is contained in this convex set, and would include $g ^ { \prime }$ such that $\breve { R } ( g ^ { \prime } ) < R ( g )$ and $\begin{array} { r } { 0 < \Delta ( g ^ { \prime } ) < } \end{array}$ $\Delta ( g )$ . Now consider a supporting tangent to the convex set at $( R ( g ) , \dot { \Delta } ( g ) )$ defined by a normal vector $( \alpha , \beta ) \in \mathbb { R } ^ { 2 }$ and margin/shift $\gamma \in \mathbb { R }$ as follows: α $R ( h ) + \beta \Delta ( h ) \geq \gamma$ , for all classifiers $h ,$ , and α $\dot { R } ( \dot { g } ) + \beta \Delta ( g ) = \dot { \gamma }$ . Let $\lambda = \beta / \alpha$ . Then $R ( g ) \dot { + } \lambda \Delta ( g ) \dot { = } R ( g ) + \beta / \alpha \cdot \Delta ( g ) =$ $( \alpha R ( g ) + \beta \Delta ( g ) ) / \alpha = \gamma / \alpha$ , whereas for any other h, we have $R ( h ) + \lambda \Delta ( h ) = \dot { R } ( h ) + \beta / \alpha \cdot \Delta ( h ) =$ $( \alpha R ( h ) + \beta \Delta ( h ) ) / \alpha \geq \gamma / \alpha$ . Therefore, g is a minimizer of $R ( h ) + \lambda g ( h )$ objective over all h, and is itself $g _ { \lambda }$ for $\lambda = \beta / \alpha$

We will show that any $g _ { \lambda }$ can be constructed as a threshold classifier, where the threshold is a function of some distribution-dependent parameters and λ. And then using Lemma 3.8, we will be able to estimate the risk and unfairness of several points on the Pareto frontier by varying the value of λ in a fixed range.

Cost sensitive classification: In classification, we generally work with the 0-1 loss. However, there can be scenarios where we need to weigh different errors unequally, such as the cost of missing a disease (it may be much riskier not to catch a disease than to falsely detect it in healthy individuals) and the cost of granting loans to undeserving candidates. This is the setting of cost-sensitive classification [47].

Lemma C.5. (Extension of[48] to the case ofsubgroups) Let $\alpha \in [ 0 , 1 ]$ . Define the $\alpha { - } C S$ risk as weighing the false negatives for a group $A = a$ with $\left( 1 - \alpha _ { a } \right)$ and thefalse positives with $\alpha _ { a } .$

$$
R _ { \alpha } ( g ) = \sum _ { a \in \mathcal { A } } \left( ( 1 - \alpha _ { a } ) \pi _ { 1 a } \cdot \mathbb { E } _ { X | Y = 1 , A = a } \left[ l _ { 0 1 } ( g ( X , a ) ) \right] + \alpha _ { a } \pi _ { 0 a } \cdot \mathbb { E } _ { X | Y = 0 , A = a } \left[ l _ { 0 1 } ( g ( X , a ) ) \right] \right)
$$

where $\pi _ { y a } = \operatorname* { P r } \left( Y = y , A = a \right) , y \in \{ 0 , 1 \}$ and $l _ { 0 1 }$ is the 0-1 loss that returns 1 whenever the prediction and the true label mismatch. Then, the cost-sensitive group-dependent Bayes optimal classifier can be written down as:

$$
g _ { \alpha } ( x , a ) = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f } } \ \eta ( x , a ) \geq \alpha _ { a } } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }
$$

The proof for Lemma C.5 is a simple group-aware extension of Scott et al. [48]. We can immediately see that the thresholds for Bayes optimal fair classifiers from Lemma 3.6 can be written down as group-dependent costs [11]. For example, for demographic parity, $\begin{array} { r } { \alpha _ { a } = \frac { 1 } { 2 } + \frac { ( 2 a - 1 ) \lambda ^ { * } ( \delta ) } { 2 \pi _ { a } } } \end{array}$ . We will exploit Lemmas 3.8 and C.5 in the coming sections to construct estimators for thresholds, risk, and disparity at arbitrary points up until $\lambda _ { \mathrm { m a x } }$ to get an efficient estimate of the Pareto frontier, only using soft labels.

## D Proofs for Section 4

Proof Sketch: We summarize the main ideas behind the proofs of the DP and EOP threshold (Theorem 4.1), and the risk and unfairness estimators (Theorems 4.4, 4.5, 4.6).

• Consistency: The consistency proofs begin with the Weak Law of Large Numbers for the empirical priors $\hat { \pi } , \hat { \pi } _ { a } , \hat { \pi } _ { y a } .$ , and ${ \hat { \pi } } _ { y \mid a }$ . The threshold estimators $\hat { \alpha } _ { a }$ are then shown to be consistent by applying the Continuous Mapping Theorem $^ 3$ to the corresponding plug-in formulas, together with the truncation operator $\Pi _ { [ 0 , 1 ] }$ . In the EOP case, we additionally require the relevant denominator to stay bounded away from zero. The later consistency results for risk and unfairness then follow by combining the consistency of $\hat { \alpha } _ { a }$ with continuity of the corresponding functionals, along with standard concentration of empirical averages over the independent sample split.

• Bias bounds: For each estimator, we define a suitable good event $\mathcal { G }$ on which the empirical prior of interest is close to its population counterpart. On ${ \mathcal { G } } .$ the estimator error is bounded deterministically in terms of the prior estimation error, using elementary algebraic control of reciprocals whenever needed. Taking expectations then reduces to controlling quantities such as $\mathbb { E } [ | \hat { \pi } - \pi | ]$ , which are bounded using explicit variance calculations and standard inequalities for bounded random variables. On the complement $\mathcal { G } ^ { C }$ , the estimator error is controlled by the uniform boundedness of the estimator, multiplied by $\operatorname* { P r } ( \mathcal G ^ { C } )$ , and $\operatorname* { P r } ( \mathcal { G } ^ { C } )$ is bounded using Hoeffding’s inequality. For the later risk and unfairness results, the threshold estimation error is propagated through the corresponding risk or disparity functional using continuity or Lipschitz bounds under Assumptions 2.1 and 2.2.

• Sample complexity: The sample complexity results are obtained by combining concentration bounds for the threshold estimators built from $D _ { \alpha }$ with concentration bounds for the empirical risk and unfairness estimators built from the independent split $D _ { \beta }$ . The total failure probability is then controlled by a union bound. For the later results, the final sample requirements inherit both the threshold-estimation complexity and the Lipschitz dependence of the downstream risk or disparity estimators on the estimated thresholds.

We now provide proofs for all of the results described in Section 4. For convenience and readability, we will prove the DP and EOP results separately whenever applicable.

Theorem D.1. (DP resultsfrom Theorem 4.1 in the main text) (Consistency and Sample Complexity for DP threshold) Given a dataset $D _ { \alpha }$ of clean soft labels $( c _ { i } = \eta ( x _ { i } , a _ { i } ) )$ and a fixed $\lambda \in \mathbb { R }$ for the case of Demographic Parity, consider the following estimator of the threshold: $\hat { \alpha } _ { a } ^ { + } \ =$ $\Pi ( \tilde { \alpha } _ { a } ^ { + } )$ , where $\begin{array} { r } { \tilde { \alpha } _ { a } ^ { + } = \frac { 1 } { 2 } + \frac { ( 2 a - 1 ) \lambda } { 2 \hat { \pi } _ { a } ^ { + } } , \forall a \in \{ 0 , 1 \} } \end{array}$ . Then thefollowing holds:

1. Consistency: $\hat { \alpha } _ { a } ^ { + } \overset { p } {  } \alpha _ { a } .$

2. Bias: $\begin{array} { r } { | \alpha _ { a } - \mathbb { E } _ { D _ { \alpha } } \left[ \hat { \alpha } _ { a } ^ { + } \right] | \leq \frac { | \lambda | } { \pi _ { a } } \sqrt { \frac { 1 - \pi _ { a } } { \pi _ { a } N _ { \alpha } } } + 2 \exp ( - \frac { \pi _ { a } ^ { 2 } } { 2 } N _ { \alpha } ) . } \end{array}$

3. Sample Complexity: With $1 - \delta$ probability and $\begin{array} { r } { N _ { \alpha } > \operatorname* { m a x } \{ \frac { 2 } { \pi _ { a } ^ { 2 } } \log \frac { 4 } { \delta } , \frac { \lambda ^ { 2 } } { 2 \epsilon ^ { 2 } \pi _ { a } ^ { 4 } } \log \frac { 4 } { \delta } , \frac { 2 } { \pi _ { a } } \} } \end{array}$ $\operatorname* { P r } \left( | \hat { \alpha } _ { a } ^ { + } - \alpha _ { a } | > \epsilon \right) \leq \delta .$

Proof. We will prove the results for an arbitrary $a \in \{ 0 , 1 \}$ . Due to Lemma 3.8 and our projection operation Π on the empirical estimator, $\alpha _ { a }$ and $\dot { \hat { \alpha } } _ { a } ^ { + }$ are both restricted to the interval $[ 0 , 1 ]$ . Using this, note that in our domain of $\lambda \in [ 0 , \lambda _ { \operatorname* { m a x } } ]$ or $\lambda \in [ - \lambda _ { \operatorname* { m i n } } , 0 ] , \alpha _ { a } = \Pi ( \alpha _ { a } )$ , which is a useful property we exploit throughout the proof.

(1) Consistency: By the weak law of large numbers, $\hat { \pi } _ { a } \overset { p } {  } \pi _ { a }$ . Consider $\hat { \pi } _ { a } ^ { + }$ . Since $\pi _ { a } > 0$ by assumption, we can pick a sufficiently large $N _ { \alpha }$ such that $1 / N _ { \alpha } \leq \pi _ { a } / 2$ . Therefore the event $\{ \hat { \pi } _ { a } <$ $1 / N _ { \alpha } \big \}$ is contained in the event $\lbrace \hat { \pi } _ { a } < \pi _ { a } / 2 \rbrace$ which is further contained in $\{ | \hat { \pi } _ { a } - \pi _ { a } | \geq \grave { \pi } _ { a } / 2 \}$ Since $\hat { \pi } _ { a } \stackrel { p } {  } \pi _ { a } , \mathrm { P r } ( | \hat { \pi } _ { a } - \pi _ { a } | \geq \pi _ { a } / 2 )  0$ and hence $\mathrm { P r } ( \hat { \pi } _ { a } ^ { + } \neq \hat { \pi } _ { a } )  0$ , and hence $\hat { \pi } _ { a } ^ { + } \overset { p } { \to } \pi _ { a }$ The estimator $\hat { \alpha } _ { a } ^ { + }$ is a continuous function of $\hat { \pi } _ { a } ^ { + }$ (including the truncation operation). We can use the Continuous Mapping Theorem <sup>4</sup> to claim that $\hat { \pi } _ { a } ^ { + } \stackrel { p } {  } \pi _ { a } \implies \hat { \alpha } _ { a } ^ { + } \stackrel { p } {  } \alpha _ { a }$

(2) Bias: We will now proceed towards bounding the bias of the threshold for an arbitrary group $A \ = \ a$ . We can break down our bias as follows: $\vert \alpha _ { a } - \mathbb { E } _ { D _ { \alpha } } [ \hat { \alpha } _ { a } ^ { + } ] \vert ~ = ~ \vert \mathbb { E } _ { D _ { \alpha } } [ \alpha _ { a } - \bar { \hat { \alpha } _ { a } ^ { + } } ] \vert ~ \bar { = } ~$ $\begin{array} { r } { | \mathbb { E } _ { D _ { \alpha } } \left[ ( \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } ) { \mathbb 1 } _ { \mathcal { G } } \right] + \mathbb { E } _ { D _ { \alpha } } \left[ ( \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } ) { \mathbb 1 } _ { \mathcal { G } ^ { c } } \right] | \le | \mathbb { E } _ { D _ { \alpha } } \left[ ( \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } ) { \mathbb 1 } _ { \mathcal { G } } \right] | + | \mathbb { E } _ { D _ { \alpha } } \left[ ( \overline { { \alpha _ { a } } } - \hat { \alpha } _ { a } ^ { + } ) { \mathbb 1 } _ { \mathcal { G } ^ { c } } \right] | } \end{array}$ where G is defined next.

Let $\mathcal { G } = \{ | \hat { \pi } _ { a } - \pi _ { a } | \leq \epsilon \}$ be the good event that our estimate of $\pi _ { a }$ is close. On this event, for sufficiently large $\begin{array} { r } { N _ { \alpha } \mathrm { ~ ( } N _ { \alpha } > \frac { 2 } { \pi _ { a } } \mathrm { ) , ~ } \bar { \pi _ { a } ^ { + } } = \hat { \pi } _ { a } } \end{array}$ and hence $\hat { \alpha } _ { a } ^ { + } = \hat { \alpha } _ { a }$ . Using Hoeffding’s inequality, we know that $\hat { \pi } _ { a } \in ( \pi _ { a } - \epsilon , \pi _ { a } + \epsilon )$ with $1 - \delta / 2$ probability over the choice of data sample $D _ { \alpha }$ . We can set $\epsilon \leq \pi _ { a } / 2$ , which means $\begin{array} { r } { \mathrm { ~ \tilde { ~ } { ~ N _ { \alpha } ~ > ~ \frac { 2 } { \pi _ { \alpha } ^ { 2 } } \log \left( \frac { 4 } { \delta } \right) ~ } } } \end{array}$ .

Consider the first term:

$$
\begin{array} { r l } { \mathbb { E } _ { D _ { \sigma } } [ ( \alpha _ { \alpha } - \hat { \alpha } _ { \alpha } ) { \mathbf 1 } _ { \mathcal { G } } ] } & { \leq \mathbb { E } _ { D _ { \sigma } } [ ( \alpha _ { \alpha } - \hat { \alpha } _ { \alpha } ) { \mathbf 1 } _ { \mathcal { G } } ] = \mathbb { E } _ { D _ { \sigma } } [ | { \mathbf 1 } ( \alpha _ { \alpha } ) - { \mathbf 1 } | ( \hat { \alpha } _ { \alpha } ) - 1 | ( \hat { \alpha } _ { \alpha } ) | { \mathbf 1 } _ { \mathcal { G } } ] } \\ & { \leq \mathbb { E } _ { D _ { \sigma } } [ \| \alpha _ { \alpha } - \hat { \alpha } _ { \alpha } \| _ { \mathcal { G } } ] ( { \mathrm { S i n c e ~ } } \operatorname { I n i s h i t s f i e s } ) } \\ & { = \mathbb { E } _ { D _ { \sigma } } [ \| \frac { \lambda ( 2 \alpha - 1 ) } { 2 \pi _ { \alpha } } - \frac { \lambda ( 2 \alpha - 1 ) } { 2 \hat { \pi } _ { \alpha } } | \ { \mathbf 1 } _ { \mathcal { G } } ] } \\ & { \leq \frac { | \lambda | } { 2 } \cdot \mathbb { E } _ { D _ { \sigma } } [ \| \frac { 1 } { \pi _ { \alpha } } - \frac { 1 } { \hat { \pi } _ { \alpha } } | \ \| \mathcal { G } ] } \\ &  = \frac { | \lambda | } { 2 } \cdot \mathbb { E } _ { D _ { \sigma } } [ \| \frac { \hat { \alpha } _ { \alpha } } { \pi _ { \alpha } \hat { \alpha } _ { \alpha } } \| { \mathbf 1 } _ { \mathcal { G } } ] = \frac { | \lambda | } { 2 } \cdot \mathbb { E } _ { D _ { \sigma } } [ \| \frac { \hat { \alpha } _ { \alpha } } { \pi _ { \alpha } \hat { \alpha } _ { \alpha } } - { \mathbf 1 } _  \mathcal  \end{array}
$$

Since $\hat { \pi } _ { a }$ is an unbiased estimate of $\pi _ { a } ,$ using Lemma B.2, $\mathbb { E } _ { D _ { \alpha } } \left[ | \hat { \pi } _ { a } - \pi _ { a } | \right] \leq \sqrt { V a r ( \hat { \pi } _ { a } } )$ . Using $\begin{array} { r } { \mathbb { E } _ { D _ { \alpha } } \left[ | \alpha _ { a } - \hat { \alpha } _ { a } | \mathbb { 1 } _ { \mathcal { G } } \right] \leq \mathbb { E } _ { D _ { \alpha } } \left[ | \alpha _ { a } - \hat { \alpha } _ { a } | \right] \leq \frac { | \lambda | \sqrt { V a r ( \hat { \pi } _ { a } ) } } { \pi _ { a } ^ { 2 } } , } \end{array}$ , we get:

$$
\left| \mathbb { E } _ { D _ { \alpha } } \left[ \left( \alpha _ { a } - \hat { \alpha } _ { a } \right) \mathbb { 1 } _ { \mathcal { G } } \right] \right| \leq \frac { | \lambda | \sqrt { V a r ( \hat { \pi } _ { a } ) } } { \pi _ { a } ^ { 2 } } = \frac { | \lambda | } { \pi _ { a } } \sqrt { \frac { 1 - \pi _ { a } } { \pi _ { a } N _ { \alpha } } }
$$

where $\begin{array} { r } { V a r ( \hat { \pi } _ { a } ) = V a r ( \frac { 1 } { N _ { \alpha } } \sum _ { i \in [ N _ { \alpha } ] } \mathbb { I } \left( a _ { i } = a \right) ) = \frac { 1 } { N _ { \alpha } ^ { 2 } } V a r ( \sum _ { i \in [ N _ { \alpha } ] } \mathbb { I } \left( a _ { i } = a \right) ) = \frac { \pi _ { a } \left( 1 - \pi _ { a } \right) } { N _ { \alpha } } . } \end{array}$

On $\mathcal G ^ { c }$ , we can use the fact that both $\alpha _ { a } , \hat { \alpha } _ { a } ^ { + } \in [ 0 , 1 ] ;$

$$
\begin{array} { r l } & { | \mathbb { E } _ { D _ { \alpha } } \left[ ( \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } ) \mathbb { 1 } _ { \mathcal { G } ^ { c } } \right] | \leq | \mathbb { E } _ { D _ { \alpha } } \left[ 1 \cdot \mathbb { 1 } _ { \mathcal { G } ^ { c } } \right] | } \\ & { = \operatorname* { P r } \big ( \mathcal { G } ^ { C } \big ) \leq 2 \exp \left( - \frac { \pi _ { a } ^ { 2 } } { 2 } N _ { \alpha } \right) ( \mathrm { U s i n g ~ H o e f f d i n g ' s ~ i n e q u a l i t y ~ o n ~ P r ~ } ( \mathcal { G } ^ { c } ) ) . } \end{array}
$$

Overall,

$$
\left| \alpha _ { a } - \mathbb { E } _ { D _ { \alpha } } \left[ \hat { \alpha } _ { a } ^ { + } \right] \right| \leq \frac { \left| \lambda \right| } { \pi _ { a } } \sqrt { \frac { 1 - \pi _ { a } } { \pi _ { a } N _ { \alpha } } } + 2 \exp \left( { - \frac { \pi _ { a } ^ { 2 } } { 2 } N _ { \alpha } } \right) .
$$

(3) Sample Complexity: We now want to study the number of samples required for a good estimator of $\alpha _ { a } ^ { + }$ . We want to ensure that $\mathrm { P r } \left( | \hat { \alpha } _ { a } ^ { + } - \alpha _ { a } | \stackrel { . } { > } \epsilon \right) \leq \delta$ . Note that $\bar { \{ | \hat { \alpha } _ { a } ^ { + } - \bar { \alpha } _ { a } | > \epsilon \} } \subseteq \bar { \{ | \tilde { \alpha } _ { a } ^ { + } - \alpha _ { a } | > } $ $\epsilon \}$

Reusing the analysis from the previous part, we can first ensure that $\begin{array} { r l } { \operatorname* { P r } \left( \mathcal { G } ^ { C } \right) } & { { } = } \end{array}$ Pr $\left( | \pi _ { a } - \hat { \pi } _ { a } | > \pi _ { a } / 2 \right) < \delta / 2$ by using Hoeffding’s inequality to obtain that $\begin{array} { r } { N _ { \alpha } > \frac { 2 } { \pi _ { a } ^ { 2 } } \log \frac { 4 } { \delta } } \end{array}$

Next, as stated earlier, on the good event ${ \mathcal { G } } ,$ as long as $\begin{array} { r } { N _ { \alpha } > \frac { 2 } { \pi _ { a } } } \end{array}$ (which means $1 / N _ { \alpha } < \pi _ { a } / 2 )$ $\hat { \pi } _ { a } ^ { + } = \hat { \pi } _ { a }$ and hence $\hat { \alpha } _ { a } ^ { + } = \tilde { \alpha } _ { a }$ . This implies that $\begin{array} { r } { | \tilde { \alpha } _ { a } ^ { + } - \alpha _ { a } | = | \frac { \lambda } { 2 } | \cdot | \frac { 1 } { \hat { \pi } _ { a } } - \frac { 1 } { \pi _ { a } } | \leq \frac { | \lambda | } { \pi _ { a } ^ { 2 } } | \hat { \pi } _ { a } - \pi _ { a } | } \end{array}$

Therefore, on $\begin{array} { r } { \mathcal { G } , \{ | \tilde { \alpha } _ { a } ^ { + } - \alpha _ { a } | > \epsilon \} \subseteq \{ | \hat { \pi } _ { a } - \pi _ { a } | > \frac { \epsilon \pi _ { a } ^ { 2 } } { | \lambda | } \} } \end{array}$ . Now, we can study the entire event and upper bound its failure probability:

$$
\begin{array} { r l } & { \operatorname* { P r } \left( | \tilde { \alpha } _ { a } ^ { + } - \alpha _ { a } | > \epsilon \right) \leq \operatorname* { P r } \left( | \tilde { \alpha } _ { a } ^ { + } - \alpha _ { a } | > \epsilon \cap \mathcal { G } \right) + \operatorname* { P r } \left( | \tilde { \alpha } _ { a } ^ { + } - \alpha _ { a } | > \epsilon \cap \mathcal { G } ^ { C } \right) } \\ & { \leq \operatorname* { P r } \left( | \hat { \pi } _ { a } - \pi _ { a } | > \displaystyle \frac { \epsilon \pi _ { a } ^ { 2 } } { | \lambda | } \right) + \operatorname* { P r } \left( \mathcal { G } ^ { C } \right) } \end{array}
$$

$\leq 2 \exp ( - \frac { 2 N _ { \alpha } \epsilon ^ { 2 } \pi _ { a } ^ { 4 } } { \lambda ^ { 2 } } ) + 2 \exp ( - \frac { N _ { \alpha } \pi _ { a } ^ { 2 } } { 2 } )$ (Using Hoeffding’s Inequality and earlier bound on ${ \mathcal { G } } ^ { C } )$

Bounding the second term by $\delta / 2$ gave us $\begin{array} { r } { N _ { \alpha } > \frac { 2 } { \pi _ { { \alpha } } ^ { 2 } } } \end{array}$ log $\frac { 4 } { \delta }$ . Doing the same for the first term gives us $\begin{array} { r } { N _ { \alpha } > \frac { \lambda ^ { 2 } } { 2 \epsilon ^ { 2 } \pi _ { a } ^ { 4 } } \log \frac { 4 } { \delta } } \end{array}$ . Overall, $\begin{array} { r } { N _ { \alpha } > \operatorname* { m a x } \{ \frac { 2 } { \pi _ { a } ^ { 2 } } \log \frac { 4 } { \delta } , \frac { \lambda ^ { 2 } } { 2 \epsilon ^ { 2 } \pi _ { a } ^ { 4 } } \log \frac { 4 } { \delta } , \frac { 2 } { \pi _ { a } } \} } \end{array}$

Theorem D.2. (EOP results from Theorem 4.1 in the main text) (Consistency and Sample Complexity for EOP threshold) Given a dataset $D _ { \alpha }$ of clean soft labels $( c _ { i } = \eta ( x _ { i } , a _ { i } ) )$ and assuming $2 -$ $\frac { ( 2 a - 1 ) \lambda } { \pi _ { 1 a } } \geq \tau > 0$ (denominator bounded awayfrom zero),for the case ofequal opportunity, consider the following estimator of the threshold: $\hat { \alpha } _ { a } ^ { + } = \Pi ( \tilde { \alpha } _ { a } ^ { + } )$ , where $\begin{array} { r } { \tilde { \alpha } _ { a } ^ { + } = \frac { 1 } { 2 - \frac { \left( 2 a - 1 \right) \lambda } { \hat { \pi } _ { 1 a } ^ { + } } } } \end{array}$ . Then the following holds:

1. Consistency: $\hat { \alpha } _ { a } ^ { + } \overset { p } { \to } \alpha _ { a }$

2. Bias: $\begin{array} { r l r } { | \alpha _ { a } ~ - ~ \mathbb { E } _ { D _ { \alpha } } \left[ \hat { \alpha } _ { a } ^ { + } \right] | } & { \leq } & { \frac { 4 | \lambda | } { \tau ^ { 2 } \pi _ { 1 , \alpha } ^ { 2 } } \sqrt { \frac { \pi _ { 1 a } \left( 1 - \pi _ { 1 a } \right) } { N _ { \alpha } } ~ + ~ 2 \exp ( - 2 \epsilon ^ { 2 } N _ { \alpha } ) } } \end{array}$ , where $\epsilon \quad \leq$ min $\bigl \{ \frac { \pi _ { 1 a } } { 2 } , \frac { \tau \pi _ { 1 a } ^ { 2 } } { 4 | \lambda | } \bigr \}$

3. Sample Complexity: With $1 - \delta$ probability over the choice of data sample $N _ { \alpha }$ , as long as $\begin{array} { r } { N _ { \alpha } \geq \frac { 1 } { 2 t ^ { 2 } } \log ( \frac { 2 } { \delta } ) f o r t = \operatorname* { m i n } \Big \{ \frac { \pi _ { 1 a } } { 2 } , \frac { \tau \pi _ { 1 a } ^ { 2 } } { 4 | \lambda | } , \frac { \epsilon \tau ^ { 2 } \pi _ { 1 a } ^ { 2 } } { 4 | \lambda | } \Big \} , \operatorname* { P r } \big ( | \hat { \alpha } _ { a } - \alpha _ { a } | > \epsilon \big ) \leq \delta . } \end{array}$

Proof. We will prove the results for an arbitrary $a \in \{ 0 , 1 \}$ . Due to Lemma 3.8 and our projection operation Π on the empirical estimator, $\alpha _ { a }$ and $\bar { \hat { \alpha } } _ { a } ^ { + }$ are both restricted to the interval [0, 1]. Using this, note that in our domain of $\lambda \in [ 0 , \lambda _ { \operatorname* { m a x } } ]$ or $\lambda \in [ - \lambda _ { \operatorname* { m i n } } , 0 ] , \alpha _ { a } = \Pi ( \alpha _ { a } )$ , which is a useful property we exploit throughout the proof.

(1) Consistency: By the weak law of large numbers, $\hat { \pi } _ { 1 a } \overset { p } {  } \pi _ { 1 a }$ . Consider $\hat { \pi } _ { 1 a } ^ { + }$ . Since $\pi _ { 1 a } > 0$ by assumption, we can pick a sufficiently large $N _ { \alpha }$ such that $1 / N _ { \alpha } \leq \pi _ { 1 a } / 2$ . Therefore the event $\{ \dot { \pi } _ { 1 a } < 1 \dot { / } N _ { \alpha } \}$ is contained in the event $\lbrace \dot { \pi } _ { 1 a } < \pi _ { 1 a } / 2 \rbrace$ which is further contained in $\left\{ \left| { \hat { \pi } } _ { 1 a } - \pi _ { 1 a } \right| \geq \right.$ $\pi _ { 1 a } / 2 \}$ . Since $\hat { \pi } _ { 1 a } \stackrel { p } {  } \pi _ { 1 a } , \mathrm { P r } ( | \hat { \pi } _ { 1 a } - \pi _ { 1 a } | \geq \pi _ { 1 a } / 2 )  0$ and hence Pr $( \hat { \pi } _ { 1 a } ^ { + } \neq \hat { \pi } _ { 1 a } )  0$ , and hence $\hat { \pi } _ { 1 a } ^ { + } \overset { p } {  } \pi _ { 1 a }$

Now, we will show consistency in two steps. First, let the denominator be defined as $\tilde { d } _ { a } = 2 -$ $\frac { ( 2 a - \mathrm { 1 } ) \lambda } { \hat { \pi } _ { 1 a } ^ { + } }$ and $\begin{array} { r } { d _ { a } = 2 - \frac { ( 2 a - 1 ) \lambda } { \pi _ { 1 a } } } \end{array}$ . Since $\pi _ { 1 a } > 0$ by assumption, and $\tilde { \pi } _ { 1 a } ^ { + } ~ \overset { p } {  } ~ \pi _ { 1 a }$ , we can use the Continuous Mapping Theorem to claim that $\tilde { d } _ { a } \stackrel { p } {  } d _ { a }$

Again, since $d _ { a } \geq \tau > 0$ , then by application of the Continuous Mapping Theorem, $\frac { 1 } { \tilde { d } _ { a } } \stackrel { p } {  } \frac { 1 } { d _ { a } }$ and hence $\hat { \alpha } _ { a } \overset { p } { \to } \alpha _ { a }$ (as the truncation operation Π is also continuous).

(2) Bias: We want to now bound the bias of the estimator: $\left. \alpha _ { a } - \mathbb { E } _ { D _ { \alpha } } \left[ \hat { \alpha } _ { a } \right] \right. :$

$$
\begin{array} { r l } & { \left| \alpha _ { a } - \mathbb { E } _ { D _ { \alpha } } \left[ \hat { \alpha } _ { a } \right] \right| = \left| \mathbb { E } _ { D _ { \alpha } } \left[ \alpha _ { a } - \hat { \alpha } _ { a } \right] \right| \leq \mathbb { E } _ { D _ { \alpha } } \left[ \left| \alpha _ { a } - \hat { \alpha } _ { a } \right| \right] = \mathbb { E } _ { D _ { \alpha } } \left[ \left| \Pi ( \alpha _ { a } ) - \Pi ( \tilde { \alpha } _ { a } ) \right| \right] } \\ & { \qquad \leq \mathbb { E } _ { D _ { \alpha } } \left[ \left| \alpha _ { a } - \tilde { \alpha } _ { a } \right| \right] \quad \mathrm { ( S i n c e ~ } \Pi \mathrm { i s ~ L i p s c h i t z ) } } \\ & { \qquad = \mathbb { E } _ { D _ { \alpha } } \left[ \left| \cfrac { 1 } { d _ { a } } - \cfrac { 1 } { \tilde { d } _ { a } } \right| \right] = \mathbb { E } _ { D _ { \alpha } } \left[ \left| \cfrac { 1 } { d _ { a } } - \cfrac { 1 } { \tilde { d } _ { a } } \right| \mathbb { 1 } _ { \mathcal { G } } \right] + \mathbb { E } _ { D _ { \alpha } } \left[ \left| \cfrac { 1 } { d _ { a } } - \cfrac { 1 } { \tilde { d } _ { a } } \right| \mathbb { 1 } _ { \mathcal { G } ^ { c } } \right] , } \end{array}
$$

where we let $\mathcal { G } = \{ | \hat { \pi } _ { 1 a } - \pi _ { 1 a } | \leq \epsilon \}$ denote the good event that $\hat { \pi } _ { 1 a }$ is close to $\pi _ { 1 a }$ . On this event, for sufficiently large $\begin{array} { r } { N _ { \alpha } ( N _ { \alpha } > \frac { 2 } { \pi _ { 1 a } } ) , \hat { \pi } _ { 1 a } ^ { + } = \hat { \pi } _ { 1 a } } \end{array}$ and hence $\hat { \alpha } _ { a } ^ { + } = \hat { \alpha } _ { a }$ . We can set $\epsilon \leq \pi _ { 1 a } / 2$ so that $\hat { \pi } _ { 1 a } \geq \pi _ { 1 a } / 2$

On $\mathcal { G } \colon$

$$
\left| d _ { a } - \tilde { d } _ { a } \right| \leq \left| \lambda \right| \left| \frac { \hat { \pi } _ { 1 a } - \pi _ { 1 a } } { \pi _ { 1 a } \hat { \pi } _ { 1 a } } \right| \leq \frac { 2 | \lambda | } { \pi _ { 1 a } ^ { 2 } } | \hat { \pi } _ { 1 a } - \pi _ { 1 a } |
$$

Similarly, we can also bound $\tilde { d } _ { a }$ on G:

$$
\tilde { d } _ { a } \geq d _ { a } - | \tilde { d } _ { a } - d _ { a } | \geq \tau - \frac { 2 | \lambda | \epsilon } { \pi _ { 1 a } ^ { 2 } } .
$$

We can ensure that $\tilde { d } _ { a } \geq \tau / 2$ on ${ \mathcal { G } } _ { : }$ , which means $\begin{array} { r } { \epsilon \leq \operatorname* { m i n } \{ \frac { \pi _ { 1 a } } { 2 } , \frac { \tau \pi _ { 1 a } ^ { 2 } } { 4 | \lambda | } \} } \end{array}$ . Finally, for the first term of the bias bound, on the good event $\mathcal { G } \mathrm { : }$

$$
\mathbb { E } _ { D _ { \alpha } } \left[ \Big \vert \frac { 1 } { d _ { a } } - \frac { 1 } { \tilde { d } _ { a } } \Big \vert \mathbb { 1 } _ { \mathcal { G } } \right] = \mathbb { E } _ { D _ { \alpha } } \left[ \frac { \vert \tilde { d } _ { a } - d _ { a } \vert } { d _ { a } \tilde { d } _ { a } } \mathbb { 1 } _ { \mathcal { G } } \right] \big \vert \leq \frac { 2 } { \tau ^ { 2 } } \cdot \frac { 2 \vert { \boldsymbol \lambda } \vert } { \pi _ { 1 a } ^ { 2 } } \mathbb { E } _ { D _ { \alpha } } \left[ \vert \hat { \pi } _ { 1 a } - \pi _ { 1 a } \vert \right] .
$$

On $\mathcal G ^ { c }$ , we can use the fact that both $\alpha _ { a } , \hat { \alpha } _ { a } ^ { + } \in [ 0 , 1 ] ;$

$$
\begin{array} { r l } & { \mathbb { E } _ { D _ { \alpha } } \left[ \left| \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } \right| \mathbb { 1 } _ { \mathcal { G } ^ { c } } \right] \leq \mathbb { E } _ { D _ { \alpha } } \left[ 1 \cdot \mathbb { 1 } _ { \mathcal { G } ^ { c } } \right] } \\ & { \quad = \operatorname* { P r } \left( \mathcal { G } ^ { C } \right) \leq 2 \exp ( - 2 \epsilon ^ { 2 } N _ { \alpha } ) ( \mathrm { U s i n g ~ H o e f f d i n g ' s ~ i n e q u a l i t y ~ o n ~ P r } \left( \mathcal { G } ^ { c } \right) ) . } \end{array}
$$

Overall,

$$
\left| \alpha _ { a } - \mathbb { E } _ { D _ { \alpha } } \left[ \hat { \alpha } _ { a } ^ { + } \right] \right| \leq \frac { 4 | \lambda | } { \tau ^ { 2 } \pi _ { 1 a } ^ { 2 } } \mathbb { E } _ { D _ { \alpha } } \left[ \left| \hat { \pi } _ { 1 a } - \pi _ { 1 a } \right| \right] + 2 \exp ( - 2 \epsilon ^ { 2 } N _ { \alpha } ) , \mathrm { ~ w h e r e ~ } \epsilon \leq \operatorname* { m i n } \left\{ \frac { \pi _ { 1 a } } { 2 } , \frac { \tau \pi _ { 1 a } ^ { 2 } } { 4 | \lambda | } \right\} .
$$

To further simplify this, using Lemma B.2, we know that $\begin{array} { r } { \mathbb { E } \left[ | \pi _ { 1 a } - \hat { \pi } _ { 1 a } | \right] \ \leq \ \sqrt { V a r ( \hat { \pi } _ { 1 a } ) } \ = } \end{array}$ $\sqrt { \frac { \pi _ { 1 a } ( 1 - \pi _ { 1 a } ) } { N _ { \alpha } } }$ , and hence:

$$
\left| \alpha _ { a } - \mathbb { E } _ { D _ { \alpha } } \left[ \hat { \alpha } _ { a } ^ { + } \right] \right| \leq \frac { 4 | \lambda | } { \tau ^ { 2 } \pi _ { 1 a } ^ { 2 } } \sqrt { \frac { \pi _ { 1 a } ( 1 - \pi _ { 1 a } ) } { N _ { \alpha } } + 2 \exp ( - 2 \epsilon ^ { 2 } N _ { \alpha } ) } , \mathrm { ~ w h e r e ~ } \epsilon \leq \operatorname* { m i n } \left\{ \frac { \pi _ { 1 a } } { 2 } , \frac { \tau \pi _ { 1 a } ^ { 2 } } { 4 | \lambda | } \right\} .
$$

(3) Sample Complexity: For sample complexity, we can proceed by returning a high probability bound on the good event derived in the previous part. Note that $\{ | \hat { \alpha } _ { a } ^ { + } - \bar { \alpha } _ { a } | > \epsilon \} \subseteq \{ | \bar { \alpha } _ { a } ^ { + } - \alpha _ { a } | > \epsilon \bar  \}$ Recall that $\mathcal { G } = \{ | \hat { \pi } _ { 1 a } - \pi _ { 1 a } | \leq \epsilon \}$ denotes the good event that $\hat { \pi } _ { 1 a }$ is close to $\pi _ { 1 a }$ . We can set $\epsilon \leq \pi _ { 1 a } / 2$ so that $\hat { \pi } _ { 1 a } \geq \pi _ { 1 a } / 2$ , whenever $1 / N _ { \alpha } \bar { \leq } \pi _ { 1 a } / 2 \implies N _ { \alpha } \geq 2 / \pi _ { 1 a }$ From the previous part, we know that on $\mathcal { G }$

$$
\begin{array} { r } { \bullet \ | d _ { a } - \tilde { d } _ { a } | \leq \frac { 2 | \lambda | \epsilon } { \pi _ { 1 a } ^ { 2 } } , } \end{array}
$$

$\begin{array} { r } { \tilde { d } _ { a } \geq \tau / 2 \mathrm { f o r } \epsilon \leq \operatorname* { m i n } \{ \frac { \pi _ { 1 a } } { 2 } , \frac { \tau \pi _ { 1 a } ^ { 2 } } { 4 | \lambda | } \} } \end{array}$ , and

• Using both of the results above, $\begin{array} { r } { | \alpha _ { a } - \tilde { \alpha } _ { a } | \le \frac { 4 | \lambda | \epsilon } { \tau ^ { 2 } \pi _ { 1 a } ^ { 2 } } } \end{array}$ on G.

Finally, if we want $| \tilde { \alpha } _ { a } - \alpha _ { a } | \le \epsilon ^ { \prime }$ , then $\begin{array} { r } { \frac { 4 | \lambda | \epsilon } { \tau ^ { 2 } \pi _ { 1 a } ^ { 2 } } \leq \epsilon ^ { \prime } \implies \epsilon \leq \frac { \epsilon ^ { \prime } \tau ^ { 2 } \pi _ { 1 a } ^ { 2 } } { 4 | \lambda | } } \end{array}$ . Combining this with the previous min bound on $\epsilon ,$ we $\begin{array} { r } { \mathrm { g e t } \epsilon = \operatorname* { m i n } \left\{ \frac { \pi _ { 1 a } } { 2 } , \frac { \tau \pi _ { 1 a } ^ { 2 } } { 4 | \lambda | } , \frac { \epsilon ^ { \prime } \tau ^ { 2 } \pi _ { 1 a } ^ { 2 } } { 4 | \lambda | } \right\} } \end{array}$

We have now shown that $\{ | { \hat { \pi } } _ { 1 a } - \pi _ { 1 a } | \ \leq \ \epsilon \} \ \subseteq \ \{ | \alpha _ { a } - { \hat { \alpha } } _ { a } | \ \leq \ \epsilon ^ { \prime } \}$ . Taking complements and probabilities on both sides, we get:

$$
\begin{array} { r } { \operatorname* { P r } \left( \left| \alpha _ { a } - \hat { \alpha } _ { a } \right| > \epsilon ^ { \prime } \right) \le \operatorname* { P r } \left( \left| \hat { \pi } _ { 1 a } - \pi _ { 1 a } \right| > \epsilon \right) \le 2 \exp ( - 2 N _ { \alpha } \epsilon ^ { 2 } ) \le \delta , } \end{array}
$$

which gives us $\begin{array} { r } { N _ { \alpha } \geq \frac { 1 } { 2 \epsilon ^ { 2 } } \log ( \frac { 2 } { \delta } ) } \end{array}$ for $\begin{array} { r } { \epsilon = \operatorname* { m i n } \left\{ \frac { \pi _ { 1 a } } { 2 } , \frac { \tau \pi _ { 1 a } ^ { 2 } } { 4 | \lambda | } , \frac { \epsilon ^ { \prime } \tau ^ { 2 } \pi _ { 1 a } ^ { 2 } } { 4 | \lambda | } \right\} } \end{array}$

Refactoring notations (using t for ϵ and ϵ for $\epsilon ^ { \prime } )$ , with $1 - \delta$ probability over the choice of data sample $N _ { \alpha }$ , as long as $\begin{array} { r } { N _ { \alpha } \geq \frac { 1 } { 2 t ^ { 2 } } \log ( \frac { 2 } { \delta } ) \mathrm { f o r } t = \operatorname* { m i n } \Big \{ \frac { \pi _ { 1 a } } { 2 } , \frac { \tau \pi _ { 1 a } ^ { 2 } } { 4 | \lambda | } , \frac { \epsilon \tau ^ { 2 } \pi _ { 1 a } ^ { 2 } } { 4 | \lambda | } \Big \} , \mathrm { P r } \left( | \hat { \alpha } _ { a } - \alpha _ { a } | > \epsilon \right) \leq \delta } \end{array}$

Theorem D.3. (Theorem 4.3 in the main text) Consider the Bayes optimal classifier for the costsensitive risk with group-dependent costs $\alpha _ { a } .$

$$
g _ { \alpha } ^ { \ast } ( x , a ) = \left\{ \begin{array} { l l } { 1 , } & { i f \eta ( x , a ) \geq \alpha _ { a } } \\ { 0 , } & { o t h e r w i s e . } \end{array} \right.
$$

1. Bayes Risk: The Bayes error with respect to the cost-sensitive risk can be expressed as $\dot { \beta _ { C S } } = \mathbb { E } _ { X , A } \left[ \operatorname* { m i n } ( ( \dot { 1 } - \alpha _ { A } ) \eta ( X , A ) , \dot { \alpha } _ { A } ( 1 - \eta ( X , A ) ) ) \right]$

2. Consistency: $\hat { \beta } _ { C S } = \textstyle { \frac { 1 } { N _ { \beta } } } \sum _ { i \in [ N _ { \beta } ] } \operatorname* { m i n } ( ( 1 - \alpha _ { i } ) c _ { i } , \alpha _ { i } ( 1 - c _ { i } ) )$ is a consistent estimator of β<sub>CS</sub>, where for the $i ^ { t h }$ instance $( c _ { i } , a _ { i } ) , c _ { i } = \eta ( x _ { i } , a _ { i } )$ and $\alpha _ { i } = \alpha _ { a _ { i } }$ , the group dependent cost.

3. Sample Complexity: For any $\delta > 0 _ { : }$ , with probability at least $1 - \delta , | \hat { \beta } _ { C S } - \beta _ { C S } | \leq$ $\textstyle { \frac { 1 } { 4 } } { \sqrt { { \frac { 1 } { N _ { \beta } } } \log ( { \frac { 2 } { \delta } } ) } }$

Proof. (1) Bayes Risk: We will adapt the standard proof technique<sup>5</sup> of the Bayes error for costsensitive settings. The expected α-cost sensitive risk for the classifier $g ^ { * }$ can be expressed as:

$$
\begin{array} { l } { \displaystyle { R _ { \alpha } ( g ^ { * } ) = \sum _ { a \in A } \Biggl ( ( 1 - \alpha _ { a } ) \cdot \mathbb { E } _ { X , Y = 1 , A = a } \left[ l _ { 0 1 } ( g ^ { * } ( X , a ) , 1 ) \right] + \alpha _ { a } \cdot \mathbb { E } _ { X , Y = 0 , A = a } \left[ l _ { 0 1 } ( g ^ { * } ( X , a ) , 0 ) \right] \Biggr ) } } \\ { \displaystyle { = \sum _ { a \in A } \mathbb { E } _ { X , a } \left[ \mathbb { E } _ { Y = 1 | X , a } \left[ ( 1 - \alpha _ { a } ) \mathbb { I } \left( Y = 1 , g ^ { * } ( X , a ) = 0 \right) \right] + \mathbb { E } _ { Y = 0 | X , a } \left[ \alpha _ { a } \mathbb { I } \left( Y = 0 , g ^ { * } ( X , a ) = 1 \right) \right] \right] } } \\ { \displaystyle { = \sum _ { a \in A } \mathbb { E } _ { X , a } \left[ ( 1 - \alpha _ { a } ) \mathbb { I } \left( \eta ( X , a ) < \alpha _ { a } \right) \eta ( X , a ) + \alpha _ { a } \mathbb { I } \left( \eta \ge \alpha _ { a } \right) \left( 1 - \eta ( X , a ) \right) \right] } } \end{array}
$$

At this point, note that $\eta < \alpha _ { a } \implies ( 1 - \alpha _ { a } ) \eta < \alpha _ { a } ( 1 - \eta )$ and similarly $\begin{array} { r } { \eta \ge \alpha _ { a } \implies \alpha _ { a } ( 1 - \eta ) \le } \end{array}$ $( 1 - \alpha _ { a } ) \eta$ . Therefore,

$$
\begin{array} { l } { \displaystyle R _ { \alpha } ( g ^ { * } ) = \sum _ { a \in \mathcal { A } } \mathbb { E } _ { X , a } \left[ \left( 1 - \alpha _ { a } \right) \mathbb { I } \left( \eta ( X , a ) < \alpha _ { a } \right) \eta ( X , a ) + \alpha _ { a } \mathbb { I } \left( \eta \geq \alpha _ { a } \right) \left( 1 - \eta ( X , a ) \right) \right] } \\ { \displaystyle = \sum _ { a \in \mathcal { A } } \mathbb { E } _ { X , a } [ \mathbb { I } \left( \eta ( X , a ) < \alpha _ { a } \right) \operatorname* { m i n } ( \left( 1 - \alpha _ { a } \right) \eta ( X , a ) , \alpha _ { a } ( 1 - \eta ( X , a ) ) ) } \\ { \displaystyle \qquad + \mathbb { I } \left( \eta ( X , a ) \geq \alpha _ { a } \right) \operatorname* { m i n } ( \alpha _ { a } ( 1 - \eta ( X , a ) ) , \eta ( X , a ) ( 1 - \alpha _ { a } ) ) ] } \\ { \displaystyle = \mathbb { E } _ { X , A } \left[ \operatorname* { m i n } ( \left( 1 - \alpha _ { A } \right) \eta ( X , A ) , \alpha _ { A } ( 1 - \eta ( X , A ) ) ) \right] . } \end{array}
$$

It remains to show that $g ^ { * }$ is the Bayes optimal classifier. Let $g ^ { \prime }$ be any arbitrary classifier. Then,

$$
\begin{array} { r l } { R _ { \alpha } ( g ^ { \prime } ) - R _ { \alpha } ( g ^ { * } ) = \mathbb { E } _ { X , a } [ ( 1 - \alpha _ { a } ) \eta ( X , a ) \mathbb { I } ( g ^ { \prime } ( X , a ) = 0 ) + \alpha _ { a } ( 1 - \eta ( X , a ) ) \mathbb { I } ( g ^ { \prime } ( X , a ) = 1 ) ] } & { } \\ { - \mathbb { E } _ { X , a } [ ( 1 - \alpha _ { a } ) \eta ( X , a ) \mathbb { I } ( g ^ { * } ( X , a ) = 0 ) + \alpha _ { a } ( 1 - \eta ( X , a ) ) \mathbb { I } ( g ^ { * } ( X , a ) = 1 ) ] } & { } \\ { = \mathbb { E } _ { X , a } [ ( \mathbb { I } ( g ^ { \prime } = 0 ) - \mathbb { I } ( g ^ { * } = 0 ) ) ( 1 - \alpha _ { a } ) \eta ( X , a ) } & { } \\ { + ( \mathbb { I } ( g ^ { \prime } = 1 ) - \mathbb { I } ( g ^ { * } = 1 ) ) \alpha _ { a } ( 1 - \eta ( X , a ) ) ] } & { } \\ { = \mathbb { E } _ { X , a } [ ( \mathbb { I } ( g ^ { \prime } = 0 ) - \mathbb { I } ( g ^ { * } = 0 ) ) ( \eta ( X , a ) - \alpha _ { a } ) ] } & { } \\ { = \mathbb { E } _ { X , a } [ \mathbb { I } ( g ^ { \prime } \neq g ^ { * } ) \cdot ( \eta ( X , a ) - \alpha _ { a } ) \cdot s g n ( \eta ( X , a ) - \alpha _ { a } ) ] , } & { } \\ { \mathrm { u s i n g ~ t h e ~ f a c t ~ t h a t ~ } g ^ { * } = \mathbb { I } ( \eta \geq \alpha _ { a } ) } & { } \\ { = \mathbb { E } _ { X , a } [ \mathbb { I } ( g ^ { \prime } \neq g ^ { * } ) \cdot | \eta ( X , a ) - \alpha _ { a } | ] \geq 0 . } \end{array}
$$

(2) Consistency:

$$
\begin{array} { r l r } {  { \mathbb { E } _ { ( c _ { i } , a _ { i } ) _ { i = 1 } ^ { N _ { \beta } } } [ \hat { \beta } _ { C S } ] = \mathbb { E } _ { ( c _ { i } , a _ { i } ) _ { i = 1 } ^ { N _ { \beta } } } [ \frac { 1 } { N _ { \beta } } \sum _ { i \in [ N _ { \beta } ] } \operatorname* { m i n } ( ( 1 - \alpha _ { i } ) c _ { i } , \alpha _ { i } ( 1 - c _ { i } ) ) ] } } \\ & { } & { = \frac { 1 } { N _ { \beta } } \sum _ { i \in [ N _ { \beta } ] } \mathbb { E } _ { ( c _ { i } , a _ { i } ) \sim D } [ \operatorname* { m i n } ( ( 1 - \alpha _ { i } ) c _ { i } , \alpha _ { i } ( 1 - c _ { i } ) ) ] } \\ & { } & { = \beta _ { C S } . } \end{array}
$$

By the weak law of large numbers, $\hat { \beta } _ { C S } \overset { p } {  } \beta _ { C S }$

(3) Sample Complexity: The proof for sample complexity is a straightforward application of Hoeffding’s inequality, by noting that mi $\mathfrak { l } \big ( ( 1 - \alpha _ { i } ) c _ { i } , \bar { \alpha } _ { i } ( 1 - c _ { i } ) \big ) \in [ 0 , \bar { \mathfrak { i } } / 4 ]$ :

$$
\begin{array} { r l } & { \operatorname* { P r } \left( | \hat { \beta } _ { C S } - \beta _ { C S } | \ge \epsilon \right) \le 2 \exp \left( \frac { - 2 N _ { \beta } \epsilon ^ { 2 } } { ( 1 / 4 ) ^ { 2 } } \right) \le \delta } \\ & { \quad \implies \epsilon \ge \cfrac { 1 } { 4 } \sqrt { \cfrac { 1 } { 2 N _ { \beta } } \log ( \cfrac 2 \delta ) } . } \end{array}
$$

Theorem D.4. (Theorem 4.4 in the main text) Suppose the dataset $D _ { \alpha }$ is used to construct a consistent estimator of the cost-sensitive threshold $\alpha _ { a }$ (Theorem 4.1). Then, given a dataset $D _ { \beta }$ the following statements are true:

1. Consistency: $\hat { \beta } _ { C S , \hat { \alpha } } = \frac { 1 } { N _ { \beta } } \sum _ { i \in [ D _ { \beta } ] } \operatorname* { m i n } ( ( 1 - \hat { \alpha } _ { a _ { i } } ^ { + } ) c _ { i } , \hat { \alpha } _ { a _ { i } } ^ { + } ( 1 - c _ { i } ) )$ is a consistent estimator of β<sub>CS</sub>.

2. Bias: For a fixed $\delta \in ( 0 , 1 )$

• Demographic Parity: $\begin{array} { r } { | \beta _ { C S } - \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ \hat { \beta } _ { C S , \hat { \alpha } } \right] | = \mathcal { O } \left( | \lambda | \cdot \sqrt { \frac { \log ( 1 / \delta ) } { N _ { \alpha } } } \right) , } \end{array}$

• Equal Opportunity: $\begin{array} { r } { | \beta _ { C S } - \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ \hat { \beta } _ { C S , \hat { \alpha } } \right] | = \mathcal { O } \left( \frac { | \lambda | } { \tau ^ { 2 } } \sqrt { \frac { \log ( 1 / \delta ) } { N _ { \alpha } } } \right) } \end{array}$

3. Sample Complexity: With 1 − δ probability, $| \beta _ { C S } - \hat { \beta } _ { C S , \hat { \alpha } } | < \epsilon$ as long as $\begin{array} { r } { N _ { \beta } \geq \frac { 1 } { 8 \epsilon ^ { 2 } } \log ( \frac { 4 } { \delta } ) } \end{array}$ and:

• Demographic Parity: $\begin{array} { r } { N _ { \alpha } = \mathcal { O } \left( \frac { \lambda ^ { 2 } } { \epsilon ^ { 2 } } \log \left( \frac { 1 } { \delta } \right) \right) } \end{array}$

• Equal Opportunity: $\begin{array} { r } { N _ { \alpha } = \mathcal { O } \left( \frac { \lambda ^ { 2 } } { \tau ^ { 4 } \epsilon ^ { 2 } } \log \left( \frac { 1 } { \delta } \right) \right) } \end{array}$

Proof. (1) Consistency: For showing consistency, note that conditioned on $D _ { \alpha }$ , our thresholds $\hat { \alpha } _ { a } ^ { + }$ are fixed, and hence by Theorem $4 . 3 , \hat { \beta } _ { C S , \hat { \alpha } } \stackrel { p } {  } \beta _ { C S , \hat { \alpha } }$ , where $\beta _ { C S , \hat { \alpha } }$ is the cost sensitive Bayes risk with respect to group-aware thresholds $\hat { \alpha } _ { a } ^ { + } , \forall a \in { \mathcal A } .$ . It remains to show that $\beta _ { C S , \hat { \alpha } } \stackrel { p } {  } \beta _ { C S }$

From Theorem $4 . 1 , \hat { \alpha } _ { a } ^ { + } \overset { p } {  } \alpha _ { a }$ . Note that $f ( x ) = \mathbb { E } \left[ m i n \{ ( 1 - x ) \eta , x ( 1 - \eta ) \} \right]$ is a continuous function of x. Hence, by the Continuous Mapping Theorem, $\beta _ { C S , \hat { \alpha } } \stackrel { p } {  } \beta _ { C S }$

(2) Bias:

$$
\begin{array} { r l } { \left| \beta _ { C S } - \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ \hat { \beta } _ { C S , \hat { \alpha } } \right] \right| = \left| \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ \beta _ { C S } - \hat { \beta } _ { C S , \hat { \alpha } } \right] \right| } & { } \\ & { = \left| \mathbb { E } _ { D _ { \alpha } } \left[ \mathbb { E } _ { D _ { \beta } } \left[ ( \beta _ { C S } - \hat { \beta } _ { C S , \hat { \alpha } } ) | D _ { \alpha } \right] \right] \right| = \left| \mathbb { E } _ { D _ { \alpha } } \left[ \beta _ { C S } - \beta _ { C S , \hat { \alpha } } \right] \right| } \\ & { \leq \mathbb { E } _ { D _ { \alpha } } \left[ \left| \beta _ { C S } - \beta _ { C S , \hat { \alpha } } \right| \right] . } \end{array}
$$

At this point, consider the following:

$$
\begin{array} { r l } { \| f _ { C ; 2 , x } - \hat { \rho } _ { C ; \infty } \| - | \mathcal { E } _ { x , \lambda } | \operatorname* { m i n } ( \theta ( 1 - x ) , x ) - \operatorname* { m i n } ( 1 - \theta ) - \operatorname* { m i n } ( \theta ( 1 - y ) , y ( 1 - n ) ) \| } & { } \\ & { = \| \mathbb { E } \left[ \frac { 1 } { 2 } + x - x y \cdot \left( 1 - 2 \right) \left( 1 - \frac { x - y } { 2 } \right) \right] - \frac { \theta } { 2 } + x \theta + \theta \eta + \left\| \frac { y - \theta } { 2 } \right\| } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { = \| \mathrm { C i a n g ~ m i n ~ ( \theta , \theta , \phi , \phi ) = \frac { \pi + \theta } { 2 } - | \frac { \theta - \theta } { 2 } - | \mathcal { E } _ { 1 } | } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { = \frac { 1 } { 2 } \| \mathbb { E } \left[ ( x - y ) ( 1 - 2 \eta ) \right] - | \eta - x | + | \mathcal { h } - y | \| } \\ & { \quad = \frac { 1 } { 2 } \left| ( x - y ) ( 1 - 2 \mathbb { E } [ \theta ] ) + \mathbb { E } \left[ \| \eta - y \right] - | \| \ : \ : \operatorname* { m i n } - x | \right| } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \quad \quad \quad \quad \quad \quad \quad \end{array}
$$

The above result shows that the Bayes risk is Lipschitz in the threshold argument α with $L = 1$ . This result will be useful later.

We know from Theorem 4.1 that the bias of $\hat { \alpha } _ { a } ^ { + }$ is bounded: $\left| \mathbb { E } \left[ \hat { \alpha } _ { a } ^ { + } \right] - { \alpha } _ { a } \right| \leq U _ { m }$ , where $U _ { m }$ depends on the fairness metric. Therefore, setting $\boldsymbol { x } \overset { \sim } { = } \mathbb { E } _ { D _ { \alpha } } \left[ \boldsymbol { \hat { \alpha } } \right]$ and $y = \alpha _ { a }$ yields us:

$$
\begin{array} { r l } { \left| \beta _ { C S } - \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ \hat { \beta } _ { C S , \hat { \alpha } } \right] \right| \leq \mathbb { E } _ { D _ { \alpha } } \left[ \left| \beta _ { C S } - \beta _ { C S , \hat { \alpha } } \right| \right] } & { } \\ & { \leq \pi _ { 0 } \mathbb { E } _ { D _ { \alpha } } \left[ \left| \beta _ { C S } - \beta _ { C S , \hat { \alpha } _ { 0 } ^ { + } } \right| \right] + \pi _ { 1 } \mathbb { E } _ { D _ { \alpha } } \left[ \left| \beta _ { C S } - \beta _ { C S , \hat { \alpha } _ { 1 } ^ { + } } \right| \right] } \\ & { \leq \pi _ { 0 } \mathbb { E } _ { D _ { \alpha } } \left[ \left| \alpha _ { 0 } - \hat { \alpha } _ { 0 } ^ { + } \right| \right] + \pi _ { 1 } \mathbb { E } _ { D _ { \alpha } } \left[ \left| \alpha _ { 1 } - \hat { \alpha } _ { 1 } ^ { + } \right| \right] } \end{array}
$$

From Theorem 4.1, we know that with $1 - \delta$ probability, $| \hat { \alpha } _ { a } ^ { + } - \alpha _ { a } | < \epsilon$ . Using this, we get:

$$
\begin{array} { r l } & { \| \beta _ { C S } - \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } [ \hat { \beta } _ { C S , \hat { \alpha } } ] | \leq \pi _ { 0 } \mathbb { E } _ { D _ { \alpha } } [ | \alpha _ { 0 } - \hat { \alpha } _ { 0 } ^ { + } | ] + \pi _ { 1 } \mathbb { E } _ { D _ { \alpha } } [ | \alpha _ { 1 } - \hat { \alpha } _ { 1 } ^ { + } | ] } \\ & { = \displaystyle \sum _ { a \in \{ 0 , 1 \} } \pi _ { a } ( \mathbb { E } _ { D _ { \alpha } } [ | \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } | \mathbb { I } ( | \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } | < \epsilon ) ] + \mathbb { E } _ { D _ { \alpha } } [ | \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } | \mathbb { I } ( | \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } | \geq \epsilon ) ] ) } \\ & { \leq \displaystyle \sum _ { a \in \{ 0 , 1 \} } \pi _ { a } ( \epsilon \cdot \operatorname* { P r } ( | \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } | < \epsilon ) + 1 \cdot \operatorname* { P r } ( | \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } | \geq \epsilon ) ) \quad \mathrm { ( S i n c e ~ } | \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } | \leq 1 ) } \\ & { \leq \displaystyle \sum _ { a \in \{ 0 , 1 \} } \pi _ { a } ( \epsilon _ { a } ( N _ { \alpha } , \delta _ { a } ) + \delta _ { a } ) . } \end{array}
$$

where, $\epsilon _ { a } ( N _ { \alpha } , \delta _ { a } )$ can de derived from the respective sample complexities in Theorem 4.1.

For DP, using the dominant term, $\begin{array} { r } { \epsilon _ { a } = \frac { | \lambda | } { \pi _ { a } ^ { 2 } } \sqrt { \frac { \log ( 4 / \delta _ { a } ) } { 2 N _ { \alpha } } } } \end{array}$ and setting $\delta = \operatorname* { m a x } \{ \delta _ { 0 } , \delta _ { 1 } \}$ , giving us:

$$
\left| \beta _ { C S } - \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ \hat { \beta } _ { C S , \hat { \alpha } } \right] \right| = \mathcal { O } \left( | \lambda | \cdot \sqrt { \frac { \log ( 1 / \delta ) } { N _ { \alpha } } } \right) .
$$

For EOP, using the ϵ dependent term, $\begin{array} { r } { \epsilon _ { a } = \frac { | \lambda | } { \tau ^ { 2 } \pi _ { 1 a } ^ { 2 } } \sqrt { \frac { 8 \log ( 2 / \delta _ { a } ) } { N _ { \alpha } } } } \end{array}$ and setting $\delta = \operatorname* { m a x } \{ \delta _ { 0 } , \delta _ { 1 } \}$ , giving us:

$$
| \beta _ { C S } - \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ \hat { \beta } _ { C S , \hat { \alpha } } \right] | = O \left( \frac { | \lambda | } { \tau ^ { 2 } } \sqrt { \frac { \log ( 1 / \delta ) } { N _ { \alpha } } } \right) .
$$

(3) Sample Complexity: We want to ensure the following:

$$
\operatorname* { P r } \left( | \beta _ { C S } - \hat { \beta } _ { C S , \hat { \alpha } } | > \epsilon \right) \leq \delta .
$$

To do so, consider the following breakdown: $| \beta _ { C S } - \hat { \beta } _ { C S , \hat { \alpha } } | \leq | \beta _ { C S } - \beta _ { C S , \hat { \alpha } } | + | \beta _ { C S , \hat { \alpha } } - \hat { \beta } _ { C S , \hat { \alpha } } | .$ Consider the random error: $\lvert \beta _ { C S , \hat { \alpha } } - \hat { \beta } _ { C S , \hat { \alpha } } \rvert$ . This will give us a sample complexity lower bound for $N _ { \beta }$ . Bounding this is a simple application of Hoeffding’s inequality with $\bar { X _ { i } } \stackrel { \cdot } { = } \operatorname* { m i n } \{ ( 1 -$ $\hat { \alpha } _ { a _ { i } } ^ { + } ) c _ { i } ^ { \top } , \hat { \alpha } _ { a _ { i } } ^ { + } ( 1 - c _ { i } \big ) \} \in [ 0 , 1 / 4 ]$

$$
\operatorname* { P r } \Big ( \vert \beta _ { C S , \hat { \alpha } _ { a } } - \hat { \beta } _ { C S , \hat { \alpha } } \vert > \epsilon / 2 \Big ) \le 2 \exp ( - 8 N _ { \beta } \epsilon ^ { 2 } ) \le \delta / 2 \implies N _ { \beta } \ge \frac { 1 } { 8 \epsilon ^ { 2 } } \log ( \frac { 4 } { \delta } )
$$

Now consider the error due to threshold estimation. We know from our analysis of bias that $| \beta _ { C S , x } - \beta _ { C S , y } | \leq | x - y | \implies | \beta _ { C S } - \beta _ { C S , \hat { \alpha } } | \leq \sum _ { a \in \{ 0 , 1 \} } \pi _ { a } \cdot | \hat { \alpha } _ { a } ^ { + } - \alpha _ { a } | .$

From the sample complexity results in Theorem 4.1, we can derive the following: with $\mathrm { P r } \left( | \hat { \alpha } _ { a } ^ { + } - \alpha _ { a } | \hat { > } \epsilon / 2 \right) \hat { \leq } U _ { m } \bar { ( } \epsilon / 2 ) \leq \delta / 2$ , where $U _ { m }$ depends on the fairness metric we choose. This will provide us with a bound on $N _ { \alpha } \ge N _ { \alpha } ( m )$ .

For demographic parity, again using the dominant term:

$$
\mathrm { P r } \left( \lvert \hat { \alpha } _ { a } ^ { + } - \alpha _ { a } \rvert > \epsilon / 4 \pi _ { a } \right) \le \delta / 4 \Longrightarrow N _ { \alpha } \ge \frac { 8 \lambda ^ { 2 } } { \epsilon ^ { 2 } \pi _ { a } ^ { 2 } } \log \left( \frac { 1 6 } { \delta } \right) = \mathcal { O } \left( \frac { \lambda ^ { 2 } } { \epsilon ^ { 2 } } \log \left( \frac { 1 } { \delta } \right) \right) ,
$$

For equal opportunity, recall that with $1 - \delta$ probability over the choice of data sample $N _ { \alpha }$ , as long as $\begin{array} { r } { N _ { \alpha } \geq \frac { 1 } { 2 t ^ { 2 } } \log ( \frac { 2 } { \delta } ) } \end{array}$ for t = min $\begin{array} { r } { \biggr \{ \frac { \pi _ { 1 a } } { 2 } , \frac { \tau \pi _ { 1 a } ^ { 2 } } { 4 | \lambda | } , \frac { \epsilon \tau ^ { 2 } \pi _ { 1 a } ^ { 2 } } { 4 | \lambda | } \biggr \} , \mathrm { P r } \left( | \hat { \alpha } _ { a } - \alpha _ { a } | > \epsilon \right) \leq \delta } \end{array}$ . Using these bounds with our new ϵ and δ we get:

$$
\begin{array} { r l } & { \operatorname* { P r } \left( \left| \hat { \alpha } _ { a } ^ { + } - \alpha _ { a } \right| > \epsilon / 4 \pi _ { a } \right) \le \delta / 4 \implies N _ { \alpha } \ge \frac { 1 } { 2 } \log \left( \frac { 8 } { \delta } \right) \operatorname* { m a x } \left\{ \frac { 2 } { \pi _ { 1 a } } , \frac { 4 | \lambda | } { \tau \pi _ { 1 a } ^ { 2 } } , \frac { 4 | \lambda | } { \epsilon \tau ^ { 2 } \pi _ { 1 a } ^ { 2 } } \right\} ^ { 2 } } \\ & { \qquad = \mathcal { O } \left( \frac { \lambda ^ { 2 } } { \tau ^ { 4 } \epsilon ^ { 2 } } \log \left( \frac { 1 } { \delta } \right) \right) . } \end{array}
$$

Proof for Theorem 4.5: Recall from Definition 3.5 the definition of expected risk:

$$
\begin{array} { l } { { \displaystyle R ( g ) = \sum _ { a \in \{ 0 , 1 \} } \pi _ { a } \cdot \Big ( \mathbb { E } _ { X \mid A = a } \left[ ( 1 - 2 \eta ( X , a ) ) g ( X , a ) \right] \Big ) + \pi } } \\ { { \displaystyle \qquad = \mathbb { E } _ { X , A } \left[ ( 1 - 2 C ) \mathbb { I } \left( C \geq \alpha _ { A } \right) \right] + \pi } } \end{array}
$$

A finite sample estimator of the above risk can be written down as follows:

$$
\hat { R } ( \hat { g } ) = \frac { 1 } { N _ { \beta } } \sum _ { i \in [ D _ { \beta } ] } ( 1 - 2 c _ { i } ) \mathbb { I } \left( c _ { i } \ge \hat { \alpha } _ { a _ { i } } ^ { + } \right) + \hat { \pi }
$$

Theorem D.5. (Proof for Theorem 4.5 from the main text) Consider $\hat { R } ( \hat { g } )$ , the empirical estimator for the expected risk from Definition 3.5. Then, the following properties hold:

1. Consistency: ${ \hat { R } } ( { \hat { g } } ) \ { \overset { p } { \to } } \ R ( g )$

2. Bias: For a fixed $\delta \in ( 0 , 1 )$

• Demographic

$$
\begin{array} { r } { \mathcal { O } \bigg ( | \lambda | \cdot \operatorname* { m a x } \{ L _ { 0 } , L _ { 1 } \} \cdot \sqrt { \frac { \log ( 1 / \delta ) } { N _ { \alpha } } } \bigg ) , } \end{array}
$$

• Equal Opportunity: $\begin{array} { r } { | R ( g ) - \mathbb { E } _ { D _ { \alpha } , D _ { \beta } } \left[ \hat { R } ( \hat { g } ) \right] | = \mathcal { O } \left( \frac { | \lambda | \operatorname* { m a x } \{ L _ { 0 } , L _ { 1 } \} } { \tau ^ { 2 } } \sqrt { \frac { \log ( 1 / \delta ) } { N _ { \alpha } } } \right) . } \end{array}$

3. Sample Complexity: With 1 − δ probability, $| R ( g ) - \hat { R } ( \hat { g } ) | < \epsilon$ as long as $\begin{array} { r } { N _ { \beta } \ge \frac { 1 8 } { \epsilon ^ { 2 } } \log \left( \frac { 6 } { \delta } \right) } \end{array}$ and:

• Demographic Parity: $\begin{array} { r } { N _ { \alpha } = \mathcal { O } \left( \frac { \lambda ^ { 2 } \operatorname* { m a x } \{ L _ { 0 } ^ { 2 } , L _ { 1 } ^ { 2 } \} } { \epsilon ^ { 2 } } \log \left( \frac { 1 } { \delta } \right) \right) } \end{array}$

• Equal Opportunity: $\begin{array} { r } { N _ { \alpha } = \mathcal { O } \left( \frac { \lambda ^ { 2 } \operatorname* { m a x } \{ L _ { 0 } ^ { 2 } , L _ { 1 } ^ { 2 } \} } { \epsilon ^ { 2 } \tau ^ { 4 } } \log \left( \frac { 1 } { \delta } \right) \right) } \end{array}$

Proof. To prove the above results, we will use an intermediate estimator $\begin{array} { r l } { R ( \hat { g } ) } & { { } = } \end{array}$ $\mathbb { E } _ { X , A } \left[ ( 1 - 2 C ) \mathbb { I } \left( C \geq \hat { \alpha } _ { A } ^ { + } \right) \right] + \pi$

1. Consistency: We want to show the following result:

$$
\begin{array} { r } { \operatorname* { P r } \Big ( | R ( g ) - \hat { R } ( \hat { g } ) | > \epsilon \Big ) \stackrel { p } {  } 0 \mathrm { ~ a s ~ } N _ { \alpha } , N _ { \beta }  \infty . } \end{array}
$$

We can decompose the consistency gap as follows:

$$
| R ( g ) - \hat { R } ( \hat { g } ) | \leq | R ( g ) - R ( \hat { g } ) | + | R ( \hat { g } ) - \hat { R } ( \hat { g } ) |
$$

Since $D _ { \alpha }$ and $D _ { \beta }$ are independently drawn samples, conditioned on $D _ { \alpha } , \hat { \alpha } _ { a } ^ { + }$ is a constant for all $A = a$ and hence ${ \hat { R } } ( { \hat { g } } ) \ { \overset { p } { \to } } \ R ( { \hat { g } } )$ by the weak law of large numbers. It remains to show how $R ( { \hat { g } } ) \ { \overset { p } { \to } } \ R ( g )$

Let $\phi _ { a } ( t ) = \mathbb { E } _ { X | A = a } \left[ \left( 1 - 2 C \right) \mathbb { I } \left( C \geq t \right) | A = a \right] , t \in ( 0 , 1 )$ , so that $R ( g ) = \sum _ { a \in \{ 0 , 1 \} } \pi _ { a } \cdot \phi _ { a } ( \alpha _ { a } )$ Let $0 \leq u < v \leq 1$ and consider the following analysis for a fixed $A = a \colon$

$$
\begin{array} { r l } & { \left| \phi _ { a } ( u ) - \phi _ { a } ( v ) \right| = \left| \mathbb { E } \left[ ( 1 - 2 C ) ( \mathbb { I } ( C \geq u ) - \mathbb { I } ( C \geq v ) ) \middle | A = a \right] \right| } \\ & { \qquad \leq \operatorname* { P r } \left( u \leq C < v | A = a \right) \mathrm { ~ ( S i n c e ~ ( \mathbb { I } ( C \geq u ) - \mathbb { I } ( C \geq v ) ) \neq 0 ~ w h e n ~ } C \in [ u , v ) ) } \\ & { \qquad = \left| F _ { a } ( v ) - F _ { a } ( u ) \right| } \end{array}
$$

Now, from Theorem 4.1, we know that $\hat { \alpha } _ { a } ^ { + } \overset { p } { \to } \alpha _ { a }$ and since $F _ { a }$ is continuous due to Assumption 2.1 and $\alpha _ { a } > 0$ , by the continuous mapping theorem, $\phi ( \hat { \alpha } _ { a } ^ { + } ) \stackrel { p } {  } \phi ( \alpha _ { a } )$ . Since $R ( g ) = \sum _ { a \in \{ 0 , 1 \} } \pi _ { a } \cdot \phi _ { a } ( \alpha _ { a } )$

$$
R ( { \hat { g } } ) \ { \overset { p } { \to } } \ R ( g )
$$

2. Bias: We want to bound the following quantity: $| R ( g ) - \mathbb { E } _ { D _ { \alpha } , D _ { \beta } } \left[ \hat { R } ( \hat { g } ) \right] | . \operatorname { L e t } R _ { 0 } ( g ) = R ( g ) - \pi ,$ and similarly define $\begin{array} { r } { R _ { 0 } ( \hat { g } ) = R ( \hat { g } ) - \pi } \end{array}$

Note that since $D _ { \alpha }$ and $D _ { \beta }$ are independently sampled, $\mathbb { E } _ { D _ { \alpha } , D _ { \beta } } \left[ \hat { R } ( \hat { g } ) \right] = \mathbb { E } _ { D _ { \alpha } } \left[ R _ { 0 } ( \hat { g } ) \right] + \mathbb { E } _ { D _ { \alpha } } \left[ \hat { \pi } \right] ,$ which means:

$$
\begin{array} { r l } & { \left| R ( g ) - \mathbb { E } _ { D _ { \alpha } , D _ { \beta } } \left[ \hat { R } ( \hat { g } ) \right] \right| \leq \left| R _ { 0 } ( g ) - \mathbb { E } _ { D _ { \alpha } } \left[ R _ { 0 } ( \hat { g } ) \right] \right| + \left| \pi - \mathbb { E } _ { D _ { \alpha } } \left[ \hat { \pi } \right] \right| } \\ & { \qquad = \left| R _ { 0 } ( g ) - \mathbb { E } _ { D _ { \alpha } } \left[ R _ { 0 } ( \hat { g } ) \right] \right| \mathrm { ( S i n c e ~ \hat { \pi } i s ~ a n ~ u n b i a s e d ~ e s t i m a t e ~ o f ~ } \pi \mathrm { ) } } \end{array}
$$

Let $\phi _ { a } ( t ) \ = \ \mathbb { E } _ { X \mid A = a } \left[ \left( 1 - 2 C \right) \mathbb { I } \left( C \geq t \right) \big | A = a \right]$ . From our consistency proof, we know that $| \phi _ { a } ( u ) - \phi _ { a } ( v ) | \leq | F _ { a } ( v ) - F _ { a } ( u ) |$ |. Using this:

$$
\left| R ( g ) - \mathbb { E } _ { D _ { \alpha } , D _ { \beta } } \left[ \hat { R } ( \hat { g } ) \right] \right| \leq \left| R _ { 0 } ( g ) - \mathbb { E } _ { D _ { \alpha } } \left[ R _ { 0 } ( \hat { g } ) \right] \right| = \big | \sum _ { a \in \{ 0 , 1 \} } \pi _ { a } \mathbb { E } _ { D _ { \alpha } } \left[ \phi _ { a } ( \alpha _ { a } ) - \phi _ { a } ( \hat { \alpha } _ { a } ^ { + } ) \right] \big |
$$

$$
\leq \sum _ { a \in \{ 0 , 1 \} } \pi _ { a } \mathbb { E } _ { D _ { \alpha } } \left[ \left| \phi _ { a } ( \alpha _ { a } ) - \phi _ { a } ( \hat { \alpha } _ { a } ^ { + } ) \right| \right] \leq \sum _ { a \in \{ 0 , 1 \} } \pi _ { a } L _ { a } \mathbb { E } _ { D _ { \alpha } } \left[ \left| \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } \right| \right] \ ( \mathrm { U s i n g ~ A s s u m p t i o n } \ 2 . 2 )
$$

From Theorem 4.1, we know that with $1 - \delta$ probability, $| \hat { \alpha } _ { a } ^ { + } - \alpha _ { a } | < \epsilon$ We also know that $| \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } | \leq 1$ . Using this, we get:

$$
\mathbb { E } _ { D _ { \alpha } } \left[ \left| \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } \right| \right] = \mathbb { E } _ { D _ { \alpha } } \left[ \left| \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } \right| \mathbb { I } \left( \left| \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } \right| < \epsilon \right) \right] + \mathbb { E } _ { D _ { \alpha } } \left[ \left| \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } \right| \mathbb { I } \left( \left| \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } \right| \geq \epsilon \right) \right]
$$

$$
\leq \epsilon \cdot \operatorname* { P r } \left( \left| \alpha _ { a } - { \hat { \alpha } } _ { a } ^ { + } \right| < \epsilon \right) + 1 \cdot \operatorname* { P r } \left( \left| \alpha _ { a } - { \hat { \alpha } } _ { a } ^ { + } \right| \geq \epsilon \right) { \mathrm { ~ } } ( { \mathrm { S i n c e ~ } } | \alpha _ { a } - { \hat { \alpha } } _ { a } ^ { + } | \leq 1 ) \leq \epsilon + \delta .
$$

and therefore,

$$
| R ( g ) - \mathbb { E } _ { D _ { \alpha } , D _ { \beta } } \left[ \hat { R } ( \hat { g } ) \right] | \leq \sum _ { a \in \{ 0 , 1 \} } \pi _ { a } L _ { a } \mathbb { E } _ { D _ { \alpha } } \left[ | \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } | \right] \leq \sum _ { a \in \{ 0 , 1 \} } \pi _ { a } L _ { a } \big ( \epsilon _ { a } + \delta _ { a } \big ) .
$$

For DP, using the sample complexity bound from Theorem 4.1 and the dominant term, $\epsilon _ { a } =$ $\frac { | \lambda | } { \pi _ { a } ^ { 2 } } \sqrt { \frac { \log ( 4 / \delta _ { a } ) } { 2 N _ { \alpha } } }$ , and setting $\delta = \operatorname* { m i n } \{ \delta _ { 0 } , \delta _ { 1 } \}$ :

$$
\left| R ( g ) - \mathbb { E } _ { D _ { \alpha } , D _ { \beta } } \left[ \hat { R } ( \hat { g } ) \right] \right| = \mathcal { O } \left( \left| \lambda \right| \cdot \operatorname* { m a x } \{ L _ { 0 } , L _ { 1 } \} \cdot \sqrt { \frac { \log ( 1 / \delta ) } { N _ { \alpha } } } \right) .
$$

For EOP, using the sample complexity bound from Theorem 4.1 and the ϵ dependent term, $\epsilon _ { a } =$ $\frac { | \lambda | } { \tau ^ { 2 } \pi _ { 1 a } ^ { 2 } } \sqrt { \frac { 8 \log ( 4 / \delta _ { a } ) } { N _ { \alpha } } }$ , and setting $\delta = \operatorname* { m i n } \{ \delta _ { 0 } , \delta _ { 1 } \}$ :

$$
\left| R ( g ) - \mathbb { E } _ { D _ { \alpha } , D _ { \beta } } \left[ \hat { R } ( \hat { g } ) \right] \right| = \mathcal { O } \left( \frac { | \lambda | \operatorname* { m a x } \{ L _ { 0 } , L _ { 1 } \} } { { \tau ^ { 2 } } } \sqrt { \frac { \log ( 1 / \delta ) } { N _ { \alpha } } } \right) .
$$

(3) Sample Complexity: We want to ensure the following:

$$
\operatorname* { P r } \Big ( | R ( g ) - \hat { R } ( \hat { g } ) | > \epsilon \Big ) \leq \delta .
$$

Note that we can decompose the absolute error as follows via the triangle inequality:

$$
\begin{array} { r } { | R ( g ) - \hat { R } ( \hat { g } ) | \le | R _ { 0 } ( g ) - R _ { 0 } ( \hat { g } ) | + | R _ { 0 } ( \hat { g } ) - \hat { R } _ { 0 } ( \hat { g } ) | + | \hat { \pi } - \pi | . } \end{array}
$$

Applying the union bound, we get:

$$
\begin{array} { r l } & { \operatorname* { P r } \Big ( | R ( g ) - \hat { R } ( \hat { g } ) | > \epsilon \Big ) \leq \operatorname* { P r } \big ( | R _ { 0 } ( g ) - R _ { 0 } ( \hat { g } ) | > \epsilon / 3 \big ) + \operatorname* { P r } \Big ( | R _ { 0 } ( \hat { g } ) - \hat { R } _ { 0 } ( \hat { g } ) | > \epsilon / 3 \Big ) } \\ & { \qquad + \operatorname* { P r } \big ( | \hat { \pi } - \pi | > \epsilon / 3 \big ) . } \end{array}
$$

Conditioned on $D _ { \alpha } , \hat { \alpha } _ { a } ^ { + }$ is a constant and hence we can control $\mathrm { P r } \left( | R _ { 0 } ( \hat { g } ) - \hat { R } _ { 0 } ( \hat { g } ) | > \epsilon / 3 \right)$ via the Hoeffding’s inequality. Note that the random variable $Z _ { i } = ( 1 - 2 c _ { i } ) \mathbb { I } \left( c _ { i } \geq \hat { \alpha } _ { a _ { i } } ^ { + } \right) \in [ - 1 , 1 ]$

$$
\operatorname* { P r } \Big ( | R _ { 0 } ( \widehat { g } ) - \widehat { R } _ { 0 } ( \widehat { g } ) | > \epsilon / 3 | D _ { \alpha } \Big ) \leq \delta / 3 \implies N _ { \beta } \geq \frac { 1 8 } { \epsilon ^ { 2 } } \log \Big ( \frac { 6 } { \delta } \Big ) .
$$

Similarly, we can also control $\operatorname* { P r } \left( | \hat { \pi } - \pi | > \epsilon / 3 \right)$

$$
\operatorname* { P r } \left( | { \hat { \pi } } - \pi | > \epsilon / 3 \right) \leq \delta / 3 \implies N _ { \alpha } \geq { \frac { 9 } { 2 \epsilon ^ { 2 } } } \log \left( { \frac { 6 } { \delta } } \right)
$$

Finally, to control $\operatorname* { P r } \left( | R _ { 0 } ( g ) - R _ { 0 } ( \hat { g } ) | > \epsilon / 3 \right)$ , we can borrow our analysis from the previous proof of bias:

$$
| R _ { 0 } ( g ) - R _ { 0 } ( \hat { g } ) | \leq \sum _ { a \in \{ 0 , 1 \} } \pi _ { a } L _ { a } | \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } | .
$$

By union bound and splitting the error budget between each group-dependent term (since there are two groups, if the sum exceeds $\epsilon / 3 ,$ at least one term exceeds $\epsilon / 6 )$

$$
\operatorname* { P r } \left( | R _ { 0 } ( g ) - R _ { 0 } ( \hat { g } ) | > \epsilon / 3 \right) \le \sum _ { a \in \{ 0 , 1 \} } \operatorname* { P r } \left( | \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } | > \frac { \epsilon } { 6 \pi _ { a } L _ { a } } \right) .
$$

For demographic parity, again using the dominant term:

$$
\operatorname* { P r } \left( \lvert \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } \rvert > \frac { \epsilon } { 6 \pi _ { a } L _ { a } } \right) \leq \delta / 6 \implies N _ { \alpha } \geq \frac { 1 8 \lambda ^ { 2 } L _ { a } ^ { 2 } } { \epsilon ^ { 2 } \pi _ { a } ^ { 2 } } \log { \left( \frac { 2 4 } { \delta } \right) } = \mathcal { O } \left( \frac { \lambda ^ { 2 } L _ { a } ^ { 2 } } { \epsilon ^ { 2 } } \log { \left( \frac { 1 } { \delta } \right) } \right) .
$$

For equal opportunity, recall that with $1 - \delta$ probability over the choice of data sample $N _ { \alpha }$ , as long as $\begin{array} { r } { N _ { \alpha } \geq \frac { 1 } { 2 t ^ { 2 } } \log ( \frac { 2 } { \delta } ) } \end{array}$ for t = min $\begin{array} { r } { \biggr \{ \frac { \pi _ { 1 a } } { 2 } , \frac { \tau \pi _ { 1 a } ^ { 2 } } { 4 | \lambda | } , \frac { \epsilon \tau ^ { 2 } \pi _ { 1 a } ^ { 2 } } { 4 | \lambda | } \biggr \} , \mathrm { P r } \left( | \hat { \alpha } _ { a } - \alpha _ { a } | > \epsilon \right) \leq \delta } \end{array}$ . Using these bounds with our new ϵ and δ we get:

$$
\begin{array} { r l } & { \operatorname* { P r } \left( \left| \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } \right| > \displaystyle \frac { \epsilon } { 6 \pi _ { a } L _ { a } } \right) \leq \delta / 6 \implies N _ { \alpha } \geq \frac { 1 } { 2 } \log \left( \frac { 1 2 } { \delta } \right) \operatorname* { m a x } \left\{ \displaystyle \frac { 2 } { \pi _ { 1 a } } , \displaystyle \frac { 4 | \lambda | } { \tau \pi _ { 1 a } ^ { 2 } } , \displaystyle \frac { 2 4 | \lambda | \pi _ { a } L _ { a } } { \epsilon \tau ^ { 2 } \pi _ { 1 a } ^ { 2 } } \right\} ^ { 2 } } \\ & { \qquad = \mathcal { O } \left( \displaystyle \frac { \lambda ^ { 2 } L _ { a } ^ { 2 } } { \epsilon ^ { 2 } \tau ^ { 4 } } \log \left( \frac { 1 } { \delta } \right) \right) . } \end{array}
$$

Taking a max over $L _ { a }$ completes our proof.

Again, for convenience and readability, we will prove the DP and EOP results for Theorem $4 . 6 .$ Theorem D.6. (Proofofthe DP part ofTheorem 4.6 in the main text) Suppose a dataset $D _ { \beta }$ is used to construct $\hat { \Delta } _ { D P } ( g _ { \hat { \alpha } _ { a } ^ { + } } )$ , where $\hat { \alpha } _ { a } ^ { + }$ is the estimator for the DP threshold (Theorem 4.1). Then the following holds:

1. Consistency: $\hat { \Delta } _ { D P } ( g _ { \hat { \alpha } _ { a } ^ { + } } ) \stackrel { p } {  } \Delta _ { D P } ( g _ { \alpha } ) )$

$$
\begin{array} { r l r l r l } { { 2 } . } & { B i a s : } & { F o r } & { \delta } & { \in } & { ( 0 , 1 ) , } & { | \Delta _ { D P } ( g ) } & { { } - } & { \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ \hat { \Delta } _ { D P } ( \hat { g } ) \right] | } & { } & { { } = } \\ { \mathcal { O } \left( \frac { 1 } { \sqrt { N _ { \beta } } } + | \lambda | \cdot \operatorname* { m a x } \{ L _ { 0 } , L _ { 1 } \} \sqrt { \frac { \log ( 1 / \delta _ { \alpha } ) } { N _ { \alpha } } } \right) . } & { } & { { } } & { } \end{array}
$$

3. Sample Complexity: With $1 - \delta$ probability, $| \hat { \Delta } ( \hat { g } ) - \Delta ( g ) | \leq \epsilon ,$ as long as $N _ { \beta } \ =$ O   <sup>1</sup><sub>ϵ2</sub> log( <sup>1</sup><sub>δ</sub> ) and N<sub>α</sub> = O  <sup>λ2·max{L2</sup>0<sup>,L2</sup>1<sup>}</sup><sub>ϵ2</sub> log   <sup>1</sup><sub>δ</sub> .

Proof. For the sake of this proof, we will denote $\Delta _ { D P } ( g _ { \alpha } )$ as $\Delta ( g )$ and $\hat { \Delta } _ { D P } ( g _ { \hat { \alpha } _ { a } ^ { + } } )$ as $\hat { \Delta } ( \hat { g } )$ . Then an intermediate estimator $\hat { \Delta } _ { D P } ( g _ { \alpha _ { a } } )$ can be denoted as $\hat { \Delta } ( g )$ . We similarly overload the notations in this fashion for the positive rate estimator $\scriptstyle \rho _ { \mathrm { P R } , a }$ with $\rho _ { a }$

(1) Consistency: Most of the inequality arguments in the proof are a direct consequence of the triangle inequality, by introducing an intermediate estimate of the PR $( \hat { \rho } _ { a } ( g ) )$ .

We want to show that P $\mathrm { ~ r ~ } \Big ( | \hat { \Delta } ( \hat { g } ) - \Delta ( g ) | > \epsilon \Big )  0$ as $N _ { \beta }  \infty$ . By Slutsky’s theorem, it will be enough to upper-bound our target event as a sum of simpler events and show that if each of them converges in probability to zero, then we can claim our final result.

Consider the absolute difference term and apply the reverse triangle inequality $( | | A | - | B | | \leq | A - B | )$ :

$$
\begin{array} { r l } & { | \Delta ( g ) - \hat { \Delta } ( \hat { g } ) | \leq | \rho _ { 1 } ( g ) - \rho _ { 0 } ( g ) - \hat { \rho } _ { 1 } ( \hat { g } ) + \hat { \rho } _ { 0 } ( \hat { g } ) | } \\ & { \leq | \rho _ { 1 } ( g ) - \hat { \rho } _ { 1 } ( g ) | + | \rho _ { 0 } ( g ) - \hat { \rho } _ { 0 } ( g ) | + | \hat { \rho } _ { 1 } ( g ) - \hat { \rho } _ { 1 } ( \hat { g } ) | + | \hat { \rho } _ { 0 } ( g ) - \hat { \rho } _ { 0 } ( \hat { g } ) | . } \end{array}
$$

Since similar arguments will work for any group $A = a ,$ , we will now study each term separately for an arbitrary group $A = a$

Consider $| \rho _ { a } ( g ) - \hat { \rho } _ { a } ( g ) |$ . By the Weak law of large numbers, $\begin{array} { r } { \frac { 1 } { N _ { \beta , a } } \sum _ { i \in [ D _ { \beta , a } ] } \mathbb { I } \left( c _ { i } \geq \alpha _ { a } \right) \ \xrightarrow { p } } \end{array}$ $\mathbb { E } _ { X \mid A = a } \left[ \mathbb { I } \left( c _ { a } \geq \alpha _ { a } \right) \right]$ , and hence $ \rho _ { a } ( g ) - \hat { \rho } _ { a } ( g )  \overset { p } {  } 0$

Next, consider the term $| \hat { \rho } _ { a } ( g ) - \hat { \rho } _ { a } ( \hat { g } )$ |. We will now attempt to study the convergence in probability properties of this term:

$$
| \hat { \rho } _ { a } ( g ) - \hat { \rho } _ { a } ( \hat { g } ) | = \left| \frac { 1 } { N _ { \beta , a } } \sum _ { i \in [ D _ { \beta , a } ] } \mathbb { I } \left( c _ { i } \geq \alpha _ { a } \right) - \frac { 1 } { N _ { \beta , a } } \sum _ { i \in [ D _ { \beta , a } ] } \mathbb { I } \left( c _ { i } \geq \hat { \alpha } _ { a } ^ { + } \right) \right| \leq \frac { 1 } { N _ { \beta , a } } \sum _ { i \in [ D _ { \beta , a } ] } Z _ { i } ,
$$

where $Z _ { i } = | \mathbb { I } ( c _ { i } \ge \alpha _ { a } ) - \mathbb { I } ( c _ { i } \ge \hat { \alpha } _ { a } ^ { + } ) | = \mathbb { I } ( c _ { i } \in ( \operatorname* { m i n } \{ \hat { \alpha } _ { a } ^ { + } , \alpha _ { a } \} , \operatorname* { m a x } \{ \hat { \alpha } _ { a } ^ { + } , \alpha _ { a } \} ) .$

We will now show that the numerator converges to zero. Note that conditioned on $D _ { \alpha } , \ \hat { \alpha } _ { a } ^ { + }$ is fixed. Since $D _ { \alpha }$ and $D _ { \beta }$ are sampled independently, $\{ Z _ { i } , i ~ \in ~ [ D _ { \beta , a } ] \}$ are iid and bounded. Hence, by the weak law of large numbers, $\begin{array} { r } { \frac { 1 } { N _ { \beta , a } } \sum _ { i \in [ D _ { \beta , a } ] } Z _ { i } \quad \xrightarrow { p } } \end{array}$ $\begin{array} { r l r l r } { \mathbb { E } _ { D _ { \delta , a } } \left[ Z _ { i } | D _ { \alpha } , A = a \right] } & { { } = { } } & { \mathbb { E } _ { D _ { \delta , a } } \left[ \mathbb { I } \left( c _ { i } \in \left( \operatorname* { m i n } \{ \hat { \alpha } _ { a } ^ { + } , \alpha _ { a } \} , \operatorname* { m a x } \{ \hat { \alpha } _ { a } ^ { + } , \alpha _ { a } \} \right) \right) | D _ { \alpha } , A = a \right] } & { { } = \frac { \alpha + c _ { i } \alpha _ { i } } { 2 } } \end{array}$ $\operatorname* { P r } \left( c \in ( \operatorname* { m i n } \{ \hat { \alpha } _ { a } ^ { + } , \alpha _ { a } \} , \operatorname* { m a x } \{ \hat { \alpha } _ { a } ^ { + } , \alpha _ { a } \} ) | D _ { \alpha } , A = a \right) \ = \ | F _ { a } ( \alpha _ { a } ) \ - \ F _ { a } ( \hat { \alpha } _ { a } ^ { + } ) |$ , where $F _ { a }$ is the group-conditional CDF of $c = \eta ( x , a )$ , and the absolute difference accounts for both scenarios $( \hat { \alpha } _ { a } ^ { + } > \alpha _ { a }$ and $\hat { \alpha } _ { a } ^ { + } \leq \alpha _ { a } )$

We are now ready to apply the Continuous Mapping Theorem, since $F _ { a }$ is continuous by Assumption 2.1. Since we ensure $\alpha _ { a } \in ( 0 , 1 )$ , by CMT, $F _ { a } ( \hat { \alpha } _ { a } ^ { + } ) \stackrel { p } {  } F _ { a } ( \alpha _ { a } )$ , since $\hat { \alpha } _ { a } ^ { + } \overset { p } { \to } \alpha _ { a }$ (Theorem 4.1), and therefore $\begin{array} { r } { \frac { 1 } { N _ { \beta , a } } \sum _ { i \in [ N _ { \beta , a } ] } Z _ { i } \stackrel { p } { \to } 0 } \end{array}$ and:

$$
| \hat { \rho } _ { a } ( g ) - \hat { \rho } _ { a } ( \hat { g } ) | \leq \frac { 1 } { N _ { \beta , a } } \sum _ { i \in [ N _ { \beta , a } ] } Z _ { i } \overset { p } {  } | F _ { a } ( \alpha _ { a } ) - F _ { a } ( \hat { \alpha } _ { a } ^ { + } ) | \overset { p } {  } 0 .
$$

With this, we now have that all our components of $| \Delta ( g ) - \hat { \Delta } ( \hat { g } ) |$ | converge to zero in probability and hence we can claim that $\mathrm { P r } ( | \hat { \Delta } _ { D P } ( \hat { g } ) - \Delta _ { D P } ( g ) | > \epsilon )  0 \mathrm { a s } N _ { \beta }  \infty$

(2) Bias: We now want to bound the bias of our estimator, $\left| \Delta _ { D P } ( g ) - \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ \hat { \Delta } _ { D P } ( \hat { g } ) \right] \right.$ |.

$$
\begin{array} { r l } & { \left| \Delta _ { D P } ( g ) - \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ \hat { \Delta } _ { D P } ( \hat { g } ) \right] \right| } \\ & { = \left| \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ \Delta _ { D P } ( g ) - \hat { \Delta } _ { D P } ( \hat { g } ) \right] \right| \leq \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ | \hat { \Delta } _ { D P } ( \hat { g } ) - \Delta _ { D P } ( g ) | \right] } \\ & { \leq \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ | \hat { \rho } _ { 1 } ( \hat { g } ) - \hat { \rho } _ { 0 } ( \hat { g } ) - \rho _ { 1 } ( g ) + \rho _ { 0 } ( g ) | \right] } \end{array}
$$

(By using the Reverse Triangle Inequality)

$$
\begin{array} { r l } & { \leq \mathbb { E } _ { D _ { \beta , 1 } , D _ { \alpha } } \left[ | \rho _ { 1 } ( g ) - \hat { \rho } _ { 1 } ( g ) | \right] + \mathbb { E } _ { D _ { \beta , 0 } , D _ { \alpha } } \left[ | \rho _ { 0 } ( g ) - \hat { \rho } _ { 0 } ( g ) | \right] } \\ & { + \mathbb { E } _ { D _ { \beta , 1 } , D _ { \alpha } } \left[ | \hat { \rho } _ { 1 } ( g ) - \hat { \rho } _ { 1 } ( \hat { g } ) | \right] + \mathbb { E } _ { D _ { \beta , 0 } , D _ { \alpha } } \left[ | \hat { \rho } _ { 0 } ( g ) - \hat { \rho } _ { 0 } ( \hat { g } ) | \right] } \end{array}
$$

$$
\hat { \rho } _ { a } ( g )
$$

and noting that the expectation is over group-specific samples $D _ { \beta . a } )$

Once again, we will only focus on deriving the statistical bias for an arbitrary group $A = a$ . Consider the first term. At this point, note that $\hat { \rho } _ { a } ( g )$ is an unbiased estimate of $\rho _ { a } ( g )$ , and hence by Lemma B.2,

$$
\mathbb { E } _ { D _ { \beta , a } , D _ { \alpha } } \left[ \left| \hat { \rho } _ { a } ( g ) - \rho _ { a } ( g ) \right| \right] \leq \sqrt { V a r ( \hat { \rho } _ { a } ( g ) ) } \leq \frac { 1 } { 2 \sqrt { N _ { \beta , a } } } ,
$$

where the last step follows due to Popoviciu’s Inequality on $\mathbb { I } \left( c _ { i } \geq \alpha _ { a } \right)$

We can now focus on the second term: E ${ D _ { \beta , a } } , { D _ { \alpha } } \left[ \left| \hat { \rho } _ { a } ( g ) - \hat { \rho } _ { a } ( \hat { g } ) \right| \right]$

$$
\begin{array} { r l r } {  { \mathbb { E } _ { D _ { \beta , a } , D _ { \alpha } } [ | \hat { \rho } _ { a } ( \hat { g } ) - \hat { \rho } _ { a } ( g ) | ] = \mathbb { E } _ { D _ { \beta , a } , D _ { \alpha } } [ | \frac { 1 } { N _ { \beta , a } } \sum _ { i \in [ D _ { \beta , a } ] } ( \mathbb { I } ( c _ { i } \geq \hat { \alpha } _ { a } ^ { + } ) - \mathbb { I } ( c _ { i } \geq \alpha _ { a } ) ) | ] } } \\ & { } & { \leq \mathbb { E } _ { D _ { \beta , a } , D _ { \alpha } } [ \frac { 1 } { N _ { \beta , a } } \sum _ { i \in [ D _ { \beta , a } ] } | \mathbb { I } ( c _ { i } \geq \hat { \alpha } _ { a } ^ { + } ) - \mathbb { I } ( c _ { i } \geq \alpha _ { a } ) | ] } \end{array}
$$

From our consistency analysis, we know that $\begin{array} { r l r } { Z _ { i } } & { { } = } & { | \mathbb { I } \left( c _ { i } \ge \alpha _ { a } \right) ~ - ~ \mathbb { I } \left( c _ { i } \ge \hat { \alpha } _ { a } ^ { + } \right) | ~ = } \end{array}$ $\mathbb { I } \left( c _ { i } \in ( \operatorname* { m i n } \{ \hat { \alpha } _ { a } ^ { + } , \alpha _ { A } \} , \operatorname* { m a x } \{ \hat { \alpha } _ { a } ^ { + } , \alpha _ { a } \} ) \right)$ , and hence:

$$
\begin{array} { r l } & { \mathbb { E } _ { D _ { \beta , a } , D _ { \alpha } } \left[ \displaystyle \frac { 1 } { N _ { \beta , a } } \sum _ { i \in [ D _ { \beta , a } ] } Z _ { i } \right] } \\ & { = \displaystyle \frac { 1 } { N _ { \beta , a } } \sum _ { i \in [ D _ { \beta , a } , D _ { \alpha } } [ Z _ { i } ] = \mathbb { E } _ { D _ { \beta , a } , D _ { \alpha } } \left[ \mathbb { I } \left( c _ { i } \in ( \operatorname* { m i n } \{ \hat { \alpha } _ { a } ^ { + } , \alpha _ { a } \} , \operatorname* { m a x } \{ \hat { \alpha } _ { a } ^ { + } , \alpha _ { a } \} ) \right) \right] } \\ & { = \mathbb { E } _ { D _ { \alpha } } \left[ \mathbb { E } _ { D _ { \beta , a } } \left[ \mathbb { I } \left( c _ { i } \in ( \operatorname* { m i n } \{ \hat { \alpha } _ { a } ^ { + } , \alpha _ { a } \} , \operatorname* { m a x } \{ \hat { \alpha } _ { a } ^ { + } , \alpha _ { a } \} ) \right) \right] | D _ { \alpha } \right] = \mathbb { E } _ { D _ { \alpha } } \left[ | F _ { a } ( \alpha _ { a } ) - F _ { a } ( \hat { \alpha } _ { a } ^ { + } ) | \right] . } \end{array}
$$

For a fixed $\gamma \in ( 0 , 1 )$ , let $\mathcal { G } = \left\{ \left| \hat { \alpha } _ { a } ^ { + } - \alpha _ { a } \right| \leq \gamma _ { a } \right\}$ be the good event that the threshold error is at most $\gamma _ { a }$ . Then,

$$
\begin{array} { r l } & { \mathbb { E } _ { D _ { \beta , a } , D _ { \alpha } } \left[ | \hat { \rho } _ { a } ( \hat { g } ) - \hat { \rho } _ { a } ( g ) | \right] \leq \mathbb { E } _ { D _ { \alpha } } \left[ | F _ { a } ( \alpha _ { a } ) - F _ { a } ( \hat { \alpha } _ { a } ^ { + } ) | \right] } \\ & { \ = \mathbb { E } _ { D _ { \alpha } } \left[ | F _ { a } ( \alpha _ { a } ) - F _ { a } ( \hat { \alpha } _ { a } ^ { + } ) | \mathbb { 1 } g \right] + \mathbb { E } _ { D _ { \alpha } } \left[ | F _ { a } ( \alpha _ { a } ) - F _ { a } ( \hat { \alpha } _ { a } ^ { + } ) | \mathbb { 1 } _ { g ^ { c } } \right] } \\ & { \ \leq L _ { a } \gamma _ { a } + \operatorname* { P r } \left( \mathcal { G } ^ { c } \right) \ \left( \operatorname { U s i n g } \operatorname { A s s u m p t i o n } 2 . 2 \mathrm { ~ a n d ~ } \operatorname* { u s i n g } | F _ { a } ( \alpha _ { a } ) - F _ { a } ( \hat { \alpha } _ { a } ^ { + } ) | \leq 1 \right) } \\ & { \ = L _ { a } \gamma _ { a } + \operatorname* { P r } \left( | \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } | > \gamma _ { a } \right) . } \end{array}
$$

Overall,

$$
\left| \Delta _ { D P } ( g ) - \mathbb { E } _ { D _ { \delta } , D _ { \alpha } } \left[ \hat { \Delta } _ { D P } ( \hat { g } ) \right] \right| \leq \sum _ { a \in \{ 0 , 1 \} } \left( \frac { 1 } { 2 \sqrt { N _ { \beta , a } } } + L _ { a } \gamma _ { a } + \operatorname* { P r } \left( | \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } | > \gamma _ { a } \right) \right) .
$$

Using the sample complexity bound from Theorem 4.1 and the dominant term, $\begin{array} { r } { \gamma _ { a } = \frac { | \lambda | } { \pi _ { a } ^ { 2 } } \sqrt { \frac { \log ( 4 / \delta _ { a } ) } { 2 N _ { \alpha } } } } \end{array}$ giving us:

$$
\left| \Delta _ { D P } ( g ) - \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ \hat { \Delta } _ { D P } ( \hat { g } ) \right] \right| \leq \sum _ { a \in \{ 0 , 1 \} } \left( \frac { 1 } { 2 \sqrt { N _ { \beta , a } } } + \frac { L _ { a } | \lambda | } { \pi _ { a } ^ { 2 } } \sqrt { \frac { \log ( 4 / \delta _ { a } ) } { 2 N _ { \alpha } } } + \delta _ { a } \right) .
$$

Setting $N _ { \beta } = \operatorname* { m i n } \{ N _ { \beta , 0 } , N _ { \beta , 1 } \}$ , we can obtain:

$$
\left| \Delta _ { D P } ( g ) - \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ \hat { \Delta } _ { D P } ( \hat { g } ) \right] \right| = \mathcal { O } \left( \frac { 1 } { \sqrt { N _ { \beta } } } + | \lambda | \cdot \operatorname* { m a x } \{ L _ { 0 } , L _ { 1 } \} \sqrt { \frac { \log ( 1 / \delta _ { a } ) } { N _ { \alpha } } } \right) .
$$

(3) Sample Complexity: To bound the sample complexity, we need to bound the following event:

$$
\operatorname* { P r } \left( | \hat { \Delta } ( \hat { g } ) - \Delta ( g ) | > \epsilon \right) \leq \delta .
$$

Let $\Delta ( \hat { g } )$ denote the true DP difference with respect to the estimated threshold $\hat { \alpha } _ { a } ^ { + }$ . Consider the following quantity and its breakdown via the triangle inequality:

$$
| \hat { \Delta } ( \hat { g } ) - \Delta ( g ) | \leq | \hat { \Delta } ( \hat { g } ) - \Delta ( \hat { g } ) | + | \Delta ( \hat { g } ) - \Delta ( g ) | \mathrm { ( B y ~ T r i a n g l e ~ I n e q u a l i t y ) }
$$

By the union bound, we can upper bound our total error

$$
\operatorname* { P r } \left( | \hat { \Delta } ( \hat { g } ) - \Delta ( g ) | > \epsilon \right) \leq \operatorname* { P r } \left( | \hat { \Delta } ( \hat { g } ) - \Delta ( \hat { g } ) | > \epsilon / 2 \right) + \operatorname* { P r } \left( | \Delta ( \hat { g } ) - \Delta ( g ) | > \epsilon / 2 \right) .
$$

We can first use the Hoeffding bound for the first term by noting that $\begin{array} { r } { \vert \hat { \Delta } ( \hat { g } ) - \Delta ( \hat { g } ) \vert \leq \sum _ { a \in \mathcal { A } } \vert \rho _ { a } ( \hat { g } ) - } \end{array}$ $\hat { \rho } _ { a } ( \hat { g } ) |$ :

$$
\operatorname* { P r } { \big ( } | \rho _ { a } ( { \widehat { g } } ) - { \widehat { \rho } } _ { a } ( { \widehat { g } } ) | > \epsilon / 4 { \big ) } \leq 2 \exp { \Big ( } { \frac { - 2 N _ { \beta , a } \epsilon ^ { 2 } } { 1 6 } } { \Big ) } \leq { \frac { \delta } { 4 } } \implies N _ { \beta , a } \geq { \frac { 8 \log ( { \frac { 8 } { \delta } } ) } { \epsilon ^ { 2 } } } .
$$

For both groups, we would need $\begin{array} { r } { N _ { \beta } = \mathcal { O } \left( \frac { 1 } { \epsilon ^ { 2 } } \log ( \frac { 1 } { \delta } ) \right) } \end{array}$ .

Next, consider the second term:

$$
\begin{array} { r l } & { | \Delta ( \hat { g } ) - \Delta ( g ) | \le \displaystyle \sum _ { a \in \mathcal { A } } | \rho _ { a } ( g ) - \rho _ { a } ( \hat { g } ) | = \displaystyle \sum _ { a \in \mathcal { A } } | \operatorname* { P r } _ { \tau } ( C \ge \alpha _ { a } | A = a ) - \operatorname* { P r } _ { \tau } \left( C \ge \hat { \alpha } _ { a } ^ { + } | A = a \right) ) | } \\ & { \quad \quad \quad \quad = \displaystyle \sum _ { a \in \mathcal { A } } | 1 - F _ { a } ( \alpha _ { a } ) - 1 + F _ { a } ( \hat { \alpha } _ { a } ^ { + } ) | \le \displaystyle \sum _ { a \in \mathcal { A } } L _ { a } \cdot | \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } | \ ( \mathrm { U s i n g ~ A s s u m p t i o n ~ } 2 . 2 ) . } \end{array}
$$

Now, we can figure out the sample complexity of each term:

$$
\operatorname* { P r } { ( | \rho _ { a } ( g ) - \rho _ { a } ( \hat { g } ) | > \epsilon / 4 ) } = \operatorname* { P r } { \biggl ( } | \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } | > \frac { \epsilon } { 4 L _ { a } } { \biggr ) } < \frac { \delta } { 4 } .
$$

Now, we can use Theorem 4.1 to get the number of samples required to upper bound the above error: $\begin{array} { r } { N _ { \alpha } = \mathcal { O } \left( \frac { \lambda ^ { 2 } \cdot \operatorname* { m a x } \{ L _ { 0 } ^ { 2 } , L _ { 1 } ^ { 2 } \} } { \epsilon ^ { 2 } } \log \left( \frac { 1 } { \delta } \right) \right) } \end{array}$ □

Theorem D.7. (Proof of the EOP part of Theorem 4.6 in the main text) Suppose a dataset $D _ { \beta }$ is used to construct $\hat { \Delta } _ { E O P } ( g _ { \hat { \alpha } _ { a } ^ { + } } )$ , where $\hat { \alpha } _ { a } ^ { + }$ is the estimatorfor the EOP threshold (Theorem 4.1). Then the following holds:

1. Consistency: $\hat { \Delta } _ { E O P } ( g _ { \hat { \alpha } _ { n } ^ { + } } ) \stackrel { p } {  } \Delta _ { E O P } ( g _ { \alpha } ) .$

$$
\begin{array} { r } { 2 . \ B i a s \colon | \ \Delta _ { E O P } ( g ) - \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ \hat { \Delta } _ { E O P } ( \hat { g } ) \right] | = { \cal O } \left( \frac { | \lambda | \cdot \operatorname* { m a x } \{ L _ { 0 } , L _ { 1 } \} } { \tau ^ { 2 } } \sqrt { \frac { \log ( 1 / \delta ) } { N _ { \alpha } } } + \frac { 1 } { \sqrt { N _ { \beta } } } \right) . } \end{array}
$$

3. Sample Complexity: With $1 - \delta$ probability, $| \hat { \Delta } ( \hat { g } ) - \Delta ( g ) | \leq \epsilon ,$ as long as $N _ { \alpha } \ =$ $\begin{array} { r } { \mathcal { O } \left( \frac { \lambda ^ { 2 } \operatorname* { m a x } \{ L _ { 0 } ^ { 2 } , L _ { 1 } ^ { 2 } \} } { \epsilon ^ { 2 } \tau ^ { 4 } } \log \left( \frac { 1 } { \delta } \right) \right) } \end{array}$ and $\begin{array} { r } { N _ { \beta } = \mathcal { O } \left\{ \frac { 1 } { \epsilon ^ { 2 } } \log \left( \frac { 1 } { \delta } \right) \right\} } \end{array}$

Proof. For the sake of this proof, we will denote $\Delta _ { E O P } ( g _ { \alpha } )$ as $\Delta ( g )$ and $\hat { \Delta } _ { E O P } ( g _ { \hat { \alpha } _ { a } ^ { + } } )$ as $\hat { \Delta } ( \hat { g } )$ Then an intermediate estimator $\hat { \Delta } _ { E O P } ( g _ { \alpha _ { a } } )$ can be denoted as $\hat { \Delta } ( g )$ . We similarly overload the notations in this fashion for the positive rate estimator $\rho _ { \mathrm { T P R } , a }$ with $\rho _ { a }$

Recalling the definitions of our estimations,

$$
\begin{array} { r l } & { \rho _ { \mathrm { T P R } , a } ( g _ { \alpha } ) = \operatorname* { P r } \left( g _ { \alpha } = 1 | Y = 1 , A = a \right) = \frac { \mathbb { E } _ { X | A = a } \left[ \mathbb { I } \left( \eta ( x , a ) \geq \alpha _ { a } \right) \cdot \operatorname* { P r } \left( Y = 1 | X = x , A = a \right) \right] } { \pi _ { 1 | a } } } \\ & { \quad \quad \quad = \frac { \mathbb { E } _ { X | A = a } \left[ \mathbb { I } \left( C \geq \alpha _ { a } \right) \cdot C \right] } { \pi _ { 1 | a } } , } \end{array}
$$

and,

$$
\hat { \rho } _ { \mathrm { T P R } , a } ( g _ { \hat { \alpha } _ { a } ^ { + } } ) = \frac { \frac { 1 } { N _ { \beta , a } } \sum _ { i \in [ D _ { \beta , a } ] } ( c _ { i } ( \mathbb { I } \left( c _ { i } \geq \hat { \alpha } _ { a } ^ { + } \right) ) } { \hat { \pi } _ { 1 | a } ^ { + } } = \frac { \hat { \mathbf { E } } _ { X | A = a } [ \mathbb { I } \left( c \geq \hat { \alpha } _ { a } ^ { + } \right) c ] } { \hat { \pi } _ { 1 | a } ^ { + } }
$$

(1) Consistency: Most of the inequality arguments in the proof are a direct consequence of the triangle inequality, by introducing an intermediate estimate of the TPR $( \hat { \rho } _ { a } ( g ) )$ .

We want to show that $\mathrm { P r } \left( | \hat { \Delta } ( \hat { g } ) - \Delta ( g ) | > \epsilon \right) \to 0$ as $N _ { \beta }  \infty$ . By Slutsky’s theorem, it will be enough to upper-bound our target event as a sum of simpler events and show that if each of them converges in probability to zero, then we can claim our final result.

Consider the absolute difference term and apply the reverse triangle inequality $( | | A | - | B | | \leq | A - B | )$ :

$$
\begin{array} { r l } & { | \Delta ( g ) - \hat { \Delta } ( \hat { g } ) | \leq | \rho _ { 1 } ( g ) - \rho _ { 0 } ( g ) - \hat { \rho } _ { 1 } ( \hat { g } ) + \hat { \rho } _ { 0 } ( \hat { g } ) | } \\ & { \leq | \rho _ { 1 } ( g ) - \rho _ { 1 } ( \hat { g } ) | + | \rho _ { 0 } ( g ) - \rho _ { 0 } ( \hat { g } ) | + | \rho _ { 1 } ( \hat { g } ) - \hat { \rho } _ { 1 } ( \hat { g } ) | + | \rho _ { 0 } ( \hat { g } ) - \hat { \rho } _ { 0 } ( \hat { g } ) | . } \end{array}
$$

Since similar arguments will work for any group $A = a$ , we will now study each term separately for an arbitrary group $A = a$

Consider $| \rho _ { a } ( \hat { g } ) - \hat { \rho } _ { a } ( \hat { g } ) | . \hat { \mathrm { E } } _ { X | A = a } [ \mathbb { I } \left( c \geq \hat { \alpha } _ { a } ^ { + } \right) c ] \to \mathrm { E } _ { X | A = a } [ \mathbb { I } \left( c \geq \hat { \alpha } _ { a } ^ { + } \right) c ]$ and $\hat { \pi } _ { 1 | a } ^ { + } \overset { p } {  } \pi _ { 1 | a }$ due to the weak law of large numbers, since conditioned on $D _ { \alpha } , \hat { \alpha } _ { a } ^ { + }$ and $\hat { \pi } _ { 1 | a } ^ { + }$ are fixed numbers.

Since $\pi _ { 1 | a } > 0$ by assumption, by Slutsky’s theorem, $\begin{array} { r } { \hat { \rho } _ { a } \bigl ( \hat { g } \bigr ) = \frac { \hat { \mathrm { E } } _ { X \mid A = a } [ \mathbb { I } ( c \geq \hat { \alpha } _ { a } ^ { + } ) c ] } { \hat { \pi } _ { 1 \mid a } ^ { + } } \stackrel { p } {  } \rho _ { a } \bigl ( \hat { g } \bigr ) . } \end{array}$

Next, consider the term $| \rho _ { a } ( \hat { g } ) - \rho _ { a } ( g )$ |. Since the denominator for both is a constant $\pi _ { 1 | a : }$ , we will focus on showing that the difference in the numerator converges to zero.

$$
\left| \mathbb { E } _ { X \mid A = a } \left[ \mathbb { I } \left( C \geq \alpha _ { a } \right) C \right] - \mathbb { E } _ { X \mid A = a } \left[ \mathbb { I } \left( C \geq \hat { \alpha } _ { a } ^ { + } \right) C \right] \right| \leq \mathbb { E } _ { X \mid A = a } \left[ Z \right]
$$

where $Z = | \mathbb { I } \left( C \geq \alpha _ { a } \right) - \mathbb { I } \left( C \geq \hat { \alpha } _ { a } ^ { + } \right) | = \mathbb { I } \left( C \in \left( \operatorname* { m i n } \{ \hat { \alpha } _ { a } ^ { + } , \alpha _ { a } \} , \operatorname* { m a x } \{ \hat { \alpha } _ { a } ^ { + } , \alpha _ { a } \} \right) \right)$ and $C \leq 1 . \mathrm { { U s } } .$ ing this we can write $\mathbb { E } _ { X | A = a } \left[ Z \right] = \operatorname* { P r } \left( \operatorname* { m i n } \{ \hat { \alpha } _ { a } ^ { + } , \alpha _ { a } \} \leq \check { C } < \operatorname* { m a x } \{ \hat { \alpha } _ { a } ^ { + } , \alpha _ { a } ^ { - } \} | A = a \right) = | F _ { a } ( \hat { \alpha } _ { a } ^ { + } ) - \check { C } | .$ $F _ { a } ( \alpha _ { a } ) |$ . Since $F _ { a }$ is continuous, and from Theorem $4 . 1 , \hat { \alpha } _ { a } ^ { + } \overset { p } {  } \alpha _ { a }$ , by the continuous mapping theorem, $F _ { a } ( \hat { \alpha } _ { a } ^ { + } ) \stackrel { p } {  } F _ { a } ( \alpha _ { a } )$ . Combining all the pieces, $\hat { \Delta } _ { E O P } ( g _ { \hat { \alpha } _ { \sigma } ^ { + } } ) \stackrel { p } {  } \Delta _ { E O P } ( g _ { \alpha } )$

(2) Bias: We now want to bound the bias of our estimator, $| \Delta _ { E O P } ( g ) - \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ \hat { \Delta } _ { E O P } ( \hat { g } ) \right] | .$

$$
\left| \Delta _ { E O P } ( g ) - \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ \hat { \Delta } _ { E O P } ( \hat { g } ) \right] \right| = \left| \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ \Delta _ { E O P } ( g ) - \hat { \Delta } _ { E O P } ( \hat { g } ) \right] \right|
$$

$$
\begin{array} { r } { \leq \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ \left| \hat { \Delta } _ { E O P } ( \hat { g } ) - \Delta _ { E O P } ( g ) \right| \right] \leq \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ \left| \hat { \rho } _ { 1 } ( \hat { g } ) - \hat { \rho } _ { 0 } ( \hat { g } ) - \rho _ { 1 } ( g ) + \rho _ { 0 } ( g ) \right| \right] } \end{array}
$$

(By using the Reverse Triangle Inequality)

$$
\leq \mathbb { E } _ { D _ { \beta , 1 } , D _ { \alpha } } \left[ \left| \rho _ { 1 } ( g ) - \rho _ { 1 } ( \hat { g } ) \right] \right| + \mathbb { E } _ { D _ { \beta , 0 } , D _ { \alpha } } \left[ \left| \rho _ { 0 } ( g ) - \rho _ { 0 } ( \hat { g } ) \right| \right]
$$

$$
+ \mathbb { E } _ { D _ { \beta , 1 } , D _ { \alpha } } \left[ | \rho _ { 1 } ( \hat { g } ) - \hat { \rho } _ { 1 } ( \hat { g } ) | \right] + \mathbb { E } _ { D _ { \beta , 0 } , D _ { \alpha } } \left[ | \rho _ { 0 } ( \hat { g } ) - \hat { \rho } _ { 0 } ( \hat { g } ) | \right]
$$

(By introducing an intermediate estimator $\rho _ { a } ( \hat { g } )$

and noting that the expectation is over group-specific samples $D _ { \beta . a } )$

Once again, we will only focus on deriving the statistical bias for an arbitrary group $A = a$ . Consider the term $\mathbb { E } _ { D _ { \beta , a } , D _ { \alpha } } \left[ | \rho _ { a } ( \hat { g } ) - \rho _ { a } ( g ) | \right]$ . Since the denominator for both is a constant $\pi _ { 1 | a } , { \mathrm { w e } }$ will focus on showing that the difference in the numerator converges to zero. Reusing our analysis from the previous proof on consistency and using Assumption 2.2, we can write:

$$
\mathbb { E } _ { D _ { \beta , a } , D _ { \alpha } } \left[ | \rho _ { a } ( \hat { g } ) - \rho _ { a } ( g ) | \right] \leq \frac { 1 } { \pi _ { 1 | a } } \mathbb { E } _ { D _ { \alpha } } \left[ | F _ { a } ( \hat { \alpha } _ { a } ^ { + } ) - F _ { a } ( \alpha _ { a } ) | \right] \leq \frac { L _ { a } } { \pi _ { 1 | a } } \mathbb { E } _ { D _ { \alpha } } \left[ | \hat { \alpha } _ { a } ^ { + } - \alpha _ { a } | \right]
$$

From Theorem 4.1, we know that with $1 - \delta$ probability, $| \hat { \alpha } _ { a } ^ { + } - \alpha _ { a } | < \epsilon$ . We also know that $| \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } | \leq 1$ . Using this, we get:

$$
\begin{array} { r l } & { \mathbb { E } _ { D _ { \alpha } } \left[ \left| \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } \right| \right] = \mathbb { E } _ { D _ { \alpha } } \left[ \left| \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } \right| \mathbb { I } \left( \left| \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } \right| < \epsilon _ { a } \right) \right] + \mathbb { E } _ { D _ { \alpha } } \left[ \left| \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } \right| \mathbb { I } \left( \left| \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } \right| \geq \epsilon _ { a } \right) \right] } \\ & { \leq \epsilon _ { a } \cdot \operatorname* { P r } \left( \left| \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } \right| < \epsilon _ { a } \right) + 1 \cdot \operatorname* { P r } \left( \left| \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } \right| \geq \epsilon _ { a } \right) \mathrm { ~ } ( \mathrm { S i n c e ~ } | \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } | \leq 1 ) \leq \epsilon _ { a } + \delta _ { a } . } \end{array}
$$

Using $\begin{array} { r } { \epsilon _ { a } = \frac { | \lambda | } { \tau ^ { 2 } \pi _ { 1 a } ^ { 2 } } \sqrt { \frac { 8 \log ( 2 / \delta _ { a } ) } { N _ { \alpha } } } } \end{array}$ , our final bound becomes:

$$
\mathbb { E } _ { D _ { \beta , a } , D _ { \alpha } } \left[ | \rho _ { a } ( \hat { g } ) - \rho _ { a } ( g ) | \right] \leq \frac { L _ { a } } { \pi _ { 1 | a } } \left( \frac { | \lambda | } { \tau ^ { 2 } \pi _ { 1 a } ^ { 2 } } \sqrt { \frac { 8 \log ( 2 / \delta _ { a } ) } { N _ { \alpha } } } + \delta _ { a } \right)
$$

Next, consider the term $\mathbb { E } _ { D _ { \beta , a } , D _ { \alpha } } \left[ \left| \rho _ { a } ( \hat { g } ) - \hat { \rho } _ { a } ( \hat { g } ) \right| \right]$

$$
\begin{array} { r l } & { \mathcal { L } _ { ( \mathcal { X } _ { n , S } , 0 ] } [ \mathcal { S } _ { n } ; \bar { X } _ { n } ^ { 0 } , \mathcal { X } _ { n } ^ { 0 } ] [ - \mathcal { K } _ { n , S } ; \mathcal { R } _ { n , S } ] \left[ \mathcal { R } _ { n } ^ { 2 } \Lambda ( \mathcal { L } _ { n } ^ { 0 } ; \mathcal { B } _ { n } ^ { \pm } ; \mathcal { T } ) _ { C } \right] } \\ & { \quad - \mathcal { E } _ { ( \mathcal { X } _ { n , S } , 0 ] } \Bigg [ \mathcal { R } _ { n } ^ { 2 } \Lambda ( \mathcal { L } _ { n } ^ { 0 } ; \mathcal { C } ) _ { C } \Bigg ] \Bigg [ \mathcal { R } _ { n } ^ { 2 } \Lambda ( \mathcal { R } _ { n } ^ { 0 } ; \mathcal { R } _ { n } ^ { \pm } ) \Bigg ] } \\ & { \quad + \frac { \mathbb { E } _ { ( \mathcal { X } _ { n , S } , 0 ] } [ \mathcal { X } _ { n } ^ { 0 } ; \mathcal { X } _ { n } ^ { 0 } ] [ \mathcal { X } _ { n } ^ { 0 } ; \mathcal { X } _ { n } ^ { 0 } ] [ - \mathcal { R } _ { n } ^ { 0 } ; \mathcal { X } _ { n } ^ { 0 } ] [ \mathcal { R } _ { n } ^ { 0 } ; \mathcal { X } _ { n } ^ { 0 } ] [ \mathcal { C } ] } { \Lambda _ { C } ^ { 0 } } } \\ & { \quad + \frac { \mathbb { E } _ { ( \mathcal { X } _ { n , S } , 0 ] } [ \mathcal { X } _ { n } ^ { 0 } ; \mathcal { X } _ { n } ^ { 0 } ] [ - \mathcal { R } _ { n } ^ { 0 } ; \mathcal { X } _ { n } ^ { 0 } ] [ \mathcal { R } _ { n } ^ { 0 } \Sigma _ { C } \bar { X } _ { n } ^ { 0 } ] [ \mathcal { X } _ { n } ^ { 0 } ; \mathcal { X } _ { n } ^ { 0 } ] } { \sqrt { \pi } } \Bigg ] \Bigg [ } \\ &  \quad \mathrm  ( B y t a c a l i u e a ~ a l i u e a ~ a l i u e s ~ a l i u \end{array}
$$

Conditioned on $D _ { \alpha } , \hat { \alpha } _ { a } ^ { + }$ is fixed. Therefore, we can use Lemma B.2 to upper bound the term $\begin{array} { r } { \mathbb { E } _ { D _ { \beta , a } , D _ { \alpha } } \left[ \left| \mathbb { E } _ { X | A = a } \left[ \mathbb { I } \left( C \geq \hat { \alpha } _ { a } ^ { + } \right) C \right] - \hat { \mathrm { E } } _ { X | A = a } [ \mathbb { I } \left( C \geq \hat { \alpha } _ { a } ^ { + } \right) C ] \right| \right] } \end{array}$

$$
\begin{array} { r l } & { \mathbb { E } _ { D _ { \beta , a } , D _ { \alpha } } \left[ \left| \mathbb { E } _ { X | A = a } \left[ \mathbb { I } \left( C \geq \hat { \alpha } _ { a } ^ { + } \right) C \right] - \hat { \boldsymbol { \mathrm { E } } } _ { X | A = a } [ \mathbb { I } \left( C \geq \hat { \alpha } _ { a } ^ { + } \right) C ] \right| \right] } \\ & { = \mathbb { E } _ { D _ { \beta , a } } \left[ \left| \mathbb { E } _ { X | A = a } \left[ \mathbb { I } \left( C \geq \hat { \alpha } _ { a } ^ { + } \right) C \right] - \hat { \boldsymbol { \mathrm { E } } } _ { X | A = a } [ \mathbb { I } \left( C \geq \hat { \alpha } _ { a } ^ { + } \right) C ] \right| | D _ { \alpha } \right] } \\ & { = \sqrt { V a r ( \hat { \boldsymbol { \mathrm { E } } } _ { X | A = a } [ \mathbb { I } \left( C \geq \hat { \alpha } _ { a } ^ { + } \right) C ] | D _ { \alpha } ) } = \sqrt { \cfrac { 1 } { N _ { \beta , a } } V a r ( \mathbb { I } \left( C \geq \alpha _ { a } ^ { + } \right) C ) } } \\ & { \leq \cfrac { 1 } { 2 \sqrt { N _ { \beta , a } } } \left( \mathrm { U s i n g ~ P o p o v i c i u ' s ~ I n e q u a l i t y } \right) . } \end{array}
$$

To bound $\begin{array} { r l } & { \mathbb { E } _ { D _ { \beta , a } , D _ { \alpha } } \left[ \left| \frac { 1 } { \pi _ { 1 | a } } - \frac { 1 } { \hat { \pi } _ { 1 | a } ^ { + } } \right| \right] = \mathbb { E } _ { D _ { \alpha } } \left[ \left| \frac { 1 } { \pi _ { 1 | a } } - \frac { 1 } { \hat { \pi } _ { 1 | a } ^ { + } } \right| \right] } \end{array}$ (Since $\hat { \pi } _ { 1 | a } ^ { + }$ is estimated using $D _ { \alpha } ) .$ , let $\mathcal { G } = \{ \hat { \pi } _ { 1 | a } \geq \frac { \pi _ { 1 | a } } { 2 } \}$ denote the event that ${ \hat { \pi } } _ { 1 | a }$ and $\pi _ { 1 | a }$ are close. Note that on $\mathcal { G } , \hat { \pi } _ { 1 | a } ^ { + } = \hat { \pi } _ { 1 | a }$ for sufficiently many samples $( 1 / N _ { \alpha , a } \leq \pi _ { 1 | a } / 2 ) \mathrm {  ; }$

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { B s } } \left[ \frac { 1 } { \tau _ { 1 , 2 } ^ { ( 1 ) } } - \frac { 1 } { \tilde { \lambda } _ { 1 , 1 } } \right] = \Xi _ { D s } \left[ \frac { 1 } { \tau _ { 1 , 2 } ^ { ( 1 ) } } - \frac { 1 } { \tilde { \lambda } _ { 1 , 1 } } \left| \Omega _ { s } \right| + \Xi _ { D s } \left[ \frac { 1 } { \tilde { \lambda } _ { 1 , 1 } } - \frac { 1 } { \tilde { \lambda } _ { 1 , 1 } } \right| \tilde { \lambda } _ { 2 , 1 } \right] } \\ & { = \Xi _ { D s } \left[ \frac { 1 } { \tilde { \lambda } _ { 1 , 1 } } - \Xi _ { 1 , 1 } \right] \cdot \frac { 1 } { \tilde { \lambda } _ { 1 , 2 } ^ { ( 1 ) } } \Bigg ] + \mathrm { E } _ { B o s } \Bigg [ \frac { 1 } { \tilde { \lambda } _ { 1 , 1 } } - \frac { 1 } { \tilde { \lambda } _ { 1 , 1 } ^ { ( 1 ) } } \left| \Delta _ { 1 s } \right| \Bigg ] } \\ & { \mathcal { L } _ { \mathrm { B s } } ^ { - 2 } \frac { 1 } { \tau _ { 1 , 2 } ^ { ( 1 ) } } - \mathcal { R } _ { 1 } \left| \tilde { \lambda } _ { 1 , 1 } - \Xi _ { 1 , 1 } \right| + \mathcal { T } _ { D s } \Bigg [ \frac { 1 } { \tau _ { 1 , 1 } ^ { ( 1 ) } } - \frac { 1 } { \tilde { \lambda } _ { 1 , 1 } ^ { ( 1 ) } } \left| \Omega _ { s } \right| ^ { 2 } \Bigg ] \left( \mathrm { E m } \tilde { \lambda } _ { 1 , 1 } - \Xi _ { 1 , 1 } \right) \cdot \mathcal { D } _ { 1 o l } \mathrm { t a l l } \mathrm { t } ^ { 2 } } \\ & { \mathcal { L } _ { \mathrm { B i } , 1 } ^ { - 1 } \frac { 1 } { \tilde { \lambda } _ { 1 , 1 } ^ { ( 1 ) } } } \\ &  \leq \frac { 1 }  \tilde { \lambda } _  1 , 1  \end{array}
$$

Our upper bound becomes:

$$
\mathbb { E } _ { D _ { \hat { p } , a } , D _ { \alpha } } \left[ \left| \rho _ { a } ( \hat { g } ) - \hat { \rho } _ { a } ( \hat { g } ) \right| \right] \leq \frac { 1 } { 2 \pi _ { 1 | a } \sqrt { N _ { \beta , a } } } + \frac { 1 } { \pi _ { 1 | a } ^ { 2 } \sqrt { N _ { \alpha , a } } } + \left( \frac { 1 } { \pi _ { 1 | a } } + N _ { \alpha , a } \right) \exp \Big ( - \frac { N _ { \alpha , a } \pi _ { 1 | a } ^ { 2 } } { 2 } \Big )
$$

and our overall upper bound on the bias of the estimator takes the form:

$$
\begin{array} { l } { { \displaystyle \left| \Delta _ { E O P } ( g ) - \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ \hat { \Delta } _ { E O P } ( \hat { g } ) \right] \right| } } \\ { { \displaystyle \leq \sum _ { a \in \{ 0 , 1 \} } \left( \frac { L _ { a } } { \pi _ { 1 | a } } \left( \frac { | \lambda | } { \tau ^ { 2 } \pi _ { 1 a } ^ { 2 } } \sqrt { \frac { 8 \log ( 2 / \delta _ { a } ) } { N _ { \alpha } } } + \delta _ { a } \right) + \frac { 1 } { 2 \pi _ { 1 | a } \sqrt { N _ { \beta , a } } } + \frac { 1 } { \pi _ { 1 | a } ^ { 2 } \sqrt { N _ { \alpha , a } } } \right. } } \\ { { \displaystyle + \left. \left( \frac { 1 } { \pi _ { 1 | a } } + N _ { \alpha , a } \right) \exp \left( - \frac { N _ { \alpha , a } \pi _ { 1 | a } ^ { 2 } } { 2 } \right) \right) } } \\ { { \displaystyle = \sum _ { a \in \{ 0 , 1 \} } \mathcal { O } \left( \frac { L _ { a } } { \pi _ { 1 | a } } \left( \frac { | \lambda | } { \tau ^ { 2 } \pi _ { 1 a } ^ { 2 } } \sqrt { \frac { \log ( 2 / \delta _ { a } ) } { N _ { \alpha } } } + \delta _ { a } \right) + \frac { 1 } { 2 \pi _ { 1 | a } \sqrt { N _ { \beta , a } } } + \frac { 1 } { \pi _ { 1 | a } ^ { 2 } \sqrt { N _ { \alpha , a } } } \right) } } \end{array}
$$

Setting δ = min{δ<sub>0</sub>, δ<sub>1</sub>}, N<sub>α</sub> = min{N<sub>α,0</sub>, N<sub>α,1</sub>} and N<sub>β</sub> = min{N<sub>β,0</sub>, N<sub>β,1</sub>}:

$$
\left| \Delta _ { E O P } ( g ) - \mathbb { E } _ { D _ { \beta } , D _ { \alpha } } \left[ \hat { \Delta } _ { E O P } ( \hat { g } ) \right] \right| = \mathcal { O } \left( \frac { | \lambda | \cdot \operatorname* { m a x } \{ L _ { 0 } , L _ { 1 } \} } { \tau ^ { 2 } } \sqrt { \frac { \log ( 1 / \delta ) } { N _ { \alpha } } } + \frac { 1 } { \sqrt { N _ { \beta } } } \right)
$$

(3) Sample Complexity: To bound the sample complexity, we need to bound the following event:

$$
\operatorname* { P r } \left( | \hat { \Delta } ( \hat { g } ) - \Delta ( g ) | > \epsilon \right) \leq \delta .
$$

Let $\Delta ( \hat { g } )$ denote the true EOP difference with respect to the estimated threshold $\hat { \alpha } _ { a } ^ { + }$ . Consider the absolute difference term and apply the reverse triangle inequality $( | | A | - | B | | \leq | A ^ {  } - B | )$

$$
\begin{array} { r l } & { | \Delta ( g ) - \hat { \Delta } ( \hat { g } ) | \leq | \rho _ { 1 } ( g ) - \rho _ { 0 } ( g ) - \hat { \rho } _ { 1 } ( \hat { g } ) + \hat { \rho } _ { 0 } ( \hat { g } ) | } \\ & { \leq | \rho _ { 1 } ( g ) - \rho _ { 1 } ( \hat { g } ) | + | \rho _ { 0 } ( g ) - \rho _ { 0 } ( \hat { g } ) | + | \rho _ { 1 } ( \hat { g } ) - \hat { \rho } _ { 1 } ( \hat { g } ) | + | \rho _ { 0 } ( \hat { g } ) - \hat { \rho } _ { 0 } ( \hat { g } ) | . } \end{array}
$$

By the union bound, we want to ensure the following:

$$
\operatorname* { P r } \Big ( | \Delta ( g ) - \hat { \Delta } ( \hat { g } ) | > \epsilon \Big ) \leq \sum _ { a \in \{ 0 , 1 \} } \operatorname* { P r } \left( | \rho _ { a } ( g ) - \rho _ { a } ( \hat { g } ) | > \epsilon / 4 \right) + \operatorname* { P r } \left( | \rho _ { a } ( \hat { g } ) - \hat { \rho } _ { a } ( \hat { g } ) | > \epsilon / 4 \right) .
$$

Since similar arguments will work for any group $A = a ,$ , we will now study each term separately for an arbitrary group $A = a$ . From our analysis in previous part, we know that $| \rho _ { a } ( g ) - \rho _ { a } ( \hat { g } ) | \leq$ $\frac { L _ { a } } { \pi _ { 1 | a } } | \hat { \alpha } _ { a } ^ { + } - \alpha _ { a } |$ and therefore Pr $\begin{array} { r } { ( | \rho _ { a } ( g ) - \rho _ { a } ( \hat { g } ) | > \epsilon / 4 ) \le \operatorname* { P r } \left( | \hat { \alpha } _ { a } ^ { + } - \alpha _ { a } | > \frac { \epsilon \pi _ { 1 | a } } { 4 L _ { a } } \right) } \end{array}$

Using Theorem 4.1, recall that with $1 - \delta$ probability over the choice of data sample $N _ { \alpha }$ , as long as $\begin{array} { r } { N _ { \alpha } \geq \frac { 1 } { 2 t ^ { 2 } } \log ( \frac { 2 } { \delta } ) \mathrm { f o r } t = \operatorname* { m i n } \Big \{ \frac { \pi _ { 1 a } } { 2 } , \frac { \tau \pi _ { 1 a } ^ { 2 } } { 4 | \lambda | } , \frac { \epsilon \tau ^ { 2 } \pi _ { 1 a } ^ { 2 } } { 4 | \lambda | } \Big \} , \operatorname* { P r } \big ( | \hat { \alpha } _ { a } - \alpha _ { a } | > \epsilon \big ) \leq \delta . } \end{array}$ . Using these bounds with our new ϵ and δ we get:

$$
\begin{array} { r l } & { \operatorname* { P r } \left( \left| \alpha _ { a } - \hat { \alpha } _ { a } ^ { + } \right| > \displaystyle \frac { \epsilon \pi _ { 1 | a } } { 4 L _ { a } } \right) \leq \delta / 4 \implies N _ { \alpha } \geq \displaystyle \frac { 1 } { 2 } \log \left( \frac { 8 } { \delta } \right) \operatorname* { m a x } \left\{ \displaystyle \frac { 2 } { \pi _ { 1 a } } , \displaystyle \frac { 4 | \lambda | } { \tau \pi _ { 1 a } ^ { 2 } } , \displaystyle \frac { 1 6 | \lambda | L _ { a } } { \epsilon \tau ^ { 2 } \pi _ { 1 | a } \pi _ { 1 a } ^ { 2 } } \right\} ^ { 2 } } \\ & { \qquad = \mathcal { O } \left( \displaystyle \frac { \lambda ^ { 2 } L _ { a } ^ { 2 } } { \epsilon ^ { 2 } \tau ^ { 4 } } \log \left( \frac { 1 } { \delta } \right) \right) . } \end{array}
$$

Now we focus on the other event. Just like the bias proof, we will proceed by introducing an intermediate estimator:

$$
\begin{array} { r l } { | \frac { \partial \xi } { \partial t } ( \vec { \xi } ) - \vec { \mu } ( \vec { \xi } ) | } & { = | \frac { \mathbb { E } _ { x \_ { \pm \pm } , \ldots , \ldots , \ldots , \ldots , \ldots , k _ { \xi } } [ ( \vec { \xi } ^ { \top } \cdot \sum _ { i = 1 } ^ { \infty } \vec { \xi } _ { i } ) C ^ { \top } ] } { \nu _ { \pm \downarrow } } - \frac { \mathbb { E } _ { x \_ { \pm \pm } , \ldots , k _ { \xi } } [ C ( \vec { \xi } ) - \vec { \mu } _ { \xi } ^ { \top } ] } { \nu _ { \pm \downarrow } } | } \\ & { = | \frac { \mathbb { E } _ { x \_ { \pm \pm } , \ldots , \ldots , \ldots , k _ { \xi } } [ ( \vec { \xi } ^ { \top } \cdot \sum _ { i = 1 } ^ { \infty } \vec { \xi } _ { i } ) C ^ { \top } ] } { \nu _ { \pm \downarrow } } - \frac { \mathbb { E } _ { x \_ { \pm \pm } , \ldots , k _ { \xi } } [ C ( \vec { \xi } ) - \vec { \mu } _ { \xi } ^ { \top } ] } { \nu _ { \pm \downarrow } } | } \\ & { \quad + \frac { \mathbb { E } _ { x \_ { \pm \pm } , \ldots , k _ { \xi } } [ ( \vec { \xi } ^ { \top } \cdot \sum _ { i = 1 } ^ { \infty } \vec { \xi } _ { i } ) C ^ { \top } ] } { \nu _ { \pm \downarrow } } - \frac { \mathbb { E } _ { x \_ { \pm \pm } , k _ { \xi } } [ C ( \vec { \xi } ^ { \top } ) - \vec { \mu } _ { \xi } ^ { \top } ] } { \nu _ { \pm \downarrow } } | } \\ &  \quad \times \frac { \mathbb { E } _ { x \_ { \pm \pm } , k _ { \xi } } [ C ( \vec { \xi } ) - \vec { \mu } _ { \xi } ^ { \top } ] }  \nu _  \pm \xi \end{array}
$$

Now we can use the union bound:

$$
\begin{array} { r l } & { \operatorname* { P r } \left( | \rho _ { a } \left( \hat { g } \right) - \hat { \rho } _ { a } \left( \hat { g } \right) | > \epsilon / 4 \right) } \\ & { \quad \le \operatorname* { P r } \left( \frac { 1 } { \pi _ { 1 | a } } | \mathbb { E } _ { X | A = a } \left[ \mathbb { I } \left( C \ge \hat { \alpha } _ { a } ^ { + } \right) C \right] - \hat { \mathrm { E } } _ { X | A = a } [ \mathbb { I } \left( C \ge \hat { \alpha } _ { a } ^ { + } \right) C ] | > \epsilon / 8 \right) } \\ & { \quad + \operatorname* { P r } \left( \Big | \frac { 1 } { \pi _ { 1 | a } } - \frac { 1 } { \hat { \pi } _ { 1 | a } ^ { + } } \Big | > \epsilon / 8 \right) } \end{array}
$$

To further simplify $\left| \frac { 1 } { \pi _ { 1 | a } } - \frac { 1 } { \hat { \pi } _ { 1 | a } ^ { + } } \right|$ , we can borrow our analysis from the bias proof. Recall $\mathcal { G } = \{ \hat { \pi } _ { 1 | a } \geq$ $\frac { \pi _ { 1 | a } } { 2 } \biggr \}$ to be the event that ${ \hat { \pi } } _ { 1 | a }$ and $\pi _ { 1 | a }$ are close. Note that on $\mathcal { G } , \hat { \pi } _ { 1 | a } ^ { + } = \hat { \pi } _ { 1 | a }$ for sufficiently many samples $( 1 / N _ { \alpha , a } \leq \pi _ { 1 | a } / 2 )$ . On this good event ${ \mathcal { G } } .$ we derived that $\begin{array} { r } { \left| \frac { 1 } { \pi _ { 1 | a } } - \frac { 1 } { \widehat { \pi } _ { 1 | a } ^ { + } } \right| \leq \frac { 2 } { \pi _ { 1 | a } ^ { 2 } } | \widehat { \pi } _ { 1 | a } - \pi _ { 1 | a } | } \end{array}$

$$
\begin{array} { r l } & { \Gamma ( \gamma _ { 1 } \{ \mathbf { S } _ { 2 } ( \mathbf { S } _ { 2 } ) - \mathbf { S } _ { 1 } \{ \mathbf { S } _ { 2 } \} \} ) \times } \\ & { \quad \le \Gamma ( \frac { 1 } { \gamma _ { 1 } ( \mathbf { S } _ { 2 } ) } ( \gamma _ { 1 } \{ \mathbf { S } _ { 2 } ( \mathbf { S } _ { 2 } ) + \mathbf { S } _ { 2 } \{ \mathbf { S } _ { 1 } ( \mathbf { S } _ { 2 } ) + \mathbf { S } _ { 2 } \{ \{ ( \mathbf { S } _ { 2 } ) + \mathbf { S } _ { 2 } \{ \mathbf { S } _ { 1 } ( \mathbf { S } _ { 2 } ) + \mathbf { S } _ { 2 } \{ \mathbf { S } _ { 1 } ( \mathbf { S } _ { 2 } ) + \mathbf { S } _ { 2 } \} \} } )  +  \\ & { \quad - \Gamma ( \frac { 1 } { \gamma _ { 1 } ( \mathbf { S } _ { 2 } ) } ( \frac { 1 } { \gamma _ { 1 } ( \mathbf { S } _ { 2 } ) } - \mathbf { S } _ { 2 } ) ) } \\ & { \quad - \Gamma ( \frac { 1 } { \gamma _ { 2 } ( \mathbf { S } _ { 2 } ) } - \frac { 1 } { \gamma _ { 1 } ( \mathbf { S } _ { 2 } ) } \sum _ { i = 1 } ^ { N } \mathrm { S } _ { i j } [ ( \gamma _ { 2 } \{ \mathbf { S } _ { 2 } ( \mathbf { S } _ { 2 } ) + \{ \{ \mathbf { S } _ { 2 } \{ S _ { 2 } ( \mathbf { S } _ { 2 } ) + \{ \mathbf { S } _ { 2 } \{ S _ { 1 } ( \mathbf { S } _ { 2 } ) + \mathbf { S } _ { 2 } \{ S _ { 2 } ( \mathbf { S } _ { 2 } ) \} } \}  )  +  \\ &  \quad - \Gamma ( \frac { 1 }  \gamma _  1  \end{array}
$$

(By using Hoeffding’s Inequality on each event).

We can now derive our sample complexities by splitting out $\delta / 4$ budget between all three terms to obtain $\begin{array} { r } { N _ { \beta , a } \geq \frac { 3 2 } { \pi _ { 1 | a } ^ { 2 } \epsilon ^ { 2 } } \log \left( \frac { 2 4 } { \delta } \right) } \end{array}$ and $\begin{array} { r } { N _ { \alpha , a } \geq \operatorname* { m a x } \left\{ \frac { 2 } { \pi _ { 1 | a } ^ { 2 } } \log \left( \frac { 2 4 } { \delta } \right) , \frac { 1 2 8 } { \pi _ { 1 | a } ^ { 4 } \epsilon ^ { 2 } } \log \left( \frac { 2 4 } { \delta } \right) \right\} } \end{array}$

Overall, our sample complexities take the following form:

$$
N _ { \alpha } = \mathcal { O } \left( \frac { \lambda ^ { 2 } \operatorname* { m a x } \{ L _ { 0 } ^ { 2 } , L _ { 1 } ^ { 2 } \} } { \epsilon ^ { 2 } \tau ^ { 4 } } \log \left( \frac { 1 } { \delta } \right) \right) \mathrm { ~ a n d ~ } N _ { \beta } = \mathcal { O } \left\{ \frac { 1 } { \epsilon ^ { 2 } } \log \left( \frac { 1 } { \delta } \right) \right\} .
$$

## E Details for the Tradeoff Estimation algorithm and Experiments

With all the tools in place, we now present the algorithm for estimating the Pareto frontier using the estimators derived using soft labels in the previous section. We will now describe the algorithm formally in Algorithm 1 and discuss each design choice in more detail.

1-2 Assuming access to sufficiently large, disjoint datasets $D _ { \alpha } . D _ { \beta }$ sampled i.i.d. from our target distribution, we obtain estimates for all the distribution-dependent quantities $\hat { \Pi } =$ $\left\{ \hat { \pi } , \hat { \pi } _ { a } , \hat { \pi } _ { y a } , \hat { \pi } _ { y | a } \right\}$ . Using these priors Π<sup>ˆ</sup> and any $\lambda ,$ , we can estimate our thresholds $\hat { \alpha } _ { a } ^ { + }$ (AlphaEst) and then our risk R and unfairness $\Delta \left( { \mathrm { E r r U n f E s t } } \right)$

3-4 If the Bayes optimal classifier (at $\lambda = 0 )$ already satisfies the required fairness $( | \hat { \Delta } _ { 0 } | \leq \gamma )$ then we are done.

5 Depending on sign of $\hat { \Delta } _ { 0 } ^ { \phantom { \dagger } }$ , we can set $\lambda _ { \mathrm { m a x } }$ from Lemma 3.8 to fix our range for the bisection search.

Algorithm 1 Tradeoff Estimation with Soft Labels   
1: Input: Disjoint datasets $D _ { \alpha } , D _ { \beta } ;$ Unfairness Tolerance $\gamma ,$ Precision $\xi ;$ Priors $\hat { \Pi }$   
2: Functions:   
$\cdot \mathrm { A l p h a E s t } ( \lambda , \hat { \Pi } ) \to \hat { \alpha } _ { a } ^ { + }$ (Function to estimate the cost-sensitive thresholds $\alpha _ { a }$ given a $\lambda$ and distribution   
priors);   
- ErrUnfEst $( \lambda ,$ AlphaEst $( \lambda , \hat { \Pi } ) )  R ( g _ { \hat { \alpha } _ { a } ^ { + } } ) , \Delta ( g _ { \hat { \alpha } _ { a } ^ { + } } )$ (Function to estimate the error and unfairness,   
given a threshold $\alpha _ { a }$ computed using the previous function)   
3: $\hat { R } _ { 0 } , \hat { \Delta } _ { 0 } \gets$ ErrUnfEst $( 0 , \hat { \alpha } _ { 0 } )$   
4: $\mathbf { i f } \ | \hat { \Delta } _ { 0 } | \leq \gamma$ then return No-Tradeoff.   
5: Set range: if $\hat { \Delta } _ { 0 } > 0$ then $\lambda \in [ 0 , \frac { 1 } { \operatorname* { i n f } w } ]$ else $\lambda \in [ - \frac { 1 } { \operatorname* { i n f } w } , 0 ]$   
6: points $\langle \mathcal { \ - } \left\{ \right\}$   
7: while $\lambda _ { \operatorname* { m a x } } - \lambda _ { \operatorname* { m i n } } > \xi$ do   
8: $\lambda \gets ( \lambda _ { \operatorname* { m a x } } + \lambda _ { \operatorname* { m i n } } ) / 2$   
9: $\hat { R } , \hat { \Delta } \gets \mathrm { E r r U n f E s t } ( \lambda , \mathrm { A l p h a E s t } ( \lambda , \hat { \Pi } ) )$   
10: Append $( \lambda , \hat { R } , \hat { \Delta } )$ to points   
11: $\mathbf { i f } \ | \hat { \Delta } | \leq \gamma$ then break loop   
12: if $\hat { \Delta } > \gamma$ then $\lambda _ { \operatorname* { m a x } }  \lambda$ else $\lambda _ { \operatorname* { m i n } }  \lambda$   
13: end while   
14: return points

8-12 We compute λ and calculate the updated $\hat { R } , \hat { \Delta }$ . If we immediately achieve our required unfairness $\gamma ,$ we are done. Otherwise, depending on the sign, we update our bound $\lambda _ { \operatorname* { m a x } } / \lambda _ { \operatorname* { m i n } }$ and continue until the gap between $\lambda _ { \operatorname* { m i n } }$ and $\lambda _ { \mathrm { m a x } }$ is under a fixed precision ξ.

Details of Data and Experiments: The synthetic dataset setup in our experiments is derived from Zeng et al. [12], which consists of a binary label $y \in \{ 0 , 1 \}$ and a binary sensitive attribute $a \in \{ 0 , 1 \}$ We sample data from subgroup-specific 100-dimensional Gaussian distributions $X \sim \mathcal { N } ( \mu _ { y a } , \sigma ^ { \tilde { 2 } } I _ { 1 0 0 } )$ where $I _ { 1 0 0 }$ denotes a 100 × 100 identity matrix and all entries of $\mu _ { y a }$ are sampled from $U ( 0 , 1 )$ . We sample 20, 000 training examples and 5000 test examples.

The overall setup of soft labels is borrowed from previous state-of-the-art studies in Bayes error estimation: Ishida et al. [6] and Ushio et al. [14]. CIFAR-10H [49] is a dataset constructed using 50 additional annotations of the CIFAR-10 test dataset. The ground-truth soft label is constructed as the average of the underlying hard labels. Since the original dataset is multi-class (10 classes), the binary labels are constructed by grouping classes into two: land $\mathbf { V S } .$ other animals, animal vs. artifacts, odd vs. even, and so on. Because we are working with unfairness constraints that also include a notion of a sensitive group, we assume one of these partitions is the class and the other is the sensitive group. Note that no dataset of soft labels exists for standard fairness datasets, and hence, we had to retrofit a popular benchmark from the soft-label-based Bayes error estimation literature. The code for setting up the CIFAR-10H experiments is from Ishida et al. [6] <sup>6</sup>. The code and data to reproduce the experiments is provided here.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The main contributions have been well motivated and highlighted in the abstract and the introduction.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The conclusion section discusses limitations and future extensions of our work. Guidelines:

• The answer [N/A] means that the paper has no limitation, while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: All assumptions required are included in the main body and all the proofs are provided in the Appendix.

## Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: A link to anonymized code to reproduce the empirical results has been included in the Appendix.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: A link to anonymized code to reproduce the empirical results has been included in the Appendix.

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: Data preparation scripts and the random seeds used are provided in the anonymized code.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The experiments are performed with bootstrapping, and the error bars are also reported with the results.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [N/A]

Justification: Our experiments are lightweight and can be run on most systems with moderate compute power.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: All points in the code of ethics have been conformed to. We do not believe our research process has caused harm — it does not involve human participation, and we use standard public datasets from the literature. Our research will help society curb discrimination by directly tackling the themes of bias and unfairness. We do not see any possible adverse impact on society due to our research work.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We discuss the benefits of our work for practitioners and the society at large, and do not foresee any negative impact.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: Our paper does not involve any data or models that can cause harm or have a risk of misuse.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We have credited the authors of the datasets and the codebases we used as baselines.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented, and is the documentation provided alongside the assets?

Answer: [N/A]

Justification: The link to the anonymized code implementation is included in the Appendix. Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: No human subjects involved.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: Human subjects not involved.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: LLMs were not used for any important, original or non-standard components. Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.