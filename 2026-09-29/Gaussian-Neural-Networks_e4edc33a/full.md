# Gaussian Neural Networks

Peter Kuhn<sup>1</sup> and Victoria Heusinger-Heß<sup>1</sup>

Fraunhofer Institute for High-Speed Dynamics, Ernst-Mach-Institut, EMI, Germany {peter.kuhn,victoria.heusinger-hess}@emi.fraunhofer.de

Abstract. Gaussian neural networks (GaNNs) are proposed as a novel regularization mechanism for neural networks. From a Bayesian perspective standard regularization techniques can be viewed as imposing priors over weight-space. Assuming priors over activation-space remains a largely unexplored possibility. GaNNs assume such priors. They do this by treating activities from earlier layers like signals with Gaussian noise and predicting the properties of the noise distribution using an additional unsupervised loss. While training, the unsupervised loss acts as a penalty on unexpected activities, allowing greater weight updates in less surprising directions. The paper demonstrates the superiority of Gaussian neural networks over standard neural networks on a variety of classification and regression tasks. We also investigate the ability of GaNNs to quantify uncertainty.

Keywords: Neural Networks · Regularization · Bayesian Deep Learning · Uncertainty Quantification.

## 1 Introduction

It is a surprising fact about standard neural networks that they cannot be surprised. They possess no explicit representation of what does and what does not constitute expected neuronal activation vectors. However, in order to learn effectively, it seems that prediction errors should drive less hypothesis revision where activities are expected versus where they come as a surprise. Some regularization techniques can be thought of as addressing this weakness implicitly, as they bias the network against learning over-complicated mappings, thereby disincentivizing networks to learn from (unsurprising) noise in the data [7]. In a similar vein, neural networks by themselves are not well suited to deal with and quantify uncertainty inherent in the data (aleatoric uncertainty) or in model parameters (epistemic uncertainty). Here various techniques, from noise induction while learning [2,1] to ensemble methods [5], have proved useful.

This paper proposes a novel regularization technique that induces learnable priors in the activation space of a neural network. This is done by assuming that the activities of hidden layers themselves can be treated as noisy or uncertain.

By amending the standard supervised learning objective by an additional unsupervised learning objective of the probability distribution over hidden node activities a model develops an intrinsic bias towards ‘normal’ activity regimes. This results in a novel architecture, the Gaussian neural network (GaNN), that is intrinsically better regularized. Two ways to model noise of hidden units are proposed, resulting in low to medium computational overhead over standard architectures. Both architectures achieve better predictive performance across a variety of classification and regression tasks as compared to standard neural architectures trained with standard regularizers. By sampling from the learned activity-priors, the method can also be used to quantify predictive uncertainty, though the results here are mixed. GaNNs can be implemented with very low computational and memory costs. Depending on the details, they have only two additional parameters per neuron. GaNNs alter some of the fundamental wiring of neural networks and can in principle be combined with any kind of neural wiring. In our experiments we intentionally restrict our attention to simple dense networks to validate the core architecture.

## 2 Related Work

## 2.1 Bayesian Approaches to Neural Networks

Early work on the deeper integration of Bayesian methods and neural networks took the form of Bayesian neural networks [7]. These work by placing Gaussian prior distributions on the weights, tripling the number of parameters. The modeled uncertainty is the uncertainty of parameters or epistemic uncertainty. These networks can be trained using approximate Bayesian methods, chief among them variational inference. Bayesian neural networks inherently generate predictive distributions and can thus quantify the epistemic uncertainty of predictions, and are less prone to overfitting. However, these advantages are accompanied by a significant increase in necessary computing power. Also, the tripling of the parameter count can be debilitating in training very large models.

Dropout ofers a way of approximating the functioning of Bayesian neural networks in standard architectures without increasing the number of model parameters. Dropout consists in the stochastic deactivation of nodes while learning. Intuitively, the induction of internal noise while training forces the network to learn a representation that is stabilized against that noise. This can be shown to approximate the functioning of Bayesian neural networks trained with variational inference [2] and can be used to quantify predictive uncertainty using Monte-Carlo dropout, i.e. leaving dropout active while inference where the distribution over predictions corresponds to the predicted distribution [1]. More recent work has emphasized diferentiating between diferent kinds of uncertainty by modelling deeper properties of the target distribution instead of attempting deeper revisions of the model structure [19,20]. Such approaches are in principle compatible rather than in conflict with the approach proposed here.

## 2.2 The Bayesian Perspective on Regularization

If a learning algorithm is too flexible the learned mappings will generalize badly to unseen data. This simplicity bias has its parallel in ordinary life in Occam’s razor. In neural learning, it typically shows up as regularization terms that are added to the loss function. Here, weight decay [15] has been established as a strong default. The base loss $\mathcal { L } _ { B }$ , depending on the weights θ is amended by the L2-norm of the weights, connected by a hyper-parameter λ:

$$
\mathcal { L } ( \theta ) = \mathcal { L } _ { B } ( \theta ) + \frac { \lambda } { 2 } \left. \theta \right. _ { 2 } ^ { 2 }\tag{1}
$$

From the Bayesian perspective, weight decay can be understood as an isotropic Gaussian prior on weights such that lower weights have a higher prior probability [7,10].

## 2.3 Gaussian Variance Heads

An approach to uncertainty parallel to Bayesian neural networks and their approximate solutions (and indeed one that can be integrated with them [6]) is to perform maximum-likelihood inference in the final model layer to model the noise inherent in the data, resulting in variance heads [3]. This approach is standardly used to predict aleatoric uncertainty.

Variance heads predicting $\sigma ^ { 2 } ( \pmb { x } _ { i } )$ are outputs of a model corresponding to data-point $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ . It is assumed that the data points $d _ { i }$ scatter around the model output $y ( \pmb { x } _ { i } )$ as a Gaussian distribution parameterized by the weights θ:

$$
P ( d _ { i } | \pmb { x } _ { i } ; \theta ) = \frac { 1 } { \sqrt { 2 \pi \sigma ^ { 2 } ( \pmb { x } _ { i } ) } } e ^ { \frac { - ( d _ { i } - y ( \pmb { x } _ { i } ) ) ^ { 2 } } { 2 \sigma ^ { 2 } ( \pmb { x } _ { i } ) } }\tag{2}
$$

In a maximum-likelihood approach we would adjust network weights in order to maximize the probability of the data given the model, i.e. $P ( d _ { i } | \mathbf { x } _ { i } ; \theta )$ . Because a function has its maxima where the logarithm of the function has its maxima, we can alternatively maximize the natural logarithm. Switching signs and taking the sample-mean we arrive at the Gaussian loss function (sometimes called Gaussian negative log-likelihood or Gaussian NLL):

$$
\mathcal { L } _ { G } = \sum _ { i } \frac { 1 } { 2 } \left( \frac { ( d _ { i } - y ( \pmb { x } _ { i } ) ) ^ { 2 } } { \sigma ^ { 2 } ( \pmb { x } _ { i } ) } + \ln \left( \sigma ^ { 2 } ( \pmb { x } _ { i } ) \right) \right)\tag{3}
$$

Note that this is a more general form of the well known mean squared error and becomes equivalent to it for a uniform variance of 1. Minimizing the full Gaussian loss a model will learn to predict both the mean $y ( \pmb { x } _ { i } )$ as well as the variance $\sigma ^ { 2 } ( \pmb { x } _ { i } )$ of the data-points $d _ { i }$

Note that the common assumption [6] that this approach quantifies solely aleatoric uncertainty can be questioned. The total uncertainty of a prediction can be conceptualized as its expected error [4]. Minimizing the expected error using maximum-likelihood inference, assuming it takes the form of a Gaussian, will be precisely equivalent to the described process of training a variance head. Learning the expected error of a model is an established method for measuring total uncertainty [5,9]. The only diference to a variance head is that it is typically integrated within one model rather than training a second higher-order model. The variance is trained based on the divergence of model output and prediction across trials and this divergence will necessarily result from both epistemic and aleatoric components.

## 3 Gaussian Neural Networks

## 3.1 Inner Gaussian Loss

We propose a new kind of neuronal architecture, the GaNN, that can be efectively trained using back-propagation and that has the promise to be intrinsically well-regularized with minimal computational overhead. GaNNs are regularized by isotropic Gaussian priors over activation-space. GaNNs consist of Gaussian hidden layers. Gaussian hidden layers operate under the assumption that the activity of the network can itself be treated like noisy signals. On a very high level this formalizes the idea that one should track the uncertainty of one’s beliefs or representations. Going deeper, neuronal activities represent real-world features (even if determining what features exactly are represented can be very hard). The assumed internal noise then reflects the uncertainty about whether the neuronal activity accurately captures the represented feature. The GaNN will learn priors over activities, i.e. activity-priors, that can also be seen as priors regarding a represented feature. The violation of this prior will come with a penalty on the loss, entailing an imperative for the network to perform weight updates that conform to their learned prior probability.

Similar to Bayesian neural networks that represent weight and bias priors, pure Gaussian hidden layers possess Gaussian activity priors for all their neurons. Every layer l consists of three vectors representing a mean-prior $\mu ^ { ( l ) }$ and a variance-prior $\pmb { \sigma } ^ { ( l ) ^ { 2 } }$ , in addition to an activity $\mathbf { \Omega } _  \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathbf { \Omega } \mathrm { ~ \Omega ~ } \mathrm { ~ \Omega ~ } \mathrm { ~ \Omega ~ } \mathrm { ~ \Omega ~ } \mathrm { ~ \Omega ~ } \mathrm { ~ \Omega ~ } \mathrm { ~ \Omega ~ } \mathrm { ~ \Omega ~ } \mathrm { ~ \Omega ~ } \mathrm { ~ \Omega ~ } \mathrm { ~ \Omega ~ } \mathrm { ~ \Omega ~ } \mathrm { ~ \Omega ~ } \mathrm { ~ \Omega ~ }$ . We here adopt the convention of using bracketed superscript indices to refer to either singular neurons or layers, which will be clear from context. The full state of a layer can be represented as a vector $\pmb { \mathscr { s } } ^ { ( l ) } = \left( \pmb { \mu } ^ { ( l ) } , \pmb { \sigma } ^ { ( l ) ^ { 2 } } , \pmb { a } ^ { ( l ) } \right) ^ { T }$ . To make the whole construction mathematically traceable we assume the noise at a layer to be independent of the noise at previous layers, resulting in the following likelihood for an N-layer model (the output layer is non-Gaussian), parameterized by the weights θ:

$$
P ( d _ { i } | \pmb { x } _ { i } ; \theta ) = P ( d _ { i } | \pmb { s } _ { i } ^ { ( N - 1 ) } ; \theta ) P ( \pmb { s } _ { i } ^ { ( N - 1 ) } | \pmb { s } _ { i } ^ { ( N - 2 ) } ; \theta ) \cdot \cdot \cdot P ( \pmb { s } _ { i } ^ { ( 1 ) } | \pmb { x } _ { i } ; \theta )\tag{4}
$$

Where $\mathbf { \mathbf { \mathbf { \mathbf { s } } } } _ { i } ^ { ( k - 1 ) }$ is the state corresponding to input sample $\mathbf { x } _ { i } .$ . We assume the probability distribution for the activities for neuron k to be Gaussian parameterized by that neuron’s activity-prior. Just as in the case of variance heads, the variance-priors will be a learnable function of the previous layer’s state, thus written as $\pmb { \sigma } ^ { ( k ) ^ { 2 } } \left( \pmb { s } _ { i } ^ { ( k - 1 ) } \right)$ . Thus, the neuron-wise likelihood will be Gaussian around the mean-prior with the respective variance:

$$
P ( s _ { i } ^ { ( k ) } | s _ { i } ^ { ( k - 1 ) } ; \theta ) = \frac { 1 } { \sqrt { 2 \pi \sigma ^ { ( k ) ^ { 2 } } \left( s _ { i } ^ { ( k - 1 ) } \right) } } e ^ { \frac { - \left( \mu ^ { ( k ) } - a \left( s _ { i } ^ { ( k - 1 ) } \right) \right) ^ { 2 } } { 2 \sigma ^ { 2 } \left( s _ { i } ^ { ( k - 1 ) } \right) } }\tag{5}
$$

We will assume that the input and output layer have no activity-priors and we will ignore them from now on. Taking the logarithm of Eq. 4, as the probabilities for activities in layers and neurons within layers are independent, with index k looping over all neurons in all layers except the input- and output-layer, we get:

$$
\ln { ( P ( d _ { i } | \pmb { x } _ { i } ; \theta ) ) } = \ln { P ( d _ { i } | \pmb { s } _ { i } ^ { ( N - 1 ) } ; \theta ) } + \sum _ { k } \ln { \left( P ( s _ { i } ^ { ( k ) } | \pmb { s } _ { i } ^ { ( k - 1 ) } ; \theta ) \right) }\tag{6}
$$

Focusing on the right term, just as for the training of variance-heads we arrive at a loss function for training activity priors:

$$
\mathcal { L } _ { I } = \sum _ { i , k } \frac { 1 } { 2 } \left( \frac { \left( \mu _ { i } ^ { ( k ) } - a ^ { ( k ) } \left( s _ { i } ^ { ( k - 1 ) } \right) \right) ^ { 2 } } { \sigma ^ { ( k ) ^ { 2 } } \left( s _ { i } ^ { ( k - 1 ) } \right) } + \ln \left( \sigma ^ { ( k ) ^ { 2 } } \left( s _ { i } ^ { ( k - 1 ) } \right) \right) \right)\tag{7}
$$

Learning Gaussian activity priors is a form of unsupervised learning as the Gaussian loss in Eq. 7 is independent of any data-points $d _ { i }$ . We will refer to it as the inner loss of the network. In order to facilitate real learning, the Gaussian loss needs to be amended with a data related base loss to get a semi-supervised total loss. Here, it is tempting to rely on the sum of unsupervised Gaussian loss and some base loss, however, this turns out to be inadvisable. Depending on the dataset, the Gaussian loss may be on a diferent order of magnitude compared to the base loss, leading the network to focus almost solely on learning activity priors. Fixed weights, due to diferences in the evolution of the losses, are similarly inadvisable. Standard solutions for these kinds of multi-goal learning tasks involve additional learnable loss-weights [21]. However, we note that it is easier to define a no-gradient operator $\Theta$ such that, for any g with $\theta \mapsto g ( \theta )$

$$
\nabla _ { \theta } \Theta { \bigl ( } g ( \theta ) { \bigr ) } = 0\tag{8}
$$

This is implemented in PyTorch natively as the no\_grad operation. We can then define the loss of our network, using an arbitrary base loss $\mathcal { L } _ { B }$ , so that the relative contribution of the two loss functions to the total loss will only depend on a new hyper-parameter α:

$$
\mathcal { L } = \mathcal { L } _ { B } + \alpha \Theta \left( \left| \frac { \mathcal { L } _ { B } } { \mathcal { L } _ { I } } \right| \right) \mathcal { L } _ { I }\tag{9}
$$

α specifies how much emphasis will be put on learning the target objective vs. learning activity-priors. As in Eq. 1, α also determines the strength of the regularization. As the inner loss can be negative, this may result in the counterintuitive case of the total loss being zero across training. Considering the impact of Θ, it should be clear that this will not negatively impact learning because there will still be non-zero derivatives owing to those parts of the equation outside the scope of Θ.

The impact of the unsupervised loss is similar to the neuronal mechanism proposed to underlie human attention (see [8]). The function of the inner loss can be conceptualized as a penalty on unexpected activities. The added network nodes learn what regime of activities is to be expected. The inner loss then quantifies and penalizes a gradient evolution outside the learned expected regime. In this way, expectations are integrated into the functioning of the network as larger steps are taken when the system is within an expected regime, while more cautious gradient updates are in order outside of it. As noted in the introduction, it should be surprising that standard neural networks do not learn any explicit representation of what constitutes normal activity. This would also result in a regularization efect as the activity priors disincentivize learning over-complicated mappings, relative to the mappings that have already been learned.

If these considerations are correct, we would expect a model that minimizes an inner loss to perform worse on the validation set in the beginning of training, as the model first has to learn reasonable activity-priors alongside minimizing the base loss. Once this initial learning was performed, the regularization efect should kick in and the network should converge more optimally as compared to a model merely learning the base-objective in a supervised fashion. We would not expect the inner loss to reduce indefinitely, as the learning of the baseobjective will also shift the expected activities around, constantly shifting what the network is trying to learn. So the inner loss should reach a relatively stable plateau at some point in training or even increase where this can be canceled out by a corresponding minimization of the base loss.

## 3.2 Dense and Sparse Gaussian Layers

So far we have introduced an additional inner loss for learning Gaussian activitypriors. It would be possible to let the additional parameters be independent of the model input, similar to a bias in a normal network. However, if we want the network to learn more complex activity priors, we should make the variance terms functions of other activities. We will consider two such architectures, one sparse, resulting in minimal computational overhead, and one dense, which is able to learn more complex patterns but results in a medium to large computational overhead. (See Fig. 1 for an overview.)

Variances are by definition positive and the inner loss becomes undefined otherwise. This can be ensured by using the right kind of activation function. Here, we will employ a softplus activation function $\zeta$ to ensure that:

$$
\zeta ( x ) = \log ( 1 + e ^ { x } )\tag{10}
$$

![](images/8f513cdaa00b87521a6dd84f0c9bae978f55e032fa6e12ba254f59ef6ea663b4.jpg)  
Fig. 1. Two models with two sparse (a) and dense (b) Gaussian hidden layers. Every box is a Gaussian neuron, containing an activity and an activity prior. In the sparse Gaussian layers, the variance-priors only depend on the associated activity. In the dense Gaussian layers, the variance-priors depend on all preceding nodes. In both architectures, the activities are independent of variances. The mean-priors only enter as constituents of the loss and are irrelevant in inference.

Furthermore, both the sparse and the dense model learn more stably using a hard-coded lower bound v for the variance-prior as otherwise a variance approaching zero would entail an inner loss approaching infinity. Thus for the sparse model the variance-prior at layer l will be (omitting dependencies on the input for readability):

$$
\pmb { \sigma } _ { s p a r s e } ^ { ( l ) ^ { 2 } } = \zeta \left( \pmb { a } ^ { ( l ) } \odot \pmb { w } ^ { ( l ) } + \pmb { b } ^ { ( l ) } \right) + v\tag{11}
$$

$\mathbf { \Delta } _ { \pmb { w } } ^ { ( l ) }$ and $\mathbf { \delta } _ { b } ( l )$ are weights and biases, of course not to be confused with the regular weights and biases mapping activities in one layer to activities in the next. ⊙ represents the Hadamard product.

The dense noise model on the other hand works just as a normal dense network layer. Note however, that the activity in a layer does not depend on the variance-priors of the previous layer in the same way. Forward-propagation for activities is defined in the usual way.

$$
\pmb { \sigma } _ { d e n s e } ^ { ( l ) ^ { 2 } } = \zeta \left( W ^ { ( l ) } \pmb { a } ^ { ( l - 1 ) } + K ^ { ( l ) } \pmb { \sigma } ^ { ( l - 1 ) } + \pmb { b } ^ { ( l ) } \right) + v\tag{12}
$$

$W ^ { ( l ) }$ and $K ^ { ( l ) }$ are again additional weight matrices.

The dense model adds considerable computational complexity and memory requirements, but can also learn more complex activity priors. The number of parameters grows quadratically with the number of Gaussian nodes. On the other hand, in case of the sparse architecture, the number of additional parameters when compared to a normal neural network is just three times the number of nodes, adding a mean-prior $\mu ^ { ( k ) }$ , one connection weight as part of $\mathbf { \Delta } _ { \pmb { w } } ^ { ( l ) }$ and one bias term in $\Breve { \mathbf { \theta } } _ { b } ( l )$ . The computational and memory overhead will thus shrink the larger the model gets, becoming vanishingly small for very large models. The computational overhead of the sparse architecture is considerably lower than for ensemble techniques or Bayesian neural networks.

All added complexity of a GaNN as opposed to a standard neural network are only relevant for training and uncertainty quantification. Once training is done, all non-standard parameters can be dropped and the remaining units can be used for feed-forward inference.

We would predict that, overall, a GaNN equipped with a dense noise model will generate more powerful efects as opposed to the sparse noise model for it can encode more complex expectations about what constitutes ordinary activities, and what constitutes a surprise and should be penalized. On the other hand we may also expect that a too powerful noise model might focus on minimizing the inner loss indefinitely delaying progress on the main learning objective, causing slow convergence or even no convergence within the relevant time frames.

## 3.3 Uncertainty Quantification

Noisy Sampling. A trained GaNN can perform uncertainty quantification by producing a distribution of predictions. This is similar, though not equivalent, to the process of drawing predictive samples in Monte Carlo dropout. As a GaNN treats its own activities like a noisy data-source, we can straight away sample from the learned noise distributions and add those to the relevant activities in inference. So, ${ s } _ { \gamma , j } ^ { ( l ) } \left( \pmb { x } _ { i } \right)$ is the jth noisy sample of the lth state vector for the ith input sample. As the state vectors at layer l only depend on the state vector at layer l − 1 we say that:

$$
\pmb { a } _ { \gamma , j } ^ { ( l ) } = \pmb { a } _ { j } ^ { ( l ) } \left( \pmb { a } _ { \gamma , j } ^ { ( l - 1 ) } \right) + \gamma \varepsilon \left( 0 , \pmb { \sigma } ^ { ( l ) ^ { 2 } } \left( \pmb { a } _ { \gamma , j } ^ { ( l - 1 ) } \right) \right)\tag{13}
$$

$\varepsilon ( \mu , \sigma ^ { 2 } )$ is a Gaussian noise term with mean $\mu$ and variance $\sigma ^ { 2 }$ . For the input layer there will be no noisy sampling of course. $\gamma$ is a hyper-parameter that controls the strength of the noise. While the resulting distribution will naturally reflect predictive uncertainty, there is no reason to assume it should be well calibrated by default. We note that, in principle, γ can be learned from data, though we do not attempt this here.

One might expect, generalizing from the example of variance heads, that the process will primarily help to quantify aleatoric uncertainty. However, as argued above, whether variance heads really solely quantify aleatoric uncertainty can be questioned. Another way to see this is to note that, if a network predicts high activity variance for some input, the relevant input will be unusual. Thus, if a data-point is distant from the training data, and thus the connected predictions are epistemically uncertain, this should lead to high predicted variance.

Performance Metrics. Uncertainty quantification sufers from the fact that the ground truth, except in the case of simulated data, is unavailable. We thus have to specify additional performance metrics to measure the quality of uncertainty quantification. In case of regression, we can simply calculate the mean and variance of the predictive distribution generated by drawing samples from the process specified by $\operatorname { E q . }$ 13. We can then rely on the Gaussian negative loglikelihood $\mathrm { o r }$ Gaussian NLL, already specified in Eq. 3, measuring the error on the test set with respect to the predicted mean and variance.

In case of classification we will start out by calculating the predicted probabilities $p _ { k , i }$ for all classes k and all samples $i \in [ 0 ,$ , n] by taking the mean across the softmax output across predictive samples j. We will define the predicted class $\ddot { d } _ { i }$ as that with the maximum probability. These we can compare to the true classes $d _ { i }$ in two ways that will tell us about how well the model captures uncertainty.

Just as in the case of the regression task, we can use the negative loglikelihood of the results, given the predicted probabilities as a guide to how well the ground truth (the true class $y _ { i } )$ was captured:

$$
{ \mathrm { N L L } } = - { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \log p _ { y _ { i } , i }\tag{14}
$$

Also, we may want to know how well we can detect whether our model is going to make an error from how uncertain it is. The relevant uncertainty per sample i is equal to the entropy:

$$
H _ { i } = - \sum _ { k } p _ { k , i } \log p _ { k , i }\tag{15}
$$

Then to assess how good a guide the uncertainty or entropy is to the probability of error we evaluate the probability that a randomly chosen error has a higher uncertainty score than a randomly chosen correct example (with ties counted as half). This results in the AUROC score (the ‘Area Under the Receiver Operating Characteristic curve’) for error detection:

$$
{ \mathrm { A U R O C } } = { \frac { 1 } { | E | \left| C \right| } } \sum _ { i \in E } \sum _ { j \in C } { \Big ( } \mathbf { 1 } ( H _ { i } > H _ { j } ) + { \frac { 1 } { 2 } } \mathbf { 1 } ( H _ { i } = H _ { j } ) { \Big ) }\tag{16}
$$

1(P) is the fraction of samples for which condition $P$ holds. C is the set of correct samples, E is the set of errors, thus $| E | + | C | = n$

Diferentiating between diferent kinds of uncertainty is not the subject of this paper and the proposed metrics allow no such disentanglement.

## 4 Experiments

## 4.1 Model Implementation

All models were implemented in $\mathrm { P y }$ Torch. For the dense noise model, a layer can be implemented using PyTorch’s standard Linear layer with a doubled number of units. After a forward pass through one layer the units are then split in half with one half representing activities, the other representing variances. Both are then fed into diferent loss functions, ReLU in the case of the activities, softplus in the case of variances. The sparse architecture is implemented with the help of a custom layer object. All Gaussian layers return their internal state vector s, a list of which is added to the network’s output for the computation of the supervised loss. The classical neural networks used for comparison are identical to the GaNNs, except for the additional structure. All models possess three hidden layers containing 1024, 512 and 256 units respectively; if dropout is used, there is one dropout layer per hidden layer. For Gaussian networks the dropout layers only attach to the activity nodes of the next layer, not to the variance nodes.

## 4.2 Training Setup

All experiments were performed five times with diferent random seeds and accumulated as a mean. Training was performed using stochastic gradient descent. A very simple and uniform learning rate schedule was used. For regression tasks, the initial learning rate was chosen to be 0.1, for classification tasks 0.01. After 30 epochs it was divided by 10. In some cases, the training was not stable (for both the standard and the Gaussian architecture) and gradient normalization to a norm of 10 was performed to produce a training setup that could be applied universally to all models. The base loss for regression is MSE, the base loss for classification is cross-entropy. As with any regularization term added to the base loss, the strength of regularization, here encoded by α, is a sensitive hyper-parameter. While $\alpha = 1$ can be a good default on many datasets, we find that this can place too strong constraints on learning. In case of simple classification tasks we thus used $\alpha = 0 . 1$ , in case of CIFAR100 and Tiny ImageNet we chose $\alpha = 0 . 0 1$ . The variance lower bound we chose to be $v = 0 . 5$ for all experiments. Where weight decay was used, we used a factor of $\lambda = 0 . 0 0 0 1$ . The chosen dropout rate was 0.3 for classification and 0.1 for regression tasks. The numbers of epochs used for specific datasets are listed in Table 1. For test set evaluation we used the weight configuration where the model achieved maximal validation performance.

## 4.3 Datasets

We use ten datasets, five involving a regression, five involving a classification task. An overview is shown in Table 1. Datasets are both visual and tabular to demonstrate the independence of performance of any specific data type. Regression datasets include the airfoil self-noise dataset created by NASA for the acoustic properties of airfoil blade sections [11], the yacht dataset where the hydrodynamic properties of yachts are predicted from dimensions and velocity [12], the UTK face dataset for the prediction of ages from portrait pictures consisting of 64x64 color images [13], the wine quality dataset [14] where features are chemical properties of wine and the target is a taste score, and finally the million songs dataset where the year of publication is to be predicted from other features of the song. On the classification side, the CIFAR100 is composed of 32x32 color images that belong to 100 diferent classes [16] and Tiny ImageNet is composed of 64x64 color images that belong to 200 classes, the MNIST datasets are 28x28 greyscale images of handwritten digits [18] and ten types of fashion articles [17].

Table 1. Summary of dataset properties and number of epochs employed.
<table><tr><td>Name</td><td>Domain</td><td>N</td><td>Features Epochs</td><td></td></tr><tr><td colspan="5">Regression</td></tr><tr><td>Airfoil</td><td>tabular</td><td>1503</td><td>5</td><td>3000</td></tr><tr><td>Yacht</td><td>tabular</td><td>308</td><td>6</td><td>5000</td></tr><tr><td>UTKFace</td><td>vision</td><td>23708</td><td>12288</td><td>120</td></tr><tr><td>Wine Regression</td><td>tabular</td><td>6497 515345</td><td>11 90</td><td>120 120</td></tr><tr><td colspan="5">YearPredictionMSD tabular</td></tr><tr><td>Classification CIFAR-100</td><td>vision</td><td>60000</td><td>3072</td><td>120</td></tr><tr><td>Digits-MNIST</td><td>vision</td><td>1797</td><td>64</td><td>120</td></tr><tr><td>Fashion-MNIST</td><td>vision</td><td>70000</td><td>784</td><td>120</td></tr><tr><td>Wine Classification</td><td>tabular</td><td>6497</td><td>11</td><td>120</td></tr><tr><td>Tiny ImageNet</td><td>vision</td><td>100000</td><td>12288</td><td>120</td></tr></table>

The MNIST datasets, Tiny ImageNet and CIFAR100 come with a predefined train-test-validation split. All other datasets were split into training and validation data (with diferent random seeds for diferent trials, see below) using an 60/20/20-split.

## 5 Results

## 5.1 Overall Performance

Quantitative predictive performance results are summarized in Table 2. We observe first of all that GaNNs outperform the basic ANN consistently. The sparse architecture outperforms the ANN on nine out of ten datasets with one tie. The dense architecture does so in seven with one tie. Merely looking at overall performance across all experiments, this makes the performance of GaNNs more consistent than either weight decay or dropout. We also observe that the evolution of the loss follows our predicted trend: The networks first learn slower than rival networks, minimizing the inner loss in addition to the base loss. At some point the inner loss plateaus and the model focuses on minimizing the base loss. The inclusion of the learned knowledge about what constitutes normal activity enables higher peak performances as compared to non-Gaussian architectures. This trend is especially clear in the Airfoil dataset, visualized in Fig. 3.

We observe that the dense noise model is not necessarily superior to the sparse one. In fact, the sparse model more often outperforms the dense variant than vice versa. Also, we observe that the dense model sometimes converges very late in training, for instance in case of the Airfoil and Yacht datasets. Both observations can be explained by an overpowerful noise model: It turns out to primarily minimize the inner loss for long stretches of training (see Fig. 3). When convergence is achieved the model typically achieves better performances than any of the comparison models. We also observe that sparse GaNNs can sometimes be combined with dropout in a beneficial way, underlining that these are not strictly rival techniques.

Table 2. Test performance (mean for best model $\pm \ \mathrm { s t d } )$ . Top: Regression (MAE). Bottom: Classification (Accuracy). Note that for regression datasets, the ANN dropout columns use the variance-head variants.
<table><tr><td>Dataset</td><td colspan="3">GaNN-dense GaNN-sparse GaNN-sparse + D</td><td>ANN</td><td> $\mathbf { A N N } + \mathbf { D }$ </td><td>ANN + WD</td><td> $\mathbf { A N N } + \mathbf { W D } + \mathbf { D }$ </td></tr><tr><td colspan="9">Regression (MAE)</td></tr><tr><td>Airfoil</td><td> $2 . 7 0 6 \pm 0 . 0 9 9$ </td><td> $2 . 3 3 5 \pm 0 . 0 7 5$ </td><td> $2 . 8 5 3 \pm 0 . 0 7 5$ </td><td> $2 . 4 9 4 \pm 0 . 0 9 0$ </td><td> $1 1 . 5 1 6 \pm 0 . 4 1 6$ </td><td> $2 . 4 9 4 \pm 0 . 0 9 0$ </td><td> $1 1 . 5 1 5 \pm 0 . 4 1 6$ </td></tr><tr><td>Yacht</td><td> $2 . 2 0 8 \pm 0 . 1 5 4$ </td><td> $0 . 8 0 4 \pm 0 . 1 1 9$ </td><td> $1 . 1 7 3 \pm 0 . 1 0 7$ </td><td> $0 . 8 7 9 \pm 0 . 1 1 9$ </td><td> $5 . 5 2 1 \pm 0 . 3 8 6$ </td><td> $0 . 8 8 0 \pm 0 . 1 2 0$ </td><td> $5 . 5 2 1 \pm 0 . 3 8 6$ </td></tr><tr><td>UTKFace</td><td> $1 0 . 1 6 4 \pm 0 . 0 9 4$ </td><td> $1 0 . 1 3 1 \pm 0 . 1 2 4$ </td><td> $1 0 . 2 2 7 \pm 0 . 1 1 6$ </td><td> $1 0 . 1 8 3 \pm 0 . 0 7 8$ </td><td>16.719 ± 0.168</td><td> $1 0 . 1 8 4 \pm 0 . 0 7 8$ </td><td> $1 6 . 7 1 9 \pm 0 . 1 6 8$ </td></tr><tr><td>Wine Regression</td><td> $0 . 5 7 3 \pm 0 . 0 0 8$ </td><td> $0 . 5 8 9 \pm 0 . 0 0 8$ </td><td> $0 . 5 9 6 \pm 0 . 0 0 9$ </td><td> $0 . 6 1 3 \pm 0 . 0 0 5$ </td><td> $0 . 9 2 9 \pm 0 . 0 2 2$ </td><td> $0 . 6 1 3 \pm 0 . 0 0 5$ </td><td> $0 . 9 2 9 \pm 0 . 0 2 2$ </td></tr><tr><td>YearPredictionMSD</td><td> $6 . 1 7 3 \pm 0 . 0 2 5$ </td><td> $6 . 1 5 9 \pm 0 . 0 2 5$ </td><td> $6 . 8 4 1 \pm 0 . 2 1 7$ </td><td> $7 . 0 2 9 \pm 0 . 0 6 2$ </td><td> $3 3 . 7 2 7 \pm 1 . 1 6 1$ </td><td> $7 . 0 2 4 \pm 0 . 0 6 1$ </td><td> $3 3 . 0 7 5 \pm 1 . 1 5 7$ </td></tr><tr><td colspan="8">Classification (Accuracy)</td></tr><tr><td>CIFAR100</td><td> $0 . 2 7 6 \dot { \pm } 0 . 0 0 3$ </td><td> $0 . 2 7 8 \pm 0 . 0 0 3$ </td><td> $0 . 2 6 9 \pm 0 . 0 0 3$ </td><td> $0 . 2 7 6 \pm 0 . 0 0 4$ </td><td> $0 . 2 6 8 \pm 0 . 0 0 1$ </td><td> $0 . 2 7 8 \pm 0 . 0 0 2$ </td><td> $0 . 2 6 7 \pm 0 . 0 0 1$ </td></tr><tr><td>MNIST-digits</td><td> $0 . 9 8 1 \pm 0 . 0 0 1$ </td><td> $0 . 9 8 0 \pm 0 . 0 0 1$ </td><td> $0 . 9 8 4 \pm 0 . 0 0 1$ </td><td> $0 . 9 8 0 \pm 0 . 0 0 1$ </td><td> $0 . 9 8 4 \pm 0 . 0 0 1$ </td><td> $0 . 9 8 0 \pm 0 . 0 0 1$ </td><td> $0 . 9 8 4 \pm 0 . 0 0 1$ </td></tr><tr><td>MNIST-fashion</td><td> $0 . 9 0 0 \pm 0 . 0 0 1$ </td><td> $0 . 8 9 8 \pm 0 . 0 0 1$ </td><td> $0 . 8 9 9 \pm 0 . 0 0 2$ </td><td> $0 . 8 9 7 \pm 0 . 0 0 2$ </td><td> $0 . 8 9 9 \pm 0 . 0 0 1$ </td><td> $0 . 8 9 8 \pm 0 . 0 0 1$ </td><td> $0 . 8 9 9 \pm 0 . 0 0 1$ </td></tr><tr><td>Wine Classification</td><td> $0 . 5 8 3 \pm 0 . 0 1 1$ </td><td> $0 . 5 7 6 \pm 0 . 0 1 3$ </td><td> $0 . 5 6 4 \pm 0 . 0 1 8$ </td><td> $0 . 5 7 4 \pm 0 . 0 1 3$ </td><td> $0 . 5 6 3 \pm 0 . 0 1 8$ </td><td> $0 . 5 7 0 \pm 0 . 0 1 4$ </td><td> $0 . 5 6 4 \pm 0 . 0 1 8$ </td></tr><tr><td>Tiny ImageNet</td><td> $0 . 1 2 1 \pm 0 . 0 0 3$ </td><td> $0 . 1 1 9 \pm 0 . 0 0 1$ </td><td> $0 . 1 2 7 \pm 0 . 0 0 1$ </td><td> $0 . 1 1 8 \pm 0 . 0 0 2$ </td><td> $0 . 1 2 7 \pm 0 . 0 0 3$ </td><td> $0 . 1 2 2 \pm 0 . 0 0 1$ </td><td> $0 . 1 2 9 \pm 0 . 0 0 3$ </td></tr></table>

CIFAR100 Validation Accuracy Evolution  
![](images/92a2e5ab091eef6e233ba179d4dff9fe94b0f2959122383be025bd240c56ee50.jpg)  
Fig. 2. Ablation studies on CIFAR100 using a sparse GaNN. Validation accuracy evolution for diferent values of α.

GaNNs introduce two novel hyperparameters, the loss-weight α and the variance lower bound v. A sweep for α on CIFAR100 (figure 2), as well as our general experience while training, indicates that α is easy to tune. Only high values (0.5 and above) cause a collapse in performance. Performance turns out, within a reasonable range, to be insensitive to v — its main function is to ensure numerical stability.

Table 3. Test UQ (means). Top: Regression (Gaussian NLL). Bottom: Classification (NLL / AUROC).
<table><tr><td>Dataset</td><td colspan="6">GaNN-dense GaNN-sparse GaNN-sparse + D ANN + varhead ANN + varhead + D</td></tr><tr><td colspan="6">Regression (Gaussian NLL)</td></tr><tr><td>Airfoil</td><td>3.36</td><td>3.28</td><td>3.26</td><td>4.05</td><td></td></tr><tr><td>Yacht</td><td>2.79</td><td>1.99</td><td>2.01</td><td>3.02</td><td>4.10 3.06</td></tr><tr><td>UTKFace</td><td>12.92</td><td>7.57</td><td>7.89</td><td>4.50</td><td>4.50</td></tr><tr><td>Wine Regression</td><td>1.23</td><td>1.27</td><td>1.28</td><td>1.61</td><td>1.63</td></tr><tr><td>YearPredictionMSD</td><td>5.05</td><td>5.05</td><td>5.18</td><td>5.38</td><td>6.06</td></tr><tr><td colspan="6">Classification (NLL</td></tr><tr><td>CIFAR-100</td><td>/AUROC)</td><td>3.17  / 0.746</td><td>3.02  / 0.734</td><td></td><td></td></tr><tr><td>Digits-MNIST</td><td>3.17  / 0.750 0.11 0.954</td><td>0.12 0.953</td><td>0.13 /0.953</td><td>3.20 / 0.751 0.08 /0.974</td><td>3.04 / 0.735 0.06 / 0.975</td></tr><tr><td>Fashion-MNIST</td><td>0.36 0.853</td><td>0.44 0.828</td><td>0.47 / 0.829</td><td>0.34 / 0.896</td><td>0.30 /0.903</td></tr><tr><td>Wine Classification</td><td>1.02 0.592</td><td>1.01 0.599</td><td>1.03 /0.577</td><td>1.02 / 0.595</td><td>1.03 /0.566</td></tr><tr><td>Tiny ImageNet</td><td>4.77 0.702</td><td>4.81 0.706</td><td>4.18 /0.706</td><td>4.93 / 0.709</td><td>4.19 0.705</td></tr></table>

![](images/d706d2d301347c2623816650e559a3443481f3960c7a818bab8c38921f0878ca.jpg)

![](images/6df235cee1bf992567cb3b095929240287b017d7edcb28d2c89e24925ca37516.jpg)  
Fig. 3. Performance on the airfoil dataset, standard deviations are visualized as shaded regions. This dataset constitutes an extreme example of the predicted delayed convergence resulting in higher final performance. Delayed convergence is caused by first optimizing inner loss (left, dashed) and then optimizing the base loss.

## 5.2 Uncertainty Quantification

GaNNs can be used for uncertainty quantification, as can be gleaned from the results shown in Table 3. In case of regression, both GaNNs outperform their competitors. The best overall performance is arguably achieved by a combination of GaNNs with dropout. In the case of classification, the results are far less consistent.

## 6 Discussion

The consistent positive performance impact of the GaNN architecture makes them both theoretically and practically interesting. In our experiments, the sparse noise model proved superior to the dense one, not just because of computational and memory eficiency, but because of a tendency towards more stable convergence.

The generally better regression results might speak to the fact that the regression models fit our Gaussian assumptions better. On the other hand, this paper focused on the usage of pure Gaussian models. In practice, it might make sense to combine standard architectures with a few Gaussian layers to reap all positive efects. Also, the investigation of deeper models as well as the interaction of internal Gaussian models with convolutions and attention mechanisms might prove useful fields of further study. Finally, we note that the specifically Gaussian assumptions may be generalized to other kinds of distributions.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Gal, Y., Ghahramani, Z.: Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. arXiv:1506.02142 (2016)

2. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R.: Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research 15(56), 1929–1958 (2014)

3. Nix, D.A., Weigend, A.S.: Estimating the mean and variance of the target probability distribution. In: Proceedings of 1994 IEEE International Conference on Neural Networks (ICNN’94), vol. 1, pp. 55–60. IEEE (1994). https://doi.org/10. 1109/ICNN.1994.374138

4. Hüllermeier, E., Waegeman, W.: Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods. Machine Learning 110(3), 457– 506 (2021)

5. Lakshminarayanan, B., Pritzel, A., Blundell, C.: Simple and scalable predictive uncertainty estimation using deep ensembles. In: Advances in Neural Information Processing Systems, vol. 30 (2017)

6. Kendall, A., Gal, Y.: What uncertainties do we need in Bayesian deep learning for computer vision? In: Proceedings of the 31st International Conference on Neural Information Processing Systems, pp. 5580–5590. Curran Associates Inc., Red Hook, NY, USA (2017)

7. MacKay, D.J.C.: A Practical Bayesian Framework for Backpropagation Networks. Neural Computation 4(3), 448–472 (1992). https://doi.org/10.1162/neco.1992.4.3. 448

8. Feldman, H., Friston, K.: Attention, Uncertainty, and Free-Energy. Frontiers in Human Neuroscience 4, 215 (2010). https://doi.org/10.3389/fnhum.2010.00215

9. Kuhn, P., Schweizer, D.: Uncertainties in Iceberg Detection from Satellite Data: Error-Modelling for the Quantification of Total Uncertainty in Image Segmentation. In: Abrahamsen, E.B., Aven, T., Bouder, F., Flage, R., Ylönën, M. (eds.) Proceedings of the 35th European Safety and Reliability & the 33rd Society for Risk Analysis Europe Conference. Research Publishing, Singapore (2025). https://doi.org/10.3850/978-981-94-3281-3\_ESREL-SRA-E2025-P0089-cd

10. Wilson, A.G., Izmailov, P.: Bayesian deep learning and a probabilistic perspective of generalization. In: Proceedings of the 34th International Conference on Neural Information Processing Systems, articleno. 394. Curran Associates Inc., Red Hook, NY, USA (2020)

11. Brooks, T.F., Pope, D.S., Marcolini, M.A.: Airfoil self-noise and prediction. Technical Report, NASA Langley Research Center (1989)

12. Gerritsma, J., Onnink, R., Versluis, A.: Yacht Hydrodynamics [Dataset]. UCI Machine Learning Repository (1981). https://doi.org/10.24432/C5XG7R

13. UTKFace: A Large-Scale Face Dataset for Age, Gender, and Ethnicity (2018). https://susanqq.github.io/UTKFace/

14. Cortez, P., Cerdeira, A., Almeida, F., Matos, T., Reis, J.: Modeling wine preferences by data mining from physicochemical properties. Decision Support Systems 47(4), 547–553 (2009)

15. Krogh, A., Hertz, J.: A Simple Weight Decay Can Improve Generalization. In: Moody, J., Hanson, S., Lippmann, R.P. (eds.) Advances in Neural Information Processing Systems, vol. 4. Morgan-Kaufmann (1991)

16. Krizhevsky, A.: Learning Multiple Layers of Features from Tiny Images. Technical Report, University of Toronto (2012)

17. Xiao, H., Rasul, K., Vollgraf, R.: Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv:1708.07747 (2017)

18. LeCun, Y., Cortes, C., Burges, C.J.C.: The MNIST database of handwritten digits (1998). http://yann.lecun.com/exdb/mnist/

19. Malinin, A., Gales, M.: Predictive uncertainty estimation via prior networks. In: Proceedings of the 32nd International Conference on Neural Information Processing Systems, pp. 7047–7058. Curran Associates Inc., Red Hook, NY, USA (2018)

20. Amini, A., Schwarting, W., Soleimany, A., Rus, D.: Deep evidential regression. In: Proceedings of the 34th International Conference on Neural Information Processing Systems, articleno. 1251. Curran Associates Inc., Red Hook, NY, USA (2020)

21. Kendall, A., Gal, Y., Cipolla, R.: Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. CoRR abs/1705.07115 (2017)