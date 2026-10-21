# Robust Representation Learning for Spiking Neural Networks based on Conditional Information Bottleneck

Yuli Zha<sup>a</sup>, Wang Fan<sup>a</sup>, Yixiao Dai<sup>a</sup>, Lingjie Wang<sup>a</sup>, Bang Hu<sup>a</sup>, Wei Cao<sup>b</sup>, Fengzhe Zhang<sup>b</sup> and Fan Zhang<sup>b,∗</sup>

<sup>a</sup>College ofComputer Science and Artificial Intelligence, Fudan University, Shanghai, 200433, China

<sup>b</sup>Institute of Big Data, Fudan University, Shanghai, 200433, China

## A R T I C L E I N F O

Keywords:   
Spiking Neural Networks   
Information Bottleneck   
Conditional Mutual Information   
Neuromorphic Computing   
Energy eficiency

## A BS T RA C T

Spiking Neural Networks (SNNs) have shown great potential for low-power neuromorphic computing due to their event-driven processing and spatiotemporal sparsity, but they remain vulnerable to input perturbations, hardware-intrinsic noise, and adversarial attacks in real-world deployment. We attribute this vulnerability partly to input-specific hidden representations learned by standard SNNs trained with conventional cross-entropy loss, as our information-plane analysis empirically shows that substantial input-related information can still be retained after label-predictive information becomes stable. To address this issue, we propose SCIB, a simple yet efective robust learning objective for SNNs based on the Conditional Information Bottleneck. Diferent from conventional variational IB methods, SCIB replaces the mutual information compression term �(�; �) with the conditional mutual information �(�; �|�) and derives a tractable variational upper bound. Moreover, it introduces a learnable class-conditional prior instead of a label-agnostic isotropic Gaussian prior, improving the expressiveness of variational compression and encouraging class-structured representations when combined with the supervised classification objective. Systematic experiments on Fashion-MNIST, CIFAR-10, and CIFAR-100 under external environmental noise, neuromorphic hardware noise, and adversarial attacks show that SCIB improves robustness over standard SNNs and IB-based baselines under most evaluated noise conditions, improves adversarial accuracy under the evaluated attacks without using adversarial training, and reduces the SynOps-based single-sample inference energy consumption by 24.9% relative to an unconstrained SNN baseline.

## 1. Introduction

Neuromorphic computing, which emulates the operating mechanisms of the biological brain, ofers a promising pathway for overcoming the energy-eficiency bottleneck of conventional von Neumann architectures [1, 2]. As the core computational paradigm of neuromorphic computing, Spiking Neural Networks (SNNs) encode and transmit information through discrete spike events, and are characterized by event-driven processing, temporal sensitivity, and inherent sparsity [3, 4]. On neuromorphic hardware platforms such as Intel Loihi [5], IBM TrueNorth [6], and Tianjic [7], SNNs can achieve remarkable energy-eficiency advantages over conventional architectures [8, 9]. In recent years, surrogate gradient-based direct training methods [10, 11] have efectively alleviated the non-diferentiability of the spike firing function, promoting the application of SNNs to image classification [12, 13], object detection [14], and other neuromorphic perception tasks.

However, SNNs still face severe robustness challenges in real-world deployment environments. The perturbations encountered by SNNs can be broadly categorized into three types. First, environmental perturbations and random input noise directly corrupt spike-encoded signals at the sensory or input-encoding stage [15]. Second, non-ideal factors intrinsic to neuromorphic hardware, such as thermal noise, synaptic weight drift, and stochastic membrane-potential fluctuations, may introduce deviations in spike timing and neuronal responses [16]. Third, surrogate gradient-based adversarial attacks can exploit diferentiable approximations of spike dynamics to construct malicious perturbations, posing serious threats to safety-critical SNN applications [17, 18].

Existing studies have developed diferent robustnessenhancement strategies for these perturbation sources. For input-level noise, methods including spike encoding optimization, crafted-noise training, and frequency-domain filtering aim to suppress perturbations before or during spike generation [19, 20, 21]. For hardware-induced neuronal perturbations, bio-inspired stochastic mechanisms and heterogeneous neuronal dynamics improve robustness by injecting structured randomness into spike transmission, membrane evolution, or synaptic dynamics [22, 23, 24]. For gradientbased adversarial perturbations, regularization-based and adversarial training methods constrain perturbation propagation by imposing spectral, Lipschitz-type, or gradient-level restrictions on SNNs [25, 26, 27]. Although these methods improve SNN robustness from diferent perspectives, they are often designed for specific perturbation models or attack types. Encoding-based methods mainly operate near the input stage, stochastic mechanisms may require carefully designed noise processes or modified neuron models, and adversarial training usually relies on attack-specific sample generation and expensive inner-loop optimization. For these reasons, it is desirable to develop a simple objective-level surrogate for SNN training that is easy to implement, does not alter the SNN architecture or neuronal dynamics, and can generalize across heterogeneous perturbation sources.

Our approach is motivated by an empirical informationplane observation: trained SNN representations can retain substantial or even redundant input-related information even after their label-predictive information becomes stable [28, 29]. This observation suggests that spike representations may contain input-specific variations, some of which can be perturbation-sensitive or redundant for the target classification task and may amplify sensitivity to environmental noise, hardware perturbations, and adversarial attacks.

Information Bottleneck (IB) theory provides a principled framework for learning minimally suficient representations by preserving label-relevant information while compressing input-dependent redundancy [30, 31]. It has been widely applied to conventional neural networks, including variational deep networks, convolutional neural networks, recurrent or sequential models, and robust representation learning frameworks, to improve compactness, generalization, and adversarial robustness [32, 33, 34, 35, 36, 37]. However, compared with its extensive use in conventional deep neural networks, the systematic use of IB principles for improving SNN robustness remains relatively underexplored.

Unfortunately, directly applying conventional IB objective or the Variational Information Bottleneck (VIB) [32] to SNNs may lead to suboptimal performance. The first dificulty lies in the compression target. Conventional IB objectives typically penalize the mutual information �(�; �) between the input � and the learned representation �. Under the supervised representation-learning Markov chain $Y  X  S$ , where � is generated from � and has no additional access to � at inference time, the total information �(�; �) can be decomposed into the label-relevant predictive term �(�; �) and the label-conditioned term �(�; �|�). Therefore, directly penalizing �(�; �) may compress useful discriminative information together with nuisance factors, potentially leading to a robustness–accuracy trade-of. [38]. The second dificulty lies in the variational prior. Standard VIB usually adopts a label-agnostic unimodal Gaussian prior, such as <sup></sup>(0, �), which pulls representations from all classes toward a single distributional anchor. Such a prior has limited expressiveness for class-structured spike representations and may weaken inter-class separability, especially when the number of categories increases or when noisy spike features are involved. These limitations motivate a more selective compression objective and a class-aware prior structure for robust SNN representation learning.

To address the above limitations, we propose SCIB, a Spiking Conditional Information Bottleneck framework for robust SNNs. Instead of using the conventional mutual information $I ( X ; S )$ as the compression objective, SCIB takes the conditional mutual information $I ( X ; S | Y )$ as the ideal compression target and implements it through a tractable variational upper-bound regularizer. We further introduce a learnable class-conditional prior, where each class is assigned an independent distributional anchor in the latent space. This design provides a more expressive prior than the label-agnostic standard Gaussian used in conventional VIB, while encouraging intra-class compactness and helping maintain inter-class separability when combined with the classification objective. SCIB can be integrated into standard surrogate-gradient training as an objective-level regularizer, without modifying the SNN architecture or neuronal dynamics. The main contributions of this paper are summarized as follows:

• Conditional information bottleneck framework for robust SNNs. Unlike perturbation-specific defense methods and conventional IB objectives based on indiscriminate compression of $I ( X ; S )$ , SCIB is motivated by $I ( X ; S | Y )$ and implements conditional compression through a tractable variational upper-bound regularizer. This design targets label-conditioned inputspecific information while relying on the supervised loss to preserve predictive information, thereby mitigating the robustness–accuracy tension observed with indiscriminate IB compression.

• Learnable class-conditional prior. We derive a tractable variational upper bound for the conditional compression term and instantiate it with a learnable classconditional prior. The resulting prior can be interpreted as a Gaussian mixture over labels, providing class-specific distributional anchors for spike representations and improving the expressiveness of variational compression.

• Systematic robustness evaluation under heterogeneous perturbations. On Fashion-MNIST, CIFAR-10, and CIFAR-100, we evaluate SCIB under environmental noise, neuromorphic hardware noise, and adversarial attacks. Results show that SCIB improves robustness over standard SNNs and existing IB-based baselines under diverse noise conditions, and improves adversarial accuracy under the evaluated attacks without adversarial training.

• Joint robustness and energy analysis. SynOps-based analysis shows that SCIB reduces spike firing activity while improving robustness in the evaluated settings, lowering the analytical single-sample computationenergy by 24.9% relative to the unconstrained SNN baseline.

## 2. Related work

## 2.1. Robustness of spiking neural networks

Spiking Neural Networks (SNNs) encode and transmit information through discrete spike events, where neuronal states evolve over time according to membrane-potential dynamics. This event-driven computation paradigm provides SNNs with temporal sparsity and high energy efficiency, but the non-diferentiability of the spike firing function makes their training more challenging than that of conventional artificial neural networks (ANNs) [3, 4, 39]. Existing training paradigms mainly include ANN-to-SNN conversion [40], membrane-potential-based learning [41], and surrogate gradient-based direct training [10, 11]. Among them, surrogate gradient learning has become the dominant approach for training deep SNNs, but the diferentiable approximations introduced during training also make SNNs susceptible to gradient-based adversarial attacks [17, 18].

Despite recent progress in classification accuracy and deployment eficiency, SNNs remain vulnerable to heterogeneous perturbations in real-world environments, including input-level sensory noise, hardware-induced neuronal or synaptic noise, and gradient-based adversarial perturbations [15, 16, 17, 18]. These perturbations can disturb spike timing, membrane-potential evolution, or input spike patterns, and may accumulate across timesteps and layers in deep SNNs.

Existing SNN robustness methods can be roughly grouped according to the perturbation source they target. For inputlevel perturbations, spike encoding optimization methods aim to suppress noise at the input or temporal encoding stage. Discrete input encoding has been shown to provide partial intrinsic robustness [19], while HIRE-SNN [20] and FEEL-SNN [21] further improve robustness through crafted input noise or frequency-domain encoding. For hardwareinduced perturbations, stochastic modeling and bio-inspired mechanisms enhance robustness by injecting structured randomness into spike transmission, membrane evolution, or synaptic dynamics. Representative examples include stochastic gating mechanisms [22], random heterogeneous neuronal dynamics [23], and dynamic synaptic noise modeling [24]. For adversarial perturbations, regularization-based and adversarial training methods constrain perturbation propagation by imposing explicit smoothness, spectral, or gradient-level restrictions. For example, SNN-RAT introduces Parseval regularization and mixed adversarial training to limit adversarial amplification across timesteps [25], while sparse-gradient methods improve robustness by suppressing adversarial gradient propagation [27].

Although these methods improve SNN robustness from diferent perspectives, most of them are closely tied to specific perturbation models. This leaves room for representationlevel learning objectives that can suppress perturbationsensitive redundancy without relying on a particular noise model, attack procedure, or architectural modification.

## 2.2. Information bottleneck in deep neural networks

Information Bottleneck (IB) theory provides a principled framework for learning compact and predictive representations [30, 31]. We denote the input by �, the class label by $Y ,$ and the learned representation by �. The mutual information $I ( X ; S )$ measures how much information about the input is retained in the representation, while $I ( S ; Y )$ measures how much label-relevant predictive information is preserved by the representation. For two continuous random variables $X$ and $\bar { Y }$ with joint probability density function $f ( x , y )$ and marginal probability density functions �(�) and �(�)

respectively, mutual information is defined as:

$$
I ( X ; Y ) = \iint f ( x , y ) \log { \frac { f ( x , y ) } { f ( x ) f ( y ) } } d x d y .\tag{1}
$$

Mutual information measures the strength of statistical dependence between � and � , with $I ( X ; Y ) = 0$ if and only if � and $Y$ are statistically independent. From the perspective of information entropy, mutual information can be equivalently decomposed as:

$$
I ( X ; Y ) = H ( X ) + H ( Y ) - H ( X , Y ) = H ( X ) - H ( X \vert Y ) ,\tag{2}
$$

where $H ( \cdot ) , H ( \cdot , \cdot )$ , and $H ( \cdot | \cdot )$ denote Shannon entropy, joint entropy, and conditional entropy, respectively. Eq. (2) reveals the intuitive meaning of mutual information: $I ( X ; Y )$ equals the reduction in uncertainty about � after observing $Y .$

The conventional IB objective, which is typically maximized, encourages � to preserve label-relevant information while compressing input-dependent information:

$$
\mathcal { I } _ { \mathrm { I B } } = \operatorname* { m a x } I ( S ; Y ) - \beta I ( X ; S ) ,\tag{3}
$$

where $\beta$ controls the trade-of between predictive suficiency and information compression.

The IB principle has been widely studied in deep neural networks (DNNs). Variational Information Bottleneck (VIB) [32] introduces variational inference to make the IB objective tractable in high-dimensional neural networks. Nonlinear IB [33], supervised IB [36], information bottleneck distillation [35], and gated IB [37] further extend the IB principle to generalization, adversarial robustness, distillation, and sequential decision-making. Recent studies have also adapted IB objectives and information estimators to specific learning tasks. Zhang et al. [42] formulated feature selection as a masked deterministic IB problem and employed matrix-based Rényi’s �-order entropy to obtain a tractable objective without variational approximation. Tang and Yu [43] employed variational IB and conditional mutual-information regularization to extract shared representations from two interdependent sequences. Informationplane studies also provide empirical tools for analyzing representation dynamics during neural network training [28, 29, 44, 45].

Compared with its extensive use in conventional DNNs, the application of IB theory to SNNs is still limited. SNIB [46] applies nonlinear information bottleneck regularization to spike-based learning and demonstrates that constraining input-representation dependence can improve SNN robustness. SeLHIB [47] extends high-order information bottleneck learning to self-supervised event-based optical flow estimation. These studies validate the feasibility of information compression in the spike domain, but existing IB-based SNN methods mainly inherit the conventional compression form from DNNs and have not systematically addressed conditional redundancy in spike representations or classstructured variational priors.

![](images/a31f3f71ca6c8abe7d6ca5c4d575c69d75ebc410f74d42fc547e396de9c6c6c5.jpg)

(a) Standard SNN on Fashion-MNIST  
![](images/bd5b7d071011e2586fa4b70fa4f9630051c867b836cef085281b54bfab54c6c5.jpg)  
(c) Conventional IB on Fashion-MNIST

![](images/ae00ebe1a1537ec82f3b14f29d113289c36bb359351171f0fc1edd77332dcca3.jpg)

(b) Standard SNN on CIFAR-10  
![](images/ae7a2eb8a34dece9c7d37b3e56e04ac6fba8c54343bcd5d8ea02c97c79645517.jpg)  
(d) Conventional IB on CIFAR-10  
Figure 1: Information-plane trajectories of standard SNNs and conventional IB-regularized SNNs on Fashion-MNIST and CIFAR 10. The first row shows the trajectories of standard SNNs, whereas the second row shows those obtained with conventional IB regularization.

## 3. Empirical observation

Before introducing the SCIB framework, we first analyze the representation dynamics of conventionally trained SNNs and IB-regularized SNNs from the perspective of the information plane [28, 29]. The information plane uses the mutual information between the input and the representation, $I ( X ; S )$ , as the horizontal axis, and the mutual information between the representation and the label, $I ( S ; Y )$ , as the vertical axis. This visualization provides an empirical view for examining whether a network mainly fits label-relevant information or also reduces input-dependent information during training. An increase in $I ( S ; Y )$ indicates that the learned representation becomes more predictive of the class label, whereas a decrease in $I ( X ; S )$ suggests a reduction of input-related information retained.

Fig. 1 compares the information-plane trajectories of standard SNNs and conventional IB-regularized SNNs on Fashion-MNIST and CIFAR-10. The first row shows the results of the standard SNNs, whereas the second row shows the results obtained after introducing conventional IB regularization.

As shown in the first row of Fig. 1, the trajectories of the standard SNNs generally move upward and to the right before gradually approaching saturation. During the early stage of training, both $I ( X ; S )$ and $I ( S ; Y )$ increase rapidly, indicating that the networks progressively extract input-dependent features that are useful for classification. During the middle and late stages, however, $I ( S ; Y )$ becomes relatively stable, while $I ( X ; S )$ remains in a highinformation region or continues to increase slowly. Under the adopted estimator and experimental settings, no evident compression phase is observed in the standard SNNs.

This behavior suggests that the learned spike representations may retain substantial input-dependent information even after the label-related information has largely stabilized. Such retained information may include input-specific details, local variations, or perturbation-sensitive components, some of which may not be essential for the final classification decision. Although the information-plane trajectories alone do not establish that all retained information is redundant or harmful, they indicate that conventionally trained SNN representations may benefit from an explicit mechanism that controls unnecessary input dependence.

In contrast, the second row of Fig. 1 shows that the trajectories obtained with conventional IB regularization exhibit a diferent pattern. During the early stage of training, the trajectories still move upward and to the right, indicating that the networks first acquire label-predictive information. During the middle and late stages, the trajectories gradually bend toward the left: $I ( X ; S )$ decreases from a relatively high-information region, while $I ( S ; Y )$ remains at a comparatively high level. This fold-back behavior is consistent with the intended efect of conventional IB regularization, which encourages the representation to reduce its dependence on the input while retaining information relevant to label prediction.

However, conventional IB achieves compression by directly penalizing the total mutual information $I ( X ; S )$ . As discussed in the following section, under the supervised Markov assumption $Y \  \ X \  \ S$ can be decomposed into the label-predictive component �(�; �) and the labelconditioned component $I ( X ; S ~ \mid ~ Y )$ . Therefore, directly compressing $I ( X ; S )$ does not explicitly distinguish predictive information from input-specific information that remains after conditioning on the label. When the compression constraint is excessively strong, useful discriminative information may also be suppressed, potentially leading to a robustness–accuracy trade-of.

These observations motivate the proposed SCIB framework. Instead of indiscriminately penalizing the total dependence between the input and the representation, SCIB uses the conditional mutual information $I ( X ; S ~ \mid ~ Y )$ as its compression target. In this way, the regularization is directed toward input-specific variations that remain after conditioning on the class label, while the predictive term encourages the retention of label-relevant information.

## 4. Methodology

This section presents the theoretical motivation, variational formulation, learnable class-conditional prior, and optimization procedure of the proposed Spiking Conditional Information Bottleneck (SCIB) framework.

## 4.1. Limitations of information bottleneck

In supervised SNN representation learning, we assume the Markov chain $Y  X  S $ : the representation � is produced from the input �, and labels are used only during training for supervision and regularization. Under this assumption, � and � are conditionally independent given �, i.e., $I ( S ; Y | X ) = 0$ . Applying the chain rule of mutual information gives

$$
I ( X ; S ) = I ( S ; Y ) + I ( X ; S \vert Y ) ,\tag{4}
$$

where �(�; �) corresponds to label-relevant predictive information, and $I ( X ; S | Y )$ measures the residual dependence between � and � after the label � is given. Thus, $I ( X ; S | Y )$ measures the input-specific information retained in � after conditioning on the label. This term may include nuisance factors or perturbation-sensitive components, but it may also contain useful intra-class variations such as pose, texture, style, subclass structure, or temporal dynamics.

This decomposition clarifies a limitation of conventional IB regularization. Standard IB compresses representations by penalizing �(�; �), but Eq. (4) shows that $I ( X ; S )$ contains both predictive information and label-conditioned redundancy. Therefore, directly minimizing $I ( X ; S )$ does not explicitly distinguish predictive information from other input-dependent factors; when the compression strength is large, it may suppress useful discriminative features together with nuisance variations. Related conditional bottleneck formulations have been used to separate task-relevant or shared information from unnecessary input dependence. The Conditional Entropy Bottleneck formalizes supervised compression through $I ( Z ; X | Y )$ [38], while recent sequential models employ conditional mutual-information terms to suppress stream-specific information from shared dynamic representations [43]. In contrast, SCIB conditions directly on the class label and formulates $I ( X ; S | Y )$ as labelconditioned redundancy in supervised spike representations.

To formalize the desired representation, we consider the notion of a minimal suficient representation. Under the Markov chain $Y  X  S$ , the data processing inequality gives $I ( S ; Y ) ~ \le ~ I ( X ; Y )$ [48]. Following the minimalsuficiency principle, a desirable representation $S ^ { * }$ should satisfy two conditions.

$$
I ( S ^ { * } ; Y ) = I ( X ; Y ) ,\tag{5}
$$

which means that $S ^ { * }$ preserves all label-relevant information contained in �. Second, among all suficient representations, it should be minimal with respect to the retained input information. Since Eq. (4) gives

$$
I ( X ; S ^ { \ast } ) = I ( S ^ { \ast } ; Y ) + I ( X ; S ^ { \ast } | Y ) ,\tag{6}
$$

any suficient representation satisfies

$$
I ( X ; S ^ { * } ) = I ( X ; Y ) + I ( X ; S ^ { * } | Y ) \ge I ( X ; Y ) .\tag{7}
$$

Therefore, the smallest possible value of $I ( X ; S ^ { * } )$ under suficiency is $I ( X ; Y )$ , which is attained when $I ( X ; S ^ { * } | Y ) =$ 0. The ideal minimal suficient representation thus satisfies

$$
I ( X ; S ^ { * } ) = I ( S ^ { * } ; Y ) = I ( X ; Y ) ,\tag{8}
$$

indicating that a minimal suficient representation should preserve predictive information while minimizing inputspecific information beyond the label.

A second limitation appears in the variational implementation of conventional IB. Standard VIB commonly adopts a label-agnostic unimodal Gaussian prior such as $\mathcal { N } ( 0 , I )$ When applied to class-structured spike representations, this prior pulls posteriors from all classes toward a single global anchor, which may limit representation expressiveness and weaken inter-class separability. These two limitations motivate SCIB to use a conditional compression objective together with a learnable class-conditional prior.

![](images/4f6d2cb590363bcffa11805dcf8f2ee45095027e80a86bc75bf833d20d9b0b03.jpg)  
Figure 2: Information diagram for the SCIB framework.

As illustrated in Fig. 2, conventional IB minimizes the whole input–representation dependence $I ( X ; S )$ , whereas SCIB aims to compress the residual input information that remains after conditioning on the label. This replaces the indiscriminate compression term $I ( X ; S )$ with a labelconditioned compression term motivated by $I ( X ; S | Y )$ .

## 4.2. SCIB objective function

Based on the analysis above, reducing $I ( X ; S | Y )$ provides a way to regularize input-specific information that remains after conditioning on the class label. Compared with the conventional compression objective $I ( X ; S )$ , this choice has two advantages. First, it separates compression from prediction more explicitly, because $I ( X ; S | Y )$ corresponds to the part of $I ( X ; S )$ beyond the predictive information $I ( S ; Y )$ . Second, it provides an interpretable compression target: a smaller $I ( X ; S | Y )$ indicates weaker dependence between individual inputs and representations after the label is given.

To apply this principle to spike representations, we use the Leaky Integrate-and-Fire (LIF) neuron as the basic feature extraction unit. At discrete timestep $t ,$ the membrane potential $V _ { t }$ and the instantaneous output spike $S _ { t }$ follow

$$
V _ { t } = \tau V _ { t - 1 } + W \cdot S _ { t - 1 } ,\tag{9}
$$

$$
S _ { t } = H ( V _ { t } - V _ { \mathrm { t h } } ) ,\tag{10}
$$

where � denotes the membrane-potential decay constant, � is the synaptic weight matrix, $V _ { \mathrm { t h } }$ is the firing threshold, and $H ( \cdot )$ is the Heaviside step function. Over � timesteps, the SNN encoder extracts a high-dimensional latent spike representation from �, modeled by the stochastic encoder distribution $p _ { \theta } ( s | x )$ .

We define the SCIB information-theoretic objective to be maximized as

$$
\mathcal { I } _ { \mathrm { S C I B } } = \operatorname* { m a x } I ( S ; Y ) - \beta \cdot I ( X ; S | Y ) ,\tag{11}
$$

where $\beta$ controls the strength of conditional compression. This objective encourages the representation to preserve label-relevant predictive information while suppressing inputspecific redundancy conditioned on the label.

## 4.3. Variational bound derivation

The SCIB information-theoretic objective in Eq. (11) involves $I ( X ; S | Y )$ and $I ( S ; Y )$ , which are intractable in highdimensional feature spaces; therefore, the practical training loss optimizes variational bounds rather than the exact information quantities. We derive tractable variational bounds for both terms.

For the conditional compression term, the Markov structure gives $p _ { \theta } ( s | x , y ) = p _ { \theta } ( s | x )$ , and

$$
I ( X ; S | Y ) = \mathbb { E } _ { p ( x , y ) } { \big [ } D _ { \mathrm { K L } } { \big ( } p _ { \theta } ( s | x ) \| p ( s | y ) { \big ) } { \big ] } .\tag{12}
$$

Since the true marginal conditional distribution $p ( s | y )$ is intractable, we introduce a variational prior $r _ { \phi } ( s | y )$ to approximate it. Using the non-negativity of KL divergence, we obtain

$$
\begin{array} { r } { I ( X ; S | Y ) = \mathbb { E } _ { p ( x , y ) } \big [ D _ { \mathrm { K L } } \big ( p _ { \theta } ( s | x ) \| r _ { \phi } ( s | y ) \big ) \big ] } \\ { - \mathbb { E } _ { p ( y ) } \big [ D _ { \mathrm { K L } } \big ( p ( s | y ) \| r _ { \phi } ( s | y ) \big ) \big ] . } \\ { \le \mathbb { E } _ { p ( x , y ) } \big [ D _ { \mathrm { K L } } \big ( p _ { \theta } ( s | x ) \| r _ { \phi } ( s | y ) \big ) \big ] } \end{array}\tag{13}
$$

Eq. (13) replaces the intractable marginal conditional distribution $p ( s | y )$ with the learnable parameterized distribution $r _ { \phi } ( s | y )$ , yielding a tractable upper bound rather than the exact conditional mutual information. The bound is tight when $r _ { \phi } ( s | y ) = p ( s | y )$ ; therefore, learning a high-quality conditional prior $r _ { \phi } ( s | y )$ is critical for tightening the upper bound.

Variational lower bound for the prediction term. For the prediction term $I ( S ; Y )$ , we introduce a variational classifier $q _ { \psi } ( y | s )$ as an approximation to the true predictive distribution �(�|�). Using the relation $I ( S ; Y ) = H ( Y ) - H ( Y \vert S )$ and the fact that cross-entropy upper-bounds conditional entropy, we derive:

$$
I ( S ; Y ) \geq H ( Y ) + \mathbb { E } _ { p ( x , y ) } \big [ \log q _ { \psi } ( y | s ) \big ] .\tag{14}
$$

Since the empirical entropy $H ( Y )$ of the dataset label distribution is constant during training, maximizing the lower bound on predictive information is equivalent to minimizing the standard cross-entropy loss $\mathcal { L } _ { \mathrm { C E } } = - \mathbb { E } _ { p ( x , y ) } [ \log q _ { \psi } ( y | s ) ]$

Combining the variational upper bound of Eq. (13) with the variational lower bound of Eq. (14), maximizing the intractable objective in Eq. (11) is converted into minimizing the following surrogate loss function amenable to end-to-end optimization:

$$
{ \mathcal { L } } _ { \mathrm { S C I B } } = { \mathcal { L } } _ { \mathrm { C E } } + \beta \mathbb { E } _ { p ( x , y ) } { \big [ } D _ { \mathrm { K L } } { \big ( } p _ { \theta } ( s | x ) \| r _ { \phi } ( s | y ) { \big ) } { \big ] }\tag{15}
$$

where the hyperparameter $\beta > 0$ controls the conditional compression strength in the feature space. The $\mathcal { L } _ { \mathrm { C E } }$ term in Eq. (15) encourages the representation � to retain predictive information for classification decisions, while the KL divergence term pulls the sample-level encoder distribution $p _ { \theta } ( s | x )$ towards the class-conditional prior $r _ { \phi } ( s | y )$ of the corresponding class, encouraging class-conditioned compactness. In this way, the representation is encouraged to reduce its dependence on input-specific variations after conditioning on the label, which may help reduce perturbationsensitive components without explicitly assuming a particular noise type.

## 4.4. Learnable class-conditional prior

The quality of the variational upper bound depends on how well $r _ { \phi } ( s | y )$ approximates $p ( s | y )$ . Conventional variational inference typically adopts a label-agnostic standard Gaussian as the prior, but this choice may encourage feature posteriors of diferent classes to share the same global anchor, potentially weakening inter-class separability in the representation space and increasing the risk of representation degeneration and inter-class confusion. To address this, we propose a Learnable Class-Conditional Prior that assigns each class an independent distributional anchor in the latent space.

For a classification task with � classes, we assign each class $k \in \{ 1 , \ldots , K \}$ a set of independent learnable parameters: a mean vector $\mu _ { k } \in \mathbb { R } ^ { d }$ and a variance vector $\sigma _ { k } ^ { 2 } \in \mathbb { R } _ { + } ^ { d }$ where � is the latent feature dimensionality. The variational prior is defined as a set of class-specific diagonal Gaussian distributions:

$$
r _ { \phi } ( s \mid y = k ) = \mathcal { N } \big ( s ; \mu _ { k } , \mathrm { d i a g } ( \sigma _ { k } ^ { 2 } ) \big ) .\tag{16}
$$

After marginalizing over labels, the class-conditional prior can be interpreted as a Gaussian mixture:

$$
r _ { \phi } ( s ) = \sum _ { k = 1 } ^ { K } p ( y = k ) r _ { \phi } ( s \mid y = k ) .\tag{17}
$$

The SNN encoder $p _ { \theta } ( s | x )$ extracts sample-level latent feature mean $\mu _ { x }$ and variance $\sigma _ { x } ^ { 2 }$ from the given input � through the spatiotemporal integration dynamics of multiple spiking layers, and constructs the variational posterior via the reparameterization trick:

$$
p _ { \theta } ( s \mid x ) = \mathcal { N } \big ( s ; \mu _ { x } , \mathrm { d i a g } ( \sigma _ { x } ^ { 2 } ) \big ) .\tag{18}
$$

Since both the variational prior and the variational posterior are parameterized as diagonal Gaussians, the KL divergence regularization term in Eq. (15) admits an eficient closed-form analytical solution, bypassing high-variance Monte Carlo sampling. For a spike representation of feature dimensionality �, given input � and its corresponding label $y = k$ , the analytical form of the KL divergence is:

$$
\begin{array} { r l } & { \quad D _ { \mathrm { K L } } \left[ p _ { \theta } ( s \mid x ) \parallel r _ { \phi } ( s \mid y = k ) \right] } \\ & { = \cfrac { 1 } { 2 } \displaystyle \sum _ { j = 1 } ^ { d } \left( \log \frac { \sigma _ { k , j } ^ { 2 } } { \sigma _ { x , j } ^ { 2 } } + \frac { \sigma _ { x , j } ^ { 2 } + ( \mu _ { x , j } - \mu _ { k , j } ) ^ { 2 } } { \sigma _ { k , j } ^ { 2 } } - 1 \right) \cdot } \end{array}\tag{19}
$$

Eq. (19) enters directly as the regularization component of the loss function, participating in the joint gradient updates of the SNN synaptic weights �, classifier parameters �, and prior parameters $\phi = \{ \bar { \mu _ { k } } , \mathrm { d i a g } ( \sigma _ { k } ^ { 2 } ) \} _ { k = 1 } ^ { K }$

From a representation-learning perspective, this regularization has two efects. First, it encourages intra-class compactness by pulling the posterior mean $\mu _ { x }$ toward the prior center $\mu _ { k }$ of the corresponding class and constraining posterior dispersion through the variance terms. Second, when combined with the cross-entropy objective, it may help maintain inter-class separability because diferent classes are assigned independent learnable distributional anchors, reducing the tendency of classes to be pulled toward a single standard Gaussian prior.

## 4.5. Overall framework and algorithm

Based on the theoretical derivations and mechanism designs presented above, this section presents the overall architecture and complete optimization algorithm of the SCIB framework. Fig. 3 illustrates the end-to-end training pipeline of SCIB.

First, the input � is processed by the SNN encoder over � timesteps to obtain the latent spike representation and the posterior statistics $( \mu _ { x } , \sigma _ { x } ^ { 2 } )$ . Second, the ground-truth label � is used only during training to select the corresponding class-conditional prior $r _ { \phi } ( s | y )$ , while the sampled latent variable is passed to the classifier for prediction. Finally, the cross-entropy loss and the closed-form KL regularizer are combined according to Eq. (15), and the SNN encoder parameters �, classifier parameters �, and prior parameters $\phi$ are jointly optimized via BPTT. During inference, the class-conditional prior is not used, and prediction follows the standard feedforward SNN pathway. The complete optimization procedure is summarized in Algorithm 1.

## 5. Experiments and results

## 5.1. Experimental setup

Environment and training configuration. All experiments are implemented using PyTorch 1.12 and Python 3.9 on Ubuntu 20.04, with training and evaluation conducted on an NVIDIA A100 GPU. All models are trained with the SGD optimizer with momentum 0.9, an initial learning rate of 0.1, and a cosine annealing schedule for learning rate decay. The batch size is 128. The compression strength hyperparameter � of SCIB is determined via validation set sensitivity analysis, with a default value of $\beta ~ = ~ 0 . 4 ;$ for hardware noise experiments, $\beta \ : = \ : 0 . 2$ is adopted to account for the diferential sensitivity of membrane potential perturbations to the compression constraint.

Datasets. To evaluate SCIB across varying levels of data complexity, three widely used image classification benchmark datasets are selected: Fashion-MNIST [49], CIFAR-10 [50], and CIFAR-100 [50]. The three datasets exhibit an increasing relationship in terms of class granularity and feature space dimensionality.

Network architecture. The backbone network is a VGG-9 SNN architecture adapted for the spike domain, comprising 7 convolutional layers (Conv + BN + LIF) and 1 fully connected layer, with the number of timesteps set to $T \ = \ 4 ;$ all spiking neurons adopt the LIF model. In adversarial attack experiments, comparisons are conducted under two timestep settings, � = 4 and $T \ = \ B ,$ to evaluate the influence of temporal depth and align with the configuration of FEEL-SNN.

![](images/0b4c318943144a0bb9bbcb48eb5719f394671f0cdb22ed801fe564c92607f920.jpg)  
Figure 3: The overall architecture of the SCIB framework.

Baseline methods. Since random perturbations and adversarial attacks correspond to diferent threat models, two groups of comparison baselines are employed. For noise robustness experiments, we compare SCIB with a standard SNN baseline and several IB-based SNN baselines: (1) Without IB (w.o. IB) [11]: a standard SNN trained with the surrogate gradient method using cross-entropy loss and no information compression constraint, serving as the ablation control; (2) IB [32]: the conventional information bottleneck applied directly to SNN training, with a standard Gaussian $\mathcal { N } ( 0 , I )$ as the variational prior to minimize the variational upper bound of �(�; �); (3) NBP-SNN [51]: a non-backpropagation SNN training method based on the IB principle, computing local losses using the membrane potential of each layer as the information representation; (4) SIB and QIB [46]: the squared and quartic variants of the nonlinear information bottleneck within the SNIB framework. For adversarial attack experiments, we compare SCIB with two reference methods: (1) Vanilla SNN [11]: standard cross-entropy training with no defense mechanism; (2) FEEL-SNN [21]: a frequency-encoding-based robust SNN method with an evolutionary leak factor, where the frequency parameter is set to � = [16, 14, 12, 10] for � = 4 and � = [16, 14, 12, 10, 8, 6, 4, 2] for � = 8. SCIB does not employ any adversarial training mechanism; its adversarial robustness is evaluated as a by-product of representationlevel regularization rather than as a replacement for dedicated adversarial training.

External environmental noise. Three representative types of external environmental noise are introduced to evaluate the interference resistance of SCIB in real physical environments. The Normalized Noise Intensity (NNI) is defined independently for each noise type.

(1) Gaussian noise simulates sensor thermal noise and electronic circuit floor noise. A perturbation drawn from a normal distribution is added to each pixel � in the image:

$$
{ \bar { x } } = x + { \mathcal { N } } ( 0 , \sigma ^ { 2 } ) ,\tag{20}
$$

where NNI is defined as the standard deviation �, with tested values ��� = 0.25 and ��� = 1.0.

(2) Salt-and-pepper noise simulates impulse interference caused by dead pixels in event camera sensors or spike transmission errors. A proportion � of pixels in the image are randomly replaced by extreme values, 0 or 1:

$$
{ \bar { x } } = \operatorname { S P } ( x , P ) ,\tag{21}
$$

where NNI is defined as the noise density �. Two pixel corruption ratios are tested: 10% (NNI = 0.1) and 30% (NNI = 0.3).

(3) Rayleigh noise is used to evaluate model robustness under non-Gaussian and asymmetric noise distributions:

$$
\begin{array} { r } { \bar { x } = x + \sigma _ { \mathrm { r a y } } \sqrt { - 2 \ln ( 1 - u ) } , \quad u \sim \mathcal { V } ( 0 , 1 ) , } \end{array}\tag{22}
$$

where $\sigma _ { \mathrm { r a y } }$ is the scale parameter of the Rayleigh distribution, and NNI is defined as $\sigma _ { \mathrm { r a y } }$ . Two perturbation levels are tested: NNI = 0.25 and NNI = 1.0.

Neuromorphic hardware noise. When deploying SNNs on edge computing devices or mixed-signal neuromorphic chips, the non-ideal characteristics of physical devices introduce unavoidable internal noise. Two representative hardware-level perturbations are considered.

Algorithm 1: Optimization of SCIB   
Input : Training dataset $D = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N } ;$ ; number   
of classes $K ;$ timestep $T ;$ parameters $\theta$   
(including decay constant $\tau ,$ threshold $V _ { \mathrm { t h } } ,$   
weights $W ) ;$ classifier parameters $\psi ;$ prior   
parameters $\boldsymbol { \phi } = \{ \mu _ { k } , \sigma _ { k } ^ { \hat { 2 } } \} _ { k = 1 } ^ { K } ;$ compression   
coeficient $\beta ;$ learning rate $\eta ;$ training   
epoch $E ;$ mini-batch �   
Output: Optimized parameters $\theta ^ { * }$ , classifier   
parameters $\psi ^ { * }$ , prior parameters $\phi ^ { * }$   
1 Initialize SNN weights $\theta$ and classifier parameters   
$\psi ;$   
2 for $k \gets 1$ to � do   
3 Initialize prior mean $\mu _ { k } \sim \mathcal { N } ( 0 , 0 . 0 1 I )$ and   
log-variance log $\sigma _ { k } ^ { 2 } \sim \mathcal { N } ( 0 , 0 . 0 1 I ) ;$   
4 end   
5 for epoch ← 1 to � do   
6 for each mini-batch $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { B } \subseteq D$ do   
7 Initialize membrane potentials $V _ { l } ^ { ( 0 ) } = 0$ and   
spikes $S _ { l } ^ { ( 0 ) } = 0$ for all layers $l ;$   
8 for $t \gets 1$ to $T$ do   
9 for $l \gets 1$ to � do   
10 LIF dynamics: $V _ { I } ^ { ( t ) } =$   
$\tau V _ { l } ^ { ( t - 1 ) } \odot ( 1 - \dot { S _ { l } } ^ { ( t - 1 ) } ) + W _ { l - 1 , l } S _ { l - 1 } ^ { ( t ) } ;$   
11 Spike emission:   
$\boldsymbol { S } _ { l } ^ { ( t ) } = \boldsymbol { H } ( \boldsymbol { V } _ { l } ^ { ( t ) } - \boldsymbol { V } _ { \mathrm { t h } } ) ;$   
12 end   
13 end   
14 Aggregate spikes over time to obtain latent   
representation ${ \bar { S } } ;$   
15 Extract posterior statistics:   
$( \mu _ { x } , \log \sigma _ { x } ^ { 2 } ) = \operatorname { E n c o d e r } ( { \bar { S } } ; \theta ) ;$   
16 Reparameterize: $z = \mu _ { x } + \sigma _ { x } \odot \varepsilon ,$   
$\varepsilon \sim { \mathcal { N } } ( 0 , I ) ;$   
17 Compute prediction: ̂� = Classifier(�; �);   
18 Compute cross-entropy loss:   
${ \mathcal { L } } _ { \mathrm { C E } } = - \log q _ { \psi } ( y \mid z ) ;$   
19 Index class-conditional prior via label $y \colon$   
$\boldsymbol { r } _ { \phi } ( z \mid \boldsymbol { y } ) = \mathcal { N } ( \boldsymbol { \mu } _ { \boldsymbol { y } } , \mathrm { d i a g } ( \sigma _ { \boldsymbol { y } } ^ { 2 } ) ) ;$   
20 Compute conditional compression   
regularizer:   
$\mathcal { R } _ { \mathrm { K L } } = D _ { \mathrm { K L } } \big ( p _ { \theta } ( s | x ) \| r _ { \phi } ( s | y ) \big )$   
21 Compute SCIB loss:   
$\mathcal { L } _ { \mathrm { S C I B } } = \mathcal { L } _ { \mathrm { C E } } + \beta \cdot \mathcal { R } _ { \mathrm { K L } }$   
22 Compute gradients via BPTT: $\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { S C I B } } .$   
$\nabla _ { \psi } \mathcal { L } _ { \mathrm { S C I B } } , \nabla _ { \phi } \mathcal { L } _ { \mathrm { S C I B } } ;$   
23 Update parameters: $\theta  \theta - \eta \nabla _ { \theta } \mathcal { L } _ { \mathrm { S C I B } } ,$   
$\psi  \psi - \eta \nabla _ { \psi } \mathcal { L } _ { \mathrm { S C I B } } , \phi  \phi - \eta \nabla _ { \phi } \mathcal { L } _ { \mathrm { S C I B } } ;$   
24 end   
25 end

(1) Synaptic weight noise simulates thermal noise and conductance drift occurring during weight updates or state readout in physical synapses. Zero-mean Gaussian perturbations are injected into synaptic weight updates during training:

$$
w _ { \mathrm { u p d } } = \Delta w + \mathcal { N } ( 0 , \sigma _ { \mathrm { S N } } ^ { 2 } ) ,\tag{23}
$$

where $\Delta w$ is the ideal update magnitude computed by backpropagation, and $w _ { \mathrm { u p d } }$ is the weight actually applied to the network after perturbation.

(2) Membrane potential noise simulates background perturbations of neuronal membrane potentials caused by crosstalk, leakage currents, or thermal fluctuations in neuromorphic circuits:

$$
\tilde { V } = V + \mathcal { N } ( 0 , \sigma _ { \mathrm { M N } } ^ { 2 } ) ,\tag{24}
$$

where � is the theoretical membrane potential determined by the LIF neuron dynamics, and $\tilde { V }$ is the perturbed membrane potential. Two NNI levels are tested for both hardware noise types: NNI = 1 and NNI = 5.

Adversarial attack. Three gradient-based adversarial attack methods are employed to evaluate the adversarial robustness of SCIB. All attacks are conducted under the $L _ { \infty }$ norm threat model with perturbation budget $\epsilon \in$ {4∕255, 8∕255}. The non-diferentiability of spiking neuron activation functions is circumvented via surrogate gradient techniques, with gradients approximated by BPTT.

(1) Fast Gradient Sign Method (FGSM) [52] is a single-step attack that perturbs the input along the sign of the loss gradient:

$$
\begin{array} { r } { \hat { x } = x + \epsilon \cdot \mathrm { s i g n } \big ( \nabla _ { x } \mathcal { L } ( f ( x ) , y ) \big ) , } \end{array}\tag{25}
$$

where  is the classification loss, � denotes the SNN model, and � is the ground-truth label.

(2) Projected Gradient Descent (PGD) [53] is an iterative extension of FGSM that starts from a random initialization within the �-ball and performs multi-step projected gradient updates:

$$
\hat { x } ^ { 0 } = x + \mathcal { V } ( - \epsilon , + \epsilon ) ,\tag{26}
$$

$$
\hat { x } ^ { k + 1 } = \Pi _ { \epsilon } \big \{ \hat { x } ^ { k } + \alpha \cdot \mathrm { s i g n } \big ( \nabla _ { \hat { x } ^ { k } } \mathcal { L } ( f ( \hat { x } ^ { k } ) , y ) \big ) \big \}\tag{27}
$$

where � is the step size and $\Pi _ { \epsilon }$ denotes the projection onto the $L _ { \infty }$ �-ball centered at �. We set $\alpha = 0 . 0 1$ with $k = 7$ iterations, with adversarial examples clipped to both the valid pixel range and the $L _ { \infty }$ ball.

(3) Basic Iterative Method (BIM) [54] is similar to PGD but starts from the original clean image rather than a random initialization:

$$
\hat { x } ^ { 0 } = x ,\tag{28}
$$

Classification accuracy (%) of all methods under 18 external environmental noise configurations. Boldface and underlining denote the best and second-best results in each row, respectively.
<table><tr><td rowspan="2">Noise Type</td><td rowspan="2">Noise Intensity</td><td rowspan="2">Dataset</td><td colspan="6">Accuracy (%)</td></tr><tr><td>w.o. IB</td><td>IB</td><td>NBP-SNN</td><td>SIB</td><td>QIB</td><td>SCIB</td></tr><tr><td rowspan="6">Salt-and-Pepper</td><td rowspan="3">NNI=0.1</td><td>Fashion-MNIST</td><td>92.96</td><td>93.09</td><td>92.85</td><td>92.94</td><td>93.08</td><td>93.28</td></tr><tr><td>CIFAR-10</td><td>88.38</td><td>89.07</td><td>88.84</td><td>88.75</td><td>89.33</td><td>89.99</td></tr><tr><td></td><td>CIFAR-100</td><td>61.52 62.32</td><td>62.84</td><td></td><td>63.27</td><td>62.92</td><td>62.91</td></tr><tr><td rowspan="3">NNI=0.3</td><td>Fashion-MNIST</td><td>92.39</td><td>92.50</td><td>92.65</td><td>92.47</td><td>92.48</td><td>92.73</td></tr><tr><td></td><td>CIFAR-10 85.59</td><td>86.61</td><td>86.01</td><td>86.88</td><td>86.31</td><td>87.13</td></tr><tr><td>CIFAR-100</td><td>55.90</td><td>57.08</td><td>56.92</td><td>57.49</td><td>56.61</td><td>58.40</td></tr><tr><td rowspan="6">Gaussian</td><td rowspan="3">NNI=0.25</td><td>Fashion-MNIST</td><td>91.26</td><td>92.67</td><td>91.71</td><td>92.69</td><td>92.68</td><td>92.77</td></tr><tr><td>CIFAR-10</td><td>88.23</td><td>88.52</td><td>87.52</td><td>88.38</td><td>88.40</td><td>88.44</td></tr><tr><td>CIFAR-100</td><td>61.19</td><td>62.28</td><td>61.89</td><td>61.79</td><td>62.51</td><td>62.89</td></tr><tr><td rowspan="3">NNI=1.0</td><td>Fashion-MNIST</td><td>87.98</td><td>88.13</td><td>87.77</td><td>89.34</td><td>89.28</td><td>89.86</td></tr><tr><td>CIFAR-10</td><td>78.48</td><td>81.44</td><td>80.38</td><td>81.21</td><td>81.36</td><td>81.78</td></tr><tr><td>CIFAR-100</td><td>50.21</td><td>50.72</td><td>50.23</td><td>50.65</td><td>51.19</td><td>51.32</td></tr><tr><td rowspan="5">Rayleigh</td><td rowspan="3">NNI=0.25</td><td>Fashion-MNIST</td><td>88.69</td><td>88.87</td><td>88.83</td><td>89.20</td><td>89.05</td><td>89.51</td></tr><tr><td>CIFAR-10</td><td>83.72</td><td>84.60</td><td>84.69</td><td>85.05</td><td>85.16</td><td>85.62</td></tr><tr><td>CIFAR-100</td><td>54.31</td><td>55.35</td><td>55.45</td><td>55.20</td><td>56.10</td><td>56.29</td></tr><tr><td rowspan="3">NNI=1.0</td><td>Fashion-MNIST</td><td>84.07</td><td>85.07</td><td>83.97</td><td>84.75</td><td>84.53</td><td>84.99</td></tr><tr><td>CIFAR-10</td><td>64.89</td><td>65.33</td><td>64.78</td><td>66.17</td><td>64.62</td><td>66.52</td></tr><tr><td>CIFAR-100</td><td>45.91</td><td>45.39</td><td>45.80</td><td>45.99</td><td>46.06</td><td>46.60</td></tr></table>

$$
\begin{array} { r } { \hat { x } ^ { k + 1 } = \mathrm { C l i p } _ { x , \epsilon } \left\{ \hat { x } ^ { k } + \alpha \cdot \mathrm { s i g n } \big ( \nabla _ { \hat { x } ^ { k } } \mathcal { L } ( f ( \hat { x } ^ { k } ) , y ) \big ) \right\} } \end{array}\tag{29}
$$

where $\mathrm { C l i p } _ { x , \epsilon }$ performs pixel-wise clipping to ensure the result always lies within the $L _ { \infty }$ �-neighborhood of the original image. The step size and number of iterations are the same as PGD.

Evaluation metric. Classification accuracy is used as the primary evaluation metric across all experiments. The energy consumption analysis additionally employs a theoretical energy model based on Synaptic Operations (SynOps).

## 5.2. External environmental noise robustness

To evaluate the interference resistance of SCIB under real-world physical environmental noise, we conduct systematic external noise robustness tests on Fashion-MNIST [49], CIFAR-10 [50], and CIFAR-100 [50], covering three representative noise distributions: Salt-and-Pepper, Gaussian, and Rayleigh. Each noise type is tested at two intensity levels. The classification accuracy of all methods across 18 noise configurations is reported in Table 1.

Salt-and-pepper noise. Under mild corruption (NNI = 0.1), the performance of all IB-based methods is comparable. On CIFAR-100, SCIB achieves 62.91%, slightly below SIB’s 63.27% and QIB’s 62.92%. When the corruption rate increases to the extreme level (NNI = 0.3), the performance gap widens: SCIB achieves the best results on all three datasets, reaching 58.40% on CIFAR-100 and surpassing the second-best SIB by 0.91 percentage points.

Gaussian noise. Under moderate perturbation (NNI = 0.25), SCIB performs comparably to all IB baselines, achieving 88.44% on CIFAR-10 versus IB’s 88.52%. Under stronger perturbation (NNI = 1.0), SCIB achieves the best results on all three datasets, leading the unconstrained baseline by approximately 3.3 percentage points on CIFAR-10 and 1.1 percentage points on CIFAR-100.

Rayleigh noise. SCIB achieves the best results across all four configurations on CIFAR-10 and CIFAR-100. Under extreme perturbation on CIFAR-100 (NNI = 1.0), IB drops 0.52 percentage points below the unconstrained baseline, whereas SCIB maintains the best accuracy at 46.60%. This contrast suggests that the class-conditional prior used by SCIB is more suitable than a label-agnostic standard Gaussian prior under asymmetric non-Gaussian perturbations.

## 5.3. Neuromorphic hardware noise robustness

When deploying SNNs on edge computing devices or mixed-signal neuromorphic chips, the non-ideal characteristics of physical devices introduce unavoidable low-level hardware noise. In this section, two representative internal perturbations are considered: synaptic weight noise and membrane potential noise, each tested at two intensity levels, NNI = 1 and NNI = 5. Figs. 4 and 5 present the accuracy convergence curves of all methods on Fashion-MNIST, CIFAR-10, and CIFAR-100 under membrane potential noise (MPN) and synaptic noise (SN), respectively. Solid and dashed lines correspond to NNI = 1 and NNI = 5. All models are trained for 200 epochs; since the accuracy stabilizes after approximately epoch 110, only the first 150 epochs are shown.

Membrane potential noise acts directly on the state of LIF neurons and perturbs spike firing timing at each timestep, making it more destructive than synaptic weight noise. As shown in Fig. 4, under moderate perturbation (NNI = 1), the performance of all IB-based methods is comparable, with SCIB essentially on par with the best baseline. For example, SCIB achieves 88.18% on CIFAR-10, close to IB’s 88.20%. Under extreme perturbation (NNI = 5), the performance gap becomes larger: on CIFAR-10, IB drops from 88.20% to 64.08%, while SCIB maintains 64.45%. On CIFAR-100, SCIB achieves the best results under both perturbation levels, with its lead becoming more evident as the noise intensity increases. This suggests that class-conditional compression helps preserve discriminative spike representations under noisy neuronal dynamics.

![](images/368fcb034e351b44fc5d1bd256209c284d7459671a7337e601a7355d431f0798.jpg)  
(a) Fashion-MNIST

![](images/0954fc24299ca557bc01db77b04c4faf384f1e780c9ccbee60cae81cd68e5cf2.jpg)  
(b) CIFAR-10

![](images/79b7dd61eeeff1f3c313c16d22f4d8a21d6c7ff9c5658d33b6e99e89a470c70f.jpg)  
(c) CIFAR-100

Figure 4: Training convergence curves of all methods under membrane potential noise (MPN). Solid and dashed lines correspond to NNI = 1 and NNI = 5, respectively.  
![](images/7661ff363fe687690c1186d03169f76a806708c13088281e45a94fd06b463dec.jpg)  
(a) Fashion-MNIST

![](images/797db1ee73644b923cb6a905945b862ee34113761fbb86d1613deb7f93d12426.jpg)  
(b) CIFAR-10

![](images/a49d8ca1b8018b934f0a4751deef993ef4aed77de4b2228d2c42bfe1c4fd0a42.jpg)  
(c) CIFAR-100  
Figure 5: Training convergence curves of all methods under synaptic weight noise (SN). Solid and dashed lines correspond to $\mathsf { N N I } = 1$ and NNI = 5, respectively.

Synaptic weight noise primarily afects the weight update and state readout processes of physical synapses. As shown in Fig. 5, SCIB achieves best or near-best results in most configurations. On CIFAR-100, the performance gap among QIB, SIB, and SCIB is small under moderate perturbation, whereas the diferences become more pronounced under extreme perturbation. When NNI increases to 5, SCIB achieves 62.98% on CIFAR-100, approximately 0.9 percentage points above the second-best method. These results indicate that the benefit of conditional compression becomes more apparent when high-intensity synaptic noise accumulates in deep spike representations.

## 5.4. Adversarial robustness

To evaluate SCIB under gradient-based adversarial perturbations, this section compares SCIB with Vanilla SNN and FEEL-SNN [21] on CIFAR-10 and CIFAR-100. The attack methods include FGSM [52], PGD [53], and BIM [54], with perturbation budgets $\epsilon \in \mathrm { ~  ~ { \left\{ ~ 4 / 2 5 5 , ~ 8 / 2 5 5 \right\} } ~ }$ and timesteps $T \ = \ 4$ and $T \ = \ B .$ . It should be particularly noted that SCIB is not explicitly optimized with adversarial examples during training; this section examines whether its conditional compression mechanism can provide implicit robustness gains in adversarial scenarios. The classification accuracy of all methods under diferent attack configurations is reported in Table 2.

Vanilla SNN without any defense mechanism collapses to near-zero accuracy under iterative PGD and BIM attacks, especially at $\epsilon = 8 / 2 5 5$ , confirming that standard SNN training provides limited inherent resistance to strong gradientbased adversarial perturbations. On CIFAR-10 with $T =$ 4 and $\epsilon \ : = \ : 4 / 2 5 5$ , SCIB improves over Vanilla SNN by 46.80 and 47.06 percentage points under PGD and BIM, respectively.

Compared with FEEL-SNN, SCIB shows stronger robustness under most iterative attacks. On CIFAR-10 with $T \ = \ 4$ , SCIB exceeds FEEL-SNN by 26.25 and 26.63 percentage points under PGD and BIM at $\epsilon = 4 / 2 5 5$ , and by 24.92 and 24.66 percentage points at $\epsilon ~ = ~ 8 / 2 5 5$ . A similar trend is observed at $T = 8 ,$ where SCIB consistently achieves the best adversarial accuracy on CIFAR-10 across all attack settings. On CIFAR-100, SCIB also outperforms FEEL-SNN under all PGD and BIM configurations, although FEEL-SNN is better under the single-step FGSM attack at � = 4 and $\epsilon = 4 / 2 5 5$

Table 2  
Classification accuracy (%) under adversarial attacks with perturbation budgets $\epsilon \in \{ 4 / 2 5 5 , 8 / 2 5 5 \}$ . SCIB does not employ adversarial training. Bold and underline denote the best and second-best results, respectively.
<table><tr><td rowspan="2">Timestep</td><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td rowspan="2">Clean</td><td colspan="3"> $\epsilon = 4 / 2 5 5$ </td><td colspan="3"> $\epsilon = 8 / 2 5 5$ </td></tr><tr><td>FGSM</td><td>PGD</td><td>BIM</td><td>FGSM</td><td>PGD</td><td>BIM</td></tr><tr><td rowspan="5">T = 4</td><td rowspan="2">CIFAR-10</td><td>Vanilla</td><td>92.94</td><td>19.63</td><td>0.78</td><td>0.67</td><td>8.29</td><td>0.02</td><td>0.01</td></tr><tr><td>FEEL-SNN</td><td>88.07</td><td>39.17</td><td>21.33</td><td>21.10</td><td>16.19</td><td>1.73</td><td>1.38</td></tr><tr><td rowspan="2"></td><td>SCIB</td><td>89.69</td><td>52.08</td><td>47.58</td><td>47.73</td><td>37.02</td><td>26.65</td><td>26.04</td></tr><tr><td>Vanilla</td><td>73.80</td><td>6.99</td><td>0.38</td><td>0.36</td><td>3.64</td><td>0.01</td><td>0.00</td></tr><tr><td rowspan="2">CIFAR-100</td><td>FEEL-SNN</td><td>63.08</td><td>19.51</td><td>9.08</td><td>8.74</td><td>8.27</td><td>0.94</td><td>0.76</td></tr><tr><td>SCIB</td><td>64.73</td><td>14.08</td><td>12.62</td><td>12.42</td><td>9.12</td><td>6.29</td><td>6.37</td></tr><tr><td rowspan="6">T = 8</td><td rowspan="3">CIFAR-10</td><td>Vanilla</td><td>93.53</td><td>19.05</td><td>0.56</td><td>0.52</td><td>7.24</td><td>0.02</td><td>0.00</td></tr><tr><td>FEEL-SNN</td><td>87.31</td><td>39.70</td><td>17.67</td><td>17.71</td><td>20.67</td><td>1.60</td><td>1.52</td></tr><tr><td>SCIB</td><td>87.73</td><td>51.03</td><td>44.88</td><td>45.10</td><td>26.74</td><td>16.09</td><td>15.24</td></tr><tr><td rowspan="3">CIFAR-100</td><td>Vanilla</td><td>74.33</td><td>7.43</td><td>0.47</td><td>0.47</td><td>3.88</td><td>0.02</td><td>0.00</td></tr><tr><td>FEEL-SNN</td><td>61.68</td><td>15.42</td><td>5.73</td><td>5.57</td><td>7.05</td><td>0.51</td><td>0.50</td></tr><tr><td>SCIB</td><td>61.53</td><td>25.39</td><td>21.94</td><td>21.65</td><td>11.91</td><td>7.55</td><td>7.18</td></tr></table>

These results suggest that SCIB can improve robustness under the evaluated gradient-based attacks, particularly iterative attacks. Since SCIB is not adversarially trained, these results should be interpreted as implicit robustness gains from representation-level regularization rather than evidence that SCIB dominates dedicated adversarial-training defenses.

## 5.5. Energy consumption

In the deployment ofedge computing devices and mixedsignal neuromorphic chips, computational complexity and dynamic power consumption are important metrics. SNNs rely on an event-driven mechanism, transmitting binary spikes to postsynaptic neurons only when the membrane potential exceeds the firing threshold. Therefore, physical energy consumption is directly related to the frequency of spike events.

To provide an algorithm-level proxy for inference eficiency, we construct a analytical computation-energy model based on Synaptic Operations (SynOps). For a hidden layer �, let the output feature map size be $H _ { l } \times W _ { l } \times C _ { l } ^ { \mathrm { o u t } }$ , the convolutional kernel size be $k _ { l } .$ , and the number of input channels be $C _ { l } ^ { \mathrm { i n } }$ . The fan-in of each neuron is $N _ { l } = k _ { l } ^ { 2 } \bar { C } _ { l } ^ { \mathrm { i n } }$ Defining the average total number of spike firings of a single neuron in layer � over the inference window as $f _ { l }$ , the total synaptic accumulation operations are

$$
\mathrm { A C } _ { l } = H _ { l } \cdot W _ { l } \cdot C _ { l } ^ { \mathrm { o u t } } \cdot N _ { l } \cdot f _ { l } .\tag{30}
$$

Following the commonly used 45-nm CMOS operationenergy estimates [55], a 32-bit floating-point MAC operation costs approximately $E _ { \mathrm { M A C } } = 4 . 6 \mathrm { p J }$ , while an AC operation costs $E _ { \mathrm { A C } } = 0 . 9 \mathrm { p J }$ . This analytical model accounts only for computation energy associated with MAC/AC operations.

$$
\begin{array} { r } { \mathbf { M A C } _ { 1 } = H _ { 1 } \cdot W _ { 1 } \cdot C _ { 1 } ^ { \mathrm { o u t } } \cdot N _ { 1 } \cdot T . } \end{array}\tag{31}
$$

Table 3  
Per-layer spike firing statistics and synaptic operations of SCIB on CIFAR-10. M denotes MAC operations; A denotes AC operations.
<table><tr><td>Layer</td><td>Neurons</td><td>Spikes</td><td>Rate</td><td>SynOps</td></tr><tr><td>Conv#1</td><td>65536</td><td>131859.7</td><td>2.0120</td><td> $\overline { { 1 . 8 \times 1 0 ^ { 6 } } }$  (M)</td></tr><tr><td>Conv#2</td><td>65536</td><td>142810.9</td><td>2.1791</td><td> $7 . 6 \times 1 0 ^ { 7 } ~ ( \mathsf { A } )$ </td></tr><tr><td>Conv#3</td><td>32768</td><td>74278.7</td><td>2.2668</td><td> $4 . 1 \times 1 0 ^ { 7 }$  (A)</td></tr><tr><td>Conv#4</td><td>32768</td><td>69054.9</td><td>2.1074</td><td> $8 . 6 \times 1 0 ^ { 7 }$  (A)</td></tr><tr><td>Conv#5</td><td>16384</td><td>37739.2</td><td>2.3034</td><td> $4 . 0 \times 1 0 ^ { 7 }$  (A)</td></tr><tr><td>Conv#6</td><td>16384</td><td>40166.4</td><td>2.4516</td><td> $8 . 7 \times 1 0 ^ { 7 }$  (A)</td></tr><tr><td>Conv#7</td><td>16384</td><td>43500.8</td><td>2.6551</td><td> $9 . 3 \times 1 0 ^ { 7 }$  (A)</td></tr><tr><td>FC1</td><td>1024</td><td>2367.6</td><td>2.3121</td><td> $1 . 1 \times 1 0 ^ { 7 } ~ ( \mathsf { A } )$ </td></tr></table>

All subsequent layers are driven by binary spikes and execute AC operations. The operation-level dynamic inferenceenergy proxy is estimated as

$$
E _ { \mathrm { t o t a l } } = E _ { \mathrm { M A C } } \cdot \mathbf { M A C _ { \mathrm { 1 } } } + E _ { \mathrm { A C } } \cdot \sum _ { l = 2 } ^ { L } \mathbf { A C } _ { l }\tag{32}
$$

From this model, SNN inference energy is linearly related to layer-wise firing rates. SCIB reduces spike activity by regularizing label-conditioned input-specific dependence, thereby lowering SynOps in deeper layers under this analytical model. Taking CIFAR-10 as an example, the perlayer spike firing statistics and synaptic operations of SCIB are reported in Table 3.

As shown in Table 3, except for Conv#1, all subsequent layers perform low-power AC operations. The average firing rates of deeper layers such as Conv#6 and Conv#7 are controlled at approximately 2.45 and 2.65, respectively.

Fig. 6 presents the average spike firing rate and perlayer energy consumption. Across all methods, Conv#1 has the same energy consumption because it directly processes analog inputs through dense MAC operations. From Conv#2 onward, the diferences among methods become evident. Compared with the baselines, SCIB maintains lower firing rates in most layers, especially in deeper convolutional layers.

![](images/ac46a6df44c73761ad0504e61dbf6cf46ec0a27f563bd22f74e4367815ec0fa5.jpg)  
Figure 6: Comparison of average spike firing rate and per-layer energy consumption during single-sample inference.

By accumulating layer-wise operation-energy estimates, the total SynOps-based single-sample inference-energy proxy is approximately $9 . 0 2 \times 1 0 ^ { 8 }$ pJ for ANN, $5 . 3 1 \times 1 0 ^ { 8 }$ pJ for w.o. $\mathrm { I B } , 6 . 1 1 \times 1 0 ^ { 8 }$ pJ for IB, 4.10×10<sup>8</sup> pJ for SIB, and 3.99× $1 0 ^ { 8 } \mathrm { p J }$ for SCIB. Overall, under this analytical SynOpsbased model, SCIB reduces the estimated operation-level inference energy by approximately 55.8% compared with ANN and 24.9% compared with w.o. IB.

## 5.6. Further analysis

## 5.6.1. Training evolution of the learnable class-conditional prior

This section further conducts a visualization analysis of the parameter evolution of the learnable class-conditional prior distribution proposed in Section 4.4, examining how the class-conditional regularizer shapes the learned prior structure in the representation space. Specifically, a VGG-9 model is trained on Fashion-MNIST, and the mean vectors and variance vectors of the 10 class prior distributions are extracted at four training stages—Epoch 0, 40, 80, and 120— and visualized from two complementary perspectives: the global topological structure of the latent space and the interclass statistical distances. Results are shown in Fig. 7.

Fig. 7(a) shows the spatial distribution of each class prior mean vector projected onto a two-dimensional plane via Principal Component Analysis (PCA), where colored dots mark the mean positions and 1� confidence ellipses represent the local coverage of each class prior. At the initial training stage (Epoch 0), the prior means of all classes are densely clustered within a small region near the origin (±0.5), with confidence ellipses heavily overlapping and classes nearly indistinguishable in the latent space. As training progresses, the class prior centers spread out in the latent space driven by the class-conditional KL regularization and the classification objective, while the confidence ellipses exhibit a pronounced shrinkage trend: by Epoch 80, the confidence ellipses appear largely separated in the PCA projection; by Epoch 120, the inter-center distances further increase and the ellipses become more compact, indicating that the prior parameters have largely converged.

Furthermore, the spatial topology learned by the prior distributions exhibits a clustering pattern highly consistent with class semantics. The top-wear categories in Fashion-MNIST—Class 0 (T-shirt), Class 2 (Pullover), Class 4 (Coat), and Class 6 (Shirt)—consistently maintain a nearestneighbor clustering relationship in the prior space; the footwear categories Class 5 (Sandal) and Class 7 (Sneaker) form a neighboring pair; Class 1 (Trouser), whose morphological features difer from the top-wear categories, remains relatively separated from those categories throughout training. This suggests that the learnable prior mechanism, together with the classification objective, encourages a classstructured latent topology that is partly consistent with semantic similarity between categories.

Fig. 7(b) provides a complementary quantitative view of the evolution process from the perspective of statistical distances. At the initial training stage (Epoch 0), all ofdiagonal elements of the KL divergence matrix are close to zero $( < ~ 0 . 5$ nats), indicating that the class priors are statistically nearly indistinguishable. As training iterates, the of-diagonal elements exhibit a generally increasing trend: the KL divergence between Class 1 (Trouser) and most other classes remains among the largest values across training stages, corroborating the spatial observations in the PCA visualization; the KL divergences between intra-cluster class pairs within the top-wear cluster are relatively low, quantitatively supporting the topological property that semantically similar classes tend to maintain nearest-neighbor relationships in the prior space.

Integrating both levels of analysis, the learnable classconditional prior mechanism exhibits a complete evolutionary trajectory during training, from random initialization to class-specific spatial partitioning: confidence ellipses progressively shrink from heavily overlapping to more compact regions; prior centers move from origin-clustered to more spatially separated configurations with increasing pairwise KL divergences; and visually similar classes tend to form neighboring groups in the prior space. This visualization provides empirical evidence that the conditional regularization term is associated with more class-structured latent priors, although the observed separation should not be interpreted as a theoretical guarantee.

## 5.6.2. Sensitivity analysis of compression strength hyperparameter �

In the optimization objective of the SCIB framework, the hyperparameter � determines the compression strength of the information bottleneck, controlling the trade-of between maintaining predictive performance and regularizing labelconditioned input-specific information. When $\beta$ approaches zero, the compression constraint relaxes and the model tends to retain more original input information to prioritize prediction accuracy; as � increases, the compression constraint strengthens and the model is forced to discard more inputspecific information—potentially achieving superior robustness in noisy environments, but also risking the loss of useful discriminative information due to over-compression. This section conducts a systematic sensitivity analysis of � on CIFAR-10 using the VGG-9 architecture.

![](images/e82c214588b01a041b45b66d43eaa75395709611d65f79047bf4d70e04161904.jpg)  
Epoch 0

![](images/abaa2b6511a2223d919619e5fba6c91cbd0971cdf315854b35b016563f279fde.jpg)  
Epoch 40

![](images/3d44931205f6c1efb51c08d7ff6531e2c37bc9f2adab97a75e175115df04bfa1.jpg)  
Epoch 80

![](images/f684326b2d9ff442dc156792461fab5731e35660bedaa04397a0774295649537.jpg)  
Epoch 120

(a) PCA projection evolution of class-conditional prior means $\mu _ { k }$  
![](images/8a970e0e2d85b96516f292cc2089d76d2e084826ea8e4f6c4da0308b10470c4e.jpg)  
Epoch 0

![](images/8548a0208e076d8a1b8ce217244e2257fb93f4c68e73c7f7acf639d53eeeabc4.jpg)  
Epoch 40

![](images/40345c9a33f8d1aa429a19e03002ce80c7862cb67a682a32a31032d74989bb3c.jpg)  
Epoch 80

![](images/8feb7ecccc5b24100b5f52484f79733b4b9a9cb108ba7fef55a4d8ca699e69b2.jpg)  
Epoch 120  
(b) Inter-class KL divergence heatmap evolution  
Figure 7: Training evolution of the class-conditional prior distribution �(�|�). Row (a): PCA projections of prior mean vectors; colored dots mark mean positions and 1� confidence ellipses indicate the local coverage of each class prior. Row (b): KL divergence heatmaps between class-conditional priors.

For the Gaussian noise experiments, NNI is set to 0.25, 0.50, 0.75, and 1.00; for the membrane potential noise experiments, NNI is set to 2, 4, 6, and 8. At each noise intensity level, � takes values of 0.2, 0.4, 0.6, 0.8, and 1.0, forming 4× 5 = 20 cross-experimental configurations. The experimental results are shown in Fig. 8.

Fig. 8(a) presents the �–NNI–accuracy response surface under Gaussian noise. The surface is relatively flat, indicating that model performance is not highly sensitive to � under input-level Gaussian noise: under low-noise conditions, the performance diferences across diferent � values are minimal; as noise intensity increases, smaller � values generally maintain higher accuracy, but moderately increasing $\beta$ at higher noise intensities can instead achieve a local optimum, suggesting that strengthening the compression constraint may help reduce perturbation-sensitive information when noise interference is stronger. Fig. 8(b) presents the response surface under membrane potential noise, where increasing � leads to a monotonically decreasing accuracy trend at all noise intensity levels, with this trend becoming more pronounced under high-noise conditions.

Synthesizing the observations across both noise paradigms, the optimal � is not fixed but requires adaptive adjustment according to the noise type and intensity: under Gaussian noise, � and performance exhibit a non-monotonic relationship, where moderate compression is beneficial under strong noise; under membrane potential noise, � and performance exhibit a monotonically decreasing relationship, and smaller compression strengths should be preferred. This analysis reveals the scenario-dependent nature of � selection in the SCIB framework, providing systematic experimental evidence for hyperparameter tuning under diferent noise environments.

## 6. Conclusion

Spiking Neural Networks (SNNs) provide an energyeficient computing paradigm for neuromorphic perception because of event-driven processing and sparse spike communication, but robustness remains a critical challenge. To address this issue, this paper proposes SCIB, a Spiking Conditional Information Bottleneck framework for robust and energy-eficient SNN representation learning. Motivated by the empirical observation that standardtrained SNNs may retain substantial or even redundant inputrelated information after their label-predictive information becomes stable, SCIB formulates robustness enhancement as a representation-level conditional compression problem. Unlike conventional IB-based methods that directly compress the mutual information �(�; �), SCIB adopts a variational upper bound of the conditional mutual information �(�; �|�) as the compression regularizer, thereby encouraging class-conditioned compactness while relying on the supervised objective to preserve label-relevant predictive information. To make this objective tractable and suitable for spike-based learning, we derive a variational upper bound for conditional compression and introduce a learnable class-conditional prior, which provides class-specific distributional anchors and improves the expressiveness of the variational prior compared with a label-agnostic standard Gaussian.

![](images/1c099cee200d5b62f5bf40bfdd33cb56532ca8e5c2febc82ed359bf60d8ad257.jpg)  
(a) Gaussian noise

![](images/305a39795651ce6a83aa0dc5fd990cefa0b2833dd86c7e27e79b99e4b6e6c6d0.jpg)  
(b) membrane potential noise  
Figure 8: Sensitivity analysis of the compression strength hyperparameter �.

Experiments demonstrate that SCIB improves robustness under external environmental noise, neuromorphic hardware noise, and adversarial attacks. Across most noise configurations, SCIB outperforms standard SNNs and existing IB-based baselines, with advantages that become more evident under stronger perturbations. In addition, the SynOps-based energy analysis shows that SCIB reduces redundant spike firing and lowers single-sample inference energy consumption by approximately 24.9% compared with the unconstrained SNN baseline, indicating that conditional compression regularization can jointly improve robustness and reduce operation-level SynOps energy estimates in the evaluated settings.

Several limitations remain for future investigation. First, SCIB does not explicitly optimize against adversarial examples, and thus its robustness under certain single-step attacks may still be limited. Second, the current class-conditional prior assigns an independent Gaussian component to each class, which may require more careful design for datasets with many fine-grained categories. Future work will explore more sample-eficient and hierarchical prior structures, investigate the complementary combination of SCIB with adversarial training, and validate the proposed framework on physical neuromorphic hardware.

## CRediT authorship contribution statement

Yuli Zha: Conceptualization, Methodology, Validation, Software, Investigation, Writing – Original draft. Wang Fan: Software, Validation. Yixiao Dai: Software, Data curation. Lingjie Wang: Methodology, Investigation. Bang Hu: Investigation, Validation. Wei Cao: Methodology, Writing – Review & Editing. Fengzhe Zhang: Conceptualization, Methodology. Fan Zhang: Conceptualization, Resources, Supervision, Writing – Review & Editing.

## References

[1] A. Mehonic and A. J. Kenyon. Brain-inspired computing needs a master plan. Nature, 604:255–260, 2022.

[2] Y. Zhang, P. Qu, Y. Ji, W. Zhang, G. Gao, G. Wang, and L. Shi. A system hierarchy for brain-inspired computing. Nature, 586:378–384, 2020.

[3] W. Maass. Networks of spiking neurons: the third generation of neural network models. Neural Netw., 10:1659–1671, 1997.

[4] K. Roy, A. Jaiswal, and P. Panda. Towards spike-based machine intelligence with neuromorphic computing. Nature, 575:607–617, 2019.

[5] M. Davies, N. Srinivasa, T. H. Lin, G. Chinya, Y. Cao, S. H. Choday, and H. Wang. Loihi: A neuromorphic manycore processor with onchip learning. IEEE Micro, 38:82–99, 2018.

[6] F. Akopyan, J. Sawada, A. Cassidy, R. Alvarez-Icaza, J. Arthur, P. Merolla, and D. S. Modha. Truenorth: Design and tool flow of a 65 mw 1 million neuron programmable neurosynaptic chip. IEEE Trans. Comput.-Aided Des. Integr. Circuits Syst., 34:1537–1557, 2015.

[7] L. Deng, G. Wang, G. Li, S. Li, L. Liang, M. Zhu, and L. Shi. Tianjic: A unified and scalable chip bridging spike-based and continuous neural computation. IEEE J. Solid-State Circuits, 55:2228–2246, 2020.

[8] M. Davies, A. Wild, G. Orchard, Y. Sandamirskaya, G. A. F. Guerra, P. Joshi, and S. R. Risbud. Advancing neuromorphic computing with loihi: A survey of results and outlook. Proc. IEEE, 109:911–934, 2021.

[9] C. D. Schuman, S. R. Kulkarni, M. Parsa, J. P. Mitchell, P. Date, and B. Kay. Opportunities for neuromorphic computing algorithms and applications. Nat. Comput. Sci., 2:10–19, 2022.

[10] Y. Wu, L. Deng, G. Li, J. Zhu, Y. Xie, and L. Shi. Direct training for spiking neural networks: Faster, larger, better. In In Proc. AAAI Conf. Artif. Intell., pages 1311–1318, 2019.

[11] E. O. Neftci, H. Mostafa, and F. Zenke. Surrogate gradient learning in spiking neural networks. IEEE Signal Process. Mag., 36:51–63, 2019.

[12] Y. Wu, L. Deng, G. Li, J. Zhu, and L. Shi. Spatio-temporal backpropagation for training high-performance spiking neural networks. Front. Neurosci., 12:331, 2018.

[13] S. Kim, S. Park, B. Na, and S. Yoon. Spiking-yolo: spiking neural network for energy-eficient object detection. In In Proc. AAAI Conf. Artif. Intell., pages 11270–11277, 2020.

[14] Y. Cao, Y. Chen, and D. Khosla. Spiking deep convolutional neural networks for energy-eficient object recognition. Int. J. Comput. Vis., 113:54–66, 2015.

[15] S. Sharmin, P. Panda, S. S. Sarwar, C. Lee, W. Ponghiran, and K. Roy. A comprehensive analysis on adversarial robustness of spiking neural networks. arXiv preprint arXiv:1905.02704, 2019.

[16] S. Yang, J. Wang, X. Hao, H. Li, X. Wei, B. Deng, and K. A. Loparo. Bicoss: Toward large-scale cognition brain with multigranular neuromorphic architecture. IEEE Trans. Neural Netw. Learn. Syst., 33: 2801–2815, 2021.

[17] L. Liang, X. Hu, L. Deng, Y. Wu, G. Li, Y. Ding, and Y. Xie. Exploring adversarial attack in spiking neural networks with spikecompatible gradient. IEEE Trans. Neural Netw. Learn. Syst., 34: 2569–2583, 2021.

[18] T. Bu, J. Ding, Z. Hao, and Z. Yu. Rate gradient approximation attack threats deep spiking neural networks. In In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 7896–7906, 2023.

[19] S. Sharmin, N. Rathi, P. Panda, and K. Roy. Inherent adversarial robustness of deep spiking neural networks: Efects of discrete input encoding and non-linear activations. In In Proc. Eur. Conf. Comput. Vis. (ECCV), pages 399–414, 2020.

[20] S. Kundu, M. Pedram, and P. A. Beerel. Hire-snn: Harnessing the inherent robustness of energy-eficient deep spiking neural networks by training with crafted input noise. In In Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), pages 5209–5218, 2021.

[21] M. Xu, D. Ma, H. Tang, Q. Zheng, and G. Pan. Feel-snn: Robust spiking neural networks with frequency encoding and evolutionary leak factor. Adv. Neural Inf. Process. Syst., 37:91930–91950, 2024.

[22] J. Ding, Z. Yu, T. Huang, and J. K. Liu. Enhancing the robustness of spiking neural networks with stochastic gating mechanisms. In In Proc. AAAI Conf. Artif. Intell., pages 492–502, 2024.

[23] J. Wang, D. Zhao, C. Du, X. He, Q. Zhang, and Y. Zeng. Random heterogeneous spiking neural network for adversarial defense. iScience, 28, 2025.

[24] Y. Garipova, S. Yonekura, and Y. Kuniyoshi. Noise and dynamical synapses as optimization tools for spiking neural networks. Entropy, 27:219, 2025.

[25] J. Ding, T. Bu, Z. Yu, T. Huang, and J. Liu. Snn-rat: Robustnessenhanced spiking neural network through regularized adversarial training. Adv. Neural Inf. Process. Syst., 35:24780–24793, 2022.

[26] H. Zhang, J. Cheng, J. Zhang, H. Liu, and Z. Wei. A regularization perspective based theoretical analysis for adversarial robustness of deep spiking neural networks. Neural Netw., 165:164–174, 2023.

[27] Y. Liu, T. Bu, J. Ding, Z. Hao, T. Huang, and Z. Yu. Enhancing adversarial robustness in snns with sparse gradients. arXiv preprint arXiv:2405.20355, 2024.

[28] R. Shwartz-Ziv and N. Tishby. Opening the black box of deep neural networks via information. arXiv preprint arXiv:1703.00810, 2017.

[29] B. C. Geiger. On information plane analyses of neural network classifiers—a review. IEEE Trans. Neural Netw. Learn. Syst., 33: 7039–7051, 2021.

[30] N. Tishby, F. C. Pereira, and W. Bialek. The information bottleneck method. arXiv preprint physics/0004057, 2000.

[31] N. Tishby and N. Zaslavsky. Deep learning and the information bottleneck principle. In In Proc. IEEE Inf. Theory Workshop (ITW), pages 1–5, 2015.

[32] A. A. Alemi, I. Fischer, J. V. Dillon, and K. Murphy. Deep variational information bottleneck. arXiv preprint arXiv:1612.00410, 2016.

[33] A. Kolchinsky, B. D. Tracey, and D. H. Wolpert. Nonlinear information bottleneck. Entropy, 21:1181, 2019.

[34] K. Kawaguchi, Z. Deng, X. Ji, and J. Huang. How does information bottleneck help deep learning? In In Proc. Int. Conf. Mach. Learn., pages 16049–16096, 2023.

[35] H. Kuang, H. Liu, Y. Wu, S. I. Satoh, and R. Ji. Improving adversarial robustness via information bottleneck distillation. Adv. Neural Inf. Process. Syst., 36:10796–10813, 2023.

[36] N. Z. Weingarten, Z. Yakhini, M. Butman, and R. Bustin. The supervised information bottleneck. Entropy, 27:452, 2025.

[37] F. Alesiani, S. Yu, and X. Yu. Gated information bottleneck for generalization in sequential environments. Knowl. Inf. Syst., 65:683– 705, 2023.

[38] I. Fischer. The conditional entropy bottleneck. Entropy, 22:999, 2020.

[39] H. Mostafa. Supervised learning based on temporal coding in spiking neural networks. IEEE Trans. Neural Netw. Learn. Syst., 29:3227– 3235, 2017.

[40] N. Rathi and K. Roy. Diet-snn: A low-latency spiking neural network with direct input encoding and leakage and threshold optimization. IEEE Trans. Neural Netw. Learn. Syst., 34:3174–3182, 2021.

[41] B. Han, G. Srinivasan, and K. Roy. Rmp-snn: Residual membrane potential neuron for enabling deeper high-accuracy and low-latency spiking neural network. In In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pages 13558–13567, 2020.

[42] Q. Zhang, M. Lu, S. Yu, J. Xin, and B. Chen. An information bottleneck approach for feature selection. Pattern Recognit., 164: 111564, 2025.

[43] S. Tang and S. Yu. Infodpcca: Information-theoretic dynamic probabilistic canonical correlation analysis. 2025.

[44] J. Li and D. Liu. Information bottleneck theory on convolutional neural networks. Neural Process. Lett., 53:1385–1400, 2021.

[45] I. Chelombiev, C. Houghton, and C. O’Donnell. Adaptive estimators show information compression in deep neural networks. arXiv preprint arXiv:1902.09037, 2019.

[46] S. Yang and B. Chen. Snib: improving spike-based machine learning using nonlinear information bottleneck. IEEE Trans. Syst. Man Cybern. Syst., 53:7852–7863, 2023.

[47] S. Yang, B. Linares-Barranco, Y. Wu, and B. Chen. Self-supervised high-order information bottleneck learning of spiking neural network for robust event-based optical flow estimation. IEEE Trans. Pattern Anal. Mach. Intell., 47:2280–2297, 2024.

[48] R. F. Nalewajski. Elements of information theory. In Perspectives in Electronic Structure Theory, pages 371–395. Springer Berlin Heidelberg, Berlin, Heidelberg, 2011.

[49] H. Xiao, K. Rasul, and R. Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.

[50] A. Krizhevsky and G. Hinton. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

[51] S. Guo and T. Lin. An eficient non-backpropagation method for training spiking neural networks. In In Proc. IEEE Int. Conf. Tools Artif. Intell. (ICTAI), pages 192–199, 2021.

[52] I. J. Goodfellow, J. Shlens, and C. Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.

[53] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu. Towards deep learning models resistant to adversarial attacks. In In Proc. Int. Conf. Learn. Represent. (ICLR), 2018.

[54] A. Kurakin, I. Goodfellow, and S. Bengio. Adversarial examples in the physical world. In In Proc. Int. Conf. Learn. Represent. Workshop, 2017.

[55] M. Horowitz. 1.1 computing’s energy problem (and what we can do about it). In In Proc. IEEE Int. Solid-State Circuits Conf. (ISSCC), pages 10–14, 2014.