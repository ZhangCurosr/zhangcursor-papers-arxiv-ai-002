# Revisiting Over-refusal in Safety-aligned Large Language Models: A Representation and Optimization Perspective

Chao Chen hfut.chenchao@gmail.com Hefei University of Technology Hefei, China

Shijie Hao hfut.hsj@gmail.com Hefei University of Technology Hefei, China

Pengyang Shao shaopymark@gmail.com National University of Singapore Singapore, Singapore

Fei Shen shenfei29@nus.edu.sg National University of Singapore Singapore, Singapore

Wei Qin qinwei.hfut@gmail.com Hefei University of Technology Hefei, China

Richang Hong hongrc.hfut@gmail.com Hefei University of Technology Hefei, China

## Abstract

Over-refusal is a critical side efect of safety alignment in large language models (LLMs), causing harmless but superficially risky queries to be mistakenly rejected. Despite the efectiveness of existing mitigation methods, why safety-aligned LLMs may still reject harmless queries under correct safety supervision remains insufficiently understood. To address this gap, we analyze over-refusal through the lens of representation and optimization. We make two observations: 1) the representations of harmful and over-refusal samples are visually distinguishable at certain intermediate layers but increasingly overlap in deeper layers; and 2) harmful-sample gradients exhibit high intra-class similarity and concentrate in a low-rank subspace. Moreover, they are more strongly aligned with refusal-target gradients than with correct helpful-target gradients on over-refusal prompts. This imbalanced gradient coupling can bias over-refusal samples toward refusal, while the reduced apparent representational separation in deeper layers may make it more dificult for the model to maintain a clear boundary between harmful and over-refusal samples. To this end, we propose GOAL (Gradient Orthogonalization with contrAstive Learning), a safety-alignment framework for mitigating over-refusal. Specifi cally, GOAL consists of two components: 1) to prevent imbalanced gradient coupling from biasing the optimization of over-refusal samples toward refusal, we orthogonalize over-refusal gradients against the low-rank refusal subspace extracted from harmful gradients; and 2) to explicitly enhance the separation between harmful and over-refusal representations, we introduce a contrastive learning objective. Extensive experiments across multiple models and benchmarks demonstrate that GOAL efectively mitigates over-refusal while preserving safety performance and general capabilities. For reproducibility, we have released the code for GOAL<sup>1</sup>.

## Keywords

Large Language Models, Safety Alignment, Over-refusal

## 1 Introduction

As Large Language Models (LLMs) are increasingly deployed in open-ended, user-facing applications, LLM safety has attracted growing attention [1, 3, 7, 32, 34, 45]. As a critical side efect of LLM safety, over-refusal, which refers to the mistaken rejection of harmless but superficially risky queries, has been widely studied [14, 23]. To clarify over-refusal, we define three types of samples [8, 14, 23]: harmful samples are genuinely unsafe queries that must be refused; benign samples are clearly harmless queries; and overrefusal samples are harmless but likely to be mistakenly rejected.

Existing eforts to mitigate over-refusal can be divided into two lines [8, 11, 14, 23, 40]. One line aims to reshape the model’s refusal boundary during optimization [14, 23]. For example, DCR introduces a contrastive-refinement stage before safety alignment, encouraging a sharper decision boundary between harmful samples and over-refusal samples [23]. The other line of work mitigates over-refusal through targeted interventions [8, 40]. For example, SCANS identifies safety-critical layers and steers their activations with refusal-related directions, suppressing over-refusal while preserving refusals on harmful inputs [8]. Despite the efectiveness of existing mitigation methods, why safety-aligned LLMs may still reject harmless queries under correct safety supervision remains insuficiently understood.

To address this gap, we analyze over-refusal through the lens of representation and optimization. We make two core observations. 1) Through layer-wise representation visualization, we observe that harmful and over-refusal samples are distinguishable at certain intermediate layers but show increasing overlap as layer depth increases. For example, as shown in Figure 1, at layer 13, these two types of samples are visually distinguishable in the representation space but increasingly overlap in deeper layers, suggesting a reduction in their apparent representational separation. Such reduced separation may make it more dificult for the model to maintain a clear boundary between harmful and over-refusal samples. 2) Through sample-level gradient statistics, as shown in Figure 2, we observe that harmful samples exhibit high intra-class gradient similarity during safety alignment, suggesting that their gradients may accumulate coherently and form a low-rank refusal subspace with highly concentrated gradient energy. In contrast, benign and over-refusal samples exhibit low intra-class similarity and more dispersed gradient energy. Moreover, benign sample gradients show low similarity to harmful sample gradients, whereas over-refusal gradients exhibit substantially higher similarity, potentially making the optimization of over-refusal samples more susceptible to the accumulated harmful gradient. Specifically, as shown in Figure 2(b), harmful sample gradients are more similar to over-refusal-ref gradients than to over-refusal gradients, where over-refusal-ref denotes the gradients induced by pairing over-refusal prompts with refusal targets. This observation suggests that the accumulated harmful gradient is more strongly coupled with the refusal-target gradient than with the correct helpful-target gradient. Such imbalanced gradient coupling may increase the likelihood that the model selects a refusal response during safety alignment. Together, these observations suggest that imbalanced gradient coupling can bias over-refusal samples toward refusal, while reduced apparent representational separation may make it more dificult for the model to maintain a clear boundary between harmful and over-refusal samples.

(b)  
Benign Over-refusal Harmful �: the �-th layer of the LLM  
![](images/ba13ea417c262fc08d859e9864fc43f65713a293f49ff80a08b148748060cafe.jpg)  
Figure 1: Layer-wise PCA visualization of last-token representations for benign, harmful, and over-refusal samples in Llama-2- 7B-Chat. Notably, harmful and over-refusal samples are visually distinguishable at layer 13 but show increasing overlap in deeper layers. Due to space limitations, representation visualization for an additional model is provided in Appendix E.

![](images/851c38a9cc3147d62afd6409c6234633b630525bfedce9efacb6faaae0e3e837.jpg)  
(a)

![](images/76461a3c1c0dec1e1b9e0cc1bacc74e9b1775c6f50ca01ca247c9c1fcbcfa9bd.jpg)  
Figure 2: Gradient analyses on Llama-2-7B-Chat. (a) Cumulative Explained Ratio (CER) of the top-� singular values for gradient subspaces of diferent sample types. (b) Pairwise gradient similarity across diferent sample types during safety alignment.

To this end, we propose GOAL (Gradient Orthogonalization with contrAstive Learning), a safety-alignment framework that utilizes gradient orthogonalization and contrastive learning to mitigate over-refusal. Specifically, GOAL fine-tunes the model on a safetyalignment dataset that includes harmful, benign, and over-refusal samples, while incorporating two complementary components during training. One component aims to mitigate imbalanced gradient coupling that biases the optimization of over-refusal samples toward refusal. Specifically, we orthogonalize over-refusal gradients against a low-rank refusal subspace extracted from harmful gradients. To obtain this subspace, we apply Singular Value Decomposition (SVD) to the per-sample gradients of harmful samples. The other component introduces a contrastive learning objective to explicitly enhance the separation between harmful and overrefusal representations. To balance computational eficiency and over-refusal mitigation, both components are applied by fine-tuning only the selected safety-relevant layers. Extensive experimental results on multiple benchmarks demonstrate the efectiveness of GOAL in mitigating over-refusal while maintaining safety performance and general capabilities. Our main contributions are summarized as follows:

• We investigate why safety-aligned LLMs still reject harmless overrefusal samples despite correct supervision, identifying increasing visual overlap between harmful and over-refusal representations in deeper layers and a low-rank harmful gradient subspace that aligns more strongly with refusal-target than helpful-target gradients on over-refusal prompts.

• We propose GOAL, a Gradient Orthogonalization with contrAstive Learning framework for mitigating over-refusal. GOAL first extracts a low-rank gradient-based refusal subspace from harmful samples via SVD and orthogonalizes over-refusal gradients against this subspace. It further introduces a contrastive learning objective to enhance the separation between harmful and over-refusal representations.

![](images/a6870f546b3e60652146f261c882270b511e40061c07282414ab4fbac09255f9.jpg)  
Figure 3: Overview of the proposed GOAL methodology.

• Extensive experiments show that GOAL mitigates over-refusal while maintaining safety performance and general capabilities, achieving relative gains of up to 14.03% in over-refusal mitigation and 1.20% in safety performance over the second-best methods.

## 2 Related Works

## 2.1 Over-refusal in Safety-aligned LLMs

As LLMs are increasingly deployed in open-ended and user-facing applications, safety alignment has received widespread attention from the research community [34, 46]. However, while efective in reducing harmful outputs, safety alignment often induces overrefusal, where aligned models reject user requests that appear potentially unsafe but are benign in nature [13, 18, 29]. To better characterize this failure mode, recent studies have introduced dedicated benchmarks that systematically evaluate false refusals across model families and prompt types [13, 29, 33, 41].

Existing eforts to mitigate over-refusal generally fall into two lines [11, 14, 23]. The first line aims to reshape the model’s refusal boundary during optimization [14, 23]. For example, DCR [23] in troduces a contrastive refinement stage before safety alignment to encourage a sharper decision boundary. The second line focuses on targeted interventions [8, 40]. For example, SCANS [8] steers activations in safety-critical layers using refusal steering vectors. Despite their efectiveness, existing methods provide limited understanding of why safety-aligned LLMs trained with correct safety labels may still reject harmless queries.

## 2.2 Decision Boundary Shifts in Optimization

Decision boundary shift under imbalanced data is an important problem in model optimization [16, 35, 43]. Related studies mainly examine such decision boundary shifts from three perspectives. The first investigates margin-based optimization, where explicit margin objectives or the implicit bias ofgradient-based training can enlarge inter-class separation and favor larger-margin solutions [28, 30, 36]. The second focuses on contrastive and metric learning, which reorganize the representation space by pulling positive samples together and pushing negative samples apart, thereby altering the relative geometry of neighboring categories [10, 21, 28]. The third analyzes biased or imbalanced training signals, particularly in long-tailed or imbalanced settings, where diferences in class frequency, classifier weights, instance dificulty, or the strength of optimization signals can systematically shift the learned boundary toward particular classes or regions of the representation space [16, 35, 43]. However, the decision boundary perspective has rarely been used to analyze over-refusal, even though over-refusal naturally matches a boundary shift pattern where harmless but safety-sensitive samples are moved toward the refusal side.

## 3 Methodology

In this section, we propose GOAL (Gradient Orthogonalization with contrAstive Learning), as illustrated in Figure 3. Specifically, GOAL consists of two complementary components: refusal-orthogonal projection in Section 3.1 and safety-aware contrastive learning in Section 3.2. Finally, Section 3.3 discusses GOAL.

## 3.1 Refusal-orthogonal Projection

To prevent imbalanced gradient coupling from attracting overrefusal samples toward the refusal side, we propose refusal-orthogonal projection, which orthogonalizes over-refusal gradients against the refusal subspace. Specifically, we first identify a low-rank refusal subspace from harmful sample gradients. We then explain why over-refusal samples are attracted to this subspace through imbalanced gradient coupling. Based on this analysis, we perform refusalorthogonal projection to remove the refusal-subspace components from over-refusal gradients, thereby decoupling their optimization from harmful gradients.

3.1.1 Identifying the Low-Rank Refusal Subspace. Let � denote the trainable parameters, and let $\mathcal { D } _ { \mathrm { h a r m f u l } } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ denote a set of harmful samples used for refusal subspace extraction, where $x _ { i }$ and $y _ { i }$ denote the prompt and the target response, respectively. A typical supervised fine-tuning (SFT) loss for safety alignment can be formulated as:

$$
\mathcal { L } _ { \mathrm { S F T } } ( \theta ) = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log p _ { \theta } ( y _ { i } \mid x _ { i } ) .\tag{1}
$$

However, the SFT loss itselfcannot prevent concentrated harmful gradients from biasing the safety-alignment boundary toward re fusal. To identify this gradient-based refusal subspace, we compute the per-sample SFT gradient with respect to the model parameters �. Specifically, given the per-sample SFT loss $\mathcal { L } _ { \mathrm { S F T } , i } ^ { \mathrm { h a r m f u l } } ( \theta , x _ { i } , y _ { i } ) =$ $- \log p _ { \theta } ( y _ { i } \mid x _ { i } )$ , the corresponding gradient is defined as:

$$
\begin{array} { r } { \mathbf { g } _ { i } ^ { \mathrm { h a r m f u l } } = \nabla _ { \theta } \mathcal { L } _ { \mathrm { S F T } , i } ^ { \mathrm { h a r m f u l } } ( \theta , x _ { i } , y _ { i } ) \in \mathbb { R } ^ { d } , } \end{array}\tag{2}
$$

where � is the gradient dimension. We then stack the gradients into a matrix

$$
\mathbf { G } = [ \mathbf { g } _ { 1 } ^ { \mathrm { h a r m f u l } } , \dots , \mathbf { g } _ { N } ^ { \mathrm { h a r m f u l } } ] ^ { \top } \in \mathbb { R } ^ { N \times d } .\tag{3}
$$

To extract the dominant refusal-gradient directions, we apply Singular Value Decomposition (SVD) to $\mathbf { G } \colon \mathbf { G } \ = \ \mathbf { U } \Sigma \mathbf { V } ^ { \intercal }$ , where $\Sigma = \mathrm { d i a g } ( \sigma _ { 1 } , . . . , \sigma _ { r } )$ contains the singular values in descending order, � = rank(G) is the number of non-zero singular values, and $\mathbf { U } \in \mathbb { R } ^ { N \times r }$ and $\mathbf { V } \in \mathbb { R } ^ { d \times r }$ denote the left and right singular vectors, respectively. The columns of V form an orthonormal basis for the row space of G.

From the singular-value spectrum in �, we observe that the refusal-gradient energy is concentrated in the top few directions, indicating a clear low-rank structure. To quantify this concentration, we compute the Cumulative Explained Ratio (CER) of the top-� singular values:

$$
\operatorname { C E R } ( k ) = \frac { \sum _ { j = 1 } ^ { k } \sigma _ { j } ^ { 2 } } { \sum _ { j = 1 } ^ { r } \sigma _ { j } ^ { 2 } } .\tag{4}
$$

We choose � such that the top-� directions capture the dominant refusal-gradient energy, and retain the corresponding top-� right singular vectors: $\mathbf { V } _ { k } = \mathbf { \bar { V } } [ : , : k ] \in \mathbb { R } ^ { d \times k }$ . We use $\mathbf { V } _ { k }$ as the orthonormal basis of the gradient-based refusal subspace, and define the corresponding orthogonal projection matrix as $\mathbf { P } = \mathbf { V } _ { k } \mathbf { V } _ { k } ^ { \top }$ . This subspace captures the dominant refusal directions while keeping the subsequent projection eficient.

3.1.2 Why Over-refusal Samples Are Atracted to the Refusal Subspace. As shown in Figure 2, harmful sample gradients exhibit high intra-class similarity, suggesting that they can accumulate coherently along a few dominant directions and form a low-rank refusal subspace. For an over-refusal sample, the accumulated harmful gradient is more strongly coupled with the gradient induced by the refusal-target than with that induced by the correct helpful target. Such imbalanced gradient coupling can increase the relative preference for the refusal response during safety alignment, poten tially biasing the over-refusal sample toward the refusal side. We formalize this efect as follows.

Proposition 1 (Refusal subspace attraction). Let $\bar { \bf g } _ { t }$ denote the aggregated harmful sample gradient at step �. For the �-th over-refusal sample, let $\mathcal { L } _ { \mathrm { S F T } , i } ^ { \mathrm { o v e r } }$ and $\mathcal { L } _ { \mathrm { S F T } , i } ^ { \prime 0 \mathrm { v e r } }$ denote its losses defined with respect to the correct helpful response �<sub>�</sub> and a refusal response $y _ { i } ^ { \prime } ,$ respectively, and let $\mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } = \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { S F T } , i } ^ { \mathrm { o v e r } } ( \boldsymbol { \theta } _ { t } , \boldsymbol { x } _ { i } , y _ { i } )$ and $\mathbf { g } _ { i , t } ^ { \prime \mathrm { o v e r } } = \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { S F T } , i } ^ { \prime \mathrm { o v e r } } ( \theta _ { t } , x _ { i } , y _ { i } ^ { \prime } )$ be their gradients at step �. Define the refusal preference margin as

$$
m _ { i } ( \theta ) = \log { p _ { \theta } ( y _ { i } ^ { \prime } \mid x _ { i } ) } - \log { p _ { \theta } ( y _ { i } \mid x _ { i } ) } .\tag{5}
$$

Define $A _ { i , t } = \left. \mathbf { P } \left( \mathbf { g } _ { i , t } ^ { \prime \mathrm { o v e r } } - \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } \right) , \mathbf { P } \bar { \mathbf { g } } _ { t } \right.$ as the refusal subspace contribution to the change in the refusal preference margin, and $R _ { i , t } =$ $\left. ( \mathbf { I } - \mathbf { P } ) \left( \mathbf { g } _ { i , t } ^ { \prime \mathrm { o v e r } } - \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } \right) , ( \mathbf { I } - \mathbf { P } ) \bar { \mathbf { g } } _ { t } \right.$ as the contribution ofthe residual components outside the refusal subspace. $I f A _ { i , t } > | R _ { i , t } | .$ , then, for a suficiently small learning rate �, the harmful gradient update increases the refusal preference margin:

$$
m _ { i } ( \theta _ { t } - \eta \bar { \bf g } _ { t } ) - m _ { i } ( \theta _ { t } ) = \eta \left( A _ { i , t } + R _ { i , t } \right) + O ( \eta ^ { 2 } ) > 0 .\tag{6}
$$

Remark. Proposition 1 formalizes the refusal subspace attraction efect in terms of the refusal preference margin. Specifically, $A _ { i , t } ~ > ~ 0$ means that, inside the refusal subspace, the aggregated harmful gradient is more strongly coupled with the refusal-target gradient than with the correct helpful-target gradient. Since P is extracted from the low-rank subspace of harmful sample gradients, the aggregated harmful gradient $\bar { \bf g } _ { t }$ is mainly contained in the refusal subspace and thus has only a small component outside it. Consequently, the residual term $R _ { i , t }$ is relatively small, making $A _ { i , t } > | R _ { i , t } |$ likely to hold. The harmful gradient update therefore increases the refusal preference margin of the over-refusal sample, progressively pulling it toward refusal behavior. The proof is provided in Appendix A.

3.1.3 Refusal-orthogonal Projection for Decoupled Optimization. Based on the above analysis, there are two natural alternative strategies for mitigating the refusal-subspace attraction efect. One strategy is to downweight harmful gradients, thereby reducing their contribution to the refusal-preference margin of over-refusal samples. However, harmful gradients provide necessary safety supervision, and directly weakening them may compromise the model’s ability to refuse genuinely harmful requests (Appendix R). Another strategy is to explicitly reduce the coupling between harmful gradients and the counterfactual refusal-target gradients of over-refusal samples. However, during safety alignment, over-refusal samples are paired with their correct helpful responses, and such counterfactual refusal-target gradients are not part of the standard training objective. We therefore adopt a more practical alternative: while fully preserving harmful gradients, we orthogonalize only the available helpful-target gradients of over-refusal samples against the refusal subspace, thereby preventing over-refusal optimization from evolving toward dominant harmful-refusal directions and being attracted to the refusal side. Moreover, our projected update yields a smaller first-order change in the refusal-preference margin than its unprojected counterpart (Appendix L).

We optimize GOAL on $\mathcal { D } _ { \mathrm { t r a i n } } = \mathcal { D } _ { \mathrm { h a r m f u l } } \cup \mathcal { D } _ { \mathrm { o v e r } } \cup \mathcal { D } _ { \mathrm { b e n i g n } }$ . Each training example is represented as a pair $( x _ { i } , y _ { i } )$ , where $x _ { i }$ is the prompt and $y _ { i }$ is the target response. At each step, we compute persample SFT gradients over a sampled minibatch $\mathcal { B } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { B } \colon$ g<sup>�</sup><sub>�</sub> = ∇<sub>�</sub> L<sup>�</sup><sub>SFT �</sub> (�, �<sub>�</sub>, �<sub>�</sub>), � ∈ {harmful, over-refusal, benign}.

(7)

For over-refusal samples, we orthogonalize their gradients against the refusal subspace, while leaving harmful and benign gradients unchanged:

$$
\tilde { \mathbf { g } } _ { i } = \left\{ \begin{array} { l l } { \mathbf { g } _ { i } ^ { c } - \mathbf { P g } _ { i } ^ { c } , } & { c = \mathrm { o v e r - r e f u s a l } , } \\ { \mathbf { g } _ { i } ^ { c } , } & { c \in \{ \mathrm { h a r m f u l } , \mathrm { b e n i g n } \} . } \end{array} \right.\tag{8}
$$

The projected supervised gradients are then aggregated as

$$
{ \bf g } _ { \mathrm { S F T } } ^ { \mathrm { t o t a l } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \tilde { \bf g } _ { i } .\tag{9}
$$

We formalize that the decoupled gradient can still efectively reduce the over-refusal loss as follows.

Proposition 2 (Decoupling by refusal-orthogonal projection). For the �-th over-refusal sample, let $\mathcal { L } _ { \mathrm { S F T } , i } ^ { \mathrm { o v e r } }$ denote its training loss, and let $\mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } = \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { S F T } , i } ^ { \mathrm { o v e r } } ( \boldsymbol { \theta } _ { t } , \boldsymbol { x } _ { i } , y _ { i } )$ be its gradient at step �. Let $\tilde { \bf g } _ { i , t } ^ { \mathrm { o v e r } } =$ $( \mathbf { I } - \dot { \mathbf { P } } ) \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } }$ be the refusal-orthogonal gradient used by GOAL. Then the projected over-refusal gradient satisfies $\left. \tilde { \bf g } _ { i , t } ^ { \mathrm { o v e r } } , { \bf P } \bar { \bf g } _ { t } \right. = 0 .$ , and its remaining overlap with the harmful gradient is given by $\left. \tilde { \bf g } _ { i , t } ^ { \mathrm { o v e r } } , \bar { \bf g } _ { t } \right. =$ $\left. ( \mathbf { I } - \mathbf { P } ) \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } , ( \mathbf { I } - \mathbf { P } ) \bar { \mathbf { g } } _ { t } \right.$ . Moreover, the projected gradient remains a first-order descent direction for the over-refusal training loss:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S F T } , i } ^ { \mathrm { o v e r } } ( \theta _ { t } - \eta \tilde { \mathbf { g } } _ { i , t } ^ { \mathrm { o v e r } } , x _ { i } , y _ { i } ) - \mathcal { L } _ { \mathrm { S F T } , i } ^ { \mathrm { o v e r } } ( \theta _ { t } , x _ { i } , y _ { i } ) = - \eta \left\| \left( \mathbf { I } - \mathbf { P } \right) \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } \right\| _ { 2 } ^ { 2 } + O ( \eta ^ { 2 } ) . } \end{array}\tag{10}
$$

Therefore, $i f \left\| ( \mathbf { I } - \mathbf { P } ) \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } \right\| _ { 2 } > 0 ;$ , then the projected update decreases the over-refusal loss to first order, while eliminating the refusal subspace component from the over-refusal update.

Remark. Proposition 2 explains why the proposed refusalorthogonal projection is reasonable. As shown in Figure 2, the over-refusal gradient is not entirely contained in the refusal subspace, implying that $\left\| \left( \mathbf { I } - \mathbf { P } \right) \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } \right\| _ { 2 } > 0 .$ Therefore, the remaining non-refusal component provides a first-order descent signal for the over-refusal training loss. Thus, GOAL decouples over-refusal optimization from the dominant refusal subspace without discarding useful task-learning signals. The proof is provided in Appendix B.

## 3.2 Safety-aware Contrastive Learning

Refusal-orthogonal projection decouples the optimization of overrefusal samples from the refusal subspace, preventing their updates from following dominant harmful-refusal directions. However, con trolling the update direction alone does not explicitly specify how harmful and over-refusal samples should be organized in the hiddenstate space. To enhance the separability of harmful and over-refusal representations in middle-to-late layers, we introduce a safety aware contrastive objective.

Specifically, we separate the hidden representations of harmful and over-refusal samples with a Circle-style margin contrastive loss [36]. Since the final hidden state directly governs token generation, we impose this contrastive constraint at the final layer, thereby obviating the need to enforce contrastive targets across all middle-to-late layers. For each harmful or over-refusal sample in the minibatch B, let $\mathbf { h } _ { i }$ denote its last-token hidden state at the final transformer layer. To construct stable and balanced contrastive pairs, we maintain a class-balanced representation memory bank M. For a given sample, its positive set $\mathcal { P } _ { i }$ and negative set ${ \cal N } _ { i }$ are constructed from $\bar { \boldsymbol { M } }$ using representations with the same or diferent labels, respectively. The similarity between $\mathbf { h } _ { i }$ and a memory-bank representation $\mathbf { z } _ { j }$ is measured by $s _ { i j } = \cos ( { \bf h } _ { i } , { \bf z } _ { j } )$ To make the contrastive objective focus more on less-optimized similarity pairs, we introduce adaptive pair weights following Circle Loss [36]. Given a relaxation margin � and a scale factor $\gamma ,$ the corresponding optimization targets and weighting ofsets are configured as $\Delta _ { \rho } = \bar { 1 } - m , \Delta _ { n } = m , { \bar { O } } _ { \rho } = 1 + m , { \bar { O } } _ { n } = { \bar { - } } m$ . Accordingly, the adaptive weights for positive and negative pairs are defined as $\alpha _ { \hat { p } } ( i , j ) \mathrm { ~ \bar { ~ } { = } ~ } [ O _ { \hat { p } } - \bar { s _ { i j } } ] _ { + } , \alpha _ { n } ( \bar { i } , j ) = [ s _ { i j } - O _ { n } ] _ { + }$ , where $[ \cdot ] _ { + } = \operatorname* { m a x } ( \cdot , 0 )$ Finally, the contrastive loss over all $N _ { \mathrm { c o n t } }$ harmful and over-refusal samples in B is formulated as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c o n t } } = \displaystyle \frac { 1 } { N _ { \mathrm { c o n t } } } \sum _ { i = 1 } ^ { N _ { \mathrm { c o n t } } } \log \left[ 1 + \Bigg ( \displaystyle \sum _ { j \in \mathcal { N } _ { i } } \exp \bigl ( \gamma \alpha _ { n } ( i , j ) \left( s _ { i j } - \Delta _ { n } \right) \bigr ) \Bigg ) \right. } \\ { \displaystyle \left. \cdot \left( \displaystyle \sum _ { j \in \mathcal { P } _ { i } } \exp \bigl ( - \gamma \alpha _ { j } \left( i , j \right) \left( s _ { i j } - \Delta _ { \mathcal { P } } \right) \bigr ) \right) \right] . } \end{array}\tag{11}
$$

Finally, GOAL combines the projected supervised gradient with a safety-aware contrastive learning gradient to form the overall update direction:

$$
\begin{array} { r } { \mathbf { g } _ { \mathrm { G O A L } } = \mathbf { g } _ { \mathrm { S F T } } ^ { \mathrm { t o t a l } } + \lambda \nabla _ { \theta } \mathcal { L } _ { \mathrm { c o n t } } , } \end{array}\tag{12}
$$

where � is a balancing hyperparameter.

## 3.3 Model Discussion

We further discuss GOAL. Specifically, we first explain why gradient orthogonalization is necessary beyond contrastive learning, then discuss which layers should be fine-tuned to mitigate over-refusal while improving computational eficiency, and finally analyze the time complexity of GOAL.

3.3.1 Why contrastive learning alone is insuficient. Contrastive learning encourages separation between harmful and over-refusal samples in the hidden-state space, but it does not directly control the SFT gradients that drive parameter updates. Since refusal responses are often highly repetitive, harmful samples can still induce concentrated harmful gradients and bias optimization toward refusal. Therefore, using contrastive learning alone would need to counteract this gradient-level bias only indirectly, which may require a large loss weight and risk harming utility or safety. In contrast, refusal-orthogonal projection directly removes refusal subspace components from over-refusal updates, while contrastive learning further improves representation-level separation. Thus, the two components address complementary failure modes: optimizationlevel coupling and reduced representational separation.

3.3.2 Choice ofSafety-relevant Layers. Our objective is to localize the layers responsible for constructing the model’s general safety boundary, as over-refusal can arise when safety-relevant layers misclassify harmless prompts as harmful. Inspired by prior findings [8, 14, 23], we identify safety-relevant layers using a perlayer separability probe based on hidden-state cosine similarity.

Table 1: Performance comparison of GOAL and baselines across over-refusal, safety, and general-capability benchmarks. Benchmark abbreviations: XST (XSTest), ORH (OR-Bench-Hard), PHT (PHTest), OKT (OKTest), CoCo (CoCoNot), MMLU, OBQA (OpenBookQA), PIQA, and ARC-C (ARC-Challenge). The best results within each model block are shown in bold, and the second-best results are underlined.
<table><tr><td rowspan="2">Method</td><td colspan="4">Over-refusal C.R. ↑</td><td colspan="3">Safety S.S. ↑</td><td colspan="4">General Capability Acc ↑</td></tr><tr><td>XST</td><td>ORH</td><td>PHT</td><td>OKT</td><td>CoCo</td><td>JBB</td><td>STL</td><td>MMLU</td><td>OBQA</td><td>PIQA</td><td>ARC-C</td></tr><tr><td colspan="10">Gemma-7B-it</td></tr><tr><td>Base</td><td>0.6977</td><td>0.6225</td><td>0.8596</td><td>0.6400</td><td>0.7652</td><td>0.90</td><td>0.9748</td><td>0.5063</td><td>0.368</td><td>0.7731</td><td>0.4778</td></tr><tr><td>SFT</td><td>0.8089</td><td>0.5808</td><td>0.8528</td><td>0.7467</td><td>0.8100</td><td>0.96</td><td>0.9767</td><td>0.5024</td><td>0.366</td><td>0.7693</td><td>0.4650</td></tr><tr><td>SCANS</td><td>0.5600</td><td>0.6998</td><td>0.8197</td><td>0.5633</td><td>0.8391</td><td>0.89</td><td>0.9612</td><td>0.5056</td><td>0.364</td><td>0.7737</td><td>0.4761</td></tr><tr><td>Surgical</td><td>0.7511</td><td>0.6745</td><td>0.8899</td><td>0.7800</td><td>0.7795</td><td>0.87</td><td>0.9399</td><td>0.5044</td><td>0.362</td><td>0.7644</td><td>0.4710</td></tr><tr><td>DCR</td><td>0.7200</td><td>0.6296</td><td>0.8699</td><td>0.8033</td><td>0.8496</td><td>0.89</td><td>0.9535</td><td>0.4997</td><td>0.348</td><td>0.7748</td><td>0.4582</td></tr><tr><td>ACTOR</td><td>0.7867</td><td>0.7066</td><td>0.8996</td><td>0.7800</td><td>0.8206</td><td>0.88</td><td>0.9593</td><td>0.5003</td><td>0.358</td><td>0.7704</td><td>0.4718</td></tr><tr><td>GOAL</td><td>0.9200</td><td>0.7006</td><td>0.9094</td><td>0.8733</td><td>0.8813</td><td>0.97</td><td>0.9884</td><td>0.5059</td><td>0.368</td><td>0.7753</td><td>0.4727</td></tr><tr><td colspan="10">Llama-2-7B-Chat</td></tr><tr><td>Base</td><td>0.8000</td><td>0.2648</td><td>0.5502</td><td>0.7600</td><td>0.7203</td><td>0.96</td><td>0.9981</td><td>0.4681</td><td>0.320</td><td>0.7628</td><td>0.4206</td></tr><tr><td>SFT</td><td>0.8978</td><td>0.6604</td><td>0.9098</td><td>0.9233</td><td>0.8813</td><td>0.96</td><td>0.9787</td><td>0.4698</td><td>0.318</td><td>0.7633</td><td>0.4019</td></tr><tr><td>SCANS</td><td>0.9467</td><td>0.4050</td><td>0.9001</td><td>0.9767</td><td>0.8790</td><td>0.94</td><td>0.9632</td><td>0.4679</td><td>0.326</td><td>0.7639</td><td>0.4138</td></tr><tr><td>Surgical</td><td>0.8889</td><td>0.6793</td><td>0.9201</td><td>0.9033</td><td>0.8100</td><td>0.95</td><td>0.9767</td><td>0.4651</td><td>0.318</td><td>0.7595</td><td>0.4113</td></tr><tr><td>DCR</td><td>0.9067</td><td>0.7297</td><td>0.8996</td><td>0.9200</td><td>0.8602</td><td>0.94</td><td>0.9651</td><td>0.4413</td><td>0.302</td><td>0.7236</td><td>0.4027</td></tr><tr><td>ACTOR</td><td>0.9333</td><td>0.7494</td><td>0.9498</td><td>0.9333</td><td>0.8522</td><td>0.95</td><td>0.9767</td><td>0.4626</td><td>0.316</td><td>0.7524</td><td>0.4078</td></tr><tr><td>GOAL</td><td>0.9511</td><td>0.7699</td><td>0.9498</td><td>0.9700</td><td>0.9024</td><td>0.96</td><td>0.9864</td><td>0.4682</td><td>0.318</td><td>0.7557</td><td>0.4138</td></tr><tr><td colspan="10">Llama-2-13B-Chat</td></tr><tr><td></td><td></td><td>0.2002</td><td>0.6501</td><td>0.7200</td><td>0.6807</td><td>0.96</td><td>0.9981</td><td>0.5366</td><td>0.354</td><td>0.7791</td><td>0.4633</td></tr><tr><td>Base SFT</td><td>0.7511 0.8978</td><td>0.5902</td><td>0.9201</td><td>0.9400</td><td>0.8496</td><td>0.97</td><td>0.9981</td><td>0.5359</td><td>0.376</td><td>0.7791</td><td>0.4693</td></tr><tr><td>SCANS</td><td>0.9022</td><td>0.3499</td><td>0.8504</td><td>0.9367</td><td>0.8496</td><td>0.98</td><td>0.9981</td><td>0.5351</td><td>0.356</td><td>0.7835</td><td>0.4718</td></tr><tr><td>Surgical</td><td>0.8178</td><td>0.6004</td><td>0.8801</td><td>0.8733</td><td>0.7784</td><td>0.95</td><td>0.9942</td><td>0.5348</td><td>0.348</td><td>0.7753</td><td>0.4608</td></tr><tr><td>DCR</td><td>0.8444</td><td>0.6194</td><td>0.8796</td><td>0.9067</td><td>0.8206</td><td>0.94</td><td>0.9787</td><td>0.5144</td><td>0.308</td><td>0.7378</td><td>0.4266</td></tr><tr><td>ACTOR</td><td>0.8667</td><td>0.6399</td><td>0.9503</td><td>0.9033</td><td>0.8179</td><td>0.95</td><td>0.9961</td><td>0.5316</td><td>0.352</td><td>0.7622</td><td>0.4522</td></tr><tr><td>GOAL</td><td>0.9289</td><td>0.7297</td><td>0.9649</td><td>0.9700</td><td>0.8892</td><td>0.97</td><td>0.9981</td><td>0.5361</td><td>0.382</td><td>0.7758</td><td>0.4659</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Specifically, at each Transformer layer, we compute the within class similarities of benign–benign and harmful–harmful prompt pairs, as well as the between-class similarity of benign–harmful prompt pairs. We then define a separation gap to quantify the margin between within-class and between-class similarities. This gap widens rapidly across layers because the between-class similarity of benign–harmful pairs drops at a much faster rate than the within-class similarities. We select the contiguous layer block where this gap grows most rapidly as the safety-relevant layers. Notably, the layer where harmful and over-refusal representations are separable in our earlier analysis (Section 1) lies near the earlier portion of the identified safety-relevant layers. This pattern suggests that the model initially retains information distinguishing over-refusal samples from harmful ones, but becomes increasingly conservative as safety-relevant representations propagate through deeper layers. Therefore, to improve computational eficiency while mitigating over-refusal, we restrict LoRA fine-tuning to the entire safety-relevant layer block, rather than fine-tuning only the later layers, where our visualization indicates increased overlap between harmful and over-refusal representations. This encourages a coherent safety boundary throughout the layer-wise safety decision process. The refusal-orthogonal projection is applied to the gradi ents of the trainable parameters in these layers. Meanwhile, the contrastive objective is computed using the final-layer hidden representations, with its gradients backpropagated solely to the trainable LoRA parameters in the selected safety-relevant layers. We defer the full details to Appendix C.

3.3.3 Time Complexity Analysis. Finally, we analyze the computational cost of GOAL. Let � denote the number of trainable parameters in the selected layers, � the number of harmful samples used for subspace extraction, � the number of retained refusalbasis directions, and $d _ { h }$ the hidden-state dimension. GOAL first performs � backward passes to collect the harmful gradient matrix $\mathbf { G } \in \mathbb { R } ^ { N \times d }$ , and computes the top-� singular vectors with a truncated SVD, which costs �(���). During training, the low-rank projection costs $O ( B _ { o } k d )$ for $B _ { o }$ over-refusal samples in a minibatch, while the contrastive memory-bank comparison costs $O ( B _ { c } | M | d _ { h } )$ for $B _ { c }$ contrastive queries. Since � and |M| are small, the extra overhead is modest in practice.

## 4 Experiments

In this section, we try to answer the following research questions:

• RQ1: How does GOAL perform compared with existing baselines across over-refusal, safety, and general-capability benchmarks (Section 4.2)?

• RQ2: How does each component of GOAL contribute to the final performance (Section 4.3)?

• RQ3: Do the selected safety-relevant layers provide an efective target for over-refusal mitigation (Section 4.4)?

• RQ4: How robust is the extracted low-rank refusal subspace across diferent extraction settings (Section 4.5)?

• RQ5: How sensitive is GOAL to key hyperparameters, including the projection rank � and contrastive loss weight � (Section 4.6)?

## 4.1 Experimental Settings

Training Data. Our 500-sample training set comprises 200 harmful queries (AdvBench [47], MaliciousInstruct [19], TDC2023 [24, 25]), 200 benign queries (Alpaca [37]), and 100 over-refusal queries (OR-Bench-Hard [13], XSTest [29], PHTest [2]). Reference responses are generated via GPT-4o [20].

Evaluation and Metrics. We evaluate over-refusal via Compliance Rate (C.R.) on the held-out test portions of the aforementioned over-refusal datasets, alongside two OOD benchmarks: OKTest [33] and CoCoNot [6]. Safety preservation is measured via Safety Score (S.S.) on JailbreakBench (JBB) [9] and STL, our shorthand for a collection of four safety benchmarks from Bianchi et al. [4]. General capability is assessed via Accuracy (Acc) on standard multiplechoice benchmarks (MMLU [17], ARC-C [12], OpenBookQA [26], PIQA [5]). GPT-5.1 is employed as the judge for evaluating C.R. and S.S. Detailed dataset descriptions, baseline information, metric definitions, training hyperparameters, and implementation details are deferred to Appendix D.

Models and Baselines. We evaluate GOAL on Gemma-7B-it, Llama-2-7B-Chat, and Llama-2-13B-Chat [38, 39] using greedy decoding. We compare against standard SFT and representative overrefusal mitigation methods, including training-based methods (AC-TOR [14], DCR [23]) and inference-time or representation-level interventions (SCANS [8], Surgical [40]). For a fair comparison, we retrain ACTOR, DCR, and SFT using the same over-refusal data construction strategy as GOAL.

Implementation Details. We fine-tune all models with LoRA on the selected safety-relevant layers only, using rank � = 16, scaling factor � = 32, and dropout rate 0.05. The selected layer ranges are layers 12–22 for Llama-2-7B-Chat, layers 13–26 for Llama-2-13B-Chat, and layers 15–23 for Gemma-7B-it. All models are trained for 5 epochs with an efective batch size of 4. We set the refusalsubspace rank to � = 10, apply the contrastive loss at the last layer, and use a contrastive weight of � = 0.1. The Circle-style contrastive loss uses margin � = 0.25 and scale � = 80, with a class-balanced FIFO memory bank of size 128. More training and implementation details are provided in Appendix D.

## 4.2 Overall Performance (RQ1)

Table 1 reports the performance of GOAL and the baselines in terms of over-refusal mitigation, safety preservation, and general capability retention across three backbone models. Overall, GOAL achieves leading over-refusal performance while maintaining strong safety and preserving general capabilities. We analyze these three aspects below.

Efective mitigation of over-refusal. GOAL consistently attains the highest or second-highest Compliance Rate (C.R.) across all over-refusal benchmarks. Specifically, it achieves the best or tiedbest performance on 13 out of 15 over-refusal settings and ranks second on the remaining two. The gains are especially pronounced on challenging OOD benchmarks such as OKTest and CoCoNot. For example, on Llama-2-13B-Chat, GOAL improves OKTest from 0.9400 to 0.9700 and CoCoNot from 0.8496 to 0.8892 compared with the strongest non-GOAL results. These results suggest that GOAL generalizes its mitigation behavior beyond in-domain overrefusal prompts, rather than merely fitting to a specific benchmark distribution.

Robust safety preservation. A common failure mode in mitigating over-refusal is the inadvertent degradation of model safety. In contrast, GOAL maintains strong safety performance on both JBB and STL. On Gemma-7B-it, GOAL achieves the best safety scores on both JBB and STL, improving JBB from 0.90 to 0.97 and STL from 0.9748 to 0.9884 compared with the base model. On Llama-2- 13B-Chat, GOAL matches the best STL score of 0.9981 while also achieving a competitive JBB score of 0.97. These results show that the improved compliance on over-refusal queries does not come at the cost of substantially weakened safety behavior against genuinely harmful instructions.

Maintaining general capabilities. Across all three backbones and four general-capability benchmarks, GOAL largely preserves the performance of the corresponding base models. The largest observed decrease is only 0.0071 in absolute accuracy. These results indicate that GOAL mitigates over-refusal without noticeably compromising general capabilities.

Further statistical significance analysis and human evaluation are provided in Appendices P and O, respectively.

## 4.3 Ablation Study (RQ2)

Ablation setup. To isolate the contribution of each component, we compare three variants: (i) Ortho only, retaining only the gradient orthogonalization against the refusal subspace; (ii) Contrast only, retaining only the contrastive learning; and (iii) GOAL, the full model. All variants are trained on the three base models with identical data and hyperparameters.

Component contributions. As demonstrated in Table 2, the full GOAL framework consistently achieves the highest C.R. on every over-refusal benchmark across all three model blocks. Compared with either orthogonal projection alone or contrastive learning alone, their combination yields more robust mitigation of overrefusal, with clear improvements on XSTest, OR-Bench-Hard, PHTest, OKTest, and CoCoNot. Meanwhile, GOAL also maintains competitive safety scores and general-capability performance. These results indicate that the two mechanisms are complementary.

## 4.4 Efectiveness of Safety-relevant Layers (RQ3)

To evaluate the efectiveness of our choice of safety-relevant layers, we compare our probe-selected middle-layer block against early-, late-, and full-layer variants on Llama-2-7B-Chat in Table 3. The middle-layer configuration consistently outperforms both the early and late variants across all over-refusal benchmarks. Furthermore, it achieves competitive performance against the full-layer configuration while updating only roughly one-third of the parameters.

Table 2: Ablation of GOAL’s components on over-refusal, safety, and general-capability benchmarks across three base models.
<table><tr><td rowspan="2">Method</td><td colspan="5">Over-refusal C.R. ↑</td><td colspan="2">Safety S.S. ↑</td><td colspan="4">General Capability Acc ↑</td></tr><tr><td>XST</td><td>ORH</td><td>PHT</td><td>OKT</td><td>CoCo</td><td>JBB</td><td>STL</td><td>MMLU</td><td>OBQA</td><td>PIQA</td><td>ARC-C</td></tr><tr><td colspan="10">Gemma-7B-it</td></tr><tr><td>Ortho only</td><td>0.8978</td><td>0.6596</td><td>0.8699</td><td>0.8533</td><td>0.8602</td><td>0.96</td><td>0.9845</td><td>0.5092</td><td>0.3720</td><td>0.7666</td><td>0.4710</td></tr><tr><td>Contrast only</td><td>0.8489</td><td>0.6399</td><td>0.8801</td><td>0.8200</td><td>0.8391</td><td>0.95</td><td>0.9787</td><td>0.4979</td><td>0.3520</td><td>0.7742</td><td>0.4599</td></tr><tr><td>GOAL</td><td>0.9200</td><td>0.7006</td><td>0.9094</td><td>0.8733</td><td>0.8813</td><td>0.97</td><td>0.9884</td><td>0.5059</td><td>0.3680</td><td>0.7753</td><td>0.4727</td></tr><tr><td colspan="10">Llama-2-7B-Chat</td></tr><tr><td>Ortho only</td><td>0.9200</td><td>0.7195</td><td>0.9001</td><td>0.9400</td><td>0.8892</td><td>0.95</td><td>0.9806</td><td>0.4717</td><td>0.3160</td><td>0.7492</td><td>0.4024</td></tr><tr><td>Contrast only</td><td>0.9067</td><td>0.6895</td><td>0.8996</td><td>0.9367</td><td>0.8839</td><td>0.96</td><td>0.9845</td><td>0.4705</td><td>0.3040</td><td>0.7612</td><td>0.4147</td></tr><tr><td>GOAL</td><td>0.9511</td><td>0.7699</td><td>0.9498</td><td>0.9700</td><td>0.9024</td><td>0.96</td><td>0.9864</td><td>0.4682</td><td>0.3180</td><td>0.7557</td><td>0.4138</td></tr><tr><td colspan="10">Llama-2-13B-Chat</td></tr><tr><td>Ortho only</td><td>0.9111</td><td>0.6998</td><td>0.9298</td><td>0.9400</td><td>0.8602</td><td>0.97</td><td>0.9884</td><td>0.5364</td><td>0.3860</td><td>0.7775</td><td>0.4514</td></tr><tr><td>Contrast only</td><td>0.8978</td><td>0.6596</td><td>0.9201</td><td>0.9200</td><td>0.8575</td><td>0.96</td><td>0.9903</td><td>0.5351</td><td>0.3660</td><td>0.7764</td><td>0.4565</td></tr><tr><td>GOAL</td><td>0.9289</td><td>0.7297</td><td>0.9649</td><td>0.9700</td><td>0.8892</td><td>0.97</td><td>0.9981</td><td>0.5361</td><td>0.3820</td><td>0.7758</td><td>0.4659</td></tr></table>

Table 3: Ablation of target layer selection on Llama-2-7B-Chat (32 transformer layers).
<table><tr><td rowspan="2">Configuration</td><td colspan="5">Over-refusal C.R. ↑</td><td colspan="2">Safety S.S. ↑</td><td colspan="4">General Capability Acc ↑</td></tr><tr><td>XST</td><td>ORH</td><td>PHT</td><td>OKT</td><td>CoCo</td><td>JBB</td><td>STL</td><td>MMLU</td><td>OBQA</td><td>PIQA</td><td>ARC-C</td></tr><tr><td>Early (layers 1–11)</td><td>0.8267</td><td>0.6194</td><td>0.9196</td><td>0.9000</td><td>0.7995</td><td>0.96</td><td>0.9923</td><td>0.4657</td><td>0.3120</td><td>0.7519</td><td>0.4053</td></tr><tr><td>Late (layers 23–32)</td><td>0.8089</td><td>0.5595</td><td>0.8996</td><td>0.8133</td><td>0.7493</td><td>0.97</td><td>0.9826</td><td>0.4679</td><td>0.3060</td><td>0.7622</td><td>0.4121</td></tr><tr><td>Full (layers 1–32)</td><td>0.9556</td><td>0.7699</td><td>0.9552</td><td>0.9733</td><td>0.8892</td><td>0.97</td><td>0.9942</td><td>0.4675</td><td>0.3180</td><td>0.7595</td><td>0.4104</td></tr><tr><td>GOAL (layers 12–22)</td><td>0.9511</td><td>0.7699</td><td>0.9498</td><td>0.9700</td><td>0.9024</td><td>0.96</td><td>0.9864</td><td>0.4682</td><td>0.3180</td><td>0.7557</td><td>0.4138</td></tr></table>

![](images/56152d1c009dd267f9b9d84d8527ef8204c25c2b39ed9fee9724d583812b00cd.jpg)  
Figure 4: Robustness to harmful sample size �. Impact of the number of harmful prompts � used for subspace extraction on model performance.

These results suggest that the selected middle-layer block serves as a parameter-eficient target for GOAL. Further analysis of the efectiveness of safety-relevant layers is provided in Appendix J.

## 4.5 Robustness of the Refusal Subspace (RQ4)

We evaluate the sensitivity of the extracted refusal subspace to the number of harmful prompts (�) used for its construction on Llama-2-7B-Chat. As shown in Figure 4, the C.R. and S.S. metrics remain stable across varied sample sizes. This demonstrates that the extracted subspace captures an intrinsic, stable refusal structure rather than sample-specific artifacts. More comprehensive experimental results and the robustness evaluation of the refusal subspace across diferent harmful datasets are provided in Appendix F.

![](images/67758fb5e203a8519e69df9bb379acd57ddc89ffddd40b658525e735aab37809.jpg)

![](images/f3970c259c57770bd24e9a2c98ccbb87724eaacdd33d82d3961b5224060d187f.jpg)  
Figure 5: Sensitivity heatmaps of projection rank � and contrastive weight �.

## 4.6 Hyperparameter Analysis (RQ5)

We analyze the sensitivity of GOAL to its two primary hyperparameters: the projection rank � and the contrastive loss weight �. Figure 5 presents heatmaps of OKTest C.R. and STL S.S. over a two-dimensional hyperparameter sweep on Llama-2-7B-Chat. The results show that GOAL remains stable across a broad range of hyperparameter settings. In particular, strong performance is consistently observed around the default setting (�=10, �=0.1), while nearby settings achieve comparable results on both C.R. and S.S. metrics. This stability reduces the need for intensive per-model tuning and supports the practical applicability of GOAL. More comprehensive experimental results are provided in Appendix F.

## 5 Conclusion

In this work, we analyze over-refusal from both representational and optimization perspectives. Our layer-wise visualization shows that harmful and over-refusal representations are separable at certain intermediate layers but become increasingly entangled in deeper layers. We further find that harmful sample gradients exhibit high intra-class similarity and concentrate in a low-rank subspace, while exhibiting stronger alignment with refusal-target gradients than with helpful-target gradients on over-refusal prompts. Based on these observations, we propose GOAL, which combines refusal orthogonal projection to decouple over-refusal optimization from harmful gradients with safety-aware contrastive learning to enhance the separation between harmful and over-refusal representations. Extensive experiments across multiple models and benchmarks demonstrate that GOAL substantially mitigates over-refusal while preserving model safety and general capabilities.

## References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023).

[2] Bang An, Sicheng Zhu, Ruiyi Zhang, Michael-Andrei Panaitescu-Liess, Yuancheng Xu, and Furong Huang. 2024. Automatic pseudo-harmful prompt generation for evaluating false refusals in large language models. arXiv preprint arXiv:2409.00598 (2024).

[3] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. 2022. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862 (2022).

[4] Federico Bianchi, Mirac Suzgun, Giuseppe Attanasio, Paul Röttger, Dan Jurafsky, Tatsunori Hashimoto, and James Y Zou. 2024. Safety-tuned llamas: Lessons from improving the safety of large language models that follow instructions. In International Conference on Learning Representations, Vol. 2024. 34196–34216.

[5] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. 2020. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 34. 7432–7439.

[6] Faeze Brahman, Sachin Kumar, Vidhisha Balachandran, Pradeep Dasigi, Valentina Pyatkin, Abhilasha Ravichander, Sarah Wiegrefe, Nouha Dziri, Khyathi Chandu, Jack Hessel, et al. 2024. The art of saying no: Contextual noncompliance in language models. Advances in Neural Information Processing Systems 37 (2024), 49706–49748.

[7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in Neural Information Processing Systems 33 (2020), 1877–1901.

[8] Zouying Cao, Yifei Yang, and Hai Zhao. 2025. Scans: Mitigating the exaggerated safety for llms via safety-conscious activation steering. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 39. 23523–23531.

[9] Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J Pappas, Florian Tramer, et al. 2024. Jailbreakbench: An open robustness bench mark for jailbreaking large language models. Advances in Neural Information Processing Systems 37 (2024), 55005–55029.

[10] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geofrey Hinton. 2020. A simple framework for contrastive learning of visual representations. In International Conference on Machine Learning. PmLR, 1597–1607.

[11] Zijun Chen, Wenbo Hu, Ya Li, Lei Miao, Guoping Hu, and Richang Hong. 2026. ProSafePrune: Projected Safety Pruning for Mitigating Over-Refusal in LLMs. In The Fourteenth International Conference on Learning Representations.

[12] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457 (2018).

[13] Justin Cui, Wei-Lin Chiang, Ion Stoica, and Cho-Jui Hsieh. 2024. Orbench: An over-refusal benchmark for large language models. arXiv preprint arXiv:2405.20947 (2024).

[14] Mahavir Dabas, Si Chen, Charles Fleming, Ming Jin, and Ruoxi Jia. 2025. Just enough shifts: Mitigating over-refusal in aligned language models with targeted representation fine-tuning. arXiv preprint arXiv:2507.04250 (2025).

[15] Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. 2023. Enhancing chat language models by

scaling high-quality instructional conversations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. 3029–3051.

[16] Naoya Hasegawa and Issei Sato. 2024. Exploring weight balancing on long-tailed recognition problem. In International Conference on Learning Representations, Vol. 2024. 40705–40732.

[17] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300 (2020).

[18] Kexin Huang, Xiangyang Liu, Qianyu Guo, Tianxiang Sun, Jiawei Sun, Yaru Wang, Zeyang Zhou, Yixu Wang, Yan Teng, Xipeng Qiu, et al. 2024. Flames: Benchmarking value alignment of llms in chinese. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 4551–4591.

[19] Yangsibo Huang, Samyak Gupta, Mengzhou Xia, Kai Li, and Danqi Chen. 2024. Catastrophic jailbreak of open-source llms via exploiting generation. In International Conference on Learning Representations, Vol. 2024. 13707–13727.

[20] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024).

[21] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. 2020. Supervised contrastive learning. Advances in Neural Information Processing Systems 33 (2020), 18661–18673.

[22] Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. Truthfulqa: Measuring how models mimic human falsehoods. In Proceedings ofthe 60th Annual Meeting of the Association for Computational Linguistics (volume 1: long papers). 3214–3252.

[23] Yuxiao Lu, Lin Xu, Yang Sun, Wenjun Li, and Jie Shi. 2026. Discern Truth from Falsehood: Reducing Over-Refusal via Contrastive Refinement. arXiv preprint arXiv:2603.03323 (2026).

[24] Mantas Mazeika, Dan Hendrycks, Huichen Li, Xiaojun Xu, Sidney Hough, Andy Zou, Arezoo Rajabi, Qi Yao, Zihao Wang, Jian Tian, et al. 2023. The trojan detection challenge. In NeurIPS 2022 Competition Track. PMLR, 279–291

[25] Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, et al. 2024. Harmbench: A standardized evaluation framework for automated red teaming and robust refusal. arXiv preprint arXiv:2402.04249 (2024).

[26] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. 2381–2391.

[27] Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson. 2024. Fine-tuning aligned language models compromises safety, even when users do not intend to!. In International Conference on Learning Representations, Vol. 2024. 30988–31043.

[28] Daniel Rho, TaeSoo Kim, Sooill Park, Jaehyun Park, and JaeHan Park. 2023. Understanding contrastive learning through the lens of margins. arXiv preprint arXiv:2306.11526 (2023).

[29] Paul Röttger, Hannah Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, and Dirk Hovy. 2024. Xstest: A test suite for identifying exaggerated safety behaviours in large language models. In Proceedings of the 2024 Conference of the North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 5377–5400.

[30] Florian Schrof, Dmitry Kalenichenko, and James Philbin. 2015. Facenet: A unified embedding for face recognition and clustering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 815–823.

[31] Omar Shaikh, Hongxin Zhang, William Held, Michael Bernstein, and Diyi Yang. 2023. On second thought, let’s not think step by step! bias and toxicity in zeroshot reasoning. In Proceedings ofthe 61st Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 4454–4470.

[32] Pengyang Shao, Naixin Zhai, Lei Chen, Yonghui Yang, Fengbin Zhu, Xun Yang, and Meng Wang. 2026. Baldro: A distributionally robust optimization based framework for large language model unlearning. In Proceedings ofthe ACM Web Conference 2026. 8874–8884.

[33] Chenyu Shi, Xiao Wang, Qiming Ge, Songyang Gao, Xianjun Yang, Tao Gui, Qi Zhang, Xuan-Jing Huang, Xun Zhao, and Dahua Lin. 2024. Navigating the overkill in large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 4602–4614.

[34] Enyi Shi, Fei Shen, Shuyi Miao, Linxia Zhu, Pengyang Shao, Jinhui Tang, and Tat-Seng Chua. 2026. Precise Shield: Explaining and Aligning VLLM Safety via Neuron-Level Guidance. arXiv preprint arXiv:2604.08881 (2026).

[35] Minseok Son, Inyong Koo, Jinyoung Park, and Changick Kim. 2025. Dificultyaware balancing margin loss for long-tailed recognition. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 39. 20522–20530.

[36] Yifan Sun, Changmao Cheng, Yuhan Zhang, Chi Zhang, Liang Zheng, Zhongdao Wang, and Yichen Wei. 2020. Circle loss: A unified perspective of pair similarity optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern recognition. 6398–6407.

[37] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. 2023. Stanford alpaca: An instruction-following llama model.

[38] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. 2024. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295 (2024)

[39] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yas mine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhos ale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023).

[40] Xinpeng Wang, Chengzhi Martin Hu, Paul Röttger, and Barbara Plank. 2025. Surgical, cheap, and flexible: Mitigating false refusal in language models via single vector ablation. In International Conference on Learning Representations, Vol. 2025. 33824–33843.

[41] Yixu Wang, Yan Teng, Kexin Huang, Chengqi Lyu, Songyang Zhang, Wenwei Zhang, Xingjun Ma, Yu-Gang Jiang, Yu Qiao, and Yingchun Wang. 2024. Fake alignment: Are llms really aligned well?. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 4696–4712.

[42] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025).

[43] Jiacheng Yang, Ruichi Zhang, Chikai Shang, Mengke Li, Xinyi Shang, Junlong Gao, Yonggang Zhang, and Yang Lu. 2026. Decision Boundary-aware Generation for Long-tailed Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 29441–29450.

[44] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. 2020. Gradient surgery for multi-task learning. Advances in neural information processing systems 33 (2020), 5824–5836.

[45] Naixin Zhai, Pengyang Shao, Binbin Zheng, Yonghui Yang, Fei Shen, Long Bai, and Xun Yang. 2026. Maximizing local entropy where it matters: Prefix-aware localized llm unlearning. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 19514–19531.

[46] Chujie Zheng, Fan Yin, Hao Zhou, Fandong Meng, Jie Zhou, Kai-Wei Chang, Minlie Huang, and Nanyun Peng. 2024. On prompt-driven safeguarding for large language models. arXiv preprint arXiv:2401.18018 (2024).

[47] Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J Zico Kolter, and Matt Fredrikson. 2023. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043 (2023).

## Limitations and Ethical Considerations

Limitations. Our experiments focus on open-weight instructiontuned models with 7–13 billion parameters and predominantly English benchmarks. Although the Qwen3-8B results support generalization to a recent model family, further evaluation is needed on larger, proprietary, and multilingual models. Moreover, our firstorder analysis provides a suficient characterization rather than establishing harmful-gradient coupling as the sole cause of over refusal. Finally, automatic evaluation partly relies on an LLM judge, and strong benchmark performance does not guarantee safety in all deployment settings.

Human-evaluation ethics. The human evaluation was conducted by three master’s students, each of whom independently assessed the model responses. The annotators were informed of the study procedure and the possibility of encountering superficially sensitive language, provided informed consent, and could discontinue participation without penalty. They were compensated at a rate of CNY 100 per hour. No personally identifying information was collected, and only aggregate results are reported.

Generative-AI assistance. OpenAI Codex was used to assist with language editing, manuscript organization, and LaTeX preparation. The authors made all research decisions, conducted the experiments, and verified the mathematical derivations, numerical results, and citations.

## A Proof of Refusal-subspace Attraction

We prove Proposition 1. For the �-th over-refusal prompt, recall that $\mathcal { L } _ { \mathrm { S F T } , i } ^ { \mathrm { o v e r } }$ and $\mathcal { L } _ { \mathrm { S F T } , i } ^ { \prime 0 \mathrm { v e r } }$ denote the losses defined with respect to the correct helpful response $y _ { i }$ and a refusal response $y _ { i } ^ { \prime } ,$ respectively. Their gradients at step � are

$$
\mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } = \nabla _ { \theta } \mathcal { L } _ { \mathrm { S F T } , i } ^ { \mathrm { o v e r } } ( \theta _ { t } , x _ { i } , y _ { i } )\tag{13}
$$

and

$$
\mathbf { g } _ { i , t } ^ { \prime \mathrm { o v e r } } = \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { S F T } , i } ^ { \prime \mathrm { o v e r } } ( \boldsymbol { \theta } _ { t } , \boldsymbol { x } _ { i } , y _ { i } ^ { \prime } ) .\tag{14}
$$

Since the SFT losses are negative log-likelihood losses, the refusal preference margin can be written as

$$
\begin{array} { r l } & { m _ { i } ( \theta ) = \log \ p _ { \theta } ( y _ { i } ^ { \prime } \mid x _ { i } ) - \log \ p _ { \theta } ( y _ { i } \mid x _ { i } ) } \\ & { \qquad = \mathcal { L } _ { \mathrm { S F T } , i } ^ { \mathrm { o v e r } } ( \theta , x _ { i } , y _ { i } ) - \mathcal { L } _ { \mathrm { S F T } , i } ^ { \prime \mathrm { o v e r } } ( \theta , x _ { i } , y _ { i } ^ { \prime } ) . } \end{array}\tag{15}
$$

Therefore, its gradient at $\theta _ { t }$ is

$$
\begin{array} { r } { \nabla _ { \boldsymbol { \theta } } m _ { i } ( \boldsymbol { \theta } _ { t } ) = \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } - \mathbf { g } _ { i , t } ^ { \prime \mathrm { o v e r } } . } \end{array}\tag{16}
$$

Consider the parameter update induced by the aggregated harmfulsample gradient:

$$
\theta _ { t + 1 } = \theta _ { t } - \eta \bar { \bf g } _ { t } .\tag{17}
$$

By a first-order Taylor expansion around $\theta _ { t }$ , we have

$$
\begin{array} { r l } & { m _ { i } ( \theta _ { t } - \eta \bar { \bf g } _ { t } ) - m _ { i } ( \theta _ { t } ) = - \eta \left. \nabla _ { \theta } m _ { i } ( \theta _ { t } ) , \bar { \bf g } _ { t } \right. + O ( \eta ^ { 2 } ) } \\ & { \quad \quad \quad \quad = \eta \left. { \bf g } _ { i , t } ^ { \prime \mathrm { o v e r } } - { \bf g } _ { i , t } ^ { \mathrm { o v e r } } , \bar { \bf g } _ { t } \right. + O ( \eta ^ { 2 } ) . } \end{array}\tag{18}
$$

Since P is an orthogonal projection matrix, we have $\mathbf { P } ^ { \top } = \mathbf { P }$ $\mathbf { P } ^ { 2 } = \mathbf { P } ,$ and $\mathbf { P } ( \mathbf { I } - \mathbf { P } ) = \mathbf { 0 }$ . Hence,

$$
\begin{array} { r l } & { \left. \mathbf { g } _ { i , t } ^ { \prime \mathrm { o v e r } } - \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } , \bar { \mathbf { g } } _ { t } \right. } \\ & { = \left. \mathbf { P } \left( \mathbf { g } _ { i , t } ^ { \prime \mathrm { o v e r } } - \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } \right) , \mathbf { P } \bar { \mathbf { g } } _ { t } \right. } \\ & { \quad + \left. \left( \mathbf { I } - \mathbf { P } \right) \left( \mathbf { g } _ { i , t } ^ { \prime \mathrm { o v e r } } - \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } \right) , \left( \mathbf { I } - \mathbf { P } \right) \bar { \mathbf { g } } _ { t } \right. } \\ & { = A _ { i , t } + R _ { i , t } . } \end{array}\tag{19}
$$

Substituting this decomposition into the Taylor expansion gives

$$
m _ { i } ( \theta _ { t } - \eta \bar { \bf g } _ { t } ) - m _ { i } ( \theta _ { t } ) = \eta \left( A _ { i , t } + R _ { i , t } \right) + O ( \eta ^ { 2 } ) .\tag{20}
$$

If $A _ { i , t } > | R _ { i , t } |$ , we have

$$
A _ { i , t } + R _ { i , t } \geq A _ { i , t } - | R _ { i , t } | > 0 .\tag{21}
$$

Therefore, for a suficiently small learning rate $\eta ,$ the positive firstorder term dominates the second-order remainder, yielding

$$
m _ { i } ( \theta _ { t } - \eta \bar { \bf g } _ { t } ) - m _ { i } ( \theta _ { t } ) > 0 .\tag{22}
$$

Thus, the harmful gradient update increases the refusal preference margin of the over-refusal prompt, pulling it toward the refusal side. This completes the proof.

## B Proof of Decoupling by Refusal-orthogonal Projection

We prove Proposition 2. For the �-th over-refusal sample, let $\mathcal { L } _ { \mathrm { S F T } , i } ^ { \mathrm { o v e r } }$ denote its training loss, and let

$$
\mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } = \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { S F T } , i } ^ { \mathrm { o v e r } } ( \theta _ { t } , x _ { i } , y _ { i } )\tag{23}
$$

be its gradient at step �. The refusal-orthogonal gradient used by GOAL is defined as

$$
\begin{array} { r } { \tilde { \mathbf { g } } _ { i , t } ^ { \mathrm { o v e r } } = ( \mathbf { I } - \mathbf { P } ) \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } . } \end{array}\tag{24}
$$

We first show that the projected gradient has zero overlap with the refusal-subspace component ofthe aggregated harmful gradient. By the definition of $\tilde { \bf g } _ { i , t } ^ { \mathrm { o v e r } }$ , we have

$$
\begin{array} { r l } & { \left. \tilde { \bf g } _ { i , t } ^ { \mathrm { o v e r } } , \mathbf { P } \bar { \bf g } _ { t } \right. = \left. ( { \bf I } - { \bf P } ) { \bf g } _ { i , t } ^ { \mathrm { o v e r } } , \mathbf { P } \bar { \bf g } _ { t } \right. } \\ & { \quad \quad \quad \quad = \left. { \bf g } _ { i , t } ^ { \mathrm { o v e r } } , ( { \bf I } - { \bf P } ) ^ { \top } { \bf P } \bar { \bf g } _ { t } \right. . } \end{array}\tag{25}
$$

Since P is an orthogonal projector, $( \mathbf { I } - \mathbf { P } ) ^ { \top } = \mathbf { I } - \mathbf { P } .$ Therefore,

$$
( \mathbf { I } - \mathbf { P } ) ^ { \top } \mathbf { P } = ( \mathbf { I } - \mathbf { P } ) \mathbf { P } = \mathbf { P } - \mathbf { P } ^ { 2 } = \mathbf { 0 } .\tag{26}
$$

Thus, $\left. \tilde { \bf g } _ { i , t } ^ { \mathrm { o v e r } } , { \bf P } \bar { \bf g } _ { t } \right. = 0 .$

Next, we characterize the remaining overlap between the projected over-refusal gradient and the aggregated harmful gradient. Decomposing the aggregated harmful gradient gives $\bar { \bf g } _ { t } ~ =$ $\mathbf { P } \bar { \mathbf { g } } _ { t } + ( \mathbf { I } - \mathbf { P } ) \bar { \mathbf { g } } _ { i }$ . Therefore,

$$
\begin{array} { r l } & { \left. \tilde { \bf g } _ { i , t } ^ { \mathrm { o v e r } } , \bar { \bf g } _ { t } \right. = \left. ( { \bf I } - { \bf P } ) { \bf g } _ { i , t } ^ { \mathrm { o v e r } } , { \bf P } \bar { \bf g } _ { t } + ( { \bf I } - { \bf P } ) \bar { \bf g } _ { t } \right. } \\ & { \quad \quad \quad \quad = \left. ( { \bf I } - { \bf P } ) { \bf g } _ { i , t } ^ { \mathrm { o v e r } } , { \bf P } \bar { \bf g } _ { t } \right. } \\ & { \quad \quad \quad \quad + \left. ( { \bf I } - { \bf P } ) { \bf g } _ { i , t } ^ { \mathrm { o v e r } } , ( { \bf I } - { \bf P } ) \bar { \bf g } _ { t } \right. . } \end{array}\tag{27}
$$

The first term is zero by the orthogonality shown above. Hence,

$$
\left. \tilde { \mathbf { g } } _ { i , t } ^ { \mathrm { o v e r } } , \bar { \mathbf { g } } _ { t } \right. = \left. ( \mathbf { I } - \mathbf { P } ) \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } , ( \mathbf { I } - \mathbf { P } ) \bar { \mathbf { g } } _ { t } \right. .\tag{28}
$$

Thus, the projection eliminates the overlap with the harmful-gradient component inside the refusal subspace, leaving only the interaction between their components outside the refusal subspace.

It remains to show that the projected gradient is still a first-order descent direction for the over-refusal training loss. By a first-order Taylor expansion around $\theta _ { t } ,$ we have

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { S F T } , i } ^ { \mathrm { o v e r } } ( \theta _ { t } - \eta \tilde { \mathbf { g } } _ { i , t } ^ { \mathrm { o v e r } } , x _ { i } , y _ { i } ) - \mathcal { L } _ { \mathrm { S F T } , i } ^ { \mathrm { o v e r } } ( \theta _ { t } , x _ { i } , y _ { i } ) } \\ & { \quad = - \eta \left. \nabla _ { \theta } \mathcal { L } _ { \mathrm { S F T } , i } ^ { \mathrm { o v e r } } ( \theta _ { t } , x _ { i } , y _ { i } ) , \tilde { \mathbf { g } } _ { i , t } ^ { \mathrm { o v e r } } \right. + O ( \eta ^ { 2 } ) } \\ & { \quad = - \eta \left. \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } , ( \mathbf { I } - \mathbf { P } ) \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } \right. + O ( \eta ^ { 2 } ) . } \end{array}\tag{29}
$$

Since I − P is also an orthogonal projector, it satisfies

$$
( \mathbf { I } - \mathbf { P } ) ^ { \top } = \mathbf { I } - \mathbf { P } , \qquad ( \mathbf { I } - \mathbf { P } ) ^ { 2 } = \mathbf { I } - \mathbf { P } .\tag{30}
$$

Therefore,

$$
\begin{array} { r l } & { \left. { { \bf { g } } _ { i , t } ^ { \mathrm { o v e r } } , ( { \bf { I } } - { \bf { P } } ) { \bf { g } } _ { i , t } ^ { \mathrm { o v e r } } } \right. = \left. { ( { \bf { I } } - { \bf { P } } ) { \bf { g } } _ { i , t } ^ { \mathrm { o v e r } } , ( { \bf { I } } - { \bf { P } } ) { \bf { g } } _ { i , t } ^ { \mathrm { o v e r } } } \right. } \\ & { \quad \quad \quad \quad \quad \quad = \left. { ( { \bf { I } } - { \bf { P } } ) { \bf { g } } _ { i , t } ^ { \mathrm { o v e r } } } \right. _ { 2 } ^ { 2 } . } \end{array}\tag{31}
$$

Substituting this identity into the Taylor expansion gives

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { S F T } , i } ^ { \mathrm { o v e r } } ( \theta _ { t } - \eta \tilde { \mathbf { g } } _ { i , t } ^ { \mathrm { o v e r } } , x _ { i } , y _ { i } ) - \mathcal { L } _ { \mathrm { S F T } , i } ^ { \mathrm { o v e r } } ( \theta _ { t } , x _ { i } , y _ { i } ) } \\ & { \qquad = - \eta \left\| ( \mathbf { I } - \mathbf { P } ) \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } \right\| _ { 2 } ^ { 2 } + O ( \eta ^ { 2 } ) . } \end{array}\tag{32}
$$

If $\begin{array} { r } { \left\| ( \mathbf { I } - \mathbf { P } ) \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } \right\| _ { 2 } > 0 , } \end{array}$ then the first-order term is strictly negative. For a suficiently small learning rate $\eta ,$ the negative first-order term dominates the higher-order remainder, and thus

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S F T } , i } ^ { \mathrm { o v e r } } ( \theta _ { t } - \eta \tilde { \mathbf { g } } _ { i , t } ^ { \mathrm { o v e r } } , x _ { i } , y _ { i } ) - \mathcal { L } _ { \mathrm { S F T } , i } ^ { \mathrm { o v e r } } ( \theta _ { t } , x _ { i } , y _ { i } ) < 0 . } \end{array}\tag{33}
$$

Therefore, the projected update eliminates the harmful-gradient component inside the refusal subspace while remaining a first-order descent direction for the over-refusal training loss, completing the proof.

## C Choice of Safety-relevant Layers

Our objective is to localize the layers responsible for constructing the model’s general safety boundary, for which benign and harmful samples provide two clean reference classes. We deliberately avoid using over-refusal samples for layer selection because these samples are harmless but are already misjudged by the model. Inspired by the success of [8, 14, 23], we identify the safety-relevant layers and apply LoRA fine-tuning to them, which allows us to better balance safety and general performance.

Per-layer separability probe. Let $\mathbf { h } _ { x } ^ { ( l ) }$ denote the last-token hidden state of prompt � at the output of transformer block �, obtained from a single forward pass on the base model. Given a benign prompt pool $\mathcal { P } _ { N }$ and a harmful prompt pool $\mathcal { P } _ { M }$ , we compute three per-layer cosine-similarity curves, each averaged over � random prompt pairs:

$$
\begin{array} { r l } & { \rho ^ { \mathrm { N N } } ( l ) = \mathbb { E } _ { x , x ^ { \prime } \sim \mathcal { P } _ { N } } \big [ \cos ( \mathbf { h } _ { x } ^ { ( l ) } , \mathbf { h } _ { x ^ { \prime } } ^ { ( l ) } ) \big ] , } \\ & { \rho ^ { \mathrm { M M } } ( l ) = \mathbb { E } _ { x , x ^ { \prime } \sim \mathcal { P } _ { M } } \big [ \cos ( \mathbf { h } _ { x } ^ { ( l ) } , \mathbf { h } _ { x ^ { \prime } } ^ { ( l ) } ) \big ] , } \\ & { \rho ^ { \mathrm { N M } } ( l ) = \mathbb { E } _ { x \sim \mathcal { P } _ { N } , x ^ { \prime } \sim \mathcal { P } _ { M } } \big [ \cos ( \mathbf { h } _ { x } ^ { ( l ) } , \mathbf { h } _ { x ^ { \prime } } ^ { ( l ) } ) \big ] , } \end{array}\tag{34}
$$

where cos(·, ·) is cosine similarity, and � and $x ^ { \prime }$ denote two prompts independently sampled from the corresponding prompt pools; $\bar { \rho } ^ { \mathrm { { N N } } }$ and $\bar { \rho } ^ { \mathrm { M M } }$ measure within-class cohesion (benign–benign and harmful harmful), and $\rho ^ { \mathrm { N M } }$ measures between-class similarity. As visualized in Figure 6, these curves provide a clear layer-by-layer trajectory of the representational separation between benign and harmful inputs.

Identifying safety-relevant layers. Let � be the number of transformer blocks, indexed by $l \in \{ 0 , \ldots , L - 1 \}$ . For each layer, we define the separation gap as $g ( l ) { \mathrm { \bar { \alpha } } } ( \rho ^ { \mathrm { N N } } ( l ) + \bar { \rho } ^ { \mathrm { M M } } ( l ) ) / 2 - \bar { \rho } ^ { \mathrm { N M } } ( l )$ where a larger value indicates stronger within-group similarity relative to cross-group similarity. Since we aim to identify where the benign–harmful separation emerges rather than where it is already maximal, we use the growth score $\Delta g ( l ) = g ( l ) - g ( l - 1 )$ . We set the window width � to approximately one-third of the total layers and select the contiguous block with the largest average growth:

$$
l _ { 0 } ^ { * } = \arg \operatorname* { m a x } _ { l _ { 0 } \in \{ 0 , \ldots , L - w \} } \frac { 1 } { w - 1 } \sum _ { l = l _ { 0 } + 1 } ^ { l _ { 0 } + w - 1 } \Delta g ( l ) ,\tag{35}
$$

In all subsequent training, LoRA updates and gradient orthogonalization are restricted to the layers in $\mathcal { L } ^ { * }$ (indicated by the gray dashed lines in Figure 6), while the contrastive objective is computed using the final-layer hidden representations.

## D Detailed Experimental Settings

Models. We evaluate the generalization and robustness of our method on three representative instruction-tuned LLMs: Gemma-7B-it, Llama-2-7B-Chat, and Llama-2-13B-Chat. Greedy decoding is used for all text generation.

Training Datasets. To train the models, we construct a dataset $\mathcal { D } _ { \mathrm { t r a i n } } = \mathcal { D } _ { \mathrm { h a r m f u l } } \cup \mathcal { D } _ { \mathrm { b e n i g n } } \cup \mathcal { D } _ { \mathrm { o v e r } }$ , comprising harmful, benign, and over-refusal samples. $\mathcal { D } _ { \mathrm { h a r m f u l } }$ consists of 200 harmful instructions drawn from AdvBench [47], MaliciousInstruct [19], and TDC-2023 [24, 25]. $\mathcal { D } _ { \mathrm { b e n i g n } }$ consists of 200 instructions drawn from the Stanford Alpaca dataset [37], covering a broad range of everyday user intents. $\mathcal { D } _ { \mathrm { o v e r } }$ is compiled by drawing 50 items from OR-Bench-Hard [13] and 25 items from each of XSTest [29] and PHTest [2], yielding 100 prompts whose surface form appears unsafe but whose underlying intent is benign. Reference answers for harmful prompts are aligned refusals, while those for benign and over-refusal prompts are helpful, compliant completions; all reference answers are generated using GPT-4o [20].

![](images/80a0aa890d746d838f58803c316d57ed40738d78e2971743ad99d4dc517ebd47.jpg)  
(a) Llama-2-7B-Chat

![](images/fb96f345d08d23bd36a32fd1c0cb8c61553f482425f27901a0be14cf8c8d5013.jpg)  
(b) Llama-2-13B-Chat

![](images/d82fa19858af67936098e0683b184111fca74299aec9bc22ac1186bffe5aaec8.jpg)  
(c) Gemma-7B-it  
Figure 6: Per-layer cosine similarity analysis across Llama-2-7B-Chat, Llama-2-13B-Chat, and Gemma-7B-it to identify safety relevant layers for orthogonal gradient projection. The $\rho ^ { \mathbf { N N } } , \rho ^ { \mathbf { M M } }$ , and $\rho ^ { \mathbf { N M } }$ curves show the similarity trends for each model, where solid lines represent the mean, shaded bands indicate the ±1 standard deviation, and gray dashed lines mark the selected safety band $\mathcal { L } ^ { * }$

Baseline Methods. We compare our proposed GOAL with the following baseline methods: SFT fine-tunes the base model on the aforementioned $\mathcal { D } _ { \mathrm { t r a i n } }$ without any additional regularization, serving as the most direct vanilla baseline. ACTOR [14] fine-tunes a single layer to suppress activation components that trigger refusal. DCR [23] introduces a preceding contrastive-refinement stage so the model can better distinguish truly toxic from superficially toxic prompts. SCANS [8] steers activations in safety-critical layers via refusal steering vectors. Surgical [40] ablates a single false-refusal direction in representation space to recover helpful responses on benign samples.

Evaluation Datasets. We evaluate over-refusal using the held-out versions of XSTest [29] (225 samples), OR-Bench-Hard [13] (1,269 samples), and PHTest [2] (2,052 samples), with OKTest [33] (300 samples) and CoCoNot [6] (379 samples) used as out-of-distribution (OOD) benchmarks. To ensure safety preservation, we further evaluate on JailbreakBench (JBB; [9]; 100 samples), targeting adversarial and jailbreak robustness, together with four harmfulness benchmarks from [4], collectively denoted as STL in this paper: I-MaliciousInstructions (98 samples), I-CoNa (178 samples), I-Controver (40 samples), and HarmfulQ [31] (200 samples)—covering hateful speech, controversial topics, malicious instructions, and common jailbreak prompts (516 samples in total). We assess general model capability using the Language Model Evaluation Harness on multiplechoice benchmarks, including MMLU [17], ARC-Challenge [12], OpenBookQA [26], and PIQA [5].

Data for Representation and Gradient Analyses. We construct balanced diagnostic sets containing 300 and 1,200 samples per category for the representation and gradient analyses, respectively (900 and $3 { , } 6 0 0$ samples in total). The benign samples are drawn from TruthfulQA [22] and UltraChat [15]; the over-refusal samples are drawn from PHTest [2] and OR-Bench-Hard [13]; and the harmful samples are drawn from AdvBench [47], Hex-Phi [27], and HH-RLHF [3]. These diagnostic sets are used only for the analyses in Figures 1 and 2 and are disjoint from the training and evaluation sets.

Metrics. Over-refusal is measured using the Compliance Rate (C.R.), defined as the ratio of compliant responses to the total number of generated responses. Similarly, the model’s safety is evaluated with the Safety Score (S.S.), calculated as the ratio of safe responses to the total number of responses. Response evaluation for both C.R. and S.S. is conducted using GPT-5.1. For general capability, we report accuracy (Acc) on all four multiple-choice benchmarks: for MMLU, this is the overall average accuracy across its 57 subjects; for ARC-Challenge, OpenBookQA, and PIQA, this is the task-level accuracy.

Training Hyperparameters and GOAL Implementation. To ensure parameter eficiency, all models are fine-tuned using Low-Rank Adaptation (LoRA) with a rank of 16, scaling factor $\alpha = 3 2 .$ and a dropout rate of 0.05. Specifically, we apply LoRA modules exclusively to a selected subset of layers tailored to each model: layers 12–22 for Llama-2-7B-Chat, layers 13–26 for Llama-2-13B-Chat, and layers 15–23 for Gemma-7B-it. During training, we use a micro-batch size of 1 and 4 gradient accumulation steps, resulting in an efective batch size of 4. All models are trained for 5 epochs using bfloat16 precision with a fixed random seed of 42. We apply a warmup ratio of 0.03, gradient clipping with a maximum norm of 1.0, and no weight decay. For the learning rates, we use $1 . 7 5 \times 1 0 ^ { - 4 }$ for both Llama-2-7B-Chat and Llama-2-13B-Chat, while a lower learning rate of $2 . 5 \times 1 0 ^ { - 5 }$ is adopted for Gemma-7B-it.

For the hyperparameters specific to the GOAL framework, we extract the gradient-based refusal subspace using a projection rank of $k = 1 0 ,$ . The contrastive representation loss is applied at the last layer of each model. The overall loss objective sets the contrastive loss weight to $\lambda = 0 . 1$ 1. For the circle loss formulation, we set the margin to 0.25, the scaling factor to $\gamma = 8 0$ , and maintain a representation bank M of size 128 to facilitate positive and negative pair comparisons. Specifically, rather than using random sampling, we employ an all-to-all deterministic comparison. To strictly control class imbalance, the bank is explicitly partitioned into two independent, class-specific FIFO queues, each with a capacity of 64 (one for harmful representations and one for over-refusal representations). New representations are routed to their respective queues, overwriting the oldest entries. For each query �, we use all currently available representations in the same-class queue, up to 64, as the positive set $\mathcal { P } _ { x }$ , and all currently available representations in the opposing-class queue, up to 64, as the negative set $N _ { x }$ . The contrastive loss is then computed directly over this full similarity matrix.

![](images/0bec5b8e24f1672a188044a853fd7f1d1eca93da3f5d04d9c16a4c96e52b5c0a.jpg)  
Figure 7: PCA-based layer-wise visualization of the last-token hidden representations of benign, harmful, and over-refusal samples on Llama-2-13B-Chat. At layer 14, harmful and over-refusal samples are separable in the representation space; however, the two types of samples show increasing visual overlap in deeper layers.

Implementation Details of Baselines. For the SFT baseline, we fine-tune the models on $\mathcal { D } _ { \mathrm { t r a i n } }$ utilizing exactly the same LoRA configurations, target layer subsets, and training hyperparameters as our GOAL framework. This ensures that any performance improvements are strictly attributable to our proposed mechanisms rather than hyperparameter tuning. For ACTOR, we utilize the same training dataset $( \mathcal { D } _ { \mathrm { t r a i n } } )$ as GOAL, while keeping its specific optimization hyperparameters and layer configurations identical to those in the original paper [14]. For DCR, which originally leverages 250 samples from XSTest for its contrastive-refinement stage, we maintain an equivalent data scale by sampling 250 instances across OR-Bench-Hard [13], XSTest [29], and PHTest [2] in accordance with the exact proportional ratio (2:1:1) used for $\mathcal { D } _ { \mathrm { o v e r } } .$ , keeping all other training settings unchanged [23]. For SCANS [8], the method selects layers based on safety relevance; following the original paper, layers 10–20 are selected for steering in the 7B models, while layers 16–26 are used for the 13B models. Furthermore, we strictly follow the original paper regarding the dataset and hyperparameter designs. Finally, for Surgical [40], we directly apply it using the default configurations and hyperparameters recommended by its original authors.

Evaluation Prompts for GPT-5.1 Judge. To automatically evaluate the Safety Score (S.S.) and Compliance Rate (C.R.), we employ GPT-5.1 as an impartial judge. We follow the evaluation protocols and prompt templates proposed by [14]. For full transparency and reproducibility, the exact prompt templates used in our evaluation pipeline are illustrated below. Specifically, Figure 11 details the template utilized for assessing model safety against harmful instructions, whereas Figure 12 provides the template for evaluating over-refusal behavior on benign and over-refusal instructions.

## E Additional Layer-wise Representation Visualization

To further examine whether the layer-wise representation pattern observed in the main text generalizes across models, we provide additional visualization results on Llama-2-13B-Chat. As shown in Figure 7, harmful and over-refusal samples are separable at layer 14. However, as the layer depth increases, the two types of samples show increasing visual overlap.

## F Detailed Results for Robustness and Hyperparameter Analysis

In this section, we provide the comprehensive quantitative results corresponding to the robustness and hyperparameter analyses discussed in the main text.

Robustness of the Gradient-based Refusal Subspace. We evaluate the sensitivity of the extracted subspace to the number of harmful prompts (�) used for its construction on Llama-2-7B-Chat. We vary � by sampling additional harmful prompts solely for constructing the harmful-gradient matrix and extracting the refusal subspace. The original training set $\mathcal { D } _ { \mathrm { t r a i n } }$ is kept fixed across all settings, and the additional prompts are not used for model optimization. As shown in Figure 8, GOAL remains highly stable across diferent values of �, suggesting that the extracted subspace is not sensitive to the prompt selection or sample size used for its estimation.

To further examine whether the extracted subspace depends on a particular harmful prompt source, we construct refusal subspaces using diferent harmful datasets. Specifically, alongside our default $\mathcal { D } _ { \mathrm { h a r m f u l } }$ , which consists of 200 harmful instructions drawn from AdvBench [47], MaliciousInstruct [19], and TDC2023 [24, 25], we also independently construct refusal subspaces using the Hex-Phi [27] benchmark and the HH-RLHF [3] dataset. As shown in Table 4, the performance of GOAL remains consistent across diferent datasets used for subspace construction. This indicates that the extracted refusal subspace efectively reflects a robust optimization � = 50 � = 100 � = 200 � = 400

Table 4: Robustness of refusal subspaces constructed from diferent harmful prompt sources. We compare subspaces built using the default harmful instruction set, Hex-Phi, and HH-RLHF.
<table><tr><td rowspan="2">Method</td><td colspan="5">Compliance Rate (C.R.) ↑</td><td colspan="2">Safety Score (S.S.) ↑</td></tr><tr><td>XST</td><td>ORH</td><td>PHT</td><td>OKT</td><td>CoCo</td><td>JBB</td><td>STL</td></tr><tr><td>GOAL (Hex-Phi)</td><td>0.9467</td><td>0.7604</td><td>0.9522</td><td>0.9667</td><td>0.8865</td><td>0.96</td><td>0.9845</td></tr><tr><td>GOAL (HH-RLHF)</td><td>0.9511</td><td>0.7526</td><td>0.9352</td><td>0.9700</td><td>0.8945</td><td>0.96</td><td>0.9884</td></tr><tr><td>GOAL (Ours)</td><td>0.9511</td><td>0.7699</td><td>0.9498</td><td>0.9700</td><td>0.9024</td><td>0.96</td><td>0.9864</td></tr></table>

![](images/5cf1e3d96d949bbfda6b714c72d6ed29d4da6eeeb424f86225c8c43e3bfc6231.jpg)

![](images/9e673c377129bb0e218bc8b77b7568afff37832c5407489c09146983e0d1042e.jpg)

![](images/d710402f0c8832d709f0b7960aea9f39e44ddb03050a746fcd65127fd2af15db.jpg)

![](images/bba5f618335e97396317672d53b496edc52ec26747d74eb149930b40fd714ed5.jpg)  
Figure 8: Robustness to harmful sample size �. Impact of the number of harmful prompts � used for subspace extraction on model performance.

![](images/9c1465e77b3ca2fb97cb1e68983aa9cb3110c4cf9e3f44f3c8a66708d3685e96.jpg)

![](images/f743db54108776140adae298953b49484e55f79b3b20e82f7477d4842beadcd3.jpg)

![](images/35c954cf8bafb48011c380c1f016f83891d1b9da72f7acbaf634fb5f31690001.jpg)

![](images/410f72ab72403e471ac094ab889f48a08549464e0459c51e4d72c96fe478b560.jpg)  
Figure 9: Sensitivity heatmaps of projection rank � and contrastive weight �.

![](images/0c46799f00a707417683131284b65eaf36c29de3bd01b0dc3b565e74b49b39fb.jpg)

Benign Over-refusal Harmful �: the �-th layer of the LLM  
![](images/d4119be06dcc4e63f9842f3017fab8a49caa4b8e95ea82c35f416d07321eb3bf.jpg)  
Figure 10: Layer-wise representation visualization across middle-to-late layers for benign, harmful, and over-refusal samples on Llama-2-7B-Chat.

structure shared across varied harmful prompt distributions, rather than being biased toward the properties of a specific dataset.

Hyperparameter analysis. We analyze the sensitivity of GOAL to its two main hyperparameters: the projection rank � and the contrastive loss weight �. Figure 9 presents the detailed results of various evaluation metrics across a two-dimensional grid search of � and � on Llama-2-7B-Chat.

## G Representation Separation after Training

As discussed in Section 1, harmful and over-refusal samples are visually distinguishable at certain intermediate layers, but their apparent representational separation decreases in deeper layers. To qualitatively examine how GOAL afects this layer-wise representation pattern, we visualize the last-token hidden representations of Llama-2-7B-Chat after fine-tuning in Figure 10. Compared with the representation pattern before fine-tuning, GOAL yields a clearer visual separation between harmful and over-refusal samples across the examined middle-to-late layers. This observation is consistent with the intended efect of the safety-aware contrastive objective, which explicitly encourages greater representational separation between the two sample types.

Algorithm 1: GOAL Training   
Input : Training set $\mathcal { D } _ { \mathrm { t r a i n } } = \mathcal { D } _ { \mathrm { h a r m f u l } } \cup \mathcal { D } _ { \mathrm { o v e r } } \cup \mathcal { D } _ { \mathrm { b e n i g n } } ;$   
trainable parameters $\theta ;$ refusal basis $\mathbf { V } _ { k } ;$ projection matrix   
$\mathbf { P } = \mathbf { V } _ { k } \mathbf { V } _ { k } ^ { \top } ;$ ; contrastive memory bank M; contrastive   
weight $\lambda ;$ Circle Loss margin � and scale .   
Output : Updated parameters $\theta .$   
Initialize a class-balanced FIFO memory bank   
$M = \left\{ { \mathcal { M } } _ { \mathrm { h a r m f u l } } , { \mathcal { M } } _ { \mathrm { o v e r } } \right\}$   
foreach training minibatch $\mathcal { B } = \{ \left( x _ { i } , y _ { i } \right) \} _ { i = 1 } ^ { B } \subset \mathcal { D } _ { \mathrm { t r a i n } }$ do   
// Safety-aware contrastive learning   
Run a batched forward pass with hidden states enabled   
For each harmful or over-refusal sample in ${ \mathcal { B } } ,$ extract its   
last-token hidden state $\mathbf { h } _ { i }$ at the final layer   
For each harmful or over-refusal sample, construct $\mathcal { P } _ { i }$ from   
representations in $\mathcal { M }$ with the same label, and ${ \cal N } _ { i }$ from   
representations with diferent labels   
if valid positive and negative pairs exist in M then   
Compute ${ \mathcal { L } } _ { \mathrm { c o n t } }$ using $\operatorname { E q . } \left( 1 1 \right)$   
else   
Set $\mathcal { L } _ { \mathrm { c o n t } }  0$   
Update $\mathcal { M }$ with detached harmful and over-refusal   
representations, where each representation is routed to its   
class-specific FIFO queue   
// Refusal-orthogonal projected SFT gradient   
foreach sample $( x _ { i } , y _ { i } ) \in { \mathcal { B } }$ do   
Compute the per-sample SFT gradient   
$\mathbf { g } _ { i } ^ { c } = \nabla _ { \theta } \mathcal { L } _ { \mathrm { S F T } , i } ^ { c } ( x _ { i } , y _ { i } ) , c \in$   
{harmful, over-refusal, benign}   
if $c = \mathrm { { o v e r } }$ -refusal then   
Orthogonalize the gradient against the refusal   
subspace: $\tilde { \mathbf { g } } _ { i }  \mathbf { g } _ { i } ^ { c } - \mathbf { P g } _ { i } ^ { c }$   
else   
Set $\tilde { \mathbf { g } } _ { i }  \mathbf { g } _ { i } ^ { c }$   
Compute the projected supervised gradient: $\begin{array} { r } { { \bf g } _ { \mathrm { S F T } } ^ { \mathrm { t o t a l } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } } \end{array}$ g˜ <sub>�</sub>   
$/ /$ GOAL update   
Compute the final GOAL update direction:   
g<sub>GOAL</sub> $= \mathbf { g } _ { \mathrm { S F T } } ^ { \mathrm { t o t a l } } + \lambda \nabla _ { \theta } \mathcal { L } _ { \mathrm { c o n t } } .$   
Update � with g<sub>GOAL</sub> using AdamW   
return �

## H Training Procedure and Runtime Overhead

Algorithm 1 summarizes the training procedure of GOAL. We measure its wall-clock cost on Llama-2-7B-Chat using a single NVIDIA A800 GPU with 80GB memory. Under the same training configuration, standard SFT takes 9.53 minutes, while the online training stage of GOAL takes 9.55 minutes, corresponding to an online overhead of only 0.21%. Before training, GOAL performs a one-time refusal-subspace extraction, which takes an additional 2.17 minutes. Therefore, the end-to-end runtime of the first GOAL run is 11.72 minutes, corresponding to a 23.0% overhead relative to standard SFT. The extracted subspace can be reused across repeated training runs that share the same base checkpoint, training data, and trainable parameterization, in which case the one-time preprocessing cost is amortized. We report both online and end-to-end costs to clearly distinguish the training-stage overhead from the complete runtime.

## I Some Qualitative Examples

To provide a more intuitive understanding of the improvements brought by our proposed GOAL framework, we present qualitative examples of the model’s text generation in this section. We select representative outputs from Llama-2-7B-Chat to illustrate the model’s behavior under two critical and distinct scenarios: truly harmful instructions and over-refusal instructions.

Figure 13 demonstrates the model’s response to a genuinely harmful query. As shown, while GOAL is designed to mitigate overrefusal, it successfully preserves the necessary safety guardrails. The model decisively rejects the malicious request with a safe and aligned refusal, suggesting that GOAL can preserve safety behavior on harmful requests.

Conversely, Figure 14 presents an over-refusal query—a benign instruction crafted with superficially sensitive keywords that typically trigger false refusals in standard aligned models. While standard baseline models conservatively refuse to answer, GOAL correctly discerns the underlying benign intent and provides a helpful, compliant response. Together, these qualitative results illustrate that GOAL achieves a superior balance between helpfulness and safety, efectively mitigating over-refusal.

## J Further Analysis of the Efectiveness of Safety-relevant Layers

Through layer-wise representation visualization, we observe that harmful and over-refusal samples are visually distinguishable at certain intermediate layers but show increasing overlap in subsequent layers. Meanwhile, GOAL applies its interventions to the entire safety-relevant layer block identified using benign and harmful samples. We conduct a controlled experiment to examine whether operating on the complete safety-relevant block is more efective than targeting only the specific intermediate layer where harmful and over-refusal representations are visually distinguishable or targeting the subsequent layers where they show increasing visual overlap.

We compare three target-layer choices. H–O Separable Layer uses only the intermediate layer at which harmful and over-refusal representations are distinguishable in our representation analysis. Overlap Layers uses a contiguous block located after the visually distinguishable layer, where harmful and over-refusal representations show increasing visual overlap. The number of layers in this block is matched to that of our default safety-relevant block. Safety-relevant Layers uses the complete layer block identified by our benign–harmful safety-boundary criterion.

Apart from the target layers, all variants use the same training data, projection rank, contrastive-loss weight, optimization hyperparameters, and decoding settings. The refusal-orthogonal projection and contrastive objective are applied using the same configuration in all three variants.

Table 5: Comparison of diferent target-layer choices on Llama-2-7B-Chat. H–O Separable Layer uses only the intermediate layer where harmful and over-refusal representations are distinguishable. Overlap Layers uses a subsequent contiguous block with the same number of layers as the safety-relevant block used by GOAL. All other training and optimization configurations are held fixed. Higher C.R. indicates better over-refusal mitigation.
<table><tr><td>Target-layer choice</td><td>Selected layer(s)</td><td>XST C.R. ↑</td><td>ORH C.R. ↑</td><td>PHT C.R. ↑</td><td>OKT C.R. ↑</td><td>CoCo C.R. ↑</td></tr><tr><td>H-O Separable Layer</td><td>13</td><td>0.7867</td><td>0.5343</td><td>0.8601</td><td>0.7867</td><td>0.7124</td></tr><tr><td>Overlap Layers</td><td>14-24</td><td>0.8400</td><td>0.6241</td><td>0.9230</td><td>0.8533</td><td>0.8047</td></tr><tr><td>Safety-relevant Layers (Ours)</td><td>12-22</td><td>0.9511</td><td>0.7699</td><td>0.9498</td><td>0.9700</td><td>0.9024</td></tr></table>

![](images/90b3a220e2cf6d660f95f6b9f0acf1d3b25855e002938bf9944e8b33744c32ee.jpg)  
Figure 11: GPT-5.1 Safety Judge Template.

As shown in Table 5, both the H–O Separable Layer and Overlap Layers variants underperform our default Safety-relevant Layers across all five over-refusal benchmarks. These results indicate that intervening only at an isolated intermediate layer or within the subsequent overlap layers is less efective than operating across the complete safety-relevant layer block. Applying GOAL to the complete block enables the gradient- and representation-level interventions to operate across a broader span of the layer-wise safety process and achieves the best overall over-refusal mitigation. This comparison supports our choice to intervene throughout the com plete safety-relevant layer block rather than only at the individual separable layer or the subsequent overlap layers.

## K Empirical Verification of Refusal-Subspace Attraction

Proposition 1 provides a suficient condition under which an ag gregated harmful-gradient update increases the refusal preference margin of an over-refusal prompt. Since the proposition does not imply that this condition holds universally, we empirically examine its prevalence across samples, training checkpoints, random seeds, and backbone models. We further evaluate whether the corresponding first-order prediction is associated with the margin change produced by a finite-step harmful-gradient update.

## K.1 Evaluation Setup

We conduct the analysis on Gemma-7B-it, Llama-2-7B-Chat, and Llama-2-13B-Chat. For each model, we use a fixed diagnostic set of $N _ { \mathrm { h a r m } } ~ = ~ 2 0 0$ harmful prompts and $N _ { \mathrm { o v e r } } ~ = ~ 2 0 0$ over-refusal prompts. These prompts are excluded from the safety-alignment training set and from the data used to construct the refusal subspace. We evaluate the checkpoint saved after each of the five training epochs and repeat the analysis over five random seeds.

At checkpoint �, we compute the aggregated harmful gradient over the diagnostic harmful set:

$$
\bar { \bf g } _ { t } = \frac { 1 } { N _ { \mathrm { h a r m } } } \sum _ { j = 1 } ^ { N _ { \mathrm { h a r m } } } { \bf g } _ { j , t } ^ { \mathrm { h a r m } } .\tag{36}
$$

For the �-th held-out over-refusal prompt, we compute the gradients induced by its correct helpful-target and a counterfactual refusal target, denoted by $\mathbf { g } _ { i , t } ^ { \mathrm { o v e r } }$ and $\mathbf { g } _ { i , t } ^ { \prime \mathrm { o v e r } }$ , respectively. We define

$$
\begin{array} { r } { \pmb { \delta } _ { i , t } = \mathbf { g } _ { i , t } ^ { \prime 0 \mathrm { v e r } } - \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } . } \end{array}\tag{37}
$$

Following Proposition 1, the refusal-subspace and residual contributions are

$$
A _ { i , t } = \left. \mathbf { P } \delta _ { i , t } , \mathbf { P } \bar { \mathbf { g } } _ { t } \right. ,\tag{38}
$$

$$
R _ { i , t } = \left. ( \mathbf { I } - \mathbf { P } ) \delta _ { i , t } , ( \mathbf { I } - \mathbf { P } ) \bar { \mathbf { g } } _ { t } \right. .\tag{39}
$$

## K.2 Evaluation Metrics

Refusal-subspace energy. We first measure the fraction of aggregated harmful-gradient energy captured by the refusal subspace:

$$
E _ { t } ^ { \mathrm { { h a r m } } } = \frac { \| \mathbf { P } \bar { \mathbf { g } } _ { t } \| _ { 2 } ^ { 2 } } { \| \bar { \mathbf { g } } _ { t } \| _ { 2 } ^ { 2 } + \epsilon } .\tag{40}
$$

A high value indicates that the aggregated harmful gradient is predominantly contained in the extracted refusal subspace, as assumed in the interpretation of Proposition 1.

Condition-satisfaction rate. To quantify the strength of the suf ficient condition, we define

$$
q _ { i , t } = \frac { A _ { i , t } } { | R _ { i , t } | + \epsilon } .\tag{41}
$$

The condition in Proposition 1 is satisfied when $q _ { i , t } > 1$ . For random seed $r ,$ the condition-satisfaction rate at the checkpoint after epoch � is

$$
S _ { e } ^ { ( r ) } = \frac { 1 } { N _ { \mathrm { o v e r } } } \sum _ { i = 1 } ^ { N _ { \mathrm { o v e r } } } \mathbb { I } \Big [ q _ { i , e } ^ { ( r ) } > 1 \Big ] .\tag{42}
$$

We report $S _ { e }$ averaged across five random seeds. We additionally report the median $q _ { i , t }$ over all evaluated sample–checkpoint pairs, averaged across the five seeds.

Finite-step verification. The proposition predicts the following first-order margin change:

$$
\widehat { \Delta m } _ { i , t } = \eta _ { \mathrm { d i a g } } \left( A _ { i , t } + R _ { i , t } \right) ,\tag{43}
$$

where $\eta _ { \mathrm { d i a g } }$ is set to the learning rate used for the corresponding backbone, namely $2 . 5 \times 1 0 ^ { - 5 }$ for Gemma-7B-it and $1 . 7 5 \times 1 0 ^ { - 4 }$ for both Llama-2-7B-Chat and Llama-2-13B-Chat.

To examine whether the first-order prediction remains informative for a finite update, we directly compute

$$
\Delta m _ { i , t } ^ { \mathrm { a c t u a l } } = m _ { i } \left( \theta _ { t } - \eta _ { \mathrm { d i a g } } \bar { \bf g } _ { t } \right) - m _ { i } ( \theta _ { t } ) .\tag{44}
$$

This controlled diagnostic update uses the raw aggregated gradient to match the Euclidean gradient setting analyzed in Proposition 1. It is used only for this analysis, after which the original checkpoint parameters are restored.

We report the Spearman correlation � betwee n Δc��<sub>,�</sub> and Δ�<sup>actual</sup><sub>�</sub> $\widehat { \Delta m } _ { i , t }$ ,� over all evaluated sample–checkpoint pairs. For each model, the reported correlation is averaged across five random seeds.

## K.3 Results

Table 6 summarizes the results. Across the three backbone models, the extracted refusal subspace captures between 0.82 and 0.89 of the aggregated harmful-gradient energy. This confirms that the harmful-gradient updates are predominantly concentrated in the extracted subspace.

The suficient condition $A _ { i , t } > | R _ { i , t } |$ holds for 70.3%–83.7% of the evaluated over-refusal samples across models and training epochs. The satisfaction rates exhibit only minor fluctuations across the five epochs, with no systematic decreasing trend. This suggests that the refusal-subspace attraction condition persists throughout safety alignment rather than being confined to its initial stage.

The median condition ratio ranges from 1.16 to 1.25, indicating that the condition is satisfied by a non-negligible margin in the median case. In addition, the predicted first-order margin changes are positively correlated with the observed finite-step changes, with Spearman correlations ranging from 0.24 to 0.37. Although these correlations are moderate, their consistently positive values across all three backbones indicate that the first-order analysis captures a systematic component of the observed margin dynamics.

Overall, these results show that the suficient condition in Proposition 1 is realized for a substantial fraction of over-refusal samples and remains broadly stable throughout training. We emphasize that the condition is suficient rather than necessary and does not imply that harmful-gradient coupling is the sole cause of over-refusal.

## L Refusal-Margin Reduction

In this section, we show that refusal-orthogonal projection can reduce the first-order increase in the refusal preference margin under a verifiable gradient-coupling condition. We further provide likelihood-based empirical evidence for the resulting margin reduction.

## L.1 Relative Reduction of the Refusal Preference Margin

We isolate the efect of refusal-orthogonal projection by comparing the projected update with its unprojected counterpart while holding all other update components fixed. This comparison characterizes the marginal contribution of the proposed projection. When the contrastive component is excluded, the unprojected counterpart reduces to the standard supervised update used by vanilla SFT.

Proposition 3 (Relative refusal-margin reduction). For the �-th over-refusal sample at optimization step �, let

$$
\mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } = \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { S F T } , i } ^ { \mathrm { o v e r } } ( \theta _ { t } , x _ { i } , y _ { i } )\tag{45}
$$

and

$$
\mathbf { g } _ { i , t } ^ { \prime \mathrm { o v e r } } = \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { S F T } , i } ^ { \prime \mathrm { o v e r } } ( \boldsymbol { \theta } _ { t } , \boldsymbol { x } _ { i } , \boldsymbol { y } _ { i } ^ { \prime } )\tag{46}
$$

denote the gradients induced by the correct helpful-target and a refusal-target, respectively. Let P be the orthogonal projector onto the refusal subspace.

Let $\mathbf { g } _ { t } ^ { \mathrm { s h a r e d } }$ collect all update components that are held fixed in this comparison, such as the harmful, benign, and, when applicable, contrastive-gradient components. Define the unprojected and projected update directions as

$$
\mathbf { g } _ { t } ^ { \mathrm { u n p r o j } } = \mathbf { g } _ { t } ^ { \mathrm { s h a r e d } } + \alpha _ { i } \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } ,\tag{47}
$$

and

$$
\mathbf { g } _ { t } ^ { \mathrm { p r o j } } = \mathbf { g } _ { t } ^ { \mathrm { s h a r e d } } + \alpha _ { i } ( \mathbf { I } - \mathbf { P } ) \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } ,
$$

where $\alpha _ { i } > 0$ is the aggregation weight ofthe sample.

(48)

Define the refusal preference margin as

$$
m _ { i } ( \theta ) = \log { p _ { \theta } ( y _ { i } ^ { \prime } \mid x _ { i } ) } - \log { p _ { \theta } ( y _ { i } \mid x _ { i } ) } .\tag{49}
$$

If

$$
\left. \mathbf { P g } _ { i , t } ^ { \prime \mathrm { o v e r } } , \mathbf { P g } _ { i , t } ^ { \mathrm { o v e r } } \right. > \left. \mathbf { P g } _ { i , t } ^ { \mathrm { o v e r } } \right. _ { 2 } ^ { 2 } ,\tag{50}
$$

then, for a suficiently small learning rate �,

$$
\Delta m _ { i , t } ^ { \mathrm { p r o j } } < \Delta m _ { i , t } ^ { \mathrm { u n p r o j } } ,\tag{51}
$$

![](images/bb5d7bb2b72e347fc2b6b9bb22e608b643d4b8f96643e945c40febcbcfa230e2.jpg)  
Figure 12: GPT-5.1 Over-Refusal Judge Template.

Table 6: Empirical verification of Proposition 1. �<sup>harm</sup> denotes the fraction of aggregated harmful-gradient energy captured by the refusal subspace. $S _ { e }$ is the percentage of over-refusal samples satisfying $A _ { i , t } > | R _ { i , t } |$ at the checkpoint after epoch �. Median � is computed over all evaluated sample–checkpoint pairs. � is the Spearman correlation between the predicted and actual finite-step margin changes. Results are averaged across five random seeds.
<table><tr><td>Model</td><td> $E ^ { \mathrm { h a r m \ . } }$  ↑</td><td>S1 (%)</td><td> $S _ { 2 } \ ( \% )$ </td><td> $S _ { 3 }$  (%)</td><td> $S _ { 4 } \left( \% \right)$ </td><td>S5 (%)</td><td>Median q</td><td>Spearman  $\rho$ </td></tr><tr><td>Gemma-7B-it</td><td>0.89</td><td>78.5</td><td>76.5</td><td>77.5</td><td>79.5</td><td>78.5</td><td>1.25</td><td>0.37</td></tr><tr><td>Llama-2-7B-Chat</td><td>0.82</td><td>72.1</td><td>70.3</td><td>72.6</td><td>74.3</td><td>72.4</td><td>1.16</td><td>0.24</td></tr><tr><td>Llama-2-13B-Chat</td><td>0.85</td><td>81.2</td><td>83.7</td><td>81.3</td><td>79.6</td><td>81.5</td><td>1.24</td><td>0.35</td></tr></table>

up to an $O ( \eta ^ { 2 } )$ remainder, where

$$
\Delta m _ { i , t } ^ { q } = m _ { i } ( \theta _ { t } - \eta \mathbf { g } _ { t } ^ { q } ) - m _ { i } ( \theta _ { t } ) , \qquad q \in \{ \mathrm { u n p r o j , p r o j } \} .\tag{52}
$$

Therefore, under condition (50), refusal-orthogonal projection reduces the first-order increase in the refusal preference margin relative to the corresponding unprojected update.

Proof. Since the SFT losses are negative log-likelihood losses, the gradient of the refusal preference margin is

$$
\begin{array} { r } { \nabla _ { \theta } m _ { i } ( \theta _ { t } ) = \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } - \mathbf { g } _ { i , t } ^ { \prime \mathrm { o v e r } } . } \end{array}\tag{53}
$$

For an update direction $\mathbf { g } _ { t } ^ { q } ,$ , a first-order Taylor expansion gives

$$
\Delta m _ { i , t } ^ { q } = - \eta \left. \nabla _ { \theta } m _ { i } ( \theta _ { t } ) , \mathbf { g } _ { t } ^ { q } \right. + O ( \eta ^ { 2 } ) .\tag{54}
$$

The diference between the projected and unprojected update directions is

$$
\mathbf { g } _ { t } ^ { \mathrm { p r o j } } - \mathbf { g } _ { t } ^ { \mathrm { u n p r o j } } = - \alpha _ { i } \mathbf { P } \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } .\tag{55}
$$

Consequently,

$$
\begin{array} { r l } & { \Delta m _ { i , t } ^ { \mathrm { p r o j } } - \Delta m _ { i , t } ^ { \mathrm { u n p r o j } } } \\ & { = - \eta \left. \nabla _ { \theta } m _ { i } ( \theta _ { t } ) , \mathbf { g } _ { t } ^ { \mathrm { p r o j } } - \mathbf { g } _ { t } ^ { \mathrm { u n p r o j } } \right. + O ( \eta ^ { 2 } ) } \\ & { = \eta \alpha _ { i } \left. \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } - \mathbf { g } _ { i , t } ^ { \prime \mathrm { o v e r } } , \mathbf { P } \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } \right. + O ( \eta ^ { 2 } ) . } \end{array}\tag{56}
$$

Since P is an orthogonal projector, $\mathbf { P } ^ { \top } = \mathbf { P }$ and $\mathbf { P } ^ { 2 } = \mathbf { P } .$ . Therefore,

$$
\left. \mathbf { g } _ { i , t } ^ { \mathrm { { o v e r } } } , \mathbf { P } \mathbf { g } _ { i , t } ^ { \mathrm { { o v e r } } } \right. = \left. \mathbf { P } \mathbf { g } _ { i , t } ^ { \mathrm { { o v e r } } } \right. _ { 2 } ^ { 2 } ,\tag{57}
$$

and

$$
\begin{array} { r l } & { \left. \mathbf { g } _ { i , t } ^ { \prime \mathrm { o v e r } } , \mathbf { P } \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } \right. } \\ & { = \left. \mathbf { P } \mathbf { g } _ { i , t } ^ { \prime \mathrm { o v e r } } , \mathbf { P } \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } \right. . } \end{array}\tag{58}
$$

It follows that

$$
\begin{array} { r l } & { \Delta m _ { i , t } ^ { \mathrm { p r o j } } - \Delta m _ { i , t } ^ { \mathrm { u n p r o j } } } \\ & { = \eta \alpha _ { i } \Big ( \| \mathbf { P g } _ { i , t } ^ { \mathrm { o v e r } } \| _ { 2 } ^ { 2 } - \langle \mathbf { P g } _ { i , t } ^ { \prime \mathrm { o v e r } } , \mathbf { P g } _ { i , t } ^ { \mathrm { o v e r } } \rangle \Big ) + O ( \eta ^ { 2 } ) . } \end{array}\tag{59}
$$

Under condition (50), the first-order term in $\operatorname { E q . }$ (59) is strictly negative. For a suficiently small learning rate, this negative firstorder term dominates the $O ( \eta ^ { 2 } )$ remainder, yielding

$$
\Delta m _ { i , t } ^ { \mathrm { p r o j } } < \Delta m _ { i , t } ^ { \mathrm { u n p r o j } } .
$$

Interpretation. The condition in Eq. (50) states that, inside the refusal subspace, the helpful-target gradient is more strongly coupled with the refusal-target gradient than with itself. Under this condition, removing the refusal-subspace component of the helpfultarget gradient reduces the first-order refusal-margin increase relative to leaving this component unchanged. This is a relative result and does not require the refusal margin to decrease absolutely after every update.

Table 7: Refusal preference margins and robustness under AdamW preconditioning. “Base”, $\mathbf { \mu ^ { * } s F T ^ { * } }$ , and “Ortho only” report the average refusal preference margins of the base model, vanilla SFT model, and refusal-orthogonally trained model, respectively. A lower margin indicates a weaker relative preference for refusal. Δ� denotes the margin reduction from SFT to Ortho only, thereby isolating the additional efect of refusal-orthogonal projection under the same training data and configuration. �<sup>Adam</sup> denotes the residual refusal-subspace leakage after AdamW preconditioning, and $Q ^ { \mathrm { A d a m } }$ denotes the coupling reduction relative to the unprojected AdamW-preconditioned gradient. Results for SFT, Ortho only, and the AdamW diagnostics are averaged over five runs with diferent random seeds; the base models are evaluated once because they involve no additional training.
<table><tr><td></td><td colspan="4">Refusal Preference Margin ↓</td><td colspan="2">AdamW Robustness</td></tr><tr><td>Model</td><td>Base</td><td>SFT</td><td>Ortho only</td><td>∆M↑</td><td> $L ^ { \mathrm { A d a m } } \downarrow$ </td><td> $Q ^ { \mathrm { A d a m } } \uparrow$ </td></tr><tr><td>Gemma-7B-it</td><td>0.187</td><td>-0.086</td><td>-0.141</td><td>0.055</td><td>0.036</td><td>0.885</td></tr><tr><td>Llama-2-7B-Chat</td><td>0.215</td><td>-0.093</td><td>-0.118</td><td>0.025</td><td>0.041</td><td>0.841</td></tr><tr><td>Llama-2-13B-Chat</td><td>0.159</td><td>-0.081</td><td>-0.113</td><td>0.032</td><td>0.047</td><td>0.858</td></tr></table>

## L.2 Empirical Analysis of Refusal Preference Margins

We provide likelihood-based empirical evidence for the refusalmargin reduction efect by comparing the base model, vanilla SFT, and the model trained using refusal-orthogonal projection (Ortho only). The evaluation is conducted on held-out over-refusal samples that are not used for training or refusal subspace construction. SFT and Ortho only use identical training data, target layers, and optimization hyperparameters, with refusal-orthogonal projection being the only diference.

For each over-refusal prompt $x _ { i } ,$ , we pair its correct helpful response $y _ { i }$ with a refusal response $y _ { i } ^ { \prime } .$ The refusal preference margin of the �-th over-refusal sample is then defined as

$$
m _ { i } ( \theta ) = \log p _ { \theta } ( y _ { i } ^ { \prime } \mid x _ { i } ) - \log p _ { \theta } ( y _ { i } \mid x _ { i } ) .\tag{60}
$$

A positive margin indicates that the model assigns a higher response log-likelihood to the refusal response, whereas a smaller margin indicates a weaker relative preference for refusal.

We report the average refusal preference margin over $N _ { \mathrm { o v e r } }$ heldout over-refusal samples:

$$
M ( \theta ) = \frac { 1 } { N _ { \mathrm { o v e r } } } \sum _ { i = 1 } ^ { N _ { \mathrm { o v e r } } } m _ { i } ( \theta ) .\tag{61}
$$

To isolate the additional efect of refusal-orthogonal projection, we define the margin reduction relative to vanilla SFT as

$$
\Delta M = M ( \theta _ { \mathrm { S F T } } ) - M ( \theta _ { \mathrm { O r t h o } } ) .\tag{62}
$$

Since SFT and Ortho only use identical training data and configurations, a positive Δ� indicates that refusal-orthogonal projection further reduces the refusal preference margin beyond the efect of helpful supervision from standard SFT. The resulting margins and margin reductions are reported in Table 7, together with the AdamW robustness results discussed in Section M.

## L.3 Empirical Verification of the Projection Condition

Proposition 3 gives a suficient condition under which removing the refusal-subspace component of an over-refusal gradient reduces the first-order increase in its refusal preference margin. We empirically examine how frequently this condition holds during training and whether its strength is associated with the sample-level benefit of refusal-orthogonal projection.

For the �-th held-out over-refusal sample at the end of epoch $e ,$ let

$$
\mathbf { g } _ { i , e } ^ { \mathrm { o v e r } } = \nabla _ { \theta } \mathcal { L } _ { \mathrm { S F T } , i } ^ { \mathrm { o v e r } } ( \theta _ { e } , x _ { i } , y _ { i } )\tag{63}
$$

denote the gradient induced by the correct helpful-target $y _ { i } ,$ and let

$$
\mathbf { g } _ { i , e } ^ { \prime \mathrm { o v e r } } = \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { S F T } , i } ^ { \prime \mathrm { o v e r } } ( \boldsymbol { \theta } _ { e } , \boldsymbol { x } _ { i } , \boldsymbol { y } _ { i } ^ { \prime } )\tag{64}
$$

denote the counterfactual gradient induced by a refusal-target $y _ { i } ^ { \prime } .$ Here, $\theta _ { e }$ denotes the Ortho only model checkpoint obtained after epoch $e ,$ and P is the fixed harmful-gradient refusal-subspace projector used during training.

We define the condition ratio as

$$
r _ { i , e } = \frac { \left. \mathbf { P g } _ { i , e } ^ { \prime \mathrm { o v e r } } , \mathbf { P g } _ { i , e } ^ { \mathrm { o v e r } } \right. } { \left\| \mathbf { P g } _ { i , e } ^ { \mathrm { o v e r } } \right\| _ { 2 } ^ { 2 } + \epsilon } ,\tag{65}
$$

where � is a small constant for numerical stability. Up to this numerical stabilizer, the suficient condition in Proposition 3 is equivalent to $r _ { i , e } > 1$

We evaluate $r _ { i , e }$ on the same fixed held-out over-refusal set at the end of each of the five training epochs. These samples are not used for training or refusal-subspace construction. For epoch $e ,$ we define the condition-satisfaction rate as

$$
S _ { e } = \frac { 1 } { N _ { \mathrm { o v e r } } } \sum _ { i = 1 } ^ { N _ { \mathrm { o v e r } } } \mathbb { I } [ r _ { i , e } > 1 ] ,\tag{66}
$$

where $N _ { \mathrm { o v e r } }$ is the number of held-out over-refusal samples.

To examine whether the condition strength is associated with the sample-level efect of projection, we compute the final refusalmargin reduction

$$
\delta _ { i } = m _ { i } ( \theta _ { \mathrm { { S F T } } } ) - m _ { i } ( \theta _ { \mathrm { { O r t h o } } } ) ,\tag{67}
$$

A larger $\delta _ { i }$ indicates that refusal-orthogonal projection produces a larger reduction in the refusal preference margin relative to vanilla SFT.

For each sample, we average its condition ratio over the five epoch checkpoints:

$$
\bar { r } _ { i } = \frac { 1 } { 5 } \sum _ { e = 1 } ^ { 5 } r _ { i , e } .\tag{68}
$$

We then compute the Spearman correlation between $\bar { r } _ { i }$ and $\delta _ { i }$ . The correlation is calculated separately for each random seed and then averaged across the five seeds.

![](images/fa9fd719a6738325c25806f7b371742fe2d9c405aef5a381c978d4700411199f.jpg)  
Figure 13: Qualitative comparison on a harmful instruction. GOAL efectively preserves safety guardrails and decisively rejects the malicious request, maintaining strong safety alignment.

Table 8: Empirical verification of the suficient condition in Proposition 3. � denotes the percentage of held-out over-refusal samples satisfying $r _ { i , e } > 1$ at the end of epoch �. Median � is computed over all held-out samples and five epoch checkpoints. Spearman $\rho$ measures the association between the sample-level condition ratio averaged across epochs and the final refusal margin reduction. Results are averaged over five random seeds.
<table><tr><td>Model</td><td>S1 (%)</td><td>S2 (%)</td><td>S3 (%)</td><td>S4 (%)</td><td>S5 (%)</td><td>Median r</td><td>Spearman ρ</td></tr><tr><td>Gemma-7B-it</td><td>75.6</td><td>73.8</td><td>72.9</td><td>75.1</td><td>74.4</td><td>1.18</td><td>0.29</td></tr><tr><td>Llama-2-7B-Chat</td><td>70.9</td><td>69.5</td><td>71.1</td><td>71.3</td><td>70.2</td><td>1.12</td><td>0.24</td></tr><tr><td>Llama-2-13B-Chat</td><td>74.2</td><td>74.8</td><td>75.9</td><td>76.1</td><td>75.4</td><td>1.21</td><td>0.33</td></tr></table>

Let

As shown in Table 8, the suficient condition holds for 69.5%– 76.1% of the held-out over-refusal samples across the three models and remains stable throughout the five training epochs. The median condition ratio ranges from 1.12 to 1.21, remaining consistently greater than one. Moreover, the condition ratio is positively associated with the sample-level refusal-margin reduction, with Spearman correlations ranging from 0.24 to 0.33. These results show that the condition in Proposition 3 is not merely a theoretical possibility: it holds for a clear majority ofthe evaluated samples and is moderately associated with their benefit from refusal-orthogonal projection. At the same time, the condition does not hold uniformly, consistent with its role as a suficient rather than necessary condition.

## M Robustness under AdamW Preconditioning

The theoretical analysis of refusal-orthogonal projection is stated in the Euclidean gradient space. AdamW applies an element-wise adaptive preconditioner, which does not necessarily preserve exact Euclidean orthogonality. We therefore examine whether the projected over-refusal gradient remains largely outside the refusal subspace after AdamW preconditioning.

$$
\mathbf { u } _ { i , t } = ( \mathbf { I } - \mathbf { P } ) \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } }\tag{69}
$$

denote the projected over-refusal gradient. By construction, $\mathbf { P u } _ { i , t } =$ 0. Let

$$
\mathbf { D } _ { t } = \mathrm { d i a g } \left( { \frac { 1 } { \sqrt { \widehat { \mathbf { v } } _ { t } } + \epsilon } } \right)\tag{70}
$$

denote the diagonal second-moment preconditioner used by AdamW.

We define the residual refusal-subspace leakage after AdamW preconditioning as

$$
L _ { i , t } ^ { \mathrm { A d a m } } = \frac { \left\| \mathbf { P D } _ { t } \mathbf { u } _ { i , t } \right\| _ { 2 } } { \left\| \mathbf { D } _ { t } \mathbf { u } _ { i , t } \right\| _ { 2 } + \epsilon } .\tag{71}
$$

A smaller $L _ { i , t } ^ { \mathrm { A d a m } }$ indicates that the AdamW-preconditioned projected gradient retains little residual component in the refusal subspace.

For comparison, we measure the refusal-subspace fraction of the corresponding unprojected AdamW-preconditioned gradient:

$$
B _ { i , t } ^ { \mathrm { A d a m } } = \frac { \left\| \mathbf { P D } _ { t } \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } \right\| _ { 2 } } { \left\| \mathbf { D } _ { t } \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } \right\| _ { 2 } + \epsilon } .\tag{72}
$$

![](images/027f52896eede54362d4f2f5f44784cb443365d9652dba189ddaf9de614c75f9.jpg)  
Figure 14: Qualitative comparison on an over-refusal query containing sensitive keywords. While the baseline triggers a false refusal, GOAL successfully discerns the benign intent and delivers a safe, informative response.

The relative coupling reduction is defined as

$$
Q _ { i , t } ^ { \mathrm { A d a m } } = 1 - \frac { L _ { i , t } ^ { \mathrm { A d a m } } } { B _ { i , t } ^ { \mathrm { A d a m } } + \epsilon } .\tag{73}
$$

A larger $Q _ { i , t } ^ { \mathrm { A d a m } }$ indicates that refusal-orthogonal projection remains efective after adaptive preconditioning.

Our implementation uses zero weight decay; therefore, no addi tional weight-decay component is introduced into the parameter update. We report �<sup>Adam</sup> and $Q ^ { \mathrm { A d a m } }$ averaged over the evaluated over-refusal gradients at the end of training.

As shown in Table 7, vanilla SFT substantially reduces the refusal preference margin relative to the base model, reflecting the contribution of correctly labeled over-refusal training samples. More importantly, Ortho only consistently achieves a lower margin than SFT across all three backbone models. Since the two variants use identical training data and configurations, the additional reductions of 0.055, 0.025, and 0.032 on Gemma-7B-it, Llama-2-7B-Chat, and Llama-2-13B-Chat, respectively, provide empirical support for the incremental efect of refusal-orthogonal projection. The average margin reduction over the three models is 0.037.

Moreover, the residual refusal-subspace leakage after AdamW preconditioning ranges from 3.6% to 4.7%, with an average of 4.13%. The corresponding coupling reduction ranges from 84.1% to 88.5%, averaging 86.1%. These results indicate that adaptive preconditioning introduces only limited subspace leakage and that the intended efect of refusal-orthogonal projection is largely preserved under AdamW.

## N Controlling for Refusal-Target Artifacts

The gradients of harmful samples exhibit a concentrated, low-rank spectrum. To determine whether this structure arises from the interaction between harmful prompts and refusal supervision, rather than merely from shared refusal-targets, we conduct a matchedtarget control experiment on Llama-2-7B-Chat, following the gradientspectrum analysis in the main text.

Control groups. We construct the following three gradient groups:

• Harmful+Refusal: harmful prompts paired with their aligned refusal responses, corresponding to the original refusal-subspace construction setting.

• Benign+Refusal: benign prompts paired with the same pool of refusal responses used in Harmful+Refusal.

• Benign+Helpful: the same benign prompts paired with their original helpful responses.

Each group contains � = 200 samples. Harmful+Refusal consists of 200 harmful prompts paired with 200 aligned refusal responses. Benign+Refusal consists of 200 benign prompts paired with the same set of 200 refusal responses used in Harmful+Refusal. Benign+Helpful uses the same 200 benign prompts as Benign+Refusal, paired with their corresponding helpful responses.

The Benign+Refusal and Benign+Helpful groups therefore use exactly the same benign prompts, isolating the efect of replacing helpful targets with refusal-targets. Moreover, Harmful+Refusal and Benign+Refusal use an identical refusal-response pool. In each run, we randomly permute this pool and assign the responses to benign prompts one-to-one without replacement. We repeat this procedure over five random seeds and report the average results. All gradients are computed at the same model checkpoint, with respect to the same trainable parameters, and using the same tokennormalized SFT objective.

Gradient-spectrum analysis. For each group $c \in$ {Harmful + Refusal, Benign + Refusal, Benign + Helpful}, we stack its � per-sample gradients into

$$
\mathbf { G } _ { c } = [ \mathbf { g } _ { 1 } ^ { c } , \ldots , \mathbf { g } _ { n } ^ { c } ] ^ { \top } \in \mathbb { R } ^ { n \times d } .\tag{74}
$$

We apply singular value decomposition to each gradient matrix:

$$
\mathbf { G } _ { c } = \mathbf { U } _ { c } \boldsymbol { \Sigma } _ { c } \mathbf { V } _ { c } ^ { \top } .\tag{75}
$$

Table 9: Matched-target control for the efect of refusaltargets on gradient low-rankness on Llama-2-7B-Chat. We report the cumulative explained ratio of the top-10 singular directions before and after within-group gradient centering. Higher values indicate stronger concentration of gradient energy in a low-dimensional subspace. Harmful+Refusal and Benign+Refusal use the same refusal-response pool, while Benign+Refusal and Benign+Helpful use the same benign prompts. Results are averaged over five runs.
<table><tr><td>Gradient group</td><td>CER(10) ↑</td><td>CER(10) ↑</td></tr><tr><td>Harmful+Refusal</td><td>0.78</td><td>0.59</td></tr><tr><td>Benign+Refusal</td><td>0.57</td><td>0.32</td></tr><tr><td>Benign+Helpful</td><td>0.29</td><td>0.24</td></tr></table>

Let $\sigma _ { c , j }$ denote the �-th singular value of $\mathbf { G } _ { c }$ . We measure the cumulative explained ratio of the top-� singular directions as

$$
\mathrm { C E R } _ { c } ( k ) = \frac { \sum _ { j = 1 } ^ { k } \sigma _ { c , j } ^ { 2 } } { \sum _ { j = 1 } ^ { r _ { c } } \sigma _ { c , j } ^ { 2 } } ,\tag{76}
$$

where $r _ { c } = \mathrm { r a n k } ( \mathbf { G } _ { c } )$ . A larger CER at the same � indicates that the gradient energy is concentrated in fewer directions and therefore exhibits a stronger low-rank structure. We use � = 10, consistent with the refusal-subspace rank used in our main experiments.

Centered-gradient control. To remove the contribution of a shared mean-gradient direction, we also center the gradients within each group:

$$
\widetilde { \mathbf { G } } _ { c } = \mathbf { G } _ { c } - \mathbf { 1 } \bar { \mathbf { g } } _ { c } ^ { \top } , \qquad \bar { \mathbf { g } } _ { c } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbf { g } _ { i } ^ { c } .\tag{77}
$$

We then compute the centered cumulative explained ratio $\widetilde { \mathrm { C E R } } _ { c } ( 1 0 )$ by applying $\operatorname { E q } .$ (76) to $\widetilde { \mathbf { G } } _ { c } .$ . This centered analysis tests whether the low-rank structure persists beyond a single shared mean-gradient component induced by similar target responses.

As shown in Table 9, replacing helpful-targets with refusaltargets increases the gradient-energy concentration of benign samples: the CER increases from 0.29 for Benign+Helpful to 0.57 for Benign+Refusal. This result confirms that shared refusal supervi sion contributes to gradient concentration.

More importantly, Harmful+Refusal achieves a substantially higher CER of 0.78 than the 0.57 obtained by Benign+Refusal, even though the two groups use the same refusal-response pool. The same ordering remains after within-group gradient centering:

$$
\widetilde { C E } \mathrm { R } _ { \mathrm { H a r m f u l + R e f u s a l } } ( 1 0 ) > \widetilde { C E } \mathrm { R } _ { \mathrm { B e n i g n + R e f u s a l } } ( 1 0 ) > \widetilde { C E } \mathrm { R } _ { \mathrm { B e n i g n + H e l p f u l } } ( 1 0 ) .\tag{78}
$$

These results indicate that shared refusal-targets account for only part of the observed gradient concentration. The low-rank structure of harmful-sample gradients persists after controlling for both the refusal-target distribution and the shared mean-gradient direction, supporting an additional interaction between harmful prompts and refusal supervision.

## O Human Evaluation and Response-Quality Analysis

The primary evaluation in our experiments uses GPT-5.1 to determine whether a model response constitutes an unwarranted refusal. Although this evaluation directly measures over-refusal, a non-refusal response is not necessarily correct, helpful, or complete. We therefore conduct an additional response-quality evaluation using both GPT-5.1 and human annotators. This analysis examines whether the reduction in refusal achieved by GOAL corresponds to genuinely useful responses and whether the GPT-5.1 evaluation is broadly consistent with human judgments.

## O.1 Evaluation Setup

We randomly sample $N _ { \mathrm { h u m a n } } = 2 0 0$ prompts from the held-out over-refusal evaluation sets. For each prompt, we collect responses generated by vanilla SFT, ACTOR, which is the strongest competing method on this backbone in our main experiments, and GOAL. All methods use the same decoding configuration, including the temperature, maximum generation length, and sampling parameters.

To reduce potential presentation bias, model identities and method names are hidden during evaluation, and the responses are randomly shufled for each prompt. The same prompt–response pairs are evaluated by GPT-5.1 and human annotators, enabling a direct comparison between the two evaluation procedures.

Each response is evaluated along the following dimensions:

• Unwarranted refusal: whether the model refuses, evades, or unnecessarily avoids answering a benign request.

• Helpfulness: the extent to which the response addresses the user’s request and provides useful information, measured on a five-point Likert scale.

• Correctness: the factual and logical correctness of the response, measured on a five-point Likert scale.

• Completeness: whether the response suficiently addresses the important aspects of the request, measured on a fivepoint Likert scale.

We additionally define Useful Compliance as

$$
\mathrm { U C } _ { i } = \mathbb { I } \left[ r _ { i } = 0 \ \wedge \ h _ { i } \ge 4 \ \wedge \ c _ { i } \ge 4 \right] ,\tag{79}
$$

where $r _ { i }$ indicates an unwarranted refusal, and $h _ { i }$ and $c _ { i }$ denote the helpfulness and correctness scores, respectively. The overall Useful Compliance rate is defined as

$$
\mathrm { U C } = \frac { 1 } { N _ { \mathrm { h u m a n } } } \sum _ { i = 1 } ^ { N _ { \mathrm { h u m a n } } } \mathrm { U C } _ { i } .\tag{80}
$$

This metric prevents a response from being counted as successful merely because it does not explicitly refuse. A high Useful Compliance rate requires the response to be both non-refusing and substantively useful.

## O.2 GPT-5.1 Quality Evaluation

In addition to the binary refusal judgment used in the main experiments, we ask GPT-5.1 to evaluate the helpfulness, correctness, and completeness of each response. The evaluator is provided with the original prompt and the corresponding model response but is not given the identity of the evaluated method.

The GPT-5.1 evaluator first determines whether the response contains an unwarranted refusal and then assigns an integer score from 1 to 5 for each quality dimension. A score of 1 indicates that the response is severely deficient, whereas a score of 5 indicates that it fully satisfies the corresponding criterion. We use a fixed evaluation prompt and deterministic decoding to improve evaluation consistency.

Table 10: GPT-5.1 and human evaluation of response quality on the same 200 held-out over-refusal prompts using Llama-2- 7B-Chat. Helpfulness, correctness, and completeness are rated on a five-point scale. Useful Compliance is the percentage of responses that do not exhibit unwarranted refusal and receive both helpfulness and correctness scores of at least 4. Agreement denotes Fleiss’ � for refusal and ordinal Krippendorf’s � for quality ratings.
<table><tr><td>Evaluator</td><td>Method</td><td>Refusal Rate ↓</td><td>Helpfulness ↑</td><td>Correctness ↑</td><td>Completeness ↑</td><td>Useful Compliance ↑</td></tr><tr><td rowspan="3">GPT-5.1</td><td>SFT</td><td>0.195</td><td>3.83</td><td>3.88</td><td>3.76</td><td>68.5%</td></tr><tr><td>ACTOR</td><td>0.145</td><td>4.01</td><td>4.03</td><td>3.84</td><td>72.5%</td></tr><tr><td>GOAL</td><td>0.090</td><td>4.27</td><td>4.23</td><td>4.18</td><td>78.5%</td></tr><tr><td rowspan="4">Human</td><td>SFT</td><td>0.215</td><td>3.76</td><td>3.92</td><td>3.69</td><td>66.5%</td></tr><tr><td>ACTOR</td><td>0.150</td><td>3.84</td><td>3.88</td><td>3.78</td><td>70.0%</td></tr><tr><td>GOAL</td><td>0.100</td><td>4.12</td><td>4.17</td><td>4.03</td><td>76.0%</td></tr><tr><td>Agreement</td><td>κ = 0.71</td><td>α = 0.77</td><td>α = 0.81</td><td>α = 0.75</td><td></td></tr></table>

## O.3 Human Evaluation

Each prompt–response pair is independently evaluated by three human annotators. The annotators are given the same definitions of unwarranted refusal, helpfulness, correctness, and completeness as the GPT-5.1 evaluator. They are blind to both the method identity and the GPT-5.1 judgment.

Non-refusal requires unanimous agreement among the three annotators; otherwise, the response is counted as refusal, while quality ratings are averaged. Across all 600 prompt–response pairs, agreement is substantial (Fleiss’ � = 0.71; ordinal Krippendorf’s $\alpha = 0 . 7 5 \substack { - 0 . 8 1 } )$ . The GPT-5.1 and human evaluations are performed on exactly the same responses, allowing us to examine whether the conclusions obtained using the automated evaluator are consistent with those obtained from human assessment.

## O.4 Results

As shown in Table 10, GOAL consistently achieves the lowest unwarranted-refusal rate under both GPT-5.1 and human evaluation. According to GPT-5.1, GOAL reduces the refusal rate from 19.5% for SFT and 14.5% for ACTOR to 9.0%. Human annotators report the same trend, with refusal rates of 21.5%, 15.0%, and 10.0% for SFT, ACTOR, and GOAL, respectively.

More importantly, the reduction in refusal is accompanied by improved response quality. GOAL obtains the highest helpfulness, correctness, and completeness scores under both evaluation procedures. Under human evaluation, GOAL achieves a Useful Compliance rate of 76.0%, compared with 66.5% for SFT and 70.0% for ACTOR. This indicates that GOAL does not reduce over-refusal merely by replacing refusals with perfunctory or low-quality responses.

The GPT-5.1 and human evaluations also produce consistent method-level conclusions: both evaluators rank GOAL first across all reported metrics, and the corresponding refusal rates and qual ity scores follow similar trends. These results provide additional evidence for the reliability of the GPT-5.1-based evaluation used in our main experiments while confirming that the improvement in over-refusal is accompanied by genuinely useful responses.

## P Multi-Seed Evaluation and Statistical Significance

To examine whether the improvements in over-refusal mitigation are robust to training randomness, we repeat the experiments using five diferent random seeds. We evaluate SFT, ACTOR, and GOAL on five over-refusal benchmarks and two safety benchmarks using Gemma-7B-it, Llama-2-7B-Chat, and Llama-2-13B-Chat. These methods are selected because SFT provides the direct fine-tuning baseline, while ACTOR is the strongest competing method in terms of average over-refusal performance in our main comparison.

For each method and backbone, the five runs use the same set of random seeds, training data, data order, and evaluation prompts. All hyperparameters and decoding configurations are held fixed across methods. We report the mean and sample standard deviation across the five runs. Over-refusal performance is measured by compliance rate (C.R.) on XSTest, OR-Bench-Hard, PHTest, OKTest, and CoCoNot. Safety preservation is measured by safety score (S.S.) on JailbreakBench (JBB) and STL.

Because diferent methods are evaluated on the same prompts under matched random seeds, we assess the statistical significance of the over-refusal results using hierarchical paired bootstrap resampling. For each bootstrap replicate, we first resample the random seeds with replacement and then resample the evaluation prompts within each selected seed. We compute the paired compliance-rate diference between GOAL and ACTOR using 10,000 bootstrap replicates. We apply the Holm–Bonferroni correction across the five over-refusal benchmarks within each backbone and regard a diference as statistically significant when the corrected �-value is below 0.05. For the two safety benchmarks, we report only the mean and sample standard deviation because these experiments are intended to verify that GOAL maintains safety performance rather than to claim a statistically significant safety improvement.

As shown in Table 11, GOAL maintains its overall advantage across random seeds and backbone models. On Gemma-7B-it, GOAL achieves the highest mean compliance rate on four of the five over-refusal benchmarks, while ACTOR performs slightly better on ORH. On Llama-2-7B-Chat, GOAL achieves the highest mean performance on all five over-refusal benchmarks, although its improvement over ACTOR on PHT is marginal. On Llama-2-13B-Chat, GOAL consistently outperforms both SFT and ACTOR across all five over-refusal benchmarks.

Table 11: Multi-seed over-refusal and safety results on three backbone models. Results are reported as mean ± sample standard deviation across five random seeds. C.R. denotes compliance rate, and S.S. denotes safety score. Higher values indicate better performance. <sup>∗</sup> indicates that GOAL significantly outperforms ACTOR on the corresponding over-refusal benchmark under hierarchical paired bootstrap resampling with 10,000 replicates $( p < 0 . 0 5$ after Holm–Bonferroni correction). Statistical significance is not assessed for the two safety benchmarks.
<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Method</td><td colspan="5">Over-refusal C.R. ↑</td><td colspan="2">Safety S.S. ↑</td></tr><tr><td>XST</td><td>ORH</td><td>PHT</td><td>OKT</td><td>CoCo</td><td>JBB</td><td>STL</td></tr><tr><td rowspan="3">Gemma-7B-it</td><td>SFT</td><td>0.8098±0.0058</td><td>0.5820±0.0045</td><td>0.8491±0.0055</td><td>0.7420±0.0090</td><td>0.8153±0.0083</td><td>0.9640±0.0055</td><td>0.9771±0.0046</td></tr><tr><td>ACTOR</td><td>0.7876±0.0085</td><td>0.7046±0.0053</td><td>0.8933±0.0076</td><td>0.7760±0.0055</td><td>0.8169±0.0071</td><td>0.8800±0.0071</td><td>0.9593±0.0049</td></tr><tr><td>GOAL</td><td>0.9218±0.0074*</td><td>0.7004±0.0046</td><td>0.9047±0.0072</td><td>0.8767±0.0062</td><td>0.8828±0.0051*</td><td>0.9740±0.0055</td><td>0.9891±0.0040</td></tr><tr><td rowspan="3">Llama-2-7B-Chat</td><td>SFT</td><td>0.8978±0.0070</td><td>0.6600±0.0073</td><td>0.9105±0.0063</td><td>0.9260±0.0086</td><td>0.8834±0.0051</td><td>0.9620±0.0045</td><td>0.9791±0.0046</td></tr><tr><td>ACTOR</td><td>0.9289±0.0083</td><td>0.7494±0.0044</td><td>0.9565±0.0063</td><td>0.9353±0.0096</td><td>0.8464±0.0082</td><td>0.9500±0.0071</td><td>0.9771±0.0065</td></tr><tr><td>GOAL</td><td>0.9502±0.0073*</td><td>0.7638±0.0069</td><td>0.9573±0.0044</td><td>0.9633±0.0041</td><td>0.9029±0.0090*</td><td>0.9640±0.0055</td><td>0.9872±0.0040</td></tr><tr><td rowspan="3">Llama-2-13B-Chat</td><td>SFT</td><td>0.8898±0.0049</td><td>0.5849±0.0075</td><td>0.9197±0.0075</td><td>0.9373±0.0086</td><td>0.8443±0.0049</td><td>0.9700±0.0071</td><td>0.9977±0.0025</td></tr><tr><td>ACTOR</td><td>0.8729±0.0074</td><td>0.6386±0.0060</td><td>0.9551±0.0082</td><td>0.9047±0.0061</td><td>0.8195±0.0085</td><td>0.9540±0.0055</td><td>0.9957±0.0037</td></tr><tr><td>GOAL</td><td>0.9271±0.0081*</td><td>0.7250±0.0084</td><td>0.9652±0.0084</td><td>0.9753±0.0069</td><td>0.8876±0.0040*</td><td>0.9740±0.0055</td><td>0.9984±0.0016</td></tr></table>

After Holm–Bonferroni correction, GOAL significantly outperforms ACTOR on XST, PHT, OKT, and CoCo using Gemma-7B-it; on XST, ORH, OKT, and CoCo using Llama-2-7B-Chat; and on all five over-refusal benchmarks using Llama-2-13B-Chat. The diference is not statistically significant on PHT using Llama-2-7B-Chat, while ACTOR obtains a slightly higher mean compliance rate on ORH using Gemma-7B-it.

The safety results further indicate that the improvements in overrefusal mitigation do not come at the cost of a noticeable reduction in safety. GOAL achieves the highest mean safety score on both JBB and STL across all three backbone models. In particular, GOAL maintains safety scores above 0.96 on JBB and above 0.98 on STL for all three backbones. Together, these results indicate that GOAL provides stable over-refusal improvements while preserving safety performance across random seeds.

## Q Comparison with Alternative Gradient Intervention Strategies

To determine whether the improvement of GOAL can be reproduced by generic gradient surgery or rank-one gradient removal, we compare the proposed low-rank projection with two alternative gradient intervention strategies: PCGrad and harmful meangradient orthogonalization. This comparison examines whether explicitly modeling the low-rank structure of harmful per-sample gradients is necessary for mitigating over-refusal.

PCGrad. PCGrad [44] is a general multi-objective gradient surgery method designed to resolve negative interference between task gradients. We treat the harmful, benign, and over-refusal objectives as three training objectives. For two objective gradients ${ \bf g } _ { a }$ and $\mathbf { g } _ { b } ,$ when $\langle { \bf g } _ { a } , { \bf g } _ { b } \rangle < 0 ,$

PCGrad projects ${ \bf g } _ { a }$ onto the orthogonal complement of g<sub>�</sub>:

$$
\mathbf { g } _ { a }  \mathbf { g } _ { a } - \frac {  \mathbf { g } _ { a } , \mathbf { g } _ { b }  } { \| \mathbf { g } _ { b } \| _ { 2 } ^ { 2 } + \epsilon } \mathbf { g } _ { b } .\tag{81}
$$

We follow the standard PCGrad procedure and randomize the order of pairwise gradient projections at each optimization step. Unlike GOAL, PCGrad operates on aggregated objective gradients and applies projection only when conventional negative gradient interference is detected.

Harmful mean-gradient orthogonalization. We additionally consider a rank-one projection baseline constructed from the mean harmful gradient. Let

$$
\bar { \mathbf { g } } _ { t } ^ { \mathrm { m e a n } } = \frac { 1 } { B _ { \mathrm { h a r m } } } \sum _ { j = 1 } ^ { B _ { \mathrm { h a r m } } } \mathbf { g } _ { j , t } ^ { \mathrm { h a r m } }\tag{82}
$$

denote the mean harmful gradient at optimization step �. Each over-refusal helpful-target gradient is projected against this single direction:

$$
\tilde { \mathbf { g } } _ { i , t } ^ { \mathrm { M e a n } } = \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } - \frac { \left. \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } } , \bar { \mathbf { g } } _ { t } ^ { \mathrm { m e a n } } \right. } { \left\| \bar { \mathbf { g } } _ { t } ^ { \mathrm { m e a n } } \right\| _ { 2 } ^ { 2 } + \epsilon } \bar { \mathbf { g } } _ { t } ^ { \mathrm { m e a n } } .\tag{83}
$$

This baseline removes only the rank-one direction represented by the mean harmful gradient. By contrast, Ortho only constructs a low-rank projector $\mathbf { P } = \mathbf { V } _ { k } \mathbf { V } _ { k } ^ { \top }$ from the top-� right singular vectors of the harmful per-sample gradient matrix and applies $\tilde { \mathbf { g } } _ { i , t } ^ { \mathrm { S V D } } ~ =$ $( \mathbf { I } - \mathbf { P } ) \mathbf { g } _ { i , t } ^ { \mathrm { o v e r } }$

All methods use identical training data, trainable parameters, target layers, optimization hyperparameters, and decoding config urations. Neither Harmful Mean Ortho nor Ortho only uses the contrastive objective, ensuring that their comparison isolates the efect of subspace construction. PCGrad is implemented using the same three training objectives and trainable parameters.

As shown in Table 12, PCGrad is less efective than Ortho only across all five over-refusal benchmarks. This result is consistent with the fact that PCGrad is triggered by negative interference between aggregated objective gradients, whereas the refusal-subspace coupling studied in this work does not necessarily appear as a conventional negative gradient conflict.

Harmful Mean Ortho improves the average compliance rate over PCGrad, but its performance varies across individual benchmarks. In particular, it performs slightly worse than PCGrad on PHT and OKT. More importantly, Ortho only consistently outperforms Harmful Mean Ortho across all five over-refusal benchmarks, increasing the average compliance rate from 0.8616 to 0.8738. This result indicates that the refusal-related gradient structure is not fully captured by a single mean direction and is better represented by multiple dominant singular directions.

Table 12: Comparison with alternative gradient intervention strategies on Llama-2-7B-Chat. PCGrad applies generic conflicttriggered gradient surgery across the harmful, benign, and over-refusal objectives. Harmful Mean Ortho projects over-refusal gradients against the rank-one mean harmful-gradient direction, while Ortho only projects them against the low-rank subspace extracted from harmful per-sample gradients. None of the variants uses the contrastive objective. All methods use identical training data, trainable parameters, target layers, and optimization settings.
<table><tr><td>Method</td><td>XST C.R. ↑</td><td>ORH C.R. ↑</td><td>PHT C.R. ↑</td><td>OKT C.R. ↑</td><td>CoCo C.R. ↑</td><td>JBB S.S. ↑</td><td>STL S.S. ↑</td></tr><tr><td>PCGrad</td><td>0.8933</td><td>0.6698</td><td>0.8904</td><td>0.9300</td><td>0.8786</td><td>0.96</td><td>0.9767</td></tr><tr><td>Harmful Mean Ortho</td><td>0.9111</td><td>0.6990</td><td>0.8899</td><td>0.9267</td><td>0.8813</td><td>0.95</td><td>0.9806</td></tr><tr><td>Ortho only</td><td>0.9200</td><td>0.7195</td><td>0.9001</td><td>0.9400</td><td>0.8892</td><td>0.95</td><td>0.9806</td></tr></table>

Table 13: Efect of harmful-gradient downweighting on Llama-2-7B-Chat. $\lambda _ { \mathrm { h a r m } }$ denotes the weight assigned to the harmful sample training objective, with $\lambda _ { \mathrm { h a r m } } = 1$ corresponding to standard SFT. Ortho only preserves the original harmful objective and projects over-refusal gradients against the low-rank refusal subspace. None of the variants uses the contrastive objective. All other training and evaluation configurations are held fixed.
<table><tr><td>Method</td><td>XST C.R. ↑</td><td>ORH C.R. ↑</td><td>PHT C.R. ↑</td><td>OKT C.R. ↑</td><td>CoCo C.R. ↑</td><td>JBB S.S. ↑</td><td>STL S.S. ↑</td></tr><tr><td> $\mathbf { S F T } \left( \lambda _ { \mathrm { h a r m } } = 1 . 0 0 \right)$ </td><td>0.8978</td><td>0.6604</td><td>0.9098</td><td>0.9233</td><td>0.8813</td><td>0.96</td><td>0.9787</td></tr><tr><td>Harmful DW  $( \lambda _ { \mathrm { h a r m } } = 0 . 7 5 )$ </td><td>0.9022</td><td>0.6761</td><td>0.9103</td><td>0.9600</td><td>0.8945</td><td>0.91</td><td>0.9360</td></tr><tr><td>Harmful DW  $( \lambda _ { \mathrm { h a r m } } = 0 . 5 0 )$ </td><td>0.8978</td><td>0.6856</td><td>0.9211</td><td>0.9667</td><td>0.9050</td><td>0.86</td><td>0.8779</td></tr><tr><td>Ortho only</td><td>0.9200</td><td>0.7195</td><td>0.9001</td><td>0.9400</td><td>0.8892</td><td>0.95</td><td>0.9806</td></tr></table>

The three strategies achieve comparable safety performance. Ortho only obtains a slightly lower JBB safety score than PCGrad (0.95 versus 0.96), while achieving a higher STL safety score (0.9806 versus 0.9767). Together with the rank sensitivity analysis, these results support the use ofan SVD-derived low-rank refusal subspace rather than generic conflict-triggered gradient surgery or rank-one mean-gradient removal.

## R Efect of Harmful-Gradient Downweighting

As discussed in Section 3.1, a natural strategy for reducing the influence of harmful gradients is to downweight the harmful-sample training objective. To examine this strategy, we scale the harmful objective by $\lambda _ { \mathrm { h a r m } }$ while leaving the benign and over-refusal objec tives unchanged. We evaluate � ∈ {0.75, 0.5}, with $\lambda _ { \mathrm { h a r m } } = 1$ corresponding to standard SFT. All other training and evaluation configurations are held fixed.

As shown in Table 13, downweighting harmful gradients improves compliance over standard SFT on most over-refusal benchmarks, but introduces a clear safety trade-of. In particular, reducing $\lambda _ { \mathrm { h a r m } }$ to 0.5 achieves strong compliance on PHT, OKT, and CoCo, but decreases the JBB and STL safety scores from 0.96 and 0.9787 to 0.86 and 0.8779, respectively. In contrast, Ortho only achieves comparable overall over-refusal mitigation while maintaining safety scores of 0.95 and 0.9806. These results support the rationality and efectiveness of our method.

## S Evaluation on a Recent Model Family

Motivation. Our main experiments use Gemma-7B-it and Llama-2-Chat models to enable controlled comparisons with prior overrefusal mitigation methods. To examine whether GOAL generalizes to a more recent model family, we additionally evaluate it on Qwen3- 8B [42]. Qwen3-8B is an instruction-tuned model with 8.2 billion parameters and 36 transformer layers. It difers from the models used in our main experiments in its architecture, tokenizer, and post-training recipe, thereby providing an out-of-family evaluation of over-refusal mitigation.

Experimental setup. We follow the same training-data construction and evaluation protocol described in Section 4. All fine-tuned methods use the same training examples, trainable parameters, and decoding configuration.

We evaluate over-refusal using Compliance Rate (C.R.) on XSTest, OR-Bench-Hard, PHTest, OKTest, and CoCoNot. To verify that improved compliance does not compromise safety, we report Safety Score (S.S.) on JBB and STL, which contain 100 and 516 harmful prompts, respectively. We additionally evaluate general-capability preservation using accuracy on MMLU, OpenBookQA, PIQA, and ARC-Challenge. All reported results are averaged across five independent runs.

Results. As shown in Table 14, GOAL achieves the highest mean compliance rate on four of the five over-refusal benchmarks and ranks second on ORH. Compared with SFT, GOAL improves the compliance rate by 1.78, 0.16, 2.34, 2.33, and 1.06 percentage points on XST, ORH, PHT, OKT, and CoCo, respectively. The largest improvement is observed on PHT, where the compliance rate increases from 0.9459 to 0.9693.

The improved compliance does not come at the cost of substantially weakened safety. On JBB, GOAL achieves a Safety Score of 0.98, tying with SFT for the best result. On STL, GOAL obtains a Safety Score of 0.9953, remaining within 0.0004 of the base model and 0.0008 of SFT, while outperforming ACTOR by 0.0085.

GOAL also largely preserves general capabilities. It achieves accuracies of 0.7597, 0.5600, 0.8107, and 0.5121 on MMLU, OpenBookQA, PIQA, and ARC-Challenge, respectively. Although its performance is slightly lower than that of the base model and SFT on the first three benchmarks, the diferences remain small, while GOAL consistently outperforms ACTOR across all four benchmarks.

Table 14: Performance comparison on Qwen3-8B across over-refusal, safety, and general-capability benchmarks. C.R. denotes Compliance Rate, S.S. denotes Safety Score, and Acc denotes accuracy. All results are averaged across five independent runs. The best mean result in each column is shown in bold, and the second-best is underlined.
<table><tr><td></td><td colspan="5">Over-refusal C.R. ↑</td><td colspan="2">Safety S.S. ↑</td><td colspan="4">General Capability Acc ↑</td></tr><tr><td>Method</td><td>XST</td><td>ORH</td><td>PHT</td><td>OKT</td><td>CoCo</td><td>JBB</td><td>STL</td><td>MMLU</td><td>OBQA</td><td>PIQA</td><td>ARC-C</td></tr><tr><td>Base</td><td>0.9600</td><td>0.8437</td><td>0.9303</td><td>0.9600</td><td>0.9393</td><td>0.97</td><td>0.9957</td><td>0.7683</td><td>0.572</td><td>0.8123</td><td>0.5119</td></tr><tr><td>SFT</td><td>0.9689</td><td>0.8597</td><td>0.9459</td><td>0.9667</td><td>0.9472</td><td>0.98</td><td>0.9961</td><td>0.7675</td><td>0.567</td><td>0.8116</td><td>0.5102</td></tr><tr><td>SCANS</td><td>0.9778</td><td>0.8512</td><td>0.9421</td><td>0.9633</td><td>0.9499</td><td>0.97</td><td>0.9845</td><td>0.7657</td><td>0.569</td><td>0.8108</td><td>0.5108</td></tr><tr><td>Surgical</td><td>0.9644</td><td>0.8575</td><td>0.9359</td><td>0.9667</td><td>0.9472</td><td>0.97</td><td>0.9849</td><td>0.7589</td><td>0.565</td><td>0.8102</td><td>0.5102</td></tr><tr><td>DCR</td><td>0.9644</td><td>0.8463</td><td>0.9365</td><td>0.9667</td><td>0.9409</td><td>0.95</td><td>0.9709</td><td>0.7442</td><td>0.544</td><td>0.8002</td><td>0.4915</td></tr><tr><td>ACTOR</td><td>0.9733</td><td>0.8629</td><td>0.9446</td><td>0.9767</td><td>0.9530</td><td>0.97</td><td>0.9868</td><td>0.7521</td><td>0.557</td><td>0.8103</td><td>0.5078</td></tr><tr><td>GOAL</td><td>0.9867</td><td>0.8613</td><td>0.9693</td><td>0.9900</td><td>0.9578</td><td>0.98</td><td>0.9953</td><td>0.7597</td><td>0.560</td><td>0.8107</td><td>0.5121</td></tr></table>

Table 15: Quantitative separation between harmful and overrefusal representations in Llama-2-7B-Chat. Values in brackets denote 95% bootstrap confidence intervals.
<table><tr><td>Layer</td><td>Probe AUC ↑</td><td> $R ^ { \left( \ell \right) } \uparrow$ </td></tr><tr><td>1</td><td>0.612 [0.574, 0.650]</td><td>0.052 [0.041, 0.065]</td></tr><tr><td>5</td><td>0.684 [0.649, 0.718]</td><td>0.087 [0.071, 0.105]</td></tr><tr><td>9</td><td>0.761 [0.742, 0.791]</td><td>0.219 [0.193, 0.248]</td></tr><tr><td>13</td><td>0.858 [0.837, 0.871]</td><td>0.357 [0.345, 0.372]</td></tr><tr><td>17</td><td>0.737 [0.712, 0.761]</td><td>0.311 [0.294, 0.321]</td></tr><tr><td>21</td><td>0.716 [0.683, 0.718]</td><td>0.288 [0.277, 0.296]</td></tr><tr><td>25</td><td>0.686 [0.675, 0.705]</td><td>0.259 [0.248, 0.269]</td></tr><tr><td>29</td><td>0.678 [0.663, 0.691]</td><td>0.207 [0.194, 0.218]</td></tr></table>

Overall, these results demonstrate that GOAL efectively mitigates over-refusal on Qwen3-8B while largely preserving safety and general capabilities, suggesting that its efectiveness generalizes beyond Llama-2 and Gemma to a more recent model family.

## T Quantitative Analysis of Representation Separation

Layer-wise separability. To verify that the PCA patterns in Figure 1 are not artifacts of two-dimensional projection, we quantitatively measure the separation between harmful and over-refusal representations in the original hidden space. We use a balanced held out set containing 200 harmful and 200 over-refusal prompts. At each layer, we extract and $\ell _ { 2 } \cdot$ -normalize the last-token hidden states. We report: (i) the AUC of an �<sub>2</sub>-regularized logistic-regression probe under stratified five-fold cross-validation; and (ii) the betweenclass/within-class separation ratio

$$
R ^ { ( \ell ) } = \frac { \left\| \pmb { \mu } _ { \mathrm { h a r m } } ^ { ( \ell ) } - \pmb { \mu } _ { \mathrm { o v e r } } ^ { ( \ell ) } \right\| _ { 2 } ^ { 2 } } { \frac { 1 } { 2 } \left( S _ { \mathrm { h a r m } } ^ { ( \ell ) } + S _ { \mathrm { o v e r } } ^ { ( \ell ) } \right) + \epsilon } ,\tag{84}
$$

where $\pmb { \mu } _ { c } ^ { ( \ell ) }$ is the class centroid and $S _ { c } ^ { ( \ell ) }$ is the average squared distance to that centroid. Higher AUC and $R ^ { ( \ell ) }$ indicate stronger separation. We report 95% confidence intervals obtained from 1,000 stratified prompt-level bootstrap replicates.

Results. As shown in Table 15, harmful and over-refusal representations are most distinguishable at layer 13, where the linear probe achieves an AUC of 0.858 (95% CI: [0.837, 0.871]) and the separation ratio reaches 0.357 (95% CI: [0.345, 0.372]). Both metrics decrease consistently in subsequent layers, with the probe AUC and separation ratio falling to 0.678 and 0.207, respectively, at layer 29. Nevertheless, the probe AUC remains above the chance level of 0.5, indicating that the distinction is weakened rather than entirely eliminated. Overall, these results quantitatively corroborate the PCA observation: harmful and over-refusal representations are most separable at an intermediate layer and become progressively less separable in deeper layers.

Subspace-specificity controls. We further examine whether the benefit of gradient projection is specific to the harmful-gradient subspace. We compare Harmful Ortho with two rank-matched controls: Random Ortho projects against a random rank-� orthonormal subspace, while Benign Ortho projects against the top-� singular directions extracted from benign-sample gradients. None of these variants uses the contrastive objective, and all other settings are identical.

Table 16: Rank-matched subspace controls on Llama-2-7B-Chat. C.R. denotes Compliance Rate. The best result is shown in bold and the second-best result is underlined.
<table><tr><td>Method</td><td>XST</td><td>ORH</td><td>PHT</td><td>OKT</td><td>CoCo</td></tr><tr><td>SFT</td><td>0.8978</td><td>0.6604</td><td>0.9098</td><td>0.9233</td><td>0.8813</td></tr><tr><td>Random Ortho</td><td>0.8489</td><td>0.5965</td><td>0.8679</td><td>0.8833</td><td>0.8575</td></tr><tr><td>Benign Ortho</td><td>0.7956</td><td>0.5122</td><td>0.7861</td><td>0.7867</td><td>0.7731</td></tr><tr><td>Harmful Ortho</td><td>0.9200</td><td>0.7195</td><td>0.9001</td><td>0.9400</td><td>0.8892</td></tr></table>

Control results. As reported in Table 16, Harmful Ortho achieves the best compliance rate on four ofthe five over-refusal benchmarks and remains competitive with SFT on PHT. In contrast, both Random Ortho and Benign Ortho consistently underperform SFT across all five benchmarks, with Benign Ortho showing the largest degradation. These results demonstrate that generic gradient projection does not inherently mitigate over-refusal. Instead, the efectiveness of orthogonal projection depends on identifying the dominant subspace induced by harmful samples with refusal supervision.