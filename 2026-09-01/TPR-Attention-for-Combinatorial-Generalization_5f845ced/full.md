# TPR-Attention for Combinatorial Generalization

Melisa Civelekoˇglu & Isabeau Pr´emont-Schwarz

Universit´e Laval

Institut intelligence et donn´ees

Mila – Qu´ebec Artificial Intelligence Institute

melisa.civelekoglu.1@ulaval.ca, isabeau.premont-schwarz@ift.ulaval.ca

## Abst<sub>r</sub>act

Systematic generalization remains a significant challenge in deep learning. In particular, combinatorial generalization – generalizing to new configurations of known factors of variation – is efortless for humans but difficult for standard neural architectures that rely on statistical correlations rather than explicit structural representations. We introduce a new architectural component that embeds structured inductive bias into deep learning: an attention mechanism operating over tensor-product representations (TPRs). Through controlled experiments on compositional tasks, we show that this TPR-Attention mechanism outperforms existing architectural components in combinatorial generalization. These results highlight the value of integrating explicit compositional structure into neural attention and point toward a promising path for models capable of systematic generalization.

## 1 Introduction

A natural ability of human intelligence is the recombination of known factors of variation into novel structures, known as combinatorial generalization (Cole et al., 2013; Smolensky et al., 2022). Rather than memorizing correlations, it relies on the compositionality of many real-world decision-making problems. For example, a human who has learned the meanings of “throw”, “catch” and “throw fast” could efortlessly understand the meaning of “catch fast”.

Traditional deep networks, on the other hand, struggle with such seemingly straightforward combinations (Lake & Baroni, 2018) as they rely on statistical patterns rather than compositional rules. A model trained on red circles and green squares, for example, should also be able to generate red squares and green circles. Despite remarkable successes, deep networks remain limited because they learn statistical co-occurrences rather than underlying compositional structure and thus are still far from achieving human-level generalization.

To address this limitation, methods for encoding symbolic structure, such as Tensor Product Representations (TPRs) (Smolensky, 1990), have been proposed. In this work, we make two contributions. First, we introduce a new neural architectural component, TPR-Attention, which uses a structured object-centric representation built from the binding of an object’s factors of variation, and whose attention mechanism explicitly performs binding and unbinding operations on these objects. Second, we provide empirical evidence that TPR-Attention achieves better combinatorial generalization than existing architectural components, particularly in the dificult setting where factors of variation interact(Montero et al., 2024).

## 2 Related Work

Many previous works have focused on learning disentangled representations (Mathieu et al., 2019; Wang et al., 2024), i.e., independent factors of variation that can be recombined to produce novel concept compositions. One motivation for disentanglement is that separating these factors may improve generalization, either by capturing inherent compositional structure (Duan et al., 2020) or by isolating causal variables (Sch¨olkopf et al., 2021).

![](images/155bba7f0a54ef4ae776cf00b0ad0a5c6eeba0fcfe0b9b6e651c7a3a141851e6.jpg)  
Figure 1: Illustration of the object matching and property extraction phases of TPR-Attention: Objects $O _ { 1 } , O _ { 2 } ,$ , and $O _ { 3 }$ are stored in memory $\begin{array} { r } { \bar { \mathbf { M } } = \overline { { \sum _ { t } \mathbf { O } _ { t } } } \otimes \mathbf { O } _ { t } } \end{array}$ . A query for a filler match $f _ { m } = f _ { \mathrm { y e l l o w } }$ and role match $r _ { m } = r _ { \mathrm { c o l } }$ retrieves the yellow triangle $O _ { 2 } .$ . Finally, a query for a target role $r _ { t } = r _ { \mathrm { s h a p e } }$ retrieves the triangle shape from the matched $O _ { 2 }$

However, the relationship between disentanglement and compositional generalization remains unclear. Several studies report limited evidence that explicit factor decoupling improves generalization (Xu et al., 2022; Montero et al., 2024). In particular, Montero et al. (2024) show that models can achieve high disentanglement by mapping perceptual inputs to factors that remain invariant across examples, yet still fail to generalize compositionally when factors interact. In this paper, we focus specifically on this hard case of interacting factors of variation, where existing approaches are known to struggle.

There exist explicit symbolic structures such as TPRs and other Vector Symbolic Architectures (Kleyko et al., 2022) which provide role–filler bindings that separate feature from content. In this paper, we provide transformation operations to be used between these structures with a focus on combinatorial generalization.

## 3 Method

## 3.1 Representation Space

Objects are represented by TPRs, which provide a vector-space embedding of symbolic structures. Each object is composed of roles, which specify an attribute slot of an object (e.g., shape, colour, position), and fillers, which are the values occupying each role (e.g., square, red, center ).

We represent each role by a role vector r and each filler by a filler vector f. An object is defined as the tensor O, where the tensor product is used as the binding operator, so that a role-filler relation is described by $r \otimes f .$ We assume orthogonality of role vectors, i.e., $r _ { i } ^ { \top } r _ { j } = \delta _ { i j }$ , to prevent interference across fillers.

The final TPR form of an object is the superposition of all its role–filler bindings:

$$
\mathbf { 0 } _ { i } = \sum _ { j } r _ { j } \otimes f _ { j } ^ { i } .
$$

## 3.2 TPR-Attention Mechanism

There are three stages in TPR-Attention: (i) generate a structured role-filler query and match relevant objects, (ii) extract a target property from each matching object and (iii) transform each extracted property and re-bind it to a new role per attention head.

To match and extract at the same index t, we define a TPR-based structured associative memory mechanism (TPR-SAM) (see Appendix C for a generalization):

$$
\mathrm { T P R \mathrm { - } S A M } ( \mathbf { 0 } _ { t } ) : = \mathbf { 0 } _ { t } \otimes \mathbf { 0 } _ { t } .
$$

Then, TPR-Attention mechanism operates on a TPR-SAM memory, represented as the superposition of all past transformed objects:

$$
\mathbf { M } _ { t } \gets \mathbf { M } _ { t - 1 } + \mathbf { O } _ { t } \otimes \mathbf { O } _ { t } = \sum _ { s = 1 } ^ { t } \mathbf { O } _ { s } \otimes \mathbf { O } _ { s } .
$$

## 3.2.1 Object Matching

Given a role $\boldsymbol { r } _ { m }$ and filler $f _ { m }$ to match, each object is weighted by a similarity score between $f _ { m }$ and the filler bound to $\boldsymbol { r } _ { m }$ (see Appendix A for notation and B.1 for a detailed derivation).

$$
\mathrm { M } _ { \mathrm { o b j } } ( \mathbf { M } , { r } _ { m } , { f } _ { m } ) : = \overbrace { \sum _ { \binom { m } { m } } \left( \int _ { m } \right) } ^ { \overline { { ( \mathbf { m } ) } } } = \sum _ { \overset { t } { t } } \overbrace { \binom { r _ { m } } { m } - \overbrace { ( \mathbf { o } _ { t } ) } - \overbrace { ( f _ { m } ) } ^ { ( \mathbf { m } ) } } ^ { - \overbrace { ( \mathbf { o } _ { t } ) } ^ { ( \mathbf { m } ) } } = \sum _ { \overset { t } { t } } ( r _ { m } ^ { \top } \mathbf { 0 } _ { t } f _ { m } ) \mathbf { 0 } _ { t } .
$$

## 3.2.2 Property Extraction

Given a target role $\mathbf { } _ { \mathbf { } ^ { r } { } _ { t } . }$ , target fillers are extracted from the matching object and weighted by their similarity to the filler bound to $\mathbf { \nabla } _ { \mathbf { r } _ { t } }$ (see Appendix B.2).

$$
\begin{array} { r l } { \mathrm { E } _ { \mathrm { p r o p } } ( \mathbf { 0 } , \boldsymbol { r } _ { t } ) : = \begin{array} { c c } { \widehat { \mathbf { \Omega } } ( \mathbf { \eta } _ { t } ) - \widehat { \mathbf { \Omega } } ( \mathbf { \eta } _ { 0 } ) - } & { = \mathbf { \Omega } r _ { t } ^ { \top } \mathbf { 0 } . } \end{array} } \end{array}
$$

## 3.2.3 Transformation and Re-binding

Finally, the extracted fillers will be transformed by some learned linear transformation H and re-bound to a new role $r _ { n }$ per attention head (see Appendix B.3). This allows multiple features to be extracted and transformed in parallel. The final output object is a superposition of these role-filler pairs per head.

$$
\begin{array} { r l r l } { \mathrm { T } _ { \mathrm { b i n d } } ( { \pmb f } , { \pmb H } , { \pmb r } _ { n } ) : = } & { \widehat { ( { \pmb r } _ { n } ) } } & { \widehat { ( { \pmb f } ) } - \widehat { ( { \pmb H } ) } - } & { = { \pmb r } _ { n } \otimes ( { \pmb f } ^ { \top } { \pmb H } ) . } \end{array}
$$

Finally, the full TPR-Attention mechanism per head is described by:

$$
\mathrm { T P R - A t t n } = \mathrm { T } _ { \mathrm { b i n d } } ( \mathrm { E } _ { \mathrm { p r o p } } ( \mathrm { M } _ { \mathrm { o b j } } ( \boldsymbol { \mathsf { M } } , \boldsymbol { r } _ { m } , f _ { m } ) , \boldsymbol { r } _ { t } ) , \boldsymbol { H } , \boldsymbol { r } _ { n } )  &  = - \overbrace { \left( \begin{array} { l l l } { \boldsymbol { r } _ { n } } & { \left( \begin{array} { l l l } { \boldsymbol { r } _ { m } } & { \left( f _ { m } \right) } & { \left( f _ { m } \right) } \\ { \boldsymbol { r } } & { \left( \boldsymbol { \mathsf { M } } \right) } & { \left( f _ { m } \right) } \end{array} \right) ^ { - 1 } } \\ & { \overbrace { \left( \boldsymbol { r } _ { t } \right) } ^ { \mathrm { L i n d } } , \quad \overbrace { \left( \boldsymbol { H } \right) } ^ { \mathrm { d i v e c a l { } } } , } \end{array} \right) } ^ { \left( \mathrm { T p o p } \right) }
$$

## 4 Experiments

We evaluate our mechanism on a controlled composition task (see Figure 3), implemented through a feature substitution, adapted from Montero et al. (2024) using the dSprites dataset (Matthey et al., 2017). In our current setup, rather than working with sprite-level images, our mechanism operates on latent representations (see Appendix E for TPR construction details of the task), allowing us to test the composition mechanism on a controlled setting, independent of perceptual representation learning.

![](images/1c586870550c4dd678b5c9407179cb393873b8b38b4f25186d50e3abdd60190d.jpg)  
Figure 2: Loss on combinatorial generalization for the out-of-distribution set square red. TPR-Attention with 4 heads (red) achieves lower loss in all tests compared to classical attention with 4 heads (yellow) and ResNet baseline with 1 layer (blue). Curves show the mean over 5 seeds, with shaded regions indicating ± 2 standard deviation.

For each experiment, we compare one layer of TPR-Attention with a baseline of one layer of regular attention, and ResNet to see how well each generalizes to unseen combinations of factors of variation.

We evaluate generalization using three out-of-distribution (OOD) splits: scale pos, square pos and square red. These sets are chosen to systematically evaluate compositional generalization under diferent combinations of numerical vs. categorical factors of variation. The scale pos split tests the holdout of numerical features, where latents with scale $> 0 . 7$ and $p o s X > 0$ are held out during training. The square pos split introduces a mixture of numerical and categorical features, holding out squares at posX > 0. Finally, to evaluate over categorical features, we additionally assign a random color encoding (red, green, or blue) to each latent input and choose the square red split, where red squares are held out.

We further evaluate our TPR-Attention on three diferent test conditions: (Test 1) only the reference input contains OOD factors, (Test 2) only the transform input contains OOD factors, and (Test 3) both the reference and the transform input are in-distribution, yet their composition yields an OOD composition in the output.

Finally, we evaluate two diferent experimental settings: non-interacting and interacting factors of variation. This is motivated by prior work which has shown that compositional generalization becomes substantially more dificult when factors interact (Montero et al., 2024). In the non-interacting setting, factors of variation are independent, such that substituting one factor does not afect another. For example, changing an object’s color does not afect its shape. In contrast, in the interacting setting, we add an extra component in the representation which is the interaction of two of the factors of variation. We consider two types of factor interactions. First, we evaluate interactions between numerical factors, scale pos. To induce the interaction, we construct an additional latent factor by adding the scale and position values. Second, we evaluate interactions between categorical factors, shape col. The interaction is introduced as a linear mixing of shape and color, mapping their representations with a random matrix M (for further information, refer to Appendix E.3).

![](images/7d90d2eb58b933bca1f626a48f7bc0363347ce3e68763d18d98e3e5d915efd10.jpg)  
Figure 3: Controlled composition task. Given a reference input, transform input and an action corresponding to a factor of variation, the output must match the transform input along the specified factor and match the reference input for the rest of the factors. In the current experiments, we operate on pre-encoded latent representations rather than raw pixel inputs to isolate the behaviour of TPR-Attention independent of perceptual representation learning.

![](images/04df6e23f371a14bef2367dc7258a330d21db4024ee84cf46b1a4f686ec32c8f.jpg)  
Figure 4: Loss on combinatorial generalization under diferent interaction settings for square red OOD condition. TPR-Attention (red) is compared against classical attention (yellow) and a ResNet baseline with one layer (blue). Curves show the mean over five seeds, with shaded regions indicating ± one standard error. Refer to Appendix F for additional results.

To operate on this composition task, we formulate an action-conditioned TPR-Attention model (described in Appendix D), where the input action determines a query q<sub>i</sub> that specifies the role-filler pair to be modified in the current object. We enable a copy mechanism, which initializes the output object with a copy of the reference input before applying action-conditioned updates. This design choice reflects the structure of the task, as the output must preserve most features of the reference object. To ensure a fair comparison, we implement analogous copy behavior in all baselines. In particular, ResNet baseline has a residual update that adds to the reference object and classical multi-head attention baseline attends over the reference and transform inputs, producing an update to be added to the reference object.

## 5 Discussion

In this work, we introduced a new attention mechanism for structured TPR representations. This mechanism operates directly on role-filler bindings, giving the model explicit access to object-centric structure and enabling binding and unbinding operations within attention.

We then evaluated TPR-Attention in a controlled setting where the latent factors of variation interact. In this hard regime – where existing methods are known to struggle – TPR-Attention generalized compositionally out of distribution substantially better than classical attention and ResNets. To our knowledge, this level of combinatorial generalization on interacting factors has not been demonstrated by prior architectures.

A natural next step is to combine TPR-Attention with a learned encoder to operate directly on pixel inputs. Prior work (Mathieu et al., 2019; Wang et al., 2024; Xu et al., 2022; Montero et al., 2024) has shown that disentangled factors of variation can be recovered from images, suggesting that the structured representations required by TPR-Attention can, in principle, be learned rather than manually provided.

Limitations. Our experiments rely on manually provided latent factors, which allows us to isolate the behavior of TPR-Attention but does not yet demonstrate performance on raw perceptual inputs. In addition, our evaluation focuses on a controlled setting with a small number of structured factors, and it remains to be seen how TPR-Attention scales to higher-dimensional or noisier domains. Finally, another major limitation of the current work is that we only compared combinatorial generalization for a single layer. There is no guarantee that the generalization advantage will persist when we compare stacked versions of the base layer.

## References

Michael W. Cole, Philippe Laurent, and Andrea Stocco. Rapid instructed task learning: A new window into the human brain’s unique capacity for flexible cognitive control. Cognitive, Afective & Behavioral Neuroscience, 13(1):1–22, March 2013. doi: 10.3758/ s13415-012-0125-7.

Sunny Duan, Loic Matthey, Andre Saraiva, Nicholas Watters, Christopher P. Burgess, Alexander Lerchner, and Irina Higgins. Unsupervised model selection for variational disentangled representation learning, 2020. URL https://arxiv.org/abs/1905.12614.

Denis Kleyko, Dmitri A. Rachkovskij, Evgeny Osipov, and Abbas Rahimi. A survey on hyperdimensional computing aka vector symbolic architectures, part i: Models and data transformations. ACM Computing Surveys, 55(6):1–40, December 2022. ISSN 1557-7341. doi: 10.1145/3538531. URL http://dx.doi.org/10.1145/3538531.

Brenden M. Lake and Marco Baroni. Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks, 2018. URL https: //arxiv.org/abs/1711.00350.

Emile Mathieu, Tom Rainforth, N. Siddharth, and Yee Whye Teh. Disentangling disentanglement in variational autoencoders, 2019. URL https://arxiv.org/abs/1812.02833.

Loic Matthey, Irina Higgins, Demis Hassabis, and Alexander Lerchner. dsprites: Disentanglement testing sprites dataset. https://github.com/deepmind/dsprites-dataset/, 2017.

Milton L. Montero, Jefrey S. Bowers, Rui Ponte Costa, Casimir J. H. Ludwig, and Gaurav Malhotra. Lost in latent space: Disentangled models and the challenge of combinatorial generalisation, 2024. URL https://arxiv.org/abs/2204.02283.

Bernhard Sch¨olkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke, Nal Kalchbrenner, Anirudh Goyal, and Yoshua Bengio. Towards causal representation learning, 2021. URL https://arxiv.org/abs/2102.11107.

Paul Smolensky. Tensor product variable binding and the representation of symbolic structures in connectionist systems. Artificial Intelligence, 46(1):159–216, 1990. ISSN 0004-3702. doi: https://doi.org/10.1016/0004-3702(90)90007-M. URL https://www. sciencedirect.com/science/article/pii/000437029090007M.

Paul Smolensky, R. Thomas McCoy, Roland Fernandez, Matthew Goldrick, and Jianfeng Gao. Neurocompositional computing: From the central paradox of cognition to a new generation of ai systems, 2022. URL https://arxiv.org/abs/2205.01128.

Xin Wang, Hong Chen, Si’ao Tang, Zihao Wu, and Wenwu Zhu. Disentangled representation learning, 2024. URL https://arxiv.org/abs/2211.11695.

Zhenlin Xu, Marc Niethammer, and Colin A Rafel. Compositional generalization in unsupervised compositional representation learning: A study on disentanglement and emergent language. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 25074–25087. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/file/9f9ecbf4062842df17ec3f4ea3ad7f54-Paper-Conference.pdf.

## A Tensor Network Diagram Notation

Tensors are multidimensional arrays which encode interactions across multiple modes. Vectors and matrices are a special case of tensors, which store information along one and two modes respectively.

We use tensor network diagrams as a graphical notation, where each node represents a tensor and each edge represents one tensor mode of the node it belongs to.

For example, a vector $_ { v , }$ a matrix M and a third-order tensor T would be represented, respectively, by the following diagrams:

![](images/8b543546147880bc1b178ac88b223afda5574b7c2e7a244ee28da0e42e3863aa.jpg)

A tensor contraction is represented by a shared edge between two nodes corresponding to the mode along which contraction occurs. Edges that do not connect to another node correspond to free indices and represent the modes in the output of the operation. Examples are shown below.

Example 1: Matrix-Vector Contraction.

![](images/00dd628c56f397ce32f6dd11ea85e455f45b6e4ac16da684c5fddb536abf463b.jpg)

Example 2: Matrix-Tensor Contraction.

![](images/6cdc5ac8a410a2ba5a3b9a5fc3487303c37bacc6c3e3876c875557376579d76d.jpg)

Example 3: Tensor-Tensor Contraction.

![](images/f29cb74eca789fb26dfa3b9e0410930eaf785f8a8ecc9e4fac071e63935602ad.jpg)

Finally, tensor product (or outer product), is represented by placing tensors next to each other without shared edges.

Example: Outer Product of Two Vectors

$$
i \stackrel { } { \longrightarrow } \stackrel { } { \bigcirc } \langle \stackrel { } { \pmb { v } } \mathbin { \vrule w \vrule h \vrule h \vrule h 0 . 6 4 d f \bigstar \bigstar \bigstar } \bigstar \bigstar
$$

## B TPR-Attention Module

There are three stages in TPR-Attention: (i) object matching (see B.1), (ii) property extraction (see B.2) and (iii) transformation and re-binding per attention head (see B.3). The full computation of TPR-Attention is summarized in Algorithm 1.

Algorithm 1 Full TPR-Attention   
Input: Current object $\mathbf { O } _ { t } .$ memory of past objects M   
Output: Updated object $\mathbf { O } _ { t + 1 }$   
$\mathbf { M } _ { t + 1 } \mathbf { \bar { \Psi } }  \mathbf { M } _ { t } \mathbf { \bar { \Psi } } + \mathrm { T P R - S A M } ( \mathbf { O } _ { t } )$   
for each head n do   
Compute $\mathbf { \Delta } { \hat { f } } _ { m } ^ { ( n ) } , \mathbf { r } _ { m } ^ { ( n ) } , \mathbf { r } _ { t } ^ { ( n ) } .$ as functions of $\mathbf { O } _ { t }$   
$\begin{array} { r } { \mathbf { 0 } _ { t + 1 , n } \gets \mathrm { T } _ { \mathrm { b i n d } } \Big ( \mathrm { E } _ { \mathrm { p r o p } } \Big ( \mathrm { M } _ { \mathrm { o b j } } \big ( \mathbf { M } _ { t + 1 } , r _ { m } ^ { ( n ) } , f _ { m } ^ { ( n ) } \big ) , r _ { t } ^ { ( n ) } \Big ) , H ^ { ( n ) } , r _ { n } \Big ) } \end{array}$   
end for   
$\begin{array} { r } { \mathbf { 0 } _ { t + 1 }  \sum _ { n } \mathbf { 0 } _ { t + 1 , n } } \end{array}$

## B.1 Object Matching

The following is the derivation of object matching in TPR-Attention:

![](images/c6eb40965b1d27538864a1cb44ee4ae7db669b8f77c201aff5a19e3e3b440774.jpg)  
First, we expand each $\mathbf { O } _ { t }$ into its role-filler decomposition. Then, using the orthonormal property of the role vectors $( \mathrm { i . e . , ~ } \langle r _ { m } , r _ { j } \rangle = \delta _ { j m } )$ , we collapse the role tensor contractions via the Kronecker delta. The Kronecker delta eliminates the sum over $j ,$ , leaving only the terms where $j = m$

Finally, for each object s, the contraction produces a similarity score between $f _ { m }$ and $\pmb { f } _ { m } ^ { s }$ The final result is then a weighted sum of objects from memory, each scaled by how well $f _ { m }$ matched $\pmb { f } _ { m } ^ { s }$

## B.2 Property Extraction

In order to retrieve a target property, we introduce a target role $\mathbf { } _ { \mathbf { } \mathbf { } \mathbf { } \mathbf { } r _ { t } }$ to extract target fillers from the retrieved objects and define:

$$
\begin{array} { r l } { \mathrm { E } _ { \mathrm { p r o p } } ( \mathbf { 0 } , r _ { t } ) : = { \bf \nabla } \widehat { \left( r _ { t } \right) } - \widehat { \left( \mathbf { 0 } \right) } - } & { = r _ { t } ^ { \top } \mathbf { 0 } . } \end{array}
$$

$$
\begin{array}{c} \operatorname { E } _ { \operatorname { p r o p } } ( \operatorname { M } _ { \mathrm { o b j } } ( \mathbf { M } , r _ { m } , f _ { m } ) , r _ { t } ) = \ \sum _ { s , i } \ { \begin{array} { l } { { \binom { r _ { t } } { c } } - { \binom { r _ { t } } { i } } \left( { \frac { f _ { i } ^ { s } } { f _ { i } ^ { s } } } \right) - } \\ { \left( { \binom { f _ { m } ^ { s } } { m } } - { \binom { f _ { m } ^ { s } } { f _ { i } ^ { s } } } \right) } \end{array} } = \ \sum _ { s } \ { \binom { s } { f _ { m } ^ { s } } } - { \binom { f _ { m } } { f _ { m } ^ { s } } } \ \left( { \binom { f _ { i } ^ { s } } { f _ { i } ^ { s } } } \right) - \sum _ { s = 1 } ^ { \infty } { \binom { f _ { m } } { f _ { i } ^ { s } } } \ \sum _ { s = 1 } ^ { \infty } { \binom { f _ { m } } { f _ { i } ^ { s } } } \left( { \binom { f _ { m } ^ { s } } { f _ { i } ^ { s } } } \right) = \sum _ { s = 1 } ^ { \infty } { \binom { f _ { m } } { f _ { i } ^ { s } } } \left( { \binom { f _ { m } } { f _ { i } ^ { s } } } \right) +  \end{array}
$$

As before, we use Kronecker delta to eliminate the sum over i, leaving only the terms where $i = t$

## B.3 Transformation and Re-binding in Multi-Head TPR-Attention

Finally, the weighted sum of target fillers across retrieved objects are transformed by a learned matrix H and re-bound to a new role $\boldsymbol { r } _ { n }$ per n attention heads:

$$
\operatorname { T } _ { \mathrm { b i n d } } ( { \pmb f } , { \pmb H } , { \pmb r } _ { n } ) : = \mathrm { ~ \ o ~ { ~ \{ ~ \pmb r } _ { n } ~ \pmb ~ { ~ \{ ~ \pmb f } ~ \} - \left( ~ { \pmb H } \right) ~ } ~  = { \pmb r } _ { n } \otimes ( { \pmb f } ^ { \top } { \pmb H } ) .
$$

$$
\mathrm { T } _ { \mathrm { b i n d } } ( \mathrm { E } _ { \mathrm { p r o p } } ( \mathrm { M } _ { \mathrm { o b j } } ( \mathbf { M } , r _ { m } , f _ { m } ) , r _ { t } ) , H , r _ { n } ) = - \widehat { \left( r _ { n } \right) } \sum _ { s } \widehat { \left( f _ { m } ^ { s } \right) } - \widehat { \left( f _ { m } \right) } \widehat { \left( f _ { t } ^ { s } \right) } - \widehat { \left( H \right) } - \widehat { \left( H \right) } - \widehat { \left( H \right) } \widehat { \left( f _ { m } \right) } = - \widehat { \left( f _ { m } \right) } \widehat { \left( f _ { m } \right) } ,
$$

## C Generalization of TPR-Attention

In previous sections, we have described a TPR-Attention mechanism which matches objects of form order-2 TPRs given a single role-filler query to match. It might often be the case that (i) structured reasoning may require the querying of multiple factors simultaneously, and (ii) objects will need to be represented as a higher–order TPR (see example in Appendix D).

To accommodate for these cases, we first define the general TPR-SAM. Given objects represented by order-s TPRs, m queries and n desired dimensions on our matched objects, we define memory M:

$$
\mathbf { \boldsymbol { \mathsf { M } } } = \sum _ { t } ( \mathbf { \boldsymbol { \mathsf { O } } } _ { t } ) \varprojlim _ { \mathbf { \theta } } ) \mathbf { \boldsymbol { \mathsf { s } } } ^ { \otimes _ { j } } = \sum _ { t } \mathbf { \boldsymbol { \mathsf { O } } } _ { t } ^ { \otimes j } ,
$$

where $j \cdot s - m = n$ must be satisfied, so that during matching we have

$$
\mathrm { M } _ { \mathrm { o b j } } \left( { \bf M } , \ { \bf Q } = \bigotimes _ { k = 1 } ^ { m } q _ { k } \right) = \bigoplus _ { { \bf Q } } \ \sum _ { \bf \Xi \ ^ { \prime } \Sigma } \left. \textbf { M } \boxed { \begin{array} { c c } { \vdots \ n } \\ { \vdots \ } \end{array} } \right. \ .
$$

As an example, consider objects represented as order-2 TPRs:

$$
\mathbf { 0 } _ { t } = \sum _ { i } ( r _ { i } \otimes f _ { i } ) .
$$

If we were to query the simultaneous conjunction of two properties $\pmb { r } _ { m } ^ { 1 } , \pmb { f } _ { m } ^ { 1 }$ and $r _ { m } ^ { 2 } , f _ { m } ^ { 2 }$ , we would need the memory to be

$$
\mathbf { \nabla } \mathbf { \mathsf { M } } = \sum _ { t } \mathbf { 0 } _ { t } \otimes \mathbf { 0 } _ { t } \otimes \mathbf { 0 } _ { t } ,
$$

so that we can match the objects having both properties simultaneously:

![](images/26cc45a6819220533ffdc2ebde6658db93ca39e05aff44b74fa778af315080cb.jpg)

## D Action Conditioned TPR-Attention Module

For the composition task described in Section 4, the goal is to construct an output object $\mathbf { O } _ { 3 }$ , whose role-filler structure selectively combines concepts from two input objects. Given a reference object $\mathbf { O } _ { 1 }$ , a transform object $\mathbf { O } _ { 2 } .$ , and an action specifying a target role, the output object should preserve all role-filler bindings from $\mathbf { O } _ { 1 }$ except for the target role, which is substituted with the corresponding binding from $\mathbf { O } _ { 2 }$

Following the general TPR-SAM (as described in Appendix C), we extend the general TPR-Attention baseline (as described in Appendix B), by (i) adding an ID tag to each object to make them identifiable, (ii) changing the query structure to accommodate for the added IDs and specific task, and (iii) constructing queries from the given action.

Given order-3 tensors $\mathbf { 0 } _ { t } ,$ two queries (for role and ID, respectively), and a desired single filler dimension on the retrieved objects, we define the memory M of our task to be the sum of objects $\mathbf { O } _ { t }$

![](images/772161860d85e916b38a7501f09d99c49e19f57132228caff934ac0ae22ad155.jpg)

where each $\mathbf { O } _ { t }$ represents an image latent as a TPR along with an ID tag bound to it with the outer product:

![](images/c224290f2f7f13e1fe5ad2fc9cdce11c4921f52e2c92f185c3ffd8d9fe70c7fa.jpg)

We express the desired output object $\mathbf { O } _ { 3 }$ as a superposition of three components: the contribution from $\mathbf { O } _ { 1 }$ , the removal of the target role-filler binding from $\mathbf { O } _ { 1 }$ , and the contribution from $\mathbf { O } _ { 2 }$ restricted to the target role.

![](images/1170b3f0295786826c10bb26e93f3669cb0c0406daa817d08c9797703bf3ce8c.jpg)

We interpret the composition operation for constructing $\mathbf { O } _ { 3 }$ as a substitution in the tensor space. Conditioning on M via $\hat { \mathbf { O } } \otimes i d ^ { 1 }$ , we learn to subtract the role-filler corresponding to our chosen action with learned negative identity matrix H. Furthermore, we ”substitute” this role-filler with the pair extracted from $\mathbf { O } _ { 2 }$

Having defined our desired composition operation for tasks using an action, we now generalize to formalize the learnable parameters of our TPR-Attention mechanism for each head i.

We define the query vector $\pmb q _ { i }$ as a function of the input action a. Let a be the chosen input action and $\pmb { H } _ { l } ^ { q }$ be a learned projection matrix. Then:

![](images/1dd047e79c563dccc853216f5b084888bdf25a360e2493d0aa9b0245647a6afa.jpg)

Similarly, we define a role $\boldsymbol { r } _ { n } ^ { i }$ , which is the role to which our queried filler is bound to in the output object.

![](images/0e772c52c519668b55082e29ac66d126a1835c888290013bf3e6b23c5d63746b.jpg)

Finally, we define the generalized attention mechanism at each head as:

![](images/c426bb3ca309ae62a7db5facf61fc288490133343cf81db8b0b9f3a3f533c0f1.jpg)

## E Composition Task Representation Details

We construct an object representation from the ground-truth latent factors of dSprites. Each object is represented as a set of role-filler pairs. Roles are fixed and represented as one-hot vectors. The construction of fillers, on the other hand, depends on the factor of variation.

## E.1 Roles.

Let $d _ { r }$ denote the number of roles and $d _ { f }$ the filler dimension. We use the canonical basis in $\mathbb { R } ^ { d _ { r } }$ :

$$
\pmb { r } _ { j } = \mathbf { e } _ { j } \in \mathbb { R } ^ { d _ { r } } , \qquad j \in \{ 1 , \dots , d _ { r } \} ,
$$

so roles are one-hot vectors.

## E.2 Non-interacting Fillers.

Color and shape. For categorical factors (color, shape), we use one-hot encodings. For example, the three colors red, green, and blue are represented as

$$
\mathbf { f } _ { \mathrm { r e d } } = \left[ 0 \right] , \quad \mathbf { f } _ { \mathrm { g r e e n } } = \left[ 0 \right] , \quad \mathbf { f } _ { \mathrm { b l u e } } = \left[ 0 \right] ,
$$

and shape categories are encoded similarly.

Orientation. Let $\theta \in \mathbb { R }$ denote the orientation angle. We represent orientation on the unit circle:

$$
\begin{array} { r } { \pmb { f } _ { \mathrm { o r i e n t } } ( \theta ) = \left[ \stackrel { \cos \theta } { \sin \theta } \right] \in \mathbb { R } ^ { 3 } . } \end{array}
$$

Position. Let $( x , y )$ denote the normalized 2D position. We encode position with:

$$
\begin{array} { r } { f _ { \mathrm { p o s } } ( x , y ) = \left[ \begin{array} { c } { x } \\ { y } \\ { 1 - \sqrt { x ^ { 2 } + y ^ { 2 } } } \end{array} \right] \in \mathbb { R } ^ { 3 } . } \end{array}
$$

Scale. We map each scale index to a point on the unit sphere using fixed angles $\phi$ and $\theta _ { s } i$

$$
f _ { \mathrm { s c a l e } } ( s ) = \left[ \sin \theta _ { s } \cos \phi \right] \in \mathbb { R } ^ { 3 } ,
$$

where $\phi$ is fixed and $\{ \theta _ { s } \} _ { s = 0 } ^ { 5 }$ are fixed values.

## E.3 Interacting Fillers.

Numerical interaction. Scale and position factors are used to construct an interacting factor, where:

$$
f _ { \mathrm { s c a l e - p o s } } = \mathrm { n o r m a l i z e } ( f _ { \mathrm { s c a l e } } + f _ { \mathrm { p o s } } )
$$

Categorical interaction. Shape and color factors are used to construct an interacting factor between two discrete factors, where:

$$
f _ { \mathrm { s h a p e - c o l o r } } [ k ] = \sum _ { i , j } f _ { \mathrm { s h a p e } } [ i ] \ M [ k , i , j ] \ f _ { \mathrm { c o l o r } } [ j ] , \qquad k \in \{ 1 , \ldots , d _ { f } \} .
$$

## F Additional Results

## F.1 Extended Results for Non-Interacting Setting

The following results show the loss on combinatorial generalization given 4 heads (left column) and 8 heads (right column) for TPR-Attention and classical attention.

![](images/6a400166561d1ea8d711665d1d4289216a4b9fe73b0239129c68ccbccfeb2322.jpg)  
(a) OOD: scale pos (4 heads)

![](images/0e71755b1fa6cbce7552ebd89c1381d1f6f29210415626cdb88f56984a8a8776.jpg)  
(b) OOD: scale pos (8 heads)

![](images/a40cb39587e2b0ea65305b9c10541e81d190f2041888707c36c43ba6ca6abfe4.jpg)

![](images/db17ff8687569580d69044e3400df25d3e7ed3444fa508ef197391509842281c.jpg)

(c) OOD: square pos (4 heads)  
![](images/dbf5603ac9809963a1b6ef99833633c5f59734d26c23d2d882e1e142422fef92.jpg)  
(e) OOD: square red (4 heads)

(d) OOD: square pos (8 heads)  
![](images/7daca7cd6f986a25a5e382d49ad5abb6318745f7cd756007a642c6e987ec7987.jpg)  
(f) OOD: square red (8 heads)  
Figure 5: Loss on combinatorial generalization across all out-of-distribution test sets. Left column corresponds to results with 4 heads and the right column to 8 heads. TPR-Attention (red) con sistently achieves lower loss than classical attention (yellow) and ResNet baselines (blue). Curves show the mean over 5 seeds with shaded regions indicating ± 2 standard deviation (4 heads) and ± 2 standard error (8 heads).

## F.2 Extended Results for Numerical Interaction Setting

![](images/bc252e36008e547a07a75836c0e84979ae00701f881bcb4ebd44c808efd4a735.jpg)  
(a) OOD: scale pos

![](images/7e5854b9f02431d4ff751113da127d1de2a5f05aed1598c3a278e3b300a6f710.jpg)  
(b) OOD: square pos

![](images/c6ca6c9b57b358c79a720907bc0a7c07a2fbd4f9152491c6de14155f4008039b.jpg)  
(c) OOD: square red  
Figure 6: Loss on combinatorial generalization with scale pos interaction across 5 seeds. TPR-Attention (8 heads) shown in red, classical attention (8 heads) in yellow and ResNet baseline with 1 layer in blue. Shaded regions show ± 1 standard error.

## F.3 Extended Results for Categorical Interaction Setting

![](images/99fc851648217e1fc478c4fc653fcc33b8f70ed365b0c53ec3807910dbe4eb18.jpg)  
(a) OOD: scale pos

![](images/fadc3e5e5121c3625862c3f781d7bad5979798e6170d4dfcd5940f51e00f0774.jpg)  
(b) OOD: square pos

![](images/6c0265bd04883cf2bc64d0ceb0801bdfdfb23a174aad801deed5d2a8dd358078.jpg)  
(c) OOD: square red  
Figure 7: Loss on combinatorial generalization with shape col interaction across 5 seeds. TPR-Attention (8 heads) shown in red, classical attention (8 heads) in yellow and ResNet baseline with 1 layer in blue. Shaded regions show ± 1 standard error.