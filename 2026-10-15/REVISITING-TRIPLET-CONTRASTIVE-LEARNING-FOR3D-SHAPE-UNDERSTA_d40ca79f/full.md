# REVISITING TRIPLET CONTRASTIVE LEARNING FOR3D SHAPE UNDERSTANDING

Chaoyang Zhu<sup>1</sup> Zhe Liao<sup>1</sup> Jiayi Zhang<sup>1</sup> Jie Li<sup>2</sup> Jiayi Ji<sup>1</sup> Liujuan Cao<sup>1,†</sup> <sup>1</sup>Xiamen University <sup>2</sup>Shanghai AI Lab

## ABSTRACT

In multi-modal 3D pre-training, aligning 3D shapes, their 2D image counterparts, and language descriptions has made tremendous progress. However, we find that supervision from rendered images is unreliable, the semantics encoded in synthetic views are distorted due to domain gap, self-occlusion, and rendering artifacts. Instead, we learn transferable 3D representations from language only. We observe that, in real world, 3D shapes are organized in a tree-like structure, ranging from general concepts to fine-grained instances with specific attributes, while language typically denotes generic concepts and subsumes these various 3D instantiations. Hence, we propose to embed 3D representations in hyperbolic space, whose geometry is well-suited for modeling tree-like data without distortion. To capture the asymmetric hierarchy between language and 3D shapes, we further impose a partial-order relation that language entails 3D shapes. Our method, Hyperbolic Contrastive Learning (HyperCL), jointly optimizes a hyperbolic geodesic-based contrastive objective and an entailment loss to learn discriminative cross-modal representations while explicitly preserving this hierarchy. We also introduce a rigorously verified benchmark for comprehensively evaluating cross-modal 3D alignment. Across two widely used benchmarks and ours, we improve the performance up to 4.6% absolute percentage on 3D shape classification and 5.5% on cross-modal text-to-3D retrieval under the same parameter budget. The code and evaluation benchmark will be made publicly available.

## 1 INTRODUCTION

Triplet contrastive learning. Multi-modal 3D contrastive pre-training (Liu et al., 2023; Wang et al., 2024; 2025; Xue et al., 2023; 2024; Zhang et al., 2024; Zhou et al., 2024; Gao et al., 2024b) has emerged as a predominant paradigm for learning transferable 3D representations across various downstream tasks, e.g., zero-shot point cloud classification and cross-modal shape retrieval. However, unlike web-scale image-text data used to pretrain 2D vision-language models (Radford et al., 2021), high-quality language annotations for 3D shapes are extremely scarce. As a result, rendered images are often used as visual proxies for generating synthetic captions via image captioning models (Li et al., 2023; Achiam et al., 2023). Given these tri-modal triplets, existing methods align 3D shapes bidirectionally with language descriptions and rendered images in a shared semantic space.

While this triplet contrastive learning paradigm substantially alleviates the need for manually labeled 3D annotations, it has several limitations. Rendered images exhibit a severe domain gap w.r.t. natural images used to pretrain 2D vision-language models, i.e., shifted foreground color statistics and large blank background areas (Liang et al., 2023; Zhang et al., 2024). Furthermore, manually positioned virtual cameras following a predefined trajectory cause severe self-occlusion in certain perspectives during rendering (Luo et al., 2024). The rendering pipeline itself could also introduce unexpected errors, such as color distortions and complete rendering failures. Consequently, learning from these incomplete and misleading semantics in rendered images inevitably has a negative impact on the alignment. Zhang et al. (2024) attempts to address the unreliable image supervision by first adapting CLIP (Radford et al., 2021) image encoder to rendered 3D views, thereby reducing the domain gap. However, how to avoid self-occlusion that fundamentally distorts the perceived 3D shape semantics still remains an open problem. For example, even a person may mistake a living room viewed behind its exterior wall for a rectangular box. Hence we advocate learning from language supervisions only to reduce the influence of low-fidelity and unreliable semantics in vision modality.

Hyperbolic representation learning. In the real world, 3D shapes often convey multifaceted semantics, ranging from various categories to intricate details like color, texture, geometry, design style, and functionality. This naturally organizes into a coarse-to-fine-grained tree that spans a wide spectrum of semantic granularities. In this hierarchy, general concepts described by language are typically positioned near the root, whereas leaf nodes represent specific 3D instantiations characterized by unique attributes. Existing approaches mainly use Euclidean space to model this tree-like structure. However, Euclidean space often struggles to preserve the underlying hierarchical structure without distortion due to its limited polynomial volume growth. As the depth of the underlying tree increases, many fine-grained concepts are compressed into a small region in the space, leading to reduced discriminative capacity. This limitation is particularly prominent in 3D shape understanding, where many real-world instances share a common coarse semantic but differ in subtle attributes.

In contrast, hyperbolic space is naturally suitable for tree-like data that has exponential volume growth w.r.t.its radius. This geometric property makes it useful to model an exponentially branching tree without losing discrimination capacity. Hence we propose to contrast 3D shapes against languages in hyperbolic space using a novel hyperbolic contrastive objective. The hyperbolic geometry also allows us to model the asymmetric relation between language and 3D shapes, i.e., language entails 3D shapes. To encode it explicitly, we further propose an entailment loss that ensures positive 3D representations lie inside the entailment cone of its paired text, and negative 3D shapes outside the cone. Experimental results demonstrate superior performance of our HyperCL on both zero-shot 3D shape recognition and cross-modal retrieval, enabling fine-grained cross-modal discrimination while preserving the vision-language hierarchy embedded in data. Overall, our contributions are summarized as the following:

\- We identify the limitations of existing triplet contrastive learning framework, and propose to learn from language supervision only.

\- We introduce HyperCL, a novel approach leveraging hyperbolic geometry to model the tree-like structure embedded in 3D data and enforce a partial-order relation in which language entails 3D shapes, enhancing fine-grained discrimination while preserving the hierarchy.

\- We create a high-quality, manually verified benchmark to rigorously evaluate 3D cross-modal alignment. Experiments show that our method surpasses existing state-of-the-art by a large margin on 3D shape classification, text-to-3D, and 3D-to-text retrieval tasks.

## 2 RELATED WORK

## 2.1 3D TRIPLET CONTRASTIVE LEARNING

Multi-modal 3D pre-training has made tremendous progress in recent years. Besides learning from natural language supervision via bi-directional contrastive loss, the point cloud encoder also contrasts with rendered images of 3D shapes, forming a triplet of (point cloud, text, image). Due to the fact that 3D captioning is underexplored, pioneering works Liu et al. (2023); Xue et al. (2023; 2024); Luo et al. (2023; 2024) focus on how to accurately describe 3D shapes. They first render 3D shapes from preset viewpoints using Blender, then prompt pre-trained image captioning models (Achiam et al., 2023; Li et al., 2023) to generate language descriptions. To cope with the scarcity of 3D pretraining data, Uni3D Zhou et al. (2024) initializes its point cloud encoder directly from pretrained 2D image encoders (Caron et al., 2021; Radford et al., 2021; Sun et al., 2023) instead of training from scratch, establishing a strong optimization prior. MM-Mixing Wang et al. (2025) mixes different point clouds to learn robust and discriminative representations. MixCon3D Gao et al. (2024b) integrates the distinct yet complementary information between rendered images and point clouds to contrast with languages. MRD Wang et al. (2024) performs multi-modal distillation to maintain the semantic relations of both intra- and inter-modalities in the embedding space.

However, given rendered images, CLIP fails to encode meaningful semantics from rendered images due to domain gap. Besides, how to position virtual cameras and select appropriate viewpoints to avoid self-occlusion is still an open problem. Furthermore, rendering pipeline sometimes leads to unexpected errors. TAMM Zhang et al. (2024) adds an extra stage that adapts CLIP image encoder via adapter (Gao et al., 2024a) to rendered images for extracting faithful semantic information. However, better alignment with rendered images during training does not necessarily translate into better generalization, possibly due to overlapped semantics with language. Even if the distorted semantics brought by domain gap can be faithfully restored, the multi-view rendering at scale is prohibitively expensive, and the fundamental self-occlusion issue inherent to any 2D projection of 3D shapes remains difficult to address. Hence we propose to learn from language only, bypassing the limitations of the supervision signal in rendered images.

## 2.2 HYPERBOLIC LEARNING

Characterized by the exponential volume expansion, hyperbolic geometry is naturally suited for modeling hierarchical data such as trees and graphs. Hyperbolic learning can be divided into two groups: 1) the first line of work builds hyperbolic neural networks (Ganea et al., 2018b; Shimizu et al., 2020) by designing hyperbolic counterparts of euclidean layers, such as convolution and multinomial logistic regression, and these networks are optimized using specialized riemannian optimizers; 2) the second group still keeps the main chunk of model in Euclidean space, but with an added exponential mapping projection to transform euclidean features onto hyperbolic data points. Foundational works Nickel & Kiela (2017; 2018) establish the efficacy of two representative models in hyperbolic geometry, i.e., poincare ball and lorentz model, for capturing lexical hierarchies´ within text modality. Ganea et al. (2018a) further introduces entailment cone to encode asymmetric is-a relations in hyperbolic space, which inspires our batch-wise entailment loss.

In computer vision and multi-modal learning, hyperbolic geometry has been adapted to various tasks including 2D representation learning (Ermolov et al., 2022; Ge et al., 2023), image and emotion classification (Chen et al., 2023; Khrulkov et al., 2020; Guo et al., 2022), image segmentation (Atigh et al., 2022), open-vocabulary perception (Doan et al., 2024; Kong et al., 2024; Peng et al., 2025), 2D vision-language models (Desai et al., 2023; Ramasinghe et al., 2024; Pal et al., 2025), and scaling models to billions of parameters (Mandica et al., 2024). However, applying hyperbolic geometry to model 3D data is less explored. Liu et al. (2025) is the first to utilize hyperbolic learning in 3D domain but without large-scale pre-training. Following prior works Desai et al. (2023); Kong et al. (2024) that enforce the partial relation such as object entails scenes or caption entails images/objects in 2D, we enforce caption entails 3D shapes to build vision-language hierarchy and adopt the lorentz model which is numerically more stable and suitable for gradient optimization than poincare ball.´

## 3 METHOD

We first briefly introduce hyperbolic geometry and the triplet contrastive learning paradigm in Section 3.1. Then we analyze the current pitfall of existing triplet contrastive learning framework in Section 3.2, followed by our proposed method HyperCL with hyperbolic geodesic-based contrastive loss and entailment loss in Section 3.3.

## 3.1 PRELIMINARIES

Hyperbolic geometry. Unlike Euclidean and hypersphere space with zero and constant positive curvature, hyperbolic space is a type of riemannian manifold with constant negative curvature. There are five equivalent models of hyperbolic space, and we focus on lorentz model in this work since it is suitable for optimization. We refer to Ratcliffe (2006) for a complete treatment.

In lorentz model, hyperbolic space $\mathbb { L } ^ { n }$ is represented as $\{ \mathbf { h } \in \mathbb { R } ^ { n + 1 } : \langle \mathbf { h } , \mathbf { h } \rangle _ { \mathcal { L } } = - 1 / c , c > 0 \}$ on the upper half of a two-sheeted hyperboloid. The curvature c measures the degree to which hyperbolic space deviates from flat Euclidean space. A hyperbolic representation h is $[ \mathbf { h } _ { \mathrm { t i m e } } ; \mathbf { h } _ { \mathrm { s p a c e } } ]$ with time and space component $\mathbf { h } _ { \mathrm { t i m e } } \in \mathbb { R } , \dot { \mathbf { h } _ { \mathrm { s p a c e } } } \in \bar { \mathbb { R } } ^ { \hat { n } } . \ \langle \cdot , \cdot \rangle \boldsymbol { \mathscr { L } }$ is lorentz inner product, computed as $\langle \mathbf { u } , \mathbf { v } \rangle _ { \mathcal { L } } = \langle \mathbf { u } _ { \mathrm { s p a c e } } , \mathbf { v } _ { \mathrm { s p a c e } } \rangle _ { \mathcal { E } } - \mathbf { u } _ { \mathrm { t i m e } } \mathbf { v } _ { \mathrm { t i m e } }$ , and $\langle \cdot , \cdot \rangle _ { \varepsilon }$ is euclidean inner product. Given the space component, the corresponding time component is calculated as $\mathbf { h } _ { \mathrm { t i m e } } = \sqrt { 1 / c + \| \mathbf { h } _ { \mathrm { s p a c e } } \| ^ { 2 } }$

Analogous to straight lines in Euclidean space, the shortest path connecting two points in hyperbolic space is called geodesic. In lorentz model, geodesics are curves traced by the intersection of the hyperboloid with hyperplanes passing through the origin O of $\mathbb { R } ^ { n + 1 }$ . The lorentz distance between

Table 1: We re-train Open-Shape on ensembled dataset. CLIP ViT-B/16 is used as the frozen image and text encoder. Rendered images are substantially worse than languages. Acc@1 is reported.
<table><tr><td>Image</td><td>Text</td><td>OLVIS</td><td>MN40</td></tr><tr><td>√</td><td>√</td><td>46.9</td><td>84.2</td></tr><tr><td>√</td><td>x</td><td> $2 6 . 0 _ { \downarrow 2 0 . 9 }$ </td><td>74.2↓10.0</td></tr><tr><td>x</td><td>√</td><td> $4 5 . { \bar { 5 } } _ { \downarrow 1 . 4 }$ </td><td>84.5↑0.3</td></tr></table>

![](images/a7a9afe7ae8b6824c83b1cba571bc7a933e67eaa0fb41e8bc3950b6a5e3b64f0.jpg)  
Table 2: Hyperbolicity of the representation space induced by Uni3D. A lower value indicates a stronger underlying tree-like hierarchy.  
Figure 1: Problems when contrasting with rendered images only.

<table><tr><td rowspan="2">Model</td><td colspan="2">δ-hyperbolicity ↓</td></tr><tr><td>OLVIS</td><td>MN40</td></tr><tr><td>Uni3D-B</td><td>0.24</td><td>0.26</td></tr><tr><td>Uni3D-L</td><td>0.20</td><td>0.24</td></tr><tr><td>Uni3D-G</td><td>0.19</td><td>0.24</td></tr></table>

two points in hyperbolic space is:

$$
d _ { \mathcal { L } } ( \mathbf { u } , \mathbf { v } ) = \sqrt { 1 / c } \cdot \cosh ^ { - 1 } ( - c \langle \mathbf { u } , \mathbf { v } \rangle _ { \mathcal { L } } ) ,\tag{1}
$$

The tangent space $\mathcal { T } _ { \mathbf { u } } \mathbb { L } ^ { n } = \left. \mathbf { h } \in \mathbb { R } ^ { n + 1 } : \left. \mathbf { h } , \mathbf { u } \right. _ { \mathcal { L } } = 0 \right.$ at a hyperbolic point u is an Euclidean space that locally approximates hyperbolic space around u. The exponential map is used to project a point $\mathbf { e } \in \mathbb { R } ^ { n }$ from tangent space onto hyperbolic manifold:

$$
\mathbf { h } _ { \mathrm { s p a c e } } = \mathrm { e x p m } _ { \mathbf { O } } ( \mathbf { e } ) = \frac { \sinh ( \sqrt { c } \| \mathbf { e } \| ) } { \sqrt { c } \| \mathbf { e } \| } \mathbf { e } .\tag{2}
$$

Triplet contrastive learning. As shown in Figure 2 (a), existing methods typically take triplets $\{ \mathbf { x } _ { i } ^ { \hat { P } } , \mathbf { x } _ { i } ^ { I } , \mathbf { x } _ { i } ^ { T } \} _ { i = 1 } ^ { B }$ with point cloud, rendered image, and language description as input. They are first encoded into $\tilde { \mathbf { e } } _ { i } ^ { P } = f _ { P } ( \mathbf { x } _ { i } ^ { P } ) , \tilde { \mathbf { e } } _ { i } ^ { I } = f _ { I } ( \mathbf { x } _ { i } ^ { I } ) , \tilde { \mathbf { e } } _ { i } ^ { T } \overset { = } { = } f _ { T } ( \mathbf { x } _ { i } ^ { T } )$ by point cloud, image, and text encoders. Then they are normalized to project onto unit sphere space: $\mathbf { e } _ { i } ^ { P } = \widetilde { \mathbf { e } } _ { i } ^ { P } / \Vert \widetilde { \mathbf { e } } _ { i } ^ { \breve { P } } \Vert , \mathbf { e } _ { i } ^ { T } = \widetilde { \mathbf { e } } _ { i } ^ { T } / \Vert \widetilde { \mathbf { e } } _ { i } ^ { T } \Vert$ $\mathbf { e } _ { i } ^ { I } = \widetilde { \mathbf { e } } _ { i } ^ { I } / | | \widetilde { \mathbf { e } } _ { i } ^ { I } | |$ . Let $\begin{array} { r } { l ( \mathbf { a } , \mathbf { b } ) = \log \frac { \exp ( \mathcal { K } ( \mathbf { a } , \mathbf { b } ) / \tau ) } { \sum _ { j = 1 } ^ { B } \exp ( \mathcal { K } ( \mathbf { a } , \mathbf { b } _ { j } ) / \tau ) } } \end{array}$ be the log-softmax operation where $\kappa ( \cdot )$ is a similarity measure and τ is temperature. In Euclidean space, $\kappa ( \cdot )$ is cosine similarity. The triplet contrastive learning objective $\mathcal { L } _ { E }$ consists of the following terms:

$$
\mathcal { L } _ { E } ^ { P  I } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } ( l ( \mathbf { e } _ { i } ^ { P } , \mathbf { e } _ { i } ^ { I } ) + l ( \mathbf { e } _ { i } ^ { I } , \mathbf { e } _ { i } ^ { P } ) ) , \quad \mathcal { L } _ { E } ^ { P  T } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } ( l ( \mathbf { e } _ { i } ^ { P } , \mathbf { e } _ { i } ^ { T } ) + l ( \mathbf { e } _ { i } ^ { T } , \mathbf { e } _ { i } ^ { P } ) ) .\tag{3}
$$

Note that only the point cloud encoder is trainable, image and text encoders are directly from pretrained CLIP Radford et al. (2021) and they remain frozen.

## 3.2 ANALYSIS ON TRIPLET CONTRASTIVE LEARNING

We investigate one representative work OpenShape Liu et al. (2023). As shown in Table 1, compared to triplet contrastive learning, the performance degrades severely when language supervision is removed. In contrast, after removing image supervision, it is on par with and can even marginally surpass triplet contrastive learning. This confirms that the gap mainly stems from the inability of CLIP to encode reliable semantics from rendered images due to domain gap (Liang et al., 2023), self-occlusion, and rendering errors. As shown in Figure 1, the frying pan might be mistaken for a shallow bowl, and CLIP is hard to extract meaningful semantics given the rendered large blank area (Liang et al., 2023). In contrast, since language descriptions are aggregated from multiple views and inconsistent descriptions are filtered out (Luo et al., 2024; Xue et al., 2024), they are more robust. Hence, we propose to learn from language supervision only.

## 3.3 HYPERCL

We diagnose whether the learned representation space resembles a tree structure. For this, we choose Uni3D Zhou et al. (2024), a state-of-the-art model pretrained for 3D understanding. We use δ- hyperbolicity in Khrulkov et al. (2020); Fournier et al. (2015) to measure how far the space deviates from a tree, which is zero for a perfect tree and increases as the underlying structure becomes less tree-like (more details are in Appendix A.3). As reported in Table 2, the representations induced by euclidean triplet contrastive learning are already intrinsically hierarchical. This reflects the fact that real-world 3D concepts form a vast taxonomy with exponential branching factor. However, the geometry in which these representations are optimized is misaligned with this tree-like structure. Euclidean space only has a polynomial growth with insufficient discrimination capacity. In contrast, hyperbolic space can embed trees without distortion due to its exponential volume growth. Motivated by this, we embed both 3D shapes and language descriptions into hyperbolic space. We next detail our proposed HyperCL with hyperbolic contrastive loss and entailment loss.

![](images/5ad98e268395ebeb6f0ed992836b58809648fafea872664f1da926c50f99461c.jpg)  
Figure 2: We embed 3D shapes in hyperbolic space to preserve exponentially expanding tree-like data. In addition, we propose an entailment loss that enforces the relation in which caption entails 3D shapes.

## 3.3.1 HYPERBOLIC CONTRASTIVE LEARNING.

As shown in Figure 2 (b), we first map $\widetilde { \mathbf e } _ { i } ^ { P } , \widetilde { \mathbf e } _ { i } ^ { T }$ onto hyperbolic space via Equation. 2:

$$
{ \bf h } _ { i , \mathrm { s p a c e } } ^ { T } = \exp \mathbf { m } _ { \bf O } \left( \widetilde { \bf e } _ { i } ^ { T } \right) .\tag{4}
$$

Note that encoders only give space component, time component can be correspondingly derived as mentioned in Section 3.1. In contrast to Euclidean space that uses cosine similarity, we adopt the negative lorentz distance in Equation 1 as similarity measure $\mathcal { K } ( \cdot ) = - d _ { \mathcal { L } } ( \mathbf { u } , \mathbf { v } )$ . Therefore, our hyperbolic contrastive loss is defined as:

$$
\mathcal { L } _ { H } ^ { P  T } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } ( l ( \mathbf { h } _ { i } ^ { P } , \mathbf { h } _ { i } ^ { T } ) + l ( \mathbf { h } _ { i } ^ { T } , \mathbf { h } _ { i } ^ { P } ) ) .\tag{5}
$$

The dynamic range of negative lorentz distance is fundamentally different from the bounded [−1, 1] cosine similarity in Euclidean space. It is unbounded towards negative infinity, allowing for fine grained semantics to be pushed arbitrarily far without saturation. This exponential growth in hyper bolic space naturally induces a larger discriminative capacity for the model.

## 3.3.2 BATCH-WISE ENTAILMENT LOSS.

We further propose to enforce the relation in which language entails 3D shapes: text denotes general concepts, whereas 3D shapes are specific instantiations of those concepts. To capture this latent vision-language hierarchy, we define an entailment cone for each text $\{ \mathbf { h } _ { i } ^ { T } \} _ { i = 1 } ^ { B }$ , see Figure 2 (b) for an illustration in two dimensions. This cone is defined by half-aperture:

$$
\mathrm { a p e r } ( \mathbf { h } _ { i } ^ { T } ) = \mathrm { s i n } ^ { - 1 } \left( \frac { 2 K } { \sqrt { c } \| \mathbf { h } _ { i , \mathrm { s p a c e } } ^ { T } \| } \right) ,\tag{6}
$$

where K is a small constant 0.1 used for setting boundary conditions near the origin following previous practice (Desai et al., 2023; Ganea et al., 2018a). We now aim to identify and penalize: 1) a positive 3D shape $\mathbf { h } _ { i } ^ { P }$ lies outside this entailment cone of its paired text; and 2) negative 3D shapes $\mathbf { h } _ { j } ^ { \bar { P } } , j = 1 , \dots , \bar { B } , j \neq$ i lie inside this cone. To achieve this, we measure:

$$
\mathrm { e x t } ( \mathbf { u } , \mathbf { v } ) = \mathrm { c o s } ^ { - 1 } \left( \frac { \mathbf { v } _ { \mathrm { t i m e } } + \mathbf { u } _ { \mathrm { t i m e } } c \langle \mathbf { u } , \mathbf { v } \rangle _ { \mathcal { L } } } { \| \mathbf { u } _ { \mathrm { s p a c e } } \| \sqrt { ( c \langle \mathbf { u } , \mathbf { v } \rangle _ { \mathcal { L } } ) ^ { 2 } - 1 } } \right) .\tag{7}
$$

If the exterior angle $\mathrm { e x t } ( \mathbf { h } _ { i } ^ { T } , \mathbf { h } _ { i } ^ { P } )$ between paired text and 3D shape is smaller than $\mathrm { a p e r } ( \mathbf { h } _ { i } ^ { T } )$ , then no loss is incurred, otherwise it will be penalized to force the partial-order relation, and vice versa for unpaired 3D shapes and texts. This is captured by the following objective:

$$
\Delta _ { i j } = \mathrm { e x t } ( \mathbf { h } _ { i } ^ { T } , \mathbf { h } _ { j } ^ { P } ) - \mathrm { a p e r } ( \mathbf { h } _ { i } ^ { T } ) , \qquad \mathscr { L } _ { H } ^ { e n t } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } h ( \Delta _ { i i } ) + \frac { 1 } { B ( B - 1 ) } \sum _ { \substack { i = 1 } } ^ { B } \sum _ { j = 1 } ^ { B } h ( \gamma - \Delta _ { i j } ) ,\tag{8}
$$

where $h ( \cdot ) = \operatorname* { m a x } ( 0 , \cdot )$ is hinge function. The second term ensures that ex $\mathrm { t } ( \mathbf { h } _ { i } ^ { T } , \{ \mathbf { h } _ { j } ^ { P } \} _ { j = 1 , j \neq i } ^ { B } ) \geq$ aper(h<sup>T</sup>)+γ, thus encouraging $\{ \mathbf { h } _ { j } ^ { P } \} _ { j = 1 , j \neq i } ^ { B }$ to lie outside the cone with a margin γ. Note that Desai et al. (2023) only imposes a pairwise entailment loss, whereas we additionally constrain negative 3D shapes in a batch to lie outside the cone.

## 4 EXPERIMENTS

## 4.1 IMPLEMENTATION DETAILS

Following Liu et al. (2023); Zhou et al. (2024), we pretrain on the ensembled dataset including Objaverse Deitke et al. (2023), ShapeNet Chang et al. (2015), 3D-FUTURE Fu et al. (2021), and ABO Collins et al. (2022). We adopt PointBERT Yu et al. (2022) initialized from EVA02 Fang et al. (2024) with 85.3M trainable parameters as our point cloud encoder unless otherwise noted. EVA02- E-14-plus from EVA Sun et al. (2023) is used as the frozen text encoder. To accelerate training, we precompute text embeddings $\{ \tilde { \mathbf e } ^ { T } \}$ and cache them offline. AdamW Loshchilov & Hutter (2017) with batch size 256 is adopted as the optimizer, and we train for 120 epochs. The final loss is $\mathcal { L } _ { H } = \mathcal { L } _ { H } ^ { P  T } + \lambda \cdot \mathcal { L } _ { H } ^ { e n t }$ , where λ is a balancing factor and set to 0.2. The margin γ in Equation 8 is set to 0.5. More implementation details and training cost are in the appendix A.1 and A.2.

## 4.2 EVALUATION DATASETS

We adopt two widely used datasets for assessment, including ModelNet40 (MN40) Wu et al. (2015) and Objaverse-LVIS (OLVIS) Deitke et al. (2023). MN40 is a synthetic CAD model dataset. It contains 2.5K testing samples that span 40 categories. OLVIS is a subset of the Objaverse dataset with human-verified category labels. It has 46K samples spanning 1,156 categories, making it suitable for more challenging open-vocabulary and zero-shot 3D shape classification and retrieval.

Our curated benchmark. Astonishingly, despite its manual labeling, many samples in OLVIS are mislabeled, likely due to varied labeling quality. Such noise can lead to misleading conclusions, as it obscures whether a model truly distinguishes fine-grained semantics or simply benefits from annotation artifacts. To obtain a more reliable testbed for cross-modal alignment, we manually verify each 3D shape in OLVIS. Detailed procedures and statistics are listed in appendix A.4 and A.5. We term the resulting benchmark as cleaned Objaverse-LVIS (c-OLVIS), it contains around 7.7K highquality samples across 846 categories. c-OLVIS is more trustworthy, by eliminating label noise, it ensures performance gains stem from genuine improvements in 3D semantic understanding rather than fitting to erroneous annotations, while also maintaining a large concept pool to assess real-world 3D understanding.

## 4.3 MAIN COMPARISON

Zero-shot 3D shape classification. We compare with strong baselines under different pre-training data regimes and model parameters in Table 3. We do not compare with MM-MixingWang et al.

Table 3: Comparison on zero-shot 3D shape classification with state-of-the-art methods. To make a comprehensive comparison, we test other methods on our c-OLVIS benchmark if the model weights are released and available. Note that Uni3D-L and Uni3D-G are 3.6× and 11.9× more parameters than ours. The best is bold, and second-best is underlined.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Data</td><td colspan="3">OLVIS</td><td colspan="3">c-OLVIS</td><td colspan="3">MN40</td></tr><tr><td>Acc@1 Acc@3 Acc@5 Acc@1 Acc@3 Acc@5 Acc@1 Acc@3 Acc@5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ReCon Qi et al. (2023) CLIP2Point Huang et al. (2023) ULIP Xue et al. (2023)</td><td rowspan="5">ShapeNet</td><td>1.1 2.7 6.2</td><td>2.7 5.8 13.6</td><td>3.7 7.9 17.9</td><td>- 9.9</td><td>19.1</td><td>24.1</td><td>61.2 49.5 60.4</td><td>73.9 71.3 79.0</td><td>78.1 81.2 84.4</td></tr><tr><td>OpenShape Liu et al. (2023)</td><td>10.8</td><td>20.2</td><td>25.0</td><td>16.5</td><td>27.3</td><td>32.1</td><td>70.3</td><td>86.9</td><td>91.3</td></tr><tr><td>TÀMM Żhang et al. (2024)</td><td>13.7</td><td>24.2</td><td>29.2</td><td>23.9</td><td>35.5</td><td>41.2</td><td>73.1</td><td>88.5</td><td>91.9</td></tr><tr><td>HyperCL (ours)</td><td>14.1</td><td>24.1</td><td>29.2</td><td>25.6</td><td>37.2</td><td>43.0</td><td>76.0</td><td>89.1</td><td>93.4</td></tr><tr><td>ULIP2 Xue et al. (2024)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ULIP Xue et al. (2023)</td><td>(w/o LVIS) + ShapeNet</td><td>46.3</td><td></td><td>75.0</td><td></td><td></td><td></td><td>84.0</td><td></td><td>97.2</td></tr><tr><td>OpenShape Liu et al. (2023) Uni3D-B Zhou et al. (2024)</td><td>Ensembled (w/o LVIS)</td><td>21.4 39.1</td><td>38.1 60.8</td><td>46.0 68.9</td><td>60.9</td><td>81.6</td><td>86.7</td><td>71.4 85.3</td><td>84.4 96.2</td><td>89.2 97.4</td></tr><tr><td></td><td>45.9</td><td>67.4</td><td></td><td>74.8</td><td>72.6</td><td>89.2</td><td>92.7</td><td>86.1</td><td>97.4</td><td>98.7</td></tr><tr><td>Uni3D-L Zhou et al. (2024)</td><td>45.9</td><td>67.2</td><td>74.3</td><td></td><td>73.4</td><td>89.6</td><td>93.0</td><td>86.8</td><td>96.6</td><td>97.8</td></tr><tr><td>TAMM Zhang et al. (2024)</td><td>42.0</td><td>63.6</td><td>71.7</td><td></td><td>67.6</td><td>85.4</td><td>89.9</td><td>86.3</td><td>96.6</td><td>98.1</td></tr><tr><td>HyperCL (ours)</td><td>46.6</td><td>67.2</td><td></td><td>74.3</td><td>74.8</td><td>89.8</td><td>92.9</td><td>88.3</td><td>97.3</td><td>98.3</td></tr><tr><td>ULIP2 Xue et al. (2024)</td><td>Objaverse</td><td>50.6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>+ ShapeNet</td><td></td><td></td><td>79.1</td><td>78.8</td><td>92.9</td><td>95.6</td><td>84.7</td><td></td><td>97.1</td></tr><tr><td>ULIP Xue et al. (2023)</td><td></td><td>26.8</td><td>44.8</td><td>52.6</td><td></td><td></td><td></td><td>75.1</td><td>88.1</td><td>93.2</td></tr><tr><td>OpenShape Liu et al. (2023)</td><td>46.8</td><td>69.1</td><td></td><td>77.0</td><td>70.3</td><td>90.0</td><td>93.5</td><td>84.4</td><td>96.5</td><td>98.0</td></tr><tr><td>Uni3D-B Zhou et al. (2024)</td><td>51.7</td><td>74.0</td><td>80.8</td><td></td><td></td><td></td><td>97.7</td><td>86.3</td><td>96.5</td><td>97.9</td></tr><tr><td></td><td>53.1</td><td></td><td></td><td></td><td>81.6</td><td>95.3</td><td></td><td></td><td></td><td></td></tr><tr><td>Uni3D-L Zhou et al. (2024)</td><td>Ensembled</td><td>74.7</td><td></td><td>81.5</td><td>83.7</td><td>96.4</td><td>98.3</td><td>87.1</td><td>96.4</td><td>97.9</td></tr><tr><td>Uni3D-G Zhou et al. (2024)</td><td>53.5</td><td>75.3</td><td></td><td>81.7</td><td>84.1</td><td>96.7</td><td>98.3</td><td>87.4</td><td>98.0</td><td>98.9</td></tr><tr><td>TAMM Zhang et al. (2024)</td><td>50.7 54.4</td><td>73.2</td><td></td><td>80.6</td><td>80.2</td><td>95.4</td><td>97.6</td><td>85.0</td><td>96.6</td><td>98.1</td></tr><tr><td>HyperCL (ours)</td><td></td><td>75.3</td><td></td><td>81.7</td><td>86.2</td><td>97.3</td><td>98.7</td><td>87.8</td><td>97.1</td><td>98.0</td></tr></table>

(2025), MixCon3D Gao et al. (2024b), and MRD Wang et al. (2024) as their model weights or codes are not released. When pretrained on ShapeNet containing only 50K objects, we use a smaller model with 21.7M parameters. As shown in Table 3, our method achieves state-of-the-art performance across all benchmarks under the same parameter budget. Previous best model Uni3D-B achieves 81.6 on our c-OLVIS dataset, whereas we boost the performance to 86.2 with 4.6 absolute improvement. Besides, Uni3D scales its model to 306.7M (Uni3D-L) and 1B parameters (Uni3D-G) that is 3.6x and 11.9x larger than our model, however we surpass both Uni3D-L and Uni3D-G with non-trivial margin. Scaling to more parameters requires excessively long training time, and a strong regularization strategy is needed to avoid overfitting, hence we do not scale models but rather focus on improving performance via hyperbolic learning with smaller parameter budget. The strong performance indicates that our HyperCL learns more discriminative 3D representations that generalize well to challenging open-world scenarios.

Zero-shot cross-modal 3D retrieval. While prior work Zhou et al. (2024) has shown the potential to perform 3D shape retrieval, we aim to quantitatively assess cross-modal 3D retrieval capabilities. To build such a dataset, we take the intersection of 3D shapes that have human written captions in Cap3D Luo et al. (2023) and those in OLVIS as test samples, leaving us with 4,421 unique shapes spanning 879 categories. As shown in Table 4, our model consistently achieves the best performance, with particularly clear advantages in the more challenging text-to-3D retrieval setting. Under the strict zero-shot setting where models are pre-trained on the ensembled dataset without LVIS (never seeing any test samples during training), our approach outperforms all methods, achieving a notable 3.3% gain on text-to-3D retrieval compared to second best method TAMM Zhang et al. (2024). When pretrained on full ensembled corpus, we further push the performance to 29.1. Both Uni3D and ULIP2 achieve competitive performance on 3D-to-text retrieval, but fail to maintain the same performance on text-to-3D retrieval. In contrast, due to our entailment loss, paired 3D shapes are enforced to lie inside the entailment cone, and vice versa for unpaired 3D shapes, making text-to-3D retrieval more effective while also enhancing 3D-to-text retrieval performance.

Table 4: Zero-shot cross-modal 3D retrieval results. For a fair comparison, when evaluating other methods, we use their sampled point clouds from 3D meshes. Recall@K is used as evaluation metric. The best is bold, second-best is underlined. $\mathrm { P }  \mathrm { T }$ denote 3D-to-text retrieval, $\mathrm { T }  \mathrm { P }$ is text-to-3D retrieval. Note Uni3D-L and Uni3D-G are 3.6× and $1 1 . 9 \times$ more parameters than ours.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Data</td><td colspan="3">P →T</td><td colspan="3"> $\mathrm { T }  \mathrm { P }$ </td></tr><tr><td>R@1</td><td>R@3</td><td>R@5</td><td>R@1</td><td>R@3</td><td>R@5</td></tr><tr><td>ULIP2 Xue et al. (2024)</td><td>ShapeNet</td><td>1.0 3.9</td><td>2.4 9.4</td><td>3.6 12.8</td><td>1.0 4.9</td><td>2.7 10.5</td><td>3.9 14.4</td></tr><tr><td>TAMM Zhang et al. (2024) HyperCL (Ours)</td><td></td><td>6.0</td><td>12.1</td><td>17.1</td><td>7.0</td><td>14.7</td><td>19.2</td></tr><tr><td>OpenShape Liu et al. (2023) Uni3D-B Zhou et al. (2024) Uni3D-L Zhou et al. (2024) Uni3D-G Zhou et al. (2024)</td><td rowspan="6">Ensembled</td><td>19.6 22.0</td><td>35.2 39.3</td><td>45.5 48.8</td><td>20.4 18.5</td><td>37.1 34.1</td><td>46.0 42.7</td></tr><tr><td></td><td>21.9</td><td>40.1</td><td>49.3</td><td>18.7</td><td>34.2</td><td>44.4</td></tr><tr><td>(w/o LVIS)</td><td>23.5</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TAMM Zhang et al. (2024)</td><td>19.8</td><td>40.9</td><td>50.4</td><td>21.1</td><td>36.6</td><td>45.6</td></tr><tr><td></td><td>37.2</td><td>46.6</td><td>21.9</td><td></td><td>38.5</td><td>48.0</td></tr><tr><td>HyperCL (Ours)</td><td>24.6</td><td>42.0</td><td>50.9</td><td>25.2</td><td>43.4</td><td>52.4</td></tr><tr><td>ULIP2 Xue et al. (2024)</td><td>Objaverse + ShapeNet</td><td>26.3</td><td>45.2</td><td>55.1</td><td>24.3</td><td>43.3</td><td>52.7</td></tr><tr><td>OpenShape Liu et al. (2023) Uni3D-B Zhou et al. (2024)</td><td></td><td>22.2 24.9</td><td>39.8</td><td>49.3</td><td>23.6</td><td>42.5</td><td>52.3</td></tr><tr><td>Uni3D-L Zhou et al. (2024)</td><td>Ensembled</td><td>26.0</td><td>43.2 44.9</td><td>53.2 54.8</td><td>21.7</td><td>39.1</td><td>49.0</td></tr><tr><td>Uni3D-G Zhou et al. (2024)</td><td></td><td></td><td></td><td></td><td>22.1</td><td>39.4</td><td>48.1</td></tr><tr><td>TAMM Zhang et al. (2024)</td><td></td><td>26.6</td><td>45.6</td><td>55.5</td><td>22.3</td><td>40.5</td><td>50.2</td></tr><tr><td></td><td></td><td>23.1</td><td>41.2</td><td>51.9</td><td>25.5</td><td>44.1</td><td>53.4</td></tr><tr><td>HyperCL (Ours)</td><td></td><td>27.6</td><td>45.9</td><td>54.8</td><td>29.1</td><td>48.4</td><td>57.4</td></tr></table>

## 4.4 ABLATION STUDY

We thoroughly investigate our proposed HyperCL in this section. To reduce training cost, the number of epochs is set to 60.

Table 5: Ablation study on loss components. $\mathcal { L } _ { E } ^ { P  T }$ is the baseline. it does not contrast with rendered images, but adopts cosine similarity instead of negative lorentz distance.

<table><tr><td></td><td> $\mathcal { L } _ { E } ^ { P  T }$ </td><td> $\mathcal { L } _ { H } ^ { P  T }$ </td><td> $\mathcal { L } _ { H } ^ { P  T } + \mathcal { L } _ { H } ^ { e n t }$ </td></tr><tr><td>Acc@1</td><td></td><td></td><td></td></tr><tr><td>OLVIS</td><td>52.3</td><td> $5 3 . 2 _ { \uparrow 0 . 9 }$ </td><td> $5 3 . 4 _ { \uparrow 1 . 1 }$ </td></tr><tr><td>c-OLVIS</td><td>82.9</td><td> $8 4 . 2 _ { \uparrow 1 . 3 }$ </td><td> $8 4 . 6 _ { \uparrow 1 . 7 }$ </td></tr><tr><td>MN40</td><td>85.7</td><td> $8 6 . 3 _ { \uparrow 0 . 6 }$ </td><td> $8 7 . 9 _ { \uparrow 2 . 2 }$ </td></tr><tr><td>R@1</td><td></td><td></td><td></td></tr><tr><td>P→T</td><td>25.2</td><td> $2 6 . 0 \substack { \uparrow 0 . 8 }$ </td><td> $2 7 . 0 \ast _ { 1 . 8 }$ </td></tr><tr><td>T→P</td><td>25.8</td><td> $2 7 . 1 \ L _ { \uparrow 1 . 3 }$ </td><td> $2 8 . 4 _ { \uparrow 2 . 6 }$ </td></tr><tr><td>R@5</td><td></td><td></td><td></td></tr><tr><td>P→T</td><td>53.4</td><td> $5 4 . 2 _ { \uparrow 0 . 8 }$ </td><td> $5 5 . 6 _ { \uparrow 2 . 2 }$ </td></tr><tr><td>T→P</td><td>54.7</td><td> $5 5 . 4 _ { \uparrow 0 . 7 }$ </td><td> $5 6 . 3 _ { \uparrow 1 . 6 }$ </td></tr></table>

![](images/6220c38c18aeb90c281d922303c79e7445ee431478573b7bb266b95a88591e36.jpg)  
OLVIS  c-OLVIS ▲ MN40 P→T  T→P -- learnable c (init. 4.0)  
Figure 3: Effect of curvature c on loss $\mathcal { L } _ { H } ^ { P  T }$ Dashed lines denote learnable curvature.

Effect of individual loss. Table 5 shows the effect of each loss. We first compare against euclidean baseline $\mathcal { L } _ { E } ^ { P  T }$ in Equation. 3. The hyperbolic loss $\mathcal { L } _ { H } ^ { P  T }$ alone surpasses baseline by a large margin. As argued in Section 1 and 3.3, this proves that the underlying geometry is tree-like structure and is suitable to be modeled via hyperbolic geometry. The entailment loss further improves the performance. Together the proposed two losses enable discriminative recognition and partial-order relation between texts and 3D instantiations.

![](images/11faf98b37ca1c0592415fd0093e79c53c1b36ad495632c11061826b1af4faf4.jpg)

![](images/f0386271c11bbb127cbabe48166aa8f35d5eb4ba4448fdc1ba84045088c33aa1.jpg)

![](images/d6f7bd802bdc07e0bcafbda6ff62025212d2a1a6a24cfd37943e2126c20f468a.jpg)  
Figure 4: Embedding distances from [root]. Text em- Figure 5: Shape traversal, moving tobeddings consistently lie closer to [root] than 3D shape ward [ROOT], our model retrieves catembeddings encoded by our model. egories at multiple levels of abstraction.

Effect of curvature c. The curvature measures how hyperbolic space deviates from flat Euclidean space. When it goes to zero, the hyperbolic space gradually degrades to Euclidean space. As shown in Figure 3, both classification and retrieval performance increase from $c = 1$ and stabilizes at $c \geq 2$ As c decreases from 1, performance initially deteriorates, but gradually recovers when it approaches 0 as the underlying geometry becoming increasingly euclidean. However, setting $c = 0 . 0 1$ still lags behind $c \geq 2$ . This v-shaped curve suggests that moderate curvature may provide neither the strong hierarchical modeling capacity of highly curved hyperbolic space nor the near-Euclidean behavior. Making c a learnable parameter (initialized from 4) is roughly the same as fixed c = 4 since c is decreasing during training and the model has to constantly adapt to this changing geometry. Hence, we set c to 4 and it remains fixed.

## 4.5 QUALITATIVE RESULTS

If the partial-order relation of language entails 3D shapes is imposed, $h _ { i } ^ { T }$ should lie closer to [ROOT] than $h _ { i } ^ { P }$ , since text denotes generic concepts that subsume specific 3D instantiations. Following Desai et al. (2023), the origin of lorentz model is used as [ROOT]. As shown in Figure 4, texts and 3D shapes are clearly separated, with text concentrated near the origin and 3D shapes scattered toward the hyperbolic boundary where the distance becomes larger to encompass fine-grained concepts. Conversely, OpenShape Liu et al. (2023) inevitably overlaps two modalities with a large portion. To further examine this vision-language hierarchy, we perform shape traversal. Between a query 3D representation and origin [ROOT], we interpolate evenly spaced points along the hyperbolic geodesic connecting them, and use each intermediate point to retrieve its nearest category. The retrieved categories should become progressively more abstract as we move toward [ROOT]. As shown in Figure 5, euclidean baselines Liu et al. (2023); Zhang et al. (2024) tend to directly jump to [ROOT], whereas our model yields a much smoother trajectory of meaningful intermediate concepts, revealing multiple levels of abstraction.

## 5 CONCLUSION

In this paper, we present HyperCL for learning transferable 3D representations. We demonstrate that learning from language supervision only effectively bypasses the inability of VLMs to encode reliable semantics from rendered images. We leverage hyperbolic geometry to enhance discrimination and model the inherent vision-language hierarchy that language entails 3D shapes. The proposed hyperbolic contrastive and batchwise entailment losses effectively preserves this hierarchical structure. Furthermore, our manually verified c-OLVIS benchmark serves as a rigorous evaluation. Experimental results show that our method achieves state-of-the-art performance on both 3D shape classification and cross-modal retrieval tasks.

Limitations Though the language descriptions are relatively robust than rendered images, the problems caused by self-occlusion and rendering artifacts may still be propagated into captions. 3D captioning is best performed without rendered images as proxy via 3D LLMs that directly take 3D shapes as input. However this is beyond our paper’s scope and we leave this to future work.

## ACKNOWLEDGEMENTS

This work was supported by National Science and Technology Major Project (No. 2025YFE0113500), the National Science Fund for Distinguished Young Scholars (No. 62525605), the National Natural Science Foundation of China (No. U25B2066, No. U22B2051, and No. 62302411), and the Xiamen Municipal Science and Technology Bureau, China (No. 3502ZC-QXT2024009).

## AI USE STATEMENT

In this work, we used generative AI tools for supporting qualitative and thematic data analysis. We have not used generative AI tools for other tasks with required disclosure, and the rest of the required disclosure tasks are not applicable to this work. Additionally, we used generative AI tools for sourcing/searching for information and editing a research paper to improve readability. We have reviewed all AI-assisted work. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Mina Ghadimi Atigh, Julian Schoep, Erman Acar, Nanne Van Noord, and Pascal Mettes. Hyperbolic image segmentation. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4443–4452. IEEE, 2022.

Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J´ egou, Julien Mairal, Piotr Bojanowski, and´ Armand Joulin. Emerging properties in self-supervised vision transformers. In 2021 IEEE/CVF international conference on computer vision (ICCV), pp. 9630–9640. IEEE, 2021.

Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015.

Chih-Yao Chen, Tun Min Hung, Yi-Li Hsu, and Lun-Wei Ku. Label-aware hyperbolic embeddings for fine-grained emotion classification. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 10947–10958, 2023.

Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu, Xi Zhang, Tomas F Yago Vicente, Thomas Dideriksen, Himanshu Arora, et al. Abo: Dataset and benchmarks for real-world 3d object understanding. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 21094–21104. IEEE, 2022.

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 13142–13153, 2023.

Karan Desai, Maximilian Nickel, Tanmay Rajpurohit, Justin Johnson, and Shanmukha Ramakrishna Vedantam. Hyperbolic image-text representations. In International Conference on Machine Learning, pp. 7694–7731. PMLR, 2023.

Thang Doan, Xin Li, Sima Behpour, Wenbin He, Liang Gou, and Liu Ren. Hyp-ow: Exploiting hierarchical structure learning with hyperbolic distance enhances open world object detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 1555–1563, 2024.

Aleksandr Ermolov, Leyla Mirvakhabova, Valentin Khrulkov, Nicu Sebe, and Ivan Oseledets. Hyperbolic vision transformers: Combining improvements in metric learning. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7399–7409. IEEE, 2022.

Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva-02: A visual representation for neon genesis. Image and Vision Computing, 149:105171, 2024.

Herve Fournier, Anas Ismail, and Antoine Vigneron. Computing the gromov hyperbolicity of a´ discrete metric space. Information Processing Letters, 115(6-8):576–579, 2015.

Huan Fu, Rongfei Jia, Lin Gao, Mingming Gong, Binqiang Zhao, Steve Maybank, and Dacheng Tao. 3d-future: 3d furniture shape with texture. International Journal of Computer Vision, 129 (12):3313–3337, 2021.

Octavian Ganea, Gary Becigneul, and Thomas Hofmann. Hyperbolic entailment cones for learn-´ ing hierarchical embeddings. In International conference on machine learning, pp. 1646–1655. PMLR, 2018a.

Octavian Ganea, Gary Becigneul, and Thomas Hofmann. Hyperbolic neural networks.´ Advances in neural information processing systems, 31, 2018b.

Peng Gao, Shijie Geng, Renrui Zhang, Teli Ma, Rongyao Fang, Yongfeng Zhang, Hongsheng Li, and Yu Qiao. Clip-adapter: Better vision-language models with feature adapters. International journal ofcomputer vision, 132(2):581–595, 2024a.

Yipeng Gao, Zeyu Wang, Wei-Shi Zheng, Cihang Xie, and Yuyin Zhou. Sculpting holistic 3d representation in contrastive language-image-3d pre-training. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 22998–23008. IEEE, 2024b.

Songwei Ge, Shlok Mishra, Simon Kornblith, Chun-Liang Li, and David Jacobs. Hyperbolic contrastive learning for visual representations beyond objects. In 2023 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 6840–6849. IEEE, 2023.

Yunhui Guo, Xudong Wang, Yubei Chen, and Stella X Yu. Clipped hyperbolic classifiers are superhyperbolic classifiers. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1–10. IEEE, 2022.

Tianyu Huang, Bowen Dong, Yunhan Yang, Xiaoshui Huang, Rynson WH Lau, Wanli Ouyang, and Wangmeng Zuo. Clip2point: Transfer clip to point cloud classification with image-depth pretraining. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 22100– 22110. IEEE, 2023.

Valentin Khrulkov, Leyla Mirvakhabova, Evgeniya Ustinova, Ivan Oseledets, and Victor Lempitsky. Hyperbolic image embeddings. In 2020 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 6417–6427. IEEE, 2020.

Fanjie Kong, Yanbei Chen, Jiarui Cai, and Davide Modolo. Hyperbolic learning with synthetic captions for open-world detection. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 16762–16771. IEEE, 2024.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pp. 19730–19742. PmLR, 2023.

Feng Liang, Bichen Wu, Xiaoliang Dai, Kunpeng Li, Yinan Zhao, Hang Zhang, Peizhao Zhang, Peter Vajda, and Diana Marculescu. Open-vocabulary semantic segmentation with mask-adapted clip. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7061–7070. IEEE, 2023.

Minghua Liu, Ruoxi Shi, Kaiming Kuang, Yinhao Zhu, Xuanlin Li, Shizhong Han, Hong Cai, Fatih Porikli, and Hao Su. Openshape: Scaling up 3d shape representation towards open-world understanding. Advances in neural information processing systems, 36:44860–44879, 2023.

Yingjie Liu, Pengyu Zhang, Ziyao He, Mingsong Chen, Xuan Tang, and Xian Wei. Hyperbolic contrastive learning for hierarchical 3d point cloud embedding. arXiv preprint arXiv:2501.02285, 2025.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Tiange Luo, Chris Rockwell, Honglak Lee, and Justin Johnson. Scalable 3d captioning with pretrained models. Advances in Neural Information Processing Systems, 36:75307–75337, 2023.

Tiange Luo, Justin Johnson, and Honglak Lee. View selection for 3d captioning via diffusion ranking. In European Conference on Computer Vision, pp. 180–197. Springer, 2024.

Paolo Mandica, Luca Franco, Konstantinos Kallidromitis, Suzanne Petryk, and Fabio Galasso. Hyperbolic learning with multimodal large language models. In European Conference on Computer Vision, pp. 382–398. Springer, 2024.

Maximillian Nickel and Douwe Kiela. Poincare embeddings for learning hierarchical representa- ´ tions. Advances in neural information processing systems, 30, 2017.

Maximillian Nickel and Douwe Kiela. Learning continuous hierarchies in the lorentz model of hyperbolic geometry. In International conference on machine learning, pp. 3779–3788. PMLR, 2018.

Avik Pal, Max Van Spengler, Guido D’Amely di Melendugno, Alessandro Flaborea, Fabio Galasso, and Pascal Mettes. Compositional entailment learning for hyperbolic vision-language models. In International Conference on Learning Representations, volume 2025, pp. 87371–87399, 2025.

Zelin Peng, Zhengqin Xu, Zhilin Zeng, Changsong Wen, Yu Huang, Menglin Yang, Feilong Tang, and Wei Shen. Understanding fine-tuning clip for open-vocabulary semantic segmentation in hyperbolic space. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4562–4572. IEEE, 2025.

Zekun Qi, Runpei Dong, Guofan Fan, Zheng Ge, Xiangyu Zhang, Kaisheng Ma, and Li Yi. Contrast with reconstruct: Contrastive 3d representation learning guided by generative pretraining. In International Conference on Machine Learning, pp. 28223–28243. PMLR, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Sameera Ramasinghe, Violetta Shevchenko, Gil Avraham, and Ajanthan Thalaiyasingam. Accept the modality gap: An exploration in the hyperbolic space. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 27253–27262. IEEE, 2024.

John G Ratcliffe. Foundations of hyperbolic manifolds. Springer, 2006.

Ryohei Shimizu, Yusuke Mukuta, and Tatsuya Harada. Hyperbolic neural networks++. arXiv preprint arXiv:2006.08210, 2020.

Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023.

Huiqun Wang, Yiping Bao, Panwang Pan, Zeming Li, Xiao Liu, Ruijie Yang, and Di Huang. Multimodal relation distillation for unified 3d representation learning. In European Conference on Computer Vision, pp. 364–381. Springer, 2024.

Jiaze Wang, Yi Wang, Ziyu Guo, Renrui Zhang, Donghao Zhou, Guangyong Chen, Anfeng Liu, and Pheng-Ann Heng. Mm-mixing: Multi-modal mixing alignment for 3d understanding. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 7744–7752, 2025.

Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Linguang Zhang, Xiaoou Tang, and Jianxiong Xiao. 3d shapenets: A deep representation for volumetric shapes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1912–1920, 2015.

Le Xue, Mingfei Gao, Chen Xing, Roberto Mart´ın-Mart´ın, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, and Silvio Savarese. Ulip: Learning a unified representation of language, images, and point clouds for 3d understanding. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1179–1189. IEEE, 2023.

Le Xue, Ning Yu, Shu Zhang, Artemis Panagopoulou, Junnan Li, Roberto Mart´ın-Mart´ın, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, et al. Ulip-2: Towards scalable multimodal pretraining for 3d understanding. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 27081–27091. IEEE, 2024.

Xumin Yu, Lulu Tang, Yongming Rao, Tiejun Huang, Jie Zhou, and Jiwen Lu. Point-bert: Pretraining 3d point cloud transformers with masked point modeling. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19291–19300. IEEE, 2022.

Zhihao Zhang, Shengcao Cao, and Yu-Xiong Wang. Tamm: Triadapter multi-modal learning for 3d shape understanding. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 21413–21423. IEEE, 2024.

Junsheng Zhou, Jinsheng Wang, Baorui Ma, Yu-Shen Liu, Tiejun Huang, and Xinlong Wang. Uni3d: Exploring unified 3d representation at scale. In International Conference on Learning Representations, volume 2024, pp. 46766–46782, 2024.

## A APPENDIX

## A.1 MORE IMPLEMENTATION DETAILS

As listed in Table 6, we mainly follow previous practices Liu et al. (2023); Zhou et al. (2024); Xue et al. (2024) in multi-modal 3D contrastive pre-training for data augmentation and model training hyperparameters.

Table 6: Hyperparameters used in our HyperCL.
<table><tr><td>Hyperparameters</td><td>Value</td><td>Descriptions</td></tr><tr><td>Normalization</td><td>√</td><td>normalize xyz coordinates of point cloud into [-1, 1]</td></tr><tr><td>Point dropout</td><td>0.875</td><td>the maximum ratio of dropped points is 0.875</td></tr><tr><td>Random scale</td><td>[0.8, 1.25]</td><td>uniformly scale points</td></tr><tr><td>Random shift</td><td>[-0.1, 0.1]</td><td>shift x, y, z coordinates independently</td></tr><tr><td>Random rotation</td><td>√</td><td>small rotation jittering around x, y, z axes</td></tr><tr><td>Random rotation around gravity-axis</td><td>√</td><td>rotate around gravity-axis for learning rotation invariance</td></tr><tr><td>Random RGB drop</td><td>√</td><td>replace the RGB values of points with 0.4</td></tr><tr><td>CLIP model</td><td>ViT-E/14+</td><td>CLIP model used to compute offline text embeddings</td></tr><tr><td>Number of points</td><td>10,000</td><td>number of sampled points from 3D mesh</td></tr><tr><td>Hyperbolic temperature τ</td><td>0.06</td><td></td></tr><tr><td>Learning rate</td><td>0.001</td><td></td></tr><tr><td>Learning rate schedule</td><td>WarmUpStepLR with decaying factor 0.967</td><td></td></tr><tr><td>Optimizer</td><td>AdamW with betas = (0.9, 0.98) and eps=1e-6</td><td></td></tr><tr><td>Weight decay</td><td>0.01</td><td>no weight decay on temperature τ, curvature c</td></tr><tr><td>Batch Size per GPU</td><td>256</td><td></td></tr><tr><td>Epochs</td><td>120</td><td></td></tr><tr><td>AMP</td><td>V</td><td>use FP16 for training</td></tr><tr><td>GPUs</td><td>8 × NVIDIA A800 (PCIe, 80GB)</td><td></td></tr></table>

## A.2 TRAINING COST

Since both our hyperbolic contrastive loss and entailment loss use the already well-optimized (inverse) hyperbolic triangle functions provided by PyTorch, and the main computation stems from point cloud encoder whose parameters still reside in Euclidean space and can be optimized via AdamW, hence both the memory footprint and FLOPs maintain the same as euclidean baselines as shown in Table 7. The speed slightly decreases to 3.36 it/s compared with $\mathcal { L } _ { E } ^ { P  T }$ , however, this is because the host-side kernel-launch overhead from a chain of fine-grained primitives such as norm, acosh, clamp, sinh rather than computational burden.

Table 7: Comparison of training cost.
<table><tr><td></td><td>Speed (it/s)</td><td>Peak Memory (GB)</td><td>GFLOPs</td></tr><tr><td> $\mathcal { L } _ { E } ^ { P  I } + \mathcal { L } _ { E } ^ { P  T }$ </td><td>3.30</td><td>46.65</td><td>72.05</td></tr><tr><td> $\mathcal { L } _ { E } ^ { P  T }$ </td><td>3.43</td><td>46.65</td><td>72.05</td></tr><tr><td> $\mathcal { L } _ { H } ^ { P  T } + \mathcal { L } _ { H } ^ { e n t }$ </td><td>3.36</td><td>46.65</td><td>72.05</td></tr></table>

## A.3 MORE DETAILS ON HYPERBOLICITY

Gromov hyperbolicity quantifies how closely a metric space resembles the structure of a tree. Let $( \mathcal { M } , d )$ be a metric space, where M is the set of representations and $d : \mathcal { M } \times \mathcal { M } $ R is the distance measure. In our experiments, each representation is a 3D shape embedding $\boldsymbol { \tilde { e } } _ { i } \in \mathbb { R } ^ { D }$ produced by Uni3D (Zhou et al., 2024), and d is euclidean metric $d ( \tilde { e } _ { i } , \tilde { e } _ { j } ) = \lVert \tilde { e } _ { i } - \tilde { e } _ { j } \rVert _ { 2 }$

For a base point $\tilde { e } _ { b } \in \mathcal { M }$ , the Gromov product of $\tilde { e } _ { i }$ and ${ \tilde { e } } _ { j }$ is defined as

$$
( \tilde { e } _ { i } \mid \tilde { e } _ { j } ) _ { \tilde { e } _ { b } } = \frac { 1 } { 2 } \left( d ( \tilde { e } _ { i } , \tilde { e } _ { b } ) + d ( \tilde { e } _ { j } , \tilde { e } _ { b } ) - d ( \tilde { e } _ { i } , \tilde { e } _ { j } ) \right) .\tag{9}
$$

In a rooted tree, it equals the length of the initial path shared by the paths from $\tilde { e } _ { b }$ to ${ \tilde { e } } _ { i }$ and ${ \tilde { e } } _ { j }$ Thus, a larger Gromov product indicates a deeper branching point, whereas a smaller value indicates branching closer to the base point.

Given three representations $\tilde { e } _ { i } , \tilde { e } _ { j }$ , and $\tilde { e } _ { k }$ , a tree satisfies the following Gromov product inequality:

$$
( \tilde { e } _ { i } \mid \tilde { e } _ { k } ) _ { \tilde { e } _ { b } } \geq \operatorname* { m i n } \left\{ ( \tilde { e } _ { i } \mid \tilde { e } _ { j } ) _ { \tilde { e } _ { b } } , ( \tilde { e } _ { j } \mid \tilde { e } _ { k } ) _ { \tilde { e } _ { b } } \right\} .\tag{10}
$$

The deviation from this tree condition is quantified by the base-point Gromov hyperbolicity:

$$
\delta _ { \tilde { e } _ { b } } = \operatorname* { m a x } _ { \tilde { e } _ { i } , \tilde { e } _ { j } , \tilde { e } _ { k } \in \mathcal { M } } \left[ \operatorname* { m i n } \left\{ ( \tilde { e } _ { i } \mid \tilde { e } _ { j } ) _ { \tilde { e } _ { b } } , ( \tilde { e } _ { j } \mid \tilde { e } _ { k } ) _ { \tilde { e } _ { b } } \right\} - ( \tilde { e } _ { i } \mid \tilde { e } _ { k } ) _ { \tilde { e } _ { b } } \right] .\tag{11}
$$

A tree metric has $\delta _ { \tilde { e } _ { b } } = 0 ;$ , since the above inequality holds for every triple of points. A positive value measures the largest violation of this condition among sampled representations. Therefore, a smaller value indicates a more tree-like structure, whereas a larger value indicates the opposite. To account for the overall scale of the embedding space, we report the relative hyperbolicity:

$$
\delta _ { \mathrm { r e l } } = \frac { 2 \delta _ { \tilde { e } _ { b } } } { \mathrm { d i a m } ( \mathcal { M } ) } , \qquad \mathrm { d i a m } ( \mathcal { M } ) = \operatorname* { m a x } _ { \tilde { e } _ { i } , \tilde { e } _ { j } \in \mathcal { M } } d \bigl ( \tilde { e } _ { i } , \tilde { e } _ { j } \bigr ) .\tag{12}
$$

In practice, we follow Khrulkov et al. (2020); Fournier et al. (2015) to compute the hyperbolicity.

## A.4 MORE DETAILS ON C-OLVIS CONSTRUCTION

Based on OLVIS, we construct a cleaned version of it through manual inspection and refinement. During inspection, for each object we utilize the official Sketchfab website stored in the metadata. The viewer supports rotation and zooming in of the 3D shape, allowing labelers to inspect it from multiple viewpoints and reduce errors caused by self-occlusion. To ensure consistent criteria, each 3D shape will display its assigned category name, associated synonyms, along with its LVIS definition. This information helps reviewers distinguish intended category from polysemy.

We develop a lightweight annotation tool that provides three possible options: accepted (a), rejected (r), and unsure (u). Annotators can submit decision by pressing corresponding key, and the tool automatically opens the online URL using chromium. We evenly split OLVIS into three disjoint subsets and assign each subset to one labeler. Candidates marked as accepted during initial screening phase are passed for a second round of review by other independent labelers. Only candidates accepted in both stages are kept in our c-OLVIS benchmark. This two stage procedure reduces the chances that an individual oversight or inconsistent interpretation propagates into the final benchmark. We encourage annotators to label it as unsure rather than accepted to avoid being deterministically filtered out in second round. Objects marked as unsure in the first round are collected and sent for committee screening as detailed next.

![](images/60b7cbd9bf258102d391c8f60a50524fe61b7e45d91360e5123b9adfd178564f.jpg)  
Figure 6: Examples from the manual verification. For four categories lollipop, birdhouse, combination lock and turkey (food), the upper row shows 3D shapes that were correctly labelled and kept in our c-OLVIS, while the lower row shows OLVIS samples that were mislabelled and removed.

We find that many categories in OLVIS are either intrinsically ambiguous or related with each other via parent-child and synonym relationships. For example, table is the parent concept of dining table and coffee table, while in the meantime, it may also be used as a synonym for desk. We first organize such ambiguous concepts into groups via WordNet and instruct reviewers to press unsure button whenever a 3D shape lies in these groups. Then all reviewers are shown the same unsure object, when the group contains parent-children relationship, we apply a strict rule that it should be assigned to the most specific subclass unless it can not be classified into any subclass in that group, then it will be assigned as the parent class. Only the vote for a concept receives more than half of the reviewers then the object will be assigned to that class, otherwise, the object is discarded.

![](images/fb39f9f7b0289ad942b65e74315dde2c0913fd4e10c3adff4199d7e1f2c3dda3.jpg)

![](images/1201bfdb6c28af01a56ebe09a99abb05c99662b9fc61d09aadf86a5ae68ca2a2.jpg)

![](images/ebe46286e2e85def20653523da66ad3e9ff8aab21bdd01f702b2a6b2581e206d.jpg)  
Figure 7: Statistics of our c-OLVIS. Left: category vocabulary of c-OLVIS, 846 categories in total. Middle: the rank–frequency distribution of class sizes on category-index axis. Right: the number of categories per class-size bin.

We asked three graduate student to independently assess their given split using the same criteria and it takes around 15 days to annotate our c-OLVIS benchmark. The time is mostly spent on waiting for online rendering and committee review for unsure objects. Examples are given in Figure 6.

## A.5 MORE STATISTICS OF C-OLVIS

Compared to OLVIS, c-OLVIS has 7,678 unique 3D shapes from 846 categories, covering 73.2% original categories of OLVIS. This diverse coverage of real-world concepts ensures that our proposed c-OLVIS benchmark remains representative for zero-shot 3D understanding. After rigorous filtering and correction, we remove substantial annotation noise, therefore it provides a more reliable evaluation, allowing the improvement to be more likely to reflect true semantic recognition rather than false exploitation on noisy labels. While at the meantime, even though the accuracy on our c-OLVIS benchmark is typically higher than that on OLVIS, the resulting c-OLVIS benchmark is not merely an easy subset that contains only frequent categories from OLVIS. As shown in Figure 7, categories that contain less 3D shapes are well preserved, the main annotation noise comes from several categories with a large amount of objects (the rightmost bar in right subfigure). Thus, the large accuracy gap between our c-OLVIS and OLVIS can not be explained simply by retaining a large portion of easily recognizable categories. Instead, it is consistent with our manual verification process that removes many noisy or ambiguous instances while preserving low-frequency concepts. This is exactly why annotation quality matters: a reliable benchmark should reward correct semantic recognition rather than penalize it because of incorrect labels.

## A.6 MORE QUALITATIVE RESULTS

We visualize cross-modal retrieval results in Figure 8. Euclidean geometry based models often produce partially correct retrievals, they may match coarse semantics but confuse fine-grained concepts such as car v.s. truck v.s. pickup truck, or miss certain attributes in the query, e.g., failing to recognize the orange color of the scooter. This behavior is consistent with the fact that euclidean geometry embeds tree that has a large amount of concepts with distortion, highlighting the benefit of hyperbolic space for preserving fine-grained semantics.

![](images/ddf569b70e9e8d05716cb6683ebc1441119ad0bcf017b3adc01deb76d9b92448.jpg)

OpenShape old push button flap screened mobile phone model with curved rectangular shape.

TAMM old push button flap screened mobile phone model with curved rectangular shape.

Ours a white calculator

![](images/d5e1a7e70d6e895d66d8f7206710f2d7dc2a23f80fe8079c2b665925883a9371.jpg)

OpenShape dark blue colored toy truck with open rectangular carrier in back.

TAMM a vector image of a white pickup truck. the truck has a black bumper and four black round wheels. Ours white joy jeep car without wheels and roof

a scooter orange in colour with handles at the front and a visible wheel at the front

![](images/fc36f8a7790117bcd505b798ec6162dd4984e76e974a2e468bc1fafb208f7138.jpg)  
Figure 8: Qualitative visualization of retrieval results. Top: 3D → text retrieval (top-1 caption). Bottom: text → 3D retrieval (top-5 shapes). Similarities decrease from left to right.