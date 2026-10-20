# PFedQFT: Personalized Federated Quantum-Inspired Fine-Tuning for Medical Large Language Models

Qihan Huang<sup>1,†</sup>, Lingxia Zhang<sup>1,3,†</sup>, Guye Zhang<sup>1,3</sup>, Zizhu Wang<sup>1,3,\*</sup>, Xiaowei Mao<sup>1,2,3,4,\*,‡</sup> <sup>1</sup>Institute of Fundamental and Frontier Sciences,

University of Electronic Science and Technology of China, Chengdu, Sichuan, China <sup>2</sup>Sichuan Provincial Key Laboratory for Human Disease Gene Study and the Center for Medical Genetics, Department of Laboratory Medicine, Sichuan Academy of Medical Sciences and Sichuan Provincial People’s Hospital, University of Electronic Science and Technology of China, Chengdu, Sichuan, China <sup>3</sup>Key Laboratory of Quantum Physics and Photonic Quantum Information, Ministry of Education, University of Electronic Science and Technology of China, Chengdu, Sichuan, China <sup>4</sup>Sichuan Quantum Science and Technology Institute, Chengdu, Sichuan, China xiaowei.mao@uestc.edu.cn

Abstract—Medical data remain fragmented across institutional silos because of privacy, governance, and ownership constraints. Federated learning enables medical institutions to adapt large language models (LLMs) collaboratively without transferring raw records. However, independently parameterized low-rank adapters incur repeated parameter and communication costs across Transformer projections, while uniform aggregation may limit adaptation to heterogeneous institutional data. We propose PFedQFT, a quantum-inspired framework for personalized federated fine-tuning with a compact shared adapter. Specifically, PFedQFT constructs reusable orthonormal adaptation bases from compact skew-symmetric generators using a truncated matrix-exponential approximation followed by QR orthonormalization. Compatible projections reuse these bases, while a globally shared full core preserves the flexibility of each projection. Each client further maintains a lightweight private residual operator, whose directions are learned globally and whose coefficients remain local. Experiments on seven public medical tasks and multiple LLM backbones show substantial reductions in trainable parameters and per-round adapter upload volume, with competitive predictive performance on most tasks.

Index Terms—Federated Learning, Parameter-Efficient Fine-Tuning, Quantum-Inspired Adaptation, Large Language Models.

## I. INTRODUCTION

Medical institutions continuously generate large and heterogeneous volumes of clinical text, diagnostic records, annotated medical documents, and question answering data. These distributed resources provide valuable evidence for adapting large language models (LLMs) to a wide range of clinical and biomedical language processing tasks [1]. However, consolidating medical records into a centralized training repository is often impractical because of privacy regulations, data governance requirements, and institutional ownership constraints [2]. Federated learning (FL) offers a practical alternative by enabling multiple institutions to collaborate while retaining sensitive records locally [2], [3]. We refer to adapting pretrained models under this distributed protocol as federated fine-tuning (FFT). Nevertheless, hospital servers and clinical edge devices often vary in computing power, memory capacity, and network bandwidth, making repeated optimization and transmission of LLMs prohibitively expensive [4], [5]. As illustrated in Fig. 1, institutional data silos motivate FFT, while high communication costs and heterogeneous local data call for efficient personalized federated adaptation.

![](images/104eacaf3a012b6a764edd522ca9adc9adc799ba83957bdedd4f135e7ad4c20b.jpg)  
Fig. 1. Challenges in FFT of medical LLMs: (a) institutional data silos, (b) high communication costs from repeated model update exchanges, and (c) heterogeneous data across institutions.

In a typical FL workflow, the server distributes the current global parameters, clients optimize them on private local data, and the resulting updates are aggregated to improve the global model [3]. Directly adapting all parameters of an LLM is often impractical because of the associated computation, memory, and communication costs. Parameter-efficient finetuning (PEFT) addresses this limitation by freezing the pretrained backbone and optimizing only a compact set of task parameters [6], [7]. Within FFT, PEFT allows clients to train and exchange lightweight adaptation modules rather than the entire model, reducing local resource consumption and communication overhead while leaving the pretrained backbone weights unchanged [5]. Low-Rank Adaptation (LoRA), for example, represents each weight update with two trainable low-rank factors [7]. However, these factors are typically learned independently for individual projection layers, and their trainable and communicated parameter counts still scale with the dimensions of the adapted modules. Moreover, the adapter payload is exchanged repeatedly over communication rounds, so even a relatively small fraction of the backbone parameters can accumulate into a substantial transmission volume. Reducing this recurring cost requires attention to how adaptation parameters are organized and reused across the model, in addition to how many parameters are trained in each individual module.

Despite these advantages, standard FFT with independently parameterized low-rank modules and fully shared aggregation presents two technical limitations. Conventional low-rank adapters learn a separate pair of projection factors for each adapted layer, requiring approximately $r ( d _ { \mathrm { i n } } + d _ { \mathrm { o u t } } )$ trainable parameters per weight matrix, where r is the adaptation rank and $d _ { \mathrm { i n } }$ and $d _ { \mathrm { o u t } }$ are the input and output dimensions [7]. Independently parameterizing these factors across Transformer layers incurs repeated parameter costs, with limited reuse of adaptation directions across compatible projections. Simply lowering the adaptation rank reduces the parameter budget but also restricts the dimension of the update space. An alternative is to reuse adaptation bases across compatible projections while retaining transformations specific to each projection within those bases. This separates the cost of specifying adaptation directions from the cost of combining them for each projection.

Conventional FL aggregation generally treats all adapter parameters as globally shareable. When medical data are not independently and identically distributed (non-IID), client updates can contain both transferable clinical knowledge and institution specific information, leading to conflicting local objectives and unstable aggregation [4], [8]. Aggregating both components can weaken local specialization, whereas keeping complete adapters private restricts knowledge transfer across institutions [9]. A lightweight private component offers a middle ground: clients can share the main adaptation parameters while retaining a small set of local corrections [9], [10]. The challenge is to provide useful local flexibility without introducing a complete private adapter for every institution. Together, these limitations motivate two related challenges: constructing reusable low-rank adaptation subspaces from compact intrinsic parameters and separating globally collaborative transformations from lightweight client specific adaptation without substantially increasing the communication payload.

Quantum-inspired parameterizations offer a structured perspective on constructing adaptation transformations [11]. In quantum computation, unitary transformations preserve inner products and norms and can be expressed as matrix exponentials of Hermitian generators, $\mathbf { Q } = \exp ( - \mathrm { i } \mathbf { H } )$ [12]. For real-valued adaptation, a corresponding construction uses a real skew-symmetric generator A, whose exponential exp(A) is orthogonal; setting H = iA establishes the connection to the unitary formulation [13]. Applying this orthogonal transformation to a set of standard basis vectors yields orthonormal directions that can serve as adaptation bases. The opportunity for parameter reduction comes from imposing a compact structure on the generator and reusing the resulting bases, rather than from orthogonality alone. This perspective motivates constructing shared adaptation subspaces through structured generator parameters while retaining separate core transformations for each projection and client-specific adjustments.

To address these challenges, we propose Personalized Federated Quantum-Inspired Fine-Tuning (PFedQFT), an efficient method for adapting medical LLMs in distributed environments. Inspired by generator-based unitary transformations, the method uses structured skew-symmetric generators and matrix exponential mappings to construct orthogonal adaptation subspaces [11]. A truncated Taylor approximation followed by thin QR orthonormalization produces the adaptation bases without explicitly forming a dense matrix exponential. Projections with matching module types and matrix shapes reuse these bases, while a full core matrix for each projection retains flexibility in combining the shared directions. All computations and optimization are performed on classical hardware.

For institutional personalization, PFedQFT combines a globally aggregated full core with a private residual operator for each client. Globally learned residual direction matrices map the client retained coefficient vector to a structured core correction, allowing each institution to adjust selected directions within the shared adaptation subspaces. This design introduces only r private parameters per adapted projection, compared with $r ^ { 2 }$ for a full private core. Clients retain these coefficients across communication rounds and exchange only the shared adapter parameters. By coupling basis reuse with compact local corrections, the framework is designed to reduce the trainable parameter budget while supporting both collaborative learning and institution specific adaptation. The main contributions are summarized as follows:

• A quantum-inspired FFT framework is formulated to adapt medical LLMs across clients with heterogeneous data and constrained resources.

• Orthonormal adaptation bases are constructed from compact skew-symmetric generators using a truncated matrixexponential approximation followed by thin QR orthonormalization, and are shared across compatible Transformer projections.

• A shared and private core decomposition combines an aggregated full core with structured residual operators controlled by locally retained coefficients, preserving institutional adaptation without additional communication payload.

• Experiments on medical datasets demonstrate that PFedQFT substantially reduces trainable parameters while maintaining competitive performance.

## II. RELATED WORK

## A. Parameter-Efficient Fine-Tuning

PEFT adapts a pretrained model by freezing most backbone weights and optimizing a small set of task parameters. Bottleneck adapters insert compact trainable modules between frozen Transformer components [6], whereas LoRA represents each weight update with two trainable low-rank factors [7]. Recent studies have refined LoRA from several complementary perspectives. LoRA+ assigns different learning rates to its two factors to improve optimization efficiency [14], while DoRA decomposes pretrained weights into magnitude and direction components and applies low-rank updates to the directional component [15]. RandLoRA constructs full-rank updates from learned combinations of fixed random low-rank bases [16], and LowRA combines LoRA with quantization below two bits per parameter to reduce memory consumption [17]. PLoP further studies where adapters should be placed and automatically selects projection types for a given model and task [18]. Efficient foundation model adaptation has also been examined in both centralized and federated settings [19]. Despite this progress, most designs retain separate parameters for individual layers or projections. For a projection of shape $d _ { o } \times d _ { i }$ , LoRA requires $r ( d _ { o } + d _ { i } )$ parameters, so repeatedly adapting multiple Transformer projections limits structural reuse and preserves dimension dependent training and communication costs.

## B. Federated Learning

FL enables distributed clients to collaboratively optimize models while retaining training data locally. FedAvg alternates local optimization with aggregation weighted by client sample counts [3]. Recent work addresses statistical heterogeneity through adaptive aggregation and personalized collaboration. FedAWA adjusts aggregation weights using client update directions to improve alignment with global optimization [20]. DPFL constructs a collaboration graph through constrained greedy selection, enabling decentralized personalized training within predefined communication and resource budgets [21]. FedMerge jointly learns multiple global models and merging weights for each client to produce personalized models suited to local data distributions [22]. These approaches improve how clients collaborate, but do not directly address the repeated parameterization of adaptation modules across Transformer projections. For medical LLM adaptation, a complementary challenge is to couple compact reusable subspaces with lightweight institutional corrections, rather than relying solely on changes to aggregation or collaboration structure.

## C. Federated Fine-Tuning

Parameter-efficient FFT integrates PEFT into FL, allowing clients to retain raw data and communicate only a small trainable parameter subset. FedPer introduced shared and private layers for personalized FL [9]. FedPETuning evaluated multiple PEFT methods under federated constraints [5]. FedPepTAO combined partial prompt tuning with adaptive optimization to mitigate client drift [23]. FFA-LoRA freezes the randomly initialized nonzero factor and updates only the zero initialized factor [24]. FedEx-LoRA introduces a residual correction to recover the exact aggregated update [25]. LoRA-FAIR jointly refines server aggregation and client initialization [26]. RoLoRA alternates optimization of the two LoRA factors [27]. RAFFT uses Riemannian parameter matching to support adaptive ranks [28]. FedALT combines individual adapters with a rest-of-world adapter for personalized training [29]. Personalized FL has also been investigated for heterogeneous medical imaging systems [30]. Nevertheless, these approaches primarily correct or personalize complete LoRA modules. They do not directly provide reusable orthogonal subspaces across compatible projections together with a shared and private core decomposition for heterogeneous medical language modeling.

## III. METHODOLOGY

Fig. 2 outlines the interaction between server-side coordination and client-side adaptation in PFedQFT. On each client, quantum-inspired subspace construction generates reusable orthonormal bases, while private core residuals introduce institution-specific adaptation within the shared subspace. The server manages client participation and aggregates only shared parameter updates, thereby separating collaborative optimization from locally retained data and personalization parameters. The following subsections formalize these components.

## A. Problem Formulation

Consider a federated system comprising K medical institutions, each represented by a client. Client k maintains a private dataset $\mathcal { D } _ { k }$ containing $n _ { k }$ samples, with $N = \textstyle \sum _ { k = 1 } ^ { K }$ n<sub>k</sub> denoting the total number of samples. Throughout this paper, personalization is defined at the institutional client level: the private parameters $\phi _ { k }$ adapt the model to institution k and its local data distribution, rather than to an individual patient. Given a pretrained language model with frozen parameters W, adaptation is performed through globally shared parameters Θ and private parameters $\phi _ { k }$ specific to client k. The personalized federated learning objective is formulated as:

$$
\operatorname* { m i n } _ { \Theta , \{ \phi _ { k } \} _ { k = 1 } ^ { K } } \sum _ { k = 1 } ^ { K } \frac { n _ { k } } { N } \mathcal { L } _ { k } ( W , \Theta , \phi _ { k } ) ,\tag{1}
$$

where $\mathcal { L } _ { k }$ denotes the empirical task loss evaluated on $\mathcal { D } _ { k }$ Under this protocol, raw medical records and private adaptation parameters remain at their respective clients, while only globally shared adaptation parameters are exchanged with the server. For each selected projection matrix ${ \bf W } _ { \ell } \in { \bf \Psi }$ $\mathbb { R } ^ { d _ { o } \times d _ { i } }$ , where $d _ { i }$ and $d _ { o }$ denote its input and output dimensions, respectively, client k uses the adapted matrix $\mathbf { W } _ { \ell } + \Delta \mathbf { W } _ { k , \ell } ( \theta , \phi _ { k } )$ . The objective is to construct a compact parameterization of $\Delta \mathbf { W } _ { k , \ell }$ that reduces trainable parameter counts and communication overhead relative to full fine-tuning while supporting knowledge sharing across institutions and institution specific adaptation.

![](images/4aee1cd8b11dcc619d460e1883fd2f90833bfe7f5fae873b466a95b0b75ce43f.jpg)  
Fig. 2. Overall architecture of $\mathrm { P F e d Q F T } .$ The framework comprises a global server that coordinates sample-weighted aggregation and multiple clients that construct reusable adaptation subspaces while retaining private core residuals for personalized local training. (a) At communication round t, the server selects the participating clients $\boldsymbol { S } _ { t }$ and broadcasts the shared parameters Θ<sup>t</sup>. (b) Each selected client transforms the generators $\mathbf { B } _ { q } ^ { \mathrm { i n } }$ and $\mathbf { B } _ { q } ^ { \mathrm { o u t } }$ associated with sharing group g into the orthonormal bases $\mathbf { V } _ { g }$ and $\mathbf { U } _ { g }$ through Taylor mapping and thin QR. (c) For each adapted projection $\ell ,$ client k constructs the private residual operator $\mathbf { D } _ { k , \ell } = \mathbf { R } _ { \ell } ^ { \mathrm { o u t } } \mathrm { d i a g } ( \mathbf { p } _ { k , \ell } ^ { - } ) ( \mathbf { R } _ { \ell } ^ { \mathrm { i n } } ) ^ { \top }$ and the personalized subspace operator $\mathbf { C } _ { k , \ell } = \mathbf { G } _ { \ell } + \mathbf { \bar { D } } _ { k , \ell }$ . (d) Local optimization updates the shared parameter copy $\Theta _ { k }$ and the private coefficients ϕ using $\mathcal { D } _ { k } .$ (e) Client k uploads only $( \Delta \Theta _ { k } ^ { t } , n _ { k } )$ , while $\mathcal { D } _ { k }$ and $\mathbf { p } _ { k , \ell }$ remain local. (f) The server performs sample-weighted aggregation to obtain the shared parameters $\Theta ^ { t + 1 }$ for the next communication round.

## B. Quantum-Inspired Subspace Adaptation

To enable parameter sharing across compatible Transformer projections, compact trainable parameters are used to construct orthonormal bases for their adaptation subspaces. The pretrained backbone remains frozen, with lightweight adapters introduced into selected attention and feed forward projections. Let $\mathcal { L }$ denote the set of adapted projections and G the set of sharing groups. Projections with the same module type and matrix shape form a group $g \in \mathcal G$ , and $g ( \ell )$ maps projection $\ell \in \mathcal L$ to its group. Each group has input and output dimensions $d _ { i , g }$ and $d _ { o , g } ,$ , respectively, and uses bases $\mathbf { U } _ { g } \in \mathbb { R } ^ { d _ { o , g } \times r }$ and $\bar { \mathbf { V } _ { g } } \in \mathbb { R } ^ { d _ { i , g } \times r }$ , with $r \leq \operatorname* { m i n } ( d _ { i , g } , d _ { o , g } )$ Both bases have orthonormal columns and are generated from parameters shared within the group.

Inspired by parameterized quantum gates that control state evolution through unitary transformations, the method generates adaptation bases from compact transformation parameters [11]. Real orthogonal mappings implement this design, producing adaptation directions shared across compatible projections. For a feature dimension $d ,$ let κ control the capacity of the structured generator, with $1 \leq \kappa \leq d .$ Define $\mathbf { P } _ { \kappa } =$ $[ \mathbf { e } _ { 1 } , \dots , \mathbf { e } _ { \kappa } ] \in \mathbb { R } ^ { d \times \kappa }$ using the first κ standard basis vectors. A trainable masked matrix $\mathbf { B } \in \mathbb { R } ^ { d \times \kappa }$ satisfies $[ \mathbf { B } ] _ { i j } = 0$ for $i \leq j$ . The resulting generator and its skew-symmetry property are given by:

$$
\mathbf { A } ( \mathbf { B } ) = \mathbf { B } \mathbf { P } _ { \kappa } ^ { \top } - \mathbf { P } _ { \kappa } \mathbf { B } ^ { \top } ,\tag{2}
$$

$$
\mathbf { A ( B ) } ^ { \top } = - \mathbf { A ( B ) } .\tag{3}
$$

Its exponential $\mathbf { Q ( B ) } \ = \ \exp ( \mathbf { A ( B ) } )$ is orthogonal and represents a real valued instance of a unitary transformation.

In particular, $\mathbf { H } ( \mathbf { B } ) \mathbf { \Phi } = \mathrm { i } \mathbf { A } ( \mathbf { B } )$ is Hermitian and ${ \bf Q } ( { \bf B } ) = { \bf \Lambda }$ exp(−iH(B)), linking the construction to the generator formulation of quantum state evolution. Applying Q(B) to ${ \bf E } _ { r } = [ { \bf e } _ { 1 } , \ldots , { \bf e } _ { r } ] \in \mathbb { R } ^ { d \times r }$ selects r orthonormal adaptation directions. To compute these directions efficiently, a degree-J Taylor approximation is applied directly to $\mathbf { E } _ { r }$ , followed by thin QR to obtain the orthonormal adaptation basis:

$$
\widetilde { \mathbf { F } } _ { d } ( \mathbf { B } ) = \sum _ { j = 0 } ^ { J } \frac { \mathbf { A } ( \mathbf { B } ) ^ { j } \mathbf { E } _ { r } } { j ! } ,\tag{4}
$$

$$
\mathbf { F } _ { d } ( \mathbf { B } ) = \mathcal { Q } \Big ( \widetilde { \mathbf { F } } _ { d } ( \mathbf { B } ) \Big ) ,\tag{5}
$$

where $\mathcal { Q } ( \cdot )$ denotes the operator that returns the orthonormal factor of a thin QR decomposition, distinct from the matrix exponential Q(B). The order J controls the truncation depth. Provided that $\bar { \mathbf { F } } _ { d } ( \mathbf { B } )$ has full column rank, thin QR yields an orthonormal basis for its column space. Each multiplication is evaluated as $\mathbf { A } ( \mathbf { B } ) \mathbf { Z } \ = \ \mathbf { B } ( \mathbf { P } _ { \kappa } ^ { \top } \mathbf { Z } ) \mathbf { - \nabla } \mathbf { P } _ { \kappa } ( \mathbf { B } ^ { \top } \hat { \mathbf { Z } } )$ for an intermediate matrix Z, avoiding construction of a dense $d \times d$ generator or exponential. For group $^ { g , }$ separate masked generators $\mathbf { B } _ { q } ^ { \mathrm { i n } } \in \mathbb { R } ^ { d _ { i , g } \times \kappa }$ and $\mathbf { B } _ { g } ^ { \mathrm { o u t } } \in \mathbb { R } ^ { d _ { o , g } \times \kappa }$ construct the input and output bases:

$$
\begin{array} { r } { \mathbf { V } _ { g } = \mathbf { F } _ { d _ { i , g } } ( \mathbf { B } _ { g } ^ { \mathrm { i n } } ) , } \end{array}\tag{6}
$$

$$
\mathbf { U } _ { g } = \mathbf { F } _ { d _ { o , g } } ( \mathbf { B } _ { g } ^ { \mathrm { o u t } } ) .\tag{7}
$$

These bases satisfy $\mathbf { V } _ { q } ^ { \top } \mathbf { V } _ { g } = \mathbf { I } _ { r }$ and $\mathbf { U } _ { q } ^ { \top } \mathbf { U } _ { g } = \mathbf { I } _ { r }$ . They are computed once per model call and reused within each sharing group, avoiding a separately parameterized basis for every layer.

C. Personalized Federated Adaptation with Private Core Residuals

Personalized federated adaptation requires local flexibility within the shared adaptation subspaces. Each adapted projection ℓ therefore uses a globally shared full core $\mathbf { G } _ { \ell } \in \mathbb { R } ^ { r \times r }$ which captures interactions between all input and output adaptation directions. Unlike the orthonormal bases reused across compatible projections, $\mathbf { G } _ { \ell }$ is specific to projection ℓ and is shared across clients through federated aggregation.

For the private component, two globally shared residual direction matrices $\mathbf { R } _ { \ell } ^ { \mathrm { i n } } , \mathbf { \bar { R } } _ { \ell } ^ { \mathrm { o u t } } \in \mathbb { R } ^ { r \times \bar { r } }$ satisfy $( \mathbf { R } _ { \ell } ^ { s } ) ^ { \top } \mathbf { R } _ { \ell } ^ { s } = \mathbf { I } _ { r }$ for $s \in \{ \mathrm { i n } , \mathrm { o u t } \}$ . Each matrix is block diagonal with $r / h$ blocks of size $h \times h$ , where $2 \leq h \leq r$ and h divides r. Within each block, an identity matrix is successively right multiplied by Givens rotations for all coordinate pairs $( a , b )$ with $1 \leq a < b \leq h .$ , in lexicographic order. Each rotation uses the $2 \times 2$ submatrix [cos ρ, − sin ρ; sin $\rho ,$ cos ρ] on its selected coordinates. The resulting angle vector $\dot { \pmb { \rho } _ { \ell } ^ { s } } \in \mathbb { R } ^ { r ( h - 1 ) / 2 }$ is initialized to zero, yielding an identity direction matrix. This construction learns residual directions within each block without mixing coordinates across blocks.

The direction angles are aggregated together with the full core and subspace generator parameters, whereas client k retains only a coefficient vector $\mathbf { p } _ { k , \ell } \in \mathbb { R } ^ { r }$ . The private residual operator, personalized subspace operator, and resulting update to the frozen projection are defined as:

$$
\mathbf { D } _ { k , \ell } = \mathbf { R } _ { \ell } ^ { \mathrm { o u t } } \mathrm { d i a g } ( \mathbf { p } _ { k , \ell } ) ( \mathbf { R } _ { \ell } ^ { \mathrm { i n } } ) ^ { \top } ,\tag{8}
$$

$$
\mathbf { C } _ { k , \ell } = \mathbf { G } _ { \ell } + \mathbf { D } _ { k , \ell } ,\tag{9}
$$

$$
\Delta \mathbf { W } _ { k , \ell } = \frac { \alpha } { r } \mathbf { U } _ { g ( \ell ) } \mathbf { C } _ { k , \ell } \mathbf { V } _ { g ( \ell ) } ^ { \top } ,\tag{10}
$$

where $g ( \ell )$ identifies the sharing group of projection ℓ and α controls the update scale.

The full core $\mathbf { G } _ { \ell }$ provides a complete shared transformation within the adaptation subspaces, whereas $\mathbf { D } _ { k , \ell }$ supplies a structured institution specific correction. For fixed residual direction matrices, the client retained coefficients control $r$ learned rank one directions rather than only the coordinate axes of the core. The private component consequently uses only r client parameters per projection, compared with $\overline { { \boldsymbol { r } ^ { 2 } } }$ for a full private core. These coefficients remain at their respective clients and do not increase the shared adapter payload. Global cores are initialized with small nonzero values to permit task gradients to reach the basis generators at the start of training. Private coefficients are initialized to zero, so the initial private residual vanishes. Under this parameterization, the task gradient with respect to the residual direction angles is zero whenever $\mathbf { p } _ { k , \ell } ~ = ~ \mathbf { 0 }$ . The private coefficients can nevertheless receive nonzero task gradients; once they become nonzero, the direction angles can also receive task gradients. A nonzero global core does not directly provide task gradients to these angles, since they appear only in the private branch.

## D. Federated Optimization

The globally shared and private parameters are organized as tuples:

$$
\begin{array} { r l } & { \boldsymbol { \Theta } = \biggl ( \bigl ( \mathbf { B } _ { g } ^ { \mathrm { i n } } , \mathbf { B } _ { g } ^ { \mathrm { o u t } } \bigr ) _ { \boldsymbol { g } \in \mathcal { G } } , } \\ & { \qquad \bigl ( \mathbf { G } _ { \ell } , \boldsymbol { \rho } _ { \ell } ^ { \mathrm { i n } } , \boldsymbol { \rho } _ { \ell } ^ { \mathrm { o u t } } \bigr ) _ { \ell \in \mathcal { L } } \biggr ) , } \\ & { \boldsymbol { \phi } _ { k } = \bigl ( \mathbf { p } _ { k , \ell } \bigr ) _ { \ell \in \mathcal { L } } . } \end{array}\tag{11}
$$

(12)

Differences and weighted averages of Θ act componentwise on the corresponding tensors. At round t, the server selects a client subset $S _ { t }$ and broadcasts $\Theta ^ { t }$ . Each participating client initializes its shared parameters from $\Theta ^ { t }$ and restores its previously retained private coefficients $\phi _ { k } ^ { t }$ . In practice, training uses a regularized version of the objective in Eq. (1), replacing each $\mathcal { L } _ { k }$ with $\mathcal { T } _ { k }$ . The client performs E local epochs of minibatch optimization using the local objective:

$$
\mathcal { I } _ { k } ( \Theta , \phi _ { k } ) = \mathcal { L } _ { k } ( W , \Theta , \phi _ { k } ) + \lambda _ { p } \sum _ { \ell \in \mathcal { L } } \| \mathbf { p } _ { k , \ell } \| _ { 2 } ^ { 2 } ,\tag{13}
$$

where $\lambda _ { p } \geq 0$ controls the regularization of private residual coefficients. Gradients update both shared and private parameters through the basis construction and core transformations, yielding $( \check { \Theta } _ { k } ^ { t + 1 } , \phi _ { k } ^ { t + 1 } )$ after the prescribed local epochs.

Client k retains $\phi _ { k } ^ { t + 1 }$ locally and uploads its shared parameter increment with the sample count $n _ { k }$ . The increment and the server update are defined as:

$$
\Delta \Theta _ { k } ^ { t } = \Theta _ { k } ^ { t + 1 } - \Theta ^ { t } ,\tag{14}
$$

$$
\Theta ^ { t + 1 } = \Theta ^ { t } + \sum _ { k \in S _ { t } } \frac { n _ { k } } { \sum _ { j \in S _ { t } } n _ { j } } \Delta \Theta _ { k } ^ { t } .\tag{15}
$$

Aggregation operates on the trainable generator parameters and residual direction angles, together with the globally shared full cores. The orthonormal bases and residual direction matrices are reconstructed from the updated parameters using their respective mappings. Private coefficients do not enter the server update, and clients outside $S _ { t }$ retain their previous state:

$$
\phi _ { k } ^ { t + 1 } = \phi _ { k } ^ { t } , \qquad k \notin { \cal S } _ { t } .\tag{16}
$$

Algorithm 1 summarizes the basis construction, local training, and federated aggregation procedure.

Algorithm 1: Federated Optimization   
Input: Frozen LLM $W ;$ datasets $\{ \mathcal { D } _ { k } \} _ { k = 1 } ^ { K } ;$ sharing groups   
$\mathcal { G } ;$ ranks $r , \kappa ;$ Taylor order $J ;$ rounds $T ;$ local epochs   
$E ;$ update scale $\alpha ;$ regularization coefficient $\Delta _ { p }$   
Output: Global adapter $\Theta ^ { T }$ and locally retained $\{ \phi _ { k } ^ { T } \} _ { k = 1 } ^ { K }$   
1 Initialize shared generators, residual direction angles, and   
full cores as $\Theta ^ { \breve { 0 } } ;$   
2 Initialize private coefficients $\phi _ { k } ^ { 0 } = \{ \mathbf { p } _ { k , \ell } ^ { 0 } = \mathbf { 0 } \} _ { \ell } ;$   
3 for $t = 0$ to $T - 1$ do   
4 Select $S _ { t }$ and broadcast $\Theta ^ { t } ;$   
5 Keep $\phi _ { k } ^ { t + 1 } \gets \phi _ { k } ^ { t }$ for $k \notin S _ { t } ;$   
6 foreach $k \in S _ { t }$ in parallel do   
7 Load $\Theta _ { k } \gets \Theta ^ { t }$ and restore $\phi _ { k }  \phi _ { k } ^ { t } ;$   
8 foreach minibatch $B \subset D _ { k }$ over $E$ local epochs do   
/<sub>\*</sub> Shared Subspace Construction /   
9 foreach input and output generator $\mathbf { B }$ in G do   
10 Apply the lower triangular mask to B;   
11 $\mathbf { Z } \gets \mathbf { E } _ { r } ; \widetilde { \mathbf { F } } \gets \mathbf { E } _ { r } ;$   
12 for $j = 1 \ { \bf t o } \ J$ do   
13 $\begin{array} { r } { \dot { \mathbf Z }  [ \mathbf B ( \mathbf P _ { \kappa } ^ { \top } \mathbf Z ) - \mathbf P _ { \kappa } ( \mathbf B ^ { \top } \mathbf Z ) ] / j ; } \end{array}$   
14 $\widetilde { \mathbf { F } } \gets \widetilde { \mathbf { F } } + \mathbf { Z } ;$   
15 Store $\mathcal { Q } ( \widetilde { \mathbf { F } } )$ as $\mathbf { V } _ { g }$ or $\mathbf { U } _ { g } ;$   
/ Core Adaptation \*/   
16 foreach adapted projection ℓ do   
17 Construct ${ \bf R } _ { \ell } ^ { \mathrm { i n } } , { \bf R } _ { \ell } ^ { \mathrm { o u t } }$ from residual direction   
angles;   
18 $\mathbf D _ { k , \ell } \gets \mathbf R _ { \ell } ^ { \mathrm { o u t } } \mathrm { d i a g } ( \mathbf p _ { k , \ell } ) ( \mathbf R _ { \ell } ^ { \mathrm { i n } } ) ^ { \top } ;$   
19 $\mathbf { C } _ { k , \ell } \gets \mathbf { G } _ { \ell } + \mathbf { D } _ { k , \ell } ;$   
20 $\mathbf { A p p l y } \Delta \mathbf { W } _ { k , \ell }$ using Eq. (10);   
21 Evaluate minibatch loss $\mathcal { T } _ { k }$ using Eq. (13);   
22 Update $\Theta _ { k } , \phi _ { k }$ by backpropagation through $\mathcal { T } _ { k } ;$   
23 Store $\phi _ { k } ^ { t + 1 } \gets \phi _ { k }$ locally;   
24 Upload $\mathbf { \bar { \Delta } } \mathbf { \Theta } _ { k } ^ { t } \xleftarrow { } \mathbf { \Theta } \mathbf { \Theta } \Theta _ { k } - \mathbf { \bar { \Theta } } \mathbf { \bar { \Theta } } ^ { t }$ and $n _ { k } ;$   
/<sub>\*</sub> Global Aggregation \*/   
25 $\Theta ^ { t + 1 } \gets \Theta ^ { t } + \sum _ { k \in \mathcal { S } _ { t } } ^ { - } \frac { n _ { k } } { \sum _ { j \in \mathcal { S } _ { t } } n _ { j } } \Delta \Theta _ { k } ^ { t } ;$   
26 return $\Theta ^ { T } \colon$ each client retains its own $\phi _ { k } ^ { T } \mathbf { ; }$

## IV. EXPERIMENTS

## A. Datasets and Models

Experiments are conducted on seven medical datasets covering diagnosis normalization, information extraction, and medical question answering. The Chinese biomedical tasks are drawn from CBLUE [31] and its subsequent releases: CHIP-CDN for diagnosis normalization, CMeEE-V2 for named entity recognition, CMeIE for entity relation extraction, and CMedCausal [32] for causal relation extraction. Medical question answering is evaluated on the MedQA-MCMLE, MedQA-TWCMLE, and MedQA-USMLE subsets [33], which contain medical licensing examination questions in simplified Chinese, traditional Chinese, and English, respectively. Each MedQA subset is evaluated as a separate task.

We evaluate PFedQFT using five backbone LLMs: Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct [34], Qwen3-1.7B, Qwen3-4B [35], and MedGemma-1.5-4B-it [36]. These models are selected to assess the applicability of PFedQFT across model families and parameter scales, encompassing both general-purpose and medically adapted backbones.

## B. Baselines and Metrics

We compare PFedQFT with four FFT baselines based on LoRA, covering exact aggregation through factor stacking, adaptive server optimization, factor freezing, and residual correction. FLoRA performs noise-free aggregation by vertically stacking the sample-weighted local $\mathbf { A } _ { k }$ factors and horizontally stacking the corresponding $\mathbf { B } _ { k }$ factors, so that the product of the global factors exactly equals the sampleweighted sum of the local LoRA updates; the stacking construction also supports heterogeneous local ranks [37]. FedAdam-LoRA applies the FedAdam server optimizer to the aggregated LoRA parameter updates [38]. FFA-LoRA fixes the randomly initialized factor A and trains only B, reducing communication and eliminating the aggregation mismatch induced by independently averaging two trainable factors [24]. FedEx-LoRA introduces a residual correction to recover the weighted average of client weight updates from the aggregated low-rank factors [25].

Predictive performance is assessed using task-specific metrics. For medical diagnosis normalization and information extraction datasets (CHIP-CDN, CMeEE-V2, CMeIE, and CMedCausal), we adopt Soft F1, a similarity-weighted micro F1 score that awards partial credit for lexical overlap. Predictions and references are parsed into sets of diagnosis terms or structured extraction units. For each example $i ,$ the Hungarian algorithm [39] identifies a maximum-weight one-to-one matching $\mathcal { M } _ { i }$ between the predicted set $\widehat { \mathcal { V } } _ { i }$ and reference set $\mathcal { \mathrm { V } } _ { i }$ . Given unit similarity $s ( u , v ) \in [ 0 , 1 ]$ , the score is computed over the evaluation set as:

$$
\mathrm { S o f t F 1 } = \frac { 2 \sum _ { i } \sum _ { ( u , v ) \in { \mathcal { M } } _ { i } } s ( u , v ) } { \sum _ { i } | \widehat { \mathcal { V } } _ { i } | + \sum _ { i } | \mathcal { V } _ { i } | } .\tag{17}
$$

For the three MedQA subsets, we report accuracy based on exact matching of predicted and reference answer option identifiers.

Parameter efficiency is measured by the number of trainable scalar parameters per client:

$$
\nu _ { k } = | \Theta | + | \phi _ { k } | ,\tag{18}
$$

where |·| counts scalar parameters, with shared tensors counted only once. The upload volume for each client in round t, measured in bytes, is:

$$
\Gamma _ { k , t } ^ { \uparrow } = \frac { b } { 8 } m _ { k , t } ^ { \uparrow } ,\tag{19}
$$

where b is the number of bits per transmitted element and $m _ { k , t } ^ { \uparrow }$ counts uploaded elements. Private residuals, protocol overhead, and initial backbone distribution are excluded.

## C. Configurations

All experiments freeze the pretrained backbone and optimize only the adaptation parameters, with Llama-3.2-3B-Instruct used by default unless otherwise specified. The LoRAbased baselines use a local rank of 16 at each client. FLoRA is evaluated with identical local ranks: every participating client trains a rank-16 adapter, and the server stacks the local factors so that the rank of the resulting global adapter equals the sum of the participating clients’ local ranks. In contrast, PFedQFT aggregates shared adapter parameters across participating clients and retains institution specific residual coefficients locally. Each dataset is converted into a unified instruction–response format: diagnosis normalization produces standardized terms, information extraction tasks use structured textual outputs, and MedQA predicts an answer option identifier. Federated clients are simulated by partitioning the training records of each dataset. All experiments are conducted on a server equipped with two NVIDIA H100 GPUs, each with 80 GB of memory; the quantum-inspired operations in PFedQFT are implemented entirely on classical hardware.

## D. Experimental Results

TABLE I  
PERFORMANCE UNDER VARYING DEGREES OF DATA HETEROGENEITY.
<table><tr><td rowspan="2">Non-IID Degree</td><td colspan="2">MedQA-MCMLE</td><td colspan="2">CHIP-CDN</td><td colspan="2">CMeEE-V2</td></tr><tr><td>FLoRA</td><td>PFedQFT</td><td>FLoRA</td><td>PFedQFT</td><td>FLoRA</td><td>PFedQFT</td></tr><tr><td>Light</td><td>81.83</td><td>81.49</td><td>78.32</td><td>76.52</td><td>77.79</td><td>76.74</td></tr><tr><td>Moderate</td><td>81.72</td><td>81.46</td><td>79.95</td><td>76.57</td><td>77.12</td><td>76.35</td></tr><tr><td>Severe</td><td>81.15</td><td>81.14</td><td>77.96</td><td>76.08</td><td>77.19</td><td>76.56</td></tr></table>

Note: The Dirichlet concentration parameter $\beta$ controls the non-IID degree, with $\beta =$ 1.0, 0.5, and 0.1 denoting light, moderate, and severe heterogeneity, respectively. MedQA-MCMLE uses Accuracy, while CHIP-CDN and CMeEE-V2 use Soft F1.

1) Overall Performance Comparison: To assess overall effectiveness across medical language understanding tasks, we compare PFedQFT with four FFT baselines based on LoRA using Llama-3.2-3B-Instruct; accuracy is reported on the three complete MedQA validation splits, whereas Soft F1 is used for the four CBLUE tasks. Figure 3 shows that PFedQFT achieves the highest accuracy on MedQA-MCMLE (81.75%), exceeding FLoRA by 0.29 percentage points, and obtains

76.91% and 65.25% on MedQA-TWCMLE and MedQA-USMLE, respectively, remaining within 1.49 points of the strongest baseline. On CHIP-CDN and CMeEE-V2, PFedQFT reaches 76.71% and 77.46% Soft F1, only 0.33 and 0.26 points below FLoRA. On CMeIE and CMedCausal, PFedQFT obtains 59.48% and 63.09% Soft F1, respectively, trailing FLoRA by 5.58 and 6.10 points. Relative to the strongest baseline on each task, the corresponding gaps are 6.35 points to FedEx-LoRA and 7.73 points to FedAdam-LoRA. These losses may reflect capacity constraints specific to each task imposed by aggressive parameter compression and basis sharing across layers when modeling complex entity relations and causal links. Overall, PFedQFT ranks first or remains within 1.5 points of the best baseline on five of the seven datasets, while the CMeIE and CMedCausal results reveal a trade-off between parameter efficiency and predictive performance that depends on the task.

2) Performance across Backbone Models: To evaluate portability across model families and parameter scales, we compare PFedQFT with FLoRA on the three complete MedQA validation splits using five general-purpose and medically adapted backbones ranging from 1.7B to 8B parameters. Across the 15 combinations in Fig. 4, PFedQFT outperforms FLoRA in seven cases, with the most consistent gains obtained by Llama-3.1-8B-Instruct: 1.28, 0.28, and 1.18 percentage points on MedQA-MCMLE, MedQA-TWCMLE, and MedQA-USMLE, respectively; this backbone also produces the highest MedQA-TWCMLE and MedQA-USMLE accuracies among the evaluated models. On MedQA-TWCMLE, PFedQFT further improves Qwen3-1.7B and Qwen3-4B by 0.64 and 0.71 points. The two methods differ by at most 0.24 points for Llama-3.2-3B on MedQA-USMLE and for MedGemma on MedQA-TWCMLE and MedQA-USMLE, whereas PFedQFT trails FLoRA on MedQA-MCMLE by 2.89, 2.04, and 7.53 points with Qwen3-1.7B, Qwen3-4B, and MedGemma. These results confirm broad applicability but also indicate that the attainable gains depend on the backbone, with Llama-3.1-8B-Instruct providing the most stable improvements.

3) Sensitivity to Data Heterogeneity: To assess sensitivity to heterogeneous client data, we evaluate FLoRA and PFedQFT under three Dirichlet concentrations $\beta \in$ {1.0, 0.5, 0.1} using five clients, with matched client assignments and identical global validation examples. Table I reports the performance of the global adapter after the final round with private residuals cleared. Across the three concentrations, the performance ranges of PFedQFT are only 0.35, 0.49, and 0.39 percentage points on MedQA-MCMLE, CHIP-CDN, and CMeEE-V2, respectively, compared with 0.68, 1.99, and 0.67 for FLoRA. The reduction is most pronounced on CHIP-CDN, where the range of PFedQFT is approximately four times smaller. Under severe heterogeneity, PFedQFT also achieves 81.14% accuracy on MedQA-MCMLE, closely matching the 81.15% of FLoRA. These consistently narrower variations demonstrate that PFedQFT maintains more stable performance across the evaluated heterogeneity levels and is less sensitive

![](images/521a40619d553ca95e555a66d95fc0793245532ff6201ca66381531d7fd0485e.jpg)  
(a) MedQA

![](images/ac47542e3e00d8b4477340d484b2f395ef189a3688c1a93d34ad8b84f99959fc.jpg)  
(b) CBLUE

Fig. 3. Overall performance comparison. The Llama-3.2-3B-Instruct backbone is frozen, and only adapter parameters are optimized with five participating clients per communication round. The LoRA baselines use local rank-16 adapters, whereas PFedQFT uses task-specific adaptation configurations with globally shared parameters and residual coefficients retained privately by each client. Results are evaluated on the complete validation sets using Accuracy for medical question answering and Soft F1 for diagnosis normalization and information extraction.  
![](images/77d8a1655c04449e4dc7f4ae7514700a38013a8c3c63a9ad0ab932b1b3471de0.jpg)  
(a) MedQA-MCMLE

![](images/9745ced4d6e5a968a1b363db9b5a95154312fe585385cd6c6963769b56bcba1e.jpg)  
(b) MedQA-TWCMLE

![](images/1775995ee747b11311ecb044c3367545d93d35764d88cd950d00e273242f818c.jpg)  
(c) MedQA-USMLE  
Fig. 4. Performance across backbone models. All backbones are frozen, and only adapter parameters are optimized with one local epoch per communication round, an effective batch size of 16, and a maximum sequence length of 768. FLoRA uses rank-16 adapters, whereas PFedQFT uses an adaptation rank of 8 and an intrinsic rank of 4 with residual coefficients retained privately by each client. Accuracy is evaluated on the complete validation splits, and ∆ denotes the accuracy of PFedQFT minus that of FLoRA in percentage points.

TABLE II  
AVERAGE TRAINABLE PARAMETER COUNT, ADAPTER STORAGE, COMMUNICATION COST, AND RUNTIME ACROSS THE SEVEN TASKS.
<table><tr><td>Method</td><td>#TP (M)</td><td>TP (%)</td><td>Shared (MB)</td><td>Private (MB)</td><td>Upload (MB)</td><td>Time</td></tr><tr><td>FLoRA†</td><td>24.3139</td><td>0.7568</td><td>97.2554</td><td></td><td>97.2554</td><td>0 h 45 min</td></tr><tr><td>FedAdam-LoRA</td><td>24.3139</td><td>0.7568</td><td>97.2554</td><td></td><td>97.2554</td><td>0 h 37 min</td></tr><tr><td>FFA-LoRA</td><td>12.3863</td><td>0.3855</td><td>49.5452</td><td></td><td>49.5452</td><td>1 h 06 min</td></tr><tr><td>FedEx-LoRA‡</td><td>24.3139</td><td>0.7568</td><td>97.2554</td><td></td><td>97.2554</td><td>0 h 52 min</td></tr><tr><td>PFedQFT</td><td>1.6524</td><td>0.0514</td><td>6.5981</td><td>0.011648</td><td>6.5981</td><td>6 h 23 min</td></tr></table>

Note: Values are arithmetic means across the seven task-specific experimental runs underlying the overall comparison in Fig. 3. Time denotes end-to-end wall-clock runtime, including training and evaluation. M denotes ${ 1 0 } ^ { 6 }$ parameters, and MB denotes 10<sup>6</sup> bytes. All adapter tensors are stored in FP32. #TP denotes the trainable parameter count, and TP (%) is normalized by the backbone parameter count. The Shared, Private, and Upload columns denote the shared adapter size, private storage for each client, and upload volume for each client per round, respectively. † FLoRA uses aggregation based on factor stacking and is evaluated with a local rank of 16 at every participating client. ‡ FedEx-LoRA excludes residual-correction traffic.

## to changes in client data distributions.

4) Parameter and Communication Efficiency: Resource efficiency is evaluated by averaging end-to-end runtime, trainable parameters, adapter storage, and upload volume for each client across the seven tasks. As reported in Table II, PFedQFT uses an average of 1.6524 million trainable parameters, or 0.0514% of the backbone, and transmits 6.5981 MB per client per round; these values represent reductions of 93.20% and 93.22%, respectively, relative to FLoRA. Its residuals retained privately by each client occupy only 0.011648 MB on average and require no transmission. The average end-to-end runtime is 6 h 23 min under the evaluated task-specific configurations, reflecting the additional structured basis construction and personalized residual optimization. Overall, PFedQFT provides a compact parameter and communication footprint across the seven medical tasks.

5) Ablation of the Personalization Mechanism: The contribution of client private adaptation is evaluated on CHIP-

![](images/8415a1d7cef0134da6370764eee5be0618c62fcd0a3e7c2cd1588cc8aefac27e.jpg)  
(a) Moderate Adaptation

![](images/2b8cada47102af510dd47ca9e6ca7efd7e14e9d66a54f5f9e2f02a36a66494e2.jpg)  
(b) Strong Adaptation  
Fig. 5. Personalization mechanism ablation on the 196-example CHIP-CDN target-client validation set. Global disables private adaptation; PFedQFT-D trains the private branch on local examples whose diagnosis concepts do not overlap with the validation set; and PFedQFT uses local support covering the target concepts. Moderate and Strong denote private-branch update settings $( \eta _ { p } , s _ { p } , \lambda _ { p } ) = ( 0 . 0 0 1 5 , 0 . 1 2 , 5 \times 1 0 ^ { - 5 } )$ and $( 0 . 0 0 2 0 , 0 . 1 6 , ^ { 2 } \times 1 0 ^ { - 5 } )$ , respectively. Shared parameters remain frozen in all personalized runs. Each distribution is obtained from 3,000 bootstrap resamples of the validation examples rather than independent training runs; dashed lines indicate Soft F1 on the complete validation set.

CDN using a target client validation set of 196 examples. All conditions start from the same global checkpoint and retain the same model structure; during eight additional rounds, the shared parameters remain frozen and only the private residual coefficients are optimized. Global disables the private residual operator and evaluates the federated global adapter directly. PFedQFT-D trains the private branch using local examples whose diagnosis concepts do not overlap with those in the target validation set, whereas PFedQFT uses a local support set that covers the target concepts. Moderate and Strong refer to the update strength of the private branch rather than the degree of data heterogeneity: they use $( \eta _ { p } , s _ { p } , \lambda _ { p } ) \ =$ $( 0 . 0 0 1 5 , 0 . 1 2 , 5 \times 1 0 ^ { - 5 } )$ and $( 0 . 0 0 2 0 , 0 . 1 6 , 2 \times 1 0 ^ { - 5 } )$ ), respectively, while keeping the adaptation rank, intrinsic rank, Taylor order, and effective batch size fixed. The plotted distributions are obtained from 3,000 bootstrap resamples of the same validation predictions and therefore characterize evaluation variability rather than variation across independent training runs.

As shown in Fig. 5, Global achieves 66.20% Soft F1 under both configurations. With Moderate adaptation, PFedQFT-D and PFedQFT obtain 67.29% and 69.81%, respectively; with Strong adaptation, their scores are 67.18% and 70.92%. Thus, PFedQFT improves over Global by 3.61 and 4.71 percentage points and over PFedQFT-D by 2.53 and 3.74 points under the Moderate and Strong settings, respectively. PFedQFT-D yields an improvement of only approximately one percentage point and does not benefit from the stronger private update, whereas PFedQFT gains a further 1.11 points. The rightward shift of the PFedQFT bootstrap distributions demonstrates that the personalization gain is primarily associated with support data relevant to the target client, and that increasing private adaptation strength is effective when the local concepts align with the target distribution.

## V. CONCLUSION

This work presents PFedQFT, a framework inspired by quantum transformations for efficient personalized FFT of medical language models. By combining shared structured adaptation subspaces, full core transformations for individual projections, and structured residual operators controlled by locally retained coefficients, the framework supports collaborative knowledge transfer and adaptation to the local data distribution of each participating institution. Experiments on medical datasets demonstrate substantial reductions in trainable parameters and communication costs while maintaining competitive predictive performance. Several limitations warrant further investigation. Broader validation across additional medical domains, larger backbone models, and deployments involving multiple institutions is needed to establish the robustness and generalizability of these findings. Retaining institution specific parameters locally does not itself provide formal privacy guarantees, and the clinical reliability of model outputs remains unverified. Future work will explore aggregation with formal privacy guarantees, further reductions in communication costs, and more effective client personalization mechanisms. Evaluation in clinical settings will also be necessary to assess reliability and suitability for medical applications before deployment.

## REFERENCES

[1] K. Singhal, S. Azizi, T. Tu et al., “Large language models encode clinical knowledge,” Nature, vol. 620, pp. 172–180, 2023.

[2] N. Rieke, J. Hancox, W. Li, F. Milletar\`ı, H. R. Roth, S. Albarqouni, S. Bakas, M. N. Galtier, B. A. Landman, K. Maier-Hein, S. Ourselin, M. Sheller, R. M. Summers, A. Trask, D. Xu, M. Baust, and M. J. Cardoso, “The future of digital health with federated learning,” npj Digital Medicine, vol. 3, no. 1, p. 119, 2020.

[3] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Aguera¨ y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in Proceedings of the 20th International Conference on Artificial Intelligence and Statistics. PMLR, 2017, pp. 1273–1282.

[4] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, “Federated optimization in heterogeneous networks,” in Proceedings of Machine Learning and Systems, vol. 2, 2020.

[5] Z. Zhang, Y. Yang, Y. Dai, Q. Wang, Y. Yu, L. Qu, and Z. Xu, “FedPETuning: When federated learning meets the parameter-efficient tuning methods of pre-trained language models,” in Findings of the Association for Computational Linguistics: ACL 2023. Association for Computational Linguistics, 2023, pp. 9963–9977.

[6] N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. De Laroussilhe, A. Gesmundo, M. Attariyan, and S. Gelly, “Parameter-efficient transfer learning for NLP,” in Proceedings of the 36th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 97. PMLR, 2019, pp. 2790–2799.

[7] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in The Tenth International Conference on Learning Representations, 2022.

[8] S. P. Karimireddy, S. Kale, M. Mohri, S. Reddi, S. Stich, and A. T. Suresh, “SCAFFOLD: Stochastic controlled averaging for federated learning,” in Proceedings of the 37th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 119. PMLR, 2020, pp. 5132–5143.

[9] M. G. Arivazhagan, V. Aggarwal, A. K. Singh, and S. Choudhary, “Federated learning with personalization layers,” 2019.

[10] L. Collins, H. Hassani, A. Mokhtari, and S. Shakkottai, “Exploiting shared representations for personalized federated learning,” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 139. PMLR, 2021, pp. 2089–2099.

[11] T. Koike-Akino, F. Tonin, Y. Wu, F. Z. Wu, L. N. Candogan, and V. Cevher, “Quantum-PEFT: Ultra parameter-efficient fine-tuning,” in The Thirteenth International Conference on Learning Representations, 2025.

[12] D. Wierichs, J. Izaac, C. Wang, and C. Y.-Y. Lin, “General parametershift rules for quantum gradients,” Quantum, vol. 6, p. 677, 2022.

[13] M. Lezcano-Casado and D. Mart´ınez-Rubio, “Cheap orthogonal constraints in neural networks: A simple parametrization of the orthogonal and unitary group,” in Proceedings of the 36th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 97. PMLR, 2019, pp. 3794–3803.

[14] S. Hayou, N. Ghosh, and B. Yu, “LoRA+: Efficient low rank adaptation of large models,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 235. PMLR, 2024, pp. 17 783–17 806.

[15] S.-Y. Liu, C.-Y. Wang, H. Yin, P. Molchanov, Y.-C. F. Wang, K.- T. Cheng, and M.-H. Chen, “DoRA: Weight-decomposed low-rank adaptation,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 235. PMLR, 2024, pp. 32 100–32 121.

[16] P. Albert, F. Zhang, H. Saratchandran, C. Rodriguez-Opazo, A. Van Den Hengel, and E. Abbasnejad, “RandLoRA: Full rank parameterefficient fine-tuning of large models,” in The Thirteenth International Conference on Learning Representations, 2025.

[17] Z. Zhou, Q. Zhang, H. Kumbong, and K. Olukotun, “LowRA: Accurate and efficient LoRA fine-tuning of LLMs under 2 bits,” in Proceedings of the 42nd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 267. PMLR, 2025, pp. 79 570–79 594.

[18] S. Hayou, N. Ghosh, and B. Yu, “PLoP: Precise LoRA placement for efficient finetuning of large models,” in The Fourteenth International Conference on Learning Representations, 2026.

[19] N. Saadati, Z. Jiang, A. Balu, C. Liu, C. Hegde, and S. Sarkar, “Foundation model efficient fine-tuning in centralized and federated settings,” in 2025 IEEE International Conference on Big Data (BigData), 2025, pp. 1857–1866.

[20] C. Shi, H. Zhao, B. Zhang, M. Zhou, D. Guo, and Y. Chang, “FedAWA: Adaptive optimization of aggregation weights in federated learning using client vectors,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 30 651–30 660.

[21] S. Kharrat, M. Canini, and S. Horvath, “DPFL: Decentralized per-´ sonalized federated learning,” in Proceedings of the 28th International Conference on Artificial Intelligence and Statistics, ser. Proceedings of Machine Learning Research, vol. 258. PMLR, 2025, pp. 5086–5094.

[22] S. Chen, T. Zhou, G. Long, J. Jiang, and C. Zhang, “FedMerge: Federated model merging for personalization,” Proceedings of the AAAI

Conference on Artificial Intelligence, vol. 40, no. 24, pp. 20 253–20 261, 2026.

[23] T. Che, J. Liu, Y. Zhou, J. Ren, J. Zhou, V. Sheng, H. Dai, and D. Dou, “Federated learning of large language models with parameter-efficient prompt tuning and adaptive optimization,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2023, pp. 7871–7888.

[24] Y. Sun, Z. Li, Y. Li, and B. Ding, “Improving LoRA in privacypreserving federated learning,” in The Twelfth International Conference on Learning Representations, 2024.

[25] R. Singhal, K. Ponkshe, and P. Vepakomma, “FedEx-LoRA: Exact aggregation for federated and efficient fine-tuning of large language models,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 2025, pp. 1316–1336.

[26] J. Bian, L. Wang, L. Zhang, and J. Xu, “LoRA-FAIR: Federated LoRA fine-tuning with aggregation and initialization refinement,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 3737–3746.

[27] S. Chen, Y. Guo, Y. Ju, H. Dalal, Z. Zhu, and A. Khisti, “Robust federated finetuning of LLMs via alternating optimization of LoRA,” in Advances in Neural Information Processing Systems, vol. 38, 2025.

[28] Z. Zhou, Y. Zhou, T. Che, Z. Zhang, J. Ren, D. Yan, Z. Jiang, Y. Shen, R. Jin, and J. Gao, “Noise-free dynamic rank-adaptation via riemannian methods in federated fine-tuning,” in Proceedings ofthe 29th International Conference on Artificial Intelligence and Statistics, ser. Proceedings of Machine Learning Research, vol. 300. PMLR, 2026, pp. 478–486.

[29] J. Bian, L. Wang, L. Zhang, and J. Xu, “FedALT: Federated finetuning through adaptive local training with rest-of-world LoRA,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 24, 2026, pp. 19 728–19 736.

[30] N. S. Shadin, X. Zhang, J. Wang, and M. Pan, “Heterogeneity-aware private personalized federated learning for medical imaging via contrastive distillation,” in 2025 IEEE International Conference on Big Data (BigData), 2025, pp. 2033–2042.

[31] N. Zhang, M. Chen, Z. Bi, X. Liang, L. Li, X. Shang, K. Yin, C. Tan, J. Xu, F. Huang, L. Si, Y. Ni, G. Xie, Z. Sui, B. Chang, H. Zong, Z. Yuan, L. Li, J. Yan, H. Zan, K. Zhang, B. Tang, and Q. Chen, “CBLUE: A chinese biomedical language understanding evaluation benchmark,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 2022, pp. 7888–7915.

[32] Z. Li, M. Chen, Z. Ma, K. Yin, Y. Tong, C. Tan, Z. Lang, B. Tang, and J. Xu, “CMedCausal: Chinese medical causal relationship extraction dataset,” Journal of Medical Informatics, vol. 43, no. 12, pp. 23–27, 31, 2022, in Chinese.

[33] D. Jin, E. Pan, N. Oufattole, W.-H. Weng, H. Fang, and P. Szolovits, “What disease does this patient have? a large-scale open domain question answering dataset from medical exams,” Applied Sciences, vol. 11, no. 14, p. 6421, 2021.

[34] A. Grattafiori et al., “The Llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[35] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.

[36] A. Sellergren, C. Gao, F. Mahvar, T. Kohlberger, F. Jamil, M. Traverse et al., “MedGemma 1.5 technical report,” arXiv preprint arXiv:2604.05081, 2026.

[37] Z. Wang, Z. Shen, Y. He, G. Sun, H. Wang, L. Lyu, and A. Li, “FLoRA: Federated fine-tuning large language models with heterogeneous lowrank adaptations,” in Advances in Neural Information Processing Systems, vol. 37, 2024.

[38] S. J. Reddi, Z. Charles, M. Zaheer, Z. Garrett, K. Rush, J. Konecnˇ y,´ S. Kumar, and H. B. McMahan, “Adaptive federated optimization,” in The Ninth International Conference on Learning Representations, 2021.

[39] H. W. Kuhn, “The Hungarian method for the assignment problem,” Naval Research Logistics Quarterly, vol. 2, no. 1–2, pp. 83–97, 1955.