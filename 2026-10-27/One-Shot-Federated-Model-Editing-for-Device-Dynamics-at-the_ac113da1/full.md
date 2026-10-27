# One-Shot Federated Model Editing for Device Dynamics at the Edge

Youming Tao, Weijun Yin, Huanyi Xie, Yanqing Yao, Dongxiao Yu, Senior Member, IEEE, Xiuzhen Cheng, Fellow, IEEE, Di Wang, Member, IEEE, Falko Dressler Fellow, IEEE

Abstract—Standard federated learning (FL) requires persistent device connectivity across multiple communication rounds, making it unsuitable for dynamic environments such as vehicular networks, where devices frequently join and leave. One-shot FL offers a promising alternative by completing training in a single communication round. However, one-shot FL suffers from model solidification and cannot readily adapt to subsequent client dynamics, including new client contributions and the withdrawal of previous contributions. We therefore consider exact model editing, which removes withdrawn contributions through unlearning or incorporates newly contributed ones through incremental learning without retraining from scratch. We identify that efficient editing fundamentally hinges on learning stability. To this end, we propose F-SOSA, the first stable one-shot FL algorithm that enables rapid and exact model editing. F-SOSA leverages sub-sampling of local models with carefully tuned rate for stable aggregation and performs curvature-aware model fusion to preserve utility. We prove that F-SOSA achieves vanishing training error and reasonably small generalization error bounds. Building on this stability, we further develop two single-round editing algorithms: F-SOSA-U for unlearning and F-SOSA-I for incremental learning. They ensure that the edited model, along with all intermediate algorithmic states, is statistically indistinguishable from retraining while incurring only an expected $O ( \bar { \rho } )$ fraction of its cost, where ρ quantifies stability. Experiments demonstrate that our approach matches retraining performance while reducing cumulative time and communication overhead by at least 36% and 22%, respectively, under realistic urban mobility simulations.

Index Terms—Federated Learning, Dynamic Participation, Model Editing, Single-Round Communication

## I. INTRODUCTION

Federated learning (FL) [1] enables collaborative model training across distributed devices without sharing raw data.

Communication efficiency is also critical in bandwidthconstrained edge environments [4]–[6]. To reduce communication rounds and reliance on persistent availability, oneshot FL [7] has emerged, which completes training in a single round. Recent studies [8]–[10] show that one-shot FL can achieve competitive performance over non-iid data. This makes it well-suited for dynamic systems. Vehicular networks provide a natural setting for collaborative learning across vehicles [11]. A representative use case is vehicular micro-clouds (VMCs) [12], where nearby vehicles temporarily pool their resources for edge collaboration. Vehicle mobility leads to changes in VMC membership as vehicles join and leave [13], making persistent multi-round participation difficult to maintain. One-shot FL is therefore a natural solution.

In mobile edge computing (MEC) scenarios such as vehicular networks, client participation is inherently dynamic. Devices frequently join or leave due to mobility or network instability. This challenges standard FL algorithms like FedAvg [1], Fed-Prox [2], and SCAFFOLD [3]. These methods rely on a largely fixed client population and multiple rounds of communication. Such assumptions rarely hold in practice.

However, one-shot FL suffers from model solidification. Once trained, the global model cannot adapt to changes in the client population. When clients arrive or depart, the set of participants contributing to the collaborative model changes. A newly arrived client can contribute its local model and benefit from the resulting global model, whereas a departing client may withdraw its previous contribution for privacy before disconnecting from the coordinator. Supporting such contribution additions and withdrawals requires the one-shot model to accommodate incremental learning and unlearning after its initial training. This issue is particularly relevant in dynamic edge applications such as VMCs, where vehicles participate only temporarily and repeated retraining after every arrival or departure would incur substantial communication and computation overhead. Approximate editing methods such as fine-tuning or model calibration can reduce this cost, but successive approximate updates may deviate from the model produced by retraining and propagate this deviation to later edits. Exact editing instead restores the same distribution as retraining after every change, preventing such history-dependent drift. These challenges motivate the following question:

Can we efficiently edit a one-shot trained FL model to reflect an updated client set without retraining?

includes two tasks: (1) client unlearning to remove a client’s previous contribution upon a withdrawal request, and (2) client incremental learning to incorporate the contribution of a newly participating client. By exactness, we require statistical indistinguishability between editing and retraining, which applies to both the final model and all intermediate algorithmic states. To achieve this goal, we propose the first exact model editing framework for one-shot FL. The framework consists of a stable one-shot FL algorithm, F-SOSA, and two editing algorithms: F-SOSA-U for client unlearning and F-SOSA-I for client incremental learning. Our main contributions are summarized as follows:

1) We formalize the definition of exact federated model editing through statistical indistinguishability between editing and retraining. Specifically, we require that the joint distributions of the obtained model and all intermediate algorithmic states produced by editing match those generated by retraining on the updated client set.

2) We design F-SOSA as an edit-friendly one-shot FL algorithm that combines two complementary mechanisms. First, we reformulate curvature-aware model fusion as a projection characterized by client-additive sufficient statistics, so that a single-client change can be incorporated by updating only the corresponding contribution rather than rerunning the full fusion procedure. Second, we introduce randomized client subsampling to make the one-shot output stable in total variation, thereby allowing many membership changes to reuse the existing model and states without recomputation.

3) We provide theoretical utility guarantees for F-SOSA. Specifically, we establish upper bounds on both training and generalization errors for over-parameterized neural networks. The training error vanishes as network width and local training increase. The generalization error converges to a constant governed by data non-degeneracy and becomes favorably small as the numbers of data samples and clients grow.

4) Building upon F-SOSA, we develop F-SOSA-U and F-SOSA-I for client unlearning and incremental learning. Thanks to F-SOSA’s total variation stability quantified by $\rho < 1$ , not all editing requests need execution. We design a lightweight verification to determine the necessity. As a result, only an expected O(ρ)-fraction of requests trigger the editing process.

5) We conduct extensive experiments on standard benchmarks to evaluate model utility, editing exactness, and efficiency. F-SOSA achieves strong accuracy compared with stateof-the-art one-shot FL methods. F-SOSA-U/I match retraining performance while significantly reducing time and communication costs. We verify the exactness of editing using membership inference attacks. We also present a case study with simulated vehicular scenarios to demonstrate the practical applicability of our proposed approach.

## II. RELATED WORK

We review three related research directions in FL: unlearning, continual learning, and dynamic client participation.

Federated unlearning. Federated unlearning was first studied in [14], which removes a client approximately by reversing its updates. While faster than retraining, it remains computationally expensive. Subsequent works improve efficiency through noise perturbation [15]–[17], knowledge distillation [18], class-discriminative pruning [19], or gradient ascent [20]. However, all these methods provide only approximate removal. As a result, the model remains vulnerable to membership inference and gradient leakage [21]–[27]. Recent studies explore exact unlearning. Liu et al. [28] propose a Newton-style retraining method that uses Fisher information for curvature approximation. However, their retraining path deviates from the original trajectory and relies on convexity assumptions. Tao et al. [29] introduce a stabilized FedAvg variant called FATS for exact unlearning via resampling and rapid retraining. Their one-shot variant, however, suffers from degraded utility due to simple averaging and does not support client addition.

Federated continual learning. Federated continual learning assumes a fixed set of clients whose local data evolve over time. Existing work considers different settings including task-incremental [30]–[33], class-incremental [34]–[36], and domain-incremental [37]–[39], which correspond to sequential tasks, new classes, and domain shifts, respectively. The primary goal is to mitigate catastrophic forgetting and preserve past knowledge. As they prioritize knowledge retention over exact alignment with new data, these methods cannot guarantee statistical equivalence to retraining. In addition, they rely on multi-round training and continued access to past clients. In contrast, we study dynamic client populations where new clients may join after others have left.

Federated Learning with dynamic participation. Recent studies consider federated learning with dynamic participants. Ruan et al. [40] and Chang et al. [41] allow clients to join or leave during training and provide convergence guarantees. Their setting differs fundamentally from ours. They study multi-round federated training where client changes occur across rounds and the goal is to adapt during training. In contrast, we focus on one-shot FL, which involves only a single round. In this setting, client dynamics arise after training rather than during it. Our goal is therefore to enable exact model editing of the trained model to accommodate changes in the client population.

Summary. Existing studies address related but distinct challenges. Federated unlearning focuses on removing clients but typically provides only approximate guarantees or suffers from utility degradation in one-shot settings. Federated continual learning assumes a fixed client set and prioritizes knowledge retention rather than exact alignment with updated data. Work on dynamic participation adapts to client churn during multiround training and does not apply to post-training model editing in one-shot FL. Consequently, none of these directions directly address our setting, where client additions and withdrawals occur after one-shot training. This challenge is amplified under repeated edits, since any residual deviation may propagate to subsequent updates. The goal is therefore not merely post-training adaptation, but efficient and exact editing that remains valid across successive client-set changes.

## III. PRELIMINARIES

## A. System Model

We consider an edge system consisting of a trusted infrastructure-based coordinator and a dynamic set of devices. The coordinator can, for example, be an RSU or MEC server in a vehicular micro-cloud (VMC), with vehicles serving as participating devices. Time proceeds in discrete rounds. Devices may join or leave at each round. We do not impose a particular distribution on client arrivals or departures; their occurrence may be non-uniform across devices and time. Let $\mathcal { M } ( t )$ denote the set of active devices at round t, and let $\boldsymbol { M } ( t ) ~ = ~ \boldsymbol { \lvert M ( t ) \rvert }$ |. The coordinator maintains knowledge of the active device set. When a new device joins, it establishes a connection with the coordinator and may submit its local model for inclusion in the collaborative model. Before an active device departs and loses connectivity to the coordinator, it may issue an unlearning request to withdraw its previous contribution. Standard authenticated communication between devices and the coordinator is assumed.

## B. Federated Learning

Let M denote the number of active devices. Each device $i \in [ M ]$ holds a local dataset $\mathcal { D } _ { i } = \{ ( x _ { i j } , y _ { i j } ) \} _ { j = 1 } ^ { n }$ , where $\pmb { x } _ { i j } \in \mathbb { R } ^ { p }$ denotes the feature vector and $y _ { i j } \in \mathbb { R }$ denotes the label. The global dataset is $\textstyle { \mathcal { D } } = \bigcup _ { i = 1 } ^ { M } { \mathcal { D } } _ { i }$ with total size $N = M n$ . The learning objective can be formulated as the following distributed optimization problem:

$$
\operatorname* { m i n } _ { W \in \mathbb { R } ^ { d } } L _ { \mathcal { D } } ^ { f , \ell } ( W ) : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } L _ { \mathcal { D } _ { i } } ^ { f , \ell } ( W ) ,\tag{1}
$$

where $\begin{array} { r } { L _ { \mathcal { D } _ { i } } ^ { f , \ell } ( W ) : = \frac { 1 } { n } \sum _ { ( \pmb { x } , y ) \in \mathcal { D } _ { i } } \ell ( f ( W , \pmb { x } ) , y ) , f ( W , \cdot ) } \end{array}$ denotes the model parameterized by W, and $\ell ( \cdot , \cdot )$ denotes the loss function. Following the standard streaming-edit setting [42], we consider editing sequences in which the active population remains within a constant factor of its initial size, i.e., $M ( 0 ) / 2 \le M ( t ) \le 2 M ( 0 )$

## C. Federated Model Editing

To support dynamic client participation, our goal is to enable the global model to adapt efficiently to changes in the client set. Specifically, we aim to (i) remove the influence of departed clients that withdraw their previous contributions (client unlearning), and (ii) incorporate newly arrived clients (client incremental learning), while preserving model utility and avoiding full retraining.

Let M denote the space of devices, W the space of model parameters, and S the space of internal algorithmic states, which will be specified later. A one-shot FL algorithm is a randomized map $\mathcal { A } : \mathfrak { M } ^ { * } \to \mathcal { W } \times \mathcal { S }$ . Given a set of devices, it produces a global model together with internal algorithmic states. A model editing algorithm is a map U : $\mathcal { W } \times \mathcal { S } \times \mathfrak { M } ^ { * } $ ${ \boldsymbol { \nu } } \times { \boldsymbol { S } }$ . It takes the current model, the associated states, and a set of client addition or removal requests, and outputs an updated model and states. Examples of internal states include sketches of device information or intermediate computations that can be reused during editing.

Definition 1 (Exact Federated Model Editing). Let M and M<sup>′</sup> denote the device sets before and after a change. Given a one-shot FL algorithm ${ \mathcal { A } } ,$ a model editing algorithm U satisfies exact model editing if for any measurable event ${ \mathcal { E } } \subseteq { \mathcal { W } } \times S ,$

$$
\begin{array} { r } { \mathbb { P } \big [ \boldsymbol { A } ( \boldsymbol { \mathcal { M } } ^ { \prime } ) \in \mathcal { E } \big ] = \mathbb { P } \big [ \boldsymbol { \mathcal { U } } ( \boldsymbol { A } ( \boldsymbol { \mathcal { M } } ) , \boldsymbol { \mathcal { M } } \oplus \boldsymbol { \mathcal { M } } ^ { \prime } ) \in \mathcal { E } \big ] , } \end{array}
$$

where $\mathcal { M }$ ⊕ $\mathcal { M } ^ { \prime }$ denotes the symmetric difference, i.e., M ⊕ $\mathcal { M } ^ { \prime } = \left( \mathcal { M } \setminus \mathcal { M } ^ { \prime } \right) \cup \left( \mathcal { M } ^ { \prime } \setminus \mathcal { M } \right)$

This definition requires that both the output model and all internal algorithmic states after editing be statistically indistinguishable from those obtained by retraining on the updated device set. Requiring exactness for the internal states is important for repeated editing: these states are reused by subsequent editing operations, and any approximation retained in them may be propagated to later model updates. If one edit produces exactly the same joint distribution as retraining, the resulting state can be treated as a fresh execution of the learning algorithm on the updated client set. The guarantee thus extends to an arbitrary sequence of edits by induction, without accumulating approximation error across requests.

## D. Total Variation (TV) Stability

Tao et al. [29] showed that exact federated unlearning can be achieved efficiently when the FL algorithm is stable in total variation (TV) distance. We extend this insight to general model editing. Let $P = \mathcal { A } ( \mathcal { M } )$ and $Q = \mathcal { A } ( \mathcal { M } ^ { \prime } )$ denote the distributions of the one-shot FL output, including the model and internal states, before and after editing. Model editing can be viewed as transporting probability mass from $P$ to $Q .$

Retraining corresponds to a trivial transport plan that samples independently from $Q ,$ but it incurs high computational cost. Efficient editing instead exploits the overlap between $P$ and $Q$ by reusing existing computations. Let $\Pi ( P , Q )$ denote the set of couplings between $P$ and $Q .$ . To quantify editing cost, we follow [29] and adopt the 0-1 cost function $c ( x , y ) = \mathbb { 1 } \{ x \neq y \}$ ， ∀x $, y \in \Omega : = \mathcal { W } \times \mathcal { S } .$ . Let $\mathrm { T V } ( P , Q )$ denote the total variation distance between $P$ and $Q \colon$

$$
\mathrm { T V } ( P , Q ) : = \operatorname* { s u p } _ { \mathrm { m e a s u r a b l e } \ Z } | P ( Z ) - Q ( Z ) | = \frac { 1 } { 2 } \| \phi _ { P } - \phi _ { Q } \| _ { 1 } ,
$$

where $\phi _ { P }$ and $\phi _ { Q }$ are the probability densities of $P$ and $Q .$ The minimal expected editing cost is then

$$
\operatorname* { i n f } _ { \pi \in \Pi ( P , Q ) } \mathbb { E } _ { ( x , y ) \sim \pi } [ \Im \left\{ x \neq y \right\} ] = \mathrm { T V } ( P , Q ) .
$$

Thus, only an $O ( \mathrm { T V } ( P , Q ) )$ fraction of editing requests requires actual recomputation. In practice, the future device changes are unknown at training time. When generating $P ,$ we do not know the corresponding target distribution $Q .$ To enable efficient editing under arbitrary future changes, a oneshot FL algorithm must guarantee uniformly bounded TV distance across all possible single-device edits. This motivates the following notion of stability.

Definition 2 (ρ-TV-Stable One-Shot FL Algorithm). A one-shot FL algorithm $\mathcal { A }$ is ρ-TV-stable if for any device sets $\mathcal { M }$ and $\mathcal { M } ^ { \prime }$ with $\vert { \mathcal { M } } \oplus { \mathcal { M } } ^ { \prime } \vert \ = \ 1$ , we have

Algorithm 1: F-SOSA: Fisher-Informed Subsample   
One-Shot Averaging   
Input: Initial model ${ \bf W } ^ { ( 0 ) }$ , number of local iterations   
$K ,$ learning rate $\eta ,$ stability $\rho ,$ sample size $P$   
1 Coordinator samples $\mathcal { P }$ uniformly from the size- $. P$   
subsets of $\mathcal { M } ;$   
2 Coordinator broadcasts ${ \bf W } ^ { ( 0 ) }$ to clients;   
3 for every client $i \in \mathcal { P }$ in parallel do   
4 Set $W _ { i } ^ { ( 0 ) }  W ^ { ( 0 ) }$ ;   
5 for k $ 1 , \cdots , K$ do   
6 $\pmb { W } _ { i } ^ { ( k ) }  \dot { \pmb { W } } _ { i } ^ { ( k - 1 ) } - \eta \nabla L _ { \mathcal { D } _ { i } } ( \pmb { W } _ { i } ^ { ( k - 1 ) } ) ;$   
7 Set $\tilde { \pmb { W } } _ { i } \gets \pmb { W } _ { i } ^ { ( K ) } ;$   
8 Compute the Fisher $\mathbf { } F _ { i } \gets$   
$\begin{array} { r l } {  { \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \mathbb { E } _ { \pmb { y } } \bigl [ \nabla \log \mathbb { P } ( \pmb { y } | \pmb { x } _ { i j } , \tilde { \pmb { W } } _ { i } ) \nabla \log \mathbb { P } ( \pmb { y } | \pmb { x } _ { i j } , \tilde { \pmb { W } } _ { i } ) ^ { \top } \bigr ] ; } \quad } & { } \end{array}$   
9 Communicate $\tilde { W } _ { i }$ and $\mathbf { \nabla } \mathbf { F } _ { i }$ to the coordinator;   
10 Coordinator computes and stores $\bar { W } _ { \mathcal { P } } , F _ { \mathcal { P } }$ and $\begin{array} { r } { b _ { \mathcal { P } } \colon } \end{array}$   
$\bar { W } _ { \mathcal { P } } \longleftarrow \sum _ { i \in \mathcal { P } } \tilde { W } _ { i } / P , F _ { \mathcal { P } } \longleftarrow \sum _ { i \in \mathcal { P } } F _ { i } , b _ { \mathcal { P } }  \sum _ { i \in \mathcal { P } } F _ { i } \tilde { W } _ { i } ;$   
11 Coordinator solves the linear system in $\pmb { \Lambda } \in \mathbb { R } ^ { d } ;$   
$F _ { \mathcal P } ^ { 2 } \pmb { \Lambda } = 2 \big ( F _ { \mathcal P } \bar { \pmb { W } } _ { \mathcal P } - \pmb { b } _ { \mathcal P } \big )$   
12 Coordinator sets the global model as   
$\begin{array} { r } { \bar { \mathbf { W } } _ { \mathcal { P } } ^ { \mathrm { F } }  \bar { \mathbf { W } } _ { \mathcal { P } } - \frac { 1 } { 2 } F _ { \mathcal { P } } \mathbf { \Lambda } \mathrm { . } } \end{array}$   
Output: $\bar { \boldsymbol { W } } _ { \mathcal { P } } ^ { \mathrm { F } }$

TV $\left( \mathcal { A } _ { 1 } ( \mathcal { M } ) , \mathcal { A } _ { 1 } ( \mathcal { M } ^ { \prime } ) \right) \leq \rho ,$ , where $\mathcal { A } _ { 1 } ( \cdot )$ denotes the model output of A.

Note that Definition 2 only requires stability of the marginal distribution of model output. The internal states need not be considered. For multiple device changes, suppose $| { \mathcal { M } } \backslash { \mathcal { M } } ^ { \prime } | =$ $k _ { 1 }$ and $| { \mathcal { M } } ^ { \prime } \setminus { \mathcal { M } } | = k _ { 2 }$ . By the triangle inequality of TV distance and repeated application of Definition 2, we have TV $\left( \mathcal { A } _ { 1 } ( \mathcal { M } ) , \mathcal { A } _ { 1 } ( \mathcal { M } ^ { \prime } ) \right) \leq ( k _ { 1 } + k _ { 2 } ) \rho .$

## IV. OUR METHOD

## A. TV-Stable One-Shot Federated Learning: F-SOSA

The federated learning admits a posterior-inference interpretation. Specifically, the learning objective in (1) corresponds to finding the mode of a global posterior distribution over model parameters. This viewpoint has recently been used to derive one-shot FL algorithms that aggregate local posterior information in a single communication round [8]. We adopt this probabilistic formulation as a technical starting point, but our design goal is fundamentally different. Existing posteriorbased one-shot FL methods primarily optimize a single fusion operation. In contrast, our setting requires the fused model to support repeated post-training edits as the client set changes. This creates two additional requirements. First, the fusion state should admit efficient single-client addition and removal without rerunning the entire fusion procedure. Second, the output should remain stable under a single-client change so that previously computed results can be reused whenever possible. We address these requirements in two stages: we first derive a curvature-aware projection characterized by client-additive statistics, and then introduce randomized client subsampling to control total-variation stability.

We first recall the probabilistic formulation. Following standard treatments in posterior-based FL [8], we assume the loss corresponds to a negative log-likelihood from an exponential-family model, which holds for common choices such as squared loss and cross-entropy.

Assumption 1. Given $\begin{array} { r l r } { z } & { { } = } & { f ( W , \pmb { x } ) , \ell ( z , \pmb { y } ) } \end{array}$ ∝ $- \log \mathbb { P } ( \pmb { y } | \pmb { z } )$ where $\mathbb { P } ( \pmb { y } | \pmb { z } ) = h ( \pmb { y } ) \exp ( \pmb { z } ^ { \top } T ( \pmb { y } ) - A ( \pmb { z } ) )$ and $h ( { \pmb y } ) , T ( { \pmb y } ) , A ( { \pmb z } )$ are some real-valued functions.

Under this assumption, the likelihood of a data point $( { \pmb x } , { \pmb y } )$ given parameters $W$ can be written as

$$
\begin{array} { r } { \mathbb { P } \big ( ( { \pmb x } , { \pmb y } ) | { \pmb W } \big ) = \mathbb { P } ( { \pmb x } ) \mathbb { P } \big ( { \pmb y } | { \pmb x } , { \pmb W } \big ) \qquad } \\ { \qquad \propto q ( { \pmb x } ) \exp \big ( - \ell \big ( f ( { \pmb W } , { \pmb x } ) , { \pmb y } \big ) \big ) , } \end{array}
$$

where $q ( \cdot )$ is a prior on x independent of W. Under a flat prior $\mathbb { P } ( W ) \propto 1$ , the global Maximum A Posteriori (MAP) estimator coincides with the minimizer of (1) [43]. Moreover, the global posterior factorizes as

$$
\mathbb { P } ( { \cal W } | \mathcal { D } ) \propto \prod _ { i = 1 } ^ { M } \mathbb { P } ( { \cal W } | \mathcal { D } _ { i } ) .
$$

This factorization suggests a one-shot protocol where each client transmits a summary of its local posterior once and the server combines these summaries to obtain a global MAP estimate. Unfortunately, an analytical expression of $\mathbb { P } ( W | \mathcal { D } _ { i } )$ is generally intractable for modern models. A standard remedy is to approximate each local posterior near its mode. Let

$$
\tilde { \pmb { W } } _ { i } \in \underset { \pmb { W } } { \arg \operatorname* { m a x } } \mathbb { P } ( \pmb { W } | \mathcal { D } _ { i } ) ,
$$

which is obtained by local training. A second-order expansion around $\tilde { W } _ { i }$ yields a Gaussian (Laplace) approximation:

$$
\mathbb { P } ( \mathbf { \boldsymbol { W } } | \mathcal { D } _ { i } ) \approx \log \mathbb { P } ( \tilde { W } _ { i } | \mathcal { D } _ { i } ) - \frac { n } { 2 } \big ( \mathbf { \boldsymbol { W } } - \tilde { \mathbf { \boldsymbol { W } } } _ { i } \big ) ^ { \top } \mathbf { \boldsymbol { H } } _ { i } \big ( \mathbf { \boldsymbol { W } } - \tilde { \mathbf { \boldsymbol { W } } } _ { i } \big ) ,
$$

where

$$
H _ { i } = - \frac 1 n \nabla ^ { 2 } \log \mathbb { P } ( { \cal W } | { \mathcal D } _ { i } ) \big | _ { { \cal W } = { \tilde { \cal W } } _ { i } } .
$$

To ensure a tractable and well-behaved curvature surrogate, $\pmb { H } _ { i }$ is commonly replaced by the Fisher information at $\tilde { W } _ { i } { : }$

$$
\pmb { F } _ { i } = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \mathbb { E } _ { \pmb { y } } \left[ \nabla \log \mathbb { P } ( \pmb { y } | \pmb { x } _ { i j } , \pmb { W } ) \nabla \log \mathbb { P } ( \pmb { y } | \pmb { x } _ { i j } , \pmb { W } ) ^ { \top } \right] ,
$$

evaluated at $\boldsymbol { W } = \tilde { \boldsymbol { W } } _ { i \cdot } \boldsymbol { F } _ { i }$ is symmetric and positive semidefinite (PSD). Computing, storing, and communicating full curvature matrices can be prohibitively expensive in largescale distributed learning [44]. In our setting, the full Fisher matrix $\boldsymbol { F } _ { i } ~ \in ~ \mathbb { R } ^ { d \times d }$ incurs quadratic storage and communication costs. Our fusion formulation does not depend on a particular Fisher representation, so structured approximations can be adopted according to the available computation, storage, and communication resources. For a layer with weight matrix of size $o _ { \ell } \times i _ { \ell }$ and $d _ { \ell } ~ = ~ o _ { \ell } i _ { \ell }$ parameters, Table I summarizes representative per-client Fisher representations. A diagonal approximation retains only the diagonal curvature and has linear representation cost, whereas K-FAC preserves richer layer-wise curvature through Kronecker factors while avoiding the quadratic representation cost of a full Fisher matrix. Thus, the choice of Fisher approximation provides a trade-off between curvature fidelity and resource overhead. Combining these local posterior approximations yields a Fisher-weighted quadratic form for the global log-posterior:

TABLE I  
STRUCTURAL COMPLEXITY OF REPRESENTATIVE PER-CLIENT FISHER REPRESENTATIONS FOR A LAYER WITH $d _ { \ell } = o _ { \ell } i _ { \ell }$ PARAMETERS.
<table><tr><td>Structure</td><td>Storage / Communication</td><td>Curvature Structure</td></tr><tr><td>Full</td><td> $\overline { { O ( d _ { \ell } ^ { 2 } ) } }$ </td><td>Full</td></tr><tr><td>Diagonal</td><td> $\dot { O ( d _ { \ell } ) }$ </td><td>Diagonal only</td></tr><tr><td>K-FAC</td><td> $O ( i _ { \ell } ^ { 2 } + o _ { \ell } ^ { 2 } )$ </td><td>Kronecker-factorized</td></tr></table>

$$
\begin{array} { r l } { \log \mathbb { P } ( { W } | \mathcal { D } ) } & { } \\ { \approx \displaystyle \sum _ { i = 1 } ^ { M } \log \mathbb { P } ( \tilde { W } _ { i } | \mathcal { D } _ { i } ) - \frac { n } { 2 } \displaystyle \sum _ { i = 1 } ^ { M } ( W - \tilde { W } _ { i } ) ^ { \top } F _ { i } ( W - \tilde { W } _ { i } ) . } \end{array}
$$

Maximizing this approximation is equivalent to solving

$$
\operatorname* { m i n } _ { W \in \mathbb { R } ^ { d } } \sum _ { i = 1 } ^ { M } ( W - \tilde { W } _ { i } ) ^ { \top } F _ { i } ( W - \tilde { W } _ { i } ) .\tag{2}
$$

Since each ${ \bf { F } } _ { i } \succeq 0$ , any minimizer satisfies the first-order condition

$$
\left( \sum _ { i = 1 } ^ { M } F _ { i } \right) W = \sum _ { i = 1 } ^ { M } F _ { i } { \tilde { W } } _ { i } .\tag{3}
$$

In over-parameterized regimes $( d \gg N )$ , the system in (3) can admit multiple solutions. To select a unique and wellbehaved estimator, an effective selection rule is to choose the feasible solution that stays closest to the collection of local optima [8]. This leads to the following constrained leastsquares formulation

$$
\operatorname* { m i n } _ { W \in \mathbb { R } ^ { d } } \left\{ \sum _ { i = 1 } ^ { M } \| W - \tilde { W } _ { i } \| _ { 2 } ^ { 2 } \mathrm { s . t . } \left( \sum _ { i = 1 } ^ { M } F _ { i } \right) W = \sum _ { i = 1 } ^ { M } F _ { i } \tilde { W } _ { i } \right\} .\tag{4}
$$

A solution to (2) can be obtained by iterative optimization [8]. Starting from the average initialization $\begin{array} { r } { { \cal W } ^ { [ 0 ] } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \tilde { \tilde { W } _ { i } } . } \end{array}$ gradient descent with step size $\eta _ { S } \leq 1 / \lambda _ { \operatorname* { m a x } } .$ , where $\lambda _ { \mathrm { m a x } }$ is the largest eigenvalue of $\sum { _ { i = 1 } ^ { M } \pmb { F } _ { i } } ,$ generates iterates $W ^ { [ t ] } \gets$ $\begin{array} { r } { W ^ { [ t - 1 ] } - \tilde { \eta } \sum _ { i = 1 } ^ { M } F _ { i } ( W ^ { [ t - 1 ] } - \tilde { W } _ { i } ) } \end{array}$ , which converge asymptotically to a minimizer $W ^ { \star }$ . While such iterative fusion is suitable for one-time aggregation, it is poorly matched to repeated model editing. When a single client changes, the fusion objective changes accordingly, and the iterative procedure generally needs to be rerun to obtain the updated solution. Moreover, the previous optimization trajectory is not directly reusable for updating only the affected client contribution. This motivates the following projection formulation. Let

$$
\bar { \boldsymbol { W } } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \tilde { \boldsymbol { W } } _ { i } .\tag{5}
$$

Using the identity

$$
\sum _ { i = 1 } ^ { M } \| \boldsymbol { W } - \tilde { \boldsymbol { W } } _ { i } \| _ { 2 } ^ { 2 } = M \| \boldsymbol { W } - \bar { \boldsymbol { W } } \| _ { 2 } ^ { 2 } + \sum _ { i = 1 } ^ { M } \| \bar { \boldsymbol { W } } - \tilde { \boldsymbol { W } } _ { i } \| _ { 2 } ^ { 2 } ,
$$

problem (4) is equivalent to

$$
\operatorname* { m i n } _ { W \in \mathbb { R } ^ { d } } \left\{ \| W - \bar { W } \| _ { 2 } ^ { 2 } \mathrm { ~ s . t . ~ } F W = b \right\} ,\tag{6}
$$

where

$$
\pmb { F } : = \sum _ { i = 1 } ^ { M } \pmb { F } _ { i } , \qquad \pmb { b } : = \sum _ { i = 1 } ^ { M } \pmb { F } _ { i } \tilde { \pmb { W } } _ { i } .\tag{7}
$$

Hence, the fusion state needed for subsequent editing is summarized by $( \bar { W } , F , b )$ , each of which decomposes over clients. Under a single-client replacement, these statistics can be updated by subtracting the outgoing contribution and adding the incoming one, while all unchanged client contributions are reused. Since each $\begin{array} { r } { \pmb { F } _ { i } \succeq 0 , \pmb { b } = \sum _ { i } \pmb { F } _ { i } \widetilde { \pmb { W } } _ { i } \in } \end{array}$ ${ \mathrm { R a n g e } } ( F )$ , so the affine constraint $F W = b$ in (6) is feasible. Thus, solving (6) reduces to computing the orthogonal projection of W<sup>¯</sup> onto the affine set $\{ W : F W = b \}$ . This projection admits an explicit characterization via the KKT conditions. Consider the Lagrangian

$$
\mathcal { L } ( W , \Lambda ) = \| W - \bar { W } \| _ { 2 } ^ { 2 } + { \Lambda ^ { \top } } ( F W - b ) ,\tag{8}
$$

where $\pmb { \Lambda } \in \mathbb { R } ^ { d }$ is the dual variable. Since the objective is strictly convex and the constraint is affine, the KKT conditions are necessary and sufficient for optimality. Taking derivatives of (8) yields the stationarity and primal feasibility conditions:

$$
\left\{ \begin{array} { r } { 2 ( W - \bar { W } ) + F \Lambda = \pmb { 0 } , } \\ { F W - b = \pmb { 0 } . } \end{array} \right.\tag{9}
$$

From the first condition,

$$
\begin{array} { r } { W = \bar { W } - \frac { 1 } { 2 } F \Lambda . } \end{array}\tag{10}
$$

Substituting into the constraint yields a linear system in the dual variable Λ:

$$
F ^ { 2 } \Lambda = 2 ( F \bar { W } - b ) .\tag{11}
$$

Therefore, model fusion reduces to solving the linear system (11). The linear system in (11) remains consistent even when the aggregated Fisher matrix F is sparse, singular, or low-rank. Since $b \in \mathrm { R a n g e } ( F )$ and $F \bar { W } \in \mathrm { R a n g e } ( F )$ , we have $F \bar { W } - b \in \mathrm { R a n g e } ( F )$ . Moreover, because $\pmb { F }$ is symmetric and positive semidefinite, $\operatorname { R a n g e } ( F ^ { 2 } ) = \operatorname { R a n g e } ( F )$ Therefore, (11) always admits a solution. Rank deficiency may make the dual solution $\Lambda ^ { \star }$ non-unique. However, if $\pmb { \Lambda } _ { 1 } ^ { \star }$ and $\boldsymbol { \Lambda } _ { 2 } ^ { \star }$ are two solutions, then $F ^ { 2 } ( \mathbf { A } _ { 1 } ^ { \star } - \mathbf { \bar { A } } _ { 2 } ^ { \star } ) = 0$ . For symmetric positive semidefinite F, $\mathrm { N u l l } ( { \cal F } ^ { 2 } ) \ = \ \mathrm { N u l l } ( { \cal F } )$ , and hence $F ( \mathbf { \vec { \Lambda } } _ { 1 } ^ { \star } - \mathbf { \Lambda } \mathbf { \vec { \Lambda } } _ { 2 } ^ { \star } ) = 0$ . Thus, all dual solutions yield the same $\pmb { F } \pmb { \Lambda } ^ { \star }$ and therefore the same projected primal solution $W ^ { \star } =$ $\bar { \bf W } \mathrm { ~ - ~ } \frac { 1 } { 2 } { \cal F } { \bf A } ^ { \star }$ . Hence, rank deficiency may affect numerical conditioning and dual uniqueness, but not the existence or uniqueness of the projected primal solution. Once a solution $\Lambda ^ { \star }$ is obtained, the optimal model follows directly from (10):

$$
\begin{array} { r } { W ^ { \star } = \bar { W } - \frac { 1 } { 2 } F \Lambda ^ { \star } . } \end{array}\tag{12}
$$

This formulation makes clear that the fusion problem decomposes into two steps: solving a linear system defined by aggregated curvature statistics, and then recovering the projected model in closed form.

The linear system (11) in Λ can be solved efficiently and accurately by standard linear solvers. For example, conjugate gradient (CG) applies here because $\pmb { F } ^ { 2 }$ is symmetric positive semidefinite and the linear system (11) is always consistent. In particular, CG minimizes the quadratic energy

$$
\operatorname* { m i n } _ { \Lambda } \ { } ^ { 1 } 2 { \Lambda } ^ { \top } F ^ { 2 } \Lambda - { \Lambda } ^ { \top } 2 ( F \bar { W } - b ) ,\tag{13}
$$

whose first-order optimality condition is exactly (11). CG terminates in at most $r = \mathrm { r a n k } ( F )$ iterations, which is typically much smaller than d in over-parameterized settings. This follows from the fact that CG minimizes the quadratic energy over an expanding Krylov subspaces and the Krylov subspace associated with the nonzero eigenspace has dimension at most $r = \mathrm { r a n k } ( F )$ . The practical cost of the projection step also depends on the Fisher representation. A diagonal approximation incurs the lowest structural overhead, while K-FAC retains richer curvature information and substantially reduces representation and communication costs relative to a full Fisher matrix. However, a sum of client-wise K-FAC matrices does not generally preserve a single Kronecker factorization. Since our coordinator stores only the aggregated curvature statistic rather than individual K-FAC factors, this structure may be lost after aggregation, and the coordinator-side projection cost can approach that of a less structured representation. Hence, the Fisher approximation can be selected according to the desired trade-off among curvature fidelity, communication overhead, and coordinator-side computation.

To enable efficient exact model editing, the one-shot FL algorithm must be ρ-TV-stable (Definition 2) for a pre-specified stability level $\rho .$ The fusion rule above uses all available clients and is deterministic. This benefits model utility, but it yields no distributional stability. In particular, for any neighboring device sets M and $\mathcal { M } ^ { j }$ with $\vert \mathcal { M } \oplus \mathcal { M } ^ { \prime } \vert = \mathrm { ~ \overset { . } { 1 } ~ }$ , the outputs $A _ { 1 } ( \mathcal { M } )$ and $\mathbf { } A _ { 1 } ( \mathcal { M } ^ { \prime } )$ are Dirac measures supported at different points. Hence, $\mathrm { T V } ( \mathcal { A } _ { 1 } ( \mathcal { M } ) , \mathcal { A } _ { 1 } ( \mathcal { M } ^ { \prime } ) ) = 1$ , which cannot meet an arbitrary target $\rho < 1$ . We address this by introducing randomized client subsampling for model aggregation. Let $M _ { 0 }$ denote the number of clients at initialization. We set $P =$ $\lfloor \rho M _ { 0 } \rfloor$ once and retain this subset size throughout the editing sequence. The coordinator samples a client subset ${ \mathcal { P } } \subseteq [ M ]$ of size $P$ for local training and model fusion. This shared randomness induces overlap between the output distributions under neighboring client sets, which is directly controlled by the sampling rate $\rho .$ Note that, for the general dynamicdevice learning setting considered in this work, we assume that client contributions have comparable aggregation priority, under which uniform random subsampling is appropriate.

Combining Fisher-informed projection with client subsampling yields our ρ-TV-stable one-shot FL algorithm, F-SOSA (Fisher-Informed Subsample One-Shot Averaging). Algorithm 1 summarizes the procedure. The coordinator samples $\mathcal { P }$ (step 1) and broadcasts the initialization ${ \bf W } ^ { ( 0 ) }$ (step 2). Here, $P ~ = ~ \rho M$ is determined at the initial execution of F-SOSA and remains fixed during all subsequent editing operations. Each client $i \in \mathcal { P }$ runs K steps of local gradient descent to obtain $\tilde { W } _ { i }$ (steps 3–7), computes its Fisher matrix $\mathbf { \nabla } F _ { i }$ at $\tilde { W } _ { i }$ (step 8), and uploads $( \tilde { W } _ { i } , \boldsymbol { F } _ { i } )$ to the coordinator (step 9). The coordinator then forms the sufficient statistics

```latex
Algorithm 2: F-SOSA-U
Input: Unlearning request from client $i ^ { - }$ with its local
contribution $( \widetilde { W } _ { i ^ { - } } , F _ { i ^ { - } } )$ , initial model ${ \pmb W } ^ { ( 0 ) }$
number of local iterations $K ,$ , learning rate η
1 if $i ^ { - } \in \mathcal { P }$ then
2 Coordinator samples $i ^ { + } \sim$ Uniform $( [ M ] \backslash { \mathcal { P } } ) ;$
3 Coordinator updates ${ \mathcal { P } } \gets { \mathcal { P } } \backslash \{ i ^ { - } \} \cup \{ i ^ { + } \}$
4 Coordinator communicates ${ \bf W } ^ { ( 0 ) }$ to client $i ^ { + } ;$
5 Client $i ^ { + }$ performs local training and Fisher
computation following Lines 4–9 of Algorithm 1,
and sends $( \tilde { W } _ { i ^ { + } } , F _ { i ^ { + } } )$ to the coordinator;
6 Coordinator updates $\bar { W } _ { \mathcal { P } } , F _ { \mathcal { P } }$ and $\begin{array} { r } { b _ { \mathcal { P } } \colon } \end{array}$
$\begin{array} { r } { \bar { W } _ { \mathcal { P } }  \bar { W } _ { \mathcal { P } } + \frac { 1 } { P } \big ( \tilde { W } _ { i ^ { + } } - \tilde { W } _ { i ^ { - } } \big ) , } \end{array}$
$\pmb { F } _ { \mathcal { P } }  \pmb { F } _ { \mathcal { P } } + ( \pmb { F } _ { i ^ { + } } - \pmb { F } _ { i ^ { - } } ) ,$
$\mathfrak { b } _ { \mathcal { P } }  \mathfrak { b } _ { \mathcal { P } } + ( \mathbf { F } _ { i ^ { + } } \tilde { \mathbf { W } } _ { i ^ { + } } - \mathbf { F } _ { i ^ { - } } \tilde { \mathbf { W } } _ { i ^ { - } } ) ;$
7 Coordinator solves the linear system for $\pmb { \Lambda } \in \mathbb { R } ^ { d } ;$
$F _ { \mathcal P } ^ { 2 } \Lambda = 2 \big ( F _ { \mathcal P } \bar { W } _ { \mathcal P } - b _ { \mathcal P } \big ) ;$
8 Coordinator sets the updated global model:
$\begin{array} { r } { \bar { \mathbf { W } } _ { \mathcal { P } } ^ { \mathrm { F } }  \bar { \mathbf { W } } _ { \mathcal { P } } - \frac { 1 } { 2 } F _ { \mathcal { P } } \mathbf { \Lambda } \mathrm { . } } \end{array}$
Output: $\bar { \boldsymbol { W } } _ { \mathcal { P } } ^ { \mathrm { F } }$
```

$$
\bar { \boldsymbol { W } } _ { \mathcal { P } } = \frac { 1 } { P } \sum _ { i \in \mathcal { P } } \tilde { \boldsymbol { W } } _ { i } , \quad \boldsymbol { F } _ { \mathcal { P } } = \sum _ { i \in \mathcal { P } } \boldsymbol { F } _ { i } , \quad b _ { \mathcal { P } } = \sum _ { i \in \mathcal { P } } \boldsymbol { F } _ { i } \tilde { \boldsymbol { W } } _ { i }
$$

as shown in step 10. Finally, it outputs the global model by projecting $\bar { W } _ { \mathcal { P } }$ onto the affine set $\left\{ W : F _ { \mathcal { P } } W = b _ { \mathcal { P } } \right\}$ . This projection is computed by solving the linear system for the dual variable Λ(steps 11) and then recovering the final fused model $\bar { \boldsymbol { W } } _ { \mathcal { P } } ^ { \mathrm { F } }$ (step 12).

## B. Efficient Editing Algorithms: F-SOSA-U and F-SOSA-I

For a fixed subset size $P ,$ F-SOSA aggregates a uniformly sampled client subset $\mathcal { P } .$ . When the client population changes, the support of the uniform size-P subset distribution changes accordingly. Exact model editing therefore reduces to constructing a coupling between these sampling distributions while retaining $P .$ For a current population of M clients, the deletion coupling retains the subset when the departing client is absent and otherwise replaces that client. The insertion coupling includes the new client with probability $P / ( M + 1 )$ by replacing a uniformly selected member; otherwise, it retains the subset. These transitions are maximal couplings of the respective subset distributions, with trigger probabilities $P / M$ and $P / ( M + 1 )$ . Thus, the trigger probabilities depend on the current population even though $P$ remains fixed. Whenever the subset is retained, its contributions, aggregate statistics, and fused model can be reused. We develop the corresponding procedures below: F-SOSA-U for client unlearning (Algorithm 2) and F-SOSA-I for client incremental learning (Algorithm 3). The coordinator persistently maintains only the sampled subset $\mathcal { P }$ and the aggregate statistics $( \bar { W } _ { \mathcal { P } } , F _ { \mathcal { P } } , b _ { \mathcal { P } } )$ Individual client contributions $( \widetilde { W } _ { i } , F _ { i } )$ need not be retained at the coordinator. While connected, each active client keeps its own contribution and transmits it only when required by an editing operation; in particular, a departing client sends its contribution together with the unlearning request before disconnecting. The coordinator then updates the aggregate statistics by subtracting the departing contribution and adding the incoming one. Hence, its persistent state does not grow with the number of clients that have participated historically.

Algorithm 3: F-SOSA-I   
Input: new client index $i ^ { + }$   
1 if Bernoulli $\cdot { \frac { P } { M + 1 } } )$ then   
2 Coordinator samples $i ^ { - } \sim$ Uniform $( { \mathcal { P } } ) ;$   
3 Coordinator requests $( \widetilde { W } _ { i ^ { - } } , F _ { i ^ { - } } )$ from client $i ^ { - }$   
and $( \widetilde { W } _ { i ^ { + } } , F _ { i ^ { + } } )$ from client $i ^ { + } ;$   
4 Coordinator updates $\mathcal { P }  \mathcal { P } \backslash \{ i ^ { - } \} \cup \{ i ^ { + } \}$   
5 Coordinator updates $\bar { W } _ { \mathcal { P } } , F _ { \mathcal { P } }$ and $\begin{array} { r } { b _ { \mathcal { P } } \colon } \end{array}$   
$\bar { W } _ { \mathcal { P } }  \bar { W } _ { \mathcal { P } } + \frac { 1 } { P } \big ( \tilde { W } _ { i ^ { + } } - \tilde { W } _ { i ^ { - } } \big ) .$   
$F _ { \mathcal { P } } \gets F _ { \mathcal { P } } + \left( F _ { i ^ { + } } - F _ { i ^ { - } } \right)$   
$\boldsymbol { b _ { \mathcal { P } } } \gets \boldsymbol { b _ { \mathcal { P } } } + \big ( \boldsymbol { F } _ { i ^ { + } } \tilde { \boldsymbol { W } } _ { i ^ { + } } - \boldsymbol { F } _ { i ^ { - } } \tilde { \boldsymbol { W } } _ { i ^ { - } } \big ) ;$   
6 Coordinator solves the linear system for $\pmb { \Lambda } \in \mathbb { R } ^ { d } ;$   
$F _ { \mathcal P } ^ { 2 } \pmb { \Lambda } = 2 \big ( F _ { \mathcal P } \pmb { \bar { W } } _ { \mathcal P } - \pmb { b } _ { \mathcal P } \big )$ ;   
7 Coordinator updates the global model:   
$\begin{array} { r } { \bar { \mathbf { W } } _ { \mathcal { P } } ^ { \mathrm { F } }  \bar { \mathbf { W } } _ { \mathcal { P } } - \frac { 1 } { 2 } F _ { \mathcal { P } } \mathbf { \Lambda } \mathrm { . } } \end{array}$   
Output: $\bar { \boldsymbol { W } } _ { \mathcal { P } } ^ { \mathrm { F } }$

• Client unlearning (F-SOSA-U). When the coordinator receives an unlearning request from client $i ^ { - }$ , the coordinator first checks whether it belongs to the sampled subset $\mathcal { P } _ { \cdot }$ If $i ^ { - } \notin \mathcal { P }$ , then under the maximal coupling the subset remains unchanged, and no editing is needed. Otherwise, we must replace the removed client to preserve the subset size and sampling distribution. The coordinator therefore draws a replacement client $i ^ { + }$ uniformly from $[ M ] \backslash \mathcal { P }$ and updates $\mathcal { P }  \mathcal { P } \backslash \{ i ^ { - } \} \cup \{ i ^ { + } \}$ . The new client performs local training and Fisher computation, after which the coordinator updates the sufficient statistics via client-additive adjustments: $\bar { W } _ { \mathcal { P } } , F _ { \mathcal { P } } , b _ { \mathcal { P } }$ . The edited global model is then obtained by solving the same projection system as in F-SOSA. Since only one client contribution changes, the update cost is independent of the total number of clients.

• Client incremental learning (F-SOSA-I). When a new client $i ^ { + }$ joins, we determine whether it would have been selected under the updated F-SOSA sampling distribution. Conditional on the current client set in the learning objective containing $M + 1$ clients, F-SOSA samples a size-P subset uniformly at random. Hence, each client in the updated set, including the newly arrived client $i ^ { + }$ , has inclusion probability $\operatorname* { P r } ( i ^ { + } \in { \mathcal { P } } ^ { \prime } ) = P / ( M + 1 )$ . This probability is determined by the conditional subset-sampling rule and is independent of the external arrival distribution. If the

Bernoulli trial fails, the sampled subset remains unchanged under the coupling. Otherwise, the coordinator replaces a uniformly chosen client $i ^ { - } \in \mathcal { P }$ with $i ^ { + }$ and updates the sufficient statistics using the same adjustments as in unlearning.

Both editing algorithms modify only a single selected-client contribution while reusing the current aggregate fusion statistics. Consequently, each triggered editing operation updates only one client contribution to the sufficient statistics and performs one projection solve. Its coordinator-side cost depends on the chosen Fisher representation, as discussed above, but does not require recomputing the contributions of all previously sampled clients. Moreover, since editing is triggered with probability at most $O ( \rho )$ , the expected editing cost per client change is $O ( \rho )$ relative to full retraining. Together, these procedures enable provably exact and communicationefficient model adaptation to dynamic client participation in one-shot federated learning. For sequential requests, F-SOSA-U/I update the model after each request, preserving fixed- $P$ exactness by induction.

Finally, we consider simultaneous requests. A batch specifies deletions $\mathcal { D } \subseteq \mathcal { M }$ and additions ${ \mathcal { A } } \cap { \mathcal { M } } = \emptyset , $ independently of the sampling randomness. Let $M = | { \mathcal { M } } | , d = | { \mathcal { D } } |$ $a = | { \mathcal { A } } | , { \mathcal { M } } ^ { \prime } = ( { \mathcal { M } } \setminus { \mathcal { D } } ) \cup { \mathcal { A } } ,$ and $M ^ { \prime } = M - d + a$ , with $1 \leq P \leq \operatorname* { m i n } \{ M , M ^ { \prime } \}$ . We condition on the shared initialization and fixed per-client training randomness, independent of subset sampling. Each identifier denotes an immutable local contribution under this configuration; a changed contribution uses a new identifier. Algorithm 4 pairs min $\{ d , a \}$ departing and arriving clients independently of the realized subset. Each selected departing identifier is replaced by its paired arriving identifier. Unmatched arrivals are handled by the fixed- $. P$ insertion coupling; unmatched departures are removed together and the subset is uniformly replenished. Pairing operates only on identifiers: each added client supplies its own model and Fisher information. Once the final subset ${ \mathcal { P } } ^ { \prime }$ is obtained, the coordinator applies only the net changes $S _ { - } = \mathcal { P } \setminus \mathcal { P } ^ { \prime }$ and $S _ { + } = \mathcal { P } ^ { \prime } \setminus \mathcal { P }$ to the original aggregate statistics. Departing selected clients provide their original contributions before disconnecting, with temporary buffering until batch completion. The model is reused if the final subset is unchanged; otherwise, one projection is performed. No intermediate model is released. Proposition 1 establishes exactness and shows that this coupling minimizes both the probability of a subset change and the expected number of replaced members.

## V. THEORETICAL RESULTS

## A. Stability of F-SOSA

We consider a general device set M with $| { \mathcal { M } } | = M ,$ and note that all results apply to any round t with device set $\mathcal { M } ( t )$ . We first prove that F-SOSA is $\rho { \mathrm { - T V } }$ -stable for any given stability level $\rho ,$ and then analyze its learning utility. Our stability analysis uses the concept of the pushforward measure: for a measurable map $f : \mathcal { X }  \mathcal { Y }$ and a measure $\mu$ on $\mathcal { X } .$ , the push-forward $f \# \mu$ is defined by $( f \# \mu ) ( \mathcal { Z } ) = \mu ( f ^ { - 1 } ( \mathcal { Z } ) )$ for any measurable set $\mathcal { Z } \subseteq \mathcal { V }$

Algorithm 4: F-SOSA-Batch: Paired Joint Editing   
Input: Current client set ${ \overline { { \mathcal { M } } } } ,$ size-P subset ${ \overline { { \mathcal { P } } } } ,$ aggregate   
statistics and fused model; deletion set D and   
addition set $\boldsymbol { A }$   
1 Set $\mathcal { M } ^ { \prime }  ( \mathcal { M } \setminus \mathcal { D } ) \cup \mathcal { A } ;$   
2 Buffer original contributions of $\mathcal { P } \cap \mathcal { D }$ before their clients   
disconnect;   
3 Choose $\mathcal { D } _ { 0 } \subseteq \mathcal { D } , \mathcal { A } _ { 0 } \subseteq \mathcal { A }$ of size min $\{ | \mathcal { D } | , | A | \}$ and a   
bijection $\pi : { \mathcal { D } } _ { 0 }  { \mathcal { A } } _ { 0 } ,$ independently of ${ \dot { \mathcal { P } } } ;$   
4 Set $\widehat { \mathcal { M } } \gets ( \mathcal { M } \setminus \mathcal { D } _ { 0 } ) \cup \mathcal { A } _ { 0 }$ and $\mathcal { P } ^ { \prime }  ( \mathcal { P } \setminus \mathcal { D } _ { 0 } ) \cup \pi ( \mathcal { P } \cap \mathcal { D } _ { 0 } )$   
5 if $\vert \mathcal { A } \vert > \vert \mathcal { D } \vert$ then   
6 foreach $i \in \mathcal { A } \backslash \mathcal { A } _ { 0 }$ in a fixed order do   
7 if Bernoull $\lfloor ( P / ( | \widehat { \mathcal { M } } | + 1 ) ) = 1$ then   
8 Sample $\dot { j } \sim \operatorname { U n i f o r m } ( \mathcal { P } ^ { \prime } )$ and set   
$\mathcal { P } ^ { \prime } \overset { \cdot } {  } \overset { \cdot } { ( } \mathcal { P } ^ { \prime } \setminus \{ j \} ) \cup \{ i \} ;$   
9 ${ \widehat { \mathcal { M } } } \gets { \widehat { \mathcal { M } } } \cup \{ i \}$   
10 else   
11 $\mathcal { P } ^ { \prime }  \mathcal { P } ^ { \prime } \setminus ( \mathcal { D } \setminus \mathcal { D } _ { 0 } ) ;$   
12 Sample $\boldsymbol { B }$ uniformly among size- $\cdot ( P - | \mathcal { P } ^ { \prime } | )$ subsets of   
$\mathcal { M } ^ { \prime } \backslash \mathcal { P } ^ { \prime } ;$ set ${ \mathcal { P } } ^ { \prime } \ i \not  { \mathcal { P } } ^ { \prime } \cup { \bf { \bar { \boldsymbol { B } } } } ;$   
13 Set $S _ { - }  \mathcal { P } \setminus \mathcal { P } ^ { \prime }$ and $S _ { + }  \mathcal { P } ^ { \prime } \setminus \mathcal { P } ;$   
14 if $S _ { - } \cup S _ { + } \stackrel { } { = } \infty$ then   
15 Retrieve original contributions for S from the buffer or   
connected clients; obtain contributions for $S _ { + }$ using the   
fixed F-SOSA training configuration;   
16 From the unchanged original statistics, compute   
$\bar { W } ^ { \prime }  \bar { W } _ { \mathcal { P } } + \frac { 1 } { P } ( \sum _ { i \in S _ { + } } \widetilde { W } _ { i } - \sum _ { i \in S _ { - } } \widetilde { W } _ { i } )$   
$\pmb { F } ^ { \prime }  \pmb { F } _ { \mathcal { P } } + \sum \pmb { F } _ { i } - \sum \pmb { F } _ { i } ,$   
$i \overline { { \in S _ { + } } } \qquad i \overline { { \in S _ { - } } }$   
$\boldsymbol { b } ^ { \prime }  \boldsymbol { b } _ { \mathcal { P } } + \sum \boldsymbol { F } _ { i } \widetilde { \boldsymbol { W } } _ { i } - \sum \boldsymbol { F } _ { i } \widetilde { \boldsymbol { W } } _ { i } .$   
i∈S<sub>+</sub> i∈S<sub>−</sub>   
Solve $( F ^ { \prime } ) ^ { 2 } \Lambda = 2 ( F ^ { \prime } \bar { W } ^ { \prime } - b ^ { \prime } )$ and set   
$\begin{array} { r } { \bar { W } ^ { \prime \mathrm { F } } \gets \bar { W } ^ { \prime } - \frac 1 2 \bar { F ^ { \prime } } \pmb { \Lambda } ; } \end{array}$   
17 else   
18 Reuse the original statistics and fused model;   
19 Commit $\mathcal { M }  \mathcal { M } ^ { \prime } , \mathcal { P }  \mathcal { P } ^ { \prime } ,$ , the statistics and model;   
discard temporary per-client buffers;   
Output: Updated model, client set, sampled subset, and   
aggregate statistics

## Theorem 1. F-SOSA is ρ-TV-stable for any $\rho \in ( 0 , 1 )$

Proof. Let M be original device set and $\mathcal { M } ^ { \prime }$ be the update device set such that $\vert \mathcal { M } \oplus \mathcal { M } ^ { \prime } \vert = 1$ . We first consider $\mathcal { M }$ contains $M - 1$ devices, where w.l.o.g. we assume device M leaves, $\mathrm { i . e . }$ $\mathcal { M } = \{ 1 , 2 , \cdots , M \}$ and $\mathcal { M } ^ { \prime } = \{ 1 , 2 , \cdots , M - 1 \}$ . Let the sigma-algebra on these sets be the power sets of M and $\mathcal { M } ^ { \prime }$ respectively, denoted by Pow(M) and Pow(M<sup>′</sup>) respectively. Let $\mu _ { ( P , M ) }$ be the sub-sampling probability measure on $M$ devices in $\mathcal { M } ,$ i.e., sub-samples a subset $\mathcal { P }$ of size $P$ out of M elements in $\mathcal { M }$ uniformly at random. Similarly we can define $ { \mu } _ { ( P , M - 1 ) }$ for $\mathcal { M } ^ { \prime }$ . We first extend the sigma-algebra for the probability spaces so that the random variables $\mu _ { ( P , M ) }$ and $ { \mu } _ { ( P , M - 1 ) }$ are defined on a common probability space. For this, we will just add an event where the device M can be sampled under $ { \mu } _ { ( P , M - 1 ) }$ with probability 0. We define

$\mu _ { ( P , M ) } ^ { \prime }$ as follows: for any set $\mathcal { P } \in \mathrm { P o w } ( \mathcal { M } ) , \mu _ { ( P , M ) } ^ { \prime } ( \mathcal { P } ) : =$ $\int \mu _ { ( P , M - 1 ) } ( \mathcal { P } )$ , if M $\notin \mathcal { P }$   
. Note that, for fixed initialization 0, otherwise   
$\grave { W } ^ { ( 0 ) }$ and other parameters, Algorithm $\boldsymbol { \mathcal { A } } ( \boldsymbol { \mathcal { M } } )$ and $\mathcal { A } ( \mathcal { M } ^ { \prime } )$ is the same deterministic map from $\begin{array} { r } { \mathcal { P } \in [ M ] ^ { P } } \end{array}$ to W. They only differ because of different measure on the input space. Hence TV distance between $\mathcal { A } ( \mathcal { M } )$ and $\mathcal { A } ( \mathcal { M } ^ { \prime } )$ is just the TV distance between the push-forward measures $\mathcal { A } ( \mathcal { M } ) \# \mu _ { ( P , M ) }$ and $\mathcal { A } ( \mathcal { M } ^ { \prime } ) \# \mu _ { ( P , M ) } ^ { \prime } .$ , which by using the fact that $\mathcal { A } ( \mathcal { M } ) \equiv \mathcal { A } ( \mathcal { M } ^ { \prime } )$ and data processing inequality is at most the TV distance between $\mu _ { ( P , M ) }$ and $\mu _ { ( P , M ) } ^ { \prime }$ . Therefore, $\mathrm { T V } ( \boldsymbol { A } ( \mathcal { M } ) , \boldsymbol { A } ( \mathcal { M } ^ { \prime } ) ) \le \mathrm { T V } ( \mu _ { ( P , M ) } , \mu _ { ( P , M ) } ^ { \prime } )$   
$\leq \mu _ { ( P , M ) } ( S$ containing $M ) = \binom { M - 1 } { P - 1 } / \binom { M } { P } = P / M .$ A similar argument works when $\mathcal { M } ^ { \prime } = \{ 1 , 2 , \cdots , M , M +$ 1}, yielding a TV bound of $\begin{array} { l l l } { { { \frac { P } { M + 1 } } } } & { { \le } } & { { { \frac { \textstyle P } { M } } . } } \end{array}$ Taking a uniform bound over all neighbouring set ${ \bar { \mathcal { M } } } ^ { \prime } .$ , we get that $\sin ( \Theta _ { | \mathcal { M } \oplus \mathcal { M ^ { \prime } } | = 1 } \operatorname { T V } ( \mathcal { A } ( \mathcal { M } ) , \mathcal { A } ( \mathcal { M ^ { \prime } } ) ) \ \leq \ P / M$ . Since in F-SOSA, $\mathrm { ~ \ P ~ }$ is set to $\rho M _ { ☉ }$ , thus $\mathrm { T V } ( \boldsymbol { A } ( \boldsymbol { \mathcal { M } } ) , \boldsymbol { A } ( \boldsymbol { \mathcal { M } } ^ { \prime } ) ) \leq \rho .$ □

## B. Learning Utility of F-SOSA

We analyze the utility guarantee of $\bar { \boldsymbol { W } } _ { \mathcal { P } } ^ { \mathrm { F } }$ obtained by F-SOSA under the following standard data assumptions:

Assumption 2 (Data Normalization). For any $( \boldsymbol { x } , \boldsymbol { y } ) \in \mathcal { D } _ { \mathrm { \Omega } }$ $| { \boldsymbol { x } } | _ { 2 } = 1$ and $| y | \le C$ for some constant $C > 0 .$

$$
( { \pmb x } ^ { \prime } , y ^ { \prime } )
$$

$$
| { \pmb x } - { \pmb x } ^ { \prime } | _ { 2 } \geq \phi > 0 .
$$

$$
( { \pmb x } , y )
$$

Assumption 4 (Data Non-Degeneracy). The minimum eigenvalue of the Gram matrix $\mathbf { H } ^ { \infty } \in \mathbb { R } ^ { N \times N }$ defined by $( \mathbf { H } ^ { \infty } ) _ { k , l } =$ $\mathbb { E } _ { \boldsymbol { w } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) } [ \pmb { x } _ { k } ^ { \top } \pmb { x } _ { l } \mathbb { 1 } \{ \pmb { w } ^ { \top } \pmb { x } _ { k } \ge 0 \} \mathbb { 1 } \{ \pmb { w } ^ { \top } \pmb { x } _ { l } \ge 0 \} ]$ , is bounded below: $\lambda _ { \operatorname* { m i n } } ( \mathbf { H } ^ { \infty } ) = \lambda _ { 0 } > 0 .$

No bounded heterogeneity assumption is made, so our results hold under arbitrary non-i.i.d. client data. We consider a two-layer ReLU neural network (NN) where the first-layer consists of m trainable weights $\{ { w _ { r } } \} _ { r = 1 } ^ { m }$ and the second-layer weights $\pmb { a } = \{ a _ { r } \} _ { r = 1 } ^ { m }$ are drawn i.i.d. from $\{ + 1 , - 1 \}$ at random and then fixed [8], [45]. Let ${ \pmb W } = \mathrm { v e c } ( { \pmb w } _ { 1 } , { \pmb w } _ { 2 } , \dots , { \pmb w } _ { m } )$ The NN is modeled as

$$
f _ { a } ( { \cal W } , x ) = \frac { 1 } { \sqrt { m } } \sum _ { r = 1 } ^ { m } a _ { r } \gamma ( { \pmb x } ^ { \top } { \pmb w } _ { r } ) ,\tag{14}
$$

where $\gamma ( x ) = \operatorname* { m a x } \{ x , 0 \}$ is the ReLU activation function. This model can be equivalently written in a linearized form:

$$
f _ { a } ( W , \pmb { x } ) = \phi _ { a } ( W , \pmb { x } ) ^ { \top } W ,\tag{15}
$$

where

$$
\phi _ { a } ( W , x ) : = \frac { 1 } { \sqrt { m } } \left[ \begin{array} { c } { a _ { 1 } x \mathbb { 1 } \{ x ^ { \top } w _ { 1 } \geq 0 \} } \\ { a _ { 2 } x \mathbb { 1 } \{ x ^ { \top } w _ { 2 } \geq 0 \} } \\ { \vdots } \\ { a _ { m } x \mathbb { 1 } \{ x ^ { \top } w _ { m } \geq 0 \} } \end{array} \right] \in \mathbb { R } ^ { m p } .\tag{16}
$$

The neural-network objective is non-convex. We analyze it using the two-layer ReLU model, a standard canonical setting for NTK-style analysis of over-parameterized neural networks, where the training dynamics can be characterized through activation stability and the associated Gram matrix. In this regime, gradient descent remains within a controlled neighborhood of the random initialization, the relevant Gram matrices remain well conditioned, and the empirical fitting error decreases geometrically with the number of local iterations.

We first record a basic bound on the feature representation that will be used repeatedly in the subsequent analysis. It follows directly from the definition of $\phi _ { a } ( W , x )$ , since $a _ { r } \in \{ - 1 , + 1 \}$ and each activation indicator is at most one.

Lemma 1. For $\forall W \in \mathbb { R } ^ { d }$ and $\| x \| _ { 2 } \leq 1 , \| \phi _ { a } ( W , x ) \| _ { 2 } \leq 1$

We initialize ${ \pmb w } _ { r } ^ { ( 0 ) } \overset { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( { \bf 0 } , \sigma ^ { 2 } { \bf I } )$ . To quantify activation pattern stability, given a real $R \geq 0 .$ , define event $\mathcal { E } _ { i j r } ( R ) : =$ $\{ \exists w : \| w - w _ { r } ^ { ( 0 ) } \| _ { 2 } \leq R , \mathbb { 1 } \{ { \pmb x } _ { i j } ^ { \top } { \pmb w } _ { r } ^ { ( 0 ) } \geq 0 \} \neq \mathbb { 1 } \{ { \pmb x } _ { i j } ^ { \top } { \pmb w } \geq 0 \} \}$ We next characterize the stability of ReLU activation patterns around the random initialization.

Lemma 2 ( [8, Lemma 3]). The random initialization of ${ \pmb w } _ { r } ^ { ( 0 ) } \overset { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( { \bf 0 } , \sigma ^ { 2 } { \bf I } )$ ensures that $\mathbb { P } ( \mathcal { E } _ { i j r } ( R ) ) \le 2 R / ( \sqrt { 2 \pi } \sigma )$

We then invoke a standard over-parameterized training result to control both the local fitting error and the distance traveled by each local model.

Lemma 3 ( [8, Lemma 2]). Let $\begin{array} { r } { m = \Omega ( \frac { N ^ { 6 } } { \lambda _ { 0 } ^ { 4 } \sigma ^ { 2 } \delta ^ { 3 } } ) } \end{array}$ and $\eta =$ $\begin{array} { r } { O \big ( \frac { \lambda _ { 0 } } { N ^ { 2 } } \big ) } \end{array}$ . With probability at least $1 - \delta$ over the initialization of $\ddot { W } ^ { ( 0 ) }$ and a, we have simultaneously for all $i \in [ M ]$

1) $\begin{array} { r } { \sum _ { j = 1 } ^ { n } ( y _ { i j } - f _ { a } ( W ^ { ( 0 ) } , \pmb { x } _ { i j } ) ) ^ { 2 } = O ( \frac { N } { \delta } ) ; } \end{array}$

$$
\begin{array} { r } { \sum _ { j = 1 } ^ { \tilde { n } } ( y _ { i j } - f _ { a } ( \tilde { W } _ { i } , x _ { i j } ) ) ^ { 2 } \leq ( 1 - \frac { \eta \lambda _ { 0 } } { 2 } ) ^ { K } \cdot O ( \frac { N } { \delta } ) ; } \end{array}
$$

3) For $\forall r \in [ m ] , \| \tilde { \pmb { w } } _ { i , r } - \pmb { w } _ { r } ^ { ( 0 ) } \| _ { 2 } \leq R _ { 0 }$ , where $R _ { 0 } =$ $\begin{array} { r } { \frac { 4 } { \lambda _ { 0 } } \sqrt { \frac { N } { m } } ( 1 - ( 1 - \frac { \eta \lambda _ { 0 } } { 4 } ) ^ { K } ) \sqrt { \sum _ { j = 1 } ^ { n } ( y _ { i j } - f _ { a } ( W ^ { ( 0 ) } , { \pmb x } _ { i j } ) ) ^ { 2 } } . } \end{array}$

Combining 1) and 3), we have $\begin{array} { r } { R _ { 0 } \leq O ( \frac { N } { \lambda _ { 0 } \sqrt { \delta m } } ) } \end{array}$

Building on these local training properties, we next bound the deviations of the trained local models and their subsampled average from the common initialization. Proofs of all auxiliary lemmas established in this paper are deferred to Appendix A.

Lemma 4. Let $\begin{array} { r } { m ~ = ~ \frac { \mathrm { p o l y } ( N , \delta ^ { - 1 } , \lambda _ { 0 } ^ { - 1 } ) } { \sigma ^ { 2 } } , ~ \eta ~ = ~ O ( \frac { \lambda _ { 0 } } { N ^ { 2 } } ) } \end{array}$ , and $\begin{array} { r } { \sigma = O ( \frac { \lambda _ { 0 } \delta } { N } ) } \end{array}$ . With probability at least $1 - \delta$ over random initialization:

1) for all $i \in [ M ] , \| \tilde { \pmb { W } } _ { i } - \pmb { W } ^ { ( 0 ) } \| _ { 2 } \le \sqrt { \pmb { y } _ { i } ^ { \top } ( \pmb { H } _ { i } ^ { \infty } ) ^ { - 1 } \pmb { y } _ { i } } +$ $\begin{array} { r } { O ( \frac { N \sigma } { \lambda _ { 0 } \delta } ) + \frac { \mathrm { p o l y } ( N , \lambda _ { 0 } ^ { - 1 } , \delta ^ { - 1 } ) } { m ^ { 1 / 4 } \sigma ^ { 1 / 2 } } ; } \end{array}$

$$
\begin{array} { l c l } { \| \bar { W } _ { \mathcal { P } } ~ - ~ W ^ { ( 0 ) } \| _ { 2 } } & { \leq } & { \sqrt { \frac { 2 y ^ { \top } ( { \bf H } ^ { \infty } ) ^ { - 1 } y } { \rho } } ~ + ~ { \cal O } ( { \frac { N \sigma } { \lambda _ { 0 } \delta } } ) ~ + } \\ { \frac { \mathrm { p o l y } ( N , \lambda _ { 0 } ^ { - 1 } , \delta ^ { - 1 } ) } { m ^ { 1 / 4 } \sigma ^ { 1 / 2 } } . } \end{array}
$$

These parameter-space bounds further allow us to control the prediction discrepancy between the averaged model and the collection of local models. Let ${ \pmb y } = \mathrm { v e c } ( { \pmb y } _ { 1 } , { \pmb y } _ { 2 } , \cdot \cdot \cdot , { \pmb y } _ { M } ) \in$ $\mathbb { R } ^ { N }$ be the true label vector and $\tilde { \pmb { y } } = \mathrm { v e c } ( \tilde { \pmb { y } } _ { 1 } , \tilde { \pmb { y } } _ { 2 } , \cdot \cdot \cdot , \tilde { \pmb { y } } _ { M } ) =$ $[ f _ { a } ( \tilde { W } _ { 1 } , x _ { 1 1 } ) , ~ f _ { a } ( \tilde { W } _ { 1 } , x _ { 1 2 } ) , \cdot \cdot \cdot , f _ { a } ( \tilde { W } _ { M } , x _ { M n } ^ { - } ) ] ^ { \top } ~ \in ~ \mathbb { R } ^ { N }$ be the predictions of trained local models. We define the proxy output $\begin{array} { r } { \tilde { \boldsymbol { f } } ( \boldsymbol { W } , \boldsymbol { x } _ { i j } ) = \phi _ { a } ( \tilde { \boldsymbol { W } } _ { i } , \boldsymbol { x } _ { i j } ) ^ { \top } \boldsymbol { W } } \end{array}$ for $\forall W \in \mathbb { R } ^ { d }$ and $\mathbf { \nabla } x _ { i j } .$

Lemma 5. Let $m = \mathrm { p o l y } ( N , \delta ^ { - 1 } , \lambda _ { 0 } ^ { - 1 } ) / \sigma ^ { 2 } , \eta = { \cal O } ( \lambda _ { 0 } / N ^ { 2 } )$ and $\sigma = O ( \lambda _ { 0 } \delta / N )$ . With probability at least $1 - \delta$ over the random initialization, we have

$$
\left\| \widetilde f ( \bar { W } _ { \mathcal { P } } , \pmb { x } _ { i j } ) - \tilde { \pmb { y } } \right\| _ { 2 } ^ { 2 } \leq O \left( \frac { N } { \rho } \pmb { y } ^ { \top } ( \mathbf { H } ^ { \infty } ) ^ { - 1 } \pmb { y } \right) .
$$

We define four matrices associated with the initial model ${ \pmb W } ^ { ( 0 ) }$ and participant set $\mathcal { P } ~ = ~ \{ i _ { 1 } , \ldots , i _ { P } \} \colon \mathbf { A } _ { \mathcal { P } } ^ { ( 0 ) } , \tilde { \mathbf { A } } _ { \mathcal { P } } ~ \in$ $\mathbb { R } ^ { ( P n ) \times ( m p ) }$ and $\bar { \mathbf { H } } _ { \mathcal { P } } ^ { ( 0 ) } , \tilde { \mathbf { H } } _ { \mathcal { P } } \in \mathbb { R } ^ { ( P n ) \times ( P n ) } ;$

$$
\begin{array} { r l r } & { \mathbf { A } _ { \mathcal { P } } ^ { ( 0 ) } : = \left[ \begin{array} { c } { \phi _ { a } ( { \cal W } ^ { ( 0 ) } , { \pmb { x } } _ { i _ { 1 } 1 } ) ^ { \top } } \\ { \phi _ { a } ( { \cal W } ^ { ( 0 ) } , { \pmb { x } } _ { i _ { 1 } 2 } ) ^ { \top } } \\ { \vdots } \\ { \phi _ { a } ( { \cal W } ^ { ( 0 ) } , { \pmb { x } } _ { i _ { P } n } ) ^ { \top } } \end{array} \right] , } & { \tilde { \mathbf { A } } _ { \mathcal { P } } : = \left[ \begin{array} { c } { \phi _ { a } ( \tilde { W } _ { 1 } , { \pmb { x } } _ { i _ { 1 } 1 } ) ^ { \top } } \\ { \phi _ { a } ( \tilde { W } _ { 1 } , { \pmb { x } } _ { i _ { 1 } 2 } ) ^ { \top } } \\ { \vdots } \\ { \phi _ { a } ( \tilde { W } _ { M } , { \pmb { x } } _ { i _ { P } n } ) ^ { \top } } \end{array} \right] } \\ & { \mathbf { H } _ { \mathcal { P } } ^ { ( 0 ) } : = \mathbf { A } _ { \mathcal { P } } ^ { ( 0 ) } \mathbf { A } _ { \mathcal { P } } ^ { ( 0 ) \top } , } & { \tilde { \mathbf { H } } _ { \mathcal { P } } : = \tilde { \mathbf { A } } _ { \mathcal { P } } \tilde { \mathbf { A } } _ { \mathcal { P } } ^ { \top } . } \end{array}
$$

We next study the conditioning of the Gram matrices associated with the sampled clients, which is needed to control the projection step.

Lemma 6. Let $\mathbf { H } _ { \mathcal { D } } ^ { \infty } \in \mathbb { R } ^ { ( P n ) \times ( P n ) }$ be the principal submatrix of $\mathbf { H } ^ { \infty }$ formed by P. If $\begin{array} { r } { m = \Omega ( \frac { ( \rho N ) ^ { 2 } \log ( \dot { \rho } N / \delta ) } { \lambda _ { 0 } ^ { 2 } } ) } \end{array}$ , with probability at least $1 - \delta , \| \mathbf { H } _ { \mathcal { P } } ^ { ( 0 ) } - \mathbf { H } _ { \mathcal { P } } ^ { \infty } \| \leq \lambda _ { 0 } / 4 , \lambda _ { \operatorname* { m i n } } ( \mathbf { H } _ { \mathcal { P } } ^ { ( 0 ) } ) \geq 3 \lambda _ { 0 } / 4$ Lemma 7. If $\begin{array} { r } { m = \Omega \big ( \frac { N ^ { 6 } } { \lambda _ { \mathrm { { n } } } ^ { 4 } \sigma ^ { 2 } \delta ^ { 3 } } \big ) } \end{array}$ , then with probability at least $1 - \delta ,$ , we have $\lambda _ { \operatorname* { m i n } } ( \tilde { \mathbf { H } } _ { \mathcal { P } } ) \geq \lambda _ { 0 } / 2$ and $\lambda _ { \operatorname* { m a x } } ( \tilde { \mathbf { H } } _ { \mathcal { P } } ) \leq \rho N$

With these spectral bounds in place, we can quantify the additional displacement introduced by Fisher-based projection. Lemma 8. Let $\begin{array} { r } { m \ge \frac { \sigma ^ { 2 } } { \mathrm { p o l y } ( \rho , N , \delta ^ { - 1 } , \lambda _ { 0 } ^ { - 1 } ) } , \sigma = O ( \frac { \lambda _ { 0 } \delta } { N } ) } \end{array}$ . With probability at least $1 - \delta$ over the initialization, we have

$$
\left\| \bar { W } _ { \mathcal { P } } ^ { \mathrm { F } } - \bar { W } _ { \mathcal { P } } \right\| _ { 2 } = O \left( \sqrt { \frac { N } { \rho \lambda _ { 0 } } \pmb { y } ^ { \top } ( \mathbf { H } ^ { \infty } ) ^ { - 1 } \pmb { y } } \right) .
$$

Lemma 9. Let $m = \mathrm { p o l y } ( N , \delta ^ { - 1 } , \lambda _ { 0 } ^ { - 1 } ) / \sigma ^ { 2 } , \eta = { \cal O } ( \lambda _ { 0 } / N ^ { 2 } )$ and $\sigma = O ( \lambda _ { 0 } \delta / N )$ . With probability at least $1 - \delta$ over the random initialization, we have

$$
\| \bar { \pmb { w } } _ { \mathcal { P } , r } ^ { \mathrm { F } } - \bar { \pmb { w } } _ { \mathcal { P } , r } \| _ { 2 } \leq R _ { 1 } \leq O \left( \frac { N } { \sqrt { m } \lambda _ { 0 } } \sqrt { { \pmb { y } } ^ { \top } ( { \bf H } ^ { \infty } ) ^ { - 1 } { \pmb { y } } } \right) .
$$

The resulting neuron-wise displacement bound allows us to control activation-pattern mismatches between the fused and local models. Given any $( i , j )$ pair, we define $S _ { i j } ^ { \mathrm { F } } : = \{ r \in$ $[ m ] : 1 \{ \pmb { x } _ { i j } ^ { \top } \pmb { \bar { w } } _ { \mathcal { P } , r } ^ { \mathrm { F } } \geq 0 \} = \mathbb { 1 } \{ \pmb { \bar { x } } _ { i j } ^ { \top } \pmb { \tilde { w } } _ { i , r } \geq 0 \} \}$ as the set of neurons with matching activation signs between the global and local models. Let $S _ { i j } ^ { \mathrm { F \perp } } : = [ m ] \setminus S _ { i j } ^ { \mathrm { F } }$ denote the mismatch set.

Lemma 10. With probability at least $1 - \delta$ over the initialization, $\begin{array} { r } { | S _ { i j } ^ { \mathrm { F } \perp } | = O ( \frac { \dot { m ^ { } } N ( R _ { 0 } + R _ { 1 } \ddot { ) } } { \delta \sigma } ) } \end{array}$ for all $i , j$ simultaneously.

Note that the mismatch size $| S _ { i j } ^ { \mathrm { F \perp } } |$ is independent of the client subsampling rate $\rho ,$ since $R _ { 0 } ^ { \bar { } }$ bounds the deviation of each local trained model $\tilde { \pmb { w } } _ { i , r }$ from initial ${ \pmb w } _ { r } ^ { ( 0 ) }$ uniformly across $i \in [ M ]$ , regardless of which clients are sampled.

We are now ready to establish the empirical fitting guarantee of F-SOSA. Lemma 3 controls the local fitting error, while Lemmas 8–10 bound the additional error induced by Fisher-based fusion through the resulting activation-pattern mismatches. Combining these two components yields the following training-error guarantee.

Theorem 2 (Training Error for Fitting the Data). Under Assumptions $2 - 4$ , let $m = \Omega ( N ^ { 6 } / ( \lambda _ { 0 } ^ { 4 } \bar { \sigma } ^ { 2 } \delta ^ { 3 } ) ) , \eta = \lambda _ { 0 } / N ^ { 2 }$ ${ \pmb w } _ { r } ^ { ( 0 ) } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } { \bf I } ) , a _ { r } \sim$ Uniform $( \{ - 1 , + 1 \} )$ , then with probability at least $1 - \delta ,$ we have

$$
L _ { \mathcal { D } } ^ { f , \ell } ( \bar { W } _ { \mathcal { P } } ^ { \mathrm { F } } ) \leq O ( ( 1 - \frac { \eta \lambda _ { 0 } } { 2 } ) ^ { K } \cdot \frac { N } { \delta } + \frac { N ^ { 6 } ( \pmb { y } ^ { \top } ( \mathbf { H } ^ { \infty } ) ^ { - 1 } \pmb { y } ) ^ { 2 } } { m \delta ^ { 4 } \sigma ^ { 2 } \lambda _ { 0 } ^ { 4 } } ) .
$$

Proof. We decompose the loss $L _ { \mathcal { D } } ^ { f , \ell } ( \bar { \boldsymbol { W } } _ { \mathcal { P } } ^ { \mathrm { F } } )$ as follows

$$
\begin{array} { l } { { \displaystyle { \cal L } ( \bar { W } _ { \mathcal { P } } ^ { \mathrm { F } } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { M } \sum _ { j = 1 } ^ { n } ( f ( \bar { W } _ { \mathcal { P } } ^ { \mathrm { F } } , x _ { i j } ) - y _ { i j } ) ^ { 2 } } } \\ { { \displaystyle = \frac { 1 } { N } \sum _ { i = 1 } ^ { M } \sum _ { j = 1 } ^ { n } ( \phi ( \bar { W } _ { \mathcal { P } } ^ { \mathrm { F } } , x _ { i j } ) ^ { \top } \bar { W } _ { \mathcal { P } } ^ { \mathrm { F } } - \tilde { y } _ { i j } + \tilde { y } _ { i j } - y _ { i j } ) ^ { 2 } } } \\ { { \displaystyle \le \frac { 2 } { N } \sum _ { i = 1 } ^ { M } \sum _ { j = 1 } ^ { n } ( \phi ( \bar { W } _ { \mathcal { P } } ^ { \mathrm { F } } , x _ { i j } ) ^ { \top } \bar { W } _ { \mathcal { P } } ^ { \mathrm { F } } - \tilde { y } _ { i j } ) ^ { 2 } } + \underbrace { \frac { 2 } { N } \sum _ { i = 1 } ^ { M } ( \tilde { y } _ { i j } - y _ { i j } ) ^ { 2 } } _ { T _ { 2 } } . }  \end{array}
$$

$T _ { 2 }$ measures how well the local models fit their local data and can be bounded via Lemma 3 as $\begin{array} { r } { T _ { 2 } = O \left( \left( 1 - \frac { \eta \lambda _ { 0 } } { 2 } \right) ^ { K } \frac { N } { \delta } \right) } \end{array}$ By taking $t \to \infty$ in (21), $\begin{array} { r } { \tilde { y } _ { i j } = \phi _ { a } ( \tilde { W } _ { i } , \tilde { x _ { i j } } ) ^ { \top } \bar { W } _ { \mathcal P } ^ { \mathrm { F } } } \end{array}$ , we have

$$
\begin{array} { r l } & { T ^ { * } - \frac { \lambda ^ { 2 } } { \mu _ { 0 } } \frac { \sum _ { j = 0 } ^ { N } \sum _ { i = 0 } ^ { N } \mu _ { j } ^ { ( k ) } } { \mu _ { 0 } } \sum _ { j = 0 } ^ { N } \mu _ { i } ^ { ( k ) } \mu _ { j } ^ { ( k ) } - \mu _ { 0 } ^ { ( k ) } \mu _ { i } ^ { ( k ) } \mu _ { j } ^ { ( k ) } } \\ & { = \frac { \lambda ^ { 2 } } { \mu _ { 0 } } \sum _ { j = 0 } ^ { N } \mu _ { i } ^ { ( k ) } \mu _ { j } ^ { ( k ) } } \\ & { \leq \sum _ { j = 0 } ^ { N } \mu _ { i } ^ { ( k ) } \mu _ { j } ^ { ( k ) } \mu _ { j } ^ { ( k ) } \mu _ { j } ^ { ( k ) } \mu _ { j } ^ { ( k ) } - \mu _ { 0 } ^ { ( k ) } \mu _ { i } ^ { ( k ) } \mu _ { j } ^ { ( k ) } } \\ & { \leq \frac { \lambda ^ { 2 } } { \mu _ { 0 } } \frac { \mu _ { i } ^ { ( k ) } - \mu _ { 0 } ^ { ( k ) } } { \mu _ { 0 } } \bigg \{ \mu _ { 0 } ^ { ( k ) } \mu _ { j } ^ { ( k ) } \mu _ { j } ^ { ( k ) } \mu _ { j } ^ { ( k ) } \mu _ { j } ^ { ( k ) } } \\ & { \leq \mu _ { 0 } ^ { ( k ) } \mu _ { j } ^ { ( k ) } \mu _ { j } ^ { ( k ) } \mu _ { j } ^ { ( k ) } \mu _ { j } ^ { ( k ) } \mu _ { j } ^ { ( k ) } \bigg \} } \\ &  = \frac { \lambda ^ { 2 } } { \mu _ { 0 } } \frac { \mu _ { i } ^ { ( k ) } - \mu _ { 0 } ^ { ( k ) } } { \mu _ { 0 } } \frac { \mu _ { i } ^ { ( k ) } } { \mu _ { 0 } } \sum _ { j = 0 } ^ { N } \mu _ { i } ^ { ( k ) } \mu _  \end{array}
$$

where the second-to-the-last inequality uses upper bounds on $R _ { 0 }$ and $R _ { 1 }$ from Lemma 3 and Lemma 9. □

Corollary 1. Under the setting of Theorem 2, given a target training error ϵ, if we set $m = \Omega ( \mathrm { p o l y } ( N , \delta ^ { - 1 } , \lambda _ { 0 } ^ { - 1 } ) / \sigma ^ { 2 } ) , \eta =$ $\lambda _ { 0 } / N ^ { 2 } , \sigma = O ( \lambda _ { 0 } \delta / N )$ , and $K = \Omega ( ( 1 / \eta \lambda _ { 0 } \bar { ) } \log ( N / \delta \epsilon ) )$ then with probability at least $1 - \delta .$ , we have $L _ { \mathcal { D } } ^ { f , \ell } ( \bar { \boldsymbol { W } } _ { \mathcal { P } } ^ { \mathrm { F } } ) \le \epsilon .$

Corollary 1 shows that the empirical risk can be made vanishingly small in the over-parameterized regime by choosing the network width $m$ and the number of local iterations $K$ sufficiently large.

Having established the empirical fitting guarantee, we next study generalization to unseen data. For this purpose, we adopt a two-level data distribution framework [46]. Let $\mathcal { Z } : = \mathcal { X } \times$ Y denote the data space, D the set of distributions over ${ \mathcal { Z } } ,$ and $\tau$ a meta-distribution over D. Each client i samples a distribution $D _ { i } \sim \tau$ and obtains n i.i.d. samples $\mathcal { D } _ { i } \sim D _ { i } ^ { n }$ Let $\textstyle { \mathcal { D } } = \bigcup _ { i = 1 } ^ { M } { \mathcal { D } } _ { i }$ be the union of local datasets. We consider the squared loss $\begin{array} { r } { \ell ( z , y ) = \frac { 1 } { 2 } ( z - y ) ^ { 2 } } \end{array}$ . The empirical risk over training data is

$$
L _ { \mathcal { D } } ^ { f , \ell } ( W ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \frac { 1 } { 2 } ( f _ { a } ( W , x _ { i j } ) - y _ { i j } ) ^ { 2 } .
$$

The population risk is defined as

$$
L _ { \mathcal { T } } ^ { f , \ell } ( W ) = \mathbb { E } _ { D \sim \mathcal { T } } \mathbb { E } _ { z \sim D } [ \frac { 1 } { 2 } ( f _ { a } ( W , x ) - y ) ^ { 2 } ] .
$$

We also define the semi-population risk:

$$
L _ { S } ^ { f , \ell } ( W ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } _ { z \sim D _ { i } } [ \frac { 1 } { 2 } ( f _ { a } ( W , x ) - y ) ^ { 2 } ] .
$$

To extend the empirical analysis to unseen data, we adopt the following federated Rademacher complexity.

Definition 3 (Federated Rademacher complexity). Let $\mathcal { F }$ be a class of functions $g : \mathcal { X }  \mathbb { R }$ . For M clients with n samples each, the federated Rademacher complexity is

$$
\mathcal { R } _ { n } ^ { M } ( \mathcal { F } ) = \mathbb { E } _ { \{ \iota _ { i j } \} } \big [ \operatorname* { s u p } _ { g \in \mathcal { F } } \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \iota _ { i j } \cdot g ( \pmb { x } _ { i j } ) \big ] ,
$$

where $\{ \iota _ { i j } \}$ are i.i.d. Rademacher variables.

When $M = 1$ , this reduces to the classical Rademacher complexity $\begin{array} { r } { \mathbb { E } _ { \{ \iota _ { j } \} } [ \operatorname* { s u p } _ { g \in \mathcal { F } } \frac 1 n \sum _ { j = 1 } ^ { n } \iota _ { j } g ( \pmb { x } _ { j } ) ] } \end{array}$ ]. We define a neural network class with weight constraints relative to initialization: $\mathcal { F } _ { W ^ { ( 0 ) } , a , R , B } : = \{ f _ { a } ( W , \cdot ) : \| w _ { r } - w _ { r } ^ { ( 0 ) } \| \leq R , \forall r \in$ $[ m ] , \| W - W ^ { ( 0 ) } \| _ { 2 } \leq B \}$ . The following existing result bounds the Rademacher complexity of the network class determined by the parameter neighborhood above.

Lemma 11 ( [8, Lemma 8]). Given $R > 0 ,$ with probability at least $1 - \delta$ over the initialization, uniformly over all $B > 0 ,$

$$
\begin{array} { r l } & { \mathcal { R } _ { n } ^ { M } ( \mathcal { F } _ { W ^ { ( 0 ) } , a , R , B } ) } \\ & { \leq \displaystyle \frac { B } { \sqrt { 2 N } } ( 1 + ( \frac { 2 \log ( 2 / \delta ) } { m } ) ^ { 1 / 4 } ) + \frac { 2 R ^ { 2 } \sqrt { m } } { \sigma } + R \sqrt { 2 \log ( 2 / \delta ) } . } \end{array}
$$

We next recall the empirical-to-semi-population generalization bound under the two-level federated data model.

Lemma 12 ( [46, Lemma 1]). Let the loss function $\ell ( f ( W , x ) , y )$ be bounded in [0, c]. Under the two-level distribution setting, with probability at least $1 - \delta$ , we have

$$
\operatorname* { s u p } _ { f \in \mathcal { F } } \{ L _ { S } ^ { f , \ell } - L _ { \mathcal { D } } ^ { f , \ell } \} \leq 2 \mathcal { R } _ { n } ^ { M } ( \mathcal { F } ) + 3 c \sqrt { \frac { \log ( 2 / \delta ) } { 2 N } } .
$$

The remaining step is to control the gap between the semi population and population risks.

Lemma 13. Let the loss $\ell ( f ( W , x ) , y )$ be bounded in $[ 0 , c ]$ With probability at least $1 - \delta .$

$$
\operatorname* { s u p } _ { f \in \mathcal { F } } \{ L _ { \mathcal { T } } ^ { f , \ell } - L _ { S } ^ { f , \ell } \} \leq 2 \mathcal { R } _ { n } ^ { M } ( \mathcal { F } ) + 2 c \sqrt { \frac { \log ( 2 / \delta ) } { 2 N } } + c \sqrt { \frac { \log ( 2 / \delta ) } { 2 M } } .
$$

Lemmas 4 and 8–9 control the fused model around the initialization, while Lemmas 11–13 bound the generalization gaps. Together with Theorem 2, these results yield the following population-risk bound.

Theorem 3 (Generalization Error for Unseen Data). Under Assumptions 2, 3 and 4, let $\begin{array} { r c l } { \sigma } & { = } & { O ( \lambda _ { 0 } \delta / N ) } \end{array}$ $\begin{array} { r l r l r l r l } { m } & { { } } & { = } & { } & { \Omega ( \mathrm { p o l y } ( N , \delta ^ { - 1 } , \lambda _ { 0 } ^ { - 1 } ) / \sigma ^ { 2 } ) , } & { \eta } & { { } } & { = } & { } & { \lambda _ { 0 } / N ^ { 2 } } \end{array}$ $\begin{array} { r l r } { { \pmb w } _ { r } ^ { ( 0 ) } } & { { } \sim } & { \mathcal { N } ( 0 , \sigma ^ { 2 } { \bf I } ) , ~ a _ { r } \sim ~ \mathrm { U n i f o r m } ( \{ - 1 , + 1 \} ) } \end{array}$ and $\begin{array} { r c l } { K } & { = } & { \Omega ( ( 1 / \eta \lambda _ { 0 } ) \log ( \rho N / ( \delta ( 1 + \rho ) \pmb { y } ^ { \top } ( \mathbf { H } ^ { \infty } ) ^ { - 1 } \pmb { y } ) ) ) } \end{array}$ , with probability at least $1 - \delta$ , we have

$$
L _ { 7 } ^ { f , \ell } ( \bar { W } _ { \mathcal { P } } ^ { \mathrm { F } } ) \leq O ( \frac { 1 } { \rho } \sqrt { \frac { y ^ { \top } ( \mathbf { H } ^ { \infty } ) ^ { - 1 } y } { \lambda _ { 0 } } } + \sqrt { \frac { \log \frac { 1 } { \delta } } { N } } + \sqrt { \frac { \log \frac { 1 } { \delta } } { M } } ) .
$$

Proof. We first bound $\| \bar { \mathbfcal { W } } _ { \mathcal { P } } ^ { \mathrm { F } } - \bar { \mathbfcal { W } } _ { \mathcal { P } } \| _ { 2 }$ via Lemmas 8 and Lemma 4, which gives

$$
\| \bar { \boldsymbol { W } } _ { \mathcal { P } } ^ { \mathrm { F } } - \bar { \boldsymbol { W } } _ { \mathcal { P } } \| _ { 2 } \le O ( \sqrt { N / ( \rho \lambda _ { 0 } ) } \cdot \sqrt { \pmb { y } ^ { \top } ( \mathbf { H } ^ { \infty } ) ^ { - 1 } \pmb { y } } ) .\tag{17}
$$

Similarly, we bound $\lVert \bar { \boldsymbol { w } } _ { \mathcal { P } , r } ^ { \mathrm { F } } - \bar { \boldsymbol { w } } _ { \mathcal { P } , r } \rVert _ { 2 }$ via Lemmas 3 and Lemma 9, which gives

$$
\| \bar { \boldsymbol { w } } _ { \mathcal { P } , r } ^ { \mathrm { F } } - \boldsymbol { w } _ { r } ^ { ( 0 ) } \| _ { 2 } \le O ( \mathrm { p o l y } ( N , \delta ^ { - 1 } , \lambda _ { 0 } ^ { - 1 } ) / \sqrt { m } )\tag{18}
$$

Using above bounds, we can conclude that $f _ { a } ( \bar { W } _ { \mathcal { P } } ^ { \mathrm { F } } , \cdot ) \in$ $\mathcal { F } _ { W ^ { ( 0 ) } , a , R , B }$ with R and B set by Equation (18) and (17) respectively. Using Lemma 12 and Lemma 13, we have

$$
\begin{array} { r l } & { \underset { f \in { \mathscr F } _ { W ^ { ( 0 ) , a , R , B } } } { \operatorname* { s u p } } \{ L _ { T } ^ { f , \ell } ( { \bar { W } } _ { \mathcal { P } } ^ { \mathrm { F } } ) - L _ { \mathcal { D } } ^ { f , \ell } ( { \bar { W } } _ { \mathcal { P } } ^ { \mathrm { F } } ) \} } \\ & { \leq \underset { f \in { \mathscr F } _ { W ^ { ( 0 ) } , a , R , B } } { \operatorname* { s u p } } \{ L _ { T } ^ { f , \ell } ( { \bar { W } } _ { \mathcal { P } } ^ { \mathrm { F } } ) - L _ { S } ^ { f , \ell } ( { \bar { W } } _ { \mathcal { P } } ^ { \mathrm { F } } ) \} } \\ & { \quad + \underset { f \in { \mathscr F } _ { W ^ { ( 0 ) } , a , R , B } } { \operatorname* { s u p } } \{ L _ { S } ^ { f , \ell } ( { \bar { W } } _ { \mathcal { P } } ^ { \mathrm { F } } ) - L _ { \mathcal { D } } ^ { f , \ell } ( { \bar { W } } _ { \mathcal { P } } ^ { \mathrm { F } } ) \} } \\ & { \leq O ( \mathcal { R } _ { n } ^ { M } ( \mathcal { F } _ { W ^ { ( 0 ) } , a , R , B } ) + \sqrt { \log ( 1 / \delta ) / N } + \sqrt { \log ( 1 / \delta ) / M } ) . } \end{array}
$$

By Lemma 11, the Rademacher complexity satisfies $\begin{array} { r l r } { \tilde { \mathcal { R } } _ { n } ^ { M } ( \mathcal { F } _ { W ^ { ( 0 ) } , a , R , B } ) } & { { } \leq } & { O ( \frac { 1 } { \rho } \sqrt { \pmb { y } ^ { \top } ( \mathbf { H } ^ { \infty } ) ^ { - 1 } \pmb { y } / \lambda _ { 0 } } ) } \end{array}$ . Since $\begin{array} { r l r } { L _ { \mathcal { T } } ^ { f , \ell } ( \bar { \pmb { W } } _ { \mathcal { P } } ^ { \mathrm { F } } ) } & { = } & { L _ { \mathcal { T } } ^ { f , \ell } ( \bar { \pmb { W } } _ { \mathcal { P } } ^ { \mathrm { F } } ) - \overset { r } { L } _ { \mathcal { D } } ^ { f , \ell } ( \bar { \pmb { W } } _ { \mathcal { P } } ^ { \mathrm { F } } ) + L _ { \mathcal { D } } ^ { f , \ell } ( \bar { \pmb { W } } _ { \mathcal { P } } ^ { \mathrm { F } } ) , } \end{array}$ , by combing with the training error established in Theorem 2, we obtain the result by union bound.

Theorem 3 shows that, with sufficiently many clients and local samples, the generalization of F-SOSA is governed by a Rademacher complexity term of order $\textstyle \frac { 1 } { \rho } { \sqrt { { \pmb y } ^ { \top } ( \mathbf { H } ^ { \infty } ) ^ { - 1 } { \pmb y } / \lambda _ { 0 } } }$ As shown in [45], the quantity $\mathbf { \boldsymbol { y } } ^ { \top } ( \mathbf { \boldsymbol { H } } ^ { \infty } ) ^ { - 1 } \mathbf { \boldsymbol { y } }$ is bounded by a constant independent of sample size for a broad class of target functions, including linear, quadratic, and cosine models. The factor $1 / \rho$ reflects a trade-off between editing efficiency and generalization: larger $\rho$ includes more clients and improves the utility/generalization bound, but weakens TV stability and increases the expected editing cost.

## C. Exactness and Efficiency of Model Editing

## F-SOSA-U/I are valid transports and efficient for editing.

Theorem 4. (F-SOSA, F-SOSA-U/I) satisfies exact federated model editing. For R editing requests, F-SOSA-U/I triggers at most $O ( \rho R )$ recomputation in expectation, with total expected runtime bounded by $O ( \operatorname* { m a x } \{ \rho R$ · Training Time, R}).

Proof. Let $\mu _ { ( P , M ) }$ be the uniform distribution over size-P subsets of [M]. Let $\mathcal { P } \sim \mu _ { ( P , M ) }$ be a sampled client subset. Assume M leaves or $M + 1$ joins, and consider a deletion map DEL $: ~ \mathcal { P } ~ \mapsto ~ \mathcal { P } ^ { \prime } ~ \subseteq ~ [ M - 1 ]$ (for client removal), and an insertion map $\mathrm { I N S ~ } : \mathcal { P } \ \mapsto \ \mathcal { P } ^ { \prime } \ \subseteq \ [ M + 1 ]$ (for new client addition). These induce pushforward distributions $\mu _ { ( P , M ) } ^ { \mathrm { D E L } } : = \mathrm { D E L } \# \mu _ { ( P , M ) }$ and $\mu _ { ( P , M ) } ^ { \mathrm { I N S } } : = \mathrm { I N S } \# \mu _ { ( P , M ) }$ . To show F-SOSA-U/I are valid transports, we prove that for any $\mathcal { P } ~ \in ~ [ M ] ^ { P } , ~ \mu _ { ( P , M ) } ^ { \mathrm { D E L } } ( \mathcal { P } ) ~ = ~ \mu _ { P , M - 1 } ( \mathcal { P } )$ and for any $\mathcal { P } \in [ M + 1 ] ^ { P } , \mu _ { ( P , M ) } ^ { \mathrm { I N S } } ( \mathcal { P } ) = \mu _ { ( P , M + 1 ) } ( \mathcal { P } )$

For F-SOSA-U, we consider two cases based on whether the membership verification succeeds. Case 1 (verification succeeds). The pushforward measure $\mu _ { ( P , M ) } ^ { \mathrm { D E L } }$ is simply $\mu _ { ( P , M ) }$ conditioned on the event $M \not \in { \mathcal { P } }$

$$
\mu _ { ( P , M ) } ^ { \mathrm { D E L } } ( \mathcal { P } ) = \mu _ { ( P , M ) } ( \mathcal { P } | M \notin \mathcal { P } ) = \frac { \mu _ { ( P , M ) } ( \mathcal { P } \cap \{ M \notin \mathcal { P } \} ) } { \mu _ { ( P , M ) } ( \{ M \notin \mathcal { P } \} ) } .
$$

The denominator $\mu _ { ( P , M ) } ( \{ M \notin \mathcal { P } \} ) = 1 - \mu _ { ( P , M ) } ( \{ M \in \mathcal { P } \} ) =$ $1 - \textstyle { \binom { M - 1 } { P - 1 } } / \binom { M } { P } \stackrel { } { = } 1 - \stackrel { \textstyle ^ { \prime } } { P } / M$ . For $\mu _ { ( P , M ) } ( { \mathcal { P } } \cap \{ M \not \in { \mathcal { P } } \} )$ it equals 0 if $M \in { \mathcal { P } } ,$ and $\mu _ { ( P , M ) } ( \mathcal { P } )$ otherwise. In both branches, we have $\mu _ { ( P , M ) } ^ { \mathrm { D E L } } ( { \mathcal P } ) \stackrel { \cdot } { = } \mu _ { P , M - 1 } ( { \mathcal P } )$ . Case 2 (verification fails). In this case, M would have been replaced by some point in $\mathcal { P }$ and there are M possibilities. Let υ be the uniform probability measure over [M]. For each $k \in [ M ]$ , let $\mathcal { E } _ { k }$ be the event that M was replaced by $i _ { k } .$ . The events $\mathcal { E } _ { k } \mathrm { { ' s } }$ are disjoint and $\mathcal { P }$ is $\cup _ { k = 1 } ^ { M } \mathcal { E } _ { k }$ . Then $\mu _ { ( P , M ) } ^ { \mathrm { D E L } } ( \mathcal { P } ) =$ $\begin{array} { r } { \mu _ { ( P , M ) } ^ { \mathrm { D E L } } ( \sum _ { k = 1 } ^ { M } \mathcal { E } _ { k } ) = \sum _ { k = 1 } ^ { M } \mu _ { ( P , M ) } ( \mathcal { P } \backslash \{ i _ { k } \} \cup \{ M \} ) v ( \{ i _ { k } \} ) = } \end{array}$ $1 \dot { / } \binom { M - 1 } { P } = \mu _ { ( P , M - 1 ) } ( \mathcal { P } )$

For F-SOSA-I, given ${ \mathcal P } ,$ we consider two cases based on whether the inserted $M + 1$ lies in $\mathcal { P }$ or not. Case 1 $( M { + } 1 \in$ ${ \mathcal { P } } ) .$ . In this case, Bernoulli $( P / ( M { + } 1 )$ must be 1, M+1 would have replaced some other points in $\ [ M + 1 ] \backslash \mathcal { P }$ and there are $M { + } 1 { - } P$ possibilities. Let ν be a uniform probability measure over $[ M { + } 1 { - } P ]$ . For each $k \in \left[ M { + } 1 { - } P \right]$ , let $\mathcal { E } _ { k }$ be the event that $M { + 1 }$ replaced $i _ { k } ,$ then $\mathcal { E } _ { k } \mathrm { { ' s } }$ are disjoint and the event $\mathcal { P }$ is $\cup _ { k = 1 } ^ { M + 1 - P } \mathcal { E } _ { k }$ . Then $\mu _ { ( P , M ) } ^ { \mathrm { I N S } } ( { \mathcal E } _ { k } )$ equals

$$
\begin{array} { r l } & { \mu _ { ( P , M ) } ^ { \mathrm { I N S } } ( \mathcal { P } \backslash \{ M + 1 \} \cup \{ i _ { k } \} | i _ { k } \mathrm { r e p l a c e d } ) \mu _ { ( P , M ) } ^ { \mathrm { I N S } } ( i _ { k } \mathrm { r e p l a c e d } ) } \\ & { \ = \mu _ { ( P , M ) } ( \mathcal { P } \backslash \{ M + 1 \} \cup \{ i _ { k } \} | \{ i _ { k } \} ) \nu ( i _ { k } ) = \frac { 1 } { { \binom { M } { P - 1 } } ( M + 1 - P ) } . } \end{array}
$$

Thus, $\begin{array} { r } { \mu _ { ( P , M ) } ^ { \mathrm { I N S } } ( \mathcal { P } ) = \frac { P } { M + 1 } \mu _ { ( P , M ) } ^ { \mathrm { I N S } } ( \cup _ { k = 1 } ^ { M + 1 - P } \mathcal { E } _ { k } ) = \mu _ { ( P , M + 1 ) } ( \mathcal { P } ) . } \end{array}$ Case $\mathsf { \pmb { 2 } } \overset { \cdot } { ( } \dot { M } + 1 \notin \mathcal { P } )$ . Bernoulli $\begin{array} { r } { \dot { \left( P / ( M + 1 ) \right) } = 0 , } \end{array}$ there is no replacement, $\begin{array} { r } { \mu _ { ( P , M ) } ^ { \mathrm { I N S } } ( \mathcal { P } ) = ( 1 - \frac { \hbar } { M + 1 } ) / \binom { M } { P } = 1 / \binom { M + 1 } { P } = } \end{array}$ $\mu _ { ( P , M + 1 ) } ( \mathcal { P } )$

Therefore, F-SOSA-U/I are both valid transports. $\mathrm { L e t } \mathcal { P } ^ { \mathrm { E D T } }$ be the post-editing sampled subset. By ρ-TV-stability of F-SOSA, $\mathbf { \bar { \mathbb { P } } } ( \mathcal { P } \neq \mathcal { P } ^ { \mathrm { E D T } } ) \leq \mathbf { \bar { \rho } } ,$ , and $M / 2 \le M ( t ) \le 2 M$ , so each request triggers recomputation with probability at most $2 \rho .$ For R total edits, the expected number of recomputations is

$O ( \rho R )$ , with total expected runtime bounded by $O ( \operatorname* { m a x } \{ \rho R$ Training Time, R}). □

Proposition 1 (Exact and Optimal Batch Coupling). In the batch setting of Section IV-B, Algorithm 4 satisfies exact model editing in Definition 1 at batch completion. Moreover, $\mathrm { P r } ( \mathcal { P ^ { \prime } } \mathit { \Pi } \ne \mathcal { P } ) \ = \ 1 - \ \binom { M - d } { P } / \operatorname* { m a x } \{ \binom { \dot { M } } { P } , \binom { M ^ { \prime } } { P } \}$ , and $\begin{array} { r } { \mathbb { E } [ | \mathcal { P } \setminus \mathcal { P } ^ { \prime } | ] = \operatorname* { m a x } \left\{ \frac { P d } { M } , \frac { P a } { M ^ { \prime } } \right\} } \end{array}$ . The algorithm simultaneously minimizes these two quantities over all couplings of the initial and final uniform size-P subset distributions, where $\binom { M - d } { P } = 0 \mathrm { ~ i f ~ } M - d < P$

Proof. Condition on the common training randomness and an independently chosen pairing. Paired replacement is a bijective relabeling and preserves uniformity. Remaining insertions preserve uniformity by the fixed-P insertion coupling. For remaining deletions, conditional on the number of survivors, the surviving selected clients form a uniform subset of $\mathcal { M } ^ { \prime } ;$ uniform replenishment therefore produces a uniform size-$P$ subset. Thus $\mathcal { P } ^ { \prime }$ has the retraining subset distribution. The net updates recover its aggregate statistics exactly, and the same fusion rule produces its model. After temporary buffers are discarded, the model and internal state are the same functions of ${ \mathcal { P } } ^ { \prime }$ as in retraining. Averaging over the common training randomness establishes Definition 1. For any coupling $( S , T )$ of the initial and final uniform subset distributions, equality is possible only on the $\binom { M - d } { P }$ subsets containing no deleted or added clients. Each such subset contributes at most the smaller of its two marginal probabilities, giving $\begin{array} { r } { \operatorname* { P r } ( S ~ = ~ T ) ~ \le ~ { \binom { M - d } { P } } / \operatorname* { m a x } \{ { \binom { M } { P } } , { \binom { M ^ { \prime } } { P } } \} } \end{array}$ }. Moreover, every selected deleted client must leave, and every selected added client must enter. Since both subsets have size $P ,$ $| S \setminus T | \geq | S \cap \mathcal { D } | , | S \setminus T | \geq | T \cap \mathcal { A } |$ . Taking expectations yields $\mathbb { E } [ | S \setminus T | ] \ \geq$ max $\left\{ \frac { P d } { M } , \frac { P a } { M ^ { \prime } } \right\}$ . The algorithm attains both bounds. If $a \ \geq \ d ,$ it adds no previously unselected common client. Hence $\mathcal { P } ^ { \prime } = \mathcal { P }$ exactly when $\mathcal { P } ^ { \prime } \cap \mathcal { A } = \emptyset$ , and $\vert { \mathcal { P } } \setminus { \mathcal { P } } ^ { \prime } \vert = \vert { \mathcal { P } } ^ { \prime } \cap { \mathcal { A } } \vert$ . Uniformity gives agreement probability $\dot { ( } { M } _ { P } ^ { - } d ) \dot { / } ( { M } ^ { \prime } )$ and expected net replacements $P a / M ^ { \prime } . \mathrm { I f } d \ge a ,$ every originally selected common client is retained. Hence equality holds exactly when $\mathcal { P } \cap \mathcal { D } = \mathcal { O }$ , and $| \mathcal { P } \backslash \mathcal { P ^ { \prime } } | = | \mathcal { P } \cap \mathcal { D } |$ giving agreement probability $\binom { M - d } { P } / \binom { M } { P }$ and expectation $P d / M$ . Since $M ^ { \prime } - M = a - d$ and $\begin{array} { r } { \frac { a } { M ^ { \prime } } - \frac { d } { M } = \frac { ( M - d ) ( a - d ) } { M M ^ { \prime } } } \end{array}$ both branches attain the two bounds simultaneously.

The batch updates $| \mathcal { P } \setminus \mathcal { P } ^ { \prime } |$ contribution pairs and performs one projection if and only i $\mathbf { f } \mathbf { \nabla } \mathcal { P } ^ { \prime } \neq \mathcal { P }$ . For $k = d { + } a ,$ , let $M _ { 0 }$ be the initial population size. If $P \le \rho M _ { 0 }$ and min $\left\{ { M , M ^ { \prime } } \right\} \ge$ $M _ { 0 } / 2$ , Proposition 1 gives $\mathbb { E } [ | \mathcal { P } \backslash \mathcal { P } ^ { \prime } | ] \leq 2 \rho \operatorname* { m a x } \{ d , a \} \leq 2 \rho k$ Since a subset change requires at least one net replacement, the projection is triggered with probability at most min{1, 2ρk}.

## VI. EXPERIMENTS

We empirically evaluate our framework with focus on: (1) one-shot learning performance, (2) model editing accuracy and efficiency, (3) trade-offs controlled by the stability level $\rho ,$ and (4) resilience to membership inference attacks. We also include a realistic case study based on vehicle mobility simulation.

## A. Experimental Setup

1) Datasets and Models: We use MNIST [47], SVHN [48], CIFAR-10 [49], and FEMNIST from LEAF [50]. For MNIST/SVHN/CIFAR-10, we simulate client heterogeneity using Dirichlet partitioning $( \alpha ~ \in ~ \{ 0 . 1 , 0 . 5 , 0 . 8 \}$ ; default $\alpha { = } 0 . 5 )$ . FEMNIST, as a FL dataset, naturally exhibits non-IID characteristics; we follow [9], sampling 10% of users and grouping into M clients. Unless otherwise specified, we use the two-layer CNN from [8] with K-FAC Fisher approximation. To evaluate F-SOSA beyond the default classification setting, we additionally consider ResNet18 on CIFAR-10 and YOLOv8s on the real-world driving dataset BDD100K; detailed results are reported in the Appendix.

2) Baselines: We compare F-SOSA with three one-shot FL baselines: FedFisher [8], which uses client-specific Fisher information for model fusion; FedLPA [9], which constructs local posteriors via layer-wise Laplace approximation; and Dense [10], which performs data-free distillation using GANgenerated samples. For dynamic-client experiments, we use a fixed-ratio implementation of F-SOSA-I/U to accommodate population changes that may extend beyond the bounded range assumed in the theoretical analysis. Specifically, we set $P _ { t } = \lfloor \rho M ( t ) \rfloor$ ⌋, where $M ( t )$ is the current active client population. This keeps the participation fraction approximately constant and adapts the aggregation size to population changes. For both unlearning and incremental learning, Retrain reruns F-SOSA on the updated client set using the same current subset size $P _ { t }$ . For unlearning, we additionally compare with FedEraser [14], which reconstructs the model using calibrated gradient updates; FedRR [28], which uses diagonal Fisher information with momentum correction; and FATS-CU [29], which stabilizes FedAvg for rapid unlearning. For incremental learning, we also compare with Fine-Tuning, which adapts the global model on the new client’s data using a tuned number of epochs selected from $\{ 5 , 1 0 , \ldots , 5 0 \}$ . All methods are evaluated under the one-shot constraint for fair comparison.

3) Hyperparameters: We follow the settings in [8], using learning rate $\eta ~ = ~ 0 . 0 1$ and default local epoch number $K = 5 0$ . For learning experiments, we set $M = 2 0$ clients; for model editing, we use at most 100 clients and initialize with 50 active clients, while the scalability experiments in the Appendix consider client populations from 20 to 500. The stability level is set to $\rho = 0 . 5$ unless otherwise specified.

4) Evaluation Metrics: We report test accuracy, training loss, total wall-clock time, and total communication bytes, including global-model broadcasts, local-model uploads, and Fisher-information uploads. Time and communication efficiency measure the average accuracy gain per unit of time and communication, respectively.

## B. Experimental Results

1) Performance of F-SOSA in Heterogeneous Settings: We evaluate F-SOSA under varying data heterogeneity levels $( \alpha \in \{ 0 . 1 , 0 . 5 , 0 . 8 \} )$ ) on MNIST/SVHN/CIFAR-10, and include FEMNIST as an inherently heterogeneous FL dataset. We compare against three state-of-the-art one-shot FL baselines. As shown in Table II and Table III, F-SOSA consistently achieves the lowest communication cost across all datasets and heterogeneity levels, leading to the highest communication efficiency. This benefit arises from its client subsampling mechanism, which reduces aggregation overhead. Time cost is also substantially lower, often by over 50% compared to FedLPA and Dense, while remaining comparable to FedFisher in most cases. In terms of accuracy, F-SOSA matches or exceeds FedFisher under moderate heterogeneity $( \alpha = 0 . 5 , 0 . 8 )$ and consistently outperforms Dense and FedLPA. Under high heterogeneity $( \alpha = 0 . 1 )$ , its performance remains robust with minimal degradation. These results demonstrate that F-SOSA offers strong utility-efficiency trade-offs, maintaining competitive accuracy while substantially reducing communication and computation costs.

TABLE II  
PERFORMANCE COMPARISON OF F-SOSA AND BASELINES ACROSS MNIST, SVHN, AND CIFAR-10 UNDER VARYING DIRICHLET HETEROGENEITY α. BEST RESULTS ARE IN BOLD; SECOND-BEST RESULTS ARE UNDERLINED.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Metrics</td><td colspan="4"> $\alpha = 0 . 1$ </td><td colspan="4"> $\alpha = 0 . 5$ </td><td colspan="4"> $\alpha = 0 . 8$ </td></tr><tr><td>FedFisher</td><td>FedLPA</td><td>Dense</td><td>F-SOSA</td><td>FedFisher</td><td>FedLPA</td><td>Dense</td><td>F-SOSA</td><td>FedFisher</td><td>FedLPA</td><td>Dense</td><td>F-SOSA</td></tr><tr><td rowspan="5">MNIST</td><td>Accuracy (%)</td><td>90.35</td><td>87.90</td><td>85.92</td><td>91.08</td><td>95.98</td><td>94.21</td><td>96.06</td><td>95.43</td><td>96.08</td><td>92.82</td><td>96.93</td><td>95.80</td></tr><tr><td>Time (s)</td><td>43.11</td><td>24.59</td><td>187.96</td><td>34.63</td><td>43.76</td><td>26.61</td><td>190.54</td><td>34.59</td><td>43.15</td><td>29.02</td><td>193.58</td><td>35.56</td></tr><tr><td>Comm. (MB)</td><td>35.30</td><td>793.31</td><td>33.76</td><td>18.46</td><td>35.30</td><td>793.31</td><td>33.76</td><td>18.46</td><td>35.30</td><td>793.31</td><td>33.76</td><td>18.46</td></tr><tr><td>Time Efficiency</td><td>2.10</td><td>3.57</td><td>0.52</td><td>2.63</td><td>2.19</td><td>3.54</td><td>0.57</td><td>2.76</td><td>2.23</td><td>3.20</td><td>0.58</td><td>2.69</td></tr><tr><td>Comm. Efficiency</td><td>2.56</td><td>0.11</td><td>2.55</td><td>4.93</td><td>2.72</td><td>0.12</td><td>2.85</td><td>5.17</td><td>2.72</td><td>0.12</td><td>2.87</td><td>5.19</td></tr><tr><td rowspan="5">SVHN</td><td>Accuracy (%)</td><td>64.89</td><td>38.81</td><td>62.78</td><td>64.88</td><td>80.52</td><td>76.24</td><td>70.63</td><td>80.33</td><td>79.64</td><td>77.75</td><td>78.76</td><td>79.40</td></tr><tr><td>Time (s)</td><td>29.15</td><td>50.50</td><td>198.85</td><td>23.54</td><td>28.98</td><td>51.61</td><td>184.83</td><td>21.29</td><td>28.53</td><td>51.33</td><td>188.70</td><td>22.14</td></tr><tr><td>Comm. (MB)</td><td>77.04</td><td>362.74</td><td>75.27</td><td>40.31</td><td>77.04</td><td>362.74</td><td>75.27</td><td>40.31</td><td>77.04</td><td>362.74</td><td>75.27</td><td>40.31</td></tr><tr><td>Time Efficiency</td><td>2.23</td><td>0.77</td><td>0.36</td><td>2.76</td><td>2.78</td><td>1.48</td><td>0.40</td><td>3.77</td><td>2.79</td><td>1.51</td><td>0.46</td><td>3.59</td></tr><tr><td>Comm. Efficiency</td><td>0.84</td><td>0.11</td><td>0.83</td><td>1.61</td><td>1.05</td><td>0.21</td><td>0.94</td><td>1.99</td><td>1.03</td><td>0.21</td><td>1.05</td><td>1.97</td></tr><tr><td rowspan="5">CIFAR10</td><td>Accuracy (%)</td><td>39.99</td><td>30.49</td><td>36.57</td><td>32.38</td><td>51.86</td><td>40.36</td><td>50.09</td><td>51.69</td><td>55.18</td><td>44.32</td><td>52.58</td><td>52.65</td></tr><tr><td>Time (s)</td><td>22.76</td><td>63.79</td><td>202.91</td><td>16.55</td><td>23.16</td><td>56.50</td><td>197.96</td><td>16.26</td><td>23.22</td><td>55.92</td><td>197.70</td><td>16.22</td></tr><tr><td>Comm. (MB)</td><td>77.04</td><td>362.74</td><td>75.27</td><td>40.31</td><td>77.04</td><td>362.74</td><td>75.27</td><td>40.31</td><td>77.04</td><td>362.74</td><td>75.27</td><td>40.31</td></tr><tr><td>Time Efficiency</td><td>1.76</td><td>0.48</td><td>0.19</td><td>1.96</td><td>2.24</td><td>0.72</td><td>0.26</td><td>3.18</td><td>2.38</td><td>0.79</td><td>0.28</td><td>3.25</td></tr><tr><td>Comm. Efficiency</td><td>0.52</td><td>0.08</td><td>0.49</td><td>0.80</td><td>0.67</td><td>0.11</td><td>0.67</td><td>1.28</td><td>0.72</td><td>0.12</td><td>0.70</td><td>1.31</td></tr></table>

![](images/6a01ae6fd7452630fa28a50a77e50a915689d342b7b959f3db43c46fda123c10.jpg)

![](images/a09550f7317ab709d8de0953423c3555622dfcd2d4e71dd45da6936b0cd710a3.jpg)

![](images/3f607aa35eb8f524b8f4f50f4b25d294877a7395f8f87e18c4f291e5fd5d547d.jpg)  
Fig. 1. Effect of the stability level ρ on F-SOSA’s accuracy and time/ communication overhead across datasets.

TABLE III  
PERFORMANCE COMPARISON ON FEMNIST UNDER A REAL-WORLD NON-IID SETTING. BEST RESULTS ARE IN BOLD; SECOND-BEST RESULTS ARE UNDERLINED.
<table><tr><td>Metrics</td><td>FedFisher</td><td>FedLPA</td><td>Dense</td><td>F-SOSA</td></tr><tr><td>Accuracy (%)</td><td>76.94</td><td>70.18</td><td>75.97</td><td>75.71</td></tr><tr><td>Time (s)</td><td>115.89</td><td>18.69</td><td>138.57</td><td>67.07</td></tr><tr><td>Comm. (MB)</td><td>554.12</td><td>1968.70</td><td>528.83</td><td>289.68</td></tr><tr><td>Time Efficiency</td><td>0.66</td><td>3.75</td><td>0.64</td><td>1.13</td></tr><tr><td>Comm. Efficiency</td><td>0.139</td><td>0.036</td><td>0.144</td><td>0.261</td></tr></table>

2) Impact and Selection of Hyperparameters: We examine how the stability parameter ρ and local epoch number K affect

![](images/ed9b029fe4de894a55eb4b1d68749e059b89da4c80eeadf67a39c274937c47e6.jpg)

![](images/a2259ffa04038f9811cd9105f8cd5ff5272e8387f0b412932fd31d6c61a76722.jpg)

![](images/e72562d0b31cb683e997ed642e87c8d77d168ca6b89dd7079bd262c8a3391b38.jpg)

![](images/9474765130c89100fdcf346a5886126365b363bf1973bfd91104f165228675ad.jpg)  
Fig. 2. Effect of local epoch number K on performance. Top-left: MNIST, top-right: SVHN, bottom-left: FEMNIST, bottom-right: CIFAR-10.

F-SOSA’s utility and efficiency. Figure 1 shows that increasing ρ improves accuracy, as more clients are aggregated, enhancing generalization. When $\rho = 1 . 0$ , F-SOSA recovers FedFisher and achieves the highest accuracy. However, this comes at a cost of increased time and communication, especially on larger datasets like FEMNIST. A moderate setting $( \rho = 0 . 5 )$ provides a strong trade-off between utility and efficiency, and is used by default. As shown in Figure 2, larger K reduces training loss, indicating better local fitting. However, test accuracy peaks around $K = 3 0 \ \mathrm { o r \ 5 0 }$ , beyond which overfitting may reduce generalization. Meanwhile, time cost grows linearly with K. We adopt K = 50 as default, balancing generalization and

TABLE IV  
MEMBERSHIP INFERENCE ATTACK (MIA) RESULTS ON UNLEARNED GLOBAL MODELS FROM DIFFERENT METHODS ON FEMNIST AND CIFAR-10
<table><tr><td rowspan="2">Method</td><td colspan="3">FEMNIST</td><td colspan="3">CIFAR-10</td></tr><tr><td>Accuracy (%)</td><td>Precision (%)</td><td> $\overline { { \mathrm { A U C } \mathrm { ~ ( \% ) } } }$ </td><td>Accuracy (%)</td><td>Precision (%)</td><td>AUC (%)</td></tr><tr><td>FedErasesr</td><td> $5 0 . 0 1 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $5 1 . 0 8 { \scriptstyle \pm 3 . 2 3 }$ </td><td> $5 0 . 0 1 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $4 8 . 9 9 { \scriptstyle \pm 0 . 7 8 }$ </td><td> $4 0 . 7 5 { \scriptstyle \pm 7 . 8 3 }$ </td><td> $4 8 . 9 9 { \scriptstyle \pm 0 . 7 8 }$ </td></tr><tr><td>FedRR</td><td> $5 0 . 4 5 { \scriptstyle \pm 0 . 4 9 }$ </td><td> $5 0 . 3 7 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $5 0 . 4 5 { \scriptstyle \pm 0 . 4 9 }$ </td><td> $4 5 . 1 7 { \scriptstyle \pm 2 . 3 3 }$ </td><td> $4 2 . 2 9 { \scriptstyle \pm 2 . 8 7 }$ </td><td> $4 5 . 1 7 { \scriptstyle \pm 2 . 3 3 }$ </td></tr><tr><td>FATS-CU</td><td> $5 0 . 4 7 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $5 0 . 3 3 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $5 0 . 4 7 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $5 3 . 5 4 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $6 1 . 2 9 { \scriptstyle \pm 2 . 9 9 }$ </td><td> $5 3 . 5 4 { \scriptstyle \pm 0 . 8 8 }$ </td></tr><tr><td>Retrain</td><td> $5 0 . 4 5 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $5 0 . 3 1 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $5 0 . 4 5 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $5 1 . 6 4 { \scriptstyle \pm 1 . 6 7 }$ </td><td> $5 2 . 0 4 { \scriptstyle \pm 2 . 0 9 }$ </td><td> $5 1 . 6 4 { \scriptstyle \pm 1 . 6 7 }$ </td></tr><tr><td>F-SOSA-U</td><td> $5 0 . 9 2 _ { \pm 0 . 3 3 }$ </td><td> $5 0 . 6 3 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $5 0 . 9 2 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $5 1 . 6 3 { \scriptstyle \pm 1 . 8 0 }$ </td><td> $5 2 . 1 7 { \scriptstyle \pm 2 . 7 2 }$ </td><td> $5 1 . 6 3 { \scriptstyle \pm 1 . 8 0 }$ </td></tr></table>

![](images/aac0c05702811990a3acad7ee09a34098fc35fef6cbdaa383f8365e5f96953f3.jpg)

![](images/5e46e24ee924f53b88e1e2bbdd3653211fd9e4dd8a8a8b2d307030a10a9d7268.jpg)

![](images/180081f3c16faf24b965102b264a2dfce62e679706c179fab8fb9e2bbcee2dcc.jpg)

![](images/bc54a075268eaa72ce12bf1885189cf95ae6b59262890d995b47b93719525635.jpg)

![](images/f350dea4ff873ac4dfa8f8d2eb8343a953361b7e54c0e7f5d3c1b0c7d9b4f8c4.jpg)

![](images/88dcac4f3a5e2ec349553f7a9bea556d24ace1f5bab57da87b39562b23454cba.jpg)

![](images/f8946ce5c2b86a507fdf3a35be90b02eddb6480674dbfd2e31cda9e3ae4d92d0.jpg)  
Fig. 3. Performance of F-SOSA-I/U vs. baselines on CIFAR-10. Columns: IL (left), UL (right). Rows: test accuracy (top), time cost (middle), communication cost (bottom).

![](images/82fc6566d7c187521c4727a8db4d4146038d8a4e0cd806da2818e0dfc4945cac.jpg)

![](images/508e0c27108bbdadf591588b16be9a7474217ba76cada230a75dfacd628c0a33.jpg)

![](images/fa05fd49e8f9f84c252ed5e45bbc3d63b7bcdbb0a1d5c734434dee438f12241b.jpg)

![](images/70d2779bd0e30b8c754621fd85e71a961c15c98477c4d457e1ff129ac8b2e054.jpg)  
Fig. 4. Effect of stability parameter ρ on accuracy and efficiency. Top-left: MNIST, top-right: SVHN, bottom-left: FEMNIST, bottom-right: CIFAR-10.  
Fig. 5. The Luxembourg map and the region considered.

efficiency across datasets.

3) Performance of F-SOSA-I/U: We evaluate F-SOSA-I/U on CIFAR-10 using a fixed sequence of randomly generated client join and leave requests. All methods are tested on the same sequence for fair comparison. Figure 3 presents the results. Note that initial accuracy and overhead differ across unlearning baselines due to their learning methods: FedEraser, FedRR, and FATS-CU use FedAvg, while F-SOSA-U and Retrain use F-SOSA. F-SOSA-I matches Retrain in accuracy while consistently outperforming Fine-Tuning by up to 18.6%. It also exhibits the lowest accuracy fluctuation (2.28%), compared to 3.20% (Retrain) and 9.12% (Fine-Tuning), benefiting from its verification mechanism that avoids unnecessary re-aggregation. In contrast, Fine-Tuning overfits to newly joined clients, degrading generalization. F-SOSA-I reduces the final cumulative communication cost by 39.73% compared with Retrain. Fine-Tuning incurs the lowest communication and time costs, but suffers from significant utility loss. F-SOSA-U reaches accuracy comparable to Retrain and surpasses FedEraser/FedRR/FATS-CU by up to 10%. This advantage arises from Fisher-aware aggregation. While FedEraser/FedRR/FATS-CU incur slightly lower time costs, their performance drops notably. F-SOSA-U also reduces accuracy fluctuation and maintains high efficiency by avoiding redundant recomputation.

4) Trade-off Between Editing Utility and Efficiency w.r.t. Stability Parameter $\rho \mathrm { : }$ We study how the stability parameter $\rho$ influences the utility-efficiency trade-off in F-SOSA-I/U across datasets, with results in Figure 4. As $\rho$ increases, accuracy improves due to greater client participation, which enhances generalization. However, this comes at the cost of increased time and communication overhead from larger aggregation sets. Notably, small values of ρ (e.g., 0.1–0.3) yield significant accuracy gains with minimal efficiency loss, suggesting that editing benefits from a small number of well-chosen clients. Around $\rho = 0 . 5$ , F-SOSA-I/U achieves a favorable balance between accuracy and resource usage across all datasets.

5) Resistance to Membership Inference Attacks: We assess unlearning effectiveness via membership inference attacks (MIA) on the global models produced by different methods on FEMNIST and CIFAR-10 (Table IV). F-SOSA-U achieves near-random metrics (accuracy, precision, AUC around 50%), closely matching Retrain across both datasets. This suggests that the model cannot reliably distinguish whether a sample was seen during training, confirming successful unlearning. On CIFAR-10, F-SOSA-U shows even smaller deviations from random guessing compared to other baselines, some of which exceed 5%, indicating a lower risk of memorization leakage. Given its much lower time and communication costs, F-SOSA-U offers an effective and efficient solution for one-shot federated unlearning.

![](images/0c283d23a8e065184f190292b70a5cbf1f935b04435c76adaca6328986680120.jpg)

![](images/d1819bb996dbbd9fa27ddb3204419ad78fe6ae90c8236861c1a61de9d48efec9.jpg)

![](images/eadab35e6f45a4a99b6ffcc193eb0c3a981ea55717a6646471337e1e079fb988.jpg)

![](images/6d77062d48b2678334a75fb4fd52b8856fbcf9437708286486b0b8dc21714f81.jpg)

![](images/9a5f3547991613244a349e98e4da9717f7eeed595fbb26c25274471dba4d035b.jpg)

![](images/1cfe2689d32227b21e407f2958d0eef62fef10adb92097ebc9fb171b79eca6d3.jpg)

![](images/7bf29d21527c205c5b96273973680ebc269cb823f8a4ab9650edef33056787a3.jpg)

![](images/bcf949aa4217274166d4b8d71471877fbea6b111f7492b0e8b4b6274f25cac90.jpg)  
Fig. 6. Test accuracy (left) and cumulative cost (right) of F-SOSA-I/U vs. Retrain under simulator-generated model editing requests. Rows: MNIST, SVHN, FEMNIST, CIFAR-10 (top to bottom).

6) F-SOSA Performance Across Model Architectures: Experiments with ResNet18 on CIFAR-10 and YOLOv8s on BDD100K extend the default CNN evaluation to deeper architectures. Across varying data heterogeneity, F-SOSA maintains competitive utility with lower communication and computation costs (Appendix B-B).

7) Scalability with the Number of Clients: We evaluate F-SOSA-I/U with total client populations ranging from 20 to 500, using 15 sequential incremental-learning and 15 sequential unlearning requests at each scale. Across the evaluated range, F-SOSA-I/U maintain accuracy close to Retrain while consistently reducing communication and computation costs.

The efficiency gains remain substantial even with 500 clients, showing that selective editing continues to provide benefits as the client population grows. Detailed results are reported in the Appendix B-C.

## C. Case Study Using Simulation of Urban Mobility (SUMO)

We further evaluate F-SOSA-I/U using the Simulation of Urban MObility (SUMO) [51], focusing on a selected city center region in the Luxembourg scenario [52] (see Figure 5). Editing requests are triggered by vehicle movements during the 08:00 morning rush hour, simulating dynamic client join/leave events. For this case study, we map each vehicle arrival to a new client contribution and each vehicle departure to an unlearning request issued before disconnection, thereby generating a mobility-driven editing sequence. We apply F-SOSA-I/U and Retrain to the resulting request sequence across four datasets. Figure 6 reports the test accuracy after each request and the cumulative time/communication costs. F-SOSA-I/U match the accuracy of Retrain while maintaining smaller fluctuations. On SVHN, for instance, Retrain exhibits swings over 5%, whereas F-SOSA-I/U keep variations within 3.96% across all datasets. This robustness stems from client subsampling and Bernoulli sampling, which avoid unnecessary updates when edits do not affect the aggregation set. In terms of efficiency, F-SOSA-I/U reduce cumulative time and communication overhead by 36.66%–53.84% and 21.97%–25.76%, respectively, where the reported communication cost includes all model transfers and Fisher-information uploads. In summary, F-SOSA-I/U deliver model editing with near-optimal accuracy and significantly improved efficiency, showing their practical benefits in dynamic environments. We further replay the same mobility-driven editing sequence on a YOLOv8s object-detection model trained on BDD100K; the corresponding results are reported in the Appendix B-D.

## VII. CONCLUSIONS

We present a principled framework for exact model editing in one-shot FL, enabling both client unlearning and incremental learning. At its core is F-SOSA, a stable learning algorithm that supports efficient post-training edits. Together with the editing algorithms F-SOSA-U and F-SOSA-I, our method achieves statistical exactness while preserving model utility and efficiency. We provide theoretical guarantees and comprehensive experiments demonstrating its effectiveness across standard benchmarks and dynamic edge scenarios.

## REFERENCES

[1] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y. Arcas, “Communication-efficient learning of deep networks from decentralized data,” in 20th International Conference on Artificial Intelligence and Statistics (AISTATS 2017), Fort Lauderdale, FL, 2017.

[2] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, “Federated optimization in heterogeneous networks,” in 3rd Conference on Machine Learning and Systems (MLSys 2020), Austin, TX, 2020.

[3] S. P. Karimireddy, S. Kale, M. Mohri, S. Reddi, S. Stich, and A. T. Suresh, “Scaffold: Stochastic controlled averaging for federated learning,” in 37th International Conference on Machine Learning (ICML 2020). Virtual Conference: PMLR, 2020.

[4] Y. Tao, S. Cui, W. Xu, H. Yin, D. Yu, W. Liang, and X. Cheng, “Byzantine-Resilient Federated Learning at Edge,” IEEE Transactions on Computers, vol. 72, no. 9, pp. 2600–2614, 9 2023.

[5] Y. Tao, S. Chen, C. Zhang, D. Wang, D. Yu, X. Cheng, and F. Dressler, “Private Over-the-Air Federated Learning at Band-Limited Edge,” IEEE Transactions on Mobile Computing, vol. 23, no. 12, pp. 12 444–12 460, 12 2024.

[6] S. Chen, Y. Tao, S. Chen, F. Dressler, Q. Ye, and D. Wang, “Fed-SubMuon: Communication-Efficient Federated LLM Fine-Tuning via Structured Subspace Muon,” in Conference on Empirical Methods in Natural Language Processing (EMNLP 2026). Budapest, Hungary: ACL, 10 2026.

[7] N. Guha, A. Talwalkar, and V. Smith, “One-shot federated learning,” arXiv, cs.LG 1902.11175, 2019.

[8] D. Jhunjhunwala, S. Wang, and G. Joshi, “FedFisher: Leveraging Fisher Information for One-Shot Federated Learning,” in 27th International Conference on Artificial Intelligence and Statistics (AISTATS 2024). Valencia, Spain: PMLR, 2024.

[9] X. Liu, L. Liu, F. Ye, Y. Shen, X. Li, L. Jiang, and J. Li, “Fedlpa: Oneshot federated learning with layer-wise posterior aggregation,” in 38th Conference on Neural Information Processing Systems (NeurIPS 2024), vol. 37. Vancouver, Canada: Curran Associates Inc., 2024.

[10] J. Zhang, C. Chen, B. Li, L. Lyu, S. Wu, S. Ding, C. Shen, and C. Wu, “Dense: Data-free one-shot federated learning,” in 36th Conference on Neural Information Processing Systems (NeurIPS 2022), vol. 35. New Orleans, LA: Curran Associates Inc., 2022.

[11] D. Yu, Z. Zou, S. Chen, Y. Tao, B. Tian, W. Lv, and X. Cheng, “Decentralized Parallel SGD With Privacy Preservation in Vehicular Networks,” IEEE Transactions on Vehicular Technology, vol. 70, no. 6, pp. 5211–5220, 6 2021.

[12] F. Dressler, G. S. Pannu, F. Hagenauer, M. Gerla, T. Higuchi, and O. Altintas, “Virtual Edge Computing Using Vehicular Micro Clouds,” in IEEE International Conference on Computing, Networking and Communications (ICNC 2019). Honolulu, HI: IEEE, 2019.

[13] Z. Zhou, Y. Tao, A. Memedi, C. Lee, S. Ucar, O. Altintas, and F. Dressler, “Optimizing Task Migration Decisions in Vehicular Edge Computing Environments,” in 1st IEEE International Conference on Meta Computing (ICMC 2024). Qingdao, China: IEEE, 6 2024.

[14] G. Liu, X. Ma, Y. Yang, C. Wang, and J. Liu, “FedEraser: Enabling efficient client-level data removal from federated learning models,” in 29th IEEE/ACM International Symposium on Quality of Service (IWQOS 2021). Chiyoda, Japan: IEEE, 2021.

[15] Y. Fraboni, M. Van Waerebeke, K. Scaman, R. Vidal, L. Kameni, and M. Lorenzi, “Sifu: Sequential informed federated unlearning for efficient and provable client unlearning in federated optimization,” in 27th International Conference on Artificial Intelligence and Statistics (AISTATS 2024). Valencia, Spain: PMLR, 2024.

[16] H. Wu, Y. Tao, A. Zhou, S. Chen, F. Dressler, and D. Yu, “Certified Unlearning in Decentralized Federated Learning,” in 45th IEEE International Conference on Computer Communications (INFOCOM 2026). Tokyo, Japan: IEEE, 5 2026, pp. 1–10.

[17] H. Wu, J. Yang, S. Chen, D. Wang, D. Yu, and Y. Tao, “Trajectory-Aware Certified Decentralized Unlearning via SGD Stability,” in 43rd International Conference on Machine Learning (ICML 2026), Poster Session. Seoul, South Korea: PMLR, 7 2026.

[18] C. Wu, S. Zhu, and P. Mitra, “Federated unlearning with knowledge distillation,” arXiv, cs.LG 2201.09441, 2022.

[19] J. Wang, S. Guo, X. Xie, and H. Qi, “Federated unlearning via classdiscriminative pruning,” in 31st ACM Web Conference (WWW 2022). Lyon, France: ACM, 2022.

[20] L. Wu, S. Guo, J. Wang, Z. Hong, J. Zhang, and Y. Ding, “Federated Unlearning: Guarantee the Right of Clients to Forget,” IEEE Network, vol. 27, no. 5, 2013.

[21] R. Shokri, M. Stronati, C. Song, and V. Shmatikov, “Membership inference attacks against machine learning models,” in 38th IEEE Symposium on Security and Privacy (S&P 2017). San Jose, CA: IEEE, 2017.

[22] M. Song, Z. Wang, Z. Zhang, Y. Song, Q. Wang, J. Ren, and H. Qi, “Analyzing user-level privacy attack against federated learning,” IEEE Journal on Selected Areas in Communications, vol. 38, no. 10, 2020.

[23] Z. Wang, M. Song, Z. Zhang, Y. Song, Q. Wang, and H. Qi, “Beyond inferring class representatives: User-level privacy leakage from federated learning,” in 38th IEEE Conference on Computer Communications (INFOCOM 2019). Paris, France: IEEE, 2019.

[24] L. Zhu, Z. Liu, and S. Han, “Deep Leakage from Gradients,” in 33rd Conference on Neural Information Processing Systems (NeurIPS 2019). Vancouver, Canada: Curran Associates Inc., 2019.

[25] C.-L. Wang, Q. Li, Z. Xiang, Y. Cao, and D. Wang, “Towards Lifecycle Unlearning Commitment Management: Measuring Sample-level Unlearning Completeness,” in 34th USENIX Security Symposium (USENIX Security 2025). Seattle, WA: USENIX, 8 2025, pp. 6481–6500.

[26] J. Chen, Z. Lin, W. Lin, W. Shi, X. Yin, and D. Wang, “Fedmua: Exploring the vulnerabilities of federated learning to malicious unlearning attacks,” IEEE Transactions on Information Forensics and Security, vol. 20, pp. 1665–1678, 2025.

[27] J. Li, Y. Tao, B. He, and M. Ding, “Not All Change Is Damage: Update-Matched Unlearning for Large Language Models,” arXiv, cs.LG 2610.21230, 10 2026. [Online]. Available: https: //arxiv.org/abs/2610.21230

[28] Y. Liu, L. Xu, X. Yuan, C. Wang, and B. Li, “The right to be forgotten in federated learning: An efficient realization with rapid retraining,” in 41st IEEE International Conference on Computer Communications (INFOCOM 2022). Virtual Conference: IEEE, 2022.

[29] Y. Tao, C.-L. Wang, M. Pan, D. Yu, X. Cheng, and D. Wang, “Communication Efficient and Provable Federated Unlearning,” Proceedings of the VLDB Endowment, vol. 17, no. 5, 2024.

[30] Y. F. Bakman, D. N. Yaldiz, Y. H. Ezzeldin, and S. Avestimehr, “Federated orthogonal training: Mitigating global catastrophic forgetting in continual federated learning,” in 12th International Conference on Learning Representations (ICLR 2024). Vienna, Austria: ICLR, 2024.

[31] Z. Zhang, B. Guo, W. Sun, Y. Liu, and Z. Yu, “Cross-FCL: Toward a Cross-Edge Federated Continual Learning Framework in Mobile Edge Computing Systems,” IEEE Transactions on Mobile Computing, vol. 23, no. 1, 2024.

[32] S. Yue, J. Ren, J. Xin, S. Lin, and J. Zhang, “Inexact-ADMM based federated meta-learning for fast and continual edge learning,” in 22nd ACM International Symposium on Theory, Algorithmic Foundations, and Protocol Design for Mobile Networks and Mobile Computing (MobiHoc 2021). Shanghai, China: ACM, 2021.

[33] M. Ding, K. Ji, D. Wang, and J. Xu, “Understanding Forgetting in Continual Learning with Linear Regression,” in 41st International Conference on Machine Learning (ICML 2024). Vienna, Austria: PMLR, 7 2024, pp. 10 978–11 001.

[34] S. Babakniya, Z. Fabian, C. He, M. Soltanolkotabi, and S. Avestimehr, “A data-free approach to mitigate catastrophic forgetting in federated class incremental learning for vision tasks,” in 37th Conference on Neural Information Processing Systems (NeurIPS 2023). New Orleans, LA: Curran Associates Inc., 2023.

[35] J. Dong, L. Wang, Z. Fang, G. Sun, S. Xu, X. Wang, and Q. Zhu, “Federated class-incremental learning,” in 33rd IEEE/CVF conference on computer vision and pattern recognition (CVPR 2022). New Orleans, LA: IEEE, 2022.

[36] J. Yoon, W. Jeong, G. Lee, E. Yang, and S. J. Hwang, “Federated continual learning with weighted inter-client transfer,” in 38th International Conference on Machine Learning (ICML 2021), vol. 139. Virtual Conference: PMLR, 2021.

[37] W. Huang, M. Ye, and B. Du, “Learn from others and be yourself in heterogeneous federated learning,” in 33rd IEEE/CVF conference on computer vision and pattern recognition (CVPR 2022). New Orleans, LA: IEEE, 2022.

[38] Z. Ma, Y. Lu, W. Li, and S. Cui, “EFL: elastic federated learning on non-IID data,” in 1st Conference on Lifelong Learning Agents (CoLLAs 2022). Montreal, Canada: PMLR, 2022.´

[39] R. Zhang, Q. Xu, J. Yao, Y. Zhang, Q. Tian, and Y. Wang, “Federated domain generalization with generalization adjustment,” in 34th IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2023). Vancouver, Canada: IEEE, 2023.

[40] Y. Ruan, X. Zhang, S.-C. Liang, and C. Joe-Wong, “Towards flexible device participation in federated learning,” in 24th International Conference on Artificial Intelligence and Statistics (AISTATS 2021). Virtual Conference: PMLR, 2021.

[41] Z.-L. Chang, D.-J. Han, S. Hosseinalipour, M. Chiang, and C. G. Brinton, “Federated Learning with Dynamic Client Arrival and Departure: Convergence and Rapid Adaptation via Initial Model Construction,” arXiv, cs.LG 2410.05662, 2024.

[42] E. Ullah, T. Mai, A. Rao, R. A. Rossi, and R. Arora, “Machine unlearning via algorithmic stability,” in 34th Annual Conference on Learning Theory (COLT 2021). Boulder, CO: PMLR, 2021.

[43] M. Al-Shedivat, J. Gillenwater, E. Xing, and A. Rostamizadeh, “Federated learning via posterior averaging: A new perspective and practical algorithms,” in 9th International Conference on Learning Representations (ICLR 2021). Virtual Conference: OpenReview.net, 5 2021.

[44] S. Chen, Y. Yuan, Y. Tao, T. Wang, Z. Cai, and D. Yu, “Adaptive pruningbased Newton’s method for distributed learning,” Theoretical Computer Science, vol. 1026, p. 114987, 2 2025.

[45] S. Arora, S. Du, W. Hu, Z. Li, and R. Wang, “Fine-grained analysis of optimization and generalization for overparameterized two-layer neural networks,” in 36th International Conference on Machine Learning (ICML 2019), vol. 97. Long Beach, CA: PMLR, 2019.

[46] X. Hu, S. Li, and Y. Liu, “Generalization bounds for federated learning: Fast rates, unparticipating clients and unbounded losses,” in 11th International Conference on Learning Representations (ICLR 2023). Kigali, Rwanda: OpenReview.net, 2023.

[47] Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, 1998.

[48] Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu, and A. Y. Ng, “Reading digits in natural images with unsupervised feature learning,” in NIPS 2011 Workshop on Deep Learning and Unsupervised Feature Learning, Granada, Spain, 2011.

[49] A. Krizhevsky, “Learning multiple layers of features from tiny images,” University of Toronto, Technical Report TR-2009, 2009.

[50] S. Caldas, S. M. K. Duddu, P. Wu, T. Li, J. Konecnˇ y, B. Mcmahan,\` V. Smith, and A. Talwalkar, “Leaf: A benchmark for federated settings,” arXiv, cs.LG 1812.01097, 2018.

[51] P. A. Lopez, E. Wiessner, M. Behrisch, L. Bieker-Walz, J. Erdmann, Y.-P. Flotterod, R. Hilbrich, L. Lucken, J. Rummel, and P. Wagner, “Microscopic Traffic Simulation using SUMO,” in 21st IEEE International Conference on Intelligent Transportation Systems (ITSC 2018). Maui, HI: IEEE, 2018.

[52] L. Codeca, R. Frank, S. Faye, and T. Engel, “Luxembourg SUMO´ Traffic (LuST) Scenario: Traffic Demand Evaluation,” IEEE Intelligent Transportation Systems Magazine, vol. 9, no. 2, 2017.

[54] F. Yu, H. Chen, X. Wang, W. Xian, Y. Chen, F. Liu, V. Madhavan, and T. Darrell, “Bdd100k: A diverse driving dataset for heterogeneous multitask learning,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Seattle, WA: IEEE, 6 2020, pp. 2633–2642.

[53] G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics yolov8,” 2023. [Online]. Available: https://github.com/ultralytics/ultralytics

## APPENDIX A PROOFS OF AUXILIARY LEMMAS

Proof of Lemma 4. Part 1) is directly from [8, Lemma 10]. For Part 2), we first apply convexity of the squared norm to bound $\lVert \bar { \mathbf { W } } _ { \mathcal { P } } - \mathbf { W } ^ { ( 0 ) } \rVert _ { 2 } ^ { 2 }$ by the average $\lVert \tilde { \boldsymbol { W } } _ { i } - \boldsymbol { W } ^ { ( 0 ) } \rVert _ { 2 } ^ { 2 }$ over $i \in \mathcal { P }$

$$
{ \left\| { \overline { { W } } _ { \mathcal { P } } } - W ^ { ( 0 ) } \right\} | _ { 2 } ^ { 2 } \leq \frac { 1 } { \rho M } \sum _ { l \in \mathcal { P } } { \left\| { \tilde { W } _ { l } } - W ^ { ( 0 ) } \right\| _ { 2 } ^ { 2 } } ,
$$

where each summation term can be bounded using Part 1. Expressing the resulting sum via the block-diagonal Gram matrix dia $\mathfrak { u g } ( \mathbf { H } _ { 1 } ^ { \infty } , \ldots , \mathbf { H } _ { M } ^ { \infty } )$ ), we have

$$
\begin{array} { r l } & { \| \overline { { W } } _ { \mathcal { P } } - W ^ { ( 0 ) } \| _ { 2 } ^ { 2 } } \\ & { \leq \frac { 2 \sum _ { l \in \mathcal { P } } y _ { l } ^ { ( 1 ) } ( \mathbf { H } _ { \mathfrak { x } } ^ { ( \cdot ) - 1 } ) y _ { l } } { \rho M } + O ( \frac { N ^ { 2 } \sigma ^ { 2 } } { \lambda _ { l } ^ { ( 3 ) / 2 } } ) + \frac { \mathrm { P o l y } ( N , \lambda _ { 0 } ^ { - 1 } , \delta ^ { - 1 } ) } { m ^ { \frac { 3 } { 2 } } \sigma } } \\ & { \leq \frac { 2 \sum _ { l = 1 } ^ { \frac { 1 } { M } } y _ { l } ^ { ( 1 ) } ( \mathbf { H } _ { \mathfrak { x } } ^ { ( \cdot ) - 1 } ) y _ { l } } { \rho M } + O ( \frac { N ^ { 2 } \sigma ^ { 2 } } { \lambda _ { l } ^ { ( 3 ) / 2 } } ) + \frac { \mathrm { P o l y } ( N , \lambda _ { 0 } ^ { - 1 } , \delta ^ { - 1 } ) } { m ^ { \frac { 1 } { 9 } } \sigma } } \\ & { = \frac { 2 y ^ { \top } ( M \mathrm { d a g } ( \mathbf { H } _ { \mathfrak { x } } ^ { ( \cdot ) } , \mathbf { H } _ { \mathfrak { x } } ^ { ( \cdot ) , \cdot } , \cdots , \mathbf { H } _ { \mathfrak { x } } ^ { ( \cdot ) } ) ) ^ { - 1 } y } { \rho } } \\ & { \quad + O ( \frac { N ^ { 2 } \sigma ^ { 2 } } { \lambda _ { 0 } ^ { 3 / 2 } } ) + \frac { \mathrm { P o l y } ( N , \lambda _ { 0 } ^ { - 1 } , \delta ^ { - 1 } ) } { m ^ { \frac { 1 } { 9 } } \sigma } } \\ &  \leq \frac { 2 y ^ { \top } ( \mathbf { H } ^ { ( \cdot ) - 1 } ) y } { \rho } + O ( \frac { N ^ { 2 } \sigma ^ { 2 } }  \lambda _   \end{array}
$$

The proof concludes by the setting of m and $\sigma .$

Proof of Lemma 5. We bound $\lVert \tilde { f } ( \bar { \pmb { W } } _ { \mathcal { P } } , \pmb { x } _ { i j } ) - \tilde { \pmb { y } } \rVert _ { 2 } ^ { 2 }$ by first expanding the difference as a sum of squared inner products and then applying $\| \phi _ { a } ( \pmb { W } , \pmb { x } ) \| _ { 2 } \leq 1$ from Lemma 1.

$$
\begin{array} { c } { \displaystyle | | \tilde { f } ( \bar { W } _ { \mathcal { P } } , x _ { i j } ) - \tilde { y } | | _ { 2 } ^ { 2 } = \displaystyle \sum _ { i = 1 } ^ { M } \sum _ { j = 1 } ^ { n } ( \phi _ { a } ( \tilde { W } _ { i } , x _ { i j } ) ^ { \top } ( \bar { W } _ { \mathcal { P } } - \tilde { W } _ { i } ) ) ^ { 2 } \medskip } \\ { \displaystyle \leq \sum _ { i = 1 } ^ { M } \sum _ { j = 1 } ^ { n } \| \phi _ { a } ( \tilde { W } _ { i } , x _ { i j } ) \| _ { 2 } ^ { 2 } \| \bar { W } _ { \mathcal { P } } - \tilde { W } _ { i } \| _ { 2 } ^ { 2 } \leq \sum _ { i = 1 } ^ { M } \sum _ { j = 1 } ^ { n } \| \bar { W } _ { \mathcal { P } } - \tilde { W } _ { i } \| _ { 2 } ^ { 2 } } \end{array}
$$

Triangle inequality gives a further bound via $\lVert \tilde { W } _ { i } - W ^ { ( 0 ) } \rVert _ { 2 }$ and $\lVert \bar { \boldsymbol { W } } _ { \mathcal { P } } - \boldsymbol { W } ^ { ( 0 ) } \rVert _ { 2 }$

$$
\begin{array} { r l } & { \quad \| \tilde { f } ( \bar { W } _ { \mathcal { P } } , \pmb { x } _ { i j } ) - \tilde { \pmb { y } } \| _ { 2 } ^ { 2 } } \\ & { \le 2 \displaystyle \sum _ { i = 1 } ^ { M } \displaystyle \sum _ { j = 1 } ^ { n } ( \| \bar { \pmb { W } } _ { \mathcal { P } } - \pmb { W } ^ { ( 0 ) } \| _ { 2 } ^ { 2 } + \| \tilde { \pmb { W } } _ { i } - \pmb { W } ^ { ( 0 ) } \| _ { 2 } ^ { 2 } ) } \end{array}
$$

By Lemma 4 and the setting of $m , \eta , \delta ,$ , we obtain

$$
\begin{array} { r l } & { \quad \| \tilde { f } ( \bar { W } _ { \mathcal { P } } , \mathbf { x } _ { i j } ) - \tilde { y } \| _ { 2 } ^ { 2 } } \\ & { \leq \displaystyle \sum _ { i = 1 } ^ { M } \sum _ { j = 1 } ^ { n } O ( \frac { y ^ { \top } ( \mathbf { H } ^ { \infty } ) ^ { - 1 } y } { \rho } + y _ { i } ^ { \top } ( \mathbf { H } _ { i } ^ { \infty } ) ^ { - 1 } y _ { i } ) } \\ & { = O ( N ( \frac { y ^ { \top } ( \mathbf { H } ^ { \infty } ) ^ { - 1 } y } { \rho } + y ^ { \top } \frac { \left( \mathrm { d a g } \left( \mathbf { H } _ { 1 } ^ { \infty } , \mathbf { H } _ { 2 } ^ { \infty } , \cdots , \mathbf { H } _ { M } ^ { \infty } \right) \right) ^ { - 1 } y } { M } ) ) } \\ & { = O \left( \frac { N } { \rho } y ^ { \top } ( \mathbf { H } ^ { \infty } ) ^ { - 1 } y \right) , } \end{array}
$$

Proof of Lemma 6. By Assumption 4 and Cauchy’s interlacing theorem, we have $\begin{array} { r } { \lambda _ { \operatorname* { m i n } } ( \mathbf { H } _ { \mathcal { P } } ^ { \infty } ) \geq \lambda _ { 0 } } \end{array}$ . For any entry (k, l),

$$
\mathbf { H } _ { \mathcal { P } } ^ { ( 0 ) } ( k , l ) = \frac { \pmb { x } _ { k ^ { \prime } k ^ { \prime \prime } } ^ { \top } \pmb { x } _ { l ^ { \prime } l ^ { \prime \prime } } } { m } \sum _ { r = 1 } ^ { m } \mathbb { 1 } \{ \pmb { x } _ { k ^ { \prime } k ^ { \prime \prime } } ^ { \top } \pmb { w } _ { r } ^ { ( 0 ) } \geq 0 \} \mathbb { 1 } \{ \pmb { x } _ { l ^ { \prime } l ^ { \prime \prime } } ^ { \top } \pmb { w } _ { r } ^ { ( 0 ) } \geq 0 \} ,
$$

where $k ^ { \prime } = i _ { \lceil k / n \rceil } , k ^ { \prime \prime } = k - \lceil k / n \rceil + 1 , l ^ { \prime } = i _ { \lceil l / n \rceil } , l ^ { \prime \prime } = l - \lceil l / n \rceil + 1$ . Thus $\mathbf { H } _ { \mathcal { P } } ^ { ( 0 ) } ( k , l )$ is an average of m bounded independent variables. Applying Hoeffding’s inequality yields

$$
| \mathbf { H } _ { \mathcal { P } } ^ { ( 0 ) } ( k , l ) - \mathbf { H } _ { \mathcal { P } } ^ { \infty } ( k , l ) | \leq 4 \sqrt { \log ( \rho N / \delta ) / m } .
$$

for all (k, l) with probability at least 1 − δ. Therefore,

$$
\begin{array} { r } { \| \mathbf { H } _ { \mathcal { P } } ^ { ( 0 ) } - \mathbf { H } _ { \mathcal { P } } ^ { \infty } \| _ { 2 } ^ { 2 } \leq \| \mathbf { H } _ { \mathcal { P } } ^ { ( 0 ) } - \mathbf { H } _ { \mathcal { P } } ^ { \infty } \| _ { F } ^ { 2 } \leq = 1 6 ( \rho N ) ^ { 2 } \log ( \rho N / \delta ) / m . } \end{array}
$$

Setting $\begin{array} { r } { m = \Omega \left( \frac { ( \rho N ) ^ { 2 } \log ( \rho N / \delta ) } { \lambda _ { 0 } ^ { 2 } } \right) } \end{array}$ completes the proof.

Proof of Lemma 7. The (k, l)-th entry of $\tilde { \mathbf { H } } _ { \mathcal { P } }$ is given by

$$
\frac { \pmb { x } _ { k ^ { \prime } k ^ { \prime \prime } } ^ { \top } \pmb { x } _ { k ^ { \prime } k ^ { \prime \prime } } } { m } \sum _ { r = 1 } ^ { m } \mathbb { 1 } \{ \pmb { x } _ { k k ^ { \prime \prime } } ^ { \top } \pmb { \tilde { w } } _ { k ^ { \prime } , r } \geq 0 \} \cdot \mathbb { 1 } \{ \pmb { x } _ { l l ^ { \prime \prime } } ^ { \top } \pmb { \tilde { w } } _ { l ^ { \prime } , r } \geq 0 \} ,
$$

where $k ^ { \prime } = i _ { \left[ k / n \right] } , k ^ { \prime \prime } = k - \left[ k / n \right] + 1 , l ^ { \prime } = i _ { \left[ l / n \right] } , l ^ { \prime \prime } = l - \left[ l / n \right] + 1$ . We have

$$
\begin{array} { r l } & { \mathbb { E } [ \left| \hat { \mathbf { H } } _ { p } ( k , l ) - \mathbf { H } _ { p } ^ { \mathrm { i } } ( \cdot , l ) \right| ] } \\ & { = \mathbb { E } [ \frac { \left| \hat { \mathbf { H } } _ { p } ^ { x } ( \cdot , l ^ { x } , \hat { \mathbf { H } } _ { p } , \cdot , l ^ { x } ) \right| } { m } \sum _ { \gamma = - 1 } ^ { m } \mathbb { 1 } \left\{ \mathbb { I } \{ ( \mathbf { T } _ { s } ^ { l } , l ^ { x } , \hat { \mathbf { H } } _ { k } , \gamma , \geq 0 ) \} \mathbb { I } \{ ( x _ { \gamma } ^ { l } , \hat { \mathbf { H } } _ { p } , \gamma > 0 ) \} \right. } \\ & { \qquad + \left. \mathbb { E } [ \left| \hat { \mathbf { a } } _ { p } ^ { k } ( \cdot , r _ { s } ^ { m } , 0 ) \right| ] \mathbb { E } [ \hat { \mathbf { H } } _ { p } ^ { x } , \hat { \mathbf { H } } _ { p } ^ { x } , \gamma , \geq 0 ) \right\} ] } \\ & { \leq \mathbb { E } [ \frac { 1 } { m } \sum _ { \gamma = - 1 } ^ { m } \mathbb { 1 } \{ \mathbb { I } \} \{ \mathbb { I } \} _ { p } ^ { x } \mathbb { I } _ { p } ^ { x } , \mathbb { I } _ { p } ^ { x } , \mathbb { I } _ { p } ^ { x } ] \leq 0 \} \mathbb { I } \{ ( \mathbf { T } _ { s } ^ { l } , \hat { \mathbf { H } } _ { p } , \gamma > 0 ) \} } \\ & { \qquad + \mathbb { E } [ \frac { 1 } { m } \sum _ { \gamma = - 1 } ^ { m } \mathbb { I } \{ \mathbb { I } \} \{ \hat { \mathbf { I } } _ { p } ^ { x } \} \mathbb { I } _ { p } ^ { x } , \mathbb { I } _ { p } ^ { x } , \mathbb { I } _ { p } ^ { x } ] \leq 0 \} ] \mathbb { I } _ { p } ^ { - 1 } } \\ &  \qquad + \mathbb { I } \{ \hat { \mathbf { H } } _ { p } ^  x \end{array}
$$

Then, we bound the Frobenius norm,

$$
\begin{array} { r l } & { \mathbb { E } [ \| \tilde { \mathbf { H } } _ { \mathcal { P } } - \mathbf { H } _ { \mathcal { P } } ^ { ( 0 ) } \| _ { \mathrm { F } } ] \le \mathbb { E } [ \displaystyle \sum _ { k , l \in \mathcal { P } } | \tilde { \mathbf { H } } _ { \mathcal { P } } ( k , l ) - \mathbf { H } _ { \mathcal { P } } ^ { ( 0 ) } ( k , l ) | ] } \\ & { \qquad \le 4 ( \rho N ) ^ { 2 } R _ { 0 } / ( \sqrt { 2 \pi } \sigma ) . } \end{array}
$$

Using Markov’s inequality, with probability $1 - \delta ,$ , we get

$$
\| \tilde { \mathbf { H } } _ { \mathcal { P } } - \mathbf { H } _ { \mathcal { P } } ^ { ( 0 ) } \| _ { \mathrm { F } } \leq \mathbb { E } [ \| \tilde { \mathbf { H } } _ { \mathcal { P } } - \mathbf { H } _ { \mathcal { P } } ^ { ( 0 ) } \| _ { \mathrm { F } } ] / \delta \leq 4 ( \rho N ) ^ { 2 } R _ { 0 } / ( \sqrt { 2 \pi } \delta \sigma ) .
$$

Thus,

$$
\| \tilde { \mathbf { H } } _ { \mathcal { P } } - \mathbf { H } _ { \mathcal { P } } ^ { ( 0 ) } \| _ { 2 } \le \| \tilde { \mathbf { H } } _ { \mathcal { P } } - \mathbf { H } _ { \mathcal { P } } ^ { ( 0 ) } \| _ { \mathrm { F } } \le 4 ( \rho N ) ^ { 2 } R _ { 0 } / ( \sqrt { 2 \pi } \delta \sigma ) .
$$

By Lemma 6 and the settng of $m .$ , we have

$$
\lambda _ { \operatorname* { m i n } } ( \tilde { \mathbf { H } } _ { \mathcal { P } } ) \geq \lambda _ { \operatorname* { m i n } } ( \mathbf { H } _ { \mathcal { P } } ^ { ( 0 ) } ) - 4 ( \rho N ) ^ { 2 } R _ { 0 } / ( \sqrt { 2 \pi } \delta \sigma ) \geq \lambda _ { 0 } / 2 .
$$

To get upper bound of $\lambda _ { \operatorname* { m a x } } \left( \tilde { \mathbf { H } } _ { \mathcal { P } } \right)$ , we have

$$
\begin{array} { r l } & { \quad \| \tilde { \mathbf { H } } _ { \mathcal { P } } \| _ { 2 } = \| \tilde { \mathbf { A } } _ { \mathcal { P } } \tilde { \mathbf { A } } _ { \mathcal { P } } ^ { \top } \| _ { 2 } = \| \tilde { \mathbf { A } } _ { \mathcal { P } } ^ { \top } \tilde { \mathbf { A } } _ { \mathcal { P } } \| _ { 2 } } \\ & { { = } \| \sum _ { i \in \mathcal { P } } \sum _ { j = 1 } ^ { n } \phi _ { a } ( \tilde { W } _ { i } , x _ { i j } ) \phi _ { a } ( \tilde { W } _ { i } , x _ { i j } ) ^ { \top } \| _ { 2 } } \\ & { { \leq } \sum _ { i \in \mathcal { P } } \sum _ { j = 1 } ^ { n } \| \phi _ { a } \left( \tilde { W } _ { i } , x _ { i j } \right) \phi _ { a } \left( \tilde { W } _ { i } , x _ { i j } \right) ^ { \top } \| _ { 2 } } \\ & { { \leq } \sum _ { i \in \mathcal { P } } \sum _ { j = 1 } ^ { n } \| \phi _ { a } \left( \tilde { W } _ { i } , x _ { i j } \right) \| _ { 2 } ^ { 2 } { \leq } \rho N , } \end{array}
$$

where the last inequality uses Lemma 1.

Proof of Lemma 8. By [8, Lemma 1], $\bar { \boldsymbol { W } } _ { \mathcal { P } } ^ { \mathrm { F } }$ is equivalent to the solution obtained via a gradient descent (GD) procedure. Let $\tilde { W } _ { \mathcal { P } } ^ { [ t ] }$ be the t-th iterate of the GD procedure with the initialization $\tilde { W } _ { \mathcal { P } } ^ { [ 0 ] } = \bar { W } _ { \mathcal { P } }$ . Define the vectorized output $\tilde { f } ( t ) \in \mathbb { R } ^ { N }$ by stacking $\tilde { f } ( \tilde { W } _ { \mathcal { P } } ^ { [ t ] } , x _ { i j } )$ across all $i \in [ M ] , j \in [ n ]$ :

$$
\tilde { f } ( t ) : = [ \tilde { f } ( \tilde { W } _ { \mathcal { P } } ^ { [ t ] } , x _ { 1 1 } ) , \tilde { f } ( \tilde { W } _ { \mathcal { P } } ^ { [ t ] } , x _ { 1 2 } ) , \cdots , \tilde { f } ( \tilde { W } _ { \mathcal { P } } ^ { [ t ] } , x _ { M n } ) ] ^ { \top } \in \mathbb { R } ^ { N } .
$$

Thus, GD produces the recurrence

$$
\tilde { \mathbf { W } } _ { \mathcal { P } } ^ { [ t ] } = \tilde { \mathbf { W } } _ { \mathcal { P } } ^ { [ t - 1 ] } - \frac { \tilde { \eta } } { n } \tilde { \mathbf { A } } _ { \mathcal { P } } ^ { \top } ( \tilde { f } ( t - 1 ) - \tilde { \pmb { y } } ) .\tag{19}
$$

which implies the prediction evolution

$$
\tilde { f } ( t ) = \tilde { f } ( t - 1 ) - \frac { \tilde { \eta } } { n } \tilde { \mathbf { H } } _ { \mathcal { P } } ( \tilde { f } ( t - 1 ) - \tilde { \mathbf { y } } ) .\tag{20}
$$

Unrolling the recurrence, we obtain for any t,

$$
\tilde { f } ( t ) - \tilde { \pmb { y } } = ( \mathbf { I } - \frac { \tilde { \eta } } { n } \tilde { \mathbf { H } } _ { \mathcal { P } } ) ( \tilde { f } ( t - 1 ) - \tilde { \pmb { y } } ) = ( \mathbf { I } - \frac { \tilde { \eta } } { n } \tilde { \mathbf { H } } _ { \mathcal { P } } ) ^ { t } ( \tilde { f } ( 0 ) - \tilde { \pmb { y } } ) .\tag{21}
$$

The distance between the final global model and the initialization is expressed as a gradient series:

$$
\bar { W } _ { \mathcal { P } } ^ { \mathrm { F } } - \bar { W } _ { \mathcal { P } } = - \frac { \tilde { \eta } } { n } \sum _ { t = 0 } ^ { \infty } \tilde { \mathbf { A } } _ { \mathcal { P } } ^ { \top } ( \mathbf { I } - \frac { \tilde { \eta } } { n } \tilde { \mathbf { H } } _ { \mathcal { P } } ) ^ { t } ( \tilde { f } ( 0 ) - \tilde { y } ) .\tag{22}
$$

Let $\begin{array} { r } { \mathbf { U } : = \frac { \tilde { \eta } } { n } ( \mathbf { I } - \frac { \tilde { \eta } } { n } \tilde { \mathbf { H } } _ { \mathcal { P } } ) ^ { t } } \end{array}$ be the geometric series matrix. Then

$$
\begin{array} { r l } & { \| \bar { \boldsymbol { W } } _ { \mathcal { P } } ^ { \mathrm { F } } - \bar { \boldsymbol { W } } _ { \mathcal { P } } \| _ { 2 } ^ { 2 } = \| \tilde { \mathbf { A } } _ { \mathcal { P } } ^ { \top } \mathbf { U } ( \tilde { f } ( 0 ) - \tilde { \pmb { y } } ) \| _ { 2 } ^ { 2 } } \\ & { \qquad \leq \left( \tilde { f } ( 0 ) - \tilde { \pmb { y } } \right) ^ { \top } \mathbf { U } \tilde { \mathbf { H } } _ { \mathcal { P } } \mathbf { U } \left( \tilde { f } ( 0 ) - \tilde { \pmb { y } } \right) . } \end{array}
$$

Using eigen-decomposition $\tilde { \mathbf { H } } _ { \mathcal { P } } = \mathbf { V } \mathbf { D } \mathbf { V } ^ { \top }$ , we obtain

$$
\mathbf { U } = { \frac { \tilde { \eta } } { n } } \mathbf { V } \sum _ { t = 1 } ^ { \infty } ( \mathbf { I } - { \frac { \tilde { \eta } } { n } } \mathbf { D } ) ^ { t } \mathbf { V } ^ { \top } = \mathbf { V } \mathbf { D } ^ { - 1 } \mathbf { V } ^ { \top } = \tilde { \mathbf { H } } _ { \mathcal { P } } ^ { - 1 } ,
$$

The squared distance becomes

$$
\| \bar { \boldsymbol { W } } _ { \mathcal { P } } ^ { \mathrm { F } } - \bar { \boldsymbol { W } } _ { \mathcal { P } } \| _ { 2 } ^ { 2 } \le ( \tilde { f } ( 0 ) - \tilde { \pmb { y } } ) ^ { \top } \tilde { \mathbf { H } } _ { \mathcal { P } } ^ { - 1 } ( \tilde { f } ( 0 ) - \tilde { \pmb { y } } ) \le \frac { 2 } { \lambda _ { 0 } } \| \tilde { f } ( 0 ) - \tilde { \pmb { y } } \| _ { 2 } ^ { 2 } ,
$$

where the last inequality is due to Lemma 7. Finally, Lemma 5 bounds $\lVert \tilde { f } ( 0 ) - \tilde { y } \rVert _ { 2 } ^ { 2 }$ , which leads to the desired results. Proof of Lemma 9. By (22), for each neuron r, we have,

$$
\begin{array} { r l } & { \quad \| \bar { w } _ { \mathcal { P } , r } ^ { \mathrm { F } } - \bar { w } _ { \mathcal { P } , r } \| _ { 2 } } \\ & { = \| \frac { \bar { \eta } } { n } \tilde { \mathbf { A } } _ { \mathcal { P } } [ ( r - 1 ) p + 1 : r p ] ^ { \top } ( \mathbf { I } - \frac { \tilde { \eta } } { n } \tilde { \mathbf { H } } _ { \mathcal { P } } ) ^ { t } ( \tilde { f } ( 0 ) - \tilde { \pmb { y } } ) \| _ { 2 } } \\ & { \le \displaystyle \frac { 2 } { \lambda _ { 0 } } \| \tilde { \mathbf { A } } _ { \mathcal { P } } [ ( r - 1 ) p + 1 : r p ] ^ { \top } \| _ { 2 } \| \tilde { f } ( 0 ) - \tilde { \pmb { y } } \| _ { 2 } } \\ & { \le \displaystyle \frac { 2 } { \lambda _ { 0 } } \| \tilde { \mathbf { A } } _ { \mathcal { P } } [ ( r - 1 ) p + 1 : r p ] ^ { \top } \| _ { \mathrm { F } } \| \tilde { f } ( 0 ) - \tilde { \pmb { y } } \| _ { 2 } } \\ & { \le \displaystyle \frac { 2 \sqrt { \rho N } } { \lambda _ { 0 } \sqrt { m } } \| \tilde { f } ( 0 ) - \tilde { \pmb { y } } \| _ { 2 } , } \end{array}
$$

where we use the notation $A [ x : y ]$ to denote the submatrix of A having row numbers x to $y ,$ and the last inequality follows from the definition of $\tilde { \mathbf { A } } _ { \mathcal { P } }$ and the data normalization assumption (Assumption 2). Combing with Lemma (5), we completes the proof. □

Proof of Lemma 10. Each mismatch requires a sign flip between $\bar { w } _ { \mathcal { P } , r } ^ { \mathrm { F } }$ or $\tilde { \pmb { w } } _ { i , r }$ and the shared initialization ${ \pmb w } _ { r } ^ { ( 0 ) }$

$$
\begin{array} { r l } & { \displaystyle \mathbb { E } [ | S _ { i j } ^ { \Gamma } | ] = \sum _ { r = 1 } ^ { m } \mathbb { P } ( \mathbb { 1 } \{ { \boldsymbol x } _ { i j } ^ { \top } \bar { { \boldsymbol w } } _ { \mathcal { P } , r } ^ { \mathrm { F } } \geq 0 \} \neq \mathbb { 1 } \{ { \boldsymbol x } _ { i j } ^ { \top } \tilde { { \boldsymbol w } } _ { i , r } \geq 0 \} ) } \\ & { \displaystyle \leq \sum _ { r = 1 } ^ { m } \mathbb { P } ( \{ \mathbb { 1 } \{ { \boldsymbol x } _ { i j } ^ { \top } \bar { { \boldsymbol w } } _ { \mathcal { P } , r } ^ { \mathrm { F } } \geq 0 \} \neq \mathbb { 1 } \{ { \boldsymbol x } _ { i j } ^ { \top } { \boldsymbol w } _ { r } ^ { ( 0 ) } \geq 0 \} \} } \\ & { ~ \cup \{ \mathbb { 1 } \{ { \boldsymbol x } _ { i j } ^ { \top } \tilde { { \boldsymbol w } } _ { i , r } \geq 0 \} \neq \mathbb { 1 } \{ { \boldsymbol x } _ { i j } ^ { \top } { \boldsymbol w } _ { r } ^ { ( 0 ) } \geq 0 \} \} ) } \end{array}
$$

$$
\begin{array} { r l } & { \leq \displaystyle \sum _ { r = 1 } ^ { m } [ \mathbb { P } ( \{ \mathbb { 1 } \{ \pmb { x } _ { i j } ^ { \top } \bar { \pmb { w } } _ { \mathcal { P } , r } ^ { \top } \geq 0 \} \neq \mathbb { 1 } \{ \pmb { x } _ { i j } ^ { \top } \pmb { w } _ { r } ^ { ( 0 ) } \geq 0 \} \} ) } \\ & { ~ + \mathbb { P } ( \{ \mathbb { 1 } \{ \pmb { x } _ { i j } ^ { \top } \tilde { \pmb { w } } _ { i , r } \geq 0 \} \neq \mathbb { 1 } \{ \pmb { x } _ { i j } ^ { \top } \pmb { w } _ { r } ^ { ( 0 ) } \geq 0 \} \} ) ] } \\ & { \leq \displaystyle \sum _ { r = 1 } ^ { m } ( \mathbb { P } ( \mathcal { E } _ { i j r } ( R _ { 0 } + R _ { 1 } ) ) + \mathbb { P } ( \mathcal { E } _ { i j r } ( R _ { 0 } ) ) ) \leq \frac { 4 m ( R _ { 0 } + R _ { 1 } ) } { \sqrt { 2 \pi } \sigma } . } \end{array}
$$

By applying Markov’s inequality with failure probability $\delta / N$ and taking a union bound over all $( i , j )$ pairs, we obtain the desired bound. □

Proof of Lemma 13. By [46, Lemma 2], sup $\rangle _ { f \in \mathcal { F } } \{ L _ { T } ^ { f , \ell } - L _ { S } ^ { f , \ell } \}$ is less than

$$
2 \mathbb { E } _ { \{ D _ { i } , \iota _ { i } \} } [ \operatorname* { s u p } _ { f \in \mathcal { F } } \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \iota _ { i } \mathbb { E } _ { z _ { i } \sim D _ { i } } [ \ell ( f ( W , \boldsymbol { x } _ { i } ) , y _ { i } ) ] ] + c \sqrt { \frac { \log \frac { 2 } { \delta } } { 2 P } } .
$$

Let $\tilde { \ell } ( W , z ) : = \ell ( f ( W , x ) , y )$ , the expectation term is

$$
\begin{array} { r l } & { \mathbb { E } _ { D _ { 1 } , \dots , D _ { P } } [ \displaystyle \operatorname* { s u p } \frac { 1 } { H } \sum _ { i = 1 } ^ { M } \varepsilon _ { i } \mathbb { E } _ { \mathfrak { s } ^ { \varepsilon } \sim D _ { i } } [ \ell ( J ( W , x _ { i } ) , y _ { i } ) ] ] } \\ & { = \mathbb { E } _ { z _ { 1 } \setminus z _ { i } \setminus \dots , z _ { P r } } [ \displaystyle \operatorname* { s u p } \frac { 1 } { H \in \mathcal { F } } \frac { M } { i \le 1 } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \varepsilon _ { i } \tilde { \ell } ( W , z _ { i j } ) ] } \\ & { \le \mathbb { E } _ { z _ { 1 } \setminus \dots , z _ { P r } } [ \displaystyle \operatorname* { s u p } \frac { 1 } { f \in \mathcal { F } } \frac { 1 } { H \in \mathcal { P } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \frac { \nu } { n } \sum _ { j = 1 } ^ { n } \varepsilon _ { i j } \tilde { \ell } ( W , z _ { i j } ) ] } \\ & { \le \mathbb { E } _ { z _ { 1 } \setminus \dots , z _ { P r } } [ \displaystyle \operatorname* { s u p } \frac { 1 } { f \in \mathcal { P } } \frac { 1 } { \varepsilon ( P ) } \sum _ { i \in \mathcal { P } } \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \varepsilon _ { i j } \tilde { \ell } ( W , z _ { i j } ) ] } \\ & { = \mathbb { E } _ { z _ { 1 } \setminus \dots , z _ { P r } } [ \mathbb { E } _ { z _ { 1 } \setminus \dots , z _ { P r } } [ \mathrm { s u p } \frac { 1 } { f \in \mathcal { F } } \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \tilde { \ell } _ { i j } \tilde { \ell } ( W , z _ { i j } ) ] ] } \\ &  = \mathbb { E } _ { z _ { 1 } \setminus \dots , z _ { P r } } [ \mathbb { E } ( \mathcal { F } ) ] \le \mathcal  \end{array}
$$

where the last inequality follows from McDiarmid’s inequality, since replacing one point $z _ { i j }$ changes $\mathcal { R } ( \mathcal { F } )$ by at most $c / N$ .

## APPENDIX B

## ADDITIONAL EXPERIMENTAL DETAILS AND RESULTS

## A. Additional Experimental Details

1) Dataset and Models: Beyond the default two-layer CNN, we further evaluate F-SOSA using an ImageNet-pretrained ResNet18 on CIFAR-10 and YOLOv8s [53] on BDD100K [54]. BDD100K is a large-scale real-world driving dataset containing 100,000 40-second videos recorded at 720p and 30 FPS across diverse road, weather, and lighting conditions. It supports ten autonomous-driving tasks, with bounding-box annotations for ten traffic-related object categories. For both settings, we simulate client heterogeneity using the same Dirichlet partitioning scheme $( \alpha \in \{ 0 . 1 , 0 . 5 , 0 . 8 \}$ ; default $\alpha { = } 0 . 5 )$

2) Hyperparameters: For experiments on CIFAR-10 with ResNet18 and BDD100K with YOLOv8s, we set the local learning rate to $\eta = 0 . 0 0 1$ and the total number of clients to $M = 2 0$ . For CIFAR-10, we set the number of local epochs to $K = 5 0$ and the batch size to 64; for BDD100K, we use $K = 2 0$ local epochs, a batch size of 32, and an input resolution of $6 4 0 \times 6 4 0$ . Both settings employ SGD, with momentum 0.9. For BDD100K, we additionally apply weight decay of $5 \times 1 0 ^ { - 4 }$ , a three-epoch warm-up, and cosine learning-rate decay.

3) Metrics: For the object-detection task on BDD100K, we use $\mathrm { m A P _ { 5 0 - 9 5 } }$ as the primary metric to evaluate the detection utility of F-SOSA and the baselines. It is computed by evaluating average precision (AP) at IoU thresholds from 0.50 to 0.95 in increments of 0.05 and averaging the results across all thresholds and ten object categories. This metric jointly reflects object recognition and bounding-box localization accuracy, with higher values indicating better overall detection performance.

## B. F-SOSA Performance Across Model Architectures

To demonstrate that F-SOSA generalizes across models with different capacities, we evaluate network architectures beyond the two-layer CNN, including ResNet18 on CIFAR-10 and YOLOv8s on the real-world autonomous-driving dataset BDD100K. We further evaluate F-SOSA under varying data heterogeneity levels $( \alpha \in \{ 0 . 1 , 0 . 5 , 0 . 8 \} )$ to assess its robustness across both model architectures and non-IID settings. As shown in Table V, on CIFAR-10 with ResNet18, F-SOSA consistently achieves the lowest communication cost of 462.44 MB, reducing communication by 47.6% relative to FedFisher, 48.3% relative to Dense, and 94.3% relative to FedLPA. It also achieves the lowest time cost across all heterogeneity levels, with reductions of up to 43.0%, 56.0%, and 91.1% compared with FedFisher, Dense, and FedLPA, respectively. Although F-SOSA trails the best-performing baseline under high heterogeneity $( \alpha = 0 . 1 )$ , it still outperforms Dense by 7.13 percentage points. At $\alpha = 0 . 5$ and 0.8, its accuracy increases to 65.98% and 69.87%, narrowing the gap with FedFisher to 3.68 and 2.74 percentage points, respectively, while continuing to outperform Dense. Given its consistently lowest time and communication costs, these modest accuracy gaps represent a reasonable utility-efficiency trade-off. On BDD100K with YOLOv8s, F-SOSA maintains competitive detection performance, achieving an $\mathrm { m A P _ { 5 0 - 9 5 } }$ only 0.20 percentage points below the best result at $\alpha = 0 . 1$ , tying for the best result at $\alpha = 0 . 5$ , and obtaining the highest $\mathrm { m A P _ { 5 0 - 9 5 } }$ of 21.30% at $\alpha = 0 . 8 .$ . Meanwhile, it consistently incurs the lowest communication cost of 0.46 GB, representing reductions of 47.1% over FedFisher, 47.7% over Dense, and 95.2% over FedLPA. F-SOSA also achieves the lowest time cost at $\alpha = 0 . 1$ and 0.5 and the second-lowest at $\alpha = 0 . 8$ , while delivering the highest communication efficiency under all settings. These results demonstrate that F-SOSA remains effective with larger classification and detection models, offering a favorable utility–efficiency trade-off across different model architectures, tasks, and degrees of data heterogeneity.

TABLE V  
PERFORMANCE COMPARISON OF F-SOSA AND BASELINES ACROSS CIFAR-10 AND BDD100K UNDER VARYING DIRICHLET HETEROGENEITY α. BEST RESULTS ARE IN BOLD; SECOND-BEST RESULTS ARE UNDERLINED.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Metrics</td><td colspan="4"> $\alpha = 0 . 1$ </td><td colspan="4"> $\alpha = 0 . 5$ </td><td colspan="4"> $\alpha = 0 . 8$ </td></tr><tr><td>FedFisher</td><td>FedLPA</td><td>Dense</td><td>F-SOSA</td><td>FedFisher</td><td>FedLPA</td><td>Dense</td><td>F-SOSA</td><td>FedFisher</td><td>FedLPA</td><td>Dense</td><td>F-SOSA</td></tr><tr><td rowspan="5">CIFAR10 (Resnet18)</td><td>Accuracy (%)</td><td>46.60</td><td>45.18</td><td>28.04</td><td>35.17</td><td>69.66</td><td>68.41</td><td>57.48</td><td>65.98</td><td>72.61</td><td>70.55</td><td>63.92</td><td>69.87</td></tr><tr><td>Time (s)</td><td>361.02</td><td>2493.05</td><td>479.82</td><td>278.22</td><td>362.14</td><td>2355.83</td><td>474.14</td><td>210.34</td><td>365.50</td><td>1713.55</td><td>473.76</td><td>208.27</td></tr><tr><td>Comm. (MB)</td><td>882.16</td><td>8066.21</td><td>895.13</td><td>462.44</td><td>882.16</td><td>8066.21</td><td>895.13</td><td>462.44</td><td>882.16</td><td>8066.21</td><td>895.13</td><td>462.44</td></tr><tr><td>Time Efficiency</td><td>0.129</td><td>0.018</td><td>0.058</td><td>0.126</td><td>0.192</td><td>0.029</td><td>0.121</td><td>0.314</td><td>0.199</td><td>0.041</td><td>0.135</td><td>0.335</td></tr><tr><td>Comm. Efficiency</td><td>0.053</td><td>0.006</td><td>0.031</td><td>0.076</td><td>0.079</td><td>0.008</td><td>0.064</td><td>0.143</td><td>0.082</td><td>0.009</td><td>0.071</td><td>0.151</td></tr><tr><td rowspan="5">BDD100K (YOLOv8s)</td><td>mAP50-95 (%)</td><td>20.82</td><td>20.98</td><td>20.76</td><td>20.78</td><td>21.18</td><td>21.17</td><td>19.54</td><td>21.18</td><td>21.21</td><td>21.22</td><td>21.13</td><td>21.30</td></tr><tr><td>Time (min)</td><td>39.88</td><td>36.41</td><td>37.28</td><td>32.52</td><td>37.77</td><td>31.32</td><td>34.67</td><td>28.96</td><td>37.69</td><td>31.10</td><td>37.11</td><td>34.31</td></tr><tr><td>Comm. (GB)</td><td>0.87</td><td>9.55</td><td>0.88</td><td>0.46</td><td>0.87</td><td>9.55</td><td>0.88</td><td>0.46</td><td>0.87</td><td>9.55</td><td>0.88</td><td>0.46</td></tr><tr><td>Time Efficiency</td><td>0.52</td><td>0.58</td><td>0.56</td><td>0.64</td><td>0.56</td><td>0.68</td><td>0.61</td><td>0.73</td><td>0.56</td><td>0.68</td><td>0.57</td><td>0.62</td></tr><tr><td>Comm. Efficiency</td><td>23.93</td><td>2.20</td><td>23.59</td><td>45.17</td><td>24.34</td><td>2.22</td><td>22.20</td><td>46.04</td><td>24.38</td><td>2.22</td><td>24.01</td><td>46.30</td></tr></table>

## C. Scalability of F-SOSA-U and F-SOSA-I Across Client Populations

To assess the scalability of F-SOSA-I/U with increasing edge-device density, we consider five total client populations, {20, 50, 100, 200, 500}. For each population, we execute 15 sequential incremental-learning requests and 15 sequential unlearning requests. Table VI reports the mean accuracy, time-cost savings relative to full retraining, and communication cost normalized to Retrain, which together capture utility preservation and system efficiency.

For incremental learning, F-SOSA-I maintains accuracy comparable to Retrain across the evaluated client scales. Even at 500 clients, the accuracy gap between F-SOSA-I and Retrain remains within 2.46 percentage points across all four datasets. In the 50-client case, F-SOSA-I achieves the highest accuracy on MNIST, SVHN, and CIFAR-10, reaching 94.35%, 68.95%, and 59.99%, respectively. As the system scales to 200 clients, it achieves the best performance on FEMNIST and CIFAR-10 with accuracies of 32.33% and 46.27%, respectively. For the largest population of 500 clients, the accuracy gap between F-SOSA-I and Retrain remains within 2.46 percentage points across all four datasets, while F-SOSA-I outperforms Fine-tuning by 2.45–10.90 percentage points. This utility advantage is particularly pronounced on the inherently heterogeneous FEMNIST dataset: as the client population increases from 100 to 500, F-SOSA-I maintains an accuracy of approximately 31%, wherea Fine-tuning remains around 20%. In terms of efficiency, the normalized communication cost of F-SOSA-I never exceeds 0.87× that of Retrain and decreases to 0.67× with 500 clients, corresponding to a 33% reduction. Although Fine-tuning incurs marginally lower communication overhead at some scales, these savings come at the cost of substantial accuracy degradation, particularly on MNIST, SVHN, and FEMNIST. F-SOSA-I therefore provides a more favorable utility–efficiency trade-off. Its time-cost savings remain positive across all configurations, ranging from 15.84% to 49.04%. Even with a population of 500 clients, F-SOSA-I saves 23.26%–49.04% of the retraining time, indicating that its computational benefits persist as the edge system becomes increasingly dense.

Regarding unlearning, F-SOSA-U exhibits similarly strong scalability. Across all evaluated client populations, its accuracy closely tracks or exceeds that of Retrain while consistently outperforming most unlearning baselines. The results for 100 clients show that F-SOSA-U achieves the highest accuracy on MNIST, SVHN, and CIFAR-10. When the client population reaches 500, F-SOSA-U surpasses Retrain on SVHN, FEMNIST, and CIFAR-10 and remains within 0.89 percentage points of Retrain on MNIST. In this largest-scale regime, F-SOSA-U also outperforms the strongest competing unlearning baseline by 1.24–4.85 percentage points across the four datasets. Its normalized communication cost remains below that of Retrain throughout the evaluated range. With 500 clients, F-SOSA-U reduces communication by 47% relative to Retrain and FATS-CU and by approximately 60% relative to FedEraser and FedRR, whose communication costs exceed 1.33× that of Retrain Moreover, F-SOSA-U continues to save 38.92%–43.35% of the retraining time at the maximum evaluated scale. Although FedEraser and FATS-CU achieve greater time-cost savings, their accuracy degrades more noticeably as the client population increases. Consequently, F-SOSA-U provides a more favorable overall trade-off among utility, communication efficiency, and computational cost.

![](images/61d8ed0d378fac612fd4f630add2da1f95914f2cae8f9c57f434cd36227a1790.jpg)

![](images/c19cfdbef3ff878a1c578c02d52b919cac71d7fd140fef327d323955ddb65e07.jpg)  
Fig. 7. mAP<sub>50-95</sub> (left) and cumulative cost (right) of F-SOSA-I/U vs. Retrain under simulator-generated model editing requests on the real driving dataset BDD100K.

In addition, we note that the absolute accuracy on MNIST, SVHN, and CIFAR-10 decreases as the number of clients increases, whereas this trend is not observed on the inherently non-IID FEMNIST dataset. This behavior is a consequence of our data-partitioning strategy: increasing the number of clients reduces the amount of training data assigned to each client, thereby limiting the local data available for one-shot FL and subsequent editing. Importantly, F-SOSA-I and F-SOSA-U still achieve the best or near-best relative accuracy compared with their respective baselines across different client scales. This suggests that the decline is primarily associated with the reduced amount of local data per client, rather than a deterioration unique to F-SOSA-I/U.

Overall, these results show that the relative communication and computation benefits of F-SOSA-I/U remain substantial as the client population increases. Across the evaluated range of up to 500 clients, both editing algorithms preserve utility close to Retrain while continuing to reduce its communication and computation costs. This indicates that the efficiency advantage of selective exact editing remains robust under increasingly dense client populations.

## D. Real-World Driving Case Study with BDD100K and SUMO Mobility

We further evaluate F-SOSA-I/U on a dynamic federated object-detection task by combining mobility traces generated by the Simulation of Urban MObility (SUMO) with the real-world BDD100K driving dataset. Following the mobility-trace setting described in section VI-C, the trace produces 31 sequential editing requests, comprising 16 client arrivals and 15 departures. We replay this request sequence on BDD100K using YOLOv8s and report the $\mathrm { m A P _ { 5 0 - 9 5 } }$ after each request, together with the cumulative computation time and communication cost, in Figure 7. F-SOSA-I/U closely tracks the utility of Retrain throughout the sequence, with a maximum $\mathrm { m A P _ { 5 0 - 9 5 } }$ difference of only 0.39 percentage points. Across the 31 editing requests, F-SOSA-I/U matches or outperforms Retrain after 19 requests and achieves a slightly higher average $\mathrm { m A P _ { 5 0 - 9 5 } }$ of 20.25%, compared with 20.21% for Retrain. After the final request, its $\mathrm { m A P _ { 5 0 - 9 5 } }$ reaches 20.27%, exceeding the 20.19% achieved by Retrain. F-SOSA-I/U also exhibits greater stability: its $\mathrm { m A P _ { 5 0 - 9 5 } }$ varies within a range of 0.35 percentage points, compared with 0.55 for Retrain, while its maximum request-to-request variation is 0.28 percentage points, versus 0.46 for Retrain.

More notably, F-SOSA-I/U provides substantial efficiency gains. As shown in the figure, its communication and time curves remain essentially flat for 8 of the 31 requests. When the client involved in a request is not selected for aggregation, Bernoull sampling avoids unnecessary computation and communication. These skipped updates save 2.74 GB of communication and 343.73 minutes of computation, accounting for 100% and approximately 78.31% of the overall communication and time savings, respectively. The savings do not arise solely from skipped updates: across the remaining 23 requests that require model updates F-SOSA-I/U saves an additional 95.19 minutes. This additional time saving occurs because some requests require F-SOSA-I/U to update only one client with a relatively small local dataset, whereas Retrain must wait for all participating clients to complete local training and is therefore limited by the slowest client. After the complete request sequence, F-SOSA-I/U incurs 8.21 GB of cumulative communication, compared with 10.94 GB for Retrain, corresponding to a 25.00% reduction. Its cumulative computation time decreases from 1335.21 to 896.28 minutes, yielding a 32.87% reduction. These absolute savings are particularly significant on BDD100K, since repeatedly retraining YOLOv8s and communicating its model updates over complex real-world driving data is considerably more expensive than editing lightweight classification models.

Taken together, F-SOSA-I/U preserves near-retraining detection utility while substantially reducing the cost of frequent recomputation, demonstrating its practical value in dynamic and communication-constrained edge environments.

TABLE VI  
SCALABILITY OVER 15 INCREMENTAL-LEARNING OR UNLEARNING REQUESTS. WE REPORT MEAN ACCURACY (ACC.), MEAN TIME-COST SAVING (T-SAVE), AND MEAN COMMUNICATION COST NORMALIZED TO RETRAIN (COMM.; RETRAIN = 1.00). T-SAVE IS RELATIVE TO RETRAIN; HIGHER IS BETTER FOR ACC. AND T-SAVE, WHILE LOWER IS BETTER FOR COMM. BEST RESULTS ARE IN BOLD; SECOND-BEST RESULTS ARE UNDERLINED.
<table><tr><td rowspan="2">Clients Method</td><td rowspan="2"></td><td colspan="3">MNIST</td><td colspan="3">SVHN</td><td colspan="3">FEMNIST</td><td colspan="3">CIFAR10</td></tr><tr><td>Acc. (%)</td><td>T-save (%)</td><td>Comm. (×)</td><td>Acc. (%)</td><td>T-save (%)</td><td>Comm. (×)</td><td>Acc. (%)</td><td>T-save (%)</td><td>Comm. (×)</td><td>Acc. (%)</td><td>T-save (%)</td><td>Comm. (×)</td></tr><tr><td colspan="10"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">0.00</td><td rowspan="2">1.00</td></tr><tr><td>20</td><td>Retrain Fine-tuning F-SOSA-I</td><td>96.33 95.74 95.62</td><td>0.00 79.68 20.83</td><td>1.00 0.78 0.75</td><td>73.07 74.67 72.82</td><td>0.00 81.48 37.84</td><td>1.00 27.06 0.78 21.65 0.75 25.64</td><td>0.00 69.13 19.67</td><td>1.00 0.78 0.75</td><td>68.51 67.86 78.89 69.34 45.26</td></tr><tr><td>50</td><td>Retrain Fine-tuning F-SOSA-I</td><td>94.11 93.30 94.35</td><td>0.00 79.67 21.64</td><td>1.00 0.70 0.72</td><td>66.61 65.95 68.95</td><td>0.00 79.70 38.63</td><td>1.00 0.70 0.72</td><td>30.22 19.72 30.09</td><td>0.00 62.36 20.47</td><td>1.00 0.70 0.72</td><td>57.11 52.88 59.99</td><td>0.00 78.29 46.08</td><td>1.00 0.70 0.72</td></tr><tr><td>100</td><td>Retrain Fine-tuning F-SOSA-I</td><td>92.81 91.32 92.94</td><td>0.00 72.55 16.93</td><td>1.00 0.68 0.87</td><td>62.55 55.83 62.18</td><td>0.00 79.64 34.86</td><td>1.00 0.68 0.87</td><td>31.24 20.28 31.05</td><td>0.00 64.89 15.84</td><td>1.00 0.68 0.87</td><td>56.01 47.70 54.22</td><td>0.00 77.37 41.87</td><td>1.00 0.68 0.87</td></tr><tr><td>200</td><td>Retrain Fine-tuning F-SOSA-I</td><td>91.30 81.98 90.22</td><td>0.00 82.80 18.25</td><td>1.00 0.67 0.87</td><td>36.56 23.08 35.94</td><td>0.00 79.03 36.04</td><td>1.00 0.67 0.87</td><td>30.65 19.22 32.33</td><td>0.00 72.35 17.16</td><td>1.00 0.67 0.87</td><td>43.65 42.41 46.27</td><td>0.00 75.61 43.18</td><td>1.00 0.67 0.87</td></tr><tr><td>500</td><td>Retrain Fine-tuning F-SOSA-I</td><td>0.00 80.26 24.17</td><td>1.00 0.67</td><td>0.67</td><td>19.60 14.77 17.88</td><td>0.00 79.95 41.63</td><td>1.00 0.67 0.67</td><td>31.88 20.15 31.05</td><td>0.00 76.82 23.26</td><td>1.00 0.67 0.67</td><td>35.65 30.74 33.19</td><td>0.00 74.48 49.04</td><td>1.00 0.67 0.67</td></tr><tr><td>Retrain</td><td>95.73</td><td colspan="10">0.00</td><td>64.75</td><td>0.00 1.00</td></tr><tr><td>20</td><td>FedEraser FedRR FATS-CU F-SOSA-U</td><td>93.95 92.20 88.72 94.87</td><td>78.64 64.18 76.92 44.73 0.00</td><td>1.35 1.35 1.00 0.75 1.00</td><td>68.48 56.26 61.25 64.95</td><td>79.12 68.79 77.36 45.81</td><td>1.35 1.35 1.00 0.75</td><td>21.17 24.78 20.11 26.51</td><td>77.85 44.18 75.94 41.62</td><td>1.35 1.35 1.00 0.75</td><td>67.74 62.03 61.85 63.91</td><td>78.31 64.10 76.58 43.79</td><td>1.35 1.35 1.00 0.75</td></tr><tr><td>50</td><td>Retrain FedEraser FedRR FATS-CU F-SOSA-U</td><td>94.87 93.21 91.97 90.87 94.28</td><td>78.21 63.76 76.49 44.16</td><td>1.34 1.34 1.00 0.87</td><td>63.73 56.80 39.11 53.63 62.88</td><td>0.00 78.77 68.31 76.94 45.22</td><td>1.00 1.34 1.34 1.00 0.87</td><td>26.49 18.59 21.22 18.38 28.01</td><td>0.00 77.46 43.76 75.61 41.08</td><td>1.00 1.34 1.34 1.00 0.87</td><td>56.93 59.14 56.10 57.10 57.86</td><td>0.00 77.94 63.69 76.17 43.25</td><td>1.00 1.34 1.34 1.00 0.87</td></tr><tr><td>100</td><td>Retrain FedEraser FedRR FATS-CU</td><td>92.26 91.31 90.04 90.24 92.33</td><td>0.00 77.83 61.21 76.12 43.58</td><td>1.00 1.34 1.34 1.00 0.80</td><td>58.58 45.50 26.75 44.65 59.32</td><td>0.00 78.39 65.18 76.58 44.76</td><td>1.00 1.34 1.34 1.00 0.80</td><td>30.68 23.82 22.48 23.51 29.81</td><td>0.00 77.02 43.11 75.18 40.54</td><td>1.00 1.34 1.34 1.00 0.80</td><td>51.64 51.57 50.38 50.67 51.72</td><td>0.00 77.56 63.02 75.83 42.61</td><td>1.00 1.34 1.34 1.00 0.80</td></tr><tr><td>200</td><td>Retrain FedEraser FedRR FATS-CU</td><td>91.21 87.56 86.95 86.23</td><td>0.00 77.34 59.84 75.67 42.91</td><td>1.00 1.33 1.33 1.00 0.67</td><td>40.09 25.56 19.59 24.08 39.62</td><td>0.00 77.95 62.46 76.11 44.13</td><td>1.00 1.33 1.33 1.00</td><td>28.41 24.00 25.90 23.24</td><td>0.00 76.58 42.42 74.72</td><td>1.00 1.33 1.33 1.00</td><td>48.61 43.29 40.06 42.64</td><td>0.00 77.18 62.74 75.41</td><td>1.00 1.33 1.33 1.00 0.67</td></tr><tr><td>500</td><td>Retrain FedEraser FedRR FATS-CU F-SOSA-U</td><td>90.28 86.62 80.99 80.71 79.63 85.73</td><td>0.00 76.82 59.37 75.13 41.76</td><td>1.00 1.33 1.33 1.00 0.53</td><td>19.66 19.59 19.59 19.66 20.90</td><td>0.00 77.46 61.92 75.62 43.35</td><td>0.67 1.00 1.33 1.33 1.00 0.53</td><td>29.09 30.45 25.11 26.18 25.38 31.03</td><td>39.86 0.00 76.11 41.57 74.29 38.92</td><td>0.67 1.00 1.33 1.33 1.00</td><td>46.22 33.71 31.90 26.84 31.71</td><td>41.94 0.00 76.73 62.48 74.96</td><td>1.00 1.33 1.33 1.00</td></tr></table>

## E. Detailed Communication-Cost Breakdown

To provide a clear breakdown of the communication costs, Table VII reports the per-round costs for selected and unselected clients in F-SOSA with M = 20 and ρ = 0.5. Under this configuration, ten clients are selected, while the remaining ten are unselected. The server first broadcasts the global model to all clients. Then, only the selected clients upload their local model and approximate Fisher information, represented by either Diag or K-FAC, whereas the unselected clients do not upload any information. Note that although four experiments in the table use two-layer CNNs, their architectures and parameter counts differ, resulting in different model sizes.

As shown in the table, the uplink cost of the Diag Fisher information is equal to that of the 16-bit quantized local model for all model architectures. Their combined size is therefore equal to the communication size of one FP32 model. This is because 16-bit quantization is applied to both the local models and the Diag Fisher information in our implementation. After receiving the quantized information, the server reconstructs the values and performs aggregation. More specifically, each loca model and its diagonal Fisher information are quantized layer by layer to 16-bit precision before aggregation. They have the same communication cost because both contain one value for each model parameter. For K-FAC, the Fisher factors are furthe compressed using low-rank decomposition and quantization. Therefore, the K-FAC communication cost depends on the model architecture and the dimensions of its Fisher factors, and may differ slightly from the Diag cost. The table also shows that unselected clients incur no uplink cost. The final column reports the total communication costs of the Diag and K-FAC variants, respectively. The two-layer CNN has the same communication cost on SVHN and CIFAR-10 because the two settings use the same model architecture and number of parameters.

TABLE VII  
DETAILED COMMUNICATION COSTS (MB) OF F-SOSA FOR $M = 2 0 ~ \mathrm { { A N D } } ~ \rho = 0 . 5$ , WHERE M IS THE TOTAL NUMBER OF CLIENTS AND ρ IS THE CLIENT SELECTION RATIO. THE DOWNLINK CORRESPONDS TO ONE BROADCAST OF THE GLOBAL MODEL TO ALL ACTIVE CLIENTS. MODEL UP., DIAG UP., AND K-FAC UP. DENOTE THE AGGREGATE UPLINK COMMUNICATION OF THE SELECTED CLIENTS FOR LOCAL MODELS, DIAGONAL FISHER INFORMATION, AND KRONECKER-FACTORED APPROXIMATE CURVATURE (K-FAC) FISHER INFORMATION, RESPECTIVELY. UNSELECTED CLIENTS INCUR NO UPLINK COMMUNICATION. THE FINAL COLUMN REPORTS THE TOTAL COMMUNICATION COSTS OF THE DIAG AND K-FAC VARIANTS.
<table><tr><td>Dataset</td><td>Model</td><td>Broadcast</td><td colspan="3">Selected Clients</td><td>Unselected Clients</td><td>Round Total</td></tr><tr><td></td><td></td><td>Downlink</td><td>Model Up.</td><td>Diag Up.</td><td>K-FAC Up.</td><td>Uplink</td><td>Diag/K-FAC</td></tr><tr><td>MNIST</td><td>Two-Layer CNN</td><td>1.61</td><td>8.04</td><td>8.04</td><td>8.81</td><td>0</td><td>17.69/18.46</td></tr><tr><td>SVHN</td><td>Two-Layer CNN</td><td>3.58</td><td>17.92</td><td>17.92</td><td>18.81</td><td>0</td><td>39.42/40.31</td></tr><tr><td>CIFAR10</td><td>Two-Layer CNN</td><td>3.58</td><td>17.92</td><td>17.92</td><td>18.81</td><td>0</td><td>39.42/40.31</td></tr><tr><td>CIFAR10</td><td>ResNet18-Pretrained</td><td>42.63</td><td>213.13</td><td>213.13</td><td>206.68</td><td>0</td><td>468.89/462.44</td></tr><tr><td>FEMNIST</td><td>Two-Layer CNN</td><td>25.18</td><td>125.91</td><td>125.91</td><td>138.59</td><td>0</td><td>277.00/289.68</td></tr><tr><td>BDD100K</td><td>YOLOv8s</td><td>42.49</td><td>212.47</td><td>212.47</td><td>211.99</td><td>0</td><td>467.43/466.95</td></tr></table>

Furthermore, as shown in Tables VIII–XII, we report a detailed breakdown of the actual amount of data transmitted during the initial training round and 31 sequential editing requests. The communication cost includes three parts: global-model broadcasts, local-model uploads from selected clients, and approximate Fisher information uploads from selected clients. Unselected clients do not upload models or Fisher information. Therefore, the reported totals include all communication required to exchange the current model and Fisher information between the server and clients. With this complete accounting, F-SOSA-I/U reduces the cumulative communication cost from 424.48 to 329.61 MB on MNIST, from 966.63 to 717.63 MB on SVHN, from 6942.18 to 5416.97 MB on FEMNIST, from 966.63 to 750.59 MB on CIFAR-10, and from 11207.23 to 8405.42 MB on BDD100K These results correspond to reductions of 22.35%, 25.76%, 21.97%, 22.35%, and 25.00%, respectively. The detailed BDD100K results further show that the F-SOSA-I/U total of 8405.42 MB consists of 1019.76 MB of global-model broadcasts, 3696.95 MB of local-model uploads, and 3688.71 MB of Fisher-information uploads. In comparison, retraining requires 1359.68, 4929.27, and 4918.28 MB for the same three parts, respectively. Thus, the reduction is obtained even after including the large Fisher-information uploads and all model transfers. When an editing request requires an update, F-SOSA-I/U has the same per-round communication components as retraining; however, it avoids the full broadcast and upload process when the reques does not require updating the selected clients. This saves 94.87, 249.00, 1525.21, 216.04, and 2801.81 MB on MNIST, SVHN, FEMNIST, CIFAR-10, and BDD100K, respectively. SVHN achieves the largest relative reduction of 25.76%, while BDD100K achieves the largest absolute reduction of 2801.81 MB. Overall, F-SOSA-I/U provides consistent communication savings acros datasets, ranging from 21.97% on FEMNIST to 25.76% on SVHN, while preserving performance comparable to retraining.

DETAILED COMMUNICATION COSTS (MB) OF SEQUENTIAL EDITING ON MNIST WITH $\rho = 0 . 5$ . EACH COMMUNICATION ENTRY IS REPORTED AS F-SOSA-I/U VERSUS RETRAIN, WITH THE FIRST VALUE CORRESPONDING TO F-SOSA-I/U AND THE SECOND TO RETRAIN. MODEL UP. AND FISHER UP. DENOTE THE COMMUNICATION COSTS OF UPLOADING THE MODEL AND THE APPROXIMATE FISHER INFORMATION. RESPECTIVELY. IN THE REOUEST COLUMN, THE NUMBER IDENTIFIES THE AFFECTED CLIENT, WHILE + AND − DENOTE CLIENT ADDITION AND CLIENT REMOVAL, RESPECTIVELY.  
TABLE VIII
<table><tr><td rowspan="2">Request</td><td>Broadcast</td><td colspan="2">Selected Clients</td><td>Unselected Clients</td><td rowspan="2">Round Total</td></tr><tr><td>Downlink</td><td>Model Up.</td><td>Fisher Up.</td><td>Uplink</td></tr><tr><td>Initial</td><td>1.61/1.61</td><td>4.82/4.82</td><td>4.82/4.82</td><td>0.00/0.00</td><td>11.25/11.25</td></tr><tr><td>12+</td><td>0.00/1.61</td><td>0.00/4.82</td><td>0.00/4.82</td><td>0.00/0.00</td><td>0.00/11.25</td></tr><tr><td>13+</td><td>1.61/1.61</td><td>5.63/5.63</td><td>5.63/5.63</td><td>0.00/0.00</td><td>12.87/12.87</td></tr><tr><td>14+</td><td>0.00/1.61</td><td>0.00/5.63</td><td>0.00/5.63</td><td>0.00/0.00</td><td>0.00/12.87</td></tr><tr><td>15+</td><td>1.61/1.61</td><td>6.43/6.43</td><td>6.43/6.43</td><td>0.00/0.00</td><td>14.47/14.47</td></tr><tr><td>16+</td><td>0.00/1.61</td><td>0.00/6.43</td><td>0.00/6.43</td><td>0.00/0.00</td><td>0.00/14.47</td></tr><tr><td>3-</td><td>0.00/1.61</td><td>0.00/6.43</td><td>0.00/6.43</td><td>0.00/0.00</td><td>0.00/14.47</td></tr><tr><td>6-</td><td>1.61/1.61</td><td>5.63/5.63</td><td>5.63/5.63</td><td>0.00/0.00</td><td>12.87/12.87</td></tr><tr><td>7-</td><td>1.61/1.61</td><td>5.63/5.63</td><td>5.63/5.63</td><td>0.00/0.00</td><td>12.87/12.87</td></tr><tr><td>17+</td><td>1.61/1.61</td><td>5.63/5.63</td><td>5.63/5.63</td><td>0.00/0.00</td><td>12.87/12.87</td></tr><tr><td>18+</td><td>1.61/1.61</td><td>6.43/6.43</td><td>6.43/6.43</td><td>0.00/0.00</td><td>14.47/14.47</td></tr><tr><td>19+</td><td>0.00/1.61</td><td>0.00/6.43</td><td>0.00/6.43</td><td>0.00/0.00</td><td>0.00/14.47</td></tr><tr><td>1-</td><td>1.61/1.61</td><td>6.43/6.43</td><td>6.43/6.43</td><td>0.00/0.00</td><td>14.47/14.47</td></tr><tr><td>11-</td><td>1.61/1.61</td><td>5.63/5.63</td><td>5.63/5.63</td><td>0.00/0.00</td><td>12.87/12.87</td></tr><tr><td>16-</td><td>1.61/1.61</td><td>5.63/5.63</td><td>5.63/5.63</td><td>0.00/0.00</td><td>12.87/12.87</td></tr><tr><td>2-</td><td>1.61/1.61</td><td>4.82/4.82</td><td>4.82/4.82</td><td>0.00/0.00</td><td>11.25/11.25</td></tr><tr><td>20+</td><td>1.61/1.61</td><td>5.63/5.63</td><td>5.63/5.63</td><td>0.00/0.00</td><td>12.87/12.87</td></tr><tr><td>21+</td><td>1.61/1.61</td><td>5.63/5.63</td><td>5.63/5.63</td><td>0.00/0.00</td><td>12.87/12.87</td></tr><tr><td>22+</td><td>1.61/1.61</td><td>6.43/6.43</td><td>6.43/6.43</td><td>0.00/0.00</td><td>14.47/14.47</td></tr><tr><td>4-</td><td>1.61/1.61</td><td>5.63/5.63</td><td>5.63/5.63</td><td>0.00/0.00</td><td>12.87/12.87</td></tr><tr><td>8-</td><td>0.00/1.61</td><td>0.00/5.63</td><td>0.00/5.63</td><td>0.00/0.00</td><td>0.00/12.87</td></tr><tr><td>9-</td><td>1.61/1.61</td><td>4.82/4.82</td><td>4.82/4.82</td><td>0.00/0.00</td><td>11.25/11.25</td></tr><tr><td>23+</td><td>1.61/1.61</td><td>5.63/5.63</td><td>5.63/5.63</td><td>0.00/0.00</td><td>12.87/12.87</td></tr><tr><td>24+</td><td>1.61/1.61</td><td>5.63/5.63</td><td>5.63/5.63</td><td>0.00/0.00</td><td>12.87/12.87</td></tr><tr><td>25+</td><td>1.61/1.61</td><td>6.43/6.43</td><td>6.43/6.43</td><td>0.00/0.00</td><td>14.47/14.47</td></tr><tr><td>26+</td><td>1.61/1.61</td><td>6.43/6.43</td><td>6.43/6.43</td><td>0.00/0.00</td><td>14.47/14.47</td></tr><tr><td>27+</td><td>1.61/1.61</td><td>7.23/7.23</td><td>7.23/7.23</td><td>0.00/0.00</td><td>16.07/16.07</td></tr><tr><td>0-</td><td>1.61/1.61</td><td>6.43/6.43</td><td>6.43/6.43</td><td>0.00/0.00</td><td>14.47/14.47</td></tr><tr><td>10-</td><td>0.00/1.61</td><td>0.00/6.43</td><td>0.00/6.43</td><td>0.00/0.00</td><td>0.00/14.47</td></tr><tr><td>17-</td><td>1.61/1.61</td><td>5.63/5.63</td><td>5.63/5.63</td><td>0.00/0.00</td><td>12.87/12.87</td></tr><tr><td>18-</td><td>1.61/1.61</td><td>5.63/5.63</td><td>5.63/5.63</td><td>0.00/0.00</td><td>12.87/12.87</td></tr><tr><td>19-</td><td>1.61/1.61</td><td>4.82/4.82</td><td>4.82/4.82</td><td>0.00/0.00</td><td>11.25/11.25</td></tr><tr><td>Total</td><td>40.25/51.52</td><td>144.68/186.48</td><td>144.68/186.48</td><td>0.00/0.00</td><td>329.61/424.48</td></tr></table>

DETAILED COMMUNICATION COSTS (MB) OE SEOUENTIAL EDITING ON SVHN WITH $\rho = 0 . 5$ . EACH COMMUNICATION ENTRY IS REPORTED AS F-SOSA-I/U VERSUS RETRAIN, WITH THE FIRST VALUE CORRESPONDING TO F-SOSA-I/U AND THE SECOND TO RETRAIN. MODEL UP. AND FISHER UP. DENOTE THE COMMUNICATION COSTS OF UPLOADING THE MODEL AND THE APPROXIMATE FISHER INFORMATION. RESPECTIVELY. IN THE REOUEST COLUMN, THE NUMBER IDENTIFIES THE AFFECTED CLIENT, WHILE + AND − DENOTE CLIENT ADDITION AND CLIENT REMOVAL, RESPECTIVELY.  
TABLE IX
<table><tr><td rowspan="2">Request</td><td>Broadcast</td><td colspan="2">Selected Clients</td><td>Unselected Clients</td><td rowspan="2">Round Total</td></tr><tr><td>Downlink</td><td>Model Up.</td><td>Fisher Up.</td><td>Uplink</td></tr><tr><td>Initial</td><td>3.58/3.58</td><td>10.75/10.75</td><td>11.28/11.28</td><td>0.00/0.00</td><td>25.61/25.61</td></tr><tr><td>12+</td><td>3.58/3.58</td><td>10.75/10.75</td><td>11.28/11.28</td><td>0.00/0.00</td><td>25.61/25.61</td></tr><tr><td>13+</td><td>3.58/3.58</td><td>12.55/12.55</td><td>13.16/13.16</td><td>0.00/0.00</td><td>29.29/29.29</td></tr><tr><td>14+</td><td>3.58/3.58</td><td>12.55/12.55</td><td>13.16/13.16</td><td>0.00/0.00</td><td>29.29/29.29</td></tr><tr><td>15+</td><td>3.58/3.58</td><td>14.34/14.34</td><td>15.04/15.04</td><td>0.00/0.00</td><td>32.96/32.96</td></tr><tr><td>16+</td><td>3.58/3.58</td><td>14.34/14.34</td><td>15.04/15.04</td><td>0.00/0.00</td><td>32.96/32.96</td></tr><tr><td>3-</td><td>0.00/3.58</td><td>0.00/14.34</td><td>0.00/15.04</td><td>0.00/0.00</td><td>0.00/32.96</td></tr><tr><td>6-</td><td>3.58/3.58</td><td>12.55/12.55</td><td>13.16/13.16</td><td>0.00/0.00</td><td>29.29/29.29</td></tr><tr><td>7-</td><td>3.58/3.58</td><td>12.55/12.55</td><td>13.16/13.16</td><td>0.00/0.00</td><td>29.29/29.29</td></tr><tr><td>17+</td><td>0.00/3.58</td><td>0.00/12.55</td><td>0.00/13.16</td><td>0.00/0.00</td><td>0.00/29.29</td></tr><tr><td>18+</td><td>3.58/3.58</td><td>14.34/14.34</td><td>15.04/15.04</td><td>0.00/0.00</td><td>32.96/32.96</td></tr><tr><td>19+</td><td>0.00/3.58</td><td>0.00/14.34</td><td>0.00/15.04</td><td>0.00/0.00</td><td>0.00/32.96</td></tr><tr><td>1-</td><td>3.58/3.58</td><td>14.34/14.34</td><td>15.04/15.04</td><td>0.00/0.00</td><td>32.96/32.96</td></tr><tr><td>11-</td><td>3.58/3.58</td><td>12.55/12.55</td><td>13.16/13.16</td><td>0.00/0.00</td><td>29.29/29.29</td></tr><tr><td>16-</td><td>0.00/3.58</td><td>0.00/12.55</td><td>0.00/13.16</td><td>0.00/0.00</td><td>0.00/29.29</td></tr><tr><td>2-</td><td>3.58/3.58</td><td>10.75/10.75</td><td>11.28/11.28</td><td>0.00/0.00</td><td>25.61/25.61</td></tr><tr><td>20+</td><td>3.58/3.58</td><td>12.55/12.55</td><td>13.16/13.16</td><td>0.00/0.00</td><td>29.29/29.29</td></tr><tr><td>21+</td><td>3.58/3.58</td><td>12.55/12.55</td><td>13.16/13.16</td><td>0.00/0.00</td><td>29.29/29.29</td></tr><tr><td>22+</td><td>3.58/3.58</td><td>14.34/14.34</td><td>15.04/15.04</td><td>0.00/0.00</td><td>32.96/32.96</td></tr><tr><td>4-</td><td>3.58/3.58</td><td>12.55/12.55</td><td>13.16/13.16</td><td>0.00/0.00</td><td>29.29/29.29</td></tr><tr><td>8-</td><td>0.00/3.58</td><td>0.00/12.55</td><td>0.00/13.16</td><td>0.00/0.00</td><td>0.00/29.29</td></tr><tr><td>9-</td><td>3.58/3.58</td><td>10.75/10.75</td><td>11.28/11.28</td><td>0.00/0.00</td><td>25.61/25.61</td></tr><tr><td>23+</td><td>3.58/3.58</td><td>12.55/12.55</td><td>13.16/13.16</td><td>0.00/0.00</td><td>29.29/29.29</td></tr><tr><td>24+</td><td>0.00/3.58</td><td>0.00/12.55</td><td>0.00/13.16</td><td>0.00/0.00</td><td>0.00/29.29</td></tr><tr><td>25+</td><td>3.58/3.58</td><td>14.34/14.34</td><td>15.04/15.04</td><td>0.00/0.00</td><td>32.96/32.96</td></tr><tr><td>26+</td><td>0.00/3.58</td><td>0.00/14.34</td><td>0.00/15.04</td><td>0.00/0.00</td><td>0.00/32.96</td></tr><tr><td>27+</td><td>3.58/3.58</td><td>16.13/16.13</td><td>16.93/16.93</td><td>0.00/0.00</td><td>36.64/36.64</td></tr><tr><td>0-</td><td>3.58/3.58</td><td>14.34/14.34</td><td>15.04/15.04</td><td>0.00/0.00</td><td>32.96/32.96</td></tr><tr><td>10-</td><td>0.00/3.58</td><td>0.00/14.34</td><td>0.00/15.04</td><td>0.00/0.00</td><td>0.00/32.96</td></tr><tr><td>17-</td><td>3.58/3.58</td><td>12.55/12.55</td><td>13.16/13.16</td><td>0.00/0.00</td><td>29.29/29.29</td></tr><tr><td>18-</td><td>3.58/3.58</td><td>12.55/12.55</td><td>13.16/13.16</td><td>0.00/0.00</td><td>29.29/29.29</td></tr><tr><td>19-</td><td>3.58/3.58</td><td>10.75/10.75</td><td>11.28/11.28</td><td>0.00/0.00</td><td>25.61/25.61</td></tr><tr><td>Total</td><td></td><td>85.92/114.56308.25/415.78 323.46/436.29</td><td></td><td>0.00/0.00</td><td>717.63/966.63</td></tr></table>

DETAILED COMMUNICATION COSTS (MB) OF SEQUENTIAL EDITING ON FEMNIST WITH $\rho = 0 . 5$ . EACH COMMUNICATION ENTRY IS REPORTED AS F-SOSA-I/U VERSUS RETRAIN, WITH THE FIRST VALUE CORRESPONDING TO F-SOSA-I/U AND THE SECOND TO RETRAIN. MODEL UP. AND FISHER UP. DENOTE THE COMMUNICATION COSTS OF UPLOADING THE MODEL AND THE APPROXIMATE FISHER INFORMATION, RESPECTIVELY. IN THE REQUEST COLUMN, THE NUMBER IDENTIFIES THE AFFECTED CLIENT, WHILE + AND – DENOTE CLIENT ADDITION AND CLIENT REMOVAL, RESPECTIVELY  
TABLE X
<table><tr><td rowspan="2">Request</td><td>Broadcast</td><td colspan="2">Selected Clients</td><td>Unselected Clients</td><td rowspan="2">Round Total</td></tr><tr><td>Downlink</td><td>Model Up.</td><td>Fisher Up.</td><td>Uplink</td></tr><tr><td>Initial</td><td>25.18/25.18</td><td>75.55/75.55</td><td>83.15/83.15</td><td>0.00/0.00</td><td>183.88/183.88</td></tr><tr><td>12+</td><td>0.00/25.18</td><td>0.00/75.55</td><td>0.00/83.15</td><td>0.00/0.00</td><td>0.00/183.88</td></tr><tr><td>13+</td><td>25.18/25.18</td><td>88.14/88.14</td><td>97.01/97.01</td><td>0.00/0.00</td><td>210.33/210.33</td></tr><tr><td>14+</td><td>0.00/25.18</td><td>0.00/88.14</td><td>0.00/97.01</td><td>0.00/0.00</td><td>0.00/210.33</td></tr><tr><td>15+</td><td>25.18/25.18</td><td>100.73/100.73</td><td>110.87/110.87</td><td>0.00/0.00</td><td>236.78/236.78</td></tr><tr><td>16+</td><td>0.00/25.18</td><td>0.00/100.73</td><td>0.00/110.87</td><td>0.00/0.00</td><td>0.00/236.78</td></tr><tr><td>3-</td><td>25.18/25.18</td><td>100.73/100.73</td><td>110.87/110.87</td><td>0.00/0.00</td><td>236.78/236.78</td></tr><tr><td>6-</td><td>25.18/25.18</td><td>88.14/88.14</td><td>97.01/97.01</td><td>0.00/0.00</td><td>210.33/210.33</td></tr><tr><td>7-</td><td>25.18/25.18</td><td>88.14/88.14</td><td>97.01/97.01</td><td>0.00/0.00</td><td>210.33/210.33</td></tr><tr><td>17+</td><td>25.18/25.18</td><td>88.14/88.14</td><td>97.01/97.01</td><td>0.00/0.00</td><td>210.33/210.33</td></tr><tr><td>18+</td><td>25.18/25.18</td><td>100.73/100.73</td><td>110.87/110.87</td><td>0.00/0.00</td><td>236.78/236.78</td></tr><tr><td>19+</td><td>25.18/25.18</td><td>100.73/100.73</td><td>110.87/110.87</td><td>0.00/0.00</td><td>236.78/236.78</td></tr><tr><td>1-</td><td>0.00/25.18</td><td>0.00/100.73</td><td>0.00/110.87</td><td>0.00/0.00</td><td>0.00/236.78</td></tr><tr><td>11-</td><td>25.18/25.18</td><td>88.14/88.14</td><td>97.01/97.01</td><td>0.00/0.00</td><td>210.33/210.33</td></tr><tr><td>16-</td><td>25.18/25.18</td><td>88.14/88.14</td><td>97.01/97.01</td><td>0.00/0.00</td><td>210.33/210.33</td></tr><tr><td>2-</td><td>25.18/25.18</td><td>75.55/75.55</td><td>83.15/83.15</td><td>0.00/0.00</td><td>183.88/183.88</td></tr><tr><td>20+</td><td>25.18/25.18</td><td>88.14/88.14</td><td>97.01/97.01</td><td>0.00/0.00</td><td>210.33/210.33</td></tr><tr><td>21+</td><td>25.18/25.18</td><td>88.14/88.14</td><td>97.01/97.01</td><td>0.00/0.00</td><td>210.33/210.33</td></tr><tr><td>22+</td><td>25.18/25.18</td><td>100.73/100.73</td><td>110.87/110.87</td><td>0.00/0.00</td><td>236.78/236.78</td></tr><tr><td>4-</td><td>25.18/25.18</td><td>88.14/88.14</td><td>97.01/97.01</td><td>0.00/0.00</td><td>210.33/210.33</td></tr><tr><td>8-</td><td>25.18/25.18</td><td>88.14/88.14</td><td>97.01/97.01</td><td>0.00/0.00</td><td>210.33/210.33</td></tr><tr><td>9-</td><td>25.18/25.18</td><td>75.55/75.55</td><td>83.15/83.15</td><td>0.00/0.00</td><td>183.88/183.88</td></tr><tr><td>23+</td><td>25.18/25.18</td><td>88.14/88.14</td><td>97.01/97.01</td><td>0.00/0.00</td><td>210.33/210.33</td></tr><tr><td>24+</td><td>0.00/25.18</td><td>0.00/88.14</td><td>0.00/97.01</td><td>0.00/0.00</td><td>0.00/210.33</td></tr><tr><td>25+</td><td>25.18/25.18</td><td>100.73/100.73</td><td>110.87/110.87</td><td>0.00/0.00</td><td>236.78/236.78</td></tr><tr><td>26+</td><td>25.18/25.18</td><td>100.73/100.73</td><td>110.87/110.87</td><td>0.00/0.00</td><td>236.78/236.78</td></tr><tr><td>27+</td><td>25.18/25.18</td><td>113.32/113.32</td><td>124.73/124.73</td><td>0.00/0.00</td><td>263.23/263.23</td></tr><tr><td>0-</td><td>25.18/25.18</td><td>100.73/100.73</td><td>110.87/110.87</td><td>0.00/0.00</td><td>236.78/236.78</td></tr><tr><td>10-</td><td>0.00/25.18</td><td>0.00/100.73</td><td>0.00/110.87</td><td>0.00/0.00</td><td>0.00/236.78</td></tr><tr><td>17-</td><td>25.18/25.18</td><td>88.14/88.14</td><td>97.01/97.01</td><td>0.00/0.00</td><td>210.33/210.33</td></tr><tr><td>18-</td><td>0.00/25.18</td><td>0.00/88.14</td><td>0.00/97.01</td><td>0.00/0.00</td><td>0.00/210.33</td></tr><tr><td>19-</td><td>25.18/25.18</td><td>75.55/75.55</td><td>83.15/83.15</td><td>0.00/0.00</td><td>183.88/183.88</td></tr><tr><td>Total</td><td>629.50/805.76</td><td>2279.04/2921.20</td><td>2508.43/3215.22</td><td>0.00/0.00</td><td>5416.97/6942.18</td></tr></table>

DETAILED COMMUNICATION COSTS (MB) OF SEQUENTIAL EDITING ON CIFAR10 WITH $\rho = 0 . 5$ . EACH COMMUNICATION ENTRY IS REPORTED AS F-SOSA-I/U VERSUS RETRAIN, WITH THE FIRST VALUE CORRESPONDING TO F-SOSA-I/U AND THE SECOND TO RETRAIN. MODEL UP. AND FISHER UP. DENOTE THE COMMUNICATION COSTS OF UPLOADING THE MODEL AND THE APPROXIMATE FISHER INFORMATION. RESPECTIVELY. IN THE REOUEST COLUMN, THE NUMBER IDENTIFIES THE AFFECTED CLIENT, WHILE + AND − DENOTE CLIENT ADDITION AND CLIENT REMOVAL, RESPECTIVELY.  
TABLE XI
<table><tr><td rowspan="2">Request</td><td>Broadcast</td><td colspan="2">Selected Clients</td><td>Unselected Clients</td><td rowspan="2">Round Total</td></tr><tr><td>Downlink</td><td>Model Up.</td><td>Fisher Up.</td><td>Uplink</td></tr><tr><td>Initial</td><td>3.58/3.58</td><td>10.75/10.75</td><td>11.28/11.28</td><td>0.00/0.00</td><td>25.61/25.61</td></tr><tr><td>12+</td><td>3.58/3.58</td><td>10.75/10.75</td><td>11.28/11.28</td><td>0.00/0.00</td><td>25.61/25.61</td></tr><tr><td>13+</td><td>3.58/3.58</td><td>12.55/12.55</td><td>13.16/13.16</td><td>0.00/0.00</td><td>29.29/29.29</td></tr><tr><td>14+</td><td>0.00/3.58</td><td>0.00/12.55</td><td>0.00/13.16</td><td>0.00/0.00</td><td>0.00/29.29</td></tr><tr><td>15+</td><td>3.58/3.58</td><td>14.34/14.34</td><td>15.04/15.04</td><td>0.00/0.00</td><td>32.96/32.96</td></tr><tr><td>16+</td><td>3.58/3.58</td><td>14.34/14.34</td><td>15.04/15.04</td><td>0.00/0.00</td><td>32.96/32.96</td></tr><tr><td>3-</td><td>3.58/3.58</td><td>14.34/14.34</td><td>15.04/15.04</td><td>0.00/0.00</td><td>32.96/32.96</td></tr><tr><td>6-</td><td>3.58/3.58</td><td>12.55/12.55</td><td>13.16/13.16</td><td>0.00/0.00</td><td>29.29/29.29</td></tr><tr><td>7-</td><td>0.00/3.58</td><td>0.00/12.55</td><td>0.00/13.16</td><td>0.00/0.00</td><td>0.00/29.29</td></tr><tr><td>17+</td><td>0.00/3.58</td><td>0.00/12.55</td><td>0.00/13.16</td><td>0.00/0.00</td><td>0.00/29.29</td></tr><tr><td>18+</td><td>3.58/3.58</td><td>14.34/14.34</td><td>15.04/15.04</td><td>0.00/0.00</td><td>32.96/32.96</td></tr><tr><td>19+</td><td>3.58/3.58</td><td>14.34/14.34</td><td>15.04/15.04</td><td>0.00/0.00</td><td>32.96/32.96</td></tr><tr><td>1-</td><td>0.00/3.58</td><td>0.00/14.34</td><td>0.00/15.04</td><td>0.00/0.00</td><td>0.00/32.96</td></tr><tr><td>11-</td><td>3.58/3.58</td><td>12.55/12.55</td><td>13.16/13.16</td><td>0.00/0.00</td><td>29.29/29.29</td></tr><tr><td>16-</td><td>3.58/3.58</td><td>12.55/12.55</td><td>13.16/13.16</td><td>0.00/0.00</td><td>29.29/29.29</td></tr><tr><td>2-</td><td>3.58/3.58</td><td>10.75/10.75</td><td>11.28/11.28</td><td>0.00/0.00</td><td>25.61/25.61</td></tr><tr><td>20+</td><td>3.58/3.58</td><td>12.55/12.55</td><td>13.16/13.16</td><td>0.00/0.00</td><td>29.29/29.29</td></tr><tr><td>21+</td><td>3.58/3.58</td><td>12.55/12.55</td><td>13.16/13.16</td><td>0.00/0.00</td><td>29.29/29.29</td></tr><tr><td>22+</td><td>3.58/3.58</td><td>14.34/14.34</td><td>15.04/15.04</td><td>0.00/0.00</td><td>32.96/32.96</td></tr><tr><td>4-</td><td>3.58/3.58</td><td>12.55/12.55</td><td>13.16/13.16</td><td>0.00/0.00</td><td>29.29/29.29</td></tr><tr><td>8-</td><td>0.00/3.58</td><td>0.00/12.55</td><td>0.00/13.16</td><td>0.00/0.00</td><td>0.00/29.29</td></tr><tr><td>9-</td><td>3.58/3.58</td><td>10.75/10.75</td><td>11.28/11.28</td><td>0.00/0.00</td><td>25.61/25.61</td></tr><tr><td>23+</td><td>3.58/3.58</td><td>12.55/12.55</td><td>13.16/13.16</td><td>0.00/0.00</td><td>29.29/29.29</td></tr><tr><td>24+</td><td>3.58/3.58</td><td>12.55/12.55</td><td>13.16/13.16</td><td>0.00/0.00</td><td>29.29/29.29</td></tr><tr><td>25+</td><td>3.58/3.58</td><td>14.34/14.34</td><td>15.04/15.04</td><td>0.00/0.00</td><td>32.96/32.96</td></tr><tr><td>26+</td><td>0.00/3.58</td><td>0.00/14.34</td><td>0.00/15.04</td><td>0.00/0.00</td><td>0.00/32.96</td></tr><tr><td>27+</td><td>3.58/3.58</td><td>16.13/16.13</td><td>16.93/16.93</td><td>0.00/0.00</td><td>36.64/36.64</td></tr><tr><td>0-</td><td>3.58/3.58</td><td>14.34/14.34</td><td>15.04/15.04</td><td>0.00/0.00</td><td>32.96/32.96</td></tr><tr><td>10-</td><td>0.00/3.58</td><td>0.00/14.34</td><td>0.00/15.04</td><td>0.00/0.00</td><td>0.00/32.96</td></tr><tr><td>17-</td><td>3.58/3.58</td><td>12.55/12.55</td><td>13.16/13.16</td><td>0.00/0.00</td><td>29.29/29.29</td></tr><tr><td>18-</td><td>3.58/3.58</td><td>12.55/12.55</td><td>13.16/13.16</td><td>0.00/0.00</td><td>29.29/29.29</td></tr><tr><td>19-</td><td>3.58/3.58</td><td>10.75/10.75</td><td>11.28/11.28</td><td>0.00/0.00</td><td>25.61/25.61</td></tr><tr><td>Total</td><td>89.50/114.56322.59/415.78</td><td></td><td>338.50/436.29</td><td>0.00/0.00</td><td>750.59/966.63</td></tr></table>

DETAILED COMMUNICATION COSTS (MB) OF SEQUENTIAL EDITING ON BDD100K WITH $\rho = 0 . 5$ . EACH COMMUNICATION ENTRY IS REPORTED AS F-SOSA-I/U VERSUS RETRAIN, WITH THE FIRST VALUE CORRESPONDING TO F-SOSA-I/U AND THE SECOND TO RETRAIN. MODEL UP. AND FISHER UP. DENOTE THE COMMUNICATION COSTS OF UPLOADING THE MODEL AND THE APPROXIMATE FISHER INFORMATION, RESPECTIVELY. IN THE REQUEST COLUMN, THE NUMBER IDENTIFIES THE AFFECTED CLIENT, WHILE + AND – DENOTE CLIENT ADDITION AND CLIENT REMOVAL, RESPECTIVELY  
TABLE XII
<table><tr><td rowspan="2">Request</td><td>Broadcast</td><td colspan="2">Selected Clients</td><td>Unselected Clients</td><td rowspan="2">Round Total</td></tr><tr><td>Downlink</td><td>Model Up.</td><td>Fisher Up.</td><td>Uplink</td></tr><tr><td>Initial</td><td>42.49/42.49</td><td>127.48/127.48</td><td>127.20/127.20</td><td>0.00/0.00</td><td>297.17/297.17</td></tr><tr><td>12+</td><td>0.00/42.49</td><td>0.00/127.48</td><td>0.00/127.20</td><td>0.00/0.00</td><td>0.00/297.17</td></tr><tr><td>13+</td><td>42.49/42.49</td><td>148.73/148.73</td><td>148.40/148.40</td><td>0.00/0.00</td><td>339.62/339.62</td></tr><tr><td>14+</td><td>0.00/42.49</td><td>0.00/148.73</td><td>0.00/148.40</td><td>0.00/0.00</td><td>0.00/339.62</td></tr><tr><td>15+</td><td>42.49/42.49</td><td>169.97/169.97</td><td>169.60/169.60</td><td>0.00/0.00</td><td>382.06/382.06</td></tr><tr><td>16+</td><td>42.49/42.49</td><td>169.97/169.97</td><td>169.60/169.60</td><td>0.00/0.00</td><td>382.06/382.06</td></tr><tr><td>3-</td><td>0.00/42.49</td><td>0.00/169.97</td><td>0.00/169.60</td><td>0.00/0.00</td><td>0.00/382.06</td></tr><tr><td>6-</td><td>42.49/42.49</td><td>148.73/148.73</td><td>148.40/148.40</td><td>0.00/0.00</td><td>339.62/339.62</td></tr><tr><td>7-</td><td>42.49/42.49</td><td>148.73/148.73</td><td>148.40/148.40</td><td>0.00/0.00</td><td>339.62/339.62</td></tr><tr><td>17+</td><td>0.00/42.49</td><td>0.00/148.73</td><td>0.00/148.40</td><td>0.00/0.00</td><td>0.00/339.62</td></tr><tr><td>18+</td><td>42.49/42.49</td><td>169.97/169.97</td><td>169.60/169.60</td><td>0.00/0.00</td><td>382.06/382.06</td></tr><tr><td>19+</td><td>0.00/42.49</td><td>0.00/169.97</td><td>0.00/169.60</td><td>0.00/0.00</td><td>0.00/382.06</td></tr><tr><td>1-</td><td>42.49/42.49</td><td>169.97/169.97</td><td>169.60/169.60</td><td>0.00/0.00</td><td>382.06/382.06</td></tr><tr><td>11-</td><td>42.49/42.49</td><td>148.73/148.73</td><td>148.40/148.40</td><td>0.00/0.00</td><td>339.62/339.62</td></tr><tr><td>16-</td><td>42.49/42.49</td><td>148.73/148.73</td><td>148.40/148.40</td><td>0.00/0.00</td><td>339.62/339.62</td></tr><tr><td>2-</td><td>42.49/42.49</td><td>127.48/127.48</td><td>127.20/127.20</td><td>0.00/0.00</td><td>297.17/297.17</td></tr><tr><td>20+</td><td>42.49/42.49</td><td>148.73/148.73</td><td>148.40/148.40</td><td>0.00/0.00</td><td>339.62/339.62</td></tr><tr><td>21+</td><td>42.49/42.49</td><td>148.73/148.73</td><td>148.40/148.40</td><td>0.00/0.00</td><td>339.62/339.62</td></tr><tr><td>22+</td><td>42.49/42.49</td><td>169.97/169.97</td><td>169.60/169.60</td><td>0.00/0.00</td><td>382.06/382.06</td></tr><tr><td>4-</td><td>42.49/42.49</td><td>148.73/148.73</td><td>148.40/148.40</td><td>0.00/0.00</td><td>339.62/339.62</td></tr><tr><td>8-</td><td>0.00/42.49</td><td>0.00/148.73</td><td>0.00/148.40</td><td>0.00/0.00</td><td>0.00/339.62</td></tr><tr><td>9-</td><td>42.49/42.49</td><td>127.48/127.48</td><td>127.20/127.20</td><td>0.00/0.00</td><td>297.17/297.17</td></tr><tr><td>23+</td><td>42.49/42.49</td><td>148.73/148.73</td><td>148.40/148.40</td><td>0.00/0.00</td><td>339.62/339.62</td></tr><tr><td>24+</td><td>0.00/42.49</td><td>0.00/148.73</td><td>0.00/148.40</td><td>0.00/0.00</td><td>0.00/339.62</td></tr><tr><td>25+</td><td>42.49/42.49</td><td>169.97/169.97</td><td>169.60/169.60</td><td>0.00/0.00</td><td>382.06/382.06</td></tr><tr><td>26+</td><td>42.49/42.49</td><td>169.97/169.97</td><td>169.60/169.60</td><td>0.00/0.00</td><td>382.06/382.06</td></tr><tr><td>27+</td><td>42.49/42.49</td><td>191.22/191.22</td><td>190.80/190.80</td><td>0.00/0.00</td><td>424.51/424.51</td></tr><tr><td>0-</td><td>42.49/42.49</td><td>169.97/169.97</td><td>169.60/169.60</td><td>0.00/0.00</td><td>382.06/382.06</td></tr><tr><td>10-</td><td>0.00/42.49</td><td>0.00/169.97</td><td>0.00/169.60</td><td>0.00/0.00</td><td>0.00/382.06</td></tr><tr><td>17-</td><td>42.49/42.49</td><td>148.73/148.73</td><td>148.40/148.40</td><td>0.00/0.00</td><td>339.62/339.62</td></tr><tr><td>18-</td><td>42.49/42.49</td><td>148.73/148.73</td><td>148.40/148.40</td><td>0.00/0.00</td><td>339.62/339.62</td></tr><tr><td>19-</td><td>42.49/42.49</td><td>127.48/127.48</td><td>127.20/127.20</td><td>0.00/0.00</td><td>297.17/297.17</td></tr><tr><td>Total</td><td>1019.76/1359.683696.95/4929.27</td><td></td><td>3688.71/4918.28</td><td>0.00/0.00</td><td>8405.42/11207.23</td></tr></table>