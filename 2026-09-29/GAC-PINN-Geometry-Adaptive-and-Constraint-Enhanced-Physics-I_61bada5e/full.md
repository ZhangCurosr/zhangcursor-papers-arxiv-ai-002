# GAC-PINN: Geometry-Adaptive and Constraint-Enhanced Physics-Informed Neural Networks

Yanxin Zhang<sup>a</sup>, Yong Zhang<sup>a</sup> and Houbiao Li<sup>a,∗</sup>

<sup>a</sup>University ofElectronic Science and Technology ofChina, Chengdu, 611731, Sichuan, China

A R T I C L E I N F O

Keywords:   
Physics-informed neural networks   
Spectral bias   
Adaptive grid mapping0   
Adaptive hard constraints Gaussian Fourier features   
Steep gradients and sharp interfaces

## A BS T R AC T

For systems with steep gradients, sharp interfaces, or severe spatio-temporal coupling, Physicsinformed neural networks (PINNs) sufer from spectral bias, geometric inflexibility, and boundary constraint conflicts, which undermine accuracy and convergence. To overcome these issues, we propose a geometry-adaptive and constraint-enhanced PINN (GAC-PINN). The framework comprises four components: a gradient-driven adaptive grid mapping (AGM) for difeomorphic point concentration with Jacobian regularization, an adaptive bandwidth hard-constraint ansatz with spatially-varying boundary transition widths, a Gaussian Fourier feature mapping as a spectral preconditioner to further enhance high-wavenumber representation, and an operatoraware router that automatically selects the appropriate hard-constraint construction based on whether the governing PDE contains temporal derivatives. An AGM callback mechanism and a three-stage training strategy ensure stable coordination. Benchmarks including the viscous Burgers equation, a sharp-peaked 2D Poisson problem, and the Allen-Cahn phase-transition equation show that GAC-PINN attains relative �<sup>2</sup> errors of $( 1 . 7 4 7 \pm 0 . 4 5 0 ) \times 1 0 ^ { - 4 }$ , (2.868 ± 0.947)×10<sup>−5</sup>, and (1.756±0.712)×10<sup>−3</sup>, respectively, consistently outperforming the baselines. Ablation studies further reveal that AGM alone yields a substantially lower error than residual based adaptive refinement (RAR), while RAR becomes beneficial only when combined with FFM, demonstrating a context-dependent module interaction. Convergence analysis verifies rapid error reduction and saturation with increasing resolution, establishing a practical adaptive framework for high-fidelity simulation of problems with localized sharp features in applied mechanics and computational physics.

## 1. Introduction

Accurate simulation of physical fields with steep gradients, sharp interfaces, and strong nonlinearities is critically important in numerous engineering disciplines. In aerospace engineering, the precise capture of shock waves and boundary layers is critical for predicting aerodynamic drag, heat flux, and structural integrity of high-speed vehicles; in electronic packaging, reliable thermal management requires high-fidelity solutions of the heat equation in the presence of localized hot spots and material interfaces; and in materials science, the modeling of phase separation and grain growth during alloy solidification hinges on the accurate resolution of propagating phase interfaces under extreme curvature-driven dynamics. These diverse applications share a common mathematical challenge: the eficient and accurate numerical solution of partial diferential equations (PDEs) whose solutions exhibit localized, multi-scale features that impose prohibitive resolution requirements on conventional mesh-based methods.

Physics-informed neural networks (PINNs), introduced by Raissi et al. [1], have emerged as a transformative paradigm for PDE solving. By embedding physical laws directly into the training loss and leveraging automatic diferentiation (AD) to compute PDE residuals, PINNs provide a mesh-free framework that circumvents the discretization and meshing burdens of traditional numerical methods. This elegant formulation has enabled promising results in various forward and inverse problems.

However, when deployed on the aforementioned problems with strong localized gradients and multiscale dynamics, standard PINNs sufer from three interrelated bottlenecks that severely compromise accuracy and convergence robustness:

• Spectral bias: Deep fully-connected networks prefer to learn low-frequency components due to the eigenvalue decay of the neural tangent kernel (NTK), causing pronounced oscillations near shocks or boundary layers.

• Geometric inflexibility: Fixed uniform collocation points fail to dynamically adapt to evolving solution features, leading to redundant sampling in smooth regions and insuficient resolution in high-gradient zones.

• Boundary/initial condition conflicts: Penalty-based soft constraints induce detrimental gradient competition between PDE residuals and boundary losses, while conventional fixed hard-constraint ansatz may introduce derivative contamination in high-order AD computations.

Existing works have achieved significant progress by tackling individual bottlenecks. For instance, Fourier feature mappings [2] mitigate spectral bias, residual-based adaptive sampling [3] optimizes collocation point distribution, and coordinate transformation techniques [4] enhance geometric flexibility. However, these strategies are often applied as independent modules or in a sequential pipeline, leaving room for a more synergistic integration that jointly considers geometric adaptation and spectral preconditioning within a unified training framework. In this work, we propose such an integration, with a particular focus on a gradient-driven difeomorphic mapping (AGM) that dynamically adapts collocation points based on real-time physical gradients. This mapping is designed to be compatible with and complementary to existing spectral embedding techniques, thereby providing a cohesive framework rather than a simple aggregation of disjoint components.

Building on this integrated perspective, this work proposes a Geometry-Adaptive and Constraint-Enhanced PINN (GAC-PINN). The framework consists of four components integrated within a unified training pipeline that directly target the three bottlenecks: an operator-aware router that selects the appropriate hard-constraint construction based on the presence of temporal derivatives; a gradient-driven adaptive grid mapping (AGM) that difeomorphically concentrates collocation points in high-residual regions while preventing geometric degeneration; a Gaussian Fourier feature mapping (FFM) that reshapes the NTK spectrum to mitigate low-frequency bias; and an adaptive hardconstraint ansatz with a learnable bandwidth and a stop-gradient operator that facilitates stable high-order automatic diferentiation. A dedicated three-stage training strategy ensures stable coordination among these components. The main contributions of this work are threefold:

• Methodologically, we establish a new PINN framework that achieves tight integration of geometric adaptation, spectral preconditioning and constraint enforcement within a single optimization pipeline;

• Algorithmically, we propose a closed-loop manifold evolution mechanism——AGM driven by real-time physical gradients, distinct from static coordinate transformations or discrete resampling;

• Experimentally, extensive benchmarks on the viscous Burgers equation, a sharp 2D Poisson problem, the Allen-Cahn phase-field equation, and the 2D Navier-Stokes equations demonstrate that GAC-PINN achieves competitive and improved accuracy compared to re-implemented baselines under aligned settings, while ablation and computational analyses confirm the synergistic contributions of each module and competitive training eficiency.

The remainder of this paper is organized as follows. Section 2 reviews the relevant preliminaries, including PINNs, NTK theory, and difeomorphic mappings. Section 3 presents the detailed architecture and algorithmic implementation of GAC-PINN. Section 4 describes the experimental setup. Section 5 reports comprehensive numerical results and comparisons. Finally, Section 6 provides discussion on the implications and limitations of the proposed approach, also outlines future research directions.

## 2. Methodology

## 2.1. Physics-informed neural networks

Consider a spatiotemporal domain $\Omega \times [ 0 , T ]$ , with spatial coordinates $x \in \mathbb { R } ^ { d }$ and time $t \in [ 0 , T ]$ . Let $X = \left( x , t \right)$ denote the spatiotemporal coordinate. The general PDE initial-boundary value problem is expressed as:

$$
\mathcal { P } [ u ] ( X ) = 0 , \quad X \in \Omega \times ( 0 , T ] ,\tag{1}
$$

$$
\begin{array} { r } { \beta [ u ] ( X ) = 0 , \quad X \in \partial \Omega \times ( 0 , T ] , } \end{array}\tag{2}
$$

$$
\begin{array} { r } { T [ u ] ( X ) = 0 , \quad x \in \Omega , t = 0 , } \end{array}\tag{3}
$$

where  denotes the diferential operator,  and  represent boundary and initial operators, respectively, and �(�) is the unknown physical field.A standard PINN [1] approximates the unknown solution by a fully connected neural network $u ( X ; \Theta )$ with trainable parameters Θ. The total loss function is a weighted sum of three components:

$$
\begin{array} { r } { \mathcal { L } _ { P I N N } = \lambda _ { p d e } \mathcal { L } _ { p d e } + \lambda _ { b c } \mathcal { L } _ { b c } + \lambda _ { i c } \mathcal { L } _ { i c } , } \end{array}\tag{4}
$$

where the PDE residual loss quantifies the deviation of the governing equation at collocation points:

$$
\mathcal { L } _ { p d e } = \frac { 1 } { N _ { p d e } } \sum _ { i = 1 } ^ { N _ { p d e } } \big \| \mathcal { P } [ u ] ( X _ { i } ) \big \| _ { 2 } ^ { 2 } ,\tag{5}
$$

and the boundary and initial losses are defined analogously. The weighting coeficients $\lambda _ { p d e } , \lambda _ { b c }$ , and $\lambda _ { i c }$ balance the competing loss terms, and the network parameters are optimized via back-propagation to minimize this composite objective.

## 2.2. Neural tangent kernel theory

The neural tangent kernel (NTK) provides a theoretical tool for analyzing the training dynamics of infinite-width neural networks. For a fully connected network $u ( X ; \Theta )$ , the NTK is defined as [5]:

$$
\Theta _ { N T K } ( X , X ^ { \prime } ) = \left. \frac { \partial u ( X ; \Theta ) } { \partial \Theta } , \frac { \partial u ( X ^ { \prime } ; \Theta ) } { \partial \Theta } \right. .\tag{6}
$$

Under gradient descent training with an infinitesimal learning rate, the NTK remains approximately constant during training, i.e., $\Theta _ { t } \approx \Theta _ { 0 }$ . For a training set of � samples,the empirical NTK matrix $\Theta \in \mathbb { R } ^ { N \times N }$ is symmetric positive semi-definite and admits the eigendecomposition [6]:

$$
\Theta = \sum _ { k = 1 } ^ { N } \lambda _ { k } \phi _ { k } \phi _ { k } ^ { T } ,\tag{7}
$$

with eigenvalues $\lambda _ { 1 } \geq \lambda _ { 2 } \geq \cdots \geq \lambda _ { N } \geq 0$ and corresponding orthogonal eigenvectors $\phi _ { k }$ . Projecting the residual $e ( t ) = u ( t ) - y$ onto this eigenbasis yields modal coeficients $\hat { \epsilon } _ { k } ( t ) = \phi _ { k } ^ { \top } e ( t )$ . Due to eigenvector orthogonality, the high-dimensional training dynamics decouple into � independent scalar ordinary diferential equations:

$$
\frac { d \hat { \epsilon } _ { k } ( t ) } { d t } = - \eta \lambda _ { k } \hat { \epsilon } _ { k } ( t ) ,\tag{8}
$$

which admits the closed-form solution

$$
\hat { \epsilon } _ { k } ( t ) = \hat { \epsilon } _ { k } ( 0 ) e ^ { - \eta \lambda _ { k } t } .\tag{9}
$$

This theory reveals that high-frequency modes corresponding to small eigenvalues converge at exponentially slower rates than low-frequency modes, mathematically establishing the spectral bias of standard PINNs — a limitation that motivates the Fourier feature embedding introduced in Section 3.5.

## 2.3. Difeomorphic coordinate transformation

## 2.3.1. Definition of difeomorphic mapping

A mapping $\mathcal { M } : \Omega _ { \Xi } \to \Omega _ { X }$ from the computational coordinates $\Xi = ( \xi , \eta )$ to the physical coordinates $X = \left( x , t \right)$ is difeomorphic [4] if it satisfies: (i) bijectivity — each computational point maps to a unique physical point and vice versa; (ii) infinite diferentiability — all derivatives exist and are continuous throughout the domain; and (iii) invertible Jacobian — the Jacobian matrix $\mathbf { J } = \partial X / \partial \boldsymbol { \Xi }$ satisfies det $\left( \mathbf { J } \right) > 0$ everywhere, preventing coordinate folding or local volume collapse.

## 2.3.2. Volume transformation and sampling density

Unlike conventional fixed-grid methods where collocation points are prescribed directly in the physical domain, this approach adopts a computational-coordinate perspective commonly used in r-adaptivity. Let $\Xi \in \Omega _ { \Xi }$ denote the

computational coordinates with a uniform reference distribution, and let $M : \Omega _ { \Xi }  \Omega _ { X }$ be a difeomorphic mapping from the computational domain to the physical domain:

$$
X = \mathcal { M } ( \Xi ) , \quad \Xi \in \Omega _ { \Xi } \subset \mathbb { R } ^ { d } , \quad X \in \Omega _ { X } \subset \mathbb { R } ^ { d } .\tag{10}
$$

The Jacobian matrix of this mapping is

$$
\mathbf { J } ( \Xi ) = \frac { \partial X } { \partial \Xi } = \left[ \begin{array} { c c c c } { \displaystyle \frac { \partial X _ { 1 } } { \partial \Xi _ { 1 } } } & { \displaystyle \frac { \partial X _ { 1 } } { \partial \Xi _ { 2 } } } & { \cdots } & { \displaystyle \frac { \partial X _ { 1 } } { \partial \Xi _ { d } } } \\ { \displaystyle \frac { \partial X _ { 2 } } { \partial \Xi _ { 1 } } } & { \displaystyle \frac { \partial X _ { 2 } } { \partial \Xi _ { 2 } } } & { \cdots } & { \displaystyle \frac { \partial X _ { 2 } } { \partial \Xi _ { d } } } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { \displaystyle \frac { \partial X _ { d } } { \partial \Xi _ { 1 } } } & { \displaystyle \frac { \partial X _ { d } } { \partial \Xi _ { 2 } } } & { \cdots } & { \displaystyle \frac { \partial X _ { d } } { \partial \Xi _ { d } } } \end{array} \right] , \quad \operatorname* { d e t } ( \mathbf { J } ( \Xi ) ) > 0 , \quad \forall \Xi \in \Omega _ { \Xi } ,\tag{11}
$$

where the positivity condition ensures bijectivity and prevents local collapse.

By the change-of-variables formula, the physical and computational volume elements satisfy

$$
d V _ { X } = \mid \operatorname* { d e t } ( \mathbf { J } ( \Xi ) ) \mid d V _ { \Xi } .\tag{12}
$$

Let $\rho _ { \Xi } ( \Xi ) \equiv \mathrm { c o n s t }$ denote the uniform sampling density in the computational domain. Since the total number of collocation points is invariant under , the physical-space density $\rho _ { X } ( X )$ satisfies

$$
\rho _ { X } ( X ) d V _ { X } = \rho _ { \Xi } ( \Xi ) d V _ { \Xi } ,\tag{13}
$$

which yields

$$
\rho _ { X } ( X ) = \frac { \rho _ { \Xi } ( \Xi ) } { | \operatorname* { d e t } ( \mathbf { J } ( \Xi ) ) | } .\tag{14}
$$

Formula (14) reveals the core principle of adaptive node refinement: the physical sampling density is inversely proportional to the local Jacobian determinant. Regions with large det(�) thus receive denser collocation points, concentrating computational resources in steep-gradient areas, while regions with small det(�) are automatically coarsened. In essence, the Jacobian determinant acts as a magnification factor that stretches the uniform computational grid into a solution-adaptive non-uniform physical grid.

Motivated by the classical equidistribution principle [7, 8], we seek a mapping that minimizes the coeficient of variation of the weighted volumes �(�)| det(�(Ξ))| across all collocation points, where �(�) is a monitor function reflecting the local gradient magnitude. The AGM module introduced in Section3.4 explicitly performs this minimization, ensuring that high-gradient regions receive a proportionally larger share of computational resources.

## 3. The GAC-PINN framework

This work aims to develop a high-precision, adaptive numerical solver for PDEs with steep gradients, strong nonlinearities, and multiscale features. This methodology makes original contributions to adaptive discretization and integrates it with spectral preconditioning into a unified PINN framework as shown in Fig. 1. Among these, the hardconstraint ansatz is employed as an engineering enhancement to stabilize high-order diferentiation.

![](images/3366629d9c4e7abe965378369faae87b9eecbe1962a665d7f19fe5453ff61b4d.jpg)  
Fig. 1. GAC-PINN Framework

## 3.1. Overall architecture

Within this architecture, the neural network serves as the flexible functional approximator, while four components collectively address the above bottlenecks:

• Operator-aware hard-constraint router: Classifies the governing PDE by its dependence on temporal derivatives and selects the appropriate hard-constraint construction (time-adaptive bandwidth for evolution equations, pure spatial distance for steady-state problems).

• Adaptive grid mapping network (AGM): Driven by physical gradients through the equidistribution principle, dynamically concentrates collocation points toward high-gradient regions with a Jacobian safety barrier against degeneration.

• Gaussian Fourier feature mapping (FFM): Reshapes the NTK spectral distribution through Gaussian random projection, breaking the low-frequency bias.

• Adaptive hard-constraint ansatz: Employs a learnable boundary transition bandwidth to facilitate stable highorder diferentiation.

## 3.2. Operator-aware hard-constraint routing

The construction of the hard-constraint ansatz depends on the temporal characteristics of the governing PDE. Evolution equations benefit from a time-adaptive bandwidth that gradually releases the initial condition as � increases,

whereas steady-state problems require only a spatial distance function. To accommodate both classes within a unified framework without manual reconfiguration, we introduce an operator-aware router that classifies the governing operator and selects the corresponding hard-constraint construction.

The classification is based on the operator’s dependence on temporal derivatives:

$$
\chi _ { \mathcal { P } } = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { i f ~ } \partial \mathcal { P } / \partial ( \partial _ { t } u ) \equiv 0 \quad \mathrm { ( s t e a d y \mathrm { - } s t a t e ) } , } \\ { 1 , } & { \mathrm { i f ~ } \partial \mathcal { P } / \partial ( \partial _ { t } u ) \not \equiv 0 \quad \mathrm { ( t i m e - d e p e n d e n t ) } . } \end{array} \right.\tag{15}
$$

The discriminant $\chi _ { \mathcal { P } }$ is determined symbolically from the PDE definition and incurs no additional computational cost. Based on $\chi _ { \mathcal { P } }$ , the framework selects the corresponding hard-constraint ansatz described in Section 3.6:

• If $\chi _ { \mathcal { P } } = 1$ (time-dependent), the time-adaptive bandwidth ansatz is applied. A learnable gating network predicts a spatially-varying bandwidth $\kappa _ { a d a p t i v e } ( X )$ from the input coordinates, and the hard constraint gradually releases the initial condition as � increases.

• If $\chi _ { \mathcal { P } } = 0$ (steady-state), a pure spatial distancefunction is used, and all temporal dependence is removed.

This routing mechanism ensures that the same framework handles both elliptic and evolution problems without manual reconfiguration, while preserving the exact satisfaction of boundary and initial conditions in both cases.

## 3.3. Gradient-driven adaptive grid mapping network (AGM)

## 3.3.1. Gradient-drivenfeedback monitoring mechanism

The driving force for manifold adaptation originates from the real-time first-order spatial derivatives of the physical network output $u _ { N N }$ with respect to spatial coordinates �. At the current training iteration, automatic diferentiation extracts the spatial gradient magnitude:

$$
\| \nabla _ { x } u _ { N N } \| _ { 2 } = \sqrt { \sum _ { i = 1 } ^ { d } \left( \frac { \partial u _ { N N } } { \partial x _ { i } } \right) ^ { 2 } } .\tag{16}
$$

To construct a dimensionless mapping from physical gradients to spatial grid density, we define a dynamic feedback indicator $\omega ( X )$ with global adaptive scaling:

$$
\omega ( X ) = 1 . 0 + \alpha _ { g r a d } \times \frac { \| \nabla _ { x } u _ { N N } \| _ { 2 } } { \operatorname* { m a x } _ { \Omega _ { x } } \| \nabla _ { x } u _ { N N } \| _ { 2 } + 1 0 ^ { - 8 } } ,\tag{17}
$$

with gradient amplification coeficient $\alpha _ { g r a d } \ = \ 4 . 0$ . This indicator normalizes by the current maximum gradient magnitude, strictly confining �(�) to the interval [1.0, 5.0].

## 3.4. Adaptive difeomorphic coordinate transformation

Given that primary high-frequency physical features evolve strongly along spatial directions, this framework maintains absolute independence of the time coordinate to prevent spatial distortion from interfering with the temporal evolution. The spatial nonlinear transformation takes the form:

$$
( x , y ) = ( \xi , \eta ) + \alpha \cdot \operatorname { t a n h } \left( \mathcal { A } _ { G M } ( \xi , \eta ; \Theta _ { M A P } ) \right) ,\tag{18}
$$

where $\mathcal { A } _ { G M }$ is a multilayer perceptron with trainable parameters $\Theta _ { M A P }$ , and $\alpha = 2 . 0$ controls the maximum local stretching magnitude.

![](images/454aefc2b9c62f109b1d13f4b1445db1a200103fecc4724d7ca91678f1df36f7.jpg)  
Fig. 2. AGM geometric coordinate transformation

As illustrated in Fig. 2, this transformation establishes a difeomorphic mapping from the uniform computational manifold (�, �) to the physical space (�, �). It acts as a space-stretching operator: the uniform lattice in the computational domain is projected onto the physical domain, yielding adaptive node clustering toward high-gradient regions (e.g., the central spike). In these regions, the Jacobian determinant det(�) increases, reflecting the local volumetric magnification that concentrates collocation points. Conversely, smooth regions correspond to smaller det(�) and are automatically coarsened. Meanwhile, the computational manifold retains a perfectly uniform mesh, which is an essential prerequisite for subsequent FFM spectral reconstruction to guarantee orthogonality and numerical conditioning of the basis functions.

To quantitatively enforce this adaptive redistribution, the Jacobian matrix � (from computational to physical coordinates) is defined as:

$$
\mathbf { J } = { \frac { \partial ( x , y ) } { \partial ( \xi , \eta ) } } = { \left[ \begin{array} { l l } { { \cfrac { \partial x } { \partial \xi } } } & { { \cfrac { \partial x } { \partial \eta } } } \\ { { \cfrac { \partial y } { \partial \xi } } } & { { \cfrac { \partial y } { \partial \eta } } } \end{array} \right] } .\tag{19}
$$

Motivated by the equidistribution principle, we construct a weighted volumetric measure $V _ { i } = \omega ( \mathbf { X } _ { i } ) \cdot \operatorname* { d e t } ( \mathbf { J } _ { i } )$ to characterize the local solution complexity. The AGM parameters are then optimized by a composite loss function that jointly minimizes the coeficient of variation of these volumes and enforces a Jacobian safety barrier:

$$
\mathcal { L } _ { A G M } = 2 . 0 \cdot \mathcal { L } _ { e q u i } + \lambda _ { b a r r i e r } \cdot \mathcal { L } _ { g u a r d } ,\tag{20}
$$

(21)

where

$$
\mathcal { L } _ { e q u i } = \frac { \sqrt { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( V _ { i } - \bar { V } ) ^ { 2 } } } { \bar { V } + \epsilon _ { 0 } } ,\tag{22}
$$

Minimizing $\mathcal { L } _ { e q u i }$ drives the physical grid volumes to conform to the equidistribution criterion, thereby automatically inducing local refinement in high-gradient regions, where the computational lattice and det(�) increases to concentrate collocation points. Conversely, in smooth regions where det(�) is small, the grid is automatically coarsened. The safety barrier, with a lower bound $J _ { s a f e \_ m i n } = 0 . 0 2$ , imposes a quadratic penalty when the local Jacobian determinant approaches zero from above:

$$
\mathcal { L } _ { g u a r d } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ \mathrm { R e L U } \big ( J _ { s a f e \_ m i n } - \mathrm { d e t } ( \mathbf { J } _ { i } ) \big ) \right] ^ { 2 } .\tag{23}
$$

This strictly prevents det(�) from vanishing or turning negative, thus rigorously preserving the bijectivity, smoothness, and topological integrity of the mapping over the entire domain.

## 3.4.1. AGM dynamic evolution callback mechanism

To eficiently realize the closed-loop iteration of physical gradient capture, geometric grid deformation, coordinate update, and manifold reconstruction during training, a unified callback mechanism is designed.

In Stage 1, the AGM callback executes the following closed loop every 100 training steps:

• Gradient field capture and forward decoupling: Pause back-propagation of the governing equation residual, pass the current collocation point batch through the physical backbone network, activate the first-order automatic diferentiation graph, capture the latest spatial gradient magnitudes, detach them from the physical optimization graph, and convert them to static driving sources.

• Monitoring operator reconstruction and manifold update: Substitute the extracted gradient field into the governing equations to compute the dynamic monitoring indicator �(�), calculate the Jacobian determinant of the current transformation, and trigger an independent variational optimization step for the mapping network parameters $\Theta _ { M A P }$ within the callback.

• Coordinate mapping dynamic overwrite: $\operatorname { A s } \Theta _ { M A P }$ updates, the manifold coordinates $\xi$ output by the mapping network are refreshed in real time and passed as new input features to the subsequent FFM, reconstructing a smoother manifold space more amenable to spectral approximation for the next 100 steps of physical field solution.

The mapping network does not participate in PDE loss back-propagation of the backbone network; its parameter updates are independently driven by the AGM callback mechanism.

## 3.5. Gaussian Fourier feature mapping (FFM)

NTK theory indicates that deep fully connected networks exhibit exponentially decaying convergence rates for highfrequency features when approximating complex physical fields with multi-scale, high-wavenumber components—the spectral bias phenomenon. To achieve "pseudo-low-frequency" mapping of flow field features, GAC-PINN introduces a Gaussian Fourier feature embedding layer [2] before the transformed manifold coordinates enter the backbone network. Let the transformed spatiotemporal manifold coordinate vector be $z _ { M A P } = [ \xi , t ] ^ { T } \in \mathbb { R } ^ { d }$ . The high-dimensiona spectral feature projection is defined as:

$$
\gamma ( z _ { M A P } ) = \left[ \frac { \sin ( 2 \pi B z _ { M A P } ) } { \cos ( 2 \pi B z _ { M A P } ) } \right] \in \mathbb { R } ^ { 2 m } ,\tag{24}
$$

where $m = 1 2 8$ is the half-dimension of the Fourier feature mapping, yielding an embedded feature dimension of $2 m = 2 5 6$ . The projection matrix $B \in \mathbb { R } ^ { m \times d }$ is generated once during initialization and fixed as a static constant, with elements sampled from an isotropic multivariate Gaussian distribution:

$$
B _ { j k } \sim \mathcal N ( 0 , \delta _ { s c a l e } ^ { 2 } ) ,\tag{25}
$$

with Gaussian standard deviation $\delta _ { s c a l e } = 5 . 0$ . By explicitly mapping the low-dimensional spatiotemporal domain to the high-dimensional spectral space spanned by the Gaussian prior basis, FFM pre-stretches the frequency response distribution of the input signal, equipping the physical backbone network with the capability to capture highwavenumber information of strongly nonlinear physical interfaces from the early stages of training.

## 3.6. Adaptive hard-constraint network

The hard-constraint ansatz has been widely applied in the field of PINN research to eliminate boundary loss and prevent weight collapse. This work employs a variant with a learnable bandwidth and a stop-gradient operator, which is an engineering enhancement intended to stabilize high-order diferentiation.

## 3.6.1. Gradient-truncated hard constraintsfor non-periodic boundary conditions

For non-periodic boundaries (e.g., homogeneous Dirichlet boundaries for 2D Poisson, fixed-value boundaries for 1D Allen-Cahn), the general hard-constraint ansatz is defined as [9, 10]:

$$
u ( X ) = g ( X ) + t \cdot { \cal D } _ { s p a c e } ( X ; \kappa ( X ) ) \cdot \mathcal { N } _ { p h y } ( \gamma ( z _ { M A P } ) ; \Theta _ { P H Y } ) ,\tag{26}
$$

where $g ( X )$ is the basis function exactly satisfying initial conditions, $\mathcal { N } _ { p h y }$ is the physical backbone network, $\gamma ( \cdot )$ is the Fourier feature embedding mapping, and $z _ { M A P }$ is the transformed manifold coordinate from AGM. The spatia boundary distance factor $\mathcal { D } _ { s p a c e } ( X )$ is constructed as:

$$
\mathcal { D } _ { s p a c e } ( X ) = 1 . 0 - \exp \left( - \lfloor \kappa _ { a d a p t i v e } ( X ) \rfloor _ { s g } \cdot \mathcal { D } _ { m e s h } ( X ) \right) ,\tag{27}
$$

where $D _ { m e s h } ( X )$ is the distance from the physical space coordinate to the boundary. The bandwidth coeficient $\kappa _ { a d a p t i v e }$ is predicted in real time by a lightweight gating network:

$$
\kappa _ { a d a p t i v e } ( X ) = \kappa _ { m i n } + ( \kappa _ { m a x } - \kappa _ { m i n } ) \cdot { \mathrm { S i g m o i d } } \left( { \mathcal { N } } _ { g a t e } ( \nu _ { g a t e } ; \Phi _ { g a t e } ) \right) ,\tag{28}
$$

with input vector $\nu _ { g a t e } = [ x , t ]$ comprising spatial and temporal coordinates; $\Phi _ { g a t e }$ denotes the trainable gating network parameters, and $\lfloor \cdot \rfloor _ { s g }$ denotes the stop-gradient operator.

When computing high-order spatial derivatives in the PDE operator [�] (e.g., $\partial ^ { 2 } u / \partial x ^ { 2 }$ or $\partial ^ { 4 } u / \partial x ^ { 4 } )$ via automatic diferentiation, $\lfloor \kappa _ { a d a p t i v e } ( X ) \rfloor _ { s g }$ is treated as a spatial constant independent of network parameters. This design cuts the gradient propagation path from $\Phi _ { g a t e }$ to the high-order residual automatic diferentiation graph, mathematically eliminating the potential influence of the bandwidth prediction network on the numerical stifness of high-order PDE derivative terms.

## 3.6.2. Intrinsic periodic constructionfor periodic boundary conditions

For the periodic boundary conditions $u ( - 1 , t ) = u ( 1 , t )$ of the 1D Burgers equation, our framework employs a fully periodic harmonic basis substitution rather than algebraic truncation terms. The periodic ansatz is defined as:

$$
u ( X ) = e ^ { - \lambda t } \cdot g ( x ) + \left( 1 . 0 - e ^ { - \lambda t } \right) \cdot \mathcal { N } _ { p h y } \left( \sin ( \pi x ) , \cos ( \pi x ) , \gamma _ { t } ( t ) ; \Theta _ { P H Y } \right) ,\tag{29}
$$

where $g ( x ) = - \sin ( \pi x )$ exactly matches the initial condition $u ( x , 0 ) = - \sin ( \pi x )$ . Since the network input layer is explicitly constructed as sin(��) and cos(��) basis forms, the output intrinsically satisfies full periodic continuity $u ( x + 2 , t ) \equiv u ( x , t )$ regardless of weight evolution. The temporal decay factor $e ^ { - \lambda t }$ ensures exact recovery of initial conditions at $t = 0$ , while $1 . 0 - e ^ { - \lambda t }$ progressively delegates solution evolution to the backbone network as � increases, simultaneously imposing initial and periodic boundary conditions without additional penalty terms.

## 3.7. Three-stage training strategy

To ensure stable convergence of the relative $L ^ { 2 }$ error to high precision for strongly nonlinear evolution equations, we adopt a three-stage training strategy:

• Stage 1: Adam with AGM cooperative evolution: Physical network weights $\Theta _ { P H Y }$ and mapping network weights $\Theta _ { M A P }$ evolve asynchronously in a dual-track manner. AGM performs ofline forward updates via the callback mechanism every 100 training steps. During this stage, the system conducts global topological exploration over the large-scale spatiotemporal domain; high-gradient regions are initially localized and stretched on the geometric manifold, establishing the fundamental solution topology.

• Residual-based adaptive refinement (RAR) [11]: After Stage 1, one RAR step is performed: 60,000 candidate points are sampled from the full domain, PDE residual magnitudes are computed, and the 2,500 points with the largest residuals are appended to the training set, focusing computational resources on regions with maximal physical residuals.

• Stage 2: Freeze manifold, fine-tune physical layers: All trainable parameters $\Theta _ { M A P }$ of AGM are frozen, eliminating potential high-frequency oscillations from coordinate transformation grids in late training and fixing the manifold geometry. Computational resources are concentrated on resolving high-gradient regions captured by RAR using Adam fine-tuning of $\Theta _ { P H Y }$

• Stage 3: L-BFGS final convergence, $N _ { 3 } = 3 \small { , } 0 0 0$ iterations: The L-BFGS second-order optimizer is employed for final convergence with full-batch computation, leveraging curvature information to achieve precise local minimization of $\mathcal { L } _ { P D E }$ . Only $\{ \Theta _ { P H Y } , \Phi _ { g a t e } \}$ are updated; $\Theta _ { M A P }$ remains frozen to preserve the established manifold geometry.

## 3.8. Algorithm

The complete GAC-PINN training procedure is summarized in Algorithm 1.   
Algorithm 1: GAC-PINN Training Algorithm   
Input: PDE operator , computational domain Ω and time interval [0, �], boundary and initial conditions,   
total collocation points $\mathcal { N } _ { p d e } ,$ hyperparameters $\lambda _ { e q u i } , \lambda _ { b a r r i e r } , J _ { s a f e _ { \mathrm { . } } }$ \_���   
Output: Solution space $u \in U$ , neural network parameters $\Theta _ { N N }$   
1 Initialize physical network $\mathcal { N } _ { p h y }$ (parameters $\Theta _ { p h y } ) .$ , AGM network $\mathcal { N } _ { A G M }$ (parameters $\Theta _ { A G M } )$ , bandwidth   
gating network $\mathcal { N } _ { g a t e }$ (parameters $\Phi _ { g a t e } ) { : }$   
2 Generate Fourier projection matrix � by fixed sampling $B _ { j k } \sim \mathcal N ( 0 , \delta _ { s c a l e } ^ { 2 } )$ , construct FFM embedding $\gamma ( \cdot ) ;$   
3 Select hard-constraint ansatz according to boundary topology, construct output transformation , bind into   
complete forward propagation chain:   
4 $u _ { N N } ( X ) = \mathcal { T } ( \mathcal { N } _ { p h y } ( \gamma ( \mathcal { N } _ { A G M } ( X ; \Theta _ { A G M } ) ) ; \Theta _ { p h y } ) ; \Phi _ { g a t e } ) ;$   
5 Stage 1 (cooperative training, $N _ { 1 } = 1 2 0 0 0 \rangle$   
6 for $i t e r = 1$ to $N _ { 1 }$ do   
7 fix $\Theta _ { A G M } ,$ , update $\Theta _ { p h y } , \Phi _ { g a t e }$ with Adam minimizing $\begin{array} { r } { \mathcal { L } _ { P D E } = \frac { 1 } { \mathcal { N } _ { p d e } } \sum _ { i = 1 } ^ { \mathcal { N } _ { p d e } } \| \mathcal { P } [ u _ { N N } ] ( \boldsymbol { X } _ { i } ) \| _ { 2 } ^ { 2 } ; } \end{array}$   
8 if ���� mod $1 0 0 = 0$ then   
9 fix $\Theta _ { p h y } , \Phi _ { g a t e } ,$ update $\Theta _ { A G M }$ with Adam minimizing $\mathcal { L } _ { A G M } = 2 . 0 \cdot \mathcal { L } _ { e q u i } + \lambda _ { b a r r i e r } \cdot \mathcal { L } _ { g u a r d } ;$   
10 end   
11 end   
12 Execute RAR: select $N _ { r e f i n e } = 2 5 0 0$ points with largest residuals from $N _ { c a n d } = 6 0 0 0 0$ candidate points,   
append to training set;   
13 Stage 2 (frozen mapping fine-tuning, $N _ { 2 } = 5 0 0 0 )$ : freeze $\Theta _ { A G M }$ , update $\Theta _ { p h y } , \Phi _ { g a t e }$ with Adam $( \mathrm { l r } = 1 0 ^ { - 4 } )$   
minimizing $\mathcal { L } _ { P D E } ;$   
14 Stage 3 (L-BFGS refinement, $N _ { 3 } = 3 0 0 0 )$ : update all parameters $\{ \Theta _ { p h y } , \Phi _ { g a t e } \}$ with L-BFGS minimizing   
$\mathcal { L } _ { P D E } ;$   
15 return trained GAC-PINN model.

## 4. Experimental setup

## 4.1. Computational environment

All experiments were developed and tested on a Linux computing platform equipped with an NVIDIA Tesla V100 GPU, utilizing PyTorch and the DeepXDE scientific computing library. The benchmark datasets and problem configurations for the Burgers equation, Allen-Cahn equation, and two-dimensional Poisson equation were adopted from the open-source code repository associated with the gradient-enhanced physics-informed neural networks (gPINN) work by Yu et al. [12], which is publicly available at https: $/ / \mathrm { g } .$ ithub.com/lu-group/gpinn. For the two-dimensional unsteady cylinder-wake benchmark, we adopt the governing-equation configuration from PINNsFormer [13], while the reference ground-truth flow-field data originate from the Nek5000 simulation database of Raissi et al. [1]. To eliminate the influence of random perturbations on optimization trajectories, all experiments were performed with five independent random seeds, and results are reported as mean ± standard deviation. For clarity, the error distribution figures and the specific error values reported in Section 5 are obtained from a single representative run with the same fixed random seed.

## 4.2. Benchmark problems and baselines

## 4.2.1. PDE benchmark problems

• 1D Burgers equation: The kinematic viscosity is set to $\nu = 0 . 0 1 / \pi .$ , testing the model’s capability for localized capture of nonlinear fluid shock fronts under convection-dominated conditions.

• 1D Allen-Cahn equation: This equation includes a high-order nonlinear cubic reaction source term $( \epsilon = 0 . 0 0 1 )$ with sharp phase interface rotation and spatiotemporal evolution, evaluating the eficacy of the time-adaptive hard constraint.

• 2D Poisson equation: The method of manufactured solutions introduces a high-order exponential parameter $( a = 1 0 )$ , constructing an extremely sharp spatial singularity peak at the domain center to assess the manifold adaptive compression performance of the model for complex static spatial gradients.

• 2D Navier-Stokes equation: The unsteady flow past a cylinder is considered with parameters $\lambda _ { 1 } = 1$ and $\lambda _ { 2 } = 0 . 0 1$ , serving as a benchmark to evaluate the model’s generalization capability for complex fluid dynamics and pressure field reconstruction.

## 4.2.2. Baseline models

Five representative baseline models spanning the evolutionary spectrum from traditional penalty methods to recent dynamic weighting and adaptive sampling strategies were selected for comparison:

• Vanilla PINN (Raissi et al., 2019): Standard fully connected architecture with mean squared error soft constraints for boundary conditions.

• GPINN (Yu et al., 2022): Explicitly embeds first-order spatial derivatives of PDE residuals as gradient supervision in the total loss.

• Causal PINN (Wang et al., 2024): Explicit temporal weighting scheme based on forward-moving temporal residuals.

• RAR-PINN (Mao & Meng, 2023): Discrete incremental greedy resampling strategy dynamically appending high-residual collocation points.

• RAMS-PINN (Ouyang et al., 2026): Dynamically modulates loss term weights or multi-scale basis functions using real-time residual magnitude evolution.

Moreover, for the 2D Navier-Stokes equation, which involves strong convective unsteadiness and vortex shedding dynamics that difer substantially from the preceding benchmark problems, we adopt a separate set of baselines specifically designed for fluid mechanics applications. These include PINNsformer [13], a transformer-based architecture for PDE solving; DD-PINN [14], which employs a domain-decomposition strategy; and TSA-PINN [15], which incorporates trainable sinusoidal activation functions. This targeted selection ensures physically meaningful comparisons tailored to Navier-Stokes problem.

Due to the unavailability of open-source code for DD-PINN and the irreproducibility of PINNsformer’s reported results in our environment, we directly cite their published numerical results under the same initial and boundary conditions, sampling strategies, and relative $L _ { 2 }$ error metrics, rendering cross-paper comparisons objectively valid.

## 4.3. Evaluation metrics

## 4.3.1. Relative $L ^ { 2 }$ error

The relative $L ^ { 2 }$ error is adopted as the primary accuracy metric:

$$
\epsilon _ { L ^ { 2 } } = \frac { | | u _ { p r e d } - u _ { t r u e } | | _ { 2 } } { | | u _ { t r u e } | | _ { 2 } } = \sqrt { \frac { \sum _ { i = 1 } ^ { N } | u _ { p r e d } ( X _ { i } ) - u _ { t r u e } ( X _ { i } ) | ^ { 2 } } { \sum _ { i = 1 } ^ { N } | u _ { t r u e } ( X _ { i } ) | ^ { 2 } } } .\tag{30}
$$

## 4.3.2. Convergence order

To quantify the error decay with mesh refinement, we define the convergence order � via the power-law relationship:

$$
E = C \cdot N _ { p d e } ^ { - p } ,\tag{31}
$$

where � is the relative $L ^ { 2 }$ error and � is a constant. Taking logarithms gives log $E = \log C - p$ log $N _ { p d e } ;$ thus $p$ is estimated as the negative slope of the linear fit in the log-log plane. For each seed, an individual $p _ { i }$ is obtained by fitting its errors across all $N _ { p d e } . \mathrm { A }$ global $p _ { \mathrm { g l o b a l } }$ is similarly estimated from the geometric mean errors at each $N _ { p d e }$

## 5. Numerical experiments

## 5.1. Burgers equation: convective shock flow field

The Burgers equation serves as a simplified nonlinear form of the Navier-Stokes equations in fluid mechanics, commonly used to model convection-dominated flow and shock formation. The governing equation is:

$$
\frac { \partial u } { \partial t } + u \frac { \partial u } { \partial x } = \nu \frac { \partial ^ { 2 } u } { \partial x ^ { 2 } } , \quad x \in [ - 1 , 1 ] , t \in [ 0 , 1 ] ,\tag{32}
$$

with $\nu = 0 . 0 1 / \pi$ , initial condition $u ( x , 0 ) = - \sin ( \pi x )$ , and periodic boundary conditions $u ( - 1 , t ) = u ( 1 , t )$

$\mathbf { A s } \ t  1 . 0 $ , the nonlinear convection term drives rapid steepening of the solution field near $x = 0$ , evolving an extremely thin fluid shock front.

To demonstrate the dynamic adaptability of the AGM framework, Fig. 3 illustrates the mesh redistribution on the 1D Burgers equation across snapshots $t = 0 . 2 , 0 . 5$ , and 0.8. The left panels show the coordinate mapping $( \xi _ { \mathrm { ~ t o ~ } x } )$ with collocation nodes, while the right panels display the corresponding field solutions $u ( x , t )$ and high-gradient shock regions. Fig. 3 reveals the core regulatory features of AGM:

![](images/c044f37ef2212675d67f9467bcd89078585af36d8cf6e3f27a5215e87efe1c13.jpg)  
Fig. 3. Adaptive grid evolution and physical field solutions for the 1D Burgers’ equation across representative time snapshots $( t = 0 . 2 , 0 . 5 , 0 . 8 )$

• Shock-tracking and dynamic concentration: As shown in the left-hand panels, as time evolves from $t = 0 . 2$ to $t = 0 . 8$ and the shock front of the physical field on the right becomes increasingly steep, the mapped grid (�) and collocation nodes (red dots) exhibit a significant and monotonically intensifying S-shaped deformation in the central high-gradient region $( x \in \left[ - 0 . 1 5 , 0 . 1 5 \right] )$ . Meanwhile, the range of the vertical computational domain coordinate dynamically expands from approximately ±1.5 to around $\pm 2 .$ , intuitively reflecting the AGM framework’s continuously enhanced local compression and node-refinement capabilities in the high-gradient region over time.

• Precise field-grid correspondence: Comparing the physical field solution $u ( x , t )$ on the right with the coordinate mapping on the left indicates that the AGM module can perceive drastic local gradient changes in real-time, precisely allocating dense computational nodes in the core shock region where physical variations are most severe, thereby achieving eficient adaptive deployment of computational resources.

• Boundary and smooth domain preservation: In the low-gradient regions near the physical boundaries $( x =$ ±1), the grid transformation smoothly transitions and approaches a linear uniform distribution without causing unnecessary distortion, thereby ensuring the numerical stability of boundary condition enforcement.

• Optimization compatibility for stif problems: While maintaining topological homeomorphy, continuous diferentiability, and a constant total number of collocation points, this method avoids the loss-function discontinuities brought by traditional dynamic remeshing techniques (such as RAR), ensures the smoothness of the variational optimization landscape, and thus perfectly matches the high-precision convergence requirements of second-order optimizers (such as L-BFGS).

GAC-PINN demonstrates superior adaptive capture capability for the Burgers shock problem. The FFM, through nonlinear stretching via Gaussian random projection, enables the backbone network to establish strong representations of high-wavenumber shock fronts from early optimization stages. AGM captures strong spatial physical gradients at the shock front in real time, driving spontaneous local high-density aggregation of computational manifold nodes at $x = 0$ through the global synergy of the equidistribution loss $\mathcal { L } _ { e q u i }$ . RAR appends 2,500 high-residual points in the shock layer as a supplementary refinement step; its individual contribution is examined in the ablation study. The GAC-PINN predicted flow field aligns closely with the analytical solution, achieving a global relative $L ^ { 2 }$ error of $6 . 7 8 5 \times 1 0 ^ { - 5 }$ and efectively eliminating non-physical numerical oscillations.

Fig. 4 presents the absolute error distribution between the predicted and reference solutions across the full spatiotemporal domain.

![](images/533f5c78f27ed4d89be5da123edeb0aa5ce52186c5f4cc7c5b331272f3181802.jpg)

![](images/18559c9a71e3910a04c140eca83fd67d080a04ac2f26222f45e70639c10735c9.jpg)  
Fig. 4. Absolute error distribution of GAC-PINN for the Burgers equation.

From the error contours, three observations can be made: (1) the maximum error is strictly localized near the shock front trajectory around $x \approx 0$ and $t \to 1 . 0 .$ , corresponding to the infinite gradient discontinuity in the Burgers equation; (2) no non-physical oscillations are observed on either side of the shock, benefiting from FFM’s NTK spectrum reshaping and AGM’s geometric widening efect; (3) the error contours vary continuously in the spatio-temporal domain without abrupt changes or discontinuities, confirming that the three-stage training strategy efectively avoids the loss landscape discontinuity caused by discrete resampling.

## 5.2. 2D Poisson equation: extreme gradient potential reconstruction

The 2D Poisson equation is a canonical elliptic PDE model widely applicable to electrostatics, heat conduction, potential flow, and elasticity. The governing equation is:

$$
\begin{array} { r } { \nabla ^ { 2 } u ( x , y ) = f ( x , y ) , \quad ( x , y ) \in [ 0 , 1 ] ^ { 2 } , } \end{array}\tag{33}
$$

with homogeneous Dirichlet boundary conditions $u | _ { \partial \Omega } = 0$ . The method of manufactured solutions specifies the reference solution:

$$
u _ { t r u e } ( x , y ) = [ 1 6 x y ( 1 - x ) ( 1 - y ) ] ^ { a } , \quad a = 1 0 ,\tag{34}
$$

with the source term $f ( x , y )$ analytically derived from $f = - \nabla ^ { 2 } u _ { t r u e } .$

In the Poisson experiment, the operator-aware router determines $\chi _ { \mathcal { P } } = 0$ since the control operator $\mathcal { P } = \nabla ^ { 2 } u - f$ contains no temporal derivatives. The framework therefore selects the pure spatial distance function as the hardconstraint construction, dedicating all computational resources to the spatial adaptive reconstruction module. AGM, through minimization of the manifold volume dispersion $\mathcal { L } _ { e q u i }$ , drives nonlinear shear and compression of physical space coordinates. At the central high-gradient peak region, the Jacobian determinant det(�) increases substantially, indicating strong local densification of collocation points. The safety barrier $J _ { s a f e \_ m i n } = 0 . 0 2$ prevents det(�) from approaching zero, thereby guaranteeing that the mapping remains bijective throughout training. Coupled with FFM and adaptive hard constraints, GAC-PINN perfectly reconstructs the sharp 2D potential field with a global relative $L ^ { 2 }$ error of only $3 . 0 7 7 \times 1 0 ^ { - 5 }$

![](images/bac5b914dbb0d6a5fcd40bf4aceb42c13b29b69bdfb544c8424a5cbd66a53264.jpg)

![](images/9478ef1d98472b2799b1923c740ca6488680d929eba4ca8d45563d59cf93e05d.jpg)

![](images/33d603bf2457906b1464c70d0753ab136fb3494333cb69619f6d3200ce8d8ea4.jpg)  
Fig. 5. Three-dimensional surface plot of the GAC-PINN predicted solution $u ( x , y )$ for the 2D Poisson equation.  
Fig. 6. Comprehensive comparison for the 2D Poisson equation: (left) analytical true solution, (middle) GAC-PINN prediction, and (right) absolute error distribution field, achieving a global relative $L ^ { 2 }$ error of $3 . 0 7 7 \times 1 0 ^ { - 5 }$

To intuitively demonstrate the spatial modeling fidelity of the proposed framework, Fig. 5 presents the threedimensional surface plot of the predicted potential field $u ( x , y )$ . As illustrated, the network accurately captures the sharp, centralized bell-shaped profile induced by the high-order manufactured solution $( a = 1 0 )$ , while exhibiting exceptional smoothness and complete compliance with the homogeneous Dirichlet boundary conditions across all perimeter edges of the computational domain [0, 1]<sup>2</sup>.

To further quantify and validate the reconstruction performance, Fig. 6 provides a comprehensive three-way visual comparison comprising the analytical true solution, the GAC-PINN predicted solution, and the spatial absolute error distribution. The qualitative comparison between the true and predicted solutions confirms a near-perfect visua alignment. Furthermore, the absolute error distribution field reveals that the minor residuals are strictly localized around the steep central peak region characterized by high local curvature and gradients, whereas the extensive outer low-gradient domains maintain near-zero error levels. This outcome confirms the efectiveness of the AGM-driven adaptive collocation point aggregation strategy in tackling extreme gradient challenges.

## 5.3. Allen-Cahn equation: sharp phase-interface flow field

The Allen-Cahn equation is a canonical nonlinear parabolic PDE describing interface evolution and reactiondifusion processes in multi-phase materials science:

$$
\frac { \partial u } { \partial t } - \epsilon \frac { \partial ^ { 2 } u } { \partial x ^ { 2 } } + 5 ( u ^ { 3 } - u ) = 0 , \quad x \in [ - 1 , 1 ] , t \in [ 0 , 1 ] ,\tag{35}
$$

with phase interface difusion bandwidth coeficient $\epsilon = 0 . 0 0 1$ , initial condition $u ( x , 0 ) = x ^ { 2 } \cos ( \pi x )$ , and boundary conditions $u ( - 1 , t ) = u ( 1 , t ) = - 1$

For the Allen-Cahn problem, the operator-aware router determines $\chi _ { \mathcal { P } } = 1$ because the governing equation contains a temporal derivative. The framework therefore selects the time-adaptive bandwidth hard constraint, which gradually releases the initial condition as � increases. Meanwhile, AGM applies its nonlinear transformation only to the spatial axis, preserving temporal independence. The global relative $L ^ { 2 }$ error remains stable at $8 . 9 9 6 \times 1 0 ^ { - 3 }$ , outperforming existing baseline models. Fig. 7 presents the error distribution of GAC-PINN for the Allen-Cahn sharp phase-field evolution.

![](images/5605f5dda19c72723b6a9259942974f8813a13244de7903be31b60f7f14b0efb.jpg)

![](images/53e8bbfd009ed1a92db831daee7e804f13876be954dd4fcb35b76ab8d26ccf50.jpg)  
Fig. 7. Absolute error distribution of GAC-PINN for the Allen-Cahn equation.

The left panel of Fig. 7 depicts the dynamic evolution from the initial state to multi-phase separation. The right panel shows that the vast majority of the spatio-temporal domain exhibits extremely low absolute error (approaching $1 0 ^ { - 4 } ) ;$ minor local error elevations are concentrated at steep phase interface gradient regions and near $t = 1 . 0$ , consistent with the numerical challenges typically encountered in deep learning-based PDE solutions.

## 5.4. 2D Navier-Stokes equation: incompressible flow field

The 2D Navier-Stokes equation is a set of parabolic partial diferential equations describing incompressible fluid dynamics. As fundamental governing equations in fluid mechanics, they are widely adopted in scientific investigations and engineering applications to model the motions of fluid media such as water and air. The governing equations are as follows:

$$
\begin{array} { r } { \displaystyle \frac { \partial u } { \partial t } + \lambda _ { 1 } \left( u \frac { \partial u } { \partial x } + v \frac { \partial u } { \partial y } \right) = - \frac { \partial p } { \partial x } + \lambda _ { 2 } \left( \frac { \partial ^ { 2 } u } { \partial x ^ { 2 } } + \frac { \partial ^ { 2 } u } { \partial y ^ { 2 } } \right) , } \\ { \displaystyle \frac { \partial v } { \partial t } + \lambda _ { 1 } \left( u \frac { \partial v } { \partial x } + v \frac { \partial v } { \partial y } \right) = - \frac { \partial p } { \partial y } + \lambda _ { 2 } \left( \frac { \partial ^ { 2 } v } { \partial x ^ { 2 } } + \frac { \partial ^ { 2 } v } { \partial y ^ { 2 } } \right) , } \end{array}\tag{36}
$$

where $u ( t , x , y )$ and $v ( t , x , y )$ stand for velocity components in the � and � directions, and $p ( t , x , y )$ is fluid pressure. In this work, the coeficients are set to $\lambda _ { 1 } = 1$ and $\lambda _ { 2 } = 0 . 0 1$ . Table 1 summarizes the relative $L _ { 2 }$ errors of all compared models for the three output fields (u, v, and p).

As shown in Table 1, Vanilla PINN achieves errors of $7 . 9 5 \times 1 0 ^ { - 3 }$ for u and for $2 . 4 2 \times 1 0 ^ { - 2 }$ v in the velocity field, but its pressure field error reaches as high as 17.27, indicating that conventional PINNs struggle to accommodate the wide amplitude span of pressure gradients in the absence of explicit boundary constraints, leading to severe degradation in pressure reconstruction. DD-PINN reduces the pressure error to 0.28 through a domain-decomposition strategy, while also improving the velocity errors over the vanilla PINN; PINNsformer similarly achieves a pressure error of 0.28. The velocity errors of TSA-PINN are on the order of $1 0 ^ { - 2 }$ , while those of our GAC-PINN attain a comparable order of magnitude with slight improvements, thereby confirming its efective reconstruction of the velocity field. Meanwhile, GAC-PINN drastically reduces the pressure error to $2 . 6 6 \times 1 0 ^ { - 2 }$ , representing a nearly oned f i d d i f h 0.28 hi d b d f hi l f ll d its capability for simultaneous accurate reconstruction of both velocity and pressure fields in unsteady flows, as well as its favorable generalization performance.

Table 1: Relative $L _ { 2 }$ errors of diferent models on 2D Navier-Stokes equation.
<table><tr><td>Models</td><td>Relative L2 Error (u)</td><td>Relative L2 Error (v)</td><td>Relative L2 Error (p)</td></tr><tr><td>Vanilla PINN</td><td> $7 . 9 5 \times 1 0 ^ { - 3 }$ </td><td> $2 . 4 2 \times 1 0 ^ { - 2 }$ </td><td>17.27</td></tr><tr><td>PINNsformer</td><td></td><td></td><td>0.28</td></tr><tr><td>DD-PINN</td><td> $6 . 2 \times 1 0 ^ { - 3 }$ </td><td> $1 . 9 \times 1 0 ^ { - 2 }$ </td><td>0.28</td></tr><tr><td>TSA-PINN</td><td> $2 . 9 4 \times 1 0 ^ { - 2 }$ </td><td> $5 . 4 4 \times 1 0 ^ { - 2 }$ </td><td></td></tr><tr><td>GAC-PINN</td><td> $5 . 4 7 \times 1 0 ^ { - 3 }$ </td><td> $1 . 6 2 \times 1 0 ^ { - 2 }$ </td><td> $2 . 6 6 \times 1 0 ^ { - 2 }$ </td></tr></table>

This improvement is primarily attributed to two factors: the gradient-adaptive balancing strategy, which dynamically adjusts the weights of individual loss terms and efectively prevents the pressure gradient from being dominated by velocity residuals during back-propagation; and the physics-constrained attention mechanism, which enhances the network’s capacity to capture localized sharp features in the pressure field.

Figure 8 plots the exact pressure field $p ( x , y )$ , the GAC-PINN prediction, and the absolute error at $t = 1 0 . 0$ . It can be observed that the predicted pressure contour faithfully reproduces the main spatial patterns, extreme-value regions and sharp gradient structures of the reference solution. The absolute error map reveals that large errors mainly concentrate near steep pressure gradients, whereas the majority of the computational domain maintains a low error level.

![](images/24ef1a32fa3d6ccfb6e440dfb751a552beb413556309cc68ea09b6236bb76fd1.jpg)

![](images/3c53a23190f052bd10cb85c140786ff6085b4ca6e76730b3628c21248d409120.jpg)

![](images/d9b151a017a92139dac94e0d5ab4fcc560a4a5163e61d78a1c49a50b01af119c.jpg)  
Fig. 8. Pressure field reconstruction results for the 2D Navier-Stokes equation.

The contour comparison further verifies the quantitative error listed in Table 1. Given the relative $L _ { 2 }$ error of $2 . 6 6 \times 1 0 ^ { - 2 }$ for pressure, the contour plots and error distributions jointly demonstrate that GAC-PINN is capable of capturing complex localized flow structures and achieving high-fidelity pressure field reconstruction for incompressible Navier-Stokes problems.

## 6. Discussion

## 6.1. Prediction accuracy comparison

Table 1 presents the relative $L ^ { 2 }$ errors of all models across the three benchmark problems. GAC-PINN achieves optimal accuracy in all tests, with particularly pronounced advantages on steep gradient or discontinuous problems.

Table 2: Relative $L ^ { 2 }$ error comparison (mean ± standard deviation over five independent runs).

<table><tr><td>Models</td><td>Burgers</td><td>Poisson</td><td>Allen-Cahn</td></tr><tr><td>Vanilla PINN</td><td> $( 2 . 6 9 6 \pm 1 . 4 2 6 ) \times 1 0 ^ { - 2 }$ </td><td> $( 5 . 2 6 9 \pm 0 . 8 7 7 ) \times 1 0 ^ { - 4 }$ </td><td> $( 5 . 1 4 5 \pm 4 . 3 1 6 ) \times 1 0 ^ { - 2 }$ </td></tr><tr><td>RAR-PINN</td><td> $( 6 . 3 8 9 \pm 9 . 9 7 0 ) \times 1 0 ^ { - 3 }$ </td><td> $( 5 . 3 1 3 \pm 0 . 8 6 4 ) \times 1 0 ^ { - 4 }$ </td><td> $( 3 . 0 2 3 \pm 0 . 9 0 2 ) \times 1 0 ^ { - 2 }$ </td></tr><tr><td>GPINN</td><td> $( 8 . 5 1 2 \pm 1 . 6 8 0 ) \times 1 0 ^ { - 3 }$ </td><td> $( 1 . 0 6 5 \pm 0 . 5 8 8 ) \times 1 0 ^ { - 3 }$ </td><td> $( 1 . 9 5 1 \pm 0 . 3 0 8 ) \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Causal PINN</td><td> $( 1 . 3 0 5 \pm 0 . 3 1 5 ) \times 1 0 ^ { - 2 }$ </td><td> $( 2 . 9 1 4 \pm 0 . 9 1 5 ) \times 1 0 ^ { - 3 }$ </td><td> $( 2 . 5 2 7 \pm 1 . 0 6 4 ) \times 1 0 ^ { - 3 }$ </td></tr><tr><td>RAMS-PINN</td><td> $( 1 . 0 8 2 \pm 1 . 2 3 1 ) \times 1 0 ^ { - 2 }$ </td><td> $( 2 . 2 4 2 \pm 0 . 4 2 0 ) \times 1 0 ^ { - 3 }$ </td><td> $( 7 . 6 8 9 \pm 0 . 0 1 3 ) \times 1 0 ^ { - 1 }$ </td></tr><tr><td>GAC-PINN</td><td> $( 1 . 7 4 7 \pm 0 . 4 5 0 ) \times 1 0 ^ { - 4 }$ </td><td> $\mathbf { ( 2 . 8 6 8 \pm 0 . 9 4 7 ) \times 1 0 ^ { - 5 } }$ </td><td> $( \mathbf { 1 . 7 5 6 \pm 0 . 7 1 2 } ) \times \mathbf { 1 0 ^ { - 3 } }$ </td></tr></table>

<sup>†</sup> Values are reported as mean ± std over five random seeds (1234, 5678, 9012, 3456, 7890).

For the low-viscosity Burgers equation, GAC-PINN achieves a relative $L ^ { 2 } \operatorname { e r r o r } \operatorname { o f } ( 1 . 7 4 7 \pm 0 . 4 5 0 ) { \times } 1 0 ^ { - 4 }$ , more than one order of magnitude lower than the optimal baseline RAMS-PINN $( ( 1 . 0 8 2 \pm 1 . 2 3 1 ) \times 1 0 ^ { - 2 } )$ and with substantially smaller variance. For the 2D Poisson problem, GAC-PINN achieves $( 2 . 8 6 8 \pm 0 . 9 4 7 ) \times 1 0 ^ { - 5 }$ , more than an order of magnitude lower than the best baseline RAR-PINN $( ( 5 . 3 1 3 \pm 0 . 8 6 4 ) \times 1 0 ^ { - 4 } )$ . For the Allen-Cahn equation, GAC-PINN attains $( 1 . 7 5 6 { \pm } 0 . 7 1 2 ) { \times } 1 0 ^ { - 3 }$ , comparable to Causal PINN $( ( 2 . 5 2 7 \pm 1 . 0 6 4 ) \times 1 0 ^ { - 3 } )$ but with noticeably smaller standard deviation, indicating more stable convergence without manual tuning of temporal weights.

## 6.2. Computational eficiency analysis

Table 2 summarizes computational cost metrics across models for the 1D Burgers problem, and Fig. 9 presents the evolution of relative $L ^ { 2 }$ error with training iterations.

Table 3: Computational cost comparison.
<table><tr><td>Model</td><td>Parameters</td><td>Peak Memory (GB)</td><td>Total Time (s)</td><td>Epochs to Converge</td><td> $L ^ { 2 }$  Error</td></tr><tr><td>Vanilla PINN</td><td>921</td><td>0.163</td><td>1103</td><td>15000</td><td> $( 2 . 6 9 6 \pm 1 . 4 2 6 ) \times 1 0 ^ { - 2 }$ </td></tr><tr><td>RAR-PINN</td><td>2,241</td><td>0.228</td><td>4213</td><td>67203</td><td> $( 6 . 3 8 9 \pm 9 . 9 7 0 ) \times 1 0 ^ { - 3 }$ </td></tr><tr><td>GPINN</td><td>21,057</td><td>0.684</td><td>1457</td><td>38308</td><td> $( 8 . 5 1 2 \pm 1 . 6 8 0 ) \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Causal PINN</td><td>52,609</td><td>0.624</td><td>728</td><td>20000</td><td> $( 1 . 3 0 5 \pm 0 . 3 1 5 ) \times 1 0 ^ { - 2 }$ </td></tr><tr><td>RAMS-PINN</td><td>20,601</td><td>6.374</td><td>989</td><td>21000</td><td> $( 1 . 0 8 2 \pm 1 . 2 3 1 ) \times 1 0 ^ { - 2 }$ </td></tr><tr><td>GAC-PINN</td><td>104,708</td><td>3.776</td><td>1117</td><td>18514</td><td> $( 1 . 7 4 7 \pm 0 . 4 5 0 ) \times 1 0 ^ { - 4 }$ </td></tr></table>

![](images/14d420509e988a73ca648b5d35bd3d5720276fbcef7618443c000b5d582dcb9e.jpg)  
Fig. 9. Evolution of relative $L ^ { 2 }$ error with training iterations across all models.

GAC-PINN converges in 18,514 epochs to a relative $L ^ { 2 }$ error of $( 1 . 7 4 7 \pm 0 . 4 5 0 ) \times 1 0 ^ { - 4 }$ , reaching the target $1 0 ^ { - 4 }$ precision level within approximately 12,000 steps. This convergence speed is faster than all re-implemented baselines. In terms of wall-clock time, GAC-PINN requires 1,117 s, which is comparable to Vanilla PINN (1,103 s) and substantially lower than RAR-PINN and GPINN. Although GAC-PINN has the largest parameter count and peak memory footprint, the cost increase is accompanied by a reduction of more than one order of magnitude in the relative $\dot { L } ^ { 2 }$ error compared with the best baseline RAMS-PINN $( ( 1 . 0 8 2 \pm 1 . 2 3 1 ) \times 1 0 ^ { - 2 } )$ , yielding a favorable accuracy-eficiency trade-of.

## 6.3. Ablation studies

To quantitatively analyze the contribution of each core module to global convergence accuracy, seven ablation experiments were conducted on the 1D Burgers equation with a fixed random seed:

A. Fixed-HC baseline: A standard fully connected network with a fixed hard constraint.

B. Adaptive-HC baseline: Replaces the fixed hard constraint with an adaptive bandwidth hard constraint.

C. B + RAR: Adds residual-based adaptive refinement on top of B.

D. B + AGM: Adds adaptive grid mapping on top of B.

E. $\mathbf { D } + \mathbf { R } \mathbf { A } \mathbf { R } ;$ Adds RAR on top of D, combining AGM and RAR.

F. $\mathbf { D } + \mathbf { F F M } \mathbf { : }$ Adds Fourier feature mapping on top of D, combining AGM and FFM.

G. GAC-PINN: Full model with AGM, FFM, RAR, and adaptive hard constraints.

Table 4 summarizes the results.

Table 4: Ablation study results for the 1D Burgers equation.
<table><tr><td>Experiment</td><td>Components</td><td>Relative L² Error</td><td>Training Time</td><td>Parameters</td></tr><tr><td>A: Fixed-HC base PINN</td><td>MLP + fixed hard constraint</td><td> $6 . 8 9 0 \times 1 0 ^ { - 3 }$ </td><td>532.07 s</td><td>67,843</td></tr><tr><td>B: Adaptive-HC base PINN</td><td>A + adaptive hard constraint</td><td> $1 . 3 6 3 \times 1 0 ^ { - 3 }$ </td><td>487.26 s</td><td>67,843</td></tr><tr><td> ${ \bf C } \colon { \bf B } + { \bf R } { \bf A } { \bf R }$ </td><td>B + residual-based adaptive refinement</td><td> $3 . 9 0 5 \times 1 0 ^ { - 3 }$ </td><td>603.58 s</td><td>67,843</td></tr><tr><td> ${ \bf D } \colon { \bf B } + { \bf A } { \bf G } { \bf M }$ </td><td>B + adaptive grid mapping</td><td> $\mathbf { 3 . 4 1 0 \times 1 0 ^ { - 4 } }$ </td><td>819.55 s</td><td>72,196</td></tr><tr><td> ${ \mathrm { E } } { \mathrm { : D } } + { \mathrm { R A R } }$ </td><td> $\mathrm { B } + \mathrm { A G M } + \mathrm { R A R }$ </td><td> $1 . 7 2 5 \times 1 0 ^ { - 3 }$ </td><td>743.32 s</td><td>72,196</td></tr><tr><td> $\mathrm { F } \colon \mathrm { D } + \mathrm { F F M }$ </td><td>B + AGM + Fourier feature mapping</td><td> $1 . 5 7 2 \times 1 0 ^ { - 4 }$ </td><td>626.61 s</td><td>104,708</td></tr><tr><td>G: GAC-PINN (Full)</td><td> $\mathrm { F } + \mathrm { R A R }$ </td><td> $\mathbf { 6 . 7 8 5 \times 1 0 ^ { - 5 } }$ </td><td>1,173.30 s</td><td>104,708</td></tr></table>

All results are obtained with a fixed random seed (4321) under identical settings.

The ablation results reveal several key insights. Firstly, the adaptive bandwidth hard constraint reduces the relative $L ^ { 2 }$ error considerably compared with the fixed-HC baseline, confirming the value of spatially-varying boundary transition widths. Secondly, AGM alone yields a substantially lower error than RAR alone, establishing geometryadaptive point redistribution as a more efective strategy than discrete residual-based resampling. This advantage arises because AGM continuously deforms the computational manifold through a diferentiable mapping, whereas RAR performs discrete insertions that leave the underlying geometry unchanged. Thirdly, combining AGM and RAR (E) results in a higher error than AGM alone (D), revealing a conflict between the two mechanisms: AGM pursues a globally balanced distribution via the equidistribution principle, whereas RAR concentrates points in locally highresidual regions. Then adding FFM to AGM (F) further reduces the error, confirming that spectral preconditioning captures high-wavenumber features that geometric concentration alone cannot resolve. Notably, RAR is beneficial only in the presence of spectral preconditioning: it increases the error when added to AGM alone but decreases the error when added to $\mathrm { A G M + F F M } \ ( \mathrm { G } , 6 . 7 8 5 \times 1 0 ^ { - 5 } \ \mathrm { v s } . \ \mathrm { F } , \ 1 . 5 7 2 \times 1 0 ^ { - 4 } )$ . This suggests that FFM reshapes the NTK spectrum so that the network can exploit the additional high-residual points without disrupting the smooth geometric mapping learned by AGM. In GAC-PINN, RAR is therefore applied as a supplementary refinement step after AGM and FFM have been established, rather than integrated directly into the geometric adaptation pipeline. Overall, the full GAC-PINN (G) achieves the lowest relative $L ^ { 2 }$ error among all configurations, with a value of $6 . 7 8 5 \times 1 0 ^ { - 5 }$

## 6.4. Grid Convergence Analysis

We further performed a grid-convergence study for the one-dimensional Burgers equation, considering spatial resolutions of $N _ { \mathrm { p d e } } \in \{ 1 0 0 0 , 3 0 0 0 , 5 0 0 0$ , 7000, 9000, 12000} and averaging over 10 independent random seeds for each case. The geometric mean of the $L ^ { 2 }$ error at each resolution is summarized in Table 4.

Table 5: Grid-convergence statistics for the 1D Burgers equation.
<table><tr><td> $N _ { p d e }$ </td><td>Geometric mean  $L ^ { 2 } \thinspace { \mathrm { e r r o r } }$ </td><td> $\mathrm { S t d . ~ d e v . ~ ( l o g _ { 1 0 } \ s p a c e ) }$ </td><td>Min / Max error</td></tr><tr><td>1,000</td><td> $3 . 4 9 \times 1 0 ^ { - 3 }$ </td><td>0.85</td><td> $4 . 8 0 \times 1 0 ^ { - 4 } / 1 . 0 1 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>3,000</td><td> $3 . 6 0 \times 1 0 ^ { - 4 }$ </td><td>0.42</td><td> $1 . 4 1 \times 1 0 ^ { - 4 } / 1 . 1 6 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>5,000</td><td> $2 . 1 9 \times 1 0 ^ { - 4 }$ </td><td>0.38</td><td> $5 . 9 3 \times 1 0 ^ { - 5 } / 6 . 3 7 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>7,000</td><td> $1 . 7 2 \times 1 0 ^ { - 4 }$ </td><td>0.29</td><td> $8 . 7 0 \times 1 0 ^ { - 5 } / 4 . 5 1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>9,000</td><td> $1 . 9 9 \times 1 0 ^ { - 4 }$ </td><td>0.35</td><td> $8 . 2 1 \times 1 0 ^ { - 5 } / 3 . 5 7 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>12,000</td><td> $1 . 3 8 \times 1 0 ^ { - 4 }$ </td><td>0.31</td><td> $\ 7 . 0 8 \times 1 0 ^ { - 5 } \ / \ 2 . 8 9 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>16,000</td><td> $1 . 2 2 \times 1 0 ^ { - 4 }$ </td><td>0.33</td><td> $6 . 5 4 \times 1 0 ^ { - 5 } / 2 . 8 8 \times 1 0 ^ { - 4 }$ </td></tr></table>

Global fitted convergence order: $p _ { \mathrm { g l o b a l } } = 1 . 1 8 ( R ^ { 2 } = 0 . 7 1 7 )$  
Mean of individual seed orders: $\bar { p } = 1 . 1 7 7 \pm 0 . 5 8 1$

Two features are noteworthy: (i) The error steadily decreases with $N _ { p d e }$ , from $3 . 4 9 \times 1 0 ^ { - 3 }$ to $1 . 2 2 \times 1 0 ^ { - 4 }$ , by over an order of magnitude. (ii) The improvement is most rapid for $N _ { p d e } \leq 9 \AA , 0 0 0$ ; subsequent increments produce only modest reductions $( 1 . 9 9 \times 1 0 ^ { - 4 } \mathrm { ~ t o ~ } 1 . 2 2 \times 1 0 ^ { - 4 } )$ . This indicates that the AGM has already densely sampled the high-gradient shock region, so that the local efective resolution saturates near the network’s expressive limit; extra points fall in smooth areas and have marginal impact.This saturation provides direct evidence of AGM’s efectiveness: by difeomorphically compressing collocation points toward steep gradients, AGM decouples the global error from the total point count, enabling high accuracy with far fewer samples than uniform refinement would require.

The global convergence order $p _ { \mathrm { g l o b a l } } = 1 . 1 8$ is lower than the theoretical value $p = 2 . 0$ for second-order finitediference schemes. This gap reflects the fact that, at this resolution range, the total error is governed not only by collocation point density but also by the network’s representation capacity and the non-convexity of the optimization landscape. The saturation observed for $N _ { p d e } \ge 9 . 0 0 0$ suggests that the expressive power of the backbone network, rather than the point count, becomes the limiting factor. The practical significance of AGM is therefore not that it improves the asymptotic convergence rate, but that it achieves an $L ^ { 2 }$ error of $\approx 1 0 ^ { - 4 }$ with only 9,000 collocation points, an eficiency that uniform sampling would require substantially more points to match.

## 7. Conclusion

This paper has systematically developed and validated GAC-PINN, a geometry-adaptive and constraint-enhanced physics-informed neural network framework for problems with steep gradients and sharp interfaces. By integrating these core modules, the framework efectively alleviates the spectral bias, geometric inflexibility, and boundary constraint conflicts that limit standard PINNs. Tested on four representative benchmarks—including the viscous Burgers equation, the sharp 2D Poisson problem, the Allen-Cahn phase-field equation, and the 2D Navier-Stokes equations for unsteady cylinder flow—GAC-PINN achieves relative $L ^ { 2 }$ errors of $( 1 . 7 4 7 \pm 0 . 4 5 0 ) \times 1 0 ^ { - 4 } , ( 2 . 8 6 8 \pm 0 . 9 4 7 ) \times 1 0 ^ { - 5 }$ and $( 1 . 7 5 6 \pm 0 . 7 1 2 ) \times 1 0 ^ { - 3 }$ on the Burgers, Poisson, and Allen–Cahn benchmarks, respectively, and $2 . 6 6 \times 1 0 ^ { - 2 }$ for the pressure field on the 2D Navier–Stokes problem, respectively, achieving competitive or improved accuracy compared to re-implemented baselines under aligned settings, with particularly pronounced gains on problems exhibiting localized steep gradients. The successful extension to the Navier-Stokes system, with its fundamentally diferent physical characteristics from the previous benchmarks, provides strong evidence of the framework’s generalization capability beyond canonical PDE problems. Ablation and computational analyses validate the synergistic contributions of each module and demonstrate competitive training eficiency despite increased parameter counts. This work provides a practical adaptive framework for high-fidelity simulation of problems with localized sharp features in applied mathematics and computational mechanics.

## 7.1. Conclusion of core mechanisms

The modules of GAC-PINN are designed to complement each other within a unified training pipeline. Each module plays a specific regulatory role in the computational manifold and optimization landscape:

• Dual dimensionality reduction through geometric adaptation and spectral reshaping. For the convective shock in the low-viscosity Burgers equation and the extreme spatial peak in the 2D Poisson equation, conventional PINNs sufer from spectral bias due to high-wavenumber component attenuation in the input space. AGM spontaneously compresses physical collocation points toward high-gradient regions through difeomorphic coordinate transformation and Jacobian regularization, spatially "stretching" geometric discontinuities on the computational manifold. FFM, through Gaussian random projection matrices, further stretches the spectral response of the feature space. The deep coupling of these two mechanisms achieves dual dimensionality reduction from physical space to feature space, transforming originally localized infinite high-wavenumber features into macroscopic smooth signals amenable to neural network capture, fundamentally suppressing Gibbs oscillation generation.

• Stable high-order diferentiation through adaptive hard constraints. For problems with non-periodic boundaries, the adaptive hard-constraint ansatz predicts a spatially-varying bandwidth $\kappa _ { a d a p t i v e } ( X )$ from the input coordinates. By treating the bandwidth as a detached constant in the high-order automatic diferentiation graph, the ansatz stabilizes the computation of higher-order PDE derivatives while maintaining exact satisfaction of boundary and initial conditions.

## 7.2. Limitations

Despite GAC-PINN’s strong performance on multiple benchmark problems, several limitations merit discussion for future work:

• Extension to 3D and irregular geometries. The current AGM is demonstrated on 1D and 2D regular domains. Extending the difeomorphic mapping and Jacobian barrier to 3D complex geometries (e.g., turbine blades, irregular patient-specific domains) remains an open challenge. The computational cost of evaluating det(�) and enforcing the anti-folding barrier in 3D, as well as the risk of local folding near sharp concave boundaries, require further investigation.

• Automated hyperparameter tuning for ultra-large-scale multiphysics coupling. Some hyperparameters in the framework (e.g., Jacobian scaling coeficients, gradient amplification factors, loss term weights) still rely on problem-specific prior tuning. Future work should incorporate automated optimization strategies to enhance algorithmic generality.

## Data availability

The benchmark datasets for the Burgers, Poisson and Allen–Cahn equations are taken from the open gPINN repository [12]. The 2D cylinder wake reference data can be obtained from the original publication of Raissi et al. [1]. No new data were generated in this study. The source code of the proposed GAC-PINN model is publicly available at https://github.com/zyx7765-sudo/GAC-PINN.

## Declaration of competing interest

The author declares that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## CRediT authorship contribution statement

Yanxin Zhang: Conceptualization, Investigation, Formal analysis, Writing - original draft.   
Yong Zhang: Supervision, Project administration, Writing - review & editing.   
Houbiao Li: Supervision, Project administration, Writing - review & editing.

## References

[1] M. Raissi, P. Perdikaris, and G.E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. J. Comput. Phys., 378:686–707, 2019. doi:10.1016/j.jcp.2018.10.045.

[2] M. Tancik, P.P. Srinivasan, B. Mildenhall, et al. Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. In Adv. Neural Inf. Process. Syst. (NeurIPS), volume 33, pages 7537–7547, 2020. URL https://proceedings.neurips.cc/paper/ 2020/hash/2fcd4e9b8a1f3b6a7a8c9d0e1f2a3b4c-Abstract.html.

[3] C.X. Wu, M. Zhu, Q.Y. Tan, et al. A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neura networks. Comput. Methods Appl. Mech. Eng., 403:115671, 2023. doi:10.1016/j.cma.2022.115671.

[4] H. Gao, L.N. Sun, and J.X. Wang. PhyGeoNet: Physics-Informed Geometry-Adaptive Convolutional Neural Networks for Solving Parametric PDEs on Irregular Domain. J. Comput. Phys., 428:110079, 2021. doi:10.1016/j.jcp.2020.110079

[5] A. Jacot, F. Gabriel, and C. Hongler. Neural Tangent Kernel: Convergence and Generalization in Neural Networks. In Advances in Neural Information Processing Systems (NeurIPS), volume 31, pages 8571–8580, 2018. URL https://proceedings.neurips.cc/paper/2018/ hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html.

[6] S. Wang, X. Yu, and P. Perdikaris. When and why PINNs fail to train: A neural tangent kernel perspective. J. Comput. Phys., 449:110768, 2022. doi:10.1016/j.jcp.2021.110768.

[7] C. de Boor. Good approximation by splines with variable knots. In Spline Functions and Approximation Theory, volume 21 of ISNM, pages 57–72, 1973. doi:10.1007/978-3-0348-5979-0\_3.

[8] C.J. Budd, W. Huang, and R.D. Russell. Adaptivity with moving grids. Acta Numerica, 18:111–241, 2009. doi:10.1017/s0962492906400015.

[9] C. Straub, P. Brendel, V. Medvedev, and A. Rosskopf. Hard-constraining Neumann boundary conditions in physics-informed neural networks via Fourier feature embeddings, 2025. URL https://arxiv.org/abs/2504.01093.

[10] L. Lu, R. Pestourie, W. Yao, Z. Wang, F. Verdugo, and S.G. Johnson. Physics-informed neural networks with hard constraints for inverse design. SIAM J. Sci. Comput., 43:B1105–B1132, 2021. doi:10.1137/21M1397908.

[11] Z.P. Mao and X.H. Meng. Physics-informed neural networks with residual/gradient-based adaptive sampling methods for solving partial diferential equations with sharp solutions. Appl. Math. Mech., 44(7):1069–1084, 2023. doi:10.1007/s10483-023-2994-7.

[12] J. Yu, L. Lu, X.H. Meng, and G.E. Karniadakis. Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems. Comput. Methods Appl. Mech. Eng., 393:114823, 2022. doi:10.1016/j.cma.2022.114823.

[13] Z. Zhao, X. Ding, and B.A. Prakash. PINNsFormer: A transformer-based framework for physics-informed neural networks. In The Twelfth International Conference on Learning Representations (ICLR), 2024. URL https://openreview.net/forum?id=DO2WFXU1Be.

[14] A. Khademi and S. Dufour. A novel discretized physics-informed neural network model applied to the Navier-Stokes equations. Phys. Scr., 99:076016, 2024. doi:10.1088/1402-4896/ad50f6.

[15] A. Khademi and S. Dufour. Physics-informed neural networks with trainable sinusoidal activation functions for approximating the solutions of the Navier-Stokes equations. Comput. Phys. Commun., 314:109672, 2025. doi:10.1016/j.cpc.2025.109672.

[16] S.F. Wang, S. Sankaran, and P. Perdikaris. Respecting causality is all you need for training physics-informed neural networks. Comput. Methods Appl. Mech. Eng., 421:116813, 2024. doi:10.1016/j.cma.2024.116813.

[17] J. Wei, F. Wu, and X. Zhang. SAGE: A Lightweight Framework for Trigger-Guided LoRA-Based Self-Adaptation in LLMs, 2025. URL https://arxiv.org/abs/2509.05385.

[18] L. Lu, X.H. Meng, Z.P. Mao, and G.E. Karniadakis. DeepXDE: A deep learning library for solving diferential equations. SIAM Rev., 63(1): 208–228, 2021. doi:10.1137/19M1274067.

[19] G.E. Karniadakis, I.G. Kevrekidis, L. Lu, et al. Physics-informed machine learning. Nat. Rev. Phys., 3:422–440, 2021. doi:10.1038/s42254- 021-00314-5.

[20] J. Hou, Y. Li, and S.H. Ying. Enhancing PINNs for solving PDEs via adaptive collocation point movement and adaptive loss weighting. Nonlinear Dyn., 111(16):15233–15261, 2023. doi:10.1007/s11071-023-08766-9.

[21] Z.K. Hao, J.C. Yao, C. Su, H. Su, Z.A. Wang, F.Z. Lu, Z.Y. Xia, Y.C. Zhang, S.M. Liu, L. Lu, and J. Zhu. PINNacle: A comprehensive benchmark of physics-informed neural networks for solving pdes. In Adv. Neural Inf. Process. Syst. (NeurIPS), 2024. doi:10.52202/079017- 2442.

[22] Y.C. Xie, H.H. Chi, H.P. Quan, et al. Spectral analysis of hard-constraint PINNs: The spatial modulation mechanism of boundary functions, 2025. URL https://arxiv.org/abs/2512.23295.

[23] X.M. Lian and L. Chen. Gaussian causal physics-informed neural networks. In Proc. 2025 3rd Int. Conf. Math. Mach. Learn., 2025. doi:10.1145/3783779.3783812.

[24] Y.H. Chen, Z. Gao, J.S. Hesthaven, Y.F. Lin, and X. Sun. A coordinate transformation-based physics-informed neural networks for hyperbolic conservation laws. J. Comput. Phys., 538:114161, 2025. doi:10.1016/j.jcp.2025.114161.

[25] R. Zare Moayedi, M. Abbaszadeh, and M. Dehghan. CuPINN: Optimizing pinns through curvature minimization and residual landscape flattening. Comput. Methods Appl. Mech. Eng., 445:118180, 2025. doi:10.1016/j.cma.2025.118180.

[26] Z.Y. Zhang, C.L. Ruan, and Z.J. Liu. Two improved physics-informed Neural Networks for solving Burgers equation. J. Comput. Sci., 93: 102756, 2025. doi:10.1016/j.jocs.2025.102756.

[27] W.H. Ouyang, M. Zhu, W. Xiong, S.W. Liu, and L. Lu. RAMS: residual-based adversarial-gradient moving sample method for scientific machine learning in solving partial diferential equations. Adv. Intell. Discovery, 2(3):e202500214, 2026. doi:10.1002/aidi.202500214.

[28] S. Khanra, V.K. Kukreja, and I. Bala. Physics-informed neural networks for diferential equation solutions: A comprehensive review. Neurocomputing, 680:133317, 2026. doi:10.1016/j.neucom.2026.133317.

[29] A. Iftakher, R. Golder, B.N. Roy, and M.M.F. Hasan. Physics-informed neural networks with hard nonlinear equality and inequality constraints. Comput. Chem. Eng., 204:109418, 2025. doi:10.1016/j.compchemeng.2025.109418.