# Physics-Informed Neural Networks to Infer the Perpendicular Energy Conductivity in the Scrape-Off Layer of Stellarator Devices

J. Gallego<sup>1</sup>, P. Protopapas<sup>2</sup>, A. Bustos<sup>1</sup>, A. Alonso<sup>3</sup>, S. Barquero<sup>3</sup>, A. Baciero<sup>3</sup>, I. Rivera<sup>3</sup>, J. A. Moríñigo<sup>1</sup>, R. Mayo-García<sup>1</sup>, and the TJ-II team.

<sup>1</sup>Departamento de Tecnología, CIEMAT, Spain. <sup>2</sup>Harvard John A. Paulson School of Engineering and Applied Sciences, USA. <sup>3</sup>Laboratorio Nacional de Fusión, CIEMAT, Spain.

## Abstract

In this work, we develop an inverse Physics-Informed Neural Network (PINN) framework to infer the dependence of the scrape-off layer (SOL) perpendicular heat conductivity on plasma density and temperature, κ (n, T). The method combines radial profile measurements of electron density and temperature with the residual of a reduced one-dimensional SOL transport equation, so that the inferred conductivity is constrained by both the measurements and the underlying transport model. Three neural networks are trained simultaneously: two reconstruct the temperature and density profiles as functions of the radial coordinate and transported power, while a third represents the effective conductivity as a function of the local density and temperature. The framework is first validated using synthetic data generated from a prescribed conductivity function, allowing the inferred (n, T) to be compared directly with the ground truth. The model recovers the imposed functional dependence with errors below 10 % in the data-constrained region. Bootstrap resampling is shown to provide a practical indicator of prediction reliability and consistency. A scan in the number of plasma profiles used for training and the number of radial measurement positions per profile identifies a practical trade-off between reconstruction accuracy and data availability. Finally, the method is applied to an experimental dataset from the TJ-II stellarator obtained with the helium-beam diagnostic. This exploratory application provides an initial estimate of the effective SOL conductivity and illustrates the potential of inverse PINNs for extracting transport information from plasma edge measurements.

## 1 Introduction

Neural networks are powerful tools for processing and generalizing large amounts of data, with successful applications in fields such as computer vision [1], healthcare [2], and civil engineering [3]. However, purely data-driven approaches rely entirely on the available data to contain the information needed to describe the system. As a result, their performance can deteriorate when the data are scarce, sparse, or not sufficiently representative of the underlying behaviour [4].

Physics-Informed Neural Networks (PINNs) [5] address this limitation by constraining the neural-network solution with the physical laws governing the system. In the PINN framework, these laws are embedded in the loss function as additional terms, so that the network is trained to satisfy both the available data and the prescribed physics. Since the original PINN formulation, the field has grown rapidly [6, 7], reflecting the increasing interest of the scientific community in this approach.

When the governing physics is expressed as a differential equation, or as a system of differential equations, a PINN can be interpreted as a neuralnetwork-based differential equation solver. In this setting, the network approximates the solution, while the initial and boundary conditions are imposed through the training dataset. PINNs have been applied as differential equation solvers in a wide range of problems, including fluid dynamics [8], heat transfer in multiphase flows [9], wave physics [10] and magnetohydrodynamic equations [11, 12].

Beyond forward problems, PINNs can also be formulated to solve inverse Partial Differential Equations (PDE) problems. In this case, the objective is to infer unknown quantities in the governing equation from sparse measurements of the solution, as also discussed in [5]. In the simplest formulation, the unknown parameter is treated as an additional trainable variable and optimized so that the predicted solution remains consistent with both the measurements and the physical model. This strategy has been used, for example, to infer unknown parameters in groundwater flow [13] and biomechanical systems [14]. A further extension consists of representing the unknown coefficient itself as the output of a neural network, allowing it to depend on selected independent variables. The inverse problem then becomes the identification of a functional relation rather than a scalar parameter, as studied in [15].

Since the original formulation of PINNs, numerous methodological developments have been introduced to improve their capabilities and extend their range of applications. Flamant et al. [16] introduced the concept of solution bundles, in which relevant parameters of the governing differential equations are included as inputs to the neural network, enabling a single model to represent a family of solutions rather than a single realization. This approach has subsequently been applied in several cosmological studies [17, 18, 19]. Further developments have addressed the difficulties associated with stiff differential equations. Tarancón-Álvarez et al. [20] proposed multi-head training combined with unimodular regularization to improve the treatment of stiff systems and to generalize the network over a latent space of solutions, while Yepes et al. [21] investigated the sensitivity of PINN performance to the choice of initial-condition embeddings in stiff problems. Advances in alternative training techniques have also been proposed; for example, Protopapas et al. [22] introduced variational boosting to enable second-order optimization strategies. In parallel, increasing attention has been devoted to the theoretical characterization of PINN accuracy. Recent work has derived error bounds for specific classes of differential equations [23], providing a more rigorous framework for quantifying the reliability of PINN solutions and improving their physical interpretability.

Building on these foundations, this work explores the application of a PINN framework to the modelling and interpretation of the physics parameters observed in the periphery of magnetically confined fusion plasmas. Nuclear fusion is a promising energy source, owing to its abundant fuel and lack of green house gas emissions. However, the practical production of fusion energy remains to be demonstrated. One of the central challenges is ensuring plasma heat and particle exhaust in the plasma-facing components that is compatible with stable, high-performance operation while maintaining the resulting heat loads within the thermal limits of the materials. For a given amount of transported power, this heat flux depends on the width $\lambda _ { q }$ of the Scrape-Off Layer (SOL) transport channel, which is set by the balance between perpendicular and parallel transport. Here, perpendicular and parallel refer to the direction relative to the magnetic field. Higher perpendicular transport leads to a wider transport channel and therefore a lower peak heat load on the target. Determining the perpendicular and parallel heat conductivities, together with their parametric dependencies, is therefore essential for accurately predicting the heat loads the target must withstand and its wet area.

The parallel heat conductivity $\kappa _ { \parallel }$ is well described by the Spitzer–Härm conductivity, which can be derived from classical collisional transport theory [24]. In contrast, the perpendicular conductivity $\kappa _ { \perp }$ is largely governed by turbulent transport and is therefore difficult to derive from first principles. One should also consider that the two main types of magnetic confinement fusion devices, tokamaks and stellarators, differ in their confinement and stability properties, and therefore perpendicular SOL transport cannot necessarily be assumed to follow the same behaviour in both designs. Previous studies have examined different aspects of perpendicular SOL transport mainly in tokamak devices. For example, Carralero et al. [25] analysed the role of filaments in perpendicular SOL transport, while Itoh et al. [26] studied the consistency and implications of applying $\kappa _ { \perp }$ scaling laws developed for the confined plasma region to the SOL. Eich et al. [27] found a multi-machine scaling law to directly predict $\lambda _ { q }$ in tokamaks. Nevertheless, robust scalings for $\kappa _ { \perp }$ and, consequently, for $\lambda _ { q }$ in the SOL of stellarator devices remain open problems.

The objective of this work is therefore to advance the study of $\kappa _ { \perp }$ in stellarator devices. Namely, we present a new methodology to infer the functional dependence of the perpendicular heat conductivity on plasma density n and temperature $T$ in the $\mathrm { S O L } , \kappa _ { \perp } ( n , T )$ , using sparse measurements of the plasma profiles. To this end, we use an inverse PINN framework based on the minimization of the residual of a reduced 1D transport model, which we implement in Python using NeuroDiffEq [28, 29].

The paper is organized as follows. Section 2 introduces the physical and computational background of the work. First, the heat transport physics of the SOL and existing scalings for $\kappa _ { \perp }$ are reviewed. The section then defines formally the inverse PINN formulation for parameter-identification problems. Section 3 presents the PINN model, including the specific PINN structure adopted to solve the problem and the optimization process. Section 4 validates the framework using synthetic data generated from numerical solutions of the governing equation. The validation assesses the robustness of the method through scans in the number of plasma profiles and measurement positions. Section 5 applies the methodology to a small experimental dataset from the TJ-II stellarator, describing the experimental setup, the measured data, and the inferred transport results. Finally, section 6 summarizes the main conclusions and outlines directions for future work.

## 2 Background

This section provides an overview of the physical and computational background of the work. First, the heat-transport physics of the SOL are introduced, including the reduced transport model used in this study and the main $\kappa _ { \perp }$ scalings reported in the literature. The inverse PINN formulation is then presented, beginning with the inference of a constant coefficient and subsequently extending the formulation to the inference of a functional dependence.

## 2.1 Heat transport physics in the SOL

## Heat transport model

Within the two-fluid framework, the SOL energy balance for ions and electrons can be written as [30]

$$
\nabla \cdot \left( \frac 5 2 n u _ { \parallel } T _ { \mathrm { i } } \mathbf { b } - \kappa _ { \parallel \mathrm { i } } \nabla _ { \parallel } T _ { \mathrm { i } } - \kappa _ { \perp \mathrm { i } } \nabla _ { \perp } T _ { \mathrm { i } } \right) = S _ { \mathrm { e i } } ,\tag{1}
$$

$$
\nabla \cdot \left( \frac 5 2 n u _ { \parallel } T _ { \mathrm { e } } \mathbf { b } - \kappa _ { \parallel \mathrm { e } } \nabla _ { \parallel } T _ { \mathrm { e } } - \kappa _ { \perp \mathrm { e } } \nabla _ { \perp } T _ { \mathrm { e } } \right) = - S _ { \mathrm { e i } } .\tag{2}
$$

Here, $u _ { \parallel }$ is the parallel flow velocity, b is the unit vector along the magnetic field, $T _ { \mathrm { e } }$ and $T _ { \mathrm { i } }$ are the electron and ion temperatures, and $S _ { \mathrm { e i } }$ is the electron– ion energy exchange term. Note that equations 1 and 2 neglect radiation and charge-exchange losses. This approximation is usually acceptable for attached plasmas in small and medium-sized devices, but it should be reconsidered for detached plasmas or larger devices, where SOL radiation becomes significant. In the SOL, the electron and ion temperatures are often comparable because of the high collisionality and the resulting thermal coupling [31]. We therefore assume $T _ { \mathrm { i } } = T _ { \mathrm { e } } = T$ . Since the parallel ion heat conductivity is much smaller than the electron one, $\kappa _ { \parallel \mathrm { i } } \ll \kappa _ { \parallel \epsilon }$ [32], we set $\kappa _ { \parallel \mathrm { i } } \approx 0 .$ . Defining the total perpendicular conductivity as $\kappa _ { \perp } = \kappa _ { \perp \mathrm { i } } + \kappa _ { \perp \mathrm { e } } ,$ the sum of equations 1 and 2 gives

$$
\nabla \cdot \Big ( 5 n \boldsymbol { u } _ { \| } T \mathbf { b } - \kappa _ { \| \mathbf { e } } \nabla _ { \| } T - \kappa _ { \perp } \nabla _ { \perp } T \Big ) = 0 .\tag{3}
$$

The relative importance of the terms in equation 3 depends on the SOL transport regime. In the sheath-limited regime, the plasma is nearly isothermal along the magnetic field lines, so that $\nabla _ { \parallel } T \approx 0 \ [ 3 3 ]$ . The parallel conductive term can then be neglected, and the parallel exhaust is mainly governed by convective transport to the sheath. On the other hand, in the conduction-limited regime, the temperature shows a maximum upstream and decreases towards the target. Parallel heat conduction then plays an important role, with $\kappa _ { \parallel \mathrm { e } } \propto T ^ { 5 / 2 }$ given by the Spitzer–Härm conductivity [24], and neither the parallel conduction nor the convective term can in general be neglected. TJ-II is known to be predominantly sheath-limited, as is typically the case for limiter SOLs in medium-sized devices [33]. Therefore, we can adopt the approximation $\kappa _ { \parallel e } \nabla _ { \parallel } T \approx 0$ in the present study.

Strictly speaking, the divergence and gradient operators in equation 3 should be evaluated in toroidal geometry. However, the SOL width is much smaller than the plasma minor radius, $\lambda _ { q } \ll a .$ Therefore, the SOL can be straightened out [33, p. 20], so that the perpendicular and parallel directions define a 2D Cartesian basis, as illustrated in figure 1. Writing equation 3 in this new basis for the sheath-limited case gives

$$
5 T \frac { \partial \left( n u _ { y } \right) } { \partial y } - \frac { \partial } { \partial x } \left( \kappa _ { \perp } \frac { \partial T } { \partial x } \right) = 0 ,\tag{4}
$$

where x denotes now the perpendicular direction and y the parallel direction.

The first term in equation 4 represents the divergence of the parallel convective heat flux. In a reduced 1D description along the perpendicular direction, this term can be interpreted as a local sink of thermal energy. Physically, the energy transported from the confined plasma into the SOL is progressively redirected along the magnetic field lines and exhausted at the target, so that the perpendicular heat flux decreases along the radial direction until it vanishes, as illustrated in figure 1.

To express this parallel loss in local form, we approximate the parallel particle flux term, $n u _ { y } ,$ by using characteristic upstream and target quantities. At the upstream position, the net parallel particle flux is zero, since particles are lost symmetrically toward the two target ends. At the sheath entrance, the downstream density is approximately $n _ { \mathrm { d } } \approx n _ { \mathrm { u p } } / 2$ and the parallel velocity $u _ { y } \approx c _ { \mathrm { i } } \ [ 3 3 ]$ , where

$$
c _ { \mathrm { i } } = { \sqrt { \frac { T } { m _ { \mathrm { i } } } } }\tag{5}
$$

is the ion sound speed, and $m _ { \mathrm { i } }$ is the ion mass. Assuming that the parallel particle flux varies linearly

![](images/f8db73f450935442a45a5fc0cda03e736c1067e563ff2cf20dd314336e5742ab.jpg)  
Figure 1: Schematic representation of the straightened SOL geometry used to derive the reduced transport model. The SOL is bounded by the two target plates separated by two times the connection length $L _ { \mathrm { c } } .$ The coordinates x and y denote the perpendicular and parallel directions, respectively, with the origin of coordinates at the upstream position of the Last-Closed Flux Surface. Heat flux q enters the SOL through the x direction, to be then redirected along the magnetic field lines and exhausted toward the targets.

along y, its derivative can be estimated as

$$
\frac { \partial \left( n u _ { y } \right) } { \partial y } \approx \frac { n _ { \mathrm { d } } c _ { \mathrm { i } } - 0 } { L _ { \mathrm { c } } } \approx \frac { n _ { \mathrm { u p } } c _ { \mathrm { i } } } { 2 L _ { \mathrm { c } } } ,\tag{6}
$$

where $L _ { \mathrm { c } }$ is the upstream-to-target connection length. Substituting equation 6 into equation 4 gives the reduced 1D perpendicular transport equation

$$
\frac { \partial } { \partial x } \left( \kappa _ { \perp } \frac { \partial T } { \partial x } \right) = \frac { 5 n _ { \mathrm { u p } } T ^ { 3 / 2 } } { 2 \sqrt { m _ { \mathrm { i } } } L _ { \mathrm { c } } } ,\tag{7}
$$

that we take to model the SOL region of seathlimited devices. It should be noted that equation 7 is a strongly simplified description of the heat transport in the SOL, possibly relevant for the conditions found in small devices such as TJ-II. However, the method presented in this work can be adapted to more sophisticated descriptions.

It is also necessary to specify the boundary condition at the interface between the confined and unconfined plasma, namely the Last-Closed Flux Surface (LCFS), located at $x = 0$ in the present coordinate system. At this boundary, the transported power $P _ { \mathrm { { t r } } }$ leaving the confined plasma is assumed to enter the SOL uniformly over the total LCFS area, $S _ { \mathrm { L C F S } }$ . This gives the following relation between the perpendicular heat flux, the perpendicular conductivity, the temperature gradient, and the transported power:

$$
q _ { x } ^ { \mathrm { L C F S } } = - \left. \kappa _ { \perp } \frac { \partial T } { \partial x } \right| _ { x _ { \mathrm { L C F S } } } = \frac { P _ { \mathrm { t r } } } { S _ { \mathrm { L C F S } } } .\tag{8}
$$

## Existing $\kappa _ { \perp }$ scalings

Here, we summarize the main scaling laws commonly used to describe perpendicular transport coefficients in the confined region of fusion plasmas.

Although these scalings were not originally derived for SOL transport, they provide useful reference trends for interpreting the results obtained in section 5, allowing us to assess whether the inferred $\kappa _ { \perp } ( n , T )$ shows trends consistent with established plasma transport regimes.

• Classical and neoclassical transport: $\begin{array} { r } { \kappa _ { \perp } \propto \frac { n ^ { 2 } } { B ^ { 2 } \sqrt { T } } } \end{array}$ [32, p. 217][34]. Both Braginskii (classical) and Pfirsch–Schlüter (neoclassical) coefficients show the same dependence on $T ,$ n and $B ,$ although the Pfirsch–Schlüter coefficient includes the enhancement of perpendicular transport caused by toroidal geometry. Both descriptions are derived for collisional, fully ionized plasmas.

• Bohm transport: $\begin{array} { r } { \kappa _ { \perp } \propto \frac { n T } { B } } \end{array}$ [35]. Bohm transport was first introduced empirically by D. Bohm in 1949. It is usually interpreted as an anomalous transport scaling in which the step size does not depend on the gyroradius $\rho _ { L }$ but on the thermal velocity of particles.

• Gyro-Bohm transport: $\begin{array} { r } { \kappa _ { \perp } \propto \frac { n T ^ { 3 / 2 } } { B ^ { 2 } } \left[ 3 6 \right] } \end{array}$ . In contrast to Bohm transport, Gyro-Bohm transport assumes that turbulent perpendicular transport is controlled by $\rho _ { L }$ . It can be obtained by multiplying the Bohm diffusivity by $\rho _ { L } / a .$ , where a is the plasma minor radius.

The ISS04 energy confinement time scaling is compatible with gyro-Bohm diffusivity [37], and therefore core perpendicular transport in stellarators is often considered to follow gyro-Bohm-like behaviour. However, it is not clear that this behaviour can be extrapolated to the ${ \mathrm { S O L } } ,$ where open field lines, lower temperature and higher collisionality can substantially modify the transport regime.

## 2.2 PINN inverse problem

The inverse PINN formulation provides a way to infer unknown parameters in a differential equation by combining sparse measurements with the physical constraints imposed by the governing equation. In contrast to a forward problem, where the coefficients of the equation are known and the solution is sought, the inverse problem aims to recover one or more unknown coefficients from partial observations of the solution. This section introduces the formulation progressively. First, the case of an unknown constant coefficient is considered, establishing the basic structure of the inverse PINN loss function. The formulation is then extended to the case in which the unknown coefficient is not a scalar, but a function of the solution variable itself.

## Constant κ

Consider a differential equation of the form

$$
\begin{array} { r } { \mathcal { N } \left[ U ( \boldsymbol { x } ) ; \boldsymbol { \kappa } \right] = 0 , \qquad \boldsymbol { x } \in \Omega \subset \mathbb { R } , } \end{array}\tag{9}
$$

where $\mathcal { N }$ is a differential operator, U is the dependent variable, x is the independent variable, and κ is an unknown constant parameter. In the inverse PINN formulation, the solution U is approximated by a neural network $\hat { U } ( \boldsymbol { x } ; \Theta _ { U } )$ , where $\Theta _ { U }$ denotes the network weights. The unknown parameter is represented by an additional trainable scalar, κˆ. The physics residual is then defined as

$$
f ( x ; \Theta _ { U } , \hat { \kappa } ) : = \mathcal { N } \left[ \hat { U } ( x ; \Theta _ { U } ) ; \hat { \kappa } \right] .\tag{10}
$$

This residual defines the physics-informed part of the neural-network model [5]. The network weights $\Theta _ { U }$ and the parameter κˆ are learned simultaneously by minimizing a loss function that combines the residual of the differential equation with the mismatch between the neural network solution and the available measurements:

$$
{ L } = \frac { 1 } { { { N } _ { f } } } \sum _ { i = 1 } ^ { { N } _ { f } } { \left| f ( { x } _ { f } ^ { i } ; \Theta _ { U } , \hat { \kappa } ) \right| ^ { 2 } } + \frac { 1 } { { { N } _ { U } } } \sum _ { i = 1 } ^ { { N } _ { U } } { \left| U ^ { i } - \hat { U } ( { x } _ { U } ^ { i } ; \Theta _ { U } ) \right| ^ { 2 } } .\tag{11}
$$

Here, $\{ x _ { U } ^ { i } , U ^ { i } \} _ { i = 1 } ^ { N _ { U } }$ denotes the labelled dataset, while $\{ x _ { f } ^ { i } \} _ { i = 1 } ^ { N _ { f } }$ denotes the collocation points at which the residual of the differential equation is evaluated. The number of available measurements, $N _ { U }$ , is determined by the experiment and is often scarce. In contrast, the number of collocation points, $N _ { f . }$ can be sampled freely during training, subject to computational cost, and is therefore treated as a hyperparameter. Both the measurement points $\{ x _ { U } ^ { i } \}$ and the collocation points $\{ x _ { f } ^ { i } \}$ should cover the domain Ω sufficiently well to constrain the solution.

## Functional κ(U)

The inverse formulation can be extended to the case in which the unknown coefficient is not a constant, but a function of the solution variable, $\kappa = \kappa ( U )$ In this case, κ(U) is represented by a second neural network, $\hat { \kappa } ( U ; \Theta _ { \kappa } )$ . The residual becomes

$$
f ( \boldsymbol { x } ; \Theta _ { U } , \Theta _ { \kappa } ) : = \mathcal { N } \left[ \hat { U } ( \boldsymbol { x } ; \Theta _ { U } ) ; \hat { \kappa } \left( \hat { U } ( \boldsymbol { x } ; \Theta _ { U } ) ; \Theta _ { \kappa } \right) \right] .\tag{12}
$$

Note that, during training, the input to the κ network is the solution predicted by U<sup>ˆ</sup> . The weights $\Theta _ { U }$ and $\Theta _ { \kappa }$ are learned simultaneously, so that both the reconstructed solution and the inferred coefficient are consistent with the measurements and with the governing equation.

Allowing κ to be an arbitrary function introduces an additional degree of freedom into the inverse problem. Consequently, the available measurements of U and the differential equation may not be sufficient to identify a unique κ(U). To avoid this degeneracy, additional information on κ can be imposed, for example its value at a boundary or at selected reference points. The loss function can then be written as

$$
\begin{array} { c } { { { \displaystyle { \cal L } = \frac { 1 } { N _ { f } } \sum _ { i = 1 } ^ { N _ { f } } \left| f ( x _ { f } ^ { i } ; \Theta _ { U } , \Theta _ { \kappa } ) \right| ^ { 2 } } } } \\ { { { } } } \\  { { \displaystyle { \vphantom { \frac { 1 } { N _ { H } } \sum _ { i = 1 } ^ { N _ { U } } \sum _ { i = 1 } ^ { N _ { U } } \left| U ^ { i } - \hat { U } ( x _ { U } ^ { i } ; \Theta _ { U } ) \right| ^ { 2 } } } } } \\ { { { \displaystyle { \vphantom { \frac { 1 } { N _ { H } } \sum _ { i = 1 } ^ { N _ { r } } \left| \kappa ^ { i } - \hat { \kappa } ( U _ { \kappa } ^ { i } ; \Theta _ { \kappa } ) \right| ^ { 2 } } } } . } } \end{array}\tag{13}
$$

The last term enforces the available constraints on the unknown coefficient, where $\{ U _ { \kappa } ^ { i } , \kappa ^ { i } \} _ { i = 1 } ^ { N _ { \kappa } }$ are reference values used to remove the non-uniqueness of the inverse problem.

## 3 PINN model

The objective of the PINN model is to infer the perpendicular heat conductivity function, $\kappa _ { \perp } ( n , T )$ while simultaneously reconstructing the temperature and density profiles from sparse measurements. The inverse PINN model used in this work was implemented in Python using NeuroDiffEq [28, 29], an open-source library for Physics-Informed Neural Network applications built on top of PyTorch [38]. This framework provides the automatic differentiation tools required to evaluate the derivatives appearing in the transport equation residual, together with the neural network training infrastructure used to optimize the model parameters.

This section first describes the architecture of the PINN model. The optimization procedure is then presented, including the update of the neural network parameters and the adaptive loss weights used to balance the different contributions to the total loss.

## 3.1 Proposed PINN architecture

The physical constraint embedded in the loss function is the reduced 1D transport equation derived in equation 7. In the following, the measurements are assumed to be taken sufficiently close to the upstream region, so that the subscript in $n _ { \mathrm { u p } }$ is dropped for simplicity. We rewrite equation 7 to show explicitly the relevant dependencies:

$$
\frac { \partial } { \partial x } \left[ \kappa _ { \perp } ( n , T ) \frac { \partial T ( x ) } { \partial x } \right] - \frac { 5 n ( x ) [ T ( x ) ] ^ { 3 / 2 } } { 2 \sqrt { m _ { \mathrm { i } } } L _ { \mathrm { c } } } = 0 .\tag{14}
$$

That is, $T = T ( x ) , n = n ( x ) , \kappa _ { \perp } = \kappa _ { \perp } ( n , T )$

Following the functional formulation introduced in section 2.2, these three quantities are approximated by neural networks. A first possibility would be to represent the plasma profiles only as functions of $x \colon { \hat { T } } ( x )$ and $\hat { n } ( x )$ . However, this would restrict the training to a single plasma scenario, corresponding to a given transported power $P _ { \mathrm { { t r } } }$ . Since the SOL temperature profile depends on the power crossing the LCFS, as expressed through the boundary condition in equation 8, the transported power $P _ { \mathrm { { t r } } }$ is introduced as an additional input, as shown in figure 2. This follows the bundle method idea first described in Flamant et al. [16]. Hence, the temperature and density networks are expressed as

$$
\hat { T } = \hat { T } ( x , P _ { \mathrm { t r } } ; \Theta _ { T } ) , \qquad \hat { n } = \hat { n } ( x , P _ { \mathrm { t r } } ; \Theta _ { n } ) ,\tag{15}
$$

where $\Theta _ { T }$ and $\Theta _ { n }$ are the corresponding trainable parameters. Both networks are built using ResNet blocks [39] to improve gradient propagation and facilitate the training of deeper architectures. In this formulation, $P _ { \mathrm { { t r } } }$ labels different plasma profiles obtained under different heating conditions. This allows a single PINN model to be trained simultaneously on several discharges, provided that the magnetic configuration is kept fixed.

The predicted profiles are then passed to a third ResNet neural network, which represents the perpendicular conductivity as a function of the temperature and density:

$$
\hat { \kappa } _ { \perp } = \hat { \kappa } _ { \perp } \left( \hat { T } , \hat { n } ; \Theta _ { \kappa } \right) ,\tag{16}
$$

where $\Theta _ { \kappa }$ denotes the trainable parameters of the conductivity network. The three networks are trained simultaneously, so that the reconstructed profiles fit the labelled measurements while the inferred conductivity makes them consistent with the transport equation.

The physics residual is obtained by substituting the neural network approximations into equation 14:

$$
\begin{array} { r l } & { f ( x , P _ { \mathrm { t r } } ; \Theta _ { T } , \Theta _ { n } , \Theta _ { \kappa } ) = } \\ & { \frac { \partial } { \partial x } \left[ \hat { \kappa } _ { \perp } \left( \hat { T } , \hat { n } ; \Theta _ { \kappa } \right) \frac { \partial \hat { T } \left( x , P _ { \mathrm { t r } } ; \Theta _ { T } \right) } { \partial x } \right] - } \\ & { \frac { 5 \hat { n } ( x , P _ { \mathrm { t r } } ; \Theta _ { n } ) [ \hat { T } ( x , P _ { \mathrm { t r } } ; \Theta _ { T } ) ] ^ { 3 / 2 } } { 2 \sqrt { m _ { \mathrm { i } } } L _ { \mathrm { c } } } . } \end{array}\tag{17}
$$

The total loss function is defined as the weighted sum of four partial losses:

$$
L = \lambda _ { f } L _ { f } + \lambda _ { T } L _ { T } + \lambda _ { n } L _ { n } + \lambda _ { \kappa } L _ { \kappa } ,\tag{18}
$$

where $L _ { f }$ is the physics loss, $L _ { T }$ and $L _ { n }$ are the data losses for temperature and density, and $L _ { \kappa }$ imposes the boundary condition on the conductivity. These terms have different physical units and numerical scales, so their relative contribution to the total loss may differ substantially. To prevent any single term from dominating the optimization, the coefficients $\lambda _ { f } , ~ \lambda _ { T } , ~ \lambda _ { n }$ , and $\lambda _ { \kappa }$ are introduced as partial loss weights. Their purpose is to balance the contribution of the four partial losses throughout training. Their update rule together with the training procedure will be described in section 3.2.

Each loss term is computed as a mean-squared error:

$$
L _ { f } = \frac { 1 } { N _ { f } } \sum _ { i = 1 } ^ { N _ { f } } \left| f ( x _ { f } ^ { i } , P _ { \mathrm { t r } , f } ^ { i } ; \Theta _ { T } , \Theta _ { n } , \Theta _ { \kappa } ) \right| ^ { 2 } ,\tag{19}
$$

$$
L _ { T } = \frac { 1 } { N _ { D } } \sum _ { i = 1 } ^ { N _ { D } } \left| T ^ { i } - \hat { T } ( x _ { D } ^ { i } , P _ { \mathrm { t r } , D } ^ { i } ; \Theta _ { T } ) \right| ^ { 2 } ,\tag{20}
$$

$$
L _ { n } = \frac { 1 } { N _ { D } } \sum _ { i = 1 } ^ { N _ { D } } \left| n ^ { i } - \hat { n } ( x _ { D } ^ { i } , P _ { \mathrm { t r } , D } ^ { i } ; \Theta _ { n } ) \right| ^ { 2 } ,\tag{21}
$$

$$
L _ { \kappa } = \frac { 1 } { N _ { \kappa } } \sum _ { i = 1 } ^ { N _ { \kappa } } \left| \kappa _ { \perp } ^ { i } - \hat { \kappa } _ { \perp } ( T _ { \kappa } ^ { i } , n _ { \kappa } ^ { i } ; \Theta _ { \kappa } ) \right| ^ { 2 } .\tag{22}
$$

The set $\{ x _ { f } ^ { i } , P _ { \mathrm { t r } , f } ^ { i } \} _ { i = 1 } ^ { N _ { f } }$ denotes the collocation points at which the residual of the transport equation is evaluated. The set $\{ x _ { D } ^ { i } , P _ { \mathrm { t r } , D } ^ { i } , T ^ { i } , n ^ { i } \bar  \} _ { i = 1 } ^ { N _ { D } }$ denotes the labelled dataset of temperature and density measurements at different radial positions and power conditions. Finally, $\{ T _ { \kappa } ^ { i } , n _ { \kappa } ^ { i } , \kappa _ { \perp } ^ { i } \} _ { i = 1 } ^ { N _ { \kappa } }$ denotes the reference values used to constrain the conductivity. Such values are imposed at the LCFS $( x = 0 )$ , where equation 8 relates the perpendicular heat flux and therefore $\kappa _ { \perp }$ to the transported power.

![](images/00628de4b1d3b94a08e1df04dee004e0529ab2675bb5308307cf6c0b8cd1b13d.jpg)  
Figure 2: Schematic representation of the inverse PINN architecture used to infer $\kappa _ { \perp } ( n , T )$ . The temperature and density profiles are computed by independent ResNet neural networks from the perpendicular coordinate x and the transported power $P _ { \mathrm { { t r } } } ,$ while the inferred profiles are used as inputs to a third ResNet network representing the perpendicular heat conductivity. The three networks are trained simultaneously by minimizing the data losses and the residual of the 1D transport equation.

## 3.2 Network optimization

Once the loss function has been defined, the trainable parameters of the three neural networks, $\Theta _ { T } , \Theta _ { n } ,$ , and $\Theta _ { \kappa } ,$ are optimized simultaneously. The optimization problem is written as

$$
\begin{array} { r } { \Theta _ { T } ^ { * } , \Theta _ { n } ^ { * } , \Theta _ { \kappa } ^ { * } = \arg \operatorname* { m i n } _ { { L } } ( \Theta _ { T } , \Theta _ { n } , \Theta _ { \kappa } ) , } \\ { \Theta _ { T } , \Theta _ { n } , \Theta _ { \kappa } ~ } \end{array}\tag{23}
$$

where L is the total loss defined in equation $^ { 1 8 , }$ and $\Theta _ { T } ^ { * } , \Theta _ { n } ^ { * } ,$ , and $\Theta _ { \kappa } ^ { * }$ denote the optimized network parameters after training. The minimization is performed using the Adam optimizer [40], a stochastic gradient-based method that adaptively updates each parameter using estimates of the first and second moments of the gradients. The gradients are computed by backpropagation using automatic differentiation, which provides both the derivatives required to evaluate the differential equation residual and the gradients to update the neural network parameters.

The weights assigned to the partial losses play an important role in the optimization. They serve two purposes: first, to bring the different partial losses to comparable numerical scales at the beginning of training; and second, to dynamically balance their contribution during training according to the magnitude of their backpropagated gradients. The loss weights are defined as

$$
\lambda _ { f } = \alpha _ { f } ,\tag{24}
$$

$$
\lambda _ { j } = \alpha _ { j } \gamma _ { j } ,\tag{25}
$$

where $j \in \{ T , n , \kappa \}$ . The coefficients $\alpha _ { j }$ provide the initial normalization of the partial losses. They are

computed from the losses evaluated at epoch zero as

$$
\alpha _ { f } = \frac { 1 } { L _ { f } ^ { ( 0 ) } } ,\tag{26}
$$

$$
\alpha _ { j } = \frac { 1 } { L _ { j } ^ { ( 0 ) } } ,\tag{27}
$$

where the superscript (0) denotes evaluation before training. This normalization initializes the contribution of each partial loss to be all equally 1.

The coefficients $\gamma _ { j }$ are then updated dynamically during training every $l _ { \gamma }$ epochs, following the balancing strategy proposed by Deguchi et al. [41]. At each update index $k ,$ an instantaneous correction factor is computed as

$$
\hat { \gamma } _ { j } ^ { ( k ) } = \frac { \left. \nabla _ { \Theta } \left( \alpha _ { f } L _ { f } ^ { ( k ) } \right) \right. _ { 2 } } { \left. \nabla _ { \Theta } \left( \alpha _ { j } L _ { j } ^ { ( k ) } \right) \right. _ { 2 } } ,\tag{28}
$$

where $\nabla _ { \Theta }$ denotes the gradient with respect to all the trainable network parameters. The adaptive coefficient is then updated using an exponential moving average:

$$
\gamma _ { j } ^ { ( k ) } = \beta \gamma _ { j } ^ { ( k - 1 ) } + \left( 1 - \beta \right) \hat { \gamma } _ { j } ^ { ( k ) } ,\tag{29}
$$

where $\beta \in [ 0 , 1 )$ controls the smoothing of the update, and $\gamma _ { j } ^ { ( 0 ) }$ is initialized to 1. In this formulation, the physics loss is taken as the reference term and therefore no additional dynamic coefficient $\gamma _ { f }$ is introduced for it. The adaptive factors $\gamma _ { j }$ rescale the data and conductivity losses so that their backpropagated gradient norms remain comparable to that of the physics residual. The exponential averaging in equation 29 reduces fluctuations in the weights and prevents noisy updates from destabilizing the training.

## 4 Validation with synthetic data

The PINN model is first validated using synthetic data. This step provides a controlled test case in which the density data and the functional form of the perpendicular conductivity are prescribed a priori, while the temperature data are obtained by solving the transport equation numerically, as described in section 4.1. The inferred conductivity, ${ \hat { \kappa } } _ { \perp } ( n , T )$ , can therefore be compared directly with the ground-truth function $\kappa _ { \perp } ( n , \bar { T } )$ used to generate the data. In section 4.2, we first present a representative application of the model. We then perform a two dimensional scan as a function of the number of profiles processed simultaneously and the number of radial positions sampled per profile. This scan allows us to assess how the PINN performance depends on the amount of available data, and whether the reconstruction accuracy saturates beyond a certain amount of data.

## 4.1 Synthetic data generation

The density profile is prescribed as an exponentially decaying function,

$$
n ( x ) = n _ { 0 } \exp \left( - \frac { x } { W } \right) ,\tag{30}
$$

where $n _ { 0 } = 1 0 ^ { 1 8 } \mathrm { m } ^ { - 3 }$ is a typical density value at the LCFS, and $W = 1$ 4 mm characterizes the radial decay length of the density profile. The perpendicular conductivity is chosen as

$$
\kappa _ { \perp } ( n , T ) = k _ { 0 } n T ,\tag{31}
$$

where $k _ { 0 } = 1 \mathrm { m } ^ { 2 } \mathrm { s } ^ { - 1 } \mathrm { e V } ^ { - 1 }$ . This imposed conductivity is not intended to represent a definitive SOL transport scaling. Instead, it provides a known dependence on both density and temperature that remains physically reasonable in light of current understanding of SOL transport, allowing the inverse method to be quantitatively assessed.

To generate the synthetic temperature profiles, we introduce the transient relaxation equation associated with the steady transport balance in equation 14:

$$
n \frac { \partial T } { \partial t } = \frac { \partial } { \partial x } \left[ \kappa _ { \perp } \frac { \partial T } { \partial x } \right] - \frac { 5 n T ^ { 3 / 2 } } { 2 \sqrt { m _ { \mathrm { i } } } L _ { \mathrm { c } } } ,\tag{32}
$$

For each transported power value, equation 32 is solved subject to the LCFS heat-flux boundary condition in equation 8 for $x _ { \mathrm { i n i t } } = 0$ . The outer boundary of the computational domain is placed sufficiently far from the LCFS, at $x _ { \mathrm { e n d } } = 1 4 0$ mm, such that the temperature decays to approximately zero before reaching it. Consequently, the solution in the region of interest is insensitive to the specific boundary condition imposed at $x = x _ { \mathrm { e n d } }$ . Non-negative temperatures, $T ( x , t ) \geq 0 ,$ are enforced throughout the simulation. The LCFS area entering equation 8 is approximated as

$$
S _ { \mathrm { L C F S } } \approx 4 \pi ^ { 2 } a R ,\tag{33}
$$

where a and R are the minor and major radii of the device, respectively. For TJ-II, this gives $S _ { \mathrm { L C F S } } \approx 1 1 . 8 \mathrm { m } ^ { 2 }$

Equation 32 is solved in Python using the method of lines. Spatial derivatives are approximated by finite differences on a uniform grid spanning $x ~ \in ~ [ 0 , 1 4 0 ]$ mm with spacing $\Delta x \ = \ 0 . 1 4$ mm, yielding a system of ordinary differential equations that is advanced with the implicit Radau solver implemented in the solve\_ivp routine from SciPy [42]. The Radau solver controls the local error ϵ according to the criterion $\epsilon < a _ { \mathrm { t o l } } + r _ { \mathrm { t o l } } | T |$ , with relative and absolute tolerances set to $r _ { \mathrm { t o l } } = 1 0 ^ { - 3 }$ and $a _ { \mathrm { t o l } } = 1 0 ^ { - 6 } .$ , respectively. The system is integrated up to $t _ { \mathrm { e n d } } = 0 . 0 5 \ : s ,$ at which point a stationary state is reached. Specifically, the time derivative term averaged over the spatial domain is ten orders of magnitude smaller than that of each of the two terms on the right-hand side of equation 32. In this limit, equation 32 reduces to equation $^ { 1 4 , }$ and the resulting stationary temperature profile is used as synthetic data.

The training dataset, $\{ x _ { D } ^ { i } , P _ { \mathrm { t r } , D } ^ { i } , T ^ { i } , n ^ { i } \} _ { i = 1 } ^ { N _ { D } } ,$ , is then obtained by sampling the stationary profiles at equally-spaced discrete values of $x _ { D } ^ { i }$ and $P _ { \mathrm { t r } , D } ^ { i } .$ The selected ranges, $x _ { D } ^ { i } ~ \in ~ [ 0 , 2 4 ]$ mm and $P _ { \mathrm { t r } , D } ^ { i } ~ \in ~ \left[ 1 0 0 , 3 0 0 \right] \mathrm { k W } .$ , are consistent with values typically observed in TJ-II [43]. Figure 3 shows an example synthetic dataset with $N _ { \mathrm { P } } = 3$ distinct values of $P _ { \mathrm { t r } , D }$ and $N _ { \mathrm { x } } = 8$ distinct radial positions. Markers indicate the sampled synthetic data. For each unique radial position, $N _ { \mathrm { R } }$ repeated measurements of $T$ and n are generated by adding independent Gaussian noise to the original synthetic data. This procedure mimics the experimental acquisition process, in which temperature and density are measured at a given time resolution and are affected by plasma fluctuations. Assuming a sampling frequency of 2 kHz, consistent with Langmuir probe diagnostics, and a discharge duration of 100 ms, we set $N _ { \mathrm { R } } = 2 0 0$ . The standard deviations of the Gaussian noise are chosen as $\sigma _ { \mathrm { T } } ~ = ~ 1$ eV for the temperature and $\sigma _ { \mathrm { n } } ~ = ~ 1 0 ^ { 1 7 } ~ \mathrm { m } ^ { - 3 }$ for the density, simulating typically observed measurement errors and plasma fluctuations [43].

![](images/895b4708e21e5634e082072a4d655098b67f7f827080c99d5fc073d36f7aa595.jpg)

![](images/6e60b0cd97d2d13811fa811a9613a8aad06706158fa8faaf26604297a375ca3a.jpg)  
Figure 3: Comparison between synthetic data and neural network reconstructions for a representative use case of the model. Left: temperature profiles $T ( x )$ for three transported power values. Right: density profiles n(x) for the same cases. Markers correspond to noisy synthetic measurements sampled at discrete radial positions, while lines represent the smooth profiles reconstructed by the neural networks.

The conductivity data at the LCFS, $\{ \kappa _ { \perp } ^ { i } \} _ { i = 1 } ^ { N \kappa }$ , are computed from equation 31 using the noiseless synthetic temperature and density values at the LCFS. One conductivity value at the LCFS is assigned to each distinct transported power case; therefore, $N _ { \kappa } = N _ { \mathrm { P } }$ . For the example shown in figure 3, this yields three LCFS conductivity values, corresponding to the three transported power profiles.

## 4.2 Numerical results

This section first presents a representative training run to illustrate the general behaviour of the PINN inverse problem solver and the reconstruction of $\kappa _ { \perp } ( n , T )$ . A bootstrap resampling procedure is then introduced to estimate the uncertainty and reproducibility of the inferred conductivity. Finally, a systematic scan is performed over the number of input profiles, $N _ { \mathrm { P } }$ , and the number of radial measurement positions per profile, $N _ { \mathrm { x } }$ The hyperparameters listed in table 1 are kept fixed throughout all numerical experiments. The collocation set is constructed as the Cartesian product of 256 $x _ { f }$ values and 64 $P _ { \mathrm { t r } , f }$ values, resulting in $N _ { f } = 2 5 6 \times 6 4 = 1 6 3 8 4$ collocation points.

## General example

We first apply the model to the synthetic data from figure 3. Figure 4 shows the training loss evolution of all loss terms. The temperature and density data losses reach a higher plateau earlier than the κ and $f$ losses. This plateau represents an irreducible error, or noise floor, caused by the added measurement noise: as shown in figure 3, the predicted profiles $\hat { T }$ and nˆ follow the mean trend of each data cluster but cannot exactly interpolate every noisy data point, so further fitting or training cannot eliminate this residual. Despite this limitation, the optimization converges to a stable loss plateau. The model requires approximately 3 hours to converge when trained on the Turgalium HPC cluster using an NVIDIA H100 GPU.

![](images/af7b2a15cfc645fda891c032ae67e3b113ccbe36f59326a49899bcd025fc5946.jpg)  
Figure 4: Evolution of the individual partial losses normalized by the static coefficients $\alpha _ { j } ,$ and the resulting total loss, during the representative synthetic data run. The losses are shown in logarithmic scale as functions of the training epoch.

We define the normalized physics-consistency score $ { \boldsymbol { S } } _ { f }$ as

$$
\mathcal { S } _ { f } = 1 - \frac { \sum _ { i = 1 } ^ { N _ { \mathrm { D } } } | f ^ { i } | } { \sum _ { i = 1 } ^ { N _ { \mathrm { D } } } \left( | \mathcal { L } ^ { i } | + | \mathcal { R } ^ { i } | \right) } ,\tag{34}
$$

Here, $\mathcal { L } ^ { i }$ and $\mathcal { R } ^ { i }$ denote the first and second physical terms of equation 17, respectively, evaluated at the i-th data point, and the local residual is $f ^ { i } = \mathcal { L } ^ { i } - \mathcal { R } ^ { i }$

Table 1: Training hyperparameters used in the PINN model.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Total number of training epochs.</td><td> $5 \cdot 1 0 ^ { 5 }$ </td></tr><tr><td>Learning rate used by the optimizer.</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Number of neurons in each hidden layer of the neural networks.</td><td>128</td></tr><tr><td>Number of ResNet blocks in the neural networks.</td><td>5</td></tr><tr><td>Activation function used in the hidden layers.</td><td>ELU</td></tr><tr><td>Activation function used in output layer.</td><td>Identity</td></tr><tr><td>Exponential smoothing factor used in the dynamic loss weight update  $( \beta )$ </td><td>0.999</td></tr><tr><td>Update interval used in the dynamic loss weight update  $( l _ { \gamma } )$ </td><td>10</td></tr><tr><td>Number of generated collocation points in the spatial coordinate.</td><td>256</td></tr><tr><td>Number of generated collocation points in the power coordinate.</td><td>64</td></tr><tr><td>Number of training batches evaluated per epoch.</td><td>1</td></tr><tr><td>Random seed used for stochastic initialization and sampling.</td><td>Not fixed</td></tr></table>

The normalized physics-consistency score $\boldsymbol { \mathcal { S } } _ { f } ,$ which ranges from 0 to 1, measures how well the differential transport equation is balanced by comparing the absolute local residual, $| f ^ { i } |$ , with the combined magnitudes of its two physical terms, $\vert \mathcal { L } ^ { i } \vert + \vert \mathcal { R } ^ { i } \vert .$ A value of $s _ { f } ~ = ~ 1$ corresponds to exact balance, while lower values indicate greater normalized imbalance. This score is used only after training as a diagnostic and interpretive measure of physical consistency; it is not a training loss, stopping criterion, or model-selection quantity. For the present case, $S _ { \mathrm { f } } ~ = ~ 0 . 9 9 8$ , indicating that the differential equation is nearly perfectly satisfied.

The main quantity of interest is the inferred conductivity $\hat { \kappa } _ { \perp } ( T , n )$ . Figure 5 compares ${ \hat { \kappa } } _ { \perp } ( T , n )$ pointwise with the prescribed ground-truth conductivity $\kappa _ { \perp } ( T , n )$ defined in equation 31 over the $( T , n )$ domain. The right panel shows the percentage difference normalized by the ground-truth, in absolute values:

$$
\delta _ { \kappa } ( T , n ) = \left| \frac { \kappa _ { \perp } ( T , n ) - \hat { \kappa } _ { \perp } ( T , n ) } { \kappa _ { \perp } ( T , n ) } \right| \cdot 1 0 0 .\tag{35}
$$

In the region constrained by the training dataset, the magnitude $\delta _ { \kappa }$ is generally below 10%.

In the synthetic data case, the reconstruction error can be evaluated directly because the ground truth conductivity is known. However, this comparison is not possible when the framework is applied to experimental data. An alternative measure is therefore needed to evaluate the model consistency in predicting $\hat { \kappa } _ { \perp } \left( T , n \right)$ without access to its true value. For this purpose, we use bootstrap resampling [44]. The basic resampling unit is a single aligned observation, defined by the tuple $\left( T , n , x , P _ { \mathrm { t r } } \right)$ . All observations are pooled across transported power values, and bootstrap samples are generated by drawing these tuples with replacement from the complete dataset. Each bootstrap sample contains the same number of observations as the original dataset, preserving the underlying distribution while differing in the specific observations selected. This procedure generates alternative realizations of the empirical dataset that can be used to quantify the sensitivity of the inferred ${ \hat { \kappa } } \perp ( T , n )$ to variations in the sampled training data.

The PINN is trained independently on each bootstrap sample, producing an ensemble of conductivity estimates, $\{ \hat { \kappa } _ { \perp } ^ { ( b ) } ( T , n ) \} _ { b = 1 } ^ { N _ { \mathrm { B } } } ,$ , where $N _ { \mathrm { B } }$ is the number of bootstrap realizations. The pointwise standard deviation of this ensemble can then be used to quantify the sensitivity to variations in the sampled measurements. Namely, figure 6 shows $\tilde { \sigma } _ { \kappa } ,$ defined as

$$
\begin{array} { l } { { \displaystyle { \tilde { \sigma } } _ { \kappa } ( T , n ) = 1 0 0 \frac { 1 } { \overline { { \kappa } } _ { \perp } ( T , n ) } } } \\ { { \displaystyle ~ \times \left[ \frac { 1 } { N _ { \mathrm { B } } - 1 } \sum _ { b = 1 } ^ { N _ { \mathrm { B } } } \left( \hat { \kappa } _ { \perp } ^ { ( b ) } ( T , n ) - \overline { { \kappa } } _ { \perp } ( T , n ) \right) ^ { 2 } \right] ^ { 1 / 2 } } , } \end{array}
$$

with

(36)

$$
\overline { { \kappa } } _ { \perp } ( T , n ) = \frac { 1 } { N _ { \mathrm { B } } } \sum _ { b = 1 } ^ { N _ { \mathrm { B } } } \hat { \kappa } _ { \perp } ^ { ( b ) } ( T , n ) ,\tag{37}
$$

This is the coefficient of variations of the predictions, expressed as a percentage.

The region covered by the measurements shows the lowest standard deviation, indicating that the PINN produces consistent conductivity estimates when interpolating within the domain constrained with data. In contrast, the bottom-right and upper-left regions of the plot show larger deviations, since no training data are available there and the model is effectively extrapolating. The largest deviations appear at the lowest values of n and $T ,$ where the conductivity approaches zero and its relative standard deviation therefore increases. It is also worth noting that the regions with higher $\tilde { \sigma } _ { \kappa }$ correspond, in general, to those with larger relative errors in figure 5. This agreement supports the use of bootstrap resampling as a practical indicator of the sampling sensitivity of the inferred conductivity.

![](images/c147bdd88454e5a55751a4611a2588e77a606c71f11d41b47ca1452ce76a2c0b.jpg)

![](images/095324c18326f6995e79fad91a962456f8a9ce8bc9aa85105166476ae04586a3.jpg)

![](images/d0c7a8a093861609cc3cee3f7207026ca7918c50a17911f075f53e038b32b371.jpg)  
Figure 5: Comparison between the ground truth $\kappa _ { \perp }$ and PINN-reconstructed $\hat { \kappa } _ { \perp }$ perpendicular thermal conductivity as a function of temperature and density. The left panel shows the imposed ground truth conductivity (equation 31). The central panel shows the PINN-reconstructed conductivity. The right panel shows the relative error between $\kappa _ { \perp }$ and $\hat { \kappa } _ { \perp }$ . Black markers indicate the synthetic training dataset $\{ T ^ { i } , n ^ { i } \}$ }.

![](images/0fe1f6ca6d1b9e03130b160309a3709866ce54ede1d6bad20e6cd2349ce5e7e7.jpg)  
Figure 6: Pointwise standard deviation of the inferred perpendicular conductivity $\hat { \kappa } _ { \perp } ( T , n )$ obtained from 10 bootstrap realizations. $\tilde { \sigma } _ { \kappa }$ is normalized by the mean $\overline { { \kappa } } _ { \perp } ( T , n )$ , and given as a percentage. Black markers indicate the synthetic training dataset $\{ T ^ { i } , { \bar { n } } ^ { i } \}$

## Scan in number of profiles and positions

Experimental data acquisition in fusion devices is often costly and technically limited. It is therefore useful to estimate how the performance of the inverse PINN depends on the amount of available data, in order to identify a suitable trade-off between reconstruction accuracy and experimental effort. In the present study, the amount of training data is controlled by $N _ { \mathrm { P } }$ and $N _ { \mathrm { x } }$ . The objective of this scan is therefore to determine how large $N _ { \mathrm { P } }$ and $N _ { \mathrm { x } }$ need to be to achieve a given accuracy in the inferred conductivity.

The performance is evaluated through the Mean Absolute Percentage Error (MAPE) of $\kappa _ { \perp } ( n , T )$ , defined as

$$
\mathrm { M A P E } _ { \kappa } [ ^ { \circ } \circ ] = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| \frac { \kappa _ { \perp } \left( n _ { i } , T _ { i } \right) - \hat { \kappa } _ { \perp } \left( n _ { i } , T _ { i } \right) } { \kappa _ { \perp } \left( n _ { i } , T _ { i } \right) } \right| \cdot 1 0 0 ,\tag{38}
$$

where $\{ n _ { i } , T _ { i } \} _ { i = 1 } ^ { N }$ form an evenly distributed 2D grid spanning the full $( n , T )$ domain shown in figure 5; that is, N is the number of evaluation points covering the entire case space, and MAPE is averaged over this whole domain rather than only the region constrained by the training data.

Figure 7 shows the median of the $\mathbf { M A P E } _ { \kappa }$ obtained for different combinations of $N _ { \mathrm { P } }$ and $N _ { \mathrm { x } } ,$ with five bootstrap realizations performed for each case. As expected, the largest error is found for the most weakly constrained cases, corresponding to $N _ { \mathrm { P } } = 1$ . Increasing both the number of profiles and the number of radial positions generally reduces the reconstruction error, although the improvement becomes less pronounced once the dataset is sufficiently informative. In this case, the error is found to saturate around $N _ { \mathrm { P } } = 5$ and $N _ { \mathrm { x } } = 6$ , beyond which additional data provide only a marginal gain in accuracy.

## 5 Application to experimental data

This section applies the proposed methodology to an experimental dataset. Only a limited amount of real experimental data was available at the time of this study, so the results presented here are meant to test whether the framework can be applied to real measurements, rather than to perform a complete systematic study of the parametric dependencies of $\kappa _ { \perp }$ . Applying the method to broader and betterconstrained experimental datasets is left for future work. Section 5.1 briefly describes the experimental setup used to obtain the measurements. Section 5.2 presents the experimental dataset considered in the analysis. Finally, section 5.3 shows and discusses the results obtained from the experimental application of the PINN framework.

![](images/a5bb010effde6bf536e252c8d7c366f1b20c2dee42d855955c94fb55d403ec6a.jpg)  
Figure 7: Mean absolute percentage error of the inferred perpendicular conductivity, $\mathrm { M A P E } _ { \kappa , }$ , as a function of the number of profiles used for training, N , and the number of radial measurement positions per profile, $N _ { \mathbf { x } } .$ . For each pair $( \dot { N } _ { \mathrm { P } } , N _ { \mathrm { x } } ) .$ , five bootstrap realizations are performed, and the plotted value corresponds to the median value of the $\mathbf { M A P E } _ { \kappa }$ distribution.

## 5.1 Experimental setup

TJ-II was selected for this exploratory application because it is a well-characterized, flexible heliac-type stellarator with a long operational history [45], and because its helium-beam diagnostic provides simultaneous, spatially resolved measurements of the SOL electron density and temperature profiles [46] required by the PINN framework. TJ-II is located at the Laboratorio Nacional de Fusión of CIEMAT, Madrid.

The helium-beam diagnostic injects a neutral helium-beam into the plasma, and the resulting helium line emission is interpreted through a collisional-radiative model to reconstruct radial profiles of the edge electron temperature and density [47]. Figure 8 illustrates the diagnostic layout, showing the helium-beam entering the plasma from the upper part of the poloidal section. The measurement path is defined by the helium-beam trajectory, along which the neutral beam is then progressively attenuated as it penetrates into the plasma.

The temporal resolution of the helium-beam diagnostic is limited to 50 ms, which provides $N _ { \mathrm { R } } = 2$ measurements per discharge. The spatial resolution is limited to 3.5 mm, corresponding to $N _ { \mathrm { x } } = 4$ measurement positions within the SOL. This temporal and spatial coverage is insufficient for a complete experimental characterization of the parametric dependencies of $\kappa _ { \perp }$ in a systematic study. Nevertheless, it supports the present exploratory objective: testing the PINN framework on real experimental profiles and obtaining an initial estimate of the inferred conductivity under realistic data constraints.

![](images/04bc3359069ce919aa59c07b62943e88c140af60206827179af313ae1fc27848.jpg)  
Figure 8: Schematic poloidal cross section of the helium-beam diagnostic setup. The helium-beam is injected vertically into the plasma through the vacuum vessel, while the emitted radiation is collected horizontally by the observation optics. The upper section shows the gas injection and pumping system, while the lower section indicates the manometer used to monitor the gas pressure near the injection line. Diagram adapted from [46].

## 5.2 Experimental dataset

The experimental dataset was obtained from 13 discharges performed during the TJ-II Spring 2026 campaign. Of these, five discharges were conducted at a heating power of $P _ { \mathrm { h } } = 3 5 0 ~ \mathrm { k W }$ , and eight discharges at $P _ { \mathrm { h } } = 5 0 0 ~ \mathrm { k W } .$ , using electron cyclotron resonance heating (ECRH). The line-averaged plasma density was kept at low values, $\overline { { n } } _ { e } \approx 5 \cdot 1 \mathrm { { 0 } } ^ { 1 8 } \mathrm { { \dot { m } } } ^ { - 3 }$ , so the measured radiation losses remained small, $P _ { \mathrm { r a d } } \sim 0 . 0 1 P _ { \mathrm { h } } .$ Radiative losses are therefore neglected, and, under the assumed steady-state power balance, the heating power is approximated as the transported power,

$$
P _ { \mathrm { h } } = P _ { \mathrm { r a d } } + P _ { \mathrm { t r } } \approx P _ { \mathrm { t r } } .\tag{39}
$$

Figure 9 shows the experimental dataset for the two power values. The temperature measurements, shown with markers in the left panel, exhibit a weaker dependence on transported power than the synthetic data presented earlier. Nevertheless, higher temperatures are generally observed for the higher transported power case. The density was intended to be kept approximately constant across all discharges, and the measurements from both power cases are therefore found to be similar, as reflected in the right panel.

The localization of the LCFS typically has an uncertainty of 5–10 mm [48]. Therefore, in the present analysis, the LCFS position is estimated using a heuristic profile-based criterion, in which the LCFS is identified as the first point among the last four measurements for which both T and n show a clear decreasing trend. This criterion provides a simple and consistent way to select reasonable SOL data to feed the PINN, adequate for the exploratory purposes of this work.

It was necessary to estimate the perpendicular conductivities at the LCFS for the two different powers in order to calculate the loss function, equation 18. To obtain these values, the experimental temperature profiles were pre-fitted in the vicinity of the LCFS. The derivative dT/dx at $x = 0$ was then computed from the fitted profiles and inserted into equation 8, allowing $\kappa _ { \mathrm { ~ | ~ } } ^ { \mathrm { L C F S } }$ to be solved directly. Using this procedure, the LCFS conductivity values obtained to constrain the

PINN were $\kappa _ { \scriptscriptstyle 1 } ^ { \mathrm { L C F S } } = ( 1 2 . 5 \pm 0 . 8 ) \cdot 1 0 ^ { 1 9 } ~ \mathrm { m } ^ { - 1 } \mathrm { s } ^ { - 1 }$ for $P _ { \mathrm { t r } } = 3 5 0 ~ \mathrm { k W }$ and $\kappa _ { \scriptscriptstyle 1 } ^ { \mathrm { L C F S } } = ( 1 6 . 1 \overset { \cdot } { \pm } 0 . 7 ) \cdot 1 0 ^ { 1 9 } \mathrm { m } ^ { - 1 } \mathrm { s } ^ { - 1 }$ for $P _ { \mathrm { t r } } = 5 0 0 ~ \mathrm { k W }$ . The reported uncertainties originate from the standard errors of the fitted profiles and are propagated to $\kappa _ { \perp } ^ { \mathrm { L C F S } }$ through equation 8.

## 5.3 Experimental results

The PINN framework was applied to the experimental dataset described in section 5.2, using the same reduced transport model and hyperparameters validated with synthetic data. Since the ground truth conductivity is not known for experimental measurements, the results are interpreted through the consistency of the reconstructed profiles and the variability of the inferred conductivity across bootstrap realizations.

The PINN model was trained and evaluated for $N _ { \mathrm { B } } ~ = ~ 1 0$ independent bootstrap realizations. As done in section 4, the experimental dataset was resampled for each bootstrap realization by drawing $\left( T , n , x , P _ { \mathrm { t r } } \right)$ tuples with replacement from the pooled dataset. In addition, the LCFS conductivity associated with each transported power value was independently sampled from a Gaussian distribution centered on its estimated value, with a standard deviation equal to the corresponding reported uncertainty. This additional sampling propagates the uncertainty in the LCFS conductivity constraints through the bootstrap analysis.

Figure 9 shows, with solid and dashed lines, the bootstrap-averaged PINN reconstructions ${ \hat { T } } ( x )$ and ${ \hat { n } } ( x )$ , while shadow regions represent the 90% confidence intervals for the two transported power values. The reconstructed profiles reproduce the experimental data reasonably well, and the confidence intervals show reproducible computations, with variations of ±2 eV in temperature and $\pm 0 . 5 ~ 1 0 ^ { 1 7 } \mathrm { m } ^ { - 3 }$ in density. For $x \gtrsim 8$ mm, density profiles become weakly varying and they reach low values, providing a first estimate of the radial extent of the measured SOL region.

The physics-consistency score $ { \boldsymbol { S } } _ { f }$ applied to the bootstrap ensemble yields ${ \cal S } _ { f } = 0 . 8 1 \bar { 4 } .$ This value is considerably lower than that obtained in the synthetic case, where $S _ { f } = 0 . 9 9 8$ . This difference may indicate that the transport physics underlying the experimental data are not fully captured by equation 14, in contrast to the synthetic case, where the governing equation is satisfied by construction. Given the limited experimental dataset, this lower score should be interpreted as an exploratory measure rather than as a calibrated acceptance criterion for the reduced transport model.

![](images/e448da0c5bf4b539f9fb121fe6ef5956b6e826d992ff2c0c1e155508e40216e6.jpg)

![](images/acdc8ac05faba8fc33f14fb12b1d247aab42254a33f60324ab98363f213a074e.jpg)  
Figure 9: Comparison between the experimental dataset and the neural network reconstructions. Left: temperature profiles $T ( x ) f o r$ two transported power values. Right: density profiles $n ( x ) f o r$ the same cases. Markers correspond to the experimental measurements, with squares corresponding to $P _ { \mathrm { t r } } = 3 5 0$ kW and triangles to $P _ { \mathrm { t r } } = 5 0 0 ~ \mathrm { k W } .$ . Lines represent the smooth profiles reconstructed by the neural networks averaged over all the bootstrapped runs. Shaded regions indicate the corresponding 90% bootstrap confidence intervals, representing the uncertainty associated with variations in the sampled training data.

![](images/5daca8ed5f42781fbb070a5b772b0451e64fa025e69f5a6ec9875de5a0f79567.jpg)

![](images/25bb73d6b3c69b38e156d8d42e752b47dee5968cef672c72b2921dd6a9263a5d.jpg)  
Figure 10: Inferred perpendicular conductivity from the TJ-II experimental dataset. Left: ensemble mean conductivity, $\overline { { \kappa } } _ { \perp } ( T , n ) ,$ obtained from the bootstrap realizations. Right: normalized ensemble standard deviation, $\tilde { \sigma } _ { \kappa } ( T , n )$ , expressed as a percentage. Markers indicate the experimental data points used for training, with squares corresponding to $P _ { \mathrm { t r } } = 3 5 0 ~ \mathrm { k W }$ and triangles to $P _ { \mathrm { t r } } = 5 0 0 ~ \mathrm { k W } .$

Figure 10 shows the ensemble mean of the inferred conductivity, $\overline { { \kappa } } _ { \perp } ( T , n )$ obtained from the 10 bootstrap realizations, together with its normalized ensemble standard deviation, $\tilde { \sigma } _ { \kappa } ( T , n )$ over the evaluated $( T , n )$ domain. As in the synthetic validation, the inferred conductivity should be interpreted primarily within the region of the $( T , n )$ space covered by the measurements, where the PINN is constrained by both the labelled data and the transport equation. The largest deviations are indeed found in the lower-right corner of the plot, where experimental data are absent.

Focusing on the left panel of figure 10, the inferred conductivity ranges approximately between $1 0 ^ { 1 8 }$ and $1 0 ^ { 2 0 } \ m ^ { - 1 } \mathrm { s } ^ { - 1 }$ This corresponds to a gyro-Bohmnormalized conductivity of approximately $\kappa _ { \perp } / \kappa _ { \parallel } ^ { \mathrm { g B } }$ ∼ 800. For comparison, values around $\kappa _ { \perp } / \kappa _ { \perp } ^ { \mathrm { g B } } \sim$ 100 have been reported for the Wendelstein $7 – \dot { X } ( W - X )$ stellarator [49, 50]. The gyro-Bohm conductivity used for this normalization is computed as

$$
\kappa _ { \perp } ^ { \mathrm { g B } } = \frac { n v _ { \mathrm { t h , i } } ^ { 3 } } { \Omega _ { \mathrm { i } } ^ { 2 } a } = \sqrt { \frac { 8 m _ { \mathrm { i } } } { e ^ { 4 } } } \frac { n T ^ { 3 / 2 } } { B ^ { 2 } a } ,\tag{40}
$$

where $v _ { \mathrm { t h , i } } = \sqrt { 2 T _ { \mathrm { i } } / m _ { \mathrm { i } } }$ is the ion thermal velocity and $\Omega _ { \mathrm { i } } ~ = ~ e B / m _ { \mathrm { i } }$ is the ion gyrofrequency. For the TJ-II gyro-Bohm reference value, $B ~ = ~ 1 ~ \mathrm { T }$ and $\ a \ = \ 0 . 2 2$ m are adopted, together with a representative SOL temperature $T = 4 0 \ \mathrm { e V } .$ . Subject to compatible definitions and normalizations, the normalized conductivity inferred for TJ-II is roughly eight times larger than the values reported for $W 7 { - } X$ This difference may partly reflect variations in the underlying SOL transport regimes, highlighting the need for a broader multi-machine analysis to better characterize and understand perpendicular heat transport across different stellarator devices.

Regarding the dependencies of $\overline { { \kappa } } _ { \perp } ( T , n )$ , the maximum values are found at high T and high $n ,$ corresponding approximately to the LCFS region. From this region outward, the inferred conductivity decreases as both T and n decrease, in qualitative agreement with the trends expected from Bohm-like or gyro-Bohm-like scalings. However, for $n < 4 \cdot 1 0 ^ { 1 7 } \mathrm { ~ m } ^ { - 3 }$ , the temperature dependence appears to weaken. This behaviour is not directly explained by the reference scalings discussed in section 2.1. It may instead reflect the increased relative uncertainty of the inferred conductivity, or the fact that $\overline { { \kappa } } _ { \perp }$ represents an effective transport coefficient that absorbs physics not explicitly included in the reduced 1D model. As stated above, these results should not be interpreted as a definitive analysis of $\kappa _ { \perp }$ and its parametric dependencies, but rather as a first exploratory application of the method to real experimental data. A broader and better-constrained experimental dataset will be required to validate these trends and perform a complete and systematic transport analysis.

## 6 Conclusions

This work has developed and tested an inverse PINN framework to infer the perpendicular heat conductivity, $\kappa _ { \perp } ( n , T )$ , in the SOL of magnetically confined fusion plasmas. The model combines sparse measurements of temperature and density profiles with a reduced 1D transport equation, allowing the conductivity to be inferred as a functional dependence on the local plasma density and temperature. The proposed architecture uses three neural networks: two reconstruct $\hat { T } ( x , P _ { \mathrm { t r } } )$ and $\hat { n } ( x , P _ { \mathrm { t r } } )$ , while a third represents $\hat { \kappa } _ { \perp } ( T , n )$ . By including the transported power as an input, the model can be trained simultaneously on several plasma profiles obtained under different heating conditions.

The method was first validated using synthetic data generated from a prescribed conductivity function. In this controlled case, the PINN recovered the imposed dependence of $\kappa _ { \perp } ( n , T )$ , with errors below 10% in the data-constrained region and a differential equation satisfaction metric of $ { \boldsymbol { S } } _ { f } = 0 . 9 9 8$ Bootstrap resampling provided a useful indicator of prediction consistency and reliability, with larger variability appearing in weakly sampled regions of the $( T , n )$ domain. A scan in the number of profiles and radial measurement positions showed that the reconstruction accuracy improves with data availability, with the error saturating around $N _ { \mathrm { P } } = 5$ and $N _ { \mathrm { x } } = 6$ for the synthetic conditions considered.

The framework was then applied to a TJ-II experimental dataset obtained with the helium-beam diagnostic. This application was intended as a first exploratory test with real data, rather than a complete systematic study of $\kappa _ { \perp }$ dependencies. The reconstructed profiles reproduced the experimental measurements reasonably well, with a physicsconsistency score of ${ \cal S } _ { f } = 0 . 8 2$ . The inferred conductivity was found in the range $1 0 ^ { 1 8 } – 1 0 ^ { 2 0 } \ \mathbf { m } ^ { - 1 } \mathbf { s } ^ { - 1 }$ corresponding to a gyro-Bohm-normalized value of order $\kappa _ { \perp } / \kappa _ { \perp } ^ { \mathrm { g B } } \sim 8 0 0$ . This value is roughly eight times larger than that reported for W7-X, a contrast that possibly reflects differences in the underlying SOL transport regimes between the two devices, and which motivates a broader multi-device analysis.

Overall, the results indicate that inverse PINNs can provide a useful framework for estimating effective SOL transport coefficients from sparse experimental profiles. However, the inferred $\kappa _ { \perp } ( n , T )$ should be interpreted within the assumptions of the reduced 1D model and mainly in the region of the $( T , n )$ domain covered by data. In this work, the uncertainty of the inferred conductivity was quantified through bootstrap resampling of the training data (section 4.2 and figure 10). A more thorough treatment of this uncertainty could be obtained within a Bayesian inference formalism, which would in addition provide a natural tool for experimental design: given an approximate prior estimate of $\kappa _ { \perp } ( n , T )$ , either from a previous inference or from a limited set of experimental values, the resulting posterior uncertainty map – analogous to that shown in figure 10 – could be used to identify the $( T , n )$ regions where new measurements would most effectively reduce the uncertainty, thereby helping to inform the design of future diagnostic campaigns. Future work should therefore explore this Bayesian extension, apply the method to broader experimental datasets, and refine the treatment of geometry and parallel losses to extend its applicability to larger devices such as W7-X and LHD.

## 7 Acknowledgements

This work was partially funded by the MINERIA project (Grant No. PID2024-157169OB-I00) and the INEXTELA project (Grant No. PID2025-169524OB-I00). Computational resources were provided by the Extremadura Research Centre for Advanced Technologies (CETA-CIEMAT), funded by the European Regional Development Fund (ERDF). CETA-CIEMAT belongs to CIEMAT and the Government of Spain. The author gratefully acknowledges the Harvard John A. Paulson School of Engineering and Applied Sciences for hosting a two-month research stay that contributed to the development of this work.

## References

[1] Boukaye Boubacar Traore, Bernard Kamsu-Foguem, and Fana Tangara. “Deep convolution neural network for image recognition”. In: Ecological informatics 48 (2018), pp. 257–268.

[2] Vijendra Singh, Vijayan K Asari, and Rajkumar Rajasekaran. “A deep neural network for early detection and prediction of chronic kidney disease”. In: Diagnostics 12.1 (2022), p. 116.

[3] Xudong Fan et al. “Machine learning based water pipe failure prediction: The effects of engineering, geology, climate and socio-economic factors”. In: Reliability Engineering & System Safety 219 (2022), p. 108185.

[4] Ms Aayushi Bansal, Dr Rewa Sharma, and Dr Mamta Kathuria. “A systematic review on data scarcity problem in deep learning: solution and applications”. In: ACM Computing Surveys (Csur) 54.10s (2022), pp. 1–29.

[5] Maziar Raissi, Paris Perdikaris, and George E Karniadakis. “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations”. In: Journal of Computational physics 378 (2019), pp. 686–707.

[6] Zaharaddeen Karami Lawal et al. “Physicsinformed neural network (PINN) evolution and beyond: A systematic literature review and bibliometric analysis”. In: Big Data and Cognitive Computing 6.4 (2022), p. 140.

[7] Yuniel Martinez et al. “Physics-informed neural networks for the structural analysis and monitoring of railway bridges: A systematic review”. In: Mathematics 13.10 (2025), p. 1571.

[8] Tommaso Botarelli et al. “Using Physics-Informed neural networks for solving Navier-Stokes equations in fluid dynamic complex scenarios”. In: Engineering Applications of Artificial Intelligence 148 (2025), p. 110347.

[9] Darioush Jalili et al. “Physics-informed neural networks for heat transfer prediction in twophase flows”. In: International Journal of Heat and Mass Transfer 221 (2024), p. 125089.

[10] Raphaël Pellegrin et al. “Transfer learning with physics-informed neural networks for efficient simulation of branched flows”. In: arXiv preprint arXiv:2211.00214 (2022).

[11] B Ph van Milligen, V Tribaldos, and JA Jiménez. “Neural network differential equation and plasma equilibrium solver”. In: Physical review letters 75.20 (1995), p. 3594.

[12] Eva Jaillon and Pavlos Protopapas. “Physics Informed Neural Networks for Magnetohydrodynamic Equations”. In: AI {\&} PDE: ICLR 2026 Workshop on AI and Partial Differential Equations. 2026.

[13] Ivan Depina et al. “Application of physicsinformed neural networks to inverse problems in unsaturated groundwater flow”. In: Georisk: Assessment and Management of Risk for Engineered Systems and Geohazards 16.1 (2022), pp. 21–36.

[14] Wenjing Li and Kok-Meng Lee. “Physics informed neural network for parameter identification and boundary force estimation of compliant and biomechanical systems”. In: International Journal of Intelligent Robotics and Applications 5.3 (2021), pp. 313–325.

[15] Alexandre M Tartakovsky et al. “Physicsinformed deep neural networks for learning parameters and constitutive relationships in subsurface flow problems”. In: Water Resources Research 56.5 (2020), e2019WR026731.

[16] Cedric Flamant, Pavlos Protopapas, and David Sondak. “Solving differential equations using neural network solution bundles”. In: arXiv preprint arXiv:2006.14372 (2020).

[17] Luca Gomez Bachar et al. “Evolution of linear matter perturbations with error-bounded bundle physics-informed neural networks”. In: Physical Review D 112.6 (2025), p. 063515.

[18] Augusto T Chantada et al. “Cosmologyinformed neural networks to solve the background dynamics of the Universe”. In: Physical Review D 107.6 (2023), p. 063523.

[19] Augusto T Chantada et al. “Faster Bayesian inference with neural network bundles and new results for f (R) models”. In: Physical Review D 109.12 (2024), p. 123514.

[20] Pedro Tarancón-Álvarez et al. “Efficient PINNs via multi-head unimodular regularization of the solutions space”. In: Communications Physics 8.1 (2025), p. 335.

[21] Isabela M Yepes and Pavlos Protopapas. “Gradient Scaling Effects in Adaptive Spectral PINNs for Stiff Nonlinear ODEs”. In: arXiv preprint arXiv:2605.04502 (2026).

[22] Pavlos Protopapas and Kaylee Vo. “Variational Boosting for Physics-Informed Neural Networks”. In: arXiv preprint arXiv:2607.23940 (2026).

[23] Augusto T Chantada et al. “Exact and approximate error bounds for physics-informed neural networks”. In: arXiv preprint arXiv:2411.13848 (2024).

[24] Lyman Spitzer and Richard Härm. “Transport Phenomena in a Completely Ionized Gas”. In: Phys. Rev. 89 (5 Mar. 1953), pp. 977–981. doi: 10. 1103/PhysRev.89.977. url: https://link. aps.org/doi/10.1103/PhysRev.89.977.

[25] D Carralero et al. “On the role of filaments in perpendicular heat transport at the scrape-off layer”. In: Nuclear Fusion 58.9 (2018), p. 096015.

[26] S-I Itoh and K Itoh. “On scaling laws in scrapeoff-layer plasmas”. In: Plasma Physics and Controlled Fusion 36.11 (1994), pp. 1845–1851.

[27] Thomas Eich et al. “Scaling of the tokamak near the scrape-off layer H-mode power width and implications for ITER”. In: Nuclear fusion 53.9 (2013), p. 093031.

[28] Feiyu Chen et al. “NeuroDiffEq: A Python package for solving differential equations with neural networks”. In: Journal of Open Source Software 5.46 (2020), p. 1931.

[29] Shuheng Liu et al. “Recent Advances of NeuroDiffEq–An Open-Source Library for Physics-Informed Neural Networks”. In: arXiv preprint arXiv:2502.12177 (2025).

[30] Y Feng and W7-X-team. “Review of magnetic islands from the divertor perspective and a simplified heat transport model for the island divertor”. In: Plasma Physics and Controlled Fusion 64.12 (2022), p. 125012.

[31] D Brunner et al. “An assessment of ion temperature measurements in the boundary of the Alcator C-Mod tokamak and implications for ion fluid heat flux limiters”. In: Plasma Physics and Controlled Fusion 55.9 (2013), p. 095010.

[32] SI Braginskii. “Transport processes in a plasma”. In: Reviews of plasma physics 1 (1965), p. 205.

[33] Peter Stangeby. “The plasma boundary of magnetic fusion devices”. In: Series in Plasma Physics (2000).

[34] P Helander. “Classical and neoclassical transport in tokamaks”. In: Fusion Science and Technology 61.2T (2012), pp. 133–141.

[35] David Bohm. “The characteristics of electrical discharges in magnetic fields”. In: Qualitative Description of the Arc Plasma in a Magnetic Field (1949).

[36] G Manfredi and Maurizio Ottaviani. “Gyro-Bohm scaling of ion thermal transport from global numerical simulations of iontemperature-gradient-driven turbulence”. In: Physical review letters 79.21 (1997), p. 4190.

[37] H Yamada et al. “Characterization of energy confinement in net-current free plasmas using the extended International Stellarator Database”. In: Nuclear Fusion 45.12 (2005), pp. 1684–1693.

[38] Adam Paszke et al. “Pytorch: An imperative style, high-performance deep learning library”. In: Advances in neural information processing systems 32 (2019).

[39] Kaiming He et al. “Deep residual learning for image recognition”. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2016, pp. 770–778.

[40] Diederik P Kingma and Jimmy Ba. “Adam: A method for stochastic optimization”. In: arXiv preprint arXiv:1412.6980 (2014).

[41] Shota Deguchi and Mitsuteru Asai. “Dynamic & norm-based weights to normalize imbalance in back-propagated gradients of physicsinformed neural networks”. In: Journal of Physics Communications 7.7 (2023), p. 075005.

[42] Pauli Virtanen et al. “SciPy 1.0: fundamental algorithms for scientific computing in Python”. In: Nature methods 17.3 (2020), pp. 261–272.

[43] P Ivanova et al. “Characterization of the TJ-II stellarator plasma by means of reciprocating Langmuir probes”. In: Journal of Physics: Conference Series. Vol. 2710. 1. IOP Publishing. 2024, p. 012032.

[44] Bradley Efron. “Bootstrap methods: another look at the jackknife”. In: Breakthroughs in statistics: Methodology and distribution. Springer, 1992, pp. 569–593.

[45] Carlos Alejaldre et al. “TJ-II project: a flexible heliac stellarator”. In: Fusion Technology 17.1 (1990), pp. 131–139.

[46] B Branas et al. “Atomic beam diagnostics for characterization of edge plasma in TJ-II stellarator”. In: Review of Scientific Instruments 72.1 (2001), pp. 602–606.

[47] A Hidalgo et al. “Testing of the collisionalradiative model by laser induced perturbation of a He beam in TJ-II plasmas”. In: PLASMA-2005: International Conference on Research and Applications of Plasmas combined with the 3. German-Polish Conference on Plasma Diagnostics for Fusion and Applications and the 5. French-Polish Seminar on Thermal Plasma in Space and Laboratory. Book of Abstracts. INIS-PL–2006-0005. 2005, pp. 57– 57.

[48] D López-Bruna, Tsv Popov, and E de la Cal. “Monte Carlo estimates of edge particle sources in TJ-II plasmas”. In: Journal of Physics: Conference Series. Vol. 700. 1. IOP Publishing. 2016, p. 012006.

[49] David Bold et al. “Impact of spatially varying transport coefficients in EMC3-Eirene simulations of W7-X and assessment of drifts”. In: Nuclear Fusion 64.12 (2024), p. 126055.

[50] Carsten Killer et al. “Turbulent transport in the scrape-off layer of Wendelstein 7-X”. In: Nuclear Fusion 61.9 (2021), p. 096038.