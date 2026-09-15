# Physics Informed Neural Network model for the dynamical study of Abdominal Aortic Aneurysm

Adrián Robles Arques, Martín Ruiz Fernandez, Javier Sanchis, Miguel A. Teruel, Juan Trujillo

Lucentia Research; Instituto Universitario de Investigación en Informática, Universidad de Alicante, Carr. San Vicent del Raspeig, Alicante, 03690, Comunidad Valenciana, España

## Abstract

We present the development and application of a three-dimensional Physics-Informed Neural Network (PINN) framework for the investigation of haemodynamic behaviour in the human aorta. The model incorporates a timeresolved simulation of pulsatile blood flow over a two-minute interval, enabling the extraction of pressure and velocity fields with high temporal fidelity. The mechanical stress exerted on the aortic wall was quantified through Laplace’s law, with temporal averaging applied to derive representative stress distributions. This approach circumvents the computational overhead associated with conventional computational fluid dynamics (CFD) methods by eliminating mesh generation and exploiting the automatic differentiation capabilities inherent to neural networks. The proposed methodology demonstrates that PINNs can serve as an eficient and accurate alternative for modelling complex vascular flow phenomena, ofering significant advantages in scalability and computational cost reduction while maintaining physical consistency.

## 1. Introduction

The simulation of fluid dynamics constitutes a central pillar in both scientific research and engineering practice. Accurate modelling of blood flow, aerodynamics, and industrial transport processes has enabled significant advances in medicine, aerospace, and energy systems. At the heart of these simulations lies the solution of partial diferential equations (PDEs), which govern the conservation of mass, momentum, and energy in fluid dynamics. Classical numerical approaches, such as finite element and finite volume methods, have long provided robust frameworks for solving these equations. Their success is evident in the widespread adoption of computational fluid dynamics (CFD) across multiple disciplines, including haemodynamic simulation. [1, 2]

Despite their utility, conventional CFD methods encounter notable limitations when applied to complex flow regimes. High-fidelity simulations often demand extremely fine mesh resolution, leading to high computational costs [3]. Furthermore, intricate boundary conditions, non-linearities, and stif problems challenge the stability and convergence of traditional solvers. [4, 5] These constraints are particularly pronounced in biomedical applications, where pulsatile flows, patient-specific geometries, and multi-scale interactions must be captured with precision. As a result, there is a growing need for alternative methodologies that can balance accuracy, eficiency, and scalability. [6]

Recent advances in machine learning, and particularly in deep learning, have introduced new paradigms for fluid dynamics simulation. [7] Among these, physics-informed neural networks (PINNs) have emerged as a promising framework. [8, 9] PINNs embed the governing PDEs directly into the loss function of a neural network, thereby enforcing physical consistency while leveraging the flexibility of deep learning. This approach eliminates the need for mesh generation, exploits automatic diferentiation for gradient computation, and ofers a scalable solution for high dimensionality problems. In the context of fluid dynamics, PINNs have demonstrated the ability to capture complex flow behaviours, including pulsatile and turbulent regimes, with reduced computational overhead compared to conventional CFD. [10]

The integration of PINNs into fluid dynamics research represents a significant step toward bridging data-driven and physics-based modelling. By combining the rigour of physical laws with the adaptability of neural networks, PINNs provide a computationally eficient and accurate alternative for simulating flows in challenging scenarios. [11] Their potential impact spans multiple domains, from biomedical engineering, where they can aid in the study of vascular haemodynamics, to industrial applications requiring real-time flow prediction. This study builds upon these developments by implementing a three-dimensional PINN framework for the analysis of blood flow in the human aorta, highlighting its advantages over traditional CFD approaches in terms of eficiency, scalability, and physical fidelity. [12, 13]

The study of blood flow within the human aorta is of great relevance in the evaluation of Abdominal Aortic Aneurysms (AAAs) and its associated comorbidity and mortality. AAAs are localized dilatations of the abdominal aorta characterized by progressive structural deterioration of the vascular wall, which predisposes the vessel to further enlargement and, ultimately, rupture. [14] The risk of rupture is strongly correlated with aneurysm diameter and growth rate, and rupture events are associated with exceptionally high mortality [15]. Epidemiological data indicate that AAAs represent one of the leading causes of death in individuals over 55 years of age in both Europe and the United States of America, ranking between the twelfth and fifteenth most common causes of mortality in this demographic.[16]

Consequently, accurate modelling of aortic haemodynamics is essential for improving risk stratification, supporting clinical decision-making, and informing therapeutic strategies aimed at reducing aneurysm-related mortality. In this context, the primary objective of the present work is to develop a model capable of predicting the pressure exerted by the internal blood flow on the wall of the abdominal aorta. To this end, the PINN framework is used to estimate both velocity and pressure fields throughout the domain, including interior points, which are necessary for reconstructing the full haemodynamic flow pattern.

The remainder of this article is structured as follows: Section 2 provides a review of related work on simpler and similar geometries, Section 3 details the procedure for the present work, including mathematical background, and describing the dataset used, technical implementation of the model and the validation method. Section 4 presents the results obtained from the trained PINN, supported by visualisations of the predicted haemodynamic fields. Section 5 discusses these findings in relation to existing literature, examines the validation metrics, and outlines the main limitations of the approach. Finally, Section 6 shows the main limitations of the current model and show potential improvements, and finally Section 7 summarises the conclusions of the study and identifies potential directions for future research.

## 2. Related Works

This section includes, structured in several subsections, an introduction to the governing fluid-dynamics equations and their treatment within the model, a review of previous PINN-based studies on simplified geometries.

## 2.1. Previous works on simpler geometries

Building on this theoretical foundation, numerous studies have explored how the PINN framework performs when applied to fluid-dynamic problems in simplified geometries. These lower-dimensional settings provide an ideal environment for evaluating training strategies, loss-balancing techniques, and the overall ability of PINNs to reproduce solutions of the Navier–Stokes equations without relying on mesh-based discretization. Early demonstrations, such as the work of Raissi et al. [8], showed that PINNs can accurately recover solutions to nonlinear PDEs in one and two dimensions, establishing a baseline for subsequent investigations into more specialized flow configurations.

Several recent contributions have focused specifically on 2D flow scenarios, using them as controlled testbeds to assess the strengths and limitations of the method. Botarelli et al. [12] applied PINNs to two-dimensional Navier–Stokes problems in complex but still planar geometries, illustrating how the mesh-free formulation facilitates the treatment of irregular boundaries and heterogeneous flow conditions. Their results highlight the flexibility of the approach in settings where traditional CFD methods require careful meshing or stabilization.

Complementing this, Wong et al. [17] introduced a multi-case training strategy in which a single PINN is exposed to multiple tube-flow configurations during training. This approach significantly improves generalization across diferent 2D geometries and reduces the computational cost associated with training separate models for each case, demonstrating how shared physical structure can be leveraged within the PINN framework.

In the context of haemodynamics, simplified arterial geometries have also served as an important proving ground for PINN-based modelling. While classical CFD studies, such as Kabir et al. [18], provide detailed simulations of pulsatile blood flow in normal and stenosed arteries, more recent PINNoriented work has sought to integrate physical constraints and data within a unified optimization process. Liu et al. [19] proposed a variable-separated PINN architecture equipped with adaptive loss weighting to address the imbalance between PDE residuals and boundary-condition terms, a common challenge in blood-flow simulations. Their results demonstrate improved convergence and accuracy in simplified flow domains, underscoring the importance of carefully designed loss-balancing strategies when applying PINNs to physiological problems.

Taken together, these studies show that simple 1D and 2D geometries play a crucial role in the development and refinement of PINN methodologies for fluid dynamics. By first validating the approach in controlled settings, it is possible to systematically identify issues such as stifness in the loss landscape, sensitivity to boundary-condition enforcement, or slow convergence. The insights gained from these lower-dimensional experiments form the basis for extending PINNs to more demanding three-dimensional and time-resolved cardiovascular applications, which will be discussed in the following section.

## 2.2. Previous works on 3D and 4D geometries

The use of physics-informed neural networks for fully three-dimensional and time-resolved (4D) flow problems has grown substantially in recent years, particularly in the context of cardiovascular modelling and medical imaging.

One of the earliest applications in this direction was presented by Fathi et al. [20], who developed a physics-informed deep learning framework for super–resolution and denoising of 4D-Flow MRI data.

By embedding the Navier-Stokes equations into the reconstruction process, their method enhances noisy or low–resolution velocity fields while preserving physical consistency, demonstrating the potential of PINNs as physics–aware post–processing tools for clinical imaging.

Another important line of research focuses on using PINNs to infer unmeasured haemodynamic quantities from sparse or incomplete 4D flow data. Kissas et al. [21] showed that PINNs can reconstruct arterial pressure fields from non-invasive 4D-Flow MRI velocity measurements by enforcing the incompressible Navier-Stokes equations as soft constraints during training. Their work illustrates how PINNs can recover clinically relevant but unobservable quantities, such as pressure gradients, without requiring full boundary condition specification or mesh generation, ofering an alternative to traditional CFD pipelines.

Beyond data assimilation, several studies have explored PINNs as direct solvers for three-dimensional Navier-Stokes problems. Alzhanov et al. [22] proposed a fully 3D PINN framework for simulating blood flow in patientspecific coronary artery trees, integrating imaging data with physics-based constraints to estimate fractional flow reserve (FFR). Their results show good agreement with CFD simulations and invasive measurements.

Similarly, Heger et al. [23] investigated the use of physics-informed deep learning to predict parametric 3D flow fields from boundary data, demonstrating that PINNs can serve as eficient surrogates for families of flow solutions parametrized by geometry or boundary conditions.

Methodological advances have also contributed to improving the robustness of PINNs in high-dimensional flow settings. Liu et al. [19] introduced a variable-separated PINN architecture with adaptive loss weighting to address the imbalance between PDE residuals and boundary constraints in blood-flow simulations. Their approach improves convergence and accuracy in complex flow regimes, highlighting the importance of architectural and optimization strategies when extending PINNs to 3D and 4D domains. Afterwards, in [24] the same team extends the Integral Conservation Physics-Informed Neural Networks (ICPINNs) framework to transient blood-flow simulations in patient-specific thoracic aortas, incorporating the integral form of the Navier–Stokes equations and Monte Carlo integration to enhance physical fidelity. The authors conduct the first systematic comparison of multiple neural network architectures.

In [25] Cruz-Gonzalez et al. present a comparative study of classical PINNs, Deep Operator Networks (DeepONets), and their physics-informed variants (PI-DeepONets) for simulating blood flow in an idealised 3D abdominal aortic aneurysm model. The authors embed the steady Navier–Stokes equations into the learning process and systematically assess accuracy and computational eficiency against high-fidelity CFD benchmarks. While in [26] Zhang et al. propose a 4D haemodynamic prediction framework that integrates CFD-generated datasets with deep learning models based on point-cloud representations and physics-informed neural networks. The study constructs two large 4D vascular datasets—one including fine coronary branches and another focused on the abdominal aorta—and evaluates multiple architectures, including PointNet, PointNet++, and PINN-enhanced variants, to determine the optimal framework for diferent vascular morphologies.

Overall, these studies demonstrate that PINNs have evolved into a versatile tool for 3D and 4D fluid-dynamic modelling, capable of performing simulation, data assimilation, super-resolution, and surrogate modelling within a unified framework. They also emphasize the key challenges, including training stability, loss balancing, and scalability; that motivate ongoing research and inform the design of the present work.

## 2.3. PINNs for Abdominal Aortic Aneurysm Modelling

In recent years, physics-informed neural networks have increasingly been applied to the study of abdominal aortic aneurysms (AAAs), where accurate haemodynamic modelling is essential for understanding aneurysm progression and rupture risk. Compared to simplified vascular domains, AAA geometries present additional challenges due to their complex morphology, pulsatile flow conditions, and the presence of highly heterogeneous velocity and pressure distributions. These characteristics make AAAs a particularly suitable benchmark for evaluating the capabilities and limitations of PINNbased haemodynamic solvers.

Recent studies have explored the use of PINNs for simulating blood flow in aneurysmal geometries and reconstructing clinically relevant haemodynamic quantities. Cruz-González et al. [27] presented a comparative framework combining classical PINNs, Deep Operator Networks (DeepONets), and physics-informed DeepONet variants for the simulation of blood flow in idealized three-dimensional abdominal aortic aneurysm geometries. Their work demonstrated that operator-learning approaches can improve computational eficiency while preserving the physical consistency imposed by the Navier-Stokes equations, highlighting the potential of hybrid architectures for vascular-flow modelling.

Beyond purely fluid-dynamic simulations, other works have investigated the interaction between blood flow and the arterial wall developing a coupled fluid-structure interaction (FSI) framework, integrating PINNs with conventional computational fluid dynamics methods to analyse pulsatile flow in arterial aneurysms [28]. Their results showed that incorporating vessel-wall deformation provides a more realistic representation of aneurysmal haemodynamics and may improve the assessment of biomechanical factors associated with aneurysm growth and rupture.

Related eforts have also focused on improving the prediction of complex haemodynamic patterns in patient-specific vascular geometries such as a fourdimensional haemodynamic prediction framework combining CFD-generated datasets with physics-informed neural networks and point-cloud representations [29]. By analysing diferent vascular morphologies, including abdominal aortic configurations, the study demonstrated that PINN-enhanced models can capture temporally resolved flow dynamics while reducing the computational cost associated with traditional CFD simulations.

Similarly, another investigation shows the combined use of CFD and PINNs for rupture-risk prediction in thoracoabdominal aneurysms through fluid-structure interaction analysis [30]. Their findings suggest that physicsinformed learning frameworks may contribute to improved biomechanical indicators for aneurysm assessment, particularly when integrated with patientspecific anatomical and flow information.

Despite these advances, several studies report persistent limitations related to training stability, convergence behaviour, sensitivity to hyperparameter selection, and scalability in complex or turbulent flow regimes [31, 32]. Although PINNs reduce the dependence on mesh generation and large simulation datasets, their computational cost during training may remain substantial, and they do not consistently outperform established CFD methods [12]. These challenges continue to motivate research into improved architectures, adaptive training strategies, and more robust optimization methods for large-scale haemodynamic applications.

## 2.4. Contributions

The present study advances the state of research on Physics-Informed Neural Networks by demonstrating their applicability to fully 3D haemodynamic simulations in the abdominal aorta. While previous work has largely focused on simplified geometries, steady-state conditions, or idealized flow regimes, and there is not yet a wide consolidated literature on more complex scenarios such as in aortic aneurysms, our model shows that PINNs can successfully reconstruct pressure and velocity fields in anatomically realistic vascular complex domains without relying on mesh-based discretization. This contributes to the growing evidence that PINNs can serve as a computationally eficient alternative to conventional CFD, particularly in scenarios where boundary conditions are uncertain or clinical data are sparse.

A second key contribution lies in the systematic evaluation of the model’s physical consistency through continuity and momentum residuals, providing quantitative evidence of the reliability of the PINN-based predictions. By comparing these metrics with those reported in the literature for both simplified and complex vascular geometries, the study positions its results within the broader research landscape and demonstrates that the proposed framework achieves error levels comparable to, or in some cases lower than, existing PINN-based haemodynamic models. This reinforces the feasibility of using PINNs for clinically relevant flow estimation tasks, even in the absence of a CFD-derived ground truth.

Finally, the study identifies and articulates several methodological pathways for future development, including the incorporation of non-Newtonian fluid modelling, improved generalisation across diverse vascular morphologies, and the exploration of more advanced PINN architectures such as CNN-PINNs or GCN-PINNs. By outlining these directions, the work not only situates itself within the current trajectory of PINN research but also provides a roadmap for enhancing the accuracy, robustness, and clinical applicability of physics-informed deep learning in cardiovascular biomechanics.

## 3. Methodology

The current section contains an extensive explanation of the theoretical background and the technical implementation needed for the current work. The section begins with mathematical explanation in 3.1, where the core concepts of fluid dynamics have been analysed, followed in Sec. 3.2 by a review of the original dataset used for the present work, then in Sec. 3.3 the technical architecture and implementations of the PINN are reviewed, and lastly in Sec. 3.4 the model validation method is presented.

## 3.1. Mathematical background

Beginning with the mathematical background needed to understand the physical principles and constraints of fluid dynamics, this sections has been divided into two separate subsections. First, the core concepts and deduction of the Navier-Stokes equations shall be reviewed in Subsect. 3.1.1, continuing in Subsect. 3.1.2 with the expansion of the 3D time dependent Navier-Stokes equations and deduction of the residual therms that will be used in the PINN composed loss function.

## 3.1.1. Introducing fluid dynamics into PINNs

The foundation of fluid dynamics lies in the system of PDEs that explain the motion of viscous fluids, namely the Navier-Stokes equations [33]. These equations, derived from the principles of conservation of mass and momentum and provide the mathematical framework for describing a wide range of flow phenomena, from laminar to turbulent regimes. As such, they constitute the cornerstone of both theoretical analysis and computational modelling in fluid mechanics.

The Navier–Stokes equations can be derived as a particular case of the Cauchy momentum equation. By expressing the Cauchy stress tensor as the sum of an isotropic pressure term and a viscous stress term, the governing equations reduce to the convective form of the momentum balance for a Newtonian fluid. This formulation provides the foundation from which the Navier-Stokes equations are obtained, linking the general principles of continuum mechanics to the specific description of viscous fluid motion.

$$
\rho { \frac { \mathrm { D } { \pmb u } } { \mathrm { D } t } } = - \nabla p + \nabla \cdot { \pmb \tau } + \rho { \pmb a } ,\tag{1}
$$

where $\textstyle { \frac { \mathrm { D } } { \mathrm { D } t } }$ is the material derivative, defined as $\begin{array} { r } { \frac { \partial } { \partial t } + \pmb { u } \cdot \nabla , \rho } \end{array}$ is the fluid density, and $p$ is the pressure, and a represents all external acceleration acting over the fluid.

From this point onward, several assumptions are introduced to derive the specific form of the Navier–Stokes equations required for this problem:

1. Blood is treated as a Newtonian fluid (constant viscosity).

2. The fluid density is considered constant (incompressible).

3. The flow takes into account voticity and turbulence due to blood’s typical Reynolds number.

Under these assumptions, the Navier–Stokes equations for an incompressible laminar flow can be derived from the Cauchy momentum equation (1), yielding the conservation of momentum and the conservation of mass [34]:

The momentum conservation equation for each component can be expressed as follows:

$$
\frac { \partial u _ { i } } { \partial t } + \sum _ { j } u _ { j } \frac { \partial u _ { i } } { \partial x _ { j } } = - \frac { 1 } { \rho } \frac { \partial p } { \partial x _ { i } } + \frac { 1 } { \rho } \sum _ { i } \frac { \partial \tau _ { i j } } { \partial x _ { j } } ,\tag{2}
$$

where $u _ { i }$ is the $i ^ { t h }$ component of the velocity vector, $x _ { i }$ is the $i ^ { t h }$ component of the coordinate system. Lastly, $\tau _ { i j }$ represents the viscous stress tensor, whose expression for an incompressible Newtonian fluid is:

$$
\tau _ { i j } = \mu \left( { \frac { \partial u _ { i } } { \partial x _ { j } } } + { \frac { \partial u _ { j } } { \partial x _ { i } } } \right) \quad \Rightarrow \quad \sum _ { j } { \frac { \partial \tau _ { i j } } { \partial x _ { j } } } = \mu \sum _ { j } { \frac { \partial ^ { 2 } u _ { i } } { \partial x _ { j } ^ { 2 } } } .\tag{3}
$$

Meanwhile, the mass conservation or continuity equation is more simply expressed as:

$$
\nabla \cdot { \pmb u } \Rightarrow \frac { \partial u _ { i } } { \partial x _ { i } } = 0 .\tag{4}
$$

Taking both eq. 2 and eq. 4 as the governing equations for a particle system in an incompressible Newtonian laminar fluid, the Physics-Informed Neural Network framework incorporates these equations directly into the loss function. The loss terms are formulated so that each governing equation contributes explicitly to the optimization problem solved during backpropagation, where automatic diferentiation is used to compute the required derivatives [35].

A PINN augments the traditional data-driven loss function with additional terms that enforce the governing physical laws. These terms are evaluated at a set of collocation points distributed throughout the spatiotemporal domain, where the neural network predictions are required to satisfy the momentum and continuity equations. Each residual corresponding to the deviation from the exact PDE solution is incorporated into the loss function, ensuring that the optimization process penalizes physically inconsistent predictions [8].

During backpropagation, automatic diferentiation is used to compute all spatial and temporal derivatives appearing in the PDE residuals. This eliminates the need for numerical discretization and allows the network to learn a solution that is continuously diferentiable across the domain. As a result, the optimization algorithm simultaneously minimizes the data mismatch and the physics-based residuals, guiding the network toward solutions that adhere to both the observed measurements and the underlying fluid dynamics. This unified treatment of data and physics is one of the key advantages of PINNs, enabling them to generalize well even in regimes where training data are sparse or noisy.

To complete the physical constraints, the boundary conditions of the geometry must also be incorporated into the loss function, ensuring that the neural network satisfies not only the governing equations within the domain but also the prescribed behaviour at the inlet, outlet, and vessel walls. These terms enforce conditions such as imposed velocity profiles, pressure values, or no-slip constraints, depending on the specific configuration of the problem. By embedding both the interior physics and the boundary information into a unified optimization framework, the PINN is guided toward solutions that remain fully consistent with the physical requirements of the system.

Therefore, the final loss function is composed as follows:

$$
L = L _ { \mathrm { m } } + L _ { \mathrm { c } } + L _ { \mathrm { b c } } + L _ { \mathrm { i c } } ,\tag{5}
$$

where $L _ { \mathrm { m } }$ is the momentum equation, the $L _ { \mathrm { c } }$ corresponds to the continuity equation, and the $L _ { \mathrm { b c } }$ and $L _ { \mathrm { i c } }$ add the boundary and initial conditions. All terms should be properly arranged so that they converge to zero individually.

## 3.1.2. Developing 4D fluid equations

For a 3D-spatial non static case of an incompressible flow, the momentum equation (2) can be expanded as follows:

$$
\left\{ \begin{array} { l } { \frac { \partial u _ { x } } { \partial t } + u _ { x } \frac { \partial u _ { x } } { \partial x } + u _ { y } \frac { \partial u _ { x } } { \partial y } + u _ { z } \frac { \partial u _ { x } } { \partial z } = - \frac { 1 } { \rho } \frac { \partial p } { \partial x } + \frac { \mu } { \rho } \left( \frac { \partial ^ { 2 } u _ { x } } { \partial x ^ { 2 } } + \frac { \partial ^ { 2 } u _ { x } } { \partial y ^ { 2 } } + \frac { \partial ^ { 2 } u _ { x } } { \partial z ^ { 2 } } \right) , } \\ { \frac { \partial u _ { y } } { \partial t } + u _ { x } \frac { \partial u _ { y } } { \partial x } + u _ { y } \frac { \partial u _ { y } } { \partial y } + u _ { z } \frac { \partial u _ { y } } { \partial z } = - \frac { 1 } { \rho } \frac { \partial p } { \partial y } + \frac { \mu } { \rho } \left( \frac { \partial ^ { 2 } u _ { y } } { \partial x ^ { 2 } } + \frac { \partial ^ { 2 } u _ { y } } { \partial y ^ { 2 } } + \frac { \partial ^ { 2 } u _ { y } } { \partial z ^ { 2 } } \right) , } \\ { \frac { \partial u _ { z } } { \partial t } + u _ { x } \frac { \partial u _ { z } } { \partial x } + u _ { y } \frac { \partial u _ { z } } { \partial y } + u _ { z } \frac { \partial u _ { z } } { \partial z } = - \frac { 1 } { \rho } \frac { \partial p } { \partial z } + \frac { \mu } { \rho } \left( \frac { \partial ^ { 2 } u _ { z } } { \partial x ^ { 2 } } + \frac { \partial ^ { 2 } u _ { z } } { \partial y ^ { 2 } } + \frac { \partial ^ { 2 } u _ { z } } { \partial z ^ { 2 } } \right) , } \end{array} \right.\tag{6}
$$

and the continuity equation can be expanded as:

$$
\frac { \partial u _ { x } } { \partial x } + \frac { \partial u _ { y } } { \partial y } + \frac { \partial u _ { z } } { \partial z } = 0 .\tag{7}
$$

After expanding the Navier–Stokes equations in three spatial dimensions, it is convenient to rewrite the system in compact vector form. Let ${ \bf u } ( x , y , z , t ) = ( u _ { x } , u _ { y } , u _ { z } )$ denote the velocity field and $p ( x , y , z , t )$ the pressure field, defined on a spatial domain $\Omega \subset \mathbb { R } ^ { 3 }$ and a time interval $t \in [ 0 , T ]$ For an incompressible Newtonian fluid with constant density $\rho$ and dynamic viscosity $\mu ,$ the governing equations can be written as:

$$
\frac { \partial { \bf { u } } } { \partial t } + ( { \bf { u } } \cdot \nabla ) { \bf { u } } = - \frac { 1 } { \rho } \nabla p + \nu \nabla ^ { 2 } { \bf { u } } , \qquad \nabla \cdot { \bf { u } } = 0 ,\tag{8}
$$

where $\nu = \mu / \rho$ is the kinematic viscosity.

In the four–dimensional spatio–temporal setting considered here, the PINN takes as input the coordinates $( x , y , z , t )$ and outputs the corresponding flow variables:

$$
( x , y , z , t ) \ \mapsto \ \big ( u _ { x } ( x , y , z , t ) , \ u _ { y } ( x , y , z , t ) , \ u _ { z } ( x , y , z , t ) , \ p ( x , y , z , t ) \big ) .\tag{9}
$$

From these outputs, the residuals of the momentum and continuity equations are constructed at each collocation point. For the three momentum components, the residuals are defined as:

$$
\left\{ \begin{array} { l l } { R _ { m , x } = \displaystyle \frac { \partial u _ { x } } { \partial t } + u _ { x } \frac { \partial u _ { x } } { \partial x } + u _ { y } \frac { \partial u _ { x } } { \partial y } + u _ { z } \frac { \partial u _ { x } } { \partial z } + \frac { 1 } { \rho } \frac { \partial p } { \partial x } - \nu \nabla ^ { 2 } u _ { x } , } \\ { R _ { m , y } = \displaystyle \frac { \partial u _ { y } } { \partial t } + u _ { x } \frac { \partial u _ { y } } { \partial x } + u _ { y } \frac { \partial u _ { y } } { \partial y } + u _ { z } \frac { \partial u _ { y } } { \partial z } + \frac { 1 } { \rho } \frac { \partial p } { \partial y } - \nu \nabla ^ { 2 } u _ { y } , } \\ { R _ { m , z } = \displaystyle \frac { \partial u _ { z } } { \partial t } + u _ { x } \frac { \partial u _ { z } } { \partial x } + u _ { y } \frac { \partial u _ { z } } { \partial y } + u _ { z } \frac { \partial u _ { z } } { \partial z } + \frac { 1 } { \rho } \frac { \partial p } { \partial z } - \nu \nabla ^ { 2 } u _ { z } , } \end{array} \right.\tag{10}
$$

and the residual of the continuity equation enforcing the incompressibility of the model is expressed as:

$$
R _ { c } = \frac { \partial u _ { x } } { \partial x } + \frac { \partial u _ { y } } { \partial y } + \frac { \partial u _ { z } } { \partial z } .\tag{11}
$$

All the previously expanded residuals are evaluated at a set of interior collocation points $\{ ( x _ { f } ^ { i } , y _ { f } ^ { i } , z _ { f } ^ { i } , t _ { f } ^ { i } ) \} _ { i = 1 } ^ { N _ { f } }$ where the corresponding physics–based loss terms are defined as:

$$
L _ { \mathrm { m } } = \frac { 1 } { N _ { f } } \sum _ { i = 1 } ^ { N _ { f } } \left( R _ { m , x } ^ { ( i ) 2 } + R _ { m , y } ^ { ( i ) 2 } + R _ { m , z } ^ { ( i ) 2 } \right) , \qquad L _ { \mathrm { c } } = \frac { 1 } { N _ { f } } \sum _ { i = 1 } ^ { N _ { f } } R _ { c } ^ { ( i ) 2 } .\tag{12}
$$

To close the problem, suitable initial and boundary conditions are imposed. At the inlet, a pulsatile pressure condition is prescribed as:

$$
P _ { \mathrm { i n } } ( t ) = P _ { \mathrm { m i n } } + P _ { \Delta } \sin ^ { 2 } \left( 2 \pi f ^ { \prime } t \right) ,\tag{13}
$$

where $P _ { \Delta } = P _ { \mathrm { m a x } } - P _ { \mathrm { m i n } }$ (so that it is a continuous, smooth function and in the range of $[ P _ { \mathrm { m i n } } , P _ { \mathrm { m a x } } ] ) , f ^ { \prime } = f / 2$ and f is the frequency of the pulse. On the other hand, a static pressure $P _ { \mathrm { m i n } }$ is enforced at the outlet. The inlet velocity profile is assumed parabolic, pulsatile and always greater than 0,

$$
u _ { z } ( r , t ) = v _ { \mathrm { r e f } } \cdot \left[ 1 - \left( \frac { r } { R _ { \mathrm { m a x } } } \right) ^ { 2 } \right] \cdot \left[ \left( 1 - v _ { \mathrm { m i n } } \right) \sin ^ { 2 } \left( 2 \pi f ^ { \prime } t \right) + v _ { \mathrm { m i n } } \right] ,\tag{14}
$$

where r is the radial distance from the vessel centerline. A no–slip condition is applied on the vessel walls,

$$
{ \bf u } = { \bf 0 } \quad \mathrm { o n } \ \partial \Omega _ { \mathrm { w a l l } } .\tag{15}
$$

The initial conditions specify the state of the system at $t = 0$ . The inlet pressure is initialized at its minimum value, (diastolic phase) for a soft start,

$$
\begin{array} { r } { P ( x , y , z , 0 ) = P _ { \mathrm { m i n } } \quad \mathrm { o n ~ } \Gamma _ { \mathrm { i n } } , } \end{array}\tag{16}
$$

and the velocity field is initialized as:

$$
u _ { x } ( x , y , z , 0 ) = 0 , \qquad u _ { y } ( x , y , z , 0 ) = 0 , \qquad u _ { z } ( x , y , z , 0 ) = v _ { \mathrm { m i n } } ( x , y , z ) .\tag{17}
$$

The boundary and initial conditions contribute with additional loss terms $L _ { \mathrm { b c } }$ and $L _ { \mathrm { i c } }$ , defined as mean–squared errors between the network predictions and the prescribed values. Extending the stationary loss definition in (5) to the time–dependent three–dimensional case, the total loss becomes:

$$
L = L _ { \mathrm { m } } + L _ { \mathrm { c } } + L _ { \mathrm { b c } } + L _ { \mathrm { i c } } ,\tag{18}
$$

with each contribution arranged to converge to zero during training, ensuring that the PINN satisfies the Navier–Stokes equations, incompressibility, and all imposed physical constraints throughout the full four–dimensional domain.

## 3.2. Dataset

The original data come from computed axial tomography scans processed by the biomedical engineering team at Avamed Synergy, who, as collaborators of the ENIA Research Chair of Artificial Intelligence at the University of Alicante, have provided their datasets for the development of the present PINN model. The biomedical engineering team extracts a 3D object containing the geometry of the abdominal aorta, which is then stored in STL file format.

![](images/64d36cb69ded62d6c06ad4f804fd6eaf4ebe9b7b0821fcb462ce3ea40e95b152.jpg)  
Figure 1: Original STL view in blender

Given that the model relies exclusively on point clouds as input, we extract only the positional data, disregarding all additional mesh information such as connectivity vectors between nodes. This allows us to isolate the points that define the vessel surface, that is, the locations where the prescribed boundary conditions will be applied.

In addition, an automatic detection system for identifying inlet and outlet regions of the flow was implemented, since it is essential to distinguish the areas where specific boundary conditions must be imposed for blood-flow entry and exit. These include the direction, profile, and magnitude of the inlet velocity, constraints on lateral velocity components, and inlet and outlet pressure conditions.

This automatic detection system is based on predefined numerical rules, taking into account that, for the region of interest considered, the inlet and outlet zones consistently lie along the boundaries of the domain. The upper boundary is treated as the inlet region, whereas the lateral and lower boundaries are considered potential outlets. Minimum distance thresholds and additional constraints were also introduced to prevent partial contacts with the domain boundary from being erroneously classified as outlet regions.

![](images/bcdd3a0b0ed99bc5d7ee0c1419628c788522168420f7540b44fde22d5dad1ea5.jpg)

![](images/dbf8dbfee94032c9fbd1e8321486b1408232f81ced2a9c6b9e88f25d570dece1.jpg)  
Figure 2: Points extracted from STL and automatic inlet and outlet detection

## 3.3. PINN implementation

The PINN developed in this work has been implemented using the Python programming language, specifically through the specialized DeepXDE library built on a PyTorch backend. Following the examination of the governing physical equations presented in the previous sections, we now turn to the concrete implementation of the practical case under study.

DeepXDE is a Python library specifically designed to make physics-informed neural networks (PINNs) practical, accessible, and flexible for solving a wide range of diferential equations. Introduced by Lu et al. (2021) [36], it provides a unified framework in which users can formulate forward and inverse problems directly from their mathematical definitions, while the library handles automatic diferentiation, loss construction, and training.

This library supports ordinary, partial, and even stochastic diferential equations, and it accommodates complex geometries through constructive solid geometry. Its design emphasizes compact, mathematically intuitive code, and it incorporates advanced features such as residual-based adaptive refinement to improve training eficiency. Overall, DeepXDE provides a robust and user-friendly environment for applying PINNs to real-world scientific machine-learning problems.

However, the handling of complex geometries is not fully addressed by the library, making it necessary to develop a set of additional components capable of operating on the available mesh data, converting them into point clouds, and enabling their use within DeepXDE without compromising accuracy.

![](images/7d0764d9d97c813b78c9f5a0600580560185f9f791ca10c1148642fb3715f91e.jpg)  
Figure 3: Representation of the inner and boundary points generated to train the model within DeepXDE

As explained in the previous section, the initial data contain only the points corresponding to the aortic lumen, without including any interior points. For this reason, a key step was the development of a geometry class compatible with DeepXDE that could generate interior points while adapting to a complex and irregular vascular shape. Additionally, it was taken into account that the number of boundary points in the STL file could overload the model during training. To address this question, the custom geometry class performs a random selection of boundary points, while always ensuring a predefined proportion of points belonging to the inlet and outlet regions.

For the generation of interior points, random point clouds are created at diferent heights, and a verification step is performed using DBSCAN [37] to determine whether each point lies inside or outside the region delimited by the boundary surface. Points classified as exterior are discarded. If the number of validated interior points is lower than required, the process is repeated until the desired quantity is reached. Figure 3 shows an example of the points generated by the specific class designed in order to properly convert the mesh data to data points needed by DeepXDE.

The fluid-dynamics equations, together with the initial and boundary conditions, were implemented in the loss function following the procedures established by the library. For the physical parameters of blood, standard average values commonly reported in the specialized literature were adopted: density of 1060 $\mathrm { k g / m ^ { 3 } }$ , dynamic viscosity of 0.004 Pa · s, characteristic aortic diameter of 0.02 m, and a maximum inlet velocity of 1 $\mathrm { m / s } .$ Maximum inlet pressure is set at mean systolic pressure in adults, 120 mmHg or 15998 $\mathrm { P a } ,$ and outlet pressure is set at mean diastolic pressure of 80 mmHg or 10665 Pa [38].

![](images/a9927e04f849022cda69f1a578f5357fab84d6da00db27a1714b0b324555fef2.jpg)  
Figure 4: Architecture of a (M,N) net used in a PINN and the composed loss.

The feed-forward neural network (FNN) is structured as a single block of 10 layers with 256 neurons each, according to the architecture shown in Figure 4, following configurations commonly reported in the related literature [22]. Training was performed using a definition of 8000 boundary points, 5,000 initial domain points, 2000 boundary points and 1000 test points, granting 10% of the domain points on the outlets and 5% on the inlet, together with a temporal simulation window of 180 seconds, using a Hammersley time distribution.

## 3.3.1. Problems of the PINN architecture

Standard neural networks are known to sufer from spectral bias, tending to learn low-frequency components faster than high-frequency structures. To mitigate this limitation, Fourier Features based on the work of Tancik et

al. [39] were incorporated into the input representation. The spatial coordinates are projected into a sinusoidal embedding space:

$$
\phi ( \mathbf { x } ) = \left[ \sin ( 2 \pi \mathbf { x } \mathbf { B } ) , \cos ( 2 \pi \mathbf { x } \mathbf { B } ) \right] ,\tag{19}
$$

where B contains randomly initialized frequency matrices with multiple scaling factors. This encoding enables the network to simultaneously capture low, intermediate, and high frequency flow structures.

To improve the ability of the PINN framework to approximate complex solutions of the Navier-Stokes equations in haemodynamic simulations, a custom neural-network architecture was developed combining Fourier Features, parallel specialized branches, residual blocks, and cross-branch interactions (Figure 5).

The main motivation behind this design is to eficiently represent both smooth global flow structures and localized high-frequency phenomena such as sharp gradients, recirculation regions, and transient pulsatile dynamics.

## 3.3.2. Dual-Branch Architecture

The final architecture consists of two parallel branches:

• A physics-enriched branch, receiving the concatenation of physical coordinates and Fourier Features:

$$
\mathbf { g } _ { 0 } = [ \mathbf { x } , \phi ( \mathbf { x } ) ] ,\tag{20}
$$

which focuses on preserving the global physical structure of the solution.

• A spectral branch, operating exclusively on Fourier Features:

$$
\mathbf { h } _ { 0 } = \phi ( \mathbf { x } ) ,\tag{21}
$$

designed to capture localized high-frequency corrections.

## 3.3.3. Residual Learning and Cross-Branch Interaction

Both branches are composed of residual blocks of the form:

$$
\mathbf { y } = \mathbf { x } + c F ( \mathbf { x } ) ,\tag{22}
$$

![](images/7d2a8df8bafe9abb3b3b3f9c8b31638c4e0fc2bac6abe92c692180a8b4ce67d5.jpg)  
Figure 5: Architecture of the new network (Modulated Fourier Network or MFN).

where $F ( \mathbf { x } )$ is a small internal neural network and $c = 0 . 5$ is a damping factor introduced to improve numerical stability. Residual learning facilitates gradient propagation, stabilizes optimization, and enables progressive refinement of the solution [40].

Unlike standard multi-branch architectures, both branches exchange information after each residual block:

$$
\mathbf { g } _ { k + 1 } = \mathbf { g } _ { k } + 0 . 1 \mathbf { P } _ { h  g } ( \mathbf { h } _ { k } ) ,\tag{23}
$$

$$
\mathbf { h } _ { k + 1 } = \mathbf { h } _ { k } + 0 . 1 \mathbf { P } _ { g  h } ( \mathbf { g } _ { k } ) ,\tag{24}
$$

where $\mathbf { P } _ { h \to g }$ and $\mathrm { { \bf P } } _ { g \to h }$ are trainable linear projections. This interaction allows the physical branch to incorporate local spectral corrections while maintaining global consistency.

## 3.3.4. Adaptive Output Modulation

The final prediction combines both branches through adaptive modulation coeficients computed from the spectral representation:

$$
\gamma = \sigma ( \mathbf { W } _ { \gamma } \mathbf { h } + \mathbf { b } _ { \gamma } ) ,
$$

$$
\beta = \mathbf { W } _ { \beta } \mathbf { h } + \mathbf { b } _ { \beta } ,\tag{25}
$$

(26)

leading to the final output:

$$
\mathbf { u } = ( 1 - \gamma ) \odot \mathbf { g } + \gamma \odot \mathbf { h } + \beta .\tag{27}
$$

This formulation enables the network to adaptively determine the contribution of each branch at every spatial location.

Overall, the proposed architecture combines spectral embeddings, residual refinement, and adaptive multi-scale representations to improve convergence, training stability, and the reconstruction of complex haemodynamic flow patterns in three-dimensional vascular geometries.

## 3.4. Model validation

The strategy followed in order to validate the model obtained for the blood flow inside the abdominal aorta consists in calculating the residuals for both, continuity and momentum equations. As these equations represent conservation laws for both mass and momentum, the ideal residuals should be nullified, meaning that the Navier-Stokes equations 6 and 7 are perfectly fulfilled.

The residuals for those equations were also previously defined in equations 10 and 11, as they are introduced in the loss function. After the training, the same expressions are applied to the test domain points to verify how close the model is to the best possible outcome.

As previously stated, from the aforementioned residual equations both physics-based loss terms are defined as equation 12 shows, and consequently common error metrics like squared loss function and infinite or maximum loss function can be defined as well, knowing that in the case of the residuals the expected value is 0. Such error metrics are defined as:

$$
\begin{array} { r l r l } & { \displaystyle | | R _ { c } | | _ { L ^ { 2 } } = \sqrt { L _ { c } } \left( \frac { D _ { r e f } } { u _ { r e f } } \right) } & & { | | R _ { m } | | _ { L ^ { 2 } } = \sqrt { L _ { m } } \left( \frac { D _ { r e f } } { \rho u _ { r e f } ^ { 2 } } \right) } \\ & { \displaystyle | | R _ { c } | | _ { L _ { \infty } } = \operatorname* { m a x } _ { j = 1 , \ldots , N } | R _ { c } ^ { j } | } & & { | | R _ { m } | | _ { L _ { \infty } } = \operatorname* { m a x } _ { j = 1 , \ldots , N } | \mathbf { R } _ { m } ^ { j } | } \end{array}\tag{28}
$$

being $L _ { m }$ and $L _ { c }$ the loss terms for momentum and continuity functions respectively, $\mathbf { R } _ { m } ^ { j }$ and $R _ { c } ^ { j }$ are the momentum vector and continuity residuals corresponding to the $j ^ { t h }$ domain test point. $D _ { r e f }$ refers to the characteristic diameter of the aorta, and $u _ { r e f }$ the reference velocity, corresponding to the inlet velocity.

Regarding the physical interpretation of these error metrics, the $L ^ { 2 }$ norm allows us to determine whether there are substantial variations in the average residual error, thereby providing a general measure of the sensitivity in the computation of first and second-order derivatives. In contrast, the $L _ { \infty }$ norm highlights the presence of isolated points where the enforcement of the Navier–Stokes equations may be failing.

## 4. Results

The following section presents the results obtained from the implementation and training of the developed PINN, applying the fluid-dynamics equations, physical constraints, and network architecture described in the preceding sections. The analysis focuses on the model’s ability to reproduce velocity and pressure fields consistent with the underlying physics, as well as its behaviour during the optimization process, including the convergence of the diferent loss components and the stability of training.

Qualitative and quantitative comparisons are also provided to assess the fidelity of the model with respect to the expected characteristics of a 3D pulsatile blood-flow regime. Taken together, these results allow us to evaluate the feasibility and limitations of the proposed PINN-based approach for haemodynamic simulation.

The distribution of pressures on the surface of the abdominal aorta is represented in Figure 6, where red colour represents high pressure, green represents mean pressure and blue represents low pressure, as seen in the colour scale. All pressures are scaled in the range of the diastolic and systolic pressure, meaning that blue areas are locally below diastolic pressure.

![](images/322e27d6533b7b264ab9cde5eae642b8e0bbb1a0d9178a1e309865a80fcfdb50.jpg)  
Figure 6: Front, lateral and elevation perspectives of 3D pressure colour map. The colormap represents normalised pressures.

Although the results are shown here as static images, the actual output of the implemented module is an interactive 3D image of the abdominal aorta with the surface coloured according to the pressure predicted by the model.

![](images/d254fe9f00e3351b121b0e079853996de769b12875abea385e03082207498330.jpg)

Figure 7: Representation of internal velocity vectors at random domain points for $t = 5 . 0$ s (diastole)  
![](images/1690923866790e1ba7a93d007a9bbea30fe943fa06602263e254ee4a0f6c24de.jpg)  
Figure 8: Representation of internal velocity vectors at random domain points for $t = 5 . 5$ s (systole)  
Figures 7 and 8 show the velocity vectors at multiple random points inside

the geometry for two diferent instants of time (the first for the diastole and the second for the systole). This vectors show that peak velocity can be found at the centre of the abdominal aorta, before the common iliac bifurcation. The velocity descends as the blood flows inside the smaller iliac arteries. It can also be seen that there is plenty of direction change, due to collisions with the walls and the appearance of vortices, leading to irregular flux.

Also, a statistical analysis of the numerical pressure and velocity results was also carried out, separating the individual vector components and subdividing the region of interest into the descending aorta and the iliac bifurcations for two instants of time (diastole and systole). This allowed us to compare the diferences in blood-flow behaviour before and after the aortic bifurcation. The outcomes of this analysis are presented in Tables 1 and 2.

Table 1: Statistical results for velocity and pressure in both subsets of the ROI for $t = 5 . 0$ s (diastole).
<table><tr><td rowspan=1 colspan=1>Mean $v _ { i } \ ( \mathbf { m } / \mathbf { s } )$ </td><td rowspan=1 colspan=1>Inlet (Abdom. Aorta)</td><td rowspan=1 colspan=1>Outlet (Iliac arteries)</td></tr><tr><td rowspan=1 colspan=1> $v _ { x }$  $v _ { y }$  $v _ { z }$  $| v |$ </td><td rowspan=1 colspan=1> $- 0 . 0 0 0 4 \pm 0 . 0 1 2 1$  $0 . 0 1 0 2 \pm 0 . 0 0 7 3$  $- 0 . 0 0 7 7 \pm 0 . 0 1 3 4$  $0 . 0 2 3 2 { \pm } 0 . 0 0 2 5$ </td><td rowspan=1 colspan=1> $\overline { { 0 . 0 1 3 0 \pm 0 . 0 0 0 3 } }$  $0 . 0 1 9 7 \pm 0 . 0 0 0 7$  $- 0 . 0 2 3 3 \pm 0 . 0 0 1 8$  $0 . 0 3 3 2 \pm 0 . 0 0 1 5$ </td></tr><tr><td rowspan=1 colspan=1>Pressure (Pa)</td><td rowspan=1 colspan=1>Abdom. Aorta</td><td rowspan=1 colspan=1>Iliac arteries</td></tr><tr><td rowspan=2 colspan=1>meanstd. dev.minmax</td><td rowspan=1 colspan=1>9176.08</td><td rowspan=2 colspan=1>9463.7572.519245.809578.57</td></tr><tr><td rowspan=1 colspan=1>116.748970.149466.23</td></tr></table>

Table 2: Statistical results for velocity and pressure in both subsets of the ROI for $t = 5 . 5$ s (systole).
<table><tr><td>Mean  $\overline { { { v _ { i } \mathrm { ~ } ( \mathbf { m } / \mathbf { s } ) } } }$ </td><td>Inlet (Abdom. Aorta)</td><td>Outlet (Iliac arteries)</td></tr><tr><td> $v _ { x }$   $v _ { y }$   $v _ { z }$ </td><td> $\overline { { - 0 . 0 2 1 3 } } \pm 0 . 0 0 3 2$   $0 . 0 1 4 8 \pm 0 . 0 2 3 7$   $- 0 . 4 8 1 9 \pm 0 . 0 5 9 7$ </td><td> $- 0 . 0 1 7 9 \pm 0 . 0 0 0 4$   $0 . 0 1 4 2 \pm 0 . 0 0 1 2$   $- 0 . 0 1 7 0 \pm 0 . 0 0 1 0$ </td></tr><tr><td> $| v |$  Pressure (Pa)</td><td> $0 . 4 8 3 2 \pm 0 . 0 5 9 7$  Abdom. Aorta</td><td> $0 . 0 2 8 5 \pm 0 . 0 0 0 2$  Iliac arteries</td></tr><tr><td>mean</td><td>13974.50</td><td>10608.95</td></tr><tr><td>std. dev. min</td><td>1122.36</td><td>782.53</td></tr></table>

As shown in Tables 1 and 2, the numerical analysis reveals a clear distinction between the haemodynamic behaviour in the descending abdominal aorta and in the iliac arteries. Mean velocity and pressure values are consistently higher generally in the pre-bifurcation segment, reflecting the larger vessel calibre and the more coherent axial flow characteristic of this region. In contrast, although the average magnitudes of the lateral velocity components $( v _ { x } \ \mathrm { a n d } \ v _ { y } )$ decrease after the bifurcation, their standard deviations increase markedly in the iliac arteries. This behaviour is consistent with the more tortuous geometry of the iliac branches, where curvature and branching efects promote greater lateral deviations, wall interactions, and local changes in flow direction.

During diastole $\left( t = 5 . 0 \mathrm { s } \right)$ , the mean pressure is relatively uniform throughout the domain, with slightly higher values observed in the iliac arteries. This behavior is consistent with the temporal propagation of the pressure wave and the small pressure gradients typically present during this phase of the cardiac cycle. As for the flow acceleration phase $\left( t = 5 . 5 \mathrm { s } \right)$ , a clearly defined pressure gradient develops between the abdominal aorta and the iliac arteries. The pressure is significantly higher upstream and decreases towards the outlets, which is consistent with the expected physiological blood flow from the inlet towards the bifurcations.

Overall, the statistical comparison highlights the expected reduction in axial flow intensity downstream of the bifurcation, together with an increase in lateral velocity variability associated with the more complex post-bifurcation geometry. Moreover, there is a clear distinction on the results for the diastolic and systolic phases. The ones for the diastole are clearly lower (both for the pressures and the velocities), while the values for the systole are visibly higher. This results are physically consisten with what is expected for the physiological values.

Figure 9 represents the temporal evolution of the pressure in our region of interest. Here we can see the evolution of the maximum, minimum, mean, standart deviation and value at the inlet and outlet of the pressure from t = 0 untill $t = 1 0 . 0 ~ \mathrm { s }$ . We can observe here that our model learns the pulsatile behaviour of the blood flow (specifically following the frequency of the inlet pulse).

![](images/2fb4036116c7fd463143a8c4ff5cd2ca5beb51f45b23d8380a6a5779499c08b3.jpg)  
Figure 9: Temporal evolution of the pressure: minimum (top left), maximum (top right), mean (center left), standard deviation (center right), pressure at an inlet point (bottom left), and pressure at an outlet point (bottom right).

Finally, the validation of the current model has been performed according to the criteria specified in the previous Section 3.4, addressing the physical consistency of the model using the corresponding residuals of the continuity and momentum equations to calculate the $L ^ { 2 }$ and $L _ { \infty }$ metrics for each, as specified in equations 28 , given the results shown in Table 3:

Table 3: Validation metrics.
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>Score</td></tr><tr><td rowspan=1 colspan=1>Continuity $\overline { { \mathrm { L } ^ { 2 } } }$ </td><td rowspan=4 colspan=1> $\overline { { 3 . 6 7 \times 1 0 ^ { - 2 } } }$  $1 . 0 8 \times 1 0 ^ { - 1 }$  $2 . 8 \times 1 0 ^ { 0 }$  $3 . 6 0 \times 1 0 ^ { 0 }$ </td></tr><tr><td rowspan=1 colspan=1>Continuity L∞</td></tr><tr><td rowspan=1 colspan=1>Momentum $\mathrm { L } ^ { 2 }$ </td></tr><tr><td rowspan=1 colspan=1>Momentum $\mathrm { L } \infty$ </td></tr></table>

## 5. Discussion

The main obstacle in assessing the results lies in the absence of a CFD based comparative model that could serve as ground truth. Nevertheless, this limitation has been mitigated by analysing the error metrics reported in Table 3 and by comparing both these metrics and the visual outputs of the model with those presented in related studies. Such works include CFDand PINN-based simulations performed on simpler geometries as well as on vascular structures of comparable complexity.

Beyond validating the physical consistency of the results, as outlined in Section 3.4, we conducted a comparative analysis with the previously discussed studies, using them as benchmarks for the expected behaviour of blood-flow simulations in a large-calibre artery such as the abdominal aorta. Although these works do not address the problem in an identical manner, all reference studies used for comparison rely on similar vascular geometries and use fully three-dimensional time-dependent formulations.

The pressure distribution obtained by the model is consistent with what has been reported in simulations involving simpler and lower-dimensional geometries. Specifically, the results align with the expected behaviour in a large-diameter vessel that branches into smaller arteries, where higher pressures concentrate in the proximal, wider regions and progressively decrease along the bifurcations. This pattern is consistent with observations reported in studies such as Song et al. (2024) [17] and Kabir et al. (2021)[18].

Building on other studies that address problems of comparable complexity, several works have explored the use of PINNs in 4D flow simulations, yielding results that are consistent with those obtained in the present model. For instance, the pressure profile reported for the abdominal aorta in Zhang et al. (2023) [26] closely resembles the distribution predicted by our model, aside from the specific geometric characteristics of the aortic segment selected in each case.

Similarly, the study by Cruz González et al. (2025) [25] follows a comparable methodological approach, albeit using a simplified vascular geometry of an Abdominal Aorta Aneurysm for their experiments. Their predicted pressure profile exhibits the same qualitative behaviour, with elevated pressures in regions of larger diameter that progressively decrease as the flow moves downstream. In this case, the reported $L ^ { 2 }$ errors are of the same order of magnitude as those obtained in the present work, and in some instances even slightly higher.

## 6. Limitations

Although the current model yields satisfactory results, several ways for improvement should be considered. This section addresses the main limitations that were faced during the development of the project, and also the future works leading to solve those known limitations.

One of the main remaining limitations of the present model is the assumption of Newtonian blood behaviour. Although this approximation is commonly adopted in large-vessel haemodynamic simulations, particularly in the aorta where shear rates are generally high, blood exhibits non-Newtonian properties that may influence local flow behaviour under complex haemodynamic conditions [41].

In abdominal aortic aneurysms, regions of recirculation, flow separation, and low wall shear stress can amplify these non-Newtonian efects, potentially afecting the accuracy of pressure and velocity predictions near the aneurysmal wall. Incorporating non-Newtonian constitutive models, such as the Carreau-Yasuda or Casson formulations [42], into the PINN framework could therefore improve the physical realism of the simulations.

However, extending PINNs to non-Newtonian haemodynamics substantially increases the complexity of the governing equations and the associated optimization process, particularly in three-dimensional and time-dependent domains.

Beyond non-Newtonian fluid dynamics modelling, another important improvement concerns the model’s ability to adapt to diferent vascular morphologies. At present, the network is trained on a single aortic geometry, and its predictions are therefore limited to that specific configuration, risking to lose much of its accuracy with diferent aortas. For clinical deployment, however, it would be essential to develop a model capable of generalising to previously unseen geometries, while sacrificing as little accuracy as possible.

Finally, it is worth considering whether more advanced PINN-based architectures could further enhance performance. Models such as CNN-PINNs or GCN-PINNs have demonstrated strong results in other PDE-based applications, including fluid dynamics [43, 44]. These architectures, however, loses the flexibility of operating directly on point clouds and instead require images or graph structures, which are considerably more demanding in terms of computational resources.

## 7. Conclusions and future works

The study has presented a complete workflow for the construction, training, and evaluation of a physics-informed neural network tailored to simulate haemodynamics in a patient-specific abdominal aorta. Beyond the formulation of the PINN itself, the work integrates several methodological components that enable the model to operate on realistic vascular geometries. The network was implemented using the DeepXDE library, which provided the computational framework for enforcing the Navier–Stokes equations throughout the domain. To support this, a dedicated preprocessing pipeline was developed to handle the STL-based anatomical data, extract and classify boundary points, and generate interior sampling points suitable for both training and testing. This pipeline ensured that the model could be trained directly on complex, irregular geometries without the need for mesh generation, while maintaining control over the distribution of points across inlet, outlet, and wall regions.

In parallel, the study incorporated an interactive visualisation system capable of rendering the predicted pressure distribution on the aortic surface in three dimensions. This tool enables dynamic inspection of the haemodynamic fields and provides an intuitive means of assessing the physical plausibility of the results, complementing the quantitative validation metrics. Together, these components form a coherent framework that demonstrates the feasibility of applying PINNs to clinically relevant vascular simulations and lays the groundwork for future extensions involving more complex flow regimes and patient-specific variability.

In conclusion, a PINN model has been successfully trained to produce the results required for future work aimed at improving the prevention and treatment of abdominal aortic aneurysms. This approach has the potential to reduce the morbidity and mortality associated with such a prevalent condition by providing clinicians with a valuable decision-support tool that facilitates diagnosis and the anticipation of possible complications.

Looking ahead, several research lines for future development emerge from the limitations identified in this study. A priority for subsequent work is the incorporation of non-Newtonian fluid dynamics into the haemodynamic formulation, as the flow patterns observed in the abdominal aorta suggest the presence of simplified behaviour. Integrating these models, would allow the network to represent more realistic flow regimes, at the cost of increased mathematical and computational complexity.

In addition, enhancing the model’s ability to generalise across diverse vascular morphologies represents an essential step toward clinical applicability. Training the network on multiple aortic geometries, rather than a single configuration, would enable more robust predictions in patient-specific scenarios. Finally, exploring more advanced PINN-based architectures, such as CNN-PINNs or GCN-PINNs, may ofer further improvements in accuracy for complex PDE-driven problems, though these approaches require more structured data and greater computational resources as well. Together, these directions outline a clear path for strengthening the model’s predictive capabilities and expanding its potential for real-world clinical integration.

## 8. Ethics Statement

All procedures performed in this study were conducted in full compliance with the relevant national legislation and institutional guidelines of the University of Alicante and Avamed Synergy.

Informed consent for the acquisition and research use of imaging data was obtained from all human subjects by the clinical provider (Avamed Synergy) prior to data anonymisation and transfer. All personal identifiers were removed before data processing, and the privacy rights of all participants were strictly protected throughout the study. No additional interventions, experiments, or procedures were performed on human subjects specifically for this research.

## References

[1] Y. Qiu, J. Wang, J. Zhao, T. Wang, T. Zheng, and D. Yuan, “Association between blood flow pattern and rupture risk of abdominal aortic aneurysm based on computational fluid dynamics,” European Journal of Vascular and Endovascular Surgery, vol. 64, no. 2, pp. 155–164, 2022.

[2] S. N. Doost, D. Ghista, B. Su, et al., “Heart blood flow simulation: a perspective review,” BioMedical Engineering OnLine, vol. 15, p. 101, 2016.

[3] V. C. Rispoli, J. F. Nielsen, K. S. Nayak, et al., “Computational fluid dynamics simulations of blood flow regularized by 3d phase contrast mri,” BioMedical Engineering OnLine, vol. 14, p. 110, 2015.

[4] D. Zhang, T. Anjum, Z. Chu, J. S. Cross, and G. Ji, “Simulation of multiphase flow with thermochemical reactions: A review of computational fluid dynamics (cfd) theory to ai integration,” Renewable and Sustainable Energy Reviews, vol. 221, p. 115895, 2025.

[5] S. Fujimura, H. Kanebayashi, K. Karagiozov, T. Sano, S. Hataoka, M. Fuga, I. Kan, H. Takao, T. Ishibashi, M. Yamamoto, and Y. Murayama, “Development of a computationally eficient cfd method for blood flow analysis following flow diverter stent deployment and its application to treatment planning,” Bioengineering, vol. 12, no. 8, 2025.

[6] C. Yang, X. Yang, and X. Xiao, “Data-driven projection method in fluid simulation,” Computer Animation and Virtual Worlds, vol. 27, no. 3-4, pp. 415–424, 2016.

[7] J. N. Kutz, “Deep learning in fluid dynamics,” Journal of Fluid Mechanics, vol. 814, p. 1–4, 2017.

[8] M. Raissi, P. Perdikaris, and G. Karniadakis, “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations,” Journal of Computational Physics, vol. 378, pp. 686–707, 2019.

[9] M. Lino, S. Fotiadis, A. A. Bharath, and C. D. Cantwell, “Current and emerging deep-learning methods for the simulation of fluid dynamics,”

Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, vol. 479, p. 20230058, 07 2023.

[10] C. Caron, P. Lauret, and A. Bastide, “Machine learning to speed up computational fluid dynamics engineering simulations for built environments: A review,” Building and Environment, vol. 267, p. 112229, 2025.

[11] J. Lee, S. Shin, T. Kim, et al., “Physics informed neural networks for fluid flow analysis with repetitive parameter initialization,” Scientific Reports, vol. 15, p. 16740, 2025.

[12] T. Botarelli, M. Fanfani, P. Nesi, and L. Pinelli, “Using physics-informed neural networks for solving navier-stokes equations in fluid dynamic complex scenarios,” Engineering Applications of Artificial Intelligence, vol. 148, p. 110347, 2025.

[13] L. Sun, H. Gao, S. Pan, and J.-X. Wang, “Surrogate modeling for fluid flows based on physics-constrained deep learning without simulation data,” Computer Methods in Applied Mechanics and Engineering, vol. 361, p. 112732, 2020.

[14] D. A. Vorp, “Biomechanics of abdominal aortic aneurysm,” Journal of Biomechanics, vol. 40, no. 9, pp. 1887–1902, 2007. Epub 2007 Jan 24.

[15] Y. Qiu, J. Wang, J. Zhao, T. Wang, T. Zheng, and D. Yuan, “Association between blood flow pattern and rupture risk of abdominal aortic aneurysm based on computational fluid dynamics,” European Journal of Vascular and Endovascular Surgery, vol. 64, no. 2, pp. 155–164, 2022.

[16] O. Peypoch, L. Calsina Juscafresa, A. Vega-Méndez, B. Lobato-Delgado, J. Fité, B. Soto, L. Nieto, M. de la Rosa Estadella, A. Uribezubia, J. Romero, E. Plana, M. Miralles, A. Clarà, J. Dilmé, J. Soria, M. Camacho, A. Martinez-Perez, and M. Sabater-Lleal, “A comprehensive analysis of the abdominal aortic aneurysm growth rate in the spanish population,” Journal of Clinical Medicine, vol. 14, p. 4720, July 2025.

[17] H. S. Wong, W. X. Chan, B. H. Li, and C. H. Yap, “Strategies for multicase physics-informed neural networks for tube flows: a study using 2d flow scenarios,” Scientific Reports, vol. 14, no. 1, p. 62117, 2024.

[18] M. A. Kabir, M. F. Alam, and M. A. Uddin, “Numerical simulation of pulsatile blood flow: a study with normal artery, and arteries with single and multiple stenosis,” Journal of Engineering and Applied Science, vol. 70, no. 1, p. 25, 2021.

[19] Y. Liu, L. Cai, Y. Chen, P. Ma, and Q. Zhong, “Variable separated physics-informed neural networks based on adaptive weighted loss functions for blood flow model,” Computers & Mathematics with Applications, vol. 153, pp. 108–122, 2024.

[20] M. F. Fathi, I. Perez-Raya, A. Baghaie, P. Berg, G. Janiga, A. Arzani, and R. M. D’Souza, “Super-resolution and denoising of 4d-flow mri using physics-informed deep neural nets,” Computer Methods and Programs in Biomedicine, vol. 197, p. 105729, 2020.

[21] G. Kissas, Y. Yang, E. Hwuang, W. R. Witschey, J. A. Detre, and P. Perdikaris, “Machine learning in cardiovascular flows modeling: Predicting arterial blood pressure from non-invasive 4d flow mri data using physics-informed neural networks,” Computer Methods in Applied Mechanics and Engineering, vol. 358, p. 112623, 2020.

[22] N. Alzhanov, E. Y. K. Ng, and Y. Zhao, “Three-dimensional physicsinformed neural network simulation in coronary artery trees,” Fluids, vol. 9, no. 7, 2024.

[23] P. Heger, D. Hilger, M. Full, and N. Hosters, “Investigation of physics-informed deep learning for the prediction of parametric, threedimensional flow based on boundary data,” Computers & Fluids, vol. 278, p. 106302, 2024.

[24] Y. Liu, L. Cai, Y. Chen, J. Xue, W. He, W. Xie, and J. Wei, “Integral conservation physics-informed neural networks with diferent network architectures for patient-specific aortic flow simulations,” International Journal of Heat and Fluid Flow, vol. 117, p. 110011, 2026.

[25] O. L. Cruz-González, V. Deplano, and B. Ghattas, “Enhanced vascular flow simulations in aortic aneurysm via physics-informed neural networks and deep operator networks,” arXiv preprint arXiv:2503.17402, 2025.

[26] X. Zhang, B. Mao, Y. Che, J. Kang, M. Luo, A. Qiao, Y. Liu, H. Anzai, M. Ohta, Y. Guo, and G. Li, “Physics-informed neural networks (pinns) for 4d hemodynamics prediction: An investigation of optimal framework based on vascular morphology,” Computers in Biology and Medicine, vol. 164, p. 107287, 2023.

[27] O. L. Cruz-González, V. Deplano, and B. Ghattas, “Enhanced vascular flow simulations in aortic aneurysm via physics-informed neural networks and deep operator networks,” Mechanics Research Communications, vol. 153, p. 104642, 2026.

[28] Ö. Abaid Ur Rehman, M. Ekici, M. A. Farooq, K. Butt, M. Ajao-Olarinoye, Z. Wang, and H. Liu, “Fluid–structure interaction analysis of pulsatile flow in arterial aneurysms with physics-informed neural networks and computational fluid dynamics,” Physics of Fluids, 2025.

[29] X. Zhang, B. Mao, Y. Che, J. Kang, M. Luo, A. Qiao, Y. Liu, H. Anzai, M. Ohta, Y. Guo, and G. Li, “Physics-informed neural networks (pinns) for 4d hemodynamics prediction: An investigation of optimal framework based on vascular morphology,” Computers in Biology and Medicine, vol. 164, p. 107287, 2023.

[30] X. Chen et al., “Application of computational fluid dynamics and physics-informed neural networks in predicting rupture risk of thoracoabdominal aneurysms with fluid-structure interaction analysis,” Chinese Journal of Physics, vol. 95, pp. 433–454, 2025.

[31] P.-Y. Chuang and L. Barba, “Experience report of physics-informed neural networks in fluid simulations: pitfalls and frustration,” in Proceedings of the SciPy Conference, pp. 28–36, 01 2022.

[32] A. Aghaee and M. O. Khan, “Pinning down the accuracy of physicsinformed neural networks under laminar and turbulent-like aortic blood flow conditions,” Computers in Biology and Medicine, 2025.

[33] R. Temam, Navier–Stokes equations: theory and numerical analysis, vol. 343. American Mathematical Society, 2024.

[34] C. Rao, H. Sun, and Y. Liu, “Physics-informed deep learning for incompressible laminar flows,” Theoretical and Applied Mechanics Letters, vol. 10, no. 3, pp. 207–212, 2020.

[35] G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, and L. Yang, “Physics-informed machine learning,” Nature Reviews Physics, vol. 3, no. 6, pp. 422–440, 2021.

[36] L. Lu, X. Meng, Z. Mao, and G. E. Karniadakis, “Deepxde: A deep learning library for solving diferential equations,” SIAM Review, vol. 63, no. 1, pp. 208–228, 2021.

[37] E. Schubert, J. Sander, M. Ester, H. P. Kriegel, and X. Xu, “Dbscan revisited, revisited: why and how you should (still) use dbscan,” ACM Transactions on Database Systems (TODS), vol. 42, no. 3, pp. 1–21, 2017.

[38] J. Garcia, R. L. F. van der Palen, E. Bollache, K. Jarvis, M. J. Rose, A. J. Barker, J. D. Collins, J. C. Carr, J. Robinson, C. K. Rigsby, and M. Markl, “Distribution of blood flow velocity in the normal aorta: Efect of age and gender,” Journal of Magnetic Resonance Imaging, vol. 47, no. 2, pp. 487–498, 2018.

[39] M. Tancik, P. P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ramamoorthi, J. T. Barron, and R. Ng, “Fourier features let networks learn high frequency functions in low dimensional domains,” in Advances in Neural Information Processing Systems (NeurIPS), 2020.

[40] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778, 2016.

[41] Y. Cho and K. Kensey, “Efects of the non-newtonian viscosity of blood on flows in a diseased arterial vessel. part 1: Steady flows,” Biorheology, vol. 28, pp. 241–62, 02 1991.

[42] F. J. H. Gijsen, F. N. van de Vosse, and J. D. Janssen, “The influence of the non-newtonian properties of blood on the flow in large arteries: steady flow in a carotid bifurcation model,” Journal of Biomechanics, vol. 32, no. 6, pp. 601–608, 1999.

[43] Y. Y. Liu, J. X. Shen, P. P. Yang, and X. W. Yang, “A cnn-pinn-drl driven method for shape optimization of airfoils,” Engineering Appli-

cations of Computational Fluid Mechanics, vol. 19, no. 1, p. 2445144, 2025.

[44] H. Gao, M. J. Zahr, and J.-X. Wang, “Physics-informed graph neural galerkin networks: A unified framework for solving pde-governed forward and inverse problems,” Computer Methods in Applied Mechanics and Engineering, vol. 390, p. 114502, 2022.