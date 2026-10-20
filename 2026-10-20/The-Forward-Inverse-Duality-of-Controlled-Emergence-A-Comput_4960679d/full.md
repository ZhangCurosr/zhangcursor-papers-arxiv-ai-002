# The Forward/Inverse Duality of Controlled Emergence: A Computational Design Program from Explanation to Engineering

Hao Li<sup>∗1</sup>, Minghui Ren<sup>1</sup>, and Shaowen Lu<sup>2</sup>

<sup>1</sup>Aufkenn Technologies Co., Ltd., Ningbo, China <sup>2</sup>Northeastern University, Shenyang, China

October 20, 2026

## Abstract

The macroscopic capabilities of intelligent systems arise from the local interactions of many simple units. When such systems are deployed in the real world, their behaviour must nevertheless be constrained, leaving designers caught between releasing adaptability and preventing loss of control, yet without formal tools to resolve the dilemma. This paper proposes a unified research program for controlled emergence that restates this tension as a pair of dual problems: the forward problem explains macroscopic behaviour from local rules, and the inverse problem solves for controllable rules and readout mechanisms from a desired target. We take an augmented quintuple as the formal object shared by both ends, and decompose “engineering” into three independently measurable coordinates: controllability, reproducibility, and degree of formalizability. On this basis we give three testable working hypotheses (critical scale; the cooperation and trade-of between constraint and adaptability; minimal constraint) and one feasibility criterion, which reduce inverse design from unconstrained optimization to interval checking. The program takes explicit failure boundaries as its premise, does not pursue complete control, and always preserves the system’s room for self-organization. The reduced pilot experiment attached to this paper serves only to verify the executability of the research protocol and does not constitute a statistical test of the working hypotheses. This design language generalizes to multi-agent artificial intelligence, the industrial internet, and emergency response systems, all of which must confront emergence and controllability at once.

Keywords: controlled emergence; computational science; inverse design; controllability; reproducibility; degree of formalizability; system-of-systems engineering

## 1 Introduction: the servitization paradox of intelligent emergence

## 1.1 A case of control failure

Consider a typical field cooperative-coverage scenario. The designer sets geofencing and obstacleavoidance priorities to the maximum; when the UAV swarm encounters a fallen tree, the whole swarm evades in lockstep, falls into oscillation, and the coverage task fails completely. Post-hoc review finds that if the constraint strength is reduced by about 40%, the swarm scatters briefly but self-organizes back to coverage within 3 seconds.<sup>1</sup>

This phenomenon is not an isolated case; it points to an engineering dilemma familiar to readers across disciplines: constraints that are too strong or too weak both fail, and what the designer really needs is exactly the right amount of control, yet where “exactly” falls currently has no formal answer (see Figure 1 for the left–right contrast of the control-failure case).

![](images/82795d7b95716ca1c584c02c9c8012fd1ede0f4ada76714fc165000e932d2aad.jpg)  
Figure 1: UAV swarm: control failure and self-organized recovery. Schematic diagram (integrating phenomena reported in the literature, e.g. Wang N et al., 2021, rather than a specific flight-test record), placing two regimes side by side to make the tension between constraint and adaptability visible. Left panel (strong constraint): upon encountering a fallen tree, the whole swarm evades in lockstep under a single control law; behaviour is highly synchronized and trajectories oscillate, and task coverage fails completely, showing that over-strong constraint suppresses the room left for self-organization. Right panel (relaxed constraint): after constraint strength is reduced by about 40%, the swarm scatters briefly and then self-organizes back to coverage within roughly 3 seconds. This case gives intuition for the paper’s trade-of view that “under weak constraint the three quantities rise together, while further tightening succeeds only at the cost of adaptability”: the balance point lies neither at zero constraint nor at full constraint, but in between.

## 1.2 Intelligence is emergent: an interdisciplinary consensus

Neuronal populations produce perception and memory, ant colonies forage without central command, and artificial networks acquire abilities that were never explicitly programmed once their scale expands. Since Lewes (1875) introduced the word “emergent” and Wertheimer (1912)’s Gestalt experiments established that “the whole is greater than the sum of its parts”, this phenomenon has been recognized repeatedly across disciplines: from dissipative structures (Prigogine and Nicolis, 1977), synergetic order parameters (Haken, 1977), hypercycles (Eigen and Schuster, 1979), and attractor networks (Hopfield, 1982), to self-organized criticality (Bak et al., 1987), the autocatalytic origins of order (Kaufman, 1993), artificial life “life as it could be” (Langton, 1989), and constrained generating procedures (Holland, 1998). Interdisciplinary research has confirmed that when many simple units interact under local rules, the whole can exhibit capabilities that no single unit possesses

(Anderson, 1972; Kelso, 1995).

## 1.3 But emergent intelligence must be controllable if it is to serve people and enterprises

Engineering practice is pushing the question from “why emergence” toward “how to make desirable capabilities emerge”. The case in Section 1.1 already reveals that over-strong constraints succeed only at the cost of adaptability; constraints that are too weak are equally dangerous. A certain multi-agent reinforcement learning system did not impose suficient constraints on resource allocation, and the agents spontaneously evolved a “hoarding and blocking” strategy, occupying communication channels to prevent other nodes from obtaining information; this behaviour was never explicitly programmed and reproduced across runs, constituting a typical case of undesirable emergence.<sup>2</sup> More insidious is constraint misalignment: a cross-factory industrial-internet platform optimized only single-factory eficiency without considering supply-chain coupling, so that single-factory utilization rose while the system-level delivery cycle grew longer, and the bottleneck shifted between factories without being eliminated.<sup>3</sup> The three failure modes (too strong, too weak, misaligned) jointly show that both the location and the strength of the constraint matter, while where “exactly” falls still lacks a formal answer.

## 1.4 The thesis of this paper: unifying the explanatory and the engineering approaches from the perspective of computational science

This paper argues that the two directions of emergence research (the explanatory and the engineering) can be unified under a single computational object.

Forward problem (explanatory). Given local rules and interactions $( R , { \mathcal { D } } , \pi , W )$ , find the macroscopic state or an explanation of it

$$
Y = \pi ( { \mathcal { D } } ( X , R , W ) ) ,\tag{1}
$$

The goal is understanding, prediction, and causal attribution.

Inverse problem (engineering). Given a target neighbourhood $T ,$ a constraint set $C ,$ and performance requirements $( \kappa _ { 0 } , \rho _ { 0 } , \phi _ { 0 } )$ , find the designable components

$$
( R , \mathcal { D } , \pi , U ) ^ { * } = \underset { R , \mathcal { D } , \pi , U } { \arg \operatorname* { m a x } } J ( \lambda ) \quad \mathrm { s . t . } \quad \kappa \ge \kappa _ { 0 } , ~ \rho \ge \rho _ { 0 } , ~ \phi \ge \phi _ { 0 } , ~ \lambda ^ { * } \le \lambda _ { \mathrm { t i p } } ,\tag{2}
$$

The goal is construction, optimization, and verification. Physics, neuroscience, and system-ofsystems engineering have historically each occupied one end: physics and neuroscience excel at explanation, while system-of-systems engineering cannot avoid confronting the inverse problem. What is distinctive about the computational-science perspective is that it asks simultaneously about constructibility, computability, verifiability, and complexity boundaries. The augmented quintuple $E = ( X , R , \mathcal { D } , Y , \pi ; U , W , T , C )$ is the formal object shared by both ends.

(See Figure 2 for the master diagram of the forward/inverse duality.)

Table 1: The forward/inverse problem duality.
<table><tr><td>Problem direction</td><td>Given</td><td>Solve for</td><td>Goal</td><td>Tools in this paper</td></tr><tr><td>Forward (explanatory)</td><td> $( R , { \mathcal { D } } , \pi , W )$ </td><td> $Y$  or an explanation  $\operatorname { o f } Y$ </td><td>Understanding, prediction, causal attribution</td><td>Quintuple, ε, three emergence criteria</td></tr><tr><td>Inverse (engineering)</td><td> $T , C , \kappa _ { 0 } , \rho _ { 0 } , \phi _ { 0 }$ </td><td> $( R , { \mathcal { D } } , \pi , U )$ </td><td>Construction, optimization, verification</td><td> $\kappa / \rho / \phi , \mathrm { H 1 \mathrm { - } H 3 , }$   $J ( \lambda ) , \lambda ^ { * } \leq \lambda _ { \mathrm { t i p } }$ </td></tr><tr><td>Unified</td><td>The same E</td><td>Forward and inverse mappings</td><td>Explainable and engineerable</td><td>Computational abstraction, information, causality, formalization, complexity</td></tr></table>

![](images/9cbf943d7953be134a6e2279a1003056544402d84fd8e7bd91f147a0a7c6a8b8.jpg)  
Figure 2: Master diagram of the forward/inverse problem duality. Taking the augmented quintuple $E = ( X , R , \mathcal { D } , Y , \pi ; U , W , T , C )$ as the common formal object, “controlled emergence” is restated as a pair of dual problems: the forward problem explains macroscopic behaviour from local rules, and the inverse problem solves for controllable rules and readouts from a desired target. Forward (explanatory): given the microscopic state set $X ,$ , the interaction rules and topology $R ,$ the microscopic dynamics ${ \mathcal { D } } ,$ the readout mapping π, and the environmental perturbations $W .$ , find the macroscopic behaviour $Y$ and its explanation; this corresponds to understanding, prediction, and causal attribution, with tools being the quintuple, the closure error ε, and the three emergence criteria. Inverse (engineering): given a target neighbourhood $T ,$ , a constraint set $C ,$ and the three-property thresholds $\kappa _ { 0 } , \rho _ { 0 } , \phi _ { 0 }$ , find the optimal rules and readout $( R , { \mathcal { D } } , \pi , U ) ^ { * } = \arg \operatorname* { m a x } J ( \lambda ) ;$ this corresponds to construction, optimization, and verification, with tools being $\kappa / \rho / \phi ,$ , hypotheses H1–H3, and the feasibility criterion $\lambda ^ { * } \leq \lambda _ { \mathrm { t i p } }$ . Three-layer structure: the language layer is the augmented quintuple (Section 2), the target layer is controllability $\kappa ,$ reproducibility $\rho ,$ and formalizability $\phi$ (Section 3), and the principle layer is H1–H3 together with the utility $J ( \lambda )$ (Section 4).

Well-posedness and complexity of the inverse problem. The inverse problem $( R , { \mathcal { D } } , \pi , U ) ^ { * } =$ arg max J(λ) is ill-posed in general; the solution may not exist, may not be unique, or may depend discontinuously on $T , C .$ . This paper constrains well-posedness by three means: (i) under the exponential phenomenological model the geometric weighting J is concave and $\lambda _ { \mathrm { t i p } }$ is unique; (ii) regularization $J _ { \eta } = J ( \lambda ) - \eta \Omega ( R , \mathcal { D } , \pi )$ , where Ω is a complexity penalty and η is chosen by the L-curve or cross-validation; (iii) the feasibility criterion $\lambda ^ { * } \leq \lambda _ { \mathrm { t i p } }$ reduces the ill-posed case from unconstrained optimization to interval checking. In terms of complexity, $\lambda ^ { * } , \lambda _ { \mathrm { t i p } }$ have closed-form solutions under the exponential model; in the non-convex case arg max is generally not polynomially solvable, and this paper reports robustness via a neutral-weight drift band rather than claiming global optimality (echoing the four boundaries in Section 6.2).

From “explaining the mechanisms by which macroscopic capabilities arise” to “inversely solving for rules and readouts”, there is a spectrum of maturity. This paper situates the scattered concepts along this axis:

Table 2: The maturity spectrum.
<table><tr><td>Stage</td><td>Question</td><td>Computational-science tool</td><td>Position in this paper</td></tr><tr><td>Explanation</td><td>Why mechanisms produce macroscopic capabilities</td><td>Abstract interpretation, coarse-graining</td><td> $\pi , \varepsilon$ </td></tr><tr><td>Prediction</td><td>Predict the macroscopic given rules</td><td>Simulation, complexity analysis</td><td>N*, d</td></tr><tr><td>Reproduction</td><td>Statistical stability</td><td>Information theory, Wasserstein</td><td>ρ</td></tr><tr><td>Control</td><td>Meeting-specification control</td><td>Control theory, RL</td><td>κ</td></tr><tr><td>Design</td><td>Inversely solving rules and readouts</td><td>Program synthesis, optimization</td><td>J(λ)</td></tr><tr><td>Verification</td><td>Formal guarantees</td><td>Model checking, theorem proving</td><td> $\phi$ </td></tr></table>

Positioning against existing programs. This paper occupies a position not yet taken in the spectrum of existing programs:

Table 3: Positioning against existing programs.
<table><tr><td>Program</td><td>Core claim</td><td>Relation to this paper</td></tr><tr><td>Cybernetics (Wiener, 1948)</td><td>Feedback control of single-system stability</td><td>No micro/macro layering, no  $N ^ { * }$  phase transition</td></tr><tr><td>Constrained generating procedures (Holland, 1998)</td><td>Cross-disciplinary mechanisms of emergence</td><td>Explanatory; no inverse design and no  $\kappa / \rho / \phi$ </td></tr><tr><td>Value alignment (Russell, 2019)</td><td>Human uncertainty about machine values</td><td>Single-agent value learning; no collective-emergence criterion</td></tr><tr><td>Superintelligence governance (Tegmark, 2017)</td><td>A governance framework for future intelligence</td><td>No formal object and no measurable coordinates</td></tr><tr><td>Systematics of creation (Qian, 2001)</td><td>Methodology for open complex giant systems</td><td>Qualitative framework; no  $\kappa / \rho / \phi$  and no λ design</td></tr><tr><td>This paper</td><td>Forward/inverse duality + κ/ρ/φ + H1-H3</td><td>The same quintuple unifying both ends</td></tr></table>

The three gaps blocking this turn correspond exactly to the three layers of design tools provided in this paper:

Table 4: Three gaps and the corresponding tools.
<table><tr><td>Gap blocking the turn Corresponding tool in this paper</td></tr><tr><td>Semantic gap: the physical vocabulary (fluc- Design language: the augmented quintuple tuations, order parameters, symmetry break- and the unified computational-science table ing, coarse-graining) and the computational (Section 2) concerns (algorithms, models, architectures) are misaligned in timescale and causal direc- tion, so that “improving robustness&quot; cannot be translated back into design constraints on</td></tr><tr><td>Missing integration: condensed-matter Design language: a unified formal object so physics, systems science, neuroscience, phi- that the three emergence criteria (descriptive, losophy of mind, and AI each have their own causal, sceptical null hypothesis) can be com- criteria; the same phenomenon is described</td></tr><tr><td>in different languages across disciplines, and one cannot judge whether two cases belong to the same kind Missing formalization: the designer can only Design objectives  $\kappa / \rho / \phi$  (Section 3) decom- set local units and rules directly, while macro- pose “engineering&quot; into measurable coordi- scopic capabilities are an indirectly induced nates; the design principles of three working</td></tr></table>

The rest of the paper proceeds layer by layer through the language, objective, and principle layers: Section 2 gives the formal object unifying forward and inverse, Section 3 decomposes “engineering” into three independently measurable properties, and Section 4 gives the inverse-design space and the feasibility criterion.

## 1.5 Positioning, contributions, and boundaries

Terminology. “Intelligent emergence” is the problem motivation of this paper and what the title refers to; “controlled emergence” is the formal object of this paper; the intelligence measure $S = w _ { 1 } A + w _ { 2 } I + w _ { 3 } K$ of Section 3.4 is the bridge connecting “intelligence” to engineering coordinates, and its full three-dimensional validation (especially the stable estimation of $I , K )$ is listed as a core task of the formal experiment; this paper does not claim to have established it.

Three sources of intelligence. By mechanism, this paper distinguishes self-organizing emergence (the core object: ant colonies, UAV swarms, collective prediction), scale-triggered emergence (the contrast case: LLMs, which have a scale threshold but are not self-organized by local rules), and engineering optimization (non-emergent: supervised learning, RL policies, designed top-down). All three may present, phenomenologically, the pattern of “absent in small systems, present in large ones”, but the mechanisms difer (see Section 2.3).

Positioning of the pilot (a one-time statement). The reduced feasibility pre-experiment reported in this paper (area coverage $N = 3 0 , M _ { \mathrm { r u n } } = 5 0$ ; foraging $N = 4 0 , M _ { \mathrm { r u n } } = 6 0 )$ verifies only the executability of the protocol and constitutes neither support for nor falsification of any hypothesis. Because the area-coverage task is too simple and the injected constraint has too little efect, the three quantities barely change with λ; this is “not yet entering a testable region” rather than “being falsified”. In the foraging task, the decreasing inflection structure of $\kappa , \rho$ near $\lambda \approx 0 . 3 6$ is only a preliminary indication. The statistical adjudication of the hypotheses is left to the formal experiment with $M _ { \mathrm { r u n } } \geq 1 0 0 0$ (ordinary scenarios) or $M _ { \mathrm { r u n } } \geq 1 0 ^ { 4 }$ (safety-critical scenarios).

Illustrative nature of the cases. The first case in Section 1.1 and the two later cases in Section 1.3 are illustrative narratives made to demonstrate the trade-of between constraint and adaptability, integrating similar phenomena reported in the published literature (see the footnotes), and are not records of any specific experiment or enterprise incident.

How to read this paper. Section 2 gives the formal object unifying forward and inverse (the language layer); Section 3 decomposes “engineering” into the three measurable coordinates $\kappa / \rho / \phi$ (the objective layer); Section 4 gives H1–H3 together with $J ( \lambda )$ and $\lambda ^ { * } \leq \lambda _ { \mathrm { t i p } }$ (the principle layer); Section 5 is the validation plan and the positioning of the pilot; Section 6 is the interdisciplinary dialogue, the limits of engineerability, and ethics. Readers with a strong mathematical background can jump from the duality table in Section 1.4 directly to Section 2; engineering readers can enter via the operationalization mapping table in Section 4.4.

What this paper does not do. It does not adjudicate whether strong emergence is possible, does not provide a universal inverse-solving algorithm, does not claim the quintuple is the only formal object, and does not adjudicate the dispute over LLM “emergent abilities”. These four points are all listed as future work.

## 2 Design language: a unified formal object for emergence and controllability

Chapter claim. Emergence research in physics, neuroscience, AI, and system-of-systems engineering can be compared under a single formal object.

## 2.1 The augmented quintuple

An emergence system that can be instantiated across domains is defined as an augmented quintuple:

$$
E = ( X , R , \mathcal { D } , Y , \pi ; U , W , T , C )\tag{3}
$$

where the original components are: $X = \{ x _ { 1 } , \ldots , x _ { N } \}$ is the set of microscopic unit states; R is the local interaction rules and topology; D is the microscopic state-dynamics update function; Y is the macroscopic state-and-capability space; π is the coarse-graining or readout mapping from micro to macro.

Engineering-controllability extension components: $U = \{ u _ { 1 } , \ldots , u _ { N } \}$ is the external control input; W is the environmental perturbation; $T \subseteq Y$ is the target macroscopic-state neighbourhood; $C$ is the constraint set (rule boundaries, permissions, and value constraints).

Microscopic dynamics equation:

$$
x _ { i } ( t + 1 ) = \mathcal { D } _ { i } \big ( x _ { i } ( t ) , \{ x _ { j } ( t ) \} _ { j \in N _ { i } } , r _ { i } , u _ { i } ( t ) , w _ { i } ( t ) \big )\tag{4}
$$

where $N _ { i }$ is defined by R. The macroscopic readout is $y ( t ) = \pi ( x ( t ) )$

The five core components correspond one-to-one with four concerns of computer science: R corresponds to algorithms and protocols, D to models, Y to system-level output, and π to architecture and aggregation mechanisms. The following table places the explanatory concepts, engineering concepts, quintuple components, computational-science tools, and this paper’s criteria side by side, so that the forward and inverse problems are compared within the same row.

Table 5: Unified computational-science table.
<table><tr><td>Explanatory concept</td><td>Engineering concept</td><td>Quintuple component</td><td>Computational- science tool</td><td>Criterion in this paper</td></tr><tr><td>Order parameter</td><td>System-level KPI</td><td>π readout</td><td>Aggregation, dimensionality</td><td>κ</td></tr><tr><td>Symmetry breaking</td><td>Pattern selection</td><td>D convergence</td><td>reduction Asymmetric attractors</td><td>Y</td></tr><tr><td>Criticality</td><td>Phase-transition threshold</td><td>N-R interaction</td><td>Complexity, random graphs</td><td>N*</td></tr><tr><td>Coarse-graining</td><td>State estimation</td><td>π</td><td>Abstract interpretation, coarse-graining</td><td>ε</td></tr><tr><td>Fluctuation</td><td>Robustness</td><td>W</td><td>Random seeds, noise</td><td>ρ</td></tr><tr><td>Constraint</td><td>Safety fencing</td><td>C</td><td>Permissions, LTL specifications</td><td>φ</td></tr><tr><td>Control input</td><td>Human takeover</td><td>U</td><td>Intervention, control</td><td>K</td></tr><tr><td>Relaxation time</td><td>Response latency</td><td>D rate</td><td>Number of convergence steps</td><td> $T _ { \mathrm { m a x } }$ </td></tr></table>

The original “physics–computation isomorphism” mapping (order parameter/stable macroscopic variables, symmetry breaking/asymmetric attractors, etc.) has been merged into the first two columns of the table above and is no longer listed separately.

Macroscopic closure error ε. Let Ψ be the hypothesized macroscopic dynamics; then the validity of coarse-graining is measured by the commutative-diagram error:

$$
\varepsilon = \operatorname* { s u p } _ { x \in X } \left. \pi ( \mathcal { D } ( x ) ) - \Psi ( \pi ( x ) ) \right. _ { Y }\tag{5}
$$

where $\left\| \cdot \right\| _ { Y }$ is a norm on the macroscopic state space Y (in practice the Euclidean norm or a task-relevant metric). ε measures the discrepancy between “evolve then coarse-grain” and “coarsegrain then evolve”: the smaller ε is, the closer the macroscopic layer is to autonomy; the larger ε is, the harder it is to predict macroscopic behaviour without reference to microscopic detail. ε serves as a complementary indicator of the degree of formalizability ϕ (see Section 3.3). See Figure 3(a) for the augmented quintuple and the coarse-graining commutative diagram.

![](images/759719f6cb72042698d4a5f029a45246b72bb8b4fd96c66293174314ca962c29.jpg)  
Figure 3: Coarse-graining commutative diagram of the augmented quintuple. (a) two-layer X → Y readout; (b) hierarchical $\widetilde { E }$ with a three-stage readout $X  { \mathcal { M } }  Y$ and two-level closure errors $\varepsilon _ { 1 } , \varepsilon _ { 2 }$

## 2.2 Comparison with existing formalizations

The following table compares existing formalizations along their core object, measurable quantity, and engineering interface:

Irreplaceability of the four elements. The four elements — micro/macro layering, the readout mapping π, the scale phase transition $N ^ { * }$ , and the continuous constraint strength λ — cannot all be expressed naturally in any of the frameworks above: remove π and the macroscopic quantity degenerates into $g ( X )$ , so ε becomes undefined; remove layering and N, R no longer enter the model, so hypothesis H1 cannot be stated; remove continuous λ and $J ( \lambda )$ and the feasibility criterion become undefined.

Proposition (minimality of the four elements, proof sketch). Let $\mathfrak { F }$ be an arbitrary formal framework. If F cannot simultaneously express (i) micro/macro layering, (ii) the readout mapping π, (iii) the scale phase transition $N ^ { * }$ , and (iv) the continuous constraint strength $\lambda ,$ then there exists an emergence property expressible by the augmented quintuple but not by F. The proof idea is to construct a counterexample system for each missing element. Without $\pi _ { \ i }$ take a commutative diagram requiring comparison of $\pi \circ \mathcal { D }$ and $\Psi \circ \pi ;$ with no $\pi$ in $\mathfrak { F } , \varepsilon$ is undefined. Without layering, take a two-level phase transition $N ^ { * } = \operatorname* { m a x } ( N _ { 1 } ^ { * } , N _ { 2 } ^ { * } )$ , which a single-layer model in $\mathfrak { F }$ cannot accommodate at once. Without $N ^ { * }$ , take a system whose performance is non-monotonic in N; with N as an external hyperparameter in ${ \mathfrak { F } } ,$ the non-monotonic point has nowhere to reside. Without continuous $\lambda ,$ take a system requiring comparison of the inflection positions of $\kappa ( \lambda )$ and $\rho ( \lambda )$ ; with the constraint as a discrete switch in ${ \mathfrak { F } } ,$ there is no two-region structure. This is a counterexample-style minimality argument, not a uniqueness proof; whether the four elements are strictly irreducible is open problem 4 in Section 6.5.

Economy of joint optimization. We concede that a patchwork of frameworks can, in terms of expressiveness, just barely cover the ground, but the quintuple’s increment lies not in “being able to express” but in gathering the three formerly scattered design variables π, N, λ into a single object so they can be optimized jointly. When they are scattered across diferent frameworks, the designer cannot tune these three quantities simultaneously and observe the joint response of $\kappa / \rho / \phi$ . This is an engineering economy, not a monopoly on expressiveness.

Table 6: Comparison with existing formalizations.
<table><tr><td rowspan=1 colspan=6>Framework          Core object               Measurable quantity Substantive difference from the aug-/ engineering inter- mented quintupleface</td></tr><tr><td rowspan=1 colspan=2>system</td><td></td><td rowspan=1 colspan=2>systems / cyber- dynamical system</td><td></td></tr><tr><td rowspan=1 colspan=5>netics1948)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=4>POMDP</td><td rowspan=1 colspan=1>Sequential decision-</td><td rowspan=1 colspan=1>Belief state / deci-</td></tr><tr><td rowspan=6 colspan=4>MAS formalisms(BDI/ATL)SoS architecture</td><td rowspan=1 colspan=1>making by a single</td><td rowspan=1 colspan=1>sion</td></tr><tr><td rowspan=1 colspan=1>decision-maker</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Multi-agent reasoning</td><td rowspan=1 colspan=1>Task performance</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>algorithmic platform</td></tr><tr><td rowspan=2 colspan=1>Static configuration</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Qualitative assess-</td></tr><tr><td></td><td rowspan=1 colspan=3>language (AADL)</td><td rowspan=1 colspan=1>of component connec-</td><td rowspan=1 colspan=1>ment     mature</td></tr><tr><td></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>tions</td><td rowspan=1 colspan=1>process</td></tr><tr><td></td><td rowspan=1 colspan=3>DEVS  (Zeigler</td><td rowspan=1 colspan=1>Hierarchical discrete-</td><td rowspan=1 colspan=1>Compositional simu-</td></tr><tr><td></td><td rowspan=1 colspan=3>et al., 2000)</td><td rowspan=1 colspan=1>event modelling</td><td rowspan=1 colspan=1>lation / simulation</td></tr><tr><td></td><td rowspan=2 colspan=3></td><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td></td><td rowspan=1 colspan=1></td></tr><tr><td></td><td rowspan=1 colspan=3>SysML (Object</td><td rowspan=1 colspan=1>Graphical specifica- N</td><td rowspan=1 colspan=1>one / specification</td></tr><tr><td></td><td rowspan=1 colspan=3>Management</td><td rowspan=1 colspan=1>tion</td><td rowspan=1 colspan=1></td></tr><tr><td></td><td rowspan=1 colspan=3>Group, 2015)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td></td><td rowspan=1 colspan=3>Networkedcon-</td><td rowspan=1 colspan=1>Feedback       under</td><td rowspan=1 colspan=1>Stability margin /</td></tr><tr><td></td><td rowspan=1 colspan=3>trol systems</td><td rowspan=1 colspan=1>bandwidth/latency</td><td rowspan=1 colspan=1>control</td></tr><tr><td></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>constraints</td><td rowspan=1 colspan=1></td></tr><tr><td></td><td rowspan=1 colspan=3>Dec-POMDP</td><td rowspan=1 colspan=1>Multi-agent sequential</td><td rowspan=1 colspan=1>Reward / learning</td></tr><tr><td></td><td rowspan=1 colspan=3>Mean-field MARL</td><td rowspan=1 colspan=1>decision-making</td><td rowspan=1 colspan=1></td></tr><tr><td></td><td rowspan=1 colspan=3>(Bernstein et al.,</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td></td><td rowspan=1 colspan=3>2002; Yang et al.,</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td></td><td rowspan=1 colspan=3>2018)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Behaviour-based</td><td rowspan=1 colspan=1>Layered subsumption</td><td rowspan=1 colspan=1>Task performance /</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>robotics (Brooks, a</td><td rowspan=1 colspan=1>rchitecture</td><td rowspan=1 colspan=1>robot platform</td></tr><tr><td rowspan=3 colspan=4>Swarm-</td><td rowspan=1 colspan=1>1986, 1991)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Ant/particle swarm al-</td><td rowspan=1 colspan=1>Optimization perfor-</td></tr><tr><td></td><td rowspan=1 colspan=3>intelligence</td><td rowspan=1 colspan=1>gorithms</td><td rowspan=1 colspan=1>mance / mature</td></tr><tr><td></td><td rowspan=1 colspan=3>engineering</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td></td><td rowspan=1 colspan=3>(Bonabeau et al.,</td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=1></td></tr><tr><td></td><td rowspan=1 colspan=3>1999)</td><td rowspan=1 colspan=1></td></tr><tr><td></td><td rowspan=1 colspan=3>Safety   control S</td><td rowspan=1 colspan=1>afety of a single-step</td><td rowspan=1 colspan=1>Safety certificate /</td></tr><tr><td></td><td rowspan=1 colspan=3>(CBF / safe RL co</td><td rowspan=1 colspan=1>ntrol law</td><td rowspan=1 colspan=1>mature tools</td></tr><tr><td rowspan=1 colspan=4>/ formal verifica-</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=4>tion)Hierarchical quituple E (this pa-</td><td rowspan=1 colspan=1></td><td></td></tr><tr><td rowspan=1 colspan=1>n- X-M-Y three layers</td><td></td></tr><tr><td rowspan=1 colspan=1></td><td></td></tr><tr><td rowspan=1 colspan=4>per&#x27;s extension)</td><td></td><td></td></tr></table>

Boundary with safety control. Control barrier functions (CBF), safe RL, and formal verification (CTL/LTL model checking) intersect with this paper’s $\kappa , \phi$ , but their goals difer: CBF concerns the strict safety of a single-step control law, while this paper’s κ is a survival probability and residence time; safe RL concerns the expected return under policy constraints, while this paper’s $J ( \lambda )$ is a weighted geometric mean; formal verification (already counted in ϕ) provides machine proofs of universal propositions, while this paper’s κ is an empirical probability. The three are complementary: in engineering practice one should first lock down hard constraints with formal verification and then measure the statistical performance of soft constraints with $\kappa / \rho$

## 2.3 A conceptual map of the three sources of intelligence

Table 7: Three sources of intelligence.
<table><tr><td>Type</td><td>Mechanism</td><td>Representative</td><td>Relation to this paper</td></tr><tr><td>Self-organizing emergence (core object)</td><td>Local-rule interaction → macroscopic capability</td><td>Ant colonies, UAV swarms, collective prediction</td><td>The object of controlled-emergence engineering</td></tr><tr><td>Scale-triggered (contrast case)</td><td>Scale stacking + gradient descent → capability jump</td><td>LLMs, large vision models</td><td>Has a scale threshold but is not self-organizing; used for boundary testing</td></tr><tr><td>Engineering optimization (non-emergent)</td><td>Explicit design objective + direct optimization</td><td>Supervised learning, RL policies</td><td>Designed top-down; not emergence</td></tr></table>

Phenomenologically all three may present “absent in small systems, present in large ones”, but the mechanisms difer. The three criteria of this paper can likewise be used to adjudicate scale-triggered phenomena (see Section 6.1), which itself is a test of the framework’s cross-domain applicability.

## 2.4 Hierarchical extension: micro, meso, and macro (outlook)

In cases such as UAV formations, foraging groups, factory clusters, and emergency coordination groups, the two-layer definition above actually contains an unformalized intermediate layer, namely the mesoscopic structure. This paper generalizes the quintuple into a hierarchical form:

$$
\widetilde { E } = ( X , \mathcal { M } , Y ; \ : R , \mathcal { D } , \pi _ { 1 } , \pi _ { 2 } ; \ : U , W , T , C )\tag{6}
$$

where M is the set of mesoscopic structures (local groups, modules, formations), $\pi _ { 1 } : X \to { \mathcal { M } }$ and $\pi _ { 2 } : { \mathcal { M } } \to Y$ , and the original readout decomposes as $\pi = \pi _ { 2 } \circ \pi _ { 1 }$ . Let $\mathcal { D } _ { \mathcal { M } }$ be the mesoscopic-layer dynamics; then the coarse-graining error decomposes as:

$$
\begin{array} { r } { \varepsilon _ { 1 } = \big \| \pi _ { 1 } \circ \mathcal { D } - \mathcal { D } _ { \mathcal { M } } \circ \pi _ { 1 } \big \| , \qquad \varepsilon _ { 2 } = \big \| \pi _ { 2 } \circ \mathcal { D } _ { \mathcal { M } } - \Psi \circ \pi _ { 2 } \big \| } \end{array}\tag{7}
$$

The total closure error is $\varepsilon = \operatorname* { m a x } ( \varepsilon _ { 1 } , \varepsilon _ { 2 } )$ . This decomposition turns the chain “constraint first changes the mesoscopic structure, which in turn influences the macroscopic” from a black box into an analysable object.

Constructive definition of $\pi _ { 1 }$ . This paper gives two operational mesoscopic readout mappings: (i) a clustering mapping based on graph community detection, running the Louvain/Leiden algorithm on the adjacency graph and labelling connected modules as mesoscopic units $m _ { k }$ , with the resolution parameter $\gamma _ { \mathrm { r e s } }$ controlling the number of modules; (ii) a DBSCAN mapping based on spatial density, labelling connected clusters with pairwise distance below $\varepsilon _ { \mathrm { d b } }$ and neighbourhood size at least $n _ { \mathrm { m i n } }$ units as $m _ { k }$ . The condition for both mappings to yield stable output under symmetric D is that the community structure does not undergo label flipping under perturbation by $W$ (measured across runs by the adjusted Rand index).

General form of $\mathcal { D } _ { \mathcal { M } }$ . The mesoscopic-layer dynamics take the form

$$
\begin{array} { r } { m _ { k } ( t + 1 ) = \mathcal { D } _ { \mathcal { M } } \big ( m _ { k } ( t ) , \{ \pi _ { 1 } ( x _ { i } ( t ) ) \} _ { i \in B _ { k } } , u _ { \mathcal { M } , k } ( t ) , w _ { \mathcal { M } , k } ( t ) \big ) , } \end{array}\tag{8}
$$

where $B _ { k }$ is the set of microscopic units inside the k-th mesoscopic unit. Two instances: threshold dynamics for formation formation/dispersal (a formation forms when the mean alignment angle within $m _ { k }$ exceeds $\theta _ { \mathrm { a l i g n } } )$ ; switching dynamics for module activation/dormancy (a module goes dormant when the local return within $m _ { k }$ falls below a threshold).

Estimation protocol for $\varepsilon _ { 1 } , \varepsilon _ { 2 } .$ . (i) Run $M _ { \mathrm { r u n } }$ microscopic trajectories and use $\pi _ { 1 }$ to obtain mesoscopic sequences; (ii) fit $\mathcal { D } _ { \mathcal { M } }$ on the mesoscopic sequences (linear or threshold switching); (iii) compare $\pi _ { 1 } ( \mathcal { D } ( \boldsymbol { x } ( t ) ) )$ with $\mathcal { D } _ { \mathcal { M } } ( \pi _ { 1 } ( \boldsymbol { x } ( t ) ) )$ ) to obtain $\varepsilon _ { 1 } ; ( \mathrm { i v } )$ fit Ψ on the mesoscopic sequences and compare $\pi _ { 2 } ( \mathcal { D } _ { \mathcal { M } } ( m ( t ) ) )$ ) with $\Psi ( \pi _ { 2 } ( m ( t ) ) )$ to obtain $\varepsilon _ { \mathrm { { 2 } } } ; ~ \mathrm { { ( v ) } }$ resample trajectories $B = 1 0 0 0$ times by bootstrap and report the 95% CI.

The main-line analysis still proceeds under the two-layer $E ;$ the hierarchical $\widetilde { E }$ provides the constructions above, while the design of mesoscopic control $U _ { \mathcal { M } }$ and mesoscopic constraints $C _ { \mathcal { M } }$ is left for future work (see Figure 3(b) for the three-stage readout commutative diagram).

Chapter summary. This chapter gives the augmented quintuple $E = ( X , R , \mathcal { D } , Y , \pi ; U , W , T , C )$ as the common formal object of the forward and inverse problems, and argues its irreplaceability via the minimality-of-four-elements proposition; the unified computational-science table places explanatory and engineering concepts side by side; the hierarchical $\widetilde { E }$ gives an operational construction of $\pi _ { 1 }$ and an estimation protocol for $\varepsilon _ { 1 } , \varepsilon _ { 2 }$ . The next chapter decomposes the “engineerin $\mathrm  | g ^ { , } \bar  \it$ on this formal object into three independently measurable coordinates.

## 3 Design objectives: three measurable properties

Chapter claim. Controlled emergence can be decomposed into three independently measurable properties — controllability κ, reproducibility $\rho ,$ and degree of formalizability $\phi ;$ the intelligence measure $S _ { i }$ , as the bridge connecting “intelligence” to engineering coordinates, is given in Section 3.4.

## 3.1 Controllability κ

Let $T \subseteq Y$ be the target neighbourhood; define:

$$
\kappa = P \mathrm { ( } \forall t \in [ 0 , T _ { \operatorname* { m a x } } ] , \ y ( t ) \in T \mid C \mathrm { ) }\tag{9}
$$

In practice two indicators are reported side by side. The strict version (survival-analytic) is $\begin{array} { r } { \widehat { \kappa } _ { \mathrm { s t r i c t } } = \frac { 1 } { M _ { \mathrm { r u n } } } \sum _ { i } \mathcal { k } \{ \tau _ { i } > T _ { \mathrm { m a x } } \} } \end{array}$ , reported with a Wilson interval; the relaxed version (residence-time proportion) is κbresidence = $= \begin{array} { r } { \frac { 1 } { M _ { \mathrm { r u n } } } \sum _ { i } \frac { 1 } { T _ { \mathrm { m a x } } } \sum _ { t } \mathcal { k } \mathcal { E } \{ y _ { i } ( t ) \in T \} } \end{array}$

Sample-size planning: to distinguish $\kappa _ { 1 } , \kappa _ { 2 }$ at two-sided confidence $1 - \alpha$ and power $1 - \beta _ { ; }$ , one needs

$$
M _ { \mathrm { r u n } } \geq \frac { ( z _ { \alpha / 2 } + z _ { \beta } ) ^ { 2 } \bigl [ \kappa _ { 1 } ( 1 - \kappa _ { 1 } ) + \kappa _ { 2 } ( 1 - \kappa _ { 2 } ) \bigr ] } { ( \kappa _ { 2 } - \kappa _ { 1 } ) ^ { 2 } } .\tag{10}
$$

Example: distinguishing $\kappa = 0 . 9 9 \mathrm { f r o m } 0 . 9 9 5 ( \alpha = 0 . 0 5 , \beta = 0 . 2 0 )$ requires $M _ { \mathrm { { r u n } } } \approx 4 . 7 \times 1 0 ^ { 3 }$ ; 90% power requires about $6 . 2 \times 1 0 ^ { 3 }$ . Under the zero-failure rule (0 violations in $M _ { \mathrm { r u n } } ~ \mathrm { r u n s } )$ , assuming runs are i.i.d. and κ is constant, the one-sided $1 - \alpha$ confidence lower bound on the true κ is $\kappa \geq \alpha ^ { 1 / M _ { \mathrm { r u n } } }$ (solved from $\kappa ^ { M _ { \mathrm { r u n } } } = \alpha )$ ; safety-critical scenarios (e.g. $\kappa _ { 0 } \geq 0 . 9 9 9 )$ require $M _ { \mathrm { r u n } } \geq 1 0 ^ { 4 }$ for the lower bound to exceed the threshold. This lower bound is not robust to distribution shift, non-independent runs, or model error; if runs are autocorrelated or κ drifts over time, one must switch to sequential testing or stratified bootstrap. Early signals of loss of control include: a persistently negative sliding slope of the residence rate, an abnormally increasing trajectory variance, and a markedly prolonged recovery time.

## 3.2 Reproducibility ρ

Convert the trajectories of $M _ { \mathrm { r u n } }$ independent runs into empirical distributions $\begin{array} { r } { \mu _ { i } = \frac { 1 } { T _ { \mathrm { m a x } } } \sum _ { t } \delta _ { y _ { i } ( t ) } } \end{array}$ and define:

$$
\widehat { \rho } = 1 - \frac { 2 } { M _ { \mathrm { r u n } } ( M _ { \mathrm { r u n } } - 1 ) } \sum _ { i < j } \frac { W _ { 2 } ( \mu _ { i } , \mu _ { j } ) } { d _ { \mathrm { m a x } } } .\tag{11}
$$

The normalization baseline $d _ { \mathrm { m a x } }$ must distinguish two choices. The theoretical diameter is diam $\begin{array} { r } { ( Y ) = \operatorname* { s u p } _ { y , y ^ { \prime } \in Y } \| y - y ^ { \prime } \| _ { Y } ; } \end{array}$ the empirical diameter is $\widehat { \mathrm { d i a m } } ( Y ) = \operatorname* { m a x } _ { i , j } \| y _ { i } - y _ { j } \|$ (the support set of macroscopic trajectories across all experimental runs). Reporting always uses the empirical diameter, which shares the same support as the $\mu _ { i }$ and guarantees ${ \widehat { \rho } } \in [ 0 , 1 ]$ : since the supports of $\mu _ { i } , \mu _ { j }$ all lie within the trajectory support set, $W _ { 2 } ( \mu _ { i } , \mu _ { j } ) \leq \widehat { \mathrm { d i a m } } ( Y )$ always holds. The theoretical diameter is used only to discuss the compactness of $Y { : }$ when $Y$ is compact and its diameter is attainable, ${ \widehat { \rho } } \in [ 0 , 1 ]$ ; when $Y$ is non-compact, discrete, or unbounded, the sup may not be attained on reachable states, in which case ${ \widehat { \rho } } \in [ 0 , 1 )$ and the theoretical diameter may be infinite and unusable, so one must normalize by the empirical diameter. In the high-dimensional case (dimension of $Y > 5 0 )$ one uses the Sinkhorn entropy-regularized approximation with $\varepsilon _ { \mathrm { s i n k } } = 0 . 0 1 \cdot { \widehat { \mathrm { d i a m } } } ( Y ) ^ { 2 }$ and must report $\varepsilon _ { \mathrm { s i n k } } .$ the approximation error is bounded by $\| W _ { 2 } - W _ { 2 , \varepsilon _ { \mathrm { s i n k } } } \| \leq C \varepsilon _ { \mathrm { s i n k } }$ (C depends on the geometry of the distribution support), so $\varepsilon _ { \mathrm { s i n k } }$ should be a small fraction of the square of $d _ { \mathrm { m a x } }$ rather than an independent constant. Numerical estimation may, due to Sinkhorn error or binned entropy estimation, push the sample $\widehat { \rho }$ slightly above 1 (e.g. 1.00–1.02); when reporting, truncate to [0, 1] and note the number of truncations and the proportion of truncated samples; this truncation introduces a small negative bias that must be stated together with the estimate, and must not be interpreted as “reproducibility exceeding perfection”.

Statistical uncertainty: the confidence interval of $\widehat { \rho }$ uses bootstrap (resample the $M _ { \mathrm { r u n } }$ trajectories with replacement $B = 1 0 0 0$ times, recomputing the pairwise $W _ { 2 }$ average each time; the cost is $O ( B M _ { \mathrm { r u n } } ^ { 2 } )$ , about $1 0 ^ { 9 } ~ W _ { 2 }$ computations when $M _ { \mathrm { r u n } } = 1 0 0 0$ , so one must use the Sinkhorn approximation or estimate on a subsampled set of trajectories). In the degenerate case $d _ { \operatorname* { m a x } } = 0$ (all run trajectories coincide exactly), $W _ { 2 } = 0 / 0 $ , so $\widehat { \rho }$ is undefined and is reported as $\widehat \rho = 1$ (perfect reproduction) with the note “degenerate case, distance between distributions not estimated”.

## 3.3 Degree of formalizability $\phi$

A proposition template library plus conditionalization is used. The template library is $P _ { \mathrm { l i b r a r y } } =$ {safety, stability, convergence, coverage, liveness, fairness, real-time, scalability}, each entry corresponding to a CTL/LTL specification or a Lyapunov condition. For a concrete system, choose the applicable subset $P _ { E } \subseteq P _ { \mathrm { l i b r a r y } } ;$ then:

$$
\phi _ { E } = { \frac { | P _ { \mathrm { p r o v a b l e } } \cap P _ { E } | } { | P _ { E } | } } .\tag{12}
$$

$P _ { \mathrm { p r o v a b l e } }$ is the subset that can be machine-verified by model checking (PRISM, nuXmv) or automated theorem proving. Both $| P _ { E } |$ and the numerator must be reported, to prevent inflating ϕ by selecting few propositions. The macroscopic closure error $\varepsilon$ (Section 2.1) serves as a complementary indicator. See Figure 4 for the three-property design coordinate system.

![](images/161e48000b0444e02f35e85accfd00cb2eac7e7d412441cb273f4675af56e991.jpg)  
Figure 4: Three-property design coordinate system (κ– $^ - \rho$ plane). The horizontal axis is controllability κ and the vertical axis is reproducibility $\rho ;$ each designed system is one point, with point size/grey level encoding formalizability $\phi .$ The scatter is schematic, not real measurement. Ideal region (lower right): both κ and $\rho$ are high, and within the weak-constraint region $\kappa , \rho$ and adaptability A rise together, making it the preferred operating point. Trade-of region (upper left / high constraint): as constraint is tightened further, $\kappa , \rho$ show diminishing marginal returns while adaptability A drops markedly, and a point must be chosen by the decision-maker’s weights $\theta , \psi$ within the framework $\mathcal { \dot { I } } ( \lambda ) = \kappa ^ { \dot { \theta } } \rho ^ { \psi } \widetilde { A } ^ { 1 - \theta - \dot { \psi } }$ . Note that $\kappa$ and $\rho$ are not mutually exclusive; the real trade-of lies between $\kappa / \rho$ and A (see Figure 5).

## 3.4 The intelligence measure S: a bridge, pending validation

S is the bridge connecting “intelligence” to engineering coordinates, not a result already established in this paper. The working definition of intelligence is: a system’s ability to maintain its own organization in a given environment and to form, through information integration, a causal structure that can guide action toward achieving goals. It decomposes into three dimensions:

$$
S = w _ { 1 } A + w _ { 2 } I + w _ { 3 } K .\tag{13}
$$

where A (adaptive behaviour) is the task’s cumulative return or a performance score; I (information integration) in this paper prefers efective information $\mathrm { E I } = I ( X _ { 0 } ; X _ { 1 } \mid$ uniform input), is not tied to IIT, and IIT’s $\Phi _ { I I T }$ (Tononi, 2004) $\left( \Phi _ { I I T } \right.$ is IIT’s original symbol and is unrelated to the D in the quintuple), geometric integrated information, and so on are also legitimate replacements; K (causal modelling) is the prediction accuracy of environmental interventions, benchmarked on the intervention and counterfactual framework of Pearl (2009), and the quantification of macroscopic causal eficacy (Hoel, 2025; Zhang and Liu, 2023) can provide candidate operationalizations for both I and K; w<sub>1</sub>, w<sub>2</sub>, w<sub>3</sub> are task-relevant weights, by default equal weights $1 / 3$ after variance normalization.

Criterion for intelligent emergence: if and only if $S ( Y ) > \operatorname* { m a x } _ { i } S ( X _ { i } )$ , and this gain cannot be explained by the nonlinearity of π alone (ruled out via the sceptical-null-hypothesis permutation test). Dimensional-comparison prerequisite: macro and micro must use the same task benchmark, the same input distribution, and the same intervention set.

Weight sensitivity: when the per-dimension gains are comparable in magnitude $( | \Delta _ { \mathrm { n e g } } | / | \Delta _ { \mathrm { p o s } } | \leq$ 2) and the weights lie within [0.1, 0.6], as long as at least two dimensions have $\Delta _ { j } > 0$ , the qualitative conclusion of the criterion does not flip with the weights; the full flip-threshold surface is left for future work. The stable estimation of $I , K$ is listed as a core task of the formal experiment (see Section 5 and Appendix C); this paper does not claim to have established it.

## 3.5 Self-organization headroom: degree of control and the ontological boundary

“Is designed self-organization still self-organization?” is turned into a measurable spectrum. Define the degree of control:

$$
{ \mathrm { d e g r e e ~ o f ~ c o n t r o l } } = { \frac { { \mathcal { I } } ( U ; X _ { t + 1 } \mid X _ { t } , W ) } { H ( X _ { t + 1 } \mid X _ { t } , W ) } } ,\tag{14}
$$

that is, the proportion of the conditional entropy accounted for by the conditional mutual information between the control input and the next state. Implicit premise: this framework applies only to open systems with residual randomness; when $H ( X _ { t + 1 } \mid X _ { t } , W ) = 0$ (a fully deterministic system), the denominator is zero, the degree of control is undefined, and the notion of self-organization headroom does not apply.

Exact estimation is infeasible in high dimensions, so a three-level strategy is used. (i) Primary method: the intervention-contrast proxy degree of control<sup>\</sup> $= 1 - \widehat { H } ( X _ { t + 1 } \mid X _ { t } , W , U = 0 ) / \widehat { H } ( X _ { t + 1 } \mid$ $X _ { t } , W )$ . Its relation to the strict value: the proxy never exceeds the strict value, and its absolute underestimation never exceeds the probability p that control is on at runtime (derivation in Appendix B); but the relative underestimation $( \mathrm { s t r i c t } - \mathrm { p r o x y } ) / \mathrm { s t r i c t }$ has no uniform upper bound, and when the strict degree of control is itself small the relative underestimation can far exceed $p .$ Hence the proxy value reported in the pilot should be understood as a lower bound on the strict degree of control, with the relative bias controllable only when the strict degree of control is not close to 0. (ii) In high dimensions one uses the MINE/InfoNCE variational lower bound, whose bias decays as $1 / M _ { \mathrm { r u n } }$ and whose variance rises with network capacity; the formal experiment must report estimates after train/test separation. (iii) Fallback: use the intervention success rate.

The self-organization headroom is $S _ { \mathrm { s e l f - o r g } } = 1 - \mathrm { d e g r e e }$ of control; the legitimate premise of controlled emergence is $S _ { \mathrm { s e l f - o r g } } > 0 ;$ as $S _ { \mathrm { s e l f - o r g } }  0$ the system slides from “emergence” toward “being choreographed”.

The distinction between design-time and runtime control. The degree of control above measures only the explanatory power of the runtime control input $U ,$ , and does not include the R, D, C chosen by the designer at design time. A system can have $S _ { \mathrm { s e l f - o r g } } \approx 1$ (almost no runtime intervention) while still being highly structured by the designer (strong design-time constraints, such as a preset topology, rules, and value boundaries). Therefore $S _ { \mathrm { s e l f - o r g } } > 0$ guarantees only that “there is still room for self-organization at runtime”, not that “the system is undesigned”; a complete self-organization assessment must also report the design-time constraint strength (e.g. the proportion of the state space occupied by C, or the degree of symmetry constraint in $R )$

Chapter summary. This chapter decomposes “engineering” into three independently measurable coordinates: κ (survival probability $/$ residence time), $\rho \mathrm { ~ \ ( } W _ { 2 }$ distance, normalized by the empirical diameter), and $\phi _ { E }$ (proportion of provable propositions); $S = w _ { 1 } A + w _ { 2 } I + w _ { 3 } K$ serves as a bridge for intelligence pending validation; the bias bound of the degree-of-control proxy and the high-dimensional estimation scheme have been given. The next chapter derives the inverse-design space and the feasibility criterion on these three coordinates.

## 4 Design principles: three testable working hypotheses

Chapter claim. The design space of controlled emergence is characterized jointly by three testable working hypotheses (critical scale; the cooperation and trade-of inflection between $\kappa / \rho$ and adaptability; minimal constraint) and one dimensionless utility function.

Note on wording. The three “propositions” below are testable working hypotheses, not proven theorems. H1b in particular must be read as an exploratory fitting conjecture within a four-premise regime; the only genuinely strict lower bound is the rank condition $N ^ { * } \geq | Y |$ (see Appendix B).

## 4.1 Hypothesis H1: critical scale and interaction density

H1a (weak form, immediately testable). Given the symmetry of R and the readout mode of $\pi _ { \ i }$ the critical scale $N ^ { * }$ decreases monotonically with the interaction density d (the number of efective connections per node): $d _ { 1 } < d _ { 2 } \Rightarrow N ^ { * } ( d _ { 1 } ) \geq N ^ { * } ( d _ { 2 } )$

H1b (strong form, exploratory scaling-law conjecture). On the premise that H1a holds, conjecture that in a specific regime

$$
N ^ { * } \approx c _ { 0 } \cdot \frac { \log | Y | } { \log ( 1 + d ) } ,\tag{15}
$$

where ≈ denotes an approximate fit within the four-premise regime and is not a strict lower bound; the strict bounds are given in Appendix B (the lower bound comes from the rank condition $N \geq | Y |$ , the upper bounds from connectivity and propagation time $N \leq \operatorname* { m i n } ( e ^ { d } , d ^ { T _ { \operatorname* { m a x } } } ) )$ . Note that the log |Y| form of H1b holds only when π is a random projection / nonlinear readout, and its fitted value may fall below $| Y | ;$ if π is a linear / low-dimensional readout, then the rank lower bound $N \geq | Y |$ dominates and H1b does not apply (see Appendix B). $c _ { 0 }$ is a constant depending on the convergence rate of $\mathcal { D } .$ , the type of $\pi ,$ and the graph structure; this paper makes no a priori assertion about its cross-task universality, and only if the measured $c _ { 0 }$ across tasks converges to a narrow interval can one speak of “approximate universality”; otherwise it degenerates into a task-dependent parameter.

Hierarchical-correction conjecture: under $\widetilde { E }$ the phase transition splits into two levels, $N ^ { * } =$ $\operatorname* { m a x } ( N _ { 1 } ^ { * } , N _ { 2 } ^ { * } )$ , which may explain the pilot observation that $^ { 6 6 } N = 1 0$ already sufices” (a few individuals sufice to form a group, but enough formations are needed to sustain system-level coverage).

## 4.2 Hypothesis H2: cooperation and the trade-of inflection between $\kappa / \rho$ and adaptability

In the weak-constraint region, strengthening the constraint raises κ, $\rho ,$ and A simultaneously (cooperation); beyond the inflection point, further strengthening succeeds only at the cost of adaptability A, with diminishing marginal returns in $\kappa , \rho$ (trade-of). A clarification: in the exponential model both $\kappa ( \lambda )$ and $\rho ( \lambda )$ rise monotonically in $\lambda ;$ the two always cooperate and are not mutually exclusive, and the real trade-of occurs between $\kappa / \rho$ and adaptability A. The inflection position is determined by the task and environment. This hypothesis is supported by the exponential phenomenological model of Section 4.4, but $^ { 6 6 } \kappa$ monotone-concave and A monotone-convex” is a working hypothesis to be tested. From the hierarchical viewpoint, $\lambda _ { \mathrm { t i p } }$ may be determined by the stability of the mesoscopic structure; if a formation disperses at some λ, κ may drop abruptly and non-smoothly. Hunt et al. (2019) observed in high-density robot swarms that a simple stigmergy strategy degrades below a random walk as density rises — an independent experimental corroboration of this hypothesis’s claim that “pressing beyond the inflection point harms adaptability”.

## 4.3 Hypothesis H3: minimal constraint $\lambda ^ { * }$

There exists a minimal constraint strength $\lambda ^ { * }$ at which κ meets the specification; beyond $\lambda ^ { * }$ controllability no longer improves, and adaptability is only monotonically lost. The optimal design should approach rather than far exceed $\lambda ^ { * }$ . From the hierarchical viewpoint, the engineering meaning of $\lambda ^ { * }$ is “the minimal constraint that keeps the mesoscopic structure from collapsing”.

## 4.4 The dimensionless utility $J ( \lambda )$ and the feasibility criterion

To avoid the dimensional and weighting problems caused by directly setting $d \kappa / d \lambda = | d A / d \lambda |$ , a dimensionless engineering utility is introduced. Normalization of A: before substitution, the task return A must first be normalized to $[ 0 , 1 ]$ , taking $A _ { \mathrm { m i n } }$ (the unconstrained baseline) and $A _ { \mathrm { m a x } }$ (a theoretical upper bound or the unperturbed optimum), and writing $\widetilde { A } = ( A - A _ { \operatorname* { m i n } } ) / ( A _ { \operatorname* { m a x } } - A _ { \operatorname* { m i n } } )$ ; $\kappa , \rho$ are already in [0, 1]. The dimensionless utility is

$$
J ( \lambda ) = \kappa ( \lambda ) ^ { \theta } \rho ( \lambda ) ^ { \psi } \widetilde { A } ( \lambda ) ^ { 1 - \theta - \psi } , \qquad \theta , \psi \geq 0 , \theta + \psi \leq 1 .\tag{16}
$$

$( \theta , \psi \geq 0$ and $\theta + \psi \leq 1$ guarantee that J is a weighted geometric mean; if $\theta + \psi > 1$ , the exponent of $\widetilde { A }$ becomes negative and the weighting loses meaning.)

θ, ψ are the decision-maker’s relative preference weights for controllability and reproducibility. The inflection point $\lambda _ { \mathrm { t i p } }$ is defined as the interior point where $d J / d \lambda = 0$ . Existence condition: under the exponential model, the left side of $d J / d \lambda = 0$ tends to +∞ as $\lambda \to 0 ^ { + }$ and to $- ( 1 - \theta - \psi ) \gamma _ { r }$ as $\lambda \to \infty ;$ hence the equation has a unique interior solution if and only if $\theta + \psi < 1$ (the decision-maker still gives adaptability positive weight), in which case $\mathrm { H 2 } \mathrm { : } \mathrm { s }$ two-region structure of cooperation and trade-of holds; when $\theta + \psi = 1$ (adaptability entirely disregarded), $J ( \lambda )$ is monotonically increasing with no interior maximum and the two-region structure degenerates, so there is no “cooperation/trade-of boundary”, and in engineering terms one should go back and adjust the weights or the task requirements.

Neutral baseline: the formal experiment first reports $\lambda _ { \mathrm { t i p } } ^ { ( 0 ) }$ under equal weights $\theta = \psi = 1 / 3$ then reports the full drift band $[ \lambda _ { \mathrm { t i p } } ^ { \mathrm { m i n } } , \lambda _ { \mathrm { t i p } } ^ { \mathrm { m a x } } ]$ of $\lambda _ { \mathrm { t i p } }$ over $( \theta , \psi ) \in [ 0 . 1 , 0 . 8 ]$ (with $\theta + \psi \leq 0 . 9 )$ . If the drift band exceeds the λ scan step, the main text must state explicitly that “the inflection position depends on the decision-maker’s risk preference and is not a pure system property”; if it is narrower than the step, then $\lambda _ { \mathrm { t i p } }$ is robust to the weights and can be reported as a system property.

Under the exponential phenomenological model (below, A always denotes the normalized $\widetilde { A }$ with the tilde omitted; after normalization $\stackrel { \sim } { A } _ { 0 } = 1$ , i.e. the normalized adaptability takes its maximum value 1 under no constraint) $\kappa = 1 - e ^ { - \alpha _ { r } \lambda } , \ \rho = 1 - e ^ { - \beta _ { r } \lambda } , \ A = e ^ { - \gamma _ { r } \lambda }$ , we have $\begin{array} { r } { \lambda ^ { * } = - { \frac { 1 } { \alpha _ { r } } } \log ( 1 - \kappa _ { 0 } ) } \end{array}$ and $\lambda _ { \mathrm { o p t } } = \arg \operatorname* { m a x } _ { \lambda \ge \lambda ^ { * } } J ( \lambda )$ . If J is multimodal, report all local maxima and explain the selection basis.

Feasibility criterion: $\lambda ^ { * } \le \lambda _ { \mathrm { t i p } } . \mathrm { ~ I f ~ } \lambda ^ { * } > \lambda _ { \mathrm { t i p } }$ , then the minimal controllable constraint already falls inside the trade-of region; meeting $\kappa _ { 0 }$ must come at the cost of adaptability, which is infeasible under the current system design, and one must lower $\kappa _ { 0 } .$ change R, or adjust D. Define the design margin feasibility $= \lambda _ { \mathrm { t i p } } - \lambda ^ { * }$ and report its bootstrap 95% confidence interval (the resampling unit is the $M _ { \mathrm { r u n } }$ trajectories at each λ scan point; for each $\lambda ,$ resample with replacement, refit the exponential model, and compute $\lambda _ { \mathrm { t i p } }$ and $\lambda ^ { * }$ , repeated $B = 1 0 0 0 ~ \mathrm { t i m e s } )$ ; a lower bound $\leq 0$ casts doubt on feasibility.

Operationalization mapping from λ to the constraint set C. λ is the scalarized “constraint strength”; the way it changes C in concrete tasks is given in Table 8. The semantic direction of λ may difer across tasks (e.g. in the collective-prediction task a high λ means high sharing, the opposite direction from the coverage task, see the table), so before comparing $\lambda _ { \mathrm { t i p } }$ across tasks one must first perform semantic normalization onto a unified “higher means more constrained” axis.

Table 8: Operationalization mapping from λ to the constraint set.
<table><tr><td>Task</td><td>Concrete channel through which λ acts on C</td><td>Direction</td></tr><tr><td>Area coverage</td><td>Geofence radius tightened, obstacle-avoidance High λ = strong con- priority raised</td><td>straint</td></tr><tr><td>Foraging</td><td>Food-search radius narrowed, synchronization rate raised</td><td>rule- High λ = strong con- straint</td></tr><tr><td></td><td>Collective prediction Gating proportion of neighbour-shared observa- High λ = high shar- tions</td><td>ing (opposite direc- tion, needs normal- ization)</td></tr></table>

(See Figure 5 for the three curves $\kappa ( \lambda ) , \rho ( \lambda ) , \widetilde { A } ( \lambda )$ as functions of $\lambda ,$ , the relative positions of $\lambda ^ { * }$ and $\lambda _ { \mathrm { t i p } }$ , and the cooperation/trade-of partition.)

## 4.5 Joint constraints of the three hypotheses and the design workflow

The three hypotheses are not independent: the interaction density d simultaneously afects $N ^ { * } \left( \mathrm { H 1 } \right)$ and the rate parameters $\alpha _ { r } , \beta _ { r } , \gamma _ { r }$ (H2); H2 and H3 are coupled through $\lambda ^ { * } \leq \lambda _ { \mathrm { t i p } }$ . Design workflow:

![](images/1cf7145aa51d2ff18188bd0b8a3beadf0aa0d30f98c82cae7993be0da76ea273.jpg)  
Figure 5: Triptych of the three hypotheses (exponential phenomenological model). In the weak-constraint region (low λ) the three quantities $\kappa , \rho ,$ A rise together; after the inflection point $\lambda _ { \mathrm { t i p } } , \kappa , \rho$ show diminishing marginal returns while A decreases monotonically, forming the trade-of between $\kappa / \rho$ and A. Left: controllability $\kappa ( \lambda ) = 1 - e ^ { - \alpha _ { r } \lambda }$ saturates concavely and monotonically with constraint, and there is a minimal constraint $\lambda ^ { * } = - \alpha _ { r } ^ { - 1 } \ln ( 1 - \kappa _ { 0 } )$ (H3). Middle: reproducibility $\rho ( \lambda ) = 1 - e ^ { - \beta _ { r } \lambda }$ is a monotonically rising curve like κ; the two always cooperate and are not mutually exclusive, and the inflection point $\lambda _ { \mathrm { t i p } }$ is defined by $d J / d \lambda = 0$ (H2). Right: adaptability $\widetilde { A } ( \lambda ) = e ^ { - \gamma _ { r } \lambda }$ decreases monotonically with constraint. Shaded partitions: the light region is the cooperative region (all three quantities rising), and the red region is the trade-of region $( \kappa , \rho$ with diminishing returns and A falling); $\lambda ^ { * } < \lambda _ { \mathrm { t i p } }$ is the feasible design interval.

(1) use H1 to determine feasible $( N , d )$ combinations; (2) under that $( N , d )$ , use H2 to determine $\lambda _ { \mathrm { t i p } } ; ( 3 )$ use H3 to determine $\lambda ^ { * } ; ( 4 )$ check $\lambda ^ { * } \leq \lambda _ { \mathrm { t i p } }$ and the feasibility confidence interval; (5) if feasible, take within $[ \lambda ^ { * } , \lambda _ { \mathrm { t i p } } ]$ the $\lambda _ { \mathrm { o p t } }$ that maximizes $J$

Chapter summary. This chapter gives three testable working hypotheses: H1 (critical scale and interaction density), H2 (cooperation and the trade-of inflection between $\kappa / \rho$ and adaptability), and H3 (the minimal-constraint principle); the dimensionless utility $J ( \lambda ) = \kappa ^ { \theta } \rho ^ { \psi } \widetilde { A } ^ { 1 - \theta - \psi }$ and the feasibility criterion $\lambda ^ { * } \leq \lambda _ { \mathrm { t i p } }$ compress inverse design into a five-step workflow. The next chapter gives the formal experimental plan for testing these three hypotheses and positions the pilot.

## 5 Validation: the formal plan and the pilot

Chapter statement. We give an executable validation plan for the inverse problem and report a reduced pilot; the latter verifies only the executability of the protocol and does not constitute a statistical test of the hypotheses.

Formal plan: three Mesa/NetLogo tasks (area coverage, foraging, collective prediction), with $\lambda \in [ 0 , 1 ]$ at 20 equally spaced points and $M _ { \mathrm { { r u n } } } = 1 0 0 0 ~ \mathrm { { p e r } }$ point $( 1 0 ^ { 4 }$ for safety-critical); measure κbstrict, κbresidence, ${ \widehat { \rho } } , A ;$ ; estimate the three curves non-parametrically by LOWESS and then fit the exponential model; the falsification conditions are no interior maximum of $J ( \lambda )$ (H2) or no saturation plateau of $\kappa ( \lambda )$ (H3); the cross-task convergence of $c _ { 0 }$ is judged by the dual condition of CI overlap $+ ~ C V < 0 . 3$ . See Appendix C for parameters, pseudocode, and the risk table.

Positioning of the pilot: verifies only protocol executability (Section 1.5). In the area-coverage task $( N = 3 0 , M _ { \mathrm { r u n } } = 5 0 )$ the three quantities barely change with $\lambda ,$ which is $^ { 6 6 } \mathrm { n o t }$ yet entering a testable $\mathrm { r e g i o n } ^ { \mathrm { * } }$ rather than “being falsified”; in the foraging task $( N = 4 0 , M _ { \mathrm { r u n } } = 6 0 )$ there are signs of a decreasing inflection in $\kappa , \rho$ near $\lambda \approx 0 . 3 6$ , with power of only 0.2–0.3. I, K are insensitive in the pilot, and their stable estimation is listed as a core task of the formal experiment. See Figure 6 for the pilot’s two-task comparison; see Appendix C.2 for the checklist of what the formal experiment must add.

![](images/5b462e4853a1f5a8a970ff1a350d81f4af165f64aa6e1ffaedb4966949dadb91.jpg)  
Figure 6: Pilot comparison of two tasks. A reduced pre-experiment that only verifies the executability of the study protocol and does not constitute a statistical test of H1–H3; data points carry bootstrap intervals. Area coverage $( N = 3 0 , M _ { \mathrm { r u n } } = 5 0 ) \colon \widehat { \kappa } \approx 0 . 9 4 4 - 0 . 9 4 7$ , and $\widehat { \rho }$ and $\widetilde { A }$ barely change with $\lambda ;$ the three quantities are flat, indicating that the task is too simple and the injected constraint has too little efect to enter a testable region. Foraging $( N = 4 0 , M _ { \mathrm { r u n } } = 6 0 )$ : $\widehat { \kappa } , \widehat { \rho }$ show signs of decreasing near $\lambda \approx 0 . 3 6$ , but the power is only 0.2–0.3, so this is only a preliminary indication, not a hypothesis test.

## 6 Discussion: cross-domain significance, failure domain, and governance

Chapter statement. This framework provides a common language, measurable coordinates, and falsifiable principles for the design of controlled emergence.

## 6.1 Cross-domain significance

Physics and complex-systems science (the instance domain of the explanatory tradition). Physics and complex-systems science provide mature exemplars of forward explanation, and this framework goes further to ask about the possibility of inverse design. The three hypotheses converse with phase-transition theory: H1’s scaling law formally echoes the mean-field phase transition of Landau (1937) and the renormalization-group universality conjecture of Wilson (1975); the quantitative universality constant of period-doubling bifurcation (Feigenbaum, 1978) and the generation of complex dynamics by a simple logistic model (May, 1976) provide early exemplars of “low-dimensional rules, high-dimensional behaviour”; deterministic chaos (Lorenz, 1963) and self-organized criticality (Bak et al., 1987) further show that phase transitions need not occur only in the thermodynamic limit; and fractal geometry (Mandelbrot, 1982) and efective complexity (Gell-Mann, 1994) provide early descriptive languages for multiscale structure and “rule length”. This paper, however, does not presuppose universal constants; the cross-task convergence of $c _ { 0 }$ is listed as a hypothesis to be tested. H2’s two-region structure of cooperation and trade-of provides a phenomenological starting point for “the trade-of between constraint and adaptability”. Wu et al. (2024), who explain pattern generation by symmetry breaking, Hoel (2025), who measures emergence by macroscopic causal eficacy, and the mesoscopic computing paradigm of Wu et al. (2025) can all be situated within the unified framework, though these are mostly principle-level works at present. The relation to this framework is: this tradition provides the mature tools on the forward side $Y = \pi ( { \mathcal { D } } ( X , R , W ) )$ , while the J(λ) and feasibility criterion on the inverse-design side are this paper’s contribution.

Neuroscience and AI (the instance domains of the explanatory and scale-triggered). Neuroscience excels at explanation and the AI side excels at scale engineering; in this framework the two serve as instances of the forward and inverse ends. The framework provides a common language for the dispute over “real emergence versus statistical mirage”: the three branches — descriptive $( \Delta M > 0$ and non-monotonic), causal $\left( \mathrm { C E } > 0 \right)$ , and sceptical null hypothesis (permutation test) run in parallel and converge on an output of “real emergence $/$ statistical mirage $/$ not adjudicated”. Within neuroscience, from the self-organization of neuronal populations in Hebb (1949), the neura Darwinism of Edelman (1987), to the autopoiesis of Maturana and Varela (1980) and the embodied cognition of Varela et al. (1991), “the whole greater than the parts” has been a core topic of theoretical biology; Tononi (2004)’s IIT attempts to quantify this intuition, but its $\Phi _ { I I T }$ is not computable on large systems, which is precisely why this paper prefers efective information EI and treats IIT as a legitimate replacement rather than the default operationalization. On the AI side, beginning with the problematization of Turing (1950), LeCun et al. (2015) and Vaswan et al. (2017) laid the foundations of deep learning and attention architectures; Brown et al. (2020), Kaplan et al. (2020), and Hofmann et al. (2022) systematically reported the scaling laws of LLMs, while Good (1965)’s early warning about superintelligent machines suggests that scale-triggered intelligence likewise needs governance coordinates. The dispute over LLM “emergent abilities” (Schaefer et al., 2023; Wei et al., 2022) precisely demonstrates the necessity of distinguishing scale-triggered from self-organizing emergence, and the three criteria of this paper can likewise be used to adjudicate that dispute. The framework does not presuppose a position but only provides the common structure needed for adjudication. On the philosophy-of-mind side, Dennett (1991), Clark (1997), Hofstadter (1979), and Minsky (1986) have long asked “whether mind emerges from mindless units”; this framework does not answer that philosophical question but provides a set of common coordinates against which the debate can be benchmarked with engineering design. The relation to this framework is: the IIT, CE, and LLM-emergence disputes can all be re-coordinated under $\kappa / \rho / \phi$ and the three criteria.

System-of-systems engineering and governance (the instance domain of inverse engineering). System-of-systems engineering is the most mature instance domain of the inverse problem: the augmented quintuple upgrades SoS from “process management” to “dynamics design”, with R corresponding to the interaction protocols of member systems, D to the microscopic mechanisms of system-level evolution, π to the system-level KPI readout, and C to the boundary constraints of members. This upgrade builds on the evolution of SoS engineering itself: from the architectural principles of Maier (1998) and the survey of US DoD SoS engineering in Dahmann and Baldwin (2008) to the mission-engineering guidance of Department of Defense (2023), the engineering community has accepted that system-level behaviour cannot be extrapolated from single-system design; in the Chinese systems tradition, the “systematics of creation” of Qian Xuesen (Qian, 2001) and the human-machine-environment system intelligence of Liu et al. (2024) likewise attempt to provide an indigenous framework for cross-level emergence. Three domains are instantiated: UAV swarms (Hunt et al., 2019; Kuckling et al., 2023; Mafettone et al., 2024; Rubenstein et al., 2014; Vásárhelyi et al., 2018), with the collective memory and spatial sorting of animal groups (Couzin et al., 2002) as their biological prototype; industrial-internet cross-factory coordination (Liu et al., 2019; Wang et al., 2025b; Zhang et al., 2023); and multi-department emergency response (Butts et al., 2012; Fan and Wang, 2025; Guo and Hu, 2025; Moynihan, 2009; Nunan, 2020; Tao and Zhang, 2023). The emergency cases require three qualifications: the microscopic units X are organizations rather than individuals, and emergent networks are driven by the homogeneity of departmental/administrative hierarchy (Tao and Zhang, 2023); real emergent networks show a structure of “cohesive small clusters + peripheral hanging trees and isolates” with a low $\rho$ baseline (Butts et al., 2012); and in multi-stage disasters $\lambda _ { \mathrm { t i p } }$ drifts across response stages. In addition, Ding et al. (2025) advocate abductive AI to assist understanding emergence in complex systems, a direction complementary to the machine-provable proposition subset within this paper’s $\phi$ . The relation to this framework is: system-of-systems engineering has in fact been solving the inverse problem, but lacks the unified criteria of $\kappa / \rho / \phi$ and $\lambda ^ { * } \leq \lambda _ { \mathrm { t i p } } ;$ its R (member-system interaction protocols) and C (member boundary constraints) are exactly instances of the $R , C$ components of the augmented quintuple.

Computational science (the methodological home of the forward/inverse duality). Program synthesis, inverse problems and regularization, formal methods, and complexity theory have each historically handled one facet of the inverse problem: program synthesis solves specifications inversely into code, inverse problems invert model parameters from observations, formal methods verify universal properties, and complexity theory adjudicates the boundaries of solvability. This paper gathers these traditions into one set of coordinates: abstract interpretation and coarse-graining on the forward side correspond to $\pi , \varepsilon ;$ program synthesis and optimization on the inverse side correspond to $J ( \lambda )$ ; and model checking on the verification side corresponds to $\phi .$ . The relation to this framework is: computational science provides the toolbox for the inverse problem, but has never taken “emergence” as the object of the inverse problem; this paper’s contribution is to aim the toolbox at this object and to provide the three pillars $\kappa / \rho / \phi$

## 6.2 The limits of engineerability: which inverse problems are solvable

The sharpest question from the computational-science perspective is: where is the inverse problem solvable and where is it not. The limits scattered across Sections 3.1, 3.2, 3.5, and Appendix B are collected here into four boundaries:

• Hard interval of rank and connectivity: $N ^ { * } ~ \geq ~ | Y |$ (rank lower bound) and $N ^ { * } \ \leq$ min $( e ^ { d } , d ^ { T _ { \mathrm { m a x } } } )$ (connectivity and propagation-time upper bound) sandwich the engineerable scale between two bounds; when the interval is empty, the inverse problem has no solution under that $( d , T _ { \operatorname* { m a x } } )$ , and one must change the design rather than tune parameters.

• The degree of control is measurable but hard to estimate in high dimensions: the mutual-information definition of Section 3.5 must use the MINE/InfoNCE variational lower bound in high dimensions, and the intervention-contrast proxy has a bounded absolute bias $( \leq p )$ but no uniform bound on its relative bias; this means the design coordinate κ is usable, but exact measurement of the degree of control itself is not guaranteed in high dimensions.

• ϕ is decidable but limited by state explosion: model checking is decidable on finite-state systems but grows exponentially in $\left| X \right|$ ; large systems can only select a subset $P _ { E }$ or use approximate verification, so $\phi$ is “partially decidable” rather than “fully decidable”.

• No global guarantee for the inverse optimization of $J ( \lambda )$ : the geometric weighting has a unique interior point under the exponential model, but in the non-convex case arg max is not unique; this paper reports robustness via a neutral-weight drift band rather than claiming global optimality.

Together these four boundaries constitute the “complexity map” of the engineerability of the inverse problem, corresponding to the forward failure domain of Section 6.3 (no identifiable local rules or no stable micro-to-macro mapping). See Figure 7 for the solvability map of the inverse problem.

Reverse-problem solvability map

![](images/1dff8dd94f14f8c80e40bb1da9b6ab0dc47a6eb4c431c23652ef30a9bb568013.jpg)  
Figure 7: Solvability map of the inverse problem. The horizontal axis is interaction density d and the vertical axis is the number of macroscopic states $| Y |$ , characterizing the feasible interval of the critical size $N ^ { * }$ . Boundaries: rank lower bound $N ^ { * } \geq | Y |$ (necessary condition for a linear readout); connectivity upper bound $N ^ { * } \leq e ^ { d } ;$ propagation-time upper bound $N ^ { * } \leq \bar { d } ^ { T _ { \mathrm { m a x } } }$ . The three lines bracket the feasible interval. The feasible region (light) lies below the two upper-bound curves; the infeasible region lies above the tighter upper bound and requires a design change rather than parameter tuning. The rank condition $N ^ { * } \geq | Y |$ is already implicitly satisfied once the design interval is chosen; an empty feasible region means the inverse problem has no solution under that operating condition. $T _ { \mathrm { m a x } }$ takes a typical value; this figure is schematic.

## 6.3 The failure domain

When a system has no identifiable local rules, or no stable mapping between micro and macro exists, the quintuple no longer applies. Among the three sources of intelligence, engineering optimization (supervised learning, RL policies) is designed top-down and is not emergence, hence outside the scope of this paper.

## 6.4 Ethical governance: value boundaries into C, human veto into U

In this framework, ethical governance is realized as design constraints on C and U: value boundaries are non-violable hard constraints within $C ,$ and the human veto is the highest-priority interruption signal within U. Ethical review can therefore be converted into a formal-verification problem over $C , U -$ for example, using CTL/LTL specifications to check the non-violability of the propositions in $P _ { E }$ and whether emergency stop has the highest priority within U.

Regarding the military background such as DARPA DICE mentioned in Section 1.3, this paper’s position is that the framework is a neutral analytical tool that can be used to examine various controlled-emergence systems including DICE; but the ethical tension faced by basic researchers under military funding — the conflict between the engineering urgency of “making desirable capabilities emerge” and the values of open science — must be explicitly acknowledged rather than set aside (Widder et al., 2024). This tension has an intellectual genealogy: nearly sixty years ago Good (1965) warned of the uncontrollable “intelligence explosion” of superintelligent machines; this paper’s $\kappa / \rho / \phi$ framework does not promise to eliminate this risk, but it provides an engineering language that turns “loss of control” from a metaphor into a measurable quantity. All work in this paper is public and simulation-oriented, with open-sourced code and data; the framework itself does not favour any deployment purpose, but its users bear the responsibility of reviewing the consequences of its application.

## 6.5 Open problems

This paper does not pursue a closed theory; the following problems are left for future work:

1. Formal experiment and the sensitivity of I, K: a three-task-parallel formal experiment with $M _ { \mathrm { r u n } } \geq 1 0 0 0$ , especially a collective-prediction task design that makes I, K sensitive to $\lambda ;$

2. Cross-task convergence of $c _ { 0 } { : }$ estimate $\widehat { c } _ { 0 }$ on at least three benchmark tasks and judge its universality by the dual condition of CI overlap $+ \ C V < 0 . 3$ ;

3. Three-level dynamics of the hierarchical ${ \widetilde { E } } { \mathrm { : } }$ mesoscopic control $U _ { \mathcal { M } }$ , mesoscopic constraints $C _ { \mathcal { M } }$ and the two-level phase transitions $N _ { 1 } ^ { * } , N _ { 2 } ^ { * }$ ;

4. Minimality theorem for the quintuple: currently a counterexample-style argument, still lacking a rigorous proof that each of the four elements is irreducible;

5. Practical estimation of the degree of control in high dimensions: a more robust high-dimensional estimator is needed (Section 3.5 has already analyzed the bias boundary of the interventioncontrast proxy).

The framework does not adjudicate the dispute over “real emergence versus statistical mirage” but provides a common language that makes that dispute comparable with engineering design under the same coordinates; this is itself one of the open problems to be tested by future work.

## 7 Conclusion

Intelligence is emergent. From neuronal populations to ant colonies to artificial networks after scale expansion, interdisciplinary research has confirmed that when many simple units interact under local rules, the whole can exhibit capabilities no single unit possesses. But for artificially emergent intelligence to serve people and enterprises, it must be controllable, and over-strong control collapses adaptability while over-weak control lets behaviour run away.

From the computational-science perspective this paper restates the tension as a pair of dual problems: the forward explanatory $R , D , \pi , W \to Y$ and the inverse engineering $T , C , \kappa _ { 0 } , \rho _ { 0 } , \phi _ { 0 } \to$ $( R , { \mathcal { D } } , \pi , U ) ^ { * }$ . Physics, neuroscience, and system-of-systems engineering have historically each occupied one end; the distinctiveness of the computational-science perspective lies in asking simulta neously about constructibility, computability, verifiability, and complexity boundaries.

What this paper provides is a starting point that can be tested and overturned, not a closed theory. The three contributions build one on another: at the language layer, the augmented quintuple E accommodates both forward explanation and inverse design; at the objective layer, $\kappa / \rho / \phi$ decomposes “engineering” into three independently measurable coordinates, with S as the intelligence bridge; at the principle layer, H1–H3 together with $J ( \lambda )$ and $\lambda ^ { * } \leq \lambda _ { \mathrm { t i p } }$ give the inversedesign space and the feasibility criterion. The four engineerability boundaries listed in Section 6.2 are an honest statement of the solvability of the inverse problem.

We do not pursue one-hundred-percent control, and we always preserve the self-organization headroom $S _ { \mathrm { s e l f - o r g } } > 0 .$ . If emergence were fully domesticated into a deterministic engineering object, “controlled emergence” would slide into becoming a synonym of “complex-systems engineering”, and the very distinctiveness of the concept of emergence would disappear.

## A Symbol table

Table 9: Symbol table.
<table><tr><td>Symbol</td><td>Meaning</td><td>First appearance</td><td>Scope</td></tr><tr><td> $E$ </td><td>Emergence system (augmented quintuple)</td><td>2.1</td><td>General</td></tr><tr><td> $X = \{ x _ { 1 } , \ldots , x _ { N } \}$ </td><td>Set of microscopic unit states</td><td>2.1</td><td>General</td></tr><tr><td> $N$ </td><td>Number of microscopic units</td><td>2.1</td><td>General</td></tr><tr><td> $N ^ { * }$ </td><td>Critical scale (phase-transition threshold)</td><td>4.1</td><td>H1</td></tr><tr><td> $R$ </td><td>Local interaction rules and topology</td><td>2.1</td><td>General</td></tr><tr><td> $d$ </td><td>Average connectivity per node (interaction density)</td><td>4.1</td><td>H1</td></tr><tr><td> $\mathcal { D }$ </td><td>Microscopic state-dynamics update function</td><td>2.1</td><td>General</td></tr><tr><td> $\mathcal { D } _ { i }$ </td><td>Update function of unit i</td><td>2.1</td><td>General</td></tr><tr><td> $Y$ </td><td>Macroscopic state-and-capability space</td><td>2.1</td><td>General</td></tr><tr><td> $| Y |$ </td><td>Number of effectively distinguishable macroscopic states of the readout space Y (number of bins of the run-trajectory support</td><td>4.1</td><td>H1, Appendix B</td></tr><tr><td> $\pi$ </td><td>set, see Appendix B) Coarse-graining / readout mapping from micro to macro</td><td>2.1</td><td>General</td></tr><tr><td> $U$ </td><td>Set of external control inputs</td><td>2.1</td><td>General</td></tr><tr><td> $W$ </td><td>Environmental perturbation</td><td>2.1</td><td>General</td></tr><tr><td> $T \subseteq Y$ </td><td>Target macroscopic-state neighbourhood</td><td>2.1</td><td>General</td></tr><tr><td> $C$ </td><td>Constraint set (set)</td><td>2.1</td><td>General</td></tr><tr><td> $y ( t ) = \pi ( x ( t ) )$ </td><td>Macroscopic trajectory</td><td>2.1</td><td>General</td></tr><tr><td> $\Psi$ </td><td>Hypothesized macroscopic dynamics</td><td>2.1</td><td>General</td></tr><tr><td> $\varepsilon$ </td><td>Macroscopic closure error  $\| \pi \circ \mathcal { D } - \Psi \circ \pi \|$ </td><td>2.1</td><td>General</td></tr><tr><td> $\widetilde { E }$ </td><td>Hierarchical augmented quintuple</td><td>2.4</td><td>Hierarchical extension</td></tr><tr><td> $\mathcal { M }$ </td><td>Set of mesoscopic structures (local groups / modules / formations)</td><td>2.4</td><td>Hierarchical extension</td></tr><tr><td> $\pi _ { 1 } { : } X  { \mathcal { M } } ,$   $\pi _ { 2 } { \colon } { \mathcal { M } } \to Y$ </td><td>Two-level readout;  $\pi = \pi _ { 2 } \circ \pi _ { 1 }$ </td><td>2.4</td><td>Hierarchical extension</td></tr><tr><td> $\mathcal { D } _ { \mathcal { M } }$ </td><td>Mesoscopic-layer dynamics</td><td>2.4</td><td>Hierarchical extension</td></tr><tr><td> $\Phi _ { I I T }$ </td><td>IIT integrated information (cited only in</td><td>3.4</td><td>IIT citation</td></tr><tr><td> $\mathfrak { F }$ </td><td>Section 3.4) Generic formal framework</td><td>2.2</td><td>Proposition</td></tr><tr><td> $H ( \cdot \mid \cdot )$ </td><td>(minimality-of-four-elements proposition) Conditional entropy</td><td>3.5</td><td>proof Degree-of-</td></tr><tr><td> $\mathcal { T } ( \cdot ; \cdot \mid \cdot )$ </td><td>Conditional mutual-information operator</td><td>3.5</td><td>control definition Degree-of-</td></tr><tr><td> $\operatorname { N o r m a l } ( \mu , \sigma ^ { 2 } )$ </td><td>Normal distribution</td><td>Appendix B</td><td>control definition Bayesian hierarchical</td></tr><tr><td> $\gamma _ { \mathrm { r e s } }$ </td><td>Louvain/Leiden resolution parameter</td><td>2.4</td><td>alternative Hierarchical</td></tr><tr><td> $n _ { \mathrm { m i n } }$ </td><td>DBSCAN minimum neighbourhood unit count</td><td>2.4</td><td>extension Hierarchical</td></tr><tr><td> $\theta _ { \mathrm { a l i g n } }$ </td><td>Formation mean-alignment-angle threshold</td><td>2.4</td><td>extension Hierarchical</td></tr><tr><td> $\varepsilon _ { 1 } , \varepsilon _ { 2 }$ </td><td>Two-level coarse-graining errors</td><td>2.4</td><td>extension Hierarchical</td></tr><tr><td> $N _ { 1 } ^ { * } , \ N _ { 2 } ^ { * }$ </td><td>Two-level critical scales: micro→meso and</td><td>4.1</td><td>extension H1 (hierarchical</td></tr><tr><td> $S$ </td><td>meso→macro Composite intelligence measure</td><td>3.4</td><td>note) Intelligence</td></tr><tr><td>A</td><td> $w _ { 1 } A + w _ { 2 } I + w _ { 3 } K$  Adaptive behaviour (task return)</td><td>3.4</td><td>measure Intelligence</td></tr><tr><td>I</td><td>Information integration (preferably effective information EI)</td><td>3.4</td><td>measure Intelligence</td></tr><tr><td> $K$ </td><td>Causal-modelling ability (intervention</td><td>3.4</td><td>measure Intelligence</td></tr><tr><td> $w _ { 1 } , w _ { 2 } , w _ { 3 }$ </td><td>prediction accuracy) Intelligence-measure weights</td><td>3.4</td><td>measure Intelligence</td></tr><tr><td>κ</td><td>Controllability</td><td>3.1</td><td>measure H2/H3</td></tr><tr><td> $\widehat { \kappa } _ { \mathrm { s t r i c t } }$ </td><td>Strict estimate of controllability</td><td>3.1</td><td>H2/H3</td></tr><tr><td> $\widehat { \kappa } _ { \mathrm { r e s i d e n c e } }$ </td><td>Relaxed estimate of controllability</td><td>3.1</td><td>H2/H3</td></tr><tr><td> $\tau _ { i }$ </td><td>First violation time of the i-th run</td><td>3.1</td><td>H2/H3</td></tr><tr><td> $\rho$ </td><td>Reproducibility</td><td>3.2</td><td>H2/H3</td></tr><tr><td> $\widehat { \rho }$ </td><td>Estimate of reproducibility</td><td>3.2</td><td>H2/H3</td></tr><tr><td> $\mu _ { i }$ </td><td>Empirical distribution of the i-th trajectory</td><td>3.2</td><td>H2/H3</td></tr><tr><td> $W _ { 2 }$ </td><td>2-Wasserstein distance</td><td>3.2</td><td>Hypothesis H2</td></tr><tr><td> $d _ { \mathrm { m a x } }$ </td><td>Diameter of the state space; by default the formulas and main text refer to the empirical diameter  $\widehat { \mathrm { d i a m } } ( Y )$  , and the theoretical</td><td>3.2</td><td>H2, ρ estimation</td></tr><tr><td> $\phi _ { E }$ </td><td>compactness discussion Conditional degree of formalizability</td><td>3.3</td><td>General</td></tr><tr><td> $P _ { \mathrm { l i b r a r y } }$ </td><td>Proposition template library (8 items)</td><td>3.3</td><td>General</td></tr><tr><td> $P _ { E }$ </td><td>Applicable proposition subset of system E</td><td>3.3</td><td>General</td></tr><tr><td> $P _ { \mathrm { p r o v a b l e } }$ </td><td>Machine-verifiable proposition subset</td><td>3.3</td><td>General</td></tr><tr><td> $\lambda$ </td><td>Constraint-strength scalar  $[ 0 , 1 ]$ </td><td>4.4</td><td>H2/H3</td></tr><tr><td> $c _ { 0 }$ </td><td>H1 fitting constant (task-dependent)</td><td>4.1</td><td>H1</td></tr><tr><td> $\lambda ^ { * }$ </td><td>Minimal constraint strength</td><td>4.4</td><td>H2/H3</td></tr><tr><td> $\lambda _ { \mathrm { t i p } }$ </td><td>Cooperation/trade-off inflection (between  $\kappa / \rho$ </td><td>4.4</td><td>H2/H3</td></tr><tr><td> $\kappa _ { 0 }$ </td><td>and A) Controllability specification threshold</td><td>4.4</td><td>H2/H3</td></tr><tr><td> $\alpha _ { r } , \beta _ { r } , \gamma _ { r }$ </td><td>Analytic-model rate parameters</td><td>4.4</td><td>H2/H3</td></tr><tr><td> $S _ { \mathrm { s e l f - o r g } }$ </td><td>Self-organization headroom</td><td>3.5</td><td>General</td></tr><tr><td> $T _ { \mathrm { m a x } }$ </td><td>Upper bound of the observation time window</td><td>3.1</td><td>General</td></tr><tr><td> $M _ { \mathrm { r u n } }$ </td><td>Number of independent runs</td><td>3.1</td><td>General</td></tr><tr><td>feasibility</td><td>Design margin  $\lambda _ { \mathrm { t i p } } - \lambda ^ { * }$ </td><td>4.5</td><td>H2/H3</td></tr><tr><td> $\widehat { c } _ { 0 }$ </td><td>Estimator of  $c _ { 0 }$  (bootstrap confidence interval)</td><td>Appendix B</td><td>H1</td></tr><tr><td> $p$ </td><td>Probability that "control is  $\mathrm { o n } ^ { \dag }$  in the degree-of-control proxy estimation</td><td>3.5</td><td>Section 3.5</td></tr><tr><td> $s , \widehat s$ </td><td>True source  $/$  group estimate in the collective-prediction task</td><td>5.1</td><td>Collective- prediction task</td></tr><tr><td> $\delta$ </td><td>Tolerance radius of the collective-prediction task (group estimate falls within the δ-neighbourhood of the true source)</td><td>Appendix C</td><td>Collective- prediction task</td></tr><tr><td> $\eta$ </td><td>Regularization weight</td><td>1.4</td><td>Inverse-problem well-posedness</td></tr><tr><td> $\Omega ( R , { \mathcal { D } } , \pi )$ </td><td>Complexity-penalty function</td><td>1.4</td><td>Inverse-problem</td></tr><tr><td> $B _ { k }$ </td><td>Set of microscopic units within the k-th mesoscopic unit</td><td>2.4</td><td>well-posedness Hierarchical</td></tr></table>

Symbol disambiguation. (1) $c _ { 0 }$ (the H1 constant) and λ (constraint strength) are diferent symbols; (2) $C$ (the constraint set) and $\lambda$ (constraint strength) are diferent symbols; (3) $\mathcal { D } .$ , Ψ, and $\phi$ are not confused visually or semantically; IIT’s $\Phi _ { I I T }$ appears only in the Section 3.4 citation.

Knowledge-type classification. Statements in this paper are graded by certainty, to avoid “working hypotheses” being misread as “theorems”.

Table 10: Knowledge-type classification.
<table><tr><td>Type</td><td>Content</td><td>Location</td></tr><tr><td>Theorem (proved)</td><td>Rank lower bound  $N ^ { * } \geq | Y | ,$   $N ^ { * } \leq e ^ { d }$  , propagation upper bound  $N ^ { * } \leq d ^ { T _ { \mathrm { m a x } } }$ </td><td>connectivity upper bound Appendix B</td></tr><tr><td>Proposition (proof sketch) Minimality of the four elements</td><td></td><td>2.2</td></tr><tr><td>Working hypothesis</td><td>H1a, H2, H3</td><td>Section 4</td></tr><tr><td>Exploratory conjecture</td><td>H1b  $N ^ { * } \approx c _ { 0 }$  log  $| Y | / \log ( 1 + d )$  , hierarchical correction 4.1  $N ^ { * } = \operatorname* { m a x } ( N _ { 1 } ^ { * } , N _ { 2 } ^ { * } )$ </td><td></td></tr><tr><td>Empirical statement</td><td>Executability of the pilot protocol (not a hypothesis Section 5 test)</td><td></td></tr></table>

## B Lower-bound exploration and numerical-validation protocol for hypothesis H1

Conjecture statement: in the augmented quintuple $E = ( X , R , \mathcal { D } , Y , \pi ; U , W , T , C )$ , let R define a random-graph topology with average degree d per node, π be a random-projection or nonlinear readout mapping, and D be bounded asynchronous update dynamics. If the space Y has a non-trivial attractor, this paper conjectures that in a specific regime the critical scale approximately obeys:

$$
N ^ { * } \approx c _ { 0 } \cdot \frac { \log | Y | } { \log ( 1 + d ) } ,\tag{17}
$$

where ≈ denotes an approximate fit within the regime listed in this appendix only, and is not a strict lower bound. $c _ { 0 } > 0$ is a fitting constant depending on the convergence rate of D, the readout type, and the graph structure. This paper makes no a priori assertion about the cross-task universality of $c _ { 0 }$

Part 1: reference bounds — one lower bound and two upper bounds. Clarification: random-graph connectivity and information-propagation time give upper bounds on N (necessary conditions for the global phase transition to be observable), and only the rank condition of linear readout gives a genuine lower bound (a necessary condition). Together the three bracket the feasible interval of $N ^ { * }$

Lower bound (necessary condition): the rank constraint. A linear readout $y = W x$ has rank at most min $( | Y | , N )$ ; for |Y | macroscopic states to be distinguishable, one needs

$$
N ^ { * } \geq | Y | .\tag{18}
$$

This is the genuine lower bound. Note that it is much stronger than log |Y | (when |Y | is large, $| Y | \gg \log | Y | )$ . If π is a strictly linear readout or a low-dimensional scalar readout (coverage rate, collision rate, etc.), the log |Y| form of H1b is trivial and the true lower bound |Y| is larger; the log |Y | form is meaningful only when π is a random projection or nonlinear readout (where Johnson–Lindenstrauss-type dimensionality reduction makes $N = O ( \log | Y | )$ possible).

Upper bound 1 (necessary condition for the global phase transition to be observable): connectivity. An Erdős–Rényi random graph with mean degree $d = N p$ is connected with high probability when $d \geq \log N$ , i.e.

$$
N ^ { * } \leq e ^ { d } .\tag{19}
$$

Connectivity is a necessary condition for global information propagation, and this bound shows that $N ^ { * }$ cannot grow without limit for fixed d. Note that when $N < 1 0 ^ { 2 }$ the connectivity constraint may still dominate; beyond $N \gtrsim 1 0 ^ { 2 }$ in the $d > e$ range, connectivity is no longer the bottleneck.

Upper bound 2 (necessary condition for the global phase transition to be observable): propagation time. On a graph of degree $d ,$ information can propagate to $O ( d ^ { T _ { \mathrm { m a x } } } )$ nodes within $T _ { \mathrm { m a x } }$ steps (fully parallel updates), hence

$$
N ^ { * } \leq d ^ { T _ { \operatorname* { m a x } } } .\tag{20}
$$

Under asynchronous updates propagation is slower, and this upper bound is tighter.

Combined feasible interval (by readout type). Under linear $/$ low-dimensional readout (coverage rate, collision rate, etc.), the rank lower bound dominates:

$$
\vert Y \vert \le N ^ { \ast } \le \operatorname* { m i n } ( e ^ { d } , \ d ^ { T _ { \operatorname* { m a x } } } ) .\tag{21}
$$

In this case the log $| Y |$ fitting form of H1b is trivial and inapplicable, and the true lower bound is $| Y |$

Under random-projection $/$ nonlinear readout (the four-premise regime of this appendix), Johnson–Lindenstrauss-type dimensionality reduction makes $N ^ { * } = O ( \log | Y | )$ a lower bound:

$$
O ( \log | Y | ) \leq N ^ { * } \leq \operatorname* { m i n } ( e ^ { d } , \ d ^ { T _ { \operatorname* { m a x } } } ) .\tag{22}
$$

H1b’s $N ^ { * } \approx c _ { 0 }$ log $| Y | / \log ( 1 + d )$ is explored only within this regime.

Non-emptiness condition: under linear readout the interval is non-empty if and only if $| Y | \leq$ min $( e ^ { d } , d ^ { T _ { \mathrm { m a x } } } )$ ; under random-projection / nonlinear readout it is non-empty if and only if $O ( \log | Y | ) \leq$ min $( e ^ { d } , d ^ { T _ { \mathrm { m a x } } } )$ If the parameters make $| Y | ~ > ~ \mathrm { m i n } ( e ^ { d } , d ^ { T _ { \mathrm { m a x } } } ) ~ ( \mathrm { e . g . } ~ \dot { d } ~ = ~ 4 , ~ | Y | ~ = ~ 1 0 ^ { \bar { 3 } }$ gives $e ^ { 4 } \ \approx \ 5 4 . 6 \ < \ 1 0 ^ { 3 } )$ , then the system cannot realize the phase transition under that $( d , T _ { \mathrm { m a x } } ) ;$ one must increase d or $T _ { \mathrm { m a x } }$ , or accept that the log $| Y |$ form is trivial (degenerating to the rank lower bound). Note that the upper bound $d ^ { T _ { \mathrm { { m a x } } } }$ grows exponentially with $T _ { \mathrm { m a x } } { \mathrm { : } }$ even if $e ^ { d } < | Y |$ as long as $T _ { \mathrm { m a x } }$ is large enough that $d ^ { T _ { \mathrm { m a x } } } \geq | Y |$ , the interval can still be non-empty. This also explains why the pairing of $d \in [ 4 , 3 2 ]$ and $| Y | \in [ 1 0 ^ { 3 } , 1 0 ^ { 6 } ]$ in the four premises of this appendix is not arbitrary.

The concise form $N ^ { * } \approx c _ { 0 } \log | Y | / \log ( 1 + d )$ is an exploratory fit of this interval in a specific regime, neither a lower bound nor an upper bound, and requires four premises: (i) π is a random projection or nonlinear readout; (ii) $| Y | \in [ 1 0 ^ { 3 } , 1 0 ^ { 6 } ] ; ( \mathrm { i i i } ) \ d \in [ 4 , 3 2 ] ; ( \mathrm { i v } )$ L is not close to 1. Note: the H1b fitted value may fall below $| Y |$ , which holds only when π is a random projection $/$ nonlinear readout and the rank lower bound does not apply; under linear readout the true $N ^ { * }$ is dominated by the rank lower bound $N ^ { * } \geq | Y |$ , and H1b does not apply.

## Part 2: numerical-validation protocol.

1. Platform and operationalization of $| Y | \colon | Y |$ denotes the number of efectively distinguishable macroscopic states of the readout space $Y$ (the number of bins of the support set of all run trajectories), not the number of task target states and not the theoretical cardinality of $Y$

2. Grid scan: $d \in \{ 2 , 4 , 8 , 1 6 , 3 2 \}$ , $N \in \{ 1 0 , 2 0 , 5 0 , 1 0 0 , 2 0 0 , 5 0 0 , 1 0 0 0 \}$ , with $M _ { \mathrm { r u n } } = 1 0 0$ per configuration.

3. Phase-transition detection: the critical $N ^ { * }$ at which macroscopic performance jumps from $< 5 0 \%$ $\mathrm { t o } > 9 0 \%$

4. Interval testing (corresponding to H1a and the strict reference bounds): in the linear-readout regime the measured $N ^ { * } ( d )$ should fall within $[ | Y |$ , min $( e ^ { d } , d ^ { T _ { \mathrm { m a x } } } ) ]$ , and if the measured point $N ^ { * } < | Y |$ , the rank lower bound is falsified; in the random-projection / nonlinear-readout regime, $N ^ { * }$ may fall below $| Y |$ (the rank lower bound does not apply), and the $O ( \log | Y | )$ lower bound should be used instead. In both regimes, if the measured point $N ^ { * } > \operatorname* { m i n } ( e ^ { d } , d ^ { T _ { \operatorname* { m a x } } } )$ , it means the system still exhibits a phase transition under fixed $d , T _ { \mathrm { m a x } }$ , and one must check whether the connectivity/propagation assumptions fail.

5. Equation fitting (corresponding to H1b, exploratory): least-squares fit of $N ^ { * } = c _ { 0 } \log | Y | / \log ( 1 +$ $d )$ , reporting $\widehat { c } _ { 0 }$ and $R ^ { 2 }$ ; a large fitting residual only means the concise form is not a good fit, and does not constitute a falsification of the reference bounds.

6. Cross-task convergence test of $c _ { 0 } { : }$ estimate $\widehat { c } _ { 0 }$ on at least three benchmark tasks (bootstrap 95% CI, resampling $B = 1 0 0 0 ) \mathrm { ; }$ ; dual-condition judgment: (i) the CIs of the tasks overlap; (ii) $C V = s d ( \widehat { c } _ { 0 } ) / m e a n ( \widehat { c } _ { 0 } ) < 0 . 3$ . The threshold 0.3 is an engineering rule of thumb, not a universal law. Even if the CV is large, c<sub>0</sub> can still be used for scale extrapolation within a family of similar tasks (same π type, same $\mathcal { D }$ class). The test is restricted: comparison is made only within families of similar tasks satisfying the four premises; tasks where $\pi$ is strictly linear or a low-dimensional scalar readout are excluded from the cross-task convergence comparison. Bayesian hierarchical alternative: as a complement to the frequentist dual condition, one may set $c _ { 0 } ^ { ( k ) } \sim \mathrm { N o r m a l } ( \mu _ { c _ { 0 } } , \sigma _ { c _ { 0 } } ^ { 2 } )$ and judge convergence by the posterior $P ( \sigma _ { c _ { 0 } } < 0 . 3 \mu _ { c _ { 0 } } \mid$ data); the two methods agree when the number of tasks is $\geq 3$ and the efect size is stable, and the Bayesian method is more robust in small samples.

7. Readout-type comparison: repeat the experiment for linear readout vs random-projection readout.

Attachment: derivation of the bias upper bound of the degree-of-control proxy. Write $H _ { 0 } : = H ( X _ { t + 1 } \mid X _ { t } , W )$ (the marginal entropy not conditioned on U), $h _ { 0 } = H ( X _ { t + 1 } \mid X _ { t } , W , U = 0 )$ $h _ { 1 } = H ( X _ { t + 1 } \mid X _ { t } , W , U = u ^ { * } )$ , with $h _ { 0 } \geq h _ { 1 }$ . Suppose U takes $u ^ { * }$ with probability p and 0 with $\mathrm { p r o b a b i l i t y } \ 1 - p ;$ then the entropy after conditioning on U is $h _ { U } : = p h _ { 1 } + ( 1 - p ) h _ { 0 }$ , and by the non-increasing property of conditional entropy $H _ { 0 } \geq h _ { U }$ . The strict degree of control is

$$
\frac { \mathscr { T } ( U ; X _ { t + 1 } \mid X _ { t } , W ) } { H _ { 0 } } = \frac { H _ { 0 } - h _ { U } } { H _ { 0 } } .\tag{23}
$$

The proxy degree of control (taking only the $U = 0$ branch) is

$$
\mathrm { d e g r e e ~ o f ~ c o n t r o l } _ { \mathrm { p r o x y } } = \frac { H _ { 0 } - h _ { 0 } } { H _ { 0 } } .\tag{24}
$$

The diference between the two is

$$
\frac { H _ { 0 } - h _ { U } } { H _ { 0 } } - \frac { H _ { 0 } - h _ { 0 } } { H _ { 0 } } = \frac { h _ { 0 } - h _ { U } } { H _ { 0 } } = \frac { p \left( h _ { 0 } - h _ { 1 } \right) } { H _ { 0 } } \in [ 0 , p ] ,\tag{25}
$$

(since $0 \leq h _ { 0 } - h _ { 1 } \leq H _ { 0 } )$ . That is, the proxy value never exceeds the strict value, and the absolute underestimation never exceeds $p ;$ but the relative underestimation $( \mathrm { s t r i c t - p r o x y } ) / \mathrm { s t r i c t } =$ $p ( h _ { 0 } - h _ { 1 } ) / ( H _ { 0 } - h _ { U } )$ has no uniform upper bound, and when the strict degree of control approaches 0 the relative underestimation can far exceed $p .$ Numerical example: with $p = 0 . 5 , H _ { 0 } = 1 . 0 , h _ { 0 } =$ $0 . 8 , h _ { 1 } = 0 . 5$ , we get $h _ { U } = 0 . 6 5$ , strict value = 0.35, proxy value = 0.20, and a relative underestimation of about $4 3 \% - \mathrm { a }$ special case for $p = 0 . 5$ and strict value = 0.35, not a general upper bound.

## C Simulation protocol and pseudocode

This appendix gives the skeletons of Algorithms 1–3; the line-by-line implementation, parameter annotations, random seeds, and unit-test scripts are hosted in the open-source repository’s README.

C.1 Experimental configuration. Python 3.10+, Mesa; three tasks (area coverage, foraging, collective prediction / distributed sensing); $\lambda \in [ 0 , 1 ]$ at 20 equally spaced points, $M _ { \mathrm { r u n } } = 1 0 0 0 \ ( 1 0 ^ { 4 }$ for safety-critical); the H1 $N \times d$ grid at $M _ { \mathrm { r u n } } = 1 0 0 ; T _ { \mathrm { m a x } } = 1 0 0 0$ ; the collective-prediction task’s δ is reported with its value and perturbed by ±50%.

C.2 Checklist bridging to the formal experiment. Relative to the pilot, the formal experiment must add: (i) three benchmark tasks in parallel to perform the cross-task convergence test of c ; (ii) a collective-prediction task that makes I, K sensitive to λ; (iii) $M _ { \mathrm { r u n } } \geq 1 0 0 0 ~ ( 1 0 ^ { 4 }$ for safety-critical) with reported confidence intervals and efect sizes; (iv) the $\lambda _ { \mathrm { t i p } } ^ { ( 0 ) }$ and drift band under the neutralweight configuration $\theta = \psi = 1 / 3 ;$ (v) the efect on the $\kappa ( \lambda )$ curve of perturbing $\delta$ (the radius of the δ-neighbourhood of the true source within which the group estimate falls in the collective-prediction task) by ±50%. The formal-experiment data will be published as an independent output of future work, and this paper does not rely on them as its empirical basis.

C.5 Reproducibility. Fixed random seeds; code/parameters/data open-sourced; model checking with PRISM, nuXmv, Lean/Coq; Sinkhorn via the POT library with $\varepsilon _ { \mathrm { s i n k } }$ reported; entropy estimation: KDE in low dimensions, k-NN (Kozachenko–Leonenko) in middle dimensions, MINE/InfoNCE in high dimensions.

C.6 Preregistration template. Register in the open repository before the formal experiment starts: (i) the concrete test forms and one-sided/two-sided nature of ${ \mathrm { H 1 a / H 2 / H 3 } } ;$ (ii) the sample-size calculation based on pilot efect sizes; (iii) the random-seed interval and the blind-analysis procedure; (iv) multiple-comparison correction (Bonferroni or FDR, preregistered by the number of tests); (v) the formal decision rule for “no inflection” in LOWESS (e.g. the number of derivative sign changes $< 1$ or AIC favouring the exponential model); (vi) the model-selection criterion (AIC/BIC comparing LOWESS and the exponential model); (vii) the falsification conditions and stopping rule.

C.7 Cross-domain validation protocols (design). Besides the three-task simulation, two validation protocols for the framework’s cross-domain applicability are listed for future work: (a) industrial-internet cross-factory coordination, with the simulation platform based on the AnyLogic model of (Wang et al., 2025b), scanning λ as the information-sharing strength, measuring the delivery cycle and single-factory utilization, with the falsification condition that $\lambda _ { \mathrm { t i p } }$ drifts by more than the scan step across stages; (b) multi-department emergency response, with the network structure generated per (Tao and Zhang, 2023), measuring the $\rho$ baseline and the drift of $\lambda _ { \mathrm { t i p } }$ across response stages, with the falsification condition $\rho < 0 . 5$ and no saturation plateau of $\kappa ( \lambda )$ . Neither protocol is within the scope of this paper’s pilot.

Table 11: Algorithm skeletons.
<table><tr><td>Algorithm</td><td>Input</td><td>Output</td><td>Main loop</td></tr><tr><td> $\mathrm { A 1 }$  measure  $\kappa / \rho / \phi$  and degree of control</td><td> $\mathrm { Q u i n t u p l e } , \lambda , M _ { \mathrm { r u n } } , P _ { E }$ </td><td> ${ \widehat K } _ { \mathrm { s t r i c t } } , { \widehat K } _ { \mathrm { r e s i d e n c e } } , \rho , \phi ,$  degree of control</td><td>Run contrast trajectories with  $U = u ^ { * }$  and  $U = 0 ;$  obtain κ from residence/violation,  $\rho$  from pairwise  $W _ { 2 } ,$  φ from the model-checking proportion,</td></tr><tr><td>A2 H2/H3 validation</td><td>20 λ values,  $M _ { \mathrm { r u n } } = 1 0 0 0 .$   $\kappa _ { 0 }$ </td><td>Three curves,  $\lambda _ { \mathrm { t i p } } , \lambda ^ { * }$  feasibility</td><td>ratio Call A1 per  $\lambda ; \mathrm { L O W E S S } +$  exponential  $\operatorname { f i t } ;$  solve the inflection analytically; output the falsification</td></tr><tr><td>A3 H1 scaling</td><td> $N \times d \mathrm { \ g r i d } , | Y | ,$   $M _ { \mathrm { r u n } } = 1 0 0$ </td><td> $N ^ { * } ( d )$  , feasible interval,  ${ \widehat { c } } _ { 0 } .$   $R ^ { 2 }$ </td><td>verdict Sweep N per d to find the phase-transition  $N ^ { * }$  ; first test whether the measured point falls within  $\left[ | Y | , \operatorname* { m i n } ( e ^ { d } , d ^ { T _ { \operatorname* { m a x } } } ) \right] \ ( \operatorname* { i f }$   $N ^ { * } < | Y |$  the rank lower bound is falsified), then</td></tr></table>

C.8 Pilot summary statistics and data table (to be completed by the formal experiment). The pilot ran only sparse λ points and did not perform the 20-point equally spaced scan; the table below lists only the summary numbers actually reported in this paper, without fabricating per-point means/SD/CI.

Table 12: Pilot summary statistics.
<table><tr><td>Task</td><td>N</td><td> $M _ { \mathrm { r u n } }$ </td><td> $\widehat { \kappa } _ { \mathrm { r e s i d e n c e } }$ </td><td> $\widehat { \rho }$ </td><td>Inflection</td><td>Power</td></tr><tr><td>Area coverage</td><td>30</td><td>50</td><td>0.944–0.947 (barely changes with λ)</td><td>High but insensitive to λ</td><td>Not yet entering a testable region</td><td> $< 0 . 2$ </td></tr><tr><td>Foraging</td><td>40</td><td>60</td><td>Decreases near  $\lambda \approx 0 . 3 6$ </td><td>Decreases near  $\lambda \approx 0 . 3 6$ </td><td> $\lambda \approx 0 . 3 6$ </td><td>(indication) 0.2-0.3</td></tr></table>

The formal experiment must be rerun at 20 equally spaced λ values with $M _ { \mathrm { r u n } } = 1 0 0 0$ , reporting per point the mean, SD, and 95% bootstrap CI of $\widehat \kappa _ { \mathrm { s t r i c t } } , \widehat \kappa _ { \mathrm { r e s i d e n c e } } , \widehat \rho , \widetilde A$ , together with the H2 power curve as a function of $M _ { \mathrm { r u n } } \in \{ 5 0 , 1 0 0 , 2 0 0 , 5 0 0 , 1 0 0 0 \}$ ; the pilot’s sparse points do not constitute a filling of this table.

## D Complete comparison table of alternative theoretical frameworks

Table 13: Comparison of alternative theoretical frameworks.
<table><tr><td>Framework</td><td>Degree of formalization</td><td>Measurability</td><td>Engineering interface</td><td>Emergence criterion</td><td>What this paper can borrow</td></tr><tr><td>Holland</td><td>Low (conceptual)</td><td>None</td><td>None</td><td>Constrained generating</td><td>Cross- disciplinary</td></tr><tr><td>Cybernetics</td><td>High (transfer function)</td><td>High (stability margin)</td><td>Mature</td><td>procedures Steady-state error</td><td>perspective Stability analysis</td></tr><tr><td>MAS</td><td>Medium (BDI logic)</td><td>Medium (task performance)</td><td>Rich</td><td>Multi-agent coordination</td><td>Algorithms and platforms</td></tr><tr><td>SoS</td><td>Medium (architectural principles)</td><td>Low (qualitative) Mature process</td><td></td><td>System-level capability emergence</td><td>Engineering methodology</td></tr><tr><td>AI alignment</td><td>Medium (reward Medium model)</td><td>(alignment score)</td><td>Active</td><td>Intent consistency</td><td>Constraint-set design</td></tr><tr><td>IIT</td><td> $\mathrm { H i g h } \left( \Phi _ { I I T } \right.$  definition)</td><td>Low (not computable on large systems)</td><td>None</td><td>Information integration  $\Phi _ { I I T } > 0$ </td><td>The I dimension of the intelligence measure (but EI</td></tr><tr><td>CAS</td><td>Low (conceptual)</td><td>Low</td><td>None</td><td>Fitness improvement</td><td>preferred) Evolution and adaptation</td></tr></table>

## Acknowledgements

We thank the relevant colleagues for the feedback and suggestions provided during the discussion of the framework. AI tools were used for English translation and language polishing during the writing of this paper.

## Data availability

The raw simulation data of the reduced feasibility pre-experiment (pilot) reported in Section 5 of this paper (area coverage $N = 3 0 , M _ { \mathrm { r u n } } = 5 0$ ; foraging $N = 4 0 , M _ { \mathrm { r u n } } = 6 0 )$ are reproducible and alone sufice to support the claim of “protocol executability”. The formal-experiment data $( M _ { \mathrm { r u n } } \ge 1 0 0 0$ including the collective-prediction task) will be published as an independent output of future work, and this paper does not rely on them as its empirical basis. The augmented-quintuple definition, simulation pseudocode, experimental protocol, parameter configuration, and pilot raw data will be open-sourced and hosted in a code repository.

## References

P. W. Anderson. More is diferent. Science, 177(4047):393–396, 1972. doi: 10.1126/science.177.4047. 393.

P. Bak, C. Tang, and K. Wiesenfeld. Self-organized criticality: an explanation of 1/f noise. Physical Review Letters, 59(4):381–384, 1987. doi: 10.1103/PhysRevLett.59.381.

D. S. Bernstein, R. Givan, N. Immerman, and S. Zilberstein. The complexity of decentralized control of markov decision processes. Mathematics of Operations Research, 27(4):819–840, 2002. doi: 10.1287/moor.27.4.819.297.

E. Bonabeau, M. Dorigo, and G. Theraulaz. Swarm Intelligence: From Natural to Artificial Systems. Oxford University Press, New York, 1999.

R. A. Brooks. A robust layered control system for a mobile robot. IEEE Journal on Robotics and Automation, 2(1):14–23, 1986. doi: 10.1109/JRA.1986.1087032.

R. A. Brooks. Intelligence without representation. Artificial Intelligence, 47(1–3):139–159, 1991. doi: 10.1016/0004-3702(91)90053-M.

T. B. Brown, B. Mann, N. Ryder, et al. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33 (NeurIPS 2020), pages 1877–1901, 2020.

C. T. Butts, R. M. Acton, and C. S. Marcum. Interorganizational collaboration in the hurricane katrina response. Journal of Social Structure, 13(1):1–36, 2012.

A. Clark. Being There: Putting Brain, Body, and World Together Again. MIT Press, Cambridge, MA, 1997.

I. D. Couzin, J. Krause, R. James, G. D. Ruxton, and N. R. Franks. Collective memory and spatial sorting in animal groups. Journal of Theoretical Biology, 218(1):1–11, 2002. doi: 10.1006/jtbi. 2002.3065.

J. Dahmann and K. J. Baldwin. Understanding the current state of us defense systems of systems engineering. INSIGHT, 11(2):26–29, 2008.

DARPA/I2O. Decentralized artificial intelligence through controlled emergence (dice). Broad Agency Announcement HR001126S0010, 2026.

D. C. Dennett. Consciousness Explained. Little, Brown, Boston, 1991.

Department of Defense. Mission engineering guide, version 2.0. Technical report, Ofice of the Under Secretary of Defense for Research and Engineering, 2023.

J. Ding, Y. Zheng, F. Xu, et al. Understanding emergence in complex systems using abductive ai. Nature Reviews Physics, 7(12):675–677, 2025. doi: 10.1038/s42254-025-00895-5.

G. M. Edelman. Neural Darwinism: The Theory of Neuronal Group Selection. Basic Books, New York, 1987.

M. Eigen and P. Schuster. The Hypercycle: A Principle of Natural Self-Organization. Springer, Berlin, 1979.

L. Fan and S. Wang. Evidence-based evaluation and optimization paths of public health emergency response plans in china. Frontiers in Public Health, 13:1673636, 2025. doi: 10.3389/fpubh.2025. 1673636.

Federal Aviation Administration. Unmanned aircraft system trafic management (utm) concept of operations, version 2.0. Technical report, Federal Aviation Administration, 2020.

M. J. Feigenbaum. Quantitative universality for a class of nonlinear transformations. Journal of Statistical Physics, 19(1):25–52, 1978. doi: 10.1007/BF01020332.

M. Gell-Mann. The Quark and the Jaguar: Adventures in the Simple and the Complex. W. H. Freeman, New York, 1994.

I. J. Good. Speculations concerning the first ultraintelligent machine. Advances in Computers, 6: 31–88, 1965.

X. Guo and W. B. Hu. Simulation study on the impact of emergency command on earthquake emergency rescue based on multi-agent modeling. Journal of Tsinghua University (Science and Technology), 65(6):1027–1039, 2025. doi: 10.16511/j.cnki.qhdxxb.2025.22.024.

H. Haken. Synergetics: An Introduction. Springer, Berlin, 2 edition, 1977.

D. O. Hebb. The Organization of Behavior: A Neuropsychological Theory. Wiley, New York, 1949.

E. Hoel. Causal emergence 2.0: quantifying emergent complexity, 2025.

J. Hofmann, S. Borgeaud, A. Mensch, et al. Training compute-optimal large language models, 2022.

D. R. Hofstadter. Gödel, Escher, Bach: An Eternal Golden Braid. Basic Books, New York, 1979.

J. H. Holland. Emergence: From Chaos to Order. Addison-Wesley, Reading, MA, 1998.

J. J. Hopfield. Neural networks and physical systems with emergent collective computational abilities. Proceedings of the National Academy of Sciences, 79(8):2554–2558, 1982. doi: 10.1073/pnas.79.8. 2554.

E. R. Hunt, S. Jones, and S. Hauert. Testing the limits of pheromone stigmergy in high-density robot swarms. Royal Society Open Science, 6(6):190225, 2019. doi: 10.1098/rsos.190225.

J. Kaplan, S. McCandlish, T. Henighan, et al. Scaling laws for neural language models, 2020.

S. A. Kaufman. The Origins of Order: Self-Organization and Selection in Evolution. Oxford University Press, New York, 1993.

J. A. S. Kelso. Dynamic Patterns: The Self-Organization of Brain and Behavior. MIT Press, Cambridge, MA, 1995.

J. Kuckling, R. Luckey, V. Avrutin, A. Vardy, A. Reina, and H. Hamann. Do we run large-scale multi-robot systems on the edge? more evidence for two-phase performance in system size scaling, 2023.

L. D. Landau. On the theory of phase transitions. Zhurnal Eksperimental’noi i Teoreticheskoi Fiziki, 7:19–32, 1937.

C. G. Langton, editor. Artificial Life: The Proceedings of an Interdisciplinary Workshop on the Synthesis and Simulation of Living Systems. Addison-Wesley, Redwood City, CA, 1989.

Y. LeCun, Y. Bengio, and G. Hinton. Deep learning. Nature, 521:436–444, 2015. doi: 10.1038/ nature14539.

G. H. Lewes. Problems of Life and Mind. Trübner, London, 1875.

C. Liu, X. Xiang, and L. Zheng. Value of information sharing in a multiple producers–distributor supply chain. Annals of Operations Research, 291:459–487, 2019. doi: 10.1007/s10479-019-03259-2.

W. Liu, W. H. Tan, and X. Liu. Human–Machine–Environment System Intelligence. Science Press, Beijing, 2024. ISBN 978-7-03-079006-4.

E. N. Lorenz. Deterministic nonperiodic flow. Journal of the Atmospheric Sciences, 20(2):130–141, 1963. doi: 10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2.

G. C. Mafettone, L. Liguori, E. Palermo, M. di Bernardo, and M. Porfiri. Mixed reality environment and high-dimensional continuification control for swarm robotics. IEEE Transactions on Control Systems Technology, 32(6):2484–2491, 2024. doi: 10.1109/tcst.2024.3430128.

M. W. Maier. Architecting principles for systems-of-systems. Systems Engineering, 1(4):267–284, 1998. doi: 10.1002/(sici)1520-6858(1998)1:4<267::aid-sys3>3.0.co;2-d.

B. B. Mandelbrot. The Fractal Geometry of Nature. W. H. Freeman, San Francisco, 1982.

H. R. Maturana and F. G. Varela. Autopoiesis and Cognition: The Realization of the Living. D. Reidel, Dordrecht, 1980.

R. M. May. Simple mathematical models with very complicated dynamics. Nature, 261:459–467, 1976. doi: 10.1038/261459a0.

M. Minsky. The Society of Mind. Simon and Schuster, New York, 1986.

D. P. Moynihan. The network governance of crisis response: Case studies of incident command systems. Journal of Public Administration Research and Theory, 19(4):895–915, 2009. doi: 10.1093/jopart/mun033.

J. Nunan. The timeline of information exchange: A service evaluation of london ambulance service front line communication. British Paramedic Journal, 4(4):40–49, 2020. doi: 10.29045/14784726. 2020.12.4.40.

Object Management Group. Omg systems modeling language (sysml), version 1.4. OMG Document formal/2015-06-03, 2015.

J. Pearl. Causality: Models, Reasoning, and Inference. Cambridge University Press, New York, 2 edition, 2009.

I. Prigogine and G. Nicolis. Self-Organization in Nonequilibrium Systems: From Dissipative Structures to Order through Fluctuations. Wiley, New York, 1977.

X. S. Qian. Create Systematics. Shanxi Science and Technology Press, Taiyuan, 2001.

M. Rubenstein, A. Cornejo, and R. Nagpal. Programmable self-assembly in a thousand-robot swarm. Science, 345(6198):795–799, 2014. doi: 10.1126/science.1254295.

S. Russell. Human Compatible: Artificial Intelligence and the Problem of Control. Viking, New York, 2019.

R. Schaefer, B. Miranda, and S. Koyejo. Are emergent abilities of large language models a mirage? In Advances in Neural Information Processing Systems 36 (NeurIPS 2023), pages 55565–55581, 2023.

Z. Tao and H. Zhang. The emergence of complex adaptive response networks in china: A case study of four disasters. Risk Analysis, 43(11):2223–2240, 2023. doi: 10.1111/risa.14121.

M. Tegmark. Life 3.0: Being Human in the Age of Artificial Intelligence. Alfred A. Knopf, New York, 2017.

G. Tononi. An information integration theory of consciousness. BMC Neuroscience, 5:42, 2004. doi: 10.1186/1471-2202-5-42.

A. M. Turing. Computing machinery and intelligence. Mind, 59(236):433–460, 1950. doi: 10.1093/ mind/LIX.236.433.

F. G. Varela, E. Thompson, and E. Rosch. The Embodied Mind: Cognitive Science and Human Experience. MIT Press, Cambridge, MA, 1991.

G. Vásárhelyi, C. Virágh, G. Somorjai, T. Nepusz, A. E. Eiben, and T. Vicsek. Optimized flocking of autonomous drones in confined environments. Science Robotics, 3(20):eaot3536, 2018. doi: 10.1126/scirobotics.aat3536.

A. Vaswani, N. Shazeer, N. Parmar, et al. Attention is all you need. In Advances in Neural Information Processing Systems 30 (NeurIPS 2017), pages 5998–6008, 2017.

J. E. Wang, D. L. Du, Z. Chen, and W. W. Tan. Assessment and future prospects of china’s low-altitude economy development index. Bulletin of Chinese Academy of Sciences, 2025a. doi: 10.3724/j.issn.1000-3045.20250520002.

L. X. Wang, Y. K. Liu, L. Zhang, T. Y. Lin, and L. H. Wang. Anylogic-based platform-enterprise collaborative scheduling simulation system for cloud manufacturing. Journal of System Simulation, 37(9):2225–2241, 2025b. doi: 10.16182/j.issn1004731x.joss.24-0393.

N. Wang, Z. Li, X. Liang, Y. Li, and F. Zhao. Cooperative target search of uav swarm with communication distance constraint. Mathematical Problems in Engineering, 2021:3794329, 2021. doi: 10.1155/2021/3794329.

J. Wei, Y. Tay, R. Bommasani, et al. Emergent abilities of large language models. Transactions on Machine Learning Research, 2022. arXiv:2206.07682.

M. Wertheimer. Experimentelle studien über das sehen von bewegung. Zeitschrift für Psychologie und Physiologie der Sinnesorgane, 61:161–265, 1912.

D. G. Widder, S. Gururaja, and L. Suchman. Basic research, lethal efects: military ai research funding as enlistment, 2024.

N. Wiener. Cybernetics: Or Control and Communication in the Animal and the Machine. MIT Press, Cambridge, MA, 1948.

K. G. Wilson. The renormalization group: Critical phenomena and the kondo problem. Reviews of Modern Physics, 47(4):773–840, 1975. doi: 10.1103/RevModPhys.47.773.

J. X. Wu, H. Zou, F. Zhang, et al. Mesoscopic computing paradigm and intelligent emergence. Science and Technology Review, 43(12):19–28, 2025.

W. H. Wu, Z. Z. Li, and W. X. Wang. Symmetry-breaking self-organization as a universal origin of complexity. International Journal of Bifurcation and Chaos, 34(6):2430012, 2024. doi: 10.1142/S021812742430012X.

Y. Yang, R. Rui, M. Li, C. Sun, T. Hastie, V. Conitzer, and Z. Wang. Mean field multi-agent reinforcement learning. In Proceedings of the 35th International Conference on Machine Learning (ICML 2018), pages 5567–5576. PMLR, 2018.

B. P. Zeigler, T. G. Kim, and H. Praehofer. Theory of Modeling and Simulation. Academic Press, San Diego, CA, 2 edition, 2000.

H. Zhang, Q. Yan, Y. Qin, S. Chen, and G. Zhang. A novel approach of resource allocation for distributed digital twin shop-floor. Information, 14(8):458, 2023. doi: 10.3390/info140800458.

J. Zhang and K. Liu. Neural information squeezer for causal emergence. Entropy, 25(1):26, 2023. doi: 10.3390/e25010026.

M. Zhang, Y. Wang, and Y. Zhang. Research on supply chain coordination decision making under the influence of lead time based on system dynamics. Systems, 12(1):32, 2024. doi: 10.3390/systems12010032.