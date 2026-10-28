# WCC: A Learning-Based Framework for Deciding When to Apply Cooperative Coevolution in Large-Scale Optimization

Wenjie Qiu, Hepei Li, Jianfeng Wang, Jun Zhang, Fellow, IEEE and Yue-Jiao Gong, Senior Member, IEEE

Abstract—Cooperative Coevolution (CC) and Non-Decomposition Algorithms (NDAs) are two primary approaches for large-scale global optimization (LSGO). Although both have achieved success, they exhibit markedly different performance across problem types and each faces approachspecific limitations, leaving the question of “when to use CC” unresolved. We address this gap with WCC, a novel learningbased framework that dynamically decides when to deploy CC in LSGO. WCC employs a streamlined design for efficient adaptive learning. We formulate the decision as a Markov decision process, designing a targeted feature set, a robust reward, and a tailored network architecture to provide the agent with rich information about the problem state and optimization progress. Our auto-regressive policy enables WCC to perform fine-grained, dynamic switching between CC and NDAs while conditionally allocating the computational budget without expert specifications. WCC shows strong competitiveness and superior generalization across problems, optimization horizons, and baseline optimizers. It realizes a “train-once and reuse” paradigm under a low budget, significantly curbing LSGO cost. Additionally, in-depth analyses yield insights not captured by existing theory, offering practical guidance on when and how to effectively deploy CC.

Index Terms—Large-scale Global Optimization, Cooperative Coevolution, Meta-Black-Box Optimization.

## I. INTRODUCTION

Large-scale global optimization (LSGO), typically involving over 1000 variables, represents a challenging class of blackbox optimization problems with widespread real-world applications [1]. The primary challenge of LSGO stems from the “curse of dimensionality,” where the rapidly expanding search space often renders conventional algorithms ineffective. Cooperative Coevolution (CC) based on the “divide-and-conquer” principle has become one of the most commonly adopted frameworks in LSGO [2], [3]. In addition, many classical algorithms have been extended to large-scale variants to tackle LSGO, which are referred to as non-decomposition algorithms (NDAs), in contrast to those developed under the CC.

Both CC and NDAs, despite their successes, face notable challenges. For CC, much research focuses on effective problem decomposition, where it often demonstrates competitive performance on conventional separable problems. However,

CC relies on strong problem assumptions (e.g., separability and high-precision evaluations). Existing decomposition strategies have inherent biases, fail to cover all forms of separability, and are prone to decomposition errors [4]. Once ideal decomposition is lost, performance drops sharply, making CC often ill-suited for real-world problems dominated by non-separability [5]. NDAs avoid complex variable decomposition and an increasing number of advanced variants have demonstrated superior performance compared to CCbased algorithms on increasingly complex problem instances. However, NDAs forgo exploiting structural priors that often exist in real problems, inflating the search space and computational cost, which can easily lead to optimization stagnation [6]. As dimensionality keeps growing, a bottleneck becomes inevitable—making decomposition unavoidable [7]. These differences and challenges result in highly divergent performance across problems and, in some cases, optimization stagnation in one approach—yielding outcomes that differ by several orders of magnitude.

The distinctly different optimization characteristics of both approaches have made the question of “when to use CC” a critical decision point. Many studies have theoretically analyzed the convergence of CC from perspectives such as Nash equilibrium [5]. However, the black-box nature, heterogeneity, and numerous unknown characteristics of LSGO make such theoretical analysis difficult and often case-specific. Some studies have attempted to combine these two approaches in practice based on expert knowledge. However, these hybrid methods further indicate that the question of when to use CC depends not only on the problem class but also on the optimization stage [8]. Applying CC at an inappropriate stage can severely degrade optimization performance, requiring analysis of phase-specific characteristics and NDA behavior, thereby compounding theoretical challenges [9]. To summarize, existing methods rely on heuristic, problem-specific switching schedules with limited cross-problem generalization, leaving the “when to use CC” decision an open bottleneck.

Recognizing that determining “when to use CC” is essentially a meta-level algorithm design problem rather than one concerning the internal mechanisms of CC, we introduce Meta-Black-Box Optimization (MetaBBO) [10], [11] into LSGO. As an emerging research direction in recent years, MetaBBO follows a bi-level structure, consisting of a metalayer and an optimization-layer, and aims to automate the design of BBO algorithms by constructing intelligent agents at the meta-layer as replacements for human experts. Many studies have demonstrated their excellent generalization ability and strong balance between exploration and exploitation [12], [13]. Furthermore, the actions of their learning-based agents often provide valuable insights to researchers [14], [15]. Since the decision of “when to use $\mathbf { C } \mathbf { C } ^ { \mathfrak { s } }$ is inherently not a oneshot choice, but a sequential decision-making process and a dynamic control problem under uncertainty, the appropriate decision relies heavily on the problem structure, the optimization stage and the current search status. To address these challenges, we formulate the process as a Markov decision process (MDP) and use deep reinforcement learning (DRL) to adaptively configure efficient algorithms for different LSGO problems. Instead of merely deciding whether to use CC or NDAs, the agent also determines the allocation of function evaluations (FEs) to CC and NDAs at each stage, enabling dynamic switching between the two approaches across optimization stages. To this end, we design an auto-regressive network as the agent architecture. Trained on a diverse set of LSGO problems, the agent learns to make optimal action selections at each optimization step based on a set of welldesigned features. We refer to this architecture as the Whento-use Cooperative Coevolution (WCC) framework. The main contributions of this paper are as follows:

• An intuitive yet effective framework to guide and analyze LSGO: We propose WCC, an innovative framework to learn when to use CC in LSGO. Framed as an MDP and trained via DRL, our agent reduces reliance on scarce expert knowledge and offers a new perspective on this longstanding question. Unlike heuristic designs, WCC offers fine-grained decisions that explicitly determine which optimization stages should employ CC or NDAs. In addition, the framework allows complex problems to be addressed with remarkable clarity, providing the possibility of drawing insightful conclusions.

• Streamlined design for efficient adaptive learning: Based on the characteristics of LSGO, we design a robust reward function and targeted feature set for WCC to provide the agent with rich signals about both the problem and the optimization progress. An auto-regressive policy is designed to first choose CC vs. NDA, then allocate FEs conditionally, enabling step-wise switching during optimization. A network architecture is carefully designed for this purpose, enabling the agent to adaptively navigate various scenarios.

• Comprehensive generalization performance: WCC demonstrates generalization not only across different LSGO problem groups, but also across optimization horizons and baseline algorithms: WCC can be trained once under low FEs and instantiated with different underlying optimizers, substantially reducing WCC’s training cost.

• Insightful analysis for future advancements: Thanks to its interpretable decision process, WCC further reveals clear patterns of when to deploy CC: as variable coupling increases, WCC increasingly favors NDAs, while its temporal preference shifts from early NDA to later CC on separable problems and from early CC to later NDA on non-separable ones. These observations indicate that effective CC deployment should jointly depend on problem structure and optimization stage rather than a fixed switching rule.

The remainder of this paper is organized as follows: Section II presents the methodology, Section III reports the results and insights, and Section IV concludes the paper.

## II. METHODOLOGY

WCC formulates CC/NDA deployment as an MDP, where the agent observes the current optimization state $s _ { i } ^ { t } ~ \in ~ \mathcal { S }$ and selects an action $a _ { t } ~ \in ~ { \cal A }$ that jointly determines the optimization mode and computational budget. The policy π<sub>θ</sub> is learned to maximize the expected discounted optimization performance over the target problem distribution ${ \mathcal { P } } ;$

$$
\pi ^ { * } \approx \underset { \pi _ { \theta } \in \Pi } { \arg \operatorname* { m a x } } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \gamma ^ { t - 1 } \operatorname { p e r f } \left( s _ { i } ^ { t } , \pi _ { \theta } ( s _ { i } ^ { t } ) \mid f _ { i } \right) ,\tag{1}
$$

where $\gamma$ is the discount factor. We optimize $\pi _ { \theta }$ using PPO [16].

## A. MDP Formulation

1) State: The state $s$ comprises four complementary feature groups: problem $S _ { p r o } \in \mathbb { R } ^ { 5 }$ , convergence $\bar { S } _ { c o n v } \in \mathbb { R } ^ { \bar { 6 } }$ resource $S _ { r e s } \in \mathbb { R } ^ { 3 }$ , and mode $S _ { m o d e } \in \mathbb { R } ^ { 2 }$ . Specifically, $ { \boldsymbol { S } } _ { p r o }$ captures problem dimensionality and structural information that characterize the suitability of CC and NDAs; $S _ { c o n v }$ describes the current optimization dynamics to support stagewise switching between their complementary search behaviors [9]; and $S _ { r e s }$ and $S _ { m o d e }$ characterize resource consumption and historical mode behavior, respectively, providing the contextual information needed for adaptive hybridization.

2) Action: The action A of WCC comprises two components: a mode selector $a _ { \mathrm { m o d e } }$ (choosing between NDA and $\mathrm { C C } )$ and a resource allocator $a _ { \mathrm { r e s } }$ (determining the FEs budget and switching timing). Assuming $a _ { \mathrm { r e s } }$ has κ levels, the composite action space is defined as:

$$
\begin{array} { r } { \mathcal { A } : = a _ { m o d e } \times a _ { r e s } , a _ { m o d e } \in \{ 0 , 1 \} , a _ { r e s } \in \{ 0 , \ldots , \kappa - 1 \} . } \end{array}\tag{2}
$$

3) Reward: To provide scale-robust and numerically stable feedback across heterogeneous LSGO problems, we adopt a log-ratio reward:

$$
r _ { t } = \ln \left( \frac { c o s t _ { t - 1 } ^ { * } + c } { c o s t _ { t } ^ { * } + c } \right) , \quad c = 1 0 ^ { - 8 } \operatorname* { m a x } \{ 1 , c o s t _ { t } ^ { * } , c o s t _ { t - 1 } ^ { * } \} ,\tag{3}
$$

where $c o s t _ { t - 1 } ^ { * }$ and cost<sup>∗</sup> denote the best costs at two consecutive steps. The log ratio mitigates sensitivity to objective scales and compresses large improvements, while the adaptive smoothing term c prevents numerical instability when $c o s t _ { t } ^ { * }$ approaches zero.

## B. Network Design

The network of WCC is composed of three components: Feature-Aware State Encoding, the Actor network, and the Critic network. Figure 1 shows the specific architecture.

![](images/ed0517cc90b7f7cb8a5901ac5123e7ab0c288f8bf1a765d531835f00278acfc4.jpg)  
Fig. 1. The network design of WCC.

1) Feature-Aware State Encoding: Because the relevance of state information varies across problem types and optimization stages, we introduce Feature-Aware State Encoding (FeatureSE) to adaptively reweight heterogeneous state features. Given the normalized state $\tilde { S } \ = \ \mathrm { L N } ( S )$ , FeatureSE learns both group-level gates $g$ over the four state categories and feature-level gates h over individual features. Their effects are combined through residual gating,

$$
w _ { j } = ( 1 - \lambda ) + \lambda g _ { k ( j ) } h _ { j } , \qquad z _ { j } = s _ { j } w _ { j } ,\tag{4}
$$

where $k ( j )$ denotes the group of feature j and λ controls the gating strength. This hierarchical reweighting allows WCC to emphasize different information according to the current optimization context while avoiding excessive feature suppression.

2) Actor: We adopt an autoregressive policy that first selects the optimization mode and then allocates resources conditioned on the selected mode:

$$
\pi ( a _ { \mathrm { m o d e } } , a _ { \mathrm { r e s } } \mid z ) = \pi _ { \theta _ { \mathrm { m o d e } } } ( a _ { \mathrm { m o d e } } \mid z ) \pi _ { \theta _ { \mathrm { r e s } } } ( a _ { \mathrm { r e s } } \mid z , a _ { \mathrm { m o d e } } ) ,\tag{5}
$$

where z is the FeatureSE-enhanced state representation. The mode head outputs a Bernoulli distribution over CC and NDA, while the resource head outputs a categorical distribution over candidate budgets using mode-specific representations to avoid cross-mode information leakage.

3) Critic: To support the autoregressive policy, we adopt a mode-conditioned critic $V ( z , a _ { \mathrm { m o d e } } )$ with two value heads $V _ { 0 } ( z )$ and $V _ { 1 } ( z )$ . For the resource policy, the chosen-mode value $V \left( z _ { t } , a _ { \mathrm { m o d e } ( t ) } \right)$ provides a baseline independent of $a _ { \mathrm { r e s } } ;$ for the mode policy, we use the mode-marginal baseline

$$
\bar { V } ( z _ { t } ) = \sum _ { m \in \{ 0 , 1 \} } \pi _ { \theta _ { \mathrm { m o d e } } } ( m \mid z _ { t } ) V ( z _ { t } , m ) ,\tag{6}
$$

which is independent of the realized mode action. These baselines reduce variance while preserving unbiased policygradient estimation.

## C. Workflow

After training, WCC follows a simple decomposition– decision–execution–update loop. The problem is first decomposed into subspaces using an existing decomposition strategy.

At each optimization step, the current state $s _ { t }$ is fed to the agent, which selects the optimization mode (CC or NDA) together with its computational budget. The selected optimizer is then executed under this budget, after which the best solution and state are updated. This process repeats until the evaluation budget is exhausted.

For each policy head $h \in \{ \mathrm { m o d e } , \mathrm { r e s } \}$ , the clipped PPO objective is

$$
\mathcal { L } _ { h } ^ { \mathrm { c l i p } } = - \mathbb { E } [ \operatorname* { m i n } ( \rho _ { h , t } A _ { h , t } , \mathrm { c l i p } ( \rho _ { h , t } , 1 - \epsilon , 1 + \epsilon ) A _ { h , t } ) ] .\tag{7}
$$

where $\rho _ { h , t }$ denotes the policy probability ratio and $A _ { h , t }$ is estimated by GAE using the corresponding critic baseline. Standard value and entropy regularization are further incorporated during training. Methodological details are provided in Appendix A.

## III. EXPERIMENTS

## A. Experimental Setup

1) Comparison algorithms: We instantiate WCC with CMA-ES [17] as the CC optimizer and MMES [7] as the NDA, denoted as WCC-MMES. We compare it with representative CC-based methods, including DG2 [18], CC-CMAES [19], ERDG [20], OEDG [21], LCC-CMAES [22]; NDAs including MMES [7], sep-CMA-ES [23], LM-MA-ES [6], FCMAES [24], and MetaES [25]; and hybrid methods CC-SHADE-ML [26] and HCC-ES [9].

2) Benchmarks: The primary benchmark is CEC2013LSGO [27], containing 15 problems spanning fully separable, partially separable, overlapping, and nonseparable structures. We treat F12 as non-separable because its chain-like structure induces 999 overlapping variables and subspaces. We further use the 1000-dimensional Auto Overlapping Benchmark (AOB) [9] to evaluate generalization across overlapping and non-separable structures.

3) Settings: WCC-MMES is trained for 90 epochs on representative training problems with $\mathrm { M a x F E s } = 1 \mathrm { E } { + } 0 6$ and is directly reused for testing without retraining. Each algorithm is independently evaluated over 30 runs, with comparison methods configured following their original settings. The training and unseen test problems may share the same separability category but differ in their underlying base functions. Detailed experimental settings are provided in Appendix B.

TABLE I  
COMPARING WCC-MMES WITH COMPARISON ALGORITHMS ON CEC2013LSGO AT 1.5E6 FES.
<table><tr><td rowspan="2">problem</td><td>WCC-based Algorithms</td><td colspan="5"></td><td colspan="5">NDAs</td><td colspan="2">Hybrid Algorithms</td></tr><tr><td>WCC-MMES DG2-CMAES</td><td>CC-CMAES</td><td>CC-based Algorithms LCC-CMAES</td><td></td><td>ERDG-CMAES</td><td>OEDG-CMAES</td><td>MMES</td><td>sep-CMA-ES</td><td>LM-MA-ES</td><td>FCMAES</td><td>MetaES</td><td>HCC-ES</td><td>CC-SHADE-SL</td></tr><tr><td rowspan="3"></td><td>2.16E-20</td><td>7.36E-20(+)</td><td>9.12E+06(+)</td><td>1.13E+05(+)</td><td>9.51E-19(+)</td><td>1.15E-18(+)</td><td>7.98E+05(+)</td><td>3.90E-17(+)</td><td>4.40E+06(+)</td><td>5.90E+06(+)</td><td>4.49E+11(+)</td><td>6.96E-20(+)</td><td>1.87E+05(+)</td></tr><tr><td>± 4.90E-20</td><td>± 8.97E-21</td><td>± 2.20E+06</td><td>± 2.54E+04</td><td>± 6.35E-19</td><td>± 3.47E-19</td><td>± 1.37E+05</td><td>± 7.95E-19</td><td>± 3.27E+05</td><td>± 8.20E+05</td><td>± 2.35E+10</td><td>± 8.87E-21</td><td>± 3.64E+05</td></tr><tr><td>2.13E+05 ± 4.72E+04</td><td>1.94E+05(-) ± 2.45E+04</td><td>1.18E+11(+) ± 5.01E+11</td><td>3.48E+10(+) ± 4.99E+09</td><td>1.49E+10(+) ± 7.32E+09</td><td>1.87E+10(+) ± 5.44E+09</td><td>4.43E+08(+) ± 7.68E+07</td><td>1.38E+11(+) ± 2.55E+10</td><td>7.33E+11(+) ± 1.14E+11</td><td>3.45E+09(+) ± 4.20E+08</td><td>2.37E+14(+) ± 5.40E+11</td><td>7.18E+06(+) ± 2.85E+06</td><td>1.25E+10(+) ± 1.10E+10</td></tr><tr><td>*5</td><td>1.65E+06</td><td>1.70E+06(+)</td><td>1.05E+07(+)</td><td>1.05E+07(+)</td><td>1.97E+06(+)</td><td>2.33E+06(+)</td><td>4.17E+06(+)</td><td>1.67E+06(+)</td><td>1.75E+06(+)</td><td>4.43E+06(+)</td><td>1.02E+08(+)</td><td>1.81E+06(+)</td><td>1.92E+06(+)</td></tr><tr><td>*8</td><td>± 3.26E+05 3.97E+09</td><td>± 3.39E+05 4.63E+09(+)</td><td>± 2.18E+06 5.04E+15(+)</td><td>± 6.31E+05 7.50E+14(+)</td><td>± 3.09E+05 4.58E+10(+)</td><td>± 4.04E+05 3.17E+15(+)</td><td>± 9.25E+05 4.21E+11(+)</td><td>± 1.52E+05 1.83E+15(+)</td><td>± 1.15E+05 2.90E+15(+)</td><td>± 8.33E+05 3.61E+12(+)</td><td>± 5.21E+06 2.49E+18(+)</td><td>± 3.12E+05 6.60E+09(+)</td><td>± 7.59E+06 2.13E+14(+)</td></tr><tr><td></td><td>± 5.21E+8</td><td>± 1.40E+09</td><td>± 7.47E+15</td><td>± 2.44E+14</td><td>± 1.99E+10</td><td>± 3.89E+15</td><td>± 4.35E+10</td><td>± 8.79E+14</td><td>± 1.47E+15</td><td>± 6.78E+11</td><td>± 1.38E+16</td><td>± 1.67E+08</td><td>± 2.55E+14</td></tr><tr><td>*9</td><td>1.70E+08 ± 5.78E+07</td><td>1.84E+08(+) ± 2.75E+07</td><td>7.62E+08(+) ± 2.18E+08</td><td>8.95E+08(+) ± 8.37E+07</td><td>1.69E+08(≈) ± 2.94E+07</td><td>1.57E+08(-) ± 4.02E+07</td><td>4.41E+08(+) ± 4.26E+07</td><td>1.41E+08(-) ± 1.70E+07</td><td>2.30E+08(+) ± 2.20E+07</td><td>5.70E+08(+) ± 9.23E+07</td><td>1.81E+10(+) ± 6.83E+07</td><td>1.51E+08(-) ± 2.76E+07</td><td>2.34E+08(+) ± 1.13E+08</td></tr><tr><td>*12</td><td>9.33E+02 ± 8.60E+01</td><td>1.78E+03(+) ± 4.15E+01</td><td>2.27E+03(+) ± 7.89E+02</td><td>1.47E+03(+) ± 1.92E+02</td><td>9.58E+02(+) ± 2.94E+00</td><td>2.06E+03(+) ± 2.29E+03</td><td>9.33E+02(≈) ± 6.74E+01</td><td>1.00E+03(+) ± 4.22E+01</td><td>8.61E+02(-) ± 3.41E+01</td><td>8.44E+02(-) ± 1.20E+02</td><td>9.19E+12(+)</td><td>1.72E+03(+)</td><td>2.10E+05(+) ± 6.60E+05</td></tr><tr><td>13</td><td>5.56E+06</td><td>9.05E+10(+)</td><td>6.70E+10(+)</td><td>4.33E+08(+)</td><td>1.70E+09(+)</td><td>1.56E+10(+)</td><td>9.36E+06(+)</td><td>2.20E+11(+)</td><td>6.76E+07(+)</td><td>2.71E+07(+)</td><td>± 3.21E+12 5.74E+18(+)</td><td>± 2.76E+01 2.88E+07(+)</td><td>3.15E+09(+)</td></tr><tr><td></td><td>± 8.57E+05 2.06E+03</td><td>± 3.45E+09 3.68E+03(+)</td><td>± 3.17E+10 2.16E+03(+)</td><td>± 2.51E+08 2.11E+03(+)</td><td>± 8.68E+08 1.33E+04(+)</td><td>± 5.55E+09 1.35E+04(+)</td><td>± 2.68E+06 1.44E+04(+)</td><td>± 1.13E+11 5.51E+03(+)</td><td>± 2.15E+06 5.82E+03(+)</td><td>± 8.49E+06 1.62E+04(+)</td><td>± 7.59E+17 1.15E+05(+)</td><td>± 2.49E+07 3.72E+03(+)</td><td>± 1.43E+09 2.54E+03(+)</td></tr><tr><td></td><td>± 4.53E+02</td><td>± 4.15E+02</td><td>± 1.39E+02 2.01E+01(≈)</td><td>± 1.30E+02</td><td>± 9.59E+02</td><td>± 7.05E+02</td><td>± 9.02E+02</td><td>± 1.59E+02</td><td>± 2.74E+02</td><td>± 8.52E+02</td><td>± 5.72E+03</td><td>± 3.71E+02</td><td>± 1.45E+03</td></tr><tr><td></td><td>2.00E+01 ± 2.31E-02</td><td>2.10E+01(+) ± 3.56E-02</td><td>± 2.50E-02</td><td>2.04E+01(+) ± 4.78E-02</td><td>2.16E+01(+) ± 5.01E-03</td><td>2.16E+01(+) ± 5.01E-03</td><td>2.01E+01(≈) ± 1.87E-01</td><td>2.16E+01(+)</td><td>2.16E+01(+)</td><td>2.09E+01(+)</td><td>2.16E+01(+)</td><td>2.10E+01(+)</td><td>2.09E+01(+)</td></tr><tr><td>6</td><td>9.95E+05</td><td>1.06E+06(+)</td><td>1.03E+06(+)</td><td>1.05E+06(+)</td><td>1.06E+06(+)</td><td>1.06E+06(+)</td><td>9.95E+05(≈)</td><td>± 6.06E-03 1.06E+06(+)</td><td>± 2.67E-03 1.06E+06(+)</td><td>± 1.10E-01 9.96E+05(≈)</td><td>± 7.97E-01 1.06E+06(+)</td><td>± 3.79E-02 1.06E+06(+)</td><td>± 8.31E-02 1.06E+06(+)</td></tr><tr><td></td><td>± 1.10E+01</td><td>± 9.01E+02</td><td>± 1.58E+04</td><td>± 2.00E+03</td><td>± 1.21E+03</td><td>± 1.31E+03</td><td>± 4.52E+01</td><td>± 1.64E+03</td><td>± 1.25E+03</td><td>± 2.18E+03</td><td>± 8.23E+03</td><td>± 8.28E+02</td><td>± 2.42E+03</td></tr><tr><td></td><td>4.41E+04</td><td>3.97E+04(-)</td><td>1.46E+09(+) ± 3.93E+09</td><td>5.36E+07(+) ± 2.38E+07</td><td>5.62E+09(+) ± 2.08E+09</td><td>7.86E+09(+)</td><td>7.56E+05(+)</td><td>6.79E+09(+)</td><td>8.23E+06(+)</td><td>4.23E+06(+)</td><td>1.09E+18(+)</td><td>1.69E+05(+)</td><td>3.91E+07(+)</td></tr><tr><td>10</td><td>± 2.37E+04 9.05E+07</td><td>± 1.45E+04 9.12E+07(+)</td><td>9.42E+07(+)</td><td>9.42E+07(+)</td><td>9.46E+07(+)</td><td>± 5.61E+09 9.50E+07(+)</td><td>± 6.72E+04</td><td>± 4.62E+09</td><td>± 9.59E+05</td><td>± 5.43E+05</td><td>± 7.24E+13</td><td>± 1.16E+04</td><td>± 4.14E+07</td></tr><tr><td></td><td>± 5.09E+02</td><td>± 2.92E+05</td><td>± 4.62E+05</td><td>± 4.46E+05</td><td>± 2.21E+05</td><td>± 2.12E+05</td><td>9.05E+07(≈) ± 2.04E+04</td><td>9.42E+07(+)</td><td>9.42E+07(+)</td><td>9.07E+07(+)</td><td>9.44E+07(+)</td><td>9.11E+07(+)</td><td>9.32E+07(+)</td></tr><tr><td>11</td><td>1.45E+07</td><td>5.74E+07(+)</td><td>5.06E+11(+)</td><td>4.81E+10(+)</td><td>1.81E+09(+)</td><td>1.63E+10(+)</td><td>1.82E+07(+)</td><td>± 2.08E+05</td><td>± 2.91E+05</td><td>± 2.83E+05</td><td>± 3.11E+05</td><td>± 4.91E+05</td><td>± 3.15E+05</td></tr><tr><td></td><td>± 1.63E+06</td><td>± 4.03E+07</td><td>± 4.85E+11</td><td>± 3.43E+10</td><td>± 9.20E+08</td><td>± 1.52E+10</td><td>± 2.44E+06</td><td>1.09E+11(+) ± 2.90E+10</td><td>7.61E+08(+) ± 1.39E+08</td><td>1.09E+08(+)</td><td>4.88E+19(+)</td><td>5.10E+07(+)</td><td>4.30E+19(+)</td></tr><tr><td>14</td><td>2.52E+07</td><td>1.46E+10(+)</td><td>1.12E+12(+)</td><td>7.31E+08(+)</td><td>2.02E+09(+)</td><td>3.10E+10(+)</td><td>3.31E+07(+)</td><td>1.22E+11(+)</td><td>3.27E+08(+)</td><td>± 1.91E+07 1.55E+08(+)</td><td>± 8.79E+18 3.41E+23(+)</td><td>± 2.79E+07 8.78E+07(+)</td><td>± 6.43E+09</td></tr><tr><td></td><td>± 3.74E+06</td><td>± 2.02E+09</td><td>± 3.36E+11</td><td>± 5.74E+08</td><td>± 4.69E+08</td><td>± 5.07E+10</td><td>± 5.57E+06</td><td>± 2.17E+11</td><td>± 3.44E+07</td><td>± 7.54E+06</td><td>± 7.81E+21</td><td>± 1.15E+07</td><td>1.44E+10(+)</td></tr><tr><td>15</td><td>2.31E+06</td><td>6.32E+13(+)</td><td>3.45E+08(+)</td><td>3.78E+07(+)</td><td>4.55E+08(+)</td><td>4.56E+08(+)</td><td>5.89E+06(+)</td><td>6.63E+08(+)</td><td>2.31E+07(+)</td><td>1.75E+07(+)</td><td>1.26E+18(+)</td><td>4.48E+13(+)</td><td>± 2.74E+10</td></tr><tr><td></td><td>± 1.47E+06</td><td>± 4.48E+13</td><td>± 3.08E+08</td><td>± 4.04E+06</td><td>± 3.74E+07</td><td>± 3.82E+07</td><td>± 9.36E+05</td><td>± 1.26E+08</td><td>± 6.26E+06</td><td>± 2.31E+06</td><td>± 6.94E+14</td><td>± 2.41E+13</td><td>1.22E+07(+) ± 5.76E+06</td></tr><tr><td>+/≈/-</td><td>NA</td><td></td><td>14/1/0</td><td>15/0/0</td><td>14/1/0</td><td>14/0/1</td><td>11/4/0</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>14/0/1</td><td>14/0/1</td><td>13/1/1</td><td>15/0/0</td><td>14/0/1</td><td></td></tr><tr><td></td><td></td><td>13/0/2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>15/0/0</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Problems marked with \* are representative problems used to form the training set.

## B. Comparison Analysis

1) Optimization Performance Comparison: Table I reports results on CEC2013LSGO at MaxFEs = 1.5E+06 (including 5E+05 for DG2). Symbols $" + ? , " - "$ , and “≈” indicate significantly better, worse, and comparable performance, respectively (Wilcoxon rank-sum test, $\alpha ~ = ~ 0 . 0 5 )$ , with W/T/L counts summarized in the final row.

Based on the results from Table I, we can analyze the following outcomes: a) WCC demonstrates superior performance: WCC-MMES consistently performs competitively against all other algorithms. In particular, DG2-CMAES achieves theoretically maximal decomposition accuracy, and MMES is among the most competitive NDAs. WCC’s strong performance thus highlights the power of hybridizing the two approaches. b) Data-driven approaches transcend the limitations of expert heuristics: With expert knowledge scarce, WCC-MMES makes finer-grained action selections in a data-driven manner. Its superior performance over hybrid algorithms further shows that WCC can transcend the limitations of expert heuristics. c) Effectively Learning the Strengths of CC and NDAs: WCC-MMES consistently excels on problems favoring both CC-based algorithms (e.g., F1, F5) and MMES (e.g., F13–F15), demonstrating its ability to effectively integrate the strengths of CC and NDAs.

2) Generalization Experiment: We evaluate WCC-MMES generalization across three settings by directly testing the trained policy without retraining.

a) Generalization across Problems: In Table I, 8 of the 15 test problems were unseen during training, yet WCC-MMES remained competitive; notably, its strong performance on F15—a complex non-separable problem—underscores its promising generalization ability. To further validate WCC-MMES’s competitiveness on overlapping and non-separable problems and its generalization across problem instances, we generated 16 new AOB problems and tested them under MaxFEs = 1.5E+06. Figure 2 shows that WCC-MMES is competitive on most problems, outperforming both MMES and DG2-CMAES. Its strength on non-separable problems is particularly notable; for example, in the Rastrigin family, WCC-MMES lags on overlapping variants but still excels on non-separable instances, further confirming its ability to tackle complex non-separable problems.

The symbols $\cdot _ { + } , \cdot _ { - } ,$ , and ‘≈’ signify that WCC-MMES is significantly better than, worse than, or comparable to its competitor, respectively.  
![](images/0bbaa1f16753cd85650c2f355a241096b02459e91ac524dc0ab09cbac48279a7.jpg)  
Fig. 2. Comparison of WCC-MMES and baselines on AOB at 1.5E6 FEs. Radial values are min–max normalized scores, with larger values indicating better relative performance.

b) Generalization across Optimization Horizons: Although trained with only MaxFEs = 1E+06, WCC-MMES remains highly competitive when directly evaluated at 3E+06 FEs without retraining. It achieves +/-/≈ records of 13/0/2, 15/0/0, 12/3/0, and 14/0/1 against DG2-CMAES, LCC-CMAES, MMES, and HCC-ES, respectively. The advantage is also substantial on several problems; for example, on F15, WCC-MMES reduces the final cost from the order of $1 0 ^ { 1 3 }$ achieved by DG2-CMAES and HCC-ES to 10<sup>5</sup>, corresponding to nearly eight orders of magnitude improvement. These results demonstrate strong generalization across optimization horizons. Complete results are provided in Appendix B.

![](images/ab612c6146b411e6e1da13cb1b18595d45d5389a44aff23e0f667eece660437e.jpg)  
Fig. 3. Generalization experiment of WCC across baseline algorithms.

c) Generalization across Baseline Algorithms: To assess whether WCC generalizes beyond MMES, we directly replace the NDA component in the trained WCC-MMES with Sep-CMAES, LM-MA-ES, and FCMAES, yielding three WCC variants without retraining. As shown in Figure 3, these variants outperform their corresponding standalone NDAs on most CEC2013LSGO problems at MaxFEs = 1E+06, often by large margins. This transferability stems from WCC’s optimizer-agnostic state and action design, which preserves the same meta-level mode-selection and budget-allocation problem across different NDAs.

## C. Ablation Study

To further examine where the performance of WCC-MMES comes from, we conduct a comprehensive ablation study from four aspects: action space, state space, reward function, and network architecture.

1) Experimental Protocol: We ablate WCC from four aspects: action, state, reward, and network design. For the action space, we randomize or fix either the mode or resource decision to examine the necessity of jointly learning both components. For the state space, each of the four feature groups $\{ S _ { p r o } , S _ { c o n v } , S _ { r e s } , S _ { m o d e } \}$ is removed individually. For the reward, the proposed log-ratio formulation is replaced with Difference, Sparse Sign, and Linear alternatives. For the network, we remove FeatureSE, replace the autoregressive policy with a simple MLP, or remove both. All variants follow the same training and evaluation protocol, and their final costs are min–max normalized within each category before averaging across problems.

2) Ablation Analysis: Figure 4 confirms the contribution of all four design components. For the action space, randomizing or fixing either decision degrades performance, showing the importance of jointly learning mode selection and resource allocation. For the state space, removing any feature group causes a clear drop, with convergence features having the largest impact, indicating that the four groups provide complementary information. For the reward, all alternative formulations underperform the proposed log-ratio reward, highlighting the importance of scale-robust and numerically stable feedback. For the network, removing FeatureSE or replacing the autoregressive policy with a simple MLP consistently worsens performance, with the largest degradation occurring

![](images/795cfab7a62b69622bb156a2d50fad448ee63491318136728b4d30b141007493.jpg)  
Fig. 4. Ablation study, including (a) action space ablation, (b) state space ablation, (c) reward function ablation, and (d) network architecture ablation.  
when both are removed, validating the joint roles of adaptive state reweighting and mode-conditioned resource allocation. Detailed ablation configurations and further analysis are provided in Appendix B.

## D. Further In-depth Analysis

Understanding when WCC selects CC is a central focus of this work. We divide each optimization run into four equal intervals and report the mean proportion of NDA selections in Figure 5, revealing clear patterns across both problem structures and optimization stages.

Across problem structures, WCC increasingly favors NDAs as variable coupling becomes stronger. CC dominates on fully and partially separable problems by exploiting decomposable structures, whereas NDA usage increases on overlapping problems and becomes dominant on non-separable ones, where decomposition is less effective. Across optimization stages, WCC generally favors NDAs early and gradually shifts toward CC on separable and overlapping problems; in contrast, on non-separable problems, CC is used mainly for limited early exploration, while NDAs dominate the middle and later stages. These results show that the appropriate use of CC depends jointly on problem structure and optimization stage rather than on a fixed switching rule.

Individual functions may nevertheless deviate from these category-level trends. For example, although F3 is categorized as fully separable, its exact separability is preserved only under one-dimensional decomposition; the 50-dimensional grouping used here therefore limits the effectiveness of CC, causing WCC to increasingly favor NDAs in later stages, similarly to non-separable problems. A similar pattern appears on F6. This further indicates that WCC adapts its mode selection to the observed optimization context rather than relying solely on nominal separability.

![](images/20eb129352de70505862c5d26dda2372d3e068024a0b18455b4ef395fb434b63.jpg)  
Fig. 5. Proportion of NDA selections across problem types and optimization stages. Each cell reports the mean proportion of $a _ { \mathrm { m o d e } } = \mathrm { N D A }$ , where darker colors indicate more frequent NDA usage. Rows denote four equal intervals divided by MaxFEs, and columns denote test functions grouped by separability.

## IV. CONCLUSION AND FUTURE WORK

We proposed WCC, a pioneering framework that learns when to use CC in LSGO, concurrently optimizing the resource allocation. WCC enabled fine-grained, dynamic selection and effective hybridization of CC and NDAs without human expertise. Extensively evaluated, WCC exhibited strong competitiveness—especially on non-separable problems, where both CC and NDAs alone typically struggle. Moreover, WCC demonstrated generalization across problems, optimization horizons, and baseline optimizers, enabling trainonce under a low FE budget and reuse with different underlying optimizers to mitigate LSGO’s computational cost. Finally, ablation studies and in-depth analyses yielded insights that go beyond existing theoretical understanding, offering practical guidance on when and how to deploy CC.

By shifting the focus from how to decompose to when to deploy CC, WCC opens a new direction for CC research. Future work will explore online separability characterization and adaptive selection among multiple CC and NDA strategies.

## REFERENCES

[1] M. N. Omidvar, X. Li, and X. Yao, “A review of population-based metaheuristics for large-scale black-box global optimization—part i,” IEEE Transactions on Evolutionary Computation, vol. 26, no. 5, pp. 802–822, 2021.

[2] X. Ma, X. Li, Q. Zhang, K. Tang, Z. Liang, W. Xie, and Z. Zhu, “A survey on cooperative co-evolutionary algorithms,” IEEE Transactions on Evolutionary Computation, vol. 23, no. 3, pp. 421–441, 2018.

[3] W. Qiu, Z. Weng, Y.-J. Gong, W.-N. Chen, and J. Zhang, “Unicc: A unified coevolutionary architecture with divergence-speedup modeling for large-scale global optimization,” IEEE Transactions on Evolutionary Computation, 2026.

[4] M. M. Komarnicki, M. W. Przewozniczek, H. Kwasnicka, and K. Walkowiak, “Incremental recursive ranking grouping for large-scale global optimization,” IEEE Transactions on Evolutionary Computation, vol. 27, no. 5, pp. 1498–1513, 2022.

[5] Q. Duan, C. Shao, G. Zhou, H. Yang, Q. Zhao, and Y. Shi, “Cooperative coevolution for non-separable large-scale black-box optimization: Convergence analyses and distributed accelerations,” Applied Soft Computing, vol. 166, p. 112232, 2024.

[6] I. Loshchilov, T. Glasmachers, and H.-G. Beyer, “Large scale black-box optimization by limited-memory matrix adaptation,” IEEE Transactions on Evolutionary Computation, vol. 23, no. 2, pp. 353–358, 2018.

[7] X. He, Z. Zheng, and Y. Zhou, “Mmes: Mixture model-based evolution strategy for large-scale optimization,” IEEE Transactions on Evolutionary Computation, vol. 25, no. 2, pp. 320–333, 2020.

[8] F. Van den Bergh and A. P. Engelbrecht, “A cooperative approach to particle swarm optimization,” IEEE transactions on evolutionary computation, vol. 8, no. 3, pp. 225–239, 2004.

[9] W. Qiu, H. Guo, Z. Ma, and Y.-J. Gong, “A novel two-phase cooperative co-evolution framework for large-scale global optimization with complex overlapping,” arXiv preprint arXiv:2503.21797, 2025.

[10] Z. Ma, H. Guo, Y.-J. Gong, J. Zhang, and K. C. Tan, “Toward automated algorithm design: A survey and practical guide to meta-black-boxoptimization,” IEEE Transactions on Evolutionary Computation, 2025.

[11] W. Qiu, H. Guo, Z. Ma, J. Zhang, and Y.-J. Gong, “Automated algorithm design for black-box optimization: Progress and challenges,” Chinese Journal of Computers, vol. 49, no. 4, pp. 855–884, 2026.

[12] Y. Song, Y. Wu, Y. Guo, R. Yan, P. N. Suganthan, Y. Zhang, W. Pedrycz, S. Das, R. Mallipeddi, O. S. Ajani et al., “Reinforcement learningassisted evolutionary algorithm: A survey and research opportunities,” Swarm and Evolutionary Computation, vol. 86, p. 101517, 2024.

[13] W. Qiu, Z. Wang, H. Fang, Z. Ma, and Y.-J. Gong, “A learningbased cooperative coevolution framework for heterogeneous large-scale global optimization,” in Proceedings of the Genetic and Evolutionary Computation Conference, 2026, pp. 938–947.

[14] Q. Zhao, T. Liu, B. Yan, Q. Duan, J. Yang, and Y. Shi, “Automated metaheuristic algorithm design with autoregressive learning,” IEEE Transactions on Evolutionary Computation, 2024.

[15] W. Yi, R. Qu, L. Jiao, and B. Niu, “Automated design of metaheuristics using reinforcement learning within a novel general search framework,” IEEE Transactions on Evolutionary Computation, vol. 27, no. 4, pp. 1072–1084, 2022.

[16] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[17] N. Hansen and A. Ostermeier, “Completely derandomized selfadaptation in evolution strategies,” Evolutionary computation, vol. 9, no. 2, pp. 159–195, 2001.

[18] M. N. Omidvar, M. Yang, Y. Mei, X. Li, and X. Yao, “Dg2: A faster and more accurate differential grouping for large-scale black-box optimization,” IEEE Transactions on evolutionary computation, vol. 21, no. 6, pp. 929–942, 2017.

[19] J. Liu and K. Tang, “Scaling up covariance matrix adaptation evolution strategy using cooperative coevolution,” in International Conference on Intelligent Data Engineering and Automated Learning. Springer, 2013, pp. 350–357.

[20] M. Yang, A. Zhou, C. Li, and X. Yao, “An efficient recursive differential grouping for large-scale continuous problems,” IEEE Transactions on Evolutionary Computation, vol. 25, no. 1, pp. 159–171, 2020.

[21] M. Tian, M. Chen, W. Du, Y. Tang, and Y. Jin, “An enhanced differential grouping method for large-scale overlapping problems,” IEEE Transactions on Evolutionary Computation, 2024.

[22] H. Guo, W. Qiu, Z. Ma, X. Zhang, J. Zhang, and Y.-J. Gong, “Advancing cma-es with learning-based cooperative coevolution for scalable optimization,” arXiv preprint arXiv:2504.17578, 2025.

[23] R. Ros and N. Hansen, “A simple modification in cma-es achieving linear time and space complexity,” in International conference on parallel problem solving from nature. Springer, 2008, pp. 296–305.

[24] Z. Li, Q. Zhang, X. Lin, and H.-L. Zhen, “Fast covariance matrix adaptation for large-scale black-box optimization,” IEEE transactions on cybernetics, vol. 50, no. 5, pp. 2073–2083, 2018.

[25] R. Lange, T. Schaul, Y. Chen, T. Zahavy, V. Dalibard, C. Lu, S. Singh, and S. Flennerhag, “Discovering evolution strategies via meta-black-box optimization,” in Proceedings of the companion conference on genetic and evolutionary computation, 2023, pp. 29–30.

[26] A. Vakhnin and E. Sopov, “A novel self-adaptive cooperative coevolution algorithm for solving continuous large-scale global optimization problems,” Algorithms, vol. 15, no. 12, p. 451, 2022.

[27] X. Li, K. Tang, M. N. Omidvar, Z. Yang, and K. Qin, “Benchmark functions for the CEC’2013 special session and competition on largescale global optimization,” Evolutionary Computation and Machine Learning Group, RMIT University, Melbourne, Australia, Technical Report, 2013.

## APPENDIX A METHODOLOGY

## A. State Representation

WCC represents the optimization context with 16 features, grouped into problem $\left( { { s _ { 1 } } - { s _ { 5 } } } \right)$ , convergence $\left( { { s _ { 6 } } - { s _ { 1 1 } } } \right)$ , resource $( s _ { 1 2 } \mathrm { - } s _ { 1 4 } )$ , and mode-history $\left( { { s _ { 1 5 } } - { s _ { 1 6 } } } \right)$ information. Table I lists their complete definitions. Here D is the problem dimension, Θ is the dependency structure matrix (DSM), S is the subspace list, and $\Omega _ { i }$ is the variable set of subspace i. The bestso-far objective value is denoted by $c o s t _ { t } ^ { * }$ , while sumFEs counts consumed function evaluations. The following definitions expand the trajectory-based and mode-history features that require more than a single expression.

1) Low-frequency energy ratio $( s _ { 7 } ) \colon$ For an observed bestcost sequence $\{ c o s t _ { k } ^ { * } \} _ { k = 0 } ^ { T - 1 }$ , compute the discrete Fourier transform and its normalized amplitude:

$$
\hat { C } _ { k } = \sum _ { u = 0 } ^ { T - 1 } c o s t _ { u } ^ { * } e ^ { - \mathrm { { i } } 2 \pi k u / T } , \qquad A _ { k } = \frac { | \hat { C } _ { k } | } { T } .\tag{1}
$$

Retain the positive-frequency indices $\mathcal { K } = \{ 0 , \ldots , \lfloor T / 2 \rfloor - 1 \}$ With $f _ { k } = k / T$ and cutoff $f _ { c } = 0 . 1$ , define

$$
\begin{array} { c } { { E _ { l o w } = \displaystyle \sum _ { k \in \mathcal { K } } A _ { k } , \qquad E _ { t o t } = \displaystyle \sum _ { k \in \mathcal { K } } A _ { k } , } } \\ { { f _ { k } \le f _ { c } } } \\ { { s _ { 7 } = \displaystyle \left\{ E _ { l o w } / E _ { t o t } , \quad E _ { t o t } > 0 , \right. } } \\ { { \displaystyle 0 , \qquad } } & { { E _ { t o t } = 0 . } } \end{array}\tag{2}
$$

The ratio summarizes the low-frequency spectral mass of the observed cost trajectory; the historical definition sums amplitudes.

2) First-order convergence trend $( s _ { 8 } ) \colon$ For the available trajectory $\{ c o s t _ { k } ^ { * } \} _ { k = 0 } ^ { t - 1 } .$ , let $g _ { k } ~ = ~ \nabla c o s t _ { k } ^ { * }$ be its numerical gradient and $\bar { g } = \mathrm { m e a n } ( \{ g _ { k } \} _ { k = 0 } ^ { t - 1 } )$ . The feature is

$$
s _ { 8 } = - \operatorname { t a n h } ( \lambda _ { 1 } \bar { g } ) , \qquad \lambda _ { 1 } = 0 . 5 .\tag{3}
$$

The sign convention makes decreasing costs correspond to positive values, while the hyperbolic tangent limits the scale of the trend statistic.

3) Second-order convergence trend $( s _ { 9 } ) \colon$ For interior points, use

$$
h _ { k } = c o s t _ { k + 1 } ^ { * } - 2 c o s t _ { k } ^ { * } + c o s t _ { k - 1 } ^ { * } , \quad k = 1 , \ldots , t - 2 .\tag{4}
$$

The two boundary expressions are

$$
\begin{array} { c } { h _ { 0 } = 2 c o s t _ { 0 } ^ { * } - 5 c o s t _ { 1 } ^ { * } + 4 c o s t _ { 2 } ^ { * } - c o s t _ { 3 } ^ { * } , } \\ { h _ { t - 1 } = 2 c o s t _ { t - 1 } ^ { * } - 5 c o s t _ { t - 2 } ^ { * } + 4 c o s t _ { t - 3 } ^ { * } - c o s t _ { t - 4 } ^ { * } . } \end{array}\tag{5}
$$

With $\bar { h } = \mathrm { m e a n } ( \{ h _ { k } \} _ { k = 0 } ^ { t - 1 } )$ , set

$$
s _ { 9 } = - \operatorname { t a n h } ( \lambda _ { 2 } \bar { h } ) , \qquad \lambda _ { 2 } = 0 . 5 .\tag{6}
$$

These boundary formulas require at least four trajectory values.

4) Robust variation of cost changes $( s _ { 1 0 } ) \colon$ Use the most recent window of length $L = 1 2 8 ,$ with $\tau _ { 0 } = \operatorname* { m a x } ( 1 , t - L )$ and define $\Delta c o s t _ { k } = c o s t _ { k } ^ { * } - c o s t _ { k - 1 } ^ { * } \mathrm { ~ f o r ~ } k = \tau _ { 0 } , \ldots , t - 1$ . Let med be the median of these changes and $I Q R = Q _ { 0 . 7 5 } - Q _ { 0 }$ .25 their interquartile range. Then

$$
s _ { 1 0 } = \operatorname { t a n h } \Biggl ( \lambda _ { 3 } \frac { I Q R } { \vert m e d \vert + \epsilon } \Biggr ) , \quad \lambda _ { 3 } = 0 . 5 , \quad \epsilon = 1 0 ^ { - 1 2 } .\tag{7}
$$

When fewer than L changes are available, use all available changes. The feature summarizes fluctuation relative to the typical recent progress.

5) Recent mode-specific reductions $( s _ { 1 6 } ) \colon$ Record the fitness reductions obtained by NDA and CC in separate histories $\mathcal { R } _ { N D A , t }$ <sub>t</sub> and ${ \mathcal { R } } _ { C C , i }$ before decision step t. For either mode m $\in \{ N D A , C C \}$ , define

$$
\bar { r } _ { m } = \left\{ \begin{array} { l l } { \operatorname* { m e a n } ( \mathcal { R } _ { m , t } ^ { [ - 5 : ] } ) , } & { | \mathcal { R } _ { m , t } | > 5 , } \\ { \operatorname* { m e a n } ( \mathcal { R } _ { m , t } ) , } & { 0 < | \mathcal { R } _ { m , t } | \leq 5 , } \\ { 0 , } & { | \mathcal { R } _ { m , t } | = 0 . } \end{array} \right.\tag{8}
$$

The feature $s _ { 1 6 } = \bar { r } _ { N D A } - \bar { r } _ { C C }$ compares the average reductions over the latest five uses of each mode, or over all its uses when fewer than five are available. Positive values indicate larger recent reductions from NDA.

## B. Network Architecture

This subsection expands the architecture in Fig. 1 of the main paper. FeatureSE reweights heterogeneous state information; the autoregressive actor conditions resource allocation on the selected mode; and the critic provides a corresponding baseline for each policy head.

1) Feature-aware state encoding: Apply layer normalization to obtain $\tilde { S } = \mathrm { L N } ( \boldsymbol { S } ) \in \mathbb { R } ^ { 1 6 }$ . For each of the four feature groups, compute its mean statistic and collect these statistics in $\bar { \mathcal { S } } = [ \bar { S } _ { p r o } , \bar { S } _ { c o n v } , \bar { S } _ { r e s } , \bar { S } _ { m o d e } ]$ . The group and feature gates

$$
\begin{array} { r l } & { g = \sigma \big ( \mathrm { M L P } _ { g } ( \bar { \mathcal { S } } ) \big ) , \quad g \in ( 0 , 1 ) ^ { 4 } , } \\ & { h = \sigma \big ( \mathrm { M L P } _ { h } ( \tilde { \mathcal { S } } ) \big ) , \quad h \in ( 0 , 1 ) ^ { 1 6 } . } \end{array}\tag{9}
$$

The two MLPs have dimensions $4 \times 8 \times 4$ and $1 6 \times 8 \times 1 6 .$ respectively, with SiLU nonlinearities and sigmoid outputs. Following Equation (4) of the main paper, residual gating uses

$$
w _ { j } = ( 1 - \lambda ) + \lambda g _ { k ( j ) } h _ { j } , \qquad z _ { j } = s _ { j } w _ { j } , \qquad \lambda = 0 . 2 .\tag{10}
$$

Thus $w _ { j } \in ( 0 . 8 , 1 )$ , limiting feature suppression. The last-layer biases of both gating MLPs are initialized to 2.0, for which $\sigma ( 2 ) \approx 0 . 8 8 .$ , to start close to the identity mapping.

2) Autoregressive actor: The policy factorization is

$$
\pi ( a _ { m o d e } , a _ { r e s } \mid z ) = \pi _ { \theta _ { m o d e } } ( a _ { m o d e } \mid z ) \pi _ { \theta _ { r e s } } ( a _ { r e s } \mid z , a _ { m o d e } ) .\tag{11}
$$

The mode head is an MLP with dimensions $1 6 { \times } 1 2 8 { \times } 2 5 6 { \times } 1$ ReLU hidden activations, and a Bernoulli output. Mode 0 denotes NDA and mode 1 denotes CC. The resource policy uses two separate MLPs, one for each mode, with dimensions $1 5 { \times } 1 6 0 { \times } 3 2 0 { \times } 3$ and ReLU hidden activations. Each resource head uses a 15-dimensional input excluding the opposite mode’s historical FE statistic: $z ^ { ( 0 ) }$ omits $z _ { 1 4 }$ for NDA, while $z ^ { ( 1 ) }$ omits $z _ { 1 3 }$ for CC. Each head produces a categorical distribution over the three resource levels in Table II. The resource logits for both modes are computed in parallel; the sampled mode determines which logits are used to sample the resource action.

TABLE I  
COMPLETE DEFINITIONS OF THE 16 WCC STATE FEATURES.
<table><tr><td>Group</td><td>Index</td><td>Formula</td><td>Explanation</td></tr><tr><td rowspan="5"> $\scriptstyle { S _ { p r o } }$ </td><td>1</td><td>1000 D  $\begin{array} { r } { 1 - \frac { \sum _ { i = 1 } ^ { D } \sum _ { j = 1 } ^ { D } \mathbb { I } ( \Theta [ i , j ] = 0 ) } { \mathbf { r } ^ { 2 } } } \end{array}$ </td><td>The relative size of the problem dimensions  $\overline { { D . } }$ </td></tr><tr><td>2</td><td> $\begin{array} { r } { \left| \{ i \} \sum _ { j = 1 } ^ { D } \Theta [ i , j ] = 1 \right\} } \end{array}$ </td><td>The sparsity of DSM Θ characterizes the interactive complexity of the problem.</td></tr><tr><td>3</td><td></td><td>The proportion of completely separable variables.</td></tr><tr><td>4</td><td> $\frac { \left| \bigcup _ { i = 1 } ^ { | \mathbb { S } | } \bigcup _ { j \neq i } ^ { D } \left( \Omega _ { i } \cap \Omega _ { j } \right) \right| } { D }$ </td><td>The proportion of overlapping variables,  $\Omega _ { i }$  is the variable set of subspace i.</td></tr><tr><td>5</td><td></td><td>The relative size of the number of subspaces  $\left| \mathbb { S } \right| .$ </td></tr><tr><td rowspan="8"> $s _ { c o n v }$ </td><td>6</td><td> $\overline { { \frac { c o s t _ { 0 } ^ { * } - c o s t _ { t - 1 } ^ { * } } { ~ } } }$ </td><td>The decrease ratio of optimal cost  $c o s t _ { t - 1 } ^ { * }$  at step  $t - 1 .$ </td></tr><tr><td>7</td><td>cost*  $L F E R ( c o s t ^ { * } )$ </td><td>The proportion of low-frequency energy (LFER) in the optimal cost curve.</td></tr><tr><td>8</td><td> $- \operatorname { t a n h } ( \lambda _ { 1 } \cdot \operatorname { m e a n } ( \nabla c o s t ^ { * } ) )$ </td><td>The normalized average first-order trend of the optimal cost trajectory, reflecting</td></tr><tr><td>9</td><td> $- \operatorname { t a n h } \left( \lambda _ { 2 } \cdot \operatorname* { m e a n } ( \nabla ^ { 2 } c o s t ^ { * } ) \right)$ </td><td>the convergence speed. The normalized second-order trend indicating convergence acceleration.</td></tr><tr><td>10</td><td> $\mathrm { t a n h } \left( \lambda _ { 3 } \frac { \mathrm { I Q R } ( \{ \Delta c o s t _ { k } \} _ { k = \tau _ { 0 } } ^ { t - 1 } ) } { \left| \mathrm { M e d i a n } ( \{ \Delta c o s t _ { k } \} _ { k = \tau _ { 0 } } ^ { t - 1 } ) \right| + \epsilon } \right)$ </td><td>The robust variation coefficient of recent optimal-cost changes, measuring the</td></tr><tr><td>11</td><td> $\frac { d m a x _ { t - 1 } } { d i a m e t e r }$ </td><td>stability of recent convergence. The ratio of the maximum distance of individuals dmax to the search diameter</td></tr><tr><td></td><td> $\mathbf { M a x F E s - } s u m F E s$ </td><td>diameter at step  $t - 1 .$ </td></tr><tr><td>12</td><td> $ \frac { \mathrm { M a x F E s } } { m e a n ( F E s _ { N D A } ) }$ </td><td>The remaining proportion of FEs.</td></tr><tr><td rowspan="4"></td><td>13</td><td> $\begin{array} { c } { m a x ( a _ { r e s } ) } \\ { m e a n ( F E s _ { C C } ) } \end{array}$ </td><td>The mean of function evaluations consumed by NDA.</td></tr><tr><td>14</td><td>max(ares)</td><td>The mean of function evaluations consumed by CC.</td></tr><tr><td>15</td><td> $\scriptstyle \overline { { \sum _ { t = 1 } ^ { T } \mathbb { I } ( a _ { t } = N D A ) } }$  T</td><td>The selection ratio of NDA across all optimization steps T.</td></tr><tr><td>16</td><td> $\bar { r } _ { \mathrm { N D A } } - \bar { r } _ { \mathrm { C C } }$ </td><td>The difference between the average recent fitness reductions achieved by NDA and CC.</td></tr></table>

3) Mode-conditioned critic: The critic uses a backbone of the same style as the actor’s mode head, followed by separate linear value heads $V _ { 0 } ( z )$ and $V _ { 1 } ( z )$ . It does not share actor parameters. The chosen-mode value is

$$
V ( z _ { t } , a _ { m o d e ( t ) } ) = \left\{ V _ { 0 } ( z _ { t } ) , \quad a _ { m o d e ( t ) } = 0 , \quad \right.\tag{12}
$$

For the resource head, this value is independent of the sampled resource action, conditional on the selected mode. For the mode head, the baseline is

$$
\bar { V } ( z _ { t } ) = \sum _ { m \in \{ 0 , 1 \} } \pi _ { { m o d e } } ( m \mid z _ { t } ) V ( z _ { t } , m ) ,\tag{13}
$$

which is independent of the realized mode action. This distinction matches the two successive decisions in the actor and reduces policy-gradient variance without introducing actiondependent baselines for either head.

## C. Training Objectives

We train the agent $\pi _ { \theta }$ by PPO [1]. At each step t, let $d _ { t } \in$ $\{ 0 , 1 \}$ be the done flag. The auto-regressive policy has two heads $h \in \{ \mathrm { m o d e } , \mathrm { r e s } \}$ with distributions $\pi _ { \boldsymbol { \theta } _ { h } } ( \cdot )$ . Using the mode-marginal value in Equation (13), the one-step target is

$R _ { \mathrm { c o n d ( } t ) } = r _ { t } + \gamma ( 1 - d _ { t } ) \bar { V } ( z _ { t + 1 } )$ . With value clipping $V _ { \mathrm { c l i p } }$ around the $V ^ { \mathrm { o l d } }$ , the value loss is:

$$
\mathcal { L } _ { V } = \mathbb { E } \left[ \operatorname* { m a x } \left( ( V - R _ { \mathrm { c o n d } } ) ^ { 2 } , ( V _ { \mathrm { c l i p } } - R _ { \mathrm { c o n d } } ) ^ { 2 } \right) \right] .\tag{14}
$$

Advantages $A _ { h , t }$ are computed with Generalized Advantage Estimation using the corresponding baseline (standardized per batch). Store old log-probabilities log $\pi _ { \theta _ { h } } ^ { \mathrm { o l d } } \left( \cdot \right)$ and recompute log-probabilities under the current policy log $\pi _ { \theta _ { h } } ^ { \mathrm { n e w } } ( \cdot )$ (no resampling) to form

$$
\rho _ { h , t } = \exp ( \log \pi _ { \theta _ { h } } ^ { n e w } \left( a _ { h , t } \mid \cdot \right) - \log \pi _ { \theta _ { h } } ^ { \mathrm { o l d } } \left( a _ { h , t } \mid \cdot \right) ) .\tag{15}
$$

The clipped surrogate per head is

$$
\mathcal { L } _ { h } ^ { \mathrm { c l i p } } = - \mathbb { E } \left[ \operatorname* { m i n } \left( \rho _ { h } A _ { h } , \mathrm { c l i p } \left( \rho _ { h } , 1 - \varepsilon , 1 + \varepsilon \right) A _ { h } \right) \right] .\tag{16}
$$

The total loss is

$$
\begin{array}{c} \mathcal { L } = \mathcal { L } _ { \mathrm { m o d e } } ^ { \mathrm { c l i p } } + \mathcal { L } _ { \mathrm { r e s } } ^ { \mathrm { c l i p } } + c _ { V } \mathcal { L } _ { V }  \\ { - \displaystyle \sum _ { h \in \{ \mathrm { m o d e } , \mathrm { r e s } \} } \beta _ { h } \mathbb { E } \left[ \mathcal { H } \left( \pi _ { \theta _ { h } } \right) \right] } \end{array}\tag{17}
$$

where $c _ { V }$ weights the value loss and $\beta _ { \mathrm { h } }$ controls per-head entropy regularization. We optimize with Adam (gradient clipping; exponentially decayed learning rate). Each iteration collects $n _ { \mathrm { s t e p } }$ transitions per environment, then performs K epochs over the full segment (masked by $( 1 - d _ { t } ) )$ ).

## D. Workflow

Algorithm 1 preserves the complete inference procedure. A decomposition strategy obtains the DSM Θ, and RDDSM derives the subspace list S. The trained agent selects the mode and resource budget from the current state. NDA operates in the full space; CC divides the selected budget among the subspaces and updates the incumbent after each subspace.

Algorithm 1 WCC   
Input: Problem instance $f$ and info I, Pre-trained Agent $\pi _ { \theta } ^ { * } .$   
Output: The best individual $X _ { b e s t } ,$ The best cost $Y _ { b e s t }$   
1: DSM $\Theta $ Decomposition $\operatorname { S t r a t e g y } ( f , I )$   
2: The list of Subspaces $\mathbb { S } \gets \mathrm { R D D S M } ( \Theta )$   
3: Initialize $X _ { b e s t } \mathrm { ~ , ~ } Y _ { b e s t }$ and state vector $s _ { 0 }$   
4: while Termination conditions not met do   
5: $a _ { m o d e ( t ) } , a _ { r e s ( t ) }  \pi _ { \theta } ^ { * } \bigl ( s _ { t } \bigr )$   
6: if $a _ { m o d e ( t ) } = 0$ then   
7: $\mathrm { N D A } ( f , I , X _ { b e s t } , a _ { r e s ( t ) } )$   
8: $X _ { b e s t } , Y _ { b e s t } , s _ { t + 1 } \gets \mathbf { U } _ { \mathrm { I } }$ pdate $X _ { b e s t } , Y _ { b e s t } , s _ { t }$   
9: else   
10: Each subspace FEs $S u b F E s = \left\lceil a _ { r e s ( t ) } / l e n ( \mathbb { S } ) \right\rceil$   
11: for sub in S do   
12: $A _ { s u b } ( f , I , s u b , X _ { b e s t } , S u b F E s )$   
13: $X _ { b e s t } , Y _ { b e s t } \gets \mathrm { U p d a t e } X _ { b e s t } , Y _ { b e s t }$   
14: end for   
15: $X _ { b e s t } , Y _ { b e s t } , s _ { t + 1 } \gets \mathrm { U p d a t e } X _ { b e s t } , Y _ { b e s t } , s _ { t }$   
16: end if   
17: end while   
18: return $X _ { b e s t } , Y _ { b e s t }$

The state is updated after the selected mode finishes, and optimization stops when no FEs remain.

## APPENDIX B EXPERIMENTS

## A. Experimental Setup

1) Benchmarks: CEC2013LSGO [2] contains 15 problems, grouped here as fully separable (F1–F3), partially separable with separable components (F4–F7), partially separable without separable components (F8–F11), overlapping (F13–F14), and non-separable (F12 and F15). As in the main paper, F12 is treated as non-separable because its chain-like dependency structure yields 999 overlapping variables and subspaces.

The Auto Overlapping Benchmark (AOB) [3] provides additional 1000-dimensional problems while keeping the dimension fixed as overlap changes. Its base functions are Ackley, Rastrigin, Schwefel, and Elliptic, following the F8– F11 settings of CEC2013LSGO. The 16 instances shown in Fig. 2 of the main paper comprise three overlapping variants per base function, with 10, 15, or 20 overlapping variables per subspace, and one globally rotated non-separable variant. Labels such as A10 indicate the base function and overlap; A Non denotes the non-separable Ackley variant.

2) Comparison algorithms: WCC-MMES uses CMA-ES [4] for subspace optimization and MMES [5] for fullspace optimization. DG2 and RDDSM provide the decomposition and subspace construction used by its CC component [3], [6]. The CC-based comparisons are DG2-CMAES, CC-CMAES [7], LCC-CMAES [8], ERDG-CMAES [9], and OEDG-CMAES [10]. The NDA comparisons are MMES, sep-CMA-ES [11], LM-MA-ES [12], FCMAES [13], and MetaES [14]. HCC-ES [3] and CC-SHADE-ML [15] are the hybrid comparisons. The original numerical table labels the latter CC-SHADE-SL; that table label is retained below to match Table I of the main paper.

TABLE II  
REPORTED WCC-MMES TRAINING AND EVALUATION SETTINGS.  
Setting Value   
Training epochs 90   
Training MaxFEs $1 0 ^ { 6 }$ per episode   
Optimizer $\mathbf { A d a m }$   
Initial learning rate $1 0 ^ { - 4 } ;$ exponential decay   
Discount factor γ 0.999   
Rollout length $ { n _ { s t e p } }$ 5   
PPO update epochs $K$ 4   
Resource budgets $\{ 5 \times 1 0 ^ { 4 } , 1 0 ^ { 5 } , 2 \times 1 0 ^ { 5 } \}$ FEs   
FeatureSE strength λ 0.2   
FeatureSE output biases 2.0   
Fully/non-separable grouping 20 subspaces   
Main CEC2013LSGO tests $1 . 5 \times 1 0 ^ { 6 }$ FEs   
DG2 decomposition allowance $5 \times 1 0 ^ { 5 }$ FEs, included in main budget   
AOB tests $1 . 5 \times 1 0 ^ { 6 }$ FEs   
Horizon tests $3 \times 1 0 ^ { 6 }$ FEs   
NDA-transfer tests $1 0 ^ { 6 }$ FEs   
Independent comparison runs 30 per algorithm and problem   
Significance test Wilcoxon rank-sum; $\bar { \alpha } = 0 . 0 5$

3) Training and evaluation: Table II collects the reported settings. The training functions are marked with an asterisk in Table I of the main paper. Training and held-out functions can share a separability category while differing in their base functions and other properties. Decomposition results are precomputed to accelerate training. Comparison methods follow the configurations in their original papers; the NDA implementations use libraries such as PyPop7 [16].

The policy trained at $1 0 ^ { 6 }$ FEs is reused for the generalization tests without retraining. Numerical comparisons report the mean and standard deviation over 30 independent runs.

## B. Ablation Study

To further examine where the performance of WCC-MMES comes from, we conduct a comprehensive ablation study from four aspects: action space, state space, reward function, and network architecture.

1) Experimental Protocol: For the action space ablation, we consider both random and fixed variants. In the random variants, either both action components, only the mode $( \mathrm { R a n d o m } ( a _ { \mathrm { m o d e } } ) )$ , or only the resource $( \mathrm { R a n d o m } ( a _ { \mathrm { r e s } } ) )$ are randomized. In the fixed variants, either the mode is fixed to CC or NDA while the resource is controlled by the agent $( { \mathrm { F i x e d } } ( a _ { \mathrm { m o d e } } ) )$ , or the resource budget is fixed while the mode is controlled by the agent $( { \mathrm { F i x e d } } ( a _ { \mathrm { r e s } } ) )$ . For the state space ablation, we remove each major state group individually, including problem features $( \mathbf { W } / \mathbf { O } \ S _ { \mathrm { p r o } } )$ , convergence features (W/O $S _ { \mathrm { c o n v } } ) _ { \cdot }$ , resource features $( \mathbf { W } / \mathbf { O \ S _ { r e s } } )$ , and mode history features $\left( \mathbf { W } / \mathbf { O } ~ S _ { \mathrm { m o d e } } \right)$ . For the reward ablation, the proposed log-ratio reward is replaced by Difference $r _ { t } = c o s t _ { t - 1 } ^ { * } - c o s t _ { t } ^ { * }$ , Sparse Sign

$$
r _ { t } = \left\{ \begin{array} { l } { { 1 , \quad c o s t _ { t - 1 } ^ { * } - c o s t _ { t } ^ { * } > \epsilon } } \\ { { 0 , \quad \mathrm { o t h e r w i s e } } } \end{array} \right. \ \epsilon = 1 \mathrm { E - 2 }
$$

, and Linear $\begin{array} { r } { r _ { t } ~ = ~ \frac { c o s t _ { t - 1 } ^ { * } - c o s t _ { t } ^ { * } } { \left| c o s t _ { t - 1 } ^ { * } \right| } } \end{array}$ rewards. For the network ablation, we remove FeatureSE (W/O FeatureSE), replace the auto-regressive policy with a simple MLP, or remove both. For fair comparison, the final costs are min–max normalized within each category, and the mean normalized performance over all problems is reported. All ablated variants are trained and evaluated under the same protocol as Section III-B1 of the main paper.

TABLE III  
WCC-MMES AND COMPARISON ALGORITHMS ON CEC2013LSGO AT 3 × $1 0 ^ { 6 }$ FES.
<table><tr><td rowspan="2">Problem</td><td rowspan="2">WCC-based Algorithms WCC-MMES</td><td colspan="3">CC-based Algorithms</td><td colspan="5"></td><td colspan="2" rowspan="2">Hybrid Algorithms</td></tr><tr><td>DG2-CMAES</td><td>CC-CMAES</td><td>LCC-CMAES</td><td>MMES sep-CMA-ES</td><td>NDAs LM-MA-ES</td><td>FCMAES</td><td>MetaES</td><td>HCC-ES</td></tr><tr><td rowspan="2"></td><td rowspan="2">2.03E-20 ± 1.78E-20</td><td>6.68E-20(+)</td><td>7.38E+06(+)</td><td>1.09E+05(+)</td><td>1.18E+04(+)</td><td>1.69E-17(+)</td><td>1.63E+05(+)</td><td>1.95E+05(+)</td><td>9.09E+09(+)</td><td>6.31E-20(+)</td><td>CC-SHADE-SL 2.58E-05(+)</td></tr><tr><td>± 3.78E-21</td><td>± 3.92E+06</td><td>± 2.71E+04</td><td>± 1.80E+03</td><td>± 7.95E-19</td><td>± 4.33E+04</td><td>± 4.04E+04</td><td>± 7.69E+08</td><td>± 3.45E-21</td><td>± 1.22E-04</td></tr><tr><td rowspan="2">2</td><td>1.35E+03</td><td>3.47E+03(+)</td><td>2.13E+03(+)</td><td>2.09E+03(+)</td><td>1.26E+04(+)</td><td>5.34E+03(+)</td><td>5.98E+03(+)</td><td>1.38E+04(+)</td><td>4.05E+04(+)</td><td>3.26E+03(+)</td><td>1.98E+03(+)</td></tr><tr><td>± 3.55E+02</td><td>± 3.49E+02</td><td>± 1.31E+02</td><td>± 1.82E+02</td><td>± 7.61E+02</td><td>± 1.59E+02</td><td>± 2.74E+02</td><td>± 8.52E+02</td><td>± 5.53E+03</td><td>± 3.25E+02</td><td>± 5.73E+02</td></tr><tr><td rowspan="2">3</td><td>2.00E+01</td><td>2.10E+01(+)</td><td>2.00E+01(≈)</td><td>2.03E+01(+)</td><td>2.00E+01(≈)</td><td>2.14E+01(+)</td><td>2.15E+01(+)</td><td>2.04E+01(+)</td><td>2.16E+01(+)</td><td>2.09E+01(+)</td><td>2.08E+01(+)</td></tr><tr><td>± 6.39E-03</td><td>± 2.10E-02</td><td>± 4.82E-03</td><td>± 3.06E-02</td><td>± 1.85E-03</td><td>± 1.09E-01</td><td>± 1.89E-01</td><td>± 1.55E-01</td><td>± 3.96E-01</td><td>± 3.41E-02</td><td>± 3.91E-02</td></tr><tr><td rowspan="3">4</td><td>1.32E+03</td><td>9.89E+02(-)</td><td>1.18E+11(+)</td><td>2.27E+10(+)</td><td>6.64E+07(+)</td><td>3.73E+10(+)</td><td></td><td>4.78E+08(+)</td><td></td><td></td><td></td></tr><tr><td></td><td>± 5.73E+00</td><td>± 6.05E+10</td><td>± 1.33E+09</td><td></td><td>± 8.85E+09</td><td>1.65E+11(+)</td><td></td><td>5.08E+12(+)</td><td>9.81E+03(+)</td><td>6.13E+09(+)</td></tr><tr><td>± 8.74E+02 1.39E+06</td><td>1.58E+06(+)</td><td></td><td>1.02E+07(+)</td><td>± 5.45E+06</td><td></td><td>± 3.41E+10</td><td>± 4.98E+07</td><td>± 2.43E+09</td><td>± 3.46E+02</td><td>± 4.68E+09</td></tr><tr><td rowspan="3">5 6</td><td>± 1.21E+05</td><td>± 1.62E+05</td><td>8.65E+06(+)</td><td></td><td>4.17E+06(+)</td><td>1.64E+06(+)</td><td>1.49E+06(+)</td><td>3.82E+06(+)</td><td>4.89E+07(+)</td><td>1.60E+06(+)</td><td>2.70E+06(+)</td></tr><tr><td></td><td></td><td>± 2.68E+06</td><td>± 1.01E+06</td><td>± 7.67E+05</td><td>± 1.52E+05</td><td>± 4.58E+05</td><td>± 8.33E+05</td><td>± 8.21E+05</td><td>± 1.32E+05</td><td>± 7.29E+05</td></tr><tr><td>9.95E+05</td><td>1.06E+06(+)</td><td>1.02E+06(+)</td><td>1.05E+06(+)</td><td>9.95E+05(≈)</td><td>1.06E+06(+)</td><td>1.05E+06(+)</td><td>9.96E+05(+)</td><td>1.06E+06(+)</td><td>1.06E+06(+)</td><td>1.05E+06(+)</td></tr><tr><td rowspan="3">7</td><td>± 4.08E-01</td><td>± 1.22E+03</td><td>± 1.56E+04</td><td>± 2.08E+03</td><td>± 2.14E-01</td><td>± 1.78E+03</td><td>± 1.35E+03</td><td>± 9.38E+02</td><td>± 1.63E+03</td><td>± 9.98E+02</td><td>± 2.25E+03</td></tr><tr><td>9.65E+01</td><td>1.38E+02(+)</td><td>1.46E+09(+)</td><td>4.91E+06(+)</td><td>2.45E+04(+)</td><td>2.36E+09(+)</td><td>3.72E+06(+)</td><td>1.65E+05(+)</td><td>1.21E+15(+)</td><td>9.01E+02(+)</td><td>9.23E+05(+)</td></tr><tr><td>± 4.97E+00</td><td>± 2.20E+01</td><td>± 2.15E+09</td><td>± 3.11E+06</td><td>± 1.58E+03</td><td>± 1.87E+09</td><td>± 4.13E+05</td><td>± 3.53E+04</td><td>± 7.78E+10</td><td>± 2.44E+01</td><td>± 8.83E+05</td></tr><tr><td rowspan="3">8</td><td>1.49E+07</td><td>2.42E+06(-)</td><td>4.53E+15(+)</td><td>3.99E+14(+)</td><td>1.84E+11(+)</td><td>5.44E+14(+)</td><td>1.71E+15(+)</td><td>9.49E+11(+)</td><td>1.46E+17(+)</td><td>3.11E+06(-)</td><td>1.21E+14(+)</td></tr><tr><td>± 3.70E+06</td><td>± 1.09E+06</td><td>± 2.66E+15</td><td>± 7.17E+13</td><td>± 7.76E+09</td><td>± 6.78E+14</td><td>± 4.20E+14</td><td>± 1.96E+11</td><td>± 6.21E+15</td><td>± 9.40E+05</td><td>± 5.46E+13</td></tr><tr><td>1.13E+08</td><td>1.20E+08(+)</td><td>7.60E+08(+)</td><td>8.19E+08(+)</td><td>4.41E+08(+)</td><td>1.41E+08(+)</td><td>2.20E+08(+)</td><td>5.61E+08(+)</td><td>3.73E+09(+)</td><td>1.85E+08(+)</td><td>2.27E+08(+)</td></tr><tr><td rowspan="3">10</td><td>± 3.12E+07</td><td>± 1.56E+07</td><td>± 1.42E+08</td><td>± 1.24E+08</td><td>± 8.42E+07</td><td>± 1.70E+07</td><td>± 2.20E+07</td><td>± 9.23E+07</td><td>± 1.74E+06</td><td>± 1.24E+07</td><td>± 1.12E+08</td></tr><tr><td>9.05E+07</td><td>9.09E+07(+)</td><td>9.41E+07(+)</td><td>9.22E+07(+)</td><td>9.05E+07(≈)</td><td>9.41E+07(+)</td><td>9.40E+07(+)</td><td>9.06E+07(≈)</td><td>9.40E+07(+)</td><td>9.10E+07(+)</td><td>9.26E+07(+)</td></tr><tr><td>± 6.43E+00</td><td>± 2.77E+05</td><td>± 5.04E+05</td><td>± 2.02E+05</td><td>± 3.47E+02</td><td>± 1.66E+05</td><td>± 2.77E+05</td><td>± 3.98E+04</td><td>± 6.90E+05</td><td>± 2.87E+05</td><td>± 3.44E+05</td></tr><tr><td rowspan="3">11 12</td><td>2.89E+04</td><td>2.94E+04(+)</td><td>3.76E+11(+)</td><td>4.38E+09(+)</td><td>2.14E+06(+)</td><td>3.85E+11(+)</td><td>8.50E+07(+)</td><td>1.41E+07(+)</td><td>6.47E+16(+)</td><td>3.34E+04(+)</td><td>1.71E+08(+)</td></tr><tr><td>± 2.54E+04</td><td>± 4.36E+04</td><td>± 2.85E+11</td><td>± 4.83E+09</td><td>± 1.87E+05</td><td>± 1.36E+10</td><td>± 1.46E+07</td><td>± 1.96E+06</td><td>± 3.23E+15</td><td>± 3.96E+04</td><td>± 9.20E+07</td></tr><tr><td>7.91E+02</td><td>1.18E+03(+)</td><td>1.29E+03(+)</td><td>1.31E+03(+)</td><td>8.62E+02(+)</td><td>8.66E+02(+)</td><td>6.14E+02(-)</td><td>5.34E+02(-)</td><td>1.04E+08(+)</td><td>1.18E+03(+)</td><td>1.32E+03(+)</td></tr><tr><td rowspan="3">13</td><td>± 1.84E+01</td><td>± 8.26E+01</td><td>± 4.91E+03</td><td>± 1.61E+02</td><td>± 5.76E+02</td><td>± 3.94E+01</td><td>± 1.68E+02</td><td>± 2.29E+02</td><td>± 6.44E+08</td><td>± 8.78E+01</td><td>± 2.89E+02</td></tr><tr><td>2.46E+05</td><td>1.53E+08(+)</td><td>5.79E+10(+)</td><td>2.56E+07(+)</td><td>2.89E+05(+)</td><td>9.63E+10(+)</td><td>8.74E+06(+)</td><td>1.23E+06(+)</td><td>9.44E+16(+)</td><td>2.27E+06(+)</td><td>9.75E+07(+)</td></tr><tr><td>± 1.45E+04</td><td>± 1.06E+08</td><td>± 3.17E+10</td><td>± 4.84E+06</td><td>± 4.27E+04</td><td>± 6.91E+10</td><td>± 9.15E+05</td><td>± 1.24E+05</td><td>± 7.49E+15</td><td></td><td></td></tr><tr><td rowspan="3">14</td><td>5.81E+06</td><td>3.04E+07(+)</td><td>8.53E+11(+)</td><td>1.53E+08(+)</td><td>7.91E+06(+)</td><td>9.01E+09(+)</td><td>1.02E+08(+)</td><td>2.14E+07(+)</td><td>9.47E+17(+)</td><td>± 1.01E+06</td><td>± 1.19E+08</td></tr><tr><td></td><td>± 8.08E+06</td><td>± 4.51E+11</td><td>± 1.82E+07</td><td>± 1.22E+06</td><td>± 5.70E+09</td><td>± 6.27E+06</td><td>± 1.32E+06</td><td></td><td>9.92E+06(+)</td><td>9.41E+07(+)</td></tr><tr><td>± 4.37E+06 3.89E+05</td><td>3.54E+13(+)</td><td>3.06E+08(+)</td><td>1.85E+07(+)</td><td>1.29E+06(+)</td><td>2.66E+08(+)</td><td>9.88E+06(+)</td><td></td><td>± 6.19E+15</td><td>± 1.19E+06</td><td>± 1.11E+08</td></tr><tr><td rowspan="2">15 +/≈1-</td><td>± 1.67E+05</td><td></td><td>± 3.06E+08</td><td>± 3.59E+06</td><td>± 1.35E+05</td><td>± 5.27E+07</td><td>± 2.35E+06</td><td>6.61E+06(+) ± 9.06E+05</td><td>2.35E+15(+) ± 1.62E+12</td><td>2.79E+13(+)</td><td>2.00E+06(+)</td></tr><tr><td>NA</td><td>± 6.35E+12 13/0/2</td><td>14/1/0</td><td>15/0/0</td><td>12/3/0</td><td>15/0/0</td><td>14/0/1</td><td>13/1/1</td><td>15/0/0</td><td>± 1.61E+13 14/0/1</td><td>± 5.73E+05 15/0/0</td></tr></table>

Entries report means and standard deviations over 30 runs. Symbols +, ≈, and − indicate WCC-MMES is significantly better, comparable, and worse, respectively (Wilcoxon rank-sum, α = 0.05). The final row reports win/tie/loss counts.

2) Normalization and reporting: For category c, let $\boldsymbol { \mathcal { A } } _ { c }$ contain WCC-MMES and the variants compared in that category, and let ${ \bar { f } } _ { p , a }$ denote the mean final cost of variant a on problem $p .$ The per-problem extremes are $f _ { p , c } ^ { \operatorname* { m i n } } = \operatorname* { m i n } _ { a \in \mathcal { A } _ { c } } \bar { f } _ { p , a }$ and $f _ { p , c } ^ { \operatorname* { m a x } } = \operatorname* { m a x } _ { a \in \mathcal { A } _ { c } } \bar { f } _ { p , a }$ . The inverse min–max score is

$$
q _ { p , a } ^ { ( c ) } = \left\{ { 1 - \frac { \bar { f } _ { p , a } - f _ { p , c } ^ { \operatorname* { m i n } } } { f _ { p , c } ^ { \operatorname* { m a x } } - f _ { p , c } ^ { \operatorname* { m i n } } } , } \quad f _ { p , c } ^ { \operatorname* { m a x } } > f _ { p , c } ^ { \operatorname* { m i n } } , \quad \right.\tag{18}
$$

For the problem set $\mathcal { P } _ { c }$ evaluated in category c, the reported mean normalized performance is

$$
Q _ { a } ^ { ( c ) } = \frac { 1 } { | \mathcal { P } _ { c } | } \sum _ { p \in \mathcal { P } _ { c } } q _ { p , a } ^ { ( c ) } .\tag{19}
$$

Costs are normalized separately for each problem and ablation category before averaging. Higher scores indicate better performance within the corresponding category.

3) Ablation Analysis: The results are shown in Figure 1. a) Action space. The key takeaway from Figure 1(a) is that WCC benefits from jointly learning both the optimization mode and the resource allocation. Fixing the mode leads to the worst performance, indicating that neither CC nor NDA is universally optimal across all problems and stages. Randomizing either the mode selector or the resource allocator also degrades performance, showing that the learned decisions cannot be replaced by arbitrary switching.

![](images/fb1ad7886362c1aa7106a5661087166fe3f6e0b96cf6d380b413d314246b2c77.jpg)  
Fig. 1. Ablation study, including (a) action space ablation, (b) state space ablation, (c) reward function ablation, and (d) network architecture ablation.

b) State space. The key takeaway from Figure 1(b) is that the four state groups provide complementary information rather than redundant signals. Removing any major state group leads to a clear performance drop: problem features provide structural priors such as dimensionality and decomposability, convergence features capture the current optimization dynamics and cause the largest degradation when removed, while resource and mode-history features describe budget consumption and historical mode preference, improving the timing and stability of the learned switching policy.

c) Reward function. The key takeaway from Figure 1(c) is that a stable and scale-aware reward is critical for learning across heterogeneous LSGO problems. Replacing the proposed log-ratio reward with Difference, Sparse Sign, or Linear rewards consistently worsens performance: Difference rewards are sensitive to objective scales, Sparse Sign rewards discard the magnitude of improvement and depend on threshold selection, and Linear rewards may still suffer from numerical instability.

d) Network architecture. The key takeaway from Figure 1(d) is that both FeatureSE and the auto-regressive policy contribute to WCC’s performance. Removing FeatureSE degrades performance, confirming the usefulness of adaptively reweighting heterogeneous state features. Replacing the auto-regressive policy with a simple MLP also hurts performance, because WCC naturally involves a hierarchical decision process: the agent should first select CC or NDA and then allocate FEs conditioned on the selected mode. The largest degradation occurs when both FeatureSE and the auto-regressive structure are removed, indicating that heterogeneous state encoding and mode-conditioned resource allocation jointly support effective CC deployment.

## REFERENCES

[1] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[2] X. Li, K. Tang, M. N. Omidvar, Z. Yang, and K. Qin, “Benchmark functions for the CEC’2013 special session and competition on largescale global optimization,” Evolutionary Computation and Machine

Learning Group, RMIT University, Melbourne, Australia, Technical Report, 2013.

[3] W. Qiu, H. Guo, Z. Ma, and Y.-J. Gong, “A novel two-phase cooperative co-evolution framework for large-scale global optimization with complex overlapping,” arXiv preprint arXiv:2503.21797, 2025.

[4] N. Hansen and A. Ostermeier, “Completely derandomized selfadaptation in evolution strategies,” Evolutionary computation, vol. 9, no. 2, pp. 159–195, 2001.

[5] X. He, Z. Zheng, and Y. Zhou, “Mmes: Mixture model-based evolution strategy for large-scale optimization,” IEEE Transactions on Evolution ary Computation, vol. 25, no. 2, pp. 320–333, 2020.

[6] M. N. Omidvar, M. Yang, Y. Mei, X. Li, and X. Yao, “Dg2: A faster and more accurate differential grouping for large-scale black-box optimization,” IEEE Transactions on evolutionary computation, vol. 21, no. 6, pp. 929–942, 2017.

[7] J. Liu and K. Tang, “Scaling up covariance matrix adaptation evolution strategy using cooperative coevolution,” in International Conference on Intelligent Data Engineering and Automated Learning. Springer, 2013, pp. 350–357.

[8] H. Guo, W. Qiu, Z. Ma, X. Zhang, J. Zhang, and Y.-J. Gong, “Advancing cma-es with learning-based cooperative coevolution for scalable optimization,” arXiv preprint arXiv:2504.17578, 2025.

[9] M. Yang, A. Zhou, C. Li, and X. Yao, “An efficient recursive differential grouping for large-scale continuous problems,” IEEE Transactions on Evolutionary Computation, vol. 25, no. 1, pp. 159–171, 2020.

[10] M. Tian, M. Chen, W. Du, Y. Tang, and Y. Jin, “An enhanced differential grouping method for large-scale overlapping problems,” IEEE Transactions on Evolutionary Computation, 2024.

[11] R. Ros and N. Hansen, “A simple modification in cma-es achieving linear time and space complexity,” in International conference on parallel problem solving from nature. Springer, 2008, pp. 296–305.

[12] I. Loshchilov, T. Glasmachers, and H.-G. Beyer, “Large scale black-box optimization by limited-memory matrix adaptation,” IEEE Transactions on Evolutionary Computation, vol. 23, no. 2, pp. 353–358, 2018.

[13] Z. Li, Q. Zhang, X. Lin, and H.-L. Zhen, “Fast covariance matrix adaptation for large-scale black-box optimization,” IEEE transactions on cybernetics, vol. 50, no. 5, pp. 2073–2083, 2018.

[14] R. Lange, T. Schaul, Y. Chen, T. Zahavy, V. Dalibard, C. Lu, S. Singh, and S. Flennerhag, “Discovering evolution strategies via meta-black-box optimization,” in Proceedings of the companion conference on genetic and evolutionary computation, 2023, pp. 29–30.

[15] A. Vakhnin and E. Sopov, “A novel self-adaptive cooperative coevolution algorithm for solving continuous large-scale global optimization problems,” Algorithms, vol. 15, no. 12, p. 451, 2022.

[16] Q. Duan, G. Zhou, C. Shao, Z. Wang, M. Feng, Y. Huang, Y. Tan, Y. Yang, Q. Zhao, and Y. Shi, “Pypop7: A pure-python library for population-based black-box optimization,” Journal ofMachine Learning Research, vol. 25, no. 296, pp. 1–28, 2024.