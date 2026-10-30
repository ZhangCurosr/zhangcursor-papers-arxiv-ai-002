# Learning Safe Flow-Matching Robot Policies from Binary Feasibility Oracles

Jianming Ma Shanghai Jiao Tong University

Rongjun Jin Shanghai Jiao Tong University

Yizhi Chen Tongji University

Yue Gao Shanghai Jiao Tong University

Abstract: Robot foundation models increasingly rely on flow-based action generation for general-purpose manipulation, yet the generated actions are not guaranteed to satisfy physical constraints such as joint limits and collision avoidance. This problem is especially challenging when the feasible set cannot be described analytically and is accessible only through black-box feasibility oracles. We study this practical oracle-only setting and propose an oracle-guided differentiable safeguard for safe flow-matching robot policies. The safeguard uses binary feasibility queries to correct unsafe updates in the forward pass, while extracting local boundary information to provide constraint-aware learning signals in the backward pass. Combined with discrete-time flow matching, the same safeguard is used during training and inference, enabling forward–backward aligned policy learning without explicit or differentiable constraint models. Experiments on obstacle-aware robotic manipulation demonstrate 100% constraint satisfaction while preserving task performance.

Keywords: safe robot learning, flow matching, feasibility oracle

## 1 Introduction

Robot foundation models, including vision-language-action (VLA) models [1, 2] and world-action models (WAMs) [3, 4], increasingly combine pretrained vision-language or video-generation backbones with flow-based action generation for general-purpose robot control. However, strong task generalization does not guarantee safe physical interaction: generated actions must also satisfy constraints such as joint limits and collision avoidance. Enforcing such scene-dependent constraints remains a key challenge for deploying flow-based robot policies.

Existing methods either correct generated actions only at inference time or incorporate constraint enforcement into training. Inference-time correction, e.g., through control barrier functions [5, 6, 7, 8], or trajectory-wise constraint guidance [9, 10], can be applied to pretrained policies but introduces a train–test mismatch. Training-aware approaches reduce this mismatch [11, 12, 13], but typically require explicit constraint functions, differentiable constraint models, or known feasible-set geometry. Such information is often difficult to obtain in robotics. For example, in robotic arm obstacle avoidance, a collision checker can readily determine whether a candidate configuration is feasible, while an analytical description of the corresponding collision-free configuration space may be unavailable.

This motivates an oracle-only setting, where safety is accessible solely through a binary feasibility oracle. We ask: how can aflow-matching robotpolicy enforce constraints and learnfrom them when only binaryfeasibility queries are available? To address this problem, we propose an oracle-guided differentiable safeguard. In the forward pass, the safeguard uses oracle queries to restrict each predicted update to the feasible region. In the backward pass, it extracts local boundary information from nearby binary queries and uses it to construct a boundary-aware surrogate gradient. Thus, the same oracle supports both feasibility enforcement and constraint-aware policy learning, without requiring an explicit or differentiable constraint model.

We integrate the safeguard with discrete-time flow matching, allowing the same correction mechanism to be used during both training and inference. This yields a forward–backward aligned learning framework in which the policy is optimized under the safety intervention encountered at deployment.

Our contributions are summarized as follows:

• We formulate safe flow-matching robot policy learning under an oracle-only constraint interface, removing the need for explicit or differentiable constraint models.

• We introduce an oracle-guided differentiable safeguard that combines oracle-based forward feasibility correction with boundary-aware surrogate gradients, enabling forward– backward aligned policy learning.

• We validate the proposed approach on an obstacle-aware robotic manipulation task, demonstrating strong constraint satisfaction while maintaining task performance.

## 2 Preliminaries and Problem Formulation

Consider a robot manipulation task at control step t, where $o _ { t }$ denotes the current multimodal observation and language instruction. The policy predicts an H-step future joint-position trajectory $A _ { t } = [ q _ { t } ^ { ( 1 ) } , \dots , q _ { t } ^ { ( H ) } ] \in \mathbb { R } ^ { H \times d }$ , where $q _ { t } ^ { ( h ) } \in \mathbb { R } ^ { d }$ denotes the predicted joint configuration at the h-th future horizon. The task-dependent safety constraints are accessible only through a binary feasibility oracle

$$
\begin{array} { r } { \mathcal { O } _ { \mathrm { s } } ( q ; \xi _ { t } ) \in \{ 0 , 1 \} , \ \quad \mathcal { C } ( \xi _ { t } ) = \{ q \in \mathbb { R } ^ { d } : \mathcal { O } _ { \mathrm { s } } ( q ; \xi _ { t } ) = 1 \} , } \end{array}\tag{1}
$$

where $\xi _ { t }$ denotes the task context, and $\mathscr { C } ( \xi _ { t } )$ denotes the feasible region. The oracle need not be differentiable and does not provide an analytical description of the feasible set. Our goal is to ensure that every configuration in the predicted action chunk is feasible: $q _ { t } ^ { ( h ) } \in \mathcal { C } ( \xi _ { t } ) , \forall h \in \{ 1 , \dots , H \}$

Discrete-time flow matching. We generate the entire action chunk through a discrete-time flow. Let $X _ { k } = [ x _ { k } ^ { ( 1 ) } , \ldots , x _ { k } ^ { ( H ) } ] \in \mathbb { R } ^ { H \times d } , k = 0 , \ldots , T$ , denote the action-chunk sample at flow step k. Starting from $X _ { 0 } \sim p _ { 0 } ( \cdot \mid o _ { t } )$ , the action expert predicts the finite update

$$
D _ { k } = \Delta _ { \theta } ( X _ { k } , k , o _ { t } ) = [ d _ { k } ^ { ( 1 ) } , \ldots , d _ { k } ^ { ( H ) } ] , \qquad X _ { k + 1 } = X _ { k } + D _ { k } .\tag{2}
$$

Given a target action chunk $A _ { t } ,$ we set $X _ { T } ~ = ~ A _ { t }$ and use the linear interpolation [14] $X _ { k } ~ =$ $( 1 - k / T ) X _ { 0 } + k / T \cdot A _ { t }$ . The corresponding discrete flow-matching objective is

$$
\mathcal { L } _ { \mathrm { D F M } } = \mathbb { E } \left[ \left. \Delta _ { \theta } ( X _ { k } , k , o _ { t } ) - ( A _ { t } - X _ { 0 } ) / T \right. _ { F } ^ { 2 } \right] ,\tag{3}
$$

where $\| \cdot \| _ { F }$ denotes the Frobenius norm over the entire action chunk.

This finite-update formulation provides a natural interface for enforcing safety during both training and inference. Starting from a feasible initial sample, feasibility can be preserved recursively by safeguarding each discrete update, thereby ensuring that the final generated sample remains in the feasible set.

## 3 Oracle-Guided Differentiable Safeguard

We introduce a safeguard that uses the binary oracle for both forward feasibility correction and backward boundary-aware learning. Since the safety oracle is defined over individual joint configurations, the safeguard is applied independently to each horizon element $( x _ { k } ^ { ( h ) } , d _ { k } ^ { ( h ) } )$ . For clarity, we omit the horizon index h below and write $x _ { k }$ and $d _ { k }$ for a single horizon element.

![](images/ed489c0fe0475470b0bb9346e968c68eed8dfbcc2528dfac34c96c95230de5ef.jpg)  
Figure 1: Left: Overview of oracle-guided differentiable safeguard. Right: safe discrete-time flowmatching training using the same safeguard.

Forward: Oracle-Guided Safety Correction Given a feasible point $x _ { k }$ and its nominal update $d _ { k }$ , we reparameterize the transition as $x _ { k + 1 } = x _ { k } + \alpha _ { k } d _ { k } , \alpha _ { k } \in [ 0 , 1 ]$ . Rather than reconstructing the feasible set, we search only along the ray $r _ { k } ( \alpha ) = x _ { k } + \alpha d _ { k } , \alpha \in [ 0 , 1 ]$ , for the first exit from the feasible region. The ideal first-exit factor is

$$
\alpha _ { k } ^ { * } = \operatorname* { s u p } \left\{ \alpha \in [ 0 , 1 ] : \mathcal { O } _ { \mathrm { s } } ( r _ { k } ( \tau ) ; \xi _ { t } ) = 1 , \forall \tau \in [ 0 , \alpha ] \right\} .\tag{4}
$$

Since only binary queries are available, we approximate $\alpha _ { k } ^ { * }$ using an oracle-based ray search, e.g., a coarse scan followed by local bisection: $\hat { \alpha } _ { k } = \mathrm { O r a c l e R a y S e a r c h } ( \mathcal { O } _ { \mathrm { s } } , x _ { k } , d _ { k } ; \xi _ { t } )$ . The safeguarded update is then $\Delta _ { k } ^ { \mathrm { s a f e } } = \hat { \alpha } _ { k } d _ { k } , x _ { k + 1 } = x _ { k } + \Delta _ { k } ^ { \mathrm { s a f e } }$

Backward: Oracle-Guided Boundary Gradient The oracle-based forward correction is nondifferentiable. To expose the encountered constraint boundary to the action expert during training, we construct a boundary-aware surrogate Jacobian.

Assuming a locally smooth boundary with outward normal $n _ { k }$ , when the update is truncated at the boundary, its Jacobian with respect to the nominal update is

$$
\frac { \partial \Delta _ { k } ^ { \mathrm { s a f e } } } { \partial d _ { k } } = \alpha _ { k } ^ { * } \left( I - \frac { d _ { k } n _ { k } ^ { \top } } { n _ { k } ^ { \top } d _ { k } } \right) .\tag{5}
$$

For an untruncated update, the Jacobian reduces to $I .$

Since $n _ { k }$ is unavailable under the oracle-only setting, we estimate it from symmetric binary feasibility queries near the detected boundary. Let $z _ { c }$ denote a point near the boundary and $\{ u _ { i } \} _ { i = 1 } ^ { \bar { K } }$ a set of unit probing directions. For each direction, we query $z _ { i } ^ { \pm } = z _ { c } \pm \rho u _ { i } .$ and define the directional vote $s _ { i } = \mathcal { O } _ { \mathrm { s } } ( z _ { i } ^ { - } ; \xi _ { t } ) - \mathcal { O } _ { \mathrm { s } } ( z _ { i } ^ { + } ; \xi _ { t } )$ . Aggregating these votes gives the approximate outward normal $\begin{array} { r } { \hat { n } _ { k } = ( \sum _ { i = 1 } ^ { K } s _ { i } u _ { i } ) / ( \Big \lVert \sum _ { i = 1 } ^ { K } s _ { i } u _ { i } \Big \rVert _ { 2 } + \epsilon _ { n } ) , ~ \hat { n } _ { k } ^ { \top } d _ { k } \ge 0 } \end{array}$ . Substituting $\hat { n } _ { k }$ into $\operatorname { E q . } \left( 5 \right)$ yields the practical surrogate Jacobian. The same oracle therefore serves two complementary roles: forward feasibility enforcement and backward estimation of local boundary geometry.

## 4 Experimental Results

We design a single-arm pick-and-place task in RoboTwin2.0 [15], as illustrated in Fig. 2. The robot moves a bottle from left to right while avoiding a stationary bottle obstacle and the tabletop, subject to joint-position limits. The feasibility oracle is implemented using privileged information from the simulator.

![](images/4499f60b43712fe149dd8bbab5f16d4973b34366d07edc94def25e764a9f3c1b.jpg)  
Figure 2: Snapshots of a rollout using π<sub>0.5</sub> + safeguard, subject to collision-avoidance constraints and joint-position limits.

Table 1: Comparison of task success, safety, motion smoothness, and inference efficiency. Best results are shown in bold.
<table><tr><td>Method</td><td>Success Rate ↑</td><td>Step Safety Rate ↑</td><td>LDLJ↑</td><td>Inference Time (s) ↓</td></tr><tr><td>π0.5 (gaussian init)</td><td>13/30</td><td>78.25%</td><td>-22.99</td><td>0.071</td></tr><tr><td>π0.5 (local init)</td><td>0/30</td><td>94.12%</td><td>-22.55</td><td>0.076</td></tr><tr><td>π0.5 + safeguard + stop grad</td><td>8/30</td><td>100%</td><td>-19.32</td><td>1.808</td></tr><tr><td>π0.5 + safeguard</td><td>11/30</td><td>100%</td><td>-18.35</td><td>1.345</td></tr></table>

We use absolute joint-position actions with horizon H = 20 and fine-tune pretrained $\pi _ { 0 . 5 }$ models using LoRA [16] on 150 expert obstacle-avoidance trajectories for 30K steps on a single NVIDIA H100 GPU.

Baselines. We compare the following variants: (1) $\pi _ { 0 . 5 }$ baseline: the original model without our safeguard. $\pi _ { 0 . 5 }$ (gaussian init) uses a standard Gaussian initial distribution, while $\pi _ { 0 . 5 }$ (local init) samples initial points around the current observed joint configuration. (2) $\pi _ { 0 . 5 } + s a f e g u a r d \colon$ the baseline augmented with our oracle-guided differentiable safeguard. (3) $\pi _ { 0 . 5 } + s a f e g u a r d + s t o p$ grad: an ablation of our method that removes the boundary-aware backward gradient estimation. Our method requires the initial sample to lie in the feasible region. We therefore sample around the current observed joint configuration and use the oracle to filter out infeasible samples.

Evaluation. We evaluate each method for 30 episodes under the clean setting. We report episode success rate, step safety rate, motion smoothness measured by LDLJ [17], and the average inference time per action chunk. As shown in Table 1, $\pi _ { 0 . 5 } ~ +$ safeguard achieves a 100% step safety rate while maintaining a success rate close to the unconstrained baseline. It also improves LDLJ from −22.99 to −18.35, indicating that the safeguard improves safety without severely sacrificing task performance and can also produce smoother motions. The $\pi _ { 0 . 5 }$ (local init) baseline shows that these gains do not come from the local initialization alone, as it yields much lower success and worse smoothness than $\pi _ { 0 . 5 }$ + safeguard. Moreover, the stop-gradient ablation degrades both success and smoothness, suggesting that propagating boundary information back to the policy helps it better adapt to the constrained feasible region.

## 5 Conclusion

We propose an oracle-guided differentiable safeguard for safe flow-matching robot policies using only binary feasibility feedback. The oracle enforces feasibility in the forward pass and provides local boundary information for backward learning, allowing the same safeguard to be used during training and inference without explicit or differentiable constraint models. Experiments on obstacle aware manipulation show improved safety while preserving task performance and motion quality.

A current limitation is that the discrete flow still relies on simple interpolation paths, which may become unreliable in highly non-convex feasible regions. Future work will explore constraint-aware flow paths and evaluate the method across more diverse tasks, robots, and real-world constraints.

## References

[1] Physical Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al. π<sub>0.5</sub>: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

[2] J. Bjorck, F. Castaneda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, ˜ S. Huang, et al. GR00T N1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

[3] S. Ye, Y. Ge, K. Zheng, S. Gao, S. Yu, G. Kurian, S. Indupuru, Y. L. Tan, C. Zhu, J. Xiang, et al. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026.

[4] T. Yuan, Z. Dong, Y. Liu, and H. Zhao. Fast-WAM: Do world action models need test-time future imagination? arXiv preprint arXiv:2603.16666, 2026.

[5] A. D. Ames, X. Xu, J. W. Grizzle, and P. Tabuada. Control barrier function based quadratic programs for safety critical systems. IEEE Transactions on Automatic Control, 62(8):3861– 3876, 2017.

[6] J. Yang, S. Jang, and S. Han. SafeFlowMatcher: Safe and fast planning using flow matching with control barrier functions. In International Conference on Learning Representations, volume 2026, pages 54058–54087, 2026.

[7] T.-Y. Huang, A. Lederer, D.-J. Wu, X. Dai, S. Zhang, H.-C. Lin, S.-H. Sun, S. Sosnowski, and S. Hirche. SAD-Flower: Flow matching for safe, admissible, and dynamically consistent planning. arXiv preprint arXiv:2511.05355, 2025.

[8] S. Hu, Z. Liu, S. Liu, J. Cen, Z. Meng, S. Wang, X. Li, and X. He. VLSA: Vision-languageaction models with plug-and-play safety constraint layer. arXiv preprint arXiv:2512.11891, 2025.

[9] U. Utkarsh, P. Cai, A. Edelman, R. Gomez-Bombarelli, and C. Rackauckas. Physicsconstrained flow matching: Sampling generative models with hard constraints. Advances in Neural Information Processing Systems, 38:160217–160252, 2025.

[10] C. Cheng, B. Han, D. Maddix, A. F. Ansari, A. Stuart, M. W. Mahoney, and B. Wang. Gradientfree generation for hard-constrained systems. In International Conference on Learning Representations, volume 2025, pages 100510–100539, 2025.

[11] J. Ma, R. Jin, X. Si, Y. Zhang, Y. Li, and Y. Gao. ActSafeGuard: Differentiable and trainingaligned constraint enforcement for flow-matching policies. arXiv preprint arXiv:2609.11697, 2026.

[12] J. K. Christopher, J. E. Warner, and F. Fioretto. Constraint-aware flow matching: Decision aligned end-to-end training for constrained sampling. arXiv preprint arXiv:2605.12754, 2026.

[13] X. Li, E. Liang, and M. Chen. Gauge flow matching: Efficient constrained generative modeling over general convex set and beyond. In The Fourteenth International Conference on Learning Representations, 2026.

[14] X. Liu, C. Gong, and Q. Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations, 2023.

[15] T. Chen, Z. Chen, B. Chen, Z. Cai, Y. Liu, Z. Li, Q. Liang, X. Lin, Y. Ge, Z. Gu, et al. RoboTwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025.

[16] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

[17] S. Balasubramanian, A. Melendez-Calderon, and E. Burdet. A robust and sensitive metric for quantifying movement smoothness. IEEE transactions on biomedical engineering, 59(8): 2126–2136, 2012.

[18] J. Chen, M. I. Jordan, and M. J. Wainwright. HopSkipJumpAttack: A query-efficient decisionbased attack. In 2020 IEEE Symposium on Security and Privacy, 2020.

[19] T. Xie, Y. Zhu, L. Yu, T. Yang, Z. Cheng, S. Zhang, X. Zhang, and C. Zhang. Reflected flow matching. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 54614–54634, 2024.

[20] R. T. Q. Chen and Y. Lipman. Flow matching on general geometries. In International Conference on Learning Representations, 2024.

## A Derivation of the Boundary-Aware Backward Pass

Following the notation in Section 3, consider a feasible point $x _ { k }$ whose update $d _ { k }$ is truncated at $x _ { k } + \alpha _ { k } ^ { * } d _ { k }$ . To derive Eq. (5), hold $x _ { k }$ fixed and perturb $d _ { k }$ . The resulting displacement of the boundary intersection lies in the local tangent plane and is therefore orthogonal to the outward normal $n _ { k }$ . This gives

$$
n _ { k } ^ { \top } \left( \alpha _ { k } ^ { * } \mathrm { d } d _ { k } + d _ { k } \mathrm { d } \alpha _ { k } ^ { * } \right) = 0 , \qquad \nabla _ { d _ { k } } \alpha _ { k } ^ { * } = - \frac { \alpha _ { k } ^ { * } n _ { k } } { n _ { k } ^ { \top } d _ { k } } ,\tag{A.1}
$$

where $n _ { k } ^ { \top } d _ { k } \neq 0$ at a transverse intersection. Applying the product rule to the ideal corrected update $\Delta _ { k } ^ { \mathrm { s a f e } } = \alpha _ { k } ^ { * } d _ { k }$ then yields

$$
\frac { \partial \Delta _ { k } ^ { \mathrm { s a f e } } } { \partial d _ { k } } = \alpha _ { k } ^ { * } I + d _ { k } \left( \nabla _ { d _ { k } } \alpha _ { k } ^ { * } \right) ^ { \top } = \alpha _ { k } ^ { * } \left( I - \frac { d _ { k } n _ { k } ^ { \top } } { n _ { k } ^ { \top } d _ { k } } \right) ,\tag{A.2}
$$

which recovers Eq. (5). For an untruncated update, $\alpha _ { k } ^ { * } = 1$ is locally constant, so the Jacobian reduces to $I .$ In practice, $\hat { \alpha } _ { k }$ and $\hat { n } _ { k }$ provide oracle-based approximations to the intersection factor and its normal, as described next.

## B Practical Implementation

Forward: coarse scanning and local bisection. Given $x _ { k }$ and $d _ { k }$ , we first query the oracle at uniformly spaced points along $r _ { k } ( \alpha ) = x _ { k } + \alpha d _ { k }$ , with $\alpha \in [ 0 , 1 ]$ ]. If all sampled points are feasible, the full update is accepted. Otherwise, the first sampled infeasible point and its preceding feasible point define a local search interval. Then we refine this interval by bisection. The final feasible endpoint approximates the first-exit point.

Backward: local boundary estimation. The final search interval also provides the neighborhood for estimating $n _ { k }$ . We place $z _ { c }$ at its midpoint and query symmetric pairs $z _ { i } ^ { \pm } = z _ { c } \pm \rho u _ { i }$ , using coordinate axes and random unit probing directions $\bar { \{ } u _ { i } \} _ { i = 1 } ^ { K }$ . When the two labels differ, the vote $s _ { i } = \mathcal { O } _ { \mathrm { s } } ( z _ { i } ^ { - } ; \xi _ { t } ) - \mathcal { O } _ { \mathrm { s } } ( z _ { i } ^ { + } ; \xi _ { t } )$ ) gives a signed contribution $s _ { i } u _ { i }$ toward the infeasible side; matching labels contribute zero. Aggregating and normalizing these contributions gives $\hat { n } _ { k }$ , with its sign chosen so that $\hat { n } _ { k } ^ { \top } d _ { k } \ge 0 .$ as in the main text. This use of binary labels to estimate boundary directions is related to decision-based methods such as HopSkipJumpAttack [18]. Here, the estimate supplies the local boundary information for the surrogate backward pass. The queries are batched across horizon elements, and normal estimation is needed only during training.

## C Experimental Details

Expert data collection. We collect demonstrations with a collision-aware MPlib planner. Inverse kinematics first provides candidate grasp configurations, after which RRT plans joint-space motions to the pre-grasp and pre-place poses. Local Cartesian planning completes the short approach and placement motions. Once the grasp is physically verified, the bottle geometry is attached to the end effector in the planning scene so that subsequent transport accounts for the carried object. The planner also uses a 4 cm inflated obstacle envelope to leave clearance along the planned motion. We retain successful, safety-checked episodes and replay their stored joint paths to record synchronized observations, actions, and privileged scene information, yielding the 150 demonstrations used in Table 1.

Oracle implementation. The recorded scene information supports a batched JAX feasibility oracle that combines joint-limit checks with collision checks. For each queried configuration, forward kinematics transforms sphere approximations of the left-arm links into world coordinates, where they are checked against the scene geometry. During transport, the carried bottle is placed using the candidate end-effector pose and the observed object-to-end-effector transform, allowing its collisions to be checked as well. The safeguard accesses the resulting binary feasibility labels without differentiating through the checker.

Feasible initial distribution. We construct $p _ { 0 } ( \cdot \mid o _ { t } )$ by randomly sampling joint configurations in a local neighborhood of the currently observed joint state and retaining candidates that satisfy the oracle constraints. An accepted configuration is repeated across the horizon to form $X _ { 0 }$ . The same initialization procedure is used during training and inference, providing a feasible starting point for the safeguarded flow.

Training parameters. All variants are fine-tuned with LoRA for 30K updates on a single H100 GPU, using a batch size of 32. We use AdamW with $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 ) , \epsilon = 1 0 ^ { - 8 }$ , weight decay $1 0 ^ { - 1 0 }$ , and gradient clipping at 1.0. The learning rate increases to $2 . 5 \times 1 0 ^ { - 5 }$ over 1K warmup steps and then follows a cosine schedule, with the decay step set to 20K and a final rate of $2 . 5 \times 1 0 ^ { - 6 }$ An exponential moving average of the parameters is maintained with decay 0.99.

## D Discussion and Limitations

## D.1 Why Maintain Feasibility at Every Generation Step?

Intermediate flow states are not executed by the robot, so their feasibility is an algorithmic choice rather than a physical execution requirement. We make this choice to construct a generation process that remains within the feasible set: starting from a feasible sample, each safeguarded update preserves feasibility, which in turn ensures that the final output satisfies the constraints. This shares the feasible-iterate motivation of interior-point methods, although our method uses oracle-based correction rather than a barrier objective. Maintaining generation trajectories within a constrained domain is also central to reflected flow matching [19], which uses boundary reflection. The resulting discrete invariance of our method is stated below.

Proposition 1 (Fixed-context feasibility preservation). For a fixed context $\xi _ { t } ,$ , suppose $X _ { 0 } \in \mathcal { C } ( \xi _ { t } ) ^ { H }$ and each horizon-wise update returns an oracle-verifiedfeasible configuration or retains itsfeasible input. Then $X _ { k } \in \mathcal { C } ( \xi _ { t } ) ^ { H }$ for every $k = 0 , \ldots , T .$

The claim follows by induction, since the forward search returns a checked feasible point at each step. Applying the same correction during training also exposes the policy to the interventions it encounters during inference. Together, these choices align policy learning with feasible generation and make final configuration feasibility a property of the sampling procedure, without requiring a separate terminal projection.

## D.2 What Are the Main Limitations?

Feasible data, initialization, and local geometry. The formulation assumes that the target action distribution lies within the feasible set and that feasible initial samples can be obtained. With linear reference paths, these endpoint conditions must be accompanied by local feasible connectivity: a sufficient condition is that each source–target pair lies within a common convex feasible neighbor hood. In a non-convex region, a straight reference path may instead cross an obstacle even when both endpoints are feasible, creating supervision that the safeguarded update cannot follow. The forward correction also preserves the predicted direction, so it cannot independently select a route around the obstacle. These limitations motivate reference paths that better capture feasible task geometry.

Oracle fidelity and computational cost. The practical correction depends on finite scans and local probes, whose resolution limits how accurately they capture complex boundaries. Its safety scope is therefore determined by the oracle and the configurations actually checked. Increasing the query budget can improve boundary resolution, but also increases computation. In Table 1, the full method requires 1.345 s per chunk compared with 0.071 s for the Gaussian baseline, approximately 18.9× as long. Reducing this overhead while retaining reliable feasibility checks is necessary for more responsive manipulation.

## D.3 Future Directions

A natural next step is to replace straight reference paths with nonlinear paths that remain within the feasible region. Riemannian flow matching [20] illustrates how known manifold geometry can inform the construction of target vector fields; in our oracle-only setting, feasible expert trajectories offer a practical source of path information. The reference can follow the demonstrated route around obstacles rather than directly interpolate between the initial sample and each target configuration. With feasible local connections, this could extend the approach to non-convex constraint regions while retaining the same oracle-guided safeguard. The remaining challenge is to connect randomly sampled initial configurations to these paths and discretize them without introducing infeasible shortcuts.

A complementary direction is to reduce the cost of oracle queries during generation. Adaptive coarse scanning and selective local refinement could concentrate queries near detected boundaries, while more efficient batched collision checking could reduce their latency. Evaluating these strategies jointly with task success and constraint satisfaction would help identify a practical balance between computational cost and safety for online manipulation.